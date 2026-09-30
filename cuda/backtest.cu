// Native backtest backend for StrategySpec candidates. Called from Python through ctypes
// (src/backends/native.py) with the same contract as src.strategies.sweep.evaluate_candidates.
//
// One source, two builds (cuda/Makefile):
//   make -C cuda       -> build/libbacktest_cuda.so  nvcc: features and candidates run on the GPU
//   make -C cuda cpu   -> build/libbacktest_cpu.so   plain C++ (optionally OpenMP): the "C++ CPU"
//                                                    backend, also used to test the logic without a GPU
//
// The semantics must match the Python reference exactly (src/strategies/features.py, operators.py,
// evaluator.py, src/backtest/engine.py, metrics.py); tests/test_native_backend.py checks parity.
//
// Pipeline for one call (one dataset, one period [start, end)):
//   1. feature buffers: every distinct (feature, field, lookback) is computed once over bars [0, end)
//      GPU: one thread per (buffer, bar); lookback is a runtime argument, never a template.
//   2. candidates: each candidate combines its buffers with its thresholds into positions and runs
//      the backtest + metrics over [start, end). GPU: one thread per candidate, sequential in time
//      (the same arithmetic order as the Python engine, so results match to rounding).
//   3. only the metrics table (n_candidates x N_METRICS) goes back to the host.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

// Must match FEATURE_CODES / OPERATOR_CODES / LOGIC_CODES in src/backends/native.py.
enum Feature {
    F_RETURNS = 1,
    F_MOMENTUM = 2,
    F_ROLLING_MEAN = 3,
    F_ROLLING_STD = 4,
    F_VOLATILITY = 5,
    F_ZSCORE = 6,
    F_ROLLING_MIN = 7,
    F_ROLLING_MAX = 8,
    F_VOLUME_CHANGE = 9,
    F_DISTANCE_TO_MAX = 10,
    F_DISTANCE_TO_MIN = 11,
    F_DISTANCE_TO_MEAN = 12,
};
enum Operator { OP_GT = 1, OP_GE = 2, OP_LT = 3, OP_LE = 4 };
enum Logic { LOGIC_AND = 1, LOGIC_OR = 2 };

// Output column order; must match METRIC_NAMES in src/backtest/metrics.py.
enum Metric {
    M_CUMULATIVE_RETURN,
    M_ANNUALIZED_RETURN,
    M_SHARPE,
    M_ANNUALIZED_VOLATILITY,
    M_MAX_DRAWDOWN,
    M_TURNOVER,
    M_ANNUAL_TURNOVER,
    M_N_TRADES,
    M_EXPOSURE,
    M_N_BARS,
    N_METRICS
};

HD inline double qnan() { return nan(""); }
HD inline bool is_nan(double x) { return x != x; }
HD inline double clean(double x) { return (x - x == 0.0) ? x : qnan(); }  // +-inf and NaN -> NaN

// ------------------------------------------------------------------------------------ features

// x[t] / x[t - lag] - 1 (momentum; returns with lag 1). Division by zero -> NaN.
HD inline double ratio_change(const double* x, long long t, long long lag) {
    if (t < lag) return qnan();
    return clean(x[t] / x[t - lag] - 1.0);
}

// A series view: the raw field, or its one-bar returns (for volatility).
struct Series {
    const double* x;
    bool returns;
    HD double operator()(long long i) const { return returns ? ratio_change(x, i, 1) : x[i]; }
};

// Trailing windows [t-L+1, t]: NaN during warm-up or if any value is NaN. Like pandas, a window
// whose values are all equal has mean exactly that value and standard deviation exactly 0.
HD double window_mean(Series s, long long t, int L) {
    if (t < L - 1) return qnan();
    const double first = s(t - L + 1);
    double sum = 0.0;
    bool same = true;
    for (long long i = t - L + 1; i <= t; ++i) {
        const double v = s(i);
        if (is_nan(v)) return qnan();
        sum += v;
        same = same && v == first;
    }
    return same ? first : clean(sum / L);
}

HD double window_std(Series s, long long t, int L) {
    if (t < L - 1) return qnan();
    const double first = s(t - L + 1);
    double sum = 0.0;
    bool same = true;
    for (long long i = t - L + 1; i <= t; ++i) {
        const double v = s(i);
        if (is_nan(v)) return qnan();
        sum += v;
        same = same && v == first;
    }
    if (same) return 0.0;
    const double mean = sum / L;
    double ss = 0.0;
    for (long long i = t - L + 1; i <= t; ++i) {
        const double d = s(i) - mean;
        ss += d * d;
    }
    return clean(sqrt(ss / (L - 1)));
}

HD double window_extreme(Series s, long long t, int L, bool want_max) {
    if (t < L - 1) return qnan();
    double best = s(t);
    for (long long i = t - L + 1; i <= t; ++i) {
        const double v = s(i);
        if (is_nan(v)) return qnan();
        if (want_max ? v > best : v < best) best = v;
    }
    return best;
}

HD double feature_value(int feature, const double* x, int L, long long t) {
    const Series raw = {x, false};
    switch (feature) {
        case F_RETURNS: return ratio_change(x, t, 1);
        case F_MOMENTUM: return ratio_change(x, t, L);
        case F_ROLLING_MEAN: return window_mean(raw, t, L);
        case F_ROLLING_STD: return window_std(raw, t, L);
        case F_VOLATILITY: return window_std(Series{x, true}, t, L);
        case F_ZSCORE: {
            const double mean = window_mean(raw, t, L), sd = window_std(raw, t, L);
            if (is_nan(mean) || is_nan(sd) || sd == 0.0) return qnan();
            return clean((x[t] - mean) / sd);
        }
        case F_ROLLING_MIN: return window_extreme(raw, t, L, false);
        case F_ROLLING_MAX: return window_extreme(raw, t, L, true);
        case F_VOLUME_CHANGE: {  // x[t] vs the mean of the previous L values (bar t excluded)
            if (t < L) return qnan();
            return clean(x[t] / window_mean(raw, t - 1, L) - 1.0);
        }
        // Distance of x[t] from its trailing max / min / mean (bar t included), as a fraction.
        case F_DISTANCE_TO_MAX: return clean(x[t] / window_extreme(raw, t, L, true) - 1.0);
        case F_DISTANCE_TO_MIN: return clean(x[t] / window_extreme(raw, t, L, false) - 1.0);
        case F_DISTANCE_TO_MEAN: return clean(x[t] / window_mean(raw, t, L) - 1.0);
    }
    return qnan();
}

// ---------------------------------------------------------------------------- candidates

struct Candidate {
    int n_conditions, logic;
    double true_position, false_position;
    const int* buffer;       // per condition: index of its feature buffer
    const int* op;           // per condition: comparison operator
    const double* threshold; // per condition
};

HD inline bool compare(int op, double v, double threshold) {
    switch (op) {
        case OP_GT: return v > threshold;
        case OP_GE: return v >= threshold;
        case OP_LT: return v < threshold;
        default: return v <= threshold;  // OP_LE
    }
}

// Target position decided at bar t's close. Any undefined condition -> flat (as in Python, where
// AND/OR propagate NaN). A candidate without conditions is always at true_position.
HD double position_at(const double* buffers, long long n_rows, const Candidate& c, long long t) {
    if (c.n_conditions == 0) return c.true_position;
    bool combined = c.logic == LOGIC_AND;
    for (int k = 0; k < c.n_conditions; ++k) {
        const double v = buffers[(size_t)c.buffer[k] * n_rows + t];
        if (is_nan(v)) return 0.0;
        const bool cond = compare(c.op[k], v, c.threshold[k]);
        combined = c.logic == LOGIC_AND ? (combined && cond) : (combined || cond);
    }
    return combined ? c.true_position : c.false_position;
}

// Backtest one candidate on [start, end) and write its metrics (same formulas as compute_metrics).
// The position set at bar t-1 is held during bar t; the first bar of the period starts flat.
HD void candidate_metrics(const double* buffers, long long n_rows, const Candidate& c, const double* returns,
                          long long start, long long end, double cost_bps, double ppy, double* out) {
    const long long n = end - start;
    double held_prev = 0.0, equity = 1.0, peak = 1.0, mdd = 0.0;
    double sum = 0.0, turnover = 0.0, trades = 0.0, exposed = 0.0;
    for (long long t = start; t < end; ++t) {
        const double held = t > start ? position_at(buffers, n_rows, c, t - 1) : 0.0;
        const double turn = fabs(held - held_prev);
        const double r = held * returns[t - start] - turn * cost_bps / 10000.0;
        held_prev = held;
        sum += r;
        turnover += turn;
        trades += turn > 0.0 ? 1.0 : 0.0;
        exposed += held != 0.0 ? 1.0 : 0.0;
        equity *= 1.0 + r;
        if (equity > peak) peak = equity;
        const double dd = equity / peak - 1.0;
        if (dd < mdd) mdd = dd;
    }
    // Second pass: sample variance around the mean (two-pass, like pandas).
    const double mean = n > 0 ? sum / n : qnan();
    double ss = 0.0;
    held_prev = 0.0;
    for (long long t = start; t < end; ++t) {
        const double held = t > start ? position_at(buffers, n_rows, c, t - 1) : 0.0;
        const double r = held * returns[t - start] - fabs(held - held_prev) * cost_bps / 10000.0;
        held_prev = held;
        ss += (r - mean) * (r - mean);
    }
    const double sd = n > 1 ? sqrt(ss / (n - 1)) : qnan();
    const double cumulative = equity - 1.0;
    const double growth = 1.0 + cumulative;
    out[M_CUMULATIVE_RETURN] = cumulative;
    out[M_ANNUALIZED_RETURN] = n == 0 ? qnan() : growth > 0.0 ? pow(growth, ppy / n) - 1.0 : -1.0;
    out[M_SHARPE] = (n > 1 && sd > 0.0 && sd - sd == 0.0) ? mean / sd * sqrt(ppy) : qnan();
    out[M_ANNUALIZED_VOLATILITY] = sd * sqrt(ppy);
    out[M_MAX_DRAWDOWN] = mdd;
    out[M_TURNOVER] = turnover;
    out[M_ANNUAL_TURNOVER] = n > 0 ? turnover / (n / ppy) : qnan();
    out[M_N_TRADES] = trades;
    out[M_EXPOSURE] = n > 0 ? exposed / n : qnan();
    out[M_N_BARS] = (double)n;
}

struct Problem {
    const double* fields;
    long long n_rows;
    const int *buf_feature, *buf_field, *buf_lookback;
    int n_buffers;
    const int *cand_offset, *cand_ncond, *cand_logic;
    const double *cand_true, *cand_false;
    int n_candidates;
    const int *cond_buffer, *cond_op;
    const double* cond_threshold;
    const double* returns;
    long long start, end;
    double cost_bps, ppy;
};

HD inline Candidate candidate_view(const Problem& p, int i) {
    const int o = p.cand_offset[i];
    return Candidate{p.cand_ncond[i], p.cand_logic[i], p.cand_true[i], p.cand_false[i],
                     p.cond_buffer + o, p.cond_op + o, p.cond_threshold + o};
}

// ------------------------------------------------------------------------- argument checks

static int check_problem(const Problem& p, int n_fields, int n_conds, char* error, int error_len) {
    if (p.start < 0 || p.end > p.n_rows || p.start >= p.end) {
        snprintf(error, error_len, "invalid period [%lld, %lld) for %lld rows", p.start, p.end, p.n_rows);
        return 2;
    }
    for (int b = 0; b < p.n_buffers; ++b) {
        if (p.buf_feature[b] < F_RETURNS || p.buf_feature[b] > F_DISTANCE_TO_MEAN || p.buf_field[b] < 0 ||
            p.buf_field[b] >= n_fields || p.buf_lookback[b] < 0) {
            snprintf(error, error_len, "invalid feature buffer %d", b);
            return 2;
        }
    }
    for (int i = 0; i < p.n_candidates; ++i) {
        if (p.cand_offset[i] < 0 || p.cand_ncond[i] < 0 || p.cand_offset[i] + p.cand_ncond[i] > n_conds ||
            (p.cand_logic[i] != LOGIC_AND && p.cand_logic[i] != LOGIC_OR)) {
            snprintf(error, error_len, "invalid candidate %d", i);
            return 2;
        }
    }
    for (int k = 0; k < n_conds; ++k) {
        if (p.cond_buffer[k] < 0 || p.cond_buffer[k] >= p.n_buffers || p.cond_op[k] < OP_GT || p.cond_op[k] > OP_LE) {
            snprintf(error, error_len, "invalid condition %d", k);
            return 2;
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------ CUDA path

#ifdef __CUDACC__

__global__ void feature_kernel(Problem p, double* buffers) {
    const long long total = (long long)p.n_buffers * p.n_rows;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)blockDim.x * gridDim.x) {
        const int b = (int)(i / p.n_rows);
        const long long t = i % p.n_rows;
        buffers[i] = feature_value(p.buf_feature[b], p.fields + (size_t)p.buf_field[b] * p.n_rows,
                                   p.buf_lookback[b], t);
    }
}

__global__ void candidate_kernel(Problem p, const double* buffers, double* out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n_candidates; i += blockDim.x * gridDim.x) {
        candidate_metrics(buffers, p.n_rows, candidate_view(p, i), p.returns, p.start, p.end, p.cost_bps, p.ppy,
                          out + (size_t)i * N_METRICS);
    }
}

// Device allocations freed automatically when the call returns (also on errors).
struct DeviceMemory {
    std::vector<void*> ptrs;
    ~DeviceMemory() {
        for (void* ptr : ptrs) cudaFree(ptr);
    }
    template <class T> cudaError_t alloc(T** ptr, size_t n) {
        cudaError_t e = cudaMalloc((void**)ptr, (n ? n : 1) * sizeof(T));
        if (e == cudaSuccess) ptrs.push_back(*ptr);
        return e;
    }
    template <class T> cudaError_t upload(const T** ptr, const T* host, size_t n) {
        T* dev = nullptr;
        cudaError_t e = alloc(&dev, n);
        if (e == cudaSuccess && n) e = cudaMemcpy(dev, host, n * sizeof(T), cudaMemcpyHostToDevice);
        *ptr = dev;
        return e;
    }
};

#define CUDA_TRY(call)                                                                         \
    do {                                                                                       \
        cudaError_t e_ = (call);                                                               \
        if (e_ != cudaSuccess) {                                                               \
            snprintf(error, error_len, "%s failed: %s", #call, cudaGetErrorString(e_));        \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

static int run(const Problem& host, int n_fields, int n_conds, int device, double* out, char* error, int error_len) {
    CUDA_TRY(cudaSetDevice(device));
    DeviceMemory mem;
    Problem p = host;
    const size_t n_returns = (size_t)(host.end - host.start);
    CUDA_TRY(mem.upload(&p.fields, host.fields, (size_t)n_fields * host.n_rows));
    CUDA_TRY(mem.upload(&p.buf_feature, host.buf_feature, host.n_buffers));
    CUDA_TRY(mem.upload(&p.buf_field, host.buf_field, host.n_buffers));
    CUDA_TRY(mem.upload(&p.buf_lookback, host.buf_lookback, host.n_buffers));
    CUDA_TRY(mem.upload(&p.cand_offset, host.cand_offset, host.n_candidates));
    CUDA_TRY(mem.upload(&p.cand_ncond, host.cand_ncond, host.n_candidates));
    CUDA_TRY(mem.upload(&p.cand_logic, host.cand_logic, host.n_candidates));
    CUDA_TRY(mem.upload(&p.cand_true, host.cand_true, host.n_candidates));
    CUDA_TRY(mem.upload(&p.cand_false, host.cand_false, host.n_candidates));
    CUDA_TRY(mem.upload(&p.cond_buffer, host.cond_buffer, n_conds));
    CUDA_TRY(mem.upload(&p.cond_op, host.cond_op, n_conds));
    CUDA_TRY(mem.upload(&p.cond_threshold, host.cond_threshold, n_conds));
    CUDA_TRY(mem.upload(&p.returns, host.returns, n_returns));
    double *buffers = nullptr, *d_out = nullptr;
    CUDA_TRY(mem.alloc(&buffers, (size_t)host.n_buffers * host.n_rows));
    CUDA_TRY(mem.alloc(&d_out, (size_t)host.n_candidates * N_METRICS));

    const long long total = (long long)host.n_buffers * host.n_rows;
    if (total > 0) {
        const long long wanted = total / 256 + 1;
        const int feature_blocks = (int)(wanted < 65535 ? wanted : 65535);  // grid-stride loop covers the rest
        feature_kernel<<<feature_blocks, 256>>>(p, buffers);
        CUDA_TRY(cudaGetLastError());
    }
    // Small blocks: each thread runs a whole backtest, so spread candidates over many SMs.
    const int threads = 64;
    const int wanted_blocks = (host.n_candidates + threads - 1) / threads;
    const int candidate_blocks = wanted_blocks < 65535 ? wanted_blocks : 65535;
    candidate_kernel<<<candidate_blocks, threads>>>(p, buffers, d_out);
    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(out, d_out, (size_t)host.n_candidates * N_METRICS * sizeof(double), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int bt_info(int device, char* buffer, int length) {
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, device);
    if (e != cudaSuccess) {
        snprintf(buffer, length, "cuda device %d unavailable: %s", device, cudaGetErrorString(e));
        return 1;
    }
    snprintf(buffer, length, "cuda:%d %s (compute %d.%d, %.0f GiB)", device, prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / 1073741824.0);
    return 0;
}

// ------------------------------------------------------------------------------- CPU path

#else

static int run(const Problem& p, int, int, int, double* out, char*, int) {
    std::vector<double> buffers((size_t)p.n_buffers * p.n_rows);
    const long long total = (long long)p.n_buffers * p.n_rows;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < total; ++i) {
        const int b = (int)(i / p.n_rows);
        buffers[i] = feature_value(p.buf_feature[b], p.fields + (size_t)p.buf_field[b] * p.n_rows,
                                   p.buf_lookback[b], i % p.n_rows);
    }
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < p.n_candidates; ++i) {
        candidate_metrics(buffers.data(), p.n_rows, candidate_view(p, i), p.returns, p.start, p.end, p.cost_bps,
                          p.ppy, out + (size_t)i * N_METRICS);
    }
    return 0;
}

extern "C" int bt_info(int, char* buffer, int length) {
#ifdef _OPENMP
    snprintf(buffer, length, "cpu (C++, OpenMP %d threads)", omp_get_max_threads());
#else
    snprintf(buffer, length, "cpu (C++, single thread)");
#endif
    return 0;
}

#endif

// -------------------------------------------------------------------------------- C API

// Returns 0 on success; otherwise non-zero with a message in `error`.
//   fields        [n_fields x n_rows] input series, bars [0, end) (row-major, field-major)
//   buf_*         [n_buffers] distinct features: feature code, field index, lookback
//   cand_*        [n_candidates] condition offset/count into cond_*, logic, true/false position
//   cond_*        [n_conds] buffer index, operator, threshold
//   returns       [end - start] period returns (src.backtest.engine.period_returns)
//   out           [n_candidates x N_METRICS]
extern "C" int bt_run(const double* fields, int n_fields, long long n_rows, const int* buf_feature,
                      const int* buf_field, const int* buf_lookback, int n_buffers, const int* cand_offset,
                      const int* cand_ncond, const int* cand_logic, const double* cand_true,
                      const double* cand_false, int n_candidates, const int* cond_buffer, const int* cond_op,
                      const double* cond_threshold, int n_conds, const double* returns, long long start,
                      long long end, double cost_bps, double periods_per_year, int device, double* out,
                      char* error, int error_len) {
    const Problem p = {fields,      n_rows,     buf_feature,  buf_field,  buf_lookback, n_buffers,
                       cand_offset, cand_ncond, cand_logic,   cand_true,  cand_false,   n_candidates,
                       cond_buffer, cond_op,    cond_threshold, returns,  start,        end,
                       cost_bps,    periods_per_year};
    const int status = check_problem(p, n_fields, n_conds, error, error_len);
    if (status != 0 || n_candidates == 0) return status;
    return run(p, n_fields, n_conds, device, out, error, error_len);
}
