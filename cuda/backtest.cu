// Batch backtester for strategy programs compiled by src/strategy/dsl.py::compile_spec.
//
// Must match the Python reference engine (src/strategy/dsl.py + src/backtest/engine.py +
// src/backtest/metrics.py); tests/test_cuda_parity.py checks this.
//
// GPU layout: one thread block per strategy. Element-wise and rolling-window instructions are
// parallel over time; EMA and the metrics pass are sequential in thread 0 of the block.
//
// Built with nvcc this runs on the GPU. Built as plain C++ (make cpu) the same code runs on the
// CPU with one-thread "blocks", which lets the parity test run on machines without CUDA.
//
// Usage: backtest <data.bin> <programs.txt> <start> <end> <cost_bps> <periods_per_year>
//   data.bin      int64 T, then 5*T float64 values: open, high, low, close, volume (field-major)
//   programs.txt  S, then per strategy a line "n_instr output_register" followed by n_instr lines
//                 "opcode dst a b int_param float_param"
//   start, end    bar range [start, end) to backtest; signals are computed on bars [0, end)
// Output (stdout): CSV header + one row of metrics per strategy.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#ifdef __CUDACC__
#define CHECK(call)                                                                              \
    do {                                                                                         \
        cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,     \
                    __LINE__);                                                                   \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)
#define HD __host__ __device__
#else
// CPU emulation: blocks run one after another with a single thread each.
struct Dim3 {
    unsigned x;
};
static Dim3 threadIdx = {0}, blockIdx = {0}, blockDim = {1};
#define __global__
#define __syncthreads()
#define HD
#endif

// Must match OPCODES in src/strategy/dsl.py.
enum Op {
    OP_CONST = 1,
    OP_ADD = 2,
    OP_SUB = 3,
    OP_MUL = 4,
    OP_DIV = 5,
    OP_NEG = 6,
    OP_ABS = 7,
    OP_SIGN = 8,
    OP_LOG = 9,
    OP_GT = 10,
    OP_LT = 11,
    OP_LAG = 12,
    OP_DIFF = 13,
    OP_PCT_CHANGE = 14,
    OP_SMA = 15,
    OP_EMA = 16,
    OP_STD = 17,
    OP_ZSCORE = 18,
    OP_MAX = 19,
    OP_MIN = 20,
};

const int N_FIELDS = 5;
[[maybe_unused]] const int THREADS = 256;

struct Instr {
    int op, dst, a, b, iparam;
    double fparam;
};

struct Metrics {
    double cumulative_return, sharpe, annualized_volatility, max_drawdown, turnover,
        annual_turnover, n_trades;
};

HD inline double qnan() { return nan(""); }
HD inline bool is_nan(double x) { return x != x; }
HD inline double finite_or_nan(double x) { return (x - x == 0.0) ? x : qnan(); }
HD inline double safe_div(double a, double b) { return b == 0.0 ? qnan() : finite_or_nan(a / b); }

// Trailing-window helpers: NaN during warm-up or if any value in the window is NaN.
HD double window_mean(const double* x, long t, int w) {
    if (t < w - 1) return qnan();
    double sum = 0.0;
    for (long i = t - w + 1; i <= t; ++i) {
        if (is_nan(x[i])) return qnan();
        sum += x[i];
    }
    return finite_or_nan(sum / w);
}

HD double window_std(const double* x, long t, int w, double mean) {
    if (is_nan(mean)) return qnan();
    double ss = 0.0;
    for (long i = t - w + 1; i <= t; ++i) ss += (x[i] - mean) * (x[i] - mean);
    return finite_or_nan(sqrt(ss / (w - 1)));
}

HD double window_extreme(const double* x, long t, int w, bool want_max) {
    if (t < w - 1) return qnan();
    double best = x[t];
    for (long i = t - w + 1; i <= t; ++i) {
        if (is_nan(x[i])) return qnan();
        if (want_max ? x[i] > best : x[i] < best) best = x[i];
    }
    return best;
}

HD double eval_point(const Instr& in, const double* a, const double* b, long t) {
    const int p = in.iparam;
    switch (in.op) {
        case OP_CONST: return in.fparam;
        case OP_ADD: return finite_or_nan(a[t] + b[t]);
        case OP_SUB: return finite_or_nan(a[t] - b[t]);
        case OP_MUL: return finite_or_nan(a[t] * b[t]);
        case OP_DIV: return safe_div(a[t], b[t]);
        case OP_NEG: return -a[t];
        case OP_ABS: return fabs(a[t]);
        case OP_SIGN: return is_nan(a[t]) ? a[t] : (double)((a[t] > 0) - (a[t] < 0));
        case OP_LOG: return a[t] > 0 ? finite_or_nan(log(a[t])) : qnan();
        case OP_GT: return (is_nan(a[t]) || is_nan(b[t])) ? qnan() : (a[t] > b[t] ? 1.0 : 0.0);
        case OP_LT: return (is_nan(a[t]) || is_nan(b[t])) ? qnan() : (a[t] < b[t] ? 1.0 : 0.0);
        case OP_LAG: return t < p ? qnan() : a[t - p];
        case OP_DIFF: return t < p ? qnan() : finite_or_nan(a[t] - a[t - p]);
        case OP_PCT_CHANGE: return t < p ? qnan() : finite_or_nan(safe_div(a[t], a[t - p]) - 1.0);
        case OP_SMA: return window_mean(a, t, p);
        case OP_STD: return window_std(a, t, p, window_mean(a, t, p));
        case OP_ZSCORE: {
            double m = window_mean(a, t, p);
            return safe_div(a[t] - m, window_std(a, t, p, m));
        }
        case OP_MAX: return window_extreme(a, t, p, true);
        case OP_MIN: return window_extreme(a, t, p, false);
    }
    return qnan();
}

HD void ema(const double* x, double* out, long n, int w) {
    const double alpha = 2.0 / (w + 1);
    double state = qnan();
    for (long t = 0; t < n; ++t) {
        if (is_nan(x[t])) {
            out[t] = qnan();
            continue;
        }
        state = is_nan(state) ? x[t] : alpha * x[t] + (1 - alpha) * state;
        out[t] = state;
    }
}

// One block = one strategy. `regs` is this block's scratch space of max_regs series of length n.
__global__ void backtest_kernel(const double* fields, long n_total, const double* returns, long start,
                                long end, const Instr* instrs, const int* prog_offset,
                                const int* prog_len, const int* prog_out, int batch_first,
                                double* scratch, int max_regs, double cost, double ppy,
                                Metrics* out) {
    const int s = batch_first + blockIdx.x;
    double* regs = scratch + (size_t)blockIdx.x * max_regs * end;
#define REG(r) ((r) < N_FIELDS ? fields + (size_t)(r) * n_total : regs + (size_t)((r) - N_FIELDS) * end)

    for (int k = 0; k < prog_len[s]; ++k) {
        const Instr in = instrs[prog_offset[s] + k];
        double* dst = regs + (size_t)(in.dst - N_FIELDS) * end;  // dst is never an input field
        const double* a = in.a >= 0 ? REG(in.a) : nullptr;
        const double* b = in.b >= 0 ? REG(in.b) : nullptr;
        if (in.op == OP_EMA) {
            if (threadIdx.x == 0) ema(a, dst, end, in.iparam);
        } else {
            for (long t = threadIdx.x; t < end; t += blockDim.x) dst[t] = eval_point(in, a, b, t);
        }
        __syncthreads();
    }
    const double* signal = REG(prog_out[s]);
#undef REG

    if (threadIdx.x != 0) return;
    // Position at bar t = sign(signal[t-1]); the first bar of the range starts flat.
    double held_prev = 0.0, equity = 1.0, peak = 1.0, mdd = 0.0;
    double mean = 0.0, m2 = 0.0, turnover = 0.0, trades = 0.0;
    long n = 0;
    for (long t = start; t < end; ++t) {
        double held = 0.0;
        if (t > start) {
            double x = signal[t - 1];
            held = is_nan(x) ? 0.0 : (double)((x > 0) - (x < 0));
        }
        double turn = fabs(held - held_prev);
        double r = held * returns[t - start] - turn * cost;
        held_prev = held;
        turnover += turn;
        trades += turn > 0 ? 1.0 : 0.0;
        equity *= 1.0 + r;
        if (equity > peak) peak = equity;
        double dd = equity / peak - 1.0;
        if (dd < mdd) mdd = dd;
        ++n;
        double delta = r - mean;
        mean += delta / n;
        m2 += delta * (r - mean);
    }
    double std = n > 1 ? sqrt(m2 / (n - 1)) : qnan();
    Metrics m;
    m.cumulative_return = equity - 1.0;
    m.sharpe = (n > 1 && std > 0) ? mean / std * sqrt(ppy) : qnan();
    m.annualized_volatility = std * sqrt(ppy);
    m.max_drawdown = mdd;
    m.turnover = turnover;
    m.annual_turnover = n > 0 ? turnover / (n / ppy) : qnan();
    m.n_trades = trades;
    out[s] = m;
}

// ------------------------------------------------------------------ host memory helpers

#ifdef __CUDACC__
template <class T> T* dev_alloc(size_t n) {
    T* p = nullptr;
    CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}
template <class T> void to_dev(T* d, const T* h, size_t n) {
    CHECK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> void to_host(T* h, const T* d, size_t n) {
    CHECK(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost));
}
template <class T> void dev_free(T* p) { CHECK(cudaFree(p)); }
#else
template <class T> T* dev_alloc(size_t n) {
    T* p = (T*)malloc(n * sizeof(T) + 1);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}
template <class T> void to_dev(T* d, const T* h, size_t n) { memcpy(d, h, n * sizeof(T)); }
template <class T> void to_host(T* h, const T* d, size_t n) { memcpy(h, d, n * sizeof(T)); }
template <class T> void dev_free(T* p) { free(p); }
#endif

static void fail(const char* msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

int main(int argc, char** argv) {
    if (argc != 7) fail("usage: backtest <data.bin> <programs.txt> <start> <end> <cost_bps> <periods_per_year>");
    const long start = atol(argv[3]), end = atol(argv[4]);
    const double cost = atof(argv[5]) / 10000.0, ppy = atof(argv[6]);

    // Price data.
    FILE* f = fopen(argv[1], "rb");
    if (!f) fail("cannot open data file");
    int64_t n_total = 0;
    if (fread(&n_total, sizeof(n_total), 1, f) != 1) fail("bad data header");
    std::vector<double> fields((size_t)N_FIELDS * n_total);
    if (fread(fields.data(), sizeof(double), fields.size(), f) != fields.size()) fail("truncated data file");
    fclose(f);
    if (start < 0 || end > n_total || start >= end) fail("invalid start/end range");

    // Simple close-to-close returns on [start, end), forward-filling missing closes within the range.
    const double* close = fields.data() + 3 * n_total;
    std::vector<double> returns(end - start, 0.0);
    double last = qnan();
    for (long t = start; t < end; ++t) {
        double c = is_nan(close[t]) ? last : close[t];
        if (t > start && !is_nan(c) && !is_nan(last)) returns[t - start] = finite_or_nan(c / last - 1.0);
        if (is_nan(returns[t - start])) returns[t - start] = 0.0;
        last = c;
    }

    // Programs.
    FILE* p = fopen(argv[2], "r");
    if (!p) fail("cannot open programs file");
    int n_strats = 0;
    if (fscanf(p, "%d", &n_strats) != 1 || n_strats < 0) fail("bad programs header");
    std::vector<Instr> instrs;
    std::vector<int> offset(n_strats), length(n_strats), output(n_strats);
    int max_regs = 1;
    for (int s = 0; s < n_strats; ++s) {
        if (fscanf(p, "%d %d", &length[s], &output[s]) != 2) fail("bad program header");
        offset[s] = (int)instrs.size();
        for (int k = 0; k < length[s]; ++k) {
            Instr in;
            if (fscanf(p, "%d %d %d %d %d %lf", &in.op, &in.dst, &in.a, &in.b, &in.iparam, &in.fparam) != 6)
                fail("bad instruction");
            if (in.op < OP_CONST || in.op > OP_MIN || in.dst != N_FIELDS + k || in.a >= in.dst || in.b >= in.dst)
                fail("invalid instruction");
            instrs.push_back(in);
        }
        if (output[s] < 0 || output[s] >= N_FIELDS + length[s]) fail("invalid output register");
        if (length[s] > max_regs) max_regs = length[s];
    }
    fclose(p);

    std::vector<Metrics> results(n_strats);
    if (n_strats > 0) {
        double* d_fields = dev_alloc<double>(fields.size());
        double* d_returns = dev_alloc<double>(returns.size());
        Instr* d_instrs = dev_alloc<Instr>(instrs.size() + 1);
        int* d_offset = dev_alloc<int>(n_strats);
        int* d_len = dev_alloc<int>(n_strats);
        int* d_out = dev_alloc<int>(n_strats);
        Metrics* d_metrics = dev_alloc<Metrics>(n_strats);
        to_dev(d_fields, fields.data(), fields.size());
        to_dev(d_returns, returns.data(), returns.size());
        to_dev(d_instrs, instrs.data(), instrs.size());
        to_dev(d_offset, offset.data(), n_strats);
        to_dev(d_len, length.data(), n_strats);
        to_dev(d_out, output.data(), n_strats);

        // Strategies per launch, limited so scratch registers use at most half of free memory.
        const size_t per_strategy = (size_t)max_regs * end * sizeof(double);
        size_t budget = (size_t)2 << 30;
#ifdef __CUDACC__
        size_t free_mem = 0, total_mem = 0;
        CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        budget = free_mem / 2;
#endif
        int batch = (int)(budget / per_strategy);
        if (batch < 1) batch = 1;
        if (batch > n_strats) batch = n_strats;
        double* d_scratch = dev_alloc<double>((size_t)batch * max_regs * end);

        for (int first = 0; first < n_strats; first += batch) {
            const int blocks = first + batch <= n_strats ? batch : n_strats - first;
#ifdef __CUDACC__
            backtest_kernel<<<blocks, THREADS>>>(d_fields, n_total, d_returns, start, end, d_instrs,
                                                 d_offset, d_len, d_out, first, d_scratch, max_regs,
                                                 cost, ppy, d_metrics);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());
#else
            for (int b = 0; b < blocks; ++b) {
                blockIdx.x = b;
                backtest_kernel(d_fields, n_total, d_returns, start, end, d_instrs, d_offset, d_len,
                                d_out, first, d_scratch, max_regs, cost, ppy, d_metrics);
            }
#endif
        }
        to_host(results.data(), d_metrics, n_strats);
        dev_free(d_fields); dev_free(d_returns); dev_free(d_instrs); dev_free(d_offset);
        dev_free(d_len); dev_free(d_out); dev_free(d_metrics); dev_free(d_scratch);
    }

    printf("index,cumulative_return,sharpe,annualized_volatility,max_drawdown,turnover,annual_turnover,n_trades\n");
    for (int s = 0; s < n_strats; ++s) {
        const Metrics& m = results[s];
        printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", s, m.cumulative_return, m.sharpe,
               m.annualized_volatility, m.max_drawdown, m.turnover, m.annual_turnover, m.n_trades);
    }
    return 0;
}
