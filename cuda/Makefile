# GPU build (on the server):   make            (override the GPU arch with e.g. make ARCH=sm_80)
# CPU build (for parity tests without CUDA):   make cpu
NVCC ?= nvcc
ARCH ?= native
CXX ?= c++

build/backtest: backtest.cu
	mkdir -p build
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

cpu: build/backtest_cpu

build/backtest_cpu: backtest.cu
	mkdir -p build
	$(CXX) -O2 -std=c++17 -Wall -Wextra -x c++ -o $@ $<

clean:
	rm -rf build

.PHONY: cpu clean
