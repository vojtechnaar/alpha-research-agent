# Native backtest backends, loaded by src/backends/native.py.
#
#   make              GPU backend  -> build/libbacktest_cuda.so  (needs nvcc; RTX A6000: ARCH=sm_86)
#   make cpu          C++ backend  -> build/libbacktest_cpu.so   (single thread)
#   make cpu OMP=1    C++ backend with OpenMP (all cores; needs a compiler with -fopenmp, e.g. gcc)
#   make clean
NVCC ?= nvcc
ARCH ?= native
CXX ?= c++
OMP ?= 0

CPU_FLAGS := -O3 -std=c++17 -shared -fPIC -Wall -Wextra -Wno-unknown-pragmas
ifeq ($(OMP),1)
CPU_FLAGS += -fopenmp
endif

build/libbacktest_cuda.so: backtest.cu
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -shared -Xcompiler -fPIC -o $@ $<

cpu: build/libbacktest_cpu.so

build/libbacktest_cpu.so: backtest.cu
	mkdir -p build
	$(CXX) $(CPU_FLAGS) -x c++ -o $@ $<

clean:
	rm -rf build

.PHONY: cpu clean
