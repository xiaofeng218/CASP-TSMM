BLAS ?= openblas
CXX ?= g++
OBJDIR ?= obj
TARGET = $(OBJDIR)/benchmark

CXXFLAGS = -O3 -march=native -std=c++17 -ffast-math -fopenmp -ipo
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -fopenmp -lm -lpthread
CXXLIBS ?= -lstdc++

ifeq ($(BLAS),mkl)
  ifndef MKLROOT
    $(error MKLROOT is not set. Run: source /opt/intel/mkl/bin/mklvars.sh intel64)
  endif
  CXXFLAGS += -DHAVE_MKL -I$(MKLROOT)/include
  LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -ldl
else ifeq ($(BLAS),openblas)
  CXXFLAGS += -DHAVE_OPENBLAS
  LDFLAGS += -lopenblas
else ifeq ($(BLAS),none)
  CXXFLAGS += -UHAVE_MKL -UHAVE_OPENBLAS
else
  $(error BLAS must be one of: mkl, openblas, none)
endif

ifeq ($(AVX512),1)
  CXXFLAGS += -mavx512f -mavx512dq -mavx512bw -mavx512vl
endif

SRC = $(wildcard src/*.cpp) $(wildcard src/tsmm/*.cpp)
OBJ = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRC))
DEP = $(OBJ:.o=.d)

.PHONY: all run run-required web clean clean-results help

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS) $(CXXLIBS)
	@echo "Built $(TARGET) [BLAS=$(BLAS)]"

$(OBJDIR)/%.o: %.cpp src/tsmm.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	bash scripts/run_local.sh --all

run-required: $(TARGET)
	bash scripts/run_local.sh --required-only

web:
	python3 web/server.py --port 8080 --results web/results

clean:
	rm -rf $(OBJDIR)
	rm -f benchmark benchmark.exe benchmark_check.exe
	rm -rf web/gflops_*.csv web/gflops_*.json

clean-results:
	rm -rf web/results logs

help:
	@echo "Targets: all run run-required web clean clean-results"
	@echo "Options: BLAS=mkl|openblas|none AVX512=1 OBJDIR=obj CXXLIBS=-lstdc++"
	@echo "Benchmark args example: ./obj/benchmark --required-only --layout row --warmup 10 --runs 20"

-include $(DEP)
