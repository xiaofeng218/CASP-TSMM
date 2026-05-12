BLAS ?= openblas
CXX ?= g++
TARGET = benchmark

CXXFLAGS = -O3 -march=native -std=c++17 -ffast-math -fopenmp
CXXFLAGS += -Wall -Wextra -Wno-unused-parameter
LDFLAGS = -fopenmp -lm -lpthread

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

SRC = $(wildcard src/*.cpp)

.PHONY: all run run-required web clean help

all: $(TARGET)

$(TARGET): $(SRC) src/tsmm.hpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)
	@echo "Built $(TARGET) [BLAS=$(BLAS)]"

run: $(TARGET)
	@mkdir -p web
	./$(TARGET) --output web/results.json --all

run-required: $(TARGET)
	@mkdir -p web
	./$(TARGET) --output web/results.json --required-only

web:
	python3 web/server.py --port 8080 --results web/results.json

clean:
	rm -f $(TARGET) $(TARGET).exe web/results.json

help:
	@echo "Targets: all run run-required web clean"
	@echo "Options: BLAS=mkl|openblas|none AVX512=1"
	@echo "Benchmark args example: ./benchmark --required-only --layout row --warmup 10 --runs 20"
