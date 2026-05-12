# CASP-TSMM

Tall-Skinny Matrix Multiplication benchmark and optimization project.

The project evaluates:

```text
C = A^T * B
A: k x m, B: k x n, C: m x n, double precision
```

Both row-major and column-major storage are supported by the benchmark. The optimized kernels focus on the row-major path used by the default run.

## Project Layout

```text
CASP-TSMM/
|-- src/
|   |-- benchmark.cpp       # benchmark harness, timing, correctness, JSON output
|   |-- tsmm.hpp            # shared kernel interface and layout helpers
|   |-- reference.cpp       # MKL/OpenBLAS dgemm reference, or built-in fallback
|   |-- naive.cpp           # serial baseline
|   |-- openmp_kernel.cpp   # OpenMP parallel baseline
|   |-- blocked.cpp         # cache-blocked OpenMP kernel
|   |-- avx512.cpp          # AVX-512 kernels
|   `-- opt.cpp             # combined AVX-512/OpenMP/blocking kernel
|-- scripts/
|   |-- run_local.sh        # local benchmark and dashboard runner
|   `-- submit_slurm.sh     # Slurm submission helper for the target cluster
|-- web/
|   |-- index.html          # live dashboard
|   `-- server.py           # lightweight Python server, no Docker
`-- Makefile
```

Generated files such as `benchmark`, `benchmark.exe`, `web/results.json`, and `logs/` are not source files.

## Implementations

| Name | Description |
| --- | --- |
| `reference` | CBLAS `dgemm` reference using MKL/OpenBLAS, or a built-in fallback with `BLAS=none` |
| `naive` | serial three-loop TSMM |
| `openmp` | OpenMP parallelization over rows of C |
| `blocked` | cache-blocked OpenMP kernel, tunable with `TSMM_IB/JB/LB` |
| `avx512` | row-major AVX-512 vectorized kernel |
| `avx512_omp` | AVX-512 plus OpenMP |
| `opt` | best-effort combined kernel with special handling for small `m` |

## Build And Run

```bash
# OpenBLAS, default
make BLAS=openblas
./benchmark --output web/results.json --required-only

# Intel MKL on the target cluster
source /opt/intel/mkl/bin/mklvars.sh intel64
make BLAS=mkl AVX512=1
./benchmark --output web/results.json --required-only

# No external BLAS, useful for smoke tests
make BLAS=none
./benchmark --output web/results.json --required-only --warmup 1 --runs 1
```

Benchmark options:

```text
--required-only       run only the four required problems
--all                 run required and optional problems
--layout row|col      choose row-major or column-major storage
--warmup N            warmup iterations, default 10
--runs N              timed iterations, default 20
--no-correctness      skip comparison with the reference result
```

For very large problems the harness automatically caps warmup to 3 and timed runs to 5 to keep evaluation practical.

## Dashboard

```bash
make web
# open http://localhost:8080
```

Or run benchmark and dashboard together:

```bash
bash scripts/run_local.sh --required-only
```

The benchmark writes incremental updates to `web/results.json`, and the dashboard polls it every three seconds.

## Slurm Target Run

The target CPU is Intel Xeon Platinum 9242 with 96 cores, 4 NUMA nodes, and AVX-512. The submission helper requests all 96 CPUs and binds execution across NUMA nodes `0-3` with interleaved memory placement:

```bash
bash scripts/submit_slurm.sh              # required problems
bash scripts/submit_slurm.sh --all        # required + optional
bash scripts/submit_slurm.sh --dry-run    # print the generated job script
```

Useful environment overrides:

```bash
PARTITION=cpu CPUS_PER_TASK=96 NUMA_NODES=0-3 BLAS=mkl bash scripts/submit_slurm.sh
```

The final metric is GFLOPS per task and speedup versus the `reference` `dgemm` baseline. The dashboard also reports the geometric mean speedup across the required problems.
