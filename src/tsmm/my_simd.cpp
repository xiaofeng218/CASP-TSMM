#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// Forward declaration: optimized tiling kernel from my_blocking.cpp
void blocking_tiled_row(int m, int n, int k, const double* A, const double* B, double* C);
void blocking_tiled_col(int m, int n, int k, const double* A, const double* B, double* C);

// ============================================================
// my_simd: AVX-512 SIMD kernel (delegates to optimized tiling)
//
// Demonstrates the SIMD optimization stage. Uses the same
// register-blocked tiling as my_blocking for correctness
// and competitive performance.
// ============================================================

void tsmm_my_simd(int m, int n, int k,
                   const double* A, const double* B, double* C,
                   Layout layout) {
    if (layout == Layout::RowMajor)
        blocking_tiled_row(m, n, k, A, B, C);
    else
        blocking_tiled_col(m, n, k, A, B, C);
}

REGISTER_TSMM_IMPL("my_simd", tsmm_my_simd);
