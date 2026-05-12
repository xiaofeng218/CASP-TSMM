#include "tsmm.hpp"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

void tsmm_openmp(int m, int n, int k,
                 const double* A, const double* B, double* C,
                 Layout layout) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < k; ++l) {
            const double av = A[idx_a(layout, l, i, k, m)];
            for (int j = 0; j < n; ++j) {
                C[idx_c(layout, i, j, m, n)] += av * B[idx_b(layout, l, j, k, n)];
            }
        }
    }
}

