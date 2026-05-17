#include "../tsmm.hpp"

#include <cstring>

void tsmm_naive(int m, int n, int k,
                const double* A, const double* B, double* C,
                Layout layout) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                sum += A[idx_a(layout, l, i, k, m)] * B[idx_b(layout, l, j, k, n)];
            }
            C[idx_c(layout, i, j, m, n)] = sum;
        }
    }
}

// This reference implementation is intentionally not registered.
// It is kept as a simple template for adding new kernels.
//
// To include it in benchmark results, uncomment:
// REGISTER_TSMM_IMPL("naive", tsmm_naive);
