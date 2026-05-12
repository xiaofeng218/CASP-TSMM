#include "tsmm.hpp"

#include <cstring>

void tsmm_naive(int m, int n, int k,
                const double* A, const double* B, double* C,
                Layout layout) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    for (int l = 0; l < k; ++l) {
        for (int i = 0; i < m; ++i) {
            const double av = A[idx_a(layout, l, i, k, m)];
            for (int j = 0; j < n; ++j) {
                C[idx_c(layout, i, j, m, n)] += av * B[idx_b(layout, l, j, k, n)];
            }
        }
    }
}

