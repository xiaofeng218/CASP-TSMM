#include "tsmm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

void tsmm_blocked(int m, int n, int k,
                  const double* A, const double* B, double* C,
                  Layout layout) {
    static int IB = 0;
    static int JB = 0;
    static int LB = 0;
    if (IB == 0) {
        IB = std::getenv("TSMM_IB") ? std::atoi(std::getenv("TSMM_IB")) : 64;
        JB = std::getenv("TSMM_JB") ? std::atoi(std::getenv("TSMM_JB")) : 512;
        LB = std::getenv("TSMM_LB") ? std::atoi(std::getenv("TSMM_LB")) : 32;
    }

    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int j0 = 0; j0 < n; j0 += JB) {
        const int jlen = std::min(JB, n - j0);
        for (int i0 = 0; i0 < m; i0 += IB) {
            const int ilen = std::min(IB, m - i0);
            for (int l0 = 0; l0 < k; l0 += LB) {
                const int llen = std::min(LB, k - l0);
                for (int l = l0; l < l0 + llen; ++l) {
                    for (int i = i0; i < i0 + ilen; ++i) {
                        const double av = A[idx_a(layout, l, i, k, m)];
                        for (int j = j0; j < j0 + jlen; ++j) {
                            C[idx_c(layout, i, j, m, n)] += av * B[idx_b(layout, l, j, k, n)];
                        }
                    }
                }
            }
        }
    }
}

