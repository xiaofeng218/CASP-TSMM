#include "tsmm.hpp"

#include <cstring>

#if defined(HAVE_MKL)
#include <mkl_cblas.h>
#elif defined(HAVE_OPENBLAS)
#include <cblas.h>
#endif

void tsmm_reference(int m, int n, int k,
                    const double* A, const double* B, double* C,
                    Layout layout) {
#if defined(HAVE_MKL) || defined(HAVE_OPENBLAS)
    if (layout == Layout::RowMajor) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m, n, k, 1.0, A, m, B, n, 0.0, C, n);
    } else {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, n, k, 1.0, A, k, B, k, 0.0, C, m);
    }
#else
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    for (int l = 0; l < k; ++l) {
        for (int i = 0; i < m; ++i) {
            const double av = A[idx_a(layout, l, i, k, m)];
            for (int j = 0; j < n; ++j) {
                C[idx_c(layout, i, j, m, n)] += av * B[idx_b(layout, l, j, k, n)];
            }
        }
    }
#endif
}

