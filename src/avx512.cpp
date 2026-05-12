#include "tsmm.hpp"

#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static void tsmm_avx512_row(int m, int n, int k,
                            const double* A, const double* B, double* C,
                            bool use_openmp) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(use_openmp)
#endif
    for (int i = 0; i < m; ++i) {
        double* c = C + static_cast<std::size_t>(i) * n;
        for (int l = 0; l < k; ++l) {
            const double av_d = A[static_cast<std::size_t>(l) * m + i];
            const double* b = B + static_cast<std::size_t>(l) * n;
            int j = 0;
#ifdef __AVX512F__
            const __m512d av = _mm512_set1_pd(av_d);
            for (; j + 31 < n; j += 32) {
                __m512d c0 = _mm512_loadu_pd(c + j);
                __m512d c1 = _mm512_loadu_pd(c + j + 8);
                __m512d c2 = _mm512_loadu_pd(c + j + 16);
                __m512d c3 = _mm512_loadu_pd(c + j + 24);
                c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), c0);
                c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 8), c1);
                c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 16), c2);
                c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 24), c3);
                _mm512_storeu_pd(c + j, c0);
                _mm512_storeu_pd(c + j + 8, c1);
                _mm512_storeu_pd(c + j + 16, c2);
                _mm512_storeu_pd(c + j + 24, c3);
            }
            for (; j + 7 < n; j += 8) {
                __m512d cv = _mm512_loadu_pd(c + j);
                cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                _mm512_storeu_pd(c + j, cv);
            }
#endif
            for (; j < n; ++j) {
                c[j] += av_d * b[j];
            }
        }
    }
}

void tsmm_avx512(int m, int n, int k,
                 const double* A, const double* B, double* C,
                 Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_avx512_row(m, n, k, A, B, C, false);
    } else {
        tsmm_naive(m, n, k, A, B, C, layout);
    }
}

void tsmm_avx512_omp(int m, int n, int k,
                     const double* A, const double* B, double* C,
                     Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_avx512_row(m, n, k, A, B, C, true);
    } else {
        tsmm_openmp(m, n, k, A, B, C, layout);
    }
}

