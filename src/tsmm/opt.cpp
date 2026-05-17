#include "../tsmm.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static void opt_avx512_row_omp(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
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

static void opt_blocked_col(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    static int IB = 0;
    static int JB = 0;
    static int LB = 0;
    if (IB == 0) {
        IB = std::getenv("OPT_COL_IB") ? std::atoi(std::getenv("OPT_COL_IB")) : 64;
        JB = std::getenv("OPT_COL_JB") ? std::atoi(std::getenv("OPT_COL_JB")) : 512;
        LB = std::getenv("OPT_COL_LB") ? std::atoi(std::getenv("OPT_COL_LB")) : 32;
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
                        const double av = A[idx_a(Layout::ColMajor, l, i, k, m)];
                        for (int j = j0; j < j0 + jlen; ++j) {
                            C[idx_c(Layout::ColMajor, i, j, m, n)] +=
                                av * B[idx_b(Layout::ColMajor, l, j, k, n)];
                        }
                    }
                }
            }
        }
    }
}

static void tsmm_opt_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m == 144 && n == 144 && k == 144) {
        opt_avx512_row_omp(m, n, k, A, B, C);
        return;
    }

    static int IB = 0;
    static int JB = 0;
    static int LB = 0;
    if (IB == 0) {
        IB = std::getenv("OPT_IB") ? std::atoi(std::getenv("OPT_IB")) : 64;
        JB = std::getenv("OPT_JB") ? std::atoi(std::getenv("OPT_JB")) : 512;
        LB = std::getenv("OPT_LB") ? std::atoi(std::getenv("OPT_LB")) : 32;
    }

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    if (m <= 64 && k <= 64) {
        opt_avx512_row_omp(m, n, k, A, B, C);
        return;
    }

    if (m <= nthreads && m < 64) {
        std::vector<std::vector<double>> tmp(nthreads, std::vector<double>(static_cast<std::size_t>(m) * n, 0.0));
#ifdef _OPENMP
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* ct = tmp[tid].data();
#pragma omp for schedule(static)
            for (int l = 0; l < k; ++l) {
                const double* a = A + static_cast<std::size_t>(l) * m;
                const double* b = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    double* c = ct + static_cast<std::size_t>(i) * n;
                    int j = 0;
#ifdef __AVX512F__
                    const __m512d av = _mm512_set1_pd(a[i]);
                    for (; j + 7 < n; j += 8) {
                        __m512d cv = _mm512_loadu_pd(c + j);
                        cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                        _mm512_storeu_pd(c + j, cv);
                    }
#endif
                    for (; j < n; ++j) {
                        c[j] += a[i] * b[j];
                    }
                }
            }
        }
#else
        double* ct = tmp[0].data();
        for (int l = 0; l < k; ++l) {
            const double* a = A + static_cast<std::size_t>(l) * m;
            const double* b = B + static_cast<std::size_t>(l) * n;
            for (int i = 0; i < m; ++i) {
                double* c = ct + static_cast<std::size_t>(i) * n;
                for (int j = 0; j < n; ++j) c[j] += a[i] * b[j];
            }
        }
#endif
        std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
        for (int t = 0; t < nthreads; ++t) {
            const double* ct = tmp[t].data();
            for (std::size_t i = 0; i < static_cast<std::size_t>(m) * n; ++i) {
                C[i] += ct[i];
            }
        }
        return;
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
                    const double* b = B + static_cast<std::size_t>(l) * n + j0;
                    for (int i = i0; i < i0 + ilen; ++i) {
                        const double av_d = A[static_cast<std::size_t>(l) * m + i];
                        double* c = C + static_cast<std::size_t>(i) * n + j0;
                        int j = 0;
#ifdef __AVX512F__
                        const __m512d av = _mm512_set1_pd(av_d);
                        for (; j + 31 < jlen; j += 32) {
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
                        for (; j + 7 < jlen; j += 8) {
                            __m512d cv = _mm512_loadu_pd(c + j);
                            cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                            _mm512_storeu_pd(c + j, cv);
                        }
#endif
                        for (; j < jlen; ++j) {
                            c[j] += av_d * b[j];
                        }
                    }
                }
            }
        }
    }
}

static void tsmm_opt_col_dot(int m, int n, int k,
                             const double* A, const double* B, double* C) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n; ++j) {
        const double* b = B + static_cast<std::size_t>(j) * k;
        double* c = C + static_cast<std::size_t>(j) * m;
        for (int i = 0; i < m; ++i) {
            const double* a = A + static_cast<std::size_t>(i) * k;
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;
            int l = 0;
            for (; l + 3 < k; l += 4) {
                sum0 += a[l] * b[l];
                sum1 += a[l + 1] * b[l + 1];
                sum2 += a[l + 2] * b[l + 2];
                sum3 += a[l + 3] * b[l + 3];
            }
            double sum = (sum0 + sum1) + (sum2 + sum3);
            for (; l < k; ++l) {
                sum += a[l] * b[l];
            }
            c[i] = sum;
        }
    }
}

static void tsmm_opt_col(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    if (m <= 64 || (m <= 256 && n <= 1024)) {
        tsmm_opt_col_dot(m, n, k, A, B, C);
        return;
    }

    opt_blocked_col(m, n, k, A, B, C);
}

void tsmm_opt(int m, int n, int k,
              const double* A, const double* B, double* C,
              Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_opt_row(m, n, k, A, B, C);
    } else {
        tsmm_opt_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("opt", tsmm_opt);
