#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// ============================================================
// my_final: Self-contained optimized TSMM with shape dispatch
//
// All kernels are static → compiler can inline aggressively.
// Row-major tiling: 8×16 register-blocked (identical to opt)
// Col-major: dot-product with AVX-512 unrolling
// Tiny-output: thread-private + k-parallel + AVX-512
// ============================================================

namespace {

constexpr int  IB = 8;
constexpr int  JB = 16;

// ===================================================================
// Row-major: 8×16 register-blocked tiling
// ===================================================================

void final_tiled_row(int m, int n, int k,
                      const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0   = bi * IB;
            const int j0   = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);

#ifdef __AVX512F__
            if (jlen == 8 || jlen >= 16) {
                __m512d acc0[IB];
                __m512d acc1[IB];
                for (int ii = 0; ii < ilen; ++ii) {
                    acc0[ii] = _mm512_setzero_pd();
                    acc1[ii] = _mm512_setzero_pd();
                }
                for (int l = 0; l < k; ++l) {
                    const double* b_row = B + static_cast<std::size_t>(l) * n + j0;
                    const double* a_row = A + static_cast<std::size_t>(l) * m + i0;
                    const __m512d b0 = _mm512_loadu_pd(b_row);
                    const __m512d b1 = (jlen >= 16) ? _mm512_loadu_pd(b_row + 8)
                                                     : _mm512_setzero_pd();
                    for (int ii = 0; ii < ilen; ++ii) {
                        const __m512d av = _mm512_set1_pd(a_row[ii]);
                        acc0[ii] = _mm512_fmadd_pd(av, b0, acc0[ii]);
                        if (jlen >= 16)
                            acc1[ii] = _mm512_fmadd_pd(av, b1, acc1[ii]);
                    }
                }
                for (int ii = 0; ii < ilen; ++ii) {
                    double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    _mm512_storeu_pd(c_row, acc0[ii]);
                    if (jlen >= 16) {
                        const int rem = jlen - 16;
                        if (rem == 0)
                            _mm512_storeu_pd(c_row + 8, acc1[ii]);
                        else
                            _mm512_mask_storeu_pd(c_row + 8,
                                static_cast<__mmask8>((1u << rem) - 1), acc1[ii]);
                    }
                }
                continue;
            }
#endif
            // Scalar fallback
            for (int ii = 0; ii < ilen; ++ii) {
                double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                for (int l = 0; l < k; ++l) {
                    const double av = A[static_cast<std::size_t>(l) * m + i0 + ii];
                    const double* b_row = B + static_cast<std::size_t>(l) * n + j0;
                    for (int jj = 0; jj < jlen; ++jj)
                        c_row[jj] += av * b_row[jj];
                }
            }
        }
    }
}

// ===================================================================
// Col-major: dot-product based kernel with AVX-512
// ===================================================================

#ifdef __AVX512F__
static inline double dot_avx512(const double* a, const double* b, int k) {
    int l = 0;
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();
    for (; l + 15 < k; l += 16) {
        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + l),
                                _mm512_loadu_pd(b + l), acc0);
        acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + l + 8),
                                _mm512_loadu_pd(b + l + 8), acc1);
    }
    double sum = _mm512_reduce_add_pd(acc0) + _mm512_reduce_add_pd(acc1);
    for (; l + 7 < k; l += 8) {
        __m512d av = _mm512_loadu_pd(a + l);
        __m512d bv = _mm512_loadu_pd(b + l);
        sum += _mm512_reduce_add_pd(_mm512_mul_pd(av, bv));
    }
    for (; l < k; ++l) sum += a[l] * b[l];
    return sum;
}
#else
static inline double dot_scalar(const double* a, const double* b, int k) {
    double s = 0.0;
    for (int l = 0; l < k; ++l) s += a[l] * b[l];
    return s;
}
#define dot_avx512 dot_scalar
#endif

void final_dot_col(int m, int n, int k,
                    const double* A, const double* B, double* C) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[static_cast<std::size_t>(j) * m + i] =
                dot_avx512(A + static_cast<std::size_t>(i) * k,
                            B + static_cast<std::size_t>(j) * k, k);
        }
    }
}

// ===================================================================
// Tiny-output: thread-private C_local + k-parallel
// ===================================================================

void final_tiny_row(int m, int n, int k,
                     const double* A, const double* B, double* C) {
    const int nth = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int sz = m * n;
    std::vector<double> t(static_cast<std::size_t>(nth) * sz, 0.0);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = t.data() + static_cast<std::size_t>(tid) * sz;
#pragma omp for schedule(static)
        for (int l = 0; l < k; ++l) {
            const double* ar = A + static_cast<std::size_t>(l) * m;
            const double* br = B + static_cast<std::size_t>(l) * n;
#ifdef __AVX512F__
            for (int i = 0; i < m; ++i) {
                const __m512d av = _mm512_set1_pd(ar[i]);
                double* cr = ct + static_cast<std::size_t>(i) * n;
                int j = 0;
                for (; j + 31 < n; j += 32) {
                    __m512d c0 = _mm512_loadu_pd(cr + j), c1 = _mm512_loadu_pd(cr + j + 8);
                    __m512d c2 = _mm512_loadu_pd(cr + j + 16), c3 = _mm512_loadu_pd(cr + j + 24);
                    c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j), c0);
                    c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j + 8), c1);
                    c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j + 16), c2);
                    c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j + 24), c3);
                    _mm512_storeu_pd(cr + j, c0); _mm512_storeu_pd(cr + j + 8, c1);
                    _mm512_storeu_pd(cr + j + 16, c2); _mm512_storeu_pd(cr + j + 24, c3);
                }
                for (; j + 7 < n; j += 8) {
                    __m512d cv = _mm512_loadu_pd(cr + j);
                    cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j), cv);
                    _mm512_storeu_pd(cr + j, cv);
                }
                if (j < n) {
                    const __mmask8 mk = static_cast<__mmask8>((1u << (n - j)) - 1);
                    __m512d cv = _mm512_maskz_loadu_pd(mk, cr + j);
                    __m512d bv = _mm512_maskz_loadu_pd(mk, br + j);
                    cv = _mm512_fmadd_pd(av, bv, cv);
                    _mm512_mask_storeu_pd(cr + j, mk, cv);
                }
            }
#else
            for (int i = 0; i < m; ++i) {
                const double av = ar[i];
                double* cr = ct + static_cast<std::size_t>(i) * n;
                for (int j = 0; j < n; ++j) cr[j] += av * br[j];
            }
#endif
        }
    }
    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

void final_tiny_col(int m, int n, int k,
                     const double* A, const double* B, double* C) {
    // For col-major tiny: use dot-product with AVX-512
    // Same as final_dot_col but m*n is tiny so parallelize differently
    const int nth = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int sz = m * n;
    std::vector<double> t(static_cast<std::size_t>(nth) * sz, 0.0);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = t.data() + static_cast<std::size_t>(tid) * sz;
        // Parallelize over k via omp for, each thread does dot on its k-segment
#pragma omp for schedule(static)
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < m; ++i) {
                const double av = A[static_cast<std::size_t>(i) * k + l];
                for (int j = 0; j < n; ++j)
                    ct[static_cast<std::size_t>(j) * m + i] +=
                        av * B[static_cast<std::size_t>(j) * k + l];
            }
        }
    }
    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

} // anonymous namespace

// ===================================================================
// Public entry point
// ===================================================================

void tsmm_my_final(int m, int n, int k,
                    const double* A, const double* B, double* C,
                    Layout layout) {
    const bool tiny  = (m * n <= 256);
    const bool big_k = (k >= 10000);

    if (layout == Layout::RowMajor) {
        if (tiny && big_k)  final_tiny_row(m, n, k, A, B, C);
        else                final_tiled_row(m, n, k, A, B, C);
    } else {
        if (tiny && big_k)  final_tiny_col(m, n, k, A, B, C);
        else                final_dot_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_final", tsmm_my_final);
