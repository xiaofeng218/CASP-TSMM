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
// my_shape_dispatch: Self-contained shape-adaptive dispatcher
//
// All kernels are static for optimal compiler inlining.
// Category A: tiny m×n + large k → thread-private + k-parallel + SIMD
// Category B: extreme k ≥ 100K → panel streaming
// Category C/D: general → 8×16 register-blocked tiling (static copy)
// ============================================================

namespace {

constexpr int IB = 8, JB = 16;
constexpr int PANEL = 2048;

// ===== 8×16 register-blocked tiling (static, for compiler optimization) =====

void dispatch_tiled_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB, j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);
#ifdef __AVX512F__
            if (jlen == 8 || jlen >= 16) {
                __m512d acc0[IB], acc1[IB];
                for (int ii = 0; ii < ilen; ++ii) {
                    acc0[ii] = _mm512_setzero_pd();
                    acc1[ii] = _mm512_setzero_pd();
                }
                for (int l = 0; l < k; ++l) {
                    const double* br = B + static_cast<std::size_t>(l) * n + j0;
                    const double* ar = A + static_cast<std::size_t>(l) * m + i0;
                    const __m512d b0 = _mm512_loadu_pd(br);
                    const __m512d b1 = (jlen >= 16) ? _mm512_loadu_pd(br + 8) : _mm512_setzero_pd();
                    for (int ii = 0; ii < ilen; ++ii) {
                        const __m512d av = _mm512_set1_pd(ar[ii]);
                        acc0[ii] = _mm512_fmadd_pd(av, b0, acc0[ii]);
                        if (jlen >= 16) acc1[ii] = _mm512_fmadd_pd(av, b1, acc1[ii]);
                    }
                }
                for (int ii = 0; ii < ilen; ++ii) {
                    double* cr = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    _mm512_storeu_pd(cr, acc0[ii]);
                    if (jlen >= 16) {
                        const int rem = jlen - 16;
                        if (rem == 0) _mm512_storeu_pd(cr + 8, acc1[ii]);
                        else _mm512_mask_storeu_pd(cr + 8, static_cast<__mmask8>((1u << rem) - 1), acc1[ii]);
                    }
                }
                continue;
            }
#endif
            for (int ii = 0; ii < ilen; ++ii) {
                double* cr = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                for (int l = 0; l < k; ++l) {
                    const double av = A[static_cast<std::size_t>(l) * m + i0 + ii];
                    const double* br = B + static_cast<std::size_t>(l) * n + j0;
                    for (int jj = 0; jj < jlen; ++jj) cr[jj] += av * br[jj];
                }
            }
        }
    }
}

#ifdef __AVX512F__
static inline double dot512(const double* a, const double* b, int k) {
    int l = 0;
    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
    for (; l + 15 < k; l += 16) {
        s0 = _mm512_fmadd_pd(_mm512_loadu_pd(a + l), _mm512_loadu_pd(b + l), s0);
        s1 = _mm512_fmadd_pd(_mm512_loadu_pd(a + l + 8), _mm512_loadu_pd(b + l + 8), s1);
    }
    double sum = _mm512_reduce_add_pd(s0) + _mm512_reduce_add_pd(s1);
    for (; l + 7 < k; l += 8)
        sum += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_loadu_pd(a + l), _mm512_loadu_pd(b + l)));
    for (; l < k; ++l) sum += a[l] * b[l];
    return sum;
}
#else
static inline double dot512(const double* a, const double* b, int k) {
    double s = 0.0; for (int l = 0; l < k; ++l) s += a[l] * b[l]; return s;
}
#endif

void dispatch_dot_col(int m, int n, int k, const double* A, const double* B, double* C) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            C[static_cast<std::size_t>(j) * m + i] =
                dot512(A + static_cast<std::size_t>(i) * k, B + static_cast<std::size_t>(j) * k, k);
}

// ===== Category A: tiny m×n + large k =====

void cat_a_row(int m, int n, int k, const double* A, const double* B, double* C) {
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
                    __m512d cv = _mm512_maskz_loadu_pd(mk, cr + j), bv = _mm512_maskz_loadu_pd(mk, br + j);
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

void cat_a_col(int m, int n, int k, const double* A, const double* B, double* C) {
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
        for (int l = 0; l < k; ++l)
            for (int i = 0; i < m; ++i) {
                const double av = A[static_cast<std::size_t>(i) * k + l];
                for (int j = 0; j < n; ++j)
                    ct[static_cast<std::size_t>(j) * m + i] += av * B[static_cast<std::size_t>(j) * k + l];
            }
    }
    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

// ===== Category B: extreme k + panel streaming =====

void cat_b_row(int m, int n, int k, const double* A, const double* B, double* C) {
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
        for (int kb = 0; kb < k; kb += PANEL) {
            const int kbe = std::min(kb + PANEL, k);
#pragma omp for schedule(static)
            for (int l = kb; l < kbe; ++l) {
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
                        __m512d cv = _mm512_maskz_loadu_pd(mk, cr + j), bv = _mm512_maskz_loadu_pd(mk, br + j);
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
    }
    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

void cat_b_col(int m, int n, int k, const double* A, const double* B, double* C) {
    cat_a_col(m, n, k, A, B, C);
}

} // anonymous namespace

// ===================================================================
// Public dispatcher
// ===================================================================

void tsmm_my_shape_dispatch(int m, int n, int k,
                             const double* A, const double* B, double* C,
                             Layout layout) {
    const int osz = m * n;
    const bool tiny = (osz <= 256);
    const bool bigk = (k >= 10000);
    const bool extr = (k >= 100000);

    if (layout == Layout::RowMajor) {
        if (tiny && extr)    cat_b_row(m, n, k, A, B, C);
        else if (tiny && bigk) cat_a_row(m, n, k, A, B, C);
        else                  dispatch_tiled_row(m, n, k, A, B, C);
    } else {
        if (tiny && extr)    cat_b_col(m, n, k, A, B, C);
        else if (tiny && bigk) cat_a_col(m, n, k, A, B, C);
        else                  dispatch_dot_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_shape_dispatch", tsmm_my_shape_dispatch);
