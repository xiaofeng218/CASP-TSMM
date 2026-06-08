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

// Forward declarations from my_blocking.cpp
void blocking_tiled_row(int m, int n, int k, const double* A, const double* B, double* C);
void blocking_tiled_col(int m, int n, int k, const double* A, const double* B, double* C);

// ============================================================
// my_shape_dispatch: Shape-adaptive dispatcher
//
//   Category A: tiny m×n + large_k    → thread-private + k-parallel + SIMD
//   Category B: extreme k (≥100K)     → panel streaming
//   Category C: square-like           → optimized tiling
//   Category D: general               → optimized tiling
// ============================================================

namespace {

constexpr int PANEL = 2048;

// ===== Category A: tiny m×n (≤256) + large k =====

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
    }
    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

void cat_b_col(int m, int n, int k, const double* A, const double* B, double* C) {
    cat_a_col(m, n, k, A, B, C); // tiny C, same strategy
}

} // anonymous namespace

// =====================================================================
// Public dispatcher
// =====================================================================

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
        else                  blocking_tiled_row(m, n, k, A, B, C);
    } else {
        if (tiny && extr)    cat_b_col(m, n, k, A, B, C);
        else if (tiny && bigk) cat_a_col(m, n, k, A, B, C);
        else                  blocking_tiled_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_shape_dispatch", tsmm_my_shape_dispatch);
