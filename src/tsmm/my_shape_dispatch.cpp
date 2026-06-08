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
// my_shape_dispatch: Shape-adaptive TSMM dispatcher
//
// Category A: tiny_output + large_k   → thread-private C_local + k-parallel
// Category B: extreme_k               → panel streaming
// Category C: square-like             → m×n tiling + packing + SIMD
// Category D: general                 → m×n tiling + k-blocking
// ============================================================

namespace {

constexpr int TILE_M = 16, TILE_N = 64;
constexpr int L2_D   = 1024 * 1024 / 8;
constexpr int PANEL  = 2048;

int tile_bk(int tm, int tn, int k) {
    int d = tm + tn; if (d <= 0) d = 1;
    int bk = (L2_D - tm * tn) / d;
    if (bk < 8) bk = 8; bk = (bk / 8) * 8;
    if (bk > k) bk = k;
    return bk;
}

// ===== Category A: tiny output (m*n <= 256) + large_k =====

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
                for (; j + 7 < n; j += 8) {
                    __m512d cv = _mm512_loadu_pd(cr + j);
                    cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j), cv);
                    _mm512_storeu_pd(cr + j, cv);
                }
                for (; j < n; ++j) cr[j] += ar[i] * br[j];
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
    cat_a_col(m, n, k, A, B, C); // delegate: tiny C, same approach
}

// ===== Category C: square-like → tiling + packing + SIMD =====

void cat_c_row(int m, int n, int k, const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int TM = TILE_M, TN = TILE_N;
    const int nbi = (m + TM - 1) / TM, nbj = (n + TN - 1) / TN;
    const int BK  = tile_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TM, j0 = bj * TN;
            const int imax = std::min(i0 + TM, m), jmax = std::min(j0 + TN, n);
            const int tm = imax - i0, tn = jmax - j0;
            for (int kb = 0; kb < k; kb += BK) {
                const int kbl = std::min(BK, k - kb);
                std::vector<double> bp(static_cast<std::size_t>(kbl) * tn);
                for (int l = 0; l < kbl; ++l)
                    std::memcpy(bp.data() + static_cast<std::size_t>(l) * tn,
                                B + static_cast<std::size_t>(kb + l) * n + j0,
                                static_cast<std::size_t>(tn) * sizeof(double));
                for (int l = 0; l < kbl; ++l) {
                    const double* ar = A + static_cast<std::size_t>(kb + l) * m;
                    const double* bd = bp.data() + static_cast<std::size_t>(l) * tn;
#ifdef __AVX512F__
                    for (int i = 0; i < tm; ++i) {
                        const __m512d av = _mm512_set1_pd(ar[i0 + i]);
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        int j = 0;
                        for (; j + 31 < tn; j += 32) {
                            __m512d c0 = _mm512_loadu_pd(cr + j), c1 = _mm512_loadu_pd(cr + j + 8);
                            __m512d c2 = _mm512_loadu_pd(cr + j + 16), c3 = _mm512_loadu_pd(cr + j + 24);
                            c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j), c0);
                            c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j + 8), c1);
                            c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j + 16), c2);
                            c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j + 24), c3);
                            _mm512_storeu_pd(cr + j, c0); _mm512_storeu_pd(cr + j + 8, c1);
                            _mm512_storeu_pd(cr + j + 16, c2); _mm512_storeu_pd(cr + j + 24, c3);
                        }
                        for (; j + 7 < tn; j += 8) {
                            __m512d cv = _mm512_loadu_pd(cr + j);
                            cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j), cv);
                            _mm512_storeu_pd(cr + j, cv);
                        }
                        if (j < tn) {
                            const __mmask8 mk = static_cast<__mmask8>((1u << (tn - j)) - 1);
                            __m512d cv = _mm512_maskz_loadu_pd(mk, cr + j);
                            __m512d bv = _mm512_maskz_loadu_pd(mk, bd + j);
                            cv = _mm512_fmadd_pd(av, bv, cv);
                            _mm512_mask_storeu_pd(cr + j, mk, cv);
                        }
                    }
#else
                    for (int i = 0; i < tm; ++i) {
                        const double av = ar[i0 + i];
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        for (int j = 0; j < tn; ++j) cr[j] += av * bd[j];
                    }
#endif
                }
            }
        }
    }
}

void cat_c_col(int m, int n, int k, const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int TM = TILE_M, TN = TILE_N;
    const int nbi = (m + TM - 1) / TM, nbj = (n + TN - 1) / TN;
    const int BK  = tile_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TM, j0 = bj * TN;
            const int imax = std::min(i0 + TM, m), jmax = std::min(j0 + TN, n);
            const int tm = imax - i0, tn = jmax - j0;
            for (int kb = 0; kb < k; kb += BK) {
                const int kbl = std::min(BK, k - kb);
                std::vector<double> ap(static_cast<std::size_t>(tm) * kbl);
                for (int i = 0; i < tm; ++i)
                    std::memcpy(ap.data() + static_cast<std::size_t>(i) * kbl,
                                A + static_cast<std::size_t>(i0 + i) * k + kb,
                                static_cast<std::size_t>(kbl) * sizeof(double));
                for (int l = 0; l < kbl; ++l)
                    for (int i = 0; i < tm; ++i) {
                        const double av = ap[static_cast<std::size_t>(i) * kbl + l];
                        for (int j = 0; j < tn; ++j)
                            C[static_cast<std::size_t>(j0 + j) * m + (i0 + i)] +=
                                av * B[static_cast<std::size_t>(j0 + j) * k + kb + l];
                    }
            }
        }
    }
}

// ===== Category D: general → tiling =====

void cat_d_row(int m, int n, int k, const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int TM = TILE_M, TN = TILE_N;
    const int nbi = (m + TM - 1) / TM, nbj = (n + TN - 1) / TN;
    const int BK  = tile_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TM, j0 = bj * TN;
            const int imax = std::min(i0 + TM, m), jmax = std::min(j0 + TN, n);
            const int tm = imax - i0, tn = jmax - j0;
            for (int kb = 0; kb < k; kb += BK) {
                const int kbe = std::min(kb + BK, k);
                for (int l = kb; l < kbe; ++l) {
                    const double* ar = A + static_cast<std::size_t>(l) * m;
                    const double* br = B + static_cast<std::size_t>(l) * n;
                    for (int i = 0; i < tm; ++i) {
                        const double av = ar[i0 + i];
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        for (int j = 0; j < tn; ++j) cr[j] += av * br[j0 + j];
                    }
                }
            }
        }
    }
}

void cat_d_col(int m, int n, int k, const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int TM = TILE_M, TN = TILE_N;
    const int nbi = (m + TM - 1) / TM, nbj = (n + TN - 1) / TN;
    const int BK  = tile_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TM, j0 = bj * TN;
            const int imax = std::min(i0 + TM, m), jmax = std::min(j0 + TN, n);
            const int tm = imax - i0, tn = jmax - j0;
            for (int kb = 0; kb < k; kb += BK) {
                const int kbe = std::min(kb + BK, k);
                for (int l = kb; l < kbe; ++l)
                    for (int i = 0; i < tm; ++i) {
                        const double av = A[static_cast<std::size_t>(i0 + i) * k + l];
                        for (int j = 0; j < tn; ++j)
                            C[static_cast<std::size_t>(j0 + j) * m + (i0 + i)] +=
                                av * B[static_cast<std::size_t>(j0 + j) * k + l];
                    }
            }
        }
    }
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
    const bool sq   = (m >= 100 && n >= 100);

    if (layout == Layout::RowMajor) {
        if (tiny && extr)    cat_b_row(m, n, k, A, B, C);
        else if (tiny && bigk) cat_a_row(m, n, k, A, B, C);
        else if (sq)         cat_c_row(m, n, k, A, B, C);
        else                  cat_d_row(m, n, k, A, B, C);
    } else {
        if (tiny && extr)    cat_b_col(m, n, k, A, B, C);
        else if (tiny && bigk) cat_a_col(m, n, k, A, B, C);
        else if (sq)         cat_c_col(m, n, k, A, B, C);
        else                  cat_d_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_shape_dispatch", tsmm_my_shape_dispatch);
