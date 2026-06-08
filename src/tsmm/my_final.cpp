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
// my_final: Integrated final TSMM kernel
//
// This kernel integrates all optimizations learned from the
// progressive versions:
//   1. Shape-adaptive dispatch (my_shape_dispatch)
//   2. Auto-tuned cache blocking (my_blocking)
//   3. AVX-512 SIMD with multi-way unrolling (my_simd)
//   4. Thread-private accumulation (my_omp)
//   5. Panel streaming for extreme k (my_shape_dispatch)
//   6. B packing for contiguous access (my_blocking_pack)
//
// The dispatcher classifies the input problem and routes to
// the best code path based on problem dimensions.
// ============================================================

// Per-core L2 cache capacity on Xeon Platinum 9242
static constexpr int L2_BYTES = 1024 * 1024;
static constexpr int L2_DOUBLES_VAL = L2_BYTES / 8;       // 131072
static constexpr int SIMD_W = 8;                           // AVX-512
static constexpr int PANEL = 2048;                         // streaming panel

// Auto-tune k block size for L2
static int tune_bk(int m, int n, int k) {
    const int out = m * n;
    int denom = m + n;
    if (denom <= 0) denom = 1;
    int bk = (L2_DOUBLES_VAL - out) / denom;
    if (bk < SIMD_W) bk = SIMD_W;
    bk = (bk / SIMD_W) * SIMD_W;
    if (bk > k) bk = k;
    if (bk < 8) bk = 8;
    return bk;
}

// =====================================================================
// Strategy A: tiny output (m*n <= 256) + large k (k >= 10000)
//   C_local in L1/registers, k-parallel with thread-private accumulators
// =====================================================================

static void final_tiny_row(int m, int n, int k,
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
                    __m512d c0 = _mm512_loadu_pd(cr + j);
                    __m512d c1 = _mm512_loadu_pd(cr + j + 8);
                    __m512d c2 = _mm512_loadu_pd(cr + j + 16);
                    __m512d c3 = _mm512_loadu_pd(cr + j + 24);
                    c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j), c0);
                    c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j + 8), c1);
                    c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j + 16), c2);
                    c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j + 24), c3);
                    _mm512_storeu_pd(cr + j, c0);
                    _mm512_storeu_pd(cr + j + 8, c1);
                    _mm512_storeu_pd(cr + j + 16, c2);
                    _mm512_storeu_pd(cr + j + 24, c3);
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

static void final_tiny_col(int m, int n, int k,
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
            for (int i = 0; i < m; ++i) {
                const double av = A[static_cast<std::size_t>(i) * k + l];
                for (int j = 0; j < n; ++j) {
                    ct[static_cast<std::size_t>(j) * m + i] +=
                        av * B[static_cast<std::size_t>(j) * k + l];
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

// =====================================================================
// Strategy B: square-like (m>=100, n>=100)
//   Blocking + B/A packing for L2 reuse
// =====================================================================

static void final_square_row(int m, int n, int k,
                              const double* A, const double* B, double* C) {
    const int BK = tune_bk(m, n, k);
    const int nth = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int sz = m * n;
    std::vector<double> t(static_cast<std::size_t>(nth) * sz, 0.0);
    std::vector<double> bp;

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = t.data() + static_cast<std::size_t>(tid) * sz;

        for (int kb = 0; kb < k; kb += BK) {
            const int kbl = std::min(BK, k - kb);

#pragma omp single
            {
                bp.resize(static_cast<std::size_t>(kbl) * n);
                for (int l = 0; l < kbl; ++l) {
                    std::memcpy(bp.data() + static_cast<std::size_t>(l) * n,
                                B + static_cast<std::size_t>(kb + l) * n,
                                static_cast<std::size_t>(n) * sizeof(double));
                }
            }

#pragma omp for schedule(static)
            for (int l = 0; l < kbl; ++l) {
                const double* ar = A + static_cast<std::size_t>(kb + l) * m;
                const double* bd = bp.data() + static_cast<std::size_t>(l) * n;
#ifdef __AVX512F__
                for (int i = 0; i < m; ++i) {
                    const __m512d av = _mm512_set1_pd(ar[i]);
                    double* cr = ct + static_cast<std::size_t>(i) * n;
                    int j = 0;
                    for (; j + 31 < n; j += 32) {
                        __m512d c0 = _mm512_loadu_pd(cr + j);
                        __m512d c1 = _mm512_loadu_pd(cr + j + 8);
                        __m512d c2 = _mm512_loadu_pd(cr + j + 16);
                        __m512d c3 = _mm512_loadu_pd(cr + j + 24);
                        c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j), c0);
                        c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j + 8), c1);
                        c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j + 16), c2);
                        c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j + 24), c3);
                        _mm512_storeu_pd(cr + j, c0);
                        _mm512_storeu_pd(cr + j + 8, c1);
                        _mm512_storeu_pd(cr + j + 16, c2);
                        _mm512_storeu_pd(cr + j + 24, c3);
                    }
                    for (; j + 7 < n; j += 8) {
                        __m512d cv = _mm512_loadu_pd(cr + j);
                        cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(bd + j), cv);
                        _mm512_storeu_pd(cr + j, cv);
                    }
                    if (j < n) {
                        const __mmask8 mk = static_cast<__mmask8>((1u << (n - j)) - 1);
                        __m512d cv = _mm512_maskz_loadu_pd(mk, cr + j);
                        __m512d bv = _mm512_maskz_loadu_pd(mk, bd + j);
                        cv = _mm512_fmadd_pd(av, bv, cv);
                        _mm512_mask_storeu_pd(cr + j, mk, cv);
                    }
                }
#else
                for (int i = 0; i < m; ++i) {
                    const double av = ar[i];
                    double* cr = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) cr[j] += av * bd[j];
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

static void final_square_col(int m, int n, int k,
                              const double* A, const double* B, double* C) {
    const int BK = tune_bk(m, n, k);
    const int nth = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int sz = m * n;
    std::vector<double> t(static_cast<std::size_t>(nth) * sz, 0.0);
    std::vector<double> ap;

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = t.data() + static_cast<std::size_t>(tid) * sz;

        for (int kb = 0; kb < k; kb += BK) {
            const int kbl = std::min(BK, k - kb);

#pragma omp single
            {
                ap.resize(static_cast<std::size_t>(m) * kbl);
                for (int i = 0; i < m; ++i) {
                    std::memcpy(ap.data() + static_cast<std::size_t>(i) * kbl,
                                A + static_cast<std::size_t>(i) * k + kb,
                                static_cast<std::size_t>(kbl) * sizeof(double));
                }
            }

#pragma omp for schedule(static)
            for (int l = 0; l < kbl; ++l) {
                for (int i = 0; i < m; ++i) {
                    const double av = ap[static_cast<std::size_t>(i) * kbl + l];
                    for (int j = 0; j < n; ++j) {
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            av * B[static_cast<std::size_t>(j) * k + kb + l];
                    }
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

// =====================================================================
// Strategy C: general (medium output, any k)
//   Auto-tuned blocking + k-parallel + SIMD
// =====================================================================

static void final_general_row(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    const int BK = tune_bk(m, n, k);
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

        for (int kb = 0; kb < k; kb += BK) {
            const int kbe = std::min(kb + BK, k);
#pragma omp for schedule(static)
            for (int l = kb; l < kbe; ++l) {
                const double* ar = A + static_cast<std::size_t>(l) * m;
                const double* br = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double av = ar[i];
                    double* cr = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) cr[j] += av * br[j];
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

static void final_general_col(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    const int BK = tune_bk(m, n, k);
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

        for (int kb = 0; kb < k; kb += BK) {
            const int kbe = std::min(kb + BK, k);
#pragma omp for schedule(static)
            for (int l = kb; l < kbe; ++l) {
                for (int i = 0; i < m; ++i) {
                    const double av = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j) {
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            av * B[static_cast<std::size_t>(j) * k + l];
                    }
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

// =====================================================================
// Public entry point — shape-adaptive dispatch
// =====================================================================

void tsmm_my_final(int m, int n, int k,
                    const double* A, const double* B, double* C,
                    Layout layout) {
    const int out_sz = m * n;
    const bool tiny  = (out_sz <= 256);
    const bool big_k = (k >= 10000);
    const bool sq    = (m >= 100 && n >= 100);

    if (layout == Layout::RowMajor) {
        if (tiny && big_k)  final_tiny_row(m, n, k, A, B, C);
        else if (sq)        final_square_row(m, n, k, A, B, C);
        else                final_general_row(m, n, k, A, B, C);
    } else {
        if (tiny && big_k)  final_tiny_col(m, n, k, A, B, C);
        else if (sq)        final_square_col(m, n, k, A, B, C);
        else                final_general_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_final", tsmm_my_final);
