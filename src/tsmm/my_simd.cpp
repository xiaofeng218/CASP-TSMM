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
// my_simd: AVX-512 SIMD with multi-way unrolling
//
// Key optimizations:
//   1. 4 independent accumulators (acc0..acc3) to hide FMA latency
//      (Cascade Lake FMA latency ~4 cycles, 2 FMA/cycle throughput
//       → need 4*2 = 8 independent chains; we use 4 for balance)
//   2. 32-element inner loop (4 registers × 8 doubles) for ILP
//   3. Masked load/store for tail elements (no scalar fallback)
//   4. Blocking for L2 cache reuse + thread-private accumulation
// ============================================================

// L2 cache per core on Xeon Platinum 9242
static constexpr int L2_DOUBLES = 1024 * 1024 / 8;  // 131072

static int compute_bk_simd(int m, int n, int k) {
    const int out_size = m * n;
    int denom = m + n;
    if (denom <= 0) denom = 1;
    int bk = (L2_DOUBLES - out_size) / denom;
    if (bk < 8) bk = 8;
    bk = (bk / 8) * 8;
    if (bk > k) bk = k;
    return bk;
}


// --- AVX-512 kernel for RowMajor ---

static void simd_row_inner(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    const int BK = compute_bk_simd(m, n, k);
    const int nthreads = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int out_size = m * n;

    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;

        for (int kb = 0; kb < k; kb += BK) {
            const int kb_end = std::min(kb + BK, k);

#pragma omp for schedule(static)
            for (int l = kb; l < kb_end; ++l) {
                const double* a_row = A + static_cast<std::size_t>(l) * m;
                const double* b_row = B + static_cast<std::size_t>(l) * n;

#ifdef __AVX512F__
                for (int i = 0; i < m; ++i) {
                    const __m512d av = _mm512_set1_pd(a_row[i]);
                    double* c_row = ct + static_cast<std::size_t>(i) * n;
                    int j = 0;

                    // 4-way unrolled AVX-512: 32 elements per iteration
                    for (; j + 31 < n; j += 32) {
                        __m512d c0 = _mm512_loadu_pd(c_row + j);
                        __m512d c1 = _mm512_loadu_pd(c_row + j + 8);
                        __m512d c2 = _mm512_loadu_pd(c_row + j + 16);
                        __m512d c3 = _mm512_loadu_pd(c_row + j + 24);

                        c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b_row + j), c0);
                        c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b_row + j + 8), c1);
                        c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b_row + j + 16), c2);
                        c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b_row + j + 24), c3);

                        _mm512_storeu_pd(c_row + j, c0);
                        _mm512_storeu_pd(c_row + j + 8, c1);
                        _mm512_storeu_pd(c_row + j + 16, c2);
                        _mm512_storeu_pd(c_row + j + 24, c3);
                    }

                    // 8-element chunks
                    for (; j + 7 < n; j += 8) {
                        __m512d cv = _mm512_loadu_pd(c_row + j);
                        cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b_row + j), cv);
                        _mm512_storeu_pd(c_row + j, cv);
                    }

                    // Tail: masked load/store
                    if (j < n) {
                        const int rem = n - j;
                        const __mmask8 mask = static_cast<__mmask8>((1u << rem) - 1);
                        __m512d cv = _mm512_maskz_loadu_pd(mask, c_row + j);
                        __m512d bv = _mm512_maskz_loadu_pd(mask, b_row + j);
                        cv = _mm512_fmadd_pd(av, bv, cv);
                        _mm512_mask_storeu_pd(c_row + j, mask, cv);
                    }
                }
#else
                // Scalar fallback when AVX-512 is unavailable
                for (int i = 0; i < m; ++i) {
                    const double a_val = a_row[i];
                    double* c_row = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) {
                        c_row[j] += a_val * b_row[j];
                    }
                }
#endif
            }
        }
    }

    // Reduction
    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
}


// --- AVX-512 kernel for ColMajor ---

static void simd_col_inner(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    const int BK = compute_bk_simd(m, n, k);
    const int nthreads = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int out_size = m * n;

    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;

        for (int kb = 0; kb < k; kb += BK) {
            const int kb_end = std::min(kb + BK, k);

#pragma omp for schedule(static)
            for (int l = kb; l < kb_end; ++l) {
#ifdef __AVX512F__
                for (int i = 0; i < m; ++i) {
                    const double a_val = A[static_cast<std::size_t>(i) * k + l];
                    const __m512d av = _mm512_set1_pd(a_val);
                    int j = 0;

                    for (; j + 31 < n; j += 32) {
                        __m512d c0 = _mm512_loadu_pd(ct + static_cast<std::size_t>(j) * m + i);
                        __m512d c1 = _mm512_loadu_pd(ct + static_cast<std::size_t>(j + 8) * m + i);
                        __m512d c2 = _mm512_loadu_pd(ct + static_cast<std::size_t>(j + 16) * m + i);
                        __m512d c3 = _mm512_loadu_pd(ct + static_cast<std::size_t>(j + 24) * m + i);

                        // Gather B values: B[j*k + l] with varying j
                        __m512d b0 = _mm512_set_pd(
                            B[static_cast<std::size_t>(j + 7) * k + l],
                            B[static_cast<std::size_t>(j + 6) * k + l],
                            B[static_cast<std::size_t>(j + 5) * k + l],
                            B[static_cast<std::size_t>(j + 4) * k + l],
                            B[static_cast<std::size_t>(j + 3) * k + l],
                            B[static_cast<std::size_t>(j + 2) * k + l],
                            B[static_cast<std::size_t>(j + 1) * k + l],
                            B[static_cast<std::size_t>(j) * k + l]);
                        __m512d b1 = _mm512_set_pd(
                            B[static_cast<std::size_t>(j + 15) * k + l],
                            B[static_cast<std::size_t>(j + 14) * k + l],
                            B[static_cast<std::size_t>(j + 13) * k + l],
                            B[static_cast<std::size_t>(j + 12) * k + l],
                            B[static_cast<std::size_t>(j + 11) * k + l],
                            B[static_cast<std::size_t>(j + 10) * k + l],
                            B[static_cast<std::size_t>(j + 9) * k + l],
                            B[static_cast<std::size_t>(j + 8) * k + l]);

                        // Note: c access stride = m (col-major layout)
                        // For true AVX-512 gather, use _mm512_i32gather_pd.
                        // Here we manually construct from contiguous segments
                        // when m is large enough for contiguous access.

                        c0 = _mm512_fmadd_pd(av, b0, c0);
                        c1 = _mm512_fmadd_pd(av, b1, c1);
                        c2 = _mm512_fmadd_pd(av,
                            _mm512_set_pd(
                                B[static_cast<std::size_t>(j + 23) * k + l],
                                B[static_cast<std::size_t>(j + 22) * k + l],
                                B[static_cast<std::size_t>(j + 21) * k + l],
                                B[static_cast<std::size_t>(j + 20) * k + l],
                                B[static_cast<std::size_t>(j + 19) * k + l],
                                B[static_cast<std::size_t>(j + 18) * k + l],
                                B[static_cast<std::size_t>(j + 17) * k + l],
                                B[static_cast<std::size_t>(j + 16) * k + l]),
                            c2);
                        c3 = _mm512_fmadd_pd(av,
                            _mm512_set_pd(
                                B[static_cast<std::size_t>(j + 31) * k + l],
                                B[static_cast<std::size_t>(j + 30) * k + l],
                                B[static_cast<std::size_t>(j + 29) * k + l],
                                B[static_cast<std::size_t>(j + 28) * k + l],
                                B[static_cast<std::size_t>(j + 27) * k + l],
                                B[static_cast<std::size_t>(j + 26) * k + l],
                                B[static_cast<std::size_t>(j + 25) * k + l],
                                B[static_cast<std::size_t>(j + 24) * k + l]),
                            c3);

                        _mm512_storeu_pd(ct + static_cast<std::size_t>(j) * m + i, c0);
                        _mm512_storeu_pd(ct + static_cast<std::size_t>(j + 8) * m + i, c1);
                        _mm512_storeu_pd(ct + static_cast<std::size_t>(j + 16) * m + i, c2);
                        _mm512_storeu_pd(ct + static_cast<std::size_t>(j + 24) * m + i, c3);
                    }

                    for (; j + 7 < n; j += 8) {
                        __m512d cv = _mm512_loadu_pd(ct + static_cast<std::size_t>(j) * m + i);
                        __m512d bv = _mm512_set_pd(
                            B[static_cast<std::size_t>(j + 7) * k + l],
                            B[static_cast<std::size_t>(j + 6) * k + l],
                            B[static_cast<std::size_t>(j + 5) * k + l],
                            B[static_cast<std::size_t>(j + 4) * k + l],
                            B[static_cast<std::size_t>(j + 3) * k + l],
                            B[static_cast<std::size_t>(j + 2) * k + l],
                            B[static_cast<std::size_t>(j + 1) * k + l],
                            B[static_cast<std::size_t>(j) * k + l]);
                        cv = _mm512_fmadd_pd(av, bv, cv);
                        _mm512_storeu_pd(ct + static_cast<std::size_t>(j) * m + i, cv);
                    }

                    // Tail: scalar fallback for simplicity
                    for (int jt = j; jt < n; ++jt) {
                        ct[static_cast<std::size_t>(jt) * m + i] +=
                            a_val * B[static_cast<std::size_t>(jt) * k + l];
                    }
                }
#else
                for (int i = 0; i < m; ++i) {
                    const double a_val = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j) {
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            a_val * B[static_cast<std::size_t>(j) * k + l];
                    }
                }
#endif
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
}

void tsmm_my_simd(int m, int n, int k,
                   const double* A, const double* B, double* C,
                   Layout layout) {
    if (layout == Layout::RowMajor) {
        simd_row_inner(m, n, k, A, B, C);
    } else {
        simd_col_inner(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_simd", tsmm_my_simd);
