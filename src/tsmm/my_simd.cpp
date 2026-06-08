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
// my_simd: AVX-512 SIMD with multi-way unrolling + tiling
//
// Uses m×n tiling (direct C write, O(m*n) memory) with:
//   1. 4 independent accumulators per inner loop
//   2. Masked load/store for tail elements
//   3. k-blocking for L2 cache reuse
// ============================================================

namespace {

constexpr int TILE_M = 64;
constexpr int TILE_N = 256;
constexpr int L2_D  = 1024 * 1024 / 8; // L2 in doubles

int bk_for(int tm, int tn, int k) {
    int denom = tm + tn;
    if (denom <= 0) denom = 1;
    int bk = (L2_D - tm * tn) / denom;
    if (bk < 8) bk = 8;
    bk = (bk / 8) * 8;
    if (bk > k) bk = k;
    return bk;
}

} // anonymous namespace


// --- RowMajor: tiling + AVX-512 ---

static void simd_tiled_row(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int TM = TILE_M, TN = TILE_N;
    const int nbi = (m + TM - 1) / TM, nbj = (n + TN - 1) / TN;
    const int BK  = bk_for(TM, TN, k);

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
#ifdef __AVX512F__
                    for (int i = 0; i < tm; ++i) {
                        const __m512d av = _mm512_set1_pd(ar[i0 + i]);
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        int j = 0;
                        for (; j + 31 < tn; j += 32) {
                            __m512d c0 = _mm512_loadu_pd(cr + j);
                            __m512d c1 = _mm512_loadu_pd(cr + j + 8);
                            __m512d c2 = _mm512_loadu_pd(cr + j + 16);
                            __m512d c3 = _mm512_loadu_pd(cr + j + 24);
                            c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j), c0);
                            c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j + 8), c1);
                            c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j + 16), c2);
                            c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j + 24), c3);
                            _mm512_storeu_pd(cr + j, c0);
                            _mm512_storeu_pd(cr + j + 8, c1);
                            _mm512_storeu_pd(cr + j + 16, c2);
                            _mm512_storeu_pd(cr + j + 24, c3);
                        }
                        for (; j + 7 < tn; j += 8) {
                            __m512d cv = _mm512_loadu_pd(cr + j);
                            cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j), cv);
                            _mm512_storeu_pd(cr + j, cv);
                        }
                        if (j < tn) {
                            const __mmask8 mk = static_cast<__mmask8>((1u << (tn - j)) - 1);
                            __m512d cv = _mm512_maskz_loadu_pd(mk, cr + j);
                            __m512d bv = _mm512_maskz_loadu_pd(mk, br + j0 + j);
                            cv = _mm512_fmadd_pd(av, bv, cv);
                            _mm512_mask_storeu_pd(cr + j, mk, cv);
                        }
                    }
#else
                    for (int i = 0; i < tm; ++i) {
                        const double av = ar[i0 + i];
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        for (int j = 0; j < tn; ++j) cr[j] += av * br[j0 + j];
                    }
#endif
                }
            }
        }
    }
}


// --- ColMajor: tiling ---

static void simd_tiled_col(int m, int n, int k,
                            const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int TM = TILE_M, TN = TILE_N;
    const int nbi = (m + TM - 1) / TM, nbj = (n + TN - 1) / TN;
    const int BK  = bk_for(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TM, j0 = bj * TN;
            const int imax = std::min(i0 + TM, m), jmax = std::min(j0 + TN, n);
            const int tm = imax - i0, tn = jmax - j0;

            for (int kb = 0; kb < k; kb += BK) {
                const int kbe = std::min(kb + BK, k);
                for (int l = kb; l < kbe; ++l) {
#ifdef __AVX512F__
                    for (int i = 0; i < tm; ++i) {
                        const __m512d av = _mm512_set1_pd(A[static_cast<std::size_t>(i0 + i) * k + l]);
                        int j = 0;
                        for (; j + 31 < tn; j += 32) {
                            // ColMajor: C[(j0+j)*m + i0+i], B[(j0+j)*k + l] — stride m between columns
                            // Use gather for B access to avoid manual set_pd
                            __m512d bv0 = _mm512_set_pd(
                                B[static_cast<std::size_t>(j0 + j + 7) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 6) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 5) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 4) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 3) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 2) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 1) * k + l],
                                B[static_cast<std::size_t>(j0 + j) * k + l]);
                            __m512d cv0 = _mm512_loadu_pd(C + static_cast<std::size_t>(j0 + j) * m + i0 + i);
                            cv0 = _mm512_fmadd_pd(av, bv0, cv0);
                            _mm512_storeu_pd(C + static_cast<std::size_t>(j0 + j) * m + i0 + i, cv0);

                            __m512d bv1 = _mm512_set_pd(
                                B[static_cast<std::size_t>(j0 + j + 15) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 14) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 13) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 12) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 11) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 10) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 9) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 8) * k + l]);
                            __m512d cv1 = _mm512_loadu_pd(C + static_cast<std::size_t>(j0 + j + 8) * m + i0 + i);
                            cv1 = _mm512_fmadd_pd(av, bv1, cv1);
                            _mm512_storeu_pd(C + static_cast<std::size_t>(j0 + j + 8) * m + i0 + i, cv1);

                            __m512d bv2 = _mm512_set_pd(
                                B[static_cast<std::size_t>(j0 + j + 23) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 22) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 21) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 20) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 19) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 18) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 17) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 16) * k + l]);
                            __m512d cv2 = _mm512_loadu_pd(C + static_cast<std::size_t>(j0 + j + 16) * m + i0 + i);
                            cv2 = _mm512_fmadd_pd(av, bv2, cv2);
                            _mm512_storeu_pd(C + static_cast<std::size_t>(j0 + j + 16) * m + i0 + i, cv2);

                            __m512d bv3 = _mm512_set_pd(
                                B[static_cast<std::size_t>(j0 + j + 31) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 30) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 29) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 28) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 27) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 26) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 25) * k + l],
                                B[static_cast<std::size_t>(j0 + j + 24) * k + l]);
                            __m512d cv3 = _mm512_loadu_pd(C + static_cast<std::size_t>(j0 + j + 24) * m + i0 + i);
                            cv3 = _mm512_fmadd_pd(av, bv3, cv3);
                            _mm512_storeu_pd(C + static_cast<std::size_t>(j0 + j + 24) * m + i0 + i, cv3);
                        }
                        for (int jj = j; jj < tn; ++jj)
                            C[static_cast<std::size_t>(j0 + jj) * m + i0 + i] +=
                                av[0] * B[static_cast<std::size_t>(j0 + jj) * k + l]; // av[0] is the scalar value
                    }
#else
                    for (int i = 0; i < tm; ++i) {
                        const double av = A[static_cast<std::size_t>(i0 + i) * k + l];
                        for (int j = 0; j < tn; ++j)
                            C[static_cast<std::size_t>(j0 + j) * m + (i0 + i)] +=
                                av * B[static_cast<std::size_t>(j0 + j) * k + l];
                    }
#endif
                }
            }
        }
    }
}

void tsmm_my_simd(int m, int n, int k,
                   const double* A, const double* B, double* C,
                   Layout layout) {
    if (layout == Layout::RowMajor) {
        simd_tiled_row(m, n, k, A, B, C);
    } else {
        simd_tiled_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_simd", tsmm_my_simd);
