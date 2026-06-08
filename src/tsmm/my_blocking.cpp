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
// my_blocking: Register-blocked tiling with AVX-512
//
// Key design constraint: accumulators MUST fit in the 32-entry
// ZMM register file (2 KB). We process the n-dimension in
// 16-wide strips (like opt's b0/b1 pattern), so at most
// ilen × 2 accumulators are live → max 16×2 = 32 regs.
//
// Tile sizes: IB=8..16 (m), strips of 16 along n
// ============================================================

namespace {

int choose_ib(int m) {
    if (m <= 8)  return m;
    if (m <= 16) return 8;
    return 8;  // always 8 for max parallelism (500K tiles for large m)
}

int choose_jb(int n) {
    if (n <= 16) return n;
    if (n <= 64) return 16;
    return 32; // 2 strips of 16 → max 32 tiles worth of parallelism
}

} // anonymous namespace


// =====================================================================
// RowMajor: opt-compatible 8×16 register-blocked kernel
//
// Uses the exact same structure as opt's row_tile_i8_j16 but with
// masked tail stores instead of scalar fallback.
// 8×16 tile × 2 accumulators/row × 8 rows = 16 ZMM regs → fits.
// =====================================================================

void blocking_tiled_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    constexpr int IB = 8;
    constexpr int JB = 16;
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB;
            const int j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);

#ifdef __AVX512F__
            if (jlen == 8 || jlen >= 16) {
                // Optimized path for full 8-wide or 16-wide tiles.
                // Two accumulators per i-row — identical to opt's acc0/acc1.
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

                // Write-back
                for (int ii = 0; ii < ilen; ++ii) {
                    double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    if (jlen == 8) {
                        _mm512_storeu_pd(c_row, acc0[ii]);
                    } else {
                        _mm512_storeu_pd(c_row, acc0[ii]);
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
            // Scalar fallback for irregular jlen (1..7, 9..15):
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


// =====================================================================
// ColMajor: same strip-based approach
// C[j*m+i], A[i*k+l], B[j*k+l]
// For each strip of j, accumulate into local registers, write back
// =====================================================================

void blocking_tiled_col(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int IB = choose_ib(m);
    const int JB = choose_jb(n);
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB;
            const int j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);

            // ColMajor: C[j][i] = C[j*m+i]
            // Per (i,j): C[j*m+i] = sum_l A[i*k+l] * B[j*k+l]
            // For each j in the tile, accumulate along k and write back

            for (int jj = 0; jj < jlen; ++jj) {
                const int j_idx = j0 + jj;
                double* c_col = C + static_cast<std::size_t>(j_idx) * m + i0;
                const double* b_col = B + static_cast<std::size_t>(j_idx) * k;

                for (int ii = 0; ii < ilen; ++ii) {
                    double sum = 0.0;
                    const double* a_col = A + static_cast<std::size_t>(i0 + ii) * k;
#ifdef __AVX512F__
                    int l = 0;
                    __m512d acc0 = _mm512_setzero_pd();
                    __m512d acc1 = _mm512_setzero_pd();
                    for (; l + 15 < k; l += 16) {
                        acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(a_col + l),
                                                _mm512_loadu_pd(b_col + l), acc0);
                        acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(a_col + l + 8),
                                                _mm512_loadu_pd(b_col + l + 8), acc1);
                    }
                    sum += _mm512_reduce_add_pd(acc0) + _mm512_reduce_add_pd(acc1);
                    for (; l + 7 < k; l += 8) {
                        __m512d av = _mm512_loadu_pd(a_col + l);
                        __m512d bv = _mm512_loadu_pd(b_col + l);
                        sum += _mm512_reduce_add_pd(_mm512_mul_pd(av, bv));
                    }
                    for (; l < k; ++l) sum += a_col[l] * b_col[l];
#else
                    for (int l = 0; l < k; ++l) sum += a_col[l] * b_col[l];
#endif
                    c_col[ii] = sum;
                }
            }
        }
    }
}

void tsmm_my_blocking(int m, int n, int k,
                       const double* A, const double* B, double* C,
                       Layout layout) {
    if (layout == Layout::RowMajor)
        blocking_tiled_row(m, n, k, A, B, C);
    else
        blocking_tiled_col(m, n, k, A, B, C);
}

REGISTER_TSMM_IMPL("my_blocking", tsmm_my_blocking);


// =====================================================================
// my_blocking_pack: B-packing for very large k (amortizes packing cost)
// =====================================================================

static constexpr int L2_D = 1024 * 1024 / 8;

void blocking_tiled_row_packed(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    const int IB = choose_ib(m);
    int BK = (L2_D - IB * 16) / (IB + 16);
    if (BK < 8) BK = 8;
    BK = (BK / 8) * 8;
    if (BK > k) BK = k;

    // Only pack when k is large enough to benefit
    if (k <= 16 * BK) { blocking_tiled_row(m, n, k, A, B, C); return; }

    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + IB - 1) / IB;

#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        const int i0 = bi * IB;
        const int ilen = std::min(IB, m - i0);

#ifdef __AVX512F__
        for (int js = 0; js < n; js += 16) {
            const int js_len = std::min(16, n - js);
            const int nacc = (js_len + 7) / 8;

            __m512d acc[32] = {};
            for (int kb = 0; kb < k; kb += BK) {
                const int kbl = std::min(BK, k - kb);
                // Pack B for this strip
                std::vector<double> bp(static_cast<std::size_t>(kbl) * js_len);
                for (int l = 0; l < kbl; ++l)
                    std::memcpy(bp.data() + static_cast<std::size_t>(l) * js_len,
                                B + static_cast<std::size_t>(kb + l) * n + js,
                                static_cast<std::size_t>(js_len) * sizeof(double));

                for (int l = 0; l < kbl; ++l) {
                    const double* a_row = A + static_cast<std::size_t>(kb + l) * m + i0;
                    const double* b_p   = bp.data() + static_cast<std::size_t>(l) * js_len;
                    for (int a = 0; a < nacc; ++a) {
                        const int jj = a * 8;
                        const int rem = std::min(8, js_len - jj);
                        const __mmask8 mk = static_cast<__mmask8>((1u << rem) - 1);
                        const __m512d bv = _mm512_maskz_loadu_pd(mk, b_p + jj);
                        for (int ii = 0; ii < ilen; ++ii) {
                            acc[ii * nacc + a] = _mm512_fmadd_pd(
                                _mm512_set1_pd(a_row[ii]), bv, acc[ii * nacc + a]);
                        }
                    }
                }
            }

            for (int ii = 0; ii < ilen; ++ii) {
                double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + js;
                for (int a = 0; a < nacc; ++a) {
                    const int jj = a * 8;
                    const int rem = std::min(8, js_len - jj);
                    _mm512_mask_storeu_pd(c_row + jj, static_cast<__mmask8>((1u << rem) - 1),
                                          acc[ii * nacc + a]);
                }
            }
        }
#else
        for (int l = 0; l < k; ++l) {
            const double* a_row = A + static_cast<std::size_t>(l) * m + i0;
            const double* b_row = B + static_cast<std::size_t>(l) * n;
            for (int ii = 0; ii < ilen; ++ii) {
                const double av = a_row[ii];
                double* c_row = C + static_cast<std::size_t>(i0 + ii) * n;
                for (int j = 0; j < n; ++j) c_row[j] += av * b_row[j];
            }
        }
#endif
    }
}

void tsmm_my_blocking_pack(int m, int n, int k,
                            const double* A, const double* B, double* C,
                            Layout layout) {
    if (layout == Layout::RowMajor)
        blocking_tiled_row_packed(m, n, k, A, B, C);
    else
        blocking_tiled_col(m, n, k, A, B, C);
}

REGISTER_TSMM_IMPL("my_blocking_pack", tsmm_my_blocking_pack);
