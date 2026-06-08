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
// my_blocking: Competitive tiling kernel with AVX-512
//
// Strategy (adapted from opt's proven 8×16 approach):
//   - Small tiles over m×n space, parallelized via collapse(2)
//   - For each tile: iterate all k, accumulate in registers
//   - AVX-512 with broadcast-FMA pattern
//   - Direct C write (tiles are exclusive → no race)
//
// Improvements over opt:
//   - Adaptive tile sizing (IB=8..16, JB=16..32 based on m,n)
//   - k-blocking + packing for very large k (amortizes well)
//   - Masked tail stores (no scalar fallback for small remainders)
//
// my_blocking_pack: same tiling + B-packing for large-k shapes
// ============================================================

namespace {

constexpr int L2_SIZE = 1024 * 1024;

int choose_ib(int m) {
    if (m <= 8)  return m;
    if (m <= 16) return 8;
    if (m <= 64) return 16;
    return 8;  // lots of tiles for parallelism
}

int choose_jb(int n) {
    if (n <= 16) return n;
    if (n <= 64) return 16;
    return 32;  // 4 AVX-512 registers per row
}

int compute_bk(int ib, int jb, int k) {
    const int l2d = L2_SIZE / static_cast<int>(sizeof(double));
    int denom = ib + jb;
    if (denom <= 0) denom = 1;
    int bk = (l2d - ib * jb) / denom;
    if (bk < 8) bk = 8;
    bk = (bk / 8) * 8;
    if (bk > k) bk = k;
    return bk;
}

} // anonymous namespace


// =====================================================================
// RowMajor tiling — opt-compatible 8×16 register-blocked kernel
// =====================================================================

void blocking_tiled_row(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int IB = choose_ib(m);
    const int JB = choose_jb(n);
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;
    // Only k-block for very large k where the benefit outweighs overhead
    const int BK = compute_bk(IB, JB, k);
    const bool do_kblock = (k > 16 * BK);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB;
            const int j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);

#ifdef __AVX512F__
            // AVX-512 register-blocked kernel for tiles with jlen >= 8
            if (jlen >= 8) {
                // Allocate accumulators: jlen/8 per i-row, up to 32 ZMM regs total
                const int nacc = (jlen + 7) / 8; // 1..4 accumulators per row
                __m512d acc[64]; // max 16 rows × 4 accumulators = 64, fits in ZMM file
                for (int ii = 0; ii < ilen; ++ii)
                    for (int a = 0; a < nacc; ++a)
                        acc[ii * nacc + a] = _mm512_setzero_pd();

                if (do_kblock) {
                    for (int kb = 0; kb < k; kb += BK) {
                        const int kbe = std::min(kb + BK, k);
                        for (int l = kb; l < kbe; ++l) {
                            const double* a_row = A + static_cast<std::size_t>(l) * m + i0;
                            const double* b_row = B + static_cast<std::size_t>(l) * n + j0;
                            for (int a = 0; a < nacc; ++a) {
                                const int jj = a * 8;
                                const int rem = std::min(8, jlen - jj);
                                const __mmask8 mk = static_cast<__mmask8>((1u << rem) - 1);
                                __m512d bv = _mm512_maskz_loadu_pd(mk, b_row + jj);
                                for (int ii = 0; ii < ilen; ++ii) {
                                    const __m512d av = _mm512_set1_pd(a_row[ii]);
                                    acc[ii * nacc + a] = _mm512_fmadd_pd(av, bv, acc[ii * nacc + a]);
                                }
                            }
                        }
                    }
                } else {
                    for (int l = 0; l < k; ++l) {
                        const double* a_row = A + static_cast<std::size_t>(l) * m + i0;
                        const double* b_row = B + static_cast<std::size_t>(l) * n + j0;
                        for (int a = 0; a < nacc; ++a) {
                            const int jj = a * 8;
                            const int rem = std::min(8, jlen - jj);
                            const __mmask8 mk = static_cast<__mmask8>((1u << rem) - 1);
                            __m512d bv = _mm512_maskz_loadu_pd(mk, b_row + jj);
                            for (int ii = 0; ii < ilen; ++ii) {
                                const __m512d av = _mm512_set1_pd(a_row[ii]);
                                acc[ii * nacc + a] = _mm512_fmadd_pd(av, bv, acc[ii * nacc + a]);
                            }
                        }
                    }
                }

                // Write-back
                for (int ii = 0; ii < ilen; ++ii) {
                    double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    for (int a = 0; a < nacc; ++a) {
                        const int jj = a * 8;
                        const int rem = std::min(8, jlen - jj);
                        const __mmask8 mk = static_cast<__mmask8>((1u << rem) - 1);
                        _mm512_mask_storeu_pd(c_row + jj, mk, acc[ii * nacc + a]);
                    }
                }
                continue;
            }
#endif
            // Scalar fallback for tiny tile remainders
            for (int l = 0; l < k; ++l) {
                const double* a_row = A + static_cast<std::size_t>(l) * m + i0;
                const double* b_row = B + static_cast<std::size_t>(l) * n + j0;
                for (int ii = 0; ii < ilen; ++ii) {
                    const double av = a_row[ii];
                    double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    for (int jj = 0; jj < jlen; ++jj) c_row[jj] += av * b_row[jj];
                }
            }
        }
    }
}


// =====================================================================
// ColMajor tiling
// =====================================================================

void blocking_tiled_col(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int IB = choose_ib(m);
    const int JB = choose_jb(n);
    const int nbi = (m + IB - 1) / IB;
    const int nbj = (n + JB - 1) / JB;
    const int BK = compute_bk(IB, JB, k);
    const bool do_kblock = (k > 16 * BK);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * IB;
            const int j0 = bj * JB;
            const int ilen = std::min(IB, m - i0);
            const int jlen = std::min(JB, n - j0);

            // ColMajor: C[j*m+i], A[i*k+l], B[j*k+l]
            // For each (i,j) pair, compute C[j*m+i] = sum_l A[i*k+l] * B[j*k+l]
            // Accumulate into a small local buffer then write back
            double local[16][32] = {}; // max tile size
            if (do_kblock) {
                for (int kb = 0; kb < k; kb += BK) {
                    const int kbe = std::min(kb + BK, k);
                    for (int l = kb; l < kbe; ++l)
                        for (int ii = 0; ii < ilen; ++ii) {
                            const double av = A[static_cast<std::size_t>(i0 + ii) * k + l];
                            for (int jj = 0; jj < jlen; ++jj)
                                local[ii][jj] += av * B[static_cast<std::size_t>(j0 + jj) * k + l];
                        }
                }
            } else {
                for (int l = 0; l < k; ++l)
                    for (int ii = 0; ii < ilen; ++ii) {
                        const double av = A[static_cast<std::size_t>(i0 + ii) * k + l];
                        for (int jj = 0; jj < jlen; ++jj)
                            local[ii][jj] += av * B[static_cast<std::size_t>(j0 + jj) * k + l];
                    }
            }
            for (int ii = 0; ii < ilen; ++ii)
                for (int jj = 0; jj < jlen; ++jj)
                    C[static_cast<std::size_t>(j0 + jj) * m + (i0 + ii)] = local[ii][jj];
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
// my_blocking_pack: tiling + B-packing for very large k
// =====================================================================

static void blocking_pack_row(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    const int IB = choose_ib(m);
    const int JB = choose_jb(n);
    const int BK = compute_bk(IB, JB, k);
    // Packing only pays off when we can reuse the packed buffer many times
    if (k <= 16 * BK) { blocking_tiled_row(m, n, k, A, B, C); return; }

    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
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
            const int nacc = (jlen + 7) / 8;
            __m512d acc[64];
            for (int ii = 0; ii < ilen; ++ii)
                for (int a = 0; a < nacc; ++a)
                    acc[ii * nacc + a] = _mm512_setzero_pd();

            for (int kb = 0; kb < k; kb += BK) {
                const int kbl = std::min(BK, k - kb);
                // Pack B
                std::vector<double> bp(static_cast<std::size_t>(kbl) * jlen);
                for (int l = 0; l < kbl; ++l)
                    std::memcpy(bp.data() + static_cast<std::size_t>(l) * jlen,
                                B + static_cast<std::size_t>(kb + l) * n + j0,
                                static_cast<std::size_t>(jlen) * sizeof(double));

                for (int l = 0; l < kbl; ++l) {
                    const double* a_row = A + static_cast<std::size_t>(kb + l) * m + i0;
                    const double* b_p = bp.data() + static_cast<std::size_t>(l) * jlen;
                    for (int a = 0; a < nacc; ++a) {
                        const int jj = a * 8;
                        const int rem = std::min(8, jlen - jj);
                        const __mmask8 mk = static_cast<__mmask8>((1u << rem) - 1);
                        __m512d bv = _mm512_maskz_loadu_pd(mk, b_p + jj);
                        for (int ii = 0; ii < ilen; ++ii) {
                            const __m512d av = _mm512_set1_pd(a_row[ii]);
                            acc[ii * nacc + a] = _mm512_fmadd_pd(av, bv, acc[ii * nacc + a]);
                        }
                    }
                }
            }
            for (int ii = 0; ii < ilen; ++ii) {
                double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                for (int a = 0; a < nacc; ++a) {
                    const int jj = a * 8;
                    const int rem = std::min(8, jlen - jj);
                    const __mmask8 mk = static_cast<__mmask8>((1u << rem) - 1);
                    _mm512_mask_storeu_pd(c_row + jj, mk, acc[ii * nacc + a]);
                }
            }
#else
            for (int l = 0; l < k; ++l) {
                const double* a_row = A + static_cast<std::size_t>(l) * m + i0;
                const double* b_row = B + static_cast<std::size_t>(l) * n + j0;
                for (int ii = 0; ii < ilen; ++ii) {
                    const double av = a_row[ii];
                    double* c_row = C + static_cast<std::size_t>(i0 + ii) * n + j0;
                    for (int jj = 0; jj < jlen; ++jj) c_row[jj] += av * b_row[jj];
                }
            }
#endif
        }
    }
}

static void blocking_pack_col(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    // Packing for col-major: delegate to non-pack for simplicity
    blocking_tiled_col(m, n, k, A, B, C);
}

void tsmm_my_blocking_pack(int m, int n, int k,
                            const double* A, const double* B, double* C,
                            Layout layout) {
    if (layout == Layout::RowMajor)
        blocking_pack_row(m, n, k, A, B, C);
    else
        blocking_pack_col(m, n, k, A, B, C);
}

REGISTER_TSMM_IMPL("my_blocking_pack", tsmm_my_blocking_pack);
