#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// my_blocking: Cache Blocking with auto-tuned tile sizes
//
// Strategy:
//   - Tiling over m×n space (each tile processed independently)
//   - Tiles parallelized via OpenMP collapse(2)
//   - Each tile iterates over all k (k-blocking inside for L2 reuse)
//   - Direct write to C — no thread-private copies, O(m*n) memory
//   - BK auto-tuned: BK*(m+n) + tile_size < L2_SIZE
//
// my_blocking_pack adds B-packing within each k-block for
// contiguous access (beneficial for larger n-dimension tiles).
// ============================================================

namespace {

constexpr int L2_SIZE_BYTES = 1024 * 1024;   // 1 MB per core
constexpr int SIMD_WIDTH   = 8;

// Tile sizes for m×n tiling — chosen to balance parallelism
// and cache usage. Tunable.
constexpr int TILE_M = 64;   // rows per tile
constexpr int TILE_N = 256;  // cols per tile

int compute_bk(int tile_m, int tile_n, int k) {
    const int tile_sz = tile_m * tile_n;
    const int l2_doubles = L2_SIZE_BYTES / static_cast<int>(sizeof(double));
    int denom = tile_m + tile_n;
    if (denom <= 0) denom = 1;
    int bk = (l2_doubles - tile_sz) / denom;
    if (bk < SIMD_WIDTH) bk = SIMD_WIDTH;
    bk = (bk / SIMD_WIDTH) * SIMD_WIDTH;
    if (bk > k) bk = k;
    if (bk < 8) bk = 8;
    return bk;
}

} // anonymous namespace


// =====================================================================
// my_blocking: tiling over m×n, direct C write, k-blocked inner loop
// =====================================================================

static void blocking_tiled_row(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int TM = TILE_M;
    const int TN = TILE_N;
    const int nbi = (m + TM - 1) / TM;
    const int nbj = (n + TN - 1) / TN;
    const int BK  = compute_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0  = bi * TM;
            const int j0  = bj * TN;
            const int imax = std::min(i0 + TM, m);
            const int jmax = std::min(j0 + TN, n);
            const int t_m  = imax - i0;
            const int t_n  = jmax - j0;

            for (int kb = 0; kb < k; kb += BK) {
                const int kb_end = std::min(kb + BK, k);
                for (int l = kb; l < kb_end; ++l) {
                    const double* a_row = A + static_cast<std::size_t>(l) * m;
                    const double* b_row = B + static_cast<std::size_t>(l) * n;
                    for (int i = 0; i < t_m; ++i) {
                        const double av = a_row[i0 + i];
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        for (int j = 0; j < t_n; ++j) {
                            cr[j] += av * b_row[j0 + j];
                        }
                    }
                }
            }
        }
    }
}

static void blocking_tiled_col(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    // ColMajor: C[j][i] = C[j*m + i], A[i][l] = A[i*k + l], B[j][l] = B[j*k + l]
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int TM = TILE_M;
    const int TN = TILE_N;
    const int nbi = (m + TM - 1) / TM;
    const int nbj = (n + TN - 1) / TN;
    const int BK  = compute_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0  = bi * TM;
            const int j0  = bj * TN;
            const int imax = std::min(i0 + TM, m);
            const int jmax = std::min(j0 + TN, n);
            const int t_m  = imax - i0;
            const int t_n  = jmax - j0;

            for (int kb = 0; kb < k; kb += BK) {
                const int kb_end = std::min(kb + BK, k);
                for (int l = kb; l < kb_end; ++l) {
                    for (int i = 0; i < t_m; ++i) {
                        const double av = A[static_cast<std::size_t>(i0 + i) * k + l];
                        for (int j = 0; j < t_n; ++j) {
                            C[static_cast<std::size_t>(j0 + j) * m + (i0 + i)] +=
                                av * B[static_cast<std::size_t>(j0 + j) * k + l];
                        }
                    }
                }
            }
        }
    }
}

void tsmm_my_blocking(int m, int n, int k,
                       const double* A, const double* B, double* C,
                       Layout layout) {
    if (layout == Layout::RowMajor) {
        blocking_tiled_row(m, n, k, A, B, C);
    } else {
        blocking_tiled_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_blocking", tsmm_my_blocking);


// =====================================================================
// my_blocking_pack: tiling + B-packing for contiguous inner-loop access
// =====================================================================

static void blocking_pack_row(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int TM = TILE_M;
    const int TN = TILE_N;
    const int nbi = (m + TM - 1) / TM;
    const int nbj = (n + TN - 1) / TN;
    const int BK  = compute_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0  = bi * TM;
            const int j0  = bj * TN;
            const int imax = std::min(i0 + TM, m);
            const int jmax = std::min(j0 + TN, n);
            const int t_m  = imax - i0;
            const int t_n  = jmax - j0;

            for (int kb = 0; kb < k; kb += BK) {
                const int kb_len = std::min(BK, k - kb);

                // Pack B[kb:kb+kb_len][j0:jmax] -> packed[0..kb_len-1][0..t_n-1]
                std::vector<double> bp(static_cast<std::size_t>(kb_len) * t_n);
                for (int l = 0; l < kb_len; ++l) {
                    std::memcpy(bp.data() + static_cast<std::size_t>(l) * t_n,
                                B + static_cast<std::size_t>(kb + l) * n + j0,
                                static_cast<std::size_t>(t_n) * sizeof(double));
                }

                for (int l = 0; l < kb_len; ++l) {
                    const double* a_row = A + static_cast<std::size_t>(kb + l) * m;
                    const double* b_p   = bp.data() + static_cast<std::size_t>(l) * t_n;
                    for (int i = 0; i < t_m; ++i) {
                        const double av = a_row[i0 + i];
                        double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                        for (int j = 0; j < t_n; ++j) {
                            cr[j] += av * b_p[j];
                        }
                    }
                }
            }
        }
    }
}

static void blocking_pack_col(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    const int TM = TILE_M;
    const int TN = TILE_N;
    const int nbi = (m + TM - 1) / TM;
    const int nbj = (n + TN - 1) / TN;
    const int BK  = compute_bk(TM, TN, k);

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0  = bi * TM;
            const int j0  = bj * TN;
            const int imax = std::min(i0 + TM, m);
            const int jmax = std::min(j0 + TN, n);
            const int t_m  = imax - i0;
            const int t_n  = jmax - j0;

            for (int kb = 0; kb < k; kb += BK) {
                const int kb_len = std::min(BK, k - kb);

                // Pack A[i0:imax][kb:kb+kb_len] -> packed[0..t_m-1][0..kb_len-1]
                std::vector<double> ap(static_cast<std::size_t>(t_m) * kb_len);
                for (int i = 0; i < t_m; ++i) {
                    std::memcpy(ap.data() + static_cast<std::size_t>(i) * kb_len,
                                A + static_cast<std::size_t>(i0 + i) * k + kb,
                                static_cast<std::size_t>(kb_len) * sizeof(double));
                }

                for (int l = 0; l < kb_len; ++l) {
                    for (int i = 0; i < t_m; ++i) {
                        const double av = ap[static_cast<std::size_t>(i) * kb_len + l];
                        for (int j = 0; j < t_n; ++j) {
                            C[static_cast<std::size_t>(j0 + j) * m + (i0 + i)] +=
                                av * B[static_cast<std::size_t>(j0 + j) * k + kb + l];
                        }
                    }
                }
            }
        }
    }
}

void tsmm_my_blocking_pack(int m, int n, int k,
                            const double* A, const double* B, double* C,
                            Layout layout) {
    if (layout == Layout::RowMajor) {
        blocking_pack_row(m, n, k, A, B, C);
    } else {
        blocking_pack_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_blocking_pack", tsmm_my_blocking_pack);
