#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// my_omp_static/dynamic/guided: OpenMP parallel TSMM
//
// Strategy:
//   - For tiny m×n (≤ 256): thread-private C_local + k-parallel
//   - For larger m×n: m×n tiling (small tiles, iterate all k)
//     → O(m*n) memory, no thread-private copies
// ============================================================

namespace {

constexpr int TINY_THRESH = 256;   // m*n threshold for thread-private approach
constexpr int TILE_M = 16;
constexpr int TILE_N = 64;

// --- Tiny m×n: thread-private C_local + k-parallel ---

template<const char* Schedule>
void omp_tiny_row(int m, int n, int k, const double* A, const double* B, double* C) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
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

        if (Schedule[0] == 's' && Schedule[1] == 't') { // static
#pragma omp for schedule(static)
            for (int l = 0; l < k; ++l) {
                const double* ar = A + static_cast<std::size_t>(l) * m;
                const double* br = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double av = ar[i];
                    double* cr = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) cr[j] += av * br[j];
                }
            }
        } else if (Schedule[0] == 'd') { // dynamic
#pragma omp for schedule(dynamic)
            for (int l = 0; l < k; ++l) {
                const double* ar = A + static_cast<std::size_t>(l) * m;
                const double* br = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double av = ar[i];
                    double* cr = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) cr[j] += av * br[j];
                }
            }
        } else { // guided
#pragma omp for schedule(guided)
            for (int l = 0; l < k; ++l) {
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

template<const char* Schedule>
void omp_tiny_col(int m, int n, int k, const double* A, const double* B, double* C) {
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
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

        if (Schedule[0] == 's' && Schedule[1] == 't') {
#pragma omp for schedule(static)
            for (int l = 0; l < k; ++l)
                for (int i = 0; i < m; ++i) {
                    const double av = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j)
                        ct[static_cast<std::size_t>(j) * m + i] += av * B[static_cast<std::size_t>(j) * k + l];
                }
        } else if (Schedule[0] == 'd') {
#pragma omp for schedule(dynamic)
            for (int l = 0; l < k; ++l)
                for (int i = 0; i < m; ++i) {
                    const double av = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j)
                        ct[static_cast<std::size_t>(j) * m + i] += av * B[static_cast<std::size_t>(j) * k + l];
                }
        } else {
#pragma omp for schedule(guided)
            for (int l = 0; l < k; ++l)
                for (int i = 0; i < m; ++i) {
                    const double av = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j)
                        ct[static_cast<std::size_t>(j) * m + i] += av * B[static_cast<std::size_t>(j) * k + l];
                }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(sz) * sizeof(double));
    for (int tid = 0; tid < nth; ++tid) {
        const double* s = t.data() + static_cast<std::size_t>(tid) * sz;
        for (int idx = 0; idx < sz; ++idx) C[idx] += s[idx];
    }
}

// --- Larger m×n: m×n tiling, direct C write, iterate all k ---

template<const char* Schedule>
void omp_tiled_row(int m, int n, int k, const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + TILE_M - 1) / TILE_M;
    const int nbj = (n + TILE_N - 1) / TILE_N;

    if (Schedule[0] == 's' && Schedule[1] == 't') {
#pragma omp parallel for collapse(2) schedule(static)
        for (int bi = 0; bi < nbi; ++bi) {
            for (int bj = 0; bj < nbj; ++bj) {
                const int i0 = bi * TILE_M, j0 = bj * TILE_N;
                const int imax = std::min(i0 + TILE_M, m), jmax = std::min(j0 + TILE_N, n);
                const int tm = imax - i0, tn = jmax - j0;
                for (int l = 0; l < k; ++l) {
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
    } else {
#pragma omp parallel for collapse(2)
        for (int bi = 0; bi < nbi; ++bi) {
            for (int bj = 0; bj < nbj; ++bj) {
                const int i0 = bi * TILE_M, j0 = bj * TILE_N;
                const int imax = std::min(i0 + TILE_M, m), jmax = std::min(j0 + TILE_N, n);
                const int tm = imax - i0, tn = jmax - j0;
                for (int l = 0; l < k; ++l) {
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

template<const char* Schedule>
void omp_tiled_col(int m, int n, int k, const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + TILE_M - 1) / TILE_M;
    const int nbj = (n + TILE_N - 1) / TILE_N;

#pragma omp parallel for collapse(2)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TILE_M, j0 = bj * TILE_N;
            const int imax = std::min(i0 + TILE_M, m), jmax = std::min(j0 + TILE_N, n);
            const int tm = imax - i0, tn = jmax - j0;
            for (int l = 0; l < k; ++l)
                for (int i = 0; i < tm; ++i) {
                    const double av = A[static_cast<std::size_t>(i0 + i) * k + l];
                    for (int j = 0; j < tn; ++j)
                        C[static_cast<std::size_t>(j0 + j) * m + (i0 + i)] +=
                            av * B[static_cast<std::size_t>(j0 + j) * k + l];
                }
        }
    }
}

// Schedule tag strings for template dispatch
static const char s_static[]  = "static";
static const char s_dynamic[] = "dynamic";
static const char s_guided[]  = "guided";

} // anonymous namespace


// --- Public entry points ---

void tsmm_my_omp_static(int m, int n, int k,
                         const double* A, const double* B, double* C,
                         Layout layout) {
    if (m * n <= TINY_THRESH) {
        if (layout == Layout::RowMajor)
            omp_tiny_row<s_static>(m, n, k, A, B, C);
        else
            omp_tiny_col<s_static>(m, n, k, A, B, C);
    } else {
        if (layout == Layout::RowMajor)
            omp_tiled_row<s_static>(m, n, k, A, B, C);
        else
            omp_tiled_col<s_static>(m, n, k, A, B, C);
    }
}
REGISTER_TSMM_IMPL("my_omp_static", tsmm_my_omp_static);

void tsmm_my_omp_dynamic(int m, int n, int k,
                          const double* A, const double* B, double* C,
                          Layout layout) {
    if (m * n <= TINY_THRESH) {
        if (layout == Layout::RowMajor)
            omp_tiny_row<s_dynamic>(m, n, k, A, B, C);
        else
            omp_tiny_col<s_dynamic>(m, n, k, A, B, C);
    } else {
        if (layout == Layout::RowMajor)
            omp_tiled_row<s_dynamic>(m, n, k, A, B, C);
        else
            omp_tiled_col<s_dynamic>(m, n, k, A, B, C);
    }
}
REGISTER_TSMM_IMPL("my_omp_dynamic", tsmm_my_omp_dynamic);

void tsmm_my_omp_guided(int m, int n, int k,
                         const double* A, const double* B, double* C,
                         Layout layout) {
    if (m * n <= TINY_THRESH) {
        if (layout == Layout::RowMajor)
            omp_tiny_row<s_guided>(m, n, k, A, B, C);
        else
            omp_tiny_col<s_guided>(m, n, k, A, B, C);
    } else {
        if (layout == Layout::RowMajor)
            omp_tiled_row<s_guided>(m, n, k, A, B, C);
        else
            omp_tiled_col<s_guided>(m, n, k, A, B, C);
    }
}
REGISTER_TSMM_IMPL("my_omp_guided", tsmm_my_omp_guided);
