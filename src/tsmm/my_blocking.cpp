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
// Design rationale:
//   - Block along the k dimension to fit working set in L2 cache
//   - Auto-compute BK (k block size) based on:
//       BK*m + BK*n + m*n < L2_SIZE / sizeof(double)
//   - For the Xeon Platinum 9242, L2 = 1024 KB per core
//   - BK is aligned to 8 (SIMD width of AVX-512)
// ============================================================

namespace {

// L2 cache size per core on Xeon Platinum 9242: 1024 KB
constexpr int L2_SIZE_BYTES = 1024 * 1024;   // 1 MB
constexpr int SIMD_WIDTH = 8;                 // AVX-512: 8 doubles

// Auto-compute optimal k-block size for a given (m, n)
int compute_bk(int m, int n, int k) {
    const int out_size = m * n;
    const int l2_doubles = L2_SIZE_BYTES / static_cast<int>(sizeof(double)); // 131072

    // Constraint: BK*(m+n) + m*n <= l2_doubles
    // → BK <= (l2_doubles - m*n) / (m+n)
    int denom = m + n;
    if (denom <= 0) denom = 1;
    int bk = (l2_doubles - out_size) / denom;

    // Align to SIMD width
    if (bk < SIMD_WIDTH) bk = SIMD_WIDTH;
    bk = (bk / SIMD_WIDTH) * SIMD_WIDTH;

    if (bk > k) bk = k;
    if (bk < 8) bk = 8;

    return bk;
}

} // anonymous namespace


// --- my_blocking: simple k-blocking without packing ---

static void blocking_row(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    const int BK = compute_bk(m, n, k);
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

        // Outer loop: iterate over k-blocks
        for (int kb = 0; kb < k; kb += BK) {
            const int kb_end = std::min(kb + BK, k);

#pragma omp for schedule(static)
            for (int l = kb; l < kb_end; ++l) {
                const double* a_row = A + static_cast<std::size_t>(l) * m;
                const double* b_row = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double a_val = a_row[i];
                    double* c_row = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) {
                        c_row[j] += a_val * b_row[j];
                    }
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
}

static void blocking_col(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    const int BK = compute_bk(m, n, k);
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
                for (int i = 0; i < m; ++i) {
                    const double a_val = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j) {
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            a_val * B[static_cast<std::size_t>(j) * k + l];
                    }
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
}

void tsmm_my_blocking(int m, int n, int k,
                       const double* A, const double* B, double* C,
                       Layout layout) {
    if (layout == Layout::RowMajor) {
        blocking_row(m, n, k, A, B, C);
    } else {
        blocking_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_blocking", tsmm_my_blocking);


// --- my_blocking_pack: blocking + B packing for contiguous access ---

static void blocking_pack_row(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    const int BK = compute_bk(m, n, k);
    const int nthreads = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int out_size = m * n;

    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

    // Pack buffer for B: one buffer per block of size BK * n
    std::vector<double> B_packed;

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;

        for (int kb = 0; kb < k; kb += BK) {
            const int kb_len = std::min(BK, k - kb);

            // Master thread packs B for this block
#pragma omp single
            {
                B_packed.resize(static_cast<std::size_t>(kb_len) * n);
                for (int l = 0; l < kb_len; ++l) {
                    const double* b_src = B + static_cast<std::size_t>(kb + l) * n;
                    double* b_dst = B_packed.data() + static_cast<std::size_t>(l) * n;
                    std::memcpy(b_dst, b_src, static_cast<std::size_t>(n) * sizeof(double));
                }
            }

            // Process packed block
#pragma omp for schedule(static)
            for (int l = 0; l < kb_len; ++l) {
                const double* a_row = A + static_cast<std::size_t>(kb + l) * m;
                const double* b_packed = B_packed.data() + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double a_val = a_row[i];
                    double* c_row = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j) {
                        c_row[j] += a_val * b_packed[j];
                    }
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
}

static void blocking_pack_col(int m, int n, int k,
                               const double* A, const double* B, double* C) {
    const int BK = compute_bk(m, n, k);
    const int nthreads = []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }();
    const int out_size = m * n;

    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);
    std::vector<double> A_packed;

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;

        for (int kb = 0; kb < k; kb += BK) {
            const int kb_len = std::min(BK, k - kb);

            // Master thread packs A for this block (A is m×k col-major = m rows of length k)
            // We need A[0..m-1][kb..kb+BK-1], packed as contiguous m × kb_len block
#pragma omp single
            {
                A_packed.resize(static_cast<std::size_t>(m) * kb_len);
                for (int i = 0; i < m; ++i) {
                    const double* a_src = A + static_cast<std::size_t>(i) * k + kb;
                    double* a_dst = A_packed.data() + static_cast<std::size_t>(i) * kb_len;
                    std::memcpy(a_dst, a_src,
                                static_cast<std::size_t>(kb_len) * sizeof(double));
                }
            }

#pragma omp for schedule(static)
            for (int l = 0; l < kb_len; ++l) {
                for (int i = 0; i < m; ++i) {
                    const double a_val = A_packed[static_cast<std::size_t>(i) * kb_len + l];
                    for (int j = 0; j < n; ++j) {
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            a_val * B[static_cast<std::size_t>(j) * k + kb + l];
                    }
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
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
