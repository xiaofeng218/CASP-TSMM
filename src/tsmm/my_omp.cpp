#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// my_omp_static: OpenMP parallel with static scheduling
//
// Strategy:
//   - Parallelize along the k dimension (no data dependencies)
//   - Each thread maintains a private C_local accumulator
//   - After all k iterations, reduce (sum) thread-private C_locals
//   - static scheduling: iterations divided evenly at compile time
//
// This is optimal for TSMM because every k-iteration does the same
// amount of work (m*n FMAs), so static gives perfect load balance
// with zero scheduling overhead.
// ============================================================

namespace {

void omp_kernel_row(int m, int n, int k,
                    const double* A, const double* B, double* C,
                    const char* schedule_kind) {
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    const int out_size = m * n;

    // Aligned thread-private accumulators to avoid false sharing.
    // Each thread gets a private segment of size out_size.
    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;

        // Different schedule clauses are applied via a helper macro.
        // We use separate functions per schedule to keep benchmarking clean.
#pragma omp for schedule(static)
        for (int l = 0; l < k; ++l) {
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

    // Reduction: sum all thread-private C_local into final C
    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) {
            C[idx] += src[idx];
        }
    }
}

void omp_kernel_col(int m, int n, int k,
                    const double* A, const double* B, double* C) {
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
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

#pragma omp for schedule(static)
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < m; ++i) {
                const double a_val = A[static_cast<std::size_t>(i) * k + l];
                for (int j = 0; j < n; ++j) {
                    ct[static_cast<std::size_t>(j) * m + i] +=
                        a_val * B[static_cast<std::size_t>(j) * k + l];
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) {
            C[idx] += src[idx];
        }
    }
}

} // anonymous namespace

// --- my_omp_static: static scheduling (default, optimal for TSMM) ---
void tsmm_my_omp_static(int m, int n, int k,
                        const double* A, const double* B, double* C,
                        Layout layout) {
#ifdef _OPENMP
    if (layout == Layout::RowMajor) {
        omp_kernel_row(m, n, k, A, B, C, "static");
    } else {
        int nthreads = omp_get_max_threads();
        const int out_size = m * n;
        std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(static)
            for (int l = 0; l < k; ++l) {
                for (int i = 0; i < m; ++i) {
                    const double a_val = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j) {
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            a_val * B[static_cast<std::size_t>(j) * k + l];
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
#else
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C; (void)layout;
#endif
}

REGISTER_TSMM_IMPL("my_omp_static", tsmm_my_omp_static);


// --- my_omp_dynamic: dynamic scheduling (for comparison) ---

void tsmm_my_omp_dynamic(int m, int n, int k,
                          const double* A, const double* B, double* C,
                          Layout layout) {
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    const int out_size = m * n;
    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

    if (layout == Layout::RowMajor) {
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(dynamic)
            for (int l = 0; l < k; ++l) {
                const double* a_row = A + static_cast<std::size_t>(l) * m;
                const double* b_row = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double a_val = a_row[i];
                    double* c_row = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j)
                        c_row[j] += a_val * b_row[j];
                }
            }
        }
    } else {
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(dynamic)
            for (int l = 0; l < k; ++l) {
                for (int i = 0; i < m; ++i) {
                    const double a_val = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j)
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            a_val * B[static_cast<std::size_t>(j) * k + l];
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
#else
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C; (void)layout;
#endif
}

REGISTER_TSMM_IMPL("my_omp_dynamic", tsmm_my_omp_dynamic);


// --- my_omp_guided: guided scheduling (for comparison) ---

void tsmm_my_omp_guided(int m, int n, int k,
                         const double* A, const double* B, double* C,
                         Layout layout) {
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    const int out_size = m * n;
    std::vector<double> tmp(static_cast<std::size_t>(nthreads) * out_size, 0.0);

    if (layout == Layout::RowMajor) {
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(guided)
            for (int l = 0; l < k; ++l) {
                const double* a_row = A + static_cast<std::size_t>(l) * m;
                const double* b_row = B + static_cast<std::size_t>(l) * n;
                for (int i = 0; i < m; ++i) {
                    const double a_val = a_row[i];
                    double* c_row = ct + static_cast<std::size_t>(i) * n;
                    for (int j = 0; j < n; ++j)
                        c_row[j] += a_val * b_row[j];
                }
            }
        }
    } else {
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* ct = tmp.data() + static_cast<std::size_t>(tid) * out_size;
#pragma omp for schedule(guided)
            for (int l = 0; l < k; ++l) {
                for (int i = 0; i < m; ++i) {
                    const double a_val = A[static_cast<std::size_t>(i) * k + l];
                    for (int j = 0; j < n; ++j)
                        ct[static_cast<std::size_t>(j) * m + i] +=
                            a_val * B[static_cast<std::size_t>(j) * k + l];
                }
            }
        }
    }

    std::memset(C, 0, static_cast<std::size_t>(out_size) * sizeof(double));
    for (int t = 0; t < nthreads; ++t) {
        const double* src = tmp.data() + static_cast<std::size_t>(t) * out_size;
        for (int idx = 0; idx < out_size; ++idx) C[idx] += src[idx];
    }
#else
    (void)m; (void)n; (void)k; (void)A; (void)B; (void)C; (void)layout;
#endif
}

REGISTER_TSMM_IMPL("my_omp_guided", tsmm_my_omp_guided);
