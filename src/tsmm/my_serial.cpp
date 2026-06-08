#include "../tsmm.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

// ============================================================
// my_serial: k-outer serial implementation with best loop order
//
// Design rationale:
//   - k as outermost loop: C accumulates in cache, A/B streamed
//   - j as innermost loop: stride-1 contiguous access on B and C
//   - Loop-invariant a_val hoisted outside the j loop
//   - RowMajor and ColMajor handled via separate code paths at entry
// ============================================================

static void tsmm_my_serial_row(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    for (int l = 0; l < k; ++l) {
        const double* a_row = A + static_cast<std::size_t>(l) * m;
        const double* b_row = B + static_cast<std::size_t>(l) * n;
        for (int i = 0; i < m; ++i) {
            const double a_val = a_row[i];
            double* c_row = C + static_cast<std::size_t>(i) * n;
            for (int j = 0; j < n; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

static void tsmm_my_serial_col(int m, int n, int k,
                                const double* A, const double* B, double* C) {
    // ColMajor: C[j][i] = C[j*m+i], A[i][l] = A[i*k+l], B[j][l] = B[j*k+l]
    //
    // Key insight for col-major: A and B are BOTH contiguous along k (stride 1).
    // So k MUST be the innermost loop for efficient access.
    // Outer loops: j (C write stride=m) and i.

    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));

    for (int j = 0; j < n; ++j) {
        double* c_col = C + static_cast<std::size_t>(j) * m;
        const double* b_col = B + static_cast<std::size_t>(j) * k;
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            const double* a_col = A + static_cast<std::size_t>(i) * k;
            for (int l = 0; l < k; ++l) {
                sum += a_col[l] * b_col[l];  // stride-1 on both!
            }
            c_col[i] = sum;
        }
    }
}

void tsmm_my_serial(int m, int n, int k,
                     const double* A, const double* B, double* C,
                     Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_my_serial_row(m, n, k, A, B, C);
    } else {
        tsmm_my_serial_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_serial", tsmm_my_serial);


// ============================================================
// my_serial_reg: register-accumulated version for tiny m×n
//
// When m*n is small (<= 256), we can keep the entire C matrix
// in registers / L1 cache by using a local accumulator array.
// This eliminates repeated read-modify-write of C in memory.
// ============================================================

static void tsmm_my_serial_reg_row(int m, int n, int k,
                                    const double* A, const double* B, double* C) {
    // Stack-allocated local accumulator — compiler will put in registers
    // if small enough, or on stack (still in L1 cache).
    std::vector<double> C_local(static_cast<std::size_t>(m) * n, 0.0);

    for (int l = 0; l < k; ++l) {
        const double* a_row = A + static_cast<std::size_t>(l) * m;
        const double* b_row = B + static_cast<std::size_t>(l) * n;
        for (int i = 0; i < m; ++i) {
            const double a_val = a_row[i];
            double* c_row = C_local.data() + static_cast<std::size_t>(i) * n;
            for (int j = 0; j < n; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }

    std::memcpy(C, C_local.data(),
                static_cast<std::size_t>(m) * n * sizeof(double));
}

static void tsmm_my_serial_reg_col(int m, int n, int k,
                                    const double* A, const double* B, double* C) {
    std::vector<double> C_local(static_cast<std::size_t>(m) * n, 0.0);

    for (int j = 0; j < n; ++j) {
        double* c_col = C_local.data() + static_cast<std::size_t>(j) * m;
        const double* b_col = B + static_cast<std::size_t>(j) * k;
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            const double* a_col = A + static_cast<std::size_t>(i) * k;
            for (int l = 0; l < k; ++l) {
                sum += a_col[l] * b_col[l];
            }
            c_col[i] = sum;
        }
    }

    std::memcpy(C, C_local.data(),
                static_cast<std::size_t>(m) * n * sizeof(double));
}

void tsmm_my_serial_reg(int m, int n, int k,
                         const double* A, const double* B, double* C,
                         Layout layout) {
    if (layout == Layout::RowMajor) {
        tsmm_my_serial_reg_row(m, n, k, A, B, C);
    } else {
        tsmm_my_serial_reg_col(m, n, k, A, B, C);
    }
}

REGISTER_TSMM_IMPL("my_serial_reg", tsmm_my_serial_reg);
