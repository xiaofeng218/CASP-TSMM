#pragma once

#include <cstddef>
#include <vector>

enum class Layout {
    RowMajor,
    ColMajor,
};

struct Problem {
    int m;
    int n;
    int k;
    const char* name;
    bool required;
};

using TsmmKernel = void (*)(int m, int n, int k,
                            const double* A, const double* B, double* C,
                            Layout layout);

struct ImplDesc {
    const char* name;
    TsmmKernel fn;
    bool is_ref;
};

std::vector<ImplDesc>& tsmm_impl_registry();

struct TsmmImplRegistrar {
    TsmmImplRegistrar(const char* name, TsmmKernel fn) {
        tsmm_impl_registry().push_back({name, fn, false});
    }
};

#define REGISTER_TSMM_IMPL(name_literal, fn_name) \
    static TsmmImplRegistrar tsmm_registrar_##fn_name(name_literal, fn_name)

inline std::size_t idx_a(Layout layout, int l, int i, int k, int m) {
    return layout == Layout::RowMajor
        ? static_cast<std::size_t>(l) * m + i
        : static_cast<std::size_t>(i) * k + l;
}

inline std::size_t idx_b(Layout layout, int l, int j, int k, int n) {
    return layout == Layout::RowMajor
        ? static_cast<std::size_t>(l) * n + j
        : static_cast<std::size_t>(j) * k + l;
}

inline std::size_t idx_c(Layout layout, int i, int j, int m, int n) {
    return layout == Layout::RowMajor
        ? static_cast<std::size_t>(i) * n + j
        : static_cast<std::size_t>(j) * m + i;
}

void tsmm_reference(int m, int n, int k, const double* A, const double* B, double* C, Layout layout);
void tsmm_naive(int m, int n, int k, const double* A, const double* B, double* C, Layout layout);
void tsmm_opt(int m, int n, int k, const double* A, const double* B, double* C, Layout layout);
