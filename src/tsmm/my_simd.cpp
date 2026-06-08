#include "../tsmm.hpp"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// ============================================================
// my_simd: Simple AVX-512 kernel for optimization progression
//
// Demonstrates basic AVX-512 usage: masked load/store,
// 4-way unrolled FMA, k-outer with stride-1 inner loop.
// Static for compiler optimization.
// ============================================================

namespace {

constexpr int TILE_M = 16;
constexpr int TILE_N = 64;

} // anonymous namespace

static void simd_row(int m, int n, int k,
                      const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + TILE_M - 1) / TILE_M;
    const int nbj = (n + TILE_N - 1) / TILE_N;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TILE_M, j0 = bj * TILE_N;
            const int imax = std::min(i0 + TILE_M, m), jmax = std::min(j0 + TILE_N, n);
            const int tm = imax - i0, tn = jmax - j0;

            for (int l = 0; l < k; ++l) {
                const double* ar = A + static_cast<std::size_t>(l) * m;
                const double* br = B + static_cast<std::size_t>(l) * n;
#ifdef __AVX512F__
                for (int i = 0; i < tm; ++i) {
                    const __m512d av = _mm512_set1_pd(ar[i0 + i]);
                    double* cr = C + static_cast<std::size_t>(i0 + i) * n + j0;
                    int j = 0;
                    for (; j + 31 < tn; j += 32) {
                        __m512d c0 = _mm512_loadu_pd(cr + j), c1 = _mm512_loadu_pd(cr + j + 8);
                        __m512d c2 = _mm512_loadu_pd(cr + j + 16), c3 = _mm512_loadu_pd(cr + j + 24);
                        c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j), c0);
                        c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j + 8), c1);
                        c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j + 16), c2);
                        c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(br + j0 + j + 24), c3);
                        _mm512_storeu_pd(cr + j, c0); _mm512_storeu_pd(cr + j + 8, c1);
                        _mm512_storeu_pd(cr + j + 16, c2); _mm512_storeu_pd(cr + j + 24, c3);
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

static void simd_col(int m, int n, int k,
                      const double* A, const double* B, double* C) {
    std::memset(C, 0, static_cast<std::size_t>(m) * n * sizeof(double));
    const int nbi = (m + TILE_M - 1) / TILE_M;
    const int nbj = (n + TILE_N - 1) / TILE_N;

#pragma omp parallel for collapse(2) schedule(static)
    for (int bi = 0; bi < nbi; ++bi) {
        for (int bj = 0; bj < nbj; ++bj) {
            const int i0 = bi * TILE_M, j0 = bj * TILE_N;
            const int imax = std::min(i0 + TILE_M, m), jmax = std::min(j0 + TILE_N, n);
            const int tm = imax - i0, tn = jmax - j0;

            for (int jj = 0; jj < tn; ++jj) {
                const int j_idx = j0 + jj;
                double* c_col = C + static_cast<std::size_t>(j_idx) * m + i0;
                const double* b_col = B + static_cast<std::size_t>(j_idx) * k;
                for (int ii = 0; ii < tm; ++ii) {
                    double sum = 0.0;
                    const double* a_col = A + static_cast<std::size_t>(i0 + ii) * k;
#ifdef __AVX512F__
                    int l = 0;
                    __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd();
                    for (; l + 15 < k; l += 16) {
                        s0 = _mm512_fmadd_pd(_mm512_loadu_pd(a_col + l), _mm512_loadu_pd(b_col + l), s0);
                        s1 = _mm512_fmadd_pd(_mm512_loadu_pd(a_col + l + 8), _mm512_loadu_pd(b_col + l + 8), s1);
                    }
                    sum += _mm512_reduce_add_pd(s0) + _mm512_reduce_add_pd(s1);
                    for (; l + 7 < k; l += 8)
                        sum += _mm512_reduce_add_pd(_mm512_mul_pd(_mm512_loadu_pd(a_col + l), _mm512_loadu_pd(b_col + l)));
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

void tsmm_my_simd(int m, int n, int k,
                   const double* A, const double* B, double* C,
                   Layout layout) {
    if (layout == Layout::RowMajor) simd_row(m, n, k, A, B, C);
    else                             simd_col(m, n, k, A, B, C);
}

REGISTER_TSMM_IMPL("my_simd", tsmm_my_simd);
