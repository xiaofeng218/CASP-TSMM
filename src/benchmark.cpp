/**
 * TSMM Benchmark: C = A^T * B
 *   A in R^{k x m}  (row-major, k rows, m cols, lda = m)
 *   B in R^{k x n}  (row-major, k rows, n cols, ldb = n)
 *   C in R^{m x n}  (row-major, m rows, n cols, ldc = n)
 *
 * C[i][j] = sum_{l=0}^{k-1} A[l][i] * B[l][j]
 *
 * Build:
 *   make BLAS=mkl       (Intel MKL, requires MKLROOT)
 *   make BLAS=openblas  (OpenBLAS)
 *   make BLAS=none      (built-in reference, no external BLAS)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#ifndef _WIN32
#include <unistd.h>   /* gethostname on Linux */
#else
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#if defined(HAVE_MKL)
  #include <mkl_cblas.h>
#elif defined(HAVE_OPENBLAS)
  #include <cblas.h>
#endif

// ============================================================
//  Types and problem definitions
// ============================================================

struct Problem {
    int m, n, k;
    const char* name;
    bool required;
};

static const Problem PROBLEMS[] = {
    // Required
    {4000,    16000,    128,  "4000x16000x128",    true},
    {8,       16,       16000,"8x16x16000",         true},
    {32,      16000,    16,   "32x16000x16",        true},
    {144,     144,      144,  "144x144x144",        true},
    // Optional
    {16,      12344,    16,   "16x12344x16",        false},
    {4,       64,       606841,"4x64x606841",       false},
    {442,     193,      11,   "442x193x11",         false},
    {40,      1127228,  40,   "40x1127228x40",      false},
};
static const int N_PROBLEMS = 8;

// ============================================================
//  Timing
// ============================================================

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ============================================================
//  Memory allocation (64-byte aligned for AVX-512)
// ============================================================

static double* alloc_mat(size_t rows, size_t cols) {
    size_t bytes = rows * cols * sizeof(double);
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, 64);
    if (!p) {
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) {
#endif
        fprintf(stderr, "alloc_mat: failed to allocate %zu bytes\n", bytes);
        exit(1);
    }
    return (double*)p;
}

static void free_mat(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static inline double tsmm_flops(int m, int n, int k) {
    return 2.0 * (double)m * (double)n * (double)k;
}

static void fill_rand(double* A, size_t n) {
    for (size_t i = 0; i < n; i++)
        A[i] = (double)rand() / RAND_MAX - 0.5;
}

// ============================================================
//  Correctness check
// ============================================================

static bool check(const double* ref, const double* got, size_t n, double rtol = 1e-8) {
    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs(ref[i] - got[i]);
        double r = fabs(ref[i]);
        if (e > max_err) max_err = e;
        if (r > max_ref) max_ref = r;
    }
    if (max_ref < 1e-30) return max_err < 1e-12;
    return (max_err / max_ref) < rtol;
}

// ============================================================
//  Implementation 0: Reference (CBLAS dgemm or built-in)
// ============================================================

#if defined(HAVE_MKL) || defined(HAVE_OPENBLAS)
static void tsmm_ref(int m, int n, int k,
                     const double* A, const double* B, double* C) {
    // C(m x n) = A^T(m x k) * B(k x n)
    // A stored as k x m row-major, lda = m
    // B stored as k x n row-major, ldb = n
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                m, n, k,
                1.0, A, m,
                     B, n,
                0.0, C, n);
}
#else
// Fallback: plain C reference (used when no BLAS available)
static void tsmm_ref(int m, int n, int k,
                     const double* A, const double* B, double* C) {
    memset(C, 0, (size_t)m * n * sizeof(double));
    for (int l = 0; l < k; l++) {
        const double* a = A + (size_t)l * m;
        const double* b = B + (size_t)l * n;
        for (int i = 0; i < m; i++) {
            double av = a[i];
            double* c = C + (size_t)i * n;
            for (int j = 0; j < n; j++)
                c[j] += av * b[j];
        }
    }
}
#endif

// ============================================================
//  Implementation 1: Naive (serial, lij loop order)
// ============================================================

static void tsmm_naive(int m, int n, int k,
                       const double* A, const double* B, double* C) {
    memset(C, 0, (size_t)m * n * sizeof(double));
    for (int l = 0; l < k; l++) {
        const double* a = A + (size_t)l * m;
        const double* b = B + (size_t)l * n;
        for (int i = 0; i < m; i++) {
            double av = a[i];
            double* c = C + (size_t)i * n;
            for (int j = 0; j < n; j++)
                c[j] += av * b[j];
        }
    }
}

// ============================================================
//  Implementation 2: OpenMP (parallel over m rows of C)
// ============================================================

static void tsmm_openmp(int m, int n, int k,
                        const double* A, const double* B, double* C) {
    memset(C, 0, (size_t)m * n * sizeof(double));
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < m; i++) {
        double* c = C + (size_t)i * n;
        for (int l = 0; l < k; l++) {
            double av = A[(size_t)l * m + i];
            const double* b = B + (size_t)l * n;
            for (int j = 0; j < n; j++)
                c[j] += av * b[j];
        }
    }
}

// ============================================================
//  Implementation 3: Cache-blocked (jb x lb x ib tiling)
//  Tuned for Xeon Platinum 9242: L1d=32KB, L2=1MB
// ============================================================

static void tsmm_blocked(int m, int n, int k,
                         const double* A, const double* B, double* C) {
    // Block sizes (tunable via env vars at runtime)
    static int IB = 0, JB = 0, LB = 0;
    if (IB == 0) {
        IB = getenv("TSMM_IB") ? atoi(getenv("TSMM_IB")) : 64;
        JB = getenv("TSMM_JB") ? atoi(getenv("TSMM_JB")) : 512;
        LB = getenv("TSMM_LB") ? atoi(getenv("TSMM_LB")) : 32;
    }

    memset(C, 0, (size_t)m * n * sizeof(double));

    // Parallel over j-blocks; each thread gets independent C columns
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int j0 = 0; j0 < n; j0 += JB) {
        int jlen = std::min(JB, n - j0);

        for (int i0 = 0; i0 < m; i0 += IB) {
            int ilen = std::min(IB, m - i0);

            for (int l0 = 0; l0 < k; l0 += LB) {
                int llen = std::min(LB, k - l0);

                for (int l = l0; l < l0 + llen; l++) {
                    const double* b = B + (size_t)l * n + j0;
                    for (int i = i0; i < i0 + ilen; i++) {
                        double av = A[(size_t)l * m + i];
                        double* c = C + (size_t)i * n + j0;
                        for (int j = 0; j < jlen; j++)
                            c[j] += av * b[j];
                    }
                }
            }
        }
    }
}

// ============================================================
//  Implementation 4: AVX-512 SIMD (vectorize j, serial)
// ============================================================

static void tsmm_avx512(int m, int n, int k,
                        const double* A, const double* B, double* C) {
    memset(C, 0, (size_t)m * n * sizeof(double));

    for (int l = 0; l < k; l++) {
        const double* a = A + (size_t)l * m;
        const double* b = B + (size_t)l * n;

        for (int i = 0; i < m; i++) {
            double av_d = a[i];
            double* c = C + (size_t)i * n;

#ifdef __AVX512F__
            __m512d av = _mm512_set1_pd(av_d);
            int j = 0;
            for (; j + 31 < n; j += 32) {
                __m512d c0 = _mm512_loadu_pd(c + j);
                __m512d c1 = _mm512_loadu_pd(c + j + 8);
                __m512d c2 = _mm512_loadu_pd(c + j + 16);
                __m512d c3 = _mm512_loadu_pd(c + j + 24);
                c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j),      c0);
                c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 8),  c1);
                c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 16), c2);
                c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 24), c3);
                _mm512_storeu_pd(c + j,      c0);
                _mm512_storeu_pd(c + j + 8,  c1);
                _mm512_storeu_pd(c + j + 16, c2);
                _mm512_storeu_pd(c + j + 24, c3);
            }
            for (; j + 7 < n; j += 8) {
                __m512d cv = _mm512_loadu_pd(c + j);
                cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                _mm512_storeu_pd(c + j, cv);
            }
            for (; j < n; j++)
                c[j] += av_d * b[j];
#else
            for (int j = 0; j < n; j++)
                c[j] += av_d * b[j];
#endif
        }
    }
}

// ============================================================
//  Implementation 5: AVX-512 + OpenMP (parallel over i rows)
// ============================================================

static void tsmm_avx512_omp(int m, int n, int k,
                             const double* A, const double* B, double* C) {
    memset(C, 0, (size_t)m * n * sizeof(double));

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < m; i++) {
        double* c = C + (size_t)i * n;

        for (int l = 0; l < k; l++) {
            double av_d = A[(size_t)l * m + i];
            const double* b = B + (size_t)l * n;

#ifdef __AVX512F__
            __m512d av = _mm512_set1_pd(av_d);
            int j = 0;
            for (; j + 31 < n; j += 32) {
                __m512d c0 = _mm512_loadu_pd(c + j);
                __m512d c1 = _mm512_loadu_pd(c + j + 8);
                __m512d c2 = _mm512_loadu_pd(c + j + 16);
                __m512d c3 = _mm512_loadu_pd(c + j + 24);
                c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j),      c0);
                c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 8),  c1);
                c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 16), c2);
                c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 24), c3);
                _mm512_storeu_pd(c + j,      c0);
                _mm512_storeu_pd(c + j + 8,  c1);
                _mm512_storeu_pd(c + j + 16, c2);
                _mm512_storeu_pd(c + j + 24, c3);
            }
            for (; j + 7 < n; j += 8) {
                __m512d cv = _mm512_loadu_pd(c + j);
                cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                _mm512_storeu_pd(c + j, cv);
            }
            for (; j < n; j++)
                c[j] += av_d * b[j];
#else
            for (int j = 0; j < n; j++)
                c[j] += av_d * b[j];
#endif
        }
    }
}

// ============================================================
//  Implementation 6: AVX-512 + Blocking + OpenMP (best effort)
//  For small m (e.g., m=8): parallelize over k with reduction
//  For large m: parallelize over j-blocks
// ============================================================

static void tsmm_opt(int m, int n, int k,
                     const double* A, const double* B, double* C) {
    static int IB = 0, JB = 0, LB = 0;
    if (IB == 0) {
        IB = getenv("OPT_IB") ? atoi(getenv("OPT_IB")) : 64;
        JB = getenv("OPT_JB") ? atoi(getenv("OPT_JB")) : 512;
        LB = getenv("OPT_LB") ? atoi(getenv("OPT_LB")) : 32;
    }

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // For very small m: keep C in cache, parallelize over k
    if (m <= nthreads && m < 64) {
        // Local per-thread C buffers
        std::vector<std::vector<double>> Ctmp(nthreads,
                                              std::vector<double>((size_t)m * n, 0.0));
#ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double* ct = Ctmp[tid].data();
            #pragma omp for schedule(static)
            for (int l = 0; l < k; l++) {
                const double* a = A + (size_t)l * m;
                const double* b = B + (size_t)l * n;
                for (int i = 0; i < m; i++) {
#ifdef __AVX512F__
                    __m512d av = _mm512_set1_pd(a[i]);
                    double* cr = ct + (size_t)i * n;
                    int j = 0;
                    for (; j + 7 < n; j += 8) {
                        __m512d cv = _mm512_loadu_pd(cr + j);
                        cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                        _mm512_storeu_pd(cr + j, cv);
                    }
                    for (; j < n; j++)
                        cr[j] += a[i] * b[j];
#else
                    double* cr = ct + (size_t)i * n;
                    for (int j = 0; j < n; j++)
                        cr[j] += a[i] * b[j];
#endif
                }
            }
        }
#else
        double* ct = Ctmp[0].data();
        for (int l = 0; l < k; l++) {
            const double* a = A + (size_t)l * m;
            const double* b = B + (size_t)l * n;
            for (int i = 0; i < m; i++) {
                double* cr = ct + (size_t)i * n;
                for (int j = 0; j < n; j++)
                    cr[j] += a[i] * b[j];
            }
        }
#endif
        // Reduce into C
        memset(C, 0, (size_t)m * n * sizeof(double));
        for (int t = 0; t < nthreads; t++) {
            const double* ct = Ctmp[t].data();
            for (size_t idx = 0; idx < (size_t)m * n; idx++)
                C[idx] += ct[idx];
        }
        return;
    }

    // General case: parallel over j-blocks with AVX-512 + blocking
    memset(C, 0, (size_t)m * n * sizeof(double));

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int j0 = 0; j0 < n; j0 += JB) {
        int jlen = std::min(JB, n - j0);

        for (int i0 = 0; i0 < m; i0 += IB) {
            int ilen = std::min(IB, m - i0);

            for (int l0 = 0; l0 < k; l0 += LB) {
                int llen = std::min(LB, k - l0);

                for (int l = l0; l < l0 + llen; l++) {
                    const double* b = B + (size_t)l * n + j0;

                    for (int i = i0; i < i0 + ilen; i++) {
                        double av_d = A[(size_t)l * m + i];
                        double* c = C + (size_t)i * n + j0;

#ifdef __AVX512F__
                        __m512d av = _mm512_set1_pd(av_d);
                        int j = 0;
                        for (; j + 31 < jlen; j += 32) {
                            __m512d c0 = _mm512_loadu_pd(c + j);
                            __m512d c1 = _mm512_loadu_pd(c + j + 8);
                            __m512d c2 = _mm512_loadu_pd(c + j + 16);
                            __m512d c3 = _mm512_loadu_pd(c + j + 24);
                            c0 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j),      c0);
                            c1 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 8),  c1);
                            c2 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 16), c2);
                            c3 = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j + 24), c3);
                            _mm512_storeu_pd(c + j,      c0);
                            _mm512_storeu_pd(c + j + 8,  c1);
                            _mm512_storeu_pd(c + j + 16, c2);
                            _mm512_storeu_pd(c + j + 24, c3);
                        }
                        for (; j + 7 < jlen; j += 8) {
                            __m512d cv = _mm512_loadu_pd(c + j);
                            cv = _mm512_fmadd_pd(av, _mm512_loadu_pd(b + j), cv);
                            _mm512_storeu_pd(c + j, cv);
                        }
                        for (; j < jlen; j++)
                            c[j] += av_d * b[j];
#else
                        for (int j = 0; j < jlen; j++)
                            c[j] += av_d * b[j];
#endif
                    }
                }
            }
        }
    }
}

// ============================================================
//  Benchmark harness
// ============================================================

struct ImplDesc {
    const char* name;
    void (*fn)(int, int, int, const double*, const double*, double*);
    bool is_ref;
};

static ImplDesc IMPLS[] = {
    {"reference",   tsmm_ref,        true},
    {"naive",       tsmm_naive,      false},
    {"openmp",      tsmm_openmp,     false},
    {"blocked",     tsmm_blocked,    false},
    {"avx512",      tsmm_avx512,     false},
    {"avx512_omp",  tsmm_avx512_omp, false},
    {"opt",         tsmm_opt,        false},
};
static const int N_IMPLS = 7;

struct BenchResult {
    std::string impl_name;
    std::string prob_name;
    int m, n, k;
    bool required;
    double time_ms;
    double gflops;
    double speedup;   // vs reference
    bool correct;
};

// Escape strings for JSON
static std::string json_str(const std::string& s) {
    return "\"" + s + "\"";
}

static void write_json(const std::string& path,
                       const std::vector<BenchResult>& results,
                       const std::string& hostname,
                       int nthreads,
                       const std::string& timestamp) {
    // Group results by problem
    struct ProbSummary {
        std::string name;
        int m, n, k;
        bool required;
        std::vector<BenchResult> impls;
    };
    std::vector<ProbSummary> probs;

    for (int p = 0; p < N_PROBLEMS; p++) {
        ProbSummary ps;
        ps.name = PROBLEMS[p].name;
        ps.m = PROBLEMS[p].m;
        ps.n = PROBLEMS[p].n;
        ps.k = PROBLEMS[p].k;
        ps.required = PROBLEMS[p].required;
        for (auto& r : results)
            if (r.prob_name == ps.name)
                ps.impls.push_back(r);
        if (!ps.impls.empty())
            probs.push_back(ps);
    }

    // Compute geometric mean speedup for required problems per impl
    std::vector<std::pair<std::string, double>> geomeans;
    for (int ii = 0; ii < N_IMPLS; ii++) {
        if (IMPLS[ii].is_ref) continue;
        double logsum = 0.0;
        int cnt = 0;
        for (auto& r : results) {
            if (r.impl_name == IMPLS[ii].name && r.required && r.correct) {
                logsum += log(r.speedup > 0 ? r.speedup : 1e-9);
                cnt++;
            }
        }
        if (cnt > 0)
            geomeans.push_back({IMPLS[ii].name, exp(logsum / cnt)});
    }

    // Write JSON
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) { perror(("write_json: " + path).c_str()); return; }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp.c_str());
    fprintf(fp, "  \"hostname\": \"%s\",\n", hostname.c_str());
    fprintf(fp, "  \"n_threads\": %d,\n", nthreads);
    fprintf(fp, "  \"avx512\": %s,\n",
#ifdef __AVX512F__
            "true"
#else
            "false"
#endif
    );
    fprintf(fp, "  \"blas\": \"%s\",\n",
#if defined(HAVE_MKL)
            "mkl"
#elif defined(HAVE_OPENBLAS)
            "openblas"
#else
            "none"
#endif
    );

    // Problems
    fprintf(fp, "  \"problems\": [\n");
    for (size_t pi = 0; pi < probs.size(); pi++) {
        auto& ps = probs[pi];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"name\": \"%s\",\n", ps.name.c_str());
        fprintf(fp, "      \"m\": %d, \"n\": %d, \"k\": %d,\n", ps.m, ps.n, ps.k);
        fprintf(fp, "      \"required\": %s,\n", ps.required ? "true" : "false");
        fprintf(fp, "      \"impls\": [\n");
        for (size_t ii = 0; ii < ps.impls.size(); ii++) {
            auto& r = ps.impls[ii];
            fprintf(fp, "        {\"name\": \"%s\", \"time_ms\": %.4f, \"gflops\": %.2f, "
                        "\"speedup\": %.4f, \"correct\": %s}%s\n",
                    r.impl_name.c_str(), r.time_ms, r.gflops, r.speedup,
                    r.correct ? "true" : "false",
                    (ii + 1 < ps.impls.size()) ? "," : "");
        }
        fprintf(fp, "      ]\n");
        fprintf(fp, "    }%s\n", (pi + 1 < probs.size()) ? "," : "");
    }
    fprintf(fp, "  ],\n");

    // Geometric mean speedups
    fprintf(fp, "  \"geomean_speedup\": {\n");
    for (size_t i = 0; i < geomeans.size(); i++) {
        fprintf(fp, "    \"%s\": %.4f%s\n",
                geomeans[i].first.c_str(), geomeans[i].second,
                (i + 1 < geomeans.size()) ? "," : "");
    }
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Results written to %s\n", path.c_str());
}

// ============================================================
//  Main
// ============================================================

int main(int argc, char** argv) {
    // Parse arguments
    const char* out_path = "web/results.json";
    bool required_only = false;
    bool skip_correctness = false;
    int warmup = 10, runs = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            out_path = argv[++i];
        else if (strcmp(argv[i], "--required-only") == 0)
            required_only = true;
        else if (strcmp(argv[i], "--no-correctness") == 0)
            skip_correctness = true;
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--output path] [--required-only] "
                   "[--no-correctness] [--warmup N] [--runs N]\n", argv[0]);
            return 0;
        }
    }

    // System info
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname));

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    // Timestamp
    time_t t0 = time(nullptr);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", localtime(&t0));

    printf("=== TSMM Benchmark ===\n");
    printf("Host: %s | Threads: %d | AVX-512: %s | BLAS: %s\n",
           hostname, nthreads,
#ifdef __AVX512F__
           "yes",
#else
           "no",
#endif
#if defined(HAVE_MKL)
           "mkl"
#elif defined(HAVE_OPENBLAS)
           "openblas"
#else
           "none"
#endif
    );
    printf("Warmup: %d  Runs: %d\n\n", warmup, runs);

    srand(42);

    std::vector<BenchResult> all_results;

    for (int pi = 0; pi < N_PROBLEMS; pi++) {
        const Problem& P = PROBLEMS[pi];
        if (required_only && !P.required) continue;

        size_t sA = (size_t)P.k * P.m;
        size_t sB = (size_t)P.k * P.n;
        size_t sC = (size_t)P.m * P.n;
        double mem_gb = (sA + sB + sC) * sizeof(double) / 1e9;

        printf("--- Problem: %s  (m=%d n=%d k=%d)  mem=%.2f GB ---\n",
               P.name, P.m, P.n, P.k, mem_gb);

        double* A  = alloc_mat(P.k, P.m);
        double* B  = alloc_mat(P.k, P.n);
        double* Cref = alloc_mat(P.m, P.n);
        double* Ctmp = alloc_mat(P.m, P.n);

        fill_rand(A, sA);
        fill_rand(B, sB);

        // Adaptive iteration count for very large problems
        int this_warmup = warmup, this_runs = runs;
        if (mem_gb > 1.0) { this_warmup = 3; this_runs = 5; }

        // Reference pass (compute Cref and time)
        double ref_time_ms = 0.0;
        {
            // Warmup
            for (int w = 0; w < this_warmup; w++)
                tsmm_ref(P.m, P.n, P.k, A, B, Cref);
            // Timed
            double t0 = now_sec();
            for (int r = 0; r < this_runs; r++)
                tsmm_ref(P.m, P.n, P.k, A, B, Cref);
            ref_time_ms = (now_sec() - t0) / this_runs * 1e3;

            double gflops = tsmm_flops(P.m, P.n, P.k) / (ref_time_ms * 1e-3) / 1e9;
            printf("  %-14s  %8.3f ms  %7.2f GFLOPS  speedup=1.000\n",
                   "reference", ref_time_ms, gflops);

            BenchResult r0;
            r0.impl_name = "reference"; r0.prob_name = P.name;
            r0.m = P.m; r0.n = P.n; r0.k = P.k; r0.required = P.required;
            r0.time_ms = ref_time_ms; r0.gflops = gflops;
            r0.speedup = 1.0; r0.correct = true;
            all_results.push_back(r0);
        }

        // Other implementations
        for (int ii = 0; ii < N_IMPLS; ii++) {
            if (IMPLS[ii].is_ref) continue;

            // Run implementation
            for (int w = 0; w < this_warmup; w++)
                IMPLS[ii].fn(P.m, P.n, P.k, A, B, Ctmp);
            double t0 = now_sec();
            for (int r = 0; r < this_runs; r++)
                IMPLS[ii].fn(P.m, P.n, P.k, A, B, Ctmp);
            double time_ms = (now_sec() - t0) / this_runs * 1e3;

            bool correct = skip_correctness || check(Cref, Ctmp, sC);
            double gflops = tsmm_flops(P.m, P.n, P.k) / (time_ms * 1e-3) / 1e9;
            double speedup = ref_time_ms / time_ms;

            printf("  %-14s  %8.3f ms  %7.2f GFLOPS  speedup=%.3f  %s\n",
                   IMPLS[ii].name, time_ms, gflops, speedup,
                   correct ? "OK" : "WRONG");

            BenchResult res;
            res.impl_name = IMPLS[ii].name; res.prob_name = P.name;
            res.m = P.m; res.n = P.n; res.k = P.k; res.required = P.required;
            res.time_ms = time_ms; res.gflops = gflops;
            res.speedup = speedup; res.correct = correct;
            all_results.push_back(res);
        }

        free_mat(A); free_mat(B); free_mat(Cref); free_mat(Ctmp);
        printf("\n");

        // Write incremental JSON after each problem so web can show progress
        write_json(out_path, all_results, hostname, nthreads, tbuf);
    }

    // Print geometric mean speedup summary
    printf("=== Geometric Mean Speedup (required problems) ===\n");
    for (int ii = 0; ii < N_IMPLS; ii++) {
        if (IMPLS[ii].is_ref) continue;
        double logsum = 0.0; int cnt = 0;
        for (auto& r : all_results)
            if (r.impl_name == IMPLS[ii].name && r.required && r.correct) {
                logsum += log(r.speedup > 0 ? r.speedup : 1e-9); cnt++;
            }
        if (cnt > 0)
            printf("  %-14s  %.3f x\n", IMPLS[ii].name, exp(logsum / cnt));
    }

    write_json(out_path, all_results, hostname, nthreads, tbuf);
    return 0;
}
