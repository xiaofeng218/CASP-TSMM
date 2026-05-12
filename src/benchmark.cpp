#include "tsmm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

static const Problem PROBLEMS[] = {
    {4000, 16000, 128, "4000x16000x128", true},
    {8, 16, 16000, "8x16x16000", true},
    {32, 16000, 16, "32x16000x16", true},
    {144, 144, 144, "144x144x144", true},
    {16, 12344, 16, "16x12344x16", false},
    {4, 64, 606841, "4x64x606841", false},
    {442, 193, 11, "442x193x11", false},
    {40, 1127228, 40, "40x1127228x40", false},
};

static ImplDesc IMPLS[] = {
    {"reference", tsmm_reference, true},
    {"naive", tsmm_naive, false},
    {"openmp", tsmm_openmp, false},
    {"blocked", tsmm_blocked, false},
    {"avx512", tsmm_avx512, false},
    {"avx512_omp", tsmm_avx512_omp, false},
    {"opt", tsmm_opt, false},
};

struct BenchResult {
    std::string impl_name;
    std::string prob_name;
    int m;
    int n;
    int k;
    bool required;
    double time_ms;
    double gflops;
    double speedup;
    bool correct;
};

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static double* alloc_mat(std::size_t rows, std::size_t cols) {
    const std::size_t bytes = rows * cols * sizeof(double);
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, 64);
    if (!p) {
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) {
#endif
        std::fprintf(stderr, "failed to allocate %.2f GiB\n", bytes / 1073741824.0);
        std::exit(1);
    }
    return static_cast<double*>(p);
}

static void free_mat(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

static void fill_rand(double* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;
    }
}

static bool check_result(const double* ref, const double* got, std::size_t n, double rtol = 1e-8) {
    double max_err = 0.0;
    double max_ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(ref[i] - got[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    return max_ref < 1e-30 ? max_err < 1e-12 : (max_err / max_ref) < rtol;
}

static double tsmm_flops(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * n * k;
}

static const char* layout_name(Layout layout) {
    return layout == Layout::RowMajor ? "row-major" : "col-major";
}

static void write_json(const char* path,
                       const std::vector<BenchResult>& results,
                       const char* hostname,
                       int nthreads,
                       const char* timestamp,
                       Layout layout) {
    FILE* fp = std::fopen(path, "w");
    if (!fp) {
        std::perror(path);
        return;
    }

    std::fprintf(fp, "{\n");
    std::fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp);
    std::fprintf(fp, "  \"hostname\": \"%s\",\n", hostname);
    std::fprintf(fp, "  \"n_threads\": %d,\n", nthreads);
    std::fprintf(fp, "  \"layout\": \"%s\",\n", layout_name(layout));
    std::fprintf(fp, "  \"avx512\": %s,\n",
#ifdef __AVX512F__
                 "true"
#else
                 "false"
#endif
    );
    std::fprintf(fp, "  \"blas\": \"%s\",\n",
#if defined(HAVE_MKL)
                 "mkl"
#elif defined(HAVE_OPENBLAS)
                 "openblas"
#else
                 "none"
#endif
    );

    std::fprintf(fp, "  \"problems\": [\n");
    bool first_problem = true;
    for (const Problem& p : PROBLEMS) {
        std::vector<BenchResult> rows;
        for (const BenchResult& r : results) {
            if (r.prob_name == p.name) rows.push_back(r);
        }
        if (rows.empty()) continue;
        if (!first_problem) std::fprintf(fp, ",\n");
        first_problem = false;
        std::fprintf(fp, "    {\n");
        std::fprintf(fp, "      \"name\": \"%s\", \"m\": %d, \"n\": %d, \"k\": %d, \"required\": %s,\n",
                     p.name, p.m, p.n, p.k, p.required ? "true" : "false");
        std::fprintf(fp, "      \"impls\": [\n");
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const BenchResult& r = rows[i];
            std::fprintf(fp,
                         "        {\"name\": \"%s\", \"time_ms\": %.6f, \"gflops\": %.6f, \"speedup\": %.6f, \"correct\": %s}%s\n",
                         r.impl_name.c_str(), r.time_ms, r.gflops, r.speedup,
                         r.correct ? "true" : "false",
                         i + 1 < rows.size() ? "," : "");
        }
        std::fprintf(fp, "      ]\n");
        std::fprintf(fp, "    }");
    }
    std::fprintf(fp, "\n  ],\n");

    std::fprintf(fp, "  \"geomean_speedup\": {\n");
    bool first_impl = true;
    for (const ImplDesc& impl : IMPLS) {
        if (impl.is_ref) continue;
        double logsum = 0.0;
        int count = 0;
        for (const BenchResult& r : results) {
            if (r.impl_name == impl.name && r.required && r.correct) {
                logsum += std::log(r.speedup > 0 ? r.speedup : 1e-12);
                ++count;
            }
        }
        if (count == 0) continue;
        if (!first_impl) std::fprintf(fp, ",\n");
        first_impl = false;
        std::fprintf(fp, "    \"%s\": %.6f", impl.name, std::exp(logsum / count));
    }
    std::fprintf(fp, "\n  }\n");
    std::fprintf(fp, "}\n");
    std::fclose(fp);
}

static void usage(const char* argv0) {
    std::printf("Usage: %s [--output PATH] [--required-only] [--all] [--layout row|col] [--warmup N] [--runs N] [--no-correctness]\n", argv0);
}

int main(int argc, char** argv) {
    const char* out_path = "web/results.json";
    bool required_only = false;
    bool skip_correctness = false;
    int warmup = 10;
    int runs = 20;
    Layout layout = Layout::RowMajor;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--required-only") == 0) {
            required_only = true;
        } else if (std::strcmp(argv[i], "--all") == 0) {
            required_only = false;
        } else if (std::strcmp(argv[i], "--no-correctness") == 0) {
            skip_correctness = true;
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--layout") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            if (std::strcmp(v, "row") == 0 || std::strcmp(v, "row-major") == 0) layout = Layout::RowMajor;
            else if (std::strcmp(v, "col") == 0 || std::strcmp(v, "col-major") == 0) layout = Layout::ColMajor;
            else {
                usage(argv[0]);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    char hostname[256] = "unknown";
#ifndef _WIN32
    gethostname(hostname, sizeof(hostname));
#endif

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    const std::time_t tt = std::time(nullptr);
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", std::localtime(&tt));

    std::printf("=== TSMM Benchmark ===\n");
    std::printf("Host: %s | Threads: %d | Layout: %s | AVX-512: %s | BLAS: %s\n",
                hostname, nthreads, layout_name(layout),
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
    std::printf("Warmup: %d | Runs: %d\n\n", warmup, runs);

    std::srand(42);
    std::vector<BenchResult> all_results;

    for (const Problem& p : PROBLEMS) {
        if (required_only && !p.required) continue;

        const std::size_t sA = static_cast<std::size_t>(p.k) * p.m;
        const std::size_t sB = static_cast<std::size_t>(p.k) * p.n;
        const std::size_t sC = static_cast<std::size_t>(p.m) * p.n;
        const double mem_gb = (sA + sB + sC) * sizeof(double) / 1e9;
        int this_warmup = warmup;
        int this_runs = runs;
        if (mem_gb > 1.0) {
            this_warmup = std::min(this_warmup, 3);
            this_runs = std::min(this_runs, 5);
        }

        std::printf("--- %s (m=%d n=%d k=%d, %.2f GB) ---\n", p.name, p.m, p.n, p.k, mem_gb);
        double* A = alloc_mat(p.k, p.m);
        double* B = alloc_mat(p.k, p.n);
        double* Cref = alloc_mat(p.m, p.n);
        double* Ctmp = alloc_mat(p.m, p.n);

        fill_rand(A, sA);
        fill_rand(B, sB);

        double ref_ms = 0.0;
        for (int w = 0; w < this_warmup; ++w) tsmm_reference(p.m, p.n, p.k, A, B, Cref, layout);
        {
            const double t0 = now_sec();
            for (int r = 0; r < this_runs; ++r) tsmm_reference(p.m, p.n, p.k, A, B, Cref, layout);
            ref_ms = (now_sec() - t0) * 1e3 / this_runs;
        }
        const double ref_gflops = tsmm_flops(p.m, p.n, p.k) / (ref_ms * 1e-3) / 1e9;
        std::printf("  %-12s %10.3f ms %10.2f GFLOPS speedup=1.000 OK\n", "reference", ref_ms, ref_gflops);
        all_results.push_back({"reference", p.name, p.m, p.n, p.k, p.required, ref_ms, ref_gflops, 1.0, true});

        for (const ImplDesc& impl : IMPLS) {
            if (impl.is_ref) continue;
            for (int w = 0; w < this_warmup; ++w) impl.fn(p.m, p.n, p.k, A, B, Ctmp, layout);
            const double t0 = now_sec();
            for (int r = 0; r < this_runs; ++r) impl.fn(p.m, p.n, p.k, A, B, Ctmp, layout);
            const double ms = (now_sec() - t0) * 1e3 / this_runs;
            const bool ok = skip_correctness || check_result(Cref, Ctmp, sC);
            const double gflops = tsmm_flops(p.m, p.n, p.k) / (ms * 1e-3) / 1e9;
            const double speedup = ref_ms / ms;
            std::printf("  %-12s %10.3f ms %10.2f GFLOPS speedup=%5.3f %s\n",
                        impl.name, ms, gflops, speedup, ok ? "OK" : "WRONG");
            all_results.push_back({impl.name, p.name, p.m, p.n, p.k, p.required, ms, gflops, speedup, ok});
        }

        free_mat(A);
        free_mat(B);
        free_mat(Cref);
        free_mat(Ctmp);
        write_json(out_path, all_results, hostname, nthreads, timestamp, layout);
        std::printf("\n");
    }

    std::printf("=== Geometric mean speedup (required problems) ===\n");
    for (const ImplDesc& impl : IMPLS) {
        if (impl.is_ref) continue;
        double logsum = 0.0;
        int count = 0;
        for (const BenchResult& r : all_results) {
            if (r.impl_name == impl.name && r.required && r.correct) {
                logsum += std::log(r.speedup > 0 ? r.speedup : 1e-12);
                ++count;
            }
        }
        if (count > 0) std::printf("  %-12s %.3fx\n", impl.name, std::exp(logsum / count));
    }
    write_json(out_path, all_results, hostname, nthreads, timestamp, layout);
    return 0;
}
