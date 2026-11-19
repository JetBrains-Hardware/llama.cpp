// micro-benchmark of the two-tier attention op (ggml_kvq_attn) on the GPU, synthetic problems
//
// usage: llama-kvq-bench [--quick] [--bench-l 1,4] [--bench-nkv N [--bench-cap C]] [--no-fa] [--no-idle-wait]
//                          GPU timings: positions per call, one row count and resident capacity, no flash attention timings,
//                          no wait for an idle GPU (it asks nvidia-smi)
//        llama-kvq-bench --profile   a fixed workload for a kernel profiler
//        llama-kvq-bench --hash      hashes of the GPU outputs of large problems (compare two builds bit for bit)

#include "../../tests/kvq-attn-problem.h"
#include "../../src/llama-kvq-codec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <thread>

#ifdef _WIN32
#define popen  _popen
#define pclose _pclose
#endif

//
// benchmark
//

// bit allocations per kv head from a synthetic fit
static void bench_bits(int G, float avg_v, std::vector<std::vector<uint8_t>> & bk, std::vector<std::vector<uint8_t>> & bv) {
    bk.assign(G, {});
    bv.assign(G, {});
    for (int g = 0; g < G; ++g) {
        if (bk[g].empty() || bv[g].empty()) {
            std::vector<double> var(D);
            for (int c = 0; c < D; ++c) {
                var[c] = std::pow(1e-4, 2.0*c/(D - 1));
            }
            bk[g] = llama_kvq_alloc_bits(var, 2.0f);
            bv[g] = llama_kvq_alloc_bits(var, avg_v);
        }
        std::sort(bk[g].begin(), bk[g].end(), std::greater<>());
        std::sort(bv[g].begin(), bv[g].end(), std::greater<>());
        const int nk = (int) std::count_if(bk[g].begin(), bk[g].end(), [](uint8_t b) { return b > 0; });
        const int nv = (int) std::count_if(bv[g].begin(), bv[g].end(), [](uint8_t b) { return b > 0; });
        printf("  kv head %d: stored components K %d, V %d\n", g, nk, nv);
    }
}

static void build_bench(kvq_problem & P, int n_kv, int L, int cap, ggml_type type_prec,
        const std::vector<std::vector<uint8_t>> & bk, const std::vector<std::vector<uint8_t>> & bv) {
    P = kvq_problem();
    P.L = L;
    P.n_crow = 8192;
    P.n_chunk = (n_kv - 512)/P.n_crow;
    P.n_coarse = P.n_chunk*P.n_crow;
    P.n_tail = n_kv - P.n_coarse;
    P.cap = cap;
    P.type_tail = type_prec;
    P.type_res = type_prec;

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> ud;
    std::normal_distribution<float> nd;

    const int H = P.n_head();
    P.q.resize((size_t) P.L*H*D);
    for (auto & x : P.q) {
        x = nd(rng);
    }
    P.kc.resize((size_t) P.G*P.n_coarse*P.nwk);
    for (auto & w : P.kc) {
        w = (int32_t) ud(rng);
    }
    P.vc.resize((size_t) P.G*P.n_coarse*(P.nwv + 1));
    for (size_t i = 0; i < P.vc.size(); ++i) {
        P.vc[i] = (int32_t) (i % (P.nwv + 1) == (size_t) P.nwv ? ggml_fp32_to_fp16(1.0f + 0.1f*nd(rng)) : ud(rng));
    }
    P.tinv.resize((size_t) P.n_chunk*2*P.G*D*D);
    for (auto & x : P.tinv) {
        x = (int8_t) std::clamp(std::lround(32.0f*nd(rng)), -127L, 127L);
    }
    P.taux.resize((size_t) P.n_chunk*2*P.G*3*D);
    for (int t = 0; t < P.n_chunk; ++t) {
        for (int m = 0; m < 2*P.G; ++m) {
            float * aux = P.taux.data() + ((size_t) t*2*P.G + m)*3*D;
            const std::vector<uint8_t> & b = m % 2 ? bv[m/2] : bk[m/2];
            for (int c = 0; c < D; ++c) {
                aux[c] = b[c] ? gauss_step[b[c]]*std::exp(-0.02f*c)/512.0f : 0.0f; // 1/512: the table scale
                aux[D + c] = 0.1f*nd(rng);
                aux[2*D + c] = b[c];
            }
        }
    }
    const size_t rs = row_size(type_prec);
    std::vector<float> x(D);
    auto fill_rows = [&](std::vector<uint8_t> & dst, size_t n) {
        dst.resize(n*rs);
        for (size_t r = 0; r < n; ++r) {
            for (auto & e : x) {
                e = nd(rng);
            }
            quantize_row(type_prec, x.data(), dst.data() + r*rs);
        }
    };
    fill_rows(P.k, (size_t) P.n_tail*P.G);
    fill_rows(P.v, (size_t) P.n_tail*P.G);
    fill_rows(P.rkv, (size_t) 2*P.G*P.cap);
    P.ridx.resize((size_t) P.G*P.cap);
    for (int g = 0; g < P.G; ++g) {
        std::vector<int32_t> rows(P.n_coarse);
        std::iota(rows.begin(), rows.end(), 0);
        std::shuffle(rows.begin(), rows.end(), rng);
        std::copy(rows.begin(), rows.begin() + P.cap, P.ridx.begin() + (size_t) g*P.cap);
    }
    P.mask.assign((size_t) P.L*n_kv, ggml_fp32_to_fp16(0.0f));
    for (int l = 0; l < P.L; ++l) {
        for (int r = P.n_tail - P.L + l + 1; r < P.n_tail; ++r) {
            P.mask[(size_t) l*n_kv + P.n_coarse + r] = ggml_fp32_to_fp16(-INFINITY);
        }
    }
}

static bool no_idle_wait = false;

// the GPU is shared: wait until nvidia-smi reports it idle for 2 samples (at most 120 s)
// --no-idle-wait: skip it (the caller holds the GPU; under nsys every nvidia-smi call is slow)
static void wait_idle() {
    if (no_idle_wait) {
        return;
    }
    int n_idle = 0;
    for (int i = 0; i < 120 && n_idle < 2; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        FILE * f = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", "r");
        int util = 100;
        if (f) {
            if (fscanf(f, "%d", &util) != 1) {
                util = 100;
            }
            pclose(f);
        }
        n_idle = util < 10 ? n_idle + 1 : 0;
    }
}

// minimum over 7 runs, after a wait for an idle GPU and a warmup
static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps) {
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    wait_idle();
    // 300 ms of work first, so that the clocks are up
    for (auto t0 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count() < 300.0; ) {
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }
    std::vector<double> ts;
    for (int i = 0; i < 7; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count()/reps);
    }
    return *std::min_element(ts.begin(), ts.end());
}

static double time_kvq(ggml_backend_t backend, const kvq_problem & P, int mode, int n_top, int reps) {
    kvq_graph * g = build_kvq(backend, P, mode, reps, n_top);
    const double t = time_graph(backend, g->gf, reps);
    delete g;
    return t;
}

static double time_top_k(ggml_backend_t backend, int n_coarse, int n_rows, int n_top, int reps) {
    ggml_init_params ip = { ggml_tensor_overhead()*(16 + 8*reps) + ggml_graph_overhead_custom(8*reps + 64, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_coarse, n_rows);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8*reps + 64, false);
    for (int r = 0; r < reps; ++r) {
        ggml_build_forward_expand(gf, ggml_top_k(ctx, s, n_top));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<float> data((size_t) n_coarse*n_rows);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(0.0f, 1e-4f);
    for (auto & x : data) {
        x = ud(rng);
    }
    ggml_backend_tensor_set(s, data.data(), 0, data.size()*sizeof(float));
    const double t = time_graph(backend, gf, reps);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return t;
}

// llama.cpp pads n_kv to 256, which the vec kernels need
static double time_fa(ggml_backend_t backend, int n_kv_real, int L, ggml_type type, int reps) {
    const int G = 4, H = 24;
    const int n_kv = GGML_PAD(n_kv_real, 256);
    ggml_init_params ip = { ggml_tensor_overhead()*(16 + 8*reps) + ggml_graph_overhead_custom(8*reps + 64, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, L);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, type, D, G, n_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, type, D, G, n_kv);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, L);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8*reps + 64, false);
    for (int r = 0; r < reps; ++r) {
        ggml_tensor * out = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
            ggml_permute(ctx, v, 0, 2, 1, 3), m, 1.0f/16.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        ggml_build_forward_expand(gf, out);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    for (ggml_tensor * t : { q, k, v }) {
        std::vector<float> x(ggml_nelements(t));
        for (auto & e : x) {
            e = nd(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, x.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<uint8_t> qd(ggml_nbytes(t));
            for (int64_t r = 0; r < ggml_nelements(t)/D; ++r) {
                quantize_row(type, x.data() + r*D, qd.data() + r*row_size(type));
            }
            ggml_backend_tensor_set(t, qd.data(), 0, qd.size());
        }
    }
    std::vector<ggml_fp16_t> md((size_t) n_kv*L, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(m, md.data(), 0, md.size()*sizeof(ggml_fp16_t));
    const double t = time_graph(backend, gf, reps);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return t;
}

static void bench(ggml_backend_t gpu, bool quick, const std::vector<int> & Ls, int nkv, int cap, bool no_fa) {
    std::vector<std::vector<uint8_t>> bk, bv;
    bench_bits(4, 1.0f, bk, bv);

    const int reps = 20;
    std::vector<int> n_kvs = { 25000, 50000, 100000, 150000 };
    std::vector<int> caps  = { 1024, 4096 };
    if (quick) {
        n_kvs = { 150000 };
        caps  = { 1024 };
    }
    if (nkv > 0) {
        n_kvs = { nkv };
        caps  = { cap > 0 ? cap : 1024 };
    }

    printf("\nms per call; %d calls per graph, minimum of 7 runs after an idle GPU\n", reps);
    printf("| n_kv | L | cap | score | score+top1024 | top1024 | attn | FA q8_0 | FA f16 |\n");
    printf("|---|---|---|---|---|---|---|---|---|\n");
    for (int n_kv : n_kvs) {
        for (int L : Ls) {
            const double fa_q8  = no_fa ? 0.0 : time_fa(gpu, n_kv, L, GGML_TYPE_Q8_0, reps);
            const double fa_f16 = no_fa ? 0.0 : time_fa(gpu, n_kv, L, GGML_TYPE_F16, reps);
            for (int cap : caps) {
                kvq_problem P;
                build_bench(P, n_kv, L, cap, GGML_TYPE_Q8_0, bk, bv);
                const double ts  = time_kvq(gpu, P, GGML_KVQ_SCORE, 0, reps);
                const double tst = time_kvq(gpu, P, GGML_KVQ_SCORE, 1024, reps);
                const double ttk = time_top_k(gpu, P.n_coarse, P.G*P.L, 1024, reps);
                const double ta  = time_kvq(gpu, P, GGML_KVQ_ATTN, 0, reps);
                printf("| %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n", n_kv, L, cap, ts, tst, ttk, ta, fa_q8, fa_f16);
                fflush(stdout);
            }
        }
    }

    // VRAM at 150k rows, per attention layer
    {
        kvq_problem P;
        build_bench(P, 150000, 1, 1024, GGML_TYPE_Q8_0, bk, bv);
        const double codes  = (double) P.G*P.n_coarse*(P.nwk + P.nwv + 1)*4;
        const double tabs   = (double) 2*P.G*(D*D + 3*D*4);
        printf("\nVRAM per layer at n_kv = 150000 (n_coarse %d, %d chunks): codes %.1f MB, tables %.2f MB per chunk, %.1f MB total\n",
            P.n_coarse, P.n_chunk, codes/1e6, tabs/1e6, (codes + P.n_chunk*tabs)/1e6);
    }
}

// fixed workload for a kernel profiler: attention and score at 150k rows, L = 1 and 4
static void profile(ggml_backend_t gpu) {
    std::vector<std::vector<uint8_t>> bk, bv;
    bench_bits(4, 1.0f, bk, bv);
    for (int L : { 1, 4 }) {
        kvq_problem P;
        build_bench(P, 150000, L, 1024, GGML_TYPE_Q8_0, bk, bv);
        for (int mode : { GGML_KVQ_ATTN, GGML_KVQ_SCORE }) {
            kvq_graph * g = build_kvq(gpu, P, mode, 20, 0);
            for (int i = 0; i < 10; ++i) {
                ggml_backend_graph_compute(gpu, g->gf);
            }
            ggml_backend_synchronize(gpu);
            delete g;
        }
    }
}

// FNV-1a of the output bits of every mode, synthetic problems as in the benchmark
static void hash_outputs(ggml_backend_t gpu) {
    std::vector<std::vector<uint8_t>> bk, bv;
    bench_bits(4, 1.0f, bk, bv);
    for (int n_kv : { 20000, 100000, -100000 }) {
        for (int L = 1; L <= 8; ++L) {
            // -100000: 100000 rows with 8192 resident slots
            const int cap = n_kv < 0 ? 8192 : 1024;
            kvq_problem P;
            build_bench(P, std::abs(n_kv), L, cap, GGML_TYPE_Q8_0, bk, bv);
            printf("  n_kv %6d cap %4d L %d:", std::abs(n_kv), cap, L);
            const int modes[] = { GGML_KVQ_ATTN, GGML_KVQ_SCORE };
            for (int mode : modes) {
                const std::vector<float> r = run_kvq(gpu, P, mode);
                uint64_t h = 1469598103934665603ull;
                const uint8_t * b = (const uint8_t *) r.data();
                for (size_t i = 0; i < r.size()*sizeof(float); ++i) {
                    h = (h ^ b[i])*1099511628211ull;
                }
                printf(" %016llx", (unsigned long long) h);
            }
            printf("\n");
            fflush(stdout);
        }
    }
}

int main(int argc, char ** argv) {
    bool do_prof  = false;
    bool do_hash  = false;
    bool quick    = false;
    bool no_fa    = false;
    int  nkv      = 0;
    int  cap      = 0;
    std::vector<int> Ls = { 1, 4 };
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--profile") == 0) {
            do_prof = true;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (strcmp(argv[i], "--hash") == 0) {
            do_hash = true;
        } else if (strcmp(argv[i], "--no-fa") == 0) {
            no_fa = true;
        } else if (strcmp(argv[i], "--no-idle-wait") == 0) {
            no_idle_wait = true;
        } else if (strcmp(argv[i], "--bench-nkv") == 0 && i + 1 < argc) {
            nkv = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-cap") == 0 && i + 1 < argc) {
            cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-l") == 0 && i + 1 < argc) {
            Ls.clear();
            for (const char * p = argv[++i]; *p; ) {
                Ls.push_back(atoi(p));
                while (*p && *p != ',') {
                    ++p;
                }
                if (*p) {
                    ++p;
                }
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) {
        fprintf(stderr, "llama-kvq-bench needs a GPU\n");
        return 1;
    }
    printf("backend: %s\n", ggml_backend_name(gpu));

    if (do_prof) {
        profile(gpu);
    } else if (do_hash) {
        hash_outputs(gpu);
    } else {
        bench(gpu, quick, Ls, nkv, cap, no_fa);
    }

    ggml_backend_free(gpu);
    return 0;
}
