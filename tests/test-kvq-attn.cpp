// two-tier attention op (ggml_kvq_attn): equivalence with decode-then-attend, and a micro-benchmark
//
// usage: test-kvq-attn                             equivalence tests on the CPU and, if present, the GPU
//        test-kvq-attn --bench [--quick] [--tables FILE]  GPU timings; FILE: codec tables (llama_kvq_save) for real bit allocations

#include "../src/llama-kvq-codec.h"
#include "../src/llama-kvq-pack.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define popen  _popen
#define pclose _pclose
#endif

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static const int D = 256;

// step in units of sigma by bits, as in src/llama-kvq-codec.cpp
static const float gauss_step[9] = { 0.0f, 1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f, 0.1041f, 0.0569f, 0.0308f };

// host copy of all inputs of one call, in the layouts of ggml_kvq_attn
struct kvq_problem {
    int G = 4, n_rep = 6, L = 1;
    int n_crow = 8192, n_chunk = 0, n_coarse = 0, n_tail = 1, cap = 0;
    int nwk = 16, nwv = 8;
    ggml_type type_tail = GGML_TYPE_Q8_0;
    ggml_type type_res  = GGML_TYPE_Q8_0;
    float scale = 1.0f/16.0f;
    bool  has_mask = true;

    std::vector<float>       q;    // [L][H][D]
    std::vector<int32_t>     kc;   // [G][n_coarse][nwk]
    std::vector<int32_t>     vc;   // [G][n_coarse][nwv + 1]
    std::vector<ggml_fp16_t> tinv; // [n_chunk][2G][D*D]
    std::vector<float>       taux; // [n_chunk][2G][3*D]
    std::vector<uint8_t>     k, v; // [n_tail][G][row], cache layout
    std::vector<uint8_t>     rkv;  // [2][G][cap][row]
    std::vector<int32_t>     ridx; // [G][cap]
    std::vector<ggml_fp16_t> mask; // [L][n_kv]

    int n_head() const { return G*n_rep; }
    int n_kv()   const { return n_coarse + n_tail; }
};

static size_t row_size(ggml_type type) {
    return ggml_row_size(type, D);
}

static void quantize_row(ggml_type type, const float * x, uint8_t * dst) {
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(x, (ggml_fp16_t *) dst, D);
    } else {
        ggml_get_type_traits(type)->from_float_ref(x, dst, D);
    }
}

static void dequantize_row(ggml_type type, const uint8_t * src, float * y) {
    ggml_get_type_traits(type)->to_float(src, y, D);
}

// one graph per call; reps copies of the op (and of top_k when n_top > 0) for timing
struct kvq_graph {
    ggml_context          * ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    ggml_cgraph           * gf  = nullptr;
    ggml_tensor           * out = nullptr;

    ~kvq_graph() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
};

static ggml_tensor * new_cache_view(ggml_context * ctx, ggml_type type, int n_rows, int G, const char * name) {
    // [D, G, n_rows] as in the KV cache, viewed as [D, n_rows, G]
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, D, G, n_rows);
    ggml_set_name(t, name);
    return ggml_permute(ctx, t, 0, 2, 1, 3);
}

static void set_named(ggml_context * ctx, const char * name, const void * data, size_t size) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t) {
        GGML_ASSERT(ggml_nbytes(t) == size);
        ggml_backend_tensor_set(t, data, 0, size);
    }
}

static kvq_graph * build_kvq(ggml_backend_t backend, const kvq_problem & P, int mode, int reps, int n_top) {
    auto * g = new kvq_graph;
    ggml_init_params ip = { ggml_tensor_overhead()*(64 + 8*reps) + ggml_graph_overhead_custom(8*reps + 64, false), nullptr, true };
    g->ctx = ggml_init(ip);
    ggml_context * ctx = g->ctx;

    const bool attn = mode & GGML_KVQ_ATTN;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, P.n_head(), P.L);
    ggml_set_name(q, "q");

    ggml_tensor * kc = nullptr, * vc = nullptr, * tinv = nullptr, * taux = nullptr;
    if (P.n_coarse > 0) {
        kc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, P.nwk, P.n_coarse, P.G);
        ggml_set_name(kc, "kc");
        if (attn) {
            vc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, P.nwv + 1, P.n_coarse, P.G);
            ggml_set_name(vc, "vc");
        }
        tinv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, D, 2*P.G, P.n_chunk);
        taux = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 3, 2*P.G, P.n_chunk);
        ggml_set_name(tinv, "tinv");
        ggml_set_name(taux, "taux");
    }
    ggml_tensor * k = new_cache_view(ctx, P.type_tail, P.n_tail, P.G, "k");
    ggml_tensor * v = attn ? new_cache_view(ctx, P.type_tail, P.n_tail, P.G, "v") : nullptr;

    ggml_tensor * rkv = nullptr, * ridx = nullptr;
    if (attn && P.cap > 0) {
        rkv  = ggml_new_tensor_4d(ctx, P.type_res, D, P.cap, P.G, 2);
        ridx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, P.cap, P.G);
        ggml_set_name(rkv, "rkv");
        ggml_set_name(ridx, "ridx");
    }
    ggml_tensor * mask = nullptr;
    if (P.has_mask) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, P.n_kv(), P.L);
        ggml_set_name(mask, "mask");
    }

    g->gf = ggml_new_graph_custom(ctx, 8*reps + 64, false);
    for (int r = 0; r < reps; ++r) {
        ggml_tensor * out = ggml_kvq_attn(ctx, q, kc, vc, tinv, taux, k, v, rkv, ridx, mask, P.scale, P.n_crow, mode);
        if (n_top > 0) {
            out = ggml_top_k(ctx, ggml_reshape_2d(ctx, out, P.n_coarse, P.G*P.L), n_top);
        }
        ggml_build_forward_expand(g->gf, out);
        g->out = out;
    }

    g->buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(g->buf);

    set_named(ctx, "q", P.q.data(), P.q.size()*sizeof(float));
    set_named(ctx, "kc", P.kc.data(), P.kc.size()*sizeof(int32_t));
    set_named(ctx, "vc", P.vc.data(), P.vc.size()*sizeof(int32_t));
    set_named(ctx, "tinv", P.tinv.data(), P.tinv.size()*sizeof(ggml_fp16_t));
    set_named(ctx, "taux", P.taux.data(), P.taux.size()*sizeof(float));
    set_named(ctx, "k", P.k.data(), P.k.size());
    set_named(ctx, "v", P.v.data(), P.v.size());
    set_named(ctx, "rkv", P.rkv.data(), P.rkv.size());
    set_named(ctx, "ridx", P.ridx.data(), P.ridx.size()*sizeof(int32_t));
    set_named(ctx, "mask", P.mask.data(), P.mask.size()*sizeof(ggml_fp16_t));

    return g;
}

static std::vector<float> run_kvq(ggml_backend_t backend, const kvq_problem & P, int mode) {
    kvq_graph * g = build_kvq(backend, P, mode, 1, 0);
    std::vector<float> res;
    if (!ggml_backend_supports_op(backend, g->out)) {
        delete g;
        return res;
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, g->gf) == GGML_STATUS_SUCCESS);
    res.resize(ggml_nelements(g->out));
    ggml_backend_tensor_get(g->out, res.data(), 0, res.size()*sizeof(float));
    delete g;
    return res;
}

// decoded coarse rows and precise rows as the op sees them, plain attention in double
struct kvq_reference {
    std::vector<float> kd, vd;   // [G][n_coarse][D] decoded coarse K, V
    std::vector<float> attn;     // [L][H][D]
    std::vector<float> score;    // [L][G][n_coarse]
};

static void reference(const kvq_problem & P, kvq_reference & R) {
    const int H = P.n_head(), n_kv = P.n_kv();
    R.attn.assign((size_t) P.L*H*D, 0.0f);
    R.score.assign((size_t) P.L*P.G*P.n_coarse, 0.0f);

    std::vector<float> kt((size_t) P.G*P.n_tail*D), vt((size_t) P.G*P.n_tail*D);
    for (int r = 0; r < P.n_tail; ++r) {
        for (int g = 0; g < P.G; ++g) {
            dequantize_row(P.type_tail, P.k.data() + ((size_t) r*P.G + g)*row_size(P.type_tail), kt.data() + ((size_t) g*P.n_tail + r)*D);
            dequantize_row(P.type_tail, P.v.data() + ((size_t) r*P.G + g)*row_size(P.type_tail), vt.data() + ((size_t) g*P.n_tail + r)*D);
        }
    }

    auto mask_at = [&](int row, int l) -> double {
        return P.has_mask ? ggml_fp16_to_fp32(P.mask[(size_t) l*n_kv + row]) : 0.0;
    };
    auto dot = [](const float * a, const float * b) {
        double s = 0.0;
        for (int i = 0; i < D; ++i) {
            s += (double) a[i]*b[i];
        }
        return s;
    };

    std::vector<float> kr(D), vr(D);
    for (int l = 0; l < P.L; ++l) {
        for (int h = 0; h < H; ++h) {
            const int g = h / P.n_rep;
            const float * qv = P.q.data() + ((size_t) l*H + h)*D;

            // keys in order: coarse rows, resident slots, tail rows
            std::vector<double> lg;
            std::vector<std::vector<float>> vals;
            std::vector<char> res(P.n_coarse, 0);
            std::vector<double> lc(P.n_coarse);

            for (int s = 0; s < P.cap; ++s) {
                const int32_t r = P.ridx[(size_t) g*P.cap + s];
                if (r >= 0 && r < P.n_coarse) {
                    res[r] = 1;
                }
            }
            for (int j = 0; j < P.n_coarse; ++j) {
                lc[j] = P.scale*dot(qv, R.kd.data() + ((size_t) g*P.n_coarse + j)*D) + mask_at(j, l);
                if (!res[j]) {
                    lg.push_back(lc[j]);
                    vals.emplace_back(R.vd.begin() + ((size_t) g*P.n_coarse + j)*D, R.vd.begin() + ((size_t) g*P.n_coarse + j + 1)*D);
                }
            }
            for (int s = 0; s < P.cap; ++s) {
                const int32_t r = P.ridx[(size_t) g*P.cap + s];
                if (r < 0 || r >= P.n_coarse) {
                    continue;
                }
                const size_t rs = row_size(P.type_res);
                dequantize_row(P.type_res, P.rkv.data() + ((size_t) (0*P.G + g)*P.cap + s)*rs, kr.data());
                dequantize_row(P.type_res, P.rkv.data() + ((size_t) (1*P.G + g)*P.cap + s)*rs, vr.data());
                lg.push_back(P.scale*dot(qv, kr.data()) + mask_at(r, l));
                vals.push_back(vr);
            }
            std::vector<double> lt(P.n_tail);
            for (int r = 0; r < P.n_tail; ++r) {
                lt[r] = P.scale*dot(qv, kt.data() + ((size_t) g*P.n_tail + r)*D) + mask_at(P.n_coarse + r, l);
                lg.push_back(lt[r]);
                vals.emplace_back(vt.begin() + ((size_t) g*P.n_tail + r)*D, vt.begin() + ((size_t) g*P.n_tail + r + 1)*D);
            }

            double mx = -INFINITY;
            for (double x : lg) {
                mx = std::max(mx, x);
            }
            if (mx > -INFINITY) {
                double z = 0.0;
                std::vector<double> acc(D, 0.0);
                for (size_t i = 0; i < lg.size(); ++i) {
                    const double w = std::exp(lg[i] - mx);
                    z += w;
                    for (int d = 0; d < D; ++d) {
                        acc[d] += w*vals[i][d];
                    }
                }
                for (int d = 0; d < D; ++d) {
                    R.attn[((size_t) l*H + h)*D + d] = (float) (acc[d]/z);
                }
            }

            // score: coarse logits of all coarse rows and the tail
            double ms = -INFINITY;
            for (double x : lc) {
                ms = std::max(ms, x);
            }
            for (double x : lt) {
                ms = std::max(ms, x);
            }
            if (ms > -INFINITY) {
                double z = 0.0;
                for (double x : lc) {
                    z += std::exp(x - ms);
                }
                for (double x : lt) {
                    z += std::exp(x - ms);
                }
                for (int j = 0; j < P.n_coarse; ++j) {
                    R.score[((size_t) l*P.G + g)*P.n_coarse + j] += (float) (std::exp(lc[j] - ms)/z);
                }
            }
        }
    }
}

static double nmse(const float * a, const float * b, size_t n) {
    double e = 0.0, s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        e += ((double) a[i] - b[i])*((double) a[i] - b[i]);
        s += (double) b[i]*b[i];
    }
    return s > 0.0 ? e/s : e;
}

// synthetic K/V rows with a wide spread of component scales, so that the allocation has 0-bit and 8-bit components
struct row_source {
    std::vector<float> mu, a, s; // x = mu + (z * s) a
    std::mt19937 rng;

    row_source(uint32_t seed) : rng(seed) {
        std::normal_distribution<float> nd;
        mu.resize(D);
        a.resize((size_t) D*D);
        s.resize(D);
        for (auto & x : mu) {
            x = 0.3f*nd(rng);
        }
        for (auto & x : a) {
            x = nd(rng)/16.0f;
        }
        for (int c = 0; c < D; ++c) {
            s[c] = 10.0f*std::pow(1e-4f, (float) c/(D - 1));
        }
    }

    void rows(int n, float * x) {
        std::normal_distribution<float> nd;
        std::vector<float> z(D);
        for (int r = 0; r < n; ++r) {
            // per-row gain, as V rows have
            const float gr = std::exp(0.3f*nd(rng));
            for (int c = 0; c < D; ++c) {
                z[c] = nd(rng)*s[c]*gr;
            }
            for (int i = 0; i < D; ++i) {
                double v = mu[i];
                for (int c = 0; c < D; ++c) {
                    v += (double) z[c]*a[(size_t) c*D + i];
                }
                x[(size_t) r*D + i] = (float) v;
            }
        }
    }
};

struct eq_case {
    const char * name;
    int G, L, n_crow, n_chunk, n_tail, cap, n_res;
    ggml_type type_tail, type_res;
    bool masked;   // masked coarse rows
    bool boundary; // resident rows at chunk boundaries
    bool no_mask;
};

// coarse rows from fitted codec tables, reference from llama_kvq_decode
static void build_equivalence(const eq_case & ec, kvq_problem & P, kvq_reference & R_f16, kvq_reference & R_f32) {
    P = kvq_problem();
    P.G = ec.G;
    P.L = ec.L;
    P.n_crow = ec.n_crow;
    P.n_chunk = ec.n_chunk;
    P.n_coarse = ec.n_chunk*ec.n_crow;
    P.n_tail = ec.n_tail;
    P.cap = ec.cap;
    P.type_tail = ec.type_tail;
    P.type_res = ec.type_res;
    P.has_mask = !ec.no_mask;
    P.nwk = 16;
    P.nwv = 8;

    std::mt19937 rng(1234 + ec.n_chunk*7 + ec.L*13 + ec.cap);
    std::normal_distribution<float> nd;

    const int H = P.n_head();
    P.q.resize((size_t) P.L*H*D);
    for (auto & x : P.q) {
        x = 1.5f*nd(rng);
    }

    std::vector<float> kx((size_t) P.G*P.n_coarse*D), vx((size_t) P.G*P.n_coarse*D);
    R_f16.kd.resize(kx.size()); R_f16.vd.resize(vx.size());
    R_f32.kd.resize(kx.size()); R_f32.vd.resize(vx.size());
    P.kc.assign((size_t) P.G*P.n_coarse*P.nwk, 0);
    P.vc.assign((size_t) P.G*P.n_coarse*(P.nwv + 1), 0);
    P.tinv.assign((size_t) P.n_chunk*2*P.G*D*D, 0);
    P.taux.assign((size_t) P.n_chunk*2*P.G*3*D, 0.0f);

    std::vector<double> metric((size_t) D*D, 0.0);
    for (int i = 0; i < D; ++i) {
        metric[(size_t) i*D + i] = 1.0;
    }

    std::vector<row_source> srcs;
    for (int g = 0; g < P.G; ++g) {
        srcs.emplace_back(100 + 2*g);
        srcs.emplace_back(101 + 2*g);
    }

    int n_bits0 = 0, n_bits8 = 0;
    for (int t = 0; t < P.n_chunk; ++t) {
        for (int g = 0; g < P.G; ++g) {
            for (int kv = 0; kv < 2; ++kv) {
                float * x = (kv ? vx.data() : kx.data()) + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*D;
                srcs[2*g + kv].rows(P.n_crow, x);

                llama_kvq_table tab;
                GGML_ASSERT(llama_kvq_fit(tab, x, P.n_crow, D, metric.data(), kv ? 1.0f : 2.0f, kv == 1, 8));
                for (int c = 0; c < D; ++c) {
                    n_bits0 += tab.bits[c] == 0;
                    n_bits8 += tab.bits[c] == 8;
                }

                std::vector<int8_t> codes((size_t) P.n_crow*D);
                std::vector<float>  gain(P.n_crow, 1.0f);
                llama_kvq_encode(tab, x, P.n_crow, codes.data(), gain.data(), 8);

                const int nw = kv ? P.nwv : P.nwk;
                std::vector<int32_t> rows((size_t) P.n_crow*(nw + kv));
                llama_kvq_pack_rows(tab, nw, codes.data(), gain.data(), P.n_crow, rows.data());
                int32_t * dstc = kv ? P.vc.data() + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*(nw + 1)
                                    : P.kc.data() + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*nw;
                std::copy(rows.begin(), rows.end(), dstc);

                const int m = 2*g + kv;
                llama_kvq_pack_table(tab, P.tinv.data() + ((size_t) t*2*P.G + m)*D*D, P.taux.data() + ((size_t) t*2*P.G + m)*3*D);

                // the gain the op reads is f16
                for (auto & gv : gain) {
                    gv = ggml_fp16_to_fp32(ggml_fp32_to_fp16(gv));
                }
                float * out32 = (kv ? R_f32.vd.data() : R_f32.kd.data()) + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*D;
                float * out16 = (kv ? R_f16.vd.data() : R_f16.kd.data()) + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*D;
                llama_kvq_decode(tab, codes.data(), gain.data(), P.n_crow, out32, 8);
                for (auto & e : tab.inv) {
                    e = ggml_fp16_to_fp32(ggml_fp32_to_fp16(e));
                }
                llama_kvq_decode(tab, codes.data(), gain.data(), P.n_crow, out16, 8);
            }
        }
    }
    if (P.n_chunk > 0) {
        printf("    tables: %.1f%% 0-bit, %.1f%% 8-bit components\n", 100.0*n_bits0/(P.n_chunk*P.G*2*D), 100.0*n_bits8/(P.n_chunk*P.G*2*D));
    }

    // resident slots: n_res distinct coarse rows at random slots, the others empty
    P.ridx.assign((size_t) P.G*P.cap, -1);
    const size_t rs_res = row_size(P.type_res);
    P.rkv.assign((size_t) 2*P.G*P.cap*rs_res, 0);
    for (int g = 0; g < P.G && P.n_coarse > 0; ++g) {
        std::vector<int32_t> rows(P.n_coarse);
        std::iota(rows.begin(), rows.end(), 0);
        std::shuffle(rows.begin(), rows.end(), rng);
        if (ec.boundary) {
            std::vector<int32_t> front;
            for (int t = 0; t < P.n_chunk; ++t) {
                if (t > 0) {
                    front.push_back(t*P.n_crow - 1);
                }
                front.push_back(t*P.n_crow);
            }
            front.push_back(P.n_coarse - 1);
            for (int32_t r : rows) {
                if (std::find(front.begin(), front.end(), r) == front.end()) {
                    front.push_back(r);
                }
            }
            rows = front;
        }
        std::vector<int32_t> slots(P.cap);
        std::iota(slots.begin(), slots.end(), 0);
        std::shuffle(slots.begin(), slots.end(), rng);
        const int n_res = std::min(ec.n_res, P.n_coarse);
        for (int i = 0; i < n_res; ++i) {
            const int s = slots[i];
            const int r = rows[i];
            P.ridx[(size_t) g*P.cap + s] = r;
            quantize_row(P.type_res, kx.data() + ((size_t) g*P.n_coarse + r)*D, P.rkv.data() + ((size_t) (0*P.G + g)*P.cap + s)*rs_res);
            quantize_row(P.type_res, vx.data() + ((size_t) g*P.n_coarse + r)*D, P.rkv.data() + ((size_t) (1*P.G + g)*P.cap + s)*rs_res);
        }
    }

    const size_t rs_tail = row_size(P.type_tail);
    P.k.assign((size_t) P.n_tail*P.G*rs_tail, 0);
    P.v.assign((size_t) P.n_tail*P.G*rs_tail, 0);
    std::vector<float> xr((size_t) P.n_tail*D);
    for (int g = 0; g < P.G; ++g) {
        for (int kv = 0; kv < 2; ++kv) {
            srcs[2*g + kv].rows(P.n_tail, xr.data());
            for (int r = 0; r < P.n_tail; ++r) {
                quantize_row(P.type_tail, xr.data() + (size_t) r*D, (kv ? P.v : P.k).data() + ((size_t) r*P.G + g)*rs_tail);
            }
        }
    }

    // causal over the tail; masked coarse rows include resident ones
    const int n_kv = P.n_kv();
    P.mask.assign((size_t) P.L*n_kv, ggml_fp32_to_fp16(0.0f));
    for (int l = 0; l < P.L; ++l) {
        for (int r = 0; r < P.n_tail; ++r) {
            if (r > P.n_tail - P.L + l) {
                P.mask[(size_t) l*n_kv + P.n_coarse + r] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        for (int j = 0; j < P.n_coarse && ec.masked; ++j) {
            if (j % 7 == 3 || (P.cap > 0 && j == P.ridx[0])) {
                P.mask[(size_t) l*n_kv + j] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
    }

    reference(P, R_f16);
    reference(P, R_f32);
}

static void test_equivalence(ggml_backend_t cpu, ggml_backend_t gpu) {
    const ggml_type F16 = GGML_TYPE_F16, Q8 = GGML_TYPE_Q8_0;
    const eq_case cases[] = {
        // name                 G  L  C    n_chunk n_tail cap  n_res tail res masked boundary no_mask
        { "no coarse",          4, 1, 512, 0,      37,    0,   0,    Q8,  Q8, false, false, false },
        { "one tail row",       4, 1, 512, 2,      1,     64,  64,   Q8,  Q8, false, false, false },
        { "L1 partial res",     4, 1, 512, 2,      100,   256, 100,  Q8,  Q8, true,  true,  false },
        { "L4 full res f16",    4, 4, 512, 2,      100,   256, 256,  F16, F16, true, true,  false },
        { "L4 empty res",       4, 4, 384, 3,      70,    64,  0,    F16, Q8, false, false, false },
        { "L4 cap 1024 mixed",  2, 4, 640, 2,      300,   1024, 700, Q8,  F16, true, true,  false },
        { "L1 no mask",         2, 1, 512, 1,      65,    64,  64,   F16, Q8, false, false, true  },
    };

    for (const eq_case & ec : cases) {
        kvq_problem P;
        kvq_reference R16, R32;
        printf("  case %-18s G=%d L=%d C=%d n_chunk=%d n_tail=%d cap=%d n_res=%d tail=%s res=%s\n", ec.name, ec.G, ec.L, ec.n_crow,
            ec.n_chunk, ec.n_tail, ec.cap, ec.n_res, ggml_type_name(ec.type_tail), ggml_type_name(ec.type_res));
        build_equivalence(ec, P, R16, R32);

        const size_t na = R16.attn.size();
        const size_t ns = R16.score.size();

        for (ggml_backend_t be : { cpu, gpu }) {
            if (!be) {
                continue;
            }
            const bool is_cpu = be == cpu;
            // CPU: same f16 tables in double, only rounding differs; GPU: f16 qt and f16 weights in the mma
            const double tol_a = is_cpu ? 1e-9 : 2e-5;
            const double tol_s = is_cpu ? 1e-9 : 2e-5;

            std::vector<float> a = run_kvq(be, P, GGML_KVQ_ATTN);
            if (a.empty()) {
                printf("    %-5s not supported\n", ggml_backend_name(be));
                continue;
            }
            const double ea   = nmse(a.data(), R16.attn.data(), na);
            const double ea32 = nmse(a.data(), R32.attn.data(), na);
            printf("    %-5s attn  nmse %.2e (f32 tables: %.2e)", ggml_backend_name(be), ea, ea32);
            CHECK(ea < tol_a, "%s %s attn nmse %e", ec.name, ggml_backend_name(be), ea);

            if (P.n_coarse > 0) {
                std::vector<float> s  = run_kvq(be, P, GGML_KVQ_SCORE);
                std::vector<float> as = run_kvq(be, P, GGML_KVQ_ATTN | GGML_KVQ_SCORE);
                const double es  = nmse(s.data(), R16.score.data(), ns);
                const double eas = std::max(nmse(as.data(), R16.attn.data(), na), nmse(as.data() + na, R16.score.data(), ns));
                printf(", score nmse %.2e, both %.2e", es, eas);
                CHECK(es < tol_s, "%s %s score nmse %e", ec.name, ggml_backend_name(be), es);
                CHECK(eas < std::max(tol_a, tol_s), "%s %s attn+score nmse %e", ec.name, ggml_backend_name(be), eas);
            }
            printf("\n");
        }
    }
}

//
// benchmark
//

// bit allocations per kv head: from tables (V re-allocated at avg_v bits) or from a synthetic fit
static void bench_bits(const std::string & tables, int G, float avg_v, std::vector<std::vector<uint8_t>> & bk, std::vector<std::vector<uint8_t>> & bv) {
    bk.assign(G, {});
    bv.assign(G, {});
    std::vector<llama_kvq_table> tabs;
    if (!tables.empty() && llama_kvq_load(tables, tabs)) {
        for (const auto & t : tabs) {
            if (t.il != tabs[0].il || t.head >= G) {
                continue;
            }
            if (!t.is_v) {
                bk[t.head] = t.bits;
            } else {
                std::vector<double> var(D);
                for (int c = 0; c < D; ++c) {
                    var[c] = (double) t.sigma[c]*t.sigma[c];
                }
                bv[t.head] = llama_kvq_alloc_bits(var, avg_v);
            }
        }
        printf("bits from %s, layer %d, V at %.1f bits\n", tables.c_str(), tabs[0].il, avg_v);
    }
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
        x = ggml_fp32_to_fp16(nd(rng)/16.0f);
    }
    P.taux.resize((size_t) P.n_chunk*2*P.G*3*D);
    for (int t = 0; t < P.n_chunk; ++t) {
        for (int m = 0; m < 2*P.G; ++m) {
            float * aux = P.taux.data() + ((size_t) t*2*P.G + m)*3*D;
            const std::vector<uint8_t> & b = m % 2 ? bv[m/2] : bk[m/2];
            for (int c = 0; c < D; ++c) {
                aux[c] = b[c] ? gauss_step[b[c]]*std::exp(-0.02f*c) : 0.0f;
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

// the GPU is shared: wait until nvidia-smi reports it idle for 2 samples (at most 120 s)
static void wait_idle() {
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

// minimum over 7 runs, after a wait for an idle GPU
static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps) {
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    wait_idle();
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

static void bench(ggml_backend_t gpu, const std::string & tables, bool quick) {
    std::vector<std::vector<uint8_t>> bk, bv;
    bench_bits(tables, 4, 1.0f, bk, bv);

    const int reps = 20;
    std::vector<int> n_kvs = { 25000, 50000, 100000, 150000 };
    std::vector<int> caps  = { 1024, 4096 };
    if (quick) {
        n_kvs = { 150000 };
        caps  = { 1024 };
    }

    printf("\nms per call; %d calls per graph, minimum of 7 runs after an idle GPU\n", reps);
    printf("| n_kv | L | cap | score | score+top1024 | top1024 | attn | attn+score | FA q8_0 | FA f16 |\n");
    printf("|---|---|---|---|---|---|---|---|---|---|\n");
    for (int n_kv : n_kvs) {
        for (int L : { 1, 4 }) {
            const double fa_q8  = time_fa(gpu, n_kv, L, GGML_TYPE_Q8_0, reps);
            const double fa_f16 = time_fa(gpu, n_kv, L, GGML_TYPE_F16, reps);
            for (int cap : caps) {
                kvq_problem P;
                build_bench(P, n_kv, L, cap, GGML_TYPE_Q8_0, bk, bv);
                const double ts  = time_kvq(gpu, P, GGML_KVQ_SCORE, 0, reps);
                const double tst = time_kvq(gpu, P, GGML_KVQ_SCORE, 1024, reps);
                const double ttk = time_top_k(gpu, P.n_coarse, P.G*P.L, 1024, reps);
                const double ta  = time_kvq(gpu, P, GGML_KVQ_ATTN, 0, reps);
                const double tas = time_kvq(gpu, P, GGML_KVQ_ATTN | GGML_KVQ_SCORE, 0, reps);
                printf("| %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n", n_kv, L, cap, ts, tst, ttk, ta, tas, fa_q8, fa_f16);
                fflush(stdout);
            }
        }
    }

    // VRAM at 150k rows, per attention layer
    {
        kvq_problem P;
        build_bench(P, 150000, 1, 1024, GGML_TYPE_Q8_0, bk, bv);
        const double codes  = (double) P.G*P.n_coarse*(P.nwk + P.nwv + 1)*4;
        const double tabs   = (double) 2*P.G*(D*D*2 + 3*D*4);
        printf("\nVRAM per layer at n_kv = 150000 (n_coarse %d, %d chunks): codes %.1f MB, tables %.2f MB per chunk, %.1f MB total\n",
            P.n_coarse, P.n_chunk, codes/1e6, tabs/1e6, (codes + P.n_chunk*tabs)/1e6);
    }
}

// fixed workload for a kernel profiler: attention and score at 150k rows, L = 1 and 4
static void profile(ggml_backend_t gpu, const std::string & tables) {
    std::vector<std::vector<uint8_t>> bk, bv;
    bench_bits(tables, 4, 1.0f, bk, bv);
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

int main(int argc, char ** argv) {
    bool do_bench = false;
    bool do_prof  = false;
    bool quick    = false;
    std::string tables;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) {
            do_bench = true;
        } else if (strcmp(argv[i], "--profile") == 0) {
            do_prof = true;
        } else if (strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (strcmp(argv[i], "--tables") == 0 && i + 1 < argc) {
            tables = argv[++i];
        }
    }

    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    GGML_ASSERT(cpu);
    printf("backends: %s%s%s\n", ggml_backend_name(cpu), gpu ? ", " : "", gpu ? ggml_backend_name(gpu) : "");

    if (do_bench || do_prof) {
        GGML_ASSERT(gpu && "the benchmark needs a GPU");
        if (do_prof) {
            profile(gpu, tables);
        } else {
            bench(gpu, tables, quick);
        }
    } else {
        printf("equivalence with decode-then-attend (llama_kvq_decode with the f16 inverse of the op; f32 for information)\n");
        test_equivalence(cpu, gpu);
    }

    if (gpu) {
        ggml_backend_free(gpu);
    }
    ggml_backend_free(cpu);

    if (n_fail > 0) {
        printf("%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
