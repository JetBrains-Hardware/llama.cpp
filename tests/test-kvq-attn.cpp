// two-tier attention op (ggml_kvq_attn): equivalence with decode-then-attend on the CPU and, if present, the GPU
// (the micro-benchmark is tools/kvq-eval/kvq-bench.cpp)

#include "kvq-ref.h"
#include "kvq-attn-problem.h"
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

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

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
    int  salt = 0; // other data for the streams of test_streams
};

// coarse rows from fitted codec tables, reference from kvq_ref_decode
static void build_equivalence(const eq_case & ec, kvq_problem & P, kvq_reference & R_i8, kvq_reference & R_f32) {
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

    std::mt19937 rng(1234 + ec.n_chunk*7 + ec.L*13 + ec.cap + 1000*ec.salt);
    std::normal_distribution<float> nd;

    const int H = P.n_head();
    P.q.resize((size_t) P.L*H*D);
    for (auto & x : P.q) {
        x = 1.5f*nd(rng);
    }

    std::vector<float> kx((size_t) P.G*P.n_coarse*D), vx((size_t) P.G*P.n_coarse*D);
    R_i8.kd.resize(kx.size());  R_i8.vd.resize(vx.size());
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
        srcs.emplace_back(100 + 2*g + 50*ec.salt);
        srcs.emplace_back(101 + 2*g + 50*ec.salt);
    }

    int n_bits0 = 0, n_bits8 = 0;
    for (int t = 0; t < P.n_chunk; ++t) {
        for (int g = 0; g < P.G; ++g) {
            for (int kv = 0; kv < 2; ++kv) {
                float * x = (kv ? vx.data() : kx.data()) + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*D;
                srcs[2*g + kv].rows(P.n_crow, x);

                llama_kvq_table tab;
                GGML_ASSERT(kvq_ref_fit(tab, x, P.n_crow, D, metric.data(), kv ? 1.0f : 2.0f, kv == 1, 8));
                for (int c = 0; c < D; ++c) {
                    n_bits0 += tab.bits[c] == 0;
                    n_bits8 += tab.bits[c] == 8;
                }

                std::vector<int8_t> codes((size_t) P.n_crow*D);
                std::vector<float>  gain(P.n_crow, 1.0f);
                kvq_ref_encode(tab, x, P.n_crow, codes.data(), gain.data(), 8);

                const int nw = kv ? P.nwv : P.nwk;
                std::vector<int32_t> rows((size_t) P.n_crow*(nw + kv));
                kvq_ref_pack_rows(tab, nw, codes.data(), gain.data(), P.n_crow, rows.data());
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
                float * out8  = (kv ? R_i8.vd.data() : R_i8.kd.data()) + ((size_t) g*P.n_coarse + (size_t) t*P.n_crow)*D;
                kvq_ref_decode(tab, codes.data(), gain.data(), P.n_crow, out32, 8);
                // the op reads inv[c][:] as int8 with one scale per component
                for (int c = 0; c < D; ++c) {
                    float * row = tab.inv.data() + (size_t) c*D;
                    float   mx  = 0.0f;
                    for (int i = 0; i < D; ++i) {
                        mx = std::max(mx, std::fabs(row[i]));
                    }
                    for (int i = 0; i < D; ++i) {
                        row[i] = mx > 0.0f ? std::nearbyint(row[i]/(mx/127.0f))*(mx/127.0f) : 0.0f;
                    }
                }
                kvq_ref_decode(tab, codes.data(), gain.data(), P.n_crow, out8, 8);
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

    reference(P, R_i8);
    reference(P, R_f32);
}

static void test_equivalence(ggml_backend_t cpu, ggml_backend_t gpu) {
    const ggml_type Q8 = GGML_TYPE_Q8_0;
    const eq_case cases[] = {
        // name                 G  L  C    n_chunk n_tail cap  n_res tail res masked boundary no_mask
        { "no coarse",          4, 1, 512, 0,      37,    0,   0,    Q8,  Q8, false, false, false },
        { "one tail row",       4, 1, 512, 2,      1,     64,  64,   Q8,  Q8, false, false, false },
        { "L1 partial res",     4, 1, 512, 2,      100,   256, 100,  Q8,  Q8, true,  true,  false },
        { "L4 full res",        4, 4, 512, 2,      100,   256, 256,  Q8,  Q8,  true, true,  false },
        { "L4 empty res",       4, 4, 384, 3,      70,    64,  0,    Q8,  Q8, false, false, false },
        { "L4 cap 1024",        2, 4, 640, 2,      300,   1024, 700, Q8,  Q8,  true, true,  false },
        { "L1 no mask",         2, 1, 512, 1,      65,    64,  64,   Q8,  Q8, false, false, true  },
        { "L5 cap 1024",        4, 5, 512, 2,      150,   1024, 600, Q8,  Q8, true,  true,  false },
        { "L7 cap 1024",        4, 7, 512, 2,      150,   1024, 600, Q8,  Q8, true,  true,  false },
        { "L8 cap 1024",        4, 8, 640, 2,      300,   1024, 900, Q8,  Q8, true,  true,  false },
    };

    for (const eq_case & ec : cases) {
        kvq_problem P;
        kvq_reference R8, R32;
        printf("  case %-18s G=%d L=%d C=%d n_chunk=%d n_tail=%d cap=%d n_res=%d tail=%s res=%s\n", ec.name, ec.G, ec.L, ec.n_crow,
            ec.n_chunk, ec.n_tail, ec.cap, ec.n_res, ggml_type_name(ec.type_tail), ggml_type_name(ec.type_res));
        build_equivalence(ec, P, R8, R32);

        const size_t na = R8.attn.size();
        const size_t ns = R8.score.size();

        for (ggml_backend_t be : { cpu, gpu }) {
            if (!be) {
                continue;
            }
            const bool is_cpu = be == cpu;
            // CPU: same int8 tables in double, only rounding differs; GPU: f16 qt and f16 weights in the mma
            const double tol_a = is_cpu ? 1e-9 : 2e-5;
            const double tol_s = is_cpu ? 1e-9 : 2e-5;

            std::vector<float> a = run_kvq(be, P, GGML_KVQ_ATTN);
            if (a.empty()) {
                printf("    %-5s not supported\n", ggml_backend_name(be));
                continue;
            }
            const double ea   = nmse(a.data(), R8.attn.data(), na);
            const double ea32 = nmse(a.data(), R32.attn.data(), na);
            printf("    %-5s attn  nmse %.2e (f32 tables: %.2e)", ggml_backend_name(be), ea, ea32);
            CHECK(ea < tol_a, "%s %s attn nmse %e", ec.name, ggml_backend_name(be), ea);

            if (P.n_coarse > 0) {
                std::vector<float> s  = run_kvq(be, P, GGML_KVQ_SCORE);
                const double es  = nmse(s.data(), R8.score.data(), ns);
                printf(", score nmse %.2e", es);
                CHECK(es < tol_s, "%s %s score nmse %e", ec.name, ggml_backend_name(be), es);
            }

            // the same inputs give the same bits (no order of float atomics in the result)
            int n_diff = 0;
            const int modes[] = { GGML_KVQ_ATTN, GGML_KVQ_SCORE };
            for (int mode : modes) {
                if (mode != GGML_KVQ_ATTN && P.n_coarse == 0) {
                    continue;
                }
                const std::vector<float> r0 = run_kvq(be, P, mode);
                for (int rep = 0; rep < 8; ++rep) {
                    const std::vector<float> r = run_kvq(be, P, mode);
                    n_diff += r.size() != r0.size() || memcmp(r.data(), r0.data(), r0.size()*sizeof(float)) != 0;
                }
            }
            printf(", repeats differ %d", n_diff);
            CHECK(n_diff == 0, "%s %s: %d repeated calls differ from the first", ec.name, ggml_backend_name(be), n_diff);

            // shuffled pages in a pool with 3 spare pages: the same bits as the identity page table
            int n_page_diff = 0;
            if (P.n_coarse > 0) {
                kvq_problem Pp = P;
                Pp.n_pages = P.n_chunk + 3;
                Pp.pt.resize(P.n_chunk);
                std::vector<int32_t> pool(Pp.n_pages);
                std::iota(pool.begin(), pool.end(), 0);
                std::mt19937 prng(77 + P.n_chunk);
                std::shuffle(pool.begin(), pool.end(), prng);
                std::copy(pool.begin(), pool.begin() + P.n_chunk, Pp.pt.begin());
                for (int mode : modes) {
                    const std::vector<float> r0 = run_kvq(be, P, mode);
                    const std::vector<float> r  = run_kvq(be, Pp, mode);
                    n_page_diff += r.size() != r0.size() || memcmp(r.data(), r0.data(), r0.size()*sizeof(float)) != 0;
                }
            }
            printf(", shuffled pages differ %d", n_page_diff);
            CHECK(n_page_diff == 0, "%s %s: the shuffled page table gives other bits", ec.name, ggml_backend_name(be));
            printf("\n");
        }
    }
}

// one call with n_stream streams against n_stream single-stream calls on the same data: the same bits per stream
// (a stream reads only its own rows and its arithmetic only depends on its own numbers), CPU and GPU
static void test_streams(ggml_backend_t cpu, ggml_backend_t gpu) {
    const ggml_type Q8 = GGML_TYPE_Q8_0;
    struct sc {
        const char * name;
        int G, L, C, n_tail, cap;
        std::vector<int> n_chunk, n_res; // per stream
    };
    const sc cases[] = {
        { "S2 L1",    2, 1, 512, 100, 256,  { 3, 2 },       { 100, 40 } },
        { "S2 L4",    4, 4, 512, 100, 256,  { 2, 3 },       { 256, 0 } },
        { "S4 L1",    2, 1, 512, 70,  1024, { 3, 1, 2, 3 }, { 300, 0, 700, 50 } },
        { "S4 L4",    4, 4, 384, 70,  1024, { 2, 3, 1, 2 }, { 100, 600, 384, 7 } },
        { "S4 L8",    4, 8, 640, 150, 1024, { 1, 2, 2, 1 }, { 900, 5, 0, 640 } },
        { "S3 L8 ring", 4, 8, 512, 33, 256, { 2, 2, 1 },    { 256, 100, 64 } },
    };
    for (const sc & c : cases) {
        const int S = (int) c.n_chunk.size();
        std::vector<kvq_problem> Ps(S);
        for (int s = 0; s < S; ++s) {
            kvq_reference R8, R32;
            eq_case ec = { "", c.G, c.L, c.C, c.n_chunk[s], c.n_tail, c.cap, c.n_res[s], Q8, Q8, true, true, false, s + 1 };
            build_equivalence(ec, Ps[s], R8, R32);
            // a ring shorter than the others: the tail rows beyond ring_s are masked (the free cells), the others causal
            {
                kvq_problem & P = Ps[s];
                const int ring = c.n_tail - s*(c.n_tail/6);
                for (int l = 0; l < P.L; ++l) {
                    for (int r = 0; r < P.n_tail; ++r) {
                        if (r >= ring || r > ring - P.L + l) {
                            P.mask[(size_t) l*P.n_kv() + P.n_coarse + r] = ggml_fp32_to_fp16(-INFINITY);
                        }
                    }
                }
            }
        }
        std::vector<const kvq_problem *> pp;
        for (auto & P : Ps) {
            pp.push_back(&P);
        }
        printf("  case %-12s streams %d n_chunk", c.name, S);
        for (int s = 0; s < S; ++s) {
            printf(" %d", c.n_chunk[s]);
        }
        printf("\n");

        for (ggml_backend_t be : { cpu, gpu }) {
            if (!be) {
                continue;
            }
            for (int mode : { (int) GGML_KVQ_ATTN, (int) GGML_KVQ_SCORE }) {
                const std::vector<float> r = run_kvq_streams(be, pp, mode);
                if (r.empty()) {
                    printf("    %-5s not supported\n", ggml_backend_name(be));
                    continue;
                }
                int n_diff = 0;
                double worst = 0.0;
                int n_coarse_max = 0;
                for (auto & P : Ps) {
                    n_coarse_max = std::max(n_coarse_max, P.n_coarse);
                }
                for (int s = 0; s < S; ++s) {
                    const kvq_problem & P = Ps[s];
                    if (mode == GGML_KVQ_SCORE && P.n_coarse == 0) {
                        continue;
                    }
                    const std::vector<float> one = run_kvq(be, P, mode);
                    if (mode == GGML_KVQ_ATTN) {
                        const size_t n = one.size();
                        const float * x = r.data() + s*n;
                        n_diff += memcmp(x, one.data(), n*sizeof(float)) != 0;
                        worst = std::max(worst, nmse(x, one.data(), n));
                    } else {
                        // [n_coarse_max, G, L] per stream: the rows of the stream as in the single call, the rest 0
                        for (int gl = 0; gl < P.G*P.L; ++gl) {
                            const float * x = r.data() + ((size_t) s*P.G*P.L + gl)*n_coarse_max;
                            const float * y = one.data() + (size_t) gl*P.n_coarse;
                            n_diff += memcmp(x, y, (size_t) P.n_coarse*sizeof(float)) != 0;
                            for (int j = P.n_coarse; j < n_coarse_max; ++j) {
                                n_diff += x[j] != 0.0f;
                            }
                        }
                    }
                }
                printf("    %-5s %-5s streams differ from single calls: %d (max nmse %.1e)\n", ggml_backend_name(be),
                    mode == GGML_KVQ_ATTN ? "attn" : "score", n_diff, worst);
                // CPU: the same double math per stream; GPU: the same kernels with the same arithmetic order
                CHECK(n_diff == 0, "%s %s %s: %d streams differ from the single-stream calls", c.name, ggml_backend_name(be),
                    mode == GGML_KVQ_ATTN ? "attn" : "score", n_diff);
            }
        }

        // CPU against GPU, all streams in one call
        if (gpu) {
            for (int mode : { (int) GGML_KVQ_ATTN, (int) GGML_KVQ_SCORE }) {
                const std::vector<float> a = run_kvq_streams(cpu, pp, mode);
                const std::vector<float> b = run_kvq_streams(gpu, pp, mode);
                if (a.size() == b.size() && !a.empty()) {
                    const double e = nmse(b.data(), a.data(), a.size());
                    printf("    gpu vs cpu %-5s nmse %.2e\n", mode == GGML_KVQ_ATTN ? "attn" : "score", e);
                    CHECK(e < 2e-5, "%s gpu vs cpu nmse %e", c.name, e);
                }
            }
        }
    }
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    GGML_ASSERT(cpu);
    printf("backends: %s%s%s\n", ggml_backend_name(cpu), gpu ? ", " : "", gpu ? ggml_backend_name(gpu) : "");

    printf("equivalence with decode-then-attend (kvq_ref_decode with the int8 inverse of the op; f32 for information)\n");
    test_equivalence(cpu, gpu);

    printf("several streams in one call against single-stream calls\n");
    test_streams(cpu, gpu);

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
