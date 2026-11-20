#pragma once

// the inputs of one ggml_kvq_attn call, its graph and its runner, shared by test-kvq-attn and llama-kvq-bench

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>

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

    // the codes and tables are given by logical chunk and uploaded into pages: chunk t goes to page pt[t]
    // of a pool of n_pages pages (the other pages are zero); empty pt: the identity, n_pages = 0: n_chunk
    int                      n_pages = 0;
    std::vector<int32_t>     pt;

    std::vector<float>       q;    // [L][H][D]
    std::vector<int32_t>     kc;   // [G][n_coarse][nwk]
    std::vector<int32_t>     vc;   // [G][n_coarse][nwv + 1]
    std::vector<int8_t>      tinv; // [n_chunk][2G][D*D]
    std::vector<float>       taux; // [n_chunk][2G][3*D]
    std::vector<uint8_t>     k, v; // [n_tail][G][row], cache layout
    std::vector<uint8_t>     rkv;  // [2][G][cap][row]
    std::vector<int32_t>     ridx; // [G][cap]
    std::vector<ggml_fp16_t> mask; // [L][n_kv]

    int n_head() const { return G*n_rep; }
    int n_kv()   const { return n_coarse + n_tail; }
    int pages()  const { return n_pages > 0 ? n_pages : n_chunk; }
    int page(int t) const { return pt.empty() ? t : pt[t]; }
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

    const bool attn = mode == GGML_KVQ_ATTN;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, P.n_head(), P.L);
    ggml_set_name(q, "q");

    ggml_tensor * kc = nullptr, * vc = nullptr, * tab = nullptr, * pt = nullptr;
    if (P.n_coarse > 0) {
        kc = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, P.nwk, P.n_crow, P.G, P.pages());
        ggml_set_name(kc, "kc");
        if (attn) {
            vc = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, P.nwv + 1, P.n_crow, P.G, P.pages());
            ggml_set_name(vc, "vc");
        }
        tab  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, D*D + 3*D*sizeof(float), 2*P.G, P.pages());
        pt   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P.n_chunk);
        ggml_set_name(tab, "tab");
        ggml_set_name(pt, "pt");
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
        ggml_tensor * out = ggml_kvq_attn(ctx, q, kc, vc, tab, pt, k, v, rkv, ridx, mask, P.scale, P.n_coarse, P.n_crow, mode);
        if (n_top > 0) {
            out = ggml_top_k(ctx, ggml_reshape_2d(ctx, out, P.n_coarse, P.G*P.L), n_top);
        }
        ggml_build_forward_expand(g->gf, out);
        g->out = out;
    }

    g->buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(g->buf);

    set_named(ctx, "q", P.q.data(), P.q.size()*sizeof(float));
    // the codes of logical chunk t into page pt[t] ([G][n_crow][nw] per page), the tables of chunk t into slot pt[t]
    auto set_pages = [&](const char * name, const void * data, size_t row_bytes, size_t chunk_bytes) {
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            return;
        }
        const bool codes = row_bytes > 0;
        const size_t page_bytes = t->nb[3];
        std::vector<uint8_t> pages_data(ggml_nbytes(t), 0);
        for (int c = 0; c < P.n_chunk; ++c) {
            uint8_t * dst = pages_data.data() + (size_t) P.page(c)*page_bytes;
            if (!codes) {
                memcpy(dst, (const uint8_t *) data + (size_t) c*chunk_bytes, chunk_bytes);
                continue;
            }
            const int rows = std::min(P.n_crow, P.n_coarse - c*P.n_crow);
            for (int gi = 0; gi < P.G; ++gi) {
                memcpy(dst + (size_t) gi*P.n_crow*row_bytes,
                        (const uint8_t *) data + ((size_t) gi*P.n_coarse + (size_t) c*P.n_crow)*row_bytes, (size_t) rows*row_bytes);
            }
        }
        ggml_backend_tensor_set(t, pages_data.data(), 0, pages_data.size());
    };
    if (P.n_coarse > 0) {
        set_pages("kc", P.kc.data(), P.nwk*sizeof(int32_t), 0);
        set_pages("vc", P.vc.data(), (P.nwv + 1)*sizeof(int32_t), 0);
        // the table slots of chunk t: the inverse table bytes, then the aux floats
        if (ggml_tensor * t = ggml_get_tensor(ctx, "tab")) {
            const size_t slot = D*D + 3*D*sizeof(float);
            std::vector<uint8_t> tab_data(ggml_nbytes(t), 0);
            for (int c = 0; c < P.n_chunk; ++c) {
                for (int m = 0; m < 2*P.G; ++m) {
                    uint8_t * dst = tab_data.data() + (size_t) P.page(c)*t->nb[2] + (size_t) m*slot;
                    memcpy(dst, P.tinv.data() + ((size_t) c*2*P.G + m)*D*D, (size_t) D*D);
                    memcpy(dst + D*D, P.taux.data() + ((size_t) c*2*P.G + m)*3*D, 3*D*sizeof(float));
                }
            }
            ggml_backend_tensor_set(t, tab_data.data(), 0, tab_data.size());
        }
        std::vector<int32_t> pt_data(P.n_chunk);
        for (int c = 0; c < P.n_chunk; ++c) {
            pt_data[c] = P.page(c);
        }
        set_named(ctx, "pt", pt_data.data(), pt_data.size()*sizeof(int32_t));
    }
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

// one call over several streams (ggml_kvq_attn_streams): stream s is the problem Ps[s]; all of them have the same G, n_rep, L,
// n_crow, nwk, nwv, cap, n_tail and types, and their own coarse rows, pages, resident rows and mask. The pages of the streams are
// concatenated in one pool (stream s after the pages of the streams before it), the resident slots, the tail rows and the mask
// rows are the slices of the stream. The result has the layout of the op: attn [D, n_head, L, S], score [n_coarse_max, G, L, S]
static std::vector<float> run_kvq_streams(ggml_backend_t backend, const std::vector<const kvq_problem *> & Ps, int mode) {
    const int S = (int) Ps.size();
    const kvq_problem & P0 = *Ps[0];
    const bool attn = mode == GGML_KVQ_ATTN;

    int n_coarse_max = 0, n_chunk_max = 0, n_pages = 0, n_kv_max = 0;
    std::vector<int> page0(S), nc(GGML_KVQ_MAX_STREAM, 0);
    for (int s = 0; s < S; ++s) {
        const kvq_problem & P = *Ps[s];
        GGML_ASSERT(P.G == P0.G && P.L == P0.L && P.n_crow == P0.n_crow && P.cap == P0.cap && P.n_tail == P0.n_tail);
        page0[s]     = n_pages;
        n_pages     += P.pages();
        nc[s]        = P.n_coarse;
        n_coarse_max = std::max(n_coarse_max, P.n_coarse);
        n_chunk_max  = std::max(n_chunk_max, P.n_chunk);
        n_kv_max     = std::max(n_kv_max, P.n_kv());
    }

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead_custom(64, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const int G = P0.G, L = P0.L, cap = P0.cap, n_crow = P0.n_crow, n_tail = P0.n_tail;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, P0.n_head(), L, S);
    ggml_tensor * kc = nullptr, * vc = nullptr, * tab = nullptr, * pt = nullptr;
    if (n_coarse_max > 0) {
        kc  = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, P0.nwk, n_crow, G, n_pages);
        vc  = attn ? ggml_new_tensor_4d(ctx, GGML_TYPE_I32, P0.nwv + 1, n_crow, G, n_pages) : nullptr;
        tab = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, D*D + 3*D*sizeof(float), 2*G, n_pages);
        pt  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_chunk_max, S);
    }
    auto cache_view = [&](const kvq_problem & P) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, P.type_tail, D, G, n_tail, S);
        return ggml_permute(ctx, t, 0, 2, 1, 3);
    };
    ggml_tensor * k = cache_view(P0);
    ggml_tensor * v = attn ? cache_view(P0) : nullptr;
    ggml_tensor * rkv = nullptr, * ridx = nullptr;
    if (attn && cap > 0) {
        rkv  = ggml_new_tensor_4d(ctx, P0.type_res, D, cap, G, 2*S);
        ridx = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, cap, G, S);
    }
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv_max, L, 1, S);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_tensor * out = ggml_kvq_attn_streams(ctx, q, kc, vc, tab, pt, k, v, rkv, ridx, mask, P0.scale, nc.data(), n_crow, mode);
    ggml_build_forward_expand(gf, out);

    std::vector<float> res;
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return res;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf);

    auto set_cat = [&](ggml_tensor * t, auto && data_of) {
        if (!t) {
            return;
        }
        std::vector<uint8_t> all(ggml_nbytes(t), 0);
        size_t off = 0;
        for (int s = 0; s < S; ++s) {
            const std::pair<const uint8_t *, size_t> d = data_of(*Ps[s]);
            GGML_ASSERT(off + d.second <= all.size());
            memcpy(all.data() + off, d.first, d.second);
            off += d.second;
        }
        ggml_backend_tensor_set(t, all.data(), 0, all.size());
    };
    set_cat(q, [](const kvq_problem & P) { return std::make_pair((const uint8_t *) P.q.data(), P.q.size()*sizeof(float)); });
    set_cat(k, [](const kvq_problem & P) { return std::make_pair(P.k.data(), P.k.size()); });
    set_cat(v, [](const kvq_problem & P) { return std::make_pair(P.v.data(), P.v.size()); });
    set_cat(rkv, [](const kvq_problem & P) { return std::make_pair(P.rkv.data(), P.rkv.size()); });
    set_cat(ridx, [](const kvq_problem & P) { return std::make_pair((const uint8_t *) P.ridx.data(), P.ridx.size()*sizeof(int32_t)); });

    // mask rows of stream s: [L][n_kv_s] into [L][n_kv_max]
    {
        std::vector<ggml_fp16_t> m((size_t) S*L*n_kv_max, ggml_fp32_to_fp16(0.0f));
        for (int s = 0; s < S; ++s) {
            const kvq_problem & P = *Ps[s];
            for (int l = 0; l < L; ++l) {
                for (int r = 0; r < P.n_kv(); ++r) {
                    m[((size_t) s*L + l)*n_kv_max + r] = P.has_mask ? P.mask[(size_t) l*P.n_kv() + r] : ggml_fp32_to_fp16(0.0f);
                }
            }
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size()*sizeof(ggml_fp16_t));
    }

    if (kc) {
        const size_t slot = D*D + 3*D*sizeof(float);
        std::vector<uint8_t> kc_d(ggml_nbytes(kc), 0), vc_d(vc ? ggml_nbytes(vc) : 0, 0), tab_d(ggml_nbytes(tab), 0);
        std::vector<int32_t> pt_d((size_t) n_chunk_max*S, 0x3fffffff);
        for (int s = 0; s < S; ++s) {
            const kvq_problem & P = *Ps[s];
            for (int c = 0; c < P.n_chunk; ++c) {
                const int page = page0[s] + P.page(c);
                pt_d[(size_t) s*n_chunk_max + c] = page;
                const int rows = std::min(P.n_crow, P.n_coarse - c*P.n_crow);
                for (int gi = 0; gi < G; ++gi) {
                    const size_t rk = P.nwk*sizeof(int32_t), rv = (P.nwv + 1)*sizeof(int32_t);
                    memcpy(kc_d.data() + (size_t) page*kc->nb[3] + (size_t) gi*kc->nb[2],
                        (const uint8_t *) P.kc.data() + ((size_t) gi*P.n_coarse + (size_t) c*n_crow)*rk, (size_t) rows*rk);
                    if (vc) {
                        memcpy(vc_d.data() + (size_t) page*vc->nb[3] + (size_t) gi*vc->nb[2],
                            (const uint8_t *) P.vc.data() + ((size_t) gi*P.n_coarse + (size_t) c*n_crow)*rv, (size_t) rows*rv);
                    }
                }
                for (int m = 0; m < 2*G; ++m) {
                    uint8_t * dst = tab_d.data() + (size_t) page*tab->nb[2] + (size_t) m*slot;
                    memcpy(dst, P.tinv.data() + ((size_t) c*2*G + m)*D*D, (size_t) D*D);
                    memcpy(dst + D*D, P.taux.data() + ((size_t) c*2*G + m)*3*D, 3*D*sizeof(float));
                }
            }
        }
        ggml_backend_tensor_set(kc, kc_d.data(), 0, kc_d.size());
        if (vc) {
            ggml_backend_tensor_set(vc, vc_d.data(), 0, vc_d.size());
        }
        ggml_backend_tensor_set(tab, tab_d.data(), 0, tab_d.size());
        ggml_backend_tensor_set(pt, pt_d.data(), 0, pt_d.size()*sizeof(int32_t));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    res.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}
