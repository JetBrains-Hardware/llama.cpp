#pragma once

// the inputs of one ggml_kvq_attn call, its graph and its runner, shared by test-kvq-attn and llama-kvq-bench

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
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

    ggml_tensor * kc = nullptr, * vc = nullptr, * tinv = nullptr, * taux = nullptr;
    if (P.n_coarse > 0) {
        kc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, P.nwk, P.n_coarse, P.G);
        ggml_set_name(kc, "kc");
        if (attn) {
            vc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, P.nwv + 1, P.n_coarse, P.G);
            ggml_set_name(vc, "vc");
        }
        tinv = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, D, D, 2*P.G, P.n_chunk);
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
    set_named(ctx, "tinv", P.tinv.data(), P.tinv.size());
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
