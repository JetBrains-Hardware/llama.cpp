#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>


template <int S_v, bool KDA, bool keep_rs_t>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}


// consecutive floats that may start on any float boundary
typedef float gdn_float4 __attribute__((ext_vector_type(4), aligned(4)));
typedef float gdn_float8 __attribute__((ext_vector_type(8), aligned(4)));

// n consecutive floats of a row vector: one wide load instead of a load per value
template <int n>
static __dpct_inline__ void gdn_load_rows(const float * __restrict__ p, float * out) {
    if constexpr (n % 8 == 0) {
#pragma unroll
        for (int j = 0; j < n; j += 8) {
            const gdn_float8 v = *reinterpret_cast<const gdn_float8 *>(p + j);
#pragma unroll
            for (int l = 0; l < 8; l++) {
                out[j + l] = v[l];
            }
        }
    } else if constexpr (n % 4 == 0) {
#pragma unroll
        for (int j = 0; j < n; j += 4) {
            const gdn_float4 v = *reinterpret_cast<const gdn_float4 *>(p + j);
#pragma unroll
            for (int l = 0; l < 4; l++) {
                out[j + l] = v[l];
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < n; j++) {
            out[j] = p[j];
        }
    }
}

// Block variant of the kernel below. The recurrence is still walked token by token, but the memory
// traffic per token is cut down, because on Intel GPUs every load is a message to the data port and
// the sequential walk spends its time waiting for them:
//  - a lane owns rows_per_lane CONSECUTIVE rows of its state column, so k and q of a token are read
//    with one wide load each instead of rows_per_lane gathers;
//  - v, beta and g of the next warp_size tokens are read with one load each (lane l reads token
//    t0 + l) and handed out with a sub-group broadcast;
//  - the outputs of warp_size tokens are collected in a register (lane l keeps token t0 + l) and
//    stored with one write.
// The math per state element is the same; sums over the rows run in another order (last-bit changes).
template <int S_v, bool KDA, bool keep_rs_t>
void gated_delta_net_block_sycl(const float *     q,
                                const float *     k,
                                const float *     v,
                                const float *     g,
                                const float *     beta,
                                const float *     curr_state,
                                float *           dst,
                                float *           state,
                                int64_t           H,
                                int64_t           n_tokens,
                                int64_t           sq1,
                                int64_t           sq2,
                                int64_t           sq3,
                                int64_t           sv1,
                                int64_t           sv2,
                                int64_t           sv3,
                                int64_t           sb1,
                                int64_t           sb2,
                                int64_t           sb3,
                                const sycl::uint3 neqk1_magic,
                                const sycl::uint3 rq3_magic,
                                float             scale,
                                int64_t           state_slot_stride,
                                int               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto           sg       = item_ct1.get_sub_group();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      lane     = item_ct1.get_local_id(2);
    const int      sg_lane  = sg.get_local_linear_id();
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = S_v / warp_size;
    const int     row0          = lane * rows_per_lane;

    float s_shard[rows_per_lane];
    gdn_load_rows<rows_per_lane>(curr_state + row0, s_shard);

    const float * qk_base_q = q + iq3 * sq3 + iq1 * sq1 + row0;
    const float * qk_base_k = k + iq3 * sq3 + iq1 * sq1 + row0;
    const float * v_base    = v + sequence * sv3 + h_idx * sv1 + col;
    const int64_t gb_base   = sequence * sb3 + h_idx * sb1;

    // one token of the recurrence; v_val, beta_val (and g_val when !KDA) are the same in every lane
    auto step = [&](int64_t t, float v_val, float beta_val, float g_val) -> float {
        float k_rows[rows_per_lane];
        float q_rows[rows_per_lane];
        gdn_load_rows<rows_per_lane>(qk_base_k + t * sq2, k_rows);
        gdn_load_rows<rows_per_lane>(qk_base_q + t * sq2, q_rows);

        float g_rows[rows_per_lane];
        if constexpr (KDA) {
            gdn_load_rows<rows_per_lane>(g + (gb_base + t * sb2) * S_v + row0, g_rows);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_rows[r] = sycl::native::exp(g_rows[r]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_rows[r] = g_val;
            }
        }

        // kv[col] = sum_i S[i][col] * k[i]   (KDA: sum_i g[i] * S[i][col] * k[i])
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if constexpr (KDA) {
                kv_shard += g_rows[r] * s_shard[r] * k_rows[r];
            } else {
                kv_shard += s_shard[r] * k_rows[r];
            }
        }
        const float kv_col = sycl::reduce_over_group(sg, kv_shard, sycl::plus<>());

        // delta[col] = (v[col] - g * kv[col]) * beta   (KDA: g is already inside kv)
        const float delta_col = KDA ? (v_val - kv_col) * beta_val : (v_val - g_val * kv_col) * beta_val;

        // S[i][col] = g * S[i][col] + k[i] * delta[col];  attn[col] = sum_i S[i][col] * q[i]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_rows[r] * s_shard[r] + k_rows[r] * delta_col;
            attn_partial += s_shard[r] * q_rows[r];
        }
        const float attn_col = sycl::reduce_over_group(sg, attn_partial, sycl::plus<>());

        if constexpr (keep_rs_t) {
            const int target_slot = (int) (n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                float * slot_state = state + target_slot * state_slot_stride + col * S_v + row0;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    slot_state[r] = s_shard[r];
                }
            }
        }
        return attn_col;
    };

    const int64_t out_stride   = S_v * H;
    const int64_t n_block_toks = n_tokens / warp_size * warp_size;

    for (int64_t t0 = 0; t0 < n_block_toks; t0 += warp_size) {
        // lane l reads the scalars of token t0 + l
        const int64_t tl       = t0 + sg_lane;
        const float   v_blk    = v_base[tl * sv2];
        const float   beta_blk = beta[gb_base + tl * sb2];
        float         g_blk    = 0.0f;
        if constexpr (!KDA) {
            g_blk = sycl::native::exp(g[gb_base + tl * sb2]);
        }
        float out_blk = 0.0f;
#pragma unroll
        for (int tt = 0; tt < warp_size; tt++) {
            const float v_val    = sycl::group_broadcast(sg, v_blk, tt);
            const float beta_val = sycl::group_broadcast(sg, beta_blk, tt);
            const float g_val    = KDA ? 0.0f : sycl::group_broadcast(sg, g_blk, tt);
            const float attn_col = step(t0 + tt, v_val, beta_val, g_val);
            out_blk              = sg_lane == tt ? attn_col : out_blk;
        }
        attn_data[tl * out_stride] = out_blk * scale;
    }

    // tail: fewer than warp_size tokens left
    for (int64_t t = n_block_toks; t < n_tokens; t++) {
        const float v_val    = v_base[t * sv2];
        const float beta_val = beta[gb_base + t * sb2];
        float       g_val    = 0.0f;
        if constexpr (!KDA) {
            g_val = sycl::native::exp(g[gb_base + t * sb2]);
        }
        const float attn_col = step(t, v_val, beta_val, g_val);
        if (sg_lane == 0) {
            attn_data[t * out_stride] = attn_col * scale;
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            state[col * S_v + row0 + r] = s_shard[r];
        }
    }
}


typedef float gdn_float16 __attribute__((ext_vector_type(16), aligned(4)));
typedef float gdn_float64 __attribute__((ext_vector_type(64), aligned(4)));

// n consecutive floats at an address that is the same in every lane: block loads of 64 (or 16) values
template <int n>
static __dpct_inline__ void gdn_load_uniform(const float * __restrict__ p, float * out) {
    static_assert(n % 16 == 0, "n must be a multiple of 16");
    if constexpr (n % 64 == 0) {
#pragma unroll
        for (int j = 0; j < n; j += 64) {
            const gdn_float64 v = *reinterpret_cast<const gdn_float64 *>(p + j);
#pragma unroll
            for (int l = 0; l < 64; l++) {
                out[j + l] = v[l];
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < n; j += 16) {
            const gdn_float16 v = *reinterpret_cast<const gdn_float16 *>(p + j);
#pragma unroll
            for (int l = 0; l < 16; l++) {
                out[j + l] = v[l];
            }
        }
    }
}

// Column kernel (scalar gate only). One sub-group owns warp_size COLUMNS of the state of one head:
// a lane is a column and the S_v rows of the state live in S_v registers. The values that are shared
// by all columns (k, q, g, beta of a token) are then plain scalars that are read once per sub-group
// with block loads, every product is one full-width multiply-add, and no reduction across the
// sub-group is needed: the sums over the rows run inside a lane. Compared with one column per
// sub-group this needs warp_size times fewer threads and no shuffles, and the per-token memory
// traffic of a head drops from 2 * S_v gathers to 2 * S_v / 16 block loads.
// S_v = 128 keeps 128 state registers plus k and q, so the kernel asks for the large register file.
template <int S_v, bool keep_rs_t>
void gated_delta_net_cols_sycl(const float *     q,
                               const float *     k,
                               const float *     v,
                               const float *     g,
                               const float *     beta,
                               const float *     curr_state,
                               float *           dst,
                               float *           state,
                               int64_t           H,
                               int64_t           n_tokens,
                               int64_t           sq1,
                               int64_t           sq2,
                               int64_t           sq3,
                               int64_t           sv1,
                               int64_t           sv2,
                               int64_t           sv3,
                               int64_t           sb1,
                               int64_t           sb2,
                               int64_t           sb3,
                               const sycl::uint3 neqk1_magic,
                               const sycl::uint3 rq3_magic,
                               float             scale,
                               int64_t           state_slot_stride,
                               int               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto           sg       = item_ct1.get_sub_group();
    const uint32_t h_idx    = item_ct1.get_group(0);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      col      = item_ct1.get_global_id(2);
    const int      sg_lane  = sg.get_local_linear_id();

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset + col * S_v;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col;

    // the column of this lane
    float s_col[S_v];
    gdn_load_rows<S_v>(curr_state, s_col);

    const float * q_base  = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base  = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base  = v + sequence * sv3 + h_idx * sv1 + col;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;

    constexpr int n_acc = S_v >= 64 ? 8 : (S_v >= 32 ? 4 : 2); // independent partial sums: the multiply-adds of one sum form a serial chain

    auto step = [&](int64_t t, float beta_val, float g_val) {
        float k_rows[S_v];
        float q_rows[S_v];
        gdn_load_uniform<S_v>(k_base + t * sq2, k_rows);
        gdn_load_uniform<S_v>(q_base + t * sq2, q_rows);
        const float v_val = v_base[t * sv2];

        // kv[col] = sum_i S[i][col] * k[i]
        float kv_acc[n_acc];
#pragma unroll
        for (int a = 0; a < n_acc; a++) {
            kv_acc[a] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            kv_acc[i % n_acc] += s_col[i] * k_rows[i];
        }
        float kv_col = 0.0f;
#pragma unroll
        for (int a = 0; a < n_acc; a++) {
            kv_col += kv_acc[a];
        }

        // delta[col] = (v[col] - g * kv[col]) * beta
        const float delta_col = (v_val - g_val * kv_col) * beta_val;

        // S[i][col] = g * S[i][col] + k[i] * delta[col];  attn[col] = sum_i S[i][col] * q[i]
        float at_acc[n_acc];
#pragma unroll
        for (int a = 0; a < n_acc; a++) {
            at_acc[a] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            s_col[i] = g_val * s_col[i] + k_rows[i] * delta_col;
            at_acc[i % n_acc] += s_col[i] * q_rows[i];
        }
        float attn_col = 0.0f;
#pragma unroll
        for (int a = 0; a < n_acc; a++) {
            attn_col += at_acc[a];
        }
        attn_data[t * (S_v * H)] = attn_col * scale;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) (n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                float * slot_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int i = 0; i < S_v; i++) {
                    slot_state[i] = s_col[i];
                }
            }
        }
    };

    constexpr int warp_size    = ggml_sycl_get_physical_warp_size();
    const int64_t n_block_toks = n_tokens / warp_size * warp_size;

    for (int64_t t0 = 0; t0 < n_block_toks; t0 += warp_size) {
        // lane l reads beta and g of token t0 + l: two loads per warp_size tokens
        const int64_t tl       = t0 + sg_lane;
        const float   beta_blk = beta[gb_base + tl * sb2];
        const float   g_blk    = sycl::native::exp(g[gb_base + tl * sb2]);
#pragma unroll
        for (int tt = 0; tt < warp_size; tt++) {
            step(t0 + tt, sycl::group_broadcast(sg, beta_blk, tt), sycl::group_broadcast(sg, g_blk, tt));
        }
    }
    for (int64_t t = n_block_toks; t < n_tokens; t++) {
        step(t, beta[gb_base + t * sb2], sycl::native::exp(g[gb_base + t * sb2]));
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[i] = s_col[i];
        }
    }
}

template <int S_v, bool keep_rs_t, typename... Args>
static void launch_gated_delta_net_cols(dpct::queue_ptr stream, int64_t H, int64_t n_seqs, Args... args) {
    const sycl::nd_range<3> ndr(sycl::range<3>((size_t) H, (size_t) n_seqs, (size_t) S_v), sycl::range<3>(1, 1, WARP_SIZE));
    stream->parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             gated_delta_net_cols_sycl<S_v, keep_rs_t>(args...);
                         });
}

// GGML_SYCL_GDN_KERNEL: 0 = per-token kernel, one column per sub-group (original)
//                       1 = block kernel, one column per sub-group
//                       2 = column kernel for batches of GDN_COLS_MIN_TOKENS and more tokens with a scalar
//                           gate, per-token kernel otherwise (default)
#define GDN_COLS_MIN_TOKENS 8
static int gdn_kernel_mode() {
    static const int mode = ggml_sycl_get_env("GGML_SYCL_GDN_KERNEL", 2);
    return mode;
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const int num_warps = 4;
    dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const int  mode      = gdn_kernel_mode();
    const bool use_block = mode == 1;
    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    if constexpr (!KDA) {
        // prompt batches only: for a few tokens the per-token kernel is faster (17 us against 13 us per
        // call at one token on Arc B65), since the column kernel moves the whole state through registers
        if (mode >= 2 && n_tokens >= GDN_COLS_MIN_TOKENS && warp_size == WARP_SIZE && WARP_SIZE == 16) {
            switch (S_v) {
                case 16:
                    launch_gated_delta_net_cols<16, keep_rs_t>(stream, H, n_seqs, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3,
                                                               sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    return;
                case 32:
                    launch_gated_delta_net_cols<32, keep_rs_t>(stream, H, n_seqs, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3,
                                                               sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    return;
                case 64:
                    launch_gated_delta_net_cols<64, keep_rs_t>(stream, H, n_seqs, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3,
                                                               sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    return;
                case 128:
                    launch_gated_delta_net_cols<128, keep_rs_t>(stream, H, n_seqs, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3,
                                                                sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    return;
                default:
                    break;
            }
        }
    }

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         if (use_block) {
 gated_delta_net_block_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 } else {
 gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 }
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         if (use_block) {
 gated_delta_net_block_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 } else {
 gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 }
                                     });
            }
            break;
        case 64: {
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            if (use_block) {
 gated_delta_net_block_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 } else {
 gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 }
                                        });
            }
            break;
        }
        case 128: {
            {
                constexpr int sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            if (use_block) {
 gated_delta_net_block_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 } else {
 gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
 }
                                        });
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
