#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, /*K=*/1);
    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il,
        ggml_tensor **       separate_history) {
    const int64_t n_seqs = ubatch.n_seqs;

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    const bool separate = separate_history && !inp->tree_parent &&
            qkv_mixed->ne[0] >= int64_t(cparams.n_rs_seq) + conv_kernel_size;
    if (separate_history) {
        *separate_history = separate ? ggml_dup(ctx0, conv_states) : nullptr;
        if (separate) {
            // Cache writes can overwrite the old history before convolution executes.
            ggml_build_forward_expand(gf, *separate_history);
        }
    }
    ggml_tensor * conv_input = separate ? qkv_mixed : ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    if (inp->tree_parent) {
        const auto * mctx_cur = inp->mctx;
        const auto kv_head = mctx_cur->get_head();
        const auto mem_size = mctx_cur->get_size();
        if (!inp->tree_conv) {
            inp->tree_conv = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, conv_kernel_size * ubatch.n_tokens);
            ggml_set_input(inp->tree_conv);
            ggml_set_name(inp->tree_conv, "tree_conv_rows");
        }
        ggml_tensor * rows = ggml_cont_2d(ctx0, ggml_transpose(ctx0, conv_input),
                conv_channels, conv_input->ne[0] * n_seqs);
        rows = ggml_get_rows(ctx0, rows, inp->tree_conv);
        rows = ggml_reshape_3d(ctx0, rows, conv_channels, conv_kernel_size, ubatch.n_tokens);
        conv_input = ggml_cont(ctx0, ggml_permute(ctx0, rows, 1, 0, 2, 3));

        const int64_t K = cparams.n_rs_seq + 1;
        GGML_ASSERT(ubatch.n_seq_tokens <= K);
        if (!inp->s_slot_rev) {
            inp->s_slot_rev = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seq_tokens);
            ggml_set_input(inp->s_slot_rev);
        }
        ggml_tensor * windows = ggml_view_4d(ctx0, conv_input,
                conv_kernel_size - 1, ubatch.n_seq_tokens, conv_channels, n_seqs,
                conv_input->nb[2], conv_input->nb[1], ubatch.n_seq_tokens * conv_input->nb[2], sizeof(float));
        const size_t row_size = ggml_row_size(conv_states_all->type, (conv_kernel_size - 1) * conv_channels);
        ggml_tensor * slots = ggml_view_4d(ctx0, conv_states_all,
                conv_kernel_size - 1, K, conv_channels, n_seqs, (size_t) mem_size * row_size,
                ggml_row_size(conv_states_all->type, conv_kernel_size - 1), conv_states_all->nb[1], kv_head * row_size);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, slots, windows, inp->s_slot_rev));
        return conv_input;
    }

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0,
            separate ? qkv_mixed->ne[0] : -1);

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    // Compact replay retains update operands instead of per-token recurrent states.
    const bool replay = inp->gdn_ctl != nullptr;
    const bool keep = cparams.n_rs_seq > 0 && !replay;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        if (replay) {
            ggml_tensor * gdn = output->view_src;
            GGML_ASSERT(gdn && gdn->op == GGML_OP_GATED_DELTA_NET);
            ggml_gated_delta_net_set_replay(gdn, mctx_cur->get_st_l(il), inp->gdn_ctl);
            if (inp->tree_parent) {
                GGML_ASSERT(g->ne[0] == 1 && n_seq_tokens <= 32);
                gdn->src[8] = inp->tree_parent;
                gdn->op_params[1] = inp->tree_slots;
            }
        }
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        if (replay && n_seq_tokens <= mctx_cur->get_gdn_width()) {
            ggml_tensor * stash = mctx_cur->get_st_l(il);
            const int64_t bank = stash->ne[0] * stash->ne[1];
            for (uint32_t seq = 0; seq < ubatch.n_seqs; ++seq) {
                const uint32_t first = seq * ubatch.n_seq_tokens;
                for (int32_t j = 1; j < ubatch.n_seq_id[first]; ++j) {
                    ggml_tensor * src = ggml_view_1d(ctx0, stash, bank, ubatch.seq_id[first][0] * stash->nb[2]);
                    ggml_tensor * dst = ggml_view_1d(ctx0, stash, bank, ubatch.seq_id[first][j] * stash->nb[2]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
                }
            }
        }

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K);
    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
