#include "models.h"
#include "llama-memory-hybrid-kvq.h"
#include "llama-memory-recurrent.h"

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // optional projection of the embeddings output
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_embd_out() }, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   { hparams.n_embd_out() },         TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            const int separate_flags = flags & TENSOR_SKIP ? flags | TENSOR_NOT_REQUIRED : flags;
            layer.wqkv_z = create_tensor(tn(LLM_TENSOR_ATTN_QKV_GATE, "weight", il), { n_embd, key_dim * 2 + value_dim * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wqkv_z) {
                layer.wqkv      = create_tensor(tn(LLM_TENSOR_ATTN_QKV,  "weight", il), { n_embd, key_dim * 2 + value_dim }, separate_flags);
                layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, separate_flags);
            }
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta_alpha_cat = create_tensor(tn(LLM_TENSOR_SSM_BETA_ALPHA_CAT, "weight", il), { n_embd, n_v_heads * 2 }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.ssm_beta_alpha_cat) {
                layer.ssm_beta  = create_tensor(tn(LLM_TENSOR_SSM_BETA,  "weight", il), { n_embd, n_v_heads }, separate_flags);
                layer.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, n_v_heads }, separate_flags);
            }
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

// every ubatch that does not take the packed path: flash attention over the scratch (the cache cells are in host memory), see llama_kvq_store
static bool kvq_use_scratch(const llama_kvq_params & p, const llama_kvq_store * store, bool packed_small) {
    return p.enabled && store != nullptr && store->scratch_ok() && !packed_small;
}

// scratch row r is position r, the rows after the host store come from the ring in position order,
// up to the last position of the ubatch padded like n_kv
static int64_t kvq_scr_rows(const llama_kvq_store * store, int32_t n_host, const llama_ubatch & ubatch) {
    llama_pos p_max = n_host - 1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        p_max = std::max(p_max, ubatch.pos[i]);
    }
    return std::min<int64_t>((int64_t) n_host + store->n_ring(), GGML_PAD((int64_t) p_max + 1, 256));
}

// the scratch attention reads the host rows [0, r0) in key blocks without a mask, the rows [r0, n_rows) with the mask
// (every query of a large ubatch sees all host rows); 0: one attention with the mask over all rows
static int64_t kvq_scr_r0(const llama_kvq_params & p, int32_t n_host) {
    return p.scr_block >= 256 ? n_host/256*256 : 0;
}

class llama_model_qwen35::llm_graph_input_kvq : public llm_graph_input_i {
public:
    llm_graph_input_kvq(const llama_memory_hybrid_kvq_context * mctx) :
        mctx(mctx), params(mctx->get_params()) {}
    virtual ~llm_graph_input_kvq() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (!params.enabled) {
            return;
        }

        if (packed_small) {
            mctx->set_input_kvq_ring(mask_attn.raw, mask_early.raw, ubatch, n_coarse_s);
            if (bias_s && bias_s->buffer) {
                // the streams: the sinks and the rows of a stream at or after its n_coarse are never selected
                const int64_t nc = bias_s->ne[0];
                std::vector<float> b(nc*seqs.size(), 0.0f);
                for (size_t s = 0; s < seqs.size(); ++s) {
                    for (int64_t r = 0; r < nc; ++r) {
                        if (r < params.n_sink || r >= n_coarse_s[s]) {
                            b[s*nc + r] = -INFINITY;
                        }
                    }
                }
                ggml_backend_tensor_set(bias_s, b.data(), 0, ggml_nbytes(bias_s));
            }
            if (pt_s && pt_s->buffer) {
                // the page table of each stream (a column), the pages of the chunks of its coarse region
                const int64_t nc = pt_s->ne[0];
                std::vector<int32_t> pt(nc*seqs.size(), 0);
                for (size_t s = 0; s < seqs.size(); ++s) {
                    const int64_t n_chunk = (n_coarse_s[s] + store->n_chunk_rows() - 1)/store->n_chunk_rows();
                    for (int64_t c = 0; c < n_chunk; ++c) {
                        pt[s*nc + c] = std::max(0, store->page_of_chunk(seqs[s], (int32_t) c));
                    }
                }
                ggml_backend_tensor_set(pt_s, pt.data(), 0, ggml_nbytes(pt_s));
            }
            mctx->get_store()->begin_ubatch(*ubatch);
        } else if (scratch) {
            if (!scr_cells) {
                mctx->set_input_kvq_scr(mask_scr.raw, ubatch, n_host, scr_r0);
            }
            if (scr_set_pos && scr_set_pos->buffer) {
                const int64_t off = store->scr_tail_off();
                std::vector<int32_t> pos(ubatch->n_tokens);
                for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                    pos[i] = (int32_t) (off + ubatch->pos[i] - scr_r0);
                }
                ggml_backend_tensor_set(scr_set_pos, pos.data(), 0, pos.size()*sizeof(int32_t));
            }
            mctx->get_store()->begin_scr(*ubatch, scr_r0);
        }
    }

    bool can_reuse(const llm_graph_params & gparams) override {
        mctx = dynamic_cast<const llama_memory_hybrid_kvq_context *>(gparams.mctx);
        if (mctx == nullptr) {
            return false;
        }

        // these change the graph, the other parameters only the cell flags
        const llama_kvq_params & p = mctx->get_params();

        bool res = p.enabled == params.enabled;

        res &= p.max_batch == params.max_batch && p.prefetch_n == params.prefetch_n && p.lead == params.lead && p.scr_block == params.scr_block;

        if (res && params.enabled) {
            res &= store == mctx->get_store();
            res &= store != nullptr && gparams.ubatch.n_seqs_unq == seqs.size() && gparams.ubatch.n_tokens % gparams.ubatch.n_seqs_unq == 0;
            bool all_packed = res;
            for (size_t s = 0; res && s < seqs.size(); ++s) {
                res &= seqs[s] == gparams.ubatch.seq_id_unq[s];
                res &= n_coarse_s[s] == store->n_coarse(seqs[s]) && n_host_s[s] == store->n_host(seqs[s]);
                all_packed &= llama_kvq_use_packed(p, gparams.ubatch.n_tokens/seqs.size(), n_coarse_s[s], n_host_s[s]);
            }
            res &= codes_gen == store->codes_gen();
            res &= packed_small == all_packed;
            res &= scratch == kvq_use_scratch(p, store, packed_small);
            const int64_t n_mask = n_coarse + mctx->get_attn()->get_n_kv();
            if (res && packed_small) {
                res &= mask_attn.raw->ne[0] == n_mask;
                res &= mask_attn.raw->ne[1] == (int64_t) (gparams.ubatch.n_tokens/seqs.size());
            }
            if (res && scratch && !scr_cells) {
                res &= scr_rows == kvq_scr_rows(store, n_host, gparams.ubatch) && scr_r0 == kvq_scr_r0(p, n_host);
                res &= mask_scr.raw->ne[1] == gparams.ubatch.n_tokens;
            }
            if (res && scratch && scr_set_pos) {
                res &= scr_set_pos->ne[0] == gparams.ubatch.n_tokens;
            }
        }

        params = p;

        return res;
    }

    // packed path, ubatch of at most max_batch tokens: the masks are over the coarse rows and then the ring cells
    bool          packed_small = false;
    llm_attn_mask mask_attn;  // F16 [n_coarse + n_kv, n_tokens]
    llm_attn_mask mask_early; // F16 [n_coarse + n_kv, n_tokens], without the cells of the ubatch (their K/V are not written yet)

    const llama_kvq_store * store = nullptr;

    // the sequence of the ubatch: its page table and query sums are the inputs of the graph (several: the streams of a packed ubatch,
    // their coarse rows and rows of the host store; n_coarse is the largest)
    int32_t seq = 0;
    std::vector<int32_t> seqs;
    std::vector<int32_t> n_coarse_s;
    std::vector<int32_t> n_host_s;
    ggml_tensor * bias_s = nullptr; // streams (S > 1): f32 [n_coarse, 1, 1, S], the candidates of the selection of each
    ggml_tensor * pt_s   = nullptr; // streams (S > 1): i32 [n_chunk, S], the page table of each

    // packed mode: the coarse region of the graph; fit mode: the graph sums the query statistics
    int32_t n_coarse = 0;
    int32_t n_host   = 0; // fit mode: the rows of the host store, the scratch attention reads them from there
    bool    fit      = false;
    uint32_t codes_gen = 0; // the views of the codes hold their device address

    // fit mode, large ubatch: the attention reads the scratch
    bool    scratch  = false;

    // scratch without host rows: the scratch rows are the cells (positions before the size of the ring), the mask of the cache
    bool    scr_cells = false;

    llm_attn_mask mask_scr; // F16 [scr_rows - scr_r0, n_tokens], scratch attention, row r = position r (see kvq_scr_rows)
    ggml_tensor * scr_set_pos = nullptr;
    int64_t       scr_rows = 0;
    int64_t       scr_r0   = 0; // see kvq_scr_r0

    const llama_memory_hybrid_kvq_context * mctx;

    llama_kvq_params params;
};

llama_model_qwen35::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    auto * inp_kvq = build_inp_kvq();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // packed two-tier attention: the attention layer ila prefetches at the input of layer max(0, ila - lead)
    const int kvq_lead = inp_kvq && inp_kvq->packed_small ? inp_kvq->params.lead : 0;

    // scratch attention: the upload of the first attention layer starts with the graph, the others after the previous attention
    if (inp_kvq && inp_kvq->scratch) {
        for (int il = 0; il < n_layer; ++il) {
            if (!hparams.is_recr(il)) {
                inp_kvq->store->build_scr_up(ctx0, gf, sched, il);
                break;
            }
        }
    }

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        for (int ila = il; kvq_lead > 0 && ila <= il + kvq_lead && ila < n_layer; ++ila) {
            if (!hparams.is_recr(ila) && std::max(0, ila - kvq_lead) == il) {
                build_kvq_prefetch(inp->get_attn(), inp_kvq, inpL, inp_pos, ila);
            }
        }

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), inp_kvq, cur, inp_pos, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv_z) {
        ggml_tensor * qkvz = build_lora_mm(model.layers[il].wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}


llama_model_qwen35::llm_graph_input_kvq * llama_model_qwen35::graph::build_inp_kvq() const {
    const auto * mctx_kvq = dynamic_cast<const llama_memory_hybrid_kvq_context *>(mctx);
    if (mctx_kvq == nullptr) {
        return nullptr;
    }

    // added also when the mode is off, so that can_reuse sees a switch
    auto inp = std::make_unique<llm_graph_input_kvq>(mctx_kvq);

    if (inp->params.enabled) {
        GGML_ASSERT(mctx_kvq->get_store() != nullptr);

        const int64_t n_kv = mctx_kvq->get_attn()->get_n_kv();
        const int64_t S    = ubatch.n_seqs_unq;
        GGML_ASSERT(S >= 1 && n_tokens % S == 0);

        inp->store        = mctx_kvq->get_store();
        inp->seq          = ubatch.seq_id_unq[0];
        inp->fit          = true;
        inp->codes_gen    = inp->store->codes_gen();
        bool all_packed = true;
        // the reserve of the packed decode graph: every stream at its largest coarse region (all rows of the store)
        const bool pk_reserve = mctx_kvq->packed_reserve();
        for (int64_t s = 0; s < S; ++s) {
            inp->seqs.push_back(ubatch.seq_id_unq[s]);
            inp->n_coarse_s.push_back(pk_reserve ? inp->store->n_rows() : inp->store->n_coarse(inp->seqs[s]));
            inp->n_host_s.push_back(pk_reserve ? inp->store->n_rows() : inp->store->n_host(inp->seqs[s]));
            all_packed &= llama_kvq_use_packed(inp->params, n_tokens/S, inp->n_coarse_s[s], inp->n_host_s[s]);
        }
        GGML_ASSERT(!pk_reserve || (all_packed && S > 1));
        inp->n_coarse     = *std::max_element(inp->n_coarse_s.begin(), inp->n_coarse_s.end());
        inp->n_host       = inp->n_host_s[0];
        inp->packed_small = all_packed;
        GGML_ASSERT((S == 1 || inp->packed_small) && "two-tier attention: a ubatch of several sequences is a packed one");
        inp->scratch      = kvq_use_scratch(inp->params, inp->store, inp->packed_small);
        inp->scr_cells    = inp->scratch && inp->n_host == 0;

        const int64_t n_mask = inp->n_coarse + n_kv;

        if (inp->packed_small) {
            inp->mask_attn.n_kv  = n_mask;
            inp->mask_early.n_kv = n_mask;
            if (S == 1) {
                inp->mask_attn.raw   = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_mask, n_tokens);
                inp->mask_early.raw  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_mask, n_tokens);
            } else {
                inp->mask_attn.raw   = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_mask, n_tokens/S, 1, S);
                inp->mask_early.raw  = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_mask, n_tokens/S, 1, S);
                const int64_t n_chunk = (inp->n_coarse + inp->store->n_chunk_rows() - 1)/inp->store->n_chunk_rows();
                inp->bias_s = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, inp->n_coarse, 1, 1, S);
                inp->pt_s   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_chunk, S);
                ggml_set_input(inp->bias_s);
                ggml_set_input(inp->pt_s);
                ggml_set_name(inp->bias_s, "kvq_bias_s");
                ggml_set_name(inp->pt_s,   "kvq_pt_s");
            }
            ggml_set_input(inp->mask_attn.raw);
            ggml_set_input(inp->mask_early.raw);
            ggml_set_name(inp->mask_attn.raw,  "kvq_mask_attn");
            ggml_set_name(inp->mask_early.raw, "kvq_mask_early");
        }
        if (inp->scratch) {
            if (!inp->scr_cells) {
                inp->scr_rows      = kvq_scr_rows(inp->store, inp->n_host, ubatch);
                inp->scr_r0        = kvq_scr_r0(inp->params, inp->n_host);
                inp->mask_scr.n_kv = inp->scr_rows - inp->scr_r0;
                inp->mask_scr.raw  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, inp->mask_scr.n_kv, n_tokens);
                ggml_set_input(inp->mask_scr.raw);
                ggml_set_name(inp->mask_scr.raw, "kvq_mask_scr");
            }
            if (inp->store->scr_tiled()) {
                inp->scr_set_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
                ggml_set_input(inp->scr_set_pos);
                ggml_set_name(inp->scr_set_pos, "kvq_scr_set_pos");
            }
        }
    } else if (mctx_kvq->get_store() != nullptr) {
        // reserve graph in fit mode: the statistics nodes of a large ubatch
        inp->store = mctx_kvq->get_store();
        inp->fit   = true;

        // and the scratch attention over the largest coarse region (flash attention keeps f16 copies of K/V in the compute buffer)
        if (inp->store->scratch_ok()) {
            inp->scratch       = true;
            inp->n_coarse      = inp->store->n_rows();
            inp->n_host        = inp->store->n_rows();
            inp->scr_rows      = inp->store->n_rows() + inp->store->n_ring();
            inp->scr_r0        = kvq_scr_r0(inp->params, inp->n_host);
            inp->mask_scr.n_kv = inp->scr_rows - inp->scr_r0;
            inp->mask_scr.raw  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, inp->mask_scr.n_kv, n_tokens);
            ggml_set_input(inp->mask_scr.raw);
            ggml_set_name(inp->mask_scr.raw, "kvq_mask_scr");
            if (inp->store->scr_tiled()) {
                inp->scr_set_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
                ggml_set_input(inp->scr_set_pos);
                ggml_set_name(inp->scr_set_pos, "kvq_scr_set_pos");
            }
        }
    }

    return (llm_graph_input_kvq *) res->add_input(std::move(inp));
}

void llama_model_qwen35::graph::build_kvq_prefetch(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_kvq *     inp_kvq,
        ggml_tensor *             x,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const auto &  layer       = model.layers[il];
    const auto *  store       = inp_kvq->store;
    const auto &  t           = store->get_layer(il);

    // layer il's query path on an earlier hidden state: input norm, query half of the q projection, q norm, RoPE
    ggml_tensor * cur = build_norm(x, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "kvq_eq_norm", il);

    ggml_tensor * qg   = nullptr;
    int64_t       nb_t = 0; // bytes per token of qg
    if (layer.wq) {
        qg = build_lora_mm(layer.wq, cur, layer.wq_s);
        if (layer.wq_b) {
            qg = ggml_add(ctx0, qg, layer.wq_b);
        }
    } else {
        GGML_ASSERT(layer.wqkv && !layer.wqkv_b);
        qg = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
    }
    nb_t = qg->nb[1];
    cb(qg, "kvq_eq_proj", il);

    ggml_tensor * q = ggml_view_3d(ctx0, qg, n_embd_head, n_head, n_tokens,
            ggml_element_size(qg)*n_embd_head*2, nb_t, 0);
    q = build_norm(q, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    q = build_rope(q, inp_pos, il);
    cb(q, "kvq_eq", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    const int64_t S = inp_kvq->seqs.size();
    ggml_tensor * k = store->get_ring(ctx0, il, false, inp->mctx->get_n_kv(), S);

    ggml_tensor * score = nullptr;
    if (S == 1) {
        score = ggml_kvq_attn(ctx0, q, store->get_kc(ctx0, il), nullptr, t.tab, store->get_pt(inp_kvq->seq), k, nullptr, nullptr, nullptr,
                inp_kvq->mask_early.raw, kq_scale, inp_kvq->n_coarse, store->n_chunk_rows(), GGML_KVQ_SCORE);
    } else {
        // the tokens of a stream are contiguous
        ggml_tensor * q4 = ggml_reshape_4d(ctx0, q, q->ne[0], q->ne[1], n_tokens/S, S);
        score = ggml_kvq_attn_streams(ctx0, q4, store->get_kc(ctx0, il), nullptr, t.tab, inp_kvq->pt_s, k, nullptr, nullptr, nullptr,
                inp_kvq->mask_early.raw, kq_scale, inp_kvq->n_coarse_s.data(), store->n_chunk_rows(), GGML_KVQ_SCORE);
    }
    cb(score, "kvq_eq_score", il);

    store->build_fetch(ctx0, gf, sched, *std::min_element(inp_kvq->n_coarse_s.begin(), inp_kvq->n_coarse_s.end()), il, score,
            inp_kvq->params.prefetch_n, inp_kvq->bias_s);
}

// fit mode: qacc[:, :, g] += sum of q q^T over the tokens and the query heads of kv head g
void llama_model_qwen35::graph::build_kvq_qstat(llm_graph_input_kvq * inp_kvq, ggml_tensor * q_all, int il) {
    // the sums of each sequence of the ubatch over its tokens (a stream: its contiguous tokens)
    const int64_t S = inp_kvq->seqs.size();
    for (int64_t s = 0; s < S; ++s) {
        ggml_tensor * acc = inp_kvq->store->get_qacc(inp_kvq->seqs[s], il);
        ggml_tensor * q   = S == 1 ? q_all : ggml_view_3d(ctx0, q_all, q_all->ne[0], q_all->ne[1], q_all->ne[2]/S, q_all->nb[1], q_all->nb[2], s*(q_all->ne[2]/S)*q_all->nb[2]);

        const int64_t D = q->ne[0];
        const int64_t G = acc->ne[2];
        const int64_t R = q->ne[1]/G;

        // q [D, R*G, T] -> [R*T, D, G], the tokens and heads of a kv head along dim 0
        ggml_tensor * x = ggml_reshape_4d(ctx0, q, D, R, G, q->ne[2]);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 2, 0, 3, 1));
        x = ggml_reshape_3d(ctx0, x, R*q->ne[2], D, G);

        ggml_tensor * gram = ggml_mul_mat(ctx0, x, x); // [D, D, G]
        cb(gram, "kvq_qgram", il);

        ggml_build_forward_expand(gf, ggml_add_inplace(ctx0, acc, gram));
    }
}

// fit mode: gacc += sum of sigmoid(gate) over the tokens
void llama_model_qwen35::graph::build_kvq_gstat(llm_graph_input_kvq * inp_kvq, ggml_tensor * gate_all, int il) {
    const int64_t S = inp_kvq->seqs.size();
    for (int64_t s = 0; s < S; ++s) {
        ggml_tensor * acc = inp_kvq->store->get_gacc(inp_kvq->seqs[s], il);
        ggml_tensor * gate_sigmoid = S == 1 ? gate_all : ggml_view_2d(ctx0, gate_all, gate_all->ne[0], gate_all->ne[1]/S, gate_all->nb[1], s*(gate_all->ne[1]/S)*gate_all->nb[1]);

        // one token (decode): the add only
        ggml_tensor * sum = gate_sigmoid;
        if (gate_sigmoid->ne[1] > 1) {
            sum = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, gate_sigmoid))); // [1, n_gate]
            cb(sum, "kvq_gsum", il);
        }

        ggml_build_forward_expand(gf, ggml_add_inplace(ctx0, acc, ggml_reshape_1d(ctx0, sum, acc->ne[0])));
    }
}

// packed two-tier attention: the coarse region through the packed rows, its resident slots precise, the tail precise from the cache
// the join of the prefetch scatters its rows before the attention, then one attention with the resident slots as they are
ggml_tensor * llama_model_qwen35::graph::build_attn_kvq_packed(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_kvq *     inp_kvq,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        float                     kq_scale,
        int                       il) {
    const auto * mctx_attn = inp->mctx;
    const auto * store     = inp_kvq->store;
    const auto & t         = store->get_layer(il);
    const auto & params    = inp_kvq->params;

    // the precise rows are the plain q8_0 cache rows
    GGML_ASSERT(!inp->self_k_rot[il] && !inp->self_v_rot[il]);
    GGML_ASSERT(!mctx_attn->get_k_scale(il) && !mctx_attn->get_v_scale(il));

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    // the cells are in the tail buffer of the layer (host ring)
    ggml_build_forward_expand(gf, store->build_ring_cpy(ctx0, il, false, k_cur, inp->get_k_idxs()));
    ggml_build_forward_expand(gf, store->build_ring_cpy(ctx0, il, true,  v_cur, inp->get_v_idxs()));

    const int64_t n_coarse = inp_kvq->n_coarse;
    const int64_t n_crow   = store->n_chunk_rows();

    const int64_t S = inp_kvq->seqs.size();
    GGML_ASSERT(n_coarse == *std::max_element(inp_kvq->n_coarse_s.begin(), inp_kvq->n_coarse_s.end()));

    // the whole ring, its mask has the cells after the coarse rows
    ggml_tensor * k    = store->get_ring(ctx0, il, false, mctx_attn->get_n_kv(), S);
    ggml_tensor * v    = store->get_ring(ctx0, il, true,  mctx_attn->get_n_kv(), S);
    ggml_tensor * kc   = store->get_kc(ctx0, il);
    ggml_tensor * vc   = store->get_vc(ctx0, il);
    ggml_tensor * ridx = store->get_ridx(ctx0, il, S);
    ggml_tensor * mask = inp_kvq->mask_attn.raw;

    // the rows of the prefetch of this layer, see build_kvq_prefetch
    if (params.lead > 0) {
        store->build_join(ctx0, gf, sched, il, params.prefetch_n, n_tokens/S, S);
    }

    ggml_tensor * out = nullptr;
    if (S == 1) {
        GGML_ASSERT(n_coarse == store->n_coarse(inp_kvq->seq));
        out = ggml_kvq_attn(ctx0, q_cur, kc, vc, t.tab, store->get_pt(inp_kvq->seq), k, v, store->n_streams() == 1 ? t.rkv : store->get_rkv(ctx0, il, 1), ridx, mask, kq_scale, (int) n_coarse, (int) n_crow, GGML_KVQ_ATTN);
    } else {
        ggml_tensor * q4 = ggml_reshape_4d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], n_tokens/S, S);
        out = ggml_kvq_attn_streams(ctx0, q4, kc, vc, t.tab, inp_kvq->pt_s, k, v, store->get_rkv(ctx0, il, S), ridx, mask, kq_scale, inp_kvq->n_coarse_s.data(), (int) n_crow, GGML_KVQ_ATTN);
    }
    cb(out, "kvq_attn", il);

    ggml_tensor * cur = ggml_reshape_2d(ctx0, out, out->ne[0]*out->ne[1], out->ne[2]*out->ne[3]);
    cb(cur, "kqv_out", il);

    return cur;
}

// the K/V of this ubatch go to the cache, the cells after the rows of the host store are copied to the scratch after them
// the copy stream uploads the rows of the host store (scratch up of the previous attention layer), the main stream joins it here
ggml_tensor * llama_model_qwen35::graph::build_attn_kvq_scratch(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_kvq *     inp_kvq,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             inp_pos,
        float                     kq_scale,
        int                       il) {
    const auto * mctx_attn = inp->mctx;
    const auto * store     = inp_kvq->store;

    GGML_ASSERT(!inp->self_k_rot[il] && !inp->self_v_rot[il]);
    GGML_ASSERT(!mctx_attn->get_k_scale(il) && !mctx_attn->get_v_scale(il));
    GGML_ASSERT(inp->self_attn_mask.rows == nullptr);

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    // the scratch rows: the host store has the rows before the ubatch, the rows of the ubatch go to the rows of their positions;
    // no host rows yet: the rows of the cells, in the order of the cells (no wrap)
    const int64_t n_host = inp_kvq->n_host;
    const int64_t n_rows = !inp_kvq->scr_cells ? inp_kvq->scr_rows : mctx_attn->get_n_kv();
    GGML_ASSERT((n_host == store->n_host(inp_kvq->seq) || !inp_kvq->params.enabled) && n_rows > n_host);

    // after the readback of the rows of the layer before (the same scratch rows)
    store->build_scr_join(ctx0, gf, sched, il);
    const bool tiled = store->scr_tiled();
    ggml_tensor * pos = tiled ? inp_kvq->scr_set_pos : ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
    ggml_build_forward_expand(gf, store->build_scr_set(ctx0, false, k_cur, pos));
    ggml_build_forward_expand(gf, store->build_scr_set(ctx0, true,  v_cur, pos));

    // the scratch is head-major, so its V view looks transposed to build_attn_mha: the flash attention part of it
    GGML_ASSERT(cparams.flash_attn);
    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);

    ggml_tensor * mask = !inp_kvq->scr_cells ? inp_kvq->mask_scr.raw : inp->self_attn_mask.raw;

    const float softcap = hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f;

    // the host rows [0, r0) in key blocks without a mask, then the rows [r0, n_rows) with the mask, so the f16 copies of K/V
    // are made per block; the partial results are merged by their log-sum-exp l: w = sigmoid(l_b - l), o += (o_b - o)*w, l += softplus(l_b - l)
    const int64_t r0 = !inp_kvq->scr_cells ? inp_kvq->scr_r0 : 0;
    const int64_t nb = tiled ? store->scr_tile_rows() : inp_kvq->params.scr_block/256*256;

    ggml_tensor * cur = nullptr;
    ggml_tensor * lse = nullptr;
    int32_t tile = 0;
    for (int64_t b0 = 0; b0 < n_rows; ) {
        const bool    last = b0 >= r0;
        const int64_t n    = last ? n_rows - b0 : std::min(nb, r0 - b0);
        if (tiled && !last && tile > 0) {
            ggml_build_forward_expand(gf, cur);
            ggml_build_forward_expand(gf, lse);
            store->build_scr_tile(ctx0, gf, sched, il, tile);
        }
        const int64_t scr_off = !tiled ? b0 : (last ? store->scr_tail_off() : (int64_t) (tile & 1)*nb);

        ggml_tensor * k = ggml_permute(ctx0, store->get_scr(ctx0, false, n, scr_off), 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(ctx0, store->get_scr(ctx0, true,  n, scr_off), 0, 2, 1, 3);

        if (r0 == 0) {
            cur = ggml_flash_attn_ext(ctx0, q, k, v, mask, kq_scale, hparams.f_max_alibi_bias, softcap);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            break;
        }

        ggml_tensor * fa = ggml_flash_attn_ext_lse(ctx0, q, k, v, last ? mask : nullptr, kq_scale, hparams.f_max_alibi_bias, softcap);
        ggml_prec_set_acc(fa, GGML_PREC_F32);

        ggml_tensor * o_b = ggml_view_3d(ctx0, fa, fa->ne[0], fa->ne[1], n_tokens, fa->nb[1], fa->nb[2], 0);
        ggml_tensor * l_b = ggml_view_3d(ctx0, fa, 1, fa->ne[1], n_tokens, sizeof(float), fa->ne[1]*sizeof(float), n_tokens*fa->nb[2]);
        if (cur == nullptr) {
            cur = o_b;
            lse = l_b;
        } else {
            ggml_tensor * d = ggml_sub(ctx0, l_b, lse);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, ggml_sub(ctx0, o_b, cur), ggml_sigmoid(ctx0, d)));
            if (!last) {
                lse = ggml_add(ctx0, lse, ggml_softplus(ctx0, d));
            }
        }
        b0 += n;
        tile++;
    }
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    ggml_build_forward_expand(gf, cur);
    cb(cur, "kqv_out", il);

    // the scratch is free for the next attention layer (host ring: after the readback of the rows of this one)
    int ila = il + 1;
    while (ila < n_layer && hparams.is_recr(ila)) {
        ila++;
    }
    if (ila < n_layer) {
        store->build_scr_up(ctx0, gf, sched, ila);
    } else {
        store->build_scr_rb(ctx0, gf, sched, il);
    }

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_kvq *     inp_kvq,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    if (inp_kvq && inp_kvq->fit) {
        build_kvq_qstat(inp_kvq, Qcur, il);
    }

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (inp_kvq && inp_kvq->packed_small) {
        cur = build_attn_kvq_packed(inp, inp_kvq, Qcur, Kcur, Vcur, kq_scale, il);
    } else if (inp_kvq && inp_kvq->scratch) {
        cur = build_attn_kvq_scratch(inp, inp_kvq, Qcur, Kcur, Vcur, inp_pos, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    if (inp_kvq && inp_kvq->fit && inp_kvq->store->get_gacc(inp_kvq->seq, il)) {
        build_kvq_gstat(inp_kvq, gate_sigmoid, il);
    }

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    if (model.layers[il].ssm_beta_alpha_cat) {
        ggml_tensor * ba = build_lora_mm(model.layers[il].ssm_beta_alpha_cat, cur);
        cb(ba, "beta_alpha", il);

        beta  = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);

        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);
    cb(alpha, "alpha", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);

        // a Hadamard-latent table stores rotated rows; restore the primal basis
        if (hdmd) {
            if (const auto it = hdmd->inv.find(tok_embd_w); it != hdmd->inv.end()) {
                tok_embd = llama_mul_mat_hadamard(ctx0, tok_embd, it->second.rot);
                if (it->second.signs) {
                    tok_embd = ggml_mul(ctx0, tok_embd, it->second.signs);
                }
            }
        }
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = build_rope(Qcur, inp_pos, il);
    Kcur = build_rope(Kcur, inp_pos, il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
