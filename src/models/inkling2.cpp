#include "models.h"
#include "../llama-kvq-store.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid-kvq.h"

#include <algorithm>
#include <cmath>
#include <cstring>

static int64_t inkling_kvq_scr_rows(const llama_kvq_store * store, int32_t n_host, const llama_ubatch & ubatch) {
    llama_pos p_max = n_host - 1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        p_max = std::max(p_max, ubatch.pos[i]);
    }
    return std::min<int64_t>((int64_t) n_host + store->n_ring(), GGML_PAD((int64_t) p_max + 1, 256));
}

static int64_t inkling_kvq_scr_r0(const llama_kvq_params & p, int32_t n_host, int32_t rel_extent) {
    const int32_t hi = std::max(0, n_host - rel_extent);
    return p.scr_block >= 256 ? hi/256*256 : 0;
}

static bool inkling_kvq_has_layer(const llama_kvq_store * store, int32_t il) {
    const auto & ids = store->layer_ids();
    return std::find(ids.begin(), ids.end(), il) != ids.end();
}

static ggml_tensor * inkling_build_kq_mask_raw(ggml_context * ctx0, int64_t n_kv, int64_t n_tokens, int64_t n_stream, const char * name) {
    ggml_tensor * raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(raw);
    ggml_set_name(raw, name);
    return raw;
}

static bool inkling_kvq_use_pos_mask(const llama_hparams & hparams, const llama_cparams & cparams, const llama_ubatch & ubatch) {
    static const bool enabled = getenv("LLAMA_ATTN_POS_MASK_DISABLE") == nullptr;

    return enabled && cparams.flash_attn && cparams.causal_attn && !hparams.use_alibi &&
        hparams.swa_type == LLAMA_SWA_TYPE_STANDARD && !ubatch.is_pos_2d() &&
        (!cparams.kv_unified || ubatch.n_seqs_unq == 1);
}

// two-tier KVQ graph inputs for inkling2 (SWA cache inputs + KVQ packed/scratch masks and positions)
class llama_model_inkling2::llm_graph_input_kvq : public llm_graph_input_i {
public:
    llm_graph_input_kvq(
            const llama_memory_hybrid_kvq_context * mctx,
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            llama_kvq_store * store,
            const llama_kvq_params & params,
            const std::vector<int32_t> & seqs,
            const std::vector<int32_t> & n_coarse_s,
            int32_t n_host,
            bool packed_small,
            bool scratch,
            bool scr_cells,
            bool fit,
            bool pos_mask,
            bool swa_scratch,
            int64_t swa_old_n,
            int64_t scr_rows,
            int64_t scr_r0) :
        mctx(mctx),
        hparams(hparams),
        cparams(cparams),
        store(store),
        params(params),
        seqs(seqs),
        n_coarse_s(n_coarse_s),
        seq(seqs[0]),
        n_coarse(*std::max_element(n_coarse_s.begin(), n_coarse_s.end())),
        n_host(n_host),
        packed_small(packed_small),
        scratch(scratch),
        scr_cells(scr_cells),
        fit(fit),
        pos_mask(pos_mask),
        swa_scratch(swa_scratch),
        swa_old_n(swa_old_n),
        codes_gen(store ? store->codes_gen() : 0),
        scr_rows(scr_rows),
        scr_r0(scr_r0) {}

    void set_input(const llama_ubatch * ubatch) override {
        const auto * mctx_swa = mctx->get_swa();
        if (mctx_swa) {
            if (swa_scratch) {
                const llama_ubatch & ub_swa = mctx_swa->get_ubatch();
                if (swa_k_idxs) {
                    mctx_swa->set_input_k_idxs(swa_k_idxs, &ub_swa);
                }
                if (swa_v_idxs) {
                    mctx_swa->set_input_v_idxs(swa_v_idxs, &ub_swa);
                }
                mctx->set_input_swa_scratch(swa_mask.raw, swa_kv_pos, ubatch, hparams.n_swa);
            } else {
                if (swa_k_idxs) {
                    mctx_swa->set_input_k_idxs(swa_k_idxs, ubatch);
                }
                if (swa_v_idxs) {
                    mctx_swa->set_input_v_idxs(swa_v_idxs, ubatch);
                }
                if (swa_mask.raw) {
                    mctx_swa->set_input_attn_mask(swa_mask, ubatch, cparams.causal_attn);
                }
                if (swa_kv_pos) {
                    mctx_swa->set_input_kv_pos(swa_kv_pos, ubatch, pos_mask);
                }
            }
        }
        if (q_pos && q_pos->buffer) {
            GGML_ASSERT(q_pos->type == GGML_TYPE_I32 && q_pos->ne[0] == ubatch->n_tokens);
            ggml_backend_tensor_set(q_pos, ubatch->pos, 0, ubatch->n_tokens*sizeof(int32_t));
        }
        if (packed_small) {
            mctx->set_input_kvq_ring(mask_attn.raw, mask_early.raw, ubatch, n_coarse_s);
            mctx->set_input_kvq_ring_tail(mask_tail.raw, kpos_tail, ubatch, n_coarse_s);
            if (pt_s && pt_s->buffer) {
                const int64_t n_chunk = pt_s->ne[0];
                const int64_t S = seqs.size();
                std::vector<int32_t> data(n_chunk*S, 0);
                for (int64_t s = 0; s < S; ++s) {
                    const int64_t nck = (n_coarse_s[s] + store->n_chunk_rows() - 1)/store->n_chunk_rows();
                    for (int64_t c = 0; c < nck; ++c) {
                        data[s*n_chunk + c] = store->page_of_chunk(seqs[s], (int32_t) c);
                    }
                }
                ggml_backend_tensor_set(pt_s, data.data(), 0, data.size()*sizeof(int32_t));
            }
            if (bias_s && bias_s->buffer) {
                const int64_t S = seqs.size();
                const int32_t n_sink = params.n_sink;
                std::vector<float> data((size_t) n_coarse*S, 0.0f);
                for (int64_t s = 0; s < S; ++s) {
                    float * d = data.data() + s*n_coarse;
                    for (int32_t j = 0; j < std::min(n_sink, n_coarse); ++j) {
                        d[j] = -INFINITY;
                    }
                    for (int32_t j = n_coarse_s[s]; j < n_coarse; ++j) {
                        d[j] = -INFINITY;
                    }
                }
                ggml_backend_tensor_set(bias_s, data.data(), 0, data.size()*sizeof(float));
            }
            mctx->get_store()->begin_ubatch(*ubatch);
        }
        if (scratch) {
            if (!scr_cells && mask_scr.raw) {
                mctx->set_input_kvq_scr(mask_scr.raw, ubatch, n_host, scr_r0);
            }
            if (kpos_scr && kpos_scr->buffer) {
                if (scr_cells) {
                    mctx->get_attn()->set_input_kv_pos(kpos_scr, ubatch, pos_mask);
                } else if (pos_mask) {
                    mctx->set_input_kvq_scr_pos(kpos_scr, ubatch, n_host, scr_r0);
                } else {
                    const int64_t n = kpos_scr->ne[0];
                    std::vector<int32_t> pos(n);
                    for (int64_t i = 0; i < n; ++i) {
                        pos[i] = (int32_t) (scr_r0 + i);
                    }
                    ggml_backend_tensor_set(kpos_scr, pos.data(), 0, n*sizeof(int32_t));
                }
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

    bool can_reuse(const llm_graph_params & p) override {
        const auto * mc = dynamic_cast<const llama_memory_hybrid_kvq_context *>(p.mctx);
        if (mc == nullptr || mc->get_store() != store) {
            return false;
        }
        if (pos_mask != inkling_kvq_use_pos_mask(hparams, p.cparams, p.ubatch)) {
            return false;
        }
        const bool cur_swa_scr = mc->swa_is_scratch(p.ubatch);
        if (cur_swa_scr != swa_scratch) {
            return false;
        }
        if (swa_scratch) {
            if ((int64_t) mc->swa_old_n(p.ubatch) != swa_old_n) {
                return false;
            }
            if (swa_kv_pos && swa_kv_pos->ne[0] != swa_old_n + p.ubatch.n_tokens) {
                return false;
            }
        } else if (swa_kv_pos && mc->get_swa() && mc->get_swa()->get_n_kv() != swa_kv_pos->ne[0]) {
            return false;
        }
        if (mc->packed_reserve() || store == nullptr) {
            return false;
        }
        const llama_kvq_params & pr = mc->get_params();
        if (pr.enabled != params.enabled || pr.prefetch_n != params.prefetch_n || pr.lead != params.lead ||
                pr.max_batch != params.max_batch || pr.scr_block != params.scr_block) {
            return false;
        }
        if (codes_gen != store->codes_gen()) {
            return false;
        }
        if (p.ubatch.n_seqs_unq != seqs.size() || p.ubatch.n_tokens % p.ubatch.n_seqs_unq != 0) {
            return false;
        }
        std::vector<int32_t> cur_seq(p.ubatch.n_seqs_unq);
        std::vector<int32_t> cur_nc(p.ubatch.n_seqs_unq);
        for (uint32_t s = 0; s < p.ubatch.n_seqs_unq; ++s) {
            cur_seq[s] = p.ubatch.seq_id_unq[s];
            cur_nc[s]  = store->n_coarse(cur_seq[s]);
        }
        const int32_t cur_nh = store->n_host(cur_seq[0]);
        const bool cur_ps  = pr.enabled && !mc->prompt_path(p.ubatch);
        const bool cur_scr = pr.enabled && !cur_ps && store->scratch_ok();
        if (cur_ps != packed_small || cur_scr != scratch || cur_seq != seqs) {
            return false;
        }
        if (packed_small) {
            if (cur_nc != n_coarse_s || mask_attn.raw->ne[0] != n_coarse + mc->get_attn()->get_n_kv() ||
                    mask_attn.raw->ne[1] != (int64_t) (p.ubatch.n_tokens/seqs.size())) {
                return false;
            }
        }
        if (scratch) {
            const bool cur_cells = cur_nh == 0;
            const int64_t cur_rows = cur_cells ? mc->get_attn()->get_n_kv() : inkling_kvq_scr_rows(store, cur_nh, p.ubatch);
            if (cur_cells != scr_cells || cur_nh != n_host || cur_rows != scr_rows) {
                return false;
            }
            if (!scr_cells && mask_scr.raw && mask_scr.raw->ne[1] != p.ubatch.n_tokens) {
                return false;
            }
            if (kpos_scr && kpos_scr->ne[0] != cur_rows - scr_r0) {
                return false;
            }
            if (scr_set_pos && scr_set_pos->ne[0] != p.ubatch.n_tokens) {
                return false;
            }
        }
        mctx = mc;
        return true;
    }

    const llama_memory_hybrid_kvq_context * mctx;
    const llama_hparams & hparams;
    const llama_cparams & cparams;
    llama_kvq_store * store;
    llama_kvq_params  params;

    std::vector<int32_t> seqs;
    std::vector<int32_t> n_coarse_s;
    int32_t seq;
    int32_t n_coarse;
    int32_t n_host;
    bool packed_small;
    bool scratch;
    bool scr_cells;
    bool fit;
    bool pos_mask;
    bool swa_scratch;
    int64_t swa_old_n;
    uint32_t codes_gen;
    int64_t scr_rows;
    int64_t scr_r0;

    // SWA cache inputs
    ggml_tensor * swa_k_idxs = nullptr;
    ggml_tensor * swa_v_idxs = nullptr;
    llm_attn_mask swa_mask;
    ggml_tensor * swa_kv_pos = nullptr;
    ggml_tensor * q_pos      = nullptr;

    // KVQ packed and scratch inputs
    llm_attn_mask mask_attn;
    llm_attn_mask mask_early;
    llm_attn_mask mask_tail;
    ggml_tensor * kpos_tail = nullptr;
    ggml_tensor * pt_s      = nullptr;
    ggml_tensor * bias_s    = nullptr;
    llm_attn_mask mask_scr;
    ggml_tensor * kpos_scr    = nullptr;
    ggml_tensor * scr_set_pos = nullptr;
};

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    prec_policy.set_acc(output, GGML_PREC_F32);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold, int il) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    // the threshold tensor only depends on the shape: fill it once per graph and layer device
    ggml_tensor *& limit = clip_limits[{ { model.dev_layer(il), norm->ne[1] }, threshold }];
    if (!limit) {
        limit = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, norm->ne[0], norm->ne[1], norm->ne[2], norm->ne[3]), threshold);
    }

    ggml_tensor * factor = ggml_div(ctx0, limit, norm);

    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

ggml_tensor * llama_model_inkling2::graph::get_zero_sinks(int64_t n_head, int il) {
    ggml_tensor *& zero = clip_limits[{ { model.dev_layer(il), -n_head }, 0.0f }];
    if (!zero) {
        zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
    }
    return zero;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// the normalized blocks, newest first, [n_embd, n, n_tokens] and transposed [n, n_embd, n_tokens]
// a block is normalized once per graph, and each block list is stacked once and shared by every mix over it
const llama_model_inkling2::graph::block_stack * llama_model_inkling2::graph::stack(const std::vector<ggml_tensor *> & blocks, int il) {
    if (blocks.empty()) {
        return nullptr;
    }
    // one stack per device, so that a split does not pull another device's stack
    const auto key = std::make_pair(il < 0 ? nullptr : model.dev_layer(il), blocks);
    auto it = block_stacks.find(key);
    if (it != block_stacks.end()) {
        return &it->second;
    }
    const std::vector<ggml_tensor *> prefix(blocks.begin(), blocks.end() - 1);
    const block_stack * prev = stack(prefix, il);

    ggml_tensor * normed   = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, blocks.back(), n_embd, 1, n_tokens), 1e-6f);
    ggml_tensor * normed_t = ggml_reshape_3d(ctx0, normed, 1, n_embd, n_tokens);

    block_stack st;
    st.states   = prev ? ggml_concat(ctx0, normed,   prev->states,   1) : normed;
    st.states_t = prev ? ggml_concat(ctx0, normed_t, prev->states_t, 0) : normed_t;
    return &block_stacks.emplace(key, st).first->second;
}

ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query, int il) {
    // rows [blocks.back(), ..., blocks.front(), cur], each RMS-normalized; shared by every mix over the same blocks and cur
    const ggml_backend_dev_t dev = il < 0 ? nullptr : model.dev_layer(il);
    auto key = std::make_pair(dev, blocks);
    key.second.push_back(cur);
    block_stack & all = mix_states[key];
    if (!all.states) {
        ggml_tensor * normed   = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), 1e-6f);
        ggml_tensor * normed_t = ggml_reshape_3d(ctx0, normed, 1, n_embd, n_tokens);

        const block_stack * st = stack(blocks, il);
        all.states   = st ? ggml_concat(ctx0, st->states,   normed,   1) : normed;
        all.states_t = st ? ggml_concat(ctx0, st->states_t, normed_t, 0) : normed_t;
    }
    ggml_tensor * states   = all.states;
    ggml_tensor * states_t = all.states_t;

    // one token: the query broadcasts as is
    ggml_tensor * queries = n_tokens == 1 ? query :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, states_t, scores);
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    // the inputs first, so that the conv chain from the concat to the scale is consecutive in the graph
    ggml_build_forward_expand(gf, cur);
    if (dynamic) {
        ggml_build_forward_expand(gf, dynamic);
    }

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        // the inputs channel-major once, [n_channels, n_conv - 1 + n_seq_tokens, n_seqs]: each lag is a row offset
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_4d(ctx0, dynamic, 1, n_heads, n_seq_tokens, n_seqs,
                    dynamic->nb[1], dynamic->nb[2], n_seq_tokens*dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_4d(ctx0, xt, n_channels/n_heads, n_heads, n_seq_tokens, n_seqs,
                    (n_channels/n_heads)*xt->nb[0], xt->nb[1], xt->nb[2], (n_conv - 1 - lag)*xt->nb[1]);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// cur_norm and message_norm are cur and message RMS-normalized
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * cur_norm, ggml_tensor * message, ggml_tensor * message_norm,
        const loop_layer & loop, int ctrl) {
    ggml_tensor * gate = ggml_concat(ctx0, cur_norm, message_norm, 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

llama_model_inkling2::llm_graph_input_kvq * llama_model_inkling2::graph::build_inp_kvq() const {
    const auto * mctx_kvq = dynamic_cast<const llama_memory_hybrid_kvq_context *>(mctx);
    if (mctx_kvq == nullptr) {
        return nullptr;
    }

    llama_kvq_store * store = mctx_kvq->get_store();
    const llama_kvq_params & pr = mctx_kvq->get_params();
    const bool packed_res = mctx_kvq->packed_reserve();
    const bool fit = store != nullptr;

    std::vector<int32_t> seqs;
    std::vector<int32_t> n_coarse_s;
    int32_t n_host = 0;
    if (packed_res) {
        const uint32_t S = ubatch.n_seqs_unq;
        seqs.resize(S);
        n_coarse_s.resize(S, store ? store->n_rows() : 0);
        for (uint32_t s = 0; s < S; ++s) {
            seqs[s] = (int32_t) s;
        }
    } else if (ubatch.n_seqs_unq > 0 && ubatch.seq_id_unq != nullptr) {
        const uint32_t S = ubatch.n_seqs_unq;
        seqs.resize(S);
        n_coarse_s.resize(S, 0);
        for (uint32_t s = 0; s < S; ++s) {
            seqs[s] = ubatch.seq_id_unq[s];
            if (store) {
                n_coarse_s[s] = store->n_coarse(seqs[s]);
            }
        }
        if (store) {
            n_host = store->n_host(seqs[0]);
        }
    } else {
        seqs.push_back(0);
        n_coarse_s.push_back(0);
    }

    const bool packed_small = packed_res || (pr.enabled && store != nullptr && !mctx_kvq->prompt_path(ubatch));
    const bool scr_reserve  = !packed_small && !pr.enabled && store != nullptr && store->scratch_ok();
    const bool scratch      = (!packed_small && pr.enabled && store != nullptr && store->scratch_ok()) || scr_reserve;
    const bool scr_cells    = scratch && !scr_reserve && n_host == 0;
    if (scr_reserve) {
        n_host = store->n_rows();
    }
    const int64_t scr_rows  = scratch ? (scr_reserve ? store->n_rows() + store->n_ring() : (scr_cells ? mctx_kvq->get_attn()->get_n_kv() : inkling_kvq_scr_rows(store, n_host, ubatch))) : 0;
    const int64_t scr_r0    = (scratch && !scr_cells) ? inkling_kvq_scr_r0(pr, n_host, (int32_t) hparams.n_relative_extent) : 0;

    const bool pos_mask = inkling_kvq_use_pos_mask(hparams, cparams, ubatch);
    const bool swa_scratch = mctx_kvq->swa_is_scratch(ubatch);
    const int64_t swa_old_n = swa_scratch ? (int64_t) mctx_kvq->swa_old_n(ubatch) : 0;

    auto inp_q = std::make_unique<llm_graph_input_kvq>(mctx_kvq, hparams, cparams, store, pr, seqs, n_coarse_s, n_host, packed_small, scratch, scr_cells, fit, pos_mask, swa_scratch, swa_old_n, scr_rows, scr_r0);

    const auto * mctx_swa = mctx_kvq->get_swa();
    const int64_t n_stream = ubatch.n_seqs_unq > 0 ? ubatch.n_seqs_unq : 1;
    if (mctx_swa) {
        if (swa_scratch) {
            const int64_t n_keep = std::min<int64_t>(n_tokens, hparams.n_swa);
            inp_q->swa_k_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_keep);
            ggml_set_input(inp_q->swa_k_idxs);
            ggml_set_name(inp_q->swa_k_idxs, "kvq_swa_k_idxs");
            inp_q->swa_v_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_keep);
            ggml_set_input(inp_q->swa_v_idxs);
            ggml_set_name(inp_q->swa_v_idxs, "kvq_swa_v_idxs");
            const int64_t n_cat = swa_old_n + n_tokens;
            if (!pos_mask) {
                inp_q->swa_mask.raw = inkling_build_kq_mask_raw(ctx0, n_cat, n_tokens, 1, "kvq_swa_mask");
            }
            inp_q->swa_kv_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_cat, 1);
            ggml_set_input(inp_q->swa_kv_pos);
            ggml_set_name(inp_q->swa_kv_pos, "kvq_swa_kv_pos");
        } else {
            inp_q->swa_k_idxs = mctx_swa->build_input_k_idxs(ctx0, ubatch);
            inp_q->swa_v_idxs = mctx_swa->build_input_v_idxs(ctx0, ubatch);
            if (!pos_mask) {
                inp_q->swa_mask.raw = inkling_build_kq_mask_raw(ctx0, mctx_swa->get_n_kv(), n_tokens, n_stream, "kvq_swa_mask");
            }
            inp_q->swa_kv_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, mctx_swa->get_n_kv(), n_stream);
            ggml_set_input(inp_q->swa_kv_pos);
            ggml_set_name(inp_q->swa_kv_pos, "kvq_swa_kv_pos");
        }
    }

    inp_q->q_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_q->q_pos);
    ggml_set_name(inp_q->q_pos, "kvq_q_pos");

    if (packed_small) {
        const int64_t S = seqs.size();
        const int64_t n_kv = mctx_kvq->get_attn()->get_n_kv();
        inp_q->mask_attn.raw = inkling_build_kq_mask_raw(ctx0, inp_q->n_coarse + n_kv, n_tokens, S, "kvq_mask");
        if (pr.lead > 0) {
            inp_q->mask_early.raw = inkling_build_kq_mask_raw(ctx0, inp_q->n_coarse + n_kv, n_tokens, S, "kvq_mask_early");
        }
        inp_q->mask_tail.raw = inkling_build_kq_mask_raw(ctx0, n_kv, n_tokens, S, "kvq_mask_tail");
        inp_q->kpos_tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, S);
        ggml_set_input(inp_q->kpos_tail);
        ggml_set_name(inp_q->kpos_tail, "kvq_kpos_tail");
        if (S > 1) {
            const int64_t n_chunk = (inp_q->n_coarse + store->n_chunk_rows() - 1)/store->n_chunk_rows();
            inp_q->pt_s = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, std::max<int64_t>(1, n_chunk), S);
            ggml_set_input(inp_q->pt_s);
            ggml_set_name(inp_q->pt_s, "kvq_pt_s");
            if (pr.lead > 0) {
                inp_q->bias_s = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, inp_q->n_coarse, 1, 1, S);
                ggml_set_input(inp_q->bias_s);
                ggml_set_name(inp_q->bias_s, "kvq_bias_s");
            }
        }
    }

    if (scratch) {
        const int64_t n_last = scr_rows - scr_r0;
        if (!scr_cells && !pos_mask) {
            inp_q->mask_scr.raw = inkling_build_kq_mask_raw(ctx0, n_last, n_tokens, 1, "kvq_mask_scr");
        }
        inp_q->kpos_scr = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_last, 1);
        ggml_set_input(inp_q->kpos_scr);
        ggml_set_name(inp_q->kpos_scr, "kvq_kpos_scr");
        if (store->scr_tiled()) {
            inp_q->scr_set_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
            ggml_set_input(inp_q->scr_set_pos);
            ggml_set_name(inp_q->scr_set_pos, "kvq_scr_set_pos");
        }
    }

    return (llm_graph_input_kvq *) res->add_input(std::move(inp_q));
}

int llama_model_inkling2::graph::kvq_lead() const {
    if (!inp_kvq || !inp_kvq->packed_small || inp_kvq->params.lead <= 0) {
        return 0;
    }
    return std::min(3, inp_kvq->params.lead);
}

void llama_model_inkling2::graph::build_kvq_prefetch(int il_tgt, ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks,
        const std::vector<ggml_tensor *> * bank, ggml_tensor * bank_out, const loop_layer * loop_tgt,
        ggml_tensor * precomputed_msg, ggml_tensor * precomputed_msg_norm) {
    if (!inp_kvq || !inp_kvq->packed_small || inp_kvq->params.lead <= 0) {
        return;
    }
    llama_kvq_store * store = inp_kvq->store;
    if (!store || !inkling_kvq_has_layer(store, il_tgt)) {
        return;
    }

    const auto & layer = model.layers[il_tgt];
    const auto & t     = store->get_layer(il_tgt);
    const int64_t n_head      = hparams.n_head(il_tgt);
    const int64_t n_embd_head = hparams.n_embd_head_k(il_tgt);
    const int64_t S           = inp_kvq->seqs.size();
    const int64_t L           = n_tokens/S;
    const int64_t n_kv        = inp_hyb->get_attn()->mctx->get_n_kv();
    const float   kq_scale    = 1.0f/sqrtf((float) n_embd_head);

    ggml_tensor * eq_in = cur;
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query, il_tgt);
        eq_in = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], 0));
    }
    if (loop_tgt && (precomputed_msg || (bank && bank_out))) {
        ggml_tensor * msg = precomputed_msg ? precomputed_msg : mix(*bank, bank_out, loop_tgt->msg_query, il_tgt);
        ggml_tensor * mq  = ggml_view_2d(ctx0, msg, n_embd, n_tokens, msg->nb[2], LOOP_CTRL_Q*msg->nb[1]);
        ggml_tensor * mq_norm = precomputed_msg_norm
            ? ggml_view_2d(ctx0, precomputed_msg_norm, n_embd, n_tokens, precomputed_msg_norm->nb[2], LOOP_CTRL_Q*precomputed_msg_norm->nb[1])
            : ggml_rms_norm(ctx0, ggml_cont(ctx0, mq), 1e-6f);
        eq_in = control(eq_in, ggml_rms_norm(ctx0, eq_in, 1e-6f), mq, mq_norm, *loop_tgt, LOOP_CTRL_Q);
    }

    eq_in = build_norm(eq_in, layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il_tgt);
    ggml_tensor * q = build_lora_mm(layer.wq, eq_in);
    q = clip(ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens), layer.attn_q_clip, hparams.f_qk_clip, il_tgt);
    q = ggml_mul(ctx0, q, tau);

    ggml_tensor * k_rg = store->get_ring(ctx0, il_tgt, false, n_kv, S);
    ggml_tensor * score = nullptr;
    if (S == 1) {
        score = ggml_kvq_attn(ctx0, q, store->get_kc(ctx0, il_tgt), nullptr, t.tab, store->get_pt(inp_kvq->seq), k_rg,
                nullptr, nullptr, nullptr, inp_kvq->mask_early.raw, kq_scale, inp_kvq->n_coarse, store->n_chunk_rows(), GGML_KVQ_SCORE);
    } else {
        ggml_tensor * q4 = ggml_reshape_4d(ctx0, q, n_embd_head, n_head, L, S);
        score = ggml_kvq_attn_streams(ctx0, q4, store->get_kc(ctx0, il_tgt), nullptr, t.tab, inp_kvq->pt_s, k_rg,
                nullptr, nullptr, nullptr, inp_kvq->mask_early.raw, kq_scale, inp_kvq->n_coarse_s.data(), store->n_chunk_rows(), GGML_KVQ_SCORE);
    }
    cb(score, "kvq_eq_score", il_tgt);

    store->build_fetch(ctx0, gf, sched, *std::min_element(inp_kvq->n_coarse_s.begin(), inp_kvq->n_coarse_s.end()), il_tgt, score,
            inp_kvq->params.prefetch_n, inp_kvq->bias_s);
}

void llama_model_inkling2::graph::build_kvq_qstat(ggml_tensor * q_all, int il) {
    llama_kvq_store * store = inp_kvq->store;
    if (!store || !inkling_kvq_has_layer(store, il)) {
        return;
    }
    const int64_t S = inp_kvq->seqs.size();
    for (int64_t s = 0; s < S; ++s) {
        ggml_tensor * acc = store->get_qacc(inp_kvq->seqs[s], il);
        if (!acc) {
            continue;
        }
        ggml_tensor * q = S == 1 ? q_all : ggml_view_3d(ctx0, q_all, q_all->ne[0], q_all->ne[1], q_all->ne[2]/S, q_all->nb[1], q_all->nb[2], s*(q_all->ne[2]/S)*q_all->nb[2]);

        const int64_t D = q->ne[0];
        const int64_t G = acc->ne[2];
        const int64_t R = q->ne[1]/G;

        ggml_tensor * x = ggml_reshape_4d(ctx0, q, D, R, G, q->ne[2]);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 2, 0, 3, 1));
        x = ggml_reshape_3d(ctx0, x, R*q->ne[2], D, G);

        ggml_tensor * gram = ggml_mul_mat(ctx0, x, x); // [D, D, G]
        cb(gram, "kvq_qgram", il);
        ggml_build_forward_expand(gf, ggml_add_inplace(ctx0, acc, gram));
    }
}

void llama_model_inkling2::graph::build_kvq_gstat(ggml_tensor * gate_all, int il) {
    llama_kvq_store * store = inp_kvq->store;
    if (!store || !inkling_kvq_has_layer(store, il)) {
        return;
    }
    const int64_t S = inp_kvq->seqs.size();
    for (int64_t s = 0; s < S; ++s) {
        ggml_tensor * acc = store->get_gacc(inp_kvq->seqs[s], il);
        if (!acc) {
            continue;
        }
        ggml_tensor * gs = S == 1 ? gate_all : ggml_view_2d(ctx0, gate_all, gate_all->ne[0], gate_all->ne[1]/S, gate_all->nb[1], s*(gate_all->ne[1]/S)*gate_all->nb[1]);
        ggml_tensor * sum = gs;
        if (gs->ne[1] > 1) {
            sum = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, gs)));
            cb(sum, "kvq_gsum", il);
        }
        ggml_build_forward_expand(gf, ggml_add_inplace(ctx0, acc, ggml_reshape_1d(ctx0, sum, acc->ne[0])));
    }
}

ggml_tensor * llama_model_inkling2::graph::build_attn_kvq_packed(
        ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,
        ggml_tensor * raw_table, ggml_tensor * sinks_raw, float kq_scale, int il) {
    llama_kvq_store * store = inp_kvq->store;
    const auto & t  = store->get_layer(il);
    const auto & pr = inp_kvq->params;
    const int64_t S = inp_kvq->seqs.size();
    const int64_t L = n_tokens/S;
    const int64_t D = q_cur->ne[0];
    const int64_t n_head = q_cur->ne[1];
    const int64_t n_kv   = inp_hyb->get_attn()->mctx->get_n_kv();

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    // store new K/V into the host ring tail
    ggml_build_forward_expand(gf, store->build_ring_cpy(ctx0, il, false, k_cur, inp_hyb->get_attn()->get_k_idxs()));
    ggml_build_forward_expand(gf, store->build_ring_cpy(ctx0, il, true,  v_cur, inp_hyb->get_attn()->get_v_idxs()));

    if (pr.lead > 0) {
        store->build_join(ctx0, gf, sched, il, pr.prefetch_n, L, S);
    }

    ggml_tensor * kc   = store->get_kc(ctx0, il);
    ggml_tensor * vc   = store->get_vc(ctx0, il);
    ggml_tensor * rkv  = store->get_rkv(ctx0, il, S);
    ggml_tensor * ridx = store->get_ridx(ctx0, il, S);

    // 1. Coarse + resident attention (all coarse rows have dist >= 1024 -> raw bias is 0, no mask needed)
    ggml_tensor * fa_c = nullptr;
    if (S == 1) {
        fa_c = ggml_kvq_attn(ctx0, q_cur, kc, vc, t.tab, store->get_pt(inp_kvq->seq), nullptr, nullptr, rkv, ridx, nullptr,
                kq_scale, inp_kvq->n_coarse, store->n_chunk_rows(), GGML_KVQ_ATTN_LSE);
    } else {
        ggml_tensor * q4 = ggml_reshape_4d(ctx0, q_cur, D, n_head, L, S);
        fa_c = ggml_kvq_attn_streams(ctx0, q4, kc, vc, t.tab, inp_kvq->pt_s, nullptr, nullptr, rkv, ridx, nullptr,
                kq_scale, inp_kvq->n_coarse_s.data(), store->n_chunk_rows(), GGML_KVQ_ATTN_LSE);
    }

    // 2. Ring tail attention with relative-position bias (per-stream when S > 1 since flash_attn_ext_lse requires ne[3] == 1)
    ggml_tensor * k_ring = store->get_ring(ctx0, il, false, n_kv, S);
    ggml_tensor * v_ring = store->get_ring(ctx0, il, true,  n_kv, S);
    const int64_t n_ext  = raw_table->ne[0];

    ggml_tensor * cur = nullptr;
    for (int64_t s = 0; s < S; ++s) {
        ggml_tensor * o_c = ggml_view_3d(ctx0, fa_c, D, n_head, L, fa_c->nb[1], fa_c->nb[2], s*fa_c->nb[3]);
        ggml_tensor * l_c = ggml_view_3d(ctx0, fa_c, 1, n_head, L, sizeof(float), n_head*sizeof(float), s*fa_c->nb[3] + L*fa_c->nb[2]);

        ggml_tensor * qs = S > 1
            ? ggml_view_3d(ctx0, q_cur, D, n_head, L, q_cur->nb[1], q_cur->nb[2], s*L*q_cur->nb[2])
            : q_cur;
        ggml_tensor * q_fa = ggml_permute(ctx0, qs, 0, 2, 1, 3); // [D, L, n_head]
        ggml_tensor * ks   = S > 1 ? ggml_view_3d(ctx0, k_ring, D, n_kv, k_ring->ne[2], k_ring->nb[1], k_ring->nb[2], s*k_ring->nb[3]) : k_ring;
        ggml_tensor * vs   = S > 1 ? ggml_view_3d(ctx0, v_ring, D, n_kv, v_ring->ne[2], v_ring->nb[1], v_ring->nb[2], s*v_ring->nb[3]) : v_ring;
        ggml_tensor * ms   = S > 1 ? ggml_view_2d(ctx0, inp_kvq->mask_tail.raw, n_kv, L, inp_kvq->mask_tail.raw->nb[1], s*inp_kvq->mask_tail.raw->nb[3]) : inp_kvq->mask_tail.raw;
        ggml_tensor * ts   = S > 1 ? ggml_cont(ctx0, ggml_view_3d(ctx0, raw_table, n_ext, n_head, L, raw_table->nb[1], raw_table->nb[2], s*L*raw_table->nb[2])) : raw_table;
        ggml_tensor * kps  = S > 1 ? ggml_view_1d(ctx0, inp_kvq->kpos_tail, n_kv, s*inp_kvq->kpos_tail->nb[1]) : ggml_reshape_1d(ctx0, inp_kvq->kpos_tail, n_kv);
        ggml_tensor * qps  = S > 1 ? ggml_view_1d(ctx0, inp_kvq->q_pos, L, s*L*sizeof(int32_t)) : inp_kvq->q_pos;

        ggml_tensor * fa_t = ggml_flash_attn_ext_lse(ctx0, q_fa, ks, vs, ms, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_add_rel_bias(fa_t, ts, kps, qps);
        ggml_prec_set_acc(fa_t, GGML_PREC_F32);

        ggml_tensor * o_t = ggml_view_3d(ctx0, fa_t, D, n_head, L, fa_t->nb[1], fa_t->nb[2], 0);
        ggml_tensor * l_t = ggml_view_3d(ctx0, fa_t, 1, n_head, L, sizeof(float), n_head*sizeof(float), L*fa_t->nb[2]);

        // merge coarse+resident (o_c, l_c) and ring tail (o_t, l_t)
        ggml_tensor * d_lt = ggml_sub(ctx0, l_t, l_c);
        ggml_tensor * o_s  = ggml_add(ctx0, o_c, ggml_mul(ctx0, ggml_sub(ctx0, o_t, o_c), ggml_sigmoid(ctx0, d_lt)));
        ggml_tensor * l_s  = ggml_add(ctx0, l_c, ggml_softplus(ctx0, d_lt));

        // fold virtual attention sink: o *= sigmoid(lse - sink)
        ggml_tensor * sk_s = S > 1
            ? ggml_view_3d(ctx0, sinks_raw, 1, n_head, L, sizeof(float), n_head*sizeof(float), s*L*sinks_raw->nb[1])
            : ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, L);
        o_s = ggml_mul(ctx0, o_s, ggml_sigmoid(ctx0, ggml_sub(ctx0, l_s, sk_s)));

        cur = cur ? ggml_concat(ctx0, cur, o_s, 2) : o_s;
    }

    return ggml_reshape_2d(ctx0, cur, D*n_head, n_tokens);
}

ggml_tensor * llama_model_inkling2::graph::build_attn_kvq_scratch(
        ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,
        ggml_tensor * raw_table, ggml_tensor * sinks_raw, float kq_scale, int il) {
    llama_kvq_store * store = inp_kvq->store;
    const auto & pr = inp_kvq->params;
    const int64_t D = q_cur->ne[0];
    const int64_t n_head = q_cur->ne[1];
    const int64_t n_rows = !inp_kvq->scr_cells ? inp_kvq->scr_rows : inp_hyb->get_attn()->mctx->get_n_kv();

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    store->build_scr_join(ctx0, gf, sched, il);
    const bool tiled = store->scr_tiled();
    ggml_tensor * pos = tiled ? inp_kvq->scr_set_pos : ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
    ggml_build_forward_expand(gf, store->build_scr_set(ctx0, false, k_cur, pos));
    ggml_build_forward_expand(gf, store->build_scr_set(ctx0, true,  v_cur, pos));

    GGML_ASSERT(cparams.flash_attn);
    ggml_tensor * q_fa = ggml_permute(ctx0, q_cur, 0, 2, 1, 3); // [D, n_tokens, n_head]
    const int64_t r0   = !inp_kvq->scr_cells ? inp_kvq->scr_r0 : 0;
    const int64_t nb   = tiled ? store->scr_tile_rows() : (pr.scr_block >= 256 ? pr.scr_block/256*256 : n_rows);

    ggml_tensor * mask_last = inp_kvq->pos_mask ? nullptr : (!inp_kvq->scr_cells ? inp_kvq->mask_scr.raw : inp_hyb->get_attn()->self_attn_mask.raw);
    if (mask_last && mask_last->type != GGML_TYPE_F16) {
        mask_last = ggml_cast(ctx0, mask_last, GGML_TYPE_F16);
    }

    ggml_tensor * cur = nullptr;
    ggml_tensor * lse = nullptr;
    int32_t tile = 0;
    for (int64_t b0 = 0; b0 < n_rows; ) {
        const bool    last = b0 >= r0;
        const int64_t n    = last ? (n_rows - b0) : std::min(nb, r0 - b0);
        if (tiled && !last && tile > 0) {
            ggml_build_forward_expand(gf, cur);
            ggml_build_forward_expand(gf, lse);
            store->build_scr_tile(ctx0, gf, sched, il, tile);
        }
        const int64_t scr_off = !tiled ? b0 : (last ? store->scr_tail_off() : (int64_t) (tile & 1)*nb);
        ggml_tensor * kb   = ggml_permute(ctx0, store->get_scr(ctx0, false, n, scr_off), 0, 2, 1, 3);
        ggml_tensor * vb   = ggml_permute(ctx0, store->get_scr(ctx0, true,  n, scr_off), 0, 2, 1, 3);

        if (r0 == 0) {
            ggml_tensor * table = ggml_sub_inplace(ctx0, raw_table, ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, n_tokens));
            ggml_tensor * zero_sinks = get_zero_sinks(n_head, il);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx0, q_fa, kb, vb, mask_last, kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(fa, zero_sinks);
            ggml_flash_attn_ext_add_rel_bias(fa, table, ggml_reshape_1d(ctx0, inp_kvq->kpos_scr, n), inp_kvq->q_pos);
            if (inp_kvq->pos_mask) {
                ggml_flash_attn_ext_set_pos_window(fa, INT32_MAX);
            }
            ggml_prec_set_acc(fa, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx0, fa, D*n_head, n_tokens);
            break;
        }

        ggml_tensor * fa = ggml_flash_attn_ext_lse(ctx0, q_fa, kb, vb, last ? mask_last : nullptr, kq_scale, 0.0f, 0.0f);
        if (last) {
            ggml_flash_attn_ext_add_rel_bias(fa, raw_table, ggml_reshape_1d(ctx0, inp_kvq->kpos_scr, n), inp_kvq->q_pos);
            if (inp_kvq->pos_mask) {
                ggml_flash_attn_ext_set_pos_window(fa, INT32_MAX);
            }
        }
        ggml_prec_set_acc(fa, GGML_PREC_F32);

        ggml_tensor * ob = ggml_view_3d(ctx0, fa, D, n_head, n_tokens, fa->nb[1], fa->nb[2], 0);
        ggml_tensor * lb = ggml_view_3d(ctx0, fa, 1, n_head, n_tokens, sizeof(float), n_head*sizeof(float), n_tokens*fa->nb[2]);

        if (!cur) {
            cur = ob;
            lse = lb;
        } else {
            ggml_tensor * dl = ggml_sub(ctx0, lb, lse);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, ggml_sub(ctx0, ob, cur), ggml_sigmoid(ctx0, dl)));
            lse = ggml_add(ctx0, lse, ggml_softplus(ctx0, dl));
        }
        b0 += n;
        tile++;
    }

    if (r0 > 0) {
        ggml_tensor * s_3d = ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, n_tokens);
        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, ggml_sub(ctx0, lse, s_3d)));
        cur = ggml_reshape_2d(ctx0, cur, D*n_head, n_tokens);
    }

    ggml_build_forward_expand(gf, cur);
    cb(cur, "kqv_out", il);

    // trigger upload of next full-attention layer in execution order, or readback after the last one
    const auto & ids = store->layer_ids();
    auto it = std::find(ids.begin(), ids.end(), il);
    if (it != ids.end() && it + 1 != ids.end()) {
        store->build_scr_up(ctx0, gf, sched, *(it + 1));
    } else {
        store->build_scr_rb(ctx0, gf, sched, il);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    if (dynamic_cast<const llama_memory_hybrid_kvq_context *>(mctx)) {
        inp_hyb = build_inp_mem_hybrid();
        recr    = inp_hyb->get_recr();
        inp_kvq = build_inp_kvq();
        if (inp_kvq && inp_kvq->scratch) {
            inp_kvq->store->build_scr_up(ctx0, gf, sched, inp_kvq->store->layer_ids()[0]);
        }
    } else {
        inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
        attn = inp->get_attn();
        recr = inp->get_recr();
    }

    inp_pos     = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop, ggml_tensor * precomputed_messages_norm) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens], and the messages RMS-normalized once
    ggml_tensor * messages_norm = precomputed_messages_norm ? precomputed_messages_norm
        : (messages ? ggml_rms_norm(ctx0, messages, 1e-6f) : nullptr);
    auto message = [&](ggml_tensor * t, int ctrl) {
        return ggml_view_2d(ctx0, t, n_embd, n_tokens, t->nb[2], ctrl*t->nb[1]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query, il);
        if (is_swa) {
            inputs[0] = inputs[1] = inputs[2] = ggml_reshape_2d(ctx0, mixed, n_embd, n_tokens);
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]));
            }
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        ggml_tensor * inputs_norm[3] = {};
        for (int j = 0; j < 3; ++j) {
            // a SWA layer normalizes its one input once
            inputs_norm[j] = j > 0 && inputs[j] == inputs[0] ? inputs_norm[0] : ggml_rms_norm(ctx0, inputs[j], 1e-6f);
        }
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], inputs_norm[j], message(messages, LOOP_CTRL_Q + j), message(messages_norm, LOOP_CTRL_Q + j),
                    *loop, LOOP_CTRL_Q + j);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        ggml_build_forward_expand(gf, Kcur);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
        ggml_build_forward_expand(gf, Vcur);
        ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip, il);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip, il);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip, il);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        static const bool kvq_force_decode_stat = []() {
            const char * env = std::getenv("LLAMA_KVQ_FORCE_DECODE_STAT");
            return env != nullptr && std::atoi(env) != 0;
        }();
        const bool do_kvq_stat = !is_swa && inp_kvq && inp_kvq->fit &&
            (kvq_force_decode_stat || n_tokens > 32 || inp_kvq->n_coarse == 0);
        if (do_kvq_stat) {
            build_kvq_qstat(Qcur, il);
        }

        ggml_tensor * sinks_raw = build_lora_mm(layer.attn_sink_proj, inputs[0]);

        // project r on the bias table, the attention gathers it by distance
        // the sink logit t is per query: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))), with a zero sink
        ggml_tensor * raw_table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
        const float   kq_scale  = 1.0f/sqrtf((float) n_embd_head);

        if (inp_kvq) {
            if (is_swa) {
                ggml_tensor * table = ggml_sub_inplace(ctx0, raw_table, ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, n_tokens));
                cb(table, "rel_table", il);
                ggml_tensor * zero_sinks = get_zero_sinks(n_head, il);

                const auto * mctx_swa = inp_kvq->mctx->get_swa();
                const int32_t n_window = inp_kvq->pos_mask ? (int32_t) hparams.n_swa : 0;

                if (inp_kvq->swa_scratch) {
                    const int64_t n_old  = inp_kvq->swa_old_n;
                    const int64_t n_cat  = n_old + n_tokens;
                    const int64_t n_keep = std::min<int64_t>(n_tokens, hparams.n_swa);

                    // per-layer scratch [D, n_head_kv, n_old + n_tokens] reused across all SWA layers
                    ggml_tensor * k_new = ggml_cast(ctx0, Kcur, mctx_swa->type_k());
                    ggml_tensor * v_new = ggml_cast(ctx0, Vcur, mctx_swa->type_v());
                    ggml_tensor * k_cat = n_old > 0 ? ggml_concat(ctx0, mctx_swa->get_k(ctx0, il, (uint32_t) n_old), k_new, 2) : k_new;
                    ggml_tensor * v_cat = n_old > 0 ? ggml_concat(ctx0, mctx_swa->get_v(ctx0, il, (uint32_t) n_old), v_new, 2) : v_new;

                    const llm_graph_rel_bias rel = { table, inp_kvq->swa_kv_pos, inp_kvq->q_pos, n_window };
                    cur = build_attn_mha(Qcur, k_cat, v_cat, nullptr, inp_kvq->swa_mask.raw, nullptr, zero_sinks, nullptr, n_cat, kq_scale, il, &rel);
                    ggml_build_forward_expand(gf, cur);

                    // persist only the trailing n_keep tokens after old SWA cells were copied into k_cat/v_cat
                    const int64_t off = n_tokens - n_keep;
                    ggml_tensor * k_keep = ggml_view_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_keep, Kcur->nb[1], Kcur->nb[2], off*Kcur->nb[2]);
                    ggml_tensor * v_keep = ggml_view_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_keep, Vcur->nb[1], Vcur->nb[2], off*Vcur->nb[2]);
                    ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, k_keep, inp_kvq->swa_k_idxs, il));
                    ggml_build_forward_expand(gf, mctx_swa->cpy_v(ctx0, v_keep, inp_kvq->swa_v_idxs, il));
                } else {
                    ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, Kcur, inp_kvq->swa_k_idxs, il));
                    ggml_build_forward_expand(gf, mctx_swa->cpy_v(ctx0, Vcur, inp_kvq->swa_v_idxs, il));

                    const int64_t n_kv_swa = mctx_swa->get_n_kv();
                    ggml_tensor * k = mctx_swa->get_k(ctx0, il);
                    ggml_tensor * v = mctx_swa->get_v(ctx0, il);

                    const llm_graph_rel_bias rel = { table, inp_kvq->swa_kv_pos, inp_kvq->q_pos, n_window };
                    cur = build_attn_mha(Qcur, k, v, nullptr, inp_kvq->swa_mask.raw, nullptr, zero_sinks, nullptr, n_kv_swa, kq_scale, il, &rel);
                }
            } else if (inp_kvq->packed_small) {
                cur = build_attn_kvq_packed(Qcur, Kcur, Vcur, raw_table, sinks_raw, kq_scale, il);
            } else if (inp_kvq->scratch) {
                cur = build_attn_kvq_scratch(Qcur, Kcur, Vcur, raw_table, sinks_raw, kq_scale, il);
            } else {
                // fallback (reserve graph / KVQ disabled): standard ring attention with rel bias
                ggml_tensor * table = ggml_sub_inplace(ctx0, raw_table, ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, n_tokens));
                cb(table, "rel_table", il);
                ggml_tensor * zero_sinks = get_zero_sinks(n_head, il);

                const auto * mctx_attn = inp_hyb->get_attn()->mctx;
                ggml_build_forward_expand(gf, mctx_attn->cpy_k(ctx0, Kcur, inp_hyb->get_attn()->get_k_idxs(), il));
                ggml_build_forward_expand(gf, mctx_attn->cpy_v(ctx0, Vcur, inp_hyb->get_attn()->get_v_idxs(), il));

                const int64_t n_kv = mctx_attn->get_n_kv();
                ggml_tensor * k = mctx_attn->get_k(ctx0, il);
                ggml_tensor * v = mctx_attn->get_v(ctx0, il);

                ggml_tensor * kq_mask = inp_hyb->get_attn()->self_attn_mask.raw;
                if (kq_mask->type != GGML_TYPE_F16) {
                    kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
                }
                const int64_t n_stream = k->ne[3];
                ggml_tensor * kpos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
                const llm_graph_rel_bias rel = { table, kpos, inp_kvq->q_pos };
                cur = build_attn_mha(Qcur, k, v, nullptr, kq_mask, nullptr, zero_sinks, nullptr, n_kv, kq_scale, il, &rel);
            }
        } else {
            ggml_tensor * table = ggml_sub_inplace(ctx0, raw_table, ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, n_tokens));
            cb(table, "rel_table", il);

            ggml_tensor * zero_sinks = get_zero_sinks(n_head, il);

            cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, table, zero_sinks, kq_scale, il);
        }

        ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0]));
        if (do_kvq_stat) {
            build_kvq_gstat(gate_sigmoid, il);
        }
        cur = ggml_mul(ctx0, cur, gate_sigmoid);
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa);
        ggml_build_forward_expand(gf, cur);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query, il), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, ggml_rms_norm(ctx0, cur, 1e-6f), message(messages, LOOP_CTRL_FFN), message(messages_norm, LOOP_CTRL_FFN),
                *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
    ggml_build_forward_expand(gf, cur);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, ffn_inp, cur);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // DSpark target features: the last one is the final norm output of every token
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    const uint32_t block_sz = hparams.attn_res_block_size;
    const int lead = kvq_lead();
    for (int il = 0; il < n_layer; ++il) {
        if (lead > 0 && il + lead < n_layer && !hparams.is_swa(il + lead) && il / block_sz == (il + lead) / block_sz) {
            build_kvq_prefetch(il + lead, cur, blocks);
        }
        // DSpark target features: the partial block at the layer input
        res->t_layer_inp[il] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
