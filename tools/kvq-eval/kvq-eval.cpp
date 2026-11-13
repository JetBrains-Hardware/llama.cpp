// teacher-forced KL harness for KV cache experiments
//
// input: int32 token file ids[0..e], ids[e] = <|im_end|>, and the prefix length s (ids[s] = <|im_start|>, ids[s+1] = assistant)
// prefill ids[0:s], then feed ids[s:e] in chunks with logits at every position
// fed position i predicts ids[s+i+1]; the first 2 fed positions are not scored, so the scored targets are ids[s+3..e]
//
// --kvq-variant replaces the cached K/V of the prefix after the prefill (fake quantization, response K/V stay exact):
//   exact          read the cache and write it back unchanged (sanity)
//   q8_0           llama.cpp Q8_0 round trip (blocks of 32, no rotation)
//   aff8           8-bit affine round trip, one f16 scale and offset per 64 channels
//   K<b>V<b>, K<b>, V<b>
//                  transform codec at b bits (src/llama-kvq-codec.h), tables fitted on the prefix:
//                  K metric = second moment of the queries at the last --kvq-qwin prefix positions
//                  V metric = o_proj weight scaled by the mean sigmoid(gate) over the prefix
//                  the first --kvq-sinks and the last --kvq-recent prefix positions stay exact
//
// --kvq-two-tier keeps two copies of the prefix K/V and attends with both (see llama_kvq_params):
// the precise copy is --kvq-precise (aff8 or exact), the coarse copy is --kvq-variant
//   all-precise    every prefix cell is precise
//   all-coarse     every prefix cell is coarse
//   none           the protected cells (the first --kvq-sinks and the last --kvq-protect prefix positions) are precise
//   <N>            the protected cells and the top N cells of the coarse attention per kv head and query position

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "gguf.h"

#include "../../src/llama-ext.h"
#include "../../src/llama-kvq-codec.h"

#include <algorithm>
#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const char    kvq_magic[8] = { 'k', 'v', 'q', 'r', 'e', 'f', '0', '1' };
static const int32_t kvq_n_skip   = 2;

enum kvq_variant_type {
    KVQ_VARIANT_NONE,
    KVQ_VARIANT_EXACT,
    KVQ_VARIANT_Q8_0,
    KVQ_VARIANT_AFF8,
    KVQ_VARIANT_CODEC,
};

struct kvq_args {
    std::string tokens_path;
    std::string ref_out_path;
    std::string ref_in_path;
    std::string ref2_in_path; // a second reference, compared after the first
    std::string csv_path;
    std::string tables_out_path;
    std::string tables_in_path;
    int32_t     prefix_len = -1;
    int32_t     chunk      = 64;

    kvq_variant_type variant = KVQ_VARIANT_NONE;

    int32_t k_bits = 0; // 0: exact
    int32_t v_bits = 0;
    int32_t sinks  = 4;
    int32_t recent = 0;
    int32_t q_win  = 2048;
    bool    stats  = false;

    bool    two_tier        = false;
    bool    tt_all_precise  = false;
    bool    tt_all_coarse   = false;
    int32_t tt_top          = 0;
    int32_t tt_protect      = 512;
    bool    tt_precise_aff8 = true;
    bool    tt_stats        = false;
};

static bool kvq_parse_two_tier(const std::string & v, kvq_args & args) {
    args.two_tier = true;
    if (v == "all-precise") {
        args.tt_all_precise = true;
        return true;
    }
    if (v == "all-coarse") {
        args.tt_all_coarse = true;
        return true;
    }
    if (v == "none") {
        return true;
    }
    char extra = 0;
    return sscanf(v.c_str(), "%d%c", &args.tt_top, &extra) == 1 && args.tt_top > 0;
}

static bool kvq_parse_variant(const std::string & v, kvq_args & args) {
    if (v == "exact") {
        args.variant = KVQ_VARIANT_EXACT;
        return true;
    }
    if (v == "q8_0") {
        args.variant = KVQ_VARIANT_Q8_0;
        return true;
    }
    if (v == "aff8") {
        args.variant = KVQ_VARIANT_AFF8;
        return true;
    }
    int  kb    = 0;
    int  vb    = 0;
    char extra = 0;
    if (sscanf(v.c_str(), "K%dV%d%c", &kb, &vb, &extra) == 2 ||
        sscanf(v.c_str(), "K%d%c",    &kb,      &extra) == 1 ||
        sscanf(v.c_str(), "V%d%c",    &vb,      &extra) == 1) {
        if (kb < 0 || kb > 8 || vb < 0 || vb > 8 || kb + vb == 0) {
            return false;
        }
        args.variant = KVQ_VARIANT_CODEC;
        args.k_bits  = kb;
        args.v_bits  = vb;
        return true;
    }
    return false;
}

// stats of one scored position
struct kvq_pos_stat {
    double kl;
    double nll_ref;
    double nll_var;
    bool   top1;
};

// remove the tool args from argv, the rest goes to common_params_parse
static bool kvq_parse_args(int & argc, char ** argv, kvq_args & args) {
    int n = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const bool has_val = i + 1 < argc;
        if (a == "--kvq-tokens" && has_val) {
            args.tokens_path = argv[++i];
        } else if (a == "--kvq-prefix-len" && has_val) {
            args.prefix_len = std::stoi(argv[++i]);
        } else if (a == "--kvq-chunk" && has_val) {
            args.chunk = std::stoi(argv[++i]);
        } else if (a == "--kvq-save-ref" && has_val) {
            args.ref_out_path = argv[++i];
        } else if (a == "--kvq-ref" && has_val) {
            args.ref_in_path = argv[++i];
        } else if (a == "--kvq-ref2" && has_val) {
            args.ref2_in_path = argv[++i];
        } else if (a == "--kvq-csv" && has_val) {
            args.csv_path = argv[++i];
        } else if (a == "--kvq-variant" && has_val) {
            if (!kvq_parse_variant(argv[++i], args)) {
                LOG_ERR("%s: bad variant '%s'\n", __func__, argv[i]);
                return false;
            }
        } else if (a == "--kvq-sinks" && has_val) {
            args.sinks = std::stoi(argv[++i]);
        } else if (a == "--kvq-recent" && has_val) {
            args.recent = std::stoi(argv[++i]);
        } else if (a == "--kvq-qwin" && has_val) {
            args.q_win = std::stoi(argv[++i]);
        } else if (a == "--kvq-save-tables" && has_val) {
            args.tables_out_path = argv[++i];
        } else if (a == "--kvq-load-tables" && has_val) {
            args.tables_in_path = argv[++i];
        } else if (a == "--kvq-stats") {
            args.stats = true;
        } else if (a == "--kvq-two-tier" && has_val) {
            if (!kvq_parse_two_tier(argv[++i], args)) {
                LOG_ERR("%s: bad two-tier mode '%s'\n", __func__, argv[i]);
                return false;
            }
        } else if (a == "--kvq-protect" && has_val) {
            args.tt_protect = std::stoi(argv[++i]);
        } else if (a == "--kvq-precise" && has_val) {
            const std::string v = argv[++i];
            if (v != "aff8" && v != "exact") {
                LOG_ERR("%s: bad precise copy '%s'\n", __func__, v.c_str());
                return false;
            }
            args.tt_precise_aff8 = v == "aff8";
        } else if (a == "--kvq-tt-stats") {
            args.tt_stats = true;
        } else if (a.rfind("--kvq-", 0) == 0) {
            LOG_ERR("%s: unknown or incomplete argument '%s'\n", __func__, a.c_str());
            return false;
        } else {
            argv[n++] = argv[i];
        }
    }
    argc = n;

    if (args.tokens_path.empty() || args.prefix_len < 0) {
        LOG_ERR("usage: %s -m model.gguf --kvq-tokens ids.bin --kvq-prefix-len s [--kvq-chunk 64] [--kvq-save-ref ref.bin] [--kvq-ref ref.bin [--kvq-csv out.csv] [--kvq-ref2 ref2.bin]]\n"
                "       [--kvq-variant exact|q8_0|aff8|K<b>V<b>|K<b>|V<b>] [--kvq-sinks 4] [--kvq-recent 0] [--kvq-qwin 2048]\n"
                "       [--kvq-save-tables tables.bin] [--kvq-load-tables tables.bin] [--kvq-stats]\n"
                "       [--kvq-two-tier all-precise|all-coarse|none|<N> [--kvq-protect 512] [--kvq-precise aff8|exact] [--kvq-tt-stats]] [common args]\n", argv[0]);
        return false;
    }
    if (args.chunk <= 0) {
        LOG_ERR("%s: --kvq-chunk must be > 0\n", __func__);
        return false;
    }
    if (args.sinks < 0 || args.recent < 0 || args.q_win <= 0) {
        LOG_ERR("%s: bad --kvq-sinks, --kvq-recent or --kvq-qwin\n", __func__);
        return false;
    }
    if ((!args.tables_out_path.empty() || !args.tables_in_path.empty() || args.stats) && args.variant != KVQ_VARIANT_CODEC) {
        LOG_ERR("%s: --kvq-save-tables, --kvq-load-tables and --kvq-stats need a codec variant\n", __func__);
        return false;
    }
    // the coarse copy codes the whole prefix, the precise cells are chosen by --kvq-protect
    if (args.two_tier && (args.variant == KVQ_VARIANT_NONE || args.recent != 0 || args.tt_protect < 0)) {
        LOG_ERR("%s: --kvq-two-tier needs a --kvq-variant for the coarse copy, no --kvq-recent and --kvq-protect >= 0\n", __func__);
        return false;
    }
    if (args.tt_stats && !args.two_tier) {
        LOG_ERR("%s: --kvq-tt-stats needs --kvq-two-tier\n", __func__);
        return false;
    }
    return true;
}

static bool kvq_read_tokens(const std::string & path, std::vector<llama_token> & ids) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        LOG_ERR("%s: failed to open %s\n", __func__, path.c_str());
        return false;
    }
    const size_t size = f.tellg();
    if (size == 0 || size % sizeof(int32_t) != 0) {
        LOG_ERR("%s: bad token file size %zu\n", __func__, size);
        return false;
    }
    ids.resize(size / sizeof(int32_t));
    f.seekg(0);
    f.read((char *) ids.data(), size);
    return (bool) f;
}

// ids[s] = <|im_start|>, ids[s+1] = assistant, ids[e] = <|im_end|>
static bool kvq_check_segment(const llama_vocab * vocab, const std::vector<llama_token> & ids, int32_t s) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t : ids) {
        if (t < 0 || t >= n_vocab) {
            LOG_ERR("%s: token id %d out of vocab range %d\n", __func__, t, n_vocab);
            return false;
        }
    }
    const int32_t e = (int32_t) ids.size() - 1;
    if (s < 1 || e - s - kvq_n_skip < 1) {
        LOG_ERR("%s: bad segment s = %d, e = %d\n", __func__, s, e);
        return false;
    }
    const std::string p_start = common_token_to_piece(vocab, ids[s]);
    const std::string p_asst  = common_token_to_piece(vocab, ids[s + 1]);
    const std::string p_end   = common_token_to_piece(vocab, ids[e]);
    if (p_start != "<|im_start|>" || p_asst != "assistant" || p_end != "<|im_end|>") {
        LOG_ERR("%s: unexpected segment tokens: ids[s] = %d, ids[s+1] = %d, ids[e] = %d\n", __func__, ids[s], ids[s + 1], ids[e]);
        return false;
    }
    return true;
}

// stable log-softmax in double, returns the argmax
static int32_t kvq_log_softmax(const float * logits, int32_t n_vocab, float * out) {
    int32_t imax = 0;
    for (int32_t v = 1; v < n_vocab; v++) {
        if (logits[v] > logits[imax]) {
            imax = v;
        }
    }
    const double max = logits[imax];
    double sum = 0.0;
    for (int32_t v = 0; v < n_vocab; v++) {
        sum += std::exp((double) logits[v] - max);
    }
    const double lse = max + std::log(sum);
    for (int32_t v = 0; v < n_vocab; v++) {
        out[v] = (float) ((double) logits[v] - lse);
    }
    return imax;
}

static int32_t kvq_argmax(const float * x, int32_t n) {
    return (int32_t) (std::max_element(x, x + n) - x);
}

// fill the memory with ids[0:n_prefix], no logits
static bool kvq_prefill(llama_context * ctx, const std::vector<llama_token> & ids, int32_t n_prefix, int32_t n_batch) {
    common_batch batch(ctx);
    for (int32_t i = 0; i < n_prefix; i++) {
        batch.add(ids[i], i, 0, false);
    }
    for (int32_t i = 0; i < n_prefix; i += n_batch) {
        const int32_t n = std::min(n_batch, n_prefix - i);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(i, n))) {
            LOG_ERR("%s: decode failed at pos %d\n", __func__, i);
            return false;
        }
    }
    llama_synchronize(ctx);
    return true;
}

// feed ids[s:e] in chunks, call on_chunk(i0, n) after each chunk of fed positions [i0, i0 + n)
// and on_pos(i, logits) for each fed position i >= kvq_n_skip
template <typename F, typename G>
static bool kvq_score(llama_context * ctx, const std::vector<llama_token> & ids, int32_t s, int32_t chunk, F && on_pos, G && on_chunk) {
    const int32_t e = (int32_t) ids.size() - 1;
    common_batch batch(ctx);
    for (int32_t p = s; p < e; p += chunk) {
        const int32_t n = std::min(chunk, e - p);
        batch.clear();
        for (int32_t j = 0; j < n; j++) {
            batch.add(ids[p + j], p + j, 0, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: decode failed at pos %d\n", __func__, p);
            return false;
        }
        if (!on_chunk(p - s, n)) {
            return false;
        }
        for (int32_t j = 0; j < n; j++) {
            const int32_t i = p + j - s;
            if (i >= kvq_n_skip && !on_pos(i, llama_get_logits_ith(ctx, j))) {
                return false;
            }
        }
    }
    return true;
}

// prefill statistics for the codec metrics, filled by the eval callback
struct kvq_capture {
    bool    enabled     = false;
    int32_t il_first    = -1; // the query of this layer starts a new ubatch
    int32_t n_head      = 0;
    int32_t n_embd_head = 0;
    int32_t win0        = 0;  // queries of positions [win0, win1) are kept
    int32_t win1        = 0;
    int64_t pos0        = 0;  // first position of the current ubatch
    int64_t pos_next    = 0;

    std::map<int32_t, std::vector<float>>  q;        // [win1 - win0][n_head][n_embd_head], after q_norm and RoPE
    std::map<int32_t, std::vector<double>> gate_sum; // [n_head*n_embd_head], sum of sigmoid(gate) over the prefix
    std::map<int32_t, int64_t>             gate_n;

    // two-tier statistics of the last chunk, tensor kvq_stats-<il> (see build_attn_kvq in src/models/qwen35.cpp)
    bool tt_stats = false;
    std::map<int32_t, std::vector<float>> tt;

    std::vector<float> buf;
};

// two-tier statistics of one layer, summed over the scored positions, each as a mean over the query heads
struct kvq_tt_stat {
    double c_pre  = 0.0; // coarse weight on the prefix
    double c_cand = 0.0; // coarse weight on the prefix cells that are not protected
    double c_top  = 0.0; // coarse weight on the chosen cells
    double w_pre  = 0.0; // weight on the precise prefix cells
    double w_resp = 0.0; // weight on the cells after the prefix (always precise)
    int64_t n     = 0;
};

// adds the positions j >= j0 of the last chunk of n positions
static bool kvq_tt_add(const kvq_capture & cap, std::map<int32_t, kvq_tt_stat> & acc, int32_t n, int32_t j0, bool has_top, int32_t n_head, int32_t n_head_kv) {
    const int64_t n_c = (int64_t) n*n_head_kv;
    const int64_t n_w = (int64_t) n*n_head;
    for (const auto & it : cap.tt) {
        const auto & v = it.second;
        if ((int64_t) v.size() != (has_top ? 3 : 2)*n_c + 2*n_w) {
            LOG_ERR("%s: layer %d: %zu stats values for a chunk of %d\n", __func__, it.first, v.size(), n);
            return false;
        }
        const float * c_pre  = v.data();
        const float * c_cand = c_pre + n_c;
        const float * c_top  = has_top ? c_cand + n_c : nullptr;
        const float * w_pre  = c_cand + (has_top ? 2 : 1)*n_c;
        const float * w_resp = w_pre + n_w;

        auto & a = acc[it.first];
        // the index of position j and head h is j + n*h
        for (int32_t j = j0; j < n; j++) {
            for (int32_t g = 0; g < n_head_kv; g++) {
                a.c_pre  += c_pre [j + n*g]/n_head;
                a.c_cand += c_cand[j + n*g]/n_head;
                a.c_top  += has_top ? c_top[j + n*g]/n_head : 0.0;
            }
            for (int32_t h = 0; h < n_head; h++) {
                a.w_pre  += w_pre [j + n*h]/n_head;
                a.w_resp += w_resp[j + n*h]/n_head;
            }
            a.n++;
        }
    }
    return true;
}

static void kvq_tt_print(const std::map<int32_t, kvq_tt_stat> & acc, bool has_top) {
    LOG("\n");
    LOG("kvq: two-tier stats, means over the scored positions and the query heads:\n");
    LOG("kvq:   c_pre = coarse weight on the prefix, c_miss = coarse weight on the coarse cells that are not protected\n");
    LOG("kvq:   c_top = coarse weight on the chosen cells (share of the unprotected coarse weight), w_prec = weight on the precise cells (w_resp after the prefix)\n");
    kvq_tt_stat tot;
    for (const auto & it : acc) {
        const auto & a = it.second;
        const double n = (double) std::max<int64_t>(1, a.n);
        LOG("kvq: tt layer %2d: c_pre %.4f  c_miss %.4f  c_top %.4f (%.4f)  w_prec %.4f  w_resp %.4f\n", it.first,
                a.c_pre/n, (a.c_cand - a.c_top)/n, a.c_top/n, has_top && a.c_cand > 0 ? a.c_top/a.c_cand : NAN, (a.w_pre + a.w_resp)/n, a.w_resp/n);
        tot.c_pre  += a.c_pre/n;
        tot.c_cand += a.c_cand/n;
        tot.c_top  += a.c_top/n;
        tot.w_pre  += a.w_pre/n;
        tot.w_resp += a.w_resp/n;
        tot.n++;
    }
    const double n = (double) std::max<int64_t>(1, tot.n);
    LOG("kvq: tt total   : c_pre %.4f  c_miss %.4f  c_top %.4f (%.4f)  w_prec %.4f  w_resp %.4f\n",
            tot.c_pre/n, (tot.c_cand - tot.c_top)/n, tot.c_top/n, has_top && tot.c_cand > 0 ? tot.c_top/tot.c_cand : NAN, (tot.w_pre + tot.w_resp)/n, tot.w_resp/n);
}

static bool kvq_layer_of(const char * name, const char * prefix, int32_t & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return false;
    }
    char * end = nullptr;
    const long v = strtol(name + n, &end, 10);
    if (end == name + n || *end != '\0') {
        return false;
    }
    il = (int32_t) v;
    return true;
}

// uses the tensor names of the qwen35 attention layer: Qcur-<il> (after RoPE), gate_sigmoid-<il>
static bool kvq_capture_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (kvq_capture *) user_data;

    int32_t il = -1;
    if (cap->tt_stats && kvq_layer_of(t->name, "kvq_stats-", il)) {
        if (!ask) {
            GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
            auto & v = cap->tt[il];
            v.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        }
        return true;
    }

    if (!cap->enabled) {
        return !ask;
    }

    const bool is_q = t->op == GGML_OP_ROPE && t->ne[0] == cap->n_embd_head && t->ne[1] == cap->n_head && kvq_layer_of(t->name, "Qcur-", il);
    const bool is_g = !is_q && t->ne[0] == (int64_t) cap->n_head*cap->n_embd_head && kvq_layer_of(t->name, "gate_sigmoid-", il);

    if (ask) {
        if (is_q && il == cap->il_first) {
            cap->pos0      = cap->pos_next;
            cap->pos_next += t->ne[2];
        }
        return is_g || (is_q && cap->pos0 < cap->win1 && cap->pos0 + t->ne[2] > cap->win0);
    }
    if (!is_q && !is_g) {
        return true;
    }

    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    cap->buf.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, cap->buf.data(), 0, ggml_nbytes(t));

    if (is_q) {
        const int64_t n_row = t->ne[0]*t->ne[1];
        auto & q = cap->q[il];
        q.resize((size_t) (cap->win1 - cap->win0)*n_row);
        for (int64_t j = 0; j < t->ne[2]; j++) {
            const int64_t p = cap->pos0 + j;
            if (p >= cap->win0 && p < cap->win1) {
                memcpy(q.data() + (p - cap->win0)*n_row, cap->buf.data() + j*n_row, n_row*sizeof(float));
            }
        }
    } else {
        auto & gs = cap->gate_sum[il];
        gs.resize(t->ne[0], 0.0);
        for (int64_t j = 0; j < t->ne[1]; j++) {
            const float * g = cap->buf.data() + j*t->ne[0];
            for (int64_t c = 0; c < t->ne[0]; c++) {
                gs[c] += g[c];
            }
        }
        cap->gate_n[il] += t->ne[1];
    }
    return true;
}

// dequantized 2D weight from the model file, ne1 rows of ne0 values
static bool kvq_read_weight(const std::string & path, const std::string & name, std::vector<float> & w, int64_t & ne0, int64_t & ne1) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), ip);
    if (gctx == nullptr) {
        LOG_ERR("%s: failed to read %s\n", __func__, path.c_str());
        return false;
    }

    bool ok = false;
    const int64_t id = gguf_find_tensor(gctx, name.c_str());
    ggml_tensor * t = id >= 0 ? ggml_get_tensor(meta, name.c_str()) : nullptr;
    if (t != nullptr && ggml_n_dims(t) == 2 && (t->type == GGML_TYPE_F32 || ggml_get_type_traits(t->type)->to_float)) {
        std::vector<uint8_t> data(ggml_nbytes(t));
        std::ifstream f(path, std::ios::binary);
        f.seekg((std::streamoff) (gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, id)));
        if (f.read((char *) data.data(), data.size())) {
            ne0 = t->ne[0];
            ne1 = t->ne[1];
            w.resize(ne0*ne1);
            if (t->type == GGML_TYPE_F32) {
                memcpy(w.data(), data.data(), data.size());
            } else {
                ggml_get_type_traits(t->type)->to_float(data.data(), w.data(), ne0*ne1);
            }
            ok = true;
        }
    }
    if (!ok) {
        LOG_ERR("%s: cannot read tensor %s\n", __func__, name.c_str());
    }

    gguf_free(gctx);
    ggml_free(meta);
    return ok;
}

// K metric: second moment of the queries of kv head g, pooled over its n_rep query heads
static void kvq_metric_k(const std::vector<float> & q, int32_t n_head, int32_t n_rep, int32_t n, int32_t g, std::vector<double> & m, int n_threads) {
    const int64_t n_win = q.size()/((size_t) n_head*n);
    m.assign((size_t) n*n, 0.0);
    for (int32_t r = 0; r < n_rep; r++) {
        llama_kvq_gram(q.data() + (size_t) (g*n_rep + r)*n, n_win, n, (int64_t) n_head*n, m.data(), n_threads);
    }
    for (double & v : m) {
        v /= (double) n_rep*n_win;
    }
}

// V metric: sum over the query heads r of kv head g of Wg_r^T Wg_r, wg = o_proj weight [n_out][n_in] with the gate already applied
static void kvq_metric_v(const std::vector<float> & wg, int64_t n_out, int64_t n_in, int32_t n_rep, int32_t n, int32_t g, std::vector<double> & m, int n_threads) {
    m.assign((size_t) n*n, 0.0);
    for (int32_t r = 0; r < n_rep; r++) {
        llama_kvq_gram(wg.data() + (size_t) (g*n_rep + r)*n, n_out, n, n_in, m.data(), n_threads);
    }
}

// llama.cpp Q8_0 round trip with the reference quantizer
static void kvq_fake_q8_0(float * x, int64_t n, std::vector<uint8_t> & tmp) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    tmp.resize(ggml_row_size(GGML_TYPE_Q8_0, n));
    tt->from_float_ref(x, tmp.data(), n);
    tt->to_float(tmp.data(), x, n);
}

// 8-bit affine round trip per group of 64: scale = (max - min)/255, offset = min, both stored as f16
static void kvq_fake_aff8(float * x, int64_t n) {
    for (int64_t i0 = 0; i0 < n; i0 += 64) {
        const int64_t i1 = std::min(n, i0 + 64);
        float mn = x[i0];
        float mx = x[i0];
        for (int64_t i = i0; i < i1; i++) {
            mn = std::min(mn, x[i]);
            mx = std::max(mx, x[i]);
        }
        const float scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16((mx - mn)/255.0f));
        const float off   = ggml_fp16_to_fp32(ggml_fp32_to_fp16(mn));
        for (int64_t i = i0; i < i1; i++) {
            const float q = scale > 0.0f ? std::clamp(std::nearbyint((x[i] - off)/scale), 0.0f, 255.0f) : 0.0f;
            x[i] = off + q*scale;
        }
    }
}

template <typename F>
static void kvq_parallel_rows(int64_t n, int n_threads, const F & f) {
    std::vector<std::thread> workers;
    const int64_t chunk = (n + n_threads - 1)/n_threads;
    for (int64_t i0 = 0; i0 < n; i0 += chunk) {
        workers.emplace_back([&f, i0, i1 = std::min(n, i0 + chunk)]() { f(i0, i1); });
    }
    for (auto & w : workers) {
        w.join();
    }
}

// codec stats of one (layer, K or V), summed over kv heads; errors over the coded rows only
struct kvq_stat {
    double err2 = 0.0; // |x' - x|^2
    double sig2 = 0.0; // |x - mu|^2
    double werr = 0.0; // d^T M d
    double wsig = 0.0; // (x - mu)^T M (x - mu)
    double gerr = 0.0; // K: d^T M d expected for Gaussian components
    double t_fit_sum = 0.0;
    double t_fit_max = 0.0;
    std::array<int64_t, 9> hist = {};
};

struct kvq_model_info {
    std::string path;
    int32_t     n_head;
    int32_t     n_head_kv;
    int32_t     n_embd_head;
};

// replace the cached K/V of the prefix [0, s) in all attention layers
static bool kvq_apply(llama_context * ctx, const kvq_args & args, const kvq_capture & cap, const kvq_model_info & mi, int32_t s) {
    llama_memory_t mem = llama_get_memory(ctx);

    const auto    layer_ids = llama_memory_kv_layer_ids(mem);
    const int32_t n         = mi.n_embd_head;
    const int32_t n_gqa     = mi.n_head_kv*n;
    const int32_t n_rep     = mi.n_head/mi.n_head_kv;
    const int     n_threads = std::max(1u, std::thread::hardware_concurrency());

    const bool  is_codec = args.variant == KVQ_VARIANT_CODEC;
    const bool  fit      = is_codec && args.tables_in_path.empty();
    const bool  do_kv[2] = { !is_codec || args.k_bits > 0, !is_codec || args.v_bits > 0 };
    const float bits[2]  = { (float) args.k_bits, (float) args.v_bits };

    // rows [r0, r1) are coded, the sinks and the recent window keep the exact values
    const int32_t r0 = std::min(s, args.sinks);
    const int32_t r1 = std::max(r0, s - args.recent);

    std::vector<llama_kvq_table> tables_in;
    std::vector<llama_kvq_table> tables_out;
    if (!args.tables_in_path.empty()) {
        if (!llama_kvq_load(args.tables_in_path, tables_in)) {
            return false;
        }
        LOG_INF("%s: loaded %zu tables from %s\n", __func__, tables_in.size(), args.tables_in_path.c_str());
    }

    LOG_INF("%s: %zu attention layers, s = %d, coded rows [%d, %d), %d threads\n", __func__, layer_ids.size(), s, r0, r1, n_threads);

    const int64_t t_start = ggml_time_us();
    double t_fit_sum = 0.0;
    double t_fit_max = 0.0;

    std::vector<float> rows[2];
    rows[0].resize((size_t) s*n_gqa);
    rows[1].resize((size_t) s*n_gqa);

    // two-tier: the variant goes to the coarse copy, both K and V of it are written
    const bool do_rw[2] = { args.two_tier || do_kv[0], args.two_tier || do_kv[1] };

    for (int32_t il : layer_ids) {
        for (int kv = 0; kv < 2; kv++) {
            if (do_rw[kv] && !llama_memory_kv_get_rows(mem, il, kv == 1, 0, 0, s, n_gqa, rows[kv].data())) {
                LOG_ERR("%s: cannot read the %s cache of layer %d\n", __func__, kv ? "V" : "K", il);
                return false;
            }
        }

        // the precise copy stays in the attention cache
        if (args.two_tier && args.tt_precise_aff8) {
            for (int kv = 0; kv < 2; kv++) {
                std::vector<float> prec = rows[kv];
                kvq_parallel_rows(s, n_threads, [&](int64_t i0, int64_t i1) {
                    for (int64_t r = i0; r < i1; r++) {
                        kvq_fake_aff8(prec.data() + r*n_gqa, n_gqa);
                    }
                });
                if (!llama_memory_kv_set_rows(mem, il, kv == 1, 0, 0, s, n_gqa, prec.data())) {
                    LOG_ERR("%s: cannot write the %s cache of layer %d\n", __func__, kv ? "V" : "K", il);
                    return false;
                }
            }
        }

        if (args.variant == KVQ_VARIANT_Q8_0 || args.variant == KVQ_VARIANT_AFF8) {
            for (int kv = 0; kv < 2; kv++) {
                kvq_parallel_rows(s, n_threads, [&](int64_t i0, int64_t i1) {
                    std::vector<uint8_t> tmp;
                    for (int64_t r = i0; r < i1; r++) {
                        float * x = rows[kv].data() + r*n_gqa;
                        if (args.variant == KVQ_VARIANT_Q8_0) {
                            kvq_fake_q8_0(x, n_gqa, tmp);
                        } else {
                            kvq_fake_aff8(x, n_gqa);
                        }
                    }
                });
            }
        }

        if (is_codec) {
            if (cap.q.count(il) == 0 || cap.gate_sum.count(il) == 0) {
                LOG_ERR("%s: no captured statistics for layer %d\n", __func__, il);
                return false;
            }

            // o_proj weight with input column j scaled by the mean sigmoid(gate)[j]
            std::vector<float> wg;
            int64_t n_in  = 0;
            int64_t n_out = 0;
            if (do_kv[1]) {
                if (!kvq_read_weight(mi.path, "blk." + std::to_string(il) + ".attn_output.weight", wg, n_in, n_out)) {
                    return false;
                }
                const auto & gs = cap.gate_sum.at(il);
                const double gn = (double) cap.gate_n.at(il);
                if (n_in != (int64_t) gs.size()) {
                    LOG_ERR("%s: o_proj input size %lld != gate size %zu\n", __func__, (long long) n_in, gs.size());
                    return false;
                }
                for (int64_t o = 0; o < n_out; o++) {
                    for (int64_t j = 0; j < n_in; j++) {
                        wg[o*n_in + j] *= (float) (gs[j]/gn);
                    }
                }
            }

            struct task {
                int  kv;
                int  g;
                bool ok;
                llama_kvq_table tab;
                kvq_stat st;
                double t_fit;
            };
            std::vector<task> tasks;
            for (int kv = 0; kv < 2; kv++) {
                for (int g = 0; do_kv[kv] && g < mi.n_head_kv; g++) {
                    tasks.push_back({ kv, g, false, {}, {}, 0.0 });
                }
            }
            const int nth = std::max<int>(1, n_threads/(int) tasks.size());

            auto run = [&](task & tk) {
                const bool is_v = tk.kv == 1;
                std::vector<float> x((size_t) s*n);
                for (int32_t r = 0; r < s; r++) {
                    memcpy(x.data() + (size_t) r*n, rows[tk.kv].data() + (size_t) r*n_gqa + tk.g*n, n*sizeof(float));
                }

                std::vector<double> m;
                if (is_v) {
                    kvq_metric_v(wg, n_out, n_in, n_rep, n, tk.g, m, nth);
                } else {
                    kvq_metric_k(cap.q.at(il), mi.n_head, n_rep, n, tk.g, m, nth);
                }

                if (fit) {
                    const int64_t t0 = ggml_time_us();
                    if (!llama_kvq_fit(tk.tab, x.data(), s, n, m.data(), bits[tk.kv], is_v, nth)) {
                        return;
                    }
                    tk.t_fit = (ggml_time_us() - t0)/1e6;
                    tk.tab.il   = il;
                    tk.tab.head = tk.g;
                } else {
                    const auto it = std::find_if(tables_in.begin(), tables_in.end(), [&](const llama_kvq_table & t) {
                        return t.il == il && t.head == tk.g && t.is_v == is_v && t.n == n;
                    });
                    if (it == tables_in.end()) {
                        return;
                    }
                    int sum = 0;
                    for (uint8_t b : it->bits) {
                        sum += b;
                    }
                    if (sum != (int) std::nearbyint(bits[tk.kv]*n)) {
                        return;
                    }
                    tk.tab = *it;
                }

                std::vector<int8_t> codes((size_t) s*n);
                std::vector<float>  gain(s);
                std::vector<float>  y((size_t) s*n);
                llama_kvq_encode(tk.tab, x.data(), s, codes.data(), gain.data(), nth);
                llama_kvq_decode(tk.tab, codes.data(), gain.data(), s, y.data(), nth);

                // MSE of the uniform quantizer with the Gaussian-optimal step, in units of sigma^2 (Max 1960)
                static const double gauss_mse[9] = { 1.0, 0.3634, 0.1188, 0.03744, 0.01154, 0.003490, 0.001035, 0.0003027, 0.00008741 };
                for (int32_t c = 0; c < n; c++) {
                    tk.st.hist[tk.tab.bits[c]]++;
                    if (!is_v) {
                        tk.st.gerr += gauss_mse[tk.tab.bits[c]]*tk.tab.sigma[c]*tk.tab.sigma[c]*(r1 - r0);
                    }
                }

                if (args.stats && r1 > r0) {
                    std::vector<float> d((size_t) (r1 - r0)*n);
                    std::vector<float> xc((size_t) (r1 - r0)*n);
                    for (int32_t r = r0; r < r1; r++) {
                        for (int32_t i = 0; i < n; i++) {
                            const size_t k = (size_t) r*n + i;
                            const size_t o = (size_t) (r - r0)*n + i;
                            d[o]  = y[k] - x[k];
                            xc[o] = x[k] - tk.tab.mu[i];
                            tk.st.err2 += (double) d[o]*d[o];
                            tk.st.sig2 += (double) xc[o]*xc[o];
                        }
                    }
                    std::vector<double> gd((size_t) n*n, 0.0);
                    std::vector<double> gx((size_t) n*n, 0.0);
                    llama_kvq_gram(d.data(),  r1 - r0, n, n, gd.data(), nth);
                    llama_kvq_gram(xc.data(), r1 - r0, n, n, gx.data(), nth);
                    for (size_t i = 0; i < m.size(); i++) {
                        tk.st.werr += gd[i]*m[i];
                        tk.st.wsig += gx[i]*m[i];
                    }
                }

                for (int32_t r = r0; r < r1; r++) {
                    memcpy(rows[tk.kv].data() + (size_t) r*n_gqa + tk.g*n, y.data() + (size_t) r*n, n*sizeof(float));
                }
                tk.ok = true;
            };

            std::vector<std::thread> workers;
            for (auto & tk : tasks) {
                workers.emplace_back([&run, &tk]() { run(tk); });
            }
            for (auto & w : workers) {
                w.join();
            }

            kvq_stat st[2];
            for (auto & tk : tasks) {
                if (!tk.ok) {
                    LOG_ERR("%s: codec failed for layer %d, %s head %d (fit failed, or no matching table)\n", __func__, il, tk.kv ? "V" : "K", tk.g);
                    return false;
                }
                auto & a = st[tk.kv];
                a.err2 += tk.st.err2;
                a.sig2 += tk.st.sig2;
                a.werr += tk.st.werr;
                a.wsig += tk.st.wsig;
                a.gerr += tk.st.gerr;
                for (int b = 0; b < 9; b++) {
                    a.hist[b] += tk.st.hist[b];
                }
                a.t_fit_sum += tk.t_fit;
                a.t_fit_max  = std::max(a.t_fit_max, tk.t_fit);
                t_fit_sum   += tk.t_fit;
                t_fit_max    = std::max(t_fit_max, tk.t_fit);
                if (fit) {
                    tables_out.push_back(std::move(tk.tab));
                }
            }

            if (args.stats) {
                for (int kv = 0; kv < 2; kv++) {
                    if (!do_kv[kv]) {
                        continue;
                    }
                    const auto & a = st[kv];
                    std::string hist;
                    for (int b = 0; b < 9; b++) {
                        hist += (b ? " " : "") + std::to_string(a.hist[b]);
                    }
                    LOG("kvq: layer %2d %s: rel err %.4f, metric rel err %.4f (Gaussian %.4f), fit %.2f s, bits 0..8: %s\n", il, kv ? "V" : "K",
                            std::sqrt(a.err2/a.sig2), std::sqrt(a.werr/a.wsig), kv ? NAN : std::sqrt(a.gerr/a.wsig), a.t_fit_sum, hist.c_str());
                }
            }
        }

        for (int kv = 0; kv < 2; kv++) {
            const bool ok = !do_rw[kv] || (args.two_tier ?
                llama_memory_kvq_set_coarse_rows(mem, il, kv == 1, 0, 0, s, n_gqa, rows[kv].data()) :
                llama_memory_kv_set_rows        (mem, il, kv == 1, 0, 0, s, n_gqa, rows[kv].data()));
            if (!ok) {
                LOG_ERR("%s: cannot write the %s %s cache of layer %d\n", __func__, args.two_tier ? "coarse" : "", kv ? "V" : "K", il);
                return false;
            }
        }
    }

    LOG_INF("%s: done in %.2f s\n", __func__, (ggml_time_us() - t_start)/1e6);
    if (fit) {
        LOG_INF("%s: fit %.2f s summed over %zu tables, max %.3f s per table\n", __func__, t_fit_sum, tables_out.size(), t_fit_max);
    }

    if (!args.tables_out_path.empty()) {
        if (!llama_kvq_save(args.tables_out_path, tables_out)) {
            return false;
        }
        LOG_INF("%s: saved %zu tables to %s\n", __func__, tables_out.size(), args.tables_out_path.c_str());
    }
    return true;
}

struct kvq_ref_header {
    char    magic[8];
    int32_t n_vocab;
    int32_t prefix_len;
    int32_t n_tokens;
};

static bool kvq_open_ref(const std::string & path, int32_t n_vocab, int32_t s, const std::vector<llama_token> & ids, std::ifstream & f) {
    f.open(path, std::ios::binary);
    kvq_ref_header hdr;
    if (!f || !f.read((char *) &hdr, sizeof(hdr)) || memcmp(hdr.magic, kvq_magic, sizeof(kvq_magic)) != 0) {
        LOG_ERR("%s: %s is not a reference file\n", __func__, path.c_str());
        return false;
    }
    if (hdr.n_vocab != n_vocab || hdr.prefix_len != s || hdr.n_tokens != (int32_t) ids.size()) {
        LOG_ERR("%s: reference mismatch: n_vocab %d vs %d, s %d vs %d, n_tokens %d vs %zu\n", __func__, hdr.n_vocab, n_vocab, hdr.prefix_len, s, hdr.n_tokens, ids.size());
        return false;
    }
    std::vector<llama_token> ref_ids(ids.size());
    if (!f.read((char *) ref_ids.data(), ref_ids.size() * sizeof(llama_token)) || ref_ids != ids) {
        LOG_ERR("%s: reference token ids do not match\n", __func__);
        return false;
    }
    return true;
}

static void kvq_print_compare(const std::vector<kvq_pos_stat> & stats, const std::string & path) {
    const size_t n = stats.size();
    std::vector<double> kl(n);
    double kl_sum = 0.0, nll_ref = 0.0, nll_var = 0.0;
    size_t n_top1 = 0;
    for (size_t i = 0; i < n; i++) {
        kl[i]    = stats[i].kl;
        kl_sum  += stats[i].kl;
        nll_ref += stats[i].nll_ref;
        nll_var += stats[i].nll_var;
        n_top1  += stats[i].top1;
    }

    // bootstrap over tokens
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    std::vector<double> means(1000);
    for (double & m : means) {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum += kl[dist(rng)];
        }
        m = sum / n;
    }
    std::sort(means.begin(), means.end());
    std::sort(kl.begin(), kl.end());

    const auto pct = [](const std::vector<double> & v, double q) {
        const double x  = q * (v.size() - 1);
        const size_t lo = (size_t) x;
        const size_t hi = std::min(lo + 1, v.size() - 1);
        return v[lo] + (x - lo) * (v[hi] - v[lo]);
    };

    LOG("\n");
    LOG("kvq: reference         = %s\n", path.c_str());
    LOG("kvq: scored positions  = %zu\n", n);
    LOG("kvq: mean KL(ref||var) = %.6e  95%% CI [%.6e, %.6e]\n", kl_sum / n, pct(means, 0.025), pct(means, 0.975));
    LOG("kvq: p90 KL            = %.6e\n", pct(kl, 0.90));
    LOG("kvq: max KL            = %.6e\n", kl.back());
    LOG("kvq: top-1 agreement   = %.4f%% (%zu / %zu)\n", 100.0 * n_top1 / n, n_top1, n);
    LOG("kvq: NLL ref           = %.6f\n", nll_ref / n);
    LOG("kvq: NLL var           = %.6f\n", nll_var / n);
    LOG("kvq: dNLL (var - ref)  = %+.6e\n", (nll_var - nll_ref) / n);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();

    kvq_args args;
    if (!kvq_parse_args(argc, argv, args)) {
        return 1;
    }

    std::vector<llama_token> ids;
    if (!kvq_read_tokens(args.tokens_path, ids)) {
        return 1;
    }
    const int32_t s = args.prefix_len;
    const int32_t e = (int32_t) ids.size() - 1;

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const int32_t n_ctx_min = GGML_PAD(e + 1, 256);
    if (params.n_ctx < n_ctx_min) {
        LOG_INF("%s: setting n_ctx = %d (was %d)\n", __func__, n_ctx_min, params.n_ctx);
        params.n_ctx = n_ctx_min;
    }
    params.n_parallel = 1;
    params.n_batch    = std::max(params.n_batch, args.chunk);

    kvq_capture cap;
    const bool capture = args.variant == KVQ_VARIANT_EXACT || args.variant == KVQ_VARIANT_CODEC;
    if (capture || args.tt_stats) {
        params.cb_eval           = kvq_capture_cb;
        params.cb_eval_user_data = &cap;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // two-tier: the memory must be created with the coarse copy, so the context is created here
    auto llama_init = common_init_from_params(params, args.two_tier);

    llama_context_ptr ctx_tt;

    auto * model = llama_init->model();
    if (model != nullptr && args.two_tier) {
        llama_model_set_kvq_two_tier(model, true);
        ctx_tt.reset(llama_init_from_model(model, common_context_params_to_llama(params)));
    }

    auto * ctx = args.two_tier ? ctx_tt.get() : llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load model or create context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (!kvq_check_segment(vocab, ids, s)) {
        return 1;
    }

    const int32_t n_scored = e - s - kvq_n_skip;
    LOG_INF("%s: n_tokens = %d, s = %d, e = %d, n_scored = %d, n_vocab = %d, chunk = %d\n", __func__, (int) ids.size(), s, e, n_scored, n_vocab, args.chunk);

    std::ofstream ref_out;
    if (!args.ref_out_path.empty()) {
        ref_out.open(args.ref_out_path, std::ios::binary);
        kvq_ref_header hdr;
        memcpy(hdr.magic, kvq_magic, sizeof(kvq_magic));
        hdr.n_vocab    = n_vocab;
        hdr.prefix_len = s;
        hdr.n_tokens   = (int32_t) ids.size();
        ref_out.write((const char *) &hdr, sizeof(hdr));
        ref_out.write((const char *) ids.data(), ids.size() * sizeof(llama_token));
        if (!ref_out) {
            LOG_ERR("%s: failed to write %s\n", __func__, args.ref_out_path.c_str());
            return 1;
        }
    }

    std::ifstream ref_in;
    if (!args.ref_in_path.empty() && !kvq_open_ref(args.ref_in_path, n_vocab, s, ids, ref_in)) {
        return 1;
    }
    std::ifstream ref2_in;
    if (!args.ref2_in_path.empty() && (!ref_in.is_open() || !kvq_open_ref(args.ref2_in_path, n_vocab, s, ids, ref2_in))) {
        LOG_ERR("%s: --kvq-ref2 needs --kvq-ref and a valid reference\n", __func__);
        return 1;
    }

    kvq_model_info mi;
    mi.path        = params.model.path;
    mi.n_head      = llama_model_n_head(model);
    mi.n_head_kv   = llama_model_n_head_kv(model);
    mi.n_embd_head = 0;
    {
        char arch[64];
        char val[64];
        if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) > 0 &&
            llama_model_meta_val_str(model, (std::string(arch) + ".attention.key_length").c_str(), val, sizeof(val)) > 0) {
            mi.n_embd_head = std::atoi(val);
        }
    }

    const auto layer_ids = llama_memory_kv_layer_ids(llama_get_memory(ctx));
    if (args.variant != KVQ_VARIANT_NONE) {
        if (layer_ids.empty() || mi.n_embd_head <= 0 || mi.n_head_kv <= 0 || mi.n_head % mi.n_head_kv != 0) {
            LOG_ERR("%s: --kvq-variant needs a model with an attention KV cache and attention.key_length\n", __func__);
            return 1;
        }
        LOG_INF("%s: variant: %d, k_bits = %d, v_bits = %d, sinks = %d, recent = %d, attention layers %d..%d\n", __func__,
                (int) args.variant, args.k_bits, args.v_bits, args.sinks, args.recent, layer_ids.front(), layer_ids.back());
    }
    cap.il_first    = layer_ids.empty() ? -1 : *std::min_element(layer_ids.begin(), layer_ids.end());
    cap.n_head      = mi.n_head;
    cap.n_embd_head = mi.n_embd_head;
    cap.win0        = std::max(0, s - args.q_win);
    cap.win1        = s;

    llama_memory_clear(llama_get_memory(ctx), true);

    const int64_t t_prefill_start = ggml_time_us();
    cap.enabled = capture;
    if (!kvq_prefill(ctx, ids, s, params.n_batch)) {
        return 1;
    }
    cap.enabled = false;
    const double t_prefill = (ggml_time_us() - t_prefill_start) / 1e6;
    LOG_INF("%s: prefill: %d tokens in %.2f s, %.1f t/s\n", __func__, s, t_prefill, s / t_prefill);

    if (capture) {
        bool ok = cap.pos_next == s && cap.q.size() == layer_ids.size() && cap.gate_sum.size() == layer_ids.size();
        for (const auto & it : cap.gate_n) {
            ok = ok && it.second == s;
        }
        if (!ok) {
            LOG_ERR("%s: capture is incomplete: %lld positions, %zu query layers, %zu gate layers\n", __func__, (long long) cap.pos_next, cap.q.size(), cap.gate_sum.size());
            return 1;
        }
        LOG_INF("%s: captured queries of positions [%d, %d) and gates of %lld positions in %zu layers\n", __func__, cap.win0, cap.win1, (long long) cap.pos_next, cap.q.size());
    }

    if (args.variant != KVQ_VARIANT_NONE && !kvq_apply(ctx, args, cap, mi, s)) {
        return 1;
    }
    cap.q.clear();

    if (args.two_tier) {
        llama_kvq_params tp;
        tp.enabled  = true;
        tp.n_top    = args.tt_top;
        tp.n_sink   = args.tt_all_coarse ? 0 : args.sinks;
        tp.n_recent = args.tt_all_precise ? s : args.tt_all_coarse ? 0 : args.tt_protect;
        tp.n_prefix = s;
        tp.stats    = args.tt_stats;
        if (tp.n_top > s - std::min(s, tp.n_sink + tp.n_recent)) {
            LOG_ERR("%s: --kvq-two-tier %d is more than the unprotected prefix cells\n", __func__, tp.n_top);
            return 1;
        }
        if (!llama_memory_kvq_set_params(llama_get_memory(ctx), tp)) {
            LOG_ERR("%s: cannot switch to two-tier mode\n", __func__);
            return 1;
        }
        LOG_INF("%s: two-tier: n_top = %d, protected: first %d and last %d prefix positions, precise copy %s\n", __func__,
                tp.n_top, tp.n_sink, tp.n_recent, args.tt_precise_aff8 ? "aff8" : "exact");
        cap.tt_stats = args.tt_stats;
    }
    std::map<int32_t, kvq_tt_stat> tt_acc;

    std::vector<float> lp(n_vocab);
    std::vector<float> lp_ref(n_vocab);
    std::vector<kvq_pos_stat> stats;
    std::vector<kvq_pos_stat> stats2;
    stats.reserve(n_scored);
    double nll = 0.0;

    const auto compare = [&](std::ifstream & f, int32_t top, llama_token target, std::vector<kvq_pos_stat> & out) {
        if (!f.read((char *) lp_ref.data(), n_vocab * sizeof(float))) {
            LOG_ERR("%s: reference file is truncated\n", __func__);
            return false;
        }
        double kl = 0.0;
        for (int32_t v = 0; v < n_vocab; v++) {
            kl += std::exp((double) lp_ref[v]) * ((double) lp_ref[v] - lp[v]);
        }
        out.push_back({ kl, -lp_ref[target], -lp[target], kvq_argmax(lp_ref.data(), n_vocab) == top });
        return true;
    };

    const int64_t t_score_start = ggml_time_us();
    const bool ok = kvq_score(ctx, ids, s, args.chunk, [&](int32_t i, const float * logits) {
        const llama_token target = ids[s + i + 1];
        const int32_t top = kvq_log_softmax(logits, n_vocab, lp.data());
        nll -= lp[target];

        if (ref_out.is_open() && !ref_out.write((const char *) lp.data(), n_vocab * sizeof(float))) {
            LOG_ERR("%s: failed to write reference\n", __func__);
            return false;
        }
        if (ref_in.is_open() && !compare(ref_in, top, target, stats)) {
            return false;
        }
        if (ref2_in.is_open() && !compare(ref2_in, top, target, stats2)) {
            return false;
        }
        return true;
    }, [&](int32_t i0, int32_t n) {
        if (!cap.tt_stats) {
            return true;
        }
        llama_synchronize(ctx);
        if (cap.tt.size() != layer_ids.size()) {
            LOG_ERR("%s: two-tier stats of %zu layers, expected %zu\n", __func__, cap.tt.size(), layer_ids.size());
            return false;
        }
        const bool ok = kvq_tt_add(cap, tt_acc, n, std::max(0, kvq_n_skip - i0), args.tt_top > 0, mi.n_head, mi.n_head_kv);
        cap.tt.clear();
        return ok;
    });
    if (!ok) {
        return 1;
    }
    const double t_score = (ggml_time_us() - t_score_start) / 1e6;
    LOG_INF("%s: score: %d tokens in %.2f s, %.1f t/s\n", __func__, e - s, t_score, (e - s) / t_score);

    LOG("\n");
    LOG("kvq: NLL = %.6f nats/token over %d positions\n", nll / n_scored, n_scored);
    if (nll / n_scored > 2.0) {
        LOG_WRN("kvq: NLL above 2.0, tokenization likely does not match the model\n");
    }

    if (ref_out.is_open()) {
        ref_out.close();
        LOG("kvq: saved reference to %s\n", args.ref_out_path.c_str());
    }

    if (args.tt_stats) {
        kvq_tt_print(tt_acc, args.tt_top > 0);
    }

    if (ref_in.is_open()) {
        kvq_print_compare(stats, args.ref_in_path);
        if (ref2_in.is_open()) {
            kvq_print_compare(stats2, args.ref2_in_path);
        }

        if (!args.csv_path.empty()) {
            FILE * fcsv = fopen(args.csv_path.c_str(), "w");
            if (fcsv == nullptr) {
                LOG_ERR("%s: failed to open %s\n", __func__, args.csv_path.c_str());
                return 1;
            }
            fprintf(fcsv, "target_pos,kl,top1,nll_ref,nll_var\n");
            for (size_t i = 0; i < stats.size(); i++) {
                const auto & st = stats[i];
                fprintf(fcsv, "%d,%.9e,%d,%.9e,%.9e\n", s + kvq_n_skip + 1 + (int) i, st.kl, st.top1 ? 1 : 0, st.nll_ref, st.nll_var);
            }
            fclose(fcsv);
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);
    common_memory_breakdown_print(ctx);

    llama_backend_free();

    return 0;
}
