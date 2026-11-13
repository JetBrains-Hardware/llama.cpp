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
//   aff<b>         the same at b < 8 bits, the first --kvq-sinks positions stay exact
//   rotK<b>V<b>    no fit: normalised Walsh-Hadamard per kv head row, uniform mid-rise quantizer with step = GAUSS_STEP[b] * row RMS, un-rotate
//   K<b>V<b>, K<b>, V<b>
//                  transform codec at b bits (src/llama-kvq-codec.h), tables fitted on the prefix:
//                  K metric = second moment of the queries at the last --kvq-qwin prefix positions
//                  V metric = o_proj weight scaled by the mean sigmoid(gate) over the prefix
//                  the first --kvq-sinks and the last --kvq-recent prefix positions stay exact
//                  --kvq-fit-chunk C: each full chunk [c*C, (c+1)*C) of the prefix gets its own tables, fitted on that chunk only
//                  (gate mean over the chunk; K metric from the chunk's own queries, or with --kvq-kmetric last from the last --kvq-qwin);
//                  the partial tail chunk stays exact
//
// --kvq-two-tier keeps two copies of the prefix K/V and attends with both (see llama_kvq_params):
// the precise copy is --kvq-precise (aff8 or exact), the coarse copy is --kvq-variant
//   all-precise    every prefix cell is precise
//   all-coarse     every prefix cell is coarse
//   none           the protected cells (the first --kvq-sinks and the last --kvq-protect prefix positions) are precise
//   <N>            the protected cells and the top N cells of the coarse attention per kv head and query position
//
// --kvq-packed: packed two-tier attention (ggml_kvq_attn, see llama_kvq_params::packed), needs a K<b>V<b> variant with --kvq-fit-chunk C,
// a q8_0 cache and flash attention; the attention rotation is turned off
//   coarse region = the full chunks of C rows that end at or before s - --kvq-protect, coded from the q8_0 rows (sinks included)
//   the q8_0 cache is not changed, its rows of the coarse region are copied to the host store
//   ubatches of up to 8 tokens attend through the packed path, larger ones through flash attention over the q8_0 cache
//   --kvq-pf-n 1024, --kvq-cap 1024, --kvq-lead 2, --kvq-sync off|count|always, --kvq-sync-m 64, --kvq-sync-hits 48,
//   --kvq-all-resident (debug: every coarse row resident), --kvq-pk-stats (top-m counters with sync off)
// --kvq-no-rot: attention rotation off (for references of the packed mode)
// --kvq-perf: no logits processing, decode times per chunk only

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
    KVQ_VARIANT_AFF,   // affine at k_bits, sinks exact
    KVQ_VARIANT_ROT,   // Hadamard + per-row uniform quantizer, no fit
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
    int32_t fit_chunk = 0;     // 0: one table set fitted on the whole prefix
    bool    k_last    = false; // per-chunk fit: K metric from the last q_win queries instead of the chunk's own
    int32_t load_kv   = -1;    // --kvq-load-only k|v: load the tables of K (0) or V (1) only, fit the other
    bool    own_mu_k  = false; // loaded K tables: replace mu by the mean of this prefix
    bool    stats  = false;

    bool    two_tier        = false;
    bool    tt_all_precise  = false;
    bool    tt_all_coarse   = false;
    int32_t tt_top          = 0;
    int32_t tt_protect      = 512;
    bool    tt_precise_aff8 = true;
    bool    tt_stats        = false;

    bool    packed          = false;
    int32_t pk_pf_n         = 1024;
    int32_t pk_cap          = 1024;
    int32_t pk_lead         = 2;
    int32_t pk_sync         = LLAMA_KVQ_SYNC_OFF;
    int32_t pk_sync_m       = 64;
    int32_t pk_sync_hits    = 48;
    bool    pk_all_res      = false;
    bool    pk_stats        = false;

    bool    no_rot          = false;
    bool    perf            = false;
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
    if (sscanf(v.c_str(), "aff%d%c", &kb, &extra) == 1 && kb >= 1 && kb < 8) {
        args.variant = KVQ_VARIANT_AFF;
        args.k_bits  = kb;
        args.v_bits  = kb;
        return true;
    }
    if (sscanf(v.c_str(), "rotK%dV%d%c", &kb, &vb, &extra) == 2 && kb >= 1 && kb <= 8 && vb >= 1 && vb <= 8) {
        args.variant = KVQ_VARIANT_ROT;
        args.k_bits  = kb;
        args.v_bits  = vb;
        return true;
    }
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
        } else if (a == "--kvq-fit-chunk" && has_val) {
            args.fit_chunk = std::stoi(argv[++i]);
        } else if (a == "--kvq-kmetric" && has_val) {
            const std::string v = argv[++i];
            if (v != "own" && v != "last") {
                LOG_ERR("%s: bad K metric source '%s'\n", __func__, v.c_str());
                return false;
            }
            args.k_last = v == "last";
        } else if (a == "--kvq-load-only" && has_val) {
            const std::string v = argv[++i];
            if (v != "k" && v != "v") {
                LOG_ERR("%s: bad --kvq-load-only '%s'\n", __func__, v.c_str());
                return false;
            }
            args.load_kv = v == "v";
        } else if (a == "--kvq-own-mean-k") {
            args.own_mu_k = true;
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
        } else if (a == "--kvq-packed") {
            args.packed = true;
        } else if (a == "--kvq-pf-n" && has_val) {
            args.pk_pf_n = std::stoi(argv[++i]);
        } else if (a == "--kvq-cap" && has_val) {
            args.pk_cap = std::stoi(argv[++i]);
        } else if (a == "--kvq-lead" && has_val) {
            args.pk_lead = std::stoi(argv[++i]);
        } else if (a == "--kvq-sync" && has_val) {
            const std::string v = argv[++i];
            if (v == "off") {
                args.pk_sync = LLAMA_KVQ_SYNC_OFF;
            } else if (v == "count") {
                args.pk_sync = LLAMA_KVQ_SYNC_COUNT;
            } else if (v == "always") {
                args.pk_sync = LLAMA_KVQ_SYNC_ALWAYS;
            } else {
                LOG_ERR("%s: bad --kvq-sync '%s'\n", __func__, v.c_str());
                return false;
            }
        } else if (a == "--kvq-sync-m" && has_val) {
            args.pk_sync_m = std::stoi(argv[++i]);
        } else if (a == "--kvq-sync-hits" && has_val) {
            args.pk_sync_hits = std::stoi(argv[++i]);
        } else if (a == "--kvq-all-resident") {
            args.pk_all_res = true;
        } else if (a == "--kvq-pk-stats") {
            args.pk_stats = true;
        } else if (a == "--kvq-no-rot") {
            args.no_rot = true;
        } else if (a == "--kvq-perf") {
            args.perf = true;
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
                "       [--kvq-variant exact|q8_0|aff8|K<b>V<b>|K<b>|V<b>] [--kvq-sinks 4] [--kvq-recent 0] [--kvq-qwin 2048] [--kvq-fit-chunk 0 [--kvq-kmetric own|last]]\n"
                "       [--kvq-save-tables tables.bin] [--kvq-load-tables tables.bin [--kvq-load-only k|v] [--kvq-own-mean-k]] [--kvq-stats]\n"
                "       [--kvq-two-tier all-precise|all-coarse|none|<N> [--kvq-protect 512] [--kvq-precise aff8|exact] [--kvq-tt-stats]]\n"
                "       [--kvq-packed [--kvq-pf-n 1024] [--kvq-cap 1024] [--kvq-lead 2] [--kvq-sync off|count|always] [--kvq-sync-m 64] [--kvq-sync-hits 48]\n"
                "                     [--kvq-all-resident] [--kvq-pk-stats]] [--kvq-no-rot] [--kvq-perf] [common args]\n", argv[0]);
        return false;
    }
    if (args.chunk <= 0) {
        LOG_ERR("%s: --kvq-chunk must be > 0\n", __func__);
        return false;
    }
    if (args.sinks < 0 || args.recent < 0 || args.q_win <= 0 || args.fit_chunk < 0) {
        LOG_ERR("%s: bad --kvq-sinks, --kvq-recent, --kvq-qwin or --kvq-fit-chunk\n", __func__);
        return false;
    }
    if (args.fit_chunk > 0 && (args.variant != KVQ_VARIANT_CODEC || !args.tables_out_path.empty() || !args.tables_in_path.empty())) {
        LOG_ERR("%s: --kvq-fit-chunk needs a codec variant and no --kvq-save-tables or --kvq-load-tables\n", __func__);
        return false;
    }
    if ((args.load_kv >= 0 || args.own_mu_k) && args.tables_in_path.empty()) {
        LOG_ERR("%s: --kvq-load-only and --kvq-own-mean-k need --kvq-load-tables\n", __func__);
        return false;
    }
    if ((!args.tables_out_path.empty() || !args.tables_in_path.empty() || args.stats) && args.variant != KVQ_VARIANT_CODEC) {
        LOG_ERR("%s: --kvq-save-tables, --kvq-load-tables and --kvq-stats need a codec variant\n", __func__);
        return false;
    }
    // the precise cells are chosen by --kvq-protect; uncoded rows of the coarse copy (sinks, --kvq-recent, tail chunk) stay exact
    if (args.two_tier && (args.variant == KVQ_VARIANT_NONE || args.tt_protect < 0)) {
        LOG_ERR("%s: --kvq-two-tier needs a --kvq-variant for the coarse copy and --kvq-protect >= 0\n", __func__);
        return false;
    }
    if (args.tt_stats && !args.two_tier) {
        LOG_ERR("%s: --kvq-tt-stats needs --kvq-two-tier\n", __func__);
        return false;
    }
    if (args.packed && (args.two_tier || args.variant != KVQ_VARIANT_CODEC || args.k_bits == 0 || args.v_bits == 0 || args.fit_chunk <= 0 ||
                        args.k_last || !args.tables_in_path.empty() || args.recent != 0)) {
        LOG_ERR("%s: --kvq-packed needs a K<b>V<b> variant with --kvq-fit-chunk, the own K metric, no --kvq-recent, and no --kvq-two-tier\n", __func__);
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

// feed ids[s:e] in chunks, call on_pre() before and on_chunk(i0, n) after each chunk of fed positions [i0, i0 + n)
// and on_pos(i, logits) for each fed position i >= kvq_n_skip
template <typename F, typename G, typename H>
static bool kvq_score(llama_context * ctx, const std::vector<llama_token> & ids, int32_t s, int32_t chunk, F && on_pos, G && on_chunk, H && on_pre) {
    const int32_t e = (int32_t) ids.size() - 1;
    common_batch batch(ctx);
    for (int32_t p = s; p < e; p += chunk) {
        const int32_t n = std::min(chunk, e - p);
        batch.clear();
        for (int32_t j = 0; j < n; j++) {
            batch.add(ids[p + j], p + j, 0, true);
        }
        on_pre();
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
    int32_t gate_chunk = 0; // > 0: gate sums per chunk of this many positions
    std::map<int32_t, std::vector<double>> gate_sum; // [n_chunk][n_head*n_embd_head], sum of sigmoid(gate) over the prefix or per chunk
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
        // the gate rows of a layer arrive in position order
        auto & gs = cap->gate_sum[il];
        const int64_t p0 = cap->gate_n[il];
        for (int64_t j = 0; j < t->ne[1]; j++) {
            const int64_t ic = cap->gate_chunk > 0 ? (p0 + j)/cap->gate_chunk : 0;
            gs.resize(std::max(gs.size(), (size_t) ((ic + 1)*t->ne[0])), 0.0);
            const float * g   = cap->buf.data() + j*t->ne[0];
            double      * dst = gs.data() + ic*t->ne[0];
            for (int64_t c = 0; c < t->ne[0]; c++) {
                dst[c] += g[c];
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
// q = n_win positions of [n_head][n]
static void kvq_metric_k(const float * q, int64_t n_win, int32_t n_head, int32_t n_rep, int32_t n, int32_t g, std::vector<double> & m, int n_threads) {
    m.assign((size_t) n*n, 0.0);
    for (int32_t r = 0; r < n_rep; r++) {
        llama_kvq_gram(q + (size_t) (g*n_rep + r)*n, n_win, n, (int64_t) n_head*n, m.data(), n_threads);
    }
    for (double & v : m) {
        v /= (double) n_rep*n_win;
    }
}

// gram[r] = Wo_r^T Wo_r for the query heads r of kv head g, wo = o_proj weight [n_out][n_in]
static void kvq_gram_v(const std::vector<float> & wo, int64_t n_out, int64_t n_in, int32_t n_rep, int32_t n, int32_t g, std::vector<std::vector<double>> & gram, int n_threads) {
    gram.assign(n_rep, std::vector<double>((size_t) n*n, 0.0));
    for (int32_t r = 0; r < n_rep; r++) {
        llama_kvq_gram(wo.data() + (size_t) (g*n_rep + r)*n, n_out, n, n_in, gram[r].data(), n_threads);
    }
}

// V metric: sum over r of Wg_r^T Wg_r with Wg = Wo scaled by the gate mean gbar[n_in] per input column, i.e. D_r gram[r] D_r
static void kvq_metric_v(const std::vector<std::vector<double>> & gram, const double * gbar, int32_t n, int32_t g, std::vector<double> & m) {
    const int32_t n_rep = (int32_t) gram.size();
    m.assign((size_t) n*n, 0.0);
    for (int32_t r = 0; r < n_rep; r++) {
        const double * d = gbar + (size_t) (g*n_rep + r)*n;
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < n; j++) {
                m[(size_t) i*n + j] += d[i]*gram[r][(size_t) i*n + j]*d[j];
            }
        }
    }
}

// llama.cpp Q8_0 round trip with the reference quantizer
static void kvq_fake_q8_0(float * x, int64_t n, std::vector<uint8_t> & tmp) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    tmp.resize(ggml_row_size(GGML_TYPE_Q8_0, n));
    tt->from_float_ref(x, tmp.data(), n);
    tt->to_float(tmp.data(), x, n);
}

// affine round trip per group of 64: scale = (max - min)/(2^bits - 1), offset = min, both stored as f16
static void kvq_fake_aff(float * x, int64_t n, int bits) {
    const float n_lv = (float) ((1 << bits) - 1);
    for (int64_t i0 = 0; i0 < n; i0 += 64) {
        const int64_t i1 = std::min(n, i0 + 64);
        float mn = x[i0];
        float mx = x[i0];
        for (int64_t i = i0; i < i1; i++) {
            mn = std::min(mn, x[i]);
            mx = std::max(mx, x[i]);
        }
        const float scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16((mx - mn)/n_lv));
        const float off   = ggml_fp16_to_fp32(ggml_fp32_to_fp16(mn));
        for (int64_t i = i0; i < i1; i++) {
            const float q = scale > 0.0f ? std::clamp(std::nearbyint((x[i] - off)/scale), 0.0f, n_lv) : 0.0f;
            x[i] = off + q*scale;
        }
    }
}

static void kvq_fake_aff8(float * x, int64_t n) {
    kvq_fake_aff(x, n, 8);
}

// in-place normalised Walsh-Hadamard transform, n a power of 2 (self-inverse)
static void kvq_hadamard(float * x, int32_t n) {
    for (int32_t h = 1; h < n; h *= 2) {
        for (int32_t i = 0; i < n; i += 2*h) {
            for (int32_t j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float s = 1.0f/std::sqrt((float) n);
    for (int32_t i = 0; i < n; i++) {
        x[i] *= s;
    }
}

// rotate, quantize with a uniform mid-rise quantizer, step = Gaussian-optimal step * row RMS (f16), rotate back
static void kvq_fake_rot(float * x, int32_t n, int bits) {
    static const float gauss_step[9] = { 0.0f, 1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f, 0.1041f, 0.0569f, 0.0308f };
    kvq_hadamard(x, n);
    double ss = 0.0;
    for (int32_t i = 0; i < n; i++) {
        ss += (double) x[i]*x[i];
    }
    const float step = ggml_fp16_to_fp32(ggml_fp32_to_fp16(gauss_step[bits]*(float) std::sqrt(ss/n)));
    const float lo   = (float) -(1 << (bits - 1));
    const float hi   = (float) ((1 << (bits - 1)) - 1);
    for (int32_t i = 0; i < n; i++) {
        x[i] = step > 0.0f ? (std::clamp(std::floor(x[i]/step), lo, hi) + 0.5f)*step : 0.0f;
    }
    kvq_hadamard(x, n);
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

    // table sets are fitted on the whole prefix, or on each full chunk [c*C, (c+1)*C)
    // packed: the chunks that end at or before s - tt_protect, every row coded, the cache is not changed
    const int32_t fc      = args.fit_chunk;
    const int32_t n_chunk = args.packed ? (s - args.tt_protect)/fc : fc > 0 ? s/fc : 1;

    // rows [r0, r1) are coded, the sinks, the recent window and the partial tail chunk keep the exact values
    const int32_t r0 = args.packed ? 0 : std::min(s, args.sinks);
    const int32_t r1 = std::max(r0, std::min(s - args.recent, fc > 0 ? n_chunk*fc : s));

    if (args.packed) {
        if (n_chunk < 1) {
            LOG_ERR("%s: the prefix of %d tokens has no full chunk of %d before the last %d\n", __func__, s, fc, args.tt_protect);
            return false;
        }
        if (!llama_memory_kvq_packed_alloc(mem, n_chunk*fc, fc, args.k_bits, args.v_bits, args.pk_cap, args.pk_all_res, args.sinks)) {
            LOG_ERR("%s: cannot allocate the packed state\n", __func__);
            return false;
        }
    }

    std::vector<llama_kvq_table> tables_in;
    std::vector<llama_kvq_table> tables_out;
    if (!args.tables_in_path.empty()) {
        if (!llama_kvq_load(args.tables_in_path, tables_in)) {
            return false;
        }
        LOG_INF("%s: loaded %zu tables from %s\n", __func__, tables_in.size(), args.tables_in_path.c_str());
    }

    LOG_INF("%s: %zu attention layers, s = %d, coded rows [%d, %d), %d table sets (fit chunk %d, K metric %s), %d threads\n", __func__,
            layer_ids.size(), s, r0, r1, n_chunk, fc, fc > 0 && !args.k_last ? "own chunk" : "last queries", n_threads);

    const int64_t t_start = ggml_time_us();
    double t_fit_sum = 0.0;
    double t_fit_max = 0.0;

    std::vector<float> rows[2];
    rows[0].resize((size_t) s*n_gqa);
    rows[1].resize((size_t) s*n_gqa);

    // two-tier: the variant goes to the coarse copy, both K and V of it are written
    const bool do_rw[2] = { args.two_tier || do_kv[0], args.two_tier || do_kv[1] };
    const bool do_wr[2] = { do_rw[0] && !args.packed, do_rw[1] && !args.packed };

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

        if (args.variant == KVQ_VARIANT_AFF || args.variant == KVQ_VARIANT_ROT) {
            for (int kv = 0; kv < 2; kv++) {
                const int b = kv ? args.v_bits : args.k_bits;
                kvq_parallel_rows(r1 - r0, n_threads, [&](int64_t i0, int64_t i1) {
                    for (int64_t r = r0 + i0; r < r0 + i1; r++) {
                        float * x = rows[kv].data() + r*n_gqa;
                        if (args.variant == KVQ_VARIANT_AFF) {
                            kvq_fake_aff(x, n_gqa, b);
                        } else {
                            for (int32_t g = 0; g < mi.n_head_kv; g++) {
                                kvq_fake_rot(x + g*n, n, b);
                            }
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

            // o_proj weight, its input column j is scaled by the mean sigmoid(gate)[j] in the V metric
            std::vector<float> wo;
            int64_t n_in  = 0;
            int64_t n_out = 0;
            if (do_kv[1]) {
                if (!kvq_read_weight(mi.path, "blk." + std::to_string(il) + ".attn_output.weight", wo, n_in, n_out)) {
                    return false;
                }
                if ((int64_t) cap.gate_sum.at(il).size() < std::max(1, n_chunk)*n_in) {
                    LOG_ERR("%s: o_proj input size %lld does not match the gate sums (%zu)\n", __func__, (long long) n_in, cap.gate_sum.at(il).size());
                    return false;
                }
            }

            struct task {
                int  kv;
                int  g;
                bool ok;
                std::vector<llama_kvq_table> tabs; // one per chunk
                kvq_stat st;
                double t_fit;
                double t_fit_max;
                std::vector<int8_t> codes;
                std::vector<float>  gain;
            };
            std::vector<task> tasks;
            for (int kv = 0; kv < 2; kv++) {
                for (int g = 0; do_kv[kv] && g < mi.n_head_kv; g++) {
                    tasks.push_back({ kv, g, false, {}, {}, 0.0, 0.0, {}, {} });
                }
            }
            const int nth = std::max<int>(1, n_threads/(int) tasks.size());

            auto run = [&](task & tk) {
                const bool is_v = tk.kv == 1;
                std::vector<float> x((size_t) s*n);
                for (int32_t r = 0; r < s; r++) {
                    memcpy(x.data() + (size_t) r*n, rows[tk.kv].data() + (size_t) r*n_gqa + tk.g*n, n*sizeof(float));
                }

                std::vector<std::vector<double>> gram;
                if (is_v) {
                    kvq_gram_v(wo, n_out, n_in, n_rep, n, tk.g, gram, nth);
                }

                std::vector<int8_t> & codes = tk.codes;
                std::vector<float>  & gain  = tk.gain;
                codes.resize((size_t) s*n);
                gain.resize(s);
                std::vector<float>  y((size_t) s*n);
                std::vector<double> m;
                std::vector<double> gbar(n_in);

                for (int32_t ic = 0; ic < n_chunk; ic++) {
                    // fit rows [a, b), coded rows [c0, c1)
                    const int32_t a  = fc > 0 ? ic*fc : 0;
                    const int32_t b  = fc > 0 ? a + fc : s;
                    const int32_t c0 = std::max(a, r0);
                    const int32_t c1 = std::min(b, r1);

                    const auto & q = cap.q.at(il);
                    if (is_v) {
                        const double gn = fc > 0 ? (double) (b - a) : (double) cap.gate_n.at(il);
                        for (int64_t j = 0; j < n_in; j++) {
                            gbar[j] = cap.gate_sum.at(il)[ic*n_in + j]/gn;
                        }
                        kvq_metric_v(gram, gbar.data(), n, tk.g, m);
                    } else if (fc > 0 && !args.k_last) {
                        kvq_metric_k(q.data() + (size_t) (a - cap.win0)*mi.n_head*n, b - a, mi.n_head, n_rep, n, tk.g, m, nth);
                    } else {
                        kvq_metric_k(q.data(), q.size()/((size_t) mi.n_head*n), mi.n_head, n_rep, n, tk.g, m, nth);
                    }

                    llama_kvq_table tab;
                    if (fit || (args.load_kv >= 0 && tk.kv != args.load_kv)) {
                        const int64_t t0 = ggml_time_us();
                        if (!llama_kvq_fit(tab, x.data() + (size_t) a*n, b - a, n, m.data(), bits[tk.kv], is_v, nth)) {
                            return;
                        }
                        const double t_fit = (ggml_time_us() - t0)/1e6;
                        tk.t_fit    += t_fit;
                        tk.t_fit_max = std::max(tk.t_fit_max, t_fit);
                        tab.il   = il;
                        tab.head = tk.g;
                    } else {
                        const auto it = std::find_if(tables_in.begin(), tables_in.end(), [&](const llama_kvq_table & t) {
                            return t.il == il && t.head == tk.g && t.is_v == is_v && t.n == n;
                        });
                        if (it == tables_in.end()) {
                            return;
                        }
                        int sum = 0;
                        for (uint8_t bt : it->bits) {
                            sum += bt;
                        }
                        if (sum != (int) std::nearbyint(bits[tk.kv]*n)) {
                            return;
                        }
                        tab = *it;
                        if (args.own_mu_k && !is_v) {
                            std::vector<double> mu(n, 0.0);
                            for (int32_t r = a; r < b; r++) {
                                for (int32_t i = 0; i < n; i++) {
                                    mu[i] += x[(size_t) r*n + i];
                                }
                            }
                            for (int32_t i = 0; i < n; i++) {
                                tab.mu[i] = (float) (mu[i]/(b - a));
                            }
                        }
                    }

                    const size_t o = (size_t) a*n;
                    llama_kvq_encode(tab, x.data() + o, b - a, codes.data() + o, gain.data() + a, nth);
                    llama_kvq_decode(tab, codes.data() + o, gain.data() + a, b - a, y.data() + o, nth);

                    // MSE of the uniform quantizer with the Gaussian-optimal step, in units of sigma^2 (Max 1960)
                    static const double gauss_mse[9] = { 1.0, 0.3634, 0.1188, 0.03744, 0.01154, 0.003490, 0.001035, 0.0003027, 0.00008741 };
                    for (int32_t c = 0; c < n; c++) {
                        tk.st.hist[tab.bits[c]]++;
                        if (!is_v) {
                            tk.st.gerr += gauss_mse[tab.bits[c]]*tab.sigma[c]*tab.sigma[c]*std::max(0, c1 - c0);
                        }
                    }

                    if (args.stats && c1 > c0) {
                        std::vector<float> d((size_t) (c1 - c0)*n);
                        std::vector<float> xc((size_t) (c1 - c0)*n);
                        for (int32_t r = c0; r < c1; r++) {
                            for (int32_t i = 0; i < n; i++) {
                                const size_t k  = (size_t) r*n + i;
                                const size_t od = (size_t) (r - c0)*n + i;
                                d[od]  = y[k] - x[k];
                                xc[od] = x[k] - tab.mu[i];
                                tk.st.err2 += (double) d[od]*d[od];
                                tk.st.sig2 += (double) xc[od]*xc[od];
                            }
                        }
                        std::vector<double> gd((size_t) n*n, 0.0);
                        std::vector<double> gx((size_t) n*n, 0.0);
                        llama_kvq_gram(d.data(),  c1 - c0, n, n, gd.data(), nth);
                        llama_kvq_gram(xc.data(), c1 - c0, n, n, gx.data(), nth);
                        for (size_t i = 0; i < m.size(); i++) {
                            tk.st.werr += gd[i]*m[i];
                            tk.st.wsig += gx[i]*m[i];
                        }
                    }

                    for (int32_t r = c0; r < c1; r++) {
                        memcpy(rows[tk.kv].data() + (size_t) r*n_gqa + tk.g*n, y.data() + (size_t) r*n, n*sizeof(float));
                    }
                    tk.tabs.push_back(std::move(tab));
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
                if (args.packed && !llama_memory_kvq_packed_set(mem, il, tk.g, tk.kv == 1, tk.tabs, tk.codes.data(), tk.gain.data())) {
                    LOG_ERR("%s: cannot set the packed rows of layer %d, %s head %d\n", __func__, il, tk.kv ? "V" : "K", tk.g);
                    return false;
                }
                tk.codes = {};
                tk.gain  = {};
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
                a.t_fit_max  = std::max(a.t_fit_max, tk.t_fit_max);
                t_fit_sum   += tk.t_fit;
                t_fit_max    = std::max(t_fit_max, tk.t_fit_max);
                if (fit && !args.tables_out_path.empty()) {
                    for (auto & tab : tk.tabs) {
                        tables_out.push_back(std::move(tab));
                    }
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
            const bool ok = !do_wr[kv] || (args.two_tier ?
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
        LOG_INF("%s: fit %.2f s summed over %zu tables (%.2f s per table set), max %.3f s per table\n", __func__,
                t_fit_sum, tables_out.size(), t_fit_sum/std::max(1, n_chunk), t_fit_max);
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

static void kvq_print_counters(const llama_kvq_counters_all & c, int64_t n_decode_tokens) {
    LOG("\n");
    LOG("kvq: packed counters: %lld ubatches, %lld positions, clean positions (no top-up firing in any layer) %.2f%%\n",
            (long long) c.n_ubatch, (long long) c.n_tokens, c.n_tokens ? 100.0*c.n_clean/c.n_tokens : NAN);
    LOG("kvq:   per layer: pf = rows requested / fetched per prefetch call; fire = firing share of (kv head, position) checks;\n");
    LOG("kvq:   stall = share of (layer, position) with a firing; per fire = rows per firing (kv head, position); top = share of the true top-m resident before the top-up\n");
    LOG("kvq:   times in ms per call: since the previous callback, readback, plan, gather (prefetch: worker), join (blocked on the worker), upload\n");
    llama_kvq_counters t;
    for (const auto & l : c.layers) {
        const double n_pf = (double) std::max<int64_t>(1, l.n_pf);
        const double n_tu = (double) std::max<int64_t>(1, l.n_tu);
        const double n_cb = (double) std::max<int64_t>(1, l.n_pf + l.n_tu + l.n_join);
        LOG("kvq: layer %2d: pf %7.1f / %7.1f  evict %9lld | fire %6.3f%%  stall %6.3f%%  redo %6.3f%%  per fire %5.1f  top %7.3f%% | "
            "%8.2f MiB | ms/call wait %.3f read %.3f plan %.3f gather %.3f join %.3f upload %.3f\n",
                l.il, l.pf_req/n_pf, l.pf_fetch/n_pf, (long long) l.evict,
                l.tu_check ? 100.0*l.tu_fire/l.tu_check : NAN, l.n_pos ? 100.0*l.tu_stall/l.n_pos : NAN, l.n_tu ? 100.0*l.tu_redo/n_tu : NAN,
                l.tu_fire ? (double) l.tu_fetch/l.tu_fire : NAN, l.top_n ? 100.0*l.top_hit/l.top_n : NAN,
                l.bytes/1048576.0, 1e3*l.t_wait/n_cb, 1e3*l.t_read/n_cb, 1e3*l.t_plan/n_cb, 1e3*l.t_gather/n_cb, 1e3*l.t_join/n_cb, 1e3*l.t_upload/n_cb);
        t.n_pf += l.n_pf; t.pf_req += l.pf_req; t.pf_fetch += l.pf_fetch; t.n_tu += l.n_tu; t.n_pos += l.n_pos;
        t.tu_check += l.tu_check; t.tu_fire += l.tu_fire; t.tu_stall += l.tu_stall; t.tu_redo += l.tu_redo; t.tu_fetch += l.tu_fetch;
        t.top_hit += l.top_hit; t.top_n += l.top_n; t.evict += l.evict; t.bytes += l.bytes;
        t.n_join += l.n_join;
        t.t_idle += l.t_idle;
        t.t_wait += l.t_wait; t.t_read += l.t_read; t.t_plan += l.t_plan; t.t_gather += l.t_gather; t.t_join += l.t_join; t.t_upload += l.t_upload;
    }
    const double nt = (double) std::max<int64_t>(1, n_decode_tokens);
    LOG("kvq: total   : pf calls %lld, rows requested %.1f / fetched %.1f per call; top-up checks %lld, fire %.3f%%, stall %.3f%%, redo %.3f%%, "
        "rows per fire %.1f, top resident %.3f%%\n",
            (long long) t.n_pf, t.pf_req/(double) std::max<int64_t>(1, t.n_pf), t.pf_fetch/(double) std::max<int64_t>(1, t.n_pf),
            (long long) t.tu_check, t.tu_check ? 100.0*t.tu_fire/t.tu_check : NAN, t.n_pos ? 100.0*t.tu_stall/t.n_pos : NAN,
            t.n_tu ? 100.0*t.tu_redo/t.n_tu : NAN, t.tu_fire ? (double) t.tu_fetch/t.tu_fire : NAN, t.top_n ? 100.0*t.top_hit/t.top_n : NAN);
    LOG("kvq: per token: rows fetched %.1f (prefetch %.1f, top-up %.1f), %.3f MiB host->device, evictions %.1f\n",
            (t.pf_fetch + t.tu_fetch)/nt, t.pf_fetch/nt, t.tu_fetch/nt, t.bytes/1048576.0/nt, t.evict/nt);
    // the prefetch plan and gather run on the worker, the top-up ones in the callback
    LOG("kvq: per token: decode thread ms: since previous callback %.3f, readback %.3f, join wait %.3f, upload %.3f; plan %.3f + gather %.3f (worker for the prefetch)\n",
            1e3*t.t_wait/nt, 1e3*t.t_read/nt, 1e3*t.t_join/nt, 1e3*t.t_upload/nt, 1e3*t.t_plan/nt, 1e3*t.t_gather/nt);
    if (t.t_idle > 0.0) {
        LOG("kvq: per token: GPU idle at the joins (LLAMA_KVQ_IDLE_PROBE) %.3f ms\n", 1e3*t.t_idle/nt);
    }
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
    const bool own_ctx = args.two_tier || args.packed || args.no_rot;
    auto llama_init = common_init_from_params(params, own_ctx);

    llama_context_ptr ctx_tt;

    auto * model = llama_init->model();
    if (model != nullptr && own_ctx) {
        llama_model_set_kvq_two_tier(model, args.two_tier);
        llama_model_set_kvq_packed(model, args.packed);
        if (args.packed || args.no_rot) {
            llama_model_set_attn_rot(model, false);
        }
        ctx_tt.reset(llama_init_from_model(model, common_context_params_to_llama(params)));
    }

    auto * ctx = own_ctx ? ctx_tt.get() : llama_init->context();
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
        if (args.variant == KVQ_VARIANT_ROT && (mi.n_embd_head & (mi.n_embd_head - 1)) != 0) {
            LOG_ERR("%s: rot variants need a power of 2 head size\n", __func__);
            return 1;
        }
        LOG_INF("%s: variant: %d, k_bits = %d, v_bits = %d, sinks = %d, recent = %d, attention layers %d..%d\n", __func__,
                (int) args.variant, args.k_bits, args.v_bits, args.sinks, args.recent, layer_ids.front(), layer_ids.back());
    }
    cap.il_first    = layer_ids.empty() ? -1 : *std::min_element(layer_ids.begin(), layer_ids.end());
    cap.n_head      = mi.n_head;
    cap.n_embd_head = mi.n_embd_head;
    cap.win0        = args.fit_chunk > 0 && !args.k_last ? 0 : std::max(0, s - args.q_win);
    cap.win1        = s;
    cap.gate_chunk  = args.fit_chunk;

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
    if (args.packed) {
        llama_kvq_params tp;
        tp.enabled       = true;
        tp.packed        = true;
        tp.n_sink        = args.sinks;
        tp.stats         = args.pk_stats;
        tp.prefetch_n    = args.pk_pf_n;
        tp.resident_cap  = args.pk_cap;
        tp.lead          = args.pk_lead;
        tp.sync_mode     = args.pk_sync;
        tp.sync_m        = args.pk_sync_m;
        tp.sync_min_hits = args.pk_sync_hits;
        tp.all_resident  = args.pk_all_res;
        if (!llama_memory_kvq_set_params(llama_get_memory(ctx), tp)) {
            LOG_ERR("%s: cannot switch to packed two-tier mode\n", __func__);
            return 1;
        }
        // the capture callback is not needed any more, and an eval callback splits the graph computation
        llama_context_set_eval_callback(ctx, nullptr, nullptr);

        llama_kvq_memory_info mi_pk;
        if (llama_memory_kvq_get_memory_info(llama_get_memory(ctx), mi_pk)) {
            LOG_INF("%s: packed: device codes %.1f MiB, tables %.1f MiB, resident %.1f MiB, other %.1f MiB; host store %.1f MiB (%s)\n", __func__,
                    mi_pk.codes/1048576.0, mi_pk.tables/1048576.0, mi_pk.resident/1048576.0, mi_pk.other_dev/1048576.0, mi_pk.host_store/1048576.0,
                    mi_pk.pinned ? "pinned" : "pageable");
        }
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

    std::vector<double> t_chunks;
    int32_t n_split_min = INT32_MAX;
    int32_t n_split_max = 0;
    int64_t t_chunk_start = 0;

    const int64_t t_score_start = ggml_time_us();
    t_chunk_start = t_score_start;
    const bool ok = kvq_score(ctx, ids, s, args.chunk, [&](int32_t i, const float * logits) {
        if (args.perf) {
            return true;
        }
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
        const int32_t n_split = llama_context_n_splits(ctx);
        n_split_min = std::min(n_split_min, n_split);
        n_split_max = std::max(n_split_max, n_split);
        // decode time only, the logits are processed after this
        llama_synchronize(ctx);
        t_chunks.push_back((ggml_time_us() - t_chunk_start)/1e3/n);
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
    }, [&]() {
        t_chunk_start = ggml_time_us();
    });
    if (!ok) {
        return 1;
    }
    const double t_score = (ggml_time_us() - t_score_start) / 1e6;
    LOG_INF("%s: score: %d tokens in %.2f s, %.1f t/s\n", __func__, e - s, t_score, (e - s) / t_score);
    LOG("kvq: scheduler splits per decode: %d .. %d\n", n_split_min, n_split_max);
    if (!t_chunks.empty()) {
        // the first chunks build and capture the graphs
        const size_t n_skip = std::min<size_t>(8, t_chunks.size()/4);
        std::vector<double> tc(t_chunks.begin() + n_skip, t_chunks.end());
        double sum = 0.0;
        for (double x : tc) {
            sum += x;
        }
        std::sort(tc.begin(), tc.end());
        LOG("kvq: perf: chunk %d, %zu chunks (first %zu skipped), decode only: ms/token mean %.3f, median %.3f, p10 %.3f, p90 %.3f -> %.2f t/s\n",
                args.chunk, tc.size(), n_skip, sum/tc.size(), tc[tc.size()/2], tc[tc.size()/10], tc[tc.size()*9/10], 1e3*tc.size()/sum);
    }
    if (args.packed) {
        llama_kvq_counters_all kc;
        if (llama_memory_kvq_get_counters(llama_get_memory(ctx), kc)) {
            kvq_print_counters(kc, e - s);
        }
    }

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
