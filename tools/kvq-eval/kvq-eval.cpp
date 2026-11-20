// teacher-forced KL harness for the two-tier KV cache (kvq)
//
// input: int32 token file ids[0..e], ids[e] = <|im_end|>, and the prefix length s (ids[s] = <|im_start|>, ids[s+1] = assistant)
// prefill ids[0:s], then feed ids[s:e] in chunks of --kvq-chunk with logits at every position
// fed position i predicts ids[s+i+1]; the first 2 fed positions are not scored, so the scored targets are ids[s+3..e]
// --kvq-save-ref writes the log-softmax of every scored position, --kvq-ref / --kvq-ref2 compare against such a file (KL, top-1, NLL)
//
// without --kvq-lib the context has the plain KV cache (references); --kvq-no-rot: attention rotation off (references of the kvq path)
// --kvq-perf: no logits processing, decode times per chunk only
//
// --kvq-lib: the two-tier cache with the setup of llama-server --kvq (common_params::kvq), K2V1 tables fitted per chunk during the prefill
//   chunks of --kvq-fit-chunk (default 8192); waits for the fitter after the prefill
//   --kvq-pf-n 1024, --kvq-cap 0 (max_batch x pf-n), --kvq-lead 2: the prefetch of the resident rows (llama_kvq_params)
//   --kvq-scr-block N: the scratch attention of a large ubatch in key blocks of N rows, 0: one attention with a mask (llama_kvq_params::scr_block)
//   the ring has max(batch, 4096) + 256 cells (--kvq-tail of the server), or --kvq-ring N cells
//   --kvq-rem own|merge, --kvq-min-rows N: the rows after the last complete chunk (llama_kvq_rem)
//   --kvq-rm-test: functional check of clear and seq_rm on the first chunk, no scoring
//   --kvq-lib-hold: after the prefill, finished chunks join the coarse region only at the end of a prompt (llama_memory_kvq_fit_set_hold)
//   --kvq-turns S1,S2,...: the prefix in turns: prefill up to S_i (batches at multiples of the batch size), the end of the prompt,
//                          --kvq-turn-gen N greedy tokens (1 per decode, not printed), back to S_i (recurrent checkpoint), the next turn
// --kvq-rb-test: ring rollbacks with and without an earlier rollback (also without --kvq-lib), no scoring
// --kvq-state-save / --kvq-state-load FNAME: the sequence state after the prefill (and the wait), or instead of the prefill;
//   --kvq-state-ckpt P: a recurrent checkpoint at P, rollback to it after the save or load and prefill again (FNAME.ckpt)
// --kvq-state-refuse F1,F2,...: each state file must be refused by the load (another version, a wrong section size), the memory must
//   stay empty, then a prefill of 2048 positions must work; exit 0 if all of that holds, no scoring

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../../src/llama-ext.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

static const char    kvq_magic[8] = { 'k', 'v', 'q', 'r', 'e', 'f', '0', '1' };
static const int32_t kvq_n_skip   = 2;

struct kvq_args {
    std::string tokens_path;
    std::string ref_out_path;
    std::string ref_in_path;
    std::string ref2_in_path; // a second reference, compared after the first
    std::string csv_path;
    int32_t     prefix_len = -1;
    int32_t     chunk      = 64;

    int32_t     pk_pf_n      = 1024;
    int32_t     pk_cap       = 0;
    int32_t     pk_lead      = 2;
    int32_t     pk_scr_block = 16384;

    bool        no_rot = false;
    bool        perf   = false;

    bool        lib_on       = false;
    int32_t     fit_chunk    = 8192;
    bool        rm_test      = false;
    bool        rb_test      = false;
    bool        lib_hold     = false;
    int32_t     lib_rem      = LLAMA_KVQ_REM_OWN;
    int32_t     lib_min_rows = 256;
    int32_t     bits_k       = 2;
    int32_t     bits_v       = 1;
    int32_t     lib_ring     = 0; // 0: max(batch, 4096) + 256
    std::vector<int32_t> turns;
    int32_t     turn_gen     = 0;

    std::string state_save;     // sequence state file after the prefill (and the wait)
    std::string state_load;     // sequence state file instead of the prefill
    std::vector<std::string> state_refuse; // state files that the load must refuse
    int32_t     state_ckpt = 0; // recurrent checkpoint at this position, rollback to it after the save or load and prefill again

    bool lib() const { return lib_on; }
};

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
        } else if (a == "--kvq-pf-n" && has_val) {
            args.pk_pf_n = std::stoi(argv[++i]);
        } else if (a == "--kvq-cap" && has_val) {
            args.pk_cap = std::stoi(argv[++i]);
        } else if (a == "--kvq-lead" && has_val) {
            args.pk_lead = std::stoi(argv[++i]);
        } else if (a == "--kvq-scr-block" && has_val) {
            args.pk_scr_block = std::stoi(argv[++i]);
        } else if (a == "--kvq-no-rot") {
            args.no_rot = true;
        } else if (a == "--kvq-perf") {
            args.perf = true;
        } else if (a == "--kvq-lib") {
            args.lib_on = true;
        } else if (a == "--kvq-fit-chunk" && has_val) {
            args.fit_chunk = std::stoi(argv[++i]);
        } else if (a == "--kvq-rm-test") {
            args.rm_test = true;
        } else if (a == "--kvq-rb-test") {
            args.rb_test = true;
        } else if (a == "--kvq-lib-hold") {
            args.lib_hold = true;
        } else if (a == "--kvq-rem" && has_val) {
            const std::string v = argv[++i];
            if (v != "own" && v != "merge") {
                LOG_ERR("%s: bad --kvq-rem '%s'\n", __func__, v.c_str());
                return false;
            }
            args.lib_rem = v == "merge" ? LLAMA_KVQ_REM_MERGE : LLAMA_KVQ_REM_OWN;
        } else if (a == "--kvq-bits-k" && has_val) {
            args.bits_k = std::stoi(argv[++i]);
        } else if (a == "--kvq-bits-v" && has_val) {
            args.bits_v = std::stoi(argv[++i]);
        } else if (a == "--kvq-min-rows" && has_val) {
            args.lib_min_rows = std::stoi(argv[++i]);
        } else if (a == "--kvq-ring" && has_val) {
            args.lib_ring = std::stoi(argv[++i]);
        } else if (a == "--kvq-turns" && has_val) {
            std::string v = argv[++i];
            for (size_t p = 0; p < v.size();) {
                const size_t q = v.find(',', p);
                args.turns.push_back(std::stoi(v.substr(p, q == std::string::npos ? std::string::npos : q - p)));
                p = q == std::string::npos ? v.size() : q + 1;
            }
        } else if (a == "--kvq-turn-gen" && has_val) {
            args.turn_gen = std::stoi(argv[++i]);
        } else if (a == "--kvq-state-save" && has_val) {
            args.state_save = argv[++i];
        } else if (a == "--kvq-state-load" && has_val) {
            args.state_load = argv[++i];
        } else if (a == "--kvq-state-refuse" && has_val) {
            const std::string v = argv[++i];
            for (size_t p = 0; p < v.size();) {
                const size_t q = v.find(',', p);
                args.state_refuse.push_back(v.substr(p, q == std::string::npos ? std::string::npos : q - p));
                p = q == std::string::npos ? v.size() : q + 1;
            }
        } else if (a == "--kvq-state-ckpt" && has_val) {
            args.state_ckpt = std::stoi(argv[++i]);
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
                "       [--kvq-no-rot] [--kvq-perf] [--kvq-rb-test] [--kvq-state-save|--kvq-state-load FNAME [--kvq-state-ckpt P]]\n"
                "       [--kvq-lib [--kvq-fit-chunk 8192] [--kvq-pf-n 1024] [--kvq-cap 0] [--kvq-lead 2] [--kvq-scr-block 16384]\n"
                "                  [--kvq-ring N] [--kvq-rem own|merge] [--kvq-min-rows 256] [--kvq-rm-test] [--kvq-lib-hold] [--kvq-turns S1,S2,... [--kvq-turn-gen N]]] [common args]\n", argv[0]);
        return false;
    }
    if (args.chunk <= 0) {
        LOG_ERR("%s: --kvq-chunk must be > 0\n", __func__);
        return false;
    }
    if (!args.lib() && (args.rm_test || args.lib_hold)) {
        LOG_ERR("%s: --kvq-rm-test and --kvq-lib-hold need --kvq-lib\n", __func__);
        return false;
    }
    if (args.lib() && (args.fit_chunk <= 0)) {
        LOG_ERR("%s: bad --kvq-fit-chunk\n", __func__);
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

// fill the memory with ids[p_begin:n_prefix], no logits, the batches end at multiples of n_batch
static bool kvq_prefill(llama_context * ctx, const std::vector<llama_token> & ids, int32_t n_prefix, int32_t n_batch, int32_t p_begin = 0) {
    common_batch batch(ctx);
    for (int32_t i = 0; i < n_prefix; i++) {
        batch.add(ids[i], i, 0, false);
    }
    for (int32_t i = p_begin, n = 0; i < n_prefix; i += n) {
        n = std::min((i/n_batch + 1)*n_batch, n_prefix) - i;
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
    LOG("kvq: packed counters: %lld ubatches, %lld positions\n", (long long) c.n_ubatch, (long long) c.n_tokens);
    LOG("kvq:   per layer: pf = rows requested / fetched per prefetch call; times in ms per call: since the previous callback, readback,\n");
    LOG("kvq:   plan, gather (worker), join (blocked on the worker), upload\n");
    llama_kvq_counters t;
    for (const auto & l : c.layers) {
        const double n_pf = (double) std::max<int64_t>(1, l.n_pf);
        const double n_cb = (double) std::max<int64_t>(1, l.n_pf + l.n_join);
        LOG("kvq: layer %2d: pf %7.1f / %7.1f  evict %9lld | %8.2f MiB | ms/call wait %.3f read %.3f plan %.3f gather %.3f join %.3f upload %.3f\n",
                l.il, l.pf_req/n_pf, l.pf_fetch/n_pf, (long long) l.evict,
                l.bytes/1048576.0, 1e3*l.t_wait/n_cb, 1e3*l.t_read/n_cb, 1e3*l.t_plan/n_cb, 1e3*l.t_gather/n_cb, 1e3*l.t_join/n_cb, 1e3*l.t_upload/n_cb);
        t.n_pf += l.n_pf; t.pf_req += l.pf_req; t.pf_fetch += l.pf_fetch;
        t.evict += l.evict; t.bytes += l.bytes; t.base_bytes += l.base_bytes; t.ring_bytes += l.ring_bytes; t.rb_bytes += l.rb_bytes; t.t_mirror += l.t_mirror;
        t.pf_sel += l.pf_sel; t.pf_cut += l.pf_cut; t.pf_lost += l.pf_lost;
        t.n_join += l.n_join;
        t.t_wait += l.t_wait; t.t_read += l.t_read; t.t_plan += l.t_plan; t.t_gather += l.t_gather; t.t_join += l.t_join; t.t_upload += l.t_upload;
    }
    const double nt = (double) std::max<int64_t>(1, n_decode_tokens);
    LOG("kvq: total   : pf calls %lld, rows requested %.1f / fetched %.1f per call\n",
            (long long) t.n_pf, t.pf_req/(double) std::max<int64_t>(1, t.n_pf), t.pf_fetch/(double) std::max<int64_t>(1, t.n_pf));
    LOG("kvq: total   : selected rows of the positions resident after the prefetch %.4f%% (%lld of %lld lost in %lld cut requests)\n",
            t.pf_sel ? 100.0 - 100.0*t.pf_lost/t.pf_sel : 100.0, (long long) t.pf_lost, (long long) t.pf_sel, (long long) t.pf_cut);
    LOG("kvq: per token: rows fetched %.1f, %.3f MiB host->device, evictions %.1f; base uploads %.3f MiB, tail uploads %.3f MiB host->device, readbacks %.3f MiB\n",
            t.pf_fetch/nt, t.bytes/1048576.0/nt, t.evict/nt, t.base_bytes/1048576.0/nt, t.ring_bytes/1048576.0/nt, t.rb_bytes/1048576.0/nt);
    LOG("kvq: per token: decode thread ms: since previous callback %.3f, readback %.3f, join wait %.3f, upload %.3f; plan %.3f + gather %.3f + mirror %.3f (worker)\n",
            1e3*t.t_wait/nt, 1e3*t.t_read/nt, 1e3*t.t_join/nt, 1e3*t.t_upload/nt, 1e3*t.t_plan/nt, 1e3*t.t_gather/nt, 1e3*t.t_mirror/nt);
}

// the scratch attention of the large ubatches, summed over the layers
static void kvq_print_scratch(llama_memory_t mem) {
    llama_kvq_counters_all kc;
    if (!llama_memory_kvq_get_counters(mem, kc)) {
        return;
    }
    llama_kvq_counters t;
    for (const auto & c : kc.layers) {
        t.n_scr      += c.n_scr;
        t.scr_bytes  += c.scr_bytes;
        t.t_scr      += c.t_scr;
    }
    LOG("kvq: scratch: %lld layer uploads, %.2f GiB, %.3f s in the callbacks\n",
            (long long) t.n_scr, t.scr_bytes/1073741824.0, t.t_scr);
}

static void kvq_print_progress(llama_memory_t mem) {
    llama_kvq_fit_progress pr;
    if (!llama_memory_kvq_fit_get_progress(mem, pr)) {
        return;
    }
    std::string tj;
    for (size_t i = 0; i < pr.t_job.size(); i++) {
        tj += string_format("%s%.2f/%.2f", i ? " " : "", pr.t_job[i], pr.t_wall[i]);
    }
    LOG("kvq: fit: coarse rows %d (%d chunks), host rows %d, running %d, ready %d, cancelled %d, drops %d%s\n",
            pr.n_coarse, pr.n_chunk, pr.n_host, pr.n_running, pr.n_ready, pr.n_cancel, pr.n_drop, pr.failed ? ", FAILED" : "");
    if (pr.n_tail_wait > 0 || pr.n_spill > 0 || pr.n_fold > 0 || pr.n_settle > 0) {
        LOG("kvq: fit: %d batches made room in the ring (%.2f s), %d spills, %d folds (%.2f s), %d end-of-prompt fits (%.2f s)\n",
                pr.n_tail_wait, pr.t_tail_wait, pr.n_spill, pr.n_fold, pr.t_fold, pr.n_settle, pr.t_settle);
    }
    if (!pr.t_job.empty()) {
        LOG("kvq: fit: decode thread per chunk: readback, metrics and queued graphs %.1f ms (with the wait for the queued ubatches), "
            "install %.1f ms; s per job since its start/since the statistics: %s\n",
                1e3*pr.t_take/pr.t_job.size(), 1e3*pr.t_install/pr.t_job.size(), tj.c_str());
    }
}

// logits of the last position of ids[p0:p1] at positions [p0, p1), as one batch
static bool kvq_decode_last(llama_context * ctx, const std::vector<llama_token> & ids, int32_t p0, int32_t p1, int32_t n_vocab, std::vector<float> & out) {
    common_batch batch(ctx);
    for (int32_t p = p0; p < p1; p++) {
        batch.add(ids[p], p, 0, p == p1 - 1);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s: decode of [%d, %d) failed\n", __func__, p0, p1);
        return false;
    }
    const float * l = llama_get_logits_ith(ctx, -1);
    out.assign(l, l + n_vocab);
    return true;
}

// KL(a || b) of the softmax of two logit vectors, and whether the argmax is the same
static double kvq_kl_logits(const std::vector<float> & a, const std::vector<float> & b, bool & same_top) {
    const int32_t n = (int32_t) a.size();
    std::vector<float> la(n), lb(n);
    same_top = kvq_log_softmax(a.data(), n, la.data()) == kvq_log_softmax(b.data(), n, lb.data());
    double kl = 0.0;
    for (int32_t v = 0; v < n; v++) {
        kl += std::exp((double) la[v])*((double) la[v] - lb[v]);
    }
    return kl;
}

// clear and seq_rm in the fit mode around the end of the first chunk C: the memory is empty and in the fit mode
// rem, min_rows, min_first: the layout of the rows after the last complete chunk (llama_kvq_rem)
static bool kvq_rm_test(llama_context * ctx, const std::vector<llama_token> & ids, int32_t C, int32_t n_vocab, int32_t rem, int32_t min_rows, int32_t min_first) {
    llama_memory_t mem = llama_get_memory(ctx);
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    if ((int32_t) ids.size() < 2*C + n_batch + 1100) {
        LOG_ERR("%s: needs at least %d tokens\n", __func__, 2*C + n_batch + 1100);
        return false;
    }

    bool ok = true;
    auto check = [&](bool cond, const char * what) {
        LOG("kvq: rm-test: %-80s %s\n", what, cond ? "ok" : "FAILED");
        ok = ok && cond;
    };
    auto coarse = [&]() {
        llama_kvq_fit_progress pr;
        llama_memory_kvq_fit_get_progress(mem, pr);
        return pr;
    };
    auto compare = [&](const std::vector<float> & a, const std::vector<float> & b, const char * what) {
        bool same = false;
        const double kl = kvq_kl_logits(a, b, same);
        LOG("kvq: rm-test: %s: KL %.3e, same top-1 %d\n", what, kl, same ? 1 : 0);
    };
    // the end of the coarse region after the end of a prompt of n positions
    auto layout_end = [&](int32_t n) {
        const int32_t n_full = n/C;
        int32_t e = n_full*C;
        if (n > e && (rem == LLAMA_KVQ_REM_MERGE || n_full == 0 || n - e >= min_rows)) {
            e = n;
        }
        return e < min_first ? 0 : e;
    };

    std::vector<float> la, lb, lc;

    // a job in flight (a large ubatch 64 positions after the chunk takes it, merge: after the next chunk), then a small batch:
    // the job joins before it; then clear
    const int32_t a0 = (rem == LLAMA_KVQ_REM_MERGE ? 2*C : C) + n_batch + 70;
    if (!kvq_prefill(ctx, ids, a0, n_batch)) {
        return false;
    }
    check(coarse().n_running + coarse().n_ready + coarse().n_chunk == 1, "prefill past the first chunk starts its job");
    check(coarse().n_host >= C, "and its rows are in the host store");
    if (!kvq_decode_last(ctx, ids, a0, a0 + 1, n_vocab, lc)) {
        return false;
    }
    check(coarse().n_coarse == C && coarse().n_running + coarse().n_ready == 0, "a small batch adds the job first");
    llama_memory_clear(mem, true);
    check(coarse().n_coarse == 0 && coarse().n_host == 0 && llama_memory_seq_pos_max(mem, 0) == -1, "clear empties the memory");

    // the end of the prompt fits the rows after the first chunk, a small batch after it
    const int32_t a = C + 497;
    if (!kvq_prefill(ctx, ids, a, n_batch) || !llama_memory_kvq_fit_wait(mem)) {
        return false;
    }
    check(coarse().n_coarse == layout_end(a) && coarse().n_running == 0, "the end of the prompt fits up to the layout end");
    if (!kvq_decode_last(ctx, ids, a, a + 4, n_vocab, la) || !llama_memory_kvq_fit_wait(mem)) {
        return false;
    }
    check(coarse().n_coarse == layout_end(a + 4), "the next end of the prompt fits the 4 new rows too");

    // into the coarse rows after the chunk: the recurrent state rolls back inside the last ubatch, the coarse region back to the grid
    check(llama_memory_seq_rm(mem, 0, a + 1, -1), "seq_rm(a + 1, end) into the coarse rows succeeds");
    check(coarse().n_coarse == std::min(C, layout_end(a + 4)) && coarse().n_host == (layout_end(a + 4) > C ? a + 1 : C), "the coarse region is back at the chunk grid, the host store keeps the rows");
    check(llama_memory_seq_pos_max(mem, 0) == a, "last position is a");
    if (!kvq_decode_last(ctx, ids, a + 1, a + 4, n_vocab, lb)) {
        return false;
    }
    compare(la, lb, "logits at a + 3 before (packed) and after the removal (scratch)");
    if (!llama_memory_kvq_fit_wait(mem)) {
        return false;
    }
    check(coarse().n_coarse == layout_end(a + 4) && coarse().n_host == std::max(layout_end(a + 4), C), "the end of the prompt fits them again");

    // the ring only (draft rollback): a ubatch of 4 after the coarse region, remove 3 of them, decode them again
    const int32_t b = a + 4;
    if (!kvq_decode_last(ctx, ids, b, b + 4, n_vocab, la)) {
        return false;
    }
    const auto pr0 = coarse();
    const int64_t t0 = ggml_time_us();
    check(llama_memory_seq_rm(mem, 0, b + 1, -1), "seq_rm(b + 1, end) in the ring succeeds");
    const double t_rm = (ggml_time_us() - t0)/1e3;
    const auto pr1 = coarse();
    check(pr1.n_coarse == pr0.n_coarse && pr1.n_host == pr0.n_host && pr1.n_cancel == pr0.n_cancel && pr1.n_drop == pr0.n_drop, "the coarse region is not changed");
    LOG("kvq: rm-test: ring seq_rm took %.3f ms\n", t_rm);
    check(llama_memory_seq_rm(mem, 0, b + 10, -1), "seq_rm of an empty range succeeds");
    if (!kvq_decode_last(ctx, ids, b + 1, b + 4, n_vocab, lb)) {
        return false;
    }
    compare(la, lb, "logits at b + 3 before and after the ring removal (packed path)");

    // too far back for the recurrent state: fails, nothing changes
    check(!llama_memory_seq_rm(mem, 0, 100, -1), "seq_rm(100, end) fails (no recurrent snapshot)");
    check(coarse().n_coarse == pr0.n_coarse && llama_memory_seq_pos_max(mem, 0) == b + 3, "and changes nothing");

    // all of the sequence, as the server's prompt clear (seq_id -1 fails in the recurrent memory)
    check(llama_memory_seq_rm(mem, 0, -1, -1), "seq_rm(0, -1, -1) succeeds");
    check(coarse().n_coarse == 0 && coarse().n_host == 0 && llama_memory_seq_pos_max(mem, 0) == -1, "the memory is empty, no coarse rows");

    // a few rows after the chunk: own with fewer than min_rows stays precise in the ring
    if (!kvq_prefill(ctx, ids, C + 8, n_batch) || !llama_memory_kvq_fit_wait(mem)) {
        return false;
    }
    check(coarse().n_coarse == layout_end(C + 8) && !coarse().failed, "a new prefill fits up to the layout end again");
    check(llama_memory_seq_pos_max(mem, 0) == C + 7, "last position is C + 7");
    if (!kvq_decode_last(ctx, ids, C + 8, C + 12, n_vocab, lc)) {
        return false;
    }

    // the complete chunk and a group or own chunk after it: across the chunk boundary
    if (!kvq_prefill(ctx, ids, C + 1000, n_batch, C + 12) || !llama_memory_kvq_fit_wait(mem)) {
        return false;
    }
    check(coarse().n_coarse == layout_end(C + 1000), "the end of the prompt fits up to the layout end");
    if (!kvq_decode_last(ctx, ids, C + 1000, C + 1002, n_vocab, lc)) {
        return false;
    }

    // the geometry of the earlier version of this test: the first chunk coarse, 2 positions in the ring, a ring removal after them
    check(llama_memory_seq_rm(mem, 0, -1, -1), "seq_rm(0, -1, -1) succeeds");
    if (!kvq_prefill(ctx, ids, C - 2, n_batch) || !kvq_decode_last(ctx, ids, C - 2, C + 2, n_vocab, lc) || !llama_memory_kvq_fit_wait(mem)) {
        return false;
    }
    check(coarse().n_coarse == layout_end(C + 2), "the end of the prompt fits up to the layout end");
    if (!kvq_decode_last(ctx, ids, C + 2, C + 6, n_vocab, la) || !llama_memory_seq_rm(mem, 0, C + 3, -1) ||
        !kvq_decode_last(ctx, ids, C + 3, C + 6, n_vocab, lb)) {
        return false;
    }
    compare(la, lb, "logits at C + 5 before and after the ring removal (packed path, earlier geometry)");
    kvq_print_progress(mem);
    LOG("kvq: rm-test: %s\n", ok ? "all checks passed" : "SOME CHECKS FAILED");
    return ok;
}

// --kvq-rb-test: a 4-token batch at b = C + 501, removal of its last 3 positions, the 3 decoded again; the logits at b + 3
// are compared; before it, with or without an earlier removal of the same kind at a = C + 497
static bool kvq_rb_test(llama_context * ctx, const std::vector<llama_token> & ids, int32_t C, int32_t n_vocab, bool lib) {
    llama_memory_t mem = llama_get_memory(ctx);
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    const int32_t a = C + 497;
    const int32_t b = a + 4;
    if ((int32_t) ids.size() < b + 8) {
        return false;
    }
    auto wait = [&]() { return !lib || llama_memory_kvq_fit_wait(mem); };
    for (int prior = 0; prior < 2; ++prior) {
        std::vector<float> la, lb, lx;
        llama_memory_clear(mem, true);
        if (!kvq_prefill(ctx, ids, a, n_batch) || !wait() || !kvq_decode_last(ctx, ids, a, a + 4, n_vocab, lx) || !wait()) {
            return false;
        }
        if (prior && (!llama_memory_seq_rm(mem, 0, a + 1, -1) || !kvq_decode_last(ctx, ids, a + 1, a + 4, n_vocab, lx) || !wait())) {
            return false;
        }
        if (!kvq_decode_last(ctx, ids, b, b + 4, n_vocab, la) || !llama_memory_seq_rm(mem, 0, b + 1, -1) ||
            !kvq_decode_last(ctx, ids, b + 1, b + 4, n_vocab, lb)) {
            return false;
        }
        bool same = false;
        const double kl = kvq_kl_logits(la, lb, same);
        LOG("kvq: rb-test: %s, earlier removal %d: logits at b + 3 before and after the removal: KL %.3e, same top-1 %d\n",
                lib ? "two-tier" : "plain cache", prior, kl, same ? 1 : 0);
    }
    // no removal: the same positions as [b, b + 1) and then [b + 1, b + 4), against [b, b + 4) in one batch
    {
        std::vector<float> la, lb, lx;
        for (int split = 0; split < 2; ++split) {
            llama_memory_clear(mem, true);
            if (!kvq_prefill(ctx, ids, a, n_batch) || !wait() || !kvq_decode_last(ctx, ids, a, a + 4, n_vocab, lx) || !wait()) {
                return false;
            }
            std::vector<float> & out = split ? lb : la;
            if (split ? !kvq_decode_last(ctx, ids, b, b + 1, n_vocab, lx) || !kvq_decode_last(ctx, ids, b + 1, b + 4, n_vocab, out)
                      : !kvq_decode_last(ctx, ids, b, b + 4, n_vocab, out)) {
                return false;
            }
        }
        bool same = false;
        const double kl = kvq_kl_logits(la, lb, same);
        LOG("kvq: rb-test: %s, no removal: logits at b + 3 from one batch of 4 and from batches of 1 and 3: KL %.3e, same top-1 %d\n",
                lib ? "two-tier" : "plain cache", kl, same ? 1 : 0);
    }
    return true;
}

// the prefix in turns, see --kvq-turns: the memory is empty
static bool kvq_turns(llama_context * ctx, const std::vector<llama_token> & ids, const std::vector<int32_t> & turns, int32_t n_gen, int32_t n_batch, int32_t n_vocab, bool lib) {
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    int32_t p = 0;
    for (size_t t = 0; t < turns.size(); ++t) {
        const int32_t s_t = turns[t];
        const int64_t t0  = ggml_time_us();
        if (!kvq_prefill(ctx, ids, s_t, n_batch, p)) {
            return false;
        }
        const int64_t t1 = ggml_time_us();
        if (lib && !llama_memory_kvq_fit_wait(mem)) {
            return false;
        }
        const int64_t t2 = ggml_time_us();
        llama_kvq_fit_progress pr;
        llama_memory_kvq_fit_get_progress(mem, pr);
        LOG("kvq: turn %zu: prefill [%d, %d) %.2f s, end-of-prompt fit %.3f s, coarse rows %d, host rows %d\n",
                t, p, s_t, (t1 - t0)*1e-6, (t2 - t1)*1e-6, pr.n_coarse, pr.n_host);
        p = s_t;
        if (t + 1 == turns.size()) {
            break;
        }
        // a reply, then back to the end of the prompt (the recurrent state from a checkpoint)
        std::vector<uint8_t> ckpt(llama_state_seq_get_size_ext(ctx, 0, fl));
        if (ckpt.empty() || llama_state_seq_get_data_ext(ctx, ckpt.data(), ckpt.size(), 0, fl) != ckpt.size()) {
            LOG_ERR("%s: cannot save the recurrent state\n", __func__);
            return false;
        }
        llama_token tok = ids[s_t];
        common_batch batch(ctx);
        for (int32_t i = 0; i < n_gen; ++i) {
            batch.clear();
            batch.add(tok, s_t + i, 0, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: decode failed at pos %d\n", __func__, s_t + i);
                return false;
            }
            tok = kvq_argmax(llama_get_logits_ith(ctx, -1), n_vocab);
        }
        if (llama_state_seq_set_data_ext(ctx, ckpt.data(), ckpt.size(), 0, fl) != ckpt.size() || !llama_memory_seq_rm(mem, 0, s_t, -1)) {
            LOG_ERR("%s: rollback to position %d failed\n", __func__, s_t);
            return false;
        }
        LOG("kvq: turn %zu: %d generated tokens, back to position %d\n", t, n_gen, s_t);
    }
    return true;
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

    // --kvq-lib: the two-tier cache of the server (common_params::kvq), set up by common_init_from_params
    if (args.lib()) {
        params.kvq.enabled      = true;
        params.kvq.chunk        = args.fit_chunk;
        params.kvq.rem          = args.lib_rem;
        params.kvq.min_rows     = args.lib_min_rows;
        params.kvq.bits_k       = args.bits_k;
        params.kvq.bits_v       = args.bits_v;
        params.kvq.tail         = args.lib_ring;
        params.kvq.prefetch_n   = args.pk_pf_n;
        params.kvq.resident_cap = args.pk_cap;
        params.kvq.lead         = args.pk_lead;
        params.kvq.scr_block    = args.pk_scr_block;
    }
    if (args.rm_test || args.rb_test) {
        // recurrent snapshots for a seq_rm into the last ubatch
        params.n_rs_seq = 4;
    }
    // no warmup decode in the measured context
    if (args.lib() || args.no_rot) {
        params.warmup = false;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // --kvq-no-rot without --kvq-lib: the attention rotation is a model flag, so the context is created after it
    const bool own_ctx = args.no_rot && !args.lib();
    auto llama_init = common_init_from_params(params, own_ctx);

    llama_context_ptr ctx_own;

    auto * model = llama_init->model();
    if (model != nullptr && own_ctx) {
        llama_model_set_attn_rot(model, false);
        ctx_own.reset(llama_init_from_model(model, common_context_params_to_llama(params)));
    }

    auto * ctx = own_ctx ? ctx_own.get() : llama_init->context();
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

    if (args.rm_test) {
        return kvq_rm_test(ctx, ids, args.fit_chunk, n_vocab, args.lib_rem, args.lib_min_rows, args.pk_pf_n + llama_kvq_params().n_sink) ? 0 : 1;
    }

    if (args.rb_test) {
        return kvq_rb_test(ctx, ids, args.fit_chunk, n_vocab, args.lib()) ? 0 : 1;
    }

    if (!args.state_refuse.empty()) {
        std::vector<llama_token> toks(ids.size());
        for (const auto & f : args.state_refuse) {
            size_t n_tok = 0;
            const size_t n_read = llama_state_seq_load_file(ctx, f.c_str(), 0, toks.data(), toks.size(), &n_tok);
            const llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
            if (n_read != 0 || pos >= 0) {
                LOG_ERR("%s: state %s: expected a refusal and an empty memory, got %zu bytes read, last position %d\n", __func__, f.c_str(), n_read, (int) pos);
                return 1;
            }
            LOG("kvq: state: refused %s, the memory is empty\n", f.c_str());
        }
        const int32_t n_chk = std::min<int32_t>(2048, s);
        if (!kvq_prefill(ctx, ids, n_chk, params.n_batch) || (args.lib() && !llama_memory_kvq_fit_wait(llama_get_memory(ctx)))) {
            LOG_ERR("%s: the prefill after the refused loads failed\n", __func__);
            return 1;
        }
        LOG("kvq: state: prefill of %d positions after the refused loads ok, last position %d\n", n_chk, (int) llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
        return 0;
    }

    const int32_t ckpt = args.state_ckpt;
    if (ckpt < 0 || ckpt >= s) {
        LOG_ERR("%s: --kvq-state-ckpt must be in [1, %d)\n", __func__, s);
        return 1;
    }
    const llama_state_seq_flags fl_part = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> ckpt_data;

    const int64_t t_prefill_start = ggml_time_us();
    if (!args.state_load.empty()) {
        std::vector<llama_token> toks(ids.size());
        size_t n_tok = 0;
        const int64_t t0 = ggml_time_us();
        const size_t n_read = llama_state_seq_load_file(ctx, args.state_load.c_str(), 0, toks.data(), toks.size(), &n_tok);
        const double t_load = (ggml_time_us() - t0)/1e6;
        if (n_read == 0 || n_tok != (size_t) s || !std::equal(toks.begin(), toks.begin() + s, ids.begin())) {
            LOG_ERR("%s: cannot load the state %s, or its tokens are not the prefix\n", __func__, args.state_load.c_str());
            return 1;
        }
        LOG("kvq: state: loaded %.1f MiB in %.2f s (%.2f GB/s), last position %d\n", n_read/1048576.0, t_load, n_read/t_load/1e9,
                (int) llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
        if (ckpt > 0) {
            std::ifstream f(args.state_load + ".ckpt", std::ios::binary | std::ios::ate);
            ckpt_data.resize(f ? (size_t) f.tellg() : 0);
            f.seekg(0);
            if (ckpt_data.empty() || !f.read((char *) ckpt_data.data(), ckpt_data.size())) {
                LOG_ERR("%s: cannot read %s.ckpt\n", __func__, args.state_load.c_str());
                return 1;
            }
        }
    } else if (ckpt > 0) {
        if (!kvq_prefill(ctx, ids, ckpt, params.n_batch)) {
            return 1;
        }
        ckpt_data.resize(llama_state_seq_get_size_ext(ctx, 0, fl_part));
        if (ckpt_data.empty() || llama_state_seq_get_data_ext(ctx, ckpt_data.data(), ckpt_data.size(), 0, fl_part) != ckpt_data.size() ||
                !kvq_prefill(ctx, ids, s, params.n_batch, ckpt)) {
            return 1;
        }
    } else if (!args.turns.empty()) {
        std::vector<int32_t> turns = args.turns;
        turns.push_back(s);
        for (size_t t = 0; t < turns.size(); ++t) {
            if (turns[t] <= (t ? turns[t - 1] : 0) || turns[t] > s || ckpt > 0) {
                LOG_ERR("%s: --kvq-turns needs ascending positions below the prefix length and no --kvq-state-ckpt\n", __func__);
                return 1;
            }
        }
        if (!kvq_turns(ctx, ids, turns, args.turn_gen, params.n_batch, n_vocab, args.lib())) {
            return 1;
        }
    } else if (!kvq_prefill(ctx, ids, s, params.n_batch)) {
        return 1;
    }
    const double t_prefill = (ggml_time_us() - t_prefill_start) / 1e6;
    LOG_INF("%s: prefill: %d tokens in %.2f s, %.1f t/s\n", __func__, s, t_prefill, s / t_prefill);
    if (args.lib()) {
        kvq_print_scratch(llama_get_memory(ctx));
    }

    if (args.lib()) {
        const int64_t t0 = ggml_time_us();
        if (!llama_memory_kvq_fit_wait(llama_get_memory(ctx))) {
            return 1;
        }
        LOG_INF("%s: waited %.2f s for the fitter\n", __func__, (ggml_time_us() - t0)/1e6);
        kvq_print_progress(llama_get_memory(ctx));
    }

    if (!args.state_save.empty()) {
        const int64_t t0 = ggml_time_us();
        const size_t n_write = llama_state_seq_save_file(ctx, args.state_save.c_str(), 0, ids.data(), s);
        const double t_save = (ggml_time_us() - t0)/1e6;
        if (n_write == 0) {
            LOG_ERR("%s: cannot save the state to %s\n", __func__, args.state_save.c_str());
            return 1;
        }
        LOG("kvq: state: saved %.1f MiB in %.2f s (%.2f GB/s)\n", n_write/1048576.0, t_save, n_write/t_save/1e9);
        if (ckpt > 0) {
            std::ofstream f(args.state_save + ".ckpt", std::ios::binary);
            if (!f.write((const char *) ckpt_data.data(), ckpt_data.size())) {
                LOG_ERR("%s: cannot write %s.ckpt\n", __func__, args.state_save.c_str());
                return 1;
            }
        }
        if (args.lib()) {
            kvq_print_progress(llama_get_memory(ctx));
        }
    }

    // rollback to the checkpoint (into a coarse chunk), the rest of the prefix again
    if (ckpt > 0) {
        llama_memory_t mem = llama_get_memory(ctx);
        if (llama_state_seq_set_data_ext(ctx, ckpt_data.data(), ckpt_data.size(), 0, fl_part) != ckpt_data.size() || !llama_memory_seq_rm(mem, 0, ckpt, -1)) {
            LOG_ERR("%s: rollback to position %d failed\n", __func__, ckpt);
            return 1;
        }
        if (args.lib()) {
            kvq_print_progress(mem);
        }
        if (!kvq_prefill(ctx, ids, s, params.n_batch, ckpt) || (args.lib() && !llama_memory_kvq_fit_wait(mem))) {
            return 1;
        }
        LOG("kvq: state: rolled back to position %d and processed the prefix again\n", ckpt);
        if (args.lib()) {
            kvq_print_progress(mem);
        }
    }
    if (args.lib_hold && !llama_memory_kvq_fit_set_hold(llama_get_memory(ctx), true)) {
        return 1;
    }

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
    std::vector<int>    busy_chunks; // fit mode: a job was running at the start of the chunk
    std::vector<int>    coarse_chunks;
    int  busy   = 0;
    int  coarse = 0;
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
        busy_chunks.push_back(busy);
        coarse_chunks.push_back(coarse);
        return true;
    }, [&]() {
        llama_kvq_fit_progress pr;
        if (args.lib() && llama_memory_kvq_fit_get_progress(llama_get_memory(ctx), pr)) {
            busy   = pr.n_running > 0;
            coarse = pr.n_coarse;
        }
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
        if (args.lib()) {
            // by fitter state and coarse region
            std::map<std::pair<int, int>, std::vector<double>> by;
            for (size_t i = n_skip; i < t_chunks.size(); i++) {
                by[{ coarse_chunks[i], busy_chunks[i] }].push_back(t_chunks[i]);
            }
            for (auto & it : by) {
                auto & v = it.second;
                double sv = 0.0;
                for (double x : v) {
                    sv += x;
                }
                std::sort(v.begin(), v.end());
                LOG("kvq: perf: coarse rows %d, fitter %s: %zu chunks, ms/token mean %.3f, median %.3f\n",
                        it.first.first, it.first.second ? "busy" : "idle", v.size(), sv/v.size(), v[v.size()/2]);
            }
        }
    }
    if (args.lib()) {
        kvq_print_progress(llama_get_memory(ctx));
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
