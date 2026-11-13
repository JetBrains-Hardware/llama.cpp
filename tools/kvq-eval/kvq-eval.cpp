// teacher-forced KL harness for KV cache experiments
//
// input: int32 token file ids[0..e], ids[e] = <|im_end|>, and the prefix length s (ids[s] = <|im_start|>, ids[s+1] = assistant)
// prefill ids[0:s], then feed ids[s:e] in chunks with logits at every position
// fed position i predicts ids[s+i+1]; the first 2 fed positions are not scored, so the scored targets are ids[s+3..e]

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static const char    kvq_magic[8] = { 'k', 'v', 'q', 'r', 'e', 'f', '0', '1' };
static const int32_t kvq_n_skip   = 2;

struct kvq_args {
    std::string tokens_path;
    std::string ref_out_path;
    std::string ref_in_path;
    std::string csv_path;
    int32_t     prefix_len = -1;
    int32_t     chunk      = 64;
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
        } else if (a == "--kvq-csv" && has_val) {
            args.csv_path = argv[++i];
        } else if (a.rfind("--kvq-", 0) == 0) {
            LOG_ERR("%s: unknown or incomplete argument '%s'\n", __func__, a.c_str());
            return false;
        } else {
            argv[n++] = argv[i];
        }
    }
    argc = n;

    if (args.tokens_path.empty() || args.prefix_len < 0) {
        LOG_ERR("usage: %s -m model.gguf --kvq-tokens ids.bin --kvq-prefix-len s [--kvq-chunk 64] [--kvq-save-ref ref.bin] [--kvq-ref ref.bin [--kvq-csv out.csv]] [common args]\n", argv[0]);
        return false;
    }
    if (args.chunk <= 0) {
        LOG_ERR("%s: --kvq-chunk must be > 0\n", __func__);
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

// feed ids[s:e] in chunks, call on_pos(i, logits) for each fed position i >= kvq_n_skip
template <typename F>
static bool kvq_score(llama_context * ctx, const std::vector<llama_token> & ids, int32_t s, int32_t chunk, F && on_pos) {
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

static void kvq_print_compare(const std::vector<kvq_pos_stat> & stats) {
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

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
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

    llama_memory_clear(llama_get_memory(ctx), true);

    const int64_t t_prefill_start = ggml_time_us();
    if (!kvq_prefill(ctx, ids, s, params.n_batch)) {
        return 1;
    }
    const double t_prefill = (ggml_time_us() - t_prefill_start) / 1e6;
    LOG_INF("%s: prefill: %d tokens in %.2f s, %.1f t/s\n", __func__, s, t_prefill, s / t_prefill);

    std::vector<float> lp(n_vocab);
    std::vector<float> lp_ref(n_vocab);
    std::vector<kvq_pos_stat> stats;
    stats.reserve(n_scored);
    double nll = 0.0;

    const int64_t t_score_start = ggml_time_us();
    const bool ok = kvq_score(ctx, ids, s, args.chunk, [&](int32_t i, const float * logits) {
        const llama_token target = ids[s + i + 1];
        const int32_t top = kvq_log_softmax(logits, n_vocab, lp.data());
        nll -= lp[target];

        if (ref_out.is_open() && !ref_out.write((const char *) lp.data(), n_vocab * sizeof(float))) {
            LOG_ERR("%s: failed to write reference\n", __func__);
            return false;
        }
        if (ref_in.is_open()) {
            if (!ref_in.read((char *) lp_ref.data(), n_vocab * sizeof(float))) {
                LOG_ERR("%s: reference file is truncated\n", __func__);
                return false;
            }
            double kl = 0.0;
            for (int32_t v = 0; v < n_vocab; v++) {
                kl += std::exp((double) lp_ref[v]) * ((double) lp_ref[v] - lp[v]);
            }
            stats.push_back({ kl, -lp_ref[target], -lp[target], kvq_argmax(lp_ref.data(), n_vocab) == top });
        }
        return true;
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

    if (ref_in.is_open()) {
        kvq_print_compare(stats);

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
