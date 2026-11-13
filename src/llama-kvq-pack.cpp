#include "llama-kvq-pack.h"

#include "llama-impl.h"

#include <algorithm>
#include <numeric>

std::vector<int32_t> llama_kvq_pack_perm(const llama_kvq_table & t) {
    std::vector<int32_t> perm(t.n);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int32_t a, int32_t b) { return t.bits[a] > t.bits[b]; });
    return perm;
}

int32_t llama_kvq_pack_words(const llama_kvq_table & t) {
    int32_t sum = 0;
    for (int32_t c = 0; c < t.n; c++) {
        sum += t.bits[c];
    }
    return (sum + 31)/32;
}

void llama_kvq_pack_rows(const llama_kvq_table & t, int32_t n_words, const int8_t * codes, const float * gain, int64_t n_tok, int32_t * rows) {
    GGML_ASSERT(llama_kvq_pack_words(t) <= n_words);
    GGML_ASSERT(!t.is_v || gain != nullptr);

    const std::vector<int32_t> perm = llama_kvq_pack_perm(t);
    const int32_t n_row = n_words + (t.is_v ? 1 : 0);

    for (int64_t r = 0; r < n_tok; r++) {
        uint32_t     * row = (uint32_t *) rows + r*n_row;
        const int8_t * cr  = codes + r*t.n;
        std::fill(row, row + n_row, 0u);

        int32_t o = 0;
        for (int32_t c = 0; c < t.n; c++) {
            const int32_t b = t.bits[perm[c]];
            if (b == 0) {
                continue;
            }
            // offset binary: u = code + 2^(b-1) in [0, 2^b)
            const uint32_t u = (uint32_t) (cr[perm[c]] + (1 << (b - 1))) & ((1u << b) - 1);
            row[o/32] |= u << (o % 32);
            if (o % 32 + b > 32) {
                row[o/32 + 1] |= u >> (32 - o % 32);
            }
            o += b;
        }
        if (t.is_v) {
            row[n_words] = ggml_fp32_to_fp16(gain[r]);
        }
    }
}

void llama_kvq_pack_table(const llama_kvq_table & t, ggml_fp16_t * inv, float * aux) {
    const int32_t n = t.n;
    const std::vector<int32_t> perm = llama_kvq_pack_perm(t);

    for (int32_t c = 0; c < n; c++) {
        const int32_t pc = perm[c];
        for (int32_t i = 0; i < n; i++) {
            const ggml_fp16_t x = ggml_fp32_to_fp16(t.inv[(size_t) pc*n + i]);
            if (t.is_v) {
                inv[(size_t) c*n + i] = x; // [i, c]: i0 = dim
            } else {
                inv[(size_t) i*n + c] = x; // [c, i]: i0 = component
            }
        }
        aux[c]       = t.bits[pc] ? t.step[pc] : 0.0f;
        aux[2*n + c] = (float) t.bits[pc];
    }
    for (int32_t i = 0; i < n; i++) {
        aux[n + i] = t.mu[i];
    }
}
