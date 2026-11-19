#include "llama-kvq-pack.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
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

void llama_kvq_pack_table(const llama_kvq_table & t, int8_t * inv, float * aux) {
    const int32_t n = t.n;
    const std::vector<int32_t> perm = llama_kvq_pack_perm(t);

    for (int32_t c = 0; c < n; c++) {
        const int32_t pc  = perm[c];
        const float * row = t.inv.data() + (size_t) pc*n;
        // one scale per component: max |inv[c][:]| -> 127; a 0-bit component or a row of zeros gives zeros and step 0
        float m = 0.0f;
        for (int32_t i = 0; i < n && t.bits[pc]; i++) {
            m = std::max(m, std::fabs(row[i]));
        }
        const float s = m/127.0f;
        for (int32_t i = 0; i < n; i++) {
            const int8_t x = s > 0.0f ? (int8_t) std::clamp(std::nearbyint(row[i]/s), -127.0f, 127.0f) : 0;
            if (t.is_v) {
                inv[(size_t) c*n + i] = x; // [i, c]: i0 = dim
            } else {
                inv[(size_t) i*n + c] = x; // [c, i]: i0 = component
            }
        }
        aux[c]       = t.bits[pc] ? t.step[pc]*s : 0.0f;
        aux[2*n + c] = (float) t.bits[pc];
    }
    for (int32_t i = 0; i < n; i++) {
        aux[n + i] = t.mu[i];
    }
}
