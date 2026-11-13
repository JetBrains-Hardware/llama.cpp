#pragma once

// transform codec for cached K/V rows (CPU, used by tools/kvq-eval)
//
// one table per (layer, kv head), separately for K and V
// error cost of a row is d^T M d; y = (x - mu) fwd is decorrelated and white in M
// each component c gets bits[c] (reverse water-filling) and a uniform mid-rise quantizer with step[c]
// V rows are split into a per-token gain and a shape before quantization

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_kvq_table {
    int32_t il   = -1;
    int32_t head = -1;
    bool    is_v = false; // per-token gain
    int32_t n    = 0;

    float gain_den = 0.0f; // sum of component variances before the gain split

    std::vector<float>   mu;    // [n]
    std::vector<float>   fwd;   // [n*n], y[c] = sum_i (x[i] - mu[i]) * fwd[i*n + c]
    std::vector<float>   inv;   // [n*n], x[i] = sum_c y[c] * inv[c*n + i] + mu[i]
    std::vector<float>   sigma; // [n], scale of y (after the gain split for V)
    std::vector<float>   step;  // [n], 0 if bits == 0
    std::vector<uint8_t> bits;  // [n]
};

// out[n*n] += sum over rows of x x^T, row r starts at x + r*stride
LLAMA_API void llama_kvq_gram(const float * x, int64_t n_rows, int32_t n, int64_t stride, double * out, int n_threads);

// symmetric eigen decomposition: a[n*n] row-major in, eigenvectors as columns out; w = eigenvalues, ascending
LLAMA_API bool llama_kvq_eigh(int32_t n, std::vector<double> & a, std::vector<double> & w);

// integer bits per component, sum == round(avg_bits * n), each in [0, max_bits]
LLAMA_API std::vector<uint8_t> llama_kvq_alloc_bits(const std::vector<double> & var, float avg_bits, int max_bits = 8);

// fit a table on x[n_tok*n] with error metric[n*n]
LLAMA_API bool llama_kvq_fit(llama_kvq_table & t, const float * x, int64_t n_tok, int32_t n, const double * metric, float avg_bits, bool is_v, int n_threads);

// llama_kvq_fit, then llama_kvq_encode of the same rows (V: the fit's transformed rows are used again)
LLAMA_API bool llama_kvq_fit_encode(llama_kvq_table & t, const float * x, int64_t n_tok, int32_t n, const double * metric, float avg_bits, bool is_v,
        int8_t * codes, float * gain, int n_threads);

// codes[n_tok*n] in [-2^(b-1), 2^(b-1) - 1], 0 for b == 0; gain[n_tok] is used for V only (may be null for K)
LLAMA_API void llama_kvq_encode(const llama_kvq_table & t, const float * x, int64_t n_tok, int8_t * codes, float * gain, int n_threads);
LLAMA_API void llama_kvq_decode(const llama_kvq_table & t, const int8_t * codes, const float * gain, int64_t n_tok, float * x, int n_threads);

LLAMA_API bool llama_kvq_save(const std::string & path, const std::vector<llama_kvq_table> & tables);
LLAMA_API bool llama_kvq_load(const std::string & path, std::vector<llama_kvq_table> & tables);
