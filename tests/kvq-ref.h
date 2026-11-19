#pragma once

// CPU references of the kvq codec for the tests (test-kvq-codec, test-kvq-attn); the library fits and codes on the device

#include "../src/llama-kvq-codec.h"
#include "../src/llama-kvq-pack.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// codes [n_tok][t.n] in codec order -> rows [n_tok][n_words] (K) or [n_tok][n_words + 1] (V, gain [n_tok]) in the layout of ggml_kvq_attn
// n_words >= llama_kvq_pack_words(t), the unused high bits are 0
static void kvq_ref_pack_rows(const llama_kvq_table & t, int32_t n_words, const int8_t * codes, const float * gain, int64_t n_tok, int32_t * rows) {
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

// the fit and the codes of rows x[n_tok*n] on the CPU (the library does both on the device, src/llama-kvq-fit.cpp)
// kvq_ref_fit: a table on the rows with the error metric[n*n]; kvq_ref_encode: codes[n_tok*n] in [-2^(b-1), 2^(b-1) - 1],
// 0 for b == 0, gain[n_tok] for V only (may be null for K); kvq_ref_fit_encode: both (V: the fit's transformed rows are used again)

// f(ith, i0, i1) on contiguous ranges of [0, n)
template <typename F>
static void kvq_parallel_for(int64_t n, int n_threads, const F & f) {
    const int64_t nth = std::max<int64_t>(1, std::min<int64_t>(n_threads, n));
    if (nth == 1) {
        f(0, 0, n);
        return;
    }
    const int64_t chunk = (n + nth - 1) / nth;
    std::vector<std::thread> workers;
    for (int64_t ith = 0; ith < nth; ith++) {
        const int64_t i0 = ith * chunk;
        const int64_t i1 = std::min(n, i0 + chunk);
        if (i0 >= i1) {
            break;
        }
        workers.emplace_back([&f, ith, i0, i1]() { f((int) ith, i0, i1); });
    }
    for (auto & w : workers) {
        w.join();
    }
}

// rows [r0, r1) of c[n_rows*m] = a[n_rows*k] b[k*m]
// plain arguments, not lambda captures: the compiler cannot vectorize through captured references
static void kvq_matmul_rows(const float * a, int64_t r0, int64_t r1, int32_t k, const float * b, int32_t m, float * c) {
    int64_t r = r0;
#if defined(__AVX2__)
    // 4 rows x 32 columns in registers, the same sums in the same order (mul, then add)
    if (m % 32 == 0) {
        for (; r + 4 <= r1; r += 4) {
            for (int32_t j0 = 0; j0 < m; j0 += 32) {
                __m256 acc[4][4];
                for (int q = 0; q < 4; q++) {
                    for (int u = 0; u < 4; u++) {
                        acc[q][u] = _mm256_setzero_ps();
                    }
                }
                for (int32_t i = 0; i < k; i++) {
                    const float * bi = b + (int64_t) i*m + j0;
                    const __m256 b0 = _mm256_loadu_ps(bi);
                    const __m256 b1 = _mm256_loadu_ps(bi + 8);
                    const __m256 b2 = _mm256_loadu_ps(bi + 16);
                    const __m256 b3 = _mm256_loadu_ps(bi + 24);
                    for (int q = 0; q < 4; q++) {
                        const __m256 v = _mm256_broadcast_ss(a + (r + q)*k + i);
                        acc[q][0] = _mm256_add_ps(acc[q][0], _mm256_mul_ps(v, b0));
                        acc[q][1] = _mm256_add_ps(acc[q][1], _mm256_mul_ps(v, b1));
                        acc[q][2] = _mm256_add_ps(acc[q][2], _mm256_mul_ps(v, b2));
                        acc[q][3] = _mm256_add_ps(acc[q][3], _mm256_mul_ps(v, b3));
                    }
                }
                for (int q = 0; q < 4; q++) {
                    float * cr = c + (r + q)*m + j0;
                    for (int u = 0; u < 4; u++) {
                        _mm256_storeu_ps(cr + 8*u, acc[q][u]);
                    }
                }
            }
        }
    }
#endif
    for (; r < r1; r++) {
        const float * ar = a + r*k;
        float       * cr = c + r*m;
        std::fill(cr, cr + m, 0.0f);
        for (int32_t i = 0; i < k; i++) {
            const float v = ar[i];
            if (v == 0.0f) {
                continue;
            }
            const float * bi = b + (int64_t) i*m;
            for (int32_t j = 0; j < m; j++) {
                cr[j] += v*bi[j];
            }
        }
    }
}

// c[n_rows*m] = a[n_rows*k] b[k*m]
static void kvq_matmul(const float * a, int64_t n_rows, int32_t k, const float * b, int32_t m, float * c, int n_threads) {
    kvq_parallel_for(n_rows, n_threads, [&](int, int64_t r0, int64_t r1) {
        kvq_matmul_rows(a, r0, r1, k, b, m, c);
    });
}

// y_out: V only, the transformed rows (x - mu) fwd
static bool kvq_fit_impl(llama_kvq_table & t, const float * x, int64_t n_tok, int32_t n, const double * metric, float avg_bits, bool is_v,
        std::vector<float> * y_out, int n_threads) {
    if (n_tok < 1 || n < 1) {
        return false;
    }
    const size_t nn = (size_t) n*n;

    std::vector<float> mu(n);
    {
        std::vector<double> s(n, 0.0);
        for (int64_t r = 0; r < n_tok; r++) {
            for (int i = 0; i < n; i++) {
                s[i] += x[r*n + i];
            }
        }
        for (int i = 0; i < n; i++) {
            mu[i] = (float) (s[i]/n_tok);
        }
    }

    std::vector<float> xc((size_t) n_tok*n);
    kvq_parallel_for(n_tok, n_threads, [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            for (int i = 0; i < n; i++) {
                xc[r*n + i] = x[r*n + i] - mu[i];
            }
        }
    });

    std::vector<double> cov(nn, 0.0);
    llama_kvq_gram(xc.data(), n_tok, n, n, cov.data(), n_threads);

    if (!llama_kvq_fit_cov(t, mu.data(), cov.data(), n_tok, n, metric, is_v)) {
        return false;
    }

    if (is_v) {
        std::vector<float> y_tmp;
        std::vector<float> & y = y_out ? *y_out : y_tmp;
        y.resize((size_t) n_tok*n);
        kvq_matmul(xc.data(), n_tok, n, t.fwd.data(), n, y.data(), n_threads);

        // scale of the shape y / g
        std::vector<std::vector<double>> acc(std::max<int64_t>(1, std::min<int64_t>(n_threads, n_tok)));
        kvq_parallel_for(n_tok, n_threads, [&](int ith, int64_t r0, int64_t r1) {
            std::vector<double> & a = acc[ith];
            a.assign(n, 0.0);
            for (int64_t r = r0; r < r1; r++) {
                const float * yr = y.data() + r*n;
                double ss = 0.0;
                for (int c = 0; c < n; c++) {
                    ss += (double) yr[c]*yr[c];
                }
                const float g = (float) std::sqrt(ss/t.gain_den) + 1e-8f;
                for (int c = 0; c < n; c++) {
                    const float v = yr[c]/g;
                    a[c] += (double) v*v;
                }
            }
        });
        std::vector<double> sq(n, 0.0);
        for (const auto & a : acc) {
            for (int c = 0; c < (int) a.size(); c++) {
                sq[c] += a[c];
            }
        }
        for (int c = 0; c < n; c++) {
            t.sigma[c] = (float) std::sqrt(sq[c]/n_tok);
        }
    }

    llama_kvq_fit_bits(t, avg_bits);

    return true;
}

static bool kvq_ref_fit(llama_kvq_table & t, const float * x, int64_t n_tok, int32_t n, const double * metric, float avg_bits, bool is_v, int n_threads) {
    return kvq_fit_impl(t, x, n_tok, n, metric, avg_bits, is_v, nullptr, n_threads);
}

// codes of one transformed row y (V: y is divided by the gain)
static void kvq_code_row(const llama_kvq_table & t, float * y, int8_t * cr, float * gain) {
    const int n = t.n;
    if (t.is_v) {
        double ss = 0.0;
        for (int c = 0; c < n; c++) {
            ss += (double) y[c]*y[c];
        }
        const float g = (float) std::sqrt(ss/t.gain_den) + 1e-8f;
        for (int c = 0; c < n; c++) {
            y[c] /= g;
        }
        *gain = g;
    } else if (gain) {
        *gain = 1.0f;
    }
    for (int c = 0; c < n; c++) {
        const int b = t.bits[c];
        if (b == 0) {
            cr[c] = 0;
            continue;
        }
        const float half = (float) (1 << (b - 1));
        const float q    = std::floor(y[c]/t.step[c]);
        cr[c] = (int8_t) std::clamp(q, -half, half - 1.0f);
    }
}

static void kvq_ref_encode(const llama_kvq_table & t, const float * x, int64_t n_tok, int8_t * codes, float * gain, int n_threads) {
    const int n = t.n;
    GGML_ASSERT(!t.is_v || gain != nullptr);

    kvq_parallel_for(n_tok, n_threads, [&](int, int64_t r0, int64_t r1) {
        // blocks of rows for the matrix product
        const int64_t blk = 64;
        std::vector<float> xc(blk*n);
        std::vector<float> y(blk*n);
        for (int64_t b0 = r0; b0 < r1; b0 += blk) {
            const int64_t nb = std::min(blk, r1 - b0);
            for (int64_t r = 0; r < nb; r++) {
                for (int i = 0; i < n; i++) {
                    xc[r*n + i] = x[(b0 + r)*n + i] - t.mu[i];
                }
            }
            kvq_matmul_rows(xc.data(), 0, nb, n, t.fwd.data(), n, y.data());
            for (int64_t r = 0; r < nb; r++) {
                kvq_code_row(t, y.data() + r*n, codes + (b0 + r)*n, gain ? gain + b0 + r : nullptr);
            }
        }
    });
}

static bool kvq_ref_fit_encode(llama_kvq_table & t, const float * x, int64_t n_tok, int32_t n, const double * metric, float avg_bits, bool is_v,
        int8_t * codes, float * gain, int n_threads) {
    GGML_ASSERT(!is_v || gain != nullptr);
    if (!is_v) {
        if (!kvq_fit_impl(t, x, n_tok, n, metric, avg_bits, false, nullptr, n_threads)) {
            return false;
        }
        kvq_ref_encode(t, x, n_tok, codes, gain, n_threads);
        return true;
    }
    std::vector<float> y;
    if (!kvq_fit_impl(t, x, n_tok, n, metric, avg_bits, true, &y, n_threads)) {
        return false;
    }
    kvq_parallel_for(n_tok, n_threads, [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            kvq_code_row(t, y.data() + r*n, codes + r*n, gain + r);
        }
    });
    return true;
}

static void kvq_ref_decode(const llama_kvq_table & t, const int8_t * codes, const float * gain, int64_t n_tok, float * x, int n_threads) {
    const int n = t.n;

    kvq_parallel_for(n_tok, n_threads, [&](int, int64_t r0, int64_t r1) {
        std::vector<float> yq(n);
        for (int64_t r = r0; r < r1; r++) {
            const int8_t * cr = codes + r*n;
            const float g = t.is_v ? gain[r] : 1.0f;
            for (int c = 0; c < n; c++) {
                yq[c] = t.bits[c] ? (cr[c] + 0.5f)*t.step[c]*g : 0.0f;
            }
            float * xr = x + r*n;
            kvq_matmul(yq.data(), 1, n, t.inv.data(), n, xr, 1);
            for (int i = 0; i < n; i++) {
                xr[i] += t.mu[i];
            }
        }
    });
}
