#include "llama-kvq-codec.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// step in units of sigma of the MSE-optimal uniform quantizer for a Gaussian, by bits (Max 1960)
static const float kvq_gauss_step[9] = { 0.0f, 1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f, 0.1041f, 0.0569f, 0.0308f };

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

// part[i*n + j], j >= i, = sum over the rows [r0, r1) of x[r][i] x[r][j] in float, in row order (the lower triangle is undefined)
static void kvq_gram_block(const float * x, int64_t r0, int64_t r1, int32_t n, int64_t stride, float * part) {
#if defined(__AVX2__)
    // 4 x 32 tiles in registers, the same sums in the same order (mul, then add)
    if (n % 32 == 0) {
        for (int32_t i0 = 0; i0 < n; i0 += 4) {
            for (int32_t j0 = (i0/32)*32; j0 < n; j0 += 32) {
                __m256 acc[4][4];
                for (int q = 0; q < 4; q++) {
                    for (int u = 0; u < 4; u++) {
                        acc[q][u] = _mm256_setzero_ps();
                    }
                }
                for (int64_t r = r0; r < r1; r++) {
                    const float * xr = x + r*stride;
                    const __m256 y0 = _mm256_loadu_ps(xr + j0);
                    const __m256 y1 = _mm256_loadu_ps(xr + j0 + 8);
                    const __m256 y2 = _mm256_loadu_ps(xr + j0 + 16);
                    const __m256 y3 = _mm256_loadu_ps(xr + j0 + 24);
                    for (int q = 0; q < 4; q++) {
                        const __m256 v = _mm256_broadcast_ss(xr + i0 + q);
                        acc[q][0] = _mm256_add_ps(acc[q][0], _mm256_mul_ps(v, y0));
                        acc[q][1] = _mm256_add_ps(acc[q][1], _mm256_mul_ps(v, y1));
                        acc[q][2] = _mm256_add_ps(acc[q][2], _mm256_mul_ps(v, y2));
                        acc[q][3] = _mm256_add_ps(acc[q][3], _mm256_mul_ps(v, y3));
                    }
                }
                for (int q = 0; q < 4; q++) {
                    float * pi = part + (size_t) (i0 + q)*n + j0;
                    for (int u = 0; u < 4; u++) {
                        _mm256_storeu_ps(pi + 8*u, acc[q][u]);
                    }
                }
            }
        }
        return;
    }
#endif
    std::fill(part, part + (size_t) n*n, 0.0f);
    for (int64_t r = r0; r < r1; r++) {
        const float * xr = x + r*stride;
        for (int32_t i = 0; i < n; i++) {
            const float v = xr[i];
            if (v == 0.0f) {
                continue;
            }
            float * pi = part + (size_t) i*n;
            for (int32_t j = i; j < n; j++) {
                pi[j] += v*xr[j];
            }
        }
    }
}

void llama_kvq_gram(const float * x, int64_t n_rows, int32_t n, int64_t stride, double * out, int n_threads) {
    const int64_t blk   = 256; // rows summed in float before the double accumulator
    const int64_t n_blk = (n_rows + blk - 1) / blk;
    const size_t  nn    = (size_t) n*n;

    std::vector<std::vector<double>> acc(std::max<int64_t>(1, std::min<int64_t>(n_threads, n_blk)));

    kvq_parallel_for(n_blk, n_threads, [&](int ith, int64_t b0, int64_t b1) {
        std::vector<double> & ad = acc[ith];
        ad.assign(nn, 0.0);
        std::vector<float> part(nn);
        for (int64_t b = b0; b < b1; b++) {
            kvq_gram_block(x, b*blk, std::min(n_rows, (b + 1)*blk), n, stride, part.data());
            for (int32_t i = 0; i < n; i++) {
                for (int32_t j = i; j < n; j++) {
                    ad[(size_t) i*n + j] += part[(size_t) i*n + j];
                }
            }
        }
    });

    for (const auto & ad : acc) {
        if (ad.empty()) {
            continue;
        }
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = i; j < n; j++) {
                const double v = ad[(size_t) i*n + j];
                out[(size_t) i*n + j] += v;
                if (j != i) {
                    out[(size_t) j*n + i] += v;
                }
            }
        }
    }
}

// Householder tridiagonalization + implicit QL (EISPACK tred2/tql2)
bool llama_kvq_eigh(int32_t n, std::vector<double> & a, std::vector<double> & w) {
    if (n <= 0 || a.size() != (size_t) n*n) {
        return false;
    }
    // tred2 runs on the transpose, so that its inner loops over k read contiguous memory; it reads the lower triangle of V,
    // which is the upper triangle of a: mirror the lower one there, then the arithmetic is that of the plain layout
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            a[(size_t) j*n + i] = a[(size_t) i*n + j];
        }
    }
    auto V = [&](int i, int j) -> double & { return a[(size_t) j*n + i]; };

    std::vector<double> & d = w;
    std::vector<double> e(n, 0.0);
    d.assign(n, 0.0);

    for (int j = 0; j < n; j++) {
        d[j] = V(n - 1, j);
    }

    // tred2
    for (int i = n - 1; i > 0; i--) {
        double scale = 0.0;
        double h     = 0.0;
        for (int k = 0; k < i; k++) {
            scale += std::fabs(d[k]);
        }
        if (scale == 0.0) {
            e[i] = d[i - 1];
            for (int j = 0; j < i; j++) {
                d[j]    = V(i - 1, j);
                V(i, j) = 0.0;
                V(j, i) = 0.0;
            }
        } else {
            for (int k = 0; k < i; k++) {
                d[k] /= scale;
                h += d[k]*d[k];
            }
            double f = d[i - 1];
            double g = std::sqrt(h);
            if (f > 0) {
                g = -g;
            }
            e[i]     = scale*g;
            h        = h - f*g;
            d[i - 1] = f - g;
            for (int j = 0; j < i; j++) {
                e[j] = 0.0;
            }
            for (int j = 0; j < i; j++) {
                f       = d[j];
                V(j, i) = f;
                g       = e[j] + V(j, j)*f;
                for (int k = j + 1; k <= i - 1; k++) {
                    g    += V(k, j)*d[k];
                    e[k] += V(k, j)*f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (int j = 0; j < i; j++) {
                e[j] /= h;
                f += e[j]*d[j];
            }
            const double hh = f/(h + h);
            for (int j = 0; j < i; j++) {
                e[j] -= hh*d[j];
            }
            for (int j = 0; j < i; j++) {
                f = d[j];
                g = e[j];
                for (int k = j; k <= i - 1; k++) {
                    V(k, j) -= (f*e[k] + g*d[k]);
                }
                d[j]    = V(i - 1, j);
                V(i, j) = 0.0;
            }
        }
        d[i] = h;
    }

    for (int i = 0; i < n - 1; i++) {
        V(n - 1, i) = V(i, i);
        V(i, i)     = 1.0;
        const double h = d[i + 1];
        if (h != 0.0) {
            for (int k = 0; k <= i; k++) {
                d[k] = V(k, i + 1)/h;
            }
            for (int j = 0; j <= i; j++) {
                double g = 0.0;
                for (int k = 0; k <= i; k++) {
                    g += V(k, i + 1)*V(k, j);
                }
                for (int k = 0; k <= i; k++) {
                    V(k, j) -= g*d[k];
                }
            }
        }
        for (int k = 0; k <= i; k++) {
            V(k, i + 1) = 0.0;
        }
    }
    for (int j = 0; j < n; j++) {
        d[j]        = V(n - 1, j);
        V(n - 1, j) = 0.0;
    }
    V(n - 1, n - 1) = 1.0;
    e[0] = 0.0;

    // tql2 works on the transpose too, so that the rotations read contiguous rows
    auto T = [&](int i, int j) -> double & { return a[(size_t) j*n + i]; };

    // tql2
    for (int i = 1; i < n; i++) {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    double f    = 0.0;
    double tst1 = 0.0;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int l = 0; l < n; l++) {
        tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
        int m = l;
        while (m < n - 1 && std::fabs(e[m]) > eps*tst1) {
            m++;
        }
        if (m > l) {
            int iter = 0;
            do {
                if (++iter > 100) {
                    return false;
                }
                double g = d[l];
                double p = (d[l + 1] - g)/(2.0*e[l]);
                double r = std::hypot(p, 1.0);
                if (p < 0) {
                    r = -r;
                }
                d[l]     = e[l]/(p + r);
                d[l + 1] = e[l]*(p + r);
                const double dl1 = d[l + 1];
                double h = g - d[l];
                for (int i = l + 2; i < n; i++) {
                    d[i] -= h;
                }
                f += h;

                p = d[m];
                double c  = 1.0;
                double c2 = c;
                double c3 = c;
                const double el1 = e[l + 1];
                double s  = 0.0;
                double s2 = 0.0;
                for (int i = m - 1; i >= l; i--) {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g  = c*e[i];
                    h  = c*p;
                    r  = std::hypot(p, e[i]);
                    e[i + 1] = s*r;
                    s  = e[i]/r;
                    c  = p/r;
                    p  = c*d[i] - s*g;
                    d[i + 1] = h + s*(c*g + s*d[i]);
                    double * vi  = &T(0, i);
                    double * vi1 = &T(0, i + 1);
                    for (int k = 0; k < n; k++) {
                        h = vi1[k];
                        vi1[k] = s*vi[k] + c*h;
                        vi[k]  = c*vi[k] - s*h;
                    }
                }
                p    = -s*s2*c3*el1*e[l]/dl1;
                e[l] = s*p;
                d[l] = c*p;
            } while (std::fabs(e[l]) > eps*tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0.0;
    }

    // ascending, stable for equal eigenvalues
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        double p = d[i];
        for (int j = i + 1; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int j = 0; j < n; j++) {
                std::swap(T(j, i), T(j, k));
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            std::swap(a[(size_t) i*n + j], a[(size_t) j*n + i]);
        }
    }

    return true;
}

std::vector<uint8_t> llama_kvq_alloc_bits(const std::vector<double> & var, float avg_bits, int max_bits) {
    const int n = (int) var.size();
    std::vector<uint8_t> res(n, 0);
    if (n == 0) {
        return res;
    }

    const int64_t target = std::clamp<int64_t>((int64_t) std::nearbyint((double) avg_bits*n), 0, (int64_t) n*max_bits);

    std::vector<double> ideal(n);
    for (int c = 0; c < n; c++) {
        ideal[c] = 0.5*std::log2(std::max(var[c], 1e-30));
    }

    // water level th such that sum(clip(ideal - th, 0, max_bits)) == target
    double lo = *std::min_element(ideal.begin(), ideal.end()) - 20;
    double hi = *std::max_element(ideal.begin(), ideal.end()) + 20;
    double th = 0.0;
    for (int it = 0; it < 60; it++) {
        th = (lo + hi)/2;
        double sum = 0.0;
        for (int c = 0; c < n; c++) {
            sum += std::clamp(ideal[c] - th, 0.0, (double) max_bits);
        }
        if (sum > target) {
            lo = th;
        } else {
            hi = th;
        }
    }

    std::vector<int> b(n);
    int64_t sum = 0;
    for (int c = 0; c < n; c++) {
        ideal[c] -= th;
        b[c] = (int) std::clamp(std::nearbyint(ideal[c]), 0.0, (double) max_bits); // round half to even
        sum += b[c];
    }

    // one bit per pass; ties go to the lowest index
    while (sum != target) {
        int best = -1;
        for (int c = 0; c < n; c++) {
            const double d = b[c] - ideal[c];
            if (sum > target) {
                if (b[c] > 0 && (best < 0 || d > b[best] - ideal[best])) {
                    best = c;
                }
            } else {
                if (b[c] < max_bits && (best < 0 || d < b[best] - ideal[best])) {
                    best = c;
                }
            }
        }
        if (best < 0) {
            break;
        }
        if (sum > target) {
            b[best]--;
            sum--;
        } else {
            b[best]++;
            sum++;
        }
    }

    for (int c = 0; c < n; c++) {
        res[c] = (uint8_t) b[c];
    }
    return res;
}

// a = L L^T, L is written to the lower triangle, the upper one is zeroed
static bool kvq_cholesky(int n, std::vector<double> & a) {
    for (int j = 0; j < n; j++) {
        double s = a[(size_t) j*n + j];
        for (int k = 0; k < j; k++) {
            s -= a[(size_t) j*n + k]*a[(size_t) j*n + k];
        }
        if (!(s > 0.0)) {
            return false;
        }
        const double ljj = std::sqrt(s);
        a[(size_t) j*n + j] = ljj;
        for (int i = j + 1; i < n; i++) {
            double v = a[(size_t) i*n + j];
            for (int k = 0; k < j; k++) {
                v -= a[(size_t) i*n + k]*a[(size_t) j*n + k];
            }
            a[(size_t) i*n + j] = v/ljj;
        }
        for (int i = 0; i < j; i++) {
            a[(size_t) i*n + j] = 0.0;
        }
    }
    return true;
}

static std::vector<double> kvq_lower_inverse(int n, const std::vector<double> & l) {
    std::vector<double> r((size_t) n*n, 0.0);
    for (int j = 0; j < n; j++) {
        r[(size_t) j*n + j] = 1.0/l[(size_t) j*n + j];
        for (int i = j + 1; i < n; i++) {
            double s = 0.0;
            for (int k = j; k < i; k++) {
                s += l[(size_t) i*n + k]*r[(size_t) k*n + j];
            }
            r[(size_t) i*n + j] = -s/l[(size_t) i*n + i];
        }
    }
    return r;
}

bool llama_kvq_fit_cov(llama_kvq_table & t, const float * mu, const double * cov, int64_t n_tok, int32_t n, const double * metric, bool is_v) {
    if (n_tok < 1 || n < 1) {
        return false;
    }
    const size_t nn = (size_t) n*n;

    double tr = 0.0;
    for (int i = 0; i < n; i++) {
        tr += metric[(size_t) i*n + i];
    }
    // a metric from few queries is near singular, the rounding of its sums can make it indefinite: a larger ridge then
    std::vector<double> l;
    bool chol = false;
    for (const double ridge : { 1e-6, 1e-4, 1e-2 }) {
        l.assign(metric, metric + nn);
        for (int i = 0; i < n; i++) {
            l[(size_t) i*n + i] += ridge*tr/n;
        }
        chol = kvq_cholesky(n, l);
        if (chol) {
            if (ridge > 1e-6) {
                LLAMA_LOG_WARN("%s: the metric needed a ridge of %.0e of its mean diagonal\n", __func__, ridge);
            }
            break;
        }
    }
    if (!chol) {
        LLAMA_LOG_ERROR("%s: metric is not positive definite\n", __func__);
        return false;
    }

    t.n    = n;
    t.is_v = is_v;
    t.mu.assign(mu, mu + n);

    // S = L^T cov L / n_tok
    std::vector<double> tmp(nn, 0.0);
    std::vector<double> s(nn, 0.0);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const double v = cov[(size_t) i*n + k]/n_tok;
            for (int j = 0; j <= k; j++) {
                tmp[(size_t) i*n + j] += v*l[(size_t) k*n + j];
            }
        }
    }
    for (int k = 0; k < n; k++) {
        for (int i = 0; i <= k; i++) {
            const double v = l[(size_t) k*n + i];
            for (int j = 0; j < n; j++) {
                s[(size_t) i*n + j] += v*tmp[(size_t) k*n + j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            const double v = 0.5*(s[(size_t) i*n + j] + s[(size_t) j*n + i]);
            s[(size_t) i*n + j] = v;
            s[(size_t) j*n + i] = v;
        }
    }

    std::vector<double> lam;
    if (!llama_kvq_eigh(n, s, lam)) {
        LLAMA_LOG_ERROR("%s: eigen solver did not converge\n", __func__);
        return false;
    }
    const std::vector<double> & u = s;

    // fwd = L U, inv = U^T L^-1
    const std::vector<double> linv = kvq_lower_inverse(n, l);
    t.fwd.assign(nn, 0.0f);
    t.inv.assign(nn, 0.0f);
    // the sums run over k in increasing order as in a plain dot product, the loops over the output are inside
    {
        std::vector<double> acc(n);
        for (int i = 0; i < n; i++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = 0; k <= i; k++) {
                const double v = l[(size_t) i*n + k];
                const double * uk = u.data() + (size_t) k*n;
                double * pa = acc.data();
                for (int c = 0; c < n; c++) {
                    pa[c] += v*uk[c];
                }
            }
            for (int c = 0; c < n; c++) {
                t.fwd[(size_t) i*n + c] = (float) acc[c];
            }
        }
        for (int c = 0; c < n; c++) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int k = 0; k < n; k++) {
                const double v = u[(size_t) k*n + c];
                const double * lk = linv.data() + (size_t) k*n;
                double * pa = acc.data();
                for (int j = 0; j <= k; j++) {
                    pa[j] += v*lk[j];
                }
            }
            for (int j = 0; j < n; j++) {
                t.inv[(size_t) c*n + j] = (float) acc[j];
            }
        }
    }

    t.sigma.resize(n);
    for (int c = 0; c < n; c++) {
        t.sigma[c] = (float) std::sqrt(std::max(lam[c], 1e-30));
    }

    t.gain_den = 0.0f;
    if (is_v) {
        double den = 0.0;
        for (int c = 0; c < n; c++) {
            den += (double) t.sigma[c]*t.sigma[c];
        }
        t.gain_den = (float) den;
    }

    return true;
}

void llama_kvq_fit_bits(llama_kvq_table & t, float avg_bits) {
    const int n = t.n;
    std::vector<double> var(n);
    for (int c = 0; c < n; c++) {
        var[c] = (double) t.sigma[c]*t.sigma[c];
    }
    t.bits = llama_kvq_alloc_bits(var, avg_bits);
    t.step.resize(n);
    for (int c = 0; c < n; c++) {
        t.step[c] = kvq_gauss_step[t.bits[c]]*t.sigma[c];
    }
}
