#include "llama-kvq-codec.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

// step in units of sigma of the MSE-optimal uniform quantizer for a Gaussian, by bits (Max 1960)
static const float kvq_gauss_step[9] = { 0.0f, 1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f, 0.1041f, 0.0569f, 0.0308f };

static const char kvq_table_magic[8] = { 'k', 'v', 'q', 't', 'a', 'b', '0', '1' };

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

// c[n_rows*m] = a[n_rows*k] b[k*m]
static void kvq_matmul(const float * a, int64_t n_rows, int32_t k, const float * b, int32_t m, float * c, int n_threads) {
    kvq_parallel_for(n_rows, n_threads, [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
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
    });
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
            std::fill(part.begin(), part.end(), 0.0f);
            const int64_t r1 = std::min(n_rows, (b + 1)*blk);
            for (int64_t r = b*blk; r < r1; r++) {
                const float * xr = x + r*stride;
                for (int32_t i = 0; i < n; i++) {
                    const float v = xr[i];
                    if (v == 0.0f) {
                        continue;
                    }
                    float * pi = part.data() + (size_t) i*n;
                    for (int32_t j = i; j < n; j++) {
                        pi[j] += v*xr[j];
                    }
                }
            }
            for (size_t i = 0; i < nn; i++) {
                ad[i] += part[i];
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
    auto V = [&](int i, int j) -> double & { return a[(size_t) i*n + j]; };

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
                    for (int k = 0; k < n; k++) {
                        h = V(k, i + 1);
                        V(k, i + 1) = s*V(k, i) + c*h;
                        V(k, i)     = c*V(k, i) - s*h;
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
                std::swap(V(j, i), V(j, k));
            }
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

bool llama_kvq_fit(llama_kvq_table & t, const float * x, int64_t n_tok, int32_t n, const double * metric, float avg_bits, bool is_v, int n_threads) {
    if (n_tok < 1 || n < 1) {
        return false;
    }
    const size_t nn = (size_t) n*n;

    std::vector<double> l(metric, metric + nn);
    double tr = 0.0;
    for (int i = 0; i < n; i++) {
        tr += l[(size_t) i*n + i];
    }
    for (int i = 0; i < n; i++) {
        l[(size_t) i*n + i] += 1e-6*tr/n;
    }
    if (!kvq_cholesky(n, l)) {
        LLAMA_LOG_ERROR("%s: metric is not positive definite\n", __func__);
        return false;
    }

    t.n    = n;
    t.is_v = is_v;
    t.mu.assign(n, 0.0f);
    {
        std::vector<double> mu(n, 0.0);
        for (int64_t r = 0; r < n_tok; r++) {
            for (int i = 0; i < n; i++) {
                mu[i] += x[r*n + i];
            }
        }
        for (int i = 0; i < n; i++) {
            t.mu[i] = (float) (mu[i]/n_tok);
        }
    }

    std::vector<float> xc((size_t) n_tok*n);
    kvq_parallel_for(n_tok, n_threads, [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            for (int i = 0; i < n; i++) {
                xc[r*n + i] = x[r*n + i] - t.mu[i];
            }
        }
    });

    std::vector<double> cov(nn, 0.0);
    llama_kvq_gram(xc.data(), n_tok, n, n, cov.data(), n_threads);

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
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < n; c++) {
            double vf = 0.0;
            for (int k = 0; k <= i; k++) {
                vf += l[(size_t) i*n + k]*u[(size_t) k*n + c];
            }
            t.fwd[(size_t) i*n + c] = (float) vf;
        }
    }
    for (int c = 0; c < n; c++) {
        for (int j = 0; j < n; j++) {
            double vi = 0.0;
            for (int k = j; k < n; k++) {
                vi += u[(size_t) k*n + c]*linv[(size_t) k*n + j];
            }
            t.inv[(size_t) c*n + j] = (float) vi;
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

        std::vector<float> y((size_t) n_tok*n);
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

    std::vector<double> var(n);
    for (int c = 0; c < n; c++) {
        var[c] = (double) t.sigma[c]*t.sigma[c];
    }
    t.bits = llama_kvq_alloc_bits(var, avg_bits);
    t.step.resize(n);
    for (int c = 0; c < n; c++) {
        t.step[c] = kvq_gauss_step[t.bits[c]]*t.sigma[c];
    }

    return true;
}

void llama_kvq_encode(const llama_kvq_table & t, const float * x, int64_t n_tok, int8_t * codes, float * gain, int n_threads) {
    const int n = t.n;
    GGML_ASSERT(!t.is_v || gain != nullptr);

    kvq_parallel_for(n_tok, n_threads, [&](int, int64_t r0, int64_t r1) {
        std::vector<float> xc(n);
        std::vector<float> y(n);
        for (int64_t r = r0; r < r1; r++) {
            for (int i = 0; i < n; i++) {
                xc[i] = x[r*n + i] - t.mu[i];
            }
            kvq_matmul(xc.data(), 1, n, t.fwd.data(), n, y.data(), 1);
            if (t.is_v) {
                double ss = 0.0;
                for (int c = 0; c < n; c++) {
                    ss += (double) y[c]*y[c];
                }
                const float g = (float) std::sqrt(ss/t.gain_den) + 1e-8f;
                for (int c = 0; c < n; c++) {
                    y[c] /= g;
                }
                gain[r] = g;
            } else if (gain) {
                gain[r] = 1.0f;
            }
            int8_t * cr = codes + r*n;
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
    });
}

void llama_kvq_decode(const llama_kvq_table & t, const int8_t * codes, const float * gain, int64_t n_tok, float * x, int n_threads) {
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

bool llama_kvq_save(const std::string & path, const std::vector<llama_kvq_table> & tables) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: failed to open %s\n", __func__, path.c_str());
        return false;
    }
    bool ok = fwrite(kvq_table_magic, sizeof(kvq_table_magic), 1, f) == 1;
    const int32_t n_tables = (int32_t) tables.size();
    ok = ok && fwrite(&n_tables, sizeof(n_tables), 1, f) == 1;
    for (const auto & t : tables) {
        if (!ok) {
            break;
        }
        const size_t n  = t.n;
        const int32_t hdr[4] = { t.il, t.head, t.is_v ? 1 : 0, t.n };
        ok = ok && fwrite(hdr, sizeof(hdr), 1, f) == 1;
        ok = ok && fwrite(&t.gain_den, sizeof(float), 1, f) == 1;
        ok = ok && fwrite(t.mu.data(),    sizeof(float), n,   f) == n;
        ok = ok && fwrite(t.fwd.data(),   sizeof(float), n*n, f) == n*n;
        ok = ok && fwrite(t.inv.data(),   sizeof(float), n*n, f) == n*n;
        ok = ok && fwrite(t.sigma.data(), sizeof(float), n,   f) == n;
        ok = ok && fwrite(t.step.data(),  sizeof(float), n,   f) == n;
        ok = ok && fwrite(t.bits.data(),  1,             n,   f) == n;
    }
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
        LLAMA_LOG_ERROR("%s: failed to write %s\n", __func__, path.c_str());
    }
    return ok;
}

bool llama_kvq_load(const std::string & path, std::vector<llama_kvq_table> & tables) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LLAMA_LOG_ERROR("%s: failed to open %s\n", __func__, path.c_str());
        return false;
    }
    char magic[8];
    int32_t n_tables = 0;
    bool ok = fread(magic, sizeof(magic), 1, f) == 1 && memcmp(magic, kvq_table_magic, sizeof(magic)) == 0;
    ok = ok && fread(&n_tables, sizeof(n_tables), 1, f) == 1 && n_tables >= 0;
    tables.clear();
    for (int32_t it = 0; ok && it < n_tables; it++) {
        llama_kvq_table t;
        int32_t hdr[4];
        ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[3] > 0 && hdr[3] <= 4096;
        if (!ok) {
            break;
        }
        t.il   = hdr[0];
        t.head = hdr[1];
        t.is_v = hdr[2] != 0;
        t.n    = hdr[3];
        const size_t n = t.n;
        t.mu.resize(n);
        t.fwd.resize(n*n);
        t.inv.resize(n*n);
        t.sigma.resize(n);
        t.step.resize(n);
        t.bits.resize(n);
        ok = ok && fread(&t.gain_den, sizeof(float), 1, f) == 1;
        ok = ok && fread(t.mu.data(),    sizeof(float), n,   f) == n;
        ok = ok && fread(t.fwd.data(),   sizeof(float), n*n, f) == n*n;
        ok = ok && fread(t.inv.data(),   sizeof(float), n*n, f) == n*n;
        ok = ok && fread(t.sigma.data(), sizeof(float), n,   f) == n;
        ok = ok && fread(t.step.data(),  sizeof(float), n,   f) == n;
        ok = ok && fread(t.bits.data(),  1,             n,   f) == n;
        for (size_t c = 0; ok && c < n; c++) {
            ok = t.bits[c] <= 8;
        }
        if (ok) {
            tables.push_back(std::move(t));
        }
    }
    fclose(f);
    if (!ok) {
        LLAMA_LOG_ERROR("%s: %s is not a valid table file\n", __func__, path.c_str());
        tables.clear();
    }
    return ok;
}
