// tests for the K/V transform codec in src/llama-kvq-codec.h

#include "../src/llama-kvq-codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static std::vector<double> random_sym(int n, std::mt19937 & rng) {
    std::normal_distribution<double> nd;
    std::vector<double> a((size_t) n*n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            a[(size_t) i*n + j] = a[(size_t) j*n + i] = nd(rng);
        }
    }
    return a;
}

// max |A V - V diag(w)| / max |A| and max |V^T V - I|
static void check_eigh(const char * name, int n, const std::vector<double> & a0) {
    std::vector<double> v = a0;
    std::vector<double> w;
    const bool ok = llama_kvq_eigh(n, v, w);
    CHECK(ok, "%s: eigh failed", name);
    if (!ok) {
        return;
    }
    double amax = 1e-300;
    for (double x : a0) {
        amax = std::max(amax, std::fabs(x));
    }
    double res = 0.0;
    double orth = 0.0;
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < n; c++) {
            double av = 0.0;
            double vv = 0.0;
            for (int k = 0; k < n; k++) {
                av += a0[(size_t) i*n + k]*v[(size_t) k*n + c];
                vv += v[(size_t) k*n + i]*v[(size_t) k*n + c];
            }
            res  = std::max(res,  std::fabs(av - v[(size_t) i*n + c]*w[c]));
            orth = std::max(orth, std::fabs(vv - (i == c ? 1.0 : 0.0)));
        }
    }
    bool sorted = true;
    for (int c = 1; c < n; c++) {
        sorted = sorted && w[c - 1] <= w[c];
    }
    printf("eigh %-12s n = %3d: residual %.2e, orthogonality %.2e\n", name, n, res/amax, orth);
    CHECK(res/amax < 1e-11*n, "%s: residual %e", name, res/amax);
    CHECK(orth < 1e-11*n, "%s: orthogonality %e", name, orth);
    CHECK(sorted, "%s: eigenvalues not ascending", name);
}

static void test_eigh() {
    std::mt19937 rng(1);

    {
        std::vector<double> a = { 2, 1, 1, 2 };
        std::vector<double> w;
        llama_kvq_eigh(2, a, w);
        CHECK(std::fabs(w[0] - 1) < 1e-14 && std::fabs(w[1] - 3) < 1e-14, "2x2: w = %f %f", w[0], w[1]);
        check_eigh("2x2", 2, { 2, 1, 1, 2 });
    }
    {
        std::vector<double> a = { 5 };
        std::vector<double> w;
        CHECK(llama_kvq_eigh(1, a, w) && w[0] == 5 && a[0] == 1, "1x1");
    }

    // diagonal with known eigenvalues, out of order
    {
        const int n = 16;
        std::vector<double> a((size_t) n*n, 0.0);
        for (int i = 0; i < n; i++) {
            a[(size_t) i*n + i] = (i*7) % n - 3.0;
        }
        std::vector<double> v = a;
        std::vector<double> w;
        llama_kvq_eigh(n, v, w);
        for (int i = 0; i < n; i++) {
            CHECK(std::fabs(w[i] - (i - 3.0)) < 1e-13, "diag: w[%d] = %f", i, w[i]);
        }
        check_eigh("diag", n, a);
    }

    // identity: all eigenvalues equal
    {
        const int n = 32;
        std::vector<double> a((size_t) n*n, 0.0);
        for (int i = 0; i < n; i++) {
            a[(size_t) i*n + i] = 1.0;
        }
        check_eigh("identity", n, a);
    }

    // Q diag(l) Q^T with a known spectrum incl. repeated and zero eigenvalues
    {
        const int n = 64;
        std::vector<double> q = random_sym(n, rng);
        std::vector<double> w0;
        llama_kvq_eigh(n, q, w0); // q = random orthogonal matrix
        std::vector<double> l(n);
        for (int i = 0; i < n; i++) {
            l[i] = i < 10 ? 0.0 : (i < 20 ? 2.0 : std::pow(10.0, (i - 40)/6.0));
        }
        std::vector<double> a((size_t) n*n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += q[(size_t) i*n + k]*l[k]*q[(size_t) j*n + k];
                }
                a[(size_t) i*n + j] = s;
            }
        }
        std::vector<double> v = a;
        std::vector<double> w;
        llama_kvq_eigh(n, v, w);
        std::vector<double> ls = l;
        std::sort(ls.begin(), ls.end());
        double err = 0.0;
        for (int i = 0; i < n; i++) {
            err = std::max(err, std::fabs(w[i] - ls[i]));
        }
        CHECK(err < 1e-10, "known spectrum: max error %e", err);
        check_eigh("spectrum", n, a);
    }

    check_eigh("random", 256, random_sym(256, rng));

    // ill-conditioned Gram matrix of correlated data
    {
        const int n = 256;
        std::normal_distribution<double> nd;
        std::vector<double> a((size_t) n*n, 0.0);
        std::vector<double> x(n);
        for (int t = 0; t < 300; t++) {
            for (int i = 0; i < n; i++) {
                x[i] = nd(rng)*std::pow(0.9, i);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    a[(size_t) i*n + j] += x[i]*x[j];
                }
            }
        }
        check_eigh("gram", n, a);
    }
}

static void test_alloc_bits() {
    struct tcase {
        std::vector<double>  var;
        float                avg;
        std::vector<uint8_t> want;
    };
    // expected values from the reference implementation (allocate_bits in kvcodec.py)
    const std::vector<tcase> cases = {
        { { 1, 1, 1, 1 },                    2.0f,      { 2, 2, 2, 2 } },
        { { 1, 1 },                          1.5f,      { 1, 2 } },             // tie after rounding
        { { 1, 1, 1 },                       1.0f/3,    { 1, 0, 0 } },          // tie, lowest index wins
        { { 1e12, 1, 1, 1 },                 3.0f,      { 8, 2, 1, 1 } },       // clip at 8
        { { 1e12, 1e12, 1, 1e-40 },          2.0f,      { 4, 4, 0, 0 } },
        { { 4, 1, 0.25, 0.0625 },            2.0f,      { 4, 2, 2, 0 } },       // ideal 3.5 2.5 1.5 0.5, half to even
        { { 1, 2, 3, 4, 5, 6, 7, 8 },        2.0f,      { 1, 2, 2, 2, 2, 2, 2, 3 } },
        { { 16, 16, 1, 1 },                  1.0f,      { 2, 2, 0, 0 } },
        { { 0, 0, 0, 1 },                    2.0f,      { 0, 0, 0, 8 } },       // zero variance, clip
        { { 1024, 64, 4, 1, 1.0/16, std::pow(2.0, -20) }, 2.5f, { 6, 4, 3, 2, 0, 0 } },
    };
    for (size_t i = 0; i < cases.size(); i++) {
        const auto b = llama_kvq_alloc_bits(cases[i].var, cases[i].avg);
        CHECK(b == cases[i].want, "alloc_bits case %zu", i);
    }

    std::mt19937 rng(2);
    std::lognormal_distribution<double> ld(0.0, 3.0);
    for (int it = 0; it < 200; it++) {
        const int n = 1 + it % 300;
        std::vector<double> var(n);
        for (double & v : var) {
            v = ld(rng);
        }
        for (float avg : { 0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 7.5f }) {
            const auto b = llama_kvq_alloc_bits(var, avg);
            int sum = 0;
            bool mono = true;
            for (int c = 0; c < n; c++) {
                sum += b[c];
                for (int k = 0; k < n && mono; k++) {
                    mono = !(var[c] > var[k] && b[c] + 1 < b[k]);
                }
            }
            CHECK(sum == (int) std::nearbyint(avg*n), "alloc_bits sum %d, want %d (n = %d)", sum, (int) std::nearbyint(avg*n), n);
            CHECK(mono, "alloc_bits: larger variance got fewer bits (n = %d)", n);
        }
    }
    printf("alloc_bits ok\n");
}

// T x n rows with a random correlated covariance and a non-zero mean
static std::vector<float> make_data(int64_t T, int n, std::mt19937 & rng) {
    std::normal_distribution<float> nd;
    std::vector<float> mix((size_t) n*n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mix[(size_t) i*n + j] = nd(rng)*std::pow(0.93f, (float) j);
        }
    }
    std::vector<float> x((size_t) T*n);
    std::vector<float> z(n);
    for (int64_t t = 0; t < T; t++) {
        for (int j = 0; j < n; j++) {
            z[j] = nd(rng);
        }
        for (int i = 0; i < n; i++) {
            float s = 0.3f*i/n;
            for (int j = 0; j < n; j++) {
                s += mix[(size_t) i*n + j]*z[j];
            }
            x[t*n + i] = s;
        }
    }
    return x;
}

static std::vector<double> random_spd(int n, std::mt19937 & rng) {
    std::normal_distribution<double> nd;
    std::vector<double> b((size_t) n*n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[(size_t) i*n + j] = nd(rng)*(1.0 + 4.0*(i % 7 == 0));
        }
    }
    std::vector<double> m((size_t) n*n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += b[(size_t) k*n + i]*b[(size_t) k*n + j];
            }
            m[(size_t) i*n + j] = s/n;
        }
    }
    return m;
}

// sum over rows of d^T M d with d = a - b
static double weighted_err(const std::vector<float> & a, const std::vector<float> & b, int n, const std::vector<double> & m) {
    const int64_t T = a.size()/n;
    std::vector<float> d(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        d[i] = a[i] - b[i];
    }
    std::vector<double> g((size_t) n*n, 0.0);
    llama_kvq_gram(d.data(), T, n, n, g.data(), 4);
    double s = 0.0;
    for (size_t i = 0; i < g.size(); i++) {
        s += g[i]*m[i];
    }
    return s;
}

static void test_gram() {
    std::mt19937 rng(3);
    const int n = 7;
    const int64_t T = 1000;
    std::vector<float> x = make_data(T, n, rng);
    std::vector<double> g((size_t) n*n, 1.0); // accumulates
    llama_kvq_gram(x.data(), T, n, n, g.data(), 3);
    double err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 1.0;
            for (int64_t t = 0; t < T; t++) {
                s += (double) x[t*n + i]*x[t*n + j];
            }
            err = std::max(err, std::fabs(s - g[(size_t) i*n + j])/std::fabs(s));
        }
    }
    CHECK(err < 1e-5, "gram: rel error %e", err);
    printf("gram ok\n");
}

static void test_codec() {
    std::mt19937 rng(4);
    const int n = 64;
    const int64_t T = 20000;
    const std::vector<float> x = make_data(T, n, rng);

    // identity metric: fwd is an orthogonal PCA basis up to the regularisation scale, sigma^2 = covariance eigenvalues
    {
        std::vector<double> eye((size_t) n*n, 0.0);
        for (int i = 0; i < n; i++) {
            eye[(size_t) i*n + i] = 1.0;
        }
        llama_kvq_table t;
        CHECK(llama_kvq_fit(t, x.data(), T, n, eye.data(), 2.0f, false, 4), "fit identity");

        std::vector<double> mu(n, 0.0);
        for (int64_t r = 0; r < T; r++) {
            for (int i = 0; i < n; i++) {
                mu[i] += x[r*n + i];
            }
        }
        std::vector<float> xc(x.size());
        for (int64_t r = 0; r < T; r++) {
            for (int i = 0; i < n; i++) {
                xc[r*n + i] = x[r*n + i] - (float) (mu[i]/T);
            }
        }
        std::vector<double> cov((size_t) n*n, 0.0);
        llama_kvq_gram(xc.data(), T, n, n, cov.data(), 4);
        for (double & v : cov) {
            v /= T;
        }
        std::vector<double> w;
        llama_kvq_eigh(n, cov, w);

        double orth = 0.0;
        double sig  = 0.0;
        double pca  = 0.0;
        for (int c = 0; c < n; c++) {
            for (int k = 0; k < n; k++) {
                double s = 0.0;
                for (int i = 0; i < n; i++) {
                    s += (double) t.fwd[(size_t) i*n + c]*t.fwd[(size_t) i*n + k];
                }
                orth = std::max(orth, std::fabs(s - (c == k ? 1.0 : 0.0)));
            }
            sig = std::max(sig, std::fabs(t.sigma[c]*t.sigma[c] - w[c])/w[n - 1]);
            if (c >= n - 8) {
                // the leading components match the PCA eigenvectors up to sign
                double dot = 0.0;
                for (int i = 0; i < n; i++) {
                    dot += t.fwd[(size_t) i*n + c]*cov[(size_t) i*n + c];
                }
                pca = std::max(pca, std::fabs(1.0 - std::fabs(dot)));
            }
        }
        printf("identity metric: orthogonality %.2e, sigma^2 vs eigenvalues %.2e, 1 - |cos| top-8 %.2e\n", orth, sig, pca);
        CHECK(orth < 1e-4, "identity metric: fwd not orthogonal (%e)", orth);
        CHECK(sig  < 1e-4, "identity metric: sigma^2 != eigenvalues (%e)", sig);
        CHECK(pca  < 1e-4, "identity metric: not the PCA basis (%e)", pca);
    }

    const std::vector<double> m = random_spd(n, rng);

    // general metric: y = (x - mu) fwd is decorrelated with variance sigma^2, inv fwd = I, d^T M d = |dy|^2
    {
        llama_kvq_table t;
        CHECK(llama_kvq_fit(t, x.data(), T, n, m.data(), 2.0f, false, 4), "fit metric");
        std::vector<float> xc(x.size());
        for (int64_t r = 0; r < T; r++) {
            for (int i = 0; i < n; i++) {
                xc[r*n + i] = x[r*n + i] - t.mu[i];
            }
        }
        std::vector<double> gy((size_t) n*n, 0.0);
        std::vector<float> y(x.size());
        for (int64_t r = 0; r < T; r++) {
            for (int c = 0; c < n; c++) {
                double s = 0.0;
                for (int i = 0; i < n; i++) {
                    s += (double) xc[r*n + i]*t.fwd[(size_t) i*n + c];
                }
                y[r*n + c] = (float) s;
            }
        }
        llama_kvq_gram(y.data(), T, n, n, gy.data(), 4);
        double off = 0.0;
        double dia = 0.0;
        double id  = 0.0;
        for (int c = 0; c < n; c++) {
            for (int k = 0; k < n; k++) {
                const double v = gy[(size_t) c*n + k]/T;
                const double s = (double) t.sigma[n - 1]*t.sigma[n - 1];
                if (c == k) {
                    dia = std::max(dia, std::fabs(v - (double) t.sigma[c]*t.sigma[c])/s);
                } else {
                    off = std::max(off, std::fabs(v)/s);
                }
                double p = 0.0;
                for (int i = 0; i < n; i++) {
                    p += (double) t.inv[(size_t) c*n + i]*t.fwd[(size_t) i*n + k];
                }
                id = std::max(id, std::fabs(p - (c == k ? 1.0 : 0.0)));
            }
        }

        std::vector<int8_t> codes(x.size());
        std::vector<float>  xr(x.size());
        llama_kvq_encode(t, x.data(), T, codes.data(), nullptr, 4);
        llama_kvq_decode(t, codes.data(), nullptr, T, xr.data(), 4);
        double ey = 0.0;
        for (int64_t r = 0; r < T; r++) {
            for (int c = 0; c < n; c++) {
                const float q = t.bits[c] ? (codes[r*n + c] + 0.5f)*t.step[c] : 0.0f;
                ey += (double) (y[r*n + c] - q)*(y[r*n + c] - q);
            }
        }
        const double ex = weighted_err(x, xr, n, m);
        printf("metric: cov(y) - diag(sigma^2): diag %.2e, off-diag %.2e (relative to max), |inv fwd - I| %.2e, d^T M d / |dy|^2 = %.6f\n", dia, off, id, ex/ey);
        CHECK(dia < 1e-4 && off < 1e-4, "y is not decorrelated (diag %e, off %e)", dia, off);
        CHECK(id < 1e-3, "inv fwd != I (%e)", id);
        CHECK(std::fabs(ex/ey - 1.0) < 1e-3, "weighted error != y-space error (%f)", ex/ey);
    }

    // quantizer round trip
    for (bool is_v : { false, true }) {
        for (float bits : { 1.0f, 2.0f, 4.0f }) {
            llama_kvq_table t;
            CHECK(llama_kvq_fit(t, x.data(), T, n, m.data(), bits, is_v, 4), "fit");
            std::vector<int8_t> codes(x.size());
            std::vector<int8_t> codes2(x.size());
            std::vector<float>  gain(T);
            std::vector<float>  gain2(T);
            std::vector<float>  y(x.size());
            llama_kvq_encode(t, x.data(), T, codes.data(), gain.data(), 4);
            bool range = true;
            for (int64_t r = 0; r < T; r++) {
                for (int c = 0; c < n; c++) {
                    const int b = t.bits[c];
                    const int q = codes[r*n + c];
                    range = range && (b == 0 ? q == 0 : (q >= -(1 << (b - 1)) && q < (1 << (b - 1))));
                }
            }
            CHECK(range, "codes out of range");
            llama_kvq_decode(t, codes.data(), gain.data(), T, y.data(), 4);
            // decoded rows sit at the cell centres: encoding them again gives the same codes
            llama_kvq_encode(t, y.data(), T, codes2.data(), gain2.data(), 4);
            size_t n_diff = 0;
            for (size_t i = 0; i < codes.size(); i++) {
                n_diff += codes[i] != codes2[i];
            }
            CHECK(is_v || n_diff == 0, "K: re-encode changed %zu codes", n_diff);
            // the gain of a decoded V row is close to, not equal to, the original
            CHECK(!is_v || n_diff < codes.size()/10, "V: re-encode changed %zu codes", n_diff);

            // fit_encode gives the same table, codes and gains as fit and encode
            llama_kvq_table t2;
            CHECK(llama_kvq_fit_encode(t2, x.data(), T, n, m.data(), bits, is_v, codes2.data(), gain2.data(), 4), "fit_encode");
            CHECK(t2.fwd == t.fwd && t2.bits == t.bits && t2.step == t.step, "fit_encode: other table");
            CHECK(codes2 == codes && (!is_v || gain2 == gain), "fit_encode: other codes");
        }
    }

    // weighted error at 2 bits: transform codec vs per-channel quantization with the same quantizer
    {
        llama_kvq_table t;
        llama_kvq_fit(t, x.data(), T, n, m.data(), 2.0f, false, 4);
        std::vector<int8_t> codes(x.size());
        std::vector<float>  y(x.size());
        llama_kvq_encode(t, x.data(), T, codes.data(), nullptr, 4);
        llama_kvq_decode(t, codes.data(), nullptr, T, y.data(), 4);
        const double e_tc = weighted_err(x, y, n, m);

        std::vector<float> yc(x.size());
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            double s2 = 0.0;
            for (int64_t r = 0; r < T; r++) {
                s += x[r*n + i];
            }
            const float mu = (float) (s/T);
            for (int64_t r = 0; r < T; r++) {
                s2 += (double) (x[r*n + i] - mu)*(x[r*n + i] - mu);
            }
            const float step = 0.9957f*(float) std::sqrt(s2/T);
            for (int64_t r = 0; r < T; r++) {
                const float q = std::clamp(std::floor((x[r*n + i] - mu)/step), -2.0f, 1.0f);
                yc[r*n + i] = (q + 0.5f)*step + mu;
            }
        }
        const double e_ch = weighted_err(x, yc, n, m);

        std::vector<float> mu_only(x.size());
        for (int64_t r = 0; r < T; r++) {
            for (int i = 0; i < n; i++) {
                mu_only[r*n + i] = t.mu[i];
            }
        }
        const double e_0 = weighted_err(x, mu_only, n, m);
        printf("2 bits, weighted error / signal: transform %.4f, per-channel %.4f\n", e_tc/e_0, e_ch/e_0);
        CHECK(e_tc < e_ch, "transform codec is not better than per-channel quantization");
    }

    // save / load
    {
        std::vector<llama_kvq_table> tables(2);
        llama_kvq_fit(tables[0], x.data(), T, n, m.data(), 2.0f, false, 4);
        llama_kvq_fit(tables[1], x.data(), T, n, m.data(), 3.0f, true,  4);
        tables[0].il = 3;  tables[0].head = 1;
        tables[1].il = 63; tables[1].head = 2;
        const std::string path = "test-kvq-codec.tmp";
        CHECK(llama_kvq_save(path, tables), "save");
        std::vector<llama_kvq_table> loaded;
        CHECK(llama_kvq_load(path, loaded), "load");
        CHECK(loaded.size() == 2, "load size");
        for (size_t i = 0; i < loaded.size() && i < 2; i++) {
            const auto & a = tables[i];
            const auto & b = loaded[i];
            CHECK(a.il == b.il && a.head == b.head && a.is_v == b.is_v && a.n == b.n && a.gain_den == b.gain_den &&
                  a.mu == b.mu && a.fwd == b.fwd && a.inv == b.inv && a.sigma == b.sigma && a.step == b.step && a.bits == b.bits,
                  "table %zu differs after load", i);
        }
        std::remove(path.c_str());
    }
}

int main() {
    test_eigh();
    test_alloc_bits();
    test_gram();
    test_codec();

    if (n_fail) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
