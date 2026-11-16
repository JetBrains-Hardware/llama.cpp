// Audit of an int8 (s8 x s8 -> s32) GEMM as a replacement for the f16 GEMM of prompt batches.
// Runs next to the f16 route on the same real call and logs the NMSE of each int8 scale scheme
// against the f16 result. Diagnostic only: simple kernels, explicit waits, never changes dst.
//   GGML_SYCL_INT8_AUDIT="nchunk/nout/clip/metric,..."  schemes (see ggml_sycl_int8_audit_scheme)
//   GGML_SYCL_INT8_AUDIT_LOG=<file>    append one line per call and scheme
//   GGML_SYCL_INT8_AUDIT_EVERY=<N>     audit every Nth eligible call (default 1)
//   GGML_SYCL_INT8_AUDIT_MIN_COLS=<n>  smallest batch audited (default 512)
#ifndef GGML_SYCL_INT8_AUDIT_HPP
#define GGML_SYCL_INT8_AUDIT_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>

struct ggml_sycl_int8_audit_scheme {
    int   nchunk = 1;   // K split into this many chunks, each with its own row and column scales
    int   nout   = 0;   // number of activation dimensions taken out of the int8 operand (exact product instead)
    float clip   = 0;   // > 0: column scale = min(absmax, clip * rms) with saturation
    int   metric = 0;   // outlier choice: 0 = sum_j x^2, 1 = sum_j (x / rms_j)^4
    float alpha  = 0;   // > 0: per-dimension equalization (SmoothQuant): x[d] /= s[d], w[:,d] *= s[d], s = xstat^alpha / wmax^(1-alpha)
    int   stat   = 0;   // xstat: 0 = max over the batch, 1 = rms over the batch
};

struct ggml_sycl_int8_audit_state {
    bool                                     init = false;
    std::vector<ggml_sycl_int8_audit_scheme> schemes;
    FILE *                                   log      = nullptr;
    int                                      every    = 1;
    int                                      min_cols = 512;
    int64_t                                  calls    = 0;
    // device scratch
    size_t   cap_w = 0, cap_x = 0, cap_c = 0, cap_m = 0, cap_n = 0, cap_k = 0;
    int8_t * W8 = nullptr; int8_t * X8 = nullptr; int32_t * C32 = nullptr; float * Cacc = nullptr; float * Cxmx = nullptr;
    float * rs = nullptr; float * cs = nullptr; float * crms = nullptr; float * pn = nullptr; float * pd = nullptr;
    float * en = nullptr; float * sd = nullptr; sycl::half * Xo = nullptr; uint8_t * mask = nullptr; int32_t * oidx = nullptr; float * werr = nullptr;
    bool    xmx_valid = false;
    bool    last_audited = false;   // the last call seen was audited
};

static ggml_sycl_int8_audit_state & ggml_sycl_int8_audit() {
    static ggml_sycl_int8_audit_state s;
    if (!s.init) {
        s.init = true;
        const char * e = getenv("GGML_SYCL_INT8_AUDIT");
        if (e && *e) {
            std::string str(e);
            size_t pos = 0;
            while (pos < str.size()) {
                size_t end = str.find(',', pos);
                if (end == std::string::npos) end = str.size();
                ggml_sycl_int8_audit_scheme sc;
                if (sscanf(str.substr(pos, end - pos).c_str(), "%d/%d/%f/%d/%f/%d", &sc.nchunk, &sc.nout, &sc.clip, &sc.metric, &sc.alpha, &sc.stat) >= 1) {
                    s.schemes.push_back(sc);
                }
                pos = end + 1;
            }
            const char * lf = getenv("GGML_SYCL_INT8_AUDIT_LOG");
            s.log = lf ? fopen(lf, "a") : stderr;
            if (getenv("GGML_SYCL_INT8_AUDIT_EVERY")) s.every = std::max(1, atoi(getenv("GGML_SYCL_INT8_AUDIT_EVERY")));
            if (getenv("GGML_SYCL_INT8_AUDIT_MIN_COLS")) s.min_cols = atoi(getenv("GGML_SYCL_INT8_AUDIT_MIN_COLS"));
        }
    }
    return s;
}

static inline bool ggml_sycl_int8_audit_on() { return !ggml_sycl_int8_audit().schemes.empty(); }

static void ggml_sycl_int8_audit_reserve(ggml_sycl_int8_audit_state & s, sycl::queue & q, size_t m, size_t n, size_t k) {
    auto grow = [&](auto *& p, size_t count) {
        if (p) sycl::free(p, q);
        p = (std::remove_reference_t<decltype(p)>) sycl::malloc_device(count * sizeof(*p), q);
        GGML_ASSERT(p != nullptr);
    };
    if (m * k > s.cap_w) { s.cap_w = m * k; grow(s.W8, s.cap_w); }
    if (k * n > s.cap_x) { s.cap_x = k * n; grow(s.X8, s.cap_x); grow(s.Xo, s.cap_x); }
    if (m * n > s.cap_c) { s.cap_c = m * n; grow(s.C32, s.cap_c); grow(s.Cacc, s.cap_c); grow(s.Cxmx, s.cap_c); s.xmx_valid = false; }
    if (m > s.cap_m) { s.cap_m = m; grow(s.rs, m); grow(s.werr, 2 * m); }
    if (n > s.cap_n) { s.cap_n = n; grow(s.cs, n); grow(s.crms, n); grow(s.pn, n); grow(s.pd, n); }
    if (k > s.cap_k) { s.cap_k = k; grow(s.en, k); grow(s.sd, k); grow(s.mask, k); grow(s.oidx, k); }
}

// sum of squared differences / sum of squared reference, accumulated per column on the device
static double ggml_sycl_int8_audit_nmse(ggml_sycl_int8_audit_state & s, sycl::queue & q, const float * c, const float * ref, size_t m, size_t n) {
    float * pn = s.pn; float * pd = s.pd;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> j) {
        float num = 0, den = 0;
        const float * a = c + j[0] * m; const float * b = ref + j[0] * m;
        for (size_t i = 0; i < m; ++i) { const float d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
        pn[j] = num; pd[j] = den;
    }).wait();
    std::vector<float> hn(n), hd(n);
    q.memcpy(hn.data(), pn, n * sizeof(float)).wait();
    q.memcpy(hd.data(), pd, n * sizeof(float)).wait();
    double num = 0, den = 0;
    for (size_t j = 0; j < n; ++j) { num += hn[j]; den += hd[j]; }
    return den > 0 ? num / den : 0;
}

// w16: m x k row-major f16 weights; x: k x n column-major f32 activations; ref: m x n column-major f16-route result
static void ggml_sycl_int8_audit_call(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const sycl::half * w16, const float * x,
                                      const float * ref, int64_t m_, int64_t n_, int64_t k_, const queue_ptr & stream) {
    auto & s = ggml_sycl_int8_audit();
    s.last_audited = false;
    if (n_ < s.min_cols) { s.xmx_valid = false; return; }
    if ((s.calls++ % s.every) != 0) { s.xmx_valid = false; return; }
    s.last_audited = true;
    sycl::queue & q = *stream;
    const size_t m = m_, n = n_, k = k_;
    q.wait();
    // keep the XMX reference before the scratch is (possibly) regrown
    const bool have_xmx = s.xmx_valid && m * n <= s.cap_c;
    ggml_sycl_int8_audit_reserve(s, q, m, n, k);
    auto eng = ctx.engine_dnnl(stream);
    auto strm = ctx.stream_dnnl(stream);
    int8_t * W8 = s.W8; int8_t * X8 = s.X8; int32_t * C32 = s.C32; float * Cacc = s.Cacc;
    float * rs = s.rs; float * cs = s.cs; float * crms = s.crms; float * en = s.en; float * sd = s.sd; uint8_t * mask = s.mask; int32_t * oidx = s.oidx; float * werr = s.werr;

    if (have_xmx && s.xmx_valid) {
        const double e = ggml_sycl_int8_audit_nmse(s, q, s.Cxmx, ref, m, n);
        fprintf(s.log, "AUD name=%s type=%s m=%zu k=%zu n=%zu scheme=xmx nmse=%.4e werr=0 xerr=0\n", src0->name, ggml_type_name(src0->type), m, k, n, e);
    }
    s.xmx_valid = false;

    // column rms over all dimensions
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> j) {
        float ss = 0; const float * p = x + j[0] * k;
        for (size_t d = 0; d < k; ++d) ss += p[d] * p[d];
        crms[j] = sycl::sqrt(ss / (float) k) + 1e-30f;
    }).wait();

    for (const auto & sc : s.schemes) {
        const int nchunk = std::max(1, sc.nchunk);
        const size_t nout = std::min<size_t>(std::max(0, sc.nout), k / 2);
        // outlier dimensions
        q.fill(mask, (uint8_t) 0, k).wait();
        if (nout > 0) {
            const int metric = sc.metric;
            q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> d) {
                float e = 0;
                for (size_t j = 0; j < n; ++j) {
                    float v = x[d[0] + j * k];
                    if (metric == 1) { v /= crms[j]; v = v * v; }
                    e += v * v;
                }
                en[d] = e;
            }).wait();
            std::vector<float> he(k);
            q.memcpy(he.data(), en, k * sizeof(float)).wait();
            std::vector<int32_t> idx(k);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + nout, idx.end(), [&](int32_t a, int32_t b) { return he[a] > he[b]; });
            std::vector<uint8_t> hm(k, 0);
            for (size_t t = 0; t < nout; ++t) hm[idx[t]] = 1;
            q.memcpy(mask, hm.data(), k).wait();
            q.memcpy(oidx, idx.data(), nout * sizeof(int32_t)).wait();
            // exact part: f16 GEMM of the full weights with the activations of the outlier dimensions only
            sycl::half * Xo = s.Xo;
            q.parallel_for(sycl::range<2>(n, k), [=](sycl::id<2> id) {
                const size_t i = id[0] * k + id[1];
                Xo[i] = mask[id[1]] ? (sycl::half) x[i] : (sycl::half) 0.0f;
            }).wait();
            DnnlGemmWrapper::row_gemm(ctx, m, n, k, w16, DnnlGemmWrapper::to_dt<sycl::half>(), Xo, DnnlGemmWrapper::to_dt<sycl::half>(), Cacc,
                                      DnnlGemmWrapper::to_dt<float>(), stream);
            q.wait();
        } else {
            q.fill(Cacc, 0.0f, m * n).wait();
        }
        if (sc.alpha > 0) {
            const float alpha = sc.alpha; const int stat = sc.stat;
            q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> d) {
                float xm = 0, wm = 0;
                for (size_t j = 0; j < n; ++j) { const float v = x[d[0] + j * k]; xm = stat == 1 ? xm + v * v : sycl::fmax(xm, sycl::fabs(v)); }
                if (stat == 1) xm = sycl::sqrt(xm / (float) n);
                for (size_t i = 0; i < m; ++i) wm = sycl::fmax(wm, sycl::fabs((float) w16[i * k + d[0]]));
                const float sv = (xm > 0 && wm > 0) ? sycl::pow(xm, alpha) / sycl::pow(wm, 1.0f - alpha) : 1.0f;
                sd[d] = (sycl::isfinite(sv) && sv > 0) ? sv : 1.0f;
            }).wait();
        } else {
            q.fill(sd, 1.0f, k).wait();
        }
        double werr_n = 0, werr_d = 0, xerr_n = 0, xerr_d = 0;
        const size_t kc = k / nchunk;
        for (int c = 0; c < nchunk; ++c) {
            const size_t c0 = c * kc, c1 = (c == nchunk - 1) ? k : c0 + kc, kl = c1 - c0;
            // weights: one scale per row (and chunk)
            q.parallel_for(sycl::range<1>(m), [=](sycl::id<1> r) {
                const sycl::half * p = w16 + r[0] * k;
                float mx = 0;
                for (size_t d = c0; d < c1; ++d) mx = sycl::fmax(mx, sycl::fabs((float) p[d] * sd[d]));
                const float sc_ = mx > 0 ? mx / 127.0f : 1.0f;
                float en_ = 0, ed_ = 0;
                for (size_t d = c0; d < c1; ++d) {
                    const float v = (float) p[d] * sd[d]; const float qv = sycl::rint(v / sc_);
                    W8[r[0] * k + d] = (int8_t) qv;
                    const float df = qv * sc_ - v; en_ += df * df; ed_ += v * v;
                }
                rs[r] = sc_; werr[2 * r[0]] = en_; werr[2 * r[0] + 1] = ed_;
            }).wait();
            { std::vector<float> h(2 * m); q.memcpy(h.data(), werr, 2 * m * sizeof(float)).wait(); for (size_t r = 0; r < m; ++r) { werr_n += h[2 * r]; werr_d += h[2 * r + 1]; } }
            // activations: one scale per column (and chunk), outlier dimensions zeroed
            const float clip = sc.clip;
            float * pn = s.pn; float * pd = s.pd;
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> j) {
                const float * p = x + j[0] * k;
                float mx = 0, ss = 0;
                for (size_t d = c0; d < c1; ++d) { if (!mask[d]) { const float v = p[d] / sd[d]; mx = sycl::fmax(mx, sycl::fabs(v)); ss += v * v; } }
                if (clip > 0) mx = sycl::fmin(mx, clip * sycl::sqrt(ss / (float) (c1 - c0)));
                const float sc_ = mx > 0 ? mx / 127.0f : 1.0f;
                float en_ = 0, ed_ = 0;
                for (size_t d = c0; d < c1; ++d) {
                    const float v = mask[d] ? 0.0f : p[d] / sd[d]; const float pv = p[d] / sd[d];
                    const float qv = sycl::clamp(sycl::rint(v / sc_), -127.0f, 127.0f);
                    X8[j[0] * k + d] = (int8_t) qv;
                    const float df = qv * sc_ - v; en_ += df * df; ed_ += pv * pv;
                }
                cs[j] = sc_; pn[j] = en_; pd[j] = ed_;
            }).wait();
            { std::vector<float> h(n), g(n); q.memcpy(h.data(), pn, n * sizeof(float)).wait(); q.memcpy(g.data(), pd, n * sizeof(float)).wait();
              for (size_t j = 0; j < n; ++j) { xerr_n += h[j]; xerr_d += g[j]; } }
            // s8 x s8 -> s32 on the chunk
            using dt = dnnl::memory::data_type;
            dnnl::memory::desc a_md({ 1, (dnnl_dim_t) m, (dnnl_dim_t) kl }, dt::s8, dnnl::memory::dims{ (dnnl_dim_t) (m * k), (dnnl_dim_t) k, 1 });
            dnnl::memory::desc b_md({ 1, (dnnl_dim_t) kl, (dnnl_dim_t) n }, dt::s8, dnnl::memory::dims{ (dnnl_dim_t) (n * k), 1, (dnnl_dim_t) k });
            dnnl::memory::desc c_md({ 1, (dnnl_dim_t) m, (dnnl_dim_t) n }, dt::s32, dnnl::memory::dims{ (dnnl_dim_t) (m * n), 1, (dnnl_dim_t) m });
            dnnl::matmul::primitive_desc pd_(eng, a_md, b_md, c_md);
            dnnl::matmul prim(pd_);
            std::unordered_map<int, dnnl::memory> args = { { DNNL_ARG_SRC, dnnl::memory(a_md, eng, W8 + c0) },
                                                           { DNNL_ARG_WEIGHTS, dnnl::memory(b_md, eng, X8 + c0) },
                                                           { DNNL_ARG_DST, dnnl::memory(c_md, eng, C32) } };
            prim.execute(strm, args);
            strm.wait();
            q.parallel_for(sycl::range<2>(n, m), [=](sycl::id<2> id) {
                const size_t i = id[1] + id[0] * m;
                Cacc[i] += (float) C32[i] * rs[id[1]] * cs[id[0]];
            }).wait();
        }
        const double e = ggml_sycl_int8_audit_nmse(s, q, Cacc, ref, m, n);
        fprintf(s.log, "AUD name=%s type=%s m=%zu k=%zu n=%zu scheme=c%d_o%zu_clip%g_m%d_a%g_s%d nmse=%.4e werr=%.4e xerr=%.4e\n", src0->name,
                ggml_type_name(src0->type), m, k, n, nchunk, nout, sc.clip, sc.metric, sc.alpha, sc.stat, e, werr_d > 0 ? werr_n / werr_d : 0, xerr_d > 0 ? xerr_n / xerr_d : 0);
    }
    fflush(s.log);
}

#endif // GGML_SYCL_INT8_AUDIT_HPP
