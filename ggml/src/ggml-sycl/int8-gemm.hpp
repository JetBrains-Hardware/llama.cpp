// int8 GEMM for prompt batches of reordered q4_K / q6_K weights: C = W * X as s8 x s8 -> s32 inside oneDNN,
// about twice the f16 GEMM rate on XMX hardware.
//
// A library GEMM only knows one scale per weight row and one per activation column, which is far too coarse
// for the residual stream (a handful of dimensions carry values tens of times above the rest). The path
// therefore equalizes per input dimension d first (SmoothQuant): x[d] /= s[d], W[:, d] *= s[d] with
// s[d] = sqrt(rms_batch(x[d] / rms(x column)) / max_rows|W[:, d]|), which leaves the product unchanged. The scales are a pure
// function of the call's inputs (no calibration state), so results do not depend on earlier requests.
//
// Used only for wide expanding projections (rows >= 3 x columns: the FFN gate / up projections). Measured on
// two 27B models: projections that read intermediate activations (rows <= columns) are far outside the error
// of the per-32 XMX kernel with any row / column scale scheme (int8-audit.hpp), and the attention-side
// projections (rows 1.2-2.4 x columns) pass that per-call error check but raised perplexity by up to 7% on
// some inputs. Both stay on the f16 route.
//
// Scales. The equalization factors are a statistic of the activations that changes little from one batch to
// the next, so by default they are kept per weight tensor together with the row scales (calibration) and, while
// device memory allows, with the s8 copy of the weight itself; a call then only quantizes the activations.
// Every call still derives the factors its own batch would give and estimates the quantization noise power of
// the product for both sets of factors (ggml_sycl_int8_gemm: step 3). When the kept factors are noisier than the
// fresh ones by more than the threshold, the tensor is calibrated again on the spot, so a call is never served
// with factors much worse than its own.
// With GGML_SYCL_INT8_CALIB=0 nothing is kept: every call derives all scales from its own inputs.
//   GGML_SYCL_INT8_GEMM=0            turn the path off
//   GGML_SYCL_INT8_GEMM_MIN_COLS=<n> smallest batch that takes it (default 512)
//   GGML_SYCL_INT8_CALIB=0           per-call scales, no state kept between calls
//   GGML_SYCL_INT8_DRIFT=<x>         calibrate again when the estimated noise power with the kept factors is
//                                    above x times the one with fresh factors (default 1.15; huge: never)
//   GGML_SYCL_INT8_WCACHE_MB=<n>     cap of the s8 weight copies in MiB (0: keep none; default: no cap)
//   GGML_SYCL_INT8_WCACHE_RESERVE_MB=<n>  free device memory to leave untouched by the copies (default 2048)
//   GGML_SYCL_INT8_GEMM_STATS=1      print call / calibration / cache counters at exit (2: also per-stage
//                                    times, with waits)
//   GGML_SYCL_INT8_GEMM_MIN_RATIO / _MAX_RATIO   limit the path by rows / columns of the weight (diagnostics)
#ifndef GGML_SYCL_INT8_GEMM_HPP
#define GGML_SYCL_INT8_GEMM_HPP

#if GGML_SYCL_DNNL

#include <memory>
#include <chrono>
#include <cstdlib>

struct ggml_sycl_int8_gemm_cfg {
    bool    enabled  = true;
    int     min_cols = 512;
    float   min_ratio = 3.0f;     // rows / columns of the weight must be above 1 and within [min_ratio, max_ratio]
    float   max_ratio = 8.0f;     // FFN expansions are 2.7 .. 8; the vocabulary projection (logits) stays f16
    bool    calib    = true;      // keep the scales per weight tensor
    float   drift    = 1.15f;     // calibrate again above this ratio of estimated noise power, kept / fresh factors
    size_t  wcache_cap     = SIZE_MAX;            // bytes of s8 weight copies allowed
    size_t  wcache_reserve = (size_t) 2048 << 20; // free device memory the copies must leave
    // counters
    int64_t calls = 0, calibrations = 0, recalibrations = 0, cached_calls = 0, wcache_tensors = 0;
    size_t  wcache_bytes = 0, wcache_peak_bytes = 0;
    int64_t wcache_peak_tensors = 0;
    bool    wcache_mem_known = true;
};

static ggml_sycl_int8_gemm_cfg & ggml_sycl_int8_gemm_config() {
    static ggml_sycl_int8_gemm_cfg cfg = [] {
        ggml_sycl_int8_gemm_cfg c;
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM")) c.enabled = atoi(e) != 0;
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM_MIN_COLS")) c.min_cols = std::max(1, atoi(e));
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM_MIN_RATIO")) c.min_ratio = (float) atof(e);
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM_MAX_RATIO")) c.max_ratio = (float) atof(e);
        if (const char * e = getenv("GGML_SYCL_INT8_CALIB")) c.calib = atoi(e) != 0;
        if (const char * e = getenv("GGML_SYCL_INT8_DRIFT")) c.drift = (float) atof(e);
        if (const char * e = getenv("GGML_SYCL_INT8_WCACHE_MB")) c.wcache_cap = (size_t) std::max(0ll, atoll(e)) << 20;
        if (const char * e = getenv("GGML_SYCL_INT8_WCACHE_RESERVE_MB")) c.wcache_reserve = (size_t) std::max(0ll, atoll(e)) << 20;
        if (getenv("GGML_SYCL_INT8_GEMM_STATS")) {
            atexit([] {
                const auto & c = ggml_sycl_int8_gemm_config();
                fprintf(stderr, "ggml-sycl int8 GEMM calls: %lld\n", (long long) c.calls);
                fprintf(stderr, "ggml-sycl int8 GEMM calibrations: %lld first + %lld on drift; calls on a kept s8 weight: %lld; kept s8 weights: %lld tensors, %.0f MiB%s\n",
                        (long long) c.calibrations, (long long) c.recalibrations, (long long) c.cached_calls, (long long) c.wcache_peak_tensors,
                        c.wcache_peak_bytes / 1048576.0, c.wcache_mem_known ? "" : " (free device memory unknown)");
            });
        }
        return c;
    }();
    return cfg;
}

// host time of the blocking steps since the last report (GGML_SYCL_GRAPH_WALL diagnostics)
struct ggml_sycl_int8_host_times { double drift_us = 0, alloc_us = 0, meminfo_us = 0; };
static ggml_sycl_int8_host_times & ggml_sycl_int8_host() { static ggml_sycl_int8_host_times t; return t; }
struct ggml_sycl_int8_host_scope {
    double & acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ggml_sycl_int8_host_scope(double & a) : acc(a) {}
    ~ggml_sycl_int8_host_scope() { acc += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count(); }
};

// stage timers, only with GGML_SYCL_INT8_GEMM_STATS=2 (each stage then waits for the queue)
struct ggml_sycl_int8_gemm_timer {
    static constexpr int NSTAGE = 9;
    double us[NSTAGE] = {}; int64_t n = 0; bool on = false;
    std::chrono::steady_clock::time_point t0;
};
static ggml_sycl_int8_gemm_timer & ggml_sycl_int8_gemm_timers() {
    static ggml_sycl_int8_gemm_timer t = [] {
        ggml_sycl_int8_gemm_timer v;
        const char * e = getenv("GGML_SYCL_INT8_GEMM_STATS");
        v.on = e && atoi(e) >= 2;
        if (v.on) {
            atexit([] {
                auto & tt = ggml_sycl_int8_gemm_timers();
                static const char * names[ggml_sycl_int8_gemm_timer::NSTAGE] = { "state", "x rms", "factors+col stats", "decide", "row max", "weights->s8", "x->s8", "pd create", "gemm" };
                for (int i = 0; i < ggml_sycl_int8_gemm_timer::NSTAGE; ++i) fprintf(stderr, "ggml-sycl int8 GEMM stage %-14s %9.1f us/call\n", names[i], tt.us[i] / std::max<int64_t>(1, tt.n));
            });
        }
        return v;
    }();
    return t;
}
#define GGML_SYCL_INT8_STAGE(i) do { if (tm.on) { q.wait(); const auto t1 = std::chrono::steady_clock::now(); tm.us[i] += std::chrono::duration<double, std::micro>(t1 - tm.t0).count(); tm.t0 = t1; } } while (0)

// shape and type conditions (the weight layout is checked by the caller)
static inline bool ggml_sycl_int8_gemm_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, int64_t ncols) {
    const auto & cfg = ggml_sycl_int8_gemm_config();
    const float ratio = (float) src0->ne[1] / (float) src0->ne[0];
    return cfg.enabled && ratio >= cfg.min_ratio && ratio <= cfg.max_ratio && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) && src1->type == GGML_TYPE_F32 &&
           ncols >= cfg.min_cols && src0->ne[1] > src0->ne[0] && src0->ne[0] % QK_K == 0 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
           ggml_is_contiguous(src0);
}

static inline void ggml_sycl_int8_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// what the path keeps per weight tensor (in its extra)
struct ggml_sycl_int8_state {
    sycl::queue q;
    float *  wcm   = nullptr;  // [2k]  max over the rows of |W[:, d]|, then mean over the rows of W[:, d]^2
    float *  cal   = nullptr;  // [2k + 2m]  s, 1/s, 127 / row max, row max / 127 of the current calibration
    int8_t * w8    = nullptr;  // [m * k]  s8 weights of the current calibration, when kept
    size_t   w8_bytes = 0;
    bool     calibrated = false;
    double   cal_w2 = 0;       // current calibration: mean over the rows of sum_d (W s)^2
    double   cal_rowmax2 = 0;  // current calibration: mean over the rows of (max_d |W s|)^2
    int64_t  calls = 0;
    explicit ggml_sycl_int8_state(const sycl::queue & q_) : q(q_) {}
    ~ggml_sycl_int8_state() {
        try {
            if (wcm) sycl::free(wcm, q);
            if (cal) sycl::free(cal, q);
            if (w8) {
                ggml_sycl_free_device(w8, q);
                auto & cfg = ggml_sycl_int8_gemm_config();
                cfg.wcache_bytes -= w8_bytes;
                cfg.wcache_tensors--;
            }
        } catch (...) {
        }
    }
};

// the state of a weight tensor; max over the rows of |W[:, d]| is computed once, on first use
static ggml_sycl_int8_state * ggml_sycl_int8_gemm_state(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, ggml_tensor * dst,
                                                       const char * w, int64_t m, int64_t k, const queue_ptr & stream) {
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (extra->int8) {
        return extra->int8.get();
    }
    extra->int8 = std::make_shared<ggml_sycl_int8_state>(*stream);
    ggml_sycl_int8_state * st = extra->int8.get();
    float * cm = sycl::malloc_device<float>(2 * k, *stream);
    GGML_ASSERT(cm != nullptr);
    st->wcm = cm;
    st->cal = sycl::malloc_device<float>(2 * k + 2 * m, *stream);
    GGML_ASSERT(st->cal != nullptr);

    ggml_sycl_pool_alloc<sycl::half> w16(ctx.pool(), m * k);
    const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
    GGML_ASSERT(to_fp16_sycl != nullptr);
    to_fp16_sycl(w, w16.get(), m * k, stream);
    const sycl::half * wp = w16.get();
    const size_t mm = m, kk = k;
    // 64 rows per work-item along a column keeps the strided reads short; a second pass folds the partial maxima
    const size_t nseg = (mm + 63) / 64;
    ggml_sycl_pool_alloc<float> part(ctx.pool(), 2 * nseg * kk);
    float * pp = part.get();
    float * ps = pp + nseg * kk;
    stream->parallel_for(sycl::range<2>(nseg, kk), [=](sycl::id<2> id) {
        const size_t r0 = id[0] * 64, r1 = sycl::min(r0 + 64, mm);
        float mx = 0, ss = 0;
        for (size_t r = r0; r < r1; ++r) {
            const float v = (float) wp[r * kk + id[1]];
            mx = sycl::fmax(mx, sycl::fabs(v));
            ss += v * v;
        }
        pp[id[0] * kk + id[1]] = mx;
        ps[id[0] * kk + id[1]] = ss;
    });
    stream->parallel_for(sycl::range<1>(kk), [=](sycl::id<1> d) {
        float mx = 0, ss = 0;
        for (size_t sg = 0; sg < nseg; ++sg) {
            mx = sycl::fmax(mx, pp[sg * kk + d[0]]);
            ss += ps[sg * kk + d[0]];
        }
        cm[d]      = mx;
        cm[kk + d] = ss / (float) mm;
    });
    stream->wait();  // one time per tensor; the temporaries go back to the pool
    return st;
}

// May this tensor keep an s8 copy of itself (bytes)? Only while the copies stay under the cap and the device
// keeps the reserve free. The caller asks from the third call of a tensor on: the first ones are usually the
// load-time warm-up, when the companions of the model (draft model, second context) may not be loaded yet and
// the free memory would be measured too early.
static bool ggml_sycl_int8_wcache_admit(ggml_backend_sycl_context & ctx, size_t bytes) {
    auto & cfg = ggml_sycl_int8_gemm_config();
    if (cfg.wcache_cap == 0 || cfg.wcache_bytes + bytes > cfg.wcache_cap) {
        return false;
    }
    size_t free_b = 0, total_b = 0;
    ggml_sycl_int8_host_scope host_t(ggml_sycl_int8_host().meminfo_us);
    if (!sycl_get_mem_info(ctx.device, &free_b, &total_b) || total_b == 0 || free_b >= total_b) {
        // no usable reading of the free memory: keep copies only under an explicit cap
        cfg.wcache_mem_known = false;
        return cfg.wcache_cap != SIZE_MAX;
    }
    return free_b >= bytes + cfg.wcache_reserve;
}

// dst (m x n, column-major, f32) = W (reordered q4_K / q6_K, m x k) * x (k x n, column-major, f32)
static bool ggml_sycl_int8_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, ggml_tensor * dst, const char * w, const float * x,
                                float * out, int64_t m_, int64_t n_, int64_t k_, const queue_ptr & stream) {
    const size_t m = m_, n = n_, k = k_;
    auto & tm = ggml_sycl_int8_gemm_timers();
    if (tm.on) { stream->wait(); tm.t0 = std::chrono::steady_clock::now(); tm.n++; }
    const size_t nb = m * k / QK_K, bpr = k / QK_K;
    const bool   q4 = src0->type == GGML_TYPE_Q4_K;
    ggml_sycl_int8_state * st = ggml_sycl_int8_gemm_state(ctx, src0, dst, w, m_, k_, stream);
    const float * wcm = st->wcm;
    auto & cfg = ggml_sycl_int8_gemm_config();
    cfg.calls++;
    st->calls++;
    const bool keep = cfg.calib;   // scales (and, memory allowing, the s8 weight) live in the tensor state

    ggml_sycl_pool_alloc<int8_t> x8_alloc(ctx.pool(), k * n);
    // small per-call arrays in one allocation; K = with the kept factors, F = with the factors of this batch:
    // 127/colmax and colmax/127 per column for K and F [4n]; what the host reads back: colmax^2 and sum of squares
    // per column for K and F [4n], partial sums of (s_F W)^2 [k/64], of the squared row maxima [m/64 rounded up];
    // partial maxima and sums of squares per 64 dimensions [4 * n * k/64], xss [n/32 * k], fresh s and 1/s [2k],
    // and without kept scales s, 1/s, rowq, rows
    const size_t kc = k / 64, mseg = (m + 63) / 64;
    ggml_sycl_pool_alloc<float> f_alloc(ctx.pool(), 8 * n + kc + mseg + 4 * n * kc + ((n + 31) / 32) * k + 2 * k + (keep ? 0 : 2 * k + 2 * m));
    int8_t * x8 = x8_alloc.get();
    float * cinvK = f_alloc.get(); float * colsK = cinvK + n; float * cinvF = colsK + n; float * colsF = cinvF + n;
    float * stat = colsF + n;        // mk2[n], qk[n], mf2[n], qf[n], apart[kc]
    float * apart = stat + 4 * n; float * bpart = apart + kc;
    float * part = bpart + mseg;
    float * xss = part + 4 * n * kc;
    float * sF = xss + ((n + 31) / 32) * k; float * sFinv = sF + k;
    float * sd = keep ? st->cal : sFinv + k; float * sinv = sd + k; float * rowq = sinv + k; float * rows = rowq + m;
    float * cinv = cinvK;   // per-column sums of squares of the raw activations while the rms is taken (step 1)
    float * cmax = part;
    const float * wms = wcm + k;
    sycl::queue & q = *stream;

    // rms of every dimension over the batch: one copy per activation tensor of the graph (gate and up read the same one)
    auto & xc = ctx.int8_xr_cache;
    const ggml_tensor * node_src1 = ctx.src1_f16_cache.node_src1;
    const bool xr_cacheable = node_src1 != nullptr && node_src1->type == GGML_TYPE_F32;
    const bool xr_hit = xr_cacheable && xc.tensor == node_src1 && xc.data == (const void *) x && xc.n == n && xc.k == k;
    if (xc.cap < k) {
        if (xc.buf != nullptr) {
            q.wait(); // queued kernels may still read the old copy
            sycl::free(xc.buf, q);
        }
        xc.tensor = nullptr;
        xc.cap    = 0;
        xc.buf    = sycl::malloc_device<float>(k, q);
        GGML_ASSERT(xc.buf != nullptr);
        xc.cap = k;
    }
    float * xr = xc.buf;
    GGML_SYCL_INT8_STAGE(0);

    // 1. per-dimension equalization factor from the rms of the dimension over the batch and the weight column max
    //    (sums of squares over 32 columns x 16 dimensions per work-item: contiguous reads; a strided per-dimension
    //    loop over the batch measured 0.8 ms per call)
    //    Every column (token) is first normalized by its own mean square, so that each token weighs the same:
    //    a plain rms over the batch is dominated by the few tokens with a very large norm.
    if (!xr_hit) {
    q.parallel_for(sycl::range<2>(n, kc), [=](sycl::id<2> id) {
        const float * p = x + id[0] * k + id[1] * 64;
        float ss = 0;
        for (int j = 0; j < 64; ++j) ss += p[j] * p[j];
        cmax[id[0] * kc + id[1]] = ss;
    });
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> c) {
        float ss = 0;
        for (size_t j = 0; j < kc; ++j) ss += cmax[c[0] * kc + j];
        cinv[c] = ss > 0 ? (float) k / ss : 0.0f;   // 1 / mean square of the column (cinv is overwritten in step 4)
    });
    const size_t nseg = (n + 31) / 32;
    q.parallel_for(sycl::range<2>(nseg, k / 16), [=](sycl::id<2> id) {
        const size_t j0 = id[0] * 32, j1 = sycl::min(j0 + 32, n);
        // scalar accesses: the activations are only guaranteed the tensor alignment (32 bytes), a 16-float
        // vector load assumes 64 and silently reads from the rounded-down address when it does not hold
        float acc[16] = { 0 };
        for (size_t j = j0; j < j1; ++j) {
            const float * p = x + j * k + id[1] * 16;
            const float wj = cinv[j];
            for (int l = 0; l < 16; ++l) acc[l] += p[l] * p[l] * wj;
        }
        float * o = xss + id[0] * k + id[1] * 16;
        for (int l = 0; l < 16; ++l) o[l] = acc[l];
    });
    q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> d) {
        float ss = 0;
        for (size_t sg = 0; sg < nseg; ++sg) ss += xss[sg * k + d[0]];
        xr[d] = sycl::sqrt(ss / (float) n);
    });
    xc.tensor = xr_cacheable ? node_src1 : nullptr;
    xc.data   = x;
    xc.n      = n;
    xc.k      = k;
    }
    GGML_SYCL_INT8_STAGE(1);
    // 2. factors this batch would give, and the column statistics of x / s for the kept and for the fresh factors
    q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> d) {
        const float xv = xr[d], wm = wcm[d];
        float sv = (xv > 0 && wm > 0) ? sycl::sqrt(xv / wm) : 1.0f;
        if (!(sv > 1e-18f && sv < 1e18f)) sv = 1.0f;
        sF[d] = sv; sFinv[d] = 1.0f / sv;
    });
    const bool have_kept = keep && st->calibrated;
    const float * sKinv = have_kept ? sinv : sFinv;
    q.parallel_for(sycl::range<2>(n, kc), [=](sycl::id<2> id) {
        const float * p = x + id[0] * k + id[1] * 64; const float * sk = sKinv + id[1] * 64; const float * sf = sFinv + id[1] * 64;
        float mk = 0, qk = 0, mf = 0, qf = 0;
        for (int j = 0; j < 64; ++j) {
            const float vk = p[j] * sk[j], vf = p[j] * sf[j];
            mk = sycl::fmax(mk, sycl::fabs(vk)); qk += vk * vk;
            mf = sycl::fmax(mf, sycl::fabs(vf)); qf += vf * vf;
        }
        float * o = part + 4 * (id[0] * kc + id[1]);
        o[0] = mk; o[1] = qk; o[2] = mf; o[3] = qf;
    });
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> c) {
        float mk = 1e-30f, qk = 0, mf = 1e-30f, qf = 0;
        for (size_t j = 0; j < kc; ++j) {
            const float * o = part + 4 * (c[0] * kc + j);
            mk = sycl::fmax(mk, o[0]); qk += o[1];
            mf = sycl::fmax(mf, o[2]); qf += o[3];
        }
        colsK[c] = mk / 127.0f; cinvK[c] = 127.0f / mk;
        colsF[c] = mf / 127.0f; cinvF[c] = 127.0f / mf;
        stat[c] = mk * mk; stat[n + c] = qk; stat[2 * n + c] = mf * mf; stat[3 * n + c] = qf;
    });
    GGML_SYCL_INT8_STAGE(2);
    // 3. calibrate: always without kept scales, else on the first call of the tensor and when the kept factors
    //    are noisier than the fresh ones by more than the threshold. Noise power of the product, summed over the
    //    output, up to a constant: the int8 step of a column is its largest |x / s|, that of a row its largest |W s|,
    //        N = mean_c(colmax^2) * mean_r(sum_d (W s)^2)  +  mean_r(rowmax^2) * mean_c(sum_d (x / s)^2).
    //    For the kept factors both row terms are known from the calibration; for the fresh ones the first follows
    //    from the mean square of the weight columns and the second is scaled like the first.
    bool cal = !have_kept;
    double w2F = 0;
    if (keep) {
        q.parallel_for(sycl::range<1>(kc), [=](sycl::id<1> c) {
            float a = 0;
            for (size_t d = c[0] * 64; d < c[0] * 64 + 64; ++d) a += sF[d] * sF[d] * wms[d];
            apart[c] = a;
        });
        std::vector<float> h(4 * n + kc);
        {
            ggml_sycl_int8_host_scope host_t(ggml_sycl_int8_host().drift_us);
            q.memcpy(h.data(), stat, sizeof(float) * (4 * n + kc)).wait();
        }
        double mk2 = 0, qk = 0, mf2 = 0, qf = 0;
        for (size_t c = 0; c < n; ++c) {
            mk2 += h[c]; qk += h[n + c]; mf2 += h[2 * n + c]; qf += h[3 * n + c];
        }
        for (size_t c = 0; c < kc; ++c) w2F += h[4 * n + c];
        if (cal) {
            cfg.calibrations++;
        } else if (st->cal_w2 > 0 && w2F > 0) {
            const double nK = mk2 * st->cal_w2 + st->cal_rowmax2 * qk;
            const double nF = mf2 * w2F + st->cal_rowmax2 * (w2F / st->cal_w2) * qf;
            if (nK > (double) cfg.drift * nF) {
                cal = true;
                cfg.recalibrations++;
            }
        }
    }
    if (cal) {
        q.memcpy(sd, sF, sizeof(float) * 2 * k);   // s and 1/s
        st->calibrated = keep;
    }
    const float * cols = cal ? colsF : colsK;
    const float * cinvx = cal ? cinvF : cinvK;
    // s8 weights: the kept copy when there is one (a tensor asks for it from its third call on), else a per-call one
    bool convert = cal;
    int8_t * w8 = nullptr;
    if (keep) {
        if (st->w8 == nullptr && st->calls >= 3 && ggml_sycl_int8_wcache_admit(ctx, m * k)) {
            {
                ggml_sycl_int8_host_scope host_t(ggml_sycl_int8_host().alloc_us);
                st->w8 = (int8_t *) ggml_sycl_malloc_device(m * k, q);
            }
            if (st->w8 != nullptr) {
                st->w8_bytes = m * k;
                cfg.wcache_bytes += m * k;
                cfg.wcache_tensors++;
                cfg.wcache_peak_bytes   = std::max(cfg.wcache_peak_bytes, cfg.wcache_bytes);
                cfg.wcache_peak_tensors = std::max(cfg.wcache_peak_tensors, cfg.wcache_tensors);
                convert = true;
            }
        }
        w8 = st->w8;
    }
    ggml_sycl_pool_alloc<int8_t> w8_alloc(ctx.pool());
    if (w8 == nullptr) {
        w8      = w8_alloc.alloc(m * k);
        convert = true;
    } else if (!convert) {
        cfg.cached_calls++;
    }
    ggml_sycl_pool_alloc<float> bmax_alloc(ctx.pool());
    float * bmax = cal ? bmax_alloc.alloc(4 * nb) : nullptr;
    GGML_SYCL_INT8_STAGE(3);
    // 4. row scales: exact max of |W[r, d] * s[d]| over the row (partial maxima per 64 weights, then per row).
    //    A bound from the block scales alone is cheaper but costs about one bit: measured NMSE 2-4x higher.
    const uint8_t * base = (const uint8_t *) w;
    if (q4) {
        const uint8_t * scp = base + nb * (QK_K / 2); const ggml_half * dmp = (const ggml_half *) (base + nb * (QK_K / 2) + nb * K_SCALE_SIZE);
        const uint8_t * qs0 = base;
        if (cal) {
        q.parallel_for(sycl::range<1>(nb * 4), [=](sycl::id<1> id) {
            const size_t i = id[0] / 4; const int il = id[0] % 4;
            const float dall = (float) sycl::bit_cast<sycl::half>(dmp[2 * i]), dmin = (float) sycl::bit_cast<sycl::half>(dmp[2 * i + 1]);
            uint8_t sc, mn;
            ggml_sycl_int8_scale_min_k4(2 * il, scp + i * K_SCALE_SIZE, sc, mn); const float d1 = dall * sc, m1 = dmin * mn;
            ggml_sycl_int8_scale_min_k4(2 * il + 1, scp + i * K_SCALE_SIZE, sc, mn); const float d2 = dall * sc, m2 = dmin * mn;
            const uint8_t * qp = qs0 + i * (QK_K / 2) + 32 * il; const float * sp = sd + (i % bpr) * QK_K + 64 * il;
            float mx = 0;
            for (int l = 0; l < 32; ++l) {
                const uint8_t v = qp[l];
                mx = sycl::fmax(mx, sycl::fmax(sycl::fabs((d1 * (v & 0xF) - m1) * sp[l]), sycl::fabs((d2 * (v >> 4) - m2) * sp[l + 32])));
            }
            bmax[id] = mx;
        });
        q.parallel_for(sycl::range<1>(m), [=](sycl::id<1> r) {
            float mx = 1e-30f;
            for (size_t j = 0; j < bpr * 4; ++j) mx = sycl::fmax(mx, bmax[r[0] * bpr * 4 + j]);
            rowq[r] = 127.0f / mx; rows[r] = mx / 127.0f;
        });
        }
        GGML_SYCL_INT8_STAGE(4);
        // 5. weights -> s8: 32 work-items per block of 256, 8 weights each
        const uint8_t * qs = base;
        if (convert)
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nb * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_group(0); const int tid = it.get_local_id(0), il = tid / 8, ir = tid % 8;
            const float rq = rowq[i / bpr];
            const float dall = (float) sycl::bit_cast<sycl::half>(dmp[2 * i]) * rq, dmin = (float) sycl::bit_cast<sycl::half>(dmp[2 * i + 1]) * rq;
            uint8_t sc, mn;
            ggml_sycl_int8_scale_min_k4(2 * il, scp + i * K_SCALE_SIZE, sc, mn); const float d1 = dall * sc, m1 = dmin * mn;
            ggml_sycl_int8_scale_min_k4(2 * il + 1, scp + i * K_SCALE_SIZE, sc, mn); const float d2 = dall * sc, m2 = dmin * mn;
            const size_t col = (i % bpr) * QK_K + 64 * il + 4 * ir, off = i * QK_K + 64 * il + 4 * ir;
            const sycl::vec<uint8_t, 4> qv = *reinterpret_cast<const sycl::vec<uint8_t, 4> *>(qs + i * (QK_K / 2) + 32 * il + 4 * ir);
            const sycl::vec<float, 4> s0 = *reinterpret_cast<const sycl::vec<float, 4> *>(sd + col), s1 = *reinterpret_cast<const sycl::vec<float, 4> *>(sd + col + 32);
            sycl::vec<int8_t, 4> lo, hi;
            for (int l = 0; l < 4; ++l) {
                lo[l] = (int8_t) sycl::rint((d1 * (qv[l] & 0xF) - m1) * s0[l]);
                hi[l] = (int8_t) sycl::rint((d2 * (qv[l] >> 4) - m2) * s1[l]);
            }
            *reinterpret_cast<sycl::vec<int8_t, 4> *>(w8 + off) = lo;
            *reinterpret_cast<sycl::vec<int8_t, 4> *>(w8 + off + 32) = hi;
        });
    } else {
        const uint8_t * ql = base; const uint8_t * qh = base + nb * (QK_K / 2); const int8_t * scp = (const int8_t *) (base + nb * (QK_K / 2) + nb * (QK_K / 4));
        const ggml_half * dp = (const ggml_half *) (base + nb * (QK_K / 2 + QK_K / 4 + QK_K / 16));
        if (cal) {
        q.parallel_for(sycl::range<1>(nb * 4), [=](sycl::id<1> id) {
            const size_t i = id[0] / 4; const int c4 = id[0] % 4, ip = c4 / 2, e0 = 64 * (c4 % 2);   // 64 weights: half ip, elements e0..e0+63
            const int shift = e0 >= 64 ? 4 : 0;
            const float d = (float) sycl::bit_cast<sycl::half>(dp[i]);
            const uint8_t * qlp = ql + i * (QK_K / 2) + 64 * ip; const uint8_t * qhp = qh + i * (QK_K / 4) + 32 * ip;
            const float * sp = sd + (i % bpr) * QK_K + 128 * ip + e0;
            float mx = 0;
            for (int e = 0; e < 64; ++e) {
                const int eh = e0 + e, qb = 2 * (eh / 32);
                const int qv = (int) (int8_t) (((qlp[eh & 63] >> shift) & 0xF) | (((qhp[eh & 31] >> qb) & 3) << 4)) - 32;
                mx = sycl::fmax(mx, sycl::fabs(d * (float) scp[i * 16 + 8 * ip + eh / 16] * (float) qv * sp[e]));
            }
            bmax[id] = mx;
        });
        q.parallel_for(sycl::range<1>(m), [=](sycl::id<1> r) {
            float mx = 1e-30f;
            for (size_t j = 0; j < bpr * 4; ++j) mx = sycl::fmax(mx, bmax[r[0] * bpr * 4 + j]);
            rowq[r] = 127.0f / mx; rows[r] = mx / 127.0f;
        });
        }
        GGML_SYCL_INT8_STAGE(4);
        // 64 work-items per block of 256, 4 weights each (same decomposition as dequantize_block_q6_K_reorder)
        if (convert)
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nb * 64), sycl::range<1>(64)), [=](sycl::nd_item<1> it) {
            const size_t i = it.get_group(0); const int tid = it.get_local_id(0), ip = tid / 32, eh0 = 4 * (tid - 32 * ip);
            const int shift = eh0 >= 64 ? 4 : 0, qb = 2 * (eh0 / 32);
            const float f = (float) sycl::bit_cast<sycl::half>(dp[i]) * (float) scp[i * 16 + 8 * ip + eh0 / 16] * rowq[i / bpr];
            const uint8_t * ql4 = ql + i * (QK_K / 2) + 64 * ip + (eh0 & 63); const uint8_t * qh4 = qh + i * (QK_K / 4) + 32 * ip + (eh0 & 31);
            const size_t col = (i % bpr) * QK_K + 128 * ip + eh0;
            const sycl::vec<float, 4> s0 = *reinterpret_cast<const sycl::vec<float, 4> *>(sd + col);
            sycl::vec<int8_t, 4> v;
            for (int j = 0; j < 4; ++j) {
                const int qv = (int) (int8_t) (((ql4[j] >> shift) & 0xF) | (((qh4[j] >> qb) & 3) << 4)) - 32;
                v[j] = (int8_t) sycl::rint(f * (float) qv * s0[j]);
            }
            *reinterpret_cast<sycl::vec<int8_t, 4> *>(w8 + i * QK_K + 128 * ip + eh0) = v;
        });
    }
    if (cal && keep) {
        // row terms of the noise estimate for this calibration
        q.parallel_for(sycl::range<1>(mseg), [=](sycl::id<1> g) {
            float b = 0;
            for (size_t r = g[0] * 64; r < sycl::min(g[0] * 64 + 64, m); ++r) b += rows[r] * rows[r];
            bpart[g] = b;
        });
        std::vector<float> hb(mseg);
        {
            ggml_sycl_int8_host_scope host_t(ggml_sycl_int8_host().drift_us);
            q.memcpy(hb.data(), bpart, sizeof(float) * mseg).wait();
        }
        double b = 0;
        for (float v : hb) b += v;
        st->cal_rowmax2 = b * 127.0 * 127.0 / (double) m;
        st->cal_w2      = w2F;
    }
    GGML_SYCL_INT8_STAGE(5);
    // 6. activations -> s8 with one scale per column
    q.parallel_for(sycl::range<2>(n, k / 4), [=](sycl::id<2> id) {
        const float ci = cinvx[id[0]]; const size_t i = id[0] * k + id[1] * 4;
        const sycl::vec<float, 4> v = *reinterpret_cast<const sycl::vec<float, 4> *>(x + i), si = *reinterpret_cast<const sycl::vec<float, 4> *>(sinv + id[1] * 4);
        sycl::vec<int8_t, 4> o;
        for (int l = 0; l < 4; ++l) o[l] = (int8_t) sycl::clamp(sycl::rint(v[l] * si[l] * ci), -127.0f, 127.0f);
        *reinterpret_cast<sycl::vec<int8_t, 4> *>(x8 + i) = o;
    });
    GGML_SYCL_INT8_STAGE(6);
    // 7. s8 x s8 -> f32 with the row and column scales applied inside the primitive
    using dt = dnnl::memory::data_type;
    auto strm = ctx.stream_dnnl(stream);
    auto eng = ctx.engine_dnnl(stream);
    const dnnl::memory::desc a_md({ 1, (dnnl_dim_t) m, (dnnl_dim_t) k }, dt::s8, dnnl::memory::dims{ (dnnl_dim_t) (m * k), (dnnl_dim_t) k, 1 });
    const dnnl::memory::desc b_md({ 1, (dnnl_dim_t) k, (dnnl_dim_t) n }, dt::s8, dnnl::memory::dims{ (dnnl_dim_t) (n * k), 1, (dnnl_dim_t) k });
    const dnnl::memory::desc c_md({ 1, (dnnl_dim_t) m, (dnnl_dim_t) n }, dt::f32, dnnl::memory::dims{ (dnnl_dim_t) (m * n), 1, (dnnl_dim_t) m });
    const dnnl::memory::desc rs_md({ 1, (dnnl_dim_t) m, 1 }, dt::f32, dnnl::memory::format_tag::abc);
    const dnnl::memory::desc cs_md({ 1, 1, (dnnl_dim_t) n }, dt::f32, dnnl::memory::format_tag::abc);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_mul, rs_md);
    po.append_binary(dnnl::algorithm::binary_mul, cs_md);
    attr.set_post_ops(po);
    const auto pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    const auto scratchpad_md = pd.scratchpad_desc();
    ggml_sycl_pool_alloc<uint8_t> scratchpad(ctx.pool());
    void * scratchpad_ptr = scratchpad_md.get_size() > 0 ? scratchpad.alloc(scratchpad_md.get_size()) : nullptr;
    std::unordered_map<int, dnnl::memory> args = {
        { DNNL_ARG_SRC, dnnl::memory(a_md, eng, w8) },
        { DNNL_ARG_WEIGHTS, dnnl::memory(b_md, eng, x8) },
        { DNNL_ARG_DST, dnnl::memory(c_md, eng, out) },
        { DNNL_ARG_SCRATCHPAD, dnnl::memory(scratchpad_md, eng, scratchpad_ptr) },
        { DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, dnnl::memory(rs_md, eng, rows) },
        { DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, dnnl::memory(cs_md, eng, (void *) cols) },
    };
    const dnnl::matmul prim(pd);
    GGML_SYCL_INT8_STAGE(7);
    prim.execute(strm, args);
    GGML_SYCL_INT8_STAGE(8);
    return true;
}

#endif // GGML_SYCL_DNNL
#endif // GGML_SYCL_INT8_GEMM_HPP
