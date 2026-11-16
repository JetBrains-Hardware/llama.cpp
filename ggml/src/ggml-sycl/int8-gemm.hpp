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
//   GGML_SYCL_INT8_GEMM=0            turn the path off
//   GGML_SYCL_INT8_GEMM_MIN_COLS=<n> smallest batch that takes it (default 512)
//   GGML_SYCL_INT8_GEMM_STATS=1      print the number of calls at exit (2: also per-stage times, with waits)
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
    float   max_ratio = 1e9f;
    int64_t calls    = 0;
};

static ggml_sycl_int8_gemm_cfg & ggml_sycl_int8_gemm_config() {
    static ggml_sycl_int8_gemm_cfg cfg = [] {
        ggml_sycl_int8_gemm_cfg c;
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM")) c.enabled = atoi(e) != 0;
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM_MIN_COLS")) c.min_cols = std::max(1, atoi(e));
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM_MIN_RATIO")) c.min_ratio = (float) atof(e);
        if (const char * e = getenv("GGML_SYCL_INT8_GEMM_MAX_RATIO")) c.max_ratio = (float) atof(e);
        if (getenv("GGML_SYCL_INT8_GEMM_STATS")) {
            atexit([] { fprintf(stderr, "ggml-sycl int8 GEMM calls: %lld\n", (long long) ggml_sycl_int8_gemm_config().calls); });
        }
        return c;
    }();
    return cfg;
}

// stage timers, only with GGML_SYCL_INT8_GEMM_STATS=2 (each stage then waits for the queue)
struct ggml_sycl_int8_gemm_timer {
    static constexpr int NSTAGE = 8;
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
                static const char * names[ggml_sycl_int8_gemm_timer::NSTAGE] = { "alloc+colmax", "x rms, s", "row max", "weights->s8", "col max", "x->s8", "pd create", "gemm" };
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

// max over the rows of |W[:, d]|, computed once per weight tensor and kept in its extra
static const float * ggml_sycl_int8_gemm_wcolmax(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, ggml_tensor * dst,
                                                 const char * w, int64_t m, int64_t k, const queue_ptr & stream) {
    auto * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (extra->int8_wcolmax) {
        return extra->int8_wcolmax.get();
    }
    float * cm = sycl::malloc_device<float>(k, *stream);
    GGML_ASSERT(cm != nullptr);
    const sycl::context sctx = stream->get_context();
    extra->int8_wcolmax = std::shared_ptr<float>(cm, [sctx](float * p) { sycl::free(p, sctx); });

    ggml_sycl_pool_alloc<sycl::half> w16(ctx.pool(), m * k);
    const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
    GGML_ASSERT(to_fp16_sycl != nullptr);
    to_fp16_sycl(w, w16.get(), m * k, stream);
    const sycl::half * wp = w16.get();
    const size_t mm = m, kk = k;
    // 64 rows per work-item along a column keeps the strided reads short; a second pass folds the partial maxima
    const size_t nseg = (mm + 63) / 64;
    ggml_sycl_pool_alloc<float> part(ctx.pool(), nseg * kk);
    float * pp = part.get();
    stream->parallel_for(sycl::range<2>(nseg, kk), [=](sycl::id<2> id) {
        const size_t r0 = id[0] * 64, r1 = sycl::min(r0 + 64, mm);
        float mx = 0;
        for (size_t r = r0; r < r1; ++r) mx = sycl::fmax(mx, sycl::fabs((float) wp[r * kk + id[1]]));
        pp[id[0] * kk + id[1]] = mx;
    });
    stream->parallel_for(sycl::range<1>(kk), [=](sycl::id<1> d) {
        float mx = 0;
        for (size_t sg = 0; sg < nseg; ++sg) mx = sycl::fmax(mx, pp[sg * kk + d[0]]);
        cm[d] = mx;
    });
    stream->wait();  // one time per tensor; the temporaries go back to the pool
    return cm;
}

// dst (m x n, column-major, f32) = W (reordered q4_K / q6_K, m x k) * x (k x n, column-major, f32)
static bool ggml_sycl_int8_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, ggml_tensor * dst, const char * w, const float * x,
                                float * out, int64_t m_, int64_t n_, int64_t k_, const queue_ptr & stream) {
    const size_t m = m_, n = n_, k = k_;
    auto & tm = ggml_sycl_int8_gemm_timers();
    if (tm.on) { stream->wait(); tm.t0 = std::chrono::steady_clock::now(); tm.n++; }
    const size_t nb = m * k / QK_K, bpr = k / QK_K;
    const bool   q4 = src0->type == GGML_TYPE_Q4_K;
    const float * wcm = ggml_sycl_int8_gemm_wcolmax(ctx, src0, dst, w, m_, k_, stream);
    ggml_sycl_int8_gemm_config().calls++;

    ggml_sycl_pool_alloc<int8_t> w8_alloc(ctx.pool(), m * k);
    ggml_sycl_pool_alloc<int8_t> x8_alloc(ctx.pool(), k * n);
    // small arrays in one allocation: s[k], 1/s[k], rowq[m], rows[m], cinv[n], cols[n], cmax[n * k/64], bmax[4 * nb], xss[n/32 * k]
    const size_t kc = k / 64;
    ggml_sycl_pool_alloc<float> f_alloc(ctx.pool(), 2 * k + 2 * m + 2 * n + n * kc + 4 * nb + ((n + 31) / 32) * k);
    int8_t * w8 = w8_alloc.get();
    int8_t * x8 = x8_alloc.get();
    float * sd = f_alloc.get(); float * sinv = sd + k; float * rowq = sinv + k; float * rows = rowq + m;
    float * cinv = rows + m; float * cols = cinv + n; float * cmax = cols + n; float * bmax = cmax + n * kc; float * xss = bmax + 4 * nb;
    sycl::queue & q = *stream;
    GGML_SYCL_INT8_STAGE(0);

    // 1. per-dimension equalization factor from the rms of the dimension over the batch and the weight column max
    //    (sums of squares over 32 columns x 16 dimensions per work-item: contiguous reads; a strided per-dimension
    //    loop over the batch measured 0.8 ms per call)
    //    Every column (token) is first normalized by its own mean square, so that each token weighs the same:
    //    a plain rms over the batch is dominated by the few tokens with a very large norm.
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
        const float xr = sycl::sqrt(ss / (float) n), wm = wcm[d];
        float sv = (xr > 0 && wm > 0) ? sycl::sqrt(xr / wm) : 1.0f;
        if (!(sv > 1e-18f && sv < 1e18f)) sv = 1.0f;
        sd[d] = sv; sinv[d] = 1.0f / sv;
    });
    GGML_SYCL_INT8_STAGE(1);
    // 2. row scales: exact max of |W[r, d] * s[d]| over the row (partial maxima per 64 weights, then per row).
    //    A bound from the block scales alone is cheaper but costs about one bit: measured NMSE 2-4x higher.
    const uint8_t * base = (const uint8_t *) w;
    if (q4) {
        const uint8_t * scp = base + nb * (QK_K / 2); const ggml_half * dmp = (const ggml_half *) (base + nb * (QK_K / 2) + nb * K_SCALE_SIZE);
        const uint8_t * qs0 = base;
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
        GGML_SYCL_INT8_STAGE(2);
        // 3. weights -> s8: 32 work-items per block of 256, 8 weights each
        const uint8_t * qs = base;
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
        GGML_SYCL_INT8_STAGE(2);
        // 64 work-items per block of 256, 4 weights each (same decomposition as dequantize_block_q6_K_reorder)
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
    GGML_SYCL_INT8_STAGE(3);
    // 4. activations: column absmax of x / s, then s8 with one scale per column
    q.parallel_for(sycl::range<2>(n, kc), [=](sycl::id<2> id) {
        const float * p = x + id[0] * k + id[1] * 64; const float * si = sinv + id[1] * 64;
        float mx = 0;
        for (int j = 0; j < 64; ++j) mx = sycl::fmax(mx, sycl::fabs(p[j] * si[j]));
        cmax[id[0] * kc + id[1]] = mx;
    });
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> c) {
        float mx = 1e-30f;
        for (size_t j = 0; j < kc; ++j) mx = sycl::fmax(mx, cmax[c[0] * kc + j]);
        cols[c] = mx / 127.0f; cinv[c] = 127.0f / mx;
    });
    GGML_SYCL_INT8_STAGE(4);
    q.parallel_for(sycl::range<2>(n, k / 4), [=](sycl::id<2> id) {
        const float ci = cinv[id[0]]; const size_t i = id[0] * k + id[1] * 4;
        const sycl::vec<float, 4> v = *reinterpret_cast<const sycl::vec<float, 4> *>(x + i), si = *reinterpret_cast<const sycl::vec<float, 4> *>(sinv + id[1] * 4);
        sycl::vec<int8_t, 4> o;
        for (int l = 0; l < 4; ++l) o[l] = (int8_t) sycl::clamp(sycl::rint(v[l] * si[l] * ci), -127.0f, 127.0f);
        *reinterpret_cast<sycl::vec<int8_t, 4> *>(x8 + i) = o;
    });
    GGML_SYCL_INT8_STAGE(5);
    // 5. s8 x s8 -> f32 with the row and column scales applied inside the primitive
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
        { DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, dnnl::memory(cs_md, eng, cols) },
    };
    const dnnl::matmul prim(pd);
    GGML_SYCL_INT8_STAGE(6);
    prim.execute(strm, args);
    GGML_SYCL_INT8_STAGE(7);
    return true;
}

#endif // GGML_SYCL_DNNL
#endif // GGML_SYCL_INT8_GEMM_HPP
