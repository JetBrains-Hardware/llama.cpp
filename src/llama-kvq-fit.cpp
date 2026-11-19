#include "llama-kvq-fit.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-kvq-pack.h"
#include "llama-kvq-store.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#elif defined(__linux__)
    #include <sys/resource.h>
#endif

static constexpr int64_t KVQ_D   = llama_kvq_store::D;
static constexpr size_t  KVQ_ROW = llama_kvq_store::ROW;
static constexpr int64_t KVQ_NP  = KVQ_D + 5; // rows of the parameters of a task: fwd [D], mu, step, bits, offsets, den

// the pool runs behind the decode thread and the prefetch worker
static void kvq_fit_lower_priority() {
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#elif defined(__linux__)
    // per thread on Linux
    setpriority(PRIO_PROCESS, 0, 10);
#endif
}

static double kvq_fit_now() {
    return ggml_time_us()*1e-6;
}

enum kvq_fit_stage {
    KVQ_STAGE_FIT,    // phase 1 queued, fit tasks
    KVQ_STAGE_BITS,   // the V fit tasks are done: phase 2 and the V bits tasks queued (per-chunk V), the K fit tasks may still run
    KVQ_STAGE_READY,  // all phases queued
    KVQ_STAGE_FAILED,
};

// task i = kv*n_pair + layer index*G + kv head, kv = 0 for K, 1 for V; the host arrays are per (layer index, K or V, kv head)
struct llama_kvq_fitter::job {
    int32_t r0    = 0; // rows [r0, r1), the table set of the chunk slots r0/C .. (r1 - 1)/C
    int32_t r1    = 0;
    int     stage = KVQ_STAGE_FIT; // decode thread

    int32_t n() const { return r1 - r0; }

    std::vector<float> gram; // [n_layer][G][D*D]
    std::vector<float> gsum; // [n_layer][n_gate], per-chunk V only

    std::atomic<bool> cancel{false};
    std::atomic<bool> failed{false};

    // pool mutex
    int32_t n_left[2] = {}; // tasks not finished: K (fit), V (fit, then bits)
    int32_t n_run  = 0; // tasks running
    double  t_cpu  = 0.0;

    double t_snap = 0.0;
    double t_take = 0.0;

    // the end of phase 1 (and later of phase 3), of phase 2
    ggml_backend_event_ptr ev[2];

    // pinned host memory
    ggml_backend_buffer_ptr buf;
    float       * hstat = nullptr; // per (layer index, K or V): cov [G][D*D], mu [G][D]
    float       * hpar  = nullptr; // per (layer index, K or V, kv head): [KVQ_NP][D]
    float       * hsig  = nullptr; // per (layer index, kv head): V scale of the shape [D]
    int8_t      * htinv = nullptr; // per task [D*D]
    float       * htaux = nullptr; // per task [3*D]

    // per-chunk V: the tables between the fit and the bits tasks, per (layer index, kv head)
    std::vector<llama_kvq_table> vt;
};

llama_kvq_fitter::llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams,
        const std::vector<const ggml_tensor *> & wo, int32_t n_head, int32_t min_first) :
    store(store), fparams(fparams), min_first(min_first), layer_ids(store.layer_ids()), n_head(n_head), wo(wo) {
    const int32_t G     = store.n_head_kv();
    const int32_t C     = store.n_chunk_rows();
    const int32_t n_chk = store.n_chunks();

    if (store.backend() == nullptr) {
        throw std::runtime_error("the fitter needs the backend of the device");
    }
    if (fparams.rem != LLAMA_KVQ_REM_OWN && fparams.rem != LLAMA_KVQ_REM_MERGE) {
        throw std::runtime_error(format("unknown policy %d for the rows after the last complete chunk", fparams.rem));
    }
    // merge: a group of the last complete chunk and up to C - 1 rows after it
    n_stage = std::min<int32_t>(store.n_rows(), fparams.rem == LLAMA_KVQ_REM_MERGE ? 2*C - 1 : C);
    slot_fit.assign(n_chk, range());

    n_rep = n_head/G;
    if (n_rep < 1 || n_rep*G != n_head || wo.size() != layer_ids.size()) {
        throw std::runtime_error("per-chunk V needs the o_proj weight of every attention layer");
    }
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        const ggml_tensor * w  = wo[li];
        const ggml_tensor * ga = store.get_layer(layer_ids[li]).gacc;
        if (w == nullptr || ga == nullptr || ggml_n_dims(w) != 2 || w->ne[0] != (int64_t) n_head*KVQ_D || ga->ne[0] != w->ne[0] ||
                (w->type != GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float == nullptr) ||
                (n_rep*KVQ_D) % ggml_blck_size(w->type) != 0) {
            throw std::runtime_error(format("per-chunk V: unsupported o_proj weight or no gate statistics in layer %d", layer_ids[li]));
        }
    }
    wo_gram.resize(layer_ids.size()*G);
    wo_once.reset(new std::once_flag[layer_ids.size()*G]);

    graph_meta.resize(ggml_tensor_overhead()*128 + ggml_graph_overhead());
    dev_alloc();
    compute_wo_gram_dev();

    for (int i = 0; i < fparams.n_threads; ++i) {
        threads.emplace_back([this]() { loop(); });
    }
    const double mib_dev = (ggml_backend_buffer_get_size(buf_dev.get()) + ggml_gallocr_get_buffer_size(galloc.get(), 0))/1048576.0;
    const std::string rem = fparams.rem == LLAMA_KVQ_REM_MERGE ? "merged into the last complete chunk" : format("own chunk of at least %d rows", fparams.min_rows);
    LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits and V %d bits fitted per chunk on the device, %.1f MiB, %d CPU threads, rows after the last complete chunk: %s\n", __func__,
            C, n_chk, fparams.bits_k, fparams.bits_v, mib_dev, fparams.n_threads, rem.c_str());
}

llama_kvq_fitter::~llama_kvq_fitter() {
    {
        std::lock_guard<std::mutex> lk(m);
        stop = true;
        for (auto & jb : jobs) {
            jb->cancel = true;
        }
    }
    cv_work.notify_all();
    for (auto & t : threads) {
        t.join();
    }
    // queued phases use the buffers; the backend may be freed already, the events are not
    for (auto & jb : jobs) {
        for (auto & ev : jb->ev) {
            ggml_backend_event_synchronize(ev.get());
        }
    }
}

// device state, then the largest graphs once: the ops must run on the device, and the allocator does not grow later
void llama_kvq_fitter::dev_alloc() {
    if (buf_dev) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const int32_t G  = store.n_head_kv();

    ggml_init_params ip = { 5*ggml_tensor_overhead(), nullptr, true };
    ctx_dev.reset(ggml_init(ip));
    if (!ctx_dev) {
        throw std::runtime_error("failed to create the fitter context");
    }
    idx   = ggml_new_tensor_2d(ctx_dev.get(), GGML_TYPE_I32, n_stage, G);
    par   = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_F32, KVQ_D, KVQ_NP, 2*G);
    stage = ggml_new_tensor_4d(ctx_dev.get(), GGML_TYPE_Q8_0, KVQ_D, 2, n_stage, G);
    cst[0] = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_I32, store.n_words(false), n_stage, G);
    cst[1] = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_I32, store.n_words(true) + 1, n_stage, G);
    ggml_set_name(idx,    "kvq_fit_idx");
    ggml_set_name(par,    "kvq_fit_par");
    ggml_set_name(stage,  "kvq_fit_stage");
    ggml_set_name(cst[0], "kvq_fit_kc");
    ggml_set_name(cst[1], "kvq_fit_vc");
    buf_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev.get(), store.dev_buft()));
    if (!buf_dev) {
        throw std::runtime_error("failed to allocate the fitter device buffer");
    }
    ggml_backend_buffer_clear(buf_dev.get(), 0);

    std::vector<int32_t> ids((size_t) n_stage*G);
    for (size_t j = 0; j < ids.size(); ++j) {
        ids[j] = (int32_t) (j % n_stage);
    }
    ggml_backend_tensor_set(idx, ids.data(), 0, ggml_nbytes(idx));

    galloc.reset(ggml_gallocr_new(store.dev_buft()));

    job jb;
    jb.r0 = 0;
    jb.r1 = n_stage;
    for (int phase = 1; phase <= 3; ++phase) {
        ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_cgraph * gf = build_phase(jb, phase, 0, ctx.get());
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_NONE && !ggml_backend_supports_op(store.backend(), node)) {
                throw std::runtime_error(format("the device does not run the fit op %s (%s)", ggml_op_desc(node), node->name));
            }
        }
        if (!ggml_gallocr_reserve(galloc.get(), gf)) {
            throw std::runtime_error("failed to allocate the fitter graphs");
        }
    }
    LLAMA_LOG_INFO("%s: %.1f MiB in %.2f ms\n", __func__,
            (ggml_backend_buffer_get_size(buf_dev.get()) + ggml_gallocr_get_buffer_size(galloc.get(), 0))/1048576.0, (ggml_time_us() - t0)/1000.0);
}

void llama_kvq_fitter::dev_free() {
    if (!buf_dev) {
        return;
    }
    GGML_ASSERT(jobs_empty());
    const int64_t t0 = ggml_time_us();
    store.synchronize();
    galloc.reset();
    buf_dev.reset();
    ctx_dev.reset();
    idx     = nullptr;
    par     = nullptr;
    stage   = nullptr;
    cst[0]  = nullptr;
    cst[1]  = nullptr;
    LLAMA_LOG_INFO("%s: the device stage and the graph allocator are freed in %.2f ms\n", __func__, (ggml_time_us() - t0)/1000.0);
}

int32_t llama_kvq_fitter::n_pair() const {
    return (int32_t) layer_ids.size()*store.n_head_kv();
}

void llama_kvq_fitter::loop() {
    kvq_fit_lower_priority();
    std::unique_lock<std::mutex> lk(m);
    while (true) {
        cv_work.wait(lk, [&]() { return stop || !tasks.empty(); });
        if (stop) {
            return;
        }
        job * jb = tasks.front().first;
        const int32_t i = tasks.front().second;
        tasks.pop_front();
        jb->n_run++;
        lk.unlock();
        double t_task = 0.0;
        if (!jb->cancel && !jb->failed) {
            // the wait for the device phase does not count
            ggml_backend_event_synchronize(jb->ev[i >= 2*n_pair() ? 1 : 0].get());
            const double t0 = kvq_fit_now();
            run(*jb, i);
            t_task = kvq_fit_now() - t0;
        }
        lk.lock();
        jb->t_cpu += t_task;
        jb->n_run--;
        jb->n_left[i >= n_pair() ? 1 : 0]--;
        cv_done.notify_all();
    }
}

// pool thread, once per (layer index, kv head) p: Wo_r^T Wo_r for the query heads r of kv head g, Wo = o_proj weight [n_out][n_in]
void llama_kvq_fitter::compute_wo_gram(int32_t p) {
    const int32_t       G = store.n_head_kv();
    const int32_t       g = p % G;
    const ggml_tensor * w = wo[p / G];

    const int64_t n_out = w->ne[1];
    const int64_t nc    = (int64_t) n_rep*KVQ_D; // the columns of kv head g
    const size_t  off   = ggml_row_size(w->type, g*nc);
    const int64_t blk   = 256;

    std::vector<uint8_t> raw(ggml_nbytes(w));
    ggml_backend_tensor_get(w, raw.data(), 0, raw.size());

    std::vector<double> gram((size_t) n_rep*KVQ_D*KVQ_D, 0.0);
    std::vector<float>  x(blk*nc);
    for (int64_t o0 = 0; o0 < n_out; o0 += blk) {
        const int64_t nb = std::min(blk, n_out - o0);
        for (int64_t o = 0; o < nb; ++o) {
            const uint8_t * src = raw.data() + (o0 + o)*w->nb[1] + off;
            if (w->type == GGML_TYPE_F32) {
                memcpy(x.data() + o*nc, src, nc*sizeof(float));
            } else {
                ggml_get_type_traits(w->type)->to_float(src, x.data() + o*nc, nc);
            }
        }
        for (int32_t r = 0; r < n_rep; ++r) {
            llama_kvq_gram(x.data() + r*KVQ_D, nb, KVQ_D, nc, gram.data() + (size_t) r*KVQ_D*KVQ_D, 1);
        }
    }
    wo_gram[p].resize(gram.size());
    for (size_t j = 0; j < gram.size(); ++j) {
        wo_gram[p][j] = (float) gram[j];
    }
}

// constructor: Wo_r^T Wo_r of every (layer index, kv head) on the device, the covariance op without the mean on the dequantized
// columns; a weight that is not on the device gets it on the CPU at the first chunk (compute_wo_gram)
void llama_kvq_fitter::compute_wo_gram_dev() {
    const int32_t  G       = store.n_head_kv();
    ggml_backend_t backend = store.backend();

    wo_dev.assign(layer_ids.size()*G, 0);
    int64_t n_out = 0;
    for (const ggml_tensor * w : wo) {
        n_out = std::max(n_out, w->ne[1]);
    }

    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx_ids(ggml_init(ip));
    ggml_tensor * ids = ggml_new_tensor_2d(ctx_ids.get(), GGML_TYPE_I32, n_out, n_rep);
    ggml_backend_buffer_ptr buf_ids(ggml_backend_alloc_ctx_tensors_from_buft(ctx_ids.get(), store.dev_buft()));
    if (!buf_ids) {
        return;
    }
    std::vector<int32_t> v((size_t) n_out*n_rep);
    for (size_t j = 0; j < v.size(); ++j) {
        v[j] = (int32_t) (j % n_out);
    }
    ggml_backend_tensor_set(ids, v.data(), 0, ggml_nbytes(ids));

    const double t0 = kvq_fit_now();
    int n_dev = 0;
    for (int32_t p = 0; p < (int32_t) wo_dev.size(); ++p) {
        ggml_tensor * w = const_cast<ggml_tensor *>(wo[p / G]);
        if (w->buffer == nullptr || ggml_backend_buffer_get_type(w->buffer) != store.dev_buft()) {
            continue;
        }
        ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);

        // [D, n_out, n_rep]: the columns of query head (p % G)*n_rep + r of each row of Wo
        ggml_tensor * wv  = ggml_view_3d(ctx.get(), w, KVQ_D, w->ne[1], n_rep, w->nb[1], ggml_row_size(w->type, KVQ_D),
                ggml_row_size(w->type, (int64_t) (p % G)*n_rep*KVQ_D));
        ggml_tensor * iv  = ggml_view_2d(ctx.get(), ids, w->ne[1], n_rep, ids->nb[1], 0);
        ggml_tensor * x   = ggml_get_rows(ctx.get(), wv, iv);
        ggml_tensor * cov = ggml_kvq_fit(ctx.get(), GGML_KVQ_FIT_COV, x, nullptr, nullptr);
        ggml_set_output(cov);
        ggml_build_forward_expand(gf, cov);
        if (!ggml_backend_supports_op(backend, x) || !ggml_backend_supports_op(backend, cov) || !ggml_gallocr_alloc_graph(galloc.get(), gf) ||
                ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            continue;
        }
        wo_gram[p].resize((size_t) n_rep*KVQ_D*KVQ_D);
        ggml_backend_tensor_get(cov, wo_gram[p].data(), 0, ggml_nbytes(cov));
        wo_dev[p] = 1;
        n_dev++;
    }
    LLAMA_LOG_INFO("%s: o_proj grams of %d of %zu (layer, kv head) on the device in %.2f s\n", __func__, n_dev, wo_dev.size(), kvq_fit_now() - t0);
}

// V metric of a chunk: sum over r of D_r Wo_r^T Wo_r D_r, D_r = diag of the chunk's mean sigmoid(gate) of query head g*n_rep + r
void llama_kvq_fitter::metric_v(const job & jb, int32_t p, std::vector<double> & m) {
    const int32_t G      = store.n_head_kv();
    const int32_t li     = p / G;
    const int32_t g      = p % G;
    const int64_t n_gate = (int64_t) n_head*KVQ_D;

    if (!wo_dev[p]) {
        std::call_once(wo_once[p], [&]() { compute_wo_gram(p); });
    }

    std::vector<double> gbar((size_t) n_rep*KVQ_D);
    const float * gs = jb.gsum.data() + li*n_gate + (int64_t) g*n_rep*KVQ_D;
    double tot = 0.0;
    for (size_t j = 0; j < gbar.size(); ++j) {
        gbar[j] = (double) gs[j]/jb.n();
        tot += gbar[j];
    }
    if (!(tot > 0.0)) {
        // no gate statistics (e.g. after a removal), plain o_proj metric
        LLAMA_LOG_WARN("%s: rows [%d, %d), layer %d, kv head %d: no gate statistics, the V table is fitted without the gate\n", __func__, jb.r0, jb.r1, layer_ids[li], g);
        std::fill(gbar.begin(), gbar.end(), 1.0);
    }

    m.assign(KVQ_D*KVQ_D, 0.0);
    for (int32_t r = 0; r < n_rep; ++r) {
        const double * d  = gbar.data() + (size_t) r*KVQ_D;
        const float  * wg = wo_gram[p].data() + (size_t) r*KVQ_D*KVQ_D;
        for (int64_t i = 0; i < KVQ_D; ++i) {
            for (int64_t j = 0; j < KVQ_D; ++j) {
                m[i*KVQ_D + j] += d[i]*wg[i*KVQ_D + j]*d[j];
            }
        }
    }
}

// the parameters of a task for the device: fwd, mu, den; with bits also step, bits and the offsets of the packed row
static void kvq_fit_set_par(float * par, const llama_kvq_table & t) {
    const int64_t n = KVQ_D;
    memcpy(par, t.fwd.data(), n*n*sizeof(float));
    memcpy(par + n*n, t.mu.data(), n*sizeof(float));
    float * step = par + (n + 1)*n;
    float * bits = par + (n + 2)*n;
    float * offs = par + (n + 3)*n;
    std::fill(step, step + 3*n, 0.0f);
    if (!t.bits.empty()) {
        const std::vector<int32_t> perm = llama_kvq_pack_perm(t);
        int32_t o = 0;
        for (int64_t c = 0; c < n; ++c) {
            const int32_t pc = perm[c];
            step[pc] = t.bits[pc] ? t.step[pc] : 0.0f;
            bits[pc] = (float) t.bits[pc];
            offs[pc] = (float) o;
            o += t.bits[pc];
        }
    }
    std::fill(par + (n + 4)*n, par + (n + 5)*n, 0.0f);
    par[(n + 4)*n] = t.is_v ? t.gain_den : 0.0f;
}

void llama_kvq_fitter::run(job & jb, int32_t t) {
    const int32_t n_task = 2*(int32_t) layer_ids.size()*store.n_head_kv();
    if (t < n_task) {
        fit_task(jb, t);
    } else {
        bits_task(jb, n_task/2 + t - n_task);
    }
}

// pool thread, after phase 1: the table of K or V of one (layer, kv head) from the moments of the rows; K: also bits and the packed table
void llama_kvq_fitter::fit_task(job & jb, int32_t i) {
    const int32_t G      = store.n_head_kv();
    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    const bool    is_v   = i >= n_pair;
    const int32_t p      = i % n_pair;
    const int32_t li     = p / G;
    const int32_t g      = p % G;
    const int32_t il     = layer_ids[li];
    const int64_t n_tok  = jb.n();
    const int32_t blk    = 2*li + (is_v ? 1 : 0);
    float * par = jb.hpar + ((size_t) blk*G + g)*KVQ_NP*KVQ_D;

    std::vector<double> metric;
    if (is_v) {
        metric_v(jb, p, metric);
    } else {
        // K metric: second moment of the queries of the chunk, the fit does not depend on its scale
        metric.resize(KVQ_D*KVQ_D);
        const float * gr = jb.gram.data() + (size_t) p*KVQ_D*KVQ_D;
        double tr = 0.0;
        for (int64_t j = 0; j < KVQ_D*KVQ_D; ++j) {
            metric[j] = gr[j];
        }
        for (int64_t j = 0; j < KVQ_D; ++j) {
            tr += metric[j*KVQ_D + j];
        }
        if (!(tr > 0.0)) {
            // no queries (e.g. after a removal), plain MSE
            LLAMA_LOG_WARN("%s: rows [%d, %d), layer %d, kv head %d: no query statistics, the K table is fitted for the MSE\n", __func__, jb.r0, jb.r1, il, g);
            std::fill(metric.begin(), metric.end(), 0.0);
            for (int64_t j = 0; j < KVQ_D; ++j) {
                metric[j*KVQ_D + j] = 1.0;
            }
        }
    }
    if (jb.cancel) {
        return;
    }

    const float * st = jb.hstat + (size_t) blk*G*(KVQ_D*KVQ_D + KVQ_D);
    const float * cf = st + (size_t) g*KVQ_D*KVQ_D;
    const float * mu = st + (size_t) G*KVQ_D*KVQ_D + g*KVQ_D;
    std::vector<double> cov(cf, cf + KVQ_D*KVQ_D);

    llama_kvq_table tab;
    bool ok = llama_kvq_fit_cov(tab, mu, cov.data(), n_tok, KVQ_D, metric.data(), is_v);
    if (ok && !is_v) {
        llama_kvq_fit_bits(tab, (float) fparams.bits_k);
        ok = llama_kvq_pack_words(tab) <= store.n_words(false);
    }
    if (!ok) {
        LLAMA_LOG_ERROR("%s: rows [%d, %d): the %s fit of layer %d, kv head %d failed\n", __func__, jb.r0, jb.r1, is_v ? "V" : "K", il, g);
        jb.failed = true;
        return;
    }
    kvq_fit_set_par(par, tab);
    if (is_v) {
        jb.vt[p] = std::move(tab);
    } else {
        llama_kvq_pack_table(tab, jb.htinv + (size_t) i*KVQ_D*KVQ_D, jb.htaux + (size_t) i*3*KVQ_D);
    }
}

// pool thread, after phase 2: V bits and steps from the scale of the shape, the packed table
void llama_kvq_fitter::bits_task(job & jb, int32_t i) {
    const int32_t G      = store.n_head_kv();
    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    const int32_t p      = i - n_pair;
    const int32_t li     = p / G;
    const int32_t g      = p % G;

    llama_kvq_table & tab = jb.vt[p];
    memcpy(tab.sigma.data(), jb.hsig + (size_t) p*KVQ_D, KVQ_D*sizeof(float));
    llama_kvq_fit_bits(tab, (float) fparams.bits_v);
    if (llama_kvq_pack_words(tab) > store.n_words(true)) {
        LLAMA_LOG_ERROR("%s: rows [%d, %d): the V fit of layer %d, kv head %d failed\n", __func__, jb.r0, jb.r1, layer_ids[li], g);
        jb.failed = true;
        return;
    }
    kvq_fit_set_par(jb.hpar + ((size_t) (2*li + 1)*G + g)*KVQ_NP*KVQ_D, tab);
    llama_kvq_pack_table(tab, jb.htinv + (size_t) i*KVQ_D*KVQ_D, jb.htaux + (size_t) i*3*KVQ_D);
    tab = {};
}

// the graph of one phase of one attention layer, x = the rows of the job in the stage (from the host store):
//   1: K and V rows -> mean, covariance (outputs)
//   2: V (per-chunk V): rows -> centered -> rotated -> scale of the shape (output)
//   3: K and V: rows -> centered -> rotated -> codes in kc and vc
ggml_cgraph * llama_kvq_fitter::build_phase(const job & jb, int phase, int32_t li, ggml_context * ctx) {
    const int32_t il   = layer_ids[li];
    const int64_t G    = store.n_head_kv();
    const int64_t n    = jb.n();
    const auto &  lt   = store.get_layer(il);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    for (int kv = phase == 2 ? 1 : 0; kv < 2; ++kv) {
        // the rows of the stage [D, n_stage, G], K or V
        ggml_tensor * sv = ggml_view_3d(ctx, stage, KVQ_D, n_stage, G, stage->nb[2], stage->nb[3], kv*stage->nb[1]);
        ggml_tensor * iv = ggml_view_2d(ctx, idx, n, G, idx->nb[1], 0);
        ggml_tensor * x  = ggml_get_rows(ctx, sv, iv);

        if (phase == 1) {
            ggml_tensor * mu = ggml_kvq_fit(ctx, GGML_KVQ_FIT_MEAN, x, nullptr, nullptr);
            ggml_set_output(mu);
            ggml_tensor * cov = ggml_kvq_fit(ctx, GGML_KVQ_FIT_COV, ggml_sub(ctx, x, mu), nullptr, nullptr);
            ggml_set_output(cov);
            ggml_format_name(mu,  "kvq_fit_mu%d",  kv);
            ggml_format_name(cov, "kvq_fit_cov%d", kv);
            ggml_build_forward_expand(gf, mu);
            ggml_build_forward_expand(gf, cov);
            continue;
        }

        const size_t  o  = kv*G*par->nb[2];
        ggml_tensor * mu = ggml_view_3d(ctx, par, KVQ_D, 1, G, par->nb[1], par->nb[2], o + KVQ_D*par->nb[1]);
        ggml_tensor * fw = ggml_view_3d(ctx, par, KVQ_D, KVQ_D, G, par->nb[1], par->nb[2], o);
        ggml_tensor * pq = ggml_view_3d(ctx, par, KVQ_D, 4, G, par->nb[1], par->nb[2], o + (KVQ_D + 1)*par->nb[1]);
        ggml_tensor * y  = ggml_kvq_fit(ctx, GGML_KVQ_FIT_ROT, ggml_sub(ctx, x, mu), fw, nullptr);

        if (phase == 2) {
            ggml_tensor * sig = ggml_kvq_fit(ctx, GGML_KVQ_FIT_GAIN, y, pq, nullptr);
            ggml_set_output(sig);
            ggml_set_name(sig, "kvq_fit_sig");
            ggml_build_forward_expand(gf, sig);
        } else {
            // the codes go to the host copy (queue_phase); without the device codes through the code stage
            const bool    dev  = store.codes_on_dev();
            ggml_tensor * rows = dev ? (kv ? lt.vc : lt.kc) : cst[kv];
            ggml_tensor * rv   = ggml_view_3d(ctx, rows, rows->ne[0], n, G, rows->nb[1], rows->nb[2], dev ? (size_t) jb.r0*rows->nb[1] : 0);
            ggml_build_forward_expand(gf, ggml_kvq_fit(ctx, GGML_KVQ_FIT_CODE, y, pq, rv));
        }
    }

    return gf;
}

// host store rows [r0, r1) of layer il -> stage rows [0, r1 - r0), one copy per kv head (the same layout), main stream
void llama_kvq_fitter::upload_rows(int32_t il, int32_t r0, int32_t r1) {
    if (r1 <= r0) {
        return;
    }
    const int64_t G = store.n_head_kv();
    for (int64_t g = 0; g < G; ++g) {
        const uint8_t * src = store.host_rows(il) + ((size_t) g*store.n_rows() + r0)*2*KVQ_ROW;
        ggml_backend_tensor_set_async(store.backend(), stage, src, g*stage->nb[3], (size_t) (r1 - r0)*2*KVQ_ROW);
    }
}

// decode thread: queues one phase of a job on the main stream, no wait
// every phase reads the rows from the host store (the host ring copies the rows of the ring there)
void llama_kvq_fitter::queue_phase(job & jb, int phase) {
    ggml_backend_t backend = store.backend();
    const int64_t  G       = store.n_head_kv();
    const size_t   n_par   = (size_t) G*KVQ_NP*KVQ_D; // floats per (layer index, K or V)
    const size_t   n_stat  = (size_t) G*(KVQ_D*KVQ_D + KVQ_D);

    dev_alloc();

    // host ring: the rows of the last ubatches reach the host store on the copy stream
    store.order_rows();

    for (size_t li = 0; li < layer_ids.size(); ++li) {
        const int32_t il = layer_ids[li];
        if (phase == 2) {
            ggml_backend_tensor_set_async(backend, par, jb.hpar + (2*li + 1)*n_par, G*par->nb[2], n_par*sizeof(float));
        } else if (phase == 3) {
            ggml_backend_tensor_set_async(backend, par, jb.hpar + 2*li*n_par, 0, 2*n_par*sizeof(float));
        }
        upload_rows(il, jb.r0, jb.r1);

        ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_cgraph * gf = build_phase(jb, phase, (int32_t) li, ctx.get());
        if (!ggml_gallocr_alloc_graph(galloc.get(), gf) || ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("%s: phase %d of rows [%d, %d), layer %d failed\n", __func__, phase, jb.r0, jb.r1, il);
        }

        // the outputs, before the next graph reuses the memory
        if (phase == 1) {
            for (int kv = 0; kv < 2; ++kv) {
                float * st = jb.hstat + (2*li + kv)*n_stat;
                ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, format("kvq_fit_cov%d", kv).c_str()), st, 0, G*KVQ_D*KVQ_D*sizeof(float));
                ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, format("kvq_fit_mu%d", kv).c_str()), st + G*KVQ_D*KVQ_D, 0, G*KVQ_D*sizeof(float));
            }
        } else if (phase == 2) {
            ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, "kvq_fit_sig"), jb.hsig + li*G*KVQ_D, 0, G*KVQ_D*sizeof(float));
        } else if (phase == 3) {
            // the codes to the host copy, in the layout of kc / vc; done at the synchronize of the install
            const bool dev = store.codes_on_dev();
            for (int kv = 0; kv < 2; ++kv) {
                const ggml_tensor * rows = kv ? store.get_layer(il).vc : store.get_layer(il).kc;
                ggml_tensor       * src  = dev ? const_cast<ggml_tensor *>(rows) : cst[kv];
                uint8_t           * dst  = store.host_codes(il, kv == 1);
                for (int64_t g = 0; g < G; ++g) {
                    ggml_backend_tensor_get_async(backend, src, dst + g*rows->nb[2] + (size_t) jb.r0*rows->nb[1],
                            g*src->nb[2] + (dev ? (size_t) jb.r0*rows->nb[1] : 0), (size_t) jb.n()*rows->nb[1]);
                }
            }
        }
    }
    if (phase == 3) {
        store.codes_written(jb.r1);
    }
    if (jb.ev[phase == 2 ? 1 : 0]) {
        ggml_backend_event_record(jb.ev[phase == 2 ? 1 : 0].get(), backend);
    }
}

// decode thread: the sums of the open chunk on the device, without a reset
void llama_kvq_fitter::read_acc(std::vector<float> & gram, std::vector<float> & gsum) {
    const size_t nn = (size_t) KVQ_D*KVQ_D*store.n_head_kv();
    gram.resize(nn*layer_ids.size());
    gsum.resize((size_t) n_head*KVQ_D*layer_ids.size());
    store.synchronize();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        ggml_tensor * acc = store.get_layer(layer_ids[li]).qacc;
        ggml_backend_tensor_get(acc, gram.data() + li*nn, 0, ggml_nbytes(acc));
        ggml_tensor * gacc = store.get_layer(layer_ids[li]).gacc;
        ggml_backend_tensor_get(gacc, gsum.data() + li*gacc->ne[0], 0, ggml_nbytes(gacc));
    }
}

// decode thread, n_pos positions in the sequence: the statistics of the complete chunks, the sums of the ubatches of a chunk
// are computed when it is complete, then they start again from 0 (a ubatch counts for the chunk of its first position)
void llama_kvq_fitter::snapshots(llama_pos n_pos) {
    const int32_t C = store.n_chunk_rows();
    while (!failed && (int64_t) (n_snap + 1)*C <= store.n_rows() && n_pos >= (n_snap + 1)*C) {
        std::vector<float> gram;
        std::vector<float> gsum;
        read_acc(gram, gsum);
        set_stats(nullptr, nullptr);
        acc_pos = -1;
        grams.push_back(std::move(gram));
        gsums.push_back(std::move(gsum));
        t_snaps.push_back(kvq_fit_now());
        n_snap++;
    }
}

// the layout of [0, n): the complete chunks alone, then the rows after them (own: alone, if there are min_rows of them;
// merge: one group with the last complete chunk); nothing if the coarse region would be smaller than min_first
std::vector<llama_kvq_fitter::range> llama_kvq_fitter::layout(llama_pos n) const {
    const int32_t C      = store.n_chunk_rows();
    const int32_t n_full = std::min<int32_t>(n, store.n_rows())/C;
    const bool    merge  = fparams.rem == LLAMA_KVQ_REM_MERGE && n_full > 0;
    const int32_t n_solo = merge ? n_full - 1 : n_full;

    std::vector<range> res;
    for (int32_t k = 0; k < n_solo; ++k) {
        res.push_back({ k*C, (k + 1)*C });
    }
    const int32_t r0 = n_solo*C;
    const int32_t r1 = std::min<int32_t>(n, store.n_rows());
    if (r1 > r0 && (merge || n_full == 0 || r1 - r0 >= fparams.min_rows)) {
        res.push_back({ r0, r1 });
    }
    if (!res.empty() && res.back().r1 < min_first) {
        res.clear();
    }
    return res;
}

bool llama_kvq_fitter::fitted(const range & r) const {
    const int32_t C = store.n_chunk_rows();
    return r.r1 <= store.n_coarse() && slot_fit[r.r0/C] == r;
}

bool llama_kvq_fitter::in_flight(const range & r) const {
    std::lock_guard<std::mutex> lk(m);
    return std::any_of(jobs.begin(), jobs.end(), [&](const std::unique_ptr<job> & jb) { return jb->r0 == r.r0 && jb->r1 == r.r1; });
}

// the ring cells of the positions [p0, p1) hold them, for sequence 0 only
bool llama_kvq_fitter::check_ring(const llama_kv_cache & kv, int32_t p0, int32_t p1) {
    const auto & cells = kv.get_cells(0);
    const int64_t size = cells.size();
    if (p1 - p0 > size) {
        LLAMA_LOG_ERROR("%s: the positions [%d, %d) are not in the ring of %lld cells, the fit of the coarse region stops\n", __func__, p0, p1, (long long) size);
        failed = true;
        return false;
    }
    for (int64_t p = p0; p < p1; ++p) {
        const int64_t j = p % size;
        if (cells.is_empty(j) || cells.pos_get(j) != p || !cells.seq_has(j, 0) || cells.seq_count(j) != 1) {
            LLAMA_LOG_ERROR("%s: cell %lld does not hold position %lld of sequence 0 only, the fit of the coarse region stops\n", __func__, (long long) j, (long long) p);
            failed = true;
            return false;
        }
    }
    return true;
}

// decode thread: a new job for the rows of r with their statistics, phase 1 and the fit tasks
// the ring rows go to the host store, so n_host covers r afterwards
bool llama_kvq_fitter::take(const range & r) {
    const int32_t C  = store.n_chunk_rows();
    const int32_t G  = store.n_head_kv();
    GGML_ASSERT(r.r0 % C == 0 && r.r1 > r.r0 && r.r1 - r.r0 <= n_stage);

    auto jb = std::make_unique<job>();
    jb->r0     = r.r0;
    jb->r1     = r.r1;
    jb->t_take = kvq_fit_now();
    jb->t_snap = jb->t_take;

    // statistics: the complete chunks of the range, then the sums of the open chunk, in this order
    // a partial chunk alone also takes the previous chunk's: its own sums can hold few positions (a ubatch counts for the chunk
    // of its first position), the queries just before it stand in for them
    const bool partial = r.r1 - r.r0 < C && r.r0 >= C;
    for (int32_t k = r.r0/C - (partial ? 1 : 0); (int64_t) k*C < r.r1; ++k) {
        std::vector<float> gram;
        std::vector<float> gsum;
        if ((k + 1)*C <= r.r1) {
            GGML_ASSERT(k < n_snap);
            gram = grams[k];
            gsum = gsums[k];
            jb->t_snap = t_snaps[k];
        } else {
            GGML_ASSERT(k == n_snap);
            read_acc(gram, gsum);
            acc_pos  = r.r1;
            acc_gram = gram;
            acc_gsum = gsum;
        }
        if (jb->gram.empty()) {
            jb->gram = std::move(gram);
            jb->gsum = std::move(gsum);
            continue;
        }
        for (size_t j = 0; j < gram.size(); ++j) {
            jb->gram[j] += gram[j];
        }
        for (size_t j = 0; j < gsum.size(); ++j) {
            jb->gsum[j] += gsum[j];
        }
    }

    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    const int32_t n_task = 2*n_pair;
    const size_t  n_b    = (size_t) n_task*(KVQ_D*KVQ_D + KVQ_D)*sizeof(float) + (size_t) n_task*KVQ_NP*KVQ_D*sizeof(float) +
                           (size_t) n_pair*KVQ_D*sizeof(float) + (size_t) n_task*KVQ_D*KVQ_D*sizeof(int8_t) + (size_t) n_task*3*KVQ_D*sizeof(float);
    if (!free_bufs.empty()) {
        jb->buf = std::move(free_bufs.back());
        free_bufs.pop_back();
    } else {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(store.host_buft(), n_b);
        jb->buf.reset(b != nullptr ? b : ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), n_b));
        if (!jb->buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate %.1f MiB for a job, the fit of the coarse region stops\n", __func__, n_b/1048576.0);
            failed = true;
            return false;
        }
    }
    uint8_t * hb = (uint8_t *) ggml_backend_buffer_get_base(jb->buf.get());
    jb->hstat = (float *) hb;                         hb += (size_t) n_task*(KVQ_D*KVQ_D + KVQ_D)*sizeof(float);
    jb->hpar  = (float *) hb;                         hb += (size_t) n_task*KVQ_NP*KVQ_D*sizeof(float);
    jb->hsig  = (float *) hb;                         hb += (size_t) n_pair*KVQ_D*sizeof(float);
    jb->htinv = (int8_t *) hb;                        hb += (size_t) n_task*KVQ_D*KVQ_D*sizeof(int8_t);
    jb->htaux = (float *) hb;
    jb->vt.resize(n_pair);
    for (auto & ev : jb->ev) {
        ev.reset(ggml_backend_event_new(store.device()));
    }
    if (!jb->ev[0] || !jb->ev[1]) {
        LLAMA_LOG_ERROR("%s: failed to create an event, the fit of the coarse region stops\n", __func__);
        failed = true;
        return false;
    }

    queue_phase(*jb, 1);
    store.set_n_host(std::max(store.n_host(), r.r1));

    // V first: phase 2 and the V bits tasks can run while the K tasks are still queued
    {
        std::lock_guard<std::mutex> lk(m);
        for (int32_t i = 0; i < n_task; ++i) {
            tasks.emplace_back(jb.get(), (n_pair + i) % n_task);
        }
        jb->n_left[0] = n_pair;
        jb->n_left[1] = n_pair;
        jobs.push_back(std::move(jb));
    }
    cv_work.notify_all();
    return true;
}

void llama_kvq_fitter::spill(const llama_kv_cache & kv, llama_pos p) {
    const int32_t h0 = store.n_host();
    if (failed || p <= h0) {
        return;
    }
    if (!check_ring(kv, h0, p)) {
        return;
    }
    // the host ring: the host store has the rows of the ring (llama_kvq_store::sync_store)
    store.set_n_host(p);
    n_spill++;
    LLAMA_LOG_INFO("%s: ring rows of the positions [%d, %d) to the host store\n", __func__, h0, (int) p);
}

// decode thread: the jobs whose tasks are done get their next phase
//   V fit tasks done: phase 2 and the V bits tasks (per-chunk V); all tasks done: phase 3
void llama_kvq_fitter::advance() {
    for (size_t k = 0; ; ++k) {
        job * jb = nullptr;
        bool  v_done = false;
        bool  done   = false;
        {
            std::lock_guard<std::mutex> lk(m);
            if (k >= jobs.size()) {
                return;
            }
            jb     = jobs[k].get();
            v_done = jb->stage == KVQ_STAGE_FIT && jb->n_left[1] == 0;
            done   = jb->stage <= KVQ_STAGE_BITS && jb->n_left[0] == 0 && jb->n_left[1] == 0;
        }
        if ((v_done || done) && jb->failed) {
            if (done) {
                jb->stage = KVQ_STAGE_FAILED;
            }
            continue;
        }
        if (v_done) {
            jb->stage = KVQ_STAGE_BITS;
            queue_phase(*jb, 2);
            std::lock_guard<std::mutex> lk(m);
            for (int32_t p = 0; p < n_pair(); ++p) {
                tasks.emplace_back(jb, 2*n_pair() + p);
            }
            jb->n_left[1] = n_pair();
            done = false;
            cv_work.notify_all();
        }
        if (done) {
            queue_phase(*jb, 3);
            jb->stage = KVQ_STAGE_READY;
        }
    }
}

// decode thread: the finished jobs at the front join the coarse region, all: also when hold
// the codes and tables are queued on the main stream; the wait for them (set_n_coarse or synchronize) comes before the host buffer is reused
void llama_kvq_fitter::install(bool all) {
    if (hold && !all) {
        return;
    }
    const int32_t G      = store.n_head_kv();
    const int32_t C      = store.n_chunk_rows();
    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    while (!failed) {
        std::unique_ptr<job> jb;
        {
            std::lock_guard<std::mutex> lk(m);
            if (jobs.empty() || jobs.front()->stage < KVQ_STAGE_READY) {
                return;
            }
            jb = std::move(jobs.front());
            jobs.pop_front();
        }
        if (jb->stage == KVQ_STAGE_FAILED) {
            LLAMA_LOG_ERROR("%s: the job of the rows [%d, %d) failed, the coarse region stays at %d rows\n", __func__, jb->r0, jb->r1, store.n_coarse());
            failed = true;
            store.synchronize();
            release(*jb);
            return;
        }
        GGML_ASSERT(jb->r0 <= store.n_coarse());

        // merge: the group's table set in each chunk slot of its rows
        for (int32_t t = jb->r0/C; (int64_t) t*C < jb->r1; ++t) {
            for (size_t li = 0; li < layer_ids.size(); ++li) {
                for (int32_t g = 0; g < G; ++g) {
                    for (int kv = 0; kv < 2; ++kv) {
                        const size_t i = kv*n_pair + li*G + g;
                        store.set_chunk_table(layer_ids[li], g, kv == 1, t, jb->htinv + i*KVQ_D*KVQ_D, jb->htaux + i*3*KVQ_D, true);
                    }
                }
            }
            slot_fit[t] = { jb->r0, jb->r1 };
        }
        const int32_t n0 = store.n_coarse();
        if (jb->r1 > n0) {
            store.set_n_coarse(jb->r1);
        } else {
            store.synchronize();
        }

        const double t1 = kvq_fit_now();
        t_job.push_back(t1 - jb->t_take);
        t_wall.push_back(t1 - jb->t_snap);
        t_cpu += jb->t_cpu;
        LLAMA_LOG_INFO("%s: rows [%d, %d) fitted (coarse region %d rows), job %.2f s, %.2f s CPU, %.2f s since the statistics\n", __func__,
                jb->r0, jb->r1, store.n_coarse(), t1 - jb->t_take, jb->t_cpu, t1 - jb->t_snap);
        release(*jb);
    }
}

void llama_kvq_fitter::poll(const llama_kv_cache & kv, llama_pos n_pos, bool large) {
    const int32_t C  = store.n_chunk_rows();
    const double  t0 = kvq_fit_now();
    const int32_t s0 = n_snap;
    snapshots(n_pos);
    bool took = n_snap != s0;
    // the complete chunks, in order; also a chunk whose tables came from a group or a shorter fit
    // merge: a chunk is fitted alone once the next one is complete too, before that it is the head of the group
    const int32_t lag = fparams.rem == LLAMA_KVQ_REM_MERGE ? 1 : 0;
    for (int32_t k = 0; large && !failed && k + lag < n_snap; ++k) {
        const range r = { k*C, (k + 1)*C };
        if (!fitted(r) && !in_flight(r)) {
            took = take(r) || took;
        }
    }
    const double t1 = kvq_fit_now();
    if (took) {
        t_take += t1 - t0;
    }
    if (!failed && !jobs_empty()) {
        advance();
        install(false);
        t_install += kvq_fit_now() - t1;
    }
}

// pool mutex: the next step of the job can be queued (or it is ready)
bool llama_kvq_fitter::can_advance(const job & jb) {
    const bool all = jb.n_left[0] == 0 && jb.n_left[1] == 0;
    return jb.stage >= KVQ_STAGE_READY || all || (jb.stage == KVQ_STAGE_FIT && jb.n_left[1] == 0 && !jb.failed);
}

bool llama_kvq_fitter::jobs_empty() const {
    std::lock_guard<std::mutex> lk(m);
    return jobs.empty();
}

void llama_kvq_fitter::wait_jobs() {
    while (true) {
        advance();
        install(true);
        std::unique_lock<std::mutex> lk(m);
        if (failed || jobs.empty()) {
            return;
        }
        cv_done.wait(lk, [&]() { return can_advance(*jobs.front()); });
    }
}

llama_pos llama_kvq_fitter::settle(const llama_kv_cache & kv, llama_pos n, const std::function<void(llama_pos)> & queued) {
    snapshots(n);
    const std::vector<range> lay = layout(n);
    const llama_pos e = lay.empty() ? 0 : lay.back().r1;
    if (failed) {
        return store.n_coarse();
    }

    // e.g. merge: the job of the last complete chunk alone, which the group replaces
    cancel([&](const job & jb) {
        return std::find(lay.begin(), lay.end(), range{ jb.r0, jb.r1 }) == lay.end();
    });
    for (const range & r : lay) {
        if (!fitted(r) && !in_flight(r) && !take(r)) {
            break;
        }
    }
    if (queued && !failed) {
        queued(e);
    }
    wait_jobs();
    return failed ? store.n_coarse() : e;
}

// decode thread: cancels the jobs of pred, waits for their running tasks and drops them
void llama_kvq_fitter::cancel(const std::function<bool(const job &)> & pred) {
    std::vector<std::unique_ptr<job>> gone;
    {
        std::unique_lock<std::mutex> lk(m);
        bool any = false;
        for (auto & jb : jobs) {
            if (pred(*jb)) {
                jb->cancel = true;
                any = true;
                LLAMA_LOG_INFO("%s: the job of the rows [%d, %d) is cancelled, %d K and %d V tasks left (stage %d)\n", __func__, jb->r0, jb->r1, jb->n_left[0], jb->n_left[1], jb->stage);
            }
        }
        if (!any) {
            return;
        }
        // the queued tasks of these jobs are dropped, the running ones stop at their next check
        for (auto it = tasks.begin(); it != tasks.end();) {
            if (it->first->cancel) {
                it->first->n_left[it->second >= n_pair() ? 1 : 0]--;
                it = tasks.erase(it);
            } else {
                ++it;
            }
        }
        cv_done.wait(lk, [&]() {
            return std::all_of(jobs.begin(), jobs.end(), [](const std::unique_ptr<job> & jb) { return !jb->cancel || jb->n_run == 0; });
        });
        for (auto it = jobs.begin(); it != jobs.end();) {
            if ((*it)->cancel) {
                n_cancel++;
                gone.push_back(std::move(*it));
                it = jobs.erase(it);
            } else {
                ++it;
            }
        }
    }
    // also the queued phases of the cancelled jobs, which use their host buffers
    store.synchronize();
    for (auto & jb : gone) {
        release(*jb);
    }
}

void llama_kvq_fitter::remove(llama_pos p0) {
    const int32_t C  = store.n_chunk_rows();
    const int32_t c0 = p0 / C;

    cancel([&](const job & jb) { return jb.r1 > p0; });

    // a removal in the chunk that collects statistics keeps the removed queries, unless all of them go or the removal goes back
    // to the end of the last fit of its rows; into a complete chunk: its statistics are collected again from the ones read when it was complete
    if (n_snap <= c0) {
        if (p0 == (llama_pos) c0*C && n_snap == c0) {
            store.synchronize();
            set_stats(nullptr, nullptr);
        } else if (p0 == acc_pos && n_snap == c0) {
            store.synchronize();
            set_stats(&acc_gram, &acc_gsum);
        }
        if (p0 < acc_pos) {
            acc_pos = -1;
        }
    } else {
        acc_pos = -1;
        store.synchronize();
        if (p0 > (llama_pos) c0*C) {
            set_stats(&grams[c0], &gsums[c0]);
        } else {
            set_stats(nullptr, nullptr);
        }
        grams.resize(c0);
        gsums.resize(c0);
        t_snaps.resize(c0);
        n_snap = c0;
    }

    // the coarse rows at or after p0: back to the chunk grid, the rows before p0 stay in the host store
    if (store.n_coarse() > p0) {
        const int32_t n0 = store.n_coarse();
        store.set_n_coarse(c0*C);
        for (int32_t t = c0; t < (int32_t) slot_fit.size(); ++t) {
            slot_fit[t] = range();
        }
        n_drop++;
        LLAMA_LOG_INFO("%s: the removal from position %d drops the coarse rows [%d, %d), the host store keeps [%d, %d)\n", __func__,
                p0, c0*C, n0, c0*C, (int) p0);
    }
    store.set_n_host(std::min<int32_t>(store.n_host(), p0));
    if (p0 == 0) {
        failed = false;
    }
}

void llama_kvq_fitter::state_write_meta(llama_io_write_i & io) const {
    {
        std::lock_guard<std::mutex> lk(m);
        if (failed || !jobs.empty()) {
            throw std::runtime_error(format("the kvq fitter is not idle (failed = %d, %zu jobs), the state is not saved", failed ? 1 : 0, jobs.size()));
        }
    }
    // element 2: V fitted per chunk (1; 0 was an offline V table file), element 3: the query heads of the gate statistics,
    // elements 4, 5: the policy for the rows after the last complete chunk and its smallest chunk (they change what the next round does)
    const int32_t v[6] = { fparams.bits_k, fparams.bits_v, 1, n_head, fparams.rem, fparams.min_rows };
    io.write(v, sizeof(v));
}

void llama_kvq_fitter::state_read_meta(llama_io_read_i & io) const {
    int32_t v[6];
    io.read(v, sizeof(v));
    if (v[0] != fparams.bits_k || v[1] != fparams.bits_v || v[2] != 1 || v[3] != n_head) {
        throw std::runtime_error(format("kvq state with K %d / V %d bits, V %s (%d heads), this context: K %d / V %d bits, V fitted per chunk (%d heads)",
                v[0], v[1], v[2] ? "fitted per chunk" : "from a table file", v[3], fparams.bits_k, fparams.bits_v, n_head));
    }
    if (v[4] != fparams.rem || v[5] != fparams.min_rows) {
        throw std::runtime_error(format("kvq state fitted with rem policy %d and min rows %d, this context: rem policy %d, min rows %d "
                "(the rows after the last complete chunk would be fitted differently); use the same --kvq-rem / --kvq-min-rows or process the prompt again",
                v[4], v[5], fparams.rem, fparams.min_rows));
    }
}

static size_t kvq_fit_n_gram(const llama_kvq_store & store) {
    return (size_t) KVQ_D*KVQ_D*store.n_head_kv()*store.layer_ids().size();
}

// the fitted range of each chunk slot of the coarse region, the statistics of the complete chunks, the device sums of the open chunk
void llama_kvq_fitter::state_write_data(llama_io_write_i & io) const {
    const int32_t C      = store.n_chunk_rows();
    const int32_t n_slot = (store.n_coarse() + C - 1)/C;
    const size_t  n_gram = kvq_fit_n_gram(store);
    const size_t  n_gsum = (size_t) n_head*KVQ_D*layer_ids.size();
    io.write(&n_slot, sizeof(n_slot));
    for (int32_t t = 0; t < n_slot; ++t) {
        io.write(&slot_fit[t].r0, sizeof(int32_t));
        io.write(&slot_fit[t].r1, sizeof(int32_t));
    }
    io.write(&n_snap, sizeof(n_snap));
    for (int32_t c = 0; c < n_snap; ++c) {
        GGML_ASSERT(grams[c].size() == n_gram && gsums[c].size() == n_gsum);
        io.write(grams[c].data(), n_gram*sizeof(float));
        if (n_gsum > 0) {
            io.write(gsums[c].data(), n_gsum*sizeof(float));
        }
    }
    for (int32_t il : layer_ids) {
        const auto & t = store.get_layer(il);
        io.write_tensor(t.qacc, 0, ggml_nbytes(t.qacc));
        io.write_tensor(t.gacc, 0, ggml_nbytes(t.gacc));
    }
    io.write(&acc_pos, sizeof(acc_pos));
    if (acc_pos >= 0) {
        GGML_ASSERT(acc_gram.size() == n_gram && acc_gsum.size() == n_gsum);
        io.write(acc_gram.data(), n_gram*sizeof(float));
        if (n_gsum > 0) {
            io.write(acc_gsum.data(), n_gsum*sizeof(float));
        }
    }
}

void llama_kvq_fitter::state_read_data(llama_io_read_i & io) {
    {
        std::lock_guard<std::mutex> lk(m);
        GGML_ASSERT(jobs.empty() && tasks.empty());
    }
    const int32_t C      = store.n_chunk_rows();
    const size_t  n_gram = kvq_fit_n_gram(store);
    const size_t  n_gsum = (size_t) n_head*KVQ_D*layer_ids.size();

    int32_t n_slot = 0;
    io.read(&n_slot, sizeof(n_slot));
    if (n_slot != (store.n_coarse() + C - 1)/C) {
        throw std::runtime_error(format("kvq state with %d fitted chunk slots for a coarse region of %d rows", n_slot, store.n_coarse()));
    }
    std::fill(slot_fit.begin(), slot_fit.end(), range());
    for (int32_t t = 0; t < n_slot; ++t) {
        range r;
        io.read(&r.r0, sizeof(int32_t));
        io.read(&r.r1, sizeof(int32_t));
        if (r.r0 < 0 || r.r0 > t*C || r.r1 <= t*C || r.r1 - r.r0 > n_stage) {
            throw std::runtime_error(format("kvq state with the rows [%d, %d) fitted for chunk %d", r.r0, r.r1, t));
        }
        slot_fit[t] = r;
    }
    int32_t ns = 0;
    io.read(&ns, sizeof(ns));
    if (ns < 0 || (int64_t) ns*C > store.n_rows()) {
        throw std::runtime_error(format("kvq state with %d complete chunks", ns));
    }
    grams.assign(ns, {});
    gsums.assign(ns, {});
    t_snaps.assign(ns, kvq_fit_now());
    for (int32_t c = 0; c < ns; ++c) {
        grams[c].resize(n_gram);
        io.read(grams[c].data(), n_gram*sizeof(float));
        gsums[c].resize(n_gsum);
        if (n_gsum > 0) {
            io.read(gsums[c].data(), n_gsum*sizeof(float));
        }
    }
    n_snap = ns;
    failed = false;
    store.synchronize();
    for (int32_t il : layer_ids) {
        const auto & t = store.get_layer(il);
        io.read_tensor(t.qacc, 0, ggml_nbytes(t.qacc));
        io.read_tensor(t.gacc, 0, ggml_nbytes(t.gacc));
    }
    io.read(&acc_pos, sizeof(acc_pos));
    acc_gram.clear();
    acc_gsum.clear();
    if (acc_pos >= 0) {
        acc_gram.resize(n_gram);
        io.read(acc_gram.data(), n_gram*sizeof(float));
        acc_gsum.resize(n_gsum);
        if (n_gsum > 0) {
            io.read(acc_gsum.data(), n_gsum*sizeof(float));
        }
    }
}

// decode thread, no phase of the job queued: keeps the host buffer of a finished or cancelled job for the next one
void llama_kvq_fitter::release(job & jb) {
    if (jb.buf && free_bufs.size() < 2) {
        free_bufs.push_back(std::move(jb.buf));
    }
    jb.buf.reset();
}

// decode thread, after a synchronize: the device statistics from the host copies, or 0 (nullptr or no copy)
void llama_kvq_fitter::set_stats(const std::vector<float> * gram, const std::vector<float> * gsum) {
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        const auto & t = store.get_layer(layer_ids[li]);
        for (int k = 0; k < 2; ++k) {
            ggml_tensor * acc = k == 0 ? t.qacc : t.gacc;
            const std::vector<float> * src = k == 0 ? gram : gsum;
            if (acc == nullptr) {
                continue;
            }
            const int64_t ne = ggml_nelements(acc);
            if (src != nullptr && (int64_t) src->size() == ne*(int64_t) layer_ids.size()) {
                ggml_backend_tensor_set(acc, src->data() + li*ne, 0, ggml_nbytes(acc));
            } else {
                ggml_backend_tensor_memset(acc, 0, 0, ggml_nbytes(acc));
            }
        }
    }
}

void llama_kvq_fitter::get_progress(llama_kvq_fit_progress & out) const {
    out = {};
    out.n_coarse = store.n_coarse();
    out.n_chunk  = (store.n_coarse() + store.n_chunk_rows() - 1)/store.n_chunk_rows();
    out.n_host   = store.n_host();
    out.n_spill  = n_spill;
    out.failed   = failed;
    out.n_cancel = n_cancel;
    out.n_drop   = n_drop;
    out.t_job    = t_job;
    out.t_wall   = t_wall;
    out.t_cpu     = t_cpu;
    out.t_take    = t_take;
    out.t_install = t_install;
    std::lock_guard<std::mutex> lk(m);
    for (const auto & jb : jobs) {
        if (jb->stage < KVQ_STAGE_READY) {
            out.n_running++;
        } else {
            out.n_ready++;
        }
    }
}
