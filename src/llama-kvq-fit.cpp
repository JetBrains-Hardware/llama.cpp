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
    int32_t c     = -1;
    int     stage = KVQ_STAGE_FIT; // decode thread

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
    double t_dev  = 0.0; // LLAMA_KVQ_FIT_SYNC
    double t_ph[3] = {};

    // the end of phase 1 (and later of phase 3), of phase 2
    ggml_backend_event_ptr ev[2];

    // pinned host memory
    ggml_backend_buffer_ptr buf;
    int32_t     * hidx  = nullptr; // [G][C]
    float       * hstat = nullptr; // per (layer index, K or V): cov [G][D*D], mu [G][D]
    float       * hpar  = nullptr; // per (layer index, K or V, kv head): [KVQ_NP][D]
    float       * hsig  = nullptr; // per (layer index, kv head): V scale of the shape [D]
    int8_t      * htinv = nullptr; // per task [D*D]
    float       * htaux = nullptr; // per task [3*D]

    // per-chunk V: the tables between the fit and the bits tasks, per (layer index, kv head)
    std::vector<llama_kvq_table> vt;
};

llama_kvq_fitter::llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams, const std::vector<llama_kvq_table> & v_tabs,
        const std::vector<const ggml_tensor *> & wo, int32_t n_head, int32_t margin) :
    store(store), fparams(fparams), margin(margin), layer_ids(store.layer_ids()), v_fit(v_tabs.empty()), n_head(n_head), wo(wo) {
    const int32_t G     = store.n_head_kv();
    const int32_t C     = store.n_chunk_rows();
    const int32_t nwv   = store.n_words(true);
    const int32_t n_chk = store.n_rows()/C;

    if (store.backend() == nullptr) {
        throw std::runtime_error("the fitter needs the backend of the device");
    }

    if (v_fit) {
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
    }

    // the offline V tables are the same for every chunk, so all table slots get them now
    std::vector<int8_t>      inv(KVQ_D*KVQ_D);
    std::vector<float>       aux(3*KVQ_D);
    for (int32_t il : layer_ids) {
        for (int32_t g = 0; !v_fit && g < G; ++g) {
            const auto it = std::find_if(v_tabs.begin(), v_tabs.end(), [&](const llama_kvq_table & t) {
                return t.il == il && t.head == g && t.is_v && t.n == KVQ_D;
            });
            if (it == v_tabs.end()) {
                throw std::runtime_error(format("no V table for layer %d, kv head %d", il, g));
            }
            int sum = 0;
            for (uint8_t b : it->bits) {
                sum += b;
            }
            if (sum != fparams.bits_v*KVQ_D || llama_kvq_pack_words(*it) > nwv) {
                throw std::runtime_error(format("the V table of layer %d, kv head %d has %d bits, expected %d", il, g, sum, fparams.bits_v*(int) KVQ_D));
            }
            v_tab.push_back(*it);
            llama_kvq_pack_table(*it, inv.data(), aux.data());
            for (int32_t c = 0; c < n_chk; ++c) {
                store.set_chunk_table(il, g, true, c, inv.data(), aux.data());
            }
        }
    }

    // device state, then the largest graphs once: the ops must run on the device, and the allocator does not grow later
    {
        ggml_init_params ip = { 3*ggml_tensor_overhead(), nullptr, true };
        ctx_dev.reset(ggml_init(ip));
        if (!ctx_dev) {
            throw std::runtime_error("failed to create the fitter context");
        }
        idx   = ggml_new_tensor_2d(ctx_dev.get(), GGML_TYPE_I32, C, G);
        par   = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_F32, KVQ_D, KVQ_NP, 2*G);
        stage = ggml_new_tensor_4d(ctx_dev.get(), GGML_TYPE_F16, KVQ_ROW/2, 2, C, G);
        ggml_set_name(idx,   "kvq_fit_idx");
        ggml_set_name(par,   "kvq_fit_par");
        ggml_set_name(stage, "kvq_fit_stage");
        buf_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev.get(), store.dev_buft()));
        if (!buf_dev) {
            throw std::runtime_error("failed to allocate the fitter device buffer");
        }
        ggml_backend_buffer_clear(buf_dev.get(), 0);

        galloc.reset(ggml_gallocr_new(store.dev_buft()));
        graph_meta.resize(ggml_tensor_overhead()*128 + ggml_graph_overhead());

        job jb;
        jb.c = 0;
        for (int phase = 1; phase <= 3; ++phase) {
            if (phase == 2 && !v_fit) {
                continue;
            }
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
    }
    if (v_fit) {
        compute_wo_gram_dev();
    }
    prof = getenv("LLAMA_KVQ_FIT_SYNC") != nullptr && atoi(getenv("LLAMA_KVQ_FIT_SYNC")) != 0;

    for (int i = 0; i < fparams.n_threads; ++i) {
        threads.emplace_back([this]() { loop(); });
    }
    const double mib_dev = (ggml_backend_buffer_get_size(buf_dev.get()) + ggml_gallocr_get_buffer_size(galloc.get(), 0))/1048576.0;
    if (v_fit) {
        LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits and V %d bits fitted per chunk on the device, %.1f MiB, %d CPU threads\n", __func__,
                C, n_chk, fparams.bits_k, fparams.bits_v, mib_dev, fparams.n_threads);
    } else {
        LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits fitted per chunk on the device, %.1f MiB, V %d bits from %zu offline tables, %d CPU threads\n", __func__,
                C, n_chk, fparams.bits_k, mib_dev, fparams.bits_v, v_tab.size(), fparams.n_threads);
    }
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
        if (!ggml_backend_supports_op(backend, x) || !ggml_gallocr_alloc_graph(galloc.get(), gf) ||
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
        gbar[j] = (double) gs[j]/store.n_chunk_rows();
        tot += gbar[j];
    }
    if (!(tot > 0.0)) {
        // no gate statistics (e.g. after a removal), plain o_proj metric
        LLAMA_LOG_WARN("%s: chunk %d, layer %d, kv head %d: no gate statistics, the V table is fitted without the gate\n", __func__, jb.c, layer_ids[li], g);
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
    const int64_t C      = store.n_chunk_rows();
    const int32_t blk    = 2*li + (is_v ? 1 : 0);
    float * par = jb.hpar + ((size_t) blk*G + g)*KVQ_NP*KVQ_D;

    if (is_v && !v_fit) {
        kvq_fit_set_par(par, v_tab[p]);
        return;
    }

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
            LLAMA_LOG_WARN("%s: chunk %d, layer %d, kv head %d: no query statistics, the K table is fitted for the MSE\n", __func__, jb.c, il, g);
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
    bool ok = llama_kvq_fit_cov(tab, mu, cov.data(), C, KVQ_D, metric.data(), is_v);
    if (ok && !is_v) {
        llama_kvq_fit_bits(tab, (float) fparams.bits_k);
        ok = llama_kvq_pack_words(tab) <= store.n_words(false);
    }
    if (!ok) {
        LLAMA_LOG_ERROR("%s: chunk %d: the %s fit of layer %d, kv head %d failed\n", __func__, jb.c, is_v ? "V" : "K", il, g);
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
        LLAMA_LOG_ERROR("%s: chunk %d: the V fit of layer %d, kv head %d failed\n", __func__, jb.c, layer_ids[li], g);
        jb.failed = true;
        return;
    }
    kvq_fit_set_par(jb.hpar + ((size_t) (2*li + 1)*G + g)*KVQ_NP*KVQ_D, tab);
    llama_kvq_pack_table(tab, jb.htinv + (size_t) i*KVQ_D*KVQ_D, jb.htaux + (size_t) i*3*KVQ_D);
    tab = {};
}

// the graph of one phase of one attention layer:
//   1: K and V rows -> mean, covariance (outputs); q8_0 rows -> stage
//   2: V (per-chunk V): rows -> centered -> rotated -> scale of the shape (output)
//   3: K and V: rows -> centered -> rotated -> codes in kc and vc
ggml_cgraph * llama_kvq_fitter::build_phase(job & jb, int phase, int32_t li, ggml_context * ctx) {
    const int32_t il   = layer_ids[li];
    const int64_t G    = store.n_head_kv();
    const int64_t C    = store.n_chunk_rows();
    const auto &  lt   = store.get_layer(il);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);

    for (int kv = phase == 2 ? 1 : 0; kv < 2; ++kv) {
        // the rows of the chunk from the ring: the cache [G*D, size] seen as [D, size, G]
        ggml_tensor * cache = store.get_cache(il, kv == 1);
        ggml_tensor * cv    = ggml_view_3d(ctx, cache, KVQ_D, cache->ne[1], G, cache->nb[1], KVQ_ROW, 0);
        ggml_tensor * x     = ggml_get_rows(ctx, cv, idx);

        if (phase == 1) {
            ggml_tensor * mu = ggml_kvq_fit(ctx, GGML_KVQ_FIT_MEAN, x, nullptr, nullptr);
            ggml_set_output(mu);
            ggml_tensor * cov = ggml_kvq_fit(ctx, GGML_KVQ_FIT_COV, ggml_sub(ctx, x, mu), nullptr, nullptr);
            ggml_set_output(cov);
            ggml_format_name(mu,  "kvq_fit_mu%d",  kv);
            ggml_format_name(cov, "kvq_fit_cov%d", kv);
            ggml_build_forward_expand(gf, mu);
            ggml_build_forward_expand(gf, cov);

            // the q8_0 rows as f16 pairs, cell j0 + r -> stage row r, in at most two runs
            ggml_tensor * src  = kv ? lt.v_h : lt.k_h;
            const int64_t size = src->ne[2];
            const int64_t j0   = ((int64_t) jb.c*C) % size;
            const int64_t n0   = std::min<int64_t>(C, size - j0);
            for (int64_t r0 = 0; r0 < C; r0 += r0 == 0 ? n0 : C) {
                const int64_t n = r0 == 0 ? n0 : C - n0;
                const int64_t j = r0 == 0 ? j0 : 0;
                ggml_tensor * s = ggml_view_3d(ctx, src, KVQ_ROW/2, G, n, src->nb[1], src->nb[2], j*src->nb[2]);
                ggml_tensor * d = ggml_view_3d(ctx, stage, KVQ_ROW/2, G, n, stage->nb[3], stage->nb[2], kv*stage->nb[1] + r0*stage->nb[2]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, s, d));
            }
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
            ggml_tensor * rows = kv ? lt.vc : lt.kc;
            ggml_tensor * rv   = ggml_view_3d(ctx, rows, rows->ne[0], C, G, rows->nb[1], rows->nb[2], (size_t) jb.c*C*rows->nb[1]);
            ggml_build_forward_expand(gf, ggml_kvq_fit(ctx, GGML_KVQ_FIT_CODE, y, pq, rv));
        }
    }

    return gf;
}

// decode thread: queues one phase of a job on the main stream, no wait (unless LLAMA_KVQ_FIT_SYNC)
void llama_kvq_fitter::queue_phase(job & jb, int phase) {
    ggml_backend_t backend = store.backend();
    const int64_t  G       = store.n_head_kv();
    const int64_t  C       = store.n_chunk_rows();
    const size_t   n_par   = (size_t) G*KVQ_NP*KVQ_D; // floats per (layer index, K or V)
    const size_t   n_stat  = (size_t) G*(KVQ_D*KVQ_D + KVQ_D);

    double t0 = 0.0;
    if (prof) {
        store.synchronize();
        t0 = kvq_fit_now();
    }

    ggml_backend_tensor_set_async(backend, idx, jb.hidx, 0, ggml_nbytes(idx));
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        const int32_t il = layer_ids[li];
        if (phase == 2) {
            ggml_backend_tensor_set_async(backend, par, jb.hpar + (2*li + 1)*n_par, G*par->nb[2], n_par*sizeof(float));
        } else if (phase == 3) {
            ggml_backend_tensor_set_async(backend, par, jb.hpar + 2*li*n_par, 0, 2*n_par*sizeof(float));
        }

        ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_cgraph * gf = build_phase(jb, phase, (int32_t) li, ctx.get());
        if (!ggml_gallocr_alloc_graph(galloc.get(), gf) || ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
            GGML_ABORT("%s: phase %d of chunk %d, layer %d failed\n", __func__, phase, jb.c, il);
        }

        // the outputs, before the next graph reuses the memory
        if (phase == 1) {
            for (int kv = 0; kv < 2; ++kv) {
                float * st = jb.hstat + (2*li + kv)*n_stat;
                ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, format("kvq_fit_cov%d", kv).c_str()), st, 0, G*KVQ_D*KVQ_D*sizeof(float));
                ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, format("kvq_fit_mu%d", kv).c_str()), st + G*KVQ_D*KVQ_D, 0, G*KVQ_D*sizeof(float));
            }
            // host store [head][row][K, V]
            for (int64_t g = 0; g < G; ++g) {
                uint8_t * dst = store.host_rows(il) + ((size_t) g*store.n_rows() + (size_t) jb.c*C)*2*KVQ_ROW;
                ggml_backend_tensor_get_async(backend, stage, dst, g*stage->nb[3], stage->nb[3]);
            }
        } else if (phase == 2) {
            ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, "kvq_fit_sig"), jb.hsig + li*G*KVQ_D, 0, G*KVQ_D*sizeof(float));
        }
    }
    ggml_backend_event_record(jb.ev[phase == 2 ? 1 : 0].get(), backend);

    if (prof) {
        store.synchronize();
        jb.t_dev += kvq_fit_now() - t0;
        jb.t_ph[phase - 1] += kvq_fit_now() - t0;
    }
}

// decode thread, when the chunk is complete: the sums of the ubatches of this chunk are computed, then they start again from 0
void llama_kvq_fitter::snapshot() {
    const size_t nn = (size_t) KVQ_D*KVQ_D*store.n_head_kv();

    snap sn;
    sn.c = n_snap;
    sn.t = kvq_fit_now();
    sn.gram.resize(nn*layer_ids.size());
    sn.gsum.resize(v_fit ? (size_t) n_head*KVQ_D*layer_ids.size() : 0);

    store.synchronize();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        ggml_tensor * acc = store.get_layer(layer_ids[li]).qacc;
        ggml_backend_tensor_get(acc, sn.gram.data() + li*nn, 0, ggml_nbytes(acc));
        if (v_fit) {
            ggml_tensor * gacc = store.get_layer(layer_ids[li]).gacc;
            ggml_backend_tensor_get(gacc, sn.gsum.data() + li*gacc->ne[0], 0, ggml_nbytes(gacc));
        }
    }
    set_stats(nullptr, nullptr);

    snaps.push_back(std::move(sn));
    n_snap++;
}

// decode thread, the margin after the chunk: a new job with its statistics, phase 1 and the fit tasks
void llama_kvq_fitter::take(const llama_kv_cache & kv) {
    const int32_t c  = n_taken;
    const int32_t C  = store.n_chunk_rows();
    const int32_t G  = store.n_head_kv();
    GGML_ASSERT(!snaps.empty() && snaps.front().c == c);

    // position p is in cell p mod size (one cell per position up to n_ctx, or the ring of the tail)
    const auto & cells = kv.get_cells(0);
    const int64_t size = cells.size();
    for (int64_t p = (int64_t) c*C; p < (int64_t) (c + 1)*C; ++p) {
        const int64_t j = p % size;
        if (cells.is_empty(j) || cells.pos_get(j) != p || !cells.seq_has(j, 0) || cells.seq_count(j) != 1) {
            LLAMA_LOG_ERROR("%s: cell %lld does not hold position %lld of sequence 0 only, the fit of the coarse region stops\n", __func__, (long long) j, (long long) p);
            failed = true;
            return;
        }
    }

    auto jb = std::make_unique<job>();
    jb->c      = c;
    jb->t_snap = snaps.front().t;
    jb->t_take = kvq_fit_now();
    jb->gram   = std::move(snaps.front().gram);
    jb->gsum   = std::move(snaps.front().gsum);
    snaps.pop_front();

    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    const int32_t n_task = 2*n_pair;
    const size_t  n_b    = (size_t) G*C*sizeof(int32_t) + (size_t) n_task*(KVQ_D*KVQ_D + KVQ_D)*sizeof(float) + (size_t) n_task*KVQ_NP*KVQ_D*sizeof(float) +
                           (size_t) n_pair*KVQ_D*sizeof(float) + (size_t) n_task*KVQ_D*KVQ_D*sizeof(int8_t) + (size_t) n_task*3*KVQ_D*sizeof(float);
    if (!free_bufs.empty()) {
        jb->buf = std::move(free_bufs.back());
        free_bufs.pop_back();
    } else {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(store.host_buft(), n_b);
        jb->buf.reset(b != nullptr ? b : ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), n_b));
        if (!jb->buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate %.1f MiB for a chunk, the fit of the coarse region stops\n", __func__, n_b/1048576.0);
            failed = true;
            return;
        }
    }
    uint8_t * hb = (uint8_t *) ggml_backend_buffer_get_base(jb->buf.get());
    jb->hidx  = (int32_t *) hb;                       hb += (size_t) G*C*sizeof(int32_t);
    jb->hstat = (float *) hb;                         hb += (size_t) n_task*(KVQ_D*KVQ_D + KVQ_D)*sizeof(float);
    jb->hpar  = (float *) hb;                         hb += (size_t) n_task*KVQ_NP*KVQ_D*sizeof(float);
    jb->hsig  = (float *) hb;                         hb += (size_t) n_pair*KVQ_D*sizeof(float);
    jb->htinv = (int8_t *) hb;                        hb += (size_t) n_task*KVQ_D*KVQ_D*sizeof(int8_t);
    jb->htaux = (float *) hb;
    for (int32_t g = 0; g < G; ++g) {
        for (int32_t r = 0; r < C; ++r) {
            jb->hidx[(size_t) g*C + r] = (int32_t) (((int64_t) c*C + r) % size);
        }
    }
    jb->vt.resize(v_fit ? n_pair : 0);
    for (auto & ev : jb->ev) {
        ev.reset(ggml_backend_event_new(store.device()));
    }
    if (!jb->ev[0] || !jb->ev[1]) {
        LLAMA_LOG_ERROR("%s: failed to create an event, the fit of the coarse region stops\n", __func__);
        failed = true;
        return;
    }

    queue_phase(*jb, 1);

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
    n_taken++;
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
            if (v_fit) {
                queue_phase(*jb, 2);
                std::lock_guard<std::mutex> lk(m);
                for (int32_t p = 0; p < n_pair(); ++p) {
                    tasks.emplace_back(jb, 2*n_pair() + p);
                }
                jb->n_left[1] = n_pair();
                done = false;
                cv_work.notify_all();
            }
        }
        if (done) {
            queue_phase(*jb, 3);
            jb->stage = KVQ_STAGE_READY;
        }
    }
}

// decode thread: the finished jobs at the front join the coarse region, all: also when hold
// the codes and tables are queued on the main stream, set_n_coarse waits for them
void llama_kvq_fitter::install(bool all) {
    if (hold && !all) {
        return;
    }
    const int32_t G      = store.n_head_kv();
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
            LLAMA_LOG_ERROR("%s: the job of chunk %d failed, the coarse region stays at %d rows\n", __func__, jb->c, store.n_coarse());
            failed = true;
            store.synchronize();
            release(*jb);
            return;
        }
        GGML_ASSERT(jb->c == n_coarse_chunks);

        for (size_t li = 0; li < layer_ids.size(); ++li) {
            for (int32_t g = 0; g < G; ++g) {
                for (int kv = 0; kv < (v_fit ? 2 : 1); ++kv) {
                    const size_t i = kv*n_pair + li*G + g;
                    store.set_chunk_table(layer_ids[li], g, kv == 1, jb->c, jb->htinv + i*KVQ_D*KVQ_D, jb->htaux + i*3*KVQ_D, true);
                }
            }
        }
        store.set_n_coarse((jb->c + 1)*store.n_chunk_rows());
        n_coarse_chunks++;

        grams.resize(n_coarse_chunks);
        grams[jb->c] = std::move(jb->gram);
        gsums.resize(n_coarse_chunks);
        gsums[jb->c] = std::move(jb->gsum);

        const double t1 = kvq_fit_now();
        t_job.push_back(t1 - jb->t_take);
        t_wall.push_back(t1 - jb->t_snap);
        t_cpu += jb->t_cpu;
        t_dev += jb->t_dev;
        if (prof) {
            LLAMA_LOG_INFO("%s: chunk %d joined the coarse region (%d rows), job %.2f s, %.2f s CPU, %.1f ms device (%.1f + %.1f + %.1f), %.2f s since the statistics\n", __func__,
                    jb->c, store.n_coarse(), t1 - jb->t_take, jb->t_cpu, 1e3*jb->t_dev, 1e3*jb->t_ph[0], 1e3*jb->t_ph[1], 1e3*jb->t_ph[2], t1 - jb->t_snap);
        } else {
            LLAMA_LOG_INFO("%s: chunk %d joined the coarse region (%d rows), job %.2f s, %.2f s CPU, %.2f s since the statistics\n", __func__,
                    jb->c, store.n_coarse(), t1 - jb->t_take, jb->t_cpu, t1 - jb->t_snap);
        }
        release(*jb);
    }
}

void llama_kvq_fitter::poll(const llama_kv_cache & kv, int32_t m) {
    const int32_t C = store.n_chunk_rows();
    const double  t0 = kvq_fit_now();
    const llama_pos pos_max = kv.seq_pos_max(0);
    bool took = false;
    while (!failed && (int64_t) (n_snap + 1)*C <= store.n_rows() && pos_max >= (n_snap + 1)*C - 1) {
        snapshot();
        took = true;
    }
    while (!failed && n_taken < n_snap && pos_max >= (n_taken + 1)*C - 1 + m) {
        take(kv);
        took = true;
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

bool llama_kvq_fitter::wait_front(const llama_kv_cache & kv) {
    const int32_t n0 = n_coarse_chunks;
    poll(kv, margin);
    if (n_coarse_chunks == n0 && jobs_empty()) {
        poll(kv, 0);
    }
    while (n_coarse_chunks == n0) {
        {
            std::unique_lock<std::mutex> lk(m);
            if (failed || jobs.empty()) {
                return false;
            }
            cv_done.wait(lk, [&]() { return can_advance(*jobs.front()); });
        }
        advance();
        install(true);
    }
    return true;
}

void llama_kvq_fitter::wait(const llama_kv_cache & kv) {
    poll(kv, 0);
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

void llama_kvq_fitter::remove(llama_pos p0) {
    const int32_t C  = store.n_chunk_rows();
    const int32_t c0 = p0 / C;

    // the removal is in the chunk that collects statistics: they keep the removed queries, unless all of them go
    if (n_snap <= c0) {
        if (p0 == (llama_pos) c0*C && n_snap == c0) {
            store.synchronize();
            set_stats(nullptr, nullptr);
        }
        if (p0 == 0) {
            failed = false;
        }
        return;
    }

    // the statistics of chunk c0, to collect the rest of it again
    std::vector<float> gram_c0;
    std::vector<float> gsum_c0;
    if (c0 < (int32_t) grams.size()) {
        gram_c0 = std::move(grams[c0]);
        gsum_c0 = std::move(gsums[c0]);
    }
    // complete chunks without a job
    while (!snaps.empty() && snaps.back().c >= c0) {
        if (snaps.back().c == c0) {
            gram_c0 = std::move(snaps.back().gram);
            gsum_c0 = std::move(snaps.back().gsum);
        }
        snaps.pop_back();
    }
    std::vector<std::unique_ptr<job>> gone;
    {
        std::unique_lock<std::mutex> lk(m);
        for (auto & jb : jobs) {
            if (jb->c >= c0) {
                jb->cancel = true;
                LLAMA_LOG_INFO("%s: the job of chunk %d is cancelled, %d K and %d V tasks left (stage %d)\n", __func__, jb->c, jb->n_left[0], jb->n_left[1], jb->stage);
            }
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
        // the running tasks read the statistics of their job until here
        for (auto it = jobs.begin(); it != jobs.end();) {
            if ((*it)->cancel) {
                if ((*it)->c == c0) {
                    gram_c0 = std::move((*it)->gram);
                    gsum_c0 = std::move((*it)->gsum);
                }
                n_cancel++;
                gone.push_back(std::move(*it));
                it = jobs.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (n_coarse_chunks > c0) {
        LLAMA_LOG_INFO("%s: the removal from position %d drops %d chunks of the coarse region, %d rows left\n", __func__,
                p0, n_coarse_chunks - c0, c0*C);
        n_drop += n_coarse_chunks - c0;
        n_coarse_chunks = c0;
        store.set_n_coarse(c0*C);
    }
    grams.resize(std::min<size_t>(grams.size(), c0));
    gsums.resize(std::min<size_t>(gsums.size(), c0));

    // also the queued phases of the cancelled jobs, which use their host buffers
    store.synchronize();
    for (auto & jb : gone) {
        release(*jb);
    }
    if (p0 > (llama_pos) c0*C) {
        set_stats(&gram_c0, &gsum_c0);
    } else {
        set_stats(nullptr, nullptr);
    }
    n_taken = std::min(n_taken, c0);
    n_snap  = c0;
    // a failed chunk before c0 still blocks the later ones
    if (c0 == 0) {
        failed = false;
    }
}

static size_t kvq_fit_n_gram(const llama_kvq_store & store) {
    return (size_t) KVQ_D*KVQ_D*store.n_head_kv()*store.layer_ids().size();
}

void llama_kvq_fitter::state_write_meta(llama_io_write_i & io) const {
    {
        std::lock_guard<std::mutex> lk(m);
        if (failed || !jobs.empty() || !snaps.empty() || n_taken != n_coarse_chunks) {
            throw std::runtime_error(format("the kvq fitter is not idle (failed = %d, %zu jobs, %zu complete chunks without a job), the state is not saved",
                    failed ? 1 : 0, jobs.size(), snaps.size()));
        }
    }
    GGML_ASSERT(n_coarse_chunks*store.n_chunk_rows() == store.n_coarse() && n_snap == n_coarse_chunks);
    const int32_t v[4] = { fparams.bits_k, fparams.bits_v, v_fit ? 1 : 0, v_fit ? n_head : 0 };
    io.write(v, sizeof(v));
}

void llama_kvq_fitter::state_read_meta(llama_io_read_i & io, int32_t n_chunk) const {
    int32_t v[4];
    io.read(v, sizeof(v));
    if (v[0] != fparams.bits_k || v[1] != fparams.bits_v || v[2] != (v_fit ? 1 : 0) || v[3] != (v_fit ? n_head : 0)) {
        throw std::runtime_error(format("kvq state with K %d / V %d bits, V %s, this context: K %d / V %d bits, V %s",
                v[0], v[1], v[2] ? "fitted per chunk" : "from a table file", fparams.bits_k, fparams.bits_v, v_fit ? "fitted per chunk" : "from a table file"));
    }
    GGML_ASSERT(n_chunk >= 0 && (int64_t) n_chunk*store.n_chunk_rows() <= store.n_rows());
}

// the statistics of each coarse chunk, then the device sums of the open chunk
void llama_kvq_fitter::state_write_data(llama_io_write_i & io) const {
    const size_t n_gram = kvq_fit_n_gram(store);
    const size_t n_gsum = v_fit ? (size_t) n_head*KVQ_D*layer_ids.size() : 0;
    GGML_ASSERT((int32_t) grams.size() >= n_coarse_chunks && (int32_t) gsums.size() >= n_coarse_chunks);
    for (int32_t c = 0; c < n_coarse_chunks; ++c) {
        GGML_ASSERT(grams[c].size() == n_gram && gsums[c].size() == n_gsum);
        io.write(grams[c].data(), n_gram*sizeof(float));
        if (n_gsum > 0) {
            io.write(gsums[c].data(), n_gsum*sizeof(float));
        }
    }
    for (int32_t il : layer_ids) {
        const auto & t = store.get_layer(il);
        io.write_tensor(t.qacc, 0, ggml_nbytes(t.qacc));
        if (v_fit) {
            io.write_tensor(t.gacc, 0, ggml_nbytes(t.gacc));
        }
    }
}

void llama_kvq_fitter::state_read_data(llama_io_read_i & io, int32_t n_chunk) {
    {
        std::lock_guard<std::mutex> lk(m);
        GGML_ASSERT(jobs.empty() && tasks.empty());
    }
    GGML_ASSERT(store.n_coarse() == n_chunk*store.n_chunk_rows());
    const size_t n_gram = kvq_fit_n_gram(store);
    const size_t n_gsum = v_fit ? (size_t) n_head*KVQ_D*layer_ids.size() : 0;
    grams.assign(n_chunk, {});
    gsums.assign(n_chunk, {});
    for (int32_t c = 0; c < n_chunk; ++c) {
        grams[c].resize(n_gram);
        io.read(grams[c].data(), n_gram*sizeof(float));
        gsums[c].resize(n_gsum);
        if (n_gsum > 0) {
            io.read(gsums[c].data(), n_gsum*sizeof(float));
        }
    }
    snaps.clear();
    n_coarse_chunks = n_chunk;
    n_taken         = n_chunk;
    n_snap          = n_chunk;
    failed          = false;
    store.synchronize();
    for (int32_t il : layer_ids) {
        const auto & t = store.get_layer(il);
        io.read_tensor(t.qacc, 0, ggml_nbytes(t.qacc));
        if (v_fit) {
            io.read_tensor(t.gacc, 0, ggml_nbytes(t.gacc));
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
    out.n_chunk  = n_coarse_chunks;
    out.failed   = failed;
    out.n_cancel = n_cancel;
    out.n_drop   = n_drop;
    out.t_job    = t_job;
    out.t_wall   = t_wall;
    out.t_cpu     = t_cpu;
    out.t_take    = t_take;
    out.t_install = t_install;
    out.t_dev     = t_dev;
    std::lock_guard<std::mutex> lk(m);
    for (const auto & jb : jobs) {
        if (jb->stage < KVQ_STAGE_READY) {
            out.n_running++;
        } else {
            out.n_ready++;
        }
    }
}
