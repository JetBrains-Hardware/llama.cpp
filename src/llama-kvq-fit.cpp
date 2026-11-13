#include "llama-kvq-fit.h"

#include "llama-impl.h"
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

struct llama_kvq_fitter::job {
    int32_t c = -1;

    std::vector<float> gram; // [n_layer][G][D*D]
    std::vector<float> gsum; // [n_layer][n_gate], per-chunk V only

    std::atomic<bool> cancel{false};
    std::atomic<bool> failed{false};

    // pool mutex
    int32_t n_left  = 0; // tasks not finished
    int32_t n_run   = 0; // tasks running
    double  t_take  = 0.0;
    double  t_start = 0.0;
    double  t_end   = 0.0;
    double  t_cpu   = 0.0;
    // q8_0 rows of the chunk read on the decode thread, [attention layer][K, V][row][G*ROW] (pinned)
    ggml_backend_buffer_ptr buf;
    const uint8_t *         raw = nullptr;

    // per task i = kv*n_pair + layer index*G + kv head, kv = 0 for K, 1 for V
    std::vector<std::vector<int32_t>>     rows; // K: [n_crow][nwk], V: [n_crow][nwv + 1]
    std::vector<std::vector<ggml_fp16_t>> tinv; // [D*D], fitted tables only
    std::vector<std::vector<float>>       taux; // [3*D]
};

llama_kvq_fitter::llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams, const std::vector<llama_kvq_table> & v_tabs,
        const std::vector<const ggml_tensor *> & wo, int32_t n_head) :
    store(store), fparams(fparams), layer_ids(store.layer_ids()), v_fit(v_tabs.empty()), n_head(n_head), wo(wo) {
    const int32_t G     = store.n_head_kv();
    const int32_t nwv   = store.n_words(true);
    const int32_t n_chk = store.n_rows()/store.n_chunk_rows();

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
    std::vector<ggml_fp16_t> inv(KVQ_D*KVQ_D);
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

    for (int i = 0; i < fparams.n_threads; ++i) {
        threads.emplace_back([this]() { loop(); });
    }
    if (v_fit) {
        LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits and V %d bits fitted per chunk, %d threads\n", __func__,
                store.n_chunk_rows(), n_chk, fparams.bits_k, fparams.bits_v, fparams.n_threads);
    } else {
        LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits fitted per chunk, V %d bits from %zu offline tables, %d threads\n", __func__,
                store.n_chunk_rows(), n_chk, fparams.bits_k, fparams.bits_v, v_tab.size(), fparams.n_threads);
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
        const double t0 = kvq_fit_now();
        if (jb->t_start == 0.0) {
            jb->t_start = t0;
        }
        lk.unlock();
        if (!jb->cancel && !jb->failed) {
            run(*jb, i);
        }
        const double t1 = kvq_fit_now();
        lk.lock();
        jb->t_cpu += t1 - t0;
        jb->n_run--;
        if (--jb->n_left == 0) {
            jb->t_end = t1;
        }
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

// V metric of a chunk: sum over r of D_r Wo_r^T Wo_r D_r, D_r = diag of the chunk's mean sigmoid(gate) of query head g*n_rep + r
void llama_kvq_fitter::metric_v(const job & jb, int32_t p, std::vector<double> & m) {
    const int32_t G      = store.n_head_kv();
    const int32_t li     = p / G;
    const int32_t g      = p % G;
    const int64_t n_gate = (int64_t) n_head*KVQ_D;

    std::call_once(wo_once[p], [&]() { compute_wo_gram(p); });

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

// pool thread: K or V of one (layer, kv head) of a chunk
void llama_kvq_fitter::run(job & jb, int32_t i) {
    const int32_t G      = store.n_head_kv();
    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    const bool    is_v   = i >= n_pair;
    const int32_t p      = i % n_pair;
    const int32_t li     = p / G;
    const int32_t g      = p % G;
    const int32_t il     = layer_ids[li];
    const int64_t C      = store.n_chunk_rows();
    const int64_t r0     = (int64_t) jb.c*C;
    const size_t  nb1    = (size_t) G*KVQ_ROW;
    const int32_t nw     = store.n_words(is_v);

    const auto * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);

    // q8_0 rows of the chunk: [cell][head][ROW] -> host store [head][row][K, V], and f32 rows of this head
    std::vector<float> x(C*KVQ_D);
    uint8_t * hs = store.host_rows(il) + ((size_t) g*store.n_rows() + r0)*2*KVQ_ROW + (is_v ? KVQ_ROW : 0);
    const uint8_t * raw = jb.raw + (size_t) (2*li + (is_v ? 1 : 0))*C*nb1;
    for (int64_t r = 0; r < C; ++r) {
        const uint8_t * src = raw + r*nb1 + g*KVQ_ROW;
        memcpy(hs + r*2*KVQ_ROW, src, KVQ_ROW);
        q8->to_float(src, x.data() + r*KVQ_D, KVQ_D);
    }
    if (jb.cancel) {
        return;
    }

    std::vector<int8_t> codes(C*KVQ_D);
    std::vector<float>  gain(C);

    if (is_v && !v_fit) {
        const llama_kvq_table & vt = v_tab[p];
        llama_kvq_encode(vt, x.data(), C, codes.data(), gain.data(), 1);
        jb.rows[i].resize(C*(nw + 1));
        llama_kvq_pack_rows(vt, nw, codes.data(), gain.data(), C, jb.rows[i].data());
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

    llama_kvq_table tab;
    const float bits = (float) (is_v ? fparams.bits_v : fparams.bits_k);
    if (!llama_kvq_fit_encode(tab, x.data(), C, KVQ_D, metric.data(), bits, is_v, codes.data(), is_v ? gain.data() : nullptr, 1) ||
            llama_kvq_pack_words(tab) > nw) {
        LLAMA_LOG_ERROR("%s: chunk %d: the %s fit of layer %d, kv head %d failed\n", __func__, jb.c, is_v ? "V" : "K", il, g);
        jb.failed = true;
        return;
    }

    jb.rows[i].resize(C*(nw + (is_v ? 1 : 0)));
    llama_kvq_pack_rows(tab, nw, codes.data(), is_v ? gain.data() : nullptr, C, jb.rows[i].data());
    jb.tinv[i].resize(KVQ_D*KVQ_D);
    jb.taux[i].resize(3*KVQ_D);
    llama_kvq_pack_table(tab, jb.tinv[i].data(), jb.taux[i].data());
}

// decode thread
void llama_kvq_fitter::take(const llama_kv_cache & kv) {
    const int32_t c  = n_taken;
    const int32_t C  = store.n_chunk_rows();
    const int32_t G  = store.n_head_kv();
    const size_t  nn = (size_t) KVQ_D*KVQ_D*G;

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
    jb->c = c;
    jb->gram.resize(nn*layer_ids.size());
    jb->gsum.resize(v_fit ? (size_t) n_head*KVQ_D*layer_ids.size() : 0);

    // the sums of the ubatches of this chunk are computed, then they start again from 0
    store.synchronize();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        ggml_tensor * acc = store.get_layer(layer_ids[li]).qacc;
        ggml_backend_tensor_get(acc, jb->gram.data() + li*nn, 0, ggml_nbytes(acc));
        if (v_fit) {
            ggml_tensor * gacc = store.get_layer(layer_ids[li]).gacc;
            ggml_backend_tensor_get(gacc, jb->gsum.data() + li*gacc->ne[0], 0, ggml_nbytes(gacc));
        }
    }
    set_stats(nullptr, nullptr);

    // the rows of the chunk while the device is idle: a read from a pool thread waits behind the queued uploads of the copy stream
    const size_t nb1 = (size_t) G*KVQ_ROW;
    const size_t n_b = layer_ids.size()*2*C*nb1;
    if (!free_bufs.empty()) {
        jb->buf = std::move(free_bufs.back());
        free_bufs.pop_back();
    } else {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(store.host_buft(), n_b);
        jb->buf.reset(b != nullptr ? b : ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), n_b));
        if (!jb->buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate %.1f MiB for the rows of a chunk, the fit of the coarse region stops\n", __func__, n_b/1048576.0);
            failed = true;
            return;
        }
    }
    uint8_t * raw = (uint8_t *) ggml_backend_buffer_get_base(jb->buf.get());
    jb->raw = raw;
    const int64_t j0 = ((int64_t) c*C) % size;
    const int64_t n0 = std::min<int64_t>(C, size - j0);
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        for (int kv = 0; kv < 2; ++kv) {
            uint8_t * dst = raw + (2*li + kv)*C*nb1;
            ggml_tensor * t = store.get_cache(layer_ids[li], kv == 1);
            ggml_backend_tensor_get(t, dst, j0*nb1, n0*nb1);
            if (n0 < C) {
                ggml_backend_tensor_get(t, dst + n0*nb1, 0, (C - n0)*nb1);
            }
        }
    }

    // K and V of a kv head next to each other in the queue
    const int32_t n_pair = (int32_t) layer_ids.size()*G;
    const int32_t n_task = 2*n_pair;
    jb->rows.resize(n_task);
    jb->tinv.resize(n_task);
    jb->taux.resize(n_task);
    jb->n_left = n_task;
    jb->t_take = kvq_fit_now();

    {
        std::lock_guard<std::mutex> lk(m);
        for (int32_t p = 0; p < n_pair; ++p) {
            tasks.emplace_back(jb.get(), p);
            tasks.emplace_back(jb.get(), n_pair + p);
        }
        jobs.push_back(std::move(jb));
    }
    cv_work.notify_all();
    n_taken++;
}

// decode thread: the finished jobs at the front join the coarse region, all: also when hold
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
            if (jobs.empty() || jobs.front()->n_left > 0) {
                return;
            }
            jb = std::move(jobs.front());
            jobs.pop_front();
        }
        if (jb->failed) {
            LLAMA_LOG_ERROR("%s: the job of chunk %d failed, the coarse region stays at %d rows\n", __func__, jb->c, store.n_coarse());
            failed = true;
            return;
        }
        GGML_ASSERT(jb->c == n_coarse_chunks);

        const double t0 = kvq_fit_now();
        store.synchronize();
        for (size_t li = 0; li < layer_ids.size(); ++li) {
            for (int32_t g = 0; g < G; ++g) {
                const size_t i = li*G + g;
                store.set_chunk_rows (layer_ids[li], g, false, jb->c, jb->rows[i].data());
                store.set_chunk_rows (layer_ids[li], g, true,  jb->c, jb->rows[n_pair + i].data());
                store.set_chunk_table(layer_ids[li], g, false, jb->c, jb->tinv[i].data(), jb->taux[i].data());
                if (v_fit) {
                    store.set_chunk_table(layer_ids[li], g, true, jb->c, jb->tinv[n_pair + i].data(), jb->taux[n_pair + i].data());
                }
            }
        }
        store.set_n_coarse((jb->c + 1)*store.n_chunk_rows());
        n_coarse_chunks++;

        grams.resize(n_coarse_chunks);
        grams[jb->c] = std::move(jb->gram);
        gsums.resize(n_coarse_chunks);
        gsums[jb->c] = std::move(jb->gsum);

        t_job.push_back(jb->t_end - jb->t_start);
        t_wall.push_back(jb->t_end - jb->t_take);
        t_cpu     += jb->t_cpu;
        t_install += kvq_fit_now() - t0;
        LLAMA_LOG_INFO("%s: chunk %d joined the coarse region (%d rows), job %.2f s, %.2f s CPU, %.2f s since the statistics\n", __func__,
                jb->c, store.n_coarse(), jb->t_end - jb->t_start, jb->t_cpu, jb->t_end - jb->t_take);
        release(*jb);
    }
}

void llama_kvq_fitter::poll(const llama_kv_cache & kv) {
    const int32_t C = store.n_chunk_rows();
    const double  t0 = kvq_fit_now();
    bool took = false;
    while (!failed && (int64_t) (n_taken + 1)*C <= store.n_rows() && kv.seq_pos_max(0) >= (n_taken + 1)*C - 1) {
        take(kv);
        took = true;
    }
    if (took) {
        t_take += kvq_fit_now() - t0;
    }
    install(false);
}

void llama_kvq_fitter::wait(const llama_kv_cache & kv) {
    poll(kv);
    {
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&]() {
            return std::all_of(jobs.begin(), jobs.end(), [](const std::unique_ptr<job> & jb) { return jb->n_left == 0; });
        });
    }
    install(true);
}

void llama_kvq_fitter::remove(llama_pos p0) {
    const int32_t C  = store.n_chunk_rows();
    const int32_t c0 = p0 / C;

    // the removal is in the chunk that collects statistics: they keep the removed queries, unless all of them go
    if (n_taken <= c0) {
        if (p0 == (llama_pos) c0*C && n_taken == c0) {
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
    {
        std::unique_lock<std::mutex> lk(m);
        for (auto & jb : jobs) {
            if (jb->c >= c0) {
                jb->cancel = true;
                if (jb->c == c0) {
                    gram_c0 = std::move(jb->gram);
                    gsum_c0 = std::move(jb->gsum);
                }
            }
        }
        // the queued tasks of these jobs are dropped, the running ones stop at their next check
        for (auto it = tasks.begin(); it != tasks.end();) {
            if (it->first->cancel) {
                it->first->n_left--;
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
                release(**it);
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

    store.synchronize();
    if (p0 > (llama_pos) c0*C) {
        set_stats(&gram_c0, &gsum_c0);
    } else {
        set_stats(nullptr, nullptr);
    }
    n_taken = c0;
    // a failed chunk before c0 still blocks the later ones
    if (c0 == 0) {
        failed = false;
    }
}

// decode thread: keeps the rows buffer of a finished or cancelled job for the next one
void llama_kvq_fitter::release(job & jb) {
    if (jb.buf && free_bufs.size() < 2) {
        free_bufs.push_back(std::move(jb.buf));
    }
    jb.buf.reset();
    jb.raw = nullptr;
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
    out.t_cpu    = t_cpu;
    out.t_take    = t_take;
    out.t_install = t_install;
    std::lock_guard<std::mutex> lk(m);
    for (const auto & jb : jobs) {
        if (jb->n_left > 0) {
            out.n_running++;
        } else {
            out.n_ready++;
        }
    }
}
