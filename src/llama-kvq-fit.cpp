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

    std::atomic<bool> cancel{false};
    std::atomic<bool> failed{false};

    // pool mutex
    int32_t n_left  = 0; // tasks not finished
    int32_t n_run   = 0; // tasks running
    double  t_take  = 0.0;
    double  t_start = 0.0;
    double  t_end   = 0.0;
    double  t_cpu   = 0.0;

    // per task i = layer index*G + kv head
    std::vector<std::vector<int32_t>>     kc;   // [n_crow][nwk]
    std::vector<std::vector<int32_t>>     vc;   // [n_crow][nwv + 1]
    std::vector<std::vector<ggml_fp16_t>> kinv; // [D*D]
    std::vector<std::vector<float>>       kaux; // [3*D]
};

llama_kvq_fitter::llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams, const std::vector<llama_kvq_table> & v_tabs) :
    store(store), fparams(fparams), layer_ids(store.layer_ids()) {
    const int32_t G     = store.n_head_kv();
    const int32_t nwv   = store.n_words(true);
    const int32_t n_chk = store.n_rows()/store.n_chunk_rows();

    // the V tables are the same for every chunk, so all table slots get them now
    std::vector<ggml_fp16_t> inv(KVQ_D*KVQ_D);
    std::vector<float>       aux(3*KVQ_D);
    for (int32_t il : layer_ids) {
        for (int32_t g = 0; g < G; ++g) {
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
    LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits fitted per chunk, V %d bits from %zu offline tables, %d threads\n", __func__,
            store.n_chunk_rows(), n_chk, fparams.bits_k, fparams.bits_v, v_tab.size(), fparams.n_threads);
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

// pool thread: one (layer, kv head) of a chunk
void llama_kvq_fitter::run(job & jb, int32_t i) {
    const int32_t G    = store.n_head_kv();
    const int32_t li   = i / G;
    const int32_t g    = i % G;
    const int32_t il   = layer_ids[li];
    const int64_t C    = store.n_chunk_rows();
    const int64_t r0   = (int64_t) jb.c*C;
    const size_t  nb1  = (size_t) G*KVQ_ROW;
    const int32_t nwk  = store.n_words(false);
    const int32_t nwv  = store.n_words(true);

    const auto * q8 = ggml_get_type_traits(GGML_TYPE_Q8_0);

    // q8_0 rows of the chunk: cache [cell][head][ROW] -> host store [head][row][K, V], and f32 rows of this head
    std::vector<uint8_t> raw(C*nb1);
    std::vector<float>   x[2];
    uint8_t * hs = store.host_rows(il) + ((size_t) g*store.n_rows() + r0)*2*KVQ_ROW;
    for (int kv = 0; kv < 2; ++kv) {
        ggml_backend_tensor_get(store.get_cache(il, kv == 1), raw.data(), r0*nb1, C*nb1);
        x[kv].resize(C*KVQ_D);
        for (int64_t r = 0; r < C; ++r) {
            const uint8_t * src = raw.data() + r*nb1 + g*KVQ_ROW;
            memcpy(hs + r*2*KVQ_ROW + kv*KVQ_ROW, src, KVQ_ROW);
            q8->to_float(src, x[kv].data() + r*KVQ_D, KVQ_D);
        }
    }
    if (jb.cancel) {
        return;
    }

    // K metric: second moment of the queries of the chunk, the fit does not depend on its scale
    std::vector<double> metric(KVQ_D*KVQ_D);
    const float * gr = jb.gram.data() + (size_t) i*KVQ_D*KVQ_D;
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

    llama_kvq_table tab;
    if (!llama_kvq_fit(tab, x[0].data(), C, KVQ_D, metric.data(), (float) fparams.bits_k, false, 1) || llama_kvq_pack_words(tab) > nwk) {
        LLAMA_LOG_ERROR("%s: chunk %d: the K fit of layer %d, kv head %d failed\n", __func__, jb.c, il, g);
        jb.failed = true;
        return;
    }
    tab.il   = il;
    tab.head = g;
    if (jb.cancel) {
        return;
    }

    std::vector<int8_t> codes(C*KVQ_D);
    std::vector<float>  gain(C);

    llama_kvq_encode(tab, x[0].data(), C, codes.data(), nullptr, 1);
    jb.kc[i].resize(C*nwk);
    llama_kvq_pack_rows(tab, nwk, codes.data(), nullptr, C, jb.kc[i].data());
    jb.kinv[i].resize(KVQ_D*KVQ_D);
    jb.kaux[i].resize(3*KVQ_D);
    llama_kvq_pack_table(tab, jb.kinv[i].data(), jb.kaux[i].data());

    const llama_kvq_table & vt = v_tab[i];
    llama_kvq_encode(vt, x[1].data(), C, codes.data(), gain.data(), 1);
    jb.vc[i].resize(C*(nwv + 1));
    llama_kvq_pack_rows(vt, nwv, codes.data(), gain.data(), C, jb.vc[i].data());
}

// decode thread
void llama_kvq_fitter::take(const llama_kv_cache & kv) {
    const int32_t c  = n_taken;
    const int32_t C  = store.n_chunk_rows();
    const int32_t G  = store.n_head_kv();
    const size_t  nn = (size_t) KVQ_D*KVQ_D*G;

    const auto & cells = kv.get_cells(0);
    for (int32_t j = c*C; j < (c + 1)*C; ++j) {
        if (cells.is_empty(j) || cells.pos_get(j) != j || !cells.seq_has(j, 0) || cells.seq_count(j) != 1) {
            LLAMA_LOG_ERROR("%s: cell %d does not hold position %d of sequence 0 only, the fit of the coarse region stops\n", __func__, j, j);
            failed = true;
            return;
        }
    }

    auto jb = std::make_unique<job>();
    jb->c = c;
    jb->gram.resize(nn*layer_ids.size());

    // the sums of the ubatches of this chunk are computed, then they start again from 0
    store.synchronize();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        ggml_tensor * acc = store.get_layer(layer_ids[li]).qacc;
        ggml_backend_tensor_get(acc, jb->gram.data() + li*nn, 0, ggml_nbytes(acc));
        ggml_backend_tensor_memset(acc, 0, 0, ggml_nbytes(acc));
    }

    const int32_t n_task = (int32_t) layer_ids.size()*G;
    jb->kc.resize(n_task);
    jb->vc.resize(n_task);
    jb->kinv.resize(n_task);
    jb->kaux.resize(n_task);
    jb->n_left = n_task;
    jb->t_take = kvq_fit_now();

    {
        std::lock_guard<std::mutex> lk(m);
        for (int32_t i = 0; i < n_task; ++i) {
            tasks.emplace_back(jb.get(), i);
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
    const int32_t G = store.n_head_kv();
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
                store.set_chunk_rows (layer_ids[li], g, false, jb->c, jb->kc[i].data());
                store.set_chunk_rows (layer_ids[li], g, true,  jb->c, jb->vc[i].data());
                store.set_chunk_table(layer_ids[li], g, false, jb->c, jb->kinv[i].data(), jb->kaux[i].data());
            }
        }
        store.set_n_coarse((jb->c + 1)*store.n_chunk_rows());
        n_coarse_chunks++;

        grams.resize(n_coarse_chunks);
        grams[jb->c] = std::move(jb->gram);

        t_job.push_back(jb->t_end - jb->t_start);
        t_wall.push_back(jb->t_end - jb->t_take);
        t_cpu     += jb->t_cpu;
        t_install += kvq_fit_now() - t0;
        LLAMA_LOG_INFO("%s: chunk %d joined the coarse region (%d rows), job %.2f s, %.2f s CPU, %.2f s since the statistics\n", __func__,
                jb->c, store.n_coarse(), jb->t_end - jb->t_start, jb->t_cpu, jb->t_end - jb->t_take);
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
            for (int32_t il : layer_ids) {
                ggml_tensor * acc = store.get_layer(il).qacc;
                ggml_backend_tensor_memset(acc, 0, 0, ggml_nbytes(acc));
            }
        }
        if (p0 == 0) {
            failed = false;
        }
        return;
    }

    // the statistics of chunk c0, to collect the rest of it again
    std::vector<float> gram_c0;
    if (c0 < (int32_t) grams.size()) {
        gram_c0 = std::move(grams[c0]);
    }
    {
        std::unique_lock<std::mutex> lk(m);
        for (auto & jb : jobs) {
            if (jb->c >= c0) {
                jb->cancel = true;
                if (jb->c == c0) {
                    gram_c0 = std::move(jb->gram);
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

    const size_t nn = (size_t) KVQ_D*KVQ_D*store.n_head_kv();
    store.synchronize();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        ggml_tensor * acc = store.get_layer(layer_ids[li]).qacc;
        if (p0 > (llama_pos) c0*C && gram_c0.size() == nn*layer_ids.size()) {
            ggml_backend_tensor_set(acc, gram_c0.data() + li*nn, 0, ggml_nbytes(acc));
        } else {
            ggml_backend_tensor_memset(acc, 0, 0, ggml_nbytes(acc));
        }
    }
    n_taken = c0;
    // a failed chunk before c0 still blocks the later ones
    if (c0 == 0) {
        failed = false;
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
