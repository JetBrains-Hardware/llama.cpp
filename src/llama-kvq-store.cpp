#include "llama-kvq-store.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kvq-codec.h"
#include "llama-kvq-pack.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {

struct kvq_fetch {
    int32_t g;
    int32_t slot;
    int32_t row;
};

struct kvq_seen {
    uint32_t mark; // stamp of the last request that listed the row
    int32_t  upos; // index in the union of that request
};

struct kvq_cand {
    int32_t row;
    int32_t slot;
    float   val;
};

}

struct llama_kvq_store::layer {
    int32_t il = -1;

    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;

    layer_tensors t;

    uint8_t * store = nullptr; // [G][n_rows][2*ROW], K row then V row

    std::vector<int32_t>  ridx;       // [G + 1][cap_tot], mirror of t.ridx
    std::vector<int32_t>  row_slot;   // [G][n_rows], -1 if not resident
    std::vector<kvq_seen> row_seen;   // [G][n_rows], one load per row for the union
    std::vector<uint32_t> slot_stamp; // [G][cap_tot], stamp of the last selection
    std::vector<float>    slot_val;   // [G][cap_tot], score of the last selection
    std::vector<int32_t>  prev;       // [G][cap_tot], LRU list of the evictable slots, -1 at the ends
    std::vector<int32_t>  next;
    std::vector<int32_t>  head;       // [G], least recently selected
    std::vector<int32_t>  tail;       // [G]

    uint32_t stamp    = 0; // selection clock
    uint32_t pf_stamp = 0; // stamp of the last prefetch

    // pinned host buffers
    uint8_t * h_stage = nullptr; // [2*n][ROW], K and V row of each fetch
    int32_t * h_sel   = nullptr; // selection readback
    float   * h_val   = nullptr;
    int32_t * h_ridx  = nullptr; // [G + 1][cap_tot]
    int32_t * h_slots = nullptr; // [n_stage]

    int64_t slots_hi = 0; // entries of the device slot list that are not trash

    std::vector<kvq_fetch> fetch; // of the last request

    // request of the selection in h_sel
    int64_t n_top = 0;
    int64_t n_pos = 0;

    bool pending = false; // decode thread: a prefetch was handed to the worker and is not joined yet
    bool done    = false; // worker mutex

    ggml_backend_event_ptr ev_sel;  // main stream at the prefetch trigger
    ggml_backend_event_ptr ev_read; // readback of the selection
    ggml_backend_event_ptr ev_up;   // upload from the host buffers
    bool up_rec = false;

    // LLAMA_KVQ_IDLE_PROBE: the time the GPU reached the join and the end of the join, for the GPU idle estimate
    ggml_backend_event_ptr ev_join;
    bool   probe_busy = false; // probe mutex
    double t_reach    = 0.0;
    double t_end      = 0.0;

    llama_kvq_counters cnt;

    void touch(int32_t g, int32_t s, float v, int32_t cap_tot, int32_t n_sink) {
        const size_t o = (size_t) g*cap_tot;
        slot_stamp[o + s] = stamp;
        slot_val[o + s]   = v;
        if (s < n_sink || s == tail[g]) {
            return;
        }
        // unlink, then append at the tail
        if (prev[o + s] >= 0) {
            next[o + prev[o + s]] = next[o + s];
        } else {
            head[g] = next[o + s];
        }
        prev[o + next[o + s]] = prev[o + s];
        prev[o + s] = tail[g];
        next[o + s] = -1;
        next[o + tail[g]] = s;
        tail[g] = s;
    }

    void assign(int32_t g, int32_t s, int32_t row, float v, int32_t cap_tot, int32_t n_sink, int32_t n_rows) {
        const size_t o = (size_t) g*cap_tot;
        const int32_t old = ridx[o + s];
        if (old >= 0) {
            row_slot[(size_t) g*n_rows + old] = -1;
            cnt.evict++;
        }
        ridx[o + s] = row;
        row_slot[(size_t) g*n_rows + row] = s;
        touch(g, s, v, cap_tot, n_sink);
        fetch.push_back({ g, s, row });
    }
};

static double kvq_now() {
    return ggml_time_us()*1e-6;
}

// the gather of one request over n_th threads, the helpers sleep on a condition variable
struct llama_kvq_store::gather_pool {
    std::vector<std::thread> threads;
    std::mutex              m;
    std::mutex              m_run; // one gather at a time
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    uint64_t gen    = 0;
    int      n_left = 0;
    bool     stop   = false;

    const kvq_fetch * fetch    = nullptr;
    int64_t           n        = 0;
    uint8_t *         dst      = nullptr;
    const uint8_t *   store    = nullptr;
    int64_t           n_rows   = 0;
    const int         n_th;

    void part(int ith, int nth) const {
        const int64_t i0 = n*ith/nth;
        const int64_t i1 = n*(ith + 1)/nth;
        for (int64_t i = i0; i < i1; ++i) {
            memcpy(dst + 2*i*ROW, store + ((size_t) fetch[i].g*n_rows + fetch[i].row)*2*ROW, 2*ROW);
        }
    }

    explicit gather_pool(int n_th) : n_th(n_th) {
        for (int ith = 1; ith < n_th; ++ith) {
            threads.emplace_back([this, ith]() {
                uint64_t seen = 0;
                std::unique_lock<std::mutex> lk(m);
                while (true) {
                    cv_work.wait(lk, [&]() { return stop || gen != seen; });
                    if (stop) {
                        return;
                    }
                    seen = gen;
                    lk.unlock();
                    part(ith, this->n_th);
                    lk.lock();
                    if (--n_left == 0) {
                        cv_done.notify_one();
                    }
                }
            });
        }
    }

    ~gather_pool() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    void run(const kvq_fetch * f, int64_t n_f, uint8_t * d, const uint8_t * st, int64_t n_r) {
        std::lock_guard<std::mutex> lr(m_run);
        fetch    = f;
        n        = n_f;
        dst      = d;
        store    = st;
        n_rows   = n_r;
        // waking the helpers costs more than a small copy
        if (n_th == 1 || n < 256) {
            part(0, 1);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m);
            n_left = n_th - 1;
            gen++;
        }
        cv_work.notify_all();
        part(0, n_th);
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&]() { return n_left == 0; });
    }
};

// runs the prefetch requests (plan and gather) off the decode thread, in order
// the probe thread (LLAMA_KVQ_IDLE_PROBE) waits for the join events in order and notes when the GPU reached them
struct llama_kvq_store::worker {
    llama_kvq_store & st;

    std::mutex              m;
    std::condition_variable cv_job;
    std::condition_variable cv_done;
    std::deque<layer *>     jobs;
    bool                    stop = false;
    std::thread             th;

    std::mutex              m_probe;
    std::condition_variable cv_probe;
    std::deque<layer *>     probes;
    std::thread             th_probe;

    explicit worker(llama_kvq_store & st, bool probe) : st(st), th([this]() { loop(); }) {
        if (probe) {
            th_probe = std::thread([this]() { loop_probe(); });
        }
    }

    ~worker() {
        {
            std::lock_guard<std::mutex> lk(m);
            std::lock_guard<std::mutex> lp(m_probe);
            stop = true;
        }
        cv_job.notify_one();
        cv_probe.notify_one();
        th.join();
        if (th_probe.joinable()) {
            th_probe.join();
        }
    }

    void loop_probe() {
        std::unique_lock<std::mutex> lk(m_probe);
        while (true) {
            cv_probe.wait(lk, [&]() { return stop || !probes.empty(); });
            if (probes.empty()) {
                return;
            }
            layer * l = probes.front();
            probes.pop_front();
            lk.unlock();
            ggml_backend_event_synchronize(l->ev_join.get());
            const double t = kvq_now();
            lk.lock();
            l->t_reach    = t;
            l->probe_busy = false;
        }
    }

    // GPU idle of the previous probe of the layer, then a new probe at the current position of the stream
    double probe(layer & l, ggml_backend_t backend) {
        double idle = 0.0;
        std::lock_guard<std::mutex> lk(m_probe);
        if (l.probe_busy) {
            return idle;
        }
        if (l.t_reach > 0.0) {
            idle = std::max(0.0, l.t_end - l.t_reach);
        }
        ggml_backend_event_record(l.ev_join.get(), backend);
        l.t_reach    = 0.0;
        l.probe_busy = true;
        probes.push_back(&l);
        cv_probe.notify_one();
        return idle;
    }

    void loop() {
        std::unique_lock<std::mutex> lk(m);
        while (true) {
            cv_job.wait(lk, [&]() { return stop || !jobs.empty(); });
            if (jobs.empty()) {
                return;
            }
            layer * l = jobs.front();
            jobs.pop_front();
            lk.unlock();
            st.run_prefetch(*l);
            lk.lock();
            l->done = true;
            cv_done.notify_all();
        }
    }

    void submit(layer & l) {
        {
            std::lock_guard<std::mutex> lk(m);
            l.done = false;
            jobs.push_back(&l);
        }
        cv_job.notify_one();
    }

    void wait(layer & l) {
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&]() { return l.done; });
    }
};

static int kvq_popcount(uint64_t x) {
    int n = 0;
    for (; x; x &= x - 1) {
        n++;
    }
    return n;
}

llama_kvq_store::llama_kvq_store(const llama_kv_cache & kv, int32_t n_rows, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
        int32_t resident_cap, bool all_resident, int32_t n_sink, bool fill) :
    n_coarse_(fill ? n_rows : 0), n_rows_(n_rows), filled(fill), n_crow(n_chunk_rows), all_res(all_resident) {
    if (kv.get_n_stream() != 1) {
        throw std::runtime_error("packed two-tier mode needs a single KV stream");
    }
    if (n_rows <= 0 || n_chunk_rows <= 0 || n_rows % n_chunk_rows != 0 || n_sink < 0 || n_sink >= n_chunk_rows) {
        throw std::runtime_error("packed two-tier mode: bad coarse region");
    }
    if (bits_k < 1 || bits_k > 4 || bits_v < 1 || bits_v > 4 || (!all_resident && resident_cap <= 0) || (all_resident && !fill)) {
        throw std::runtime_error("packed two-tier mode: bad bits or resident capacity");
    }

    this->n_sink = n_sink;
    n_chunk = n_rows/n_chunk_rows;
    nwk     = (bits_k*(int32_t) D + 31)/32;
    nwv     = (bits_v*(int32_t) D + 31)/32;
    cap     = all_resident ? n_rows - n_sink : resident_cap;
    cap_tot = n_sink + cap + 1;

    const auto ids = kv.get_layer_ids();
    GGML_ASSERT(!ids.empty());

    const auto & cells = kv.get_cells(0);
    if ((int64_t) cells.size() < n_rows) {
        throw std::runtime_error("packed two-tier mode: the cache is smaller than the coarse region");
    }
    for (int32_t j = 0; j < n_coarse_; ++j) {
        if (cells.is_empty(j) || cells.pos_get(j) != j || !cells.seq_has(j, 0) || cells.seq_count(j) != 1) {
            throw std::runtime_error("packed two-tier mode: the cells of the coarse region must hold positions [0, n_coarse) of sequence 0");
        }
    }

    layers.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        layer & l = layers[i];
        l.il      = (int32_t) ids[i];
        l.k_cache = kv.get_k_storage(l.il);
        l.v_cache = kv.get_v_storage(l.il);
        l.cnt.il  = l.il;
        if (l.k_cache->type != GGML_TYPE_Q8_0 || l.v_cache->type != GGML_TYPE_Q8_0 || l.k_cache->ne[0] % D != 0 || l.v_cache->ne[0] != l.k_cache->ne[0]) {
            throw std::runtime_error("packed two-tier mode needs a q8_0 K and V cache with head size 256");
        }
        const ggml_backend_buffer_type_t b = ggml_backend_buffer_get_type(l.k_cache->buffer);
        if (buft != nullptr && b != buft) {
            throw std::runtime_error("packed two-tier mode needs all attention layers on one device");
        }
        buft = b;
        layer_of_il[l.il] = i;
    }
    G   = (int32_t) (layers[0].k_cache->ne[0]/D);
    dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr || ggml_backend_buft_is_host(buft)) {
        throw std::runtime_error("packed two-tier mode needs the attention cache on a GPU");
    }
    buft_host = ggml_backend_dev_host_buffer_type(dev);
    if (buft_host == nullptr) {
        buft_host = ggml_backend_cpu_buffer_type();
    }


    auto init_ctx = [](size_t n) {
        ggml_init_params ip = { n*ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr res(ggml_init(ip));
        if (!res) {
            throw std::runtime_error("failed to create the packed two-tier context");
        }
        return res;
    };
    ctx_dev   = init_ctx(layers.size()*7 + 3);
    ctx_alias = init_ctx(layers.size());
    ctx_host  = init_ctx(layers.size()*3);

    for (auto & l : layers) {
        layer_tensors & t = l.t;
        t.kc   = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_I32, nwk, n_rows, G);
        t.vc   = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_I32, nwv + 1, n_rows, G);
        t.tinv = ggml_new_tensor_4d(ctx_dev.get(), GGML_TYPE_F16, D, D, 2*G, n_chunk);
        t.taux = ggml_new_tensor_4d(ctx_dev.get(), GGML_TYPE_F32, D, 3, 2*G, n_chunk);
        t.rkv  = ggml_new_tensor_4d(ctx_dev.get(), GGML_TYPE_Q8_0, D, cap_tot, G, 2);
        t.ridx = ggml_new_tensor_2d(ctx_dev.get(), GGML_TYPE_I32, cap_tot, G + 1);
        ggml_format_name(t.kc,   "kvq_kc-%d",   l.il);
        ggml_format_name(t.vc,   "kvq_vc-%d",   l.il);
        ggml_format_name(t.tinv, "kvq_tinv-%d", l.il);
        ggml_format_name(t.taux, "kvq_taux-%d", l.il);
        ggml_format_name(t.rkv,  "kvq_rkv-%d",  l.il);
        ggml_format_name(t.ridx, "kvq_ridx-%d", l.il);
        if (!fill) {
            t.qacc = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_F32, D, D, G);
            ggml_format_name(t.qacc, "kvq_qacc-%d", l.il);
        }
        for (int k = 0; k < 3; ++k) {
            t.trig[k] = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, 1);
            ggml_format_name(t.trig[k], "kvq_trig%d-%d", k, l.il);
            trig_of[t.trig[k]] = { (size_t) (&l - layers.data()), k };
        }
    }
    bias    = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_F32, n_rows);
    ggml_set_name(bias,    "kvq_bias");

    buf_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev.get(), buft));
    if (!buf_dev) {
        throw std::runtime_error("failed to allocate the packed two-tier device buffer");
    }
    // the query statistics start at 0, unused table slots must not hold NaN
    ggml_backend_buffer_clear(buf_dev.get(), 0);
    // as weights: the ops that read these tensors run on their device
    ggml_backend_buffer_set_usage(buf_dev.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // set_rows has no q8_0 to q8_0 copy, so the scatter writes the rows of rkv as f16 pairs of bytes
    for (auto & l : layers) {
        l.t.rkv_h = ggml_new_tensor_2d(ctx_alias.get(), GGML_TYPE_F16, ROW/2, 2*(int64_t) G*cap_tot);
        GGML_ASSERT(ggml_nbytes(l.t.rkv_h) == ggml_nbytes(l.t.rkv));
        ggml_format_name(l.t.rkv_h, "kvq_rkv_h-%d", l.il);
        if (ggml_backend_tensor_alloc(buf_dev.get(), l.t.rkv_h, l.t.rkv->data) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to alias the resident rows");
        }
    }

    // the triggers are host weights, so the scheduler starts a split at their reader and calls the copy callback
    buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), ggml_backend_cpu_buffer_type()));
    if (!buf_host) {
        throw std::runtime_error("failed to allocate the packed two-tier triggers");
    }
    ggml_backend_buffer_clear(buf_host.get(), 0);
    ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // precise store, one buffer per layer (a single pinned CUDA buffer is limited to 4 GB)
    const size_t store_size = (size_t) G*n_rows*2*ROW;
    for (auto & l : layers) {
        ggml_backend_buffer_t b = nullptr;
        if (store_size <= ggml_backend_buft_get_max_size(buft_host)) {
            b = ggml_backend_buft_alloc_buffer(buft_host, store_size);
        }
        if (b == nullptr) {
            b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), store_size);
        }
        if (b == nullptr) {
            throw std::runtime_error("failed to allocate the precise store");
        }
        buf_store.emplace_back(b);
        l.store = (uint8_t *) ggml_backend_buffer_get_base(b);
    }

    // copy the q8_0 rows of the coarse region from the cache, [cell][head][ROW] -> [head][row][K, V]
    for (auto & l : layers) {
        GGML_ASSERT(l.k_cache->nb[1] == G*ROW && l.v_cache->nb[1] == G*ROW);
    }
    if (fill) {
        const int64_t n_step = 4096;
        const size_t  nb1    = G*ROW;
        std::vector<uint8_t> tk, tv;
        for (auto & l : layers) {
            for (int64_t r0 = 0; r0 < n_rows; r0 += n_step) {
                const int64_t n = std::min<int64_t>(n_step, n_rows - r0);
                tk.resize(n*nb1);
                tv.resize(n*nb1);
                ggml_backend_tensor_get(l.k_cache, tk.data(), r0*nb1, n*nb1);
                ggml_backend_tensor_get(l.v_cache, tv.data(), r0*nb1, n*nb1);
                for (int32_t g = 0; g < G; ++g) {
                    for (int64_t r = 0; r < n; ++r) {
                        uint8_t * dst = l.store + ((size_t) g*n_rows + r0 + r)*2*ROW;
                        memcpy(dst,       tk.data() + r*nb1 + g*ROW, ROW);
                        memcpy(dst + ROW, tv.data() + r*nb1 + g*ROW, ROW);
                    }
                }
            }
        }
    }

    {
        std::vector<float> b(n_rows, 0.0f);
        std::fill(b.begin(), b.begin() + n_sink, -INFINITY);
        ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));
    }

    for (auto & l : layers) {
        l.ridx.assign((size_t) (G + 1)*cap_tot, -1);
        l.row_slot.assign((size_t) G*n_rows, -1);
        l.row_seen.assign((size_t) G*n_rows, { 0, 0 });
        l.slot_stamp.assign((size_t) G*cap_tot, 0);
        l.slot_val.assign((size_t) G*cap_tot, 0.0f);
        l.prev.assign((size_t) G*cap_tot, -1);
        l.next.assign((size_t) G*cap_tot, -1);
        l.head.assign(G, -1);
        l.tail.assign(G, -1);
        l.ev_sel.reset(ggml_backend_event_new(dev));
        l.ev_read.reset(ggml_backend_event_new(dev));
        l.ev_up.reset(ggml_backend_event_new(dev));
        if (!l.ev_sel || !l.ev_read || !l.ev_up) {
            throw std::runtime_error("packed two-tier mode needs backend events");
        }
    }

    if (!all_res) {
        const char * env  = getenv("LLAMA_KVQ_GATHER_THREADS");
        const int    n_th = env != nullptr && atoi(env) > 0 ? atoi(env) : 4;
        backend_cp.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend_cp) {
            throw std::runtime_error("failed to create the copy backend of the packed two-tier mode");
        }
        pool = std::make_unique<gather_pool>(n_th);
        probe = getenv("LLAMA_KVQ_IDLE_PROBE") != nullptr;
        for (auto & l : layers) {
            if (probe) {
                l.ev_join.reset(ggml_backend_event_new(dev));
            }
        }
        work = std::make_unique<worker>(*this, probe);
        LLAMA_LOG_INFO("%s: gather on %d threads, events %s\n", __func__, n_th, layers[0].ev_up ? "yes" : "no");
    }

    const llama_kvq_memory_info mi = memory_info();
    LLAMA_LOG_INFO("%s: n_coarse = %d of %d rows, %d chunks of %d rows, K %d / V %d bits (%d / %d words), resident slots %d + %d sinks%s\n", __func__,
            n_coarse_, n_rows, n_chunk, n_crow, bits_k, bits_v, nwk, nwv, cap, n_sink, all_res ? " (all resident)" : "");
    LLAMA_LOG_INFO("%s: device: codes %.1f MiB, tables %.1f MiB, resident %.1f MiB, other %.1f MiB; host: store %.1f MiB (%s), other %.1f MiB\n", __func__,
            mi.codes/1048576.0, mi.tables/1048576.0, mi.resident/1048576.0, mi.other_dev/1048576.0, mi.host_store/1048576.0,
            mi.pinned ? "pinned" : "pageable", mi.host_other/1048576.0);
}

llama_kvq_store::~llama_kvq_store() {
    // the worker uses the layers
    work.reset();
    pool.reset();
    if (backend_cp) {
        ggml_backend_synchronize(backend_cp.get());
    }
}

const llama_kvq_store::layer_tensors & llama_kvq_store::get_layer(int32_t il) const {
    return layers.at(layer_of_il.at(il)).t;
}

ggml_tensor * llama_kvq_store::get_ridx(ggml_context * ctx, int32_t il) const {
    ggml_tensor * r = get_layer(il).ridx;
    return ggml_view_2d(ctx, r, cap_tot, G, r->nb[1], 0);
}

ggml_tensor * llama_kvq_store::get_kc(ggml_context * ctx, int32_t il) const {
    ggml_tensor * t = get_layer(il).kc;
    return ggml_view_3d(ctx, t, t->ne[0], n_coarse_, G, t->nb[1], t->nb[2], 0);
}

ggml_tensor * llama_kvq_store::get_vc(ggml_context * ctx, int32_t il) const {
    ggml_tensor * t = get_layer(il).vc;
    return ggml_view_3d(ctx, t, t->ne[0], n_coarse_, G, t->nb[1], t->nb[2], 0);
}

std::vector<int32_t> llama_kvq_store::layer_ids() const {
    std::vector<int32_t> res;
    for (const auto & l : layers) {
        res.push_back(l.il);
    }
    return res;
}

uint8_t * llama_kvq_store::host_rows(int32_t il) const {
    return layers.at(layer_of_il.at(il)).store;
}

ggml_tensor * llama_kvq_store::get_cache(int32_t il, bool is_v) const {
    const layer & l = layers.at(layer_of_il.at(il));
    return is_v ? l.v_cache : l.k_cache;
}

void llama_kvq_store::set_backend(const std::vector<ggml_backend_t> & backends) {
    backend_main = nullptr;
    for (ggml_backend_t b : backends) {
        if (ggml_backend_get_device(b) == dev) {
            backend_main = b;
        }
    }
}

void llama_kvq_store::synchronize() {
    if (backend_main) {
        ggml_backend_synchronize(backend_main);
    }
    if (backend_cp) {
        ggml_backend_synchronize(backend_cp.get());
    }
}

void llama_kvq_store::set_chunk_rows(int32_t il, int32_t g, bool is_v, int32_t c, const int32_t * rows) {
    GGML_ASSERT(!filled && c >= 0 && c < n_chunk && (int64_t) c*n_crow >= n_coarse_ && g >= 0 && g < G);
    ggml_tensor * dst = is_v ? get_layer(il).vc : get_layer(il).kc;
    ggml_backend_tensor_set(dst, rows, g*dst->nb[2] + (size_t) c*n_crow*dst->nb[1], (size_t) n_crow*dst->nb[1]);
}

void llama_kvq_store::set_chunk_table(int32_t il, int32_t g, bool is_v, int32_t c, const ggml_fp16_t * inv, const float * aux) {
    GGML_ASSERT(!filled && c >= 0 && c < n_chunk && (int64_t) c*n_crow >= n_coarse_ && g >= 0 && g < G);
    const layer_tensors & t = get_layer(il);
    const int64_t m = (int64_t) c*2*G + 2*g + (is_v ? 1 : 0);
    ggml_backend_tensor_set(t.tinv, inv, m*t.tinv->nb[2], t.tinv->nb[2]);
    ggml_backend_tensor_set(t.taux, aux, m*t.taux->nb[2], t.taux->nb[2]);
}

void llama_kvq_store::set_n_coarse(int32_t n) {
    GGML_ASSERT(!filled && n >= 0 && n <= n_rows_ && n % n_crow == 0);
    if (n == n_coarse_) {
        return;
    }
    drop_pending();
    synchronize();
    if (n < n_coarse_) {
        // the slots of the removed rows become empty, the sinks keep their slots
        for (auto & l : layers) {
            for (int32_t g = 0; g < G; ++g) {
                for (int32_t s = n_sink; s < cap_tot; ++s) {
                    int32_t & row = l.ridx[(size_t) g*cap_tot + s];
                    if (row >= n) {
                        l.row_slot[(size_t) g*n_rows_ + row] = -1;
                        row = -1;
                    }
                }
            }
            ggml_backend_tensor_set(l.t.ridx, l.ridx.data(), 0, ggml_nbytes(l.t.ridx));
        }
    } else if (n_coarse_ == 0) {
        // the rows of the sinks are in the host store now
        for (auto & l : layers) {
            upload_pinned(l, n_sink);
        }
    }
    n_coarse_ = n;
}

bool llama_kvq_store::set_codes(int32_t il, int32_t g, bool is_v, const std::vector<llama_kvq_table> & tabs, const int8_t * codes, const float * gain) {
    const auto it = layer_of_il.find(il);
    if (!filled || it == layer_of_il.end() || g < 0 || g >= G || (int32_t) tabs.size() != n_chunk || codes == nullptr || (is_v && gain == nullptr)) {
        LLAMA_LOG_ERROR("%s: bad arguments (il = %d, g = %d, %zu tables)\n", __func__, il, g, tabs.size());
        return false;
    }
    const layer_tensors & t = layers[it->second].t;
    const int32_t nw  = is_v ? nwv : nwk;
    const int32_t nwr = nw + (is_v ? 1 : 0);

    std::vector<int32_t>     rows((size_t) n_coarse_*nwr);
    std::vector<ggml_fp16_t> inv((size_t) D*D);
    std::vector<float>       aux((size_t) 3*D);
    for (int32_t c = 0; c < n_chunk; ++c) {
        const llama_kvq_table & tab = tabs[c];
        if (tab.n != D || tab.is_v != is_v || llama_kvq_pack_words(tab) > nw) {
            LLAMA_LOG_ERROR("%s: table %d of layer %d head %d does not fit (n = %d, %d words > %d)\n", __func__, c, il, g, tab.n, llama_kvq_pack_words(tab), nw);
            return false;
        }
        const int64_t r0 = (int64_t) c*n_crow;
        llama_kvq_pack_rows(tab, nw, codes + r0*D, is_v ? gain + r0 : nullptr, n_crow, rows.data() + r0*nwr);

        llama_kvq_pack_table(tab, inv.data(), aux.data());
        const int64_t m = 2*g + (is_v ? 1 : 0);
        ggml_backend_tensor_set(t.tinv, inv.data(), (c*2*G + m)*t.tinv->nb[2], ggml_nbytes(t.tinv)/(2*G*n_chunk));
        ggml_backend_tensor_set(t.taux, aux.data(), (c*2*G + m)*t.taux->nb[2], ggml_nbytes(t.taux)/(2*G*n_chunk));
    }
    ggml_tensor * dst = is_v ? t.vc : t.kc;
    ggml_backend_tensor_set(dst, rows.data(), g*dst->nb[2], rows.size()*sizeof(int32_t));
    return true;
}

bool llama_kvq_store::enable(const llama_kvq_params & p) {
    if (p.n_sink != n_sink || (!all_res && p.resident_cap != cap) || p.all_resident != all_res) {
        LLAMA_LOG_ERROR("%s: n_sink, resident_cap and all_resident must match the allocation (%d, %d, %d)\n", __func__, n_sink, cap, all_res);
        return false;
    }
    if (p.max_batch < 1 || p.max_batch > 8 || p.lead < 0 || p.sync_mode < LLAMA_KVQ_SYNC_OFF || p.sync_mode > LLAMA_KVQ_SYNC_ALWAYS) {
        LLAMA_LOG_ERROR("%s: bad max_batch, lead or sync_mode\n", __func__);
        return false;
    }
    // fit mode: the smallest coarse region is one chunk
    const int32_t n_cand = (filled ? n_coarse_ : n_crow) - n_sink;
    if (!all_res && (p.prefetch_n < 1 || p.prefetch_n > cap || p.prefetch_n > n_cand ||
                     p.sync_m < 1 || p.sync_m > cap || p.sync_m > n_cand || p.sync_min_hits < 0 || p.sync_min_hits > p.sync_m)) {
        LLAMA_LOG_ERROR("%s: bad prefetch_n or sync_m/sync_min_hits for %d slots and %d candidates\n", __func__, cap, n_cand);
        return false;
    }
    drop_pending();
    params = p;

    // pinned host buffers of each layer, for the largest request of these parameters
    if (backend_cp) {
        ggml_backend_synchronize(backend_cp.get());
    }
    buf_pin.reset();
    if (!all_res) {
        const int64_t n_rows = 2*G*std::max(n_req(KIND_PREFETCH, p.prefetch_n, p.max_batch), n_req(KIND_TOPUP, p.sync_m, p.max_batch));
        const int64_t n_sel  = (int64_t) std::max(p.prefetch_n, p.sync_m)*G*p.max_batch;

        // a graph that is reused keeps the tensors, so they change only with the parameters that change the graph
        if (!buf_stage || n_rows != n_stage) {
            n_stage = n_rows;
            buf_stage.reset();
            ggml_init_params ip = { layers.size()*2*ggml_tensor_overhead(), nullptr, true };
            ctx_stage.reset(ggml_init(ip));
            for (auto & l : layers) {
                l.t.stage = ggml_new_tensor_2d(ctx_stage.get(), GGML_TYPE_F16, ROW/2, n_stage);
                l.t.slots = ggml_new_tensor_1d(ctx_stage.get(), GGML_TYPE_I32, n_stage);
                ggml_format_name(l.t.stage, "kvq_stage-%d", l.il);
                ggml_format_name(l.t.slots, "kvq_slots-%d", l.il);
            }
            buf_stage.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_stage.get(), buft));
            if (!buf_stage) {
                LLAMA_LOG_ERROR("%s: failed to allocate the staging tensors\n", __func__);
                return false;
            }
            ggml_backend_buffer_set_usage(buf_stage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        // the trash entry: slot cap_tot - 1 of kv head 0, K
        std::vector<int32_t> trash(n_stage, cap_tot - 1);
        for (auto & l : layers) {
            ggml_backend_tensor_set(l.t.slots, trash.data(), 0, ggml_nbytes(l.t.slots));
            l.slots_hi = 0;
        }
        auto pad = [](size_t n) { return GGML_PAD(n, 64); };
        const size_t sz_stage = pad(n_rows*ROW);
        const size_t sz_sel   = pad(n_sel*sizeof(int32_t));
        const size_t sz_ridx  = pad((size_t) (G + 1)*cap_tot*sizeof(int32_t));
        const size_t sz_slots = pad(n_stage*sizeof(int32_t));
        const size_t sz_layer = sz_stage + 2*sz_sel + sz_ridx + sz_slots;
        buf_pin.reset(ggml_backend_buft_alloc_buffer(buft_host, sz_layer*layers.size()));
        if (!buf_pin) {
            LLAMA_LOG_ERROR("%s: failed to allocate %.1f MiB of pinned host buffers\n", __func__, sz_layer*layers.size()/1048576.0);
            return false;
        }
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf_pin.get());
        for (size_t i = 0; i < layers.size(); ++i) {
            layer & l = layers[i];
            uint8_t * b = base + i*sz_layer;
            l.h_stage = b;                                        b += sz_stage;
            l.h_sel   = (int32_t *) b;                            b += sz_sel;
            l.h_val   = (float *) b;                              b += sz_sel;
            l.h_ridx  = (int32_t *) b;                            b += sz_ridx;
            l.h_slots = (int32_t *) b;
            l.up_rec  = false;
        }
    }

    for (auto & l : layers) {
        std::fill(l.ridx.begin(), l.ridx.end(), -1);
        std::fill(l.row_slot.begin(), l.row_slot.end(), -1);
        std::fill(l.row_seen.begin(), l.row_seen.end(), kvq_seen{ 0, 0 });
        std::fill(l.slot_stamp.begin(), l.slot_stamp.end(), 0);
        std::fill(l.slot_val.begin(), l.slot_val.end(), 0.0f);
        l.stamp    = 0;
        l.pf_stamp = 0;
        l.fetch.clear();

        // pinned rows: the sinks, or every row
        const int32_t n_pin = all_res ? n_coarse_ : n_sink;
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t s = 0; s < n_pin; ++s) {
                l.ridx[(size_t) g*cap_tot + s]      = s;
                l.row_slot[(size_t) g*n_rows_ + s] = s;
            }
            // LRU list of the evictable slots, in slot order
            const size_t o = (size_t) g*cap_tot;
            for (int32_t s = n_sink; s < n_sink + cap; ++s) {
                l.prev[o + s] = s > n_sink ? s - 1 : -1;
                l.next[o + s] = s + 1 < n_sink + cap ? s + 1 : -1;
            }
            l.head[g] = all_res ? -1 : n_sink;
            l.tail[g] = all_res ? -1 : n_sink + cap - 1;
        }
        ggml_backend_tensor_set(l.t.ridx, l.ridx.data(), 0, ggml_nbytes(l.t.ridx));

        // fit mode: the sink rows are uploaded with the first chunk
        if (n_coarse_ > 0) {
            upload_pinned(l, n_pin);
        }
    }

    reset_counters();

    LLAMA_LOG_INFO("%s: max_batch = %d, prefetch_n = %d, resident = %d, lead = %d, sync = %d (m = %d, min hits = %d), stats = %d, all resident = %d\n", __func__,
            p.max_batch, p.prefetch_n, cap, p.lead, p.sync_mode, p.sync_m, p.sync_min_hits, p.stats, p.all_resident);
    return true;
}

// rows [0, n_pin) of the host store to the slots [0, n_pin) of rkv, which is [kv][g][slot][ROW]
void llama_kvq_store::upload_pinned(layer & l, int32_t n_pin) {
    std::vector<uint8_t> buf((size_t) n_pin*ROW);
    for (int kv = 0; kv < 2; ++kv) {
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t s = 0; s < n_pin; ++s) {
                memcpy(buf.data() + (size_t) s*ROW, l.store + ((size_t) g*n_rows_ + s)*2*ROW + kv*ROW, ROW);
            }
            ggml_backend_tensor_set(l.t.rkv, buf.data(), ((size_t) kv*G + g)*cap_tot*ROW, buf.size());
        }
    }
}

void llama_kvq_store::drop_pending() {
    for (auto & l : layers) {
        if (l.pending) {
            drop(l);
        }
    }
}

int64_t llama_kvq_store::n_req(int k, int64_t n_top, int64_t L) const {
    // prefetch: the union over the positions is cut at two requests, the other slots keep older rows
    return std::min<int64_t>(cap, k == KIND_TOPUP ? n_top*L : n_top*std::min<int64_t>(L, 2));
}

void llama_kvq_store::flush_ubatch() {
    if (ub_tokens > 0) {
        n_tokens += ub_tokens;
        n_clean  += ub_tokens - kvq_popcount(ub_fired);
    }
    ub_tokens = 0;
    ub_fired  = 0;
}

void llama_kvq_store::begin_ubatch(int64_t n) {
    GGML_ASSERT(n <= 64);
    // a prefetch without its join, e.g. after a failed graph compute
    for (auto & l : layers) {
        if (l.pending) {
            LLAMA_LOG_WARN("%s: dropping the prefetch of layer %d\n", __func__, l.il);
            drop(l);
        }
    }
    flush_ubatch();
    ub_tokens = n;
    n_ubatch++;
}

static ggml_backend_t kvq_backend(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    ggml_backend_t backend = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(b) == dev) {
            backend = b;
        }
    }
    GGML_ASSERT(backend != nullptr);
    return backend;
}

// the callback of the trigger fills rows [0, n) of the staging tensor and the slot list, the rest of the list is the trash slot
static void kvq_build_scatter(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * rkv_h, ggml_tensor * staging, ggml_tensor * slots, int64_t n_rows, int k, int32_t il) {
    GGML_ASSERT(staging != nullptr && n_rows <= staging->ne[1]);
    ggml_tensor * src = ggml_view_2d(ctx, staging, staging->ne[0], n_rows, staging->nb[1], 0);
    ggml_tensor * idx = ggml_view_1d(ctx, slots, n_rows, 0);
    ggml_tensor * scatter = ggml_set_rows(ctx, rkv_h, src, idx);
    ggml_format_name(scatter, "kvq_scatter%d-%d", k, il);
    ggml_build_forward_expand(gf, scatter);
}

void llama_kvq_store::build_fetch(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, kind k, ggml_tensor * score, int64_t n_top) const {
    GGML_ASSERT(k == KIND_PREFETCH || k == KIND_TOPUP);
    const layer_tensors & t = get_layer(il);

    const int64_t L = score->ne[2];
    GGML_ASSERT(score->ne[0] == n_coarse_ && score->ne[1] == G);
    GGML_ASSERT(n_top <= n_coarse_ - n_sink);

    // the sinks are never candidates
    ggml_tensor * sc = ggml_add(ctx, ggml_reshape_2d(ctx, score, n_coarse_, G*L), ggml_view_1d(ctx, bias, n_coarse_, 0));

    ggml_tensor * ids = ggml_top_k(ctx, sc, n_top); // [n_top, G*L]
    ggml_format_name(ids, "kvq_ids%d-%d", (int) k, il);
    // read in the copy callback of a later split, after its last use in the graph
    ggml_set_output(ids);

    // the scores of the selected rows, to rank the union over the positions
    ggml_tensor * vals = ggml_get_rows(ctx, ggml_reshape_3d(ctx, sc, 1, n_coarse_, G*L), ids); // [1, n_top, G*L]
    ggml_format_name(vals, "kvq_vals%d-%d", (int) k, il);

    // read back from a copy stream after the trigger, so they are outputs
    ggml_set_output(vals);

    // the trigger value is not set, the result is not used
    ggml_tensor * trig = ggml_add(ctx, ggml_view_1d(ctx, vals, 1, 0), t.trig[k]);
    ggml_format_name(trig, "kvq_trig%d-%d", (int) k, il);
    ggml_backend_sched_set_tensor_backend(sched, trig, kvq_backend(sched, dev));
    ggml_build_forward_expand(gf, trig);

    if (k == KIND_TOPUP) {
        kvq_build_scatter(ctx, gf, t.rkv_h, t.stage, t.slots, 2*G*n_req(k, n_top, L), k, il);
    }
}

void llama_kvq_store::build_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, int64_t n_top, int64_t L) const {
    const layer_tensors & t = get_layer(il);

    ggml_tensor * trig = ggml_add(ctx, ggml_view_1d(ctx, bias, 1, 0), t.trig[KIND_JOIN]);
    ggml_format_name(trig, "kvq_trig%d-%d", (int) KIND_JOIN, il);
    ggml_backend_sched_set_tensor_backend(sched, trig, kvq_backend(sched, dev));
    ggml_build_forward_expand(gf, trig);

    kvq_build_scatter(ctx, gf, t.rkv_h, t.stage, t.slots, 2*G*n_req(KIND_PREFETCH, n_top, L), KIND_JOIN, il);
}

void llama_kvq_store::select(const layer & l, const ggml_tensor * dst, ggml_cgraph * graph, const ggml_tensor ** ids, const ggml_tensor ** vals) const {
    // trig = add(view(vals), trigger), vals = get_rows(scores, ids)
    ggml_cgraph * gr = graph;
    const int n_nodes = ggml_graph_n_nodes(gr);
    const ggml_tensor * trig = nullptr;
    for (int i = 0; i < n_nodes && trig == nullptr; ++i) {
        const ggml_tensor * node = ggml_graph_node(gr, i);
        if (node->op == GGML_OP_ADD && node->src[1] == dst) {
            trig = node;
        }
    }
    GGML_ASSERT(trig != nullptr);
    *vals = trig->src[0]->view_src;
    GGML_ASSERT(*vals != nullptr && (*vals)->op == GGML_OP_GET_ROWS);
    *ids = (*vals)->src[1];
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(gr, i);
        if (node == *vals || node == *ids) {
            GGML_ABORT("the selection of layer %d is computed in the same split as its trigger", l.il);
        }
    }
    GGML_ASSERT(ggml_is_contiguous(*ids) && ggml_is_contiguous(*vals) && (*ids)->type == GGML_TYPE_I32 && (*vals)->type == GGML_TYPE_F32);
}

void llama_kvq_store::plan_prefetch(layer & l) {
    llama_kvq_counters & cnt = l.cnt;

    const int64_t n_top = l.n_top;
    const int64_t L     = l.n_pos;
    const int64_t n_u   = n_req(KIND_PREFETCH, n_top, L);

    const uint32_t now = ++l.stamp;
    l.fetch.clear();

    cnt.n_pf++;
    l.pf_stamp = now;
    std::vector<kvq_cand> u;
    for (int32_t g = 0; g < G; ++g) {
        // union over the positions, the max score of a row; the rows of one position are distinct
        u.clear();
        for (int64_t p = 0; p < L; ++p) {
            const int64_t o = (g + G*p)*n_top;
            for (int64_t i = 0; i < n_top; ++i) {
                const int32_t row = l.h_sel[o + i];
                GGML_ASSERT(row >= n_sink && row < n_coarse_);
                if (L == 1) {
                    u.push_back({ row, -1, l.h_val[o + i] });
                    continue;
                }
                kvq_seen & e = l.row_seen[(size_t) g*n_rows_ + row];
                if (e.mark != now) {
                    e = { now, (int32_t) u.size() };
                    u.push_back({ row, -1, l.h_val[o + i] });
                } else {
                    u[e.upos].val = std::max(u[e.upos].val, l.h_val[o + i]);
                }
            }
        }
        if ((int64_t) u.size() > n_u) {
            std::nth_element(u.begin(), u.begin() + n_u, u.end(), [](const kvq_cand & a, const kvq_cand & b) { return a.val > b.val; });
            u.resize(n_u);
        }
        cnt.pf_req += u.size();

        // the resident rows first, so that the victims of the missing ones are older than this request
        // independent loads first, the cache misses overlap
        const int32_t * rs = l.row_slot.data() + (size_t) g*n_rows_;
        for (auto & c : u) {
            c.slot = rs[c.row];
        }
        for (const auto & c : u) {
            if (c.slot >= 0) {
                l.touch(g, c.slot, c.val, cap_tot, n_sink);
            }
        }
        for (const auto & c : u) {
            if (c.slot < 0) {
                const int32_t s = l.head[g];
                GGML_ASSERT(s >= 0 && l.slot_stamp[(size_t) g*cap_tot + s] != now);
                l.assign(g, s, c.row, c.val, cap_tot, n_sink, n_rows_);
            }
        }
    }
    cnt.pf_fetch += l.fetch.size();
}

void llama_kvq_store::plan_topup(layer & l) {
    llama_kvq_counters & cnt = l.cnt;

    const int64_t n_top = l.n_top;
    const int64_t L     = l.n_pos;
    const int32_t * sel = l.h_sel;

    const uint32_t now = ++l.stamp;
    l.fetch.clear();

    cnt.n_tu++;
    cnt.n_pos += L;

    const bool    do_fetch = params.sync_mode != LLAMA_KVQ_SYNC_OFF;
    const int64_t thr      = params.sync_mode == LLAMA_KVQ_SYNC_ALWAYS ? n_top : params.sync_min_hits;

    std::vector<int32_t> hits(G*L, 0);
    for (int64_t r = 0; r < G*L; ++r) {
        const int32_t g = (int32_t) (r % G);
        for (int64_t i = 0; i < n_top; ++i) {
            hits[r] += l.row_slot[(size_t) g*n_rows_ + sel[r*n_top + i]] >= 0;
        }
        cnt.top_hit += hits[r];
    }
    cnt.top_n    += n_top*G*L;
    cnt.tu_check += G*L;

    // slots of this step's prefetch by ascending score, the victims once the older slots are used up
    std::vector<std::vector<int32_t>> cand(G);
    std::vector<size_t> cand_pos(G, 0);
    std::vector<bool>   cand_init(G, false);
    auto victim = [&](int32_t g) {
        const size_t o = (size_t) g*cap_tot;
        const int32_t s = l.head[g];
        if (l.slot_stamp[o + s] < l.pf_stamp || l.pf_stamp == 0) {
            return s;
        }
        if (!cand_init[g]) {
            cand_init[g] = true;
            for (int32_t t = n_sink; t < n_sink + cap; ++t) {
                if (l.slot_stamp[o + t] != now) {
                    cand[g].push_back(t);
                }
            }
            std::sort(cand[g].begin(), cand[g].end(), [&](int32_t a, int32_t b) { return l.slot_val[o + a] < l.slot_val[o + b]; });
        }
        while (cand_pos[g] < cand[g].size() && l.slot_stamp[o + cand[g][cand_pos[g]]] == now) {
            cand_pos[g]++;
        }
        GGML_ASSERT(cand_pos[g] < cand[g].size());
        return cand[g][cand_pos[g]++];
    };

    uint64_t stall = 0;
    for (int64_t r = 0; r < G*L && do_fetch; ++r) {
        const int32_t g = (int32_t) (r % G);
        const int64_t p = r / G;
        const int32_t * rows = sel + r*n_top;
        for (int64_t i = 0; i < n_top; ++i) {
            const int32_t s = l.row_slot[(size_t) g*n_rows_ + rows[i]];
            if (s >= 0) {
                l.touch(g, s, INFINITY, cap_tot, n_sink);
            }
        }
        if (hits[r] >= thr) {
            continue;
        }
        cnt.tu_fire++;
        stall |= 1ull << p;
        for (int64_t i = 0; i < n_top; ++i) {
            if (l.row_slot[(size_t) g*n_rows_ + rows[i]] < 0) {
                l.assign(g, victim(g), rows[i], INFINITY, cap_tot, n_sink, n_rows_);
            }
        }
    }
    cnt.tu_stall += kvq_popcount(stall);
    ub_fired |= stall;
    cnt.tu_fetch += l.fetch.size();

    // the flag of ggml_kvq_attn_cond
    l.ridx[(size_t) G*cap_tot] = l.fetch.empty() ? 0 : 1;
    cnt.tu_redo += !l.fetch.empty();
}

// rows, slot list and ridx of the last plan to the pinned buffers of the layer
void llama_kvq_store::gather(layer & l, int64_t n_rows) {
    const int64_t n = (int64_t) l.fetch.size();
    GGML_ASSERT(2*n <= n_rows && n_rows <= n_stage);

    // the previous upload from these buffers
    if (l.up_rec) {
        ggml_backend_event_synchronize(l.ev_up.get());
        l.up_rec = false;
    }

    // sorting the rows by address does not make the copy faster (measured)
    pool->run(l.fetch.data(), n, l.h_stage, l.store, n_rows_);

    for (int64_t i = 0; i < n; ++i) {
        l.h_slots[2*i]     = (int32_t) ((0*G + l.fetch[i].g)*cap_tot + l.fetch[i].slot);
        l.h_slots[2*i + 1] = (int32_t) ((1*G + l.fetch[i].g)*cap_tot + l.fetch[i].slot);
    }
    memcpy(l.h_ridx, l.ridx.data(), l.ridx.size()*sizeof(int32_t));
}

// on the stream of backend, after the previous readers of stage, slots and ridx of the layer
void llama_kvq_store::upload(ggml_backend_t backend, layer & l) {
    const int64_t n      = (int64_t) l.fetch.size();
    const int64_t n_list = std::max<int64_t>(2*n, l.slots_hi);
    std::fill(l.h_slots + 2*n, l.h_slots + n_list, cap_tot - 1);

    if (n > 0) {
        ggml_backend_tensor_set_async(backend, l.t.stage, l.h_stage, 0, 2*n*ROW);
    }
    if (n_list > 0) {
        ggml_backend_tensor_set_async(backend, l.t.slots, l.h_slots, 0, n_list*sizeof(int32_t));
    }
    l.slots_hi = 2*n;
    ggml_backend_tensor_set_async(backend, l.t.ridx, l.h_ridx, 0, ggml_nbytes(l.t.ridx));

    ggml_backend_event_record(l.ev_up.get(), backend);
    l.up_rec = true;

    l.cnt.bytes += 2*n*ROW + n_list*sizeof(int32_t) + ggml_nbytes(l.t.ridx);
}

// worker thread
void llama_kvq_store::run_prefetch(layer & l) {
    ggml_backend_event_synchronize(l.ev_read.get());
    const double t0 = kvq_now();
    plan_prefetch(l);
    const double t1 = kvq_now();
    gather(l, 2*G*n_req(KIND_PREFETCH, l.n_top, l.n_pos));
    const double t2 = kvq_now();
    l.cnt.t_plan   += t1 - t0;
    l.cnt.t_gather += t2 - t1;
}

void llama_kvq_store::wait(layer & l) {
    GGML_ASSERT(l.pending);
    work->wait(l);
    l.pending = false;
}

// forgets the rows of a prefetch that was not joined, the device has the previous ridx
void llama_kvq_store::drop(layer & l) {
    wait(l);
    for (const auto & f : l.fetch) {
        l.ridx[(size_t) f.g*cap_tot + f.slot] = -1;
        l.row_slot[(size_t) f.g*n_rows_ + f.row] = -1;
    }
    l.fetch.clear();
    ggml_backend_synchronize(backend_cp.get());
    l.up_rec = false;
    ggml_backend_tensor_set(l.t.ridx, l.ridx.data(), 0, ggml_nbytes(l.t.ridx));
}

bool llama_kvq_store::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    const auto it = trig_of.find(src);
    if (it == trig_of.end()) {
        return false;
    }
    layer & l = layers[it->second.first];
    const int kind = it->second.second;
    llama_kvq_counters & cnt = l.cnt;

    // the time since the previous callback of this ubatch: GPU work if the scheduler waits for it, else launches
    const double t0    = kvq_now();
    const double t_gap = cb_ubatch == n_ubatch ? t0 - cb_end : 0.0;

    if (kind == KIND_JOIN) {
        if (probe) {
            cnt.t_idle += work->probe(l, backend);
        }
        // no prefetch (e.g. dropped): nothing to scatter, the slot list is trash
        if (l.pending) {
            wait(l);
        } else {
            l.fetch.clear();
            ggml_backend_event_record(l.ev_sel.get(), backend);
            ggml_backend_event_wait(backend_cp.get(), l.ev_sel.get());
        }
        const double t1 = kvq_now();
        // the copy stream is after ev_sel of this prefetch, so after the previous readers of the layer
        upload(backend_cp.get(), l);
        ggml_backend_event_wait(backend, l.ev_up.get());
        const double t2 = kvq_now();
        if (probe) {
            std::lock_guard<std::mutex> lk(work->m_probe);
            l.t_end = t2;
        }
        cnt.n_join++;
        cnt.t_join   += t1 - t0;
        cnt.t_upload += t2 - t1;
    } else {
        const ggml_tensor * ids  = nullptr;
        const ggml_tensor * vals = nullptr;
        select(l, dst, graph, &ids, &vals);

        const int64_t n_top = ids->ne[0];
        const int64_t L     = ids->ne[1]/G;
        GGML_ASSERT(n_top*G*L <= (int64_t) std::max(params.prefetch_n, params.sync_m)*G*params.max_batch);
        GGML_ASSERT(!l.pending);

        l.n_top = n_top;
        l.n_pos = L;

        if (kind == KIND_PREFETCH) {
            // the readback on the copy stream, after the selection on the main stream
            ggml_backend_t cp = backend_cp.get();
            ggml_backend_event_record(l.ev_sel.get(), backend);
            ggml_backend_event_wait(cp, l.ev_sel.get());
            ggml_backend_tensor_get_async(cp, ids,  l.h_sel, 0, ggml_nbytes(ids));
            ggml_backend_tensor_get_async(cp, vals, l.h_val, 0, ggml_nbytes(vals));
            ggml_backend_event_record(l.ev_read.get(), cp);
            l.pending = true;
            work->submit(l);
            cnt.t_read += kvq_now() - t0;
        } else {
            ggml_backend_tensor_get_async(backend, ids,  l.h_sel, 0, ggml_nbytes(ids));
            ggml_backend_tensor_get_async(backend, vals, l.h_val, 0, ggml_nbytes(vals));
            ggml_backend_synchronize(backend);
            const double t1 = kvq_now();
            plan_topup(l);
            const double t2 = kvq_now();
            gather(l, 2*G*n_req(KIND_TOPUP, n_top, L));
            const double t3 = kvq_now();
            upload(backend, l);
            const double t4 = kvq_now();
            cnt.t_read   += t1 - t0;
            cnt.t_plan   += t2 - t1;
            cnt.t_gather += t3 - t2;
            cnt.t_upload += t4 - t3;
        }
    }

    cb_end    = kvq_now();
    cb_ubatch = n_ubatch;
    cnt.t_wait += t_gap;

    return true;
}

void llama_kvq_store::get_counters(llama_kvq_counters_all & out) {
    flush_ubatch();
    out.layers.clear();
    for (const auto & l : layers) {
        out.layers.push_back(l.cnt);
    }
    out.n_ubatch = n_ubatch;
    out.n_tokens = n_tokens;
    out.n_clean  = n_clean;
}

void llama_kvq_store::reset_counters() {
    for (auto & l : layers) {
        l.cnt = {};
        l.cnt.il = l.il;
    }
    ub_tokens = 0;
    ub_fired  = 0;
    n_ubatch  = 0;
    n_tokens  = 0;
    n_clean   = 0;
}

llama_kvq_memory_info llama_kvq_store::memory_info() const {
    llama_kvq_memory_info mi;
    for (const auto & l : layers) {
        mi.codes      += ggml_nbytes(l.t.kc) + ggml_nbytes(l.t.vc);
        mi.tables     += ggml_nbytes(l.t.tinv) + ggml_nbytes(l.t.taux);
        mi.resident   += ggml_nbytes(l.t.rkv) + ggml_nbytes(l.t.ridx);
        mi.host_store += (size_t) G*n_rows_*2*ROW;
        mi.other_dev  += l.t.qacc ? ggml_nbytes(l.t.qacc) : 0;
        mi.host_other += l.ridx.size()*sizeof(int32_t) + l.row_slot.size()*sizeof(int32_t) + l.row_seen.size()*sizeof(kvq_seen) +
                         (l.slot_stamp.size() + l.slot_val.size() + l.prev.size() + l.next.size())*sizeof(int32_t);
    }
    mi.other_dev += ggml_nbytes(bias) + (buf_stage ? ggml_backend_buffer_get_size(buf_stage.get()) : 0);
    mi.host_other += buf_pin ? ggml_backend_buffer_get_size(buf_pin.get()) : 0;
    mi.pinned = !buf_store.empty() && ggml_backend_buffer_get_type(buf_store[0].get()) != ggml_backend_cpu_buffer_type();
    return mi;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kvq_store::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (buf_dev) {
        res[ggml_backend_buffer_get_type(buf_dev.get())] += ggml_backend_buffer_get_size(buf_dev.get());
    }
    for (const auto & b : buf_store) {
        res[ggml_backend_buffer_get_type(b.get())] += ggml_backend_buffer_get_size(b.get());
    }
    return res;
}
