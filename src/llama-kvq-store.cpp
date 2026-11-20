#include "llama-kvq-store.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"

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
    int32_t st = 0; // the stream (sequence of the ubatch) of the slot
};

struct kvq_cand {
    int32_t row;
    float   val;
};

bool kvq_bit(const uint64_t * b, int32_t row) {
    return (b[row >> 6] >> (row & 63)) & 1;
}

void kvq_bit_set(uint64_t * b, int32_t row) {
    b[row >> 6] |= 1ull << (row & 63);
}

void kvq_bit_clear(uint64_t * b, int32_t row) {
    b[row >> 6] &= ~(1ull << (row & 63));
}

}

struct llama_kvq_store::layer {
    int32_t il = -1;

    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;

    layer_tensors t;

    size_t li = 0; // index in layers, for the host pages

    // the resident set of the layer is per sequence (layer_seq, see ls)

    // plan scratch
    std::vector<int32_t> miss;     // requested rows that are not resident, by row
    std::vector<int32_t> free_s;   // slots whose row is not requested, by slot
    std::vector<uint8_t> free_age; // their age in requests, 255 for 255 and more and for empty slots
    std::vector<int32_t> pick;     // the free slots that take the missing rows
    std::vector<kvq_cand> cand;    // cut: the union with the max score of each row

    // pinned host buffers
    int32_t * h_sel   = nullptr; // selection readback
    float   * h_val   = nullptr;

    // resident rows: the device buffer (bufs), rotating or own; the mirror [2][G][cap_tot][ROW] (pinned) is per sequence
    int32_t   buf     = -1;
    bool      own     = false;
    bool      own_stale = true;  // own buffer: the scratch used its memory, the mirror goes back in (dec_alloc, or the next join)
    bool      own_sinks = false; // own buffer: the sink rows of the mirror changed (reset), the next join uploads them
    std::vector<int32_t> base_next; // rotating: the layers whose base uploads the worker queues after the prefetch job of this layer
    bool      base_early = false;    // rotating: queued in the previous round, or at the first prefetch of the round when it is not
    bool      base_rec  = false; // the base upload for the next join of the layer is queued (ev_base), the join waits for it
    bool      mir_busy = false; // worker mutex: the fetched rows of the last prefetch are not in the mirror yet
    bool      joined   = false; // worker mutex: the join of the last prefetch has queued its uploads (or the prefetch was dropped)

    // host ring: the tail buffer of the layer, the layer whose tail upload the worker queues after the join of this one,
    // the previous user of the tail buffer (its readback comes before the upload)
    int32_t   tbuf    = -1;
    int32_t   tb_next = -1;
    int32_t   tb_prev = -1;
    bool      tb_rec  = false; // the tail upload for the next round of the layer is queued (ev_tb), the join of the layer before waits for it
    bool      tb_ready = false; // the main stream waited for it (layer 0: for the next round)
    ggml_backend_event_ptr ev_tb;  // tail upload
    ggml_backend_event_ptr ev_cpy; // main stream at the join, after the new rows went into the tail buffer
    ggml_backend_event_ptr ev_rb;  // copy stream after the readback of the new rows

    std::vector<kvq_fetch> fetch; // of the last request
    int32_t plan_s = 0;           // the stream the plan is making (the worker)

    // request of the selection in h_sel
    int64_t n_top = 0;
    int64_t n_pos = 0;

    bool pending = false; // decode thread: a prefetch was handed to the worker and is not joined yet
    bool done    = false; // worker mutex

    ggml_backend_event_ptr ev_sel;  // main stream at the prefetch trigger
    ggml_backend_event_ptr ev_read; // readback of the selection
    ggml_backend_event_ptr ev_up;   // upload from the host buffers
    ggml_backend_event_ptr ev_base; // base upload of the mirror into the buffer
    bool up_rec = false;

    // scratch: main stream at the scratch up, end of the uploads
    ggml_backend_event_ptr ev_su;
    ggml_backend_event_ptr ev_sd;

    llama_kvq_counters cnt;

    // the slot s of kv head g takes the row, in the resident set q of the layer
    void assign(layer_seq & q, int32_t g, int32_t s, int32_t row, int32_t cap_tot, int64_t n_bmw) {
        const size_t o = (size_t) g*cap_tot;
        uint64_t * rb = q.res.data() + g*n_bmw;
        const int32_t old = q.ridx[o + s];
        if (old >= 0) {
            kvq_bit_clear(rb, old);
            cnt.evict++;
        }
        q.ridx[o + s] = row;
        kvq_bit_set(rb, row);
        q.slot_stamp[o + s] = q.stamp;
        fetch.push_back({ g, s, row, plan_s });
    }

    // the slot becomes empty
    static void forget(layer_seq & q, int32_t g, int32_t s, int32_t cap_tot, int64_t n_bmw) {
        int32_t & row = q.ridx[(size_t) g*cap_tot + s];
        if (row >= 0) {
            kvq_bit_clear(q.res.data() + g*n_bmw, row);
            row = -1;
        }
    }
};

llama_kvq_store::layer_seq & llama_kvq_store::ls(const layer & l) {
    return seqs[cur].lay[l.li];
}

llama_kvq_store::layer_seq & llama_kvq_store::ls(int32_t s, const layer & l) {
    return seqs[s].lay[l.li];
}

static double kvq_now() {
    return ggml_time_us()*1e-6;
}

// by row (distinct rows), LSD radix sort with 8-bit digits; std::sort cost 1.5 ms per 4-token verify ubatch in the server (measured)
static void kvq_sort_rows(std::vector<kvq_cand> & u, int32_t n_rows) {
    thread_local std::vector<kvq_cand> tmp;
    tmp.resize(u.size());
    for (int sh = 0; sh < 32 && (n_rows - 1) >> sh > 0; sh += 8) {
        uint32_t cnt[257] = { 0 };
        for (const kvq_cand & c : u) {
            cnt[((uint32_t) c.row >> sh & 255) + 1]++;
        }
        for (int i = 0; i < 256; ++i) {
            cnt[i + 1] += cnt[i];
        }
        for (const kvq_cand & c : u) {
            tmp[cnt[(uint32_t) c.row >> sh & 255]++] = c;
        }
        u.swap(tmp);
    }
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
    const llama_kvq_store * st = nullptr;
    int32_t           seq      = 0;
    size_t            li       = 0;
    const int         n_th;

    void part(int ith, int nth) const {
        const int64_t i0 = n*ith/nth;
        const int64_t i1 = n*(ith + 1)/nth;
        for (int64_t i = i0; i < i1; ++i) {
            memcpy(dst + 2*i*ROW, st->row_ptr(st->ub_seq[fetch[i].st], li, fetch[i].g, fetch[i].row), 2*ROW);
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

    void run(const kvq_fetch * f, int64_t n_f, uint8_t * d, const llama_kvq_store * s, int32_t sq, size_t l) {
        std::lock_guard<std::mutex> lr(m_run);
        fetch    = f;
        n        = n_f;
        dst      = d;
        st       = s;
        seq      = sq;
        li       = l;
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
struct llama_kvq_store::worker {
    llama_kvq_store & st;

    std::mutex              m;
    std::condition_variable cv_job;
    std::condition_variable cv_done;
    std::deque<layer *>     jobs;
    bool                    stop = false;
    std::thread             th;

    explicit worker(llama_kvq_store & st) : st(st), th([this]() { loop(); }) {}

    ~worker() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv_job.notify_one();
        cv_done.notify_all();
        th.join();
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
            // the base upload of the next layer right after the uploads of the join (the copy engine takes them in order), then the mirror;
            // the staging stays until the next job
            // measured: a base upload queued before the join uploads delays them, one queued after the mirror or after the GPU passed the join costs more
            if (l->mir_busy) {
                cv_done.wait(lk, [&]() { return stop || l->joined; });
                const bool go = !stop;
                lk.unlock();
                if (go) {
                    for (int32_t j : l->base_next) {
                        st.issue_base(st.layers[j], l->ev_sel.get(), st.backend_bu.get());
                    }
                    if (l->tb_next >= 0) {
                        st.issue_tail(st.layers[l->tb_next], l->ev_sel.get(), st.backend_bu.get());
                    }
                }
                st.write_mirror(*l);
                lk.lock();
                l->mir_busy = false;
                cv_done.notify_all();
            }
        }
    }

    void submit(layer & l) {
        {
            std::lock_guard<std::mutex> lk(m);
            l.done     = false;
            l.mir_busy = true;
            l.joined   = false;
            jobs.push_back(&l);
        }
        cv_job.notify_one();
    }

    void wait(layer & l) {
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&]() { return l.done; });
    }

    void mark_joined(layer & l) {
        {
            std::lock_guard<std::mutex> lk(m);
            l.joined = true;
        }
        cv_done.notify_all();
    }

    void wait_mirror(layer & l) {
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&]() { return !l.mir_busy; });
    }
};

static int kvq_popcount(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0full;
    return (int) ((x*0x0101010101010101ull) >> 56);
}

llama_kvq_store::llama_kvq_store(const llama_kv_cache & kv, int32_t n_rows, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
        int32_t resident_cap, int32_t n_sink, int32_t layer_bufs, int32_t n_gate, size_t shared_min, bool host_prealloc, size_t host_max_bytes, size_t host_keep_bytes) :
    n_rows_(n_rows), n_crow(n_chunk_rows) {
    if (kv.get_n_stream() != 1 && kv.get_n_stream() != kv.get_n_seq_max()) {
        throw std::runtime_error("packed two-tier mode needs one KV stream or one per sequence");
    }
    if (!kv.get_ring()) {
        throw std::runtime_error("packed two-tier mode needs the attention cache as a ring (llama_kvq_config::tail)");
    }
    if (n_rows <= 0 || n_chunk_rows <= 0 || n_sink < 0 || n_sink >= n_chunk_rows) {
        throw std::runtime_error("packed two-tier mode: bad coarse region");
    }
    if (bits_k < 1 || bits_k > 4 || bits_v < 1 || bits_v > 4 || resident_cap <= 0) {
        throw std::runtime_error("packed two-tier mode: bad bits or resident capacity");
    }

    this->n_sink = n_sink;
    n_chunk = (n_rows + n_chunk_rows - 1)/n_chunk_rows;
    nwk     = (bits_k*(int32_t) D + 31)/32;
    nwv     = (bits_v*(int32_t) D + 31)/32;
    cap     = resident_cap;
    cap_tot = n_sink + cap + 1;
    n_bmw   = (n_rows + 63)/64;

    const auto ids = kv.get_layer_ids();
    GGML_ASSERT(!ids.empty());

    const auto & cells = kv.get_cells(0);

    // one state per sequence of the memory (the page table, the boundaries, the resident set, the query sums)
    const int32_t n_sq = (int32_t) std::max<uint32_t>(1, kv.get_n_seq_max());
    seqs.resize(n_sq);
    // the ring cells of a sequence are in its stream of the cache
    ring_strm.resize(n_sq);
    for (int32_t i = 0; i < n_sq; ++i) {
        ring_strm[i] = kv.get_n_stream() == 1 ? 0 : (int32_t) kv.get_stream(i);
    }
    n_ring_strm = (int32_t) kv.get_n_stream();
    // every sequence can hold one partial page of its own besides the pages of the longest one
    const int32_t n_pg = n_chunk + n_sq - 1;

    layers.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        layer & l = layers[i];
        l.il      = (int32_t) ids[i];
        l.k_cache = kv.get_k_storage(l.il);
        l.v_cache = kv.get_v_storage(l.il);
        l.cnt.il  = l.il;
        l.li      = i;
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
    // the attention layers rotate through n_buf device buffers; with no fewer buffers than layers (or layer_bufs < 0) each has its own
    const int32_t n_att = (int32_t) layers.size();
    n_rot = layer_bufs >= 0 && layer_bufs < n_att ? n_att : 0;
    n_buf = n_rot > 0 ? layer_bufs : 0;
    if (n_rot > 0 && n_buf < 2) {
        throw std::runtime_error("packed two-tier mode needs at least 2 rotating resident buffers");
    }
    {
        // test knob: every base upload k times, emulates a slower bus (the cache must work on PCIe 4.0 / 3.0)
        const char * env_ur = getenv("LLAMA_KVQ_UP_REPEAT");
        up_rep   = env_ur != nullptr && atoi(env_ur) > 0 ? atoi(env_ur) : 1;
    }
    const int32_t n_bufs = n_rot > 0 ? n_buf : n_att;
    for (int32_t i = 0; i < n_att; ++i) {
        layers[i].own = n_rot == 0;
        layers[i].buf = layers[i].own ? i : i % n_buf;
    }
    // the base upload of layer j goes after the prefetch job of the layer after the previous reader of its buffer (layer j - n_buf + 1);
    // the first layers of a round: after the last reader in the previous round, so a job of the previous round or the first prefetch
    for (int32_t j = 0; j < n_rot; ++j) {
        layer & l = layers[j];
        int32_t prev = j - n_buf;
        if (prev < 0) {
            prev = j + ((n_rot - 1 - j)/n_buf)*n_buf;
        }
        int32_t iss = prev + 1;
        l.base_early = prev >= j;
        if (iss == n_rot) {
            iss = 0;
            l.base_early = j == 0;
        }
        if (iss != j) {
            layers[iss].base_next.push_back(j);
        }
    }

    // host ring: the tail upload of layer j after the join of layer j - 3 (cyclic; ev_sel of that prefetch is after the attention of layer
    // j - 4, the previous user of the buffer), two layers before the join of layer j - 1 waits for it, so it does not queue behind the
    // resident base on a slow bus; with a layer count that is not a multiple of N_TBUF the joins queue the uploads themselves
    n_cell = (int64_t) cells.size();
    {
        const bool cyc = n_att % N_TBUF == 0;
        for (int32_t j = 0; j < n_att; ++j) {
            layers[j].tbuf    = j % N_TBUF;
            layers[j].tb_prev = (j - N_TBUF + n_att) % n_att;
            if (cyc) {
                layers[(j - 3 + n_att) % n_att].tb_next = j;
            }
        }
    }

    ctx_dev   = init_ctx(layers.size()*8 + 5 + 2*N_TBUF + (size_t) n_sq*(2*layers.size() + 1));
    ctx_codes = init_ctx(layers.size()*4);
    ctx_alias = init_ctx(n_bufs);
    n_str = std::min<int32_t>(n_sq, GGML_KVQ_MAX_STREAM);
    ctx_host  = init_ctx(layers.size()*KIND_COUNT);

    // the device buffers of the packed path share the memory of the scratch (the prompt path), see scr_alloc and dec_alloc:
    // the rotating and tail buffers hold nothing from one round to the next, an own buffer gets its mirror back after a scratch
    ctx_dec = init_ctx(n_bufs + 2*N_TBUF);
    ggml_context * ctx_res = ctx_dec.get();

    // the resident rows (rkv, an alias at its start) and the trash rows of the scatter, per device buffer
    bufs.resize(n_bufs);
    // the layers share one ridx with the staging (enable)
    for (int32_t b = 0; b < n_bufs; ++b) {
        // one slice of 2*G*cap_tot rows per stream of a packed ubatch (K and V of the slots of each kv head)
        bufs[b].rkv_h = ggml_new_tensor_2d(ctx_res, GGML_TYPE_F32, ROW/4, 2*(int64_t) G*cap_tot*n_str + N_TRASH);
        ggml_format_name(bufs[b].rkv_h, "kvq_rkv_h-b%d", b);
    }
    tbufs.resize(N_TBUF);
    for (int32_t b = 0; b < N_TBUF; ++b) {
        // the cells of the ring of stream s of a packed ubatch are the cells [s*n_cell, (s + 1)*n_cell)
        tbufs[b].k = ggml_new_tensor_2d(ctx_res, GGML_TYPE_Q8_0, D*G, n_cell*n_str);
        tbufs[b].v = ggml_new_tensor_2d(ctx_res, GGML_TYPE_Q8_0, D*G, n_cell*n_str);
        ggml_format_name(tbufs[b].k, "kvq_tail_k-b%d", b);
        ggml_format_name(tbufs[b].v, "kvq_tail_v-b%d", b);
    }

    for (auto & l : layers) {
        layer_tensors & t = l.t;
        t.kc   = ggml_new_tensor_4d(ctx_codes.get(), GGML_TYPE_I32, nwk, n_crow, G, n_pg);
        t.vc   = ggml_new_tensor_4d(ctx_codes.get(), GGML_TYPE_I32, nwv + 1, n_crow, G, n_pg);
        t.tab  = ggml_new_tensor_3d(ctx_codes.get(), GGML_TYPE_I8, TAB_SLOT, 2*G, n_pg);
        t.rkv_h = bufs[l.buf].rkv_h;
        ggml_format_name(t.kc,    "kvq_kc-%d",    l.il);
        ggml_format_name(t.vc,    "kvq_vc-%d",    l.il);
        ggml_format_name(t.tab,   "kvq_tab-%d",   l.il);
        for (auto & q : seqs) {
            q.qacc.resize(layers.size(), nullptr);
            q.gacc.resize(layers.size(), nullptr);
            q.qacc[l.li] = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_F32, D, D, G);
            ggml_format_name(q.qacc[l.li], "kvq_qacc-%d", l.il);
            if (n_gate > 0) {
                q.gacc[l.li] = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_F32, n_gate);
                ggml_format_name(q.gacc[l.li], "kvq_gacc-%d", l.il);
            }
        }
        for (int k = 0; k < KIND_COUNT; ++k) {
            t.trig[k] = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_F32, 1);
            ggml_format_name(t.trig[k], "kvq_trig%d-%d", k, l.il);
            trig_of[t.trig[k]] = { (size_t) (&l - layers.data()), k };
        }
    }
    bias    = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_F32, n_rows);
    ggml_set_name(bias,    "kvq_bias");
    for (auto & q : seqs) {
        q.pt = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_I32, n_chunk);
        ggml_set_name(q.pt, "kvq_pt");
    }

    // the scratch of the large ubatches, one row per position of the coarse region, then one per ring cell (see scr_alloc)
    n_scr   = n_rows + cells.size();
    ctx_scr = init_ctx(1);
    scr     = ggml_new_tensor_4d(ctx_scr.get(), GGML_TYPE_Q8_0, D, 2, n_scr, G);
    ggml_set_name(scr, "kvq_scr");

    buf_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev.get(), buft));
    if (!buf_dev) {
        throw std::runtime_error("failed to allocate the packed two-tier device buffer");
    }
    // the statistics start at 0, unused table slots must not hold NaN
    ggml_backend_buffer_clear(buf_dev.get(), 0);
    // as weights: the ops that read these tensors run on their device
    ggml_backend_buffer_set_usage(buf_dev.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // every write of the codes and tables goes to a pinned host copy; the device copy exists only in the decode phase with
    // prefill_free_codes (codes_alloc, codes_free), otherwise from enable on
    // the host copy and the precise store are the pages of the pool, one per chunk
    {
        llama_kvq_pool::layout pl;
        pl.n_layers  = (int32_t) layers.size();
        pl.G         = G;
        pl.C         = n_crow;
        pl.n_rows    = n_rows;
        pl.row_b     = 2*ROW;
        pl.kc_row    = layers[0].t.kc->nb[1];
        pl.vc_row    = layers[0].t.vc->nb[1];
        pl.tab_slot  = layers[0].t.tab->nb[1];
        GGML_ASSERT(pl.tab_slot == TAB_SLOT && TAB_SLOT % sizeof(float) == 0);
        if (host_prealloc && host_max_bytes > 0) {
            LLAMA_LOG_WARN("%s: --kvq-host-prealloc allocates every page at load, --kvq-host-max-mib is ignored\n", __func__);
            host_max_bytes = 0;
        }
        pages.reset(new llama_kvq_pool(buft_host, pl, n_pg, host_prealloc, host_max_bytes, host_keep_bytes));
        // no sequence has a page yet: the device page tables are zero (buf_dev is cleared)
        for (auto & q : seqs) {
            q.page_tab.assign(n_chunk, -1);
            q.n_page = 0;
        }
    }
    // set_rows has no q8_0 to q8_0 copy, so the scatter writes the rows of rkv as groups of 4 bytes (f32, half the threads of f16)
    for (size_t b = 0; b < bufs.size(); ++b) {
        bufs[b].rkv = ggml_new_tensor_4d(ctx_alias.get(), GGML_TYPE_Q8_0, D, cap_tot, G, 2*n_str);
        GGML_ASSERT(ggml_nbytes(bufs[b].rkv) == (size_t) 2*G*cap_tot*n_str*ROW);
        ggml_format_name(bufs[b].rkv, "kvq_rkv-b%d", (int) b);
    }
    for (auto & l : layers) {
        l.t.rkv = bufs[l.buf].rkv;
    }

    // the mirrors of the layers, zeroed: a slot that is not resident is skipped by the attention, but must not hold NaN
    {
        const size_t sz_mir = (size_t) 2*G*cap_tot*ROW;
        for (auto & q : seqs) {
            q.buf_mir.reset(ggml_backend_buft_alloc_buffer(buft_host, sz_mir*n_att));
            if (!q.buf_mir) {
                throw std::runtime_error("failed to allocate the mirrors of the resident rows");
            }
            ggml_backend_buffer_clear(q.buf_mir.get(), 0);
            uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(q.buf_mir.get());
            q.lay.resize(n_att);
            for (int32_t i = 0; i < n_att; ++i) {
                q.lay[i].mirror = base + i*sz_mir;
            }
        }
    }

    {
        // the resident and tail buffers after each other, the staging block after them (enable), the scratch over all of it
        const size_t align = ggml_backend_buft_get_alignment(buft);
        std::vector<std::pair<ggml_tensor *, size_t>> place;
        size_t off = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_dec.get()); t != nullptr; t = ggml_get_next_tensor(ctx_dec.get(), t)) {
            place.emplace_back(t, off);
            off += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
        }
        off_jblk = off;
        buf_scr.reset(ggml_backend_buft_alloc_buffer(buft, std::max({ ggml_backend_buft_get_alloc_size(buft, scr), off, shared_min })));
        if (!buf_scr) {
            throw std::runtime_error("failed to allocate the scratch");
        }
        ggml_backend_buffer_clear(buf_scr.get(), 0);
        ggml_backend_buffer_set_usage(buf_scr.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf_scr.get());
        bool ok = ggml_backend_tensor_alloc(buf_scr.get(), scr, base) == GGML_STATUS_SUCCESS;
        for (const auto & [t, o] : place) {
            ok = ok && ggml_backend_tensor_alloc(buf_scr.get(), t, base + o) == GGML_STATUS_SUCCESS;
        }
        for (auto & b : bufs) {
            ok = ok && ggml_backend_tensor_alloc(b.rkv_h->buffer, b.rkv, b.rkv_h->data) == GGML_STATUS_SUCCESS;
        }
        if (!ok) {
            throw std::runtime_error("failed to place the scratch and the buffers of the packed path");
        }
    }

    // the triggers are host weights, so the scheduler starts a split at their reader and calls the copy callback
    buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), ggml_backend_cpu_buffer_type()));
    if (!buf_host) {
        throw std::runtime_error("failed to allocate the packed two-tier triggers");
    }
    ggml_backend_buffer_clear(buf_host.get(), 0);
    ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (auto & l : layers) {
        GGML_ASSERT(l.k_cache->nb[1] == G*ROW && l.v_cache->nb[1] == G*ROW);
        GGML_ASSERT(l.k_cache->nb[2] == (size_t) n_cell*G*ROW && l.v_cache->nb[2] == (size_t) n_cell*G*ROW);
    }

    {
        std::vector<float> b(n_rows, 0.0f);
        std::fill(b.begin(), b.begin() + n_sink, -INFINITY);
        ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));
    }

    for (auto & q : seqs) {
        for (auto & s : q.lay) {
            s.ridx.assign((size_t) G*cap_tot, -1);
            s.res.assign((size_t) G*n_bmw, 0);
            s.req.assign((size_t) G*n_bmw, 0);
            s.slot_stamp.assign((size_t) G*cap_tot, 0);
        }
    }
    for (auto & l : layers) {
        l.ev_sel.reset(ggml_backend_event_new(dev));
        l.ev_read.reset(ggml_backend_event_new(dev));
        l.ev_up.reset(ggml_backend_event_new(dev));
        l.ev_base.reset(ggml_backend_event_new(dev));
        l.ev_su.reset(ggml_backend_event_new(dev));
        l.ev_sd.reset(ggml_backend_event_new(dev));
        l.ev_tb.reset(ggml_backend_event_new(dev));
        l.ev_cpy.reset(ggml_backend_event_new(dev));
        l.ev_rb.reset(ggml_backend_event_new(dev));
        if (!l.ev_sel || !l.ev_read || !l.ev_up || !l.ev_base || !l.ev_su || !l.ev_sd || !l.ev_tb || !l.ev_cpy || !l.ev_rb) {
            throw std::runtime_error("packed two-tier mode needs backend events");
        }
    }
    ev_ord.reset(ggml_backend_event_new(dev));
    if (!ev_ord) {
        throw std::runtime_error("packed two-tier mode needs backend events");
    }

    {
        const int n_th = 4;
        backend_cp.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend_cp) {
            throw std::runtime_error("failed to create the copy backend of the packed two-tier mode");
        }
        // the base uploads on their own stream, so that the readbacks and the uploads of the joins do not queue behind them
        backend_bu.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend_bu) {
            throw std::runtime_error("failed to create the base upload backend of the packed two-tier mode");
        }
        pool = std::make_unique<gather_pool>(n_th);
        work = std::make_unique<worker>(*this);
        LLAMA_LOG_INFO("%s: gather on %d threads, events %s\n", __func__, n_th, layers[0].ev_up ? "yes" : "no");
        if (n_rot > 0) {
            LLAMA_LOG_INFO("%s: resident rows: %d attention layers through %d rotating device buffers (pinned mirrors, base upload x %d)\n", __func__,
                    n_rot, n_buf, up_rep);
        } else {
            LLAMA_LOG_INFO("%s: resident rows: %d attention layers with their own device buffer (pinned mirrors, uploaded after a reset x %d)\n", __func__,
                    n_att, up_rep);
        }
        LLAMA_LOG_INFO("%s: ring of %lld cells per stream (%d streams) in pinned host memory, %d rotating device tail buffers of %.1f MiB (tail upload after the join of layer i - 3%s)\n", __func__,
                (long long) n_cell, n_ring_strm, N_TBUF, 2.0*ggml_nbytes(tbufs[0].k)/1048576.0, layers[0].tb_next >= 0 ? "" : ", by the joins");
    }

    const llama_kvq_memory_info mi = memory_info();

    // the scratch attention reads the host store with async copies
    if (scr) {
        scr_ok = mi.pinned;
        if (!mi.pinned) {
            LLAMA_LOG_WARN("%s: the host store is not pinned, the large ubatches attend over the cache\n", __func__);
        }
    }

    LLAMA_LOG_INFO("%s: n_coarse = %d of %d rows, %d chunks of %d rows, K %d / V %d bits (%d / %d words), resident slots %d + %d sinks\n", __func__,
            seqs[0].n_coarse, n_rows, n_chunk, n_crow, bits_k, bits_v, nwk, nwv, cap, n_sink);
    LLAMA_LOG_INFO("%s: device: codes %.1f MiB, tables %.1f MiB (%s), resident %.1f MiB, ring %.1f MiB, scratch %.1f MiB (%s), other %.1f MiB; host: store %.1f MiB (%s), codes and tables %.1f MiB, ring %.1f MiB, other %.1f MiB\n", __func__,
            mi.codes/1048576.0, mi.tables/1048576.0, "freed in a prompt only with --kvq-prefill-free-codes", mi.resident/1048576.0, mi.ring/1048576.0, mi.scratch/1048576.0, scr_ok ? "used" : "not used", mi.other_dev/1048576.0,
            mi.host_store/1048576.0, mi.pinned ? "pinned" : "pageable", mi.host_codes/1048576.0, mi.host_ring/1048576.0, mi.host_other/1048576.0);
    {
        const size_t pb = pages->bytes(0);
        LLAMA_LOG_INFO("%s: pool of %d pages (%d chunks + %d), device reserved for all of them: codes %.1f MiB + tables %.1f MiB; host: one page is %.1f MiB, %s%s%s\n", __func__,
                pages->n_pages(), n_chunk, n_sq - 1, mi.codes/1048576.0, mi.tables/1048576.0, pb/1048576.0,
                host_prealloc ? "all pages allocated at load and kept" : "pages allocated on demand and given back to the OS when released (--kvq-host-prealloc: at load, kept)",
                !host_prealloc && pages->keep_host_bytes() > 0 ? format(", up to %.1f MiB of free pages kept (--kvq-host-keep-mib)", pages->keep_host_bytes()/1048576.0).c_str() : "",
                pages->max_host_bytes() > 0 ? format(", capped at %.1f MiB", pages->max_host_bytes()/1048576.0).c_str() : "");
    }
}

llama_kvq_store::~llama_kvq_store() {
    // the worker uses the layers
    work.reset();
    pool.reset();
    if (backend_cp) {
        ggml_backend_synchronize(backend_cp.get());
    }
    if (backend_bu) {
        ggml_backend_synchronize(backend_bu.get());
    }
}

const llama_kvq_store::layer_tensors & llama_kvq_store::get_layer(int32_t il) const {
    return layers.at(layer_of_il.at(il)).t;
}

ggml_tensor * llama_kvq_store::get_ridx(ggml_context * ctx, int32_t il, int64_t S) const {
    ggml_tensor * r = get_layer(il).ridx;
    if (S > 1) {
        return ggml_view_3d(ctx, r, cap_tot, G, S, r->nb[1], r->nb[2], 0);
    }
    return ggml_view_2d(ctx, r, cap_tot, G, r->nb[1], 0);
}

ggml_tensor * llama_kvq_store::get_rkv(ggml_context * ctx, int32_t il, int64_t S) const {
    GGML_ASSERT(S >= 1 && S <= n_str);
    ggml_tensor * t = get_layer(il).rkv;
    return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], 2*S, t->nb[1], t->nb[2], t->nb[3], 0);
}

ggml_tensor * llama_kvq_store::get_kc(ggml_context * ctx, int32_t il) const {
    ggml_tensor * t = get_layer(il).kc;
    return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
}

ggml_tensor * llama_kvq_store::get_vc(ggml_context * ctx, int32_t il) const {
    ggml_tensor * t = get_layer(il).vc;
    return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
}

std::vector<int32_t> llama_kvq_store::layer_ids() const {
    std::vector<int32_t> res;
    for (const auto & l : layers) {
        res.push_back(l.il);
    }
    return res;
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
    if (backend_bu) {
        ggml_backend_synchronize(backend_bu.get());
    }
}

void llama_kvq_store::set_chunk_table(int32_t seq, int32_t il, int32_t g, bool is_v, int32_t c, const int8_t * inv, const float * aux, bool async) {
    GGML_ASSERT(c >= 0 && c < n_chunk && g >= 0 && g < G);
    GGML_ASSERT(!async || backend_main);
    const layer & l = layers.at(layer_of_il.at(il));
    const layer_tensors & t = l.t;
    const int64_t m = (int64_t) seqs[seq].page_tab[c]*2*G + 2*g + (is_v ? 1 : 0);
    memcpy(htinv_ptr(seq, l.li, c, 2*g + (is_v ? 1 : 0)), inv, (size_t) D*D);
    memcpy(htaux_ptr(seq, l.li, c, 2*g + (is_v ? 1 : 0)), aux, (size_t) D*3*sizeof(float));
    if (!codes_on_dev()) {
        return;
    }
    // one copy of the slot from the host copy; a queued upload of codes_alloc may read it now: this write comes after it on the main stream
    const uint8_t * slot = htab_ptr(seq, l.li, c, 2*g + (is_v ? 1 : 0));
    if (async) {
        ggml_backend_tensor_set_async(backend_main, t.tab, slot, m*t.tab->nb[1], TAB_SLOT);
    } else {
        ggml_backend_tensor_set(t.tab, slot, m*t.tab->nb[1], TAB_SLOT);
    }
}

void llama_kvq_store::set_n_host(int32_t seq, int32_t n) {
    seq_state & q = seqs[seq];
    GGML_ASSERT(n >= q.n_coarse && n <= n_rows_);
    // the queued tail uploads start at n_host; the host ring does not hold the rows before it
    if (n != q.n_host) {
        reset_base();
        GGML_ASSERT(n < q.n_host || q.st_ok >= n);
        if (n < q.n_host) {
            q.rg_ok = std::min(q.rg_ok, n);
        }
    }
    // the codes of a job are rows of the host store; the rows after n need no upload
    q.hc_hi = std::min(q.hc_hi, n);
    q.n_host = n;
}

void llama_kvq_store::set_n_coarse(int32_t seq, int32_t n) {
    seq_state & q = seqs[seq];
    GGML_ASSERT(n >= 0 && n <= n_rows_ && n <= std::max(q.n_host, q.n_coarse));
    if (n == q.n_coarse) {
        return;
    }
    drop_pending();
    reset_base();
    synchronize();
    q.hc_hi = std::max(q.hc_hi, n);
    if (n < q.n_coarse) {
        // the slots of the removed rows become empty, the sinks keep their slots
        // a layer gets its device ridx at every join
        for (auto & l : layers) {
            layer_seq & s_l = ls(seq, l);
            for (int32_t g = 0; g < G; ++g) {
                for (int32_t s = n_sink; s < cap_tot; ++s) {
                    if (s_l.ridx[(size_t) g*cap_tot + s] >= n) {
                        layer::forget(s_l, g, s, cap_tot, n_bmw);
                    }
                }
            }
        }
    } else if (q.n_coarse == 0) {
        // the rows of the sinks are in the host store now
        for (auto & l : layers) {
            upload_pinned(seq, l, n_sink);
        }
    }
    q.n_coarse = n;
}

// the table tensors go by their own type and slice size, so a change of the packed table format is a mismatch, not a misread
void llama_kvq_store::state_write_meta(int32_t seq, llama_io_write_i & io) const {
    const layer_tensors & t = layers[0].t;
    const int32_t v[13] = {
        (int32_t) layers.size(), G, (int32_t) D, (int32_t) ROW, n_crow, nwk, nwv, seqs[seq].n_coarse,
        (int32_t) GGML_TYPE_I8, (int32_t) (D*D), (int32_t) GGML_TYPE_F32, (int32_t) (D*3*sizeof(float)), seqs[seq].n_host,
    };
    io.write(v, sizeof(v));
    for (const auto & l : layers) {
        io.write(&l.il, sizeof(l.il));
    }
}

void llama_kvq_store::state_read_meta(int32_t seq, llama_io_read_i & io, int32_t & n_out, int32_t & n_host_out) const {
    GGML_ASSERT(seq >= 0 && seq < n_seq());
    int32_t v[13];
    io.read(v, sizeof(v));
    if (v[0] != (int32_t) layers.size() || v[1] != G || v[2] != (int32_t) D || v[3] != (int32_t) ROW) {
        throw std::runtime_error(format("kvq state of %d attention layers with %d kv heads of %d, this context: %d, %d, %d",
                v[0], v[1], v[2], (int) layers.size(), G, (int) D));
    }
    if (v[4] != n_crow || v[5] != nwk || v[6] != nwv) {
        throw std::runtime_error(format("kvq state with chunks of %d rows and %d / %d words per K / V row, this context: %d, %d / %d",
                v[4], v[5], v[6], n_crow, nwk, nwv));
    }
    if (v[8] != (int32_t) GGML_TYPE_I8 || v[9] != (int32_t) (D*D) || v[10] != (int32_t) GGML_TYPE_F32 || v[11] != (int32_t) (D*3*sizeof(float))) {
        throw std::runtime_error(format("kvq state with tables of type %d / %d (%d / %d bytes), this context: %d / %d (%d / %d bytes)",
                v[8], v[10], v[9], v[11], (int) GGML_TYPE_I8, (int) GGML_TYPE_F32, (int) (D*D), (int) (D*3*sizeof(float))));
    }
    const int32_t n = v[7];
    const int32_t h = v[12];
    if (n < 0 || h < n || h > n_rows_) {
        throw std::runtime_error(format("kvq state with a coarse region of %d rows and %d host rows, this context holds %d", n, h, n_rows_));
    }
    for (const auto & l : layers) {
        int32_t il = -1;
        io.read(&il, sizeof(il));
        if (il != l.il) {
            throw std::runtime_error(format("kvq state with attention layer %d where this context has layer %d", il, l.il));
        }
    }
    n_out      = n;
    n_host_out = h;
}

// per layer: the host rows of each kv head, the codes of each kv head, the tables of the chunks
void llama_kvq_store::state_write_data(int32_t seq, llama_io_write_i & io) {
    synchronize();
    const int32_t n = seqs[seq].n_coarse;
    for (auto & l : layers) {
        for (int32_t g = 0; g < G; ++g) {
            chunk_runs(0, seqs[seq].n_host, [&](int32_t p, int32_t m) {
                io.write(row_ptr(seq, l.li, g, p), (size_t) m*2*ROW);
            });
        }
        if (n == 0) {
            continue;
        }
        // from the host copy, the device may not have the codes
        for (int k = 0; k < 2; ++k) {
            const ggml_tensor * rows = k ? l.t.vc : l.t.kc;
            for (int32_t g = 0; g < G; ++g) {
                chunk_runs(0, n, [&](int32_t p, int32_t m) {
                    io.write(hcode_ptr(seq, l.li, k == 1, g, p), (size_t) m*rows->nb[1]);
                });
            }
        }
        // the state holds the inverse tables of the slots of a chunk, then their aux floats, as before the slots were joined
        for (int k = 2; k < 4; ++k) {
            for (int32_t c = 0; c < (n + n_crow - 1)/n_crow; ++c) {
                for (int32_t m = 0; m < 2*G; ++m) {
                    io.write(k == 3 ? htaux_ptr(seq, l.li, c, m) : htinv_ptr(seq, l.li, c, m), k == 3 ? (size_t) D*3*sizeof(float) : (size_t) D*D);
                }
            }
        }
    }
}

void llama_kvq_store::state_read_data(int32_t seq, llama_io_read_i & io, int32_t n, int32_t h) {
    seq_state & q = seqs[seq];
    GGML_ASSERT(q.n_coarse == 0 && n <= h && h <= n_rows_);
    drop_pending();
    reset_base();
    synchronize();
    for (auto & l : layers) {
        for (int32_t g = 0; g < G; ++g) {
            chunk_runs(0, h, [&](int32_t p, int32_t m) {
                io.read(row_ptr(seq, l.li, g, p), (size_t) m*2*ROW);
            });
        }
        if (n == 0) {
            continue;
        }
        for (int k = 0; k < 2; ++k) {
            const ggml_tensor * rows = k ? l.t.vc : l.t.kc;
            for (int32_t g = 0; g < G; ++g) {
                chunk_runs(0, n, [&](int32_t p, int32_t m) {
                    io.read(hcode_ptr(seq, l.li, k == 1, g, p), (size_t) m*rows->nb[1]);
                });
            }
        }
        for (int k = 2; k < 4; ++k) {
            for (int32_t c = 0; c < (n + n_crow - 1)/n_crow; ++c) {
                for (int32_t m = 0; m < 2*G; ++m) {
                    io.read(k == 3 ? htaux_ptr(seq, l.li, c, m) : htinv_ptr(seq, l.li, c, m), k == 3 ? (size_t) D*3*sizeof(float) : (size_t) D*D);
                }
            }
        }
    }
    if (codes_on_dev()) {
        codes_up(seq, n, false);
    }
    // the sink rows are read from the host store now
    q.n_host = h;
    set_n_coarse(seq, n);
}

size_t llama_kvq_store::stream_device_bytes() const {
    size_t n = 0;
    for (const auto & b : bufs) {
        n += (size_t) 2*G*cap_tot*ROW;
        (void) b;
    }
    n += (size_t) N_TBUF*2*ggml_row_size(GGML_TYPE_Q8_0, D*G)*n_cell;
    if (jblk) {
        n += ggml_nbytes(jblk)/n_str;
    }
    return n;
}

bool llama_kvq_store::enable(const llama_kvq_params & p) {
    const bool own_p = p.layer_bufs < 0 || p.layer_bufs >= (int32_t) layers.size();
    if (p.n_sink != n_sink || llama_kvq_resident_slots(p) != cap || own_p != (n_rot == 0) || (!own_p && p.layer_bufs != n_buf)) {
        LLAMA_LOG_ERROR("%s: n_sink, resident_cap and layer_bufs must match the allocation (%d, %d, %d)\n", __func__, n_sink, cap, n_rot > 0 ? n_buf : -1);
        return false;
    }
    if (p.max_batch < 1 || p.max_batch > 8 || p.lead < 0) {
        LLAMA_LOG_ERROR("%s: bad max_batch or lead\n", __func__);
        return false;
    }
    // the smallest coarse region is one chunk
    const int32_t n_cand = n_crow - n_sink;
    if (p.prefetch_n < 1 || p.prefetch_n > cap || p.prefetch_n > n_cand) {
        LLAMA_LOG_ERROR("%s: bad prefetch_n for %d slots and %d candidates\n", __func__, cap, n_cand);
        return false;
    }
    // the staging and the base uploads are ordered by the prefetch of the next layer coming after the attention of the layer before
    if (p.lead < 1 || p.lead > 3) {
        LLAMA_LOG_ERROR("%s: lead must be 1 - 3 (got %d)\n", __func__, p.lead);
        return false;
    }
    if (pf_scores(p.prefetch_n, p.max_batch) && !cut_warn) {
        LLAMA_LOG_INFO("%s: %d resident slots, fewer than max_batch x prefetch_n = %d: a request of more distinct rows is cut by the max score\n",
                __func__, cap, p.max_batch*p.prefetch_n);
        cut_warn = true;
    }
    drop_pending();
    reset_base();
    params = p;

    // pinned host buffers, for the largest request of these parameters
    if (backend_cp) {
        ggml_backend_synchronize(backend_cp.get());
    }
    if (backend_bu) {
        ggml_backend_synchronize(backend_bu.get());
    }
    buf_pin.reset();
    {
        // the staging rows (device and pinned host, one for all layers) hold a full fetch of every slot
        // (of up to n_str streams: the fetches of the streams follow each other)
        const int64_t n_rows = 2*G*n_req(p.prefetch_n, p.max_batch)*n_str;
        const int64_t n_sel  = (int64_t) p.prefetch_n*G*p.max_batch*n_str;

        // one block [ridx | slot list | staging rows] on the device and the same layout pinned, so that a join is one copy
        auto pad = [](size_t n) { return GGML_PAD(n, 256); };
        const size_t sz_ridx  = pad((size_t) G*cap_tot*n_str*sizeof(int32_t));
        const size_t sz_slots = pad(n_rows*sizeof(int32_t));
        const size_t sz_stage = pad(n_rows*ROW);
        const size_t sz_sel   = pad(n_sel*sizeof(int32_t));

        // a graph that is reused keeps the tensors, so they change only with the parameters that change the graph
        if (!jblk || n_rows != n_stage) {
            n_stage   = n_rows;
            off_slots = sz_ridx;
            off_stage = sz_ridx + sz_slots;
            buf_stage.reset();
            ggml_init_params ip = { 4*ggml_tensor_overhead(), nullptr, true };
            ctx_stage.reset(ggml_init(ip));
            jblk = ggml_new_tensor_1d(ctx_stage.get(), GGML_TYPE_I8, off_stage + sz_stage);
            ggml_set_name(jblk, "kvq_join");
            // after the resident and tail buffers in the memory of the scratch, if it has the room
            if (off_jblk + ggml_backend_buft_get_alloc_size(buft, jblk) <= ggml_backend_buffer_get_size(buf_scr.get())) {
                if (ggml_backend_tensor_alloc(buf_scr.get(), jblk, (uint8_t *) ggml_backend_buffer_get_base(buf_scr.get()) + off_jblk) != GGML_STATUS_SUCCESS) {
                    LLAMA_LOG_ERROR("%s: failed to place the staging tensors\n", __func__);
                    return false;
                }
            } else {
                buf_stage.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_stage.get(), buft));
                if (!buf_stage) {
                    LLAMA_LOG_ERROR("%s: failed to allocate the staging tensors\n", __func__);
                    return false;
                }
                ggml_backend_buffer_set_usage(buf_stage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            ggml_backend_buffer_t bs = jblk->buffer;
            ggml_tensor * ridx  = ggml_new_tensor_3d(ctx_stage.get(), GGML_TYPE_I32, cap_tot, G, n_str);
            ggml_tensor * slots = ggml_new_tensor_1d(ctx_stage.get(), GGML_TYPE_I32, n_stage);
            ggml_tensor * stage = ggml_new_tensor_2d(ctx_stage.get(), GGML_TYPE_F32, ROW/4, n_stage);
            ggml_set_name(ridx,  "kvq_ridx");
            ggml_set_name(slots, "kvq_slots");
            ggml_set_name(stage, "kvq_stage");
            uint8_t * base = (uint8_t *) jblk->data;
            if (ggml_backend_tensor_alloc(bs, ridx,  base) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(bs, slots, base + sz_ridx) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(bs, stage, base + off_stage) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to place the staging tensors\n", __func__);
                return false;
            }
            for (auto & l : layers) {
                l.t.stage = stage;
                l.t.slots = slots;
                l.t.ridx  = ridx;
            }
        }
        std::vector<int32_t> trash(n_stage);
        for (int64_t j = 0; j < n_stage; ++j) {
            trash[j] = trash_row(j);
        }
        ggml_backend_tensor_set(layers[0].t.slots, trash.data(), 0, ggml_nbytes(layers[0].t.slots));
        slots_hi = 0;
        // the shared memory: the next ubatch sets it up for its path
        scr_mode = 0;

        const size_t sz_all = off_stage + sz_stage + 2*sz_sel*layers.size();
        buf_pin.reset(ggml_backend_buft_alloc_buffer(buft_host, sz_all));
        if (!buf_pin) {
            LLAMA_LOG_ERROR("%s: failed to allocate %.1f MiB of pinned host buffers\n", __func__, sz_all/1048576.0);
            return false;
        }
        uint8_t * b = (uint8_t *) ggml_backend_buffer_get_base(buf_pin.get());
        h_ridx  = (int32_t *) b; b += sz_ridx;
        h_slots = (int32_t *) b; b += sz_slots;
        h_stage = b;             b += sz_stage;
        memcpy(h_slots, trash.data(), n_stage*sizeof(int32_t));
        for (auto & l : layers) {
            l.h_sel  = (int32_t *) b; b += sz_sel;
            l.h_val  = (float *) b;   b += sz_sel;
            l.up_rec = false;
        }
    }

    for (int32_t s = 0; s < n_seq(); ++s) {
        reset_resident(s);
    }
    reset_counters();

    // the device copy of the codes and tables lives with the context, unless a prompt frees it (scr_alloc)
    prefill_free_codes = p.prefill_free_codes;
    if (!prefill_free_codes && !codes_alloc()) {
        return false;
    }

    LLAMA_LOG_INFO("%s: max_batch = %d, prefetch_n = %d, resident = %d, lead = %d\n", __func__, p.max_batch, p.prefetch_n, cap, p.lead);
    LLAMA_LOG_INFO("%s: staging: device %.1f MiB, pinned host %.1f MiB; mirrors: pinned host %.1f MiB\n", __func__,
            ggml_nbytes(jblk)/1048576.0, ggml_backend_buffer_get_size(buf_pin.get())/1048576.0,
            seqs[0].buf_mir ? ggml_backend_buffer_get_size(seqs[0].buf_mir.get())*n_seq()/1048576.0 : 0.0);
    return true;
}

void llama_kvq_store::reset_resident(int32_t seq) {
    drop_pending();
    reset_base();
    if (backend_cp) {
        ggml_backend_synchronize(backend_cp.get());
    }
    for (auto & l : layers) {
        layer_seq & q = ls(seq, l);
        l.up_rec   = false;
        std::fill(q.ridx.begin(), q.ridx.end(), -1);
        std::fill(q.res.begin(), q.res.end(), 0);
        std::fill(q.slot_stamp.begin(), q.slot_stamp.end(), 0);
        q.stamp    = 0;
        q.pf_stamp = 0;
        l.fetch.clear();

        // the sinks are pinned rows; they are never requested, so they are not in the bitmap
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t s = 0; s < n_sink; ++s) {
                q.ridx[(size_t) g*cap_tot + s] = s;
            }
        }

        // the sink rows are uploaded with the first chunk
        if (seqs[seq].n_coarse > 0) {
            upload_pinned(seq, l, n_sink);
        }
    }
}

// rows [0, n_pin) of the host store to the slots [0, n_pin) of the mirror, which is [kv][g][slot][ROW]; an own buffer gets them at the next join
// the caller has waited for the base uploads and the mirror writes
void llama_kvq_store::upload_pinned(int32_t sq, layer & l, int32_t n_pin) {
    GGML_ASSERT(n_pin <= n_sink);
    l.own_sinks = true;
    for (int kv = 0; kv < 2; ++kv) {
        for (int32_t g = 0; g < G; ++g) {
            uint8_t * dst = ls(sq, l).mirror + ((size_t) kv*G + g)*cap_tot*ROW;
            for (int32_t s = 0; s < n_pin; ++s) {
                memcpy(dst + (size_t) s*ROW, row_ptr(sq, l.li, g, s) + kv*ROW, ROW);
            }
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

// the worker has queued its base uploads and written its mirrors
void llama_kvq_store::wait_mirrors() {
    if (!work) {
        return;
    }
    for (auto & l : layers) {
        work->wait_mirror(l);
    }
}

// the queued base uploads are forgotten (the mirrors or the buffers may change), the next round queues them again
void llama_kvq_store::reset_base() {
    if (!backend_bu) {
        return;
    }
    wait_mirrors();
    ggml_backend_synchronize(backend_bu.get());
    for (auto & l : layers) {
        l.base_rec = false;
        l.tb_rec   = false;
        l.tb_ready = false;
    }
}

// worker thread, after the gather: the fetched rows from the staging into the slots of the mirror
// the base upload of this round may read the mirror meanwhile: the slots written here are scattered from the staging after it;
// the next base upload of the layer comes from the worker after this, or from the decode thread after wait_mirrors
void llama_kvq_store::write_mirror(layer & l) {
    const double t0 = kvq_now();
    const size_t nb_kv = (size_t) G*cap_tot*ROW;
    for (size_t i = 0; i < l.fetch.size(); ++i) {
        const kvq_fetch & f = l.fetch[i];
        uint8_t * dst = ls(ub_seq[f.st], l).mirror + ((size_t) f.g*cap_tot + f.slot)*ROW;
        memcpy(dst,         h_stage + 2*i*ROW,       ROW);
        memcpy(dst + nb_kv, h_stage + (2*i + 1)*ROW, ROW);
    }
    l.cnt.t_mirror += kvq_now() - t0;
}

// the mirror of a layer into its buffer on the base stream, after ev (main stream), in one copy (pieces of 2 MiB were slower, measured);
// ev must be after the attention of the layer index i - n_buf, the previous reader of the buffer
// the worker on the base stream after the prefetch job of layer index i - n_buf + 1 and its join (ev_sel of that prefetch),
// the decode thread on the base stream with the worker idle or on the copy stream
void llama_kvq_store::issue_base(layer & l, ggml_backend_event_t ev, ggml_backend_t bu) {
    if (ev) {
        ggml_backend_event_wait(bu, ev);
    }
    // the mirror of each stream of the ubatch into its slice
    const size_t n = (size_t) 2*G*cap_tot*ROW;
    for (int r = 0; r < up_rep; ++r) {
        for (int32_t s = 0; s < ub_n; ++s) {
            ggml_backend_tensor_set_async(bu, l.t.rkv_h, ls(ub_seq[s], l).mirror, s*n, n);
        }
    }
    ggml_backend_event_record(l.ev_base.get(), bu);
    l.base_rec = true;
    l.cnt.base_bytes += (int64_t) up_rep*n*ub_n;
}

// the ring cells of the positions [p0, p1) in at most two runs, f(first cell, cells)
template <typename F>
static void kvq_cell_runs(int64_t n_cell, int64_t p0, int64_t p1, F && f) {
    GGML_ASSERT(p1 - p0 <= n_cell);
    for (int64_t p = p0; p < p1;) {
        const int64_t c = p % n_cell;
        const int64_t n = std::min(p1 - p, n_cell - c);
        f(c, n);
        p += n;
    }
}

// host ring: the cells of the positions [tb_lo, tb_hi) of layer l from the host ring into its tail buffer, after ev (main stream)
// the worker on the base stream after the join of layer index i - 3, also after the readback of the previous user of the buffer;
// the decode thread on the copy stream (in order after that readback), with ev after the readers of the buffer on the main stream
void llama_kvq_store::issue_tail(layer & l, ggml_backend_event_t ev, ggml_backend_t stream) {
    if (ev) {
        ggml_backend_event_wait(stream, ev);
    }
    if (stream != backend_cp.get()) {
        ggml_backend_event_wait(stream, layers[l.tb_prev].ev_rb.get());
    }
    const tail_buf & tb = tbufs[l.tbuf];
    const size_t nb1 = (size_t) G*ROW;
    int64_t bytes = 0;
    // the occupied cells only: all cells cost 0.25 - 0.8 ms per verify batch (measured)
    for (int r = 0; r < up_rep; ++r) {
        for (int32_t s = 0; s < ub_n; ++s) {
            const size_t  off = ring_off(ub_seq[s]);
            const size_t  dof = (size_t) s*n_cell*nb1; // the ring of stream s in the tail buffer
            const int64_t lo  = tb_lo[s].load();
            const int64_t hi  = tb_hi[s].load();
            kvq_cell_runs(n_cell, lo, hi, [&](int64_t c, int64_t n) {
                ggml_backend_tensor_set_async(stream, tb.k, (const uint8_t *) l.k_cache->data + off + c*nb1, dof + c*nb1, n*nb1);
                ggml_backend_tensor_set_async(stream, tb.v, (const uint8_t *) l.v_cache->data + off + c*nb1, dof + c*nb1, n*nb1);
                bytes += 2*n*nb1;
            });
        }
    }
    ggml_backend_event_record(l.ev_tb.get(), stream);
    l.tb_rec = true;
    l.cnt.ring_bytes += bytes;
}

// host ring, the join of layer l: the rows of the ubatch from its tail buffer into the host ring, after the main stream reached the join
void llama_kvq_store::read_ring(layer & l) {
    ggml_backend_t cp = backend_cp.get();
    ggml_backend_event_wait(cp, l.ev_cpy.get());
    const tail_buf & tb = tbufs[l.tbuf];
    const size_t nb1 = (size_t) G*ROW;
    for (int32_t s = 0; s < ub_n; ++s) {
        const size_t off = ring_off(ub_seq[s]);
        const size_t dof = (size_t) s*n_cell*nb1;
        kvq_cell_runs(n_cell, ub_lo[s], ub_hi[s], [&](int64_t c, int64_t n) {
            ggml_backend_tensor_get_async(cp, tb.k, (uint8_t *) l.k_cache->data + off + c*nb1, dof + c*nb1, n*nb1);
            ggml_backend_tensor_get_async(cp, tb.v, (uint8_t *) l.v_cache->data + off + c*nb1, dof + c*nb1, n*nb1);
            l.cnt.rb_bytes += 2*n*nb1;
        });
    }
    ggml_backend_event_record(l.ev_rb.get(), cp);
}

// host ring, scratch path: the rows of the ubatch of layer l from the scratch into the host store, on the copy stream after the main stream
// passed the attention of the layer (the same layout, one copy per kv head)
void llama_kvq_store::read_scr(layer & l) {
    const size_t n = (size_t) (ub_p1 - ub_p0)*2*ROW;
    for (int32_t g = 0; g < G; ++g) {
        chunk_runs(ub_p0, ub_p1, [&](int32_t p, int32_t m) {
            ggml_backend_tensor_get_async(backend_cp.get(), scr, row_ptr(cur, l.li, g, p), g*scr->nb[3] + p*scr->nb[2], (size_t) m*2*ROW);
        });
    }
    l.cnt.rb_bytes += (int64_t) G*n;
}

// host ring: the rows of the positions [p0, p1) of every layer between the host store [g][row][K, V] and the host ring [cell][g][ROW] (K, V)
void llama_kvq_store::ring_copy(int32_t sq, bool to_store, int32_t p0, int32_t p1) {
    if (p1 <= p0) {
        return;
    }
    const double t0 = kvq_now();
    const size_t nb1 = (size_t) G*ROW;
    const size_t off = ring_off(sq);
    for (auto & l : layers) {
        for (int32_t p = p0; p < p1; ++p) {
            uint8_t * rk = (uint8_t *) l.k_cache->data + off + (p % n_cell)*nb1;
            uint8_t * rv = (uint8_t *) l.v_cache->data + off + (p % n_cell)*nb1;
            for (int32_t g = 0; g < G; ++g) {
                uint8_t * s = row_ptr(sq, l.li, g, p);
                if (to_store) {
                    memcpy(s,       rk + g*ROW, ROW);
                    memcpy(s + ROW, rv + g*ROW, ROW);
                } else {
                    memcpy(rk + g*ROW, s,       ROW);
                    memcpy(rv + g*ROW, s + ROW, ROW);
                }
            }
        }
    }
    LLAMA_LOG_DEBUG("%s: positions [%d, %d) %s in %.3f ms\n", __func__, p0, p1, to_store ? "to the host store" : "to the host ring", 1e3*(kvq_now() - t0));
}

void llama_kvq_store::sync_store(int32_t seq, int32_t n) {
    seq_state & q = seqs[seq];
    if (q.st_ok >= n) {
        return;
    }
    synchronize();
    GGML_ASSERT(q.rg_ok >= n);
    ring_copy(seq, true, std::max(q.st_ok, q.n_host), n);
    q.st_ok = n;
}

void llama_kvq_store::sync_ring(int32_t seq, int32_t n) {
    seq_state & q = seqs[seq];
    if (q.rg_ok >= n) {
        return;
    }
    // the queued tail uploads read the host ring
    reset_base();
    synchronize();
    GGML_ASSERT(q.st_ok >= n);
    ring_copy(seq, false, std::max(q.rg_ok, q.n_host), n);
    q.rg_ok = n;
}

void llama_kvq_store::ring_loaded(int32_t seq, int32_t n) {
    reset_base();
    seqs[seq].st_ok = seqs[seq].n_host;
    seqs[seq].rg_ok = n;
}

void llama_kvq_store::order_rows() {
    ggml_backend_event_record(ev_ord.get(), backend_cp.get());
    ggml_backend_event_wait(backend_main, ev_ord.get());
}

ggml_tensor * llama_kvq_store::get_ring(ggml_context * ctx, int32_t il, bool is_v, int64_t n_kv, int64_t S) const {
    GGML_ASSERT(n_kv <= n_cell && S >= 1 && S <= n_str);
    const tail_buf & tb = tbufs[layers.at(layer_of_il.at(il)).tbuf];
    ggml_tensor * t = is_v ? tb.v : tb.k;
    if (S > 1) {
        // one ring per stream
        return ggml_view_4d(ctx, t, D, n_kv, G, S, t->nb[1], ROW, t->nb[1]*n_cell, 0);
    }
    return ggml_view_3d(ctx, t, D, n_kv, G, t->nb[1], ROW, 0);
}

ggml_tensor * llama_kvq_store::build_ring_cpy(ggml_context * ctx, int32_t il, bool is_v, ggml_tensor * cur, ggml_tensor * idxs) const {
    GGML_ASSERT(cur->ne[0]*cur->ne[1] == D*G && ggml_row_size(cur->type, cur->ne[0]) == cur->nb[1]);
    const tail_buf & tb = tbufs[layers.at(layer_of_il.at(il)).tbuf];
    cur = ggml_view_2d(ctx, cur, D*G, cur->ne[2], cur->nb[2], 0);
    return ggml_set_rows(ctx, is_v ? tb.v : tb.k, cur, idxs);
}

ggml_tensor * llama_kvq_store::build_scr_set(ggml_context * ctx, bool is_v, ggml_tensor * cur, ggml_tensor * pos) const {
    GGML_ASSERT(scr && cur->ne[0] == D && cur->ne[1] == G && pos->ne[0] == cur->ne[2]);
    ggml_tensor * dst = ggml_view_3d(ctx, scr, D, n_scr, G, scr->nb[2], scr->nb[3], is_v ? scr->nb[1] : 0);
    return ggml_set_rows(ctx, dst, ggml_permute(ctx, cur, 0, 2, 1, 3), pos);
}

int64_t llama_kvq_store::n_req(int64_t n_top, int64_t L) const {
    return std::min<int64_t>(cap, n_top*L);
}

void llama_kvq_store::flush_ubatch() {
    n_tokens += ub_tokens;
    ub_tokens = 0;
}

// the positions of the tokens [i0, i0 + n) of the ubatch, one run
static void kvq_ubatch_pos(const llama_ubatch & ub, int32_t & p0, int32_t & p1, uint32_t i0 = 0, uint32_t n = 0) {
    n = n > 0 ? n : ub.n_tokens;
    p0 = ub.pos[i0];
    p1 = ub.pos[i0];
    for (uint32_t i = i0; i < i0 + n; ++i) {
        p0 = std::min(p0, ub.pos[i]);
        p1 = std::max(p1, ub.pos[i]);
    }
    p1++;
    GGML_ASSERT(p1 - p0 == (int32_t) n && "the two-tier cache needs a ubatch of consecutive positions");
}

// the sequence of a ubatch: one
static int32_t kvq_ubatch_seq(const llama_ubatch & ub, int32_t n_seq) {
    GGML_ASSERT(ub.n_seqs_unq == 1 && ub.seq_id_unq[0] >= 0 && ub.seq_id_unq[0] < n_seq && "the two-tier cache needs a ubatch of one sequence");
    return (int32_t) ub.seq_id_unq[0];
}

void llama_kvq_store::use_seq(int32_t s) {
    use_streams(&s, 1);
}

// the streams of the ubatch: sequence ids[i] is stream i (slice i of the resident, tail and ridx buffers)
void llama_kvq_store::use_streams(const int32_t * ids, int32_t n) {
    GGML_ASSERT(n >= 1 && n <= n_str);
    bool same = n == ub_n;
    for (int32_t i = 0; i < n && same; ++i) {
        same = ids[i] == ub_seq[i];
    }
    for (int32_t i = 0; i < n; ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < n_seq());
    }
    if (same) {
        return;
    }
    // the prefetches, the base uploads and the mirror writes of the ubatch before work on its sequence
    drop_pending();
    reset_base();
    // an own device buffer of a layer holds the resident rows of the sequence before: the next join uploads the mirror of this one
    for (auto & l : layers) {
        if (l.own) {
            l.own_stale = true;
            l.own_sinks = false;
        }
    }
    cur = ids[0];
    ub_n = n;
    for (int32_t i = 0; i < n; ++i) {
        ub_seq[i] = ids[i];
    }
}

void llama_kvq_store::begin_ubatch(const llama_ubatch & ub) {
    const int64_t n = ub.n_tokens;
    GGML_ASSERT(n <= 64);
    // a prefetch without its join, e.g. after a failed graph compute
    for (auto & l : layers) {
        if (l.pending) {
            LLAMA_LOG_WARN("%s: dropping the prefetch of layer %d\n", __func__, l.il);
            drop(l);
        }
    }
    // the sequences of the ubatch are its streams, in the order of the tokens (equal runs, ascending sequence ids)
    const int32_t S = (int32_t) ub.n_seqs_unq;
    GGML_ASSERT(S >= 1 && S <= n_str && n % S == 0 && "the two-tier cache: a packed ubatch of at most n_seq_max sequences of equal length");
    int32_t ids[GGML_KVQ_MAX_STREAM];
    for (int32_t s = 0; s < S; ++s) {
        ids[s] = (int32_t) ub.seq_id_unq[s];
        GGML_ASSERT(ub.n_seq_id[s*(n/S)] == 1 && ub.seq_id[s*(n/S)][0] == ub.seq_id_unq[s]);
    }
    use_streams(ids, S);
    flush_ubatch();
    ub_tokens = n;
    n_ubatch++;

    for (int32_t s = 0; s < S; ++s) {
        seq_state & q = seqs[ids[s]];
        kvq_ubatch_pos(ub, ub_lo[s], ub_hi[s], s*(n/S), n/S);
        sync_ring(ids[s], ub_lo[s]);
        q.rg_ok = ub_hi[s];
        q.st_ok = std::min(q.st_ok, ub_lo[s]);
        tb_lo[s] = q.n_host;
        tb_hi[s] = ub_hi[s];
    }
    ub_p0 = ub_lo[0];
    ub_p1 = ub_hi[0];
    // the early query of the first layer comes before any join: its tail now, if the last join of the previous round did not wait for it
    layer & l0 = layers[0];
    if (!l0.tb_ready) {
        ggml_backend_event_record(l0.ev_cpy.get(), backend_main);
        issue_tail(l0, l0.ev_cpy.get(), backend_cp.get());
        ggml_backend_event_wait(backend_main, l0.ev_tb.get());
        l0.tb_rec = false;
    }
    l0.tb_ready = false;
}

void llama_kvq_store::begin_scr(const llama_ubatch & ub) {
    use_seq(kvq_ubatch_seq(ub, n_seq()));
    seq_state & q = seqs[cur];
    kvq_ubatch_pos(ub, ub_p0, ub_p1);
    ub_lo[0] = ub_p0;
    ub_hi[0] = ub_p1;
    sync_store(cur, ub_p0);
    q.st_ok = ub_p1;
    q.rg_ok = std::min(q.rg_ok, ub_p0);
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

// the callback of the trigger fills rows [0, n) of the staging tensor and the slot list, the rest of the list points to trash rows
static void kvq_build_scatter(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * rkv_h, ggml_tensor * staging, ggml_tensor * slots, int64_t n_rows, int k, int32_t il) {
    GGML_ASSERT(staging != nullptr && n_rows <= staging->ne[1]);
    ggml_tensor * src = ggml_view_2d(ctx, staging, staging->ne[0], n_rows, staging->nb[1], 0);
    ggml_tensor * idx = ggml_view_1d(ctx, slots, n_rows, 0);
    ggml_tensor * scatter = ggml_set_rows(ctx, rkv_h, src, idx);
    ggml_format_name(scatter, "kvq_scatter%d-%d", k, il);
    ggml_build_forward_expand(gf, scatter);
}

void llama_kvq_store::build_fetch(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t n_coarse_min, int32_t il, ggml_tensor * score, int64_t n_top, ggml_tensor * bias_s) const {
    const kind k = KIND_PREFETCH;
    const layer_tensors & t = get_layer(il);

    // the largest coarse region of the streams
    const int64_t n_coarse_ = score->ne[0];
    const int64_t L = score->ne[2];
    const int64_t S = score->ne[3];
    GGML_ASSERT(score->ne[1] == G && n_coarse_ <= n_rows_);
    GGML_ASSERT(n_top <= n_coarse_min - n_sink);

    // the sinks are never candidates; streams: the rows of a stream at or after its n_coarse are 0 in the score, they are never
    // candidates either (bias_s: f32 [n_coarse, 1, 1, S], -inf for the sinks and those rows)
    ggml_tensor * sc = nullptr;
    if (S > 1) {
        GGML_ASSERT(bias_s != nullptr && bias_s->ne[0] == n_coarse_ && bias_s->ne[3] == S);
        sc = ggml_reshape_2d(ctx, ggml_add(ctx, score, bias_s), n_coarse_, G*L*S);
    } else {
        sc = ggml_add(ctx, ggml_reshape_2d(ctx, score, n_coarse_, G*L), ggml_view_1d(ctx, bias, n_coarse_, 0));
    }

    ggml_tensor * ids = ggml_top_k(ctx, sc, n_top); // [n_top, G*L]
    ggml_format_name(ids, "kvq_ids%d-%d", (int) k, il);
    // read in the copy callback of a later split, after its last use in the graph
    ggml_set_output(ids);

    // the trigger reads a get_rows of the selection (anchor), so it runs after the selection
    ggml_tensor * sc3    = ggml_reshape_3d(ctx, sc, 1, n_coarse_, G*L*S);
    ggml_tensor * anchor = nullptr;
    if (pf_scores(n_top, L)) {
        // the scores of the selected rows, to cut the union over the positions; read back from a copy stream, so an output
        anchor = ggml_get_rows(ctx, sc3, ids); // [1, n_top, G*L]
        ggml_format_name(anchor, "kvq_vals%d-%d", (int) k, il);
        ggml_set_output(anchor);
    } else {
        // only the rows are read back, the anchor is the score of one row
        anchor = ggml_get_rows(ctx, ggml_view_3d(ctx, sc3, 1, n_coarse_, 1, sc3->nb[1], sc3->nb[2], 0), ggml_view_2d(ctx, ids, 1, 1, ids->nb[1], 0));
        ggml_format_name(anchor, "kvq_anchor%d-%d", (int) k, il);
    }

    // the trigger value is not set, the result is not used
    ggml_tensor * trig = ggml_add(ctx, ggml_view_1d(ctx, anchor, 1, 0), t.trig[k]);
    ggml_format_name(trig, "kvq_trig%d-%d", (int) k, il);
    ggml_backend_sched_set_tensor_backend(sched, trig, kvq_backend(sched, dev));
    ggml_build_forward_expand(gf, trig);
}

void llama_kvq_store::build_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, int64_t n_top, int64_t L, int64_t S) const {
    const layer_tensors & t = get_layer(il);

    ggml_tensor * trig = ggml_add(ctx, ggml_view_1d(ctx, bias, 1, 0), t.trig[KIND_JOIN]);
    ggml_format_name(trig, "kvq_trig%d-%d", (int) KIND_JOIN, il);
    ggml_backend_sched_set_tensor_backend(sched, trig, kvq_backend(sched, dev));
    ggml_build_forward_expand(gf, trig);

    kvq_build_scatter(ctx, gf, t.rkv_h, t.stage, t.slots, 2*G*n_req(n_top, L)*S, KIND_JOIN, il);
}

static void kvq_build_trigger(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, ggml_backend_dev_t dev,
        ggml_tensor * src, ggml_tensor * trig, int k, int32_t il) {
    ggml_tensor * t = ggml_add(ctx, ggml_view_1d(ctx, src, 1, 0), trig);
    ggml_format_name(t, "kvq_trig%d-%d", k, il);
    ggml_backend_sched_set_tensor_backend(sched, t, kvq_backend(sched, dev));
    ggml_build_forward_expand(gf, t);
}

void llama_kvq_store::build_scr_up(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const {
    GGML_ASSERT(scr_ok);
    kvq_build_trigger(ctx, gf, sched, dev, bias, get_layer(il).trig[KIND_SCR_UP], KIND_SCR_UP, il);
}

void llama_kvq_store::build_scr_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const {
    GGML_ASSERT(scr_ok);
    kvq_build_trigger(ctx, gf, sched, dev, bias, get_layer(il).trig[KIND_SCR_JOIN], KIND_SCR_JOIN, il);
}

void llama_kvq_store::build_scr_rb(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const {
    GGML_ASSERT(scr_ok);
    kvq_build_trigger(ctx, gf, sched, dev, bias, get_layer(il).trig[KIND_SCR_RB], KIND_SCR_RB, il);
}

ggml_tensor * llama_kvq_store::get_scr(ggml_context * ctx, bool is_v, int64_t n, int64_t r0) const {
    GGML_ASSERT(scr && r0 >= 0 && r0 + n <= n_scr);
    return ggml_view_4d(ctx, scr, D, G, n, 1, scr->nb[3], scr->nb[2], scr->nb[3]*G, r0*scr->nb[2] + (is_v ? scr->nb[1] : 0));
}

void llama_kvq_store::select(const layer & l, const ggml_tensor * dst, ggml_cgraph * graph, const ggml_tensor ** ids, const ggml_tensor ** anchor) const {
    // trig = add(view(anchor), trigger), anchor = get_rows(scores, ids or a view of ids)
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
    *anchor = trig->src[0]->view_src;
    GGML_ASSERT(*anchor != nullptr && (*anchor)->op == GGML_OP_GET_ROWS);
    *ids = (*anchor)->src[1]->view_src ? (*anchor)->src[1]->view_src : (*anchor)->src[1];
    GGML_ASSERT((*ids)->op == GGML_OP_TOP_K);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(gr, i);
        if (node == *anchor || node == *ids) {
            GGML_ABORT("the selection of layer %d is computed in the same split as its trigger", l.il);
        }
    }
    GGML_ASSERT(ggml_is_contiguous(*ids) && ggml_is_contiguous(*anchor) && (*ids)->type == GGML_TYPE_I32 && (*anchor)->type == GGML_TYPE_F32);
}

// more requested rows than slots: the n_u rows with the largest max score over the positions stay requested, ties by row
void llama_kvq_store::cut_request(layer & l, layer_seq & q, int32_t st, int32_t g, int64_t n_u) {
    uint64_t * rq = q.req.data() + g*n_bmw;
    int64_t n = 0;
    for (int64_t w = 0; w < n_bmw; ++w) {
        n += kvq_popcount(rq[w]);
    }
    if (n <= n_u) {
        return;
    }
    const int64_t n_top = l.n_top;
    const int64_t L     = l.n_pos;
    std::vector<kvq_cand> & u = l.cand;
    u.clear();
    for (int64_t p = 0; p < L; ++p) {
        const int64_t o = (g + G*(p + L*st))*n_top;
        for (int64_t i = 0; i < n_top; ++i) {
            u.push_back({ l.h_sel[o + i], l.h_val[o + i] });
        }
    }
    kvq_sort_rows(u, n_rows_);
    size_t m = 0;
    for (const kvq_cand & c : u) {
        if (m > 0 && u[m - 1].row == c.row) {
            u[m - 1].val = std::max(u[m - 1].val, c.val);
        } else {
            u[m++] = c;
        }
    }
    u.resize(m);
    std::nth_element(u.begin(), u.begin() + n_u, u.end(), [](const kvq_cand & a, const kvq_cand & b) {
        return a.val > b.val || (a.val == b.val && a.row < b.row);
    });
    std::fill(rq, rq + n_bmw, 0);
    for (int64_t i = 0; i < n_u; ++i) {
        kvq_bit_set(rq, u[i].row);
    }
    l.cnt.pf_cut++;
    for (int64_t p = 0; p < L; ++p) {
        const int64_t o = (g + G*(p + L*st))*n_top;
        for (int64_t i = 0; i < n_top; ++i) {
            l.cnt.pf_lost += !kvq_bit(rq, l.h_sel[o + i]);
        }
    }
}

// the rows of l.miss (by row, not resident) go to free slots: a slot is free if its row is not in l.req
// the free slots of the oldest requests first, then by slot, so the slot of a row does not depend on the order of the selection
// the slots of the requested rows get the stamp now; l.req is cleared
void llama_kvq_store::place(layer & l, layer_seq & q, int32_t g, uint32_t now) {
    uint64_t * rq = q.req.data() + g*n_bmw;
    const size_t o = (size_t) g*cap_tot;

    // without branches on the request: about 40% of the slots are requested, in no order
    l.free_s.resize(cap + 1);
    l.free_age.resize(cap + 1);
    int32_t  * fs = l.free_s.data();
    uint8_t  * fa = l.free_age.data();
    uint32_t * ss = q.slot_stamp.data() + o;
    int64_t    nf = 0;
    for (int32_t s = n_sink; s < n_sink + cap; ++s) {
        const int32_t  row = q.ridx[o + s];
        const bool     in  = (row >= 0) & kvq_bit(rq, std::max(row, 0));
        const uint32_t st  = ss[s];
        ss[s]  = in ? now : st;
        fs[nf] = s;
        fa[nf] = row < 0 ? 255 : (uint8_t) std::min<uint32_t>(255, now - st);
        nf += !in;
    }
    std::fill(rq, rq + n_bmw, 0);

    const int64_t n_miss = (int64_t) l.miss.size();
    GGML_ASSERT(nf >= n_miss);
    if (n_miss == 0) {
        return;
    }
    // the age a of the youngest free slot that is taken
    int64_t hist[256] = { 0 };
    for (int64_t j = 0; j < nf; ++j) {
        hist[fa[j]]++;
    }
    int     a     = 255;
    int64_t n_old = 0;
    while (n_old + hist[a] < n_miss) {
        n_old += hist[a--];
    }
    // the free slots older than a, then the first ones of age a, each in slot order
    l.pick.resize(cap + 1);
    int32_t * pk = l.pick.data();
    int64_t   np = 0;
    for (int64_t j = 0; j < nf; ++j) {
        pk[np] = fs[j];
        np += fa[j] > a;
    }
    for (int64_t j = 0; j < nf; ++j) {
        pk[np] = fs[j];
        np += (fa[j] == a) & (np < n_miss);
    }
    GGML_ASSERT(np == n_miss);
    for (int64_t k = 0; k < n_miss; ++k) {
        l.assign(q, g, pk[k], l.miss[k], cap_tot, n_bmw);
    }
}

// the plan of every stream of the ubatch, one after the other (the fetches of the streams follow each other in the list)
void llama_kvq_store::plan_prefetch(layer & l) {
    l.fetch.clear();
    for (int32_t s = 0; s < ub_n; ++s) {
        l.plan_s = s;
        plan_stream(l, s);
    }
    l.plan_s = 0;
}

void llama_kvq_store::plan_stream(layer & l, int32_t st) {
    llama_kvq_counters & cnt = l.cnt;
    layer_seq & q = ls(ub_seq[st], l);
    const int32_t n_coarse_ = seqs[ub_seq[st]].n_coarse;

    const int64_t n_top = l.n_top;
    const int64_t L     = l.n_pos;
    const bool    cut   = pf_scores(n_top, L);

    const uint32_t now = ++q.stamp;
    const size_t   f0  = l.fetch.size();

    cnt.n_pf++;
    cnt.pf_sel += G*L*n_top;
    q.pf_stamp = now;
    for (int32_t g = 0; g < G; ++g) {
        // the union over the positions; the order of top_k on the device is not fixed (atomics), the bitmap does not keep it
        uint64_t * rq = q.req.data() + g*n_bmw;
        for (int64_t p = 0; p < L; ++p) {
            const int32_t * sel = l.h_sel + (g + G*(p + L*st))*n_top;
            for (int64_t i = 0; i < n_top; ++i) {
                GGML_ASSERT(sel[i] >= n_sink && sel[i] < n_coarse_);
                kvq_bit_set(rq, sel[i]);
            }
        }
        if (cut) {
            cut_request(l, q, st, g, n_req(n_top, L));
        }
        const uint64_t * rb = q.res.data() + g*n_bmw;
        l.miss.clear();
        for (int64_t w = 0; w < n_bmw; ++w) {
            if (rq[w] == 0) {
                continue;
            }
            cnt.pf_req += kvq_popcount(rq[w]);
            for (uint64_t m = rq[w] & ~rb[w]; m != 0; m &= m - 1) {
                l.miss.push_back((int32_t) (64*w + ctz64(m)));
            }
        }
        place(l, q, g, now);
    }
    cnt.pf_fetch += l.fetch.size() - f0;
}

// rows, slot list and ridx of the last plan to the pinned staging
// the staging is shared by the layers: the prefetch of a layer is read back after the uploads of the previous join (same copy stream),
// so a gather after the readback finds them done
void llama_kvq_store::gather(layer & l, int64_t n_rows) {
    const int64_t n = (int64_t) l.fetch.size();
    GGML_ASSERT(2*n <= n_rows && n_rows <= n_stage);

    // the previous upload of this layer
    if (l.up_rec) {
        ggml_backend_event_synchronize(l.ev_up.get());
        l.up_rec = false;
    }

    // sorting the rows by address does not make the copy faster (measured)
    pool->run(l.fetch.data(), n, h_stage, this, cur, l.li);

    // the slot of stream s: the slice s of the resident rows, K then V
    for (int64_t i = 0; i < n; ++i) {
        const int64_t s = l.fetch[i].st;
        h_slots[2*i]     = (int32_t) (((2*s + 0)*G + l.fetch[i].g)*cap_tot + l.fetch[i].slot);
        h_slots[2*i + 1] = (int32_t) (((2*s + 1)*G + l.fetch[i].g)*cap_tot + l.fetch[i].slot);
    }
    fill_ridx(l);
}

// the ridx of layer l of every stream of the ubatch to the pinned block
void llama_kvq_store::fill_ridx(layer & l) {
    for (int32_t s = 0; s < ub_n; ++s) {
        const layer_seq & q = ls(ub_seq[s], l);
        memcpy(h_ridx + (size_t) s*G*cap_tot, q.ridx.data(), q.ridx.size()*sizeof(int32_t));
    }
}

// on the stream of backend, after the previous readers of stage, slots and ridx of the layer
void llama_kvq_store::upload(ggml_backend_t backend, layer & l) {
    const int64_t n      = (int64_t) l.fetch.size();
    const int64_t n_list = std::max<int64_t>(2*n, slots_hi);
    for (int64_t j = 2*n; j < n_list; ++j) {
        h_slots[j] = trash_row(j);
    }

    // ridx, the whole slot list and the rows in one copy of the block
    slots_hi = 2*n;
    const size_t nb = off_stage + 2*n*ROW;
    if (ub_n == n_str) {
        ggml_backend_tensor_set_async(backend, jblk, h_ridx, 0, nb);
    } else {
        // the ridx of the streams of the ubatch, then the slot list and the rows
        ggml_backend_tensor_set_async(backend, jblk, h_ridx, 0, (size_t) ub_n*G*cap_tot*sizeof(int32_t));
        ggml_backend_tensor_set_async(backend, jblk, (const uint8_t *) h_ridx + off_slots, off_slots, nb - off_slots);
    }
    ggml_backend_event_record(l.ev_up.get(), backend);
    l.up_rec = true;
    l.cnt.bytes += nb;
}

// worker thread
void llama_kvq_store::run_prefetch(layer & l) {
    ggml_backend_event_synchronize(l.ev_read.get());
    const double t0 = kvq_now();
    plan_prefetch(l);
    const double t1 = kvq_now();
    gather(l, 2*G*n_req(l.n_top, l.n_pos)*ub_n);
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
    work->mark_joined(l);
    work->wait_mirror(l);
    for (const auto & f : l.fetch) {
        layer::forget(ls(ub_seq[f.st], l), f.g, f.slot, cap_tot, n_bmw);
    }
    l.fetch.clear();
    ggml_backend_synchronize(backend_cp.get());
    l.up_rec = false;
    // the base uploads queued for the next joins may be stale now (the buffers are shared)
    reset_base();
    // the layer gets its device ridx at its next join
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

    if (kind == KIND_SCR_UP) {
        // after the readers of the scratch on the main stream, the rows of the coarse region, per kv head
        ggml_backend_t cp = backend_cp.get();
        ggml_backend_event_record(l.ev_su.get(), backend);
        ggml_backend_event_wait(cp, l.ev_su.get());
        // the rows of the ubatch of the attention layer before to the host store, then the rows [0, p0) of this layer
        if (it->second.first > 0) {
            read_scr(layers[it->second.first - 1]);
        }
        const size_t n = (size_t) ub_p0*2*ROW;
        for (int32_t g = 0; g < G; ++g) {
            chunk_runs(0, ub_p0, [&](int32_t p, int32_t m) {
                ggml_backend_tensor_set_async(cp, scr, row_ptr(cur, l.li, g, p), g*scr->nb[3] + p*scr->nb[2], (size_t) m*2*ROW);
            });
        }
        ggml_backend_event_record(l.ev_sd.get(), cp);
        cnt.n_scr++;
        cnt.scr_bytes += (int64_t) G*n;
        cnt.t_scr     += kvq_now() - t0;
        return true;
    }
    if (kind == KIND_SCR_JOIN) {
        ggml_backend_event_wait(backend, l.ev_sd.get());
        cnt.t_scr += kvq_now() - t0;
        return true;
    }
    if (kind == KIND_SCR_RB) {
        ggml_backend_event_record(l.ev_su.get(), backend);
        ggml_backend_event_wait(backend_cp.get(), l.ev_su.get());
        read_scr(l);
        cnt.t_scr += kvq_now() - t0;
        return true;
    }

    const double t_gap = cb_ubatch == n_ubatch ? t0 - cb_end : 0.0;

    if (kind == KIND_JOIN) {
        // the main stream is after the new rows of the layer in its tail buffer
        ggml_backend_event_record(l.ev_cpy.get(), backend);
        // no prefetch (e.g. dropped): nothing to scatter, the slot list is trash
        if (l.pending) {
            wait(l);
        } else {
            // also the tail upload that the job of the layer before may queue
            wait_mirrors();
            work->wait_mirror(l);
            l.fetch.clear();
            ggml_backend_event_record(l.ev_sel.get(), backend);
            ggml_backend_event_wait(backend_cp.get(), l.ev_sel.get());
            // the staging is not ordered by a readback here
            ggml_backend_synchronize(backend_cp.get());
            fill_ridx(l);
        }
        // a rotating layer without its base upload (no prefetch job before it queued one), an own buffer after a reset: now,
        // the main stream is after the attention of the layer before
        if (!l.base_rec && (!l.own || l.own_stale)) {
            ggml_backend_event_record(l.ev_sel.get(), backend);
            issue_base(l, l.ev_sel.get(), backend_cp.get());
        } else if (l.own && l.own_sinks) {
            ggml_backend_event_record(l.ev_sel.get(), backend);
            ggml_backend_event_wait(backend_cp.get(), l.ev_sel.get());
            for (int32_t s = 0; s < ub_n; ++s) {
                for (int32_t m = 0; m < 2*G; ++m) {
                    const size_t off = (size_t) m*cap_tot*ROW;
                    ggml_backend_tensor_set_async(backend_cp.get(), l.t.rkv_h, ls(ub_seq[s], l).mirror + off, (size_t) s*2*G*cap_tot*ROW + off, (size_t) n_sink*ROW);
                }
            }
        }
        l.own_stale = false;
        l.own_sinks = false;
        const double t1 = kvq_now();
        // the copy stream is after ev_sel of this prefetch, so after the previous readers of the layer; the delta after the base upload
        if (l.base_rec) {
            ggml_backend_event_wait(backend_cp.get(), l.ev_base.get());
            l.base_rec = false;
        }
        // the tail of the next layer (its early query comes before its join; the last layer: the first one of the next round),
        // queued by the job of this layer's predecessor, or now on the copy stream (after the readback of the previous user of the buffer)
        layer & ln = layers[(it->second.first + 1) % layers.size()];
        if (!ln.tb_rec) {
            issue_tail(ln, nullptr, backend_cp.get());
        }
        upload(backend_cp.get(), l);
        ggml_backend_event_wait(backend, l.ev_up.get());
        ggml_backend_event_wait(backend, ln.ev_tb.get());
        ln.tb_rec   = false;
        ln.tb_ready = true;
        read_ring(l);
        // the worker queues the next base upload after these uploads
        work->mark_joined(l);
        const double t2 = kvq_now();
        cnt.n_join++;
        cnt.t_join   += t1 - t0;
        cnt.t_upload += t2 - t1;
    } else {
        const ggml_tensor * ids    = nullptr;
        const ggml_tensor * anchor = nullptr;
        select(l, dst, graph, &ids, &anchor);

        const int64_t n_top = ids->ne[0];
        const int64_t L     = ids->ne[1]/G/ub_n; // per stream
        GGML_ASSERT(kind == KIND_PREFETCH && L*G*ub_n == ids->ne[1]);
        GGML_ASSERT(n_top*G*L <= (int64_t) params.prefetch_n*G*params.max_batch);
        GGML_ASSERT(!l.pending);

        l.n_top = n_top;
        l.n_pos = L;

        // the readback on the copy stream, after the selection on the main stream; the scores only for a cut
        ggml_backend_t cp = backend_cp.get();
        ggml_backend_event_record(l.ev_sel.get(), backend);
        ggml_backend_event_wait(cp, l.ev_sel.get());
        ggml_backend_tensor_get_async(cp, ids, l.h_sel, 0, ggml_nbytes(ids));
        if (pf_scores(n_top, L)) {
            GGML_ASSERT(ggml_nelements(anchor) == ggml_nelements(ids));
            ggml_backend_tensor_get_async(cp, anchor, l.h_val, 0, ggml_nbytes(anchor));
        }
        ggml_backend_event_record(l.ev_read.get(), cp);
        // the first layers of a round whose base uploads the jobs of the previous round did not queue (after a reset): now,
        // the main stream is after the previous graph; the worker is idle at the first prefetch
        if (it->second.first == 0) {
            wait_mirrors();
            for (int32_t j = 0; j < n_rot; ++j) {
                if (layers[j].base_early && !layers[j].base_rec) {
                    issue_base(layers[j], l.ev_sel.get(), backend_bu.get());
                }
            }
        }
        l.pending = true;
        work->submit(l);
        cnt.t_read += kvq_now() - t0;
    }

    cb_end    = kvq_now();
    cb_ubatch = n_ubatch;
    cnt.t_wait += t_gap;

    return true;
}

void llama_kvq_store::get_counters(llama_kvq_counters_all & out) {
    flush_ubatch();
    out.layers.clear();
    for (auto & l : layers) {
        out.layers.push_back(l.cnt);
    }
    out.n_ubatch = n_ubatch;
    out.n_tokens = n_tokens;
}

void llama_kvq_store::reset_counters() {
    for (auto & l : layers) {
        l.cnt = {};
        l.cnt.il = l.il;
    }
    ub_tokens = 0;
    n_ubatch  = 0;
    n_tokens  = 0;
}

bool llama_kvq_store::scr_alloc() {
    const int64_t t0 = ggml_time_us();
    // the scratch attention does not read the codes, the fitter writes them to the host copy (prefill_free_codes)
    codes_free();
    if (scr_mode == 1) {
        return true;
    }
    // the queued uploads of the packed path write to the same memory; the next round queues them again
    if (scr_mode == 2) {
        drop_pending();
        reset_base();
    }
    synchronize();
    // the masked rows of a scratch attention must not hold NaN
    ggml_backend_tensor_memset(scr, 0, 0, ggml_nbytes(scr));
    scr_mode = 1;
    LLAMA_LOG_INFO("%s: the scratch (%.1f MiB) in the shared memory in %.2f ms\n", __func__, ggml_nbytes(scr)/1048576.0, (ggml_time_us() - t0)/1000.0);
    return true;
}

void llama_kvq_store::fit_alloc() {
    if (scr_mode == 3) {
        return;
    }
    if (scr_mode == 2) {
        drop_pending();
        reset_base();
    }
    // the readbacks of the last graph read the same memory
    synchronize();
    scr_mode = 3;
}

bool llama_kvq_store::dec_alloc() {
    if (scr_mode == 2) {
        return true;
    }
    const int64_t t0 = ggml_time_us();
    // the readbacks of the last scratch graph read the same memory
    synchronize();
    // a slot that is not resident is skipped by the attention, but must not hold NaN
    for (const auto & b : bufs) {
        ggml_backend_tensor_memset(b.rkv_h, 0, 0, ggml_nbytes(b.rkv_h));
    }
    // the scratch overwrote the own buffers: their mirrors on the base stream now, the joins wait for them (a reset instead would
    // change the rows of a small ubatch at the end of a prompt, it comes before the reset of the end of the prompt)
    for (auto & l : layers) {
        if (l.own) {
            l.own_stale = true;
            l.own_sinks = false;
            issue_base(l, nullptr, backend_bu.get());
        }
    }
    for (const auto & b : tbufs) {
        ggml_backend_tensor_memset(b.k, 0, 0, ggml_nbytes(b.k));
        ggml_backend_tensor_memset(b.v, 0, 0, ggml_nbytes(b.v));
    }
    // a join copies the whole pinned slot list (slots_hi is its state)
    if (jblk) {
        ggml_backend_tensor_set(layers[0].t.slots, h_slots, 0, ggml_nbytes(layers[0].t.slots));
    }
    scr_mode = 2;
    LLAMA_LOG_INFO("%s: the resident, tail and staging buffers in the shared memory in %.2f ms\n", __func__, (ggml_time_us() - t0)/1000.0);
    return true;
}

void llama_kvq_store::codes_written(int32_t seq, int32_t r1) {
    seqs[seq].hc_hi = std::max(seqs[seq].hc_hi, r1);
}

// host copy -> device: the codes of the rows [0, n) of the sequence, the tables of their chunk slots; async: on the main stream (pinned host copy)
void llama_kvq_store::codes_up(int32_t s, int32_t n, bool async) {
    const std::vector<int32_t> & page_tab = seqs[s].page_tab;
    const int32_t n_tab = (n + n_crow - 1)/n_crow;
    for (auto & l : layers) {
        for (int k = 0; k < 2; ++k) {
            ggml_tensor * t = k ? l.t.vc : l.t.kc;
            // one copy per page: the host page section [G][C][row] is the device page [nwords, C, G]
            for (int32_t c = 0; c < n_tab; ++c) {
                const int32_t   pg  = page_tab[c];
                const uint8_t * src = k ? pages->vc_ptr(pg, (int32_t) l.li) : pages->kc_ptr(pg, (int32_t) l.li);
                if (async) {
                    ggml_backend_tensor_set_async(backend_main, t, src, (size_t) pg*t->nb[3], t->nb[3]);
                } else {
                    ggml_backend_tensor_set(t, src, (size_t) pg*t->nb[3], t->nb[3]);
                }
            }
        }
        // the tables: one copy per page (the slots [2G] of the host page are the device page)
        for (int32_t c = 0; c < n_tab; ++c) {
            ggml_tensor *   t   = l.t.tab;
            const uint8_t * src = htab_ptr(s, l.li, c, 0);
            if (async) {
                ggml_backend_tensor_set_async(backend_main, t, src, (size_t) page_tab[c]*t->nb[2], t->nb[2]);
            } else {
                ggml_backend_tensor_set(t, src, (size_t) page_tab[c]*t->nb[2], t->nb[2]);
            }
        }
    }
}

bool llama_kvq_store::codes_alloc() {
    if (buf_codes) {
        return true;
    }
    const int64_t t0 = ggml_time_us();
    buf_codes.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_codes.get(), buft));
    if (!buf_codes) {
        LLAMA_LOG_ERROR("%s: failed to allocate the codes and tables on the device\n", __func__);
        return false;
    }
    // the ops read only the rows of the coarse region and the tables of its chunks; zero is still the safe value elsewhere
    ggml_backend_buffer_clear(buf_codes.get(), 0);
    ggml_backend_buffer_set_usage(buf_codes.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    codes_gen_++;
    // also the rows of the jobs whose codes are in the host copy and that have not joined the coarse region yet
    int32_t n = 0;
    const bool async = backend_main != nullptr;
    const int64_t t1 = ggml_time_us();
    for (int32_t s = 0; s < n_seq(); ++s) {
        // the pages of the sequences are different pages, the chunks of the largest one are the ones logged
        const int32_t ns = std::min(n_rows_, std::max(seqs[s].n_coarse, seqs[s].hc_hi));
        codes_up(s, ns, async);
        n = std::max(n, ns);
    }
    {
        const size_t sz_rows = (size_t) ((n + n_crow - 1)/n_crow)*(layers[0].t.kc->nb[3] + layers[0].t.vc->nb[3])*layers.size();
        const size_t sz_tabs = (size_t) ((n + n_crow - 1)/n_crow)*layers[0].t.tab->nb[2]*layers.size();
        LLAMA_LOG_INFO("%s: codes and tables (%.1f MiB) on the device in %.2f ms, upload of the host copy of the rows [0, %d) and %d chunk slots (%.1f MiB) %s\n", __func__,
                ggml_backend_buffer_get_size(buf_codes.get())/1048576.0, (t1 - t0)/1000.0, n, (n + n_crow - 1)/n_crow, (sz_rows + sz_tabs)/1048576.0,
                async ? "queued" : "done");
    }
    return true;
}

void llama_kvq_store::codes_free() {
    if (!buf_codes || !prefill_free_codes) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    // the queued graphs and fitter phases read and write them
    synchronize();
    buf_codes.reset();
    for (auto & l : layers) {
        for (ggml_tensor * t : { l.t.kc, l.t.vc, l.t.tab }) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
    }
    codes_gen_++;
    LLAMA_LOG_INFO("%s: codes and tables freed on the device in %.2f ms, the host copy holds them\n", __func__, (ggml_time_us() - t0)/1000.0);
}

llama_kvq_memory_info llama_kvq_store::memory_info() const {
    llama_kvq_memory_info mi;
    for (const auto & l : layers) {
        mi.codes      += ggml_nbytes(l.t.kc) + ggml_nbytes(l.t.vc);
        mi.tables     += ggml_nbytes(l.t.tab);
    }
    for (const auto & q : seqs) {
        for (size_t li = 0; li < layers.size(); ++li) {
            mi.other_dev  += (q.qacc[li] ? ggml_nbytes(q.qacc[li]) : 0) + (q.gacc[li] ? ggml_nbytes(q.gacc[li]) : 0);
            const layer_seq & s = q.lay[li];
            mi.host_other += (s.ridx.size() + s.slot_stamp.size())*sizeof(int32_t) + (s.res.size() + s.req.size())*sizeof(uint64_t);
        }
        mi.host_other += q.buf_mir ? ggml_backend_buffer_get_size(q.buf_mir.get()) : 0;
    }
    for (const auto & b : bufs) {
        mi.resident += ggml_nbytes(b.rkv_h);
    }
    for (const auto & b : tbufs) {
        mi.ring += ggml_nbytes(b.k) + ggml_nbytes(b.v);
    }
    mi.host_ring  = layers.size()*2*(size_t) n_cell*G*ROW*n_ring_strm;
    mi.other_dev += ggml_nbytes(bias) + (buf_stage ? ggml_backend_buffer_get_size(buf_stage.get()) : 0);
    mi.scratch    = scr ? ggml_nbytes(scr) : 0;
    mi.host_other += buf_pin ? ggml_backend_buffer_get_size(buf_pin.get()) : 0;
    mi.pages       = pages->n_pages();
    mi.pages_free  = pages->n_free();
    mi.pages_shared = pages->n_shared();
    for (int32_t p = 0; p < pages->n_pages(); ++p) {
        if (pages->allocated(p)) {
            mi.pages_host++;
            mi.host_store += pages->bytes_rows(p);
            mi.host_codes += pages->bytes_codes(p);
        }
    }
    const llama_kvq_pool::stats ps = pages->get_stats();
    mi.n_page_wait = ps.n_wait;
    mi.t_page_wait = ps.t_wait;
    mi.n_page_alloc = ps.n_alloc;
    mi.t_page_alloc = ps.t_alloc;
    mi.n_page_free  = ps.n_free;
    mi.t_page_free  = ps.t_free;
    mi.pinned = pages->pinned();
    return mi;
}

// n free pages for the sequence, see the declaration
bool llama_kvq_store::acquire_pages(int32_t seq, int32_t add, int32_t * pg) {
    if (!pages->pick(add, pg)) {
        LLAMA_LOG_ERROR("%s: the kvq page pool is full: sequence %d needs %d more pages (pages of %d rows), %d of %d are free\n", __func__,
                seq, add, n_crow, pages->n_free(), pages->n_pages());
        return false;
    }
    if (!pages->host_fits(pg, add)) {
        // the memory of released pages may still be on its way back
        pages->wait_frees();
    }
    if (!pages->host_fits(pg, add)) {
        LLAMA_LOG_ERROR("%s: the kvq host memory cap is reached: sequence %d needs %d more pages of %.1f MiB, %.1f of %.1f MiB are allocated or queued (--kvq-host-max-mib)\n", __func__,
                seq, add, pages->bytes(0)/1048576.0, pages->host_bytes()/1048576.0, pages->max_host_bytes()/1048576.0);
        return false;
    }
    // the allocations run in the background thread, the pages are used after the wait (it is counted when it blocks)
    for (int32_t i = 0; i < add; ++i) {
        pages->queue(pg[i]);
    }
    for (int32_t i = 0; i < add; ++i) {
        if (!pages->wait(pg[i])) {
            LLAMA_LOG_ERROR("%s: failed to allocate the host memory of page %d (%.1f MiB) of the kvq pool\n", __func__, pg[i], pages->bytes(pg[i])/1048576.0);
            return false;
        }
    }
    return true;
}

// the pages of the sequence for the chunks of the positions [0, n_pos), see the declaration
bool llama_kvq_store::reserve(int32_t seq, int32_t n_pos) {
    seq_state & q = seqs[seq];
    if (n_pos <= 0) {
        return true;
    }
    const int32_t n_c = (n_pos + n_crow - 1)/n_crow;
    if (n_c > n_chunk) {
        LLAMA_LOG_ERROR("%s: the kvq pool cannot hold %d positions of sequence %d (%d chunks of %d rows)\n", __func__, n_pos, seq, n_chunk, n_crow);
        return false;
    }
    const int32_t add = n_c - q.n_page;
    // the next page is allocated ahead (while this batch runs) once the sequence is in the last quarter of its last page: no idle page is left behind
    // when a request ends early (the host memory of a free page is only given back at the next release)
    const bool near_end = n_pos > (n_c - 1)*n_crow + 3*(n_crow/4);
    if (add <= 0) {
        if (near_end) {
            pages->queue_ahead();
        }
        return true;
    }
    std::vector<int32_t> pg(add);
    if (!acquire_pages(seq, add, pg.data())) {
        return false;
    }
    for (int32_t i = 0; i < add; ++i) {
        pages->take(pg[i]);
        q.page_tab[q.n_page + i] = pg[i];
    }
    q.n_page = n_c;
    pt_up(seq);
    // the page after these is allocated while this batch runs (if the sequence is not near its last page, see above)
    if (near_end) {
        pages->queue_ahead();
    }
    return true;
}

void llama_kvq_store::release_from(int32_t seq, int32_t c0) {
    seq_state & q = seqs[seq];
    c0 = std::max(c0, 0);
    if (c0 >= q.n_page) {
        return;
    }
    // queued graphs and phases may still read the pages and the page table
    synchronize();
    for (int32_t c = c0; c < q.n_page; ++c) {
        pages->give(q.page_tab[c]);
        q.page_tab[c] = -1;
    }
    q.n_page = c0;
    // the host memory of the pages nobody holds goes back to the OS, on the pool thread (the streams are synchronized above)
    const int32_t n_rel = pages->trim();
    if (n_rel > 0) {
        LLAMA_LOG_INFO("%s: sequence %d let its pages from chunk %d go: the pinned host memory of %d pages (%.1f MiB) goes back to the OS, %d pages with %.1f MiB stay\n", __func__,
                seq, c0, n_rel, n_rel*pages->bytes(0)/1048576.0, pages->n_alloc(), pages->host_bytes()/1048576.0);
    }
    pt_up(seq);
}

void llama_kvq_store::release_all() {
    for (int32_t s = 0; s < n_seq(); ++s) {
        release_from(s, 0);
    }
}

void llama_kvq_store::copy_page_data(int32_t from, int32_t to, int32_t n_rows, int32_t n_code, bool sums) {
    const llama_kvq_pool::layout & lay = pages->get_layout();
    for (const auto & l : layers) {
        const int32_t li = (int32_t) l.li;
        for (int32_t g = 0; g < G; ++g) {
            if (n_rows > 0) {
                memcpy(pages->rows_ptr(to, li) + (size_t) g*n_crow*2*ROW, pages->rows_ptr(from, li) + (size_t) g*n_crow*2*ROW, (size_t) n_rows*2*ROW);
            }
            if (n_code > 0) {
                memcpy(pages->kc_ptr(to, li) + (size_t) g*n_crow*lay.kc_row, pages->kc_ptr(from, li) + (size_t) g*n_crow*lay.kc_row, (size_t) n_code*lay.kc_row);
                memcpy(pages->vc_ptr(to, li) + (size_t) g*n_crow*lay.vc_row, pages->vc_ptr(from, li) + (size_t) g*n_crow*lay.vc_row, (size_t) n_code*lay.vc_row);
            }
        }
        if (n_code > 0) {
            memcpy(pages->tab_ptr(to, li), pages->tab_ptr(from, li), (size_t) 2*G*lay.tab_slot);
        }
    }
    if (sums) {
        pages->chunk_sums(to) = pages->chunk_sums(from);
    }
}

void llama_kvq_store::page_up(int32_t pg) {
    if (!codes_on_dev()) {
        return;
    }
    for (auto & l : layers) {
        for (int k = 0; k < 2; ++k) {
            ggml_tensor * t = k ? l.t.vc : l.t.kc;
            ggml_backend_tensor_set(t, k ? pages->vc_ptr(pg, (int32_t) l.li) : pages->kc_ptr(pg, (int32_t) l.li), (size_t) pg*t->nb[3], t->nb[3]);
        }
        ggml_backend_tensor_set(l.t.tab, pages->tab_ptr(pg, (int32_t) l.li), (size_t) pg*l.t.tab->nb[2], l.t.tab->nb[2]);
    }
}

bool llama_kvq_store::fork(int32_t src, int32_t dst, int32_t n_share) {
    seq_state & s = seqs[src];
    seq_state & d = seqs[dst];
    GGML_ASSERT(src != dst && d.n_page == 0 && d.n_coarse == 0 && d.n_host == 0);
    n_share = std::max(0, std::min(n_share, s.n_page));
    const int32_t n_cp = s.n_page - n_share;
    std::vector<int32_t> pg(std::max(n_cp, 1));
    if (n_cp > 0 && !acquire_pages(dst, n_cp, pg.data())) {
        return false;
    }
    // the queued work of the sequences reads the pages and the page tables
    synchronize();
    for (int32_t c = 0; c < n_share; ++c) {
        pages->take(s.page_tab[c]);
        d.page_tab[c] = s.page_tab[c];
    }
    const int32_t hi = std::max(s.n_coarse, s.hc_hi);
    for (int32_t k = 0; k < n_cp; ++k) {
        const int32_t c = n_share + k;
        const int32_t n_rows = std::max(0, std::min(n_crow, s.n_host - c*n_crow));
        const int32_t n_code = std::max(0, std::min(n_crow, hi - c*n_crow));
        pages->take(pg[k]);
        copy_page_data(s.page_tab[c], pg[k], n_rows, n_code, true);
        d.page_tab[c] = pg[k];
        if (n_code > 0) {
            page_up(pg[k]);
        }
    }
    d.n_page = s.n_page;
    pt_up(dst);
    d.hc_hi = s.hc_hi;
    d.n_host = s.n_host;
    set_n_coarse(dst, s.n_coarse);
    LLAMA_LOG_INFO("%s: sequence %d takes %d pages of sequence %d by reference and copies %d (%d rows in the host store, %d coarse rows)\n", __func__,
            dst, n_share, src, n_cp, s.n_host, s.n_coarse);
    return true;
}

void llama_kvq_store::ring_fork(int32_t src, int32_t dst, int32_t p0, int32_t p1) {
    if (p1 <= p0) {
        return;
    }
    const size_t nb1  = (size_t) G*ROW;
    const size_t off0 = ring_off(src);
    const size_t off1 = ring_off(dst);
    for (auto & l : layers) {
        for (int32_t p = p0; p < p1; ++p) {
            memcpy((uint8_t *) l.k_cache->data + off1 + (p % n_cell)*nb1, (uint8_t *) l.k_cache->data + off0 + (p % n_cell)*nb1, nb1);
            memcpy((uint8_t *) l.v_cache->data + off1 + (p % n_cell)*nb1, (uint8_t *) l.v_cache->data + off0 + (p % n_cell)*nb1, nb1);
        }
    }
}

bool llama_kvq_store::make_private(int32_t seq, int32_t c) {
    seq_state & q = seqs[seq];
    if (c < 0 || c >= q.n_page || pages->holders(q.page_tab[c]) <= 1) {
        return true;
    }
    int32_t pg = -1;
    if (!acquire_pages(seq, 1, &pg)) {
        return false;
    }
    synchronize();
    const int32_t old    = q.page_tab[c];
    const int32_t n_rows = std::max(0, std::min(n_crow, q.n_host - c*n_crow));
    const int32_t n_code = std::max(0, std::min(n_crow, std::max(q.n_coarse, q.hc_hi) - c*n_crow));
    pages->take(pg);
    copy_page_data(old, pg, n_rows, n_code, false);
    q.page_tab[c] = pg;
    pages->give(old);
    if (n_code > 0) {
        page_up(pg);
    }
    pt_up(seq);
    LLAMA_LOG_INFO("%s: sequence %d copies the shared page of chunk %d (%d rows)\n", __func__, seq, c, n_rows);
    return true;
}

void llama_kvq_store::pt_up(int32_t seq) {
    seq_state & q = seqs[seq];
    q.pt_h.resize(n_chunk);
    for (int32_t c = 0; c < n_chunk; ++c) {
        q.pt_h[c] = std::max(q.page_tab[c], 0);
    }
    if (backend_main) {
        ggml_backend_tensor_set_async(backend_main, q.pt, q.pt_h.data(), 0, ggml_nbytes(q.pt));
    } else {
        ggml_backend_tensor_set(q.pt, q.pt_h.data(), 0, ggml_nbytes(q.pt));
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kvq_store::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (buf_dev) {
        res[ggml_backend_buffer_get_type(buf_dev.get())] += ggml_backend_buffer_get_size(buf_dev.get());
    }
    if (buf_scr) {
        res[ggml_backend_buffer_get_type(buf_scr.get())] += ggml_backend_buffer_get_size(buf_scr.get());
    }
    if (buf_codes) {
        res[ggml_backend_buffer_get_type(buf_codes.get())] += ggml_backend_buffer_get_size(buf_codes.get());
    }
    for (int32_t p = 0; p < pages->n_pages(); ++p) {
        if (pages->allocated(p)) {
            res[ggml_backend_buffer_get_type(pages->buffer(p).get())] += ggml_backend_buffer_get_size(pages->buffer(p).get());
        }
    }
    return res;
}
