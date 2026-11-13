#include "llama-kvq-store.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kvq-codec.h"
#include "llama-kvq-pack.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

struct llama_kvq_store::layer {
    int32_t il = -1;

    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;

    layer_tensors t;

    uint8_t * store = nullptr; // [G][n_coarse][2*ROW], K row then V row

    std::vector<int32_t>  ridx;       // [G + 1][cap_tot], mirror of t.ridx
    std::vector<int32_t>  row_slot;   // [G][n_coarse], -1 if not resident
    std::vector<uint32_t> row_mark;   // [G][n_coarse], stamp of the last request that listed the row
    std::vector<int32_t>  row_upos;   // [G][n_coarse], index in the union of that request
    std::vector<uint32_t> slot_stamp; // [G][cap_tot], stamp of the last selection
    std::vector<float>    slot_val;   // [G][cap_tot], score of the last selection
    std::vector<int32_t>  prev;       // [G][cap_tot], LRU list of the evictable slots, -1 at the ends
    std::vector<int32_t>  next;
    std::vector<int32_t>  head;       // [G], least recently selected
    std::vector<int32_t>  tail;       // [G]

    uint32_t pf_stamp = 0; // stamp of the last prefetch

    llama_kvq_counters cnt;
};

namespace {

struct kvq_fetch {
    int32_t g;
    int32_t slot;
    int32_t row;
};

struct kvq_cand {
    int32_t row;
    float   val;
};

}

static double kvq_now() {
    return ggml_time_us()*1e-6;
}

// LLAMA_KVQ_GATHER_THREADS > 1: the gather is split over spinning worker threads (experiment)
struct llama_kvq_store::gather_pool {
    std::vector<std::thread> threads;
    std::atomic<int>  gen  = 0;
    std::atomic<int>  done = 0;
    std::atomic<bool> stop = false;
    const kvq_fetch * fetch = nullptr;
    int64_t   n     = 0;
    uint8_t * dst   = nullptr;
    const uint8_t * store = nullptr;
    int64_t   n_coarse = 0;
    int       n_th  = 1;

    void part(int ith) {
        const int64_t i0 = n*ith/n_th;
        const int64_t i1 = n*(ith + 1)/n_th;
        for (int64_t i = i0; i < i1; ++i) {
            memcpy(dst + 2*i*ROW, store + ((size_t) fetch[i].g*n_coarse + fetch[i].row)*2*ROW, 2*ROW);
        }
    }

    explicit gather_pool(int n_th) : n_th(n_th) {
        for (int ith = 1; ith < n_th; ++ith) {
            threads.emplace_back([this, ith]() {
                int seen = 0;
                while (!stop.load(std::memory_order_acquire)) {
                    const int g = gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        std::this_thread::yield();
                        continue;
                    }
                    seen = g;
                    part(ith);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        }
    }

    ~gather_pool() {
        stop = true;
        for (auto & t : threads) {
            t.join();
        }
    }

    void run() {
        done = 0;
        gen.fetch_add(1, std::memory_order_acq_rel);
        part(0);
        while (done.load(std::memory_order_acquire) < n_th - 1) {
            std::this_thread::yield();
        }
    }
};

static int kvq_popcount(uint64_t x) {
    int n = 0;
    for (; x; x &= x - 1) {
        n++;
    }
    return n;
}

llama_kvq_store::llama_kvq_store(const llama_kv_cache & kv, int32_t n_coarse, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
        int32_t resident_cap, bool all_resident, int32_t n_sink) :
    n_coarse_(n_coarse), n_crow(n_chunk_rows), all_res(all_resident) {
    if (kv.get_n_stream() != 1) {
        throw std::runtime_error("packed two-tier mode needs a single KV stream");
    }
    if (n_coarse <= 0 || n_chunk_rows <= 0 || n_coarse % n_chunk_rows != 0 || n_sink < 0 || n_sink >= n_coarse) {
        throw std::runtime_error("packed two-tier mode: bad coarse region");
    }
    if (bits_k < 1 || bits_k > 4 || bits_v < 1 || bits_v > 4 || (!all_resident && resident_cap <= 0)) {
        throw std::runtime_error("packed two-tier mode: bad bits or resident capacity");
    }

    this->n_sink = n_sink;
    n_chunk = n_coarse/n_chunk_rows;
    nwk     = (bits_k*(int32_t) D + 31)/32;
    nwv     = (bits_v*(int32_t) D + 31)/32;
    cap     = all_resident ? n_coarse - n_sink : resident_cap;
    cap_tot = n_sink + cap + 1;

    const auto ids = kv.get_layer_ids();
    GGML_ASSERT(!ids.empty());

    const auto & cells = kv.get_cells(0);
    if ((int64_t) cells.size() < n_coarse) {
        throw std::runtime_error("packed two-tier mode: the cache is smaller than the coarse region");
    }
    for (int32_t j = 0; j < n_coarse; ++j) {
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

    n_stage = all_res ? 2 : 2*(int64_t) G*cap;

    auto init_ctx = [](size_t n) {
        ggml_init_params ip = { n*ggml_tensor_overhead(), nullptr, true };
        ggml_context_ptr res(ggml_init(ip));
        if (!res) {
            throw std::runtime_error("failed to create the packed two-tier context");
        }
        return res;
    };
    ctx_dev   = init_ctx(layers.size()*6 + 3);
    ctx_alias = init_ctx(layers.size());
    ctx_host  = init_ctx(layers.size()*2);

    for (auto & l : layers) {
        layer_tensors & t = l.t;
        t.kc   = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_I32, nwk, n_coarse, G);
        t.vc   = ggml_new_tensor_3d(ctx_dev.get(), GGML_TYPE_I32, nwv + 1, n_coarse, G);
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
        for (int k = 0; k < 2; ++k) {
            t.trig[k] = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, 1);
            ggml_format_name(t.trig[k], "kvq_trig%d-%d", k, l.il);
            trig_of[t.trig[k]] = { (size_t) (&l - layers.data()), k };
        }
    }
    bias    = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_F32, n_coarse);
    staging = ggml_new_tensor_2d(ctx_dev.get(), GGML_TYPE_F16, ROW/2, n_stage);
    slots   = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_I32, n_stage);
    ggml_set_name(bias,    "kvq_bias");
    ggml_set_name(staging, "kvq_staging");
    ggml_set_name(slots,   "kvq_slots");

    buf_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev.get(), buft));
    if (!buf_dev) {
        throw std::runtime_error("failed to allocate the packed two-tier device buffer");
    }
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

    buf_stage.reset(ggml_backend_buft_alloc_buffer(buft_host, n_stage*ROW));
    if (!buf_stage) {
        throw std::runtime_error("failed to allocate the packed two-tier staging buffer");
    }
    stage_host = (uint8_t *) ggml_backend_buffer_get_base(buf_stage.get());

    // precise store, one buffer per layer (a single pinned CUDA buffer is limited to 4 GB)
    const size_t store_size = (size_t) G*n_coarse*2*ROW;
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
    {
        const int64_t n_step = 4096;
        std::vector<uint8_t> tk, tv;
        for (auto & l : layers) {
            const size_t nb1 = l.k_cache->nb[1];
            GGML_ASSERT(nb1 == G*ROW && l.v_cache->nb[1] == nb1);
            for (int64_t r0 = 0; r0 < n_coarse; r0 += n_step) {
                const int64_t n = std::min<int64_t>(n_step, n_coarse - r0);
                tk.resize(n*nb1);
                tv.resize(n*nb1);
                ggml_backend_tensor_get(l.k_cache, tk.data(), r0*nb1, n*nb1);
                ggml_backend_tensor_get(l.v_cache, tv.data(), r0*nb1, n*nb1);
                for (int32_t g = 0; g < G; ++g) {
                    for (int64_t r = 0; r < n; ++r) {
                        uint8_t * dst = l.store + ((size_t) g*n_coarse + r0 + r)*2*ROW;
                        memcpy(dst,       tk.data() + r*nb1 + g*ROW, ROW);
                        memcpy(dst + ROW, tv.data() + r*nb1 + g*ROW, ROW);
                    }
                }
            }
        }
    }

    {
        std::vector<float> b(n_coarse, 0.0f);
        std::fill(b.begin(), b.begin() + n_sink, -INFINITY);
        ggml_backend_tensor_set(bias, b.data(), 0, ggml_nbytes(bias));
    }

    for (auto & l : layers) {
        l.ridx.assign((size_t) (G + 1)*cap_tot, -1);
        l.row_slot.assign((size_t) G*n_coarse, -1);
        l.row_mark.assign((size_t) G*n_coarse, 0);
        l.row_upos.assign((size_t) G*n_coarse, 0);
        l.slot_stamp.assign((size_t) G*cap_tot, 0);
        l.slot_val.assign((size_t) G*cap_tot, 0.0f);
        l.prev.assign((size_t) G*cap_tot, -1);
        l.next.assign((size_t) G*cap_tot, -1);
        l.head.assign(G, -1);
        l.tail.assign(G, -1);
    }
    slots_host.assign(n_stage, 0);

    if (const char * env = getenv("LLAMA_KVQ_GATHER_THREADS"); env != nullptr && atoi(env) > 1) {
        pool = std::make_unique<gather_pool>(atoi(env));
        LLAMA_LOG_INFO("%s: gather on %d threads\n", __func__, atoi(env));
    }

    const llama_kvq_memory_info mi = memory_info();
    LLAMA_LOG_INFO("%s: n_coarse = %d, %d chunks of %d rows, K %d / V %d bits (%d / %d words), resident slots %d + %d sinks%s\n", __func__,
            n_coarse, n_chunk, n_crow, bits_k, bits_v, nwk, nwv, cap, n_sink, all_res ? " (all resident)" : "");
    LLAMA_LOG_INFO("%s: device: codes %.1f MiB, tables %.1f MiB, resident %.1f MiB, other %.1f MiB; host: store %.1f MiB (%s), other %.1f MiB\n", __func__,
            mi.codes/1048576.0, mi.tables/1048576.0, mi.resident/1048576.0, mi.other_dev/1048576.0, mi.host_store/1048576.0,
            mi.pinned ? "pinned" : "pageable", mi.host_other/1048576.0);
}

llama_kvq_store::~llama_kvq_store() = default;

const llama_kvq_store::layer_tensors & llama_kvq_store::get_layer(int32_t il) const {
    return layers.at(layer_of_il.at(il)).t;
}

ggml_tensor * llama_kvq_store::get_ridx(ggml_context * ctx, int32_t il) const {
    ggml_tensor * r = get_layer(il).ridx;
    return ggml_view_2d(ctx, r, cap_tot, G, r->nb[1], 0);
}

bool llama_kvq_store::set_codes(int32_t il, int32_t g, bool is_v, const std::vector<llama_kvq_table> & tabs, const int8_t * codes, const float * gain) {
    const auto it = layer_of_il.find(il);
    if (it == layer_of_il.end() || g < 0 || g >= G || (int32_t) tabs.size() != n_chunk || codes == nullptr || (is_v && gain == nullptr)) {
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
    if (!all_res && (p.prefetch_n < 1 || p.prefetch_n > cap || p.prefetch_n > n_coarse_ - n_sink ||
                     p.sync_m < 1 || p.sync_m > cap || p.sync_m > n_coarse_ - n_sink || p.sync_min_hits < 0 || p.sync_min_hits > p.sync_m)) {
        LLAMA_LOG_ERROR("%s: bad prefetch_n or sync_m/sync_min_hits for %d slots and %d candidates\n", __func__, cap, n_coarse_ - n_sink);
        return false;
    }
    params = p;

    stamp = 0;
    for (auto & l : layers) {
        std::fill(l.ridx.begin(), l.ridx.end(), -1);
        std::fill(l.row_slot.begin(), l.row_slot.end(), -1);
        std::fill(l.row_mark.begin(), l.row_mark.end(), 0);
        std::fill(l.slot_stamp.begin(), l.slot_stamp.end(), 0);
        std::fill(l.slot_val.begin(), l.slot_val.end(), 0.0f);
        l.pf_stamp = 0;

        // pinned rows: the sinks, or every row
        const int32_t n_pin = all_res ? n_coarse_ : n_sink;
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t s = 0; s < n_pin; ++s) {
                l.ridx[(size_t) g*cap_tot + s]     = s;
                l.row_slot[(size_t) g*n_coarse_ + s] = s;
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

        // upload the pinned rows: rkv is [kv][g][slot][ROW]
        std::vector<uint8_t> buf((size_t) n_pin*ROW);
        for (int kv = 0; kv < 2; ++kv) {
            for (int32_t g = 0; g < G; ++g) {
                for (int32_t s = 0; s < n_pin; ++s) {
                    memcpy(buf.data() + (size_t) s*ROW, l.store + ((size_t) g*n_coarse_ + s)*2*ROW + kv*ROW, ROW);
                }
                ggml_backend_tensor_set(l.t.rkv, buf.data(), ((size_t) kv*G + g)*cap_tot*ROW, buf.size());
            }
        }
    }

    // the trash entry: slot cap_tot - 1 of kv head 0, K
    std::fill(slots_host.begin(), slots_host.end(), cap_tot - 1);
    ggml_backend_tensor_set(slots, slots_host.data(), 0, ggml_nbytes(slots));
    slots_hi = 0;

    reset_counters();

    LLAMA_LOG_INFO("%s: max_batch = %d, prefetch_n = %d, lead = %d, sync = %d (m = %d, min hits = %d), stats = %d, all resident = %d\n", __func__,
            p.max_batch, p.prefetch_n, p.lead, p.sync_mode, p.sync_m, p.sync_min_hits, p.stats, p.all_resident);
    return true;
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
    flush_ubatch();
    ub_tokens = n;
    n_ubatch++;
}

void llama_kvq_store::build_fetch(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, kind k, ggml_tensor * score, int64_t n_top) const {
    const layer_tensors & t = get_layer(il);

    const int64_t L = score->ne[2];
    GGML_ASSERT(score->ne[0] == n_coarse_ && score->ne[1] == G);
    GGML_ASSERT(n_top <= n_coarse_ - n_sink);

    // the sinks are never candidates
    ggml_tensor * sc = ggml_add(ctx, ggml_reshape_2d(ctx, score, n_coarse_, G*L), bias);

    ggml_tensor * ids = ggml_top_k(ctx, sc, n_top); // [n_top, G*L]
    ggml_format_name(ids, "kvq_ids%d-%d", (int) k, il);
    // read in the copy callback of a later split, after its last use in the graph
    ggml_set_output(ids);

    // the scores of the selected rows, to rank the union over the positions
    ggml_tensor * vals = ggml_get_rows(ctx, ggml_reshape_3d(ctx, sc, 1, n_coarse_, G*L), ids); // [1, n_top, G*L]
    ggml_format_name(vals, "kvq_vals%d-%d", (int) k, il);

    ggml_tensor * trig = ggml_get_rows(ctx, ggml_reshape_2d(ctx, vals, n_top*G*L, 1), t.trig[k]);
    ggml_format_name(trig, "kvq_trig%d-%d", (int) k, il);

    ggml_backend_t backend = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(b) == dev) {
            backend = b;
        }
    }
    GGML_ASSERT(backend != nullptr);
    ggml_backend_sched_set_tensor_backend(sched, trig, backend);
    ggml_build_forward_expand(gf, trig);

    // the callback of the trigger fills rows [0, n) of the staging tensor and the slot list, the rest of the list is the trash slot
    const int64_t n_rows = 2*G*std::min<int64_t>(cap, n_top*L);
    GGML_ASSERT(n_rows <= n_stage);
    ggml_tensor * src = ggml_view_2d(ctx, staging, staging->ne[0], n_rows, staging->nb[1], 0);
    ggml_tensor * idx = ggml_view_1d(ctx, slots, n_rows, 0);
    ggml_tensor * scatter = ggml_set_rows(ctx, t.rkv_h, src, idx);
    ggml_format_name(scatter, "kvq_scatter%d-%d", (int) k, il);
    ggml_build_forward_expand(gf, scatter);
}

bool llama_kvq_store::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    const auto it = trig_of.find(src);
    if (it == trig_of.end()) {
        return false;
    }
    layer & l = layers[it->second.first];
    const int kind = it->second.second;
    llama_kvq_counters & cnt = l.cnt;

    const double t0 = kvq_now();

    // trig = get_rows(reshape(vals), trigger), vals = get_rows(scores, ids)
    const int n_nodes = ggml_graph_n_nodes(graph);
    const ggml_tensor * trig = nullptr;
    for (int i = 0; i < n_nodes && trig == nullptr; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GET_ROWS && node->src[1] == dst) {
            trig = node;
        }
    }
    GGML_ASSERT(trig != nullptr);
    const ggml_tensor * vals = trig->src[0]->view_src;
    GGML_ASSERT(vals != nullptr && vals->op == GGML_OP_GET_ROWS);
    const ggml_tensor * ids = vals->src[1];
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == vals || node == ids) {
            GGML_ABORT("the selection of layer %d is computed in the same split as its trigger", l.il);
        }
    }
    GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(vals) && ids->type == GGML_TYPE_I32 && vals->type == GGML_TYPE_F32);

    const int64_t n_top = ids->ne[0];
    const int64_t L     = ids->ne[1]/G;

    ggml_backend_synchronize(backend);
    const double t1 = kvq_now();

    // the scheduler waits for the GPU before it calls the callback, so the wait is the gap since the previous callback of this ubatch
    const double t_gap = cb_ubatch == n_ubatch ? t0 - cb_end : 0.0;

    std::vector<int32_t> sel(n_top*G*L);
    std::vector<float>   val(n_top*G*L);
    ggml_backend_tensor_get_async(backend, ids,  sel.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_get_async(backend, vals, val.data(), 0, ggml_nbytes(vals));
    ggml_backend_synchronize(backend);
    const double t2 = kvq_now();

    const uint32_t now = ++stamp;
    std::vector<kvq_fetch> fetch;

    auto touch = [&](int32_t g, int32_t s, float v) {
        const size_t o = (size_t) g*cap_tot;
        l.slot_stamp[o + s] = now;
        l.slot_val[o + s]   = v;
        if (s < n_sink || s == l.tail[g]) {
            return;
        }
        // unlink, then append at the tail
        if (l.prev[o + s] >= 0) {
            l.next[o + l.prev[o + s]] = l.next[o + s];
        } else {
            l.head[g] = l.next[o + s];
        }
        l.prev[o + l.next[o + s]] = l.prev[o + s];
        l.prev[o + s] = l.tail[g];
        l.next[o + s] = -1;
        l.next[o + l.tail[g]] = s;
        l.tail[g] = s;
    };

    auto assign = [&](int32_t g, int32_t s, int32_t row, float v) {
        const size_t o = (size_t) g*cap_tot;
        const int32_t old = l.ridx[o + s];
        if (old >= 0) {
            l.row_slot[(size_t) g*n_coarse_ + old] = -1;
            cnt.evict++;
        }
        l.ridx[o + s] = row;
        l.row_slot[(size_t) g*n_coarse_ + row] = s;
        touch(g, s, v);
        fetch.push_back({ g, s, row });
    };

    if (kind == KIND_PREFETCH) {
        cnt.n_pf++;
        l.pf_stamp = now;
        std::vector<kvq_cand> u;
        for (int32_t g = 0; g < G; ++g) {
            // union over the positions, the max score of a row
            u.clear();
            for (int64_t p = 0; p < L; ++p) {
                const int64_t o = (g + G*p)*n_top;
                for (int64_t i = 0; i < n_top; ++i) {
                    const int32_t row = sel[o + i];
                    GGML_ASSERT(row >= n_sink && row < n_coarse_);
                    const size_t ir = (size_t) g*n_coarse_ + row;
                    if (l.row_mark[ir] != now) {
                        l.row_mark[ir] = now;
                        l.row_upos[ir] = (int32_t) u.size();
                        u.push_back({ row, val[o + i] });
                    } else {
                        u[l.row_upos[ir]].val = std::max(u[l.row_upos[ir]].val, val[o + i]);
                    }
                }
            }
            if ((int64_t) u.size() > cap) {
                std::nth_element(u.begin(), u.begin() + cap, u.end(), [](const kvq_cand & a, const kvq_cand & b) { return a.val > b.val; });
                u.resize(cap);
            }
            cnt.pf_req += u.size();

            // the resident rows first, so that the victims of the missing ones are older than this request
            for (const auto & c : u) {
                const int32_t s = l.row_slot[(size_t) g*n_coarse_ + c.row];
                if (s >= 0) {
                    touch(g, s, c.val);
                }
            }
            for (const auto & c : u) {
                if (l.row_slot[(size_t) g*n_coarse_ + c.row] < 0) {
                    const int32_t s = l.head[g];
                    GGML_ASSERT(s >= 0 && l.slot_stamp[(size_t) g*cap_tot + s] != now);
                    assign(g, s, c.row, c.val);
                }
            }
        }
        cnt.pf_fetch += fetch.size();
    } else {
        cnt.n_tu++;
        cnt.n_pos += L;

        const bool    do_fetch = params.sync_mode != LLAMA_KVQ_SYNC_OFF;
        const int64_t thr      = params.sync_mode == LLAMA_KVQ_SYNC_ALWAYS ? n_top : params.sync_min_hits;

        std::vector<int32_t> hits(G*L, 0);
        for (int64_t r = 0; r < G*L; ++r) {
            const int32_t g = (int32_t) (r % G);
            for (int64_t i = 0; i < n_top; ++i) {
                hits[r] += l.row_slot[(size_t) g*n_coarse_ + sel[r*n_top + i]] >= 0;
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
            const int32_t * rows = sel.data() + r*n_top;
            for (int64_t i = 0; i < n_top; ++i) {
                const int32_t s = l.row_slot[(size_t) g*n_coarse_ + rows[i]];
                if (s >= 0) {
                    touch(g, s, INFINITY);
                }
            }
            if (hits[r] >= thr) {
                continue;
            }
            cnt.tu_fire++;
            stall |= 1ull << p;
            for (int64_t i = 0; i < n_top; ++i) {
                if (l.row_slot[(size_t) g*n_coarse_ + rows[i]] < 0) {
                    assign(g, victim(g), rows[i], INFINITY);
                }
            }
        }
        cnt.tu_stall += kvq_popcount(stall);
        ub_fired |= stall;
        cnt.tu_fetch += fetch.size();

        // the flag of ggml_kvq_attn_cond
        l.ridx[(size_t) G*cap_tot] = fetch.empty() ? 0 : 1;
        cnt.tu_redo += !fetch.empty();
    }

    // gather the rows and their slots
    const int64_t n = (int64_t) fetch.size();
    GGML_ASSERT(2*n <= n_stage);
    if (pool && n >= 64) {
        pool->fetch    = fetch.data();
        pool->n        = n;
        pool->dst      = stage_host;
        pool->store    = l.store;
        pool->n_coarse = n_coarse_;
        pool->run();
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const kvq_fetch & f = fetch[i];
            memcpy(stage_host + 2*i*ROW, l.store + ((size_t) f.g*n_coarse_ + f.row)*2*ROW, 2*ROW);
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        slots_host[2*i]     = (int32_t) ((0*G + fetch[i].g)*cap_tot + fetch[i].slot);
        slots_host[2*i + 1] = (int32_t) ((1*G + fetch[i].g)*cap_tot + fetch[i].slot);
    }
    const int64_t n_list = std::max<int64_t>(2*n, slots_hi);
    std::fill(slots_host.begin() + 2*n, slots_host.begin() + n_list, cap_tot - 1);
    const double t3 = kvq_now();

    if (n > 0) {
        ggml_backend_tensor_set_async(backend, staging, stage_host, 0, 2*n*ROW);
    }
    if (n_list > 0) {
        ggml_backend_tensor_set_async(backend, slots, slots_host.data(), 0, n_list*sizeof(int32_t));
    }
    slots_hi = 2*n;
    ggml_backend_tensor_set_async(backend, l.t.ridx, l.ridx.data(), 0, ggml_nbytes(l.t.ridx));
    // the index read by the trigger
    ggml_backend_tensor_set_async(backend, dst, &zero, 0, sizeof(int32_t));

    static const bool sync_upload = getenv("LLAMA_KVQ_SYNC_UPLOAD") != nullptr;
    if (sync_upload) {
        ggml_backend_synchronize(backend);
    }
    const double t4 = kvq_now();

    cb_end    = t4;
    cb_ubatch = n_ubatch;

    cnt.bytes    += 2*n*ROW + n_list*sizeof(int32_t) + ggml_nbytes(l.t.ridx);
    cnt.t_wait   += t_gap + (t1 - t0);
    cnt.t_read   += t2 - t1;
    cnt.t_gather += t3 - t2;
    cnt.t_upload += t4 - t3;

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
        mi.host_store += (size_t) G*n_coarse_*2*ROW;
        mi.host_other += l.ridx.size()*sizeof(int32_t) + (l.row_slot.size() + l.row_mark.size() + l.row_upos.size())*sizeof(int32_t) +
                         (l.slot_stamp.size() + l.slot_val.size() + l.prev.size() + l.next.size())*sizeof(int32_t);
    }
    mi.other_dev  = ggml_nbytes(bias) + ggml_nbytes(staging) + ggml_nbytes(slots);
    mi.host_other += (size_t) n_stage*ROW + slots_host.size()*sizeof(int32_t);
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
