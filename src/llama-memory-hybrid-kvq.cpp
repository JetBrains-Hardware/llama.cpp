#include "llama-memory-hybrid-kvq.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_memory_hybrid_kvq
//

// the cells of the ring of the attention cache
static uint32_t kvq_ring_cells(uint32_t n_tail, uint32_t kv_size) {
    GGML_ASSERT(n_tail > 0 && "the two-tier cache needs the attention cache as a ring (llama_kvq_config::tail)");
    return std::min(n_tail, kv_size);
}

llama_memory_hybrid_kvq::llama_memory_hybrid_kvq(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   rs_replay,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                 uint32_t   n_tail) :
    llama_memory_hybrid(
        model,
        no_alloc,
        type_k, type_v, v_trans, kvq_ring_cells(n_tail, kv_size), n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, rs_replay, offload, unified,
        filter_attn, filter_recr,
        /* attn_rot */ false),
    model(model),
    n_ctx_max(kv_size) {
    get_mem_attn()->set_ring(true);
    LLAMA_LOG_INFO("%s: tail: the attention cache is a ring of %u cells for %u positions\n", "llama_memory_hybrid_kvq", get_mem_attn()->get_size(), n_ctx_max);
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    std::vector<llama_ubatch> ubatches;
    slot_info_vec_t heads_attn;

    // the slots of the whole batch are placed here, so the ring needs the room for all of it now
    if (fitter && !tail_wait(balloc)) {
        return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    if (!prepare_batch(balloc, n_ubatch, embd_all, ubatches, heads_attn)) {
        return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_memory_hybrid_kvq_context>(this, std::move(heads_attn), std::move(ubatches));
}

llama_pos llama_memory_hybrid_kvq::n_pos() const {
    const llama_pos p = get_mem_attn()->seq_pos_max(0);
    return std::max<llama_pos>(p + 1, store ? store->n_host() : 0);
}

bool llama_memory_hybrid_kvq::tail_wait(const llama_batch_allocr & balloc) {
    const llama_pos p0    = balloc.seq_pos_min(0);
    const llama_pos p_end = balloc.seq_pos_max(0);
    const int64_t   size  = get_mem_attn()->get_size();
    if (p_end < 0) {
        return true;
    }
    const llama_kv_cache & kv = *get_mem_attn();
    const int64_t t0 = ggml_time_us();

    // small batches attend through the packed path: the jobs in flight join first, so the path does not depend on their timing
    const bool small = balloc.get_n_tokens() <= (uint32_t) params.max_batch;
    if (small && fitter->has_jobs()) {
        fitter->wait_jobs();
        tail_release();
    }

    // generation: the fold position depends on the first position of the batch only
    const llama_pos last = small ? std::max<llama_pos>(p_end, p0 + params.max_batch - 1) : p_end;
    auto room = [&]() { return last + 1 - (int64_t) n_released <= size; };
    const bool fold_now = small && fold_every > 0 && p0 - store->n_coarse() >= fold_every;

    // host ring: a large batch reads the rows before it from the host store
    if (!small) {
        store->sync_store(n_pos());
    }

    bool waited = false;
    if (!room() && !small) {
        // the complete chunks before the batch, then the rest of the ring rows to the host store
        fitter->poll(kv, n_pos(), true);
        tail_release();
        if (!room()) {
            fitter->spill(kv, p0);
            tail_release();
        }
        waited = true;
    }
    if ((!room() || fold_now) && small) {
        // the rows after the last complete chunk are fitted in line
        const int64_t t1 = ggml_time_us();
        if (!settle(p0)) {
            return false;
        }
        n_fold++;
        t_fold += (ggml_time_us() - t1)*1e-6;
        LLAMA_LOG_INFO("%s: fold at position %d: coarse region %d rows, %d positions in the ring, %.3f s (%d folds, %.2f s in all)\n", __func__,
                p0, store->n_coarse(), (int) (p0 - n_released), (ggml_time_us() - t1)*1e-6, n_fold, t_fold);
    }
    if (!room()) {
        llama_kvq_fit_progress pr;
        fitter->get_progress(pr);
        LLAMA_LOG_ERROR("%s: the ring of %lld cells cannot hold the positions [%d, %d] of the batch: %s\n", __func__,
                (long long) size, n_released, last, pr.failed ? "the fit failed" : "the batch is too large for the ring (see llama_kvq_config::tail)");
        return false;
    }
    if (waited) {
        n_tail_wait++;
        t_tail_wait += (ggml_time_us() - t0)*1e-6;
        LLAMA_LOG_DEBUG("%s: the batch up to position %d made room in the ring in %.3f s (%d batches, %.2f s in all)\n", __func__,
                p_end, (ggml_time_us() - t0)*1e-6, n_tail_wait, t_tail_wait);
    }
    return true;
}

void llama_memory_hybrid_kvq::tail_release() {
    if (!store) {
        return;
    }
    const llama_pos n = store->n_host();
    if (n > n_released) {
        get_mem_attn()->seq_rm(0, n_released, n);
        n_released = n;
    }
}

bool llama_memory_hybrid_kvq::tail_restore(llama_pos p_lo, llama_pos p_hi) {
    if (!store || p_hi <= p_lo) {
        return true;
    }
    llama_kv_cache * kv = get_mem_attn();
    const int64_t size = kv->get_size();
    if (n_pos() - p_lo > size) {
        LLAMA_LOG_ERROR("%s: the ring of %lld cells cannot hold the positions [%d, %d)\n", __func__, (long long) size, p_lo, n_pos());
        return false;
    }
    GGML_ASSERT(p_lo >= store->n_coarse() && p_hi == store->n_host() && n_released == p_hi);
    // the host store keeps the rows, the next packed ubatch copies them into the host ring
    kv->ring_add(0, p_lo, p_hi);
    store->set_n_host(p_lo);
    n_released = p_lo;
    return true;
}

bool llama_memory_hybrid_kvq::settle(llama_pos n, bool end) {
    store->sync_store(n_pos());
    // the end of a prompt: the context frees its prompt buffers, then the device codes are allocated and the host copy uploaded
    // while the CPU fits the last rows
    const llama_pos e = fitter->settle(*get_mem_attn(), n, [&](llama_pos e_lay) {
        if (end && e_lay > 0) {
            if (tg_cb) {
                tg_cb();
            }
            // the result is ignored on purpose: a failure is retried before the first decode graph (llama-context.cpp, process_ubatch, kctx->codes_alloc) and returns ALLOC_FAILED
            store->codes_alloc();
        }
    });
    tail_release();
    if (store->n_host() > e && !tail_restore(e, store->n_host())) {
        return false;
    }
    // the decoding after a prompt depends on the memory state, not on the resident rows of an earlier reply
    store->reset_resident();
    return true;
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_full() {
    return std::make_unique<llama_memory_hybrid_kvq_context>(this);
}

bool llama_memory_hybrid_kvq::get_can_shift() const {
    return false;
}

void llama_memory_hybrid_kvq::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (fitter) {
        fitter->remove(0);
        n_released = 0;
        return;
    }

    store.reset();
    params.enabled = false;
}

bool llama_memory_hybrid_kvq::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (fitter) {
        if (seq_id > 0) {
            GGML_ABORT("seq_rm of sequence %d is not supported by the two-tier KV cache (--kvq): the cache holds one sequence only (seq 0). Use n_seq_max = 1 (server: -np 1), or run without --kvq.", seq_id);
        }
        // the attention positions: a restored recurrent checkpoint can end before them; the ring is empty when they end with the host store
        const llama_pos pos_max = n_pos() - 1;
        const llama_pos q0      = std::max<llama_pos>(p0, 0);
        const llama_pos q1      = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
        if (q0 >= q1 || q0 > pos_max) {
            return llama_memory_hybrid::seq_rm(seq_id, p0, p1);
        }
        if (q1 <= pos_max) {
            GGML_ABORT("seq_rm of positions [%d, %d) is not supported by the two-tier KV cache (--kvq): it can only remove the end of the sequence (here [p0, -1) with p0 <= %d). Removing a middle range is not supported. Disable the context shift and --cache-reuse, or run without --kvq.", p0, p1, pos_max);
        }
        if (!llama_memory_hybrid::seq_rm(seq_id, p0, p1)) {
            return false;
        }
        // the rows before q0 that leave the coarse region stay in the host store, the next fit_wait fits them again
        fitter->remove(q0);
        n_released = std::min(n_released, q0);
        return true;
    }

    if (store && (p1 < 0 || p1 > 0) && (p0 < 0 || p0 < store->n_coarse())) {
        GGML_ABORT("seq_rm of positions [%d, %d) is not supported by the two-tier KV cache (--kvq): it would remove the packed coarse region [0, %d), only the end of the sequence after it can be removed. Disable the context shift and --cache-reuse, or run without --kvq.", p0, p1, store->n_coarse());
    }

    return llama_memory_hybrid::seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_kvq::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (params.enabled || store) {
        GGML_ABORT("seq_cp is not supported by the two-tier KV cache (--kvq): the cache holds one sequence only (seq 0). Use n_seq_max = 1 (server: -np 1), or run without --kvq.");
    }

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_kvq::seq_keep(llama_seq_id seq_id) {
    if ((params.enabled || store) && seq_id != 0) {
        GGML_ABORT("seq_keep of sequence %d is not supported by the two-tier KV cache (--kvq): the cache holds one sequence only (seq 0). Use n_seq_max = 1 (server: -np 1), or run without --kvq.", seq_id);
    }

    llama_memory_hybrid::seq_keep(seq_id);
}

void llama_memory_hybrid_kvq::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (params.enabled) {
        GGML_ABORT("seq_add is not supported by the two-tier KV cache (--kvq): qwen35 uses IMROPE, its keys cannot be shifted in any KV cache (llama_memory_can_shift is false without --kvq too). Disable the context shift and --cache-reuse.");
    }

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_kvq::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (params.enabled) {
        GGML_ABORT("seq_div is not supported by the two-tier KV cache (--kvq): qwen35 uses IMROPE, its keys cannot be shifted in any KV cache (llama_memory_can_shift is false without --kvq too). Disable the context shift (self-extend).");
    }

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_kvq::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (store) {
        for (const auto & buft_size : store->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

llama_pos llama_memory_hybrid_kvq::seq_pos_max(llama_seq_id seq_id) const {
    const llama_pos res = llama_memory_hybrid::seq_pos_max(seq_id);
    if (store && seq_id == 0 && res < 0 && store->n_host() > 0) {
        return std::min<llama_pos>(store->n_host() - 1, get_mem_recr()->seq_pos_max(seq_id));
    }
    return res;
}

// state of the fit mode, sequence 0 (seq_id -1 is the same):
//   magic, version, then six sections, each as (u64 byte size, data):
//     store meta, fitter meta (layout, bits, rem policy, min rows; checked before any data is read)
//     ring: the plain state of the attention cache, the positions after the rows of the host store
//     recurrent: the plain state
//     store data: host rows [0, n_host), codes and tables of the coarse region [0, n_coarse) (it can end inside a chunk)
//     fitter data: the fitted range of each chunk slot, statistics of the complete chunks, sums of the open chunk
//   a reader checks that each section consumed exactly its size and names the section otherwise
// a plain state starts with the stream count of the attention cache, so neither reader takes the state of the other
// version 2: the coarse region ends at any row, the host store can hold rows after it
// version 3: a size per section; rem policy and min rows in the fitter meta
static constexpr uint32_t KVQ_STATE_MAGIC   = 0x5351564b; // "KVQS"
static constexpr uint32_t KVQ_STATE_VERSION = 3;

// counts the bytes a section will write; the sizes are written before the data, the writers cannot seek back
namespace {
class kvq_io_count : public llama_io_write_i {
public:
    void write(const void *, size_t size) override { n += size; }
    void write_tensor(ggml_tensor *, size_t, size_t size) override { n += size; }
    size_t n_bytes() override { return n; }
private:
    size_t n = 0;
};

// the section is written twice: counted first (no copies), then for real, and the real size is checked against the count
template <typename F>
void kvq_section_write(llama_io_write_i & io, const char * name, F && f) {
    kvq_io_count c;
    f(c);
    const uint64_t size = c.n_bytes();
    io.write(&size, sizeof(size));
    const size_t b0 = io.n_bytes();
    f(io);
    if (io.n_bytes() - b0 != size) {
        throw std::runtime_error(format("kvq state section '%s' wrote %zu bytes, counted %zu", name, io.n_bytes() - b0, (size_t) size));
    }
}

template <typename F>
void kvq_section_read(llama_io_read_i & io, const char * name, F && f) {
    uint64_t size = 0;
    io.read(&size, sizeof(size));
    const size_t b0 = io.n_bytes();
    f();
    const size_t got = io.n_bytes() - b0;
    if (got != size) {
        throw std::runtime_error(format("kvq state section '%s' holds %zu bytes, this build read %zu (saved by another build of the two-tier cache; process the prompt again)",
                name, (size_t) size, got));
    }
}
} // namespace

void llama_memory_hybrid_kvq::state_check(llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (!fitter) {
        throw std::runtime_error("the K/V state of the two-tier cache can be saved and loaded only in the fit mode (--kvq)");
    }
    if (seq_id > 0) {
        throw std::runtime_error(format("the two-tier cache holds sequence 0 only, not %d", seq_id));
    }
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        throw std::runtime_error("the K/V state of the two-tier cache cannot be kept on the device");
    }
}

void llama_memory_hybrid_kvq::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) || (!params.enabled && !store)) {
        llama_memory_hybrid::state_write(io, seq_id, flags);
        return;
    }
    state_check(seq_id, flags);

    // no job in flight in a saved state; the state is not fitted further
    const int64_t t0 = ggml_time_us();
    fitter->wait_jobs();
    const_cast<llama_memory_hybrid_kvq *>(this)->tail_release();
    const double t_wait = (ggml_time_us() - t0)*1e-6;
    if (t_wait >= 0.001) {
        LLAMA_LOG_INFO("%s: waited %.3f s for the fitter, the coarse region has %d rows\n", __func__, t_wait, store->n_coarse());
    }

    io.write(&KVQ_STATE_MAGIC,   sizeof(KVQ_STATE_MAGIC));
    io.write(&KVQ_STATE_VERSION, sizeof(KVQ_STATE_VERSION));
    kvq_section_write(io, "store meta",  [&](llama_io_write_i & w) { store->state_write_meta(w); });
    kvq_section_write(io, "fitter meta", [&](llama_io_write_i & w) { fitter->state_write_meta(w); });

    // host ring: the ring cells hold the rows after the host store, the readbacks of the last reply are done
    store->sync_ring(n_pos());
    store->synchronize();
    kvq_section_write(io, "ring",      [&](llama_io_write_i & w) { get_mem_attn()->state_write(w, 0, flags); });
    kvq_section_write(io, "recurrent", [&](llama_io_write_i & w) { get_mem_recr()->state_write(w, seq_id, flags); });

    kvq_section_write(io, "store data",  [&](llama_io_write_i & w) { store->state_write_data(w); });
    kvq_section_write(io, "fitter data", [&](llama_io_write_i & w) { fitter->state_write_data(w); });
}

void llama_memory_hybrid_kvq::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) || (!params.enabled && !store)) {
        llama_memory_hybrid::state_read(io, seq_id, flags);
        return;
    }
    state_check(seq_id, flags);

    // the memory starts empty and stays empty after a failed read: cancels the jobs, drops the coarse region
    clear(false);

    try {
        uint32_t magic   = 0;
        uint32_t version = 0;
        io.read(&magic, sizeof(magic));
        if (magic != KVQ_STATE_MAGIC) {
            throw std::runtime_error("not a state of the two-tier cache (saved without --kvq?)");
        }
        io.read(&version, sizeof(version));
        if (version != KVQ_STATE_VERSION) {
            throw std::runtime_error(format("version %u of the two-tier cache state, this build reads version %u "
                    "(saved by another build of the two-tier cache; process the prompt again)", version, KVQ_STATE_VERSION));
        }
        int32_t n = 0;
        int32_t h = 0;
        kvq_section_read(io, "store meta",  [&]() { store->state_read_meta(io, n, h); });
        kvq_section_read(io, "fitter meta", [&]() { fitter->state_read_meta(io); });

        // host ring: no readback of an earlier reply may land in the ring cells after they are read
        store->synchronize();
        kvq_section_read(io, "ring",      [&]() { get_mem_attn()->state_read(io, 0, flags); });
        kvq_section_read(io, "recurrent", [&]() { get_mem_recr()->state_read(io, seq_id, flags); });

        const llama_pos p_min = get_mem_attn()->seq_pos_min(0);
        const llama_pos p_max = get_mem_attn()->seq_pos_max(0);
        if (p_max >= 0 && (p_min != h || p_max >= (llama_pos) n_ctx_max)) {
            throw std::runtime_error(format("the ring holds positions [%d, %d], expected the positions after the %d rows of the host store, below %u", p_min, p_max, h, n_ctx_max));
        }

        kvq_section_read(io, "store data",  [&]() { store->state_read_data(io, n, h); });
        kvq_section_read(io, "fitter data", [&]() { fitter->state_read_data(io); });
        n_released = h;
        store->ring_loaded(p_max >= 0 ? p_max + 1 : h);

        // the resident slots start empty
        if (!store->enable(params)) {
            throw std::runtime_error("failed to reset the resident rows");
        }
        LLAMA_LOG_INFO("%s: coarse region of %d rows, %d rows in the host store, %d positions in the ring\n", __func__,
                n, h, p_max >= 0 ? p_max - p_min + 1 : 0);
    } catch (...) {
        clear(false);
        throw;
    }
}

bool llama_memory_hybrid_kvq::set_params(const llama_kvq_params & p) {
    if (p.enabled) {
        // one stream: the cell flags come from the positions of a single sequence
        if (get_mem_attn()->get_n_stream() != 1) {
            LLAMA_LOG_ERROR("%s: two-tier mode needs a single KV stream\n", __func__);
            return false;
        }
        if (p.n_sink < 0) {
            LLAMA_LOG_ERROR("%s: bad parameters\n", __func__);
            return false;
        }
        if (!store || !store->enable(p)) {
            LLAMA_LOG_ERROR("%s: packed mode needs llama_memory_kvq_fit_enable and matching parameters\n", __func__);
            return false;
        }
    }

    params = p;

    LLAMA_LOG_INFO("%s: enabled = %d, n_sink = %d\n", __func__, params.enabled, params.n_sink);

    return true;
}

bool llama_memory_hybrid_kvq::fit_enable(const llama_kvq_fit_params & fp, const llama_kvq_params & p) {
    if (!p.enabled) {
        LLAMA_LOG_ERROR("%s: the fit mode needs enabled parameters\n", __func__);
        return false;
    }
    const llama_kv_cache * kv = get_mem_attn();
    if (kv->get_n_stream() != 1 || kv->seq_pos_max(0) >= 0) {
        LLAMA_LOG_ERROR("%s: the fit mode needs an empty memory with a single KV stream\n", __func__);
        return false;
    }
    // the smallest coarse region: the prefetch selects prefetch_n rows that are not sinks
    const int32_t min_first = p.prefetch_n + p.n_sink;
    if (fp.chunk <= p.n_sink || fp.chunk < min_first || fp.min_rows < 1 || (int64_t) n_ctx_max < fp.chunk ||
            (int64_t) kv->get_size() < 2*p.max_batch + 256) {
        LLAMA_LOG_ERROR("%s: bad chunk (%d, %u positions, cache of %u cells) or min_rows (%d)\n", __func__, fp.chunk, n_ctx_max, kv->get_size(), fp.min_rows);
        return false;
    }
    if (backends.empty()) {
        LLAMA_LOG_ERROR("%s: the memory has no backends\n", __func__);
        return false;
    }

    // V is fitted per chunk with the o_proj weight and the gate statistics
    std::vector<const ggml_tensor *> wo;
    int32_t n_head = 0;
    for (int32_t il : kv->get_layer_ids()) {
        const auto & layer = model.layers[il];
        if (layer.wo == nullptr || layer.wo_s != nullptr || model.hdmd.rot.count(layer.wo) != 0 || (n_head != 0 && n_head != (int32_t) model.hparams.n_head(il))) {
            LLAMA_LOG_ERROR("%s: per-chunk V does not support the o_proj weight of layer %d\n", __func__, il);
            return false;
        }
        wo.push_back(layer.wo);
        n_head = model.hparams.n_head(il);
    }

    params.enabled = false;
    fitter.reset();
    store.reset();
    try {
        const int32_t n_rows = (int32_t) n_ctx_max;
        const int32_t n_gate = n_head*(int32_t) llama_kvq_store::D;
        // the fitter works in the shared memory of the store
        const int32_t G      = (int32_t) model.hparams.n_head_kv(kv->get_layer_ids()[0]);
        const size_t  sh_min = llama_kvq_fitter::shared_min(G, llama_kvq_fitter::stage_rows(n_rows, fp), fp.bits_k, fp.bits_v);
        store = std::make_unique<llama_kvq_store>(*kv, n_rows, fp.chunk, fp.bits_k, fp.bits_v, llama_kvq_resident_slots(p), p.n_sink, p.layer_bufs, n_gate, sh_min);
        store->set_backend(backends);
        if (!store->scratch_ok()) {
            throw std::runtime_error("the two-tier cache needs the scratch attention (pinned host memory, LLAMA_KVQ_SCRATCH not 0)");
        }
        fitter = std::make_unique<llama_kvq_fitter>(*store, fp, wo, n_head, min_first);
        // the graph does not read the ring cells, see llama_kvq_store (host ring)
        get_mem_attn()->move_bufs(store->host_buft());
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        fitter.reset();
        store.reset();
        return false;
    }
    if (!set_params(p)) {
        fitter.reset();
        store.reset();
        return false;
    }
    n_released  = 0;
    n_tail_wait = 0;
    t_tail_wait = 0.0;
    n_fold      = 0;
    t_fold      = 0.0;
    n_settle    = 0;
    t_settle    = 0.0;
    fold_every  = getenv("LLAMA_KVQ_FOLD_EVERY") ? std::max(0, atoi(getenv("LLAMA_KVQ_FOLD_EVERY"))) : 0;
    if (fold_every > 0) {
        LLAMA_LOG_WARN("%s: LLAMA_KVQ_FOLD_EVERY = %d: the reply rows are fitted in line when %d of them follow the coarse region\n", __func__, fold_every, fold_every);
    }
    return true;
}

int32_t llama_memory_hybrid_kvq::fit_graph_kind() const {
    if (!params.enabled || !store) {
        return 0;
    }
    return store->get_layer(store->layer_ids()[0]).gacc ? 2 : 1;
}

void llama_memory_hybrid_kvq::fit_poll(uint32_t n_tokens) {
    if (fitter) {
        fitter->poll(*get_mem_attn(), n_pos(), n_tokens > (uint32_t) params.max_batch);
        tail_release();
    }
}

bool llama_memory_hybrid_kvq::fit_wait() {
    if (!fitter) {
        return false;
    }
    const int64_t t0 = ggml_time_us();
    const int32_t c0 = store->n_coarse();
    const bool ok = settle(n_pos(), true);
    if (store->n_coarse() != c0) {
        n_settle++;
        t_settle += (ggml_time_us() - t0)*1e-6;
    }
    return ok;
}

bool llama_memory_hybrid_kvq::prompt_bufs_alloc() {
    return !store || !store->scratch_ok() || store->scr_alloc();
}


bool llama_memory_hybrid_kvq::fit_get_progress(llama_kvq_fit_progress & out) const {
    if (!fitter) {
        return false;
    }
    fitter->get_progress(out);
    out.n_tail_wait = n_tail_wait;
    out.t_tail_wait = t_tail_wait;
    out.n_fold      = n_fold;
    out.t_fold      = t_fold;
    out.n_settle    = n_settle;
    out.t_settle    = t_settle;
    return true;
}

bool llama_memory_hybrid_kvq::fit_set_hold(bool hold) {
    if (!fitter) {
        return false;
    }
    fitter->set_hold(hold);
    return true;
}

void llama_memory_hybrid_kvq::set_backends(const std::vector<ggml_backend_t> & b) {
    backends = b;
    if (store) {
        store->set_backend(backends);
    }
}

bool llama_memory_hybrid_kvq::sched_copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return store && store->copy(backend, src, dst, graph);
}

bool llama_memory_kvq_fit_enable(llama_memory_t mem, const llama_kvq_fit_params & fparams, const llama_kvq_params & params) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_enable(fparams, params);
}

bool llama_memory_kvq_fit_get_progress(llama_memory_t mem, llama_kvq_fit_progress & out) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_get_progress(out);
}

bool llama_memory_kvq_fit_wait(llama_memory_t mem) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_wait();
}

bool llama_memory_kvq_fit_set_hold(llama_memory_t mem, bool hold) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_set_hold(hold);
}

bool llama_memory_kvq_get_counters(llama_memory_t mem, llama_kvq_counters_all & out) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (!kvq || !kvq->get_store()) {
        return false;
    }
    kvq->get_store()->get_counters(out);
    return true;
}

void llama_memory_kvq_reset_counters(llama_memory_t mem) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (kvq && kvq->get_store()) {
        kvq->get_store()->reset_counters();
    }
}

//
// llama_memory_hybrid_kvq_context
//

// the reserve graph is not in two-tier mode, but sizes the scratch attention of the mode
static llama_kvq_params kvq_params_reserve(const llama_kvq_params & p) {
    llama_kvq_params res;
    res.scr_block = p.scr_block;
    return res;
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_status status) :
    llama_memory_hybrid_context(status), status(status) {}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem) :
    llama_memory_hybrid_context(mem),
    kv_attn(mem->get_mem_attn()),
    params(kvq_params_reserve(mem->get_params())),
    store(mem->fit_graph_kind() > 0 ? mem->get_store() : nullptr),
    status(llama_memory_hybrid_context::get_status()) {
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(
        llama_memory_hybrid_kvq * mem,
                slot_info_vec_t   sinfos_attn,
      std::vector<llama_ubatch>   ubatches) :
    llama_memory_hybrid_context(mem, sinfos_attn, ubatches),
    kv_attn(mem->get_mem_attn()),
    mem_kvq(mem),
    params(mem->get_params()),
    store(mem->get_params().enabled ? mem->get_store() : nullptr),
    status(llama_memory_hybrid_context::get_status()) {
}

bool llama_memory_hybrid_kvq_context::next() {
    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_kvq_context::apply() {
    // the previous ubatches are queued: the fit mode can start a job or move the coarse boundary now
    if (mem_kvq) {
        mem_kvq->fit_poll(get_ubatch().n_tokens);
    }

    bool res = llama_memory_hybrid_context::apply();

    // the scratch attention of a prompt ubatch, the resident and tail buffers of a packed one
    if (res && mem_kvq && store) {
        if (prompt_path(get_ubatch())) {
            res = !store->scratch_ok() || store->scr_alloc();
        } else {
            res = store->dec_alloc();
        }
    }

    return res;
}

bool llama_memory_hybrid_kvq_context::codes_alloc(const llama_ubatch & ubatch) const {
    if (!mem_kvq || !store || prompt_path(ubatch)) {
        return true;
    }
    return store->codes_alloc();
}

bool llama_memory_hybrid_kvq_context::prompt_path(const llama_ubatch & ubatch) const {
    if (!params.enabled || store == nullptr) {
        return false;
    }
    return !llama_kvq_use_packed(params, ubatch.n_tokens, store->n_coarse(), store->n_host());
}

llama_memory_status llama_memory_hybrid_kvq_context::get_status() const {
    return status;
}

void llama_memory_hybrid_kvq_context::set_input_kvq_ring(ggml_tensor * mask, ggml_tensor * early, const llama_ubatch * ubatch, int32_t n_coarse) const {
    if (mask == nullptr || mask->buffer == nullptr) {
        return;
    }
    GGML_ASSERT(kv_attn->get_n_stream() == 1 && ggml_backend_buffer_is_host(mask->buffer));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && mask->ne[0] > n_coarse && mask->ne[1] == ubatch->n_tokens);
    GGML_ASSERT(early == nullptr || (ggml_backend_buffer_is_host(early->buffer) && ggml_are_same_shape(early, mask) && ggml_is_contiguous(early)));

    const auto & cells = kv_attn->get_cells(0);

    const int64_t n_kv  = mask->ne[0] - n_coarse;
    const int64_t n_tok = mask->ne[1];
    GGML_ASSERT(n_kv <= (int64_t) cells.size());

    llama_pos p_min = ubatch->pos[0];
    for (uint32_t i = 1; i < ubatch->n_tokens; ++i) {
        p_min = std::min(p_min, ubatch->pos[i]);
    }

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);

    // the positions before the ubatch, and the cells at or after it
    std::vector<ggml_fp16_t> row(mask->ne[0], ninf);
    std::fill(row.begin(), row.begin() + n_coarse, zero);
    std::vector<std::pair<int64_t, llama_pos>> late;
    for (int64_t j = 0; j < n_kv; ++j) {
        if (cells.is_empty(j) || !cells.seq_has(j, 0)) {
            continue;
        }
        const llama_pos p = cells.pos_get(j);
        if (p < n_coarse) {
            continue;
        }
        if (p < p_min) {
            row[n_coarse + j] = zero;
        } else {
            late.push_back({ n_coarse + j, p });
        }
    }

    ggml_fp16_t * dst = (ggml_fp16_t *) mask->data;
    for (int64_t i = 0; i < n_tok; ++i) {
        ggml_fp16_t * d = dst + i*mask->ne[0];
        memcpy(d, row.data(), row.size()*sizeof(ggml_fp16_t));
        for (const auto & c : late) {
            if (c.second <= ubatch->pos[i]) {
                d[c.first] = zero;
            }
        }
        if (early) {
            memcpy((ggml_fp16_t *) early->data + i*mask->ne[0], row.data(), row.size()*sizeof(ggml_fp16_t));
        }
    }
}

void llama_memory_hybrid_kvq_context::set_input_kvq_scr(ggml_tensor * mask, const llama_ubatch * ubatch, int32_t n_host, int64_t r0) const {
    if (mask == nullptr || mask->buffer == nullptr) {
        return;
    }
    GGML_ASSERT(kv_attn->get_n_stream() == 1 && ggml_backend_buffer_is_host(mask->buffer));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && r0 >= 0 && r0 <= n_host && r0 + mask->ne[0] > n_host && mask->ne[1] == ubatch->n_tokens);

    const auto & cells = kv_attn->get_cells(0);

    // mask column c is scratch row r0 + c
    const int64_t n_cols = mask->ne[0];
    const int64_t n_rows = r0 + n_cols;
    const int64_t size   = cells.size();
    GGML_ASSERT(n_rows - n_host <= size);

    llama_pos p_min = ubatch->pos[0];
    for (uint32_t i = 1; i < ubatch->n_tokens; ++i) {
        p_min = std::min(p_min, ubatch->pos[i]);
    }

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);

    auto has = [&](int64_t r) {
        const int64_t j = r % size;
        return !cells.is_empty(j) && cells.seq_has(j, 0) && cells.pos_get(j) == r;
    };

    // the rows of the positions before the ubatch, then per token the positions of the ubatch up to its own
    std::vector<ggml_fp16_t> row(n_cols, ninf);
    std::fill(row.begin(), row.begin() + (n_host - r0), zero);
    for (int64_t r = n_host; r < std::min<int64_t>(p_min, n_rows); ++r) {
        if (has(r)) {
            row[r - r0] = zero;
        }
    }
    std::vector<int64_t> late;
    for (int64_t r = std::max<int64_t>(n_host, p_min); r < n_rows; ++r) {
        if (has(r)) {
            late.push_back(r);
        }
    }

    ggml_fp16_t * dst = (ggml_fp16_t *) mask->data;
    for (int64_t i = 0; i < mask->ne[1]; ++i) {
        ggml_fp16_t * d = dst + i*n_cols;
        memcpy(d, row.data(), n_cols*sizeof(ggml_fp16_t));
        for (const int64_t r : late) {
            if (r > ubatch->pos[i]) {
                break;
            }
            d[r - r0] = zero;
        }
    }
}
