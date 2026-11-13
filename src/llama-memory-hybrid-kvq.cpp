#include "llama-memory-hybrid-kvq.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_memory_hybrid_kvq
//

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
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                     bool   packed) :
    llama_memory_hybrid(
        model,
        no_alloc,
        type_k, type_v, v_trans, kv_size, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    packed(packed),
    mem_coarse([&]() -> llama_kv_cache * {
        if (packed) {
            return nullptr;
        }
        LLAMA_LOG_INFO("%s: creating coarse KV cache, size = %u cells\n", "llama_memory_hybrid_kvq", kv_size);

        // mem_other: the cells are the ones of the attention cache, this cache only adds the tensors
        return new llama_kv_cache(
            model, no_alloc, model.hparams, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_swa, swa_type,
            get_mem_attn(),
            filter_attn == nullptr ? [&](int32_t il) { return !model.hparams.is_recr(il); } : filter_attn,
            nullptr, nullptr, "kvq_");
    }()) {
    GGML_ASSERT(packed || mem_coarse->get_layer_ids() == get_mem_attn()->get_layer_ids());
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    std::vector<llama_ubatch> ubatches;
    slot_info_vec_t heads_attn;

    if (!prepare_batch(balloc, n_ubatch, embd_all, ubatches, heads_attn)) {
        return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_memory_hybrid_kvq_context>(this, std::move(heads_attn), std::move(ubatches));
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_full() {
    return std::make_unique<llama_memory_hybrid_kvq_context>(this);
}

bool llama_memory_hybrid_kvq::get_can_shift() const {
    return false;
}

void llama_memory_hybrid_kvq::clear(bool data) {
    llama_memory_hybrid::clear(data);

    // the cells are already reset, this clears the data
    if (mem_coarse) {
        mem_coarse->clear(data);
    }

    if (fitter) {
        fitter->remove(0);
        return;
    }

    store.reset();
    params.enabled = false;
}

bool llama_memory_hybrid_kvq::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (fitter) {
        if (seq_id > 0) {
            GGML_ABORT("seq_rm of sequence %d: the fit mode supports sequence 0 only", seq_id);
        }
        const llama_pos pos_max = get_mem_attn()->seq_pos_max(0);
        const llama_pos q0      = std::max<llama_pos>(p0, 0);
        const llama_pos q1      = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
        if (q0 >= q1 || q0 > pos_max) {
            return llama_memory_hybrid::seq_rm(seq_id, p0, p1);
        }
        if (q1 <= pos_max) {
            GGML_ABORT("seq_rm of positions [%d, %d): the fit mode can remove the end of the sequence only (last position %d)", p0, p1, pos_max);
        }
        if (!llama_memory_hybrid::seq_rm(seq_id, p0, p1)) {
            return false;
        }
        fitter->remove(q0);
        return true;
    }

    if (store && (p1 < 0 || p1 > 0) && (p0 < 0 || p0 < store->n_coarse())) {
        GGML_ABORT("seq_rm of positions [%d, %d) would remove the packed coarse region [0, %d)", p0, p1, store->n_coarse());
    }

    return llama_memory_hybrid::seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_kvq::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (params.enabled || store) {
        GGML_ABORT("seq_cp is not supported in two-tier mode");
    }

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_kvq::seq_keep(llama_seq_id seq_id) {
    if ((params.enabled || store) && seq_id != 0) {
        GGML_ABORT("seq_keep of sequence %d: two-tier mode supports sequence 0 only", seq_id);
    }

    llama_memory_hybrid::seq_keep(seq_id);
}

void llama_memory_hybrid_kvq::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (params.enabled) {
        GGML_ABORT("seq_add is not supported in two-tier mode");
    }

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_kvq::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (params.enabled) {
        GGML_ABORT("seq_div is not supported in two-tier mode");
    }

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_kvq::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_coarse) {
        for (const auto & buft_size : mem_coarse->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    if (store) {
        for (const auto & buft_size : store->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

// the partial state is the recurrent one, the attention cache writes nothing for it
// a failed state write looks like an empty state to the caller, so the other cases abort
void llama_memory_hybrid_kvq::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((params.enabled || store) && !(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        GGML_ABORT("state write of the K/V is not supported in two-tier mode");
    }

    llama_memory_hybrid::state_write(io, seq_id, flags);
}

void llama_memory_hybrid_kvq::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((params.enabled || store) && !(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        GGML_ABORT("state read of the K/V is not supported in two-tier mode");
    }

    llama_memory_hybrid::state_read(io, seq_id, flags);
}

llama_kv_cache * llama_memory_hybrid_kvq::get_mem_coarse() const {
    return mem_coarse.get();
}

bool llama_memory_hybrid_kvq::set_params(const llama_kvq_params & p) {
    if (p.enabled) {
        // one stream: the cell flags come from the positions of a single sequence
        if (get_mem_attn()->get_n_stream() != 1) {
            LLAMA_LOG_ERROR("%s: two-tier mode needs a single KV stream\n", __func__);
            return false;
        }
        if (p.n_top < 0 || p.n_sink < 0 || p.n_recent < 0 || p.n_prefix < 0) {
            LLAMA_LOG_ERROR("%s: bad parameters\n", __func__);
            return false;
        }
        if (p.packed != packed) {
            LLAMA_LOG_ERROR("%s: the memory was created %s packed mode\n", __func__, packed ? "for" : "without");
            return false;
        }
        if (packed && (!store || !store->enable(p))) {
            LLAMA_LOG_ERROR("%s: packed mode needs llama_memory_kvq_packed_alloc and matching parameters\n", __func__);
            return false;
        }
    }

    params = p;

    LLAMA_LOG_INFO("%s: enabled = %d, n_top = %d, n_sink = %d, n_recent = %d, n_prefix = %d, stats = %d\n", __func__,
            params.enabled, params.n_top, params.n_sink, params.n_recent, params.n_prefix, params.stats);

    return true;
}

bool llama_memory_hybrid_kvq::packed_alloc(int32_t n_coarse, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v, int32_t resident_cap, bool all_resident, int32_t n_sink) {
    if (!packed) {
        LLAMA_LOG_ERROR("%s: the memory was created without packed mode\n", __func__);
        return false;
    }
    params.enabled = false;
    fitter.reset();
    store.reset();
    try {
        store = std::make_unique<llama_kvq_store>(*get_mem_attn(), n_coarse, n_chunk_rows, bits_k, bits_v, resident_cap, all_resident, n_sink, true);
        store->set_backend(backends);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }
    return true;
}

bool llama_memory_hybrid_kvq::fit_enable(const llama_kvq_fit_params & fp, const llama_kvq_params & p) {
    if (!packed || !p.enabled || !p.packed || p.all_resident) {
        LLAMA_LOG_ERROR("%s: the fit mode needs a memory created in packed mode and enabled packed parameters without all_resident\n", __func__);
        return false;
    }
    const llama_kv_cache * kv = get_mem_attn();
    if (kv->get_n_stream() != 1 || kv->seq_pos_max(0) >= 0) {
        LLAMA_LOG_ERROR("%s: the fit mode needs an empty memory with a single KV stream\n", __func__);
        return false;
    }
    if (fp.chunk <= p.n_sink || fp.n_threads < 1 || (int64_t) kv->get_size() < fp.chunk) {
        LLAMA_LOG_ERROR("%s: bad chunk (%d, cache of %u cells) or thread count\n", __func__, fp.chunk, kv->get_size());
        return false;
    }
    if (backends.empty()) {
        LLAMA_LOG_ERROR("%s: the memory has no backends\n", __func__);
        return false;
    }

    std::vector<llama_kvq_table> v_tabs;
    if (!llama_kvq_load(fp.v_tables, v_tabs)) {
        return false;
    }

    params.enabled = false;
    fitter.reset();
    store.reset();
    try {
        const int32_t n_rows = (int32_t) (kv->get_size()/fp.chunk)*fp.chunk;
        store = std::make_unique<llama_kvq_store>(*kv, n_rows, fp.chunk, fp.bits_k, fp.bits_v, p.resident_cap, false, p.n_sink, false);
        store->set_backend(backends);
        fitter = std::make_unique<llama_kvq_fitter>(*store, fp, v_tabs);
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
    return true;
}

void llama_memory_hybrid_kvq::fit_poll() {
    if (fitter) {
        fitter->poll(*get_mem_attn());
    }
}

bool llama_memory_hybrid_kvq::fit_wait() {
    if (!fitter) {
        return false;
    }
    fitter->wait(*get_mem_attn());
    return true;
}

bool llama_memory_hybrid_kvq::fit_get_progress(llama_kvq_fit_progress & out) const {
    if (!fitter) {
        return false;
    }
    fitter->get_progress(out);
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

bool llama_memory_kvq_set_params(llama_memory_t mem, const llama_kvq_params & params) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->set_params(params);
}

bool llama_memory_kvq_set_coarse_rows(llama_memory_t mem, int32_t il, bool is_v, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int64_t n_embd, const float * src) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && src && kvq->get_mem_coarse() && kvq->get_mem_coarse()->copy_rows(il, is_v, seq_id, p0, p1, n_embd, nullptr, src);
}

bool llama_memory_kvq_packed_alloc(llama_memory_t mem, int32_t n_coarse, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
        int32_t resident_cap, bool all_resident, int32_t n_sink) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->packed_alloc(n_coarse, n_chunk_rows, bits_k, bits_v, resident_cap, all_resident, n_sink);
}

bool llama_memory_kvq_packed_set(llama_memory_t mem, int32_t il, int32_t g, bool is_v, const std::vector<llama_kvq_table> & tabs,
        const int8_t * codes, const float * gain) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->get_store() && kvq->get_store()->set_codes(il, g, is_v, tabs, codes, gain);
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

bool llama_memory_kvq_get_memory_info(llama_memory_t mem, llama_kvq_memory_info & out) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (!kvq || !kvq->get_store()) {
        return false;
    }
    out = kvq->get_store()->memory_info();
    return true;
}

//
// llama_memory_hybrid_kvq_context
//

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_status status) :
    llama_memory_hybrid_context(status), status(status) {}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem) :
    llama_memory_hybrid_context(mem),
    kv_attn(mem->get_mem_attn()),
    ctx_coarse(mem->get_mem_coarse() ? mem->get_mem_coarse()->init_full() : nullptr),
    params(),
    status(llama_memory_status_combine(llama_memory_hybrid_context::get_status(), ctx_coarse ? ctx_coarse->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(
        llama_memory_hybrid_kvq * mem,
                slot_info_vec_t   sinfos_attn,
      std::vector<llama_ubatch>   ubatches) :
    llama_memory_hybrid_context(mem, sinfos_attn, ubatches),
    kv_attn(mem->get_mem_attn()),
    mem_kvq(mem),
    // same slots as the attention cache, the cells are shared
    ctx_coarse(mem->get_mem_coarse() ? new llama_kv_cache_context(mem->get_mem_coarse(), std::move(sinfos_attn), std::move(ubatches)) : nullptr),
    params(mem->get_params()),
    store(mem->get_params().enabled ? mem->get_store() : nullptr),
    status(llama_memory_status_combine(llama_memory_hybrid_context::get_status(), ctx_coarse ? ctx_coarse->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

bool llama_memory_hybrid_kvq_context::next() {
    if (ctx_coarse) {
        ctx_coarse->next();
    }

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_kvq_context::apply() {
    // the previous ubatches are queued: the fit mode can start a job or move the coarse boundary now
    if (mem_kvq) {
        mem_kvq->fit_poll();
    }

    // the attention cache updates the shared cells first, then the coarse context reads n_kv from them
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_coarse) {
        res = res & ctx_coarse->apply();
    }

    return res;
}

llama_memory_status llama_memory_hybrid_kvq_context::get_status() const {
    return status;
}

const llama_kv_cache_context * llama_memory_hybrid_kvq_context::get_coarse() const {
    return ctx_coarse ? static_cast<const llama_kv_cache_context *>(ctx_coarse.get()) : nullptr;
}

void llama_memory_hybrid_kvq_context::set_input_kvq(ggml_tensor * pre, ggml_tensor * prot, ggml_tensor * bias) const {
    GGML_ASSERT(params.enabled && kv_attn->get_n_stream() == 1);

    const auto & cells = kv_attn->get_cells(0);

    const llama_pos p_recent = params.n_prefix - params.n_recent;

    ggml_tensor * dst[3] = { pre, prot, bias };

    for (int i = 0; i < 3; ++i) {
        if (dst[i] == nullptr || dst[i]->buffer == nullptr) {
            continue;
        }

        GGML_ASSERT(ggml_backend_buffer_is_host(dst[i]->buffer));
        GGML_ASSERT(dst[i]->type == GGML_TYPE_F32 && dst[i]->ne[0] <= (int64_t) cells.size());

        float * data = (float *) dst[i]->data;

        for (int64_t j = 0; j < dst[i]->ne[0]; ++j) {
            const llama_pos p = cells.is_empty(j) ? -1 : cells.pos_get(j);

            const bool is_pre  = p >= 0 && p < params.n_prefix;
            const bool is_prot = is_pre && (p < params.n_sink || p >= p_recent);

            switch (i) {
                case 0: data[j] = is_pre;  break;
                case 1: data[j] = is_prot; break;
                case 2: data[j] = is_pre && !is_prot ? 0.0f : -INFINITY; break;
            }
        }
    }
}

void llama_memory_hybrid_kvq_context::set_input_kvq_early(ggml_tensor * early, const ggml_tensor * mask, const llama_ubatch * ubatch) const {
    if (early == nullptr || early->buffer == nullptr) {
        return;
    }
    GGML_ASSERT(kv_attn->get_n_stream() == 1);
    GGML_ASSERT(ggml_backend_buffer_is_host(early->buffer) && ggml_backend_buffer_is_host(mask->buffer));
    GGML_ASSERT(early->type == GGML_TYPE_F16 && ggml_are_same_shape(early, mask) && ggml_is_contiguous(early) && ggml_is_contiguous(mask));

    const auto & cells = kv_attn->get_cells(0);

    llama_pos p_min = ubatch->pos[0];
    for (uint32_t i = 1; i < ubatch->n_tokens; ++i) {
        p_min = std::min(p_min, ubatch->pos[i]);
    }

    const ggml_fp16_t   ninf = ggml_fp32_to_fp16(-INFINITY);
    const int64_t       n_kv = early->ne[0];
    const ggml_fp16_t * src  = (const ggml_fp16_t *) mask->data;
    ggml_fp16_t       * dst  = (ggml_fp16_t *) early->data;

    memcpy(dst, src, ggml_nbytes(early));
    for (int64_t j = 0; j < n_kv; ++j) {
        if (!cells.is_empty(j) && cells.pos_get(j) >= p_min) {
            for (int64_t i = 0; i < early->ne[1]; ++i) {
                dst[i*n_kv + j] = ninf;
            }
        }
    }
}
