#include "llama-memory-hybrid-kvq.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <cmath>
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
    const layer_filter_cb & filter_recr) :
    llama_memory_hybrid(
        model,
        no_alloc,
        type_k, type_v, v_trans, kv_size, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    mem_coarse([&] {
        LLAMA_LOG_INFO("%s: creating coarse KV cache, size = %u cells\n", "llama_memory_hybrid_kvq", kv_size);

        // mem_other: the cells are the ones of the attention cache, this cache only adds the tensors
        return new llama_kv_cache(
            model, no_alloc, model.hparams, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_swa, swa_type,
            get_mem_attn(),
            filter_attn == nullptr ? [&](int32_t il) { return !model.hparams.is_recr(il); } : filter_attn,
            nullptr, nullptr, "kvq_");
    }()) {
    GGML_ASSERT(mem_coarse->get_layer_ids() == get_mem_attn()->get_layer_ids());
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
    return !params.enabled && llama_memory_hybrid::get_can_shift();
}

void llama_memory_hybrid_kvq::clear(bool data) {
    llama_memory_hybrid::clear(data);

    // the cells are already reset, this clears the data
    mem_coarse->clear(data);

    params.enabled = false;
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

    for (const auto & buft_size : mem_coarse->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }

    return mb;
}

void llama_memory_hybrid_kvq::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (params.enabled) {
        throw std::runtime_error("state write is not supported in two-tier mode");
    }

    llama_memory_hybrid::state_write(io, seq_id, flags);
}

void llama_memory_hybrid_kvq::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (params.enabled) {
        throw std::runtime_error("state read is not supported in two-tier mode");
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
    }

    params = p;

    LLAMA_LOG_INFO("%s: enabled = %d, n_top = %d, n_sink = %d, n_recent = %d, n_prefix = %d, stats = %d\n", __func__,
            params.enabled, params.n_top, params.n_sink, params.n_recent, params.n_prefix, params.stats);

    return true;
}

bool llama_memory_kvq_set_params(llama_memory_t mem, const llama_kvq_params & params) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->set_params(params);
}

bool llama_memory_kvq_set_coarse_rows(llama_memory_t mem, int32_t il, bool is_v, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int64_t n_embd, const float * src) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && src && kvq->get_mem_coarse()->copy_rows(il, is_v, seq_id, p0, p1, n_embd, nullptr, src);
}

//
// llama_memory_hybrid_kvq_context
//

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_status status) :
    llama_memory_hybrid_context(status), status(status) {}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem) :
    llama_memory_hybrid_context(mem),
    kv_attn(mem->get_mem_attn()),
    ctx_coarse(mem->get_mem_coarse()->init_full()),
    params(),
    status(llama_memory_status_combine(llama_memory_hybrid_context::get_status(), ctx_coarse->get_status())) {
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(
        llama_memory_hybrid_kvq * mem,
                slot_info_vec_t   sinfos_attn,
      std::vector<llama_ubatch>   ubatches) :
    llama_memory_hybrid_context(mem, sinfos_attn, ubatches),
    kv_attn(mem->get_mem_attn()),
    // same slots as the attention cache, the cells are shared
    ctx_coarse(new llama_kv_cache_context(mem->get_mem_coarse(), std::move(sinfos_attn), std::move(ubatches))),
    params(mem->get_params()),
    status(llama_memory_status_combine(llama_memory_hybrid_context::get_status(), ctx_coarse->get_status())) {
}

bool llama_memory_hybrid_kvq_context::next() {
    ctx_coarse->next();

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_kvq_context::apply() {
    // the attention cache updates the shared cells first, then the coarse context reads n_kv from them
    bool res = llama_memory_hybrid_context::apply();

    res = res & ctx_coarse->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_kvq_context::get_status() const {
    return status;
}

const llama_kv_cache_context * llama_memory_hybrid_kvq_context::get_coarse() const {
    return static_cast<const llama_kv_cache_context *>(ctx_coarse.get());
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
