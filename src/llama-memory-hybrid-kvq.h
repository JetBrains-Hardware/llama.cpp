#pragma once

#include "llama-ext.h"
#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_kvq
//

// llama_memory_hybrid plus a coarse copy of the attention K/V, for two-tier attention (see llama_kvq_params)
// the coarse cache shares the cells of the attention cache, so cell j is one token in both
// the attention cache holds the precise copy; in two-tier mode the graph writes the new tokens to both copies

class llama_memory_hybrid_kvq : public llama_memory_hybrid {
public:
    llama_memory_hybrid_kvq(
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
    const layer_filter_cb & filter_recr);

    ~llama_memory_hybrid_kvq() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    // a K-shift would rotate the precise copy only
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // the coarse copy is not saved, so a state write or read in two-tier mode throws
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_kvq specific API
    //

    llama_kv_cache * get_mem_coarse() const;

    bool set_params(const llama_kvq_params & params);

    const llama_kvq_params & get_params() const { return params; }

private:
    const std::unique_ptr<llama_kv_cache> mem_coarse;

    llama_kvq_params params;
};

class llama_memory_hybrid_kvq_context : public llama_memory_hybrid_context {
public:
    // used for errors
    explicit llama_memory_hybrid_kvq_context(llama_memory_status status);

    // used to create a full-cache context, two-tier mode is off
    explicit llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_kvq_context(
            llama_memory_hybrid_kvq * mem,
                    slot_info_vec_t   sinfos_attn,
          std::vector<llama_ubatch>   ubatches);

    bool next()  override;
    bool apply() override;

    llama_memory_status get_status() const override;

    //
    // llama_memory_hybrid_kvq_context specific API
    //

    const llama_kv_cache_context * get_coarse() const;

    const llama_kvq_params & get_params() const { return params; }

    // F32 [n_kv] each, unallocated tensors are skipped:
    //   pre:  1 for the prefix cells, 0 elsewhere
    //   prot: 1 for the protected prefix cells, 0 elsewhere
    //   bias: 0 for the other prefix cells (the candidates of the selection), -inf elsewhere
    void set_input_kvq(ggml_tensor * pre, ggml_tensor * prot, ggml_tensor * bias) const;

private:
    const llama_kv_cache * kv_attn = nullptr;

    const llama_memory_context_ptr ctx_coarse;

    const llama_kvq_params params;

    const llama_memory_status status;
};
