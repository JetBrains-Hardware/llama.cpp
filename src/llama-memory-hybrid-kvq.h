#pragma once

#include "llama-ext.h"
#include "llama-kvq-fit.h"
#include "llama-kvq-store.h"
#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_kvq
//

// llama_memory_hybrid plus a coarse copy of the attention K/V, for two-tier attention (see llama_kvq_params)
// the coarse cache shares the cells of the attention cache, so cell j is one token in both
// the attention cache holds the precise copy; in two-tier mode the graph writes the new tokens to both copies
// packed: no coarse cache, the packed state (llama_kvq_store) is created by llama_memory_kvq_packed_alloc or llama_memory_kvq_fit_enable
// tail (packed, fit mode): the attention cache is a ring of tail cells, the positions of the coarse region are removed from it

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
    const layer_filter_cb & filter_recr,
                     bool   packed,
                 uint32_t   n_tail = 0);

    ~llama_memory_hybrid_kvq() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    // a K-shift would rotate the precise copy only
    bool get_can_shift() const override;

    // fit mode: the fitted chunks are dropped, the mode stays on
    void clear(bool data) override;

    // packed: the coarse region cannot be removed
    // fit mode: the end of the sequence only, see llama_kvq_fit_params
    // tail: the positions of the removed coarse chunks before p0 go back to the ring from the host store
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    // two-tier mode: abort (seq_keep: unless it keeps sequence 0)
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // the coarse copy is not saved: in two-tier mode a state write or read aborts, unless it is of the partial state only (recurrent)
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_kvq specific API
    //

    // nullptr in packed mode
    llama_kv_cache * get_mem_coarse() const;

    bool set_params(const llama_kvq_params & params);

    const llama_kvq_params & get_params() const { return params; }

    bool is_packed() const { return packed; }

    bool packed_alloc(int32_t n_coarse, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v, int32_t resident_cap, bool all_resident, int32_t n_sink);

    bool fit_enable(const llama_kvq_fit_params & fparams, const llama_kvq_params & params);

    // fit mode, the decode thread before each ubatch (see llama_kvq_fitter::poll)
    void fit_poll();
    bool fit_wait();
    bool fit_get_progress(llama_kvq_fit_progress & out) const;
    bool fit_set_hold(bool hold);

    // nullptr before packed_alloc or fit_enable
    llama_kvq_store * get_store() const { return store.get(); }

    // the statistics nodes that the fit mode adds to every graph: 0 none, 1 queries, 2 queries and gates (for the reserve)
    int32_t fit_graph_kind() const;

    // the backends of the context
    void set_backends(const std::vector<ggml_backend_t> & backends);

    // ggml_backend_sched copy callback, see llama_kvq_store::copy
    bool sched_copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

private:
    const llama_model & model;

    const bool packed;

    const uint32_t n_ctx_max; // positions of the context
    const bool     tail;

    // tail: the positions [0, n_released) are not in the ring (the coarse region)
    llama_pos n_released = 0;

    int32_t n_tail_wait = 0;
    double  t_tail_wait = 0.0;

    // tail: removes the new coarse rows from the ring
    void tail_release();

    // tail: positions [n_coarse, min(p0, n_released)) back into the ring, after the coarse region shrank
    void tail_restore(llama_pos p0);

    // tail: before the batch is placed, waits for the fitter until its positions fit the ring
    bool tail_wait(const llama_batch_allocr & balloc);

    const std::unique_ptr<llama_kv_cache> mem_coarse;

    std::vector<ggml_backend_t> backends;

    // the fitter uses the store
    std::unique_ptr<llama_kvq_store>  store;
    std::unique_ptr<llama_kvq_fitter> fitter;

    llama_kvq_params params;
};

class llama_memory_hybrid_kvq_context : public llama_memory_hybrid_context {
public:
    // used for errors
    explicit llama_memory_hybrid_kvq_context(llama_memory_status status);

    // used to create a full-cache context, two-tier mode is off (fit mode: the store is set, for the statistics nodes of the reserve)
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

    // nullptr in packed mode
    const llama_kv_cache_context * get_coarse() const;

    const llama_kvq_params & get_params() const { return params; }

    // packed mode, nullptr if not enabled
    llama_kvq_store * get_store() const { return store; }

    // F32 [n_kv] each, unallocated tensors are skipped:
    //   pre:  1 for the prefix cells, 0 elsewhere
    //   prot: 1 for the protected prefix cells, 0 elsewhere
    //   bias: 0 for the other prefix cells (the candidates of the selection), -inf elsewhere
    void set_input_kvq(ggml_tensor * pre, ggml_tensor * prot, ggml_tensor * bias) const;

    // packed: early = mask (F16 [n_kv, n_tokens]) with -inf at the cells of the positions of the ubatch and later
    void set_input_kvq_early(ggml_tensor * early, const ggml_tensor * mask, const llama_ubatch * ubatch) const;

    // tail: mask F16 [n_coarse + n_kv, n_tokens] over the coarse rows (kept) and the ring cells (causal by position)
    // early (if not null): the same without the positions of the ubatch and later
    void set_input_kvq_ring(ggml_tensor * mask, ggml_tensor * early, const llama_ubatch * ubatch, int32_t n_coarse) const;

    // tail: mask F16 [n_rows, n_tokens] over the scratch rows, row r = position r: the coarse rows kept, the others causal if the ring has them
    void set_input_kvq_scr(ggml_tensor * mask, const llama_ubatch * ubatch, int32_t n_coarse) const;

private:
    const llama_kv_cache * kv_attn = nullptr;

    // batch processing context only
    llama_memory_hybrid_kvq * mem_kvq = nullptr;

    const llama_memory_context_ptr ctx_coarse;

    const llama_kvq_params params;

    llama_kvq_store * store = nullptr;

    const llama_memory_status status;
};
