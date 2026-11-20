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

// llama_memory_hybrid plus the packed two-tier state (llama_kvq_store), created by llama_memory_kvq_fit_enable
// tail (fit mode): the attention cache is a ring of tail cells, the positions of the coarse region are removed from it

// packed mode: a ubatch of up to max_batch tokens attends through ggml_kvq_attn, if there are coarse rows
// the ring has no coarse rows: always packed, unless the host store has rows after the coarse region (they are not in the ring)
static inline bool llama_kvq_use_packed(const llama_kvq_params & p, uint32_t n_tokens, int32_t n_coarse, int32_t n_host) {
    return p.enabled && n_tokens <= (uint32_t) p.max_batch && n_coarse > 0 && n_host == n_coarse;
}

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
                     bool   rs_replay,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                 uint32_t   n_tail);

    ~llama_memory_hybrid_kvq() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    // the reserve of the packed decode graph: n_streams() sequences of max_batch tokens each, every coarse region at its largest
    // (0 if the packed path has one stream only, or no store)
    llama_memory_context_ptr init_full_packed();
    uint32_t                 n_packed_streams() const;

    // the packed rows cannot be shifted
    bool get_can_shift() const override;

    // fit mode: the fitted chunks are dropped, the mode stays on
    void clear(bool data) override;

    // packed: the coarse region cannot be removed
    // fit mode: the end of the sequence only, see llama_kvq_fit_params; the rows before p0 that leave the coarse region stay in the host store
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    // fit mode: seq_rm of the end per sequence (seq_id -1: every sequence), seq_keep empties the other sequences
    // seq_cp copies a whole sequence (complete chunk pages shared by reference, the last page copied; a range aborts), seq_add and seq_div abort in two-tier mode
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    // tail: the ring is empty when the sequence ends with the rows of the host store
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // fit mode: the state of one sequence (see state_write in the .cpp; seq_id -1: sequence 0, with several sequences an id is needed), the save waits for the jobs in flight
    // the partial state is the recurrent one; other two-tier modes: a state write or read of the K/V throws
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_kvq specific API
    //

    bool set_params(const llama_kvq_params & params);

    const llama_kvq_params & get_params() const { return params; }

    bool fit_enable(const llama_kvq_fit_params & fparams, const llama_kvq_params & params);

    // fit mode, the decode thread before each ubatch of n_tokens tokens of sequence seq (see llama_kvq_fitter::poll)
    void fit_poll(llama_seq_id seq, uint32_t n_tokens);
    // the end of a prompt of the sequence, see llama_memory_kvq_fit_wait
    bool fit_wait(llama_seq_id seq);
    bool fit_get_progress(llama_seq_id seq, llama_kvq_fit_progress & out) const;
    bool fit_set_hold(bool hold);

    // nullptr before fit_enable
    llama_kvq_store * get_store() const { return store.get(); }

    // the statistics nodes that the fit mode adds to every graph: 0 none, 1 queries, 2 queries and gates (for the reserve)
    int32_t fit_graph_kind() const;

    // the backends of the context
    void set_backends(const std::vector<ggml_backend_t> & backends);

    // ggml_backend_sched copy callback, see llama_kvq_store::copy
    bool sched_copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // fit mode: the scratch, for a reserve graph (the fitter and the packed path share its memory)
    bool prompt_bufs_alloc();

    // the context: switch to the buffers of the generation phase, called by fit_wait before the device codes are allocated
    void set_tg_cb(std::function<void()> cb) { tg_cb = std::move(cb); }

private:
    std::function<void()> tg_cb;

    const llama_model & model;

    const uint32_t n_ctx_max; // positions of the context

    // the state of one sequence (one per sequence of the memory, sized in fit_enable)
    struct seq_state {
        // tail: the positions [0, n_released) are not in the ring (their rows are in the host store)
        llama_pos n_released = 0;

        int32_t n_tail_wait = 0;
        double  t_tail_wait = 0.0;
        int32_t n_fold      = 0;
        double  t_fold      = 0.0;
        int32_t n_settle    = 0;
        double  t_settle    = 0.0;
    };
    std::vector<seq_state> seqs;

    int32_t fold_every  = 0; // LLAMA_KVQ_FOLD_EVERY (test knob): fold the reply rows when this many follow the coarse region

    // positions in the sequence: the ring, or the host store when the ring is empty
    llama_pos n_pos(llama_seq_id seq) const;

    // tail: removes the rows of the host store of the sequence from the ring
    void tail_release(llama_seq_id seq);

    // tail: host store rows [p_lo, p_hi) of the sequence back into the ring, the host store ends at p_lo
    bool tail_restore(llama_seq_id seq, llama_pos p_lo, llama_pos p_hi);

    // the layout of the positions [0, n) of the sequence (llama_kvq_fitter::settle), the rows after its coarse region in the ring
    // end: the end of a prompt, the decode phase starts during the fit (see tg_cb)
    bool settle(llama_seq_id seq, llama_pos n, bool end = false);

    // tail: before the batch is placed: the room in the ring for its positions (large batches: the jobs of the complete chunks, then
    // a spill of the ring rows to the host store; small batches: the jobs in flight are added first, a fold at a position set by p0)
    // sequence seq with its n_tokens tokens of the batch (a ubatch is one sequence, see init_batch)
    bool tail_wait(const llama_batch_allocr & balloc, llama_seq_id seq, uint32_t n_tokens);

    // the sequence holds nothing: its ring cells and recurrent cell, jobs, coarse region, host rows and pages
    void seq_clear(llama_seq_id seq);

    // throws if the K/V state cannot be saved or loaded
    void state_check(llama_seq_id seq_id, llama_state_seq_flags flags) const;

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

    // used to create the full-cache context of the packed decode reserve (two-tier mode is on, see init_full_packed)
    llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem, bool packed_reserve);

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

    const llama_kvq_params & get_params() const { return params; }

    // the packed decode reserve: the graph is built for the largest coarse region of every stream, nothing is executed
    bool packed_reserve() const { return is_packed_reserve; }

    // nullptr if not enabled
    llama_kvq_store * get_store() const { return store; }

    // tail: mask F16 [n_coarse + n_kv, n_tokens] over the coarse rows (kept) and the ring cells (causal by position)
    // early (if not null): the same without the positions of the ubatch and later
    // the ubatch is S sequences of L tokens (equal split): mask F16 [n_coarse_max + n_kv, L, 1, S], n_coarse_s the coarse rows of each
    void set_input_kvq_ring(ggml_tensor * mask, ggml_tensor * early, const llama_ubatch * ubatch, const std::vector<int32_t> & n_coarse_s) const;

    // tail: mask F16 [n_rows - r0, n_tokens] over the scratch rows [r0, n_rows), row r = position r: the rows of the host store [0, n_host) kept, the others causal if the ring has them
    void set_input_kvq_scr(ggml_tensor * mask, const llama_ubatch * ubatch, int32_t n_host, int64_t r0) const;

    // packed mode: the ubatch does not take the packed path (prompt ubatches; small ones after a rollback into the coarse region)
    bool prompt_path(const llama_ubatch & ubatch) const;

    // fit mode, a packed ubatch: the device codes (llama_kvq_store::codes_alloc), after the context switched its compute buffers
    bool codes_alloc(const llama_ubatch & ubatch) const;

private:
    const llama_kv_cache * kv_attn = nullptr;

    // batch processing context only
    llama_memory_hybrid_kvq * mem_kvq = nullptr;

    const llama_kvq_params params;

    bool is_packed_reserve = false;

    llama_kvq_store * store = nullptr;

    const llama_memory_status status;
};
