#pragma once

#include "llama-ext.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_kvq_table;
class llama_kv_cache;

// packed two-tier state of the attention layers of one sequence (see llama_kvq_params::packed)
//   device: packed coarse rows and tables of the coarse region, resident precise rows (rkv, ridx) per layer, top-k bias
//   host:   precise store (the q8_0 K/V rows of the coarse region), LRU state of the resident slots
// the tensors and host arrays are allocated for n_rows coarse rows, the coarse region is [0, n_coarse)
//
// resident slots of a (layer, kv head): [0, n_sink) hold the sinks, then cap evictable slots, then one trash slot
// the host fills the slots in the scheduler copy callbacks of host trigger tensors, each read by a get_rows:
//   prefetch: top prefetch_n of the early query of layer il, built at the input of layer il - lead
//             the callback reads the selection back and returns, a worker thread plans the slots and gathers the rows
//   join:     before the attention of layer il, waits for the worker and uploads the rows, their slot list and ridx
//   top-up:   top sync_m of the true query of layer il, before ggml_kvq_attn_cond, plans, gathers and uploads in the callback
// a set_rows after the join and top-up triggers scatters the staging rows into rkv
// the callbacks run before the scheduler waits for the GPU (ggml_backend_sched_set_copy_callback_early)
// the readback and the uploads of the prefetch run on a copy stream (a second backend of the device), ordered by events:
//   main: ev_sel at the prefetch trigger -> copy: readback, ev_read (worker) ... join: uploads, ev_up -> main waits ev_up
// a memcpy between two graphs on the main stream costs a GPU stall of about 50 us (measured, Windows), events do not
// the trigger tensors are not written: the trigger node is an add whose result is not used
//
// fit mode, scratch attention of the large ubatches (prefill): one device scratch holds the q8_0 K/V rows that flash attention reads,
// the coarse region [0, n_coarse) uploaded from the host store per layer, the rest copied on the device from the cache
//   scratch up:   after the attention of the previous attention layer (or at the graph start), the copy stream waits for the main
//                 stream and uploads the rows of layer il, one copy per kv head
//   scratch join: before the attention of layer il, the main stream waits for the uploads
class llama_kvq_store {
public:
    static constexpr int64_t D   = 256;
    static constexpr size_t  ROW = 272; // q8_0 row of D values

    enum kind {
        KIND_PREFETCH = 0,
        KIND_TOPUP    = 1,
        KIND_JOIN     = 2,
        KIND_SCR_UP   = 3,
        KIND_SCR_JOIN = 4,
        KIND_COUNT    = 5,
    };

    struct layer_tensors {
        ggml_tensor * kc    = nullptr; // i32 [nwk, n_rows, G]
        ggml_tensor * vc    = nullptr; // i32 [nwv + 1, n_rows, G]
        ggml_tensor * tinv  = nullptr; // f16 [D, D, 2*G, n_chunk]
        ggml_tensor * taux  = nullptr; // f32 [D, 3, 2*G, n_chunk]
        ggml_tensor * rkv   = nullptr; // q8_0 [D, cap_tot, G, 2]
        ggml_tensor * rkv_h = nullptr; // f16 [ROW/2, 2*G*cap_tot], alias of rkv, row (kv*G + g)*cap_tot + slot
        ggml_tensor * ridx  = nullptr; // i32 [cap_tot, G + 1], column G holds the flag of ggml_kvq_attn_cond
        ggml_tensor * stage = nullptr; // f16 [ROW/2, n_stage], rows to scatter (set by enable)
        ggml_tensor * slots = nullptr; // i32 [n_stage], their rows of rkv_h, the rest is the trash slot
        ggml_tensor * trig[KIND_COUNT] = {}; // host f32 [1], per kind
        ggml_tensor * qacc  = nullptr; // f32 [D, D, G], sum of q q^T over the query heads of each kv head (fit mode)
        ggml_tensor * gacc  = nullptr; // f32 [n_gate], sum of sigmoid(gate) over the tokens (fit mode with n_gate > 0)
        ggml_tensor * k_h   = nullptr; // f16 [ROW/2, G, kv_size], alias of the K cache (fit mode)
        ggml_tensor * v_h   = nullptr; // f16 [ROW/2, G, kv_size], alias of the V cache (fit mode)
    };

    // the tensors hold n_rows coarse rows (a multiple of n_chunk_rows)
    // fill: the coarse region is [0, n_rows), its q8_0 rows are copied from the cache now (llama_memory_kvq_packed_alloc)
    // else: fit mode, the region starts empty and grows by chunks (set_n_coarse), the query statistics tensors are allocated
    // n_gate > 0 (fit mode): also the gate statistics tensors, n_gate = n_head*D
    llama_kvq_store(const llama_kv_cache & kv, int32_t n_rows, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
            int32_t resident_cap, bool all_resident, int32_t n_sink, bool fill, int32_t n_gate = 0);
    ~llama_kvq_store();

    bool set_codes(int32_t il, int32_t g, bool is_v, const std::vector<llama_kvq_table> & tabs, const int8_t * codes, const float * gain);

    // fit mode: the rows (llama_kvq_pack_rows) and the table (llama_kvq_pack_table) of chunk c, for (il, g, K or V)
    // only for chunks at or after the coarse region; the decode thread, between graphs
    void set_chunk_rows (int32_t il, int32_t g, bool is_v, int32_t c, const int32_t * rows);
    void set_chunk_table(int32_t il, int32_t g, bool is_v, int32_t c, const ggml_fp16_t * inv, const float * aux);

    // fit mode: the coarse region becomes [0, n), n a multiple of n_chunk_rows; new chunks must have rows, tables and host rows set
    // the resident rows at or after n are dropped; the decode thread, between graphs
    void set_n_coarse(int32_t n);

    // the main backend of the device (one of the backends of the context)
    void set_backend(const std::vector<ggml_backend_t> & backends);

    // waits for the main stream and the copy stream
    void synchronize();

    // host precise store of layer il: [G][n_rows][2*ROW], K row then V row
    uint8_t * host_rows(int32_t il) const;

    // the q8_0 K or V cache tensor of layer il
    ggml_tensor * get_cache(int32_t il, bool is_v) const;

    // pinned host memory of the device, or the CPU buffer type
    ggml_backend_buffer_type_t host_buft() const { return buft_host; }

    // checks the parameters against the allocation and resets the resident slots and the counters
    bool enable(const llama_kvq_params & params);

    // called once per ubatch of the packed path
    void begin_ubatch(int64_t n_tokens);

    // ggml_backend_sched copy callback, false if src is not a trigger
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // the graph of the selection of one trigger: score [n_coarse, G, L] -> top n_top per (kv head, position) -> trigger
    // top-up: then the scatter
    void build_fetch(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, kind k, ggml_tensor * score, int64_t n_top) const;

    // the join of the prefetch of layer il for L positions: trigger -> scatter, before the attention
    void build_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, int64_t n_top, int64_t L) const;

    const layer_tensors & get_layer(int32_t il) const;

    // fit mode: the scratch attention can be used (scratch allocated, pinned host store, LLAMA_KVQ_SCRATCH is not 0)
    bool scratch_ok() const { return scr_ok; }

    // the scratch triggers of layer il, see the class comment
    void build_scr_up  (ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const;
    void build_scr_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const;

    // K or V of scratch rows [0, n): [D, G, n, 1], like a cache view
    ggml_tensor * get_scr(ggml_context * ctx, bool is_v, int64_t n) const;

    // copies cache rows [c0, c0 + n) of layer il to scratch rows [r0, r0 + n), K or V
    ggml_tensor * build_scr_copy(ggml_context * ctx, int32_t il, bool is_v, int64_t c0, int64_t n, int64_t r0) const;

    // ridx without the flag column
    ggml_tensor * get_ridx(ggml_context * ctx, int32_t il) const;

    // kc and vc of the current coarse region
    ggml_tensor * get_kc(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_vc(ggml_context * ctx, int32_t il) const;

    int32_t n_coarse()     const { return n_coarse_; }
    int32_t n_rows()       const { return n_rows_; }
    int32_t n_chunk_rows() const { return n_crow; }
    int32_t n_head_kv()    const { return G; }
    int32_t n_words(bool is_v) const { return is_v ? nwv : nwk; }
    bool    all_resident() const { return all_res; }
    bool    fit_mode()     const { return !filled; }

    std::vector<int32_t> layer_ids() const;

    void get_counters(llama_kvq_counters_all & out);
    void reset_counters();

    llama_kvq_memory_info memory_info() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct layer;
    struct gather_pool;
    struct worker;

    int32_t n_coarse_ = 0;
    int32_t n_rows_   = 0; // allocated rows, the stride of the per-row host arrays
    bool    filled    = false;
    int32_t n_crow    = 0;
    int32_t n_chunk   = 0;
    int32_t G         = 0;
    int32_t nwk       = 0;
    int32_t nwv       = 0;
    int32_t cap       = 0; // evictable slots
    int32_t n_sink    = 0;
    int32_t cap_tot   = 0;
    bool    all_res   = false;

    llama_kvq_params params;

    ggml_backend_buffer_type_t buft      = nullptr;
    ggml_backend_buffer_type_t buft_host = nullptr;
    ggml_backend_dev_t         dev       = nullptr;

    std::vector<layer>                 layers;
    std::unordered_map<int32_t, size_t> layer_of_il;

    std::unordered_map<const ggml_tensor *, std::pair<size_t, int>> trig_of; // trigger -> layer index, kind

    ggml_context_ptr        ctx_dev;
    ggml_context_ptr        ctx_alias;
    ggml_context_ptr        ctx_host;
    ggml_backend_buffer_ptr buf_dev;
    ggml_backend_buffer_ptr buf_host; // triggers
    ggml_backend_buffer_ptr buf_pin;  // pinned host buffers of the layers (staging rows, readback, ridx, slot list)
    ggml_context_ptr        ctx_stage;
    ggml_backend_buffer_ptr buf_stage; // device staging rows and slot lists of the layers
    ggml_backend_ptr        backend_cp; // copy stream
    ggml_backend_t          backend_main = nullptr;
    std::vector<ggml_backend_buffer_ptr> buf_store; // precise store, one per layer

    ggml_tensor * bias    = nullptr; // f32 [n_coarse], -inf for the sinks
    int64_t       n_stage = 0;       // rows of the staging tensors

    ggml_tensor * scr    = nullptr; // q8_0 [D, 2, n_scr, G], K and V row of each scratch row (fit mode)
    ggml_tensor * scr_h  = nullptr; // f16 [ROW/2, 2, n_scr, G], alias of scr
    int64_t       n_scr  = 0;
    bool          scr_ok = false;


    // fired positions of the current ubatch, for the clean token count
    int64_t  ub_tokens = 0;
    uint64_t ub_fired  = 0;
    int64_t  n_ubatch  = 0;
    int64_t  n_tokens  = 0;
    int64_t  n_clean   = 0;

    const int32_t zero = 0;

    // end of the last callback and its ubatch
    double  cb_end    = 0.0;
    int64_t cb_ubatch = -1;

    std::unique_ptr<gather_pool> pool;
    std::unique_ptr<worker>      work;
    bool                         probe = false;

    // rows per kv head of one request
    int64_t n_req(int k, int64_t n_top, int64_t L) const;

    void flush_ubatch();
    void select(const layer & l, const ggml_tensor * dst, ggml_cgraph * graph, const ggml_tensor ** ids, const ggml_tensor ** vals) const;
    void plan_prefetch(layer & l);
    void plan_topup(layer & l);
    void gather(layer & l, int64_t n_rows);
    void upload(ggml_backend_t backend, layer & l);
    void run_prefetch(layer & l);
    void wait(layer & l);
    void drop(layer & l);
    void drop_pending();
    void upload_pinned(layer & l, int32_t n_pin);
};
