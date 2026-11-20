#pragma once

#include "llama-ext.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_ubatch;
class llama_kv_cache;
class llama_io_write_i;
class llama_io_read_i;

// packed two-tier state of the attention layers of one sequence (see llama_kvq_params::packed)
//   device: packed coarse rows and tables of the coarse region, resident precise rows (rkv, ridx) of the layers in rotating buffers or one per layer, top-k bias
//   host:   precise store (the q8_0 K/V rows of the coarse region), mirrors of the resident rows, slot state and row bitmaps of the resident slots
// the tensors and host arrays are allocated for n_rows coarse rows, the coarse region is [0, n_coarse)
//
// resident slots of a (layer, kv head): [0, n_sink) hold the sinks, then cap evictable slots, then one unused slot (the trash slot of the scatter before the trash rows)
// where the resident rows live: each attention layer keeps its rows in a pinned host mirror [2][G][cap_tot][ROW]
// the layers share n_buf rotating device buffers rkv, buffer = layer index mod n_buf (llama_kvq_params::layer_bufs);
// with a buffer per layer (n_rot = 0) the device rows stay from one round to the next, the join uploads the mirror only after a reset
//   base:  the mirror of layer i into its buffer on a base stream, after the attention of the previous reader of the buffer (layer i - n_buf)
//          the worker queues it after the prefetch job and the join uploads of layer i - n_buf + 1 (see base_next), event ev_base
//   delta: the join uploads the fetched rows after ev_base and scatters them as before; the worker writes them into the mirror after the gather
// the host fills the slots in the scheduler copy callbacks of host trigger tensors, each read by a get_rows:
//   prefetch: top prefetch_n of the early query of layer il, built at the input of layer il - lead
//             the callback reads the selection back and returns, a worker thread plans the slots and gathers the rows
//             the request is the union of the selections of the positions; with more rows than cap it is cut by the max score
//             a slot is free if its row is not in the request; the missing rows go to the free slots of the oldest requests
//   join:    before the attention of layer il, waits for the worker and uploads the rows, their slot list and ridx
// a set_rows after the join trigger scatters the staging rows into rkv
// the callbacks run before the scheduler waits for the GPU (ggml_backend_sched_set_copy_callback_early)
// the readback and the uploads of the prefetch run on a copy stream (a second backend of the device), ordered by events:
//   main: ev_sel at the prefetch trigger -> copy: readback, ev_read (worker) ... join: uploads, ev_up -> main waits ev_up
//   base: worker after the join uploads of layer j -> base stream waits ev_sel of j, mirror upload of layer j + n_buf - 1, ev_base -> its join waits
// a memcpy between two graphs on the main stream costs a GPU stall of about 50 us (measured, Windows), events do not
// the trigger tensors are not written: the trigger node is an add whose result is not used
//
// host ring: the attention cache is a ring (llama_kv_cache::get_ring), its cells are pinned host memory (llama_kv_cache::move_bufs),
// the device has N_TBUF rotating tail buffers [K cells | V cells], buffer = layer index mod N_TBUF
//   tail upload: the occupied cells [n_host, end) of layer i from the host ring, on the base stream by the worker after the join uploads
//                of layer i - 3 (after ev_sel of it and the readback of layer i - N_TBUF); the main stream waits for it at the join of layer i - 1,
//                as the early query of layer i reads the cells
//   new rows:    the graph writes the K/V of the ubatch into the tail buffer, the join reads them back into the host ring (copy stream)
//   prefill:     the scratch path writes the K/V of the ubatch into the scratch rows of their positions and reads them back into the host
//                store, the scratch rows [0, p0) come from the host store; so the store holds every row after a large ubatch,
//                the host ring after a packed one (st_ok, rg_ok), and the other side is copied on the host at the switch
// scratch attention of the large ubatches (prefill): one device scratch holds the q8_0 K/V rows that flash attention reads,
// row r = position r: the rows before the ubatch uploaded from the host store per layer, the rows of the ubatch written by the graph
//   scratch up:   after the attention of the previous attention layer (or at the graph start), the copy stream waits for the main
//                 stream and uploads the rows [0, p0) of layer il, one copy per kv head
//   scratch join: before the attention of layer il, the main stream waits for the uploads
class llama_kvq_store {
public:
    static constexpr int64_t D   = 256;
    static constexpr size_t  ROW = 272; // q8_0 row of D values

    enum kind {
        KIND_PREFETCH = 0,
        KIND_JOIN     = 1,
        KIND_SCR_UP   = 2,
        KIND_SCR_JOIN = 3,
        KIND_SCR_RB   = 4,
        KIND_COUNT    = 5,
    };

    // kc, vc, tinv, taux: in their own device buffer, with a pinned host copy in fit mode (see codes_alloc)
    struct layer_tensors {
        ggml_tensor * kc    = nullptr; // i32 [nwk, n_rows, G]
        ggml_tensor * vc    = nullptr; // i32 [nwv + 1, n_rows, G]
        ggml_tensor * tinv  = nullptr; // i8 [D, D, 2*G, n_chunk]
        ggml_tensor * taux  = nullptr; // f32 [D, 3, 2*G, n_chunk]
        ggml_tensor * rkv   = nullptr; // q8_0 [D, cap_tot, G, 2], alias of the start of rkv_h (of the device buffer of the layer)
        ggml_tensor * rkv_h = nullptr; // f32 [ROW/4, 2*G*cap_tot + N_TRASH], row (kv*G + g)*cap_tot + slot, then the trash rows
        ggml_tensor * ridx  = nullptr; // i32 [cap_tot, G]
        ggml_tensor * stage = nullptr; // f32 [ROW/4, n_stage], rows to scatter (set by enable, one for all layers)
        ggml_tensor * slots = nullptr; // i32 [n_stage], their rows of rkv_h, the rest trash rows (trash_row)
        ggml_tensor * trig[KIND_COUNT] = {}; // host f32 [1], per kind
        ggml_tensor * qacc  = nullptr; // f32 [D, D, G], sum of q q^T over the query heads of each kv head (fit mode)
        ggml_tensor * gacc  = nullptr; // f32 [n_gate], sum of sigmoid(gate) over the tokens (fit mode with n_gate > 0)
    };

    // the tensors hold n_rows coarse rows, table sets of n_chunk_rows rows (fit mode)
    // the region starts empty and grows (set_n_coarse), the query statistics tensors are allocated
    // n_gate > 0: also the gate statistics tensors, n_gate = n_head*D
    // layer_bufs: device buffers of the resident rows (llama_kvq_params::layer_bufs)
    // shared_min: the smallest shared memory (the fitter works in it, llama_kvq_fitter::shared_min)
    llama_kvq_store(const llama_kv_cache & kv, int32_t n_rows, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
            int32_t resident_cap, int32_t n_sink, int32_t layer_bufs, int32_t n_gate = 0, size_t shared_min = 0);
    ~llama_kvq_store();

    // fit mode: the table (llama_kvq_pack_table) of chunk c, for (il, g, K or V)
    // the decode thread, between graphs; a coarse chunk must get its new codes before the next graph
    // async: a copy on the main stream, the memory must stay valid until the next synchronize (pinned memory for a real async copy)
    void set_chunk_table(int32_t il, int32_t g, bool is_v, int32_t c, const int8_t * inv, const float * aux, bool async = false);

    // fit mode: the coarse region becomes [0, n), n <= n_host; new rows must have codes and tables set
    // the resident rows at or after n are dropped; the decode thread, between graphs
    void set_n_coarse(int32_t n);

    // fit mode: the positions [0, n) have their q8_0 rows in the host store (or will have, by work queued on the main stream)
    // the scratch attention reads these rows from the host store, the later ones from the cache
    void set_n_host(int32_t n);

    // the main backend of the device (one of the backends of the context)
    void set_backend(const std::vector<ggml_backend_t> & backends);

    // waits for the main stream and the copy stream
    void synchronize();

    // fit mode, state save and load of the coarse region [0, n_coarse): host rows, codes and tables, the decode thread
    // host rows [0, n_host), codes [0, n_coarse), tables of the chunks of the coarse region
    // meta: shapes and types, state_read_meta throws on a mismatch and returns n_coarse and n_host of the state
    // state_read_data: the coarse region must be empty, it becomes [0, n), the host rows [0, n_host)
    void state_write_meta(llama_io_write_i & io) const;
    void state_write_data(llama_io_write_i & io);
    void state_read_meta (llama_io_read_i  & io, int32_t & n, int32_t & n_host) const;
    void state_read_data (llama_io_read_i  & io, int32_t n, int32_t n_host);

    // the main backend (set_backend), the device and the buffer type of the device tensors
    ggml_backend_t             backend() const { return backend_main; }
    ggml_backend_dev_t         device()  const { return dev; }
    ggml_backend_buffer_type_t dev_buft() const { return buft; }

    // host precise store of layer il: [G][n_rows][2*ROW], K row then V row
    uint8_t * host_rows(int32_t il) const;

    // pinned host memory of the device, or the CPU buffer type
    ggml_backend_buffer_type_t host_buft() const { return buft_host; }

    // checks the parameters against the allocation and resets the resident slots and the counters
    bool enable(const llama_kvq_params & params);

    // the resident slots start empty (the sinks stay), so the packed attention after this depends on the memory state only
    void reset_resident();

    // called once per ubatch of the packed path
    void begin_ubatch(const llama_ubatch & ubatch);

    // called once per ubatch of the scratch path
    void begin_scr(const llama_ubatch & ubatch);

    // the rows of the positions [n_host, n) in the host store, or in the host ring (the copy on the host at a switch, see the class comment)
    void sync_store(int32_t n);
    void sync_ring(int32_t n);

    // after a state read, the host ring holds the positions [n_host, n)
    void ring_loaded(int32_t n);

    // the main stream waits for the readbacks into the host store (before it uploads host rows)
    void order_rows();

    // K or V of the tail cells [0, n_kv) of layer il, like a view of the cache: [D, n_kv, G]
    ggml_tensor * get_ring(ggml_context * ctx, int32_t il, bool is_v, int64_t n_kv) const;

    // the K or V of the ubatch [D, G, n_tokens] into the tail buffer of layer il (cells idxs), like llama_kv_cache::cpy_k
    ggml_tensor * build_ring_cpy(ggml_context * ctx, int32_t il, bool is_v, ggml_tensor * cur, ggml_tensor * idxs) const;

    // the K or V of the ubatch [D, G, n_tokens] into the scratch rows of their positions pos (i32 [n_tokens])
    ggml_tensor * build_scr_set(ggml_context * ctx, bool is_v, ggml_tensor * cur, ggml_tensor * pos) const;

    // after the attention of the last attention layer of a scratch graph, the readback of its rows
    void build_scr_rb(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const;

    // ggml_backend_sched copy callback, false if src is not a trigger
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // the graph of the selection of the prefetch trigger: score [n_coarse, G, L] -> top n_top per (kv head, position) -> trigger
    void build_fetch(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, ggml_tensor * score, int64_t n_top) const;

    // the join of the prefetch of layer il for L positions: trigger -> scatter, before the attention
    void build_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, int64_t n_top, int64_t L) const;

    const layer_tensors & get_layer(int32_t il) const;

    // the scratch attention can be used (pinned host store)
    bool scratch_ok() const { return scr_ok; }

    // the device memory of the scratch (allocated at the ctor, never freed): the resident rows, the tail buffers and the staging block
    // of the packed path hold no state from one round to the next (own buffers: the mirrors hold it), so they share it; scr_alloc before
    // a scratch graph drops the queued prefetch, base and tail uploads (the next round queues them again) and zeroes the scratch, dec_alloc
    // before a packed graph zeroes the resident and tail buffers (the joins upload the mirrors of the own buffers)
    bool scr_alloc();
    bool dec_alloc();

    // the fitter works in the shared memory between the ubatches: drops the queued uploads of the packed path and waits for the
    // streams; the next scr_alloc zeroes the scratch again, the next dec_alloc restores the resident buffers
    void fit_alloc();
    ggml_backend_buffer_t shared_buffer() const { return buf_scr.get(); }

    // every write of the codes and tables goes to a pinned host copy first (same bytes as kc, vc, tinv, taux); the device copy exists
    // only for the packed path: scr_alloc frees it, codes_alloc (end of a prompt, or dec_alloc) allocates it and queues the upload
    // of the host copy on the main stream
    uint8_t * host_codes(int32_t il, bool is_v) const;
    void      codes_written(int32_t r1); // the fitter queued the codes of rows before r1 into the host copy
    bool      codes_alloc();
    void      codes_free();
    bool      codes_on_dev() const { return buf_codes != nullptr; }
    uint32_t  codes_gen()    const { return codes_gen_; } // changes when the device addresses of the codes change

    // the scratch triggers of layer il, see the class comment
    void build_scr_up  (ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const;
    void build_scr_join(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il) const;

    // K or V of scratch rows [r0, r0 + n): [D, G, n, 1], like a cache view
    ggml_tensor * get_scr(ggml_context * ctx, bool is_v, int64_t n, int64_t r0 = 0) const;

    // ridx without the flag column
    ggml_tensor * get_ridx(ggml_context * ctx, int32_t il) const;

    // kc and vc of the current coarse region
    ggml_tensor * get_kc(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_vc(ggml_context * ctx, int32_t il) const;

    int32_t n_coarse()     const { return n_coarse_; }
    int32_t n_host()       const { return n_host_; }
    int32_t n_chunks()     const { return n_chunk; }
    int32_t n_rows()       const { return n_rows_; }
    int32_t n_chunk_rows() const { return n_crow; }
    int32_t n_head_kv()    const { return G; }
    int32_t n_words(bool is_v) const { return is_v ? nwv : nwk; }
    int64_t n_ring()       const { return n_scr - n_rows_; } // cells of the ring cache

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
    int32_t n_host_   = 0;
    int32_t n_rows_   = 0; // allocated rows, the stride of the per-row host arrays
    int32_t n_crow    = 0;
    int32_t n_chunk   = 0;
    int32_t G         = 0;
    int32_t nwk       = 0;
    int32_t nwv       = 0;
    int32_t cap       = 0; // evictable slots
    int32_t n_sink    = 0;
    int32_t cap_tot   = 0;
    int64_t n_bmw     = 0; // 64-bit words of a row bitmap
    bool    cut_warn  = false;
    int32_t n_rot     = 0; // attention layers through the rotating buffers, 0: each layer has its own buffer
    int32_t n_buf     = 0; // rotating device buffers (0 with a buffer per layer)
    int32_t up_rep    = 1; // LLAMA_KVQ_UP_REPEAT (test knob): every base upload k times, emulates a slower bus

    llama_kvq_params params;

    ggml_backend_buffer_type_t buft      = nullptr;
    ggml_backend_buffer_type_t buft_host = nullptr;
    ggml_backend_dev_t         dev       = nullptr;

    std::vector<layer>                 layers;
    std::unordered_map<int32_t, size_t> layer_of_il;

    std::unordered_map<const ggml_tensor *, std::pair<size_t, int>> trig_of; // trigger -> layer index, kind

    ggml_context_ptr        ctx_dev;
    ggml_context_ptr        ctx_codes; // kc, vc, tinv, taux
    ggml_backend_buffer_ptr buf_codes;
    std::vector<ggml_backend_buffer_ptr> buf_hcodes; // host copy, one per layer
    int32_t                 hc_hi      = 0;     // the host copy holds the codes of the rows that a later upload needs before this
    uint32_t                codes_gen_ = 0;
    ggml_context_ptr        ctx_alias;
    ggml_context_ptr        ctx_host;
    ggml_backend_buffer_ptr buf_dev;
    ggml_backend_buffer_ptr buf_host; // triggers
    ggml_backend_buffer_ptr buf_pin;  // pinned host buffers (staging rows, ridx, slot list; readback per layer)
    ggml_backend_buffer_ptr buf_mir;  // pinned host mirrors of the resident rows
    ggml_context_ptr        ctx_stage;
    ggml_backend_buffer_ptr buf_stage; // device staging rows and slot list
    ggml_backend_ptr        backend_cp; // copy stream
    ggml_backend_ptr        backend_bu; // base upload stream
    ggml_backend_t          backend_main = nullptr;
    std::vector<ggml_backend_buffer_ptr> buf_store; // precise store, one per layer

    // the device buffers of the resident rows, n_buf rotating ones or one per layer (their ridx is the one in jblk)
    struct res_buf {
        ggml_tensor * rkv_h = nullptr;
        ggml_tensor * rkv   = nullptr;
    };
    std::vector<res_buf> bufs;

    // the rotating tail buffers, q8_0 [D*G, n_cell] K and V each, like the cache tensors
    static constexpr int32_t N_TBUF = 4;
    struct tail_buf {
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
    };
    std::vector<tail_buf> tbufs;
    int64_t n_cell = 0; // cells of the ring
    int32_t st_ok  = 0; // decode thread: the host store holds the rows of the positions [n_host, st_ok)
    int32_t rg_ok  = 0; // decode thread: the host ring holds the rows of the positions [n_host, rg_ok)
    int32_t ub_p0  = 0; // decode thread: positions [ub_p0, ub_p1) of the current ubatch (packed or scratch)
    int32_t ub_p1  = 0;
    std::atomic<int32_t> tb_lo{0}; // the worker uploads the cells of the positions [tb_lo, tb_hi)
    std::atomic<int32_t> tb_hi{0};
    ggml_backend_event_ptr ev_ord; // copy stream, for order_rows

    ggml_tensor * bias    = nullptr; // f32 [n_coarse], -inf for the sinks
    int64_t       n_stage = 0;       // rows of the staging tensors
    ggml_tensor * jblk    = nullptr; // i8, the device block [ridx | slot list | staging rows] of the rotating layers, one copy per join
    size_t      off_stage = 0;       // byte offset of the staging rows in jblk and in the pinned block at h_ridx
    size_t      off_jblk  = 0;       // shared memory: byte offset of jblk in the scratch buffer

    // pinned staging [h_ridx | h_slots | h_stage] with the layout of jblk, shared by the layers (see gather)
    uint8_t * h_stage  = nullptr; // [n_stage][ROW], K and V row of each fetch
    int32_t * h_ridx   = nullptr; // [G][cap_tot]
    int32_t * h_slots  = nullptr; // [n_stage]
    int64_t   slots_hi = 0;       // entries of the device slot list that are not trash

    ggml_tensor * scr    = nullptr; // q8_0 [D, 2, n_scr, G], K and V row of each scratch row (fit mode)
    int64_t       n_scr  = 0;
    bool          scr_ok = false;
    ggml_context_ptr        ctx_scr;
    ggml_backend_buffer_ptr buf_scr;

    int                     scr_mode = 0;     // shared memory: set up for 1 the scratch, 2 the packed path, 3 the fitter, 0 none
    ggml_context_ptr        ctx_dec;          // the resident and tail buffers in the shared memory


    // positions of the current ubatch, for the token count
    int64_t  ub_tokens = 0;
    int64_t  n_ubatch  = 0;
    int64_t  n_tokens  = 0;

    const int32_t zero = 0;

    // end of the last callback and its ubatch
    double  cb_end    = 0.0;
    int64_t cb_ubatch = -1;

    std::unique_ptr<gather_pool> pool;
    std::unique_ptr<worker>      work;

    // rows per kv head of one request
    int64_t n_req(int64_t n_top, int64_t L) const;

    // the entries of the slot list after the fetched rows write to trash rows after rkv in rkv_h, spread so that the writes do not wait on one row
    static constexpr int64_t N_TRASH = 1024;
    int32_t trash_row(int64_t j) const { return (int32_t) (2*G*cap_tot + j % N_TRASH); }

    // the prefetch of L positions needs the scores of the selection: the union can have more rows than the slots
    bool pf_scores(int64_t n_top, int64_t L) const { return n_top*L > cap; }

    void flush_ubatch();
    void select(const layer & l, const ggml_tensor * dst, ggml_cgraph * graph, const ggml_tensor ** ids, const ggml_tensor ** anchor) const;
    void cut_request(layer & l, int32_t g, int64_t n_u);
    void place(layer & l, int32_t g, uint32_t now);
    void plan_prefetch(layer & l);
    void gather(layer & l, int64_t n_rows);
    void upload(ggml_backend_t backend, layer & l);
    void run_prefetch(layer & l);
    void wait(layer & l);
    void drop(layer & l);
    void drop_pending();
    void upload_pinned(layer & l, int32_t n_pin);
    void write_mirror(layer & l);
    void issue_base(layer & l, ggml_backend_event_t ev, ggml_backend_t stream);
    void issue_tail(layer & l, ggml_backend_event_t ev, ggml_backend_t stream);
    void read_ring(layer & l);
    void read_scr(layer & l);
    void ring_copy(bool to_store, int32_t p0, int32_t p1);
    void wait_mirrors();
    void reset_base();
    void codes_up(int32_t n, bool async);
};
