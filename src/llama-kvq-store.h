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
//
// resident slots of a (layer, kv head): [0, n_sink) hold the sinks, then cap evictable slots, then one trash slot
// the host fills the slots in the scheduler copy callback of a host trigger tensor, read by a get_rows after the selection:
//   prefetch: top prefetch_n of the early query of layer il, built at the input of layer il - lead
//   top-up:   top sync_m of the true query of layer il, before ggml_kvq_attn_cond
// the callback reads the selection, gathers the missing rows into a staging tensor and its slot list,
// and a set_rows after the trigger scatters them into rkv
class llama_kvq_store {
public:
    static constexpr int64_t D   = 256;
    static constexpr size_t  ROW = 272; // q8_0 row of D values

    enum kind {
        KIND_PREFETCH = 0,
        KIND_TOPUP    = 1,
    };

    struct layer_tensors {
        ggml_tensor * kc    = nullptr; // i32 [nwk, n_coarse, G]
        ggml_tensor * vc    = nullptr; // i32 [nwv + 1, n_coarse, G]
        ggml_tensor * tinv  = nullptr; // f16 [D, D, 2*G, n_chunk]
        ggml_tensor * taux  = nullptr; // f32 [D, 3, 2*G, n_chunk]
        ggml_tensor * rkv   = nullptr; // q8_0 [D, cap_tot, G, 2]
        ggml_tensor * rkv_h = nullptr; // f16 [ROW/2, 2*G*cap_tot], alias of rkv, row (kv*G + g)*cap_tot + slot
        ggml_tensor * ridx  = nullptr; // i32 [cap_tot, G + 1], column G holds the flag of ggml_kvq_attn_cond
        ggml_tensor * trig[2] = {};    // host i32 [1]
    };

    llama_kvq_store(const llama_kv_cache & kv, int32_t n_coarse, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
            int32_t resident_cap, bool all_resident, int32_t n_sink);
    ~llama_kvq_store();

    bool set_codes(int32_t il, int32_t g, bool is_v, const std::vector<llama_kvq_table> & tabs, const int8_t * codes, const float * gain);

    // checks the parameters against the allocation and resets the resident slots and the counters
    bool enable(const llama_kvq_params & params);

    // called once per ubatch of the packed path
    void begin_ubatch(int64_t n_tokens);

    // ggml_backend_sched copy callback, false if src is not a trigger
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // the graph of the selection of one trigger: score [n_coarse, G, L] -> top n_top per (kv head, position) -> trigger -> scatter
    void build_fetch(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int32_t il, kind k, ggml_tensor * score, int64_t n_top) const;

    const layer_tensors & get_layer(int32_t il) const;

    // ridx without the flag column
    ggml_tensor * get_ridx(ggml_context * ctx, int32_t il) const;

    int32_t n_coarse()     const { return n_coarse_; }
    int32_t n_chunk_rows() const { return n_crow; }
    int32_t n_head_kv()    const { return G; }
    bool    all_resident() const { return all_res; }

    void get_counters(llama_kvq_counters_all & out);
    void reset_counters();

    llama_kvq_memory_info memory_info() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct layer;
    struct gather_pool;

    int32_t n_coarse_ = 0;
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
    ggml_backend_buffer_ptr buf_host;  // triggers
    ggml_backend_buffer_ptr buf_stage; // pinned staging rows
    std::vector<ggml_backend_buffer_ptr> buf_store; // precise store, one per layer

    ggml_tensor * bias    = nullptr; // f32 [n_coarse], -inf for the sinks
    ggml_tensor * staging = nullptr; // f16 [ROW/2, n_stage]
    ggml_tensor * slots   = nullptr; // i32 [n_stage]
    int64_t       n_stage = 0;       // rows of the staging tensor

    uint8_t *            stage_host = nullptr;
    std::vector<int32_t> slots_host;
    int64_t              slots_hi = 0; // entries of the device slot list that are not trash

    uint32_t stamp = 0; // selection clock

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

    void flush_ubatch();
};
