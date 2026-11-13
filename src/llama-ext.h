#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// layers of the attention KV cache of a plain or hybrid memory, empty if there is none
LLAMA_API std::vector<int32_t> llama_memory_kv_layer_ids(llama_memory_t mem);

// copy the K (is_v = false) or V rows of layer il for positions [p0, p1) of seq_id, as F32 (p1 - p0) x n_embd
// the rows are found through the cells of the sequence; F32/F16/BF16/Q8_0 caches only, V must not be transposed (flash attention)
// fails if n_embd is not the row size of the cache (n_embd_k_gqa or n_embd_v_gqa)
LLAMA_API bool llama_memory_kv_get_rows(llama_memory_t mem, int32_t il, bool is_v, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int64_t n_embd, float * dst);
LLAMA_API bool llama_memory_kv_set_rows(llama_memory_t mem, int32_t il, bool is_v, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int64_t n_embd, const float * src);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

//
// two-tier attention (KV cache compression experiments)
//

// the contexts created from this model keep a coarse copy of the attention K/V next to the precise one (qwen35 only)
LLAMA_API void llama_model_set_kvq_two_tier(struct llama_model * model, bool value);

// cells with pos < n_prefix are the prefix, later cells hold exact K/V in both copies and are always precise
// the protected cells (the first n_sink and the last n_recent of the prefix) are precise
// of the other prefix cells, the n_top with the largest coarse attention are precise, per kv head and query position
//
// packed mode (llama_model_set_kvq_packed, ggml_kvq_attn): the coarse region is set by llama_memory_kvq_packed_alloc (or grows, llama_memory_kvq_fit_enable),
// the precise rows of the coarse region come from a host store into a resident cache per (layer, kv head)
// ubatches of up to max_batch tokens attend through ggml_kvq_attn, larger ones through the normal attention
// the first n_sink rows are always resident; n_top, n_recent and n_prefix are not used
enum llama_kvq_sync {
    LLAMA_KVQ_SYNC_OFF    = 0, // attend with the resident rows as they are
    LLAMA_KVQ_SYNC_COUNT  = 1, // fetch the missing rows of the true top sync_m if fewer than sync_min_hits are resident
    LLAMA_KVQ_SYNC_ALWAYS = 2, // fetch every missing row of the true top sync_m
};

struct llama_kvq_params {
    bool      enabled  = false;
    int32_t   n_top    = 0;
    int32_t   n_sink   = 4;
    int32_t   n_recent = 512;
    llama_pos n_prefix = 0;
    bool      stats    = false; // per layer statistics in the graph, as tensor "kvq_stats-<il>"; packed: top-m counters with sync off

    bool    packed        = false;
    int32_t max_batch     = 8;
    int32_t prefetch_n    = 1024; // rows per (kv head, position) requested by the early query
    int32_t resident_cap  = 1024; // evictable resident slots per (layer, kv head), the sinks have their own, least recently selected eviction; more slots cost attention time
    int32_t lead          = 2;    // the early query of layer il runs at the input of layer il - lead, 0: no prefetch
    int32_t sync_mode     = LLAMA_KVQ_SYNC_OFF;
    int32_t sync_m        = 64;
    int32_t sync_min_hits = 48;
    bool    all_resident  = false; // debug: every coarse row resident, no prefetch and no top-up
};

// false if the memory has no coarse copy or does not support the parameters (single stream only)
// the coarse copy is written only in two-tier mode: the caller sets the coarse rows of the cells that exist before the switch
LLAMA_API bool llama_memory_kvq_set_params(llama_memory_t mem, const llama_kvq_params & params);

// like llama_memory_kv_set_rows, for the coarse copy
LLAMA_API bool llama_memory_kvq_set_coarse_rows(llama_memory_t mem, int32_t il, bool is_v, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int64_t n_embd, const float * src);

struct llama_kvq_table; // src/llama-kvq-codec.h

// packed mode: the contexts created from this model attend through ggml_kvq_attn, no coarse copy of the cache (qwen35 only)
LLAMA_API void llama_model_set_kvq_packed(struct llama_model * model, bool value);

// packed mode: the coarse region is the cells [0, n_coarse) of sequence 0, which must hold positions [0, n_coarse)
// allocates the device tensors for codes of bits_k/bits_v bits per value and n_coarse/n_chunk_rows table sets,
// and copies the q8_0 K/V rows of the region from the cache to the host store
// resident_cap and all_resident size the resident cache, the other parameters are read by llama_memory_kvq_set_params
LLAMA_API bool llama_memory_kvq_packed_alloc(llama_memory_t mem, int32_t n_coarse, int32_t n_chunk_rows, int32_t bits_k, int32_t bits_v,
        int32_t resident_cap, bool all_resident, int32_t n_sink);

// packed mode: the codec tables (one per chunk) and the codes [n_coarse][n] (llama_kvq_encode) of layer il, kv head g, K or V
LLAMA_API bool llama_memory_kvq_packed_set(llama_memory_t mem, int32_t il, int32_t g, bool is_v, const std::vector<llama_kvq_table> & tabs,
        const int8_t * codes, const float * gain);

// packed mode counters, per attention layer; times in seconds
struct llama_kvq_counters {
    int32_t il = -1;
    int64_t n_pf       = 0; // prefetch calls
    int64_t pf_req     = 0; // rows requested (union over the positions, per kv head)
    int64_t pf_fetch   = 0; // rows fetched (not resident)
    int64_t n_tu       = 0; // top-up calls
    int64_t n_pos      = 0; // (layer, position) executions with a top-up check
    int64_t tu_check   = 0; // (kv head, position) checks
    int64_t tu_fire    = 0; // (kv head, position) firings
    int64_t tu_stall   = 0; // (layer, position) with at least one firing kv head
    int64_t tu_redo    = 0; // calls that ran the second attention
    int64_t tu_fetch   = 0; // rows fetched by the top-up
    int64_t top_hit    = 0; // rows of the true top sync_m resident before the top-up
    int64_t top_n      = 0;
    int64_t evict      = 0;
    int64_t n_join     = 0; // prefetch joins
    int64_t bytes      = 0; // host -> device, rows and slot lists
    double  t_wait     = 0.0; // since the previous callback of the ubatch: launches (and GPU work with LLAMA_KVQ_SCHED_SYNC)
    double  t_read     = 0.0; // readback of the selection, top-up: with the wait for the GPU
    double  t_plan     = 0.0; // slots and eviction, prefetch: worker thread
    double  t_gather   = 0.0; // rows to the staging buffer, prefetch: worker thread
    double  t_join     = 0.0; // decode thread blocked on the worker
    double  t_idle     = 0.0; // LLAMA_KVQ_IDLE_PROBE: GPU at the join before the decode thread queued the upload (estimate)
    double  t_upload   = 0.0;
};

struct llama_kvq_counters_all {
    std::vector<llama_kvq_counters> layers;
    int64_t n_ubatch   = 0; // ubatches through the packed path
    int64_t n_tokens   = 0;
    int64_t n_clean    = 0; // positions with no top-up firing in any layer
};

LLAMA_API bool llama_memory_kvq_get_counters(llama_memory_t mem, llama_kvq_counters_all & out);
LLAMA_API void llama_memory_kvq_reset_counters(llama_memory_t mem);

// device and host memory of the packed state, bytes
struct llama_kvq_memory_info {
    size_t codes      = 0; // packed coarse rows
    size_t tables     = 0;
    size_t resident   = 0; // rkv and ridx
    size_t other_dev  = 0; // bias, staging and slot lists
    size_t host_store = 0;
    size_t host_other = 0;
    bool   pinned     = false;
};

LLAMA_API bool llama_memory_kvq_get_memory_info(llama_memory_t mem, llama_kvq_memory_info & out);

// packed mode, the library fits the codec tables (single sequence 0, positions from 0 without gaps):
//   the graph sums q q^T of the queries (after q_norm and RoPE) per (attention layer, kv head) on the device;
//   a ubatch counts for the chunk of its first position
//   when the positions [c*chunk, (c+1)*chunk) are in the cache, the decode thread reads the sums and starts a job on CPU threads:
//   K table fitted on the chunk's rows with the metric of the chunk's queries, K and V rows coded (V: one offline table set),
//   packed, and the q8_0 rows copied to the host store
//   a finished chunk joins the coarse region on the decode thread before the next ubatch (the chunks in order)
//   the positions after the coarse region attend precisely (the q8_0 cache)
// memory operations: clear and seq_rm of the end of the sequence; a seq_rm into the coarse region or into a chunk that
// is fitting drops these chunks, they are fitted again; the other ones abort
struct llama_kvq_fit_params {
    int32_t     chunk     = 8192;
    int32_t     bits_k    = 2;
    int32_t     bits_v    = 1;
    std::string v_tables;      // llama_kvq_save file with a V table for each (attention layer, kv head), used for every chunk
    int32_t     n_threads = 8; // CPU threads of the jobs, the chunks are fitted in order
};

// the memory must be empty and created in packed mode; params as for llama_memory_kvq_set_params (packed = true)
// the coarse region can grow to the full chunks of the cache; enabled until the memory is freed, clear() keeps it
LLAMA_API bool llama_memory_kvq_fit_enable(llama_memory_t mem, const llama_kvq_fit_params & fparams, const llama_kvq_params & params);

struct llama_kvq_fit_progress {
    int32_t n_coarse  = 0; // rows of the coarse region
    int32_t n_chunk   = 0; // chunks in the coarse region
    int32_t n_running = 0; // jobs queued or running
    int32_t n_ready   = 0; // jobs done, not yet in the coarse region
    int32_t n_cancel  = 0; // jobs cancelled by seq_rm or clear
    int32_t n_drop    = 0; // chunks removed from the coarse region by seq_rm or clear
    bool    failed    = false; // a job failed, no more chunks join the coarse region
    std::vector<double> t_job;  // seconds per job that joined the coarse region: from the start of the job to its end
    std::vector<double> t_wall; // seconds per job that joined: from the readback of the statistics to its end (with the queue)
    double  t_cpu     = 0.0;    // CPU seconds of the tasks of these jobs
    double  t_take    = 0.0;    // decode thread seconds of the readbacks of the statistics (with the wait for the queued ubatches)
    double  t_install = 0.0;    // decode thread seconds of the installs
};

LLAMA_API bool llama_memory_kvq_fit_get_progress(llama_memory_t mem, llama_kvq_fit_progress & out);

// starts the job of a complete chunk whose statistics were not read yet, waits for all jobs and adds their chunks to the coarse region
// call it between decodes, from the decode thread
LLAMA_API bool llama_memory_kvq_fit_wait(llama_memory_t mem);

// debug: hold = true: finished chunks join the coarse region only in llama_memory_kvq_fit_wait
LLAMA_API bool llama_memory_kvq_fit_set_hold(llama_memory_t mem, bool hold);

// scheduler splits of the last graph
LLAMA_API int32_t llama_context_n_splits(const struct llama_context * ctx);

// replace the eval callback, used from the next graph build on
LLAMA_API void llama_context_set_eval_callback(struct llama_context * ctx, ggml_backend_sched_eval_callback cb, void * user_data);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// model/context data extraction
//

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);
