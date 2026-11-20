#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"

#include <algorithm>
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

LLAMA_API std::vector<ggml_tensor *> llama_memory_get_tensors(const llama_memory_t mem);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes. (The two-tier KV cache does not use it: llama_init_from_model_kvq turns the rotation off in the
// attention cache of its own context only.)
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

//
// two-tier KV cache (kvq)
//

// packed mode (llama_init_from_model_kvq, ggml_kvq_attn): the coarse region grows with the fit (llama_memory_kvq_fit_enable),
// the precise rows of the coarse region come from a host store into a resident cache per (layer, kv head)
// ubatches of up to max_batch tokens attend through ggml_kvq_attn, larger ones through the normal attention
// the first n_sink rows are always resident; the other resident rows are the ones the prefetch fetched
struct llama_kvq_params {
    bool    enabled = false;
    int32_t n_sink  = 4;

    int32_t max_batch     = 8;
    int32_t prefetch_n    = 1024; // rows per (kv head, position) requested by the early query
    int32_t resident_cap  = 0;    // evictable resident slots per (layer, kv head), the sinks have their own; 0: max_batch*prefetch_n (llama_kvq_resident_slots); more slots cost attention time
    int32_t lead          = 2;    // the early query of layer il runs at the input of layer il - lead, 0: no prefetch (packed mode needs 1 - 3)
    int32_t layer_bufs    = -1;   // device buffers of the resident rows, the attention layers rotate through them (each round uploads every layer's
                                  // resident set); < 0 or at least the attention layer count: one buffer per layer, its rows stay on the device
    int32_t scr_block     = 16384; // fit mode, tail: the scratch attention reads the host rows in key blocks of this many rows (rounded down to 256) without a mask,
                                   // the rest with a mask, merged by the log-sum-exp; 0: one attention with a mask over all rows
};

// the evictable resident slots of the parameters; the default holds the whole prefetch request of every packed ubatch,
// with fewer slots a request of more distinct rows is cut by the max score over the positions (6 x prefetch_n is as good
// at 8 per batch and saves 71 MB of pinned host memory, resident cap sweep 2026-11-18; 8192 kept: 0.4 ms faster at n_max 7)
static inline int32_t llama_kvq_resident_slots(const llama_kvq_params & p) {
    return p.resident_cap > 0 ? p.resident_cap : p.max_batch*p.prefetch_n;
}

// whether the two-tier KV cache can serve the model: arch qwen35 and the shape its code assumes (K and V head size 256 in every
// attention layer, the same number of query and kv heads in all of them, a whole number of query heads per kv head);
// reason (optional) says what fails; the kv type (q8_0) is a context setting and is checked when the memory is created
LLAMA_API bool llama_model_supports_kvq(const struct llama_model * model, std::string * reason = nullptr);

// packed mode counters, per attention layer; times in seconds
struct llama_kvq_counters {
    int32_t il = -1;
    int64_t n_pf       = 0; // prefetch calls
    int64_t pf_req     = 0; // rows requested (union over the positions, per kv head)
    int64_t pf_fetch   = 0; // rows fetched (not resident)
    int64_t pf_sel     = 0; // rows of the selections of the positions, (kv head, position) lists x prefetch_n
    int64_t pf_cut     = 0; // (prefetch call, kv head) requests cut to the resident slots
    int64_t pf_lost    = 0; // rows of the selections of the positions that are not resident after a cut
    int64_t evict      = 0;
    int64_t n_join     = 0; // prefetch joins
    int64_t bytes      = 0; // host -> device, rows and slot lists
    int64_t base_bytes = 0; // host -> device, base uploads of the resident rows from the mirror (rotating layers)
    int64_t ring_bytes = 0; // host -> device, tail uploads of the ring cells (host ring)
    int64_t rb_bytes   = 0; // device -> host, the new rows into the host ring or the host store (host ring)
    double  t_wait     = 0.0; // since the previous callback of the ubatch: launches
    double  t_read     = 0.0; // readback of the selection
    double  t_plan     = 0.0; // slots and eviction, worker thread
    double  t_gather   = 0.0; // rows to the staging buffer, worker thread
    double  t_mirror   = 0.0; // fetched rows into the mirror, worker thread after the gather
    double  t_join     = 0.0; // decode thread blocked on the worker
    double  t_upload   = 0.0;
    int64_t n_scr      = 0;   // scratch attention: uploads of the coarse region (fit mode, large ubatches)
    int64_t scr_bytes  = 0;
    double  t_scr      = 0.0; // decode thread time in the scratch callbacks
};

struct llama_kvq_counters_all {
    std::vector<llama_kvq_counters> layers;
    int64_t n_ubatch   = 0; // ubatches through the packed path
    int64_t n_tokens   = 0;
};

LLAMA_API bool llama_memory_kvq_get_counters(llama_memory_t mem, llama_kvq_counters_all & out);
LLAMA_API void llama_memory_kvq_reset_counters(llama_memory_t mem);

// device and host memory of the packed state, bytes
struct llama_kvq_memory_info {
    size_t codes      = 0; // packed coarse rows
    size_t tables     = 0;
    size_t host_codes = 0;     // fit mode: the pinned host copy of the codes and tables
    size_t resident   = 0; // rkv and ridx
    size_t ring       = 0; // host ring: the rotating tail buffers (device), the ring cells are pinned host memory of the attention cache
    size_t host_ring  = 0;
    size_t other_dev  = 0; // bias, staging and slot lists
    size_t scratch    = 0; // fit mode: K/V rows of the large ubatches
    size_t host_store = 0;
    size_t host_other = 0;
    bool   pinned     = false;
};

// packed mode, the library fits the codec tables (single sequence 0, positions from 0 without gaps):
//   the graph sums q q^T of the queries (after q_norm and RoPE) per (attention layer, kv head) on the device;
//   a ubatch counts for the chunk of its first position
//   when the positions [c*chunk, (c+1)*chunk) are in the cache, the decode thread reads the sums; at the next large ubatch
//   (prefill) it starts a job, see llama_kvq_fitter: the work on the rows runs on the device between the ubatches, the fit of
//   the tables from the moments of the rows on a few CPU threads:
//   K table fitted on the rows with the metric of their queries, V table fitted on the rows with the o_proj
//   columns scaled by the mean sigmoid(gate) (also summed on the device);
//   K and V rows coded and packed on the device; the q8_0 rows go to the host store when the job starts
//   a finished job joins the coarse region on the decode thread before a later ubatch (in order); a small batch (packed
//   attention) waits for the jobs in flight first, so its result does not depend on their timing
//   the positions after the coarse region attend precisely (the q8_0 cache)
//   tail (llama_kvq_config::tail): the cache is a ring of the positions whose rows are not in the host store; a large batch
//   that does not fit it starts the jobs of the complete chunks, then copies the ring rows to the host store (spill);
//   an error if that cannot free the room
// memory operations: clear and seq_rm of the end of the sequence; a seq_rm into the coarse region or into a chunk that
// is fitting drops the coarse rows from the chunk of the removal on, the rows before the removal stay in the host store and are
// fitted again by the next llama_memory_kvq_fit_wait
// not supported, each aborts with "not supported by the two-tier KV cache (--kvq)" and the reason: seq_cp, seq_keep and seq_rm
// of a sequence > 0 (one sequence only), seq_rm of a middle range [p0, p1) with p1 >= 0; seq_add, seq_div (qwen35 uses IMROPE,
// no KV cache can shift its keys, llama_memory_can_shift is false)
// the same list is in the "Not supported" section of docs/development/kvq-two-tier-cache-guide.md
//
// the rows after the last complete chunk (rem): llama_memory_kvq_fit_wait (the end of a prompt) fits them as well, so the
// coarse region ends with the prompt; the next fit_wait fits them again from the host store with all rows they have then
//   own:   a chunk of their own, if it has at least min_rows rows (else they stay precise in the ring)
//   merge: one group with the last complete chunk, [(n_full - 1)*chunk, n), one table set; the head chunk is fitted alone again
//          when the next chunk is complete
// the rows of the reply stay precise in the ring until the next fit_wait; when the ring is nearly full during generation,
// the decode thread fits the rows after the last complete chunk in line (at a position set by the token count)
enum llama_kvq_rem {
    LLAMA_KVQ_REM_OWN   = 0,
    LLAMA_KVQ_REM_MERGE = 1,
};

struct llama_kvq_fit_params {
    int32_t     chunk     = 8192;
    int32_t     bits_k    = 2;
    int32_t     bits_v    = 1;
    int32_t     rem       = LLAMA_KVQ_REM_OWN;
    int32_t     min_rows  = 256; // own: the smallest chunk of the rows after the last complete chunk (a covariance of fewer than D + 1 rows is rank deficient)
};

// the memory must be empty and created in packed mode (llama_init_from_model_kvq calls this, use it instead)
// the coarse region can grow to the full chunks of the cache; enabled until the memory is freed, clear() keeps it
LLAMA_API bool llama_memory_kvq_fit_enable(llama_memory_t mem, const llama_kvq_fit_params & fparams, const llama_kvq_params & params);

// everything that makes a context a two-tier KV cache context; the setting belongs to the context, not to the model
// (a second context of the same model, e.g. an MTP drafter, keeps the attention rotation and a normal memory)
struct llama_kvq_config {
    // cells of the ring of the attention cache, in pinned host memory: position p in cell p mod tail, at most one cell per position of
    // the context; the fit removes the positions of the coarse region from it; the context size still limits the positions;
    // the large ubatches attend over the scratch; 0: the larger of n_batch and 4096, plus 256, padded to 256
    int32_t              tail = 0;
    llama_kvq_fit_params fit;    // the fit of the tables (chunk, bits, threads, the rows after the last chunk)
    llama_kvq_params     params; // `enabled` is set by llama_init_from_model_kvq
};

// the one way to create a two-tier KV cache context (qwen35, see llama_model_supports_kvq): the context gets the packed memory
// with the given tail, the attention rotation is off in its attention cache only (the model setting and the other contexts of the
// model are untouched), and the fit is enabled; cparams.n_rs_seq, rs_replay and phase_bufs are taken as they are (common turns
// phase_bufs on for --kvq, it is a policy with a user override); cparams.type_k and type_v must be q8_0, ctx_type the default
// returns nullptr and logs the reason when the model or the settings do not fit, or the memory cannot be created
LLAMA_API struct llama_context * llama_init_from_model_kvq(
                 struct llama_model * model,
        struct llama_context_params   cparams,
              const llama_kvq_config & cfg);

struct llama_kvq_fit_progress {
    int32_t n_coarse  = 0; // rows of the coarse region
    int32_t n_chunk   = 0; // chunks in the coarse region
    int32_t n_running = 0; // jobs queued or running
    int32_t n_ready   = 0; // jobs done, not yet in the coarse region
    int32_t n_cancel  = 0; // jobs cancelled by seq_rm or clear
    int32_t n_drop    = 0; // chunks removed from the coarse region by seq_rm or clear
    bool    failed    = false; // a job failed, no more chunks join the coarse region
    std::vector<double> t_job;  // seconds per job that joined the coarse region: from the start of the job to the join
    std::vector<double> t_wall; // seconds per job that joined: from the readback of the statistics to the join (with the margin)
    double  t_take    = 0.0;    // decode thread seconds of the readbacks of the statistics (with the wait for the queued ubatches), the metrics and the queued graphs
    double  t_install = 0.0;    // decode thread seconds of the joins
    int32_t n_tail_wait = 0;    // tail mode: batches that waited for the fitter before they fit the ring
    double  t_tail_wait = 0.0;  // decode thread seconds of these waits
    int32_t n_host      = 0;    // positions [0, n_host) have their q8_0 rows in the host store
    int32_t n_fold      = 0;    // tail mode: in-line fits of the rows after the last complete chunk because the ring was full (generation)
    double  t_fold      = 0.0;
    int32_t n_spill     = 0;    // tail mode: ring rows copied to the host store without a fit because the ring was full (prefill)
    int32_t n_settle    = 0;    // llama_memory_kvq_fit_wait calls that fitted something
    double  t_settle    = 0.0;  // decode thread seconds of these calls
};

LLAMA_API bool llama_memory_kvq_fit_get_progress(llama_memory_t mem, llama_kvq_fit_progress & out);

// the end of a prompt: fits every chunk that is not fitted yet and the rows after the last complete chunk (see llama_kvq_rem),
// waits for all of it; afterwards no fit work runs. Call it between decodes, from the decode thread
LLAMA_API bool llama_memory_kvq_fit_wait(llama_memory_t mem);

// debug: hold = true: finished chunks join the coarse region only in llama_memory_kvq_fit_wait
LLAMA_API bool llama_memory_kvq_fit_set_hold(llama_memory_t mem, bool hold);

// scheduler splits of the last graph
LLAMA_API int32_t llama_context_n_splits(const struct llama_context * ctx);

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

// Questions packed after a shared state in one sequence, contexts without memory only
// 0 (default) is the state, k > 0 the k-th question; full attention layers hide other questions from a question
LLAMA_API bool llama_batch_ext_set_decision_group(struct llama_batch_ext * batch, int32_t idx, int32_t group);

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
