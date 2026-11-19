#pragma once

#include "llama-ext.h"
#include "llama-kvq-codec.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class llama_kv_cache;
class llama_kvq_store;
class llama_io_write_i;
class llama_io_read_i;

// fits and codes the coarse region of a packed store in fit mode (see llama_kvq_fit_params)
// a job is a range of rows with one table set (a complete chunk, the rows after the last complete chunk, or a group of the
// last complete chunk and these rows), its tasks are the (attention layer, kv head, K or V) triples
// the work on the rows runs on the device, on the main stream between the ubatches; a small pool of CPU threads
// fits the tables from the moments of the rows (K: metric of the queries of the range; V: o_proj columns scaled by the
// mean sigmoid(gate)); one graph per attention layer and phase, the rows of a layer go through a device stage:
//   phase 1 (take):    rows [r0, r1) from the host store (the host ring keeps it complete) -> stage; dequantize, mean, covariance -> host
//   CPU, V tasks first: metric, llama_kvq_fit_cov (Cholesky, eigen solver, tables); K: bits, packed table
//   phase 2 (V tasks done, per-chunk V): rows from the host store; V fwd tables up, rows rotated, scale of the shape -> host
//   V bits (CPU):      bits, steps, packed table
//   phase 3, install (all tasks done): rows from the host store; tables up, rows rotated, codes packed into kc and vc, the range
//                      joins the coarse region (a range that is coarse already gets its new tables and codes in the same poll)
// the pool threads wait for the events of the device phases; the decode thread (poll, wait, remove) never waits for
// device work it queued in the same poll, except in settle and wait_jobs
// a range is always fitted from scratch: its tables are a function of its rows and statistics only
class llama_kvq_fitter {
public:
    // V is fitted per chunk: wo = o_proj weight per attention layer (store with gate statistics), n_head query heads
    // min_first: the smallest coarse region (the prefetch selects from its rows)
    llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams,
            const std::vector<const ggml_tensor *> & wo, int32_t n_head, int32_t min_first);
    ~llama_kvq_fitter();

    // before each ubatch, n_pos positions in the sequence: reads the statistics of the complete chunks; large (prefill): starts
    // the jobs of the complete chunks (a later removal into one cancels it); moves the jobs on, adds the finished ones (unless hold)
    void poll(const llama_kv_cache & kv, llama_pos n_pos, bool large);

    // waits for the jobs in flight and adds them
    void wait_jobs();

    // the layout of the positions [0, n) (see llama_kvq_rem): starts the jobs that are missing, cancels the ones that the
    // layout does not have, waits for all and adds them. Returns the end of the coarse region of the layout
    // queued (optional): called with that end after the jobs are started, before the wait (the GPU is idle during the CPU tasks)
    llama_pos settle(const llama_kv_cache & kv, llama_pos n, const std::function<void(llama_pos)> & queued = nullptr);

    // the ring rows of the positions [n_host, p) to the host store, no fit
    void spill(const llama_kv_cache & kv, llama_pos p);

    // the positions [p0, inf) were removed: cancels the jobs with rows there, the coarse region drops back to the chunk of p0
    void remove(llama_pos p0);

    // the device stage and the graph allocator: allocated at the ctor and again by the first phase after dev_free
    // dev_free: no jobs in flight, after the queued phases
    void dev_free();
    bool dev_allocated() const { return buf_dev != nullptr; }

    bool has_jobs() const { return !jobs_empty(); }

    void get_progress(llama_kvq_fit_progress & out) const;

    void set_hold(bool value) { hold = value; }

    // state save and load, the decode thread: the fitted ranges, the statistics of the complete chunks and the sums of the open chunk
    // the save needs a fitter without jobs (after wait_jobs), else it throws
    // state_read_meta throws on a mismatch; state_read_data after remove(0) and the store's state_read_data
    void state_write_meta(llama_io_write_i & io) const;
    void state_write_data(llama_io_write_i & io) const;
    void state_read_meta (llama_io_read_i  & io) const;
    void state_read_data (llama_io_read_i  & io);

private:
    struct job;

    struct range {
        int32_t r0 = -1;
        int32_t r1 = -1;
        bool operator==(const range & o) const { return r0 == o.r0 && r1 == o.r1; }
    };

    llama_kvq_store & store;

    const llama_kvq_fit_params fparams;

    const int32_t min_first;
    int32_t       n_stage = 0; // rows of the largest job

    std::vector<int32_t> layer_ids;

    // per-chunk V: Wo_r^T Wo_r of the n_rep query heads r of each (layer index, kv head), computed once by the first task
    int32_t n_head = 0;
    int32_t n_rep  = 0;
    std::vector<const ggml_tensor *>  wo;
    std::vector<std::vector<float>>   wo_gram; // [n_rep][D*D]
    std::vector<char>                 wo_dev;  // wo_gram computed on the device
    std::unique_ptr<std::once_flag[]> wo_once;

    // statistics of the complete chunks [0, n_snap), read when the chunk was complete: to fit it, and to restore them after a seq_rm into it
    int32_t n_snap = 0;
    std::vector<std::vector<float>> grams;
    std::vector<std::vector<float>> gsums;
    std::vector<double>             t_snaps;

    // per chunk slot: the range whose fit gave its tables, r0 = -1: none
    std::vector<range> slot_fit;

    // the device sums of the open chunk when the rows up to acc_pos were fitted, -1: none; a removal back to acc_pos restores them
    int32_t            acc_pos = -1;
    std::vector<float> acc_gram;
    std::vector<float> acc_gsum;

    bool failed = false;
    bool hold   = false;

    // device state of the phases: the identity rows, the tables of one layer, the q8_0 rows of one layer
    ggml_context_ptr        ctx_dev;
    ggml_backend_buffer_ptr buf_dev;
    ggml_tensor *           idx     = nullptr; // i32 [n_stage, G], 0 .. n_stage - 1
    ggml_tensor *           par     = nullptr; // f32 [D, D + 5, 2*G], per (K or V, kv head): fwd, mu, step, bits, offsets, den
    ggml_tensor *           stage   = nullptr; // q8_0 [D, 2, n_stage, G], K and V rows in the layout of the host store
    ggml_tensor *           cst[2]  = {};      // i32 [nwk or nwv + 1, n_stage, G], phase 3 codes when the device has no kc / vc
    ggml_gallocr_ptr        galloc;
    std::vector<uint8_t>    graph_meta;

    mutable std::mutex      m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<std::pair<job *, int32_t>> tasks;
    std::deque<std::unique_ptr<job>>      jobs; // in the order of the takes
    std::vector<std::thread> threads;
    std::vector<ggml_backend_buffer_ptr> free_bufs; // host buffers of finished jobs
    bool stop = false;

    // progress
    int32_t n_cancel = 0;
    int32_t n_drop   = 0;
    int32_t n_spill  = 0;
    std::vector<double> t_job;
    std::vector<double> t_wall;
    double t_cpu     = 0.0;
    double t_take    = 0.0;
    double t_install = 0.0;

    // (layer, kv head) pairs, task i < n_pair is K, then V; i >= 2*n_pair: V bits task of V task i - n_pair
    int32_t n_pair() const;

    std::vector<range> layout(llama_pos n) const;
    bool fitted(const range & r) const;
    bool in_flight(const range & r) const;
    bool jobs_empty() const;
    static bool can_advance(const job & jb);
    void snapshots(llama_pos n_pos);
    void read_acc(std::vector<float> & gram, std::vector<float> & gsum);
    bool check_ring(const llama_kv_cache & kv, int32_t p0, int32_t p1);
    bool take(const range & r);
    void cancel(const std::function<bool(const job &)> & pred);
    void advance();
    void install(bool all);
    void dev_alloc();
    void upload_rows(int32_t il, int32_t r0, int32_t r1);
    void queue_phase(job & jb, int phase);
    ggml_cgraph * build_phase(const job & jb, int phase, int32_t li, ggml_context * ctx);
    void run(job & jb, int32_t i);
    void fit_task(job & jb, int32_t i);
    void bits_task(job & jb, int32_t i);
    void release(job & jb);
    void compute_wo_gram(int32_t p);
    void compute_wo_gram_dev();
    void metric_v(const job & jb, int32_t p, std::vector<double> & m);
    void set_stats(const std::vector<float> * gram, const std::vector<float> * gsum);
    void loop();
};
