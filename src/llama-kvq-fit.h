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
// the whole fit runs on the device, on the main stream between the ubatches (ggml_kvq_fit), in the shared memory of the store
// (llama_kvq_store::fit_alloc: idle between the ubatches), the attention layers in groups of n_grp:
//   take: the metrics from the statistics on the host (K: the queries of the range; V: o_proj columns scaled by the mean
//         sigmoid(gate)); per group: per layer rows [r0, r1) from the host store (the host ring keeps it complete) -> stage,
//         mean and covariance -> the tables of the group; one graph for the tables of the group (Cholesky, eigen solver);
//         per layer the rows again: K bits; V rotated, scale of the shape, V bits; packed tables and codes -> the host copies
//   install (the job's event): the range joins the coarse region (a range that is coarse already gets its new tables and codes)
// a waiter thread waits for the events; the decode thread (poll, wait, remove) never waits for device work it queued in the
// same poll, except in settle and wait_jobs
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

    // the smallest shared memory of the store for a fit of one layer per group, n_stage rows per job
    static size_t shared_min(int32_t n_head_kv, int32_t n_stage, int32_t bits_k, int32_t bits_v);

    // the rows of the largest job
    static int32_t stage_rows(int32_t n_rows, const llama_kvq_fit_params & fparams);

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

    // per-chunk V: Wo_r^T Wo_r of the n_rep query heads r of each (layer index, kv head), computed by the ctor
    int32_t n_head = 0;
    int32_t n_rep  = 0;
    std::vector<const ggml_tensor *> wo;
    std::vector<std::vector<float>>  wo_gram; // [n_rep][D*D]

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

    // device state: own buffer (idx, tinv); in the shared memory of the store at fixed offsets: stage, cst, tab, then the
    // intermediates of each graph from plan_lo on (plan)
    ggml_context_ptr        ctx_own;
    ggml_backend_buffer_ptr buf_own;
    ggml_context_ptr        ctx_shr;
    ggml_tensor *           idx     = nullptr; // i32 [n_stage, G], 0 .. n_stage - 1
    ggml_tensor *           tinv    = nullptr; // i8 [D, D, 2*G], the packed inverse tables of one layer
    ggml_tensor *           stage   = nullptr; // q8_0 [D, 2, n_stage, G], K and V rows in the layout of the host store
    ggml_tensor *           cst[2]  = {};      // i32 [nwk or nwv + 1, n_stage, G], the codes when the device has no kc / vc
    ggml_tensor *           tab     = nullptr; // f32 [D, 2*D + 3, 2*G*n_grp], the tables of a group: K of its layers, then V
    int32_t                 n_grp   = 0;       // attention layers per group
    size_t                  plan_lo = 0;       // the graphs place their intermediates in the shared memory from here
    std::vector<uint8_t>    graph_meta;

    mutable std::mutex      m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<std::unique_ptr<job>>     jobs; // in the order of the takes
    std::thread                          waiter;
    std::vector<ggml_backend_buffer_ptr> free_bufs; // host buffers of finished jobs
    bool stop = false;

    // progress
    int32_t n_cancel = 0;
    int32_t n_drop   = 0;
    int32_t n_spill  = 0;
    std::vector<double> t_job;
    std::vector<double> t_wall;
    double t_take    = 0.0;
    double t_install = 0.0;

    // (layer, kv head) pairs, task i < n_pair is K, then V
    int32_t n_pair() const;

    std::vector<range> layout(llama_pos n) const;
    bool fitted(const range & r) const;
    bool in_flight(const range & r) const;
    bool jobs_empty() const;
    void snapshots(llama_pos n_pos);
    void read_acc(std::vector<float> & gram, std::vector<float> & gsum);
    bool check_ring(const llama_kv_cache & kv, int32_t p0, int32_t p1);
    bool take(const range & r);
    void cancel(const std::function<bool(const job &)> & pred);
    void install(bool all);
    void dev_init();
    size_t plan(ggml_cgraph * gf, size_t lo, bool place);
    void compute(ggml_cgraph * gf, const char * what);
    void upload_rows(int32_t il, int32_t r0, int32_t r1);
    void host_metrics(job & jb);
    void queue_job(job & jb);
    ggml_tensor * tab_view(int kv, int32_t j, int32_t n_lay, size_t row0, int64_t n_rows, ggml_context * ctx) const;
    ggml_cgraph * build_moments(const job & jb, int32_t j, int32_t n_lay, ggml_context * ctx);
    ggml_cgraph * build_tables(const job & jb, int32_t n_lay, ggml_context * ctx);
    ggml_cgraph * build_codes(const job & jb, int32_t li, int32_t j, int32_t n_lay, ggml_context * ctx);
    void release(job & jb);
    void compute_wo_gram(int32_t p);
    void compute_wo_gram_dev(std::vector<char> & on_dev);
    void metric_v(const job & jb, int32_t p, float * m);
    void set_stats(const std::vector<float> * gram, const std::vector<float> * gsum);
    void loop();
};
