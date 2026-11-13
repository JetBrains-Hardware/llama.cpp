#pragma once

#include "llama-ext.h"
#include "llama-kvq-codec.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class llama_kv_cache;
class llama_kvq_store;

// fits and codes the chunks of the coarse region of a packed store in fit mode (see llama_kvq_fit_params)
// a job is one chunk, its tasks are the (attention layer, kv head, K or V) triples, run in order by a pool of CPU threads
// the decode thread reads the q8_0 rows of the chunk from the cache when it starts the job
// a task takes the K or V rows of its kv head, writes them to the host store, fits the table
// (K: metric of the chunk's queries; V: o_proj columns scaled by the chunk's mean sigmoid(gate)), codes and packs the rows
// the decode thread (poll, wait, remove) reads the statistics, starts the jobs and adds the finished chunks to the store
class llama_kvq_fitter {
public:
    // v_tabs: one V table per (attention layer, kv head), used for every chunk
    // or empty: V is fitted per chunk, wo = o_proj weight per attention layer (store with gate statistics), n_head query heads
    llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams, const std::vector<llama_kvq_table> & v_tabs,
            const std::vector<const ggml_tensor *> & wo, int32_t n_head);
    ~llama_kvq_fitter();

    // before each ubatch: starts the jobs of the complete chunks, adds the finished ones (unless hold)
    void poll(const llama_kv_cache & kv);

    // poll, then waits for all jobs and adds their chunks
    void wait(const llama_kv_cache & kv);

    // the positions [p0, inf) were removed: cancels the jobs and drops the coarse chunks at or after the chunk of p0
    void remove(llama_pos p0);

    void get_progress(llama_kvq_fit_progress & out) const;

    void set_hold(bool value) { hold = value; }

private:
    struct job;

    llama_kvq_store & store;

    const llama_kvq_fit_params fparams;

    std::vector<int32_t> layer_ids;

    // offline V tables, per (layer index, kv head)
    std::vector<llama_kvq_table> v_tab;

    // per-chunk V: Wo_r^T Wo_r of the n_rep query heads r of each (layer index, kv head), computed once by the first task
    bool    v_fit  = false;
    int32_t n_head = 0;
    int32_t n_rep  = 0;
    std::vector<const ggml_tensor *>  wo;
    std::vector<std::vector<float>>   wo_gram; // [n_rep][D*D]
    std::unique_ptr<std::once_flag[]> wo_once;

    int32_t n_taken = 0; // chunks whose statistics were read
    int32_t n_coarse_chunks = 0;
    bool    failed  = false;
    bool    hold    = false;

    // statistics of the chunks in the coarse region, to restore them after a seq_rm into a chunk
    std::vector<std::vector<float>> grams;
    std::vector<std::vector<float>> gsums;

    mutable std::mutex      m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<std::pair<job *, int32_t>> tasks;
    std::deque<std::unique_ptr<job>>      jobs; // in chunk order
    std::vector<std::thread> threads;
    std::vector<ggml_backend_buffer_ptr> free_bufs; // rows buffers of finished jobs
    bool stop = false;

    // progress
    int32_t n_cancel = 0;
    int32_t n_drop   = 0;
    std::vector<double> t_job;
    std::vector<double> t_wall;
    double t_cpu    = 0.0;
    double t_take    = 0.0;
    double t_install = 0.0;

    void take(const llama_kv_cache & kv);
    void install(bool all);
    void run(job & jb, int32_t i);
    void release(job & jb);
    void compute_wo_gram(int32_t p);
    void metric_v(const job & jb, int32_t p, std::vector<double> & m);
    void set_stats(const std::vector<float> * gram, const std::vector<float> * gsum);
    void loop();
};
