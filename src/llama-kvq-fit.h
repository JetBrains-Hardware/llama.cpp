#pragma once

#include "llama-ext.h"
#include "llama-kvq-codec.h"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class llama_kv_cache;
class llama_kvq_store;

// fits and codes the chunks of the coarse region of a packed store in fit mode (see llama_kvq_fit_params)
// a job is one chunk, its tasks are the (attention layer, kv head) pairs, run in order by a pool of CPU threads
// a task reads the q8_0 K/V rows of its kv head from the cache, writes them to the host store, fits the K table,
// codes and packs the K and V rows
// the decode thread (poll, wait, remove) reads the query statistics, starts the jobs and adds the finished chunks to the store
class llama_kvq_fitter {
public:
    // v_tabs: one V table per (attention layer, kv head)
    llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams, const std::vector<llama_kvq_table> & v_tabs);
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

    // V tables, per (layer index, kv head)
    std::vector<llama_kvq_table> v_tab;

    int32_t n_taken = 0; // chunks whose statistics were read
    int32_t n_coarse_chunks = 0;
    bool    failed  = false;
    bool    hold    = false;

    // statistics of the chunks in the coarse region, to restore them after a seq_rm into a chunk
    std::vector<std::vector<float>> grams;

    mutable std::mutex      m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<std::pair<job *, int32_t>> tasks;
    std::deque<std::unique_ptr<job>>      jobs; // in chunk order
    std::vector<std::thread> threads;
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
    void loop();
};
