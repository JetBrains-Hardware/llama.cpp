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
class llama_io_write_i;
class llama_io_read_i;

// fits and codes the chunks of the coarse region of a packed store in fit mode (see llama_kvq_fit_params)
// a job is one chunk, its tasks are the (attention layer, kv head, K or V) triples
// the work on the rows of a chunk runs on the device, on the main stream between the ubatches; a small pool of CPU threads
// fits the tables from the moments of the rows (K: metric of the chunk's queries; V: o_proj columns scaled by the chunk's
// mean sigmoid(gate)); one graph per attention layer and phase:
//   phase 1 (take):    q8_0 rows of the chunk from the ring -> host store; dequantize, mean, covariance -> host
//   CPU, V tasks first: metric, llama_kvq_fit_cov (Cholesky, eigen solver, tables); K: bits, packed table
//   phase 2 (V tasks done, per-chunk V): V fwd tables up, rows rotated, scale of the shape -> host
//   V bits (CPU):      bits, steps, packed table
//   phase 3, install (all tasks done): tables up, rows rotated, codes packed into kc and vc, the chunk joins the coarse region
// the pool threads wait for the events of the device phases; the decode thread (poll, wait, remove) never waits for
// device work it queued in the same poll
class llama_kvq_fitter {
public:
    // v_tabs: one V table per (attention layer, kv head), used for every chunk
    // or empty: V is fitted per chunk, wo = o_proj weight per attention layer (store with gate statistics), n_head query heads
    // margin: a chunk is taken when the sequence is this many positions past its end (poll)
    llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams, const std::vector<llama_kvq_table> & v_tabs,
            const std::vector<const ggml_tensor *> & wo, int32_t n_head, int32_t margin);
    ~llama_kvq_fitter();

    // before each ubatch: starts the jobs of the complete chunks, moves the jobs on, adds the finished ones (unless hold)
    void poll(const llama_kv_cache & kv) { poll(kv, margin); }

    // starts the jobs of all complete chunks, waits for them and adds their chunks
    void wait(const llama_kv_cache & kv);

    // adds at least one more chunk to the coarse region (waits for the first job), false if there is no job to wait for
    bool wait_front(const llama_kv_cache & kv);

    // the positions [p0, inf) were removed: cancels the jobs and drops the coarse chunks at or after the chunk of p0
    void remove(llama_pos p0);

    void get_progress(llama_kvq_fit_progress & out) const;

    void set_hold(bool value) { hold = value; }

    // state save and load, the decode thread: the statistics of the coarse chunks (for a later removal into a chunk) and the sums of the open chunk
    // the save needs a fitter without jobs and without complete chunks outside the coarse region (after wait), else it throws
    // state_read_meta throws on a mismatch; state_read_data after remove(0) and the store's state_read_data, n_chunk coarse chunks
    void state_write_meta(llama_io_write_i & io) const;
    void state_write_data(llama_io_write_i & io) const;
    void state_read_meta (llama_io_read_i  & io, int32_t n_chunk) const;
    void state_read_data (llama_io_read_i  & io, int32_t n_chunk);

private:
    struct job;

    llama_kvq_store & store;

    const llama_kvq_fit_params fparams;

    const int32_t margin;

    std::vector<int32_t> layer_ids;

    // offline V tables, per (layer index, kv head)
    std::vector<llama_kvq_table> v_tab;

    // per-chunk V: Wo_r^T Wo_r of the n_rep query heads r of each (layer index, kv head), computed once by the first task
    bool    v_fit  = false;
    int32_t n_head = 0;
    int32_t n_rep  = 0;
    std::vector<const ggml_tensor *>  wo;
    std::vector<std::vector<float>>   wo_gram; // [n_rep][D*D]
    std::vector<char>                 wo_dev;  // wo_gram computed on the device
    std::unique_ptr<std::once_flag[]> wo_once;

    int32_t n_taken = 0; // chunks whose jobs were started
    int32_t n_snap  = 0; // chunks whose statistics were read

    // statistics of the complete chunks whose jobs did not start yet (the margin), in chunk order
    struct snap {
        int32_t c;
        double  t;
        std::vector<float> gram;
        std::vector<float> gsum;
    };
    std::deque<snap> snaps;
    int32_t n_coarse_chunks = 0;
    bool    failed  = false;
    bool    hold    = false;

    // statistics of the chunks in the coarse region, to restore them after a seq_rm into a chunk
    std::vector<std::vector<float>> grams;
    std::vector<std::vector<float>> gsums;

    // device state of the phases: the cells of the chunk, the tables of one layer, the q8_0 rows of one layer for the host store
    ggml_context_ptr        ctx_dev;
    ggml_backend_buffer_ptr buf_dev;
    ggml_tensor *           idx   = nullptr; // i32 [C, G], cell of each row of the chunk
    ggml_tensor *           par   = nullptr; // f32 [D, D + 5, 2*G], per (K or V, kv head): fwd, mu, step, bits, offsets, den
    ggml_tensor *           stage = nullptr; // f16 [ROW/2, 2, C, G], K and V rows in the layout of the host store
    ggml_gallocr_ptr        galloc;
    std::vector<uint8_t>    graph_meta;
    bool                    prof = false; // LLAMA_KVQ_FIT_SYNC: synchronize around the device phases, device time per chunk

    mutable std::mutex      m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<std::pair<job *, int32_t>> tasks;
    std::deque<std::unique_ptr<job>>      jobs; // in chunk order
    std::vector<std::thread> threads;
    std::vector<ggml_backend_buffer_ptr> free_bufs; // host buffers of finished jobs
    bool stop = false;

    // progress
    int32_t n_cancel = 0;
    int32_t n_drop   = 0;
    std::vector<double> t_job;
    std::vector<double> t_wall;
    double t_cpu     = 0.0;
    double t_take    = 0.0;
    double t_install = 0.0;
    double t_dev     = 0.0;

    // (layer, kv head) pairs, task i < n_pair is K, then V; i >= 2*n_pair: V bits task of V task i - n_pair
    int32_t n_pair() const;

    void poll(const llama_kv_cache & kv, int32_t margin);
    bool jobs_empty() const;
    static bool can_advance(const job & jb);
    void snapshot();
    void take(const llama_kv_cache & kv);
    void advance();
    void install(bool all);
    void queue_phase(job & jb, int phase);
    ggml_cgraph * build_phase(job & jb, int phase, int32_t li, ggml_context * ctx);
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
