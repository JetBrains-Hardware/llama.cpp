#include "llama-memory-hybrid-kvq.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_memory_hybrid_kvq
//

// the cells of the ring of the attention cache
static uint32_t kvq_ring_cells(uint32_t n_tail, uint32_t kv_size) {
    GGML_ASSERT(n_tail > 0 && "the two-tier cache needs the attention cache as a ring (llama_kvq_config::tail)");
    return std::min(n_tail, kv_size);
}

llama_memory_hybrid_kvq::llama_memory_hybrid_kvq(
        const llama_model & model,
                     bool   no_alloc,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   rs_replay,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                 uint32_t   n_tail,
                     bool   swa_full,
                 uint32_t   n_ubatch) :
    llama_memory_hybrid(
        model,
        no_alloc,
        type_k, type_v, v_trans, kvq_ring_cells(n_tail, kv_size),
        swa_type != LLAMA_SWA_TYPE_NONE && model.hparams.is_swa_any() ? 0u : n_swa,
        swa_type != LLAMA_SWA_TYPE_NONE && model.hparams.is_swa_any() ? LLAMA_SWA_TYPE_NONE : swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, rs_replay, offload, /* one ring stream per sequence, whatever the context does with its cells */ false,
        swa_type != LLAMA_SWA_TYPE_NONE && model.hparams.is_swa_any() ?
            layer_filter_cb([&model, filter_attn](int32_t il) {
                return !model.hparams.is_swa(il) && (filter_attn ? filter_attn(il) : !model.hparams.is_recr(il));
            }) : filter_attn,
        filter_recr,
        /* attn_rot */ false),
    model(model),
    n_ctx_max(kv_size),
    rel_keep((llama_pos) model.hparams.n_relative_extent) {
    GGML_UNUSED(unified);
    if (swa_type != LLAMA_SWA_TYPE_NONE && model.hparams.is_swa_any()) {
        const uint32_t kv_size_swa = swa_full ? kv_size : GGML_PAD(std::min(kv_size, n_swa + n_ubatch), 256);
        mem_swa = std::make_unique<llama_kv_cache>(
            model,
            no_alloc,
            model.hparams,
            type_k,
            type_v,
            v_trans,
            offload,
            /* unified */ false,
            kv_size_swa,
            n_seq_max,
            n_swa,
            swa_type,
            nullptr,
            [&model, filter_attn](int32_t il) {
                return model.hparams.is_swa(il) && (filter_attn ? filter_attn(il) : !model.hparams.is_recr(il));
            },
            nullptr,
            nullptr,
            "SWA");
    }
    get_mem_attn()->set_ring(true);
    LLAMA_LOG_INFO("%s: tail: the attention cache is a ring of %u cells per stream (%u streams) for %u positions%s\n", "llama_memory_hybrid_kvq",
            get_mem_attn()->get_size(), get_mem_attn()->get_n_stream(), n_ctx_max,
            mem_swa ? format(", SWA cache of %u cells per stream", mem_swa->get_size()).c_str() : "");
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    std::vector<llama_ubatch> ubatches;
    slot_info_vec_t heads_attn;

    // the sequences of the batch and the number of their tokens: a ubatch is one sequence, the batch is split by sequence
    std::vector<llama_seq_id> bseq;
    std::vector<uint32_t>     bnum;
    if (fitter) {
        const llama_batch & b = balloc.get_batch();
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            if (b.n_seq_id[i] != 1 || b.seq_id[i][0] < 0 || b.seq_id[i][0] >= store->n_seq()) {
                LLAMA_LOG_ERROR("%s: the two-tier cache needs one sequence id below n_seq_max (%d) for every token\n", __func__, store->n_seq());
                return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
            }
            const auto it = std::find(bseq.begin(), bseq.end(), b.seq_id[i][0]);
            if (it == bseq.end()) {
                bseq.push_back(b.seq_id[i][0]);
                bnum.push_back(1);
            } else {
                bnum[it - bseq.begin()]++;
            }
        }
    }

    // the pages of the chunks of the batch positions first: a full pool fails the batch like a full cache
    std::vector<int32_t> n_pg0(bseq.size(), 0);
    auto release = [&](size_t n) {
        for (size_t i = 0; i < n; ++i) {
            store->release_from(bseq[i], n_pg0[i]);
        }
    };
    for (size_t i = 0; i < bseq.size(); ++i) {
        n_pg0[i] = store->n_page_chunks(bseq[i]);
        const llama_pos p_end = balloc.seq_pos_max(bseq[i]);
        if (p_end >= 0 && !store->reserve(bseq[i], p_end + 1)) {
            release(i);
            return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
    }

    // the slots of the whole batch are placed here, so the ring needs the room for all of it now
    for (size_t i = 0; i < bseq.size(); ++i) {
        if (!tail_wait(balloc, bseq[i], bnum[i])) {
            release(bseq.size());
            return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
    }

    // the sequences whose tokens of the batch take the packed path (a whole decode / verify run each) with the same number of tokens
    // go through one ubatch (the weights are read once for all of them), at most n_streams per ubatch; every other sequence has its
    // own ubatches. A group of one sequence is no group
    std::vector<std::vector<llama_seq_id>> groups;
    if (fitter && !embd_all && store->n_streams() > 1) {
        std::vector<std::pair<uint32_t, std::vector<llama_seq_id>>> by_len;
        for (size_t i = 0; i < bseq.size(); ++i) {
            if (!llama_kvq_use_packed(params, bnum[i], store->n_coarse(bseq[i]), store->n_host(bseq[i]))) {
                continue;
            }
            auto it = std::find_if(by_len.begin(), by_len.end(), [&](const auto & e) { return e.first == bnum[i]; });
            if (it == by_len.end()) {
                by_len.push_back({ bnum[i], {} });
                it = by_len.end() - 1;
            }
            it->second.push_back(bseq[i]);
        }
        for (auto & e : by_len) {
            std::sort(e.second.begin(), e.second.end());
            for (size_t i = 0; i < e.second.size(); i += store->n_streams()) {
                const size_t n = std::min<size_t>(store->n_streams(), e.second.size() - i);
                if (n > 1) {
                    groups.emplace_back(e.second.begin() + i, e.second.begin() + i + n);
                }
            }
        }
    }

    if (!prepare_batch(balloc, n_ubatch, embd_all, ubatches, heads_attn, fitter ? 1 : UINT32_MAX, &groups)) {
        release(bseq.size());
        return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    slot_info_vec_t heads_swa;
    if (mem_swa) {
        heads_swa = mem_swa->prepare(ubatches);
        if (heads_swa.empty()) {
            release(bseq.size());
            return std::make_unique<llama_memory_hybrid_kvq_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
    }

    return std::make_unique<llama_memory_hybrid_kvq_context>(this, std::move(heads_attn), std::move(heads_swa), std::move(ubatches));
}

llama_pos llama_memory_hybrid_kvq::n_pos(llama_seq_id seq) const {
    const llama_pos p = get_mem_attn()->seq_pos_max(seq);
    return std::max<llama_pos>(p + 1, store ? store->n_host(seq) : 0);
}

bool llama_memory_hybrid_kvq::tail_wait(const llama_batch_allocr & balloc, llama_seq_id seq, uint32_t n_tokens) {
    seq_state & sq = seqs[seq];
    const llama_pos p0    = balloc.seq_pos_min(seq);
    const llama_pos p_end = balloc.seq_pos_max(seq);
    const int64_t   size  = get_mem_attn()->get_size();
    if (p_end < 0) {
        return true;
    }
    const llama_kv_cache & kv = *get_mem_attn();
    const int64_t t0 = ggml_time_us();

    // small batches attend through the packed path: the jobs in flight join first, so the path does not depend on their timing
    const bool small = n_tokens <= (uint32_t) params.max_batch;
    if (small && fitter->has_jobs(seq)) {
        fitter->wait_jobs(seq);
        tail_release(seq);
    }

    // generation: the fold position depends on the first position of the batch only
    const llama_pos last = small ? std::max<llama_pos>(p_end, p0 + params.max_batch - 1) : p_end;
    auto room = [&]() { return last + 1 - (int64_t) sq.n_released <= size; };
    const bool fold_now = small && fold_every > 0 && p0 - store->n_coarse(seq) >= fold_every;

    // host ring: a large batch reads the rows before it from the host store
    if (!small) {
        store->sync_store(seq, n_pos(seq));
    }

    bool waited = false;
    if (!room() && !small) {
        // the complete chunks before the batch, then the rest of the ring rows to the host store
        fitter->poll(seq, kv, n_pos(seq), true);
        tail_release(seq);
        if (!room()) {
            fitter->spill(seq, kv, p0);
            tail_release(seq);
        }
        waited = true;
    }
    if ((!room() || fold_now) && small) {
        // the rows after the last complete chunk are fitted in line
        const int64_t t1 = ggml_time_us();
        if (!settle(seq, p0)) {
            return false;
        }
        sq.n_fold++;
        sq.t_fold += (ggml_time_us() - t1)*1e-6;
        LLAMA_LOG_INFO("%s: fold at position %d: coarse region %d rows, %d positions in the ring, %.3f s (%d folds, %.2f s in all)\n", __func__,
                p0, store->n_coarse(seq), (int) (p0 - sq.n_released), (ggml_time_us() - t1)*1e-6, sq.n_fold, sq.t_fold);
    }
    if (!room()) {
        llama_kvq_fit_progress pr;
        fitter->get_progress(seq, pr);
        LLAMA_LOG_ERROR("%s: the ring of %lld cells cannot hold the positions [%d, %d] of the batch: %s\n", __func__,
                (long long) size, sq.n_released, last, pr.failed ? "the fit failed" : "the batch is too large for the ring (see llama_kvq_config::tail)");
        return false;
    }
    if (waited) {
        sq.n_tail_wait++;
        sq.t_tail_wait += (ggml_time_us() - t0)*1e-6;
        LLAMA_LOG_DEBUG("%s: the batch up to position %d made room in the ring in %.3f s (%d batches, %.2f s in all)\n", __func__,
                p_end, (ggml_time_us() - t0)*1e-6, sq.n_tail_wait, sq.t_tail_wait);
    }
    return true;
}

void llama_memory_hybrid_kvq::tail_release(llama_seq_id seq) {
    if (!store) {
        return;
    }
    const llama_pos n = store->n_host(seq);
    if (n > seqs[seq].n_released) {
        get_mem_attn()->seq_rm(seq, seqs[seq].n_released, n);
        seqs[seq].n_released = n;
    }
}

bool llama_memory_hybrid_kvq::tail_restore(llama_seq_id seq, llama_pos p_lo, llama_pos p_hi) {
    if (!store || p_hi <= p_lo) {
        return true;
    }
    llama_kv_cache * kv = get_mem_attn();
    const int64_t size = kv->get_size();
    if (n_pos(seq) - p_lo > size) {
        LLAMA_LOG_ERROR("%s: the ring of %lld cells cannot hold the positions [%d, %d)\n", __func__, (long long) size, p_lo, n_pos(seq));
        return false;
    }
    GGML_ASSERT(p_lo >= store->n_coarse(seq) && p_hi == store->n_host(seq) && seqs[seq].n_released == p_hi);
    // the host store keeps the rows, the next packed ubatch copies them into the host ring
    kv->ring_add(seq, p_lo, p_hi);
    store->set_n_host(seq, p_lo);
    seqs[seq].n_released = p_lo;
    return true;
}

bool llama_memory_hybrid_kvq::settle(llama_seq_id seq, llama_pos n, bool end) {
    store->sync_store(seq, n_pos(seq));
    // models with relative-position bias keep the trailing rel_keep positions precise in the ring so all coarse rows have dist >= rel_keep
    const llama_pos n_settle = rel_keep > 0 ? std::max<llama_pos>(0, n - rel_keep) : n;
    // the end of a prompt: the context frees its prompt buffers, then the device codes are allocated and the host copy uploaded
    // while the CPU fits the last rows
    const llama_pos e = fitter->settle(seq, *get_mem_attn(), n_settle, [&](llama_pos e_lay) {
        if (end && e_lay > 0) {
            if (tg_cb) {
                tg_cb();
            }
            // the result is ignored on purpose: a failure is retried before the first decode graph (llama-context.cpp, process_ubatch, kctx->codes_alloc) and returns ALLOC_FAILED
            store->codes_alloc();
        }
    });
    tail_release(seq);
    if (store->n_host(seq) > e && !tail_restore(seq, e, store->n_host(seq))) {
        return false;
    }
    // the decoding after a prompt depends on the memory state, not on the resident rows of an earlier reply
    store->reset_resident(seq);
    return true;
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_full() {
    return std::make_unique<llama_memory_hybrid_kvq_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_kvq_context>(this, lctx, optimize);
}

llama_memory_context_ptr llama_memory_hybrid_kvq::init_full_packed() {
    return std::make_unique<llama_memory_hybrid_kvq_context>(this, true);
}

uint32_t llama_memory_hybrid_kvq::n_packed_streams() const {
    return store && params.enabled && store->scratch_ok() ? (uint32_t) store->n_streams() : 0;
}

bool llama_memory_hybrid_kvq::get_can_shift() const {
    return false;
}

void llama_memory_hybrid_kvq::clear(bool data) {
    llama_memory_hybrid::clear(data);
    if (mem_swa) {
        mem_swa->clear(data);
    }

    if (fitter) {
        for (llama_seq_id s = 0; s < (llama_seq_id) seqs.size(); ++s) {
            fitter->remove(s, 0);
            seqs[s].n_released = 0;
        }
        // the sequences hold no rows: their pages go back to the pool
        store->release_all();
        return;
    }

    store.reset();
    params.enabled = false;
}

// the sequence holds nothing: its ring cells and recurrent cell, jobs, coarse region, host rows, pages (the other sequences stay)
void llama_memory_hybrid_kvq::seq_clear(llama_seq_id seq) {
    if (store->n_seq() == 1) {
        clear(false);
        return;
    }
    llama_memory_hybrid::seq_rm(seq, -1, -1);
    if (mem_swa) {
        mem_swa->seq_rm(seq, -1, -1);
    }
    fitter->remove(seq, 0);
    seqs[seq].n_released = 0;
    store->release_from(seq, 0);
}

bool llama_memory_hybrid_kvq::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (fitter) {
        if (seq_id >= store->n_seq()) {
            GGML_ABORT("seq_rm of sequence %d: the two-tier KV cache (--kvq) has %d sequences (n_seq_max)", seq_id, store->n_seq());
        }
        if (seq_id < 0 && store->n_seq() > 1) {
            // every sequence
            bool ok = true;
            for (llama_seq_id s = 0; s < store->n_seq(); ++s) {
                ok = seq_rm(s, p0, p1) && ok;
            }
            return ok;
        }
        // the attention positions: a restored recurrent checkpoint can end before them; the ring is empty when they end with the host store
        // (seq_id -1 is every sequence, here the one)
        const llama_seq_id seq = std::max<llama_seq_id>(seq_id, 0);
        const llama_pos pos_max = n_pos(seq) - 1;
        const llama_pos q0      = std::max<llama_pos>(p0, 0);
        const llama_pos q1      = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
        if (q0 >= q1 || q0 > pos_max) {
            // nothing to remove, the chunks after the positions have no rows: their pages (a failed batch) go back
            if (q0 < q1) {
                store->release_from(seq, (pos_max + 1 + store->n_chunk_rows() - 1)/store->n_chunk_rows());
            }
            if (mem_swa) {
                mem_swa->seq_rm(seq_id, p0, p1);
            }
            return llama_memory_hybrid::seq_rm(seq_id, p0, p1);
        }
        if (q1 <= pos_max) {
            GGML_ABORT("seq_rm of positions [%d, %d) is not supported by the two-tier KV cache (--kvq): it can only remove the end of the sequence (here [p0, -1) with p0 <= %d). Removing a middle range is not supported. Disable the context shift and --cache-reuse, or run without --kvq.", p0, p1, pos_max);
        }
        if (!llama_memory_hybrid::seq_rm(seq_id, p0, p1)) {
            return false;
        }
        if (mem_swa) {
            mem_swa->seq_rm(seq_id, p0, p1);
        }
        // the rows before q0 that leave the coarse region stay in the host store, the next fit_wait fits them again
        fitter->remove(seq, q0);
        seqs[seq].n_released = std::min(seqs[seq].n_released, q0);
        // the pages of the chunks without rows go back to the pool, the chunk that keeps rows stays
        store->release_from(seq, (q0 + store->n_chunk_rows() - 1)/store->n_chunk_rows());
        // the chunk that keeps rows gets new rows, the next end of prompt fits it again: it must not be a page that another sequence holds
        // (copy on write; its kept sums and fitted range were dropped above, the copy has the rows before q0)
        if (q0 % store->n_chunk_rows() != 0 && !store->make_private(seq, q0/store->n_chunk_rows())) {
            LLAMA_LOG_ERROR("%s: no page for a private copy of the shared chunk %d of sequence %d: the page pool is full, the sequence is emptied\n", __func__, q0/store->n_chunk_rows(), seq);
            seq_clear(seq);
            return false;
        }
        return true;
    }

    if (store && (p1 < 0 || p1 > 0) && (p0 < 0 || p0 < store->n_coarse(std::max<llama_seq_id>(seq_id, 0)))) {
        GGML_ABORT("seq_rm of positions [%d, %d) is not supported by the two-tier KV cache (--kvq): it would remove the packed coarse region [0, %d), only the end of the sequence after it can be removed. Disable the context shift and --cache-reuse, or run without --kvq.", p0, p1, store->n_coarse(std::max<llama_seq_id>(seq_id, 0)));
    }

    return llama_memory_hybrid::seq_rm(seq_id, p0, p1);
}

// the sequence dst becomes a copy of src, like the plain unified cache (shared cells) as far as the data goes: the complete chunk pages
// (fitted as a whole, nothing writes them any more) are shared by reference, the last page (the chunk that is being filled or fitted) is
// copied, the ring cells and the recurrent cell are copied, the fit state of the open chunk too; the resident set of dst starts empty
// (as after the end of a prompt)
// the range is the whole sequence: [0, -1) or [0, p1) with p1 beyond its last position
void llama_memory_hybrid_kvq::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (!seq_cp_impl(seq_id_src, seq_id_dst, p0, p1, false)) {
        // not reached: without try_pages every failure aborts inside
        GGML_ABORT("seq_cp of sequence %d to %d failed", seq_id_src, seq_id_dst);
    }
}

// all or nothing: false, with the memory unchanged, if the pool has no pages for the copy (the caller can purge a sequence and ask again,
// or report an error)
bool llama_memory_hybrid_kvq::seq_cp_try(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    return seq_cp_impl(seq_id_src, seq_id_dst, 0, -1, true);
}

bool llama_memory_hybrid_kvq::seq_cp_impl(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1, bool try_pages) {
    if (!fitter) {
        if (params.enabled || store) {
            GGML_ABORT("seq_cp is not supported by the two-tier KV cache (--kvq) without the fit mode");
        }
        llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);
        return true;
    }
    if (seq_id_src < 0 || seq_id_src >= store->n_seq() || seq_id_dst < 0 || seq_id_dst >= store->n_seq()) {
        GGML_ABORT("seq_cp of sequence %d to %d: the two-tier KV cache (--kvq) has %d sequences (n_seq_max)", seq_id_src, seq_id_dst, store->n_seq());
    }
    if (seq_id_src == seq_id_dst) {
        return true;
    }
    const llama_seq_id src = seq_id_src;
    const llama_seq_id dst = seq_id_dst;

    // no job in flight in src, its ring and the host store complete (the same steps as a state save)
    fitter->wait_jobs(src);
    tail_release(src);
    const llama_pos n_src = n_pos(src);
    if (p0 > 0 || (p1 >= 0 && p1 < n_src)) {
        GGML_ABORT("seq_cp of positions [%d, %d) is not supported by the two-tier KV cache (--kvq): it copies a whole sequence ([0, -1), or up to a position at or after its last one, here %d). Copy the whole sequence and remove the end with seq_rm, or run without --kvq.", p0, p1, n_src - 1);
    }
    if (n_src > 0 && fitter->has_failed(src)) {
        GGML_ABORT("seq_cp of sequence %d: its fit failed, the sequence cannot be copied", src);
    }

    const int32_t n_share = fitter->n_shared(src);
    if (try_pages && n_src > 0 && store->n_page_chunks(src) - n_share > store->get_pool().n_free()) {
        // the pages dst holds are not counted (the caller empties dst first): nothing was changed
        return false;
    }

    const int64_t t0 = ggml_time_us();
    seq_clear(dst);
    if (n_src <= 0) {
        return true;
    }
    store->sync_ring(src, n_src);
    store->synchronize();

    // the pages first (it is the step that can fail, with nothing taken): the host memory cap, or or the pool
    if (!store->fork(src, dst, n_share)) {
        seq_clear(dst);
        if (try_pages) {
            return false;
        }
        GGML_ABORT("seq_cp of sequence %d to %d: the page pool of the two-tier cache has no page for the copy of the last page of the sequence (%d of %d pages free; --kvq-host-max-mib?). llama-server and other callers that can fail should call llama_memory_kvq_seq_cp_try (src/llama-ext.h), which returns false and changes nothing.", src, dst, store->get_pool().n_free(), store->get_pool().n_pages());
    }

    // the ring cells and the recurrent cell (the data of the ring: here, the plain copy of the stream is deferred to the next update)
    llama_memory_hybrid::seq_cp(src, dst, -1, -1);
    if (mem_swa) {
        mem_swa->seq_cp(src, dst, -1, -1);
    }
    fitter->fork(src, dst);
    seqs[dst].n_released = seqs[src].n_released;
    store->ring_fork(src, dst, seqs[src].n_released, n_src);
    store->ring_loaded(dst, n_src);
    // the decoding after a copy depends on the memory state, as after the end of a prompt
    store->reset_resident(dst);
    LLAMA_LOG_INFO("%s: sequence %d continues from sequence %d: %d positions, %d complete pages shared, %d pages held in all, %d shared, %.3f s\n", __func__,
            dst, src, n_src, n_share, store->get_pool().n_pages() - store->get_pool().n_free(), store->get_pool().n_shared(), (ggml_time_us() - t0)*1e-6);
    return true;
}

void llama_memory_hybrid_kvq::seq_keep(llama_seq_id seq_id) {
    if (fitter) {
        // the other sequences are emptied: their pages, fitter, store, ring and recurrent state
        for (llama_seq_id s = 0; s < store->n_seq(); ++s) {
            if (s != seq_id) {
                seq_rm(s, -1, -1);
            }
        }
        if (seq_id >= 0 && seq_id < store->n_seq()) {
            llama_memory_hybrid::seq_keep(seq_id);
            if (mem_swa) {
                mem_swa->seq_keep(seq_id);
            }
        }
        return;
    }
    if ((params.enabled || store) && seq_id != 0) {
        GGML_ABORT("seq_keep of sequence %d is not supported by the two-tier KV cache (--kvq) without the fit mode", seq_id);
    }

    llama_memory_hybrid::seq_keep(seq_id);
    if (mem_swa) {
        mem_swa->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_kvq::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (params.enabled) {
        GGML_ABORT("seq_add is not supported by the two-tier KV cache (--kvq): qwen35 uses IMROPE, its keys cannot be shifted in any KV cache (llama_memory_can_shift is false without --kvq too). Disable the context shift and --cache-reuse.");
    }

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_kvq::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (params.enabled) {
        GGML_ABORT("seq_div is not supported by the two-tier KV cache (--kvq): qwen35 uses IMROPE, its keys cannot be shifted in any KV cache (llama_memory_can_shift is false without --kvq too). Disable the context shift (self-extend).");
    }

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_kvq::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_swa) {
        for (const auto & buft_size : mem_swa->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    if (store) {
        for (const auto & buft_size : store->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

llama_pos llama_memory_hybrid_kvq::seq_pos_max(llama_seq_id seq_id) const {
    const llama_pos res = llama_memory_hybrid::seq_pos_max(seq_id);
    if (store && seq_id >= 0 && seq_id < store->n_seq() && res < 0 && store->n_host(seq_id) > 0) {
        return std::min<llama_pos>(store->n_host(seq_id) - 1, get_mem_recr()->seq_pos_max(seq_id));
    }
    return res;
}

// state of the fit mode, one sequence (seq_id -1 is sequence 0 with one sequence, with several an id is needed):
//   magic, version, then six sections, each as (u64 byte size, data):
//     store meta, fitter meta (layout, bits, rem policy, min rows; checked before any data is read)
//     ring: the plain state of the attention cache, the positions after the rows of the host store
//     recurrent: the plain state
//     store data: host rows [0, n_host), codes and tables of the coarse region [0, n_coarse) (it can end inside a chunk)
//     fitter data: the fitted range of each chunk slot, statistics of the complete chunks, sums of the open chunk
//   a reader checks that each section consumed exactly its size and names the section otherwise
// a plain state starts with the stream count of the attention cache, so neither reader takes the state of the other
// version 2: the coarse region ends at any row, the host store can hold rows after it
// version 3: a size per section; rem policy and min rows in the fitter meta
static constexpr uint32_t KVQ_STATE_MAGIC   = 0x5351564b; // "KVQS"
static constexpr uint32_t KVQ_STATE_VERSION = 3;

// counts the bytes a section will write; the sizes are written before the data, the writers cannot seek back
namespace {
class kvq_io_count : public llama_io_write_i {
public:
    void write(const void *, size_t size) override { n += size; }
    void write_tensor(ggml_tensor *, size_t, size_t size) override { n += size; }
    size_t n_bytes() override { return n; }
private:
    size_t n = 0;
};

// the section is written twice: counted first (no copies), then for real, and the real size is checked against the count
template <typename F>
void kvq_section_write(llama_io_write_i & io, const char * name, F && f) {
    kvq_io_count c;
    f(c);
    const uint64_t size = c.n_bytes();
    io.write(&size, sizeof(size));
    const size_t b0 = io.n_bytes();
    f(io);
    if (io.n_bytes() - b0 != size) {
        throw std::runtime_error(format("kvq state section '%s' wrote %zu bytes, counted %zu", name, io.n_bytes() - b0, (size_t) size));
    }
}

template <typename F>
void kvq_section_read(llama_io_read_i & io, const char * name, F && f) {
    uint64_t size = 0;
    io.read(&size, sizeof(size));
    const size_t b0 = io.n_bytes();
    f();
    const size_t got = io.n_bytes() - b0;
    if (got != size) {
        throw std::runtime_error(format("kvq state section '%s' holds %zu bytes, this build read %zu (saved by another build of the two-tier cache; process the prompt again)",
                name, (size_t) size, got));
    }
}
} // namespace

void llama_memory_hybrid_kvq::state_check(llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (!fitter) {
        throw std::runtime_error("the K/V state of the two-tier cache can be saved and loaded only in the fit mode (--kvq)");
    }
    if (seq_id >= store->n_seq()) {
        throw std::runtime_error(format("the two-tier cache has %d sequences, not %d", store->n_seq(), seq_id));
    }
    if (seq_id < 0 && store->n_seq() > 1) {
        throw std::runtime_error("the K/V state of the two-tier cache with several sequences needs a sequence id");
    }
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        throw std::runtime_error("the K/V state of the two-tier cache cannot be kept on the device");
    }
}

void llama_memory_hybrid_kvq::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) || (!params.enabled && !store)) {
        llama_memory_hybrid::state_write(io, seq_id, flags);
        return;
    }
    state_check(seq_id, flags);
    const llama_seq_id seq = std::max<llama_seq_id>(seq_id, 0);

    // no job in flight in a saved state; the state is not fitted further
    const int64_t t0 = ggml_time_us();
    fitter->wait_jobs(seq);
    const_cast<llama_memory_hybrid_kvq *>(this)->tail_release(seq);
    const double t_wait = (ggml_time_us() - t0)*1e-6;
    if (t_wait >= 0.001) {
        LLAMA_LOG_INFO("%s: waited %.3f s for the fitter, the coarse region has %d rows\n", __func__, t_wait, store->n_coarse(seq));
    }

    io.write(&KVQ_STATE_MAGIC,   sizeof(KVQ_STATE_MAGIC));
    io.write(&KVQ_STATE_VERSION, sizeof(KVQ_STATE_VERSION));
    kvq_section_write(io, "store meta",  [&](llama_io_write_i & w) { store->state_write_meta(seq, w); });
    kvq_section_write(io, "fitter meta", [&](llama_io_write_i & w) { fitter->state_write_meta(seq, w); });

    // host ring: the ring cells hold the rows after the host store, the readbacks of the last reply are done
    store->sync_ring(seq, n_pos(seq));
    store->synchronize();
    kvq_section_write(io, "ring",      [&](llama_io_write_i & w) { get_mem_attn()->state_write(w, seq, flags); });
    kvq_section_write(io, "recurrent", [&](llama_io_write_i & w) { get_mem_recr()->state_write(w, seq_id, flags); });
    if (mem_swa) {
        kvq_section_write(io, "swa",   [&](llama_io_write_i & w) { mem_swa->state_write(w, seq, flags); });
    }

    kvq_section_write(io, "store data",  [&](llama_io_write_i & w) { store->state_write_data(seq, w); });
    kvq_section_write(io, "fitter data", [&](llama_io_write_i & w) { fitter->state_write_data(seq, w); });
}

void llama_memory_hybrid_kvq::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) || (!params.enabled && !store)) {
        llama_memory_hybrid::state_read(io, seq_id, flags);
        return;
    }
    state_check(seq_id, flags);
    const llama_seq_id seq = std::max<llama_seq_id>(seq_id, 0);

    // the sequence starts empty and stays empty after a failed read: cancels its jobs, drops its coarse region
    seq_clear(seq);

    try {
        uint32_t magic   = 0;
        uint32_t version = 0;
        io.read(&magic, sizeof(magic));
        if (magic != KVQ_STATE_MAGIC) {
            throw std::runtime_error("not a state of the two-tier cache (saved without --kvq?)");
        }
        io.read(&version, sizeof(version));
        if (version != KVQ_STATE_VERSION) {
            throw std::runtime_error(format("version %u of the two-tier cache state, this build reads version %u "
                    "(saved by another build of the two-tier cache; process the prompt again)", version, KVQ_STATE_VERSION));
        }
        int32_t n = 0;
        int32_t h = 0;
        kvq_section_read(io, "store meta",  [&]() { store->state_read_meta(seq, io, n, h); });
        kvq_section_read(io, "fitter meta", [&]() { fitter->state_read_meta(io); });

        // host ring: no readback of an earlier reply may land in the ring cells after they are read
        store->synchronize();
        kvq_section_read(io, "ring",      [&]() { get_mem_attn()->state_read(io, seq, flags); });
        kvq_section_read(io, "recurrent", [&]() { get_mem_recr()->state_read(io, seq_id, flags); });
        if (mem_swa) {
            kvq_section_read(io, "swa",   [&]() { mem_swa->state_read(io, seq, flags); });
        }

        const llama_pos p_min = get_mem_attn()->seq_pos_min(seq);
        const llama_pos p_max = get_mem_attn()->seq_pos_max(seq);
        if (p_max >= 0 && (p_min != h || p_max >= (llama_pos) n_ctx_max)) {
            throw std::runtime_error(format("the ring holds positions [%d, %d], expected the positions after the %d rows of the host store, below %u", p_min, p_max, h, n_ctx_max));
        }

        // the pages of the rows of the state
        if (!store->reserve(seq, h)) {
            throw std::runtime_error("no pages for the rows of the state: the page pool of the two-tier cache is full");
        }
        kvq_section_read(io, "store data",  [&]() { store->state_read_data(seq, io, n, h); });
        kvq_section_read(io, "fitter data", [&]() { fitter->state_read_data(seq, io); });
        seqs[seq].n_released = h;
        store->ring_loaded(seq, p_max >= 0 ? p_max + 1 : h);

        // the resident slots start empty (one sequence: all the parameters again, as before; several: the slots of this sequence only)
        if (store->n_seq() == 1) {
            if (!store->enable(params)) {
                throw std::runtime_error("failed to reset the resident rows");
            }
        } else {
            store->reset_resident(seq);
        }
        LLAMA_LOG_INFO("%s: coarse region of %d rows, %d rows in the host store, %d positions in the ring\n", __func__,
                n, h, p_max >= 0 ? p_max - p_min + 1 : 0);
    } catch (...) {
        seq_clear(seq);
        throw;
    }
}

bool llama_memory_hybrid_kvq::set_params(const llama_kvq_params & p) {
    if (p.enabled) {
        if (p.n_sink < 0) {
            LLAMA_LOG_ERROR("%s: bad parameters\n", __func__);
            return false;
        }
        if (!store || !store->enable(p)) {
            LLAMA_LOG_ERROR("%s: packed mode needs llama_memory_kvq_fit_enable and matching parameters\n", __func__);
            return false;
        }
    }

    params = p;

    LLAMA_LOG_INFO("%s: enabled = %d, n_sink = %d\n", __func__, params.enabled, params.n_sink);

    return true;
}

bool llama_memory_hybrid_kvq::fit_enable(const llama_kvq_fit_params & fp, const llama_kvq_params & p) {
    if (!p.enabled) {
        LLAMA_LOG_ERROR("%s: the fit mode needs enabled parameters\n", __func__);
        return false;
    }
    const llama_kv_cache * kv = get_mem_attn();
    bool empty = kv->get_n_stream() == 1 || kv->get_n_stream() == kv->get_n_seq_max();
    for (uint32_t s = 0; s < kv->get_n_seq_max(); ++s) {
        empty = empty && kv->seq_pos_max(s) < 0;
    }
    if (!empty) {
        LLAMA_LOG_ERROR("%s: the fit mode needs an empty memory with one KV stream or one per sequence\n", __func__);
        return false;
    }
    if (p.prefill_free_codes && kv->get_n_seq_max() > 1) {
        LLAMA_LOG_ERROR("%s: --kvq-prefill-free-codes needs n_seq_max = 1 (got %u)\n", __func__, kv->get_n_seq_max());
        return false;
    }
    if (fp.rem == LLAMA_KVQ_REM_MERGE && kv->get_n_seq_max() > 1) {
        LLAMA_LOG_ERROR("%s: --kvq-rem merge needs n_seq_max = 1 (got %u): a merged group rewrites the tables of the last complete chunk, and the complete chunk pages that sequences share (seq_cp) must never change; use --kvq-rem own\n", __func__, kv->get_n_seq_max());
        return false;
    }
    // the smallest coarse region: the prefetch selects prefetch_n rows that are not sinks
    const int32_t min_first = p.prefetch_n + p.n_sink;
    if (fp.chunk <= p.n_sink || fp.chunk < min_first || fp.min_rows < 1 || (int64_t) n_ctx_max < fp.chunk ||
            (int64_t) kv->get_size() < 2*p.max_batch + 256) {
        LLAMA_LOG_ERROR("%s: bad chunk (%d, %u positions, cache of %u cells) or min_rows (%d)\n", __func__, fp.chunk, n_ctx_max, kv->get_size(), fp.min_rows);
        return false;
    }
    if (backends.empty()) {
        LLAMA_LOG_ERROR("%s: the memory has no backends\n", __func__);
        return false;
    }

    // V is fitted per chunk with the o_proj weight and the gate statistics
    auto order = model.kvq_layer_order();
    const auto kv_ids = kv->get_layer_ids();
    if (order.empty()) {
        order.assign(kv_ids.begin(), kv_ids.end());
    }
    std::vector<const ggml_tensor *> wo;
    int32_t n_head = 0;
    for (int32_t il : order) {
        const auto & layer = model.layers[il];
        if (layer.wo == nullptr || layer.wo_s != nullptr || model.hdmd.rot.count(layer.wo) != 0 || (n_head != 0 && n_head != (int32_t) model.hparams.n_head(il))) {
            LLAMA_LOG_ERROR("%s: per-chunk V does not support the o_proj weight of layer %d\n", __func__, il);
            return false;
        }
        wo.push_back(layer.wo);
        n_head = model.hparams.n_head(il);
    }

    params.enabled = false;
    fitter.reset();
    store.reset();
    try {
        const int64_t D      = (int64_t) model.hparams.n_embd_head_k(order[0]);
        const int32_t n_rows = (int32_t) n_ctx_max;
        const int32_t n_gate = n_head*(int32_t) D;
        // the fitter works in the shared memory of the store
        const int32_t G      = (int32_t) model.hparams.n_head_kv(order[0]);
        const size_t  sh_min = llama_kvq_fitter::shared_min(D, G, llama_kvq_fitter::stage_rows(n_rows, fp), fp.bits_k, fp.bits_v);
        store = std::make_unique<llama_kvq_store>(*kv, D, n_rows, fp.chunk, fp.bits_k, fp.bits_v, llama_kvq_resident_slots(p), p.n_sink, p.layer_bufs, n_gate, sh_min,
                p.host_prealloc, (size_t) std::max(0, p.host_max_mib)*1024*1024, (size_t) std::max(0, p.host_keep_mib)*1024*1024, order);
        store->set_backend(backends);
        if (!store->scratch_ok()) {
            throw std::runtime_error("the two-tier cache needs the scratch attention, which reads the host store with async copies, so the host store must be pinned memory (it is not)");
        }
        fitter = std::make_unique<llama_kvq_fitter>(*store, fp, wo, n_head, min_first);
        // the graph does not read the ring cells, see llama_kvq_store (host ring)
        get_mem_attn()->move_bufs(store->host_buft());
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        fitter.reset();
        store.reset();
        return false;
    }
    if (!set_params(p)) {
        fitter.reset();
        store.reset();
        return false;
    }
    seqs.assign(store->n_seq(), seq_state());
    fold_every  = getenv("LLAMA_KVQ_FOLD_EVERY") ? std::max(0, atoi(getenv("LLAMA_KVQ_FOLD_EVERY"))) : 0;
    if (fold_every > 0) {
        LLAMA_LOG_WARN("%s: LLAMA_KVQ_FOLD_EVERY = %d: the reply rows are fitted in line when %d of them follow the coarse region\n", __func__, fold_every, fold_every);
    }
    return true;
}

int32_t llama_memory_hybrid_kvq::fit_graph_kind() const {
    if (!params.enabled || !store) {
        return 0;
    }
    return store->get_gacc(0, store->layer_ids()[0]) ? 2 : 1;
}

void llama_memory_hybrid_kvq::fit_poll(llama_seq_id seq, uint32_t n_tokens) {
    if (fitter) {
        fitter->poll(seq, *get_mem_attn(), n_pos(seq), n_tokens > (uint32_t) params.max_batch);
        tail_release(seq);
    }
}

bool llama_memory_hybrid_kvq::fit_wait(llama_seq_id seq) {
    if (!fitter || seq < 0 || seq >= store->n_seq()) {
        return false;
    }
    const int64_t t0 = ggml_time_us();
    const int32_t c0 = store->n_coarse(seq);
    const bool ok = settle(seq, n_pos(seq), true);
    if (store->n_coarse(seq) != c0) {
        seqs[seq].n_settle++;
        seqs[seq].t_settle += (ggml_time_us() - t0)*1e-6;
    }
    return ok;
}

bool llama_memory_hybrid_kvq::prompt_bufs_alloc() {
    return !store || !store->scratch_ok() || store->scr_alloc();
}


bool llama_memory_hybrid_kvq::fit_get_progress(llama_seq_id seq, llama_kvq_fit_progress & out) const {
    if (!fitter || seq < 0 || seq >= store->n_seq()) {
        return false;
    }
    fitter->get_progress(seq, out);
    out.n_tail_wait = seqs[seq].n_tail_wait;
    out.t_tail_wait = seqs[seq].t_tail_wait;
    out.n_fold      = seqs[seq].n_fold;
    out.t_fold      = seqs[seq].t_fold;
    out.n_settle    = seqs[seq].n_settle;
    out.t_settle    = seqs[seq].t_settle;
    return true;
}

bool llama_memory_hybrid_kvq::fit_set_hold(bool hold) {
    if (!fitter) {
        return false;
    }
    fitter->set_hold(hold);
    return true;
}

void llama_memory_hybrid_kvq::set_backends(const std::vector<ggml_backend_t> & b) {
    backends = b;
    if (store) {
        store->set_backend(backends);
    }
}

bool llama_memory_hybrid_kvq::sched_copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return store && store->copy(backend, src, dst, graph);
}

bool llama_memory_kvq_fit_enable(llama_memory_t mem, const llama_kvq_fit_params & fparams, const llama_kvq_params & params) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_enable(fparams, params);
}

bool llama_memory_kvq_fit_get_progress(llama_memory_t mem, llama_seq_id seq_id, llama_kvq_fit_progress & out) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_get_progress(seq_id, out);
}

bool llama_memory_kvq_fit_wait(llama_memory_t mem, llama_seq_id seq_id) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_wait(seq_id);
}

bool llama_memory_kvq_fit_set_hold(llama_memory_t mem, bool hold) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    return kvq && kvq->fit_set_hold(hold);
}

bool llama_memory_kvq_get_counters(llama_memory_t mem, llama_kvq_counters_all & out) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (!kvq || !kvq->get_store()) {
        return false;
    }
    kvq->get_store()->get_counters(out);
    return true;
}

bool llama_memory_kvq_get_memory_info(llama_memory_t mem, llama_kvq_memory_info & out) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (!kvq || !kvq->get_store()) {
        return false;
    }
    out = kvq->get_store()->memory_info();
    return true;
}

bool llama_memory_kvq_seq_cp_try(llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (!kvq || !kvq->get_store()) {
        return false;
    }
    return kvq->seq_cp_try(seq_id_src, seq_id_dst);
}

void llama_memory_kvq_reset_counters(llama_memory_t mem) {
    auto * kvq = dynamic_cast<llama_memory_hybrid_kvq *>(mem);
    if (kvq && kvq->get_store()) {
        kvq->get_store()->reset_counters();
    }
}

//
// llama_memory_hybrid_kvq_context
//

// the reserve graph is not in two-tier mode, but sizes the scratch attention of the mode
static llama_kvq_params kvq_params_reserve(const llama_kvq_params & p) {
    llama_kvq_params res;
    res.scr_block = p.scr_block;
    return res;
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_status status) :
    llama_memory_hybrid_context(status), status(status) {}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem) :
    llama_memory_hybrid_context(mem),
    kv_attn(mem->get_mem_attn()),
    ctx_swa(mem->get_mem_swa() ? mem->get_mem_swa()->init_full() : nullptr),
    params(kvq_params_reserve(mem->get_params())),
    store(mem->fit_graph_kind() > 0 ? mem->get_store() : nullptr),
    status(llama_memory_hybrid_context::get_status()) {
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(llama_memory_hybrid_kvq * mem, bool packed_reserve) :
    llama_memory_hybrid_context(mem),
    kv_attn(mem->get_mem_attn()),
    ctx_swa(mem->get_mem_swa() ? mem->get_mem_swa()->init_full() : nullptr),
    params(mem->get_params()),
    is_packed_reserve(packed_reserve),
    store(mem->get_store()),
    status(llama_memory_hybrid_context::get_status()) {
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(
        llama_memory_hybrid_kvq * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    kv_attn(mem->get_mem_attn()),
    ctx_swa(mem->get_mem_swa() ? mem->get_mem_swa()->init_update(lctx, optimize) : nullptr),
    params(mem->get_params()),
    store(nullptr),
    status(llama_memory_hybrid_context::get_status()) {
}

llama_memory_hybrid_kvq_context::llama_memory_hybrid_kvq_context(
        llama_memory_hybrid_kvq * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_swa,
      std::vector<llama_ubatch>   ubatches) :
    llama_memory_hybrid_context(mem, sinfos_attn, ubatches),
    kv_attn(mem->get_mem_attn()),
    ctx_swa(mem->get_mem_swa() ? std::make_unique<llama_kv_cache_context>(mem->get_mem_swa(), std::move(sinfos_swa), ubatches) : nullptr),
    mem_kvq(mem),
    params(mem->get_params()),
    store(mem->get_params().enabled ? mem->get_store() : nullptr),
    status(llama_memory_hybrid_context::get_status()) {
}

const llama_kv_cache_context * llama_memory_hybrid_kvq_context::get_swa() const {
    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}

bool llama_memory_hybrid_kvq_context::next() {
    bool res = llama_memory_hybrid_context::next();
    if (ctx_swa) {
        const bool res_swa = ctx_swa->next();
        GGML_ASSERT(res == res_swa);
    }
    return res;
}

bool llama_memory_hybrid_kvq_context::apply() {
    // the previous ubatches are queued: the fit mode can start a job or move the coarse boundary now
    if (mem_kvq) {
        const llama_ubatch & ub = get_ubatch();
        for (uint32_t s = 0; s < ub.n_seqs_unq; ++s) {
            mem_kvq->fit_poll(ub.seq_id_unq[s], ub.n_tokens/ub.n_seqs_unq);
        }
    }

    bool res = llama_memory_hybrid_context::apply();
    if (ctx_swa) {
        res = res && ctx_swa->apply();
    }

    // the scratch attention of a prompt ubatch, the resident and tail buffers of a packed one
    if (res && mem_kvq && store) {
        if (prompt_path(get_ubatch())) {
            res = !store->scratch_ok() || store->scr_alloc();
        } else {
            res = store->dec_alloc();
        }
    }

    return res;
}

bool llama_memory_hybrid_kvq_context::codes_alloc(const llama_ubatch & ubatch) const {
    if (!mem_kvq || !store || prompt_path(ubatch)) {
        return true;
    }
    return store->codes_alloc();
}

bool llama_memory_hybrid_kvq_context::prompt_path(const llama_ubatch & ubatch) const {
    if (!params.enabled || store == nullptr) {
        return false;
    }
    // a ubatch of several sequences was made of the sequences that are on the packed path (init_batch), the others stay single
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq = ubatch.seq_id_unq[s];
        if (!llama_kvq_use_packed(params, ubatch.n_tokens/ubatch.n_seqs_unq, store->n_coarse(seq), store->n_host(seq))) {
            GGML_ASSERT(ubatch.n_seqs_unq == 1 && "the sequences of a packed ubatch must stay on the packed path");
            return true;
        }
    }
    return false;
}

llama_memory_status llama_memory_hybrid_kvq_context::get_status() const {
    return status;
}

void llama_memory_hybrid_kvq_context::set_input_kvq_ring(ggml_tensor * mask, ggml_tensor * early, const llama_ubatch * ubatch, const std::vector<int32_t> & n_coarse_s) const {
    if (mask == nullptr || mask->buffer == nullptr) {
        return;
    }
    // the sequences of the ubatch are the streams of the mask [n_coarse_max + n_kv, L, 1, S]; stream s: the coarse rows [0, n_coarse[s]),
    // then the ring cells (tail row r of the stream is cache row n_coarse[s] + r)
    const int64_t S = ubatch->n_seqs_unq;
    const int64_t L = ubatch->n_tokens/S;
    GGML_ASSERT((int64_t) n_coarse_s.size() == S && ggml_backend_buffer_is_host(mask->buffer));
    const int32_t n_coarse_max = *std::max_element(n_coarse_s.begin(), n_coarse_s.end());
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && mask->ne[0] > n_coarse_max && mask->ne[1] == L && mask->ne[3] == S);
    GGML_ASSERT(early == nullptr || (ggml_backend_buffer_is_host(early->buffer) && ggml_are_same_shape(early, mask) && ggml_is_contiguous(early)));

    const int64_t n_kv = mask->ne[0] - n_coarse_max;

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);

    for (int64_t st = 0; st < S; ++st) {
        const int32_t n_coarse = n_coarse_s[st];
        // the cells of the ring stream of the sequence
        const llama_seq_id seq = ubatch->seq_id_unq[st];
        const auto & cells = kv_attn->get_cells(seq);
        GGML_ASSERT(n_kv <= (int64_t) cells.size());

        const llama_pos * pos = ubatch->pos + st*L;
        llama_pos p_min = pos[0];
        for (int64_t i = 1; i < L; ++i) {
            p_min = std::min(p_min, pos[i]);
        }

        // the positions before the ubatch, and the cells at or after it
        std::vector<ggml_fp16_t> row(mask->ne[0], ninf);
        std::fill(row.begin(), row.begin() + n_coarse, zero);
        std::vector<std::pair<int64_t, llama_pos>> late;
        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                continue;
            }
            const llama_pos p = cells.pos_get(j);
            if (p < n_coarse) {
                continue;
            }
            if (p < p_min) {
                row[n_coarse + j] = zero;
            } else {
                late.push_back({ n_coarse + j, p });
            }
        }

        ggml_fp16_t * dst = (ggml_fp16_t *) mask->data + st*L*mask->ne[0];
        for (int64_t i = 0; i < L; ++i) {
            ggml_fp16_t * d = dst + i*mask->ne[0];
            memcpy(d, row.data(), row.size()*sizeof(ggml_fp16_t));
            for (const auto & c : late) {
                if (c.second <= pos[i]) {
                    d[c.first] = zero;
                }
            }
            if (early) {
                memcpy((ggml_fp16_t *) early->data + (st*L + i)*mask->ne[0], row.data(), row.size()*sizeof(ggml_fp16_t));
            }
        }
    }
}

void llama_memory_hybrid_kvq_context::set_input_kvq_ring_tail(ggml_tensor * mask, ggml_tensor * kpos, const llama_ubatch * ubatch, const std::vector<int32_t> & n_coarse_s) const {
    const int64_t S = ubatch->n_seqs_unq;
    const int64_t L = ubatch->n_tokens/S;
    GGML_ASSERT((int64_t) n_coarse_s.size() == S);

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);

    if (mask && mask->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(mask->buffer));
        GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && mask->ne[1] == L && mask->ne[3] == S);
        const int64_t n_kv = mask->ne[0];

        for (int64_t st = 0; st < S; ++st) {
            const int32_t n_coarse = n_coarse_s[st];
            const llama_seq_id seq = ubatch->seq_id_unq[st];
            const auto & cells = kv_attn->get_cells(seq);
            GGML_ASSERT(n_kv <= (int64_t) cells.size());

            const llama_pos * pos = ubatch->pos + st*L;
            llama_pos p_min = pos[0];
            for (int64_t i = 1; i < L; ++i) {
                p_min = std::min(p_min, pos[i]);
            }

            std::vector<ggml_fp16_t> row(n_kv, ninf);
            std::vector<std::pair<int64_t, llama_pos>> late;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                    continue;
                }
                const llama_pos p = cells.pos_get(j);
                if (p < n_coarse) {
                    continue;
                }
                if (p < p_min) {
                    row[j] = zero;
                } else {
                    late.push_back({ j, p });
                }
            }

            ggml_fp16_t * dst = (ggml_fp16_t *) mask->data + st*L*n_kv;
            for (int64_t i = 0; i < L; ++i) {
                ggml_fp16_t * d = dst + i*n_kv;
                memcpy(d, row.data(), n_kv*sizeof(ggml_fp16_t));
                for (const auto & c : late) {
                    if (c.second <= pos[i]) {
                        d[c.first] = zero;
                    }
                }
            }
        }
    }

    if (kpos && kpos->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(kpos->buffer));
        GGML_ASSERT(kpos->type == GGML_TYPE_I32 && ggml_is_contiguous(kpos) && kpos->ne[1] == S);
        const int64_t n_kv = kpos->ne[0];
        int32_t * dst = (int32_t *) kpos->data;
        for (int64_t st = 0; st < S; ++st) {
            const llama_seq_id seq = ubatch->seq_id_unq[st];
            const auto & cells = kv_attn->get_cells(seq);
            GGML_ASSERT(n_kv <= (int64_t) cells.size());
            for (int64_t j = 0; j < n_kv; ++j) {
                dst[st*n_kv + j] = (!cells.is_empty(j) && cells.seq_has(j, seq)) ? (int32_t) cells.pos_get(j) : -1;
            }
        }
    }
}

void llama_memory_hybrid_kvq_context::set_input_kvq_scr(ggml_tensor * mask, const llama_ubatch * ubatch, int32_t n_host, int64_t r0) const {
    if (mask == nullptr || mask->buffer == nullptr) {
        return;
    }
    GGML_ASSERT(ubatch->n_seqs_unq == 1 && ggml_backend_buffer_is_host(mask->buffer));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ggml_is_contiguous(mask) && r0 >= 0 && r0 <= n_host && r0 + mask->ne[0] > n_host && mask->ne[1] == ubatch->n_tokens);

    const llama_seq_id seq = ubatch->seq_id_unq[0];
    const auto & cells = kv_attn->get_cells(seq);

    // mask column c is scratch row r0 + c
    const int64_t n_cols = mask->ne[0];
    const int64_t n_rows = r0 + n_cols;
    const int64_t size   = cells.size();
    GGML_ASSERT(n_rows - n_host <= size);

    llama_pos p_min = ubatch->pos[0];
    for (uint32_t i = 1; i < ubatch->n_tokens; ++i) {
        p_min = std::min(p_min, ubatch->pos[i]);
    }

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);

    auto has = [&](int64_t r) {
        const int64_t j = r % size;
        return !cells.is_empty(j) && cells.seq_has(j, seq) && cells.pos_get(j) == r;
    };

    // the rows of the positions before the ubatch, then per token the positions of the ubatch up to its own
    std::vector<ggml_fp16_t> row(n_cols, ninf);
    std::fill(row.begin(), row.begin() + (n_host - r0), zero);
    for (int64_t r = n_host; r < std::min<int64_t>(p_min, n_rows); ++r) {
        if (has(r)) {
            row[r - r0] = zero;
        }
    }
    std::vector<int64_t> late;
    for (int64_t r = std::max<int64_t>(n_host, p_min); r < n_rows; ++r) {
        if (has(r)) {
            late.push_back(r);
        }
    }

    ggml_fp16_t * dst = (ggml_fp16_t *) mask->data;
    for (int64_t i = 0; i < mask->ne[1]; ++i) {
        ggml_fp16_t * d = dst + i*n_cols;
        memcpy(d, row.data(), n_cols*sizeof(ggml_fp16_t));
        for (const int64_t r : late) {
            if (r > ubatch->pos[i]) {
                break;
            }
            d[r - r0] = zero;
        }
    }
}

void llama_memory_hybrid_kvq_context::set_input_kvq_scr_pos(ggml_tensor * kpos, const llama_ubatch * ubatch, int32_t n_host, int64_t r0) const {
    if (kpos == nullptr || kpos->buffer == nullptr) {
        return;
    }
    GGML_ASSERT(ubatch->n_seqs_unq == 1 && kpos->type == GGML_TYPE_I32 && ggml_is_contiguous(kpos));
    GGML_ASSERT(r0 >= 0 && r0 <= n_host && r0 + kpos->ne[0] > n_host);

    const llama_seq_id seq = ubatch->seq_id_unq[0];
    const auto & cells = kv_attn->get_cells(seq);

    const int64_t n    = kpos->ne[0];
    const int64_t size = cells.size();
    GGML_ASSERT(r0 + n - n_host <= size);

    std::vector<int32_t> pos(n);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = r0 + i;
        if (r < n_host) {
            pos[i] = (int32_t) r;
        } else {
            const int64_t j = r % size;
            pos[i] = (!cells.is_empty(j) && cells.seq_has(j, seq) && cells.pos_get(j) == r) ? (int32_t) r : INT32_MAX;
        }
    }
    ggml_backend_tensor_set(kpos, pos.data(), 0, n*sizeof(int32_t));
}
