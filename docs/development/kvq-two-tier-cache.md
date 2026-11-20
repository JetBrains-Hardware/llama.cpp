# Two-tier KV cache for qwen35 (kvq): implementation notes, knobs, measurements, gaps

State: branch `erokhins/kvq` of ours (the JetBrains-Hardware fork of ggml-org/llama.cpp), HEAD d7887c1ad, 2026-11-20. The branch is ggml-org/llama.cpp master plus the GDN replay of ggml-org PR 31669 (merged locally at 4218b5b9f) plus the kvq series. This file is the single document of the feature: what it is and how the code works (sections 1 - 4), every setting (section 5), how to build and run (section 6), all measurements with their build and date (section 7), the decisions and the history behind them (section 8), and what is not implemented, with the status of each item against the code (section 9). It replaces the former `kvq-two-tier-cache.md` (prototype state) and `kvq-two-tier-cache-guide.md` (user guide).

Open item while this was written: step 3c of the multi-sequence work (the device memory for packed decoding reserved at load) is being implemented in parallel and is not in HEAD; section 3.7 describes it from its specification and carries a TODO for its numbers.

Contents: 1 What it is. 2 How it works (one sequence). 3 Several sequences. 4 Code map. 5 Settings. 6 Build and run. 7 Measurements. 8 Decisions and history. 9 Not implemented, known gaps, ideas. 10 Material outside the repository.

## 1. What it is

A KV cache for Qwen3.8-27B (GGUF arch `qwen35`: 64 trunk blocks, 16 attention layers il = 3, 7, ..., 63, 48 gated-delta-net layers, 24 query heads, 4 kv heads, head dim 256) that keeps most of the attention K/V out of VRAM. The model facts are checked at load (`llama_model_supports_kvq`: arch qwen35, K and V head size 256 in every attention layer, equal head counts in all of them, a whole number of query heads per kv head). The MTP block (block 64) runs in its own context with its own plain cache and uses none of this.

| Tier | Where | What |
|---|---|---|
| Coarse | VRAM (`kc`, `vc`, tables, pages) | every row of a fitted chunk as transform-coded low-bit codes: K 2 bit, V 1 bit plus a gain. The packed attention op computes on the codes. Also a pinned host copy of the codes and tables |
| Precise | pinned host RAM (the store, pages) | the q8_0 rows of the same positions |
| Resident set | VRAM, per (layer, kv head), mirrored in pinned host memory | 8192 exact rows selected per position by an early query (top 1024), fetched from host RAM while the GPU runs the layers before the attention |
| Ring | pinned host RAM, rotating device tail buffers | q8_0 rows of the positions whose rows are not in the host store yet: the reply since the end of the last prompt, and the rows of a prompt while it is processed |

Phases: prefill (a prompt, ubatches larger than 8 tokens) attends over exact rows only, streamed from the host store into a device scratch, in mask-free key blocks merged by a log-sum-exp output; the codes are not read during a prompt. Decode and MTP / DFlash verify (ubatches of up to 8 tokens) use the packed attention.

What it buys: `llama-server --kvq` at 250k context with the DFlash2 drafter fits 15.5 GiB (15,872 MiB) of dedicated VRAM in every phase (section 7.1), and with several slots (`-np N`) the slots share one pool of pages and decode together in one packed ubatch (section 3, 7.9).

Platform: CUDA only. The attention op, the fit op and the LSE flash attention have CPU references for tests and CUDA kernels for running; the other backends refuse the kvq ops and the flash attention with the LSE output in `supports_op` (the meta backend, i.e. tensor split, too), and `--kvq` refuses to start on a model or device that cannot run them.

## 2. How it works (one sequence)

### 2.1 Chunks and the three boundaries

The sequence is cut in chunks of C positions (`--kvq-chunk`, default 8192): chunk c covers [c*C, (c+1)*C). A complete chunk is fitted alone: a K table and a V table per (layer, kv head) on the chunk's own rows; its rows are coded and join the coarse region. A fit is always made from scratch from its rows (q8_0, from the host store) and its statistics, so its tables do not depend on how often or when the range was fitted before.

The rows after the last complete chunk (`--kvq-rem`). At the end of every prompt (`llama_memory_kvq_fit_wait(mem, seq_id)`, the server calls it when the prompt of a slot is done) the fitter fits the layout of the token count S, waits for all of it, and nothing of the fit runs afterwards. With n_full = floor(S/C):

- `own` (default): chunks 0 .. n_full - 1 alone, then [n_full*C, S) as a chunk of its own if it has at least `--kvq-min-rows` rows (default 256); fewer rows stay precise in the ring.
- `merge`: chunks 0 .. n_full - 2 alone, then [(n_full - 1)*C, S) as one group (C to 2C - 1 rows) with one table set written to the table slots of both chunks (the op takes the table of a row by its chunk, so no kernel change). When the next chunk is complete, the head of the group is fitted alone again.
- n_full = 0: [0, S) as one partial chunk, both policies. A coarse region smaller than prefetch_n + n_sink (1028) is not made (the prefetch selects from its rows); those rows stay precise.

At the next end of a prompt the partial chunk or group is fitted again with all rows it has then, from the host store. The rows of a reply stay precise in the ring until the next end of a prompt. The resident set is emptied at the end of every prompt, so the decoding after it depends only on the memory state (also after a slot restore).

Three boundaries per sequence. n_coarse: positions [0, n_coarse) are coded. n_host >= n_coarse: positions [0, n_host) have their q8_0 rows in the host store. The ring holds the positions [n_host, end), position p in cell p mod size; its cells are freed as soon as a job has started on their rows. The packed attention (small batches) reads coarse rows + resident rows + the ring, so it needs n_host == n_coarse; otherwise a small batch attends over the scratch like the prefill (correct, slower; only after a rollback or a spill before the next end of a prompt).

### 2.2 Tables and codec

Per (page, layer, kv head, K or V) a 256 x 256 inverse table and a small aux array (step, mu, bits), stored in one i8 tensor per layer, `tab [D*D + 3*D*4, 2G, n_pages]` (a slot is the D*D inverse table bytes, then the 3*D aux floats). The inverse table is int8 with one scale per component (row divided by its max abs, times 127); the scale is folded into the step the op reads, so the decode formula is unchanged. The codec (fit, encode, bits) works in float. K uses 2 bits, V 1 bit plus a gain (`llama_kvq_fit_params`: bits_k 2, bits_v 1); the first n_sink = 4 rows are always resident.

### 2.3 Fitter (`llama_kvq_fitter`)

A job is a row range [r0, r1) of one sequence with one table set (a chunk, a partial chunk or a group). The whole fit runs on the GPU (since 9f9d1e082, 2026-11-20), on the main stream between two ubatches, in the shared memory of the store (the scratch / resident / tail region, idle between the ubatches: `llama_kvq_store::fit_alloc`; the next prompt ubatch zeroes the scratch again, the next packed ubatch restores the resident buffers). The fitter places a stage (q8_0, C rows; 2C - 1 with merge), a code stage and the tables of a group of layers there; a small liveness planner (`llama_kvq_fitter::plan`) places the intermediates of each graph after them. The attention layers go in groups (as many as fit: 10 at 24k, all 16 at 250k; lower bound `llama_kvq_fitter::shared_min`, about 134 MB own, 255 MB merge):

1. The metrics on the host from the statistics (K: the queries of the range; V: the o_proj columns scaled by the mean sigmoid(gate)), into the tables of the group.
2. Per layer: rows [r0, r1) from the host store into the stage, mean and covariance (`ggml_kvq_fit`).
3. One graph per group (`ggml_kvq_fit_table`, 8 tables per layer): Cholesky of the metric in double, `L^T cov L` (cuBLAS, double), the eigen solver (the tred2 / tql2 of the CPU codec in float: tridiagonal form on one block per table, the QL steps on one thread, their rotations replayed by 32-row blocks), fwd = L U and inv = U^T L^-1 (cuBLAS). The result row of each table (status, ridge) goes to the host.
4. Per layer: the rows again, K: bits and steps from sigma (`ggml_kvq_fit_bits`); V: rows rotated, the scale of the shape, bits; the packed tables (`ggml_kvq_fit_pack`) and the codes to the pinned host copies (and into `kc` / `vc` through the page table). A waiter thread waits for the job's event; the next poll installs it: the tables go to the host copy of every slot of the range and to the device copy; `set_n_coarse`. A range that is coarse already gets its new codes and tables in the same poll, so no graph sees a mix.

Not bit-identical to the CPU fit that preceded it (float eigen solver); the CPU fit and `--kvq-threads` are gone. Per layer the table graph is about 10 ms (tql2 3.1, tred2 3.2, replay 1.3, Cholesky 1.2, cuBLAS 1), latency bound (8 blocks on the GPU).

When jobs start. Prefill (ubatches larger than max_batch): a complete chunk is taken at the first ubatch after it is complete (merge: once the next chunk is complete too). Generation (small ubatches): no job starts; the jobs in flight of the sequence are waited for before its small batch, so the path of the small batch does not depend on the fitter. At the end of a prompt: the layout as above, synchronously.

Statistics. K metric: second moment of the queries summed on the device during decoding (`qacc`), per sequence; a ubatch counts for the chunk of its first position; when a chunk is complete its sums are read and kept (in the page, `llama_kvq_pool::sums`) and the device sums start again from 0. A group uses the kept sums of its complete chunk plus the device sums of the open chunk; an own partial chunk also takes the kept sums of the chunk before it (its own sums can cover very few positions and the metric would be indefinite). The device sums of the open chunk are read without a reset and kept with their position; a rollback to exactly that position restores them. V metric: o_proj columns scaled by the mean sigmoid(gate) (`gacc`), the same way.

Ring overflow (`tail_wait`, per sequence). A large batch: first the jobs of the complete chunks start (their ring cells are free at once), then the remaining ring rows before the batch are copied to the host store without a fit (spill). A small batch (generation): when p0 + max_batch - n_host > ring size, the rows after the last complete chunk are fitted in line on the decode thread (fold: the layout of p0, sync, then the batch); the position of a fold depends only on the first position of the batch and the state, so generation stays reproducible.

### 2.4 Generation and verify (ubatch <= 8 tokens), per attention layer il

1. At the input of layer il - lead (default 2) the graph computes layer il's query from the earlier hidden state, scores all coarse rows, and takes the top 1024 per (kv head, position).
2. A scheduler split with an early copy callback queues the readback of the selection (row indices only; the scores only when the request can be cut) on a copy stream. A worker thread plans the resident slots and gathers the rows from the host store. The GPU keeps running layers il - 2 and il - 1.
   - The request of a kv head is the union of the selections of all positions of the ubatch, deduplicated in a bitmap of n_coarse bits (the word scan gives the distinct rows in row order; and-not with the bitmap of the resident rows gives the missing rows).
   - A slot is free if its row is not in the request. The missing rows (in row order) go to the free slots of the oldest requests (a per-slot stamp of the last request that listed its row, 8 bits, 255 and older and empty slots equal), ties by slot. So the row of every slot depends only on the sets, not on the order the GPU wrote the selection or on thread timing.
   - Default capacity (`--kvq-resident-cap 0`) = max_batch x prefetch_n = 8192, the whole request of the largest packed ubatch. A request of more distinct rows than slots is cut to the rows with the largest max score over the positions (ties by row; the scores are read back for this case only; logged once at enable, counted as `pf_cut` / `pf_lost`).
3. Before layer il's attention a join uploads the fetched rows and scatters them into the resident tensor. By default (`--kvq-layer-bufs all`) every attention layer has its own device buffer in the shared memory; a prompt leaves them stale and the mirrors go back in once on the base stream. With `--kvq-layer-bufs N` (N >= 2) the layers rotate through N buffers and the worker uploads the whole pinned mirror of the next layer into the other buffer every round (about 17 MiB per layer, 272 MiB per batch, at cap 8192).
4. `ggml_kvq_attn` computes the attention over coarse rows + resident precise rows + the ring. The ring rows of the layer are uploaded into one of four rotating device tail buffers (occupied cells only; buffer = layer index mod 4, because the early query of layer i scores the ring K of layer i one layer before the resident rows are needed); the new rows of the ubatch are written into the tail buffer by the cpy nodes and read back into the host ring on the copy stream.

The op (`ggml_kvq_attn` and `ggml_kvq_attn_streams`, `ggml/src/ggml-cuda/kvq-attn.cu`) has ten sources: q, kc, vc, tab, k, v, rkv, ridx, mask, pt (GGML_MAX_SRC stays 10; ggml-rpc serializes `src[GGML_MAX_SRC]`). Modes ATTN (attention output) and SCORE (the coarse scores for the selection). `pt` is the page table: logical chunk t -> physical page pt[t]; rows, masks, scores and top-k stay logical. Kernels: `kvq_prep`, `kvq_prep_c`, `kvq_coarse`, `kvq_score`, `kvq_precise` / `kvq_precise_m` (resident rows in blocks of 256), `kvq_merge`, `kvq_combine`, `kvq_final`; the precise rows run on a side stream concurrently with the coarse rows. Summation orders are fixed (no atomics), so the op is deterministic.

### 2.5 Prefill

Per attention layer: the rows [0, n_host) are uploaded from the pinned host store into one device scratch buffer (async uploads on the copy stream, overlapping the layers before; per-chunk runs through the page table), the ring rows after them are copied in on the device, and flash attention runs over the scratch. The K/V of the ubatch go straight into the scratch rows of their positions and are read back into the host store at the next scratch upload, so the host store holds every row after a large ubatch and the fitter reads all its rows from it. With `--kvq-scr-block 0` it is one flash attention with an F16 mask over all scratch rows, bit-identical to attention over a full device cache. By default the host rows [0, r0), r0 = floor(n_host / 256) * 256, are attended in key blocks of `--kvq-scr-block` rows (16384) without a mask by `ggml_flash_attn_ext_lse`, which also returns the log-sum-exp l of each query and head; only the rest [r0, n_rows) (at most 255 host rows plus the ring rows) has a mask. The partial results are merged in graph order with standard ops: d = l_b - l, o += (o_b - o) * sigmoid(d), l += softplus(d). The F16 copies of K/V that the CUDA flash attention makes exist for one block at a time and the big mask is gone; the result differs from the single attention in rounding (settled in 7.7). The op is FLASH_ATTN_EXT with op param 5 = 1; the result has ceil(n_batch / DV) more rows in dim 2 that hold the F32 [n_head, n_batch] log-sum-exp. The GQA kernel variants of the MMA kernel load the mask only if present and are used without a mask for the LSE op. The number of key blocks follows the number of stored rows, so the graph is re-planned at almost every prompt ubatch (the buffer does not grow).

### 2.6 Rollback, state, GDN replay

Rollback (`seq_rm` of the end at p0). Inside the ring (MTP drafts): only the ring cells. Into the coarse region: the jobs of the sequence with rows at or after p0 are cancelled, the coarse region drops back to the chunk grid (n_coarse = floor(p0/C)*C), n_host = p0: the rows before p0 stay in the host store, nothing is copied back; the next end of a prompt fits them again (until then they are prefilled over the scratch). `clear` drops everything and the fitter starts again.

Save and restore (`llama_memory_hybrid_kvq::state_write` / `state_read`, per sequence; an id is needed with several sequences). The state is exact: nothing is fitted again on load. A save waits for the jobs in flight and fits nothing. Format (magic `KVQS`, version 3): six sections, each (u64 byte size, data): store meta (attention layers, kv heads, head dim, chunk rows, words per K / V row, n_coarse, type and slice size of the table tensors, n_host), fitter meta (K / V bits, the `rem` policy and `min_rows`), the ring (the plain state of the attention cache, the positions after n_host), the recurrent state, the store data (per attention layer the q8_0 host rows [0, n_host) per kv head, the codes of [0, n_coarse), the tables of the coarse slots, in logical page order), the fitter data (the fitted range of each slot, the statistics of the complete chunks, the device sums of the open chunk, the kept sums of the last partial fit and their position). The loader checks the meta before any data and that each section consumed exactly its size (the error names the section). Another version, or a state fitted with another `rem` / `min_rows`, is refused and the sequence is empty afterwards (`kvq-eval --kvq-state-refuse`). A plain state starts with the stream count of the attention cache, so a plain context refuses a kvq state ("n_stream mismatch") and a kvq context refuses a plain one ("not a state of the two-tier cache"). The resident set starts empty after a load. The server's slot file appends the context checkpoints and the MTP draft state; the RAM prompt cache serializes the same state (an entry over `--cache-ram` is skipped). A kvq state is about 29% larger than a plain one.

GDN rollback by replay (ggml-org PR 31669): the recurrent memory keeps the committed state plus the pending inputs of at most n_rs_seq + 1 tokens per cell, and the GDN op recomputes the state at a rollback; the n_rs_seq snapshot rows are not needed. `--kvq` turns it on (context param `rs_replay`; env `LLAMA_RS_REPLAY=0` keeps the snapshots). Cost: a context checkpoint, a slot file or a RAM cache entry grows by 27.1 MiB (up to 32 checkpoints = 868 MiB of RAM); the stored layout of the pending inputs is not final upstream, so slot files and checkpoints of this build may not load in the next one.

### 2.7 Compute buffers per phase and device memory sharing

Compute buffers per phase (`llama_context`, context param `phase_bufs`; `--kvq` and a DFlash drafter turn it on, `--[no-]phase-buffers` overrides; it needs one sequence, a device buffer and n_tokens_tg < n_ubatch, so it is off with `-np` > 1). `ggml_gallocr_resize` frees a compute buffer and allocates it again with a given size while the node plan stays; `sched_reserve` reserves the pp graph and a tg graph of n_tokens_tg tokens; per ubatch a tg ubatch (at most n_tokens_tg tokens and not on the prompt path, `llama_kvq_use_packed`) switches the device compute buffers through `phase_set`. With `--kvq` the reserve graph is the scratch attention, so the tg sizes start at 0 and the largest size reached is kept (1 - 3 graph reallocations in the first reply after load). The DFlash drafter's pp reserve is the injection graph (120 MiB) instead of a 1024-token graph with full-vocabulary logits (1,099 MiB); that no longer depends on the phase buffers (405c751da). The layer input buffer of the drafter is sized per batch (n_batch rows in a prompt, n_tokens_tg rows in decode).

The shared device memory region of the store holds, one use at a time and re-zeroed or restored at each switch (`scr_alloc` / `fit_alloc` / `dec_alloc`): the prompt scratch (one layer of exact rows, 540 MiB at 256k), the fitter's stage and tables, and the decode-only buffers (resident rows, tail buffers, staging block). Separate buffers cost 5 - 12 ms of cudaFree / cudaMalloc per prompt on WDDM.

Codes in host memory during the prompt (only with `--kvq-prefill-free-codes`): every write of the codes and tables goes to a pinned host copy with the bytes of the device tensors (1,830 MiB at n_ctx 256000). With the flag the device copy exists only for the packed path: the first prompt-path ubatch frees it, the end of a prompt (`llama_memory_kvq_fit_wait`) allocates it after the context switched to the tg compute buffers (`set_tg_cb`), uploads the host copy of [0, n) and the tables. A packed ubatch that comes before the fit_wait (the server's last 4-token batch of a prompt) triggers the allocation in `llama_context::process_ubatch` after the phase switch. The graph input checks a generation count of the device copy. Without the flag (default) the device copy is allocated once at enable and stays. The flag is refused with `n_seq_max` > 1.

## 3. Several sequences

Commits 811195375 .. d7887c1ad (2026-11-20, design in the owner's design note of the same day; steps 1 - 3b of its four steps). The memory takes `n_seq_max` sequences; the server runs `-np N`.

### 3.1 Pool of chunk pages

The host side of the store is a pool of pages (`llama_kvq_pool`, `src/llama-kvq-pool.{h,cpp}`): one page = one chunk (C rows) of every attention layer and kv head: the pinned q8_0 rows `[G][C][K, V]`, the host copy of the codes `kc` / `vc`, the table slots, and the kept statistics sums of that chunk. At 16 layers x 4 kv heads and 8192 rows a page is 330.4 MiB pinned (rows 272 MiB, host codes 50 MiB, tables 8.6 MiB); the kept sums add about 16 MiB (design estimate). The device has the same pages: `kc [nwk, C, G, n_pages]`, `vc [nwv + 1, C, G, n_pages]` per layer, tables `tab [.., 2G, n_pages]` with slot = page. Every page has C rows (the last chunk of a context has a full page).

n_pages = ceil(n_ctx / C) + n_seq_max - 1 (each sequence may hold one partial page). The device part of all pages (codes and tables) is reserved at load; the host part is allocated on demand (3.3). The attention op reads the codes and tables through the page table `pt`; the host functions that touch rows resolve a position through the chunk -> page table of the sequence (`row_ptr`, `hcode_ptr`, `chunk_runs` for ranges crossing a chunk).

### 3.2 Page allocator

Every page has a holder count (0 or 1 now). A sequence takes the pages for the chunks of a batch in `init_batch` (`llama_kvq_store::reserve`): the free page with the lowest id (so one sequence keeps the identity mapping). It gives them back in `seq_rm` of the end (the chunk that keeps rows stays), `clear`, a failed state load and a failed batch (`release_from` / `release_all`). Take and release reset the kept sums of the page and upload the page table of the sequence (async on the main stream, only when it changed). A state load takes the pages of its rows.

Pool full (no free page, the host allocation failed, or `--kvq-host-max-mib` would be passed): `init_batch` returns a failed prepare with one log line (`the kvq page pool is full` / `the kvq host memory cap is reached`), `llama_decode` returns 1, nothing aborts, a failed batch gives back the pages of all its sequences.

### 3.3 Host RAM on demand and given back

The pinned buffer of a page is allocated by a background thread of the pool when the page is reserved, and for the next free page ahead (only when the sequence is in the last quarter of its last page, so an idle sequence holds no spare page); the reservation waits for the pages it takes (counted when it blocks). When the last holder lets a page go, `trim()` (called by `release_from` after the streams are synchronized) frees the host memory of the free pages beyond `--kvq-host-keep-mib` (highest ids first) on the same thread (cudaFreeHost synchronizes the device, so never on the decode thread); the page is `ST_RELEASING` until done and is not handed out meanwhile; allocations go before frees in the thread; a reservation prefers a free page that has memory, then queued, then without, last those being freed. The page stays in the pool with its device part reserved. `--kvq-host-prealloc` allocates all pages at load and never frees. `llama_kvq_memory_info` and the server's `kvq: pool:` line report pages, held pages, allocated host bytes, allocations, frees and the waits that blocked.

### 3.4 Per-sequence state

Per sequence (structs `llama_kvq_store::seq_state`, `llama_kvq_fitter::seq_fit`, `llama_memory_hybrid_kvq::seq_state`): page table, n_coarse, n_host, the device page table input, the query and gate sums (`qacc`, `gacc`), per layer the resident set (ridx, slot stamps, pinned mirror), the fitter's jobs, open-chunk sums, snapshot, progress counters, the fold / settle / tail counters. Shared: the page pool, the shared device memory and its modes, the rotating resident and tail buffers, the staging block, the worker, the gather pool and the streams. A ubatch of another sequence marks every own resident device buffer stale (`use_seq`), so the next join uploads the mirror of the new sequence. The API takes a sequence id: `llama_memory_kvq_fit_wait(mem, seq_id)`, `llama_memory_kvq_fit_get_progress(mem, seq_id, out)`; the counters and the memory info are totals.

### 3.5 Ring streams and memory operations

The kvq memory builds its attention (ring) cache with one stream per sequence whatever `kv_unified` says; cell = position mod size per stream; the context gives every sequence the whole context (`n_ctx_seq = n_ctx`). The ring costs one ring per sequence in pinned host memory (144.5 MiB at the default size). Memory operations: `seq_rm` per sequence (-1: every sequence), `seq_rm(s, -1, -1)` empties a sequence (pages, fitter, store, ring, recurrent), `seq_keep(s)` empties the others, state save / load per sequence, tokens with several sequence ids are refused. `seq_cp`, `seq_rm` of a middle range and `seq_add` / `seq_div` abort with a message that starts with `not supported by the two-tier KV cache (--kvq)`, then the reason and what to do; the server never calls them with its `--kvq` settings.

### 3.6 Splitting a batch and packed decoding

The kvq memory splits a batch itself (`llama_memory_hybrid::prepare_batch`, `llama_batch_allocr::split_seqs`): the sequences whose tokens of the batch are a whole decode / verify run on the packed path (at most max_batch = 8 tokens, the same count for each, ascending ids) go through one ubatch of up to min(n_seq_max, 8) sequences (one pass over the weights; `ggml_kvq_attn_streams`); every other sequence (a prompt) has its own ubatches (`split_equal` with one sequence, so the n_rs_seq keep-tail rule of the recurrent part holds); the packed groups come first. A lone sequence stays one sequence per ubatch as before and the graph is unchanged for one stream.

Op stream dimension (5032d4c33): one call serves S = q->ne[3] <= GGML_KVQ_MAX_STREAM (8) sequences of L positions each. Layout: q [D, n_head, L, S], k / v [D, n_tail, G, S] (one ring per stream), pt [n_chunk_max, S], rkv [D, cap, G, 2S] (K at i3 = 2s, V at 2s + 1), ridx [cap, G, S], mask [n_kv, L, 1, S]; result [D, n_head, L, S] or score [n_coarse_max, G, L, S] (rows at or after a stream's n_coarse are 0). The per-stream n_coarse are op parameters (op_params[3] = the largest, [4] = S, [5 + s] = n_coarse of stream s), so pt keeps its shape. In CUDA every grid has the stream in a grid index and each block builds the view of its stream from its own logical numbers, so a stream's output has the same bits as the call alone and does not depend on the other streams; the kernels are templates on MS (more than one stream), and S = 1 compiles to the code as before (bit-identical and as fast).

Store and graph (d7887c1ad): resident buffers (`rkv_h`), the ridx / staging block, `h_sel` and the tail buffers have one slice per stream (n_str = min(n_seq_max, 8)); per stream the prefetch plan, request cut, fetch list, mirror write, base upload, tail upload and readback, join scatter; a block upload copies only the ridx slices of the ubatch streams. The mask fillers work per stream (rows beyond a stream's n_coarse are -inf). qwen35 builds q / out with the stream dimension, per-stream masks, the page table input [n_chunk, S] and the selection bias [n_coarse, 1, 1, S] as graph inputs, and per-sequence query / gate sums. The matmuls see another batch size than a solo run, so the logits of a packed run differ from a solo run in rounding; two runs of the same schedule are identical, and the output of a sequence depends only on its own state, not on thread timing.

### 3.7 Reserve for packed decoding (step 3c, in progress, not in HEAD)

Specification of the owner's choice: predictability over optimality, a server must never grow VRAM mid-run. At load the context reserves the device memory for packed decoding: resident rows per stream for min(n_seq_max, 8) streams (capped at 8192 rows), plus the packed compute graph in the reserve (`llama_memory_hybrid_kvq::init_full_packed`: n_streams() sequences of max_batch tokens each, every coarse region at its largest, nothing executed). In HEAD the resident and tail buffers are already sized per stream at load, but the packed graph of several sequences is not part of the reserve, so the compute buffer can still grow at the first packed decode of several slots.

TODO (fill in after step 3c reports): VRAM reserved per stream (resident rows, tail buffer, packed compute graph) at -np 1 / 2 / 4 / 8; max `-c` per `-np` that fits 15,872 MiB with the reserve; whether the load-time number now equals the run maximum (no growth after the first packed decode); prompt and decode speed against d7887c1ad (section 7.9).

### 3.8 Server with `-np N`

`server_kvq_prepare_params` no longer forces one slot. With n_parallel > 1: `kv_unified` is set (every slot sees the whole context, the slots share the pool of pages; `-np` auto = 4 and `--kv-unified-per-slot` work as for the plain cache); `--kvq-prefill-free-codes` fails at startup; `cache_idle_slots` is off (idle slots are not saved to the prompt cache; with several slots a full pool purges idle slots through `try_clear_idle_slots`); n > 1 completions (children continue from `seq_cp` of the parent slot) get an invalid request error. `fit_wait` is called per slot at the end of its prompt. `decode()`: `llama_decode` returning 1 with `--kvq` means the pool is full (the batch is all or nothing): purge an idle slot and retry; else cut the view to the tokens of the first slot and retry slot by slot; a slot that alone gets no page gets the error "kvq: the pool of pages is full ..." (HTTP 500 / error event), is released and its sequence emptied, its remaining tokens are skipped and the other slots continue. The per-slot kvq log lines use the slot's sequence id. The drafter's context (a plain cache, `-c` positions in total) is not paged: when it is full the slots of the batch stop (section 9).

## 4. Code map

| Part | Files |
|---|---|
| Codec (tables from the moments, bit allocation, packed layout) | `src/llama-kvq-codec.{h,cpp}`, `src/llama-kvq-pack.{h,cpp}`; CPU fit / codes reference of the tests in `tests/kvq-ref.h` |
| Attention op, CPU reference and CUDA | `ggml_kvq_attn`, `ggml_kvq_attn_streams` in `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cuda/kvq-attn.{cu,cuh}`; CPU in `ggml/src/ggml-cpu/ops.cpp` |
| Flash attention with the log-sum-exp (prompt scratch in key blocks) | `ggml_flash_attn_ext_lse`, `ggml_flash_attn_ext_get_lse` in ggml.h / ggml.c; CPU in `ops.cpp`; CUDA MMA kernel and stream-k fixup in `fattn-mma-f16.cuh`, `fattn-common.cuh`, dispatch (GQA variants without a mask) in `fattn.cu`; blocks and merge in `build_attn_kvq_scratch` in `src/models/qwen35.cpp`, the mask of the rest in `set_input_kvq_scr` in `src/llama-memory-hybrid-kvq.cpp` |
| Fit ops (mean, covariance, rotation, V gain, code, pack, table, bits), CPU reference and CUDA | `ggml_kvq_fit`, `ggml_kvq_fit_table`, `_bits`, `_pack` (KVQ_FIT modes) in ggml.h, `ggml_compute_forward_kvq_fit` in `ops.cpp`, `ggml/src/ggml-cuda/kvq-fit.{cu,cuh}` |
| Store: host rows, resident cache, prefetch worker, scratch uploads, per-sequence state | `src/llama-kvq-store.{h,cpp}` (`seq_state`, `reserve`, `release_from`, `row_ptr`, `codes_up`, `set_chunk_table`, `fit_alloc` / `scr_alloc` / `dec_alloc`) |
| Page pool (pinned buffers, holder counts, allocator thread, trim) | `src/llama-kvq-pool.{h,cpp}` |
| Fitter (device graphs, waiter thread, per-sequence jobs) | `src/llama-kvq-fit.{h,cpp}` (`plan`, `build_codes`, `queue_job`) |
| Memory (hybrid subclass), batch split, seq ops, state, mask fillers | `src/llama-memory-hybrid-kvq.{h,cpp}`; ring-shaped device cache `src/llama-kv-cache.{h,cpp}`; split `llama_batch_allocr::split_seqs`, `llama_memory_hybrid::prepare_batch` |
| Graph | `src/models/qwen35.cpp` |
| Scheduler: copy callback before the wait | `ggml_backend_sched_set_copy_callback_early` in `ggml/src/ggml-backend.cpp` |
| Compute buffers per phase (pp / tg) | `llama_context::phase_set`, `ubatch_tg`, `sched_reserve` in `src/llama-context.cpp`; `ggml_gallocr_resize` in `ggml-alloc.c`; `ggml_backend_sched_resize_buffers` in `ggml-backend.cpp` |
| GDN rollback by replay (ggml-org PR 31669) | `src/llama-memory-recurrent.cpp`, `src/models/delta-net-base.cpp`, `ggml/src/ggml-cuda/gated_delta_net.cu`; turned on in `tools/server/server.cpp` |
| Internal API | `src/llama-ext.h` (`llama_kvq_params`, `llama_kvq_fit_params`, `llama_kvq_config`, `llama_init_from_model_kvq`, `llama_memory_kvq_*`, `llama_kvq_memory_info`) |
| Settings, forced settings | `common/arg.cpp`, `common/common.{h,cpp}`, `tools/server/server.cpp` (`server_kvq_prepare_params`), `tools/server/server-context.cpp` (fit_wait per slot, pool-full handling, log lines, n > 1 refusal) |
| Evaluation and benchmark tools | `tools/kvq-eval` (`llama-kvq-eval`, `llama-kvq-bench`) |
| Tests | `tests/test-kvq-codec.cpp`, `tests/test-kvq-attn.cpp` (with `kvq-ref.h`, `kvq-attn-problem.h`; one call with 2 - 4 streams against single-stream calls), `KVQ_ATTN` (52 cases incl. shuffled pages and 2 / 4 streams), `KVQ_FIT` (25), `FLASH_ATTN_EXT_LSE` (84), `GATED_DELTA_NET` cases in `tests/test-backend-ops.cpp` |

## 5. Settings

All server settings below also work in `llama-server` only (`LLAMA_EXAMPLE_SERVER`); the env names are the `LLAMA_ARG_*` of `common/arg.cpp`. Defaults are `common_params::kvq` in `common/common.h`. Costs are for 250k, RTX 5090, unless said otherwise.

### 5.1 `--kvq*` flags

| Flag (env) | Default | What it does, when to change it, cost |
|---|---|---|
| `--kvq` (`LLAMA_ARG_KVQ`) | off | Turns the two-tier cache on, with the forced and refused settings of 5.3. Without it nothing here applies |
| `--kvq-chunk N` (`LLAMA_ARG_KVQ_CHUNK`) | 8192 | Rows per chunk = rows per page = one table set. 16384 halves the tables (268 -> 135 MiB at 256k) and the fit stage but costs a little quality (mean KL ratio 1.12, median 1.02); 4096 is 8% better in KL at equal table memory but doubles tables and fit work and decodes about 2% slower. Also sets the page size (host ~330 MiB per page at 8192) and so the host RAM granularity |
| `--kvq-rem own\|merge` (`LLAMA_ARG_KVQ_REM`) | own | How the rows after the last complete chunk are fitted at the end of a prompt (2.1). `merge` is not better in quality (no systematic difference), costs 82 MiB more during the fit and doubles the end-of-prompt wait of a turn that completes a chunk; keep `own`. Not yet refused with several sequences (it will have to be when pages are shared, section 9) |
| `--kvq-min-rows N` (`LLAMA_ARG_KVQ_MIN_ROWS`) | 256 | `own`: fewer rows after the last complete chunk stay precise in the ring. 256 is the rank limit of the covariance (D + 1 = 257 rows); no quality difference measured between fitting and keeping 10 - 1000 rows. Lower: more rows coded, fitting a tiny chunk costs the same as a big one |
| `--kvq-tail N` (`LLAMA_ARG_KVQ_TAIL`) | max(n_batch, 4096) + 256 | Cells of the ring (pinned host memory, per sequence; 144.5 MiB at 4352). N > 0, not below n_batch + 256; a context of at most N positions gets one cell per position. A reply folds (fits the rows after the last chunk in line, 0.09 s at 24k with the GPU fit) every ring size of tokens; a larger ring folds less often, costs host RAM per sequence |
| `--kvq-prefetch-n N` (`LLAMA_ARG_KVQ_PREFETCH_N`) | 1024 | Rows per (kv head, position) selected by the early query. Must be >= 1; the server raises `--kvq-resident-cap` to it if smaller. More rows: better quality, more fetch traffic and attention time (the resident cap scales with it) |
| `--kvq-resident-cap N` (`LLAMA_ARG_KVQ_RESIDENT_CAP`) | 0 = 8 x prefetch_n = 8192 | Resident slots per (layer, kv head); 0 holds the whole request of a verify batch of 8; a larger union is cut by the max score. 6144 is as good at 8 per batch and saves 71 MB of pinned memory (12 MiB VRAM with rotating buffers), but is 0.4 ms slower at n_max 7. The traffic of rotating buffers is linear in it. A fallback on a slow bus. With several sequences the resident memory is per stream (3.6) |
| `--kvq-lead N` (`LLAMA_ARG_KVQ_LEAD`) | 2 | Layers between the early query and its attention; 0 = no prefetch (packed mode needs 1 - 3). Lead 1 halves the fetch window, lead 3 fetches 37% more rows (KL 1.04e-4 against 8.0e-5) with no speed gain |
| `--kvq-layer-bufs N\|all` (`LLAMA_ARG_KVQ_LAYER_BUFS`) | all | Device buffers of the resident rows. `all`: one per attention layer, rows stay on the device, a verify batch uploads only fetched rows. N >= 2: rotate through N buffers, every batch re-uploads each layer's resident set from the pinned mirror (272 MiB per batch at cap 8192): 110 MiB less VRAM below 250k, never faster, +2 to +4 ms per verify batch at PCIe 3.0 speed. Same output |
| `--kvq-scr-block N` (`LLAMA_ARG_KVQ_SCR_BLOCK`) | 16384 | Prompt attention over host rows in mask-free key blocks of N rows (multiple of 256). 0 = one flash attention with a mask over all rows: bit-identical to a full device cache, 30% slower at 250k, 1.3 GB more prompt-phase VRAM. 8k saves 64 MiB for +3.5% prompt time; 32k, 64k are slower. Any non-zero value changes the generated ids against 0 and against each other |
| `--kvq-prefill-free-codes` (`LLAMA_ARG_KVQ_PREFILL_FREE_CODES`) | off | Free the device copy of the codes and tables while a prompt is processed, allocate it again at its end; the pinned host copy holds them meanwhile. Saves 734 MiB at ctx 102400, 1,831 MiB at 250k in the prompt phase; costs one allocation and an upload of the host copy per prompt (about 57 ms at 250k, 20 ms at 100k), which sits on the critical path of a short follow-up prompt. Needed for 250k in 15.5 GiB. Refused with `-np` > 1. Same output with and without. At short contexts leave it off |
| `--kvq-host-prealloc` (`LLAMA_ARG_KVQ_HOST_PREALLOC`) | off | Allocate the pinned host memory of every page at load (3.4 GB + 0.7 GB at ctx 102400, 8.5 + 1.8 GB at 256k), never free it. Default: a page gets host memory in a background thread when a sequence first needs it and gives it back to the OS when no sequence holds it. Use for a single long-lived request when allocation time during the first prompt matters (the on-demand path costs +0.5% on a 250k prompt) |
| `--kvq-host-max-mib N` (`LLAMA_ARG_KVQ_HOST_MAX_MIB`) | 0 = no cap | Cap of the pinned host memory of the pages (allocated and queued), MiB. A batch whose sequence would need a page over the cap fails like a full cache (3.2). Ignored with prealloc. Set it to keep RAM for the rest of the machine |
| `--kvq-host-keep-mib N` (`LLAMA_ARG_KVQ_HOST_KEEP_MIB`) | 0 = give everything back | Keep up to N MiB of free pages' host memory for reuse instead of freeing it. The first request after a release pays its allocation (about 20 - 50 ms per page). The server RAM prompt cache (`--cache-ram`) is separate pageable memory and not affected. Ignored with prealloc |

Parse rules (`common/arg.cpp`): `--kvq-scr-block` is 0 or a positive multiple of 256, `--kvq-tail`, `--kvq-chunk`, `--kvq-min-rows`, `--kvq-prefetch-n` >= 1, the others >= 0; `--kvq-rem` is `own` or `merge`. `--kvq-layer-bufs` is `all` or an integer.

### 5.2 Other flags that change the kvq budget

| Flag | Effect |
|---|---|
| `--[no-]phase-buffers` (context param `phase_bufs`, default auto: on with `--kvq` and for a DFlash drafter, off otherwise) | `--no-phase-buffers`: one compute buffer for all ubatches, no switch between the prompt and decode phase; decode then holds the prompt buffers again (about +1.7 GB at 250k). Off automatically with more than one sequence. `--phase-buffers` turns the switch on for a plain context |
| `--ubatch-size N` | Prompt ubatch; sets the prompt compute buffer (485 MiB at 1024, 1,570 at 4096 with blocks) and the host compute buffer (89 / 354 MiB). 4096 gains about 2% prompt time over 1024 |
| `--batch-size N` | Tokens per `llama_decode`; sets the drafter's layer input buffer (400 MiB at 4096, 100 at 1024; prompt phase) and the default ring size. Changes the drafter's batch boundaries and so the ids |
| `--spec-draft-n-ubatch N` | Ubatch of the drafter; 1024 keeps its prompt compute at 120 MiB with `--ubatch-size 4096` (480 without) |
| `--spec-draft-n-max N` | Verify batch is n_max + 1 tokens, at most 8: the server clamps n_max to 7. Resident capacity and the tg compute buffer follow it |
| `--ctx-size N` | Device: codes 6.1 MiB + tables 1.05 MiB per 1000 positions (1,562 + 268 MiB at 256k), scratch about 2.1 MiB per 1000 positions (540 MiB at 256k); host: store 34.8 KB per position (8.5 GB at 256k). With several slots the pool is sized by the total context |
| `--parallel N` | Slots; see 3.8. `--kv-unified-per-slot` sizes the pool as for the plain cache |
| `--fit` | Does not know the kvq memory; use `--fit off` with `--ctx-size` and `-ngl` set |

### 5.3 What `--kvq` forces or refuses in the server (`server_kvq_prepare_params`)

Refused at startup (error, the server does not start): `--mmproj` (any of path, repo, url), `--embedding`, `--kvq-prefill-free-codes` with `-np` > 1. A model that is not qwen35 or a device that cannot run the ops fails when the context is created (`llama_init_from_model_kvq` returns null and logs the reason).

Forced with a warning: `ctx_shift` off, `--cache-reuse` 0, flash attention on, K and V cache type q8_0, `--spec-draft-n-max` clamped to 7 (drafts above that are verified with the scratch attention), `--kvq-resident-cap` raised to `--kvq-prefetch-n` if smaller, recurrent rollback by replay (`rs_replay`; `LLAMA_RS_REPLAY=0` overrides). Set without a warning: `cache_idle_slots` off, `kv_unified` on with `-np` > 1. Warned only: `--ctx-checkpoints` <= 0 (a changed conversation is processed again from the start), `--fit` on, drafts of more than 7 tokens. Per request: n > 1 completions are refused with an invalid request error. Attention rotation of quantized caches is off in the kvq context only (an MTP context of the same model keeps it).

### 5.4 Environment variables

| Variable | Kind | Effect |
|---|---|---|
| `LLAMA_ARG_KVQ*` | settings | The env names of 5.1 |
| `LLAMA_RS_REPLAY` | setting | Overrides the context param `rs_replay`: 0 keeps the GDN snapshot rows (+1,020 MiB at n_max 7) instead of the replay (+27 MiB host per checkpoint). Read in `llama_context` for every context |
| `LLAMA_KVQ_UP_REPEAT=k` | test knob | Every base upload of the resident rows (every round with rotating buffers; the mirror after a prompt with own buffers) and every tail upload is issued k times: emulates a slower bus (k = 2 about PCIe 4.0, k = 3 about PCIe 3.0 on the RTX 5090). The join uploads of the fetched rows are not repeated. Default 1 |
| `LLAMA_KVQ_FOLD_EVERY=N` | test knob | A fold whenever N reply rows follow the coarse region. The only way to exercise folds in kvq-eval, whose replies never fill the ring (the fold determinism check uses it). Default off |

Dropped (no longer in the code): `LLAMA_KVQ_NBUF` (now `--kvq-layer-bufs`), `LLAMA_KVQ_IDLE_PROBE`, `LLAMA_KVQ_GATHER_THREADS`, `LLAMA_KVQ_SCRATCH`, `LLAMA_KVQ_FIT_SYNC`, `LLAMA_KVQ_SCHED_SYNC`, `LLAMA_KVQ_UP_CHUNK`, `GGML_KVQ_ROWS_PER_BLOCK`, `LLAMA_SERVER_GEN_TIMES`, `LLAMA_KVQ_FOLD_FIXED`, `LLAMA_KVQ_DIAG_NORESET`, `GGML_KVQ_COND`, and the flags `--kvq-threads`, `--kvq-tail 0`, `--kvq-v-tables`, the top-up modes, the kvq-eval history modes. (One error message in `src/llama-memory-hybrid-kvq.cpp` still names `LLAMA_KVQ_SCRATCH`.)

### 5.5 `llama-kvq-eval` (teacher-forced KL against saved reference logits)

`llama-kvq-eval -m MODEL -ngl 99 -fa on -fit off -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --kvq-tokens TOKENS.bin --kvq-prefix-len S --kvq-chunk 1 --kvq-ref REF --kvq-lib`. Tool arguments are removed from argv before `common_params_parse`; an unknown `--kvq-*` is an error.

| Flag | Default | Meaning |
|---|---|---|
| `--kvq-tokens F`, `--kvq-prefix-len S` | required | Token file (ids) and the length of the prefix; the rest is the scored reply |
| `--kvq-chunk N` | 64 | Tokens per scoring decode (not the table chunk; 1 = one token per decode, 8 = a packed verify batch) |
| `--kvq-save-ref F`, `--kvq-ref F`, `--kvq-ref2 F`, `--kvq-csv F` | - | Write the logits of the run, score against a reference (and a second one), per-position CSV |
| `--kvq-no-rot` | off | Attention rotation off (for the plain q8_0 teacher; without `--kvq-lib`) |
| `--kvq-perf` | off | Timing output |
| `--kvq-lib` | off | The library path (the same as the server); without it the tool runs a plain cache |
| `--kvq-fit-chunk N`, `--kvq-rem own\|merge`, `--kvq-min-rows N`, `--kvq-ring N` | 8192, own, 256, 0 (= max(batch, 4096) + 256) | Table chunk, remainder policy, minimum rows, ring cells |
| `--kvq-pf-n N`, `--kvq-cap N`, `--kvq-lead N`, `--kvq-scr-block N` | 1024, 0, 2, 16384 | As `--kvq-prefetch-n`, `--kvq-resident-cap`, `--kvq-lead`, `--kvq-scr-block` |
| `--kvq-prefill-free-codes`, `--kvq-host-prealloc`, `--kvq-host-keep-mib N`, `--kvq-host-max-mib N` | off, off, 0, 0 | As the server flags |
| `--kvq-lib-hold` | off | No chunk joins during the scoring (chunks join in `fit_wait` only) |
| `--kvq-rm-test`, `--kvq-rb-test` | off | Removal cases of both policies; ring rollbacks at C + 501 with and without an earlier rollback and the same positions in two batch splits (`--kvq-rb-test` also runs on a plain cache) |
| `--kvq-turns S1,S2,... [--kvq-turn-gen N]` | - | Process the prefix in turns (prefill to S_i at batch multiples, end of prompt, N greedy tokens one per decode, back to S_i through a recurrent checkpoint, next turn); also on a plain cache |
| `--kvq-state-save F`, `--kvq-state-load F`, `--kvq-state-ckpt P`, `--kvq-state-refuse F1,F2` | - | Save the state after the prefill and the wait, load instead of the prefill, a recurrent checkpoint at P with a rollback to it after save or load, files whose load must be refused (the sequence stays empty and usable) |
| `--kvq-multi-seq N [--kvq-multi-off K] [--kvq-offset K] [--kvq-multi-ex] [--kvq-multi-short M] [--kvq-multi-full] [--kvq-multi-pack]` | - | N sequences (the token file from k*K on, K default 997) interleaved in the ubatches of one run, logits of each in `<save-ref>.sK`; `--kvq-offset K` drops the first K tokens (the matching single-sequence run); `-ex` runs rollback, empty, state round trip, `seq_keep(0)`; `-short M` lets the first M sequences prefill one batch only; `-full` uses a pool too small for all (a failed decode ends that sequence, it is run alone at the end); `-pack` decodes the score steps of sequences with the same token count as one packed ubatch |

`llama-kvq-bench` (`tools/kvq-eval/kvq-bench.cpp`) times the attention op on the GPU: `--quick`, `--profile` (a fixed workload for a profiler), `--hash` (hashes of GPU outputs of large problems, to compare two builds bit for bit), `--bench-nkv N`, `--bench-cap C`, `--bench-l 1,4`, `--no-fa`, `--no-idle-wait`.

### 5.6 Which settings change the generated ids

Known batch-boundary effect, not a bug: `--ubatch-size`, `--batch-size`, `--spec-draft-n-ubatch`, `--spec-draft-n-max`, `--kvq-scr-block` (any non-zero value against 0 and against each other), the slot mix with `-np` > 1 (the matmuls see another batch size). No change: `--no-phase-buffers`, `LLAMA_RS_REPLAY`, `--kvq-prefill-free-codes`, `--kvq-host-*`, `--kvq-layer-bufs`, slot save and restore.

## 6. Build and run

Windows, Visual Studio 2022 build tools, CUDA 12.9, Ninja, from a VS x64 developer shell:

```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cuda --target llama-server llama-kvq-eval test-backend-ops test-kvq-codec test-kvq-attn llama-kvq-bench
```

Tests (GPU, `-b CUDA0`): `test-backend-ops -o FLASH_ATTN_EXT_LSE`, `-o FLASH_ATTN_EXT`, `-o KVQ_ATTN`, `-o KVQ_FIT`, `-o GATED_DELTA_NET`, `test-kvq-codec`, `test-kvq-attn`.

Measured configuration (IQ3_S target, DFlash2 Q4_K_M drafter, 250k):

```
llama-server --fit off -m Qwen3.8-27B-UD-IQ3_S.gguf -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf ^
  --ctx-size 256000 --n-gpu-layers all -ngld all --parallel 1 ^
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja ^
  --spec-type draft-dflash --spec-draft-n-max 7 -ctkd f16 -ctvd f16 ^
  --batch-size 4096 --ubatch-size 1024 --kvq --kvq-prefill-free-codes
```

Two configurations fit 15,872 MiB at 250k: A (above, ubatch 1024), largest sample 15,271 MiB in the decode phase, prompt 139 s; B (`--ubatch-size 4096 --spec-draft-n-ubatch 1024`), 15,372 MiB in the prompt phase, 136 s. B without the drafter flag is 15,876 MiB (4 over). A is the recommendation. For several slots drop `--kvq-prefill-free-codes` and add `--parallel N` (section 7.9 for what fits).

Log lines to expect: the store line at load (`llama_kvq_store: device: codes ... ; host: ... on demand`), per prompt `kvq: waited ... at the end of the prompt, coarse region N rows` and `kvq: pool: H of P pages held, pinned host memory ...`, with `--kvq-prefill-free-codes` the `scr_alloc` / `codes_free` / `codes_alloc` lines, and the phase switches (`sched_reserve: ... compute buffer size of the tg phase`).

## 7. Measurements

Machine: RTX 5090 32 GB, Ryzen 9950X3D, Windows 11, CUDA 12.9. VRAM is dedicated usage of the process (5 Hz sampler), MiB, n_max 7, budget 15,872 unless noted. Single runs unless noted; treat them as such. Builds: "prototype" = branch `kvq-two-tier-cache` (commit named in each row); "kvq" = branch `erokhins/kvq` (ours), commit named. Builds before 2026-11-19 are history unless a number is still the reference (marked).

### 7.1 Memory per phase at 250k (n_ctx 256000), 2026-11-19, prototype 6d2ecbfab, same on kvq

| configuration | load | prompt peak | decode median | follow-up peak | headroom to 15,872 |
|---|---|---|---|---|---|
| before this round (a724f2299), ub 1024 | 19,277 | 19,483 | 19,579 | 19,581 | -3,709 |
| A: ub 1024, block 16k, codes out | 14,039 | 14,205 (15,187 at the switch into decode) | 15,159 | 15,271 | +601 |
| ub 1024, block 0, codes out | 15,364 | 15,546 | 15,168 | 15,584 | +288 |
| phases + replay, no codes out, no blocks (15b59d61d), ub 4096 | 19,628 | 20,300 | 15,310 | 20,297 | -4,428 |
| B: ub 4096 + drafter ub 1024, block 16k | 15,126 | 15,372 | 15,176 | 15,354 | +500 |
| ub 4096, block 16k, no drafter flag | 15,186 | 15,876 | 15,320 | 15,408 | -4 |
| ub 4096 + drafter ub 1024, block 0 | 17,746 | 17,974 | 15,172 | 17,854 | -2,102 |

In A the decode phase sets the headroom (the codes come back on the device), in B the prompt phase. 8k blocks give nothing over 16k at ub 4096 (1,570 MiB is the floor of the other activations).

At 100k (n_ctx 102400): before 16,950 / 17,126 / 17,196; A 13,707 / 13,873 / 13,631 (follow-up 13,903); block 0 14,132 / 14,314 / 13,640; B 14,794 / 15,040 / 13,682 (load / prompt peak / decode).

The reference for the current default build (`kvq` 9f9d1e082, R7 GPU fit, 2026-11-20, server session `--kvq-prefill-free-codes`, ub 1024): 250k load 13,951, prompt peak 15,129, run max 15,269; 24k: -88 MiB in every phase against the CPU-fit build. The latest suite (d7887c1ad, `--kvq-prefill-free-codes`, `-c 102400` for 20k / 50k / 100k): load 14,481, r1 prompt 14,650 - 14,654, decode 13,726 - 13,766, follow-up 14,602 - 14,620; 250k: load 13,931, prompt 15,165, decode 15,193, follow-up 15,305 (nvidia-smi whole GPU 15,293).

What the device holds at 250k, configuration A (MiB): target weights (IQ3_S) 10,617 in both phases; drafter weights (Q4_K_M) 1,080; drafter KV (f16) 60; recurrent state (GDN, replay) 177 (1,197 with snapshot rows); kvq codes + tables 0 in the prompt phase / 1,831 in decode (pinned host copy meanwhile); kvq scratch shared with resident / tail / staging 540 both (decode uses about 330 of it, 88 with `--kvq-layer-bufs 2`); kvq other 17; kvq fitter 1 (86 before the GPU fit); target compute 485 prompt / 72 decode (1,570 at ub 4096); drafter compute 120 / 9; layer input buffer 400 / 1; remainder (CUDA context, cuBLAS, ggml CUDA pool) 620 - 705 / 755 - 772. Measured totals 14,205 / 15,159.

Host memory alongside at 256k (pinned unless noted): precise store 8,500, codes and tables 1,831, ring 145, other kvq 281, host compute 89 (354 at ub 4096), drafter host compute 100, mmapped model 521 (not pinned), plus 27 MiB per context checkpoint, slot file or RAM cache entry. About 11.5 GB in all.

Where the 4.4 GB went (250k, ub 1024, prompt phase): GDN replay instead of snapshot rows -1,020 (both phases); compute buffers per phase (drafter prompt buffer is the injection graph 1,099 -> 120, decode holds no prompt buffers, fitter and scratch prompt-only) -980 at the prompt peak and -4,400 in decode; codes in host memory during the prompt -1,830; blocked scratch attention (prompt compute 1,810 -> 485) -1,325 plus -500 of pinned host compute; decode compute learned per phase 1,810 -> 72.

Left to squeeze if ever needed: `-b 1024` at ub 1024 sizes the layer input buffer by the batch (400 -> 100; ids change); the scratch as a ring of blocks (540 -> about 105, section 9); the pool and cuBLAS part of the remainder (100 - 250 MiB, unmeasured); a smaller drafter quant (about 300).

### 7.2 Memory of rows and buffers by design step (history, 2026-11-16 to 2026-11-18)

| Step | Before -> after | Notes |
|---|---|---|
| Ring redesign (host store takes ring rows at once) | ring 688.5 -> 144.5 MiB (default one batch + 256 cells) | prototype fa865aee3 and later; details in 8.3 |
| Resident rows in pinned host mirrors, 2 rotating buffers (2026-11-18) | 278.9 + 276.0 -> 26.1 + 13.1 MiB device, VRAM in generation -514 MiB at 100k; +0.25 - 0.35 ms per verify batch | superseded by per-layer buffers (7.8): the default is now own buffers, +110 MiB below 250k for -0.3 ms |
| Ring in pinned host memory with 4 rotating tail buffers (2026-11-18) | 144.5 -> 36.1 MiB device, -94 to -104 MiB VRAM; +0.12 - 0.32 ms per verify batch at long replies | bit-identical logits |
| Whole union of the verify batch (2026-11-17) | resident 34.5 -> 274.7 MiB, staging 36 -> 276 MiB (+504 MiB) | later shrunk by the two steps above |

### 7.3 Speed (single sequence)

Interleaved pairs, 300-token replies, IQ3_S target, DFlash2 drafter n_max 7, 2026-11-19 (prototype 6d2ecbfab against 15b59d61d; "before" there is phases + replay, speed-neutral against a724f2299: 250k prompt 198 s against 206 s on another day):

| cell | prompt before -> now | decode ms per verify batch | follow-up prompt (500 tokens) |
|---|---|---|---|
| 250k, A (ub 1024, 16k) | 199.3 -> 139.1 s (-30%) | 37.5 -> 37.2 | 1.71 -> 1.50 s (-12%) |
| 250k, ub 1024, block 0 | 199.3 -> 201.9 s (+1.3%) | 37.5 -> 37.4 | 1.71 -> 1.73 s (+1.5%) |
| 250k, B (ub 4096 + drafter 1024, 16k) | 164.0 -> 136.1 s (-17%) | 38.4 -> 37.5 | 1.76 -> 1.49 s |
| 100k, A | 42.9 -> 37.3 s (-13%) | 25.8 -> 25.9 | 1.28 -> 1.14 s |
| 100k, B | 39.5 -> 37.2 s (-6%) | 26.0 -> 26.3 | 1.36 -> 1.25 s |

The gain of the blocked attention: the single flash attention filled an F16 mask of n_rows x n_ubatch on the CPU (512 MiB per ubatch at 250k), uploaded it and read it in every attention layer; the mask-less GQA kernel is a few percent faster once the dispatch was fixed. The same structure exists in ggml-org's plain KV cache, so the gain is not specific to kvq. 16k is the fastest block (8k +3.5%, 32k +4%, 64k +24% at 250k). With `--kvq-prefill-free-codes` the follow-up prompt pays the codes upload (1.75 GB at about 34 GB/s, about 57 ms at 250k, 20 ms at 100k). Phase switches cost 0.5 - 6 ms per compute buffer, about 18 ms into a prompt in all.

Current server suite (kvq d7887c1ad, `--kvq-prefill-free-codes`, DFlash2 n_max 7, ub 1024, 2026-11-20, `kvq-scratch/ms/suite-s3b`): r1 = the whole prompt, r2 = a 501-token follow-up.

| context | r1 prompt s | r1 gen tok/s | r2 prompt s | r2 gen tok/s |
|---|---|---|---|---|
| 24k (ctx 102400) | 7.24 | 166.2 | 0.55 | 142.7 |
| 50k | 15.97 | 127.2 | 0.55 | 184.4 (44 tokens) |
| 100k | 37.67 | 150.6 | 0.65 | 125.0 |
| 250k (ctx 256000) | 140.66 | 71.9 | 0.95 | 92.8 |

Ids are equal to the builds since 8b6ac9880 in every cell (the suite hashes: 8dd6361dfcbc / 84b54f2b51aa at 24k, 097bd6474bb5 at 50k, f6400c101d9d at 100k). The 250k gen tok/s is low because the replies are 300 tokens with acceptance 0.28 on that trace. Milestone of R7 (2026-11-20, 9f9d1e082 against the CPU-fit build): 250k prompt 139.2 against 137.8 s (+1.0%), follow-up 0.95 against 1.44 s, verify batches 40.7 / 43.4 against 40.3 / 43.0 ms (n_max 7 / 3); 24k prompt 7.07 against 7.33 s, follow-up 0.57 against 1.02 s; acceptance set tau 4.513 [4.24, 4.78] against 4.501 [4.25, 4.81].

The GPU fit end-of-prompt wait: 0.098 s at 24k (CPU fit 0.30 s with 8 threads, 0.55 s with 4). The prefill with the GPU fit is +1.4% at 24k (6.50 s against 6.41; the CPU pool overlapped the prompt).

Decode against a plain f16 / q8_0 / q4_0 cache (2026-11-17, prototype build 2758a0a64; the kernels have since sped up, so this is history of the comparison): decode ms per verify batch at n_max 7 / 3, differences to the f16 cache A (16.57 / 14.01 at 24k, 17.53 / 15.05 at 50k, 19.64 / 17.17 at 100k): q8_0 +1.2 / +1.0, +3.0 / +2.9, +6.7 / +6.5; kvq +4.6 / +3.4, +5.0 / +3.1, +6.0 / +2.8; q4_0 +1.6 / +1.5, +3.1 / +3.2, +6.4 / +6.4. The two-tier cost does not shrink with the context (about 4 ms per round: `kvq_precise` over the 8192 resident slots, `kvq_final`, the early-q projection 0.45, the scatter, trigger splits 0.39, 10 top-k kernels per layer). Final prototype numbers of the packed kernels (2026-11-17): 25.96 - 26.43 ms per batch at 100k n_max 7 against 26.1 for our build without `--kvq`, acceptance set 3506 tokens, tau 4.399 [4.07, 4.77], 161.8 t/s against 178.0 for ggml-org master a49db739f, ids equal on 12 of 12.

### 7.4 Quality

Metric: teacher-forced KL on the recorded reply tokens of dataset rows (kvq-eval), against a q8_0 plain cache without rotation (rows `ref/row*.q8c1`) unless noted; KL has a noise floor: two deterministic runs that differ only in rounding are 3.6e-5 - 5.7e-5 apart at 24k and 1.5e-4 at 100k (q8_0 with and without rotation). Top-1 flips are positions where the argmax differs.

Wide test (prototype fa865aee3, 2026-11-16, 42 requests of 12k to 121k, 22,760 scored positions, nothing failed): top-1 flips 56 (0.246%) against 41 (0.180%) for the q8_0 baseline, Wilson intervals overlap; every flip is at a position with teacher top probability below 0.6. KL 1.33e-4 against 6.8e-5; the ratio grows with context, from 1.1 at 10-20k to 2.3 - 3 above 70k (110-150k: 4 requests only). KL by bucket (two-tier / baseline): 10-20k 1.14e-4 / 1.04e-4, 20-40k 1.07e-4 / 6.9e-5, 40-70k 9.3e-5 / 6.3e-5, 70-90k 1.75e-4 / 7.7e-5, 90-110k 1.75e-4 / 6.4e-5, 110-150k 8.9e-5 / 2.9e-5.

Against other caches (prototype 2758a0a64, 2026-11-17), KL at 1 per decode / 8 per batch, top-1 flips at 1 per decode: 

| context (scored) | f16 8 per batch (floor) | q8_0 | kvq | q4_0 |
|---|---|---|---|---|
| 24k (245) | 6.2e-5 | 1.30e-4 / 9.3e-5, 0 | 1.07e-4 / 1.07e-4, 0 | 8.9e-4 / 1.02e-3, 2 |
| 50k (289) | 6.5e-5 | 1.77e-4 / 1.83e-4, 2 | 2.30e-4 / 2.27e-4, 3 | 1.73e-3 / 1.81e-3, 4 |
| 100k (471) | 6.7e-5 | 1.77e-4 / 1.78e-4, 1 | 1.23e-4 / 1.34e-4, 1 | 1.01e-3 / 1.02e-3, 2 |

Against f16 kvq is about as far as plain q8_0; q4_0 is 5 - 10x further. Memory at n_ctx 102400 in generation (that build): f16 22,023, q8_0 19,345, kvq 17,802, q4_0 17,745 MiB; per 1000 positions of n_ctx 64.5 / 39.1 / 15.1 / 23.4 MiB. Against q4_0 the same VRAM at 102400, 854 MiB less at 204800, 7 - 8x lower KL.

Current default (block 16k): KL to the f16 teacher on a 100k trace (row2660, 8 per batch, 2026-11-19): old path (block 0) 1.63 / 1.49 / 1.34 / 1.41e-4 at ub 512 / 1024 / 2048 / 4096, block 16k 2.78 / 3.03 / 2.08 / 2.58e-4; at 60k and 80k the ratio is 0.9 - 1.34 on KL of 1e-5 - 4e-5 (noise); at 20k 2.65e-4 (old 2.57 / 2.84e-4), 50k 2.74e-4 (old 2.27 / 2.55e-4). Taken alone this says the blocked path is worse; it is not (7.7).

Packed multi-sequence quality (d7887c1ad, 24k, 4 per decode, 2 and 4 packed sequences against their solo runs): mean KL 4e-5 - 8e-5, top-1 same 0.996 - 1.0; two runs of the same schedule are byte-identical. A 24k gate (row310, kvq-eval, GPU fit, 2026-11-20): KL 6.36e-2 against 6.40e-2 of the CPU-fit build on the same reference, top-1 identical (96.73%).

### 7.5 Bit identity and determinism

Identical generated ids, slot files and kvq-eval logits to the earlier build in every session checked (24k, 50k, 100k, 250k, n_max 7 and 3, follow-up prompts, a 5000-token reply with a fold, slot save and restore) for: the replay, compute buffers per phase, codes in host memory, `--kvq-scr-block 0`, the pinned ring and mirrors, the page pool and allocator, per-sequence state, host RAM on demand (also with `--kvq-host-prealloc` and `--kvq-host-keep-mib 400`), the stream dimension of the op with S = 1. With block 16k the ids differ from block 0 (rounding) but equal the attention split alone. Gate hashes of the current tree (`kvq-scratch/ms/base`): eval24 logits cd5f0191..., state a3be80ab..., server 24k ids 8dd6361dfcbc / 84b54f2b51aa.

Multi-sequence: `n_seq_max` 2 and 3, each sequence byte-identical to its single-sequence run (also with the fold env and the rollback / rm / state / keep script, c5467ce09); with packed decoding the logits differ in rounding (7.4) but are reproducible. Pool full: the decode of one sequence returns 1, the other goes on identical, the failed one run again alone is identical.

Reproducibility history (2026-11-16): two loads of the same state used to differ (KL 5.1e-5 at 24k, 5.7e-5 at 100k) because `ggml_top_k` on CUDA wrote the selection in atomic order and `ggml_kvq_attn` summed warp partials with float shared-memory atomics. Fixed: the plan sorts the selection (since 2026-11-17 a bitmap), the kernels sum in a fixed order; cost up to 0.1 ms per token. Batch independence holds only for the same batch boundaries (multi-turn with turn ends at batch boundaries is byte-identical to one prefill); other splits move the model's own rows by 1.6e-4 at 100k even for plain q8_0.

### 7.6 Save and restore (prototype, 2026-11-16, at 100k; files read right after writing)

| | kvq | plain q8_0 |
|---|---|---|
| kvq-eval state (prefix 99,395) | 4,459.6 MiB, save 1.84 - 2.33 s, load 0.65 - 0.75 s | 3,452.5 MiB, 2.20 s, 1.30 s |
| server slot file (99,598 tokens, 3 checkpoints, MTP draft) | 5.56 GB, save 2.3 - 2.5 s, restore 1.22 - 1.25 s | 4.51 GB, 2.9 - 3.0 s, 1.96 s |
| RAM prompt cache entry | 4,863 MiB, 0.52 s | not run |

A restored state saved again is byte-identical (24k and 100k); logits continued from a restored state are bit-identical to the run that saved it. The server sequence 100k request, trim, slot save, restart, restore, continuation gave equal ids (300 of 300).

### 7.7 The blocked attention and the teacher's bias (2026-11-19)

A wider check at 100k gave KL to the A1 f16 teacher 1.34 - 1.63e-4 for the old path against 2.1 - 3.0e-4 for block 16k, growing with the number of blocks (1 block + masked tail 1.29e-4, 2 blocks 1.30, 3 blocks 1.63, 4 blocks 2.3 - 2.4, 13 blocks 2.80e-4), while the last prefill position equals the old path to 1e-13. Diagnosis: no bug; the merge chain is f32 end to end and the CUDA LSE error against the CPU is 0.7 - 2.1e-4. The teacher computes its prefill as one flash attention, like the old path, and on Turing and later the CUDA MMA kernel holds the probabilities and the P * V accumulator in half precision against a running maximum over all rows, so the probability mass far below the row maximum is lost over a long tail. Exact f64 attention on dumped rows (layer 31, last 100k ubatch, 99,584 rows): single attention nmse 3.34e-4, mean abs error 3.1e-3, output bias -1.02%; 16k-blocked graph 2.57e-5, 1.0e-3, -0.31%, closer on 1,520 of 1,536 (query, head) pairs. Teachers with the plain path's prefill blocked the same way (16k, 8k, 4k, 2k) are themselves 1.99 - 2.69e-4 from A1; scored against them block 16k is 1.14 - 1.82e-4 against the old path 1.64 - 2.81e-4. Consequences: block 16k is the default; a KL against a single-attention teacher cannot rank attention variants that change the reduction order, and at 100k a KL is only good to about 1e-4; use blocked teacher logits or the exact check on dumped rows (`kvq-scratch/mem/attnsplit`, `ref/A1b*`, `scripts/exact.py`). The bias is in ggml's standard CUDA flash attention for any long prefill, not in this branch.

### 7.8 PCIe and resident buffers per layer

Interleaved pairs, 300-token replies with `ignore_eos`, `LLAMA_KVQ_UP_REPEAT=k` (k = 2 about PCIe 4.0, k = 3 about PCIe 3.0), 2026-11-19 on 92e56429a; ms per verify batch (drafter included), rotating -> own buffers, first request / follow-up (RTX 5070 Ti field test, PCIe 3.0, 256k: 48.0 -> 39.0 ms per batch from the same idea):

| context, per batch | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| 24k, 8 | 25.33 -> 25.36 / 24.96 -> 25.03 | 25.76 -> 25.19 / 25.67 -> 24.96 | 27.78 -> 25.37 / 27.93 -> 25.20 |
| 24k, 4 | 21.10 -> 20.94 / 21.14 -> 21.10 | 21.77 -> 21.08 / 22.01 -> 21.05 | 25.20 -> 21.05 / 25.62 -> 21.08 |
| 100k, 8 | 29.53 -> 29.48 / 31.53 -> 31.44 | 30.12 -> 29.61 / 32.44 -> 31.72 | 31.36 -> 29.68 / 34.15 -> 31.84 |
| 100k, 4 | 23.78 -> 23.64 / 24.60 -> 24.38 | 24.40 -> 23.69 / 25.05 -> 24.39 | 27.26 -> 23.91 / 28.25 -> 24.57 |
| 250k, 8 | 40.96 -> 40.34 / 43.58 -> 42.90 | 41.55 -> 40.77 / 44.19 -> 43.59 | 43.08 -> 40.81 / 45.76 -> 43.40 |
| 250k, 4 | 28.41 -> 28.36 / 30.32 -> 30.12 | 29.17 -> 28.57 / 31.18 -> 30.48 | 30.73 -> 28.72 / 32.99 -> 30.52 |

With own buffers the time no longer depends on the bus (k = 3 costs at most +0.4 ms against k = 1; rotating buffers pay +2 to +4 ms). At k = 1 they are equal within 0.1 ms at 24k and 100k and own buffers are 0.6 ms faster at 250k; ids equal. VRAM rotating -> own: 24k and 100k +110 MiB in every phase (the 16 resident sets of 285 MiB plus tail buffers exceed the 221.5 MiB scratch at ctx 102400); at 250k the 540 MiB scratch holds them and nothing changes. Earlier (2026-11-18, prototype, rotating buffers): the cost of base uploads was +0.30 / +0.37 ms (4 / 8 tokens) at k = 1, +0.64 / +1.24 at k = 2, +1.61 / +3.22 at k = 3 against own device rows; the tail on a slow bus added +0.2 - 0.8 ms at k = 3. Not measured on real PCIe 4.0 / 3.0 hardware except the 5070 Ti report. Packed multi-sequence decoding on the PCIe emulation has not been measured (section 9).

### 7.9 Multi-sequence

Machine and builds as above. Memory: `-np` 1 to 4 at `-c 65536`, DFlash2, 512-token prompt, VRAM max (MiB), `kvq-scratch/ms/s2e`, build 405c751da (one sequence per ubatch, resident buffers per stream already allocated at load): 14,197 / 14,635 / 14,927 / 15,219 (np 1 / 2 / 3 / 4), +438 MiB for the second slot and +292 MiB for each one after; before 405c751da (the drafter pp reserve was a 1,100 MiB token graph when the phase buffers were off): 14,197 / 15,579 / 15,913 / 16,207. `-np 1 --no-phase-buffers` 14,637 (was 15,579).

Maximum context at `-np 2` in 15,872 MiB (build 405c751da, load VRAM, 512-token prompt): -c 131072 15,103; 163,840 15,353; 196,608 15,655; 212,992 15,807 (fits); 217,088 15,873 (4 over, 1 MiB). Before that commit about 82k. With mixed traffic (a prompt of slot A while slot B decodes; codes stay on the device with `-np` > 1; s2c, build 5d3a52126, before the drafter fix): -c 65,536 15,697; 81,920 15,815; 86,016 15,873.

Packed decoding (d7887c1ad): VRAM at `-c 65536`: `-np 2` 15,099, `-np 4` 16,427 MiB (the resident rows are per stream in the shared device memory, about 270 MiB per extra stream at the default cap; max `-c` per `-np` and the reserve of step 3c: TODO, section 3.7).

Throughput (`kvq-scratch/ms/s3b`, server, DFlash2, IQ3_S target, `-c 102400`, ub 1024, 20k-token prompts of different traces, 2000-token replies per slot with `ignore_eos` (400 at `-np 1`), all slots at once; total tokens per second over the generation window; "before" = s2e (405c751da, one sequence per ubatch), "now" = d7887c1ad):

| `-np` | draft n_max 7 before -> now | n_max 3 | no drafter | VRAM max now (7 / 3 / 0) |
|---|---|---|---|---|
| 1 | 219.8 -> 219.4 | 164.6 -> 164.2 | 76.4 -> 76.1 | 14,655 / 14,641 / 12,841 |
| 2 | 162.3 -> 417.8 | 96.3 -> 271.4 | 33.1 -> 124.8 | 15,407 / 15,339 / 13,417 |
| 4 | 158.2 -> 465.0 | 96.4 -> 350.1 | 33.3 -> 165.2 | 16,743 / 16,611 / 14,541 |

`-np 4` at n_max 7 exceeds 15,872 MiB at `-c 102400` (16,743). Per-slot rates at `-np 4` n_max 7 before: 35 - 43 tok/s each. Before packed decoding, concurrency gave no throughput over queuing (one forward pass per slot per step); the packed ubatch does one pass over the weights for all slots. Op timing (5032d4c33, 150k rows, cap 1024): L4 attention 0.167 ms and score 0.119 ms, L8 0.315 / 0.247, unchanged against the single-stream op.

Host RAM shrink (523876fcc, 24k request then a 300-token one in the slot): pinned pool 1,321.5 -> 330.4 MiB (one page of the short prompt); process private bytes 18,436 -> 17,693 MiB at `-np 1` (short prompt alone 17,322) and 20,994 -> 19,748 at `-np 2` (alone 19,417). Cost of freeing: 13 - 15 ms per 330.4 MiB page on the pool thread; with `-np 2`, slot 1 decoding 800 tokens while slot 0 is replaced (4 pages freed) the largest gap between streamed chunks was 163 ms with one page kept, 133 ms with nothing freed, 210 ms with all freed (the new request also waits about 50 ms for its page allocation). Pinned host memory after a 250k prompt 10.5 GiB; the on-demand pool costs a 250k prompt +0.5% (140.67 against 139.98 s).

Pool full (5d3a52126, `-c 32768`, no drafter): the second 20k prompt gets the error, the 24k one finishes with the `-np 1` ids and a later request is served; a 700 MiB host cap gives HTTP 500 for the 24k request (twice) and the 300-token requests after each are served; kvq-eval with `--kvq-host-max-mib 400`: decode failed at position 8192, exit 1, no abort.

Single-slot cost of the pool (s2c / s2d, `-np 1`): 24k prompt 7.15 - 7.25 s, 501-token follow-up 0.39 - 0.42 s, VRAM max 14,655 MiB, the same as before the pool.

### 7.10 Prototype measurements kept for what they explain (history)

Chunk size (prototype fa865aee3, 10 requests of 17k - 101k): mean KL ratio of 16384-row chunks to 8192 is 1.12 (median 1.02), of 4096 to 8192 is 0.88. At equal table memory int8 tables with 4096-row chunks are 8% better in KL than fp16 tables at 8192 (14 requests, 6,437 positions: 1.29e-4 against 1.41e-4); 4096 decodes about 2% slower and doubles the fit work.

Ring redesign (2026-11-16): own against merge over four traces: no systematic difference (24k 4.38e-5 against 7.99e-5, 12k 9.9e-5 against 1.04e-4, 50k 2.36e-4 against 2.29e-4, 100k 1.35e-4 against 1.33e-4, all intervals overlap). Minimum rows of an own partial chunk: fitted against precise rows after the last chunk (9 / 99 / 300 / 999 rows): 6.62 / 5.61 / 7.01 / 6.76e-5 against 6.91 / 6.23 / 5.82 / 5.66e-5 (24k trace), 1.53 / 1.49 / 1.35 / 1.36e-4 against 1.37 / 1.29 / 1.31 / 1.39e-4 (100k), inside the interval. The ring does not need a tail of recent prompt positions: mean KL 1.13e-4 .. 0.79e-4 for 8 .. 8,100 precise recent positions (row2660), 1.92e-4 .. 1.43e-4 (row1830), all intervals overlap. Coding the reply rows with existing tables on append costs +16% (50k) to +29% (100k) KL; refitting every 64 rows +9 to +15% (row2660); the reply stays precise (25 - 37% of the attention mass is on generated tokens). Multi-turn (`--kvq-turns`): turn ends at batch boundaries give a byte-identical state file and identical logits to the single prefill; inside batches KL 2.4e-4 against the teacher (plain q8_0 control 1.64e-4). Ring rollback at C + 501: KL 1.06e-3 (two-tier) against 6.4e-4 - 1.2e-3 (plain q8_0) and 3.6e-3 for decoding the same tokens in another batch split: the model's own batch-shape rounding, not the rollback.

Whole union of the verify batch (2026-11-17): the request used to be cut to 1024 rows by max score while 8 positions want about 3,230 distinct rows (max seen 6,230); each position had on average 52% of its own top 1024 resident, KL at 8 tokens per batch 2.87e-4 against 1.35e-4 at one per decode. With the whole union (cap 8192, plan on bitmaps, oldest-request slot choice, readback of indices only) KL 8.0e-5 [6.0, 10.3], 470 / 471 top-1; decode 33.7 -> 32.7 ms at 100k n_max 7 (the GPU idle at the joins 1.7 -> 0.06 ms), resident +504 MiB at that time. A slot choice by age against plain slot order: missing rows per kv head 719 against 1,185 at n_max 7. The scatter with one trash slot took 3.06 ms per round, 0.41 ms with 1024 trash rows.

Kernels at verify batches of 8 (2026-11-17, bit-identical changes in `kvq-attn.cu`): coarse kernel in column groups (3 x 16 at 48 columns, 168 registers, no stack), `kvq_score` 16-byte loads, `kvq_prep_c`, `kvq_final` with early loads, `kvq_precise` in groups of 8 columns with cp.async: per round at 8 tokens `kvq_precise` 3.72 -> 1.24 ms, at 4 tokens 1.96 -> 1.05; isolated op 9.2 / 9.9 / 16.5 us against 28.3 / 32.7 / 55.3 us at 6 / 24 / 48 columns; decode 0.17 - 0.45 ms faster. Precise rows in blocks of 256 (`kvq_precise_m`, not bit-identical): 0.94 + 0.61 ms against 1.24 + 0.76 per round at 8 tokens; KL inside the interval (7.66e-5 against 8.01e-5). Per round at 8 tokens (nsys, ms): early-q projection 0.451, prep 0.345 + 0.452, SCORE pass 1.971, kvq_score 0.370, top-k 0.735, ATTN pass 2.978, kvq_precise 1.242, final + merge + combine 0.764, scatter 0.646, idle at the trigger splits 0.390; round 26.24 (4 tokens: 20.53). Not kept: q8_0 through registers without the staging buffer; single-pass top-k (1.52 against 0.73 ms); top-k in one kernel with a cluster of 8 blocks (59 against 46 us per call); the early query on the query rows only (-0.15 ms, +188 MiB, other ids); the prep with mma (60 against 25 us).

GPU fitter against the CPU fitter (2026-11-16, 8k chunk at 100k): device 35 ms, CPU 2.0 s on one core; server prefill equal to production within the end-of-prompt wait (44.93 against 44.01 s at 100k, 0.55 s of it the wait); a 554-token follow-up 1.10 against 0.49 s. Replaced by the full GPU fit (2.3, 7.3).

## 8. Decisions and history

### 8.1 Decisions, with the reason (owner's unless noted)

- K 2 bit, V 1 bit: K2V1 is as good as K2V2 in the two-tier setting. Tables per 8k chunk, fitted on the chunk's own rows and queries: no quality cost against one fit over the whole prefix; K tables from another conversation do not work; V fitted per chunk too (offline V tables cost 3.5x KL at 100k). Inverse tables as int8 with one scale per component (2026-11-16): halves the tables (459 -> 235 MiB at ctx 230000), KL inside the noise; one scale per table costs 5x the error on K, 6 bits costs 10 - 50% on K.
- Early query with lead 2: twice the fetch time of lead 1, KL difference inside the noise. Synchronous top-up off (removed): no KL gain, costs a scheduler split.
- Resident capacity: the request is the whole union, default max_batch x prefetch_n = 8192 (2026-11-17); 6144 is as good in quality and 71 MB smaller in pinned memory but 0.4 ms slower at n_max 7 (cap sweep 2026-11-18).
- Prefill uses exact q8_0 rows, not the two-tier attention: two-tier prefill was never measured for quality.
- Copies between GPU graph launches go on a separate copy stream: a memcpy on the compute stream stalled the GPU 55 - 65 us each, about 47 times per token.
- Ring redesign (2026-11-16): no deliberate precise tail of recent prompt positions; fixed chunk grid; the rows after the last complete chunk are fitted at the end of every prompt from scratch and prefill waits for it; the reply stays precise until the next end of a prompt; rollback into the coded region leaves the rows in the host store and refits at the next end of a prompt; the resident set is emptied at the end of every prompt (decoding depends only on the memory state; cost: the first verify batch after a prompt is cold); `--kvq-min-rows 256`; `own` default, `merge` an option; ring of one batch by default (max(n_batch, 4096) + 256 cells, 544 MiB less than before; a long reply folds once per ring size of tokens).
- Compute buffers per phase (2026-11-18): the budget is 15.5 GiB at 250k; prefill and decode never overlap with one slot; the decode-only kvq buffers share the scratch memory; the plain server gets the same phases (13 ms per switch into the prompt phase accepted); the GDN replay of PR 31669 is merged although the PR is WIP (bit-identical output, -1,020 MiB).
- 250k fit (2026-11-19): codes and tables leave the device during the prompt (the +3.3% on the 250k follow-up prompt accepted: "simple is better"; a per-layer overlapped upload hid nothing and was dropped); the blocked scratch attention is the default after the diagnosis in 7.7; `--spec-draft-n-ubatch 1024` is the way to run ub 4096 at 250k.
- 2026-11-20: the codes-out phase is opt-in (`--kvq-prefill-free-codes`, 1d0b92481; without it 250k with one slot no longer fits 15.5 GiB); the fit of the tables runs on the GPU (R7, own kernels, not cuSOLVER: syevjBatched is limited to 32x32 and a new dependency); several sequences with a shared pool of pages like ggml-org's unified cache (VRAM reserved, host RAM on demand and given back); GGML_MAX_SRC stays 10 (one table tensor); packed decoding of several sequences before shared prefixes (step 3 before step 4); behave like the plain cache where it has an answer (unified pool, `split_equal`-like splitting, a full cache fails the batch, the server halves and retries); `seq_add` / `seq_div` stay refused as a ggml-org limit (IMROPE); middle `seq_rm` by masked rows is the chosen design, not built.

### 8.2 Design of the host store and the pages, why a pool

Pool of pages against one fixed store per sequence: any slot can grow to n_ctx while the pool has pages (like the unified cache); a fixed split would give `-np 4` at 250k only 62k per slot; shared prefixes (`seq_cp`) can share complete pages by reference and copy one partial page, against a full copy of about 3.4 GB and 0.5 s for a 100k prefix. Cost: a page-table lookup per chunk in four kernels and about 15 host functions that resolve rows through the page; the estimate was about a quarter more than fixed stores for G1, and fixed stores would have had to be redone for G2.

### 8.3 Ring and resident rows in pinned host memory (2026-11-16 to 2026-11-18)

The ring cells moved to pinned host memory at `fit_enable` (`llama_kv_cache::move_bufs`); the device has 4 rotating tail buffers (36.1 MiB), buffer = attention layer index mod 4, because the early query of layer i scores the ring K of layer i one layer before the resident rows; consecutive users must be at least 3 layers apart, which 3 buffers cannot do over 16 layers. All 4352 cells (9 MiB per layer) would cost +0.25 - 1.0 ms per verify batch at PCIe 5.0; only the occupied cells are uploaded. The resident rows moved to pinned mirrors `[2][G][cap_tot][ROW]` (13.4 MB per layer at 6144 slots) with two rotating device buffers and a base upload per round; where the base upload goes was measured (queued after the join uploads of the layer before: 4.50 ms per token at 4 tokens against 4.43 base; the cost is the copy itself); one staging block for all layers `[ridx | slot list | rows]` (13.1 MiB). The per-layer own buffers (default since 2026-11-19, 7.8) replaced the rotation on the device side and kept the mirrors for restoring after a prompt.

### 8.4 Where each phase's memory went (summary of the 250k work, 2026-11-18 / 19)

An inventory of the process at n_ctx 102400 (39 rows: where, size, formula, lifetime, needed in prefill / decode) is `kvq-scratch/mem/inventory/INVENTORY.md`. Starting point at 250k: 19,277 MiB after load, 19,483 in the prompt, 19,579 in generation. Reasons: the n_rs_seq snapshot rows (7 x 149.6 MiB), compute buffers of the prompt phase held during the reply, the drafter's 1,099 MiB token graph, the F16 K/V copies and mask of the single flash attention (1,810 MiB at ub 1024), and the codes and tables during the prompt (1,830 MiB). Each was removed by one change in 7.1.

## 9. Not implemented, known gaps, ideas

Status key: done, partly done, stale (no longer true, with the reason), open. Order: release blockers R1 .. R7 as the owner numbered them (2026-11-19), then the multi-sequence items in the owner's priority (G1 first, shared prefix last), then known limits and costs, bottlenecks, ideas in the owner's order, upstream candidates U1 .. U6, items that were never exercised.

### 9.1 Release blockers (from `RELEASE-BLOCKERS.md`, 2026-11-19 / 20)

| # | Item | Status | Now |
|---|---|---|---|
| R1 | Memory ops: `seq_cp`, `seq_add`, `seq_div`, `seq_keep` (seq > 0), middle `seq_rm`, several sequences | partly done | `seq_keep(s)` and `seq_rm(s, ...)` work per sequence (c5467ce09); several sequences, `-np N`, packed decoding work (3). `seq_add` / `seq_div`: stale as a kvq item, a ggml-org limit (qwen35 uses IMROPE, `get_can_shift()` is false without `--kvq` too, the server disables ctx shift and cache reuse; decision 2026-11-19). Open: `seq_cp` (step 4, 9.2), middle `seq_rm` |
| R1 middle `seq_rm` | open | Chosen design: gaps as masked rows (bias -inf like the sinks, positions unchanged, the plain hybrid memory also accepts it). Larger than expected: the store's `bias` only feeds the prefetch selection, the coarse attention (`KVQ_ATTN`) has no per-row exclusion, the prompt scratch attention runs mask-free key blocks over the host rows. Needs a coarse-row mask in `KVQ_ATTN`, masks for the scratch blocks that hold a gap, the gap set in the state format, and ring <-> host store copies that skip empty cells. Aborts with a message today |
| R2 | `--fit` with `--kvq` | open | `--fit` measures a plain q8_0 K/V; the server warns. Without `--ctx-size` and `-ngl`, `--fit` silently shrinks the context or moves layers to the CPU. Needs `--fit` to know the codes, tables, scratch, resident sets and host store (and the pool) |
| R3 | MTP drafter (`--spec-type draft-mtp`) with `--kvq` | partly done | The kvq setting is per context since 9031a60a4 (`llama_init_from_model_kvq`): the MTP context keeps rotation and a normal memory. Not tested in a session with `--kvq` on this branch (all measurements use DFlash2). Owner: "potentially interesting"; decide whether it blocks the release |
| R4 | Store structure: delete the rotating-buffer path (`--kvq-layer-bufs N`), the pinned mirrors of the resident rows (about 272 MiB pinned RAM per sequence), `LLAMA_KVQ_UP_REPEAT` as a base knob (our only PCIe 3.0 / 4.0 check on the 5090); maybe split `llama-kvq-store.cpp` (now 2,395 lines: constructor, `copy()`, `enable()` mix host store, codes, resident, ring, scratch) | open | Undecided; both rotating-buffer flag and env still exist. The mirrors are now per sequence (about 272 MiB pinned per stream), which makes the decision larger. The split is deferred until the design settles; a map comment at the top of the file is optional |
| R5 | Review of the remaining memory optimizations | open | Items in 9.5; plus the unmeasured pool / cuBLAS remainder (100 - 250 MiB), the host side from the 5070 Ti field report (codes host copy 1.8 GB, checkpoints about 0.27 GB each up to 32, `-lm none` default for docs). The host-side codes copy item is partly done: pages are on demand and given back (3.3) |
| R6 | CUDA side stream and events of the kvq attention (`kvq-attn.cu`) | open | Documented limit: one kvq context per GPU per process. The side stream is stream slot `GGML_CUDA_MAX_STREAMS - 1` taken without reserving it; the fork / join events are function-static per device (`ev_fork`, `ev_join`). Whether the graphs are captured as CUDA graphs is not verified. Fix: per-context events and an owned side stream (about 30 - 40 lines; gate KVQ_ATTN with CUDA graphs on and off); required before any upstream PR |
| R7 | The table fit on the GPU | done | 9f9d1e082 (2026-11-20). KL 6.36e-2 against 6.40e-2, end-of-prompt fit 0.098 s at 24k, prefill +1.4%, VRAM -88 MiB, 250k all 16 layers in one group, follow-up 0.95 against 1.44 s, acceptance set equal. Open follow-ups of the note: VRAM per phase after R7 was measured (7.1); the prefill cost (+1.4% at 24k, the GPU work no longer overlaps) could be cut by the TABLE op on a side stream or tables of 4 layers in one op (not done) |

### 9.2 Multi-sequence (design of 2026-11-20, steps 1 - 4)

| Item | Status | Now |
|---|---|---|
| G1 `-np N` served at once; pool of pages; per-sequence state; host RAM on demand and given back; ring streams; `fit_wait(seq)`; server `-np N`; packed decode of several sequences; op stream dimension | done | 3.1 - 3.8, 7.9 |
| Reserve of the packed decode memory and graph at load (step 3c) | in progress | 3.7, TODO |
| Step 4: shared prefix: `seq_cp`, `seq_keep(seq > 0)` with sharing, refcounts (holder count is 0 / 1 today), copy-on-write of one partial page (a `seq_cp` up to p1 inside a chunk: copy, then roll dst back to p1 so the copy happens in one place), server `n > 1` completions (refused today with an invalid request error), full-context state with shared pages (sharing survives a save) | open | Gates planned: fork at the end and in the middle of a chunk equal to the same sequence processed alone, pool accounting returns to empty. `--kvq-rem merge` must then be refused with N > 1 (a merged group rewrites the tables of the last complete chunk and complete pages must never change); today it is accepted |
| State format version 4 (per-sequence pages in logical order, full state with page tables) | stale | The format stayed at version 3 and is written per sequence from the pages (the host bytes are the same); a full-context state with several sequences and shared pages has no design yet because nothing is shared |
| Drafter (draft model) cache full aborts all slots of the batch | open | The drafter context is a plain cache of `-c` positions in total, not the pool: when it is full the slots of the batch stop. The pool-full path only covers the target |
| `cache_idle_slots` off with `--kvq` | open | Set off in `server_kvq_prepare_params`: idle slots are not saved to the prompt cache and not cleared when a task starts; with several slots a full pool purges idle slots. Per-sequence state exists, so saving idle slots could be enabled |
| D3 mixed traffic: a prompt of one slot with other slots decoding keeps the codes on the device; the codes-out phase is `-np 1` only | done | The peak is higher (7.9); option (b) exclusive prompts (other slots stall during a prompt) not built; option (c) only the pages of generating sequences on the device (a device page pool) not built; both only if 250k with several slots is a real need |
| Several sequences in the packed ubatch: at most 8 streams, the same token count per stream, at most max_batch = 8 tokens | done / limit | With `n_seq_max` > 8 only 8 sequences share one ubatch (`n_str = min(n_seq_max, 8)`); sequences with different counts are decoded in separate ubatches |
| `--fit` unsupported, one kvq context per GPU per process | open | R2, R6 |
| Quality and speed of packed decoding on slow PCIe (`LLAMA_KVQ_UP_REPEAT`), at 4 and 8 tokens per batch (the owner's target-batch rule) | open | Throughput is measured at PCIe 5.0 only (7.9). The per-stream resident rows are uploaded from per-sequence mirrors; the cost per extra stream on a slow bus is unknown |
| Milestone suite per step | done | Sessions at 20k / 50k / 100k / 250k after each step; ids identical since 8b6ac9880 (7.3) |

### 9.3 Known limits and costs (guide section 8 and old "Gaps before this can replace production")

| Item | Status | Now |
|---|---|---|
| Not supported: `seq_cp`, middle `seq_rm`, `seq_add` / `seq_div`, multimodal and embeddings, `--fit`, more than one kvq context per GPU per process, non-CUDA backends | open (limits) | Abort messages start with `not supported by the two-tier KV cache (--kvq)`. "One sequence only" is stale (3) |
| State of the K/V outside the fit mode errors; a plain state into a kvq context and the other way round errors; the `ON_DEVICE` state flag on a kvq state errors | open (documented) | Fit mode is the only mode now (`--kvq-tail 0` is gone), so the first part is stale; the other two are limits; there is no migration from plain snapshots; the launcher's snapshot directory hash does not know `--kvq` and kvq snapshots need their own directory (outside this repository) |
| The graph is re-planned at almost every prompt ubatch (number of key blocks follows the stored rows) | open | The buffer does not grow; the cost is inside the prompt times |
| The follow-up prompt pays the codes upload (57 ms at 250k) | stale for the default | Only with `--kvq-prefill-free-codes` (opt-in since 1d0b92481); a per-layer overlapped upload hid nothing and was dropped. The prompt-size policy (9.5) would remove it |
| Host memory: 8.5 GB pinned store + 1.8 GB codes at 256k per context | partly done | On demand and given back with `-np N` (3.3); the pool total is ceil(n_ctx / C) + n_seq_max - 1 pages and a long request still pins its pages; the ring is 144.5 MiB per sequence; the mirrors about 272 MiB per sequence; plus 27 MiB per checkpoint |
| Pool full | done | 3.2, 3.8 |
| Quality measured on one 100k trace for the final library path and four shorter ones; the wide test (42 requests, to 121k) was on the prototype fa865aee3; nothing at 150k or above; the 250k sessions check ids and speed only | open | A wider KL check at 150k+ with blocked teachers is needed |
| Speed of the final build at 24k / 50k / 100k / 250k | done | 7.3 (suite 2026-11-20); not measured with several slots at 150k+ |
| Slot save and restore and the RAM prompt cache at 150k+ (about 10 GiB per state at 230k, estimated) | open | Checked at 24k and 100k only |
| Reproducibility | done | 7.5; no job starts during generation; folds at positions set by the token count |
| Batch independence only for the same batch boundaries | limit | Inherent in the model's batch-shape rounding |
| With the ring of one batch a long reply folds once per ring size of tokens | partly done | The old figure (0.57 s per fold, CPU fit, 4 threads) is stale: with the GPU fit a fold takes about 0.09 s at 24k; not re-measured at 100k+ |
| The slots of a batch are placed for the whole batch at once, so the ring cannot be smaller than one batch (per-ubatch placement would allow n_ubatch + 256) | open | Not built |
| Fitter device work runs on the main stream between ubatches (the old 570 MB of host uploads per chunk) | partly done | Still on the main stream; the work is now graphs in the shared memory (R7); the upload volume was not re-measured |
| ggml changes (new ops, the scheduler callback order, `ggml_gallocr_resize`, CUDA flash-attention kernels, GDN replay merge); rebases conflict there | open | The branch sits on ggml-org/llama.cpp master 951d465f5 plus the merge of the WIP PR 31669: check the upstream state of the PR (the stored layout of the pending inputs may change and the final PR will have to replace the merged one) |
| No automated tests of the store, the fitter, the ring cache and the server flows; no long session (hours, many rollbacks and refits) | partly done | Op, codec, fit ops and multi-stream tests are in `tests/`; the store and the server flows are covered by the gate scripts in `kvq-scratch/ms` (kvq-eval multi-seq gates, the suite, rollback / rm scripts), not in the repository |
| The code was written by agents and reviewed at the level of reports and diffs | open | Needs a real review before anyone depends on it |
| What to do with the freed VRAM (a longer context, a larger quantization) | open | Not decided |
| Deferred tuning: top-up rules and thresholds, prefetch size | stale | Top-up modes are removed; lead 3 and the resident capacity were measured (5.1) |

### 9.4 Open bottlenecks (old list, largest first, reviewed)

| # | Item | Status | Now |
|---|---|---|---|
| 1 | Prefill against production: 44.93 against 44.01 s at 100k (0.55 s the end-of-prompt wait) | stale | The wait is 0.1 s now and the blocked attention made the prompt 13 - 30% faster than the build before; not compared to ggml-org's plain cache since |
| 2 | End-of-prompt fit 0.55 s whatever the row count | done | R7: 0.098 s at 24k, a 501-token follow-up at 250k 0.95 s |
| 3 | Job latency set by the polls, CPU tasks | stale | There are no CPU tasks; one graph per layer group and a waiter thread |
| 4 | Target compute buffer of the prompt phase (1.81 GB at 256k) | done | Blocked attention, phases, codes out: 7.1 |
| 5 | MTP context cache 899 MiB f16 for the full context | open | With the DFlash2 drafter the drafter KV is 60 MiB (the configuration measured); for `draft-mtp` it is still a plain f16 cache |
| 6 | Verify batches of 8: about 4 ms per round that do not shrink with the context (21.6 against 17.45 ms at 24k): `kvq_precise` over 8192 slots, `kvq_final`, the early-q projection, the scatter, the trigger splits and 10 top-k kernels per layer | open | Kernel work done (7.10); the trigger splits (0.39 ms idle per round) need a new ggml op (device-to-host copy of the selection and a host function node) |
| 7 | Tables 243 MiB at ctx 230000; 16k chunks would halve them | open | Costs a little quality (mean ratio 1.12); not planned |
| 8 | Overlap of the precise and coarse kernels: with int8 tables the op is 12 us slower per layer (the precise rows on the side stream take the SMs first) | open | Fragile; forking after prep made it worse |

### 9.5 Ideas, in the owner's order

1. Scratch as a ring of 16k-row blocks fed on the copy stream (540 -> about 105 MiB in both phases; the one item that moves the decode phase, which sets the headroom in configuration A; 16 blocks x 16 layers = 256 waits per ubatch against 16, a block is 1 ms on the bus against about 2.4 ms of attention at ub 1024; the masked tail needs a small buffer of about 20 MiB; 10 - 20 ms of alloc / free per phase switch is acceptable). Open (R5-1).
2. Prompt-size policy for follow-up prompts (after item 1): layer input buffer sized by the batch (no setting; a 1k follow-up needs 100 MiB, not 400); codes and tables kept on the device when a request has fewer than N new tokens (a server setting, about 4k, safe at 250k only after item 1: 16,036 MiB for a 4k follow-up today, 15,736 for 1k); ubatch 4096 as a speed knob only (about 2%). Open (R5-2). Partly overtaken: the default now keeps the codes on the device (`--kvq-prefill-free-codes` is opt-in).
3. For huge requests, run the prompt through two halves of the layers in turn, moving weights and cache between RAM and VRAM. Open (R5-3).
4. Optional, from ggml-org: PR 31802 (fused q8_0 dequantization in the MMA flash attention kernels) would remove the f16 K/V copies of the prompt attention; its base is far from ours. Open (R5-4).
5. Selection steps on a side stream (nothing built): the early-query projection, the SCORE pass, `kvq_score` and top-k (about 4 ms at 8 tokens and 100k) beside the GDN kernels; limited by the worker window (0.5 ms of 0.56 ms) and by kernels slowing each other. Open.
6. More than two rotating buffers, the union bitmap and compaction on the GPU, a device row count for the scatter: small gains (about 13 MiB), need ggml changes. Open.
7. The TABLE op on a side stream or tables of 4 layers in one op to cut the prefill cost of the GPU fit (+1.4% at 24k). Open.
8. Coding new rows on append (investigated, not built): the packed graph has n_coarse in its shapes, so every appended row rebuilds the graph; the ring floor is set by one prefill batch, so it only removes the folds of a long reply. Not planned.

### 9.6 Upstream candidates to ggml-org/llama.cpp (not release blockers, not scheduled; nothing is sent without the owner's approval)

| # | Candidate | Where | Status |
|---|---|---|---|
| U1 | Flash attention with the log-sum-exp output (`ggml_flash_attn_ext_lse`), CUDA | `kvq` e6035d123 | open; needs U2 in the same PR; a review may change the op API |
| U2 | Refuse the LSE output in the backends that do not write it | `kvq` 41454e08f | open; 11 backends, the non-CUDA ones never compiled here; the meta backend was added (57279c6b6) |
| U3 | GQA kernels of the MMA flash attention without a mask | `kvq` 11af7a218 | open; the most independent one, also helps ggml-org's normal path |
| U4 | Blocked causal prefill for the plain KV cache (mask-free key blocks + a small masked tail, merged via U1) | idea; the kvq scratch attention does it | open; judge quality with blocked teachers (7.7) |
| U5 | Compute buffers per phase (`ggml_gallocr_resize`, `ggml_backend_sched_resize_buffers`, cparam `phase_bufs`) | `kvq` 04ff3e772, b79454267, 74db3fd67 | open; the opt-in change is done (f3473da6e) |
| U6 | Early copy callback of the scheduler | `kvq` 56d411694 | open; only useful with a consumer like the kvq store, low priority |

### 9.7 Not exercised (reviewed)

| Item | Status |
|---|---|
| Two-tier without MTP; drafts of 7 tokens on the packed path | done: DFlash2 n_max 7 and 3 and no drafter are all measured (7.9) |
| `draft-mtp` with `--kvq` (R3); model reload after sleep; a chat-endpoint conversation with user-message checkpoints | open |
| Save and restore: a load into a smaller context or ring than the state needs; the full-context state API (`llama_state_save_file`); a sequence that ends exactly at a chunk boundary (empty ring, handled by `seq_pos_max`); the launcher with `--kvq` | open |
| Top-up modes, `--kvq-tail 0` and other tail sizes, the tool-driven packed modes of kvq-eval, the offline V path | stale: removed |
| Lead 1 after the async prefetch change | open |
| MoE models with the early copy callback | open (only qwen35 is supported) |
| Several sequences: `n_seq_max` above 8, a mix of sequences with different token counts in one batch, `-np` > 1 with the MTP drafter, with `--kvq-rem merge`, with slot save and restore under load | open |

## 10. Material outside the repository

`../kvq-scratch` (next to the checkout): `gpurun.sh` (GPU lock for parallel agents), `tok/` (token files of private traces: never print or decode them; `cat281k.bin` = 281,551 tokens for 250k prompts), `ref/` (reference logits), `docs-archive/` (`kvq-cache-comparison.md` with the four caches compared, `dflash2-vram-baseline.md` with the launch line and acceptance set), `mem/` (one folder per work package of 2026-11-17 to 2026-11-19, each with `RESULTS.md`: `inventory`, `vram`, `phase`, `replay`, `m250`, `comb250`, `codes`, `attnsplit` with `QUALITY.md` and `DIAG.md`, `final`), `ms/` (the multi-sequence work of 2026-11-20: `base`, `s1a` .. `s1c`, `s2a` .. `s2e`, `s3a`, `s3b`, `s3c`, the `suite-*` sessions at 20k / 50k / 100k / 250k and the gate scripts `tests.sh`, `eval24.sh`, `srv24.sh`, `rmtest.sh`), `MULTISEQ-DESIGN.md` (the design with the owner's decisions), `CLEANUP-PLAN.md`, `RELEASE-BLOCKERS.md` (now a pointer to section 9 of this file). Research background, codec definition and the original plan: the `qwen3.8-exp` repository (`notes/kvq_implementation_plan.md`, `notes/kv_prefetch_results.md`, `src/qexp/kvcodec.py`, `experiments/14_kv_prefetch.py`). The token files come from private data: never print or log prompt text, generated text or decoded tokens.
