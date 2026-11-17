# Two-tier KV cache for qwen35 (kvq) - prototype state

State of branch `kvq-two-tier-cache` at commit cefce74bc (2026-11-13), with the GPU fitter (2026-11-16) on top of 7604e664e, and the ring redesign (2026-11-16 evening, on top of fa865aee3, not committed when this was written: the rows after the last complete chunk are fitted at the end of every prompt, the ring holds only rows that are not coded yet). On top of 209ab7775, not committed (2026-11-17, worktree `kvq-wt-comb`, branch `kvq-comb`): the whole union of the verify batch and the attention op kernel changes (see Results). The CPU fitter is kept on branch `kvq-fit-cpu`. The branch starts from ggml-org/master 951d465f5 and is pushed to the JetBrains-Hardware fork. This file is the starting point for the next work session.

Research background, codec definition and the original plan are in the `qwen3.8-exp` repository: `notes/kvq_implementation_plan.md`, `notes/kv_prefetch_results.md`, `src/qexp/kvcodec.py`, `experiments/14_kv_prefetch.py`.

## What it is

A KV cache for Qwen3.8-27B (GGUF arch `qwen35`) that keeps most of the attention K/V out of VRAM:

- Coarse tier, VRAM: every row of a fitted chunk as transform-coded low-bit codes (K 2 bit, V 1 bit plus a gain). The attention is computed on the packed codes by a custom op.
- Precise tier, system RAM: the q8_0 rows of the same positions in a pinned host store.
- Resident set, VRAM: 8192 precise rows per (layer, kv head) (8 positions x 1024), selected per position by an early query and fetched from RAM while the GPU runs the two layers before the attention; every position of a packed ubatch has all of its own top 1024 rows resident.
- Ring, VRAM: q8_0 rows of the positions whose rows are not in the host store yet: the reply since the end of the last prompt, and the rows of a prompt while it is processed (ring-shaped device cache, `--kvq-tail`, default one batch + 256 cells, at least 4352).

Model facts used everywhere: 64 trunk blocks, 16 attention layers (il = 3, 7, ..., 63), 24 query heads, 4 kv heads, head dim 256. The MTP block (block 64) runs in its own context with its own plain cache and does not use any of this.

## How it works

Chunks. The sequence is cut in chunks of C positions (`--kvq-chunk`, default 8192): chunk c covers [c*C, (c+1)*C). A complete chunk is fitted alone: a K table and a V table per (layer, kv head) on the chunk's own rows; its rows are coded and join the coarse region. A fit is always made from scratch from its rows (q8_0, from the host store or the ring) and its statistics, so its tables do not depend on how often or when the range was fitted before.

The rows after the last complete chunk (`--kvq-rem`). At the end of every prompt (`llama_memory_kvq_fit_wait`, the server calls it when the prompt is done) the fitter fits the layout of the token count S, waits for all of it, and nothing of the fit runs afterwards. With n_full = floor(S/C):

- `own` (default): chunks 0 .. n_full - 1 alone, then [n_full*C, S) as a chunk of its own if it has at least `--kvq-min-rows` rows (default 256); fewer rows stay precise in the ring.
- `merge`: chunks 0 .. n_full - 2 alone, then [(n_full - 1)*C, S) as one group (C to 2C - 1 rows) with one table set, written to the table slots of both chunks (the attention op takes the table of a row by its chunk, so no kernel change). When the next chunk is complete, the head of the group is fitted alone again.
- n_full = 0: [0, S) as one partial chunk, both policies.
- A coarse region smaller than prefetch_n + n_sink (1028) is not made (the prefetch selects from its rows); those rows stay precise.

At the next end of a prompt the partial chunk or group is fitted again with all rows it has then, from the host store. The rows of a reply stay precise in the ring until the next end of a prompt. The resident set is emptied at the end of every prompt, so the decoding after it depends only on the memory state (also after a slot restore).

Three boundaries. n_coarse: positions [0, n_coarse) are coded (any row, not a chunk multiple). n_host >= n_coarse: positions [0, n_host) have their q8_0 rows in the host store. The ring (device q8_0 cache, `--kvq-tail`) holds the positions [n_host, end), position p in cell p mod size; its cells are freed as soon as a job has started on their rows, not when the job ends. The packed attention (small batches) reads coarse rows + resident rows + the ring, so it needs n_host == n_coarse; otherwise a small batch attends over the scratch like the prefill (correct, slower; only after a rollback or a spill before the next end of a prompt).

Tables. Per (chunk slot, layer, kv head, K or V) a 256 x 256 inverse table and a small aux array (step, mu, bits). The inverse table is stored as int8 with one scale per component c (row inv[c][:] divided by its max abs, times 127); the scale is folded into the step that the attention op reads, so the decode formula is unchanged. The codec itself (fit, encode, bits, the step of the device CODE mode) still works in float. There are ceil(n_ctx/C) slots and the coarse region can reach n_ctx rows (before: floor(n_ctx/C) full chunks only).

Fitter (`llama_kvq_fitter`). A job is a row range [r0, r1) with one table set (a chunk, a partial chunk or a group). The work on the rows runs on the GPU, on the main stream between two ubatches; only the 256 x 256 problem per (layer, kv head, K or V) runs on CPU threads. One graph per attention layer and phase, the rows of a layer go through a device stage (q8_0, C rows; 2C - 1 rows with `merge`):

1. Phase 1, when the job starts: rows [r0, n_host) are uploaded from the host store into the stage, the ring rows [n_host, r1) are copied into it on the device and from there to the host store (one download per (layer, kv head)); mean and covariance (`ggml_kvq_fit`) of the stage rows go to the host. n_host becomes r1 and the ring cells are free.
2. CPU tasks (`--kvq-threads`, low priority), the V tasks first: metric (from the statistics), `llama_kvq_fit_cov` (Cholesky, `L^T cov L`, eigen solver, fwd/inv tables), for K also bits, steps and the packed table. The tasks wait for the event of phase 1, not the decode thread.
3. Phase 2, at the first poll after the V tasks: the rows again from the host store (K and V, 285 MB per 8k chunk), V fwd tables up, rows rotated, the scale of the shape to the host. CPU tasks for V: bits, steps, packed table.
4. Phase 3 and install, at the first poll after all tasks: the rows again from the host store, fwd tables up, rows rotated, K and V codes packed straight into `kc` and `vc`, the tables uploaded to every slot of the range, `set_n_coarse` (one synchronize per job). A range that is coarse already gets its new codes and tables in the same poll, so no graph sees a mix.

When jobs start. Prefill (ubatches larger than max_batch): a complete chunk is taken at the first ubatch after it is complete (`merge`: once the next chunk is complete too). Generation (small ubatches): no job starts; the jobs in flight are waited for before a small batch, so the path of the small batch (packed or scratch) does not depend on the fitter threads. At the end of a prompt: the layout as above, sync.

Statistics. K metric: second moment of the queries summed on the device during decoding (`qacc`), a ubatch counts for the chunk of its first position; when a chunk is complete its sums are read and kept on the host (for its fit and for a rollback into it) and the device sums start again from 0. A group uses the kept sums of its complete chunk plus the device sums of the open chunk; an own partial chunk also takes the kept sums of the chunk before it (its own sums can cover very few positions: in a server session the 1024-token ubatch that crossed the boundary counted for the previous chunk and the partial chunk of 152 rows had the queries of 4 positions, the metric was indefinite and the fit failed; the queries just before stand in). The device sums of the open chunk are read without a reset and kept with their position; a rollback to exactly that position restores them, so a reply that is rolled back leaves no queries behind. V metric: o_proj columns scaled by the mean sigmoid(gate), summed the same way (`gacc`).

Ring overflow. Before a batch is placed (`tail_wait`), the ring needs room for it. A large batch: first the jobs of the complete chunks start (their ring cells are free at once), then the remaining ring rows before the batch are copied to the host store without a fit (spill). A small batch (generation): when p0 + max_batch - n_host > ring size, the rows after the last complete chunk are fitted in line on the decode thread (fold: the layout of p0, sync, then the batch); the position of a fold depends only on the first position of the batch and the state, so generation stays reproducible.

Generation and MTP verify (ubatch <= 8 tokens), per attention layer il:

1. At the input of layer il - 2 the graph computes layer il's query from the earlier hidden state, scores all coarse rows, and takes the top 1024 per (kv head, position).
2. A scheduler split with a copy callback queues the readback of the selection (the row indices only; the scores only when the request can be cut, see below) on a copy stream. A worker thread plans the resident slots and gathers the rows from the host store. The GPU keeps running layers il - 2 and il - 1.
   - The request of a kv head is the union of the selections of all positions of the ubatch, deduplicated in a bitmap of n_coarse bits (set the bits of the L lists, the word scan gives the distinct rows in row order; and-not with the bitmap of the resident rows gives the missing rows).
   - A slot is free if its row is not in the request. The missing rows (in row order) go to the free slots of the oldest requests (a per-slot stamp of the last request that listed its row; age in requests, 255 and older and empty slots count as 255), ties by slot. So the row of every slot depends only on the sets, not on the order the GPU wrote the selection or on thread timing.
   - The default capacity, `--kvq-resident-cap 0` = max_batch x prefetch_n = 8192, holds the whole request of the largest packed ubatch, so nothing is cut. With a smaller capacity a request of more distinct rows than slots is cut to the rows with the largest max score over the positions (ties by row, the scores are read back for this case only; logged once at enable, counted as `pf_cut` / `pf_lost`).
3. Before layer il's attention a join uploads the rows and scatters them into the resident tensor.
4. `ggml_kvq_attn` computes the attention over coarse rows + resident precise rows + the ring.

Prefill (larger ubatches), per attention layer: the rows [0, n_host) are uploaded from the pinned host store into one device scratch buffer (4 async uploads on the copy stream, overlapping the layers before), the ring rows after them are copied in on the device, and normal flash attention runs over the scratch. Logits are bit-identical to attention over a full device cache.

Rollback (`seq_rm` of the end at p0). Inside the ring (MTP drafts): only the ring cells. Into the coarse region: the jobs with rows at or after p0 are cancelled, the coarse region drops back to the chunk grid (n_coarse = floor(p0/C)*C), n_host = p0: the rows before p0 stay in the host store, nothing is copied back. The next end of a prompt fits them again (they are prefilled over the scratch until then). If the layout at the end of a prompt leaves rows precise that are only in the host store (fewer than min_rows), they are copied back to the ring then. `clear` drops everything and the fitter starts again.

Save and restore (`llama_memory_hybrid_kvq::state_write` / `state_read`, fit mode with the tail ring only). The state is exact: nothing is fitted again on load. A save waits for the jobs in flight (there are none after the end of a prompt) and does not fit anything. Layout of the state of sequence 0 (the full-context state is the same with the arch string in front), version 2:

1. Magic `KVQS` and version 2, then the meta of the store (attention layers, kv heads, head dim, chunk rows, words per K / V row, n_coarse, type and slice size of the two table tensors, n_host) and of the fitter (K / V bits, V fitted per chunk or from a file). The loader checks all of it before it reads any data. A version 1 state fails with "version 1 of the two-tier cache state, expected 2", the sequence is empty afterwards.
2. The ring: the plain state of the attention cache (the positions after n_host), read back with `find_slot`, which places a position at cell `pos mod size`.
3. The recurrent state, plain.
4. Per attention layer: the q8_0 host rows [0, n_host) per kv head (raw blocks from and into the pinned store), the codes (`kc`, `vc`) of [0, n_coarse) per kv head, the tables of the ceil(n_coarse/C) slots of the coarse region (by the tensors' own type and size, so a change of the table format is a mismatch and not a misread).
5. The fitter: the fitted range of each slot of the coarse region (so a partial chunk and its row count round-trip), the statistics of the complete chunks, the device sums of the open chunk, the kept sums of the last partial fit and their position.

A plain state starts with the stream count of the attention cache, so a plain context refuses a kvq state ("n_stream mismatch") and a kvq context refuses a plain one ("not a state of the two-tier cache"); after a failed load the sequence is empty. The resident set starts empty after a load. The server's slot file appends the context checkpoints and the MTP draft state as before. The RAM prompt cache serializes the same state into RAM (an entry over `--cache-ram` is skipped).

## Code map

| Part | Files |
|---|---|
| Codec (fit, encode, decode, tables) | `src/llama-kvq-codec.{h,cpp}`, `src/llama-kvq-pack.{h,cpp}` |
| Attention op, CPU reference and CUDA | `ggml_kvq_attn` in `ggml/include/ggml.h`, `ggml/src/ggml-cuda/kvq-attn.{cu,cuh}` |
| Fit op (mean, covariance, rotation, V gain statistics, code and pack), CPU reference and CUDA | `ggml_kvq_fit` in `ggml/include/ggml.h`, `ggml_compute_forward_kvq_fit` in `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/kvq-fit.{cu,cuh}` |
| Host store, resident cache, prefetch worker, scratch uploads | `src/llama-kvq-store.{h,cpp}` |
| Fitter (device phases, CPU tasks) | `src/llama-kvq-fit.{h,cpp}` |
| Memory (hybrid subclass), ring-shaped device cache | `src/llama-memory-hybrid-kvq.{h,cpp}`, `src/llama-kv-cache.{h,cpp}` |
| Graph | `src/models/qwen35.cpp` |
| Scheduler: copy callback before the wait | `ggml_backend_sched_set_copy_callback_early` in `ggml/src/ggml-backend.cpp` |
| Internal API | `src/llama-ext.h` (`llama_kvq_params`, `llama_kvq_fit_params`, `llama_memory_kvq_*`, `llama_model_set_kvq_*`) |
| Server options and forced settings | `common/arg.cpp`, `common/common.cpp`, `tools/server/server.cpp`, `tools/server/server-context.cpp` |
| Evaluation tool | `tools/kvq-eval` |
| Tests | `tests/test-kvq-codec.cpp`, `tests/test-kvq-attn.cpp`, `KVQ_ATTN` and `KVQ_FIT` cases in `tests/test-backend-ops.cpp` |

## How to run

Server, production configuration plus `--kvq`:

```
llama-server --fit off -m Qwen3.8-3.6-27B-blend-Q4_K_M.gguf --ctx-size 230000 --n-gpu-layers all --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16 --batch-size 4096 --ubatch-size 1024 --kvq
```

Options: `--kvq`, `--kvq-threads N` (CPU threads of the table fit, default 4, at most the physical cores), `--kvq-prefetch-n` (1024), `--kvq-resident-cap` (0 = 8 x prefetch_n = 8192, the whole request of a verify batch of 8), `--kvq-lead` (2), `--kvq-tail N` (ring cells, default max(n_batch, 4096) + 256 = one batch, 0 = full device cache; not below n_batch + 256), `--kvq-chunk N` (rows per chunk, 8192), `--kvq-rem own|merge` (the rows after the last complete chunk, default own), `--kvq-min-rows N` (own: fewer rows after the last complete chunk stay precise, default 256), `--kvq-v-tables FNAME` (offline V tables, only for comparison).

With `--kvq` the server forces, with a warning: 1 slot, context shift and cache reuse off, flash attention on, q8_0 K/V, draft n_max <= 7; with `--kvq-tail 0` also the RAM prompt cache off and the slot save path cleared. At the end of each prompt the server waits until the prompt is fitted up to its end (log line `kvq: waited ... at the end of the prompt, coarse region N rows`); no fit work runs during the reply. It refuses to start for a model that is not qwen35, `--mmproj`, embeddings. Attention rotation of quantized caches is turned off for the model (also for the MTP context). `--fit` does not count the kvq memory.

Evaluation tool (teacher-forced KL against saved reference logits):

```
llama-kvq-eval -m MODEL -ngl 99 -fa on -fit off -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --kvq-tokens TOKENS.bin --kvq-prefix-len S --kvq-chunk 1 --kvq-ref REF --kvq-lib
```

`--kvq-lib` is the library path (same as the server), `--kvq-lib-full` keeps the full device cache, `--kvq-ring N` sets the ring (default chunk + batch + 256), `--kvq-rem` / `--kvq-min-rows` as in the server, `--kvq-rm-test` checks the removal cases (both policies), `--kvq-turns S1,S2,... --kvq-turn-gen N` processes the prefix in turns (prefill to S_i with batches at multiples of the batch size, the end of the prompt, N greedy tokens one per decode, back to S_i through a recurrent checkpoint, the next turn), `--kvq-ref2` compares with a second reference (e.g. the single-prefill run), `--kvq-scr-check N` compares scratch and cache prefill logits bit for bit. State checks: `--kvq-state-save FILE` (after the prefill and the wait), `--kvq-state-load FILE` (instead of the prefill), `--kvq-state-ckpt P` (a recurrent checkpoint at P; after the save or load, roll back to P and process the rest of the prefix again), `--kvq-lib-hold` (no chunk joins during the scoring). Compare two runs with `--kvq-save-ref` files. The older `--kvq-variant`, `--kvq-two-tier`, `--kvq-packed` modes are tool-driven experiments from earlier stages.

Environment variables for debugging: `LLAMA_KVQ_SCHED_SYNC`, `LLAMA_KVQ_GATHER_THREADS`, `LLAMA_KVQ_IDLE_PROBE`, `LLAMA_KVQ_SCRATCH`, `LLAMA_SERVER_GEN_TIMES`, `LLAMA_KVQ_FIT_SYNC` (synchronize around the device phases of the fitter, logs the device time per job and phase). `LLAMA_KVQ_FOLD_EVERY=N`: a fold whenever N reply rows follow the coarse region; kept because it is the only way to exercise folds in kvq-eval, whose replies (about 500 tokens) never fill the ring (the fold determinism check uses it). Removed after the measurements: `LLAMA_KVQ_FOLD_FIXED` (the fold coded the new rows with the tables of the last fit, the append experiment in Results) and `LLAMA_KVQ_DIAG_NORESET` (no reset of the resident set at the end of a prompt). kvq-eval `--kvq-rb-test`: ring rollbacks at C + 501 with and without an earlier rollback, and the same positions in two batch splits without a rollback, also on a plain cache (`--kvq-no-rot` without `--kvq-lib`); `--kvq-turns` also works on a plain cache.

## Results

Machine: RTX 5090 32 GB, Ryzen 9950X3D, Windows 11, CUDA 12.9. One trace of 100k tokens unless noted; treat single runs as such.

### Attention op kernels at verify batches of 8 (2026-11-17)

Target IQ3_S with the DFlash2 drafter, n_ctx 102400, restored 100k slot (99,395 coarse rows), on top of the whole union below (8192 resident slots). All changes are in `kvq-attn.cu` and bit-identical: the op test outputs and the hashes of 24 large problems (`test-kvq-attn --hash`, 20k / 100k rows, capacity 1024 and 8192) are equal, kvq-eval logits from one saved state are equal at 8, 4 and 1 tokens per decode, and the server ids are equal.

- Coarse kernel in column groups (`kvq_launch_coarse`): at 48 columns ATTN as 3 x 16 (168 registers, no stack; before 255 registers and a 72 B stack), SCORE as 2 x 24, at 32 columns SCORE as 2 x 16. The groups of one launch take the same rows in adjacent blocks, so the codes are read once from DRAM.
- `kvq_score`: 2-byte loads of rows 96 B apart thrashed L1 at 48 columns; the score scratch has rows of N halves now and a thread reads its row in 16-byte loads.
- `kvq_prep_c` (from 16 columns): one thread per component, 8 or 4 columns per block, the same summation order as `kvq_prep` (before, the prep of 48 columns had 222 registers, 96 KB of shared memory and one block per SM).
- `kvq_final`: the loads of 32 precise partials are issued before their sums (same order). Against the plain loop at 8192 slots (same binary, alternating sessions): one token per decode 0.17 ms faster (6 of 6 pairs), n_max 3 0.20 ms, n_max 7 0.45 ms; isolated op 9.2 / 9.9 / 16.5 us against 28.3 / 32.7 / 55.3 us at 6 / 24 / 48 columns. Batches of 16 were 1 - 2 us slower than 32. (At 1024 slots the change looked like +0.1 ms at one token per decode, inside the noise.)
- `kvq_precise`: the columns in groups of 8, 4 warps per group, all groups of a block share the K and V tile (each group computes its columns as a block of 8 columns would, so the sums are the same); K is fetched (cp.async into the q8_0 staging) while q is staged, V while K . q is computed. Per call at 8192 slots: 52 us at 48 columns, 38 us at 24 (76 us before), 30 us at 12 (48 us), 36 us at 6 (37 us). The overlap of the loads alone gave 1 - 2%; the column groups give the rest (more warps per SM: the block of 84 KB of shared memory is one per SM). Per round at 8 tokens 3.72 -> 1.24 ms, at 4 tokens 1.96 -> 1.05 ms.
- Not kept: the q8_0 rows through registers without the staging buffer (so that two blocks fit per SM). The SM has 100 KB of shared memory (1 KB reserved per block), so two blocks fit only up to 16 columns; that variant took 77 us at 6 columns (255 registers with spills) against 36 us, 48 us against 38 us at 24 columns. The diff is in `kvq-scratch\mem\comb\kvq-attn-p2b-nost.cu`.

Per round (nsys, ms, 4 tokens / 8 tokens per verify, final build): early-q projection 0.457 / 0.451, prep 0.352 + 0.345 / 0.445 + 0.452, SCORE pass 1.082 / 1.971, kvq_score 0.156 / 0.370, top-k 0.597 / 0.735, ATTN pass 2.100 / 2.978, kvq_precise 1.046 / 1.242, kvq_final + merge + combine 0.453 / 0.764, scatter 0.378 / 0.646, idle at the trigger splits 0.371 / 0.390, round 20.53 / 26.24. Only kvq_score is above 2x from 4 to 8 tokens (2.37: its work is per row and column, and its scratch rows are twice as wide). Before the kernel changes (209ab7775 code with the whole union): ATTN 5.43, SCORE 2.73, kvq_score 1.34, the two preps 1.90 ms at 8 tokens.

Server, decode ms per verify batch (2 - 6 sessions each, `LLAMA_KVQ_IDLE_PROBE=1`): n_max 7 25.96 - 26.43 (whole union only: 33.34 - 33.69 at cap 1024 / 32.7 at 8192 in the union tree; plain cache 26.3 - 27.8 from the plain slot, 26.08 after a real prefill), n_max 3 20.33 - 20.89 (before 21.4 - 21.9, plain 23.6 - 25.4), one token per decode 14.77 - 14.86 (before 14.73 - 14.89, plain 15.6 - 15.7). Real prefill, n_max 7: 25.95 ms, 146.7 t/s against plain 26.08 ms, 143.2 t/s; prompt 43.1 s against 42.0 s. GPU idle at the joins 0.02 - 0.09 ms. VRAM in generation 17,820 MiB (dedicated usage of the process), the same as the whole union alone.

Shorter contexts: the two-tier cost does not shrink with the context. 24k prompt (real prefill, n_max 7): 21.6 ms per verify batch against 17.45 ms for plain. Acceptance set, ms per verify step with the drafter (two-tier / ggml-org master a49db739f): 25.1 / 22.0 at 12k, 25.8 / 21.3 at 24k, 26.6 / 24.0 at 50k, 29.3 / 26.6 at 65k, 30.0 / 30.2 at 99k. The fixed part per round is about 4 ms: kvq_precise over the 8192 resident slots (1.2 ms at any context; at 12k they are 70% of the rows), kvq_final over the partials of the slots, the early-q projection (0.45), the scatter (0.65), the trigger splits (0.39), the preps and top-k (top-k over n_coarse, but 10 kernels of about 4 us each per layer).

Acceptance set with the final build: 3506 tokens, tau 4.399 [4.07, 4.77], 161.8 t/s (whole union with the 209ab7775 kernels 140.6, ggml-org master 178.0), ids equal on 12 of 12 requests; paired tau difference to ggml-org master +0.005 [-0.246, +0.225].

Not built or not kept:

- Single-pass top-k (one block per row, passes of 12, 12 and 8 bits, the same set and tie rule as the radix select): correct (test-backend-ops TOP_K incl. ties, bit-identical logits), but 1.52 ms per round at 8 tokens against 0.73 for the 10-kernel radix select, decode 1.0 ms slower at 8, 4 and 1 tokens. One SM per row reads the row 4 times and the warp aggregation of the histogram atomics (`__match_any_sync`) dominates. A faster version needs several blocks per row (a cluster with the histograms in distributed shared memory); not attempted. Diff: `kvq-scratch\mem\comb\topk-row-failed.diff`.
- The trigger splits (0.39 ms of GPU idle per round): the trigger is a host tensor added to a view of the selection, so the scheduler ends the GPU split there and calls the copy callback; the next split is launched afterwards. Without the split the readback needs a new ggml op (a device-to-host copy of the selection into the pinned buffer and a host function node that wakes the worker, with a CPU implementation, CUDA graph capture of both, and the copy on a side stream joined back into the graph) and the store would have to drop its trigger tensors; that is a change of the ggml core API, not done.
- Early-query projection on the query rows only (owner's experiment, not in the tree): a copy of the query rows of wq per attention layer for the early query took the projection from 0.45 to 0.30 ms per round (matmul and quantize, nsys), +188 MiB of VRAM with the IQ3_S target; the server ids differ at n_max 7 with IQ3_S (the half matrix gives other bits at 8 columns), equal at n_max 3 and one token per decode. The code is kept outside the repository as `kvq-scratch\mem\comb\eq-half.diff`.

Scripts and logs: `kvq-scratch\mem\comb` (`part1-table.txt`, `acc-per-request.txt`, `m*.txt`, `runs\TAG`, `op\TAG`, `acc\runs`, `dfl\runs`, `comb.diff`).

### Precise rows in blocks of 256 (2026-11-17, not bit-identical)

Owner's decision: variants that change the summation order are allowed if they add little to the error of `--kvq`. `kvq_precise_m` takes the q8_0 resident rows in blocks of 256 rows (4 tiles with an online softmax inside the block; V of a tile is fetched during K . q, K of the next tile during P V), so the 8192 slots give 32 partials per kv head instead of 128; tail rows and f16 resident rows keep `kvq_precise` with 64 rows. Per call at 8192 slots and 48 columns: kvq_precise_m 31 us + kvq_final 7 us against 51 + 17 us (128 / 192 / 512 rows per block: 42 / 42 / 53 us, 256 is the fastest). Per round (nsys): 8 tokens 0.94 + 0.61 ms against 1.24 + 0.76 ms (kvq_precise, kvq_final with merge and combine), 4 tokens 0.89 + 0.39 against 1.05 + 0.45, 24k and 4 tokens 0.80 + 0.30 against 0.98 + 0.35.

Decode ms per verify batch (3 sessions each at 100k, restored slot, 2 at 24k, real prefill; the same binary with 64 and 256 rows): 100k n_max 7 25.63 - 26.23 against 25.92 - 26.58, n_max 3 19.74 - 20.16 against 20.31 - 20.70, one token per decode 14.72 - 14.80 against 14.75 - 14.89; 24k n_max 7 21.14 / 21.19 against 21.79 / 21.81, n_max 3 17.22 / 17.24 against 17.81 / 17.77 (plain 17.5 and 14.9).

Error (kvq-eval from the saved state, row2660):

| | KL to the q8_0 teacher | KL to the 64-row build (same positions) | max KL to it | top-1 equal |
|---|---|---|---|---|
| 8 tokens per batch | 7.657e-5 [5.63, 9.84] (64 rows: 8.006e-5) | 5.77e-5 | 1.84e-3 | 470 / 471 |
| 1 token per decode | 8.940e-5 [6.82, 11.4] (64 rows: 7.913e-5) | 5.64e-5 | 1.38e-3 | 471 / 471 |

The distance to the 64-row build is far above the owner's line of 8e-7, but it is not the error of the new sums: the first scored position differs by KL 1.1e-10, and within a few positions the rounding moves the selection near the cut (the effect described under Reproducibility, where two runs that differed only in rounding were 3.4e-5 - 5.7e-5 apart); the KL to the teacher moves inside its interval in both directions. The op test passes against the unchanged CPU reference with its tolerance (nmse < 2e-5; 5.55e-7 against 5.54e-7 before). Acceptance set: tau 4.411 [4.18, 4.74] against 4.399, paired +0.012 [-0.214, +0.261], 163.9 t/s; ids differ on 6 of 12 requests. Deterministic: repeated runs give bit-identical logits from the state and the same server ids.

Tried and not kept (the same tree, numbers in `kvq-scratch\mem\comb`): the prep with mma (q as f16): 60 us against 25 us per call at 48 columns, slower; top-k in one kernel with a cluster of 8 blocks per list (the same set, bit-identical logits, 1 launch per layer instead of 10): 59 us per call at 32 lists against about 46 us, decode +0.1 - 0.2 ms at 100k, -0.1 ms at 24k; the histograms of the blocks are exchanged through distributed shared memory, whose reads are slow on this GPU (a first version that read all 8 histograms in every block took 0.4 - 1 ms per call).

### Whole union of the verify batch (2026-11-17)

Problem (DFlash2 drafter, `--spec-draft-n-max 7`, verify batches of up to 8 positions, n_ctx 102400, 99,395-token prompt): the request of a kv head was cut to 1024 rows by max score (`min(cap, prefetch_n x min(L, 2))`), while the 8 positions want about 3,230 distinct rows (max seen 6,230). Each position had on average 52% of its own top 1024 resident (worst 11%), KL at 8 tokens per batch 2.87e-4 against 1.35e-4 at one per decode. The merge (stamp table with random lookups) took about 300 us of the 470 us plan per layer, and the GPU waited at the joins.

Change (owner's decision): the request is the whole union, the default capacity is max_batch x prefetch_n = 8192 (`--kvq-resident-cap 0`), the plan works on bitmaps, the slot choice is oldest request first, the readback is the row indices only, the scatter of the join writes 4-byte groups and its trash entries go to 1024 trash rows after rkv. Smaller capacities keep the cut by max score as a fallback (logged once at enable).

- Slot choice: the free slots of the oldest requests first, ties by slot, against plain slot order (first free slots). Missing rows per (kv head, verify round): 719 against 1,185 at n_max 7, 112 against 469 at one token per decode (no drafter); uploads 26.8 against 42.5 MiB per round. With slot order the low slots churn every round and rows needed again a round later are gone. The age is the stamp of the last request that listed the slot's row, 8 bits (255 and older and empty slots are equal, then slot order), so the slot of every row is a function of the sets only.
- Scatter: the join's set_rows runs over the whole static slot list, 2 x G x min(cap, prefetch_n x L) = 65,536 entries per layer at L = 8, nearly all of them trash entries. With one trash slot every trash write went to the same 272 bytes; k_set_rows took 3.06 ms per round (f16 view; 2.85 with the f32 view). With the trash entries spread over 1024 trash rows (4.25 MiB more VRAM) it takes 0.406 ms per round (0.355 ms before the change at cap 1024). A scatter that covers only the fetched rows needs the row count on the device (a set_rows that skips negative indices or a kvq scatter op), a kernel change that was not made.
- Readback: the scores (`vals`) are read back only when the request can be cut (cap < prefetch_n x L); otherwise the trigger is anchored on a get_rows of one score, which waits for the selection. 2 MiB per round at n_max 7 instead of 4 MiB.
- Top-up modes (`count`, `always`, not default): they work, but their victims are now the free slots of the oldest requests and, when those are used up, the slots of this step's prefetch in slot order; before, the latter were taken by the lowest early-query score. The top-up readback is indices only (the scores were never used there).

Server, slot restore of the 99,394-token state, prompt 99,395, 300 tokens, `LLAMA_KVQ_IDLE_PROBE=1 LLAMA_SERVER_GEN_TIMES=1`, 2 - 3 sessions each (spread within 0.25 ms of decode), before = ours at 209ab7775 (cap 1024), after = the final build; per packed ubatch (16 layers) unless noted:

| | n_max 7 before | n_max 7 after | n_max 3 before | n_max 3 after | no drafter before | no drafter after |
|---|---|---|---|---|---|---|
| decode ms per batch | 33.74 - 33.82 | 32.70 - 32.95 | 21.81 - 21.96 | 21.41 - 21.51 | 14.73 - 14.89 | 14.92 - 14.99 |
| generation t/s | 115.4 - 115.6 | 116.8 - 117.3 | 128.9 - 129.4 | 129.5 - 130.3 | 66.2 - 67.0 | 65.9 - 66.3 |
| GPU idle at the joins, ms | 1.71 - 1.79 | 0.06 - 0.08 | 0.53 - 0.62 | 0.02 | 0.02 | 0.02 |
| plan ms (worker) | 7.45 - 7.53 | 2.37 - 2.39 | 4.91 - 4.96 | 1.99 - 2.01 | 1.19 - 1.25 | 1.56 - 1.57 |
| gather ms (worker) | 1.59 - 1.60 | 2.23 - 2.25 | 1.50 - 1.52 | 1.35 - 1.36 | 1.25 - 1.28 | 0.52 - 0.53 |
| requested rows per kv head | 1024 (cut) | 3,218 | 1024 (cut) | 2,150 | 1024 | 1024 |
| fetched rows per kv head | 517 | 719 | 525 | 410 | 469 | 112 |
| MiB uploaded | 17.8 | 26.8 | 18.0 | 16.4 | 16.2 | 6.3 |
| own top 1024 resident | 52% mean (investigator) | 100% | | 100% | 100% | 100% |

Plain cache (ours without `--kvq`, investigator's sessions): 26.1 / 23.35 / 15.65 ms per batch. Plan per layer after (profiling build, n_max 7 / 3 / no drafter): 161 / 131 / 102 us = union bits 41 / 23 / 6 + missing scan 26 / 21 / 13 + slot scan 45 / 42 / 40 + choice and assign 49 / 46 / 42. The first bitmap build (branches in the slot scan) took 213 us at n_max 7, the slot scan alone 111 us. Lead 3 (n_max 7): 32.97 ms per batch, idle 0.30 ms, 984 fetched rows per kv head, no gain; 8 / 12 gather threads (intermediate build): idle 0.13 / 0.08 against 0.17 ms, decode unchanged.

Kernel time per verify round (nsys, target stream): n_max 7 before / after: kvq_precise 1.708 / 3.276 ms, kvq_final 0.183 / 0.883 ms, k_set_rows 0.355 / 0.406 ms, total target kernels 34.73 / 36.25 ms; n_max 3: kvq_precise 1.012 / 1.920, kvq_final 0.150 / 0.475, k_set_rows 0.339 / 0.246. kvq_precise and kvq_final grow with the resident rows (8192 slots, all of them filled after a few rounds); they were kernel work, done since (see the attention op kernels above: kvq_precise 1.24 ms, kvq_final 0.76 ms per round at 8 tokens). The decode time still went down because the GPU no longer waits at the joins.

8 against 4 tokens per verify (n_max 7 / n_max 3, after; before in brackets): plan 1.19 (1.52), union bits 1.81, missing scan 1.27, slot scan 1.07, choice and assign 1.07, gather 1.65 (1.06), distinct rows 1.50 (1.00, cut), fetched rows 1.75 (0.98), MiB uploaded 1.63 (0.99), HtoD copy time 1.61 (1.02), k_set_rows 1.65 (1.05), readback 2.0 (2.0, linear in the positions), idle 3.5 (3.0; 0.07 against 0.02 ms), decode 1.52 (1.54). Only the idle is above 2, at a level that is noise (under 0.1 ms per round).

Quality (kvq-eval, row2660 100k, 471 positions, against the q8_0 teacher):

| | 8 tokens per batch | 1 token per decode |
|---|---|---|
| before (cap 1024, cut at two requests) | 2.87e-4 | 1.35e-4 |
| investigator's switch (whole union, cap 8192, LRU) | 7.5e-5 | |
| after, first bitmap build | 7.04e-5 [5.12, 9.48], 469 / 471 top-1 | 8.53e-5 [6.33, 10.9], 470 / 471 |
| after, final build | 8.01e-5 [6.01, 10.3], 470 / 471 | 7.91e-5 [5.70, 10.4], 470 / 471 |
| after, lead 3 | 1.04e-4 [0.76, 1.34] | 1.11e-4 [0.81, 1.46] |
| after, `--kvq-cap 1024` (fallback, cut at 1024 by max score) | 2.83e-4, coverage of the own top 1024: 53.6% | |

The two after builds differ only in the order the chosen slots take the missing rows (rounding of the sums over the slots); their intervals overlap. At one token per decode the resident set now keeps up to 8192 rows of the recent rounds, so more rows of each step are precise (112 instead of 469 fetched per kv head, KL 8.5e-5 against 1.35e-4); decode is 0.1 - 0.2 ms slower there (the precise kernel over more filled slots).

Reproducibility: two server sessions from the same slot file give the same ids hash on every configuration (n_max 7, 3, no drafter, lead 3); kvq-eval from one saved state: logits of the run that saved and two loads bit-identical at 8 tokens per batch, two loads bit-identical at 1 per decode (both the first bitmap build and the final build); test-kvq-codec, test-kvq-attn, `--kvq-rm-test` (own and merge) and `--kvq-rb-test` pass. Save / restore: the state format is unchanged; the resident set starts empty after a load as before; the slot file saved by the old build restores into the new one.

Memory (n_ctx 102400, launch line of `dflash2-vram-baseline.md` with `--kvq`): resident 274.7 MiB (+4.25 MiB trash rows in the final build) against 34.5 MiB, staging device 276.0 MiB, pinned host staging 282.5 MiB (before about 36 MiB each). VRAM during generation 17,820 MiB (dedicated usage, r1 of the memory session) against 17,315.5 MiB, +504.5 MiB. These buffers are sized for the worst case (the union of 8 positions with nothing resident, every slot fetched in one round) and can be made smaller later: they can live in the prefill scratch (217 - 486 MiB, idle during generation), the fetch can be uploaded in pieces, or they can be sized by the largest verify batch in use (n_max + 1). Not done.

Acceptance set (12 requests, ours with `--kvq`, n_max 7): before 3979 tokens, rate 0.502, tau 4.500 [4.17, 4.89], 138.4 t/s; after 3506 tokens, rate 0.488, tau 4.399 [4.07, 4.77], 140.6 / 140.5 t/s (two runs, identical ids on all 12 requests); paired tau difference -0.101 [-0.226, +0.038], not significant. The ids differ from before on most requests (other rounding in the target), so the replies have other lengths. Lead 3 against lead 2 (intermediate build): tau +0.001 [-0.087, +0.072], 128.8 against 125.8 t/s with fewer generated tokens (3584 against 3976).

Scripts and logs: `kvq-scratch\mem\union` (`matrix*.txt`, `runs\TAG`, `logs\`, `acc\runs`, `dfl\runs`, `union.diff`).

### Ring redesign (2026-11-16 evening)

What forced 2 chunks before: phases 2 and 3 of a job read the chunk's rows from the ring (`get_rows` by cell), so the chunk stayed in the ring until it was installed, about 1 - 2.5 s after it was complete, while the next chunk filled (prefill makes a chunk every 3.3 - 3.6 s). Ring = the chunk being fitted + the chunk being filled + a batch. Now phase 1 copies the ring rows to the host store and the cells are free at once; phases 2 and 3 upload the rows from the host store into the fitter's stage, layer by layer (2 x 285 MB per 8k chunk). With the spill (ring rows to the host store without a fit) the ring can be as small as one batch.

Ring and VRAM (34,816 B per cell; codes 6,400 B per row; tables per slot 8.4 MiB int8):

| | ring cells | ring MiB | coarse rows (ctx 230000 / 100352) | tables | scratch |
|---|---|---|---|---|---|
| before: 2 chunks + n_batch + 256 | 20,736 | 688.5 | 229,376 / 98,304 rows | 28 / 12 slots | 519 / 247 MiB |
| chunk + n_batch + 256 (`--kvq-tail 12544`, the default of commit 3705fe400) | 12,544 | 416.5 | 230,000 / 100,352 rows (1404 / 613 MiB) | 29 / 13 slots (243 / 109 MiB) | 503 / 234 MiB |
| now, default: max(n_batch, 4096) + 256 (one batch) | 4,352 | 144.5 | same | same | 486 / 217 MiB |

(-b 4096; the coarse region now reaches n_ctx rows and a slot for the last partial chunk.) Saving against before: 272 MiB of ring with the default, 544 MiB with one batch, the scratch 16 - 33 MiB more; the tables 8 MiB and the codes 4 MiB less saving because of the last slot.

In the results of this section "default ring" is the default of that build, chunk + n_batch + 256 (12,544 cells at -b 4096, 10,496 at -b 2048); the default since is one batch.

kvq-eval, -b 2048 -ub 2048, 1 token per decode, deterministic, against the q8_0 teacher (`ref/row*.q8c1`); KL with the 95% bootstrap CI, flips = positions where the top-1 differs:

| Trace | before (fa865aee3) | own | merge |
|---|---|---|---|
| row310, 24k (245 scored) | 4.47e-5 | 4.38e-5 [3.01, 5.90], 1 flip (first build: 6.02e-5, 0 flips) | 7.99e-5 [4.78, 12.1], 0 flips |
| row2337, 12k (347) | not run | 9.87e-5 [7.35, 12.8], 1 flip | 1.041e-4 [0.77, 1.32], 3 flips |
| row1830, 50k (289) | not run | 2.364e-4 [1.48, 3.39], 0 flips | 2.288e-4 [1.46, 3.22], 0 flips |
| row2660, 100k (471) | 1.291e-4 | 1.349e-4 [1.01, 1.74], 1 flip (first build: 1.478e-4) | 1.332e-4 [0.98, 1.74], 1 flip |
| row2660, ring of one batch (2304 cells, 36 spills) | | 1.349e-4, the same logits bit for bit as the default ring | |

Merge against own over the four traces: worse at 24k (7.99e-5 against 4.38e-5) and at 12k (1.04e-4 against 9.9e-5), better at 50k and 100k; no systematic difference. At 24k the merged group is 15,991 rows ([8192, 24183)) under one table set, against chunk 1 alone plus a partial chunk of 7,799 rows with its own tables; the intervals overlap ([4.78, 12.1] against [3.01, 5.90]), so it is one trace's spread, not an established cost. The deciding question (do 16k tables cost quality) is the chunk16 test.

Final build: the own partial chunk takes the previous chunk's statistics into its metric (see Statistics); all numbers in this section are from the final build except the reply-coding table (first build), merge is unaffected by that change. The precise set changed: the rows after the last complete chunk (7,799 at 24k, 1,091 at 100k, 585 at 50k) are now coded. All intervals overlap with the earlier numbers; at 24k the earlier spread of 13 non-deterministic runs was 3.6e-5 - 5.7e-5.

Prefill and the end-of-prompt wait (kvq-eval, 8 fit threads, the GPU otherwise idle): 100k prefill 40.08 s (own), 40.15 s (merge, 9 spills), 40.26 s (own, one-batch ring, 36 spills, all chunks read back from the host store in phase 1); wait at the end of the prompt 0.60 s (own, a partial chunk of 1,091 rows), 0.62 s (merge, a group of 9,283 rows), 0.31 - 0.35 s at 24k. The CPU part of a job does not depend on its row count, so a partial chunk of 9 rows costs the same 0.28 s as one of 1,000. The eval prefill of the version before was not measured with the same flags, so the prefill cost of the redesign is only measured in the server (below).

Minimum rows of an own partial chunk. The codec handles few rows: the covariance has rank n - 1, the eigen solver gives zero variances for the rest, the bit allocation gives them 0 bits (if fewer than 64 components have variance, the spare K bits go to zero-variance components with a step of about 1e-15 and code zeros), the rows are coded inside their own span; the K metric from few positions is near-singular but regularized (1e-6 of its trace). Measured by choosing the chunk size so that about 10, 100, 300 and 1000 rows follow the last complete chunk, fitted (min_rows 1) against kept precise in the ring:

| rows after the last chunk | row310 fitted / precise | row2660 fitted / precise |
|---|---|---|
| 9 / 11 (C 8058 / 8282) | 6.62e-5 / 6.91e-5 | 1.529e-4 / 1.372e-4 |
| 99 / 107 (C 8028 / 8274) | 5.61e-5 / 6.23e-5 | 1.487e-4 / 1.290e-4 |
| 300 / 299 (C 7961 / 8258) | 7.01e-5 / 5.82e-5 | 1.353e-4 / 1.307e-4 |
| 999 / 1007 (C 7728 / 8199) | 6.76e-5 / 5.66e-5 | 1.362e-4 / 1.386e-4 |

(final build; the first build, without the previous chunk's statistics in the metric, gave 6.58 / 8.07 / 8.37 / 5.71e-5 and 1.371 / 1.453 / 1.240 / 1.488e-4 fitted.) At 100k the fitted tiny partial chunks (11 and 107 rows) are 11 - 15% above precise, inside the interval; one more reason for the 256-row minimum.

Flips 0 - 2 everywhere, every pair inside the run-to-run interval: no measurable quality difference. Default `--kvq-min-rows 256`: below D + 1 = 257 rows the covariance of the rows is rank deficient, and the fit of a few rows costs the full 0.28 s of end-of-prompt wait. The first coarse region needs prefetch_n + n_sink = 1028 rows (the prefetch selects 1024 non-sink rows).

Reproducibility and state:

- Two processes, same flags: identical logits files (row310, own). Two processes with 7 in-line folds each (row2660, `LLAMA_KVQ_FOLD_EVERY=64`): identical logits (first and final build).
- State at 100k (own, first and final build): 4,395 MiB (before 4,460), saved in 1.7 - 1.8 s, loaded in 0.67 - 0.76 s; the logits of the run from the loaded state are identical to the run that saved it, and the loaded state saved again is byte-identical.
- A version-1 state (the version field of a v2 file set to 1): "error loading sequence state file: version 1 of the two-tier cache state, expected 2", the load returns 0, no crash.
- `--kvq-rm-test` passes for own and merge (rollback into the coded rows after a chunk back to the grid, refit at the next end of the prompt, ring rollback 0.02 ms, a small batch after a prefill waits for the job in flight, clear, recurrent rollback failure, the layout end after each prompt). The ring rollback in the geometry of the earlier test (first chunk coarse, 2 ring rows) gives KL 1.5e-6 between the two decodes (2.0e-6 before); about 500 rows after a chunk the same check gives 1.0e-3 - 2.5e-3. That is the model's own sensitivity to the batch split at that position, not the rollback and not the two-tier cache (table below).

Ring rollback at b = C + 501 (row310, logits at b + 3 compared between a 4-token batch and, after removing its last 3 positions, a 3-token batch):

| Run | KL |
|---|---|
| version before (fa865aee3, own build of the commit with this check added to its rm-test) | 7.1e-4 |
| two-tier, the rows after the chunk precise in the ring | 1.06e-3 |
| two-tier, every coarse row resident (`--kvq-pf-n 8188 --kvq-cap 8188`), rows precise / coded | 1.63e-3 / 1.98e-3 |
| plain q8_0 cache (`--kvq-rb-test`), without / after an earlier rollback | 6.4e-4 / 1.06e-3 |
| two-tier (`--kvq-rb-test`), without / after an earlier rollback | 1.20e-3 / 1.06e-3 |
| plain q8_0 cache, no rollback: [b, b + 4) in one batch against [b] then [b + 1, b + 4) | 3.6e-3 |
| any of the above at C + 5 (2 ring rows) | 1.5e-6 - 2.6e-6 |

A rollback followed by the re-decode is as exact as decoding the same tokens in another batch split; the size is that of the model's batch-shape rounding at the position (recurrent layers and kernels of other shapes), which varies by position and token. MTP verify batches change the batch composition the same way in production.

Multi-turn (`--kvq-turns`, 64 greedy tokens and a rollback to the turn end between turns):

- row310, turn ends at 10240 and 18432 (batch boundaries, the same batches as the single prefill): the final state file is byte-identical (first and final build) to the single prefill and the logits are identical (KL 0), for own and for merge. So a chunk first fitted as a partial chunk (or a group) and later refitted complete equals the chunk fitted complete at once, and the rolled-back reply leaves nothing behind (the sums of the open chunk are restored to the end of the last fit).
- row2660, turn ends at 40000 and 81000 (inside batches): KL against the teacher 2.41e-4 (own) and 2.37e-4 (merge), against the single prefill of the same policy 1.79e-4 / 1.58e-4, 0 - 2 flips. The control, the same turns on a plain q8_0 cache: 1.64e-4 against the teacher (the teacher is the single prefill of that cache). The difference comes from the batch split itself (K/V rows and recurrent state computed in other batch shapes), not from the two-tier cache.

Batch independence: achieved in the sense above when the batch boundaries are the same (the state is a function of the rows and statistics only, the rollback restores the statistics); not achievable bit for bit across different batch splits, because the model's own K/V rows and query sums change with the batch shape (plain q8_0 moves by 1.6e-4 at 100k).

Reply rows coded during generation (proxy for the coordinator's question; row2660 / row1830):

| | row2660 | row1830 |
|---|---|---|
| reply precise in the ring (default) | 1.478e-4, 1 flip (final build 1.349e-4) | 2.354e-4, 0 flips (final 2.364e-4) |
| every 64 reply rows refitted with the partial chunk (in-line fold, 0.29 s each) | 1.608e-4, 0 flips (final 1.546e-4, 1 flip) | 2.391e-4, 0 flips |
| every 64 reply rows coded with the existing tables of the partial chunk (5 - 6 ms each) | 1.909e-4 [1.41, 2.54], 1 flip | 2.737e-4 [1.83, 3.77], 0 flips |

Coding the reply with the existing tables costs +16% (50k) to +29% (100k) KL here, the intervals overlap; one trace each.

Server, ctx 230000, MTP 3 drafts, -b 4096 -ub 1024, default ring (12,544 cells), 4 fit threads; three requests per session: r1 = 100k prompt (99,398 tokens) + up to 300 tokens, r2 = r1 + 50 of its tokens + 500 new (a rollback into the coded rows after the last chunk, checkpoint restore at the end of r1's prompt), r3 = r2 + 100 of its tokens + 6,600 new (crosses the boundary of chunk 13) + 1000 tokens. Idle GPU before every session: 750 MiB (nvidia-smi, other processes).

| | kvq, final build, session A / B | kvq, first build (A / B) | production (same day) |
|---|---|---|---|
| VRAM loaded (nvidia-smi) | 23,125 MiB, 23,189 after r1 | 23,137 MiB, 23,201 after r1 | 27,951 MiB, 27,997 after r1 |
| r1 prompt (100k) | 44.93 / 44.94 s (wait at the end 0.553 s) | 45.37 / 45.30 s (wait at the end 0.548 s) | 44.01 s |
| r1 generation | 129.3 / 128.9 t/s | 128.4 / 127.7 t/s | 119.2 t/s |
| r2 prompt (554 tokens) | 1.10 / 1.10 s (wait 0.559 s) | 1.10 s (wait 0.549 s) | 0.49 s |
| r2 generation (300) | 131.6 / 131.8 t/s | 130.1 / 129.9 t/s | 126.7 t/s |
| r3 prompt (6,704 tokens) | 5.44 / 5.45 s (wait 0.574 s; the 152 rows after chunk 13 stay precise, fewer than 256) | 5.75 / 5.74 s (wait 0.823 s) | 4.89 s |
| r3 generation (1000) | 128.9 / 129.8 t/s | 128.0 / 128.0 t/s | 122.2 t/s |
| generated ids | equal between A and B and to production on all 3 requests | equal between A and B and to production on all 3 requests (265, 300, 1000) | |

More sessions with the final build (same requests; ids compared with the default-ring session of the same build):

| | VRAM after r1 | r1 prompt (100k) | wait at the end of r1 / r2 / r3 | generation r1 / r2 / r3 | ids |
|---|---|---|---|---|---|
| default ring, 4 fit threads | 23,189 MiB | 45.18 s | 0.551 / 0.554 / 0.573 s | 128.9 / 131.0 / 129.1 t/s | reference |
| one-batch ring (`--kvq-tail 4352`), 13 spills in r1, 1 in r3 | 22,839 MiB | 45.14 s | 0.557 / 0.553 / 0.572 s | 129.6 / 131.9 / 128.9 t/s | equal on all 3 |
| merge | 23,271 MiB | 45.35 s | 0.585 / 0.586 / 1.131 s | 129.3 / 131.1 / 129.0 t/s | equal on all 3 |
| 8 fit threads | 23,189 MiB | 44.97 s | 0.297 / 0.298 / 0.316 s | 129.0 / 131.1 / 129.4 t/s | equal on all 3 |
| 12 fit threads | 23,189 MiB | 44.92 s | 0.214 / 0.215 / 0.233 s | 128.7 / 130.4 / 128.5 t/s | equal on all 3 |

The end-of-prompt wait against `--kvq-threads` (the CPU part of a job, the same for the 100k prompt and the 554-token follow-up): 0.55 s with 4 threads, 0.30 s with 8, 0.21 s with 12; the 554-token follow-up prompt takes 1.10 / 0.85 / 0.76 s (production 0.49 s). Merge: the group refit costs 0.03 s more per prompt, and 1.13 s when the prompt completes a chunk (the head chunk alone and the new group). The one-batch ring saves 350 MiB against the default ring at the same prefill time.

A fold in a real session (one-batch ring, r1 = 100k prompt + 5000 tokens with eos ignored): one fold at position 103743, 0.566 s on the decode thread, generation 127.3 t/s (default ring, no fold: 127.1 t/s). Two more such sessions (separate processes): the fold at the same position 103743 (0.57 s), generated ids identical on all 5000 + 300 tokens. Against the default ring the ids differ from token 611 on, before the fold: with the one-batch ring the second prefill batch already attends over the scratch (after a spill) instead of over the ring, which rounds differently; the 300 tokens of the follow-up r2 are equal.


Allocation line (`llama_kvq_store`): `n_coarse = 0 of 230144 rows, 29 chunks of 8192 rows ... device: codes 1404.7 MiB, tables 242.9 MiB, resident 34.5 MiB, scratch 503.6 MiB (used), other 17.3 MiB; host: store 7641.5 MiB (pinned), other 169.9 MiB`, the ring `a ring of 12544 cells for 230144 positions`. No batch had to make room in the ring, no spill, no fold.

Prefill against production the same day: +0.92 s at 100k (44.93 against 44.01 s), of which 0.55 s is the end-of-prompt wait (the partial chunk of 1,094 rows; the CPU part of a job is about 0.55 s on 4 threads whatever its row count) and 0.37 s the rest (0.8% of the prefill: the fitter's device work and the 570 MB of host uploads per chunk in phases 2 and 3 on the main stream, 12 chunks). The version before was 44.0 - 44.2 s against production 43.4 - 43.8 s on another day, so the rest is at about the earlier cost; inside the budget of the wait + 1%. A 554-token follow-up prompt takes 1.10 s against 0.49 s in production: the end-of-prompt fit dominates short prompts. Generation is 1 - 8% faster than production in these sessions (MTP acceptance identical), as before. The first build failed the fit of r3's 152 rows after chunk 13 (metric not positive definite, see Statistics), and a failed fit stops the fitter; fixed, the final sessions above use the fix (and min_rows 256, so these 152 rows stayed precise).


### Coding the reply rows on append (investigated, not built)

Idea: a new row (generated token, or a prompt row) is coded at once with tables that exist (the current partial chunk's or group's), its q8_0 row goes to the host store, and the resident set serves it like any coded row; the next end of a prompt refits as before. The device keeps only a small precise window.

Feasibility in the current structure:

- The codes can be written by the fitter's CODE op (rotation and packing on the device, any row count); the rows go to the host store with the spill path. What is missing: the fwd tables of the last table set must stay on the device (16 layers x 4 kv heads x K/V x 256 x 256 fp32 = 33.5 MiB, or 17 MiB fp16), and the append step must run per ubatch.
- The real obstacle is the graph: the packed graph has n_coarse in its shapes (`kc`/`vc` views, mask width, top-k input, score output), and `can_reuse` rebuilds the graph when n_coarse changes. Appending one row per token rebuilds the graph every token (about 1 - 3 ms of CPU per token against 8 ms per token at 125 t/s). To avoid it the op would need n_coarse as a runtime value with fixed-size views and masks (a change of `ggml_kvq_attn` and the prefetch graph), or the rows would be appended in blocks (every W tokens, one graph rebuild per block; that is the fold with existing tables, below).
- Ring floor: the generation window would be the rows too new for the prefetch (the early query runs 2 layers ahead, so rows of the current ubatch), the MTP rollback depth (up to 7) and the most recent positions (a few dozen; W for block appends): a few hundred cells. But the prefill places a whole batch in the ring (`tail_wait` before `prepare_batch`), so the floor is set by the batch: n_batch + 256 = 4352 cells = 144.5 MiB at -b 4096 (the smallest ring the code accepts today, see Results), 1280 cells = 42.5 MiB if the slots were placed per ubatch of 1024 (not built). Coding on append does not lower the floor below one prefill batch; it removes the in-line folds of a long generation.
- Cost per generated token (estimate, not measured): per attention layer one rotation (256 x 256 x n rows, K and V) and one CODE kernel, about 32 small kernels per token at 5 - 10 us each = 0.2 - 0.3 ms per token if done per token (2 - 4%); with blocks of 64 rows, one fold of about 1 ms device work and one graph rebuild per 64 tokens, under 0.05 ms per token. Plus 35 KB of rows per token to the host (async).

Quality, measured as a proxy (LLAMA_KVQ_FOLD_EVERY=64 with LLAMA_KVQ_FOLD_FIXED=1, a switch removed since: every 64 reply rows are coded with the existing tables of the partial chunk, the last up to 63 stay precise; and without FOLD_FIXED: refitted every 64 rows instead), see Results.

### Wide test and the measurements behind the redesign (2026-11-16, build fa865aee3 unless noted)

All against a q8_0 teacher (rotation off, 1 token per decode), teacher forced on the reply tokens of dataset rows, `--kvq-lib --kvq-lib-hold`, `-b 2048`. The noise baseline is the same q8_0 cache with 4 tokens per decode. Scripts and per-row results: `kvq-scratch\wide`, `tailt`, `topk`, `chunk16`, `repro`.

Wide test, 42 requests of 12k to 121k tokens, 22,760 scored reply tokens (up to 600 per request), nothing failed:

| Context | Requests | Positions | Top-1 flips two-tier | Top-1 flips baseline | KL two-tier | KL baseline |
|---|---|---|---|---|---|---|
| 10-20k | 7 | 2874 | 11 (0.38%) | 12 (0.42%) | 1.14e-4 | 1.04e-4 |
| 20-40k | 7 | 3647 | 9 (0.25%) | 6 (0.17%) | 1.07e-4 | 6.9e-5 |
| 40-70k | 7 | 4107 | 10 (0.24%) | 6 (0.15%) | 9.3e-5 | 6.3e-5 |
| 70-90k | 7 | 3901 | 9 (0.23%) | 7 (0.18%) | 1.75e-4 | 7.7e-5 |
| 90-110k | 10 | 5831 | 13 (0.22%) | 9 (0.15%) | 1.75e-4 | 6.4e-5 |
| 110-150k | 4 | 2400 | 4 (0.17%) | 1 (0.04%) | 8.9e-5 | 2.9e-5 |
| all | 42 | 22760 | 56 (0.246%) | 41 (0.180%) | 1.33e-4 | 6.8e-5 |

The Wilson intervals of the flip rates overlap (0.190 - 0.319% against 0.133 - 0.244%; positions of one request are correlated, so they are optimistic). Every flip of both configurations is at a position where the teacher's top probability is below 0.6 (median 0.44 and 0.43), none at 0.6 or above. The worst request has KL 3.36e-4, the largest KL at a flip is 2.7e-2. The KL ratio to the baseline grows with the context, from 1.1 at 10-20k to 2.3 - 3 above 70k. The 110-150k bucket has only 2 task groups.

Precise recent prompt positions (two-tier, the number t of precise prompt positions at the start of the reply set by the prefix length): mean KL 1.13e-4, 1.66e-4, 1.33e-4, 1.29e-4, 0.80e-4, 0.79e-4 at t = 8, 70, 300, 1091, 4000, 8100 on row2660 (471 positions) and 1.92e-4, 2.25e-4, 1.95e-4, 1.74e-4, 1.45e-4, 1.43e-4 at t = 8, 70, 300, 1000, 4000, 8100 on row1830 (289 positions); all intervals overlap, at most 1 flip per run, the two largest t have one chunk fewer. The reply tokens stayed precise in every run. This is why the ring keeps no tail of prompt positions.

Where the top attended positions are (plain q8_0 cache, exact attention from the captured queries and the cached K rows, first 150 reply positions, all 16 layers and 24 heads, `--kvq-attn-top` of kvq-eval in the worktree `kvq-wt-save`, not in this branch): of the top 64 positions, 12.3 / 16.9 / 18.7 / 31.1 / 46.0 / 53.9 lie in the last 64 / 256 / 1024 / 2048 / 4096 / 8192 positions at 100k and 11.1 / 18.2 / 19.3 / 20.2 / 30.9 / 40.2 at 50k; the attention mass on these windows is 23 / 27 / 30 / 46 / 67 / 79% and 20 / 31 / 33 / 34 / 48 / 61%. The top 64 hold about 60% of the mass, the top 128 about 68%. Heads differ widely (one head has 60 of its top 64 in the last 1024, others about 1 of 64 in the last 2048).

Chunk size (10 requests of 17k to 101k with the same coarse region for every size): the mean KL ratio of 16384-row chunks to 8192 is 1.12 (median 1.02, 4 requests better, 6 worse, per request 0.76 to 1.88), of 4096 to 8192 it is 0.88 (median 0.86, 9 comparable requests, 7 better, none worse by more than 3%). Flips: baseline 12, 8192: 16, 16384: 17, 4096: 14. The CPU part of the fit is about 2.2 s per chunk at every size, prefill and decode time are equal. So `--kvq-rem merge` (a 16k group) has no quality argument and 8192 stays; 4096 would cost twice the tables and fit CPU.

Chunk size against table type at equal table memory (14 requests of 17k to 118k, 6,437 scored positions, same coarse region in all four runs of a request, int8 build fa865aee3 and fp16 build 47b92eb4b, which differ only in the type of the decode tables; `kvq-scratch\tab4k`). KL over all positions and flips: int8 8192: 1.48e-4 and 18, int8 4096: 1.29e-4 and 15, fp16 8192: 1.41e-4 and 15, fp16 4096: 1.30e-4 and 13. Tables at 101k: 100.5, 201.0, 196.5, 393.0 MiB. So int8 with 4096-row chunks is 8% better in KL than fp16 with 8192-row chunks at the same memory (better on 12 of 14 requests; the interval of the per-request ratio is 0.76 - 1.07 on all 14 and 0.88 - 0.97 on the 11 with at least 398 positions), and the flips do not differ. The gain is from the chunk size: int8 costs 5% of KL at 8192 and nothing at 4096. The 4096 runs decode about 2% slower at 100k (17.6 against 17.2 ms per token) and prefill 0.1 - 0.3 s longer; the fit CPU per chunk is 2.1 - 2.2 s in all four. The three requests above 110k have replies of 96 to 188 positions and are noise.

One teacher run of the chunk test (row 2663, plain q8_0) gave NLL 0.111682 where the wide test and five repeats give 0.111541, with identical logged settings; not explained, not seen again. The two-tier runs of that row are byte-identical run to run on fa865aee3 and on c08afcc49 (3 runs each), the teacher runs too (2 each, and equal across the builds).

### Before the ring redesign

Server, ctx 230000, 100k prompt, MTP with 3 drafts:

| | Two-tier, int8 tables, 4 CPU threads | Two-tier, GPU fitter, fp16 tables, 4 / 1 CPU threads | Two-tier, CPU fitter (16 threads, before the codec speed fix) | Production (q8_0, flash attention) |
|---|---|---|---|---|
| Total VRAM (nvidia-smi after the request) | 22.79 GB | 23.02 GB | 24.06 GB | 28.4 GB |
| Prefill | 44.1 - 44.2 s | 44.0 - 44.2 s / 44.2 s | 47.9 s | 43.4 - 43.8 s |
| Generation, first request | 125.9 - 129.6 t/s | 128 - 129 / 122 - 124 t/s | 109 - 112 t/s | 120.5 t/s |
| Generation, follow-up request | 123.8 - 126.8 t/s | 125.5 - 127.5 / 126.2 t/s | 122.2 t/s | 124.3 t/s |

int8 tables: two sessions (2026-11-16), token ids matched production on every request of both. GPU fitter: two sessions per thread count (2026-11-16); prefill never waited for the fitter (no batch waited for the ring, also with 1 thread). The production numbers are from another day and were not run again, so the gain over production in generation is not established. Generated token ids matched production on every request in three of the four sessions (265, 1500, 300 tokens); in one session the first request diverged at token 211 of 265 (before the packed path was made deterministic, see Reproducibility).

GPU fitter cost per 8k chunk at 100k (kvq-eval, -ub 2048; device time with `LLAMA_KVQ_FIT_SYNC`, the GPU otherwise idle):

- Device: 35 ms (phase 1 20 ms with the 285 MB download of the q8_0 rows to the host store, phase 2 9 - 10 ms, phase 3 6 ms).
- CPU: 2.0 s on one core (2.1 s on 4 threads), the eigen solver about half of it. A job takes 2.1 - 2.6 s with 1 thread and 0.6 - 1.0 s with 4 threads in the server (statistics to join 2.2 - 3.2 s / 0.9 - 1.1 s); prefill makes a chunk about every 3.6 s.
- Decode thread: 60 - 85 ms in the snapshot and phase 1 (mostly the existing synchronize of the statistics readback), 25 - 50 ms in the later phases and the join (the synchronize of `set_n_coarse`).
- PCIe: 285 MB q8_0 rows, 34 MB moments device to host; 51 MB parameters (fwd tables), 9 MB packed tables host to device. The codes do not cross.
- Memory: 83 MiB device (fitter tensors 20 MiB, graph buffer 63 MiB), about 85 MB pinned host per job in flight (the CPU fitter used 285 MB per job).
- The o_proj grams of the V metric: 0.08 s on the GPU once, at the start (about 4 CPU-seconds on the first chunk before).

Against the CPU fitter on the same chunk (chunk 5 of the 100k trace, all 128 tasks): q8_0 rows and K metric identical, V metric max relative difference 2e-6 (o_proj grams on the GPU), means identical, covariance max relative difference 4.1e-6, bits identical, steps 3e-5, 300 of 32768 eigenvectors with a flipped sign, sign-aligned inverse tables max difference 0.4% of the largest entry, codes 0.34% different, 3.0e-5 after the sign alignment, V gains 139 of 524288 differ by one f16 step.

Quality, KL against the q8_0 reference (q8_0 cache without rotation, `ref/row*.q8c1`), 471 scored tokens, 1 token per decode. The rows before 2026-11-16 (evening) are single runs of the non-deterministic packed path, each one draw with a spread of about +-2e-5 (see Reproducibility):

| Cache | KL | ms/token |
|---|---|---|
| Two-tier, library path, CPU fitter | 1.32e-4 | 16.86 |
| Two-tier, library path, GPU fitter (5 runs: 4 or 8 threads, 1 thread) | 1.25e-4 - 1.40e-4 | 16.69 - 16.73 |
| Two-tier, library path, GPU fitter, int8 tables (3 runs) | 1.24e-4 - 1.41e-4 | 16.87 - 16.97 |
| Two-tier, deterministic packed path, fp16 tables (from a saved state, `--kvq-lib-hold`) | 1.254e-4 | 17.01 - 17.06 (median 16.80) |
| q8_0, 1 against 4 tokens per decode (both deterministic, only the rounding differs) | 5.1e-5 | |
| q8_0 with rotation on (two q8_0 configurations against each other) | 1.5e-4 | 17.70 |
| Two-tier with offline V tables from another conversation | 4.4e-4 | 16.88 |
| Plain q4_0, rotation on | 1.17e-3 | 24.15 |
| Plain q4_0, rotation off | 2.69e-3 | 24.22 |

int8 tables cost about 0.15 ms per token in kvq-eval (fp16 build 16.71 ms in the same hour). nsys: the kernels that read the tables got faster (prep 12.2 -> 11.0 us, combine 7.8 -> 6.0 us), but the attention op of a layer takes 111 instead of 99 us: the resident precise rows (side stream) now take the SMs first and delay the coarse kernel, which is on the critical path (coarse 66 -> 81 us, precise 59 -> 25 us). Forking the side stream after prep made it worse (17.33 ms). The overlap of the two streams is fragile; a fix is open.

VRAM as allocated at ctx 230000 before the ring redesign (MiB): coarse rows 1400, tables 235 (459 with fp16 tables), tail ring 689, scratch 519, resident and other 52; together about 2880 (about 2.0 GiB at 150k). Main compute buffer 1642. MTP context: f16 cache 899 (production uses f16 there too), compute 586. Host: 7.6 GB pinned store. After the redesign see the ring table above.

Per format at 150k, main model only (arithmetic): f16 9.16 GiB, q8_0 4.86 GiB, q4_0 2.57 GiB.

Save and restore at 100k (2026-11-16, measured; the files were read back right after they were written, so the loads probably came from the OS file cache):

| | Two-tier | Plain q8_0 (same binary, no `--kvq`) |
|---|---|---|
| kvq-eval state (prefix 99395) | 4459.6 MiB, save 1.84 - 2.33 s, load 0.65 - 0.75 s | 3452.5 MiB, save 2.20 s, load 1.30 s |
| Server slot file (99598 tokens, with 3 checkpoints 449 MiB and the MTP draft 391 MiB) | 5.56 GB, save 2.3 - 2.5 s, restore 1.22 - 1.25 s | 4.51 GB, save 2.9 - 3.0 s, restore 1.96 s |
| RAM prompt cache entry (with the draft) | 4863 MiB, saved in 0.52 s | not run |

- Restore is exact: a restored state saved again is byte-identical to the file (24k and 100k), and two saves of the same state are identical. Logits of a run continued from a restored state are bit-identical to the run that saved it when the scoring uses ubatches of 64 tokens (scratch attention); this holds also for a rollback to a recurrent checkpoint at 90000 (two coarse chunks dropped and fitted again) after the restore, and for a load into a context with a larger ring (`-b 4096`).
- With one token per decode (packed attention) the logits of a run from a restored state are bit-identical to the run that saved it and to every other load of the file (24k and 100k, since the packed path is deterministic, see Reproducibility). The fit itself is deterministic: two runs wrote identical state files.
- The wait at the end of the prompt: 0.011 - 0.015 s in the server after a 100k prompt (the last chunk was nearly done), 0.41 - 0.48 s in kvq-eval (-ub 2048). First-request generation 128.9 t/s (two sessions), the same as before the change. A save while a chunk was in flight waited 0.435 s for it.
- Server sequence: 100k request, trim to the confirmed prefix (`n_predict` 0), slot save, process restart, restore, continuation of 600 + 300 tokens: generated ids equal to the continuation without the restart, 300 of 300, also after a restore into a non-empty slot and after a switch through the RAM prompt cache.

## Reproducibility

Measured 2026-11-16 with kvq-eval from saved states (24k: row310, 245 positions; 100k: row2660, 471 positions; 1 token per decode, `--kvq-lib-hold`).

Before, two loads of the same state never gave the same logits: KL between two runs 5.1e-5 (24k) and 5.7e-5 (100k), max |difference of a log-prob| 5.9 - 9.0 (rare tokens), median over positions 0.2, top-1 different at 0 or 1 position. Two sources, each enough on its own for the whole effect:

1. `ggml_top_k` on CUDA (radix select, 4 rows) writes the rows above the threshold with a global atomic counter, so the order of the selection changes from run to run. The plan assigned the missing rows to the LRU slots in that order, so the row of each resident slot, and with it the order of the sums of the precise attention, changed. The selected set itself was the same.
2. `ggml_kvq_attn` summed the partials of the 8 warps with shared-memory float atomics in `kvq_prep` (qt, used by SCORE and ATTN) and `kvq_combine` (coarse output). The kernel gave different bits on identical inputs in every repeated call (test-kvq-attn).

Each difference is a rounding, but it grows: after the first token the scores of later layers differ, the top 1024 near the cut change, and from then on 70 - 93% of the selections (per layer, kv head and token) differ between two runs. With only one source left (selection sorted with the atomic sums, or the kernel fixed with the device order) the KL between two runs was still 3.4e-5 - 5.2e-5; with only `kvq_combine` atomic 2.0e-5, the first 16 positions identical.

Fix: the plan sorts the selection by row (radix sort, ties of the MTP union broken by row; since 2026-11-17 a bitmap, which has no order, and the slot choice by age and slot, see Generation and MTP verify), the kernel writes the warp partials to shared memory and sums them in a fixed order. Now two loads give bit-identical logits and identical selections and slot maps at 24k and 100k, and the kernel gives the same bits on repeated calls. Cost: kvq-eval 24k median 16.08 - 16.14 against 16.07 - 16.12 ms/token (alternating runs); 100k median 16.80 against 16.70 (mean 17.01 - 17.06 against 16.90), not alternated, so up to 0.1 ms per token (0.6%). Server (100k prompt, MTP 3 drafts, alternating sessions): 125.5 / 127.0 t/s first request and 124.8 / 125.5 follow-up, against 126.5 / 126.9 and 125.3 / 125.6 for the old code; the final build 128.0 and 126.7 t/s in one more session. Generated ids equal in all sessions, old code included. A std::sort instead of the radix sort cost 5% in the server (1.5 ms of plan per verify ubatch).

What this means for the quality numbers: the model itself moves by about the same amount when only the rounding changes. Plain q8_0 with 1 against 4 tokens per decode, both deterministic: KL 3.6e-5 (24k), 5.1e-5 (100k). Against the q8_0 reference the deterministic two-tier path gives 5.09e-5 (24k) and 1.254e-4 (100k); 13 earlier runs at 24k gave 3.6e-5 - 5.7e-5, two at 100k 1.34e-4 and 1.48e-4. So at 24k the loss of the two-tier cache is not distinguishable from rounding noise; at 100k a real loss of about 1e-4 is left, and the earlier spread (1.06e-4 - 1.48e-4) is the run-to-run noise. A deterministic run is still one draw of that rounding: a change in any kernel can move the KL by a few 1e-5, so compare configurations on more traces, not on a single number.

## Decisions taken, with the reason

- K 2 bit, V 1 bit: K2V1 is as good as K2V2 in the two-tier setting.
- Tables per 8k chunk, fitted on the chunk's own rows and queries: no quality cost against one fit over the whole prefix; chunk size 4k to 16k makes no difference. K tables from another conversation do not work.
- V fitted per chunk too: offline V tables cost 3.5x KL at 100k.
- Early query with lead 2: twice the fetch time of lead 1, KL difference inside the noise.
- Synchronous top-up off: no KL gain measured, it costs a scheduler split. `count` and `always` stay as settings.
- Resident capacity 1024: 4096 halves the KL but costs 0.6 - 0.8 ms per token in the attention kernel. Replaced 2026-11-17 (owner's decision): every position of a packed ubatch gets all of its own top 1024 rows, so the capacity is max_batch x prefetch_n = 8192 by default and the request is the whole union (see Results, whole union).
- Prefill uses exact q8_0 rows, not the two-tier attention: two-tier prefill was never measured for quality.
- Inverse tables as int8 with one scale per component (2026-11-16): halves the tables (459 -> 235 MiB at ctx 230000). Offline on one chunk the codec error in the metric grows 0.8% mean, 3.2% worst for K and 0.0% for V; KL inside the run-to-run noise (see Results). One scale per whole table costs 5x the error on K, 6 bits costs 10 - 50% on K.
- Copies between GPU graph launches go on a separate copy stream: a memcpy on the compute stream stalled the GPU 55 - 65 us each, about 47 times per token.
- Ring redesign (2026-11-16, owner's decisions): no deliberate precise tail of recent prompt positions (quality flat from 8 to 8,100 precise recent positions); the fixed chunk grid stays; the rows after the last complete chunk are fitted at the end of every prompt from scratch and prefill waits for it (stability over about 0.3 - 0.6 s per prompt); the reply stays precise until the next end of a prompt (25 - 37% of the attention mass is on generated tokens; coding them with the existing tables cost 16 - 29% KL in one run per trace, see Results).
- Rollback into the coded region leaves the rows in the host store and refits at the next end of a prompt (instead of copying them back into the ring): a ring smaller than a chunk could not take them, and the server always processes a prompt after a rollback.
- The resident set is emptied at the end of every prompt: the decoding after it depends only on the memory state, which makes multi-turn equal to one prefill and a slot restore equal to the original run. Cost: the first verify batch after a prompt starts cold.
- `--kvq-min-rows 256` (own): no quality difference measured between fitting and keeping precise 10 - 1000 rows; 256 is the rank limit of the covariance (D + 1 = 257 rows).
- `own` is the default, `merge` stays as an option (owner's idea, decided 2026-11-16 by the chunk size test): a merged group is a 16k-row fit, which is not better than 8k (mean KL ratio 1.12, median 1.02), costs 82 MiB more during the fit and doubles the wait on a turn that completes a chunk.
- Ring of one batch by default (owner's decision 2026-11-16): max(n_batch, 4096) + 256 cells, 144.5 MiB at -b 4096, 544 MiB less than before the redesign. Same prefill time, generation speed and ids as chunk + batch + 256 in the server on the three standard requests, bit-identical logits in kvq-eval. Cost: a long reply is fitted in line once per ring size of tokens (0.57 s every 4.3k tokens with 4 threads); the floor of 4096 keeps that rate with a smaller batch (my addition).
- `--kvq-threads` stays 4 by default (owner's decision 2026-11-16). They are CPU threads: the GPU part of a fit is about 35 ms per chunk, the CPU part is 128 problems of 256 x 256 per job (16 layers x 4 kv heads x K and V), about 2 CPU seconds. The end-of-prompt wait is 0.55 s with 4 threads, 0.30 s with 8, 0.21 s with 12, about 2 s with 1; generation and ids do not change. This is the one place where the feature depends on the CPU; a device eigen solver would remove it (not started).

## Open bottlenecks, largest first

1. Prefill: 44.93 s against 44.01 s of production the same day at 100k (0.55 s of it the end-of-prompt fit); the fitter does not stall it. What is left per chunk: the device work and 570 MB of host uploads of phases 2 and 3 on the main stream, two synchronizes (statistics readback, join), the D2H of the q8_0 rows. A second stream (or the copy stream for the transfers) could hide most of it; V-only rows in phase 2 would save a quarter of the uploads.
2. The end-of-prompt fit: 0.55 s per prompt on 4 threads whatever the row count (the 256 x 256 problems); it dominates short follow-up prompts (554 tokens: 1.10 s against 0.49 s). More threads at the end of the prompt, or a device eigen solver, would shorten it.
3. The latency of a job is set by the polls (one per ubatch): after the CPU tasks, a job needs one or two more ubatches. With 1 thread and -ub 2048 (kvq-eval) 3 batches still waited 0.35 s at 100k.
4. Main compute buffer 1.64 GB: mostly the F16 copies of K and V that flash attention makes at full length.
5. MTP context cache: 899 MiB f16 for the full context.
6. Verify batches of 8 (DFlash2): 26.0 ms per batch at 100k, the same as the plain cache (26.1), but about 4 ms per round that do not shrink with the context (21.6 against 17.45 ms at 24k; see Results, attention op kernels): kvq_precise over the 8192 resident slots, kvq_final, the early-q projection, the scatter, the trigger splits and the 10 top-k kernels per layer. The resident and staging buffers are sized for the worst case (about 550 MiB of VRAM, 280 MiB pinned).
7. Tables: 243 MiB at ctx 230000; 16k chunks would halve them.
8. Overlap of the precise and coarse kernels in the attention op: with int8 tables the op is 12 us slower per layer, see Results.

## Optimisation ideas (not started, nothing measured, except item 2)

Sizes are for ctx 230000 as allocated today.

1. 16k chunks. Halves the tables (243 -> about 125 MiB), and the ring of one batch does not grow with the chunk; the fitter stage and graph buffers double (about 17 + 63 MiB more). Measured since (chunk size test, see Results): mean KL ratio to 8k chunks 1.12, median 1.02, so it costs a little quality or nothing; not planned.
2. Done: 8-bit tables instead of fp16 (see Decisions).
3. Partly done (ring redesign): rows that do not fit in the ring go to the host store uncompressed (spill), prefill reads them through the scratch; the fit is done before generation (at the end of the prompt), not during it. Open point: generation needs those rows precise in VRAM until their chunk is fitted (a per-token upload is too expensive), so they need a place there; item 4 is the candidate. With the CPU fitter on 16 threads the backlog was about one chunk, 285 MB of rows.
4. Reuse the prefill buffers during generation. The scratch (519 MiB) and the flash attention F16 copies in the compute buffer (most of 1642 MiB) are only used by large ubatches and sit idle during generation. Candidates to live there: the not yet fitted rows of item 3, a larger resident set, or the ring itself, so that this memory is allocated once and used by both phases.
5. From the bottleneck list: chunked flash attention to shrink the F16 copies, a compressed or q8_0 cache for the MTP context, lower fetch traffic for multi-token verify batches.
6. Selection steps on the side stream (idea of 2026-11-17, nothing built or measured). The selection steps that run on the main stream in series with the GDN layers (early-query projection 0.45, the SCORE pass with its prep 2.4, kvq_score 0.37, top-k 0.73 ms per round at 8 tokens and 100k) could run on a side stream beside the GDN kernels; only kvq_precise runs on the side stream today (the coarse passes, prep, merge, combine and kvq_final are on the main stream). Their minimum time per launch would then not add to the round. The ATTN pass, kvq_precise and kvq_final (about 5 ms at 8 tokens) produce the output of the layer and must stay in series. Limits: the selection must end early enough for the worker (readback, plan, gather and upload take about 0.5 ms of the 0.56 ms window of the two GDN layers); beside the first GDN layer the window is about half, so the GPU would idle at the joins again unless the lead is 3 (37% more rows fetched, KL 1.04e-4 against 8.0e-5); kernels that overlap slow each other (kvq_precise 3.28 -> 3.72 ms per round when it began to overlap the shorter coarse pass). A literal fusion of kvq kernels into the GDN kernels (blocks with different jobs in one launch) was considered and not pursued: one launch configuration (registers, shared memory) for both jobs, a fused op or a fusion pass over the graph in ggml, and passes that depend on each other (the top-k passes) cannot share a launch anyway.

## Gaps before this can replace production

- Quality is measured on few traces: the final library path on one 100k trace, earlier stages on four (12k to 100k). Nothing was run at 150k or above, which is the target length. A wider check over more traces and lengths is needed.
- Speed of the final build was measured only at 100k. Not measured at 24k / 50k and at 150k+.
- Slot save and restore and the RAM prompt cache work (see Results), checked at 24k and 100k only, not at 150k+ (about 10 GiB per state at 230k, estimated). A kvq state is about 29% larger than the plain one. The launcher must give kvq snapshots their own directory (its snapshot directory hash does not know `--kvq`), plain snapshots cannot be loaded.
- Reproducibility: the packed path is deterministic from the same state (see Reproducibility). A chunk can no longer join during generation at a time set by the fitter threads: no job starts during generation, small batches wait for the jobs in flight, and a fold happens at a position set by the token count. A fold in a real session (one-batch ring, 5000-token reply) gave identical ids in two processes.
- Batch independence holds only for the same batch boundaries (multi-turn with turn ends at batch boundaries is byte-identical to one prefill); with other boundaries the model's own rows differ in rounding (plain q8_0 moves by 1.6e-4 at 100k).
- CUDA only. The attention op and the fit op have a CPU reference and a CUDA kernel, nothing for Metal or Vulkan; the fit mode refuses to start when the device cannot run the fit graphs. The fitter needs about 2 CPU-seconds per 8k chunk (one core is enough at 100k on this GPU, see Results).
- The fitter's device work runs on the main stream between ubatches, now with 570 MB of host uploads per 8k chunk (phases 2 and 3 read the rows from the host store; phase 2 uploads K too). A second stream or V-only uploads would hide or halve it, not built.
- With the ring of one batch (the default) a long reply folds every ring size of tokens (about 4.3k at -b 4096, 0.57 s each in line with 4 threads, measured in two server sessions with identical ids); with `--kvq-tail 12544` every 12k.
- The slots of a batch are placed for the whole batch at once, so the ring cannot be smaller than one batch (per-ubatch placement would allow n_ubatch + 256, not built).
- Host memory: 7.6 GB of pinned RAM at ctx 230000, per sequence.
- Deferred tuning: top-up rules and thresholds, prefetch size. Lead 3 was measured with the whole union (no gain, worse KL); the resident capacity follows the largest verify batch.
- No automated tests for the store, the fitter, the ring cache or the server flows; only the codec and the kernel have tests. The session flows were checked by hand once.
- No long session was run (hours, many rollbacks and refits).
- The code was written by agents and reviewed only at the level of reports, diffs of the scheduler change and test results. It needs a real review before anyone depends on it.
- `--fit` and the memory estimate do not know about the kvq memory. What to do with the freed VRAM (longer context, larger model quantization) is not decided.
- The branch sits on ggml-org/master 951d465f5 and changes ggml (new op, scheduler callback order, CUDA kernel); rebases will conflict there.

## Not exercised

- Two-tier without MTP, drafts of 7 tokens on the packed path, model reload after sleep, a chat-endpoint conversation with user-message checkpoints.
- Save and restore: a load into a smaller context or ring than the state needs, the full-context state API (`llama_state_save_file`), a sequence that ends exactly at a chunk boundary (empty ring, handled by `seq_pos_max`), the launcher itself with `--kvq`.
- Top-up modes `count` / `always` (their victim order changed with the whole union, see Results), lead 1 after the async prefetch change.
- The server with `--kvq-tail 0` or other tail sizes, the hard error when the ring is too small.
- The tool-driven `--kvq-packed` modes of kvq-eval after the tail-mode change.
- The offline V path after the K/V task split.
- MoE models with the early copy callback.

## Not supported (aborts with a message)

More than one sequence, `seq_cp`, `seq_add` / `seq_div` (context shift, cache reuse), removal of a middle range.

State save and load of the K/V outside the fit mode with a tail ring (`--kvq-tail 0`, the tool-driven packed and two-tier modes): an error, the save or load returns 0. A plain state (saved without `--kvq`) into a kvq context and a kvq state into a plain context: an error, the sequence is empty afterwards; there is no migration from plain snapshots. The `ON_DEVICE` state flag on a kvq state: an error.

## Working material outside the repository

`../kvq-scratch`: build scripts (`build.bat`, `reconfigure.bat`, build dir `build-cuda`), token files of four traces (`tok/`, prefix lengths 11705, 24183, 49737, 99395), reference logits (`ref/`), server session scripts (`srv/`), logs. The token files come from private data: never print or log prompt text, generated text or decoded tokens.
