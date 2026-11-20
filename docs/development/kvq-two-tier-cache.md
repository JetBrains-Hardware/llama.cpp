# Two-tier KV cache for qwen35 (kvq) - prototype state

State of branch `kvq-two-tier-cache` (2026-11-19): the GPU fitter, the ring redesign (the rows after the last complete chunk are fitted at the end of every prompt, the ring holds only rows that are not coded yet), the whole union of the verify batch, the resident rows and the ring in pinned host memory, since fba885494 the compute buffers per phase and the GDN replay of ggml-org PR 31669 (merged at 4218b5b9f, see Results, memory per phase), since affa1fa81 the codes and tables in host memory during the prompt, and since 6d2ecbfab the prompt's scratch attention in mask-free key blocks merged by the log-sum-exp (see Results, combined build; its quality is settled in Results, quality of the blocked attention). With these the server fits 250k context into 15.5 GiB of VRAM in every phase, at ub 1024 and, with `--spec-draft-n-ubatch 1024`, at ub 4096. The CPU fitter is kept on branch `kvq-fit-cpu`. The branch starts from ggml-org/master 951d465f5 and is pushed to the JetBrains-Hardware fork. This file is the starting point for the next work session.

A self-contained user guide with the final numbers of 2026-11-19 (build, run, every knob, memory per phase, speed, quality, limits) is `kvq-two-tier-cache-guide.md` next to this file.

Research background, codec definition and the original plan are in the `qwen3.8-exp` repository: `notes/kvq_implementation_plan.md`, `notes/kv_prefetch_results.md`, `src/qexp/kvcodec.py`, `experiments/14_kv_prefetch.py`.

## Removed relative to the prototype (2026-11-19)

Branch `kvq` (ours; the prototype is `kvq-two-tier-cache` 045ef35e5) removes the code that the default `--kvq` path does not run; the default path computes the same bits (same ids, slot files and kvq-eval logits as `mem/final/bin-final`). The Results and decision sections below keep the history and still name the removed modes.

- kvq-eval history modes: the fake-quant `--kvq-variant` family (`--kvq-qwin`, `--kvq-sinks`, `--kvq-recent`, `--kvq-kmetric`, the offline `--kvq-fit-chunk`, `--kvq-save-tables` / `--kvq-load-tables`, `--kvq-load-only`, `--kvq-own-mean-k`, `--kvq-stats`), `--kvq-two-tier`, `--kvq-precise`, `--kvq-protect`, `--kvq-tt-stats`, `--kvq-packed` with offline tables, `--kvq-sync*`, `--kvq-all-resident`, `--kvq-pk-stats`, `--kvq-scr-check`, `--kvq-lib-full`, `--kvq-lib-busy`, `--kvq-lib-nowait`, `--kvq-lib-vtab`.
- The explicit two-tier mode (coarse copy of the cache, `build_attn_kvq`, `llama_model_set_kvq_two_tier`), the fill mode with offline tables (`llama_memory_kvq_packed_alloc` / `packed_set`, `set_codes`), `all_resident`, the layers with their own resident buffer and `LLAMA_KVQ_NBUF`, the top-up and sync modes (`KIND_TOPUP`, `sync_mode`, `ggml_kvq_attn_cond`, `GGML_KVQ_COND`), the fused `GGML_KVQ_ATTN | GGML_KVQ_SCORE` mode, `llama_kvq_params::scratch` and `max_tail`, and the ext API only kvq-eval used (`llama_memory_kv_get_rows` / `set_rows` / `layer_ids`, `llama_memory_kvq_set_params`, `set_coarse_rows`, `get_memory_info`, `llama_context_set_eval_callback`).
- `--kvq-tail 0` (full device cache) and every path without the host ring (device ring copies of the fitter, `build_scr_copy`, `kvq_tail_view`, the k_h / v_h aliases, `set_input_kvq_early`, the separate scratch allocation); `--kvq-tail N` needs N > 0 and a context of at most N positions now uses a ring of one cell per position.
- `--kvq-v-tables` (offline V tables) and the codec table files (`llama_kvq_save` / `llama_kvq_load`).
- The f16 tail and resident variants of `ggml_kvq_attn` (q8_0 only).
- The CPU codec reference (`llama_kvq_fit`, `llama_kvq_encode`, `llama_kvq_decode`, `llama_kvq_fit_encode`, `llama_kvq_pack_rows`) moved to `tests/kvq-ref.h`; the benchmark, profile and hash modes of test-kvq-attn moved to `llama-kvq-bench` (`tools/kvq-eval/kvq-bench.cpp`).
- Debug probes: `LLAMA_KVQ_IDLE_PROBE`, `LLAMA_KVQ_GATHER_THREADS`, `LLAMA_KVQ_SCRATCH`, `LLAMA_KVQ_FIT_SYNC`, `LLAMA_KVQ_SCHED_SYNC`, `LLAMA_KVQ_UP_CHUNK`, `GGML_KVQ_ROWS_PER_BLOCK`, `LLAMA_SERVER_GEN_TIMES`; the server prints one kvq line per request. Test knobs that stay: `LLAMA_KVQ_FOLD_EVERY`, `LLAMA_KVQ_UP_REPEAT`.

## What it is

A KV cache for Qwen3.8-27B (GGUF arch `qwen35`) that keeps most of the attention K/V out of VRAM:

- Coarse tier, VRAM: every row of a fitted chunk as transform-coded low-bit codes (K 2 bit, V 1 bit plus a gain). The attention is computed on the packed codes by a custom op.
- Precise tier, system RAM: the q8_0 rows of the same positions in a pinned host store.
- Resident set: 8192 precise rows per (layer, kv head), selected per position by an early query (top 1024) and fetched from RAM while the GPU runs the two layers before the attention; the union of the positions of a packed ubatch is requested, cut by the max score when it has more distinct rows than slots. Since 2026-11-18 the rows of each layer live in a pinned host mirror and are uploaded per round into one of two rotating device buffers (see Results, resident rows from pinned host memory).
- Ring, VRAM: q8_0 rows of the positions whose rows are not in the host store yet: the reply since the end of the last prompt, and the rows of a prompt while it is processed (ring-shaped device cache, `--kvq-tail`, default one batch + 256 cells, at least 4352). Since 2026-11-18 the ring cells are pinned host memory and each attention layer uploads its occupied cells into one of four rotating device tail buffers (see Results, the ring in pinned host memory).

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

Fitter (`llama_kvq_fitter`). A job is a row range [r0, r1) with one table set (a chunk, a partial chunk or a group). The whole fit runs on the GPU (since 2026-11-20, release blocker R7), on the main stream between two ubatches, in the shared memory of the store (the scratch / resident / tail region, idle between the ubatches: `llama_kvq_store::fit_alloc`; the next prompt ubatch zeroes the scratch again, the next packed ubatch restores the resident buffers). The fitter places a stage (q8_0, C rows; 2C - 1 with `merge`), a code stage and the tables of a group of layers there; a small liveness planner places the intermediates of each graph after them. The attention layers go in groups (as many as fit: 10 at 24k, all 16 at 250k):

1. The metrics on the host from the statistics (K: the queries of the range; V: the o_proj columns scaled by the mean sigmoid(gate)), into the tables of the group.
2. Per layer of the group: rows [r0, r1) from the host store into the stage (the host ring keeps the store complete), mean and covariance (`ggml_kvq_fit`) into the tables of the group.
3. One graph for the tables of the group (`ggml_kvq_fit_table`, 8 per layer): Cholesky of the metric in double, `L^T cov L` (cuBLAS, double), the eigen solver (the tred2 / tql2 of the CPU codec in float: tridiagonal form on one block per table, the QL steps on one thread, their rotations replayed by 32-row blocks), fwd = L U and inv = U^T L^-1 (cuBLAS). The result row of each table (status, ridge) goes to the host.
4. Per layer: the rows again, K: bits and steps from sigma (`ggml_kvq_fit_bits`); V: rows rotated, the scale of the shape, bits; the packed tables (`ggml_kvq_fit_pack`) and the codes to the pinned host copies (or into `kc` / `vc` when the device copy exists, decode phase). A waiter thread waits for the job's event; the next poll installs it: the tables go to the host copy of every slot of the range and, if it exists, to the device copy; `set_n_coarse`. A range that is coarse already gets its new codes and tables in the same poll, so no graph sees a mix.

Not bit-identical to the CPU fit (float eigen solver); at 24k the KL to the same reference is 6.36e-2 against 6.40e-2, the end-of-prompt fit 0.10 s against 0.30 s (8 CPU threads) / 0.55 s (4), prefill +1.4%.

Codes and tables per phase (since 2026-11-18, see Results, codes in host memory during the prompt). Every write of the codes and tables goes to a pinned host copy with the bytes of the device tensors (1,830 MiB at n_ctx 256000). The device copy (`kc`, `vc`, `tinv`, `taux` in their own buffer) exists only for the packed path: the first prompt-path ubatch frees it (`scr_alloc`), and the end of a prompt (`llama_memory_kvq_fit_wait`) allocates it after the jobs of the last rows are started: the context first switches to the compute buffers of the tg phase (a callback of the context, `set_tg_cb`), then the buffer is allocated, zeroed, and the host copy of the rows [0, n) and the tables of their slots are queued on the main stream (n: the coarse region, or the end of the codes of jobs not joined yet); the CPU part of the fit runs meanwhile. When a packed ubatch comes before the fit_wait (the server's last 4-token batch of a prompt, or no fit_wait at all) the context allocates it after its phase switch, before that ubatch (`llama_context::process_ubatch`), never while the pp compute buffer exists. The graph input of the two-tier cache checks a generation count of the device copy, so no graph keeps a view of a freed buffer. `--kvq-tail 0` (no host ring, no shared memory) and the tool-driven fill modes keep the device copy always.

When jobs start. Prefill (ubatches larger than max_batch): a complete chunk is taken at the first ubatch after it is complete (`merge`: once the next chunk is complete too). Generation (small ubatches): no job starts; the jobs in flight are waited for before a small batch, so the path of the small batch (packed or scratch) does not depend on the fitter threads. At the end of a prompt: the layout as above, sync.

Statistics. K metric: second moment of the queries summed on the device during decoding (`qacc`), a ubatch counts for the chunk of its first position; when a chunk is complete its sums are read and kept on the host (for its fit and for a rollback into it) and the device sums start again from 0. A group uses the kept sums of its complete chunk plus the device sums of the open chunk; an own partial chunk also takes the kept sums of the chunk before it (its own sums can cover very few positions: in a server session the 1024-token ubatch that crossed the boundary counted for the previous chunk and the partial chunk of 152 rows had the queries of 4 positions, the metric was indefinite and the fit failed; the queries just before stand in). The device sums of the open chunk are read without a reset and kept with their position; a rollback to exactly that position restores them, so a reply that is rolled back leaves no queries behind. V metric: o_proj columns scaled by the mean sigmoid(gate), summed the same way (`gacc`).

Ring overflow. Before a batch is placed (`tail_wait`), the ring needs room for it. A large batch: first the jobs of the complete chunks start (their ring cells are free at once), then the remaining ring rows before the batch are copied to the host store without a fit (spill). A small batch (generation): when p0 + max_batch - n_host > ring size, the rows after the last complete chunk are fitted in line on the decode thread (fold: the layout of p0, sync, then the batch); the position of a fold depends only on the first position of the batch and the state, so generation stays reproducible.

Generation and MTP verify (ubatch <= 8 tokens), per attention layer il:

1. At the input of layer il - 2 the graph computes layer il's query from the earlier hidden state, scores all coarse rows, and takes the top 1024 per (kv head, position).
2. A scheduler split with a copy callback queues the readback of the selection (the row indices only; the scores only when the request can be cut, see below) on a copy stream. A worker thread plans the resident slots and gathers the rows from the host store. The GPU keeps running layers il - 2 and il - 1.
   - The request of a kv head is the union of the selections of all positions of the ubatch, deduplicated in a bitmap of n_coarse bits (set the bits of the L lists, the word scan gives the distinct rows in row order; and-not with the bitmap of the resident rows gives the missing rows).
   - A slot is free if its row is not in the request. The missing rows (in row order) go to the free slots of the oldest requests (a per-slot stamp of the last request that listed its row; age in requests, 255 and older and empty slots count as 255), ties by slot. So the row of every slot depends only on the sets, not on the order the GPU wrote the selection or on thread timing.
   - The default capacity, `--kvq-resident-cap 0` = max_batch x prefetch_n = 8192, holds the whole request of the largest packed ubatch (6144 is as good at 8 per batch and saves 71 MB of pinned memory, cap sweep 2026-11-18; 8192 kept because it is 0.4 ms faster at n_max 7 and keeps the output of the earlier builds). A request of more distinct rows than slots is cut to the rows with the largest max score over the positions (ties by row, the scores are read back for this case only; logged once at enable, counted as `pf_cut` / `pf_lost`).
3. Before layer il's attention a join uploads the rows and scatters them into the resident tensor. The resident tensor is one of two rotating device buffers (layer index mod 2); the worker queues the upload of the whole pinned mirror of the next layer into the other buffer right after the join uploads (a base stream, one copy of 13.4 MB), and the join of that layer waits for it before its own rows. The worker writes the fetched rows into the mirror after the gather.
4. `ggml_kvq_attn` computes the attention over coarse rows + resident precise rows + the ring.

Prefill (larger ubatches), per attention layer: the rows [0, n_host) are uploaded from the pinned host store into one device scratch buffer (4 async uploads on the copy stream, overlapping the layers before), the ring rows after them are copied in on the device, and flash attention runs over the scratch. With `--kvq-scr-block 0` (also without host rows of a full 256 yet, or without the tail ring) it is one flash attention with an F16 mask over all scratch rows, and the logits are bit-identical to attention over a full device cache. By default (since 2026-11-19, see Results, attention split) the host rows [0, r0), r0 = floor(n_host / 256) * 256, are attended in key blocks of `--kvq-scr-block` rows (16384) without a mask (every query of a prompt ubatch sees every host row) by `ggml_flash_attn_ext_lse`, which also returns the log-sum-exp l of each query and head; only the rest [r0, n_rows) (at most 255 host rows plus the ring rows) has a mask. The partial results are merged in graph order with standard ops: d = l_b - l, o += (o_b - o) * sigmoid(d), l += softplus(d). So the F16 copies of K/V that the CUDA kernel makes exist for one block at a time and the big mask is gone; the result differs from the single attention in rounding.

Rollback (`seq_rm` of the end at p0). Inside the ring (MTP drafts): only the ring cells. Into the coarse region: the jobs with rows at or after p0 are cancelled, the coarse region drops back to the chunk grid (n_coarse = floor(p0/C)*C), n_host = p0: the rows before p0 stay in the host store, nothing is copied back. The next end of a prompt fits them again (they are prefilled over the scratch until then). If the layout at the end of a prompt leaves rows precise that are only in the host store (fewer than min_rows), they are copied back to the ring then. `clear` drops everything and the fitter starts again.

Save and restore (`llama_memory_hybrid_kvq::state_write` / `state_read`, fit mode with the tail ring only). The state is exact: nothing is fitted again on load. A save waits for the jobs in flight (there are none after the end of a prompt) and does not fit anything. Layout of the state of sequence 0 (the full-context state is the same with the arch string in front), version 2:

1. Magic `KVQS` and version 3, then six sections, each as (u64 byte size, data): the meta of the store (attention layers, kv heads, head dim, chunk rows, words per K / V row, n_coarse, type and slice size of the two table tensors, n_host) and of the fitter (K / V bits, V fitted per chunk or from a file, the `rem` policy and `min_rows`), the ring, the recurrent state, the store data, the fitter data. The loader checks the meta before it reads any data and checks that each section consumed exactly its size (the error names the section). A state of another version fails with "version 2 of the two-tier cache state, this build reads version 3 (saved by another build of the two-tier cache; process the prompt again)", a state fitted with another `rem` or `min_rows` is refused too; the sequence is empty afterwards. `kvq-eval --kvq-state-refuse F1,F2` tests that.
2. The ring: the plain state of the attention cache (the positions after n_host), read back with `find_slot`, which places a position at cell `pos mod size`.
3. The recurrent state, plain.
4. Per attention layer: the q8_0 host rows [0, n_host) per kv head (raw blocks from and into the pinned store), the codes (`kc`, `vc`) of [0, n_coarse) per kv head, the tables of the ceil(n_coarse/C) slots of the coarse region (by the tensors' own type and size, so a change of the table format is a mismatch and not a misread). Codes and tables are written from and read into the host copy (the same bytes as before); a load in the decode phase also uploads them to the device copy.
5. The fitter: the fitted range of each slot of the coarse region (so a partial chunk and its row count round-trip), the statistics of the complete chunks, the device sums of the open chunk, the kept sums of the last partial fit and their position.

A plain state starts with the stream count of the attention cache, so a plain context refuses a kvq state ("n_stream mismatch") and a kvq context refuses a plain one ("not a state of the two-tier cache"); after a failed load the sequence is empty. The resident set starts empty after a load. The server's slot file appends the context checkpoints and the MTP draft state as before. The RAM prompt cache serializes the same state into RAM (an entry over `--cache-ram` is skipped).

## Code map

| Part | Files |
|---|---|
| Codec (tables from the moments of the rows, bit allocation, packed table layout) | `src/llama-kvq-codec.{h,cpp}`, `src/llama-kvq-pack.{h,cpp}`; the CPU fit and codes reference of the tests in `tests/kvq-ref.h` |
| Attention op, CPU reference and CUDA | `ggml_kvq_attn` in `ggml/include/ggml.h`, `ggml/src/ggml-cuda/kvq-attn.{cu,cuh}` |
| Flash attention with the log-sum-exp (prompt scratch in key blocks) | `ggml_flash_attn_ext_lse`, `ggml_flash_attn_ext_get_lse` in `ggml/include/ggml.h`, `ggml/src/ggml.c`; CPU in `ggml/src/ggml-cpu/ops.cpp`; CUDA MMA kernel and stream-k fixup in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `fattn-common.cuh`, dispatch (GQA variants without a mask) in `fattn.cu`; the blocks and the merge in `build_attn_kvq_scratch` in `src/models/qwen35.cpp`, the mask of the rest in `set_input_kvq_scr` in `src/llama-memory-hybrid-kvq.cpp` |
| Fit op (mean, covariance, rotation, V gain statistics, code and pack), CPU reference and CUDA | `ggml_kvq_fit` in `ggml/include/ggml.h`, `ggml_compute_forward_kvq_fit` in `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/kvq-fit.{cu,cuh}` |
| Host store, resident cache, prefetch worker, scratch uploads | `src/llama-kvq-store.{h,cpp}` |
| Fitter (device phases, CPU tasks) | `src/llama-kvq-fit.{h,cpp}` |
| Memory (hybrid subclass), ring-shaped device cache | `src/llama-memory-hybrid-kvq.{h,cpp}`, `src/llama-kv-cache.{h,cpp}` |
| Graph | `src/models/qwen35.cpp` |
| Scheduler: copy callback before the wait | `ggml_backend_sched_set_copy_callback_early` in `ggml/src/ggml-backend.cpp` |
| Compute buffers per phase (pp / tg) | `llama_context::phase_set`, `ubatch_tg`, `sched_reserve` in `src/llama-context.cpp`; `ggml_gallocr_resize` in `ggml/src/ggml-alloc.c`, `ggml_backend_sched_resize_buffers` in `ggml/src/ggml-backend.cpp`; prompt-only kvq buffers: `prompt_bufs_alloc` / `prompt_bufs_release` in `src/llama-memory-hybrid-kvq.cpp`, `scr_alloc` / `dec_alloc` in `src/llama-kvq-store.cpp`, `dev_alloc` / `dev_free` in `src/llama-kvq-fit.cpp`; codes in host memory during the prompt: `codes_alloc` / `codes_free` / `codes_up` in `src/llama-kvq-store.cpp`, `set_tg_cb` and `settle(n, end)` in `src/llama-memory-hybrid-kvq.cpp`, the allocation after `phase_set` in `llama_context::process_ubatch` |
| GDN rollback by replay (ggml-org PR 31669, merged) | `src/llama-memory-recurrent.cpp`, `src/models/delta-net-base.cpp`, `ggml/src/ggml-cuda/gated_delta_net.cu`; turned on by the server in `tools/server/server.cpp` |
| Internal API | `src/llama-ext.h` (`llama_kvq_params`, `llama_kvq_fit_params`, `llama_kvq_config`, `llama_init_from_model_kvq`, `llama_memory_kvq_*`) |
| Server options and forced settings | `common/arg.cpp`, `common/common.cpp`, `tools/server/server.cpp`, `tools/server/server-context.cpp` |
| Evaluation and benchmark tools | `tools/kvq-eval` (`llama-kvq-eval`, `llama-kvq-bench`) |
| Tests | `tests/test-kvq-codec.cpp`, `tests/test-kvq-attn.cpp` (with `tests/kvq-ref.h`, `tests/kvq-attn-problem.h`), `KVQ_ATTN`, `KVQ_FIT` and `FLASH_ATTN_EXT_LSE` cases in `tests/test-backend-ops.cpp` |

## How to run

Server, production configuration plus `--kvq`:

```
llama-server --fit off -m Qwen3.8-3.6-27B-blend-Q4_K_M.gguf --ctx-size 230000 --n-gpu-layers all --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16 --batch-size 4096 --ubatch-size 1024 --kvq
```

Options: `--kvq`, `--kvq-prefetch-n` (1024), `--kvq-resident-cap` (0 = max_batch x prefetch_n = 8192, the whole request of a verify batch of 8; a larger union is cut by the max score), `--kvq-lead` (2), `--kvq-tail N` (ring cells, default max(n_batch, 4096) + 256 = one batch; N > 0, not below n_batch + 256), `--kvq-chunk N` (rows per chunk, 8192), `--kvq-rem own|merge` (the rows after the last complete chunk, default own), `--kvq-min-rows N` (own: fewer rows after the last complete chunk stay precise, default 256), `--kvq-scr-block N` (prompt scratch attention over the host rows in mask-free key blocks of N rows, rounded down to 256, default 16384; 0 = one attention with a mask over all rows, the old numerics; env `LLAMA_ARG_KVQ_SCR_BLOCK`).

With `--kvq` the server forces, with a warning: 1 slot, context shift and cache reuse off, flash attention on, q8_0 K/V, draft n_max <= 7, recurrent rollback by replay (context param `rs_replay`; `LLAMA_RS_REPLAY=0` keeps the snapshot rows). At the end of each prompt the server waits until the prompt is fitted up to its end (log line `kvq: waited ... at the end of the prompt, coarse region N rows`); no fit work runs during the reply. It refuses to start for a model that is not qwen35, `--mmproj`, embeddings. Attention rotation of quantized caches is turned off for the model (also for the MTP context). `--fit` does not count the kvq memory.

Evaluation tool (teacher-forced KL against saved reference logits):

```
llama-kvq-eval -m MODEL -ngl 99 -fa on -fit off -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --kvq-tokens TOKENS.bin --kvq-prefix-len S --kvq-chunk 1 --kvq-ref REF --kvq-lib
```

`--kvq-lib` is the library path (same as the server), `--kvq-ring N` sets the ring (default max(batch, 4096) + 256), `--kvq-rem` / `--kvq-min-rows` / `--kvq-scr-block` as in the server, `--kvq-rm-test` checks the removal cases (both policies), `--kvq-turns S1,S2,... --kvq-turn-gen N` processes the prefix in turns (prefill to S_i with batches at multiples of the batch size, the end of the prompt, N greedy tokens one per decode, back to S_i through a recurrent checkpoint, the next turn), `--kvq-ref2` compares with a second reference (e.g. the single-prefill run). State checks: `--kvq-state-save FILE` (after the prefill and the wait), `--kvq-state-load FILE` (instead of the prefill), `--kvq-state-ckpt P` (a recurrent checkpoint at P; after the save or load, roll back to P and process the rest of the prefix again), `--kvq-lib-hold` (no chunk joins during the scoring). Compare two runs with `--kvq-save-ref` files. The kernel benchmark is `llama-kvq-bench` (`--quick`, `--profile`, `--hash`).

Settings of the memory layout: `--no-phase-buffers` (context param `phase_bufs` off: one compute buffer for all ubatches, no switch between the pp and the tg phase; also for the plain server), `LLAMA_RS_REPLAY` (see above). Test knobs: `LLAMA_KVQ_UP_REPEAT=k` (every base upload of the resident rows k times, emulates a slower bus; the cache must work on PCIe 4.0 / 3.0), `LLAMA_KVQ_FOLD_EVERY=N`: a fold whenever N reply rows follow the coarse region; kept because it is the only way to exercise folds in kvq-eval, whose replies (about 500 tokens) never fill the ring (the fold determinism check uses it). Removed after the measurements: `LLAMA_KVQ_FOLD_FIXED` (the fold coded the new rows with the tables of the last fit, the append experiment in Results) and `LLAMA_KVQ_DIAG_NORESET` (no reset of the resident set at the end of a prompt). kvq-eval `--kvq-rb-test`: ring rollbacks at C + 501 with and without an earlier rollback, and the same positions in two batch splits without a rollback, also on a plain cache (`--kvq-no-rot` without `--kvq-lib`); `--kvq-turns` also works on a plain cache.

## Results

Machine: RTX 5090 32 GB, Ryzen 9950X3D, Windows 11, CUDA 12.9. One trace of 100k tokens unless noted; treat single runs as such.

### Combined build at 250k: codes out, attention split, drafter ubatch (2026-11-19, commits affa1fa81 and 6d2ecbfab)

Both changes below in one build. The default block (`--kvq-scr-block 16384`) is confirmed (owner's decision 2026-11-19): a wider check first found its KL to the f16 teacher at 100k about 1.8x the old path, and the diagnosis (see the next section) showed that the teacher carries the single attention's own rounding bias and that the blocked attention is the more exact one. The numbers are given for block 16k and for block 0 (the old attention, kept as an option). Consistency: with block 16k the ids equal the attention split alone in every session (100k, 250k with the follow-up prompt, 24k slot save + restore with byte-identical slot files, a 5000-token reply with a fold), with block 0 they equal the build before (15b59d61d) at ub 1024 and ub 4096; kvq-eval 9 / 9 checks bit-identical in both pairings; FLASH_ATTN_EXT_LSE, FLASH_ATTN_EXT, KVQ_ATTN, KVQ_FIT, GATED_DELTA_NET, test-kvq-codec, test-kvq-attn pass.

VRAM at 250k (n_ctx 256000, n_max 7, MiB, budget 15,872; "d1024" = `--spec-draft-n-ubatch 1024`, the drafter's pp compute buffer sized by its own ubatch, 480 -> 120 MiB at ub 4096):

| cell | load | prompt | decode median | follow-up prompt peak | largest vs budget |
|---|---|---|---|---|---|
| build before, ub 1024 | 17,187 | 17,369 | 15,163 | 17,299 - 17,411 | +1,539 |
| ub 1024, block 16k | 14,039 | 14,205 (15,105 - 15,187 at the switch to decode) | 15,159 | 15,247 - 15,271 | -601 |
| ub 1024, block 0 | 15,364 | 15,546 | 15,168 | 15,472 - 15,584 | -288 |
| build before, ub 4096 | 19,628 | 20,300 | 15,310 | 19,824 - 20,297 | +4,428 |
| ub 4096, block 16k, d1024 | 15,126 | 15,372 | 15,176 | 15,354 | -500 |
| ub 4096, block 16k | 15,186 | 15,876 | 15,320 | 15,408 | +4 |
| ub 4096, block 8k, d1024 | 15,126 | 15,394 | 15,194 | 15,372 | -478 |
| ub 4096, block 0, d1024 | 17,746 | 17,974 | 15,172 | 17,854 | +2,102 |

Fit the budget in every phase: ub 1024 with block 16k or block 0, ub 4096 with block 16k or 8k and d1024. ub 4096 needs both the attention split and d1024 (block 0 is 2.1 GB over); 8k gives nothing over 16k at ub 4096 (1,570 MiB of pp compute is the floor of the other activations). With block 16k at ub 1024 the decode phase (the codes back on the device) is larger than the prompt, so decode and the switch into it set the headroom.

Speed at 250k (interleaved pairs): ub 1024 block 16k prompt 139.1 s against 199.3 s of the build before (-30%), decode 37.2 against 37.5 ms per verify batch, follow-up prompt 1.50 s against 1.71 s (-12%: the split saves more than the codes upload costs). ub 1024 block 0: prompt 201.9 s (+1.3%, the pair was not adjacent), decode -0.1%, follow-up 1.73 s (+1.5%, the accepted cost of the codes upload). ub 4096 block 16k d1024: prompt 136.1 s against 164.0 s of the build before at ub 4096 (-17%), follow-up 1.49 against 1.76 s; ub 4096 block 0 d1024 165.4 s. 100k: ub 1024 block 16k 37.3 against 42.9 s (-13%), ub 4096 d1024 37.2 against 39.5 s (-6%), decode within 0.3% where the ids are equal.

`--spec-draft-n-ubatch 1024` changes the generated ids at 250k ub 4096 with block 0 and 8k (not with 16k, where the drafts happened to be the same): the drafter prefills in other ubatches, proposes other drafts, and the target's output depends on the verify batch boundaries. Numbers, logs and notes: `kvq-scratch\mem\final` (`RESULTS.md`, `NOTES.md`).

What the device holds at 250k with block 16k (MiB; "remainder" = measured total minus the logged buffers: CUDA context, cuBLAS, the ggml CUDA pool of kernel temporaries): target weights 10,617, drafter weights 1,080, drafter KV 60, recurrent state 177, codes + tables 0 in the prompt / 1,831 in decode, scratch shared with the resident, tail and staging buffers 540 (both phases), kvq other 17, fitter 86 (prompt only), target compute 485 at ub 1024 / 1,570 at ub 4096 in the prompt and 72 in decode, drafter compute 120 / 9, layer input buffer 400 / 1, remainder 620 - 705 in the prompt and 755 - 772 in decode. Left to squeeze if ever needed: `-b 1024` at ub 1024 sizes the layer input buffer by the batch (400 -> 100, the drafter reads the rows per target batch, so the ubatch does not matter; ids change through the drafter's batch boundaries); the scratch as a ring of two or three 16k-row blocks fed on the copy stream while the attention of the block before runs (540 -> about 105 in both phases, the same bytes over the bus, 16x more sync points); the pool and cuBLAS part of the remainder (100 - 250, measure first); a smaller drafter quant (about 300).

### Quality of the blocked attention: the teacher carries the single attention's bias (2026-11-19)

A wider check of the blocked attention at 100k (row2660, kvq-eval fresh prefill, 8 per batch, KL to the A1 f16 teacher; `kvq-scratch\mem\attnsplit\QUALITY.md`) gave old path 1.63 / 1.49 / 1.34 / 1.41e-4 at ub 512 / 1024 / 2048 / 4096 (sd 0.13e-4) against block 16k 2.78 / 3.03 / 2.08 / 2.58e-4, block 8k 2.80e-4, 32k 2.41e-4, 64k 1.30e-4 (ub 2048): every new run above every old run, about 9 sd of the old path's ubatch spread, growing with the number of blocks (1 block + masked tail 1.29e-4, 2 blocks 1.30, 3 blocks 1.63, 4 blocks 2.3 - 2.4, 7 blocks 2.08, 13 blocks 2.80e-4) while the last prefill position is equal to the old path to 1e-13. At 60k and 80k (row1094, row2711) the ratio new / old is 0.9 - 1.34 on KL of 1e-5 - 4e-5, inside the noise. Top-1 agreement unchanged (0 - 2 of 471 either way).

Diagnosis (`DIAG.md`): no bug. The merge chain is F32 end to end (results, views, sub / sigmoid / mul / add / softplus), output and LSE come from the same f32 row sum, the CUDA LSE error against the CPU is 0.7 - 2.1e-4 (a block misweighted by 0.02%). The cause is the teacher: the A1 reference computes its prefill as one flash attention over all rows, like the old path, and on Turing and later the CUDA MMA kernel holds the probabilities and the P * V accumulator in half precision against a running maximum over all rows, so the probability mass far below the row maximum is lost over a long tail. In attention layer 31 of the last 100k ubatch (real data, 99,584 rows, 64 queries, 24 heads, exact f64 attention on the CPU as the reference): the single attention has nmse 3.34e-4, mean abs error 3.1e-3 and an output bias of -1.02% (in head 0, 8.6% of the mass sits below 6.1e-5 of the row maximum); the 16k-blocked graph has nmse 2.57e-5, 1.0e-3 and -0.31%, closer on 1,520 of 1,536 (query, head) pairs. Teachers built from the same f16 model with the plain path's prefill attention blocked the same way (`LLAMA_DIAG_FA_BLOCK`, a diagnostic switch kept in `attnsplit-diag-dump.diff`, not committed) are themselves 1.99 / 2.31 / 2.46 / 2.69e-4 from A1 at 16k / 8k / 4k / 2k and 0.9 - 1.5e-4 from each other, so at 100k a KL is only good to about 1e-4 and any change of the reduction order moves it by that much. Scored against those teachers the ranking reverses: block 16k 1.17 - 1.52e-4 (16k teacher), 1.14 - 1.47e-4 (8k), 1.26 - 1.82e-4 (4k), 1.36 - 1.78e-4 (2k) against the old path 1.64 - 2.53, 1.91 - 2.77, 2.14 - 2.62, 2.06 - 2.81e-4; the worst new run beats the best old run against every blocked teacher. Consequences: block 16k stays the default; the KL numbers of this document are against teachers with the single attention's bias and are comparable among themselves, not a distance to exact attention; a KL against a single-attention teacher cannot rank attention variants that change the reduction order, for that the blocked teacher logits (`ref/A1b{16,8,4,2}k-row2660.ref`) or the exact check on dumped rows (`scripts/exact.py`) are the tools. Smaller blocks would be closer to exact still (synthetic peaked attention at 98k rows against f32: single 2.5e-5, 64k 1.6e-5, 16k 4.2e-6 nmse), traded against speed and memory. The bias itself is in ggml's standard CUDA flash attention for any long prefill on Turing and later, not specific to this branch.

### Scratch attention of the prompt in key blocks (2026-11-19, commit 6d2ecbfab, not bit-identical)

Measured alone on 15b59d61d (without the codes change below). The target's pp compute buffer was mostly the F16 copies of K/V and the F16 mask that flash attention makes over all n_ctx + ring scratch rows (1,809.5 MiB at 250k, ub 1024). Now the host rows go through `ggml_flash_attn_ext_lse` in mask-free key blocks (`--kvq-scr-block`, default 16384) and only the rest has a mask (see How it works, prefill). The new op is FLASH_ATTN_EXT with op param 5 = 1: the result has ceil(n_batch / DV) more rows in dim 2 that hold the F32 [n_head, n_batch] log-sum-exp (scale, softcap, mask, sinks included). CPU: the one-chunk and tiled paths. CUDA: the MMA kernel only (Turing+); a whole tile writes KQ_max + log(KQ_rowsum) from its shared memory meta, a tile split by stream-k gets it from the fixup kernels; no new combine kernel. The GQA kernel variants (ncols2 > 1) loaded the mask unconditionally, so a mask-less attention ran the ncols2 = 1 variant, 1.7x slower; they now load the mask only if present, and the dispatch uses them without a mask for the LSE op only (other users keep their kernel). The reserve graph uses the same blocks (`kvq_params_reserve` passes `scr_block`).

VRAM, dedicated usage of the process (MiB), build before -> B 16k (target pp compute buffer in brackets):

| | load | prompt peak | generation |
|---|---|---|---|
| 250k ub 1024 | 17,187 -> 15,862 | 17,369 -> 16,028 (1,809.5 -> 484.7) | 15,163 -> 15,154 |
| 250k ub 1024, B 8k | -> 15,798 | -> 15,964 (420.7) | -> 15,154 |
| 250k ub 4096 | 19,628 -> 17,008 | 20,300 -> 17,698 (4,187.0 -> 1,570.0) | 15,310 -> 15,314 |
| 100k ub 1024 | 14,860 -> 14,436 | 15,042 -> 14,602 (909.5 -> 484.7) | 13,638 -> 13,630 |
| 100k ub 4096 | 16,512 -> 15,582 | 17,182 -> 16,271 (2,498.0 -> 1,570.0) | 13,814 -> 13,822 |

The pinned host compute buffer drops too (250k ub 1024: 588.5 -> 88.5 MiB). At ub 4096, 1,570 MiB is the floor of the other activations (8k and 16k give the same). Speed (server, first prompt): 250k ub 1024 199.9 -> 140.1 s (-30%), ub 4096 165.5 -> 136.9 s (-17%), 100k ub 1024 43.0 -> 37.4 s (-13%), ub 4096 39.9 -> 37.3 s (-6%); 16k is the fastest block size measured (8k +3.5%, 32k +4%, 64k +24% at 250k). The gain is the CPU fill and upload of the big mask (500 MB per ubatch at 250k) and the GQA kernel without a mask. Decode unchanged (the packed path is not touched). Follow-up prompt at 100k 1.09 - 1.24 s against 1.18 - 1.35 s.

Quality (kvq-eval, fresh prefill -b 2048 -ub 2048, 8 per batch, KL to the f16 teacher): at the first scored position every block size is equal to the old path (KL 1e-14 - 2e-13); the runs diverge later because the chunks are fitted from rows with other rounding. KL at 20k 2.65e-4 (old 2.57 / 2.84e-4 at ub 2048 / 1024), 50k 2.74e-4 (old 2.27 / 2.55e-4), 100k 2.08e-4 (old 1.34 / 1.49e-4); KL to the old path 2.0e-4 - 2.3e-4, about the spread of the old path at another ubatch (1.4e-4 - 2.0e-4). At 100k three of four new runs are above both old runs; settled by the diagnosis above (the teacher carries the single attention's bias, the blocked attention is closer to exact). Server ids differ from the build before at the default block (100k r1 a9d499cedd8f against f6400c101d9d); `--kvq-scr-block 0` gives the old ids. Tests: FLASH_ATTN_EXT_LSE 84/84 (with `lse_only` cases), FLASH_ATTN_EXT 4382/4382 after the mask-load change, KVQ_ATTN, KVQ_FIT, test-kvq-codec, test-kvq-attn pass. Open: the node count depends on n_host (number of blocks), so the scheduler reallocates the graph at almost every prompt ubatch (the buffer does not grow; the cost is inside the prompt times); other backends do not know op param 5 and need a supports_op check; the op needs ne3 == 1. Numbers, scripts and notes: `kvq-scratch\mem\attnsplit` (`RESULTS.md`, `NOTES.md`, `attnsplit.diff`).

### Codes in host memory during the prompt (2026-11-18, commit affa1fa81)

Owner's decision: the prompt never reads the codes and tables (1,830 MiB at n_ctx 256000; prefill attends over the q8_0 scratch), so during the prompt they live in pinned host memory only and the device copy exists only in the decode phase (see How it works, codes and tables per phase). VRAM, dedicated usage of the process (MiB), build before -> this build:

| | load | prompt peak | generation | follow-up prompt peak |
|---|---|---|---|---|
| 250k n_max 7 (n_ctx 256000) | 17,187 -> 15,364 | 17,369 -> 15,546 | 15,163 -> 15,168 | 17,299 -> 15,472 - 15,584 |
| 250k n_max 3 | 17,173 -> 15,350 | 17,355 -> 15,532 | 15,061 -> 15,066 | 17,235 -> 15,408 - 15,520 |
| 100k n_max 7 (n_ctx 102400) | 14,860 -> 14,132 | 15,042 -> 14,314 | 13,638 -> 13,640 | 14,916 - 15,028 -> 14,184 - 14,296 |

So 250k with `--kvq` fits the 15.5 GiB budget in every phase (288 MiB of headroom at the prompt peak). Pinned host memory +1,830 MiB at 256k. Generated ids identical to the build before in every session (24k, 50k, 100k, 250k, n_max 7 and 3, follow-up prompts, a 5000-token reply with a fold, slot save and restore, the old build's slot file; saved slot files byte-identical); kvq-eval bit-identical (chunk 1 smoke run, folds every 64, state save / load with rollback, turns, rb-test, `--kvq-lib-full`); KVQ_ATTN, KVQ_FIT, test-kvq-codec, test-kvq-attn pass. Speed: first prompts, decode per verify batch and the end-of-prompt wait within 0.3% (100k pairs and 4 pairs at 250k). The follow-up prompt pays the allocation and the upload of the device copy: the server sends the last 4 prompt tokens as their own batch, which takes the packed path (it needs all codes) before the end-of-prompt fit, so the upload (about 34 GB/s) is on its critical path: +20 ms at 100k (1.6% of 1.25 s), +55 ms at 250k (3.2% of 1.71 s), plus 12 - 26 ms of cudaFree at the start of the next prompt. Only when the last prompt batch takes the scratch path (first prompts at 24k and 50k here) does the upload overlap the CPU part of the end-of-prompt fit. A first version allocated the device copy in `apply()` of that 4-token batch, before the context freed the pp compute buffer, and reached 17,016 MiB in a follow-up prompt; the context now allocates it after its phase switch. An upload per layer on its own stream, with the main stream waiting for a layer at the join of the layer before (all layers queued at once, or 1 - 4 layers ahead of the reader), was built and is bit-identical but hides nothing: the 4-token batch plus the wait takes 661 - 680 ms against 660 with the upload in front of it and 640 in the build before at 100k, 742 - 747 against 741 and 684 at 250k. The H2D copies of all streams appear to serialize on the copy engine with the batch's own resident uploads (not verified with nsys), so it was not kept. The cudaFree at the start of a prompt is all driver time (no wait for queued work). Owner's decision: the +3.3% (57 ms) of the 250k follow-up prompt is accepted. Not measured: PCIe 4.0 / 3.0. Numbers, scripts and notes: `kvq-scratch\mem\codes` (`RESULTS.md`, `NOTES.md`, `codes-v2.diff`).

### Memory per phase and GDN replay (2026-11-18, commits 4218b5b9f, 9a754beaa, fba885494)

Owner's goal: the server with `--kvq` fits at 250k context into 15.5 GiB of VRAM (15,872 MiB of dedicated usage), prefill and decode speed within +-2%. The server has one slot, so a prompt (ubatches of 1024 tokens over the scratch) and a reply (packed verify ubatches of at most 8 tokens) never overlap: each phase should hold only its own buffers. Starting point: a memory inventory of the process at n_ctx 102400 (39 rows: where, size, formula, lifetime, needed in prefill / decode; `kvq-scratch\mem\inventory\INVENTORY.md`). At 250k the build before (a724f2299) needed 19,277 MiB after load, 19,483 in the prompt and 19,579 in generation (n_max 7).

GDN replay. ggml-org PR 31669 (am17an, open, WIP) replaces the n_rs_seq recurrent snapshot rows (7 x 149.6 MiB at n_max 7) by replay: the committed state plus the pending inputs of at most n_rs_seq + 1 tokens per cell, the GDN op recomputes the state at a rollback. Merged as 4218b5b9f (textually clean; our hybrid memory now passes `rs_replay`), the server turns it on with `--kvq`. Recurrent buffer 1197.0 -> 176.8 MiB at n_max 7 (598.5 -> 163.2 at n_max 3). The output is bit-identical (32 sessions: 24k, 50k, 100k, n_max 7 and 3, with and without `--kvq`, checkpoint restore into a shorter prefix, slot save and restore), decode 0.6 - 1.4% faster, prompt equal. Cost: a context checkpoint, a slot file or a RAM cache entry grows by 27.1 MiB (the pending inputs are written whole), up to 32 checkpoints = 868 MiB of RAM. The PR's stored layout of the pending inputs is still under discussion upstream; if it changes, slot files and checkpoints of this build cannot be loaded by the next one.

Compute buffers per phase (`llama_context`, every context with one sequence, a device buffer and n_tokens_tg < n_ubatch; context param `phase_bufs`, off by default; `--kvq` and the DFlash drafter turn it on, `--[no-]phase-buffers` overrides). `ggml_gallocr_resize` frees a compute buffer and allocates it again with a given size while the node plan stays; `ggml_gallocr_needs_realloc` asks for a reserve when a buffer is missing or smaller than the plan, so a graph that does not fit reserves as usual. `sched_reserve` reserves the pp graph (exact sizes) and a tg graph of n_tokens_tg tokens (n_outputs_max = n_max + 1 for the target, the block size for a DFlash draft). Per ubatch, after `mctx->apply()`: a tg ubatch has at most n_tokens_tg tokens and is not on the prompt path of the two-tier cache (the same predicate as the graph, `llama_kvq_use_packed`); on a change `phase_set` waits for the backends and the store's streams and resizes the device compute buffers (host buffers keep their size). With `--kvq` the reserve graph is the scratch attention, so the tg sizes start at 0 and the largest size reached is kept (1 - 3 graph reallocations in the first reply after load). Two schedulers were rejected: the scheduler is captured by the kvq triggers, the callbacks and the graph params.

Other buffers. The drafter's pp reserve is the injection graph (120 MiB) instead of the 1024-token graph with full-vocabulary logits (1098.6 MiB); its token graph of 8 tokens is its tg phase (8.6 MiB). The +108 MiB re-reserve of the first request (the layer input copies join the graph) happens at load. The layer input buffer is sized per batch (the drafter reads it per target batch): n_batch rows (400 MiB) in prefill, n_tokens_tg rows (0.78 MiB) in decode. The scratch is set up by `apply()` of every prompt-path ubatch and freed after the end-of-prompt fit (or a fold) once the small ubatches take the packed path; the fitter's device stage and graph allocator (83 MiB) are created by its first job and freed with it. The decode-only resident rotating buffers, tail buffers and staging block (88 MiB) hold nothing between rounds and share the memory of the scratch (one buffer of max(scratch, 88 MiB), zeroed at each switch): separate buffers cost 5 - 12 ms of cudaFree / cudaMalloc per prompt on WDDM.

VRAM, dedicated usage of the process (MiB), build before -> this build:

| | load | prompt peak | generation | follow-up prompt peak | process peak |
|---|---|---|---|---|---|
| 100k n_max 7 | 16,950 -> 14,860 | 17,126 -> 15,042 | 17,196 -> 13,638 | - | 17,196 -> 15,042 |
| 100k n_max 3 | 16,352 -> 14,846 | 16,528 -> 15,028 | 16,574 -> 13,584 | - | 16,576 -> 15,028 |
| 250k n_max 7 (n_ctx 256000, 250,000-token prompt) | 19,277 -> 17,187 | 19,483 -> 17,369 | 19,579 -> 15,163 | 19,579 -> 17,299 | 19,581 -> 17,369 |
| 250k n_max 3 | 18,679 -> 17,173 | 18,885 -> 17,355 | 18,929 -> 15,061 | 18,931 -> 17,235 | 18,931 -> 17,355 |
| plain 24k, no `--kvq` | 19,161 -> 18,150 | 19,295 -> 18,316 | 19,345 -> 17,377 | - | 19,345 -> 18,316 |

Buffer sizes at 250k, this build: recurrent 176.8; target compute 1809.5 in the pp phase, 0 -> 72.4 in the tg phase; drafter compute 120.0 / 8.6; layer input 400 / 0.78; codes 1562.5, tables 268.0, scratch 540.3 (the decode-only buffers inside it), host store 8500 pinned, target host compute 588.5 pinned. Against the budget of 15,872 MiB: generation at 250k is 709 MiB under it, the prompt phase 1,497 over and the idle state after load 1,315 over (the pp buffers are allocated at load). What the prompt phase still holds that decode does not: the target pp compute buffer (1809.5, of which about 1,490 are the f16 K and V copies and the f16 mask of the flash attention over n_ctx + ring rows for a 1024-token ubatch), the scratch (540), the layer input buffer (400), the drafter injection graph (120), the fitter (83).

Speed. 100k, same-day pairs of two sessions: prompt 42.69 -> 42.63 s (n_max 7) and 42.76 -> 42.67 s (n_max 3); decode per verify batch 26.01 -> 25.62 ms (-1.5%) and 20.17 -> 20.05 ms (-0.6%). 250k (against a baseline from earlier the same day): prompt 205.8 -> 198.2 s, 300-token reply 4080 -> 4082 ms (n_max 7). Plain 24k equal (6.50 / 6.48 s, 17.66 / 17.65 ms). Switch cost at 100k: into the pp phase 17.6 ms per prompt (12 ms of it the zeroing of the scratch after an idle reply, 3.2 ms target, 1.7 ms drafter, 0.6 ms layer input), into the tg phase about 5.8 ms per reply; plain mode 13 ms into pp (free 404 + allocate 880 MiB; cudaMalloc time on WDDM is not proportional to the size). Generated ids identical to the build before in every session (24k, 50k, 100k, 250k with the follow-up prompt, plain mode, slot save and restore, also the reference's slot file loaded by this build). Tests: `test-backend-ops` GATED_DELTA_NET 59/59 (16 new replay cases), KVQ_ATTN 36/36, KVQ_FIT 14/14; `test-kvq-codec`, `test-kvq-attn`.

Owner's decisions (2026-11-18): the budget is 15.5 GiB; plain mode keeps the phases (the 13 ms switch is 3% of a short follow-up prompt, which was still not slower end to end); the WIP state of PR 31669 is accepted. Not exercised: folds (the fitter allocated on demand in decode), a rollback into the coarse region during generation (the scratch on demand), `--kvq-tail 0`, the MTP drafter, kvq-eval, more than one slot (phases off). Not done: the layer input buffer at n_ubatch rows in prefill (-300 MiB, the drafter would have to inject per target ubatch). Numbers, scripts and the agents' notes: `kvq-scratch\mem\inventory`, `mem\phase` (buffers per phase, `RESULTS.md`, `NOTES.md`, `wp1.diff`), `mem\replay` (the PR merge), `mem\m250` (250k baseline, the 281k token file), `mem\comb250` (the combined build).

### The ring in pinned host memory (2026-11-18)

Owner's goal: the ring (the q8_0 rows of the positions after the host store, 4352 cells, 144.5 MiB of VRAM for 16 layers) leaves the device like the resident rows, with one wait point per attention layer on the main stream, no kernel change, and no cost at verify batches of 4 and 8 at PCIe 5.0 / 4.0. Mode name in the code: host ring (tail mode with the rotating buffers); `--kvq-tail 0` keeps the old on-device path.

- The ring cells of the attention cache move to pinned host memory at `fit_enable` (`llama_kv_cache::move_bufs`); cells, masks, `find_slot`, the state format and the spill and fold logic are unchanged. The device has 4 rotating tail buffers (36.1 MiB), buffer = attention layer index mod 4, each with the layout of the cache tensors, so the op reads the tail rows with the same strides and block partition as before (needed for bit identity).
- Why 4 buffers and not the `[tail | resident]` block of the phase 1 notes: the early query of layer i (at the input of layer il - lead) scores against the ring K of layer i, so its tail must be on the device one layer before the resident rows. The main stream waits for it at the join of layer i - 1 (the same callback as the join wait); the worker uploads it after the join of layer i - 3 (`ev_sel` of that prefetch is after the attention of layer i - 4, the previous user of the buffer, plus the readback event of that layer). Consecutive users must be at least 3 layers apart cyclically, which 3 buffers cannot do over 16 layers.
- Upload: the occupied cells [n_host, end) of K and V (two copies, four on a wrap), i.e. the reply so far. All 4352 cells (9 MiB per layer) cost +0.25 - 1.0 ms per verify batch at PCIe 5.0 and about +2 ms at the PCIe 4.0 emulation, rejected.
- New rows: the cpy nodes write the K/V of the ubatch into the tail buffer (same set_rows as the cache); the join reads those cells back into the host ring on the copy stream (two small copies after the delta upload, after an event of the main stream at the join).
- Prefill (the scratch path, now every non-packed ubatch): the K/V of the ubatch go straight into the scratch rows of their positions, read back into the host store per kv head at the next scratch up (before the uploads of the next layer; a trigger after the last attention layer); the scratch up uploads the store rows [0, p0). So the host store holds every row after a large ubatch and the fitter no longer copies ring rows on the device (phase 1 of a job uploads all its rows from the store, a spill only moves n_host); the host ring holds the rows after a packed ubatch; the missing side is copied on the host at the switch. Without host rows (the first ubatches of a prompt, before: the normal attention over the cache) the scratch uses the cells as rows and the cache's mask.

| | before (phase 1) | after |
|---|---|---|
| ring on the device | 144.5 MiB | 36.1 MiB (4 tail buffers) |
| ring in pinned host memory | - | 144.5 MiB |
| VRAM in generation (process, n_ctx 102400), n_max 7 / 3 | 17,286 - 17,300 / 16,676 - 16,678 MiB | 17,188 - 17,196 / 16,572 - 16,584 (-94 to -104 MiB) |
| host to device per round (300-token replies) | 285 MB base + 15 - 40 MiB delta | + 2 - 5.6 MiB tail |

Bit identity (kvq-eval logits files, against the phase 1 build): six rows from the states at k8 and k1 (cap 6144, also equal to the build before phase 1), five rows at k8 at the default cap, folds (`LLAMA_KVQ_FOLD_EVERY=64`), state save / load with a checkpoint and rollback, turns with 300 generated tokens, `--kvq-rb-test`, `--kvq-lib-full`; fresh prefills of 20k and 100k give byte-identical state files. Server ids equal in every cell (24k / 50k / 100k, n_max 7 and 3, slow-bus emulation, 3000-token replies). test-kvq-attn and test-backend-ops KVQ_ATTN / KVQ_FIT pass.

Server, DFlash2, real prefill, decode ms per verify batch (phase 1 build / this build, ids equal):

| | 24k n_max 7 / 3 | 50k n_max 7 / 3 | 100k n_max 7 / 3 |
|---|---|---|---|
| 300 tokens or EOS, 2 sessions each | 21.47 - 21.50 / 17.59 - 17.60 against 21.51 - 21.64 / 17.51 - 17.65 | 22.66 - 22.72 / 18.35 - 18.36 against 22.68 - 22.75 / 18.28 - 18.37 | 25.77 - 25.91 / 20.13 - 20.14 against 25.78 - 25.89 / 20.06 - 20.14 |
| prompt s | 7.30 - 7.37 against 7.22 - 7.24 | 16.80 - 16.87 against 16.61 - 16.64 | 43.23 - 43.31 against 42.69 - 42.78 |

kvq-eval (row1830, ms per token): 4 per batch 4.518 / 4.523 against 4.498 / 4.501, 8 per batch 3.060 / 3.068 against 3.062 / 3.047. Slow bus (`LLAMA_KVQ_UP_REPEAT=k`, base and tail uploads repeated): k = 2 equal (server 50k 23.15 / 18.97 against 23.15 / 19.03); k = 3 +0.2 - 0.8 ms per verify batch (24.67 / 21.53 against 24.90 - 25.31 / 22.09 - 22.35), the tail bytes on a bus the resident base already fills. Long replies (3000 tokens, `ignore_eos`, the ring fills to 3000 cells, the tail adds about 47 MB per round on average): +0.32 ms per verify batch at n_max 7 (24.13 against 24.45), +0.12 at n_max 3, +0.63 at k = 2, +1.6 at k = 3. Numbers, scripts and the design notes (race argument, alternatives): `kvq-scratch\mem\vram\tail` (`RESULTS.md`, `NOTES.md`).

### Resident rows from pinned host memory (2026-11-18)

Owner's design: the whole union of the verify batch stays the request, the capacity was to go to 6144 (the cap sweep: p1024 c6144 k8 as good as c8192 at 20k - 100k; the default stays 8192, see the decision below), and the resident rows no longer live on the device for all 16 attention layers. Per layer the rows live in a pinned host mirror `[2][G][cap_tot][ROW]` (13.4 MB at 6144 slots, 204 MiB for 16 layers); the device has two rotating buffers, buffer = attention layer index mod 2. Per round and layer: the base upload of the whole mirror into the buffer, then the delta as before (plan on the bitmaps, gather, one upload, set_rows scatter), the attention op reads the buffer (no kernel change, the graph passes the buffer's rkv / ridx). The slot assignment is unchanged, so at the same capacity the logits are bit-identical to the build before.

- Base upload of layer i + 1: queued by the worker on a third stream of the device right after the join uploads of layer i (the copy engine takes the copies in order), after the event of layer i's prefetch, which the main stream records after the attention of layer i - 1, the previous reader of the buffer (needs lead 1 - 3; enable refuses others now). General n_buf works (checked bit-identical with 3 and 4 buffers). The join of layer i + 1 waits for it on the copy stream before its own upload; the main stream still waits only at the join.
- Mirror: the worker copies the fetched rows from the staging into the mirror after the job (the join does not wait for it). The base upload of the same round may read the mirror meanwhile: those slots are scattered from the staging after it. The next base upload of the layer comes from the worker after this job.
- One staging block for all layers, device `[ridx | slot list | rows]` (13.1 MiB) and the same layout pinned (17.1 MiB): a join is one copy. A layer's prefetch is read back after the join upload and the attention of the layer before, so the shared block is free.
- Where the base goes was measured (kvq-eval, 4 tokens per batch, ms per token, base 4.43): at the prefetch of the layer before 4.60 (the delta upload of that layer queued behind the 13 MB copy), before the join uploads 4.77 - 5.07, right after them 4.50, after the mirror write 4.55, after the GPU passed the join 4.56; pieces of 2 MiB / 1 MiB / 256 KiB slower than one copy (4.55 / 4.67 / 5.08). With the DMA skipped (wrong logits) 4.43, so the cost is the copy itself: in nsys the kernel time is equal and the GPU gaps at the joins grow while a base copy is in flight (888 -> 1,398 us per round).

| | before (8192 slots per layer on the device) | after (cap 6144, 2 buffers) |
|---|---|---|
| resident + device staging | 278.9 + 276.0 MiB | 26.1 + 13.1 MiB |
| pinned host (staging, mirrors) | 282.5 MiB | 17.1 + 204.2 MiB |
| VRAM in generation (process, n_ctx 102400) | 17,788 - 17,802 MiB (n_max 7), 17,180 (n_max 3) | 17,274 - 17,288, 16,666 (-514 MiB) |
| host to device per round | 14 - 38 MiB of fetched rows | + 214 MB of base uploads (16 x 13.4 MB) |

Server, DFlash2, real prefill, 300 tokens, decode ms per verify batch (2 sessions each, ids of the two builds):

| | 24k n_max 7 / 3 | 50k n_max 7 / 3 | 100k n_max 7 / 3 |
|---|---|---|---|
| before (cap 8192) | 21.19 / 17.29 | 22.39 / 18.03 | 25.54 / 19.83 |
| after (cap 6144) | 21.84 / 17.52 (ids equal) | 23.10 / 18.20 (ids differ) | 26.27 / 20.04 (ids equal) |
| after, `--kvq-resident-cap 8192` | | 22.72 / - (ids equal) | 25.84 / 20.17 (ids equal) |

The base uploads cost 0.25 - 0.35 ms per verify batch at equal replies (also 50k against the old build at `--kvq-resident-cap 6144`: +0.36 / +0.24 ms, ids equal); at n_max 7 the cap 6144 adds about 0.4 ms against cap 8192 (the cut path: the scores are read back, more rows are fetched). Prompt time unchanged, GPU idle at the joins 0.003 - 0.06 ms per ubatch (before 0.02 - 0.09). Cap 8192 with the rotating buffers costs 12 MiB more VRAM than 6144 and 71 MB more pinned. Owner's decisions (2026-11-18): the 0.3 ms per verify batch is accepted for the 514 MiB; the default capacity stays 8192 (faster at n_max 7, output equal to the builds before; VRAM is no longer the reason for 6144, which remains available through `--kvq-resident-cap`).

Quality (kvq-eval from the saved states, KL to the f16 teacher): at cap 6144 the logits files of the new build and the old build at `--kvq-cap 6144` are identical on all six rows at 8 per batch and 1 per decode; the KL equals the c6144 row of the cap sweep: 8 per batch 2.30e-4 / 6.44e-5 / 2.21e-4 / 1.15e-5 / 3.38e-5 / 1.39e-4 at 20k / 40k / 50k / 60k / 80k / 100k (c8192: 2.57e-4 / 8.55e-5 / 2.27e-4 / 1.98e-5 / 3.61e-5 / 1.34e-4). At one token per decode cap 6144 is worse on all six (e.g. 7.7e-5 against 6.4e-5 at 40k), inside the intervals; not a production case.

Slow bus (`LLAMA_KVQ_UP_REPEAT=k` issues every base upload k times): kvq-eval ms per verify batch over the old build, 4 / 8 tokens: k = 1 +0.30 / +0.37, k = 2 (PCIe 4.0-like) +0.64 / +1.24, k = 3 (PCIe 3.0-like) +1.61 / +3.22; server 50k k = 3: +2.38 ms at n_max 7 (GPU idle at the joins 1.05 ms), +1.24 at n_max 3. The traffic is linear in the cap; a smaller cap is the fallback on a slow bus (owner's decision: no layers kept on the device).

Possible later improvements (owner's decisions: skipped for now). More than two rotating buffers (`LLAMA_KVQ_NBUF`, 17.8 MB of VRAM each at cap 8192): the base upload of a layer may then start earlier than the attention of the previous reader of its buffer, which widens the window for the copy on a slow bus; measured bit-identical with 3 and 4 buffers and no speed difference on this PCIe 5.0 machine, where the bytes per round, not the window, set the cost; worth a measurement on PCIe 4.0 / 3.0 hardware, together with the cap. The union bitmap and the compaction on the GPU, so the device knows the distinct row count of a verify batch and the staging and the scatter are sized exactly. The plan is not on the critical path (GPU idle at the joins 0.02 - 0.09 ms per verify batch), the shared staging is one 13 MiB block, so the gain is about 13 MiB of VRAM and 13 MiB of pinned memory plus the trash entries of the scatter; it needs the row count on the device (a set_rows that skips negative indices or a kvq scatter op, a ggml change) and, for a cut at cap < union, the max-score order on the device. Numbers, scripts and the design notes: `kvq-scratch\mem\vram` (`RESULTS.md`, `NOTES.md`).

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

### Against plain f16, q8_0 and q4_0 caches (2026-11-17, build 2758a0a64)

The same binary with four caches: A plain f16 K/V, B plain q8_0 (default attention rotation, as the server runs it), C `--kvq` (defaults), D plain q4_0 (default rotation). IQ3_S target. Full tables with the intervals, t/s and the replies: `kvq-scratch/docs-archive/kvq-cache-comparison.md` (outside the repository); raw numbers: `kvq-scratch\mem\comb\final-compare.txt`.

Quality (kvq-eval, teacher forced on the recorded replies, teacher A at one token per decode; KL at 1 per decode / 8 per batch, top-1 flips at 1 per decode):

| Context (scored positions) | A, 8 per batch (floor) | B q8_0 | C --kvq | D q4_0 | C against q8_0 without rotation |
|---|---|---|---|---|---|
| 24k (245) | 6.2e-5 | 1.30e-4 / 9.3e-5, 0 | 1.07e-4 / 1.07e-4, 0 | 8.9e-4 / 1.02e-3, 2 | 4.5e-5 / 3.7e-5 |
| 50k (289) | 6.5e-5 | 1.77e-4 / 1.83e-4, 2 | 2.30e-4 / 2.27e-4, 3 | 1.73e-3 / 1.81e-3, 4 | 1.17e-4 / 1.09e-4 |
| 100k (471) | 6.7e-5 | 1.77e-4 / 1.78e-4, 1 | 1.23e-4 / 1.34e-4, 1 | 1.01e-3 / 1.02e-3, 2 | 8.1e-5 / 8.9e-5 |

Against f16 the two-tier cache is about as far as plain q8_0 (lower at 24k and 100k, higher at 50k, all intervals overlap); q4_0 is 5 - 10x further. Plain q8_0 with the rotation is worse than without it at 24k and 100k (1.02e-4 and 1.22e-4 without).

Speed (llama-server with the DFlash2 drafter, n_ctx 102400, real prefill, 300 tokens, two sessions each; decode ms per verify batch at n_max 7 / n_max 3, difference to A):

| Context | A f16 | B q8_0 | C --kvq | D q4_0 |
|---|---|---|---|---|
| 24k | 16.57 / 14.01 | +1.2 / +1.0 | +4.6 / +3.4 | +1.6 / +1.5 |
| 50k | 17.53 / 15.05 | +3.0 / +2.9 | +5.0 / +3.1 | +3.1 / +3.2 |
| 100k | 19.64 / 17.17 | +6.7 / +6.5 | +6.0 / +2.8 | +6.4 / +6.4 |

Prefill against A: C +0.8 / +1.8 / +3.5 s, B and D 0 / +0.2 / +2.2 s at 24k / 50k / 100k. The continuations differ between the caches in most cells, so t/s (in the file) compares different replies.

Memory (dedicated usage of the process, MiB): at n_ctx 102400 in generation A 22,023, B 19,345, C 17,802, D 17,745 (against A: -2,678, -4,221, -4,278); after load at 51200 / 204800: A 18,538 / 28,439, B 17,160 / 23,161, C 16,788 / 19,107, D 16,360 / 19,961, so per 1000 positions of n_ctx A 64.5, B 39.1, C 15.1, D 23.4 MiB. C also pins the q8_0 store in host RAM (about 4.3 GB at 102400).

So against q4_0, the other way to shrink the cache: the same VRAM at 102400 (C 57 MiB more), 854 MiB less at 204800, 7 - 8x lower KL against f16; decode faster than q4_0 at 100k (-0.4 / -3.6 ms) and slower at 24k (+3.0 / +1.9 ms) and 50k (+1.9 / 0). The f16 cache is the fastest at every length (no dequantize in the attention).

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

Memory (n_ctx 102400, launch line of `kvq-scratch/docs-archive/dflash2-vram-baseline.md` with `--kvq`): resident 274.7 MiB (+4.25 MiB trash rows in the final build) against 34.5 MiB, staging device 276.0 MiB, pinned host staging 282.5 MiB (before about 36 MiB each). VRAM during generation 17,820 MiB (dedicated usage, r1 of the memory session) against 17,315.5 MiB, +504.5 MiB. These buffers are sized for the worst case (the union of 8 positions with nothing resident, every slot fetched in one round) and can be made smaller later: they can live in the prefill scratch (217 - 486 MiB, idle during generation), the fetch can be uploaded in pieces, or they can be sized by the largest verify batch in use (n_max + 1). Not done.

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
- Compute buffers per phase (2026-11-18, owner's decisions): the budget is 15.5 GiB of VRAM at 250k; prefill and decode never overlap (one slot), so each phase holds only its buffers; the plain server gets the same phases (13 ms per switch into the prompt phase, accepted); the decode-only kvq buffers share the scratch memory instead of their own buffer (no cudaMalloc per prompt, 133 MiB more in generation, the peak is the prompt phase anyway); the GDN replay of ggml-org PR 31669 is merged although the PR is WIP (bit-identical output, -1,020 MiB).
- 250k fit (2026-11-19, owner's decisions): the codes and tables leave the device during the prompt (their upload sits on the server's final 4-token packed batch, +3.3% on the 250k follow-up prompt phase, accepted: "simple is better", the per-layer overlapped upload hid nothing and was dropped); the prompt's scratch attention runs in mask-free 16k key blocks merged by the log-sum-exp, confirmed as the default after the diagnosis showed the single-attention teacher's bias (see Results, quality of the blocked attention); `--kvq-scr-block 0` keeps the old, byte-identical attention; `--spec-draft-n-ubatch 1024` is the way to run ub 4096 at 250k.
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
- The table fit runs on the GPU since 2026-11-20 (release blocker R7); `--kvq-threads` is gone. Before, 128 problems of 256 x 256 per job ran on CPU threads (about 2 CPU seconds, end-of-prompt wait 0.55 s with the default 4 threads).

## Open bottlenecks, largest first

1. Prefill: 44.93 s against 44.01 s of production the same day at 100k (0.55 s of it the end-of-prompt fit); the fitter does not stall it. What is left per chunk: the device work and 570 MB of host uploads of phases 2 and 3 on the main stream, two synchronizes (statistics readback, join), the D2H of the q8_0 rows. A second stream (or the copy stream for the transfers) could hide most of it; V-only rows in phase 2 would save a quarter of the uploads.
2. The end-of-prompt fit: 0.55 s per prompt on 4 threads whatever the row count (the 256 x 256 problems); it dominates short follow-up prompts (554 tokens: 1.10 s against 0.49 s). More threads at the end of the prompt, or a device eigen solver, would shorten it.
3. The latency of a job is set by the polls (one per ubatch): after the CPU tasks, a job needs one or two more ubatches. With 1 thread and -ub 2048 (kvq-eval) 3 batches still waited 0.35 s at 100k.
4. Target compute buffer of the prompt phase, 1.81 GB at n_ctx 256000: mostly the F16 copies of K and V and the F16 mask that flash attention makes over n_ctx + ring rows for a 1024-token ubatch. Since 2026-11-18 it is freed during the reply (compute buffers per phase), so it only counts in the prompt phase. Since 2026-11-19 the scratch attention runs in key blocks (485 MiB at ub 1024, 1,570 at ub 4096, see Results, attention split) and the codes and tables leave the device during the prompt, so 250k fits in every phase; at ub 1024 the decode phase (codes back on the device, 15,159 MiB) now sets the headroom (601 MiB), at ub 4096 with `--spec-draft-n-ubatch 1024` the prompt peak does (500 MiB). What is left per phase at 250k and what could still be squeezed is listed in Results, combined build (the scratch as a ring of blocks is the one item that moves decode); beyond that the owner's idea of running the prompt through two halves of the layers in turn for huge requests.
5. MTP context cache: 899 MiB f16 for the full context.
6. Verify batches of 8 (DFlash2): 26.0 ms per batch at 100k, the same as the plain cache (26.1), but about 4 ms per round that do not shrink with the context (21.6 against 17.45 ms at 24k; see Results, attention op kernels): kvq_precise over the 8192 resident slots, kvq_final, the early-q projection, the scatter, the trigger splits and the 10 top-k kernels per layer. The resident and staging buffers took about 550 MiB of VRAM; since 2026-11-18 39 MiB (rotating buffers, pinned mirrors), at a cost of 0.2 - 0.35 ms per verify batch for the uploads of the mirrors.
7. Tables: 243 MiB at ctx 230000; 16k chunks would halve them.
8. Overlap of the precise and coarse kernels in the attention op: with int8 tables the op is 12 us slower per layer, see Results.

## Optimisation ideas (not started, nothing measured, except item 2)

Sizes are for ctx 230000 as allocated today, items 7 and 8 for ctx 256000 (the 250k runs of 2026-11-19).

1. 16k chunks. Halves the tables (243 -> about 125 MiB), and the ring of one batch does not grow with the chunk; the fitter stage and graph buffers double (about 17 + 63 MiB more). Measured since (chunk size test, see Results): mean KL ratio to 8k chunks 1.12, median 1.02, so it costs a little quality or nothing; not planned.
2. Done: 8-bit tables instead of fp16 (see Decisions).
3. Partly done (ring redesign): rows that do not fit in the ring go to the host store uncompressed (spill), prefill reads them through the scratch; the fit is done before generation (at the end of the prompt), not during it. Open point: generation needs those rows precise in VRAM until their chunk is fitted (a per-token upload is too expensive), so they need a place there; item 4 is the candidate. With the CPU fitter on 16 threads the backlog was about one chunk, 285 MB of rows.
4. Partly done 2026-11-18 (compute buffers per phase, see Results): the prompt-only buffers are freed during the reply and the decode-only buffers live in the scratch memory; what is not done is a use of the freed memory for a larger resident set or the not yet fitted rows. Original idea: reuse the prefill buffers during generation. The scratch (519 MiB) and the flash attention F16 copies in the compute buffer (most of 1642 MiB) are only used by large ubatches and sit idle during generation. Candidates to live there: the not yet fitted rows of item 3, a larger resident set, or the ring itself, so that this memory is allocated once and used by both phases.
5. From the bottleneck list: chunked flash attention to shrink the F16 copies, a compressed or q8_0 cache for the MTP context, lower fetch traffic for multi-token verify batches.
6. Selection steps on the side stream (idea of 2026-11-17, nothing built or measured). The selection steps that run on the main stream in series with the GDN layers (early-query projection 0.45, the SCORE pass with its prep 2.4, kvq_score 0.37, top-k 0.73 ms per round at 8 tokens and 100k) could run on a side stream beside the GDN kernels; only kvq_precise runs on the side stream today (the coarse passes, prep, merge, combine and kvq_final are on the main stream). Their minimum time per launch would then not add to the round. The ATTN pass, kvq_precise and kvq_final (about 5 ms at 8 tokens) produce the output of the layer and must stay in series. Limits: the selection must end early enough for the worker (readback, plan, gather and upload take about 0.5 ms of the 0.56 ms window of the two GDN layers); beside the first GDN layer the window is about half, so the GPU would idle at the joins again unless the lead is 3 (37% more rows fetched, KL 1.04e-4 against 8.0e-5); kernels that overlap slow each other (kvq_precise 3.28 -> 3.72 ms per round when it began to overlap the shorter coarse pass). A literal fusion of kvq kernels into the GDN kernels (blocks with different jobs in one launch) was considered and not pursued: one launch configuration (registers, shared memory) for both jobs, a fused op or a fusion pass over the graph in ggml, and passes that depend on each other (the top-k passes) cannot share a launch anyway.

7. Scratch as a ring of key blocks (owner's decision 2026-11-19: do it later). The scratch holds the exact q8_0 K and V rows of one attention layer for all positions (540 MiB at 256k) and is refilled from the pinned host store for every attention layer of every prompt ubatch (8.5 GB over the bus per ubatch at the end of a 250k prompt, overlapped on the copy stream with per-layer triggers). Since the attention runs in 16k key blocks it no longer needs a whole layer: a ring of 2 or 3 blocks of 16k rows (35 MiB each, 70 - 105 MiB), the upload of block b + 1 on the copy stream while the attention of block b runs, the same trigger pattern as the resident upload but per block (16 blocks x 16 layers = 256 waits per ubatch instead of 16; a block is 1 ms on the bus against about 2.4 ms of attention at ub 1024, so a ring of 2 should stay ahead). The masked tail (ring rows plus the host remainder, at most about 9k rows) needs its own small buffer, about 20 MiB. The same bytes cross the bus, so no bandwidth change. Gain: the shared scratch region becomes max(about 105, 88) MiB and both phases drop by about 435 MiB; at 250k ub 1024 this is the one item that moves the decode phase, which sets the headroom today. The owner accepts 10 - 20 ms of cudaFree / cudaMalloc per switch between the prompt and decode phases, so the scratch ring and the decode-only buffers (resident, tail, staging) may also be allocated and freed per phase instead of sharing one region, whichever is simpler.
8. Prompt-size policy for follow-up prompts (owner's idea 2026-11-19, to do later, after item 7). Three parts, in this order. (a) Layer input buffer sized by the batch being decoded instead of n_batch: the drafter reads the rows by batch index, so a 1k follow-up needs 100 MiB, not 400; no setting, a few lines in `decode`. (b) Codes and tables stay on the device when the request has fewer than N new tokens to prefill (N a server setting, about 4k): the server knows the count after the cache reuse and passes it as a hint to the kvq memory before the first decode of the request; the device copy is then not freed and the follow-up skips the upload, the alloc and the free (57 ms at 250k, 20 ms at 100k, the accepted +3.3% of today) and the stall of the final 4-token packed batch. Budget at 250k ub 1024 with the codes kept: 16,036 MiB for a 4k follow-up and 15,736 for a 1k one with today's scratch, so N can only be about 1k until item 7 lands (then about 15,300 - 15,600, and 4k is safe); at 100k anything fits. (c) Large prompts (N tokens or more) keep today's path: codes out, the full layer input buffer, and optionally ub 4096 with `--spec-draft-n-ubatch 1024`. Since the attention split, ub 4096 gains only about 2% of prompt time over ub 1024 at 250k (136.1 against 139.1 s; 37.2 against 37.3 s at 100k; before the split 165 against 198 s), so a per-request ubatch switch is a speed knob, not a memory one, and may not be worth a second compute configuration (the ubatch is fixed at context creation today; a switch needs a pp reserve per size and the resize API).

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
- Host memory: 7.6 GB of pinned RAM at ctx 230000, 8.5 GB at 256000, per sequence; plus 27 MiB per context checkpoint, slot file or RAM cache entry for the replay's pending inputs.
- VRAM at 250k: every phase fits 15.5 GiB at ub 1024 (largest sample 15,271 MiB with block 16k, 15,584 with block 0) and at ub 4096 with block 16k and `--spec-draft-n-ubatch 1024` (15,372); ub 4096 with block 0 does not (17,974). The follow-up prompt pays about 55 ms for the upload of the codes (see Results, combined build).
- The attention split changes the rounding of the prompt (other ids than the single attention); its quality is checked on few traces only (see Results, attention split).
- Deferred tuning: top-up rules and thresholds, prefetch size. Lead 3 was measured with the whole union (no gain, worse KL); the resident capacity follows the largest verify batch.
- No automated tests for the store, the fitter, the ring cache or the server flows; only the codec and the kernel have tests. The session flows were checked by hand once.
- No long session was run (hours, many rollbacks and refits).
- The code was written by agents and reviewed only at the level of reports, diffs of the scheduler change and test results. It needs a real review before anyone depends on it.
- `--fit` and the memory estimate do not know about the kvq memory. What to do with the freed VRAM (longer context, larger model quantization) is not decided.
- The branch sits on ggml-org/master 951d465f5 and changes ggml (new op, scheduler callback order, CUDA kernel, `ggml_gallocr_resize`); rebases will conflict there. It also contains the merge of ggml-org PR 31669 (GDN replay) while that PR is open and marked WIP: its stored layout of the pending inputs may change upstream, and the final PR will have to replace the merged version.

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

State save and load of the K/V outside the fit mode: an error, the save or load returns 0. A plain state (saved without `--kvq`) into a kvq context and a kvq state into a plain context: an error, the sequence is empty afterwards; there is no migration from plain snapshots. The `ON_DEVICE` state flag on a kvq state: an error.

## Working material outside the repository

`../kvq-scratch`: build scripts (`build.bat`, `reconfigure.bat`, build dir `build-cuda`), token files of four traces (`tok/`, prefix lengths 11705, 24183, 49737, 99395; `tok/cat281k.bin` = four traces one after the other, 281,551 tokens, for 250k prompts), reference logits (`ref/`), server session scripts (`srv/`), logs; `mem/` holds the memory work of 2026-11-17 to 2026-11-19 (`inventory`, `vram`, `phase`, `replay`, `m250`, `comb250`, `codes`, `attnsplit`, `final`, each with a `RESULTS.md`; `attnsplit` also has `QUALITY.md`, `DIAG.md`, the blocked-teacher logits under `ref/`, `scripts/exact.py` and the diagnostic diffs; `comb250/UBATCH.md` is the ubatch sweep). The token files come from private data: never print or log prompt text, generated text or decoded tokens.
