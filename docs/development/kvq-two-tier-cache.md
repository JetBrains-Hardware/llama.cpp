# Two-tier KV cache for qwen35 (kvq) - prototype state

State of branch `kvq-two-tier-cache` at commit cefce74bc (2026-11-13), with the GPU fitter (2026-11-16) on top of 7604e664e. The CPU fitter is kept on branch `kvq-fit-cpu`. The branch starts from ggml-org/master 951d465f5 and is pushed to the JetBrains-Hardware fork. This file is the starting point for the next work session.

Research background, codec definition and the original plan are in the `qwen3.8-exp` repository: `notes/kvq_implementation_plan.md`, `notes/kv_prefetch_results.md`, `src/qexp/kvcodec.py`, `experiments/14_kv_prefetch.py`.

## What it is

A KV cache for Qwen3.8-27B (GGUF arch `qwen35`) that keeps most of the attention K/V out of VRAM:

- Coarse tier, VRAM: every row of a fitted chunk as transform-coded low-bit codes (K 2 bit, V 1 bit plus a gain). The attention is computed on the packed codes by a custom op.
- Precise tier, system RAM: the q8_0 rows of the same positions in a pinned host store.
- Resident set, VRAM: 1024 precise rows per (layer, kv head), selected per token by an early query and fetched from RAM while the GPU runs the two layers before the attention.
- Precise tail, VRAM: q8_0 rows of positions that are not in a fitted chunk yet (ring-shaped device cache).

Model facts used everywhere: 64 trunk blocks, 16 attention layers (il = 3, 7, ..., 63), 24 query heads, 4 kv heads, head dim 256. The MTP block (block 64) runs in its own context with its own plain cache and does not use any of this.

## How it works

Chunks. The sequence is cut in chunks of 8192 positions. When a chunk is complete (plus a margin of 64 positions), the fitter fits a K table and a V table per (layer, kv head) on the chunk's own rows, codes the rows, and the chunk joins the coarse region. Then the chunk's rows leave the device cache.

Tables. Per (chunk, layer, kv head, K or V) a 256 x 256 inverse table and a small aux array (step, mu, bits). The inverse table is stored as int8 with one scale per component c (row inv[c][:] divided by its max abs, times 127); the scale is folded into the step that the attention op reads, so the decode formula is unchanged. The codec itself (fit, encode, bits, the step of the device CODE mode) still works in float.

Fitter (`llama_kvq_fitter`). The work on the rows runs on the GPU, on the main stream between two ubatches; only the small 256 x 256 problem per (layer, kv head, K or V) runs on CPU threads. A job is a small state machine over several polls of the decode thread (before each ubatch), one graph per attention layer and phase:

1. Phase 1, when the job starts: the chunk's q8_0 rows are read from the ring (`get_rows`, wrap-around by index), copied to the pinned host store (device relayout, one download per (layer, kv head)), and their mean and covariance (`ggml_kvq_fit`) go to the host, 33 MiB per chunk.
2. CPU tasks (`--kvq-threads`, low priority), the V tasks first: metric (from `qacc` / `gacc`), `llama_kvq_fit_cov` (Cholesky, `L^T cov L`, eigen solver, fwd/inv tables), for K also bits, steps and the packed table. The tasks wait for the event of phase 1, not the decode thread.
3. Phase 2, at the first poll after the V tasks (the K tasks may still run): V fwd tables up, rows rotated, the scale of the shape (gain per row, column statistics) goes to the host. CPU tasks for V: bits, steps, packed table.
4. Phase 3 and install, at the first poll after all tasks: fwd tables up, rows rotated, K and V codes packed straight into `kc` and `vc`, the tables uploaded, `set_n_coarse` (one synchronize per chunk, as before).

The decode thread never waits in a poll for device work it queued in the same poll. The packed codes never go through the host. The eigen step is the function `llama_kvq_fit_cov`, a device eigen solver could replace it. The o_proj grams of the V metric are computed once on the GPU when the fit mode starts.

- K metric: second moment of the queries at the chunk's own positions, summed on the device during decoding (`qacc`), read back once per chunk.
- V metric: o_proj columns scaled by the chunk's mean sigmoid(gate), summed on the device (`gacc`).
- Codec: Cholesky of the metric, eigen-decomposition, reverse water-filling bit allocation, uniform quantizer with Gaussian steps. Exact definition: `src/llama-kvq-codec.cpp` and the research code.

Generation and MTP verify (ubatch <= 8 tokens), per attention layer il:

1. At the input of layer il - 2 the graph computes layer il's query from the earlier hidden state, scores all coarse rows, and takes the top 1024 per (kv head, position).
2. A scheduler split with a copy callback queues the readback of the selection on a copy stream. A worker thread plans the resident slots (least recently selected eviction) and gathers the rows from the host store. The GPU keeps running layers il - 2 and il - 1.
3. Before layer il's attention a join uploads the rows and scatters them into the resident tensor.
4. `ggml_kvq_attn` computes the attention over coarse rows + resident precise rows + the tail.

Prefill (larger ubatches), per attention layer: the installed region is uploaded from the pinned host store into one device scratch buffer (4 async uploads on the copy stream, overlapping the layers before), the tail rows are copied in on the device, and normal flash attention runs over the scratch. Logits are bit-identical to attention over a full device cache.

Rollback. Removing the end of the sequence is cheap when it stays in the tail (MTP draft rollback). When it cuts into an installed chunk, the chunk is dropped, its rows before the cut are restored from the host store into the ring, and it is fitted again later. `clear` drops everything and the fitter starts again.

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

Options: `--kvq`, `--kvq-threads N` (CPU threads of the table fit, default 4, at most the physical cores), `--kvq-prefetch-n` (1024), `--kvq-resident-cap` (1024), `--kvq-lead` (2), `--kvq-tail N` (ring cells, default 2 * 8192 + n_batch + 256, 0 = full device cache), `--kvq-v-tables FNAME` (offline V tables, only for comparison).

With `--kvq` the server forces, with a warning: 1 slot, RAM prompt cache off, slot save path cleared, context shift and cache reuse off, flash attention on, q8_0 K/V, draft n_max <= 7. It refuses to start for a model that is not qwen35, `--mmproj`, embeddings. Attention rotation of quantized caches is turned off for the model (also for the MTP context). `--fit` does not count the kvq memory.

Evaluation tool (teacher-forced KL against saved reference logits):

```
llama-kvq-eval -m MODEL -ngl 99 -fa on -fit off -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --kvq-tokens TOKENS.bin --kvq-prefix-len S --kvq-chunk 1 --kvq-ref REF --kvq-lib
```

`--kvq-lib` is the library path (same as the server), `--kvq-lib-full` keeps the full device cache, `--kvq-rm-test` checks the removal cases, `--kvq-scr-check N` compares scratch and cache prefill logits bit for bit. The older `--kvq-variant`, `--kvq-two-tier`, `--kvq-packed` modes are tool-driven experiments from earlier stages.

Environment variables for debugging: `LLAMA_KVQ_SCHED_SYNC`, `LLAMA_KVQ_GATHER_THREADS`, `LLAMA_KVQ_IDLE_PROBE`, `LLAMA_KVQ_SCRATCH`, `LLAMA_SERVER_GEN_TIMES`, `LLAMA_KVQ_FIT_SYNC` (synchronize around the device phases of the fitter, logs the device time per chunk and phase).

## Results

Machine: RTX 5090 32 GB, Ryzen 9950X3D, Windows 11, CUDA 12.9. One trace of 100k tokens unless noted; treat single runs as such.

Server, ctx 230000, 100k prompt, MTP with 3 drafts:

| | Two-tier, int8 tables, 4 CPU threads | Two-tier, GPU fitter, fp16 tables, 4 / 1 CPU threads | Two-tier, CPU fitter (16 threads, before the codec speed fix) | Production (q8_0, flash attention) |
|---|---|---|---|---|
| Total VRAM (nvidia-smi after the request) | 22.79 GB | 23.02 GB | 24.06 GB | 28.4 GB |
| Prefill | 44.1 - 44.2 s | 44.0 - 44.2 s / 44.2 s | 47.9 s | 43.4 - 43.8 s |
| Generation, first request | 125.9 - 129.6 t/s | 128 - 129 / 122 - 124 t/s | 109 - 112 t/s | 120.5 t/s |
| Generation, follow-up request | 123.8 - 126.8 t/s | 125.5 - 127.5 / 126.2 t/s | 122.2 t/s | 124.3 t/s |

int8 tables: two sessions (2026-11-16), token ids matched production on every request of both. GPU fitter: two sessions per thread count (2026-11-16); prefill never waited for the fitter (no batch waited for the ring, also with 1 thread). The production numbers are from another day and were not run again, so the gain over production in generation is not established. Generated token ids matched production on every request in three of the four sessions (265, 1500, 300 tokens); in one session the first request diverged at token 211 of 265. The packed decode is not deterministic from run to run, because the coarse region depends on fitter timing.

GPU fitter cost per 8k chunk at 100k (kvq-eval, -ub 2048; device time with `LLAMA_KVQ_FIT_SYNC`, the GPU otherwise idle):

- Device: 35 ms (phase 1 20 ms with the 285 MB download of the q8_0 rows to the host store, phase 2 9 - 10 ms, phase 3 6 ms).
- CPU: 2.0 s on one core (2.1 s on 4 threads), the eigen solver about half of it. A job takes 2.1 - 2.6 s with 1 thread and 0.6 - 1.0 s with 4 threads in the server (statistics to join 2.2 - 3.2 s / 0.9 - 1.1 s); prefill makes a chunk about every 3.6 s.
- Decode thread: 60 - 85 ms in the snapshot and phase 1 (mostly the existing synchronize of the statistics readback), 25 - 50 ms in the later phases and the join (the synchronize of `set_n_coarse`).
- PCIe: 285 MB q8_0 rows, 34 MB moments device to host; 51 MB parameters (fwd tables), 9 MB packed tables host to device. The codes do not cross.
- Memory: 83 MiB device (fitter tensors 20 MiB, graph buffer 63 MiB), about 85 MB pinned host per job in flight (the CPU fitter used 285 MB per job).
- The o_proj grams of the V metric: 0.08 s on the GPU once, at the start (about 4 CPU-seconds on the first chunk before).

Against the CPU fitter on the same chunk (chunk 5 of the 100k trace, all 128 tasks): q8_0 rows and K metric identical, V metric max relative difference 2e-6 (o_proj grams on the GPU), means identical, covariance max relative difference 4.1e-6, bits identical, steps 3e-5, 300 of 32768 eigenvectors with a flipped sign, sign-aligned inverse tables max difference 0.4% of the largest entry, codes 0.34% different, 3.0e-5 after the sign alignment, V gains 139 of 524288 differ by one f16 step.

Quality, KL against the q8_0 reference, 471 scored tokens, 1 token per decode:

| Cache | KL | ms/token |
|---|---|---|
| Two-tier, library path, CPU fitter | 1.32e-4 | 16.86 |
| Two-tier, library path, GPU fitter (5 runs: 4 or 8 threads, 1 thread) | 1.25e-4 - 1.40e-4 | 16.69 - 16.73 |
| Two-tier, library path, GPU fitter, int8 tables (3 runs) | 1.24e-4 - 1.41e-4 | 16.87 - 16.97 |
| q8_0 with rotation on (two q8_0 configurations against each other) | 1.5e-4 | 17.70 |
| Two-tier with offline V tables from another conversation | 4.4e-4 | 16.88 |
| Plain q4_0, rotation on | 1.17e-3 | 24.15 |
| Plain q4_0, rotation off | 2.69e-3 | 24.22 |

int8 tables cost about 0.15 ms per token in kvq-eval (fp16 build 16.71 ms in the same hour). nsys: the kernels that read the tables got faster (prep 12.2 -> 11.0 us, combine 7.8 -> 6.0 us), but the attention op of a layer takes 111 instead of 99 us: the resident precise rows (side stream) now take the SMs first and delay the coarse kernel, which is on the critical path (coarse 66 -> 81 us, precise 59 -> 25 us). Forking the side stream after prep made it worse (17.33 ms). The overlap of the two streams is fragile; a fix is open.

VRAM as allocated at ctx 230000 (MiB): coarse rows 1400, tables 235 (459 with fp16 tables), tail ring 689, scratch 519, resident and other 52; together about 2880 (about 2.0 GiB at 150k). Main compute buffer 1642. MTP context: f16 cache 899 (production uses f16 there too), compute 586. Host: 7.6 GB pinned store.

Per format at 150k, main model only (arithmetic): f16 9.16 GiB, q8_0 4.86 GiB, q4_0 2.57 GiB.

## Decisions taken, with the reason

- K 2 bit, V 1 bit: K2V1 is as good as K2V2 in the two-tier setting.
- Tables per 8k chunk, fitted on the chunk's own rows and queries: no quality cost against one fit over the whole prefix; chunk size 4k to 16k makes no difference. K tables from another conversation do not work.
- V fitted per chunk too: offline V tables cost 3.5x KL at 100k.
- Early query with lead 2: twice the fetch time of lead 1, KL difference inside the noise.
- Synchronous top-up off: no KL gain measured, it costs a scheduler split. `count` and `always` stay as settings.
- Resident capacity 1024: 4096 halves the KL but costs 0.6 - 0.8 ms per token in the attention kernel.
- Prefill uses exact q8_0 rows, not the two-tier attention: two-tier prefill was never measured for quality.
- Inverse tables as int8 with one scale per component (2026-11-16): halves the tables (459 -> 235 MiB at ctx 230000). Offline on one chunk the codec error in the metric grows 0.8% mean, 3.2% worst for K and 0.0% for V; KL inside the run-to-run noise (see Results). One scale per whole table costs 5x the error on K, 6 bits costs 10 - 50% on K.
- Copies between GPU graph launches go on a separate copy stream: a memcpy on the compute stream stalled the GPU 55 - 65 us each, about 47 times per token.

## Open bottlenecks, largest first

1. Prefill: 44.0 - 44.2 s against 43.4 - 43.8 s at 100k; the fitter no longer stalls it. What is left per chunk: 35 ms of device work on the main stream, two synchronizes (statistics readback, join), the D2H of the q8_0 rows on the main stream. A second stream (or the copy stream for the D2H) could hide most of it.
2. The latency of a job is set by the polls (one per ubatch): after the CPU tasks, a job needs one or two more ubatches. With 1 thread and -ub 2048 (kvq-eval) 3 batches still waited 0.35 s at 100k.
3. Main compute buffer 1.64 GB: mostly the F16 copies of K and V that flash attention makes at full length.
4. MTP context cache: 899 MiB f16 for the full context.
5. No speed gain from the packed path at 100k: a 4-token verify fetches 43k - 48k rows (65 - 73% of the request misses the resident set), GPU idle at the joins 1.7 - 2.2 ms per round.
6. Tables: 235 MiB at ctx 230000; 16k chunks would halve them.
7. Overlap of the precise and coarse kernels in the attention op: with int8 tables the op is 12 us slower per layer, see Results.

## Optimisation ideas (not started, nothing measured, except item 2)

Sizes are for ctx 230000 as allocated today.

1. 16k chunks. Halves the tables (235 -> about 120 MiB); chunk size 4k to 16k made no quality difference. Catch: the ring is sized 2 chunks + n_batch, so with the current rule it grows from 689 to about 1230 MiB and the net result is worse. It only pays off together with a ring that does not scale with the chunk (item 3), or by fitting 8k chunks first and merging two of them into one 16k table in the background later.
2. Done: 8-bit tables instead of fp16 (see Decisions).
3. Do not wait for the fitter. With the GPU fitter prefill no longer waits at 100k on this machine, so this matters only for a slower GPU or CPU, or a ring smaller than 2 chunks. Idea: rows that do not fit in the ring go to the host store uncompressed, prefill reads them through the scratch like the installed region, and the fitter replaces the full version by the compressed one during generation. Open point: generation needs those rows precise in VRAM until their chunk is fitted (a per-token upload is too expensive), so they need a place there; item 4 is the candidate. With the CPU fitter on 16 threads the backlog was about one chunk, 285 MB of rows.
4. Reuse the prefill buffers during generation. The scratch (519 MiB) and the flash attention F16 copies in the compute buffer (most of 1642 MiB) are only used by large ubatches and sit idle during generation. Candidates to live there: the not yet fitted rows of item 3, a larger resident set, or the ring itself, so that this memory is allocated once and used by both phases.
5. From the bottleneck list: chunked flash attention to shrink the F16 copies, a compressed or q8_0 cache for the MTP context, lower fetch traffic for multi-token verify batches.

## Gaps before this can replace production

- Quality is measured on few traces: the final library path on one 100k trace, earlier stages on four (12k to 100k). Nothing was run at 150k or above, which is the target length. A wider check over more traces and lengths is needed.
- Speed of the final build was measured only at 100k. Not measured at 24k / 50k and at 150k+.
- Slot save and the RAM prompt cache do not work (full state save aborts). The production launch passes `--slot-save-path`, so saving and restoring a session is lost with `--kvq`.
- Output is not reproducible from run to run (the coarse region depends on fitter timing). A deterministic install rule would be needed for exact comparisons.
- CUDA only. The attention op and the fit op have a CPU reference and a CUDA kernel, nothing for Metal or Vulkan; the fit mode refuses to start when the device cannot run the fit graphs. The fitter needs about 2 CPU-seconds per 8k chunk (one core is enough at 100k on this GPU, see Results).
- The fitter's device work runs on the main stream between ubatches. A second stream would hide its ~35 ms per chunk, not built.
- Host memory: 7.6 GB of pinned RAM at ctx 230000, per sequence.
- Deferred tuning: resident capacity (1024 vs 4096), top-up rules and thresholds, lead, prefetch size.
- No automated tests for the store, the fitter, the ring cache or the server flows; only the codec and the kernel have tests. The session flows were checked by hand once.
- No long session was run (hours, many rollbacks and refits).
- The code was written by agents and reviewed only at the level of reports, diffs of the scheduler change and test results. It needs a real review before anyone depends on it.
- `--fit` and the memory estimate do not know about the kvq memory. What to do with the freed VRAM (longer context, larger model quantization) is not decided.
- The branch sits on ggml-org/master 951d465f5 and changes ggml (new op, scheduler callback order, CUDA kernel); rebases will conflict there.

## Not exercised

- Two-tier without MTP, drafts of 7 tokens on the packed path, model reload after sleep, a chat-endpoint conversation with user-message checkpoints.
- Top-up modes `count` / `always`, lead 1 and 3 after the async prefetch change.
- The server with `--kvq-tail 0` or other tail sizes, the hard error when the ring is too small.
- The tool-driven `--kvq-packed` modes of kvq-eval after the tail-mode change.
- The offline V path after the K/V task split.
- MoE models with the early copy callback.

## Not supported (aborts with a message)

Full state save and load (RAM prompt cache, slot save files), more than one sequence, `seq_cp`, `seq_add` / `seq_div` (context shift, cache reuse), removal of a middle range. Recurrent-only state (context checkpoints) works.

## Working material outside the repository

`../kvq-scratch`: build scripts (`build.bat`, `reconfigure.bat`, build dir `build-cuda`), token files of four traces (`tok/`, prefix lengths 11705, 24183, 49737, 99395), reference logits (`ref/`), server session scripts (`srv/`), logs. The token files come from private data: never print or log prompt text, generated text or decoded tokens.
