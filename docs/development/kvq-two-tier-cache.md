# Two-tier KV cache for qwen35 (kvq) - prototype state

State of branch `kvq-two-tier-cache` at commit cefce74bc (2026-11-13). The branch starts from ggml-org/master 951d465f5 and is pushed to the JetBrains-Hardware fork. This file is the starting point for the next work session.

Research background, codec definition and the original plan are in the `qwen3.8-exp` repository: `notes/kvq_implementation_plan.md`, `notes/kv_prefetch_results.md`, `src/qexp/kvcodec.py`, `experiments/14_kv_prefetch.py`.

## What it is

A KV cache for Qwen3.8-27B (GGUF arch `qwen35`) that keeps most of the attention K/V out of VRAM:

- Coarse tier, VRAM: every row of a fitted chunk as transform-coded low-bit codes (K 2 bit, V 1 bit plus a gain). The attention is computed on the packed codes by a custom op.
- Precise tier, system RAM: the q8_0 rows of the same positions in a pinned host store.
- Resident set, VRAM: 1024 precise rows per (layer, kv head), selected per token by an early query and fetched from RAM while the GPU runs the two layers before the attention.
- Precise tail, VRAM: q8_0 rows of positions that are not in a fitted chunk yet (ring-shaped device cache).

Model facts used everywhere: 64 trunk blocks, 16 attention layers (il = 3, 7, ..., 63), 24 query heads, 4 kv heads, head dim 256. The MTP block (block 64) runs in its own context with its own plain cache and does not use any of this.

## How it works

Chunks. The sequence is cut in chunks of 8192 positions. When a chunk is complete (plus a margin of 64 positions), background CPU threads fit a K table and a V table per (layer, kv head) on the chunk's own rows, encode the rows, and the decode thread installs codes and tables between two decodes. Then the chunk's rows leave the device cache.

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
| Host store, resident cache, prefetch worker, scratch uploads | `src/llama-kvq-store.{h,cpp}` |
| Background fitter | `src/llama-kvq-fit.{h,cpp}` |
| Memory (hybrid subclass), ring-shaped device cache | `src/llama-memory-hybrid-kvq.{h,cpp}`, `src/llama-kv-cache.{h,cpp}` |
| Graph | `src/models/qwen35.cpp` |
| Scheduler: copy callback before the wait | `ggml_backend_sched_set_copy_callback_early` in `ggml/src/ggml-backend.cpp` |
| Internal API | `src/llama-ext.h` (`llama_kvq_params`, `llama_kvq_fit_params`, `llama_memory_kvq_*`, `llama_model_set_kvq_*`) |
| Server options and forced settings | `common/arg.cpp`, `common/common.cpp`, `tools/server/server.cpp`, `tools/server/server-context.cpp` |
| Evaluation tool | `tools/kvq-eval` |
| Tests | `tests/test-kvq-codec.cpp`, `tests/test-kvq-attn.cpp`, `KVQ_ATTN` cases in `tests/test-backend-ops.cpp` |

## How to run

Server, production configuration plus `--kvq`:

```
llama-server --fit off -m Qwen3.8-3.6-27B-blend-Q4_K_M.gguf --ctx-size 230000 --n-gpu-layers all --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16 --batch-size 4096 --ubatch-size 1024 --kvq
```

Options: `--kvq`, `--kvq-threads N` (fitter threads, default physical cores), `--kvq-prefetch-n` (1024), `--kvq-resident-cap` (1024), `--kvq-lead` (2), `--kvq-tail N` (ring cells, default 2 * 8192 + n_batch + 256, 0 = full device cache), `--kvq-v-tables FNAME` (offline V tables, only for comparison).

With `--kvq` the server forces, with a warning: 1 slot, RAM prompt cache off, slot save path cleared, context shift and cache reuse off, flash attention on, q8_0 K/V, draft n_max <= 7. It refuses to start for a model that is not qwen35, `--mmproj`, embeddings. Attention rotation of quantized caches is turned off for the model (also for the MTP context). `--fit` does not count the kvq memory.

Evaluation tool (teacher-forced KL against saved reference logits):

```
llama-kvq-eval -m MODEL -ngl 99 -fa on -fit off -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 --kvq-tokens TOKENS.bin --kvq-prefix-len S --kvq-chunk 1 --kvq-ref REF --kvq-lib
```

`--kvq-lib` is the library path (same as the server), `--kvq-lib-full` keeps the full device cache, `--kvq-rm-test` checks the removal cases, `--kvq-scr-check N` compares scratch and cache prefill logits bit for bit. The older `--kvq-variant`, `--kvq-two-tier`, `--kvq-packed` modes are tool-driven experiments from earlier stages.

Environment variables for debugging: `LLAMA_KVQ_SCHED_SYNC`, `LLAMA_KVQ_GATHER_THREADS`, `LLAMA_KVQ_IDLE_PROBE`, `LLAMA_KVQ_SCRATCH`, `LLAMA_SERVER_GEN_TIMES`.

## Results

Machine: RTX 5090 32 GB, Ryzen 9950X3D, Windows 11, CUDA 12.9. One trace of 100k tokens unless noted; treat single runs as such.

Server, ctx 230000, 100k prompt, MTP with 3 drafts:

| | Two-tier | Production (q8_0, flash attention) |
|---|---|---|
| Total VRAM | 24.06 GB | 28.4 GB |
| Prefill | 47.9 s | 43.4 - 43.8 s |
| Generation, first request (fitter busy) | 109 - 112 t/s | 120.5 t/s |
| Generation, follow-up request | 122.2 t/s | 124.3 t/s |

Generated token ids matched production on every request in one session (265, 1500, 300, 55 tokens); in another session the first request diverged at token 211 of 265. The packed decode is not deterministic from run to run, because the coarse region depends on fitter timing.

Quality, KL against the q8_0 reference, 471 scored tokens, 1 token per decode:

| Cache | KL | ms/token |
|---|---|---|
| Two-tier, library path | 1.32e-4 | 16.86 |
| q8_0 with rotation on (two q8_0 configurations against each other) | 1.5e-4 | 17.70 |
| Two-tier with offline V tables from another conversation | 4.4e-4 | 16.88 |
| Plain q4_0, rotation on | 1.17e-3 | 24.15 |
| Plain q4_0, rotation off | 2.69e-3 | 24.22 |

VRAM as allocated at ctx 230000 (MiB): coarse rows 1400, tables 459, tail ring 689, scratch 519, resident and other 52; together about 3100 (about 2.2 GiB at 150k). Main compute buffer 1642. MTP context: f16 cache 899 (production uses f16 there too), compute 586. Host: 7.6 GB pinned store.

Per format at 150k, main model only (arithmetic): f16 9.16 GiB, q8_0 4.86 GiB, q4_0 2.57 GiB.

## Decisions taken, with the reason

- K 2 bit, V 1 bit: K2V1 is as good as K2V2 in the two-tier setting.
- Tables per 8k chunk, fitted on the chunk's own rows and queries: no quality cost against one fit over the whole prefix; chunk size 4k to 16k makes no difference. K tables from another conversation do not work.
- V fitted per chunk too: offline V tables cost 3.5x KL at 100k.
- Early query with lead 2: twice the fetch time of lead 1, KL difference inside the noise.
- Synchronous top-up off: no KL gain measured, it costs a scheduler split. `count` and `always` stay as settings.
- Resident capacity 1024: 4096 halves the KL but costs 0.6 - 0.8 ms per token in the attention kernel.
- Prefill uses exact q8_0 rows, not the two-tier attention: two-tier prefill was never measured for quality.
- Copies between GPU graph launches go on a separate copy stream: a memcpy on the compute stream stalled the GPU 55 - 65 us each, about 47 times per token.

## Open bottlenecks, largest first

1. Generation while the fitter works (after a long prefill or a restore): 109 - 112 t/s against 120 - 125. Steady state is about 98% of production.
2. Prefill waits for the fitter: about 3.4 s of the extra 4 s at 100k. A chunk takes 3.4 - 3.8 s on 16 threads (about 56 CPU-seconds), prefill produces one about every 3.6 s.
3. Main compute buffer 1.64 GB: mostly the F16 copies of K and V that flash attention makes at full length.
4. MTP context cache: 899 MiB f16 for the full context.
5. No speed gain from the packed path at 100k: a 4-token verify fetches 43k - 48k rows (65 - 73% of the request misses the resident set), GPU idle at the joins 1.7 - 2.2 ms per round.
6. Tables: 16k chunks would halve them.

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
