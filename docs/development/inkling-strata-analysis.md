# Inkling-2-Mini: Strata-style O(1) attention compute buffer (analysis)

Branch `erokhins/inkling-strata` (worktree `wt-inkling-strata`), based on `erokhins/inkling` @ `f929fd85c`
(which already carries hridaya423 `68cd7533a`, fused relative-position bias in FlashAttention).

Goal of this branch: re-create the Strata optimization of the attention compute buffer in llama.cpp, i.e. make the
compute buffer independent of context length (`O(1)` in `n_ctx`) and keep results exact.

Sources of this document: Junie session `session-261123-154730-118q` (`~/.junie/sessions/`, tasks 1-14 of that session),
the Strata checkout `..\Strata` (commit `6a6db7f`), and GitHub PR pages checked on 2026-11-23. Numbers marked (J) come
from the Junie session and were not re-measured here. Anything marked TODO is unverified.

## 1. History (what was asked before, what was found)

Junie session `session-261123-154730-118q`, project dir `wt-inkling`, in order:

1. Why does Inkling use so much VRAM? The hypothesis in `docs/development/inkling-2-mini.md` (SWA layers get full-context KV)
   is wrong: SWA layers are capped at a 1,536-cell window (~240 MiB). ~75% of the per-token cost is the compute buffer.
2. Why does the compute buffer grow with context? Inkling feeds a relative-position bias into attention. `llama.cpp`
   materialized it as a dense `[n_kv, n_tokens, n_head]` F32 tensor (`kq_b`, built with `ggml_get_rows(table, rel_pos)` +
   `ggml_cont(ggml_permute(...))`), plus an F16 mask copy, so FlashAttention was fed a per-head mask.
3. What is the ideal buffer? `O(1)` in `n_ctx`, about 0.3-0.4 GiB at `-ub 512` (J).
4. How did Strata fix it? See section 3.
5. Is the Strata approach exact? Yes (section 4).
6. Is there a fix in llama.cpp forks/PRs? Yes, several, none merged upstream (section 5).
7. "Any of the patches the ultimate Strata fix?" No: all keep the standard 1-head F16 `attn_mask` (section 6).
8. Took hridaya423 `68cd7533a`, rebuilt, measured on the two target requests, committed on `erokhins/inkling`
   (`f929fd85c`). Result (from `inkling-2-mini.md`, RTX 5090, `-c 98304 -np 1 -fa on`):

   | Request | Mode | VRAM | Prefill |
   |---|---|---|---|
   | flow 144 (32k) | original | 26.5-26.7 GiB | 1,400-1,432 t/s |
   | flow 144 (32k) | + hridaya423 | 17.0 GiB | 2,703-2,739 t/s |
   | flow 197 (76k) | original | 26.4 GiB | 916-917 t/s |
   | flow 197 (76k) | + hridaya423 | 17.0-17.1 GiB | 2,401-2,492 t/s |

9. Generated token counts differed after the fix; checked with teacher forcing (`tools/teacher_force.py`): no
   correctness bug, the new path is at least as close to the `-fa off` F32 reference as the old one.
   The difference comes from FP32 bias accumulation instead of an F16 mask.
10. (Running at the time of writing, separate task) KV cache offloading check on this model, see `erokhins/kvq`.

Key point for this branch: after hridaya423, VRAM is already flat across 32k and 76k (17.0 GiB both). The remaining
gap to Strata is small in absolute terms (section 6). TODO: measure the real compute buffer split before choosing work.

## 2. Where the memory goes

Inkling-2-Mini, `UD-Q3_K_XL`: 24 layers (loop core 6..18 run 3 times), `n_head = 24`, head dim 64 (Strata `kInkHeadDim`),
`rel_extent = 1024` (distances clamped to 1024), `n_rel = 16`.

Relative bias: `bias[q, k, h] = table[min(max(qpos[q] - kpos[k], 0), 1024), h, q]`, where
`table[d, h, q] = sum_c r[c, h, q] * rel_proj[c, d]` (F32, `[1025, 24, n_ubatch]` = 48 MiB at `-ub 512`).
The per-query attention sink `t` is folded into the table (`table - sink`) with a zero sink in our graph.

Per-context-token compute scaling at `-ub 512` (J):

| Version | Scaling | 131k ctx | 1M ctx |
|---|---:|---:|---:|
| ggml-org master / our `8f2537201` | ~99 KiB/tok | 12.4 GiB | 96.7 GiB |
| + fused bias in FA, shared F16 mask (hridaya423, our current HEAD) | ~1 KiB/tok (mask) + 4 B/tok (`kpos`) | ~0.13 GiB scaling part | ~1 GiB scaling part |
| Strata-style, mask computed in-kernel | 0 B/tok | 0 | 0 |

Fixed part (independent of ctx): table 48 MiB + activations ~250-350 MiB (J, not re-measured).

## 3. How Strata does it (`..\Strata\engine\src\inkling2\cuda\`, repo `Niko1221/Strata`, MIT, commit `6a6db7f`)

Strata is a standalone engine (own CUDA kernels in `engine/src/inkling2/cuda/`), not a llama.cpp patch. Files:
`ink_attn.cuh` (kernels), `ink_runner.cuh` (orchestration), `ink_cuda.cuh` (constants, `kInkRelExtent = 1024` at line 144).

Step A: small bias table only. `k_ink_rel_table` (`ink_attn.cuh:316-337`) computes `d_rel_table[M, 24, 1025]` in F32
(column 1024 is `0.0f`), allocated in `ink_runner.cuh:2086` (`MB * kInkHeads * (kInkRelExtent + 1)`). This is the same
table llama.cpp already builds in `src/models/inkling2.cpp:354`.

Step B: look the bias up inside attention, never materialize `[n_kv, M, n_head]`.

- Decode / batched decode / tree verify: `k_ink_flash_gqa` (`ink_attn.cuh:352`), `k_ink_flash_gqa_splitk` (`:479`, split KV +
  `k_ink_flash_gqa_merge` `:604`), `k_ink_flash_gqa_gqa3` (`:642`, 3 q-heads per kv head). In the KV loop:
  `dist = min(max(qp - k_pos, 0), kInkRelExtent); sc = dot + rt[dist];` (`:397`, `:539`, `:688`). Zero `O(n_ctx)` buffer.
- Long prefill (Tensor Core path, `ink_runner.cuh:3029-3113`, taken when `mode == 0 && M >= 128 && cc >= 70`):
  tiles queries (`Q_CH = 256`) and KV (`kInkPfKvTile = 2048`, `ink_runner.cuh:1672`) with online softmax:
  cuBLAS `Q @ K^T` for one `[24, 256, 2048]` tile (F32 scores, 48 MiB) -> `k_ink_pf_softmax_tile_online`
  (`ink_attn.cuh:995`; adds `rt[min(dist, 1024)]`, applies causal and SWA masks by index math, updates `m`, `l`, rescale
  factors, writes BF16/FP16 `P`, 24 MiB) -> cuBLAS `P @ V` -> `k_ink_pf_accum_tile` (`:1067`) rescales and accumulates.
  Workspace is `ensure_pf_attn` (`ink_runner.cuh:2810-2842`): fixed ~114 MiB for any context (comment in source:
  "O(1)-VRAM KV tile"). If VRAM is tight it falls back to the zero-extra-VRAM `k_ink_flash_gqa_gqa3` kernel.
- Sink: folded into the online-softmax init (`m = sink, l = 1.0f`, around `ink_attn.cuh:390` and `:988`), equivalent to
  one extra key with logit `sink` and value 0.
- Masks (causal, SWA window) are index arithmetic in the kernels. No mask tensor exists. Strata targets a single
  sequence (plus tree verify), so it does not need per-sequence mask logic.

Take-away: the part of Strata that matters for llama.cpp is only "bias by index math + mask by index math inside the FA
kernel". Its tiled prefill via cuBLAS is not needed: ggml FlashAttention kernels are already tiled.

## 4. Is it exact?

Mathematically exact (Junie task 12, checked in `ink_attn.cuh`):
- Bias is added in F32 to the F32 score. Upstream llama.cpp (before the fused patch) cast `kq_b + attn_mask` to F16
  (`src/llama-graph.cpp`), so the fused path is slightly more precise, not less.
- Sink folding and tiled online softmax are the standard exact recurrences. Differences are only FP associativity,
  `__expf`, and the cast of `P` to BF16/FP16 before `P @ V` (same as any FlashAttention).
Our measured check (`inkling-2-mini.md`, teacher forcing): 2.7k prompt 95/100 top-1 vs `-fa off`, median
`|d_logprob|` 0.031; prefill sweep 10/10 top-1; CPU vs CUDA 20/20.

## 5. Existing llama.cpp work (no one is merged upstream as of 2026-11-23)

| Candidate | Approach | Backends | Status / numbers |
|---|---|---|---|
| `hridaya423/llama.cpp` branch `inkling-fast`, commit `68cd7533a` | `ggml_flash_attn_ext_add_rel_bias(a, table, kpos, qpos)`; in-kernel `d = qpos - kpos` in `fattn-mma-f16`, `fattn-tile`, `fattn-vec` | CPU + CUDA | 14 files. 64k: ~6.5 GiB -> ~1 GiB (J). **Applied on our branch.** |
| `ggml-org/llama.cpp` #32740 (`pwilkin:codex/inkling2-flash-attention`) | compact indexed bias on existing `FLASH_ATTN_EXT` (index tensor + kv_rows) | ggml contract + CPU reference only | Open; approved by am17an, IMbackK, 0cc4m; CISC and ggerganov pending (checked 2026-11-23). Likely upstream contract. Per-backend follow-ups wanted. |
| `ggml-org/llama.cpp` #32808 (`Michael-Glaser:metal-indexed-bias-fa`) | Metal backend for #32740 | Metal | Open draft |
| `ggml-org/llama.cpp` #32752 (`pcuenca:inkling2-buffers`) | build the dense mask in chunks, no new kernel | all | Open. 90k ub512: 8.69 -> 3.19 GiB; still `O(n_ctx)` |
| `ggml-org/llama.cpp` #32883 (`sanmai/2026-query-chunking-inkling2`) | sub-ubatch query chunking, 1 GiB mask budget | all | Open prototype |
| `unslothai/llama.cpp` #386 (`inkling2-attn-banded`, danielhanchen) | new `GGML_OP_FLASH_ATTN_EXT_BANDED` + shared banded mask + per-query sinks + q8_0/q4_0 in-tile dequant; stacked on #378 and #385 | CPU, CUDA, HIP | Open, 65 files. 256k: 2400 -> 560 MiB; 1M: 9312 -> 2096 MiB. Dense fallback for ubatch <= 32, `-np > 1`, non-contiguous cells, wrapped SWA, so load-time reserve is higher. No context shift or `--cache-reuse`. |
| `vengeful180-dot/llama.cpp` branch `inkling2-vulkan`, commit `0f202c363` | `GGML_OP_REL_MASK` builds F16 mask+bias from table, split per KV head group | CPU + Vulkan | Fork only |

Related: `hridaya423/inkling-2-mini-12gb`, `alesha-pro/maya-inkling-mini` (`src/ik2`, another Inkling implementation,
referenced in the Junie session, not analysed), issue `ggml-org/llama.cpp#18606` (referenced in session, content not reviewed).

## 6. Gap between our HEAD and the "ultimate" Strata fix

Feature comparison (J, verified at the PR/commit level by Junie):

| Feature | Strata | our HEAD (hridaya423) | unsloth #386 |
|---|---|---|---|
| no `kq_b` (96 KiB/tok) | yes | yes | yes (banded path) |
| no 2D `self_rel_pos` (2 KiB/tok) | yes | yes (`kpos` 4 B/tok + `qpos`) | yes |
| no 2D `attn_mask` (1 KiB/tok at ub 512) | **yes** | **no** (1-head F16 mask kept) | no (shared banded mask) |
| FP32 bias add inside FA | yes | yes | yes |
| sinks inside online softmax | yes | equivalent (`table - sink`, sinks = 0) | yes (per-query sinks) |
| `-np > 1`, non-contiguous KV | no (single seq / tree) | yes | falls back to dense |

So the only `O(n_ctx)` item left in our HEAD is the F16 `self_attn_mask`: `n_kv * n_ubatch * 2 B` = 1 KiB/tok at
`-ub 512` (128 MiB at 131k, 1 GiB at 1M, 4 KiB/tok = 4 GiB at 1M with `-ub 2048`). Real gain of the "ultimate" step is
therefore small for typical Junie contexts (<= 128k) and grows with `-ub` and context. Confirmed in step 0 (section 10):
the slope at HEAD was 1.004 KiB/tok at `-ub 512` on both CUDA0 and host. After 3692ecb40 (pos mask) the gap is closed except
for `kpos` (4 B/tok) and the dense fallback for a unified cache with several sequences in one ubatch.

## 7. Candidate designs for this branch

Decision: A, implemented in 3692ecb40 as in `inkling-strata-plan.md` (window in op_params slot 5, mask by position inside the existing bias lookup, `kpos = INT32_MAX` for hidden cells, dense mask as fallback, switch `LLAMA_ATTN_POS_MASK_DISABLE`). Tile skipping from positions (plan step 4) was not needed: no loss at `-ub 2048`. `fattn-tile.cuh` is not reached with a bias and was not changed.

A. Mask-free causal/SWA FA path: pass `mask = nullptr` plus `qpos`/`kpos` (already available, `llm_graph_rel_bias`
   in `src/llama-graph.h:779`, `kpos [n_kv, n_stream]`, `qpos [n_tokens]`) and apply `kpos <= qpos` and the SWA window
   (`qpos - kpos < n_swa`) in the kernels (`fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-vec.cuh`).
   This is the direct Strata equivalent. Needs: skip-masked-tile logic (mask-based skipping of KV tiles is currently
   driven by the mask), sequence-id handling for `-np > 1` (fall back to the existing mask path then), padded/empty
   cells (`kpos = -1` style sentinel), CPU reference + `test-backend-ops` cases.
B. Bit-packed or per-tile mask: shrink the mask 16x or more (1 bit per cell) but keep the mask tensor; smaller change, still
   `O(n_ctx)`.
C. Query chunking (like #32883): bound the mask by splitting the ubatch; no kernel work, but still scales until the
   chunk is small, and costs extra graph nodes.
D. Stay on hridaya423 and track upstream #32740 + CUDA follow-up; spend effort only on the measured-gap items below.
E. Port the unsloth banded op: biggest diff (65 files), best long-context numbers, dense fallback complicates reserve.

Recommendation to evaluate first: A, limited to single-sequence contiguous caches, with the existing mask path as fallback
(same trade-off unsloth #386 made, but without a new op). It must keep results bit-compatible with HEAD in the
mask-present case and be checked with `teacher_force.py` and `test-backend-ops`.

## 8. Test and measurement plan (scripts are shared, see `AGENTS.local.md`)

1. Baseline: build HEAD in this worktree (`build-inkling.bat`), run `session.sh` at 32768 / 131072 / 262144 and read the
   `compute buffer size` lines from `server.log`; record per-token slope.
2. Replay the two target requests (`replays\test2\req_32k.json`, `req_76k.json`) with `replay.py`: VRAM, prefill, gen.
3. Correctness: `test-backend-ops -o FLASH_ATTN_EXT -p extent=` (add mask-free cases), `teacher_force.py` against
   the baseline binary (`--teacher-exe`, built from `erokhins/inkling`) on 32k, and `--sweep-lengths`.
4. Long context: 262144 and above, `-ub 512` and `-ub 2048`, to see the slope change.
5. Always run GPU work through `gpurun.sh` (shared GPU lock).

## 9. Open questions

- Answered: compute buffer at HEAD, see section 10 step 0 (mask = 1 KiB/tok at `-ub 512`, fixed part about 130 MiB CUDA0).
- Answered: `-np 2` works in both modes and gives the same text as the baseline (section 10, step 3 correctness).
- Open: context shift and `--cache-reuse` with the pos mask are not tested (plan step 3.6).
- Open, **correctness risk**: only CUDA and CPU are correct for Inkling-2 on this branch. Other backends (Vulkan, Metal, SYCL, OpenCL, CANN, OpenVINO, WebGPU, Hexagon, zDNN, spacemit) do not check `src[6]` (the rel bias) of FLASH_ATTN_EXT in `supports_op`: they accept the node and ignore the bias. This was already wrong at f929fd85c (no bias). With the pos mask it is worse: there is no mask tensor, so such a backend also attends to every cell (future and stale cells). No crash, only wrong output. Fix (plan step 5, not done, decision 2026-11-23: CUDA/CPU only for now): refuse `src[6]` in each of these `supports_op` (one line each, Metal needs its own check), then the scheduler falls back to CPU. Must be done before any use on these backends or any upstream PR.
- Open: K/V types other than F16 make the mma kernel convert K/V in a temporary that scales with `n_kv` (not measured).
- Upstream direction: #32740 contract (indexed bias + F16 mask) differs from hridaya423's `kpos`/`qpos` API. Rebase cost
  if upstream merges first.

## Links

- Strata: https://github.com/Niko1221/Strata (local: `..\Strata`, commit `6a6db7f`); docs: `docs/HOW_IT_WORKS.md`, `docs/MODELS.md`
- hridaya423 patch: https://github.com/hridaya423/llama.cpp/commit/68cd7533a (branch `inkling-fast`), https://github.com/hridaya423/inkling-2-mini-12gb
- upstream: https://github.com/ggml-org/llama.cpp/pull/32740, /pull/32752, /pull/32808, /pull/32883, /pull/32884 to /pull/32887 (seen in session, not reviewed), /issues/18606
- unsloth: https://github.com/unslothai/llama.cpp/pull/386 (stacked on #378, #385)
- vulkan fork: https://github.com/vengeful180-dot/llama.cpp (branch `inkling2-vulkan`, commit `0f202c363`)
- other Inkling impl: https://github.com/alesha-pro/maya-inkling-mini/tree/main/src/ik2
- Junie session: `C:\Users\erokh\.junie\sessions\session-261123-154730-118q\` (`transcript.md`, `tasks.json`)

## 10. Results

### Step 0 baseline

Build: base worktree `wt-inkling-strata-base` at f929fd85c, `build-cuda`, llama-server and test-backend-ops built without errors.
Teacher binary: `tmp/inkling/baseline-f929fd85c/bin` (exe and dll files copied from that build).
Logs: `tmp/inkling/logs/strata-s0`.

Tests (CUDA0, base build):

| run | result |
|---|---|
| FLASH_ATTN_EXT -p extent= | 40/40 passed |
| FLASH_ATTN_EXT full | 4470/4470 passed (finished inside 5 min) |

Compute buffer reserve (`sched_reserve` lines, server run with `-lv 4`; at default verbosity these lines are not printed):

| ctx | ub | CUDA0 MiB | host MiB |
|---|---|---|---|
| 32768 | 512 | 162.87 | 42.65 |
| 131072 | 512 | 259.25 | 139.03 |
| 262144 | 512 | 387.75 | 267.53 |
| 131072 | 2048 | 1041.23 | 560.60 |

Slope at ub 512 from 131072 to 262144: CUDA0 128.5 MiB per 131072 tokens (1.004 KiB per token), host 128.5 MiB per 131072 tokens (1.004 KiB per token).
From 32768 to 131072: CUDA0 96.4 MiB per 98304 tokens (1.004 KiB per token), host 96.4 MiB (1.004 KiB per token).
The slope matches the expected 1 KiB per token on both buffers. The remaining fixed part is about 130 MiB on CUDA0 and about 10 MiB on host.

Total GPU memory used after load (nvidia-smi, ub 512): 15155 MiB at 32768, 18321 MiB at 131072, 22545 MiB at 262144. At ctx 131072 with ub 2048: 19345 MiB.

Speed: server `-c 98304 -np 1 -fa on -ngl 99`, `replay.py req_32k.json 4000 3` (prompt 32144 tokens, 1727 generated):

| rep | prefill t/s | decode t/s | VRAM MiB |
|---|---|---|---|
| 0 | 2690 | 81.0 | 17302 |
| 1 | 2730 | 81.1 | 17295 |
| 2 | 2721 | 80.9 | 17295 |

VRAM after load was 17266 MiB.

### Steps 1 and 2 (pos mask), gates 1 and 2

Code: uncommitted diff on f929fd85c, 11 files, 149 lines added and 56 removed (about 83 without tests). No compile fixes were needed.
Logs: `tmp/inkling/logs/strata-gate` (buffers), `tmp/inkling/logs/strata-gate/tf` (teacher forcing), `tmp/g1a.log`, `tmp/g1b.log`.

Gate 1 (CUDA0):

| run | result |
|---|---|
| FLASH_ATTN_EXT -p extent= | 288/288 passed (window 0/INT32_MAX/100 x ns 1/2, nb 20 added) |
| FLASH_ATTN_EXT full | 4718/4718 passed (4470 + 248 new) |

The `window=0, ns=2` cases pass, so multi-stream bias indexing at HEAD is correct.

Gate 2 item 1, compute buffers at ub 512 (MiB, CUDA0 / host):

| ctx | pos mask | `LLAMA_ATTN_POS_MASK_DISABLE=1` | step 0 |
|---|---|---|---|
| 32768 | 129.37 / 9.15 | 162.87 / 42.65 | 162.87 / 42.65 |
| 131072 | 129.75 / 9.53 | 259.25 / 139.03 | 259.25 / 139.03 |

With the pos mask the slope is about 4 B per context token on both buffers (`kpos`). With the switch the numbers equal step 0.

Gate 2 items 2-5, teacher = baseline binary f929fd85c, both `-fa on`:

| item | run | top-1 | max \|d_logprob\| |
|---|---|---|---|
| 2 | 32k request (23709 tokens), pos mask | 100/100 | 0.00000 |
| 3 | same, `LLAMA_ATTN_POS_MASK_DISABLE=1` | 100/100 | 0.00000 |
| 4 | default prompt | 100/100 | 0.00000 |
| 5 | sweep 7..2100 vs `-fa off`, with and without the switch | 12/12 both | output identical line by line |

All exact. Step 3 results are in the subsections below.

### Step 3 speed and VRAM

New = 3692ecb40 (pos mask), base = f929fd85c. Logs: `tmp/inkling/logs/strata-s3` (`all.sh`, `sess.sh`, `*.out`).
Server `-ngl 99 -fa on -np 1 --jinja`, one session per request and binary, GPU lock held for the whole session.
Values are means over the reps. VRAM is nvidia-smi after load; it also moves by 100-250 MiB between sessions of the same binary (other processes), so use the compute buffer numbers below for the exact delta.

Replay, `-c 98304`, `-ub 512`:

| request | binary | prefill t/s | decode t/s | VRAM after load MiB |
|---|---|---|---|---|
| 32k (3 reps) | base | 2615 (2586, 2573, 2687) | 78.5 | 17496 |
| 32k (3 reps) | new | 2673 (2699, 2725, 2596) | 77.8 | 17261 |
| 32k | diff | +2.2% | -0.9% | -235 |
| 76k (2 reps) | base | 2416 (2404, 2428) | 71.7 | 18049 |
| 76k (2 reps) | new | 2491 (2475, 2506) | 72.9 | 17577 |
| 76k | diff | +3.1% | +1.6% | -472 |

Step 0 baseline 32k was 2690/2730/2721 t/s and 17266 MiB; the re-run of the baseline today is slower (noise between runs is about 3%).
Prefill is equal or faster, decode is equal inside noise.

`-ub 2048`, `-c 98304`, replay 32k (3 reps, 1863 tokens generated):

| binary | prefill t/s | decode t/s | VRAM after load MiB |
|---|---|---|---|
| base | 3301 (3249, 3324, 3331) | 79.8 | 18257 |
| new | 3332 (3306, 3350, 3341) | 79.3 | 17994 |
| diff | +0.9% | -0.6% | -263 |

No loss at `-ub 2048`, so decision D8 does not need step 4 from this measurement.

Compute buffers (`sched_reserve`, `-lv 4`, MiB CUDA0 / host), new binary vs step 0:

| ctx | ub | step 0 | new | diff |
|---|---|---|---|---|
| 262144 | 512 | 387.75 / 267.53 | 130.25 / 10.03 | -257.50 / -257.50 |
| 131072 | 2048 | 1041.23 / 560.60 | 517.23 / 36.60 | -524.00 / -524.00 |
| 262144 | 2048 | not measured | 517.73 / 37.10 | |

At ctx 98304, ub 512 the step 0 slope gives about 227 MiB on CUDA0 against about 129.6 MiB with the pos mask: about 97 MiB lower, as expected.
The ctx growth of the new buffers is 0.5 MiB per 131072 tokens (4 B per token).
GPU memory after load (nvidia-smi): 22330 MiB at 262144/512 (step 0: 22545), 22956 MiB at 262144/2048, 18860 MiB at 131072/2048 (step 0: 19345).

### Step 3 correctness

Items 4 and 5 of step 3. Logs: `tmp/inkling/logs/strata-s3c`. Prompts A (C linked list reversal) and B (why the sky is blue), `/v1/chat/completions`, temperature 0, seed 1, max_tokens 200, `cache_prompt` false, `-fa on -ngl 99 -c 16384`.
Compared field: the whole message (content and reasoning_content), byte for byte.
The log of the new binary shows `n_slots = 2, kv_unified = 'false'` for `--no-kv-unified` and `kv_unified = 'true'` for `--kv-unified`.

Item 4, parallel slots:

| comparison | A | B |
|---|---|---|
| baseline -np 1 alone vs baseline -np 2 (both modes) alone | identical | identical |
| baseline -np 1 alone vs new -np 2 (both modes) alone | identical | identical |
| baseline -np 1 alone vs any concurrent run (baseline or new) | differs | differs |
| baseline --no-kv-unified concurrent, round 1 vs round 2 | identical | identical |
| baseline --kv-unified concurrent, round 1 vs round 2 | identical | identical |
| new --no-kv-unified concurrent vs baseline --no-kv-unified concurrent (3 rounds) | identical | identical |
| new --kv-unified concurrent vs baseline --kv-unified concurrent (2 rounds) | identical | identical |

Concurrent output differs from alone output already at baseline (batched decoding of 2 sequences changes float results). So the valid comparison is new concurrent against baseline concurrent: identical in both modes.
The concurrent output of baseline is deterministic across rounds. Both modes of the new binary are identical to the baseline of the same mode.
The new_ns session run with the buggy script lost round 2 (empty files, session killed by timeout); two later rounds with the fixed script were taken.

Item 5, CPU reference end to end (`teacher_force.py --ngl 0 --n-tokens 20`, prompt "Write a C function that reverses a linked list.", student new binary, teacher baseline, both `-fa on`):

| run | top-1 | max \|d_logprob\| | mean NLL |
|---|---|---|---|
| pos mask | 20/20 | 0.00000 | 0.28323 |
| `LLAMA_ATTN_POS_MASK_DISABLE=1` | 20/20 | 0.00000 | 0.28323 |

Both runs are identical to each other and an exact match vs the teacher.

### Step 3.6 position shifts

Logs and scripts: `tmp/inkling/logs/strata-s36`. Model Inkling-2-Mini UD-Q3_K_XL, `-ngl 99 -fa on -np 1 --jinja`, `/completion` raw prompt, temperature 0, seed 1. Configs: baseline `f929fd85c`, new pos mask, new with `LLAMA_ATTN_POS_MASK_DISABLE=1` (dense).

What is supported:
- By default both `--cache-reuse` and `--context-shift` are refused. Log: "cache_reuse is not supported by this context, it will be disabled".
- Cause: `llama_kv_cache_iswa::get_can_shift()` also needs `kv_base->get_size() == kv_swa->get_size()`. The default SWA cache is smaller (1536 against 8192 cells).
- With `--swa-full` both caches have equal size and both features are enabled.
- Cache reuse: the chunk is shifted (`seq_rm` + `seq_add`), but the server then logs "forcing full prompt re-processing due to lack of cache data" and clears the sequence. The model has a recurrent state (checkpoints with pos_min == pos_max), which cannot follow a moved chunk. So the shifted cells are never attended to. This scenario does not exercise kpos on shifted cells.
- Context shift (`--context-shift -c 4096 --swa-full`): works, shifted KV is used for the rest of the generation. This is the real test.

Runs (all `--swa-full`):
- S1: `-c 8192 --cache-reuse 256 -lv 4`. P1 = A+B+C (3337 tokens), P2 = A+C (2671 tokens). Log: "reusing chunk with size 1127, shifting KV cache [2210, 3337) -> [1544, 2671)", then the forced full re-processing.
- S2: `-c 4096 --context-shift -lv 4`, prompt 2669 tokens, n_predict 3600, ignore_eos. Log: "slot context shift, n_keep = 0, n_left = 4095, n_discard = 2047", 2 shifts in every config.
- Fresh: P2 on a fresh server, reference only.

| comparison | text | identical |
|---|---|---|
| S1 P2: pos vs dense | 499ac5c7 | yes |
| S1 P2: pos vs baseline | 499ac5c7 | yes |
| S1 P2: pos vs fresh server | 499ac5c7 | yes |
| S2 (12680 bytes, 2 shifts): pos vs dense | 90de4a52 | yes |
| S2: pos vs baseline | 90de4a52 | yes |

No finding. Caveat: S1 proves only that reuse with forced re-processing is stable. S2 is the position-moving test.

## 11. Strata engine comparison (SQ-15 vs our pos-mask build)

Goal: measure the real Strata engine (not just read its source) on the same two requests, to see how much of the
prefill/decode gap to Strata our pos-mask work (section 10, step 3) actually closed.

### Setup

- Strata source checkout `..\Strata` (commit `6a6db7f`, terminal/engine build 0.2.20/0.1.66; the installed app was 0.2.9
  and does not know the Inkling family, so the source checkout was used directly: `terminal/.venv` created, Strata's own
  `terminal/requirements.txt` installed into it).
- Model: `strata-engine/Inkling-2-Mini-Strata` build **SQ-15** (14.3 GB, downloaded this session to
  `C:\Users\erokh\AppData\Local\Strata\Strata-data\models\inkling-2-mini\Inkling-2-Mini-SQ-15.strata`), engine package
  `0.1.66-windows-x86_64-cuda-cuda13` (417 MB, downloaded alongside it).
- Launch (`terminal/setup.py`, what `strata launch`/`START-HERE.bat` call): `--family inkling-2-mini --model SQ-15
  --context 98304 --kv int8 --vision no --mtp off --drafter off --parallel 1 --host 127.0.0.1 --port 8096 --no-browser
  --yes --output jsonl`. `--mtp off --drafter off` deliberately turn off Strata's speculative decoding stack (MTP /
  DSpark) so the comparison is attention/compute-buffer against attention/compute-buffer, not drafting against no
  drafting. `--kv int8` is Strata's default and was left as-is: our side runs F16 KV, so **the VRAM numbers below are
  not apples-to-apples on KV format**, only the compute-buffer/attention part is the thing this branch changed.
- llama.cpp side: this worktree's `build-cuda/bin/llama-server.exe` (commit fc73aaa58, i.e. 3692ecb40 pos-mask + later
  docs-only commits), `-m Inkling-2-Mini-UD-Q3_K_XL.gguf -ngl 99 -fa on -c 98304 -ub 512 -np 1 --port 8765 --jinja`,
  same convention as step 3.
- Both engines were given the exact same two captured requests (`replays/test2/req_32k.json`, `req_76k.json`), one
  server session per (engine, request), GPU lock held for the whole session (`gpurun.sh`, `GPU_OWNER=strata-bench`).
  Reps: 3 for the 32k request, 2 for the 76k request, `max_tokens` cap 4000, same as the established convention.

**Anti-cache nonce.** Strata has a prompt/prefix cache (`MESSAGE_BOUNDARY_CACHE.md`, `PROMPT_CACHE_TAIL.md`) and no
documented per-request "force full prefill" flag was found. Resending the identical captured body would hit this cache
on reps after the first and give a meaningless near-zero prefill. Fix: before sending, a short unique nonce comment
(`<!-- bench <12 hex chars> -->\n`) is prepended to the system message of every rep, on **both** engines, so the two
engines see literally the same input tokens and no rep can reuse another rep's KV. llama.cpp's own `cache_prompt:false`
already forces full reprocessing regardless, so the nonce is harmless there; it changes `prompt_tokens` by about 10-16
tokens out of 32k/76k, negligible.

**`model` field bug (found and fixed during this run).** The captured requests carry
`"model": "Inkling-2-Mini-UD-Q3_K_XL"` (the Junie-side alias). llama-server ignores this field. Strata's OpenAI endpoint
does not: it 404'd on every real (non-sanity) request until the field was dropped from the body before sending. The
first attempt at this measurement lost both the 32k and the 76k Strata runs to this (`HTTP Error 404: Not Found`,
`bench_client.py`), and because the failure happened inside a `set -e` script before the stop/cleanup lines ran, it
also left two orphaned `strata-inkling.exe` + `server.py` + `setup.py` process chains running (not cleaned by
`strata launch --stop`, since they were started by `setup.py` directly, not tracked by `launch`); they were found with
`Get-CimInstance Win32_Process` and ended with `taskkill //PID <pid> //F //T`. The session script was changed to always
tree-kill its server chain via a `trap ... EXIT`, but the trap itself had a second bug: it tree-killed `$SP`, the PID
Git Bash's `$!` gives for a backgrounded native `.exe` — on this MSYS setup that is **not** the real Windows PID
`taskkill` needs (`ps`'s own `WINPID` column has the real one), so every automated cleanup in this session silently
no-opped (`taskkill` against a nonexistent PID, swallowed by `|| true`) and left one orphaned
`setup.py -> setup.py -> server.py -> server.py -> strata-inkling.exe` chain per Strata session. All of them (one
after the first crash, one after the final 76kB run) were found by matching `CommandLine` in
`Get-CimInstance Win32_Process` (port `8096` / the model's run-config name) and tree-killed from the real PID. A
correct fix would resolve the real PID from `$SP`'s `WINPID` (or from the command-line match) before trapping, not
implemented here since the runs were already done by the time this was found; flagging it so the next session's
scripts don't repeat it.

Strata's chat completion response does carry the same non-standard `timings` object as llama.cpp
(`prompt_per_second`, `predicted_per_second`, `prompt_ms`, `predicted_ms`) plus a standard OpenAI `usage` block — its
engine really is llama.cpp-derived under the hood (`engine/third_party/strata-llama`) — so the existing
`replay.py`-style parsing applies unchanged once the `model` field is dropped.

**Host RAM caveat.** The plan called for an idle-before / after-load / after-gen triple per session; only
after-load/after-gen were actually captured (whole-system `TotalVisibleMemorySize - FreePhysicalMemory` via
PowerShell, not a per-process number), and the llama.cpp and Strata sessions ran at different points in the overall
session with different ambient processes (venv setup, downloads). The absolute RAM numbers below are reported as
measured but should be read as rough/qualitative, not a controlled delta; only the within-session after-load-to-after-gen
growth is reasonably trustworthy.

### Results

Prefill/decode are the mean of the per-rep `timings.prompt_per_second` / `timings.predicted_per_second`; VRAM is
`nvidia-smi` read right after each response.

32k request (prompt ~32,138-32,160 tokens after the nonce):

| engine | rep | prompt tok | completion tok | prefill t/s | decode t/s | VRAM after response MiB |
|---|---|---|---|---|---|---|
| llama.cpp (pos-mask) | 0 | 32160 | 1618 | 2673.90 | 80.80 | 17116 |
| llama.cpp (pos-mask) | 1 | 32158 | 1939 | 2753.80 | 80.72 | 17110 |
| llama.cpp (pos-mask) | 2 | 32157 | 364 | 2750.75 | 79.65 | 17110 |
| Strata SQ-15 | 0 | 32141 | 1725 | 4951.5 | 90.8 | 17364 |
| Strata SQ-15 | 1 | 32138 | 1354 | 5008.1 | 91.6 | 17371 |
| Strata SQ-15 | 2 | 32139 | 1594 | 4961.3 | 85.6 | 18230 |

76k request (prompt ~75,967-75,999 tokens after the nonce):

| engine | rep | prompt tok | completion tok | prefill t/s | decode t/s | VRAM after response MiB |
|---|---|---|---|---|---|---|
| llama.cpp (pos-mask) | 0 | 75999 | 3739 | 2482.04 | 73.48 | 17117 |
| llama.cpp (pos-mask) | 1 | 75998 | 3533 | 2499.03 | 73.94 | 17117 |
| Strata SQ-15 | 0 | 75967 | 3283 | 3676.9 | 75.6 | 18784 |
| Strata SQ-15 | 1 | 75968 | 100 | 3907.5 | 77.1 | 18783 |

Summary (mean over reps; VRAM after load is the nvidia-smi reading right after the model finished loading, before any request):

| request | engine | prefill t/s (mean) | decode t/s (mean) | VRAM after load MiB | VRAM after last rep MiB | RAM after load | RAM after last rep |
|---|---|---|---|---|---|---|---|
| 32k | llama.cpp pos-mask | 2726.2 | 80.4 | 17080 | 17110 | 45.88 GiB | 49.90 GiB |
| 32k | Strata SQ-15 | 4973.6 | 89.3 | 17394 | 18230 | 34.79 GiB | 35.93 GiB |
| 76k | llama.cpp pos-mask | 2490.5 | 73.7 | 17074 | 17117 | 45.37 GiB | 50.30 GiB |
| 76k | Strata SQ-15 | 3792.2 | 76.4 | 17650 | 18784 | 35.99 GiB | 35.80 GiB |

Strata/llama.cpp speedup: 32k prefill **1.82x**, 32k decode **1.11x**; 76k prefill **1.52x**, 76k decode **1.04x**.

### Reading the numbers

- **Prefill**: Strata is still 1.5-1.8x faster than our pos-mask build on both requests. Section 6 already said the
  pos-mask closes the *buffer* gap (no more `O(n_ctx)` mask), not necessarily the *compute* gap; this measurement
  confirms a real prefill throughput gap remains, likely Strata's tiled cuBLAS prefill path (section 3) and/or other
  engine-level differences (int8 KV means less data to move, different batching/thread tuning, etc.) — not isolated
  further here.
- **Decode**: much closer, 4-11% in Strata's favor, inside the kind of run-to-run noise seen in section 10 (step 3
  noted "noise between runs is about 3%"), so decode speed is roughly at parity with drafting off on both sides.
- **VRAM after load** is close (within a few hundred MiB) despite Strata's KV being int8 against our F16 — expected,
  since this is not an apples-to-apples KV-format comparison (see setup notes) and Strata reserves a fixed
  `vram-experts-gb` workspace (12.28 GB in the run config) independent of KV format.
  **VRAM after generation grows noticeably more for Strata** across reps of the same session (32k: 17394 -> 18230 MiB,
  +836 MiB over 3 distinct nonce'd prompts; 76k: 17650 -> 18784 MiB, +1134 MiB over 2) while llama.cpp stays flat
  (within normal noise, 17080 -> 17110-17117 MiB). Plausible explanation: Strata's prompt/message-boundary cache keeps
  more than one distinct conversation's KV resident (each nonce'd request looks like a brand new conversation to it),
  while our pos-mask build's one `-np 1` slot simply overwrites. Not confirmed further; worth a follow-up if VRAM
  headroom at long sessions matters.
- 76k Strata rep1 generated only 100 tokens against rep0's 3283 (and llama.cpp's 3533-3739); this kind of generated-length
  difference between runs/engines was already seen and is not a correctness signal by itself (section 10, step 0 item 9) —
  temperature/seed reproduce the request but not across different engines or even across FP-nondeterministic reruns.

Logs and scripts (git-ignored, not committed): `C:\Users\erokh\junie-local-project\kvq-scratch\strata-bench\` —
`bench_client.py` (shared nonce'd client, both engines), `llama_session_32k.sh`/`llama_session_76k.sh`,
`strata_session.sh`, and the `*_session.log` / `*_results.log` / `*_server.log` files per run.

### 11.1 Strata with its built-in drafter (decode reference, not comparable)

Inkling-2-Mini SQ-15 ships a built-in speculative drafter (DSpark) with the model, no extra download. The measurement
above deliberately ran with `--mtp off --drafter off` to isolate the attention/compute-buffer comparison from
speculative decoding. This subsection re-runs decode with the drafter on, as a separate reference number only.

**Not comparable to the llama.cpp decode numbers above or anywhere else in this doc.** Our llama.cpp side has no
active speculative decoding in this comparison and will not get one here: `ggml-org/llama.cpp` PR #32218
(`pwilkin/inkling2-dspark`, "feat: Inkling 2 DSpark support") was checked and ruled out, out of scope, not pursued.
As of this writing it is open, not merged, and not close: ggerganov (code owner) has requested changes that are
still blocking despite two approvals (ngxson, ruixiang63). It already went through three rewrites (the author's own,
one from ggerganov that tried to reuse more existing code, then ngxson's parallel branch cherry-picked on top).
Reviewer ruixiang63 flagged that it touches broad files seemingly unrelated to Inkling2 (`common/sampling.cpp`,
`common/speculative.cpp`, `src/llama-context.cpp`) in ways "unclear... could break existing DSpark models," and
noted the root complication: Inkling2's DSpark variant uses a markov-rnn head instead of the standard Markov head
llama.cpp's DSpark sampler expects, and lacks the confidence head that sampler relies on, so this is new sampling
logic, not just wiring up an existing mechanism. Community reports on the PR also include a GGUF tensor-shape
mismatch that requires reconverting the draft model, and a tensor-split assertion failure on multi-GPU setups. The
numbers below are Strata's own best case with drafting on, nothing more.

**Config:** same launch as section 11 except drafting was left on, explicitly as `--drafter dspark` (checked valid
via `strata launch --plan`, which reported `"drafting": "Drafting: DSpark Block-8 (built into the model)"` for the
default too — `--mtp` was simply omitted, no `off`). Confirmed what actually loaded three ways: the server's own run
config `strata-inkling-2-mini-sq-15.json` has `"drafting": "dspark"` and no `--draft off` in the engine args (unlike
section 11's config, which had `--draft off`); the startup log advertises "embedded DSpark drafter"; and every
response's `timings` carries non-zero `draft_n`/`draft_n_accepted` (e.g. the sanity call: `"draft_n": 19,
"draft_n_accepted": 1`). Same requests, same rep counts (3 / 2), same `max_tokens` 4000 cap, same nonce anti-cache,
same GPU lock discipline, one session per request.

| request | rep | prompt tok | completion tok | prefill t/s | decode t/s | VRAM after response MiB |
|---|---|---|---|---|---|---|
| 32k | 0 | 32139 | 1722 | 4973.7 | 110.2 | 18417 |
| 32k | 1 | 32138 | 115 | 5002.6 | 146.0 | 18417 |
| 32k | 2 | 32139 | 1488 | 5002.8 | 109.4 | 18385 |
| 76k | 0 | 75967 | 100 | 3858.4 | 109.9 | 19819 |
| 76k | 1 | 75968 | 100 | 3901.9 | 118.8 | 19817 |

Mean: 32k decode **121.9 t/s** (prefill 4993.0 t/s, as expected prefill is not materially affected by drafting,
within noise of the no-drafter 4973.6 t/s), 76k decode **114.4 t/s** (prefill 3880.2 t/s, also within noise of the
no-drafter 3792.2 t/s). VRAM after load: 17195 MiB (32k session), 17163 MiB (76k session) — about in line with the
no-drafter runs; VRAM after generation is higher (18385-18417 MiB at 32k, 19817-19819 MiB at 76k) since the drafter
model and its own working state are resident.

Against this doc's no-drafter Strata decode numbers (section 11: 89.3 t/s at 32k, 76.4 t/s at 76k), the built-in
drafter is **1.36x faster at 32k and 1.50x faster at 76k** — but again, there is nothing on the llama.cpp side of
this doc to compare that ratio against; it describes Strata's own drafting benefit, not a cross-engine gap. Both 76k
reps and one 32k rep stopped at a short, round completion length (100-115 tokens, `finish_reason: tool_calls`),
so per-rep decode t/s there is a small-sample estimate (the model's own early-stop choice, same caveat as the
generated-length differences noted in section 11).

Section 11 above is already committed (`faa9edfa0`); this subsection (11.1) is a working-tree change on top of it,
uncommitted per `AGENTS.md` (no agent commit without explicit human approval).

## 12. DSpark drafter on llama.cpp

Analysis of Strata's drafter and the implementation plan are in `inkling-dspark-plan.md` (2026-11-23, plan only, nothing implemented).
It supersedes the "ruled out" note on #32218 in 11.1: the PR was rewritten, its head `9f57b4453` is 692 lines on a base that our HEAD already contains, and Unsloth publishes drafter GGUFs for it.
Results of that plan go below this line.

## 13. DSpark drafter measurements (plan step 3, 2026-11-23)

Setup. Target `Inkling-2-Mini-UD-Q3_K_XL.gguf`. Server flags `-c 98304 -ub 512 -np 1 -fa on -ngl 99 --jinja`, drafter flags `-md <file> --spec-type draft-dspark --spec-draft-n-max K -ngld 99`.
Spec binary: this branch (HEAD `07d22ba2e`) plus PR #32218 at `9f57b4453` (staged diff, CUDA build). Baseline binary: `3692ecb40` (no PR).
Drafter files: `dspark-Inkling-2-Mini-BF16-mainline.gguf` 749,371,936 bytes (0.75 GB), `dspark-Inkling-2-Mini-Q8_0-mainline.gguf` 503,889,952 bytes (0.50 GB). BF16 unless a table says Q8_0.
Requests: `req_32k.json` (32,144 tokens) and `req_76k.json` (75,985 tokens), the captured Junie requests, seed 100000, stop string `</COMMAND>`, tools set.
Sampling: the requests carry no sampling fields. The server uses the GGUF defaults: temperature 1.0, top_k 0, top_p 1.0, min_p 0. "Request sampling" below means these.
Draft sampling: default (`--spec-draft-sampling greedy`) unless a row says otherwise.
Client: `tmp/inkling/tools/dspark_client.py` (own client; `cache_prompt=false` forces a full prefill every rep, so there is no prompt cache hit between reps). Two reps per session.
Generation length: `max_tokens` 4000 for 32k and 2000 for 76k (a 76k session with 4000 tokens does not fit the 5 minute cap). Many replies end earlier by themselves (tool call).
One session per request and configuration, each under the GPU lock, wall cap 300 s. Logs: `tmp/inkling/logs/dspark-s3/` (`<tag>.client.txt`, `<tag>.server.log`). Table script: `tmp/inkling/tools/dspark_s3_table.py`.
`-lv 4` (needed for the per-position acceptance line) does not change speed: 32k, K=3, 2 reps each, decode 100.78 and 101.16 t/s without it and 100.82 and 101.58 t/s with it. All spec sessions use `-lv 4`.

### 13.1 Step 2 result (lossless gate)

Accepted by the user as passed. Run 1: 298 of 300 tokens are the top-1 of the baseline binary. The misses are at step 5 (gap 0.0187) and step 20 (gap 0.0695, the teacher margin there was 0.013).
The user accepted the step 20 miss as a near tie, because a control run with the baseline binary only (ngram-simple drafts, no PR code, run c2) also flipped a token, with gap 0.0921.
Runs 2 to 5: 99 of 100, 300 of 300, run 4 identical to run 1, 20 of 20. Logs: `tmp/inkling/logs/dspark-s2/`.

### 13.2 Scope

The user cut the batch after the first results ("overtesting, the drafter speed will change next"). Deferred until the drafter speed work: M1 (synthetic round cost; the first M1 try failed because `ignore_eos` cannot be combined with the tool grammar), M8 (top draft candidates), the extra `-lv 4` VRAM load, the baseline re-run at the end of the batch (B1).
Measured before the cut and kept: `--spec-draft-sampling probabilistic`, temperature 0, Q8_0 drafter (13.7 to 13.9).

### 13.3 Baseline (B, binary `3692ecb40`, no drafter)

| Request | Prompt tokens | Gen tokens | Prefill t/s (rep0, rep1) | Decode t/s (rep0, rep1) | VRAM MiB after load | VRAM MiB after gen |
|---|---|---|---|---|---|---|
| 32k | 32144 | 1727 | 2705, 2747 | 80.75, 80.70 | 17264 | 17300 |
| 76k | 75985 | 2000 (cap) | 2495, 2508 | 74.03, 74.09 | 17264 | 17300, 17307 |

Speedups below use the mean of the two reps: 80.72 t/s (32k) and 74.06 t/s (76k). Baseline B at the start only.

### 13.4 M3: decode speed, BF16 drafter, request sampling, draft sampling greedy

| Request | K | Decode t/s (rep0, rep1) | Speedup vs B | Gen tokens | Prefill t/s (rep0, rep1) | VRAM MiB load / gen |
|---|---|---|---|---|---|---|
| 32k | 1 | 92.69, 92.30 | 1.15 | 1962 | 2624, 2660 | 19239 / 19329 |
| 32k | 2 | 111.59, 111.70 | 1.38 | 1690 | 2605, 2657 | 19243 / 19333 |
| 32k | 3 | 92.74, 93.29 | 1.15 (disturbed, see notes) | 1098 | 2492, 2467 | 19949 / 19661 |
| 32k | 5 | 99.34, 104.84 | 1.26 | 1529 | 2406, 2476 | 19535 / 19622 |
| 32k | 7 | 100.86, 101.72 | 1.26 | 1684 | 2606, 2653 | 19510 / 19601 |
| 76k | 1 | 80.70, 80.78 | 1.09 | 2000 | 2417, 2432 | 19442 / 19529 |
| 76k | 2 | 91.01, 91.30 | 1.23 | 2000 | 2409, 2434 | 19297 / 19391 |
| 76k | 3 | 95.04, 95.10 | 1.28 | 2000 | 2409, 2426 | 19303 / 19452 |
| 76k | 5 | 97.52, 97.85 | 1.32 | 2000 | 2417, 2433 | 19365 / 19414 |
| 76k | 7 | 101.33, 101.60 | 1.37 | 2000 | 2409, 2432 | 19246 / 19531 |

Notes.
- The 32k K=3 row looks low. The same configuration ran four times in the `-lv` check (K=3, 32k): 100.78, 101.16, 100.82, 101.58 t/s (mean 101.1, speedup 1.25), with the same 1098 generated tokens. The M3 session of K=3 had 700 MiB more VRAM after load (19949 against 19247), so another process was using the GPU. Treat 101.1 as the K=3 value for 32k. The prefill of that session is low too (2467 to 2492 t/s).
- The text differs between K values on the 32k request (1962, 1690, 1098, 1529, 1684 tokens, baseline 1727), because verify batches use other kernels than single-token decode (D3) and the sampling is random with a fixed seed. Speed and acceptance between K values on 32k compare different texts. On 76k all runs end at the 2000 token cap, but the text differs there too.
- The idle desktop already uses about 1.1 GiB, and other processes add a few hundred MiB, so single VRAM readings have a spread of 300 to 700 MiB.

### 13.5 Round time split (rep1 of each M3 session)

`round_ms = decode_s * 1000 / (draft_n / K)`, `draft_ms = dur(g) / #gen drafts` from the `statistics` line (cumulative over both reps), `verify_ms = round_ms - draft_ms` (target batch of K+1 tokens, rollback and acceptance included), `c = round_ms * baseline_tps / 1000` (round cost in single-token decodes, with the real acceptance, not the synthetic one). BF16, request sampling.

| Request | K | round ms | draft ms | verify ms | draft share of round | c | Strata c(K+1) |
|---|---|---|---|---|---|---|---|
| 32k | 1 | 18.5 | 1.60 | 16.9 | 8.6% | 1.50 | 1.51 |
| 32k | 2 | 20.5 | 2.04 | 18.5 | 10.0% | 1.66 | 1.74 |
| 32k | 3 (undisturbed run, rep1 of the check) | 24.0 | 2.43 | 21.6 | 10.1% | 1.94 | 1.96 |
| 32k | 5 | 28.1 | 3.47 | 24.6 | 12.3% | 2.27 | 2.35 |
| 32k | 7 | 29.5 | 4.00 | 25.5 | 13.6% | 2.38 | 2.69 |
| 76k | 1 | 21.6 | 1.78 | 19.8 | 8.2% | 1.60 | 1.51 |
| 76k | 2 | 23.4 | 2.22 | 21.2 | 9.5% | 1.74 | 1.74 |
| 76k | 3 | 27.3 | 2.65 | 24.7 | 9.7% | 2.03 | 1.96 |
| 76k | 5 | 30.1 | 3.47 | 26.7 | 11.5% | 2.23 | 2.35 |
| 76k | 7 | 33.4 | 4.23 | 29.1 | 12.7% | 2.47 | 2.69 |

The round cost in llama.cpp is close to the Strata model (that row is for Strata kernels). `draft_n` counts K per round, so a round that stopped early would make `round_ms` slightly too small. The baseline single-token time is 12.4 ms (32k) and 13.5 ms (76k).

### 13.6 M2 and M4: acceptance (statistics lines, both reps, same text in both reps)

Acceptance rate per draft position and mean accepted length (includes the target token). BF16, request sampling, greedy draft sampling.

| Request | K | Rate at positions 1..K | Mean acc len |
|---|---|---|---|
| 32k | 1 | 0.717 | 1.72 |
| 32k | 2 | 0.752, 0.546 | 2.30 |
| 32k | 3 | 0.676, 0.440, 0.324 | 2.44 |
| 32k | 5 | 0.698, 0.483, 0.344, 0.235, 0.179 | 2.94 |
| 32k | 7 | 0.677, 0.459, 0.312, 0.225, 0.163, 0.107, 0.062 | 3.01 |
| 76k | 1 | 0.747 | 1.75 |
| 76k | 2 | 0.690, 0.450 | 2.14 |
| 76k | 3 | 0.711, 0.516, 0.376 | 2.60 |
| 76k | 5 | 0.679, 0.465, 0.353, 0.258, 0.196 | 2.95 |
| 76k | 7 | 0.704, 0.504, 0.364, 0.288, 0.230, 0.166, 0.125 | 3.38 |

M4 (block size, K=3 against K=7, positions 1..3): 32k 0.676, 0.440, 0.324 against 0.677, 0.459, 0.312. 76k 0.711, 0.516, 0.376 against 0.704, 0.504, 0.364. The difference is at most 1.9 points (32k position 2), 1.2 points or less in the other cells, and the signs are mixed. The texts differ, so this is only a rough bound.
No code prompt was run (M2 asked for one); the two requests are Junie agent turns.

### 13.7 Temperature 0 (request overridden to temperature 0, BF16, greedy draft)

The replies are short (148 to 178 tokens, the first tool call), so t/s is a small sample. No baseline at temperature 0 was run, so no speedup is claimed.

| Request | K | Decode t/s (rep0, rep1) | Gen tokens | Rate at positions 1..K | Mean acc len |
|---|---|---|---|---|---|
| 32k | 3 | 116.88, 117.85 | 148 | 0.833, 0.688, 0.542 | 3.06 |
| 32k | 7 | 140.93, 142.12 | 156 | 0.914, 0.714, 0.571, 0.457, 0.371, 0.286, 0.114 | 4.43 |
| 76k | 3 | 103.38, 106.05 | 178 | 0.833, 0.617, 0.500 | 2.95 |
| 76k | 7 | 117.29, 118.53 | 178 | 0.822, 0.600, 0.444, 0.356, 0.267, 0.267, 0.178 | 3.93 |

Positions 1..3 at K=3 against K=7: 76k 0.833, 0.617, 0.500 against 0.822, 0.600, 0.444. 32k 0.833, 0.688, 0.542 against 0.914, 0.714, 0.571. Only 35 to 60 rounds per session, so not reliable to one point.

### 13.8 `--spec-draft-sampling probabilistic` (BF16, request sampling)

| Request | K | Decode t/s | Gen tokens | Mean acc len | Rate at positions 1..K |
|---|---|---|---|---|---|
| 32k | 3 | 20.60, 20.77 | 1596 | 2.70 | 0.761, 0.555, 0.384 |
| 32k | 7 | 18.88 (one rep, the 300 s cap cut the session) | 2475 | 3.22 | 0.713, 0.514, 0.357, 0.253, 0.179, 0.117, 0.084 |
| 76k | 3 | 23.32, 23.32 | 168 | 3.13 | 0.907, 0.704, 0.519 |
| 76k | 7 | 21.74, 21.97 | 426 | 3.19 | 0.806, 0.537, 0.388, 0.224, 0.119, 0.090, 0.022 |

Odd: decode is 4 to 5 times slower than the baseline (0.23 to 0.31 of B). The round takes 130 to 170 ms (`verify_ms` 127 to 165 against 25 for greedy draft), while the draft itself costs 3.3 to 5.3 ms. The slowdown is in the probabilistic path (the draft distribution `q` that goes to the verifier), not in the drafter. Not analysed here. The acceptance is not worse than with greedy draft sampling.

### 13.9 M5: Q8_0 drafter (request sampling, greedy draft)

| Request | K | Decode t/s (rep0, rep1) | Speedup vs B | Mean acc len | Rate at positions 1..3 | Draft ms |
|---|---|---|---|---|---|---|
| 32k | 3 | 101.14, 101.00 | 1.25 | 2.45 | 0.679, 0.441, 0.325 | 2.30 |
| 32k | 7 | 100.55, 101.07 | 1.25 | 2.99 | 0.674, 0.457, 0.311 | 3.88 |
| 76k | 3 | 94.76, 94.35 | 1.28 | 2.59 | 0.710, 0.510, 0.366 | 2.50 |
| 76k | 7 | 100.24, 100.73 | 1.36 | 3.36 | 0.705, 0.502, 0.360 | 4.14 |

Same text and acceptance as BF16 within 0.5 points (K=7 positions 1..3 on 32k: 0.674, 0.457, 0.311 against 0.677, 0.459, 0.312). Decode speed is equal within noise. The draft step is 0.1 to 0.3 ms shorter. Q8_0 saves 234 MiB of weights.

### 13.10 M6: VRAM (K=7, nvidia-smi of the whole GPU)

| Config | After load MiB | After gen MiB |
|---|---|---|
| B (no drafter) | 17264 | 17300 |
| BF16, K=7 | 19246 to 19510 | 19347 to 19601 |
| Q8_0, K=7 | 19508 to 19521 | 19608 to 19624 |

Increase over B: about +2.0 GiB after load (19246 against 17264, the lowest BF16 reading) and +2.0 to +2.3 GiB after generation. Reserve lines of the K=7 BF16 session (32k), from the server log:

| Item | MiB |
|---|---|
| drafter weights, BF16 file (Q8_0 file: 469.11) | 703.25 |
| drafter KV, two caches (384 + 14) | 398.00 |
| drafter compute buffer (graph pp bs=512, tg bs=1) | 787.86 (+ 102.27 host) |
| target recurrent state with 7 snapshots (K=3: 14.62) | 29.25 |
| target compute buffer, at load, then after warmup | 129.62, 138.63 |
| target KV (unchanged) | 3072 + 240 |

The plan estimate (1.0 to 1.2 GiB, section 5) was too low by the draft compute buffer, which the plan had as unknown: 788 MiB for the drafter graph at `-ub 512`. The KV estimate (384 MiB + 12 MiB) was right. The listed lines add up to about 1.9 GiB, which matches the measurement. The baseline server was not run with `-lv 4`, so its reserve lines were not collected. The extra `-lv 4` load was skipped (13.2), but every spec session already runs with `-lv 4`.

### 13.11 M7: prefill

Prefill t/s, mean of two reps (baseline 2726 on 32k, 2502 on 76k). Drafter, K=1, 2, 7 on 32k: 2642, 2631, 2630 (-3.1 to -3.5%). On 76k: 2425, 2422, 2420 (-3.1 to -3.3%). The disturbed 32k sessions (K=3, K=5, see 13.4) show 2406 to 2492. The prefill loss with the drafter is about 3.1 to 3.5% on both requests, for all K.

### 13.12 Decision table (plan step 3)

| Finding | Fires? | Numbers |
|---|---|---|
| best fixed K gives 1.2x or more | yes, go to step 4 (draft length rule) | 32k: K=2 1.38, K=3 1.25 (undisturbed), K=5 1.26, K=7 1.26. 76k: K=7 1.37, K=5 1.32, K=3 1.28. K=1 gives 1.15 and 1.09 |
| M4 shows 2 points or more for the full block | no | at most 1.9 points, mixed sign, different texts |
| draft is more than 10% of `t_round` at the best K | yes for K=5 and K=7 (step 6, ask the user first) | K=7 13.6% (32k) and 12.7% (76k), K=5 12.3% and 11.5%. K=2 and K=3 sit at 9.5 to 10.1%, on the line. Draft ms: 1.6 (K=1) to 4.0 and 4.2 (K=7) |
| `c(K)` far above the Strata row and speedup below 1.1x | no | c is at or below the Strata row except 76k K=1 (1.60 against 1.51) and 76k K=3 (2.03 against 1.96). Speedup is below 1.1 only at 76k K=1 (1.09) |
| acceptance at position 1 below 50% on the code prompt | no | position 1 is 0.68 to 0.75 on both requests (no code prompt run) |
| prefill loss above 3% | yes, narrowly | 3.1 to 3.5% on both requests. Candidates: a larger `--spec-draft-n-ubatch`, reading `h_nextn` only for the last 9 rows of a chunk |

Other findings. On 76k the speedup grows with K (1.09 to 1.37). On 32k it is flat above K=2, but the 32k texts differ per K. The verify cost per round is nearly flat (K=1: 17 to 20 ms, K=7: 26 to 29 ms), so more K pays while acceptance holds; positions 6 and 7 accept 6 to 17% and add little. The probabilistic draft mode is 4 to 5 times slower than the baseline (13.8).

### 13.13 Memory traffic per round against the bandwidth floor

Question: how close is a round to "read the memory it needs once"? GPU RTX 5090, 1792 GB/s peak. Sizes from the target GGUF (12.68 GiB weights).

| Item | Size | Read per forward pass |
|---|---|---|
| attention + router weights of one layer | 26.4 MiB | 48 visits |
| loop control weights (layers 6..17) | 9.0 MiB | 36 loop visits |
| one expert of one layer (gate_up IQ3_XXS + down IQ4_NL) | 3.79 MiB | 8 per token per visit |
| lm head (Q5_K) | 304 MiB | once |
| global KV, 8 layer visits, f16 | 32 KiB per context token | 1.0 GiB at 32k, 2.3 GiB at 76k |
| SWA KV, 40 layer visits, window 1024 | 160 MiB | once |

No shared expert. Every visit reads its weights again from VRAM: between two visits of a loop layer about 700 MiB of other weights pass, the 96 MB L2 cannot keep them.
For a verify batch of M tokens a visit touches `128 * (1 - (15/16)^M)` distinct experts if routing were random (8, 29, 51 for M = 1, 4, 8). Real routing is correlated, so the byte counts below are an upper bound.

| Tokens per pass | Bytes read (32k) | Time at peak bandwidth | Measured (13.5) | Share of peak |
|---|---|---|---|---|
| 1 (plain decode) | 4.7 GB | 2.6 ms | 12.4 ms | 21% |
| 1, 76k | 6.2 GB | 3.4 ms | 13.5 ms | 26% |
| 4 (K=3 verify) | 8.7 GB | 4.9 ms | 21.6 ms | 23% |
| 8 (K=7 verify) | 13.0 GB | 7.2 ms | 25.5 ms | 28% |
| all experts, loop layers 3 times (upper bound) | about 26 GB | 14.5 ms | - | - |

Findings.
- The plain decode runs 4 to 5 times above its bandwidth floor. From 32k to 76k the KV read grows by 1.4 GB and the time by 1.1 ms, so attention runs close to the bandwidth; the rest (about 11 ms for 3.5 GB of weights, about 330 GB/s) looks bound by latency and launches (48 visits of small matrices, n_embd 2304). Not profiled yet.
- Strata without the drafter is at the same level (89.3 t/s at 32k = 11.2 ms per token). Its advantage is the drafter and the draft policy, not the target pass.
- The verify cost per extra token is small (c in 13.5 is at or below the Strata model). The draft is the cheapest part to fix: 4.0 ms of a 29.5 ms round at K=7, about 9 times the 0.45 ms that Strata claims. A rough floor for this drafter (3 BF16 layers, 7 Markov steps over the full vocabulary) is about 1.5 ms (estimate, not checked).

## 14. Drafter cost: llama.cpp against Strata (plan step 6 diagnosis, 2026-11-24)

Diagnosis only, no code change kept. Setup as in section 13: 32k request, K=7 and K=3, BF16 drafter, `-lv 4`.
llama.cpp side: nsys (`--trace=cuda,nvtx`, osrt does not exist on Windows) with temporary NVTX ranges and stderr timers in `common/speculative.cpp` and `src/llama-context.cpp`, reverted afterwards.
Strata side: nsys of the real engine (SQ-15, `--drafter dspark`, same request). The trace shows the greedy beam-tree draft, 33 drafts, always the full block 8, so its cost does not depend on K.
nsys captured the whole session. Kernel durations are reliable. Host times come from the runs without nsys.

### 14.1 One llama.cpp draft (K=7, K=3 in brackets)

| Stage | ms | Detail |
|---|---|---|
| Host batch build | 0.006 | f_rnn row from the host ring, 8 rows x 2304 f32 = 73,728 B H2D |
| llama_decode glue | about 0.25 | remainder of decode_call |
| ggml graph build | 0.100 (0.063) | rebuilt every draft, 647 nodes (355) |
| sched_alloc_graph | 0.172 (0.105) | re-allocated every draft |
| set_inputs | 0.078 | 18 small H2D copies, most followed by a stream sync |
| compute call (2 cudaGraphLaunch) | 0.46 (0.31) | 2 splits |
| Host before the first kernel | about 1.05 (decode_call 1.07, K=3 0.74) | GPU idles meanwhile |
| GPU backbone, 3 layers | 0.557 | L0 0.17, L1 0.23 (global flash-attn 0.088), L2 0.16 |
| GPU lm head Q5_K, 8 rows | 0.200 | 319 MB, 1.5 TB/s |
| GPU 7 Markov steps | 2.081 | 0.297 per step |
| - w2 matmul, BF16 [256 x 200058], ncols=1 | 1.274 | 0.182 per step, 0.56 TB/s |
| - argmax over 200058 | 0.212 | 0.030 per step |
| - draw (top_k 64, argsort, softmax, cumsum) | about 0.24 | built in greedy mode too, nothing reads it |
| - joint_proj | 0.047 | |
| - about 45 tiny dependent kernels per step | 0.316 | about 1 us each |
| GPU backend top-10 sampler, 8 rows | 0.27 | 8 x 33 us, serial top_k_radix, anchor row unused |
| GPU total | 3.13 (K=3 1.81) | 512 kernels, gaps 0.1 |
| D2H | 17 small copies | 73,728 B (h_nextn) + 16 x 40 B. The 8 x 200058 logits stay on the GPU |
| Syncs | 93 per draft | one long wait of 3.09 ms at the end |
| Measured draft | 4.1 (K=3 2.5) | host 1.07 + wait 3.10 (0.74 + 1.80) |

Per round, outside dur(g): the injection decode on the draft context. K+1 rows, 53 nodes, rebuilt every time (build 0.025, alloc 0.029, call 0.105 ms), then 73,728 B D2H and a sync of about 0.13 ms. About 0.3 ms per round, counted in verify_ms of 13.5.

### 14.2 Graph reuse and CUDA graphs per context

- Target context: the 12,622-node graph is reused every round (140 of 146 ubatches). The 6 rebuilds are at session start and end. One rebuild costs 1.9 ms build and 3.3 ms alloc on CUDA, and the first compute takes 55 to 60 ms.
- Draft context: 2 llama_decode calls per round (draft block and injection), 0 reuse out of about 290. The two graphs alternate in the single `gf_res_prev` slot, so `can_reuse` never matches. Cost per round: about 0.33 ms of build and alloc.
- CUDA graph capture is active on both contexts. Per draft there are 2 cudaGraphLaunch (1 for the injection) and about 0 cudaGraphInstantiate: ggml-cuda sees identical node properties after each rebuild and relaunches the same CUDA graph. The launch of about 650 nodes costs about 0.4 ms of CPU.
- `GGML_CUDA_GRAPH_DEBUG` printed nothing at `-lv 4`. The evidence is the launch and instantiate counts in the trace.

### 14.3 Strata draft (nsys, 32k, beam tree)

| Stage | ms | Detail |
|---|---|---|
| Host | about 0.1 | 3 pinned H2D (4, 4, 32 B), 1 cudaGraphLaunch (0.12 ms CPU), 5 tiny D2H, 1 sync. GPU idle 0.084 before, 0.148 after. No graph build, f_rnn ring on the device |
| GEMMs (cuBLAS BF16, fused QKV and gate/up) | 0.478 | |
| Block attention layers 0 and 2 (window 1025) | 0.125 each | |
| Block attention layer 1 (global, 32k) | 3.272 | `k_dspark_block_attn`, 24 x 8 CTAs each re-read the whole K/V |
| int8 dp4a screen of the 7 slots (230 MB) | 0.316 | 0.73 TB/s |
| Top-512 candidates | 0.024 | |
| 7 Markov steps over 512 candidates (W=4 beam) | about 0.32 | expand 0.129, int8 score 0.044, prep 0.031 |
| Draft total | 4.67 | 1.13 without the block attention. Draft to draft period 27.9 ms median, so about 17% of a round |

The Strata claim of below 0.45 ms is not reproduced. At 32k the draft is 4.67 ms, and 1.13 ms without its attention. The claim is also below the floor of its own parts: the GEMMs alone are 0.48 ms and the screen 0.32 ms.

### 14.4 Side by side (32k, block 8 or K=7)

| Stage | llama.cpp ms | Strata ms | Bytes read, llama.cpp / Strata | Floor at 1.79 TB/s, llama.cpp / Strata |
|---|---|---|---|---|
| Host before GPU | 1.05 | about 0.1 | - | - |
| Backbone weights | 0.38 | 0.58 | 496 MB BF16 / 496 MB BF16 | 0.28 / 0.28 |
| Attention over the context | 0.09 | 3.56 | 142 MB / 142 MB | 0.08 / 0.08 |
| lm head or base logits | 0.200 | 0.34 | 319 MB Q5_K / 230 MB int8 | 0.18 / 0.13 |
| Markov chain, 7 steps | 2.08 | 0.32 | 717 MB w2 + 55 MB / about 1 MB per step (candidates, L2) | 0.43 / 0.02 |
| Draw and argmax over 200k | 0.45 (inside the 2.08) | 0 | - | - |
| Backend top-10 on 8 rows | 0.27 | 0 | - | - |
| GPU total | 3.13 | 4.67 (1.13 without attention) | 1.73 GB / 0.9 GB | 0.98 / 0.50 |

Differences: Strata scores only the top-512 candidates (approximate), draws nothing, takes the drafted ids from the chain, keeps one persistent CUDA graph and keeps f_rnn on the device. Its attention kernel is 37 times slower than ours on the global layer, so there is nothing to copy there.

### 14.5 Ranked fix candidates (ms per draft at K=7, K=3 is about 40 to 50 percent; estimates, not measured)

Correction to the first report: the Q8_0 drafter file does NOT quantize `markov_w2`. `ginfo.py` shows `markov_w2.weight`, `markov_w1.weight` and `joint_proj.weight` as BF16 in the Q8_0 file, only the 3 layers are Q8_0. That is why 13.9 saw only 3.88 against 4.00 ms: it saves only the backbone bytes.

| # | Fix | Saves | Where | Touches ggml |
|---|---|---|---|---|
| 1 | Top-512 candidate scoring in the Markov head (`ggml_top_k`, `get_rows` on w2 rows and bias, small `mul_mat`, argmax over 512). Approximate drafts, remeasure acceptance | about 1.3 net | `src/models/dspark.cpp` | no |
| 2 | Skip `draw` unless drafting is probabilistic (flag from the driver) | about 0.40 | `src/models/dspark.cpp`, `common/speculative.cpp` | no |
| 3 | Read the drafted chain tokens the graph already has, drop the backend top-10 on 8 rows | about 0.33 | `common/speculative.cpp`, `src/models/dspark.cpp` | no |
| 4 | Reuse the draft-context graph: two result slots (draft block, injection) in llama_context | about 0.33 per round | `src/llama-context.cpp` | no |
| 5 | Keep f_rnn on the device (existing h_nextn_dev path), no 73,728 B D2H and H2D | about 0.15 to 0.2 per round | `src/llama-context.cpp`, `common/speculative.cpp` | no |
| 6 | Quantize `markov_w2` to Q8_0 in the drafter GGUF (converter or quantize override; the BF16 file folds w2 in fp64, requantizing may cost acceptance). ncols=1 `mul_mat_vec_q` should reach about 1.2 TB/s. Alternative to 1 | about 0.9 | converter, no graph change | no |
| 7 | Q8_0 backbone (already in the Q8_0 file) | 0.1 to 0.3 (measured 0.12 in 13.9) | none | no |
| 8 | Fuse the tiny Markov kernels | about 0.2 to 0.3 | ggml-cuda | yes |
| 9 | Faster BF16 `mul_mat_vec_f` for ncols=1 (0.56 to about 1.4 TB/s) | about 0.75 if w2 stays BF16 | ggml-cuda | yes |
| 10 | Async set_inputs without a sync per tensor | about 0.05 | backend input path | yes |

Fixes 1 to 4 together should bring the draft from about 4.1 to about 1.6 to 1.8 ms. Fixes 1 and 6 overlap, take one. Proposed measurement, not run: a K=7 and K=3 session after fix 6 (or 1) for time and acceptance.

### 14.6 Files

All under `tmp/inkling/` (git-ignored).
- `logs/dspark-s6/prof-k7.server.log`, `prof-k3.server.log`: timing runs with UB lines (build, alloc, set_inputs, compute call per decode) and DSPARK_PROF lines.
- `logs/dspark-s6/nsys-k7.nsys-rep` and `.sqlite`, `nsys-k3.nsys-rep` and `.sqlite`, `strata-nsys.nsys-rep` and `.sqlite`: traces.
- `logs/dspark-s6/draft-kernel-seq.txt`, `strata-draft-kernel-seq.txt`: ordered kernels of one draft.
- `tools/dspark_s6_nsys.sh`, `dspark_s6_strata.sh`, `dspark_s6_an.py`: session scripts and trace analysis (its host-window lines are labelled ms but are us).

## 15. Drafter speed fixes (2026-11-24)

Fixes from the list in 14.5, each measured alone. 32k and 76k requests, K=7, request sampling, greedy draft, `-c 98304 -ub 512`. Logs: `tmp/inkling/logs/dspark-s7/`, `dspark-s8/`.

| Change | Exact | 32k draft ms | 32k decode t/s | 76k draft ms | 76k decode t/s |
|---|---|---|---|---|---|
| PR #32218 as is (13.5) | - | 4.01 | 101.3 | 4.24 | 101.5 |
| F1: no `draw` subgraph in greedy mode | yes | 3.37 | 102.9 | - | - |
| F1 + F2: drafted tokens from the argmax chain (packed into `h_nextn`), no logits, no backend sampler | yes | 3.12 | 103.6 | 3.34 | 103.4 |
| F1 + F2 + `markov_w2` in Q8_0 (new drafter file, no code) | no, acceptance equal to 3 decimals | 2.06 | 107.7 | 2.29 | 107.2 |

- Exact means: same generated token count (1684 on 32k) and the same `acc rate/pos` as before. Teacher forcing with F1 + F2 (step 2 run 1): 298 of 300, the same two misses as before. `--spec-draft-sampling probabilistic` keeps the old path (18.8 t/s, as in 13.8).
- Drafter file with Q8_0 `markov_w2`: `dspark-Inkling-2-Mini-BF16-w2q8.gguf` (657 MiB), made with `llama-quantize --pure --allow-requantize --tensor-type 'markov_w2\.weight=q8_0' --tensor-type '\.weight$=bf16' <BF16-mainline.gguf> <out> Q8_0`. Only `markov_w2.weight` changes type. Acceptance at 76k: (0.706, 0.506, 0.364, 0.286, 0.228, 0.166, 0.125) against (0.704, 0.504, 0.364, 0.288, 0.230, 0.166, 0.125).
- Tried and dropped: scoring the Markov steps only on the top 512 candidates of the base logits (Strata `K_cand`). Draft 1.75 ms, but acceptance falls from position 2 on (32k mean acc len 2.55 against 3.01), decode 94.3 (32k) and 86 t/s (76k). Patch kept in `tmp/inkling/patches/dspark-f1f2f5.diff`.
- Not done: F3 (reuse of the two draft-context graphs, about 0.3 ms per round): `gf_res_prev` already has two slots, but one scheduler serves both graphs, and allocating one invalidates the other; real reuse needs a second scheduler in `llama_context`. Cheaper option: merge the injection into the block decode as one mixed batch. F4 (`f_rnn` on the device, about 0.15 ms): needs a mixed state input in `llama-graph.cpp`.
- Against Strata (121.9 / 114.4 t/s): 32k -12%, 76k -6% at fixed K=7. Next: windowed K (plan step 4).

## 16. Windowed draft length (plan step 4), 2026-11-24: implemented, loses, removed

Design: plan step 4, "windowed K". Per sequence a window of 64 rounds, `a = n_acc / (n_acc + n_rej)`, `K = argmax E(K) / c(K)`, switch only above +3%, the draft block is K + 1 tokens. Env `LLAMA_SPEC_DRAFT_COST`, `LLAMA_SPEC_DRAFT_WINDOW`; unset = off. About 100 lines in `common_speculative_impl_draft_dspark` (uses the existing `accept()` callback). Patch: `tmp/inkling/patches/dspark-windowed-k.diff`. Not in the tree. Logs: `tmp/inkling/logs/dspark-s9/`.

Cost table from synthetic acceptance (`--spec-synth-len K+1`, 32k request, tools stripped, `ignore_eos`, w2q8 drafter, text not valid output; single-token decode 80.76 t/s = 12.38 ms):

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| round ms | 18.4 | 20.5 | 25.2 | 26.7 | 29.4 | 30.4 | 33.1 |
| draft ms | 1.29 | 1.49 | 1.70 | 1.83 | 2.03 | 2.09 | 2.33 |
| c (synthetic) | 1.48 | 1.66 | 2.03 | 2.16 | 2.37 | 2.46 | 2.68 |
| c (real acceptance, verify of 13.5 + draft above) | 1.47 | 1.62 | 1.88 | - | 2.15 | - | 2.25 |

Gate: teacher forcing step 2 run 1 with the rule on (window 16): 300 of 300. Speed, `--spec-draft-n-max 7`, w2q8 drafter, request sampling:

| Config | 32k t/s | 76k t/s | Mean acc len (32k / 76k) |
|---|---|---|---|
| fixed K=7 (section 15) | 107.7 | 107.2 | 3.01 / 3.38 |
| fixed K=2 | 114.0 | not run | 2.30 / - |
| windowed K | 108.0 | 96.2 | 2.48 / 2.39 |

K sequence, the same in both requests and reps: 3 -> 2 -> 6 -> 2 -> 6 -> 2.

Why it loses.
- The synthetic table makes long drafts look too expensive: c(7) is 2.68 synthetic against 2.25 with real acceptance. A synthetic round commits 8 tokens, a real one about 3, so each committed token costs about 1 ms more than a verified one ((33.1 - 27.8) ms / 5 tokens). The rule therefore goes to small K too often (76k: 96 against 107 t/s).
- Even with the real costs the choice at 32k is a tie: a = 0.68 gives E/c = 1.32 for both K = 2 and K = 7. A window of 64 rounds measures a to about +-0.05, so the rule flips between K = 2 and K = 6; the 3% hysteresis is far below that noise.
- Upper bound of any length rule on these two requests: +6% on 32k (fixed K = 2 against K = 7), about 0 on 76k (K = 7 is best). A fixed K gets most of it.

Open finding: about 1 ms per committed token outside the verify graph. Not measured directly. Candidate: the server samples every verified position on the host (copy of 200k logits per position, softmax and sampling over the full vocabulary on the CPU). With about 3 committed tokens per round that is about 3 ms of a 28 ms round, and plain decode pays it once per token. If confirmed, backend sampling for the target (the drafter already samples on the device) would remove most of it. Revisit the rule after that, with costs from real rounds.

## 17. Where a target pass spends its GPU time (nsys, 2026-11-24)

Trace `tmp/inkling/logs/dspark-s6/nsys-k7.sqlite` (32k request, K=7, before the drafter fixes; target unchanged since). Verify passes of 8 tokens, median of 76 passes. Kernel times under nsys, which inflates very short kernels, so the small-kernel total is an upper bound. Busy time (25.1 ms) is above the span (22.6 ms) because some graph nodes overlap.

| Group | ms per pass | Kernels per pass | Note |
|---|---|---|---|
| expert matmuls (`mul_mat_vec_q_moe`, IQ3_XXS / IQ4_NL, q8_1 activations, int8 dot products) | 5.0 | 96 | 52 us each, close to the bandwidth |
| attention (`flash_attn_ext_f16`, 8 global + 40 SWA visits) | 3.1 | 48 | the global layers read about 1 GiB, 0.6 ms at peak bandwidth: about 5x above the floor |
| small F32 / BF16 matmuls (`mul_mat_f`, `mul_mat_vec_f`: router, loop control, small projections) | 4.5 | 544 | 6 to 10 us each, too small to fill the GPU |
| Q6_K projections, lm head, other quantized matmuls | about 1.6 | about 275 | |
| elementwise and glue (`k_bin_bcast` 1492, `scale_f32` 819, `rms_norm_f32` 514, `unary_op_kernel` 577, concat 548, `quantize_q8_1` 293, `k_set_rows` 292, `ssm_conv_f32` 192, ...) | about 9 | about 6300 | 1.2 to 2 us each, almost all fixed cost |
| total | 22.6 (span) | 7583 | |

The arithmetic is not the limit: a token needs about 4.8 G multiply-adds (about 90 M weights per visit x 48 visits + lm head), an 8-token verify about 77 GFLOP, well below 1 ms on this GPU. The memory bandwidth is the limit only in the expert matmuls. Most of the pass is a chain of about 7600 dependent, mostly tiny kernels with a fixed cost of about 1.5 us each on the GPU (CUDA graphs remove the CPU launch cost, not this). The same chain at batch 1 is the "about 11 ms fixed" part of plain decode in 13.13 (21% of peak bandwidth). Strata (11.2 ms per token without the drafter) fuses each stage into hand-written kernels; it is also well above the bandwidth floor.

Levers, all ggml / kernel work (not in the plan so far):
- Fuse the glue chains (norm + scale + mul + add, gates, sigmoid chains) in ggml-cuda: several ms per pass, for decode and verify. A comment in ggml-org PR #32773 mentions a branch "realugbun/inkling2-speed" with fused CUDA glue ops for Inkling2: look at it first.
- BF16 instead of F32 for the router and loop control weights: fewer bytes, but these kernels are latency bound, so small.
- Flash attention with 8 queries over 32k positions is about 5x above its bandwidth floor: split the global layers across more blocks, check the positional mask and rel-bias paths.
