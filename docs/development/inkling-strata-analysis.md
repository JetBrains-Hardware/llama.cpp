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
