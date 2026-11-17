# VRAM of llama-server with the DFlash2 drafter (Qwen3.8-27B UD-IQ3_S target), 2026-11-17

Machine: RTX 5090 32 GB (32607 MiB), Ryzen 9 9950X3D, Windows 11, CUDA 12.9, driver WDDM. Idle GPU before every session: 10 MiB (nvidia-smi).
Baseline of the DFlash2 memory work: ggml-org master, ours (kvq-two-tier-cache, see `kvq-two-tier-cache.md`) and the vvsotnikov branch `perf/cuda-qwen-dflash2`. Harness, logs and raw data are outside the repo in `../kvq-scratch/mem` (the paths `mem\...` below).
All numbers are MiB (2^20 bytes). "measured" = 5 Hz time series of the process, "log" = allocation line of the server log (`-lv 4`), "computed" = formula from the code, "diff" = difference between two measured runs.

## Builds

| Name in this file | Path | Commit (from `--version`) |
|---|---|---|
| ours with --kvq | llama.cpp\build-cuda (rebuilt at HEAD, only the UI and server objects relinked) | ours kvq-two-tier-cache f06af3fe1, run with `--kvq` |
| ours plain (control) | same binary | ours kvq-two-tier-cache f06af3fe1, without `--kvq` |
| ggml-org master | wt-up-master\build-cuda | ggml-org master a49db739f |
| ggml-org 951d (control) | wt-up-951d\build-cuda | ggml-org 951d465f5 (base of our branch) |
| ggml-org 259d (control) | wt-up-259d\build-cuda | ggml-org 259d91709 (base of the vvsotnikov branch) |
| vvsotnikov V1-ship | wt-vv-dflash2\build-cuda | vvsotnikov perf/cuda-qwen-dflash2 5ce463333, no environment variables, checkpoints in VRAM as shipped |
| vvsotnikov V2-ship | same | V1-ship with `LLAMA_GDN_REPLAY=0` |
| vvsotnikov V3-ship | same | the recommended environment and flags, n_max 15 (below) |

Not done: vvsotnikov V1 / V2 / V3 with `LLAMA_CKPT_HOST=1` (checkpoints in host memory). The one-line patch of `wt-vv-dflash2\tools\server\server-context.cpp` (create_checkpoint, `device_storage` false when `LLAMA_CKPT_HOST` is set) was not applied, so the tree is unchanged and not rebuilt. The VRAM of the patched build can be computed from the V1-ship runs: V1-ship minus its device checkpoint rows (exact, see the table), because nothing else in the code path depends on where a checkpoint lives (the checkpoint copy is made after the decode).

Every binary knows every flag of the launch line (checked in `--help`, `mem\dfl\bins\*.flags.txt`).

## Launch line (same for every build)

```
llama-server --fit off -m Qwen3.8-27B-UD-IQ3_S.gguf -md incoai\Qwen3.8-27B-DFlash2-Q4_K_M.gguf --ctx-size N --n-gpu-layers all -ngld all --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja --spec-type draft-dflash --spec-draft-n-max 7 -ctkd f16 -ctvd f16 --batch-size 4096 --ubatch-size 1024 --host 127.0.0.1 --port P -lv 4 [--kvq]
```

DFlash2 settings: the documented defaults. Draft sampling `greedy` (default of `--spec-draft-sampling`), draft backend sampling on by default (not used by DFlash2: `common/speculative.cpp` builds no backend chain when `is_dflash2`), draft K/V f16 (default, written explicitly as `-ctkd f16 -ctvd f16`), draft context = the target n_ctx (no option; `common_speculative_init_result` sets `cparams.n_ctx = llama_n_ctx(ctx_tgt)`), draft ubatch = `--ubatch-size` (`--spec-draft-n-ubatch 0`). The drafter's block size is 8 (`dflash.block_size`), so n_max 7 is the largest chain. Server defaults kept: `--ctx-checkpoints` 32, min spacing 8192, `--cache-ram` 8192, `cache_prompt` true in every request. `-lv 4` is only for the allocation lines.

V3-ship: environment `LLAMA_DFLASH_TREE=1 GGML_CUDA_TREE_SLOTS=1 GGML_CUDA_ASYNC_INPUTS=1 LLAMA_DFLASH_ASYNC_INJECT=1 GGML_CUDA_GLU_MMQ_FUSION=1 GGML_CUDA_IQ4_FRAGMENT=1` and on top of the common line `--spec-draft-n-max 15 --kv-unified --spec-draft-sampling probabilistic --spec-draft-backend-sampling --backend-sampling --cache-prompt --cache-ram 0 --no-cache-idle-slots`, with our batch sizes (-b 4096 -ub 1024) and q8_0 K/V. The tree ran with these batch sizes ("DFlash2 trees enabled: budget=15, block=16").

## Session (memory runs)

One server process per session, inside one GPU lock hold: idle baseline (nvidia-smi, 5 samples), start, `/health`, 5 s idle, r1 = the first P tokens of a token file as token ids, `n_predict` 300, temperature 0, streamed; 8 s idle; r2 = r1 prompt + r1 reply + the first 500 ids of row310.bin, 300 tokens; 8 s idle; kill. P = 99,395 (row2660.bin) or 49,737 (row1830.bin). Points: n_ctx 51200 / 102400 / 204800 with the 50k prompt, 102400 / 204800 with the 100k prompt.
Sampling: per-process counters `\GPU Process Memory(pid_N*)\Dedicated Usage` and `Shared Usage`, private bytes and working set of the process, every 200 ms (median gap 204 ms, largest gap 0.33 s except one 4.4 s gap during the load of one kvq session, while the pinned store was allocated); nvidia-smi `memory.used` every 200 ms. Phases from the client's event times: load (after /health), prefill (request sent to first streamed token), generation (first to last token), after the request, and the same for r2.

Dedicated usage and nvidia-smi minus the idle baseline agree within 2 MiB for every build without --kvq. With --kvq the dedicated counter is 5 / 8.5 / 15.6 MiB above nvidia-smi at n_ctx 51200 / 102400 / 204800, about 2 MiB per GiB of pinned host memory of the process (best guess: GPU page tables for the mapped pinned store, booked to the process but not to the device total). Shared usage did not grow during any session beyond the pinned buffers the logs announce (no spill of VRAM to system RAM).

The scripts (`mem\dfl\sess.sh BIN ...`, `runmatrix.sh`, `matrix.txt`, `py\analyze.py`, `py\components.py`, `py\remainder.py`) take the binary, the environment and the flags as parameters. Raw data per session: `mem\dfl\runs\TAG\`.

## Main table: VRAM by component at n_ctx 102400, prompt 99,395 tokens

Values are MiB. One value per row where the component does not change between prefill and generation (all ggml buffers are allocated at load or at the first request). "pf" = peak during the r1 prefill, "gen" = peak during generation (generation is steady: max - min <= 6 MiB in every session).
Class: FIXED, LINEAR (in n_ctx, allocated at load), TOKENS (grows with the cached tokens). Slopes are per 1000 tokens of n_ctx or of cached tokens.

| Context | Component | ours with --kvq | ggml-org master (= ours plain = 951d = 259d) | vvsotnikov V1-ship | Class, slope | How we know |
|---|---|---|---|---|---|---|
| target | weights on the GPU | 10616.9 | 10616.9 | 10616.9 | FIXED | log `load_tensors: CUDA0 model buffer`; the token embedding (521.0) is CPU_Mapped, the MTP block 64 is not loaded ("unused tensor ... ignoring") |
| target | K/V cache q8_0 (16 layers) | 144.5 (the ring, 4352 cells) | 3400.0 | 3400.0 | plain: LINEAR 33.20/1k; ring: FIXED (max(n_batch,4096)+256 cells) | log `llama_kv_cache`; computed n_ctx x 16 x 2 x 4 x 256 x 34/32 B = 34,816 B per cell |
| target | kvq coarse codes | 625.0 | - | - | LINEAR 6.10/1k (6400 B per row) | log `llama_kvq_store: device` |
| target | kvq tables | 108.9 | - | - | LINEAR, piecewise: 8.375 per 8192 rows (ceil(n_ctx/8192) slots), 1.02/1k | log |
| target | kvq scratch (prefill attention) | 221.5 | - | - | LINEAR 2.07/1k (2176 B per row, one layer q8_0) + 9 | log; 115.3 / 221.5 / 434.0 at the three n_ctx |
| target | kvq resident set + other | 34.5 + 16.8 | - | - | FIXED | log |
| target | kvq fitter | 83.2 | - | - | FIXED (one 8192-row stage) | log `llama_kvq_fitter` |
| target | kvq device memory not in any log line | about 85 | - | - | FIXED | diff: load remainder 589.5 with --kvq against 504.2 without, the same 85 in the runs without drafter |
| target | recurrent state: 1 cell + 7 rollback copies (n_rs_seq 7) | 1197.0 | 1197.0 | 213.3 (GDN replay) | FIXED, in proportion to 1 + n_max: 149.6 per copy (R 5.6 + S 144.0) | log `llama_memory_recurrent`; without drafter (n_rs_seq 0) 149.6 |
| target | attention rotation | - | 4.25 | 4.25 | FIXED | log `init_rotation` (off with --kvq) |
| target | compute buffer | 909.5 (801.5 at load) | 880.05 (800.04 at load) | 880.05 | LINEAR 5.86/1k: kvq 309.5 + 5.86/1k, plain 280.05 + 5.86/1k | log `sched_reserve`; re-reserved at the first request when the layer input copies join the graph (+80 plain, +108 kvq) |
| shared | layer input buffer (target hidden states of 5 layers for the drafter) | 400.0 | 400.0 | 400.0 | FIXED, in proportion to n_batch: 5 x 5120 x n_batch x 4 B | log `layer_inp_dev_alloc`; owned by the target context, read by the drafter |
| draft | drafter weights | 1079.6 | 1079.6 | 1079.6 | FIXED | log; equals the tensor sum of the GGUF |
| draft | drafter K/V (5 SWA layers, f16) | 60.0 | 60.0 | 60.0 | FIXED: (n_swa 2048 + draft ubatch) cells x 20,480 B | log `llama_kv_cache_iswa: SWA KV cache, size = 3072 cells`; the non-SWA part has 0 layers |
| draft | drafter compute buffer | 1098.6 | 1098.6 | 1098.6 | FIXED, in proportion to the draft ubatch (137.5 with `--spec-draft-n-ubatch 128`, measured) | log; cause below |
| shared | prompt checkpoints in VRAM | 0 (host) | 0 (host) | pf: 189.6 (1); gen r1: 379.3 (2); after r2: 761.5 (4) | TOKENS: one per 8192 tokens of a prompt plus one near each prompt end, 189.6 each (149.6 target + 40.0 drafter), budget 1 GiB | log `finish: allocated 'CUDA0' buffer 149.625 / 40.000 MiB` per checkpoint |
| - | CUDA context at process start | 431.6 | 431.6 | 431.6 | FIXED | measured: the first sample of every process, before the first buffer (also 74 MiB shared) |
| - | other load-time device memory not in a log line | 157.9 (73 + 85 kvq row above) | 72.7 | 72.4 | FIXED | diff (load measured minus logged buffers); best guess: cuBLAS/cuBLASLt handle and workspace, 2 MiB rounding of about 15 buffers |
| - | runtime growth during the first prefill | 56.1 | 56.0 | 56.0 | FIXED (one time, the same at 50k and 100k) | diff (prefill peak minus load minus the compute re-reserve); best guess: ggml CUDA memory pool and lazily loaded kernels |
| - | runtime growth at the first generation | 74 (+2 in r2) | 48 | 48 | FIXED (one time) | diff; CUDA graphs are about 6 of it (measured: 19338.6 instead of 19344.6 with GGML_CUDA_DISABLE_GRAPHS=1), the rest pool and kernels of the verify and draft graphs |
| | sum of the logged rows (after r2) | 16595.9 | 18736.3 | 18514.1 incl. 761.5 checkpoints | | |
| | measured, after load | 17077.4 | 19160.6 | 18176.6 | | dedicated usage; nvidia-smi minus baseline 17069 / 19159 / 18175 |
| | measured, prefill peak | 17241.5 | 19296.6 (master 19292.6) | 18502.6 | | |
| | measured, generation peak = steady | 17315.5 | 19344.6 (master 19340.6) | 18740.6 | | |
| | measured, after r1 | 17315.5 | 19344.6 | 18740.6 | | |
| | measured, after r2 (follow-up of 801 new tokens) | 17317.5 | 19344.6 (master 19342.6) | 19124.6 | | |
| | unattributed remainder after r2 | 721.6 | 608.3 | 610.5 | | = the four "-" rows (431.6 + 72.7 + 56 + 48 = 608.3 for plain) |

Split by context at 102400 after r2 (logged rows only, remainder separate):
- ours with --kvq: target 14060.6 (weights 10616.9, ring 144.5, kvq 1089.9, recurrent 1197.0, compute 909.5), shared 400.0, draft 2238.2, remainder 721.6 (incl. the kvq 85).
- ggml-org master: target 16098.2 (weights 10616.9, KV 3400.0, recurrent 1197.0, rotation 4.25, compute 880.05), shared 400.0, draft 2238.2, remainder 608.3.
- vvsotnikov V1-ship: target 15114.5, shared 400.0 + checkpoints 761.5, draft 2238.2, remainder 610.5.

What DFlash2 itself costs (measured as the difference to the same binary with no drafter, `--spec-type` none, no `-md`, 102400, 100k prompt):
- plain: 19344.6 - 15502.3 = 3842.3 MiB = drafter weights 1079.6 + drafter compute 1098.6 + rollback copies of the recurrent state 1047.4 (1197.0 - 149.6) + layer input buffer 400.0 + drafter K/V 60.0 + the larger target compute buffer 80.0 + runtime 76.8 (remainder 608.3 against 531.5).
- with --kvq: 17317.5 - 13421.2 = 3896.3 MiB, the same rows with the compute re-reserve +108 and the runtime +102.7.

## Classes and slopes (measured at 3 n_ctx and 2 prompt lengths)

The two prompt lengths at the same n_ctx differ by at most 2 MiB (master, ours plain, vvsotnikov, at every n_ctx) and by 9.9 MiB with --kvq (generation and after the requests). So in VRAM nothing grows with the cached tokens except the vvsotnikov device checkpoints and about 0.2 MiB per 1000 tokens with --kvq (cause not identified; the packed attention over more coarse rows). Everything else is FIXED or LINEAR IN n_ctx and allocated at load or at the first request.

| Build | after load (51200 / 102400 / 204800) | after r2 at 50k prompt | after r2 at 100k prompt (102400 / 204800) |
|---|---|---|---|
| ours with --kvq | 16303.8 / 17077.4 / 18622.4 | 16534.1 / 17307.7 / 18852.7 | 17317.5 / 18862.6 |
| ours plain | 17160.3 / 19160.6 / 23161.0 | 17342.4 / 19342.6 / 23343.0 | 19344.6 / 23345.0 |
| ggml-org master | 17160.3 / 19160.6 / 23161.0 | 17340.4 / 19340.6 / 23341.0 | 19342.6 / 23343.0 |
| ggml-org 951d / 259d | - / 19160.6 / 23161.0 | - | 19344.6 / 23345.0 (both) |
| vvsotnikov V1-ship | 16176.3 / 18176.6 / 22177.0 | 17122.4 / 19122.6 / 23123.0 | 19124.6 / 23125.0 |
| vvsotnikov V2-ship | - / 19160.6 / - | - | 20104.6 / - |
| vvsotnikov V3-ship | - / 18246.6 / 22247.0 | - | 21708.7 / 25709.1 |

Slopes in n_ctx (from the components, they reproduce the totals): plain 39.06 MiB per 1000 (KV 33.20 + compute 5.86); --kvq 14.11 MiB per 1000 plus the table steps (codes 6.10 + scratch 2.07 + compute 5.86 + tables 8.375 per 8192 rows; fitted slope of the totals 14.11 without the tables). Piecewise: the kvq tables in steps of 8.375 MiB per 8192 positions of n_ctx; the ring and the layer input buffer in steps of n_batch; the drafter K/V and compute in steps of the draft ubatch. Note the scale: going from 102400 to 204800 costs 4000 MiB without --kvq and 1545 MiB with it.

## Formulas VRAM(n_ctx, T) and their error

n = n_ctx / 1000, T = cached tokens / 1000 at the end of the session (50.8 or 100.5), value = dedicated usage after the follow-up request (= the session peak, except where noted).

- ggml-org master, ours plain, 951d, 259d: VRAM = 15344.6 + 39.0625 n. Error at the 5 + 5 + 2 + 2 points: -0.4 to +4.2 MiB (the model is high by up to 4 MiB on master). After load: 15160.6 + 39.0625 n (error under 1 MiB). Prefill peak: load + 130 to 136.
- ours with --kvq: VRAM = 15753.6 + 14.11 n + 8.375 ceil(n_ctx/8192) + 0.2 (T - 50.8). Error at the 5 points: within +-0.8 MiB (fitted; the component sum with a constant remainder of 715 is high by 15 to 26 MiB because the remainder grows by about 0.07 MiB per 1000 n_ctx). After load: 15523.4 + 14.11 n + 8.375 ceil(n_ctx/8192) (error within 0.8 MiB).
- vvsotnikov V1-ship: VRAM = master - 983.7 + 190.4 k, k = number of device checkpoints (4 in our session: two at the end of r1's prompt, two in r2), up to the 1 GiB budget (5 checkpoints), then pinned host. Error at the 5 points: -2.6 to 0 MiB. The same without device checkpoints (the patched V1): master - 983.7 (computed).
- vvsotnikov V2-ship (LLAMA_GDN_REPLAY=0): load identical to ggml-org 259d (19160.6); after r2 = 259d + 760 MiB of device checkpoints (measured 20104.6 against 19344.6). So the vvsotnikov memory differences against its base are exactly the GDN replay (-983.7) and the device checkpoints.
- vvsotnikov V3-ship: V1-ship + 69.3 (recurrent state with a 16-token window: 282.6 instead of 213.3) + 2470.6 (IQ4_XS fragment copies, 86 of 86 tensors) + about 44 (tree runtime), measured 21708.7 / 25709.1 at 102400 / 204800. Free VRAM before the first request 13936 / 9936 MiB, so every fragment was made (each needs free > size + 1 GiB); the fragments appear at the first 16-token verify, not at load.

## Other observations about the components

- Drafter compute buffer, 1098.6 MiB: the token-batch graph of the drafter (`src/models/dflash.cpp`, `llama_model_dflash::graph<false>`, the branch after the injection) computes the LM head logits for every token of the batch ([248320 x n_tokens] f32) and `ggml_top_k` on them, without an output selection. The scheduler reserves it at n_tokens = n_ubatch = 1024: 248320 x 1024 x 4 B = 970 MiB of the buffer. A real token batch has n_max + 1 = 8 tokens; only the injection batches have up to n_ubatch rows, and they do not compute logits. Measured: `--spec-draft-n-ubatch 128` gives 137.5 MiB (and 45 MiB drafter K/V), the same draft and accepted counts on both requests (473 / 231, 507 / 225), prefill 42.28 s against 42.05 s (one run each).
- Recurrent rollback copies, 1047 MiB: `n_rs_seq` = 7 recurrent-state copies (one per draft position) for the verify rollback. vvsotnikov replaces them by a replay of the 8-token window (213.3 MiB incl. a 24.3 MiB stash); with n_max 15 its window is 16 tokens (282.6).
- Layer input buffer, 400 MiB: sized by n_batch (4096) although the drafter consumes it per target ubatch (1024); f32.
- Prompt checkpoints: 189.67 MiB each (target 149.6 = one recurrent cell, drafter 40.0 = the 2048-cell SWA window). ggml-org, ours (both modes): plain host memory (std::vector; no device or pinned allocation line). vvsotnikov: VRAM (149.625 and 40.000 MiB device buffers per checkpoint).
- The other on-device state copies of ggml-org (`n_rs_seq` copies) are the row "recurrent state"; no `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` snapshot was allocated in any ggml-org or ours session (no such log line, and the totals are explained without it).
- ggml-org 951d465f5 and 259d91709 and master a49db739f and ours plain have identical buffers and totals within 4 MiB: the 284 upstream commits between 259d and 951d and the newer master changed nothing in memory for this configuration; ours plain equals its base 951d to the MiB (19344.6 both), so our branch adds nothing without --kvq.

## Host memory (secondary table, n_ctx 102400, 100k prompt)

Private bytes of the process include the dedicated VRAM on WDDM (the commit of the device allocations), so "host" = private bytes - dedicated usage. Shared usage = pinned host memory mapped for the GPU (CUDA_Host buffers and the kvq store).

| | ours with --kvq | ggml-org master | vvsotnikov V1-ship | vvsotnikov V3-ship |
|---|---|---|---|---|
| private bytes after load | 22598 | 20908 | 20200 | 20324 |
| private bytes after r2 | 24129 | 22221 | 21540 | 24220 |
| host = private - dedicated, after load / after r2 | 5521 / 6812 | 1747 / 2878 | 2023 / 2415 | 2077 / 2511 |
| shared usage (pinned) after load / peak | 3942 / 4172 | 470 / 476 | 470 / 476 | 480 / 538 |
| of it, logged pinned buffers | kvq store 3400.0 (LINEAR 33.20/1k), kvq other 76.3, target host compute 288.55 (LINEAR 1.95/1k), drafter host compute 26.04, outputs 2 x 0.95 | target host compute 280.05, drafter 26.04, outputs | same as master | |
| prompt checkpoints in host memory after r2 | 4 x 189.67 = 758.7 | 4 x 189.67 = 758.7 | 0 (in VRAM) | 0 (in VRAM) |
| token embedding (CPU_Mapped, file-backed, not private) | 521.0 | 521.0 | 521.0 | 521.0 |

At 204800: kvq store 6800 (shared 7534 after load), master shared 670. The private-minus-dedicated growth from load to after r2 (1131 MiB on master) is the 4 checkpoints (759) plus the slot's token and output arrays and the prompt cache bookkeeping (the RAM prompt cache saved nothing: one slot, the second prompt extends the first).

## --kvq with DFlash2

Works, no fallback. Checked in every --kvq session:
- The kvq graph exports the layer inputs: `res->t_layer_inp[il] = inpL` is set for every layer in `src/models/qwen35.cpp` in both paths, and `build_layer_inp_dev` (llama-graph.cpp) is called for every model after the graph; the log says "DFlash target features via device buffer".
- Drafts are produced and accepted: r1 473 drafted / 231 accepted, r2 507 / 225 at 102400, identical to ours plain and to ggml-org master on the same requests.
- The verify batches use the packed attention: r1 "kvq: packed ubatches = 69 (545 tokens)" for 300 generated tokens (68 verify steps + 1), 7.9 tokens per ubatch, at most 8 = max_batch; no packed ubatch fell back to the scratch (the end-of-prompt fit makes n_host == n_coarse; "waited 0.545 s for the fitter at the end of the prompt, coarse region 99395 rows").
- Cost: generation 118.3 t/s against 143.3 (ours plain) at 100k in r1, 105.7 against 134.9 in r2. An 8-token verify requests 65,536 rows per ubatch and fetches about half (r1: 4.52 M rows requested, 2.27 M fetched, 1221.5 MiB uploaded for 69 ubatches); "per packed ubatch: join 26.2 ms" on the decode thread. Not a memory issue, but with DFlash2 the packed path is about 17 - 22% slower than the plain cache here; with the MTP drafter (3 drafts) it was faster than production.

## Reducible, largest first (VRAM at 102400)

1. Recurrent rollback copies, 1047 MiB (FIXED, 149.6 per draft position): the vvsotnikov GDN replay shows that 984 MiB of it can go (213.3 instead of 1197.0).
2. Drafter compute buffer, 961 MiB of 1098.6: the drafter's token-batch graph is reserved at n_ubatch tokens with full-vocabulary logits; real token batches have 8 tokens. `--spec-draft-n-ubatch 128` saves 961 + 15 MiB today without code changes (measured, same acceptance on 2 requests); a code fix would reserve that graph at n_outputs_max tokens.
3. Target K/V, 3400 MiB at 102400 (33.2 per 1k): the two-tier cache takes it to 144.5 + 1090 + the 85 unlogged (2027 MiB saved at 102400, 4498 at 204800).
4. Target compute buffer, 880 MiB (5.86 per 1k of n_ctx, 1480 at 204800): the flash attention F16 copies of K and V and the F16 mask at full length for a 1024-token ubatch; only prefill needs it.
5. Device prompt checkpoints of vvsotnikov, 190 MiB each up to about 1 GiB: host memory is enough (ggml-org does it).
6. Layer input buffer, 400 MiB: sized by n_batch in f32; n_ubatch rows (100 MiB) or f16 would do if the drafter consumed it per ubatch.
7. CUDA context, 432 MiB: per process, set by the driver and the kernel image; hard to change.
8. kvq: codes 625 (needed), scratch 221.5 (prefill only, could share memory with the compute buffer), ring 144.5 (one batch), tables 109, fitter 83, unlogged 85 (find where it is).
9. Drafter K/V, 60 MiB: small (2048-cell window + one ubatch).

## Strange or inconsistent, and corrections to the brief

- The brief says the token files have 99,395 / 49,737 / 24,183 / 11,705 tokens: those are the prefix lengths. The files hold 99,869 / 50,029 / 24,431 / 12,055 ids (prefix and the recorded reply). Memory runs use the prefix lengths; the acceptance set uses the prefix + 3 (the prompt then ends with the assistant turn header).
- Vocabulary: the target, the drafter and the Qwen3.8-3.6 blend file have identical token strings, token types and merges (sha256 of each array), so the ids are valid; the padding id differs (248055 in the target, 248044 in the others), the chat templates differ (not used: the requests are token ids). Largest id in the token files: 248069.
- The incoai and z-lab drafter files: all 81 tensors are byte-identical (sha256 per tensor); only `general.organization` and `general.source.url` differ, the header is 2 bytes shorter in z-lab and the alignment padding makes the files the same size. So the different file SHA256 comes only from those two strings.
- "the drafter's context and its caches" grow with n_ctx: no. The drafter has a 2048-cell sliding window on all 5 layers, its K/V is 60 MiB at every n_ctx, its non-SWA cache has 0 layers; its compute buffer does not depend on n_ctx either.
- ggml-org master has 1197 MiB of recurrent state at n_max 7, as the vvsotnikov doc says (verified).
- With --kvq the dedicated counter is 5 - 16 MiB above nvidia-smi (see above).
- The first kvq session of the day (t1) had a 62 s prefill and a 4.06 s end-of-prompt wait while other agents compiled; the matrix sessions had 43 s and 0.55 s. Memory was the same.
- Without drafter the plain run stopped at 194 tokens (eos) where every drafted run generated 300: greedy speculative decoding does not reproduce the non-speculative ids here (batch-shape rounding), so drafts are verified against a slightly different continuation than plain decoding.

## Draft acceptance (baseline for later drafter or setting changes)

Set: 12 requests in a fixed order (`mem\acc\requests.txt`): the four trace files of `tok\` and 8 rows of the earlier wide test (prompt files rebuilt from the same source rows into `mem\acc\tok\`, prompt only), prompts of 11.7k to 99.4k tokens, each prompt = the first ids up to and including the assistant turn header (prefix + 3). Request: `/completion`, token ids, `n_predict` 512, `temperature` 0 (greedy), `seed` 42, `cache_prompt` true, `return_tokens` true, not streamed. One server session per run, n_ctx 102400, the launch line above. Script: `mem\acc\accset.sh TAG BIN [flags]` (env `DRAFTER=` for another drafter file, `REQS=`, `NPRED=`, `SEED=`), runner `runacc.sh accmatrix.txt`, analysis `accana.py TAG... --ref TAG`, `paired.py A B`. Generated ids are never stored: the client keeps a salted sha256 chain per position (`results.jsonl`, salt in `salt.txt`), which gives the first position where two runs differ.

Fields: generated = `tokens` of the response (= `timings.predicted_n` on all 84 requests); drafted = `timings.draft_n`; accepted = `timings.draft_n_accepted`; verify steps = the steps with a draft from the server log line "draft acceptance = ... mean len = M" (steps = accepted / (M - 1)); acceptance rate = accepted / drafted; acceptance length tau = 1 + accepted / steps (the server's mean len, tokens per target verify step). Check: generated / steps gives 4.405 against tau 4.394 pooled on master (the first token comes from the prompt and an eos can cut the last step), the accepted count of the log line equals `draft_n_accepted` on every request (asserted).

Cells: generated / drafted / accepted / steps, tau, generation t/s.

| request | prompt | ggml-org master a49db739f | ours with --kvq | vvsotnikov V1-ship | ours plain | vvsotnikov V3-ship (tree 15, not comparable) |
|---|---|---|---|---|---|---|
| r2337 | 11708 | 182 / 259 / 145 / 37, 4.92, 224 t/s | 234 / 343 / 184 / 49, 4.76, 184 t/s | 182 / 273 / 142 / 39, 4.64, 214 t/s | 182 / 259 / 145 / 37, 4.92, 225 t/s | 182 / 525 / 146 / 35, 5.17, 210 t/s |
| w1121 | 13592 | 220 / 364 / 167 / 52, 4.21, 213 t/s | 112 / 161 / 88 / 23, 4.83, 192 t/s | 318 / 553 / 238 / 79, 4.01, 206 t/s | 220 / 364 / 167 / 52, 4.21, 210 t/s | 318 / 1185 / 238 / 79, 4.01, 185 t/s |
| w2449 | 17875 | 512 / 805 / 395 / 116, 4.41, 216 t/s | 512 / 779 / 399 / 112, 4.56, 178 t/s | 512 / 838 / 391 / 120, 4.26, 213 t/s | 512 / 805 / 395 / 116, 4.41, 215 t/s | 512 / 1662 / 399 / 112, 4.56, 202 t/s |
| r310 | 24186 | 124 / 175 / 98 / 25, 4.92, 232 t/s | 134 / 231 / 100 / 33, 4.03, 144 t/s | 134 / 224 / 101 / 32, 4.16, 202 t/s | 124 / 175 / 98 / 25, 4.92, 230 t/s | 134 / 480 / 101 / 32, 4.16, 180 t/s |
| w2586 | 31836 | 512 / 877 / 385 / 126, 4.06, 185 t/s | 512 / 880 / 384 / 127, 4.02, 142 t/s | 512 / 842 / 389 / 122, 4.19, 194 t/s | 512 / 877 / 385 / 126, 4.06, 185 t/s | 512 / 1750 / 393 / 118, 4.33, 179 t/s |
| w1393 | 41354 | 354 / 448 / 290 / 64, 5.53, 235 t/s | 484 / 574 / 402 / 82, 5.90, 183 t/s | 354 / 448 / 290 / 64, 5.53, 243 t/s | 354 / 448 / 290 / 64, 5.53, 236 t/s | 354 / 840 / 297 / 56, 6.30, 250 t/s |
| r1830 | 49740 | 271 / 462 / 205 / 66, 4.11, 171 t/s | 291 / 581 / 207 / 83, 3.49, 108 t/s | 262 / 483 / 192 / 69, 3.78, 162 t/s | 271 / 462 / 205 / 66, 4.11, 169 t/s | 262 / 990 / 195 / 66, 3.95, 152 t/s |
| w2088 | 64777 | 78 / 126 / 60 / 18, 4.33, 163 t/s | 78 / 126 / 60 / 18, 4.33, 121 t/s | 78 / 126 / 60 / 18, 4.33, 165 t/s | 78 / 126 / 60 / 18, 4.33, 163 t/s | 78 / 270 / 59 / 18, 4.28, 150 t/s |
| w579 | 79090 | 86 / 119 / 68 / 17, 5.00, 174 t/s | 86 / 119 / 68 / 17, 5.00, 137 t/s | 86 / 119 / 68 / 17, 5.00, 178 t/s | 86 / 119 / 68 / 17, 5.00, 174 t/s | 86 / 225 / 70 / 15, 5.67, 184 t/s |
| w142 | 91071 | 512 / 836 / 391 / 120, 4.26, 148 t/s | 512 / 796 / 397 / 114, 4.48, 116 t/s | 512 / 847 / 389 / 122, 4.19, 147 t/s | 512 / 836 / 391 / 120, 4.26, 147 t/s | 512 / 1695 / 396 / 115, 4.44, 143 t/s |
| w1643 | 97197 | 512 / 810 / 394 / 116, 4.40, 147 t/s | 512 / 780 / 399 / 112, 4.56, 119 t/s | 512 / 798 / 396 / 115, 4.44, 153 t/s | 512 / 810 / 394 / 116, 4.40, 148 t/s | 512 / 1640 / 400 / 110, 4.64, 145 t/s |
| r2660 | 99398 | 192 / 350 / 141 / 50, 3.82, 127 t/s | 512 / 775 / 399 / 112, 4.56, 120 t/s | 512 / 784 / 399 / 112, 4.56, 156 t/s | 192 / 350 / 141 / 50, 3.82, 128 t/s | 512 / 1590 / 404 / 107, 4.78, 148 t/s |
| pooled | | 3555 / 5631 / 2739 / 807, rate 0.486, tau 4.394 [4.20, 4.70], 178.0 t/s | 3979 / 6145 / 3087 / 882, rate 0.502, tau 4.500 [4.17, 4.89], 138.4 t/s | 3974 / 6335 / 3055 / 909, rate 0.482, tau 4.361 [4.16, 4.62], 179.4 t/s | identical to master, 177.7 t/s | 3974 / 12852 / 3098 / 863, rate 0.241, tau 4.590 [4.33, 4.94], 170.3 t/s |

[ ] = 95% bootstrap interval over the 12 requests (4000 resamples). Pooled t/s = all generated tokens / all generation time.

Repeatability and noise:
- Two runs of the same configuration are identical: master run a and run b, and --kvq run a and run b, have equal ids on all 12 requests (hash chains equal to the last position) and equal counts. Greedy target and greedy DFlash2 drafting are deterministic here, so a second seed is meaningless (the seed is not used) and the run-to-run spread is 0.
- The noise that matters is the choice of requests: the interval of the pooled tau is about +-0.25 (+-6%). Two drafters compared on the same build and the same requests should be compared paired (per request); the paired 95% interval of a tau difference here is about +-0.16 (vvsotnikov V1-ship - master: -0.033 [-0.195, +0.136]; ours --kvq - master: +0.106 [-0.135, +0.289]; V3 tree - master: +0.196 [+0.009, +0.365]). So with this set a change of tau smaller than about 0.2 (5%) is not significant; a larger set (more requests, not longer ones) narrows it as 1/sqrt(n).
- Ids across builds: ours plain equals master on all 12 requests. vvsotnikov V1-ship differs from master on 9 of 12 requests (first difference at positions 15 to 278), ours --kvq on 10 of 12 (positions 5 to 147): the target computes slightly different hidden states (other kernels; the coded K/V of the two-tier cache), so acceptance is measured on a different continuation there. The generated lengths change accordingly (r2660: 192 tokens on master, 512 on both others). V3 equals V1-ship on 8 of 12.
- Context length: tau for the 5 requests under 40k against the 5 at 60k and more: master 4.34 / 4.28, --kvq 4.36 / 4.55, V1-ship 4.22 / 4.42. No dependence on the prompt length is visible in this data; the spread between requests (3.5 to 5.9) is much larger.
- Generation speed: --kvq 138 t/s against 178 on master (-22%) in this set; V1-ship 179; the tree (V3) 170 t/s with tau 4.59 but 2.1x the drafted tokens.

## Files

- `mem\dfl\` harness and data: `sess.sh`, `memsample.ps1`, `runmatrix.sh`, `matrix.txt`, `matrix_dx.txt` (diagnostics: no CUDA graphs, `--spec-draft-n-ubatch 128`, no drafter), `py\` (analyze, components, remainder, steps, grid, ggufmeta, ggufdiff, vocabhash), `runs\TAG\`, `bins\` (`--version` and `--help` of each binary), `meta_*.txt` (GGUF metadata).
- `mem\acc\` acceptance set: `requests.txt`, `tok\` (prompt-only token files of the 8 wide-test rows, private data), `accreq.py`, `accset.sh`, `runacc.sh`, `accmatrix.txt`, `accana.py`, `paired.py`, `runs\TAG\`.
