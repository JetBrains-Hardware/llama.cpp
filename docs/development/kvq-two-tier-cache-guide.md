# Two-tier KV cache (kvq) at 250k context in 15.5 GiB: user guide and final results

Branch `kvq` of the JetBrains-Hardware fork of ggml-org/llama.cpp (master + the merged ggml-org PR 31669 + the kvq series), state of 2026-11-19. It is the default path of the prototype branch `kvq-two-tier-cache` (last code commit 6d2ecbfab) without the dead modes and debug probes, see section 11, and keeps the resident rows in one device buffer per layer (section 6.1). This file is self-contained: how to build and run the branch, every knob, the memory of each phase, the speed and the quality, the known costs and limits. The design, the history of the decisions and the raw measurements are in `kvq-two-tier-cache.md` next to it.

## 1. What the branch does

For Qwen3.8-27B (GGUF arch `qwen35`: 64 trunk blocks, 16 attention layers, 48 gated-delta-net layers) the attention K/V cache is split in two tiers:

- VRAM holds coarse codes of every fitted position (K 2 bit, V 1 bit plus a gain, tables per 8k chunk) and a small resident set of exact rows chosen per token by an early query; a custom attention op reads both during decoding.
- Pinned host RAM holds the exact q8_0 rows of all positions (the precise store), the reply rows (the ring) and, during a prompt, the codes and tables too.
- Prefill (the prompt) attends over exact rows only: each attention layer streams its rows from the host store into a device scratch and runs flash attention in mask-free key blocks of 16k rows, merged through a log-sum-exp output. The codes are not read during a prompt.

Prefill and decode are two phases that never overlap (one slot). Each phase holds only its own buffers: the prompt phase has the prompt compute buffer, the scratch, the fitter and the drafter's layer input rows, the decode phase has the codes and tables, the small decode compute buffers and the resident / tail buffers. The GDN recurrent rollback uses replay (ggml-org PR 31669, merged) instead of snapshot rows.

Result: `llama-server --kvq` at 250k context with the DFlash2 drafter fits 15.5 GiB (15,872 MiB) of dedicated VRAM in every phase, at ubatch 1024 and, with one more flag, at ubatch 4096, and the 250k prompt is 30% faster than before this round.

Commits of this round on top of a724f2299: 4218b5b9f (merge of ggml-org PR 31669), 9a754beaa (replay wiring), fba885494 (compute buffers per phase), 15b59d61d (docs), affa1fa81 (codes and tables in host memory during the prompt), 6d2ecbfab (blocked scratch attention with the flash-attention LSE output), 04320bec2, dfa7c26f0, 35035fdaf (docs).

## 2. Build

Windows, Visual Studio 2022 build tools, CUDA 12.9, Ninja. From a VS x64 developer shell:

```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-cuda --target llama-server llama-kvq-eval test-backend-ops test-kvq-codec test-kvq-attn llama-kvq-bench
```

CUDA only: the attention op, the fit op and the LSE flash attention have CPU references for the tests and CUDA kernels for running; Metal and Vulkan are not supported (the fit mode refuses to start when the device cannot run the fit graphs; the other backends and the meta backend (tensor split) refuse the flash attention with the LSE output in `supports_op`, the meta backend also the kvq ops).

Tests that cover this branch (run on the GPU, `-b CUDA0`): `test-backend-ops -o FLASH_ATTN_EXT_LSE`, `-o FLASH_ATTN_EXT`, `-o KVQ_ATTN`, `-o KVQ_FIT`, `-o GATED_DELTA_NET`, `test-kvq-codec`, `test-kvq-attn`. All pass on `kvq`. `llama-kvq-bench` times the attention op on the GPU (`--quick`, `--profile`, `--hash`).

## 3. Run

Measured configuration (IQ3_S target, DFlash2 Q4_K_M drafter, 250k context):

```
llama-server --fit off -m Qwen3.8-27B-UD-IQ3_S.gguf -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf ^
  --ctx-size 256000 --n-gpu-layers all -ngld all --parallel 1 ^
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja ^
  --spec-type draft-dflash --spec-draft-n-max 7 -ctkd f16 -ctvd f16 ^
  --batch-size 4096 --ubatch-size 1024 --kvq
```

The two configurations that fit 15,872 MiB at 250k:

| | flags | largest VRAM sample at 250k | 250k prompt | why |
|---|---|---|---|---|
| A, default | `--ubatch-size 1024 --kvq` | 15,271 MiB (decode phase) | 139 s | smallest footprint |
| B, large ubatch | `--ubatch-size 4096 --spec-draft-n-ubatch 1024 --kvq` | 15,372 MiB (prompt phase) | 136 s | 2% faster prompt |

B without `--spec-draft-n-ubatch 1024` is 15,876 MiB (4 MiB over). Since the blocked attention, ubatch 4096 gains only about 2% over 1024, so A is the recommendation; B is a speed knob.

What `--kvq` forces, with a warning in the log: one slot (`--parallel 1`), context shift and cache reuse off, flash attention on, q8_0 K and V, draft n_max at most 7, recurrent rollback by replay (context param `rs_replay`; `LLAMA_RS_REPLAY=0` keeps the snapshots). It refuses to start for a model that is not `qwen35`, with `--mmproj`, or with embeddings. `--fit` does not know the kvq memory, so use `--fit off`.

What to expect in the log: the memory line of the store at load (`llama_kvq_store: device: codes ... (decode phase only) ...; host: store ... (pinned), codes and tables ...`), per prompt `scr_alloc` / `codes_free` at the start and `codes_alloc` / the fit wait at the end (`kvq: waited ... at the end of the prompt, coarse region N rows`), and the phase switches of the compute buffers (`sched_reserve: ... compute buffer size of the tg phase`).

Slot save and restore and the RAM prompt cache work with the default settings (a kvq state is about 29% larger than a plain one, about 1.77 GB at 24k; a plain snapshot cannot be loaded into a kvq context and the other way round).

## 4. Knobs

Server flags of the two-tier cache (`common/arg.cpp`); defaults are the measured values:

| flag | default | effect |
|---|---|---|
| `--kvq` | off | turns the two-tier cache on, with the forced settings above |
| `--kvq-scr-block N` | 16384 | prompt attention over the host rows in mask-free key blocks of N rows (rounded down to 256). 0 = one flash attention with a mask over all rows: the old numerics, bit-identical to a full device cache, 30% slower at 250k and 1.3 GB more prompt-phase VRAM. 8k saves 64 MiB for +3.5% prompt time; 32k and 64k are slower. Env `LLAMA_ARG_KVQ_SCR_BLOCK` |
| `--kvq-chunk N` | 8192 | rows per fitted chunk (one table set per chunk); 16k halves the tables (268 -> about 135 MiB at 256k) for a small quality cost (mean KL ratio 1.12) |
| `--kvq-rem own|merge` | own | how the rows after the last complete chunk are fitted at the end of a prompt |
| `--kvq-min-rows N` | 256 | `own`: fewer rows after the last complete chunk stay precise in the ring |
| `--kvq-tail N` | max(n_batch, 4096) + 256 | ring cells (pinned host memory, 144.5 MiB at 4352); a long reply folds every ring size of tokens (0.57 s each); not below n_batch + 256; must be > 0 |
| `--kvq-prefetch-n N` | 1024 | rows per (kv head, position) selected by the early query |
| `--kvq-resident-cap N` | 0 = 8192 | resident slots per (layer, kv head); 0 = max_batch x prefetch_n; 6144 is as good at 8 per batch and saves 71 MB of pinned memory |
| `--kvq-lead N` | 2 | how many layers ahead the early query runs |
| `--kvq-layer-bufs N\|all` | all | device buffers of the resident rows. `all`: one per attention layer, the rows stay on the device and a verify batch uploads only the fetched rows. N >= 2: the layers rotate through N buffers and every verify batch re-uploads each layer's resident set from its pinned mirror (272 MiB at cap 8192); slower on a slow bus, 110 MiB less VRAM below 250k, same output (section 6.1) |

Flags of the rest of the stack that matter for the budget:

| flag | default | effect |
|---|---|---|
| `--ubatch-size N` | 1024 in the measured config | prompt ubatch; sets the prompt compute buffer (485 MiB at 1024, 1,570 at 4096 with the blocked attention) and the host compute buffer (89 / 354 MiB) |
| `--batch-size N` | 4096 | tokens per `llama_decode`; sets the drafter's layer input buffer (400 MiB at 4096, 100 at 1024; prompt phase only) and the ring size. Changes the drafter's batch boundaries and so the generated ids |
| `--spec-draft-n-ubatch N` | = `--ubatch-size` | ubatch of the drafter context; 1024 keeps its prompt compute at 120 MiB with `--ubatch-size 4096` (480 without). Changes ids at ubatch 4096 in some cells (other drafts) |
| `--spec-draft-n-max N` | 7 measured (3 also measured) | verify batch is n_max + 1 tokens; the decode compute of the tg phase and the resident capacity follow it |
| `--ctx-size N` | 256000 measured | codes 1,562 + tables 268 MiB on the device in decode and in host memory always, host store 34.8 KB per position (8.5 GB at 256k), scratch 540 MiB (one layer of exact rows) |

Memory layout and test knobs:

| setting | default | effect |
|---|---|---|
| `--[no-]phase-buffers` (context param `phase_bufs`) | on with `--kvq` and for the DFlash drafter, off otherwise | `--no-phase-buffers`: one compute buffer for all ubatches, no switch between the prompt and the decode phase; then decode holds the prompt buffers again (about +1.7 GB at 250k). `--phase-buffers` turns the switch on for a plain context |
| `LLAMA_RS_REPLAY` | unset | overrides the context param `rs_replay`, which `--kvq` turns on: 0 keeps the GDN snapshot rows (+1,020 MiB at n_max 7) instead of the replay (+27 MiB of host memory per checkpoint or slot file) |
| `LLAMA_KVQ_FOLD_EVERY=N` | off | test knob: a fold whenever N reply rows follow the coarse region (the only way to exercise folds in kvq-eval) |
| `LLAMA_KVQ_UP_REPEAT=k` | 1 | test knob: every base upload of the resident rows (rotating buffers: every round; own buffers: the mirror after a prompt) and every tail upload k times, emulates a slower bus; k = 2 is about PCIe 4.0, k = 3 about PCIe 3.0 on the RTX 5090 (the join uploads of the fetched rows are not repeated) |

Which knobs change the generated ids (the known batch-boundary effect, not a bug): `--ubatch-size`, `--batch-size`, `--spec-draft-n-ubatch`, `--kvq-scr-block` (any non-zero value against 0 and against each other), `--spec-draft-n-max`. Which do not: `--no-phase-buffers`, `LLAMA_RS_REPLAY`, the codes-in-host-memory mechanism, slot save and restore.

## 5. Memory

Dedicated VRAM of the process in MiB (5 Hz sampler), RTX 5090, n_max 7, budget 15,872. "Prompt peak" is the largest sample of the first prompt up to its first token (it includes the switch into decode), "follow-up" the peak of a 500-token follow-up prompt. "Before" is the build at the start of this round (a724f2299).

### 5.1 Per phase, 250k (n_ctx 256000)

| configuration | load | prompt peak | decode median | follow-up peak | headroom |
|---|---|---|---|---|---|
| before, ubatch 1024 | 19,277 | 19,483 | 19,579 | 19,581 | -3,709 |
| A: ubatch 1024, block 16k | 14,039 | 14,205 (15,187 at the switch into decode) | 15,159 | 15,271 | +601 |
| ubatch 1024, block 0 | 15,364 | 15,546 | 15,168 | 15,584 | +288 |
| 15b59d61d (phases + replay, no codes out, no blocks), ubatch 4096 | 19,628 | 20,300 | 15,310 | 20,297 | -4,428 |
| B: ubatch 4096 + drafter ubatch 1024, block 16k | 15,126 | 15,372 | 15,176 | 15,354 | +500 |
| ubatch 4096, block 16k, no drafter flag | 15,186 | 15,876 | 15,320 | 15,408 | -4 |
| ubatch 4096 + drafter ubatch 1024, block 0 | 17,746 | 17,974 | 15,172 | 17,854 | -2,102 |

Headroom = 15,872 minus the largest sample of the session. In A the decode phase sets it (the codes come back on the device), in B the prompt phase.

### 5.2 Per phase, 100k (n_ctx 102400)

| configuration | load | prompt peak | decode median | follow-up peak |
|---|---|---|---|---|
| before, ubatch 1024 | 16,950 | 17,126 | 17,196 | - |
| A: ubatch 1024, block 16k | 13,707 | 13,873 | 13,631 | 13,903 |
| ubatch 1024, block 0 | 14,132 | 14,314 | 13,640 | 14,184 |
| B: ubatch 4096 + drafter ubatch 1024, block 16k | 14,794 | 15,040 | 13,682 | 15,000 |

### 5.3 What the device holds at 250k, configuration A (MiB)

| group | prompt phase | decode phase | note |
|---|---|---|---|
| target weights (IQ3_S) | 10,617 | 10,617 | |
| drafter weights (Q4_K_M) | 1,080 | 1,080 | |
| drafter KV cache (f16) | 60 | 60 | |
| recurrent state (GDN, replay) | 177 | 177 | 1,197 with snapshot rows |
| kvq codes + tables | 0 | 1,831 | pinned host copy during the prompt |
| kvq scratch, shared with resident / tail / staging | 540 | 540 | one layer of exact rows; decode uses about 330 of it (88 with `--kvq-layer-bufs 2`) |
| kvq other | 17 | 17 | slot lists, counters |
| kvq fitter | 1 | 1 | since the GPU table fit (2026-11-20) it works in the kvq scratch memory between the ubatches; was 86 in the prompt phase (session 250k: load 14,039 -> 13,951, run max 15,269 unchanged; 24k: -88 everywhere) |
| target compute buffer | 485 | 72 | 1,570 at ubatch 4096; decode size is learned (high-water mark) |
| drafter compute buffer | 120 | 9 | injection graph |
| layer input buffer (5 layers for the drafter) | 400 | 1 | n_batch rows in the prompt, 8 in decode |
| remainder (CUDA context, cuBLAS, ggml CUDA pool) | 620 - 705 | 755 - 772 | not broken down |
| measured total | 14,205 | 15,159 | |

Host memory alongside, pinned unless noted, at 256k: precise store 8,500, codes and tables 1,831, ring 145, other kvq 281, host compute 89 (354 at ubatch 4096), drafter host compute 100, memory-mapped model 521 (not pinned), plus 27 MiB per context checkpoint, slot file or RAM cache entry for the replay's pending inputs. About 11.5 GB in all.

### 5.4 Where the 4.4 GB went (250k, ubatch 1024, prompt phase, before -> now)

| change | MiB | phase |
|---|---|---|
| GDN rollback by replay instead of snapshot rows (PR 31669) | -1,020 | both |
| compute buffers per phase: drafter prompt buffer is the injection graph (1,099 -> 120), decode holds no prompt buffers, fitter and scratch prompt-only | -980 at the prompt peak, -4,400 in decode | |
| codes and tables in host memory during the prompt | -1,830 | prompt |
| blocked scratch attention (no big mask, f16 copies per block) | -1,325 (prompt compute 1,810 -> 485); -500 of pinned host compute | prompt |
| decode compute buffer learned per phase (1,810 -> 72) | | decode |

## 6. Speed

Interleaved pairs against the build before, same session block, 300-token replies, IQ3_S target, DFlash2 drafter n_max 7. "Before" here is 15b59d61d (phases and replay, which are speed-neutral: 250k prompt 198 s against 206 s of a724f2299 on another day, decode within 1.5%).

| cell | prompt before | prompt now | decode ms per verify batch, before -> now | follow-up prompt (500 tokens), before -> now |
|---|---|---|---|---|
| 250k, A (ubatch 1024, 16k) | 199.3 s | 139.1 s (-30%) | 37.5 -> 37.2 | 1.71 -> 1.50 s (-12%) |
| 250k, ubatch 1024, block 0 | 199.3 s | 201.9 s (+1.3%, pair not adjacent) | 37.5 -> 37.4 | 1.71 -> 1.73 s (+1.5%) |
| 250k, B (ubatch 4096 + drafter 1024, 16k) | 164.0 s | 136.1 s (-17%) | 38.4 -> 37.5 | 1.76 -> 1.49 s |
| 100k, A | 42.9 s | 37.3 s (-13%) | 25.8 -> 25.9 | 1.28 -> 1.14 s |
| 100k, B | 39.5 s | 37.2 s (-6%) | 26.0 -> 26.3 | 1.36 -> 1.25 s |

Decode is unchanged within noise wherever the ids are equal (the decode path is not touched by this round). The end-of-prompt fit wait is 0.55 - 0.59 s at every length (4 threads).

Where the prompt speed comes from: the single flash attention over the whole scratch filled an F16 mask of n_rows x n_ubatch on the CPU (512 MiB per ubatch at 250k), uploaded it and read it in every attention layer, and the mask-less GQA kernel is a few percent faster than the masked one once the dispatch was fixed. The same structure exists in ggml-org's plain KV cache, so the gain is not specific to the two-tier cache.

Costs of this round, accepted by the owner: the follow-up prompt pays the re-upload of the codes (1.75 GB at about 34 GB/s plus an alloc and a free, about 57 ms at 250k, 20 ms at 100k); with block 0 that is +1.5 to +3.3% of the follow-up prompt phase, with block 16k the follow-up is still 12% faster than before. The switch between phases costs 0.5 - 6 ms per compute buffer and about 18 ms into a prompt in all.

### 6.1 Resident buffers per layer (`--kvq-layer-bufs`, step 1b)

Default since step 1b: each attention layer keeps its resident rows in its own device buffer, placed in the memory the prompt scratch shares with the decode path. After a prompt the mirrors go back in once (on the base upload stream); a reset of the resident set re-uploads only the sink rows. Before, the 16 layers rotated through 2 buffers and every verify batch re-uploaded every layer's resident set (272 MiB at cap 8192). Idea and first patch from the RTX 5070 Ti field test (PCIe 3.0, 256k: 48.0 -> 39.0 ms per verify batch, +2 MiB peak VRAM).

RTX 5090, interleaved pairs in one session block, 300-token replies with `ignore_eos`, `LLAMA_KVQ_UP_REPEAT=k` (k = 2 about PCIe 4.0, k = 3 about PCIe 3.0). ms per verify batch = predicted_ms / packed ubatches of the request (drafter included, so not the same metric as the table above). Rotating -> own, first request / follow-up:

| context | per batch | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| 24k | 8 | 25.33 -> 25.36 / 24.96 -> 25.03 | 25.76 -> 25.19 / 25.67 -> 24.96 | 27.78 -> 25.37 / 27.93 -> 25.20 |
| 24k | 4 | 21.10 -> 20.94 / 21.14 -> 21.10 | 21.77 -> 21.08 / 22.01 -> 21.05 | 25.20 -> 21.05 / 25.62 -> 21.08 |
| 100k | 8 | 29.53 -> 29.48 / 31.53 -> 31.44 | 30.12 -> 29.61 / 32.44 -> 31.72 | 31.36 -> 29.68 / 34.15 -> 31.84 |
| 100k | 4 | 23.78 -> 23.64 / 24.60 -> 24.38 | 24.40 -> 23.69 / 25.05 -> 24.39 | 27.26 -> 23.91 / 28.25 -> 24.57 |
| 250k | 8 | 40.96 -> 40.34 / 43.58 -> 42.90 | 41.55 -> 40.77 / 44.19 -> 43.59 | 43.08 -> 40.81 / 45.76 -> 43.40 |
| 250k | 4 | 28.41 -> 28.36 / 30.32 -> 30.12 | 29.17 -> 28.57 / 31.18 -> 30.48 | 30.73 -> 28.72 / 32.99 -> 30.52 |

These numbers are from 92e56429a. On a30f3a44c (the restore on the base stream) four pairs were repeated: 250k 8 per batch k = 1 41.20 -> 40.41, k = 3 43.52 -> 40.62; 24k 4 per batch k = 1 21.11 -> 21.01, k = 3 25.45 -> 21.37. With own buffers the time no longer depends on the bus: k = 3 costs at most +0.4 ms against k = 1, where the rotating buffers pay +2 to +4 ms. At k = 1 the two are equal within 0.1 ms at 24k and 100k, and own buffers are 0.6 ms faster at 250k. Ids are equal to the rotating path in every cell, and to bin-final in the checkpoint set (24k, 100k, 250k, 100k block 0, slot save / restore, fold at 50k, kvq-eval checks).

VRAM (MiB, n_ctx 102400 for 24k and 100k, 256000 for 250k):

| context | load | prompt peak | decode median | follow-up peak |
|---|---|---|---|---|
| 24k, rotating -> own | 13,707 -> 13,817 | 13,873 -> 13,983 | 13,599 -> 13,709 | 13,739 -> 13,849 |
| 100k, rotating -> own | 13,707 -> 13,817 | 13,873 -> 13,983 | 13,631 -> 13,741 | 13,751 -> 13,861 - 13,973 |
| 250k, rotating -> own | 14,039 -> 14,039 | 15,105 - 15,217 both | 15,159 -> 15,159 | 15,247 - 15,271 both |

Below 250k the 16 resident sets (285 MiB) plus the tail buffers are larger than the scratch (221.5 MiB at n_ctx 102400), so the shared memory grows by 110 MiB. That is still below the 250k peak. At 250k the scratch is 540 MiB and holds them all, so nothing changes.

## 7. Quality

### 7.1 Bit identity

Identical generated ids, slot files and kvq-eval logits to the build before, in every session checked (24k, 50k, 100k, 250k, n_max 7 and 3, follow-up prompts, a 5000-token reply with a fold, slot save and restore): the replay, the compute buffers per phase, the codes in host memory, and `--kvq-scr-block 0`. So with block 0 the whole round is a pure memory change.

### 7.2 The blocked attention (default block 16k)

It changes the rounding of the prompt, so the ids differ from the single attention. Its quality was checked with kvq-eval (fresh prefill of a recorded trace, teacher-forced KL on the recorded reply, 8 tokens per batch) on traces of 20k, 50k, 60k, 80k and 100k tokens. At the first scored position every block size equals the old path to 1e-13; the runs diverge later through the fitted codes.

KL to the f16 reference (A1, single attention) on the 100k trace:

| ubatch | old path (block 0) | block 16k |
|---|---|---|
| 512 | 1.63e-4 | 2.78e-4 |
| 1024 | 1.49e-4 | 3.03e-4 |
| 2048 | 1.34e-4 | 2.08e-4 |
| 4096 | 1.41e-4 | 2.58e-4 |

Taken alone this says the new path is worse. It is not: the reference itself computes its prefill as one flash attention over all rows, and the CUDA MMA flash-attention kernel keeps the probabilities and the P x V accumulator in half precision against a running maximum over all rows, so the probability mass far below the maximum is lost over a 100k tail. Checked on dumped rows of the real prompt (attention layer 31, last ubatch, 99,584 rows, exact f64 attention on the CPU as the reference): the single attention has nmse 3.3e-4 and an output bias of -1.0%, the 16k-blocked graph nmse 2.6e-5 and -0.3%, closer on 1,520 of 1,536 (query, head) pairs. Against references built the same way but with blocked prefill attention (16k, 8k, 4k, 2k) the ranking reverses: block 16k 1.1 - 1.8e-4 against the old path 1.6 - 2.8e-4, the worst new run better than the best old run against each of them. Top-1 agreement is the same either way (0 - 2 of 471 positions differ).

Consequences: block 16k is the default; a KL against a single-attention reference cannot rank attention variants that change the reduction order; at 100k a KL is only good to about 1e-4 (references that differ only in rounding order are 1 - 1.5e-4 apart). The bias is in ggml's standard CUDA flash attention for any long prompt on Turing and later, not in this branch. The blocked reference logits and the exact-attention script are in `kvq-scratch/mem/attnsplit` (`ref/A1b*-row2660.ref`, `scripts/exact.py`).

## 8. Known costs, limits and gaps

- Not supported: see the next section (one sequence only, no context shift, no cache reuse, CUDA only, ...).
- The graph is re-planned at almost every prompt ubatch because the number of key blocks follows the number of stored rows (the buffer does not grow; the cost is inside the prompt times above).
- The follow-up prompt pays the codes upload (section 6). A per-layer overlapped upload was built and hid nothing (the batch it would hide behind is shorter than the upload), so the simple version is in the tree.
- Host memory: 8.5 GB of pinned RAM for the store at 256k plus 1.8 GB for the codes copy, per sequence.
- Quality measured on one 100k trace and four shorter ones; nothing at 150k or above; no long session (hours, many rollbacks and refits); no automated tests of the store, the fitter, the ring or the server flows beyond the codec and kernel tests.
- The branch changes ggml (two new ops, the scheduler callback order, `ggml_gallocr_resize`, the CUDA flash-attention kernels) and carries the merge of the open PR 31669; rebases will conflict there and the final upstream version of the PR will have to replace the merged one.

### 8.1 Not supported

Each of these is a documented limit (release blockers R1, R2 and R6 in `kvq-scratch/RELEASE-BLOCKERS.md` list what supporting them takes). The memory operations abort with a message that starts with `not supported by the two-tier KV cache (--kvq)`, then the reason and what to do; the server never calls them with its `--kvq` settings (one slot, no context shift, no cache reuse).

- `seq_cp`, `seq_keep` of a sequence > 0, `seq_rm` of a sequence > 0, and more than one sequence: the store and the fitter hold sequence 0 only. Use `n_seq_max = 1` (server: `-np 1`).
- `seq_add` and `seq_div`: qwen35 uses IMROPE, and no KV cache can shift its keys (`llama_memory_can_shift` is false without `--kvq` too, so the server already disables the context shift and `--cache-reuse` for this model). This is a limit of ggml-org's KV cache, not of the two-tier cache.
- `seq_rm` of a middle range: only the end of the sequence can be removed, `[p0, -1)` with `p0 <= last position`. The plain hybrid memory accepts it (the attention cache keeps a gap); for the two-tier cache this needs a row mask in the coarse attention and in the blocked scratch attention (release blocker R1).
- Multimodal input and embeddings: the memory serves text generation of the qwen35 target only.
- `--fit`: it measures a plain q8_0 K/V cache and does not know the kvq memory (device codes, tables, scratch, resident sets, host store). Use `--fit off`, with `--ctx-size` and `-ngl` set (release blocker R2).
- More than one kvq context per GPU per process: the CUDA attention uses one side stream and function-static fork / join events per device (release blocker R6).
- CUDA only: the other ggml backends refuse the kvq ops and the flash attention with the LSE output, and `--kvq` refuses to start on a model or device that cannot run them.

### 8.2 Using it from a program

One entry point, `llama_init_from_model_kvq(model, cparams, llama_kvq_config)` in `src/llama-ext.h` (the server and kvq-eval go through `common_init_from_params`, which calls it). The setting belongs to the context, not to the model: the context gets the packed memory, the ring of `llama_kvq_config::tail` cells and the fit, and the attention rotation is off in its own attention cache only. A second context of the same model (an MTP drafter, a plain context) keeps the rotation and a normal memory. `llama_model_set_attn_rot` (the ggml-org setter, for tests) is not used by `--kvq`.

## 9. Ideas kept for later (owner's order)

1. Scratch as a ring of 16k-row blocks fed on the copy stream: 540 -> about 105 MiB in both phases (the one item that moves the decode phase, which sets the headroom in configuration A); 10 - 20 ms of alloc / free per phase switch is acceptable.
2. Prompt-size policy for follow-up prompts: layer input buffer sized by the batch (no setting); codes and tables kept on the device when a request has fewer than N new tokens (server setting, about 4k, safe at 250k only after item 1); ubatch 4096 as a speed knob only.
3. For huge requests: run the prompt through two halves of the layers in turn, moving weights and cache between RAM and VRAM.
4. Optional, from ggml-org: PR 31802 (fused q8_0 dequantization in the MMA flash-attention kernels) would remove the f16 K/V copies of the prompt attention; its base is far from ours.
5. Upstream candidates: the LSE flash attention, the GQA dispatch fix for mask-less attention, and the blocked causal prefill for the plain KV cache.

## 10. Where the material is

Outside the repository, `../kvq-scratch`: `gpurun.sh` (GPU lock for parallel agents), `tok/` (token files of private traces: never print or decode them; `cat281k.bin` = 281,551 tokens for 250k prompts), `ref/` (reference logits), `docs-archive/` (`kvq-cache-comparison.md` with the four caches compared, `dflash2-vram-baseline.md` with the launch line and the acceptance set; both moved out of the repository), `mem/` (one folder per work package of 2026-11-17 to 2026-11-19, each with `RESULTS.md`: `inventory`, `vram`, `phase`, `replay`, `m250`, `comb250` with the ubatch sweep `UBATCH.md`, `codes`, `attnsplit` with `QUALITY.md` and `DIAG.md`, `final` with the combined build, its binaries in `bin-final` and the measurement scripts `scripts/sess.sh`, `runmatrix.sh`, `phases.py`, `ssum.py`).

## 11. Removed relative to the prototype

Branch `kvq` (ours; the prototype is `kvq-two-tier-cache` 045ef35e5) keeps the default `--kvq` path bit for bit (same ids, slot files and kvq-eval logits as `mem/final/bin-final`) and removes the rest:

- `--kvq-tail 0` (full device cache) and every path without the host ring; `--kvq-tail` must be > 0, a context of at most N positions gets a ring of one cell per position.
- `--kvq-v-tables` (offline V tables) and the codec table files.
- The explicit two-tier mode, the fill mode with offline tables, `all_resident`, the top-up / sync modes, `ggml_kvq_attn_cond` and the fused ATTN | SCORE mode, the f16 tail and resident variants of the attention op (q8_0 only), `params.scratch`, `max_tail`, the ext API that only kvq-eval used.
- kvq-eval history modes (`--kvq-variant` family, `--kvq-two-tier`, `--kvq-packed` with offline tables, `--kvq-sync*`, `--kvq-all-resident`, `--kvq-scr-check`, `--kvq-lib-full`, `--kvq-lib-busy`, `--kvq-lib-nowait`, `--kvq-lib-vtab`); the CPU codec reference moved to `tests/kvq-ref.h`, the benchmark of test-kvq-attn to `llama-kvq-bench`.
- Debug probes: `LLAMA_KVQ_IDLE_PROBE`, `LLAMA_KVQ_GATHER_THREADS`, `LLAMA_KVQ_SCRATCH`, `LLAMA_KVQ_FIT_SYNC`, `LLAMA_KVQ_SCHED_SYNC`, `LLAMA_KVQ_UP_CHUNK`, `LLAMA_KVQ_NBUF` (now `--kvq-layer-bufs`, step 1b), `GGML_KVQ_ROWS_PER_BLOCK`, `LLAMA_SERVER_GEN_TIMES`. The server prints one `kvq:` line per request instead of four.
