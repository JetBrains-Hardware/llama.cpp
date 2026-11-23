# Inkling-2-Mini: DSpark drafter (Strata analysis and implementation plan)

Branch `erokhins/inkling-strata`, worktree `wt-inkling-strata`, HEAD `07d22ba2e` (code = `3692ecb40`, pos mask). Read `inkling-strata-analysis.md` first (sections 3, 10, 11, 11.1).
This file is analysis and plan only. No code is changed. Line numbers are valid for `07d22ba2e` (llama.cpp) and `6a6db7f` (`..\Strata`); if a line moved, search by the quoted text.

Marks used below: **Decision** = a choice I made and why. **Attention** = a place where a mistake is easy and hard to see. TODO = not verified.

## 1. Summary

Inkling-2-Mini ships its own drafter: a 3-layer block model with a recurrent Markov head (`mtp.safetensors`, 0.64 GB, block size 8). It reads four hidden states of the target ("taps") and predicts up to 7 tokens in one pass. Strata runs it by default; we measured Strata decode with it at 121.9 t/s (32k) and 114.4 t/s (76k) against 89.3 and 76.4 t/s without it (analysis 11.1): 1.36x and 1.50x. Our llama.cpp build decodes at 80.4 and 73.7 t/s and has no drafter.

Result of the analysis:

- llama.cpp already has almost everything: the DFlash/DSpark drafter family, target tap extraction on the device, bounded rollback of the recurrent state (`n_rs_seq`, Inkling2 is in `llm_arch_supports_rs_rollback`), verification and rejection sampling in the server. Only the Inkling2 variant of the drafter is missing.
- `ggml-org/llama.cpp` PR #32218 adds that variant: 692 lines, 16 files, head `9f57b4453` (2026-11-23), base `cbb1c1c50`, which is already an ancestor of our HEAD. `git apply --check` of its diff on our HEAD is clean (checked, nothing was applied).
- Unsloth publishes ready drafter GGUFs for that PR (0.50 GB Q8_0, 0.75 GB BF16). No conversion and no 20 GB checkpoint is needed.
- I compared the PR graph with the Strata kernels formula by formula (section 4): the math is the same. The differences are policy and engineering (section 5).

So the plan is not "port Strata". It is: take the PR, prove it is lossless on our branch, measure where it loses against Strata, then add the Strata ideas that the numbers ask for, smallest first (adaptive draft length, full block decode). Tree verification and Strata's int8 kernels are documented but not planned.

Corrections to `inkling-strata-analysis.md`:
- Section 11.1 says #32218 was ruled out. Its state changed: it was rewritten (7 commits on master of 2026-11-22), it is small, and it does not touch `common/sampling.cpp` or `src/llama-context.cpp` any more. Reviews on 2026-11-23: approved by ngxson and ruixiang63, changes requested by ggerganov (2026-11-21), open, not a draft.
- "loop core 6..18" means layers 6 to 17: 6 prelude + 12 x 3 + 6 coda = 48 layer visits (`ink_runner.cuh:3278-3345`, `inkling2-loop.cpp:24`).
- Strata drafts only when exactly one request is active (`ink_serve.h:763`, `n_active == 1`).

## 2. How Strata does it

Files: `engine/src/inkling2/cuda/ink_dspark.cuh` (drafter), `ink_spec.cuh` (verify and commit), `ink_runner.cuh` (taps), `ink_cuda_backend.cu` (policy), `engine/src/inkling2/ink_serve.h` (serve loop), `ink_gguf_loader.h` (drafter files).

### 2.1 Drafter model

Constants (`ink_cuda.cuh:152-168`, `ink_manifest.hpp:66-74`, same as `mtp_config` of `thinkingmachines/Inkling-2-Mini`): 3 layers, hidden 2304, 24 heads, 8 KV heads, head dim 128, FFN 9216 (SwiGLU), block 8, Markov rank 256, RoPE theta 1e6 (NeoX half-split), layers 0 and 2 local (window 1024 + anchor), layer 1 global, taps `[5, 23, 41, 48]`, logit softcap 30, vocabulary 200058 (201024 padded).

Context side, once per committed token (`ingest_taps_from`, `ink_dspark.cuh:1436`):
1. Taps: the layer input at visits 5, 23, 41 (`maybe_tap` is called before `run_visit`, `ink_runner.cuh:3263-3284`) and the final normed hidden state (`:3358`).
2. Per drafter layer `l`: `x_l = rms_norm(sum_i softmax(fuse_weight[l])[i] * tap_i) * dflare_norm_l` (`k_dspark_mix_taps`, `:52`).
3. `K = rope(rms_norm_head(k_proj_ctx(x_l)) * k_norm_ctx)`, `V = v_proj_ctx(x_l)` (no norm), written to the context KV of layer `l` at the token position (`:1448-1466`).
4. Row 3 of `fuse_weight` with `markov_head.dflare_norm` gives `f_rnn`, kept in a small ring (`:1468`).

Draft, once per round (`emit_dspark_backbone` `:1479`, `emit_draft_block8` `:1560`):
1. Block of 8 at positions `apos .. apos+7`: slot 0 = `embed_norm(embed(anchor))`, slots 1..7 = `mask_embed`.
2. 3 layers: `input_layernorm`, q/k/v, `q_norm`/`k_norm` per head + RoPE, attention over the context window and all 8 block tokens (non-causal, scale 1/sqrt(128), no sink, no bias; `k_dspark_block_attn` `:127`), `o_proj`, residual, `post_attention_layernorm`, SwiGLU, residual. Then `dspark.norm`.
3. Markov head, 7 steps, state starts at 0 (`:1572-1598`): `joint = [state, markov_w1[prev], h_s, f_rnn(apos-1)]` (5120), `p = rnn_proj(joint) + b` (768), `g = sigmoid(p0)`, `cand = tanh(p1)`, `state = cand + g*(state - cand)`, `lat = tanh(p2)`, `logits_s = softcap(unembed(h_s + hidden_out(lat) + b))`, `prev = argmax`.

Strata-only engineering inside the drafter: an int8 copy of the unembedding (230 MB) to get base logits of the 7 slots in 0.3 ms, the Markov head scores only the top 512 candidates of each slot (`K_cand`, `:254-515`), weights pre-packed to BF16 and fused (QKV, gate/up), the whole draft captured in one CUDA graph. Claimed draft time: below 0.45 ms (comment in source, not measured by us).

### 2.2 Verify and commit

- Chain: `[anchor, d1..dK]` in one target pass. The pass does not change the committed KV or the ShortConv history; K, V and the 4 conv inputs of every node go to staging buffers. After the check, one kernel commits the accepted prefix (`k_ink_commit_kv_all48`, `k_ink_commit_sconv_all48`, `ink_spec.cuh:27-95`) and the taps of the accepted nodes are ingested into the drafter (`:186-189`).
- Tree (greedy requests only): a 4-wide beam over the Markov head makes up to 12 nodes (`draft_beam_tree`, `ink_dspark.cuh:1696`; `dspark_spec_step_greedy`, `ink_cuda_backend.cu:759`). The target verifies all nodes in one pass and walks the tree greedily.

### 2.3 Policy (`ink_serve.h:762-881`)

| Case | What Strata does |
|---|---|
| more than one active request | no drafting |
| greedy, no penalties | adaptive beam tree, 3 to 12 nodes |
| any other sampling | chain of 2..7 tokens, deterministic draft, target accepts token `d` with probability `p_target(d)`, else samples the rest (`verify_draft_logits`, `:137`) |

Draft length is chosen per round (`ink_cuda_backend.cu:740-754`): `K* = argmax_K E(K) / c(K+1)` with `E(K) = 1 + sum_{k<=K} prod_{j<=k} p_j` (`p_j` = softmax probability of the drafted token over its 512 candidates) and the cost of verifying `M` tokens `c(M) = 1 + 4.2 * (1 - (15/16)^M)`. The cost model says: one more token in the batch costs extra only for the experts that no other token uses (128 experts, 8 per token).

| drafted K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Strata round cost `c(K+1)`, in single-token decodes | 1.51 | 1.74 | 1.96 | 2.16 | 2.35 | 2.53 | 2.69 |

A round with K = 7 must emit more than 2.69 tokens on average to win. This table is Strata's model for its own kernels. llama.cpp has another cost curve; it is not known yet (step 3).

### 2.4 Memory

Drafter context KV: FP16, 4 KiB per token for the global layer (grown in steps of 8192 tokens, `ensure_ctx_cap` `:1398`), two rings of about 1100 cells for the local layers, `f_rnn` ring below 5 MiB. Measured: about +1.0 GiB VRAM after generation with the drafter on (analysis 11.1).

## 3. What llama.cpp has

### 3.1 On our HEAD

| Piece | Where | State |
|---|---|---|
| DFlash drafter family (inject target features into the draft KV, block decode) | `src/models/dflash.cpp`, `common/speculative.cpp:984` | ready |
| DSpark Markov head, also an RNN variant (Mistral Large 4) | `src/models/dspark.cpp:180` (`build_dspark_markov_head`, `is_rnn` at `:192`) | ready, but no `f_rnn` input and bias is added without softcap |
| Target taps on the device, no host copy | `llama_set_embeddings_layer_inp`, `layer_inp_dev` (`src/llama-context.cpp:206`, `:1515`), `add_embd_row` (`speculative.cpp:1232-1249`) | ready; index `n_layer` (= 48) is allowed |
| Rollback of the conv state without a checkpoint | `n_rs_seq` snapshots (`llama-context.cpp:186`, `llama-memory-recurrent.cpp:102`); `llm_arch_supports_rs_rollback` lists `INKLING2` and `INKLING2_LOOP` (`llama-arch.cpp`); set from `--spec-draft-n-max` (`common/common.h:417`, `common.cpp:1748`) | ready |
| Verify, partial accept, `seq_rm`, rejection sampling, per-position acceptance stats | `tools/server/server-context.cpp:3818-3924`, `:4866-4978` | ready |
| Synthetic acceptance for timing | `--spec-synth-len`, `--spec-synth-rates` (`docs/speculative.md`) | ready |
| Inkling2 taps (`t_layer_inp` by visit, final norm) | `src/models/inkling2.cpp`, `inkling2-loop.cpp:76` | missing |
| Inkling2 drafter: per-layer context K/V from mixed taps, `f_rnn`, softcap after the bias | - | missing |

### 3.2 PR #32218 at `9f57b4453` (local ref `ggml-org/pwilkin/inkling2-dspark`)

Commits `6ed422257 .. 9f57b4453` on `cbb1c1c50`. What it adds:
- Converter `Inkling2DSparkModel` (`conversion/inkling2.py`, `--dspark`): arch `dflash`; `fuse_weight` is written as softmax rows (`blk.N.ctx_mix`, `markov_ctx_mix`); `markov_w2 = unembed @ hidden_out` and its bias are folded in fp64 (the head becomes the usual DSpark "bias on logits"); `mask_embed` is copied to 7 rows; `sliding_window = 1025`, pattern from `local_layer_ids`; `sample_from_anchor = false`, `has_confidence_head = false`, `has_markov_context = true`. The token embedding and the lm head are not in the file: the drafter uses the target's (`get_tok_embd`, `get_lm_head`, `dflash.cpp:239-260`).
- Target: `res->t_layer_inp[visit] = cur` before each `build_layer`, `t_layer_inp[n_layer]` = normed output (`inkling2.cpp`, `inkling2-loop.cpp`, 13 lines).
- Draft graph: `build_inject_ctx_mix` (per-layer mix, norm, `k_ctx`/`v_ctx`, K norm + RoPE; outputs `f_rnn` as `h_nextn`), `build_dspark_inp_block` (normed anchor + mask rows), `f_rnn` appended to `joint`, softcap after the bias, optional in-graph sampling of the chain at the request temperature (`draw`).
- Driver (`common/speculative.cpp`, +146): keeps the `f_rnn` rows of the last 9 positions per sequence on the host (`store_context_features`), passes the row of position `pos0 - 1` back on the anchor token (`set_embd_state`), reads the drawn tokens and their distribution for rejection sampling.

Not in the PR: tests for the Inkling2 drafter, any change in ggml or the server.

### 3.3 Drafter files (checked 2026-11-23 through the Hugging Face API)

| File (`unsloth/Inkling-2-Mini-GGUF`) | Size | Format |
|---|---|---|
| `Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file` | 0.50 GB | PR #32218 (their README names `e4b1cb531`) |
| `dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file` | 0.75 GB | same, projections bit exact |
| `dspark-Inkling-2-Mini-Q8_0.gguf`, `dspark/dspark-Inkling-2-Mini-BF16.gguf` | 0.40 / 0.65 GB | old Unsloth format (padded `markov_w1`, `dspark_hidden_out`); Strata reads it (`ink_gguf_loader.h:742`), the PR rejects it |

Source: `thinkingmachines/Inkling-2-Mini` has `mtp.safetensors` (0.64 GB) next to a 19.5 GB model file; the converter needs both, so converting ourselves costs a 20 GB download. Not needed.

**Attention.** The Unsloth files were made for PR commit `e4b1cb531`. The head is 3 commits later. One of them (`3dd164fcd`) makes the loader read the key `dflash.has_markov_context` instead of looking for the tensor `markov_ctx_norm.weight`. The converter at `e4b1cb531` already wrote that key, so the files should load, but this is TODO until step 0.2 checks it.

## 4. Math check: PR graph against Strata kernels

| Item | Strata (`ink_dspark.cuh`) | PR #32218 | Same |
|---|---|---|---|
| taps | layer input at visits 5, 23, 41; normed output | `t_layer_inp[visit]`, `t_layer_inp[n_layer]`; ids from GGUF | yes |
| tap mix | `softmax(fuse_weight[l])`, RMS norm (eps 1e-6) * `dflare_norm` | softmax done by the converter, `ctx_mix` matmul, `build_norm` | yes |
| context K/V | `k_proj_ctx` -> per-head norm * `k_norm_ctx` -> RoPE; `v_proj_ctx` raw | `dflash_wk_ctx` -> `attn_k_ctx_norm` -> `build_rope`; `dflash_wv_ctx` raw | yes |
| block input | `embed_norm(embed(anchor))`, 7 x `mask_embed` | `tok_norm` on the anchor, `mask_embd` rows | yes |
| layers | pre-norm, q/k norm + RoPE, non-causal, SwiGLU | DFlash base layers (`build_layers`) | yes |
| local window | positions `apos-1024 .. apos-1`, then the block | `n_swa = 1025`, whole block uses the mask row of the anchor (`block_uniform`) | yes |
| `f_rnn` | fuse row 3, ring, read at `apos-1` | `markov_ctx_mix`, host ring, read at `pos0-1` | yes |
| `joint` order | state, `w1[prev]`, hidden, `f_rnn` (`:589-600`) | state, `w1_prev`, `hidden_i`, `f_rnn` | yes |
| state update | `cand + g*(state - cand)` | same expression (`dspark.cpp:261`) | yes |
| logits | `softcap(unembed . (h + hidden_out(lat) + b))` | `softcap(lm_head(h) + w2 . lat + w2_b)` | yes, see note |
| candidates | top 512 of the base logits per slot | full vocabulary | PR is exact |
| chain | argmax | argmax, or a draw at the request temperature | PR has more |

Note on the logits: the PR folds `w2` with the original BF16 unembedding but takes the base logits from the target's quantized lm head. Strata uses one unembedding for both. The difference only moves draft scores a little; it cannot change the output of the target.

**Decision.** No reason to write our own drafter graph or to port Strata kernels for correctness. What is left to compare is behavior, section 5.

## 5. Differences that can cost speed

| # | Topic | Strata | llama.cpp + PR | Plan |
|---|---|---|---|---|
| F1 | cost of verifying K+1 tokens | one pass, cost model of 2.3 | unknown. Batch 1 uses the `vec` attention kernel, batch > 1 the `mma` kernel; MoE runs `mul_mat_id` per token and expert | measure first (step 3) |
| F2 | draft length | adaptive per round from the drafter's own probabilities | fixed `--spec-draft-n-max`; `--spec-draft-p-min` throws without a confidence head (`speculative.cpp:1395`); the PR docs and Unsloth use `n-max 3` | step 4 |
| F3 | block size at draft time | always the trained 8, then cut | `n_max + 1` tokens since `22a56e3e5` (before: full block) | measure, then step 5 |
| F4 | verification shape | tree for greedy | chain only | not planned (step 8) |
| F5 | draft cost | below 0.45 ms claimed | lm head on 8 rows over 200k tokens, 7 chained full-vocabulary steps, `f_rnn` over the host | measure, step 6 only if it matters |
| F6 | sampled requests | deterministic draft, accept with `p_target(d)` | same by default; `--spec-draft-sampling probabilistic` draws the draft and accepts with `min(1, p/q)` | measure both |
| F7 | prefill | taps go straight into the drafter | one draft decode per target ubatch, and `h_nextn` (n_ubatch x 2304 floats) is read back to the host each time (`nextn_masked = false`) | measure |
| F8 | VRAM | grows with use | draft context is allocated for the full `n_ctx` at load: 4 KiB per token for layer 1 | measure, existing flags only (step 7) |
| F9 | parallel requests | off | one block per slot in one draft decode | check only |

Expected extra VRAM at `-c 98304`, to confirm in step 3: weights 0.50 GB (Q8_0), draft KV 384 MiB + about 12 MiB, target conv snapshots 8 x 3.7 MiB (`48 * 6656 * 3 * 4 B` each), tap buffer `4 * 2304 * n_batch * 4 B` = 72 MiB at `-b 2048`, draft compute buffer unknown. About 1.0 to 1.2 GiB.

## 6. Decisions

**D1. Base on PR #32218 at `9f57b4453`, applied as one diff.** Same pattern as hridaya423 `68cd7533a` for the bias. It is reviewed by the people who own this code, it follows the upstream contract, and the drafter files exist for it. Our own implementation would be 700 lines that diverge from upstream on the day it merges.

**D2. Drafter file: BF16 mainline for the gates, Q8_0 for the numbers.** BF16 removes one variable while we check correctness. Unsloth reports the same acceptance for both; step 3 confirms it on our requests.

**D3. Losslessness is proven by teacher forcing, not by comparing text.** With drafts the target verifies in batches, without drafts it decodes one token. These two paths use different attention kernels, so logits differ in the last bits and a near tie can flip (the same effect made `-np 2` concurrent output differ from single output, analysis section 10). The gate therefore is: every token of a speculative run must be the top-1 of the baseline binary on the same prefix, except documented near ties.

**D4. Costs are measured with synthetic acceptance, not taken from Strata.** `--spec-synth-len` gives the round time for each K with the real drafter and the real verify batch.

**D5. First own change is the draft length rule (F2), in the driver only.** It is the cheapest Strata idea and the one the PR authors worked around with `n-max 3`. Prototype switch is an environment variable, like `LLAMA_ATTN_POS_MASK_DISABLE`; a real option only after it proves a gain and the user wants it.

**D6. No tree verification.** In llama.cpp it needs tree positions and ancestor masks in the batch and KV cache, a conv history per branch, a commit of one path, and server changes. The pos mask of this branch cannot express a tree (two nodes at one position with different ancestors), so the dense mask would come back for verify batches. This is a new subsystem, against `AGENTS.md`. Step 3 collects one number that says how much a tree could give (M8); the decision can be reopened with it.

**D7. No port of the int8 screening plane, `K_cand` scoring or CUDA graph capture of the draft.** They cut a cost (F5) that is not measured yet, and they are kernel work in ggml. Step 6 lists smaller options.

**D8. The drafter keeps the standard dense mask.** The drafter is arch `dflash`, it does not use the rel-bias attention input, so the pos mask does not apply to it. Its mask is `[n_kv, n_tokens]` F16; a block has at most 8 tokens per slot.

## 7. Rules for the agents that execute this

- Follow `AGENTS.md`: ASCII only, short comments, no wrapping to a column, code must look like the code around it. Read `skills/code-review/SKILL.md` before writing code.
- No commit, no push, no PR, no comment on #32218. Stop after the gate of each step and report. The user commits.
- GPU work only through `GPU_OWNER=inkling-dspark bash tmp/inkling/tools/gpurun.sh ...`. One test run is at most 5 minutes (`timeout 300`). Long jobs (build, download) go to the background; no foreground sleep loops, no duplicate runs.
- `tmp\inkling` is shared with `wt-inkling`. Add files (`tmp/inkling/logs/dspark-s<N>`), do not change existing scripts in a way that breaks the other worktree.
- Stop servers by their real Windows PID (`ps` column `WINPID` or a command line match), not by `$!` of Git Bash (analysis section 11).
- If a gate fails: stop and report the numbers. Do not relax the gate, do not explain the difference away.
- Do not touch: ggml, `llama-graph.cpp`, `llama-kv-cache.cpp`, the server, the pos mask. Steps 1 to 5 need none of them.

## 8. Steps

### Step 0. Files and baseline (no source change)

1. Download to `..\models\inkling\` and rename to `.gguf`: `dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file` and `Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file` from `unsloth/Inkling-2-Mini-GGUF` (1.25 GB together, background job).
2. Print metadata and tensors with `tmp/inkling/tools/ginfo.py`. Expected: `general.architecture = dflash`, `dflash.block_size = 8`, `dflash.target_layers = [5, 23, 41, 48]`, `dflash.sample_from_anchor = false`, `dflash.has_confidence_head = false`, `dflash.has_markov_context = true`, sliding window 1025 with pattern `[true, false, true]`, tensors `markov_w1.weight [256, 200058]`, `markov_w2.weight`, `markov_w2.bias`, `joint_proj.weight [5120, 768]`, `markov_ctx_mix.weight [4]`, `markov_ctx_norm.weight`, `mask_embd.weight [2304, 7]`, `token_embd_norm.weight`, and per layer `ctx_mix`, `ctx_norm`, `attn_k_ctx`, `attn_v_ctx`, `attn_k_ctx_norm`. No `token_embd.weight`, no `output.weight`. If `has_markov_context` is missing: stop and report (options: add the key with gguf-py, or convert from the 20 GB checkpoint).
3. Teacher binary: copy `build-cuda\bin` (exe and dll) to `tmp\inkling\baseline-3692ecb40\bin` before any edit.
4. Baseline decode on the day: server line of `AGENTS.local.md`, `replay.py req_32k.json 4000 3` and `req_76k.json 4000 2`. Reference: 80.4 and 73.7 t/s, prefill 2726 and 2490 t/s, 17080 MiB after load.
5. Greedy references for later: both requests and prompts A and B of analysis section 10 with `temperature 0`, `max_tokens 300`; keep the texts.

### Step 1. Apply PR #32218

```
git fetch ggml-org pwilkin/inkling2-dspark
git rev-parse ggml-org/pwilkin/inkling2-dspark   # must print 9f57b4453570c6a5fef01fec8e65a44d783a63e0
git diff cbb1c1c50 9f57b4453 > tmp/pr32218.diff
git apply --3way tmp/pr32218.diff
```

If the head moved: stop and report the new commits; do not take a newer head without a look at what changed (the GGUF format changed once already). Build `llama-server`.

**Attention.**
- The patch is not ours. Read it before the build: `src/models/dspark.cpp` (head), `src/models/dflash.cpp` (`build_inject_ctx_mix`), `common/speculative.cpp` (`store_context_features`, `draft`).
- `docs/speculative.md` gets a paragraph from the PR; keep it.
- The drafter must be on the same device as the target: it reads the target's embedding, lm head and tap buffer. Use `-ngld 99`, no `-devd`.

#### Gate 1

1. No drafter given: `teacher_force.py --teacher-exe tmp\inkling\baseline-3692ecb40\bin\llama-server.exe --teacher-fa on` on the 32k request (`--ctx 40960`) and on the default prompt: 100/100 and max `|d_logprob|` 0. The patch only stores pointers in `t_layer_inp`, so anything else is a finding.
2. Compute buffers at `-c 131072 -ub 512 -lv 4` without a drafter are equal to analysis section 10 (129.75 / 9.53 MiB).
3. With the drafter: `-md <BF16 drafter> --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99`. The log must show: `DFlash with DSpark markov head (rank = 256)`, `DFlash extract_layers = [5, 23, 41, 48]`, `sample_from_anchor=false, has_markov_context=true`, `DFlash target features via device buffer`, the `n_rs_seq` line of the target context with the value 7, and no `speculative decoding not supported by this context`.
4. One short greedy request: the statistics line shows accepted tokens above 0.

### Step 2. Lossless gate

Use the tool with the roles swapped: the new binary with the drafter generates the greedy trajectory (teacher role), the baseline binary predicts one token at a time on each prefix (student role).

```
python tmp\inkling\tools\teacher_force.py --teacher-exe <abs path>\build-cuda\bin\llama-server.exe --teacher-fa on --teacher-extra "-md <drafter> --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99" --student-exe <abs path>\tmp\inkling\baseline-3692ecb40\bin\llama-server.exe --student-fa on --n-tokens 300
```

| Run | Prompt | Tokens | Pass |
|---|---|---|---|
| 1 | default 2.7k prompt | 300 | top-1 300/300, or every miss is a near tie |
| 2 | 32k request (`--request ... --ctx 40960`) | 100 | same |
| 3 | run 1 with `--spec-draft-n-max 3` | 300 | same |
| 4 | run 1 with `LLAMA_ATTN_POS_MASK_DISABLE=1` | 300 | same |
| 5 | run 1 with `--ngl 0`, 20 tokens, short `--prompt-text` | 20 | same |

Near tie: the baseline gives the generated token a logprob within 0.05 of its own top-1. Report every miss with both logprobs. If a run does not fit in 5 minutes, split it with `--n-tokens`; do not shorten the total. Also report the server statistics of each run (rounds, accepted per position); a run with fewer than 20 rejected rounds does not test the rollback, repeat it with another prompt.

**Attention.**
- A broken rollback does not crash. The conv state or a KV cell of a rejected token stays, and every later token is slightly wrong. It shows as many misses and a growing `|d_logprob|` after the first rejected round, not as one flipped token. Do not judge by reading the text.
- A wrong drafter (bad taps, wrong mix) cannot fail this gate. It only lowers acceptance. Acceptance is judged in step 3.
- Run 4 separates the two features: if 1 fails and 4 passes, the pos mask is wrong for verify batches (`kpos` of removed cells must be `INT32_MAX`); if both fail, look at the rollback.
- If the tool cannot swap the roles as written (check `--teacher-extra` reaches only the teacher server), fix the tool in a backward compatible way and say so in the report.

Not covered, state it in the results: `-np 2`, context shift and `--cache-reuse` (off for Inkling without `--swa-full`), slot save and restore, images.

### Step 3. Measurements (milestone, ask the user before the long runs)

Server: `-c 98304 -ub 512 -np 1 -fa on -ngl 99 --jinja` plus the drafter flags, one session per request and configuration, nonce'd client `..\kvq-scratch\strata-bench\bench_client.py`. Write all numbers into `inkling-strata-analysis.md` (new section 12).

- **M1, round cost.** For K = 1..7: `--spec-draft-n-max K --spec-synth-len <K+1>` (every draft accepted). `t_round(K) = (K+1) / decode_tps`; cost in single-token decodes `c(K) = t_round(K) * baseline_tps`. Add `ignore_eos` and a fixed `max_tokens` (the text is not valid model output, only the time counts). Repeat K = 3 and 7 with `--spec-synth-len 1` (nothing accepted) as a cross-check of the same `t_round`, now with rollback. Both requests. Also take the draft time from the `statistics` line (`dur(b,g,a)`).

  | K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|
  | Strata model | 1.51 | 1.74 | 1.96 | 2.16 | 2.35 | 2.53 | 2.69 |
  | llama.cpp 32k | | | | | | | |
  | llama.cpp 76k | | | | | | | |
  | of that, draft ms | | | | | | | |

- **M2, acceptance.** `--spec-draft-n-max 7`, real verification, both requests and one code prompt, greedy and the request's own sampling: acceptance per position (printed at slot release), mean accepted length.
- **M3, speed.** Decode t/s for K in {1, 2, 3, 5, 7}, for `--spec-draft-sampling greedy` and `probabilistic`, at `temperature 0` and with the request's sampling. Compare with step 0.4 and with Strata (121.9 / 114.4 t/s).
- **M4, block size (F3).** Acceptance at positions 1..3 with `n-max 3` against positions 1..3 of the `n-max 7` run, greedy, same prompts.
- **M5, drafter quant.** M2 greedy with the Q8_0 file.
- **M6, VRAM.** After load and after generation, `-lv 4` reserve lines and KV sizes of both contexts, against the estimate in section 5.
- **M7, prefill (F7).** Prefill t/s with the drafter loaded against step 0.4.
- **M8, tree potential (F4).** With `-lv` debug the driver prints the top 3 draft candidates per position. Count the rejected rounds where the target token is candidate 2 or 3 at the first rejected position.

Decision table:

| Finding | Next |
|---|---|
| best fixed K gives 1.2x or more | step 4, then report |
| M4 shows 2 points or more for the full block | step 5 before step 4 |
| draft is more than 10% of `t_round` at the best K | step 6 |
| `c(K)` far above the Strata row and speedup below 1.1x | stop. The loss is in the target's small-batch decode (F1). Report the table; that is ggml kernel work and needs its own analysis |
| acceptance at position 1 below 50% on the code prompt | suspect the drafter path; compare taps and `f_rnn` with a dump before any tuning |
| prefill loss above 3% | report; candidates are a larger `--spec-draft-n-ubatch` and reading `h_nextn` only for the last 9 rows of a chunk |

### Step 4. Draft length rule (F2)

File: `common/speculative.cpp`, `common_speculative_impl_draft_dspark` only. Two parts, measure each against the best fixed K of M3.

4a. Let `--spec-draft-p-min` work without a confidence head. The base DFlash driver already cuts on the top-1 probability of the sampler (`cur_p->data[0].p < params.p_min`, `speculative.cpp:1352`). Do the same in the DSpark loop when the draft has no confidence head, and throw only if it has neither a confidence head nor a Markov context.

**Decision (2026-11-23, user, replaces the per-round rule below): windowed K.** Reasons: ggml-org PR #32773 measured that every change of the verify batch size rebuilds and re-allocates the target graph (14 to 20 ms on Vulkan; their per-round 1..4 rule ran 47 t/s against 80 for fixed 4), and section 13.6 shows that acceptance is close to geometric (conditional rate 0.66 to 0.72 at positions 1..6 on 32k, 0.70 to 0.80 on 76k), so one number per window is enough.
- Load with `--spec-draft-n-max 7` (sizes `n_rs_seq`); the rule picks the K actually drafted, 1..7.
- Per sequence, over a window of W rounds (about 50 to 100): `a = accepted / (accepted + rejected rounds)` (MLE of a geometric acceptance; it is measurable at any K, so no exploration rounds).
- At the window end: `K = argmax_K E(K) / c(K)` with `E(K) = 1 + a + ... + a^K` and `c` from `LLAMA_SPEC_DRAFT_COST` (unset = rule off). Change K only if the predicted gain is above a few percent (hysteresis).
- K changes once per window, so a graph rebuild is rare.
- Check on 13.6 data: 32k a = 0.68 gives K = 2 (measured best 2), 76k a = 0.74 gives K = 5..7, flat (measured best 7).
- Do it after the drafter speed work (step 6), because `c` depends on the draft time; measure `c` once after that, then run the two traces only.

4b (superseded, kept for reference). Strata rule. In the read loop keep `p_k` = top-1 probability of position k. After the loop:

```cpp
// cost[k]: time of a round with k draft tokens, in single-token decodes
double p = 1.0, e = 1.0, best = 1.0;
size_t n_keep = 0;
for (size_t k = 1; k <= result.size(); ++k) {
    p *= p_top[k - 1];
    e += p;
    if (e / cost[k] > best) {
        best   = e / cost[k];
        n_keep = k;
    }
}
result.resize(n_keep);
```

`cost` comes from M1 through `LLAMA_SPEC_DRAFT_COST` (comma list for k = 1..7, read once, unset = rule off). `best` starts at 1.0, the rate of a round without a draft, so the rule can also choose no draft.

**Attention.**
- `p_top` is a softmax over the 10 candidates of the draft sampler, so it is too high when the distribution is flat. Check the calibration with M2 before tuning: predicted `prod p_j` against measured acceptance per position.
- For greedy requests the true acceptance is "the target has the same argmax", which is higher than `p`. One cost table cannot serve both; if the rule loses against fixed K on greedy requests, report it, do not add a second knob.
- On the probabilistic path (`sampled[seq_id]`) the token and its distribution come from `h_nextn`; cut `result` and `dp.result_q` together or the server assert `spec_draft_q.size() == spec_draft.size()` fires.
- `--spec-draft-n-min` still applies after the cut.

Gate 4: step 2 runs 1 and 2 pass with the rule on; decode t/s on both requests is not below the best fixed K (noise 3%).

### Step 5 (only if M4 says so). Decode the full trained block

Commit `22a56e3e5` of the PR changed the block from the trained size to `n_max + 1`. Undo it for drafts with a Markov context: in `draft()` use `block_size` tokens and read the first `n_max`, and restore the `n_outputs_max` hunk next to `cparams.n_rs_seq = 0` (`speculative.cpp:3123`). `git show 22a56e3e5` is the exact reverse of what is needed. Gate: M4 numbers reproduce at `n-max 3`, step 2 run 3 passes.

### Step 6 (only if the draft is more than 10% of a round, ask the user). Draft cost

Candidates, smallest first. Measure each alone.
1. Skip the anchor row in the lm head: `build_logits` runs on all 8 rows, row 0 is never read (`sample_from_anchor = false`).
2. Keep `f_rnn` on the device: today it goes device -> host ring -> `embd_state` -> device.
3. Limit the Markov steps to the top candidates of the base logits (Strata `K_cand`): `ggml_top_k` exists and the PR already uses it in `draw`.
4. Strata's int8 screening copy of the unembedding. Kernel work, 230 MB VRAM. Last.

### Step 7 (measurements only). Drafter memory

No code. Try what exists: `-ctkd q8_0 -ctvd q8_0` (halves the 4 KiB per token; check acceptance with M2), `--spec-draft-n-ubatch` (draft compute buffer against prefill speed, M7). Report VRAM and acceptance for each.

### Step 8 (not planned). Tree verification

See D6. Reopen only if M8 shows that candidate 2 or 3 would have matched in a large share of the rejected rounds and the user asks for it. It then needs its own plan.

## 9. Review checklist

- Step 1: the applied diff is byte equal to `git diff cbb1c1c50 9f57b4453`; no other file changed; the pos mask files are untouched.
- Step 2: roles are really swapped (the drafter flags are in the teacher's server log, not in the student's); each run has at least 20 rejected rounds; misses are listed with logprobs.
- Step 3: every table says which drafter file, K, sampling and request; synthetic runs are marked as not valid output.
- Step 4: the change is inside `common_speculative_impl_draft_dspark`; `result` and `result_q` have equal length; the rule is off when the variable is unset; comments short and ASCII.

## 10. Known limits and risks

- #32218 is not merged and has a "changes requested" review. Its GGUF format changed once and can change again; then the Unsloth files and our applied diff go stale together. Pin both (commit `9f57b4453`, file sizes and dates in the results).
- Only CUDA and CPU are correct for Inkling-2 on this branch (analysis section 9). The drafter does not change that.
- Exact equality with a run without drafts is not reachable in llama.cpp: verify batches and single-token decode use different kernels. Strata gets bitwise parity because its verify pass is built for it. We prove top-1 agreement up to near ties instead (D3).
- The target must keep `n_rs_seq = n_max` conv snapshots; a draft longer than that falls back to a full checkpoint in the server (`server-context.cpp:3899`). `--spec-draft-n-max` at load time is therefore the hard upper limit of the round.
- Real Junie requests rarely generate much (analysis: a handful of 190 above 600 tokens), and prefill dominates their time. A decode gain of 1.3x moves the end-to-end time of such a request by a few percent. Check M7 first: a prefill loss can cost more than the drafter wins.
- Strata's 1.36x and 1.50x include its adaptive length and, for greedy requests, the tree. The sampling of our two test requests on Strata is not known (no temperature in the request; TODO), so those numbers are a target, not a prediction.
