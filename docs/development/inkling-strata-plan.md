# Inkling-2-Mini: O(1) attention compute buffer (implementation plan)

Branch `erokhins/inkling-strata`, worktree `wt-inkling-strata`, base `f929fd85c`. Read `inkling-strata-analysis.md` first.
This file is the plan only. No code is changed yet. Line numbers are valid for `f929fd85c`; if a line moved, search by the quoted text.

Marks used below: **Decision** = a choice I made and why. **Attention** = a place where a mistake is easy and hard to see.

## 1. What is left to do, in one paragraph

After hridaya423 `68cd7533a` the only compute tensor that grows with the context is the KQ mask of the full-attention layers:
F16 `[n_kv, n_ubatch, 1, n_stream]` (`build_attn_inp_kq_mask`, `src/llama-graph.cpp:46`). At reserve time `n_kv = n_ctx`.
The mask is an input tensor, so it lives two times: in the host compute buffer (it is filled on the CPU,
`set_input_attn_mask` asserts a host buffer) and as a copy in the CUDA compute buffer. It is also filled and uploaded for every ubatch.
Strata has no mask tensor: its kernels decide by index math, `ok = (kp <= qp) && (!is_swa || qp - kp < win)`
(`..\Strata\engine\src\inkling2\cuda\ink_attn.cuh:1025`, `:385-386`, `:425`).
We do the same inside the existing ggml FlashAttention kernels: the kernels already read `qpos` and `kpos` for the bias, so the mask
can come from the same two reads. Name in this plan: **positional mask** (pos mask).

Expected gain (to confirm in step 0): `n_ctx * n_ubatch * 2` bytes of VRAM and the same amount of host RAM.

| `-c` | `-ub` | mask today |
|---|---|---|
| 98304 | 512 | 96 MiB |
| 131072 | 512 | 128 MiB |
| 262144 | 2048 | 1024 MiB |
| 1048576 | 512 | 1024 MiB |

What stays after the change: `kpos`, I32 `[n_kv]`, 4 bytes per context token (0.5 MiB at 131k). See decision D3.

Corrections to the analysis doc found while reading the code:
- Head dim is 128, not 64 (`kInkHeadDim = 128`, `ink_cuda.cuh:140`; the backend test uses `hs = 128, nh = 8, gqa = 3`). No effect on the plan.
- The mask costs host RAM too, not only VRAM (see above).
- Other backends (Vulkan, Metal, SYCL, OpenCL, ...) do not check `src[6]` of `GGML_OP_FLASH_ATTN_EXT`. They accept the node and ignore the bias. This is already true at HEAD. See step 5.

## 2. Design

### 2.1 ggml contract (small addition to the hridaya423 API)

One new setter, one relaxed assert. No new op, no new `src` slot.

```c
// mask by position, needs ggml_flash_attn_ext_add_rel_bias(): cell k is visible to query q only if
//   0 <= qpos[q] - kpos[k] < n_window
// set kpos to INT32_MAX for the cells that no query may see. the mask is optional then
GGML_API void ggml_flash_attn_ext_set_pos_window(struct ggml_tensor * a, int32_t n_window);
```

- Stored in `op_params` i32 slot 5. Slots in use by this op: 0..2 floats (scale, max_bias, logit_softcap), 3 precision, 4 `n_kv_max`. Slot 5 is free. 0 means off (today's behavior).
- `n_window = INT32_MAX` is "causal only" (full-attention layers). `n_window = n_swa` gives the standard sliding window (`is_masked_swa`: masked if `p1 - p0 >= n_swa`, `src/llama-hparams.h:572`).
- The mask tensor stays legal together with the window (both are applied). llama never passes both.
- Masked cells get `-INFINITY`, exactly what the F16 mask gives today, so the softmax code is not touched.

### 2.2 llama side

- Decide per graph: pos mask or dense mask. In pos mask mode the two mask tensors of `llm_graph_input_mem_hybrid_iswa_rel` are not created at all.
- `kpos` carries everything the mask knew about a cell except the distance: empty cell or cell of another sequence -> `INT32_MAX`.
- Dense mask path stays as the fallback. It is needed for: `-fa off`, non-causal attention, unified KV cache with more than one sequence in the ubatch.

### 2.3 Decisions

**D1. Mask inside the existing bias lookup, not a new kernel path.** `fattn_rel_bias_get` / `fattn_rel_bias_tile_get` already compute `d = qpos - kpos` per element. Returning `-INFINITY` there costs two compares and no extra memory read. Alternative (synthesize the F16 mask tile in shared memory from positions) keeps more code unchanged but adds a load stage with cp_async variants (4 code paths in `flash_attn_ext_f16_load_mask`). Rejected: more code, more ways to break.

**D2. Both caches (full and SWA) use the pos mask.** The SWA mask is small (1536 x 512 x 2 B) so SWA-only gain is zero, but one mode for the whole graph is simpler than "full layers pos mask, SWA layers dense mask", and `--swa-full` (SWA cache = full size) is covered for free.

**D3. Keep `kpos` (4 B per token). Do not assume contiguous cells.** Strata gets 0 B per token because position = cell index (single sequence, no holes). llama.cpp cells can have holes, shifted positions (context shift, `--cache-reuse`), several sequences. With `kpos` all of these work with no special case. 4 B per token is 0.5 MiB at 131k. Not worth a second code path.

**D4. Fallback instead of support for "unified cache + several sequences in one ubatch".** There `n_stream = 1` and queries of different sequences share one `kpos` row, so sequence membership cannot be put into `kpos`. Strata and unsloth #386 do not support this case either. Effect: with `-np N` and a unified cache the reserved compute buffer keeps the mask (worst case graph has N sequences). `-np 1` and `-np N --no-kv-unified` get the full gain. Junie uses `-np 1` (`kv_unified = 'false'` in the server logs).

**D5. No new fused-op probe.** The pos mask is implemented in exactly the kernels that implement the bias (CPU reference, CUDA vec, CUDA mma). Where the bias node is supported, the window is supported.

**D6. Debug switch `LLAMA_ATTN_POS_MASK_DISABLE` (env).** Same style as `LLAMA_KV_ROWS_DISABLE`. It lets one binary run both paths. The correctness gates depend on it.

**D7. No port of the Strata cuBLAS tiled prefill.** ggml kernels are already tiled; the tensor that scaled was the mask, not the scores.

**D8. KV-tile skipping from positions is optional (step 4).** Today `launch_fattn` scans the mask to skip trailing all-masked KV tiles, only if `n_q >= 1024` or `n_stream > 1` (`fattn-common.cuh:1226`). With `-ub 512 -np 1` (our case) the scan is already off, so nothing is lost. Do step 4 only if step 3 measures a loss.

## 3. Rules for the agents that execute this

- Follow `AGENTS.md`: ASCII only, short comments, no line wrapping to a column, code must look like the code around it. Read `skills/code-review/SKILL.md` before writing code.
- No commit, no push, no PR. Stop after the gate of each step and report. The user commits.
- GPU work only through `GPU_OWNER=inkling-strata bash tmp/inkling/tools/gpurun.sh ...`. One test run is at most 5 minutes (`timeout 300`). Long jobs (first build) go to the background; no foreground sleep loops.
- `tmp\inkling` is a junction shared with `wt-inkling`. Add files, do not change existing scripts in a way that breaks the other worktree.
- If a gate fails: stop and report the numbers. Do not relax the gate, do not explain the difference away.
- Do not touch: `fattn-tile.cuh` logic, kernel selection in `fattn.cu`, `set_input_attn_mask`, the non-FA path, GQA settings (`ncols2` must stay 1 with a bias), other models.

## 4. Steps

### Step 0. Baseline (no source change)

1. Build: `cmd /c "tmp\inkling\tools\build-inkling.bat llama-server test-backend-ops"` (first run configures `build-cuda`, run it in the background).
2. Copy `build-cuda\bin` to `tmp\inkling\baseline-f929fd85c\bin` (exe and dll files). This is the teacher binary for all later gates. Do this before any edit.
3. `test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p extent=` must pass (40 cases).
4. Record the reserve numbers. Start the server with `-c <ctx> -ub <ub> -np 1 -fa on -ngl 99` and read the lines `sched_reserve: ... compute buffer size = ... MiB` (`src/llama-context.cpp:920`, one line for CUDA0, one for the host). `session.sh <ctx>` does this for `-ub 512` and writes `server.log` in the current directory.

   | ctx | ub | CUDA0 MiB | host MiB |
   |---|---|---|---|
   | 32768 | 512 | | |
   | 131072 | 512 | | |
   | 262144 | 512 | | |
   | 131072 | 2048 | | |

   Expected slope: about 1 KiB per context token at `-ub 512`, on both buffers. If the slope is much larger, something else scales too: stop and report, the plan is then incomplete.
5. Record prefill and decode speed: `replay.py req_32k.json 4000 3` with the server line from `AGENTS.local.md` (`-c 98304`).

Write the numbers into `inkling-strata-analysis.md` (new section "10. Results").

### Step 1. ggml: API, CPU reference, CUDA kernels, tests

One step, four parts, one gate at the end (test-backend-ops compares CUDA with the CPU reference, so the parts cannot be gated alone). The build must pass after each part.

#### 1a. `ggml/include/ggml.h`, `ggml/src/ggml.c`

- `ggml.h:2295-2305`: change the comment line "the mask must not have a head dimension" to "the mask is optional with ggml_flash_attn_ext_set_pos_window(), it must not have a head dimension". Add the declaration from 2.1 after `ggml_flash_attn_ext_add_rel_bias`.
- `ggml.c:5560`: `GGML_ASSERT(mask && mask->ne[2] == 1);` -> `GGML_ASSERT(!mask || mask->ne[2] == 1);`
- `ggml.c`, after `ggml_flash_attn_ext_add_rel_bias`:

```c
void ggml_flash_attn_ext_set_pos_window(
        struct ggml_tensor * a,
        int32_t              n_window) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_ASSERT(a->src[6] != NULL);
    GGML_ASSERT(n_window > 0);

    ggml_set_op_params_i32(a, 5, n_window);
}
```

**Attention.** The setter must be called after `add_rel_bias` (it asserts `src[6]`). Mask NULL plus window 0 is legal for the op and means "every cell is visible"; llama must never build that (guard in step 2).

#### 1b. CPU reference, `ggml/src/ggml-cpu/ops.cpp`

In `ggml_compute_forward_flash_attn_ext_f16_one_chunk`:
- after line 8987: `const int32_t rel_window = rel_table ? ggml_get_op_params_i32(dst, 5) : 0;`
- lines 9105-9108 become:

```cpp
            if (rb) {
                const int64_t d = int64_t(qp) - kp[ic];
                if (rel_window > 0 && (d < 0 || d >= rel_window)) {
                    continue;
                }
                mv += rb[d >= 0 && d < rel_extent ? d : rel_extent];
            }
```

Nothing else: `use_ref` is already forced when `src[6]` is set (`ops.cpp:9616`), so the tiled and split-KV paths are never used with a bias.

#### 1c. CUDA, `ggml/src/ggml-cuda/`

`fattn-common.cuh`:
- `struct fattn_rel_bias` (line 21): add last member `int window; // > 0: mask by position`.
- `fattn_rel_bias_get` (line 28): after `d` is computed add `if (rb.window > 0 && (d < 0 || d >= rb.window)) { return -INFINITY; }`.
- `struct fattn_rel_bias_tile` (line 34): add last member `int window;`. In `fattn_rel_bias_tile_make` add `rb.window` as the last initializer.
- `fattn_rel_bias_tile_get` (line 55): keep the `k >= t.n_kv` early return and the `q` clamp first, then the same window check after `d`.
- `launch_fattn` (line 1094): add a fifth initializer `rel_table ? ggml_get_op_params_i32(KQV, 5) : 0`. Move the initializer below the line `ggml_tensor * KQV = dst;` or use `dst`.

`fattn-vec.cuh:304-309`: today the bias is added only inside `if (mask && ...)`. New shape:

```cpp
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    if (mask) {
                        sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                    }
                    if (rel_bias.table) {
                        sum += fattn_rel_bias_get(rel_bias, sequence, head, ne02, ic0 + j, int(ne01.z), k_VKQ_0 + i_KQ, ne11);
                    }
                }
```

`fattn-mma-f16.cuh`, two places add the mask to `KQ_C`: line 785 (`cols_per_warp == 8`) and line 850 (else branch, with an `RDNA3` and a non-RDNA3 variant). For both:
- outer condition `if (ncols2 > 1 || mask_h)` -> `if (ncols2 > 1 || mask_h || rbt.table)`
- wrap each read of `tile_mask` (lines 794, 863, 874-876) in `if (ncols2 > 1 || mask_h) { ... }`; the `if (rbt.table)` blocks stay as they are.

Do not change the three `flash_attn_ext_f16_load_mask` call sites (lines 685, 1043, 1470): they must stay off when there is no mask.

**Attention.**
- Without a mask `tile_mask` is never loaded: it is uninitialized shared memory. Every read must be behind `mask_h`. A missed read gives random masking, not a crash.
- No change in `fattn.cu`. With a bias the selection is already: `n_q == 1` -> vec kernel, else mma with `ncols2 == 1`; sparse and GQA paths need `mask` and stay off. `scan_mask` is false without a mask, so `KV_max` is NULL and the kernels walk all of `n_kv`.
- `fattn-tile.cuh` is not reached with a bias (`fattn.cu:672` returns NONE without Turing MMA). Leave it.
- The 5090 uses the second mma place for `n_q > 8` and the first for `n_q <= 8`. Both must be edited; the test sizes below hit both.

#### 1d. Tests, `tests/test-backend-ops.cpp`

Extend `test_flash_attn_ext_rel_bias` (line 9194) with two parameters and add them to `vars()` (check which `VARS_TO_STRn` macros exist):
- `int32_t window` (0 = today's test with the mask tensor; > 0 = no mask tensor, `ggml_flash_attn_ext(ctx, q, k, v, nullptr, ...)`, then `add_rel_bias`, then `set_pos_window`).
- `int64_t ns` (number of streams, dim 3 of `q`, `k`, `v`, `m`, `rt`, `kp`, `qp`). Use `q0 = kv - nb - 3 - 7*s` for stream `s` in both the `qp` and the `m` initialization.

Cases (line 13373): keep the loops, add `window` in `{ 0, INT32_MAX, 100 }` and `ns` in `{ 1, 2 }`, and add `nb = 20` (mma `ncols1 = 32`).

**Attention.**
- Keep the sentinel cells (`i % 97 == 5 -> INT32_MAX`) in `kp`: they test "empty cell" in pos mask mode.
- Every query must see at least one cell, else CPU gives zeros and CUDA gives NaN. With window >= 2 and the current `q0` this holds.
- If a `window = 0, ns = 2` case fails, that is a bug that exists at HEAD (multi-stream bias indexing was never tested). Stop and report it, do not hide it by removing `ns`.

#### Gate 1

- `test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT -p extent=` : all pass (window and non-window cases).
- `test-backend-ops.exe test -b CUDA0 -o FLASH_ATTN_EXT` (all cases, background, can take longer than 5 minutes: ask before running if unsure): same pass count as at step 0 plus the new cases. This shows the non-bias kernels did not change.
- No llama change yet, so `session.sh 32768` must give the same compute buffer lines as step 0.

### Step 2. llama: build the graph without the mask

Files: `src/llama-graph.h`, `src/llama-graph.cpp`, `src/llama-kv-cache.h`, `src/llama-kv-cache.cpp`. `src/models/inkling2.cpp` needs no change (check with grep that it does not read `self_attn_mask`).

#### 2a. Decision helper, `llama-graph.cpp` (next to `can_reuse_kq_mask`, line 53)

```cpp
// the KQ mask follows from the positions if it is causal and each stream holds one sequence
static bool rel_use_pos_mask(const llama_hparams & hparams, const llama_cparams & cparams, const llama_ubatch & ubatch) {
    static const bool enabled = getenv("LLAMA_ATTN_POS_MASK_DISABLE") == nullptr;

    return enabled && cparams.flash_attn && cparams.causal_attn && !hparams.use_alibi &&
        hparams.swa_type == LLAMA_SWA_TYPE_STANDARD && !ubatch.is_pos_2d() &&
        (!cparams.kv_unified || ubatch.n_seqs_unq == 1);
}
```

**Attention.** This runs for the reserve graphs too (`ubatch_reserve`, `src/llama-batch.cpp:542`), where `seq_id` holds no data. Use only `n_seqs_unq`, `is_pos_2d()` and the params. Do not read `ubatch.seq_id` or `ubatch.pos` here.

Why each condition: `flash_attn` - only the FA op has the window. `causal_attn` - the non-causal masks have other rules (`set_input_attn_mask`, `llama-kv-cache.cpp:1945`). `use_alibi` - the mask then carries values, not 0/-inf. `swa_type` - only the standard window is `d < n_swa`. `is_pos_2d` - M-RoPE has an extra causal rule. Last term - decision D4.

#### 2b. `llama-graph.h`

- `struct llm_graph_rel_bias` (line 779): add `int32_t n_window; // > 0: mask by position, no KQ mask`.
- `class llm_graph_input_mem_hybrid_iswa_rel`: add `bool pos_mask = false; // the masks follow from the positions, the mask tensors are not built`. Fix the class comment ("the KQ masks are always dense ...").
- `set_input_kv_pos` gets a third parameter (2d).

#### 2c. `llama-graph.cpp`

`build_inp_mem_hybrid_iswa_rel` (line 4372):
- `const bool pos_mask = rel_use_pos_mask(hparams, cparams, ubatch);`
- create `self_attn_mask.raw` and `self_attn_mask_swa.raw` only if `!pos_mask`.
- line 4407 reads `n_stream` from the mask. Replace with `const int64_t n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;` (same formula as `build_attn_inp_kq_mask`, line 41).
- set `inp->pos_mask = pos_mask;`

`llm_graph_input_mem_hybrid_iswa_rel::set_input` (line 1404): pass `pos_mask` to both `set_input_kv_pos` calls. The mask calls are already guarded by `raw && raw->buffer`.

`llm_graph_input_mem_hybrid_iswa_rel::can_reuse` (line 1460):

```cpp
    const int64_t n_stream = params.cparams.kv_unified ? 1 : params.ubatch.n_seqs_unq;

    res &= pos_mask == rel_use_pos_mask(inp_attn->hparams, params.cparams, params.ubatch);
```

- guard both `can_reuse_kq_mask(...)` calls with `if (inp_attn->self_attn_mask.raw)` / `if (inp_attn->self_attn_mask_swa.raw)` (`res &= f(nullptr)` still calls `f`).
- in the two `else` branches (FA path) add `res &= self_kv_pos->ne[1] == n_stream;` and `res &= self_kv_pos_swa->ne[1] == n_stream;`

**Attention.** Today the stream count of a reused graph is checked only through the mask (`kq_mask->ne[3] == n_stream`). Without the mask nothing checks it, and a graph built for 1 stream would be reused for 2. The two new `ne[1]` checks close this. Add them for the dense path too, they are true there anyway.

`build_attn` for the rel input (line 4433), FA branch:

```cpp
    if (cparams.flash_attn) {
        GGML_ASSERT(inp->pos_mask == (kq_mask == nullptr));
        if (kq_mask && kq_mask->type != GGML_TYPE_F16) {
            kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F16);
        }
        const int32_t n_window = !inp->pos_mask ? 0 : is_swa ? (int32_t) hparams.n_swa : INT32_MAX;
        const llm_graph_rel_bias rel = { rel_table, is_swa ? inp->self_kv_pos_swa : inp->self_kv_pos, inp->self_q_pos, n_window };
        ...
```

`build_attn_mha` (line 3185), inside `if (rel)` after `ggml_flash_attn_ext_add_rel_bias`: `if (rel->n_window > 0) { ggml_flash_attn_ext_set_pos_window(cur, rel->n_window); }`

**Attention.** Do not add a general "mask must exist" assert to `build_attn_mha`: other models pass no mask on purpose. The assert in the rel `build_attn` above is the guard.

#### 2d. `llama-kv-cache.cpp` / `.h`

`set_input_kv_pos(ggml_tensor * dst, const llama_ubatch * ubatch, bool seq_only)` in `llama_kv_cache` (line 2024) and the forwarding `llama_kv_cache_context` method (line 3147):

```cpp
        const llama_seq_id seq_id = ubatch->seq_id[s*n_tps][0];
        const auto & cells = v_cells.at(seq_to_stream.at(seq_id));
        for (int64_t j = 0; j < n_kv; ++j) {
            const bool hidden = cells.is_empty(j) || (seq_only && !cells.seq_has(j, seq_id));
            data[s*n_kv + j] = hidden ? INT32_MAX : cells.pos_get(j);
        }
```

**Attention.** `seq_only` must be false on the dense mask path. With a unified cache and several sequences in the ubatch one `kpos` row serves queries of several sequences; hiding the cells of "other" sequences there would give those queries the wrong bias. This is the one place where "always filter, it is harmless" is wrong.

#### Gate 2 (each run under 5 minutes, through `gpurun.sh`)

1. Buffer: `session.sh 131072`. CUDA0 and host compute buffer are each about 128 MiB smaller than at step 0. With `LLAMA_ATTN_POS_MASK_DISABLE=1` both are equal to step 0. Repeat at 32768: the drop is about 32 MiB, i.e. the remaining slope is about 4 B per token. If the slope is above about 16 B per token, report which tensor scales.
2. Same result as the baseline binary, 32k request:
   `python tmp\inkling\tools\teacher_force.py --teacher-exe tmp\inkling\baseline-f929fd85c\bin\llama-server.exe --teacher-fa on --request tmp\inkling\replays\test2\req_32k.json --ctx 40960`
   Expected: top-1 100/100 and `|d_logprob|` max 0. The pos mask adds `0.0f` less and nothing else, the arithmetic for visible cells is the same, so anything else than an exact match is a finding. Report it, do not accept "close".
3. Same command with `LLAMA_ATTN_POS_MASK_DISABLE=1` set in the shell (the tool passes the environment to both servers; the baseline binary ignores it). Must also match exactly. If 3 fails, step 1 broke the dense path. If 3 passes and 2 fails, the pos mask is wrong.
4. Default 2.7k prompt against the baseline binary (`--teacher-exe ... --teacher-fa on`, no `--request`): exact match.
5. Length sweep: `--sweep-lengths 7,15,20,32,64,128,256,512,700,1200,1600,2100`, once with and once without the env switch. The two outputs must be identical line by line (the sweep compares with `-fa off`, so its own numbers are not zero). Lengths above 1024 cross the SWA window and the bias extent; 1600 and 2100 cross a second ubatch.

**Attention.** A missing or wrong mask does not crash and short answers still look fine: the model just sees future or stale cells. Only the exact-match gates catch it. Do not judge by reading the generated text.

### Step 3. Milestone measurements (ask the user before long runs)

1. Replay both requests (`replay.py req_32k.json 4000 3`, `req_76k.json 4000 2`, server `-c 98304`). Compare VRAM, prefill, decode with step 0 and with the table in `inkling-2-mini.md`. Expected: VRAM about 96 MiB lower, prefill equal or a few percent faster (no mask fill, no mask upload per ubatch), decode equal. A prefill loss above 2% is a finding.
2. Long context: reserve numbers at `-c 262144` with `-ub 512` and `-ub 2048`.
3. `-ub 2048` speed: replay `req_32k.json` with `-ub 2048` on baseline and new. This is the case of decision D8. A loss above 3% -> do step 4.
4. Parallel slots, short prompts only: `-np 2 --no-kv-unified` (pos mask, 2 streams) and `-np 2 --kv-unified` (dense fallback when both slots are in one ubatch). Two different requests at the same time must give the same text as each request alone with the baseline binary (greedy, `temperature 0`).
5. CPU reference end to end: `teacher_force.py --ngl 0 --n-tokens 20` with a short `--prompt-text`, with and without the env switch: identical output.
6. Context shift or `--cache-reuse` if the user wants it: any run that moves positions; compare with the env switch on and off.

Write all numbers into `inkling-strata-analysis.md` section 10 and fix sections 6, 7, 9 there (decision taken, what is still open).

### Step 4 (optional, only if step 3.3 or 3.4 shows a loss). Skip trailing KV tiles from positions

`fattn-common.cuh`: add `flash_attn_pos_to_KV_max`, a copy of `flash_attn_mask_to_KV_max` (line 764) with these differences only:
- parameters `const int * kpos_ptr, const int * qpos_ptr, int * KV_max_ptr, const int ne30, const int n_q, const int n_kv`
- before the loop: `qpos_max` = max of `qpos[sequence*n_q + jt*ncols1 + j]` over `j < ncols1` with `jt*ncols1 + j < n_q`
- in the loop: `int all_inf = kpos[sequence*n_kv + KV_max_sj + 2*tid] > qpos_max && kpos[sequence*n_kv + KV_max_sj + 2*tid + 1] > qpos_max;`

In `launch_fattn` next to `scan_mask` (line 1226): `const bool scan_pos = !use_sparse && !mask && rel_bias.window > 0 && n_kv_cols % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1);` and launch the new kernel with the same grid and `KV_max.alloc` as the mask scan.

**Attention.** The test must stay one-sided: a tile may be skipped only if every cell is hidden for every query of the tile. `kpos > qpos_max` gives that. Do not use the window lower bound here: `KV_max` can only cut the tail. The kernels already consume `KV_max`, do not change them.
Gate: step 1 tests with `nb = 1024, 2048` added, gate 2 items 2 and 5 at `-ub 2048`, and the speed of step 3.3.

### Step 5 (optional, ask the user). Other backends must refuse the node

Pre-existing gap from `f929fd85c`, made worse by the pos mask (a backend that ignores `src[6]` and gets no mask attends to every cell). In each `supports_op` for `GGML_OP_FLASH_ATTN_EXT` that already refuses `src[5]`, also refuse `src[6]`: `ggml-cann.cpp:2683`, `ggml-cpu/spacemit/ime.cpp:1051`, `ggml-hexagon/htp-kparams.cpp:359`, `ggml-opencl.cpp:12666`, `ggml-openvino.cpp:1909`, `ggml-sycl.cpp:7745`, `ggml-vulkan.cpp:16778`, `ggml-webgpu.cpp:5090`, `ggml-zdnn/fattn.cpp:24`; Metal supports `src[5]` and needs its own check (`ggml-metal-device.m:1748`). One line each, cannot be tested here (CUDA build only). Decision for the user: touch 10 backend files for safety, or keep the fork CUDA and CPU only.

## 5. Review checklist (for the reviewer after each step)

- Step 1: no read of `tile_mask` outside `mask_h`; `window` is the last struct member in both structs and in both initializers; the CPU `continue` is before the `ik1` lookup; no edit in `fattn.cu` and `fattn-tile.cuh`.
- Step 2: `rel_use_pos_mask` reads no ubatch arrays; `can_reuse` has the mode check, the null guards and the two `ne[1]` checks; `seq_only` is `pos_mask`, not `true`; the dense path is byte for byte the old code when the env switch is set.
- All: comments short and ASCII, no new files besides tests and docs, diff stays below about 150 lines without tests.

## 6. Known limits after this work (state them in the results section)

- 4 B per context token remain (`kpos`), on the device and on the host.
- `-np N` with a unified cache keeps the dense mask in the reserved buffer (D4).
- K/V cache types other than F16 make the mma kernel convert K and V to F16 in a temporary that scales with `n_kv` (`ggml_cuda_flash_attn_ext_get_alloc_size`, `fattn.cu:789`). Not measured, not part of this plan; relevant only with `-ctk` / `-ctv`.
- `-ub >= 1024` or several streams lose the tail skip unless step 4 is done.
- Upstream `ggml-org/llama.cpp` #32740 uses another bias contract (index tensor). If it merges, this branch needs a rebase; the pos mask idea carries over (window + position sentinel), the code does not.
