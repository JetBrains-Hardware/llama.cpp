# Inkling-2-Mini on llama.cpp (16/24 GB card work)

Branch `erokhins/inkling` (rebased on `erokhins/kvq-rebase`): the fork CI and `--kvq` commits on top of ggml-org/master, which already supports the model.
No loader changes were needed for the Unsloth `UD-Q3_K_XL` file.

## Support in ggml-org/llama.cpp

| PR | What |
|---|---|
| #32217 | Inkling 2 text + vision |
| #32694 | Inkling 2 Mini, arch `inkling2_loop` (24 layers, loop core 6..18 run 3 times, 48 layer visits) |
| #32657 | `ffn_gate_up_exps` scales (needed by the Unsloth NVFP4 expert tensors) |
| #32704 | loop scale fix |
| #32751 (open) | `-sm tensor` for inkling2 |
| #32218 (open, draft) | DSpark drafter |

Known issues: #32755 (garbage with loop-layer experts on CPU, `--n-cpu-moe`, long prompts), #32710 (perplexity NaN on CUDA).

## GGUF

Unsloth: `Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file` (13.6 GB, rename to `.gguf`) is already in the mainline layout
(`inkling2_loop`, `block_count` 24, 744 tensors, NVFP4 experts in layers 6..19 with `.scale` tensors). The files in the repo root
use the pre-merge layout (48 blocks) and do not load. Quant sizes (GB): UD-IQ3_XXS 11.66, UD-Q3_K_XL 13.63, UD-IQ4_XS 15.73,
UD-Q4_K_M 16.61, UD-Q4_K_XL 17.63, UD-Q8_K_XL 22.87. Other sources: bartowski (imatrix quants), ggml-org (IQ4_NL 17.6 GB, Q8_0, BF16, mmproj).

## Measurements (RTX 5090, CUDA, flash attention on, build 8f2537201)

llama-bench, UD-Q3_K_XL (12.68 GiB): pp512 3184 t/s, pp2048 2521 t/s, tg128 92 t/s.

VRAM (`-np 1`, includes ~0.4 GB idle): 14.6 GB at 4k context, 18.2 GB at 32k, 30.8 GB at 131k, i.e. 0.128 MiB per context token.
SWA layers are properly capped to a fixed 1,536-cell window (~240 MiB total); the ~96 KiB/tok compute scaling came from
materializing the dense per-head relative-position bias (`kq_b` + `ggml_cont(ggml_permute(...))`, F32) before FlashAttention.

### Real Junie requests (replayed)

Two requests captured from Junie CLI (profile `local-inkling-2-mini-q3kxl`, through mitmproxy) are replayed with
`cache_prompt=false` (full prefill each time) and the original sampling and seed. Server: `-c 98304 -np 1 -fa on`.

Baseline (`8f2537201`) vs fused relative-position bias in FlashAttention (`hridaya423/llama.cpp` `68cd7533a`, `ggml_flash_attn_ext_add_rel_bias`) vs two-tier KV cache (`--kvq`):

| Request | Mode | VRAM | Prefill | Generation |
|---|---|---|---|---|
| flow 144 (32k, 32,144 tok, 3 reps) | original (`8f2537201`) | 27,110-27,390 MiB (26.5-26.7 GiB) | 1,400-1,432 t/s (22.4-23.0 s) | 77.9-78.8 t/s (1,939 tok) |
| flow 144 (32k, 32,144 tok, 3 reps) | + hridaya423 (`68cd7533a`) | 17,387-17,394 MiB (17.0 GiB) | 2,703-2,739 t/s (11.7-11.9 s) | 81.3-81.4 t/s (1,727 tok) |
| flow 144 (32k, 32,144 tok, 3 reps) | + kvq (`--kvq`) | 14,473-14,480 MiB (14.1 GiB) | 2,609-2,653 t/s (12.1-12.3 s) | 70.1-70.2 t/s (1,799 tok) |
| flow 197 (76k, 75,985 tok, 2 reps) | original (`8f2537201`) | 26,984-27,006 MiB (26.4 GiB) | 916-917 t/s (82.9 s) | 66.7-67.1 t/s (802 tok) |
| flow 197 (76k, 75,985 tok, 2 reps) | + hridaya423 (`68cd7533a`) | 17,386-17,530 MiB (17.0-17.1 GiB) | 2,401-2,492 t/s (30.5-31.7 s) | 71.6-73.6 t/s (2,832 tok) |
| flow 197 (76k, 75,985 tok, 2 reps) | + kvq (`--kvq`) | 14,486 MiB (14.1 GiB) | 2,523-2,538 t/s (29.9-30.1 s) | 66.4 t/s (210-225 tok) |

Server VRAM after load drops from ~27.0 GiB to 17,358 MiB (16.95 GiB) with `+ hridaya423` and to ~14.1 GiB flat with `--kvq` at `-c 98304` (fits comfortably on a 16 GB card), with `0` graph reallocations.
Generated lengths differ from baseline (and from the original cached-prefix run: 1,489 and 3,174) because relative bias is
accumulated in FP32 inside FlashAttention instead of cast through an FP16 mask; repeats remain deterministic.
Real Junie traffic rarely generates much: of 190 requests only a handful produced more than 600 tokens.

### Correctness verification (teacher forcing)

To verify that the different generated token counts come from normal FP32-vs-FP16 numerical sensitivity rather than a kernel bug,
we ran teacher-forced greedy checks on the same `UD-Q3_K_XL` GGUF (using `original -fa on` as teacher, and `-fa off` unfused FP32
attention on the same quantized weights as the exact-attention reference):

- **2.7k prompt (`2,745` tok, crosses `n_swa=1024` and `rel_extent=1024`, 100 steps):** 95/100 top-1 exact match, 100/100 top-2/top-5 match, median `|d_logprob| = 0.031` (mean `0.052`), `ppl_ratio = 0.9969`. Against `-fa off`, `+ hridaya423` had lower median drift than `original` (`0.023` vs `0.030`).
- **32k prompt (`flow 144`, `23,709` tok, 100 steps):** 95/100 top-1 exact match, 100/100 top-2/top-5 match, median `|d_logprob| = 0.0026` (mean `0.022`), `ppl_ratio = 1.0089`. Every mismatch across both runs (10/10) was a near-tie (`margin = 0.01-0.38`) with the teacher token at rank 2.
- **CPU vs CUDA 6-way check (20-tok prompt, 20 steps):** 20/20 exact top-1 match across all 6 combinations (`original`/`+ hridaya423`/`-fa off` on CPU and CUDA). On CPU, `+ hridaya423 -fa on` was closer to `-fa off` than `original -fa on` (`0.0091` vs `0.0122` mean `|d_logprob|`), while even `CUDA -fa off` vs `CPU -fa off` differed by `0.0081`.
- **Prefill length sweep (`L = 7, 15, 20, 32, 64, 128, 256, 512, 700, 1200` on CUDA):** 10/10 exact top-1 match vs `-fa off` for both builds; `+ hridaya423` had smaller `|d_logprob|` vs `-fa off` on 7/10 lengths (e.g. `L=1200`: `0.000074` vs `0.000207`).
- **Kernel & CUDA graph isolation:** `GGML_CUDA_DISABLE_GRAPHS=1` was bit-identical; 1-token decode (`fattn-vec`) and full prefill (`fattn-mma-f16`) agreed on spot checks at step 50.

## KVQ two-tier cache feasibility (vs Qwen3.8-27B)

All core Qwen3.8-27B KV cache properties required by `--kvq` (`erokhins/kvq`) hold on Inkling-2-Mini. Verified on both `req_32k.json` (32,144 tok) and `req_76k.json` (75,985 tok) across all 8 full-attention slots (`{5, 11, 17, 29, 35, 41, 47, 23}`) over 64 greedy decode steps using the same generalized-KLT codec (`K` 2-bit, `V` 1-bit, 8192-row chunks with `M_K` query and `M_V` `W_o * gate` metrics):

| Property | Qwen3.8-27B (25k/50k) | Inkling-2-Mini `req_32k` | Inkling-2-Mini `req_76k` | Holds? |
|---|---|---|---|---|
| Query anisotropy (`max/med`, top-8, mean share) | `39x`, `29%`, `27-49%` | `75.3x`, `42.5%`, `53.3%` | `73.2x`, `41.9%`, `50.7%` | Better |
| K / V mean energy | K `8-22%`, V `3-45%` | K `26.6%`, V `16.7%` | K `23.0%`, V `15.6%` | Yes |
| K / V norm CV | K `1-4%`, V `15-38%` (needs V gain) | K `1.1%`, V `0.0%` (RMSNorm) | K `1.7%`, V `0.0%` (RMSNorm) | Cleaner (no V gain needed) |
| Token-0 sink outlier | `15-19 sigma` | `1.03x` (virtual sink `2.1%`) | `1.01x` (virtual sink `1.6%`) | Cleaner |
| R2 coarse-K `N=1024` (`recall@16 / @64`) | `1.000 / 1.000` | `1.000 / 1.000` | `0.999 / 0.999` | Yes |
| R2 missed mass (`N=256 / 1024 / 4096`) | `0.225 / 0.114 / 0.040` (50k: `0.133`) | `0.283 / 0.152 / 0.056` | `0.364 / 0.207 / 0.087` | Yes (slightly broader tail) |
| Two-tier `K2/V1` attn NMSE (`0p -> R2@1024`) | KL `0.010 -> 0.00016` | `0.0300 -> 0.00020` (`0.00017` w/ V gain) | `0.0641 -> 0.00033` (`0.00031` w/ V gain) | Yes (~150-200x drop) |
| Early-Q `j=1` (`cos` / `r@64 in 1024` / worst layer) | `0.950` / `99.6%` / `98.3%` | `0.961` / `99.6%` / `98.6%` | `0.957` / `98.9%` / `95.9%` | Matches/better |
| Early-Q `j=2` (`cos` / `r@64 in 1024` / worst layer) | `0.917` / `98.3%` / `89.9%` | `0.924` / `99.0%` / `97.3%` | `0.917` / `96.7%` / `92.8%` | Better (even worst layer) |
| Early-Q `j=3` (`cos` / `r@64 in 1024` / worst layer) | `0.852` / `92.9%` / `16.8%` | `0.891` / `97.9%` / `95.0%` | `0.879` / `94.6%` / `89.3%` | Much better |
| Fallback predictors (`N=1024` `r@64` / `miss`) | `B2`: `79.5%` / `0.255`; `A`: `46%` ovlp | `B2`: `88.4%` / `0.253`; `A`: `37.6%` ovlp | `B2`: `76.9%` / `0.390`; `A`: `32.5%` ovlp | Same ranking (`D > B2 > A`) |

Architectural notes for implementing `--kvq` on Inkling-2-Mini:
- **Early-Q overlap:** all 8 full-attention slots sit at `ip % 6 == 5`, so early-query leads `j = 1, 2, 3` share the same `blocks` and loop `bank` with 1-3 full SWA+MoE layers of prefetch overlap.
- **No relative-bias table in coarse KVQ:** for all old coarse keys (`dist >= 1024`), `rel_table[1024] = -sink` is constant per query head, so coarse KVQ needs no relative-bias lookup table (only the recent-1024 precise ring uses `rel_table[0..1023]`).
- **Smaller footprint:** `head_dim = 128` makes KLT tables 4x smaller and coarse `K2/V1` only ~`3 KiB/tok` (~`384 MiB` at 131k).
- **Caveat for end-to-end KL tuning:** Inkling-2-Mini's attention tail is slightly broader than Qwen3.8-27B (~250-263 prompt tokens `>1e-3`/step vs 80-95, and coda slot `23` has the highest NMSE at `0.0008-0.0009`), so `prefetch_n = 1024` vs `2048` should be checked on token-level KL at 76k-131k+.

### KVQ implementation and verification

Implemented `--kvq` for `inkling2` and `inkling2_loop` on top of `erokhins/kvq-rebase` in two layers:

- **Generic KVQ refactoring (`f8c9f8a4a`):**
  - Generalized `ggml_kvq_attn` (CPU + CUDA `kvq-attn.cu`), `llama_kvq_store`, `llama_kvq_fitter`, and `llama_memory_hybrid_kvq` from fixed `D = 256` to `D in {128, 256}` (using 8-byte `cp.async` for 136-byte `q8_0` rows at `D = 128`).
  - Added `GGML_KVQ_ATTN_LSE` mode (returns `[D, n_head, L + ceil(L/D), S]` with per-head log-sum-exp) and allowed null tail (`k == nullptr`) so coarse+resident attention can be merged with a separate relative-bias tail attention.
  - Added optional SWA KV cache inside `llama_memory_hybrid_kvq` (for hybrid-ISWA models), `rel_keep` (keeps the trailing `n_relative_extent` tokens precise in the ring), `llama_model::kvq_layer_order()` (execution-order slot list for looped models), and full-row `o_proj` dequantization when `n_rep * D` is not a multiple of the quant block size (`Q6_K`).
- **Inkling-2 integration (`767d84ef0`):**
  - Wired KVQ graph inputs, SWA attention, query/gate stats, and early-Q prefetch (`lead = 2`) in `src/models/inkling2.cpp` and `src/models/inkling2-loop.cpp`.
  - Prefill uses scratch attention with `r0 = max(0, n_host - 1024)/256*256` so all tokens within `rel_extent = 1024` stay in the precise ring.
  - Packed decode splits coarse+resident rows (`GGML_KVQ_ATTN_LSE`, constant `-sink` bias per head) from the precise ring tail (`ggml_flash_attn_ext_lse` with relative-position bias table), merges `(o_c, l_c)` and `(o_t, l_t)` in log-sum-exp space, and folds the virtual attention sink.
- **Verification:**
  - `test-kvq-attn` (CPU + CUDA0): passed all single-stream and multi-stream cases (GPU-vs-CPU NMSE `< 8e-7`).
  - `test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT -p extent=`: 40/40 passed (`f16` and `q8_0`).
  - Replays at `-c 98304 -np 1 -fa on --kvq`: `14.1 GiB` VRAM across both 32k (`2,609-2,653 t/s` prefill, `70.1-70.2 t/s` decode, deterministic `1,799` tok, `0.00 ms` prefetch join, `0` failed fits) and 76k (`2,523-2,538 t/s` prefill, `66.4 t/s` decode).
  - Teacher forcing on `flow 144` (`23,709` prompt tokens, 100 greedy steps, `--kvq` vs exact `-ctk q8_0 -ctv q8_0 -fa on`): **96/100** top-1 match, **100/100** top-5 match, median `|d_logprob| = 0.00456`, `ppl_ratio = 1.0075`, exact prefill/decode agreement at step 50.

## 16 GB card (15.5 GiB) budget, vision, and DSpark planning

After cherry-picking `posmask` (`1e29b33ea`, drops the `kq_mask` tensor; `-c 98304` load VRAM drops to `17,080 MiB` standard / `14,385 MiB` `--kvq`), the `--kvq` VRAM breakdown for `UD-Q3_K_XL` (`-ngl 99 -fa on -np 1 -ub 512`) is:

- **Fixed base (`~13.63 GiB` / `~13,960 MiB`, O(1) in `n_ctx`):** `12,984 MiB` weights + `128 MiB` SWA KV (40 layers, `q8_0`, 1,536 cells) + `36 MiB` full-attn ring tail (8 layers, `q8_0`, 2,048 cells) + `4 MiB` short-conv state + `130 MiB` compute buffer (`4 B/tok` growth) + `~680 MiB` CUDA/WDDM overhead.
- **Per-token scaling (`5.65 KiB/tok` above ~72k):** `3.52 KiB/tok` coarse `K2/V1` codes + KLT tables (`28.2 MiB` per 8,192-token chunk) + `2.13 KiB/tok` 1-layer `q8_0` prefill scratch `kvq_scr` (shared with the fixed `154 MiB` decode resident buffers). Lazy host RAM for pinned KV pages is `~20.5 KiB/tok` (`~2.6 GiB` at 128k, `~5.2 GiB` at 256k).
- Note: using `-ub 2048` instead of `-ub 512` adds `+388 MiB` to the fixed compute buffer (`~-70k` max context).

### Max context in 15.5 GiB (`15,872 MiB`)

| Configuration (`UD-Q3_K_XL`, `-fa on -np 1 -ub 512`) | Fixed base | Scaling | Est. VRAM @ 128k | Max context in 15.5 GiB |
|---|---:|---:|---:|---:|
| Standard F16 KV (`+ hridaya423 + posmask`) | `~13.68 GiB` | `32.0 KiB/tok` | `~17.68 GiB` (OOM) | `~56k-60k` |
| Standard Q8_0 KV (`-ctk q8_0 -ctv q8_0`) | `~13.57 GiB` | `17.0 KiB/tok` | `~15.70 GiB` (OOM) | `~110k-118k` |
| `--kvq` text only | `~13.63 GiB` | `5.65 KiB/tok` | `~14.34 GiB` (`~15.05 GiB` @ 256k) | `~330k-345k` |
| `--kvq` + Vision on CPU (`--no-mmproj-offload`) | `~13.63 GiB` | `5.65 KiB/tok` | `~14.34 GiB` | `~330k-345k` |
| `--kvq` + Vision on GPU (`mmproj Q8_0`) | `~14.2 GiB` | `5.65 KiB/tok` | `~14.9 GiB` | `~230k-245k` |
| `--kvq` + DSpark drafter (`Q8_0` / `F16` draft KV) | `~14.25 GiB` | `7.8-9.7 KiB/tok` | `~15.2-15.5 GiB` | `~130k-160k` (`~220k` if draft KV capped) |
| `--kvq` + Vision on GPU + DSpark | `~14.8 GiB` | `7.8-9.7 KiB/tok` | `~15.8-16.0 GiB` (OOM) | `~70k-90k` |

### Vision (`--mmproj`) and DSpark (`-md`) notes

- **Vision (`inkling2v`):** supported in this branch (upstream PR `#32217`, `tools/mtmd/models/inkling2v.cpp`), not enabled in our runs yet. Costs `~0.5-0.9 GiB` fixed on GPU (`mmproj Q8_0` `~350 MiB` or `F16/BF16` `~640 MiB` + `~150-300 MiB` vision compute; `0 B/tok` scaling), or `0 MiB` VRAM with `--no-mmproj-offload` (CPU).
- **Drafter (DSpark, not MTP):** Inkling-2-Mini uses a co-trained 3-layer DSpark drafter (DFlash variant: block-8 non-causal attention + Markov RNN head, attention pattern `SWA-1025, Global, SWA-1025`, tapping 4 target layers `[5, 23, 41, 48]`). Standard MTP (`n_layer_nextn`) is not used. Upstream support is still open draft PR `#32218` (Strata implements it in `ink_dspark.cuh`). Costs `~0.5-0.8 GiB` fixed (`~0.4-0.65 GiB` weights + `~12 MiB` SWA KV + `~80-120 MiB` scratch) plus `2.13 KiB/tok` (`Q8_0`) or `4.0 KiB/tok` (`F16`) for its global draft KV layer unless capped to a smaller window. Note: `--kvq` packed decode already supports `L <= 8`, but draft-rejection KV rollback with `--kvq` will need verification when enabling DSpark.

### Possible solutions for GPU vision without permanent VRAM overhead

Vision encoding runs strictly before LLM prefill/decode and never overlaps with it. Relevant upstream work (none merged yet):

- `ggml-org/llama.cpp#28320` (open draft, `--mmproj-vram-swap`): calls `llama_sched_release()` to free idle LLM compute buffers during image encode, runs a transient GPU `mtmd_context`, then re-reserves. Limitation here: `posmask` already shrank our LLM compute buffer to `~130 MiB`, which is not enough on its own for `mmproj-Q8_0` (`~350 MiB`) + vision compute (`~150-300 MiB`).
- `ggml-org/llama.cpp#28346` (open draft, `--mmproj-evict-draft`): keeps `mmproj` in host RAM and temporarily evicts draft model GPU weights to host RAM during image/audio encode (`llama_model_weights_swap_prepare/evict/restore`), then restores them. Directly solves the Vision + DSpark case.
- `ggml-org/llama.cpp#20246` (discussion): temporarily evicting a subset of LLM layers to RAM during vision encode (no PR yet).

Candidate approaches to revisit later on this branch (ordered by simplicity):

1. **Borrow pure scratch (`llama_sched` + `kvq_scr`, zero copy cost):** at 256k-320k `--kvq` context, the LLM compute buffer (`~130 MiB`) plus `kvq_scr` (`~544-680 MiB`) provide `~670-810 MiB` of stateless scratch that can be freed and reallocated around image encode with no D2H/H2D copy.
2. **Swap with DSpark draft weights (`#28346` pattern):** when running DSpark, evicting `~500 MiB` of draft weights to host RAM and restoring them over PCIe 4.0/5.0 x16 takes `~20-40 ms` each way.
3. **Evict a few LLM layer buffers (extension of `#28346`):** works at any context size and without a drafter; evicting `~1 GiB` (2-3 MoE layers) and restoring takes `~50-100 ms` round-trip over PCIe using the same `ggml_backend_buffer` evict/restore mechanism scoped to a subset of layers.

## DSpark speculative decoding + `--kvq` (`w2q8` drafter)

Cherry-picked `9181600bc..d5f770140` from `erokhins/inkling-strata` onto `erokhins/inkling` (`58bbe32bb..4f906547c`: PR `#32218` Inkling-2 DSpark drafter + `fbb525132` greedy argmax-chain draft optimization, which cuts draft compute from `788 MiB` to `36 MiB`).

Measured on RTX 5090 with `Inkling-2-Mini-UD-Q3_K_XL.gguf` + `dspark-Inkling-2-Mini-BF16-w2q8.gguf` (`-c 98304 -ub 512 -np 1 -fa on -ngl 99`, `--spec-type draft-dspark -ngld 99`, `cache_prompt=false`, idle GPU `1,181 MiB`):

| Request | Mode | VRAM (`nvidia-smi` / ~proc) | RAM WS / Priv (`kvq` pinned) | Prefill | Decode (gen tok, mean acc) |
|---|---|---|---|---|---|
| 32k (32,144 tok) | `--kvq` (no draft) | 15,160 MiB (~14.0 GiB) | 15.1 / 17.1 GiB (821 MiB) | 2,432 t/s (13.2s) | 64.3 t/s (1,799 tok) |
| 32k (32,144 tok) | `--kvq` + DSpark `K=2` | 16,290 MiB (~15.1 GiB) | 16.0 / 18.5 GiB (821 MiB) | 2,376 t/s (13.5s) | 98.0 t/s (495 tok, 2.26) |
| 32k (32,144 tok) | `--kvq` + DSpark `K=3` | 16,293 MiB (~15.1 GiB) | 16.0 / 18.5 GiB (821 MiB) | 2,365 t/s (13.6s) | 92.3 t/s (1,607 tok, 2.47) |
| 32k (32,144 tok) | `--kvq` + DSpark `K=4` | 16,310 MiB (~15.1 GiB) | 16.0 / 18.5 GiB (821 MiB) | 2,373 t/s (13.6s) | **99.5 t/s** (2,135 tok, 2.80) |
| 32k (32,144 tok) | `--kvq` + DSpark `K=7` | 16,320 MiB (~15.1 GiB) | 16.0 / 18.5 GiB (821 MiB) | 2,385 t/s (13.5s) | 93.9 t/s (1,687 tok, 3.01) |
| 32k (32,144 tok) | `std` + DSpark `K=7` | 19,116 MiB (~17.9 GiB) | 15.3 / 20.6 GiB (n/a) | 2,466 t/s (13.0s) | 102.3 t/s (1,684 tok, 3.01) |
| 76k (75,985 tok) | `--kvq` (no draft) | 15,174 MiB (~14.0 GiB) | 16.0 / 18.0 GiB (1,642 MiB) | 2,345 t/s (32.4s) | 60.8 t/s (225 tok) |
| 76k (75,985 tok) | `--kvq` + DSpark `K=2` | 16,296 MiB (~15.1 GiB) | 16.9 / 19.5 GiB (1,642 MiB) | 2,300 t/s (33.0s) | 116.6 t/s (802 tok, 2.86*) |
| 76k (75,985 tok) | `--kvq` + DSpark `K=3` | 16,314 MiB (~15.1 GiB) | 16.9 / 19.4 GiB (1,642 MiB) | 2,293 t/s (33.1s) | 88.1 t/s (2,000 tok, 2.45) |
| 76k (75,985 tok) | `--kvq` + DSpark `K=4` | 16,318 MiB (~15.1 GiB) | 16.9 / 19.4 GiB (1,642 MiB) | 2,300 t/s (33.0s) | **96.7 t/s** (2,000 tok, 2.84) |
| 76k (75,985 tok) | `--kvq` + DSpark `K=7` | 16,354 MiB (~15.2 GiB) | 16.9 / 19.4 GiB (1,642 MiB) | 2,305 t/s (33.0s) | 90.2 t/s (2,000 tok, 3.04) |
| 76k (75,985 tok) | `std` + DSpark `K=7` | 19,118 MiB (~17.9 GiB) | 15.5 / 20.8 GiB (n/a) | 2,275 t/s (33.4s) | 101.4 t/s (2,000 tok, 3.38) |

*Note: `K=2` on 76k hit an unusually high-acceptance 802-token trajectory (`2.86` mean acc out of `K+1=3`).*

Key findings:

- **Recommended `K` for `--kvq` + DSpark: `--spec-draft-n-max 4` (`K=4`).** It delivers the best consistent decode throughput (`99.5 t/s` at 32k, `96.7 t/s` at 76k; `+55%` to `+59%` over `--kvq` no-draft) with only `~2-2.5%` prefill overhead. With `n_rep = 3`, `K=4` gives `n_rep * (K + 1) = 15` query columns per KV head, filling `15/16` of `kvq-attn`'s `N=16` MMA tile without spilling into a second tile (whereas `K=7` spills to `24` columns and slows down to `93.9 / 90.2 t/s`).
- **VRAM overhead:** DSpark (`w2q8`) adds `+1,150 MiB` (`+1.12 GiB`) total at `-c 98304` (`657.5 MiB` draft weights + `398 MiB` draft KV + `36 MiB` draft compute + `18 MiB` RS snapshots + `10.5 MiB` target compute), keeping `--kvq` + DSpark at `~15.1 GiB` process VRAM (`~2.8 GiB` below `std` + DSpark, fitting inside a 16 GB card at 96k).
- **Host RAM:** WorkingSet increases by `~0.9 GiB` with DSpark (`15.1-16.0 GiB -> 16.0-16.9 GiB`, including `821 MiB` / `1,642 MiB` of `--kvq` pinned KV pages); Private bytes stay at `18.5-19.5 GiB` (`~1.4-2.1 GiB` below `std` + DSpark's `20.6-20.8 GiB`).
- **Correctness note:** Acceptance rates on `--kvq` + DSpark (`3.01` at `K=7` on 32k) match `std` + DSpark (`3.01`), indicating healthy draft verification; formal teacher-forcing rollback verification specifically for `--kvq` + DSpark is still TODO.
- **Recommended launch flags (`--kvq` + DSpark):** `llama-server -m <model.gguf> -md <dspark-w2q8.gguf> --spec-type draft-dspark --spec-draft-n-max 4 -ngl 99 -ngld 99 -fa on -c 98304 -ub 512 -np 1 --kvq --port 8765 --jinja`

## Local tools (not committed)

Everything lives in the git-ignored `tmp/inkling/` of this worktree (`/tmp/` is in `.gitignore`):

- `tools/` - `replay.py` (replay a captured request, print prefill/decode speed), `teacher_force.py` (teacher-forcing and prefill-sweep correctness checks), `kv_analyze.py` (KVQ generalized-KLT codec, attention sparsity, and early-Q/B2/A predictor analysis), `dump_reqs.py` (mitmdump addon: write request bodies),
  `reqs.py` and `join.py` (match server log lines to flows), `ginfo.py` (print GGUF metadata and tensor types), `session.sh`
  (server session with VRAM), `build-inkling.bat`, `gpurun.sh` (GPU lock; the lock directory is shared in `kvq-scratch`).
- `replays/test2/req_32k.json`, `req_76k.json` - the two test requests; `replays/picked`, `reqs*` - all dumped bodies.
- `logs/` - Junie run logs, server logs, replay results.

Junie setup: profile `~/.junie/models/local-inkling-2-mini-q3kxl.json` points at `http://localhost:8091/inkling/v1/chat/completions`;
mitmproxy-utils (`windows-support` branch, local) routes `localhost:8091/inkling` to `localhost:8765` (plain http, no CA needed).
Select with `junie --model custom:local-inkling-2-mini-q3kxl`; `--slow-model` is accepted but undocumented.

Reproduce: `build-inkling.bat llama-server`, start `llama-server -m <gguf> -ngl 99 -fa on -c 98304 -np 1 --kvq --port 8765 --jinja` (omit `--kvq` for standard full-attention KV cache),
then `python tmp/inkling/tools/replay.py tmp/inkling/replays/test2/req_32k.json 4000 3`.
