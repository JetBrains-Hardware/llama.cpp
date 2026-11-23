# Inkling-2-Mini on llama.cpp (16/24 GB card work)

Branch `erokhins/inkling` (ours): the fork CI commits on top of ggml-org/master, which already supports the model.
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

Baseline (`8f2537201`) vs fused relative-position bias in FlashAttention (`hridaya423/llama.cpp` `68cd7533a`, `ggml_flash_attn_ext_add_rel_bias`):

| Request | Mode | VRAM | Prefill | Generation |
|---|---|---|---|---|
| flow 144 (32k, 32,144 tok, 3 reps) | original (`8f2537201`) | 27,110-27,390 MiB (26.5-26.7 GiB) | 1,400-1,432 t/s (22.4-23.0 s) | 77.9-78.8 t/s (1,939 tok) |
| flow 144 (32k, 32,144 tok, 3 reps) | + hridaya423 (`68cd7533a`) | 17,387-17,394 MiB (17.0 GiB) | 2,703-2,739 t/s (11.7-11.9 s) | 81.3-81.4 t/s (1,727 tok) |
| flow 197 (76k, 75,985 tok, 2 reps) | original (`8f2537201`) | 26,984-27,006 MiB (26.4 GiB) | 916-917 t/s (82.9 s) | 66.7-67.1 t/s (802 tok) |
| flow 197 (76k, 75,985 tok, 2 reps) | + hridaya423 (`68cd7533a`) | 17,386-17,530 MiB (17.0-17.1 GiB) | 2,401-2,492 t/s (30.5-31.7 s) | 71.6-73.6 t/s (2,832 tok) |

Server VRAM after load drops from ~27.0 GiB to 17,358 MiB (16.95 GiB) with `-c 98304`, with `0` graph reallocations.
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

Reproduce: `build-inkling.bat llama-server`, start `llama-server -m <gguf> -ngl 99 -fa on -c 98304 -np 1 --port 8765 --jinja`,
then `python tmp/inkling/tools/replay.py tmp/inkling/replays/test2/req_32k.json 4000 3`.
