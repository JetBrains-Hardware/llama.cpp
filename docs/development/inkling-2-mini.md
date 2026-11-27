# Inkling-2-Mini on llama.cpp (16/24 GB card work)

Branch `erokhins/inkling` (rebased on `erokhins/kvq-rebase`): the fork CI and `--kvq` commits on top of ggml-org/master, which already supports the model.
No loader changes were needed for the Unsloth `UD-Q3_K_XL` file.

## Recommended default settings (`llama-server`)

Best tested configuration for **Inkling-2-Mini `UD-Q3_K_XL` + DSpark** on a 16 GB card (`15.5 GiB` usable VRAM).

GGUF files (Hugging Face):

| Role | File | Size | SHA-256 (first 12) |
|---|---|---:|---|
| Target (`-m`) | [unsloth/Inkling-2-Mini-GGUF: `Inkling-2-Mini-UD-Q3_K_XL.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Inkling-2-Mini-UD-Q3_K_XL.gguf) | 13,631,779,296 B | `e2d03e8c92ac` |
| DSpark drafter (`-md`) | [Ninnix96/Inkling-2-Mini-DSpark-GGUF: `Inkling-2-Mini-DSpark-333M.gguf`](https://huggingface.co/Ninnix96/Inkling-2-Mini-DSpark-GGUF/blob/main/Inkling-2-Mini-DSpark-333M.gguf) | 333,053,536 B | `89cddd423e0b` |
| Vision projector (`-mm`, optional) | [unsloth/Inkling-2-Mini-GGUF: `mmproj-F16.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/mmproj-F16.gguf) | 639,348,160 B | `3bff75ffeb2d` |
| Previous drafter (fallback) | [unsloth/Inkling-2-Mini-GGUF: `dspark-Inkling-2-Mini-Q8_0.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark-Inkling-2-Mini-Q8_0.gguf) (same file as `Mainline_Support/dspark-Inkling-2-Mini-Q8_0.gguf_file`) | 503,889,952 B | `3da00cf7c31b` |

The target file in the repo root is the same file as `Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file`. Do not use the `Old_Support/` files or `dspark-Inkling-2-Mini-Q8_0_Old.gguf` (old drafter layout, `markov_w1` shape error).

```sh
llama-server \
  -m Inkling-2-Mini-UD-Q3_K_XL.gguf \
  -md Inkling-2-Mini-DSpark-333M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 4 --spec-draft-n-ubatch 512 \
  -ngl 99 -ngld 99 -fa on -c 98304 -b 2048 -ub 2048 -np 1 \
  --kvq --kvq-prefill-free-codes --kvq-resident-cap 4096 \
  -lm none --cache-ram 2048 \
  --port 8765 --jinja
```

Long context with vision (tested at `-c 500000` with a 485k-490k prompt and a native 4K image, `12.7 GiB` peak host RAM, see [Host RAM](#host-ram-windows--lm-none)); **note:** on a 16 GB card (RTX 5070 Ti, `~15.4-15.9 GiB` usable), `-c 500000` filled to 478k overflows physical VRAM at decode by `~578 MiB` on the CI CUDA 13.5 build - use `-c 380000`-`400000` to stay strictly in physical VRAM (see [16 GB (RTX 5070 Ti) verification and VRAM measurement](#16-gb-rtx-5070-ti-verification-and-vram-measurement-2026-11-27)):

```sh
LLAMA_DSPARK_KV_CAP=8192 llama-server \
  -m Inkling-2-Mini-UD-Q3_K_XL.gguf \
  -md Inkling-2-Mini-DSpark-333M.gguf \
  --spec-type draft-dspark --spec-draft-n-max 4 --spec-draft-n-ubatch 512 \
  -ngl 99 -ngld 99 -fa on -c 500000 -b 2048 -ub 2048 -np 1 \
  --kvq --kvq-prefill-free-codes --kvq-resident-cap 4096 \
  -mm mmproj-F16.gguf --mmproj-vram-swap --image-max-tokens 5184 \
  -lm none --cache-ram 2048 \
  --port 8765 --jinja
```

The 500k numbers were measured with the previous `Q8_0` drafter; the Ninnix96 drafter needs `~160 MiB` less VRAM.

**`LLAMA_DSPARK_KV_CAP=8192` is required at 500k.** Without it the drafter's global-layer KV grows at `4.0 KiB/tok` (`384 MiB` at `-c 98304`, `1,954 MiB` at `-c 500000`) and the card does not fit. Measured on `0a9f9219c` with the command above, a 485k text prompt and a 300-token answer (RTX 5090; **caveat:** the "process" column below subtracted `441 MiB` measured *after* `llama-server` started, which hid the server's own CUDA context - true server VRAM delta above idle is `16,075 MiB` on local CUDA 12.9 and `16,189 MiB` on CI CUDA 13.5, see [16 GB verification](#16-gb-rtx-5070-ti-verification-and-vram-measurement-2026-11-27)):

| `-c 500000`, 485k prompt | Drafter KV | Prefill peak | Decode peak (process) | Decode peak (`nvidia-smi` total) | Decode |
|---|---:|---:|---:|---:|---:|
| Uncapped | `1,954 MiB` | `17,073 MiB` total | `17,564 MiB` (over budget) | `18,005 MiB` | `79 t/s` |
| **`LLAMA_DSPARK_KV_CAP=8192`** | **`34 MiB`** | - | **`15,644 MiB`** (`228 MiB` under `15,872 MiB`) | `16,085 MiB` | **`103 t/s`** |

- Decode sets the peak: at the end of the prompt the `966 MiB` target compute buffer is freed and the `1,748 MiB` of kvq codes and tables (`3.5 KiB/tok`) are uploaded back to the device.
- The cap also makes long-context decode faster (`103` vs `79 t/s`): the drafter reads an 8k window instead of the whole context.
- Uncapped at `-c 98304` (32k prompt) the same build peaks at `14,914 MiB` process (`384 MiB` drafter KV), so the default command without the cap is right for contexts up to `~230k`.

- **Measured on RTX 5090 (`-c 98304`, greedy 32k / 76k):** `14.87-14.90 GiB` peak PP / `14.51-14.54 GiB` TG process VRAM, `3,060 / 2,974 t/s` prefill, `108.5 / 113.6 t/s` decode (`61.1% / 62.3%` acceptance).
- **Max context in `15.5 GiB` (`15,872 MiB`, 16 GB card):**
  - Default (`-ub 2048`, uncapped draft KV): **`~230k-235k`** (`4.00 KiB/tok` PP / `7.52 KiB/tok` TG; `-c 131072` fits at `~15.01 GiB` PP / `~14.75 GiB` TG).
  - With `LLAMA_DSPARK_KV_CAP=8192` (`-ub 2048`): **`~380k-400k`** strictly in physical VRAM on an RTX 5070 Ti with the CI CUDA 13.5 build (`~350k-380k` with `LLAMA_DSPARK_KV_CAP=32768`; earlier `~475k-480k` estimates were based on the `441 MiB` post-startup baseline subtraction and shorter 76k fill depth - see [16 GB verification](#16-gb-rtx-5070-ti-verification-and-vram-measurement-2026-11-27)).
- **Why each knob:**
  - `--kvq`: two-tier KV cache for the 8 full-attention slots (`3.52 KiB/tok` TG, `0 B/tok` PP with tiled `80.2 MiB` `kvq_scr` vs `32 KiB/tok` FP16), plus compact `1,280`-cell SWA cache (`106.25 MiB`) using per-layer SWA prefill scratch so `-ub 2048` adds zero permanent SWA VRAM.
  - `--kvq-prefill-free-codes`: frees KVQ `codes + tables` during prefill and re-uploads from pinned host RAM (`~5-15 ms`), offsetting the `-ub 2048` compute buffer at PP peak.
  - `--kvq-resident-cap 4096`: halves KVQ `resident` from `137.1` to `69.1 MiB` (`-68 MiB` VRAM, `-70 MiB` host RAM) with `<1%` speed impact and unchanged acceptance.
  - `-b 2048 -ub 2048`: raises MoE utilization from `32` to `128` tokens/expert (`+32%` faster prefill vs `-ub 512`).
  - `-md Inkling-2-Mini-DSpark-333M.gguf` (Ninnix96, the official DSpark drafter requantized: Q6_K FFN and Markov head, Q8_0 projections): `~160 MiB` less VRAM than the Unsloth `Q8_0` drafter, the same speed or better (see [Drafter comparison](#drafter-comparison-2026-11-26)). There is only one official DSpark drafter (Thinking Machines); all GGUFs on Hugging Face are conversions of it.
  - `--spec-draft-n-max 4`: `K=4` gives `n_rep * (K+1) = 15` query columns per KV head, filling `15/16` of `kvq-attn`'s `N=16` MMA tile.
  - `--spec-draft-n-ubatch 512`: bit-exact; saves `134 MiB` PP / `24 MiB` TG with no speed loss.
  - `layer_inp_dev` is `F16` by default in this branch (saves `36 MiB` PP at `-b 2048`, zero acceptance loss).
  - **Do NOT pass `-ctkd` / `-ctvd` (`q8_0`/`q4_0`):** DSpark block-8 decode selects `MMA_F16`, which dequantizes the entire draft KV cache into `384 MiB` of TG compute buffer on every step (more VRAM and slower).
- **Long context (`256k-500k` on 16 GB):** main default stays uncapped (best acceptance up to `~230k`); for long contexts up to `~500k`, add `LLAMA_DSPARK_KV_CAP=8192` in the environment to cap DSpark's global layer-1 KV to an 8k sliding window (`+16 MiB` over `4096`, recovers `+4.6%` 76k acceptance to `58.8%`, unaffected PP).
- **`-lm none` (Windows, host RAM):** loads the GGUFs without mmap. With mmap the mapped files (`13.0 GiB`) stay in the process working set after the weights are copied to VRAM, because `unmap_fragment` is a no-op on Windows (`src/llama-mmap.cpp`; Linux munmaps the offloaded ranges). Same prefill speed and VRAM; peak working set at 385k drops from `22.8` to `10.1 GiB`. There is no `--no-mmap` in this build. See [Host RAM](#host-ram-windows--lm-none) for the 500k numbers.
- **`--cache-ram 2048` (host RAM):** the default `8192` lets the server keep up to `8 GiB` of saved conversation states in RAM (`~25 KiB/tok` with `--kvq`). `2048` keeps one ~80k conversation for a fast switch back; use `0` to turn it off (a switched-back conversation is processed again).
- **Vision (optional):** add `-mm mmproj-F16.gguf --mmproj-vram-swap` for GPU-accelerated vision with `0 MiB` permanent VRAM and `0 MiB` host RAM (reads the weights from the mmproj file at each image, `~0.3 s` from the OS file cache, and borrows the freed LLM compute + KVQ `buf_codes` window during encode; add `--image-max-tokens 5184` for unscaled native `3840x2160` 4K). Use `--no-mmproj-offload` for CPU-only fallback.

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

## VRAM breakdown and optimization roadmap (DSpark + `--kvq` + larger `-ub`)

From `bench_kvq_dspark_k4.log` (`UD-Q3_K_XL`, `-c 98304 -b 2048 -ub 512 -np 1 --kvq`, DSpark `w2q8` `K=4`), measured process VRAM is **`15,129 MiB` (`14.77 GiB`)** in prefill (PP) and `~14,820 MiB` (`14.47 GiB`) in decode (TG):

| Component | PP (`-ub 512`) | TG (`K=4`) | Scales with `-c` | Scales with `-ub` | Notes |
|---|---:|---:|---|---|---|
| Target weights (`UD-Q3_K_XL`) | `12,685 MiB` | `12,685 MiB` | No | No | Fixed (`12.39 GiB`) |
| Target SWA KV (`q8_0`, 1536 cells) + rot + RS | `149 MiB` | `149 MiB` | No | `+85 MiB` / `+1k ub` | `127.5` SWA + `3.1` rot + `18.3` RS |
| Target KVQ codes + tables | `338 MiB` | `338 MiB` | `3.52 KiB/tok` (`901 MiB` @ 256k) | No | Unused in PP (`--kvq-prefill-free-codes` frees it) |
| Target KVQ shared (`max(scr, dec)`) | `213 MiB` (`scr`) | `213 MiB` (`dec=191`) | `2.13 KiB/tok` (`551 MiB` @ 256k) | Tiny | `scr`, `fitter`, and `dec` share one pool |
| Target `layer_inp_dev` (4 layers F32) | `72 MiB` (`2048` rows) | `0.2 MiB` (`5` rows) | No | Scales with `-b` | Shrinks in TG; `F32` for full `n_batch` in PP |
| Target compute (`sched_reserve`) | `240 MiB` | `~20 MiB` (`phase_set`) | `~4 B/tok` | `~450 MiB` / `+1k ub` | `raw_table` + `table` (`F32` x 2) is `96 MiB` (`384 MiB` @ `ub=2048`) |
| DSpark draft weights (`w2q8`) | `657 MiB` | `657 MiB` | No | No | 3 backbone layers still `BF16` (`~450 MiB`); `markov_w2` (`~207 MiB`) unused in PP |
| DSpark draft KV (1 global + 2 SWA) | `398 MiB` | `398 MiB` | `4.0 KiB/tok` (`1,024 MiB` @ 256k) | Tiny | Currently `F16`; costs more than all 8 target KVQ layers combined |
| DSpark draft compute | `36 MiB` | `19 MiB` | No | Tiny | Shrinks via `phase_set` |
| KVQ misc (`qacc`/`gacc`/bias) + CUDA/cuBLAS | `~340 MiB` | `~340 MiB` | No | No | Fixed overhead |
| **Total Process VRAM** | **`15,129 MiB`** | **`~14,820 MiB`** | **`9.65 KiB/tok`** | | |

Why larger `-ub` matters: Inkling-2-Mini is a 128-expert top-8 MoE, so `-ub 512` averages only `32` tokens/expert (bandwidth-bound) vs `128` tokens/expert at `-ub 2048`. Raising `-ub` from `512` to `2048` currently adds `~680 MiB` to target PP compute (of which `384 MiB` is two live `F32` copies of `raw_table`/`table`) plus `+127.5 MiB` to SWA KV (`3072` cells).

### Potential optimizations (3 tiers)

- **Tier 1 (zero code changes - CLI flags + quantized drafter GGUF):** saves `~785 MiB` @ 96k, `~957 MiB` @ 128k, `~1,648 MiB` @ 256k:
  1. `--kvq-prefill-free-codes` (already implemented in `src/llama-kvq-store.cpp`): frees KVQ `codes + tables` during PP and re-uploads from pinned host RAM at end of prompt (`~5-15 ms`). Saves `338 MiB` @ 96k, `450 MiB` @ 128k, `901 MiB` @ 256k at PP peak.
  2. `-ctkd q8_0 -ctvd q8_0` (or `q4_0`, already wired in `common/arg.cpp`): cuts DSpark global KV from `F16` (`384 MiB` @ 96k) to `204 MiB` (`q8_0`, `-180 MiB`) or `108 MiB` (`q4_0`, `-276 MiB`) in both PP and TG.
  3. Full `Q8_0` DSpark GGUF (quantize the 3 `BF16` backbone layers as well as `markov_w2`): cuts drafter weights from `657 MiB` to `~390 MiB` (`-267 MiB` in both PP and TG).
- **Tier 2 (small code changes - makes `-ub 2048` cheap and caps draft KV scaling):**
  4. `ggml_sub_inplace` for sink subtraction on `raw_table` (`src/models/inkling2.cpp`): bit-exact; saves `48 MiB` @ `ub=512`, `192 MiB` @ `ub=2048`, `384 MiB` @ `ub=4096`.
  5. Store `layer_inp_dev` in `F16`/`BF16` or size to `n_ubatch` (`src/llama-context.cpp`): saves `36-54 MiB` at `b=2048`.
  6. Cap DSpark global KV window (e.g. `8k-16k` SWA, Strata-style `ink_dspark.cuh:1147`): turns `204-544 MiB` (`q8_0`) into a fixed `17-34 MiB` (`O(1)` in `n_ctx`); needs acceptance check on `req_76k.json`.
  7. Drop `mask_scr` in `build_attn_kvq_scratch` using `posmask`: saves `~5 MiB` @ `ub=512` / `~20 MiB` @ `ub=2048`.
- **Tier 3 (temporary eviction / scratch tiling - only needed for `-ub 4096`, `>256k` context, or GPU vision):**
  8. Evict DSpark during long bulk prefill (`~390 MiB` weights + `36 MiB` compute; restore over PCIe in `~20-30 ms` for the final capped window, or evict `markov_w2` `~207 MiB` alone).
  9. Tile `kvq_scr` on GPU in fixed `16k-32k` row blocks: caps `kvq_scr` at `~35-70 MiB` instead of `2.13 KiB/tok` (`551 MiB` @ 256k), keeping `kvq_shared` at `dec` size (`~191 MiB`) in both PP and TG.

### Projected process VRAM (15.5 GiB target)

| Configuration | PP (`-c 98k`) | TG (`-c 98k`) | PP (`-c 128k`) | PP / TG (`-c 256k`) |
|---|---:|---:|---:|---:|
| Current (`-ub 512`, `w2q8`, `F16` draft KV) | `14.77 GiB` | `14.47 GiB` | `15.08 GiB` | `16.29 / 15.99 GiB` (OOM) |
| Tier 1 (`-ub 512`, `--kvq-prefill-free-codes`, `-ctkd/v q8_0`, full `Q8_0` draft) | `14.01 GiB` | `14.04 GiB` | `14.14 GiB` | `14.67 / 15.22 GiB` |
| Tier 1 + Tier 2 (`sub_inplace`) at `-ub 2048` | `14.61 GiB` | `14.16 GiB` | `14.74 GiB` | `15.27 / 15.32 GiB` |
| Tier 1 + Tier 2 (+ cap draft KV `16k`) at `-ub 2048` | `14.44 GiB` | `13.99 GiB` | `14.51 GiB` | `14.77 / 14.82 GiB` |
| + Tier 3 (evict DSpark in PP + tile `kvq_scr`) at `-ub 2048` | `~14.05 GiB` | `~13.99 GiB` | `~14.12 GiB` | `~14.03 / ~14.47 GiB` |

Note on Tier 3 vs `--kvq-prefill-free-codes`: with `--kvq-prefill-free-codes` + Tier 2 (`sub_inplace` + capped draft KV), Tier 3 is not needed for `-ub 2048` up to `256k` on a 16 GB card (`~14.4-14.8 GiB` in both PP and TG). At `256k`, freeing `codes` (`901 MiB`) during PP already offsets the `-ub 2048` compute buffer (`~728 MiB`), making TG slightly higher than PP (`14.82` vs `14.77 GiB`). Tier 3 only becomes useful for `-ub 4096` (PP jumps another `+~730 MiB`), `>256k-320k` context (where tiling `kvq_scr` reduces the shared pool in both PP and TG), or GPU vision (`mmproj` swap).

### Bit-exact VRAM optimizations (measured results)

Implemented the bit-exact, drafter-precision-preserving optimizations in `src/models/inkling2.cpp` and `src/llama-memory-hybrid-kvq.{h,cpp}`:

- In-place sink subtraction (`ggml_sub_inplace`) on all four `raw_table` sites (saves `48 MiB` at `ub=512`, `192 MiB` at `ub=2048`).
- In-place `ggml_rms_norm_inplace` in `mix()` when `!blocks.empty()`, and zero-copy `ggml_reshape_2d` replacing 3 redundant `ggml_cont` copies on SWA `mixed` inputs.
- Position-based FlashAttention masking (`pos_mask` / `pos_window`, guarded by `LLAMA_ATTN_POS_MASK_DISABLE`) extended to `--kvq` SWA layers and `--kvq` scratch prefill (`set_input_kvq_scr_pos`), dropping `swa_mask.raw` and `mask_scr.raw` (`~24 MiB` at `ub=2048`) and removing per-ubatch CPU mask fills / H2D uploads.

Measured on RTX 5090 (`-c 98304 -b 2048 -np 1 -fa on -ngl 99 --kvq`, DSpark `w2q8` `K=4`, `cache_prompt=false`, idle GPU `1,184 MiB`, peak VRAM polled every `0.25 s` during PP):

| Mode | Request | Compute buf (`scr`) | Peak VRAM (`nvidia-smi` / ~proc) | End VRAM (TG) | Prefill | Decode |
|---|---|---:|---:|---:|---|---|
| Before (`-ub 512`, no `free-codes`) | 32k / 76k | `240.0 MiB` | `16,310 / 16,318` (`~14.77 GiB`)* | `16,310 / 16,318` | `2,373 / 2,300 t/s` | `99.5 / 96.7 t/s` |
| After (`-ub 512`, no `free-codes`) | 32k / 76k | `222.7 MiB` | `16,657 / 16,695` (`~15.11-15.15 GiB`) | `16,369 / 16,381` | `2,514 / 2,476 t/s` (+6-8%) | `97.5 / 144.6* t/s` |
| After (`-ub 512` + `--kvq-prefill-free-codes`) | 32k / 76k | `222.7 MiB` | **`16,361 / 16,393` (`~14.82-14.85 GiB`)** | `16,361 / 16,381` | `2,516 / 2,467 t/s` (+6-7%) | `97.3 / 143.5* t/s` |
| After (`-ub 1024` + `--kvq-prefill-free-codes`) | 32k / 76k | `320.3 MiB` | **`16,496 / 16,589` (`~14.95-15.04 GiB`)** | `16,360 / 16,441` | **`2,925 / 2,843 t/s` (+23%)** | `101.3 / 92.1 t/s` |
| After (`-ub 2048` + `--kvq-prefill-free-codes`) | 32k / 76k | `558.5 MiB` | **`16,991 / 17,003` (`~15.44-15.45 GiB`)** | `16,489 / 16,489` | **`3,120 / 3,029 t/s` (+32%)** | `97.8 / 97.7 t/s` |

*Notes: the earlier `-ub 512` baseline only sampled post-request VRAM (no `0.25 s` PP peak polling); `76k` at `-ub 512` hit an `802`-token high-acceptance trajectory (`619/732`, `143.5-144.6 t/s`). Bit-exactness confirmed on `req_32k.json`: `--kvq` no-draft (`-ub 512`) reproduced the exact `1,799` generated tokens (`2,432 -> 2,555 t/s` PP), and `--kvq` + DSpark `K=4` (`-ub 512`) produced identical token counts and acceptances with and without `--kvq-prefill-free-codes` (`1,831 tok`, `1145/2740` on 32k; `802 tok`, `619/732` on 76k).*

### Drafter VRAM optimizations (measured results)

Evaluated drafter-side VRAM optimizations at `-c 98304 -b 2048 -ub 2048 -np 1 -fa on --kvq --kvq-prefill-free-codes` (`K=4`, greedy `temp=0`, 512 gen tokens on `req_32k.json` and `req_76k.json`, `0.25 s` peak VRAM polling):

| Config | Peak proc VRAM (32k / 76k) | End proc VRAM (32k / 76k) | PP (32k / 76k) | TG (32k / 76k) | Acceptance (32k / 76k) | Verdict |
|---|---:|---:|---|---|---|---|
| Baseline (`w2q8`, `dub=2048`, `F16` KV, `F32` inp) | `15,807 / 15,819 MiB` | `15,301 / 15,313 MiB` | `3,090 / 3,004 t/s` | `110.7 / 97.9 t/s` | `59.2% / 51.8%` | Baseline |
| + `--spec-draft-n-ubatch 512` (`-134` PP / `-24` TG) | `15,673 / 15,687 MiB` | `15,277 / 15,287 MiB` | `3,067 / 2,990 t/s` | `109.1 / 97.0 t/s` | `59.2% / 51.8%` | Take (bit-exact) |
| + `layer_inp_dev` `F16` (`-36` PP) | `15,637 / 15,645 MiB` | `15,277 / 15,283 MiB` | `3,078 / 2,993 t/s` | `106.1 / 97.1 t/s` | `59.2% / 51.8%` | Take (zero loss) |
| `Q8_0-mainline` + `dub=512` (`-188` weights) | `15,495 / 15,505 MiB` | `15,097 / 15,107 MiB` | `3,073 / 2,991 t/s` | `105.6 / 94.5 t/s` | `59.2% / 51.8%` | Take (zero loss) |
| **Recommended default (`Q8_0` + `dub=512` + `F16` inp)** | **`15,460 / 15,472 MiB` (`15.10 GiB`)** | **`15,098 / 15,110 MiB` (`14.75 GiB`)** | **`3,085 / 2,997 t/s`** | **`107.6 / 97.0 t/s`** | **`59.2% / 51.8%`** | **Take (default)** |
| `w2q8` + `dub=512` + `cap=16384` (`-318` KV) | `15,355 / 15,369 MiB` | `14,959 / 14,971 MiB` | `3,063 / 2,977 t/s` | `105.5 / 84.0 t/s` | `57.2% / 40.8%` | Worse than `4k` |
| `w2q8` + `dub=512` + `cap=8192` (`-350` KV) | `15,323 / 15,337 MiB` | `14,927 / 14,939 MiB` | `3,062 / 2,983 t/s` | `107.0 / 84.2 t/s` | `59.1% / 39.4%` | Opt-in (`>=128k`) |
| `w2q8` + `dub=512` + `cap=4096` (`-366` KV) | `15,307 / 15,329 MiB` | `14,911 / 14,931 MiB` | `3,068 / 2,991 t/s` | `108.8 / 87.4 t/s` | `57.2% / 41.5%` | Opt-in (`>=128k`) |
| `w2q8` + `dub=512` + `cap=1025` (`-378` KV) | `15,297 / 15,309 MiB` | `14,899 / 14,911 MiB` | `3,061 / 2,989 t/s` | `106.2 / 72.8 t/s` | `55.2% / 28.7%` | Reject (`-25%` 76k) |
| **Combined long-ctx (`Q8_0` + `dub=512` + `F16` inp + `cap=4096`)** | **`15,092 / 15,102 MiB` (`14.74 GiB`)** | **`14,730 / 14,737 MiB` (`14.39 GiB`)** | **`3,073 / 2,989 t/s`** | **`101.9 / 83.8 t/s`** | **`56.7% / 41.5%`** | **Take (`128k-256k`)** |
| `-ctkd q8_0 -ctvd q8_0` (sampled run, `MMA_F16` dequant) | `+0` to `-110` PP, **`+187 MiB` TG** | `15,471 / 15,486 MiB` | `3,084 / 3,001 t/s` | `92.4 / 88.0 t/s` | `41.2% / 41.1%` | Reject (strictly worse) |

Key takeaways:

- **Zero-loss defaults (`-347 MiB` PP / `-203 MiB` TG):** `Q8_0-mainline` drafter + `--spec-draft-n-ubatch 512` + `F16` `layer_inp_dev` (now default in `src/llama-context.cpp`) gave identical greedy output hashes and identical acceptance (`59.2% / 51.8%`) with `<1%` speed difference, bringing `-ub 2048` at 96k down to `15.10 GiB` peak / `14.75 GiB` TG.
- **Opt-in draft KV cap (`LLAMA_DSPARK_KV_CAP=4096` in `src/llama-kv-cache-iswa.cpp`):** caps DSpark layer-1 global KV to a 4k sliding window (`-366 MiB` @ 96k, `-0.5 to -1.0 GiB` @ 128k-256k) at the cost of ~10-14% TG speed on 76k (PP unaffected).
- **Why `-ctkd/-ctvd` (`q8_0`/`q4_0`) was rejected:** in `ggml/src/ggml-cuda/fattn.cu:805-810`, DSpark's block-8 decode picks `BEST_FATTN_KERNEL_MMA_F16`, which allocates a `384 MiB` (`192+192 MiB`) `F16` dequantization buffer in the draft TG compute graph (`19.25 -> 385.92 MiB`) and dequantizes the entire draft KV cache on every draft step.

### Compact SWA KV cache + per-layer prefill scratch (Option B)

Previously, raising `-ub` from `512` to `2048` permanently inflated the 40-layer SWA KV cache (`q8_0`) from `1,536` cells (`127.5 MiB`) to `3,072` cells (`255.0 MiB`) even though decode only ever reads the last `1,024` tokens. Implemented Option B across `src/llama-kv-cache.{h,cpp}`, `src/llama-memory-hybrid-kvq.{h,cpp}`, and `src/models/inkling2.cpp`:

- Keeps persistent `mem_swa` permanently at `GGML_PAD(min(kv_size, n_swa + 256), 256) = 1,280` cells (`106.25 MiB`) when `!swa_full && !v_trans && n_ubatch > 256`.
- For single-sequence prefill ubatches `> 256` tokens, concatenates `[k_old, Kcur]` / `[v_old, Vcur]` in a single per-layer `q8_0` scratch tensor (reused across all 40 SWA layers), runs FlashAttention over the concatenated window, and writes only the trailing `min(n_tokens, n_swa)` tokens back into `mem_swa`.

Measured on RTX 5090 (`-c 98304 -b 2048 -ub 2048 -np 1 -fa on --kvq --kvq-prefill-free-codes`, `Q8_0` DSpark `K=4`, `dub=512`, `F16` `layer_inp_dev`, greedy `temp=0`):

| Config | SWA KV buf | Compute buf | Peak proc VRAM (32k / 76k) | End proc VRAM (32k / 76k) | PP (32k / 76k) | Est. max ctx (`15.5 GiB`) |
|---|---:|---:|---:|---:|---|---|
| Before (default, uncapped draft KV) | `255.0 MiB` | `558.5 MiB` | `15,460 / 15,472 MiB` (`15.10 GiB`) | `15,098 / 15,110 MiB` (`14.75 GiB`) | `3,085 / 2,997 t/s` | `~160k-165k` |
| **After Option B (default, uncapped)** | **`106.25 MiB`** | **`540.4 MiB`** | **`15,317 / 15,347 MiB` (`14.96-14.99 GiB`)** | **`14,945 / 14,955 MiB` (`14.59-14.60 GiB`)** | **`3,096 / 3,011 t/s`** | **`~180k-185k`** |
| Before (`LLAMA_DSPARK_KV_CAP=4096`) | `255.0 MiB` | `558.5 MiB` | `15,092 / 15,102 MiB` (`14.74 GiB`) | `14,730 / 14,737 MiB` (`14.39 GiB`) | `3,073 / 2,989 t/s` | `~295k-305k` |
| **After Option B (`LLAMA_DSPARK_KV_CAP=4096`)** | **`106.25 MiB`** | **`540.4 MiB`** | **`14,945 / 14,975 MiB` (`14.59-14.62 GiB`)** | **`14,573 / 14,589 MiB` (`14.23-14.25 GiB`)** | **`3,100 / 3,004 t/s`** | **`~320k-328k`** |

Verification: on the 32,144-token prompt (`16` SWA-scratch ubatches), first-token logprobs match `--swa-full` to `0.0018` (`-23.2332` vs `-23.2350`), and multi-turn prefix-cache continuation (`swa_scratch` -> direct `<= 256` path) with `LLAMA_ATTN_POS_MASK_DISABLE=1` passed cleanly.

### Tiled KVQ prefill scratch (Option 1b)

Previously, `kvq_scr` in `llama_kvq_store` allocated `n_rows + n_cell` rows (`2.13 KiB/tok` -> `551 MiB` @ 256k, `1,058 MiB` @ 500k), forcing the shared `buf_scr` pool to stay inflated even during decode (`dec` only uses `~191 MiB`). Implemented Option 1b across `src/llama-kvq-store.{h,cpp}`, `src/llama-memory-hybrid-kvq.cpp`, `src/models/inkling2.cpp`, and `src/models/qwen35.cpp`:

- Caps `kvq_scr` to two ping-pong `scr_block` (`16,384`-row) host tiles plus a fixed tail region (`GGML_PAD(n_cell + rel_keep + 512, 256)` rows): **`80.2 MiB` fixed** for Inkling-2 regardless of `n_ctx` (`0 B/tok` PP scaling; `buf_scr` never grows past `dec` size).
- Uses per-tile host triggers (`build_scr_tile`) so `backend_cp` prefetches tile `t + 1` into the opposite ping-pong slot while FlashAttention runs on tile `t` (disabled cleanly when `--kvq-scr-block 0` or `n_rows + n_cell` already fits).

Measured on RTX 5090 (`-b 2048 -ub 2048 -np 1 -fa on --kvq --kvq-prefill-free-codes`, `Q8_0` DSpark `K=4`, `dub=512`, `F16` `layer_inp_dev`, greedy `temp=0`):

| Config | `kvq_scr` | Loaded / PP proc VRAM | Peak / TG proc VRAM | PP (`32k` / `76k`) | TG (`32k` / `76k`) |
|---|---:|---:|---:|---|---|
| `-c 98304` (default, uncapped draft KV) | `80.2 MiB` | `15,087 MiB` (`14.73 GiB`) | `15,295-15,325` PP / `14,923-14,947` TG | `3,059 / 2,977 t/s` | `109.5 / 113.9 t/s` (`58.8% / 62.8%`) |
| `-c 262144` (`LLAMA_DSPARK_KV_CAP=4096`) | `80.2 MiB` | `14,721 MiB` (`14.38 GiB`) | `15,135 MiB` (`14.78 GiB`) | `2,959 t/s` (`76k`) | `66.2 t/s` (64 tok, `44.3%`) |
| `-c 500000` (`LLAMA_DSPARK_KV_CAP=4096`) | `80.2 MiB` | `14,715 MiB` (`14.37 GiB`) | `15,991 MiB` (`15.62 GiB`) | `2,956 t/s` (`76k`) | `65.4 t/s` (64 tok, `44.3%`, bit-exact w/ 256k) |

### KVQ resident cap and embedding offload (measured results)

Tested `-ot ^token_embd\.weight$=CPU` and `--kvq-resident-cap 4096` on `req_32k.json` and `req_76k.json` at `-c 98304` and `-c 500000` (`LLAMA_DSPARK_KV_CAP=4096`, greedy `temp=0`):

| Config (`-b 2048 -ub 2048`, `Q8_0` DSpark `K=4`, `dub=512`, `--kvq --kvq-prefill-free-codes`) | `resident` | Loaded proc VRAM | Peak / End proc VRAM | PP (`32k` / `76k`) | TG (`32k` / `76k`, acc) |
|---|---:|---:|---:|---|---|
| Baseline `-c 98304` (tiled `121bd1080`) | `137.1 MiB` | `15,087 MiB` | `15,295-15,325` / `14,923-14,947 MiB` | `3,058.7 / 2,977.0 t/s` | `109.5 / 113.9 t/s` (`58.8% / 62.8%`) |
| `-c 98304` + `-ot ^token_embd\.weight$=CPU` | `137.1 MiB` | `15,088 MiB` (`+1`) | `15,296-15,333` / `14,924-14,955 MiB` | `3,064.6 / 2,977.1 t/s` | `110.7 / 111.6 t/s` (`58.8% / 62.8%`) |
| `-c 98304` + `--kvq-resident-cap 4096` | **`69.1 MiB`** | **`15,023 MiB` (`-64`)** | **`15,231-15,261` / `14,861-14,885 MiB`** | **`3,059.5 / 2,974.4 t/s`** | **`108.5 / 113.6 t/s` (`61.1% / 62.3%`)** |
| Baseline `-c 500000` (`cap=4096`, `76k` 64 tok) | `137.1 MiB` | `14,715 MiB` | `15,991 MiB` (`15.62 GiB`) | `2,956.0 t/s` | `65.4 t/s` (`44.3%`, `cfa946bbe33e`) |
| `-c 500000` + `-ot ^token_embd\.weight$=CPU` | `137.1 MiB` | `14,715 MiB` (`0`) | `15,989 MiB` (`15.61 GiB`) | `2,952.2 t/s` | `66.0 t/s` (`44.3%`, `cfa946bbe33e`) |
| `-c 500000` + `--kvq-resident-cap 4096` | **`69.1 MiB`** | **`14,651 MiB` (`-64`)** | **`15,924 MiB` (`15.55 GiB`, `-67`)** | **`2,948.8 t/s`** | **`64.3 t/s` (`44.3%`, `cfa946bbe33e`)** |

- **`-ot ^token_embd\.weight$=CPU` saves `0 MiB`:** `llama.cpp` already maps `token_embd.weight` (`303.67 MiB`) to `CPU_Mapped` by default (`output.weight` is the `303.67 MiB` tensor on GPU).
- **`--kvq-resident-cap 4096` adopted as default:** cuts device `resident` by `-68.0 MiB` and host `other` by `-70.0 MiB` with `<1%` speed delta, unchanged acceptance, and bit-identical 500k output hash (`cfa946bbe33e`), pushing `15.5 GiB` max context to `~480k-485k` (`~230k-235k` uncapped).

### DSpark KV cap sweep (`4k` / `8k` / `16k` under new defaults)

Re-tested `LLAMA_DSPARK_KV_CAP` under the adopted defaults (`-c 98304 -b 2048 -ub 2048 --kvq --kvq-prefill-free-codes --kvq-resident-cap 4096`, `Q8_0` DSpark `K=4`, `dub=512`, greedy `temp=0`; identical output hashes across `4096`/`8192`/`16384` on both 32k `c2d31af20b81` and 76k `c38c6d9814dd`):

| `LLAMA_DSPARK_KV_CAP` | Peak proc VRAM (32k / 76k) | End proc VRAM (32k / 76k) | Delta vs `4096` | PP (32k / 76k) | TG (32k / 76k) | Acceptance (32k / 76k) |
|---|---:|---:|---:|---|---|---|
| `None` (default, uncapped) | `15,226 / 15,256 MiB` | `14,856 / 14,880 MiB` | `+353 to +360 MiB` | `3,056.1 / 2,978.7 t/s` | `105.6 / 111.8 t/s` | `61.1% / 62.3%` |
| `4096` | `14,866 / 14,903 MiB` | `14,496 / 14,527 MiB` | baseline | `3,044.2 / 2,960.3 t/s` | `106.4 / 105.5 t/s` | `59.2% / 54.2%` |
| **`8192` (rec. for `~500k`)** | **`14,882 / 14,912 MiB`** | **`14,512 / 14,536 MiB`** | **`+9 to +16 MiB`** | **`3,038.3 / 2,967.7 t/s`** | **`105.6 / 106.9 t/s`** | **`59.2% / 58.8%`** |
| `16384` | `14,912 / 14,937 MiB` | `14,542 / 14,561 MiB` | `+34 to +46 MiB` | `3,040.5 / 2,959.4 t/s` | `111.7 / 105.4 t/s` | `62.0% / 58.8%` |

- Main default stays uncapped (`~230k-235k` max context in `15.5 GiB`). For long contexts up to `~500k`, `LLAMA_DSPARK_KV_CAP=8192` is the recommended opt-in: `+16 MiB` over `4096` (`~475k-480k` in `15.5 GiB`) while capturing all of `16384`'s `76k` acceptance gain (`58.8%` vs `54.2%`).

### 4K vision + 475k context (`--mmproj-vram-swap`)

Implemented `--mmproj-vram-swap` (env `LLAMA_ARG_MMPROJ_VRAM_SWAP`) across `llama-context`, `mtmd`, `common`, and `llama-server` (and lifted the blanket `--kvq` multimodal rejection in `tools/server/server.cpp`):

- No resident `mmproj` weights (`0 MiB` permanent GPU VRAM, skips GPU warmup and `mtmd_backends` CUDA init at load). Since 2026-11-26 also no host copy: after load the weights are freed and only their file offsets are kept (`mtmd_context::vram_swap_models`); each encode reads them from the GGUF into the transient GPU buffer. Before that the host copy cost `610 MiB` of RAM and the upload took `~47-100 ms`; now the read takes `~320 ms` from the OS file cache (a cold read from the SSD is slower), same output.
- Inside `encode_mtmd_chunk`, calls `llama_context_release_transient_vram(ctx_tgt)` (frees LLM device compute buffers and `--kvq-prefill-free-codes` `buf_codes`), transiently creates `mtmd`'s GPU backend, reads `mmproj` weights from the file into it (`609.7 MiB` `F16`), runs FlashAttention ViT encode, and immediately frees weights, compute buffers, and the GPU backend (`~6-8 ms`, releasing its cuBLAS workspace / CUDA pool so TG VRAM stays minimal) before LLM prefill resumes.

Measured on RTX 5090 at `-c 475000 -b 2048 -ub 2048 -np 1 -fa on --kvq --kvq-prefill-free-codes --kvq-resident-cap 4096`, `LLAMA_DSPARK_KV_CAP=8192`, `Q8_0` DSpark `K=4`, `dub=512`, `mmproj-Inkling-2-Mini-F16.gguf` (`609.7 MiB`), `--mmproj-vram-swap`, greedy `temp=0`, running a `3840x2160` 4K PNG -> `req_76k` text -> second 4K PNG:

| Resolution mode | Loaded proc VRAM | Transient vision VRAM (wt + comp) | Upload / Encode / Release | Peak proc VRAM (4K / 76k / 4K-after-76k) | PP / TG (4K img) | 76k text PP |
|---|---:|---:|---|---:|---|---|
| Default 4K (`4096` cap, `3400x1920`, `4,122` tok) | `14,673 MiB` (`14.33 GiB`) | `609.7 + 393.6 MiB` (`0.98 GiB`) | `47.5 / ~307 / 6.4 ms` | **`15,813 / 15,825 / 15,825 MiB` (`15.44-15.45 GiB`)** | `1,799-1,991` / `138.4-140.2 t/s` | `3,181.9 t/s` |
| Native 4K (`--image-max-tokens 5184`, `3840x2160`, `5,226` tok) | `14,673 MiB` (`14.33 GiB`) | `609.7 + 500.1 MiB` (`1.08 GiB`) | `47.1 / ~579 / 7.9 ms` | **`15,812 / 15,840 / 15,840 MiB` (`15.44-15.47 GiB`)** | `1,804-1,971` / `133.5-136.8 t/s` | `3,180.4 t/s` |

- Both default 4K and native `3840x2160` 4K fit cleanly under `15.5 GiB` (`15.44-15.47 GiB` peak) at `-c 475000` with zero LLM or drafter weight eviction, read all shapes and verbatim text (including small `36pt` text) accurately, and produced identical deterministic outputs before and after the 76k text turn.

### Host RAM (Windows, `-lm none`)

Measured on RTX 5090 / Windows 11, commit `cff196458`, recommended settings above plus `LLAMA_DSPARK_KV_CAP=8192 -lm none -mm mmproj-Inkling-2-Mini-F16.gguf --mmproj-vram-swap --image-max-tokens 5184`. Requests: War and Peace text prefill (max_tokens 1), then the same text + a native `3840x2160` PNG (`5,203` tokens) + a question (cache reuse, only the image is new). Working set of `llama-server` (`PeakWorkingSet64` for peaks):

| | `-c 409600`, 385k text | `-c 500000`, 490k text |
|---|---:|---:|
| After load (incl. `610 MiB` mmproj weights in host RAM) | `1.9 GiB` | `1.9 GiB` |
| End of text prefill | `10.7 GiB` | `12.7 GiB` |
| kvq pinned pool | `7.7 GiB` (48 pages) | `9.8 GiB` (61 of 62 pages) |
| 4K image (encode + `5,203` image tokens), peak | **`11.2 GiB`** (`+0.5 GiB`) | **`13.4 GiB`** (`+0.7 GiB`) |
| After the image | `11.1 GiB` | `13.3 GiB` |
| Commit charge (private bytes), peak | `27.6 GiB` | `30.3 GiB` |
| GPU (whole card, incl. `441 MiB` desktop), peak in decode | `15,853 MiB` | `16,253 MiB` (process `~15.4 GiB`) |
| Prefill text / image tokens | `3,941 / 1,501 t/s` | `3,486 / 1,334 t/s` |

- Where the RAM goes: `~1.3 GiB` process + CUDA, `0.6 GiB` mmproj, the pinned pool (`164.2 MiB` per 8192-row page = `20.5 KiB/tok`, the q8_0 rows of the 8 global layers, allocated on demand), `~1 GiB` prompt-path overhead (`~0.3 GiB` on the first request, context checkpoints `~90 MiB`, rest grows slowly with the prompt), and the image (`+0.5-0.7 GiB`, most of it stays).
- **Formula (recommended settings, `-lm none`, mmproj read from the file), peak working set of `llama-server`:**

  `RAM ~= 2.6 GiB + 20.5 KiB x n_tok` (+ `0.5 GiB` for each image turn, + the prompt cache up to `--cache-ram`)

  - `n_tok` = tokens in the context (the pool pages are allocated on demand); worst case `n_tok = -c`. `20.5 KiB/tok` = one `164.2 MiB` page per 8192 tokens (q8_0 rows of the 8 global layers + host copy of the codes and tables), rounded up to whole pages, one page ahead.
  - `2.6 GiB` = `1.3 GiB` after load + `~0.3 GiB` on the first request + `~0.5-0.9 GiB` prompt path + context checkpoints (`~0.1-0.2 GiB`) + page rounding. It is `~2.4 GiB` at 100k and `~2.7 GiB` at 490k.
  - Check: 385k -> `2.6 + 7.5 = 10.1 GiB` (measured `10.1`); 490k -> `2.6 + 9.6 = 12.2 GiB` (measured `12.2`), with the image `12.7` (measured `12.7`). 500k worst case with images: `~12.9 GiB`.
  - Commit charge (private bytes) ~= process VRAM + working set + `~0.7 GiB`.
- **Host RAM for a full 500k context with images: `~12.8 GiB` for `llama-server`** (`~13.5 GiB` before the mmproj file read; text only, no `-mm`: `~12.1 GiB`). A machine with 32 GB RAM is fine; 16 GB is not enough.
- Commit charge is not RAM: under WDDM the device allocations count toward the process commit (private bytes = VRAM + private working set + `~0.5-0.9 GiB` in every sample). The commit limit (RAM + page file) must cover `~31 GiB` for `llama-server` alone at 500k, so do not disable or cap the page file.
- **mmproj read from the file (no host copy, see the 4K vision section):** same 500k + image run, identical reply: after load `1.3 GiB` (`1.9` during load, the loader still reads the weights once), end of prefill `12.2 GiB`, **image peak `12.7 GiB` (`13,046 MiB`, `-631 MiB`)**, after the image `12.6 GiB`, commit peak `29.6 GiB`, GPU unchanged (`16,253 MiB`). Image read + upload `319 ms` (was `50-101 ms`), image prefill `1,253 t/s` (was `1,334`).
- Breakdown of the `13,677 MiB` peak (500k + image, before the mmproj change; now the mmproj row is `0`), from the load log, the kvq pool lines and the sampler:

  | Item | MiB |
  |---|---:|
  | CUDA runtime / driver, DLLs, process heap (after load, not traced) | `~850` |
  | mmproj `F16` weights in host RAM (`--mmproj-vram-swap`) | `610` |
  | `token_embd.weight` (`CUDA_Host`, the embedding lookup runs on the CPU) | `304` |
  | kvq ring (`72`) + staging and resident mirrors (`81`) | `153` |
  | host compute / output buffers (target + drafter) | `~55` |
  | **kvq pool, 61 pages:** q8_0 rows `136.0 MiB`/page (`8,296`), host copy of the codes `26.0`/page (`1,587`), tables `2.2`/page (`134`) | **`10,015`** |
  | context checkpoints (`3.8 + 4 x 43.8`) | `179` |
  | prompt path (`~330` on the first request + `~550` with a long prompt, fixed, not traced) | `~877` |
  | image (`48 MiB` embeddings, `~95 MiB` f32 preprocessed image, rest not traced; `~535` stays) | `~634` |
- **Prompt cache (`--cache-ram`, default `8192 MiB`) is not in the numbers above:** with `--kvq` the server still saves the whole slot state when a request does not continue the cached prompt (a different conversation). 50k conversation -> unrelated 20k one: `1,248 MiB` saved (`~25 KiB/tok`) and kept in RAM. In real use with several conversations, add up to `8 GiB`. Use `--cache-ram 0` (or e.g. `2048`) on a 32 GB machine; the cost is a new prefill when the user goes back to an older conversation.
- Without `-lm none` add `13.0 GiB` to every working set row above (clean file pages, the OS can trim them under pressure, but they show as used).

### Drafter comparison (2026-11-26)

Recommended settings (`-c 98304`, uncapped draft KV, `--spec-draft-n-ubatch 512`), greedy, `cache_prompt=false`, 2 reps; decode speedup over no drafter. Junie = the two replayed requests (short generations, 120-180 tokens ending in a tool call); book = Moby Dick 32k / War and Peace 76k summary prompts, 2,500 generated tokens (`x` = "chapter-by-chapter, at least 1500 words").

| Drafter (file size) | Junie 32k / 76k | book32 / book32x | book76 / book76x | Peak VRAM 76k |
|---|---|---|---|---:|
| none (`102` / `95` t/s) | - | - | - | `14,581 MiB` |
| **Ninnix96 DSpark (333 MB)** | `1.61x / 1.51x` | `1.24x / 1.56x` | `1.31x / 1.19x` | **`15,449 MiB`** |
| Unsloth DSpark `Q8_0` (504 MB) | `1.57x / 1.64x` | `1.21x / 1.56x` | `1.11x / 1.15x` | `15,611 MiB` |
| Unsloth DSpark `BF16` mainline (749 MB) | `1.54x / 1.45x` | | | `15,837 MiB` |
| DSpark `BF16-w2q8` (701 MB, local) | `1.55x / 1.53x` | | | `15,791 MiB` |
| DFlash2 `K=4` (notfakerv, 538 MB, needs the port on `erokhins/inkling-dflash2`) | `1.85x / 1.33x` | `1.06x / 1.18x` | `1.22x / 1.22x` | `15,283 MiB` |
| DFlash2 `K=7` | `1.91x / -` | `0.95x / 1.10x` | `1.05x / 0.91x` | `15,235 MiB` |

- AtomicChat `Q8_0` behaves exactly like the Unsloth `Q8_0`. IsValorum `Q4_0` and InterpreticAI `Q8_0` use the old layout and do not load (`markov_w1` shape `201024`).
- Ninnix96 and Unsloth `Q8_0` are within rep noise (up to `23 t/s` between two reps of one config); Ninnix96 is chosen for the lower VRAM. DFlash2 is not worth it: it trails DSpark on the book prompts at 32k and `K=7` is slower than no drafter on some prompts.
- With a drafter the greedy output moves away from the no-drafter output within the first `~60-220` characters (also with the standard KV cache, so not a `--kvq` effect), and drafter reps differ from each other; each config is timed on a slightly different text. Raw data: `tmp/inkling/drafters/` (`RESULTS.md`, `RESEARCH.md`).

### 16 GB (RTX 5070 Ti) verification and VRAM measurement (2026-11-27)

Re-tested `-c 500000` on `6d775739d` with `Inkling-2-Mini-UD-Q3_K_XL.gguf`, `Inkling-2-Mini-DSpark-333M.gguf` (Ninnix96), and `mmproj-Inkling-2-Mini-F16.gguf` (`--mmproj-vram-swap`), using a `~478k`-token `war_and_peace.txt` prompt + 64 generated tokens on RTX 5090 (`10 MiB` idle baseline measured *before* starting `llama-server`, `0.25 s` `nvidia-smi` polling):

| Build / Config | Loaded (delta) | 21-tok (delta) | 12k TG (delta) | 478k PP (delta) | 478k TG (delta / total) |
|---|---:|---:|---:|---:|---:|
| Local CUDA 12.9 (`120a-real`), `-c 500000`, `CAP=8192` | `14,871 MiB` | `14,883 MiB` | `15,785 MiB` | - | **`16,075 MiB`** (`16,085` total) |
| CI CUDA 13.5 (multi-arch `b12774-6d77573`), `-c 500000`, `CAP=8192` | `14,925 MiB` | `14,995 MiB` | `15,897 MiB` | `15,259 MiB` | **`16,189 MiB`** (`16,199` total) |
| CI CUDA 13.5 (`b12774-6d77573`), `-c 510000`, `CAP=32768`, `-ctk/-ctv q8_0` | `14,973 MiB` | `15,043 MiB` | `16,021 MiB` | `15,351 MiB` | **`16,307 MiB`** (`16,317` total) |

Key findings:

- **Why the earlier 500k `"process"` figure (`15,644 MiB`) was `~431 MiB` too low:** the `0a9f9219c` measurement subtracted `441 MiB` taken *right after* `llama-server` started, when the server had already created its own `~430 MiB` CUDA context. Both runs hit the exact same `16,085 MiB` `nvidia-smi` total on local CUDA 12.9; the true VRAM added by `llama-server` above idle is **`16,075 MiB`** (not `15,644 MiB`).
- **CI CUDA 13.5 multi-arch vs local CUDA 12.9 single-arch (`+114 MiB`):** explicit `CUDA0` buffers from `-lv 4` are identical (`15,045 MiB` at `-c 500000`, `CAP=8192`). The flat `+112-114 MiB` gap is CUDA 13.5 context init (`+54 MiB` at load) plus `cublasLt64_13` workspace/kernels on the first request (`+58 MiB` at 21 tok; also `+3.6%` PP / `+5.0%` TG at 12k), **not** multi-arch fatbins (`CUDA_MODULE_LOADING=LAZY` is active by default; forcing `EAGER` adds `+422 MiB` on single-arch local and `+506 MiB` on CI multi-arch).
- **Why filled depth matters (`+290 MiB` from 12k to 478k):** on prompts `<8k` (`n_coarse == 0`), `codes_alloc()` is never called. Any prompt `>8k` allocates the full `-c` `buf_codes` (`1,748 MiB` at `-c 500000`, `1,776 MiB` at `-c 510000`), and filling from 12k to 478k adds another `~290 MiB` in decode (`kvq-attn` scratch `scp_a`/`po_a`, masks/scores, and `ggml_cuda_pool` VMM retention). Raising `LLAMA_DSPARK_KV_CAP` from `8192` to `32768` adds `+96 MiB` (`24k x 4 KiB/tok`).
- **RTX 5070 Ti (16 GB) physical VRAM test:** `nvidia-smi memory.total` on an RTX 5070 Ti is **`16,303 MiB`** (`15.92 GiB`), minus NVIDIA driver `memory.reserved` (`~330-417 MiB`) -> **`~15,886-15,974 MiB` usable headless** (`~15.6 GiB` on Windows), or **`~15,400-15,600 MiB`** with Windows DWM/desktop. Simulating this cap on the RTX 5090 (`..\occupy.ps1 15.6` -> `16,415 MiB` held, `15,776 MiB` free) with the CI CUDA 13.5 binary (`-c 500000`, `CAP=8192`, 478k prompt):
  - Load (`+14,925 MiB`) and prefill (`+15,259 MiB`, `3,654 t/s`) stayed in physical VRAM.
  - Decode hit the physical ceiling at `32,026 MiB` total (`+15,611 MiB` on GPU) and **silently spilled `~578 MiB` into Windows Shared GPU Memory**, slowing `codes_alloc` by **10.9x** (`17.2 -> 187.6 ms`) and cutting TG speed by **`~19%`** (`64.2-65.6 -> 52.9 t/s`). On Linux (no WDDM Shared GPU Memory fallback), this fails with a hard CUDA OOM.
  - To stay strictly inside physical VRAM on an RTX 5070 Ti with the CI CUDA 13.5 build, use **`-c 380000`-`400000`** with `LLAMA_DSPARK_KV_CAP=8192` (or **`-c 350000`-`380000`** with `LLAMA_DSPARK_KV_CAP=32768`).

#### How to measure VRAM correctly

1. **Take the baseline before starting `llama-server`:** query `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` while the GPU is idle (or with only the desktop/occupier running) *before* launching `llama-server`. Never take the baseline after the server has started, as that hides `~430-550 MiB` of `llama-server`'s own CUDA context and library allocations.
2. **Compare against real usable VRAM, not `16,384 MiB`:** check `nvidia-smi --query-gpu=memory.total,memory.reserved,memory.used,memory.free --format=csv`. On a 16 GB RTX 5070 Ti, usable VRAM is `memory.total (16,303) - memory.reserved (~330-417) - desktop` (`~15,400-15,900 MiB`).
3. **Fill the context to `~95%` of `-c` and run decode:** short prompts (`<8k`) skip `codes_alloc()`, and medium prompts (`12k-76k`) miss `~250-290 MiB` of full-depth decode scratch/pool growth. Always send a prompt near `-c` and generate at least 32-64 tokens.
4. **Poll continuously across the whole request:** sample `nvidia-smi` every `0.25 s` from request start until `1-3 s` after the reply so both the prefill peak and the post-`codes_alloc` decode peak are captured.
5. **Measure unconstrained first, then verify under an occupier (Windows WDDM spill trap):** on Windows, WDDM silently spills VRAM overflow into Shared GPU Memory and caps `nvidia-smi memory.used` near `memory.total - memory.reserved`, making an overflowing run look like it "fit". Always confirm `unconstrained_peak - baseline <= target_usable_vram`, and when testing under a VRAM occupier, check that `codes_alloc` latency (`-lv 4`) and TG tok/s do not regress vs the unconstrained run (or verify on Linux where overflow hard-OOMs).
6. **Measure the actual release binary:** CI CUDA 13.5 builds use `~114 MiB` more VRAM than local CUDA 12.9 builds on the same commit, so budget against the binary you ship.

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
