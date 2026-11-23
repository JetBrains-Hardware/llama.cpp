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

## Local tools (not committed)

Everything lives in the git-ignored `tmp/inkling/` of this worktree (`/tmp/` is in `.gitignore`):

- `tools/` - `replay.py` (replay a captured request, print prefill/decode speed), `dump_reqs.py` (mitmdump addon: write request bodies),
  `reqs.py` and `join.py` (match server log lines to flows), `ginfo.py` (print GGUF metadata and tensor types), `session.sh`
  (server session with VRAM), `build-inkling.bat`, `gpurun.sh` (GPU lock; the lock directory is shared in `kvq-scratch`).
- `replays/test2/req_32k.json`, `req_76k.json` - the two test requests; `replays/picked`, `reqs*` - all dumped bodies.
- `logs/` - Junie run logs, server logs, replay results.

Junie setup: profile `~/.junie/models/local-inkling-2-mini-q3kxl.json` points at `http://localhost:8091/inkling/v1/chat/completions`;
mitmproxy-utils (`windows-support` branch, local) routes `localhost:8091/inkling` to `localhost:8765` (plain http, no CA needed).
Select with `junie --model custom:local-inkling-2-mini-q3kxl`; `--slow-model` is accepted but undocumented.

Reproduce: `build-inkling.bat llama-server`, start `llama-server -m <gguf> -ngl 99 -fa on -c 98304 -np 1 --port 8765 --jinja`,
then `python tmp/inkling/tools/replay.py tmp/inkling/replays/test2/req_32k.json 4000 3`.
