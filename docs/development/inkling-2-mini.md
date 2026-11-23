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
KV looks large for a model where only 1 layer in 6 is global attention (hypothesis, not yet checked: sliding-window layers
may get full-context cache slots).

### Real Junie requests (replayed)

Two requests captured from Junie CLI (profile `local-inkling-2-mini-q3kxl`, through mitmproxy) are replayed with
`cache_prompt=false` (full prefill each time) and the original sampling and seed. Server: `-c 98304 -np 1`.

| Request | Prompt tokens | Prefill | Decode | Generated |
|---|---|---|---|---|
| flow 144 (32k) | 32,144 | 1,400-1,460 t/s (22.0-23.0 s) | 78.5 t/s | 1,939 (3 reps, identical) |
| flow 197 (76k) | 75,985 | 916 t/s (82.9 s) | 66.9 t/s | 802 (2 reps, identical) |

Server VRAM with `-c 98304` is about 27 GB in both cases. The generated length differs from the original run (1,489 and 3,174)
because the original had cached prefixes; speeds, not lengths, are the comparison.
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
