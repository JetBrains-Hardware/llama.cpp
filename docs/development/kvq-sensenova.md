# Two-tier KV cache (kvq) on SenseNova-Loop-9B

Branch `erokhins/kvq-sensenova`: our `kvq` branch (see `kvq-two-tier-cache-guide.md`) with ggml-org PR 32588 (SenseNova Loop support, open) merged, and `--kvq` enabled for the `sensenova_loop` architecture. Prototype state of 2026-11-20.

## What changes for this model

SenseNova-Loop-9B is a qwen35 hybrid (gated delta net + gated attention) with a looped trunk: 56 logical layers, layers 4..27 run twice with shared weights, each visit has its own KV cache; mHC two-stream routing on layers 4..27. That gives 14 attention visits with 4 kv heads of dim 256, so the KV cache per token is 87.5% of Qwen3.8-27B's on a model a third the size.

The two-tier cache works per logical layer, so each of the 14 visits has its own coarse codes, tables, resident set and early query. The early query of attention layer L is computed at the input of layer L - 2 from the mean of the two mHC streams. The kvq code needs no other change for this model.

## Build

As in the kvq guide (Windows, VS 2022, CUDA 12.9, Ninja):

```
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_HTML=OFF
cmake --build build-cuda --target llama-server llama-kvq-eval
```

`-DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_HTML=OFF`: the prebuilt UI download of this base does not match the asset script and breaks the build.

## Run (recommended settings)

Model: `SenseNova-Loop-9B-Q5_K_M.gguf` from the Hugging Face repo `sensenova/SenseNova-Loop-9B-GGUF`.

```
llama-server --fit off -m SenseNova-Loop-9B-Q5_K_M.gguf --alias SenseNova-Loop-9B-Q5_K_M ^
  --host 127.0.0.1 --port 8080 --ctx-size 256000 --n-gpu-layers all --parallel 1 ^
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja ^
  --batch-size 4096 --ubatch-size 1024 --kvq ^
  --spec-type draft-mtp --spec-draft-n-max 1 -ctkd f16 -ctvd f16
```

- MTP draft of one token (the model's own MTP layer, blk.56). Greedy decode t/s, RTX 5090 (short: 4 generic prompts of up to 400 tokens; 24k / 100k: prompts of 24,183 / 99,395 tokens, 300 tokens each, `--verbose`); draft acceptance 0.70 - 0.95:

  | config | short | 24k | 100k |
  |---|---|---|---|
  | plain | 116 | 107 | 84 |
  | plain + MTP n_max 1 | 160 | 141 | 108 |
  | `--kvq` | 106 | 91 | 80 |
  | `--kvq` + MTP n_max 1 | 135 - 157 | 132 | 120 |

- The MTP draft context has no compute buffers per phase: it alternates 1- and 2-token ubatches, and with the per-phase buffers it switched the phase at every step (about 7 ms per step, `--kvq` + MTP 88 / 85 / 80 t/s). n_max 1 is the best: every further draft token costs about 6.5 ms per step and is accepted less often (plain + MTP n_max 3: 143 t/s, acceptance 0.48).
- This needs the commit that adds `sensenova_loop` to `llm_arch_supports_rs_rollback` (PR 32588 does not have it). Without it the server checkpoints the whole recurrent state (88 MiB) every step and restores it and decodes again after a rejected draft, and MTP makes the decode slower (plain + MTP n_max 3: 71 t/s). With it, the greedy output of MTP n_max 1 is byte-identical to the greedy output without MTP, with and without `--kvq`.
- Sampling of the model card: temperature 1.0, top_p 0.95.
- Memory at 256k: VRAM holds the weights (6.7 GB) and the coarse codes (K 2 bit, V 1 bit plus a gain: about 5.6 KB per token, 1.4 GB at 256k) instead of a q8_0 KV cache of about 30 KB per token (7.4 GB). The q8_0 rows of all positions are in pinned host RAM (about 7.4 GB at 256k).
- The limits of the kvq guide apply: one slot, CUDA only, no context shift or cache reuse, no `--mmproj`, `--fit off`.

### Junie

Profile `~/.junie/models/local-sensenova-loop-9b-kvq.json` (user scope; `.junie/models/` of a project works too):

```
{
  "displayName": "SenseNova Loop 9B (kvq)",
  "providerName": "Local",
  "id": "SenseNova-Loop-9B-Q5_K_M",
  "baseUrl": "http://localhost:8080/v1/chat/completions",
  "apiType": "OpenAICompletion",
  "temperature": 1.0,
  "maxContextLength": 240000,
  "extraBody": { "top_p": 0.95 }
}
```

Select it with `junie --model custom:local-sensenova-loop-9b-kvq` or `/model` -> Custom.

## Quality

Teacher-forced KL of the next-token distribution against a plain f16 cache (`llama-kvq-eval`, 7 private traces of 20k - 100k tokens, 2505 scored reply positions, one token per batch):

| cache | same top-1 token | mean KL | KL / q8_0 |
|---|---|---|---|
| plain q8_0 | 2499 / 2505 | 2.41e-4 | 1.00 |
| kvq K2V1 (default) | 2498 / 2505 | 2.75e-4 | 1.14 |
| kvq K2V2 | | 2.57e-4 | 1.07 |
| kvq K3V2 | | 2.54e-4 | 1.06 |
| plain q4_0 | 2486 / 2505 | 1.23e-3 | 5.1 |

No trace shows a significant difference between kvq and plain q8_0; plain q4_0 is clearly worse. At 8 tokens per batch the result is the same. `llama-kvq-eval --kvq-bits-k N --kvq-bits-v N` sets other bit counts (there is no server flag yet).

Speed in `llama-kvq-eval` at 8 tokens per batch: kvq 5 - 12% slower than plain q8_0 (100k: 268 against 281 t/s).

## What the attention looks like

Measured on dumps of both models with the same code (24k and 50k traces): SenseNova has the properties the two-tier cache relies on, like Qwen3.8-27B. Sparse decode attention (about 115 keys above 1e-3 per query head), the early query at lead 2 finds 93 - 97% of the true top-64 keys in its top 1024, K is as compressible in the query metric. The loop does not help: the two visits of a layer attend to mostly different keys (top-1024 overlap about 0.3), so every visit needs its own selection, and sharing the K tables between the visits costs 1.2 - 1.4x KL.

## Not tested

The server with long prompts at 256k (the VRAM of each phase), long sessions, slot save / restore, `--kvq-layer-bufs`, and other quantizations of the model. Measurement material: `../kvq-scratch/sensenova/` (analysis, codec simulation, port gates, e2e KL).
