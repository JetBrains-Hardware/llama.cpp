# Optimized tree configuration for Qwen3.8-27B

This branch contains the Q3 prefill and DFlash2 tree optimization groups and their required correctness support. Prepared IQ3 prefill weights, prepared IQ3/IQ4 verification kernels, separate DFlash workspaces, tree-forward/commit-once experiments, the optional BF16 recurrent provider and the fork's mixed FP8/NVFP4 pipelines are not included. Their implementations and switches have been removed, rather than disabled by a launcher.

Use the Unsloth `Qwen3.8-27B-UD-Q3_K_XL.gguf` target and the original Inco Qwen3.8 DFlash2 drafter converted to Q4_K_M. Keep both files in the Hugging Face cache, outside this checkout.

## Build and run

Build on the CUDA device that will serve the model. Set `CUDA_ARCH` to `89-real` for RTX 4090 or `121a-real` for RTX Spark N1X. Put build output outside this checkout.

```sh
cmake -S . -B "$LLAMA_BUILD_DIR" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$LLAMA_BUILD_DIR" --target llama-server -j 4

scripts/serve-qwen-dflash2-tree.sh \
    "$LLAMA_BUILD_DIR/bin/llama-server" "$TARGET_GGUF" "$DRAFTER_GGUF"
```

For multiple requests, pass `-np N` and a context capacity covering their total unified KV budget, for example `-np 4 -c 163840`.

Additional server options are passed through after the defaults, for example `--port 8081` or `--seed 43`. Run in a clean shell without experimental environment overrides or preloaded profiling libraries. Use the libraries produced by the same build as the server.

## Enabled configuration

| Area | Configuration |
| --- | --- |
| Prefill | GLU/activation-quantization fusion, GQA grouping 2, and split convolution |
| Speculation | DFlash2 tree with 15 draft nodes plus the pending-token root and shared recurrent-state slots; normal quantized matrix dispatch |
| Decode | Compact recurrent replay, asynchronous CUDA inputs and target-feature injection; the branch's normal few-row matrix dispatch, IQ4 calibration and graph reuse |
| Prompt reuse | Enabled; checkpoint allocation reuse, restored-checkpoint retention, pinned/device checkpoint storage and sampler lifetime optimizations retain their normal branch behavior |
| Serving | C1 defaults, with multiple request slots supported; 40,960 context capacity, batch and microbatch 2,048, full CUDA layer placement, unified F16 K/V |
| Sampling | Temperature 1, top-p .95, top-k 20, min-p 0; zero presence/frequency penalties, repeat penalty 1; thinking enabled, low effort |

Device checkpoints use additional VRAM and fall back to the host when their allocation and headroom checks fail.

The output limit remains a request option. Use `max_tokens: 1024` to match the grouped performance measurements, with natural stops retained. Prompt-processing timing and checkpoint-inclusive server prompt-phase timing are different metrics.

See [DFlash2 tree execution](dflash-tree.md) for tree construction, verification and state handling.
