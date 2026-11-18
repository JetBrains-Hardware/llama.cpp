# Experimental CUDA input pipeline

Two opt-in switches control separate parts of input preparation:

- `GGML_CUDA_ASYNC_INPUTS=1` stages small contiguous host inputs before queuing their GPU copies.
- `LLAMA_DFLASH_ASYNC_INJECT=1` prepares DFlash2 target-feature injection before the server waits for target completion.

The staging path copies the caller's bytes into owned pinned memory before returning. The caller can overwrite or release its input immediately. Copies run on the consuming CUDA stream, after earlier users of the destination and before the next graph. Each CUDA buffer using this path owns 64 slots of 64 KiB, totaling 4 MiB of pinned staging data, plus completion events. A slot can be reused after its copy event completes. A full ring waits for a copy to finish; buffer destruction drains pending copies before freeing the staging data. Larger or noncontiguous inputs and unavailable pinned memory retain the synchronous path.

The scheduler discovers this operation through the optional `ggml_backend_copy_input_async` registry extension. A successful call must capture host input bytes before returning and order the destination copy on the backend stream. Backends without this extension retain their existing input-copy behavior.

The DFlash2 switch requires C1, device target features, operation offload, and all draft layers and output on the same CUDA device as the target features. After allocation, a single colocated CUDA split can wait for the producer event after input copies and before graph execution. Other split plans wait on the host before execution. Existing reader-completion events still protect the target feature buffer from reuse. Metal retains its host wait.

These changes preserve tensor bytes, model arithmetic, sampling and cache policy. They do not change the proposal width or the drafter checkpoint. Both switches are disabled by default.

Build with the normal CUDA configuration. Test input ownership under queued GPU work, immediate host overwrite and release, ring wrap, unsupported layouts and pinned-memory fallback:

```sh
GGML_CUDA_ASYNC_INPUTS=1 ./build/bin/test-backend-tensor
GGML_CUDA_ASYNC_INPUTS=1 GGML_CUDA_NO_PINNED=1 ./build/bin/test-backend-tensor
```

Enable the switches on the same server command used for the control. This example makes the Qwen3.8-27B cache and sampling settings explicit; retain any independent optimization flags used by the control:

```sh
GGML_CUDA_ASYNC_INPUTS=1 LLAMA_DFLASH_ASYNC_INJECT=1 ./build/bin/llama-server \
    -m "$TARGET_GGUF" -md "$DRAFTER_GGUF" \
    --spec-type draft-dflash --spec-draft-n-max 7 \
    --spec-draft-sampling probabilistic --spec-draft-backend-sampling \
    --backend-sampling --kv-unified -ngl 99 -ngld 99 -np 1 -c 40960 \
    -b 2048 -ub 512 --fit off -fa on -ctk f16 -ctv f16 --cache-prompt \
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 0 --repeat-penalty 1 --reasoning on --reasoning-effort low
```

Keep the model, quantization, sampling parameters, context capacity and input requests fixed between arms. Measure newly processed prompt tokens and generation separately. Inspect acceptance counts and numerical outputs as well as timing. Qualification is specific to the tested model, hardware and serving configuration.
