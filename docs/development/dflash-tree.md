# Experimental CUDA DFlash2 tree verification

`LLAMA_DFLASH_TREE=1` enables per-request tree proposals and batched verification for dense Qwen35 targets with a DFlash2 drafter. It requires unified KV, the existing CUDA compact GDN replay path, full CUDA placement of recurrent layers, and a 128-element recurrent state dimension. It is disabled by default.

For example, retain the usual model paths and serving options, and add:

```sh
LLAMA_DFLASH_TREE=1 ./build/bin/llama-server \
    -m TARGET.gguf -md DFLASH2.gguf \
    --spec-type draft-dflash2 --spec-draft-n-max 15 \
    --spec-draft-sampling probabilistic --spec-draft-backend-sampling \
    --backend-sampling --kv-unified -np 1 -ngl 99 -ngld 99 \
    -c 40960 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on \
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
    --repeat-penalty 1 --presence-penalty 0 \
    --reasoning on --reasoning-effort low --cache-prompt
```

This example's sampling settings are for the Qwen3.8-27B workload. Use the target model's own recommended sampling settings. Existing independent CUDA optimization switches may be retained in both control and candidate runs.

The node budget includes draft nodes only; the pending token is an additional root row. A budget of 15 therefore verifies at most 16 rows. The implementation bounds a window to 32 rows. The runtime drafter block can exceed its training block size; this is a proposal-policy experiment, and the target still verifies every emitted token. The selector packs every requested runtime row instead of truncating the result to the training block size.

## Execution

The drafter's existing conditional candidate lattice supplies a best-first tree. Candidate priority uses normalized path scores, with position/token-dependent Gumbel noise shared with the target sampler. Proposal ranking uses `(unary + 0.6 * selector_edge) / temperature`, adds noise weighted by 0.7, then divides by 1.5. The separate unary and edge values are retained after the original chain scores in the packed lattice. The noise counter uses TensorFold's `keyBits`/`uniform24` arithmetic and clamps the F32 draw below one; candidate ties retain candidate order and open-node ties use parent, token and row. These are proposal heuristics adapted from TensorFold's CUDA DFlash2 implementation. They do not change the target's configured temperature or probability filters.

Each request owns its tree and pending recurrent-update tape. Drafting can serve several requests, and target verification batches their windows; each accepted path commits to its own sequence. The library accepts batch-global parent indices and rejects cross-sequence parents. `llama_tree_commit_seq()` selects a path for a named sequence; `llama_tree_commit()` remains a shorthand for sequence 0. Ordinary prompt rows can share the submitted batch and are marked with parent -2.

Use `-np N` to configure concurrent requests and size `-c` for their total unified KV capacity. For example, `-np 4 -c 163840` retains the 40,960-token budget per request of the C1 example. All requests share model weights; capacity admission still depends on available memory.

Target verification uses one batch. Storage positions identify distinct tree rows, while RoPE positions follow tree depth. The attention mask exposes the committed context and a node's ancestors. Convolution windows gather the same ancestor path. The GDN kernel starts from the committed state, replays the pending prefix, and evaluates each node's ancestor path; it records compact update operands instead of storing a full state for every node.

After sampling, only the selected root-to-leaf path remains in the KV metadata. Its keys were already encoded at their logical depth positions, so commit needs no RoPE correction or KV movement. Convolution rollback selects the retained node's snapshot. The next recurrent pass replays the retained node indices from the compact operand tape. State serialization writes those operands in logical sequence order. Only selected target feature rows are injected into the drafter cache.

The target uses Gumbel-max over its filtered probability distribution. Noise is keyed by request seed, absolute token position and token ID, so changing the tree's shape does not change a position's random draw. For grammar sampling, it retains the normal single-token validity check and computes a full grammar-filtered distribution only on rejection; the fallback uses a separate counter key. Histories with repetition/presence/frequency penalties, DRY, adaptive sampling, Mirostat or XTC retain the chain path. The conventional probabilistic chain remains the default control when the environment switch is unset.

The tree path currently consumes filtered probabilities from the existing sampler pipeline; it does not introduce a fused GPU sampler. The recurrent kernel repeats ancestor updates for different nodes. Both costs must be included in downstream measurements.

Tree verification uses a recurrent tree scan. A bounded depth-first schedule assigns shared-memory slots to parent states and reuses a slot after its last child reads it. The CUDA kernel folds the pending prefix once, visits each tree node once, and writes results in the original batch order. Slots hold only the state shard owned by each block; no per-node full-state tensor is allocated. Slot-count buckets are part of graph reuse validation.

Contiguous F32 inputs to IQ4_XS projections that are not aligned for the native vectorized quantizer are copied to aligned scratch before normal dispatch.

## Validation

```sh
cmake --build build --target llama-server test-dflash-tree test-speculative-tree -j8
./build/bin/test-speculative-tree
./build/bin/test-dflash-tree TARGET.gguf
TREE_TEST_KV_Q8=1 ./build/bin/test-dflash-tree TARGET.gguf
```

The target-model test supplies hand-built trees and uses real target weights. It does not exercise the neural drafter or prove proposal-policy equivalence. It checks multiple requests, cross-request isolation, independent commits, compaction, shared prefixes, sequence copies and whole/partial state restore. It also checks irregular, wide and deep trees, including 32 rows and all shared-slot bucket sizes. It changes every non-ancestor token to test sibling isolation, compares each node against a serial ancestor path, checks five seeded sampled tokens per node, commits a selected branch, and compares both live and restored continuations. The selected sequence-state bytes must equal the ordinary serial path's state bytes. Invalid parent graphs are rejected before decoding.

The serial reference reserves masked KV slots to match the tree's attention reduction layout. Otherwise, small floating-point changes caused by different reduction layouts can grow through a deeply quantized model and obscure a state or masking defect. This alignment does not reuse the tree kernel for the serial calculation.

The model-free test also isolates the selector weight with a regression that fails for an unweighted edge. It checks 64,000 tree schedules against the original ancestor paths and a two-position categorical distribution over 200,000 seeds. These checks validate the tested execution and sampling paths; they do not replace a downstream benchmark or model-quality evaluation.

Measure freshly processed prompt tokens and decode separately, with prompt caching enabled. Keep models, sampling, input requests, output limit, seeds and CUDA settings fixed. Report useful output tokens per verification round and time per round alongside accepted/proposed counts: unused sibling nodes reduce the latter fraction without necessarily reducing useful progress.

Reference implementation inspected: [TensorFold v1.2.5 DFlash2 tree policy](https://github.com/ashhart/TensorFold/blob/8be9c408c2825102b475dc6049c6aa3cf5d9e910/zig/src/families/qwen3_5/cuda_draft_tree.zig).
