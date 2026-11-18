#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Qwen35 forest verification with unified KV. Each sequence starts at its pending token.
// Parents are batch indices and must precede their child in the same sequence.
// Use -1 for roots and -2 for ordinary prompt/chain rows. Tree positions are temporary row IDs.
LLAMA_API bool llama_tree_supported(const struct llama_context * ctx);
LLAMA_API int llama_tree_decode(struct llama_context * ctx, struct llama_batch_ext * batch, const int32_t * parents, int32_t n);
// Keep a root-to-leaf path, including row 0, without recomputing target layers.
LLAMA_API bool llama_tree_commit_seq(struct llama_context * ctx, llama_seq_id seq_id, const int32_t * rows, int32_t n);
// Compatibility shorthand for sequence 0.
LLAMA_API bool llama_tree_commit(struct llama_context * ctx, const int32_t * rows, int32_t n);

#ifdef __cplusplus
}
#endif
