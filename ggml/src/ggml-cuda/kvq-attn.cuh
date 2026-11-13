#include "common.cuh"

void ggml_cuda_op_kvq_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_kvq_attn_supported(int device, const ggml_tensor * op);
