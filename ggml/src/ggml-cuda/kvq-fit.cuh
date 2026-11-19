#include "common.cuh"

void ggml_cuda_op_kvq_fit(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_kvq_fit_supported(const ggml_tensor * op);
