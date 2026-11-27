#include "common.cuh"

void ggml_cuda_top_k_f32(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream);
void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
