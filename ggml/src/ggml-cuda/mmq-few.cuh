#pragma once

#include "common.cuh"

// ids != nullptr: MUL_MAT_ID
bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, int cc);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

struct ggml_cuda_mmq_few_state;

class ggml_cuda_mmq_few_scope {
public:
    ggml_cuda_mmq_few_scope(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, bool graph_mode);
    ~ggml_cuda_mmq_few_scope();
    ggml_cuda_mmq_few_scope(const ggml_cuda_mmq_few_scope &) = delete;
    ggml_cuda_mmq_few_scope & operator=(const ggml_cuda_mmq_few_scope &) = delete;

private:
    ggml_cuda_mmq_few_state * previous;
};

void ggml_cuda_mmq_few_free(ggml_backend_cuda_context & ctx);

void ggml_cuda_graph_capture_begin();
void ggml_cuda_graph_capture_end();
