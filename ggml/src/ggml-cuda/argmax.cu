#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static bool fast_argmax_disabled() {
    static const bool disabled = []() {
        const char * env = std::getenv("GGML_CUDA_DISABLE_FAST_ARGMAX");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return disabled;
}

static __global__ void argmax_f32_stage1(
        const float * __restrict__ x,
        float * __restrict__ part_val,
        int32_t * __restrict__ part_idx,
        const int64_t ncols) {
    float maxval = -FLT_MAX;
    int32_t argmax = -1;

    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t col = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; col < ncols; col += stride) {
        const float val = x[col];
        if (val > maxval) {
            maxval = val;
            argmax = (int32_t) col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float   val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
        const int32_t col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col < argmax)) {
            maxval = val;
            argmax = col;
        }
    }

    constexpr int max_warps = 256 / WARP_SIZE;
    __shared__ float   s_val[max_warps];
    __shared__ int32_t s_idx[max_warps];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    if (lane_id == 0) {
        s_val[warp_id] = maxval;
        s_idx[warp_id] = argmax;
    }
    __syncthreads();

    if (warp_id == 0) {
        if (lane_id < n_warps) {
            maxval = s_val[lane_id];
            argmax = s_idx[lane_id];
        } else {
            maxval = -FLT_MAX;
            argmax = -1;
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float   val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
            const int32_t col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
            if (val > maxval || (val == maxval && col < argmax)) {
                maxval = val;
                argmax = col;
            }
        }
        if (lane_id == 0) {
            part_val[blockIdx.x] = maxval;
            part_idx[blockIdx.x] = argmax;
        }
    }
}

static __global__ void argmax_f32_stage2(
        const float * __restrict__ part_val,
        const int32_t * __restrict__ part_idx,
        int32_t * __restrict__ dst,
        const int nparts) {
    float   maxval = -FLT_MAX;
    int32_t argmax = -1;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) {
        const float   val = part_val[i];
        const int32_t col = part_idx[i];
        if (val > maxval || (val == maxval && col < argmax)) {
            maxval = val;
            argmax = col;
        }
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const float   val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
        const int32_t col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
        if (val > maxval || (val == maxval && col < argmax)) {
            maxval = val;
            argmax = col;
        }
    }
    constexpr int max_warps = 128 / WARP_SIZE;
    __shared__ float   s_val[max_warps];
    __shared__ int32_t s_idx[max_warps];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    if (lane_id == 0) {
        s_val[warp_id] = maxval;
        s_idx[warp_id] = argmax;
    }
    __syncthreads();
    if (warp_id == 0) {
        if (lane_id < n_warps) {
            maxval = s_val[lane_id];
            argmax = s_idx[lane_id];
        } else {
            maxval = -FLT_MAX;
            argmax = -1;
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float   val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
            const int32_t col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
            if (val > maxval || (val == maxval && col < argmax)) {
                maxval = val;
                argmax = col;
            }
        }
        if (lane_id == 0) {
            dst[0] = argmax;
        }
    }
}

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
        const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = ggml_cuda_shfl_xor(maxval, offset, WARP_SIZE);
                const int   col = ggml_cuda_shfl_xor(argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (nrows == 1 && ne00 >= 8192 && !fast_argmax_disabled()) {
        constexpr int nparts = 128;
        ggml_cuda_pool_alloc<float>   part_val(ctx.pool(), nparts);
        ggml_cuda_pool_alloc<int32_t> part_idx(ctx.pool(), nparts);
        argmax_f32_stage1<<<nparts, 256, 0, stream>>>(src0_d, part_val.get(), part_idx.get(), ne00);
        argmax_f32_stage2<<<1, 128, 0, stream>>>(part_val.get(), part_idx.get(), dst_d, nparts);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
