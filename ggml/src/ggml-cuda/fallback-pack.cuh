// Included inside native_prefill, where scale_index defines the cuBLASLt layout.
static bool fallback_type(ggml_type type) {
    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K;
}

template<ggml_type Type>
static __global__ void pack_fallback_weight(const void * weights, uint8_t * codes, uint8_t * scales, int k) {
#ifdef BLACKWELL_MMA_AVAILABLE
    constexpr int groups = 16;
    constexpr int workers = 256;
    __shared__ float scratch[workers/32][QK_K];
    const int lane = threadIdx.x%32;
    const int warp = threadIdx.x/32;
    const int block = blockIdx.x*(workers/32)+warp;
    float * decoded = scratch[warp];
    if constexpr (Type == GGML_TYPE_IQ2_S) {
        dequantize_iq2_s(weights,block,decoded,lane);
    } else if constexpr (Type == GGML_TYPE_Q3_K) {
        dequantize_q3_K(weights,block,decoded,lane);
        dequantize_q3_K(weights,block,decoded,lane+32);
    } else {
        dequantize_q4_K(weights,block,decoded,lane);
    }
    __syncwarp();
#pragma unroll
    for (int step=0;step<QK_K;step+=32) {
        const size_t i = size_t(block)*QK_K+step+lane;
        const float value = decoded[step+lane]*64.0f;
        float amax = fabsf(value);
#pragma unroll
        for (int d=groups/2;d;d/=2) {
            amax = fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d,groups));
        }
        const __nv_fp8_e4m3 scale(amax>0 ? fmaxf(amax/6,0x1p-9f) : 1.0f);
        const unsigned code = __nv_fp4_e2m1(value*(1.0f/float(scale))).__x;
        const unsigned next = __shfl_xor_sync(0xffffffff,code,1,groups);
        if (lane%2==0) {
            codes[i/2] = code | (next<<4);
        }
        if (lane%groups==0) {
            scales[scale_index(i/k,(i%k)/groups,k/groups)] = scale.__x;
        }
    }
#else
    GGML_UNUSED_VARS(weights,codes,scales,k);
    NO_DEVICE_CODE;
#endif
}

static void launch_fallback_weights(const ggml_tensor * w, uint8_t * codes, uint8_t * scales, int k, int n,
                                    cudaStream_t stream) {
    const dim3 grid(size_t(k)*n/(8*QK_K));
    switch (w->type) {
        case GGML_TYPE_IQ2_S:
            pack_fallback_weight<GGML_TYPE_IQ2_S><<<grid,256,0,stream>>>(w->data,codes,scales,k);
            break;
        case GGML_TYPE_Q3_K:
            pack_fallback_weight<GGML_TYPE_Q3_K><<<grid,256,0,stream>>>(w->data,codes,scales,k);
            break;
        case GGML_TYPE_Q4_K:
            pack_fallback_weight<GGML_TYPE_Q4_K><<<grid,256,0,stream>>>(w->data,codes,scales,k);
            break;
        default: GGML_ABORT("unexpected fallback type");
    }
    CUDA_CHECK(cudaGetLastError());
}
