// Experimental per-projection work buffers for native FP4 prefill.
namespace native_prefill {

static __device__ size_t scale_index(int row, int group, int groups) {
    return (size_t(row/128)*(groups/4)+group/4)*512 +
        (row%32)*16+(row%128/32)*4+group%4;
}

#ifdef BLACKWELL_MMA_AVAILABLE
template<ggml_type Type>
static __device__ __forceinline__ compact_nvfp4_group convert(const void * w, int row_group) {
    const int i = row_group/16;
    const int g = row_group%16;
    if constexpr (Type == GGML_TYPE_IQ3_S) {
        return compact_nvfp4_from_iq3(static_cast<const block_iq3_s *>(w)[i],g,64.0f);
    } else if constexpr (Type == GGML_TYPE_IQ3_XXS) {
        return compact_nvfp4_from_iq3_xxs(static_cast<const block_iq3_xxs *>(w)[i],g,64.0f);
    } else {
        return compact_nvfp4_from_iq4(static_cast<const block_iq4_xs *>(w)[i],g,64.0f);
    }
}
#endif

template<ggml_type Type>
static __global__ void pack_weight(const void * w, uint2 * values, uint32_t * scales, int groups, int rows) {
#ifdef BLACKWELL_MMA_AVAILABLE
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= rows*groups) {
        return;
    }
    const int g = i%groups;
    const auto q = convert<Type>(w,i);
    values[i] = make_uint2(q.lo,q.hi);
    const uint32_t lane = uint32_t(q.scale) << (8*(g%4));
    const uint32_t pair = lane | __shfl_xor_sync(0xffffffff,lane,1);
    const uint32_t word = pair | __shfl_xor_sync(0xffffffff,pair,2);
    if (g%4 == 0) {
        scales[scale_index(i/groups,g,groups)/4] = word;
    }
#else
    GGML_UNUSED_VARS(w,values,scales,groups,rows);
    NO_DEVICE_CODE;
#endif
}

template<bool Swiglu>
static __global__ void pack_input(const float * x, const float * up, uint8_t * values,
                                  uint8_t * scales, int k, int m) {
#ifdef BLACKWELL_MMA_AVAILABLE
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int row = i/k;
    const int within = threadIdx.x%16;
    float value = row < m ? x[i] : 0.0f;
    if constexpr (Swiglu) {
        if (row < m) {
            value = ggml_cuda_op_silu_single(value)*up[i];
        }
    }
    float amax = fabsf(value);
#pragma unroll
    for (int d = 8; d; d/=2) {
        amax = fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d,16));
    }
    const __nv_fp8_e4m3 scale(amax > 0 ? fmaxf(amax/6,0x1p-9f) : 1.0f);
    const uint32_t code = __nv_fp4_e2m1(value*(1.0f/float(scale))).__x;
    const uint32_t adjacent = __shfl_xor_sync(0xffffffff,code,1,16);
    if (within == 0) {
        scales[scale_index(row,(i%k)/16,k/16)] = scale.__x;
    }
    if (within%2 == 0) {
        values[i/2] = code | (adjacent << 4);
    }
#else
    GGML_UNUSED_VARS(x, up, values, scales, k, m);
    NO_DEVICE_CODE;
#endif
}

#include "fallback-pack.cuh"

struct plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t algorithm{};

    ~plan() {
        if (d) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d));
        if (b) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b));
        if (a) CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a));
        if (op) CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
    }

    bool prepare(cublasLtHandle_t h, int k, int n, int m, void * sa, void * sb, size_t work_bytes, int device) {
        constexpr auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        constexpr auto dtype = CUDA_R_4F_E2M1;
        const cublasOperation_t transpose = CUBLAS_OP_T;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa,sizeof(sa)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb,sizeof(sb)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a,dtype,k,n,k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b,dtype,k,m,k));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d,CUDA_R_32F,n,m,n));

        using key = std::tuple<int, int, int, int>;
        static thread_local std::map<key,std::optional<cublasLtMatmulAlgo_t>> algorithms;
        const key shape{device, k, n, m};
        const auto cached = algorithms.find(shape);
        if (cached != algorithms.end()) {
            if (!cached->second) return false;
            algorithm = *cached->second;
            return true;
        }
        cublasLtMatmulPreference_t prefer;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&prefer));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(prefer,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &work_bytes,sizeof(work_bytes)));
        const uint32_t reductions = CUBLASLT_REDUCTION_SCHEME_NONE;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(prefer,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                         &reductions,sizeof(reductions)));
        cublasLtMatmulHeuristicResult_t found[8]{};
        int count = 0;
        const auto status = cublasLtMatmulAlgoGetHeuristic(h,op,a,b,d,d,prefer,8,found,&count);
        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(prefer));
        if (status == CUBLAS_STATUS_SUCCESS) {
            for (int i = 0; i < count; ++i) {
                if (found[i].state == CUBLAS_STATUS_SUCCESS) {
                    algorithm = found[i].algo;
                    algorithms.emplace(shape, algorithm);
                    return true;
                }
            }
        }
        algorithms.emplace(shape,std::nullopt);
        return false;
    }

    void multiply(cublasLtHandle_t h, const void * w, const void * x, float * dst,
                  void * work, size_t work_bytes, cudaStream_t stream) const {
        constexpr float alpha = 1.0f / 64.0f;
        constexpr float beta = 0;
        CUBLAS_CHECK(cublasLtMatmul(h,op,&alpha,w,a,x,b,&beta,dst,d,dst,d,&algorithm,work,work_bytes,stream));
    }
};

static void launch_weight(const ggml_tensor * w, uint8_t * codes, uint8_t * scales, int k, int n, cudaStream_t stream) {
    auto * words = reinterpret_cast<uint2 *>(codes);
    auto * scale_words = reinterpret_cast<uint32_t *>(scales);
    constexpr int threads = 256;
    const dim3 grid((n*(k/16)+threads-1)/threads);
    switch (w->type) {
        case GGML_TYPE_IQ3_S:
            pack_weight<GGML_TYPE_IQ3_S><<<grid,threads,0,stream>>>(w->data,words,scale_words,k/16,n);
            break;
        case GGML_TYPE_IQ3_XXS:
            pack_weight<GGML_TYPE_IQ3_XXS><<<grid,threads,0,stream>>>(w->data,words,scale_words,k/16,n);
            break;
        case GGML_TYPE_IQ4_XS:
            pack_weight<GGML_TYPE_IQ4_XS><<<grid,threads,0,stream>>>(w->data,words,scale_words,k/16,n);
            break;
        default: GGML_ABORT("unexpected compact type");
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool supported_type(ggml_type type) {
    return fallback_type(type) || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS;
}

static bool run(ggml_backend_cuda_context & ctx, const ggml_tensor * weight, const ggml_tensor * input,
                ggml_tensor * output, const ggml_tensor * up, const ggml_tensor * other, ggml_tensor * other_output) {
    const int k = input->ne[0];
    const int m = input->ne[1];
    const int padded = GGML_PAD(m,128);
    const int n = weight->ne[1];
    const int other_n = other ? static_cast<int>(other->ne[1]) : 0;
    const size_t w_count = size_t(k)*std::max(n,other_n);
    const size_t x_count = size_t(k)*padded;
    const auto values_size = [](size_t count) { return count / 2; };
    const auto total_size = [&](size_t count) { return values_size(count) + count / 16; };
    if (total_size(w_count) > 64ull * 1024 * 1024) {
        return false;
    }
    constexpr size_t work_bytes = 32ull<<20;
    ggml_cuda_pool_alloc<uint8_t> weights(ctx.pool(),total_size(w_count));
    ggml_cuda_pool_alloc<uint8_t> activations(ctx.pool(),total_size(x_count));
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(),work_bytes);
    auto * wc = weights.get();
    auto * ws = wc+values_size(w_count);
    auto * xc = activations.get();
    auto * xs = xc+values_size(x_count);
    const auto handle = ctx.cublaslt_handle();

    static const bool pad_output = ggml_cuda_env_flag("GGML_CUDA_COMPACT_NVFP4_PAD_OUTPUT");
    static const bool trace = ggml_cuda_env_flag("GGML_CUDA_COMPACT_NVFP4_TRACE");
    plan first_aligned, second_aligned, first_tail, second_tail;
    auto prepare = [&](plan & first, plan & second, int rows) {
        return first.prepare(handle,k,n,rows,ws,xs,work_bytes,ctx.device) &&
               (!other || second.prepare(handle,k,other_n,rows,ws,xs,work_bytes,ctx.device));
    };
    const bool direct_tail = !pad_output && padded != m && prepare(first_tail,second_tail,m);
    if (!direct_tail && !prepare(first_aligned,second_aligned,padded)) {
        return false;
    }
    if (direct_tail && trace) {
        fprintf(stderr, "PREFILL_DIRECT_TAIL %s %d\n", weight->name, m);
    }
    const auto & first = direct_tail ? first_tail : first_aligned;
    const auto & second = direct_tail ? second_tail : second_aligned;
    ggml_cuda_pool_alloc<float> padded_output(ctx.pool());
    float * extra = !direct_tail && padded != m ? padded_output.alloc(size_t(std::max(n,other_n))*padded) : nullptr;

    const auto stream = ctx.stream();
    const float * x = static_cast<const float *>(input->data);
    const auto * u = up ? static_cast<const float *>(up->data) : nullptr;
    if (up) {
        pack_input<true><<<x_count/256,256,0,stream>>>(x,u,xc,xs,k,m);
    } else {
        pack_input<false><<<x_count/256,256,0,stream>>>(x,nullptr,xc,xs,k,m);
    }
    CUDA_CHECK(cudaGetLastError());
    const auto project = [&](const plan & p, const ggml_tensor * w, ggml_tensor * dst) {
        const int rows = w->ne[1];
        if (trace) {
            fprintf(stderr, "PREFILL_PRECISION fp4 %s %d\n", w->name, m);
        }
        if (fallback_type(w->type)) {
            launch_fallback_weights(w,wc,ws,k,rows,stream);
        } else {
            launch_weight(w,wc,ws,k,rows,stream);
        }
        auto * out = static_cast<float *>(dst->data);
        p.multiply(handle,wc,xc,extra ? extra : out,workspace.get(),work_bytes,stream);
        if (extra) {
            CUDA_CHECK(cudaMemcpyAsync(out,extra,size_t(rows)*m*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        }
    };
    project(first,weight,output);
    if (other) {
        project(second,other,other_output);
    }
    return true;
}

} // namespace native_prefill
