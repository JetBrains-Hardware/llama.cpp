#include "kvq-fit.cuh"

// device steps of the kvq codec fit, see ggml_kvq_fit
// the code is built with -use_fast_math, so the float divisions use the _rn intrinsics: the results must be those of the CPU codec
// the gain of a row sums in double in the order of the CPU; the column sums over the rows add partial sums in double in a fixed order,
// so they round to the float of the sequential sum of the CPU but in rare cases; the products of two floats are exact in double

#define KVQ_FIT_RB    16 // rows per block of the row kernels
#define KVQ_FIT_PARTS 64 // row ranges of the column sums
#define KVQ_FIT_LANES 8  // rows summed at once per column in a block

// column sums in double over the rows of one of KVQ_FIT_PARTS ranges: s[c] = sum of x[r][c] (or (x[r][c]/g[r])^2 with g)
// block (32 columns, KVQ_FIT_LANES), grid (columns/32, KVQ_FIT_PARTS, B); the partials of a part are added in a fixed order
static __global__ void kvq_fit_col_part(const float * x, const float * g, double * part, const int D, const int n, const int64_t sx1, const int64_t sx2) {
    __shared__ double acc[KVQ_FIT_LANES][32];

    const int c  = blockIdx.x*32 + threadIdx.x;
    const int b  = blockIdx.z;
    const int nr = (n + KVQ_FIT_PARTS - 1)/KVQ_FIT_PARTS;
    const int r0 = blockIdx.y*nr;
    const int r1 = min(n, r0 + nr);

    double s = 0.0;
    if (c < D) {
        const float * xb = x + b*sx2 + c;
        for (int r = r0 + threadIdx.y; r < r1; r += KVQ_FIT_LANES) {
            const float v = g ? __fdiv_rn(xb[r*sx1], g[(int64_t) b*n + r]) : xb[r*sx1];
            s += g ? (double) v*v : (double) v;
        }
    }
    acc[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && c < D) {
        for (int l = 1; l < KVQ_FIT_LANES; l++) {
            s += acc[l][threadIdx.x];
        }
        part[((int64_t) b*KVQ_FIT_PARTS + blockIdx.y)*D + c] = s;
    }
}

// one thread per (column, task): MEAN: sum/n, GAIN: sqrt(sum/n)
static __global__ void kvq_fit_col_final(const double * part, float * dst, const int D, const int n, const bool root) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= D) {
        return;
    }
    double s = 0.0;
    for (int p = 0; p < KVQ_FIT_PARTS; p++) {
        s += part[((int64_t) b*KVQ_FIT_PARTS + p)*D + c];
    }
    dst[(int64_t) b*D + c] = root ? (float) sqrt(s/n) : (float) (s/n);
}

static __device__ float kvq_fit_gain(const float * y, const int D, const float den) {
    if (den == 0.0f) {
        return 1.0f;
    }
    double ss = 0.0;
    for (int c = 0; c < D; c++) {
        ss += (double) y[c]*y[c];
    }
    return __fadd_rn((float) sqrt(ss/den), 1e-8f);
}

// KVQ_FIT_RB rows of one task per block: the gain of each row, in the order of the CPU (as the code kernel)
static __global__ void kvq_fit_gain_rows(const float * y, const float * par, float * g, const int D, const int n,
        const int64_t sy1, const int64_t sy2, const int64_t sp1, const int64_t sp2) {
    extern __shared__ float ys[]; // [KVQ_FIT_RB][D + 1]

    const int b  = blockIdx.y;
    const int r0 = blockIdx.x*KVQ_FIT_RB;
    const int nr = min(KVQ_FIT_RB, n - r0);

    for (int j = threadIdx.x; j < nr*D; j += blockDim.x) {
        ys[(j / D)*(D + 1) + j % D] = y[b*sy2 + (r0 + j / D)*sy1 + j % D];
    }
    __syncthreads();
    if (threadIdx.x < nr) {
        g[(int64_t) b*n + r0 + threadIdx.x] = kvq_fit_gain(ys + threadIdx.x*(D + 1), D, par[b*sp2 + 3*sp1]);
    }
}

// KVQ_FIT_RB rows of one task per block, a thread per component
static __global__ void kvq_fit_code(const float * y, const float * par, int32_t * rows, const int D, const int n, const int nw,
        const int64_t sy1, const int64_t sy2, const int64_t sp1, const int64_t sp2, const int64_t sr1, const int64_t sr2) {
    extern __shared__ float smem[];
    float    * ys = smem;                                         // [KVQ_FIT_RB][D + 1]
    float    * gs = ys + KVQ_FIT_RB*(D + 1);                      // [KVQ_FIT_RB]
    uint32_t * ws = (uint32_t *) (gs + KVQ_FIT_RB);               // [KVQ_FIT_RB][nw]

    const int b    = blockIdx.y;
    const int r0   = blockIdx.x*KVQ_FIT_RB;
    const int nr   = min(KVQ_FIT_RB, n - r0);
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;

    const float * pb   = par + b*sp2;
    const float   den  = pb[3*sp1];

    for (int j = tid; j < nr*D; j += nthr) {
        const int rr = j / D;
        const int c  = j % D;
        ys[rr*(D + 1) + c] = y[b*sy2 + (r0 + rr)*sy1 + c];
    }
    for (int j = tid; j < KVQ_FIT_RB*nw; j += nthr) {
        ws[j] = 0;
    }
    __syncthreads();

    if (tid < nr) {
        gs[tid] = kvq_fit_gain(ys + tid*(D + 1), D, den);
    }
    __syncthreads();

    for (int c = tid; c < D; c += nthr) {
        const int nb = (int) pb[sp1 + c];
        if (nb == 0) {
            continue;
        }
        const float step = pb[c];
        const int   o    = (int) pb[2*sp1 + c];
        const float half = (float) (1 << (nb - 1));
        for (int rr = 0; rr < nr; rr++) {
            const float    v    = den != 0.0f ? __fdiv_rn(ys[rr*(D + 1) + c], gs[rr]) : ys[rr*(D + 1) + c];
            const float    q    = floorf(__fdiv_rn(v, step));
            const int      code = (int) fminf(fmaxf(q, -half), half - 1.0f);
            const uint32_t u    = (uint32_t) (code + (1 << (nb - 1))) & ((1u << nb) - 1);
            atomicOr(ws + rr*nw + o/32, u << (o % 32));
            if (o % 32 + nb > 32) {
                atomicOr(ws + rr*nw + o/32 + 1, u >> (32 - o % 32));
            }
        }
    }
    __syncthreads();

    for (int j = tid; j < nr*nw; j += nthr) {
        const int rr = j / nw;
        const int w  = j % nw;
        uint32_t val = ws[j];
        if (den != 0.0f && w == nw - 1) {
            val = (uint32_t) __half_as_ushort(__float2half_rn(gs[rr]));
        }
        rows[b*sr2 + (r0 + rr)*sr1 + w] = (int32_t) val;
    }
}

void ggml_cuda_op_kvq_fit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * par = dst->src[1];

    const int mode = ggml_get_op_params_i32(dst, 0);

    const int     D   = (int) x->ne[0];
    const int     n   = (int) x->ne[1];
    const int     B   = (int) x->ne[2];
    const int64_t sx1 = x->nb[1]/sizeof(float);
    const int64_t sx2 = x->nb[2]/sizeof(float);
    const int64_t sp1 = par ? par->nb[1]/sizeof(float) : 0;
    const int64_t sp2 = par ? par->nb[2]/sizeof(float) : 0;

    const float * x_d   = (const float *) x->data;
    const float * par_d = par ? (const float *) par->data : nullptr;

    cudaStream_t stream = ctx.stream();

    switch (mode) {
        case GGML_KVQ_FIT_MEAN: {
            ggml_cuda_pool_alloc<double> part(ctx.pool(), (size_t) KVQ_FIT_PARTS*D*B);
            kvq_fit_col_part<<<dim3((D + 31)/32, KVQ_FIT_PARTS, B), dim3(32, KVQ_FIT_LANES), 0, stream>>>(x_d, nullptr, part.get(), D, n, sx1, sx2);
            kvq_fit_col_final<<<dim3((D + 255)/256, B), 256, 0, stream>>>(part.get(), (float *) dst->data, D, n, false);
        } break;
        case GGML_KVQ_FIT_COV:
        case GGML_KVQ_FIT_ROT: {
            // column-major views: x of a task is D x n with ld sx1; COV: x x^T, ROT: F x with F(c, i) = par[i*sp1 + c]
            const float one  = 1.0f;
            const float zero = 0.0f;
            cublasHandle_t h = ctx.cublas_handle();
            CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
            if (mode == GGML_KVQ_FIT_COV) {
                CUBLAS_CHECK(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, D, D, n, &one, x_d, (int) sx1, sx2, x_d, (int) sx1, sx2,
                        &zero, (float *) dst->data, D, (int64_t) D*D, B));
            } else {
                CUBLAS_CHECK(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, D, n, D, &one, par_d, (int) sp1, sp2, x_d, (int) sx1, sx2,
                        &zero, (float *) dst->data, D, (int64_t) D*n, B));
            }
            // back to the mode of common.cuh
            CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
        } break;
        case GGML_KVQ_FIT_GAIN: {
            ggml_cuda_pool_alloc<float>  g(ctx.pool(), (size_t) n*B);
            ggml_cuda_pool_alloc<double> part(ctx.pool(), (size_t) KVQ_FIT_PARTS*D*B);
            const size_t nsm = KVQ_FIT_RB*(D + 1)*sizeof(float);
            kvq_fit_gain_rows<<<dim3((n + KVQ_FIT_RB - 1)/KVQ_FIT_RB, B), 256, nsm, stream>>>(x_d, par_d, g.get(), D, n, sx1, sx2, sp1, sp2);
            kvq_fit_col_part<<<dim3((D + 31)/32, KVQ_FIT_PARTS, B), dim3(32, KVQ_FIT_LANES), 0, stream>>>(x_d, g.get(), part.get(), D, n, sx1, sx2);
            kvq_fit_col_final<<<dim3((D + 255)/256, B), 256, 0, stream>>>(part.get(), (float *) dst->data, D, n, true);
        } break;
        case GGML_KVQ_FIT_CODE: {
            const ggml_tensor * rows = dst->src[2];
            const int     nw  = (int) rows->ne[0];
            const int64_t sr1 = rows->nb[1]/sizeof(int32_t);
            const int64_t sr2 = rows->nb[2]/sizeof(int32_t);
            const size_t  nsm = (KVQ_FIT_RB*(D + 1) + KVQ_FIT_RB)*sizeof(float) + KVQ_FIT_RB*nw*sizeof(uint32_t);
            const dim3 grid((n + KVQ_FIT_RB - 1)/KVQ_FIT_RB, B);
            kvq_fit_code<<<grid, 256, nsm, stream>>>(x_d, par_d, (int32_t *) rows->data, D, n, nw, sx1, sx2, sp1, sp2, sr1, sr2);
        } break;
        default:
            GGML_ABORT("unknown kvq_fit mode %d", mode);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_kvq_fit_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const int mode = ggml_get_op_params_i32(op, 0);
    if (x->ne[0] > 1024 || x->ne[1] > INT32_MAX/x->ne[0]) {
        return false;
    }
    if (mode == GGML_KVQ_FIT_CODE && op->src[2]->ne[0] > 64) {
        return false;
    }
    return true;
}
