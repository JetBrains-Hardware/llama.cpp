#include "mhc.cuh"

#define MHC_ROUTE_BLOCK 256

template <int N_ROUTE, bool VEC4>
static __global__ void mhc_route_f32_kernel(
        const char * __restrict__ residual,
        const float * __restrict__ w,
        const float * __restrict__ bias,
        float * __restrict__ dst,
        const int64_t d,
        const size_t nb_r1,
        const size_t nb_r2,
        const size_t nb_dst1,
        const int32_t iterations,
        const float pre_scale) {
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;

    const float * va = (const float *) (residual + t * nb_r2);
    const float * vb = (const float *) (residual + t * nb_r2 + nb_r1);
    float       * po = (float *)       ((char *) dst + t * nb_dst1);

    float acc[N_ROUTE + 1];
#pragma unroll
    for (int k = 0; k <= N_ROUTE; ++k) acc[k] = 0.0f;

    if constexpr (VEC4) {
        const int64_t d4 = d >> 2;
        const float4 * va4 = (const float4 *) va;
        const float4 * vb4 = (const float4 *) vb;
        const float4 * w4  = (const float4 *) w;
        const int64_t w_stride4 = d4 << 1;
        for (int64_t i = tid; i < d4; i += blockDim.x) {
            const float4 xa = va4[i];
            const float4 xb = vb4[i];
            acc[0] += xa.x*xa.x + xa.y*xa.y + xa.z*xa.z + xa.w*xa.w
                    + xb.x*xb.x + xb.y*xb.y + xb.z*xb.z + xb.w*xb.w;
#pragma unroll
            for (int k = 0; k < N_ROUTE; ++k) {
                const float4 wa = w4[k * w_stride4 + i];
                const float4 wb = w4[k * w_stride4 + d4 + i];
                acc[k + 1] += wa.x*xa.x + wa.y*xa.y + wa.z*xa.z + wa.w*xa.w
                            + wb.x*xb.x + wb.y*xb.y + wb.z*xb.z + wb.w*xb.w;
            }
        }
    } else {
        for (int64_t i = tid; i < d; i += blockDim.x) {
            const float xa = va[i];
            const float xb = vb[i];
            acc[0] += xa*xa + xb*xb;
#pragma unroll
            for (int k = 0; k < N_ROUTE; ++k) {
                acc[k + 1] += w[k * 2 * d + i] * xa + w[k * 2 * d + d + i] * xb;
            }
        }
    }

    __shared__ float red[N_ROUTE + 1][MHC_ROUTE_BLOCK / WARP_SIZE];
    __shared__ float pre_gates[2];
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
#pragma unroll
    for (int k = 0; k <= N_ROUTE; ++k) {
        float v = warp_reduce_sum(acc[k]);
        if (lane == 0) red[k][warp] = v;
    }
    __syncthreads();

    if (tid == 0) {
        float tot[N_ROUTE + 1];
        const int nw = blockDim.x / WARP_SIZE;
#pragma unroll
        for (int k = 0; k <= N_ROUTE; ++k) {
            float v = 0.0f;
            for (int j = 0; j < nw; ++j) v += red[k][j];
            tot[k] = v;
        }
        const float rms = rsqrtf(tot[0] / (2.0f * (float) d) + 1e-6f);
        float lg[N_ROUTE];
#pragma unroll
        for (int k = 0; k < N_ROUTE; ++k) {
            lg[k] = tot[k + 1] * rms + bias[k];
        }

        pre_gates[0] = (1.0f / (1.0f + expf(-lg[0]))) * pre_scale;
        pre_gates[1] = (1.0f / (1.0f + expf(-lg[1]))) * pre_scale;
        po[d + 0] = 2.0f / (1.0f + expf(-lg[2]));
        po[d + 1] = 2.0f / (1.0f + expf(-lg[3]));

        if constexpr (N_ROUTE == 4) {
            po[d + 2] = 1.0f; po[d + 3] = 0.0f;
            po[d + 4] = 0.0f; po[d + 5] = 1.0f;
        } else {
            float m[2][2] = { { lg[4], lg[5] }, { lg[6], lg[7] } };
            for (int it = 0; it < iterations; ++it) {
                for (int r = 0; r < 2; ++r) {
                    const float mx  = fmaxf(m[r][0], m[r][1]);
                    const float lse = mx + logf(expf(m[r][0] - mx) + expf(m[r][1] - mx));
                    m[r][0] -= lse; m[r][1] -= lse;
                }
                for (int c = 0; c < 2; ++c) {
                    const float mx  = fmaxf(m[0][c], m[1][c]);
                    const float lse = mx + logf(expf(m[0][c] - mx) + expf(m[1][c] - mx));
                    m[0][c] -= lse; m[1][c] -= lse;
                }
            }
            po[d + 2] = expf(m[0][0]); po[d + 3] = expf(m[0][1]);
            po[d + 4] = expf(m[1][0]); po[d + 5] = expf(m[1][1]);
        }
        po[d + 6] = 0.0f;
        po[d + 7] = 0.0f;
    }
    __syncthreads();

    const float pre0 = pre_gates[0];
    const float pre1 = pre_gates[1];
    if constexpr (VEC4) {
        const int64_t d4 = d >> 2;
        const float4 * va4 = (const float4 *) va;
        const float4 * vb4 = (const float4 *) vb;
        float4       * po4 = (float4 *)       po;
        for (int64_t i = tid; i < d4; i += blockDim.x) {
            const float4 xa = va4[i];
            const float4 xb = vb4[i];
            po4[i] = make_float4(
                pre0 * xa.x + pre1 * xb.x,
                pre0 * xa.y + pre1 * xb.y,
                pre0 * xa.z + pre1 * xb.z,
                pre0 * xa.w + pre1 * xb.w);
        }
    } else {
        for (int64_t i = tid; i < d; i += blockDim.x) {
            po[i] = pre0 * va[i] + pre1 * vb[i];
        }
    }
}

template <bool VEC4>
static __global__ void mhc_merge_f32_kernel(
        const char * __restrict__ residual,
        const char * __restrict__ branch,
        const char * __restrict__ packed,
        char * __restrict__ dst,
        const int64_t d,
        const size_t nb_r1, const size_t nb_r2,
        const size_t nb_b1, const size_t nb_p1,
        const size_t nb_o1, const size_t nb_o2) {
    const int64_t t = blockIdx.y;
    const float * vp = (const float *) (packed + t * nb_p1);
    const float p0 = vp[d + 0], p1 = vp[d + 1];
    const float m0 = vp[d + 2], m1 = vp[d + 3], m2 = vp[d + 4], m3 = vp[d + 5];

    const float * va = (const float *) (residual + t * nb_r2);
    const float * vb = (const float *) (residual + t * nb_r2 + nb_r1);
    const float * vx = (const float *) (branch   + t * nb_b1);
    float       * oa = (float *)       (dst      + t * nb_o2);
    float       * ob = (float *)       (dst      + t * nb_o2 + nb_o1);

    if constexpr (VEC4) {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (d >> 2)) return;
        const float4 xa = ((const float4 *) va)[i];
        const float4 xb = ((const float4 *) vb)[i];
        const float4 xx = ((const float4 *) vx)[i];
        ((float4 *) oa)[i] = make_float4(
            m0*xa.x + m1*xb.x + p0*xx.x,
            m0*xa.y + m1*xb.y + p0*xx.y,
            m0*xa.z + m1*xb.z + p0*xx.z,
            m0*xa.w + m1*xb.w + p0*xx.w);
        ((float4 *) ob)[i] = make_float4(
            m2*xa.x + m3*xb.x + p1*xx.x,
            m2*xa.y + m3*xb.y + p1*xx.y,
            m2*xa.z + m3*xb.z + p1*xx.z,
            m2*xa.w + m3*xb.w + p1*xx.w);
    } else {
        const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= d) return;
        const float a = va[i], b = vb[i], x = vx[i];
        oa[i] = m0*a + m1*b + p0*x;
        ob[i] = m2*a + m3*b + p1*x;
    }
}

void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual = dst->src[0];
    const ggml_tensor * weight   = dst->src[1];
    const ggml_tensor * bias     = dst->src[2];
    const int32_t iterations     = ggml_get_op_params_i32(dst, 0);
    const bool    fixed_identity = ggml_get_op_params_i32(dst, 1) != 0;
    const float   pre_scale      = ggml_get_op_params_f32(dst, 2);

    const int64_t d  = residual->ne[0];
    const int64_t nt = residual->ne[2];
    if (nt == 0) return;

    const bool vec4 = (d % 4 == 0) &&
        (((uintptr_t) residual->data % 16) == 0) && ((residual->nb[1] % 16) == 0) && ((residual->nb[2] % 16) == 0) &&
        (((uintptr_t) weight->data % 16) == 0) &&
        (((uintptr_t) dst->data % 16) == 0) && ((dst->nb[1] % 16) == 0);

    const dim3 grid(nt, 1, 1);
    if (fixed_identity) {
        if (vec4) {
            mhc_route_f32_kernel<4, true><<<grid, MHC_ROUTE_BLOCK, 0, ctx.stream()>>>(
                (const char *) residual->data, (const float *) weight->data, (const float *) bias->data,
                (float *) dst->data, d, residual->nb[1], residual->nb[2], dst->nb[1], iterations, pre_scale);
        } else {
            mhc_route_f32_kernel<4, false><<<grid, MHC_ROUTE_BLOCK, 0, ctx.stream()>>>(
                (const char *) residual->data, (const float *) weight->data, (const float *) bias->data,
                (float *) dst->data, d, residual->nb[1], residual->nb[2], dst->nb[1], iterations, pre_scale);
        }
    } else {
        if (vec4) {
            mhc_route_f32_kernel<8, true><<<grid, MHC_ROUTE_BLOCK, 0, ctx.stream()>>>(
                (const char *) residual->data, (const float *) weight->data, (const float *) bias->data,
                (float *) dst->data, d, residual->nb[1], residual->nb[2], dst->nb[1], iterations, pre_scale);
        } else {
            mhc_route_f32_kernel<8, false><<<grid, MHC_ROUTE_BLOCK, 0, ctx.stream()>>>(
                (const char *) residual->data, (const float *) weight->data, (const float *) bias->data,
                (float *) dst->data, d, residual->nb[1], residual->nb[2], dst->nb[1], iterations, pre_scale);
        }
    }
}

void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual = dst->src[0];
    const ggml_tensor * branch   = dst->src[1];
    const ggml_tensor * packed   = dst->src[2];
    const int64_t d  = residual->ne[0];
    const int64_t nt = residual->ne[2];
    if (nt == 0) return;

    const bool vec4 = (d % 4 == 0) &&
        (((uintptr_t) residual->data % 16) == 0) && ((residual->nb[1] % 16) == 0) && ((residual->nb[2] % 16) == 0) &&
        (((uintptr_t) branch->data % 16) == 0) && ((branch->nb[1] % 16) == 0) &&
        (((uintptr_t) dst->data % 16) == 0) && ((dst->nb[1] % 16) == 0) && ((dst->nb[2] % 16) == 0);

    if (vec4) {
        const dim3 grid(((d >> 2) + 255) / 256, nt, 1);
        mhc_merge_f32_kernel<true><<<grid, 256, 0, ctx.stream()>>>(
            (const char *) residual->data, (const char *) branch->data, (const char *) packed->data,
            (char *) dst->data, d,
            residual->nb[1], residual->nb[2], branch->nb[1], packed->nb[1], dst->nb[1], dst->nb[2]);
    } else {
        const dim3 grid((d + 255) / 256, nt, 1);
        mhc_merge_f32_kernel<false><<<grid, 256, 0, ctx.stream()>>>(
            (const char *) residual->data, (const char *) branch->data, (const char *) packed->data,
            (char *) dst->data, d,
            residual->nb[1], residual->nb[2], branch->nb[1], packed->nb[1], dst->nb[1], dst->nb[2]);
    }
}
