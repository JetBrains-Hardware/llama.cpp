#pragma once

#if defined(BLACKWELL_MMA_AVAILABLE) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080

#include <cuda_fp4.h>
#include <cuda_fp8.h>

struct compact_nvfp4_group {
    uint32_t lo;
    uint32_t hi;
    uint8_t scale;
};

static __device__ __forceinline__ unsigned compact_nvfp4_gather_nibbles(unsigned bytes) {
    return (bytes & 0x0f) | ((bytes >> 4) & 0x00f0) | ((bytes >> 8) & 0x0f00) | ((bytes >> 12) & 0xf000);
}

static __device__ __forceinline__ unsigned compact_nvfp4_spread_signs(unsigned bits) {
    bits = (bits | (bits << 6)) & 0x0303;
    bits = (bits | (bits << 3)) & 0x1111;
    return bits << 3;
}

static __device__ __forceinline__ compact_nvfp4_group compact_nvfp4_from_iq3(const block_iq3_s & block, int group, float global_scale) {
    const int sub = group / 2;
    const int first = 4 * group;
    const unsigned high = block.qh[sub] >> (4 * (group & 1));
    uint32_t values[4];
    uint32_t maxima = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned index = block.qs[first + j] | (((high >> j) & 1u) << 8);
        values[j] = iq3s_grid[index];
        maxima = __vmaxu4(maxima, values[j]);
    }
    maxima = __vmaxu4(maxima, maxima >> 16);
    const int amax = __vmaxu4(maxima, maxima >> 8) & 0xff;
    const int ls = (block.scales[sub / 2] >> (4 * (sub & 1))) & 15;
    const float d = __half2float(block.d) * (1 + 2 * ls) * global_scale;
    const float peak = fabsf(d) * amax;
    const __nv_fp8_e4m3 scale(peak > 0 ? fmaxf(peak / 6.0f, 0x1p-9f) : 1.0f);
    const float inv = 1.0f / float(scale);
    const uint16_t signs = uint16_t(block.signs[2 * group]) | (uint16_t(block.signs[2 * group + 1]) << 8);
    uint32_t packed[4];
    uint32_t palette[2] = {};
    const float mag = fabsf(d);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 pair_values = make_float2((4 * pair + 1) * mag * inv, (4 * pair + 3) * mag * inv);
        const unsigned bits = __nv_cvt_float2_to_fp4x2(pair_values, __NV_E2M1, cudaRoundNearest);
        const unsigned bytes = (bits & 0x0f) | ((bits & 0xf0) << 4);
        palette[pair / 2] |= bytes << (16 * (pair % 2));
    }
    const unsigned polarity = (__float_as_uint(d) >> 31) ? 0x8888 : 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned selectors = compact_nvfp4_gather_nibbles((values[j] >> 1) & 0x07070707);
        const unsigned nibbles = compact_nvfp4_gather_nibbles(__byte_perm(palette[0], palette[1], selectors));
        packed[j] = nibbles ^ compact_nvfp4_spread_signs((signs >> (4 * j)) & 15) ^ polarity;
    }
    return {packed[0] | (packed[1] << 16), packed[2] | (packed[3] << 16), scale.__x};
}

static __device__ __forceinline__ compact_nvfp4_group compact_nvfp4_from_iq4(const block_iq4_xs & block, int group, float global_scale) {
    const int sub = group / 2;
    const int shift = 4 * (group % 2);
    const int ls = ((block.scales_l[sub / 2] >> (4 * (sub % 2))) & 15) |
        (((block.scales_h >> (2 * sub)) & 3) << 4);
    const float d = __half2float(block.d) * (ls - 32) * global_scale;
    int values[16];
    int peak_code = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        values[j] = kvalues_iq4nl[(block.qs[16 * sub + j] >> shift) & 15];
        peak_code = max(peak_code, abs(values[j]));
    }
    const float peak = fabsf(d) * peak_code;
    const __nv_fp8_e4m3 scale(peak > 0 ? fmaxf(peak / 6.0f, 0x1p-9f) : 1.0f);
    const float inv = 1.0f / float(scale);
    uint32_t packed[2] = {};
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const float2 v = make_float2((values[2 * pair] * d) * inv, (values[2 * pair + 1] * d) * inv);
        const unsigned bits = __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
        packed[pair / 4] |= bits << (8 * (pair % 4));
    }
    return {packed[0], packed[1], scale.__x};
}

template<ggml_type compact_type, int J>
static __device__ __forceinline__ void ggml_cuda_mmq_load_compact_nvfp4(
        const char * __restrict__ x, int * __restrict__ dst, const int kbx0, const int stride) {
    static_assert(compact_type == GGML_TYPE_IQ3_S || compact_type == GGML_TYPE_IQ4_XS, "unsupported compact weight type");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr auto config = ggml_cuda_mmq_get_config(GGML_TYPE_NVFP4, J, false, GGML_PREC_Q4);
    constexpr int blocks_per_row = config.K_vram / QK_NVFP4;
    constexpr int rows_per_warp = warp_size / blocks_per_row;
    constexpr int rows_per_step = rows_per_warp * (config.nthreads / warp_size);
    constexpr int shared_stride = ggml_cuda_mmq_get_sram_stride(config.sram_layout);
    constexpr int groups_per_block = QK_NVFP4 / QK_NVFP4_SUB;
    constexpr float global_scale = 64.0f;
    const int block = threadIdx.x % blocks_per_row;
    const int lane_row = threadIdx.x / blocks_per_row;
    auto * words = reinterpret_cast<uint32_t *>(dst);

#pragma unroll
    for (int i = 0; i < config.I; i += rows_per_step) {
        const int row = i + threadIdx.y * rows_per_warp + lane_row;
        const int index = kbx0 + row*stride + block;
        uint32_t scales = 0;
#pragma unroll
        for (int sub = 0; sub < groups_per_block; ++sub) {
            const int group = (index % (QK_K / QK_NVFP4))*groups_per_block + sub;
            const int source_block = index / (QK_K / QK_NVFP4);
            compact_nvfp4_group q;
            if constexpr (compact_type == GGML_TYPE_IQ3_S) {
                q = compact_nvfp4_from_iq3(reinterpret_cast<const block_iq3_s *>(x)[source_block], group, global_scale);
            } else {
                q = compact_nvfp4_from_iq4(reinterpret_cast<const block_iq4_xs *>(x)[source_block], group, global_scale);
            }
            const int offset = row*shared_stride + 8*block + 2*sub;
            words[offset] = q.lo;
            words[offset + 1] = q.hi;
            scales |= uint32_t(q.scale) << (8*sub);
        }
        words[row*shared_stride + 2*MMQ_TILE_NE_K + block] = scales;
    }
}

#endif
