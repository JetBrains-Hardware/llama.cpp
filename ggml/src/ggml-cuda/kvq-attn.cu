#include "kvq-attn.cuh"
#include "cp-async.cuh"

#include <cstdlib>

// two-tier attention, see ggml_kvq_attn
//
// kernels, per call:
//   prep:     per (chunk, kv head): qt = (inv q)*step, q . mu, max |qt| per query, unpack fields
//   coarse:   per (block of rows of one chunk, kv head): logits in the code domain with mma (codes x qt, f16 with a
//             power-of-2 scale per query), online softmax, V accumulated in the code domain with mma (codes^T x P),
//             partial (m, z, O^T)
//   precise:  the same for the resident slots and the tail rows with f16/q8_0 K/V, O in the dim domain
//   merge:    per (chunk, kv head): merge the coarse partials of the chunk, y = O*step
//   combine:  per (chunk, kv head): out = y inv + z mu, once per chunk
//   final:    per (query, kv head): merge chunks and precise partials, write the output and the score normalizers
//   score:    per coarse row: sum over the query heads of the normalized coarse weights
//
// a column n of a block is one query: head g*n_rep + n % n_rep at position n / n_rep, N = columns padded to 8
// logits are kept in log2 units (scaled by log2(e)) and exponentiated with exp2

#define KVQ_D        256
#define KVQ_T        64          // rows per tile
#define KVQ_WARPS    4
#define KVQ_NTHREADS (KVQ_WARPS*WARP_SIZE)
#define KVQ_LDT      (KVQ_D + 8) // halves per row of a smem tile
#define KVQ_LDP      (KVQ_T + 8) // halves per column of the P tile
#define KVQ_LDR      (KVQ_T + 8) // words per word of the word-major staging of packed rows
#define KVQ_NBUF     2           // staging buffers of the coarse kernel
#define KVQ_PROWS    64          // precise rows per block
#define KVQ_Q8_ROW   (KVQ_D/QK8_0*sizeof(block_q8_0))
#define KVQ_LOG2E    1.4426950408889634f

struct kvq_args {
    int G, n_rep, L, cols, N, n_head;
    int n_coarse, n_crow, n_chunk, n_cbpc, rows_blk, n_tiles_blk, n_cblk;
    int n_tail, cap, n_rblk, n_tblk, n_part;
    int nwk, nwv, nrk, nrv; // words per packed K/V row, word rows of the K/V staging
    float scale2;           // scale*log2(e)

    const float * q; int64_t q_nb1, q_nb2;          // in elements
    const char  * kc; int64_t kc_nb1, kc_nb2;
    const char  * vc; int64_t vc_nb1, vc_nb2;
    const half  * tinv; int64_t tinv_nb2, tinv_nb3; // in elements
    const float * taux; int64_t taux_nb2, taux_nb3; // in elements
    const int32_t * ridx; int64_t ridx_nb1;         // in elements
    const half  * mask; int64_t mask_nb1;           // in elements

    float    * qtf;   // [n_chunk*G][N][D] qt in f32
    uint32_t * kmax;  // [n_chunk*G][N] max |qt| as f32 bits
    float    * kmu;   // [n_chunk*G][N] q . mu
    uint4    * kpair; // [n_chunk*G][D/2] per component pair: word*LDR, shift | bits of the first << 8, masks, offsets (half2)
    uint2    * vfld;  // [n_chunk*G][D] per component: word*LDR | shift << 16 | mask << 24, offset (half2)
    int      * nst;   // [n_chunk*G][2]: stored components of K, V
    uint32_t * resmap; int nres_words;

    float * po; float * pm; float * pz; // partials [G][n_part][N][D], [G][n_part][N]
    float * pmc; float * pzc;           // coarse score normalizers [G][n_cblk][N]
    float * cy;                         // merged coarse partials y = O*step [G][n_chunk][N][D]
    float * co; float * cm; float * cz; // chunks [G][n_chunk][N][D], [G][n_chunk][N]
    half  * scp; float * scm;           // score weights [G][n_coarse][cols], tile max [G][n_cblk*n_tiles_blk][cols]
    float * scs;                        // score normalizers [G][2][N]

    float * dst_attn;
    float * dst_score;
};

// precise row source: resident slots (row index from ridx) or tail rows
struct kvq_src {
    const char * k; int64_t k_nb1, k_nb2;
    const char * v; int64_t v_nb1, v_nb2;
    int  n_rows;
    bool res;
};

// f16 scale of the qt of one column: max |qt| to [2^14, 2^15)
static __device__ __forceinline__ float kvq_qscale(const uint32_t mx_bits) {
    const float mx = __uint_as_float(mx_bits);
    int e = 0;
    frexpf(mx, &e);
    return mx > 0.0f ? ldexpf(1.0f, 15 - e) : 1.0f;
}

static __device__ __forceinline__ float kvq_exp2(const float x) {
    return x == -INFINITY ? 0.0f : exp2f(x);
}

#ifdef AMPERE_MMA_AVAILABLE

static __device__ __forceinline__ void kvq_mma(float * d, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void kvq_ldsm_x4(uint32_t * r, const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ void kvq_ldsm_x4_trans(uint32_t * r, const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ void kvq_cp4(uint32_t * dst, const void * src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"((uint32_t) __cvta_generic_to_shared(dst)), "l"(src));
}

// a field u is extracted one bit early (the staging has a leading word) into bits 1.. of a 16-bit lane,
// 0x6001 | u << 1 is the f16 value 512 + u + 0.5, minus sub = 512 + 2^(bits-1) (512.5 for 0 bits) gives code + 0.5 exactly
static __device__ __forceinline__ uint32_t kvq_to_half2(const uint32_t y, const uint32_t sub) {
    const half2 h = __hsub2(*(const half2 *) &y, *(const half2 *) &sub);
    return *(const uint32_t *) &h;
}

// code + 0.5 of a component pair from the 32 bits at its offset
static __device__ __forceinline__ uint32_t kvq_pair(const uint32_t lo, const uint32_t hi, const uint4 m) {
    const uint32_t x = __funnelshift_r(lo, hi, m.y);
    const uint32_t y = (__byte_perm(x, x >> (m.y >> 8), 0x5410) & m.z) | 0x60016001u;
    return kvq_to_half2(y, m.w);
}

// code + 0.5 of one component in two rows (lo/hi words of both rows), sh = shift, mk = masks
static __device__ __forceinline__ uint32_t kvq_rows(const uint2 lo, const uint2 hi, const uint32_t sh, const uint32_t mk, const uint32_t sub) {
    const uint32_t ua = __funnelshift_r(lo.x, hi.x, sh);
    const uint32_t ub = __funnelshift_r(lo.y, hi.y, sh);
    return kvq_to_half2((__byte_perm(ua, ub, 0x5410) & mk) | 0x60016001u, sub);
}

// max over the rows of a tile per column of this lane, written to s_red [WARPS][N] by lanes 0..3
template <int NT>
static __device__ __forceinline__ void kvq_tile_max(const float (&x)[NT][4], float * s_red, const int w, const int lane) {
    constexpr int N = NT*8;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float mx = fmaxf(x[j][c], x[j][c + 2]);
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 4));
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 8));
            mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 16));
            if (lane < 4) {
                s_red[w*N + 8*j + 2*lane + c] = mx;
            }
        }
    }
}

// online softmax update after kvq_tile_max and a barrier; x is overwritten with p, returns whether any alpha != 1 in the warp
template <int NT>
static __device__ __forceinline__ bool kvq_tile_update(float (&x)[NT][4], float (&m)[NT][2], float (&z)[NT][2], float (&alpha)[NT][2],
        const float * s_red, const int lane) {
    constexpr int N = NT*8;
    const int t4 = lane % 4;
    bool rescale = false;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float mx = m[j][c];
#pragma unroll
            for (int ww = 0; ww < KVQ_WARPS; ++ww) {
                mx = fmaxf(mx, s_red[ww*N + 8*j + 2*t4 + c]);
            }
            alpha[j][c] = mx == m[j][c] ? 1.0f : kvq_exp2(m[j][c] - mx);
            rescale = rescale || mx != m[j][c];
            m[j][c] = mx;
            const float pa = mx == -INFINITY ? 0.0f : kvq_exp2(x[j][c]     - mx);
            const float pb = mx == -INFINITY ? 0.0f : kvq_exp2(x[j][c + 2] - mx);
            x[j][c]     = pa;
            x[j][c + 2] = pb;
            z[j][c] = z[j][c]*alpha[j][c] + pa + pb;
        }
    }
    return __any_sync(0xFFFFFFFF, rescale);
}

// sum of z over the rows of the block, written by the lanes of warp 0 that own the columns
template <int NT>
static __device__ __forceinline__ void kvq_write_mz(const float (&m)[NT][2], const float (&z)[NT][2], float * s_red, float * pm, float * pz,
        const int w, const int lane) {
    constexpr int N = NT*8;
    const int t4 = lane % 4;

    __syncthreads();
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            float s = z[j][c];
            s += __shfl_xor_sync(0xFFFFFFFF, s, 4);
            s += __shfl_xor_sync(0xFFFFFFFF, s, 8);
            s += __shfl_xor_sync(0xFFFFFFFF, s, 16);
            if (lane < 4) {
                s_red[w*N + 8*j + 2*t4 + c] = s;
            }
        }
    }
    __syncthreads();
    if (w == 0 && lane < 4) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int n = 8*j + 2*t4 + c;
                float s = 0.0f;
#pragma unroll
                for (int ww = 0; ww < KVQ_WARPS; ++ww) {
                    s += s_red[ww*N + n];
                }
                pm[n] = m[j][c];
                pz[n] = s;
            }
        }
    }
}

#endif // AMPERE_MMA_AVAILABLE

// per (chunk, kv head) and slice of 32 components: qt = (inv q)*step in f32, max |qt| per column;
// slice 0 also q . mu, slice 1 the unpack fields; grid (n_chunk*G, D/32), D threads, warp w sums dims [32w, 32w + 32)
template <int N>
static __global__ void __launch_bounds__(KVQ_D) kvq_prep(const kvq_args a, const bool attn) {
    const int tab   = blockIdx.x;
    const int t     = tab / a.G;
    const int g     = tab % a.G;
    const int slice = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int w     = tid / WARP_SIZE;

    extern __shared__ __align__(16) char kvq_smem[];
    float * s_q   = (float *) kvq_smem;   // [N][D]
    float * s_acc = s_q + N*KVQ_D;        // [N][32]

    const half  * invk = a.tinv + t*a.tinv_nb3 + (2*g)*a.tinv_nb2;
    const float * auxk = a.taux + t*a.taux_nb3 + (2*g)*a.taux_nb2;
    const float * auxv = auxk + a.taux_nb2;

    for (int e = tid; e < N*KVQ_D; e += KVQ_D) {
        const int n = e / KVQ_D;
        const int i = e % KVQ_D;
        float qv = 0.0f;
        if (n < a.cols) {
            const int h = g*a.n_rep + n % a.n_rep;
            qv = a.q[i + h*a.q_nb1 + (n / a.n_rep)*a.q_nb2];
        }
        s_q[e] = qv;
    }
    for (int e = tid; e < N*WARP_SIZE; e += KVQ_D) {
        s_acc[e] = 0.0f;
    }
    __syncthreads();

    {
        const int c = WARP_SIZE*slice + lane;
        float acc[N];
#pragma unroll
        for (int n = 0; n < N; ++n) {
            acc[n] = 0.0f;
        }
        float x[WARP_SIZE];
#pragma unroll
        for (int ii = 0; ii < WARP_SIZE; ++ii) {
            x[ii] = __half2float(invk[c + KVQ_D*(WARP_SIZE*w + ii)]);
        }
#pragma unroll
        for (int ii = 0; ii < WARP_SIZE; ++ii) {
#pragma unroll
            for (int n = 0; n < N; ++n) {
                acc[n] += x[ii]*s_q[n*KVQ_D + WARP_SIZE*w + ii];
            }
        }
#pragma unroll
        for (int n = 0; n < N; ++n) {
            atomicAdd(&s_acc[n*WARP_SIZE + lane], acc[n]);
        }
    }
    __syncthreads();

    {
        const int   c    = WARP_SIZE*slice + lane;
        const float step = auxk[2*KVQ_D + c] > 0.0f ? auxk[c] : 0.0f;
        for (int n = w; n < N; n += KVQ_D/WARP_SIZE) {
            const float qt = s_acc[n*WARP_SIZE + lane]*step;
            a.qtf[((int64_t) tab*N + n)*KVQ_D + c] = qt;
            const float mx = warp_reduce_max(fabsf(qt));
            if (lane == 0) {
                atomicMax(&a.kmax[tab*N + n], __float_as_uint(mx));
            }
        }
    }

    if (slice == 0) {
        for (int n = w; n < N; n += KVQ_D/WARP_SIZE) {
            float sm = 0.0f;
            for (int i = lane; i < KVQ_D; i += WARP_SIZE) {
                sm += s_q[n*KVQ_D + i]*auxk[KVQ_D + i];
            }
            sm = warp_reduce_sum(sm);
            if (lane == 0) {
                a.kmu[tab*N + n] = sm;
            }
        }
    }

    if (slice == 1) {
        int * s_bits = (int *) s_q;          // [2][D], s_q is no longer used
        int * s_off  = s_bits + 2*KVQ_D;     // [2][D]
        int * s_nst  = s_off + 2*KVQ_D;      // [2]
        __syncthreads();
        s_bits[tid]         = (int) auxk[2*KVQ_D + tid];
        s_bits[KVQ_D + tid] = attn ? (int) auxv[2*KVQ_D + tid] : 0;
        if (tid < 2) {
            s_nst[tid] = 0;
        }
        __syncthreads();
        if (tid < 2) {
            int o = 0;
            for (int i = 0; i < KVQ_D; ++i) {
                s_off[tid*KVQ_D + i] = o;
                o += s_bits[tid*KVQ_D + i];
            }
        }
        __syncthreads();
        for (int kv = 0; kv < 2; ++kv) {
            if (s_bits[kv*KVQ_D + tid] > 0) {
                atomicMax(&s_nst[kv], tid + 1);
            }
        }

        auto sub = [](const int b) { return b > 0 ? 512.0f + (float) (1 << (b - 1)) : 512.5f; };

        // offsets one bit early in the staging with its leading word: o + 31 = 32*(word) + shift
        if (tid < KVQ_D/2) {
            const int b0 = s_bits[2*tid];
            const int b1 = s_bits[2*tid + 1];
            const int o  = s_off[2*tid] + 31;
            const half2 sb = __floats2half2_rn(sub(b0), sub(b1));
            a.kpair[tab*(KVQ_D/2) + tid] = make_uint4((o / 32)*KVQ_LDR, (o % 32) | (b0 << 8),
                (((1u << b0) - 1) << 1) | (((1u << b1) - 1) << 17), *(const uint32_t *) &sb);
        }
        if (attn) {
            const int b = s_bits[KVQ_D + tid];
            const int o = s_off[KVQ_D + tid] + 31;
            const half2 sb = __floats2half2_rn(sub(b), sub(b));
            a.vfld[tab*KVQ_D + tid] = make_uint2((o / 32)*KVQ_LDR | ((o % 32) << 16) | (((1u << b) - 1) << 24), *(const uint32_t *) &sb);
        }
        __syncthreads();
        if (tid < 2) {
            a.nst[tab*2 + tid] = s_nst[tid];
        }
    }
}

// coarse rows; grid (n_cbpc, n_chunk, G)
// staging of a tile: packed rows word-major, word i of row r at [i*LDR + r], so that rows r and r + 8 share the address computation
template <int N, bool ATTN, bool SCORE>
static __global__ void __launch_bounds__(KVQ_NTHREADS) kvq_coarse(const kvq_args a) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int NT = N/8;

    const int bx   = blockIdx.x;
    const int t    = blockIdx.y;
    const int g    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int w    = tid / WARP_SIZE;
    const int gid  = lane / 4;
    const int t4   = lane % 4;
    const int tab  = t*a.G + g;
    const int blk  = t*a.n_cbpc + bx;

    const int64_t c1 = min((int64_t) a.n_coarse, (int64_t) (t + 1)*a.n_crow);
    const int64_t r0 = (int64_t) t*a.n_crow + (int64_t) bx*a.rows_blk;
    const int64_t r1 = min(c1, r0 + a.rows_blk);

    const int ld_st = (a.nrk + (ATTN ? a.nrv : 0))*KVQ_LDR; // words per staging buffer

    extern __shared__ __align__(16) char kvq_smem[];
    half     * s_qt   = (half *) kvq_smem;                      // [N][LDT]
    half     * s_p    = s_qt + N*KVQ_LDT;                       // [N][LDP]
    uint4    * s_kp   = (uint4 *) (s_p + N*KVQ_LDP);            // [D/2]
    float    * s_red  = (float *) (s_kp + KVQ_D/2);             // [2][WARPS][N]
    uint32_t * s_st   = (uint32_t *) (s_red + 2*KVQ_WARPS*N);   // [NBUF][ld_st]
    half     * s_mask = (half *) (s_st + KVQ_NBUF*ld_st);       // [NBUF][L][T]

    const int nwv1 = a.nwv + 1;
    auto stage = [&](const int64_t tr, const int buf) {
        uint32_t * sk = s_st + buf*ld_st;
        const int nr = (int) min((int64_t) KVQ_T, r1 - tr);
        {
            const int dr = KVQ_NTHREADS / a.nwk;
            const int di = KVQ_NTHREADS % a.nwk;
            int r = tid / a.nwk;
            int i = tid % a.nwk;
            const char * src = a.kc + tr*a.kc_nb1 + g*a.kc_nb2;
            while (r < nr) {
                kvq_cp4(sk + (i + 1)*KVQ_LDR + r, src + r*a.kc_nb1 + 4*i);
                i += di;
                r += dr;
                if (i >= a.nwk) {
                    i -= a.nwk;
                    r++;
                }
            }
        }
        if (ATTN) {
            uint32_t * sv = sk + a.nrk*KVQ_LDR;
            const int dr = KVQ_NTHREADS / nwv1;
            const int di = KVQ_NTHREADS % nwv1;
            int r = tid / nwv1;
            int i = tid % nwv1;
            const char * src = a.vc + tr*a.vc_nb1 + g*a.vc_nb2;
            while (r < nr) {
                kvq_cp4(sv + (i + 1)*KVQ_LDR + r, src + r*a.vc_nb1 + 4*i);
                i += di;
                r += dr;
                if (i >= nwv1) {
                    i -= nwv1;
                    r++;
                }
            }
        }
    };

    // mask values of a tile, [L][T] with KVQ_T*L/NTHREADS <= 4 per thread, loaded one tile ahead
    unsigned short mpre[4];
    auto load_mask = [&](const int64_t tr) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int e = tid + k*KVQ_NTHREADS;
            const int l = e / KVQ_T;
            const int r = e % KVQ_T;
            mpre[k] = 0;
            if (a.mask && l < a.L && tr + r < r1) {
                mpre[k] = __half_as_ushort(a.mask[tr + r + l*a.mask_nb1]);
            }
        }
    };
    auto store_mask = [&](const int buf) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int e = tid + k*KVQ_NTHREADS;
            if (e < a.L*KVQ_T) {
                s_mask[buf*a.L*KVQ_T + e] = __ushort_as_half(mpre[k]);
            }
        }
    };

    // resident bits of the rows of this lane, loaded one tile ahead
    uint32_t rpre[2] = { 0, 0 };
    auto load_res = [&](const int64_t tr) {
        const int64_t ra = tr + 16*w + gid;
        rpre[0] = 0;
        rpre[1] = 0;
        if (ATTN && a.cap > 0) {
            if (ra < r1) {
                rpre[0] = (a.resmap[g*a.nres_words + ra/32] >> (ra % 32)) & 1;
            }
            if (ra + 8 < r1) {
                rpre[1] = (a.resmap[g*a.nres_words + (ra + 8)/32] >> ((ra + 8) % 32)) & 1;
            }
        }
    };

    if (r0 < r1) {
        stage(r0, 0);
    }
    load_mask(r0);
    store_mask(0);
    load_res(r0);

    for (int e = tid; e < N*(KVQ_D/2); e += KVQ_NTHREADS) {
        const int n = e / (KVQ_D/2);
        const int i = e % (KVQ_D/2);
        const float  qs = kvq_qscale(a.kmax[tab*N + n]);
        const float2 qt = ((const float2 *) a.qtf)[((int64_t) tab*N + n)*(KVQ_D/2) + i];
        *(half2 *) (s_qt + n*KVQ_LDT + 2*i) = __floats2half2_rn(qt.x*qs, qt.y*qs);
    }
    if (tid < KVQ_D/2) {
        s_kp[tid] = a.kpair[tab*(KVQ_D/2) + tid];
    }
    uint2 vma[4];
    uint2 vmb[4];
    if (ATTN) {
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            vma[mi] = a.vfld[tab*KVQ_D + 16*(w + 4*mi) + gid];
            vmb[mi] = a.vfld[tab*KVQ_D + 16*(w + 4*mi) + gid + 8];
        }
    }

    // per column of this lane: query position (-1: padding), logit = (acc*kinv + kmu)*scale2 + mask*log2(e)
    int   ln[NT][2];
    float kinv[NT][2];
    float kmu[NT][2];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int n = 8*j + 2*t4 + c;
            ln[j][c]   = n < a.cols ? n / a.n_rep : -1;
            kinv[j][c] = a.scale2/kvq_qscale(a.kmax[tab*N + n]);
            kmu[j][c]  = a.scale2*a.kmu[tab*N + n];
        }
    }

    const int nst_k = a.nst[tab*2 + 0];
    const int nst_v = a.nst[tab*2 + 1];

    float m_a[NT][2], z_a[NT][2], m_c[NT][2], z_c[NT][2], alpha[NT][2];
    float o[4][NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            m_a[j][c] = -INFINITY;
            m_c[j][c] = -INFINITY;
            z_a[j][c] = 0.0f;
            z_c[j][c] = 0.0f;
        }
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o[mi][j][e] = 0.0f;
            }
        }
    }

    int it = 0;
    for (int64_t tr = r0; tr < r1; tr += KVQ_T, ++it) {
        const int buf = it % KVQ_NBUF;
        const int nxt = (it + 1) % KVQ_NBUF;
        const uint32_t resa = rpre[0];
        const uint32_t resb = rpre[1];
        cp_async_wait_all();
        __syncthreads();
        // the other buffer was last read in the previous iteration
        if (tr + KVQ_T < r1) {
            stage(tr + KVQ_T, nxt);
        }
        load_mask(tr + KVQ_T);
        load_res(tr + KVQ_T);

        const uint32_t * sk = s_st + buf*ld_st;
        const uint32_t * sv = sk + a.nrk*KVQ_LDR;
        const half     * sm = s_mask + buf*a.L*KVQ_T;

        // logits of 16 rows per warp: codes x qt
        float s[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                s[j][e] = 0.0f;
            }
        }
        {
            const uint32_t * rk = sk + 16*w + gid;
#pragma unroll
            for (int kk2 = 0; kk2 < KVQ_D/32; ++kk2) {
                if (32*kk2 >= nst_k) {
                    break;
                }
                uint32_t bq[NT][4];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    kvq_ldsm_x4(bq[j], s_qt + (8*j + lane % 8)*KVQ_LDT + 32*kk2 + 8*(lane / 8));
                }
#pragma unroll
                for (int hk = 0; hk < 2; ++hk) {
                    const int kk = 2*kk2 + hk;
                    const uint4 ma = s_kp[8*kk + t4];
                    const uint4 mb = s_kp[8*kk + t4 + 4];
                    const uint32_t * pa = rk + ma.x;
                    const uint32_t * pb = rk + mb.x;
                    const uint32_t af[4] = {
                        kvq_pair(pa[0], pa[KVQ_LDR],     ma), kvq_pair(pa[8], pa[KVQ_LDR + 8], ma),
                        kvq_pair(pb[0], pb[KVQ_LDR],     mb), kvq_pair(pb[8], pb[KVQ_LDR + 8], mb),
                    };
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        kvq_mma(s[j], af, bq[j][2*hk], bq[j][2*hk + 1]);
                    }
                }
            }
        }

        const int     la = 16*w + gid;
        const int64_t ra = tr + la;
        const bool    va = ra < r1;
        const bool    vb = ra + 8 < r1;

        float sc[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int l = ln[j][e & 1];
                float x = -INFINITY;
                if ((e < 2 ? va : vb) && l >= 0) {
                    x = s[j][e]*kinv[j][e & 1] + kmu[j][e & 1] + KVQ_LOG2E*__half2float(sm[l*KVQ_T + la + (e < 2 ? 0 : 8)]);
                }
                sc[j][e] = x;
                s[j][e]  = (e < 2 ? resa : resb) ? -INFINITY : x;
            }
        }

        if (SCORE) {
            kvq_tile_max<NT>(sc, s_red + KVQ_WARPS*N, w, lane);
        }
        if (ATTN) {
            kvq_tile_max<NT>(s, s_red, w, lane);
        }
        __syncthreads();

        if (SCORE) {
            kvq_tile_update<NT>(sc, m_c, z_c, alpha, s_red + KVQ_WARPS*N, lane);
            const int tile = blk*a.n_tiles_blk + it;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n = 8*j + 2*t4 + (e & 1);
                    if ((e < 2 ? va : vb) && ln[j][e & 1] >= 0) {
                        a.scp[((int64_t) g*a.n_coarse + ra + (e < 2 ? 0 : 8))*a.cols + n] = __float2half_rn(sc[j][e]);
                    }
                    if (e < 2 && w == 0 && gid == 0 && ln[j][e] >= 0) {
                        a.scm[((int64_t) g*a.n_cblk*a.n_tiles_blk + tile)*a.cols + n] = m_c[j][e];
                    }
                }
            }
        }

        if (ATTN) {
            const bool rescale = kvq_tile_update<NT>(s, m_a, z_a, alpha, s_red, lane);

            const float gaina = va ? __half2float(__ushort_as_half((unsigned short) (sv[(a.nwv + 1)*KVQ_LDR + la] & 0xFFFF))) : 0.0f;
            const float gainb = vb ? __half2float(__ushort_as_half((unsigned short) (sv[(a.nwv + 1)*KVQ_LDR + la + 8] & 0xFFFF))) : 0.0f;

#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n = 8*j + 2*t4 + (e & 1);
                    s_p[n*KVQ_LDP + la + (e < 2 ? 0 : 8)] = __float2half_rn(s[j][e]*(e < 2 ? gaina : gainb));
                }
            }
            if (rescale) {
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            o[mi][j][e] *= alpha[j][e & 1];
                        }
                    }
                }
            }
            __syncthreads();

            // V in the code domain: codes^T x P^T
            uint32_t bp[NT][4][2];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int ks2 = 0; ks2 < 2; ++ks2) {
                    uint32_t r4[4];
                    kvq_ldsm_x4(r4, s_p + (8*j + lane % 8)*KVQ_LDP + 32*ks2 + 8*(lane / 8));
                    bp[j][2*ks2 + 0][0] = r4[0];
                    bp[j][2*ks2 + 0][1] = r4[1];
                    bp[j][2*ks2 + 1][0] = r4[2];
                    bp[j][2*ks2 + 1][1] = r4[3];
                }
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int mt = w + 4*mi;
                if (16*mt >= nst_v) {
                    continue;
                }
                const uint32_t * pa = sv + (vma[mi].x & 0xFFFF) + 2*t4;
                const uint32_t * pb = sv + (vmb[mi].x & 0xFFFF) + 2*t4;
                const uint32_t sha = vma[mi].x >> 16;
                const uint32_t shb = vmb[mi].x >> 16;
                const uint32_t mka = (vma[mi].x >> 24)*0x20002u;
                const uint32_t mkb = (vmb[mi].x >> 24)*0x20002u;
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    const int r = 16*ks;
                    const uint32_t af[4] = {
                        kvq_rows(*(const uint2 *) (pa + r),     *(const uint2 *) (pa + r + KVQ_LDR),     sha, mka, vma[mi].y),
                        kvq_rows(*(const uint2 *) (pb + r),     *(const uint2 *) (pb + r + KVQ_LDR),     shb, mkb, vmb[mi].y),
                        kvq_rows(*(const uint2 *) (pa + r + 8), *(const uint2 *) (pa + r + 8 + KVQ_LDR), sha, mka, vma[mi].y),
                        kvq_rows(*(const uint2 *) (pb + r + 8), *(const uint2 *) (pb + r + 8 + KVQ_LDR), shb, mkb, vmb[mi].y),
                    };
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        kvq_mma(o[mi][j], af, bp[j][ks][0], bp[j][ks][1]);
                    }
                }
            }
        }
        store_mask(nxt);
    }
    cp_async_wait_all();

    if (SCORE) {
        kvq_write_mz<NT>(m_c, z_c, s_red, a.pmc + ((int64_t) g*a.n_cblk + blk)*N, a.pzc + ((int64_t) g*a.n_cblk + blk)*N, w, lane);
    }
    if (ATTN) {
        const int64_t ip = (int64_t) g*a.n_part + blk;
        kvq_write_mz<NT>(m_a, z_a, s_red, a.pm + ip*N, a.pz + ip*N, w, lane);
        float * po = a.po + ip*N*KVQ_D;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n  = 8*j + 2*t4 + (e & 1);
                    const int cc = 16*(w + 4*mi) + gid + (e < 2 ? 0 : 8);
                    po[n*KVQ_D + cc] = o[mi][j][e];
                }
            }
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// precise rows: resident slots or tail rows; grid (n_blocks, G)
template <int N, ggml_type type, bool ATTN>
static __global__ void __launch_bounds__(KVQ_NTHREADS) kvq_precise(const kvq_args a, const kvq_src src, const int part0) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int NT = N/8;

    const int bx   = blockIdx.x;
    const int g    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int w    = tid / WARP_SIZE;
    const int gid  = lane / 4;
    const int t4   = lane % 4;

    const int r0 = bx*KVQ_PROWS;
    const int r1 = min(src.n_rows, r0 + KVQ_PROWS);

    extern __shared__ __align__(16) char kvq_smem[];
    half  * s_q    = (half *) kvq_smem;             // [N][LDT]
    half  * s_tile = s_q + N*KVQ_LDT;               // [T][LDT]
    half  * s_p    = s_tile + KVQ_T*KVQ_LDT;        // [N][LDP]
    float * s_red  = (float *) (s_p + N*KVQ_LDP);   // [WARPS][N]
    int   * s_row  = (int *) (s_red + KVQ_WARPS*N); // [T]
    char  * s_st   = (char *) (s_row + KVQ_T);      // [T][KVQ_Q8_ROW], q8_0 only

    for (int e = tid; e < N*KVQ_D; e += KVQ_NTHREADS) {
        const int n = e / KVQ_D;
        const int i = e % KVQ_D;
        float qv = 0.0f;
        if (n < a.cols) {
            const int h = g*a.n_rep + n % a.n_rep;
            qv = a.q[i + h*a.q_nb1 + (n / a.n_rep)*a.q_nb2];
        }
        s_q[n*KVQ_LDT + i] = __float2half_rn(qv);
    }

    // rows of a tile into s_tile as f16, zeros for invalid rows
    auto load_tile = [&](const char * base, const int64_t nb1, const int tr) {
        if constexpr (type == GGML_TYPE_F16) {
            for (int e = tid; e < KVQ_T*(KVQ_D/8); e += KVQ_NTHREADS) {
                const int r = e / (KVQ_D/8);
                const int i = e % (KVQ_D/8);
                half * dst = s_tile + r*KVQ_LDT + 8*i;
                if (tr + r < r1) {
                    cp_async_cg_16<0>((unsigned int) __cvta_generic_to_shared(dst), base + (int64_t) (tr + r)*nb1 + 16*i);
                } else {
                    *(uint4 *) dst = make_uint4(0, 0, 0, 0);
                }
            }
            cp_async_commit_group();
            cp_async_wait_all();
        } else {
            constexpr int nseg = KVQ_Q8_ROW/16;
            for (int e = tid; e < KVQ_T*nseg; e += KVQ_NTHREADS) {
                const int r = e / nseg;
                const int i = e % nseg;
                if (tr + r < r1) {
                    cp_async_cg_16<0>((unsigned int) __cvta_generic_to_shared(s_st + r*KVQ_Q8_ROW + 16*i), base + (int64_t) (tr + r)*nb1 + 16*i);
                }
            }
            cp_async_commit_group();
            cp_async_wait_all();
            __syncthreads();
            for (int e = tid; e < KVQ_T*(KVQ_D/QK8_0); e += KVQ_NTHREADS) {
                const int r = e / (KVQ_D/QK8_0);
                const int ib = e % (KVQ_D/QK8_0);
                half2 * dst = (half2 *) (s_tile + r*KVQ_LDT + QK8_0*ib);
                if (tr + r < r1) {
                    const char * blkp = s_st + r*KVQ_Q8_ROW + ib*sizeof(block_q8_0);
                    const half   d  = *(const half *) blkp;
                    const int8_t * qs = (const int8_t *) (blkp + sizeof(half));
#pragma unroll
                    for (int i = 0; i < QK8_0/2; ++i) {
                        dst[i] = __hmul2(__halves2half2(__int2half_rn(qs[2*i]), __int2half_rn(qs[2*i + 1])), __half2half2(d));
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < QK8_0/2; ++i) {
                        dst[i] = __float2half2_rn(0.0f);
                    }
                }
            }
        }
    };

    float m[NT][2], z[NT][2], alpha[NT][2];
    float o[4][NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            m[j][c] = -INFINITY;
            z[j][c] = 0.0f;
        }
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o[mi][j][e] = 0.0f;
            }
        }
    }

    const char * kbase = src.k + g*src.k_nb2;
    const char * vbase = ATTN ? src.v + g*src.v_nb2 : nullptr;

    for (int tr = r0; tr < r1; tr += KVQ_T) {
        if (tid < KVQ_T) {
            const int sr = tr + tid;
            int row = -1;
            if (sr < r1) {
                if (src.res) {
                    const int32_t idx = a.ridx[sr + g*a.ridx_nb1];
                    row = idx >= 0 && idx < a.n_coarse ? idx : -1;
                } else {
                    row = a.n_coarse + sr;
                }
            }
            s_row[tid] = row;
        }
        load_tile(kbase, src.k_nb1, tr);
        __syncthreads();

        float s[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                s[j][e] = 0.0f;
            }
        }
#pragma unroll
        for (int kk2 = 0; kk2 < KVQ_D/32; ++kk2) {
            uint32_t bq[NT][4];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                kvq_ldsm_x4(bq[j], s_q + (8*j + lane % 8)*KVQ_LDT + 32*kk2 + 8*(lane / 8));
            }
#pragma unroll
            for (int hk = 0; hk < 2; ++hk) {
                uint32_t af[4];
                kvq_ldsm_x4(af, s_tile + (16*w + lane % 16)*KVQ_LDT + 32*kk2 + 16*hk + 8*(lane / 16));
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    kvq_mma(s[j], af, bq[j][2*hk], bq[j][2*hk + 1]);
                }
            }
        }

        const int la = 16*w + gid;
        const int rowa = s_row[la];
        const int rowb = s_row[la + 8];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int n   = 8*j + 2*t4 + (e & 1);
                const int row = e < 2 ? rowa : rowb;
                float x = -INFINITY;
                if (row >= 0 && n < a.cols) {
                    x = s[j][e]*a.scale2 + (a.mask ? KVQ_LOG2E*__half2float(a.mask[row + (n / a.n_rep)*a.mask_nb1]) : 0.0f);
                }
                s[j][e] = x;
            }
        }

        kvq_tile_max<NT>(s, s_red, w, lane);
        __syncthreads();
        const bool rescale = kvq_tile_update<NT>(s, m, z, alpha, s_red, lane);

        if (ATTN) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n = 8*j + 2*t4 + (e & 1);
                    s_p[n*KVQ_LDP + la + (e < 2 ? 0 : 8)] = __float2half_rn(s[j][e]);
                }
            }
            if (rescale) {
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            o[mi][j][e] *= alpha[j][e & 1];
                        }
                    }
                }
            }
            __syncthreads();
            load_tile(vbase, src.v_nb1, tr);
            __syncthreads();

            uint32_t bp[NT][4][2];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int ks2 = 0; ks2 < 2; ++ks2) {
                    uint32_t r4[4];
                    kvq_ldsm_x4(r4, s_p + (8*j + lane % 8)*KVQ_LDP + 32*ks2 + 8*(lane / 8));
                    bp[j][2*ks2 + 0][0] = r4[0];
                    bp[j][2*ks2 + 0][1] = r4[1];
                    bp[j][2*ks2 + 1][0] = r4[2];
                    bp[j][2*ks2 + 1][1] = r4[3];
                }
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int mt = w + 4*mi;
#pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    uint32_t af[4];
                    kvq_ldsm_x4_trans(af, s_tile + (16*ks + lane % 8 + 8*(lane / 16))*KVQ_LDT + 16*mt + 8*((lane / 8) % 2));
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        kvq_mma(o[mi][j], af, bp[j][ks][0], bp[j][ks][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    const int64_t ip = (int64_t) g*a.n_part + part0 + bx;
    kvq_write_mz<NT>(m, z, s_red, a.pm + ip*N, a.pz + ip*N, w, lane);
    if (ATTN) {
        float * po = a.po + ip*N*KVQ_D;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int n  = 8*j + 2*t4 + (e & 1);
                    const int cc = 16*(w + 4*mi) + gid + (e < 2 ? 0 : 8);
                    po[n*KVQ_D + cc] = o[mi][j][e];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(a, src, part0);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// merge the coarse partials of a chunk: y = O*step, per column the max and the sum of weights
// grid (n_chunk*G, N), D threads (component)
static __global__ void __launch_bounds__(KVQ_D) kvq_merge(const kvq_args a) {
    const int tab = blockIdx.x;
    const int t   = tab / a.G;
    const int g   = tab % a.G;
    const int n   = blockIdx.y;
    const int c   = threadIdx.x;
    const int N   = a.N;

    const float * aux = a.taux + t*a.taux_nb3 + (2*g + 1)*a.taux_nb2;

    const int64_t ip0 = (int64_t) g*a.n_part + (int64_t) t*a.n_cbpc;
    const int64_t ic  = (int64_t) g*a.n_chunk + t;

    float mx = -INFINITY;
    for (int b = 0; b < a.n_cbpc; ++b) {
        mx = fmaxf(mx, a.pm[(ip0 + b)*N + n]);
    }
    float oc = 0.0f;
    float zs = 0.0f;
    for (int b = 0; b < a.n_cbpc && mx > -INFINITY; ++b) {
        const float wb = kvq_exp2(a.pm[(ip0 + b)*N + n] - mx);
        oc += wb*a.po[((ip0 + b)*N + n)*KVQ_D + c];
        zs += wb*a.pz[(ip0 + b)*N + n];
    }
    a.cy[(ic*N + n)*KVQ_D + c] = aux[2*KVQ_D + c] > 0.0f ? oc*aux[c] : 0.0f;
    if (c == 0) {
        a.cm[ic*N + n] = mx;
        a.cz[ic*N + n] = zs;
    }
}

// out = y inv + z mu per chunk; grid (n_chunk*G, D/32), D threads: lane = dim of the slice, warp w sums components [32w, 32w + 32)
template <int N>
static __global__ void __launch_bounds__(KVQ_D) kvq_combine(const kvq_args a) {
    const int tab   = blockIdx.x;
    const int t     = tab / a.G;
    const int g     = tab % a.G;
    const int slice = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int w     = tid / WARP_SIZE;

    __shared__ float s_y[8][KVQ_D];
    __shared__ float s_out[8][WARP_SIZE];

    const half  * inv = a.tinv + t*a.tinv_nb3 + (2*g + 1)*a.tinv_nb2;
    const float * aux = a.taux + t*a.taux_nb3 + (2*g + 1)*a.taux_nb2;

    const int64_t ic = (int64_t) g*a.n_chunk + t;
    const int     i  = WARP_SIZE*slice + lane;

    float x[WARP_SIZE];
#pragma unroll
    for (int k = 0; k < WARP_SIZE; ++k) {
        x[k] = __half2float(inv[i + KVQ_D*(WARP_SIZE*w + k)]);
    }

    for (int n0 = 0; n0 < N; n0 += 8) {
        for (int e = tid; e < 8*KVQ_D; e += KVQ_D) {
            s_y[e / KVQ_D][e % KVQ_D] = a.cy[(ic*N + n0)*KVQ_D + e];
        }
        s_out[w][lane] = 0.0f;
        __syncthreads();

        float acc[8] = { 0.0f };
#pragma unroll
        for (int k = 0; k < WARP_SIZE; ++k) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j] += s_y[j][WARP_SIZE*w + k]*x[k];
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            atomicAdd(&s_out[j][lane], acc[j]);
        }
        __syncthreads();

        const int n = n0 + w;
        a.co[(ic*N + n)*KVQ_D + i] = s_out[w][lane] + a.cz[ic*N + n]*aux[KVQ_D + i];
        __syncthreads();
    }
}

// block-wide max and sum over the D threads of kvq_final
static __device__ __forceinline__ float kvq_block_max(float x, float * s_r) {
    x = warp_reduce_max(x);
    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        s_r[threadIdx.x / WARP_SIZE] = x;
    }
    __syncthreads();
    x = -INFINITY;
#pragma unroll
    for (int w = 0; w < KVQ_D/WARP_SIZE; ++w) {
        x = fmaxf(x, s_r[w]);
    }
    return x;
}

static __device__ __forceinline__ float kvq_block_sum(float x, float * s_r) {
    x = warp_reduce_sum(x);
    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        s_r[threadIdx.x / WARP_SIZE] = x;
    }
    __syncthreads();
    x = 0.0f;
#pragma unroll
    for (int w = 0; w < KVQ_D/WARP_SIZE; ++w) {
        x += s_r[w];
    }
    return x;
}

// grid (cols, G), D threads
// sources of the output: the chunks (m, z, out in cm, cz, co) and the precise partials (pm, pz, po from n_cblk on)
template <bool ATTN, bool SCORE>
static __global__ void __launch_bounds__(KVQ_D) kvq_final(const kvq_args a) {
    const int n   = blockIdx.x;
    const int g   = blockIdx.y;
    const int tid = threadIdx.x;
    const int N   = a.N;

    __shared__ float s_r[KVQ_D/WARP_SIZE];

    const int64_t ip0 = (int64_t) g*a.n_part;
    const int64_t ic0 = (int64_t) g*a.n_chunk;
    const int     n_p = a.n_part - a.n_cblk;   // precise partials
    const int     n_s = a.n_chunk + n_p;       // all sources

    // source s: chunk s < n_chunk, else precise partial
    auto src_m = [&](const int s) { return s < a.n_chunk ? a.cm[(ic0 + s)*N + n] : a.pm[(ip0 + a.n_cblk + s - a.n_chunk)*N + n]; };
    auto src_z = [&](const int s) { return s < a.n_chunk ? a.cz[(ic0 + s)*N + n] : a.pz[(ip0 + a.n_cblk + s - a.n_chunk)*N + n]; };

    if (ATTN) {
        float mx = -INFINITY;
        for (int s = tid; s < n_s; s += KVQ_D) {
            mx = fmaxf(mx, src_m(s));
        }
        mx = kvq_block_max(mx, s_r);

        float zs = 0.0f;
        for (int s = tid; s < n_s && mx > -INFINITY; s += KVQ_D) {
            zs += kvq_exp2(src_m(s) - mx)*src_z(s);
        }
        zs = kvq_block_sum(zs, s_r);

        float acc = 0.0f;
        if (mx > -INFINITY) {
#pragma unroll 4
            for (int t = 0; t < a.n_chunk; ++t) {
                acc += kvq_exp2(a.cm[(ic0 + t)*N + n] - mx)*a.co[((ic0 + t)*N + n)*KVQ_D + tid];
            }
#pragma unroll 4
            for (int p = a.n_cblk; p < a.n_part; ++p) {
                acc += kvq_exp2(a.pm[(ip0 + p)*N + n] - mx)*a.po[((ip0 + p)*N + n)*KVQ_D + tid];
            }
        }
        const int h = g*a.n_rep + n % a.n_rep;
        const int l = n / a.n_rep;
        a.dst_attn[tid + KVQ_D*(h + (int64_t) a.n_head*l)] = zs > 0.0f ? acc/zs : 0.0f;
    }

    if (SCORE) {
        // all coarse rows (coarse normalizers of the coarse blocks) and the tail partials
        const int nt0 = a.n_cblk + a.n_rblk;
        const int n_c = a.n_cblk + (a.n_part - nt0);
        auto sc_m = [&](const int s) { return s < a.n_cblk ? a.pmc[((int64_t) g*a.n_cblk + s)*N + n] : a.pm[(ip0 + nt0 + s - a.n_cblk)*N + n]; };
        auto sc_z = [&](const int s) { return s < a.n_cblk ? a.pzc[((int64_t) g*a.n_cblk + s)*N + n] : a.pz[(ip0 + nt0 + s - a.n_cblk)*N + n]; };

        float mx = -INFINITY;
        for (int s = tid; s < n_c; s += KVQ_D) {
            mx = fmaxf(mx, sc_m(s));
        }
        mx = kvq_block_max(mx, s_r);
        float zs = 0.0f;
        for (int s = tid; s < n_c && mx > -INFINITY; s += KVQ_D) {
            zs += kvq_exp2(sc_m(s) - mx)*sc_z(s);
        }
        zs = kvq_block_sum(zs, s_r);
        if (tid == 0) {
            a.scs[(g*2 + 0)*N + n] = mx;
            a.scs[(g*2 + 1)*N + n] = zs;
        }
    }
}

// grid (ceil(n_coarse/256), G), 256 threads
static __global__ void kvq_score(const kvq_args a) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int     g = blockIdx.y;
    if (j >= a.n_coarse) {
        return;
    }
    const int t    = (int) (j / a.n_crow);
    const int jj   = (int) (j - (int64_t) t*a.n_crow);
    const int b    = jj / a.rows_blk;
    const int tile = (t*a.n_cbpc + b)*a.n_tiles_blk + (jj - b*a.rows_blk)/KVQ_T;

    const half  * p  = a.scp + ((int64_t) g*a.n_coarse + j)*a.cols;
    const float * mt = a.scm + ((int64_t) g*a.n_cblk*a.n_tiles_blk + tile)*a.cols;

    for (int l = 0; l < a.L; ++l) {
        float s = 0.0f;
        for (int r = 0; r < a.n_rep; ++r) {
            const int n = r + a.n_rep*l;
            const float mx = a.scs[(g*2 + 0)*a.N + n];
            const float zs = a.scs[(g*2 + 1)*a.N + n];
            if (zs > 0.0f) {
                s += __half2float(p[n])*kvq_exp2(mt[n] - mx)/zs;
            }
        }
        a.dst_score[j + (int64_t) a.n_coarse*(g + (int64_t) a.G*l)] = s;
    }
}

static __global__ void kvq_resmap(const kvq_args a) {
    const int s = blockIdx.x*blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (s >= a.cap) {
        return;
    }
    const int32_t idx = a.ridx[s + g*a.ridx_nb1];
    if (idx >= 0 && idx < a.n_coarse) {
        atomicOr(&a.resmap[g*a.nres_words + idx/32], 1u << (idx % 32));
    }
}

static int kvq_rows_per_block() {
    static const int rows = [] {
        const char * env = getenv("GGML_KVQ_ROWS_PER_BLOCK");
        const int r = env ? atoi(env) : 1024;
        return r >= KVQ_T && r % KVQ_T == 0 ? r : 1024;
    }();
    return rows;
}

template <auto kernel>
static void kvq_set_smem() {
    static bool done[GGML_CUDA_MAX_DEVICES] = { false };
    const int id = ggml_cuda_get_device();
    if (!done[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) ggml_cuda_info().devices[id].smpbo));
        done[id] = true;
    }
}

static size_t kvq_coarse_smem(const int N, const int ld_st, const int L) {
    return N*KVQ_LDT*sizeof(half) + N*KVQ_LDP*sizeof(half) + KVQ_D/2*sizeof(uint4) + 2*KVQ_WARPS*N*sizeof(float)
        + KVQ_NBUF*ld_st*sizeof(uint32_t) + KVQ_NBUF*L*KVQ_T*sizeof(half);
}

static size_t kvq_precise_smem(const int N) {
    return N*KVQ_LDT*sizeof(half) + KVQ_T*KVQ_LDT*sizeof(half) + N*KVQ_LDP*sizeof(half) + KVQ_WARPS*N*sizeof(float)
        + KVQ_T*sizeof(int) + KVQ_T*KVQ_Q8_ROW;
}

template <int N, ggml_type type, bool ATTN>
static void kvq_launch_precise(const kvq_args & a, const kvq_src & src, const int part0, const int n_blk, cudaStream_t stream) {
    if (n_blk == 0) {
        return;
    }
    kvq_set_smem<kvq_precise<N, type, ATTN>>();
    kvq_precise<N, type, ATTN><<<dim3(n_blk, a.G), KVQ_NTHREADS, kvq_precise_smem(N), stream>>>(a, src, part0);
}

template <int N, bool ATTN>
static void kvq_launch_precise_t(const kvq_args & a, const kvq_src & src, const ggml_type type, const int part0, const int n_blk, cudaStream_t stream) {
    if (type == GGML_TYPE_F16) {
        kvq_launch_precise<N, GGML_TYPE_F16, ATTN>(a, src, part0, n_blk, stream);
    } else {
        kvq_launch_precise<N, GGML_TYPE_Q8_0, ATTN>(a, src, part0, n_blk, stream);
    }
}

// the precise rows run on a side stream, concurrently with the coarse rows
template <int N>
static void kvq_launch(const kvq_args & a, const bool attn, const bool score,
        const kvq_src & src_res, const ggml_type type_res, const kvq_src & src_tail, const ggml_type type_tail,
        cudaStream_t stream, cudaStream_t side) {
    static cudaEvent_t ev_fork[GGML_CUDA_MAX_DEVICES] = { nullptr };
    static cudaEvent_t ev_join[GGML_CUDA_MAX_DEVICES] = { nullptr };
    const int id = ggml_cuda_get_device();
    if (!ev_fork[id]) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork[id], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_join[id], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(ev_fork[id], stream));
    CUDA_CHECK(cudaStreamWaitEvent(side, ev_fork[id], 0));
    if (attn) {
        kvq_launch_precise_t<N, true>(a, src_res, type_res, a.n_cblk, a.n_rblk, side);
        kvq_launch_precise_t<N, true>(a, src_tail, type_tail, a.n_cblk + a.n_rblk, a.n_tblk, side);
    } else {
        kvq_launch_precise_t<N, false>(a, src_tail, type_tail, a.n_cblk + a.n_rblk, a.n_tblk, side);
    }
    CUDA_CHECK(cudaEventRecord(ev_join[id], side));

    if (a.n_chunk > 0) {
        CUDA_CHECK(cudaMemsetAsync(a.kmax, 0, (size_t) a.n_chunk*a.G*N*sizeof(uint32_t), stream));
        kvq_set_smem<kvq_prep<N>>();
        kvq_prep<N><<<dim3(a.n_chunk*a.G, KVQ_D/WARP_SIZE), KVQ_D, (N*KVQ_D + N*WARP_SIZE)*sizeof(float), stream>>>(a, attn);

        const int    ld_st = (a.nrk + (attn ? a.nrv : 0))*KVQ_LDR;
        const size_t smem  = kvq_coarse_smem(N, ld_st, a.L);
        const dim3   grid(a.n_cbpc, a.n_chunk, a.G);
        if (attn && score) {
            kvq_set_smem<kvq_coarse<N, true, true>>();
            kvq_coarse<N, true, true><<<grid, KVQ_NTHREADS, smem, stream>>>(a);
        } else if (attn) {
            kvq_set_smem<kvq_coarse<N, true, false>>();
            kvq_coarse<N, true, false><<<grid, KVQ_NTHREADS, smem, stream>>>(a);
        } else {
            kvq_set_smem<kvq_coarse<N, false, true>>();
            kvq_coarse<N, false, true><<<grid, KVQ_NTHREADS, smem, stream>>>(a);
        }
    }

    if (attn && a.n_chunk > 0) {
        kvq_merge<<<dim3(a.n_chunk*a.G, N), KVQ_D, 0, stream>>>(a);
        kvq_combine<N><<<dim3(a.n_chunk*a.G, KVQ_D/WARP_SIZE), KVQ_D, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, ev_join[id], 0));

    const dim3 grid_f(a.cols, a.G);
    if (attn && score) {
        kvq_final<true, true><<<grid_f, KVQ_D, 0, stream>>>(a);
    } else if (attn) {
        kvq_final<true, false><<<grid_f, KVQ_D, 0, stream>>>(a);
    } else {
        kvq_final<false, true><<<grid_f, KVQ_D, 0, stream>>>(a);
    }

    if (score) {
        kvq_score<<<dim3((a.n_coarse + 255)/256, a.G), 256, 0, stream>>>(a);
    }
}

void ggml_cuda_op_kvq_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * kc   = dst->src[1];
    const ggml_tensor * vc   = dst->src[2];
    const ggml_tensor * tinv = dst->src[3];
    const ggml_tensor * taux = dst->src[4];
    const ggml_tensor * k    = dst->src[5];
    const ggml_tensor * v    = dst->src[6];
    const ggml_tensor * rkv  = dst->src[7];
    const ggml_tensor * ridx = dst->src[8];
    const ggml_tensor * mask = dst->src[9];

    const int  mode  = ggml_get_op_params_i32(dst, 2);
    const bool attn  = mode & GGML_KVQ_ATTN;
    const bool score = mode & GGML_KVQ_SCORE;

    cudaStream_t stream = ctx.stream();
    cudaStream_t side   = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - 1);
    ggml_cuda_pool & pool = ctx.pool();

    kvq_args a = {};
    a.G        = (int) k->ne[2];
    a.n_head   = (int) q->ne[1];
    a.n_rep    = a.n_head / a.G;
    a.L        = (int) q->ne[2];
    a.cols     = a.n_rep*a.L;
    a.N        = GGML_PAD(a.cols, 8);
    a.n_coarse = kc ? (int) kc->ne[1] : 0;
    a.n_crow   = ggml_get_op_params_i32(dst, 1);
    a.n_chunk  = kc ? (a.n_coarse + a.n_crow - 1)/a.n_crow : 0;
    a.rows_blk = kvq_rows_per_block();
    a.n_cbpc   = a.n_chunk > 0 ? (std::min(a.n_crow, a.n_coarse) + a.rows_blk - 1)/a.rows_blk : 0;
    a.n_tiles_blk = a.rows_blk/KVQ_T;
    a.n_cblk   = a.n_chunk*a.n_cbpc;
    a.n_tail   = (int) k->ne[1];
    a.cap      = attn && rkv ? (int) rkv->ne[1] : 0;
    a.n_rblk   = (a.cap + KVQ_PROWS - 1)/KVQ_PROWS;
    a.n_tblk   = (a.n_tail + KVQ_PROWS - 1)/KVQ_PROWS;
    a.n_part   = a.n_cblk + a.n_rblk + a.n_tblk;
    a.nwk      = kc ? (int) kc->ne[0] : 1;
    a.nwv      = attn && kc ? (int) vc->ne[0] - 1 : 1;
    a.nrk      = a.nwk + 2;
    a.nrv      = a.nwv + 2;
    a.scale2   = ggml_get_op_params_f32(dst, 0)*KVQ_LOG2E;

    a.q     = (const float *) q->data;
    a.q_nb1 = q->nb[1]/sizeof(float);
    a.q_nb2 = q->nb[2]/sizeof(float);
    if (kc) {
        a.kc = (const char *) kc->data; a.kc_nb1 = kc->nb[1]; a.kc_nb2 = kc->nb[2];
        a.tinv = (const half *) tinv->data; a.tinv_nb2 = tinv->nb[2]/sizeof(half); a.tinv_nb3 = tinv->nb[3]/sizeof(half);
        a.taux = (const float *) taux->data; a.taux_nb2 = taux->nb[2]/sizeof(float); a.taux_nb3 = taux->nb[3]/sizeof(float);
        if (attn) {
            a.vc = (const char *) vc->data; a.vc_nb1 = vc->nb[1]; a.vc_nb2 = vc->nb[2];
        }
    }
    if (a.cap > 0) {
        a.ridx = (const int32_t *) ridx->data;
        a.ridx_nb1 = ridx->nb[1]/sizeof(int32_t);
    }
    if (mask) {
        a.mask = (const half *) mask->data;
        a.mask_nb1 = mask->nb[1]/sizeof(half);
    }

    const int N  = a.N;
    const int G  = a.G;
    const int nc = std::max(1, a.n_chunk);

    ggml_cuda_pool_alloc<float>    qtf_a(pool, (size_t) nc*G*N*KVQ_D);
    ggml_cuda_pool_alloc<uint32_t> kmax_a(pool, (size_t) nc*G*N);
    ggml_cuda_pool_alloc<float>    kmu_a(pool, (size_t) nc*G*N);
    ggml_cuda_pool_alloc<uint4>    kpair_a(pool, (size_t) nc*G*KVQ_D/2);
    ggml_cuda_pool_alloc<uint2>    vfld_a(pool, (size_t) nc*G*KVQ_D);
    ggml_cuda_pool_alloc<int>      nst_a(pool, (size_t) nc*G*2);
    a.nres_words = (a.n_coarse + 31)/32;
    ggml_cuda_pool_alloc<uint32_t> resmap_a(pool, (size_t) std::max(1, G*a.nres_words));
    ggml_cuda_pool_alloc<float>    po_a(pool, attn ? (size_t) G*a.n_part*N*KVQ_D : 1);
    ggml_cuda_pool_alloc<float>    pm_a(pool, (size_t) G*a.n_part*N);
    ggml_cuda_pool_alloc<float>    pz_a(pool, (size_t) G*a.n_part*N);
    ggml_cuda_pool_alloc<float>    pmc_a(pool, (size_t) std::max(1, G*a.n_cblk*N));
    ggml_cuda_pool_alloc<float>    pzc_a(pool, (size_t) std::max(1, G*a.n_cblk*N));
    ggml_cuda_pool_alloc<float>    cy_a(pool, attn ? (size_t) G*nc*N*KVQ_D : 1);
    ggml_cuda_pool_alloc<float>    co_a(pool, attn ? (size_t) G*nc*N*KVQ_D : 1);
    ggml_cuda_pool_alloc<float>    cm_a(pool, (size_t) G*nc*N);
    ggml_cuda_pool_alloc<float>    cz_a(pool, (size_t) G*nc*N);
    ggml_cuda_pool_alloc<half>     scp_a(pool, score ? (size_t) G*a.n_coarse*a.cols : 1);
    ggml_cuda_pool_alloc<float>    scm_a(pool, score ? (size_t) G*a.n_cblk*a.n_tiles_blk*a.cols : 1);
    ggml_cuda_pool_alloc<float>    scs_a(pool, (size_t) G*2*N);

    a.qtf = qtf_a.get(); a.kmax = kmax_a.get(); a.kmu = kmu_a.get();
    a.kpair = kpair_a.get(); a.vfld = vfld_a.get(); a.nst = nst_a.get();
    a.resmap = resmap_a.get();
    a.po = po_a.get(); a.pm = pm_a.get(); a.pz = pz_a.get(); a.pmc = pmc_a.get(); a.pzc = pzc_a.get();
    a.cy = cy_a.get(); a.co = co_a.get(); a.cm = cm_a.get(); a.cz = cz_a.get();
    a.scp = scp_a.get(); a.scm = scm_a.get(); a.scs = scs_a.get();

    float * dst_d = (float *) dst->data;
    a.dst_attn  = attn ? dst_d : nullptr;
    a.dst_score = score ? dst_d + (attn ? (int64_t) KVQ_D*a.n_head*a.L : 0) : nullptr;

    if (a.cap > 0 && a.n_coarse > 0) {
        CUDA_CHECK(cudaMemsetAsync(a.resmap, 0, (size_t) G*a.nres_words*sizeof(uint32_t), stream));
        kvq_resmap<<<dim3((a.cap + 255)/256, G), 256, 0, stream>>>(a);
    }

    kvq_src src_res = {};
    if (a.cap > 0) {
        src_res.k = (const char *) rkv->data; src_res.k_nb1 = rkv->nb[1]; src_res.k_nb2 = rkv->nb[2];
        src_res.v = (const char *) rkv->data + rkv->nb[3]; src_res.v_nb1 = rkv->nb[1]; src_res.v_nb2 = rkv->nb[2];
        src_res.n_rows = a.cap;
        src_res.res = true;
    }
    kvq_src src_tail = {};
    src_tail.k = (const char *) k->data; src_tail.k_nb1 = k->nb[1]; src_tail.k_nb2 = k->nb[2];
    if (attn) {
        src_tail.v = (const char *) v->data; src_tail.v_nb1 = v->nb[1]; src_tail.v_nb2 = v->nb[2];
    }
    src_tail.n_rows = a.n_tail;
    src_tail.res = false;

    const ggml_type type_res  = a.cap > 0 ? rkv->type : GGML_TYPE_F16;
    const ggml_type type_tail = k->type;

    switch (N) {
        case  8: kvq_launch< 8>(a, attn, score, src_res, type_res, src_tail, type_tail, stream, side); break;
        case 16: kvq_launch<16>(a, attn, score, src_res, type_res, src_tail, type_tail, stream, side); break;
        case 24: kvq_launch<24>(a, attn, score, src_res, type_res, src_tail, type_tail, stream, side); break;
        case 32: kvq_launch<32>(a, attn, score, src_res, type_res, src_tail, type_tail, stream, side); break;
        case 40: kvq_launch<40>(a, attn, score, src_res, type_res, src_tail, type_tail, stream, side); break;
        case 48: kvq_launch<48>(a, attn, score, src_res, type_res, src_tail, type_tail, stream, side); break;
        default: GGML_ABORT("unsupported number of queries per kv head");
    }
}

bool ggml_cuda_kvq_attn_supported(int device, const ggml_tensor * op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!ampere_mma_available(cc)) {
        return false;
    }

    const ggml_tensor * q    = op->src[0];
    const ggml_tensor * kc   = op->src[1];
    const ggml_tensor * vc   = op->src[2];
    const ggml_tensor * k    = op->src[5];
    const ggml_tensor * v    = op->src[6];
    const ggml_tensor * rkv  = op->src[7];

    // 64 rows of mask per thread block of 128 threads: at most 8 positions
    if (q->ne[0] != KVQ_D || q->ne[2] > 8 || (q->ne[1]/k->ne[2])*q->ne[2] > 48) {
        return false;
    }
    if (kc && (kc->ne[0] > 32 || kc->nb[1] % 4 != 0 || kc->nb[2] % 4 != 0)) {
        return false;
    }
    if (vc && (vc->ne[0] > 33 || vc->nb[1] % 4 != 0 || vc->nb[2] % 4 != 0)) {
        return false;
    }
    // precise rows are copied in 16 byte pieces
    for (const ggml_tensor * t : { k, v, rkv }) {
        if (t && (t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0)) {
            return false;
        }
    }
    return true;
}
