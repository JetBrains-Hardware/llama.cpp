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

// the table of a task (ggml_kvq_fit_table), D <= 256: Cholesky and L^-1 in double, S = L^T C L in double (cuBLAS), the eigen solver
// in float (the tred2 / tql2 of the CPU codec: tridiagonal form and transformations per task, the QL steps on one thread, their
// rotations replayed on the rows by many blocks), fwd and inv in double (cuBLAS)

#define KVQ_TAB_NT 1024 // threads of the per-task kernels
#define KVQ_TAB_P  4    // threads per column (n <= 256)

// block sum, every thread gets the same value
template <typename T>
static __device__ T kvq_tab_sum(T v, T * red) {
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = v;
    }
    __syncthreads();
    T s = 0;
    for (int w = 0; w < KVQ_TAB_NT/32; w++) {
        s += red[w];
    }
    return s;
}

// L (column-major, lower) of metric + ridge, the first ridge that works
// thread t works on row or column t % 256 with the part t / 256 of each sum
static __global__ void __launch_bounds__(KVQ_TAB_NT) kvq_tab_chol(const float * metric, const int64_t sm1, const int64_t sm2, double * L_all,
        int * st_all, float * ridge_all, const int n) {
    const int     b    = blockIdx.x;
    const int     tid  = threadIdx.x;
    const int     jj   = tid % 256;
    const int     part = tid / 256;
    const float * M    = metric + b*sm2;
    double      * Lc   = L_all + (size_t) b*n*n;

    __shared__ double pd[KVQ_TAB_P][256];
    __shared__ double sh_tr;
    __shared__ double sh_ljj;
    __shared__ int    sh_fail;

    if (tid == 0) {
        double tr = 0.0;
        for (int i = 0; i < n; i++) {
            tr += M[i*sm1 + i];
        }
        sh_tr = tr;
    }
    __syncthreads();

    const double ridges[3] = { 1e-6, 1e-4, 1e-2 };
    bool ok = false;
    for (int q = 0; q < 3 && !ok; q++) {
        const double add = ridges[q]*sh_tr/n;
        // Lc[k*n + i] = element (i, k) = M[k][i] = M[i][k]
        for (int idx = tid; idx < n*n; idx += KVQ_TAB_NT) {
            const int k = idx / n;
            const int i = idx % n;
            Lc[idx] = (double) M[k*sm1 + i] + (i == k ? add : 0.0);
        }
        __syncthreads();
        ok = true;
        for (int j = 0; j < n; j++) {
            if (tid < 32) {
                double s = 0.0;
                for (int k = tid; k < j; k += 32) {
                    const double v = Lc[(size_t) k*n + j];
                    s += v*v;
                }
                for (int o = 16; o > 0; o >>= 1) {
                    s += __shfl_xor_sync(0xffffffff, s, o);
                }
                if (tid == 0) {
                    s = Lc[(size_t) j*n + j] - s;
                    sh_fail = !(s > 0.0);
                    if (!sh_fail) {
                        sh_ljj = sqrt(s);
                        Lc[(size_t) j*n + j] = sh_ljj;
                    }
                }
            }
            __syncthreads();
            if (sh_fail) {
                ok = false;
                break;
            }
            const int i = j + 1 + jj;
            if (i < n) {
                double v = 0.0;
                for (int k = part; k < j; k += KVQ_TAB_P) {
                    v += Lc[(size_t) k*n + i]*Lc[(size_t) k*n + j];
                }
                pd[part][jj] = v;
            }
            __syncthreads();
            if (i < n && part == 0) {
                Lc[(size_t) j*n + i] = (Lc[(size_t) j*n + i] - ((pd[0][jj] + pd[1][jj]) + (pd[2][jj] + pd[3][jj])))/sh_ljj;
            }
            __syncthreads();
        }
        if (tid == 0) {
            ridge_all[b] = (float) ridges[q];
        }
        __syncthreads();
    }
    if (tid == 0) {
        st_all[b] = ok ? 0 : 1;
    }
    if (!ok) {
        return;
    }
    // the upper triangle of column-major L
    for (int idx = tid; idx < n*n; idx += KVQ_TAB_NT) {
        if (idx % n < idx / n) {
            Lc[idx] = 0.0;
        }
    }
}

// the pointers of the batched triangular solve: A = L of task b, B = U of task b
static __global__ void kvq_tab_ptrs(const double * L_all, double * U_all, const double ** pa, double ** pb, const int n, const int B) {
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b < B) {
        pa[b] = L_all + (size_t) b*n*n;
        pb[b] = U_all + (size_t) b*n*n;
    }
}

// C (double, column-major) = cov/n_tok; cov is symmetric
static __global__ void kvq_tab_cov(const float * cov, const int64_t sc1, const int64_t sc2, double * C_all, const int n, const double inv_n) {
    const int b = blockIdx.y;
    for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < n*n; idx += gridDim.x*blockDim.x) {
        C_all[(size_t) b*n*n + idx] = (double) cov[b*sc2 + (idx / n)*sc1 + idx % n]*inv_n;
    }
}

// A (float, row-major) = the symmetric part of S (double, column-major)
static __global__ void kvq_tab_sym(const double * S_all, float * A_all, const int n) {
    const int b = blockIdx.y;
    const double * S = S_all + (size_t) b*n*n;
    for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < n*n; idx += gridDim.x*blockDim.x) {
        const int i = idx / n;
        const int j = idx % n;
        A_all[(size_t) b*n*n + idx] = (float) (0.5*(S[(size_t) j*n + i] + S[(size_t) i*n + j]));
    }
}

// tred2 of the CPU codec in float: A (row-major, symmetric) -> the accumulated transformations in A, the tridiagonal form in D, E
// (E shifted for tql2); the active block [0, i) x [0, i) is kept symmetric, so e = A d reads columns
static __global__ void __launch_bounds__(KVQ_TAB_NT) kvq_tab_tred2(float * A_all, float * D_all, float * E_all, const int * st_all, const int n) {
    const int b    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int jj   = tid % 256;
    const int part = tid / 256;
    if (st_all[b] != 0) {
        return;
    }
    float * a = A_all + (size_t) b*n*n;

    __shared__ float d[256];
    __shared__ float e[256];
    __shared__ float pp[KVQ_TAB_P][256];
    __shared__ float red[KVQ_TAB_NT/32];

#define KVQ_V(i, j) a[(size_t) (i)*n + (j)]

    for (int j = tid; j < n; j += KVQ_TAB_NT) {
        d[j] = KVQ_V(n - 1, j);
    }
    __syncthreads();

    for (int i = n - 1; i > 0; i--) {
        float part_s = 0.0f;
        for (int k = tid; k < i; k += KVQ_TAB_NT) {
            part_s += fabsf(d[k]);
        }
        const float scale = kvq_tab_sum(part_s, red);
        if (scale == 0.0f) {
            if (tid == 0) {
                e[i] = d[i - 1];
            }
            __syncthreads();
            for (int j = tid; j < i; j += KVQ_TAB_NT) {
                d[j] = KVQ_V(i - 1, j);
                KVQ_V(i, j) = 0.0f;
                KVQ_V(j, i) = 0.0f;
            }
            if (tid == 0) {
                d[i] = 0.0f;
            }
            __syncthreads();
            continue;
        }
        part_s = 0.0f;
        for (int k = tid; k < i; k += KVQ_TAB_NT) {
            const float v = __fdiv_rn(d[k], scale);
            d[k] = v;
            part_s += v*v;
        }
        float h = kvq_tab_sum(part_s, red);
        const float f = d[i - 1];
        float g = __fsqrt_rn(h);
        if (f > 0.0f) {
            g = -g;
        }
        h = h - f*g;
        __syncthreads();
        if (tid == 0) {
            e[i]     = scale*g;
            d[i - 1] = f - g;
        }
        __syncthreads();
        if (jj < i) {
            float s = 0.0f;
#pragma unroll 8
            for (int k = part; k < i; k += KVQ_TAB_P) {
                s += KVQ_V(k, jj)*d[k];
            }
            pp[part][jj] = s;
        }
        __syncthreads();
        part_s = 0.0f;
        if (tid < i) {
            const float v = __fdiv_rn((pp[0][tid] + pp[1][tid]) + (pp[2][tid] + pp[3][tid]), h);
            e[tid] = v;
            KVQ_V(tid, i) = d[tid];
            part_s = v*d[tid];
        }
        const float hh = __fdiv_rn(kvq_tab_sum(part_s, red), h + h);
        if (tid < i) {
            e[tid] -= hh*d[tid];
        }
        __syncthreads();
        // the same two products for (k, j) and (j, k), no contraction: the block stays symmetric
        if (jj < i) {
            const float dj = d[jj];
            const float ej = e[jj];
#pragma unroll 8
            for (int k = part; k < i; k += KVQ_TAB_P) {
                KVQ_V(k, jj) -= __fadd_rn(__fmul_rn(dj, e[k]), __fmul_rn(ej, d[k]));
            }
        }
        __syncthreads();
        if (tid < i) {
            d[tid] = KVQ_V(i - 1, tid);
            KVQ_V(i, tid) = 0.0f;
        }
        if (tid == 0) {
            d[i] = h;
        }
        __syncthreads();
    }

    // the transformations
    for (int i = 0; i < n - 1; i++) {
        if (tid == 0) {
            KVQ_V(n - 1, i) = KVQ_V(i, i);
            KVQ_V(i, i) = 1.0f;
        }
        __syncthreads();
        const float h = d[i + 1];
        if (h != 0.0f) {
            if (tid <= i) {
                d[tid] = __fdiv_rn(KVQ_V(tid, i + 1), h);
            }
            __syncthreads();
            if (jj <= i) {
                float s = 0.0f;
#pragma unroll 8
                for (int k = part; k <= i; k += KVQ_TAB_P) {
                    s += KVQ_V(k, i + 1)*KVQ_V(k, jj);
                }
                pp[part][jj] = s;
            }
            __syncthreads();
            if (jj <= i) {
                const float g = (pp[0][jj] + pp[1][jj]) + (pp[2][jj] + pp[3][jj]);
#pragma unroll 8
                for (int k = part; k <= i; k += KVQ_TAB_P) {
                    KVQ_V(k, jj) -= g*d[k];
                }
            }
            __syncthreads();
        }
        if (tid <= i) {
            KVQ_V(tid, i + 1) = 0.0f;
        }
        __syncthreads();
    }
    if (tid < n) {
        D_all[(size_t) b*n + tid] = KVQ_V(n - 1, tid);
        E_all[(size_t) b*n + tid] = tid + 1 < n ? e[tid + 1] : 0.0f;
    }
    __syncthreads();
    if (tid < n) {
        KVQ_V(n - 1, tid) = tid == n - 1 ? 1.0f : 0.0f;
    }
#undef KVQ_V
}

// tql2 of the CPU codec on one thread per task: the eigenvalues (sorted, W), the order (perm) and the rotations of the
// eigenvectors: per QL step a header (l, m, offset) and the m - l rotations (c, s) of the columns i, i + 1 for i = m - 1 .. l;
// more than cap_h steps or cap rotations is a failure (status 2)
static __global__ void kvq_tab_ql(float * D_all, float * E_all, int4 * hdr_all, int * nh_all, float2 * cs_all, int * perm_all, float * W_all,
        int * st_all, const int n, const int cap_h, const int cap) {
    // a block per task, one thread; d and e in shared memory (a global store would evict the L1 line of the next load)
    __shared__ float d[256];
    __shared__ float e[256];
    const int b = blockIdx.x;
    if (threadIdx.x != 0 || st_all[b] != 0) {
        return;
    }
    for (int i = 0; i < n; i++) {
        d[i] = D_all[(size_t) b*n + i];
        e[i] = E_all[(size_t) b*n + i];
    }
    int4   * hdr  = hdr_all + (size_t) b*cap_h;
    float2 * rcs  = cs_all + (size_t) b*cap;
    int    * perm = perm_all + (size_t) b*n;

    const float eps  = 1.1920929e-07f;
    float       f    = 0.0f;
    float       tst1 = 0.0f;
    int         nh   = 0;
    int         nr   = 0;
    for (int l = 0; l < n; l++) {
        tst1 = fmaxf(tst1, fabsf(d[l]) + fabsf(e[l]));
        int m = l;
        while (m < n - 1 && fabsf(e[m]) > eps*tst1) {
            m++;
        }
        if (m > l) {
            int iter = 0;
            do {
                if (++iter > 100 || nh >= cap_h || nr + (m - l) > cap) {
                    st_all[b] = 2;
                    return;
                }
                hdr[nh++] = make_int4(l, m, nr, 0);
                float g = d[l];
                float p = __fdiv_rn(d[l + 1] - g, 2.0f*e[l]);
                float r = hypotf(p, 1.0f);
                if (p < 0.0f) {
                    r = -r;
                }
                d[l]     = __fdiv_rn(e[l], p + r);
                d[l + 1] = e[l]*(p + r);
                const float dl1 = d[l + 1];
                float h = g - d[l];
                for (int i = l + 2; i < n; i++) {
                    d[i] -= h;
                }
                f += h;
                p = d[m];
                float c  = 1.0f;
                float c2 = c;
                float c3 = c;
                const float el1 = e[l + 1];
                float s  = 0.0f;
                float s2 = 0.0f;
                // the loads of the next step do not depend on this one: issued early
                float en = e[m - 1];
                float dn = d[m - 1];
                for (int i = m - 1; i >= l; i--) {
                    const float ei = en;
                    const float di = dn;
                    if (i > l) {
                        en = e[i - 1];
                        dn = d[i - 1];
                    }
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c*ei;
                    h = c*p;
                    // r = hypot(p, ei) by one rsqrt (2 ulp: the rotations stay orthogonal far below the quantization error)
                    const float q = p*p + ei*ei;
                    float ir;
                    if (q > 1e-30f && q < 1e30f) {
                        ir = rsqrtf(q);
                        r  = q*ir;
                    } else {
                        r  = hypotf(p, ei);
                        ir = 1.0f/r;
                    }
                    e[i + 1] = s*r;
                    s = ei*ir;
                    c = p*ir;
                    p = c*di - s*g;
                    d[i + 1] = h + s*(c*g + s*di);
                    rcs[nr++] = make_float2(c, s);
                }
                p    = __fdiv_rn(-s*s2*c3*el1*e[l], dl1);
                e[l] = s*p;
                d[l] = c*p;
            } while (fabsf(e[l]) > eps*tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0.0f;
    }
    nh_all[b] = nh;

    // ascending, the selection sort of the CPU codec
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    for (int i = 0; i < n - 1; i++) {
        int   k = i;
        float p = d[i];
        for (int j = i + 1; j < n; j++) {
            if (d[j] < p) {
                k = j;
                p = d[j];
            }
        }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            const int t = perm[i];
            perm[i] = perm[k];
            perm[k] = t;
        }
    }
    for (int c = 0; c < n; c++) {
        W_all[(size_t) b*n + c] = d[c];
    }
}

#define KVQ_TAB_ROWS 32 // rows of V per block of the replay

// the rotations of tql2 on KVQ_TAB_ROWS rows of V per block (a thread per row, the row in shared memory), then U (double, column-major)
// with the columns in the order of the eigenvalues: U[c*n + k] = V[k][perm[c]]
static __global__ void kvq_tab_rot(const float * A_all, const int4 * __restrict__ hdr_all, const int * nh_all, const float2 * __restrict__ cs_all,
        const int * perm_all, const int * st_all, double * U_all, const int n, const int cap_h, const int cap) {
    extern __shared__ float rows[]; // [KVQ_TAB_ROWS][n + 1]

    const int b  = blockIdx.y;
    const int k0 = blockIdx.x*KVQ_TAB_ROWS;
    const int r  = threadIdx.x;
    if (st_all[b] != 0) {
        return;
    }
    const float * a = A_all + (size_t) b*n*n;
    for (int idx = r; idx < KVQ_TAB_ROWS*n; idx += KVQ_TAB_ROWS) {
        const int rr = idx / n;
        const int i  = idx % n;
        rows[rr*(n + 1) + i] = k0 + rr < n ? a[(size_t) (k0 + rr)*n + i] : 0.0f;
    }
    __syncthreads();

    float        * row = rows + r*(n + 1);
    const int4   * hdr = hdr_all + (size_t) b*cap_h;
    const float2 * rcs = cs_all + (size_t) b*cap;
    const int      nh  = nh_all[b];
    // the rotations are one sequence j = 0, 1, ...: lane r holds rotation base + r, the others come by shuffle
    int    base = 0;
    float2 cl   = rcs[r];
    for (int t = 0; t < nh; t++) {
        const int4 hd = hdr[t];
        float zn = row[hd.y];
        for (int i = hd.y - 1, j = hd.z; i >= hd.x; i--, j++) {
            if (j >= base + 32) {
                base += 32;
                cl = base + r < cap ? rcs[base + r] : make_float2(0.0f, 0.0f);
            }
            const float c  = __shfl_sync(0xffffffff, cl.x, j - base);
            const float s  = __shfl_sync(0xffffffff, cl.y, j - base);
            const float zi = row[i];
            row[i + 1] = s*zi + c*zn;
            zn = c*zi - s*zn;
        }
        row[hd.x] = zn;
    }
    __syncthreads();

    const int * perm = perm_all + (size_t) b*n;
    double    * U    = U_all + (size_t) b*n*n;
    if (k0 + r < n) {
        for (int c = 0; c < n; c++) {
            U[(size_t) c*n + k0 + r] = row[perm[c]];
        }
    }
}

// a warp per eigenvector (column of U): the largest |entry| (the first on ties) positive
static __global__ void kvq_tab_sign(double * U_all, const int * st_all, const int n) {
    const int b = blockIdx.y;
    const int c = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    const int l = threadIdx.x % 32;
    if (st_all[b] != 0 || c >= n) {
        return;
    }
    double * u  = U_all + (size_t) b*n*n + (size_t) c*n;
    double   vm = -1.0;
    int      im = n;
    for (int i = l; i < n; i += 32) {
        if (fabs(u[i]) > vm) {
            vm = fabs(u[i]);
            im = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const double vo = __shfl_xor_sync(0xffffffff, vm, o);
        const int    io = __shfl_xor_sync(0xffffffff, im, o);
        if (vo > vm || (vo == vm && io < im)) {
            vm = vo;
            im = io;
        }
    }
    if (u[im] < 0.0) {
        for (int i = l; i < n; i += 32) {
            u[i] = -u[i];
        }
    }
}

// rows [r0, r0 + n) of the table of each task from R (double, row-major n x n); 0 for a failed task
static __global__ void kvq_tab_rows(const double * R_all, const int * st_all, float * tab, const int64_t st1, const int64_t st2, const int n, const int r0) {
    const int b = blockIdx.y;
    const bool ok = st_all[b] == 0;
    for (int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < n*n; idx += gridDim.x*blockDim.x) {
        tab[b*st2 + (int64_t) (r0 + idx / n)*st1 + idx % n] = ok ? (float) R_all[(size_t) b*n*n + idx] : 0.0f;
    }
}

// mu, sigma and the result row of each task
static __global__ void kvq_tab_final(const float * mu, const int64_t smu2, const float * W_all, const int * st_all, const float * ridge_all,
        float * tab, const int64_t st1, const int64_t st2, const int n, const int n_k) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const bool ok = st_all[b] == 0;
    float * t = tab + b*st2;
    for (int c = tid; c < n; c += blockDim.x) {
        t[(int64_t) 2*n*st1 + c]       = ok ? mu[b*smu2 + c] : 0.0f;
        t[(int64_t) (2*n + 1)*st1 + c] = ok ? __fsqrt_rn(fmaxf(W_all[(size_t) b*n + c], 1e-30f)) : 0.0f;
        t[(int64_t) (2*n + 2)*st1 + c] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        double den = 0.0;
        if (ok && b >= n_k) {
            for (int c = 0; c < n; c++) {
                const double sg = t[(int64_t) (2*n + 1)*st1 + c];
                den += sg*sg;
            }
        }
        float * r = t + (int64_t) (2*n + 2)*st1;
        r[0] = (float) den;
        r[1] = (float) st_all[b];
        r[2] = ridge_all[b];
    }
}

static void kvq_fit_table(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * work = dst->src[1];

    const int n     = (int) dst->ne[0];
    const int B     = (int) dst->ne[2];
    const int n_k   = ggml_get_op_params_i32(dst, 1);
    const int n_tok = ggml_get_op_params_i32(dst, 2);
    const size_t nn = (size_t) n*n;

    // in place: rows [0, n) cov, [n, 2n) metric, 2n mu
    const int64_t st1 = dst->nb[1]/sizeof(float);
    const int64_t st2 = dst->nb[2]/sizeof(float);
    float * tab = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    // the QL steps (about 1.3 per eigenvalue) and their rotations (about 0.8 n^2 measured), see ggml_kvq_fit_table_work_size
    const int cap_h = 16*n;
    const int cap   = 3*n*n/2;

    // the scratch: the work tensor, or one block of the pool; 16 arrays, each aligned to 256 bytes
    const size_t ws = ggml_kvq_fit_table_work_size(n, B);
    ggml_cuda_pool_alloc<char> wpool(ctx.pool());
    char * wp = work ? (char *) work->data : wpool.alloc(ws);
    size_t off = 0;
    auto take = [&](size_t bytes) {
        char * p = wp + off;
        off += (bytes + 255) & ~(size_t) 255;
        GGML_ASSERT(off <= ws);
        return p;
    };
    double  * Cd = (double  *) take(nn*B*sizeof(double));
    double  * Lc = (double  *) take(nn*B*sizeof(double));
    double  * U  = (double  *) take(nn*B*sizeof(double));
    float   * A  = (float   *) take(nn*B*sizeof(float));
    float   * Dv = (float   *) take((size_t) n*B*sizeof(float));
    float   * Ev = (float   *) take((size_t) n*B*sizeof(float));
    float   * W  = (float   *) take((size_t) n*B*sizeof(float));
    int     * pm = (int     *) take((size_t) n*B*sizeof(int));
    int4    * hd = (int4    *) take((size_t) cap_h*B*sizeof(int4));
    float2  * rc = (float2  *) take((size_t) cap*B*sizeof(float2));
    int     * nh = (int     *) take(B*sizeof(int));
    float   * rg = (float   *) take(B*sizeof(float));
    int     * st = (int     *) take(B*sizeof(int));
    double ** pp = (double **) take(2*B*sizeof(double *));

    kvq_tab_chol<<<B, KVQ_TAB_NT, 0, stream>>>(tab + n*st1, st1, st2, Lc, st, rg, n);
    const dim3 grid_nn((unsigned) std::min<size_t>((nn + 255)/256, 1024), B);
    kvq_tab_cov<<<grid_nn, 256, 0, stream>>>(tab, st1, st2, Cd, n, 1.0/n_tok);

    // column-major: T = C L (in U), S = L^T T (in C)
    const double one  = 1.0;
    const double zero = 0.0;
    cublasHandle_t h = ctx.cublas_handle();
    CUBLAS_CHECK(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, Cd, n, nn, Lc, n, nn, &zero, U, n, nn, B));
    CUBLAS_CHECK(cublasDgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, Lc, n, nn, U, n, nn, &zero, Cd, n, nn, B));
    kvq_tab_sym<<<grid_nn, 256, 0, stream>>>(Cd, A, n);

    kvq_tab_tred2<<<B, KVQ_TAB_NT, 0, stream>>>(A, Dv, Ev, st, n);
    kvq_tab_ql<<<B, 32, 0, stream>>>(Dv, Ev, hd, nh, rc, pm, W, st, n, cap_h, cap);
    kvq_tab_rot<<<dim3((n + KVQ_TAB_ROWS - 1)/KVQ_TAB_ROWS, B), KVQ_TAB_ROWS, KVQ_TAB_ROWS*(n + 1)*sizeof(float), stream>>>(
            A, hd, nh, rc, pm, st, U, n, cap_h, cap);
    kvq_tab_sign<<<dim3((n + 7)/8, B), 256, 0, stream>>>(U, st, n);

    // fwd row-major = (U^T L^T) column-major; inv row-major = (L^-T U) column-major, a triangular solve in place of U
    CUBLAS_CHECK(cublasDgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &one, U, n, nn, Lc, n, nn, &zero, Cd, n, nn, B));
    kvq_tab_rows<<<grid_nn, 256, 0, stream>>>(Cd, st, tab, st1, st2, n, 0);
    const double ** pa = (const double **) pp;
    double       ** pb = pp + B;
    kvq_tab_ptrs<<<(B + 31)/32, 32, 0, stream>>>(Lc, U, pa, pb, n, B);
    CUBLAS_CHECK(cublasDtrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, n, &one, pa, n, pb, n, B));
    kvq_tab_rows<<<grid_nn, 256, 0, stream>>>(U, st, tab, st1, st2, n, n);

    kvq_tab_final<<<B, 256, 0, stream>>>(tab + 2*n*st1, st2, W, st, rg, tab, st1, st2, n, n_k);
}

// step in units of sigma of the MSE-optimal uniform quantizer for a Gaussian, by bits (Max 1960)
__constant__ float kvq_fit_gauss_step[9] = { 0.0f, 1.596f, 0.9957f, 0.5860f, 0.3352f, 0.1881f, 0.1041f, 0.0569f, 0.0308f };

#define KVQ_BITS_NT 256 // threads of the bits kernel

// block reduction of the bits kernel, every thread gets the result; op 0 sum, 1 min, 2 max
template <int op>
static __device__ double kvq_bits_red(double v, double * red) {
    for (int o = 16; o > 0; o >>= 1) {
        const double w = __shfl_xor_sync(0xffffffff, v, o);
        v = op == 0 ? v + w : op == 1 ? fmin(v, w) : fmax(v, w);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = v;
    }
    __syncthreads();
    double r = red[0];
    for (int w = 1; w < KVQ_BITS_NT/32; w++) {
        r = op == 0 ? r + red[w] : op == 1 ? fmin(r, red[w]) : fmax(r, red[w]);
    }
    return r;
}

// a block per task: the reverse water-filling of the CPU codec (the sums as block reductions), then the packed order on one thread
static __global__ void kvq_fit_bits_k(const float * sigma, const int64_t ss2, const float * par, const int64_t sp1, const int64_t sp2,
        float * out, const int64_t so1, const int64_t so2, const int n, const float avg_bits) {
    extern __shared__ double ideal[]; // [n], then the bits as int [n]
    int * bb = (int *) (ideal + n);

    __shared__ double red[KVQ_BITS_NT/32];
    __shared__ int    sh_best;

    const int     b        = blockIdx.x;
    const int     tid      = threadIdx.x;
    const int     max_bits = 8;
    const float * sg       = sigma + b*ss2;

    const int64_t target = min(max((int64_t) nearbyint((double) avg_bits*n), (int64_t) 0), (int64_t) n*max_bits);
    double lo = DBL_MAX;
    double hi = -DBL_MAX;
    for (int c = tid; c < n; c += KVQ_BITS_NT) {
        const double var = (double) sg[c]*sg[c];
        ideal[c] = 0.5*log2(fmax(var, 1e-30));
        lo = fmin(lo, ideal[c]);
        hi = fmax(hi, ideal[c]);
    }
    lo = kvq_bits_red<1>(lo, red) - 20;
    hi = kvq_bits_red<2>(hi, red) + 20;
    double th = 0.0;
    for (int it = 0; it < 60; it++) {
        th = (lo + hi)/2;
        double s = 0.0;
        for (int c = tid; c < n; c += KVQ_BITS_NT) {
            s += fmin(fmax(ideal[c] - th, 0.0), (double) max_bits);
        }
        if (kvq_bits_red<0>(s, red) > target) {
            lo = th;
        } else {
            hi = th;
        }
    }
    double s = 0.0;
    for (int c = tid; c < n; c += KVQ_BITS_NT) {
        ideal[c] -= th;
        bb[c] = (int) fmin(fmax(nearbyint(ideal[c]), 0.0), (double) max_bits);
        s += bb[c];
    }
    int64_t sum = (int64_t) kvq_bits_red<0>(s, red);
    // one bit per pass; ties go to the lowest index
    while (sum != target) {
        const bool down = sum > target;
        double vb = down ? -DBL_MAX : DBL_MAX;
        int    ib = n;
        for (int c = tid; c < n; c += KVQ_BITS_NT) {
            if (down ? bb[c] > 0 : bb[c] < max_bits) {
                const double dc = bb[c] - ideal[c];
                if (down ? dc > vb : dc < vb) {
                    vb = dc;
                    ib = c;
                }
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const double vo = __shfl_xor_sync(0xffffffff, vb, o);
            const int    io = __shfl_xor_sync(0xffffffff, ib, o);
            if ((down ? vo > vb : vo < vb) || (vo == vb && io < ib)) {
                vb = vo;
                ib = io;
            }
        }
        __shared__ int ibs[KVQ_BITS_NT/32];
        __syncthreads();
        if (tid % 32 == 0) {
            red[tid / 32] = vb;
            ibs[tid / 32] = ib;
        }
        __syncthreads();
        if (tid == 0) {
            double v = red[0];
            int    i = ibs[0];
            for (int w = 1; w < KVQ_BITS_NT/32; w++) {
                if ((down ? red[w] > v : red[w] < v) || (red[w] == v && ibs[w] < i)) {
                    v = red[w];
                    i = ibs[w];
                }
            }
            sh_best = i < n ? i : -1;
            if (i < n) {
                bb[i] += down ? -1 : 1;
            }
        }
        __syncthreads();
        if (sh_best < 0) {
            break;
        }
        sum += down ? -1 : 1;
    }

    float * o = out + b*so2;
    for (int c = tid; c < n; c += KVQ_BITS_NT) {
        o[3*so1 + c] = 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        int off = 0;
        for (int nb = max_bits; nb >= 0; nb--) {
            for (int c = 0; c < n; c++) {
                if (bb[c] != nb) {
                    continue;
                }
                o[c]         = nb ? kvq_fit_gauss_step[nb]*sg[c] : 0.0f;
                o[so1 + c]   = (float) nb;
                o[2*so1 + c] = (float) off;
                off += nb;
            }
        }
        o[3*so1] = par[b*sp2 + 3*sp1];
    }
}

// a block per task, a warp per packed position: the int8 inverse table (inv != nullptr) or tab_aux
static __global__ void kvq_fit_pack_k(const float * x, const int64_t sx1, const int64_t sx2, const float * par, const int64_t sp1, const int64_t sp2,
        int8_t * inv, const int64_t si1, const int64_t si2, float * aux, const int64_t sa1, const int64_t sa2, const int n) {
    extern __shared__ int pperm[]; // [n] packed position -> component

    const int     b    = blockIdx.x;
    const int     tid  = threadIdx.x;
    const float * pb   = par + b*sp2;
    const bool    is_v = pb[3*sp1] != 0.0f;

    if (tid == 0) {
        int j = 0;
        for (int nb = 8; nb >= 0; nb--) {
            for (int c = 0; c < n; c++) {
                if ((int) pb[sp1 + c] == nb) {
                    pperm[j++] = c;
                }
            }
        }
    }
    __syncthreads();

    for (int c = tid / 32; c < n; c += blockDim.x/32) {
        const int     pc   = pperm[c];
        const float   bits = pb[sp1 + pc];
        const float * row  = x + b*sx2 + (int64_t) pc*sx1;
        float m = 0.0f;
        for (int i = tid % 32; i < n && bits != 0.0f; i += 32) {
            m = fmaxf(m, fabsf(row[i]));
        }
        for (int o = 16; o > 0; o >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        }
        const float s = __fdiv_rn(m, 127.0f);
        if (inv) {
            for (int i = tid % 32; i < n; i += 32) {
                const int8_t q = s > 0.0f ? (int8_t) fminf(fmaxf(rintf(__fdiv_rn(row[i], s)), -127.0f), 127.0f) : 0;
                inv[b*si2 + (is_v ? (int64_t) c*si1 + i : (int64_t) i*si1 + c)] = q;
            }
        } else if (tid % 32 == 0) {
            aux[b*sa2 + c]         = bits != 0.0f ? pb[pc]*s : 0.0f;
            aux[b*sa2 + 2*sa1 + c] = bits;
        }
    }
    if (!inv) {
        for (int i = tid; i < n; i += blockDim.x) {
            aux[b*sa2 + sa1 + i] = x[b*sx2 + (int64_t) n*sx1 + i];
        }
    }
}

void ggml_cuda_op_kvq_fit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * par = dst->src[1];

    const int mode = ggml_get_op_params_i32(dst, 0);

    switch (mode) {
        case GGML_KVQ_FIT_TABLE: {
            kvq_fit_table(ctx, dst);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        case GGML_KVQ_FIT_BITS: {
            const int n = (int) x->ne[0];
            const int B = (int) x->ne[2];
            kvq_fit_bits_k<<<B, KVQ_BITS_NT, n*(sizeof(double) + sizeof(int)), ctx.stream()>>>((const float *) x->data, x->nb[2]/sizeof(float),
                    (const float *) par->data, par->nb[1]/sizeof(float), par->nb[2]/sizeof(float),
                    (float *) dst->data, dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), n, ggml_get_op_params_f32(dst, 1));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        case GGML_KVQ_FIT_PACK_INV:
        case GGML_KVQ_FIT_PACK_AUX: {
            const int  n   = (int) x->ne[0];
            const int  B   = (int) x->ne[2];
            const bool pin = mode == GGML_KVQ_FIT_PACK_INV;
            kvq_fit_pack_k<<<B, 256, n*sizeof(int), ctx.stream()>>>((const float *) x->data, x->nb[1]/sizeof(float), x->nb[2]/sizeof(float),
                    (const float *) par->data, par->nb[1]/sizeof(float), par->nb[2]/sizeof(float),
                    pin ? (int8_t *) dst->data : nullptr, pin ? dst->nb[1] : 0, pin ? dst->nb[2] : 0,
                    pin ? nullptr : (float *) dst->data, pin ? 0 : dst->nb[1]/sizeof(float), pin ? 0 : dst->nb[2]/sizeof(float), n);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        default:
            break;
    }

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
    // TABLE: a thread per column (256 per block); BITS and PACK hold D values per task
    if ((mode == GGML_KVQ_FIT_TABLE && x->ne[0] > 256) || (mode > GGML_KVQ_FIT_TABLE && x->ne[0] > 1024)) {
        return false;
    }
    return true;
}
