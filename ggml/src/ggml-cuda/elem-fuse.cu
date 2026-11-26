#include "elem-fuse.cuh"
#include "reduce_rows.cuh"

#include <cstring>

// ---------------------------------------------------------------------------------------------------------------------
// y = c -/+ tanh(a)*b
// each value is rounded as in the separate TANH (tanhf), MUL and SUB/ADD kernels: the product and the sum are not
// contracted into an FMA

template <bool sub>
static __global__ void tanh_mul_addsub_f32(
        const char * a_ptr, const char * b_ptr, const char * c_ptr, float * dst_ptr,
        const int ne0, const int ne1, const int ne2,
        const int64_t a1, const int64_t a2, const int64_t a3,
        const int64_t b1, const int64_t b2, const int64_t b3,
        const int64_t c1, const int64_t c2, const int64_t c3) {
    const char * GGML_CUDA_RESTRICT a   = a_ptr;
    const char * GGML_CUDA_RESTRICT b   = b_ptr;
    const char * GGML_CUDA_RESTRICT c   = c_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z % ne2;
    const int i3 = blockIdx.z / ne2;

    ggml_cuda_pdl_sync();
    if (i0 >= ne0) {
        return;
    }

    const float av = *(const float *) (a + i1*a1 + i2*a2 + i3*a3 + i0*sizeof(float));
    const float bv = *(const float *) (b + i1*b1 + i2*b2 + i3*b3 + i0*sizeof(float));
    const float cv = *(const float *) (c + i1*c1 + i2*c2 + i3*c3 + i0*sizeof(float));

    const float m = __fmul_rn(tanhf(av), bv);

    dst[((int64_t(i3)*ne2 + i2)*ne1 + i1)*ne0 + i0] = sub ? __fsub_rn(cv, m) : __fadd_rn(cv, m);
}

bool ggml_cuda_elem_overlap_ok(const ggml_tensor * dst, const ggml_tensor * src) {
    const char * d0 = (const char *) dst->data;
    const char * d1 = d0 + ggml_nbytes(dst);
    const char * s0 = (const char *) src->data;
    const char * s1 = s0 + ggml_nbytes(src);
    if (d1 <= s0 || s1 <= d0) {
        return true;
    }
    return d0 == s0 && ggml_is_contiguous(dst) && ggml_is_contiguous(src) && ggml_nelements(dst) == ggml_nelements(src) &&
        dst->type == src->type;
}

static bool elem_is_view_op(const ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool elem_f32_rows_ok(const ggml_tensor * t, const ggml_tensor * shape) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ggml_are_same_shape(t, shape);
}

int ggml_cuda_match_tanh_mul_addsub(const ggml_cgraph * cgraph, int i, ggml_cuda_tanh_mul_addsub_match & match) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    // the views of b (no kernels) may sit between TANH and MUL
    int j = i + 1;
    while (j < i + 3 && j < cgraph->n_nodes && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * t = cgraph->nodes[i];
    const ggml_tensor * m = cgraph->nodes[j];
    ggml_tensor       * y = cgraph->nodes[j + 1];
    if (t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_TANH || m->op != GGML_OP_MUL ||
        (y->op != GGML_OP_SUB && y->op != GGML_OP_ADD)) {
        return 0;
    }

    const ggml_tensor * b = nullptr;
    if (m->src[0] == t) {
        b = m->src[1];
    } else if (m->src[1] == t) {
        b = m->src[0];
    } else {
        return 0;
    }

    const ggml_tensor * c = nullptr;
    bool sub = false;
    if (y->op == GGML_OP_SUB) {
        if (y->src[1] != m || y->src[0] == m) {
            return 0;
        }
        c   = y->src[0];
        sub = true;
    } else {
        if (y->src[0] == m && y->src[1] != m) {
            c = y->src[1];
        } else if (y->src[1] == m && y->src[0] != m) {
            c = y->src[0];
        } else {
            return 0;
        }
    }

    const ggml_tensor * a = t->src[0];
    if (!elem_f32_rows_ok(a, y) || !elem_f32_rows_ok(b, y) || !elem_f32_rows_ok(c, y) || !elem_f32_rows_ok(t, y) ||
        !elem_f32_rows_ok(m, y) || !ggml_is_contiguous(y) || !ggml_is_contiguous(t) || b == t || c == t) {
        return 0;
    }
    if (y->ne[1] > 65535 || y->ne[2]*y->ne[3] > 65535 || y->ne[0] > INT_MAX) {
        return 0;
    }

    // the skipped views must not be views of an elided node
    for (int k = i + 1; k < j; ++k) {
        if (cgraph->nodes[k]->view_src == t || cgraph->nodes[k]->view_src == m) {
            return 0;
        }
    }

    const ggml_op ops[3]  = { GGML_OP_UNARY, GGML_OP_MUL, y->op };
    const int     idxs[3] = { i, j, j + 1 };
    const int     outs[1] = { j + 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 3, ops, outs, 1)) {
        return 0;
    }

    if (!ggml_cuda_elem_overlap_ok(y, a) || !ggml_cuda_elem_overlap_ok(y, b) || !ggml_cuda_elem_overlap_ok(y, c)) {
        return 0;
    }

    match.a   = a;
    match.b   = b;
    match.c   = c;
    match.dst = y;
    match.sub = sub;
    return j + 2 - i;
}

void ggml_cuda_op_tanh_mul_addsub(ggml_backend_cuda_context & ctx, const ggml_cuda_tanh_mul_addsub_match & match) {
    const ggml_tensor * a = match.a;
    const ggml_tensor * b = match.b;
    const ggml_tensor * c = match.c;
    ggml_tensor       * y = match.dst;

    const int block = 256;
    const dim3 grid((y->ne[0] + block - 1)/block, y->ne[1], y->ne[2]*y->ne[3]);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, dim3(block, 1, 1), 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            (const char *) a->data, (const char *) b->data, (const char *) c->data, (float *) y->data,
            (int) y->ne[0], (int) y->ne[1], (int) y->ne[2],
            (int64_t) a->nb[1], (int64_t) a->nb[2], (int64_t) a->nb[3],
            (int64_t) b->nb[1], (int64_t) b->nb[2], (int64_t) b->nb[3],
            (int64_t) c->nb[1], (int64_t) c->nb[2], (int64_t) c->nb[3]);
    };
    if (match.sub) {
        launch(tanh_mul_addsub_f32<true>);
    } else {
        launch(tanh_mul_addsub_f32<false>);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// x / f(sum_rows(x)) [* scale + bias], f = SCALE (x*s + b) or CLAMP
// one block per row with the block size of sum_rows_f32_cuda, so the row sum is reduce_row_f32's for the same shape;
// SCALE as scale_f32's FMA, the division as op_div's a / b

template <bool clamp, bool has_scale>
static __global__ void row_norm_div_f32(const float * x_ptr, float * dst_ptr, const int ncols,
        const float p0, const float p1, const float out_scale, const float out_bias) {
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int64_t row = blockIdx.x;

    const float * xr = x + row*ncols;
    const float sum = reduce_row_f32(xr, ncols); // waits for the previous kernel
    ggml_cuda_pdl_lc();

    const float d = clamp ? fminf(fmaxf(sum, p0), p1) : __fmaf_rn(p0, sum, p1);

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float v = xr[col] / d;
        dst[row*ncols + col] = has_scale ? __fmaf_rn(out_scale, v, out_bias) : v;
    }
}

int ggml_cuda_match_row_norm_div(const ggml_cgraph * cgraph, int i, ggml_cuda_row_norm_div_match & match) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sum = cgraph->nodes[i];
    const ggml_tensor * f   = cgraph->nodes[i + 1];
    const ggml_tensor * div = cgraph->nodes[i + 2];
    if (sum->op != GGML_OP_SUM_ROWS || (f->op != GGML_OP_SCALE && f->op != GGML_OP_CLAMP) || div->op != GGML_OP_DIV) {
        return 0;
    }
    const ggml_tensor * x = sum->src[0];
    if (f->src[0] != sum || div->src[0] != x || div->src[1] != f) {
        return 0;
    }
    if (x->type != GGML_TYPE_F32 || sum->type != GGML_TYPE_F32 || f->type != GGML_TYPE_F32 || div->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(div) || !ggml_are_same_shape(x, div) || x->ne[0] > INT_MAX ||
        ggml_nrows(x) > INT_MAX || !ggml_is_contiguous(sum) || !ggml_is_contiguous(f) || ggml_nelements(sum) != ggml_nrows(x)) {
        return 0;
    }

    int n = 3;
    ggml_op ops[5] = { GGML_OP_SUM_ROWS, f->op, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE };
    ggml_tensor * out = cgraph->nodes[i + 2];
    bool has_scale = false;
    if (i + 4 < cgraph->n_nodes) {
        const ggml_tensor * r = cgraph->nodes[i + 3];
        ggml_tensor       * s = cgraph->nodes[i + 4];
        if (r->op == GGML_OP_RESHAPE && r->src[0] == div && s->op == GGML_OP_SCALE && s->src[0] == r &&
            s->type == GGML_TYPE_F32 && ggml_is_contiguous(s) && ggml_nelements(s) == ggml_nelements(div)) {
            n = 5;
            out = s;
            has_scale = true;
        }
    }

    const int outs[1] = { i + n - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs, 1)) {
        if (n == 5) {
            // the plain division alone
            n = 3;
            out = cgraph->nodes[i + 2];
            has_scale = false;
            const int outs3[1] = { i + 2 };
            if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs3, 1)) {
                return 0;
            }
        } else {
            return 0;
        }
    }

    // each thread reads x only at the columns it writes, after the row sum
    if (!ggml_cuda_elem_overlap_ok(out, x)) {
        return 0;
    }

    match.x     = x;
    match.dst   = out;
    match.clamp = f->op == GGML_OP_CLAMP;
    memcpy(&match.p0, (const float *) f->op_params + 0, sizeof(float));
    memcpy(&match.p1, (const float *) f->op_params + 1, sizeof(float));
    match.has_scale = has_scale;
    if (has_scale) {
        memcpy(&match.out_scale, (const float *) out->op_params + 0, sizeof(float));
        memcpy(&match.out_bias,  (const float *) out->op_params + 1, sizeof(float));
    }
    return n;
}

void ggml_cuda_op_row_norm_div(ggml_backend_cuda_context & ctx, const ggml_cuda_row_norm_div_match & match) {
    const ggml_tensor * x = match.x;
    const int     ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    // the block size of sum_rows_f32_cuda for this shape
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int block_size = (nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128);

    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, (float *) match.dst->data, ncols,
            match.p0, match.p1, match.out_scale, match.out_bias);
    };
    if (match.clamp) {
        match.has_scale ? launch(row_norm_div_f32<true, true>) : launch(row_norm_div_f32<true, false>);
    } else {
        match.has_scale ? launch(row_norm_div_f32<false, true>) : launch(row_norm_div_f32<false, false>);
    }
}

// =================================================================================================
// Inkling-2 looped-layer control fusion:
//   gate = W_down @ concat(cur_norm, msg_norm, dim=0)  [R, T]
//   dst  = cur - tanh(W_forget @ gate) * cur + tanh(W_gain @ gate) * msg  [D, T]
// Eliminates CONCAT + 3x small F32 cuBLAS GEMMs + tanh_mul_addsub -> 2 fused bandwidth-optimal kernels.
// =================================================================================================

static bool loop_ctrl_vec_layout_ok(const ggml_tensor * t, int64_t D, int64_t T) {
    return t != nullptr && t->type == GGML_TYPE_F32 &&
           t->ne[0] == D && t->ne[1] == T && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->nb[0] == sizeof(float) && (t->nb[1] % sizeof(float)) == 0;
}

int ggml_cuda_match_loop_control(const ggml_cgraph * cgraph, int i, ggml_cuda_loop_control_match & match) {
    if (i + 9 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * n_cat = cgraph->nodes[i];
    if (n_cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(n_cat, 0) != 0) {
        return 0;
    }
    if (cgraph->nodes[i + 1]->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const ggml_tensor * cur_norm = n_cat->src[0];
    const ggml_tensor * msg_norm = n_cat->src[1];
    if (!cur_norm || !msg_norm || cur_norm->type != GGML_TYPE_F32 || msg_norm->type != GGML_TYPE_F32) {
        return 0;
    }
    const int64_t D = cur_norm->ne[0];
    const int64_t T = cur_norm->ne[1];
    if (D <= 0 || (D % 4) != 0 || T < 1 || T > 32 ||
        !loop_ctrl_vec_layout_ok(cur_norm, D, T) || !loop_ctrl_vec_layout_ok(msg_norm, D, T)) {
        return 0;
    }

    ggml_tensor * n_down = cgraph->nodes[i + 1];
    if (n_down->op != GGML_OP_MUL_MAT || n_down->src[1] != n_cat || n_down->type != GGML_TYPE_F32 || !ggml_is_contiguous(n_down)) {
        return 0;
    }
    const ggml_tensor * w_down = n_down->src[0];
    if (!w_down || w_down->type != GGML_TYPE_F32 || !ggml_is_contiguous(w_down) ||
        w_down->ne[0] != 2 * D || w_down->ne[2] != 1 || w_down->ne[3] != 1) {
        return 0;
    }
    const int64_t R = w_down->ne[1];
    if (R < 1 || R > 128) {
        return 0;
    }

    // Match the forget branch: [VIEW*], MUL_MAT(w_forget, n_down), UNARY(TANH), [VIEW*], MUL(tanh, cur), SUB(cur, mul)
    int j = i + 2;
    while (j < cgraph->n_nodes && j < i + 5 && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j >= cgraph->n_nodes) return 0;
    const int idx_fmm = j;
    const ggml_tensor * n_fmm = cgraph->nodes[j++];
    if (n_fmm->op != GGML_OP_MUL_MAT || n_fmm->src[1] != n_down) {
        return 0;
    }
    const ggml_tensor * w_forget = n_fmm->src[0];
    if (!w_forget || w_forget->type != GGML_TYPE_F32 || !ggml_is_contiguous(w_forget) ||
        w_forget->ne[0] != R || w_forget->ne[1] != D || w_forget->ne[2] != 1 || w_forget->ne[3] != 1) {
        return 0;
    }
    if (j >= cgraph->n_nodes) return 0;
    const int idx_ftanh = j;
    const ggml_tensor * n_ftanh = cgraph->nodes[j++];
    if (n_ftanh->op != GGML_OP_UNARY || ggml_get_unary_op(n_ftanh) != GGML_UNARY_OP_TANH || n_ftanh->src[0] != n_fmm) {
        return 0;
    }
    while (j < cgraph->n_nodes && j < i + 9 && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) return 0;
    const int idx_fmul = j;
    const ggml_tensor * n_fmul = cgraph->nodes[j++];
    const int idx_sub  = j;
    const ggml_tensor * n_sub  = cgraph->nodes[j++];
    if (n_fmul->op != GGML_OP_MUL || n_sub->op != GGML_OP_SUB) {
        return 0;
    }
    const ggml_tensor * cur = nullptr;
    if (n_fmul->src[0] == n_ftanh) {
        cur = n_fmul->src[1];
    } else if (n_fmul->src[1] == n_ftanh) {
        cur = n_fmul->src[0];
    } else {
        return 0;
    }
    if (n_sub->src[0] != cur || n_sub->src[1] != n_fmul || !loop_ctrl_vec_layout_ok(cur, D, T)) {
        return 0;
    }

    // Match the gain branch: [VIEW*], MUL_MAT(w_gain, n_down), UNARY(TANH), [VIEW*], MUL(tanh, msg), ADD(n_sub, mul)
    while (j < cgraph->n_nodes && j < i + 12 && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j + 3 >= cgraph->n_nodes) return 0;
    const int idx_gmm = j;
    const ggml_tensor * n_gmm = cgraph->nodes[j++];
    if (n_gmm->op != GGML_OP_MUL_MAT || n_gmm->src[1] != n_down) {
        return 0;
    }
    const ggml_tensor * w_gain = n_gmm->src[0];
    if (!w_gain || w_gain->type != GGML_TYPE_F32 || !ggml_is_contiguous(w_gain) ||
        w_gain->ne[0] != R || w_gain->ne[1] != D || w_gain->ne[2] != 1 || w_gain->ne[3] != 1) {
        return 0;
    }
    const int idx_gtanh = j;
    const ggml_tensor * n_gtanh = cgraph->nodes[j++];
    if (n_gtanh->op != GGML_OP_UNARY || ggml_get_unary_op(n_gtanh) != GGML_UNARY_OP_TANH || n_gtanh->src[0] != n_gmm) {
        return 0;
    }
    while (j < cgraph->n_nodes && j < i + 16 && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) return 0;
    const int idx_gmul = j;
    const ggml_tensor * n_gmul = cgraph->nodes[j++];
    const int idx_add  = j;
    ggml_tensor       * n_add  = cgraph->nodes[j++];
    if (n_gmul->op != GGML_OP_MUL || n_add->op != GGML_OP_ADD) {
        return 0;
    }
    const ggml_tensor * msg = nullptr;
    if (n_gmul->src[0] == n_gtanh) {
        msg = n_gmul->src[1];
    } else if (n_gmul->src[1] == n_gtanh) {
        msg = n_gmul->src[0];
    } else {
        return 0;
    }
    const bool add_ok = (n_add->src[0] == n_sub && n_add->src[1] == n_gmul) ||
                        (n_add->src[1] == n_sub && n_add->src[0] == n_gmul);
    if (!add_ok || !loop_ctrl_vec_layout_ok(msg, D, T) || !loop_ctrl_vec_layout_ok(n_add, D, T) || !ggml_is_contiguous(n_add)) {
        return 0;
    }
    if (!ggml_cuda_elem_overlap_ok(n_add, cur) || !ggml_cuda_elem_overlap_ok(n_add, msg)) {
        return 0;
    }

    const int n = j - i;
    const ggml_op ops[10] = {
        GGML_OP_CONCAT, GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB,
        GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD,
    };
    const int idxs[10] = { i, i + 1, idx_fmm, idx_ftanh, idx_fmul, idx_sub, idx_gmm, idx_gtanh, idx_gmul, idx_add };
    const int outs[1]  = { idx_add };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 10, ops, outs, 1)) {
        return 0;
    }

    match.cur_norm = cur_norm;
    match.msg_norm = msg_norm;
    match.cur      = cur;
    match.msg      = msg;
    match.w_down   = w_down;
    match.w_forget = w_forget;
    match.w_gain   = w_gain;
    match.gate     = n_down;
    match.dst      = n_add;
    return n;
}

template <int TILE_T>
static __global__ void loop_ctrl_down_f32(
        const float * cur_norm_ptr,
        const float * msg_norm_ptr,
        const float * w_down_ptr,
        float       * gate_ptr,
        const int D, const int R, const int T,
        const int64_t s_cn, const int64_t s_mn) {
    const float * GGML_CUDA_RESTRICT cur_norm = cur_norm_ptr;
    const float * GGML_CUDA_RESTRICT msg_norm = msg_norm_ptr;
    const float * GGML_CUDA_RESTRICT w_down   = w_down_ptr;
    float       * GGML_CUDA_RESTRICT gate     = gate_ptr;
    constexpr int NTHREADS = 128;
    constexpr int NWARPS   = NTHREADS / WARP_SIZE;

    const int r  = blockIdx.x;
    const int t0 = blockIdx.y * TILE_T;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const float * wd_a = w_down + (int64_t) r * (2 * D);
    const float * wd_b = wd_a + D;

    float acc[TILE_T];
#pragma unroll
    for (int it = 0; it < TILE_T; ++it) {
        acc[it] = 0.0f;
    }

    const int D4 = D / 4;
    const float4 * wd_a4 = (const float4 *) wd_a;
    const float4 * wd_b4 = (const float4 *) wd_b;
    const bool aligned_w  = ((uintptr_t) w_down % 16 == 0);
    const bool aligned_cn = ((uintptr_t) cur_norm % 16 == 0) && ((s_cn % 4) == 0);
    const bool aligned_mn = ((uintptr_t) msg_norm % 16 == 0) && ((s_mn % 4) == 0);

    ggml_cuda_pdl_sync();

    if (aligned_w && aligned_cn && aligned_mn) {
        for (int k4 = tid; k4 < D4; k4 += NTHREADS) {
            const float4 wa = wd_a4[k4];
            const float4 wb = wd_b4[k4];
#pragma unroll
            for (int it = 0; it < TILE_T; ++it) {
                const int t = t0 + it;
                if (t < T) {
                    const float4 xa = ((const float4 *) (cur_norm + t * s_cn))[k4];
                    const float4 xb = ((const float4 *) (msg_norm + t * s_mn))[k4];
                    acc[it] += wa.x*xa.x + wa.y*xa.y + wa.z*xa.z + wa.w*xa.w +
                               wb.x*xb.x + wb.y*xb.y + wb.z*xb.z + wb.w*xb.w;
                }
            }
        }
    } else {
        for (int k = tid; k < D; k += NTHREADS) {
            const float wa = wd_a[k];
            const float wb = wd_b[k];
#pragma unroll
            for (int it = 0; it < TILE_T; ++it) {
                const int t = t0 + it;
                if (t < T) {
                    acc[it] += wa * cur_norm[t * s_cn + k] + wb * msg_norm[t * s_mn + k];
                }
            }
        }
    }

    __shared__ float smem[TILE_T][NWARPS];
#pragma unroll
    for (int it = 0; it < TILE_T; ++it) {
        const float ws = warp_reduce_sum(acc[it]);
        if (lane == 0) {
            smem[it][warp] = ws;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < TILE_T) {
        const int t = t0 + lane;
        if (t < T) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                sum += smem[lane][w];
            }
            gate[t * R + r] = sum;
        }
    }
    ggml_cuda_pdl_lc();
}

template <int TILE_T>
static __global__ void loop_ctrl_up_apply_f32(
        const float * gate_ptr,
        const float * w_forget_ptr,
        const float * w_gain_ptr,
        const float * cur_ptr,
        const float * msg_ptr,
        float       * dst_ptr,
        const int D, const int R, const int T,
        const int64_t s_cur, const int64_t s_msg, const int64_t s_dst) {
    const float * GGML_CUDA_RESTRICT gate     = gate_ptr;
    const float * GGML_CUDA_RESTRICT w_forget = w_forget_ptr;
    const float * GGML_CUDA_RESTRICT w_gain   = w_gain_ptr;
    const float * GGML_CUDA_RESTRICT cur      = cur_ptr;
    const float * GGML_CUDA_RESTRICT msg      = msg_ptr;
    float       * GGML_CUDA_RESTRICT dst      = dst_ptr;
    constexpr int NWARPS = 4;
    const int d    = blockIdx.x * NWARPS + threadIdx.y;
    const int t0   = blockIdx.y * TILE_T;
    const int lane = threadIdx.x;
    const int tid  = threadIdx.y * WARP_SIZE + lane;

    ggml_cuda_pdl_sync();

    __shared__ float s_gate[TILE_T][128];
    const int total_g = TILE_T * R;
    for (int idx = tid; idx < total_g; idx += NWARPS * WARP_SIZE) {
        const int it = idx / R;
        const int r  = idx % R;
        const int t  = t0 + it;
        s_gate[it][r] = (t < T) ? gate[t * R + r] : 0.0f;
    }
    __syncthreads();

    if (d >= D) {
        return;
    }

    const float * wf_row = w_forget + (int64_t) d * R;
    const float * wg_row = w_gain   + (int64_t) d * R;

    float f_acc[TILE_T];
    float g_acc[TILE_T];
#pragma unroll
    for (int it = 0; it < TILE_T; ++it) {
        f_acc[it] = 0.0f;
        g_acc[it] = 0.0f;
    }

    for (int r = lane; r < R; r += WARP_SIZE) {
        const float wf = wf_row[r];
        const float wg = wg_row[r];
#pragma unroll
        for (int it = 0; it < TILE_T; ++it) {
            const float gv = s_gate[it][r];
            f_acc[it] += wf * gv;
            g_acc[it] += wg * gv;
        }
    }

#pragma unroll
    for (int it = 0; it < TILE_T; ++it) {
        const float f = warp_reduce_sum(f_acc[it]);
        const float g = warp_reduce_sum(g_acc[it]);
        const int t = t0 + it;
        if (lane == 0 && t < T) {
            const float c = cur[t * s_cur + d];
            const float m = msg[t * s_msg + d];
            dst[t * s_dst + d] = c - tanhf(f) * c + tanhf(g) * m;
        }
    }
    ggml_cuda_pdl_lc();
}

void ggml_cuda_op_loop_control(ggml_backend_cuda_context & ctx, const ggml_cuda_loop_control_match & match) {
    const int D = (int) match.cur->ne[0];
    const int T = (int) match.cur->ne[1];
    const int R = (int) match.w_down->ne[1];

    const int64_t s_cn  = match.cur_norm->nb[1] / sizeof(float);
    const int64_t s_mn  = match.msg_norm->nb[1] / sizeof(float);
    const int64_t s_cur = match.cur->nb[1]      / sizeof(float);
    const int64_t s_msg = match.msg->nb[1]      / sizeof(float);
    const int64_t s_dst = match.dst->nb[1]      / sizeof(float);

    // Reuse graph-allocated n_down->data [R, T] instead of ctx.pool() so captured CUDA graphs stay pointer-stable.
    float * gate = (float *) match.gate->data;
    cudaStream_t stream = ctx.stream();

    if (T <= 8) {
        constexpr int TILE_T = 1;
        ggml_cuda_kernel_launch(loop_ctrl_down_f32<TILE_T>,
            ggml_cuda_kernel_launch_params(dim3(R, T, 1), dim3(128, 1, 1), 0, stream),
            (const float *) match.cur_norm->data, (const float *) match.msg_norm->data,
            (const float *) match.w_down->data, gate, D, R, T, s_cn, s_mn);
        ggml_cuda_kernel_launch(loop_ctrl_up_apply_f32<TILE_T>,
            ggml_cuda_kernel_launch_params(dim3((D + 3) / 4, T, 1), dim3(WARP_SIZE, 4, 1), 0, stream),
            gate, (const float *) match.w_forget->data, (const float *) match.w_gain->data,
            (const float *) match.cur->data, (const float *) match.msg->data, (float *) match.dst->data,
            D, R, T, s_cur, s_msg, s_dst);
    } else {
        constexpr int TILE_T = 4;
        const int nty = (T + TILE_T - 1) / TILE_T;
        ggml_cuda_kernel_launch(loop_ctrl_down_f32<TILE_T>,
            ggml_cuda_kernel_launch_params(dim3(R, nty, 1), dim3(128, 1, 1), 0, stream),
            (const float *) match.cur_norm->data, (const float *) match.msg_norm->data,
            (const float *) match.w_down->data, gate, D, R, T, s_cn, s_mn);
        ggml_cuda_kernel_launch(loop_ctrl_up_apply_f32<TILE_T>,
            ggml_cuda_kernel_launch_params(dim3((D + 3) / 4, nty, 1), dim3(WARP_SIZE, 4, 1), 0, stream),
            gate, (const float *) match.w_forget->data, (const float *) match.w_gain->data,
            (const float *) match.cur->data, (const float *) match.msg->data, (float *) match.dst->data,
            D, R, T, s_cur, s_msg, s_dst);
    }
}
