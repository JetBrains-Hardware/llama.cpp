#include "llama-kvq-fit.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-kvq-store.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <unordered_map>
#include <stdexcept>

static constexpr int64_t KVQ_D   = llama_kvq_store::D;
static constexpr size_t  KVQ_ROW = llama_kvq_store::ROW;

static double kvq_fit_now() {
    return ggml_time_us()*1e-6;
}

// task i = kv*n_pair + layer index*G + kv head, kv = 0 for K, 1 for V; the host arrays are per task
struct llama_kvq_fitter::job {
    int32_t r0 = 0; // rows [r0, r1), the table set of the chunk slots r0/C .. (r1 - 1)/C
    int32_t r1 = 0;

    int32_t n() const { return r1 - r0; }

    std::vector<float> gram; // [n_layer][G][D*D]
    std::vector<float> gsum; // [n_layer][n_gate], per-chunk V only

    // pool mutex
    bool done   = false; // the device work and its readbacks are done
    bool cancel = false;

    double t_snap = 0.0;
    double t_take = 0.0;

    // after the readbacks of the last layer
    ggml_backend_event_ptr ev;

    // pinned host memory
    ggml_backend_buffer_ptr buf;
    float  * hmet  = nullptr; // per layer index [2*G][D*D]: the metrics, K then V
    float  * hres  = nullptr; // per layer index [2*G][D]: the result row of each table
    int8_t * htinv = nullptr; // per task [D*D]
    float  * htaux = nullptr; // per task [3*D]
};

int32_t llama_kvq_fitter::stage_rows(int32_t n_rows, const llama_kvq_fit_params & fparams) {
    // merge: a group of the last complete chunk and up to C - 1 rows after it
    return std::min<int32_t>(n_rows, fparams.rem == LLAMA_KVQ_REM_MERGE ? 2*fparams.chunk - 1 : fparams.chunk);
}

size_t llama_kvq_fitter::shared_min(int32_t n_head_kv, int32_t n_stage, int32_t bits_k, int32_t bits_v) {
    const size_t G     = n_head_kv;
    const size_t rows  = (size_t) n_stage*G;
    const size_t words = (size_t) (bits_k*KVQ_D + 31)/32 + (bits_v*KVQ_D + 31)/32 + 1;
    const size_t fixed = rows*2*KVQ_ROW + rows*words*sizeof(int32_t);
    const size_t tab1  = 2*G*(2*KVQ_D + 3)*KVQ_D*sizeof(float);
    // a graph of one layer holds at most three f32 copies of the rows of one of K and V
    const size_t inter = 3*rows*KVQ_D*sizeof(float);
    return fixed + tab1 + std::max(inter, ggml_kvq_fit_table_work_size(KVQ_D, 2*G)) + (8u << 20);
}

llama_kvq_fitter::llama_kvq_fitter(llama_kvq_store & store, const llama_kvq_fit_params & fparams,
        const std::vector<const ggml_tensor *> & wo, int32_t n_head, int32_t min_first) :
    store(store), fparams(fparams), min_first(min_first), layer_ids(store.layer_ids()), n_head(n_head), wo(wo) {
    const int32_t G     = store.n_head_kv();
    const int32_t C     = store.n_chunk_rows();
    const int32_t n_chk = store.n_chunks();

    if (store.backend() == nullptr || store.shared_buffer() == nullptr) {
        throw std::runtime_error("the fitter needs the backend of the device and the shared memory of the store");
    }
    if (fparams.rem != LLAMA_KVQ_REM_OWN && fparams.rem != LLAMA_KVQ_REM_MERGE) {
        throw std::runtime_error(format("unknown policy %d for the rows after the last complete chunk", fparams.rem));
    }
    // the bits of a row: round(bits*D), ggml_kvq_fit_bits
    for (int kv = 0; kv < 2; ++kv) {
        const int64_t nb = (int64_t) std::nearbyint((double) (kv ? fparams.bits_v : fparams.bits_k)*KVQ_D);
        if (nb < 0 || nb > 8*KVQ_D || (nb + 31)/32 > store.n_words(kv == 1)) {
            throw std::runtime_error(format("%d bits per value of %s do not fit the %d words of a packed row", kv ? fparams.bits_v : fparams.bits_k,
                    kv ? "V" : "K", store.n_words(kv == 1)));
        }
    }
    n_stage = stage_rows(store.n_rows(), fparams);
    slot_fit.assign(n_chk, range());

    n_rep = n_head/G;
    if (n_rep < 1 || n_rep*G != n_head || wo.size() != layer_ids.size()) {
        throw std::runtime_error("per-chunk V needs the o_proj weight of every attention layer");
    }
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        const ggml_tensor * w  = wo[li];
        const ggml_tensor * ga = store.get_layer(layer_ids[li]).gacc;
        if (w == nullptr || ga == nullptr || ggml_n_dims(w) != 2 || w->ne[0] != (int64_t) n_head*KVQ_D || ga->ne[0] != w->ne[0] ||
                (w->type != GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float == nullptr) ||
                (n_rep*KVQ_D) % ggml_blck_size(w->type) != 0) {
            throw std::runtime_error(format("per-chunk V: unsupported o_proj weight or no gate statistics in layer %d", layer_ids[li]));
        }
    }
    wo_gram.resize(layer_ids.size()*G);

    graph_meta.resize(ggml_tensor_overhead()*128 + ggml_graph_overhead());
    dev_init();
    std::vector<char> on_dev;
    compute_wo_gram_dev(on_dev);
    int n_host = 0;
    const double t0 = kvq_fit_now();
    for (int32_t p = 0; p < (int32_t) on_dev.size(); ++p) {
        if (!on_dev[p]) {
            compute_wo_gram(p);
            n_host++;
        }
    }
    if (n_host > 0) {
        LLAMA_LOG_INFO("%s: o_proj grams of %d (layer, kv head) on the CPU in %.2f s\n", __func__, n_host, kvq_fit_now() - t0);
    }

    waiter = std::thread([this]() { loop(); });

    const std::string rem = fparams.rem == LLAMA_KVQ_REM_MERGE ? "merged into the last complete chunk" : format("own chunk of at least %d rows", fparams.min_rows);
    LLAMA_LOG_INFO("%s: chunks of %d rows, up to %d chunks, K %d bits and V %d bits fitted per chunk on the device, rows after the last complete chunk: %s\n", __func__,
            C, n_chk, fparams.bits_k, fparams.bits_v, rem.c_str());
}

llama_kvq_fitter::~llama_kvq_fitter() {
    {
        std::lock_guard<std::mutex> lk(m);
        stop = true;
    }
    cv_work.notify_all();
    waiter.join();
    // queued graphs use the buffers; the backend may be freed already, the events are not
    for (auto & jb : jobs) {
        ggml_backend_event_synchronize(jb->ev.get());
    }
}

// places the tensors of gf without memory (not views) in the shared memory from lo: first fit, a tensor is free after its
// last reader (outputs stay to the end); place = false: only the size. Returns the end of the used memory, 0 if it does not fit
size_t llama_kvq_fitter::plan(ggml_cgraph * gf, size_t lo, bool place) {
    ggml_backend_buffer_t buf  = store.shared_buffer();
    const size_t          size = place ? ggml_backend_buffer_get_size(buf) : SIZE_MAX/2;
    const size_t          al   = 256;

    auto owner = [](ggml_tensor * t) {
        while (t->view_src) {
            t = t->view_src;
        }
        return t;
    };
    const int n = ggml_graph_n_nodes(gf);
    std::unordered_map<const ggml_tensor *, int> last;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s]) {
                last[owner(node->src[s])] = i;
            }
        }
    }

    std::map<size_t, size_t> free_mem = { { lo, size - lo } }; // offset -> bytes
    std::vector<std::pair<ggml_tensor *, size_t>> at;
    std::unordered_map<const ggml_tensor *, std::pair<size_t, size_t>> where;
    size_t hi = lo;
    bool ok = true;
    auto take = [&](ggml_tensor * t) {
        if (t->data || t->view_src || where.count(t)) {
            return;
        }
        const size_t sz = GGML_PAD(ggml_nbytes(t), al);
        for (auto it = free_mem.begin(); it != free_mem.end(); ++it) {
            if (it->second >= sz) {
                const size_t off = it->first;
                const size_t len = it->second;
                free_mem.erase(it);
                if (len > sz) {
                    free_mem[off + sz] = len - sz;
                }
                where[t] = { off, sz };
                at.emplace_back(t, off);
                hi = std::max(hi, off + sz);
                return;
            }
        }
        ok = false;
    };
    auto give = [&](const ggml_tensor * t) {
        auto w = where.find(t);
        if (w == where.end() || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return;
        }
        size_t off = w->second.first;
        size_t sz  = w->second.second;
        auto nx = free_mem.lower_bound(off);
        if (nx != free_mem.end() && nx->first == off + sz) {
            sz += nx->second;
            free_mem.erase(nx);
        }
        auto pv = free_mem.lower_bound(off);
        if (pv != free_mem.begin()) {
            --pv;
            if (pv->first + pv->second == off) {
                off = pv->first;
                sz += pv->second;
                free_mem.erase(pv);
            }
        }
        free_mem[off] = sz;
    };
    for (int i = 0; i < n && ok; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s]) {
                take(owner(node->src[s]));
            }
        }
        take(node);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] && last[owner(node->src[s])] == i) {
                give(owner(node->src[s]));
            }
        }
    }
    if (!ok) {
        return 0;
    }
    if (place) {
        uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
        for (const auto & [t, off] : at) {
            if (ggml_backend_tensor_alloc(buf, t, base + off) != GGML_STATUS_SUCCESS) {
                return 0;
            }
        }
        // the views, their sources first
        std::function<void(ggml_tensor *)> init_view = [&](ggml_tensor * t) {
            if (t->view_src == nullptr || t->buffer != nullptr) {
                return;
            }
            init_view(t->view_src);
            ggml_backend_view_init(t);
        };
        for (int i = 0; i < n; ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            init_view(node);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s]) {
                    init_view(node->src[s]);
                }
            }
        }
    }
    return hi;
}

void llama_kvq_fitter::compute(ggml_cgraph * gf, const char * what) {
    if (!plan(gf, plan_lo, true) || ggml_backend_graph_compute_async(store.backend(), gf) != GGML_STATUS_SUCCESS) {
        GGML_ABORT("%s: %s failed\n", __func__, what);
    }
}

// the own buffer and the layout in the shared memory: as many layers per group as fit next to the largest graph of one layer
void llama_kvq_fitter::dev_init() {
    const int64_t G      = store.n_head_kv();
    const int32_t n_lay  = (int32_t) layer_ids.size();
    const size_t  al     = 256;
    const size_t  shared = ggml_backend_buffer_get_size(store.shared_buffer());

    ggml_init_params ip = { 2*ggml_tensor_overhead(), nullptr, true };
    ctx_own.reset(ggml_init(ip));
    idx  = ggml_new_tensor_2d(ctx_own.get(), GGML_TYPE_I32, n_stage, G);
    tinv = ggml_new_tensor_3d(ctx_own.get(), GGML_TYPE_I8, KVQ_D, KVQ_D, 2*G);
    ggml_set_name(idx,  "kvq_fit_idx");
    ggml_set_name(tinv, "kvq_fit_tinv");
    buf_own.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_own.get(), store.dev_buft()));
    if (!buf_own) {
        throw std::runtime_error("failed to allocate the fitter device buffer");
    }
    std::vector<int32_t> ids((size_t) n_stage*G);
    for (size_t j = 0; j < ids.size(); ++j) {
        ids[j] = (int32_t) (j % n_stage);
    }
    ggml_backend_tensor_set(idx, ids.data(), 0, ggml_nbytes(idx));

    // the largest graphs, measured with tab for one layer, then the group
    ggml_init_params sp = { 4*ggml_tensor_overhead(), nullptr, true };
    auto make_shr = [&](int32_t ng) {
        ctx_shr.reset(ggml_init(sp));
        stage  = ggml_new_tensor_4d(ctx_shr.get(), GGML_TYPE_Q8_0, KVQ_D, 2, n_stage, G);
        cst[0] = ggml_new_tensor_3d(ctx_shr.get(), GGML_TYPE_I32, store.n_words(false), n_stage, G);
        cst[1] = ggml_new_tensor_3d(ctx_shr.get(), GGML_TYPE_I32, store.n_words(true) + 1, n_stage, G);
        tab    = ggml_new_tensor_3d(ctx_shr.get(), GGML_TYPE_F32, KVQ_D, 2*KVQ_D + 3, 2*G*ng);
        ggml_set_name(stage,  "kvq_fit_stage");
        ggml_set_name(cst[0], "kvq_fit_kc");
        ggml_set_name(cst[1], "kvq_fit_vc");
        ggml_set_name(tab,    "kvq_fit_tab");
    };
    job jb;
    jb.r0 = 0;
    jb.r1 = n_stage;
    auto measure = [&](int which, int32_t ng) {
        ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_cgraph * gf = which == 0 ? build_moments(jb, 0, ng, ctx.get()) : which == 1 ? build_codes(jb, 0, 0, ng, ctx.get()) : build_tables(jb, ng, ctx.get());
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE && node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE &&
                    !ggml_backend_supports_op(store.backend(), node)) {
                throw std::runtime_error(format("the device does not run the fit op %s (%s)", ggml_op_desc(node), node->name));
            }
        }
        return plan(gf, 0, false);
    };
    make_shr(1);
    const size_t inter  = std::max(measure(0, 1), measure(1, 1));
    const size_t fixed  = GGML_PAD(ggml_nbytes(stage), al) + GGML_PAD(ggml_nbytes(cst[0]), al) + GGML_PAD(ggml_nbytes(cst[1]), al);
    const size_t tab1   = GGML_PAD(ggml_nbytes(tab), al);
    n_grp = 0;
    for (int32_t ng = n_lay; ng >= 1; --ng) {
        make_shr(ng);
        if (fixed + ng*tab1 + std::max(inter, measure(2, ng)) <= shared) {
            n_grp = ng;
            break;
        }
    }
    if (n_grp == 0) {
        throw std::runtime_error(format("the shared memory of the store (%.1f MiB) is too small for the fitter", shared/1048576.0));
    }
    make_shr(n_grp);
    ggml_backend_buffer_t buf  = store.shared_buffer();
    uint8_t *             base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    size_t off = 0;
    for (ggml_tensor * t : { stage, cst[0], cst[1], tab }) {
        if (ggml_backend_tensor_alloc(buf, t, base + off) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to place the fitter tensors in the shared memory");
        }
        off += GGML_PAD(ggml_nbytes(t), al);
    }
    plan_lo = off;
    const size_t tables = measure(2, n_grp);
    LLAMA_LOG_INFO("%s: %d attention layers per group, %.1f MiB of the %.1f MiB shared memory (stage %.1f, tables %.1f, graphs %.1f / table graph %.1f), own %.1f MiB\n", __func__,
            n_grp, (plan_lo + std::max(inter, tables))/1048576.0, shared/1048576.0, fixed/1048576.0, n_grp*tab1/1048576.0, inter/1048576.0, tables/1048576.0,
            ggml_backend_buffer_get_size(buf_own.get())/1048576.0);
}

int32_t llama_kvq_fitter::n_pair() const {
    return (int32_t) layer_ids.size()*store.n_head_kv();
}

// waits for the events of the jobs in order, the decode thread installs the done ones
void llama_kvq_fitter::loop() {
    std::unique_lock<std::mutex> lk(m);
    while (true) {
        job * jb = nullptr;
        cv_work.wait(lk, [&]() {
            if (stop) {
                return true;
            }
            for (auto & j : jobs) {
                if (!j->done) {
                    jb = j.get();
                    return true;
                }
            }
            return false;
        });
        if (stop) {
            return;
        }
        // a cancel waits for the job to be done, so it stays in the queue until then
        lk.unlock();
        ggml_backend_event_synchronize(jb->ev.get());
        lk.lock();
        jb->done = true;
        cv_done.notify_all();
    }
}

// ctor, once per (layer index, kv head) p whose o_proj weight is not on the device: Wo_r^T Wo_r for the query heads r of
// kv head g, Wo = o_proj weight [n_out][n_in]
void llama_kvq_fitter::compute_wo_gram(int32_t p) {
    const int32_t       G = store.n_head_kv();
    const int32_t       g = p % G;
    const ggml_tensor * w = wo[p / G];

    const int64_t n_out = w->ne[1];
    const int64_t nc    = (int64_t) n_rep*KVQ_D; // the columns of kv head g
    const size_t  off   = ggml_row_size(w->type, g*nc);

    std::vector<uint8_t> raw(ggml_nbytes(w));
    ggml_backend_tensor_get(w, raw.data(), 0, raw.size());

    std::vector<double> gram((size_t) n_rep*KVQ_D*KVQ_D, 0.0);
    std::vector<float>  x(nc);
    for (int64_t o = 0; o < n_out; ++o) {
        const uint8_t * src = raw.data() + o*w->nb[1] + off;
        if (w->type == GGML_TYPE_F32) {
            memcpy(x.data(), src, nc*sizeof(float));
        } else {
            ggml_get_type_traits(w->type)->to_float(src, x.data(), nc);
        }
        for (int32_t r = 0; r < n_rep; ++r) {
            const float * xr = x.data() + r*KVQ_D;
            double      * gr = gram.data() + (size_t) r*KVQ_D*KVQ_D;
            for (int64_t i = 0; i < KVQ_D; ++i) {
                for (int64_t j = 0; j < KVQ_D; ++j) {
                    gr[i*KVQ_D + j] += (double) xr[i]*xr[j];
                }
            }
        }
    }
    wo_gram[p].resize(gram.size());
    for (size_t j = 0; j < gram.size(); ++j) {
        wo_gram[p][j] = (float) gram[j];
    }
}

// ctor: Wo_r^T Wo_r of every (layer index, kv head) on the device, the covariance op without the mean on the dequantized
// columns, in the shared memory
void llama_kvq_fitter::compute_wo_gram_dev(std::vector<char> & on_dev) {
    const int32_t  G       = store.n_head_kv();
    ggml_backend_t backend = store.backend();

    on_dev.assign(layer_ids.size()*G, 0);
    int64_t n_out = 0;
    for (const ggml_tensor * w : wo) {
        n_out = std::max(n_out, w->ne[1]);
    }

    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx_ids(ggml_init(ip));
    ggml_tensor * ids = ggml_new_tensor_2d(ctx_ids.get(), GGML_TYPE_I32, n_out, n_rep);
    ggml_backend_buffer_ptr buf_ids(ggml_backend_alloc_ctx_tensors_from_buft(ctx_ids.get(), store.dev_buft()));
    if (!buf_ids) {
        return;
    }
    std::vector<int32_t> v((size_t) n_out*n_rep);
    for (size_t j = 0; j < v.size(); ++j) {
        v[j] = (int32_t) (j % n_out);
    }
    ggml_backend_tensor_set(ids, v.data(), 0, ggml_nbytes(ids));

    store.fit_alloc();
    const double t0 = kvq_fit_now();
    int n_dev = 0;
    for (int32_t p = 0; p < (int32_t) on_dev.size(); ++p) {
        ggml_tensor * w = const_cast<ggml_tensor *>(wo[p / G]);
        if (w->buffer == nullptr || ggml_backend_buffer_get_type(w->buffer) != store.dev_buft()) {
            continue;
        }
        ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
        ggml_context_ptr ctx(ggml_init(gp));
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 16, false);

        // [D, n_out, n_rep]: the columns of query head (p % G)*n_rep + r of each row of Wo
        ggml_tensor * wv  = ggml_view_3d(ctx.get(), w, KVQ_D, w->ne[1], n_rep, w->nb[1], ggml_row_size(w->type, KVQ_D),
                ggml_row_size(w->type, (int64_t) (p % G)*n_rep*KVQ_D));
        ggml_tensor * iv  = ggml_view_2d(ctx.get(), ids, w->ne[1], n_rep, ids->nb[1], 0);
        ggml_tensor * x   = ggml_get_rows(ctx.get(), wv, iv);
        ggml_tensor * cov = ggml_kvq_fit(ctx.get(), GGML_KVQ_FIT_COV, x, nullptr, nullptr);
        ggml_set_output(cov);
        ggml_build_forward_expand(gf, cov);
        if (!ggml_backend_supports_op(backend, x) || !ggml_backend_supports_op(backend, cov) || !plan(gf, 0, true) ||
                ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            continue;
        }
        wo_gram[p].resize((size_t) n_rep*KVQ_D*KVQ_D);
        ggml_backend_tensor_get(cov, wo_gram[p].data(), 0, ggml_nbytes(cov));
        on_dev[p] = 1;
        n_dev++;
    }
    LLAMA_LOG_INFO("%s: o_proj grams of %d of %zu (layer, kv head) on the device in %.2f s\n", __func__, n_dev, on_dev.size(), kvq_fit_now() - t0);
}

// V metric of a range: sum over r of D_r Wo_r^T Wo_r D_r, D_r = diag of the range's mean sigmoid(gate) of query head g*n_rep + r
void llama_kvq_fitter::metric_v(const job & jb, int32_t p, float * m) {
    const int32_t G      = store.n_head_kv();
    const int32_t li     = p / G;
    const int32_t g      = p % G;
    const int64_t n_gate = (int64_t) n_head*KVQ_D;

    std::vector<float> gbar((size_t) n_rep*KVQ_D);
    const float * gs = jb.gsum.data() + li*n_gate + (int64_t) g*n_rep*KVQ_D;
    double tot = 0.0;
    for (size_t j = 0; j < gbar.size(); ++j) {
        gbar[j] = (float) ((double) gs[j]/jb.n());
        tot += gbar[j];
    }
    if (!(tot > 0.0)) {
        // no gate statistics (e.g. after a removal), plain o_proj metric
        LLAMA_LOG_WARN("%s: rows [%d, %d), layer %d, kv head %d: no gate statistics, the V table is fitted without the gate\n", __func__, jb.r0, jb.r1, layer_ids[li], g);
        std::fill(gbar.begin(), gbar.end(), 1.0f);
    }

    std::fill(m, m + KVQ_D*KVQ_D, 0.0f);
    for (int32_t r = 0; r < n_rep; ++r) {
        const float * d  = gbar.data() + (size_t) r*KVQ_D;
        const float * wg = wo_gram[p].data() + (size_t) r*KVQ_D*KVQ_D;
        for (int64_t i = 0; i < KVQ_D; ++i) {
            float       * mi = m + i*KVQ_D;
            const float * wi = wg + i*KVQ_D;
            for (int64_t j = 0; j < KVQ_D; ++j) {
                mi[j] += d[i]*wi[j]*d[j];
            }
        }
    }
}

// decode thread: the metric of every task of the job into its pinned buffer
void llama_kvq_fitter::host_metrics(job & jb) {
    const int32_t G = store.n_head_kv();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        for (int32_t g = 0; g < G; ++g) {
            const int32_t p = (int32_t) li*G + g;
            // K: second moment of the queries of the range, the fit does not depend on its scale
            float       * mk = jb.hmet + ((size_t) li*2*G + g)*KVQ_D*KVQ_D;
            const float * gr = jb.gram.data() + (size_t) p*KVQ_D*KVQ_D;
            double tr = 0.0;
            for (int64_t j = 0; j < KVQ_D; ++j) {
                tr += gr[j*KVQ_D + j];
            }
            if (tr > 0.0) {
                memcpy(mk, gr, KVQ_D*KVQ_D*sizeof(float));
            } else {
                // no queries (e.g. after a removal), plain MSE
                LLAMA_LOG_WARN("%s: rows [%d, %d), layer %d, kv head %d: no query statistics, the K table is fitted for the MSE\n", __func__, jb.r0, jb.r1, layer_ids[li], g);
                std::fill(mk, mk + KVQ_D*KVQ_D, 0.0f);
                for (int64_t j = 0; j < KVQ_D; ++j) {
                    mk[j*KVQ_D + j] = 1.0f;
                }
            }
            metric_v(jb, p, jb.hmet + ((size_t) li*2*G + G + g)*KVQ_D*KVQ_D);
        }
    }
}

// rows [row0, row0 + n_rows) of the tables of layer j of a group of n_lay layers, K or V of its kv heads: [D, n_rows, G]
ggml_tensor * llama_kvq_fitter::tab_view(int kv, int32_t j, int32_t n_lay, size_t row0, int64_t n_rows, ggml_context * ctx) const {
    const int64_t G = store.n_head_kv();
    return ggml_view_3d(ctx, tab, KVQ_D, n_rows, G, tab->nb[1], tab->nb[2], (size_t) (kv*n_lay*G + j*G)*tab->nb[2] + row0*tab->nb[1]);
}

// the rows of the stage [D, n, G], K or V
static ggml_tensor * kvq_fit_rows(ggml_context * ctx, ggml_tensor * stage, ggml_tensor * idx, int kv, int64_t n) {
    ggml_tensor * sv = ggml_view_3d(ctx, stage, stage->ne[0], stage->ne[2], stage->ne[3], stage->nb[2], stage->nb[3], kv*stage->nb[1]);
    ggml_tensor * iv = ggml_view_2d(ctx, idx, n, idx->ne[1], idx->nb[1], 0);
    return ggml_get_rows(ctx, sv, iv);
}

// layer j of a group: mean and covariance of the rows of the job (from the stage) -> its tables (rows [0, D) and 2D)
ggml_cgraph * llama_kvq_fitter::build_moments(const job & jb, int32_t j, int32_t n_lay, ggml_context * ctx) {
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    for (int kv = 0; kv < 2; ++kv) {
        ggml_tensor * x   = kvq_fit_rows(ctx, stage, idx, kv, jb.n());
        ggml_tensor * mu  = ggml_kvq_fit(ctx, GGML_KVQ_FIT_MEAN, x, nullptr, nullptr);
        ggml_tensor * cov = ggml_kvq_fit(ctx, GGML_KVQ_FIT_COV, ggml_sub(ctx, x, mu), nullptr, nullptr);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cov, tab_view(kv, j, n_lay, 0, KVQ_D, ctx)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, mu, tab_view(kv, j, n_lay, 2*KVQ_D, 1, ctx)));
    }
    return gf;
}

// the tables of a group of n_lay layers (the metrics are in rows [D, 2D)), the result rows to the output
ggml_cgraph * llama_kvq_fitter::build_tables(const job & jb, int32_t n_lay, ggml_context * ctx) {
    const int64_t G = store.n_head_kv();
    const int64_t B = 2*n_lay*G;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_tensor * t    = ggml_view_3d(ctx, tab, KVQ_D, 2*KVQ_D + 3, B, tab->nb[1], tab->nb[2], 0);
    ggml_tensor * work = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, ggml_kvq_fit_table_work_size(KVQ_D, B));
    ggml_tensor * tb   = ggml_kvq_fit_table(ctx, t, work, (int) jb.n(), (int) (n_lay*G));
    ggml_tensor * res  = ggml_cont(ctx, ggml_view_3d(ctx, tb, KVQ_D, 1, B, tb->nb[1], tb->nb[2], (2*KVQ_D + 2)*tb->nb[1]));
    ggml_set_output(res);
    ggml_set_name(res, "kvq_fit_res");
    ggml_build_forward_expand(gf, res);
    return gf;
}

// layer li (index j in a group of n_lay): rows of the job (from the stage) -> K: bits; V: rows rotated, scale of the shape, bits
// -> packed tables (outputs); rows rotated -> codes in kc and vc
ggml_cgraph * llama_kvq_fitter::build_codes(const job & jb, int32_t li, int32_t j, int32_t n_lay, ggml_context * ctx) {
    const int64_t G  = store.n_head_kv();
    const int64_t n  = jb.n();
    const auto &  lt = store.get_layer(layer_ids[li]);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    for (int kv = 0; kv < 2; ++kv) {
        ggml_tensor * fwd = tab_view(kv, j, n_lay, 0, KVQ_D, ctx);
        ggml_tensor * gp  = tab_view(kv, j, n_lay, 2*KVQ_D - 1, 4, ctx);
        ggml_tensor * mu  = tab_view(kv, j, n_lay, 2*KVQ_D, 1, ctx);
        ggml_tensor * y   = ggml_kvq_fit(ctx, GGML_KVQ_FIT_ROT, ggml_sub(ctx, kvq_fit_rows(ctx, stage, idx, kv, n), mu), fwd, nullptr);
        ggml_tensor * sig = kv ? ggml_kvq_fit(ctx, GGML_KVQ_FIT_GAIN, y, gp, nullptr) : tab_view(kv, j, n_lay, 2*KVQ_D + 1, 1, ctx);
        ggml_tensor * p4  = ggml_kvq_fit_bits(ctx, sig, gp, (float) (kv ? fparams.bits_v : fparams.bits_k));

        ggml_tensor * xi  = tab_view(kv, j, n_lay, KVQ_D, KVQ_D + 1, ctx);
        ggml_tensor * ti  = ggml_view_3d(ctx, tinv, KVQ_D, KVQ_D, G, tinv->nb[1], tinv->nb[2], kv*G*tinv->nb[2]);
        ggml_build_forward_expand(gf, ggml_kvq_fit_pack(ctx, xi, p4, ti));
        ggml_tensor * aux = ggml_kvq_fit_pack(ctx, xi, p4, nullptr);
        ggml_set_output(aux);
        ggml_format_name(aux, "kvq_fit_aux%d", kv);
        ggml_build_forward_expand(gf, aux);

        // the codes go to the host copy (queue_job); without the device codes through the code stage
        const bool    dev  = store.codes_on_dev();
        ggml_tensor * rows = dev ? (kv ? lt.vc : lt.kc) : cst[kv];
        ggml_tensor * rv   = ggml_view_3d(ctx, rows, rows->ne[0], n, G, rows->nb[1], rows->nb[2], dev ? (size_t) jb.r0*rows->nb[1] : 0);
        ggml_build_forward_expand(gf, ggml_kvq_fit(ctx, GGML_KVQ_FIT_CODE, y, p4, rv));
    }
    return gf;
}

// host store rows [r0, r1) of layer il -> stage rows [0, r1 - r0), one copy per kv head (the same layout), main stream
void llama_kvq_fitter::upload_rows(int32_t il, int32_t r0, int32_t r1) {
    if (r1 <= r0) {
        return;
    }
    const int64_t G = store.n_head_kv();
    for (int64_t g = 0; g < G; ++g) {
        const uint8_t * src = store.host_rows(il) + ((size_t) g*store.n_rows() + r0)*2*KVQ_ROW;
        ggml_backend_tensor_set_async(store.backend(), stage, src, g*stage->nb[3], (size_t) (r1 - r0)*2*KVQ_ROW);
    }
}

// decode thread: queues the graphs of a job on the main stream and the readbacks into its pinned buffer, no wait
// every graph reads the rows from the host store (the host ring copies the rows of the ring there)
void llama_kvq_fitter::queue_job(job & jb) {
    ggml_backend_t backend = store.backend();
    const int64_t  G       = store.n_head_kv();
    const size_t   nn      = (size_t) KVQ_D*KVQ_D;
    const int32_t  n_layer = (int32_t) layer_ids.size();

    // the shared memory is idle between the ubatches; this also waits for the readbacks of the last ubatches into the host store
    store.fit_alloc();

    for (int32_t g0 = 0; g0 < n_layer; g0 += n_grp) {
        const int32_t n_lay = std::min(n_grp, n_layer - g0);
        for (int32_t j = 0; j < n_lay; ++j) {
            upload_rows(layer_ids[g0 + j], jb.r0, jb.r1);
            ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
            ggml_context_ptr ctx(ggml_init(gp));
            compute(build_moments(jb, j, n_lay, ctx.get()), "the moments");
            // the metrics into rows [D, 2D) of the tables, task kv*n_lay*G + j*G + g
            for (int kv = 0; kv < 2; ++kv) {
                for (int64_t g = 0; g < G; ++g) {
                    const size_t t = (size_t) (kv*n_lay*G + j*G + g);
                    ggml_backend_tensor_set_async(backend, tab, jb.hmet + ((size_t) (g0 + j)*2*G + kv*G + g)*nn, t*tab->nb[2] + KVQ_D*tab->nb[1], nn*sizeof(float));
                }
            }
        }
        {
            ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
            ggml_context_ptr ctx(ggml_init(gp));
            ggml_cgraph * gf = build_tables(jb, n_lay, ctx.get());
            compute(gf, "the tables");
            ggml_tensor * res = ggml_graph_get_tensor(gf, "kvq_fit_res");
            for (int kv = 0; kv < 2; ++kv) {
                for (int32_t j = 0; j < n_lay; ++j) {
                    ggml_backend_tensor_get_async(backend, res, jb.hres + ((size_t) (g0 + j)*2*G + kv*G)*KVQ_D,
                            (size_t) (kv*n_lay*G + j*G)*KVQ_D*sizeof(float), G*KVQ_D*sizeof(float));
                }
            }
        }
        for (int32_t j = 0; j < n_lay; ++j) {
            const int32_t li = g0 + j;
            const int32_t il = layer_ids[li];
            upload_rows(il, jb.r0, jb.r1);
            ggml_init_params gp = { graph_meta.size(), graph_meta.data(), true };
            ggml_context_ptr ctx(ggml_init(gp));
            ggml_cgraph * gf = build_codes(jb, li, j, n_lay, ctx.get());
            compute(gf, "the codes");

            // the outputs, before the next graph reuses the memory; task i = kv*n_pair + li*G + g
            for (int kv = 0; kv < 2; ++kv) {
                const size_t i0 = (size_t) kv*n_pair() + li*G;
                ggml_backend_tensor_get_async(backend, ggml_graph_get_tensor(gf, format("kvq_fit_aux%d", kv).c_str()), jb.htaux + i0*3*KVQ_D, 0, G*3*KVQ_D*sizeof(float));
                ggml_backend_tensor_get_async(backend, tinv, jb.htinv + i0*nn, kv*G*tinv->nb[2], G*nn);
            }
            // the codes to the host copy, in the layout of kc / vc
            const bool dev = store.codes_on_dev();
            for (int kv = 0; kv < 2; ++kv) {
                const ggml_tensor * rows = kv ? store.get_layer(il).vc : store.get_layer(il).kc;
                ggml_tensor       * src  = dev ? const_cast<ggml_tensor *>(rows) : cst[kv];
                uint8_t           * dst  = store.host_codes(il, kv == 1);
                for (int64_t g = 0; g < G; ++g) {
                    ggml_backend_tensor_get_async(backend, src, dst + g*rows->nb[2] + (size_t) jb.r0*rows->nb[1],
                            g*src->nb[2] + (dev ? (size_t) jb.r0*rows->nb[1] : 0), (size_t) jb.n()*rows->nb[1]);
                }
            }
        }
    }
    store.codes_written(jb.r1);
    ggml_backend_event_record(jb.ev.get(), backend);
}

// decode thread: the sums of the open chunk on the device, without a reset
void llama_kvq_fitter::read_acc(std::vector<float> & gram, std::vector<float> & gsum) {
    const size_t nn = (size_t) KVQ_D*KVQ_D*store.n_head_kv();
    gram.resize(nn*layer_ids.size());
    gsum.resize((size_t) n_head*KVQ_D*layer_ids.size());
    store.synchronize();
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        ggml_tensor * acc = store.get_layer(layer_ids[li]).qacc;
        ggml_backend_tensor_get(acc, gram.data() + li*nn, 0, ggml_nbytes(acc));
        ggml_tensor * gacc = store.get_layer(layer_ids[li]).gacc;
        ggml_backend_tensor_get(gacc, gsum.data() + li*gacc->ne[0], 0, ggml_nbytes(gacc));
    }
}

// decode thread, n_pos positions in the sequence: the statistics of the complete chunks, the sums of the ubatches of a chunk
// are computed when it is complete, then they start again from 0 (a ubatch counts for the chunk of its first position)
void llama_kvq_fitter::snapshots(llama_pos n_pos) {
    const int32_t C = store.n_chunk_rows();
    while (!failed && (int64_t) (n_snap + 1)*C <= store.n_rows() && n_pos >= (n_snap + 1)*C) {
        std::vector<float> gram;
        std::vector<float> gsum;
        read_acc(gram, gsum);
        set_stats(nullptr, nullptr);
        acc_pos = -1;
        grams.push_back(std::move(gram));
        gsums.push_back(std::move(gsum));
        t_snaps.push_back(kvq_fit_now());
        n_snap++;
    }
}

// the layout of [0, n): the complete chunks alone, then the rows after them (own: alone, if there are min_rows of them;
// merge: one group with the last complete chunk); nothing if the coarse region would be smaller than min_first
std::vector<llama_kvq_fitter::range> llama_kvq_fitter::layout(llama_pos n) const {
    const int32_t C      = store.n_chunk_rows();
    const int32_t n_full = std::min<int32_t>(n, store.n_rows())/C;
    const bool    merge  = fparams.rem == LLAMA_KVQ_REM_MERGE && n_full > 0;
    const int32_t n_solo = merge ? n_full - 1 : n_full;

    std::vector<range> res;
    for (int32_t k = 0; k < n_solo; ++k) {
        res.push_back({ k*C, (k + 1)*C });
    }
    const int32_t r0 = n_solo*C;
    const int32_t r1 = std::min<int32_t>(n, store.n_rows());
    if (r1 > r0 && (merge || n_full == 0 || r1 - r0 >= fparams.min_rows)) {
        res.push_back({ r0, r1 });
    }
    if (!res.empty() && res.back().r1 < min_first) {
        res.clear();
    }
    return res;
}

bool llama_kvq_fitter::fitted(const range & r) const {
    const int32_t C = store.n_chunk_rows();
    return r.r1 <= store.n_coarse() && slot_fit[r.r0/C] == r;
}

bool llama_kvq_fitter::in_flight(const range & r) const {
    std::lock_guard<std::mutex> lk(m);
    return std::any_of(jobs.begin(), jobs.end(), [&](const std::unique_ptr<job> & jb) { return jb->r0 == r.r0 && jb->r1 == r.r1; });
}

// the ring cells of the positions [p0, p1) hold them, for sequence 0 only
bool llama_kvq_fitter::check_ring(const llama_kv_cache & kv, int32_t p0, int32_t p1) {
    const auto & cells = kv.get_cells(0);
    const int64_t size = cells.size();
    if (p1 - p0 > size) {
        LLAMA_LOG_ERROR("%s: the positions [%d, %d) are not in the ring of %lld cells, the fit of the coarse region stops\n", __func__, p0, p1, (long long) size);
        failed = true;
        return false;
    }
    for (int64_t p = p0; p < p1; ++p) {
        const int64_t j = p % size;
        if (cells.is_empty(j) || cells.pos_get(j) != p || !cells.seq_has(j, 0) || cells.seq_count(j) != 1) {
            LLAMA_LOG_ERROR("%s: cell %lld does not hold position %lld of sequence 0 only, the fit of the coarse region stops\n", __func__, (long long) j, (long long) p);
            failed = true;
            return false;
        }
    }
    return true;
}

// decode thread: a new job for the rows of r with their statistics, phase 1 and the fit tasks
// the ring rows go to the host store, so n_host covers r afterwards
bool llama_kvq_fitter::take(const range & r) {
    const int32_t C  = store.n_chunk_rows();
    const int32_t G  = store.n_head_kv();
    GGML_ASSERT(r.r0 % C == 0 && r.r1 > r.r0 && r.r1 - r.r0 <= n_stage);

    auto jb = std::make_unique<job>();
    jb->r0     = r.r0;
    jb->r1     = r.r1;
    jb->t_take = kvq_fit_now();
    jb->t_snap = jb->t_take;

    // statistics: the complete chunks of the range, then the sums of the open chunk, in this order
    // a partial chunk alone also takes the previous chunk's: its own sums can hold few positions (a ubatch counts for the chunk
    // of its first position), the queries just before it stand in for them
    const bool partial = r.r1 - r.r0 < C && r.r0 >= C;
    for (int32_t k = r.r0/C - (partial ? 1 : 0); (int64_t) k*C < r.r1; ++k) {
        std::vector<float> gram;
        std::vector<float> gsum;
        if ((k + 1)*C <= r.r1) {
            GGML_ASSERT(k < n_snap);
            gram = grams[k];
            gsum = gsums[k];
            jb->t_snap = t_snaps[k];
        } else {
            GGML_ASSERT(k == n_snap);
            read_acc(gram, gsum);
            acc_pos  = r.r1;
            acc_gram = gram;
            acc_gsum = gsum;
        }
        if (jb->gram.empty()) {
            jb->gram = std::move(gram);
            jb->gsum = std::move(gsum);
            continue;
        }
        for (size_t j = 0; j < gram.size(); ++j) {
            jb->gram[j] += gram[j];
        }
        for (size_t j = 0; j < gsum.size(); ++j) {
            jb->gsum[j] += gsum[j];
        }
    }

    const int32_t n_task = 2*n_pair();
    const size_t  n_b    = (size_t) n_task*KVQ_D*KVQ_D*sizeof(float) + (size_t) n_task*KVQ_D*sizeof(float) +
                           (size_t) n_task*KVQ_D*KVQ_D*sizeof(int8_t) + (size_t) n_task*3*KVQ_D*sizeof(float);
    if (!free_bufs.empty()) {
        jb->buf = std::move(free_bufs.back());
        free_bufs.pop_back();
    } else {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(store.host_buft(), n_b);
        jb->buf.reset(b != nullptr ? b : ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), n_b));
        if (!jb->buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate %.1f MiB for a job, the fit of the coarse region stops\n", __func__, n_b/1048576.0);
            failed = true;
            return false;
        }
    }
    uint8_t * hb = (uint8_t *) ggml_backend_buffer_get_base(jb->buf.get());
    jb->hmet  = (float *) hb;  hb += (size_t) n_task*KVQ_D*KVQ_D*sizeof(float);
    jb->hres  = (float *) hb;  hb += (size_t) n_task*KVQ_D*sizeof(float);
    jb->htinv = (int8_t *) hb; hb += (size_t) n_task*KVQ_D*KVQ_D*sizeof(int8_t);
    jb->htaux = (float *) hb;
    jb->ev.reset(ggml_backend_event_new(store.device()));
    if (!jb->ev) {
        LLAMA_LOG_ERROR("%s: failed to create an event, the fit of the coarse region stops\n", __func__);
        failed = true;
        return false;
    }

    host_metrics(*jb);
    queue_job(*jb);
    store.set_n_host(std::max(store.n_host(), r.r1));

    {
        std::lock_guard<std::mutex> lk(m);
        jobs.push_back(std::move(jb));
    }
    cv_work.notify_all();
    return true;
}

void llama_kvq_fitter::spill(const llama_kv_cache & kv, llama_pos p) {
    const int32_t h0 = store.n_host();
    if (failed || p <= h0) {
        return;
    }
    if (!check_ring(kv, h0, p)) {
        return;
    }
    // the host ring: the host store has the rows of the ring (llama_kvq_store::sync_store)
    store.set_n_host(p);
    n_spill++;
    LLAMA_LOG_INFO("%s: ring rows of the positions [%d, %d) to the host store\n", __func__, h0, (int) p);
}

// decode thread: the finished jobs at the front join the coarse region, all: also when hold
// the tables are queued on the main stream; the wait for them (set_n_coarse or synchronize) comes before the host buffer is reused
void llama_kvq_fitter::install(bool all) {
    if (hold && !all) {
        return;
    }
    const int32_t G = store.n_head_kv();
    const int32_t C = store.n_chunk_rows();
    while (!failed) {
        std::unique_ptr<job> jb;
        {
            std::lock_guard<std::mutex> lk(m);
            if (jobs.empty() || !jobs.front()->done) {
                return;
            }
            jb = std::move(jobs.front());
            jobs.pop_front();
        }
        GGML_ASSERT(jb->r0 <= store.n_coarse());

        // the result row of each table (ggml_kvq_fit_table): status, ridge of the metric
        int n_ridge = 0;
        for (size_t li = 0; li < layer_ids.size() && !failed; ++li) {
            for (int32_t t = 0; t < 2*G; ++t) {
                const float * res = jb->hres + (li*2*G + t)*KVQ_D;
                if (res[1] != 0.0f) {
                    LLAMA_LOG_ERROR("%s: rows [%d, %d): the %s fit of layer %d, kv head %d failed (%s), the coarse region stays at %d rows\n", __func__,
                            jb->r0, jb->r1, t < G ? "K" : "V", layer_ids[li], t % G, res[1] == 1.0f ? "metric not positive definite" : "eigen solver",
                            store.n_coarse());
                    failed = true;
                    break;
                }
                n_ridge += res[2] > 1e-6f ? 1 : 0;
            }
        }
        if (failed) {
            store.synchronize();
            release(*jb);
            return;
        }
        if (n_ridge > 0) {
            LLAMA_LOG_WARN("%s: rows [%d, %d): %d metrics needed a ridge above 1e-6 of their mean diagonal\n", __func__, jb->r0, jb->r1, n_ridge);
        }

        // merge: the group's table set in each chunk slot of its rows
        for (int32_t t = jb->r0/C; (int64_t) t*C < jb->r1; ++t) {
            for (size_t li = 0; li < layer_ids.size(); ++li) {
                for (int32_t g = 0; g < G; ++g) {
                    for (int kv = 0; kv < 2; ++kv) {
                        const size_t i = kv*n_pair() + li*G + g;
                        store.set_chunk_table(layer_ids[li], g, kv == 1, t, jb->htinv + i*KVQ_D*KVQ_D, jb->htaux + i*3*KVQ_D, true);
                    }
                }
            }
            slot_fit[t] = { jb->r0, jb->r1 };
        }
        const int32_t n0 = store.n_coarse();
        if (jb->r1 > n0) {
            store.set_n_coarse(jb->r1);
        } else {
            store.synchronize();
        }

        const double t1 = kvq_fit_now();
        t_job.push_back(t1 - jb->t_take);
        t_wall.push_back(t1 - jb->t_snap);
        LLAMA_LOG_INFO("%s: rows [%d, %d) fitted (coarse region %d rows), job %.3f s, %.2f s since the statistics\n", __func__,
                jb->r0, jb->r1, store.n_coarse(), t1 - jb->t_take, t1 - jb->t_snap);
        release(*jb);
    }
}

void llama_kvq_fitter::poll(const llama_kv_cache & kv, llama_pos n_pos, bool large) {
    const int32_t C  = store.n_chunk_rows();
    const double  t0 = kvq_fit_now();
    const int32_t s0 = n_snap;
    snapshots(n_pos);
    bool took = n_snap != s0;
    // the complete chunks, in order; also a chunk whose tables came from a group or a shorter fit
    // merge: a chunk is fitted alone once the next one is complete too, before that it is the head of the group
    const int32_t lag = fparams.rem == LLAMA_KVQ_REM_MERGE ? 1 : 0;
    for (int32_t k = 0; large && !failed && k + lag < n_snap; ++k) {
        const range r = { k*C, (k + 1)*C };
        if (!fitted(r) && !in_flight(r)) {
            took = take(r) || took;
        }
    }
    const double t1 = kvq_fit_now();
    if (took) {
        t_take += t1 - t0;
    }
    if (!failed && !jobs_empty()) {
        install(false);
        t_install += kvq_fit_now() - t1;
    }
}

bool llama_kvq_fitter::jobs_empty() const {
    std::lock_guard<std::mutex> lk(m);
    return jobs.empty();
}

void llama_kvq_fitter::wait_jobs() {
    while (true) {
        install(true);
        std::unique_lock<std::mutex> lk(m);
        if (failed || jobs.empty()) {
            return;
        }
        cv_done.wait(lk, [&]() { return jobs.front()->done; });
    }
}

llama_pos llama_kvq_fitter::settle(const llama_kv_cache & kv, llama_pos n, const std::function<void(llama_pos)> & queued) {
    snapshots(n);
    const std::vector<range> lay = layout(n);
    const llama_pos e = lay.empty() ? 0 : lay.back().r1;
    if (failed) {
        return store.n_coarse();
    }

    // e.g. merge: the job of the last complete chunk alone, which the group replaces
    cancel([&](const job & jb) {
        return std::find(lay.begin(), lay.end(), range{ jb.r0, jb.r1 }) == lay.end();
    });
    for (const range & r : lay) {
        if (!fitted(r) && !in_flight(r) && !take(r)) {
            break;
        }
    }
    if (queued && !failed) {
        queued(e);
    }
    wait_jobs();
    return failed ? store.n_coarse() : e;
}

// decode thread: cancels the jobs of pred, waits for their running tasks and drops them
void llama_kvq_fitter::cancel(const std::function<bool(const job &)> & pred) {
    std::vector<std::unique_ptr<job>> gone;
    {
        std::unique_lock<std::mutex> lk(m);
        bool any = false;
        for (auto & jb : jobs) {
            if (pred(*jb)) {
                jb->cancel = true;
                any = true;
                LLAMA_LOG_INFO("%s: the job of the rows [%d, %d) is cancelled (%s)\n", __func__, jb->r0, jb->r1, jb->done ? "done" : "running");
            }
        }
        if (!any) {
            return;
        }
        // the device work of a job cannot stop; the waiter may hold a pointer to it until it is done
        cv_done.wait(lk, [&]() {
            return std::all_of(jobs.begin(), jobs.end(), [](const std::unique_ptr<job> & jb) { return !jb->cancel || jb->done; });
        });
        for (auto it = jobs.begin(); it != jobs.end();) {
            if ((*it)->cancel) {
                n_cancel++;
                gone.push_back(std::move(*it));
                it = jobs.erase(it);
            } else {
                ++it;
            }
        }
    }
    // also the queued phases of the cancelled jobs, which use their host buffers
    store.synchronize();
    for (auto & jb : gone) {
        release(*jb);
    }
}

void llama_kvq_fitter::remove(llama_pos p0) {
    const int32_t C  = store.n_chunk_rows();
    const int32_t c0 = p0 / C;

    cancel([&](const job & jb) { return jb.r1 > p0; });

    // a removal in the chunk that collects statistics keeps the removed queries, unless all of them go or the removal goes back
    // to the end of the last fit of its rows; into a complete chunk: its statistics are collected again from the ones read when it was complete
    if (n_snap <= c0) {
        if (p0 == (llama_pos) c0*C && n_snap == c0) {
            store.synchronize();
            set_stats(nullptr, nullptr);
        } else if (p0 == acc_pos && n_snap == c0) {
            store.synchronize();
            set_stats(&acc_gram, &acc_gsum);
        }
        if (p0 < acc_pos) {
            acc_pos = -1;
        }
    } else {
        acc_pos = -1;
        store.synchronize();
        if (p0 > (llama_pos) c0*C) {
            set_stats(&grams[c0], &gsums[c0]);
        } else {
            set_stats(nullptr, nullptr);
        }
        grams.resize(c0);
        gsums.resize(c0);
        t_snaps.resize(c0);
        n_snap = c0;
    }

    // the coarse rows at or after p0: back to the chunk grid, the rows before p0 stay in the host store
    if (store.n_coarse() > p0) {
        const int32_t n0 = store.n_coarse();
        store.set_n_coarse(c0*C);
        for (int32_t t = c0; t < (int32_t) slot_fit.size(); ++t) {
            slot_fit[t] = range();
        }
        n_drop++;
        LLAMA_LOG_INFO("%s: the removal from position %d drops the coarse rows [%d, %d), the host store keeps [%d, %d)\n", __func__,
                p0, c0*C, n0, c0*C, (int) p0);
    }
    store.set_n_host(std::min<int32_t>(store.n_host(), p0));
    if (p0 == 0) {
        failed = false;
    }
}

void llama_kvq_fitter::state_write_meta(llama_io_write_i & io) const {
    {
        std::lock_guard<std::mutex> lk(m);
        if (failed || !jobs.empty()) {
            throw std::runtime_error(format("the kvq fitter is not idle (failed = %d, %zu jobs), the state is not saved", failed ? 1 : 0, jobs.size()));
        }
    }
    // element 2: V fitted per chunk (1; 0 was an offline V table file), element 3: the query heads of the gate statistics,
    // elements 4, 5: the policy for the rows after the last complete chunk and its smallest chunk (they change what the next round does)
    const int32_t v[6] = { fparams.bits_k, fparams.bits_v, 1, n_head, fparams.rem, fparams.min_rows };
    io.write(v, sizeof(v));
}

void llama_kvq_fitter::state_read_meta(llama_io_read_i & io) const {
    int32_t v[6];
    io.read(v, sizeof(v));
    if (v[0] != fparams.bits_k || v[1] != fparams.bits_v || v[2] != 1 || v[3] != n_head) {
        throw std::runtime_error(format("kvq state with K %d / V %d bits, V %s (%d heads), this context: K %d / V %d bits, V fitted per chunk (%d heads)",
                v[0], v[1], v[2] ? "fitted per chunk" : "from a table file", v[3], fparams.bits_k, fparams.bits_v, n_head));
    }
    if (v[4] != fparams.rem || v[5] != fparams.min_rows) {
        throw std::runtime_error(format("kvq state fitted with rem policy %d and min rows %d, this context: rem policy %d, min rows %d "
                "(the rows after the last complete chunk would be fitted differently); use the same --kvq-rem / --kvq-min-rows or process the prompt again",
                v[4], v[5], fparams.rem, fparams.min_rows));
    }
}

static size_t kvq_fit_n_gram(const llama_kvq_store & store) {
    return (size_t) KVQ_D*KVQ_D*store.n_head_kv()*store.layer_ids().size();
}

// the fitted range of each chunk slot of the coarse region, the statistics of the complete chunks, the device sums of the open chunk
void llama_kvq_fitter::state_write_data(llama_io_write_i & io) const {
    const int32_t C      = store.n_chunk_rows();
    const int32_t n_slot = (store.n_coarse() + C - 1)/C;
    const size_t  n_gram = kvq_fit_n_gram(store);
    const size_t  n_gsum = (size_t) n_head*KVQ_D*layer_ids.size();
    io.write(&n_slot, sizeof(n_slot));
    for (int32_t t = 0; t < n_slot; ++t) {
        io.write(&slot_fit[t].r0, sizeof(int32_t));
        io.write(&slot_fit[t].r1, sizeof(int32_t));
    }
    io.write(&n_snap, sizeof(n_snap));
    for (int32_t c = 0; c < n_snap; ++c) {
        GGML_ASSERT(grams[c].size() == n_gram && gsums[c].size() == n_gsum);
        io.write(grams[c].data(), n_gram*sizeof(float));
        if (n_gsum > 0) {
            io.write(gsums[c].data(), n_gsum*sizeof(float));
        }
    }
    for (int32_t il : layer_ids) {
        const auto & t = store.get_layer(il);
        io.write_tensor(t.qacc, 0, ggml_nbytes(t.qacc));
        io.write_tensor(t.gacc, 0, ggml_nbytes(t.gacc));
    }
    io.write(&acc_pos, sizeof(acc_pos));
    if (acc_pos >= 0) {
        GGML_ASSERT(acc_gram.size() == n_gram && acc_gsum.size() == n_gsum);
        io.write(acc_gram.data(), n_gram*sizeof(float));
        if (n_gsum > 0) {
            io.write(acc_gsum.data(), n_gsum*sizeof(float));
        }
    }
}

void llama_kvq_fitter::state_read_data(llama_io_read_i & io) {
    {
        std::lock_guard<std::mutex> lk(m);
        GGML_ASSERT(jobs.empty());
    }
    const int32_t C      = store.n_chunk_rows();
    const size_t  n_gram = kvq_fit_n_gram(store);
    const size_t  n_gsum = (size_t) n_head*KVQ_D*layer_ids.size();

    int32_t n_slot = 0;
    io.read(&n_slot, sizeof(n_slot));
    if (n_slot != (store.n_coarse() + C - 1)/C) {
        throw std::runtime_error(format("kvq state with %d fitted chunk slots for a coarse region of %d rows", n_slot, store.n_coarse()));
    }
    std::fill(slot_fit.begin(), slot_fit.end(), range());
    for (int32_t t = 0; t < n_slot; ++t) {
        range r;
        io.read(&r.r0, sizeof(int32_t));
        io.read(&r.r1, sizeof(int32_t));
        if (r.r0 < 0 || r.r0 > t*C || r.r1 <= t*C || r.r1 - r.r0 > n_stage) {
            throw std::runtime_error(format("kvq state with the rows [%d, %d) fitted for chunk %d", r.r0, r.r1, t));
        }
        slot_fit[t] = r;
    }
    int32_t ns = 0;
    io.read(&ns, sizeof(ns));
    if (ns < 0 || (int64_t) ns*C > store.n_rows()) {
        throw std::runtime_error(format("kvq state with %d complete chunks", ns));
    }
    grams.assign(ns, {});
    gsums.assign(ns, {});
    t_snaps.assign(ns, kvq_fit_now());
    for (int32_t c = 0; c < ns; ++c) {
        grams[c].resize(n_gram);
        io.read(grams[c].data(), n_gram*sizeof(float));
        gsums[c].resize(n_gsum);
        if (n_gsum > 0) {
            io.read(gsums[c].data(), n_gsum*sizeof(float));
        }
    }
    n_snap = ns;
    failed = false;
    store.synchronize();
    for (int32_t il : layer_ids) {
        const auto & t = store.get_layer(il);
        io.read_tensor(t.qacc, 0, ggml_nbytes(t.qacc));
        io.read_tensor(t.gacc, 0, ggml_nbytes(t.gacc));
    }
    io.read(&acc_pos, sizeof(acc_pos));
    acc_gram.clear();
    acc_gsum.clear();
    if (acc_pos >= 0) {
        acc_gram.resize(n_gram);
        io.read(acc_gram.data(), n_gram*sizeof(float));
        acc_gsum.resize(n_gsum);
        if (n_gsum > 0) {
            io.read(acc_gsum.data(), n_gsum*sizeof(float));
        }
    }
}

// decode thread, no phase of the job queued: keeps the host buffer of a finished or cancelled job for the next one
void llama_kvq_fitter::release(job & jb) {
    if (jb.buf && free_bufs.size() < 2) {
        free_bufs.push_back(std::move(jb.buf));
    }
    jb.buf.reset();
}

// decode thread, after a synchronize: the device statistics from the host copies, or 0 (nullptr or no copy)
void llama_kvq_fitter::set_stats(const std::vector<float> * gram, const std::vector<float> * gsum) {
    for (size_t li = 0; li < layer_ids.size(); ++li) {
        const auto & t = store.get_layer(layer_ids[li]);
        for (int k = 0; k < 2; ++k) {
            ggml_tensor * acc = k == 0 ? t.qacc : t.gacc;
            const std::vector<float> * src = k == 0 ? gram : gsum;
            if (acc == nullptr) {
                continue;
            }
            const int64_t ne = ggml_nelements(acc);
            if (src != nullptr && (int64_t) src->size() == ne*(int64_t) layer_ids.size()) {
                ggml_backend_tensor_set(acc, src->data() + li*ne, 0, ggml_nbytes(acc));
            } else {
                ggml_backend_tensor_memset(acc, 0, 0, ggml_nbytes(acc));
            }
        }
    }
}

void llama_kvq_fitter::get_progress(llama_kvq_fit_progress & out) const {
    out = {};
    out.n_coarse = store.n_coarse();
    out.n_chunk  = (store.n_coarse() + store.n_chunk_rows() - 1)/store.n_chunk_rows();
    out.n_host   = store.n_host();
    out.n_spill  = n_spill;
    out.failed   = failed;
    out.n_cancel = n_cancel;
    out.n_drop   = n_drop;
    out.t_job    = t_job;
    out.t_wall   = t_wall;
    out.t_take    = t_take;
    out.t_install = t_install;
    std::lock_guard<std::mutex> lk(m);
    for (const auto & jb : jobs) {
        if (jb->done) {
            out.n_ready++;
        } else {
            out.n_running++;
        }
    }
}
