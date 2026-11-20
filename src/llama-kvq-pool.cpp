#include "llama-kvq-pool.h"

#include <algorithm>
#include <stdexcept>

llama_kvq_pool::llama_kvq_pool(ggml_backend_buffer_type_t buft_host, const layout & lay, int32_t n_pages, bool prealloc, size_t max_bytes) :
    lay(lay), buft(buft_host), max_b(max_bytes), bufs(n_pages), base(n_pages, nullptr), offs(n_pages), nrow(n_pages), kept(n_pages), hold(n_pages, 0),
    st(new std::atomic<int>[n_pages > 0 ? n_pages : 1]) {
    if (lay.C <= 0 || n_pages <= 0 || (int64_t) n_pages*lay.C < lay.n_rows) {
        throw std::runtime_error("kvq pool: bad page layout");
    }
    for (int32_t p = 0; p < n_pages; ++p) {
        st[p].store(ST_NONE);
        const size_t G = lay.G;
        const size_t R = nrow[p] = lay.C;
        const size_t sz[4] = { G*R*lay.row_b, G*R*lay.kc_row, G*R*lay.vc_row, 2*G*lay.tab_slot };
        size_t off = 0;
        for (int k = 0; k < 4; ++k) {
            offs[p][k] = off;
            off += sz[k];
        }
        offs[p][4] = off;
    }
    if (prealloc) {
        for (int32_t p = 0; p < n_pages; ++p) {
            if (!alloc_now(p)) {
                throw std::runtime_error("failed to allocate a page of the precise store and the host codes");
            }
            st[p].store(ST_READY, std::memory_order_release);
            n_commit++;
        }
    }
}

llama_kvq_pool::~llama_kvq_pool() {
    {
        std::lock_guard<std::mutex> lk(m);
        stop = true;
    }
    cv_work.notify_all();
    if (thr.joinable()) {
        thr.join();
    }
}

size_t llama_kvq_pool::bytes(int32_t page) const {
    return (size_t) lay.n_layers*offs[page][4];
}

size_t llama_kvq_pool::bytes_rows(int32_t page) const {
    return (size_t) lay.n_layers*offs[page][1];
}

size_t llama_kvq_pool::bytes_codes(int32_t page) const {
    return bytes(page) - bytes_rows(page);
}

int32_t llama_kvq_pool::n_free() const {
    int32_t n = 0;
    for (int32_t h : hold) {
        n += h == 0;
    }
    return n;
}

bool llama_kvq_pool::pick(int32_t n, int32_t * out) const {
    int32_t k = 0;
    for (int32_t p = 0; p < n_pages() && k < n; ++p) {
        if (hold[p] == 0) {
            out[k++] = p;
        }
    }
    return k == n;
}

bool llama_kvq_pool::host_fits(const int32_t * pg, int32_t n) const {
    if (max_b == 0) {
        return true;
    }
    int32_t k = 0;
    for (int32_t i = 0; i < n; ++i) {
        const int s = st[pg[i]].load();
        k += s == ST_NONE || s == ST_FAILED;
    }
    std::lock_guard<std::mutex> lk(m);
    return (size_t) (n_commit + k)*bytes(0) <= max_b;
}

void llama_kvq_pool::take(int32_t page) {
    GGML_ASSERT(page >= 0 && page < n_pages());
    hold[page]++;
    if (hold[page] == 1) {
        kept[page] = sums();
    }
}

void llama_kvq_pool::give(int32_t page) {
    GGML_ASSERT(page >= 0 && page < n_pages() && hold[page] > 0);
    if (--hold[page] == 0) {
        kept[page] = sums();
    }
}

void llama_kvq_pool::queue(int32_t page) {
    std::lock_guard<std::mutex> lk(m);
    const int s = st[page].load();
    if (s != ST_NONE && s != ST_FAILED) {
        return;
    }
    st[page].store(ST_QUEUED);
    n_commit++;
    todo.push_back(page);
    if (!thr.joinable()) {
        thr = std::thread([this]() { run(); });
    }
    cv_work.notify_one();
}

bool llama_kvq_pool::wait(int32_t page) {
    if (st[page].load(std::memory_order_acquire) == ST_READY) {
        return true;
    }
    queue(page);
    const int64_t t0 = ggml_time_us();
    std::unique_lock<std::mutex> lk(m);
    cv_done.wait(lk, [&]() { return st[page].load() >= ST_READY; });
    stat.n_wait++;
    stat.t_wait += (ggml_time_us() - t0)*1e-6;
    return st[page].load() == ST_READY;
}

void llama_kvq_pool::queue_ahead() {
    for (int32_t p = 0; p < n_pages(); ++p) {
        if (hold[p] == 0 && st[p].load() == ST_NONE) {
            if (host_fits(&p, 1)) {
                queue(p);
            }
            return;
        }
    }
}

// allocates the pinned host buffer of the page, or pageable memory if that fails
bool llama_kvq_pool::alloc_now(int32_t page) {
    const size_t sz = bytes(page);
    ggml_backend_buffer_t b = nullptr;
    if (sz <= ggml_backend_buft_get_max_size(buft)) {
        b = ggml_backend_buft_alloc_buffer(buft, sz);
    }
    if (b == nullptr) {
        b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), sz);
    }
    if (b == nullptr) {
        return false;
    }
    bufs[page].reset(b);
    base[page] = (uint8_t *) ggml_backend_buffer_get_base(b);
    return true;
}

void llama_kvq_pool::run() {
    std::unique_lock<std::mutex> lk(m);
    while (true) {
        cv_work.wait(lk, [&]() { return stop || !todo.empty(); });
        if (stop) {
            return;
        }
        const int32_t p = todo.front();
        todo.pop_front();
        lk.unlock();
        const int64_t t0 = ggml_time_us();
        const bool ok = alloc_now(p);
        const double dt = (ggml_time_us() - t0)*1e-6;
        lk.lock();
        stat.n_alloc++;
        stat.t_alloc += dt;
        if (!ok) {
            n_commit--;
        }
        st[p].store(ok ? ST_READY : ST_FAILED, std::memory_order_release);
        cv_done.notify_all();
    }
}

int32_t llama_kvq_pool::n_alloc() const {
    int32_t n = 0;
    for (int32_t p = 0; p < n_pages(); ++p) {
        n += st[p].load(std::memory_order_acquire) == ST_READY;
    }
    return n;
}

size_t llama_kvq_pool::host_bytes() const {
    return (size_t) n_alloc()*bytes(0);
}

bool llama_kvq_pool::pinned() const {
    bool any = false;
    for (int32_t p = 0; p < n_pages(); ++p) {
        if (st[p].load(std::memory_order_acquire) == ST_READY) {
            any = true;
            if (ggml_backend_buffer_get_type(bufs[p].get()) == ggml_backend_cpu_buffer_type()) {
                return false;
            }
        }
    }
    return any || buft != ggml_backend_cpu_buffer_type();
}

llama_kvq_pool::stats llama_kvq_pool::get_stats() const {
    std::lock_guard<std::mutex> lk(m);
    return stat;
}

void llama_kvq_pool::reset_wait_stats() {
    std::lock_guard<std::mutex> lk(m);
    stat.n_wait = 0;
    stat.t_wait = 0.0;
}
