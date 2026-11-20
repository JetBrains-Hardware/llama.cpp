#include "llama-kvq-pool.h"

#include <algorithm>
#include <stdexcept>

llama_kvq_pool::llama_kvq_pool(ggml_backend_buffer_type_t buft_host, const layout & lay, int32_t n_pages, bool prealloc, size_t max_bytes, size_t keep_bytes) :
    lay(lay), buft(buft_host), max_b(max_bytes), keep_b(prealloc ? SIZE_MAX : keep_bytes), bufs(n_pages), base(n_pages, nullptr), offs(n_pages), nrow(n_pages), kept(n_pages), hold(n_pages, 0),
    st(new std::atomic<int>[n_pages > 0 ? n_pages : 1]), want(n_pages > 0 ? n_pages : 0, 0), drop(n_pages > 0 ? n_pages : 0, 0) {
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

int32_t llama_kvq_pool::n_shared() const {
    int32_t n = 0;
    for (int32_t h : hold) {
        n += h > 1;
    }
    return n;
}

bool llama_kvq_pool::pick(int32_t n, int32_t * out) const {
    // free pages that have host memory first, then those queued for it, then those without, last those that are being freed (the lowest ids
    // within each: with no host memory kept, a page that is just freed is not waited for)
    int32_t k = 0;
    for (int cls = 0; cls < 4 && k < n; ++cls) {
        for (int32_t p = 0; p < n_pages() && k < n; ++p) {
            if (hold[p] != 0) {
                continue;
            }
            const int s = st[p].load();
            const int c = s == ST_READY ? 0 : s == ST_QUEUED ? 1 : s == ST_RELEASING ? 3 : 2;
            if (c == cls) {
                out[k++] = p;
            }
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

int32_t llama_kvq_pool::trim() {
    if (keep_b == SIZE_MAX) {
        return 0;
    }
    size_t kept_b = 0;
    int32_t n_rel = 0;
    std::lock_guard<std::mutex> lk(m);
    for (int32_t p = 0; p < n_pages(); ++p) {
        const int s = st[p].load();
        if (hold[p] != 0 || (s != ST_READY && s != ST_QUEUED)) {
            continue;
        }
        if (kept_b + bytes(p) <= keep_b) {
            kept_b += bytes(p);
            continue;
        }
        if (s == ST_QUEUED) {
            // queued ahead, nobody took it: the thread drops it
            drop[p] = 1;
            continue;
        }
        // the caller synchronized the streams: nothing reads or writes the page any more
        st[p].store(ST_RELEASING);
        want[p] = 0;
        todo_free.push_back(p);
        n_rel++;
    }
    if (n_rel > 0) {
        if (!thr.joinable()) {
            thr = std::thread([this]() { run(); });
        }
        cv_work.notify_one();
    }
    return n_rel;
}

void llama_kvq_pool::wait_frees() {
    std::unique_lock<std::mutex> lk(m);
    cv_done.wait(lk, [&]() {
        for (int32_t p = 0; p < n_pages(); ++p) {
            if (st[p].load() == ST_RELEASING) {
                return false;
            }
        }
        return true;
    });
}

void llama_kvq_pool::queue(int32_t page) {
    std::lock_guard<std::mutex> lk(m);
    const int s = st[page].load();
    if (s == ST_QUEUED) {
        drop[page] = 0;
        return;
    }
    if (s == ST_RELEASING) {
        // the thread allocates it again when the free is done
        want[page] = 1;
        return;
    }
    if (s != ST_NONE && s != ST_FAILED) {
        return;
    }
    drop[page] = 0;
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
    cv_done.wait(lk, [&]() { const int s = st[page].load(); return s == ST_READY || s == ST_FAILED; });
    stat.n_wait++;
    stat.t_wait += (ggml_time_us() - t0)*1e-6;
    return st[page].load() == ST_READY;
}

void llama_kvq_pool::queue_ahead() {
    int32_t p = -1;
    if (!pick(1, &p) || st[p].load() != ST_NONE) {
        return; // none free, or the page the next reservation takes has its host memory or is on its way
    }
    if (host_fits(&p, 1)) {
        queue(p);
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

void llama_kvq_pool::free_now(int32_t page) {
    bufs[page].reset();
    base[page] = nullptr;
}

void llama_kvq_pool::run() {
    std::unique_lock<std::mutex> lk(m);
    while (true) {
        cv_work.wait(lk, [&]() { return stop || !todo.empty() || !todo_free.empty(); });
        if (stop) {
            return;
        }
        // allocations go before frees: a sequence may wait for a page
        std::deque<int32_t> & q = todo.empty() ? todo_free : todo;
        const int32_t p = q.front();
        q.pop_front();
        if (st[p].load() == ST_RELEASING) {
            // cudaFreeHost synchronizes the device: here, not on the thread that decodes
            lk.unlock();
            const int64_t t0 = ggml_time_us();
            free_now(p);
            const double dt = (ggml_time_us() - t0)*1e-6;
            lk.lock();
            stat.n_free++;
            stat.t_free += dt;
            n_commit--;
            if (want[p]) {
                want[p] = 0;
                st[p].store(ST_QUEUED);
                n_commit++;
                todo.push_back(p);
            } else {
                st[p].store(ST_NONE);
            }
            cv_done.notify_all();
            continue;
        }
        if (drop[p]) {
            drop[p] = 0;
            n_commit--;
            st[p].store(ST_NONE);
            cv_done.notify_all();
            continue;
        }
        lk.unlock();
        const int64_t t0 = ggml_time_us();
        const bool ok = alloc_now(p);
        const double dt = (ggml_time_us() - t0)*1e-6;
        lk.lock();
        stat.n_alloc++;
        stat.t_alloc += dt;
        if (ok && drop[p]) {
            // taken by nobody while it was allocated
            drop[p] = 0;
            lk.unlock();
            const int64_t t1 = ggml_time_us();
            free_now(p);
            const double df = (ggml_time_us() - t1)*1e-6;
            lk.lock();
            stat.n_free++;
            stat.t_free += df;
            n_commit--;
            st[p].store(ST_NONE);
            cv_done.notify_all();
            continue;
        }
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
