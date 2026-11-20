#include "llama-kvq-pool.h"

#include <algorithm>
#include <stdexcept>

llama_kvq_pool::llama_kvq_pool(ggml_backend_buffer_type_t buft_host, const layout & lay, int32_t n_pages) :
    lay(lay), buft(buft_host), bufs(n_pages), base(n_pages, nullptr), offs(n_pages), nrow(n_pages) {
    if (lay.C <= 0 || n_pages <= 0 || (int64_t) n_pages*lay.C < lay.n_rows || (int64_t) (n_pages - 1)*lay.C >= lay.n_rows) {
        throw std::runtime_error("kvq pool: bad page layout");
    }
    for (int32_t p = 0; p < n_pages; ++p) {
        const size_t G = lay.G;
        const size_t R = nrow[p] = std::min(lay.C, lay.n_rows - p*lay.C);
        const size_t sz[5] = { G*R*lay.row_b, G*R*lay.kc_row, G*R*lay.vc_row, 2*G*lay.tinv_slot, 2*G*lay.taux_slot };
        size_t off = 0;
        for (int k = 0; k < 5; ++k) {
            offs[p][k] = off;
            off += sz[k];
        }
        offs[p][5] = off;
    }
    for (int32_t p = 0; p < n_pages; ++p) {
        if (!alloc(p)) {
            throw std::runtime_error("failed to allocate a page of the precise store and the host codes");
        }
    }
}

size_t llama_kvq_pool::bytes(int32_t page) const {
    return (size_t) lay.n_layers*offs[page][5];
}

size_t llama_kvq_pool::bytes_rows(int32_t page) const {
    return (size_t) lay.n_layers*offs[page][1];
}

size_t llama_kvq_pool::bytes_codes(int32_t page) const {
    return bytes(page) - bytes_rows(page);
}

bool llama_kvq_pool::alloc(int32_t page) {
    if (bufs[page]) {
        return true;
    }
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

void llama_kvq_pool::free(int32_t page) {
    bufs[page].reset();
    base[page] = nullptr;
}

bool llama_kvq_pool::pinned() const {
    for (const auto & b : bufs) {
        if (!b || ggml_backend_buffer_get_type(b.get()) == ggml_backend_cpu_buffer_type()) {
            return false;
        }
    }
    return !bufs.empty();
}
