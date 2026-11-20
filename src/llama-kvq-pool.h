#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// pool of chunk pages of the host side of the two-tier store
// a page is the C rows of one chunk of every attention layer and kv head in one pinned host buffer, per layer:
//   rows:  q8_0 rows [G][rows][K row, V row]
//   kc:    host codes of K [G][rows][kc_row bytes], vc: of V [G][rows][vc_row bytes] (the row layout of the device kc / vc)
//   tinv:  the tables of the chunk slot [2*G][tinv_slot bytes], taux: [2*G][taux_slot bytes]
// every page has C rows (the last page of n_rows is not shortened)
class llama_kvq_pool {
public:
    struct layout {
        int32_t n_layers = 0;
        int32_t G        = 0; // kv heads
        int32_t C        = 0; // rows of a chunk
        int32_t n_rows   = 0; // rows of the sequence, the pages are ceil(n_rows/C)
        size_t  row_b    = 0; // bytes of the K and V row of a position
        size_t  kc_row   = 0;
        size_t  vc_row   = 0;
        size_t  tinv_slot = 0;
        size_t  taux_slot = 0;
    };

    // allocates every page, throws if one fails
    llama_kvq_pool(ggml_backend_buffer_type_t buft_host, const layout & lay, int32_t n_pages);

    int32_t n_pages() const { return (int32_t) bufs.size(); }

    // rows of a page (C)
    int32_t rows(int32_t page) const { return nrow[page]; }

    // allocates the page (a pinned host buffer, or pageable if it fails), false if both fail
    bool alloc(int32_t page);
    void free(int32_t page);

    // bytes of a page, and of its rows and of its codes and tables
    size_t bytes(int32_t page) const;
    size_t bytes_rows(int32_t page) const;
    size_t bytes_codes(int32_t page) const;

    // the host buffer of every page is pinned
    bool pinned() const;

    // the buffer types of the allocated pages, for the memory report
    const ggml_backend_buffer_ptr & buffer(int32_t page) const { return bufs[page]; }

    // sections of layer li of an allocated page
    uint8_t * rows_ptr(int32_t page, int32_t li) const { return sec(page, li, 0); } // [G][rows(page)][2*ROW]
    uint8_t * kc_ptr  (int32_t page, int32_t li) const { return sec(page, li, 1); } // [G][rows(page)][kc_row]
    uint8_t * vc_ptr  (int32_t page, int32_t li) const { return sec(page, li, 2); } // [G][rows(page)][vc_row]
    uint8_t * tinv_ptr(int32_t page, int32_t li) const { return sec(page, li, 3); } // [2*G][tinv_slot]
    uint8_t * taux_ptr(int32_t page, int32_t li) const { return sec(page, li, 4); } // [2*G][taux_slot]

    const layout & get_layout() const { return lay; }

private:
    layout lay;
    ggml_backend_buffer_type_t buft = nullptr;
    std::vector<ggml_backend_buffer_ptr> bufs; // one per page
    std::vector<uint8_t *>               base; // their base addresses
    std::vector<std::array<size_t, 6>>   offs; // per page: the offsets of the sections in the block of a layer, then its size
    std::vector<int32_t>                 nrow; // per page: its rows

    uint8_t * sec(int32_t page, int32_t li, int k) const {
        return base[page] + (size_t) li*offs[page][5] + offs[page][k];
    }
};
