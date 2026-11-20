#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// pool of chunk pages of the host side of the two-tier store
// a page is the C rows of one chunk of every attention layer and kv head in one pinned host buffer, per layer:
//   rows:  q8_0 rows [G][rows][K row, V row]
//   kc:    host codes of K [G][rows][kc_row bytes], vc: of V [G][rows][vc_row bytes] (the row layout of the device kc / vc)
//   tab:   the tables of the chunk slot [2*G][tab_slot bytes], a slot is the inverse table bytes then the aux floats (the device tab)
// and, in pageable host memory, the kept sums of the chunk (struct sums): they are a part of the fitted chunk, so a page carries them
// every page has C rows (the last page of n_rows is not shortened)
//
// a page has a holder count (the sequences whose page table has it; 0 or 1 until pages are shared): a page is free at 0
// the host memory of a page is allocated on demand, by a background thread: queue() at the reservation of a page (and for the
// next free page, ahead), wait() before the page is used; a free page keeps its host memory (the free list is the free pages
// that have it), so the host memory only grows to the high-water mark of the pages in use
// max_bytes: cap of the host memory of the pages (allocated and queued), 0: none
class llama_kvq_pool {
public:
    struct layout {
        int32_t n_layers = 0;
        int32_t G        = 0; // kv heads
        int32_t C        = 0; // rows of a chunk
        int32_t n_rows   = 0; // rows of the longest sequence, it needs ceil(n_rows/C) pages
        size_t  row_b    = 0; // bytes of the K and V row of a position
        size_t  kc_row   = 0;
        size_t  vc_row   = 0;
        size_t  tab_slot = 0;
    };

    // prealloc: allocates every page here, throws if one fails
    llama_kvq_pool(ggml_backend_buffer_type_t buft_host, const layout & lay, int32_t n_pages, bool prealloc, size_t max_bytes);
    ~llama_kvq_pool();

    int32_t n_pages() const { return (int32_t) bufs.size(); }

    // rows of a page (C)
    int32_t rows(int32_t page) const { return nrow[page]; }

    // holders
    int32_t holders(int32_t page) const { return hold[page]; }
    int32_t n_free() const;

    // the n free pages with the lowest ids into out, false if there are fewer
    bool    pick(int32_t n, int32_t * out) const;

    // the host memory the n pages need, new: those that have no host memory yet; false if it would pass max_bytes
    bool    host_fits(const int32_t * pg, int32_t n) const;

    // a holder takes the page (its kept sums are reset) / gives it back (at 0 the kept sums are reset, the host memory stays)
    void    take(int32_t page);
    void    give(int32_t page);

    // the host memory of the page: queues the allocation, if it has none
    void    queue(int32_t page);
    // waits for the allocation (counted when it blocks), false if it failed
    bool    wait(int32_t page);
    // queues the free page with the lowest id that has no host memory (the next one a sequence takes), if the cap allows
    void    queue_ahead();

    // bytes of a page, and of its rows and of its codes and tables
    size_t bytes(int32_t page) const;
    size_t bytes_rows(int32_t page) const;
    size_t bytes_codes(int32_t page) const;

    // pages whose host memory is allocated, its bytes
    int32_t n_alloc() const;
    size_t  host_bytes() const;
    size_t  max_host_bytes() const { return max_b; }

    // the host memory of every allocated page is pinned
    bool pinned() const;

    // allocations, and the waits that blocked
    struct stats {
        int32_t n_alloc  = 0;   // allocations done
        double  t_alloc  = 0.0; // s in them (the thread's time)
        int32_t n_wait   = 0;   // wait() calls that blocked
        double  t_wait   = 0.0; // s blocked
    };
    stats get_stats() const;
    void  reset_wait_stats();

    // the buffer of an allocated page, for the memory report
    const ggml_backend_buffer_ptr & buffer(int32_t page) const { return bufs[page]; }
    bool allocated(int32_t page) const { return st[page].load(std::memory_order_acquire) == ST_READY; }

    // sections of layer li of an allocated page
    uint8_t * rows_ptr(int32_t page, int32_t li) const { return sec(page, li, 0); } // [G][rows(page)][2*ROW]
    uint8_t * kc_ptr  (int32_t page, int32_t li) const { return sec(page, li, 1); } // [G][rows(page)][kc_row]
    uint8_t * vc_ptr  (int32_t page, int32_t li) const { return sec(page, li, 2); } // [G][rows(page)][vc_row]
    uint8_t * tab_ptr (int32_t page, int32_t li) const { return sec(page, li, 3); } // [2*G][tab_slot]

    // the statistics of the queries of a complete chunk, read when it was complete (llama_kvq_fitter); empty: none
    struct sums {
        std::vector<float> gram;         // [n_layer][G][D*D]
        std::vector<float> gsum;         // [n_layer][n_gate], per-chunk V only
        double             t_snap = 0.0; // when they were read
    };
    sums       & chunk_sums(int32_t page)       { return kept[page]; }
    const sums & chunk_sums(int32_t page) const { return kept[page]; }

    const layout & get_layout() const { return lay; }

private:
    enum { ST_NONE = 0, ST_QUEUED = 1, ST_READY = 2, ST_FAILED = 3 };

    layout lay;
    ggml_backend_buffer_type_t buft = nullptr;
    size_t max_b = 0;
    std::vector<ggml_backend_buffer_ptr> bufs; // one per page, set when the page is ready
    std::vector<uint8_t *>               base; // their base addresses
    std::vector<std::array<size_t, 5>>   offs; // per page: the offsets of the sections in the block of a layer, then its size
    std::vector<int32_t>                 nrow; // per page: its rows
    std::vector<sums>                    kept; // per page: the kept sums of its chunk
    std::vector<int32_t>                 hold; // per page: holders
    std::unique_ptr<std::atomic<int>[]>  st;   // per page: ST_*

    mutable std::mutex      m;
    std::condition_variable cv_work;
    std::condition_variable cv_done;
    std::deque<int32_t>     todo;
    bool                    stop = false;
    int32_t                 n_commit = 0;      // pages that are queued or ready
    stats                   stat;
    std::thread             thr;

    bool alloc_now(int32_t page);
    void run();

    uint8_t * sec(int32_t page, int32_t li, int k) const {
        GGML_ASSERT(page >= 0 && base[page] != nullptr && "kvq pool: the page has no host memory");
        return base[page] + (size_t) li*offs[page][4] + offs[page][k];
    }
};
