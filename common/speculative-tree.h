#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

// Counter-based draws keep a token's noise independent of tree shape and batching.
inline uint64_t common_tree_mix(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

inline double common_tree_noise(uint64_t seed, uint32_t position, uint32_t token) {
    const uint64_t base = common_tree_mix(common_tree_mix(seed + UINT64_C(0x9e3779b97f4a7c15)) ^
                                          (uint64_t(position) * UINT64_C(0xd1b54a32d192ed03)));
    const uint32_t top = uint32_t(common_tree_mix(base ^ token) >> 40);
    // Match TensorFold's uniform24 F32 arithmetic, including its clamp below 1.
    const float u = (float(top) + 0.5f) * (1.0f / 16777216.0f);
    const double uniform = std::min(u, std::nextafter(1.0f, 0.0f));
    return -std::log(-std::log(uniform));
}

struct common_tree_node {
    int32_t token;
    int32_t parent;  // verification row: 0 is the pending token
    double own;     // conditional log-chance used by the proposal policy
};

// Packed rows retain the chain scores, then the separate unary and edge values.
// Best-first policy follows TensorFold v1.2.5 cuda_draft_tree: four children,
// selector weight 0.6, noise weight 0.7 and softmax scale 1.5.
inline std::vector<common_tree_node> common_tree_grow(
        const float * lattice, int rows, int width, int top_k, int budget,
        uint64_t seed, uint32_t anchor_position, double temperature) {
    assert(rows > 1 && top_k > 0 && top_k <= 20 && budget >= 0 && budget <= 31);
    assert(width >= 2 * top_k * (top_k + 1));
    struct candidate {
        double cost;
        int32_t parent;
        int32_t token;
        int32_t depth;
        int32_t index;
        double own;
    };
    std::vector<candidate> open;
    std::vector<common_tree_node> result;
    open.reserve(4 * (budget + 1));
    result.reserve(budget);
    int depth = 1, parent = 0, predecessor = 0;
    double path = 0;
    while (true) {
        if (depth < rows) {
            const float * row = lattice + size_t(depth) * width;
            const float * unary = row + top_k * (top_k + 1);
            const float * edge = unary + top_k + predecessor * top_k;
            std::vector<double> lc(top_k);
            double most = -INFINITY;
            for (int j = 0; j < top_k; ++j) {
                lc[j] = ((double(unary[j]) + 0.6 * double(edge[j])) / std::max(temperature, 1e-6) +
                         0.7 * common_tree_noise(seed, anchor_position + depth, uint32_t(row[j]))) / 1.5;
                most = std::max(most, lc[j]);
            }
            double total = 0;
            for (double v : lc) {
                total += std::exp(v - most);
            }
            const double log_total = most + std::log(total);
            for (double & v : lc) {
                v -= log_total;
            }
            std::vector<int> order(top_k);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return lc[a] > lc[b] || (lc[a] == lc[b] && a < b);
            });
            for (int j = 0; j < std::min(4, top_k); ++j) {
                const int k = order[j];
                if (!std::isfinite(lc[k])) {
                    break;
                }
                open.push_back({path - lc[k], parent, int32_t(row[k]), depth, k, lc[k]});
            }
        }
        if (int(result.size()) == budget || open.empty()) {
            break;
        }
        const auto best = std::min_element(open.begin(), open.end(), [](const candidate & a, const candidate & b) {
            if (a.cost != b.cost) return a.cost < b.cost;
            if (a.parent != b.parent) return a.parent < b.parent;
            if (a.token != b.token) return a.token < b.token;
            return a.depth < b.depth;
        });
        const candidate node = *best;
        *best = open.back();
        open.pop_back();
        result.push_back({node.token, node.parent, node.own});
        parent = int(result.size());
        predecessor = node.index;
        depth = node.depth + 1;
        path = node.cost;
    }
    return result;
}
