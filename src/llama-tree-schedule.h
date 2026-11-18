#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

struct llama_tree_schedule {
    std::array<int32_t, 96> steps{}; // row, parent slot, destination slot
    int32_t slots = 1;
};

inline llama_tree_schedule llama_tree_schedule_make(const int32_t * parents, int32_t n) {
    assert(n >= 1 && n <= 32 && parents[0] == -1);
    llama_tree_schedule result;
    std::array<int32_t, 32> children{}, remaining{}, node_slot{}, stack{};
    std::array<bool, 32> used{};
    node_slot.fill(-1);
    for (int32_t i = 1; i < n; ++i) {
        assert(parents[i] >= 0 && parents[i] < i);
        ++children[parents[i]];
    }
    remaining = children;
    int32_t pending = 1, count = 0, maximum = 0;
    stack[0] = 0;
    while (pending) {
        const int32_t row = stack[--pending];
        const int32_t parent = parents[row];
        const int32_t read = parent < 0 ? -1 : node_slot[parent];
        assert(parent < 0 || read >= 0);
        if (parent >= 0 && --remaining[parent] == 0 && node_slot[parent] >= 0) {
            // The last child can overwrite its parent's slot after loading it.
            used[node_slot[parent]] = false;
        }
        int32_t write = -1;
        if (children[row] > 0) {
            write = std::find(used.begin(), used.end(), false) - used.begin();
            assert(write < 32);
            used[write] = true;
            node_slot[row] = write;
            maximum = std::max(maximum, write + 1);
        }
        result.steps[3 * count] = row;
        result.steps[3 * count + 1] = read;
        result.steps[3 * count + 2] = write;
        ++count;
        for (int32_t i = n - 1; i > row; --i) {
            if (parents[i] == row) {
                stack[pending++] = i;
            }
        }
    }
    assert(count == n);
    while (result.slots < maximum) {
        result.slots *= 2;
    }
    return result;
}
