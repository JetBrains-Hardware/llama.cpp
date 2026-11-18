#include "speculative-tree.h"
#include "../src/llama-tree-schedule.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static void test_schedule() {
    std::mt19937 rng(42);
    for (int n = 1; n <= 32; ++n) {
        for (int trial = 0; trial < 2000; ++trial) {
            std::array<int32_t, 32> parents{};
            std::array<uint32_t, 32> expected{}, slots{};
            parents[0] = -1;
            expected[0] = 1;
            for (int i = 1; i < n; ++i) {
                parents[i] = trial % 1000 == 0 ? i - 1 : trial % 1000 == 1 ? 0 : rng() % i;
                expected[i] = expected[parents[i]] | (uint32_t(1) << i);
            }
            const auto schedule = llama_tree_schedule_make(parents.data(), n);
            if (schedule.slots > 16) {
                std::abort();
            }
            uint32_t seen = 0;
            for (int i = 0; i < n; ++i) {
                const int row = schedule.steps[3 * i];
                const int read = schedule.steps[3 * i + 1];
                const int write = schedule.steps[3 * i + 2];
                if (row < 0 || row >= n || read < -1 || read >= schedule.slots || write < -1 || write >= schedule.slots) {
                    std::abort();
                }
                const uint32_t bit = uint32_t(1) << row;
                if (seen & bit) {
                    std::abort();
                }
                seen |= bit;
                const uint32_t value = (read < 0 ? 0 : slots[read]) | bit;
                if (value != expected[row]) {
                    std::abort();
                }
                if (write >= 0) {
                    slots[write] = value;
                }
            }
        }
    }
    puts("PASS: 64000 tree schedules preserve every ancestor path");
}

static int draw(uint32_t seed, uint32_t position, const std::array<double, 3> & p) {
    int result = 0;
    double best = -INFINITY;
    for (int i = 0; i < 3; ++i) {
        const double noise = common_tree_noise(seed, position, i);
        if (!std::isfinite(noise)) {
            std::abort();
        }
        const double score = std::log(p[i]) + noise;
        if (score > best) {
            best = score;
            result = i;
        }
    }
    return result;
}

static void test_selector_weight() {
    // Cancel known noise so this isolates unary + 0.6*edge from unary + edge.
    constexpr int k = 2, width = 12;
    std::array<float, 2 * width> lattice{};
    float * row = lattice.data() + width;
    row[0] = 10; row[1] = 20;
    row[6] = -0.7 * common_tree_noise(42, 101, 10);
    row[7] = 1.5 - 0.7 * common_tree_noise(42, 101, 20);
    row[8] = 2; row[9] = 0;
    const auto nodes = common_tree_grow(lattice.data(), 2, width, k, 2, 42, 100, 1);
    if (nodes.size() != 2 || nodes[0].token != 20 || nodes[0].parent != 0 || nodes[1].parent != 0) {
        std::abort();
    }
    const double wrong_a = row[6] + row[8] + 0.7 * common_tree_noise(42, 101, 10);
    const double wrong_b = row[7] + row[9] + 0.7 * common_tree_noise(42, 101, 20);
    if (wrong_a <= wrong_b) {
        std::abort();
    }
    puts("PASS: selector weighting regression distinguishes 0.6 from 1.0");
}

int main() {
    test_selector_weight();
    test_schedule();
    const std::array<double, 3> p = {0.2, 0.3, 0.5};
    const std::array<std::array<double, 3>, 3> next = {{{0.1, 0.8, 0.1}, {0.9, 0.05, 0.05}, {0.2, 0.2, 0.6}}};
    constexpr uint32_t trials = 200000;
    std::array<std::array<uint32_t, 3>, 3> counts{};
    for (uint32_t seed = 0; seed < trials; ++seed) {
        const int first = draw(seed, 101, p);
        const int second = draw(seed, 102, next[first]);
        ++counts[first][second];
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double expected = p[i] * next[i][j];
            const double observed = double(counts[i][j]) / trials;
            const double tolerance = 6 * std::sqrt(expected * (1 - expected) / trials);
            printf("%d,%d expected=%g observed=%g tolerance=%g\n", i, j, expected, observed, tolerance);
            if (std::fabs(observed - expected) > tolerance) {
                return 1;
            }
        }
    }
    printf("PASS: two-position categorical distribution, %u seeds\n", trials);
    return 0;
}
