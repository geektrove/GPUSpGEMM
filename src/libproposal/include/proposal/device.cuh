#pragma once

#include <cstdint>

struct Device {
    std::int32_t n_sm{};
    std::int32_t max_threads_per_sm{};
    std::int32_t max_threads_per_block{};
    std::int32_t max_blocks_per_sm{};
    std::int32_t smem_per_sm{};
    std::int32_t smem_per_block_reserved{};
    std::int32_t max_smem_per_block{};

    std::int32_t optimal_block_size{};
    std::int32_t min_block_size{};
};
