#pragma once

#include <proposal/device.cuh>

inline auto get_smem_size(const Device& device, const std::int32_t n_blocks)
    -> std::int32_t {
    auto smem = device.smem_per_sm;                    // Total SMEM per SM
    smem -= n_blocks * device.smem_per_block_reserved; // Available SMEM per SM
    smem /= n_blocks;                                  // SMEM per block
    smem = std::min(smem, device.max_smem_per_block);  // Max SMEM per block (opt in)
    return smem;
}
