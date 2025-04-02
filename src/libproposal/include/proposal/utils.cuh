#pragma once

#include <cstdint>
#include <type_traits>

#include <proposal/device.cuh>

inline auto get_smem_size(const Device& device, const std::int32_t n_blocks)
    -> std::int32_t {
    auto smem = device.smem_per_sm;                    // Total SMEM per SM
    smem -= n_blocks * device.smem_per_block_reserved; // Available SMEM per SM
    smem /= n_blocks;                                  // SMEM per block
    smem = std::min(smem, device.max_smem_per_block);  // Max SMEM per block (opt in)
    return smem;
}

template<std::int32_t begin, std::int32_t end, std::int32_t step, typename F>
inline void constexpr_for(F&& f) {
    static_assert(step == 1 || step == -1);
    static_assert((begin <= end && step > 0) || (begin >= end && step < 0));
    if constexpr (begin != end) {
        f(std::integral_constant<std::int32_t, begin>{});
        constexpr_for<begin + step, end, step>(std::forward<F>(f));
    }
}
