#include <cassert>
#include <cstdint>

#include <cooperative_groups.h>
#include <gsl/gsl-lite.hpp>

#include <proposal/num_binning.cuh>

namespace cg = cooperative_groups;

namespace {

__forceinline__ __device__ auto find_bin(const std::int32_t* const __restrict__ ranges,
                                         const std::int32_t n_bins,
                                         const std::int32_t x) -> std::int32_t {
    for (std::int32_t i = 0; i < n_bins; i++) {
        if (x <= ranges[i])
            return i;
    }
    assert(false);
    __builtin_unreachable();
}

} // namespace

__global__ void k_num_binning1(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ values,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes) {
    extern __shared__ std::int32_t s_bin_sizes[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (tib < n_bins)
        s_bin_sizes[tib] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    if (row < m) {
        const auto bin_idx = find_bin(ranges, n_bins, values[row + 1] - values[row]);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (tib < n_bins)
        atomicAdd(bin_sizes + tib, s_bin_sizes[tib]);
}

__global__ void k_num_binning2(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ values,
    const __grid_constant__ std::int32_t m,
    const __grid_constant__ std::int32_t* const __restrict__ bin_offsets,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
    __grid_constant__ std::int32_t* const __restrict__ bins) {
    extern __shared__ std::int32_t smem[];
    auto* s_bin_sizes = smem;
    auto* s_bin_offsets = s_bin_sizes + n_bins;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (tib < n_bins)
        s_bin_sizes[tib] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    std::int32_t bin_idx = 0;
    if (row < m) {
        bin_idx = find_bin(ranges, n_bins, values[row + 1] - values[row]);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (tib < n_bins) {
        s_bin_offsets[tib] = atomicAdd(bin_sizes + tib, s_bin_sizes[tib]);
        s_bin_offsets[tib] += bin_offsets[tib];
        s_bin_sizes[tib] = 0;
    }
    block.sync();

    if (row < m) {
        const auto index = atomicAdd_block(s_bin_sizes + bin_idx, 1);
        bins[s_bin_offsets[bin_idx] + index] = row;
    }
}
