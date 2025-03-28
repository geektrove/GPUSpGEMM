#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/functional>
#include <gsl/gsl-lite.hpp>

#include <proposal/sym_binning.cuh>

namespace cg = cooperative_groups;

namespace {

__forceinline__ __device__ auto find_bin(const std::int32_t* const __restrict__ ranges,
                                         const std::int32_t n_bins,
                                         const std::int32_t x) -> std::int32_t {
    for (int i = 0; i < n_bins; i++) {
        if (x <= ranges[i])
            return i;
    }
    __builtin_unreachable();
}

} // namespace

__global__ void k_sym_binning1(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ nips,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes) {
    extern __shared__ std::int32_t s_bin_sizes[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto bid = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (bid < n_bins)
        s_bin_sizes[block.thread_rank()] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    if (row < m) {
        const auto bin_idx = find_bin(ranges, n_bins, nips[row]);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (bid < n_bins)
        atomicAdd(bin_sizes + bid, s_bin_sizes[bid]);
}

__global__ void k_sym_binning2(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ nips,
    const __grid_constant__ std::int32_t m,
    const __grid_constant__ std::int32_t* const __restrict__ bin_offsets,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
    __grid_constant__ std::int32_t* const __restrict__ bins) {
    extern __shared__ std::int32_t smem[];
    auto* s_bin_sizes = smem;
    auto* s_bin_offsets = s_bin_sizes + n_bins;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto bid = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (bid < n_bins)
        s_bin_sizes[block.thread_rank()] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    int bin_idx = 0;
    if (row < m) {
        bin_idx = find_bin(ranges, n_bins, nips[row]);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (bid < n_bins) {
        s_bin_offsets[bid] = atomicAdd(bin_sizes + bid, s_bin_sizes[bid]);
        s_bin_offsets[bid] += bin_offsets[bid];
        s_bin_sizes[bid] = 0;
    }
    block.sync();

    if (bid < m) {
        const auto index = atomicAdd_block(s_bin_sizes + bin_idx, 1);
        bins[s_bin_offsets[bin_idx] + index] = row;
    }
}
