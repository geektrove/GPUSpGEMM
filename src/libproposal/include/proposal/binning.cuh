#pragma once

#include <cstdint>
#include <cuda/std/concepts>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <gsl/gsl-lite.hpp>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

namespace cg = cooperative_groups;

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

template<typename GetValueF>
__global__ void k_binning1(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
    GetValueF get_value) {
    extern __shared__ std::int32_t s_bin_sizes[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (tib < n_bins)
        s_bin_sizes[tib] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    if (row < m) {
        const auto bin_idx = find_bin(ranges, n_bins, get_value(row));
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (tib < n_bins)
        atomicAdd(bin_sizes + tib, s_bin_sizes[tib]);
}

template<typename GetValueF>
__global__ void k_binning2(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t m,
    const __grid_constant__ std::int32_t* const __restrict__ bin_offsets,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
    __grid_constant__ std::int32_t* const __restrict__ bins,
    GetValueF get_value) {
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
        bin_idx = find_bin(ranges, n_bins, get_value(row));
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

inline void small_binning(const std::int32_t m, Meta& meta) {
    auto op = [d_bins = meta.d_bins] __device__(int i) {
        d_bins[i] = static_cast<std::int32_t>(i);
    };

    // Perform iota operation to fill the smallest bin with row indices
    utils::handle_cuda_error(
        cub::DeviceFor::Bulk(meta.d_cub_storage, meta.cub_storage_size, m, op));

    // Set bin sizes and offsets
    meta.h_bin_sizes[0] = m;
    for (int i = 1; i < meta.n_bins; i++)
        meta.h_bin_sizes[i] = 0;
    meta.h_bin_offsets[0] = 0;
    for (int i = 1; i < meta.n_bins; i++)
        meta.h_bin_offsets[i] = m;

    utils::stream_sync();
}
