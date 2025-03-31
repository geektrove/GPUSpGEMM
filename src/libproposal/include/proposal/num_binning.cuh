#pragma once

#include <cassert>
#include <concepts>
#include <cstdlib>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/binning.cuh>
#include <proposal/device.cuh>
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

template<std::floating_point T>
void num_binning(utils::DeviceCSR<T>& C, Meta& meta, const Device& device) {
    NVTX3_FUNC_RANGE();

    if (*meta.h_max_row_nnz <= meta.h_bin_ranges[0]) {
        // If all rows fall into the smallest bin, we can skip the binning process
        // and directly assign the row indices to the smallest bin
        small_binning(C.m, meta);
        return;
    }

    // Perform full two-stage numeric binning
    utils::memset_async(meta.d_bin_sizes, 0, meta.n_bins * sizeof(std::int32_t));

    utils::launch_kernel(k_num_binning1,
                         cuda::ceil_div(C.m, device.optimal_block_size),
                         device.optimal_block_size,
                         meta.n_bins * sizeof(std::int32_t),
                         cudaStreamDefault,
                         meta.d_bin_ranges,
                         meta.n_bins,
                         C.rpt,
                         C.m,
                         meta.d_bin_sizes);

    utils::memcpy_async(meta.h_bin_sizes,
                        meta.d_bin_sizes,
                        meta.n_bins * sizeof(std::int32_t));
    utils::event_record(meta.events[0]);
    utils::memset_async(meta.d_bin_sizes, 0, meta.n_bins * sizeof(std::int32_t));

    SPDLOG_DEBUG("Numeric bin sizes");
    SPDLOG_DEBUG("{:>12s} {:>12s}", "Bin", "Size");
    for (std::int32_t i = 0; i < meta.n_bins; i++) {
        SPDLOG_DEBUG("{:12d} {:12d}", i, meta.h_bin_sizes[i]);
    }

    utils::event_sync(meta.events[0]);
    meta.h_bin_offsets[0] = 0;
    for (int i = 0; i + 1 < meta.n_bins; i++)
        meta.h_bin_offsets[i + 1] = meta.h_bin_offsets[i] + meta.h_bin_sizes[i];

    utils::memcpy_async(meta.d_bin_offsets,
                        meta.h_bin_offsets,
                        meta.n_bins * sizeof(std::int32_t));

    utils::launch_kernel(k_num_binning2,
                         cuda::ceil_div(C.m, device.optimal_block_size),
                         device.optimal_block_size,
                         2 * meta.n_bins * sizeof(std::int32_t),
                         cudaStreamDefault,
                         meta.d_bin_ranges,
                         meta.n_bins,
                         C.rpt,
                         C.m,
                         meta.d_bin_offsets,
                         meta.d_bin_sizes,
                         meta.d_bins);

    utils::stream_sync();
}
