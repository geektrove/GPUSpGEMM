#pragma once

#include <cassert>
#include <concepts>
#include <cstdint>

#include <cub/cub.cuh>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/binning.cuh>
#include <proposal/device.cuh>
#include <proposal/meta.cuh>

__global__ void k_num_binning1(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ values,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes);

__global__ void k_num_binning2(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ values,
    const __grid_constant__ std::int32_t m,
    const __grid_constant__ std::int32_t* const __restrict__ bin_offsets,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
    __grid_constant__ std::int32_t* const __restrict__ bins);

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
