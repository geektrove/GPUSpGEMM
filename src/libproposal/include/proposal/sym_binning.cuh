#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/device.cuh>
#include <proposal/meta.cuh>

__global__ void k_sym_binning1(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ nips,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes);

__global__ void k_sym_binning2(
    const __grid_constant__ std::int32_t* const __restrict__ ranges,
    const __grid_constant__ std::int32_t n_bins,
    const __grid_constant__ std::int32_t* const __restrict__ nips,
    const __grid_constant__ std::int32_t m,
    const __grid_constant__ std::int32_t* const __restrict__ bin_offsets,
    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
    __grid_constant__ std::int32_t* const __restrict__ bins);

template<std::floating_point T>
void sym_binning(utils::DeviceCSR<T>& C, Meta& meta, const Device& device) {
    NVTX3_FUNC_RANGE();

    if (*meta.h_max_row_nnz <= meta.h_sym_bin_ranges[0]) {
        // If all rows fall into the smallest bin, we can skip the binning process
        // and directly assign the row indices to the smallest bin
        auto op = [d_bins = meta.d_bins] __device__(int i) {
            d_bins[i] = static_cast<std::int32_t>(i);
        };

        if constexpr (utils::IS_DEBUG) {
            size_t cub_requested{};
            cub::DeviceFor::Bulk(nullptr, cub_requested, C.m, op);
            assert(cub_requested <= meta.cub_storage_size);
        }

        // Perform iota operation to fill the smallest bin with row indices
        utils::handle_cuda_error(
            cub::DeviceFor::Bulk(meta.d_cub_storage, meta.cub_storage_size, C.m, op));

        // Set bin sizes and offsets
        meta.h_bin_sizes[0] = C.m;
        for (int i = 1; i < meta.n_bins; i++)
            meta.h_bin_sizes[i] = 0;
        meta.h_bin_offsets[0] = 0;
        for (int i = 1; i < meta.n_bins; i++)
            meta.h_bin_offsets[i] = C.m;

        utils::stream_sync();
        return;
    }

    // Perform full two-stage symbolic binning
    utils::memset_async(meta.d_bin_sizes, 0, meta.n_bins * sizeof(std::int32_t));

    const auto n_blocks = cuda::ceil_div(C.m, device.optimal_block_size);
    auto smem = meta.n_bins * sizeof(std::int32_t);
    k_sym_binning1<<<n_blocks, device.optimal_block_size, smem>>>(meta.d_sym_bin_ranges,
                                                                  meta.n_bins,
                                                                  C.rpt,
                                                                  C.m,
                                                                  meta.d_bin_sizes);

    utils::memcpy_async(meta.h_bin_sizes,
                        meta.d_bin_sizes,
                        meta.n_bins * sizeof(std::int32_t));
    utils::event_record(meta.events[0]);
    utils::memset_async(meta.d_bin_sizes, 0, meta.n_bins * sizeof(std::int32_t));

    utils::event_sync(meta.events[0]);
    meta.h_bin_offsets[0] = 0;
    for (int i = 0; i + 1 < meta.n_bins; i++)
        meta.h_bin_offsets[i + 1] = meta.h_bin_offsets[i] + meta.h_bin_sizes[i];

    utils::memcpy_async(meta.d_bin_offsets,
                        meta.h_bin_offsets,
                        meta.n_bins * sizeof(std::int32_t));

    smem = 2 * meta.n_bins * sizeof(std::int32_t);
    k_sym_binning2<<<n_blocks, device.optimal_block_size, smem>>>(meta.d_sym_bin_ranges,
                                                                  meta.n_bins,
                                                                  C.rpt,
                                                                  C.m,
                                                                  meta.d_bin_offsets,
                                                                  meta.d_bin_sizes,
                                                                  meta.d_bins);

    utils::stream_sync();
}
