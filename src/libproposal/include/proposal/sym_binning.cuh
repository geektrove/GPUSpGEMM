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

#include <proposal/binning.cuh>
#include <proposal/device.cuh>
#include <proposal/meta.cuh>

template<std::floating_point T>
void sym_binning(utils::DeviceCSR<T>& C, Meta& meta, const Device& device) {
    NVTX3_FUNC_RANGE();

    if (*meta.h_max_row_nnz <= meta.h_sym_bin_ranges[0]) {
        // If all rows fall into the smallest bin, we can skip the binning process
        // and directly assign the row indices to the smallest bin
        small_binning(C.m, meta);
        return;
    }

    // Perform full two-stage symbolic binning
    utils::memset_async(meta.d_bin_sizes, 0, meta.n_bins * sizeof(std::int32_t));

    utils::launch_kernel(k_binning1,
                         cuda::ceil_div(C.m, device.optimal_block_size),
                         device.optimal_block_size,
                         meta.n_bins * sizeof(std::int32_t),
                         cudaStreamDefault,
                         meta.d_sym_bin_ranges,
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

    utils::launch_kernel(k_binning2,
                         cuda::ceil_div(C.m, device.optimal_block_size),
                         device.optimal_block_size,
                         2 * meta.n_bins * sizeof(std::int32_t),
                         cudaStreamDefault,
                         meta.d_sym_bin_ranges,
                         meta.n_bins,
                         C.rpt,
                         C.m,
                         meta.d_bin_offsets,
                         meta.d_bin_sizes,
                         meta.d_bins);

    utils::stream_sync();

    if constexpr (utils::IS_DEBUG) {
        auto op = [d_bins = meta.d_bins, m = C.m] __device__(int i) {
            assert(d_bins[i] >= 0 && d_bins[i] < m);
        };
        size_t cub_requested{};
        cub::DeviceFor::Bulk(nullptr, cub_requested, C.m, op);
        assert(cub_requested <= meta.cub_storage_size);
        utils::handle_cuda_error(
            cub::DeviceFor::Bulk(meta.d_cub_storage, meta.cub_storage_size, C.m, op));
    }
}

template<std::floating_point T>
void sym_binning2(utils::DeviceCSR<T>& C, Meta& meta, const Device& device) {
    NVTX3_FUNC_RANGE();

    utils::handle_cuda_error(cub::DeviceReduce::Max(meta.d_cub_storage,
                                                    meta.cub_storage_size,
                                                    C.rpt,
                                                    meta.d_max_row_nnz,
                                                    C.m));
    utils::memcpy_async(meta.h_max_row_nnz,
                        meta.d_max_row_nnz,
                        sizeof(*meta.h_max_row_nnz));
    utils::handle_cuda_error(cub::DeviceReduce::Sum(meta.d_cub_storage,
                                                    meta.cub_storage_size,
                                                    C.rpt,
                                                    meta.d_total_nnz,
                                                    C.m));
    utils::memcpy_async(meta.h_total_nnz, meta.d_total_nnz, sizeof(*meta.h_total_nnz));
    utils::stream_sync();

    C.nnz = *meta.h_total_nnz;
    auto* col_ptr = utils::malloc_async(C.nnz * sizeof(*C.col), meta.streams[0]);
    C.col = static_cast<std::int32_t*>(col_ptr);

    sym_binning(C, meta, device);

    utils::stream_sync(meta.streams[0]);

    utils::handle_cuda_error(cub::DeviceScan::ExclusiveSum(meta.d_cub_storage,
                                                           meta.cub_storage_size,
                                                           C.rpt,
                                                           C.rpt,
                                                           C.m + 1));

    utils::stream_sync();
}
