#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/device.cuh>
#include <proposal/meta.cuh>

__global__ void k_sym_smem_pwarp(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ std::int32_t* const __restrict__ nnzs);

__global__ void k_sym_smem(const __grid_constant__ std::int32_t table_size,
                           const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ a_col,
                           const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ b_col,
                           const __grid_constant__ std::int32_t* const __restrict__ bins,
                           __grid_constant__ std::int32_t* const __restrict__ nnzs);

__global__ void k_sym_smem_max(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ nnzs,
    __grid_constant__ std::int32_t* const __restrict__ fail_bin,
    __grid_constant__ std::int32_t* const __restrict__ fail_bin_size);

__global__ void k_sym_global(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ tables,
    __grid_constant__ std::int32_t* const __restrict__ nnzs);

template<std::floating_point T>
void sym(const utils::DeviceCSR<T>& A,
         const utils::DeviceCSR<T>& B,
         utils::DeviceCSR<T>& C,
         Meta& meta,
         const Device& device) {
    NVTX3_FUNC_RANGE();

    static constexpr auto IdxByteSize = gsl::narrow_cast<std::int32_t>(
        sizeof(std::int32_t));

    // Handle the largest bin first as it may fail to fit in the shared memory
    const auto last_bin_idx = meta.n_bins - 1;
    const auto last_bin_size = meta.h_bin_sizes[last_bin_idx];
    std::int32_t h_fail_bin_size{};
    std::int32_t* d_fail_bin{};
    std::int32_t* d_fail_bin_size{};
    SPDLOG_DEBUG("Sym: bin {} size is {}", last_bin_idx, last_bin_size);
    if (last_bin_size > 0) {
        auto tmp_mem_size = (last_bin_size + 1) * IdxByteSize;
        if (tmp_mem_size <= gsl::narrow_cast<std::int32_t>(meta.cub_storage_size)) {
            SPDLOG_DEBUG("CUB storage is enough for fail bins: {} <= {}",
                         tmp_mem_size,
                         meta.cub_storage_size);
            d_fail_bin = static_cast<std::int32_t*>(meta.d_cub_storage);
        } else {
            SPDLOG_DEBUG("CUB storage is not enough for fail bins: {} > {}",
                         tmp_mem_size,
                         meta.cub_storage_size);
            auto* d_fail_ptr = utils::malloc_async<utils::Location::Device>(
                tmp_mem_size,
                meta.streams[last_bin_idx]);
            d_fail_bin = static_cast<std::int32_t*>(d_fail_ptr);
        }
        d_fail_bin_size = d_fail_bin + last_bin_size;
        utils::memset_async(d_fail_bin_size,
                            0,
                            sizeof(std::int32_t),
                            meta.streams[last_bin_idx]);
        const auto smem = (meta.sym_table_sizes[last_bin_idx] + 1) * IdxByteSize;
        utils::handle_cuda_error(
            cudaFuncSetAttribute(k_sym_smem_max,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem));
        utils::launch_kernel(k_sym_smem_max,
                             last_bin_size,
                             meta.sym_block_sizes[last_bin_idx],
                             smem,
                             meta.streams[last_bin_idx],
                             meta.sym_table_sizes[last_bin_idx],
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             meta.d_bins + meta.h_bin_offsets[last_bin_idx],
                             C.rpt,
                             d_fail_bin,
                             d_fail_bin_size);
        utils::memcpy_async(&h_fail_bin_size,
                            d_fail_bin_size,
                            sizeof(std::int32_t),
                            meta.streams[last_bin_idx]);
        utils::event_record(meta.events[0], meta.streams[last_bin_idx]);
    }

    // Handle the rest of the bins
    utils::handle_cuda_error(
        cudaFuncSetAttribute(k_sym_smem,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             meta.sym_table_sizes[meta.n_bins - 2] * IdxByteSize));
    for (std::int32_t i = meta.n_bins - 2; i > 0; i--) {
        SPDLOG_DEBUG("Sym: bin {} size is {}", i, meta.h_bin_sizes[i]);
        if (meta.h_bin_sizes[i] > 0) {
            utils::launch_kernel(k_sym_smem,
                                 meta.h_bin_sizes[i],
                                 meta.sym_block_sizes[i],
                                 meta.sym_table_sizes[i] * IdxByteSize,
                                 meta.streams[i],
                                 meta.sym_table_sizes[i],
                                 A.rpt,
                                 A.col,
                                 B.rpt,
                                 B.col,
                                 meta.d_bins + meta.h_bin_offsets[i],
                                 C.rpt);
        }
    }
    SPDLOG_DEBUG("Sym: bin 0 size is {}", meta.h_bin_sizes[0]);
    if (meta.h_bin_sizes[0] > 0) {
        const auto rows_per_block = device.optimal_block_size / PWARP_SIZE;
        const auto smem = (meta.sym_table_sizes[0] + rows_per_block) * IdxByteSize;
        utils::handle_cuda_error(
            cudaFuncSetAttribute(k_sym_smem_pwarp,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem));
        utils::launch_kernel(k_sym_smem_pwarp,
                             cuda::ceil_div(meta.h_bin_sizes[0], rows_per_block),
                             device.optimal_block_size,
                             smem,
                             meta.streams[0],
                             meta.sym_table_sizes[0] / rows_per_block,
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             meta.d_bins + meta.h_bin_offsets[0],
                             meta.h_bin_sizes[0],
                             C.rpt);
    }

    // Handle the fail bin
    if (last_bin_size > 0) {
        utils::event_sync(meta.events[0]);
        SPDLOG_DEBUG("Sym: fail bin size is {}", h_fail_bin_size);
        if (h_fail_bin_size > 0) {
            const auto table_size = *meta.h_max_row_nnz;
            meta.mem_pool_size = gsl::narrow_cast<std::size_t>(h_fail_bin_size
                                                               * table_size)
                                 * IdxByteSize;
            meta.d_mem_pool = utils::malloc_async(meta.mem_pool_size,
                                                  meta.streams[last_bin_idx]);
            utils::launch_kernel(k_sym_global,
                                 h_fail_bin_size,
                                 meta.sym_block_sizes[last_bin_idx],
                                 0,
                                 meta.streams[last_bin_idx],
                                 table_size,
                                 A.rpt,
                                 A.col,
                                 B.rpt,
                                 B.col,
                                 d_fail_bin,
                                 static_cast<std::int32_t*>(meta.d_mem_pool),
                                 C.rpt);
        }
    }

    // Wait for all bins to finish
    for (std::int32_t i = 0; i < meta.n_bins; i++)
        utils::stream_sync(meta.streams[i]);

    // No need to wait for the fail bin deallocation
    if (d_fail_bin != nullptr && d_fail_bin != meta.d_cub_storage)
        utils::free_async(d_fail_bin, meta.streams[last_bin_idx]);
}
