#pragma once

#include <concepts>
#include <cstdint>
#include <cstdlib>

#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/device.cuh>
#include <proposal/meta.cuh>
#include <proposal/utils.cuh>

__global__ void k_sym2_smem_pwarp(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ std::int32_t* const __restrict__ c_col);

__global__ void k_sym2_smem(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ c_col);

__global__ void k_sym2_smem_max(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ c_col);

__global__ void k_sym2_global(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ tables,
    __grid_constant__ std::int32_t* const __restrict__ c_col);

template<std::floating_point T>
void sym2(const utils::DeviceCSR<T>& A,
          const utils::DeviceCSR<T>& B,
          utils::DeviceCSR<T>& C,
          Meta& meta,
          const Device& device) {
    NVTX3_FUNC_RANGE();

    static constexpr auto IdxByteSize = gsl::narrow_cast<std::int32_t>(
        sizeof(std::int32_t));

    // Handle the global memory bin
    const auto gl_mem_bin_idx = meta.n_bins - 1;
    SPDLOG_DEBUG("Sym2: bin {} size is {}",
                 gl_mem_bin_idx,
                 meta.h_bin_sizes[gl_mem_bin_idx]);
    if (meta.h_bin_sizes[gl_mem_bin_idx] > 0) {
        utils::launch_kernel(k_sym2_global,
                             meta.h_bin_sizes[gl_mem_bin_idx],
                             meta.block_sizes[gl_mem_bin_idx],
                             0,
                             meta.streams[gl_mem_bin_idx],
                             meta.table_sizes[gl_mem_bin_idx],
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             C.rpt,
                             meta.d_bins + meta.h_bin_offsets[gl_mem_bin_idx],
                             static_cast<std::int32_t*>(meta.d_mem_pool),
                             C.col);
    }

    // Handle the max shared memory bin
    const auto max_smem_bin_idx = meta.n_bins - 2;
    SPDLOG_DEBUG("Sym2: bin {} size is {}",
                 max_smem_bin_idx,
                 meta.h_bin_sizes[max_smem_bin_idx]);
    if (meta.h_bin_sizes[max_smem_bin_idx] > 0) {
        const auto smem = (meta.table_sizes[max_smem_bin_idx] + 1) * IdxByteSize;
        utils::handle_cuda_error(
            cudaFuncSetAttribute(k_sym2_smem_max,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem));
        utils::launch_kernel(k_sym2_smem_max,
                             meta.h_bin_sizes[max_smem_bin_idx],
                             meta.block_sizes[max_smem_bin_idx],
                             smem,
                             meta.streams[max_smem_bin_idx],
                             meta.table_sizes[max_smem_bin_idx],
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             C.rpt,
                             meta.d_bins + meta.h_bin_offsets[max_smem_bin_idx],
                             C.col);
    }

    // Handle the rest of the bins
    utils::handle_cuda_error(
        cudaFuncSetAttribute(k_sym2_smem,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             meta.table_sizes[meta.n_bins - 3] * IdxByteSize));
    for (std::int32_t i = meta.n_bins - 3; i > 0; i--) {
        SPDLOG_DEBUG("Sym2: bin {} size is {}", i, meta.h_bin_sizes[i]);
        if (meta.h_bin_sizes[i] > 0) {
            utils::launch_kernel(k_sym2_smem,
                                 meta.h_bin_sizes[i],
                                 meta.block_sizes[i],
                                 meta.table_sizes[i] * IdxByteSize,
                                 meta.streams[i],
                                 meta.table_sizes[i],
                                 A.rpt,
                                 A.col,
                                 B.rpt,
                                 B.col,
                                 C.rpt,
                                 meta.d_bins + meta.h_bin_offsets[i],
                                 C.col);
        }
    }
    SPDLOG_DEBUG("Sym2: bin 0 size is {}", meta.h_bin_sizes[0]);
    if (meta.h_bin_sizes[0] > 0) {
        const auto rows_per_block = device.optimal_block_size / SYM_PWARP_SIZE;
        const auto smem = (meta.table_sizes[0] + rows_per_block) * IdxByteSize;
        utils::handle_cuda_error(
            cudaFuncSetAttribute(k_sym2_smem_pwarp,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem));
        utils::launch_kernel(k_sym2_smem_pwarp,
                             cuda::ceil_div(meta.h_bin_sizes[0], rows_per_block),
                             device.optimal_block_size,
                             smem,
                             meta.streams[0],
                             meta.table_sizes[0] / rows_per_block,
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             C.rpt,
                             meta.d_bins + meta.h_bin_offsets[0],
                             meta.h_bin_sizes[0],
                             C.col);
    }

    // Allocate C.val
    auto* val_ptr = utils::malloc_async(C.nnz * sizeof(T));
    C.val = reinterpret_cast<T*>(val_ptr);

    // Update the table sizes and ranges for numeric binning
    auto calculate_num_table_size = [&](std::int32_t n_blocks) {
        static constexpr auto ItemByteSize = gsl::narrow_cast<std::int32_t>(
            sizeof(std::int32_t) + sizeof(T));
        const auto smem = get_smem_size(device, n_blocks);
        auto table_size = smem / ItemByteSize;
        return table_size;
    };

    meta.table_sizes[0] = calculate_num_table_size(device.max_threads_per_sm
                                                   / meta.block_sizes[0]);
    {
        std::int32_t i = 1;
        for (; meta.block_sizes[i] < device.max_threads_per_block; i++)
            meta.table_sizes[i] = calculate_num_table_size(device.max_threads_per_sm
                                                           / meta.block_sizes[i]);
        for (auto n_blocks = device.max_threads_per_sm / device.max_threads_per_block;
             n_blocks > 0;
             n_blocks /= 2)
            meta.table_sizes[i++] = calculate_num_table_size(n_blocks);
        meta.table_sizes[i++] = calculate_num_table_size(1);
        meta.table_sizes[i] = *meta.h_max_row_nnz;
    }
    meta.h_bin_ranges[0] = meta.table_sizes[0] / (meta.block_sizes[0] / SYM_PWARP_SIZE);
    for (std::int32_t i = 1; i < meta.n_bins; i++)
        meta.h_bin_ranges[i] = meta.table_sizes[i];
    utils::memcpy_async(meta.d_bin_ranges,
                        meta.h_bin_ranges,
                        meta.n_bins * sizeof(std::int32_t));

    SPDLOG_DEBUG("Numeric bins");
    SPDLOG_DEBUG("* Bins with MAX_THREADS_PER_BLOCK are combined into one");
    SPDLOG_DEBUG("{:>12s} {:>12s} {:>12s} {:>12s}",
                 "Bin",
                 "Block size",
                 "Table size",
                 "Range");
    for (std::int32_t i = 0; i < meta.n_bins; i++) {
        SPDLOG_DEBUG("{:12d} {:12d} {:12d} {:12d}",
                     i,
                     meta.block_sizes[i],
                     meta.table_sizes[i],
                     meta.h_bin_ranges[i]);
    }

    // Wait for all bins to finish
    for (std::int32_t i = 0; i + 1 < meta.n_bins; i++)
        utils::stream_sync(meta.streams[i]);

    utils::stream_sync();
}
