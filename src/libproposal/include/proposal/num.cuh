#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda/std/cstddef>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/device.cuh>
#include <proposal/meta.cuh>

namespace cg = cooperative_groups;

__forceinline__ __device__ auto find_key(const std::int32_t* const __restrict__ cols,
                                         const std::int32_t size,
                                         const std::int32_t key) -> std::int32_t {
    auto low = 0;
    auto high = size - 1;
    while (low < high) {
        const auto mid = low + ((high - low) / 2);
        if (cols[mid] < key)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

template<std::floating_point T>
__global__ void k_num_smem_pwarp(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ T* const __restrict__ a_val,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ T* const __restrict__ b_val,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ c_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ T* const __restrict__ c_val) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tile = cg::tiled_partition<NUM_PWARP_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto pib = utils::divpow2(tib, NUM_PWARP_SIZE);
    const auto tip = utils::modpow2(tib, NUM_PWARP_SIZE);
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    const auto rows_per_block = utils::divpow2(block_size, NUM_PWARP_SIZE);
    const auto total_table_size = table_size * rows_per_block;

    auto* s_vals_all = reinterpret_cast<T*>(smem);
    auto* s_cols_all = reinterpret_cast<std::int32_t*>(s_vals_all + total_table_size);
    auto* s_vals = s_vals_all + (pib * table_size);
    auto* s_cols = s_cols_all + static_cast<ptrdiff_t>(pib * table_size);

    const auto row_id = utils::divpow2(tig, NUM_PWARP_SIZE);
    if (row_id >= bin_size)
        return;
    const auto row = bins[row_id];
    const auto c_offset = c_rpt[row];
    const auto size = c_rpt[row + 1] - c_offset;

    cg::memcpy_async(tile, s_cols, c_col + c_offset, size * sizeof(std::int32_t));
    for (auto i = tip; i < size; i += NUM_PWARP_SIZE)
        s_vals[i] = 0;
    cg::wait(tile);
    tile.sync();

    for (auto i = a_rpt[row] + tip; i < a_rpt[row + 1]; i += NUM_PWARP_SIZE) {
        const auto a_value = a_val[i];
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow]; k < b_rpt[colrow + 1]; k++) {
            const auto idx = find_key(s_cols, size, b_col[k]);
            atomicAdd_block(s_vals + idx, a_value * b_val[k]);
        }
    }
    tile.sync();

    cg::memcpy_async(tile, c_val + c_offset, s_vals, size * sizeof(T));
    cg::wait(tile);
}

template<std::floating_point T>
__global__ void k_num_smem(const __grid_constant__ std::int32_t table_size,
                           const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ a_col,
                           const __grid_constant__ T* const __restrict__ a_val,
                           const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ b_col,
                           const __grid_constant__ T* const __restrict__ b_val,
                           const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ c_col,
                           const __grid_constant__ std::int32_t* const __restrict__ bins,
                           __grid_constant__ T* const __restrict__ c_val) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* s_vals = reinterpret_cast<T*>(smem);
    auto* s_cols = reinterpret_cast<std::int32_t*>(s_vals + table_size);

    const auto row = bins[grid.block_rank()];
    const auto c_offset = c_rpt[row];
    const auto size = c_rpt[row + 1] - c_offset;

    cg::memcpy_async(block, s_cols, c_col + c_offset, size * sizeof(std::int32_t));
    for (auto i = tib; i < size; i += block_size)
        s_vals[i] = 0;
    cg::wait(block);
    block.sync();

    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(block_size, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto a_value = a_val[i];
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto idx = find_key(s_cols, size, b_col[k]);
            atomicAdd_block(s_vals + idx, a_value * b_val[k]);
        }
    }
    block.sync();

    cg::memcpy_async(block, c_val + c_offset, s_vals, size * sizeof(T));
    cg::wait(block);
}

template<std::floating_point T>
__global__ void k_num_global(
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ T* const __restrict__ a_val,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ T* const __restrict__ b_val,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ c_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ T* const __restrict__ c_val) {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    const auto row = bins[grid.block_rank()];
    const auto c_offset = c_rpt[row];
    const auto size = c_rpt[row + 1] - c_offset;

    auto* vals = c_val + c_offset;
    const auto* cols = c_col + c_offset;
    for (auto i = tib; i < size; i += block_size)
        vals[i] = 0;

    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(block_size, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto a_value = a_val[i];
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto idx = find_key(cols, size, b_col[k]);
            atomicAdd_block(vals + idx, a_value * b_val[k]);
        }
    }
}

template<std::floating_point T>
void num(const utils::DeviceCSR<T>& A,
         const utils::DeviceCSR<T>& B,
         utils::DeviceCSR<T>& C,
         Meta& meta) {
    NVTX3_FUNC_RANGE();

    static constexpr auto IdxByteSize = gsl::narrow_cast<std::int32_t>(
        sizeof(std::int32_t));
    static constexpr auto ValueByteSize = gsl::narrow_cast<std::int32_t>(sizeof(T));
    static constexpr auto ItemByteSize = IdxByteSize + ValueByteSize;

    // Handle the global memory bin
    const auto gl_mem_bin_idx = meta.n_bins - 1;
    SPDLOG_DEBUG("Num: bin {} size is {}",
                 gl_mem_bin_idx,
                 meta.h_bin_sizes[gl_mem_bin_idx]);
    if (meta.h_bin_sizes[gl_mem_bin_idx] > 0) {
        utils::launch_kernel(k_num_global<T>,
                             meta.h_bin_sizes[gl_mem_bin_idx],
                             meta.block_sizes[gl_mem_bin_idx],
                             0,
                             meta.streams[gl_mem_bin_idx],
                             A.rpt,
                             A.col,
                             A.val,
                             B.rpt,
                             B.col,
                             B.val,
                             C.rpt,
                             C.col,
                             meta.d_bins + meta.h_bin_offsets[gl_mem_bin_idx],
                             C.val);
    }

    // Handle the rest of the bins
    utils::handle_cuda_error(
        cudaFuncSetAttribute(k_num_smem<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             meta.table_sizes[meta.n_bins - 2] * ItemByteSize));
    for (std::int32_t i = meta.n_bins - 2; i > 0; i--) {
        SPDLOG_DEBUG("Num: bin {} size is {}", i, meta.h_bin_sizes[i]);
        if (meta.h_bin_sizes[i] > 0) {
            utils::launch_kernel(k_num_smem<T>,
                                 meta.h_bin_sizes[i],
                                 meta.block_sizes[i],
                                 meta.table_sizes[i] * ItemByteSize,
                                 meta.streams[i],
                                 meta.table_sizes[i],
                                 A.rpt,
                                 A.col,
                                 A.val,
                                 B.rpt,
                                 B.col,
                                 B.val,
                                 C.rpt,
                                 C.col,
                                 meta.d_bins + meta.h_bin_offsets[i],
                                 C.val);
        }
    }
    SPDLOG_DEBUG("Num: bin 0 size is {}", meta.h_bin_sizes[0]);
    if (meta.h_bin_sizes[0] > 0) {
        const auto rows_per_block = utils::divpow2(meta.block_sizes[0], NUM_PWARP_SIZE);
        const auto smem = meta.table_sizes[0] * ItemByteSize;
        utils::handle_cuda_error(
            cudaFuncSetAttribute(k_num_smem_pwarp<T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem));
        utils::launch_kernel(k_num_smem_pwarp<T>,
                             cuda::ceil_div(meta.h_bin_sizes[0], rows_per_block),
                             meta.block_sizes[0],
                             smem,
                             meta.streams[0],
                             utils::divpow2(meta.table_sizes[0], rows_per_block),
                             A.rpt,
                             A.col,
                             A.val,
                             B.rpt,
                             B.col,
                             B.val,
                             C.rpt,
                             C.col,
                             meta.d_bins + meta.h_bin_offsets[0],
                             meta.h_bin_sizes[0],
                             C.val);
    }

    // Wait for all bins to finish
    for (std::int32_t i = 0; i + 1 < meta.n_bins; i++)
        utils::stream_sync(meta.streams[i]);
}
