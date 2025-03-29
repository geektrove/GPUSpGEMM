#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/device.cuh>
#include <proposal/meta.cuh>

namespace cg = cooperative_groups;

inline constexpr std::int32_t WARP_SIZE = 32;
inline constexpr std::int32_t HASH_SCALE = 107;

__forceinline__ __device__ auto insert_to_table(std::int32_t* const __restrict__ table,
                                                const std::int32_t size,
                                                const std::int32_t key) -> bool {
    auto hash = (key * HASH_SCALE) % size;
    while (true) {
        const auto old = atomicCAS_block(table + hash, -1, key);
        if (old == -1)
            return true;
        if (old == key)
            return false;
        hash = (hash + 1) % size;
    }
}

__forceinline__ __device__ auto fill_table(const std::int32_t* const __restrict__ a_rpt,
                                           const std::int32_t* const __restrict__ a_col,
                                           const std::int32_t* const __restrict__ b_rpt,
                                           const std::int32_t* const __restrict__ b_col,
                                           const std::int32_t threads_per_row_a,
                                           const std::int32_t threads_per_row_b,
                                           const std::int32_t thread_idx,
                                           const std::int32_t row,
                                           std::int32_t* const __restrict__ table,
                                           const std::int32_t size,
                                           std::int32_t* const __restrict__ nnz) {
    assert(utils::ispow2(threads_per_row_a));
    assert(utils::ispow2(threads_per_row_b));
    assert(threads_per_row_a % threads_per_row_b == 0);

    const auto i_offset = (thread_idx % threads_per_row_a) / threads_per_row_b;
    const auto i_step = threads_per_row_a / threads_per_row_b;
    const auto k_offset = thread_idx % threads_per_row_b;
    const auto k_step = threads_per_row_b;
    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            if (insert_to_table(table, size, b_col[k]))
                atomicAdd_block(nnz, 1);
        }
    }
}

__global__ void k_sym_smem_pwarp(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    extern __shared__ std::int32_t smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto& tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    const auto rows_per_block = block_size / PWARP;
    const auto total_table_size = table_size * rows_per_block;

    auto* s_tables = smem;
    auto* s_nnzs = smem + total_table_size;

    for (auto i = tib; i < total_table_size; i += block_size)
        s_tables[i] = -1;
    if (tib % PWARP == 0)
        s_nnzs[tib / PWARP] = 0;
    const auto row_id = tig / PWARP;
    if (row_id >= bin_size)
        return;
    block.sync();

    auto* s_table = s_tables + (static_cast<ptrdiff_t>((tib / PWARP) * table_size));
    auto* s_nnz = s_nnzs + (tib / PWARP);
    const auto row = bins[row_id];
    fill_table(a_rpt,
               a_col,
               b_rpt,
               b_col,
               PWARP,
               1,
               tib,
               row,
               s_table,
               table_size,
               s_nnz);
    block.sync();

    if (block.thread_rank() % PWARP == 0)
        nnzs[row] = *s_nnz;
}

__global__ void k_sym_smem(const __grid_constant__ std::int32_t table_size,
                           const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ a_col,
                           const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ b_col,
                           const __grid_constant__ std::int32_t* const __restrict__ bins,
                           __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    extern __shared__ std::int32_t s_table[];
    __shared__ std::int32_t s_nnz;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    for (auto i = tib; i < table_size; i += block_size)
        s_table[i] = -1;
    cg::invoke_one(block, [&] { s_nnz = 0; });
    block.sync();

    const auto row = bins[grid.block_rank()];
    fill_table(a_rpt,
               a_col,
               b_rpt,
               b_col,
               block_size,
               WARP_SIZE,
               tib,
               row,
               s_table,
               table_size,
               &s_nnz);
    block.sync();

    cg::invoke_one(block, [&] { nnzs[row] = s_nnz; });
}

template<std::floating_point T>
void sym(const utils::DeviceCSR<T>& A,
         const utils::DeviceCSR<T>& B,
         utils::DeviceCSR<T>& C,
         Meta& meta,
         const Device& device) {
    NVTX3_FUNC_RANGE();

    static constexpr auto IdxByteSize = gsl::narrow_cast<std::int32_t>(
        sizeof(std::int32_t));

    utils::handle_cuda_error(
        cudaFuncSetAttribute(k_sym_smem,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             meta.sym_table_sizes[meta.n_bins - 2] * IdxByteSize));
    for (std::int32_t i = meta.n_bins - 2; i > 0; i--) {
        if (meta.h_bin_sizes[i] > 0) {
            SPDLOG_DEBUG("Sym: bin {} size is {}", i, meta.h_bin_sizes[i]);
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
    if (meta.h_bin_sizes[0] > 0) {
        SPDLOG_DEBUG("Sym: bin 0 size is {}", meta.h_bin_sizes[0]);
        const auto rows_per_block = device.optimal_block_size / PWARP;
        utils::launch_kernel(k_sym_smem_pwarp,
                             cuda::ceil_div(meta.h_bin_sizes[0], rows_per_block),
                             device.optimal_block_size,
                             (meta.sym_table_sizes[0] + rows_per_block) * IdxByteSize,
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

    for (std::int32_t i = 0; i < meta.n_bins; i++)
        utils::stream_sync(meta.streams[i]);
}
