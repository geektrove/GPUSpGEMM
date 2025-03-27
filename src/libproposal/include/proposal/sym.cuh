#pragma once

#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdlib>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

namespace cg = cooperative_groups;

inline constexpr std::int32_t PWARP = 4;
inline constexpr std::int32_t PWARP_BLOCK_SIZE = 512;
inline constexpr auto PWARP_ROWS = PWARP_BLOCK_SIZE / PWARP;
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
    __builtin_assume(utils::ispow2(threads_per_row_a));
    __builtin_assume(utils::ispow2(threads_per_row_b));
    __builtin_assume(threads_per_row_a % threads_per_row_b == 0);

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

template<std::int32_t TABLE_SIZE>
__global__ void k_sym_shmem_pwarp(
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    __shared__ std::int32_t s_tables[PWARP_ROWS * TABLE_SIZE];
    __shared__ std::int32_t s_nnzs[PWARP_ROWS];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();

    for (auto i = block.thread_rank(); i < PWARP_ROWS * TABLE_SIZE;
         i += block.num_threads())
        s_tables[i] = -1;
    if (block.thread_rank() % PWARP == 0)
        s_nnzs[block.thread_rank() / PWARP] = 0;
    const auto row_id = gsl::narrow_cast<std::int32_t>(grid.thread_rank() / PWARP);
    if (row_id >= bin_size)
        return;
    block.sync();

    auto* s_table = s_tables + (block.thread_rank() / PWARP) * TABLE_SIZE;
    auto* s_nnz = s_nnzs + (block.thread_rank() / PWARP);
    const auto row = bins[row_id];
    fill_table(a_rpt,
               a_col,
               b_rpt,
               b_col,
               PWARP,
               1,
               block.thread_rank(),
               row,
               s_table,
               TABLE_SIZE,
               s_nnz);
    block.sync();

    if (block.thread_rank() % PWARP == 0)
        nnzs[row] = *s_nnz;
}

template<std::int32_t TABLE_SIZE>
__global__ void k_sym_shmem(
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    __shared__ std::int32_t s_table[TABLE_SIZE];
    __shared__ std::int32_t s_nnz;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    for (auto i = block.thread_rank(); i < TABLE_SIZE; i += block.num_threads())
        s_table[i] = -1;
    cg::invoke_one(block, [&] { s_nnz = 0; });
    block.sync();

    const auto row = bins[grid.block_rank()];
    fill_table(a_rpt,
               a_col,
               b_rpt,
               b_col,
               block.num_threads(),
               warp.num_threads(),
               block.thread_rank(),
               row,
               s_table,
               TABLE_SIZE,
               &s_nnz);
    block.sync();

    cg::invoke_one(block, [&] { nnzs[row] = s_nnz; });
}

template<std::floating_point T>
void sym(const utils::DeviceCSR<T>& A,
         const utils::DeviceCSR<T>& B,
         utils::DeviceCSR<T>& C,
         Meta& meta) {
    NVTX3_FUNC_RANGE();

    if (meta.h_bin_sizes[5] > 0) {
        SPDLOG_DEBUG("Sym: bin 5 size is {}", meta.h_bin_sizes[5]);
        k_sym_shmem<8192><<<meta.h_bin_sizes[5], 1024, 0, meta.streams[5]>>>(
            A.rpt,
            A.col,
            B.rpt,
            B.col,
            meta.d_bins + meta.h_bin_offsets[5],
            C.rpt);
    }
    if (meta.h_bin_sizes[0] > 0) {
        SPDLOG_DEBUG("Sym: bin 0 size is {}", meta.h_bin_sizes[0]);
        const auto n_blocks = cuda::ceil_div(meta.h_bin_sizes[0], PWARP_ROWS);
        k_sym_shmem_pwarp<32><<<n_blocks, PWARP_BLOCK_SIZE, 0, meta.streams[0]>>>(
            A.rpt,
            A.col,
            B.rpt,
            B.col,
            meta.d_bins + meta.h_bin_offsets[0],
            meta.h_bin_sizes[0],
            C.rpt);
    }

    for (const auto& stream : meta.streams)
        utils::stream_sync(stream);
}
