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

__global__ void k_sym_smem_max(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ nnzs,
    __grid_constant__ std::int32_t* const __restrict__ fail_bin,
    __grid_constant__ std::int32_t* const __restrict__ fail_bin_size) {
    extern __shared__ std::int32_t s_table[];
    auto* s_nnz = s_table + table_size;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    for (auto i = tib; i < table_size; i += block_size)
        s_table[i] = -1;
    cg::invoke_one(block, [&] { s_nnz = 0; });

    const auto row = bins[grid.block_rank()];
    const auto i_offset = tib / WARP_SIZE;
    const auto i_step = block_size / WARP_SIZE;
    const auto k_offset = tib % WARP_SIZE;
    const auto k_step = WARP_SIZE;
    const auto threshold = table_size * SYM_RANGE_RATIO;
    block.sync();

    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (*s_nnz <= threshold) {
                const auto old = atomicCAS_block(s_table + hash, -1, key);
                if (old == -1) {
                    atomicAdd_block(s_nnz, 1);
                    break;
                }
                if (old == key)
                    break;
                hash = hash + 1 < table_size ? hash + 1 : 0;
            }
        }
    }
    block.sync();

    cg::invoke_one(block, [&] {
        const auto nnz = *s_nnz;
        if (nnz <= threshold) {
            nnzs[row] = nnz;
        } else {
            const auto idx = atomicAdd(fail_bin_size, 1);
            fail_bin[idx] = row;
        }
    });
}

__global__ void k_sym_global(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ tables,
    __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    __shared__ std::int32_t s_nnz;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* table = tables + (static_cast<ptrdiff_t>(grid.block_rank()) * table_size);
    for (auto i = tib; i < table_size; i += block_size)
        table[i] = -1;
    cg::invoke_one(block, [&] { s_nnz = 0; });

    const auto row = bins[grid.block_rank()];
    const auto i_offset = tib / WARP_SIZE;
    const auto i_step = block_size / WARP_SIZE;
    const auto k_offset = tib % WARP_SIZE;
    const auto k_step = WARP_SIZE;
    block.sync();

    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (true) {
                const auto old = atomicCAS_block(table + hash, -1, key);
                if (old == -1) {
                    atomicAdd_block(&s_nnz, 1);
                    break;
                }
                if (old == key)
                    break;
                hash = hash + 1 < table_size ? hash + 1 : 0;
            }
        }
    }
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

    // Handle the fail bin
    SPDLOG_DEBUG("Sym: fail bin size is {}", h_fail_bin_size);
    if (last_bin_size > 0) {
        utils::event_sync(meta.events[0]);
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
        if (d_fail_bin != meta.d_cub_storage)
            utils::free_async(d_fail_bin, meta.streams[last_bin_idx]);
    }

    for (std::int32_t i = 0; i < meta.n_bins; i++)
        utils::stream_sync(meta.streams[i]);
}
