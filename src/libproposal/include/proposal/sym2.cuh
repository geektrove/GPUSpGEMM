#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/functional>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>
#include <proposal/parameters.cuh>
#include <proposal/utils.cuh>

namespace proposal {

namespace cg = cooperative_groups;

template<std::int32_t BLOCK_SIZE, std::int32_t PWARP_SIZE, std::int32_t TOTAL_TABLE_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_sym2_smem_pwarp(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ a_col,
                           const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ b_col,
                           const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ bins,
                           const __grid_constant__ std::int32_t bin_size,
                           __grid_constant__ std::int32_t* const __restrict__ c_col) {
    static constexpr auto ROWS_PER_BLOCK = utils::divpow2(BLOCK_SIZE, PWARP_SIZE);
    static constexpr auto TABLE_SIZE = utils::divpow2(TOTAL_TABLE_SIZE, ROWS_PER_BLOCK);
    static constexpr auto ITEMS_PER_THREAD = TABLE_SIZE / PWARP_SIZE;

    using SortT = cub::WarpMergeSort<std::uint32_t, ITEMS_PER_THREAD, PWARP_SIZE>;
    using SortTempStorageT = typename SortT::TempStorage;
    using StoreT = cub::
        WarpStore<std::int32_t, ITEMS_PER_THREAD, cub::WARP_STORE_TRANSPOSE, PWARP_SIZE>;
    using StoreTempStorageT = typename StoreT::TempStorage;

    static constexpr auto SMEM = cuda::std::max(
        {TOTAL_TABLE_SIZE * sizeof(std::int32_t),
         ROWS_PER_BLOCK * sizeof(SortTempStorageT),
         ROWS_PER_BLOCK * sizeof(StoreTempStorageT)});
    static constexpr auto PWARP_SMEM = SMEM / ROWS_PER_BLOCK;

    static_assert(utils::ispow2(BLOCK_SIZE));
    static_assert(utils::ispow2(TOTAL_TABLE_SIZE));
    static_assert(PWARP_SIZE <= WARP_SIZE);
    static_assert(BLOCK_SIZE % PWARP_SIZE == 0);
    static_assert(TOTAL_TABLE_SIZE % ROWS_PER_BLOCK == 0);
#ifndef NDEBUG
    assert(dynamic_smem_size() == SMEM);
#endif

    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto pwarp = cg::tiled_partition<PWARP_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto tip = utils::modpow2(tib, PWARP_SIZE);
    const auto pib = utils::divpow2(tib, PWARP_SIZE);

    // Get row id
    const auto row_id = utils::divpow2(tig, PWARP_SIZE);
    if (row_id >= bin_size)
        return;

    // Initialize hash table
    auto* p_smem = smem + (pib * PWARP_SMEM);
    auto* p_table = reinterpret_cast<std::int32_t*>(p_smem);
    for (auto i = tip; i < TABLE_SIZE; i += PWARP_SIZE)
        p_table[i] = HASH_EMPTY;
    pwarp.sync();

    // Aggregate column indices in hash table
    const auto row = bins[row_id];
    assert(c_rpt[row + 1] - c_rpt[row] <= TABLE_SIZE);
    const auto a_rpt_end = a_rpt[row + 1];
    for (auto i = a_rpt[row] + tip; i < a_rpt_end; i += PWARP_SIZE) {
        const auto colrow = a_col[i];
        const auto b_rpt_end = b_rpt[colrow + 1];
        for (auto k = b_rpt[colrow]; k < b_rpt_end; k++) {
            const auto key = b_col[k];
            auto hash = utils::modpow2(key * HASH_SCALE, TABLE_SIZE);
            while (true) {
                const auto old = atomicCAS_block(p_table + hash, HASH_EMPTY, key);
                if (old == HASH_EMPTY || old == key)
                    break;
                hash = utils::modpow2(hash + 1, TABLE_SIZE);
            }
        }
    }
    pwarp.sync();

    // Prepare cols for sorting
    std::int32_t l_cols[ITEMS_PER_THREAD];
    for (auto i = tip; i < TABLE_SIZE; i += PWARP_SIZE)
        l_cols[utils::divpow2(i, PWARP_SIZE)] = p_table[i];
    pwarp.sync();

    // Sort using merge sort
    auto* l_ucols = reinterpret_cast<std::uint32_t(*)[ITEMS_PER_THREAD]>(&l_cols);
    auto* sort_storage = reinterpret_cast<SortTempStorageT*>(p_smem);
    SortT(*sort_storage).Sort(*l_ucols, cuda::std::less<std::uint32_t>{});
    pwarp.sync();

    // Write to C.col
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;
    auto* store_storage = reinterpret_cast<StoreTempStorageT*>(p_smem);
    StoreT(*store_storage).Store(&c_col[c_offset], l_cols, nnz);
}

template<std::int32_t BLOCK_SIZE, std::int32_t TABLE_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_sym2_smem(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                     const __grid_constant__ std::int32_t* const __restrict__ a_col,
                     const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                     const __grid_constant__ std::int32_t* const __restrict__ b_col,
                     const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
                     const __grid_constant__ std::int32_t* const __restrict__ bins,
                     __grid_constant__ std::int32_t* const __restrict__ c_col) {
    static constexpr auto ITEMS_PER_THREAD = TABLE_SIZE / BLOCK_SIZE;

    using SortT = cub::BlockMergeSort<std::uint32_t, BLOCK_SIZE, ITEMS_PER_THREAD>;
    using SortTempStorageT = typename SortT::TempStorage;
    using StoreT = cub::BlockStore<std::int32_t,
                                   BLOCK_SIZE,
                                   ITEMS_PER_THREAD,
                                   cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using StoreTempStorageT = typename StoreT::TempStorage;

    static_assert(utils::ispow2(BLOCK_SIZE));
    static_assert(TABLE_SIZE % BLOCK_SIZE == 0);
#ifndef NDEBUG
    static constexpr auto SMEM = cuda::std::max({TABLE_SIZE * sizeof(std::int32_t),
                                                 sizeof(SortTempStorageT),
                                                 sizeof(StoreTempStorageT)});
    assert(dynamic_smem_size() == SMEM);
#endif

    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    // Initialize hash table
    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    for (auto i = tib; i < TABLE_SIZE; i += BLOCK_SIZE)
        s_table[i] = HASH_EMPTY;
    block.sync();

    // Aggregate column indices in hash table
    const auto row = bins[grid.block_rank()];
    assert(c_rpt[row + 1] - c_rpt[row] <= TABLE_SIZE);
    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    const auto a_rpt_end = a_rpt[row + 1];
    for (auto i = a_rpt[row] + i_offset; i < a_rpt_end; i += i_step) {
        const auto colrow = a_col[i];
        const auto b_rpt_end = b_rpt[colrow + 1];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt_end; k += k_step) {
            const auto key = b_col[k];
            auto hash = key * HASH_SCALE;
            if constexpr (utils::ispow2(TABLE_SIZE))
                hash = utils::modpow2(hash, TABLE_SIZE);
            else
                hash = hash % TABLE_SIZE;
            while (true) {
                const auto old = atomicCAS_block(s_table + hash, HASH_EMPTY, key);
                if (old == HASH_EMPTY || old == key)
                    break;
                if constexpr (utils::ispow2(TABLE_SIZE))
                    hash = utils::modpow2(hash + 1, TABLE_SIZE);
                else
                    hash = hash + 1 < TABLE_SIZE ? hash + 1 : 0;
            }
        }
    }
    block.sync();

    // Prepare cols for sorting
    std::int32_t l_cols[ITEMS_PER_THREAD];
    for (auto i = tib; i < TABLE_SIZE; i += BLOCK_SIZE)
        l_cols[utils::divpow2(i, BLOCK_SIZE)] = s_table[i];
    block.sync();

    // Sort using merge sort
    auto* l_ucols = reinterpret_cast<std::uint32_t(*)[ITEMS_PER_THREAD]>(&l_cols);
    auto* sort_storage = reinterpret_cast<SortTempStorageT*>(smem);
    SortT(*sort_storage).Sort(*l_ucols, cuda::std::less<std::uint32_t>{});
    block.sync();

    // Write to C.col
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;
    auto* store_storage = reinterpret_cast<StoreTempStorageT*>(smem);
    StoreT(*store_storage).Store(&c_col[c_offset], l_cols, nnz);
}

template<std::int32_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_sym2_global(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                       const __grid_constant__ std::int32_t* const __restrict__ a_col,
                       const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                       const __grid_constant__ std::int32_t* const __restrict__ b_col,
                       const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
                       const __grid_constant__ std::int32_t* const __restrict__ bins,
                       const __grid_constant__ std::int32_t common_table_size,
                       __grid_constant__ std::int32_t* const __restrict__ tables,
                       __grid_constant__ std::int32_t* const __restrict__ c_col) {
    static_assert(utils::ispow2(BLOCK_SIZE));

    __shared__ std::int32_t s_offset;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto big = gsl::narrow_cast<std::int32_t>(grid.block_rank());

    const auto row = bins[big];
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;

    // Initialize hash table
    auto* table = tables + static_cast<ptrdiff_t>(big * common_table_size);
    const auto table_size = gsl::narrow_cast<std::int32_t>(nnz / SYM2_RANGE_RATIO);
    assert(table_size <= common_table_size);
    for (auto i = tib; i < table_size; i += BLOCK_SIZE)
        table[i] = HASH_EMPTY;
    cg::invoke_one(block, [&] { s_offset = 0; });
    block.sync();

    // Aggregate column indices in hash table
    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    const auto a_rpt_end = a_rpt[row + 1];
    for (auto i = a_rpt[row] + i_offset; i < a_rpt_end; i += i_step) {
        const auto colrow = a_col[i];
        const auto b_rpt_end = b_rpt[colrow + 1];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt_end; k += k_step) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (true) {
                const auto old = atomicCAS_block(table + hash, HASH_EMPTY, key);
                if (old == HASH_EMPTY || old == key)
                    break;
                hash = hash + 1 < table_size ? hash + 1 : 0;
            }
        }
    }
    block.sync();

    // Condense
    for (auto offset = 0; offset < table_size; offset += BLOCK_SIZE) {
        const auto i = offset + tib;
        const auto col = i < table_size ? table[i] : HASH_EMPTY;
        block.sync();
        if (col != HASH_EMPTY)
            table[atomicAdd_block(&s_offset, 1)] = col;
    }
    block.sync();

    // Write to C.col
    for (auto i = tib; i < nnz; i += BLOCK_SIZE) {
        const auto col = table[i];
        std::int32_t n_less = 0;
        for (std::int32_t j = 0; j < nnz; j++) {
            if (table[j] < col)
                n_less++;
        }
        c_col[c_offset + n_less] = col;
    }
}

// Main function for the symbolic 2 phase
// Builds column indices arrays for the result matrix using the sparsity pattern from sym1
// Uses specialized kernels based on row characteristics (organized by bins)
template<std::floating_point T, typename Params>
void sym2(const utils::DeviceCSR<T>& A,
          const utils::DeviceCSR<T>& B,
          utils::DeviceCSR<T>& C,
          Meta<Params>& meta) {
    NVTX3_FUNC_RANGE();

    // Handle global memory bin
    const auto gmem_bin_size = meta.h_bin_sizes[Params::SYM2_GLOBAL_MEM_BIN];
    SPDLOG_DEBUG("SYM2 bin {} size is {}",
                 Params::SYM2_GLOBAL_MEM_BIN,
                 meta.h_bin_sizes[Params::SYM2_GLOBAL_MEM_BIN]);
    const auto gmem_table_size = gsl::narrow_cast<std::int32_t>(meta.h_max_row_nnz
                                                                / SYM2_RANGE_RATIO);
    if (gmem_bin_size > 0) {
        static constexpr auto BLOCK_SIZE =
            Params::SYM2_BLOCK_SIZES[Params::SYM2_GLOBAL_MEM_BIN];

        const auto gmem_size = gmem_bin_size * gmem_table_size * sizeof(std::int32_t);
        if (meta.mem_pool_size < gmem_size) {
            SPDLOG_DEBUG("Memory pool size {} is less than required {}",
                         meta.mem_pool_size,
                         gmem_size);
            utils::free_async(meta.d_mem_pool, meta.streams[Params::SYM2_GLOBAL_MEM_BIN]);
            meta.mem_pool_size = gmem_size;
            meta.d_mem_pool = utils::malloc_async(
                meta.mem_pool_size,
                meta.streams[Params::SYM2_GLOBAL_MEM_BIN]);
        }

        utils::launch_kernel(k_sym2_global<BLOCK_SIZE>,
                             gmem_bin_size,
                             BLOCK_SIZE,
                             0,
                             meta.streams[Params::SYM2_GLOBAL_MEM_BIN],
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             C.rpt,
                             meta.d_bins
                                 + meta.h_bin_offsets[Params::SYM2_GLOBAL_MEM_BIN],
                             gmem_table_size,
                             static_cast<std::int32_t*>(meta.d_mem_pool),
                             C.col);
    }

    // Handle regular bins
    constexpr_for<Params::SYM2_GLOBAL_MEM_BIN - 1, -1, -1>(
        [&]<std::int32_t I>(std::integral_constant<std::int32_t, I> ARG) {
            static constexpr auto BIN = ARG.value;
            SPDLOG_DEBUG("SYM2 bin {} size is {}", BIN, meta.h_bin_sizes[BIN]);
            if (meta.h_bin_sizes[BIN] == 0)
                return;

            static constexpr auto BLOCK_SIZE = Params::SYM2_BLOCK_SIZES[BIN];
            static constexpr auto PWARP_SIZE = Params::SYM2_PWARP_SIZES[BIN];
            static constexpr auto TABLE_SIZE = Params::SYM2_TABLE_SIZES[BIN];
            static constexpr auto SMEM = Params::SYM2_SMEM_SIZES[BIN];

            if constexpr (PWARP_SIZE > 0) {
                static constexpr auto ROWS_PER_BLOCK = utils::divpow2(BLOCK_SIZE,
                                                                      PWARP_SIZE);

                utils::handle_cuda_error(cudaFuncSetAttribute(
                    k_sym2_smem_pwarp<BLOCK_SIZE, PWARP_SIZE, TABLE_SIZE>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SMEM));
                utils::launch_kernel(
                    k_sym2_smem_pwarp<BLOCK_SIZE, PWARP_SIZE, TABLE_SIZE>,
                    cuda::ceil_div(meta.h_bin_sizes[BIN], ROWS_PER_BLOCK),
                    BLOCK_SIZE,
                    SMEM,
                    meta.streams[BIN],
                    A.rpt,
                    A.col,
                    B.rpt,
                    B.col,
                    C.rpt,
                    meta.d_bins + meta.h_bin_offsets[BIN],
                    meta.h_bin_sizes[BIN],
                    C.col);
            } else {
                utils::handle_cuda_error(
                    cudaFuncSetAttribute(k_sym2_smem<BLOCK_SIZE, TABLE_SIZE>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM));
                utils::launch_kernel(k_sym2_smem<BLOCK_SIZE, TABLE_SIZE>,
                                     meta.h_bin_sizes[BIN],
                                     BLOCK_SIZE,
                                     SMEM,
                                     meta.streams[BIN],
                                     A.rpt,
                                     A.col,
                                     B.rpt,
                                     B.col,
                                     C.rpt,
                                     meta.d_bins + meta.h_bin_offsets[BIN],
                                     C.col);
            }
        });

    // Allocate C.val
    auto* val_ptr = utils::malloc_async(C.nnz * sizeof(T));
    C.val = static_cast<T*>(val_ptr);

    // Wait for all bins to finish
    for (std::int32_t i = 0; i < Params::SYM2_N_BINS; i++)
        utils::stream_sync(meta.streams[i]);

    utils::stream_sync();
}

} // namespace proposal
