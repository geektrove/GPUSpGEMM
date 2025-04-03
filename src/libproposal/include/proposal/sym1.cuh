#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <type_traits>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>
#include <proposal/parameters.cuh>
#include <proposal/utils.cuh>

namespace cg = cooperative_groups;

template<std::int32_t BLOCK_SIZE, std::int32_t PWARP_SIZE, std::int32_t TOTAL_TABLE_SIZE>
__launch_bounds__(BLOCK_SIZE, get_minctapersm(BLOCK_SIZE)) __global__
    void k_sym1_smem_pwarp(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ a_col,
                           const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ b_col,
                           const __grid_constant__ std::int32_t* const __restrict__ bins,
                           const __grid_constant__ std::int32_t bin_size,
                           __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    static_assert(utils::ispow2(BLOCK_SIZE));
    static_assert(utils::ispow2(TOTAL_TABLE_SIZE));
    static_assert(utils::ispow2(PWARP_SIZE));
    static_assert(PWARP_SIZE <= WARP_SIZE);

    static constexpr auto ROWS_PER_BLOCK = utils::divpow2(BLOCK_SIZE, PWARP_SIZE);
    static constexpr auto TABLE_SIZE = utils::divpow2(TOTAL_TABLE_SIZE, ROWS_PER_BLOCK);
    static_assert(TOTAL_TABLE_SIZE % ROWS_PER_BLOCK == 0);

    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tile = cg::tiled_partition<PWARP_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto tip = utils::modpow2(tib, PWARP_SIZE);
    const auto pib = utils::divpow2(tib, PWARP_SIZE);

    auto* s_tables = reinterpret_cast<std::int32_t*>(smem);

    for (auto i = tib; i < TOTAL_TABLE_SIZE; i += BLOCK_SIZE)
        s_tables[i] = HASH_EMPTY;
    block.sync();

    const auto row_id = utils::divpow2(tig, PWARP_SIZE);
    if (row_id >= bin_size)
        return;
    auto* s_table = s_tables + (static_cast<ptrdiff_t>(pib * TABLE_SIZE));

    //Aggregate column indices in the hash table
    const auto row = bins[row_id];
    assert(nnzs[row] <= TABLE_SIZE);
    auto l_nnz = 0;
    for (auto i = a_rpt[row] + tip; i < a_rpt[row + 1]; i += PWARP_SIZE) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow]; k < b_rpt[colrow + 1]; k++) {
            const auto key = b_col[k];
            auto hash = utils::modpow2(key * HASH_SCALE, TABLE_SIZE);
            while (true) {
                const auto old = atomicCAS_block(s_table + hash, HASH_EMPTY, key);
                if (old == HASH_EMPTY) {
                    l_nnz++;
                    break;
                }
                if (old == key)
                    break;
                hash = utils::modpow2(hash + 1, TABLE_SIZE);
            }
        }
    }
    tile.sync();

    const auto sum_nnz = cg::reduce(tile, l_nnz, cg::plus<std::int32_t>{});
    cg::invoke_one(tile, [&] { nnzs[row] = sum_nnz; });
}

template<std::int32_t BLOCK_SIZE, std::int32_t TABLE_SIZE>
__launch_bounds__(BLOCK_SIZE, get_minctapersm(BLOCK_SIZE)) __global__
    void k_sym1_smem(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                     const __grid_constant__ std::int32_t* const __restrict__ a_col,
                     const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                     const __grid_constant__ std::int32_t* const __restrict__ b_col,
                     const __grid_constant__ std::int32_t* const __restrict__ bins,
                     __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    static_assert(utils::ispow2(TABLE_SIZE));

    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    auto* s_nnz = s_table + TABLE_SIZE;

    for (auto i = tib; i < TABLE_SIZE; i += BLOCK_SIZE)
        s_table[i] = HASH_EMPTY;
    cg::invoke_one(block, [&] { *s_nnz = 0; });
    block.sync();

    const auto row = bins[grid.block_rank()];
    assert(nnzs[row] <= TABLE_SIZE);
    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto key = b_col[k];
            auto hash = utils::modpow2(key * HASH_SCALE, TABLE_SIZE);
            while (true) {
                const auto old = atomicCAS_block(s_table + hash, HASH_EMPTY, key);
                if (old == HASH_EMPTY) {
                    atomicAdd_block(s_nnz, 1);
                    break;
                }
                if (old == key)
                    break;
                hash = utils::modpow2(hash + 1, TABLE_SIZE);
            }
        }
    }
    block.sync();

    cg::invoke_one(block, [&] { nnzs[row] = *s_nnz; });
}

template<std::int32_t BLOCK_SIZE, std::int32_t TABLE_SIZE>
__launch_bounds__(BLOCK_SIZE, get_minctapersm(BLOCK_SIZE)) __global__
    void k_sym1_smem_max(
        const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
        const __grid_constant__ std::int32_t* const __restrict__ a_col,
        const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
        const __grid_constant__ std::int32_t* const __restrict__ b_col,
        const __grid_constant__ std::int32_t* const __restrict__ bins,
        __grid_constant__ std::int32_t* const __restrict__ nnzs,
        __grid_constant__ std::int32_t* const __restrict__ fail_bin,
        __grid_constant__ std::int32_t* const __restrict__ fail_bin_size) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    auto* s_nnz = s_table + TABLE_SIZE;

    for (auto i = tib; i < TABLE_SIZE; i += BLOCK_SIZE)
        s_table[i] = HASH_EMPTY;
    cg::invoke_one(block, [&] { *s_nnz = 0; });

    const auto row = bins[grid.block_rank()];
    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    const auto threshold = TABLE_SIZE * SYM1_RANGE_RATIO;
    block.sync();

    cuda::std::invoke([&] {
        for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
            const auto colrow = a_col[i];
            for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
                const auto key = b_col[k];
                auto hash = (key * HASH_SCALE) % TABLE_SIZE;
                while (true) {
                    const auto old = atomicCAS_block(s_table + hash, HASH_EMPTY, key);
                    if (old == HASH_EMPTY) {
                        if (atomicAdd_block(s_nnz, 1) >= threshold)
                            return;
                        break;
                    }
                    if (old == key)
                        break;
                    hash = hash + 1 < TABLE_SIZE ? hash + 1 : 0;
                }
            }
        }
    });
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

template<std::int32_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, get_minctapersm(BLOCK_SIZE)) __global__
    void k_sym1_global(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                       const __grid_constant__ std::int32_t* const __restrict__ a_col,
                       const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                       const __grid_constant__ std::int32_t* const __restrict__ b_col,
                       const __grid_constant__ std::int32_t* const __restrict__ bins,
                       const __grid_constant__ std::int32_t table_size,
                       __grid_constant__ std::int32_t* const __restrict__ tables,
                       __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    __shared__ std::int32_t s_nnz;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    auto* table = tables + (static_cast<ptrdiff_t>(grid.block_rank()) * table_size);
    for (auto i = tib; i < table_size; i += BLOCK_SIZE)
        table[i] = HASH_EMPTY;
    cg::invoke_one(block, [&] { s_nnz = 0; });
    block.sync();

    const auto row = bins[grid.block_rank()];
    assert(nnzs[row] <= table_size);
    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
    const auto k_offset = utils::modpow2(tib, WARP_SIZE);
    const auto k_step = WARP_SIZE;
    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (true) {
                const auto old = atomicCAS_block(table + hash, HASH_EMPTY, key);
                if (old == HASH_EMPTY) {
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

template<std::floating_point T, typename Params>
void sym1(const utils::DeviceCSR<T>& A,
          const utils::DeviceCSR<T>& B,
          utils::DeviceCSR<T>& C,
          Meta<Params>& meta) {
    NVTX3_FUNC_RANGE();

    static constexpr auto IdxByteSize = gsl::narrow_cast<std::int32_t>(
        sizeof(std::int32_t));

    // Handle max smem bin first as it may fail to fit in shared memory
    const auto max_smem_bin_size = meta.h_bin_sizes[Params::SYM1_MAX_SMEM_BIN];
    std::int32_t h_fail_bin_size{};
    std::int32_t* d_fail_bin{};
    std::int32_t* d_fail_bin_size{};
    SPDLOG_DEBUG("SYM1 bin {} size is {}", Params::SYM1_MAX_SMEM_BIN, max_smem_bin_size);
    if (max_smem_bin_size > 0) {
        auto tmp_mem_size = (max_smem_bin_size + 1) * IdxByteSize;
        if (tmp_mem_size <= gsl::narrow_cast<std::int32_t>(meta.cub_storage_size)) {
            SPDLOG_DEBUG("CUB storage is enough for fail bin: {} <= {}",
                         tmp_mem_size,
                         meta.cub_storage_size);
            d_fail_bin = static_cast<std::int32_t*>(meta.d_cub_storage);
        } else {
            SPDLOG_DEBUG("CUB storage is not enough for fail bin: {} > {}",
                         tmp_mem_size,
                         meta.cub_storage_size);
            auto* d_fail_ptr = utils::malloc_async<utils::Location::Device>(
                tmp_mem_size,
                meta.streams[Params::SYM1_MAX_SMEM_BIN]);
            d_fail_bin = static_cast<std::int32_t*>(d_fail_ptr);
        }
        d_fail_bin_size = d_fail_bin + max_smem_bin_size;
        utils::memset_async(d_fail_bin_size,
                            0,
                            sizeof(std::int32_t),
                            meta.streams[Params::SYM1_MAX_SMEM_BIN]);

        static constexpr auto BLOCK_SIZE =
            Params::SYM1_BLOCK_SIZES[Params::SYM1_MAX_SMEM_BIN];
        static constexpr auto TABLE_SIZE =
            Params::SYM1_TABLE_SIZES[Params::SYM1_MAX_SMEM_BIN];
        static constexpr auto SMEM = Params::SYM1_SMEM_SIZES[Params::SYM1_MAX_SMEM_BIN];

        utils::handle_cuda_error(
            cudaFuncSetAttribute(k_sym1_smem_max<BLOCK_SIZE, TABLE_SIZE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 SMEM));
        utils::launch_kernel(k_sym1_smem_max<BLOCK_SIZE, TABLE_SIZE>,
                             max_smem_bin_size,
                             BLOCK_SIZE,
                             SMEM,
                             meta.streams[Params::SYM1_MAX_SMEM_BIN],
                             A.rpt,
                             A.col,
                             B.rpt,
                             B.col,
                             meta.d_bins + meta.h_bin_offsets[Params::SYM1_MAX_SMEM_BIN],
                             C.rpt,
                             d_fail_bin,
                             d_fail_bin_size);
        utils::memcpy_async(&h_fail_bin_size,
                            d_fail_bin_size,
                            sizeof(std::int32_t),
                            meta.streams[Params::SYM1_MAX_SMEM_BIN]);
    }

    // Handle remaining bins
    constexpr_for<Params::SYM1_MAX_SMEM_BIN - 1, -1, -1>(
        [&]<std::int32_t I>(std::integral_constant<std::int32_t, I> ARG) {
            static constexpr auto BIN = ARG.value;
            SPDLOG_DEBUG("SYM1 bin {} size is {}", BIN, meta.h_bin_sizes[BIN]);
            if (meta.h_bin_sizes[BIN] == 0)
                return;

            static constexpr auto BLOCK_SIZE = Params::SYM1_BLOCK_SIZES[BIN];
            static constexpr auto PWARP_SIZE = Params::SYM1_PWARP_SIZES[BIN];
            static constexpr auto TABLE_SIZE = Params::SYM1_TABLE_SIZES[BIN];
            static constexpr auto SMEM = Params::SYM1_SMEM_SIZES[BIN];

            if constexpr (PWARP_SIZE > 0) {
                static constexpr auto ROWS_PER_BLOCK = utils::divpow2(BLOCK_SIZE,
                                                                      PWARP_SIZE);
                utils::handle_cuda_error(cudaFuncSetAttribute(
                    k_sym1_smem_pwarp<BLOCK_SIZE, PWARP_SIZE, TABLE_SIZE>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SMEM));
                utils::launch_kernel(
                    k_sym1_smem_pwarp<BLOCK_SIZE, PWARP_SIZE, TABLE_SIZE>,
                    cuda::ceil_div(meta.h_bin_sizes[BIN], ROWS_PER_BLOCK),
                    BLOCK_SIZE,
                    SMEM,
                    meta.streams[BIN],
                    A.rpt,
                    A.col,
                    B.rpt,
                    B.col,
                    meta.d_bins + meta.h_bin_offsets[BIN],
                    meta.h_bin_sizes[BIN],
                    C.rpt);
            } else {
                utils::handle_cuda_error(
                    cudaFuncSetAttribute(k_sym1_smem<BLOCK_SIZE, TABLE_SIZE>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM));
                utils::launch_kernel(k_sym1_smem<BLOCK_SIZE, TABLE_SIZE>,
                                     meta.h_bin_sizes[BIN],
                                     BLOCK_SIZE,
                                     SMEM,
                                     meta.streams[BIN],
                                     A.rpt,
                                     A.col,
                                     B.rpt,
                                     B.col,
                                     meta.d_bins + meta.h_bin_offsets[BIN],
                                     C.rpt);
            }
        });

    // Handle fail bin
    if (max_smem_bin_size > 0) {
        utils::stream_sync(meta.streams[Params::SYM1_MAX_SMEM_BIN]);
        SPDLOG_DEBUG("SYM1 fail bin size is {}", h_fail_bin_size);
        if (h_fail_bin_size > 0) {
            static constexpr auto BLOCK_SIZE =
                Params::SYM1_BLOCK_SIZES[Params::SYM1_GLOBAL_MEM_BIN];

            const auto table_size = meta.h_max_row_nnz;
            meta.mem_pool_size = gsl::narrow_cast<std::size_t>(h_fail_bin_size
                                                               * table_size)
                                 * IdxByteSize;
            meta.d_mem_pool = utils::malloc_async(
                meta.mem_pool_size,
                meta.streams[Params::SYM1_GLOBAL_MEM_BIN]);
            utils::launch_kernel(k_sym1_global<BLOCK_SIZE>,
                                 h_fail_bin_size,
                                 BLOCK_SIZE,
                                 0,
                                 meta.streams[Params::SYM1_GLOBAL_MEM_BIN],
                                 A.rpt,
                                 A.col,
                                 B.rpt,
                                 B.col,
                                 d_fail_bin,
                                 table_size,
                                 static_cast<std::int32_t*>(meta.d_mem_pool),
                                 C.rpt);
        }
    }

    // Wait for all bins to finish
    for (std::int32_t i = 0; i < Params::SYM1_N_BINS; i++)
        utils::stream_sync(meta.streams[i]);

    // No need to wait for fail bin deallocation
    if (d_fail_bin != nullptr && d_fail_bin != meta.d_cub_storage)
        utils::free_async(d_fail_bin, meta.streams[Params::SYM1_GLOBAL_MEM_BIN]);
}
