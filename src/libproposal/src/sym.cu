#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda/std/cstddef>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/functional>
#include <gsl/gsl-lite.hpp>

#include <utils/utils.cuh>

#include <proposal/sym.cuh>

namespace cg = cooperative_groups;

inline constexpr std::int32_t HASH_SCALE = 107;

__global__ void k_sym_smem_pwarp(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tile = cg::tiled_partition<SYM_PWARP_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto tip = tib % SYM_PWARP_SIZE;
    const auto pib = tib / SYM_PWARP_SIZE;
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    const auto rows_per_block = block_size / SYM_PWARP_SIZE;
    const auto total_table_size = table_size * rows_per_block;

    auto* s_tables = reinterpret_cast<std::int32_t*>(smem);

    for (auto i = tib; i < total_table_size; i += block_size)
        s_tables[i] = -1;
    block.sync();

    const auto row_id = tig / SYM_PWARP_SIZE;
    if (row_id >= bin_size)
        return;

    auto* s_table = s_tables + (static_cast<ptrdiff_t>(pib * table_size));
    const auto row = bins[row_id];

    auto l_nnz = 0;
    for (auto i = a_rpt[row] + tip; i < a_rpt[row + 1]; i += SYM_PWARP_SIZE) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow]; k < b_rpt[colrow + 1]; k++) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (true) {
                const auto old = atomicCAS_block(s_table + hash, -1, key);
                if (old == -1) {
                    l_nnz++;
                    break;
                }
                if (old == key)
                    break;
                hash = (hash + 1) % table_size;
            }
        }
    }
    tile.sync();

    const auto sum_nnz = cg::reduce(tile, l_nnz, cg::plus<std::int32_t>{});
    cg::invoke_one(tile, [&] { nnzs[row] = sum_nnz; });
}

__global__ void k_sym_smem(const __grid_constant__ std::int32_t table_size,
                           const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ a_col,
                           const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                           const __grid_constant__ std::int32_t* const __restrict__ b_col,
                           const __grid_constant__ std::int32_t* const __restrict__ bins,
                           __grid_constant__ std::int32_t* const __restrict__ nnzs) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    auto* s_nnz = s_table + table_size;

    for (auto i = tib; i < table_size; i += block_size)
        s_table[i] = -1;
    cg::invoke_one(block, [&] { *s_nnz = 0; });
    block.sync();

    const auto row = bins[grid.block_rank()];
    const auto i_offset = tib / WARP_SIZE;
    const auto i_step = block_size / WARP_SIZE;
    const auto k_offset = tib % WARP_SIZE;
    const auto k_step = WARP_SIZE;
    for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (true) {
                const auto old = atomicCAS_block(s_table + hash, -1, key);
                if (old == -1) {
                    atomicAdd_block(s_nnz, 1);
                    break;
                }
                if (old == key)
                    break;
                hash = (hash + 1) % table_size;
            }
        }
    }
    block.sync();

    cg::invoke_one(block, [&] { nnzs[row] = *s_nnz; });
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
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    auto* s_nnz = s_table + table_size;

    for (auto i = tib; i < table_size; i += block_size)
        s_table[i] = -1;
    cg::invoke_one(block, [&] { *s_nnz = 0; });

    const auto row = bins[grid.block_rank()];
    const auto i_offset = tib / WARP_SIZE;
    const auto i_step = block_size / WARP_SIZE;
    const auto k_offset = tib % WARP_SIZE;
    const auto k_step = WARP_SIZE;
    const auto threshold = table_size * SYM_RANGE_RATIO;
    block.sync();

    cuda::std::invoke([&] {
        for (auto i = a_rpt[row] + i_offset; i < a_rpt[row + 1]; i += i_step) {
            const auto colrow = a_col[i];
            for (auto k = b_rpt[colrow] + k_offset; k < b_rpt[colrow + 1]; k += k_step) {
                const auto key = b_col[k];
                auto hash = (key * HASH_SCALE) % table_size;
                while (true) {
                    const auto old = atomicCAS_block(s_table + hash, -1, key);
                    if (old == -1) {
                        if (atomicAdd_block(s_nnz, 1) >= threshold)
                            return;
                        break;
                    }
                    if (old == key)
                        break;
                    hash = hash + 1 < table_size ? hash + 1 : 0;
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
