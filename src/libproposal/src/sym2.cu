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

#include <proposal/sym2.cuh>

namespace cg = cooperative_groups;

inline constexpr std::int32_t HASH_SCALE = 107;

__global__ void k_sym2_smem_pwarp(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    const __grid_constant__ std::int32_t bin_size,
    __grid_constant__ std::int32_t* const __restrict__ c_col) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto tip = tib % SYM_PWARP_SIZE;
    const auto pib = tib / SYM_PWARP_SIZE;
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    const auto rows_per_block = block_size / SYM_PWARP_SIZE;
    const auto total_table_size = table_size * rows_per_block;

    auto* s_tables = reinterpret_cast<std::int32_t*>(smem);
    auto* s_offsets = s_tables + total_table_size;

    for (auto i = tib; i < total_table_size; i += block_size)
        s_tables[i] = -1;
    if (tib < rows_per_block)
        s_offsets[tib] = 0;
    block.sync();

    const auto row_id = tig / SYM_PWARP_SIZE;
    if (row_id >= bin_size)
        return;

    auto* s_table = s_tables + static_cast<ptrdiff_t>(pib * table_size);
    auto* s_offset = s_offsets + pib;

    // Aggregate the column indices in the hash table
    const auto row = bins[row_id];
    for (auto i = a_rpt[row] + tip; i < a_rpt[row + 1]; i += SYM_PWARP_SIZE) {
        const auto colrow = a_col[i];
        for (auto k = b_rpt[colrow]; k < b_rpt[colrow + 1]; k++) {
            const auto key = b_col[k];
            auto hash = (key * HASH_SCALE) % table_size;
            while (true) {
                const auto old = atomicCAS_block(s_table + hash, -1, key);
                if (old == -1 || old == key)
                    break;
                hash = (hash + 1) % table_size;
            }
        }
    }
    warp.sync();

    // Condense the column indices
    for (auto offset = 0; offset < table_size; offset += SYM_PWARP_SIZE) {
        const auto i = offset + tip;
        const auto col = i < table_size ? s_table[i] : -1;
        warp.sync();
        if (col != -1)
            s_table[atomicAdd_block(s_offset, 1)] = col;
    }
    warp.sync();

    // Write the column indices to the output
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;
    for (auto i = tip; i < nnz; i += SYM_PWARP_SIZE) {
        const auto col = s_table[i];
        std::int32_t n_less = 0;
        for (std::int32_t j = 0; j < nnz; j++) {
            if (s_table[j] < col)
                n_less++;
        }
        c_col[c_offset + n_less] = col;
    }
}

__global__ void k_sym2_smem(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ c_col) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    auto* s_offset = s_table + table_size;

    for (auto i = tib; i < table_size; i += block_size)
        s_table[i] = -1;
    cg::invoke_one(block, [&] { *s_offset = 0; });
    block.sync();

    // Aggregate the column indices in the hash table
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
                if (old == -1 || old == key)
                    break;
                hash = (hash + 1) % table_size;
            }
        }
    }
    block.sync();

    // Condense the column indices
    for (auto offset = 0; offset < table_size; offset += block_size) {
        const auto i = offset + tib;
        const auto col = i < table_size ? s_table[i] : -1;
        block.sync();
        if (col != -1)
            s_table[atomicAdd_block(s_offset, 1)] = col;
    }
    block.sync();

    // Write the column indices to the output
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;
    for (auto i = tib; i < nnz; i += block_size) {
        const auto col = s_table[i];
        std::int32_t n_less = 0;
        for (std::int32_t j = 0; j < nnz; j++) {
            if (s_table[j] < col)
                n_less++;
        }
        c_col[c_offset + n_less] = col;
    }
}

__global__ void k_sym2_smem_max(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ c_col) {
    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* s_table = reinterpret_cast<std::int32_t*>(smem);
    auto* s_offset = s_table + table_size;

    for (auto i = tib; i < table_size; i += block_size)
        s_table[i] = -1;
    cg::invoke_one(block, [&] { *s_offset = 0; });
    block.sync();

    // Aggregate the column indices in the hash table
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
                if (old == -1 || old == key)
                    break;
                hash = hash + 1 < table_size ? hash + 1 : 0;
            }
        }
    }
    block.sync();

    // Condense the column indices
    for (auto offset = 0; offset < table_size; offset += block_size) {
        const auto i = offset + tib;
        const auto col = i < table_size ? s_table[i] : -1;
        block.sync();
        if (col != -1)
            s_table[atomicAdd_block(s_offset, 1)] = col;
    }
    block.sync();

    // Write the column indices to the output
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;
    for (auto i = tib; i < nnz; i += block_size) {
        const auto col = s_table[i];
        std::int32_t n_less = 0;
        for (std::int32_t j = 0; j < nnz; j++) {
            if (s_table[j] < col)
                n_less++;
        }
        c_col[c_offset + n_less] = col;
    }
}

__global__ void k_sym2_global(
    const __grid_constant__ std::int32_t table_size,
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ b_col,
    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ bins,
    __grid_constant__ std::int32_t* const __restrict__ tables,
    __grid_constant__ std::int32_t* const __restrict__ c_col) {
    __shared__ std::int32_t s_offset;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto& tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto& block_size = gsl::narrow_cast<std::int32_t>(block.num_threads());

    auto* table = tables + (static_cast<ptrdiff_t>(grid.block_rank()) * table_size);
    for (auto i = tib; i < table_size; i += block_size)
        table[i] = -1;
    cg::invoke_one(block, [&] { s_offset = 0; });
    block.sync();

    // Aggregate the column indices in the hash table
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
                const auto old = atomicCAS_block(table + hash, -1, key);
                if (old == -1 || old == key)
                    break;
                hash = hash + 1 < table_size ? hash + 1 : 0;
            }
        }
    }
    block.sync();

    // Condense the column indices
    for (auto offset = 0; offset < table_size; offset += block_size) {
        const auto i = offset + tib;
        const auto col = i < table_size ? table[i] : -1;
        block.sync();
        if (col != -1)
            table[atomicAdd_block(&s_offset, 1)] = col;
    }
    block.sync();

    // Write the column indices to the output
    const auto c_offset = c_rpt[row];
    const auto nnz = c_rpt[row + 1] - c_offset;
    for (auto i = tib; i < nnz; i += block_size) {
        const auto col = table[i];
        std::int32_t n_less = 0;
        for (std::int32_t j = 0; j < nnz; j++) {
            if (table[j] < col)
                n_less++;
        }
        c_col[c_offset + n_less] = col;
    }
}
