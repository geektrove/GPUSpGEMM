#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda/std/cstddef>
#include <type_traits>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>
#include <proposal/parameters.cuh>
#include <proposal/utils.cuh>

namespace proposal {

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

template<std::floating_point T,
         std::int32_t BLOCK_SIZE,
         std::int32_t PWARP_SIZE,
         std::int32_t TOTAL_ARRAY_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_num_smem_pwarp(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
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
    static constexpr auto ROWS_PER_BLOCK = utils::divpow2(BLOCK_SIZE, PWARP_SIZE);
    static constexpr auto ARRAY_SIZE = utils::divpow2(TOTAL_ARRAY_SIZE, ROWS_PER_BLOCK);

    static_assert(utils::ispow2(BLOCK_SIZE));
    static_assert(PWARP_SIZE <= WARP_SIZE);
    static_assert(TOTAL_ARRAY_SIZE % ROWS_PER_BLOCK == 0);
#ifndef NDEBUG
    static constexpr auto SMEM = TOTAL_ARRAY_SIZE * (sizeof(T) + sizeof(std::int32_t));
    assert(dynamic_smem_size() == SMEM);
#endif

    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tile = cg::tiled_partition<PWARP_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());
    const auto tip = utils::modpow2(tib, PWARP_SIZE);
    const auto pib = utils::divpow2(tib, PWARP_SIZE);

    const auto row_id = utils::divpow2(tig, PWARP_SIZE);
    if (row_id >= bin_size)
        return;

    auto* s_vals_all = reinterpret_cast<T*>(smem);
    auto* s_cols_all = reinterpret_cast<std::int32_t*>(s_vals_all + TOTAL_ARRAY_SIZE);
    auto* s_vals = s_vals_all + (pib * ARRAY_SIZE);
    auto* s_cols = s_cols_all + static_cast<ptrdiff_t>(pib * ARRAY_SIZE);

    const auto row = bins[row_id];
    const auto c_offset = c_rpt[row];
    const auto size = c_rpt[row + 1] - c_offset;
    assert(size <= ARRAY_SIZE);

    cg::memcpy_async(tile, s_cols, c_col + c_offset, size * sizeof(*s_cols));
    for (auto i = tip; i < size; i += PWARP_SIZE)
        s_vals[i] = 0;
    cg::wait(tile);
    tile.sync();

    for (auto i = a_rpt[row] + tip; i < a_rpt[row + 1]; i += PWARP_SIZE) {
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

template<std::floating_point T, std::int32_t BLOCK_SIZE, std::int32_t ARRAY_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_num_smem(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                    const __grid_constant__ std::int32_t* const __restrict__ a_col,
                    const __grid_constant__ T* const __restrict__ a_val,
                    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                    const __grid_constant__ std::int32_t* const __restrict__ b_col,
                    const __grid_constant__ T* const __restrict__ b_val,
                    const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
                    const __grid_constant__ std::int32_t* const __restrict__ c_col,
                    const __grid_constant__ std::int32_t* const __restrict__ bins,
                    __grid_constant__ T* const __restrict__ c_val) {
    static_assert(utils::ispow2(BLOCK_SIZE));
#ifndef NDEBUG
    static constexpr auto SMEM = ARRAY_SIZE * (sizeof(T) + sizeof(std::int32_t));
    assert(dynamic_smem_size() == SMEM);
#endif

    extern __shared__ cuda::std::byte smem[];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    auto* s_vals = reinterpret_cast<T*>(smem);
    auto* s_cols = reinterpret_cast<std::int32_t*>(s_vals + ARRAY_SIZE);

    const auto row = bins[grid.block_rank()];
    const auto c_offset = c_rpt[row];
    const auto size = c_rpt[row + 1] - c_offset;

    cg::memcpy_async(block, s_cols, c_col + c_offset, size * sizeof(*s_cols));
    for (auto i = tib; i < size; i += BLOCK_SIZE)
        s_vals[i] = 0;
    cg::wait(block);
    block.sync();

    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
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

template<std::floating_point T, std::int32_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_num_global(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                      const __grid_constant__ std::int32_t* const __restrict__ a_col,
                      const __grid_constant__ T* const __restrict__ a_val,
                      const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                      const __grid_constant__ std::int32_t* const __restrict__ b_col,
                      const __grid_constant__ T* const __restrict__ b_val,
                      const __grid_constant__ std::int32_t* const __restrict__ c_rpt,
                      const __grid_constant__ std::int32_t* const __restrict__ c_col,
                      const __grid_constant__ std::int32_t* const __restrict__ bins,
                      __grid_constant__ T* const __restrict__ c_val) {
    static_assert(utils::ispow2(BLOCK_SIZE));

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    const auto row = bins[grid.block_rank()];
    const auto c_offset = c_rpt[row];
    const auto size = c_rpt[row + 1] - c_offset;

    auto* vals = c_val + c_offset;
    const auto* cols = c_col + c_offset;
    for (auto i = tib; i < size; i += BLOCK_SIZE)
        vals[i] = 0;

    const auto i_offset = utils::divpow2(tib, WARP_SIZE);
    const auto i_step = utils::divpow2(BLOCK_SIZE, WARP_SIZE);
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

template<std::floating_point T, typename Params>
void num(const utils::DeviceCSR<T>& A,
         const utils::DeviceCSR<T>& B,
         utils::DeviceCSR<T>& C,
         Meta<Params>& meta) {
    NVTX3_FUNC_RANGE();

    // Handle global memory bin
    const auto global_mem_bin_size = meta.h_bin_sizes[Params::NUM_GLOBAL_MEM_BIN];
    SPDLOG_DEBUG("NUM bin {} size is {}",
                 Params::NUM_GLOBAL_MEM_BIN,
                 meta.h_bin_sizes[Params::NUM_GLOBAL_MEM_BIN]);
    if (global_mem_bin_size > 0) {
        static constexpr auto BLOCK_SIZE =
            Params::NUM_BLOCK_SIZES[Params::NUM_GLOBAL_MEM_BIN];

        utils::launch_kernel(k_num_global<T, BLOCK_SIZE>,
                             global_mem_bin_size,
                             BLOCK_SIZE,
                             0,
                             meta.streams[Params::NUM_GLOBAL_MEM_BIN],
                             A.rpt,
                             A.col,
                             A.val,
                             B.rpt,
                             B.col,
                             B.val,
                             C.rpt,
                             C.col,
                             meta.d_bins + meta.h_bin_offsets[Params::NUM_GLOBAL_MEM_BIN],
                             C.val);
    }

    // Handle regular bins
    constexpr_for<Params::NUM_GLOBAL_MEM_BIN - 1, -1, -1>(
        [&]<std::int32_t I>(std::integral_constant<std::int32_t, I> ARG) {
            static constexpr auto BIN = ARG.value;
            SPDLOG_DEBUG("NUM bin {} size is {}", BIN, meta.h_bin_sizes[BIN]);
            if (meta.h_bin_sizes[BIN] == 0)
                return;

            static constexpr auto BLOCK_SIZE = Params::NUM_BLOCK_SIZES[BIN];
            static constexpr auto PWARP_SIZE = Params::NUM_PWARP_SIZES[BIN];
            static constexpr auto ARRAY_SIZE = std::is_same_v<T, float>
                                                   ? Params::NUM_ARRAY_SIZES_F32[BIN]
                                                   : Params::NUM_ARRAY_SIZES_F64[BIN];
            static constexpr auto SMEM = std::is_same_v<T, float>
                                             ? Params::NUM_SMEM_SIZES_F32[BIN]
                                             : Params::NUM_SMEM_SIZES_F64[BIN];

            if constexpr (PWARP_SIZE > 0) {
                static constexpr auto ROWS_PER_BLOCK = utils::divpow2(BLOCK_SIZE,
                                                                      PWARP_SIZE);

                utils::handle_cuda_error(cudaFuncSetAttribute(
                    k_num_smem_pwarp<T, BLOCK_SIZE, PWARP_SIZE, ARRAY_SIZE>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SMEM));
                utils::launch_kernel(
                    k_num_smem_pwarp<T, BLOCK_SIZE, PWARP_SIZE, ARRAY_SIZE>,
                    cuda::ceil_div(meta.h_bin_sizes[BIN], ROWS_PER_BLOCK),
                    BLOCK_SIZE,
                    SMEM,
                    meta.streams[BIN],
                    A.rpt,
                    A.col,
                    A.val,
                    B.rpt,
                    B.col,
                    B.val,
                    C.rpt,
                    C.col,
                    meta.d_bins + meta.h_bin_offsets[BIN],
                    meta.h_bin_sizes[BIN],
                    C.val);
            } else {
                utils::handle_cuda_error(
                    cudaFuncSetAttribute(k_num_smem<T, BLOCK_SIZE, ARRAY_SIZE>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SMEM));
                utils::launch_kernel(k_num_smem<T, BLOCK_SIZE, ARRAY_SIZE>,
                                     meta.h_bin_sizes[BIN],
                                     BLOCK_SIZE,
                                     SMEM,
                                     meta.streams[BIN],
                                     A.rpt,
                                     A.col,
                                     A.val,
                                     B.rpt,
                                     B.col,
                                     B.val,
                                     C.rpt,
                                     C.col,
                                     meta.d_bins + meta.h_bin_offsets[BIN],
                                     C.val);
            }
        });

    // Wait for all bins to finish
    for (std::int32_t i = 0; i < Params::NUM_N_BINS; i++)
        utils::stream_sync(meta.streams[i]);
}

} // namespace proposal
