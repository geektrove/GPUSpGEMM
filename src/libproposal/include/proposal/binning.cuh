#pragma once

#include <cassert>
#include <concepts>
#include <cstdint>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/functional>

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>
#include <proposal/parameters.cuh>

namespace cg = cooperative_groups;

namespace proposal {

template<auto RANGES>
__forceinline__ __device__ auto find_bin(const std::int32_t x) -> std::int32_t {
    static constexpr auto N_BINS = gsl::narrow_cast<std::int32_t>(RANGES.size());
    for (std::int32_t i = 0; i < N_BINS; i++) {
        if (x <= RANGES[i])
            return i;
    }
    assert(false);
    __builtin_unreachable();
}

template<std::int32_t BLOCK_SIZE, BinningType BinType, typename GetValueF>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_binning1(const __grid_constant__ std::int32_t m,
                    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
                    GetValueF get_value) {
    static constexpr auto RANGES = get_ranges<BinType>();
    static constexpr auto N_BINS = gsl::narrow_cast<std::int32_t>(RANGES.size());

    __shared__ std::int32_t s_bin_sizes[N_BINS];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (tib < N_BINS)
        s_bin_sizes[tib] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    if (row < m) {
        const auto value = get_value(row);
        if (value > 0) {
            const auto bin_idx = find_bin<RANGES>(value);
            atomicAdd_block(s_bin_sizes + bin_idx, 1);
        }
    }
    block.sync();

    if (tib < N_BINS)
        atomicAdd(bin_sizes + tib, s_bin_sizes[tib]);
}

template<std::int32_t BLOCK_SIZE, BinningType BinType, typename GetValueF>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_binning2(const __grid_constant__ std::int32_t m,
                    const __grid_constant__ std::int32_t* const __restrict__ bin_offsets,
                    __grid_constant__ std::int32_t* const __restrict__ bin_sizes,
                    __grid_constant__ std::int32_t* const __restrict__ bins,
                    GetValueF get_value) {
    static constexpr auto RANGES = get_ranges<BinType>();
    static constexpr auto N_BINS = gsl::narrow_cast<std::int32_t>(RANGES.size());

    __shared__ std::int32_t s_bin_sizes[N_BINS];
    __shared__ std::int32_t s_bin_offsets[N_BINS];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto tib = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (tib < N_BINS)
        s_bin_sizes[tib] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    std::int32_t bin_idx = 0;
    const auto value = row < m ? get_value(row) : 0;
    if (row < m && value > 0) {
        bin_idx = find_bin<RANGES>(value);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (tib < N_BINS) {
        s_bin_offsets[tib] = atomicAdd(bin_sizes + tib, s_bin_sizes[tib]);
        s_bin_offsets[tib] += bin_offsets[tib];
        s_bin_sizes[tib] = 0;
    }
    block.sync();

    if (row < m && value > 0) {
        const auto index = atomicAdd_block(s_bin_sizes + bin_idx, 1);
        bins[s_bin_offsets[bin_idx] + index] = row;
    }
}

template<typename Params, std::int32_t N_BINS>
void small_binning(const std::int32_t m, Meta<Params>& meta) {
    NVTX3_FUNC_RANGE();

    // Perform iota operation to fill the smallest bin with row indices
    utils::handle_cuda_error(
        cub::DeviceFor::Bulk(meta.d_cub_storage,
                             meta.cub_storage_size,
                             m,
                             [bins = meta.d_bins] __device__(int i) {
                                 bins[i] = gsl::narrow_cast<std::int32_t>(i);
                             }));

    // Set bin sizes and offsets
    meta.h_bin_sizes[0] = m;
    for (int i = 1; i < N_BINS; i++)
        meta.h_bin_sizes[i] = 0;
    meta.h_bin_offsets[0] = 0;
    for (int i = 1; i < N_BINS; i++)
        meta.h_bin_offsets[i] = m;

    utils::stream_sync();
}

template<std::floating_point T, typename Params, BinningType BinType, typename GetValueF>
void binning(utils::DeviceCSR<T>& C, Meta<Params>& meta, GetValueF get_value) {
    NVTX3_FUNC_RANGE();

    static constexpr auto RANGES = get_ranges<Params, BinType>();
    static constexpr auto N_BINS = gsl::narrow_cast<std::int32_t>(RANGES.size());

    if (meta.h_max_row_nnz <= RANGES[0]) {
        // If all rows fall into the smallest bin, we can skip the binning process
        // and directly assign the row indices to the smallest bin
        small_binning<Params, N_BINS>(C.m, meta);
        return;
    }

    // Perform full two-stage symbolic binning
    utils::memset_async(meta.d_bin_sizes, 0, N_BINS * sizeof(*meta.d_bin_sizes));
    utils::launch_kernel(k_binning1<Params::OPTIMAL_BLOCK_SIZE, BinType, GetValueF>,
                         cuda::ceil_div(C.m, Params::OPTIMAL_BLOCK_SIZE),
                         Params::OPTIMAL_BLOCK_SIZE,
                         0,
                         cudaStreamDefault,
                         C.m,
                         meta.d_bin_sizes,
                         get_value);
    utils::memcpy_async(meta.h_bin_sizes,
                        meta.d_bin_sizes,
                        N_BINS * sizeof(*meta.h_bin_sizes));
    utils::stream_sync();
    utils::memset_async(meta.d_bin_sizes, 0, N_BINS * sizeof(*meta.d_bin_sizes));

    meta.h_bin_offsets[0] = 0;
    for (int i = 0; i + 1 < N_BINS; i++)
        meta.h_bin_offsets[i + 1] = meta.h_bin_offsets[i] + meta.h_bin_sizes[i];

    SPDLOG_DEBUG("{:>6s} {:>10s} {:>10s} {:>10s}", "Bin", "Range", "Size", "Offset");
    for (std::int32_t i = 0; i < N_BINS; i++)
        SPDLOG_DEBUG("{:>6d} {:10d} {:>10d} {:>10d}",
                     i,
                     RANGES[i],
                     meta.h_bin_sizes[i],
                     meta.h_bin_offsets[i]);

    utils::memcpy_async(meta.d_bin_offsets,
                        meta.h_bin_offsets,
                        N_BINS * sizeof(*meta.d_bin_offsets));
    utils::launch_kernel(k_binning2<Params::OPTIMAL_BLOCK_SIZE, BinType, GetValueF>,
                         cuda::ceil_div(C.m, Params::OPTIMAL_BLOCK_SIZE),
                         Params::OPTIMAL_BLOCK_SIZE,
                         0,
                         cudaStreamDefault,
                         C.m,
                         meta.d_bin_offsets,
                         meta.d_bin_sizes,
                         meta.d_bins,
                         get_value);

    utils::stream_sync();
}

template<std::floating_point T, typename Params, typename GetValueF>
void sym_binning2(utils::DeviceCSR<T>& C, Meta<Params>& meta, GetValueF get_value) {
    NVTX3_FUNC_RANGE();

    // Calculate max NNZ per row
    utils::handle_cuda_error(cub::DeviceReduce::Max(meta.d_cub_storage,
                                                    meta.cub_storage_size,
                                                    C.rpt,
                                                    meta.d_max_row_nnz,
                                                    C.m));

    // Scan C.rpt
    utils::handle_cuda_error(cub::DeviceScan::ExclusiveSum(meta.d_cub_storage,
                                                           meta.cub_storage_size,
                                                           C.rpt,
                                                           C.rpt,
                                                           C.m + 1));

    // Copy max and total NNZ to host
    utils::memcpy_async(&meta.h_max_row_nnz,
                        meta.d_max_row_nnz,
                        sizeof(meta.h_max_row_nnz));
    utils::memcpy_async(&C.nnz, C.rpt + C.m, sizeof(C.nnz));
    utils::stream_sync();
    SPDLOG_DEBUG("Max NNZ per row is {}", meta.h_max_row_nnz);
    SPDLOG_DEBUG("Total NNZ in C is {}", C.nnz);

    // Allocate C.col
    auto* col_ptr = utils::malloc_async(C.nnz * sizeof(*C.col), meta.streams[0]);
    C.col = static_cast<std::int32_t*>(col_ptr);

    // Bin rows over NNZ
    binning<T, Params, BinningType::SYM2>(C, meta, get_value);

    utils::stream_sync(meta.streams[0]);
}

} // namespace proposal
