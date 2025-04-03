#pragma once

#include <concepts>
#include <cstdint>
#include <cuda/cmath>
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

namespace cg = cooperative_groups;

template<std::int32_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, get_minctapersm(BLOCK_SIZE)) __global__
    void k_compute_nip(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                       const __grid_constant__ std::int32_t* const __restrict__ a_col,
                       const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                       const __grid_constant__ std::int32_t m,
                       __grid_constant__ std::int32_t* const __restrict__ nips,
                       __grid_constant__ std::int32_t* const __restrict__ max_nip) {
    using ReduceT = cub::BlockReduce<std::int32_t, BLOCK_SIZE>;

    __shared__ typename ReduceT::TempStorage s_storage;

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto tile = cg::tiled_partition<BLOCK_SIZE>(block);
    const auto tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto tiw = gsl::narrow_cast<std::int32_t>(warp.thread_rank());

    const auto row_length = tig < m ? a_rpt[tig + 1] - a_rpt[tig] : 0;
    const auto row_length_max = cg::reduce(warp, row_length, cg::greater<std::int32_t>{});
    auto l_nip = cuda::std::invoke([&] {
        // If the maximum row length across warp is greater than threshold,
        // use a warp per row
        if (row_length_max >= WARP_SIZE) {
            std::int32_t thread_row_nip = 0;
            auto row_begin = utils::divpow2(tig, WARP_SIZE) * WARP_SIZE;
            for (std::int32_t row_offset = 0; row_offset < WARP_SIZE; row_offset++) {
                const auto row = row_begin + row_offset;
                if (row >= m)
                    break;
                std::int32_t l_row_nip = 0;
                for (auto j = a_rpt[row] + tiw; j < a_rpt[row + 1]; j += WARP_SIZE) {
                    const auto col = a_col[j];
                    l_row_nip += b_rpt[col + 1] - b_rpt[col];
                }
                const auto row_nip = cg::reduce(warp,
                                                l_row_nip,
                                                cg::plus<std::int32_t>{});
                if (row_offset == tiw) {
                    thread_row_nip = row_nip;
                    nips[row] = row_nip;
                }
            }
            return thread_row_nip;
        }

        // Otherwise, use a single thread per row
        const auto row = tig;
        if (row >= m)
            return 0;
        std::int32_t row_nip = 0;
        for (auto j = a_rpt[row]; j < a_rpt[row + 1]; j++) {
            const auto col = a_col[j];
            row_nip += b_rpt[col + 1] - b_rpt[col];
        }
        nips[row] = row_nip;
        return row_nip;
    });

    auto l_max_nip = ReduceT(s_storage).Reduce(l_nip, cg::greater<std::int32_t>{});
    cg::invoke_one(block, [&] { atomicMax(max_nip, l_max_nip); });
}

template<typename Params>
void allocate_device_mem(const std::int32_t m, Meta<Params>& meta) {
    // Estimate CUB storage size
    size_t cub_requested{};
    cub::DeviceFor::Bulk(nullptr, cub_requested, m, [] __device__(int) {});
    meta.cub_storage_size = cub_requested;
    cub::DeviceReduce::Max(nullptr,
                           cub_requested,
                           static_cast<std::int32_t*>(nullptr),
                           static_cast<std::int32_t*>(nullptr),
                           m);
    meta.cub_storage_size = std::max(meta.cub_storage_size, cub_requested);
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  cub_requested,
                                  static_cast<std::int32_t*>(nullptr),
                                  static_cast<std::int32_t*>(nullptr),
                                  m + 1);
    meta.cub_storage_size = std::max(meta.cub_storage_size, cub_requested);

    // Allocate device memory
    const auto d_memsize = ((m + 2 * Params::MAX_N_BINS + 2) * sizeof(std::int32_t))
                           + meta.cub_storage_size;
    meta.d_ptr = utils::malloc_async(d_memsize, meta.streams[0]);

    // Assign pointers to the allocated memory
    meta.d_bins = static_cast<std::int32_t*>(meta.d_ptr);
    meta.d_bin_sizes = meta.d_bins + m;
    meta.d_bin_offsets = meta.d_bin_sizes + Params::MAX_N_BINS;
    meta.d_max_row_nnz = meta.d_bin_offsets + Params::MAX_N_BINS;
    meta.d_cub_storage = meta.d_max_row_nnz + 1;

    SPDLOG_DEBUG("Allocated memory on device: {}", d_memsize);
    SPDLOG_DEBUG("-- CUB memory on device: {}", meta.cub_storage_size);
}

template<std::floating_point T, typename Params>
void setup(const utils::DeviceCSR<T>& A,
           const utils::DeviceCSR<T>& B,
           utils::DeviceCSR<T>& C,
           Meta<Params>& meta) {
    NVTX3_FUNC_RANGE();

    // Inititalize C dimensions
    C.m = A.m;
    C.n = B.n;

    // Allocate memory for C.rpt
    auto* rpt = utils::malloc_async((C.m + 1) * sizeof(std::int32_t));
    C.rpt = static_cast<std::int32_t*>(rpt);

    // Compute NIP per row in C and find the maximum
    utils::memset_async(C.rpt + C.m, 0, sizeof(*C.rpt));
    utils::launch_kernel(k_compute_nip<Params::OPTIMAL_BLOCK_SIZE>,
                         cuda::ceil_div(C.m, Params::OPTIMAL_BLOCK_SIZE),
                         Params::OPTIMAL_BLOCK_SIZE,
                         0,
                         cudaStreamDefault,
                         A.rpt,
                         A.col,
                         B.rpt,
                         C.m,
                         C.rpt,
                         C.rpt + C.m);
    utils::memcpy_async(&meta.h_max_row_nnz, C.rpt + C.m, sizeof(meta.h_max_row_nnz));

    // Create CUDA streams
    for (auto& stream : meta.streams)
        utils::handle_cuda_error(cudaStreamCreate(&stream));

    // Allocate device memory
    allocate_device_mem(C.m, meta);

    // Create CUDA events
    for (auto& event : meta.events)
        utils::handle_cuda_error(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    // Synchronize streams
    utils::stream_sync(meta.streams[0]);
    utils::stream_sync();

    SPDLOG_DEBUG("Max NIP per row is {}", meta.h_max_row_nnz);
}
