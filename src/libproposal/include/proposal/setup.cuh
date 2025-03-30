#pragma once

#include <bit>
#include <concepts>
#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda/cmath>
#include <cuda/std/functional>
#include <gsl/gsl-lite.hpp>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/device.cuh>
#include <proposal/meta.cuh>

namespace cg = cooperative_groups;

template<std::int32_t BLOCK_SIZE>
__global__ void k_compute_nip(
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ nips,
    __grid_constant__ std::int32_t* const __restrict__ max_nip) {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<WARP_SIZE>(block);
    const auto tile = cg::tiled_partition<BLOCK_SIZE>(block);

    const auto& tig = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto& tiw = gsl::narrow_cast<std::int32_t>(warp.thread_rank());

    const auto row_length = tig < m ? a_rpt[tig + 1] - a_rpt[tig] : 0;
    const auto row_length_max = cg::reduce(warp, row_length, cg::greater<std::int32_t>{});
    const auto l_nip = cuda::std::invoke([&] {
        // If the maximum row length across wap is greater than threshold,
        // use a warp per row
        if (row_length_max >= WARP_SIZE) {
            std::int32_t thread_row_nip = 0;
            auto row_begin = (tig / WARP_SIZE) * WARP_SIZE;
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

    cuda::atomic_ref<std::int32_t, cuda::thread_scope_device> max_nip_ref{*max_nip};
    cg::reduce_update_async(tile, max_nip_ref, l_nip, cg::greater<std::int32_t>{});
}

template<std::floating_point T>
void h_compute_nip(const utils::DeviceCSR<T>& A,
                   const utils::DeviceCSR<T>& B,
                   utils::DeviceCSR<T>& C,
                   const Device& device) {
    static constexpr std::int32_t BLOCK512 = 512;
    static constexpr std::int32_t BLOCK1024 = 1024;

    // Realistically only 512 and 1024 block sizes are optimal
    // starting from compute capability 1.2, but in any other case,
    // we can use 1024 threads per block as a fallback
    switch (device.optimal_block_size) {
    case BLOCK512:
        utils::launch_kernel(k_compute_nip<BLOCK512>,
                             cuda::ceil_div(C.m, BLOCK512),
                             BLOCK512,
                             0,
                             cudaStreamDefault,
                             A.rpt,
                             A.col,
                             B.rpt,
                             C.m,
                             C.rpt,
                             C.rpt + C.m);
        break;
    default:
        utils::launch_kernel(k_compute_nip<BLOCK1024>,
                             cuda::ceil_div(C.m, BLOCK1024),
                             BLOCK1024,
                             0,
                             cudaStreamDefault,
                             A.rpt,
                             A.col,
                             B.rpt,
                             C.m,
                             C.rpt,
                             C.rpt + C.m);
        break;
    }
}

template<std::floating_point T>
void setup(const utils::DeviceCSR<T>& A,
           const utils::DeviceCSR<T>& B,
           utils::DeviceCSR<T>& C,
           Meta& meta,
           Device& device) {
    NVTX3_FUNC_RANGE();

    // Initialize C dimensions
    C.m = A.m;
    C.n = B.n;

    // Allocate memory for C.rpt and initialize it to zero
    auto* rpt = utils::malloc_async((C.m + 1) * sizeof(std::int32_t));
    C.rpt = static_cast<std::int32_t*>(rpt);
    utils::memset_async(C.rpt + C.m, 0, sizeof(std::int32_t));

    // Get device properties and compute optimal block size
    int id{};
    utils::handle_cuda_error(cudaGetDevice(&id));
    utils::handle_cuda_error(
        cudaDeviceGetAttribute(&device.n_sm, cudaDevAttrMultiProcessorCount, id));
    utils::handle_cuda_error(
        cudaDeviceGetAttribute(&device.max_threads_per_sm,
                               cudaDevAttrMaxThreadsPerMultiProcessor,
                               id));
    utils::handle_cuda_error(cudaDeviceGetAttribute(&device.max_threads_per_block,
                                                    cudaDevAttrMaxThreadsPerBlock,
                                                    id));
    utils::handle_cuda_error(cudaDeviceGetAttribute(&device.max_blocks_per_sm,
                                                    cudaDevAttrMaxBlocksPerMultiprocessor,
                                                    id));
    utils::handle_cuda_error(
        cudaDeviceGetAttribute(&device.smem_per_sm,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                               id));
    utils::handle_cuda_error(
        cudaDeviceGetAttribute(&device.smem_per_block_reserved,
                               cudaDevAttrReservedSharedMemoryPerBlock,
                               id));
    utils::handle_cuda_error(
        cudaDeviceGetAttribute(&device.max_smem_per_block,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               id));

    device.optimal_block_size = std::invoke([&] {
        auto block_size = device.max_threads_per_block;
        while (device.max_threads_per_sm % block_size != 0)
            block_size /= 2;
        return block_size;
    });

    SPDLOG_DEBUG("Number of SMs: {}", device.n_sm);
    SPDLOG_DEBUG("Max threads per SM: {}", device.max_threads_per_sm);
    SPDLOG_DEBUG("Max threads per block: {}", device.max_threads_per_block);
    SPDLOG_DEBUG("Max blocks per SM: {}", device.max_blocks_per_sm);
    SPDLOG_DEBUG("Shared memory per SM: {}", device.smem_per_sm);
    SPDLOG_DEBUG("Shared memory per block (reserved): {}",
                 device.smem_per_block_reserved);
    SPDLOG_DEBUG("Shared memory per block (max opt in): {}", device.max_smem_per_block);
    SPDLOG_DEBUG("Optimal block size: {}", device.optimal_block_size);

    // Compute NIP per row in C
    h_compute_nip(A, B, C, device);

    // Calculate number of bins
    meta.n_bins = 2; // First bin (PWARP) and last bin (max SMEM) are always present

    // Minimum block size is MAX_THREADS_PER_SM / MAX_BLOCKS_PER_SM
    // rounded up to the nearest power of 2
    const auto min_block_size = gsl::narrow_cast<std::int32_t>(
        std::bit_ceil(gsl::narrow_cast<std::uint32_t>(device.max_threads_per_sm
                                                      / device.max_blocks_per_sm)));

    // Add one bin for each power of 2 block size
    // [MIN_BLOCK_SIZE, MAX_THREADS_PER_BLOCK]
    meta.n_bins += utils::ilog2(device.max_threads_per_block / min_block_size) + 1;

    // Add one bin for each power of 2 table size with MAX_THREADS_PER_BLOCK
    meta.n_bins += utils::ilog2(device.max_threads_per_sm / device.max_threads_per_block);

    SPDLOG_DEBUG("Number of bins: {}", meta.n_bins);

    // Create CUDA streams
    auto* streams_ptr = utils::malloc<utils::Location::Host>(meta.n_bins
                                                             * sizeof(cudaStream_t));
    meta.streams = static_cast<cudaStream_t*>(streams_ptr);
    for (std::int32_t i = 0; i < meta.n_bins; i++)
        utils::handle_cuda_error(cudaStreamCreate(&meta.streams[i]));

    // Estimate CUB storage size
    size_t cub_requested{};
    cub::DeviceFor::Bulk(nullptr, cub_requested, C.m, [] __device__(int) {});
    meta.cub_storage_size = cub_requested;
    cub::DeviceReduce::Max(nullptr,
                           cub_requested,
                           static_cast<std::int32_t*>(nullptr),
                           static_cast<std::int32_t*>(nullptr),
                           C.m);
    meta.cub_storage_size = std::max(meta.cub_storage_size, cub_requested);
    cub::DeviceReduce::Sum(nullptr,
                           cub_requested,
                           static_cast<std::int32_t*>(nullptr),
                           static_cast<std::int32_t*>(nullptr),
                           C.m);
    meta.cub_storage_size = std::max(meta.cub_storage_size, cub_requested);
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  cub_requested,
                                  static_cast<std::int32_t*>(nullptr),
                                  static_cast<std::int32_t*>(nullptr),
                                  C.m + 1);
    meta.cub_storage_size = std::max(meta.cub_storage_size, cub_requested);

    // Allocate device memory
    const auto d_memsize = (C.m + 3 * meta.n_bins + 2) * sizeof(std::int32_t)
                           + meta.cub_storage_size;
    meta.d_ptr = utils::malloc_async(d_memsize, meta.streams[0]);
    meta.d_bins = static_cast<std::int32_t*>(meta.d_ptr);
    meta.d_sym_bin_ranges = meta.d_bins + C.m;
    meta.d_bin_sizes = meta.d_sym_bin_ranges + meta.n_bins;
    meta.d_bin_offsets = meta.d_bin_sizes + meta.n_bins;
    meta.d_max_row_nnz = meta.d_bin_offsets + meta.n_bins;
    meta.d_total_nnz = meta.d_max_row_nnz + 1;
    meta.d_cub_storage = meta.d_total_nnz + 1;

    SPDLOG_DEBUG("Allocated memory on device: {}", d_memsize);
    SPDLOG_DEBUG("-- CUB memory on device: {}", meta.cub_storage_size);

    // Allocate host memory
    const auto h_memsize = (5 * meta.n_bins + 2) * sizeof(std::int32_t);
    meta.h_ptr = utils::malloc<utils::Location::Host>(h_memsize);
    meta.sym_block_sizes = static_cast<std::int32_t*>(meta.h_ptr);
    meta.sym_table_sizes = meta.sym_block_sizes + meta.n_bins;
    meta.h_sym_bin_ranges = meta.sym_table_sizes + meta.n_bins;
    meta.h_bin_sizes = meta.h_sym_bin_ranges + meta.n_bins;
    meta.h_bin_offsets = meta.h_bin_sizes + meta.n_bins;
    meta.h_max_row_nnz = meta.h_bin_offsets + meta.n_bins;
    meta.h_total_nnz = meta.h_max_row_nnz + 1;

    SPDLOG_DEBUG("Allocated memory on host: {}", h_memsize);

    // Copy the maximum NIP per row in C to the host
    utils::memcpy_async(meta.h_max_row_nnz, C.rpt + C.m, sizeof(std::int32_t));

    // Calculate table sizes and ranges
    auto calculate_smem_size = [&](std::int32_t n_blocks) {
        auto smem = device.smem_per_sm;                    // Total SMEM per SM
        smem -= n_blocks * device.smem_per_block_reserved; // Available SMEM per SM
        smem /= n_blocks;                                  // SMEM per block
        smem = std::min(smem, device.max_smem_per_block);  // Max SMEM per block (opt in)
        return smem;
    };
    auto calculate_sym_table_size = [&](std::int32_t n_blocks, bool round_down = true) {
        const auto smem = calculate_smem_size(n_blocks);
        auto table_size = smem / sizeof(std::int32_t);
        if (round_down)
            // Round down to the nearest power of 2
            table_size = std::bit_floor(table_size);
        return table_size;
    };

    meta.sym_block_sizes[0] = device.optimal_block_size;
    meta.sym_table_sizes[0] = calculate_sym_table_size(device.max_threads_per_sm
                                                       / meta.sym_block_sizes[0]);
    meta.sym_block_sizes[1] = min_block_size;
    meta.sym_table_sizes[1] = calculate_sym_table_size(device.max_threads_per_sm
                                                       / meta.sym_block_sizes[1]);
    std::int32_t i = 2;
    for (; meta.sym_block_sizes[i - 1] < device.max_threads_per_block; i++) {
        meta.sym_block_sizes[i] = meta.sym_block_sizes[i - 1] * 2;
        meta.sym_table_sizes[i] = calculate_sym_table_size(device.max_threads_per_sm
                                                           / meta.sym_block_sizes[i]);
    }
    for (auto n_blocks = device.max_threads_per_sm / device.max_threads_per_block / 2;
         n_blocks > 0;
         n_blocks /= 2) {
        meta.sym_block_sizes[i] = device.max_threads_per_block;
        meta.sym_table_sizes[i] = calculate_sym_table_size(n_blocks);
        i++;
    }
    meta.sym_block_sizes[meta.n_bins - 1] = device.max_threads_per_block;
    meta.sym_table_sizes[meta.n_bins - 1] = calculate_sym_table_size(1, false) - 1;

    meta.h_sym_bin_ranges[0] = gsl::narrow_cast<std::int32_t>(
        SYM_RANGE_RATIO
        * gsl::narrow_cast<double>(meta.sym_table_sizes[0]
                                   / (meta.sym_block_sizes[0] / PWARP_SIZE)));
    for (i = 1; i + 1 < meta.n_bins; i++)
        meta.h_sym_bin_ranges[i] = gsl::narrow_cast<std::int32_t>(
            SYM_RANGE_RATIO * meta.sym_table_sizes[i]);
    meta.h_sym_bin_ranges[meta.n_bins - 1] = std::numeric_limits<std::int32_t>::max();

    SPDLOG_DEBUG("{:>12s} {:>12s} {:>12s} {:>12s}",
                 "Bin",
                 "Block size",
                 "Table size",
                 "Sym range");
    for (i = 0; i < meta.n_bins; i++) {
        SPDLOG_DEBUG("{:12d} {:12d} {:12d} {:12d}",
                     i,
                     meta.sym_block_sizes[i],
                     meta.sym_table_sizes[i],
                     meta.h_sym_bin_ranges[i]);
    }

    // Copy the symbolic bin ranges to the device
    utils::memcpy_async(meta.d_sym_bin_ranges,
                        meta.h_sym_bin_ranges,
                        meta.n_bins * sizeof(std::int32_t),
                        meta.streams[0]);

    // Create CUDA events
    for (auto& event : meta.events)
        utils::handle_cuda_error(
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    // Synchronize streams
    utils::stream_sync(meta.streams[0]);
    utils::stream_sync();

    SPDLOG_DEBUG("Maximum NIP per row is {}", *meta.h_max_row_nnz);
}
