#pragma once

#include <concepts>
#include <cstdint>

#include <cub/cub.cuh>
#include <cuda/cmath>
#include <nvtx3/nvtx3.hpp>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

__global__ void k_compute_nip(
    const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
    const __grid_constant__ std::int32_t* const __restrict__ a_col,
    const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
    const __grid_constant__ std::int32_t m,
    __grid_constant__ std::int32_t* const __restrict__ nips,
    __grid_constant__ std::int32_t* const __restrict__ max_nip);

template<std::floating_point T>
void setup(const utils::DeviceCSR<T>& A,
           const utils::DeviceCSR<T>& B,
           utils::DeviceCSR<T>& C,
           Meta& meta) {
    NVTX3_FUNC_RANGE();

    C.m = A.m;
    C.n = B.n;
    auto* rpt = utils::malloc_async((C.m + 1) * sizeof(std::int32_t));
    C.rpt = static_cast<std::int32_t*>(rpt);
    utils::memset_async(C.rpt + C.m, 0, sizeof(std::int32_t));

    // TODO: Compute optimal grid dimensions
    static constexpr auto BLOCK_SIZE = 1024;
    const auto n_blocks = cuda::ceil_div(C.m, BLOCK_SIZE);
    k_compute_nip<<<n_blocks, BLOCK_SIZE>>>(A.rpt, A.col, B.rpt, C.m, C.rpt, C.rpt + C.m);

    for (auto& stream : meta.streams)
        utils::handle_cuda_error(cudaStreamCreate(&stream));

    cub::DeviceScan::ExclusiveSum(nullptr,
                                  meta.cub_storage_size,
                                  static_cast<std::int32_t*>(nullptr),
                                  static_cast<std::int32_t*>(nullptr),
                                  C.m + 1);
    const auto d_memsize = (C.m + 2 * N_BINS + 2) * sizeof(std::int32_t)
                           + meta.cub_storage_size;
    auto* d_ptr = utils::malloc_async(d_memsize, meta.streams[0]);
    meta.d_bins = static_cast<std::int32_t*>(d_ptr);
    meta.d_bin_sizes = meta.d_bins + C.m;
    meta.d_bin_offsets = meta.d_bin_sizes + N_BINS;
    meta.d_max_row_nnz = meta.d_bin_offsets + N_BINS;
    meta.d_total_nnz = meta.d_max_row_nnz + 1;

    auto* h_ptr = utils::malloc<utils::Location::Host>((2 * N_BINS + 2)
                                                       * sizeof(std::int32_t));
    meta.h_bin_sizes = static_cast<std::int32_t*>(h_ptr);
    meta.h_bin_offsets = meta.h_bin_sizes + N_BINS;
    meta.h_max_row_nnz = meta.h_bin_offsets + N_BINS;
    meta.h_total_nnz = meta.h_max_row_nnz + 1;

    utils::memcpy_async(meta.h_max_row_nnz, C.rpt + C.m, sizeof(std::int32_t));

    utils::stream_sync(meta.streams[0]);
    utils::stream_sync();
}
