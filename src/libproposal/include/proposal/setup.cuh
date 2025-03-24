#pragma once

#include <concepts>
#include <cstdint>

#include <cub/cub.cuh>
#include <cuda/cmath>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

__global__ void k_compute_nip(const std::int32_t* __restrict__ a_rpt,
                              const std::int32_t* __restrict__ a_col,
                              const std::int32_t* __restrict__ b_rpt,
                              std::int32_t m,
                              std::int32_t* __restrict__ nips,
                              std::int32_t* __restrict__ max_nip);

template<std::floating_point T>
void setup(const utils::DeviceCSR<T>& A,
           const utils::DeviceCSR<T>& B,
           utils::DeviceCSR<T>& C,
           Meta& meta) {
    C.m = A.m;
    C.n = B.n;

    utils::handle_cuda_error(
        cudaMallocAsync(&C.rpt, (C.m + 1) * sizeof(std::int32_t), cudaStreamDefault));
    utils::handle_cuda_error(cudaMemsetAsync(C.rpt + C.m, 0, sizeof(std::int32_t)));

    // TODO: Compute optimal grid dimensions
    static constexpr auto BLOCK_SIZE = 1024;
    const auto n_blocks = cuda::ceil_div(C.m, BLOCK_SIZE);
    k_compute_nip<<<n_blocks, BLOCK_SIZE>>>(A.rpt, A.col, B.rpt, C.m, C.rpt, C.rpt + C.m);

    for (auto& stream : meta.streams)
        utils::handle_cuda_error(cudaStreamCreate(&stream));

    utils::handle_cuda_error(
        cub::DeviceScan::ExclusiveSum(nullptr,
                                      meta.cub_storage_size,
                                      static_cast<std::int32_t*>(nullptr),
                                      static_cast<std::int32_t*>(nullptr),
                                      C.m + 1));
    const auto d_memsize = (C.m + 2 * N_BINS + 2) * sizeof(std::int32_t)
                           + meta.cub_storage_size;
    utils::handle_cuda_error(cudaMallocAsync(&meta.d_bins, d_memsize, meta.streams[0]));

    const auto h_memsize = (2 * N_BINS + 2) * sizeof(std::int32_t);
    utils::handle_cuda_error(cudaMallocHost(&meta.h_bin_sizes, h_memsize));
    meta.h_bin_offsets = meta.h_bin_sizes + N_BINS;
    meta.h_max_row_nnz = meta.h_bin_offsets + N_BINS;
    meta.h_total_nnz = meta.h_max_row_nnz + 1;

    utils::handle_cuda_error(cudaMemcpyAsync(meta.h_max_row_nnz,
                                             C.rpt + C.m,
                                             sizeof(std::int32_t),
                                             cudaMemcpyDeviceToHost));

    utils::handle_cuda_error(cudaStreamSynchronize(meta.streams[0]));
    meta.d_bin_sizes = meta.d_bins + C.m;
    meta.d_bin_offsets = meta.d_bin_sizes + N_BINS;
    meta.d_max_row_nnz = meta.d_bin_offsets + N_BINS;
    meta.d_total_nnz = meta.d_max_row_nnz + 1;

    utils::handle_cuda_error(cudaStreamSynchronize(cudaStreamDefault));
}
