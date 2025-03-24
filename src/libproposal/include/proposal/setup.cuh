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
                              std::int32_t* __restrict__ nip,
                              std::int32_t* __restrict__ max_nip);

template<std::floating_point T>
void setup(const utils::DeviceCSR<T>& A,
           const utils::DeviceCSR<T>& B,
           utils::DeviceCSR<T>& C,
           Meta& meta) {
    C.m = A.m;
    C.n = B.n;

    for (auto* stream : meta.streams)
        utils::handle_cuda_error(cudaStreamCreate(&stream));

    utils::handle_cuda_error(
        cudaMallocAsync(&C.rpt, (C.m + 1) * sizeof(std::int32_t), meta.streams[0]));
    utils::handle_cuda_error(
        cudaMemsetAsync(C.rpt + C.m, 0, sizeof(std::int32_t), meta.streams[0]));

    // TODO: Compute optimal grid dimensions
    constexpr auto BLOCK_SIZE = 1024;
    const auto n_blocks = cuda::ceil_div(C.m, BLOCK_SIZE);
    k_compute_nip<<<n_blocks, BLOCK_SIZE, 0, meta.streams[0]>>>(A.rpt,
                                                                A.col,
                                                                B.rpt,
                                                                C.m,
                                                                C.rpt,
                                                                C.rpt + C.m);
    utils::handle_cuda_error(cudaDeviceSynchronize());

    cub::DeviceScan::ExclusiveSum(nullptr,
                                  meta.cub_storage_size,
                                  static_cast<std::int32_t*>(nullptr),
                                  static_cast<std::int32_t*>(nullptr),
                                  C.m + 1);
    const auto device_mem_size = C.m * sizeof(std::int32_t) + meta.cub_storage_size;
    utils::handle_cuda_error(
        cudaMallocAsync(&meta.device_ptr, device_mem_size, meta.streams[1]));

    const auto managed_mem_size = (2 * N_BINS + 2) * sizeof(std::int32_t);
    utils::handle_cuda_error(cudaMallocManaged(&meta.managed_ptr, managed_mem_size));
    meta.bin_sizes = static_cast<std::int32_t*>(meta.managed_ptr);
    meta.bin_offsets = meta.bin_sizes + N_BINS;
    meta.max_row_nnz = meta.bin_offsets + N_BINS;
    meta.total_nnz = meta.max_row_nnz + 1;

    utils::handle_cuda_error(cudaMemPrefetchAsync(meta.max_row_nnz,
                                                  sizeof(std::int32_t),
                                                  cudaCpuDeviceId,
                                                  meta.streams[0]));
    utils::handle_cuda_error(cudaMemcpyAsync(meta.max_row_nnz,
                                             C.rpt + C.m,
                                             sizeof(std::int32_t),
                                             cudaMemcpyDeviceToHost,
                                             meta.streams[0]));

    utils::handle_cuda_error(cudaStreamSynchronize(meta.streams[1]));
    meta.d_bins = static_cast<std::int32_t*>(meta.device_ptr);
    meta.d_cub_storage = static_cast<void*>(meta.d_bins + C.m);

    utils::handle_cuda_error(cudaStreamSynchronize(meta.streams[0]));
}
