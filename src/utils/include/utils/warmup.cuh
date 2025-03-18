#pragma once

#include <utils/errors.cuh>

namespace utils {

inline void cudaruntime_warmup() {
    int* d{};
    handle_cuda_error(cudaMalloc(&d, 4));
    handle_cuda_error(cudaFree(d));
    handle_cuda_error(cudaDeviceSynchronize());
}

} // namespace utils
