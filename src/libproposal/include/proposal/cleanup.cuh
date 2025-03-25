#pragma once

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

inline auto cleanup(Meta& meta) -> void {
    NVTX3_FUNC_RANGE();
    SPDLOG_DEBUG("Free device memory asynchronously");
    utils::handle_cuda_error(cudaFreeAsync(meta.d_bins, cudaStreamDefault));
    utils::handle_cuda_error(cudaDeviceSynchronize());
    SPDLOG_DEBUG("Free host memory");
    utils::handle_cuda_error(cudaFreeHost(meta.h_bin_sizes));
    utils::handle_cuda_error(cudaDeviceSynchronize());
    SPDLOG_DEBUG("Destroy streams");
    for (auto& stream : meta.streams)
        utils::handle_cuda_error(cudaStreamDestroy(stream));
    utils::handle_cuda_error(cudaStreamSynchronize(cudaStreamDefault));
}
