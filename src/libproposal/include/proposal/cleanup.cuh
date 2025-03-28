#pragma once

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

inline auto cleanup(Meta& meta) -> void {
    NVTX3_FUNC_RANGE();
    SPDLOG_DEBUG("Free device memory asynchronously");
    utils::free_async(meta.d_ptr);
    SPDLOG_DEBUG("Free host memory");
    utils::free<utils::Location::Host>(meta.h_ptr);
    SPDLOG_DEBUG("Destroy streams");
    for (std::int32_t i = 0; i < meta.n_bins; i++)
        utils::handle_cuda_error(cudaStreamDestroy(meta.streams[i]));
    utils::free<utils::Location::Host>(static_cast<void*>(meta.streams));
    SPDLOG_DEBUG("Destroy events");
    for (auto& event : meta.events)
        utils::handle_cuda_error(cudaEventDestroy(event));
    utils::stream_sync();
}
