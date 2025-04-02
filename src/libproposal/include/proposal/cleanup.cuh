#pragma once

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>
#include <proposal/parameters.cuh>

template<typename Params>
auto cleanup(Meta<Params>& meta) -> void {
    NVTX3_FUNC_RANGE();

    SPDLOG_DEBUG("Free device memory asynchronously");
    utils::free_async(meta.d_ptr);
    if (meta.d_mem_pool != nullptr) {
        SPDLOG_DEBUG("Free device memory pool asynchronously");
        utils::free_async(meta.d_mem_pool);
    }

    SPDLOG_DEBUG("Destroy streams");
    for (auto& stream : meta.streams)
        utils::handle_cuda_error(cudaStreamDestroy(stream));
    SPDLOG_DEBUG("Destroy events");
    for (auto& event : meta.events)
        utils::handle_cuda_error(cudaEventDestroy(event));

    utils::stream_sync();
}
