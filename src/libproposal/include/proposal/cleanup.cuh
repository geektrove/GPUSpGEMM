#pragma once

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>

inline auto cleanup(Meta& meta) -> void {
    NVTX3_FUNC_RANGE();
    SPDLOG_DEBUG("Free device memory asynchronously");
    utils::free_async(meta.d_bins);
    SPDLOG_DEBUG("Free host memory");
    utils::free<utils::Location::Host>(meta.h_bin_sizes);
    SPDLOG_DEBUG("Destroy streams");
    for (auto& stream : meta.streams)
        utils::handle_cuda_error(cudaStreamDestroy(stream));
    utils::stream_sync();
}
