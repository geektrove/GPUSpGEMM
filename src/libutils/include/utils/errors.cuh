#pragma once

#include <source_location>

#include <cusparse.h>
#include <fmt/format.h>

namespace utils {

// Checks CUDA error code and throws exception with detailed source location on failure
inline auto handle_cuda_error(
    const cudaError_t status,
    const std::source_location location = std::source_location::current()) -> void {
    if (status == cudaSuccess) [[likely]]
        return;
    const auto* reason{cudaGetErrorString(status)};
    throw std::runtime_error(fmt::format("CUDA error ({}:{}:{}): {}\n",
                                         location.file_name(),
                                         location.line(),
                                         location.function_name(),
                                         reason));
}

// Checks cuSPARSE error code and throws exception with detailed source location on failure
inline auto handle_cusparse_error(
    const cusparseStatus_t status,
    const std::source_location location = std::source_location::current()) -> void {
    if (status == CUSPARSE_STATUS_SUCCESS) [[likely]]
        return;
    const auto* reason{cusparseGetErrorString(status)};
    throw std::runtime_error(fmt::format("cuSPARSE error ({}:{}:{}): {}",
                                         location.file_name(),
                                         location.line(),
                                         location.function_name(),
                                         reason));
}

} // namespace utils
