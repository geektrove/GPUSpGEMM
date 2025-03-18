#pragma once

#include <source_location>

#include <cusparse.h>
#include <fmt/core.h>

#define CHECK_CUDA(value) utils::check_cuda_error((value), #value, __FILE__, __LINE__)
#define CHECK_LAST_CUDA() utils::check_last_cuda_error(__FILE__, __LINE__)

namespace utils {

inline auto check_cuda_error(const cudaError_t error,
                             const char* const function,
                             const char* const file,
                             const int line) -> void {
    if (error == cudaSuccess)
        return;
    const auto* reason{cudaGetErrorString(error)};
    const auto message{
        fmt::format("CUDA error ({}:{}:{}): {}\n", file, line, function, reason)};
    throw std::runtime_error(message);
}

inline auto check_last_cuda_error(const char* const file, const int line) -> void {
    const cudaError_t error{cudaGetLastError()};
    check_cuda_error(error, "LAST CUDA ERROR", file, line);
}

inline auto handle_cusparse_error(
    const cusparseStatus_t status,
    const std::source_location location = std::source_location::current()) -> void {
    if (status == CUSPARSE_STATUS_SUCCESS)
        return;
    const auto* reason{cusparseGetErrorString(status)};
    throw std::runtime_error(fmt::format("cuSPARSE error ({}:{}:{}): {}",
                                         location.file_name(),
                                         location.line(),
                                         location.function_name(),
                                         reason));
}

} // namespace utils
