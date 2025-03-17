#pragma once

#include <fmt/core.h>

#define CHECK_CUDA(value) check_cuda_error((value), #value, __FILE__, __LINE__)
#define CHECK_LAST_CUDA() check_last_cuda_error(__FILE__, __LINE__)

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
