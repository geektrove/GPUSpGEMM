#pragma once

#include <cusparse.h>
#include <fmt/core.h>

#define CHECK_CUSPARSE(value) check_cusparse_error((value), #value, __FILE__, __LINE__)

inline auto check_cusparse_error(const cusparseStatus_t status,
                                 const char* const function,
                                 const char* const file,
                                 const int line) -> void {
    if (status == CUSPARSE_STATUS_SUCCESS)
        return;
    const auto* reason{cusparseGetErrorString(status)};
    const auto message{
        fmt::format("CUSPARSE error ({}:{}:{}): {}", file, line, function, reason)};
    throw std::runtime_error(message);
}
