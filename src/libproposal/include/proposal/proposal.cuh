#pragma once

#include <concepts>
#include <cstdlib>

#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/meta.cuh>
#include <proposal/setup.cuh>
#include <proposal/sym_binning.cuh>

template<std::floating_point T>
auto proposal(const utils::DeviceCSR<T>& A, const utils::DeviceCSR<T>& B)
    -> utils::DeviceCSR<T> {
    Meta meta;
    utils::DeviceCSR<T> C;

    // Get device properties

    // Setup
    setup(A, B, C, meta);
    SPDLOG_INFO("Maximum NIP per row in C: {}", *meta.h_max_row_nnz);

    // Symbolic binning
    sym_binning(C, meta);

    return C;
}
