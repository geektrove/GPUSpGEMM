#pragma once

#include <concepts>
#include <cstdlib>

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/cleanup.cuh>
#include <proposal/meta.cuh>
#include <proposal/setup.cuh>
#include <proposal/sym_binning.cuh>

template<std::floating_point T>
auto proposal(const utils::DeviceCSR<T>& A, const utils::DeviceCSR<T>& B)
    -> utils::DeviceCSR<T> {
    NVTX3_FUNC_RANGE();

    utils::DeviceCSR<T> C;
    Meta meta;

    // Get device properties

    // Setup
    setup(A, B, C, meta);

    // Symbolic binning
    sym_binning(C, meta);

    // Cleanup
    cleanup(meta);

    return C;
}
