#pragma once

#include <concepts>
#include <cstdlib>

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/cleanup.cuh>
#include <proposal/device.cuh>
#include <proposal/meta.cuh>
#include <proposal/setup.cuh>
#include <proposal/sym.cuh>
#include <proposal/sym_binning.cuh>

template<std::floating_point T>
auto proposal(const utils::DeviceCSR<T>& A, const utils::DeviceCSR<T>& B)
    -> utils::DeviceCSR<T> {
    NVTX3_FUNC_RANGE();

    utils::DeviceCSR<T> C;
    Meta meta;
    Device device;

    // Setup
    setup(A, B, C, meta, device);

    // Symbolic binning
    sym_binning(C, meta, device);

    // Symbolic
    sym(A, B, C, meta, device);

    // Cleanup
    cleanup(meta);

    return C;
}
