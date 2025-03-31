#pragma once

#include <concepts>
#include <cstdlib>

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/cleanup.cuh>
#include <proposal/device.cuh>
#include <proposal/meta.cuh>
#include <proposal/num.cuh>
#include <proposal/num_binning.cuh>
#include <proposal/setup.cuh>
#include <proposal/sym.cuh>
#include <proposal/sym2.cuh>
#include <proposal/sym_binning.cuh>

template<std::floating_point T>
auto proposal(const utils::DeviceCSR<T>& A, const utils::DeviceCSR<T>& B)
    -> utils::DeviceCSR<T> {
    NVTX3_FUNC_RANGE();

    utils::DeviceCSR<T> C;
    Meta meta;
    Device device;

    // Setup
    SPDLOG_DEBUG("Starting setup phase");
    setup(A, B, C, meta, device);
    SPDLOG_DEBUG("Finished setup phase");

    // Symbolic binning
    SPDLOG_DEBUG("Starting symbolic binning phase");
    sym_binning(C, meta, device);
    SPDLOG_DEBUG("Finished symbolic binning phase");

    // Symbolic
    SPDLOG_DEBUG("Starting symbolic phase");
    sym(A, B, C, meta, device);
    SPDLOG_DEBUG("Finished symbolic phase");

    // Symbolic binning 2
    SPDLOG_DEBUG("Starting symbolic binning 2 phase");
    sym_binning2(C, meta, device);
    SPDLOG_DEBUG("Finished symbolic binning 2 phase");

    // Symbolic 2
    SPDLOG_DEBUG("Starting symbolic 2 phase");
    sym2(A, B, C, meta, device);
    SPDLOG_DEBUG("Finished symbolic 2 phase");

    // Numeric binning
    SPDLOG_DEBUG("Starting numeric binning phase");
    num_binning(C, meta, device);
    SPDLOG_DEBUG("Finished numeric binning phase");

    // Numeric
    SPDLOG_DEBUG("Starting numeric phase");
    num(A, B, C, meta);
    SPDLOG_DEBUG("Finished numeric phase");

    // Cleanup
    SPDLOG_DEBUG("Starting cleanup phase");
    cleanup(meta);
    SPDLOG_DEBUG("Finished cleanup phase");

    return C;
}
