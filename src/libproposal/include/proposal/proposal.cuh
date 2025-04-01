#pragma once

#include <concepts>
#include <cstdlib>

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/binning.cuh>
#include <proposal/cleanup.cuh>
#include <proposal/device.cuh>
#include <proposal/meta.cuh>
#include <proposal/num.cuh>
#include <proposal/setup.cuh>
#include <proposal/sym.cuh>
#include <proposal/sym2.cuh>

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
    auto get_value_sym = [values = C.rpt] __device__(const std::int32_t row) {
        return values[row];
    };
    binning(C, meta, device, get_value_sym);
    SPDLOG_DEBUG("Finished symbolic binning phase");

    // Symbolic
    SPDLOG_DEBUG("Starting symbolic phase");
    sym(A, B, C, meta);
    SPDLOG_DEBUG("Finished symbolic phase");

    // Symbolic binning 2
    SPDLOG_DEBUG("Starting symbolic binning 2 phase");
    sym_binning2(C, meta, device, get_value_sym);
    SPDLOG_DEBUG("Finished symbolic binning 2 phase");

    // Symbolic 2
    SPDLOG_DEBUG("Starting symbolic 2 phase");
    sym2(A, B, C, meta, device);
    SPDLOG_DEBUG("Finished symbolic 2 phase");

    // Numeric binning
    SPDLOG_DEBUG("Starting numeric binning phase");
    auto get_value_num = [values = C.rpt] __device__(const std::int32_t row) {
        return values[row + 1] - values[row];
    };
    binning(C, meta, device, get_value_num);
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
