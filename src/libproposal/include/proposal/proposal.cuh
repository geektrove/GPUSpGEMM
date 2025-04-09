#pragma once

#include <concepts>
#include <cstdlib>
#include <type_traits>

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

#include <proposal/binning.cuh>
#include <proposal/cleanup.cuh>
#include <proposal/meta.cuh>
#include <proposal/num.cuh>
#include <proposal/setup.cuh>
#include <proposal/sym1.cuh>
#include <proposal/sym2.cuh>

namespace proposal {

template<std::floating_point T, typename Params>
requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
auto proposal_inner(const utils::DeviceCSR<T>& A, const utils::DeviceCSR<T>& B)
    -> utils::DeviceCSR<T> {
    NVTX3_FUNC_RANGE();

    utils::DeviceCSR<T> C;
    Meta<Params> meta;

    // Setup
    SPDLOG_DEBUG("Starting setup phase");
    setup<T, Params>(A, B, C, meta);
    SPDLOG_DEBUG("Finished setup phase");

    // Symbolic binning
    SPDLOG_DEBUG("Starting symbolic binning phase");
    auto get_value_individual = [values = C.rpt] __device__(const std::int32_t row) {
        return values[row];
    };
    binning<T, Params, BinningType::SYM1>(C, meta, get_value_individual);
    SPDLOG_DEBUG("Finished symbolic binning phase");

    // Symbolic
    SPDLOG_DEBUG("Starting symbolic phase");
    sym1<T, Params>(A, B, C, meta);
    SPDLOG_DEBUG("Finished symbolic phase");

    // Symbolic binning 2
    SPDLOG_DEBUG("Starting symbolic binning 2 phase");
    auto get_value_difference = [values = C.rpt] __device__(const std::int32_t row) {
        return values[row + 1] - values[row];
    };
    sym_binning2(C, meta, get_value_difference);
    SPDLOG_DEBUG("Finished symbolic binning 2 phase");

    // Symbolic 2
    SPDLOG_DEBUG("Starting symbolic 2 phase");
    sym2<T, Params>(A, B, C, meta);
    SPDLOG_DEBUG("Finished symbolic 2 phase");

    // Numeric binning
    SPDLOG_DEBUG("Starting numeric binning phase");
    if constexpr (std::is_same_v<T, float>) {
        binning<T, Params, BinningType::NUM_F32>(C, meta, get_value_difference);
    } else {
        binning<T, Params, BinningType::NUM_F64>(C, meta, get_value_difference);
    }
    SPDLOG_DEBUG("Finished numeric binning phase");

    // Numeric
    SPDLOG_DEBUG("Starting numeric phase");
    num<T, Params>(A, B, C, meta);
    SPDLOG_DEBUG("Finished numeric phase");

    // Cleanup
    SPDLOG_DEBUG("Starting cleanup phase");
    cleanup(meta);
    SPDLOG_DEBUG("Finished cleanup phase");

    return C;
}

template<std::floating_point T>
requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
auto proposal(const utils::DeviceCSR<T>& A, const utils::DeviceCSR<T>& B)
    -> utils::DeviceCSR<T> {
    NVTX3_FUNC_RANGE();

    const auto CC = std::invoke([&] {
        static constexpr std::int32_t MAJOR_SHIFT = 100;
        static constexpr std::int32_t MINOR_SHIFT = 10;

        int device{};
        utils::handle_cuda_error(cudaGetDevice(&device));
        int major{};
        int minor{};
        utils::handle_cuda_error(
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        utils::handle_cuda_error(
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        return (major * MAJOR_SHIFT) + (minor * MINOR_SHIFT);
    });

#if defined(COMPUTE_80)
    if (CC == CC80)
        return proposal_inner<T, Parameters<CC80>>(A, B);
#endif

#if defined(COMPUTE_86)
    if (CC == CC86)
        return proposal_inner<T, Parameters<CC86>>(A, B);
#endif

    throw std::runtime_error("Unsupported compute capability");
}

} // namespace proposal
