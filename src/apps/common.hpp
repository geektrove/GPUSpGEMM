#pragma once

#include <cstdlib>
#include <functional>
#include <thread>

#include <CLI/CLI.hpp>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <cusparse/cusparse.cuh>
#include <utils/utils.cuh>

// Template function that implements the common application workflow for sparse matrix multiplication algorithms
// Parameters:
// - ValueType: The floating-point type used (float or double)
// - ConvertToAppDeviceCSR: Function that converts host matrix to device matrix in the format needed by the algorithm
// - RunT: Function that executes the matrix multiplication and returns result
// - MeasureT: Function that measures execution time
template<typename ValueType,
         typename ConvertToAppDeviceCSR,
         typename RunT,
         typename MeasureT>
auto run_app(const std::string& name,
             ConvertToAppDeviceCSR convert_to_app_device_csr,
             RunT run,
             MeasureT measure,
             int argc,
             char** argv) -> int {
    using Clock = std::chrono::steady_clock;

    static constexpr auto DEFAULT_PAUSE_MS = 50;

    //
    // Parse command line arguments
    //

    CLI::App app{name};
    app.require_subcommand(1, 1); // Require exactly one subcommand

    app.add_option("inputA")
        ->description("Path to input matrix A")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("inputB")
        ->description("Path to input matrix B")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--loglevel")
        ->description("Set the logging level")
        ->check(
            CLI::IsMember({"trace", "debug", "info", "warn", "error", "critical", "off"}))
        ->default_str("info");

    // Subcommand to validate results against cuSPARSE 2 reference implementation
    app.add_subcommand("validate")
        ->description("Validate the result matrix using cuSPARSE 2");

    // Subcommand to save the result matrix to a file
    auto* save = app.add_subcommand("save")->description(
        "Save the result matrix to a file");
    save->add_option("output")->description("Path to output matrix")->required();

    // Subcommand to benchmark the algorithm
    auto* benchmark =
        app.add_subcommand("benchmark")->description("Benchmark the algorithm");
    benchmark->add_option("--runs")
        ->description("Number of runs to perform")
        ->default_val(1)
        ->check(CLI::NonNegativeNumber);
    benchmark->add_option("--warmups")
        ->description("Number of warmup iterations before the actual runs")
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    benchmark->add_option("--pause")
        ->description("Pause between iterations in milliseconds")
        ->default_val(DEFAULT_PAUSE_MS)
        ->check(CLI::NonNegativeNumber);

    CLI11_PARSE(app, argc, argv);

    //
    // Set logging level
    //

    const auto& loglevel = app["--loglevel"]->as<std::string>();
    spdlog::set_level(spdlog::level::from_str(loglevel));

    //
    // Load matrices
    //

    const auto& inputA = app["inputA"]->as<std::string>();
    const auto& inputB = app["inputB"]->as<std::string>();
    const auto h_a = utils::HostCSR<ValueType>::load_from_filename(inputA);
    const auto h_b = utils::HostCSR<ValueType>::load_from_filename(inputB);
    if (h_a.n != h_b.m) {
        SPDLOG_ERROR("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }

    //
    // Validate
    //

    if (app.got_subcommand("validate")) {
        // Run the implementation
        const auto h_c = std::invoke([&] {
            const auto app_d_a = convert_to_app_device_csr(h_a);
            const auto app_d_b = convert_to_app_device_csr(h_b);
            return run(app_d_a, app_d_b);
        });
        // Run cuSPARSE 2 as reference implementation
        const auto h_c_cusparse = std::invoke([&] {
            const auto d_a = h_a.template to<utils::Location::Device>();
            const auto d_b = h_b.template to<utils::Location::Device>();
            return cusparse2(d_a, d_b).template to<utils::Location::Host>();
        });

        // Compare results
        if (h_c == h_c_cusparse) {
            fmt::println("Validation succeeded");
        } else {
            fmt::println("Validation failed");
            fmt::println("Matrix C: {} x {} ({} non-zero elements)",
                         h_c.m,
                         h_c.n,
                         h_c.nnz);
            fmt::println("Matrix C (cuSPARSE): {} x {} ({} non-zero elements)",
                         h_c_cusparse.m,
                         h_c_cusparse.n,
                         h_c_cusparse.nnz);
        }
    }

    //
    // Save
    //

    if (app.got_subcommand("save")) {
        const auto& output = save->get_option("output")->as<std::string>();
        fmt::println("Saving result to {}", output);
        // Run the implementation and save the result
        const auto h_c = std::invoke([&] {
            const auto app_d_a = convert_to_app_device_csr(h_a);
            const auto app_d_b = convert_to_app_device_csr(h_b);
            return run(app_d_a, app_d_b);
        });
        h_c.save_to_filename(output);
    }

    //
    // Benchmark
    //

    if (app.got_subcommand("benchmark")) {
        // Initialize NVTX for NVIDIA profiling tools
#ifndef NVTX_DISABLE
        nvtxInitialize(nullptr);
#endif

        const auto& runs = benchmark->get_option("--runs")->as<int>();
        const auto& warmups = benchmark->get_option("--warmups")->as<int>();
        const auto& pause = benchmark->get_option("--pause")->as<int>();
        const auto pause_ms = std::chrono::milliseconds(pause);

        // Convert matrices to device format once
        const auto app_d_a = convert_to_app_device_csr(h_a);
        const auto app_d_b = convert_to_app_device_csr(h_b);

        // Warmup runs to stabilize GPU performance
        for (int i = 0; i < warmups; i++) {
            std::this_thread::sleep_for(pause_ms);
            measure(app_d_a, app_d_b);
        }

        // Execute the actual benchmark runs
        std::vector<Clock::duration> times(runs);
        for (int i = 0; i < runs; i++) {
            std::this_thread::sleep_for(pause_ms);
            times[i] = measure(app_d_a, app_d_b);
        }

        // Compute NIP, FLOPs and NNZ
        const auto nip = std::invoke([&] {
            const auto d_a = h_a.template to<utils::Location::Device>();
            const auto d_b = h_b.template to<utils::Location::Device>();
            return utils::get_nip(d_a, d_b);
        });
        const auto flop = 2.0 * nip; // Each inner product requires a multiply and add
        const auto h_c = run(app_d_a, app_d_b);

        // Print benchmark results
        fmt::println("A: {} x {} ({} non-zero elements)", h_a.m, h_a.n, h_a.nnz);
        fmt::println("B: {} x {} ({} non-zero elements)", h_b.m, h_b.n, h_b.nnz);
        fmt::println("NIP: {}", nip);
        fmt::println("C: {} x {} ({} non-zero elements)", h_c.m, h_c.n, h_c.nnz);
        for (int i = 0; i < runs; i++) {
            const auto seconds = std::chrono::duration<double>(times[i]).count();
            const auto flops = flop / seconds;
            fmt::println("Run {:2d}: {} | {} FLOPS", i + 1, times[i], flops);
        }
    }

    return EXIT_SUCCESS;
}
