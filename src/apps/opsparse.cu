#include <cstdlib>
#include <functional>
#include <thread>

#include <CLI/CLI.hpp>
#include <fmt/chrono.h>
#include <fmt/core.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <cusparse/cusparse.cuh>
#include <opsparse/conversion.cuh>
#include <opsparse/opsparse.h>
#include <utils/utils.cuh>

constexpr auto DEFAULT_PAUSE_MS = 100;

auto main(int argc, char** argv) -> int {
    using ValueType = double;
    using Clock = std::chrono::steady_clock;

    // Parse command line arguments
    CLI::App app{"OpSparse"};
    app.require_subcommand(1, 1);

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

    app.add_subcommand("validate")
        ->description("Validate the result matrix using cuSPARSE");

    auto* save = app.add_subcommand("save")->description(
        "Save the result matrix to a file");
    save->add_option("output")->description("Path to output matrix")->required();

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

    // Set logging level
    const auto& loglevel = app["--loglevel"]->as<std::string>();
    spdlog::set_level(spdlog::level::from_str(loglevel));

    // Load matrices
    const auto& inputA = app["inputA"]->as<std::string>();
    const auto& inputB = app["inputB"]->as<std::string>();
    auto h_a = utils::HostCSR<ValueType>::load_from_filename(inputA);
    auto h_b = utils::HostCSR<ValueType>::load_from_filename(inputB);
    if (h_a.n != h_b.m) {
        SPDLOG_ERROR("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }
    const auto d_a = h_a.to<utils::Location::Device>();
    const auto d_b = h_b.to<utils::Location::Device>();

    // Convert to OpSparse CSR format
    CSR A = convertFromUtilsCSR(h_a);
    CSR B = convertFromUtilsCSR(h_b);
    h_a.release();
    h_b.release();
    A.H2D();
    B.H2D();

    // Validate
    if (app.got_subcommand("validate")) {
        const auto h_c = std::invoke([&] {
            CSR C;
            Meta meta;
            Timings timing;
            opsparse(A, B, C, meta, timing);
            C.D2H();
            return convertToUtilsCSR(C);
        });
        const auto h_c_cusparse = std::invoke(
            [&] { return cusparse(d_a, d_b).to<utils::Location::Host>(); });
        if (h_c == h_c_cusparse) {
            fmt::println("Validation succeeded");
        } else {
            fmt::println("Validation failed");
            fmt::println("Matrix A: {} x {}", h_a.m, h_a.n);
            fmt::println("Matrix B: {} x {}", h_b.m, h_b.n);
            fmt::println("Matrix C: {} x {}", h_c.m, h_c.n);
            fmt::println("Matrix C (cuSPARSE): {} x {}", h_c_cusparse.m, h_c_cusparse.n);
            fmt::println("Non-zero elements in C: {}", h_c.nnz);
            fmt::println("Non-zero elements in C (cuSPARSE): {}", h_c_cusparse.nnz);
        }
    }

    // Save
    if (app.got_subcommand("save")) {
        const auto& output = save->get_option("output")->as<std::string>();
        fmt::println("Saving result to {}", output);
        const auto h_c = std::invoke([&] {
            CSR C;
            Meta meta;
            Timings timing;
            opsparse(A, B, C, meta, timing);
            C.D2H();
            return convertToUtilsCSR(C);
        });
        h_c.save_to_filename(output);
    }

    // Benchmark
    if (app.got_subcommand("benchmark")) {
        // Initialize NVTX
#ifndef NVTX_DISABLE
        nvtxInitialize(nullptr);
#endif

        const auto& runs = benchmark->get_option("--runs")->as<int>();
        const auto& warmups = benchmark->get_option("--warmups")->as<int>();
        const auto& pause = benchmark->get_option("--pause")->as<int>();

        // Warmup
        for (int i = 0; i < warmups; i++) {
            CSR C;
            Meta meta;
            Timings timing;
            std::this_thread::sleep_for(std::chrono::milliseconds(pause));
            opsparse(A, B, C, meta, timing);
            utils::device_sync();
        }

        // Execute
        std::vector<Clock::duration> times(runs);
        for (int i = 0; i < runs; i++) {
            CSR C;
            Meta meta;
            Timings timing;
            std::this_thread::sleep_for(std::chrono::milliseconds(pause));
            const auto start = Clock::now();
            opsparse(A, B, C, meta, timing);
            utils::device_sync();
            const auto end = Clock::now();
            times[i] = end - start;
        }

        // Print results
        fmt::println("Timestamp: {}", std::chrono::system_clock::now());
        for (int i = 0; i < runs; i++)
            fmt::println("Run {:2d}: {}", i + 1, times[i]);
    }

    return EXIT_SUCCESS;
}
