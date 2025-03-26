#include <chrono>
#include <cstdlib>
#include <thread>

#include <benchmark/benchmark.h>
#include <fmt/base.h>
#include <spdlog/cfg/env.h>

#include <opsparse/conversion.cuh>
#include <opsparse/opsparse.h>
#include <utils/utils.cuh>

namespace {

constexpr int WARMUP_ITERATIONS = 10;
constexpr auto SLEEP_TIME = std::chrono::milliseconds(100);

template<std::floating_point T>
void benchmark_opsparse(benchmark::State& state,
                        const utils::HostCSR<T>& a,
                        const utils::HostCSR<T>& b) {
    // Convert to OpSparse CSR format
    CSR A = convertFromUtilsCSR(a);
    CSR B = convertFromUtilsCSR(b);
    A.H2D();
    B.H2D();

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        CSR C;
        Meta meta;
        Timings timing;
        std::this_thread::sleep_for(SLEEP_TIME);
        opsparse(A, B, C, meta, timing);
        utils::device_sync();
    }

    for (auto _ : state) {
        CSR C;
        Meta meta;
        Timings timing;

        // Very important observation:
        // It seems like even after the SpGEMM call returns,
        // and the device is synchronized, the device is still
        // busy behind the scenes. So, we need to wait a little
        // bit and let the device finish operations from the
        // previous iteration. It significantly improves performance
        // of each iteration.
        std::this_thread::sleep_for(SLEEP_TIME);

        const auto start = std::chrono::high_resolution_clock::now();
        opsparse(A, B, C, meta, timing);
        utils::device_sync();
        const auto end = std::chrono::high_resolution_clock::now();
        const auto seconds = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(seconds);
    }
}

} // namespace

auto main(int argc, char** argv) -> int {
    using ValueType = double;

    // Initialize logging
    spdlog::cfg::load_env_levels();

    // Load the matrices
    if (argc < 3) {
        fmt::println("Usage: {} <input:A> <input:B> [benchmark-options]", argv[0]);
        return EXIT_FAILURE;
    }
    auto h_a = utils::HostCSR<ValueType>::load_from_filename(argv[1]);
    auto h_b = utils::HostCSR<ValueType>::load_from_filename(argv[2]);
    if (h_a.n != h_b.m) {
        fmt::println("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }

    // Register the benchmark
    benchmark::RegisterBenchmark("opsparse", benchmark_opsparse<ValueType>, h_a, h_b)
        ->UseManualTime();

    // Run the benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::AddCustomContext("Matrix A", argv[1]);
    benchmark::AddCustomContext("Matrix B", argv[2]);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return EXIT_SUCCESS;
}
