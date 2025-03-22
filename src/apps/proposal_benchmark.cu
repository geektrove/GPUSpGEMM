#include <chrono>
#include <concepts>
#include <cstdlib>
#include <thread>

#include <benchmark/benchmark.h>
#include <fmt/base.h>

#include <proposal/proposal.cuh>
#include <utils/utils.cuh>

namespace {

constexpr auto SLEEP_TIME = std::chrono::milliseconds(100);

template<std::floating_point T>
void benchmark_proposal(benchmark::State& state,
                        const utils::DeviceCSR<T>& a,
                        const utils::DeviceCSR<T>& b) {
    for (auto _ : state) {
        // Very important observation:
        // It seems like even after the SpGEMM call returns,
        // and the device is synchronized, the device is still
        // busy behind the scenes. So, we need to wait a little
        // bit and let the device finish operations from the
        // previous iteration. It significantly improves performance
        // of each iteration.
        std::this_thread::sleep_for(SLEEP_TIME);

        const auto start = std::chrono::high_resolution_clock::now();
        auto c = proposal(a, b);
        utils::handle_cuda_error(cudaDeviceSynchronize());
        const auto end = std::chrono::high_resolution_clock::now();
        const auto seconds = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(seconds);
    }
}

} // namespace

auto main(int argc, char** argv) -> int {
    using ValueType = double;

    // Load the matrices
    if (argc < 3) {
        fmt::println("Usage: {} <input:A> <input:B> [benchmark-options]", argv[0]);
        return EXIT_FAILURE;
    }
    const auto h_a = utils::HostCSR<ValueType>::load_from_filename(argv[1]);
    const auto h_b = utils::HostCSR<ValueType>::load_from_filename(argv[2]);
    if (h_a.n != h_b.m) {
        fmt::println("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }
    const auto d_a = h_a.to<utils::Location::Device>();
    const auto d_b = h_b.to<utils::Location::Device>();

    // Register the benchmark
    benchmark::RegisterBenchmark("cusparse", benchmark_proposal<ValueType>, d_a, d_b)
        ->UseManualTime();

    // Run the benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::AddCustomContext("Matrix A", argv[1]);
    benchmark::AddCustomContext("Matrix B", argv[2]);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return EXIT_SUCCESS;
}
