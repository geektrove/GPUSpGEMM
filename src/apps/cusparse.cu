#include <cstdlib>
#include <thread>

#include <fmt/base.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/cfg/env.h>

#include <cusparse/cusparse.cuh>
#include <utils/utils.cuh>

constexpr auto SLEEP_TIME = std::chrono::milliseconds(100);

auto main(int argc, char** argv) -> int {
    using ValueType = double;

    // Initialize logging
    spdlog::cfg::load_env_levels();

    // Initialize NVTX
    nvtxInitialize(nullptr);

    // Load the matrices
    if (argc != 4) {
        fmt::println("Usage: {} <input:A> <input:B> <output>", argv[0]);
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

    // Warm up the GPU
    cusparse(d_a, d_b);
    utils::device_sync();
    std::this_thread::sleep_for(SLEEP_TIME);

    // Execute
    const auto d_c = cusparse(d_a, d_b);

    // Save the result
    auto h_c = d_c.to<utils::Location::Host>();
    h_c.save_to_filename(argv[3]);

    return EXIT_SUCCESS;
}
