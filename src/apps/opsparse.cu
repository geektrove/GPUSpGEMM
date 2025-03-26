#include <cstdlib>
#include <thread>

#include <fmt/base.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/cfg/env.h>

#include <opsparse/conversion.cuh>
#include <opsparse/opsparse.h>
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
    auto h_a = utils::HostCSR<ValueType>::load_from_filename(argv[1]);
    auto h_b = utils::HostCSR<ValueType>::load_from_filename(argv[2]);
    if (h_a.n != h_b.m) {
        fmt::println("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }

    // Convert to OpSparse CSR format
    CSR A = convertFromUtilsCSR(h_a);
    CSR B = convertFromUtilsCSR(h_b);
    h_a.release();
    h_b.release();
    A.H2D();
    B.H2D();

    // Warm up the GPU
    {
        CSR C;
        Meta meta;
        Timings timing;
        opsparse(A, B, C, meta, timing);
    }
    utils::handle_cuda_error(cudaDeviceSynchronize());
    std::this_thread::sleep_for(SLEEP_TIME);

    // Execute
    CSR C;
    Meta meta;
    Timings timing;
    opsparse(A, B, C, meta, timing);

    // Save the result
    C.D2H();
    const auto h_c = convertToUtilsCSR(C);
    h_c.save_to_filename(argv[3]);

    return EXIT_SUCCESS;
}
