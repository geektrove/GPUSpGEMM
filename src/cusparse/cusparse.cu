#include <chrono>
#include <concepts>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <fmt/base.h>
#include <fmt/core.h>
#include <gsl/gsl-lite.hpp>

#include <utils/utils.cuh>

namespace {

template<std::floating_point T>
auto spgemm_cusparse(const utils::DeviceCSR<T>& a, const utils::DeviceCSR<T>& b)
    -> utils::DeviceCSR<T> {
    // Create cuSPARSE matrix descriptors
    cusparseSpMatDescr_t desc_a{};
    utils::handle_cusparse_error(cusparseCreateCsr(&desc_a,
                                                   a.m,
                                                   a.n,
                                                   a.nnz,
                                                   a.rows_ptr,
                                                   a.cols,
                                                   a.values,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_BASE_ZERO,
                                                   CUDA_R_64F));
    cusparseSpMatDescr_t desc_b{};
    utils::handle_cusparse_error(cusparseCreateCsr(&desc_b,
                                                   b.m,
                                                   b.n,
                                                   b.nnz,
                                                   b.rows_ptr,
                                                   b.cols,
                                                   b.values,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_BASE_ZERO,
                                                   CUDA_R_64F));
    cusparseSpMatDescr_t desc_c{};
    utils::handle_cusparse_error(cusparseCreateCsr(&desc_c,
                                                   a.m,
                                                   b.n,
                                                   0,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_BASE_ZERO,
                                                   CUDA_R_64F));

    // Set parameters for the algorithm
    const double alpha = 1.0;
    const double beta = 0.0;

    // Initialize cuSPARSE
    cusparseHandle_t handle{};
    utils::handle_cusparse_error(cusparseCreate(&handle));

    // Initialize cuSPARSE SpGEMM descriptor
    cusparseSpGEMMDescr_t spgemm_desc{};
    utils::handle_cusparse_error(cusparseSpGEMM_createDescr(&spgemm_desc));

    // First stage
    size_t buffer1_size{};
    utils::handle_cusparse_error(
        cusparseSpGEMM_workEstimation(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      desc_a,
                                      desc_b,
                                      &beta,
                                      desc_c,
                                      CUDA_R_64F,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemm_desc,
                                      &buffer1_size,
                                      nullptr));
    void* buffer1{};
    utils::handle_cuda_error(cudaMalloc(&buffer1, buffer1_size));
    utils::handle_cusparse_error(
        cusparseSpGEMM_workEstimation(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      desc_a,
                                      desc_b,
                                      &beta,
                                      desc_c,
                                      CUDA_R_64F,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemm_desc,
                                      &buffer1_size,
                                      buffer1));

    // Second stage
    size_t buffer2_size{};
    utils::handle_cusparse_error(cusparseSpGEMM_compute(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        desc_a,
                                                        desc_b,
                                                        &beta,
                                                        desc_c,
                                                        CUDA_R_64F,
                                                        CUSPARSE_SPGEMM_DEFAULT,
                                                        spgemm_desc,
                                                        &buffer2_size,
                                                        nullptr));
    void* buffer2{};
    utils::handle_cuda_error(cudaMalloc(&buffer2, buffer2_size));
    utils::handle_cusparse_error(cusparseSpGEMM_compute(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        desc_a,
                                                        desc_b,
                                                        &beta,
                                                        desc_c,
                                                        CUDA_R_64F,
                                                        CUSPARSE_SPGEMM_DEFAULT,
                                                        spgemm_desc,
                                                        &buffer2_size,
                                                        buffer2));

    // Extract the result matrix size
    int64_t m_c{};
    int64_t n_c{};
    int64_t nnz_c{};
    utils::handle_cusparse_error(cusparseSpMatGetSize(desc_c, &m_c, &n_c, &nnz_c));

    // Allocate the result matrix
    utils::DeviceCSR<T> d_c(gsl::narrow_cast<std::int32_t>(nnz_c),
                            gsl::narrow_cast<std::int32_t>(m_c),
                            gsl::narrow_cast<std::int32_t>(n_c));

    // Extract the result matrix
    utils::handle_cusparse_error(
        cusparseCsrSetPointers(desc_c, d_c.rows_ptr, d_c.cols, d_c.values));
    utils::handle_cusparse_error(cusparseSpGEMM_copy(handle,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &alpha,
                                                     desc_a,
                                                     desc_b,
                                                     &beta,
                                                     desc_c,
                                                     CUDA_R_64F,
                                                     CUSPARSE_SPGEMM_DEFAULT,
                                                     spgemm_desc));

    // Clean up
    utils::handle_cuda_error(cudaFree(buffer2));
    utils::handle_cuda_error(cudaFree(buffer1));
    utils::handle_cusparse_error(cusparseSpGEMM_destroyDescr(spgemm_desc));
    utils::handle_cusparse_error(cusparseDestroy(handle));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_c));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_b));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_a));

    return d_c;
}

} // namespace

auto main(int argc, char** argv) -> int {
    using clock = std::chrono::high_resolution_clock;
    constexpr auto NANO = 1'000'000'000.0;
    constexpr auto GIGA = 1'000'000'000.0;

    // Load the matrices
    if (argc != 4) {
        fmt::println("Usage: {} <input:A> <input:B> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    const auto h_a = utils::HostCSR<double>::load_from_filename(argv[1]);
    const auto h_b = utils::HostCSR<double>::load_from_filename(argv[2]);
    if (h_a.n != h_b.m) {
        fmt::println("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }
    const auto d_a = h_a.to<utils::Location::Device>();
    const auto d_b = h_b.to<utils::Location::Device>();

    // Compute NIP
    const auto nip = utils::get_nip(d_a, d_b);
    fmt::println("NIP: {}", nip);
    const auto flop = 2 * nip;
    fmt::println("FLOP: {}", flop);

    // Warm up the GPU
    utils::cudaruntime_warmup();
    constexpr int N_WARMUP = 5;
    for (int i = 0; i < N_WARMUP; i++)
        auto d_c = spgemm_cusparse(d_a, d_b);

    // Benchmark
    constexpr int N_ITERS = 10;
    std::intmax_t total_time_ns = 0;
    for (int i = 0; i < N_ITERS; i++) {
        const auto start = clock::now();
        const auto d_c = spgemm_cusparse(d_a, d_b);
        const auto end = clock::now();

        const auto elapsed = end - start;
        const auto elapsed_ns = std::chrono::nanoseconds(elapsed).count();
        total_time_ns += elapsed_ns;
    }
    const auto average_time_ns = gsl::narrow_cast<double>(total_time_ns) / N_ITERS;
    const auto average_time_s = average_time_ns / NANO;

    fmt::println("Average time over {} iterations: {} ns", N_ITERS, average_time_ns);
    fmt::println("Average performance: {:.6f} GFLOPS",
                 gsl::narrow_cast<double>(flop) / (GIGA * average_time_s));

    // Save the result
    auto d_c = spgemm_cusparse(d_a, d_b);
    auto h_c = d_c.to<utils::Location::Host>();
    h_c.save_to_filename(argv[3]);

    return EXIT_SUCCESS;
}
