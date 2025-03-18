#include <concepts>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <fmt/base.h>
#include <fmt/core.h>
#include <gsl/gsl-lite.hpp>

#include <utils/utils.cuh>

template<std::floating_point T>
using HostCSR = utils::CSR<T, utils::Location::Host>;
template<std::floating_point T>
using DeviceCSR = utils::CSR<T, utils::Location::Device>;

namespace {

template<std::floating_point T>
auto spgemm_cusparse(const DeviceCSR<T>& a, const DeviceCSR<T>& b) -> DeviceCSR<T> {
    // Initialize cuSPARSE
    cusparseHandle_t handle{};
    utils::handle_cusparse_error(cusparseCreate(&handle));

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

    // Initialize cuSPARSE SpGEMM descriptor
    cusparseSpGEMMDescr_t spgemm_desc{};
    utils::handle_cusparse_error(cusparseSpGEMM_createDescr(&spgemm_desc));

    // Set parameters for the algorithm
    const double alpha = 1.0;
    const double beta = 0.0;

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
    DeviceCSR<T> d_c(gsl::narrow_cast<std::int32_t>(nnz_c),
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
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_c));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_b));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_a));
    utils::handle_cusparse_error(cusparseDestroy(handle));

    return d_c;
}

} // namespace

auto main(int argc, char** argv) -> int {
    if (argc != 4) {
        fmt::println("Usage: {} <input:A> <input:B> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    auto h_a = HostCSR<double>::load_from_filename(argv[1]);
    auto h_b = HostCSR<double>::load_from_filename(argv[2]);
    if (h_a.n != h_b.m) {
        fmt::println("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }

    auto d_a = h_a.to<utils::Location::Device>();
    auto d_b = h_b.to<utils::Location::Device>();

    const auto nip = get_nip(d_a, d_b);
    fmt::print("Number of intermediate products: {}\n", nip);

    auto d_c = spgemm_cusparse(d_a, d_b);
    auto h_c = d_c.to<utils::Location::Host>();

    h_c.save_to_filename(argv[3]);

    return EXIT_SUCCESS;
}
