#include <concepts>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <fmt/core.h>
#include <gsl/gsl-lite.hpp>

#include <utils/utils.cuh>

using namespace utils;

#define CHECK_CUSPARSE(value) check_cusparse_error((value), #value, __FILE__, __LINE__)

auto check_cusparse_error(const cusparseStatus_t status,
                          const char* const function,
                          const char* const file,
                          const int line) -> void {
    if (status == CUSPARSE_STATUS_SUCCESS)
        return;
    const auto* reason{cusparseGetErrorString(status)};
    const auto message{
        fmt::format("CUSPARSE error ({}:{}:{}): {}", file, line, function, reason)};
    throw std::runtime_error(message);
}

template<std::floating_point T>
auto get_nips_square(const CSR<T, Location::Device>& a) -> int64_t {
    // Initialize cuSPARSE
    cusparseHandle_t handle{};
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create cuSPARSE matrix descriptors
    cusparseSpMatDescr_t desc_a{};
    CHECK_CUSPARSE(cusparseCreateCsr(&desc_a,
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
    cusparseSpMatDescr_t desc_c{};
    CHECK_CUSPARSE(cusparseCreateCsr(&desc_c,
                                     a.m,
                                     a.n,
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
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemm_desc));

    // Set parameters for the algorithm
    const double alpha = 1.0;
    const double beta = 0.0;

    // First stage
    size_t buffer1_size{};
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 desc_a,
                                                 desc_a,
                                                 &beta,
                                                 desc_c,
                                                 CUDA_R_64F,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemm_desc,
                                                 &buffer1_size,
                                                 nullptr));
    void* buffer1{};
    CHECK_CUDA(cudaMalloc(&buffer1, buffer1_size));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 desc_a,
                                                 desc_a,
                                                 &beta,
                                                 desc_c,
                                                 CUDA_R_64F,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemm_desc,
                                                 &buffer1_size,
                                                 buffer1));

    // Extract the number of intermediate products
    int64_t nip{};
    CHECK_CUSPARSE(cusparseSpGEMM_getNumProducts(spgemm_desc, &nip));

    // Clean up
    CHECK_CUDA(cudaFree(buffer1));
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemm_desc));
    CHECK_CUSPARSE(cusparseDestroySpMat(desc_c));
    CHECK_CUSPARSE(cusparseDestroySpMat(desc_a));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return nip;
}

template<std::floating_point T>
auto spgemm_cusparse_square(const CSR<T, Location::Device>& a)
    -> CSR<T, Location::Device> {
    // Initialize cuSPARSE
    cusparseHandle_t handle{};
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create cuSPARSE matrix descriptors
    cusparseSpMatDescr_t desc_a{};
    CHECK_CUSPARSE(cusparseCreateCsr(&desc_a,
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
    cusparseSpMatDescr_t desc_c{};
    CHECK_CUSPARSE(cusparseCreateCsr(&desc_c,
                                     a.m,
                                     a.n,
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
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemm_desc));

    // Set parameters for the algorithm
    const double alpha = 1.0;
    const double beta = 0.0;

    // First stage
    size_t buffer1_size{};
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 desc_a,
                                                 desc_a,
                                                 &beta,
                                                 desc_c,
                                                 CUDA_R_64F,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemm_desc,
                                                 &buffer1_size,
                                                 nullptr));
    void* buffer1{};
    CHECK_CUDA(cudaMalloc(&buffer1, buffer1_size));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 desc_a,
                                                 desc_a,
                                                 &beta,
                                                 desc_c,
                                                 CUDA_R_64F,
                                                 CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemm_desc,
                                                 &buffer1_size,
                                                 buffer1));

    // Second stage
    size_t buffer2_size{};
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          desc_a,
                                          desc_a,
                                          &beta,
                                          desc_c,
                                          CUDA_R_64F,
                                          CUSPARSE_SPGEMM_DEFAULT,
                                          spgemm_desc,
                                          &buffer2_size,
                                          nullptr));
    void* buffer2{};
    CHECK_CUDA(cudaMalloc(&buffer2, buffer2_size));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          desc_a,
                                          desc_a,
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
    CHECK_CUSPARSE(cusparseSpMatGetSize(desc_c, &m_c, &n_c, &nnz_c));

    // Allocate the result matrix
    CSR<double, Location::Device> d_c(gsl::narrow_cast<std::int32_t>(nnz_c),
                                      gsl::narrow_cast<std::int32_t>(m_c),
                                      gsl::narrow_cast<std::int32_t>(n_c));

    // Extract the result matrix
    CHECK_CUSPARSE(cusparseCsrSetPointers(desc_c, d_c.rows_ptr, d_c.cols, d_c.values));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha,
                                       desc_a,
                                       desc_a,
                                       &beta,
                                       desc_c,
                                       CUDA_R_64F,
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       spgemm_desc));

    // Clean up
    CHECK_CUDA(cudaFree(buffer2));
    CHECK_CUDA(cudaFree(buffer1));
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemm_desc));
    CHECK_CUSPARSE(cusparseDestroySpMat(desc_c));
    CHECK_CUSPARSE(cusparseDestroySpMat(desc_a));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return d_c;
}

auto main(int argc, char** argv) -> int {
    if (argc != 3) {
        fmt::println("Usage: {} <input> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    const auto* input = argv[1];
    const auto* output = argv[2];

    auto h_a = CSR<double, Location::Host>::load_from_filename(input);
    auto d_a = h_a.to<Location::Device>();

    const auto nip = get_nips_square(d_a);
    fmt::print("Number of intermediate products: {}\n", nip);

    auto d_c = spgemm_cusparse_square(d_a);

    auto h_c = d_c.to<Location::Host>();
    h_c.save_to_filename(output);

    h_a.free();
    d_a.free();
    d_c.free();
    h_c.free();

    return EXIT_SUCCESS;
}
