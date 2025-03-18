#pragma once

#include <cusparse.h>

#include <utils/csr.cuh>
#include <utils/errors.cuh>

namespace utils {

template<std::floating_point T>
auto get_nip(const CSR<T, Location::Device>& a, const CSR<T, Location::Device>& b)
    -> int64_t {
    // Initialize cuSPARSE
    cusparseHandle_t handle{};
    handle_cusparse_error(cusparseCreate(&handle));

    // Create cuSPARSE matrix descriptors
    cusparseSpMatDescr_t desc_a{};
    handle_cusparse_error(cusparseCreateCsr(&desc_a,
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
    handle_cusparse_error(cusparseCreateCsr(&desc_b,
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
    handle_cusparse_error(cusparseCreateCsr(&desc_c,
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
    handle_cusparse_error(cusparseSpGEMM_createDescr(&spgemm_desc));

    // Set parameters for the algorithm
    const double alpha = 1.0;
    const double beta = 0.0;

    // First stage
    size_t buffer1_size{};
    handle_cusparse_error(cusparseSpGEMM_workEstimation(handle,
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
    handle_cuda_error(cudaMalloc(&buffer1, buffer1_size));
    handle_cusparse_error(cusparseSpGEMM_workEstimation(handle,
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

    // Extract the number of intermediate products
    int64_t nip{};
    handle_cusparse_error(cusparseSpGEMM_getNumProducts(spgemm_desc, &nip));

    // Clean up
    handle_cuda_error(cudaFree(buffer1));
    handle_cusparse_error(cusparseSpGEMM_destroyDescr(spgemm_desc));
    handle_cusparse_error(cusparseDestroySpMat(desc_c));
    handle_cusparse_error(cusparseDestroySpMat(desc_b));
    handle_cusparse_error(cusparseDestroySpMat(desc_a));
    handle_cusparse_error(cusparseDestroy(handle));

    return nip;
}

} // namespace utils
