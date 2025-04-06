#include <concepts>
#include <cstdlib>
#include <type_traits>

#include <cusparse.h>
#include <gsl/gsl-lite.hpp>
#include <spdlog/spdlog.h>

#include <utils/utils.cuh>

template<std::floating_point T>
auto cusparse(const utils::DeviceCSR<T>& a, const utils::DeviceCSR<T>& b)
    -> utils::DeviceCSR<T> {
    static constexpr auto DATA_TYPE = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
    static constexpr T ALPHA = 1.0;
    static constexpr T BETA = 0.0;

    // Create cuSPARSE matrix descriptors
    cusparseSpMatDescr_t desc_a{};
    utils::handle_cusparse_error(cusparseCreateCsr(&desc_a,
                                                   a.m,
                                                   a.n,
                                                   a.nnz,
                                                   a.rpt,
                                                   a.col,
                                                   a.val,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_BASE_ZERO,
                                                   DATA_TYPE));
    cusparseSpMatDescr_t desc_b{};
    utils::handle_cusparse_error(cusparseCreateCsr(&desc_b,
                                                   b.m,
                                                   b.n,
                                                   b.nnz,
                                                   b.rpt,
                                                   b.col,
                                                   b.val,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_32I,
                                                   CUSPARSE_INDEX_BASE_ZERO,
                                                   DATA_TYPE));
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
                                                   DATA_TYPE));

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
                                      &ALPHA,
                                      desc_a,
                                      desc_b,
                                      &BETA,
                                      desc_c,
                                      DATA_TYPE,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemm_desc,
                                      &buffer1_size,
                                      nullptr));
    SPDLOG_DEBUG("Buffer 1 size is {}", buffer1_size);
    void* buffer1 = utils::malloc<utils::Location::Device>(buffer1_size);
    utils::handle_cusparse_error(
        cusparseSpGEMM_workEstimation(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &ALPHA,
                                      desc_a,
                                      desc_b,
                                      &BETA,
                                      desc_c,
                                      DATA_TYPE,
                                      CUSPARSE_SPGEMM_DEFAULT,
                                      spgemm_desc,
                                      &buffer1_size,
                                      buffer1));

    // Second stage
    size_t buffer2_size{};
    utils::handle_cusparse_error(cusparseSpGEMM_compute(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &ALPHA,
                                                        desc_a,
                                                        desc_b,
                                                        &BETA,
                                                        desc_c,
                                                        DATA_TYPE,
                                                        CUSPARSE_SPGEMM_DEFAULT,
                                                        spgemm_desc,
                                                        &buffer2_size,
                                                        nullptr));
    SPDLOG_DEBUG("Buffer 2 size is {}", buffer2_size);
    void* buffer2 = utils::malloc<utils::Location::Device>(buffer2_size);
    utils::handle_cusparse_error(cusparseSpGEMM_compute(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &ALPHA,
                                                        desc_a,
                                                        desc_b,
                                                        &BETA,
                                                        desc_c,
                                                        DATA_TYPE,
                                                        CUSPARSE_SPGEMM_DEFAULT,
                                                        spgemm_desc,
                                                        &buffer2_size,
                                                        buffer2));

    // Extract the result matrix size
    std::int64_t m_c{};
    std::int64_t n_c{};
    std::int64_t nnz_c{};
    utils::handle_cusparse_error(cusparseSpMatGetSize(desc_c, &m_c, &n_c, &nnz_c));

    // Allocate the result matrix
    utils::DeviceCSR<T> d_c(gsl::narrow_cast<std::int32_t>(nnz_c),
                            gsl::narrow_cast<std::int32_t>(m_c),
                            gsl::narrow_cast<std::int32_t>(n_c));

    // Extract the result matrix
    utils::handle_cusparse_error(
        cusparseCsrSetPointers(desc_c, d_c.rpt, d_c.col, d_c.val));
    utils::handle_cusparse_error(cusparseSpGEMM_copy(handle,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &ALPHA,
                                                     desc_a,
                                                     desc_b,
                                                     &BETA,
                                                     desc_c,
                                                     DATA_TYPE,
                                                     CUSPARSE_SPGEMM_DEFAULT,
                                                     spgemm_desc));

    // Clean up
    utils::free<utils::Location::Device>(buffer2);
    utils::free<utils::Location::Device>(buffer1);
    utils::handle_cusparse_error(cusparseSpGEMM_destroyDescr(spgemm_desc));
    utils::handle_cusparse_error(cusparseDestroy(handle));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_c));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_b));
    utils::handle_cusparse_error(cusparseDestroySpMat(desc_a));

    return d_c;
}
