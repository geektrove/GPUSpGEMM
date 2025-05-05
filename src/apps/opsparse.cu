#include <opsparse/conversion.cuh>
#include <opsparse/opsparse.h>
#include <utils/utils.cuh>

#include "common.hpp"

auto main(int argc, char** argv) -> int {
    using Clock = std::chrono::steady_clock;
#ifdef USE_DOUBLE_PRECISION
    using ValueType = double;
#else
    using ValueType = float;
#endif

    // Function to convert from the utilities CSR format to the OpSparse-specific CSR format
    // and transfer data from host to device
    auto convert_to_app_device_csr = [](const utils::HostCSR<ValueType>& h_a) -> CSR {
        auto A = convertFromUtilsCSR(h_a);
        A.H2D();
        return A;
    };

    // Function that performs sparse matrix multiplication using OpSparse
    // Returns the result matrix back in the utility CSR format on host
    auto run = [](const CSR& A, const CSR& B) -> utils::HostCSR<ValueType> {
        CSR C;
        Meta meta;
        Timings timing;
        opsparse(A, B, C, meta, timing);
        C.D2H();
        return convertToUtilsCSR(C);
    };

    // Function to measure the execution time of the OpSparse algorithm
    auto measure = [](const CSR& A, const CSR& B) -> Clock::duration {
        const auto start = Clock::now();
        CSR C;
        Meta meta;
        Timings timing;
        opsparse(A, B, C, meta, timing);
        utils::device_sync();
        const auto end = Clock::now();
        return end - start;
    };

    // Run the application with the OpSparse implementation
    return run_app<ValueType>("OpSparse",
                              convert_to_app_device_csr,
                              run,
                              measure,
                              argc,
                              argv);
}
