#include <cusparse/cusparse.cuh>
#include <utils/utils.cuh>

#include "common.hpp"

auto main(int argc, char** argv) -> int {
    using Clock = std::chrono::steady_clock;
#ifdef USE_DOUBLE_PRECISION
    using ValueType = double;
#else
    using ValueType = float;
#endif

    // Function to convert from host CSR format to device CSR format
    auto convert_to_app_device_csr =
        [](const utils::HostCSR<ValueType>& h_a) -> utils::DeviceCSR<ValueType> {
        return h_a.template to<utils::Location::Device>();
    };

    // Function that performs sparse matrix multiplication using cuSPARSE 1
    // Returns the result matrix back on host
    auto run = [](const utils::DeviceCSR<ValueType>& d_a,
                  const utils::DeviceCSR<ValueType>& d_b) -> utils::HostCSR<ValueType> {
        return cusparse1(d_a, d_b).template to<utils::Location::Host>();
    };

    // Function to measure the execution time of the cuSPARSE 1 operation
    auto measure = [](const utils::DeviceCSR<ValueType>& d_a,
                      const utils::DeviceCSR<ValueType>& d_b) -> Clock::duration {
        const auto start = Clock::now();
        auto c = cusparse1(d_a, d_b);
        utils::device_sync();
        const auto end = Clock::now();
        return end - start;
    };

    // Run the application with the cuSPARSE 1 implementation
    return run_app<ValueType>("cuSPARSE 1",
                              convert_to_app_device_csr,
                              run,
                              measure,
                              argc,
                              argv);
}
