#include <cusparse/cusparse.cuh>
#include <utils/utils.cuh>

#include "common.hpp"

auto main(int argc, char** argv) -> int {
    using Clock = std::chrono::steady_clock;
    using ValueType = double;

    auto convert_to_app_device_csr =
        [](const utils::HostCSR<ValueType>& h_a) -> utils::DeviceCSR<ValueType> {
        return h_a.template to<utils::Location::Device>();
    };

    auto run = [](const utils::DeviceCSR<ValueType>& d_a,
                  const utils::DeviceCSR<ValueType>& d_b) -> utils::HostCSR<ValueType> {
        return cusparse2(d_a, d_b).template to<utils::Location::Host>();
    };

    auto measure = [](const utils::DeviceCSR<ValueType>& d_a,
                      const utils::DeviceCSR<ValueType>& d_b) -> Clock::duration {
        const auto start = Clock::now();
        auto c = cusparse2(d_a, d_b);
        utils::device_sync();
        const auto end = Clock::now();
        return end - start;
    };

    return run_app<ValueType>("cuSPARSE 2",
                              convert_to_app_device_csr,
                              run,
                              measure,
                              argc,
                              argv);
}
