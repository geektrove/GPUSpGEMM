#include <opsparse/conversion.cuh>
#include <opsparse/opsparse.h>
#include <utils/utils.cuh>

#include "common.hpp"

auto main(int argc, char** argv) -> int {
    using Clock = std::chrono::steady_clock;
    using ValueType = double;

    auto convert_to_app_device_csr = [](const utils::HostCSR<ValueType>& h_a) -> CSR {
        auto A = convertFromUtilsCSR(h_a);
        A.H2D();
        return A;
    };

    auto run = [](const CSR& A, const CSR& B) -> utils::HostCSR<ValueType> {
        CSR C;
        Meta meta;
        Timings timing;
        opsparse(A, B, C, meta, timing);
        C.D2H();
        return convertToUtilsCSR(C);
    };

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

    return run_app<ValueType>("OpSparse",
                              convert_to_app_device_csr,
                              run,
                              measure,
                              argc,
                              argv);
}
