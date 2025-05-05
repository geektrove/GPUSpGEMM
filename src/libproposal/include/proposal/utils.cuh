#pragma once

#include <cstdint>
#include <type_traits>

namespace proposal {

// Compile-time loop unrolling utility that executes function f for compile-time constants
// from begin to end (exclusive) with step size
template<std::int32_t begin, std::int32_t end, std::int32_t step, typename F>
void constexpr_for(F&& f) {
    static_assert(step == 1 || step == -1);
    static_assert((begin <= end && step > 0) || (begin >= end && step < 0));
    if constexpr (begin != end) {
        f(std::integral_constant<std::int32_t, begin>{});
        constexpr_for<begin + step, end, step>(std::forward<F>(f));
    }
}

// Retrieves the dynamic shared memory size allocated to the current kernel
// Used primarily for debugging purposes
__forceinline__ __device__ int dynamic_smem_size() {
    int ret{0};
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(ret));
    return ret;
}

} // namespace proposal
