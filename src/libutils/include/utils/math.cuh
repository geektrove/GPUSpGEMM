#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>

#include <cuda/cmath>
#include <cuda/std/bit>
#include <gsl/gsl-lite.hpp>

namespace utils {

constexpr auto ispow2(std::integral auto a) -> bool {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    return (a & (a - 1)) == 0;
}

constexpr auto fastmodpow2(std::integral auto a, std::integral auto b) {
    assert(b > 0);
    if (b <= 0)
        __builtin_unreachable();
    return a & (b - 1);
}

constexpr auto fastdivpow2(std::integral auto a, std::integral auto b) {
    assert(b > 0);
    if (b <= 0)
        __builtin_unreachable();
    return a >> ilog2(b);
}

template<std::integral T>
constexpr auto ilog2(T a) {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    if constexpr (std::is_signed_v<T>)
        return cuda::std::bit_width(gsl::narrow_cast<std::make_unsigned_t<T>>(a)) - 1;
    else
        return cuda::std::bit_width(a) - 1;
}

} // namespace utils
