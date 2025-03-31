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

constexpr auto modpow2(std::integral auto a, std::integral auto b) {
    assert(b > 0);
    if (b <= 0)
        __builtin_unreachable();
    return a & (b - 1);
}

constexpr auto divpow2(std::integral auto a, std::integral auto b) {
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
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    const auto result = cuda::std::bit_width(unsigned_a) - 1;
    return gsl::narrow_cast<T>(result);
}

template<std::integral T>
constexpr auto bitceil(T a) {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    const auto result = cuda::std::bit_ceil(unsigned_a);
    return gsl::narrow_cast<T>(result);
}

template<std::integral T>
constexpr auto bitfloor(T a) {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    const auto result = cuda::std::bit_floor(unsigned_a);
    return gsl::narrow_cast<T>(result);
}

} // namespace utils
