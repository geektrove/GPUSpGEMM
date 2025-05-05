#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>

#include <cuda/cmath>
#include <cuda/std/bit>
#include <gsl/gsl-lite.hpp>

namespace utils {

// Returns the smallest power of 2 greater than or equal to a
template<std::integral T>
constexpr auto bitceil(T a) {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    const auto result = cuda::std::bit_ceil(unsigned_a);
    return gsl::narrow_cast<T>(result);
}

// Returns the largest power of 2 less than or equal to a
template<std::integral T>
constexpr auto bitfloor(T a) {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    const auto result = cuda::std::bit_floor(unsigned_a);
    return gsl::narrow_cast<T>(result);
}

// Checks if a is a power of 2
template<std::integral T>
constexpr auto ispow2(T a) -> bool {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    return cuda::std::has_single_bit(unsigned_a);
}

// Returns the base-2 logarithm of a (floor value)
template<std::integral T>
constexpr auto ilog2(T a) {
    assert(a > 0);
    if (a <= 0)
        __builtin_unreachable();
    const auto unsigned_a = gsl::narrow_cast<std::make_unsigned_t<T>>(a);
    const auto result = cuda::std::bit_width(unsigned_a) - 1;
    return gsl::narrow_cast<T>(result);
}

// Computes a modulo b, where b is a power of 2
constexpr auto modpow2(std::integral auto a, std::integral auto b) {
    assert(b > 0);
    assert(ispow2(b));
    if (b <= 0)
        __builtin_unreachable();
    return a & (b - 1);
}

// Computes a divided by b, where b is a power of 2
constexpr auto divpow2(std::integral auto a, std::integral auto b) {
    assert(b > 0);
    assert(ispow2(b));
    if (b <= 0)
        __builtin_unreachable();
    return a >> ilog2(b);
}

} // namespace utils
