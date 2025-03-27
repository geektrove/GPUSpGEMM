#pragma once

#include <concepts>
#include <cuda/std/bit>

namespace utils {

constexpr auto ispow2(std::integral auto a) -> bool {
    return (a & (a - 1)) == 0;
}

constexpr auto fastmodpow2(std::integral auto a, std::integral auto b) {
    return a & (b - 1);
}

constexpr auto fastdivpow2(std::integral auto a, std::unsigned_integral auto b) {
    return a >> cuda::std::bit_width(b);
}

} // namespace utils
