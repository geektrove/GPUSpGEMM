#pragma once

#include <type_traits>

template<typename T>
requires std::is_integral_v<T>
constexpr auto narrow(auto a) -> T {
    return static_cast<T>(a);
}
