#pragma once

#include <cstdint>

namespace utils {

// Indicates whether data resides in host (CPU) or device (GPU) memory
enum class Location : std::uint8_t {
    Host,
    Device,
};

} // namespace utils
