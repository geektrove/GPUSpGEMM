#pragma once

namespace utils {

#ifdef NDEBUG
inline constexpr bool IS_DEBUG = false;
#else
inline constexpr bool IS_DEBUG = true;
#endif

} // namespace utils
