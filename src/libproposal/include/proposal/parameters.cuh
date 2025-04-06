#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda/std/array>

#include <gsl/gsl-lite.hpp>

namespace proposal {

inline constexpr std::int32_t N_CUDA_EVENTS = 1;
inline constexpr std::int32_t WARP_SIZE = 32;
inline constexpr std::int32_t HASH_EMPTY = -1;
inline constexpr std::int32_t HASH_SCALE = 107;
inline constexpr double SYM1_RANGE_RATIO = 1 / 1.2;
inline constexpr double SYM2_RANGE_RATIO = 1 / 1.2;

inline constexpr std::int32_t CC86 = 860;

template<std::int32_t ComputeCapability>
struct Parameters;

template<>
struct Parameters<CC86> {
    //
    // Symbolic 1
    //

    static constexpr cuda::std::array SYM1_BLOCK_SIZES =
        {512, 128, 256, 512, 1024, 1024, 1024};
    static constexpr cuda::std::array SYM1_PWARP_SIZES = {4, 0, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array SYM1_TABLE_SIZES =
        {8192, 1024, 2048, 8192, 16384, 25343, 0};
    static constexpr cuda::std::array SYM1_SMEM_SIZES =
        {32768, 4096, 8192, 32768, 65536, 101'376, 0};
    static constexpr cuda::std::array SYM1_RANGES =
        {53, 853, 1706, 6826, 13653, INT32_MAX, INT32_MAX};

    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_PWARP_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_TABLE_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_SMEM_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_RANGES.size());

    static constexpr auto SYM1_N_BINS = gsl::narrow_cast<std::int32_t>(
        SYM1_BLOCK_SIZES.size());
    static constexpr auto SYM1_GLOBAL_MEM_BIN = SYM1_N_BINS - 1;
    static constexpr auto SYM1_MAX_SMEM_BIN = SYM1_N_BINS - 2;

    //
    // Symbolic 2
    //

    static constexpr cuda::std::array SYM2_BLOCK_SIZES =
        {512, 512, 128, 256, 512, 1024, 1024, 1024};
    static constexpr cuda::std::array SYM2_PWARP_SIZES = {4, 4, 0, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array SYM2_TABLE_SIZES =
        {4096, 8192, 1024, 2048, 8192, 16384, 24576, 0};
    static constexpr cuda::std::array SYM2_SMEM_SIZES =
        {17408, 34304, 4240, 8464, 33808, 67600, 98320, 0};
    static constexpr cuda::std::array SYM2_RANGES =
        {26, 53, 853, 1706, 6826, 13653, 20480, INT32_MAX};

    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_PWARP_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_TABLE_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_SMEM_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_RANGES.size());

    static constexpr auto SYM2_N_BINS = gsl::narrow_cast<std::int32_t>(
        SYM2_BLOCK_SIZES.size());
    static constexpr auto SYM2_GLOBAL_MEM_BIN = SYM2_N_BINS - 1;
    static constexpr auto SYM2_MAX_SMEM_BIN = SYM2_N_BINS - 2;

    //
    // Numeric
    //

    static constexpr cuda::std::array NUM_BLOCK_SIZES = {512, 128, 256, 512, 1024, 1024};
    static constexpr cuda::std::array NUM_PWARP_SIZES = {8, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array NUM_ARRAY_SIZES_F32 =
        {4096, 938, 2005, 4138, 12672, 0};
    static constexpr cuda::std::array NUM_SMEM_SIZES_F32 =
        {32768, 7504, 16040, 33104, 101'376, 0};
    static constexpr cuda::std::array NUM_RANGES_F32 =
        {64, 938, 2005, 4138, 12672, INT32_MAX};
    static constexpr cuda::std::array NUM_ARRAY_SIZES_F64 =
        {2752, 625, 1336, 2759, 8448, 0};
    static constexpr cuda::std::array NUM_SMEM_SIZES_F64 =
        {33024, 7500, 16032, 33108, 101'376, 0};
    static constexpr cuda::std::array NUM_RANGES_F64 =
        {43, 624, 1336, 2758, 8448, INT32_MAX};

    static_assert(NUM_BLOCK_SIZES.size() == NUM_PWARP_SIZES.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_ARRAY_SIZES_F32.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_RANGES_F32.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_ARRAY_SIZES_F64.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_RANGES_F64.size());

    static constexpr auto NUM_N_BINS = gsl::narrow_cast<std::int32_t>(
        NUM_BLOCK_SIZES.size());
    static constexpr auto NUM_GLOBAL_MEM_BIN = NUM_N_BINS - 1;

    //
    // General
    //

    static constexpr std::int32_t MAX_N_BINS = std::max(
        {SYM1_N_BINS, SYM2_N_BINS, NUM_N_BINS});

    static constexpr std::int32_t MAX_THREADS_PER_SM = 1536;
    static constexpr std::int32_t OPTIMAL_BLOCK_SIZE = 512;
};

enum class BinningType : std::int8_t {
    SYM1,
    SYM2,
    NUM_F32,
    NUM_F64
};

template<typename Params, BinningType BinType>
__host__ consteval auto get_ranges() {
    if constexpr (BinType == BinningType::SYM1) {
        return Params::SYM1_RANGES;
    } else if constexpr (BinType == BinningType::SYM2) {
        return Params::SYM2_RANGES;
    } else if constexpr (BinType == BinningType::NUM_F32) {
        return Params::NUM_RANGES_F32;
    } else if constexpr (BinType == BinningType::NUM_F64) {
        return Params::NUM_RANGES_F64;
    }
}

template<BinningType BinType>
__device__ consteval auto get_ranges() {
#ifdef __CUDA_ARCH__
    if constexpr (BinType == BinningType::SYM1) {
        return Parameters<__CUDA_ARCH__>::SYM1_RANGES;
    } else if constexpr (BinType == BinningType::SYM2) {
        return Parameters<__CUDA_ARCH__>::SYM2_RANGES;
    } else if constexpr (BinType == BinningType::NUM_F32) {
        return Parameters<__CUDA_ARCH__>::NUM_RANGES_F32;
    } else if constexpr (BinType == BinningType::NUM_F64) {
        return Parameters<__CUDA_ARCH__>::NUM_RANGES_F64;
    }
#endif
}

} // namespace proposal
