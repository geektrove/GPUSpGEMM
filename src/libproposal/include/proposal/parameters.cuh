#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda/std/array>
#include <functional>

#include <gsl/gsl-lite.hpp>

namespace proposal {

inline constexpr std::int32_t WARP_SIZE = 32;
inline constexpr std::int32_t HASH_EMPTY = -1;
inline constexpr std::int32_t HASH_SCALE = 107;
inline constexpr double SYM1_RANGE_RATIO = 1 / 1.2;
inline constexpr double SYM2_RANGE_RATIO = 1 / 1.2;

inline constexpr std::int32_t CC80 = 800;
inline constexpr std::int32_t CC86 = 860;

template<std::int32_t ComputeCapability>
struct Parameters;

template<>
struct Parameters<CC80> {
    //
    // Symbolic 1
    //

    static constexpr cuda::std::array SYM1_BLOCK_SIZES =
        {1024, 128, 256, 512, 1024, 1024, 1024, 1024, 1024};
    static constexpr cuda::std::array SYM1_PWARP_SIZES = {4, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array SYM1_TABLE_SIZES =
        {16384, 1024, 2048, 4096, 8192, 16384, 32768, 41727, 0};
    static constexpr cuda::std::array SYM1_SMEM_SIZES =
        {65536, 4096, 8192, 16384, 32768, 65536, 131'072, 166'912, 0};

    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_PWARP_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_TABLE_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_SMEM_SIZES.size());

    static constexpr auto SYM1_N_BINS = gsl::narrow_cast<std::int32_t>(
        SYM1_BLOCK_SIZES.size());
    static constexpr auto SYM1_GLOBAL_MEM_BIN = SYM1_N_BINS - 1;
    static constexpr auto SYM1_MAX_SMEM_BIN = SYM1_N_BINS - 2;

    static constexpr cuda::std::array SYM1_RANGES = std::invoke([] {
        cuda::std::array<int, SYM1_N_BINS> ranges{};
        for (auto i = 0; i < SYM1_MAX_SMEM_BIN; i++) {
            ranges[i] = SYM1_TABLE_SIZES[i];
            if (SYM1_PWARP_SIZES[i] > 0)
                ranges[i] /= (SYM1_BLOCK_SIZES[i] / SYM1_PWARP_SIZES[i]);
            ranges[i] = gsl::narrow_cast<int>(ranges[i] * SYM1_RANGE_RATIO);
        }
        ranges[SYM1_MAX_SMEM_BIN] = INT32_MAX;
        ranges[SYM1_GLOBAL_MEM_BIN] = INT32_MAX;
        return ranges;
    });

    //
    // Symbolic 2
    //

    static constexpr cuda::std::array SYM2_BLOCK_SIZES =
        {1024, 1024, 128, 256, 512, 1024, 1024, 1024, 1024, 1024};
    static constexpr cuda::std::array SYM2_PWARP_SIZES = {4, 4, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array SYM2_TABLE_SIZES =
        {8192, 16384, 1024, 2048, 4096, 8192, 16384, 32768, 40960, 0};
    static constexpr cuda::std::array SYM2_SMEM_SIZES =
        {34816, 68608, 4240, 8464, 16912, 33808, 67600, 135'184, 163'856, 0};

    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_PWARP_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_TABLE_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_SMEM_SIZES.size());

    static constexpr auto SYM2_N_BINS = gsl::narrow_cast<std::int32_t>(
        SYM2_BLOCK_SIZES.size());
    static constexpr auto SYM2_GLOBAL_MEM_BIN = SYM2_N_BINS - 1;

    static constexpr cuda::std::array SYM2_RANGES = std::invoke([] {
        cuda::std::array<int, SYM2_N_BINS> ranges{};
        for (auto i = 0; i < SYM2_GLOBAL_MEM_BIN; i++) {
            ranges[i] = SYM2_TABLE_SIZES[i];
            if (SYM2_PWARP_SIZES[i] > 0)
                ranges[i] /= (SYM2_BLOCK_SIZES[i] / SYM2_PWARP_SIZES[i]);
            ranges[i] = gsl::narrow_cast<int>(ranges[i] * SYM2_RANGE_RATIO);
        }
        ranges[SYM2_GLOBAL_MEM_BIN] = INT32_MAX;
        return ranges;
    });

    //
    // Numeric
    //

    static constexpr cuda::std::array NUM_BLOCK_SIZES =
        {1024, 128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024};
    static constexpr cuda::std::array NUM_PWARP_SIZES = {8, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array NUM_ARRAY_SIZES_F32 =
        {10240, 528, 1184, 2496, 5120, 10368, 20864, 20736, 41728, 0};
    static constexpr cuda::std::array NUM_SMEM_SIZES_F32 =
        {81920, 4224, 9472, 19968, 40960, 82944, 166'912, 82944, 166'912, 0};
    static constexpr cuda::std::array NUM_ARRAY_SIZES_F64 =
        {6912, 352, 788, 1664, 3412, 6912, 13908, 20736, 41728, 0};
    static constexpr cuda::std::array NUM_SMEM_SIZES_F64 =
        {82944, 4224, 9456, 19968, 40944, 82944, 166'896, 82944, 166'912, 0};

    static_assert(NUM_BLOCK_SIZES.size() == NUM_PWARP_SIZES.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_ARRAY_SIZES_F32.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_SMEM_SIZES_F32.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_ARRAY_SIZES_F64.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_SMEM_SIZES_F64.size());

    static constexpr auto NUM_N_BINS = gsl::narrow_cast<std::int32_t>(
        NUM_BLOCK_SIZES.size());
    static constexpr auto NUM_GLOBAL_MEM_BIN = NUM_N_BINS - 1;
    static constexpr auto NUM_SMEM_COLS_BIN_BEGIN = NUM_N_BINS - 2;
    static constexpr auto NUM_SMEM_REGULAR_BIN_BEGIN = NUM_N_BINS - 4;

    static constexpr cuda::std::array NUM_RANGES_F32 = std::invoke([] {
        cuda::std::array ranges = NUM_ARRAY_SIZES_F32;
        ranges[NUM_GLOBAL_MEM_BIN] = INT32_MAX;
        for (auto i = NUM_SMEM_REGULAR_BIN_BEGIN; i >= 0; i--) {
            if (NUM_PWARP_SIZES[i] > 0)
                ranges[i] /= (NUM_BLOCK_SIZES[i] / NUM_PWARP_SIZES[i]);
        }
        return ranges;
    });
    static constexpr cuda::std::array NUM_RANGES_F64 = std::invoke([] {
        cuda::std::array ranges = NUM_ARRAY_SIZES_F64;
        ranges[NUM_GLOBAL_MEM_BIN] = INT32_MAX;
        for (auto i = NUM_SMEM_REGULAR_BIN_BEGIN; i >= 0; i--) {
            if (NUM_PWARP_SIZES[i] > 0)
                ranges[i] /= (NUM_BLOCK_SIZES[i] / NUM_PWARP_SIZES[i]);
        }
        return ranges;
    });

    //
    // General
    //

    static constexpr std::int32_t MAX_N_BINS = std::max(
        {SYM1_N_BINS, SYM2_N_BINS, NUM_N_BINS});
    static constexpr std::int32_t OPTIMAL_BLOCK_SIZE = 1024;
};

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

    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_PWARP_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_TABLE_SIZES.size());
    static_assert(SYM1_BLOCK_SIZES.size() == SYM1_SMEM_SIZES.size());

    static constexpr auto SYM1_N_BINS = gsl::narrow_cast<std::int32_t>(
        SYM1_BLOCK_SIZES.size());
    static constexpr auto SYM1_GLOBAL_MEM_BIN = SYM1_N_BINS - 1;
    static constexpr auto SYM1_MAX_SMEM_BIN = SYM1_N_BINS - 2;

    static constexpr cuda::std::array SYM1_RANGES = std::invoke([] {
        cuda::std::array<int, SYM1_N_BINS> ranges{};
        for (auto i = 0; i < SYM1_MAX_SMEM_BIN; i++) {
            ranges[i] = SYM1_TABLE_SIZES[i];
            if (SYM1_PWARP_SIZES[i] > 0)
                ranges[i] /= (SYM1_BLOCK_SIZES[i] / SYM1_PWARP_SIZES[i]);
            ranges[i] = gsl::narrow_cast<int>(ranges[i] * SYM1_RANGE_RATIO);
        }
        ranges[SYM1_MAX_SMEM_BIN] = INT32_MAX;
        ranges[SYM1_GLOBAL_MEM_BIN] = INT32_MAX;
        return ranges;
    });

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

    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_PWARP_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_TABLE_SIZES.size());
    static_assert(SYM2_BLOCK_SIZES.size() == SYM2_SMEM_SIZES.size());

    static constexpr auto SYM2_N_BINS = gsl::narrow_cast<std::int32_t>(
        SYM2_BLOCK_SIZES.size());
    static constexpr auto SYM2_GLOBAL_MEM_BIN = SYM2_N_BINS - 1;

    static constexpr cuda::std::array SYM2_RANGES = std::invoke([] {
        cuda::std::array<int, SYM2_N_BINS> ranges{};
        for (auto i = 0; i < SYM2_GLOBAL_MEM_BIN; i++) {
            ranges[i] = SYM2_TABLE_SIZES[i];
            if (SYM2_PWARP_SIZES[i] > 0)
                ranges[i] /= (SYM2_BLOCK_SIZES[i] / SYM2_PWARP_SIZES[i]);
            ranges[i] = gsl::narrow_cast<int>(ranges[i] * SYM2_RANGE_RATIO);
        }
        ranges[SYM2_GLOBAL_MEM_BIN] = INT32_MAX;
        return ranges;
    });

    //
    // Numeric
    //

    static constexpr cuda::std::array NUM_BLOCK_SIZES =
        {512, 128, 256, 512, 1024, 768, 1024, 1024};
    static constexpr cuda::std::array NUM_PWARP_SIZES = {8, 0, 0, 0, 0, 0, 0, 0};
    static constexpr cuda::std::array NUM_ARRAY_SIZES_F32 =
        {4096, 938, 2004, 4138, 12672, 12544, 25344, 0};
    static constexpr cuda::std::array NUM_SMEM_SIZES_F32 =
        {32768, 7504, 16032, 33104, 101'376, 50176, 101'376, 0};
    static constexpr cuda::std::array NUM_ARRAY_SIZES_F64 =
        {2688, 624, 1336, 2758, 8448, 12544, 25344, 0};
    static constexpr cuda::std::array NUM_SMEM_SIZES_F64 =
        {32256, 7488, 16032, 33096, 101'376, 50176, 101'376, 0};

    static_assert(NUM_BLOCK_SIZES.size() == NUM_PWARP_SIZES.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_ARRAY_SIZES_F32.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_SMEM_SIZES_F32.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_ARRAY_SIZES_F64.size());
    static_assert(NUM_BLOCK_SIZES.size() == NUM_SMEM_SIZES_F64.size());

    static constexpr auto NUM_N_BINS = gsl::narrow_cast<std::int32_t>(
        NUM_BLOCK_SIZES.size());
    static constexpr auto NUM_GLOBAL_MEM_BIN = NUM_N_BINS - 1;
    static constexpr auto NUM_SMEM_COLS_BIN_BEGIN = NUM_N_BINS - 2;
    static constexpr auto NUM_SMEM_REGULAR_BIN_BEGIN = NUM_N_BINS - 4;

    static constexpr cuda::std::array NUM_RANGES_F32 = std::invoke([] {
        cuda::std::array ranges = NUM_ARRAY_SIZES_F32;
        ranges[NUM_GLOBAL_MEM_BIN] = INT32_MAX;
        for (auto i = NUM_SMEM_REGULAR_BIN_BEGIN; i >= 0; i--) {
            if (NUM_PWARP_SIZES[i] > 0)
                ranges[i] /= (NUM_BLOCK_SIZES[i] / NUM_PWARP_SIZES[i]);
        }
        return ranges;
    });
    static constexpr cuda::std::array NUM_RANGES_F64 = std::invoke([] {
        cuda::std::array ranges = NUM_ARRAY_SIZES_F64;
        ranges[NUM_GLOBAL_MEM_BIN] = INT32_MAX;
        for (auto i = NUM_SMEM_REGULAR_BIN_BEGIN; i >= 0; i--) {
            if (NUM_PWARP_SIZES[i] > 0)
                ranges[i] /= (NUM_BLOCK_SIZES[i] / NUM_PWARP_SIZES[i]);
        }
        return ranges;
    });

    //
    // General
    //

    static constexpr std::int32_t MAX_N_BINS = std::max(
        {SYM1_N_BINS, SYM2_N_BINS, NUM_N_BINS});
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
