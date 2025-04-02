#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda/std/array>

inline constexpr std::int32_t N_CUDA_EVENTS = 1;
inline constexpr std::int32_t WARP_SIZE = 32;
inline constexpr std::int32_t HASH_EMPTY = -1;
inline constexpr std::int32_t HASH_SCALE = 107;
inline constexpr double SYM_RANGE_RATIO = 1 / 1.2;

inline constexpr std::int32_t CC86 = 860;

template<std::int32_t ComputeCapability>
struct Parameters;

template<>
struct Parameters<CC86> {
    static constexpr std::int32_t MAX_THREADS_PER_SM = 1536;

    // Optimal block size for 100% occupancy
    static constexpr std::int32_t OPTIMAL_BLOCK_SIZE = 512;

    //
    // Symbolic 1
    //

    static constexpr std::int32_t SYM1_N_BINS = 7;

    static constexpr std::int32_t SYM1_PWARP_BIN = 0;
    static constexpr std::int32_t SYM1_PWARP_SIZE = 4;
    static constexpr std::int32_t SYM1_SMEM_BIN_BEGIN = 4;
    static constexpr std::int32_t SYM1_MAX_SMEM_BIN = 5;
    static constexpr std::int32_t SYM1_GLOBAL_MEM_BIN = 6;

    using SYM1_CUDA_ARRAY = cuda::std::array<std::int32_t, SYM1_N_BINS>;
    static constexpr SYM1_CUDA_ARRAY SYM1_BLOCK_SIZES =
        {512, 128, 256, 512, 1024, 1024, 1024};
    static constexpr SYM1_CUDA_ARRAY SYM1_TABLE_SIZES =
        {8192, 1024, 2048, 8192, 16384, 25343, INT32_MAX};
    static constexpr SYM1_CUDA_ARRAY SYM1_RANGES =
        {53, 853, 1706, 6826, 13653, INT32_MAX, INT32_MAX};

    //
    // Symbolic 2
    //

    static constexpr std::int32_t SYM2_N_BINS = 7;

    static constexpr std::int32_t SYM2_PWARP_BIN = 0;
    static constexpr std::int32_t SYM2_PWARP_SIZE = 4;
    static constexpr std::int32_t SYM2_SMEM_BIN_BEGIN = 4;
    static constexpr std::int32_t SYM2_MAX_SMEM_BIN = 5;
    static constexpr std::int32_t SYM2_GLOBAL_MEM_BIN = 6;

    using SYM2_CUDA_ARRAY = cuda::std::array<std::int32_t, SYM2_N_BINS>;
    static constexpr SYM2_CUDA_ARRAY SYM2_BLOCK_SIZES =
        {512, 128, 256, 512, 1024, 1024, 1024};
    static constexpr SYM2_CUDA_ARRAY SYM2_TABLE_SIZES =
        {4096, 1024, 2048, 8192, 16384, 25343, INT32_MAX};
    static constexpr SYM2_CUDA_ARRAY SYM2_RANGES =
        {26, 853, 1706, 6826, 13653, 21119, INT32_MAX};

    //
    // Numeric
    //

    static constexpr std::int32_t NUM_N_BINS = 6;

    static constexpr std::int32_t NUM_PWARP_BIN = 0;
    static constexpr std::int32_t NUM_PWARP_SIZE = 8;
    static constexpr std::int32_t NUM_SMEM_BIN_BEGIN = 4;
    static constexpr std::int32_t NUM_GLOBAL_MEM_BIN = 5;

    using NUM_CUDA_ARRAY = cuda::std::array<std::int32_t, NUM_N_BINS>;
    static constexpr NUM_CUDA_ARRAY NUM_BLOCK_SIZES = {512, 128, 256, 512, 1024, 1024};
    static constexpr NUM_CUDA_ARRAY NUM_ARRAY_SIZES =
        {2752, 624, 1336, 2758, 8448, INT32_MAX};
    static constexpr NUM_CUDA_ARRAY NUM_RANGES = {43, 624, 1336, 2758, 8448, INT32_MAX};

    //
    // General
    //

    static constexpr std::int32_t MAX_N_BINS = std::max(
        {SYM1_N_BINS, SYM2_N_BINS, NUM_N_BINS});
};

enum class BinningType : std::int8_t {
    SYM1,
    SYM2,
    NUM
};

template<typename Params, BinningType BinType>
__host__ consteval auto get_ranges() {
    if constexpr (BinType == BinningType::SYM1) {
        return Params::SYM1_RANGES;
    } else if constexpr (BinType == BinningType::SYM2) {
        return Params::SYM2_RANGES;
    } else if constexpr (BinType == BinningType::NUM) {
        return Params::NUM_RANGES;
    }
}

template<BinningType BinType>
__device__ consteval auto get_ranges() {
#ifdef __CUDA_ARCH__
    if constexpr (BinType == BinningType::SYM1) {
        return Parameters<__CUDA_ARCH__>::SYM1_RANGES;
    } else if constexpr (BinType == BinningType::SYM2) {
        return Parameters<__CUDA_ARCH__>::SYM2_RANGES;
    } else if constexpr (BinType == BinningType::NUM) {
        return Parameters<__CUDA_ARCH__>::NUM_RANGES;
    }
#endif
}

__device__ consteval auto get_minctapersm(const std::int32_t block_size) -> std::int32_t {
#ifdef __CUDA_ARCH__
    return Parameters<__CUDA_ARCH__>::MAX_THREADS_PER_SM / block_size;
#else
    return 1;
#endif
}
