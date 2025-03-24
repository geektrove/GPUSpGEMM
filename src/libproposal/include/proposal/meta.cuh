#pragma once

#include <array>
#include <cstdint>

inline constexpr std::int32_t N_BINS = 8;

struct Meta {
    // Host memory
    std::array<cudaStream_t, N_BINS> streams{};
    std::size_t cub_storage_size{};

    // Device memory
    void* device_ptr{};
    std::int32_t* d_bins{};
    void* d_cub_storage{};

    // Managed memory
    void* managed_ptr{};
    std::int32_t* bin_sizes{};
    std::int32_t* bin_offsets{};
    std::int32_t* max_row_nnz{};
    std::int32_t* total_nnz{};
};
