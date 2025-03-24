#pragma once

#include <array>
#include <cstdint>

inline constexpr std::int32_t N_BINS = 8;

struct Meta {
    std::array<cudaStream_t, N_BINS> streams{};

    // Host memory
    // Combined in h_bin_sizes
    std::int32_t* h_bin_sizes{};   // size N_BINS
    std::int32_t* h_bin_offsets{}; // size N_BINS
    std::int32_t* h_max_row_nnz{}; // size 1
    std::int32_t* h_total_nnz{};   // size 1

    // Device memory
    // Combined in d_bins
    std::int32_t* d_bins{};        // size M
    std::int32_t* d_bin_sizes{};   // size N_BINS
    std::int32_t* d_bin_offsets{}; // size N_BINS
    std::int32_t* d_max_row_nnz{}; // size 1
    std::int32_t* d_total_nnz{};   // size 1
    void* d_cub_storage{};         // size cub_storage_size
    std::size_t cub_storage_size{};
};
