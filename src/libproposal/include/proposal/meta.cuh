#pragma once

#include <array>
#include <cstdint>

inline constexpr std::int32_t PWARP_SIZE = 4;

inline constexpr double SYM_RANGE_RATIO = 1 / 1.2;

struct Meta {
    std::array<cudaEvent_t, 1> events{}; // size 1
    cudaStream_t* streams{};             // size n_bins

    std::int32_t n_bins{};
    std::size_t cub_storage_size{};
    std::size_t mem_pool_size{};

    // Host memory (combined)
    void* h_ptr{};
    std::int32_t* sym_block_sizes{};  // size n_bins
    std::int32_t* sym_table_sizes{};  // size n_bins
    std::int32_t* h_sym_bin_ranges{}; // size n_bins
    std::int32_t* h_bin_sizes{};      // size n_bins
    std::int32_t* h_bin_offsets{};    // size n_bins
    std::int32_t* h_max_row_nnz{};    // size 1
    std::int32_t* h_total_nnz{};      // size 1

    // Device memory (combined)
    void* d_ptr{};
    std::int32_t* d_bins{};           // size M
    std::int32_t* d_sym_bin_ranges{}; // size n_bins
    std::int32_t* d_bin_sizes{};      // size n_bins
    std::int32_t* d_bin_offsets{};    // size n_bins
    std::int32_t* d_max_row_nnz{};    // size 1
    std::int32_t* d_total_nnz{};      // size 1
    void* d_cub_storage{};            // size cub_storage_size

    // Device memory (separate)
    void* d_mem_pool{};
};
