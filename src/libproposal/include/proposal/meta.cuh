#pragma once

#include <array>
#include <cstdint>

#include <proposal/parameters.cuh>

namespace proposal {

template<typename Params>
struct Meta {
    std::array<cudaEvent_t, N_CUDA_EVENTS> events{};        // size 1
    std::array<cudaStream_t, Params::MAX_N_BINS> streams{}; // size n_bins

    std::size_t cub_storage_size{};
    std::size_t mem_pool_size{};

    // Host memory
    std::int32_t h_bin_sizes[Params::MAX_N_BINS]{};
    std::int32_t h_bin_offsets[Params::MAX_N_BINS]{};
    std::int32_t h_max_row_nnz{};

    // Device memory (combined)
    void* d_ptr{};
    std::int32_t* d_bins{};        // size M
    std::int32_t* d_bin_sizes{};   // size n_bins
    std::int32_t* d_bin_offsets{}; // size n_bins
    std::int32_t* d_max_row_nnz{}; // size 1
    void* d_cub_storage{};         // size cub_storage_size

    // Device memory (separate)
    void* d_mem_pool{};
};

} // namespace proposal
