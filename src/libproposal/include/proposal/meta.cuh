#pragma once

#include <array>
#include <cstdint>

#include <proposal/parameters.cuh>

namespace proposal {

// Metadata structure for the proposal algorithm, containing streams, memory pools,
// and temporary working arrays used across different algorithm phases
template<typename Params>
struct Meta {
    std::array<cudaStream_t, Params::MAX_N_BINS> streams{};

    std::size_t cub_storage_size{}; // Size of CUB library temporary storage
    std::size_t mem_pool_size{};    // Size of algorithm's memory pool

    // Host memory
    std::int32_t h_bin_sizes[Params::MAX_N_BINS]{};   // Number of rows in each bin
    std::int32_t h_bin_offsets[Params::MAX_N_BINS]{}; // Starting offset of each bin
    std::int32_t h_max_row_nnz{};                     // Maximum non-zeros in any row

    // Device memory (combined)
    void* d_ptr{};                 // Base pointer for all allocated device memory
    std::int32_t* d_bins{};        // Bin assignments for each row
    std::int32_t* d_bin_sizes{};   // Number of rows in each bin (device copy)
    std::int32_t* d_bin_offsets{}; // Starting offset of each bin (device copy)
    std::int32_t* d_max_row_nnz{}; // Maximum non-zeros in any row (device copy)
    void* d_cub_storage{};         // Temporary storage for CUB operations

    // Device memory (separate)
    void* d_mem_pool{}; // Memory pool for temporary allocations
};

} // namespace proposal
