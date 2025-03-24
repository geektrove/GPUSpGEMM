#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/functional>
#include <gsl/gsl-lite.hpp>

#include <proposal/sym_binning.cuh>

namespace cg = cooperative_groups;

namespace {

constexpr __device__ __constant__ std::int32_t D_SYM_BIN_RANGES[N_BINS] =
    {26, 426, 853, 1706, 3413, 6826, 10240, INT_MAX};

__device__ auto find_bin(const std::int32_t x) -> std::int32_t {
    for (int i = 0; i < N_BINS; i++) {
        if (x <= D_SYM_BIN_RANGES[i])
            return i;
    }
    __builtin_unreachable();
}

} // namespace

__global__ void k_sym_binning1(const std::int32_t* __restrict__ nips,
                               std::int32_t m,
                               std::int32_t* __restrict__ bin_sizes) {
    __shared__ std::int32_t s_bin_sizes[N_BINS];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto bid = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (bid < N_BINS)
        s_bin_sizes[block.thread_rank()] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    if (row < m) {
        const auto bin_idx = find_bin(nips[row]);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (bid < N_BINS)
        atomicAdd(bin_sizes + bid, s_bin_sizes[bid]);
}

__global__ void k_sym_binning2(const std::int32_t* __restrict__ nips,
                               std::int32_t m,
                               const std::int32_t* __restrict__ bin_offsets,
                               std::int32_t* __restrict__ bin_sizes,
                               std::int32_t* __restrict__ bins) {
    __shared__ std::int32_t s_bin_sizes[N_BINS];
    __shared__ std::int32_t s_bin_offsets[N_BINS];

    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto bid = gsl::narrow_cast<std::int32_t>(block.thread_rank());

    if (bid < N_BINS)
        s_bin_sizes[block.thread_rank()] = 0;
    block.sync();

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    int bin_idx = 0;
    if (row < m) {
        bin_idx = find_bin(nips[row]);
        atomicAdd_block(s_bin_sizes + bin_idx, 1);
    }
    block.sync();

    if (bid < N_BINS) {
        s_bin_offsets[bid] = atomicAdd(bin_sizes + bid, s_bin_sizes[bid]);
        s_bin_offsets[bid] += bin_offsets[bid];
        s_bin_sizes[bid] = 0;
    }
    block.sync();

    if (bid < m) {
        const auto index = atomicAdd_block(s_bin_sizes + bin_idx, 1);
        bins[s_bin_offsets[bin_idx] + index] = row;
    }
}
