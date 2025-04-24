#pragma once

#include <cstdint>
#include <cuda/atomic>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <utils/csr.cuh>
#include <utils/errors.cuh>
#include <utils/runtime.cuh>

namespace utils {

namespace cg = cooperative_groups;

template<std::int32_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void k_get_nip(const __grid_constant__ std::int32_t* const __restrict__ a_rpt,
                   const __grid_constant__ std::int32_t* const __restrict__ a_col,
                   const __grid_constant__ std::int32_t* const __restrict__ b_rpt,
                   const __grid_constant__ std::int32_t m,
                   __grid_constant__ std::int64_t* const __restrict__ nip) {
    using ReduceT = cub::
        BlockReduce<std::int32_t, BLOCK_SIZE, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ typename ReduceT::TempStorage s_storage;

    const auto tig = gsl::narrow_cast<std::int32_t>(cg::this_grid().thread_rank());

    const auto a_rpt_start = tig < m ? a_rpt[tig] : 0;
    const auto a_rpt_end = tig < m ? a_rpt[tig + 1] : 0;
    std::int32_t row_nip = 0;
    for (auto j = a_rpt_start; j < a_rpt_end; j++) {
        const auto col = a_col[j];
        row_nip += b_rpt[col + 1] - b_rpt[col];
    }

    const std::int64_t sum = ReduceT(s_storage).Sum(row_nip);
    cg::invoke_one(cg::this_thread_block(), [&] {
        cuda::atomic_ref<std::int64_t, cuda::thread_scope_device> ref(*nip);
        ref.fetch_add(sum);
    });
}

template<std::floating_point T>
auto get_nip(const DeviceCSR<T>& a, const DeviceCSR<T>& b) -> std::int64_t {
    static constexpr std::int32_t BLOCK_SIZE = 512;

    std::int64_t nip = 0;
    auto* tmp = utils::malloc_async<Location::Device>(sizeof(std::int64_t));
    auto* d_nip = static_cast<std::int64_t*>(tmp);
    utils::memset_async(d_nip, 0, sizeof(std::int64_t));
    utils::launch_kernel(k_get_nip<BLOCK_SIZE>,
                         cuda::ceil_div(a.m, BLOCK_SIZE),
                         BLOCK_SIZE,
                         0,
                         cudaStreamDefault,
                         a.rpt,
                         a.col,
                         b.rpt,
                         a.m,
                         d_nip);
    utils::memcpy_async(&nip, d_nip, sizeof(std::int64_t));
    utils::free_async(d_nip);

    utils::stream_sync();

    return nip;
}

} // namespace utils
