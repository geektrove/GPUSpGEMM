#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/functional>
#include <gsl/gsl-lite.hpp>

#include <proposal/setup.cuh>

namespace cg = cooperative_groups;

__global__ void k_compute_nip(const std::int32_t* __restrict__ a_rpt,
                              const std::int32_t* __restrict__ a_col,
                              const std::int32_t* __restrict__ b_rpt,
                              const std::int32_t m,
                              std::int32_t* __restrict__ nips,
                              std::int32_t* __restrict__ max_nip) {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    // TODO: Compute optimal block dimensions
    const auto tile = cg::tiled_partition<1024>(block);

    const auto row = gsl::narrow_cast<std::int32_t>(grid.thread_rank());
    const auto l_max_nip = cuda::std::invoke([&] {
        if (row >= m)
            return 0;
        std::int32_t row_nip = 0;
        for (auto j = a_rpt[row]; j < a_rpt[row + 1]; j++) {
            const auto col = a_col[j];
            row_nip += b_rpt[col + 1] - b_rpt[col];
        }
        nips[row] = row_nip;
        return row_nip;
    });

    const auto s_max_nip = cg::reduce(tile, l_max_nip, cg::greater<std::int32_t>{});
    cg::invoke_one(block, [&] { atomicMax(max_nip, s_max_nip); });
}
