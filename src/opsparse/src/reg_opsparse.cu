#include <cub/cub.cuh>
#include <cuda_profiler_api.h>

#include <utils/utils.cuh>

#include "conversion.cuh"
#include "kernel_wrapper.cuh"
#include "Timings.h"

void opsparse(const CSR& A, const CSR& B, CSR& C, Meta& meta, Timings& timing) {
    double t0, t1;
    t1 = t0 = fast_clock_time();
    C.M = A.M;
    C.N = B.N;
    C.nnz = 0;
    h_setup(A, B, C, meta, timing);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.setup = fast_clock_time() - t0;

    // symbolic binning
    t0 = fast_clock_time();
    h_symbolic_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic_binning = fast_clock_time() - t0;

    // symbolic phase
    t0 = fast_clock_time();
    h_symbolic(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.symbolic = fast_clock_time() - t0;

    // numeric binning, exclusive sum, and allocate C
    meta.memset_all(0);
    mint BS = 1024;
    mint GS = div_up(C.M, BS);
    k_numeric_binning<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt,
                                                     C.M,
                                                     meta.d_bin_size,
                                                     meta.d_total_nnz,
                                                     meta.d_max_row_nnz);
    meta.D2H_all(0);
    CHECK_ERROR(cudaStreamSynchronize(meta.stream[0]));
    C.nnz = *meta.total_nnz;

    if (*meta.max_row_nnz <= 16) {
        k_binning_small<<<GS, BS>>>(meta.d_bins, C.M);
        CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
        meta.bin_size[0] = C.M;
        for (int i = 1; i < NUM_BIN; i++) {
            meta.bin_size[i] = 0;
        }
        meta.bin_offset[0] = 0;
        for (int i = 1; i < NUM_BIN; i++) {
            meta.bin_offset[i] = C.M;
        }
    } else {
        meta.memset_bin_size(0);
        meta.bin_offset[0] = 0;
        for (int i = 0; i < NUM_BIN - 1; i++) {
            meta.bin_offset[i + 1] = meta.bin_offset[i] + meta.bin_size[i];
        }
        meta.H2D_bin_offset(0);

        k_numeric_binning2<<<GS, BS, 0, meta.stream[0]>>>(C.d_rpt,
                                                          C.M,
                                                          meta.d_bins,
                                                          meta.d_bin_size,
                                                          meta.d_bin_offset);
        CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    }
    CHECK_ERROR(cudaDeviceSynchronize());

    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage,
                                  meta.cub_storage_size,
                                  C.d_rpt,
                                  C.d_rpt,
                                  C.M + 1);
    CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    CHECK_ERROR(cudaDeviceSynchronize());

    // numeric
    t0 = fast_clock_time();
    h_numeric_full_occu(A, B, C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric = fast_clock_time() - t0;

    // cleanup
    t0 = fast_clock_time();
    meta.release();
    timing.cleanup = fast_clock_time() - t0;
    timing.total = fast_clock_time() - t1;
}

int main(int argc, char** argv) {
    using HostCSR = utils::CSR<double, utils::Location::Host>;

    if (argc != 4) {
        fmt::println("Usage: {} <input:A> <input:B> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    CSR A = convertFromUtilsCSR(HostCSR::load_from_filename(argv[1]));
    CSR B = convertFromUtilsCSR(HostCSR::load_from_filename(argv[2]));

    A.H2D();
    B.H2D();

    long total_flop = compute_flop(A, B);
    CSR C;
    cudaruntime_warmup();
    Meta meta;
    {
        Timings timing;
        opsparse(A, B, C, meta, timing);
        C.release();
    }

    mint iter = 10;
    Timings timing, bench_timing;
    for (mint i = 0; i < iter; i++) {
        opsparse(A, B, C, meta, timing);
        bench_timing += timing;
        if (i < iter - 1) {
            C.release();
        }
    }
    bench_timing /= iter;

    bench_timing.reg_print(total_flop * 2);

    // save the result
    C.D2H();
    const auto c_h = convertToUtilsCSR(C);
    c_h.save_to_filename(argv[3]);

    A.release();
    B.release();
    C.release();

    return 0;
}
