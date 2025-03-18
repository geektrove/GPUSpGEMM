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

    // numeric binning
    t0 = fast_clock_time();
    h_numeric_binning(C, meta);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.numeric_binning = fast_clock_time() - t0;

    // malloc C
    t0 = fast_clock_time();
    C.nnz = *meta.total_nnz;
    CHECK_ERROR(cudaMalloc(&C.d_val, C.nnz * sizeof(mdouble)));
    CHECK_ERROR(cudaMalloc(&C.d_col, C.nnz * sizeof(mint)));
    timing.allocate = fast_clock_time() - t0;

    // prefix sum and malloc
    t0 = fast_clock_time();
    cub::DeviceScan::ExclusiveSum(meta.d_cub_storage,
                                  meta.cub_storage_size,
                                  C.d_rpt,
                                  C.d_rpt,
                                  C.M + 1);
    CHECK_ERROR(cudaDeviceSynchronize());
    timing.prefix = fast_clock_time() - t0;

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
    // Load the matrices
    if (argc != 4) {
        fmt::println("Usage: {} <input:A> <input:B> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    const auto h_a = HostCSR::load_from_filename(argv[1]);
    const auto h_b = HostCSR::load_from_filename(argv[2]);
    if (h_a.n != h_b.m) {
        fmt::println("Matrix A columns ({}) must match matrix B rows ({})", h_a.n, h_b.m);
        return EXIT_FAILURE;
    }
    const auto d_a = h_a.to<utils::Location::Device>();
    const auto d_b = h_b.to<utils::Location::Device>();

    // Compute NIP
    const auto nip = utils::get_nip(d_a, d_b);
    fmt::println("NIP: {}", nip);
    fmt::println("FLOP: {}", 2 * nip);

    // Convert to OpSparse CSR format
    CSR A = convertFromUtilsCSR(h_a);
    CSR B = convertFromUtilsCSR(h_b);
    A.H2D();
    B.H2D();

    // Compute FLOP by OpSparse
    long total_flop = compute_flop(A, B);
    fmt::println("Total FLOP (OpSparse): {}", total_flop);

    // Warm up the GPU
    CSR C;
    Meta meta;
    {
        utils::cudaruntime_warmup();
        Timings timing;
        opsparse(A, B, C, meta, timing);
        C.release();
    }

    // Benchmark
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
    bench_timing.print(total_flop * 2);

    // save the result
    C.D2H();
    const auto c_h = convertToUtilsCSR(C);
    c_h.save_to_filename(argv[3]);

    A.release();
    B.release();
    C.release();

    return 0;
}
