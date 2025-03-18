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
    using clock = std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    using DurationMS = duration<double, milliseconds::period>;

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
    constexpr int N_ITERS = 10;
    double total_time = 0.0;
    Timings timing;
    for (int i = 0; i < N_ITERS; i++) {
        const auto start = clock::now();
        opsparse(A, B, C, meta, timing);
        const auto end = clock::now();
        if (i < N_ITERS - 1)
            C.release();
        const auto elapsed = DurationMS(end - start).count();
        total_time += elapsed;
        fmt::println("Iteration {}: elapsed time = {:.3f} ms", i + 1, elapsed);
    }
    fmt::println("Average time over {} iterations: {:.3f} ms",
                 N_ITERS,
                 total_time / N_ITERS);

    // save the result
    C.D2H();
    const auto c_h = convertToUtilsCSR(C);
    c_h.save_to_filename(argv[3]);

    A.release();
    B.release();
    C.release();

    return 0;
}
