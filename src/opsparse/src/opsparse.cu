#include <fstream>

#include <cub/cub.cuh>
#include <cuda_profiler_api.h>

#include <utils/csr.cuh>

#include "cusparse_spgemm.h"
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
    if (argc != 3) {
        fmt::println("Usage: {} <input> <output>", argv[0]);
        return EXIT_FAILURE;
    }
    const auto* input = argv[1];
    const auto* output = argv[2];

    auto h_a = utils::CSR<double, utils::Location::Host>::load_from_filename(input);

    CSR A;
    A.M = h_a.m;
    A.N = h_a.n;
    A.nnz = h_a.nnz;
    A.rpt = new mint[A.M + 1];
    A.col = new mint[A.nnz];
    A.val = new mdouble[A.nnz];
    std::copy(h_a.rows_ptr, h_a.rows_ptr + A.M + 1, A.rpt);
    std::copy(h_a.cols, h_a.cols + A.nnz, A.col);
    std::copy(h_a.values, h_a.values + A.nnz, A.val);
    A.d_rpt = nullptr;
    A.d_col = nullptr;
    A.d_val = nullptr;

    CSR B = A;

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

    // printf("%s ", mat1.c_str());
    bench_timing.print(total_flop * 2);

    // compare result

    //C.D2H();
    //CSR C_ref;
    //cusparse_spgemm(&A, &B, &C_ref);
    //C_ref.D2H();
    //if(C == C_ref){
    //    printf("pass\n");
    //}
    //else{
    //    printf("error\n");
    //}

    A.release();
    B.release();
    C.release();

    return 0;
}
