#pragma once

#include <CSR.h>
#include <utils/utils.cuh>

inline auto convertFromUtilsCSR(const utils::HostCSR<mdouble>& from) -> CSR {
    CSR to;
    to.M = from.m;
    to.N = from.n;
    to.nnz = from.nnz;
    to.rpt = new mint[to.M + 1];
    to.col = new mint[to.nnz];
    to.val = new mdouble[to.nnz];
    std::copy_n(from.rows_ptr, to.M + 1, to.rpt);
    std::copy_n(from.cols, to.nnz, to.col);
    std::copy_n(from.values, to.nnz, to.val);
    to.d_rpt = nullptr;
    to.d_col = nullptr;
    to.d_val = nullptr;
    return to;
}

inline auto convertToUtilsCSR(const CSR& from) -> utils::HostCSR<mdouble> {
    utils::HostCSR<mdouble> to(from.nnz, from.M, from.N);
    std::copy_n(from.rpt, from.M + 1, to.rows_ptr);
    std::copy_n(from.col, from.nnz, to.cols);
    std::copy_n(from.val, from.nnz, to.values);
    return to;
}
