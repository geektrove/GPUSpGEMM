#pragma once

#include <utils/utils.cuh>

#include "CSR.h"

inline auto convertFromUtilsCSR(const utils::HostCSR<mdouble>& from) -> CSR {
    CSR to;
    to.M = from.m;
    to.N = from.n;
    to.nnz = from.nnz;
    to.rpt = new mint[to.M + 1];
    to.col = new mint[to.nnz];
    to.val = new mdouble[to.nnz];
    std::copy_n(from.rpt, to.M + 1, to.rpt);
    std::copy_n(from.col, to.nnz, to.col);
    std::copy_n(from.val, to.nnz, to.val);
    to.d_rpt = nullptr;
    to.d_col = nullptr;
    to.d_val = nullptr;
    return to;
}

inline auto convertToUtilsCSR(const CSR& from) -> utils::HostCSR<mdouble> {
    utils::HostCSR<mdouble> to(from.nnz, from.M, from.N);
    std::copy_n(from.rpt, from.M + 1, to.rpt);
    std::copy_n(from.col, from.nnz, to.col);
    std::copy_n(from.val, from.nnz, to.val);
    return to;
}
