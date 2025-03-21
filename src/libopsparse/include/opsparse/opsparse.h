#pragma once

#include "CSR.h"
#include "Meta.h"
#include "Timings.h"

void opsparse(const CSR& A, const CSR& B, CSR& C, Meta& meta, Timings& timing);
