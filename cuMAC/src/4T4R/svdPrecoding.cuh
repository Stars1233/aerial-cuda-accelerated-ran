/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "api.h"
#include "cumac.h"

// cuMAC namespace
namespace cumac {

#define CUSOLVER_CHECK(call)                                                   \
{                                                                              \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSolver error in file '%s' in line %i.\n",           \
            __FILE__, __LINE__);                                           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

class svdPrecoding {
public:
    svdPrecoding(cumacCellGrpPrms* cellGrpPrms);
    ~svdPrecoding();
    svdPrecoding(const svdPrecoding&)            = delete;
    svdPrecoding& operator=(const svdPrecoding&) = delete;

    // setup() function for per TTI execution
    void setup(cumacCellGrpPrms* cellGrpPrms, cudaStream_t strm);

    // run() function for per TTI execution
    void run(cumacCellGrpPrms* cellGrpPrms);

    void destroy();

private:
    // indicator for DL
    uint8_t DL;

    // channel matrix dimensions and number of channel matrices
    int M;
    int N;
    int lda;
    int ldu;
    int ldv;
    int nChanMat;
    
    // buffers for batched cuSOLVER SVD solver
    int *devInfo;       
    cuComplex* d_hCpy;   

    // Parameters configuration of Jacobi-based SVD
    double tol;
    int    maxSweeps;
    int    econ; // --- econ = 1 for economy size 

    // handles for batched cuSOLVER SVD solver and parameter structure
    cusolverDnHandle_t solver_handle;
    gesvdjInfo_t gesvdj_params;

    // buffer for workspace of batched cuSOLVER SVD solver
    int work_size;
    cuComplex *d_work;
};

typedef struct svdPrecoding*                svdPrdHndl_t;
}