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
 
 #define _USE_MATH_DEFINES

 #include <deque>
 #include "NvInfer.h"
 #include "NvOnnxParser.h"
 #include "nv_utils.h"
 #include "nv_ipc.h"
 #include "nv_ipc_utils.h"
 #include "nv_lockfree.hpp"
 #include <cstdio>
 #include <cstdlib>
 #include <cstdint>
 #include <pthread.h>
 #include <cerrno>
 #include <cstdarg>
 #include <cstddef>
 #include <unistd.h>
 #include <termios.h>
 #include <fcntl.h>
 #include <stdatomic.h>
 #include <inttypes.h>
 #include <bit>
 #include <iostream>
 #include <fstream>
 #include <random>
 #include <climits>
 #include <memory>
 #include <string>
 #include <cstring>
 #include <complex>
 #include <cmath>
 #include <filesystem>
 #include <queue>
 #include <array>
 #include <sys/queue.h>
 #include <sys/epoll.h>
 #include <semaphore.h>
 #include <thread>
 #include <algorithm>
 #include <cassert>
 #include <vector>
 #include <ctime>
 #include <chrono>
 #include <cassert>
 #include <stdexcept>
 #include <limits>
 #include <cstdlib>
 #include <cusolverDn.h>
 #include "H5Cpp.h"
 #include "cuda_fp16.h"
 #include "cuda_bf16.h"
 #include "cuda.h"
 #include <cuda_runtime.h>
 #include <curand_kernel.h>
 #include <cuComplex.h>
 #include <cooperative_groups.h>

 // cuMAC namespace
 namespace cumac {
    // Data strucures for cuMAC internal performance simulation /////////////////////////////////////////////////
    struct cumacSimParam {
        //----------------- parameters -----------------
        uint16_t    totNumCell; // total number of cells in the network (including coordinated cells)
    };
    
    // MCS tables  /////////////////////////////////////////////////
    static float L1T1B024[28] = {-4.57, -3.02, -0.6, 1.26, 2.87, 4.89, 5.9, 6.67, 7.72, 8.41, 9.36, 10.56, 11.65, 12.64, 13.23, 14.25, 15.23, 16.14, 16.72, 17.79, 19.19, 19.65, 20.95, 21.75, 22.6, 23.91, 24.38, 25.99};
    static float L1T1B050PRGS01_GTC25[28] = {-4.3, -2.44, -0.46, 1.24, 2.94, 4.89, 5.72, 6.84, 7.67, 8.67, 9.23, 10.48, 11.49, 12.37, 13.42, 14.43, 15.04, 16.0, 16.98, 18.03, 19.24, 19.88, 20.74, 21.71, 22.94, 23.5, 24.98, 26.08};
    static float minSinrCqi[15] = {-5.7456, -2.7627, 1.2773, 5.1605, 6.9945, 8.6601, 10.9361, 12.6731, 14.3437, 16.3933, 18.0984, 19.8376, 22.2465, 24.0897, 26.2837};
    
    static int mcsTableRowSizes[28] = {12, 12, 12, 12, 11, 8, 7, 7, 9, 8, 7, 9, 8, 10, 7, 7, 7, 8, 7, 7, 8, 9, 9, 8, 8, 5, 6, 7};

    static float snrMcs0[] = {-3, -3.25, -3.5, -3.75, -4, -4.25, -4.5, -4.75, -5, -5.25, -5.5, -5.75};
    static float snrMcs1[] = {-1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75, -4};
    static float snrMcs2[] = {1.5, 1.25, 1, 0.75, 0.5, 0.25, 0, -0.25, -0.5, -0.75, -1, -1.25};
    static float snrMcs3[] = {3.5, 3, 3.25, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1, 0.75};
    static float snrMcs4[] = {4.5, 4.25, 4, 3.75, 3.5, 3.25, 3, 2.75, 2.5, 2.25, 2};
    static float snrMcs5[] = {5.75, 5.5, 5.25, 5, 4.75, 4.5, 4.25, 4};
    static float snrMcs6[] = {6.5, 6.25, 6, 5.75, 5.5, 5.25, 5};
    static float snrMcs7[] = {7.25, 7, 6.75, 6.5, 6.25, 6, 5.75};
    static float snrMcs8[] = {8.75, 8.5, 8.25, 8, 7.75, 7.5, 7.25, 7, 6.75};
    static float snrMcs9[] = {9.25, 9, 8.75, 8.5, 8.25, 8, 7.75, 7.5};
    static float snrMcs10[] = {10, 9.75, 9.5, 9.25, 9, 8.75, 8.55};
    static float snrMcs11[] = {11.75, 11.5, 11.25, 11, 10.75, 10.5, 10.25, 10, 9.75};
    static float snrMcs12[] = {12.5, 12.25, 12, 11.75, 11.5, 11.25, 11, 10.75};
    static float snrMcs13[] = {13.75, 13.5, 13.25, 13, 12.75, 12.5, 12.25, 12, 11.75, 11.5};
    static float snrMcs14[] = {14, 13.75, 13.5, 13.25, 13, 12.75, 12.5};
    static float snrMcs15[] = {15, 14.75, 14.5, 14.25, 14, 13.75, 13.5};
    static float snrMcs16[] = {16, 15.75, 15.5, 15.25, 15, 14.75, 14.5};
    static float snrMcs17[] = {17, 16.75, 16.5, 16.25, 16, 15.75, 15.5, 15.25};
    static float snrMcs18[] = {17.5, 17.25, 17, 16.75, 16.5, 16.25, 16};
    static float snrMcs19[] = {18.5, 18.25, 18, 17.75, 17.5, 17.25, 17};
    static float snrMcs20[] = {20.25, 20, 19.75, 19.5, 19.25, 19, 18.75, 18.5};
    static float snrMcs21[] = {21.25, 20.5, 20.25, 20, 19.75, 19.5, 19.25, 19, 18.75};
    static float snrMcs22[] = {22, 21.75, 21.5, 21.25, 21, 20.75, 20.5, 20.25, 20};
    static float snrMcs23[] = {22.75, 22.5, 22.25, 22, 21.75, 21.5, 21.25, 21};
    static float snrMcs24[] = {23.5, 23.25, 23, 22.75, 22.5, 22.25, 22, 21.5};
    static float snrMcs25[] = {24.25, 24, 23.75, 23.5, 23.25};
    static float snrMcs26[] = {25, 24.75, 24.5, 24.25, 24, 23.75};
    static float snrMcs27[] = {26.75, 26.5, 26.25, 26, 25.75, 25.5, 25.25};

    static float blerMcs0[] = {0.00848, 0.01112, 0.01412, 0.01784, 0.02476, 0.03732, 0.07168, 0.17728, 0.42324, 0.7258, 0.92464, 0.9898};
    static float blerMcs1[] = {0.00376, 0.0046, 0.00584, 0.00788, 0.01056, 0.01412, 0.02332, 0.08384, 0.3422, 0.74496, 0.95904, 0.9982};
    static float blerMcs2[] = {0.00084, 0.00096, 0.0016, 0.00236, 0.00312, 0.00488, 0.0066, 0.01096, 0.03352, 0.20448, 0.65944, 0.95844};
    static float blerMcs3[] = {0.00016, 0.00024, 0.00028, 0.00036, 0.00068, 0.00104, 0.00152, 0.00264, 0.0076, 0.10476, 0.60312, 0.96944};
    static float blerMcs4[] ={4.0e-05, 0.00012, 0.00016, 0.00024, 0.0004, 0.00092, 0.00572, 0.18384, 0.8092, 0.99616, 0.99996};
    static float blerMcs5[] = {0, 4.0e-05, 0.00064, 0.0134, 0.21124, 0.79344, 0.99432, 1};
    static float blerMcs6[] = {0, 0.00088, 0.01932, 0.2198, 0.78464, 0.99264, 1};
    static float blerMcs7[] = {0, 0.00112, 0.02292, 0.25448, 0.81236, 0.99568, 1};
    static float blerMcs8[] = {0, 4.0e-05, 0.0004, 0.00476, 0.05988, 0.41568, 0.91896, 0.99908, 1};
    static float blerMcs9[] = {0, 4.0e-05, 0.00024, 0.00892, 0.26044, 0.91652, 0.99992, 1};
    static float blerMcs10[] = {0, 0.00016, 0.00612, 0.17688, 0.8322, 0.99872, 1};
    static float blerMcs11[] = {0, 4.0e-05, 8.0e-05, 0.00112, 0.01196, 0.12876, 0.63684, 0.98184, 1};
    static float blerMcs12[] = {0, 0.00036, 0.00304, 0.02516, 0.20844, 0.73136, 0.98916, 0.99996};
    static float blerMcs13[] = {0, 4.0e-05, 0.00012, 0.00192, 0.02188, 0.2058, 0.7292, 0.99004, 0.99992, 1};
    static float blerMcs14[] = {0, 4.0e-05, 0.00208, 0.06104, 0.60688, 0.99032, 1};
    static float blerMcs15[] = {0, 0.0002, 0.00408, 0.0966, 0.64696, 0.99036, 1};
    static float blerMcs16[] = {0, 0.00016, 0.00376, 0.06976, 0.555, 0.98308, 1};
    static float blerMcs17[] = {0, 4.0e-05, 0.00044, 0.01, 0.22384, 0.88576, 0.99964, 1};
    static float blerMcs18[] = {0, 8.0e-05, 0.00092, 0.04312, 0.59976, 0.99444, 1};
    static float blerMcs19[] = {0, 4.0e-05, 0.0032, 0.11948, 0.766, 0.99852, 1};
    static float blerMcs20[] = {0, 4.0e-05, 0.00012, 0.0016, 0.02624, 0.31976, 0.91032, 0.99988};
    static float blerMcs21[] = {0, 4.0e-05, 0.00016, 0.00096, 0.01744, 0.22232, 0.83812, 0.99856, 1};
    static float blerMcs22[] = {0, 4.0e-05, 0.00028, 0.00232, 0.03628, 0.37684, 0.94368, 0.99988, 1};
    static float blerMcs23[] = {0, 0.00012, 0.00044, 0.00604, 0.09064, 0.59456, 0.98468, 1};
    static float blerMcs24[] = {0, 4.0e-05, 0.0004, 0.00792, 0.16468, 0.81412, 0.99904, 1};
    static float blerMcs25[] = {0, 0.00864, 0.25544, 0.91956, 0.99992};
    static float blerMcs26[] = {0, 8.0e-05, 0.00564, 0.197, 0.88056, 0.99968};
    static float blerMcs27[] = {0, 0.0002, 0.00428, 0.07912, 0.5296, 0.96752, 0.99988};

    static float mcsTable_codeRate[] = {120.0000, 193.0000, 308.0000, 449.0000, 602.0000, 378.0000, 434.0000, 490.0000, 553.0000, 616.0000, 658.0000, 466.0000, 
    517.0000, 567.0000, 616.0000, 666.0000, 719.0000, 772.0000, 822.0000, 873.0000, 682.5000, 711.0000, 754.0000, 797.0000, 841.0000, 885.0000, 916.5000, 948.0000}; 

    static uint8_t mcsTable_qamOrder[] = {2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8};

    static uint32_t TBS_table[] = {24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,208,224,240,256,272,288,304,320,336,352,368,384,408,432,456,480,504,528,552,576,608,640,672,704,736,768,808,848,888,928,984,1032,1064,1128,1160,1192,1224,1256,1288,1320,1352,1416,1480,1544,1608,1672,1736,1800,1864,1928,2024,2088,2152,2216,2280,2408,2472,2536,2600,2664,2728,2792,2856,2976,3104,3240,3368,3496,3624,3752,3824};

    static uint32_t TBS_table_size = 93;

    typedef enum {
        STATUS_SUCCESS = 0,  /*!< The API call returned with no errors.                                    */
        STATUS_ERROR   = 1
    } status_t;

    #define CUDA_CHECK_ERR(call)                                                 \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

    #define CUDA_CHECK_RES(call)                                                    \
    {                                                                               \
        CUresult runStatus = call;                                                  \
        if (runStatus != CUDA_SUCCESS) {                                            \
            const char* errorName;                                                  \
            const char* errorMessage;                                               \
            cuGetErrorName(runStatus, &errorName);                                  \
            cuGetErrorString(runStatus, &errorMessage);                             \
            fprintf(stderr, "CUDA result error in file '%s' in line %i: %s - %s.\n",\
                    __FILE__, __LINE__, errorName, errorMessage);                   \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

    // launch configuration structure /////////////////////////////////////////////////
    typedef struct {
        CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
        void*                   kernelArgs[2];
    } launchCfg_t;
 }
 #include "cumac_pfm_sort.h"
 #include "cumac_muUeGrp.h"
 #include "cumac_msg.h"
 #include "cpuMatAlg/cpuMatAlg.h"
 #include "4T4R/cellAssociation.cuh"
 #include "4T4R/cellAssociationCpu.h"
 #include "4T4R/mcsSelectionLUT.cuh"
 #include "4T4R/mcsSelectionLUTCpu.h"
 #include "4T4R/multiCellScheduler.cuh"
 #include "4T4R/multiCellSchedulerCpu.h"
 #include "4T4R/roundRobinScheduler.cuh"
 #include "4T4R/roundRobinSchedulerCpu.h"
 #include "4T4R/singleCellScheduler.cuh"
 #include "4T4R/singleCellSchedulerCpu.h"
 #include "4T4R/svdPrecoding.cuh"
 #include "4T4R/multiCellUeSelection.cuh"
 #include "4T4R/multiCellLayerSel.cuh"
 #include "4T4R/multiCellUeSelectionCpu.h"
 #include "4T4R/multiCellLayerSelCpu.h"
 #include "4T4R/roundRobinUeSel.cuh"
 #include "4T4R/roundRobinUeSelCpu.h"
 #include "64T64R/multiCellMuUeSort.cuh"
 #include "64T64R/multiCellMuUeGrp.cuh"
 #include "64T64R/multiCellBeamform.cuh"   
 #include "64T64R/muMimoSchedulerBaseCpu.h"
 #include "pfmSort/pfmSort.cuh"
 #include "multiCellSrsScheduler/multiCellSrsScheduler.cuh"
 #include "tools/multiCellSinrCal.cuh"
 #include "cumacSubcontext.h"