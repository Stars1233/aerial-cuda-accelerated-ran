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

#include "multiCellSrsScheduler.cuh"

// cuMAC namespace
namespace cumac {

// #define SCHEDULER_KERNEL_TIME_MEASURE_ 
#ifdef SCHEDULER_KERNEL_TIME_MEASURE_
constexpr uint16_t numRunSchKnlTimeMsr 1000;
#endif

#define dir 0

__device__ __constant__ uint16_t d_SRS_BW_TABLE[64][8]; 
    
static uint16_t SRS_BW_TABLE[64][8] =
   {{4,1,4,1,4,1,4,1},
    {8,1,4,2,4,1,4,1},
    {12,1,4,3,4,1,4,1},
    {16,1,4,4,4,1,4,1},
    {16,1,8,2,4,2,4,1},
    {20,1,4,5,4,1,4,1},
    {24,1,4,6,4,1,4,1},
    {24,1,12,2,4,3,4,1},
    {28,1,4,7,4,1,4,1},
    {32,1,16,2,8,2,4,2},
    {36,1,12,3,4,3,4,1},
    {40,1,20,2,4,5,4,1},
    {48,1,16,3,8,2,4,2},
    {48,1,24,2,12,2,4,3},
    {52,1,4,13,4,1,4,1},
    {56,1,28,2,4,7,4,1},
    {60,1,20,3,4,5,4,1},
    {64,1,32,2,16,2,4,4},
    {72,1,24,3,12,2,4,3},
    {72,1,36,2,12,3,4,3},
    {76,1,4,19,4,1,4,1},
    {80,1,40,2,20,2,4,5},
    {88,1,44,2,4,11,4,1},
    {96,1,32,3,16,2,4,4},
    {96,1,48,2,24,2,4,6},
    {104,1,52,2,4,13,4,1},
    {112,1,56,2,28,2,4,7},
    {120,1,60,2,20,3,4,5},
    {120,1,40,3,8,5,4,2},
    {120,1,24,5,12,2,4,3},
    {128,1,64,2,32,2,4,8},
    {128,1,64,2,16,4,4,4},
    {128,1,16,8,8,2,4,2},
    {132,1,44,3,4,11,4,1},
    {136,1,68,2,4,17,4,1},
    {144,1,72,2,36,2,4,9},
    {144,1,48,3,24,2,12,2},
    {144,1,48,3,16,3,4,4},
    {144,1,16,9,8,2,4,2},
    {152,1,76,2,4,19,4,1},
    {160,1,80,2,40,2,4,10},
    {160,1,80,2,20,4,4,5},
    {160,1,32,5,16,2,4,4},
    {168,1,84,2,28,3,4,7},
    {176,1,88,2,44,2,4,11},
    {184,1,92,2,4,23,4,1},
    {192,1,96,2,48,2,4,12},
    {192,1,96,2,24,4,4,6},
    {192,1,64,3,16,4,4,4},
    {192,1,24,8,8,3,4,2},
    {208,1,104,2,52,2,4,13},
    {216,1,108,2,36,3,4,9},
    {224,1,112,2,56,2,4,14},
    {240,1,120,2,60,2,4,15},
    {240,1,80,3,20,4,4,5},
    {240,1,48,5,16,3,8,2},
    {240,1,24,10,12,2,4,3},
    {256,1,128,2,64,2,4,16},
    {256,1,128,2,32,4,4,8},
    {256,1,16,16,8,2,4,2},
    {264,1,132,2,44,3,4,11},
    {272,1,136,2,68,2,4,17},
    {272,1,68,4,4,17,4,1},
    {272,1,16,17,8,2,4,2}};

static __device__ __constant__ uint16_t pow2NArr[2048] = {1,2,4,4,8,8,8,8,16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,512,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,1024,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048,2048};

 inline __device__ void bitonicSort(int32_t* valueArr, uint16_t* idArr, uint16_t n)
  {
    for (int size = 2; size < n; size*=2) {
        int d=dir^((threadIdx.x & (size / 2)) != 0);
       
        for (int stride = size / 2; stride > 0; stride/=2) {
           __syncthreads(); 

           if(threadIdx.x<n/2) {
              int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

              float t;
              int t_id;

              if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == d) {
                  t = valueArr[pos];
                  valueArr[pos] = valueArr[pos + stride];
                  valueArr[pos + stride] = t;
                  t_id = idArr[pos];
                  idArr[pos] = idArr[pos + stride];
                  idArr[pos + stride] = t_id;
              }
           }
        }
    }
    
    for (int stride = n / 2; stride > 0; stride/=2) {
        __syncthreads(); 
        if(threadIdx.x<n/2) {
           int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

           float t;
           int t_id;

           if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == dir) {
               t = valueArr[pos];
               valueArr[pos] = valueArr[pos + stride];
               valueArr[pos + stride] = t;
             
               t_id = idArr[pos];
               idArr[pos] = idArr[pos + stride];
               idArr[pos + stride] = t_id;
           }
        }
    }

    __syncthreads(); 
 }

multiCellSrsScheduler::multiCellSrsScheduler()
{
    // allocate memory for dynamic descriptors
    pCpuDynDesc = std::make_unique<mcSrsSchedulerDynDescr_t>();
    CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(mcSrsSchedulerDynDescr_t)));

    pLaunchCfg = std::make_unique<launchCfg_t>();
    
}

multiCellSrsScheduler::~multiCellSrsScheduler()
{
    CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
}

 __forceinline__ __device__ void multiCellSrsSchedulerTpcKernel(mcSrsSchedulerDynDescr_t* pDynDescr, uint16_t ueIdx, float W_SRS_LAST)
{
    float       srsTxPwrMax                    = pDynDescr->srsTxPwrMax[ueIdx];
    float       srsPwr0                        = pDynDescr->srsPwr0[ueIdx];
    float       srsPwrAlpha                    = pDynDescr->srsPwrAlpha[ueIdx];
    uint8_t     srsTpcAccumulationFlag         = pDynDescr->srsTpcAccumulationFlag[ueIdx];
    float       srsPowerControlAdjustmentState = pDynDescr->srsPowerControlAdjustmentState[ueIdx]; 
    
    uint8_t     srsConfigIndex                 = pDynDescr->srsConfigIndex[ueIdx];
    uint8_t     srsBwIndex                     = pDynDescr->srsBwIndex[ueIdx];
    uint8_t     srsCombSize                    = pDynDescr->srsCombSize[ueIdx];
    
    float       srsWbSnr                       = pDynDescr->srsWbSnr[ueIdx];
    float       srsWbSnrThreshold              = pDynDescr->srsWbSnrThreshold[ueIdx];
    
    float       srsTxPwr                       = pDynDescr->srsTxPwr[ueIdx];
    float       srsWidebandSignalEnergy        = fmaxf(pDynDescr->srsWidebandSignalEnergy[ueIdx], 1e-9f);
     
    float       pathLossEst                    = srsTxPwr - 10.0f*log10f(srsWidebandSignalEnergy/W_SRS_LAST); 
    float       M_SRS                          = d_SRS_BW_TABLE[63][0]*12.0f/srsCombSize;
    
    if((srsConfigIndex<=63)&&(srsBwIndex<=3))
    {
        M_SRS = d_SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f/srsCombSize;
    }
    
    if(srsTpcAccumulationFlag==1) //TPC commands via accumulation, large dynamic range for power control
    {
        float srsTxPwrGap = srsTxPwrMax - srsPwr0 - 10.0f*log10f(2.0f*M_SRS) - srsPwrAlpha * pathLossEst - srsPowerControlAdjustmentState;
        if(srsTxPwrGap<=0.0f)
        {
            pDynDescr->srsPowerControlAdjustmentState[ueIdx] = (srsPowerControlAdjustmentState + floorf(srsTxPwrGap));
        }
        else
        {
            float srsWbSnrGap = srsWbSnrThreshold - srsWbSnr;
            float srsTxPwrDelta = fminf(ceilf(srsWbSnrGap), srsTxPwrGap);
            pDynDescr->srsPowerControlAdjustmentState[ueIdx] = (srsPowerControlAdjustmentState + floorf(srsTxPwrDelta));
        }
        pDynDescr->srsTxPwr[ueIdx] = fminf(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + pDynDescr->srsPowerControlAdjustmentState[ueIdx]); 
    }
    else if(srsTpcAccumulationFlag==0) //TPC commands via without accumulation, small dynamic range for power control
    {
        float srsTxPwrGap = srsTxPwrMax - srsPwr0 - 10.0f*log10f(2.0f*M_SRS) - srsPwrAlpha * pathLossEst;
        if(srsTxPwrGap<=0.0f)
        {
            if(srsTxPwrGap>=-1.0f)
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = -1.0f;
                pDynDescr->srsTxPwr[ueIdx] = fminf(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + pDynDescr->srsPowerControlAdjustmentState[ueIdx]); 
            }
            else if(srsTxPwrGap>=-4.0f)
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = -4.0f;
                pDynDescr->srsTxPwr[ueIdx] = fminf(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + pDynDescr->srsPowerControlAdjustmentState[ueIdx]); 
            }
            else
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = -4.0f;
                pDynDescr->srsTxPwr[ueIdx] = pDynDescr->srsTxPwrMax[ueIdx];
            }
        }
        else
        {
            float srsWbSnrGap = srsWbSnrThreshold - (srsWbSnr-srsPowerControlAdjustmentState);
            float srsTxPwrDelta = fminf(ceilf(srsWbSnrGap), srsTxPwrGap);
            if(srsTxPwrDelta>=4.0f)
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = 4.0f;
            }
            else if(srsTxPwrDelta>=1.0f)
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = 1.0f;
            }
            else if(srsTxPwrDelta>=-1.0f)
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = -1.0f;
            }
            else
            {
                pDynDescr->srsPowerControlAdjustmentState[ueIdx] = -4.0f;
            }
            pDynDescr->srsTxPwr[ueIdx] = fminf(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + pDynDescr->srsPowerControlAdjustmentState[ueIdx]); 
        }
    }
}

static __global__ void multiCellSrsSchedulerKernel_v0(mcSrsSchedulerDynDescr_t* pDynDescr)
{
   uint16_t cIdx = blockIdx.x;
   uint16_t cellIdx = pDynDescr->cellId[cIdx];
   
   pDynDescr->nSrsScheduledUePerCell[cellIdx] = pDynDescr->nSymbsPerSlot*SRS_COMB_SIZE; 

   __shared__ int32_t srsAging[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];

   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      srsAging[eIdx] = -100;
      ueIds[eIdx] = 0xFFFF;
   }

   __shared__ int nAssocUeFound;
   if (threadIdx.x == 0) {
      nAssocUeFound = 0;
   }
   __syncthreads();

   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) 
   {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx]) 
      {
         int storeIdx = atomicAdd(&nAssocUeFound, 1);
         srsAging[storeIdx] = pDynDescr->srsLastTxCounter[uIdx];
         ueIds[storeIdx] = uIdx; 
         ///// update srsLastTxCounter
         pDynDescr->srsLastTxCounter[uIdx]+=1;
         ////  update RX cell
         pDynDescr->srsRxCell[uIdx]=cellIdx;
      }
   }
   __syncthreads();

   if (nAssocUeFound > 0) 
   {
      if(pDynDescr->nSrsScheduledUePerCell[cellIdx] > nAssocUeFound)
      {
          pDynDescr->nSrsScheduledUePerCell[cellIdx] = nAssocUeFound;
      }
      uint16_t pow2N = pow2NArr[nAssocUeFound-1];

      bitonicSort(srsAging, ueIds, pow2N); // internal synchronization
   }
   else
   {
       pDynDescr->nSrsScheduledUePerCell[cellIdx] = 0;
       return;
   }
   
   
   if (threadIdx.x < pDynDescr->nSrsScheduledUePerCell[cellIdx]) 
   {  
      uint8_t srsConfigIndex = pDynDescr->srsConfigIndex[ueIds[threadIdx.x]];
      uint8_t srsBwIndex = pDynDescr->srsBwIndex[ueIds[threadIdx.x]];
      float W_SRS_LAST = d_SRS_BW_TABLE[63][0]*12.0f;
      if((srsConfigIndex<=63)&&(srsBwIndex<=3))
      {
          W_SRS_LAST = d_SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
      }
      
      pDynDescr->srsTxUe[cIdx*pDynDescr->nMaxActUePerCell + threadIdx.x] = ueIds[threadIdx.x];
      pDynDescr->srsLastTxCounter[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsNumSymb[ueIds[threadIdx.x]] = 1;
      pDynDescr->srsTimeStart[ueIds[threadIdx.x]] = static_cast<uint8_t>(floorf(threadIdx.x/SRS_COMB_SIZE)); 
      pDynDescr->srsNumRep[ueIds[threadIdx.x]] = 1;
      pDynDescr->srsCombSize[ueIds[threadIdx.x]] = SRS_COMB_SIZE;
      pDynDescr->srsCombOffset[ueIds[threadIdx.x]] = static_cast<uint8_t>(floorf(threadIdx.x%SRS_COMB_SIZE)); 
      pDynDescr->srsSequenceId[ueIds[threadIdx.x]] = (40+cellIdx);
      pDynDescr->srsGroupOrSequenceHopping[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsFreqHopping[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsConfigIndex[ueIds[threadIdx.x]] = 63;
      pDynDescr->srsBwIndex[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsFreqStart[ueIds[threadIdx.x]] = 0;  
      pDynDescr->srsFreqShift[ueIds[threadIdx.x]] = 0; 
      pDynDescr->srsCyclicShift[ueIds[threadIdx.x]] = 0;   
      
      multiCellSrsSchedulerTpcKernel(pDynDescr, ueIds[threadIdx.x], W_SRS_LAST);
          
   }
}

static __global__ void multiCellSrsSchedulerKernel_v1(mcSrsSchedulerDynDescr_t* pDynDescr)
{
   uint16_t cIdx = blockIdx.x;
   uint16_t cellIdx = pDynDescr->cellId[cIdx];
   
   pDynDescr->nSrsScheduledUePerCell[cellIdx] = pDynDescr->nSymbsPerSlot*SRS_COMB_SIZE; 
   
   uint16_t nSrsScheduledMuMimoUe = (uint16_t)floorf(pDynDescr->nSrsScheduledUePerCell[cellIdx]/3.0);
   uint16_t nSrsScheduledSuMimoUe = (uint16_t)floorf(pDynDescr->nSrsScheduledUePerCell[cellIdx]/3.0);
   
   __shared__ uint16_t nSrsScheduledUe;
   
   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) 
   {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx]) 
      { 
         ///// update srsLastTxCounter
         pDynDescr->srsLastTxCounter[uIdx]+=1;
         ////  update RX cell
         pDynDescr->srsRxCell[uIdx]=cellIdx;
      }
   }
   __syncthreads();
   
   ///////// Schedule muMIMO UEs ////////////////////////////////
   if(threadIdx.x == 0)
   {
       nSrsScheduledUe = 0;
       for(uint16_t idx = 0; idx < pDynDescr->nMaxActUePerCell; idx++)
       {
           uint16_t ueIdx = pDynDescr->sortedUeList[cIdx][idx];
           if((pDynDescr->srsLastTxCounter[ueIdx]>0)&&(pDynDescr->muMimoInd[ueIdx]==1))
           {    
               uint8_t srsConfigIndex = pDynDescr->srsConfigIndex[ueIdx];
               uint8_t srsBwIndex = pDynDescr->srsBwIndex[ueIdx];
               float W_SRS_LAST = d_SRS_BW_TABLE[63][0]*12.0f;
               if((srsConfigIndex<=63)&&(srsBwIndex<=3))
               {
                   W_SRS_LAST = d_SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
               } 
               
               pDynDescr->srsTxUe[cIdx*pDynDescr->nMaxActUePerCell + nSrsScheduledUe] = ueIdx;
               pDynDescr->srsLastTxCounter[ueIdx] = 0;
               pDynDescr->srsNumSymb[ueIdx] = 1;
               pDynDescr->srsTimeStart[ueIdx] = static_cast<uint8_t>(floorf(nSrsScheduledUe/SRS_COMB_SIZE)); 
               pDynDescr->srsNumRep[ueIdx] = 1;
               pDynDescr->srsCombSize[ueIdx] = SRS_COMB_SIZE;
               pDynDescr->srsCombOffset[ueIdx] = static_cast<uint8_t>(floorf(nSrsScheduledUe%SRS_COMB_SIZE)); 
               pDynDescr->srsSequenceId[ueIdx] = (40+cellIdx);
               pDynDescr->srsGroupOrSequenceHopping[ueIdx] = 0;
               pDynDescr->srsFreqHopping[ueIdx] = 0;
               pDynDescr->srsConfigIndex[ueIdx] = 63;
               pDynDescr->srsBwIndex[ueIdx] = 0;
               pDynDescr->srsFreqStart[ueIdx] = 0;  
               pDynDescr->srsFreqShift[ueIdx] = 0; 
               pDynDescr->srsCyclicShift[ueIdx] = 0;
               
               multiCellSrsSchedulerTpcKernel(pDynDescr, ueIdx, W_SRS_LAST);  
               
               nSrsScheduledUe += 1;  
               
               if(nSrsScheduledUe == nSrsScheduledMuMimoUe)
               {
                   break;
               }
               
               if(nSrsScheduledUe == pDynDescr->nMaxActUePerCell)
               {
                   return;
               }
           }
       }
   }
   __syncthreads();
   
   ///////// Schedule suMIMO UEs ///////////////////////////////////
   if(threadIdx.x==0)
   {
       for(uint16_t idx = 0; idx < pDynDescr->nMaxActUePerCell; idx++)
       {
           uint16_t ueIdx = pDynDescr->sortedUeList[cIdx][idx];
           if((pDynDescr->srsLastTxCounter[ueIdx]>0)&&(pDynDescr->muMimoInd[ueIdx]==0))
           {   
               uint8_t srsConfigIndex = pDynDescr->srsConfigIndex[ueIdx];
               uint8_t srsBwIndex = pDynDescr->srsBwIndex[ueIdx];
               float W_SRS_LAST = d_SRS_BW_TABLE[63][0]*12.0f;
               if((srsConfigIndex<=63)&&(srsBwIndex<=3))
               {
                   W_SRS_LAST = d_SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
               }       
               
               pDynDescr->srsTxUe[cIdx*pDynDescr->nMaxActUePerCell + nSrsScheduledUe] = ueIdx;
               pDynDescr->srsLastTxCounter[ueIdx] = 0;
               pDynDescr->srsNumSymb[ueIdx] = 1;
               pDynDescr->srsTimeStart[ueIdx] = static_cast<uint8_t>(floor(nSrsScheduledUe/SRS_COMB_SIZE)); 
               pDynDescr->srsNumRep[ueIdx] = 1;
               pDynDescr->srsCombSize[ueIdx] = SRS_COMB_SIZE;
               pDynDescr->srsCombOffset[ueIdx] = static_cast<uint8_t>(floor(nSrsScheduledUe%SRS_COMB_SIZE)); 
               pDynDescr->srsSequenceId[ueIdx] = (40+cellIdx);
               pDynDescr->srsGroupOrSequenceHopping[ueIdx] = 0;
               pDynDescr->srsFreqHopping[ueIdx] = 0;
               pDynDescr->srsConfigIndex[ueIdx] = 63;
               pDynDescr->srsBwIndex[ueIdx] = 0;
               pDynDescr->srsFreqStart[ueIdx] = 0;  
               pDynDescr->srsFreqShift[ueIdx] = 0; 
               pDynDescr->srsCyclicShift[ueIdx] = 0;  
               
               multiCellSrsSchedulerTpcKernel(pDynDescr, ueIdx, W_SRS_LAST);  
               
               nSrsScheduledUe += 1;  
               
               if(nSrsScheduledUe == (nSrsScheduledMuMimoUe + nSrsScheduledSuMimoUe))
               {
                   break;
               }
               
                if(nSrsScheduledUe == pDynDescr->nMaxActUePerCell)
               {
                   return;
               }
               
           }
       }
   }
   __syncthreads();
   
   ////// Schedule UEs based on the age //////////////////////////////
   
   __shared__ int32_t srsAging[maxNumActUePerCell_];
   __shared__ uint16_t ueIds[maxNumActUePerCell_];

   for (int eIdx = threadIdx.x; eIdx < maxNumActUePerCell_; eIdx += blockDim.x) {
      srsAging[eIdx] = -100;
      ueIds[eIdx] = 0xFFFF;
   }

   __shared__ int nAssocUeFound;
   if (threadIdx.x == 0) {
      nAssocUeFound = 0;
   }
   __syncthreads();

   for (int uIdx = threadIdx.x; uIdx < pDynDescr->nActiveUe; uIdx += blockDim.x) 
   {
      if (pDynDescr->cellAssocActUe[cellIdx*pDynDescr->nActiveUe + uIdx]) 
      {
          if(pDynDescr->srsLastTxCounter[uIdx]>0)
          {
              int storeIdx = atomicAdd(&nAssocUeFound, 1);
              srsAging[storeIdx] = pDynDescr->srsLastTxCounter[uIdx];
              ueIds[storeIdx] = uIdx; 
          }
      }
   }
   __syncthreads();

   if (nAssocUeFound > 0) 
   {
      if(pDynDescr->nSrsScheduledUePerCell[cellIdx] > (nAssocUeFound + nSrsScheduledUe))
      {
          pDynDescr->nSrsScheduledUePerCell[cellIdx] = (nAssocUeFound + nSrsScheduledUe);
      }
      uint16_t pow2N = pow2NArr[nAssocUeFound-1];

      bitonicSort(srsAging, ueIds, pow2N); // internal synchronization
   }
   else
   {
       pDynDescr->nSrsScheduledUePerCell[cellIdx] = nSrsScheduledUe;
       return;
   }
   
   
   if (threadIdx.x < (pDynDescr->nSrsScheduledUePerCell[cellIdx]-nSrsScheduledUe)) 
   {    
      uint8_t srsConfigIndex = pDynDescr->srsConfigIndex[ueIds[threadIdx.x]];
      uint8_t srsBwIndex = pDynDescr->srsBwIndex[ueIds[threadIdx.x]];
      float W_SRS_LAST = d_SRS_BW_TABLE[63][0]*12.0f;
      if((srsConfigIndex<=63)&&(srsBwIndex<=3))
      {
          W_SRS_LAST = d_SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
      }
      
      pDynDescr->srsTxUe[cIdx*pDynDescr->nMaxActUePerCell + threadIdx.x + nSrsScheduledUe] = ueIds[threadIdx.x];
      pDynDescr->srsLastTxCounter[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsNumSymb[ueIds[threadIdx.x]] = 1;
      pDynDescr->srsTimeStart[ueIds[threadIdx.x]] = static_cast<uint8_t>(floorf((threadIdx.x+nSrsScheduledUe)/SRS_COMB_SIZE)); 
      pDynDescr->srsNumRep[ueIds[threadIdx.x]] = 1;
      pDynDescr->srsCombSize[ueIds[threadIdx.x]] = SRS_COMB_SIZE;
      pDynDescr->srsCombOffset[ueIds[threadIdx.x]] = static_cast<uint8_t>(floorf((threadIdx.x+nSrsScheduledUe)%SRS_COMB_SIZE)); 
      pDynDescr->srsSequenceId[ueIds[threadIdx.x]] = (40+cellIdx);
      pDynDescr->srsGroupOrSequenceHopping[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsFreqHopping[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsConfigIndex[ueIds[threadIdx.x]] = 63;
      pDynDescr->srsBwIndex[ueIds[threadIdx.x]] = 0;
      pDynDescr->srsFreqStart[ueIds[threadIdx.x]] = 0;  
      pDynDescr->srsFreqShift[ueIds[threadIdx.x]] = 0; 
      pDynDescr->srsCyclicShift[ueIds[threadIdx.x]] = 0; 
      
      multiCellSrsSchedulerTpcKernel(pDynDescr, ueIds[threadIdx.x], W_SRS_LAST);       
   }
}

void multiCellSrsScheduler::kernelSelect(uint8_t kernsl_version_sel)
{
    if(kernsl_version_sel==0)
    {
        void* kernelFunc = reinterpret_cast<void*>(multiCellSrsSchedulerKernel_v0);
        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));
    }
    else
    {
        void* kernelFunc = reinterpret_cast<void*>(multiCellSrsSchedulerKernel_v1);
        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));
    }
  
    // launch geometry
    gridDim  = {numThrdBlk, 1, 1};
    blockDim = {numThrdPerBlk, 1, 1};
  
    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver   = pLaunchCfg->kernelNodeParamsDriver;
  
    kernelNodeParamsDriver.blockDimX                  = blockDim.x;
    kernelNodeParamsDriver.blockDimY                  = blockDim.y;
    kernelNodeParamsDriver.blockDimZ                  = blockDim.z;
  
    kernelNodeParamsDriver.gridDimX                   = gridDim.x;
    kernelNodeParamsDriver.gridDimY                   = gridDim.y;
    kernelNodeParamsDriver.gridDimZ                   = gridDim.z;
  
    kernelNodeParamsDriver.extra                      = nullptr;
    kernelNodeParamsDriver.sharedMemBytes             = 0;    
}

void multiCellSrsScheduler::setup(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
                                  cumacSrsCellGrpPrms*        srsCellGrpPrms,
                                  cumacSchdSol*               schdSol, 
                                  cumacSrsSchdSol*            srsSchdSol,
                                  cudaStream_t                strm)
{
    pCpuDynDesc->nMaxActUePerCell               = srsCellGrpPrms->nMaxActUePerCell;
    pCpuDynDesc->nActiveUe                      = srsCellGrpPrms->nActiveUe;
    pCpuDynDesc->nCell                          = srsCellGrpPrms->nCell;
    pCpuDynDesc->nSymbsPerSlot                  = srsCellGrpPrms->nSymbsPerSlot;
    pCpuDynDesc->cellId                         = srsCellGrpPrms->cellId;
    pCpuDynDesc->cellAssocActUe                 = srsCellGrpPrms->cellAssocActUe;
    pCpuDynDesc->newDataActUe                   = srsCellGrpUeStatus->newDataActUe;
    pCpuDynDesc->srsLastTxCounter               = srsCellGrpUeStatus->srsLastTxCounter;
    pCpuDynDesc->srsNumAntPorts                 = srsCellGrpUeStatus->srsNumAntPorts;
    pCpuDynDesc->srsResourceType                = srsCellGrpUeStatus->srsResourceType;
    pCpuDynDesc->srsWbSnr                       = srsCellGrpUeStatus->srsWbSnr;
    pCpuDynDesc->srsWbSnrThreshold              = srsCellGrpUeStatus->srsWbSnrThreshold;
    pCpuDynDesc->srsWidebandSignalEnergy        = srsCellGrpUeStatus->srsWidebandSignalEnergy;
    pCpuDynDesc->srsTxPwrMax                    = srsCellGrpUeStatus->srsTxPwrMax;
    pCpuDynDesc->srsPwr0                        = srsCellGrpUeStatus->srsPwr0;
    pCpuDynDesc->srsPwrAlpha                    = srsCellGrpUeStatus->srsPwrAlpha;
    pCpuDynDesc->srsTpcAccumulationFlag         = srsCellGrpUeStatus->srsTpcAccumulationFlag;
    pCpuDynDesc->srsPowerControlAdjustmentState = srsCellGrpUeStatus->srsPowerControlAdjustmentState;  
    pCpuDynDesc->srsPowerHeadroomReport         = srsCellGrpUeStatus->srsPowerHeadroomReport;    
    if((srsCellGrpPrms->nBsAnt==64)&&(schdSol!= nullptr))
    {
        pCpuDynDesc->muMimoInd              = schdSol->muMimoInd;
        pCpuDynDesc->sortedUeList           = schdSol->sortedUeList;
    }
    pCpuDynDesc->nSrsScheduledUePerCell     = srsSchdSol->nSrsScheduledUePerCell;
    pCpuDynDesc->srsTxUe                    = srsSchdSol->srsTxUe;
    pCpuDynDesc->srsRxCell                  = srsSchdSol->srsRxCell;
    pCpuDynDesc->srsNumSymb                 = srsSchdSol->srsNumSymb;
    pCpuDynDesc->srsTimeStart               = srsSchdSol->srsTimeStart;
    pCpuDynDesc->srsNumRep                  = srsSchdSol->srsNumRep;
    pCpuDynDesc->srsConfigIndex             = srsSchdSol->srsConfigIndex;
    pCpuDynDesc->srsBwIndex                 = srsSchdSol->srsBwIndex;
    pCpuDynDesc->srsCombSize                = srsSchdSol->srsCombSize;
    pCpuDynDesc->srsCombOffset              = srsSchdSol->srsCombOffset;
    pCpuDynDesc->srsFreqStart               = srsSchdSol->srsFreqStart;
    pCpuDynDesc->srsFreqShift               = srsSchdSol->srsFreqShift;
    pCpuDynDesc->srsFreqHopping             = srsSchdSol->srsFreqHopping;
    pCpuDynDesc->srsSequenceId              = srsSchdSol->srsSequenceId;
    pCpuDynDesc->srsGroupOrSequenceHopping  = srsSchdSol->srsGroupOrSequenceHopping;
    pCpuDynDesc->srsCyclicShift             = srsSchdSol->srsCyclicShift;
    pCpuDynDesc->srsTxPwr                   = srsSchdSol->srsTxPwr;

    numThrdPerBlk = 1024;
    numThrdBlk    = srsCellGrpPrms->nCell;

    CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, (void*)pCpuDynDesc.get(), sizeof(mcSrsSchedulerDynDescr_t), cudaMemcpyHostToDevice, strm));
    CUDA_CHECK_ERR(cudaMemcpyToSymbolAsync(d_SRS_BW_TABLE, SRS_BW_TABLE, sizeof(SRS_BW_TABLE), 0, cudaMemcpyHostToDevice, strm));

    // select kernel 
    uint8_t kernel_version_sel = 0;
    if((srsCellGrpPrms->nBsAnt==64)&&(srsCellGrpPrms->srsSchedulingSel==1))
    {
        assert(schdSol!= nullptr);
        kernel_version_sel = 1;
    }
    kernelSelect(kernel_version_sel);
 
    pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void multiCellSrsScheduler::run(cudaStream_t strm)
{
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
    cudaEvent_t start, stop;
    CUDA_CHECK_ERR(cudaEventCreate(&start));
    CUDA_CHECK_ERR(cudaEventCreate(&stop));
    float milliseconds = 0;
    CUDA_CHECK_ERR(cudaEventRecord(start));
    for (int exeIdx = 0; exeIdx < numRunSchKnlTimeMsr; exeIdx++) {
    #endif

    CUDA_CHECK_RES(cuLaunchKernel(kernelNodeParamsDriver.func,
                                  kernelNodeParamsDriver.gridDimX,
                                  kernelNodeParamsDriver.gridDimY, 
                                  kernelNodeParamsDriver.gridDimZ,
                                  kernelNodeParamsDriver.blockDimX, 
                                  kernelNodeParamsDriver.blockDimY, 
                                  kernelNodeParamsDriver.blockDimZ,
                                  kernelNodeParamsDriver.sharedMemBytes,
                                  strm,
                                  kernelNodeParamsDriver.kernelParams,
                                  kernelNodeParamsDriver.extra));   
    #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
    }
    CUDA_CHECK_ERR(cudaEventRecord(stop));
    CUDA_CHECK_ERR(cudaEventSynchronize(stop));
    CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Multi-cell SRS scheduling ext time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
    #endif 
}

void multiCellSrsScheduler::cpuSrsSchedulerTpc(cumacSrsCellGrpUeStatus* srsCellGrpUeStatus, cumacSrsSchdSol* srsSchdSol, uint16_t ueIdx, float W_SRS_LAST)
{
    float       srsTxPwrMax                    = srsCellGrpUeStatus->srsTxPwrMax[ueIdx];
    float       srsPwr0                        = srsCellGrpUeStatus->srsPwr0[ueIdx];
    float       srsPwrAlpha                    = srsCellGrpUeStatus->srsPwrAlpha[ueIdx];
    uint8_t     srsTpcAccumulationFlag         = srsCellGrpUeStatus->srsTpcAccumulationFlag[ueIdx];
    float       srsPowerControlAdjustmentState = srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx]; 
    
    uint8_t     srsConfigIndex                 = srsSchdSol->srsConfigIndex[ueIdx];
    uint8_t     srsBwIndex                     = srsSchdSol->srsBwIndex[ueIdx];
    uint8_t     srsCombSize                    = srsSchdSol->srsCombSize[ueIdx];
    
    float       srsWbSnr                       = srsCellGrpUeStatus->srsWbSnr[ueIdx];
    float       srsWbSnrThreshold              = srsCellGrpUeStatus->srsWbSnrThreshold[ueIdx];
    
    float       srsTxPwr                       = srsSchdSol->srsTxPwr[ueIdx];
    float       srsWidebandSignalEnergy        = std::max(srsCellGrpUeStatus->srsWidebandSignalEnergy[ueIdx], 1e-9f);
     
    float       pathLossEst                    = srsTxPwr - 10.0f*log10f(srsWidebandSignalEnergy/W_SRS_LAST); 
    float       M_SRS                          = SRS_BW_TABLE[63][0]*12.0f/srsCombSize;
    
    if((srsConfigIndex<=63)&&(srsBwIndex<=3))
    {
        M_SRS = SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f/srsCombSize;
    }
    
    if(srsTpcAccumulationFlag==1) //TPC commands via accumulation, large dynamic range for power control
    {
        float srsTxPwrGap = srsTxPwrMax - srsPwr0 - 10.0f*log10f(2.0f*M_SRS) - srsPwrAlpha * pathLossEst - srsPowerControlAdjustmentState;
        if(srsTxPwrGap<=0.0f)
        {
            srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = (srsPowerControlAdjustmentState + std::floor(srsTxPwrGap));
        }
        else
        {
            float srsWbSnrGap = srsWbSnrThreshold - srsWbSnr;
            float srsTxPwrDelta = std::min(std::ceil(srsWbSnrGap), srsTxPwrGap);
            srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = (srsPowerControlAdjustmentState + std::floor(srsTxPwrDelta));
        }
        srsSchdSol->srsTxPwr[ueIdx] = std::min(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx]); 
    }
    else if(srsTpcAccumulationFlag==0) //TPC commands via without accumulation, small dynamic range for power control
    {
        float srsTxPwrGap = srsTxPwrMax - srsPwr0 - 10.0f*log10f(2.0f*M_SRS) - srsPwrAlpha * pathLossEst;
        if(srsTxPwrGap<=0.0f)
        {
            if(srsTxPwrGap>=-1.0f)
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = -1.0f;
                srsSchdSol->srsTxPwr[ueIdx] = std::min(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx]); 
            }
            else if(srsTxPwrGap>=-4.0f)
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = -4.0f;
                srsSchdSol->srsTxPwr[ueIdx] = std::min(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx]); 
            }
            else
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = -4.0f;
                srsSchdSol->srsTxPwr[ueIdx] = srsCellGrpUeStatus->srsTxPwrMax[ueIdx];
            }
        }
        else
        {
            float srsWbSnrGap = srsWbSnrThreshold - (srsWbSnr-srsPowerControlAdjustmentState);
            float srsTxPwrDelta = std::min(std::ceil(srsWbSnrGap), srsTxPwrGap);
            if(srsTxPwrDelta>=4.0f)
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = 4.0f;
            }
            else if(srsTxPwrDelta>=1.0f)
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = 1.0f;
            }
            else if(srsTxPwrDelta>=-1.0f)
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = -1.0f;
            }
            else
            {
                srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx] = -4.0f;
            }
            srsSchdSol->srsTxPwr[ueIdx] = std::min(srsTxPwrMax, srsPwr0 + 10.0f*log10f(2.0f*M_SRS) + srsPwrAlpha * pathLossEst + srsCellGrpUeStatus->srsPowerControlAdjustmentState[ueIdx]); 
        }
    }
}

void multiCellSrsScheduler::cpuScheduler_v0(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
                                                   cumacSrsCellGrpPrms*        srsCellGrpPrms,
                                                   cumacSrsSchdSol*            srsSchdSol)
{
       uint16_t    nMaxActUePerCell = srsCellGrpPrms->nMaxActUePerCell; 
       uint16_t    nActiveUe        = srsCellGrpPrms->nActiveUe; 
       uint16_t    nCell            = srsCellGrpPrms->nCell; 
       uint16_t    nSymbsPerSlot    = srsCellGrpPrms->nSymbsPerSlot;
       
       for(uint16_t cIdx = 0; cIdx < nCell; cIdx++)
       {
           srsSchdSol->nSrsScheduledUePerCell[cIdx] = nSymbsPerSlot*SRS_COMB_SIZE; 
           
           int nAssocUeFound = 0;
   
           for (int uIdx = 0; uIdx < nActiveUe; uIdx++) 
           {
              if (srsCellGrpPrms->cellAssocActUe[cIdx*nActiveUe + uIdx]) 
              { 
                 ///// update srsLastTxCounter
                 srsCellGrpUeStatus->srsLastTxCounter[uIdx]+=1;
                 ////  update RX cell
                 srsSchdSol->srsRxCell[uIdx]=cIdx;
                 
                 nAssocUeFound++;
              }
           }
           
           if(nAssocUeFound>0)
           {
               if(srsSchdSol->nSrsScheduledUePerCell[cIdx] > nAssocUeFound)
                   srsSchdSol->nSrsScheduledUePerCell[cIdx] = nAssocUeFound;
           }
           else 
           {
               srsSchdSol->nSrsScheduledUePerCell[cIdx] = 0;
               continue;
           }
           
           ////// Schedule UEs based on the age //////////////////////////////
           std::vector<std::pair<uint32_t, uint16_t>> srsAging;
           for(uint16_t uIdx = cIdx*nMaxActUePerCell; uIdx < (cIdx+1)*nMaxActUePerCell; uIdx++)
           {
               srsAging.push_back({srsCellGrpUeStatus->srsLastTxCounter[uIdx], uIdx});
           }
           
           
           std::sort(srsAging.begin(), srsAging.end(), [](std::pair<uint32_t, uint16_t> a, std::pair<uint32_t, uint16_t> b)
           {
               return (a.first > b.first) || (a.first == b.first && a.second < b.second);
           });
           
           uint16_t count=0;
           for(uint16_t uIdx = 0; uIdx < (srsSchdSol->nSrsScheduledUePerCell[cIdx]); uIdx++)
           {
               uint16_t ueIdx = srsAging[uIdx].second;
               if(srsCellGrpUeStatus->srsLastTxCounter[ueIdx]>0)
               {
                   uint8_t srsConfigIndex = srsSchdSol->srsConfigIndex[ueIdx];
                   uint8_t srsBwIndex = srsSchdSol->srsBwIndex[ueIdx];
                   float W_SRS_LAST = SRS_BW_TABLE[63][0]*12.0f;
                   if((srsConfigIndex<=63)&&(srsBwIndex<=3))
                   {
                       W_SRS_LAST = SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
                   }
                   
                   srsSchdSol->srsTxUe[cIdx*nMaxActUePerCell + uIdx] = ueIdx;
                   srsCellGrpUeStatus->srsLastTxCounter[ueIdx] = 0;
                   srsSchdSol->srsNumSymb[ueIdx] = 1;
                   srsSchdSol->srsTimeStart[ueIdx] = static_cast<uint8_t>(std::floor(uIdx/SRS_COMB_SIZE)); 
                   srsSchdSol->srsNumRep[ueIdx] = 1;
                   srsSchdSol->srsCombSize[ueIdx] = SRS_COMB_SIZE;
                   srsSchdSol->srsCombOffset[ueIdx] = static_cast<uint8_t>(std::floor(uIdx%SRS_COMB_SIZE)); 
                   srsSchdSol->srsSequenceId[ueIdx] = (40+cIdx);
                   srsSchdSol->srsGroupOrSequenceHopping[ueIdx] = 0;
                   srsSchdSol->srsFreqHopping[ueIdx] = 0;
                   srsSchdSol->srsConfigIndex[ueIdx] = 63;
                   srsSchdSol->srsBwIndex[ueIdx] = 0;
                   srsSchdSol->srsFreqStart[ueIdx] = 0;  
                   srsSchdSol->srsFreqShift[ueIdx] = 0; 
                   srsSchdSol->srsCyclicShift[ueIdx] = 0; 
                   
                   cpuSrsSchedulerTpc(srsCellGrpUeStatus, srsSchdSol, ueIdx, W_SRS_LAST);
                   
                   count += 1;
               }
           }
           
           srsSchdSol->nSrsScheduledUePerCell[cIdx] = count;
            
       } // cell loop
} // cpuScheduler_v0

void multiCellSrsScheduler::cpuScheduler_v1(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
                                            cumacSrsCellGrpPrms*        srsCellGrpPrms,
                                            cumacSchdSol*               schdSol, 
                                            cumacSrsSchdSol*            srsSchdSol)
{      
       uint16_t    nMaxActUePerCell = srsCellGrpPrms->nMaxActUePerCell; 
       uint16_t    nActiveUe        = srsCellGrpPrms->nActiveUe; 
       uint16_t    nCell            = srsCellGrpPrms->nCell; 
       uint16_t    nSymbsPerSlot    = srsCellGrpPrms->nSymbsPerSlot;
       
       for(uint16_t cIdx = 0; cIdx < nCell; cIdx++)
       {
           srsSchdSol->nSrsScheduledUePerCell[cIdx] = nSymbsPerSlot*SRS_COMB_SIZE; 
           
           uint16_t nSrsScheduledMuMimoUe = (uint16_t)std::floor(srsSchdSol->nSrsScheduledUePerCell[cIdx]/3.0);
           uint16_t nSrsScheduledSuMimoUe = (uint16_t)std::floor(srsSchdSol->nSrsScheduledUePerCell[cIdx]/3.0);
   
           uint16_t nSrsScheduledUe = 0;
           
           int nAssocUeFound = 0;
   
           for (int uIdx = 0; uIdx < nActiveUe; uIdx++) 
           {
              if (srsCellGrpPrms->cellAssocActUe[cIdx*nActiveUe + uIdx]) 
              { 
                 ///// update srsLastTxCounter
                 srsCellGrpUeStatus->srsLastTxCounter[uIdx]+=1;
                 ////  update RX cell
                 srsSchdSol->srsRxCell[uIdx]=cIdx;
                 
                 nAssocUeFound++;
              }
           }
           
           if(nAssocUeFound>0)
           {
               if(srsSchdSol->nSrsScheduledUePerCell[cIdx] > nAssocUeFound)
                   srsSchdSol->nSrsScheduledUePerCell[cIdx] = nAssocUeFound;
           }
           else 
           {
               srsSchdSol->nSrsScheduledUePerCell[cIdx] = 0;
               continue;
           }
           
           //// schedule SRS for muMIMO UEs
           for(uint16_t idx = 0; idx < nMaxActUePerCell; idx++)
           {
               uint16_t ueIdx = schdSol->sortedUeList[cIdx][idx];
               if((srsCellGrpUeStatus->srsLastTxCounter[ueIdx]>0)&&(schdSol->muMimoInd[ueIdx]==1))
               { 
                   uint8_t srsConfigIndex = srsSchdSol->srsConfigIndex[ueIdx];
                   uint8_t srsBwIndex = srsSchdSol->srsBwIndex[ueIdx];
                   float W_SRS_LAST = SRS_BW_TABLE[63][0]*12.0f;
                   if((srsConfigIndex<=63)&&(srsBwIndex<=3))
                   {
                       W_SRS_LAST = SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
                   }
                     
                   srsSchdSol->srsTxUe[cIdx*nMaxActUePerCell + nSrsScheduledUe] = ueIdx;
                   srsCellGrpUeStatus->srsLastTxCounter[ueIdx] = 0;
                   srsSchdSol->srsNumSymb[ueIdx] = 1;
                   srsSchdSol->srsTimeStart[ueIdx] = static_cast<uint8_t>(std::floor(nSrsScheduledUe/SRS_COMB_SIZE)); 
                   srsSchdSol->srsNumRep[ueIdx] = 1;
                   srsSchdSol->srsCombSize[ueIdx] = SRS_COMB_SIZE;
                   srsSchdSol->srsCombOffset[ueIdx] = static_cast<uint8_t>(std::floor(nSrsScheduledUe%SRS_COMB_SIZE)); 
                   srsSchdSol->srsSequenceId[ueIdx] = (40+cIdx);
                   srsSchdSol->srsGroupOrSequenceHopping[ueIdx] = 0;
                   srsSchdSol->srsFreqHopping[ueIdx] = 0;
                   srsSchdSol->srsConfigIndex[ueIdx] = 63;
                   srsSchdSol->srsBwIndex[ueIdx] = 0;
                   srsSchdSol->srsFreqStart[ueIdx] = 0;  
                   srsSchdSol->srsFreqShift[ueIdx] = 0; 
                   srsSchdSol->srsCyclicShift[ueIdx] = 0;  
                   
                   cpuSrsSchedulerTpc(srsCellGrpUeStatus, srsSchdSol, ueIdx, W_SRS_LAST);
                   
                   nSrsScheduledUe += 1;  
                   
                   if(nSrsScheduledUe == nSrsScheduledMuMimoUe)
                   {
                       break;
                   }
                   
                   if(nSrsScheduledUe == nMaxActUePerCell)
                   {
                       continue;
                   }
               }
           }

           //// schedule SRS for suMIMO UEs
           for(uint16_t idx = 0; idx < nMaxActUePerCell; idx++)
           {
               uint16_t ueIdx = schdSol->sortedUeList[cIdx][idx];
               if((srsCellGrpUeStatus->srsLastTxCounter[ueIdx]>0)&&(schdSol->muMimoInd[ueIdx]==0))
               {
                   uint8_t srsConfigIndex = srsSchdSol->srsConfigIndex[ueIdx];
                   uint8_t srsBwIndex = srsSchdSol->srsBwIndex[ueIdx];
                   float W_SRS_LAST = SRS_BW_TABLE[63][0]*12.0f;
                   if((srsConfigIndex<=63)&&(srsBwIndex<=3))
                   {
                       W_SRS_LAST = SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
                   }
                   
                   srsSchdSol->srsTxUe[cIdx*nMaxActUePerCell + nSrsScheduledUe] = ueIdx;
                   srsCellGrpUeStatus->srsLastTxCounter[ueIdx] = 0;
                   srsSchdSol->srsNumSymb[ueIdx] = 1;
                   srsSchdSol->srsTimeStart[ueIdx] = static_cast<uint8_t>(std::floor(nSrsScheduledUe/SRS_COMB_SIZE)); 
                   srsSchdSol->srsNumRep[ueIdx] = 1;
                   srsSchdSol->srsCombSize[ueIdx] = SRS_COMB_SIZE;
                   srsSchdSol->srsCombOffset[ueIdx] = static_cast<uint8_t>(std::floor(nSrsScheduledUe%SRS_COMB_SIZE)); 
                   srsSchdSol->srsSequenceId[ueIdx] = (40+cIdx);
                   srsSchdSol->srsGroupOrSequenceHopping[ueIdx] = 0;
                   srsSchdSol->srsFreqHopping[ueIdx] = 0;
                   srsSchdSol->srsConfigIndex[ueIdx] = 63;
                   srsSchdSol->srsBwIndex[ueIdx] = 0;
                   srsSchdSol->srsFreqStart[ueIdx] = 0;  
                   srsSchdSol->srsFreqShift[ueIdx] = 0; 
                   srsSchdSol->srsCyclicShift[ueIdx] = 0;  
                   
                   cpuSrsSchedulerTpc(srsCellGrpUeStatus, srsSchdSol, ueIdx, W_SRS_LAST);
                   
                   nSrsScheduledUe += 1;  
                   
                   if(nSrsScheduledUe == (nSrsScheduledMuMimoUe + nSrsScheduledSuMimoUe))
                   {
                       break;
                   }
                   
                   if(nSrsScheduledUe == nMaxActUePerCell)
                   {
                       continue;
                   }
                   
               }
           }
           
           ////// Schedule UEs based on the age //////////////////////////////
           std::vector<std::pair<uint32_t, uint16_t>> srsAging;
           for(uint16_t uIdx = cIdx*nMaxActUePerCell; uIdx < (cIdx+1)*nMaxActUePerCell; uIdx++)
           {
               srsAging.push_back({srsCellGrpUeStatus->srsLastTxCounter[uIdx], uIdx});
           }
           
           
           std::sort(srsAging.begin(), srsAging.end(), [](std::pair<uint32_t, uint16_t> a, std::pair<uint32_t, uint16_t> b)
           {
               return (a.first > b.first) || (a.first == b.first && a.second < b.second);
           });
           
           uint16_t count=0;
           for(uint16_t uIdx = 0; uIdx < (srsSchdSol->nSrsScheduledUePerCell[cIdx]-nSrsScheduledUe); uIdx++)
           {
               uint16_t ueIdx = srsAging[uIdx].second;
               if(srsCellGrpUeStatus->srsLastTxCounter[ueIdx]>0)
               {
                   uint8_t srsConfigIndex = srsSchdSol->srsConfigIndex[ueIdx];
                   uint8_t srsBwIndex = srsSchdSol->srsBwIndex[ueIdx];
                   float W_SRS_LAST = SRS_BW_TABLE[63][0]*12.0f;
                   if((srsConfigIndex<=63)&&(srsBwIndex<=3))
                   {
                       W_SRS_LAST = SRS_BW_TABLE[srsConfigIndex][2*srsBwIndex]*12.0f;
                   }
                   
                   srsSchdSol->srsTxUe[cIdx*nMaxActUePerCell + uIdx + nSrsScheduledUe] = ueIdx;
                   srsCellGrpUeStatus->srsLastTxCounter[ueIdx] = 0;
                   srsSchdSol->srsNumSymb[ueIdx] = 1;
                   srsSchdSol->srsTimeStart[ueIdx] = static_cast<uint8_t>(std::floor((nSrsScheduledUe+uIdx)/SRS_COMB_SIZE)); 
                   srsSchdSol->srsNumRep[ueIdx] = 1;
                   srsSchdSol->srsCombSize[ueIdx] = SRS_COMB_SIZE;
                   srsSchdSol->srsCombOffset[ueIdx] = static_cast<uint8_t>(std::floor((nSrsScheduledUe+uIdx)%SRS_COMB_SIZE)); 
                   srsSchdSol->srsSequenceId[ueIdx] = (40+cIdx);
                   srsSchdSol->srsGroupOrSequenceHopping[ueIdx] = 0;
                   srsSchdSol->srsFreqHopping[ueIdx] = 0;
                   srsSchdSol->srsConfigIndex[ueIdx] = 63;
                   srsSchdSol->srsBwIndex[ueIdx] = 0;
                   srsSchdSol->srsFreqStart[ueIdx] = 0;  
                   srsSchdSol->srsFreqShift[ueIdx] = 0; 
                   srsSchdSol->srsCyclicShift[ueIdx] = 0; 
                   
                   cpuSrsSchedulerTpc(srsCellGrpUeStatus, srsSchdSol, ueIdx, W_SRS_LAST);
                   
                   count += 1;
               }
           }
           
           srsSchdSol->nSrsScheduledUePerCell[cIdx] = nSrsScheduledUe + count;
            
       } // cell loop
} // cpuScheduler_v1

void multiCellSrsScheduler::debugLog()
{

}
}