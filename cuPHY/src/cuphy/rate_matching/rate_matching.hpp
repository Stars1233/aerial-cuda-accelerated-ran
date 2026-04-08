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

#include <functional>
#include "cuphy.h"

constexpr float    LLR_MAX_ABS_VALUE = 10000.0f;
constexpr uint32_t DERM_BLK_DIM      = 96;  // block dimension for de-rate-matching kernels

struct puschRxRateMatchDescr
{
    const void*        llr_vec_in[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED]; // Rm input LLRs    //ToDo change to void**
    uint16_t           schUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
    void**             out;        // Rm output LLRs
    const PerTbParams* tbPrmsArray;
    int                descramblingOn;

};
typedef struct puschRxRateMatchDescr puschRxRateMatchDescr_t;


typedef struct _puschRxRateMatchLaunchGeo
{
    dim3      gridDim;
    dim3      blockDim;
    uint32_t  shMemBytes;
}puschRxRateMatchLaunchGeo_t;


struct puschRxRateMatchLaunchPrms;
using puschRxRateMatchKernelLauncher_t = std::function<void(puschRxRateMatchLaunchPrms&, cudaStream_t&)>;


struct puschRxRateMatchLaunchPrms
{
    puschRxRateMatchKernelLauncher_t launcher;
    puschRxRateMatchDescr_t*         args;
    puschRxRateMatchLaunchGeo_t      geo;

    // Graph
    void* kernelArgs;
    void* kernelFunc;
};
typedef struct puschRxRateMatchLaunchPrms puschRxRateMatchLaunchPrms_t;


class puschRxRateMatch : public cuphyPuschRxRateMatch {
public:
    puschRxRateMatch()                                   = default;
    ~puschRxRateMatch()                                  = default;
    puschRxRateMatch(puschRxRateMatch const&)            = delete;
    puschRxRateMatch& operator=(puschRxRateMatch const&) = delete;

    static void getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes);

    void init(int   rmFPconfig,          // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
              int   descramblingOn);   // enable/disable descrambling


    void setup( uint16_t                          nSchUes,                      // number of users with sch data
                uint16_t*                         pSchUserIdxsCpu,              // indicies of users with SCH data
                const PerTbParams*                pTbPrmsCpu,                   // starting adress of transport block paramters (CPU)
                const PerTbParams*                pTbPrmsGpu,                   // starting adress of transport block paramters (GPU)
                cuphyTensorPrm_t*                 pTPrmRmIn,                    // starting adress of input LLR tensor parameters
                cuphyTensorPrm_t*                 pTPrmCdm1RmIn, 
                void**                            ppRmOut,                      // array of rm outputs (GPU)
                void*                             pCpuDesc,                     // pointer to descriptor in cpu
                void*                             pGpuDesc,                     // pointer to descriptor in gpu
                uint8_t                           enableCpuToGpuDescrAsyncCpy,  // option to copy cpu descriptors from cpu to gpu
                cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,                   // pointer to rate matching launch configuration
                cudaStream_t                      strm);                        // stream to perform copy

private:
    // class state modifed by setup saved in data member.
    CUfunction m_kernelFunc;
    CUfunction m_resetBufferKernelFunc;  // Reset buffer kernel dispatcher
    CUfunction m_clampBufferKernelFunc;  // Clamp buffer kernel dispatcher
    int        m_descramblingOn;
    int        m_rmFPconfig{};  // Store FP configuration for kernel selection
};
