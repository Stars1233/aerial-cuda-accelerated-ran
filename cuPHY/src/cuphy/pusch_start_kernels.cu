/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuphy.h"
#include "cuphy_api.h"
#include "common_utils.hpp"
#include "ch_est/ch_est.hpp"

#include "pusch_start_kernels.hpp"

namespace
{
__device__ __forceinline__ unsigned long long get_ptimer_ns()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}
}

namespace pusch {

// kernel to wait until required number of symbol bits are ready
template <bool SUB_SLOT>
static __global__ void
symbolWaitKernel(ch_est::puschRxChEstStatDescr_t* pStatDescr, ch_est::puschRxChEstDynDescr_t* pDynDescr)
{
    if(threadIdx.x==0)
    {
        bool     bitsReady  = false;
        bool     timedOut   = false;
        uint64_t startTime, maxWaitTime;
        uint32_t nSymbols;
        if (SUB_SLOT)
        {
            startTime   = get_ptimer_ns();
            maxWaitTime = startTime + static_cast<uint64_t>(pDynDescr->waitTimeOutPreEarlyHarqUs) * 1000;
            nSymbols    = (uint32_t)(pDynDescr->nSymPreSubSlotWaitKernel);
            pDynDescr->mPuschStartTimeNs = startTime; // record the start time to be used in full-slot (i.e. SUB_SLOT == false)
        }
        else
        {
            startTime   = pDynDescr->mPuschStartTimeNs; // note that this is being initialized to 0, but updated if sub-slot is enabled
            // do not wait for full-slot symbols if sub-slot mode is disabled
            if (startTime == 0)
            {
                bitsReady = true;
            }
            else
            {
                maxWaitTime = startTime + static_cast<uint64_t>(pDynDescr->waitTimeOutPostEarlyHarqUs) * 1000;
                nSymbols    = (uint32_t)(pDynDescr->nSymPostSubSlotWaitKernel);
            }
        }

        while (!bitsReady && !timedOut)
        {
            bitsReady = true;
            for(int i = 0; i < nSymbols; i++)
            {
                volatile uint32_t* pstat = (volatile uint32_t*)&(pStatDescr->pSymbolRxStatus[i]);
                if (*pstat != SYM_RX_DONE)
                {
                    bitsReady = false;
                    break;
                }
            }
            timedOut = get_ptimer_ns() > maxWaitTime;   // to ensure the kernel won't get stuck in while loop
        }

        uint8_t* pTimeOutFlag = SUB_SLOT? pDynDescr->pPreSubSlotWaitKernelStatusGpu  : pDynDescr->pPostSubSlotWaitKernelStatusGpu;
        if (pTimeOutFlag)
        {
            *pTimeOutFlag = (timedOut && !bitsReady) ? PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT : PUSCH_RX_WAIT_KERNEL_STATUS_DONE;
        }
    }
}

template <bool ENABLE_DEVICE_GRAPH_LAUNCH, bool SUB_SLOT>
static __global__ void
deviceGraphLaunchKernel(ch_est::puschRxChEstDynDescr_t* pDynDescr, CUgraphExec deviceGraphExec)
{
    if(ENABLE_DEVICE_GRAPH_LAUNCH && threadIdx.x==0)
    {
        bool timedOut = false;
        if (SUB_SLOT)
        {
            if(pDynDescr->pPreSubSlotWaitKernelStatusGpu)
            {
                timedOut = *(pDynDescr->pPreSubSlotWaitKernelStatusGpu) == PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT;
            }
        }
        else
        {
            if(pDynDescr->pPostSubSlotWaitKernelStatusGpu)
            {
                timedOut = *(pDynDescr->pPostSubSlotWaitKernelStatusGpu) == PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT;
            }
        }
        
        if(!timedOut)
        {
            cudaGraphLaunch(deviceGraphExec, cudaStreamGraphFireAndForget);
        }
    }
}

template <typename CFG, typename GetFunc>
static inline void setKernelParamsCommon(CFG& launchCfg,
                                         void* kernelArgs0,
                                         void* kernelArgs1,
                                         GetFunc&& getFunc)
{
    dim3 gridDims(1);
    dim3 blockDims(32);

    CUDA_KERNEL_NODE_PARAMS& p = launchCfg.kernelNodeParamsDriver;
    { MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&p.func, getFunc())); }

    p.blockDimX = blockDims.x; p.blockDimY = blockDims.y; p.blockDimZ = blockDims.z;
    p.gridDimX  = gridDims.x;  p.gridDimY  = gridDims.y;  p.gridDimZ  = gridDims.z;
    p.extra = nullptr; p.sharedMemBytes = 0;

    void* placeholderPtr = nullptr;
    launchCfg.kernelArgs[0] = kernelArgs0 ? kernelArgs0 : &placeholderPtr;
    launchCfg.kernelArgs[1] = kernelArgs1 ? kernelArgs1 : &placeholderPtr;
    p.kernelParams = launchCfg.kernelArgs;
}

template <typename CFG, typename FUNC>
void setKernelParamsImpl(CFG& launchCfg, const uint8_t puschRxFullSlotMode, void* kernelArgs0, void* kernelArgs1, FUNC&& kernelFunc)
{
    setKernelParamsCommon(launchCfg, kernelArgs0, kernelArgs1, [&]{ return kernelFunc(puschRxFullSlotMode); });
}

template <typename CFG, typename FUNC>
void setDGLKernelParamsImpl(CFG& launchCfg, const uint8_t enableDeviceGraphLaunch, const uint8_t puschRxFullSlotMode, void* kernelArgs0, void* kernelArgs1, FUNC&& kernelFunc)
{
    setKernelParamsCommon(launchCfg, kernelArgs0, kernelArgs1, [&]{ return kernelFunc(enableDeviceGraphLaunch, puschRxFullSlotMode); });

}

// setup for symbolWaitKernel, where it waits for certain number of symbols to arrive before proceeding to start the PUSCH pipeline
void StartKernels::setWaitKernelParams(cuphyPuschRxWaitLaunchCfg_t* pLaunchCfg,
                                       const uint8_t                puschRxFullSlotMode,
                                       void*                        ppStatDescr,
                                       void*                        ppDynDescr)
{
    setKernelParamsImpl(*pLaunchCfg,
                        puschRxFullSlotMode,
                        ppStatDescr,
                        ppDynDescr,
                        [](const auto puschRxFullSlotMode) -> void* {
                            const auto ret = (puschRxFullSlotMode == CUPHY_PUSCH_FULL_SLOT_PATH ?
                                                  symbolWaitKernel<false> :
                                                  symbolWaitKernel<true>);
                            return reinterpret_cast<void*>(ret);
                        });
}

// used in kernel node that launches full-slot or sub-slot PUSCH graph from device
void StartKernels::setDeviceGraphLaunchKernelParams(cuphyPuschRxDglLaunchCfg_t* pLaunchCfg,
                                                    const uint8_t               enableDeviceGraphLaunch,
                                                    const uint8_t               puschRxFullSlotMode,
                                                    void*                       ppDynDescr,
                                                    void*                       ppDeviceGraph)
{
    setDGLKernelParamsImpl(*pLaunchCfg,
                           enableDeviceGraphLaunch,
                           puschRxFullSlotMode,
                           ppDynDescr,
                           ppDeviceGraph,
                           [](const auto enableDeviceGraphLaunch, const auto puschRxFullSlotMode) -> void* {
                               const auto ret = enableDeviceGraphLaunch ? 
                                              (puschRxFullSlotMode == CUPHY_PUSCH_FULL_SLOT_PATH ? deviceGraphLaunchKernel<true, false>  : deviceGraphLaunchKernel<true, true>) : 
                                              (puschRxFullSlotMode == CUPHY_PUSCH_FULL_SLOT_PATH ? deviceGraphLaunchKernel<false, false> : deviceGraphLaunchKernel<false, true>);
                               return reinterpret_cast<void*>(ret);
                        });
}
} // namespace pusch
