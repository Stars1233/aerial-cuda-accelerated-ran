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

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include "cuphy_kernel_util.cuh"
#include "simplex_decoder.hpp"

namespace cg = cooperative_groups;

const int SIMPLEX_DECODER_TPB = 32;

// The use of cuda::memcpy_async, cuda::pipeline, and cuda::barrier should be disabled for A100 GPU 
// because under the Ampere architecture, there's a potential issue with the wait/arrive logic of CUDA barriers, 
// which will cause errors/CICD failures when testing with the compute-sanitizer synccheck tool.
// For more info refer to NVBUG-4114257 and JIRA BUG DTCS-1070
// To re-enable the use of cuda::memcpy_async and cuda::pipeline for Simplex decoder, uncomment the following definition (keep it commented out for A100):

// #define USE_ASYNC

template <typename T>
__device__ void simplex_decode_core(int cw, cg::thread_block &cta, const T* __restrict__ s_in, uint32_t * __restrict__ x_out, uint8_t K, uint32_t E, uint8_t Qm, float* __restrict__ confLevelFloat)
{
    const T llrscale = static_cast<T>(1.0) / static_cast<T>(SIMPLEX_DECODER_MAX_E);
    T c0Sum = 0.;
    T c1Sum = 0.;
    T c2Sum = 0.;
    uint16_t counter = 0;
    float totalPwr = 0.0;

    cg::thread_block_tile<SIMPLEX_DECODER_TPB> tile = cg::tiled_partition<SIMPLEX_DECODER_TPB>(cta);

    // Cooperative accumulation for K = 1
    if (K == 1)
    {
        if (Qm < 3) // Qm == 1 || Qm == 2
        {
            for (int e=cta.thread_rank(); e<E; e+=SIMPLEX_DECODER_TPB)
            {
                c0Sum += llrscale*s_in[e];
                counter += 1;
                //totalPwr += pow((float)s_in[e], 2);
                totalPwr += ((float)s_in[e] * (float)s_in[e]);
            }
        }
        else // (Qm >= 3)
        {
            for (int e=cta.thread_rank(); e<E; e+=SIMPLEX_DECODER_TPB)
            {
                if (((e % Qm) == 0) || ((e % Qm) == 1))
                {
                    c0Sum += llrscale*s_in[e];
                    counter += 1;
                    //totalPwr += pow((float)s_in[e], 2);
                    totalPwr += ((float)s_in[e] * (float)s_in[e]);
                }
            }
        }
        c0Sum = cg::reduce(tile, c0Sum, cg::plus<T>());
        counter = cg::reduce(tile, counter, cg::plus<uint16_t>());
        totalPwr = cg::reduce(tile, totalPwr, cg::plus<float>());
    }

    // thread0 of each warp does the simplex decoding
    if (cta.thread_rank() == 0)
    {
        if (K == 1)
        {
            *x_out = c0Sum < static_cast<T>(0);
            *confLevelFloat = (abs((float)c0Sum*SIMPLEX_DECODER_MAX_E)/counter)/sqrtf(totalPwr/counter);
        }
        else // K == 2
        {
            if (Qm == 1)
            {
                for (int e=0; e<E; e++) 
                {
                    int cwQamIdx = e % 3;

                    switch(cwQamIdx)
                    {
                        case 0:
                        {
                            c0Sum += llrscale*s_in[e];
                            break;
                        }
                        case 1:
                        {
                            c1Sum += llrscale*s_in[e];
                            break; 
                        }
                        case 2:
                        {
                            c2Sum += llrscale*s_in[e];
                            break; 
                        }
                    }
                    totalPwr += ((float)s_in[e] * (float)s_in[e]);
                    counter  += 1;
                }
            }
            else
            {
                for(int rmQamIdx = 0; rmQamIdx < E / Qm; ++rmQamIdx)
                {
                    int cwQamIdx = rmQamIdx % 3;
                    switch(cwQamIdx)
                    {
                        case 0:
                        {
                            c0Sum += llrscale*s_in[Qm*rmQamIdx];
                            c1Sum += llrscale*s_in[Qm*rmQamIdx + 1];
                            break;
                        }
                        case 1:
                        {
                            c2Sum += llrscale*s_in[Qm*rmQamIdx];
                            c0Sum += llrscale*s_in[Qm*rmQamIdx + 1];
                            break;
                        }
                        case 2:
                        {
                            c1Sum += llrscale*s_in[Qm*rmQamIdx];
                            c2Sum += llrscale*s_in[Qm*rmQamIdx + 1];
                            break;
                        }
                    }
                    totalPwr += ((float)s_in[Qm*rmQamIdx] * (float)s_in[Qm*rmQamIdx]);
                    totalPwr += ((float)s_in[Qm*rmQamIdx + 1] * (float)s_in[Qm*rmQamIdx + 1]);
                    counter  += 2;
                }
            }
            
            // matrix of all possible codeword [c0 c1 c2; c0 c1 c2; c0 c1 c2; c0 c1 c2], c2 = mod(c0 + c1, 2)
            // [0 0 0; 0 1 1; 1 0 1; 1 1 0];
            T cost[4] = {0.};
            T cost_min = 0.;
            T cost_max = 0.;
            int idx_min = 0;

            //cost0 = 0
            cost[2] = c1Sum + c2Sum;
            cost[1] = c0Sum + c2Sum;
            cost[3] = c0Sum + c1Sum;

            //printf("   ");
            for (int k=1; k<4; k++)
            {
                if (cost[k] < cost_min)
                {
                    cost_min = cost[k];
                    idx_min = k;
                }
                //printf("cost[%d]: %f  ",k,__half2float(cost[k]));
                if (cost[k] > cost_max)
                {
                    cost_max = cost[k];
                }
            }
            //printf("\n");
            *x_out = idx_min;
            *confLevelFloat = ((float)(cost_max-cost_min)*SIMPLEX_DECODER_MAX_E/counter)/sqrtf(totalPwr/counter)*3.0f/2.0f;
        }

        //printf("   cxSum: %f %f %f => x_out = %d\n",__half2float(c0Sum),__half2float(c1Sum),__half2float(c2Sum),*x_out);
    }
}

// The kernel does a grid-stride over all the codewords.
template <typename T>
__global__ void simplex_decoder_kernel(simplexDecoderDynDescr_t* pDynDescr)
{
    uint16_t   numCW                 =  pDynDescr->nCws;
    cuphySimplexCwPrm_t* pCwPrmsGpu  =  pDynDescr->pCwPrmsGpu;

    cg::thread_block cta = cg::this_thread_block();
#ifdef USE_ASYNC
    __shared__ T shmem[SIMPLEX_DECODER_MAX_E*2];
    T *s_in[2] = {&shmem[0], &shmem[SIMPLEX_DECODER_MAX_E]};
    int wp=1;
    int rp=0;
#else
    __shared__ T shmem[SIMPLEX_DECODER_MAX_E];
    T *s_in = &shmem[0];
#endif

    // Add a margin to CUPHY_NOISE_REGULARIZER when using it for threshold
    // checks to account for floating point precision variation.
    constexpr float CUPHY_NOISE_REGULARIZER_THRESH = CUPHY_NOISE_REGULARIZER + 1.0e-10f;

#ifdef USE_ASYNC
#if CUDART_VERSION >= 11060
#pragma nv_diag_suppress static_var_with_dynamic_init
#else
#pragma diag_suppress static_var_with_dynamic_init
#endif
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> shared_state;
    auto pipeline = cuda::make_pipeline(cta, &shared_state);

    // Prefetch first codeword
    if(pCwPrmsGpu[blockIdx.x].exitFlag == 0)
    {
        pipeline.producer_acquire();
        uint32_t E_clipped = pCwPrmsGpu[blockIdx.x].E;
        if(E_clipped > SIMPLEX_DECODER_MAX_E)
        {
            uint32_t maxBitsPerCw = 3 * pCwPrmsGpu[blockIdx.x].nBitsPerQam;
            E_clipped = maxBitsPerCw * (SIMPLEX_DECODER_MAX_E / maxBitsPerCw);
            if (cta.thread_rank() == 0)
            {
                printf("WARNING: E_clipped (=%d) exceeded max E (=%d)\n", E_clipped, SIMPLEX_DECODER_MAX_E);
            }
        }
        cuda::memcpy_async(cta, s_in[0], pCwPrmsGpu[blockIdx.x].d_LLRs, sizeof(T)*E_clipped, pipeline);
        pipeline.producer_commit();
    }

#endif

    for (int cw=blockIdx.x; cw<numCW; cw += gridDim.x)
    {
#ifdef USE_ASYNC
        if (cw+gridDim.x < numCW)
        {
            if(pCwPrmsGpu[cw+gridDim.x].exitFlag == 0)
            {
                // Prefetch next codeword
                if(pCwPrmsGpu[cw+gridDim.x].exitFlag == 0)
                {
                    pipeline.producer_acquire();
                    uint32_t E_clipped = pCwPrmsGpu[cw+gridDim.x].E;
                    if(E_clipped > SIMPLEX_DECODER_MAX_E)
                    {
                        uint32_t maxBitsPerCw = 3 * pCwPrmsGpu[cw+gridDim.x].nBitsPerQam;
                        E_clipped = maxBitsPerCw * (SIMPLEX_DECODER_MAX_E / maxBitsPerCw);
                        if (cta.thread_rank() == 0)
                        {
                            printf("WARNING: E_clipped (=%d) exceeded max E (=%d)\n", E_clipped, SIMPLEX_DECODER_MAX_E);
                        }
                    }
                    cuda::memcpy_async(cta, s_in[(wp++) % 2], pCwPrmsGpu[cw+gridDim.x].d_LLRs, sizeof(T)*E_clipped, pipeline);
                    pipeline.producer_commit();
                }
            }
        }

        // Decode current codeword
        if(pCwPrmsGpu[cw].exitFlag == 0)
        {
            pipeline.consumer_wait();
            uint32_t E_clipped = pCwPrmsGpu[cw].E;
            if(E_clipped > SIMPLEX_DECODER_MAX_E)
            {
                uint32_t maxBitsPerCw = 3 * pCwPrmsGpu[cw].nBitsPerQam;
                E_clipped = maxBitsPerCw * (SIMPLEX_DECODER_MAX_E / maxBitsPerCw);
                if (cta.thread_rank() == 0)
                {
                    printf("WARNING: E_clipped (=%d) exceeded max E (=%d)\n", E_clipped, SIMPLEX_DECODER_MAX_E);
                }
            }
            float confLevelFloat;
            simplex_decode_core(cw, cta, s_in[(rp++) % 2], pCwPrmsGpu[cw].d_cbEst, pCwPrmsGpu[cw].K, E_clipped, pCwPrmsGpu[cw].nBitsPerQam, &confLevelFloat);
            if(threadIdx.x==0)
            {
                if((pCwPrmsGpu[cw].en_DTXest&CUPHY_DTX_EN) == CUPHY_DTX_EN)
                {   
                    uint8_t d_DTXEst;
                    if(*pCwPrmsGpu[cw].d_noiseVar<=CUPHY_NOISE_REGULARIZER_THRESH)
                    {
                        d_DTXEst = 1;  
                        //*pCwPrmsGpu[cw].d_DTXStatus = 3;  //DTX
                        //printf("simplex_decoder: noisevar = %f, d_chEst = %d, d_DTXEst = %d\n", *pCwPrmsGpu[cw].d_noiseVar, *pCwPrmsGpu[cw].d_cbEst, *pCwPrmsGpu[cw].d_DTXEst);
                    }
                    else
                    {
                        float confLevelThrFloat = CUPHY_DTX_THRESHOLD_ADJ_SIMPLEX_DECODER * pCwPrmsGpu[cw].DTXthreshold;
                        d_DTXEst = 0;
                        //*pCwPrmsGpu[cw].d_DTXStatus = 4;  //no DTX
                        if(confLevelFloat < confLevelThrFloat)
                        {
                            d_DTXEst = 1;
                            //*pCwPrmsGpu[cw].d_DTXStatus = 3;  //DTX
                        }
                        //printf("simplex_decoder: noisevar = %f, DTXthreshold = %f, confLevel = %f, confLevelThr = %f, rmDTX = %d\n", *pCwPrmsGpu[cw].d_noiseVar, pCwPrmsGpu[cw].DTXthreshold, confLevelFloat, confLevelThrFloat, *pCwPrmsGpu[cw].d_DTXEst);
                    }
                    if((pCwPrmsGpu[cw].en_DTXest&CUPHY_DET_EN) == CUPHY_DET_EN)
                    {
                        if(d_DTXEst)
                        {
                            *pCwPrmsGpu[cw].d_DTXStatus = CUPHY_FAPI_DTX;
                            //printf("simplex_decoder: DTXStatus: %d\n", *pCwPrmsGpu[cw].d_DTXStatus);
                        }
                        else
                        {
                            *pCwPrmsGpu[cw].d_DTXStatus = CUPHY_FAPI_NO_DTX;
                            //printf("simplex_decoder: DTXStatus: %d\n", *pCwPrmsGpu[cw].d_DTXStatus);
                        }
                    }
                    //////pCwPrmsGpu[cw].en_DTXest = 0;
                }
            }
            pipeline.consumer_release();

        }

#else
        if(pCwPrmsGpu[cw].exitFlag == 0)
        {
            // Clip E to avoid shared memory buffer overflow
            uint32_t E_clipped = pCwPrmsGpu[cw].E;
            if(E_clipped > SIMPLEX_DECODER_MAX_E)
            {
                uint32_t maxBitsPerCw = 3 * pCwPrmsGpu[cw].nBitsPerQam;
                E_clipped = maxBitsPerCw * (SIMPLEX_DECODER_MAX_E / maxBitsPerCw);
                if (cta.thread_rank() == 0)
                {
                    printf("WARNING: E_clipped (=%d) exceeded max E (=%d)\n", E_clipped, SIMPLEX_DECODER_MAX_E);
                }
            }

            // Cooperative load codeword
            for (int e=cta.thread_rank(); e<E_clipped; e+=cta.size())
            {
                s_in[e] = pCwPrmsGpu[cw].d_LLRs[e];
            }
            __syncthreads();

            // Decode codeword
            float confLevelFloat;
            simplex_decode_core(cw, cta, s_in, pCwPrmsGpu[cw].d_cbEst, pCwPrmsGpu[cw].K, E_clipped, pCwPrmsGpu[cw].nBitsPerQam, &confLevelFloat);

            if(cta.thread_rank() == 0)
            {
                if((pCwPrmsGpu[cw].en_DTXest&CUPHY_DTX_EN) == CUPHY_DTX_EN)
                {   
                    uint8_t d_DTXEst;
                    if(*pCwPrmsGpu[cw].d_noiseVar<=CUPHY_NOISE_REGULARIZER_THRESH)
                    {
                        d_DTXEst = 1;  
                        //*pCwPrmsGpu[cw].d_DTXStatus = 3;  //DTX
                        //printf("simplex_decoder: noisevar = %f, d_chEst = %d, d_DTXEst = %d\n", *pCwPrmsGpu[cw].d_noiseVar, *pCwPrmsGpu[cw].d_cbEst, *pCwPrmsGpu[cw].d_DTXEst);
                    }
                    else
                    {
                        float confLevelThrFloat = CUPHY_DTX_THRESHOLD_ADJ_SIMPLEX_DECODER * pCwPrmsGpu[cw].DTXthreshold;
                        d_DTXEst = 0;
                        //*pCwPrmsGpu[cw].d_DTXStatus = 4;  //no DTX
                        if(confLevelFloat < confLevelThrFloat)
                        {
                            d_DTXEst = 1;
                            //*pCwPrmsGpu[cw].d_DTXStatus = 3;  //DTX
                        }
                        //printf("simplex_decoder: noisevar = %f, DTXthreshold = %f, confLevel = %f, confLevelThr = %f, rmDTX = %d\n", *pCwPrmsGpu[cw].d_noiseVar, pCwPrmsGpu[cw].DTXthreshold, confLevelFloat, confLevelThrFloat, *pCwPrmsGpu[cw].d_DTXEst);
                    }
                    if((pCwPrmsGpu[cw].en_DTXest&CUPHY_DET_EN) == CUPHY_DET_EN)
                    {
                        if(d_DTXEst)
                        {
                            *pCwPrmsGpu[cw].d_DTXStatus = CUPHY_FAPI_DTX;
                            //printf("simplex_decoder: DTXStatus: %d\n", *pCwPrmsGpu[cw].d_DTXStatus);
                        }
                        else
                        {
                            *pCwPrmsGpu[cw].d_DTXStatus = CUPHY_FAPI_NO_DTX;
                            //printf("simplex_decoder: DTXStatus: %d\n", *pCwPrmsGpu[cw].d_DTXStatus);
                        }
                    }
                    //////pCwPrmsGpu[cw].en_DTXest = 0;
                }
            }
            __syncthreads();
        }
#endif
    }
}

 SimplexDecoder::SimplexDecoder()
{
    CW_PER_BLOCK_ = 4;
}

void SimplexDecoder::kernelSelect(uint16_t  nCws, cuphySimplexDecoderLaunchCfg_t* pLaunchCfg)
{
   // launch geometry
   int bpk = (nCws + CW_PER_BLOCK_ - 1) / CW_PER_BLOCK_;
   int tpb = SIMPLEX_DECODER_TPB;

   dim3 gridDim(bpk);
   dim3 blockDim(tpb);

    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(simplex_decoder_kernel<__half>);
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

void SimplexDecoder::setup( uint16_t                        nCws,
                            cuphySimplexCwPrm_t*            pCwPrmsCpu, 
                            cuphySimplexCwPrm_t*            pCwPrmsGpu, 
                            bool                            enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                            simplexDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                            void*                           pGpuDynDesc,                 // pointer to descriptor in gpu
                            cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                            cudaStream_t                    strm)                        // stream to perform copy
{
    // Populate dynamic descriptor:
    pCpuDynDesc->nCws       = nCws;
    pCpuDynDesc->pCwPrmsGpu = pCwPrmsGpu;

    // save pointer to GPU descriptor
    simplexDecoderKernelArgs_t& kernelArgs = m_kernelArgs;
    kernelArgs.pDynDescr                   = reinterpret_cast<simplexDecoderDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(simplexDecoderDynDescr_t), cudaMemcpyHostToDevice, strm);
    }

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect(nCws, pLaunchCfg);
    pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}



void SimplexDecoder::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
    dynDescrSizeBytes  = sizeof(simplexDecoderDynDescr_t);
    dynDescrAlignBytes = alignof(simplexDecoderDynDescr_t);
}