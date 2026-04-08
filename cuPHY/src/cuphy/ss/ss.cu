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

#include <stdio.h>
#include <cuda_runtime.h>

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "ss.hpp"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"
#include "descrambling.hpp"
#include "descrambling.cuh"
#include "polar_encoder.hpp"

using namespace cuphy_i;
using namespace descrambling;

// clang-format off
//SSB_PSS_X_EXT, SSB_SSS_X0 and SSB_SSS_X1 each are CUPHY_SSB_N_SS_SEQ_BITS elements wide and do not depend on any config. parameters.
//FIXME could have packed them in 4 uint32_t elements
// SSB_PSS_X_EXT has its first 86 elements repeated to avoid a modulo operation.
static __device__ __constant__  uint8_t SSB_PSS_X_EXT[] =
{
0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
1, 1, 0, 1, 1, 0, 1, 0, 1, 1,
1, 0, 1, 0, 0, 0, 1, 1, 0, 0,
1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 1, 0, 0, 1, 0, 0, 1, 1, 0,
1, 0, 0, 1, 1, 1, 1, 0, 1, 1,
1, 0, 0, 0, 0, 1
};


static __device__ __constant__  uint8_t SSB_SSS_X0[] =
{
1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0
};

static __device__ __constant__  uint8_t SSB_SSS_X1[] =
{
1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
};
// clang-format on

/* Notes on PBCH scrambling sequence:
   * seed depends on NID only (so cell specific)
   * The generated sequence is CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS * (v+1)
     where v is [0, 3] or [0, 7]
     depending on Lmax (SSB specific) and blockIndx (SSB specific)
     So up to 8 (v+1)nSCRAM scrambling sequence will be generated
     but only the last one is what we care about.
     //Note to self, maybe we could leverage this and store them somewhere?

     CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS is 864 bits = 108B

*/

__device__ __inline__ uint32_t compute_qam_index_v2(const uint32_t NID, const uint16_t nF)
{
    const uint8_t indices[4][9] = {{1, 2, 3, 5, 6, 7, 9, 10, 11},
                                   {0, 2, 3, 4, 6, 7, 8, 10, 11},
                                   {0, 1, 3, 4, 5, 7, 8, 9, 11},
                                   {0, 1, 2, 4, 5, 6, 8, 9, 10}};

    const int tid       = threadIdx.x - 128;
    const int tid_div_9 = tid / 9;
    const int tid_mod_9 = tid - (tid_div_9 * 9);
    int       qam_re    = indices[NID & 0x3][tid_mod_9];
    int       qam_idx   = 0;
    if(tid < 180)
    { // entire 2nd symbol (240REs, 20RBs, 20 * 9 = 180 QAM REs)
        qam_idx = nF;
    }
    else if(tid >= (180 + 2 * 36))
    { // entire 4th symbol (240 REs; 180 QAM REs)
        qam_idx = 3 * nF - 28 * 12;
    }
    else if(tid < 216)
    { // 3rd symbol first 48 REs = 36 QAM REs
        qam_idx = 2 * nF - 20 * 12;
    }
    else
    {                              // 3rd symbol last 48 REs = 36 QAM REs
        qam_idx = 2 * nF - 8 * 12; // -24*12 + 16*12
    }
    qam_idx += (tid_div_9 * 12 + qam_re);
    return qam_idx;
}

__device__ __inline__ uint32_t compute_dmrs_index_v2(const uint16_t nF)
{
    const int idxDmrs   = threadIdx.x - 576;
    const int idx_div_3 = idxDmrs / 3;
    const int dmrs_re   = (idxDmrs - (idx_div_3 * 3)) * 4;
    int       dmrs_idx  = 0;
    if(idxDmrs < 60)
    { // entire 2nd symbol (240REs, 20RBs, 20 * 3 = 60 DMRS REs)
        dmrs_idx = nF;
    }
    else if(idxDmrs >= (60 + 2 * 12))
    { // entire 4th symbol (240 REs; 60 DMRS REs)
        dmrs_idx = 3 * nF - 28 * 12;
    }
    else if(idxDmrs < 72)
    { // 3rd symbol first 48 REs = 12 DMRS REs
        dmrs_idx = 2 * nF - 20 * 12;
    }
    else
    {                               // 3rd symbol last 48 REs = 12 DRMS REs
        dmrs_idx = 2 * nF - 8 * 12; // -24*12 + 16*12
    }
    dmrs_idx += (idx_div_3 * 12 + dmrs_re);
    // dmrs_re also needs + (NID & 0x3) added to ret. value

    return dmrs_idx;
}

// Kernel for reading coded bits, and modulating PBCH, DMRS, PSS and SSS sequences and mapping them to time frequency domain subcarriers
template <typename TComplex>
__global__ void ssbModTfSigKernel(TComplex** __restrict__ d_tfSignal,
                                  const uint8_t* __restrict__ d_x_tx,
                                  const cuphyPerSsBlockDynPrms_t* __restrict__ d_ssb_params,
                                  const cuphyPerCellSsbDynPrms_t* __restrict__ d_per_cell_params,
                                  const cuphyPmWOneLayer_t* __restrict__  d_pmw_params)
{
    const int                                            SSB_idx = blockIdx.x;
    const int                                            tid     = threadIdx.x;
    typedef typename scalar_from_complex<TComplex>::type scalar_t;

    const uint8_t  block_index = d_ssb_params[SSB_idx].blockIndex;
    const float    beta_pss    = d_ssb_params[SSB_idx].beta_pss;
    const float    beta_sss    = d_ssb_params[SSB_idx].beta_sss;
    const uint16_t cell_index  = d_ssb_params[SSB_idx].cell_index;
    const uint8_t enablePrcdBf = d_ssb_params[SSB_idx].enablePrcdBf;
    uint16_t pmwPrmIdx         = 0xFFFF;//d_ssb_params[SSB_idx].pmwPrmIdx;
    uint8_t  nPorts            = 0;
    if(enablePrcdBf)
    {
        pmwPrmIdx = d_ssb_params[SSB_idx].pmwPrmIdx;
        nPorts    = d_pmw_params[pmwPrmIdx].nPorts;
    }

    const uint16_t NID  = d_per_cell_params[cell_index].NID;
    const uint16_t nHF  = d_per_cell_params[cell_index].nHF;  // can be 0 or 1
    const uint16_t Lmax = d_per_cell_params[cell_index].Lmax; // can be 4,8 or 64

    // DMRS and scrambling sequence generation for PBCH part of SSB.
    const int           max_scrambling_elements = 27; // CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS/32=864/32
    __shared__ uint32_t scrambling_seq[max_scrambling_elements];
    uint32_t            v = (Lmax == 4) ? (block_index & 0x3) : (block_index & 0x7);

    const int           max_dmrs_elements = 9; // 288/32 = CUPHY_SSB_N_DMRS_SEQ_BITS/32
    __shared__ uint32_t dmrs_seq[max_dmrs_elements];

    int      i_ssb  = v + ((Lmax == 4) ? 4 * nHF : 0);
    uint32_t c_init = (0x1 << 11) * (i_ssb + 1) * ((NID >> 2) + 1) +
                      (0x1 << 6) * (i_ssb + 1) + (NID & 0x3);

    if(tid < max_scrambling_elements)
    {
        scrambling_seq[tid] = gold32(NID, (tid + max_scrambling_elements * v) * 32);
    }
    else if((tid >= 32) && (tid < (32 + max_dmrs_elements)))
    {
        dmrs_seq[tid - 32] = gold32(c_init, (tid - 32) * 32);
    }
    __syncthreads();

    //Could do the PSS and SSS before syncthreads as there's not dependency to reduce idle warps

    // Used for PSS and SSS seq. generation later
    const uint16_t NID1 = NID / 3;
    const uint16_t NID2 = NID - NID1 * 3;
    const int16_t  m0   = NID1 / 112 * 15 + 5 * NID2;
    const int16_t  m1   = NID1 % 112;

    const uint16_t f0                = d_ssb_params[SSB_idx].f0;
    const uint8_t  t0                = d_ssb_params[SSB_idx].t0;
    const uint16_t nF                = d_per_cell_params[cell_index].nF;
    const uint16_t slot_buffer_idx   = d_per_cell_params[cell_index].slotBufferIdx;
    const float    beta_sss_factor   = beta_sss * 0.70710678f;
    TComplex* __restrict__ tf_signal = d_tfSignal[slot_buffer_idx] + (t0 * nF + f0);
    const uint16_t offset_per_port = nF * OFDM_SYMBOLS_PER_SLOT;

    const uint32_t tx_offset = SSB_idx * 108;
    const TComplex zeroValue = make_complex<TComplex>::create(0,0);

    if(threadIdx.x < CUPHY_SSB_N_SS_SEQ_BITS)
    { // Map PSS and SSS
        //Pss_idx[0, 126] are [56, 182]
        const int idxPssSss = threadIdx.x;
        // Reminder NID2 can be 0, 1, 2. Max length of the seq. is 127
        // max 43*2 = 86. So round up at most once.. 127 + 86 = 213 to avoid modulo operation.
        // Also numbers are 0 or 1. If 0 -> output is 1. If 1 output is -1)
        //tf_signal[idxPssSss + 56].x = static_cast<scalar_t>(beta_pss*(1 - 2 * SSB_PSS_X[(idxPssSss + 43* NID2) % CUPHY_SSB_N_SS_SEQ_BITS]));
        scalar_t tmpSS0 = static_cast<scalar_t>(beta_pss * (1 - 2*SSB_PSS_X_EXT[idxPssSss + 43*NID2]));
        if(enablePrcdBf)
        {
            for(int idx = 0; idx < nPorts; idx++)
            {
                tf_signal[idxPssSss + 56 + offset_per_port*idx].x = tmpSS0*static_cast<scalar_t>(d_pmw_params[pmwPrmIdx].matrix[idx].x);
                tf_signal[idxPssSss + 56 + offset_per_port*idx].y = tmpSS0*static_cast<scalar_t>(d_pmw_params[pmwPrmIdx].matrix[idx].y);
            }
        }
        else
        {
            tf_signal[idxPssSss + 56].x = tmpSS0;
        }

        //SSS_idx[0,126] are  [2*240 + 56, 2 * 240 + 182]
        int16_t lower_half = 1 - 2*SSB_SSS_X0[(idxPssSss + m0) % CUPHY_SSB_N_SS_SEQ_BITS];
        int16_t upper_half = 1 - 2*SSB_SSS_X1[(idxPssSss + m1) % CUPHY_SSB_N_SS_SEQ_BITS];
        scalar_t tmpSS1 = static_cast<scalar_t>(beta_sss* (lower_half * upper_half));
        if(enablePrcdBf)
        {
            for(int idx = 0; idx < nPorts; idx++)
            {
                tf_signal[2*nF + 56 + idxPssSss + offset_per_port*idx].x = tmpSS1*static_cast<scalar_t>(d_pmw_params[pmwPrmIdx].matrix[idx].x);
                tf_signal[2*nF + 56 + idxPssSss + offset_per_port*idx].y = tmpSS1*static_cast<scalar_t>(d_pmw_params[pmwPrmIdx].matrix[idx].y);
            }
        }
        else
        {
            tf_signal[2*nF + 56 + idxPssSss].x = tmpSS1;
        }
    }
    else if((threadIdx.x >= 128) && (threadIdx.x < (128 + (CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS / 2))))
    { // Map PBCH - 432 REs
        const int pbch_qam_tid = threadIdx.x - 128;
        uint32_t  bIdx         = 2 * pbch_qam_tid;
        uint32_t  idxByte      = bIdx >> 3;  // divide by 8
        uint32_t  idxBit       = bIdx & 0x7; // remainder
        uint32_t  input_x      = (d_x_tx[tx_offset + idxByte] >> idxBit) & 0x1;
        uint32_t  input_y      = (d_x_tx[tx_offset + idxByte] >> (idxBit + 1)) & 0x1;

        uint32_t scrambling_val   = scrambling_seq[(pbch_qam_tid >> 4)];
        uint16_t scrambling_val_x = (scrambling_val >> (((2 * pbch_qam_tid) & 0x1F))) & 0x1;
        uint16_t scrambling_val_y = (scrambling_val >> (((2 * pbch_qam_tid + 1) & 0x1F))) & 0x1;
        TComplex qam;

        int x = (input_x + scrambling_val_x) & 0x1;
        int y = (input_y + scrambling_val_y) & 0x1;
        qam.x = static_cast<scalar_t>(beta_sss_factor * (1 - 2 * x));
        qam.y = static_cast<scalar_t>(beta_sss_factor * (1 - 2 * y));

        //computation to remove QAM index computation from host
        int qam_idx = compute_qam_index_v2(NID, nF);
        if(enablePrcdBf)
        {
            for(int idx = 0; idx < nPorts; idx++)
                tf_signal[qam_idx + offset_per_port*idx] = __hcmadd(qam, d_pmw_params[pmwPrmIdx].matrix[idx], zeroValue); // uncoalesced writes
        }
        else
        {
            tf_signal[qam_idx] = qam; // uncoalesced writes
        }
    }
    else if((threadIdx.x >= 576) && (threadIdx.x < (576 + (CUPHY_SSB_N_DMRS_SEQ_BITS / 2))))
    { // Map DMRS - 144 REs
        const int32_t idxDmrs = threadIdx.x - 576;

        uint32_t dmrs_val = dmrs_seq[(idxDmrs >> 4)]; // (2*idxDmrs) >> 5
        uint16_t x        = (dmrs_val >> (((2 * idxDmrs) & 0x1F))) & 0x1;
        uint16_t y        = (dmrs_val >> (((2 * idxDmrs + 1) & 0x1F))) & 0x1;
        TComplex qam      = make_complex<TComplex>::create(beta_sss_factor * (1 - 2 * x),
                                                      beta_sss_factor * (1 - 2 * y));

        //PBCH –> entire 2nd and 4th symbols. 240 REs for each, and the first 48 and last 48 REs of the 3rd symbol. Total = 2* 240 +2 * 48 = 576 REs. Included DMRS.
        int dmrs_idx = compute_dmrs_index_v2(nF) + (NID & 0x3); // computation to remove DMRS index computation from the host
        if(enablePrcdBf)
        {
            for(int idx = 0; idx < nPorts; idx++)
                tf_signal[dmrs_idx + offset_per_port*idx] = __hcmadd(qam, d_pmw_params[pmwPrmIdx].matrix[idx], zeroValue); //uncoalesced writes
        }
        else
        {
            tf_signal[dmrs_idx] = qam; //uncoalesced writes
        }
    }
}



cuphyStatus_t kernelSelectSsbMapper(cuphySsbMapperLaunchCfg_t* pLaunchCfg,
                                    uint16_t                   num_SSBs)
{
    if (pLaunchCfg == nullptr) return CUPHY_STATUS_INVALID_ARGUMENT;
    // kernel
    void* kernelFunc = reinterpret_cast<void*>(ssbModTfSigKernel<__half2>);
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));}
    // launch geometry
    dim3 gridDim(num_SSBs);
    dim3 blockDim(736);

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

    return CUPHY_STATUS_SUCCESS;
}

// SSB TX pipeline to launch kernels on GPU
cuphyStatus_t cuphyRunSsbMapper(const uint8_t*                    d_x_tx,
                                __half2**                         d_tfSignal,
                                const cuphyPerSsBlockDynPrms_t*   d_ssb_params,
                                const cuphyPerCellSsbDynPrms_t*   d_per_cell_params,
                                const cuphyPmWOneLayer_t*         d_pmw_params,
                                uint16_t                          num_SSBs,
                                uint16_t                          num_cells,
                                cudaStream_t                      stream,
                                cuphySsbMapperLaunchCfg_t*        pSsbMapperCfg)
 {

    if ((d_x_tx == nullptr) || (d_tfSignal == nullptr) || (d_ssb_params == nullptr) || (d_per_cell_params == nullptr) || d_pmw_params == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    // modulate PBCH, DMRS, PSS and SSS sequences and map to time frequency domain subcarriers
    CUresult e = launch_kernel(pSsbMapperCfg->kernelNodeParamsDriver, stream);
    return (e == CUDA_SUCCESS) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

cuphyStatus_t CUPHYWINAPI cuphySSBsKernelSelect(cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
                                                cuphySsbMapperLaunchCfg_t*                pSsbMapperLaunchCfg,
                                                uint16_t                                  nSSBs)
{
    cuphyStatus_t ssbEncdrStatus = polar_encoder::kernelSelectEncodeRateMatchMultiSSBs(pEncdRMLaunchCfg, nSSBs);
    if(ssbEncdrStatus != CUPHY_STATUS_SUCCESS)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cuphyStatus_t ssbMapperStatus = kernelSelectSsbMapper(pSsbMapperLaunchCfg, nSSBs);
    if(ssbMapperStatus != CUPHY_STATUS_SUCCESS)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return CUPHY_STATUS_SUCCESS;
}
