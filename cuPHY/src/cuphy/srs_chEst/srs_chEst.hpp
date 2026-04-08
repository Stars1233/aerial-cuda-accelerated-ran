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

#include "cuphy.h"
#include "tensor_desc.hpp"
#include "cuphy_api.h"

#define MAX_N_ANT_PORTS (4)
#define MAX_N_ANT_PORTS_PER_COMB (12)
#define MAX_N_SYM (4)
#define MAX_N_HOPS (4)
#define MAX_N_REPS (4)
#define MAX_N_COMB_PER_UE (2)


#define MAX_N_SRS_CELL (24) 
#define MAX_N_COMP_BLOCKS (CUPHY_SRS_MAX_N_USERS * 68) // CUPHY_SRS_MAX_N_USERS * nComputeBlocks_perUe (e.g. 272/4)
// nComputeBlocks_perUe = nHops * nPrbsPerHop / 4 * nCombs
#define MAX_N_SRS_RKHS_COMP_BLOCKS (MAX_N_SRS_CELL * MAX_N_ANT_PORTS * MAX_N_SYM * 4 * 2)

#define N_SYM_PER_SLOT 14
#define N_SC_PER_PRB 12
#define N_PRIMES 303
//#define MAX_N_SC 24
#define N_PRB_PER_COMP_BLK 4
//#define N_GRP_PER_COMP_BLK 2

#define POINT_ONE_PERCENT 0.001
#define MIN_NOISE_ENERGY 1e-6

#define SRS_CHEST_BLOCK_SZ 128

struct cuphySrsChEst
{};

// RKHS grid descriptor
struct rkhsGridDesc_t
{
    tensor_ref_any<CUPHY_R_16F> tEigenVecs;   // eigen vectors. Dim: gridSize x nEigs
    tensor_ref_any<CUPHY_R_16F> tEigenValues; // eigen values. Dim: nEigs x 1 
    tensor_ref_any<CUPHY_C_16F> tEigenCorr;   // correlation of modulated eigenvectors

    tensor_ref_any<CUPHY_R_8U>  tSecondStageFourierPerm;    // input permutation for input to second stage Fourier transform
    tensor_ref_any<CUPHY_C_16F> tSecondStageTwiddleFactors; // twiddle factors for second stage of FFT
};


// Channel estimator static descriptor
struct srsChEstStatDescr_t
{
    tensor_ref_any<CUPHY_C_16F>  tFocc_table;
    tensor_ref_any<CUPHY_C_16F>  tFocc_comb2_table;
    tensor_ref_any<CUPHY_C_16F>  tFocc_comb4_table;

    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts1_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts2_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts4_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts8_wide;

    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts1_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts2_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts4_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts6_wide;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts12_wide;

    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts1_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts2_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts4_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb2_nPorts8_narrow;

    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts1_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts2_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts4_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts6_narrow;
    tensor_ref_any<CUPHY_C_16F>  tW_comb4_nPorts12_narrow;

    float noisEstDebias_comb2_nPorts1;
    float noisEstDebias_comb2_nPorts2;
    float noisEstDebias_comb2_nPorts4;
    float noisEstDebias_comb2_nPorts8;

    float noisEstDebias_comb4_nPorts1;
    float noisEstDebias_comb4_nPorts2;
    float noisEstDebias_comb4_nPorts4;
    float noisEstDebias_comb4_nPorts6;
    float noisEstDebias_comb4_nPorts12;

    // RKHS paramaters:
    rkhsGridDesc_t rkhsGridDescs[NUM_RKHS_GRIDS];
    
    uint8_t chEstToL2NormalizationAlgo;
    float   chEstToL2ConstantScaler;
    uint8_t enableDelayOffsetCorrection;
};


struct __align__(32) rkhsCompBlockDescriptor_t
{
    uint16_t ueIdx;
    uint8_t  combIdx;
    uint8_t  portIdx;
    uint8_t  polIdx;

    tensor_ref<CUPHY_C_16F, 4> tFreqProjCoeffs;
    tensor_ref<CUPHY_C_16F, 3> tHammingProjCoeffs;
};





struct cellDescr_t{
    uint8_t                      mu;
    uint16_t                     nRxAntSrs;
    tensor_ref_any<CUPHY_C_16F>  tDataRx;
};


struct __align__(32) compBlockDescr_t{
    uint16_t    ueIdx;
    uint8_t     combIdx;
    uint8_t     hopIdx;
    uint16_t    blockStartPrb;

    uint16_t ueGroupIdx;
    uint16_t nRxAntSrs;
    uint16_t blockStartAnt;
};

struct ueGroupDescr_t{
    uint16_t nUes;
    uint8_t  nAntPorts;
    uint16_t ueIdxs[MAX_N_ANT_PORTS_PER_COMB];
    uint8_t  ueCombIdxs[MAX_N_ANT_PORTS_PER_COMB];
    uint8_t  ueHopIdxs[MAX_N_ANT_PORTS_PER_COMB];
    uint8_t  portToFoccMap[MAX_N_ANT_PORTS_PER_COMB];
    uint16_t portToUeIdxWithinBlock[MAX_N_ANT_PORTS_PER_COMB];
    uint8_t  blockPortToUePortMap[MAX_N_ANT_PORTS_PER_COMB];
};

struct ueDescr_t{
        // compute parameters:
        uint8_t  repSymIdxs[MAX_N_HOPS][MAX_N_REPS];
        uint16_t hopStartPrbs[MAX_N_HOPS];
        uint8_t  nRepPerHop[MAX_N_HOPS]; 
        uint16_t nPrbsPerHop;
        uint8_t  u[MAX_N_SYM];
        float    q[MAX_N_SYM];
        float    alphaCommon;
        uint8_t  n_SRS_cs_max;
        uint8_t  lowPaprTableIdx;
        uint16_t lowPaprPrime;
        uint8_t  nPorts;
        uint8_t  nPortsPerComb;
        uint8_t  portToFoccMap[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS];
        uint8_t  combSize; 
        uint8_t  combOffsets[MAX_N_COMB_PER_UE];
        uint8_t  nCombScPerPrb; 
        uint8_t  portToUeAntMap[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS];
        uint8_t  portToL2OutUeAntMap[MAX_N_COMB_PER_UE][MAX_N_ANT_PORTS];
        uint8_t  cellIdx;
        uint16_t prgSize;
        uint16_t prgSizeL2;

        // ue group parameters:
        uint32_t ueBlockCntr;
        uint32_t ueNumBlocks;

        // temp wideband report
        float   tmpWidebandNoiseEnergy  = 0;
        float   tmpWidebandSignalEnergy = 0;
        __half2 tmpWidebandScCorr       = __floats2half2_rn(0.f, 0.f);
        float   tmpWidebandCsCorrUse    = 0;
        float   tmpWidebandCsCorrNotUse = 0;

        // output buffers:
        float*                       pUeRbSnr;
        cuphySrsReport_t*            pUeSrsReport;
#ifdef ASIM_CUPHY_SRS_OUTPUT_FP32
        tensor_ref_any<CUPHY_C_32F>  tChEstBuff;   
        tensor_ref_any<CUPHY_C_32F>  tChEstToL2Inner;
        tensor_ref_any<CUPHY_C_32F>  tChEstToL2;
#else
        tensor_ref_any<CUPHY_C_16F>  tChEstBuff; 
        tensor_ref_any<CUPHY_C_16F>  tChEstToL2Inner;  
        tensor_ref_any<CUPHY_C_16I>  tChEstToL2;
#endif        
        uint16_t                     chEstBuffStartPrbGrp;
        
        // for tChEstToL2 normalization
        uint16_t nRxAntSrsL2;
        uint16_t nPrbGrpsL2;
        uint8_t  nAntPortsL2;
        uint16_t prgIdxMappingL2[CUPHY_SRS_MAX_N_PRGS_SUPPORTED];
        uint8_t  portIdxMappingL2[MAX_N_ANT_PORTS];
};

struct srsChEstDynDescr_t{
    ueDescr_t                 ueDescrs[CUPHY_SRS_MAX_N_USERS];
    ueGroupDescr_t            ueGroupDescrs[CUPHY_SRS_MAX_N_USERS];
    cellDescr_t               cellDescrs[MAX_N_SRS_CELL];
    compBlockDescr_t          compBlockDescrs[MAX_N_COMP_BLOCKS];
    rkhsCompBlockDescriptor_t rkhsCompBlockDescrs[MAX_N_SRS_RKHS_COMP_BLOCKS];
    int                       nSrsUes;
};

//  srsChEst kernel arguments (Supplied via descriptors)
struct srsChEstKernelArgs_t
{
    srsChEstStatDescr_t* pStatDescr; 
    srsChEstDynDescr_t*  pDynDescr;
};

class srsChEst : public cuphySrsChEst
{
public:
    srsChEst();
    ~srsChEst()                           = default;
    srsChEst(srsChEst const&)            = delete;
    srsChEst& operator=(srsChEst const&) = delete;

    void init(cuphySrsFilterPrms_t*   pSrsFilterPrms,
              cuphySrsRkhsPrms_t*     pRkhsPrms,
              cuphySrsChEstAlgoType_t chEstAlgo,
              uint8_t                 chEstToL2NormalizationAlgo,
              float                   chEstToL2ConstantScaler,
              uint8_t                 enableDelayOffsetCorrection,
              bool                    enableCpuToGpuDescrAsyncCpy,
              srsChEstStatDescr_t*    pCpuStatDesc,
              void*                   pGpuStatDesc,
              cudaStream_t            strm);

    

    cuphyStatus_t setup( uint16_t                     nSrsUes,
                        cuphyUeSrsPrm_t*              h_srsUePrms,
                        uint16_t                      nCells,
                        cuphyTensorPrm_t*             pTDataRx, 
                        cuphySrsCellPrms_t*           h_srsCellPrms,
                        float*                        d_rbSnrBuff,
                        uint32_t*                     h_rbSnrBuffOffsets,
                        cuphySrsReport_t*             d_pSrsReports,
                        cuphySrsChEstBuffInfo_t*      h_chEstBuffInfo,
                        void**                        d_addrsChEstToL2InnerBuff,
                        void**                        d_addrsChEstToL2Buff,
                        cuphySrsChEstToL2_t*          h_chEstToL2,
                        void*                         d_workspace,
                        bool                          enableCpuToGpuDescrAsyncCpy,
                        srsChEstDynDescr_t*           pCpuDynDesc,
                        void*                         pGpuDynDesc,
                        cuphySrsChEstLaunchCfg_t*     pLaunchCfg,
                        cuphySrsChEstNormalizationLaunchCfg_t* pNormalizationLaunchCfg,
                        cudaStream_t                  strm);

    void kernelSelect(srsChEstDynDescr_t*            pCpuDynDesc, 
                      uint16_t                       nSrsUes, 
                      uint16_t                       nCompBlocks, 
                      uint16_t                       nRkhsCompBlocks, 
                      cuphySrsChEstLaunchCfg_t*      pLaunchCfg, 
                      cuphySrsChEstNormalizationLaunchCfg_t* pNormalizationLaunchCfg);

    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    srsChEstKernelArgs_t    m_kernelArgs;
    cuphySrsChEstAlgoType_t m_chEstAlgo;
    uint8_t m_chEstToL2NormalizationAlgo;

};
