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

constexpr uint16_t VSIZE = 1024; // shared memory size used in type-1 allocation kernels

// Architectures with higher register pressure need fewer blocks per SM
#if defined(CUDA_ARCH_86_89) || defined(CUDA_ARCH_100_120_121)
constexpr uint8_t MinBlkPerSM_ = 1;
#else
constexpr uint8_t MinBlkPerSM_ = 2;
#endif

// dynamic descriptor for CUDA kernels
typedef struct mcDynDescr {
  //----------------- input buffers ----------------- 
  // ************ DL only *************
  cuComplex*  estH_fr; 
  cuComplex*  prdMat;
  cuComplex*  detMat;
  // half-precision buffers
  __nv_bfloat162*    estH_fr_half; 
  //----------------- Aerial Sim specific data buffers -----------------
  cuComplex** srsEstChan; 
  cuComplex*  prdMat_asim;
  cuComplex*  detMat_asim;
  float*      wbSinr; // global wideband SINR array
  // ***********************************

  // **** common for both DL and UL ****
  uint16_t*   cellId; // IDs of coordinated cells
  uint8_t*    cellAssoc;
  float*      avgRates;
  float*      postEqSinr; // global post-eq SINR array
  float*      sinVal; 
  uint16_t*   setSchdUePerCellTTI; // global UE IDs of scheduled UEs
  float*      pfMetricArr; // for storing computed PF metrices
  uint16_t*   pfIdArr; // for storing indices (indicating PRB and UE indecies) of computed PF metrices
  int*        numCompleteBlk; // for type-1 allocate. Memory allocated when creating scheduler object

  //----------------- Aerial Sim specific data buffers -----------------
  float*      sinVal_asim;

  // HARQ related buffers
  uint8_t**   prgMsk; // PRG availability bit musk
  int8_t*     newDataActUe;
  int16_t*    allocSolLastTx; 

  //----------------- output buffers ----------------- 
  int16_t*    allocSol; // -1 indicates unallocated
  // ***********************************

  //----------------- parameters (common for both DL and UL) ----------------- 
  uint16_t    nUe; // total number of selected UEs in the coordinated cells
  uint16_t    nCell; // number of coordinated cells
  uint8_t     numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
  uint16_t    totNumCell; // number of all cells in the network. (not needed if channel buffer only contains channels within coordinated cells)
  uint16_t    nPrbGrp; // number of PRB groups
  uint8_t     nBsAnt; // number of BS antennas
  uint8_t     nUeAnt; // number of UE antennas, assumption's that nUeAnt <= nBsAnt
  float       W; // frequency bandwidth (Hz) of a PRB group
  float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power, limitation: SNR (per antenna) should be <= 111 dB
  uint16_t    nMaxSchdUePerRnd; // maximum number of UEs per cell that can be scheduled per round 
  float       betaCoeff; // coefficient for improving cell edge UEs' performance in multi-cell scheduling
} mcDynDescr_t;

class multiCellScheduler {
public:
  // constructor
  multiCellScheduler(cumacCellGrpPrms* cellGrpPrms);

  // constructor for Aerial Sim. in_Asim should be set to 1 when called from Aerial Sim
  multiCellScheduler(cumacCellGrpPrms* cellGrpPrms, uint8_t in_Asim);

  // destructor
  ~multiCellScheduler();

  multiCellScheduler(multiCellScheduler const&)            = delete;
  multiCellScheduler& operator=(multiCellScheduler const&) = delete;

  // default setup() function for per-TTI algorithm execution
  void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
             cumacSchdSol*               schdSol,
             cumacCellGrpPrms*           cellGrpPrms,
             cumacSimParam*              simParam,
             uint8_t                     in_columnMajor,
             uint8_t                     in_halfPrecision,
             uint8_t                     in_lightWeight,
             float                       in_percSmNumThrdBlk,
             cudaStream_t                strm); 
  // in_columnMajor: 0 - row-major channel access, 1 - column-major channel access
  // in_halfPrecision: 0 - call FP32 floating type kernel, 1 - call FP16 (bfloat162) half-precision kernel
  // in_lightWeight: 0 - call heavy-weight kernel, 1 - call (SINR computation based) light-weight kernel, 2 - call (SINR loading based) light-weight kernel
  // in_enableHarq: 0 - HARQ disabled, 1 - HARQ enabled
  // requires externel synchronization
             
  // setup function for Aerial Sim for per-TTI algorithm execution
  void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
             cumacSchdSol*               schdSol,
             cumacCellGrpPrms*           cellGrpPrms,
             uint8_t                     in_columnMajor,
             uint8_t                     in_halfPrecision,
             cudaStream_t                strm);  
  // in_columnMajor: should be set to 0. For Aerial Sim, only row-major channel access is supported.
  // in_halfPrecision: should be set to 0 for now. For Aerial Sim, half-precision (FP16) multi-cell scheduler kernel is not available yet.
  // requires externel synchronization

  // run() function for per-TTI algorithm execution
  void run(cudaStream_t strm); // requires externel synchronization     

  // parameter/data buffer logging function for debugging purpose
  void debugLog(); // for debugging only, printing out dynamic descriptor parameters

private:
  // indicator for DL/UL
  uint8_t DL; // 1 for DL, 0 for UL

  // precision: 0 - float2, 1 - __nv_bfloat162
  uint8_t halfPrecision;

  // allocate type: 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
  uint8_t allocType;

  // column-major or row-major channel matrix access: 0 - row major, 1 - column major
  uint8_t columnMajor;

  // precoding type: 0 - no precoding, 1 - SVD precoding
  uint8_t precodingScheme;

  // indicator for calling light-weight kernel
  uint8_t lightWeight;

  // indicator for HARQ
  uint8_t enableHarq;

  // Aerial Sim indicator
  uint8_t Asim;

  // number of SMs in the device
  uint16_t numSM;

  // percentage of SMs to determine the number of thread blocks used in light-weight kernel
  float percSmNumThrdBlk;

  // dynamic descriptors
  std::unique_ptr<mcDynDescr_t> pCpuDynDesc;
  mcDynDescr_t* pGpuDynDesc;

  // CUDA kernel parameters
  uint16_t numThrdBlk;
  uint16_t numThrdPerBlk;

  dim3 gridDim;
  dim3 blockDim;

  // launch configuration structure
  std::unique_ptr<launchCfg_t> pLaunchCfg;

  // for type-1 allocate
  int* numCompleteBlk_d; // variable in GPU global memory to indicate the number of thread blocks that have completed compute job
  std::vector<int> numCompleteBlk_h; // storing zero value in CPU memory for initializing numCompleteBlk_d per setup call

  void kernelSelect();
};

typedef struct multiCellScheduler*          mcSchdHndl_t;

// DL: 
// multi-cell scheduler kernel for Type-0 allocation with no precoding and MMSE-IRC equalizer
static __global__ void multiCellSchedulerKernel_noPrdMmseIrc(mcDynDescr_t* pDynDescr);

// multi-cell scheduler kernel for Type-0 allocation with SVD precoding and MMSE-IRC equalizer
static __global__ void multiCellSchedulerKernel_svdMmseIrc(mcDynDescr_t* pDynDescr);

// multi-cell scheduler kernel for Type-1 allocation (consecutive RB allocation) with no precoding and MMSE-IRC equalizer
// column-major channel access
static __global__ void multiCellSchedulerKernel_type1_NoPrdMmseIrc_cm(mcDynDescr_t* pDynDescr); // with no precoding and MMSE-IRC equalizer
// column-major channel access
static __global__ void multiCellSchedulerKernel_type1_svdPrdMmseIrc_cm(mcDynDescr_t* pDynDescr); // with SVD precoding and MMSE-IRC equalizer

// row-major channel access
static __global__ void multiCellSchedulerKernel_type1_NoPrdMmseIrc_rm(mcDynDescr_t* pDynDescr);

// half-precision kernels
static __global__ void multiCellSchedulerKernel_half_noPrdMmseIrc(mcDynDescr_t* pDynDescr);

// light-weight kernels
// type-0 SINR computation light-weight PF kernel
static __global__ void lwPfSchedulerKernel_noPrdSinrCompute(mcDynDescr_t* pDynDescr);
// type-0 SINR loading light-weight kernel
static __global__ void lwPfSchedulerKernel_noPrdSinrLoad(mcDynDescr_t* pDynDescr);


// Aerial Sim
// static __global__ void multiCellSchedulerKernel_Asim_type1_svdPrdMmseIrc_rm(mcDynDescr_t* pDynDescr);
// static __global__ void multiCellSchedulerKernel_Asim_type1_svdPrdMmse_rm(mcDynDescr_t* pDynDescr);
static __global__ void multiCellSchedulerKernel_Asim_type1_wbSinr(mcDynDescr_t* pDynDescr);

// Aerial Sim - HARQ
// static __global__ void multiCellSchedulerKernel_Asim_type1_svdPrdMmseIrc_rm_harq(mcDynDescr_t* pDynDescr);
// static __global__ void multiCellSchedulerKernel_Asim_type1_svdPrdMmse_rm_harq(mcDynDescr_t* pDynDescr);
static __global__ void multiCellSchedulerKernel_Asim_type1_wbSinr_harq(mcDynDescr_t* pDynDescr);

// UL:
// multi-cell scheduler kernel for Type-1 allocation (consecutive RB allocation)
static __global__ void multiCellSchedulerKernel_type1_svdPrdMmseIrc_UL(mcDynDescr_t* pDynDescr); //  with SVD precoding and MMSE-IRC equalizer
// multi-cell scheduler kernel for Type-1 allocation (consecutive RB allocation)
static __global__ void multiCellSchedulerKernel_type1_svdPrdMmseIrc_harq_UL(mcDynDescr_t* pDynDescr); //  with SVD precoding and MMSE-IRC equalizer
}
