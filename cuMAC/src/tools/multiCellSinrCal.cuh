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

 // dynamic descriptor for CUDA kernels
 typedef struct mcSinrCalDynDescr {
   //----------------- input buffers ----------------- 
   // ************ DL only *************
   cuComplex*  estH_fr_actUe;
   cuComplex*  prdMat_actUe; // precoder matrix coefficients per cell-UE link, per PRG, per symbol-Tx antenna pair
   cuComplex*  detMat_actUe;
   //----------------- Aerial Sim specific data buffers -----------------
   cuComplex** srsEstChan; 
   cuComplex*  prdMat_asim;
   cuComplex*  detMat_asim;
   // ***********************************

   // **** common for both DL and UL ****
   uint16_t*   cellId; // IDs of coordinated cells
   uint8_t*    cellAssocActUe;
   float*      sinVal_actUe; // singular values associated with each precoder matrix, per cell-UE link, per PRG, per layer (maximum number of layers is equal to the number of UE antennas) for all active UEs

   //----------------- Aerial Sim specific data buffers (common for both DL and UL) -----------------
   uint8_t*    cellAssoc;
   float*      sinVal_asim;

   //----------------- output buffers ----------------- 
   float*      postEqSinr;
   float*      wbSinr;
   // ***********************************

   //----------------- parameters (common for both DL and UL) -----------------
   uint16_t    nUe; // total number of selected UEs in the coordinated cells
   uint16_t    nActiveUe; // number of active UEs for all coordinated cells
   uint16_t    nCell;
   uint16_t    nPrbGrp;
   uint8_t     nBsAnt;
   uint8_t     nUeAnt;
   float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power, limitation: SNR (per antenna) should be <= 111 dB
   uint16_t    nMaxUeSinrCalPerRnd;
} mcSinrCalDynDescr_t;

 class multiCellSinrCal {
 public:
    // default constructor
    multiCellSinrCal(cumacCellGrpPrms* cellGrpPrms);

    // destructor
    ~multiCellSinrCal();
    
    multiCellSinrCal(multiCellSinrCal const&)            = delete;
    multiCellSinrCal& operator=(multiCellSinrCal const&) = delete;
    
    // default setup() function
    void setup(cumacCellGrpPrms*           cellGrpPrms,
               uint8_t                     in_columnMajor,
               cudaStream_t                strm); // requires externel synchronization 
    // set in_columnMajor to 1 if channel matrix access is column-major; 0 otherwise

    // setup() for calculating wideband SINR
    void setup_wbSinr(cumacCellGrpPrms*           cellGrpPrms,
                      cudaStream_t                strm); // requires externel synchronization   
    
    // run() function (for both subband and wideband SINR calculations)
    void run(cudaStream_t strm);

    void debugLog();

 private:
    // CUDA kernel selection
    void kernelSelect();

    // indicator for DL/UL
    uint8_t DL; // 1 for DL, 0 for UL

    // column-major or row-major channel matrix access: 0 - row major, 1 - column major
    uint8_t columnMajor;

    // precoding type: 0 - no precoding, 1 - SVD precoding
    uint8_t precodingScheme;

    // dynamic descriptors
    std::unique_ptr<mcSinrCalDynDescr_t> pCpuDynDesc;
    mcSinrCalDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;

    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;
    
    // for debugging
    uint16_t    nActiveUe;
    uint16_t    nPrbGrp;
    uint8_t     nUeAnt;
 };

 typedef struct multiCellSinrCal*            mcSinrCalHndl_t;
 
 // DL:
 // column major channel access kernels
 static __global__ void multiCellSinrCalKernel_noPrdMmseIrc_cm(mcSinrCalDynDescr_t* pDynDescr);
 static __global__ void multiCellSinrCalKernel_svdPrdMmseIrc_cm(mcSinrCalDynDescr_t* pDynDescr);

 // row major channel access kernels
 static __global__ void multiCellSinrCalKernel_noPrdMmseIrc_rm(mcSinrCalDynDescr_t* pDynDescr);

 // calculate wideband eq-SINR
 static __global__ void multiCellWideBandSinrCalKernel(mcSinrCalDynDescr_t* pDynDescr);

 // UL:
 // column major channel access kernels
 static __global__ void multiCellSinrCalKernel_svdPrdMmse_UL(mcSinrCalDynDescr_t* pDynDescr);
}