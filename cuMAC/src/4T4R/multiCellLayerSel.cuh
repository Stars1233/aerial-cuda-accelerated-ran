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
 typedef struct mcLayerSelDynDescr {
   //----------------- input buffers (common for both DL and UL) ----------------- 
   int16_t*    allocSol; // PRB allocation solution
   float*      sinVal; // array of singular values
   uint16_t*   setSchdUePerCellTTI; // global UE IDs of scheduled UEs
   uint8_t*    cellAssoc; // only for type-0 allocation
   int8_t*     riActUe; 

   // HARQ related buffers
   int8_t*     newDataActUe;
   uint8_t*    layerSelSolLastTx; 

   //----------------- Aerial Sim specific data buffers -----------------
   cuComplex** srsEstChan;

   //----------------- output buffers (common for both DL and UL) ----------------- 
   uint8_t*    layerSelSol; // layer selection solution for the selected UEs per TTI in the coordinated cells
   
   //----------------- parameters (common for both DL and UL) -----------------
   uint16_t    nUe;
   uint16_t    nPrbGrp;
   uint8_t     nUeAnt;
   uint16_t    nCell;
   uint8_t     nTxAnt;
   uint8_t     nRxAnt;

   // parameters not included in the API data structures
   // (set up internally in cuMAC setup() function)
   uint16_t    nUePerBlk;
   float       sinValThr;
   float       corrThr;
 } mcLayerSelDynDescr_t;

 class multiCellLayerSel {
 public:
    // default constructor
    multiCellLayerSel(cumacCellGrpPrms* cellGrpPrms);

    // constructor for Aerial Sim
    multiCellLayerSel(cumacCellGrpPrms* cellGrpPrms, uint8_t in_Asim); // in_Asim: indicator for Aerial Sim

    // desctructor
    ~multiCellLayerSel();

    multiCellLayerSel(multiCellLayerSel const&)            = delete;
    multiCellLayerSel& operator=(multiCellLayerSel const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacCellGrpUeStatus*       cellGrpUeStatus,
               cumacSchdSol*               schdSol,
               cumacCellGrpPrms*           cellGrpPrms,
               uint8_t                     in_ri,
               cudaStream_t                strm); // requires externel synchronization

    // run() function for per-TTI algorithm execution
    void run(cudaStream_t strm);

    // parameter/data buffer logging function for debugging purpose
    void debugLog(); // for debugging only, printing out dynamic descriptor parameters

 private:
    // indicator for DL/UL
    uint8_t          DL; // 1 for DL, 0 for UL

    // allocate type
    uint8_t          allocType;

    // precoding type: 0 - no precoding, 1 - SVD precoding
    uint8_t          precodingScheme;

    // indicator for HARQ
    uint8_t          enableHarq;

    // indicator for UE reported RI-based layer selection
    uint8_t          RI; // 1 for UE reported RI based, 0 otherwise

    // Aerial Sim indicator
    uint8_t          Asim;

    // dynamic descriptors
    std::unique_ptr<mcLayerSelDynDescr_t> pCpuDynDesc;
    mcLayerSelDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    void kernelSelect();
 };

 typedef struct multiCellLayerSel*           mcLayerSelHndl_t;

 static __global__ void mcLayerSelKernel_type0(mcLayerSelDynDescr_t* pDynDescr);
 static __global__ void mcLayerSelKernel_type1(mcLayerSelDynDescr_t* pDynDescr);
 static __global__ void mcLayerSelKernel_ri(mcLayerSelDynDescr_t* pDynDescr);

 // HARQ
 // static __global__ void mcLayerSelKernel_type0_harq(mcLayerSelDynDescr_t* pDynDescr);
 static __global__ void mcLayerSelKernel_type1_harq(mcLayerSelDynDescr_t* pDynDescr);
 static __global__ void mcLayerSelKernel_ri_harq(mcLayerSelDynDescr_t* pDynDescr);
}
