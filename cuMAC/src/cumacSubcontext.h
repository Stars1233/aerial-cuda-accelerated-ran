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
 #include "h5TvCreate.h"
 #include "h5TvLoad.h"

// cuMAC namespace
namespace cumac {

 class cumacSubcontext {
 public:
    // default constructor
    cumacSubcontext(const std::string&    tvFilename, 
                    uint8_t               in_GPU, 
                    uint8_t               in_halfPrecision, 
                    uint8_t               in_layerRi,
                    uint8_t               in_aodtInd, 
                    uint8_t               in_heteroUeSel, 
                    uint8_t               in_schAlg, 
                    const uint8_t*        in_modulesCalled, 
                    cudaStream_t          strm);
    // tvFilename: H5 TV file name
    // in_GPU: whether to use GPU or CPU scheduler, 0 - CPU scheduler, 1 - GPU scheduler
    // in_halfPrecision: whether to use FP16 CUDA kernels, 0 - FP32 kernels, 1 - FP16 (half-precision) kernels
    // in_layerRi: whether to use RI-based layer selection, 0 - not use, 1 - use
    // in_aodtInd: whether the cuMAC subcontext is created for AODT testing, 0 - not for AODT testing, 1 - for AODT testing
    // in_heteroUeSel: whether to enable heterogeneous UE selection config. across cells, 0 - disabled, 1 - enabled
    // in_schAlg: choose scheduling algorithm: 0 - round-robin; 1 - multi-cell proportional fair; 2 - single-cell proportional fair
    // in_modulesCalled: array of indicators for scheduler modules being called;
    // ** size of in_modulesCalled is fixed to 4;
    // ** Entry 0 is for UE selection:       0 - not being called, 1 - being called
    // ** Entry 1 is for PRG allocation:     0 - not being called, 1 - being called
    // ** Entry 2 is for layer selection:    0 - not being called, 1 - being called
    // ** Entry 3 is for MCS selection:      0 - not being called, 1 - being called
    // ** for Phase-3 testing, in_modulesCalled should be hardcoded to 1111 (calling UE selection, PRG allocation, layer selection, and MCS selection)
    // requires external synchronization

    // destructor
    ~cumacSubcontext();
    
    // in_halfPrecision: indicator for half-precision processing, 0 - float2/cuComplex, 1 - __nv_bfloat162
    void setup(const std::string& tvFilename, uint8_t lightWeight, float percSmNumThrdBlk, cudaStream_t strm); 
    // tvFilename: H5 TV file name
    // lightWeight: whether to use heavy-weight (SRS-based) or light-weight PRG allocation kernel, 0 - heavy-weight, 1 - SINR computation based light-weight, 2 - SINR loading based light-weight
    // percSmNumThrdBlk: percentage of SMs to determine the number of thread blocks used in light-weight kernels
    // requires external synchronization     

    void run(cudaStream_t strm); // requires external synchronization    

    void debugLog();

    // handlers for multi-cell scheduler modules
    mcUeSelHndl_t         mcUeSelGpu;
    mcRRUeSelHndl_t       mcRRUeSelGpu;
    mcSchdHndl_t          mcSchGpu;
    mcRRSchdHndl_t        mcRRSchGpu;
    mcsSelLUTHndl_t       mcMcsSelGpu;
    mcLayerSelHndl_t      mcLayerSelGpu;
    mcMuUeSortHndl_t      mcMuUeSortGpu;
    mcMuUeGrpHndl_t       mcMuUeGrpGpu;

    mcUeSelCpuHndl_t      mcUeSelCpu;
    rrUeSelCpuHndl_t      rrUeSelCpu;
    mcSchdCpuHndl_t       mcSchCpu;
    rrSchdCpuHndl_t       rrSchCpu;
    mcsSelLUTCpuHndl_t    mcMcsSelCpu;
    mcLayerSelCpuHndl_t   mcLayerSelCpu;

    // API structures
    std::unique_ptr<cumacCellGrpUeStatus>        cellGrpUeStatusGpu;
    std::unique_ptr<cumacSchdSol>                schdSolGpu;
    std::unique_ptr<cumacCellGrpPrms>            cellGrpPrmsGpu;

    std::unique_ptr<cumacCellGrpUeStatus>        cellGrpUeStatusCpu;
    std::unique_ptr<cumacSchdSol>                schdSolCpu;
    std::unique_ptr<cumacCellGrpPrms>            cellGrpPrmsCpu;

    // cuMAC simulation parameter structure
    std::unique_ptr<cumacSimParam>               simParam;

 private:
    // TV parameter structure
    cumacSchedulerParam          data;

    // indicator for GPU/CPU scheduler
    uint8_t                      GPU;

    // precision
    uint8_t                      halfPrecision;

    // scheduling algorithm: 0 - round-robin; 1 - multi-cell proportional fair; 2 - single-cell proportional fair
    uint8_t                      schAlg;

    // column-major or row-major channel matrix access: 0 - row major, 1 - column major
    uint8_t                      columnMajor;

    // indicator for whether to use RI-based layer selection
    uint8_t                      layerRi;

    // indicator for heterogeneous UE selection config. across cells
    uint8_t                      heteroUeSelCells;

    // Aerial Sim indicator
    uint8_t                      aodtInd;

    // indication of scheduler modules being called
    std::vector<uint8_t>         modulesCalled;
    const uint8_t                numSchedulerModules = 4;

    // buffers
    uint8_t**                    prgMsk;
    cuComplex**                  srsEstChan;
    
    // buffer sizes
    uint32_t perCellPrgMskSize;
    uint32_t ndActSize;
    uint32_t allocLTSize;
    uint32_t mcsSelLTSize;
    uint32_t layerSelLTSize;
    uint32_t tbeSize;
    uint32_t tbeActUeSize;
    uint32_t setSchdUeSize;
    uint32_t sinrSize;
    uint32_t wbSinrSize;
    uint32_t cidSize;
    uint32_t assocSize;
    uint32_t assocActUeSize;
    uint32_t mcsSelSolSize;
    uint32_t layerSize;
    uint32_t arSize;
    uint32_t arActUeSize;
    uint32_t sinValSize;
    uint32_t prgMskSize;
    uint32_t ueMapSize;
    uint32_t hfrSize;
    uint32_t perCellHfrLen;
    uint32_t perCellHfrSize;
    uint32_t prdSize;
    uint32_t detSize;
    uint32_t prdLen;
    uint32_t detLen;
    uint32_t hSize;
    uint32_t hLen;
    uint32_t hHalfSize;
    uint32_t gpuAllocSolSize;
    uint32_t pfMetricSize;
    uint32_t pfIdSize;
    uint32_t prioWeightActUeSize;
    uint32_t numUeSchdArrSize;
    uint32_t blerTargetActUeSize;
 };
}