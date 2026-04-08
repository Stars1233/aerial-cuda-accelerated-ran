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

 #include "cumacSubcontext.h"
 #include "api.h"
 #include "cumac.h"

// cuMAC namespace
namespace cumac {

 cumacSubcontext::cumacSubcontext(const std::string&    tvFilename, 
                                  uint8_t               in_GPU, 
                                  uint8_t               in_halfPrecision, 
                                  uint8_t               in_layerRi,
                                  uint8_t               in_aodtInd, 
                                  uint8_t               in_heteroUeSel, 
                                  uint8_t               in_schAlg, 
                                  const uint8_t*        in_modulesCalled, 
                                  cudaStream_t          strm)
 {    
    // GPU/GPU scheduler indicator 
    GPU = in_GPU;

    // AODT testing indicator
    aodtInd = in_aodtInd;

    // heterogeneous UE selection config. across cells indicator
    heteroUeSelCells = in_heteroUeSel;

    // RI-based layer selection
    layerRi = in_layerRi;

    // precision
    halfPrecision = in_halfPrecision;

    // scheduling algorithm
    schAlg = in_schAlg;

    modulesCalled.resize(numSchedulerModules);
    std::memcpy(modulesCalled.data(), in_modulesCalled, numSchedulerModules*sizeof(uint8_t));

    if (GPU == 0) { // CPU scheduler
        if (aodtInd == 1) {
            printf("\nERROR: Aerial Sim format CPU scheduler is not implementd\n");
            return;
        }
    }

    uint8_t numberModulesCalled = 0;
    for (int i = 0; i < numSchedulerModules; i++) {
        if (modulesCalled[i] == 1) {
            switch(i) {
              case 0:
                if(schAlg == 0) {
                    printf("cuMAC scheduler module UE selection is called, using round-robin\n");
                }
                else {
                    printf("cuMAC scheduler module UE selection is called, using proportional fair\n");
                }
                break;
              case 1:
                if(schAlg == 0) {
                    printf("cuMAC scheduler module PRG allocation is called, using round-robin\n");
                }
                else {
                    printf("cuMAC scheduler module PRG allocation is called, using proportional fair\n");
                }
                break;
              case 2:
                printf("cuMAC scheduler module layer selection is called\n");
                break;
              case 3:
                printf("cuMAC scheduler module MCS selection is called\n");
                break;
            }
            
            numberModulesCalled++;
        } else if (modulesCalled[i] != 1 && modulesCalled[i] != 0) {
            printf("\nERROR: format of scheduler module indication is incorrect:\n");
            printf("\nSize of the indication array should be equal to 4;\n");
            printf("Each entry of the array can be either 0 or 1\n");
            printf("Entry 0 is for UE selection:       0 - not being called, 1 - being called\n");
            printf("Entry 1 is for PRG allocation:     0 - not being called, 1 - being called\n");
            printf("Entry 2 is for layer selection:    0 - not being called, 1 - being called\n");
            printf("Entry 3 is for MCS selection:      0 - not being called, 1 - being called\n");
            printf("Examples: 0100 - only call PRG allocation; 1111 - call all scheduler modules\n");
            printf("0000 is invalid because no scheduler module is being called\n");
            return;
        }
    }
    
    if (numberModulesCalled == 0) {
        printf("ERROR: no scheduler modules being called\n");
        return;
    }

    // initialize API structures and data buffers with TV
    // open TV H5 file
    H5::H5File file(tvFilename, H5F_ACC_RDONLY);

    // Open the dataset
    H5::DataSet dataset = file.openDataSet("cumacSchedulerParam");

    // Get the compound data type
    H5::CompType compoundType = dataset.getCompType();

    // Read the data from the dataset
    dataset.read(&data, compoundType);

    // create simulator control structure
    simParam = std::make_unique<cumacSimParam>();

    simParam->totNumCell = data.totNumCell;

    columnMajor = data.columnMajor;

    // determine buffer sizes
    tbeSize = sizeof(int8_t)*data.nUe; 
    tbeActUeSize = sizeof(int8_t)*data.nActiveUe;
    setSchdUeSize = sizeof(uint16_t)*data.nCell*data.numUeSchdPerCellTTI;
    sinrSize = sizeof(float)*data.nActiveUe*data.nPrbGrp*data.nUeAnt;
    wbSinrSize = sizeof(float)*data.nActiveUe*data.nUeAnt;
    cidSize = sizeof(uint16_t)*data.nCell;
    assocSize = sizeof(uint8_t)*data.nCell*data.nUe; 
    assocActUeSize = sizeof(uint8_t)*data.nCell*data.nActiveUe;
    mcsSelSolSize = sizeof(int16_t)*data.nUe;
    blerTargetActUeSize = sizeof(float)*data.nActiveUe;
    layerSize = sizeof(uint8_t)*data.nUe;
    arSize = sizeof(float)*data.nUe;
    arActUeSize = sizeof(float)*data.nActiveUe;
    sinValSize = sizeof(float)*data.nUe*data.nPrbGrp*data.nUeAnt;
    ndActSize = sizeof(int8_t)*data.nActiveUe;
    allocLTSize = sizeof(int16_t)*2*data.nUe; 
    mcsSelLTSize = sizeof(int16_t)*data.nUe; 
    layerSelLTSize = sizeof(uint8_t)*data.nUe; 
    prgMskSize = sizeof(uint8_t*)*data.nCell;
    perCellPrgMskSize = sizeof(uint8_t)*data.nPrbGrp;
    ueMapSize = sizeof(int32_t)*data.nUe; 
    hfrSize = sizeof(cuComplex*)*data.nCell;
    perCellHfrLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nUeAnt;
    perCellHfrSize = sizeof(cuComplex)*perCellHfrLen;
    prdLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nBsAnt;
    prdSize = sizeof(cuComplex)*prdLen;
    detLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nBsAnt;
    detSize = sizeof(cuComplex)*detLen;
    hLen = data.nPrbGrp*data.nUe*data.nCell*data.nBsAnt*data.nUeAnt;
    hSize = sizeof(cuComplex)*hLen;
    hHalfSize = sizeof(__nv_bfloat162)*hLen;
    prioWeightActUeSize = sizeof(uint16_t)*data.nActiveUe;
    numUeSchdArrSize = sizeof(uint8_t)*data.nCell;

    if (data.allocType == 1) {
        gpuAllocSolSize = sizeof(int16_t)*2*data.nUe;

        uint16_t pfSize = data.nPrbGrp*data.numUeSchdPerCellTTI;
        uint16_t pow2N = 2;
 
        while(pow2N<pfSize) {
          pow2N = pow2N << 1;
        }

        pfMetricSize = sizeof(float)*data.nCell*pow2N;
        pfIdSize     = sizeof(uint16_t)*data.nCell*pow2N;
    } else {
        gpuAllocSolSize = sizeof(int16_t)*data.nCell*data.nPrbGrp;
        pfMetricSize = 0;
        pfIdSize = 0;
    } 

    std::vector<float> blerTargetActUe(data.nActiveUe, 0.1);

    // create API structures /////////////////////////////////
    if (GPU == 1) { // GPU scheduler
        cellGrpUeStatusGpu = std::make_unique<cumacCellGrpUeStatus>();
        schdSolGpu         = std::make_unique<cumacSchdSol>();
        cellGrpPrmsGpu     = std::make_unique<cumacCellGrpPrms>();

        cellGrpUeStatusCpu = nullptr;
        schdSolCpu         = nullptr;
        cellGrpPrmsCpu     = nullptr;

        cellGrpPrmsGpu->nUe = data.nUe;
        cellGrpPrmsGpu->nCell = data.nCell;
        cellGrpPrmsGpu->nPrbGrp = data.nPrbGrp;
        cellGrpPrmsGpu->nBsAnt = data.nBsAnt;
        cellGrpPrmsGpu->nUeAnt = data.nUeAnt;
        cellGrpPrmsGpu->W = data.W;
        cellGrpPrmsGpu->sigmaSqrd = data.sigmaSqrd;
        cellGrpPrmsGpu->betaCoeff = data.betaCoeff;
        cellGrpPrmsGpu->precodingScheme = data.precodingScheme;
        cellGrpPrmsGpu->receiverScheme = data.receiverScheme;
        cellGrpPrmsGpu->allocType = data.allocType;
        cellGrpPrmsGpu->numUeSchdPerCellTTI = data.numUeSchdPerCellTTI;
        cellGrpPrmsGpu->nActiveUe = data.nActiveUe;
        cellGrpPrmsGpu->sinValThr = data.sinValThr;
        cellGrpPrmsGpu->harqEnabledInd = data.harqEnabledInd;
        cellGrpPrmsGpu->dlSchInd = data.dlSchInd;

        // allocate GPU memory
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellId, cidSize));
        if (heteroUeSelCells == 1) {
            CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->numUeSchdPerCellTTIArr, numUeSchdArrSize));
        }
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellAssoc, assocSize)); 
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellAssocActUe, assocActUeSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->avgRates, arSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->avgRatesActUe, arActUeSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->tbErrLast, tbeSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->tbErrLastActUe, tbeActUeSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->postEqSinr, sinrSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->wbSinr, wbSinrSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->blerTargetActUe, blerTargetActUeSize));
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->blerTargetActUe, blerTargetActUe.data(), blerTargetActUeSize, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->allocSol, gpuAllocSolSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->setSchdUePerCellTTI, setSchdUeSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->mcsSelSol, mcsSelSolSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->layerSelSol, layerSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->newDataActUe, ndActSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->prioWeightActUe, prioWeightActUeSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->allocSolLastTx, allocLTSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->mcsSelSolLastTx, mcsSelLTSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->layerSelSolLastTx, layerSelLTSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->sinVal_asim, sinValSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->sinVal, sinValSize));
    
        if (data.allocType) {
            CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->pfMetricArr, pfMetricSize));
            CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->pfIdArr, pfIdSize));
        } else {
            schdSolGpu->pfMetricArr = nullptr;
            schdSolGpu->pfIdArr = nullptr; 
        }

        CUDA_CHECK_ERR(cudaMallocHost((void **)&prgMsk, prgMskSize));
        for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMalloc((void **)&prgMsk[cIdx], perCellPrgMskSize));
        }
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prgMsk, prgMskSize));
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->prgMsk, prgMsk, prgMskSize, cudaMemcpyHostToDevice));
    
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr, hSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr_half, hHalfSize));
    
        CUDA_CHECK_ERR(cudaMallocHost((void **)&srsEstChan, hfrSize));     
        for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMalloc((void **)&srsEstChan[cIdx], perCellHfrSize));
        }
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsEstChan, hfrSize));
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->srsEstChan, srsEstChan, hfrSize, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prdMat, prdSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prdMat_asim, prdSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->detMat, detSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->detMat_asim, detSize));
    } else { // CPU scheduler
        srsEstChan    = nullptr;
        prgMsk          = nullptr;
        
        cellGrpUeStatusCpu = std::make_unique<cumacCellGrpUeStatus>();
        schdSolCpu         = std::make_unique<cumacSchdSol>();
        cellGrpPrmsCpu     = std::make_unique<cumacCellGrpPrms>();
        
        cellGrpUeStatusGpu = nullptr;
        schdSolGpu         = nullptr;
        cellGrpPrmsGpu     = nullptr;

        cellGrpPrmsCpu->nUe = data.nUe;
        cellGrpPrmsCpu->nCell = data.nCell;
        cellGrpPrmsCpu->nPrbGrp = data.nPrbGrp;
        cellGrpPrmsCpu->nBsAnt = data.nBsAnt;
        cellGrpPrmsCpu->nUeAnt = data.nUeAnt;
        cellGrpPrmsCpu->W = data.W;
        cellGrpPrmsCpu->sigmaSqrd = data.sigmaSqrd;
        cellGrpPrmsCpu->betaCoeff = data.betaCoeff;
        cellGrpPrmsCpu->precodingScheme = data.precodingScheme;
        cellGrpPrmsCpu->receiverScheme = data.receiverScheme;
        cellGrpPrmsCpu->allocType = data.allocType;
        cellGrpPrmsCpu->numUeSchdPerCellTTI = data.numUeSchdPerCellTTI;
        cellGrpPrmsCpu->nActiveUe = data.nActiveUe;
        cellGrpPrmsCpu->sinValThr = data.sinValThr;
        cellGrpPrmsCpu->harqEnabledInd = data.harqEnabledInd;
        cellGrpPrmsCpu->dlSchInd = data.dlSchInd;

        // allocate CPU memory
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->cellId, cidSize));
        if (heteroUeSelCells == 1) {
            CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->numUeSchdPerCellTTIArr, numUeSchdArrSize));
        }
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->cellAssoc, assocSize)); 
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->cellAssocActUe, assocActUeSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->avgRates, arSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->avgRatesActUe, arActUeSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->tbErrLast, tbeSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->tbErrLastActUe, tbeActUeSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->postEqSinr, sinrSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->wbSinr, wbSinrSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->blerTargetActUe, blerTargetActUeSize));
        std::memcpy(cellGrpPrmsCpu->blerTargetActUe, blerTargetActUe.data(), blerTargetActUeSize);
        CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->allocSol, gpuAllocSolSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->setSchdUePerCellTTI, setSchdUeSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->mcsSelSol, mcsSelSolSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->layerSelSol, layerSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->newDataActUe, ndActSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->allocSolLastTx, allocLTSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->mcsSelSolLastTx, mcsSelLTSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->layerSelSolLastTx, layerSelLTSize));       
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpUeStatusCpu->prioWeightActUe, prioWeightActUeSize));

        if (data.allocType) {
            CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->pfMetricArr, pfMetricSize));
            CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->pfIdArr, pfIdSize));
        } else {
            schdSolCpu->pfMetricArr = nullptr;
            schdSolCpu->pfIdArr = nullptr; 
        }

        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->prgMsk, prgMskSize));
        for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->prgMsk[cIdx], perCellPrgMskSize));
        }
            
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->estH_fr, hSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->estH_fr_half, hHalfSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->prdMat, prdSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->detMat, detSize));
        CUDA_CHECK_ERR(cudaMallocHost((void **)&cellGrpPrmsCpu->sinVal, sinValSize));
    }
     
    // create scheduler module objects
    if (modulesCalled[0] == 1) {
        if (GPU == 1) { // GPU scheduler
            mcUeSelCpu = nullptr;
            rrUeSelCpu = nullptr;
            if (schAlg == 0) {
                mcUeSelGpu = nullptr;
                mcRRUeSelGpu = new multiCellRRUeSel(cellGrpPrmsGpu.get());
            } else {
                mcUeSelGpu = new multiCellUeSelection(cellGrpPrmsGpu.get());
                mcRRUeSelGpu = nullptr;
            }
        } else { // CPU scheduler
            mcUeSelGpu = nullptr;
            mcRRUeSelGpu = nullptr;
            if (schAlg == 0) {
                mcUeSelCpu = nullptr;
                rrUeSelCpu = new roundRobinUeSelCpu(cellGrpPrmsCpu.get());
            } else {
                mcUeSelCpu = new multiCellUeSelectionCpu(cellGrpPrmsCpu.get());
                rrUeSelCpu = nullptr;
            }
        }    
    } else {
        mcUeSelGpu = nullptr;
        mcRRUeSelGpu = nullptr;
        mcUeSelCpu = nullptr;
        rrUeSelCpu = nullptr;
    }
      
    if (modulesCalled[1] == 1) {
        if (GPU == 1) { // GPU scheduler
            if (schAlg == 0) {
                mcSchGpu = nullptr;
                mcRRSchGpu = new multiCellRRScheduler(cellGrpPrmsGpu.get());
            } else {
                if (aodtInd == 1) { // for AODT testing
                    mcSchGpu = new multiCellScheduler(cellGrpPrmsGpu.get(), aodtInd);
                } else {
                    mcSchGpu = new multiCellScheduler(cellGrpPrmsGpu.get());
                }
                mcRRSchGpu = nullptr;
            }
            mcSchCpu = nullptr;
            rrSchCpu = nullptr;
        } else { // CPU scheduler
            if (schAlg == 0) {
                mcSchCpu = nullptr;
                rrSchCpu = new roundRobinSchedulerCpu(cellGrpPrmsCpu.get());
            } else {
                mcSchCpu = new multiCellSchedulerCpu(cellGrpPrmsCpu.get());
                rrSchCpu = nullptr;
            }
            mcSchGpu = nullptr;
            mcRRSchGpu = nullptr;
        }    
    } else {
        mcSchGpu = nullptr;
        mcRRSchGpu = nullptr;
        mcSchCpu = nullptr;
        rrSchCpu = nullptr;
    }
      
    if (modulesCalled[2] == 1) {
        if (GPU == 1) { // GPU scheduler
            mcLayerSelGpu = new multiCellLayerSel(cellGrpPrmsGpu.get(), aodtInd);
            mcLayerSelCpu = nullptr;
        } else { // CPU scheduler
            mcLayerSelGpu = nullptr;
            mcLayerSelCpu = new multiCellLayerSelCpu(cellGrpPrmsCpu.get());
        }    
    } else {
        mcLayerSelGpu = nullptr;
        mcLayerSelCpu = nullptr;
    }
      
    if (modulesCalled[3] == 1) {
        if (GPU == 1) { // GPU scheduler
            mcMcsSelGpu = new mcsSelectionLUT(cellGrpPrmsGpu.get(), strm);
            mcMcsSelCpu = nullptr;
        } else { // CPU scheduler
            mcMcsSelGpu = nullptr;
            mcMcsSelCpu = new mcsSelectionLUTCpu(cellGrpPrmsCpu.get());
        }
    } else {
        mcMcsSelGpu = nullptr;
        mcMcsSelCpu = nullptr;
    }
 }

 cumacSubcontext::~cumacSubcontext()
 {
    if (mcUeSelGpu) delete mcUeSelGpu;
    if (mcRRUeSelGpu) delete mcRRUeSelGpu;
    if (mcSchGpu) delete mcSchGpu;
    if (mcRRSchGpu) delete mcRRSchGpu;
    if (mcLayerSelGpu) delete mcLayerSelGpu;
    if (mcMcsSelGpu) delete mcMcsSelGpu;
    if (mcUeSelCpu) delete mcUeSelCpu;
    if (rrUeSelCpu) delete rrUeSelCpu;
    if (mcSchCpu) delete mcSchCpu;
    if (rrSchCpu) delete rrSchCpu;
    if (mcLayerSelCpu) delete mcLayerSelCpu;
    if (mcMcsSelCpu) delete mcMcsSelCpu;

    if (cellGrpPrmsGpu) {
        if (cellGrpPrmsGpu->cellId)                 CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellId));
        if (cellGrpPrmsGpu->numUeSchdPerCellTTIArr) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->numUeSchdPerCellTTIArr));
        if (cellGrpPrmsGpu->cellAssoc)              CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellAssoc));
        if (cellGrpPrmsGpu->cellAssocActUe)         CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellAssocActUe));
        if (cellGrpPrmsGpu->postEqSinr)             CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->postEqSinr));
        if (cellGrpPrmsGpu->wbSinr)                 CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->wbSinr));
        if (cellGrpPrmsGpu->blerTargetActUe)        CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->blerTargetActUe));
        if (cellGrpPrmsGpu->prdMat)                 CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->prdMat));
        if (cellGrpPrmsGpu->detMat)                 CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->detMat));
        if (cellGrpPrmsGpu->prgMsk)                 CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->prgMsk));
        if (cellGrpPrmsGpu->srsEstChan)             CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsEstChan));
        if (cellGrpPrmsGpu->prdMat_asim)            CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->prdMat_asim));
        if (cellGrpPrmsGpu->detMat_asim)            CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->detMat_asim));
        if (cellGrpPrmsGpu->sinVal_asim)            CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->sinVal_asim));   
        if (cellGrpPrmsGpu->sinVal)                 CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->sinVal));
        if (cellGrpPrmsGpu->estH_fr)                CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr));
        if (cellGrpPrmsGpu->estH_fr_half)           CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr_half));
    }

    if (cellGrpUeStatusGpu) {
        if (cellGrpUeStatusGpu->avgRates)           CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->avgRates));
        if (cellGrpUeStatusGpu->avgRatesActUe)      CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->avgRatesActUe));
        if (cellGrpUeStatusGpu->tbErrLast)          CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->tbErrLast));
        if (cellGrpUeStatusGpu->tbErrLastActUe)     CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->tbErrLastActUe));
        if (cellGrpUeStatusGpu->prioWeightActUe)    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->prioWeightActUe));
        if (cellGrpUeStatusGpu->newDataActUe)       CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->newDataActUe));
        if (cellGrpUeStatusGpu->allocSolLastTx)     CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->allocSolLastTx));
        if (cellGrpUeStatusGpu->mcsSelSolLastTx)    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->mcsSelSolLastTx));
        if (cellGrpUeStatusGpu->layerSelSolLastTx)  CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->layerSelSolLastTx));
    }

    if (schdSolGpu) {
        if (schdSolGpu->allocSol)                   CUDA_CHECK_ERR(cudaFree(schdSolGpu->allocSol));
        if (schdSolGpu->setSchdUePerCellTTI)        CUDA_CHECK_ERR(cudaFree(schdSolGpu->setSchdUePerCellTTI));
        if (schdSolGpu->mcsSelSol)                  CUDA_CHECK_ERR(cudaFree(schdSolGpu->mcsSelSol));
        if (schdSolGpu->layerSelSol)                CUDA_CHECK_ERR(cudaFree(schdSolGpu->layerSelSol));
        if (schdSolGpu->pfMetricArr)                CUDA_CHECK_ERR(cudaFree(schdSolGpu->pfMetricArr));
        if (schdSolGpu->pfIdArr)                    CUDA_CHECK_ERR(cudaFree(schdSolGpu->pfIdArr));
    }
    
    if (prgMsk) {
        for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            if (prgMsk[cIdx]) CUDA_CHECK_ERR(cudaFree(prgMsk[cIdx]));
        }
        CUDA_CHECK_ERR(cudaFreeHost(prgMsk));
    }

    if (srsEstChan) {
        for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            if (srsEstChan[cIdx]) CUDA_CHECK_ERR(cudaFree(srsEstChan[cIdx]));
        }
        CUDA_CHECK_ERR(cudaFreeHost(srsEstChan));   
    }
    
    if (cellGrpPrmsCpu) {
        if (cellGrpPrmsCpu->cellId)                 CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->cellId));
        if (cellGrpPrmsCpu->numUeSchdPerCellTTIArr) CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->numUeSchdPerCellTTIArr));
        if (cellGrpPrmsCpu->cellAssoc)              CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->cellAssoc));
        if (cellGrpPrmsCpu->cellAssocActUe)         CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->cellAssocActUe));
        if (cellGrpPrmsCpu->postEqSinr)             CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->postEqSinr));
        if (cellGrpPrmsCpu->wbSinr)                 CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->wbSinr));
        if (cellGrpPrmsCpu->blerTargetActUe)        CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->blerTargetActUe));
        for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            if (cellGrpPrmsCpu->prgMsk[cIdx])       CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->prgMsk[cIdx]));
        }
        if (cellGrpPrmsCpu->prgMsk)                 CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->prgMsk));
        if (cellGrpPrmsCpu->estH_fr)                CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->estH_fr));
        if (cellGrpPrmsCpu->estH_fr_half)           CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->estH_fr_half));
        if (cellGrpPrmsCpu->prdMat)                 CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->prdMat));
        if (cellGrpPrmsCpu->detMat)                 CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->detMat));
        if (cellGrpPrmsCpu->sinVal)                 CUDA_CHECK_ERR(cudaFreeHost(cellGrpPrmsCpu->sinVal));
    }

    if (cellGrpUeStatusCpu) {
        if (cellGrpUeStatusCpu->avgRates)           CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->avgRates));
        if (cellGrpUeStatusCpu->avgRatesActUe)      CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->avgRatesActUe));
        if (cellGrpUeStatusCpu->tbErrLast)          CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->tbErrLast));
        if (cellGrpUeStatusCpu->tbErrLastActUe)     CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->tbErrLastActUe));
        if (cellGrpUeStatusCpu->newDataActUe)       CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->newDataActUe));
        if (cellGrpUeStatusCpu->allocSolLastTx)     CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->allocSolLastTx));
        if (cellGrpUeStatusCpu->mcsSelSolLastTx)    CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->mcsSelSolLastTx));
        if (cellGrpUeStatusCpu->layerSelSolLastTx)  CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->layerSelSolLastTx));
        if (cellGrpUeStatusCpu->prioWeightActUe)    CUDA_CHECK_ERR(cudaFreeHost(cellGrpUeStatusCpu->prioWeightActUe));
    }

    if (schdSolCpu) {
        if (schdSolCpu->allocSol)                   CUDA_CHECK_ERR(cudaFreeHost(schdSolCpu->allocSol));
        if (schdSolCpu->setSchdUePerCellTTI)        CUDA_CHECK_ERR(cudaFreeHost(schdSolCpu->setSchdUePerCellTTI));
        if (schdSolCpu->mcsSelSol)                  CUDA_CHECK_ERR(cudaFreeHost(schdSolCpu->mcsSelSol));
        if (schdSolCpu->layerSelSol)                CUDA_CHECK_ERR(cudaFreeHost(schdSolCpu->layerSelSol));
        if (schdSolCpu->pfMetricArr)                CUDA_CHECK_ERR(cudaFreeHost(schdSolCpu->pfMetricArr));
        if (schdSolCpu->pfIdArr)                    CUDA_CHECK_ERR(cudaFreeHost(schdSolCpu->pfIdArr));
    }
 }

 void cumacSubcontext::setup(const std::string& tvFilename, uint8_t lightWeight, float percSmNumThrdBlk, cudaStream_t strm)
 {
    if (GPU == 1) { // GPU scheduler
        loadFromH5(tvFilename, cellGrpUeStatusGpu.get(), cellGrpPrmsGpu.get(), schdSolGpu.get());

        // setup scheduler components
        if (aodtInd == 1) { // Aerial Sim
            if (mcUeSelGpu) mcUeSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
            if (mcRRUeSelGpu) mcRRUeSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
            if (mcSchGpu) mcSchGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), columnMajor, halfPrecision, strm);
            if (mcRRSchGpu) mcRRSchGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
            if (mcLayerSelGpu) mcLayerSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), layerRi, strm);
            if (mcMcsSelGpu) mcMcsSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
        } else {
            if (mcUeSelGpu) mcUeSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
            if (mcRRUeSelGpu) mcRRUeSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
            if (mcSchGpu) mcSchGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), simParam.get(), columnMajor, halfPrecision, lightWeight, percSmNumThrdBlk, strm);
            if (mcRRSchGpu) mcRRSchGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
            if (mcLayerSelGpu) mcLayerSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), layerRi, strm);
            if (mcMcsSelGpu) mcMcsSelGpu->setup(cellGrpUeStatusGpu.get(), schdSolGpu.get(), cellGrpPrmsGpu.get(), strm);
        }
    } else { // CPU scheduler
        loadFromH5_CPU(tvFilename, cellGrpUeStatusCpu.get(), cellGrpPrmsCpu.get(), schdSolCpu.get());

        if (aodtInd == 1) { // Aerial Sim
            printf("\nERROR: Aerial Sim format CPU scheduler is not implementd\n");
            return;
        } else {
            if (mcUeSelCpu) mcUeSelCpu->setup(cellGrpUeStatusCpu.get(), schdSolCpu.get(), cellGrpPrmsCpu.get());
            if (rrUeSelCpu) rrUeSelCpu->setup(cellGrpUeStatusCpu.get(), schdSolCpu.get(), cellGrpPrmsCpu.get());
            if (mcSchCpu) mcSchCpu->setup(cellGrpUeStatusCpu.get(), schdSolCpu.get(), cellGrpPrmsCpu.get(), simParam.get(), columnMajor);
            if (rrSchCpu) rrSchCpu->setup(cellGrpUeStatusCpu.get(), schdSolCpu.get(), cellGrpPrmsCpu.get());
            if (mcLayerSelCpu) mcLayerSelCpu->setup(cellGrpUeStatusCpu.get(), schdSolCpu.get(), cellGrpPrmsCpu.get());
            if (mcMcsSelCpu) mcMcsSelCpu->setup(cellGrpUeStatusCpu.get(), schdSolCpu.get(), cellGrpPrmsCpu.get());
        }
    }
 }

 void cumacSubcontext::run(cudaStream_t strm)
 {
    if (mcUeSelGpu) {
        mcUeSelGpu->run(strm);
    }

    if (mcRRUeSelGpu) {
        mcRRUeSelGpu->run(strm);
    }

    if (mcSchGpu) {
        mcSchGpu->run(strm);
    }

    if (mcRRSchGpu) {
        mcRRSchGpu->run(strm);
    }
    
    if (mcLayerSelGpu) {
        mcLayerSelGpu->run(strm);
    }

    if (mcMcsSelGpu) {
        mcMcsSelGpu->run(strm);
    }

    if (mcUeSelCpu) {
        mcUeSelCpu->run();
    }

    if (rrUeSelCpu){
        rrUeSelCpu->run();
    }

    if (mcSchCpu) {
        mcSchCpu->run();
    }

    if (rrSchCpu) {
        rrSchCpu->run();
    }

    if (mcLayerSelCpu) {
        mcLayerSelCpu->run();
    }

    if (mcMcsSelCpu) {
        mcMcsSelCpu->run();
    }
 }

 void cumacSubcontext::debugLog()
 {
    if (GPU == 0) {
        return;
    }

    printf("********************************************\n");
    printf("** cuMAC TV loading test parameters:\n\n");
    printf("nUe: %d\n", cellGrpPrmsGpu->nUe);
    printf("nActiveUe: %d\n", cellGrpPrmsGpu->nActiveUe);
    printf("numUeSchdPerCellTTI: %d\n", cellGrpPrmsGpu->numUeSchdPerCellTTI);
    printf("nCell: %d\n", cellGrpPrmsGpu->nCell);
    printf("totNumCell: %d\n", simParam->totNumCell);
    printf("nPrbGrp: %d\n", cellGrpPrmsGpu->nPrbGrp);
    printf("nBsAnt: %d\n", cellGrpPrmsGpu->nBsAnt);
    printf("nUeAnt: %d\n", cellGrpPrmsGpu->nUeAnt);
    printf("W: %f\n", cellGrpPrmsGpu->W);
    printf("sigmaSqrd: %4.3e\n", cellGrpPrmsGpu->sigmaSqrd);
    printf("allocType: %d\n", cellGrpPrmsGpu->allocType);
    printf("precodingScheme: %d\n", cellGrpPrmsGpu->precodingScheme);
    printf("receiverScheme: %d\n", cellGrpPrmsGpu->receiverScheme);
    printf("betaCoeff: %f\n", cellGrpPrmsGpu->betaCoeff);
    printf("columnMajor: %d\n", columnMajor);
    
    int16_t*   allocSol;
    uint16_t*  cellId;
    uint8_t*   cellAssoc;

    CUDA_CHECK_ERR(cudaMallocHost((void **)&cellId, cidSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&allocSol, gpuAllocSolSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&cellAssoc, assocSize)); 

    CUDA_CHECK_ERR(cudaMemcpy(allocSol, schdSolGpu->allocSol, gpuAllocSolSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellAssoc, cellGrpPrmsGpu->cellAssoc, assocSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellId, cellGrpPrmsGpu->cellId, cidSize, cudaMemcpyDeviceToHost));

    if (cellGrpPrmsGpu->allocType) { // allocation type-1
      for (int cIdx = 0; cIdx < cellGrpPrmsGpu->nCell; cIdx++) {
        printf("*********** cell %d ***********\n", cellId[cIdx]);
        printf(" Associated UEs: \n");
        for (int uIdx = 0; uIdx < cellGrpPrmsGpu->nUe; uIdx++) {
            if (cellAssoc[cIdx*cellGrpPrmsGpu->nUe + uIdx]) {
               printf("(%d) ", uIdx);
            }
        }
        printf("\n");

        printf(" allocSol: \n");
        for (int uIdx = 0; uIdx < cellGrpPrmsGpu->nUe; uIdx++) {
            if (cellAssoc[cIdx*cellGrpPrmsGpu->nUe + uIdx]) {
               printf(" UE %d: (%d, %d)", uIdx, allocSol[2*uIdx], allocSol[2*uIdx+1]);
            }
        }
        printf("\n");
      }
    } else { // allocation type-0
      for (int cIdx = 0; cIdx < cellGrpPrmsGpu->nCell; cIdx++) {
        printf("*********** cell %d ***********\n", cellId[cIdx]);
        printf(" Associated UEs: \n");
        for (int uIdx = 0; uIdx < cellGrpPrmsGpu->nUe; uIdx++) {
            if (cellAssoc[cIdx*cellGrpPrmsGpu->nUe + uIdx]) {
               printf("(%d) ", uIdx);
            }
        }
        printf("\n");
        
        printf(" allocSol: \n");
        for (int prgIdx = 0; prgIdx < cellGrpPrmsGpu->nPrbGrp; prgIdx++) {
            printf(" PRG %d: (%d)", prgIdx, allocSol[prgIdx*cellGrpPrmsGpu->nCell + cellId[cIdx]]);
        }
        printf("\n");
      }
    }
    
    if (cellAssoc) CUDA_CHECK_ERR(cudaFreeHost(cellAssoc));
    if (allocSol) CUDA_CHECK_ERR(cudaFreeHost(allocSol));
    if (cellId) CUDA_CHECK_ERR(cudaFreeHost(cellId));
 }
}