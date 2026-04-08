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

 #include "network.h"
 #include "../src/api.h"
 #include "../src/cumac.h"

// cuMAC namespace
namespace cumac {

 network::network(uint8_t in_DL, uint8_t extSchedulerType, uint8_t fixCellAssoc, uint8_t fastFadingType, bool en_traffic_gen, cudaStream_t strm) : m_en_traffic_gen(en_traffic_gen)
 {
    /* Initialization */
    // determine MIMO mode
    if (nBsAntConst == 4 && nUeAntConst == 4) {
        mimoMode = 0;
    } else if (nBsAntConst == 64 && nUeAntConst == 4) {
        mimoMode = 1;
    } else {
        throw std::runtime_error("Error: invalid MIMO mode for the simulation. Currently only support 4TR or 64TR");
    }

    // specify scheduler type
    schedulerType           = extSchedulerType; // 0 - single-cell scheduler, 1 - multi-cell scheduler
    DL                      = in_DL; // 1 for DL, 0 for UL
    heteroUeSelCells        = heteroUeSelCellsConst;

    // specify whether to use fixed cell association
    fixCellAssociation      = fixCellAssoc;

    // 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)
    m_fastFadingType = fastFadingType;

    // initialize parameters /////////////////////////////////////
    totNumCell             = numCellConst; // total number of cells in the network, including coordinated cells and interfering cells
    nUe                    = totNumUesConst; // total number of scheduled UEs per TTI in the coordinated cells
    nActiveUe              = totNumActiveUesConst; // total number of active UEs in the coordinated cells
    numUeSchdPerCellTTI    = numUePerCellConst; // number of UEs scheduled per TTI per cell
    numUeForGrpPerCell     = numUeForGrpConst; // number of UEs considered for MU-MIMO UE grouping per TTI per cell
    nMaxActUePerCell       = maxNumActUePerCell_; // maximum number of active UEs per cell currently supported
    nCell                  = numCellConst; // number of coordinated cells
    nInterfCell            = totNumCell - nCell;
    nPrbGrp                = nPrbGrpsConst;
    nPrbPerGrp             = nPrbsPerGrpConst;
    nBsAnt                 = nBsAntConst;
    bsAntSize              = bsAntSizeConst;
    bsAntSpacing           = bsAntSpacingConst;
    bsAntPolarAngles       = bsAntPolarAnglesConst;
    bsAntPattern           = bsAntPatternConst;
    nUeAnt                 = nUeAntConst;
    ueAntSize              = ueAntSizeConst;
    ueAntSpacing           = ueAntSpacingConst;
    ueAntPolarAngles       = ueAntPolarAnglesConst;
    ueAntPattern           = ueAntPatternConst;
    vDirection             = vDirectionConst;
    W                      = WConst;
    slotDuration           = slotDurationConst;
    Pt_Rbg                 = PtRbgConst;
    Pt_rbgAnt              = PtRbgAntConst;
    precodingScheme        = prdSchemeConst;
    receiverScheme         = rxSchemeConst;
    gpuAllocType           = gpuAllocTypeConst;
    cpuAllocType           = cpuAllocTypeConst;
    betaCoeff              = betaCoeffConst;
    sinValThr              = sinValThrConst;
    prioWeightStep         = prioWeightStepConst;
    netData                = std::make_unique<networkData>();
    // muCoeff set to default value 
    // srsSnrThr set to default value 
    // chanCorrThr set to default value 

    // randomness seed
    seed = std::chrono::system_clock::now().time_since_epoch().count();
    randomEngine = std::default_random_engine(seed);
    uniformRealDist = std::uniform_real_distribution<float>(0.0,1.0);
    // PF scheduling
     pfAvgRateUpd           = pfAvgRateUpdConst;
     initAvgRate            = initAvgRateConst;
     sumCellThrRecordsCpu   = std::make_unique<float []>(numSimChnRlz);
     sumCellThrRecordsGpu   = std::make_unique<float []>(numSimChnRlz);
     sumInsThrRecordsCpu    = std::make_unique<float []>(numSimChnRlz);
     sumInsThrRecordsGpu    = std::make_unique<float []>(numSimChnRlz);
     sumCellPfRecordsCpu    = std::make_unique<float []>(numSimChnRlz);
     sumCellPfRecordsGpu    = std::make_unique<float []>(numSimChnRlz);

    // SNR-BLER tables for all MCS levels
     snrMcsArr.push_back(snrMcs0);
     snrMcsArr.push_back(snrMcs1);
     snrMcsArr.push_back(snrMcs2);
     snrMcsArr.push_back(snrMcs3);
     snrMcsArr.push_back(snrMcs4);
     snrMcsArr.push_back(snrMcs5);
     snrMcsArr.push_back(snrMcs6);
     snrMcsArr.push_back(snrMcs7);
     snrMcsArr.push_back(snrMcs8);
     snrMcsArr.push_back(snrMcs9);
     snrMcsArr.push_back(snrMcs10);
     snrMcsArr.push_back(snrMcs11);
     snrMcsArr.push_back(snrMcs12);
     snrMcsArr.push_back(snrMcs13);
     snrMcsArr.push_back(snrMcs14);
     snrMcsArr.push_back(snrMcs15);
     snrMcsArr.push_back(snrMcs16);
     snrMcsArr.push_back(snrMcs17);
     snrMcsArr.push_back(snrMcs18);
     snrMcsArr.push_back(snrMcs19);
     snrMcsArr.push_back(snrMcs20);
     snrMcsArr.push_back(snrMcs21);
     snrMcsArr.push_back(snrMcs22);
     snrMcsArr.push_back(snrMcs23);
     snrMcsArr.push_back(snrMcs24);
     snrMcsArr.push_back(snrMcs25);
     snrMcsArr.push_back(snrMcs26);
     snrMcsArr.push_back(snrMcs27);
    
     blerMscArr.push_back(blerMcs0);
     blerMscArr.push_back(blerMcs1);
     blerMscArr.push_back(blerMcs2);
     blerMscArr.push_back(blerMcs3);
     blerMscArr.push_back(blerMcs4);
     blerMscArr.push_back(blerMcs5);
     blerMscArr.push_back(blerMcs6);
     blerMscArr.push_back(blerMcs7);
     blerMscArr.push_back(blerMcs8);
     blerMscArr.push_back(blerMcs9);
     blerMscArr.push_back(blerMcs10);
     blerMscArr.push_back(blerMcs11);
     blerMscArr.push_back(blerMcs12);
     blerMscArr.push_back(blerMcs13);
     blerMscArr.push_back(blerMcs14);
     blerMscArr.push_back(blerMcs15);
     blerMscArr.push_back(blerMcs16);
     blerMscArr.push_back(blerMcs17);
     blerMscArr.push_back(blerMcs18);
     blerMscArr.push_back(blerMcs19);
     blerMscArr.push_back(blerMcs20);
     blerMscArr.push_back(blerMcs21);
     blerMscArr.push_back(blerMcs22);
     blerMscArr.push_back(blerMcs23);
     blerMscArr.push_back(blerMcs24);
     blerMscArr.push_back(blerMcs25);
     blerMscArr.push_back(blerMcs26);
     blerMscArr.push_back(blerMcs27);

     // PDSCH parameters
     m_pdschNrOfDataSymb = pdschNrOfDataSymb;
    
     // initialize sizes /////////////////////////////////////////
     hSize                  = sizeof(cuComplex)*nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst;
     hHalfSize              = sizeof(__nv_bfloat162)*nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst;
     hActUePrdSize          = sizeof(cuComplex)*totNumActiveUesConst*nPrbGrpsConst*nBsAntConst*nUeAntConst;
     hPerUeBufferPtrSize    = sizeof(cuComplex*)*totNumUesConst;
     hPerUeBufferSize       = sizeof(cuComplex)*nPrbGrpsConst*nBsAntConst*nUeAntConst;
     prdActUeSize           = sizeof(cuComplex)*totNumActiveUesConst*nPrbGrpsConst*nBsAntConst*nBsAntConst;
     detActUeSize           = sizeof(cuComplex)*totNumActiveUesConst*nPrbGrpsConst*nBsAntConst*nBsAntConst;
     prdSize                = sizeof(cuComplex)*totNumUesConst*nPrbGrpsConst*nBsAntConst*nBsAntConst;
     detSize                = sizeof(cuComplex)*totNumUesConst*nPrbGrpsConst*nBsAntConst*nBsAntConst;
     sinValActUeSize        = sizeof(float)*totNumActiveUesConst*nPrbGrpsConst*nUeAntConst;

     // initialize buffers ///////////////////////////////////////
     estH_fr                = std::make_unique<cuComplex []>(nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst);
     estH_fr_actUe          = std::make_unique<cuComplex []>(nPrbGrpsConst*totNumActiveUesConst*numCellConst*nBsAntConst*nUeAntConst);
     estH_fr_perUeBuffer    = std::make_unique<std::unique_ptr<cuComplex []> []>(totNumUesConst);
     estH_fr_perUeBufferGpu = std::make_unique<cuComplex* []>(totNumUesConst);
     prdMat                 = std::make_unique<cuComplex []>(totNumUesConst*nPrbGrpsConst*nBsAntConst*nBsAntConst);
     detMat                 = std::make_unique<cuComplex []>(totNumUesConst*nPrbGrpsConst*nBsAntConst*nBsAntConst);
     sinVal                 = std::make_unique<float []>(totNumUesConst*nPrbGrpsConst*nUeAntConst);
     sortedUeListGpu        = std::make_unique<uint16_t* []>(numCellConst);
     muGrpListGpu           = std::make_unique<multiCellMuGrpList>();

     for (int uIdx = 0; uIdx < totNumUesConst; uIdx++) {
         estH_fr_perUeBuffer[uIdx] = std::make_unique<cuComplex []>(nPrbGrpsConst*nBsAntConst*nUeAntConst);
     }
    
     hActUeSize             = sizeof(cuComplex)*nPrbGrpsConst*totNumActiveUesConst*numCellConst*nBsAntConst*nUeAntConst;
     cidSize                = sizeof(uint16_t)*numCellConst;
     assocSize              = sizeof(uint8_t)*numCellConst*totNumUesConst; 
     assocActUeSize         = sizeof(uint8_t)*numCellConst*totNumActiveUesConst;
     arSize                 = sizeof(float)*totNumUesConst;
     arActUeSize            = sizeof(float)*totNumActiveUesConst;
     tbeActUeSize           = sizeof(int8_t)*totNumActiveUesConst;
     ndActUeSize            = sizeof(int8_t)*totNumActiveUesConst;
     prioActUeSize          = sizeof(uint16_t)*totNumActiveUesConst;
     postEqSinrSize         = sizeof(float)*totNumActiveUesConst*nPrbGrpsConst*nUeAntConst;
     wbSinrSize             = sizeof(float)*totNumActiveUesConst*nUeAntConst;
     
     if (mimoMode == 0) { // 4TR
        tbeSize         = sizeof(int8_t)*totNumUesConst;
        mcsSelSolSize   = sizeof(int16_t)*nUe;
        mcsSelSol       = std::make_unique<int16_t []>(nUe);
        for (int uIdx = 0; uIdx<nUe; uIdx++) {
            mcsSelSol[uIdx] = -1;
        }

        layerSize       = sizeof(uint8_t)*nUe;
        layerSelSol     = std::make_unique<uint8_t []>(nUe);
        sinValSize      = sizeof(float)*totNumUesConst*nPrbGrpsConst*nUeAntConst;
        setSchdUePerCellTTI    = std::make_unique<uint16_t []>(numCellConst*numUeSchdPerCellTTI);
        setSchdUeSolSize       = sizeof(uint16_t)*numCellConst*numUeSchdPerCellTTI;
     } else if (mimoMode == 1) { // 64TR
        tbeSize         = sizeof(int8_t)*nActiveUe;
        mcsSelSolSize   = sizeof(int16_t)*nActiveUe;
        mcsSelSol       = std::make_unique<int16_t []>(nActiveUe);
        for (int uIdx = 0; uIdx<nActiveUe; uIdx++) {
            mcsSelSol[uIdx] = -1;
        }

        layerSize       = sizeof(uint8_t)*nActiveUe;
        layerSelSol     = std::make_unique<uint8_t []>(nActiveUe);
        sinValSize      = sizeof(float)*totNumActiveUesConst*nPrbGrpsConst*nUeAntConst;
        srsEstChan      = std::make_unique<cuComplex* []>(numCellConst);
        srsUeMap        = std::make_unique<int32_t* []>(numCellConst);
        setSchdUePerCellTTI    = std::make_unique<uint16_t []>(numCellConst*numUeForGrpPerCell);
        setSchdUeSolSize       = sizeof(uint16_t)*numCellConst*numUeForGrpPerCell;
     }
     
     numUeSchdArrSize       = sizeof(uint8_t)*numCellConst;
     srsWbSnrSize           = sizeof(float)*totNumActiveUesConst;
     nSCIDSize              = sizeof(uint8_t)*totNumActiveUesConst;
     muMimoIndSize          = sizeof(uint8_t)*totNumActiveUesConst;

     currSlotIdxPerCell     = std::make_unique<uint32_t []>(numCellConst);
     cellId                 = std::make_unique<uint16_t []>(numCellConst);
     avgRates               = std::make_unique<float []>(totNumUesConst);
     avgRatesActUe          = std::make_unique<float []>(totNumActiveUesConst);
     newDataActUe           = std::make_unique<int8_t []>(totNumActiveUesConst);
     std::fill(newDataActUe.get(), newDataActUe.get()+totNumActiveUesConst, 1);
     tbErrLast              = std::make_unique<int8_t []>(totNumUesConst);
     tbErrLastActUe         = std::make_unique<int8_t []>(totNumActiveUesConst);
     prioWeightActUe        = std::make_unique<uint16_t []>(totNumActiveUesConst);
     // Each element of cellAssoc stores the cell index (not cell ID) for a UE
     cellAssoc              = std::make_unique<uint8_t []>(numCellConst*totNumUesConst); 
     cellAssocActUe         = std::make_unique<uint8_t []>(numCellConst*totNumActiveUesConst);
     postEqSinr             = std::make_unique<float []>(totNumActiveUesConst*nPrbGrpsConst*nUeAntConst);
     wbSinr                 = std::make_unique<float []>(totNumActiveUesConst*nUeAntConst);
     
     if (heteroUeSelCells == 1) { // heterogeneous UE selection config. across cells
        numUeSchdPerCellTTIArr = std::make_unique<uint8_t []>(numCellConst);
        for (int cIdx = 0; cIdx < numCellConst; cIdx++) {
            numUeSchdPerCellTTIArr[cIdx] = 6 + cIdx % 5;
        }
     }
     
     for (int uIdx = 0; uIdx < totNumUesConst; uIdx++) {
        avgRates[uIdx] = initAvgRate;
        tbErrLast[uIdx] = -1;
     }

     for (int uIdx = 0; uIdx < totNumActiveUesConst; uIdx++) {
        avgRatesActUe[uIdx] = initAvgRate;
        tbErrLastActUe[uIdx] = -1;
        prioWeightActUe[uIdx] = 0;
     }

     // assignment of coordinated cell IDs may be changed
     for (int cIdx = 0; cIdx < numCellConst; cIdx++) {
        cellId[cIdx] = cIdx;
     }

     // sanity check
     if (mimoMode == 1 && gpuAllocType == 0) { // does not support type-0 allocation for 64TR
        throw std::runtime_error("Error: invalid MIMO mode and allocation type for the simulation. Currently only support type-1 allocation for 64TR");
     }

     floatRandomArr = std::make_unique<float[]>(nUe);

     netData->scenarioUma = 1; // UMa
     netData->carrierFreq = 2.5; // carrier frequency in unit of GHz
     netData->sfStd = 3.0; // shadow fading STD in dB
     netData->noiseFloor = -98.9; // dBm
     netData->bsHeight = 25;
     netData->ueHeight = 1.5;
     netData->bsAntDownTilt = 102.0; // degree
     netData->GEmax = 9.0; // dBi
     netData->cellRadius = cellRadiusConst;
     netData->bsTxPower = 49.0; // dBm, 79.4328 W
     netData->bsTxPower_perAntPrg = netData->bsTxPower - 10.0*log10(static_cast<float>(nBsAnt)*static_cast<float>(nPrbGrp)); // dBm
     netData->ueTxPower = 23.0; // dBm, 79.4328 W
     netData->ueTxPower_perAntPrg = netData->ueTxPower - 10.0*log10(static_cast<float>(nUeAnt)*static_cast<float>(nPrbGrp)); // dBm
     netData->numCell = nCell; // assume all cells are coordinated cells
     netData->numActiveUesPerCell = numActiveUePerCellConst;
     netData->sectorOrien[0] = M_PI/3.0;
     netData->sectorOrien[1] = M_PI;
     netData->sectorOrien[2] = M_PI*5.0/3.0;
     netData->minD2Bs = 30;
     netData->rho = 0.9938; // for 2.5 GHz carrier frequency, 1 ms time slot duration, 3 m/s UE moving speed
     netData->rhoPrime = 0.0786; // sqrt(1-0.9938^2)*sqrt(0.5), for 2.5 GHz carrier frequency, 1 ms time slot duration, 3 m/s UE moving speed

     netData->bsPos.resize(netData->numCell);
     netData->rxSigPowDB = std::make_unique<float []>(netData->numCell*netData->numCell*netData->numActiveUesPerCell);
     netData->rxSigPowDB_UL = std::make_unique<float []>(netData->numCell*netData->numCell*netData->numActiveUesPerCell);
     netData->uePos.resize(netData->numCell*netData->numActiveUesPerCell);
     netData->avgRatesGPU = std::make_unique<float []>(netData->numCell*netData->numActiveUesPerCell);
     netData->avgRatesCPU = std::make_unique<float []>(netData->numCell*netData->numActiveUesPerCell);
     for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
        netData->bsPos[cIdx].resize(3);
        for (int uIdx = 0; uIdx < netData->numActiveUesPerCell; uIdx++) {
            netData->uePos[cIdx*netData->numActiveUesPerCell + uIdx].resize(3);
            netData->avgRatesGPU[cIdx*netData->numActiveUesPerCell + uIdx] = 1.0;
            netData->avgRatesCPU[cIdx*netData->numActiveUesPerCell + uIdx] = 1.0;
        }
     }

     netData->numThrdBlk = nPrbGrp*netData->numCell;
     netData->numThrdPerBlk = nBsAnt*nUeAnt*floor(1024.0/static_cast<float>(nBsAnt*nUeAnt));
     
     CUDA_CHECK_ERR(cudaMalloc((void **)&netData->rxSigPowDBGpu, netData->numCell*netData->numCell*netData->numActiveUesPerCell*sizeof(float)));
     CUDA_CHECK_ERR(cudaMalloc((void **)&netData->rxSigPowDBGpu_UL, netData->numCell*netData->numCell*netData->numActiveUesPerCell*sizeof(float)));
     CUDA_CHECK_ERR(cudaMalloc((void **)&netData->states, netData->numThrdBlk*netData->numThrdPerBlk*sizeof(curandState_t)));
     
     init_curand<<<netData->numThrdBlk, netData->numThrdPerBlk>>>(time(NULL), 0, netData->states);

     CUDA_CHECK_ERR(cudaMalloc((void **)&estH_fr_GPUforCpuSchd, hSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&estH_fr_GPUforCpuSchd_half, hHalfSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&setSchdUePerCellTTIGpuforCpuSchd, setSchdUeSolSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&avgRatesGpuforCpuSchd, arSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellAssocGpuforCpuSchd, assocSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&avgRatesActUeGpuforCpuSchd, arActUeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&tbErrLastGpuforCpuSchd, tbeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&tbErrLastActUeGpuforCpuSchd, tbeActUeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&prdMatGpuforCpuSchd, prdSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&detMatGpuforCpuSchd, detSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&sinValGpuforCpuSchd, sinValSize));

     if (fixCellAssociation) {
        for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
            for (int uIdx = 0; uIdx < totNumUesConst; uIdx++) {
                int cellIdx = floor(static_cast<float>(uIdx)/static_cast<float>(numUeSchdPerCellTTI));
                if (cellIdx == cIdx)
                    cellAssoc[cIdx*totNumUesConst + uIdx] = 1;
                else
                    cellAssoc[cIdx*totNumUesConst + uIdx] = 0;
            }

            for (int uIdx = 0; uIdx < totNumActiveUesConst; uIdx++) {
                int cellIdx = floor(static_cast<float>(uIdx)/static_cast<float>(netData->numActiveUesPerCell));
                if (cellIdx == cIdx)
                    cellAssocActUe[cIdx*totNumActiveUesConst + uIdx] = 1;
                else
                    cellAssocActUe[cIdx*totNumActiveUesConst + uIdx] = 0;
            }
        }
     }

     // noise variance
     sigma     = pow(10.0, ((netData->noiseFloor - 30.0)/20.0));
     sigmaSqrd = pow(AFTER_SCALING_SIGMA_CONST, 2.0); // after scaling
     scalingFactor = AFTER_SCALING_SIGMA_CONST/sigma;

     // MCS selection records
     mcsSelRecordsCpu       = std::make_unique<std::unique_ptr<int16_t []> []>(nUe);
     tbErrRecordsCpu        = std::make_unique<std::unique_ptr<int8_t []> []>(nUe);
     mcsSelRecordsGpu       = std::make_unique<std::unique_ptr<int16_t []> []>(nUe);
     tbErrRecordsGpu        = std::make_unique<std::unique_ptr<int8_t []> []>(nUe);
     layerSelRecordsCpu     = std::make_unique<std::unique_ptr<uint8_t []> []>(nUe);
     layerSelRecordsGpu     = std::make_unique<std::unique_ptr<uint8_t []> []>(nUe);
     for (int ueIdx = 0; ueIdx < nUe; ueIdx++) {
        mcsSelRecordsCpu[ueIdx] = std::make_unique<int16_t []>(numSimChnRlz);
        tbErrRecordsCpu[ueIdx] = std::make_unique<int8_t []>(numSimChnRlz);
        mcsSelRecordsGpu[ueIdx] = std::make_unique<int16_t []>(numSimChnRlz);
        tbErrRecordsGpu[ueIdx] = std::make_unique<int8_t []>(numSimChnRlz);
        layerSelRecordsCpu[ueIdx] = std::make_unique<uint8_t []>(numSimChnRlz);
        layerSelRecordsGpu[ueIdx] = std::make_unique<uint8_t []>(numSimChnRlz);
     }

     if (nInterfCell > 0) {
        interfCellId = std::make_unique<uint16_t []>(nInterfCell);

        uint16_t interfCellIdx = 0;
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {

            bool coordinatedCell = false;
            for (int j = 0; j<nCell; j++) {
                if (cIdx == cellId[j]) {
                    coordinatedCell = true;
                    break;
                }
            }

            if (coordinatedCell)
                continue;

            interfCellId[interfCellIdx] = cIdx;
            interfCellIdx++;

            if (interfCellIdx == nInterfCell) 
                break;
        }
     }

     if (gpuAllocType == 1) {
        if (mimoMode == 0) { // 4TR
            gpuAllocSolSize = sizeof(int16_t)*2*nUe;
            allocSol = std::make_unique<int16_t []>(2*nUe);
            allocSolPrb = std::make_unique<int16_t []>(2*nUe);
            for (int uIdx = 0; uIdx<nUe; uIdx++) {
                allocSol[2*uIdx]   = -1;
                allocSol[2*uIdx+1] = -1;
                allocSolPrb[2*uIdx]   = -1;
                allocSolPrb[2*uIdx+1] = -1;
            }
        } else { // 64TR
            gpuAllocSolSize = sizeof(int16_t)*2*nActiveUe;
            allocSol = std::make_unique<int16_t []>(2*nActiveUe);
            allocSolPrb = std::make_unique<int16_t []>(2*nActiveUe);
            for (int uIdx = 0; uIdx<nActiveUe; uIdx++) {
                allocSol[2*uIdx]   = -1;
                allocSol[2*uIdx+1] = -1;
                allocSolPrb[2*uIdx]   = -1;
                allocSolPrb[2*uIdx+1] = -1;
            }
        }
     } else {
        gpuAllocSolSize = sizeof(int16_t)*totNumCell*nPrbGrp;
        allocSol = std::make_unique<int16_t []>(totNumCell*nPrbGrp);
        allocSolPrb = std::make_unique<int16_t []>(totNumCell*nPrbGrp);
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol[rbgIdx*totNumCell + cIdx] = -1;
                allocSolPrb[rbgIdx*totNumCell + cIdx] = -1;
            }
        }
     }   

     if (cpuAllocType) {
        cpuAllocSolSize = sizeof(int16_t)*2*nUe;
     } else {
        cpuAllocSolSize = sizeof(int16_t)*totNumCell*nPrbGrp;
     }

     if (schedulerType == 0) { // single-cell scheduling
        throw std::runtime_error("Error: single-cell scheduling option is no longer maintained");
        printf("Single-cell scheduler, Type-%d allocate\n", gpuAllocType);
        if (gpuAllocType) {
            uint16_t pfSize = nPrbGrp*numUeSchdPerCellTTI;
            uint16_t pow2N = 2;
 
            while(pow2N<pfSize) {
                pow2N = pow2N << 1;
            }

            pfMetricSize = sizeof(float)*pow2N;
            pfIdSize     = sizeof(uint16_t)*pow2N;
        } else {
            pfMetricSize = 0;
            pfIdSize = 0;
        }
     } else {
        printf("Multi-cell scheduler, Type-%d allocate\n", gpuAllocType);
        if (gpuAllocType) {
            uint16_t pfSize = nPrbGrp*numUeSchdPerCellTTI;
            uint16_t pow2N = 2;
 
            while(pow2N<pfSize) {
                pow2N = pow2N << 1;
            }

            pfMetricSize = sizeof(float)*nCell*pow2N;
            pfIdSize     = sizeof(uint16_t)*nCell*pow2N;
        } else {
            pfMetricSize = 0;
            pfIdSize = 0;
        }
     }

     if (precodingScheme == 0) {
        printf("No precoding + MMSE-IRC\n");
     } else {
        printf("SVD precoding + MMSE-IRC\n");
     }

     printf("nBsAnt X nUeAnt = %d X %d\n", nBsAnt, nUeAnt);

     // initialize execution status ///////////////////////////////
     execStatus = std::make_unique<cumacExecStatus>();
     execStatus.get()->channelRenew       = true;
     execStatus.get()->cellIdRenew        = true;
     execStatus.get()->cellAssocRenew     = true;
     execStatus.get()->averageRateRenew   = true;
    
    // CPU matrix operation algorithms
     matAlg = std::make_unique<cpuMatAlg>();

    // GPU TDL config, class, ptr to CFR
    if(m_fastFadingType == 1 || m_fastFadingType == 2) {
        m_tdlCfg = std::make_unique<tdlConfig_t>(); 
        // default TDLA30-5-Low, CFO 200
        m_tdlCfg->useSimplifiedPdp = true; // true for simplified pdp in 38.141, false for 38.901
        m_tdlCfg->delayProfile     = 'A';
        m_tdlCfg->delaySpread      = 30;
        m_tdlCfg->maxDopplerShift  = 5;
        m_tdlCfg->cfoHz            = 200.0f;
        // network size, TODO: currently TDL generates CFR for all UEs and all cells
        m_tdlCfg->nCell            = totNumCell;
        m_tdlCfg->nUe              = nActiveUe;
        m_tdlCfg->nBsAnt           = nBsAnt;
        m_tdlCfg->nUeAnt           = nUeAnt;
        m_tdlCfg->sigLenPerAnt = 0; // no need to process input signals
        m_tdlCfg->N_sc             = nPrbGrp * 12 * nPrbPerGrp; // total number of Scs
        m_tdlCfg->N_sc_Prbg        = nPrbPerGrp * 12; // number of Scs per Prbg
        m_tdlCfg->runMode          = m_fastFadingType; 
        // runMode 0: TDL time channel
        // runMode 1: TDL time channel and frequency channel on Prbg only
        // runMode 2: TDL time channel and frequency channel on Sc and Prbg
        m_tdl_chan = std::make_unique<tdlChan<float, cuComplex>>(m_tdlCfg.get() /*tdlConfig_t*/, 0 /*random seed*/, strm /*cudaStream_t*/);
        printf("GPU TDL: TDL%c%d-%d-Low, CFO %d Hz, using mode %s\n", m_tdlCfg->delayProfile, int(m_tdlCfg->delaySpread), int(m_tdlCfg->maxDopplerShift), int(m_tdlCfg->cfoHz), (m_fastFadingType == 1) ? "1 (CFR on Prbg only)" : "2 (CFR on Sc and Prbg)");
    }
    else if(m_fastFadingType == 3 || m_fastFadingType == 4) {
        // GPU CDL config, class, ptr to CFR
        m_cdlCfg = std::make_unique<cdlConfig_t>();
        // check antenna size configuration
        assert(nBsAnt == std::accumulate(bsAntSize.begin(), bsAntSize.end(), 1U, std::multiplies<uint32_t>()));
        assert(nUeAnt == std::accumulate(ueAntSize.begin(), ueAntSize.end(), 1U, std::multiplies<uint32_t>()));
        // default CDLA30-5-Low
        m_cdlCfg->delayProfile = 'A';
        m_cdlCfg->delaySpread = 30;
        m_cdlCfg->maxDopplerShift = 10;
        m_cdlCfg->cfoHz = 200.0f;
        m_cdlCfg->numRay = 20;
        // network size, TODO: currently CDL generates CFR for all UEs and all cells
        m_cdlCfg->nCell = totNumCell;
        m_cdlCfg->nUe = nActiveUe;
        m_cdlCfg->bsAntSize = bsAntSize;
        m_cdlCfg->bsAntSpacing = bsAntSpacing;
        m_cdlCfg->bsAntPolarAngles = bsAntPolarAngles;
        m_cdlCfg->bsAntPattern = bsAntPattern;
        m_cdlCfg->ueAntSize = ueAntSize;
        m_cdlCfg->ueAntSpacing = ueAntSpacing;
        m_cdlCfg->ueAntPolarAngles = ueAntPolarAngles;
        m_cdlCfg->ueAntPattern = ueAntPattern;
        m_cdlCfg->vDirection = vDirection;
        m_cdlCfg->sigLenPerAnt = 0; // no need to process input signals
        m_cdlCfg->N_sc = nPrbGrp * 12 * nPrbPerGrp; // total number of Scs
        m_cdlCfg->N_sc_Prbg = nPrbPerGrp * 12; // number of Scs per Prbg
        m_cdlCfg->runMode          = m_fastFadingType - 2;
        // runMode 0: CDL time channel
        // runMode 1: CDL time channel and frequency channel on Prbg only
        // runMode 2: CDL time channel and frequency channel on Sc and Prbg
        m_cdl_chan = std::make_unique<cdlChan<float, cuComplex>>(m_cdlCfg.get() /*cdlConfig_t*/, 0, strm);
        printf("GPU CDL: CDL%c%d-%d, CFO %d Hz, using mode %s\n", m_cdlCfg->delayProfile, int(m_cdlCfg->delaySpread), int(m_cdlCfg->maxDopplerShift), int(m_cdlCfg->cfoHz),  (m_fastFadingType == 3) ? "1 (CFR on Prbg only)" : "2 (CFR on Sc and Prbg)");
    }

    // GPU address of CFR on Prg for all cell and UE if TDL or CDL is used
    // getFreqChanPrbg() returns the GPU address of CFR on Prg for all cell and UE, a 1D array: nCell * nUE * nUeAnt * nBsAnt * nPrg
    if (m_fastFadingType == 1 || m_fastFadingType == 2) { // GPU TDL
        m_externFastFadingPrbgPtr = m_tdl_chan->getFreqChanPrbg();
        m_externFastFadingScPtr = m_tdl_chan->getFreqChanSc();
    } else if (m_fastFadingType == 3 || m_fastFadingType == 4) { // GPU CDL
        m_externFastFadingPrbgPtr = m_cdl_chan->getFreqChanPrbg();
        m_externFastFadingScPtr = m_cdl_chan->getFreqChanSc();
    } else {
        m_externFastFadingPrbgPtr = nullptr;
        m_externFastFadingScPtr = nullptr;
    }
 }

 network::~network()
 {
     CUDA_CHECK_ERR(cudaFree(estH_fr_GPUforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(estH_fr_GPUforCpuSchd_half));
     CUDA_CHECK_ERR(cudaFree(setSchdUePerCellTTIGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(avgRatesGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(avgRatesActUeGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(tbErrLastGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(tbErrLastActUeGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(cellAssocGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(prdMatGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(sinValGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(detMatGpuforCpuSchd));
     CUDA_CHECK_ERR(cudaFree(netData->rxSigPowDBGpu));
     CUDA_CHECK_ERR(cudaFree(netData->rxSigPowDBGpu_UL));
     CUDA_CHECK_ERR(cudaFree(netData->states));
 }

 void network::createAPI()
 {
     // create API structures /////////////////////////////////
     cellGrpUeStatusGpu = std::make_unique<cumacCellGrpUeStatus>();
     schdSolGpu         = std::make_unique<cumacSchdSol>();
     cellGrpPrmsGpu     = std::make_unique<cumacCellGrpPrms>();

     // create API for CPU scheduler //////////////////////////
     cellGrpUeStatusCpu = std::make_unique<cumacCellGrpUeStatus>();
     schdSolCpu         = std::make_unique<cumacSchdSol>();
     cellGrpPrmsCpu     = std::make_unique<cumacCellGrpPrms>();

     // create cuMAC simulation parameter structure ///////////
     simParam           = std::make_unique<cumacSimParam>();

     //GPU mem allocate ///////////////////////////////////////
     if (mimoMode == 0) { // 4TR
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->avgRates, arSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prdMat, prdSize));
     } else { // 64TR 
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prdMat, sizeof(cuComplex)*numCellConst*nPrbGrpsConst*nBsAntConst*maxNumLayerPerGrpDL_));  
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsWbSnr, srsWbSnrSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->muGrpList, sizeof(multiCellMuGrpList)));   
        auto muGrpListCpu = std::make_unique<multiCellMuGrpList>();
        CUDA_CHECK_ERR(cudaMalloc((void **)&muGrpListGpu->numUeInGrp, sizeof(uint16_t)*maxNumCoorCells_*maxNumUegPerCell_));
        CUDA_CHECK_ERR(cudaMalloc((void **)&muGrpListGpu->ueId, sizeof(uint16_t)*maxNumCoorCells_*maxNumUegPerCell_*maxNumLayerPerGrpDL_));
        CUDA_CHECK_ERR(cudaMalloc((void **)&muGrpListGpu->subbandId, sizeof(int16_t)*maxNumCoorCells_*maxNumUegPerCell_));
        CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->muGrpList, muGrpListGpu.get(), sizeof(multiCellMuGrpList), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->ueOrderInGrp, sizeof(uint16_t)*totNumActiveUesConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->muMimoInd, muMimoIndSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->sortedUeList, sizeof(uint16_t*)*numCellConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->nSCID, nSCIDSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsEstChan, sizeof(cuComplex*)*numCellConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsUeMap, sizeof(int32_t*)*numCellConst));
        for (int cIdx = 0; cIdx < numCellConst; cIdx++) {
            CUDA_CHECK_ERR(cudaMalloc((void **)&sortedUeListGpu[cIdx], sizeof(uint16_t)*nMaxActUePerCell));
            CUDA_CHECK_ERR(cudaMalloc((void **)&srsEstChan[cIdx], sizeof(cuComplex)*numCellConst*numUeForGrpConst*nPrbGrpsConst*nUeAntConst*nBsAntConst));
            CUDA_CHECK_ERR(cudaMalloc((void **)&srsUeMap[cIdx], sizeof(int32_t)*totNumActiveUesConst));
        }
        CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->sortedUeList, sortedUeListGpu.get(), sizeof(uint16_t*)*numCellConst, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->srsEstChan, srsEstChan.get(), sizeof(cuComplex*)*numCellConst, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->srsUeMap, srsUeMap.get(), sizeof(int32_t*)*numCellConst, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->allocSolLastTx, sizeof(int16_t)*2*totNumActiveUesConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->mcsSelSolLastTx, sizeof(int16_t)*totNumActiveUesConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->layerSelSolLastTx, sizeof(uint8_t)*totNumActiveUesConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->beamformGainCurrTx, sizeof(float)*totNumActiveUesConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->bfGainPrgCurrTx, sizeof(float)*totNumActiveUesConst*nPrbGrpsConst));
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->beamformGainLastTx, sizeof(float)*totNumActiveUesConst));   
        auto rsrpTx = std::make_unique<float []>(totNumActiveUesConst);
        for (int uIdx = 0; uIdx < totNumActiveUesConst; uIdx++) {
            rsrpTx[uIdx] = -1000.0;
        }
     }
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->lastSchdSlotActUe, sizeof(uint32_t)*totNumActiveUesConst));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->tbErrLast, tbeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->newDataActUe, ndActUeSize));
     CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->newDataActUe, newDataActUe.get(), ndActUeSize, cudaMemcpyHostToDevice));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->avgRatesActUe, arActUeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->tbErrLastActUe, tbeActUeSize));
     // TODO - size works, but should be defined in type(bufferSize) i.e. uint32 rather than float
     if(m_en_traffic_gen){
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->bufferSize, arActUeSize));
     }
     CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->allocSol, gpuAllocSolSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->mcsSelSol, mcsSelSolSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->layerSelSol, layerSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->setSchdUePerCellTTI, setSchdUeSolSize));
     if (gpuAllocType) {
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->pfMetricArr, pfMetricSize));
        CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->pfIdArr, pfIdSize));
     } else {
        schdSolGpu->pfMetricArr = nullptr;
        schdSolGpu->pfIdArr = nullptr; 
     }
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr, hSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr_half, hHalfSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr_actUe_prd, hActUePrdSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr_perUeBuffer, hPerUeBufferPtrSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->estH_fr_actUe, hActUeSize));
     
     for (int uIdx = 0; uIdx < nUe; uIdx++) {
        CUDA_CHECK_ERR(cudaMalloc((void **)&estH_fr_perUeBufferGpu[uIdx], hPerUeBufferSize));
     }
     CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->estH_fr_perUeBuffer, estH_fr_perUeBufferGpu.get(), hPerUeBufferPtrSize, cudaMemcpyHostToDevice));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellId, cidSize));
     if (heteroUeSelCells == 1) { // heterogeneous UE selection config. across cells
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->numUeSchdPerCellTTIArr, numUeSchdArrSize));
     }
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->currSlotIdxPerCell, sizeof(uint32_t)*totNumCell)); 
     std::vector<float> blerTargetActUe(nActiveUe, 0.1);
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->blerTargetActUe, sizeof(float)*nActiveUe));
     CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->blerTargetActUe, blerTargetActUe.data(), sizeof(float)*nActiveUe, cudaMemcpyHostToDevice));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellAssoc, assocSize));   
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellAssocActUe, assocActUeSize));   
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->postEqSinr, postEqSinrSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->wbSinr, wbSinrSize));
     if (fixCellAssociation) {
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->cellAssoc, cellAssoc.get(), assocSize, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->cellAssocActUe, cellAssocActUe.get(), assocActUeSize, cudaMemcpyHostToDevice));
     } 
     
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prdMat_actUe, prdActUeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->sinVal, sinValSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->sinVal_actUe, sinValActUeSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->detMat, detSize));
     CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->detMat_actUe, detActUeSize));

    // CPU mem allocate (or assign pointers) ///////////////////////////////////////
     if (cpuAllocType) {
        schdSolCpu->allocSol                = new int16_t[2*nUe];
     } else {
        schdSolCpu->allocSol                = new int16_t[totNumCell*nPrbGrp];
     }
     schdSolCpu->mcsSelSol                  = new int16_t[nUe];
     if (mimoMode == 0) { // 4TR
        schdSolCpu->setSchdUePerCellTTI        = new uint16_t[totNumCell*numUeSchdPerCellTTI];
     } else { // 64TR   
        schdSolCpu->setSchdUePerCellTTI        = new uint16_t[totNumCell*numUeForGrpPerCell];
     }
     schdSolCpu->layerSelSol                = new uint8_t[nUe];
     cellGrpUeStatusCpu->avgRates           = new float[nUe];
     cellGrpUeStatusCpu->avgRatesActUe      = new float[nActiveUe];
     cellGrpUeStatusCpu->tbErrLast          = new int8_t[nUe];
     cellGrpUeStatusCpu->tbErrLastActUe     = new int8_t[nActiveUe];
     cellGrpUeStatusCpu->prioWeightActUe    = new uint16_t[nActiveUe];
     if(m_en_traffic_gen){
        cellGrpUeStatusCpu->bufferSize         = new uint32_t[nActiveUe];
        memset(cellGrpUeStatusCpu->bufferSize,0,nActiveUe*sizeof(uint32_t));
     }
     cellGrpPrmsCpu->estH_fr                = new cuComplex[nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst];
     //cellGrpPrmsCpu->estH_fr_perUeBuffer    = estH_fr_perUeBuffer;
     //cellGrpPrmsCpu->estH_fr_actUe          = estH_fr_actUe;
     cellGrpPrmsCpu->estH_fr_actUe_prd      = nullptr;
     cellGrpPrmsCpu->cellId                 = cellId.get();
     if (heteroUeSelCells == 1) { // heterogeneous UE selection config. across cells
        cellGrpPrmsCpu->numUeSchdPerCellTTIArr = new uint8_t[totNumCell];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            cellGrpPrmsCpu->numUeSchdPerCellTTIArr[cIdx] = numUeSchdPerCellTTIArr[cIdx];
        }
     }
     cellGrpPrmsCpu->cellAssoc              = new uint8_t[totNumCell*nUe];
     cellGrpPrmsCpu->cellAssocActUe         = new uint8_t[totNumCell*nActiveUe];
     cellGrpPrmsCpu->postEqSinr             = new float[nActiveUe*nPrbGrp*nUeAnt];
     cellGrpPrmsCpu->wbSinr                 = new float[nActiveUe*nUeAnt];
     cellGrpPrmsCpu->blerTargetActUe        = new float[nActiveUe];
     std::memcpy(cellGrpPrmsCpu->blerTargetActUe, blerTargetActUe.data(), nActiveUe * sizeof(float));
     
     if (fixCellAssociation) {
        for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
            for (int uIdx = 0; uIdx < totNumUesConst; uIdx++) {
                int cellIdx = floor(static_cast<float>(uIdx)/static_cast<float>(numUeSchdPerCellTTI));
                if (cellIdx == cIdx)
                    cellGrpPrmsCpu->cellAssoc[cIdx*totNumUesConst + uIdx] = 1;
                else
                    cellGrpPrmsCpu->cellAssoc[cIdx*totNumUesConst + uIdx] = 0;
            }

            for (int uIdx = 0; uIdx < totNumActiveUesConst; uIdx++) {
                int cellIdx = floor(static_cast<float>(uIdx)/static_cast<float>(netData->numActiveUesPerCell));
                if (cellIdx == cIdx)
                    cellGrpPrmsCpu->cellAssocActUe[cIdx*totNumActiveUesConst + uIdx] = 1;
                else
                    cellGrpPrmsCpu->cellAssocActUe[cIdx*totNumActiveUesConst + uIdx] = 0;
            }
        }
     }
     cellGrpPrmsCpu->prdMat       = new cuComplex[nUe*nPrbGrp*nBsAnt*nBsAnt];
     cellGrpPrmsCpu->prdMat_actUe = new cuComplex[nActiveUe*nPrbGrp*nBsAnt*nBsAnt];
     cellGrpPrmsCpu->sinVal       = new float[nUe*nPrbGrp*nUeAnt];
     cellGrpPrmsCpu->sinVal_actUe = new float[nActiveUe*nPrbGrp*nUeAnt];
     cellGrpPrmsCpu->detMat       = new cuComplex[nUe*nPrbGrp*nBsAnt*nBsAnt];
     cellGrpPrmsCpu->detMat_actUe = new cuComplex[nActiveUe*nPrbGrp*nBsAnt*nBsAnt]; 
     for (int uIdx = 0; uIdx < nUe; uIdx++) {
        cellGrpUeStatusCpu->avgRates[uIdx] = initAvgRate;
        cellGrpUeStatusCpu->tbErrLast[uIdx] = -1;
     }

     for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
        cellGrpUeStatusCpu->avgRatesActUe[uIdx] = initAvgRate;
        cellGrpUeStatusCpu->tbErrLastActUe[uIdx] = -1;
        cellGrpUeStatusCpu->prioWeightActUe[uIdx] = 0;
     }
    // set constant parameters
    // GPU
     cellGrpPrmsGpu->dlSchInd              = DL;
     cellGrpPrmsGpu->nUe                   = nUe; // number of scheduled UEs per time slot for all coordinated cells
     cellGrpPrmsGpu->nActiveUe             = nActiveUe; // number of active UEs for all coordinated cells
     cellGrpPrmsGpu->numUeSchdPerCellTTI   = numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
     cellGrpPrmsGpu->nCell                 = nCell; // number of coordinated cells
     cellGrpPrmsGpu->nPrbGrp               = nPrbGrp;
     cellGrpPrmsGpu->nBsAnt                = nBsAnt;
     cellGrpPrmsGpu->nUeAnt                = nUeAnt;
     cellGrpPrmsGpu->W                     = W;
     cellGrpPrmsGpu->sigmaSqrd             = sigmaSqrd;
     cellGrpPrmsGpu->Pt_Rbg                = Pt_Rbg;
     cellGrpPrmsGpu->Pt_rbgAnt             = Pt_rbgAnt;
     cellGrpPrmsGpu->precodingScheme       = precodingScheme;
     cellGrpPrmsGpu->receiverScheme        = receiverScheme;
     cellGrpPrmsGpu->allocType             = gpuAllocType;
     cellGrpPrmsGpu->betaCoeff             = betaCoeff;
     cellGrpPrmsGpu->sinValThr             = sinValThr;
     cellGrpPrmsGpu->prioWeightStep        = prioWeightStep;
     cellGrpPrmsGpu->harqEnabledInd        = 0; // disable HARQ by default

    // CPU
     cellGrpPrmsCpu->dlSchInd              = DL;
     cellGrpPrmsCpu->nUe                   = nUe; // total number of UEs
     cellGrpPrmsCpu->nActiveUe             = nActiveUe; // number of active UEs for all coordinated cells
     cellGrpPrmsCpu->numUeSchdPerCellTTI   = numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
     cellGrpPrmsCpu->nCell                 = nCell; // number of coordinated cells
     cellGrpPrmsCpu->nPrbGrp               = nPrbGrp;
     cellGrpPrmsCpu->nBsAnt                = nBsAnt;
     cellGrpPrmsCpu->nUeAnt                = nUeAnt;
     cellGrpPrmsCpu->W                     = W;
     cellGrpPrmsCpu->sigmaSqrd             = sigmaSqrd;
     cellGrpPrmsCpu->Pt_Rbg                = Pt_Rbg;
     cellGrpPrmsCpu->Pt_rbgAnt             = Pt_rbgAnt;
     cellGrpPrmsCpu->precodingScheme       = precodingScheme;
     cellGrpPrmsCpu->receiverScheme        = receiverScheme;
     cellGrpPrmsCpu->allocType             = cpuAllocType;
     cellGrpPrmsCpu->betaCoeff             = betaCoeff;
     cellGrpPrmsCpu->sinValThr             = sinValThr;
     cellGrpPrmsCpu->prioWeightStep        = prioWeightStep;
     cellGrpPrmsCpu->harqEnabledInd        = 0; // disable HARQ by default

    // Simulation parameter
     simParam.get()->totNumCell            = totNumCell;
 }

 void network::setupAPI(cudaStream_t strm)
 {
    if (execStatus.get()->cellIdRenew) {
        CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->cellId, cellId.get(), cidSize, cudaMemcpyHostToDevice, strm));
        if (heteroUeSelCells == 1) { // heterogeneous UE selection config. across cells
            CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->numUeSchdPerCellTTIArr, numUeSchdPerCellTTIArr.get(), numUeSchdArrSize, cudaMemcpyHostToDevice, strm));
        }
    }

    if (execStatus.get()->averageRateRenew) {
        CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->avgRatesActUe, avgRatesActUe.get(), arActUeSize, cudaMemcpyHostToDevice, strm));
        CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->tbErrLastActUe, tbErrLastActUe.get(), tbeActUeSize, cudaMemcpyHostToDevice, strm));
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->currSlotIdxPerCell, currSlotIdxPerCell.get(), sizeof(uint32_t)*totNumCell, cudaMemcpyHostToDevice, strm));   
 }

 void network::updateCurrSlotIdxPerCell(uint32_t slotIdx)
 {
    for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
        currSlotIdxPerCell[cIdx] = slotIdx;
    }
 }  

 void network::destroyAPI()
 {
    // free GPU mem allocate /////////////////////////////////
     if (cellGrpUeStatusGpu->avgRates) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->avgRates));
     if (cellGrpUeStatusGpu->avgRatesActUe) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->avgRatesActUe));
     if (cellGrpUeStatusGpu->lastSchdSlotActUe) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->lastSchdSlotActUe));    
     if (cellGrpUeStatusGpu->tbErrLast) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->tbErrLast));
     if (cellGrpUeStatusGpu->tbErrLastActUe) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->tbErrLastActUe));
     if (cellGrpUeStatusGpu->newDataActUe) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->newDataActUe));
     if (cellGrpUeStatusGpu->allocSolLastTx) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->allocSolLastTx));
     if (cellGrpUeStatusGpu->mcsSelSolLastTx) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->mcsSelSolLastTx));
     if (cellGrpUeStatusGpu->layerSelSolLastTx) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->layerSelSolLastTx));
     if (cellGrpUeStatusGpu->beamformGainCurrTx) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->beamformGainCurrTx));  
     if (cellGrpUeStatusGpu->bfGainPrgCurrTx) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->bfGainPrgCurrTx));
     if (cellGrpUeStatusGpu->beamformGainLastTx) CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->beamformGainLastTx));
     if (schdSolGpu->ueOrderInGrp) CUDA_CHECK_ERR(cudaFree(schdSolGpu->ueOrderInGrp));
     if (schdSolGpu->allocSol) CUDA_CHECK_ERR(cudaFree(schdSolGpu->allocSol));
     if (schdSolGpu->mcsSelSol) CUDA_CHECK_ERR(cudaFree(schdSolGpu->mcsSelSol));
     if (schdSolGpu->layerSelSol) CUDA_CHECK_ERR(cudaFree(schdSolGpu->layerSelSol));
     if (schdSolGpu->setSchdUePerCellTTI) CUDA_CHECK_ERR(cudaFree(schdSolGpu->setSchdUePerCellTTI));
     if (cellGrpPrmsGpu->estH_fr) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr));
     if (cellGrpPrmsGpu->estH_fr_half) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr_half));
     if (cellGrpPrmsGpu->estH_fr_actUe) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr_actUe));
     if (cellGrpPrmsGpu->estH_fr_actUe_prd) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr_actUe_prd));
     if (cellGrpPrmsGpu->currSlotIdxPerCell) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->currSlotIdxPerCell));  
     if (cellGrpPrmsGpu->cellId) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellId));
     if (cellGrpPrmsGpu->numUeSchdPerCellTTIArr) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->numUeSchdPerCellTTIArr));
     if (cellGrpPrmsGpu->cellAssoc) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellAssoc));
     if (cellGrpPrmsGpu->cellAssocActUe) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellAssocActUe));
     if (cellGrpPrmsGpu->postEqSinr) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->postEqSinr));
     if (cellGrpPrmsGpu->wbSinr) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->wbSinr));
     if (cellGrpPrmsGpu->blerTargetActUe) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->blerTargetActUe));
     if (cellGrpPrmsGpu->prdMat) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->prdMat));
     if (cellGrpPrmsGpu->prdMat_actUe) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->prdMat_actUe));
     if (cellGrpPrmsGpu->detMat) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->detMat));
     if (cellGrpPrmsGpu->detMat_actUe) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->detMat_actUe));
     if (cellGrpPrmsGpu->sinVal) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->sinVal));
     if (cellGrpPrmsGpu->sinVal_actUe) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->sinVal_actUe));
     if (muGrpListGpu->numUeInGrp) CUDA_CHECK_ERR(cudaFree(muGrpListGpu->numUeInGrp));
     if (muGrpListGpu->ueId) CUDA_CHECK_ERR(cudaFree(muGrpListGpu->ueId));
     if (muGrpListGpu->subbandId) CUDA_CHECK_ERR(cudaFree(muGrpListGpu->subbandId));
     if (schdSolGpu->muGrpList) CUDA_CHECK_ERR(cudaFree(schdSolGpu->muGrpList));    
     for (int cIdx = 0; cIdx < numCellConst; cIdx++) {
        if (sortedUeListGpu[cIdx]) CUDA_CHECK_ERR(cudaFree(sortedUeListGpu[cIdx]));
        if (mimoMode == 1) { // 64TR
            if (srsEstChan[cIdx]) CUDA_CHECK_ERR(cudaFree(srsEstChan[cIdx]));
            if (srsUeMap[cIdx]) CUDA_CHECK_ERR(cudaFree(srsUeMap[cIdx]));
        }
     }
     if (schdSolGpu->sortedUeList) CUDA_CHECK_ERR(cudaFree(schdSolGpu->sortedUeList));
     if (cellGrpPrmsGpu->srsEstChan) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsEstChan));
     if (cellGrpPrmsGpu->srsUeMap) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsUeMap));
     for (int uIdx = 0; uIdx < nUe; uIdx++) {
        if (estH_fr_perUeBufferGpu[uIdx]) CUDA_CHECK_ERR(cudaFree(estH_fr_perUeBufferGpu[uIdx]));
     }
     if (cellGrpPrmsGpu->estH_fr_perUeBuffer) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->estH_fr_perUeBuffer));
     if (cellGrpPrmsGpu->srsWbSnr) CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsWbSnr));
     if (schdSolGpu->muMimoInd) CUDA_CHECK_ERR(cudaFree(schdSolGpu->muMimoInd));
     if (schdSolGpu->nSCID) CUDA_CHECK_ERR(cudaFree(schdSolGpu->nSCID));
     if (schdSolGpu->pfMetricArr) CUDA_CHECK_ERR(cudaFree(schdSolGpu->pfMetricArr));
     if (schdSolGpu->pfIdArr) CUDA_CHECK_ERR(cudaFree(schdSolGpu->pfIdArr));
     
     // free CPU mem allocate /////////////////////////////////
     if (schdSolCpu->allocSol) delete schdSolCpu->allocSol;
     if (schdSolCpu->mcsSelSol) delete schdSolCpu->mcsSelSol;
     if (schdSolCpu->layerSelSol) delete schdSolCpu->layerSelSol;
     if (schdSolCpu->setSchdUePerCellTTI) delete schdSolCpu->setSchdUePerCellTTI;
     if (cellGrpUeStatusCpu->avgRates) delete cellGrpUeStatusCpu->avgRates;
     if (cellGrpUeStatusCpu->avgRatesActUe) delete cellGrpUeStatusCpu->avgRatesActUe;
     if (cellGrpUeStatusCpu->tbErrLast) delete cellGrpUeStatusCpu->tbErrLast;
     if (cellGrpUeStatusCpu->tbErrLastActUe) delete cellGrpUeStatusCpu->tbErrLastActUe;
     if (cellGrpUeStatusCpu->prioWeightActUe) delete cellGrpUeStatusCpu->prioWeightActUe;
     if (cellGrpPrmsCpu->prdMat) delete cellGrpPrmsCpu->prdMat;
     if (cellGrpPrmsCpu->prdMat_actUe) delete cellGrpPrmsCpu->prdMat_actUe;
     if (cellGrpPrmsCpu->detMat) delete cellGrpPrmsCpu->detMat;
     if (cellGrpPrmsCpu->detMat_actUe) delete cellGrpPrmsCpu->detMat_actUe;
     if (cellGrpPrmsCpu->sinVal) delete cellGrpPrmsCpu->sinVal;
     if (cellGrpPrmsCpu->sinVal_actUe) delete cellGrpPrmsCpu->sinVal_actUe;
     if (cellGrpPrmsCpu->numUeSchdPerCellTTIArr) delete cellGrpPrmsCpu->numUeSchdPerCellTTIArr;
     if (cellGrpPrmsCpu->cellAssoc) delete cellGrpPrmsCpu->cellAssoc;
     if (cellGrpPrmsCpu->cellAssocActUe) delete cellGrpPrmsCpu->cellAssocActUe;
     if (cellGrpPrmsCpu->postEqSinr) delete cellGrpPrmsCpu->postEqSinr;
     if (cellGrpPrmsCpu->wbSinr) delete cellGrpPrmsCpu->wbSinr;
     if (cellGrpPrmsCpu->estH_fr) delete cellGrpPrmsCpu->estH_fr;
     if (cellGrpPrmsCpu->estH_fr_actUe_prd) delete cellGrpPrmsCpu->estH_fr_actUe_prd;
     if (cellGrpPrmsCpu->blerTargetActUe) delete cellGrpPrmsCpu->blerTargetActUe;
 }

 void network::popHfrToPerUeBuffer()
 {
     for (int uIdx = 0; uIdx < nUe; uIdx++) {
        int assocCellIdx;
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            if (cellAssoc[cIdx*nUe + uIdx]) {
                assocCellIdx = cIdx;
                break;
            }
        }

        for (int rIdx = 0; rIdx < nPrbGrp; rIdx++) {
            for (int txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) {
                for (int rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) {
                    estH_fr_perUeBuffer[uIdx][rIdx*nBsAnt*nUeAnt + txAntIdx*nUeAnt + rxAntIdx].x = estH_fr[rIdx*nUe*totNumCell*nBsAnt*nUeAnt + uIdx*totNumCell*nBsAnt*nUeAnt + assocCellIdx*nBsAnt*nUeAnt + txAntIdx*nUeAnt + rxAntIdx].x;
                    estH_fr_perUeBuffer[uIdx][rIdx*nBsAnt*nUeAnt + txAntIdx*nUeAnt + rxAntIdx].y = estH_fr[rIdx*nUe*totNumCell*nBsAnt*nUeAnt + uIdx*totNumCell*nBsAnt*nUeAnt + assocCellIdx*nBsAnt*nUeAnt + txAntIdx*nUeAnt + rxAntIdx].y;
                }
            }
        }
     }
 }


 void network::copyPrdMatGpu2Cpu(cudaStream_t strm)
 {
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsCpu->prdMat_actUe, cellGrpPrmsGpu->prdMat_actUe, prdActUeSize, cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsCpu->detMat_actUe, cellGrpPrmsGpu->detMat_actUe, detActUeSize, cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsCpu->sinVal_actUe, cellGrpPrmsGpu->sinVal_actUe, sinValActUeSize, cudaMemcpyDeviceToHost, strm));
 }
 
void network::copyCellAssocResGpu2Cpu(cudaStream_t strm)
{
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellAssoc.get(), cellGrpPrmsGpu->cellAssoc, assocSize, cudaMemcpyDeviceToHost, strm));
}

 /* --------------------------------Need to check dimension: [TimeSlots PRBs txAnt RxAnt NumUEs NumCells]---------------------------------------------------------*/
 void network::genRandomChannel()
 {
     float stddev = 0.5*sqrt(2);
     std::normal_distribution<double> distribution(0.0, stddev);
    
    #pragma unroll
     for (int prbIdx = 0; prbIdx < nPrbGrp; prbIdx++) {
         for (int ueIdx = 0; ueIdx < nUe; ueIdx++) {
             for (int cellIdx = 0; cellIdx < totNumCell; cellIdx++) {
                 // generate channel coefficients per antenna pair
                 for (int txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) {
                     for (int rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) {
                         int index = prbIdx*nUe*totNumCell*nBsAnt*nUeAnt;
                         index += ueIdx*totNumCell*nBsAnt*nUeAnt;
                         index += cellIdx*nBsAnt*nUeAnt;
                         index += txAntIdx*nUeAnt;
                         index += rxAntIdx;
                         estH_fr[index].x = distribution(randomEngine);
                         estH_fr[index].y = distribution(randomEngine);
                         // printf("estH_fr[index].x = %f, estH_fr[index].y = %f, index = %d\n", float(estH_fr[index].x), float(estH_fr[index].y), index);
                     }
                 }
             }
         }
     }
 }

 void network::run(cudaStream_t strm)
 {
    CUDA_CHECK_ERR(cudaMemcpyAsync(allocSol.get(), schdSolGpu->allocSol, gpuAllocSolSize, cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK_ERR(cudaMemcpyAsync(layerSelSol.get(), schdSolGpu->layerSelSol, layerSize, cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK_ERR(cudaMemcpyAsync(mcsSelSol.get(), schdSolGpu->mcsSelSol, mcsSelSolSize, cudaMemcpyDeviceToHost, strm));
    if (~fixCellAssociation) {
        CUDA_CHECK_ERR(cudaMemcpyAsync(cellAssoc.get(), cellGrpPrmsGpu->cellAssoc, assocSize, cudaMemcpyDeviceToHost, strm));
    }
 }

 void network::phyAbstract(uint8_t gpuInd, int slotIdx)
 {
    // verify slotIdx
    if (slotIdx < 0) {
        throw std::runtime_error("Error: invalid time slot index");
    }

    for (int idx = 0; idx < nUe; idx++) {
        floatRandomArr[idx] = uniformRealDist(randomEngine);
    }

    if (gpuInd == 1) { // for GPU
        updateDataRatePdschGpu(slotIdx);
    } else if (gpuInd == 0) { // for CPU
        updateDataRatePdschCpu(slotIdx);
    } else if (gpuInd == 2) { // for both CPU and GPU
        updateDataRatePdschGpu(slotIdx);
        updateDataRatePdschCpu(slotIdx);
    } else {
        throw std::runtime_error("Error: invalid gpuInd value, 0 - CPU, 1 - GPU, 2 - both CPU and GPU");
    }
 }

 void network::updateDataRatePdschGpu(int slotIdx)
 {
    uint32_t nBsAntSqrd         = nBsAnt*nBsAnt;
    cuComplex* CMat             = new cuComplex[nBsAntSqrd];
    cuComplex* BMat             = new cuComplex[nBsAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nBsAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];
    float* perUeThr             = new float[nUe];

    if (gpuAllocType) {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine RBG-to-UE mapping
        int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;

                for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                    if (assocCellIdx[ueIdx] != cIdx)
                        continue;

                    if (rbgIdx >= allocSol[2*ueIdx] && rbgIdx < allocSol[2*ueIdx+1]) {
                        allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                    }  
                }
            }
        }

        // determine CRC and update average data rate
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            int nrAllocRbg = allocSol[2*ueIdx+1] - allocSol[2*ueIdx];
            if (assocCellIdx[ueIdx] == -1) {
                nrAllocRbg = 0;
            }
            int nrAllocPrb = nrAllocRbg*nPrbPerGrp;
            if (nrAllocRbg == 0) { // not scheduled for the last time slot
                avgRates[ueIdx] = (1.0-pfAvgRateUpd)*avgRates[ueIdx];
                perUeThr[ueIdx] = 0;
                tbErrRecordsGpu[ueIdx][slotIdx] = -1;
            } else { // scheduled for the last time slot
                int mcsSel = mcsSelSol[ueIdx];
                // calculate average SINR over the allocated PRBs
                float avgSinr = 0;

                for (int rbgIdx = allocSol[2*ueIdx]; rbgIdx < allocSol[2*ueIdx+1]; rbgIdx++) {
                    if (DL == 1) { // DL
                        uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                        for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                            for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                                if (rowIdx == colIdx) {
                                    CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                } else {
                                    CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                }
                            }
                        }

                        for (int l = 0; l < totNumCell; l++) {
                            if (l == assocCellIdx[ueIdx]) 
                                continue;

                            int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                            if (uePrimeIdx < 0)
                                continue;

                            uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                            matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                        }
                        matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                        uint32_t hMatStart = hTemp + assocCellIdx[ueIdx]*nBsAnt*nUeAnt;

                        if (precodingScheme == 0) { // no precoding
                            matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                        } else { 
                            uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                            matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, prdMat.get()+vMatStart, nBsAnt, BMat);
                            matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                        }

                        for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                            EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                        }
                        matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                        for (int layerIdx = 0; layerIdx < layerSelSol[ueIdx]; layerIdx++) {
                            avgSinr += 1.0/EInvMat[layerIdx*nBsAnt+layerIdx].x - 1.0;
                        }
                    } else { // UL
                        uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ assocCellIdx[ueIdx]*nBsAnt*nUeAnt;

                        for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                            for (int colIdx = 0; colIdx < nBsAnt; colIdx++) {
                                if (rowIdx == colIdx) {
                                    CMat[colIdx*nBsAnt + rowIdx].x = sigmaSqrd;
                                    CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                } else {
                                    CMat[colIdx*nBsAnt + rowIdx].x = 0;
                                    CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                }
                            }
                        }  
                            
                        for (int l = 0; l < totNumCell; l++) {
                            if (l == assocCellIdx[ueIdx]) 
                                continue;
        
                            int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                            if (uePrimeIdx < 0)
                                continue;
        
                            uint32_t hInterfMatStart = hTemp+ uePrimeIdx*totNumCell*nBsAnt*nUeAnt;
                            matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nBsAnt, nUeAnt, CMat);
                        }
                        matAlg->matInverseEigen(CMat, nBsAnt, CInvMat);

                        uint32_t hMatStart = hTemp + ueIdx*totNumCell*nBsAnt*nUeAnt;
        
                        if (precodingScheme == 0) { // no precoding
                            printf("Error: Currently only support SVD precoding for UL");
                            return;
                        } else { 
                            uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nUeAnt*nUeAnt;
                            matAlg->matMultiplication_ab(&estH_fr[hMatStart], nBsAnt, nUeAnt, prdMat.get()+vMatStart, nUeAnt, BMat);
    
                            matAlg->matMultiplication_aHb(BMat, nBsAnt, nUeAnt, CInvMat, nBsAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nUeAnt, nBsAnt, BMat, nUeAnt, EMat);
                        }

                        for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                            EMat[rowIdx*nUeAnt+rowIdx].x += 1.0;
                        }
                        matAlg->matInverseEigen(EMat, nUeAnt, EInvMat);

                        for (int layerIdx = 0; layerIdx < layerSelSol[ueIdx]; layerIdx++) {
                            avgSinr += 1.0/EInvMat[layerIdx*nUeAnt+layerIdx].x - 1.0;
                        }  
                    }
                }
                        
                avgSinr /= (nrAllocRbg*layerSelSol[ueIdx]);

                float avgSinrDB = 10.0*log10(avgSinr);

                if (avgSinrDB < -4.75) {
                    avgSinrDB = -4.75;
                }

                int NrEleSnrMcsArr = mcsTableRowSizes[mcsSel];
                float* snrArrCurr = snrMcsArr[mcsSel];
                float* blerArrCurr = blerMscArr[mcsSel];
                float blerCurr = 0;
                
                if (avgSinrDB < snrArrCurr[NrEleSnrMcsArr-1]) {
                    blerCurr = 1.0;
                } else if (avgSinrDB > snrArrCurr[0]) { 
                    blerCurr = 0;
                } else {
                    for (int blerIdx = NrEleSnrMcsArr-1; blerIdx > 0; blerIdx--) {
                        if (avgSinrDB <= snrArrCurr[blerIdx-1]) {
                            float relDist= (avgSinrDB - snrArrCurr[blerIdx]) / (snrArrCurr[blerIdx-1] - snrArrCurr[blerIdx]);
                            blerCurr = blerArrCurr[blerIdx] + relDist*(blerArrCurr[blerIdx-1] - blerArrCurr[blerIdx]);
                            break;
                        }
                    }
                }

                float rndNum = floatRandomArr[ueIdx];
                int tbErr = 0;
                if (rndNum < blerCurr) {
                    tbErr = 1;
                }

                tbErrLast[ueIdx] = tbErr;
                tbErrRecordsGpu[ueIdx][slotIdx] = tbErr;

                uint32_t TBS = determineTbsPdsch(nrAllocPrb, pdschNrOfDataSymb, layerSelSol[ueIdx], mcsTable_codeRate[mcsSel]/1024.0, mcsTable_qamOrder[mcsSel]);
                float insRate = static_cast<float>(TBS)*(1-tbErr)/slotDuration;
                // CPU and GPU buffer sizes hold the same values (data managed on CPU side)
                if(0 != cellGrpUeStatusCpu->bufferSize){
                    auto sched_bytes = std::min({TBS,cellGrpUeStatusCpu->bufferSize[ueIdx]});
                    insRate = static_cast<float>(sched_bytes)*(1-tbErr)/slotDuration;
                }
                avgRates[ueIdx] = (1.0-pfAvgRateUpd)*avgRates[ueIdx] + pfAvgRateUpd*insRate;
                perUeThr[ueIdx] = insRate;
            }

            // MCS selection record
            mcsSelRecordsGpu[ueIdx][slotIdx] = mcsSelSol[ueIdx];
            layerSelRecordsGpu[ueIdx][slotIdx] = layerSelSol[ueIdx];
        }

        delete assocCellIdx;
        delete allocSol_rbg2Ue;
    } else {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < nCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx] == 1) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine CRC and update average data rate
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            int nrAllocRbg = 0;

            if (assocCellIdx[ueIdx] == -1) {
                nrAllocRbg = 0;
            } else {
                for (int prgIdx = 0; prgIdx < nPrbGrp; prgIdx++) {
                    if (allocSol[prgIdx*nCell + assocCellIdx[ueIdx]] == ueIdx) {
                        nrAllocRbg++;
                    }
                }
            }
            
            int nrAllocPrb = nrAllocRbg*nPrbPerGrp;
            if (nrAllocRbg == 0) { // not scheduled for the last time slot
                avgRates[ueIdx] = (1.0-pfAvgRateUpd)*avgRates[ueIdx];
                perUeThr[ueIdx] = 0;
                tbErrRecordsGpu[ueIdx][slotIdx] = -1;
            } else { // scheduled for the last time slot
                int mcsSel = mcsSelSol[ueIdx];
                // calculate average SINR over the allocated PRBs
                float avgSinr = 0;

                for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                    if (allocSol[rbgIdx*nCell + assocCellIdx[ueIdx]] != ueIdx) {
                        continue;
                    }

                    if (DL == 1) { // DL
                        uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                        for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                            for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                                if (rowIdx == colIdx) {
                                    CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                } else {
                                    CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                }
                            }
                        }

                        for (int l = 0; l < totNumCell; l++) {
                            if (l == assocCellIdx[ueIdx]) 
                                continue;

                            int uePrimeIdx = allocSol[rbgIdx*totNumCell+l];
                            if (uePrimeIdx < 0)
                                continue;

                            uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                            matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                        }
                        matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                        uint32_t hMatStart = hTemp + assocCellIdx[ueIdx]*nBsAnt*nUeAnt;

                        if (precodingScheme == 0) { // no precoding
                            matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                        } else { 
                            uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                            matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, prdMat.get()+vMatStart, nBsAnt, BMat);
                            matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                        }

                        for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                            EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                        }
                        matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                        for (int layerIdx = 0; layerIdx < layerSelSol[ueIdx]; layerIdx++) {
                            avgSinr += 1.0/EInvMat[layerIdx*nBsAnt+layerIdx].x - 1.0;
                        }
                    } else { // UL
                        uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ assocCellIdx[ueIdx]*nBsAnt*nUeAnt;

                        for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                            for (int colIdx = 0; colIdx < nBsAnt; colIdx++) {
                                if (rowIdx == colIdx) {
                                    CMat[colIdx*nBsAnt + rowIdx].x = sigmaSqrd;
                                    CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                } else {
                                    CMat[colIdx*nBsAnt + rowIdx].x = 0;
                                    CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                }
                            }
                        }  
                            
                        for (int l = 0; l < totNumCell; l++) {
                            if (l == assocCellIdx[ueIdx]) 
                                continue;
        
                            int uePrimeIdx = allocSol[rbgIdx*totNumCell+l];
                            if (uePrimeIdx < 0)
                                continue;
        
                            uint32_t hInterfMatStart = hTemp+ uePrimeIdx*totNumCell*nBsAnt*nUeAnt;
                            matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nBsAnt, nUeAnt, CMat);
                        }
                        matAlg->matInverseEigen(CMat, nBsAnt, CInvMat);

                        uint32_t hMatStart = hTemp + ueIdx*totNumCell*nBsAnt*nUeAnt;
        
                        if (precodingScheme == 0) { // no precoding
                            printf("Error: Currently only support SVD precoding for UL");
                            return;
                        } else { 
                            uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nUeAnt*nUeAnt;
                            matAlg->matMultiplication_ab(&estH_fr[hMatStart], nBsAnt, nUeAnt, prdMat.get()+vMatStart, nUeAnt, BMat);
    
                            matAlg->matMultiplication_aHb(BMat, nBsAnt, nUeAnt, CInvMat, nBsAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nUeAnt, nBsAnt, BMat, nUeAnt, EMat);
                        }

                        for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                            EMat[rowIdx*nUeAnt+rowIdx].x += 1.0;
                        }
                        matAlg->matInverseEigen(EMat, nUeAnt, EInvMat);

                        for (int layerIdx = 0; layerIdx < layerSelSol[ueIdx]; layerIdx++) {
                            avgSinr += 1.0/EInvMat[layerIdx*nUeAnt+layerIdx].x - 1.0;
                        }  
                    }
                }
                        
                avgSinr /= (nrAllocRbg*layerSelSol[ueIdx]);

                float avgSinrDB = 10.0*log10(avgSinr);

                if (avgSinrDB < -4.75) {
                    avgSinrDB = -4.75;
                }

                int NrEleSnrMcsArr = mcsTableRowSizes[mcsSel];
                float* snrArrCurr = snrMcsArr[mcsSel];
                float* blerArrCurr = blerMscArr[mcsSel];
                float blerCurr = 0;
                
                if (avgSinrDB < snrArrCurr[NrEleSnrMcsArr-1]) {
                    blerCurr = 1.0;
                } else if (avgSinrDB > snrArrCurr[0]) { 
                    blerCurr = 0;
                } else {
                    for (int blerIdx = NrEleSnrMcsArr-1; blerIdx > 0; blerIdx--) {
                        if (avgSinrDB <= snrArrCurr[blerIdx-1]) {
                            float relDist= (avgSinrDB - snrArrCurr[blerIdx]) / (snrArrCurr[blerIdx-1] - snrArrCurr[blerIdx]);
                            blerCurr = blerArrCurr[blerIdx] + relDist*(blerArrCurr[blerIdx-1] - blerArrCurr[blerIdx]);
                            break;
                        }
                    }
                }

                float rndNum = floatRandomArr[ueIdx];
                int tbErr = 0;
                if (rndNum < blerCurr) {
                    tbErr = 1;
                }

                tbErrLast[ueIdx] = tbErr;
                tbErrRecordsGpu[ueIdx][slotIdx] = tbErr;

                uint32_t TBS = determineTbsPdsch(nrAllocPrb, pdschNrOfDataSymb, layerSelSol[ueIdx], mcsTable_codeRate[mcsSel]/1024.0, mcsTable_qamOrder[mcsSel]);
                float insRate = static_cast<float>(TBS)*(1-tbErr)/slotDuration;
                // CPU and GPU buffer sizes hold the same values (data managed on CPU side)
                if(0 != cellGrpUeStatusCpu->bufferSize){
                    auto sched_bytes = std::min({TBS,cellGrpUeStatusCpu->bufferSize[ueIdx]});
                    insRate = static_cast<float>(sched_bytes)*(1-tbErr)/slotDuration;
                }
                avgRates[ueIdx] = (1.0-pfAvgRateUpd)*avgRates[ueIdx] + pfAvgRateUpd*insRate;
                perUeThr[ueIdx] = insRate;
                
            }

            // MCS selection record
            mcsSelRecordsGpu[ueIdx][slotIdx] = mcsSelSol[ueIdx];
            layerSelRecordsGpu[ueIdx][slotIdx] = layerSelSol[ueIdx];
        }

        delete assocCellIdx;
    }

    sumInsThrRecordsGpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        bool coordinatedCellUe = false;
        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            if (cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                coordinatedCellUe = true;
                break;
            }
        }

        if (coordinatedCellUe) {
            sumInsThrRecordsGpu[slotIdx] += perUeThr[uIdx];
        }
    }

    delete perUeThr;
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
 }

 uint32_t network::getAllocBytes(int ueIdx)
 {
    // TODO only handles type 1 allocation
    int nrAllocRbg = allocSol[2*ueIdx+1] - allocSol[2*ueIdx];
    int nrAllocPrb = nrAllocRbg*nPrbPerGrp;
    int mcsSel = mcsSelSol[ueIdx];
    // Note: TBS seems to have a minimum of 48, so clip when no RBs allocated
    uint32_t TBS = (nrAllocPrb>0)*determineTbsPdsch(nrAllocPrb, pdschNrOfDataSymb, layerSelSol[ueIdx], mcsTable_codeRate[mcsSel]/1024.0, mcsTable_qamOrder[mcsSel]);
    return TBS*(1-tbErrLast[ueIdx])/8; // Return value in bytes
 }

 void network::updateDataRatePdschCpu(int slotIdx)
 {
    uint32_t nBsAntSqrd         = nBsAnt*nBsAnt;

    cuComplex* CMat             = new cuComplex[nBsAntSqrd];
    cuComplex* CInvMat          = new cuComplex[nBsAntSqrd];
    cuComplex* BMat             = new cuComplex[nBsAntSqrd];
    cuComplex* DMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EMat             = new cuComplex[nBsAntSqrd];
    cuComplex* EInvMat          = new cuComplex[nBsAntSqrd];
    float* perUeThr             = new float[nUe];

    if (cpuAllocType) {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine RBG-to-UE mapping
        int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;

                for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                    if (assocCellIdx[ueIdx] != cIdx)
                        continue;

                    if (rbgIdx >= schdSolCpu->allocSol[2*ueIdx] && rbgIdx < schdSolCpu->allocSol[2*ueIdx+1]) {
                        allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                    }  
                }
            }
        }

        // determine CRC and update average data rate
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            int nrAllocRbg = schdSolCpu->allocSol[2*ueIdx+1] - schdSolCpu->allocSol[2*ueIdx];
            if (assocCellIdx[ueIdx] == -1) {
                nrAllocRbg = 0;
            }
            int nrAllocPrb = nrAllocRbg*nPrbPerGrp;
            if (nrAllocRbg == 0) { // not scheduled for the last time slot
                cellGrpUeStatusCpu->avgRates[ueIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[ueIdx];
                tbErrRecordsCpu[ueIdx][slotIdx] = -1;
                perUeThr[ueIdx] = 0;
            } else { // scheduled for the last time slot
                int mcsSel = schdSolCpu->mcsSelSol[ueIdx];
                // calculate average SINR over the allocated PRBs
                float avgSinr = 0;

                for (int rbgIdx = schdSolCpu->allocSol[2*ueIdx]; rbgIdx < schdSolCpu->allocSol[2*ueIdx+1]; rbgIdx++) {
                        if (DL == 1) { // DL
                            uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;

                            for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                                for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                                    if (rowIdx == colIdx) {
                                        CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                        CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                    } else {
                                        CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                        CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                    }
                                }
                            }
        
                            for (int l = 0; l < totNumCell; l++) {
                                if (l == assocCellIdx[ueIdx]) 
                                    continue;
        
                                int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                                if (uePrimeIdx < 0)
                                    continue;
        
                                uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                                matAlg->matMultiplication_aaHplusb(&cellGrpPrmsCpu->estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                            }
                            matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);
        
                            uint32_t hMatStart = hTemp + assocCellIdx[ueIdx]*nBsAnt*nUeAnt;
        
                            if (precodingScheme == 0) { // no precoding
                                matAlg->matMultiplication_aHb(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                                matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &cellGrpPrmsCpu->estH_fr[hMatStart], nBsAnt, EMat);
                            } else { 
                                uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                                matAlg->matMultiplication_ab(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);
    
                                matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                                matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                            }
        
                            for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                                EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                            }
                            matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                            for (int layerIdx = 0; layerIdx < schdSolCpu->layerSelSol[ueIdx]; layerIdx++) {
                                avgSinr += 1.0/EInvMat[layerIdx*nBsAnt+layerIdx].x - 1.0;
                            }
                        } else { // UL
                            uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ assocCellIdx[ueIdx]*nBsAnt*nUeAnt;

                            for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                                for (int colIdx = 0; colIdx < nBsAnt; colIdx++) {
                                    if (rowIdx == colIdx) {
                                        CMat[colIdx*nBsAnt + rowIdx].x = sigmaSqrd;
                                        CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                    } else {
                                        CMat[colIdx*nBsAnt + rowIdx].x = 0;
                                        CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                    }
                                }
                            }  
                            
                            for (int l = 0; l < totNumCell; l++) {
                                if (l == assocCellIdx[ueIdx]) 
                                    continue;
        
                                int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                                if (uePrimeIdx < 0)
                                    continue;
        
                                uint32_t hInterfMatStart = hTemp+ uePrimeIdx*totNumCell*nBsAnt*nUeAnt;
                                matAlg->matMultiplication_aaHplusb(&cellGrpPrmsCpu->estH_fr[hInterfMatStart], nBsAnt, nUeAnt, CMat);
                            }
                            matAlg->matInverseEigen(CMat, nBsAnt, CInvMat);

                            uint32_t hMatStart = hTemp + ueIdx*totNumCell*nBsAnt*nUeAnt;
        
                            if (precodingScheme == 0) { // no precoding
                                printf("Error: Currently only support SVD precoding for UL");
                                return;
                            } else { 
                                uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nUeAnt*nUeAnt;
                                matAlg->matMultiplication_ab(&cellGrpPrmsCpu->estH_fr[hMatStart], nBsAnt, nUeAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nUeAnt, BMat);
    
                                matAlg->matMultiplication_aHb(BMat, nBsAnt, nUeAnt, CInvMat, nBsAnt, DMat);
                                matAlg->matMultiplication_ab(DMat, nUeAnt, nBsAnt, BMat, nUeAnt, EMat);
                            }

                            for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                                EMat[rowIdx*nUeAnt+rowIdx].x += 1.0;
                            }
                            matAlg->matInverseEigen(EMat, nUeAnt, EInvMat);

                            for (int layerIdx = 0; layerIdx < schdSolCpu->layerSelSol[ueIdx]; layerIdx++) {
                                avgSinr += 1.0/EInvMat[layerIdx*nUeAnt+layerIdx].x - 1.0;
                            }  
                        }
                }
                avgSinr /= (nrAllocRbg*schdSolCpu->layerSelSol[ueIdx]);

                float avgSinrDB = 10.0*log10(avgSinr);

                if (avgSinrDB < -4.75) {
                    avgSinrDB = -4.75;
                }

                int NrEleSnrMcsArr = mcsTableRowSizes[mcsSel];
                float* snrArrCurr = snrMcsArr[mcsSel];
                float* blerArrCurr = blerMscArr[mcsSel];
                float blerCurr = 0;
                
                if (avgSinrDB < snrArrCurr[NrEleSnrMcsArr-1]) {
                    blerCurr = 1.0;
                } else if (avgSinrDB > snrArrCurr[0]) { 
                    blerCurr = 0;
                } else {
                    for (int blerIdx = NrEleSnrMcsArr-1; blerIdx > 0; blerIdx--) {
                        if (avgSinrDB <= snrArrCurr[blerIdx-1]) {
                            float relDist= (avgSinrDB - snrArrCurr[blerIdx]) / (snrArrCurr[blerIdx-1] - snrArrCurr[blerIdx]);
                            blerCurr = blerArrCurr[blerIdx] + relDist*(blerArrCurr[blerIdx-1] - blerArrCurr[blerIdx]);
                            break;
                        }
                    }
                }

                float rndNum = floatRandomArr[ueIdx];
                int tbErr = 0;
                if (rndNum < blerCurr) {
                    tbErr = 1;
                }
                
                cellGrpUeStatusCpu->tbErrLast[ueIdx] = tbErr;
                tbErrRecordsCpu[ueIdx][slotIdx] = tbErr;

                uint32_t TBS = determineTbsPdsch(nrAllocPrb, pdschNrOfDataSymb, schdSolCpu->layerSelSol[ueIdx], mcsTable_codeRate[mcsSel]/1024.0, mcsTable_qamOrder[mcsSel]);
                float insRate = static_cast<float>(TBS)*(1-tbErr)/slotDuration;
                if(0 != cellGrpUeStatusCpu->bufferSize){
                    auto sched_bytes = std::min({TBS,cellGrpUeStatusCpu->bufferSize[ueIdx]});
                    cellGrpUeStatusCpu->bufferSize[ueIdx] -= sched_bytes;
                    insRate = static_cast<float>(sched_bytes)*(1-tbErr)/slotDuration;
                }
                cellGrpUeStatusCpu->avgRates[ueIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[ueIdx] + pfAvgRateUpd*insRate;
                perUeThr[ueIdx] = insRate;
            }

            // MCS selection record
            mcsSelRecordsCpu[ueIdx][slotIdx] = schdSolCpu->mcsSelSol[ueIdx];
            layerSelRecordsCpu[ueIdx][slotIdx] = schdSolCpu->layerSelSol[ueIdx];
        }

        delete assocCellIdx;
        delete allocSol_rbg2Ue;
    } else {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine CRC and update average data rate
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            int nrAllocRbg = 0;

            if (assocCellIdx[ueIdx] == -1) {
                nrAllocRbg = 0;
            } else {
                for (int prgIdx = 0; prgIdx < nPrbGrp; prgIdx++) {
                    if (schdSolCpu->allocSol[prgIdx*nCell + assocCellIdx[ueIdx]] == ueIdx) {
                        nrAllocRbg++;
                    }
                }
            }

            int nrAllocPrb = nrAllocRbg*nPrbPerGrp;
            if (nrAllocRbg == 0) { // not scheduled for the last time slot
                cellGrpUeStatusCpu->avgRates[ueIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[ueIdx];
                tbErrRecordsCpu[ueIdx][slotIdx] = -1;
                perUeThr[ueIdx] = 0;
            } else { // scheduled for the last time slot
                int mcsSel = schdSolCpu->mcsSelSol[ueIdx];
                // calculate average SINR over the allocated PRBs
                float avgSinr = 0;
                
                for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                        if (schdSolCpu->allocSol[rbgIdx*nCell + assocCellIdx[ueIdx]] != ueIdx) {
                            continue;
                        }

                        if (DL == 1) { // DL
                            uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;

                            for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                                for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                                    if (rowIdx == colIdx) {
                                        CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                        CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                    } else {
                                        CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                        CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                    }
                                }
                            }
        
                            for (int l = 0; l < totNumCell; l++) {
                                if (l == assocCellIdx[ueIdx]) 
                                    continue;
        
                                int uePrimeIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+l];
                                if (uePrimeIdx < 0)
                                    continue;
        
                                uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                                matAlg->matMultiplication_aaHplusb(&cellGrpPrmsCpu->estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                            }
                            matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);
        
                            uint32_t hMatStart = hTemp + assocCellIdx[ueIdx]*nBsAnt*nUeAnt;
        
                            if (precodingScheme == 0) { // no precoding
                                matAlg->matMultiplication_aHb(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                                matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &cellGrpPrmsCpu->estH_fr[hMatStart], nBsAnt, EMat);
                            } else { 
                                uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                                matAlg->matMultiplication_ab(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);
    
                                matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                                matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                            }
        
                            for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                                EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                            }
                            matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                            for (int layerIdx = 0; layerIdx < schdSolCpu->layerSelSol[ueIdx]; layerIdx++) {
                                avgSinr += 1.0/EInvMat[layerIdx*nBsAnt+layerIdx].x - 1.0;
                            }
                        } else { // UL
                            uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ assocCellIdx[ueIdx]*nBsAnt*nUeAnt;

                            for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                                for (int colIdx = 0; colIdx < nBsAnt; colIdx++) {
                                    if (rowIdx == colIdx) {
                                        CMat[colIdx*nBsAnt + rowIdx].x = sigmaSqrd;
                                        CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                    } else {
                                        CMat[colIdx*nBsAnt + rowIdx].x = 0;
                                        CMat[colIdx*nBsAnt + rowIdx].y = 0;
                                    }
                                }
                            }  
                            
                            for (int l = 0; l < totNumCell; l++) {
                                if (l == assocCellIdx[ueIdx]) 
                                    continue;
        
                                int uePrimeIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+l];
                                if (uePrimeIdx < 0)
                                    continue;
        
                                uint32_t hInterfMatStart = hTemp+ uePrimeIdx*totNumCell*nBsAnt*nUeAnt;
                                matAlg->matMultiplication_aaHplusb(&cellGrpPrmsCpu->estH_fr[hInterfMatStart], nBsAnt, nUeAnt, CMat);
                            }
                            matAlg->matInverseEigen(CMat, nBsAnt, CInvMat);

                            uint32_t hMatStart = hTemp + ueIdx*totNumCell*nBsAnt*nUeAnt;
        
                            if (precodingScheme == 0) { // no precoding
                                printf("Error: Currently only support SVD precoding for UL");
                                return;
                            } else { 
                                uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nUeAnt*nUeAnt;
                                matAlg->matMultiplication_ab(&cellGrpPrmsCpu->estH_fr[hMatStart], nBsAnt, nUeAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nUeAnt, BMat);
    
                                matAlg->matMultiplication_aHb(BMat, nBsAnt, nUeAnt, CInvMat, nBsAnt, DMat);
                                matAlg->matMultiplication_ab(DMat, nUeAnt, nBsAnt, BMat, nUeAnt, EMat);
                            }

                            for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                                EMat[rowIdx*nUeAnt+rowIdx].x += 1.0;
                            }
                            matAlg->matInverseEigen(EMat, nUeAnt, EInvMat);

                            for (int layerIdx = 0; layerIdx < schdSolCpu->layerSelSol[ueIdx]; layerIdx++) {
                                avgSinr += 1.0/EInvMat[layerIdx*nUeAnt+layerIdx].x - 1.0;
                            }  
                        }
                }
                avgSinr /= (nrAllocRbg*schdSolCpu->layerSelSol[ueIdx]);

                float avgSinrDB = 10.0*log10(avgSinr);

                if (avgSinrDB < -4.75) {
                    avgSinrDB = -4.75;
                }

                int NrEleSnrMcsArr = mcsTableRowSizes[mcsSel];
                float* snrArrCurr = snrMcsArr[mcsSel];
                float* blerArrCurr = blerMscArr[mcsSel];
                float blerCurr = 0;
                
                if (avgSinrDB < snrArrCurr[NrEleSnrMcsArr-1]) {
                    blerCurr = 1.0;
                } else if (avgSinrDB > snrArrCurr[0]) { 
                    blerCurr = 0;
                } else {
                    for (int blerIdx = NrEleSnrMcsArr-1; blerIdx > 0; blerIdx--) {
                        if (avgSinrDB <= snrArrCurr[blerIdx-1]) {
                            float relDist= (avgSinrDB - snrArrCurr[blerIdx]) / (snrArrCurr[blerIdx-1] - snrArrCurr[blerIdx]);
                            blerCurr = blerArrCurr[blerIdx] + relDist*(blerArrCurr[blerIdx-1] - blerArrCurr[blerIdx]);
                            break;
                        }
                    }
                }

                float rndNum = floatRandomArr[ueIdx];
                int tbErr = 0;
                if (rndNum < blerCurr) {
                    tbErr = 1;
                }
                
                cellGrpUeStatusCpu->tbErrLast[ueIdx] = tbErr;
                tbErrRecordsCpu[ueIdx][slotIdx] = tbErr;

                uint32_t TBS = determineTbsPdsch(nrAllocPrb, pdschNrOfDataSymb, schdSolCpu->layerSelSol[ueIdx], mcsTable_codeRate[mcsSel]/1024.0, mcsTable_qamOrder[mcsSel]);
                float insRate = static_cast<float>(TBS)*(1-tbErr)/slotDuration;
                if(0 != cellGrpUeStatusCpu->bufferSize){
                    auto sched_bytes = std::min({TBS,cellGrpUeStatusCpu->bufferSize[ueIdx]});
                    cellGrpUeStatusCpu->bufferSize[ueIdx] -= sched_bytes;
                    insRate = static_cast<float>(sched_bytes)*(1-tbErr)/slotDuration;
                }
                cellGrpUeStatusCpu->avgRates[ueIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[ueIdx] + pfAvgRateUpd*insRate;
                perUeThr[ueIdx] = insRate;
            }

            // MCS selection record
            mcsSelRecordsCpu[ueIdx][slotIdx] = schdSolCpu->mcsSelSol[ueIdx];
            layerSelRecordsCpu[ueIdx][slotIdx] = schdSolCpu->layerSelSol[ueIdx];
        }

        delete assocCellIdx;
    }

    sumInsThrRecordsCpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        bool coordinatedCellUe = false;
        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            if (cellGrpPrmsCpu->cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                coordinatedCellUe = true;
                break;
            }
        }
        
        if (coordinatedCellUe) {
            sumInsThrRecordsCpu[slotIdx] += perUeThr[uIdx];
        }
    }

    delete perUeThr;
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
 }

 void network::updateDataRateUeSelCpu(int slotIdx)
 {
        // verify slotIdx
        if (slotIdx < 0) {
            printf("Error: invalid time slot index");
            return;
        }

        uint32_t nBsAntSqrd         = nBsAnt*nBsAnt;
        float* perUeThr     = new float[nUe];
        cuComplex* BMat     = new cuComplex[nBsAntSqrd];
        cuComplex* CMat     = new cuComplex[nBsAntSqrd];
        cuComplex* CInvMat  = new cuComplex[nBsAntSqrd];
        cuComplex* DMat     = new cuComplex[nBsAntSqrd];
        cuComplex* EMat     = new cuComplex[nBsAntSqrd];
        cuComplex* EInvMat  = new cuComplex[nBsAntSqrd];
    
        for (int uIdx = 0; uIdx < nUe; uIdx++) {
            perUeThr[uIdx] = 0;
        }
    
        // determine per-UE instantaneous data rates based on PRB allocate type (type-0 or type-1)
        if (cpuAllocType) {
            // determine cell associate 
            int* assocCellIdx = new int[nUe];
            for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                assocCellIdx[ueIdx] = -1;
                for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                    if (cellGrpPrmsCpu->cellAssoc[cIdx*nUe + ueIdx]) {
                        assocCellIdx[ueIdx] =cIdx;
                        break;
                    }
                }
            }
    
            // determine RBG-to-UE mapping
            int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                    allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;
    
                    for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                        if (assocCellIdx[ueIdx] != cIdx)
                            continue;
    
                        if (rbgIdx >= schdSolCpu->allocSol[2*ueIdx] && rbgIdx < schdSolCpu->allocSol[2*ueIdx+1]) {
                            allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                        }  
                    }
                }
            }
     
            // determine per-UE instantaneous data rates
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                    int ueIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx];
    
                    if (ueIdx >= 0) {
                        uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                        for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                            for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                                if (rowIdx == colIdx) {
                                    CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                } else {
                                    CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                }
                            }
                        }
    
                        for (int l = 0; l < totNumCell; l++) {
                            if (l == cIdx) 
                                continue;
    
                            int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                            if (uePrimeIdx < 0)
                                continue;
    
                            uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                            matAlg->matMultiplication_aaHplusb(&cellGrpPrmsCpu->estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                        }
                        matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);
    
                        uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;
    
                        if (precodingScheme == 0) { // no precoding
                            matAlg->matMultiplication_aHb(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &cellGrpPrmsCpu->estH_fr[hMatStart], nBsAnt, EMat);
                        } else { 
                            uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                            matAlg->matMultiplication_ab(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);

                            matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                        }
    
                        for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                            EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                        }
                        matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);
    
                        float dataRate = 0;
                        for (int j = 0; j < nUeAnt; j++) {
                            dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                        }
                        perUeThr[ueIdx] += dataRate;
                    }
                }
            }
    
            delete assocCellIdx;
            delete allocSol_rbg2Ue;
        } else {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                    int ueIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx];
    
                    if (ueIdx >= 0) {
                        uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                        for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                            for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                                if (rowIdx == colIdx) {
                                    CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                } else {
                                    CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                    CMat[colIdx*nUeAnt + rowIdx].y = 0;
                                }
                            }
                        }
    
                        for (int l = 0; l < totNumCell; l++) {
                            if (l == cIdx) 
                                continue;
    
                            int uePrimeIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+l];
                            if (uePrimeIdx < 0)
                                continue;
    
                            uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                            matAlg->matMultiplication_aaHplusb(&cellGrpPrmsCpu->estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                        }
                        matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);
    
                        uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;
    
                        if (precodingScheme == 0) { // no precoding
                            matAlg->matMultiplication_aHb(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &cellGrpPrmsCpu->estH_fr[hMatStart], nBsAnt, EMat);
                        } else if (precodingScheme == 1) { // SVD precoding
                            uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                            matAlg->matMultiplication_ab(&cellGrpPrmsCpu->estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);
                            matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                            matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                        } else { 
                            printf("Error: precoding type is not supported.");
                            return;
                        }
    
                        for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                            EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                        }
                        matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);
    
                        float dataRate = 0;
                        for (int j = 0; j < nUeAnt; j++) {
                            dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                        }
                        perUeThr[ueIdx] += dataRate;
                    }
                }
            }
        }
    
        sumInsThrRecordsCpu[slotIdx] = 0;
        for (int uIdx = 0; uIdx < nUe; uIdx++) {
            bool coordinatedCellUe = false;
            for (int cIdx = 0; cIdx < nCell; cIdx++) {
                if (cellGrpPrmsCpu->cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                    coordinatedCellUe = true;
                    break;
                }
            }
    
            cellGrpUeStatusCpu->avgRates[uIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[uIdx] + pfAvgRateUpd*perUeThr[uIdx];
    
            if (coordinatedCellUe) {
                sumInsThrRecordsCpu[slotIdx] += perUeThr[uIdx];
            }
        }

        printf("CPU scheduler sum instantaneous rate: %4.3e\n", sumInsThrRecordsCpu[slotIdx]);
    
        delete perUeThr;
        delete BMat;
        delete CMat;
        delete CInvMat;
        delete DMat;
        delete EMat;
        delete EInvMat;
 }

 void network::updateDataRateCpu(int slotIdx)
 {
    // verify slotIdx
    if (slotIdx < 0) {
        printf("Error: invalid time slot index");
        return;
    }

    float* perUeThr     = new float[nUe];
    cuComplex* BMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* CMat     = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* CInvMat  = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* DMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* EMat     = new cuComplex[nBsAnt*nBsAnt];
    cuComplex* EInvMat  = new cuComplex[nBsAnt*nBsAnt];

    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        perUeThr[uIdx] = 0;
    }

    // determine per-UE instantaneous data rates based on PRB allocate type (type-0 or type-1)
    if (cpuAllocType) {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellGrpPrmsCpu->cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine RBG-to-UE mapping
        int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;

                for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                    if (assocCellIdx[ueIdx] != cIdx)
                        continue;

                    if (rbgIdx >= schdSolCpu->allocSol[2*ueIdx] && rbgIdx < schdSolCpu->allocSol[2*ueIdx+1]) {
                        allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                    }  
                }
            }
        }
 
        // determine per-UE instantaneous data rates
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    for (int l = 0; l < totNumCell; l++) {
                        if (l == cIdx) 
                            continue;

                        int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                        if (uePrimeIdx < 0)
                            continue;

                        uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                        matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                    }
                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }

        delete assocCellIdx;
        delete allocSol_rbg2Ue;
    } else {
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    for (int l = 0; l < totNumCell; l++) {
                        if (l == cIdx) 
                            continue;

                        int uePrimeIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+l];
                        if (uePrimeIdx < 0)
                            continue;

                        uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                        matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                    }
                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else if (precodingScheme == 1) { // SVD precoding
                        uint32_t vMatStart = rbgIdx*nUe*totNumCell*nBsAnt*nBsAnt + ueIdx*totNumCell*nBsAnt*nBsAnt + cIdx*nBsAnt*nBsAnt;
                        matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);
                        matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }
    }

    sumCellThrRecordsCpu[slotIdx] = 0;
    sumInsThrRecordsCpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        bool coordinatedCellUe = false;
        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            if (cellGrpPrmsCpu->cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                coordinatedCellUe = true;
                break;
            }
        }

        cellGrpUeStatusCpu->avgRates[uIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[uIdx] + pfAvgRateUpd*perUeThr[uIdx];

        if (coordinatedCellUe) {
            sumCellThrRecordsCpu[slotIdx] += cellGrpUeStatusCpu->avgRates[uIdx];
            sumInsThrRecordsCpu[slotIdx] += perUeThr[uIdx];
        }
    }

    delete perUeThr;
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
 }

 void network::updateDataRateGpu(int slotIdx)
 {
    // verify slotIdx
    if (slotIdx < 0) {
        printf("Error: invalid time slot index");
        return;
    }

/*    
    for (int cidx = 0; cidx < nCell; cidx++) {
        printf("Cell %d: ", cidx);
        for (int uidx = 0; uidx < nUe; uidx++) {
            printf("%d ", cellAssoc[cidx*nUe + uidx]);
        }
        printf("\n");
    }
*/    

    float* perUeThr     = new float[nUe];
    cuComplex* BMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* CMat     = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* CInvMat  = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* DMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* EMat     = new cuComplex[nBsAnt*nBsAnt];
    cuComplex* EInvMat  = new cuComplex[nBsAnt*nBsAnt];

    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        perUeThr[uIdx] = 0;
    }

    // determine per-UE instantaneous data rates based on PRB allocate type (type-0 or type-1)
    if (gpuAllocType) {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine RBG-to-UE mapping
        int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;

                for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                    if (assocCellIdx[ueIdx] != cIdx)
                        continue;

                    if (rbgIdx >= allocSol[2*ueIdx] && rbgIdx < allocSol[2*ueIdx+1]) {
                        allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                    }  
                }
            }
        }

        // determine per-UE instantaneous data rates
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    for (int l = 0; l < totNumCell; l++) {
                        if (l == cIdx) 
                            continue;

                        int uePrimeIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+l];
                        if (uePrimeIdx < 0)
                            continue;

                        uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                        matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                    }
                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else { 
                        uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                        matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, prdMat.get()+vMatStart, nBsAnt, BMat);
                        matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }

        delete assocCellIdx;
        delete allocSol_rbg2Ue;
    } else {
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = allocSol[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    for (int l = 0; l < totNumCell; l++) {
                        if (l == cIdx) 
                            continue;

                        int uePrimeIdx = allocSol[rbgIdx*totNumCell+l];
                        if (uePrimeIdx < 0)
                            continue;

                        uint32_t hInterfMatStart = hTemp+ l*nBsAnt*nUeAnt;
                        matAlg->matMultiplication_aaHplusb(&estH_fr[hInterfMatStart], nUeAnt, nBsAnt, CMat);
                    }
                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else if (precodingScheme == 1) { // SVD precoding
                        uint32_t vMatStart = (ueIdx*nPrbGrp + rbgIdx)*nBsAnt*nBsAnt;
                        matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, prdMat.get()+vMatStart, nBsAnt, BMat);
                        matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }
    }

    // update long-term average data rates
    sumInsThrRecordsGpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        bool coordinatedCellUe = false;
        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            if (cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                coordinatedCellUe = true;
                break;
            }
        }

        avgRates[uIdx] = (1.0-pfAvgRateUpd)*avgRates[uIdx] + pfAvgRateUpd*perUeThr[uIdx];

        if (coordinatedCellUe) {
            sumInsThrRecordsGpu[slotIdx] += perUeThr[uIdx];
        }
    }

    printf("GPU scheduler sum instantaneous rate: %4.3e\n", sumInsThrRecordsGpu[slotIdx]);

    delete perUeThr;
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
 }

 void network::updateDataRateUpperBndCpu(int slotIdx)
 {
    // verify slotIdx
    if (slotIdx < 0) {
        printf("Error: invalid time slot index");
        return;
    }

    float* perUeThr     = new float[nUe];
    cuComplex* BMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* CMat     = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* CInvMat  = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* DMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* EMat     = new cuComplex[nBsAnt*nBsAnt];
    cuComplex* EInvMat  = new cuComplex[nBsAnt*nBsAnt];

    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        perUeThr[uIdx] = 0;
    }

    // determine per-UE instantaneous data rates based on PRB allocate type (type-0 or type-1)
    if (cpuAllocType) {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine RBG-to-UE mapping
        int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;

                for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                    if (assocCellIdx[ueIdx] != cIdx)
                        continue;

                    if (rbgIdx >= schdSolCpu->allocSol[2*ueIdx] && rbgIdx < schdSolCpu->allocSol[2*ueIdx+1]) {
                        allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                    }  
                }
            }
        }
 
        // determine per-UE instantaneous data rates
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }

        delete assocCellIdx;
        delete allocSol_rbg2Ue;
    } else {
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else if (precodingScheme == 1) { // SVD precoding
                        uint32_t vMatStart = rbgIdx*nUe*totNumCell*nBsAnt*nBsAnt + ueIdx*totNumCell*nBsAnt*nBsAnt + cIdx*nBsAnt*nBsAnt;
                        matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);
                        matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }
    }

    sumCellThrRecordsCpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        bool coordinatedCellUe = false;
        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            if (cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                coordinatedCellUe = true;
                break;
            }
        }

        cellGrpUeStatusCpu->avgRates[uIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRates[uIdx] + pfAvgRateUpd*perUeThr[uIdx];

        if (coordinatedCellUe) {
            sumCellThrRecordsCpu[slotIdx] += cellGrpUeStatusCpu->avgRates[uIdx];
        }
    }

    delete perUeThr;
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
 }


 void network::updateDataRateUpperBndGpu(int slotIdx)
 {
    // verify slotIdx
    if (slotIdx < 0) {
        printf("Error: invalid time slot index");
        return;
    }

    float* perUeThr     = new float[nUe];
    cuComplex* BMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* CMat     = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* CInvMat  = new cuComplex[nUeAnt*nUeAnt];
    cuComplex* DMat     = new cuComplex[nBsAnt*nUeAnt];
    cuComplex* EMat     = new cuComplex[nBsAnt*nBsAnt];
    cuComplex* EInvMat  = new cuComplex[nBsAnt*nBsAnt];

    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        perUeThr[uIdx] = 0;
    }

    // determine per-UE instantaneous data rates based on PRB allocate type (type-0 or type-1)
    if (gpuAllocType) {
        // determine cell associate 
        int* assocCellIdx = new int[nUe];
        for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
            assocCellIdx[ueIdx] = -1;
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) { 
                if (cellAssoc[cIdx*nUe + ueIdx]) {
                    assocCellIdx[ueIdx] =cIdx;
                    break;
                }
            }
        }

        // determine RBG-to-UE mapping
        int16_t* allocSol_rbg2Ue = new int16_t[totNumCell*nPrbGrp];
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = -1;

                for (int ueIdx = 0; ueIdx<nUe; ueIdx++) {
                    if (assocCellIdx[ueIdx] != cIdx)
                        continue;

                    if (rbgIdx >= allocSol[2*ueIdx] && rbgIdx < allocSol[2*ueIdx+1]) {
                        allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx] = ueIdx;
                    }  
                }
            }
        }

        // determine per-UE instantaneous data rates
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = allocSol_rbg2Ue[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }

        delete assocCellIdx;
        delete allocSol_rbg2Ue;
    } else {
        for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                int ueIdx = allocSol[rbgIdx*totNumCell+cIdx];

                if (ueIdx >= 0) {
                    uint32_t hTemp = rbgIdx*nUe*totNumCell*nBsAnt*nUeAnt+ ueIdx*totNumCell*nBsAnt*nUeAnt;
                    for (int rowIdx = 0; rowIdx < nUeAnt; rowIdx++) {
                        for (int colIdx = 0; colIdx < nUeAnt; colIdx++) {
                            if (rowIdx == colIdx) {
                                CMat[colIdx*nUeAnt + rowIdx].x = sigmaSqrd;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            } else {
                                CMat[colIdx*nUeAnt + rowIdx].x = 0;
                                CMat[colIdx*nUeAnt + rowIdx].y = 0;
                            }
                        }
                    }

                    matAlg->matInverseEigen(CMat, nUeAnt, CInvMat);

                    uint32_t hMatStart = hTemp + cIdx*nBsAnt*nUeAnt;

                    if (precodingScheme == 0) { // no precoding
                        matAlg->matMultiplication_aHb(&estH_fr[hMatStart], nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, &estH_fr[hMatStart], nBsAnt, EMat);
                    } else if (precodingScheme == 1) { // SVD precoding
                        uint32_t vMatStart = rbgIdx*nUe*totNumCell*nBsAnt*nBsAnt + ueIdx*totNumCell*nBsAnt*nBsAnt + cIdx*nBsAnt*nBsAnt;
                        matAlg->matMultiplication_ab(&estH_fr[hMatStart], nUeAnt, nBsAnt, &cellGrpPrmsCpu->prdMat[vMatStart], nBsAnt, BMat);
                        matAlg->matMultiplication_aHb(BMat, nUeAnt, nBsAnt, CInvMat, nUeAnt, DMat);
                        matAlg->matMultiplication_ab(DMat, nBsAnt, nUeAnt, BMat, nBsAnt, EMat);
                    } else { 
                        printf("Error: precoding type is not supported.");
                        return;
                    }

                    for (int rowIdx = 0; rowIdx < nBsAnt; rowIdx++) {
                        EMat[rowIdx*nBsAnt+rowIdx].x += 1.0;
                    }
                    matAlg->matInverseEigen(EMat, nBsAnt, EInvMat);

                    float dataRate = 0;
                    for (int j = 0; j < nUeAnt; j++) {
                        dataRate += W*log2f(1.0/EInvMat[j*nBsAnt+j].x);
                    }
                    perUeThr[ueIdx] += dataRate;
                }
            }
        }
    }

    // update long-term average data rates
    sumCellThrRecordsGpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        bool coordinatedCellUe = false;
        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            if (cellAssoc[cellId[cIdx]*nUe + uIdx]) {
                coordinatedCellUe = true;
                break;
            }
        }

        avgRates[uIdx] = (1.0-pfAvgRateUpd)*avgRates[uIdx] + pfAvgRateUpd*perUeThr[uIdx];

        if (coordinatedCellUe) {
            sumCellThrRecordsGpu[slotIdx] += avgRates[uIdx];
        }
    }

    delete perUeThr;
    delete BMat;
    delete CMat;
    delete CInvMat;
    delete DMat;
    delete EMat;
    delete EInvMat;
 }


 uint8_t* network::getCellAssoc()
 {
    return cellAssoc.get();
 }

 cuComplex* network::getEstH_fr()
 {
    return estH_fr.get();
 }

 uint16_t* network::getCellID()
 {
    return cellId.get();
 }

 uint16_t* network::getInterfCellID()
 {
    return interfCellId.get();
 }

 int16_t* network::getGpuAllocSol()
 {
    return allocSol.get();
 }

 bool network::compareCpuGpuAllocSol()
 {
    if (gpuAllocType != cpuAllocType) {
        throw std::runtime_error("CPU and GPU schedulers are of different allocation types");
        return false;
    }

    CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTI.get(), schdSolGpu->setSchdUePerCellTTI, setSchdUeSolSize, cudaMemcpyDeviceToHost)); 

    std::vector<uint16_t> ueSelArrGpu(nUe);
    std::vector<uint16_t> ueSelArrCpu(nUe);
    std::vector<uint16_t> ueSelIdxGpu(nUe);
    std::vector<uint16_t> ueSelIdxCpu(nUe);
    for (int idx = 0; idx < nUe; idx++) {
        ueSelArrGpu[idx] = setSchdUePerCellTTI[idx];
        ueSelArrCpu[idx] = schdSolCpu->setSchdUePerCellTTI[idx];
        ueSelIdxGpu[idx] = idx;
        ueSelIdxCpu[idx] = idx;
    }

    std::sort(ueSelIdxGpu.begin(), ueSelIdxGpu.end(),
              [&ueSelArrGpu](uint16_t i1, uint16_t i2) {
                  return ueSelArrGpu[i1] > ueSelArrGpu[i2];
              });

    std::sort(ueSelIdxCpu.begin(), ueSelIdxCpu.end(),
              [&ueSelArrCpu](uint16_t i1, uint16_t i2) {
                  return ueSelArrCpu[i1] > ueSelArrCpu[i2];
              });

    /*
    printf("GPU UE sel.: ");
    for (int idx = 0; idx < nUe; idx++) {
        printf("(%d, %d) ", ueSelArrGpu[ueSelIdxGpu[idx]], ueSelIdxGpu[idx]);
    }
    printf("\n");

    printf("CPU UE sel.: ");
    for (int idx = 0; idx < nUe; idx++) {
        printf("(%d, %d) ", ueSelArrCpu[ueSelIdxCpu[idx]], ueSelIdxCpu[idx]);
    }
    printf("\n");
    */

    bool matchUeSel     = true;
    bool matchLayerSel  = true;
    bool matchMcsSel    = true;
    bool matchPrg       = true;

    // Per-cell set comparison for UE selection (order of selected UEs in each cell can be different)
    for (int cIdx = 0; cIdx < nCell; cIdx++) {
        int offset = cIdx * numUeSchdPerCellTTI;

        std::vector<uint16_t> cpuSet(schdSolCpu->setSchdUePerCellTTI + offset,
                                     schdSolCpu->setSchdUePerCellTTI + offset + numUeSchdPerCellTTI);
        std::vector<uint16_t> gpuSet(setSchdUePerCellTTI.get() + offset,
                                     setSchdUePerCellTTI.get() + offset + numUeSchdPerCellTTI);

        std::sort(cpuSet.begin(), cpuSet.end());
        std::sort(gpuSet.begin(), gpuSet.end());

        if (cpuSet != gpuSet) {
            matchUeSel = false;
            printf("Failure: CPU and GPU UE selection mismatch at cell %d\n", cIdx);
            for (int uIdx = 0; uIdx < numUeSchdPerCellTTI; uIdx++) {
                printf("  cell %d, uIdx %d: CPU = %d, GPU = %d\n", cIdx, uIdx, cpuSet[uIdx], gpuSet[uIdx]);
            }
        }
    }
    
    for (int uIdx = 0; uIdx<nUe; uIdx++) {
        if (schdSolCpu->layerSelSol[ueSelIdxCpu[uIdx]] != layerSelSol[ueSelIdxGpu[uIdx]]) {
            matchLayerSel = false;
        }

        // printf("CPU uIdx = %d, GPU uIdx = %d, schdSolCpu->mcsSelSol[uIdx] = %d, mcsSelSol[uIdx] = %d\n", ueSelIdxCpu[uIdx], ueSelIdxGpu[uIdx], schdSolCpu->mcsSelSol[ueSelIdxCpu[uIdx]], mcsSelSol[ueSelIdxGpu[uIdx]]);
        if (schdSolCpu->mcsSelSol[ueSelIdxCpu[uIdx]] != mcsSelSol[ueSelIdxGpu[uIdx]]) {
            matchMcsSel = false;
        }
    }

    if (gpuAllocType == 1) {
        for (int uIdx = 0; uIdx<nUe; uIdx++) {
            // printf("uIdx = %d, CPU-(%d, %d), GPU-(%d, %d)\n", uIdx, schdSolCpu->allocSol[2*ueSelIdxCpu[uIdx]], schdSolCpu->allocSol[2*ueSelIdxCpu[uIdx]+1], allocSol[2*ueSelIdxGpu[uIdx]], allocSol[2*ueSelIdxGpu[uIdx]+1]);
            if (schdSolCpu->allocSol[2*ueSelIdxCpu[uIdx]] != allocSol[2*ueSelIdxGpu[uIdx]] || schdSolCpu->allocSol[2*ueSelIdxCpu[uIdx]+1] != allocSol[2*ueSelIdxGpu[uIdx]+1]) {
                matchPrg = false;
                break;
            }
        }
    } else {
        for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
            for (int rbgIdx = 0; rbgIdx < nPrbGrp; rbgIdx++) {
                //printf("rbgIdx*totNumCell+cIdx = %d, schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx] = %d, allocSol[rbgIdx*totNumCell+cIdx] = %d\n", rbgIdx*totNumCell+cIdx, schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx], allocSol[rbgIdx*totNumCell+cIdx]);
                if (schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx] != allocSol[rbgIdx*totNumCell+cIdx]) {
                    matchPrg = false;
                    //printf("cellAssoc[cIdx*nUe + allocSol[rbgIdx*totNumCell+cIdx]] = %d\n", cellAssoc[cIdx*nUe + allocSol[rbgIdx*totNumCell+cIdx]]);
                    //printf("cellAssoc[cIdx*nUe + schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx]] = %d\n", cellAssoc[cIdx*nUe + schdSolCpu->allocSol[rbgIdx*totNumCell+cIdx]]);
                    break;
                }
            }
        }
    }

    if (matchUeSel) {
        printf("Success: CPU and GPU UE selection solutions match\n");
    } else {
        printf("Failure: CPU and GPU UE selection solutions do not match\n");
    }
    
    if (matchPrg) {
        printf("Success: CPU and GPU PRG allocation solutions match\n");
    } else {
        printf("Failure: CPU and GPU PRG allocation solutions do not match\n");
    }

    if (matchLayerSel) {
        printf("Success: CPU and GPU layer selection solutions match\n");
    } else {
        printf("Failure: CPU and GPU layer selection solutions do not match\n");
    }

    if (matchMcsSel) {
        printf("Success: CPU and GPU MCS selection solutions match\n");
    } else {
        printf("Failure: CPU and GPU MCS selection solutions do not match\n");
    }

    bool matchH     = true;
    for (int pIdx = 0; pIdx < nPrbGrp; pIdx++) {
        for (int uIdx = 0; uIdx < nUe; uIdx++) {
            for (int cIdx = 0; cIdx < totNumCell; cIdx++) {
                for (int aIdx = 0; aIdx < nBsAnt*nUeAnt; aIdx++) {
                    if (cellGrpPrmsCpu->estH_fr[pIdx*nUe*totNumCell*nBsAnt*nUeAnt+
                                            ueSelIdxCpu[uIdx]*totNumCell*nBsAnt*nUeAnt+
                                            cIdx*nBsAnt*nUeAnt+aIdx].x != 
                                    estH_fr[pIdx*nUe*totNumCell*nBsAnt*nUeAnt+
                                            ueSelIdxGpu[uIdx]*totNumCell*nBsAnt*nUeAnt+
                                            cIdx*nBsAnt*nUeAnt+aIdx].x ||
                        cellGrpPrmsCpu->estH_fr[pIdx*nUe*totNumCell*nBsAnt*nUeAnt+
                                            ueSelIdxCpu[uIdx]*totNumCell*nBsAnt*nUeAnt+
                                            cIdx*nBsAnt*nUeAnt+aIdx].y != 
                                    estH_fr[pIdx*nUe*totNumCell*nBsAnt*nUeAnt+
                                            ueSelIdxGpu[uIdx]*totNumCell*nBsAnt*nUeAnt+
                                            cIdx*nBsAnt*nUeAnt+aIdx].y) {
                        matchH = false;
                        break;
                    }
                    if (!matchH) {
                        break;
                    }
                }
                if (!matchH) {
                    break;
                }
            }
            if (!matchH) {
                break;
            }
        }
        if (!matchH) {
            break;
        }
    }

    if (matchH) {
        printf("Success: CPU and GPU channels match\n");
    } else {
        printf("Failure: CPU and GPU channels do not match\n");
    }

    return matchH & matchUeSel & matchLayerSel & matchMcsSel & matchPrg;
 }

 bool network::compareCpuGpuSchdPerf()
 {
    bool perUePerfCheckPass;

    // Kolmogorov–Smirnov test of CPU and GPU per-UE perf CDF curves
    float* avgRatesActUeGpuTemp = new float[nActiveUe];
    float* avgRatesActUeCpuTemp = new float[nActiveUe];
    std::memcpy(avgRatesActUeGpuTemp, avgRatesActUe.get(), sizeof(float)*nActiveUe);
    std::memcpy(avgRatesActUeCpuTemp, cellGrpUeStatusCpu->avgRatesActUe, sizeof(float)*nActiveUe);
    std::sort(avgRatesActUeGpuTemp, avgRatesActUeGpuTemp + nActiveUe);
    std::sort(avgRatesActUeCpuTemp, avgRatesActUeCpuTemp + nActiveUe);

    int numTestPerc = 200;
    while (numTestPerc >= nActiveUe) {
        numTestPerc /= 10;
    }
    numTestPerc = numTestPerc >= 1 ? numTestPerc : 1;    

    float* xPoints = new float[numTestPerc];
    float* percentilesCpu = new float[numTestPerc];
    float* percentilesGpu = new float[numTestPerc];
    float xGap = (avgRatesActUeCpuTemp[nActiveUe-1] - avgRatesActUeCpuTemp[0])/static_cast<float>(numTestPerc);

    for (int xIdx = 0; xIdx < numTestPerc; xIdx++) {
        xPoints[xIdx] = avgRatesActUeCpuTemp[0] + (xIdx+1)*xGap;
        percentilesCpu[xIdx] = 0;
        percentilesGpu[xIdx] = 0;
    }

    for (int idx = 0; idx < nActiveUe; idx++) {
        for (int xIdx = 0; xIdx < numTestPerc; xIdx++) {
            if (avgRatesActUeCpuTemp[idx] <= xPoints[xIdx]) { 
                percentilesCpu[xIdx] += 1.0;
            }

            if (avgRatesActUeGpuTemp[idx] <= xPoints[xIdx]) { 
                percentilesGpu[xIdx] += 1.0;
            }
        }
    }

    float maxGap = 0;
    for (int xIdx = 0; xIdx < numTestPerc; xIdx++) {
        percentilesGpu[xIdx] /= static_cast<float>(nActiveUe);
        percentilesCpu[xIdx] /= static_cast<float>(nActiveUe);
        float tempGap = fabs(percentilesGpu[xIdx] - percentilesCpu[xIdx]);

        if (tempGap > maxGap) {
            maxGap = tempGap;
        }
    }

    if (maxGap > cpuGpuPerfGapPerUeConst) {
        perUePerfCheckPass = false;
        printf("CPU and GPU scheduler per-UE throughput performance check result: FAIL\n");
    } else {
        perUePerfCheckPass = true;
        printf("CPU and GPU scheduler per-UE throughput performance check result: PASS\n");
    }
    printf("Largest gap (in percentage) between CPU and GPU per-UE throughput CDFs = %f%%\n", 100.0*maxGap);
    /*
    // test
    printf("CPU: ");
    for (int xIdx = 0; xIdx < numTestPerc; xIdx++) {
        printf("%f ", percentilesCpu[xIdx]);
    }
    printf("\n");

    printf("GPU: ");
    for (int xIdx = 0; xIdx < numTestPerc; xIdx++) {
        printf("%f ", percentilesGpu[xIdx]);
    }
    printf("\n");
    //
    */

    bool sumRPerfCheckPass;
    float cpuGpuGapSumR = 0;
    for (int tIdx = floor(numSimChnRlz*0.1); tIdx < numSimChnRlz; tIdx++) {
        float temp = fabs(sumCellThrRecordsGpu[tIdx] - sumCellThrRecordsCpu[tIdx])/sumCellThrRecordsCpu[tIdx];
        if (temp > cpuGpuGapSumR) {
            cpuGpuGapSumR = temp;
        }
    }
    
    if (cpuGpuGapSumR > cpuGpuPerfGapSumRConst) {
        sumRPerfCheckPass = false;
        printf("CPU and GPU scheduler sum throughput performance check result: FAIL\n");
    } else {
        sumRPerfCheckPass = true;
        printf("CPU and GPU scheduler sum throughput performance check result: PASS\n");
    }
    printf("Largest gap (in percentage) between CPU and GPU sum throughput curves = %f%%\n", 100.0*cpuGpuGapSumR);

    delete avgRatesActUeGpuTemp;
    delete avgRatesActUeCpuTemp;
    delete xPoints;
    delete percentilesCpu;
    delete percentilesGpu;

    return perUePerfCheckPass & sumRPerfCheckPass;
 }

 void network::compareCpuGpuCellAssocSol()
 {
    for(int ueIdx = 0; ueIdx < nUe; ueIdx++) 
    {
        for (int cellIdx = 0; cellIdx < totNumCell; cellIdx++) 
        {
            if(cellAssoc[cellIdx*nUe + ueIdx] != cellGrpPrmsCpu->cellAssoc[cellIdx*nUe + ueIdx]) // either 0 or same with preset cell associate
            {
                printf("Failure: CPU and GPU cell association solutions do not match\n");
                return;
            }
        }
    }
    printf("Success: CPU and GPU cell association solutions match\n");
}

void network::writeToFile()
{
    std::ofstream file;
    file.open(mcOutputFile, std::fstream::out);

    if (schedulerType == 0) { // single-cell scheduler
        if (cpuAllocType) { // type-1 consecutive allocate
            file << "scType1SumCellThrRecordsCpu = [";
        } else { // type-0 consecutive allocate
            file << "scType0SumCellThrRecordsCpu = [";
        }
        
    } else { // multi-cell scheduler
        if (cpuAllocType) { // type-1 consecutive allocate
            file << "mcType1SumCellThrRecordsCpu = [";
        } else { // type-0 consecutive allocate
            file << "mcType0SumCellThrRecordsCpu = [";
        }
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellThrRecordsCpu[slotIdx] << " ";
    }
    file << sumCellThrRecordsCpu[numSimChnRlz-1] << "];\n";

    if (schedulerType == 0) { // single-cell scheduler
        if (gpuAllocType) { // type-1 consecutive allocate
            file << "scType1SumCellThrRecordsGpu = [";
        } else { // type-0 consecutive allocate
            file << "scType0SumCellThrRecordsGpu = [";
        }
    } else { // multi-cell scheduler
        if (gpuAllocType) { // type-1 consecutive allocate
            file << "mcType1SumCellThrRecordsGpu = [";
        } else { // type-0 consecutive allocate
            file << "mcType0SumCellThrRecordsGpu = [";
        }
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellThrRecordsGpu[slotIdx] << " ";
    }
    file << sumCellThrRecordsGpu[numSimChnRlz-1] << "];\n";

    ////////////////////
    if (schedulerType == 0) { // single-cell scheduler
        if (cpuAllocType) { // type-1 consecutive allocate
            file << "scType1SumInsThrRecordsCpu = [";
        } else { // type-0 consecutive allocate
            file << "scType0SumInsThrRecordsCpu = [";
        }
        
    } else { // multi-cell scheduler
        if (cpuAllocType) { // type-1 consecutive allocate
            file << "mcType1SumInsThrRecordsCpu = [";
        } else { // type-0 consecutive allocate
            file << "mcType0SumInsThrRecordsCpu = [";
        }
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumInsThrRecordsCpu[slotIdx] << " ";
    }
    file << sumInsThrRecordsCpu[numSimChnRlz-1] << "];\n";

    if (schedulerType == 0) { // single-cell scheduler
        if (gpuAllocType) { // type-1 consecutive allocate
            file << "scType1SumInsThrRecordsGpu = [";
        } else { // type-0 consecutive allocate
            file << "scType0SumInsThrRecordsGpu = [";
        }
    } else { // multi-cell scheduler
        if (gpuAllocType) { // type-1 consecutive allocate
            file << "mcType1SumInsThrRecordsGpu = [";
        } else { // type-0 consecutive allocate
            file << "mcType0SumInsThrRecordsGpu = [";
        }
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumInsThrRecordsGpu[slotIdx] << " ";
    }
    file << sumInsThrRecordsGpu[numSimChnRlz-1] << "];\n";
    ////////////////////

    file << "mcType1AvgRatesCpu = [";
    for (int uIdx = 0; uIdx < netData->numCell*netData->numActiveUesPerCell-1; uIdx++) {
        file << netData->avgRatesCPU[uIdx] << " ";
    }
    file << netData->avgRatesCPU[netData->numCell*netData->numActiveUesPerCell-1] << "];\n";

    file << "mcType1AvgRatesGpu = [";
    for (int uIdx = 0; uIdx < netData->numCell*netData->numActiveUesPerCell-1; uIdx++) {
        file << netData->avgRatesGPU[uIdx] << " ";
    }
    file << netData->avgRatesGPU[netData->numCell*netData->numActiveUesPerCell-1] << "];\n";

    file.close();
}

void network::convertPrbgAllocToPrbAlloc()
{
    for(uint idx = 0; idx < gpuAllocSolSize / sizeof(int16_t); idx ++)
    {
        allocSolPrb[idx] = (allocSol[idx] < 0) ? -1 : (allocSol[idx] * nPrbPerGrp);
    }
}

void network::saveSolH5()
{
    try {
        /*---------    Start saving solution ---------------*/
        // Create a new HDF5 file
        H5::H5File file("cumacSol.h5", H5F_ACC_TRUNC);

        /*---------    Save PRB allocations ---------------*/
        // Define the data type for PRB allocations 
        H5::DataType datatype(H5::PredType::NATIVE_INT16);

        // Create the dataspace for the dataset
        hsize_t dims[1] = {static_cast<hsize_t>(nUe*2)};
        H5::DataSpace dataspace(1, dims);

        // Create the dataset in the file
        H5::DataSet dataset = file.createDataSet("prbAlloc", datatype, dataspace);

        // Write PRB allocations  to the dataset
        dataset.write(allocSolPrb.get(), datatype);

        // Close the dataset
        dataset.close();

        /*---------    End saving solution ---------------*/
        printf("Solution saved to cumacSol.h5: %ld prbAlloc\n", gpuAllocSolSize / sizeof(int16_t));
        file.close();
    }
    catch (H5::Exception& error) {
        // Handle any HDF5 errors that may occur
        std::cerr << "Error: " << error.getDetailMsg() << std::endl;
        exit(-1);
    }
}

void network::saveSolMatx()
{/*
    try
    {
        // TODO: need to get nUE in current time slot; May consider use GPU memory ptr "schdSolGpu->allocSol" later
        auto prbAllocSolTensor = matx::make_tensor<int16_t>({nUe, 2}); // make_tensor must use device ptr 
        for(int ueIdx = 0; ueIdx < nUe; ueIdx ++)
        {
            prbAllocSolTensor(ueIdx, 0) = allocSolPrb[ueIdx * 2];
            prbAllocSolTensor(ueIdx, 1) = allocSolPrb[ueIdx * 2 + 1];
        }
        matx::io::write_mat(prbAllocSolTensor, "cumacSol.mat", "prbAlloc");
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }*/
}


uint32_t network::determineTbsPdsch(int rbSize, int nDataSymb, int nrOfLayers, float codeRate, int qam)
{
    uint32_t TBS = 0;
    // current assumption is that resource allocation is type-1
    float Ninfo = static_cast<float>(rbSize*12*nDataSymb*nrOfLayers*codeRate*qam);

    if (Ninfo <= 3824.0) { // for small size
        int temp = floor(log2(Ninfo)) - 6;
        int n = temp > 3 ? temp : 3;
        temp = static_cast<int>(pow(2.0, static_cast<float>(n)) * floor(Ninfo / pow(2.0, static_cast<float>(n))));
        int Ninfo_prime = temp > 24 ? temp : 24;

         
        for (int tbsTbIdx = 0; tbsTbIdx < TBS_table_size; tbsTbIdx++) {
            if (TBS_table[tbsTbIdx] >= Ninfo_prime) {
                TBS = TBS_table[tbsTbIdx];
                break;
            }
        }
    } else { // for large size
        int n = floor(log2(Ninfo-24.0)) - 5;
        int temp = static_cast<int>(pow(2.0, static_cast<float>(n))*round((Ninfo-24.0)/pow(2.0, static_cast<float>(n))));
        float Ninfo_prime = static_cast<float>(temp > 3840 ? temp : 3840);

        if (codeRate < 0.25) {
            int C = ceil( (Ninfo + 24.0) / 3816.0);
            TBS = 8*C*ceil( (Ninfo_prime + 24.0) / (8.0*C) ) - 24;
        } else {
            if (Ninfo_prime > 8424.0) {
                int C = ceil( (Ninfo_prime + 24.0) / 8424.0);
                TBS = 8*C*ceil( (Ninfo_prime + 24.0) / (8.0*C) ) - 24;
            } else {
                int C = 1;
                TBS = 8*C*ceil( (Ninfo_prime + 24.0) / (8.0*C) ) - 24;
            }
        }
    }

    return TBS;
}

void network::genNetTopology()
{
    float Angle = M_PI/6.0;
    int bsIdx = 0;

    for (int celli = 0; celli < 7; celli++) {
        float coorX = 0;
        float coorY = 0;
        if (celli > 0) {
            coorX = sin(Angle)*2.0*netData->cellRadius;
            coorY = cos(Angle)*2.0*netData->cellRadius;
            Angle += M_PI/3.0;
        }

        for (int secIdx = 0; secIdx < 3; secIdx++) {
            if (bsIdx < netData->numCell) {
                netData->bsPos[bsIdx][0] = coorX;
                netData->bsPos[bsIdx][1] = coorY;
                netData->bsPos[bsIdx][2] = netData->bsHeight;

                for (int uIdx = 0; uIdx < netData->numActiveUesPerCell; uIdx++) {
                    float randomAngle = 2.0*M_PI*uniformRealDist(randomEngine)/3.0 - M_PI/3.0; // centered at 0 degree
                    randomAngle += netData->sectorOrien[secIdx];
                    float randomDistance = (netData->cellRadius - netData->minD2Bs)*uniformRealDist(randomEngine) + netData->minD2Bs;

                    netData->uePos[bsIdx*netData->numActiveUesPerCell+uIdx][0] = cos(randomAngle)*randomDistance+netData->bsPos[bsIdx][0];
                    netData->uePos[bsIdx*netData->numActiveUesPerCell+uIdx][1] = sin(randomAngle)*randomDistance+netData->bsPos[bsIdx][1];
                    netData->uePos[bsIdx*netData->numActiveUesPerCell+uIdx][2] = netData->ueHeight;
                }
                bsIdx++;
            } else
                break;
        }

        if (bsIdx == netData->numCell)
            break;
    }
}

void network::genLSFading()
{
    std::lognormal_distribution<float> ln_distribution(0.0, netData->sfStd);

    // for testing
    float** snrDBAssoc = new float*[netData->numCell];

    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) { // loop through all cells
        snrDBAssoc[cIdx] = new float[netData->numActiveUesPerCell]; 
        int sectorIdx = cIdx % 3;
        float sectorOrien = netData->sectorOrien[sectorIdx];
        for (int uIdx = 0; uIdx < netData->numCell*netData->numActiveUesPerCell; uIdx++) { // loop through all active UEs
            
            float distanceBsUe_2D = sqrt(pow(netData->bsPos[cIdx][0] - netData->uePos[uIdx][0], 2.0)+
                                    pow(netData->bsPos[cIdx][1] - netData->uePos[uIdx][1], 2.0));
            float deltaH = netData->bsPos[cIdx][2] - netData->uePos[uIdx][2];
            float distanceBsUe_3D = sqrt(pow(distanceBsUe_2D, 2.0) + pow(deltaH, 2.0));

            // antenna attenuation
            float theta = 180.0 - atan(distanceBsUe_2D/deltaH)*180.0/M_PI;
            float phi = atan2(netData->uePos[uIdx][1] - netData->bsPos[cIdx][1], 
                        netData->uePos[uIdx][0] - netData->bsPos[cIdx][0]);
            phi = phi>=0?phi:(2.0*M_PI+phi);
            float degreeGap = phi - sectorOrien;
            if (degreeGap > M_PI) {
                degreeGap = 2.0*M_PI - degreeGap;
            } else if (degreeGap < -M_PI) {
                degreeGap = 2.0*M_PI + degreeGap;
            }
            degreeGap = degreeGap*180.0/M_PI;

            float antAttenVerti = -std::min(12.0*pow((theta-netData->bsAntDownTilt)/65.0, 2.0), 30.0);
            float antAttenHoriz = -std::min(12.0*pow(degreeGap/65.0, 2.0), 30.0);
            float antAtten = -std::min(-(antAttenVerti+antAttenHoriz), float(30.0));
            float antGain = netData->GEmax + antAtten;
            // pathloss + shadow fading
            float PL = 32.4+20.0*log10(netData->carrierFreq)+30.0*log10(distanceBsUe_3D);
            float SF=10.0*log10(ln_distribution(randomEngine));

            netData->rxSigPowDB[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx] = netData->bsTxPower_perAntPrg + antGain - PL - SF;
            netData->rxSigPowDB_UL[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx] = netData->ueTxPower_perAntPrg + antGain - PL - SF;

            if (floor(uIdx/netData->numActiveUesPerCell) == cIdx) {
                if (DL == 1) { // DL
                    snrDBAssoc[cIdx][uIdx%netData->numActiveUesPerCell] = netData->rxSigPowDB[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx] - netData->noiseFloor;
                } else { // UL
                    snrDBAssoc[cIdx][uIdx%netData->numActiveUesPerCell] = netData->rxSigPowDB_UL[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx] - netData->noiseFloor;
                }
            }
        }
    }

    std::ofstream file;
    file.open("snr.txt", std::fstream::out);

    file << "snrDBAssoc = [";
    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
        for (int uIdx = 0; uIdx < netData->numActiveUesPerCell; uIdx++) {
            if ((cIdx*netData->numActiveUesPerCell + uIdx)<(netData->numCell*netData->numActiveUesPerCell-1)) {
                file << snrDBAssoc[cIdx][uIdx] << " ";
            } else {
                file << snrDBAssoc[cIdx][uIdx] << "];\n\n";
            }
        }
    }
/*
    file << "snrDB = [";
    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
        for (int uIdx = 0; uIdx < netData->numCell*netData->numActiveUesPerCell; uIdx++) {
            if ((cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx)<(netData->numCell*netData->numCell*netData->numActiveUesPerCell-1)) {
                file << netData->rxSigPowDB[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx] - netData->noiseFloor<< " ";
            } else {
                file << netData->rxSigPowDB[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx] - netData->noiseFloor<< "];\n";
            }
        }
    }
*/
    file.close();

    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) delete snrDBAssoc[cIdx];
    delete snrDBAssoc;
}



__global__ void ueDownSel4TrKernel(uint16_t*        setSchdUePerCellTTIGpu,
                                   uint16_t*        cellIdArr,
                                   uint8_t*         cellAssocArr,
                                   cuComplex*       channMatGpu, 
                                   cuComplex*       estH_frGpu,
                                   __nv_bfloat162*  estH_frGpu_half,
                                   float*           avgRatesActUeGpu,
                                   float*           avgRatesGpu,
                                   int8_t*          tbErrLastActUe,
                                   int8_t*          tbErrLast,
                                   cuComplex*       prdMat_actUe,
                                   cuComplex*       prdMat,
                                   cuComplex*       detMat_actUe,
                                   cuComplex*       detMat,
                                   float*           sinVal_actUe,
                                   float*           sinVal,
                                   const int        nPrbGrp, 
                                   const int        numCell, 
                                   const int        nActiveUe,
                                   const int        numSchdUePerCell,
                                   const int        numTxAnt,
                                   const int        numRxAnt)
{
    uint16_t prgIdx               = floor(static_cast<float>(blockIdx.x)/numCell);
    uint16_t cIdx                 = blockIdx.x - prgIdx*numCell;
    uint16_t nTxRxAntPrd          = numTxAnt*numRxAnt;
    uint16_t nTxAntSqrd           = numTxAnt*numTxAnt;
    uint16_t nRxAntSqrd           = numRxAnt*numRxAnt;
    uint16_t maxAntDim            = numTxAnt > numRxAnt ? numTxAnt : numRxAnt;
    uint16_t maxAntDimSqrd        = maxAntDim*maxAntDim;
    uint16_t nUePerRnd            = floor(static_cast<float>(blockDim.x)/static_cast<float>(maxAntDimSqrd));
    uint16_t assocUeIdxInBlk      = floor(static_cast<float>(threadIdx.x)/static_cast<float>(maxAntDimSqrd));
    uint16_t eIdx                 = threadIdx.x - assocUeIdxInBlk*maxAntDimSqrd;
    uint16_t totNumSchdUe         = numCell*numSchdUePerCell;
    uint16_t numLayer             = numTxAnt > numRxAnt ? numRxAnt : numTxAnt;
    
    
    if (prgIdx == 0) {
        int schdUeIdStart = cIdx*numSchdUePerCell;
        int schdUeIdEnd = schdUeIdStart + numSchdUePerCell;

        for (int idx = threadIdx.x; idx < totNumSchdUe; idx += blockDim.x) {
            if (idx >= schdUeIdStart && idx < schdUeIdEnd) {
                if (setSchdUePerCellTTIGpu[idx] != 0xFFFF) {
                    cellAssocArr[cIdx*totNumSchdUe + idx] = 1;
                } else {
                    cellAssocArr[cIdx*totNumSchdUe + idx] = 0;
                }
            } else {
                cellAssocArr[cIdx*totNumSchdUe + idx] = 0;
            }
        }
    }

    if (threadIdx.x < nUePerRnd*maxAntDimSqrd) {
        for (int uIdx = assocUeIdxInBlk; uIdx < totNumSchdUe; uIdx += nUePerRnd) {
            int globalUeId = setSchdUePerCellTTIGpu[uIdx];
            if (globalUeId == 0xFFFF) {
                continue;
            }

            if (cIdx == 0 && prgIdx == 0 && eIdx == 0) {
                avgRatesGpu[uIdx] = avgRatesActUeGpu[globalUeId];
                tbErrLast[uIdx] = tbErrLastActUe[globalUeId];
            }

            if (eIdx < nTxRxAntPrd) {
                int globalChannIdx = prgIdx*numCell*nActiveUe*nTxRxAntPrd;
                globalChannIdx += globalUeId*numCell*nTxRxAntPrd;
                globalChannIdx += cIdx*nTxRxAntPrd;
                globalChannIdx += eIdx;

                int hChannIdx = prgIdx*numCell*totNumSchdUe*nTxRxAntPrd;
                hChannIdx += uIdx*numCell*nTxRxAntPrd;
                hChannIdx += cIdx*nTxRxAntPrd;
                hChannIdx += eIdx;

                estH_frGpu[hChannIdx].x = channMatGpu[globalChannIdx].x;
                estH_frGpu[hChannIdx].y = channMatGpu[globalChannIdx].y;

                estH_frGpu_half[hChannIdx] = __float22bfloat162_rn(channMatGpu[globalChannIdx]);
            }

            if (eIdx < nTxAntSqrd) {
                int globalPrdIdx = globalUeId*nPrbGrp*nTxAntSqrd + prgIdx*nTxAntSqrd + eIdx;
                int prdIdx = uIdx*nPrbGrp*nTxAntSqrd + prgIdx*nTxAntSqrd + eIdx;
            
                prdMat[prdIdx].x = prdMat_actUe[globalPrdIdx].x;
                prdMat[prdIdx].y = prdMat_actUe[globalPrdIdx].y;
            }
            
            if (eIdx < nRxAntSqrd) {
                int globalDetIdx = globalUeId*nPrbGrp*nRxAntSqrd + prgIdx*nRxAntSqrd + eIdx;
                int detIdx = uIdx*nPrbGrp*nRxAntSqrd + prgIdx*nRxAntSqrd + eIdx;

                detMat[detIdx].x = detMat_actUe[globalDetIdx].x;
                detMat[detIdx].y = detMat_actUe[globalDetIdx].y;
            }

            if (eIdx < numLayer) {
                int globalSinValIdx = globalUeId*nPrbGrp*numLayer + prgIdx*numLayer + eIdx;
                int sinValIdx = uIdx*nPrbGrp*numLayer + prgIdx*numLayer + eIdx;

                sinVal[sinValIdx] = sinVal_actUe[globalSinValIdx];
            }
        }
    }
}

__global__ void ueDownSel64TrKernel(uint16_t**       sortedUeList,
                                    cuComplex*       channMatGpu, 
                                    cuComplex**      srsEstChan,
                                    int32_t**        srsUeMap,
                                    const int        nPrbGrp, 
                                    const int        numCell, 
                                    const int        nActiveUe,
                                    const int        numUeForGrpPerCell,
                                    const int        numTxAnt,
                                    const int        numRxAnt)
{ // total number of threads = numTxAnt*numRxAnt
    uint16_t prgIdx               = floor(static_cast<float>(blockIdx.x)/numCell);
    uint16_t cIdx                 = blockIdx.x - prgIdx*numCell;
    uint16_t nTxRxAntPrd          = numTxAnt*numRxAnt;
    uint32_t nPrgBsUeAntPrd       = nPrbGrp*nTxRxAntPrd;

    if (prgIdx == 0) {
        for (int idx = threadIdx.x; idx < nActiveUe; idx += blockDim.x) {
            srsUeMap[cIdx][idx] = -1;
        }
    }
    __syncthreads(); 

    if (prgIdx == 0 && threadIdx.x < numUeForGrpPerCell) {
        srsUeMap[cIdx][sortedUeList[cIdx][threadIdx.x]] = threadIdx.x;
    }

    for (int uei = 0; uei < numUeForGrpPerCell; uei++) {
        for (int idx = threadIdx.x; idx < nTxRxAntPrd; idx += blockDim.x) {
            int globalChannIdx = prgIdx*numCell*nActiveUe*nTxRxAntPrd;
            globalChannIdx += sortedUeList[cIdx][uei]*numCell*nTxRxAntPrd;
            globalChannIdx += cIdx*nTxRxAntPrd;
            globalChannIdx += idx;
            srsEstChan[cIdx][uei*nPrgBsUeAntPrd + prgIdx*nTxRxAntPrd + idx].x = channMatGpu[globalChannIdx].x;
            srsEstChan[cIdx][uei*nPrgBsUeAntPrd + prgIdx*nTxRxAntPrd + idx].y = channMatGpu[globalChannIdx].y;
        }
    }
}

void network::ueDownSelectCpu()
{
    uint16_t numThrdPerBlk = nBsAnt*nBsAnt*numUeSchdPerCellTTI;
    numThrdPerBlk = numThrdPerBlk > 1024 ? 1024 : numThrdPerBlk;
    dim3 gridDim = {static_cast<uint16_t>(nPrbGrp*nCell), 1, 1};
    dim3 blockDim = {numThrdPerBlk, 1, 1};

    CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTIGpuforCpuSchd, schdSolCpu->setSchdUePerCellTTI, setSchdUeSolSize, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERR(cudaMemcpy(avgRatesActUeGpuforCpuSchd, cellGrpUeStatusCpu->avgRatesActUe, arActUeSize, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERR(cudaMemcpy(tbErrLastActUeGpuforCpuSchd, cellGrpUeStatusCpu->tbErrLastActUe, tbeActUeSize, cudaMemcpyHostToDevice));
    if (DL == 1) { // DL
        ueDownSel4TrKernel<<<gridDim, blockDim>>>(setSchdUePerCellTTIGpuforCpuSchd, cellGrpPrmsGpu->cellId, 
            cellAssocGpuforCpuSchd, cellGrpPrmsGpu->estH_fr_actUe, estH_fr_GPUforCpuSchd, estH_fr_GPUforCpuSchd_half,
            avgRatesActUeGpuforCpuSchd, avgRatesGpuforCpuSchd, tbErrLastActUeGpuforCpuSchd, tbErrLastGpuforCpuSchd, 
            cellGrpPrmsGpu->prdMat_actUe, prdMatGpuforCpuSchd, cellGrpPrmsGpu->detMat_actUe, detMatGpuforCpuSchd, 
            cellGrpPrmsGpu->sinVal_actUe, sinValGpuforCpuSchd, cellGrpPrmsCpu->nPrbGrp,
            cellGrpPrmsCpu->nCell, cellGrpPrmsCpu->nActiveUe, cellGrpPrmsCpu->numUeSchdPerCellTTI, 
            cellGrpPrmsCpu->nBsAnt, cellGrpPrmsCpu->nUeAnt);
    } else { // UL
        ueDownSel4TrKernel<<<gridDim, blockDim>>>(setSchdUePerCellTTIGpuforCpuSchd, cellGrpPrmsGpu->cellId, 
            cellAssocGpuforCpuSchd, cellGrpPrmsGpu->estH_fr_actUe, estH_fr_GPUforCpuSchd, estH_fr_GPUforCpuSchd_half,
            avgRatesActUeGpuforCpuSchd, avgRatesGpuforCpuSchd, tbErrLastActUeGpuforCpuSchd, tbErrLastGpuforCpuSchd, 
            cellGrpPrmsGpu->prdMat_actUe, prdMatGpuforCpuSchd, cellGrpPrmsGpu->detMat_actUe, detMatGpuforCpuSchd, 
            cellGrpPrmsGpu->sinVal_actUe, sinValGpuforCpuSchd, cellGrpPrmsCpu->nPrbGrp,
            cellGrpPrmsCpu->nCell, cellGrpPrmsCpu->nActiveUe, cellGrpPrmsCpu->numUeSchdPerCellTTI, 
            cellGrpPrmsCpu->nUeAnt, cellGrpPrmsCpu->nBsAnt);
    }
    
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->cellAssoc, cellAssocGpuforCpuSchd, assocSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->estH_fr, estH_fr_GPUforCpuSchd, hSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusCpu->avgRates, avgRatesGpuforCpuSchd, arSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusCpu->tbErrLast, tbErrLastGpuforCpuSchd, tbeSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->prdMat, prdMatGpuforCpuSchd, prdSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->sinVal, sinValGpuforCpuSchd, sinValSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->detMat, detMatGpuforCpuSchd, detSize, cudaMemcpyDeviceToHost));
    /*
    std::ofstream file;
    file.open("channels.txt", std::fstream::out);
    file << "channel_real = [";
    for (int idx = 0; idx < nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst; idx++) {
            if (idx<(nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst-1)) {
                file << cellGrpPrmsCpu->estH_fr[idx].x << " ";
            } else {
                file << cellGrpPrmsCpu->estH_fr[idx].x << "];\n\n";
            }
    }

    file.close();
    */
}

void network::ueDownSelectGpu()
{ 
    if (mimoMode == 0) { // 4TR
        uint16_t numThrdPerBlk = nBsAnt*nBsAnt*numUeSchdPerCellTTI;
        numThrdPerBlk = numThrdPerBlk > 1024 ? 1024 : numThrdPerBlk;
        dim3 gridDim = {static_cast<uint16_t>(nPrbGrp*nCell), 1, 1};
        dim3 blockDim = {numThrdPerBlk, 1, 1};

        if (DL == 1) { // DL
            ueDownSel4TrKernel<<<gridDim, blockDim>>>(schdSolGpu->setSchdUePerCellTTI, cellGrpPrmsGpu->cellId, 
                cellGrpPrmsGpu->cellAssoc, cellGrpPrmsGpu->estH_fr_actUe, cellGrpPrmsGpu->estH_fr, cellGrpPrmsGpu->estH_fr_half,
                cellGrpUeStatusGpu->avgRatesActUe, cellGrpUeStatusGpu->avgRates, cellGrpUeStatusGpu->tbErrLastActUe, cellGrpUeStatusGpu->tbErrLast, 
                cellGrpPrmsGpu->prdMat_actUe, cellGrpPrmsGpu->prdMat, cellGrpPrmsGpu->detMat_actUe, cellGrpPrmsGpu->detMat,
                cellGrpPrmsGpu->sinVal_actUe, cellGrpPrmsGpu->sinVal, cellGrpPrmsGpu->nPrbGrp,
                cellGrpPrmsGpu->nCell, cellGrpPrmsGpu->nActiveUe, cellGrpPrmsGpu->numUeSchdPerCellTTI, 
                cellGrpPrmsGpu->nBsAnt, cellGrpPrmsGpu->nUeAnt);
        } else { // UL
            ueDownSel4TrKernel<<<gridDim, blockDim>>>(schdSolGpu->setSchdUePerCellTTI, cellGrpPrmsGpu->cellId, 
                cellGrpPrmsGpu->cellAssoc, cellGrpPrmsGpu->estH_fr_actUe, cellGrpPrmsGpu->estH_fr, cellGrpPrmsGpu->estH_fr_half,
                cellGrpUeStatusGpu->avgRatesActUe, cellGrpUeStatusGpu->avgRates, cellGrpUeStatusGpu->tbErrLastActUe, cellGrpUeStatusGpu->tbErrLast, 
                cellGrpPrmsGpu->prdMat_actUe, cellGrpPrmsGpu->prdMat, cellGrpPrmsGpu->detMat_actUe, cellGrpPrmsGpu->detMat,
                cellGrpPrmsGpu->sinVal_actUe, cellGrpPrmsGpu->sinVal, cellGrpPrmsGpu->nPrbGrp,
                cellGrpPrmsGpu->nCell, cellGrpPrmsGpu->nActiveUe, cellGrpPrmsGpu->numUeSchdPerCellTTI, 
                cellGrpPrmsGpu->nUeAnt, cellGrpPrmsGpu->nBsAnt);
        }        
    
        CUDA_CHECK_ERR(cudaMemcpy(estH_fr.get(), cellGrpPrmsGpu->estH_fr, hSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERR(cudaMemcpy(avgRates.get(), cellGrpUeStatusGpu->avgRates, arSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERR(cudaMemcpy(tbErrLast.get(), cellGrpUeStatusGpu->tbErrLast, tbeSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERR(cudaMemcpy(cellAssoc.get(), cellGrpPrmsGpu->cellAssoc, assocSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERR(cudaMemcpy(prdMat.get(), cellGrpPrmsGpu->prdMat, prdSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERR(cudaMemcpy(sinVal.get(), cellGrpPrmsGpu->sinVal, sinValSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERR(cudaMemcpy(detMat.get(), cellGrpPrmsGpu->detMat, detSize, cudaMemcpyDeviceToHost));
    } else if (mimoMode == 1) { // 64 TR
        dim3 gridDim = {static_cast<uint16_t>(nPrbGrp*nCell), 1, 1};
        dim3 blockDim = {static_cast<uint16_t>(nBsAnt*nUeAnt), 1, 1};

        if (DL == 1) { // DL
            ueDownSel64TrKernel<<<gridDim, blockDim>>>(schdSolGpu->sortedUeList,
                                cellGrpPrmsGpu->estH_fr_actUe, 
                                cellGrpPrmsGpu->srsEstChan,
                                cellGrpPrmsGpu->srsUeMap,
                                cellGrpPrmsGpu->nPrbGrp, 
                                cellGrpPrmsGpu->nCell, 
                                cellGrpPrmsGpu->nActiveUe,
                                cellGrpPrmsGpu->numUeForGrpPerCell,
                                cellGrpPrmsGpu->nBsAnt, 
                                cellGrpPrmsGpu->nUeAnt);

        } else { // UL
            ueDownSel64TrKernel<<<gridDim, blockDim>>>(schdSolGpu->sortedUeList,
                                cellGrpPrmsGpu->estH_fr_actUe, 
                                cellGrpPrmsGpu->srsEstChan,
                                cellGrpPrmsGpu->srsUeMap,
                                cellGrpPrmsGpu->nPrbGrp, 
                                cellGrpPrmsGpu->nCell, 
                                cellGrpPrmsGpu->nActiveUe,
                                cellGrpPrmsGpu->numUeForGrpPerCell,
                                cellGrpPrmsGpu->nUeAnt,
                                cellGrpPrmsGpu->nBsAnt);
        }
    }
}

void network::rrUeSelectionCpu(const int TTIidx)
{
    // assume that the cell assocation of all active UEs are fixed and that numActiveUesPerCell is no less than numUeSchdPerCellTTI
    // round-robin UE selection
    if (TTIidx == 0) { // first TTI
        for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
            for (int uIdx = 0; uIdx < numUeSchdPerCellTTI; uIdx++) {
                schdSolCpu->setSchdUePerCellTTI[cIdx*numUeSchdPerCellTTI + uIdx] = cIdx*netData->numActiveUesPerCell + uIdx; // global UE ID
                cellGrpUeStatusCpu->avgRates[cIdx*numUeSchdPerCellTTI + uIdx] = netData->avgRatesCPU[schdSolCpu->setSchdUePerCellTTI[cIdx*numUeSchdPerCellTTI + uIdx]];
            }
        }
    } else {
        for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
            uint16_t lastSchdUeId = schdSolCpu->setSchdUePerCellTTI[cIdx*numUeSchdPerCellTTI + numUeSchdPerCellTTI-1];
            for (int uIdx = 0; uIdx < numUeSchdPerCellTTI; uIdx++) {
                uint16_t schdUeId = lastSchdUeId+uIdx+1;
                if (schdUeId >= (cIdx+1)*netData->numActiveUesPerCell) {
                    schdUeId -= netData->numActiveUesPerCell;
                }

                schdSolCpu->setSchdUePerCellTTI[cIdx*numUeSchdPerCellTTI + uIdx] = schdUeId;
                cellGrpUeStatusCpu->avgRates[cIdx* + uIdx] = netData->avgRatesCPU[schdUeId];
            }
        }
    }
}

void network::genFastFadingGpu(const int TTIidx)
{
    if(m_fastFadingType == 1 || m_fastFadingType == 2) // GPU TDL
    {
        // optional: reset initial phase
        // m_tdl_chan -> reset();
        m_tdl_chan->run(TTIidx * slotDuration, !DL /*enableSwapTxRx*/);  // channel is always generated in DL, but processing tx signal depends on enableSwapTxRx
        // optional: print sample GPU TDL freq channel on Prbg
        // m_tdl_chan->printFreqPrbgChan();
    }
    else if (m_fastFadingType == 3 || m_fastFadingType == 4)  // GPU CDL
    {
        // optional: reset initial phase and ray coupling
        // m_cdl_chan -> reset();
        m_cdl_chan->run(TTIidx * slotDuration, !DL /*enableSwapTxRx*/);  // channel is always generated in DL, but processing tx signal depends on enableSwapTxRx
        // optional: print sample GPU CDL freq channel on Prbg
        // m_cdl_chan->printFreqPrbgChan();
    }

    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->estH_fr_actUe, estH_fr_actUe.get(), hActUeSize, cudaMemcpyHostToDevice));
    if (DL == 1) { // DL
        CUDA_CHECK_ERR(cudaMemcpy(netData->rxSigPowDBGpu, netData->rxSigPowDB.get(), netData->numCell*netData->numCell*netData->numActiveUesPerCell*sizeof(float), cudaMemcpyHostToDevice));
    } else { // UL
        CUDA_CHECK_ERR(cudaMemcpy(netData->rxSigPowDBGpu_UL, netData->rxSigPowDB_UL.get(), netData->numCell*netData->numCell*netData->numActiveUesPerCell*sizeof(float), cudaMemcpyHostToDevice));
    }
    
    if (DL == 1) { // DL
        genChann4TrKernel<<<netData->numThrdBlk, netData->numThrdPerBlk>>>(cellGrpPrmsGpu->estH_fr_actUe, cellGrpPrmsGpu->estH_fr_actUe_prd, netData->rxSigPowDBGpu, nPrbGrp, netData->numCell, 
            netData->numActiveUesPerCell, nBsAnt, nUeAnt, netData->rho, netData->rhoPrime, scalingFactor, 
            TTIidx, netData->states, 1 /*DL = 1*/, m_externFastFadingPrbgPtr);
    } else { // UL
        genChann4TrKernel<<<netData->numThrdBlk, netData->numThrdPerBlk>>>(cellGrpPrmsGpu->estH_fr_actUe, cellGrpPrmsGpu->estH_fr_actUe_prd, netData->rxSigPowDBGpu_UL, nPrbGrp, netData->numCell, 
            netData->numActiveUesPerCell, nBsAnt, nUeAnt, netData->rho, netData->rhoPrime, scalingFactor, 
            TTIidx, netData->states, 0 /*DL = 0*/, m_externFastFadingPrbgPtr);
    }
    
    CUDA_CHECK_ERR(cudaMemcpy(estH_fr_actUe.get(), cellGrpPrmsGpu->estH_fr_actUe, hActUeSize, cudaMemcpyDeviceToHost));                                    
}

void network::genSrsChanEstGpu()
{
    dim3 gridDim = {1, 1, 1};
    dim3 blockDim = {1024, 1, 1};

    genSrsChanEstGpuKernel<<<gridDim, blockDim>>>(cellGrpPrmsGpu->wbSinr, cellGrpPrmsGpu->srsWbSnr, cellGrpPrmsGpu->nActiveUe, cellGrpPrmsGpu->nUeAnt);
}

__global__ void genSrsChanEstGpuKernel(float*        wbSinr,
                                       float*        srsWbSnr,
                                       const int     nActiveUe,
                                       const int     numUeAnt)
{
    for (int uIdx = threadIdx.x; uIdx < nActiveUe; uIdx += blockDim.x) {
        srsWbSnr[uIdx] = static_cast<float>(uIdx % 15);
    }
}

__global__ void init_curand(unsigned int t_seed, int id_offset, curandState *state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(t_seed, idx + id_offset, 0, &state[idx]);
}

__global__ void genChann4TrKernel(cuComplex*       channMatGpu, 
                                  cuComplex*       channMatPrdGpu,
                                  float*           rxSigPowDBGpu,
                                  const int        nPrbGrp, 
                                  const int        numCell, 
                                  const int        numActUePerCell, 
                                  const int        numBsAnt,
                                  const int        numUeAnt,
                                  const float      rho,
                                  const float      rhoPrime,
                                  const float      scalingFactor,
                                  const int        TTIidx,
                                  curandState_t*   states,
                                  const uint8_t    DL,
                                  cuComplex*       freqChanPrgPtr)
{
    int globalThdIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int prgIdx = floor(static_cast<float>(blockIdx.x)/static_cast<float>(numCell));
    int cIdx = blockIdx.x - prgIdx*numCell; 
    int totNumActUe = numCell*numActUePerCell;
    int nBsUeAntPrd = numBsAnt*numUeAnt;
    int uIdx = floor(static_cast<float>(threadIdx.x)/static_cast<float>(nBsUeAntPrd));
    int channIdx = prgIdx*numCell*totNumActUe*nBsUeAntPrd;
    channIdx += cIdx*nBsUeAntPrd;
    int eIdx = threadIdx.x - uIdx*nBsUeAntPrd;
    channIdx += eIdx;
    // eIdx is bsAntIdx * numUeAnt + ueAntIdx, i.e., bsAntIdx = 0, [ueAntIdx = 0, 1, ..., numUeAnt-1]; bsAntIdx = 1, [numUeAnt = 0, 1, ..., numUeAnt-1] ...
    // tdlCdlTxUeAntOffset is ueAntIdx * nBsAnt + bsAntIdx, i.e., ueAntIdx = 0, [bsAntIdx = 0, 1, ..., nBsAnt-1]; ueAntIdx = 1, [bsAntIdx = 0, 1, ..., nBsAnt-1] ...
    // For UL/DL: ueAntIdx = eIdx % numUeAnt, nBsAnt = numBsAnt, bsAntIdx = eIdx / numUeAnt
    int tdlCdlTxUeAntOffset = ((eIdx % numUeAnt) * numBsAnt + (eIdx / numUeAnt)); 

    int numUePerRnd = floor(static_cast<float>(blockDim.x)/static_cast<float>(nBsUeAntPrd));
    int numRound = ceil(static_cast<float>(totNumActUe)/static_cast<float>(numUePerRnd));
    
    float channMatGpu_real;
    float channMatGpu_imag;
    for (int rndIdx = 0; rndIdx < numRound; rndIdx++) {
        int currUeIdx = uIdx + rndIdx*numUePerRnd; 
        if (currUeIdx < totNumActUe) {
            int currChannIdx = channIdx + currUeIdx*numCell*nBsUeAntPrd;
            float amplDBm = rxSigPowDBGpu[cIdx*totNumActUe + currUeIdx];
            float sqrtAmpl = pow(10.0, ((amplDBm - 30.0)/20.0));
            sqrtAmpl *= scalingFactor;

            if(freqChanPrgPtr == nullptr) // use Rayleigh fading
            {
                if (TTIidx > 0) {
                    channMatGpu_real = rho*channMatGpu[currChannIdx].x + rhoPrime*sqrtAmpl*curand_normal(&states[globalThdIdx]);
                    channMatGpu_imag = rho*channMatGpu[currChannIdx].y + rhoPrime*sqrtAmpl*curand_normal(&states[globalThdIdx]);
                } else {
                    channMatGpu_real = sqrtAmpl*curand_normal(&states[globalThdIdx]);
                    channMatGpu_imag = sqrtAmpl*curand_normal(&states[globalThdIdx]);
                }
            }
            else
            {
                // get global tdl chan idx to read tdl channel coe on Prg
                uint32_t globalTdlChanIdx = ((cIdx * totNumActUe + currUeIdx) * nBsUeAntPrd + tdlCdlTxUeAntOffset) * nPrbGrp + prgIdx; // GPU TDL channel 1D dims: [nCell, nActiveUe, nUeAnt, nBsAnt, nPrg]; cuMAC channel dims: [nPrbGrp, nActiveUe, nCell, numBsAnt, numUeAnt]
                channMatGpu_real = sqrtAmpl * freqChanPrgPtr[globalTdlChanIdx].x;
                channMatGpu_imag = sqrtAmpl * freqChanPrgPtr[globalTdlChanIdx].y;
            }
            channMatGpu[currChannIdx].x = channMatGpu_real;
            channMatGpu[currChannIdx].y = channMatGpu_imag;

            int asscCellIdx = floor(static_cast<float>(currUeIdx)/static_cast<float>(numActUePerCell));
            if (asscCellIdx == cIdx) {
                channMatPrdGpu[currUeIdx*nPrbGrp*nBsUeAntPrd + prgIdx*nBsUeAntPrd + eIdx].x = channMatGpu_real;
                channMatPrdGpu[currUeIdx*nPrbGrp*nBsUeAntPrd + prgIdx*nBsUeAntPrd + eIdx].y = channMatGpu_imag;
            }
        }
    }
}

void network::genFastFading()
{
    float stddev = 0.5*sqrt(2);
    std::normal_distribution<double> distribution(0.0, stddev);

    for (int prgIdx = 0; prgIdx < nPrbGrp; prgIdx++) { // loop through PRGs
        for (int uIdx = 0; uIdx < netData->numCell*numUeSchdPerCellTTI; uIdx++) {
            for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
                int assocCellId = floor(static_cast<float>(uIdx)/static_cast<float>(numUeSchdPerCellTTI));

                int globalUeId = setSchdUePerCellTTI[assocCellId*numUeSchdPerCellTTI + uIdx%numUeSchdPerCellTTI];
                
                float amplDBm = netData->rxSigPowDB[cIdx*netData->numCell*netData->numActiveUesPerCell + globalUeId];
                float sqrtAmpl = pow(10.0, ((amplDBm - 30.0)/20.0));
                for (int txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) {
                    for (int rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) {
                        int index = prgIdx*netData->numCell*numUeSchdPerCellTTI*netData->numCell*nBsAnt*nUeAnt;
                        index += uIdx*netData->numCell*nBsAnt*nUeAnt;
                        index += cIdx*nBsAnt*nUeAnt;
                        index += txAntIdx*nUeAnt;
                        index += rxAntIdx;

                        estH_fr[index].x = sqrtAmpl*distribution(randomEngine);
                        estH_fr[index].y = sqrtAmpl*distribution(randomEngine);
                    }
                }
            }
        }
    }
}

void network::cpySinrGpu2Cpu()
{
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->postEqSinr, cellGrpPrmsGpu->postEqSinr, postEqSinrSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsCpu->wbSinr, cellGrpPrmsGpu->wbSinr, wbSinrSize, cudaMemcpyDeviceToHost));
}

void network::updateDataRateAllActiveUeGpu(const int slotIdx)
{
    CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTI.get(), schdSolGpu->setSchdUePerCellTTI, setSchdUeSolSize, cudaMemcpyDeviceToHost)); 

    for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
        bool ueFound = false;
        for (int schdUidx = 0; schdUidx < nUe; schdUidx++) {
            if (uIdx == setSchdUePerCellTTI[schdUidx]) {
                avgRatesActUe[uIdx] = avgRates[schdUidx];
                tbErrLastActUe[uIdx] = tbErrLast[schdUidx];
                ueFound = true;
                break;
            }
        }

        if (!ueFound)
            avgRatesActUe[uIdx] = (1.0-pfAvgRateUpd)*avgRatesActUe[uIdx];
    }

    sumCellThrRecordsGpu[slotIdx] = 0;
    sumCellPfRecordsGpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
        sumCellThrRecordsGpu[slotIdx] += avgRatesActUe[uIdx];
        sumCellPfRecordsGpu[slotIdx] += log2f(avgRatesActUe[uIdx]);
        //printf("%f ", avgRatesActUe[uIdx]);
    }
    //printf("\n");

    printf("GPU scheduler sum cell throughput: %4.3e\n", sumCellThrRecordsGpu[slotIdx]);
}

void network::updateDataRateAllActiveUeCpu(const int slotIdx)
{
    for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
        bool ueFound = false;
        for (int schdUidx = 0; schdUidx < nUe; schdUidx++) {
            if (uIdx == schdSolCpu->setSchdUePerCellTTI[schdUidx]) {
                cellGrpUeStatusCpu->avgRatesActUe[uIdx] = cellGrpUeStatusCpu->avgRates[schdUidx];
                cellGrpUeStatusCpu->tbErrLastActUe[uIdx] = cellGrpUeStatusCpu->tbErrLast[schdUidx];
                ueFound = true;
                break;
            }
        }

        if (!ueFound)
            cellGrpUeStatusCpu->avgRatesActUe[uIdx] = (1.0-pfAvgRateUpd)*cellGrpUeStatusCpu->avgRatesActUe[uIdx];
    }

    sumCellThrRecordsCpu[slotIdx] = 0;
    sumCellPfRecordsCpu[slotIdx] = 0;
    for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
        sumCellThrRecordsCpu[slotIdx] += cellGrpUeStatusCpu->avgRatesActUe[uIdx];
        sumCellPfRecordsCpu[slotIdx] += log2f(cellGrpUeStatusCpu->avgRatesActUe[uIdx]);
    }

    printf("CPU scheduler sum cell throughput: %4.3e\n", sumCellThrRecordsCpu[slotIdx]);
}

void network::testChannGen()
{
    float stddev = 0.5*sqrt(2);
    std::normal_distribution<double> distribution(0.0, stddev);

    for (int prgIdx = 0; prgIdx < nPrbGrp; prgIdx++) { // loop through PRGs
        for (int uIdx = 0; uIdx < netData->numCell*netData->numActiveUesPerCell; uIdx++) {
            for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {                
                float amplDBm = netData->rxSigPowDB[cIdx*netData->numCell*netData->numActiveUesPerCell + uIdx];
                float sqrtAmpl = pow(10.0, ((amplDBm - 30.0)/20.0));
                for (int txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) {
                    for (int rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) {
                        int index = prgIdx*netData->numCell*netData->numActiveUesPerCell*netData->numCell*nBsAnt*nUeAnt;
                        index += uIdx*netData->numCell*nBsAnt*nUeAnt;
                        index += cIdx*nBsAnt*nUeAnt;
                        index += txAntIdx*nUeAnt;
                        index += rxAntIdx;
                        estH_fr_actUe[index].x = sqrtAmpl*distribution(randomEngine);
                        estH_fr_actUe[index].y = sqrtAmpl*distribution(randomEngine);
                    }
                }
            }
        }
    }
}

void network::writetoFileLargeNumActUe_short()
{
    std::ofstream file;
    file.open(mcOutputFileShort, std::fstream::out);

    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellThrRecordsCpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellThrRecordsCpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellThrRecordsCpu[slotIdx] << " ";
    }
    file << sumCellThrRecordsCpu[numSimChnRlz-1] << "];\n";

    if (gpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellThrRecordsGpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellThrRecordsGpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellThrRecordsGpu[slotIdx] << " ";
    }
    file << sumCellThrRecordsGpu[numSimChnRlz-1] << "];\n";

    ////////////////////
    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellPfRecordsCpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellPfRecordsCpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellPfRecordsCpu[slotIdx] << " ";
    }
    file << sumCellPfRecordsCpu[numSimChnRlz-1] << "];\n";

    if (gpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellPfRecordsGpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellPfRecordsGpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellPfRecordsGpu[slotIdx] << " ";
    }
    file << sumCellPfRecordsGpu[numSimChnRlz-1] << "];\n";
    ////////////////////

    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumInsThrRecordsCpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumInsThrRecordsCpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumInsThrRecordsCpu[slotIdx] << " ";
    }
    file << sumInsThrRecordsCpu[numSimChnRlz-1] << "];\n";

    if (gpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumInsThrRecordsGpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumInsThrRecordsGpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumInsThrRecordsGpu[slotIdx] << " ";
    }
    file << sumInsThrRecordsGpu[numSimChnRlz-1] << "];\n";
    ////////////////////

    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1AvgRatesCpu = [";
    } else {
        file << "mcType0AvgRatesCpu = [";
    }
    for (int uIdx = 0; uIdx < nActiveUe-1; uIdx++) {
        file << cellGrpUeStatusCpu->avgRatesActUe[uIdx] << " ";
    }
    file << cellGrpUeStatusCpu->avgRatesActUe[nActiveUe-1] << "];\n";

    if (gpuAllocType) { // type-1
        file << "mcType1AvgRatesGpu = [";
    } else {
        file << "mcType0AvgRatesGpu = [";
    }
    for (int uIdx = 0; uIdx < nActiveUe-1; uIdx++) {
        file << avgRatesActUe[uIdx] << " ";
    }
    file << avgRatesActUe[nActiveUe-1] << "];\n";

    file.close();
}

void network::writeToFileLargeNumActUe()
{
    std::ofstream file;
    file.open(mcOutputFile, std::fstream::out);

    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellThrRecordsCpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellThrRecordsCpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellThrRecordsCpu[slotIdx] << " ";
    }
    file << sumCellThrRecordsCpu[numSimChnRlz-1] << "];\n";

    if (gpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellThrRecordsGpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellThrRecordsGpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellThrRecordsGpu[slotIdx] << " ";
    }
    file << sumCellThrRecordsGpu[numSimChnRlz-1] << "];\n";

    ////////////////////
    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellPfRecordsCpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellPfRecordsCpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellPfRecordsCpu[slotIdx] << " ";
    }
    file << sumCellPfRecordsCpu[numSimChnRlz-1] << "];\n";

    if (gpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumCellPfRecordsGpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumCellPfRecordsGpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumCellPfRecordsGpu[slotIdx] << " ";
    }
    file << sumCellPfRecordsGpu[numSimChnRlz-1] << "];\n";
    ////////////////////

    if (cpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumInsThrRecordsCpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumInsThrRecordsCpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumInsThrRecordsCpu[slotIdx] << " ";
    }
    file << sumInsThrRecordsCpu[numSimChnRlz-1] << "];\n";

    if (gpuAllocType) { // type-1 consecutive allocate
        file << "mcType1SumInsThrRecordsGpu = [";
    } else { // type-0 consecutive allocate
        file << "mcType0SumInsThrRecordsGpu = [";
    }
    
    for (int slotIdx = 0; slotIdx < numSimChnRlz-1; slotIdx++){
        file << sumInsThrRecordsGpu[slotIdx] << " ";
    }
    file << sumInsThrRecordsGpu[numSimChnRlz-1] << "];\n";
    ////////////////////
    file << "mcType1AvgRatesCpu = [";
    for (int uIdx = 0; uIdx < nActiveUe-1; uIdx++) {
        file << cellGrpUeStatusCpu->avgRatesActUe[uIdx] << " ";
    }
    file << cellGrpUeStatusCpu->avgRatesActUe[nActiveUe-1] << "];\n";

    file << "mcType1AvgRatesGpu = [";
    for (int uIdx = 0; uIdx < nActiveUe-1; uIdx++) {
        file << avgRatesActUe[uIdx] << " ";
    }
    file << avgRatesActUe[nActiveUe-1] << "];\n";


    file << "mcType1McsSelCpu"<<" = [";
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        for (int slotIdx = 0; slotIdx < numSimChnRlz; slotIdx++){
            file << mcsSelRecordsCpu[uIdx][slotIdx] << " ";
        }
    }
    file <<"];\n";

    file << "mcType1McsSelGpu"<<" = [";
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        for (int slotIdx = 0; slotIdx < numSimChnRlz; slotIdx++){
            file << mcsSelRecordsGpu[uIdx][slotIdx] << " ";
        }
    }
    file <<"];\n";
/*
    file << "mcType1LayerSelCpu"<<" = [";
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        for (int slotIdx = 0; slotIdx < numSimChnRlz; slotIdx++){
            if (static_cast<int>(layerSelRecordsCpu[uIdx][slotIdx]) < 0xFF)
                file << static_cast<int>(layerSelRecordsCpu[uIdx][slotIdx]) << " ";
        }
    }
    file <<"];\n";

    file << "mcType1LayerSelGpu"<<" = [";
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        for (int slotIdx = 0; slotIdx < numSimChnRlz; slotIdx++){
            if (static_cast<int>(layerSelRecordsGpu[uIdx][slotIdx]) < 0xFF)
                file << static_cast<int>(layerSelRecordsGpu[uIdx][slotIdx]) << " ";
        }
    }
    file <<"];\n";
*/
    file.close();
}
}