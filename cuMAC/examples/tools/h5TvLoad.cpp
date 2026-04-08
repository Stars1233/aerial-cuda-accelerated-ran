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

#include "parameters.h"
#include "api.h"
#include "cumac.h"
#include "h5TvLoad.h"

using namespace H5;

void loadFromH5(const std::string&                 filename,
                cumac::cumacCellGrpUeStatus*       cellGrpUeStatusGpu,
                cumac::cumacCellGrpPrms*           cellGrpPrmsGpu,
                cumac::cumacSchdSol*               schdSolGpu) 
{
    cumac::cumacSchedulerParam data;

    // open TV H5 file
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // Open the dataset
    H5::DataSet dataset = file.openDataSet("cumacSchedulerParam");

    // Get the compound data type
    H5::CompType compoundType = dataset.getCompType();

    // Read the data from the dataset
    dataset.read(&data, compoundType);

    uint8_t DL = data.dlSchInd;

    // buffer sizes
    uint32_t prgMskSize = sizeof(uint8_t*)*data.nCell;
    uint32_t perCellPrgMskSize = sizeof(uint8_t)*data.nPrbGrp;
    uint32_t tbeSize;
    uint32_t tbeActUeSize = sizeof(int8_t)*data.nActiveUe;
    uint32_t pwActSize = sizeof(uint16_t)*data.nActiveUe;
    uint32_t ndActSize = sizeof(int8_t)*data.nActiveUe;
    uint32_t allocLTSize;
    uint32_t mcsSelLTSize; 
    uint32_t layerSelLTSize; 
    uint32_t gpuAllocSolSize;
    uint32_t mcsSelSolSize;
    uint32_t layerSize;
    uint32_t ueMapSize = sizeof(int32_t)*data.nUe; 
    uint32_t hfrSize = sizeof(cuComplex*)*data.nCell;
    uint32_t perCellHfrLen;
    uint32_t sinValLen;
    uint32_t setSchdUeSize;
    if (data.nBsAnt == 4) { // 4TR
        perCellHfrLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nUeAnt;
        sinValLen = data.nUe*data.nPrbGrp*data.nUeAnt;
        allocLTSize = sizeof(int16_t)*2*data.nUe; 
        mcsSelLTSize = sizeof(int16_t)*data.nUe;
        layerSelLTSize = sizeof(uint8_t)*data.nUe;
        if (data.allocType == 1) {
            gpuAllocSolSize = sizeof(int16_t)*2*data.nUe;
        } else {
            gpuAllocSolSize = sizeof(int16_t)*data.nCell*data.nPrbGrp;
        }    
        mcsSelSolSize = sizeof(int16_t)*data.nUe;
        layerSize = sizeof(uint8_t)*data.nUe;
        setSchdUeSize = sizeof(uint16_t)*data.nCell*data.numUeSchdPerCellTTI;
        tbeSize = sizeof(int8_t)*data.nUe; 
    } else if (data.nBsAnt == 64) { // 64TR
        perCellHfrLen = data.nCell*data.numUeForGrpPerCell*data.nPrbGrp*data.nUeAnt*data.nBsAnt;
        sinValLen = data.nActiveUe*data.nPrbGrp*data.nUeAnt;
        allocLTSize = sizeof(int16_t)*2*data.nActiveUe;
        mcsSelLTSize = sizeof(int16_t)*data.nActiveUe;
        layerSelLTSize = sizeof(uint8_t)*data.nActiveUe;
        gpuAllocSolSize = sizeof(int16_t)*2*data.nActiveUe;
        mcsSelSolSize = sizeof(int16_t)*data.nActiveUe;
        layerSize = sizeof(uint8_t)*data.nActiveUe;
        setSchdUeSize = sizeof(uint16_t)*data.nCell*data.numUeForGrpPerCell;
        tbeSize = sizeof(int8_t)*data.nActiveUe;
    }
    uint32_t perCellHfrSize = sizeof(cuComplex)*perCellHfrLen;
    uint32_t perCellHfrRealImagSize = sizeof(float)*perCellHfrLen;
    uint32_t prdLen;
    uint32_t detLen;
    if (DL == 1) { // DL
        prdLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nBsAnt;
        detLen = data.nUe*data.nPrbGrp*data.nUeAnt*data.nUeAnt;
    } else {
        prdLen = data.nUe*data.nPrbGrp*data.nUeAnt*data.nUeAnt;
        detLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nBsAnt; 
    }
    uint32_t prdSize = sizeof(cuComplex)*prdLen;
    uint32_t prdRealImagSize = sizeof(float)*prdLen;
    uint32_t detSize = sizeof(cuComplex)*detLen;
    uint32_t detRealImagSize = sizeof(float)*detLen;
    uint32_t sinValSize = sizeof(float)*sinValLen;
    uint32_t assocActUeSize = sizeof(uint8_t)*data.nCell*data.nActiveUe;
    uint32_t arActUeSize = sizeof(float)*data.nActiveUe;
    uint32_t sinrSize = sizeof(float)*data.nActiveUe*data.nPrbGrp*data.nUeAnt;
    uint32_t wbSinrSize = sizeof(float)*data.nActiveUe*data.nUeAnt;
    uint32_t cidSize = sizeof(uint16_t)*data.nCell;
    uint32_t numUeSchdArrSize = sizeof(uint8_t)*data.nCell;
    uint32_t assocSize = sizeof(uint8_t)*data.nCell*data.nUe; 
    uint32_t arSize = sizeof(float)*data.nUe;
    uint32_t hLen = data.nPrbGrp*data.nUe*data.nCell*data.nBsAnt*data.nUeAnt;
    uint32_t hSize = sizeof(cuComplex)*hLen;
    uint32_t hHalfSize = sizeof(__nv_bfloat162)*hLen;

    // pre-allocate CPU data structures and buffers
    // buffers
    uint16_t*                    cellId; // IDs of coordinated cells
    uint8_t*                     numUeSchdPerCellTTIArr;
    uint8_t*                     cellAssoc;
    uint8_t*                     cellAssocActUe;
    float*                       avgRates;
    float*                       avgRatesActUe;
    int8_t*                      tbErrLast;
    int8_t*                      tbErrLastActUe;
    uint16_t*                    prioWeightActUe;
    float*                       postEqSinr;
    float*                       wbSinr;
    int16_t*                     allocSol; // -1 indicates unallocated
    uint16_t*                    setSchdUePerCellTTI;
    int16_t*                     mcsSelSol; 
    uint8_t*                     layerSelSol; 

    // HARQ related buffers
    int8_t*                      newDataActUe; 
    int16_t*                     allocSolLastTx; 
    int16_t*                     mcsSelSolLastTx; 
    uint8_t*                     layerSelSolLastTx; 
    uint8_t**                    prgMsk;
    uint8_t*                     perCellPrgMsk;

    // buffers for channels and precoders
    std::vector<cuComplex>          prdMat64(data.nCell*data.nPrbGrp*data.nBsAnt*cumac::maxNumLayerPerGrpDL_);
    std::vector<float>              prdMat64_real(data.nCell*data.nPrbGrp*data.nBsAnt*cumac::maxNumLayerPerGrpDL_);
    std::vector<float>              prdMat64_imag(data.nCell*data.nPrbGrp*data.nBsAnt*cumac::maxNumLayerPerGrpDL_);
    std::vector<uint16_t>           ueOrderInGrp(data.nActiveUe);
    std::vector<uint32_t>           lastSchdSlotActUe(data.nActiveUe);  
    std::vector<uint32_t>           currSlotIdxPerCell(data.nCell); 
    std::vector<float>              beamformGainCurrTx(data.nActiveUe);
    std::vector<float>              bfGainPrgCurrTx(data.nActiveUe*data.nPrbGrp);
    std::vector<float>              beamformGainLastTx(data.nActiveUe);
    std::vector<uint16_t>           numUeInGrp(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
    std::vector<uint16_t>           ueId(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_);
    std::vector<int16_t>            subbandId(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
    auto muGrpListGpu = std::make_unique<cumac::multiCellMuGrpList>();  
    std::unique_ptr<cuComplex* []>  srsEstChan              = std::make_unique<cuComplex* []>(data.nCell);
    std::unique_ptr<cuComplex []>   srsEstChan_perCell      = std::make_unique<cuComplex []>(perCellHfrLen);
    std::unique_ptr<float []>       srsEstChan_perCell_real = std::make_unique<float []>(perCellHfrLen);
    std::unique_ptr<float []>       srsEstChan_perCell_imag = std::make_unique<float []>(perCellHfrLen);
    std::unique_ptr<float []>       srsWbSnr                = std::make_unique<float []>(data.nActiveUe);
    std::unique_ptr<int32_t* []>    srsUeMap                = std::make_unique<int32_t* []>(data.nCell);
    std::unique_ptr<int32_t []>     srsUeMap_perCell        = std::make_unique<int32_t []>(data.nActiveUe);
    std::unique_ptr<float []>       sinVal                  = std::make_unique<float []>(sinValLen);
    std::unique_ptr<uint8_t []>     muMimoInd               = std::make_unique<uint8_t []>(data.nActiveUe);
    std::unique_ptr<uint16_t* []>   sortedUeList            = std::make_unique<uint16_t* []>(data.nCell);
    std::unique_ptr<uint16_t []>    sortedUeList_perCell    = std::make_unique<uint16_t []>(data.nMaxActUePerCell);
    std::unique_ptr<uint8_t []>     nSCID                   = std::make_unique<uint8_t []>(data.nActiveUe);
    std::vector<float>              bsTxPow(data.nCell);
    std::vector<float>              ueTxPow(data.nActiveUe);
    std::vector<float>              noiseVarActUe(data.nActiveUe);
    std::vector<float>              blerTargetActUe(data.nActiveUe);

    float*                       estH_fr_real;
    float*                       estH_fr_imag;
    __nv_bfloat162*              estH_fr_half; 
    cuComplex*                   estH_fr;
    cuComplex*                   prdMat;
    float*                       prdMat_real;
    float*                       prdMat_imag;
    cuComplex*                   detMat;
    float*                       detMat_real;
    float*                       detMat_imag;
    cuComplex*                   prdMat_asim;
    float*                       prdMat_asim_real;
    float*                       prdMat_asim_imag;
    cuComplex*                   detMat_asim;
    float*                       detMat_asim_real;
    float*                       detMat_asim_imag;
    float*                       sinVal_asim;

    CUDA_CHECK_ERR(cudaMallocHost((void **)&estH_fr_real, hLen*sizeof(float)));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&estH_fr_imag, hLen*sizeof(float)));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&estH_fr_half, hHalfSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&estH_fr, hSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&cellId, cidSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&numUeSchdPerCellTTIArr, numUeSchdArrSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&cellAssoc, assocSize)); 
    CUDA_CHECK_ERR(cudaMallocHost((void **)&cellAssocActUe, assocActUeSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&avgRates, arSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&avgRatesActUe, arActUeSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&tbErrLast, tbeSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&tbErrLastActUe, tbeActUeSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&postEqSinr, sinrSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&wbSinr, wbSinrSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&allocSol, gpuAllocSolSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&setSchdUePerCellTTI, setSchdUeSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&mcsSelSol, mcsSelSolSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&layerSelSol, layerSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&newDataActUe, ndActSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prioWeightActUe, pwActSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&allocSolLastTx, allocLTSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&mcsSelSolLastTx, mcsSelLTSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&layerSelSolLastTx, layerSelLTSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&perCellPrgMsk, perCellPrgMskSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prgMsk, prgMskSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&sinVal_asim, sinValSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_asim, prdSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_asim_real, prdRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_asim_imag, prdRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_asim, detSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_asim_real, detRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_asim_imag, detRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat, prdSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_real, prdRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_imag, prdRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat, detSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_real, detRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_imag, detRealImagSize));

    // assign values for constant parameters
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
    cellGrpPrmsGpu->numUeForGrpPerCell = data.numUeForGrpPerCell;
    cellGrpPrmsGpu->chanCorrThr = data.chanCorrThr;
    cellGrpPrmsGpu->muCoeff = data.muCoeff;
    cellGrpPrmsGpu->srsSnrThr = data.srsSnrThr;
    cellGrpPrmsGpu->nMaxActUePerCell = data.nMaxActUePerCell;
    cellGrpPrmsGpu->zfCoeff = data.zfCoeff;
    cellGrpPrmsGpu->nMaxUePerGrpUl = data.nMaxUePerGrpUl;
    cellGrpPrmsGpu->nMaxUePerGrpDl = data.nMaxUePerGrpDl;
    cellGrpPrmsGpu->nMaxLayerPerGrpUl = data.nMaxLayerPerGrpUl;
    cellGrpPrmsGpu->nMaxLayerPerGrpDl = data.nMaxLayerPerGrpDl;
    cellGrpPrmsGpu->nMaxLayerPerUeSuUl = data.nMaxLayerPerUeSuUl;
    cellGrpPrmsGpu->nMaxLayerPerUeSuDl = data.nMaxLayerPerUeSuDl;
    cellGrpPrmsGpu->nMaxLayerPerUeMuUl = data.nMaxLayerPerUeMuUl;
    cellGrpPrmsGpu->nMaxLayerPerUeMuDl = data.nMaxLayerPerUeMuDl;    
    cellGrpPrmsGpu->nMaxUegPerCellDl = data.nMaxUegPerCellDl;
    cellGrpPrmsGpu->nMaxUegPerCellUl = data.nMaxUegPerCellUl;
    cellGrpPrmsGpu->mcsSelSinrCapThr = data.mcsSelSinrCapThr; 
    cellGrpPrmsGpu->muGrpSrsSnrMaxGap = data.muGrpSrsSnrMaxGap; 
    cellGrpPrmsGpu->muGrpSrsSnrSplitThr = data.muGrpSrsSnrSplitThr; 
    cellGrpPrmsGpu->bfPowAllocScheme = data.bfPowAllocScheme; 
    cellGrpPrmsGpu->muGrpUpdate = data.muGrpUpdate; 
    cellGrpPrmsGpu->mcsSelLutType = data.mcsSelLutType; 
    cellGrpPrmsGpu->semiStatFreqAlloc = data.semiStatFreqAlloc;   
    cellGrpPrmsGpu->harqEnabledInd = data.harqEnabledInd;
    cellGrpPrmsGpu->mcsSelCqi = data.mcsSelCqi; 
    cellGrpPrmsGpu->dlSchInd = data.dlSchInd;
    
    if (cellGrpPrmsGpu->prgMsk) {
        CUDA_CHECK_ERR(cudaMemcpy(prgMsk, cellGrpPrmsGpu->prgMsk, prgMskSize, cudaMemcpyDeviceToHost));
    }

    if (cellGrpPrmsGpu->srsEstChan) {
        CUDA_CHECK_ERR(cudaMemcpy(srsEstChan.get(), cellGrpPrmsGpu->srsEstChan, hfrSize, cudaMemcpyDeviceToHost));
    }

    if (cellGrpPrmsGpu->srsUeMap) {
        CUDA_CHECK_ERR(cudaMemcpy(srsUeMap.get(), cellGrpPrmsGpu->srsUeMap, data.nCell*sizeof(int32_t*), cudaMemcpyDeviceToHost));
    }

    if (schdSolGpu->sortedUeList) {
        CUDA_CHECK_ERR(cudaMemcpy(sortedUeList.get(), schdSolGpu->sortedUeList, data.nCell*sizeof(uint16_t*), cudaMemcpyDeviceToHost));
    }

    if (schdSolGpu->muGrpList) {
        CUDA_CHECK_ERR(cudaMemcpy(muGrpListGpu.get(), schdSolGpu->muGrpList, sizeof(cumac::multiCellMuGrpList), cudaMemcpyDeviceToHost));
    }   

    // load all existing datasets
    if (H5Lexists(file.getId(), "cellId", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("cellId");
        dataset.read(cellId, H5::PredType::NATIVE_UINT16);
        if (cellGrpPrmsGpu->cellId)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->cellId, cellId, cidSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "numUeSchdPerCellTTIArr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("numUeSchdPerCellTTIArr");
        dataset.read(numUeSchdPerCellTTIArr, H5::PredType::NATIVE_UINT8);
        if (cellGrpPrmsGpu->numUeSchdPerCellTTIArr)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->numUeSchdPerCellTTIArr, numUeSchdPerCellTTIArr, numUeSchdArrSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "cellAssoc", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("cellAssoc");
        dataset.read(cellAssoc, H5::PredType::NATIVE_UINT8);
        if (cellGrpPrmsGpu->cellAssoc)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->cellAssoc, cellAssoc, assocSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "cellAssocActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("cellAssocActUe");
        dataset.read(cellAssocActUe, H5::PredType::NATIVE_UINT8);
        if (cellGrpPrmsGpu->cellAssocActUe)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->cellAssocActUe, cellAssocActUe, assocActUeSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "blerTargetActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("blerTargetActUe");
        dataset.read(blerTargetActUe.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->blerTargetActUe) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->blerTargetActUe, blerTargetActUe.data(), data.nActiveUe*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    if (H5Lexists(file.getId(), "avgRates", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("avgRates");
        dataset.read(avgRates, H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->avgRates)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->avgRates, avgRates, arSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "avgRatesActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("avgRatesActUe");
        dataset.read(avgRatesActUe, H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->avgRatesActUe)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->avgRatesActUe, avgRatesActUe, arActUeSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "tbErrLast", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("tbErrLast");
        dataset.read(tbErrLast, H5::PredType::NATIVE_INT8);
        if (cellGrpUeStatusGpu->tbErrLast)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->tbErrLast, tbErrLast, tbeSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "tbErrLastActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("tbErrLastActUe");
        dataset.read(tbErrLastActUe, H5::PredType::NATIVE_INT8);
        if (cellGrpUeStatusGpu->tbErrLastActUe)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->tbErrLastActUe, tbErrLastActUe, tbeActUeSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "postEqSinr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("postEqSinr");
        dataset.read(postEqSinr, H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->postEqSinr)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->postEqSinr, postEqSinr, sinrSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "wbSinr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("wbSinr");
        dataset.read(wbSinr, H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->wbSinr)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->wbSinr, wbSinr, wbSinrSize, cudaMemcpyHostToDevice));
    }

    if (data.nBsAnt == 64 && H5Lexists(file.getId(), "prdMat_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prdMat_real");
        dataset.read(prdMat64_real.data(), H5::PredType::NATIVE_FLOAT);
    }

    if (data.nBsAnt == 64 && H5Lexists(file.getId(), "prdMat_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prdMat_imag");
        dataset.read(prdMat64_imag.data(), H5::PredType::NATIVE_FLOAT);
    }   

    if (H5Lexists(file.getId(), "numUeInGrp", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "ueId", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "subbandId", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("numUeInGrp");
        dataset.read(numUeInGrp.data(), H5::PredType::NATIVE_UINT16);
        if (muGrpListGpu->numUeInGrp) {
            CUDA_CHECK_ERR(cudaMemcpy(muGrpListGpu->numUeInGrp, numUeInGrp.data(), cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(uint16_t), cudaMemcpyHostToDevice));
        }

        dataset = file.openDataSet("ueId");
        dataset.read(ueId.data(), H5::PredType::NATIVE_UINT16);
        if (muGrpListGpu->ueId) {
            CUDA_CHECK_ERR(cudaMemcpy(muGrpListGpu->ueId, ueId.data(), cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_*sizeof(uint16_t), cudaMemcpyHostToDevice));
        }

        dataset = file.openDataSet("subbandId");
        dataset.read(subbandId.data(), H5::PredType::NATIVE_INT16);
        if (muGrpListGpu->subbandId) {
            CUDA_CHECK_ERR(cudaMemcpy(muGrpListGpu->subbandId, subbandId.data(), cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(int16_t), cudaMemcpyHostToDevice));
        }   
    }

    if (H5Lexists(file.getId(), "ueOrderInGrp", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("ueOrderInGrp");
        dataset.read(ueOrderInGrp.data(), H5::PredType::NATIVE_UINT16);
        if (schdSolGpu->ueOrderInGrp)
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->ueOrderInGrp, ueOrderInGrp.data(), sizeof(uint16_t)*data.nActiveUe, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "allocSol", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("allocSol");
        dataset.read(allocSol, H5::PredType::NATIVE_INT16);
        if (schdSolGpu->allocSol)
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->allocSol, allocSol, gpuAllocSolSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "muMimoInd", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("muMimoInd");
        dataset.read(muMimoInd.get(), H5::PredType::NATIVE_UINT8);
        if (schdSolGpu->muMimoInd) {
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->muMimoInd, muMimoInd.get(), data.nActiveUe*sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
    }

    if (H5Lexists(file.getId(), "nSCID", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("nSCID");
        dataset.read(nSCID.get(), H5::PredType::NATIVE_UINT8);
        if (schdSolGpu->nSCID) {
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->nSCID, nSCID.get(), data.nActiveUe*sizeof(uint8_t), cudaMemcpyHostToDevice));
        }
    }

    if (H5Lexists(file.getId(), "setSchdUePerCellTTI", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("setSchdUePerCellTTI");
        dataset.read(setSchdUePerCellTTI, H5::PredType::NATIVE_UINT16);
        if (schdSolGpu->setSchdUePerCellTTI)
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->setSchdUePerCellTTI, setSchdUePerCellTTI, setSchdUeSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "mcsSelSol", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("mcsSelSol");
        dataset.read(mcsSelSol, H5::PredType::NATIVE_INT16);
        if (schdSolGpu->mcsSelSol)
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->mcsSelSol, mcsSelSol, mcsSelSolSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "layerSelSol", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("layerSelSol");
        dataset.read(layerSelSol, H5::PredType::NATIVE_UINT8);
        if (schdSolGpu->layerSelSol)
            CUDA_CHECK_ERR(cudaMemcpy(schdSolGpu->layerSelSol, layerSelSol, layerSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "lastSchdSlotActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("lastSchdSlotActUe");
        dataset.read(lastSchdSlotActUe.data(), H5::PredType::NATIVE_UINT32);
        if (cellGrpUeStatusGpu->lastSchdSlotActUe) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->lastSchdSlotActUe, lastSchdSlotActUe.data(), sizeof(uint32_t)*data.nActiveUe, cudaMemcpyHostToDevice));    
        }
    }   

    if (H5Lexists(file.getId(), "currSlotIdxPerCell", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("currSlotIdxPerCell");
        dataset.read(currSlotIdxPerCell.data(), H5::PredType::NATIVE_UINT32);
        if (cellGrpPrmsGpu->currSlotIdxPerCell) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->currSlotIdxPerCell, currSlotIdxPerCell.data(), sizeof(uint32_t)*data.nCell, cudaMemcpyHostToDevice));    
        }
    }       

    if (H5Lexists(file.getId(), "beamformGainCurrTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("beamformGainCurrTx");
        dataset.read(beamformGainCurrTx.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->beamformGainCurrTx) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->beamformGainCurrTx, beamformGainCurrTx.data(), sizeof(float)*data.nActiveUe, cudaMemcpyHostToDevice));    
        }
    }   

    if (H5Lexists(file.getId(), "beamformGainLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("beamformGainLastTx");
        dataset.read(beamformGainLastTx.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->beamformGainLastTx) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->beamformGainLastTx, beamformGainLastTx.data(), sizeof(float)*data.nActiveUe, cudaMemcpyHostToDevice));    
        }
    }   

    if (H5Lexists(file.getId(), "bfGainPrgCurrTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("bfGainPrgCurrTx");
        dataset.read(bfGainPrgCurrTx.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->bfGainPrgCurrTx) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->bfGainPrgCurrTx, bfGainPrgCurrTx.data(), sizeof(float)*data.nActiveUe*data.nPrbGrp, cudaMemcpyHostToDevice));    
        }
    }   

    if (H5Lexists(file.getId(), "prioWeightActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prioWeightActUe");
        dataset.read(prioWeightActUe, H5::PredType::NATIVE_UINT16);
        if (cellGrpUeStatusGpu->prioWeightActUe)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->prioWeightActUe, prioWeightActUe, pwActSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "newDataActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("newDataActUe");
        dataset.read(newDataActUe, H5::PredType::NATIVE_INT8);
        if (cellGrpUeStatusGpu->newDataActUe)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->newDataActUe, newDataActUe, ndActSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "allocSolLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("allocSolLastTx");
        dataset.read(allocSolLastTx, H5::PredType::NATIVE_INT16);
        if (cellGrpUeStatusGpu->allocSolLastTx)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->allocSolLastTx, allocSolLastTx, allocLTSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "mcsSelSolLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("mcsSelSolLastTx");
        dataset.read(mcsSelSolLastTx, H5::PredType::NATIVE_INT16);
        if (cellGrpUeStatusGpu->mcsSelSolLastTx)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->mcsSelSolLastTx, mcsSelSolLastTx, mcsSelLTSize, cudaMemcpyHostToDevice));    
    }

    if (H5Lexists(file.getId(), "layerSelSolLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("layerSelSolLastTx");
        dataset.read(layerSelSolLastTx, H5::PredType::NATIVE_UINT8);
        if (cellGrpUeStatusGpu->layerSelSolLastTx)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->layerSelSolLastTx, layerSelSolLastTx, layerSelLTSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "srsWbSnr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("srsWbSnr");
        dataset.read(srsWbSnr.get(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->srsWbSnr) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->srsWbSnr, srsWbSnr.get(), data.nActiveUe*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    if (H5Lexists(file.getId(), "bsTxPow", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("bsTxPow");
        dataset.read(bsTxPow.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->bsTxPow) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->bsTxPow, bsTxPow.data(), data.nCell*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    if (H5Lexists(file.getId(), "ueTxPow", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("ueTxPow");
        dataset.read(ueTxPow.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->ueTxPow) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->ueTxPow, ueTxPow.data(), data.nActiveUe*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    if (H5Lexists(file.getId(), "noiseVarActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("noiseVarActUe");
        dataset.read(noiseVarActUe.data(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpUeStatusGpu->noiseVarActUe) {
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->noiseVarActUe, noiseVarActUe.data(), data.nActiveUe*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            std::string prgMskFieldName = "prgMsk" + std::to_string(cIdx);

            if (H5Lexists(file.getId(), prgMskFieldName.c_str(), H5P_DEFAULT) > 0) {
                dataset = file.openDataSet(prgMskFieldName);
                dataset.read(perCellPrgMsk, H5::PredType::NATIVE_UINT8);
                if (prgMsk[cIdx])
                    CUDA_CHECK_ERR(cudaMemcpy(prgMsk[cIdx], perCellPrgMsk, perCellPrgMskSize, cudaMemcpyHostToDevice));
            }

            std::string cfrRealFieldName = "estH_fr_real_cell" + std::to_string(cIdx);

            if (H5Lexists(file.getId(), cfrRealFieldName.c_str(), H5P_DEFAULT) > 0) {
                dataset = file.openDataSet(cfrRealFieldName);
                dataset.read(srsEstChan_perCell_real.get(), H5::PredType::NATIVE_FLOAT);
            }

            std::string cfrImagFieldName = "estH_fr_imag_cell" + std::to_string(cIdx);

            if (H5Lexists(file.getId(), cfrImagFieldName.c_str(), H5P_DEFAULT) > 0) {
                dataset = file.openDataSet(cfrImagFieldName);
                dataset.read(srsEstChan_perCell_imag.get(), H5::PredType::NATIVE_FLOAT);
            }

            if (H5Lexists(file.getId(), cfrRealFieldName.c_str(), H5P_DEFAULT) > 0 && H5Lexists(file.getId(), cfrImagFieldName.c_str(), H5P_DEFAULT) > 0) {
                for (int hIdx = 0; hIdx < perCellHfrLen; hIdx++) {
                    srsEstChan_perCell[hIdx].x = srsEstChan_perCell_real[hIdx];
                    srsEstChan_perCell[hIdx].y = srsEstChan_perCell_imag[hIdx];
                }
                if (srsEstChan[cIdx]) {
                    CUDA_CHECK_ERR(cudaMemcpy(srsEstChan[cIdx], srsEstChan_perCell.get(), perCellHfrSize, cudaMemcpyHostToDevice));
                }
            }

            std::string sortUeListFieldName = "sortedUeList_cell" + std::to_string(cIdx);
            if (H5Lexists(file.getId(), sortUeListFieldName.c_str(), H5P_DEFAULT) > 0) {
                dataset = file.openDataSet(sortUeListFieldName);
                dataset.read(sortedUeList_perCell.get(), H5::PredType::NATIVE_UINT16);
                if (sortedUeList[cIdx]) {
                    CUDA_CHECK_ERR(cudaMemcpy(sortedUeList[cIdx], sortedUeList_perCell.get(), data.nMaxActUePerCell*sizeof(uint16_t), cudaMemcpyHostToDevice));
                }
            }

            std::string srsUeMapFieldName = "srsUeMap_cell" + std::to_string(cIdx);
            if (H5Lexists(file.getId(), srsUeMapFieldName.c_str(), H5P_DEFAULT) > 0) {
                dataset = file.openDataSet(srsUeMapFieldName);
                dataset.read(srsUeMap_perCell.get(), H5::PredType::NATIVE_INT32);
                if (srsUeMap[cIdx]) {
                    CUDA_CHECK_ERR(cudaMemcpy(srsUeMap[cIdx], srsUeMap_perCell.get(), data.nActiveUe*sizeof(int32_t), cudaMemcpyHostToDevice));
                }
            }
    }

    if (H5Lexists(file.getId(), "sinVal_asim", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("sinVal_asim");
        dataset.read(sinVal_asim, H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->sinVal_asim)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->sinVal_asim, sinVal_asim, sinValSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "sinVal", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("sinVal");
        dataset.read(sinVal.get(), H5::PredType::NATIVE_FLOAT);
        if (cellGrpPrmsGpu->sinVal)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->sinVal, sinVal.get(), sinValSize, cudaMemcpyHostToDevice));
    }

    if (data.nBsAnt == 4 && H5Lexists(file.getId(), "prdMat_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prdMat_real");
        dataset.read(prdMat_asim_real, H5::PredType::NATIVE_FLOAT);
        dataset.read(prdMat_real, H5::PredType::NATIVE_FLOAT);
    }

    if (data.nBsAnt == 4 && H5Lexists(file.getId(), "prdMat_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prdMat_imag");
        dataset.read(prdMat_asim_imag, H5::PredType::NATIVE_FLOAT);
        dataset.read(prdMat_imag, H5::PredType::NATIVE_FLOAT);
    }

    if (data.nBsAnt == 4 && H5Lexists(file.getId(), "prdMat_real", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "prdMat_imag", H5P_DEFAULT) > 0) {
        for (int pIdx = 0; pIdx < prdLen; pIdx++) {
            prdMat_asim[pIdx].x = prdMat_asim_real[pIdx];
            prdMat_asim[pIdx].y = prdMat_asim_imag[pIdx];
            prdMat[pIdx].x = prdMat_real[pIdx];
            prdMat[pIdx].y = prdMat_imag[pIdx];
        }
        if (cellGrpPrmsGpu->prdMat_asim)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->prdMat_asim, prdMat_asim, prdSize, cudaMemcpyHostToDevice));

        if (cellGrpPrmsGpu->prdMat)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->prdMat, prdMat, prdSize, cudaMemcpyHostToDevice));
    }

    if (data.nBsAnt == 4 && H5Lexists(file.getId(), "detMat_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("detMat_real");
        dataset.read(detMat_asim_real, H5::PredType::NATIVE_FLOAT);
        dataset.read(detMat_real, H5::PredType::NATIVE_FLOAT);
    }

    if (data.nBsAnt == 4 && H5Lexists(file.getId(), "detMat_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("detMat_imag");
        dataset.read(detMat_asim_imag, H5::PredType::NATIVE_FLOAT);
        dataset.read(detMat_imag, H5::PredType::NATIVE_FLOAT);
    }

    if (data.nBsAnt == 4 && H5Lexists(file.getId(), "detMat_real", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "detMat_imag", H5P_DEFAULT) > 0) {
        for (int dIdx = 0; dIdx < detLen; dIdx++) {
            detMat_asim[dIdx].x = detMat_asim_real[dIdx];
            detMat_asim[dIdx].y = detMat_asim_imag[dIdx];
            detMat[dIdx].x = detMat_real[dIdx];
            detMat[dIdx].y = detMat_imag[dIdx];
        }
        if (cellGrpPrmsGpu->detMat_asim)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->detMat_asim, detMat_asim, detSize, cudaMemcpyHostToDevice));

        if (cellGrpPrmsGpu->detMat)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->detMat, detMat, detSize, cudaMemcpyHostToDevice));
    }

    if (H5Lexists(file.getId(), "estH_fr_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("estH_fr_real");
        dataset.read(estH_fr_real, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "estH_fr_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("estH_fr_imag");
        dataset.read(estH_fr_imag, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "estH_fr_real", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "estH_fr_imag", H5P_DEFAULT) > 0) {
        for (int hIdx = 0; hIdx < hLen; hIdx++) {
            estH_fr_half[hIdx].x = __float2bfloat16(estH_fr_real[hIdx]);
            estH_fr_half[hIdx].y = __float2bfloat16(estH_fr_imag[hIdx]);
            estH_fr[hIdx].x = estH_fr_real[hIdx];
            estH_fr[hIdx].y = estH_fr_imag[hIdx];
        }

        if (cellGrpPrmsGpu->estH_fr_half)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->estH_fr_half, estH_fr_half, hHalfSize, cudaMemcpyHostToDevice));

        if (cellGrpPrmsGpu->estH_fr)
            CUDA_CHECK_ERR(cudaMemcpy(cellGrpPrmsGpu->estH_fr, estH_fr, hSize, cudaMemcpyHostToDevice));

    }

    // free allocated memories
    if (estH_fr_real) CUDA_CHECK_ERR(cudaFreeHost(estH_fr_real));
    if (estH_fr_imag) CUDA_CHECK_ERR(cudaFreeHost(estH_fr_imag));
    if (estH_fr_half) CUDA_CHECK_ERR(cudaFreeHost(estH_fr_half));
    if (estH_fr) CUDA_CHECK_ERR(cudaFreeHost(estH_fr));
    if (cellId) CUDA_CHECK_ERR(cudaFreeHost(cellId));
    if (numUeSchdPerCellTTIArr) CUDA_CHECK_ERR(cudaFreeHost(numUeSchdPerCellTTIArr));
    if (cellAssoc) CUDA_CHECK_ERR(cudaFreeHost(cellAssoc));
    if (cellAssocActUe) CUDA_CHECK_ERR(cudaFreeHost(cellAssocActUe));
    if (avgRates) CUDA_CHECK_ERR(cudaFreeHost(avgRates));
    if (avgRatesActUe) CUDA_CHECK_ERR(cudaFreeHost(avgRatesActUe));
    if (tbErrLast) CUDA_CHECK_ERR(cudaFreeHost(tbErrLast));
    if (tbErrLastActUe) CUDA_CHECK_ERR(cudaFreeHost(tbErrLastActUe));
    if (postEqSinr) CUDA_CHECK_ERR(cudaFreeHost(postEqSinr));
    if (wbSinr) CUDA_CHECK_ERR(cudaFreeHost(wbSinr));
    if (allocSol) CUDA_CHECK_ERR(cudaFreeHost(allocSol));
    if (mcsSelSol) CUDA_CHECK_ERR(cudaFreeHost(mcsSelSol));
    if (layerSelSol) CUDA_CHECK_ERR(cudaFreeHost(layerSelSol));
    if (setSchdUePerCellTTI) CUDA_CHECK_ERR(cudaFreeHost(setSchdUePerCellTTI));
    if (newDataActUe) CUDA_CHECK_ERR(cudaFreeHost(newDataActUe));
    if (prioWeightActUe) CUDA_CHECK_ERR(cudaFreeHost(prioWeightActUe));
    if (allocSolLastTx) CUDA_CHECK_ERR(cudaFreeHost(allocSolLastTx));
    if (layerSelSolLastTx) CUDA_CHECK_ERR(cudaFreeHost(layerSelSolLastTx));
    if (mcsSelSolLastTx) CUDA_CHECK_ERR(cudaFreeHost(mcsSelSolLastTx)); 
    if (perCellPrgMsk) CUDA_CHECK_ERR(cudaFreeHost(perCellPrgMsk));
    if (prgMsk) CUDA_CHECK_ERR(cudaFreeHost(prgMsk));
    if (sinVal_asim) CUDA_CHECK_ERR(cudaFreeHost(sinVal_asim));
    if (prdMat_asim) CUDA_CHECK_ERR(cudaFreeHost(prdMat_asim));
    if (prdMat_asim_real) CUDA_CHECK_ERR(cudaFreeHost(prdMat_asim_real));
    if (prdMat_asim_imag) CUDA_CHECK_ERR(cudaFreeHost(prdMat_asim_imag));
    if (detMat_asim) CUDA_CHECK_ERR(cudaFreeHost(detMat_asim));
    if (detMat_asim_real) CUDA_CHECK_ERR(cudaFreeHost(detMat_asim_real));
    if (detMat_asim_imag) CUDA_CHECK_ERR(cudaFreeHost(detMat_asim_imag));
    if (detMat) CUDA_CHECK_ERR(cudaFreeHost(detMat));
    if (detMat_real) CUDA_CHECK_ERR(cudaFreeHost(detMat_real));
    if (detMat_imag) CUDA_CHECK_ERR(cudaFreeHost(detMat_imag));
    if (prdMat) CUDA_CHECK_ERR(cudaFreeHost(prdMat));
    if (prdMat_real) CUDA_CHECK_ERR(cudaFreeHost(prdMat_real));
    if (prdMat_imag) CUDA_CHECK_ERR(cudaFreeHost(prdMat_imag));
}


void loadFromH5_CPU(const std::string&                 filename,
                    cumac::cumacCellGrpUeStatus*       cellGrpUeStatusCpu,
                    cumac::cumacCellGrpPrms*           cellGrpPrmsCpu,
                    cumac::cumacSchdSol*               schdSolCpu) {
    
    cumac::cumacSchedulerParam data;

    // open TV H5 file
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // Open the dataset
    H5::DataSet dataset = file.openDataSet("cumacSchedulerParam");

    // Get the compound data type
    H5::CompType compoundType = dataset.getCompType();

    // Read the data from the dataset
    dataset.read(&data, compoundType);

    uint8_t DL = data.dlSchInd;

    // pre-allocate CPU data structures and buffers
    float*                       estH_fr_real;
    float*                       estH_fr_imag;
    float*                       prdMat_real;
    float*                       prdMat_imag;
    float*                       detMat_real;
    float*                       detMat_imag;

    // buffer sizes
    uint32_t prgMskSize = sizeof(uint8_t*)*data.nCell;
    uint32_t perCellPrgMskSize = sizeof(uint8_t)*data.nPrbGrp;
    uint32_t tbeSize = sizeof(int8_t)*data.nUe; 
    uint32_t tbeActUeSize = sizeof(int8_t)*data.nActiveUe;
    uint32_t ndActSize = sizeof(int8_t)*data.nActiveUe;
    uint32_t allocLTSize = sizeof(int16_t)*2*data.nUe; 
    uint32_t mcsSelLTSize = sizeof(int16_t)*data.nUe; 
    uint32_t layerSelLTSize = sizeof(uint8_t)*data.nUe; 
    uint32_t ueMapSize = sizeof(int32_t)*data.nUe; 
    uint32_t hfrSize = sizeof(cuComplex*)*data.nCell;
    uint32_t perCellHfrLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nUeAnt;
    uint32_t perCellHfrSize = sizeof(cuComplex)*perCellHfrLen;
    uint32_t perCellHfrRealImagSize = sizeof(float)*perCellHfrLen;
    uint32_t prdLen;
    uint32_t detLen;
    if (DL == 1) { // DL
        prdLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nBsAnt;
        detLen = data.nUe*data.nPrbGrp*data.nUeAnt*data.nUeAnt;
    } else {
        prdLen = data.nUe*data.nPrbGrp*data.nUeAnt*data.nUeAnt;
        detLen = data.nUe*data.nPrbGrp*data.nBsAnt*data.nBsAnt; 
    }
    uint32_t prdSize = sizeof(cuComplex)*prdLen;
    uint32_t prdRealImagSize = sizeof(float)*prdLen;
    uint32_t detSize = sizeof(cuComplex)*detLen;
    uint32_t detRealImagSize = sizeof(float)*detLen;
    uint32_t sinValSize; 
    if (data.nBsAnt == 4) { // 4TR
        sinValSize = sizeof(float)*data.nUe*data.nPrbGrp*data.nUeAnt;
    } else if (data.nBsAnt == 64) { // 64TR
        sinValSize = sizeof(float)*data.nActiveUe*data.nPrbGrp*data.nUeAnt;
    }
    uint32_t assocActUeSize = sizeof(uint8_t)*data.nCell*data.nActiveUe;
    uint32_t arActUeSize = sizeof(float)*data.nActiveUe;
    uint32_t setSchdUeSize = sizeof(uint16_t)*data.nCell*data.numUeSchdPerCellTTI;
    uint32_t sinrSize = sizeof(float)*data.nActiveUe*data.nPrbGrp*data.nUeAnt;
    uint32_t wbSinrSize = sizeof(float)*data.nActiveUe*data.nUeAnt;
    uint32_t cidSize = sizeof(uint16_t)*data.nCell;
    uint32_t assocSize = sizeof(uint8_t)*data.nCell*data.nUe; 
    uint32_t arSize = sizeof(float)*data.nUe;
    uint32_t mcsSelSolSize = sizeof(int16_t)*data.nUe;
    uint32_t layerSize = sizeof(uint8_t)*data.nUe;
    uint32_t hLen = data.nPrbGrp*data.nUe*data.nCell*data.nBsAnt*data.nUeAnt;
    uint32_t hSize = sizeof(cuComplex)*hLen;
    uint32_t hHalfSize = sizeof(__nv_bfloat162)*hLen;
    uint32_t gpuAllocSolSize;

    if (data.allocType == 1) {
        gpuAllocSolSize = sizeof(int16_t)*2*data.nUe;
    } else {
        gpuAllocSolSize = sizeof(int16_t)*data.nCell*data.nPrbGrp;
    }    

    CUDA_CHECK_ERR(cudaMallocHost((void **)&estH_fr_real, hLen*sizeof(float)));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&estH_fr_imag, hLen*sizeof(float)));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_real, prdRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&prdMat_imag, prdRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_real, detRealImagSize));
    CUDA_CHECK_ERR(cudaMallocHost((void **)&detMat_imag, detRealImagSize));

    // assign values for constant parameters
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

    // load all existing datasets
    if (H5Lexists(file.getId(), "cellId", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("cellId");
        dataset.read(cellGrpPrmsCpu->cellId, H5::PredType::NATIVE_UINT16);
    }

    if (H5Lexists(file.getId(), "numUeSchdPerCellTTIArr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("numUeSchdPerCellTTIArr");
        if (cellGrpPrmsCpu->numUeSchdPerCellTTIArr)
            dataset.read(cellGrpPrmsCpu->numUeSchdPerCellTTIArr, H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(file.getId(), "cellAssoc", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("cellAssoc");
        dataset.read(cellGrpPrmsCpu->cellAssoc, H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(file.getId(), "cellAssocActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("cellAssocActUe");
        dataset.read(cellGrpPrmsCpu->cellAssocActUe, H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(file.getId(), "avgRates", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("avgRates");
        dataset.read(cellGrpUeStatusCpu->avgRates, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "avgRatesActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("avgRatesActUe");
        dataset.read(cellGrpUeStatusCpu->avgRatesActUe, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "tbErrLast", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("tbErrLast");
        dataset.read(cellGrpUeStatusCpu->tbErrLast, H5::PredType::NATIVE_INT8);
    }

    if (H5Lexists(file.getId(), "tbErrLastActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("tbErrLastActUe");
        dataset.read(cellGrpUeStatusCpu->tbErrLastActUe, H5::PredType::NATIVE_INT8);
    }

    if (H5Lexists(file.getId(), "postEqSinr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("postEqSinr");
        dataset.read(cellGrpPrmsCpu->postEqSinr, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "wbSinr", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("wbSinr");
        dataset.read(cellGrpPrmsCpu->wbSinr, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "allocSol", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("allocSol");
        dataset.read(schdSolCpu->allocSol, H5::PredType::NATIVE_INT16);
    }

    if (H5Lexists(file.getId(), "setSchdUePerCellTTI", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("setSchdUePerCellTTI");
        dataset.read(schdSolCpu->setSchdUePerCellTTI, H5::PredType::NATIVE_UINT16);   
    }

    if (H5Lexists(file.getId(), "mcsSelSol", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("mcsSelSol");
        dataset.read(schdSolCpu->mcsSelSol, H5::PredType::NATIVE_INT16);   
    }

    if (H5Lexists(file.getId(), "layerSelSol", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("layerSelSol");
        dataset.read(schdSolCpu->layerSelSol, H5::PredType::NATIVE_UINT8); 
    }

    if (H5Lexists(file.getId(), "newDataActUe", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("newDataActUe");
        dataset.read(cellGrpUeStatusCpu->newDataActUe, H5::PredType::NATIVE_INT8);   
    }

    if (H5Lexists(file.getId(), "allocSolLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("allocSolLastTx");
        dataset.read(cellGrpUeStatusCpu->allocSolLastTx, H5::PredType::NATIVE_INT16);   
    }

    if (H5Lexists(file.getId(), "mcsSelSolLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("mcsSelSolLastTx");
        dataset.read(cellGrpUeStatusCpu->mcsSelSolLastTx, H5::PredType::NATIVE_INT16);   
    }

    if (H5Lexists(file.getId(), "layerSelSolLastTx", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("layerSelSolLastTx");
        dataset.read(cellGrpUeStatusCpu->layerSelSolLastTx, H5::PredType::NATIVE_UINT8);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
            std::string prgMskFieldName = "prgMsk" + std::to_string(cIdx);

            if (H5Lexists(file.getId(), prgMskFieldName.c_str(), H5P_DEFAULT) > 0) {
                dataset = file.openDataSet(prgMskFieldName);
                dataset.read(cellGrpPrmsCpu->prgMsk[cIdx], H5::PredType::NATIVE_UINT8);
            }
    }


    if (H5Lexists(file.getId(), "sinVal", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("sinVal");
        dataset.read(cellGrpPrmsCpu->sinVal, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "prdMat_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prdMat_real");
        dataset.read(prdMat_real, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "prdMat_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("prdMat_imag");
        dataset.read(prdMat_imag, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "prdMat_real", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "prdMat_imag", H5P_DEFAULT) > 0) {
        for (int pIdx = 0; pIdx < prdLen; pIdx++) {
            cellGrpPrmsCpu->prdMat[pIdx].x = prdMat_real[pIdx];
            cellGrpPrmsCpu->prdMat[pIdx].y = prdMat_imag[pIdx];
        }
    }

    if (H5Lexists(file.getId(), "detMat_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("detMat_real");
        dataset.read(detMat_real, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "detMat_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("detMat_imag");
        dataset.read(detMat_imag, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "detMat_real", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "detMat_imag", H5P_DEFAULT) > 0) {
        for (int dIdx = 0; dIdx < detLen; dIdx++) {
            cellGrpPrmsCpu->detMat[dIdx].x = detMat_real[dIdx];
            cellGrpPrmsCpu->detMat[dIdx].y = detMat_imag[dIdx];
        }
    }

    if (H5Lexists(file.getId(), "estH_fr_real", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("estH_fr_real");
        dataset.read(estH_fr_real, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "estH_fr_imag", H5P_DEFAULT) > 0) {
        dataset = file.openDataSet("estH_fr_imag");
        dataset.read(estH_fr_imag, H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(file.getId(), "estH_fr_real", H5P_DEFAULT) > 0 && H5Lexists(file.getId(), "estH_fr_imag", H5P_DEFAULT) > 0) {
        for (int hIdx = 0; hIdx < hLen; hIdx++) {
            cellGrpPrmsCpu->estH_fr[hIdx].x = estH_fr_real[hIdx];
            cellGrpPrmsCpu->estH_fr[hIdx].y = estH_fr_imag[hIdx];
        }
    }

    // free allocated memories
    if (estH_fr_real) CUDA_CHECK_ERR(cudaFreeHost(estH_fr_real));
    if (estH_fr_imag) CUDA_CHECK_ERR(cudaFreeHost(estH_fr_imag));
    if (detMat_real) CUDA_CHECK_ERR(cudaFreeHost(detMat_real));
    if (detMat_imag) CUDA_CHECK_ERR(cudaFreeHost(detMat_imag));
    if (prdMat_real) CUDA_CHECK_ERR(cudaFreeHost(prdMat_real));
    if (prdMat_imag) CUDA_CHECK_ERR(cudaFreeHost(prdMat_imag));
}
