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
#include "h5TvCreate.h"

using namespace H5;

void saveToH5_Asim(const std::string&                 filename,
                   cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
                   cumac::cumacCellGrpPrms*           cellGrpPrms,
                   cumac::cumacSchdSol*               schdSol,
                   uint8_t                            saveBfWeights) {

    cumac::cumacSchedulerParam param;
        
    param.nUe = cellGrpPrms->nUe;
    param.nCell = cellGrpPrms->nCell;
    param.totNumCell = cellGrpPrms->nCell;
    param.nPrbGrp = cellGrpPrms->nPrbGrp;
    param.nBsAnt = cellGrpPrms->nBsAnt;
    param.nUeAnt = cellGrpPrms->nUeAnt;
    param.W  = cellGrpPrms->W;
    param.sigmaSqrd = cellGrpPrms->sigmaSqrd;
    param.betaCoeff = cellGrpPrms->betaCoeff;
    param.precodingScheme = cellGrpPrms->precodingScheme;
    param.receiverScheme = cellGrpPrms->receiverScheme;
    param.allocType = cellGrpPrms->allocType;
    param.columnMajor = 0;
    param.nActiveUe = cellGrpPrms->nActiveUe;
    param.numUeSchdPerCellTTI = cellGrpPrms->numUeSchdPerCellTTI;
    param.sinValThr = cellGrpPrms->sinValThr;
    param.numUeForGrpPerCell = cellGrpPrms->numUeForGrpPerCell;
    param.chanCorrThr = cellGrpPrms->chanCorrThr;
    param.muCoeff = cellGrpPrms->muCoeff;
    param.srsSnrThr = cellGrpPrms->srsSnrThr;
    param.nMaxActUePerCell = cellGrpPrms->nMaxActUePerCell;
    param.zfCoeff = cellGrpPrms->zfCoeff;
    param.nMaxUePerGrpUl = cellGrpPrms->nMaxUePerGrpUl;
    param.nMaxUePerGrpDl = cellGrpPrms->nMaxUePerGrpDl;
    param.nMaxLayerPerGrpUl = cellGrpPrms->nMaxLayerPerGrpUl;
    param.nMaxLayerPerGrpDl = cellGrpPrms->nMaxLayerPerGrpDl;
    param.nMaxLayerPerUeSuUl = cellGrpPrms->nMaxLayerPerUeSuUl;
    param.nMaxLayerPerUeSuDl = cellGrpPrms->nMaxLayerPerUeSuDl;
    param.nMaxLayerPerUeMuUl = cellGrpPrms->nMaxLayerPerUeMuUl;
    param.nMaxLayerPerUeMuDl = cellGrpPrms->nMaxLayerPerUeMuDl;
    param.nMaxUegPerCellDl = cellGrpPrms->nMaxUegPerCellDl;
    param.nMaxUegPerCellUl = cellGrpPrms->nMaxUegPerCellUl;
    param.mcsSelSinrCapThr = cellGrpPrms->mcsSelSinrCapThr;
    param.muGrpSrsSnrMaxGap = cellGrpPrms->muGrpSrsSnrMaxGap;
    param.muGrpSrsSnrSplitThr = cellGrpPrms->muGrpSrsSnrSplitThr;
    param.bfPowAllocScheme = cellGrpPrms->bfPowAllocScheme;
    param.muGrpUpdate = cellGrpPrms->muGrpUpdate;
    param.mcsSelLutType = cellGrpPrms->mcsSelLutType;
    param.semiStatFreqAlloc = cellGrpPrms->semiStatFreqAlloc;
    param.harqEnabledInd = cellGrpPrms->harqEnabledInd;
    param.mcsSelCqi = cellGrpPrms->mcsSelCqi;
    param.dlSchInd = cellGrpPrms->dlSchInd;

    // Create a compound data type
    H5::CompType compType(sizeof(cumac::cumacSchedulerParam));
    compType.insertMember("nUe", HOFFSET(cumac::cumacSchedulerParam, nUe), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nCell", HOFFSET(cumac::cumacSchedulerParam, nCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("totNumCell", HOFFSET(cumac::cumacSchedulerParam, totNumCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nPrbGrp", HOFFSET(cumac::cumacSchedulerParam, nPrbGrp), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nBsAnt", HOFFSET(cumac::cumacSchedulerParam, nBsAnt), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nUeAnt", HOFFSET(cumac::cumacSchedulerParam, nUeAnt), H5::PredType::NATIVE_UINT8);
    compType.insertMember("W", HOFFSET(cumac::cumacSchedulerParam, W), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("sigmaSqrd", HOFFSET(cumac::cumacSchedulerParam, sigmaSqrd), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("betaCoeff", HOFFSET(cumac::cumacSchedulerParam, betaCoeff), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("precodingScheme", HOFFSET(cumac::cumacSchedulerParam, precodingScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("receiverScheme", HOFFSET(cumac::cumacSchedulerParam, receiverScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("allocType", HOFFSET(cumac::cumacSchedulerParam, allocType), H5::PredType::NATIVE_UINT8);
    compType.insertMember("columnMajor", HOFFSET(cumac::cumacSchedulerParam, columnMajor), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nActiveUe", HOFFSET(cumac::cumacSchedulerParam, nActiveUe), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nMaxActUePerCell", HOFFSET(cumac::cumacSchedulerParam, nMaxActUePerCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("numUeSchdPerCellTTI", HOFFSET(cumac::cumacSchedulerParam, numUeSchdPerCellTTI), H5::PredType::NATIVE_UINT8);
    compType.insertMember("sinValThr", HOFFSET(cumac::cumacSchedulerParam, sinValThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("numUeForGrpPerCell", HOFFSET(cumac::cumacSchedulerParam, numUeForGrpPerCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("chanCorrThr", HOFFSET(cumac::cumacSchedulerParam, chanCorrThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("muCoeff", HOFFSET(cumac::cumacSchedulerParam, muCoeff), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("srsSnrThr", HOFFSET(cumac::cumacSchedulerParam, srsSnrThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("zfCoeff", HOFFSET(cumac::cumacSchedulerParam, zfCoeff), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("nMaxUePerGrpUl", HOFFSET(cumac::cumacSchedulerParam, nMaxUePerGrpUl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxUePerGrpDl", HOFFSET(cumac::cumacSchedulerParam, nMaxUePerGrpDl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxLayerPerGrpUl", HOFFSET(cumac::cumacSchedulerParam, nMaxLayerPerGrpUl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxLayerPerGrpDl", HOFFSET(cumac::cumacSchedulerParam, nMaxLayerPerGrpDl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxLayerPerUeSuUl", HOFFSET(cumac::cumacSchedulerParam, nMaxLayerPerUeSuUl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxLayerPerUeSuDl", HOFFSET(cumac::cumacSchedulerParam, nMaxLayerPerUeSuDl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxLayerPerUeMuUl", HOFFSET(cumac::cumacSchedulerParam, nMaxLayerPerUeMuUl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxLayerPerUeMuDl", HOFFSET(cumac::cumacSchedulerParam, nMaxLayerPerUeMuDl), H5::PredType::NATIVE_UINT8);   
    compType.insertMember("nMaxUegPerCellDl", HOFFSET(cumac::cumacSchedulerParam, nMaxUegPerCellDl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxUegPerCellUl", HOFFSET(cumac::cumacSchedulerParam, nMaxUegPerCellUl), H5::PredType::NATIVE_UINT8);  
    compType.insertMember("mcsSelSinrCapThr", HOFFSET(cumac::cumacSchedulerParam, mcsSelSinrCapThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("muGrpSrsSnrMaxGap", HOFFSET(cumac::cumacSchedulerParam, muGrpSrsSnrMaxGap), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("muGrpSrsSnrSplitThr", HOFFSET(cumac::cumacSchedulerParam, muGrpSrsSnrSplitThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("bfPowAllocScheme", HOFFSET(cumac::cumacSchedulerParam, bfPowAllocScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("muGrpUpdate", HOFFSET(cumac::cumacSchedulerParam, muGrpUpdate), H5::PredType::NATIVE_UINT8);   
    compType.insertMember("mcsSelLutType", HOFFSET(cumac::cumacSchedulerParam, mcsSelLutType), H5::PredType::NATIVE_UINT8); 
    compType.insertMember("semiStatFreqAlloc", HOFFSET(cumac::cumacSchedulerParam, semiStatFreqAlloc), H5::PredType::NATIVE_UINT8);
    compType.insertMember("harqEnabledInd", HOFFSET(cumac::cumacSchedulerParam, harqEnabledInd), H5::PredType::NATIVE_UINT8);
    compType.insertMember("mcsSelCqi", HOFFSET(cumac::cumacSchedulerParam, mcsSelCqi), H5::PredType::NATIVE_UINT8);
    compType.insertMember("dlSchInd", HOFFSET(cumac::cumacSchedulerParam, dlSchInd), H5::PredType::NATIVE_UINT8);   

    // Open the HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Create a dataset
    H5::DataSet dataset = file.createDataSet("cumacSchedulerParam", compType, H5::DataSpace());

    // Write the data to the dataset
    dataset.write(&param, compType);

    uint16_t numUe;
    int numCfrPerCell;
    int numPrdPerCell;
    int numDetPerCell;

    if (cellGrpPrms->nBsAnt == 4) { // 4TR
        numCfrPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt;
        numPrdPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        numDetPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        numUe = cellGrpPrms->nUe;
    } else if (cellGrpPrms->nBsAnt == 64) { // 64TR
        numCfrPerCell = cellGrpPrms->nCell*cellGrpPrms->numUeForGrpPerCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nBsAnt;
        numUe = cellGrpPrms->nActiveUe;
    }
    
    hsize_t dims[] = {static_cast<hsize_t>(numCfrPerCell)};

    if (cellGrpPrms->srsEstChan) {
        cuComplex** srsEstChan              = new cuComplex*[cellGrpPrms->nCell];
        cuComplex*  estH_fr_perCell         = new cuComplex[numCfrPerCell];
        float*      estH_fr_real            = new float[numCfrPerCell];
        float*      estH_fr_imag            = new float[numCfrPerCell];

        CUDA_CHECK_ERR(cudaMemcpy(srsEstChan, cellGrpPrms->srsEstChan, cellGrpPrms->nCell*sizeof(cuComplex*), cudaMemcpyDeviceToHost));
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMemcpy(estH_fr_perCell, srsEstChan[cIdx], numCfrPerCell*sizeof(cuComplex), cudaMemcpyDeviceToHost));
            for (int hIdx = 0; hIdx < numCfrPerCell; hIdx++) {
                estH_fr_real[hIdx] = estH_fr_perCell[hIdx].x;
                estH_fr_imag[hIdx] = estH_fr_perCell[hIdx].y;
            }

            std::string cfrRealFieldName = "estH_fr_real_cell" + std::to_string(cIdx);
            H5::DataSpace dataspaceEstH_fr_real(1, dims);
            dataset = file.createDataSet(cfrRealFieldName, H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_real);
            dataset.write(estH_fr_real, H5::PredType::NATIVE_FLOAT);

            std::string cfrImagFieldName = "estH_fr_imag_cell" + std::to_string(cIdx);
            H5::DataSpace dataspaceEstH_fr_imag(1, dims);
            dataset = file.createDataSet(cfrImagFieldName, H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_imag);
            dataset.write(estH_fr_imag, H5::PredType::NATIVE_FLOAT);
        }

        delete[] srsEstChan;
        delete[] estH_fr_perCell;
        delete[] estH_fr_real;
        delete[] estH_fr_imag;
    }

    if (cellGrpPrms->srsWbSnr) {
        std::unique_ptr<float []>  srsWbSnr = std::make_unique<float []>(cellGrpPrms->nActiveUe);
        CUDA_CHECK_ERR(cudaMemcpy(srsWbSnr.get(), cellGrpPrms->srsWbSnr, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceSrsWbSnr(1, dims);
        dataset = file.createDataSet("srsWbSnr", H5::PredType::NATIVE_FLOAT, dataspaceSrsWbSnr);
        dataset.write(srsWbSnr.get(), H5::PredType::NATIVE_FLOAT);
    }

    if (cellGrpPrms->srsUeMap) {
        std::unique_ptr<int32_t* []> srsUeMap = std::make_unique<int32_t* []>(cellGrpPrms->nCell);
        std::unique_ptr<int32_t []> srsUeMap_perCell = std::make_unique<int32_t []>(cellGrpPrms->nActiveUe);
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);

        CUDA_CHECK_ERR(cudaMemcpy(srsUeMap.get(), cellGrpPrms->srsUeMap, cellGrpPrms->nCell*sizeof(int32_t*), cudaMemcpyDeviceToHost));
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMemcpy(srsUeMap_perCell.get(), srsUeMap[cIdx], cellGrpPrms->nActiveUe*sizeof(int32_t), cudaMemcpyDeviceToHost));
            
            std::string srsUeMapFieldName = "srsUeMap_cell" + std::to_string(cIdx);
            H5::DataSpace dataspaceSrsUeMap(1, dims);
            dataset = file.createDataSet(srsUeMapFieldName, H5::PredType::NATIVE_INT32, dataspaceSrsUeMap);
            dataset.write(srsUeMap_perCell.get(), H5::PredType::NATIVE_INT32);
        }
    }

    if (cellGrpPrms->bsTxPow) {
        std::vector<float> bsTxPow(cellGrpPrms->nCell);
        CUDA_CHECK_ERR(cudaMemcpy(bsTxPow.data(), cellGrpPrms->bsTxPow, cellGrpPrms->nCell*sizeof(float), cudaMemcpyDeviceToHost));

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell);
        H5::DataSpace dataspaceBsTxPow(1, dims);
        dataset = file.createDataSet("bsTxPow",  H5::PredType::NATIVE_FLOAT, dataspaceBsTxPow);
        dataset.write(bsTxPow.data(), H5::PredType::NATIVE_FLOAT);
    }

    if (cellGrpPrms->blerTargetActUe) {
        std::vector<float> blerTargetActUe(cellGrpPrms->nActiveUe);
        CUDA_CHECK_ERR(cudaMemcpy(blerTargetActUe.data(), cellGrpPrms->blerTargetActUe, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceBlerTargetActUe(1, dims);
        dataset = file.createDataSet("blerTargetActUe", H5::PredType::NATIVE_FLOAT, dataspaceBlerTargetActUe);
        dataset.write(blerTargetActUe.data(), H5::PredType::NATIVE_FLOAT);
    }

    if (cellGrpPrms->nBsAnt == 64 && cellGrpPrms->sinVal) {
        float* sinVal = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(sinVal, cellGrpPrms->sinVal, cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost));

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceSinVal(1, dims);
        dataset = file.createDataSet("sinVal", H5::PredType::NATIVE_FLOAT, dataspaceSinVal);
        dataset.write(sinVal, H5::PredType::NATIVE_FLOAT);

        delete[] sinVal;
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->sinVal_asim) {
        float* sinVal_asim = new float[cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(sinVal_asim, cellGrpPrms->sinVal_asim, cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost));
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceSinVal_asim(1, dims);
        dataset = file.createDataSet("sinVal_asim", H5::PredType::NATIVE_FLOAT, dataspaceSinVal_asim);
        dataset.write(sinVal_asim, H5::PredType::NATIVE_FLOAT);

        delete[] sinVal_asim;
    }

    if (saveBfWeights == 1 && cellGrpPrms->prdMat) {
        std::vector<cuComplex> prdMat(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);
        std::vector<float> prdMat_real(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);
        std::vector<float> prdMat_imag(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);

        CUDA_CHECK_ERR(cudaMemcpy(prdMat.data(), cellGrpPrms->prdMat, cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_*sizeof(cuComplex), cudaMemcpyDeviceToHost));
        for (int hIdx = 0; hIdx < cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_; hIdx++) {
            prdMat_real[hIdx] = prdMat[hIdx].x;
            prdMat_imag[hIdx] = prdMat[hIdx].y;
        }
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);
        H5::DataSpace dataspacePrdMat_real(1, dims);
        dataset = file.createDataSet("prdMat_real", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_real);
        dataset.write(prdMat_real.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspacePrdMat_imag(1, dims);
        dataset = file.createDataSet("prdMat_imag", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_imag);
        dataset.write(prdMat_imag.data(), H5::PredType::NATIVE_FLOAT); 
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->prdMat_asim) {
        cuComplex*  prdMat_asim             = new cuComplex[numPrdPerCell];
        float*      prdMat_real             = new float[numPrdPerCell];
        float*      prdMat_imag             = new float[numPrdPerCell];

        CUDA_CHECK_ERR(cudaMemcpy(prdMat_asim, cellGrpPrms->prdMat_asim, numPrdPerCell*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int hIdx = 0; hIdx < numPrdPerCell; hIdx++) {
            prdMat_real[hIdx] = prdMat_asim[hIdx].x;
            prdMat_imag[hIdx] = prdMat_asim[hIdx].y;
        }

        dims[0] = static_cast<hsize_t>(numPrdPerCell);
        H5::DataSpace dataspacePrdMat_real(1, dims);
        dataset = file.createDataSet("prdMat_real", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_real);
        dataset.write(prdMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspacePrdMat_imag(1, dims);
        dataset = file.createDataSet("prdMat_imag", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_imag);
        dataset.write(prdMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] prdMat_asim;
        delete[] prdMat_real;
        delete[] prdMat_imag;
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->detMat_asim) {
        cuComplex*  detMat_asim             = new cuComplex[numDetPerCell]; 
        float*      detMat_real             = new float[numDetPerCell];
        float*      detMat_imag             = new float[numDetPerCell];

        CUDA_CHECK_ERR(cudaMemcpy(detMat_asim, cellGrpPrms->detMat_asim, numDetPerCell*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int hIdx = 0; hIdx < numDetPerCell; hIdx++) {
            detMat_real[hIdx] = detMat_asim[hIdx].x;
            detMat_imag[hIdx] = detMat_asim[hIdx].y;
        }

        dims[0] = static_cast<hsize_t>(numDetPerCell);
        H5::DataSpace dataspaceDetMat_real(1, dims);
        dataset = file.createDataSet("detMat_real", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_real);
        dataset.write(detMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceDetMat_imag(1, dims);
        dataset = file.createDataSet("detMat_imag", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_imag);
        dataset.write(detMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] detMat_asim;
        delete[] detMat_real;
        delete[] detMat_imag;
    }

    if (cellGrpPrms->currSlotIdxPerCell) {
        std::vector<uint32_t> currSlotIdxPerCell(cellGrpPrms->nCell);
        CUDA_CHECK_ERR(cudaMemcpy(currSlotIdxPerCell.data(), cellGrpPrms->currSlotIdxPerCell, cellGrpPrms->nCell*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell);
        H5::DataSpace dataspaceCurrSlotIdxPerCell(1, dims);
        dataset = file.createDataSet("currSlotIdxPerCell", H5::PredType::NATIVE_UINT32, dataspaceCurrSlotIdxPerCell);
        dataset.write(currSlotIdxPerCell.data(), H5::PredType::NATIVE_UINT32);  
    }   

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->cellId) {
        uint16_t* cellId = new uint16_t[cellGrpPrms->nCell];
        CUDA_CHECK_ERR(cudaMemcpy(cellId, cellGrpPrms->cellId, cellGrpPrms->nCell*sizeof(uint16_t), cudaMemcpyDeviceToHost));

        dims[0] = {static_cast<hsize_t>(cellGrpPrms->nCell)};
        H5::DataSpace dataspaceCellId(1, dims);
        dataset = file.createDataSet("cellId", H5::PredType::NATIVE_UINT16, dataspaceCellId);
        dataset.write(cellId, H5::PredType::NATIVE_UINT16);

        delete[] cellId;
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->numUeSchdPerCellTTIArr) {
        uint8_t* numUeSchdPerCellTTIArr = new uint8_t[cellGrpPrms->nCell];
        CUDA_CHECK_ERR(cudaMemcpy(numUeSchdPerCellTTIArr, cellGrpPrms->numUeSchdPerCellTTIArr, cellGrpPrms->nCell*sizeof(uint8_t), cudaMemcpyDeviceToHost));

        dims[0] = {static_cast<hsize_t>(cellGrpPrms->nCell)};
        H5::DataSpace dataspaceNumUeSchdArr(1, dims);
        dataset = file.createDataSet("numUeSchdPerCellTTIArr", H5::PredType::NATIVE_UINT8, dataspaceNumUeSchdArr);
        dataset.write(numUeSchdPerCellTTIArr, H5::PredType::NATIVE_UINT8);

        delete[] numUeSchdPerCellTTIArr;
    }

    if (cellGrpUeStatus) {
        if (cellGrpUeStatus->lastSchdSlotActUe) {
            std::vector<uint32_t> lastSchdSlotActUe(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(lastSchdSlotActUe.data(), cellGrpUeStatus->lastSchdSlotActUe, cellGrpPrms->nActiveUe*sizeof(uint32_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceLastSchdSlotActUe(1, dims);
            dataset = file.createDataSet("lastSchdSlotActUe", H5::PredType::NATIVE_UINT32, dataspaceLastSchdSlotActUe);
            dataset.write(lastSchdSlotActUe.data(), H5::PredType::NATIVE_UINT32);   
        }

        if (cellGrpUeStatus->beamformGainCurrTx) {
            std::vector<float> beamformGainCurrTx(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(beamformGainCurrTx.data(), cellGrpUeStatus->beamformGainCurrTx, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));
            
            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceBeamformGainCurrTx(1, dims);
            dataset = file.createDataSet("beamformGainCurrTx", H5::PredType::NATIVE_FLOAT, dataspaceBeamformGainCurrTx);
            dataset.write(beamformGainCurrTx.data(), H5::PredType::NATIVE_FLOAT);
        }

        if (cellGrpUeStatus->bfGainPrgCurrTx) { 
            std::vector<float> bfGainPrgCurrTx(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp);
            CUDA_CHECK_ERR(cudaMemcpy(bfGainPrgCurrTx.data(), cellGrpUeStatus->bfGainPrgCurrTx, cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp);
            H5::DataSpace dataspaceBfGainPrgCurrTx(1, dims);
            dataset = file.createDataSet("bfGainPrgCurrTx", H5::PredType::NATIVE_FLOAT, dataspaceBfGainPrgCurrTx);
            dataset.write(bfGainPrgCurrTx.data(), H5::PredType::NATIVE_FLOAT);
        }   

        if (cellGrpUeStatus->beamformGainLastTx) {
            std::vector<float> beamformGainLastTx(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(beamformGainLastTx.data(), cellGrpUeStatus->beamformGainLastTx, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceBeamformGainLastTx(1, dims); 
            dataset = file.createDataSet("beamformGainLastTx", H5::PredType::NATIVE_FLOAT, dataspaceBeamformGainLastTx);
            dataset.write(beamformGainLastTx.data(), H5::PredType::NATIVE_FLOAT);   
        }

        if (cellGrpUeStatus->ueTxPow) {
            std::vector<float> ueTxPow(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(ueTxPow.data(), cellGrpUeStatus->ueTxPow, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceUeTxPow(1, dims);
            dataset = file.createDataSet("ueTxPow", H5::PredType::NATIVE_FLOAT, dataspaceUeTxPow);
            dataset.write(ueTxPow.data(), H5::PredType::NATIVE_FLOAT);
        }

        if (cellGrpUeStatus->noiseVarActUe) {
            std::vector<float> noiseVarActUe(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(noiseVarActUe.data(), cellGrpUeStatus->noiseVarActUe, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceNoiseVarActUe(1, dims);
            dataset = file.createDataSet("noiseVarActUe", H5::PredType::NATIVE_FLOAT, dataspaceNoiseVarActUe);
            dataset.write(noiseVarActUe.data(), H5::PredType::NATIVE_FLOAT);
        }

        if (cellGrpPrms->nBsAnt == 4 && cellGrpUeStatus->avgRates) {
            float* avgRates = new float[cellGrpPrms->nUe];
            CUDA_CHECK_ERR(cudaMemcpy(avgRates, cellGrpUeStatus->avgRates, cellGrpPrms->nUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceAvgRates(1, dims);
            dataset = file.createDataSet("avgRates", H5::PredType::NATIVE_FLOAT, dataspaceAvgRates);
            dataset.write(avgRates, H5::PredType::NATIVE_FLOAT);

            delete[] avgRates;
        }

        if (cellGrpUeStatus->avgRatesActUe) {
            float* avgRatesActUe = new float[cellGrpPrms->nActiveUe];
            CUDA_CHECK_ERR(cudaMemcpy(avgRatesActUe, cellGrpUeStatus->avgRatesActUe, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceAvgRatesActUe(1, dims);
            dataset = file.createDataSet("avgRatesActUe", H5::PredType::NATIVE_FLOAT, dataspaceAvgRatesActUe);
            dataset.write(avgRatesActUe, H5::PredType::NATIVE_FLOAT);

            delete[] avgRatesActUe;
        }

        if (cellGrpUeStatus->tbErrLast) {
            int8_t* tbErrLast = new int8_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(tbErrLast, cellGrpUeStatus->tbErrLast, numUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceTbErrLast(1, dims);
            dataset = file.createDataSet("tbErrLast", H5::PredType::NATIVE_INT8, dataspaceTbErrLast);
            dataset.write(tbErrLast, H5::PredType::NATIVE_INT8);

            delete[] tbErrLast;
        }

        if (cellGrpUeStatus->tbErrLastActUe) {
            int8_t*     tbErrLastActUe = new int8_t[cellGrpPrms->nActiveUe];
            CUDA_CHECK_ERR(cudaMemcpy(tbErrLastActUe, cellGrpUeStatus->tbErrLastActUe, cellGrpPrms->nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceTbErrLastActUe(1, dims);
            dataset = file.createDataSet("tbErrLastActUe", H5::PredType::NATIVE_INT8, dataspaceTbErrLastActUe);
            dataset.write(tbErrLastActUe, H5::PredType::NATIVE_INT8);

            delete[] tbErrLastActUe;
        }

        if (cellGrpUeStatus->prioWeightActUe) {
            uint16_t* prioWeightActUe = new uint16_t[cellGrpPrms->nActiveUe];
            CUDA_CHECK_ERR(cudaMemcpy(prioWeightActUe, cellGrpUeStatus->prioWeightActUe, cellGrpPrms->nActiveUe*sizeof(uint16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspacePwActUe(1, dims);
            dataset = file.createDataSet("prioWeightActUe", H5::PredType::NATIVE_UINT16, dataspacePwActUe);
            dataset.write(prioWeightActUe, H5::PredType::NATIVE_UINT16);

            delete[] prioWeightActUe;
        }

        if (cellGrpUeStatus->newDataActUe) {
            int8_t* newDataActUe = new int8_t[cellGrpPrms->nActiveUe]; 
            CUDA_CHECK_ERR(cudaMemcpy(newDataActUe, cellGrpUeStatus->newDataActUe, cellGrpPrms->nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceNdActUe(1, dims);
            dataset = file.createDataSet("newDataActUe", H5::PredType::NATIVE_INT8, dataspaceNdActUe);
            dataset.write(newDataActUe, H5::PredType::NATIVE_INT8);

            delete[] newDataActUe;
        }

        if (cellGrpUeStatus->allocSolLastTx) {
            int16_t* allocSolLastTx = new int16_t[2*numUe];
            CUDA_CHECK_ERR(cudaMemcpy(allocSolLastTx, cellGrpUeStatus->allocSolLastTx, 2*numUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(2*numUe);
            H5::DataSpace dataspaceAllocSolLastTx(1, dims);
            dataset = file.createDataSet("allocSolLastTx", H5::PredType::NATIVE_INT16, dataspaceAllocSolLastTx);
            dataset.write(allocSolLastTx, H5::PredType::NATIVE_INT16);

            delete[] allocSolLastTx;
        }

        if (cellGrpUeStatus->mcsSelSolLastTx) {
            int16_t* mcsSelSolLastTx = new int16_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(mcsSelSolLastTx, cellGrpUeStatus->mcsSelSolLastTx, numUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceMcsSelSolLastTx(1, dims);
            dataset = file.createDataSet("mcsSelSolLastTx", H5::PredType::NATIVE_INT16, dataspaceMcsSelSolLastTx);
            dataset.write(mcsSelSolLastTx, H5::PredType::NATIVE_INT16);

            delete[] mcsSelSolLastTx;
        }

        if (cellGrpUeStatus->layerSelSolLastTx) {
            uint8_t* layerSelSolLastTx = new uint8_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(layerSelSolLastTx, cellGrpUeStatus->layerSelSolLastTx, numUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceLayerSelSolLastTx(1, dims);
            dataset = file.createDataSet("layerSelSolLastTx", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSolLastTx);
            dataset.write(layerSelSolLastTx, H5::PredType::NATIVE_UINT8);

            delete[] layerSelSolLastTx;
        }
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->cellAssoc) {
        uint8_t* cellAssoc = new uint8_t[cellGrpPrms->nCell*cellGrpPrms->nUe];
        CUDA_CHECK_ERR(cudaMemcpy(cellAssoc, cellGrpPrms->cellAssoc, cellGrpPrms->nCell*cellGrpPrms->nUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nUe);
        H5::DataSpace dataspaceCellAssoc(1, dims);
        dataset = file.createDataSet("cellAssoc", H5::PredType::NATIVE_UINT8, dataspaceCellAssoc);
        dataset.write(cellAssoc, H5::PredType::NATIVE_UINT8);

        delete[] cellAssoc;
    }

    if (cellGrpPrms->cellAssocActUe) {
        uint8_t* cellAssocActUe = new uint8_t[cellGrpPrms->nCell*cellGrpPrms->nActiveUe];
        CUDA_CHECK_ERR(cudaMemcpy(cellAssocActUe, cellGrpPrms->cellAssocActUe, cellGrpPrms->nCell*cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceCellAssocActUe(1, dims);
        dataset = file.createDataSet("cellAssocActUe", H5::PredType::NATIVE_UINT8, dataspaceCellAssocActUe);
        dataset.write(cellAssocActUe, H5::PredType::NATIVE_UINT8);

        delete[] cellAssocActUe;
    }

    if (cellGrpPrms->postEqSinr) {
        float* postEqSinr = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(postEqSinr, cellGrpPrms->postEqSinr, cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspacePostEqSinr(1, dims);
        dataset = file.createDataSet("postEqSinr", H5::PredType::NATIVE_FLOAT, dataspacePostEqSinr);
        dataset.write(postEqSinr, H5::PredType::NATIVE_FLOAT);

        delete[] postEqSinr;
    }

    if (cellGrpPrms->wbSinr) {
        float*      wbSinr = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(wbSinr, cellGrpPrms->wbSinr, cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceWbSinr(1, dims);
        dataset = file.createDataSet("wbSinr", H5::PredType::NATIVE_FLOAT, dataspaceWbSinr);
        dataset.write(wbSinr, H5::PredType::NATIVE_FLOAT);

        delete[] wbSinr;
    }
    
    if (schdSol) {
        if (cellGrpPrms->nBsAnt == 64 && schdSol->muGrpList) {
            auto muGrpListGpu           = std::make_unique<cumac::multiCellMuGrpList>();
            CUDA_CHECK_ERR(cudaMemcpy(muGrpListGpu.get(), schdSol->muGrpList, sizeof(cumac::multiCellMuGrpList), cudaMemcpyDeviceToHost));

            std::vector<uint16_t> numUeInGrp(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            CUDA_CHECK_ERR(cudaMemcpy(numUeInGrp.data(), muGrpListGpu->numUeInGrp, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(uint16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            H5::DataSpace dataspaceNumUeInGrp(1, dims);
            dataset = file.createDataSet("numUeInGrp", H5::PredType::NATIVE_UINT16, dataspaceNumUeInGrp);
            dataset.write(numUeInGrp.data(), H5::PredType::NATIVE_UINT16);  

            std::vector<uint16_t> ueId(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_);
            CUDA_CHECK_ERR(cudaMemcpy(ueId.data(), muGrpListGpu->ueId, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_*sizeof(uint16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_);
            H5::DataSpace dataspaceUeId(1, dims);
            dataset = file.createDataSet("ueId", H5::PredType::NATIVE_UINT16, dataspaceUeId);
            dataset.write(ueId.data(), H5::PredType::NATIVE_UINT16);

            std::vector<int16_t> subbandId(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            CUDA_CHECK_ERR(cudaMemcpy(subbandId.data(), muGrpListGpu->subbandId, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(int16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            H5::DataSpace dataspaceSubbandId(1, dims);
            dataset = file.createDataSet("subbandId", H5::PredType::NATIVE_INT16, dataspaceSubbandId);
            dataset.write(subbandId.data(), H5::PredType::NATIVE_INT16);
        }

        if (schdSol->ueOrderInGrp) {
            std::vector<uint16_t> ueOrderInGrp(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(ueOrderInGrp.data(), schdSol->ueOrderInGrp, cellGrpPrms->nActiveUe*sizeof(uint16_t), cudaMemcpyDeviceToHost));    

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceUeOrderInGrp(1, dims);
            dataset = file.createDataSet("ueOrderInGrp", H5::PredType::NATIVE_UINT16, dataspaceUeOrderInGrp);
            dataset.write(ueOrderInGrp.data(), H5::PredType::NATIVE_UINT16);
        }
        
        if (schdSol->muMimoInd) {
            std::unique_ptr<uint8_t []> muMimoInd = std::make_unique<uint8_t []>(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(muMimoInd.get(), schdSol->muMimoInd, cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceMuMimoInd(1, dims);
            dataset = file.createDataSet("muMimoInd", H5::PredType::NATIVE_UINT8, dataspaceMuMimoInd);
            dataset.write(muMimoInd.get(), H5::PredType::NATIVE_UINT8);
        }

        if (schdSol->sortedUeList) {
            std::unique_ptr<uint16_t* []> sortedUeList  = std::make_unique<uint16_t* []>(cellGrpPrms->nCell);
            std::unique_ptr<uint16_t []> sortedUeList_perCell = std::make_unique<uint16_t []>(cellGrpPrms->nMaxActUePerCell);

            CUDA_CHECK_ERR(cudaMemcpy(sortedUeList.get(), schdSol->sortedUeList, cellGrpPrms->nCell*sizeof(uint16_t*), cudaMemcpyDeviceToHost));
            dims[0] = static_cast<hsize_t>(cellGrpPrms->nMaxActUePerCell);
            for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
                CUDA_CHECK_ERR(cudaMemcpy(sortedUeList_perCell.get(), sortedUeList[cIdx], cellGrpPrms->nMaxActUePerCell*sizeof(uint16_t), cudaMemcpyDeviceToHost));

                std::string sortedUeListFieldName = "sortedUeList_cell" + std::to_string(cIdx);
                H5::DataSpace dataspaceSortedUeList(1, dims);
                dataset = file.createDataSet(sortedUeListFieldName, H5::PredType::NATIVE_UINT16, dataspaceSortedUeList);
                dataset.write(sortedUeList_perCell.get(), H5::PredType::NATIVE_UINT16);
            }
        }

        if (schdSol->setSchdUePerCellTTI) {
            int numUeSchd;
            if (cellGrpPrms->nBsAnt == 4) { // 4TR
                numUeSchd = cellGrpPrms->numUeSchdPerCellTTI;
            } else { // 64TR
                numUeSchd = cellGrpPrms->numUeForGrpPerCell;
            }

            uint16_t* setSchdUePerCellTTI = new uint16_t[cellGrpPrms->nCell*numUeSchd];
            CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTI, schdSol->setSchdUePerCellTTI, cellGrpPrms->nCell*numUeSchd*sizeof(uint16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*numUeSchd);
            H5::DataSpace dataspaceSetSchdUe(1, dims);
            dataset = file.createDataSet("setSchdUePerCellTTI", H5::PredType::NATIVE_UINT16, dataspaceSetSchdUe);
            dataset.write(setSchdUePerCellTTI, H5::PredType::NATIVE_UINT16);

            delete[] setSchdUePerCellTTI;
        }

        if (schdSol->allocSol) {
            int16_t* allocSol;
            if (param.allocType == 1) { // type-1 allocation
                allocSol = new int16_t[2*numUe];
                CUDA_CHECK_ERR(cudaMemcpy(allocSol, schdSol->allocSol, 2*numUe*sizeof(int16_t), cudaMemcpyDeviceToHost));
                dims[0] = static_cast<hsize_t>(2*numUe);
            } else { // type-0 allocation
                allocSol = new int16_t[cellGrpPrms->nCell*cellGrpPrms->nPrbGrp];
                CUDA_CHECK_ERR(cudaMemcpy(allocSol, schdSol->allocSol, cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*sizeof(int16_t), cudaMemcpyDeviceToHost));
                dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp);
            }
             
            H5::DataSpace dataspaceAllocSol(1, dims);
            dataset = file.createDataSet("allocSol", H5::PredType::NATIVE_INT16, dataspaceAllocSol);
            dataset.write(allocSol, H5::PredType::NATIVE_INT16);

            delete[] allocSol;
        }

        if (schdSol->mcsSelSol) {
            int16_t*    mcsSelSol = new int16_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(mcsSelSol, schdSol->mcsSelSol, numUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceMcsSelSol(1, dims);
            dataset = file.createDataSet("mcsSelSol", H5::PredType::NATIVE_INT16, dataspaceMcsSelSol);
            dataset.write(mcsSelSol, H5::PredType::NATIVE_INT16);

            delete[] mcsSelSol;
        }

        if (schdSol->layerSelSol) {
            uint8_t*    layerSelSol = new uint8_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(layerSelSol, schdSol->layerSelSol, numUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceLayerSelSol(1, dims);
            dataset = file.createDataSet("layerSelSol", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSol);
            dataset.write(layerSelSol, H5::PredType::NATIVE_UINT8);

            delete[] layerSelSol;
        }

        if (schdSol->nSCID) {
            std::unique_ptr<uint8_t []> nSCID = std::make_unique<uint8_t []>(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(nSCID.get(), schdSol->nSCID, cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceNSCID(1, dims);
            dataset = file.createDataSet("nSCID", H5::PredType::NATIVE_UINT8, dataspaceNSCID);
            dataset.write(nSCID.get(), H5::PredType::NATIVE_UINT8);
        }
    }

    if (cellGrpPrms->prgMsk) {
        uint8_t** prgMsk = new uint8_t*[cellGrpPrms->nCell];
        uint8_t* perCellPrgMsk = new uint8_t[cellGrpPrms->nPrbGrp];

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nPrbGrp);
        CUDA_CHECK_ERR(cudaMemcpy(prgMsk, cellGrpPrms->prgMsk, cellGrpPrms->nCell*sizeof(uint8_t*), cudaMemcpyDeviceToHost));
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMemcpy(perCellPrgMsk, prgMsk[cIdx], cellGrpPrms->nPrbGrp*sizeof(uint8_t), cudaMemcpyDeviceToHost));
            std::string prgMskFieldName = "prgMsk" + std::to_string(cIdx);
            H5::DataSpace dataspacePrgMsk(1, dims);
            dataset = file.createDataSet(prgMskFieldName, H5::PredType::NATIVE_UINT8, dataspacePrgMsk);
            dataset.write(perCellPrgMsk, H5::PredType::NATIVE_UINT8);
        }

        delete[] prgMsk;
        delete[] perCellPrgMsk;
    }
}

void saveToH5(const std::string&                 filename,
              cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
              cumac::cumacCellGrpPrms*           cellGrpPrms,
              cumac::cumacSchdSol*               schdSol) 
{
    cumac::cumacSchedulerParam param;
        
    param.nUe = cellGrpPrms->nUe;
    param.nCell = cellGrpPrms->nCell;
    param.totNumCell = cellGrpPrms->nCell;
    param.nPrbGrp = cellGrpPrms->nPrbGrp;
    param.nBsAnt = cellGrpPrms->nBsAnt;
    param.nUeAnt = cellGrpPrms->nUeAnt;
    param.W  = cellGrpPrms->W;
    param.sigmaSqrd = cellGrpPrms->sigmaSqrd;
    param.betaCoeff = cellGrpPrms->betaCoeff;
    param.precodingScheme = cellGrpPrms->precodingScheme;
    param.receiverScheme = cellGrpPrms->receiverScheme;
    param.allocType = cellGrpPrms->allocType;
    param.columnMajor = 1;
    param.nActiveUe = cellGrpPrms->nActiveUe;
    param.numUeSchdPerCellTTI = cellGrpPrms->numUeSchdPerCellTTI;
    param.sinValThr = cellGrpPrms->sinValThr;
    param.numUeForGrpPerCell = cellGrpPrms->numUeForGrpPerCell;
    param.chanCorrThr = cellGrpPrms->chanCorrThr;
    param.muCoeff = cellGrpPrms->muCoeff;
    param.srsSnrThr = cellGrpPrms->srsSnrThr;
    param.nMaxActUePerCell = cellGrpPrms->nMaxActUePerCell;
    param.nMaxUegPerCellDl = cellGrpPrms->nMaxUegPerCellDl;
    param.nMaxUegPerCellUl = cellGrpPrms->nMaxUegPerCellUl;
    param.mcsSelSinrCapThr = cellGrpPrms->mcsSelSinrCapThr;
    param.muGrpSrsSnrMaxGap = cellGrpPrms->muGrpSrsSnrMaxGap;
    param.muGrpSrsSnrSplitThr = cellGrpPrms->muGrpSrsSnrSplitThr;
    param.bfPowAllocScheme = cellGrpPrms->bfPowAllocScheme;
    param.muGrpUpdate = cellGrpPrms->muGrpUpdate;
    param.mcsSelLutType = cellGrpPrms->mcsSelLutType;   
    param.semiStatFreqAlloc = cellGrpPrms->semiStatFreqAlloc;
    param.harqEnabledInd = cellGrpPrms->harqEnabledInd;
    param.mcsSelCqi = cellGrpPrms->mcsSelCqi;
    param.dlSchInd = cellGrpPrms->dlSchInd;

    uint8_t DL = param.dlSchInd;

    // Create a compound data type
    H5::CompType compType(sizeof(cumac::cumacSchedulerParam));
    compType.insertMember("nUe", HOFFSET(cumac::cumacSchedulerParam, nUe), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nCell", HOFFSET(cumac::cumacSchedulerParam, nCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("totNumCell", HOFFSET(cumac::cumacSchedulerParam, totNumCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nPrbGrp", HOFFSET(cumac::cumacSchedulerParam, nPrbGrp), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nBsAnt", HOFFSET(cumac::cumacSchedulerParam, nBsAnt), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nUeAnt", HOFFSET(cumac::cumacSchedulerParam, nUeAnt), H5::PredType::NATIVE_UINT8);
    compType.insertMember("W", HOFFSET(cumac::cumacSchedulerParam, W), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("sigmaSqrd", HOFFSET(cumac::cumacSchedulerParam, sigmaSqrd), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("betaCoeff", HOFFSET(cumac::cumacSchedulerParam, betaCoeff), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("precodingScheme", HOFFSET(cumac::cumacSchedulerParam, precodingScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("receiverScheme", HOFFSET(cumac::cumacSchedulerParam, receiverScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("allocType", HOFFSET(cumac::cumacSchedulerParam, allocType), H5::PredType::NATIVE_UINT8);
    compType.insertMember("columnMajor", HOFFSET(cumac::cumacSchedulerParam, columnMajor), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nActiveUe", HOFFSET(cumac::cumacSchedulerParam, nActiveUe), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nMaxActUePerCell", HOFFSET(cumac::cumacSchedulerParam, nMaxActUePerCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("numUeSchdPerCellTTI", HOFFSET(cumac::cumacSchedulerParam, numUeSchdPerCellTTI), H5::PredType::NATIVE_UINT8);
    compType.insertMember("sinValThr", HOFFSET(cumac::cumacSchedulerParam, sinValThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("numUeForGrpPerCell", HOFFSET(cumac::cumacSchedulerParam, numUeForGrpPerCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("chanCorrThr", HOFFSET(cumac::cumacSchedulerParam, chanCorrThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("muCoeff", HOFFSET(cumac::cumacSchedulerParam, muCoeff), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("srsSnrThr", HOFFSET(cumac::cumacSchedulerParam, srsSnrThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("nMaxUegPerCellDl", HOFFSET(cumac::cumacSchedulerParam, nMaxUegPerCellDl), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nMaxUegPerCellUl", HOFFSET(cumac::cumacSchedulerParam, nMaxUegPerCellUl), H5::PredType::NATIVE_UINT8);   
    compType.insertMember("mcsSelSinrCapThr", HOFFSET(cumac::cumacSchedulerParam, mcsSelSinrCapThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("muGrpSrsSnrMaxGap", HOFFSET(cumac::cumacSchedulerParam, muGrpSrsSnrMaxGap), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("muGrpSrsSnrSplitThr", HOFFSET(cumac::cumacSchedulerParam, muGrpSrsSnrSplitThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("bfPowAllocScheme", HOFFSET(cumac::cumacSchedulerParam, bfPowAllocScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("muGrpUpdate", HOFFSET(cumac::cumacSchedulerParam, muGrpUpdate), H5::PredType::NATIVE_UINT8);
    compType.insertMember("mcsSelLutType", HOFFSET(cumac::cumacSchedulerParam, mcsSelLutType), H5::PredType::NATIVE_UINT8); 
    compType.insertMember("semiStatFreqAlloc", HOFFSET(cumac::cumacSchedulerParam, semiStatFreqAlloc), H5::PredType::NATIVE_UINT8);   
    compType.insertMember("harqEnabledInd", HOFFSET(cumac::cumacSchedulerParam, harqEnabledInd), H5::PredType::NATIVE_UINT8);
    compType.insertMember("mcsSelCqi", HOFFSET(cumac::cumacSchedulerParam, mcsSelCqi), H5::PredType::NATIVE_UINT8);
    compType.insertMember("dlSchInd", HOFFSET(cumac::cumacSchedulerParam, dlSchInd), H5::PredType::NATIVE_UINT8);   

    // Open the HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Create a dataset
    H5::DataSet dataset = file.createDataSet("cumacSchedulerParam", compType, H5::DataSpace());

    // Write the data to the dataset
    dataset.write(&param, compType);

    uint16_t numUe;
    int numCfr = cellGrpPrms->nPrbGrp*cellGrpPrms->nUe*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt;
    int numCfrPerCell = cellGrpPrms->nCell*cellGrpPrms->numUeForGrpPerCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nBsAnt;

    if (cellGrpPrms->nBsAnt == 4) { // 4TR
        numUe = cellGrpPrms->nUe;
    } else if (cellGrpPrms->nBsAnt == 64) { // 64TR
        numUe = cellGrpPrms->nActiveUe;
    }

    hsize_t dims[] = {static_cast<hsize_t>(numCfr)};

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->estH_fr) {
        cuComplex*  estH_fr      = new cuComplex[numCfr];
        float*      estH_fr_real = new float[numCfr];
        float*      estH_fr_imag = new float[numCfr];

        CUDA_CHECK_ERR(cudaMemcpy(estH_fr, cellGrpPrms->estH_fr, numCfr*sizeof(cuComplex), cudaMemcpyDeviceToHost));
        
        for (int hIdx = 0; hIdx < numCfr; hIdx++) {
            estH_fr_real[hIdx] = estH_fr[hIdx].x;
            estH_fr_imag[hIdx] = estH_fr[hIdx].y;
        }

        H5::DataSpace dataspaceEstH_fr_real(1, dims);
        dataset = file.createDataSet("estH_fr_real", H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_real);
        dataset.write(estH_fr_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceEstH_fr_imag(1, dims);
        dataset = file.createDataSet("estH_fr_imag", H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_imag);
        dataset.write(estH_fr_imag, H5::PredType::NATIVE_FLOAT);

        delete[] estH_fr;
        delete[] estH_fr_real;
        delete[] estH_fr_imag;
    }

    if (cellGrpPrms->nBsAnt == 64 && cellGrpPrms->srsEstChan) {
        cuComplex** srsEstChan              = new cuComplex*[cellGrpPrms->nCell];
        cuComplex*  estH_fr_perCell    = new cuComplex[numCfrPerCell];
        float*      estH_fr_real            = new float[numCfrPerCell];
        float*      estH_fr_imag            = new float[numCfrPerCell];

        dims[0] = static_cast<hsize_t>(numCfrPerCell);

        CUDA_CHECK_ERR(cudaMemcpy(srsEstChan, cellGrpPrms->srsEstChan, cellGrpPrms->nCell*sizeof(cuComplex*), cudaMemcpyDeviceToHost));
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMemcpy(estH_fr_perCell, srsEstChan[cIdx], numCfrPerCell*sizeof(cuComplex), cudaMemcpyDeviceToHost));
            for (int hIdx = 0; hIdx < numCfrPerCell; hIdx++) {
                estH_fr_real[hIdx] = estH_fr_perCell[hIdx].x;
                estH_fr_imag[hIdx] = estH_fr_perCell[hIdx].y;
            }

            std::string cfrRealFieldName = "estH_fr_real_cell" + std::to_string(cIdx);
            H5::DataSpace dataspaceEstH_fr_real(1, dims);
            dataset = file.createDataSet(cfrRealFieldName, H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_real);
            dataset.write(estH_fr_real, H5::PredType::NATIVE_FLOAT);

            std::string cfrImagFieldName = "estH_fr_imag_cell" + std::to_string(cIdx);
            H5::DataSpace dataspaceEstH_fr_imag(1, dims);
            dataset = file.createDataSet(cfrImagFieldName, H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_imag);
            dataset.write(estH_fr_imag, H5::PredType::NATIVE_FLOAT);
        }

        delete[] srsEstChan;
        delete[] estH_fr_perCell;
        delete[] estH_fr_real;
        delete[] estH_fr_imag;
    }

    if (cellGrpPrms->srsWbSnr) {
        std::unique_ptr<float []>  srsWbSnr = std::make_unique<float []>(cellGrpPrms->nActiveUe);
        CUDA_CHECK_ERR(cudaMemcpy(srsWbSnr.get(), cellGrpPrms->srsWbSnr, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceSrsWbSnr(1, dims);
        dataset = file.createDataSet("srsWbSnr", H5::PredType::NATIVE_FLOAT, dataspaceSrsWbSnr);
        dataset.write(srsWbSnr.get(), H5::PredType::NATIVE_FLOAT);
    }

    if (cellGrpPrms->srsUeMap) {
        std::unique_ptr<int32_t* []> srsUeMap = std::make_unique<int32_t* []>(cellGrpPrms->nCell);
        std::unique_ptr<int32_t []> srsUeMap_perCell = std::make_unique<int32_t []>(cellGrpPrms->nActiveUe);
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);

        CUDA_CHECK_ERR(cudaMemcpy(srsUeMap.get(), cellGrpPrms->srsUeMap, cellGrpPrms->nCell*sizeof(int32_t*), cudaMemcpyDeviceToHost));
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMemcpy(srsUeMap_perCell.get(), srsUeMap[cIdx], cellGrpPrms->nActiveUe*sizeof(int32_t), cudaMemcpyDeviceToHost));
            
            std::string srsUeMapFieldName = "srsUeMap_cell" + std::to_string(cIdx);
            H5::DataSpace dataspaceSrsUeMap(1, dims);
            dataset = file.createDataSet(srsUeMapFieldName, H5::PredType::NATIVE_INT32, dataspaceSrsUeMap);
            dataset.write(srsUeMap_perCell.get(), H5::PredType::NATIVE_INT32);
        }
    }

    if (cellGrpPrms->sinVal) {
        float* sinVal = new float[numUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(sinVal, cellGrpPrms->sinVal, numUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost));
        
        dims[0] = static_cast<hsize_t>(numUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceSinVal(1, dims);
        dataset = file.createDataSet("sinVal", H5::PredType::NATIVE_FLOAT, dataspaceSinVal);
        dataset.write(sinVal, H5::PredType::NATIVE_FLOAT);

        delete[] sinVal;
    }
    
    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->prdMat) { // 4T4R SU-MIMO  
        int numPrdPerCell;
        if (DL == 1) { // DL
            numPrdPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        } else { // UL
            numPrdPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
        }

        cuComplex*  prdMat                  = new cuComplex[numPrdPerCell];
        float*      prdMat_real             = new float[numPrdPerCell];
        float*      prdMat_imag             = new float[numPrdPerCell];

        CUDA_CHECK_ERR(cudaMemcpy(prdMat, cellGrpPrms->prdMat, numPrdPerCell*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int hIdx = 0; hIdx < numPrdPerCell; hIdx++) {
            prdMat_real[hIdx] = prdMat[hIdx].x;
            prdMat_imag[hIdx] = prdMat[hIdx].y;
        }

        dims[0] = static_cast<hsize_t>(numPrdPerCell);
        H5::DataSpace dataspacePrdMat_real(1, dims);
        dataset = file.createDataSet("prdMat_real", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_real);
        dataset.write(prdMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspacePrdMat_imag(1, dims);
        dataset = file.createDataSet("prdMat_imag", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_imag);
        dataset.write(prdMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] prdMat;
        delete[] prdMat_real;
        delete[] prdMat_imag;
    }

    if (cellGrpPrms->nBsAnt == 64 && cellGrpPrms->prdMat) {// 64T64R MU-MIMO
        std::vector<cuComplex> prdMat(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);
        std::vector<float> prdMat_real(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);
        std::vector<float> prdMat_imag(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);

        CUDA_CHECK_ERR(cudaMemcpy(prdMat.data(), cellGrpPrms->prdMat, cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_*sizeof(cuComplex), cudaMemcpyDeviceToHost));
        for (int hIdx = 0; hIdx < cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_; hIdx++) {
            prdMat_real[hIdx] = prdMat[hIdx].x;
            prdMat_imag[hIdx] = prdMat[hIdx].y;
        }
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cumac::maxNumLayerPerGrpDL_);
        H5::DataSpace dataspacePrdMat_real(1, dims);
        dataset = file.createDataSet("prdMat_real", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_real);
        dataset.write(prdMat_real.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspacePrdMat_imag(1, dims);
        dataset = file.createDataSet("prdMat_imag", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_imag);
        dataset.write(prdMat_imag.data(), H5::PredType::NATIVE_FLOAT); 
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->detMat) {
        int numDetPerCell;
        if (DL == 1) { // DL
            numDetPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
        } else { // UL
            numDetPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        }
         
        cuComplex*  detMat      = new cuComplex[numDetPerCell]; 
        float*      detMat_real = new float[numDetPerCell];
        float*      detMat_imag = new float[numDetPerCell];

        CUDA_CHECK_ERR(cudaMemcpy(detMat, cellGrpPrms->detMat, numDetPerCell*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int hIdx = 0; hIdx < numDetPerCell; hIdx++) {
            detMat_real[hIdx] = detMat[hIdx].x;
            detMat_imag[hIdx] = detMat[hIdx].y;
        }

        dims[0] = static_cast<hsize_t>(numDetPerCell);
        H5::DataSpace dataspaceDetMat_real(1, dims);
        dataset = file.createDataSet("detMat_real", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_real);
        dataset.write(detMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceDetMat_imag(1, dims);
        dataset = file.createDataSet("detMat_imag", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_imag);
        dataset.write(detMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] detMat;
        delete[] detMat_real;
        delete[] detMat_imag;
    }

    if (cellGrpPrms->currSlotIdxPerCell) {
        std::vector<uint32_t> currSlotIdxPerCell(cellGrpPrms->nCell);
        CUDA_CHECK_ERR(cudaMemcpy(currSlotIdxPerCell.data(), cellGrpPrms->currSlotIdxPerCell, cellGrpPrms->nCell*sizeof(uint32_t), cudaMemcpyDeviceToHost));
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell);
        H5::DataSpace dataspaceCurrSlotIdxPerCell(1, dims);
        dataset = file.createDataSet("currSlotIdxPerCell", H5::PredType::NATIVE_UINT32, dataspaceCurrSlotIdxPerCell);
        dataset.write(currSlotIdxPerCell.data(), H5::PredType::NATIVE_UINT32);  
    }

    if (cellGrpPrms->blerTargetActUe) {
        std::vector<float> blerTargetActUe(cellGrpPrms->nActiveUe);
        CUDA_CHECK_ERR(cudaMemcpy(blerTargetActUe.data(), cellGrpPrms->blerTargetActUe, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceBlerTargetActUe(1, dims);
        dataset = file.createDataSet("blerTargetActUe", H5::PredType::NATIVE_FLOAT, dataspaceBlerTargetActUe);
        dataset.write(blerTargetActUe.data(), H5::PredType::NATIVE_FLOAT);
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->cellId) {
        uint16_t* cellId = new uint16_t[cellGrpPrms->nCell];
        CUDA_CHECK_ERR(cudaMemcpy(cellId, cellGrpPrms->cellId, cellGrpPrms->nCell*sizeof(uint16_t), cudaMemcpyDeviceToHost));

        dims[0] = {static_cast<hsize_t>(cellGrpPrms->nCell)};
        H5::DataSpace dataspaceCellId(1, dims);
        dataset = file.createDataSet("cellId", H5::PredType::NATIVE_UINT16, dataspaceCellId);
        dataset.write(cellId, H5::PredType::NATIVE_UINT16);

        delete[] cellId;
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->numUeSchdPerCellTTIArr) {
        uint8_t* numUeSchdPerCellTTIArr = new uint8_t[cellGrpPrms->nCell];
        CUDA_CHECK_ERR(cudaMemcpy(numUeSchdPerCellTTIArr, cellGrpPrms->numUeSchdPerCellTTIArr, cellGrpPrms->nCell*sizeof(uint8_t), cudaMemcpyDeviceToHost));

        dims[0] = {static_cast<hsize_t>(cellGrpPrms->nCell)};
        H5::DataSpace dataspaceNumUeSchdArr(1, dims);
        dataset = file.createDataSet("numUeSchdPerCellTTIArr", H5::PredType::NATIVE_UINT8, dataspaceNumUeSchdArr);
        dataset.write(numUeSchdPerCellTTIArr, H5::PredType::NATIVE_UINT8);

        delete[] numUeSchdPerCellTTIArr;
    }
    
    if (cellGrpUeStatus) {
        if (cellGrpUeStatus->lastSchdSlotActUe) {
            std::vector<uint32_t> lastSchdSlotActUe(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(lastSchdSlotActUe.data(), cellGrpUeStatus->lastSchdSlotActUe, cellGrpPrms->nActiveUe*sizeof(uint32_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceLastSchdSlotActUe(1, dims);
            dataset = file.createDataSet("lastSchdSlotActUe", H5::PredType::NATIVE_UINT32, dataspaceLastSchdSlotActUe);
            dataset.write(lastSchdSlotActUe.data(), H5::PredType::NATIVE_UINT32);   
        }   

        if (cellGrpUeStatus->beamformGainCurrTx) {
            std::vector<float> beamformGainCurrTx(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(beamformGainCurrTx.data(), cellGrpUeStatus->beamformGainCurrTx, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));
            
            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceBeamformGainCurrTx(1, dims);
            dataset = file.createDataSet("beamformGainCurrTx", H5::PredType::NATIVE_FLOAT, dataspaceBeamformGainCurrTx);
            dataset.write(beamformGainCurrTx.data(), H5::PredType::NATIVE_FLOAT);
        }

        if (cellGrpUeStatus->bfGainPrgCurrTx) { 
            std::vector<float> bfGainPrgCurrTx(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp);
            CUDA_CHECK_ERR(cudaMemcpy(bfGainPrgCurrTx.data(), cellGrpUeStatus->bfGainPrgCurrTx, cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp);
            H5::DataSpace dataspaceBfGainPrgCurrTx(1, dims);
            dataset = file.createDataSet("bfGainPrgCurrTx", H5::PredType::NATIVE_FLOAT, dataspaceBfGainPrgCurrTx);
            dataset.write(bfGainPrgCurrTx.data(), H5::PredType::NATIVE_FLOAT);
        }   

        if (cellGrpUeStatus->beamformGainLastTx) {
            std::vector<float> beamformGainLastTx(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(beamformGainLastTx.data(), cellGrpUeStatus->beamformGainLastTx, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceBeamformGainLastTx(1, dims); 
            dataset = file.createDataSet("beamformGainLastTx", H5::PredType::NATIVE_FLOAT, dataspaceBeamformGainLastTx);
            dataset.write(beamformGainLastTx.data(), H5::PredType::NATIVE_FLOAT);   
        }

        if (cellGrpPrms->nBsAnt == 4 && cellGrpUeStatus->avgRates) {
            float* avgRates = new float[cellGrpPrms->nUe];
            CUDA_CHECK_ERR(cudaMemcpy(avgRates, cellGrpUeStatus->avgRates, cellGrpPrms->nUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceAvgRates(1, dims);
            dataset = file.createDataSet("avgRates", H5::PredType::NATIVE_FLOAT, dataspaceAvgRates);
            dataset.write(avgRates, H5::PredType::NATIVE_FLOAT);

            delete[] avgRates;
        }

        if (cellGrpUeStatus->avgRatesActUe) {
            float* avgRatesActUe = new float[cellGrpPrms->nActiveUe];
            CUDA_CHECK_ERR(cudaMemcpy(avgRatesActUe, cellGrpUeStatus->avgRatesActUe, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceAvgRatesActUe(1, dims);
            dataset = file.createDataSet("avgRatesActUe", H5::PredType::NATIVE_FLOAT, dataspaceAvgRatesActUe);
            dataset.write(avgRatesActUe, H5::PredType::NATIVE_FLOAT);

            delete[] avgRatesActUe;
        }

        if (cellGrpPrms->nBsAnt == 4 && cellGrpUeStatus->tbErrLast) {
            int8_t* tbErrLast = new int8_t[cellGrpPrms->nUe];
            CUDA_CHECK_ERR(cudaMemcpy(tbErrLast, cellGrpUeStatus->tbErrLast, cellGrpPrms->nUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceTbErrLast(1, dims);
            dataset = file.createDataSet("tbErrLast", H5::PredType::NATIVE_INT8, dataspaceTbErrLast);
            dataset.write(tbErrLast, H5::PredType::NATIVE_INT8);

            delete[] tbErrLast;
        }

        if (cellGrpUeStatus->tbErrLastActUe) {
            int8_t*     tbErrLastActUe = new int8_t[cellGrpPrms->nActiveUe];
            CUDA_CHECK_ERR(cudaMemcpy(tbErrLastActUe, cellGrpUeStatus->tbErrLastActUe, cellGrpPrms->nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceTbErrLastActUe(1, dims);
            dataset = file.createDataSet("tbErrLastActUe", H5::PredType::NATIVE_INT8, dataspaceTbErrLastActUe);
            dataset.write(tbErrLastActUe, H5::PredType::NATIVE_INT8);

            delete[] tbErrLastActUe;
        }

        if (cellGrpUeStatus->prioWeightActUe) {
            uint16_t* prioWeightActUe = new uint16_t[cellGrpPrms->nActiveUe];
            CUDA_CHECK_ERR(cudaMemcpy(prioWeightActUe, cellGrpUeStatus->prioWeightActUe, cellGrpPrms->nActiveUe*sizeof(uint16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspacePwActUe(1, dims);
            dataset = file.createDataSet("prioWeightActUe", H5::PredType::NATIVE_UINT16, dataspacePwActUe);
            dataset.write(prioWeightActUe, H5::PredType::NATIVE_UINT16);

            delete[] prioWeightActUe;
        }

        if (cellGrpUeStatus->newDataActUe) {
            int8_t* newDataActUe = new int8_t[cellGrpPrms->nActiveUe]; 
            CUDA_CHECK_ERR(cudaMemcpy(newDataActUe, cellGrpUeStatus->newDataActUe, cellGrpPrms->nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceNdActUe(1, dims);
            dataset = file.createDataSet("newDataActUe", H5::PredType::NATIVE_INT8, dataspaceNdActUe);
            dataset.write(newDataActUe, H5::PredType::NATIVE_INT8);

            delete[] newDataActUe;
        }

        if (cellGrpUeStatus->allocSolLastTx) {
            int16_t* allocSolLastTx = new int16_t[2*numUe];
            CUDA_CHECK_ERR(cudaMemcpy(allocSolLastTx, cellGrpUeStatus->allocSolLastTx, 2*numUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(2*numUe);
            H5::DataSpace dataspaceAllocSolLastTx(1, dims);
            dataset = file.createDataSet("allocSolLastTx", H5::PredType::NATIVE_INT16, dataspaceAllocSolLastTx);
            dataset.write(allocSolLastTx, H5::PredType::NATIVE_INT16);

            delete[] allocSolLastTx;
        }

        if (cellGrpUeStatus->mcsSelSolLastTx) {
            int16_t* mcsSelSolLastTx = new int16_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(mcsSelSolLastTx, cellGrpUeStatus->mcsSelSolLastTx, numUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceMcsSelSolLastTx(1, dims);
            dataset = file.createDataSet("mcsSelSolLastTx", H5::PredType::NATIVE_INT16, dataspaceMcsSelSolLastTx);
            dataset.write(mcsSelSolLastTx, H5::PredType::NATIVE_INT16);

            delete[] mcsSelSolLastTx;
        }

        if (cellGrpUeStatus->layerSelSolLastTx) {
            uint8_t* layerSelSolLastTx = new uint8_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(layerSelSolLastTx, cellGrpUeStatus->layerSelSolLastTx, numUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceLayerSelSolLastTx(1, dims);
            dataset = file.createDataSet("layerSelSolLastTx", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSolLastTx);
            dataset.write(layerSelSolLastTx, H5::PredType::NATIVE_UINT8);

            delete[] layerSelSolLastTx;
        }
    }

    if (cellGrpPrms->nBsAnt == 4 && cellGrpPrms->cellAssoc) {
        uint8_t* cellAssoc = new uint8_t[cellGrpPrms->nCell*cellGrpPrms->nUe];
        CUDA_CHECK_ERR(cudaMemcpy(cellAssoc, cellGrpPrms->cellAssoc, cellGrpPrms->nCell*cellGrpPrms->nUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nUe);
        H5::DataSpace dataspaceCellAssoc(1, dims);
        dataset = file.createDataSet("cellAssoc", H5::PredType::NATIVE_UINT8, dataspaceCellAssoc);
        dataset.write(cellAssoc, H5::PredType::NATIVE_UINT8);

        delete[] cellAssoc;
    }

    if (cellGrpPrms->cellAssocActUe) {
        uint8_t* cellAssocActUe = new uint8_t[cellGrpPrms->nCell*cellGrpPrms->nActiveUe];
        CUDA_CHECK_ERR(cudaMemcpy(cellAssocActUe, cellGrpPrms->cellAssocActUe, cellGrpPrms->nCell*cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceCellAssocActUe(1, dims);
        dataset = file.createDataSet("cellAssocActUe", H5::PredType::NATIVE_UINT8, dataspaceCellAssocActUe);
        dataset.write(cellAssocActUe, H5::PredType::NATIVE_UINT8);

        delete[] cellAssocActUe;
    }

    if (cellGrpPrms->postEqSinr) {
        float* postEqSinr = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(postEqSinr, cellGrpPrms->postEqSinr, cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspacePostEqSinr(1, dims);
        dataset = file.createDataSet("postEqSinr", H5::PredType::NATIVE_FLOAT, dataspacePostEqSinr);
        dataset.write(postEqSinr, H5::PredType::NATIVE_FLOAT);

        delete[] postEqSinr;
    }

    if (cellGrpPrms->wbSinr) {
        float*      wbSinr = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt];
        CUDA_CHECK_ERR(cudaMemcpy(wbSinr, cellGrpPrms->wbSinr, cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost)); 

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceWbSinr(1, dims);
        dataset = file.createDataSet("wbSinr", H5::PredType::NATIVE_FLOAT, dataspaceWbSinr);
        dataset.write(wbSinr, H5::PredType::NATIVE_FLOAT);

        delete[] wbSinr;
    }
    
    if (schdSol) {
        if (cellGrpPrms->nBsAnt == 64 && schdSol->muGrpList) {
            auto muGrpListGpu           = std::make_unique<cumac::multiCellMuGrpList>();
            CUDA_CHECK_ERR(cudaMemcpy(muGrpListGpu.get(), schdSol->muGrpList, sizeof(cumac::multiCellMuGrpList), cudaMemcpyDeviceToHost));

            std::vector<uint16_t> numUeInGrp(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            CUDA_CHECK_ERR(cudaMemcpy(numUeInGrp.data(), muGrpListGpu->numUeInGrp, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(uint16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            H5::DataSpace dataspaceNumUeInGrp(1, dims);
            dataset = file.createDataSet("numUeInGrp", H5::PredType::NATIVE_UINT16, dataspaceNumUeInGrp);
            dataset.write(numUeInGrp.data(), H5::PredType::NATIVE_UINT16);  

            std::vector<uint16_t> ueId(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_);
            CUDA_CHECK_ERR(cudaMemcpy(ueId.data(), muGrpListGpu->ueId, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_*sizeof(uint16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_);
            H5::DataSpace dataspaceUeId(1, dims);
            dataset = file.createDataSet("ueId", H5::PredType::NATIVE_UINT16, dataspaceUeId);
            dataset.write(ueId.data(), H5::PredType::NATIVE_UINT16);

            std::vector<int16_t> subbandId(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            CUDA_CHECK_ERR(cudaMemcpy(subbandId.data(), muGrpListGpu->subbandId, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(int16_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_);
            H5::DataSpace dataspaceSubbandId(1, dims);
            dataset = file.createDataSet("subbandId", H5::PredType::NATIVE_INT16, dataspaceSubbandId);
            dataset.write(subbandId.data(), H5::PredType::NATIVE_INT16);    
        }

        if (schdSol->ueOrderInGrp) {
            std::vector<uint16_t> ueOrderInGrp(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(ueOrderInGrp.data(), schdSol->ueOrderInGrp, cellGrpPrms->nActiveUe*sizeof(uint16_t), cudaMemcpyDeviceToHost));    

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceUeOrderInGrp(1, dims);
            dataset = file.createDataSet("ueOrderInGrp", H5::PredType::NATIVE_UINT16, dataspaceUeOrderInGrp);
            dataset.write(ueOrderInGrp.data(), H5::PredType::NATIVE_UINT16);
        }

        if (schdSol->muMimoInd) {
            std::unique_ptr<uint8_t []> muMimoInd = std::make_unique<uint8_t []>(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(muMimoInd.get(), schdSol->muMimoInd, cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost));

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceMuMimoInd(1, dims);
            dataset = file.createDataSet("muMimoInd", H5::PredType::NATIVE_UINT8, dataspaceMuMimoInd);
            dataset.write(muMimoInd.get(), H5::PredType::NATIVE_UINT8);
        }

        if (schdSol->sortedUeList) {
            std::unique_ptr<uint16_t* []> sortedUeList  = std::make_unique<uint16_t* []>(cellGrpPrms->nCell);
            std::unique_ptr<uint16_t []> sortedUeList_perCell = std::make_unique<uint16_t []>(cellGrpPrms->nMaxActUePerCell);

            CUDA_CHECK_ERR(cudaMemcpy(sortedUeList.get(), schdSol->sortedUeList, cellGrpPrms->nCell*sizeof(uint16_t*), cudaMemcpyDeviceToHost));
            dims[0] = static_cast<hsize_t>(cellGrpPrms->nMaxActUePerCell);
            for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
                CUDA_CHECK_ERR(cudaMemcpy(sortedUeList_perCell.get(), sortedUeList[cIdx], cellGrpPrms->nMaxActUePerCell*sizeof(uint16_t), cudaMemcpyDeviceToHost));

                std::string sortedUeListFieldName = "sortedUeList_cell" + std::to_string(cIdx);
                H5::DataSpace dataspaceSortedUeList(1, dims);
                dataset = file.createDataSet(sortedUeListFieldName, H5::PredType::NATIVE_UINT16, dataspaceSortedUeList);
                dataset.write(sortedUeList_perCell.get(), H5::PredType::NATIVE_UINT16);
            }
        }

        if (schdSol->setSchdUePerCellTTI) {
            int numUeSchd;
            if (cellGrpPrms->nBsAnt == 4) { // 4TR
                numUeSchd = cellGrpPrms->numUeSchdPerCellTTI;
            } else { // 64TR
                numUeSchd = cellGrpPrms->numUeForGrpPerCell;
            }

            uint16_t* setSchdUePerCellTTI = new uint16_t[cellGrpPrms->nCell*numUeSchd];
            CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTI, schdSol->setSchdUePerCellTTI, cellGrpPrms->nCell*numUeSchd*sizeof(uint16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*numUeSchd);
            H5::DataSpace dataspaceSetSchdUe(1, dims);
            dataset = file.createDataSet("setSchdUePerCellTTI", H5::PredType::NATIVE_UINT16, dataspaceSetSchdUe);
            dataset.write(setSchdUePerCellTTI, H5::PredType::NATIVE_UINT16);

            delete[] setSchdUePerCellTTI;
        }

        if (schdSol->allocSol) {
            int16_t* allocSol;
            if (param.allocType == 1) { // type-1 allocation
                allocSol = new int16_t[2*numUe];
                CUDA_CHECK_ERR(cudaMemcpy(allocSol, schdSol->allocSol, 2*numUe*sizeof(int16_t), cudaMemcpyDeviceToHost));
                dims[0] = static_cast<hsize_t>(2*numUe);
            } else { // type-0 allocation
                allocSol = new int16_t[cellGrpPrms->nCell*cellGrpPrms->nPrbGrp];
                CUDA_CHECK_ERR(cudaMemcpy(allocSol, schdSol->allocSol, cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*sizeof(int16_t), cudaMemcpyDeviceToHost));
                dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp);
            }

            H5::DataSpace dataspaceAllocSol(1, dims);
            dataset = file.createDataSet("allocSol", H5::PredType::NATIVE_INT16, dataspaceAllocSol);
            dataset.write(allocSol, H5::PredType::NATIVE_INT16);

            delete[] allocSol;
        }

        if (schdSol->mcsSelSol) {
            int16_t*    mcsSelSol = new int16_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(mcsSelSol, schdSol->mcsSelSol, numUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceMcsSelSol(1, dims);
            dataset = file.createDataSet("mcsSelSol", H5::PredType::NATIVE_INT16, dataspaceMcsSelSol);
            dataset.write(mcsSelSol, H5::PredType::NATIVE_INT16);

            delete[] mcsSelSol;
        }

        if (schdSol->layerSelSol) {
            uint8_t*    layerSelSol = new uint8_t[numUe];
            CUDA_CHECK_ERR(cudaMemcpy(layerSelSol, schdSol->layerSelSol, numUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(numUe);
            H5::DataSpace dataspaceLayerSelSol(1, dims);
            dataset = file.createDataSet("layerSelSol", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSol);
            dataset.write(layerSelSol, H5::PredType::NATIVE_UINT8);

            delete[] layerSelSol;
        }

        if (schdSol->nSCID) {
            std::unique_ptr<uint8_t []> nSCID = std::make_unique<uint8_t []>(cellGrpPrms->nActiveUe);
            CUDA_CHECK_ERR(cudaMemcpy(nSCID.get(), schdSol->nSCID, cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceNSCID(1, dims);
            dataset = file.createDataSet("nSCID", H5::PredType::NATIVE_UINT8, dataspaceNSCID);
            dataset.write(nSCID.get(), H5::PredType::NATIVE_UINT8);
        }
    }
    

    if (cellGrpPrms->prgMsk) {
        uint8_t** prgMsk = new uint8_t*[cellGrpPrms->nCell];
        uint8_t* perCellPrgMsk = new uint8_t[cellGrpPrms->nPrbGrp];

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nPrbGrp);
        CUDA_CHECK_ERR(cudaMemcpy(prgMsk, cellGrpPrms->prgMsk, cellGrpPrms->nCell*sizeof(uint8_t*), cudaMemcpyDeviceToHost));
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            CUDA_CHECK_ERR(cudaMemcpy(perCellPrgMsk, prgMsk[cIdx], cellGrpPrms->nPrbGrp*sizeof(uint8_t), cudaMemcpyDeviceToHost));
            std::string prgMskFieldName = "prgMsk" + std::to_string(cIdx);
            H5::DataSpace dataspacePrgMsk(1, dims);
            dataset = file.createDataSet(prgMskFieldName, H5::PredType::NATIVE_UINT8, dataspacePrgMsk);
            dataset.write(perCellPrgMsk, H5::PredType::NATIVE_UINT8);
        }

        delete[] prgMsk;
        delete[] perCellPrgMsk;
    }
}

void saveToH5_CPU(const std::string&             filename,
                  cumac::cumacCellGrpUeStatus*   cellGrpUeStatus,
                  cumac::cumacCellGrpPrms*       cellGrpPrms,
                  cumac::cumacSchdSol*           schdSol) {

    cumac::cumacSchedulerParam param;
        
    param.nUe = cellGrpPrms->nUe;
    param.nCell = cellGrpPrms->nCell;
    param.totNumCell = cellGrpPrms->nCell;
    param.nPrbGrp = cellGrpPrms->nPrbGrp;
    param.nBsAnt = cellGrpPrms->nBsAnt;
    param.nUeAnt = cellGrpPrms->nUeAnt;
    param.W  = cellGrpPrms->W;
    param.sigmaSqrd = cellGrpPrms->sigmaSqrd;
    param.betaCoeff = cellGrpPrms->betaCoeff;
    param.precodingScheme = cellGrpPrms->precodingScheme;
    param.receiverScheme = cellGrpPrms->receiverScheme;
    param.allocType = cellGrpPrms->allocType;
    param.columnMajor = 1;
    param.nActiveUe = cellGrpPrms->nActiveUe;
    param.numUeSchdPerCellTTI = cellGrpPrms->numUeSchdPerCellTTI;
    param.sinValThr = cellGrpPrms->sinValThr;
    param.harqEnabledInd = cellGrpPrms->harqEnabledInd;
    param.mcsSelCqi = cellGrpPrms->mcsSelCqi;
    param.dlSchInd = cellGrpPrms->dlSchInd;

    uint8_t DL = param.dlSchInd;

    // Create a compound data type
    H5::CompType compType(sizeof(cumac::cumacSchedulerParam));
    compType.insertMember("nUe", HOFFSET(cumac::cumacSchedulerParam, nUe), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nCell", HOFFSET(cumac::cumacSchedulerParam, nCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("totNumCell", HOFFSET(cumac::cumacSchedulerParam, totNumCell), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nPrbGrp", HOFFSET(cumac::cumacSchedulerParam, nPrbGrp), H5::PredType::NATIVE_UINT16);
    compType.insertMember("nBsAnt", HOFFSET(cumac::cumacSchedulerParam, nBsAnt), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nUeAnt", HOFFSET(cumac::cumacSchedulerParam, nUeAnt), H5::PredType::NATIVE_UINT8);
    compType.insertMember("W", HOFFSET(cumac::cumacSchedulerParam, W), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("sigmaSqrd", HOFFSET(cumac::cumacSchedulerParam, sigmaSqrd), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("betaCoeff", HOFFSET(cumac::cumacSchedulerParam, betaCoeff), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("precodingScheme", HOFFSET(cumac::cumacSchedulerParam, precodingScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("receiverScheme", HOFFSET(cumac::cumacSchedulerParam, receiverScheme), H5::PredType::NATIVE_UINT8);
    compType.insertMember("allocType", HOFFSET(cumac::cumacSchedulerParam, allocType), H5::PredType::NATIVE_UINT8);
    compType.insertMember("columnMajor", HOFFSET(cumac::cumacSchedulerParam, columnMajor), H5::PredType::NATIVE_UINT8);
    compType.insertMember("nActiveUe", HOFFSET(cumac::cumacSchedulerParam, nActiveUe), H5::PredType::NATIVE_UINT16);
    compType.insertMember("numUeSchdPerCellTTI", HOFFSET(cumac::cumacSchedulerParam, numUeSchdPerCellTTI), H5::PredType::NATIVE_UINT8);
    compType.insertMember("sinValThr", HOFFSET(cumac::cumacSchedulerParam, sinValThr), H5::PredType::NATIVE_FLOAT);
    compType.insertMember("harqEnabledInd", HOFFSET(cumac::cumacSchedulerParam, harqEnabledInd), H5::PredType::NATIVE_UINT8);
    compType.insertMember("mcsSelCqi", HOFFSET(cumac::cumacSchedulerParam, mcsSelCqi), H5::PredType::NATIVE_UINT8);
    compType.insertMember("dlSchInd", HOFFSET(cumac::cumacSchedulerParam, dlSchInd), H5::PredType::NATIVE_UINT8);   
    
    // Open the HDF5 file
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Create a dataset
    H5::DataSet dataset = file.createDataSet("cumacSchedulerParam", compType, H5::DataSpace());

    // Write the data to the dataset
    dataset.write(&param, compType);

    int numCfr = cellGrpPrms->nPrbGrp*cellGrpPrms->nUe*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt;
    hsize_t dims[] = {static_cast<hsize_t>(numCfr)};

    if (cellGrpPrms->estH_fr) {
        cuComplex*  estH_fr      = cellGrpPrms->estH_fr;
        float*      estH_fr_real = new float[numCfr];
        float*      estH_fr_imag = new float[numCfr];
        
        for (int hIdx = 0; hIdx < numCfr; hIdx++) {
            estH_fr_real[hIdx] = estH_fr[hIdx].x;
            estH_fr_imag[hIdx] = estH_fr[hIdx].y;
        }

        H5::DataSpace dataspaceEstH_fr_real(1, dims);
        dataset = file.createDataSet("estH_fr_real", H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_real);
        dataset.write(estH_fr_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceEstH_fr_imag(1, dims);
        dataset = file.createDataSet("estH_fr_imag", H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_imag);
        dataset.write(estH_fr_imag, H5::PredType::NATIVE_FLOAT);

        delete[] estH_fr_real;
        delete[] estH_fr_imag;
    }

    if (cellGrpPrms->sinVal) {
        float* sinVal = cellGrpPrms->sinVal;
        
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceSinVal(1, dims);
        dataset = file.createDataSet("sinVal", H5::PredType::NATIVE_FLOAT, dataspaceSinVal);
        dataset.write(sinVal, H5::PredType::NATIVE_FLOAT);
    }
    
    if (cellGrpPrms->prdMat) {
        int numPrdPerCell;
        if (DL == 1) { // DL
            numPrdPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        } else { // UL
            numPrdPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
        }

        cuComplex*  prdMat                  = cellGrpPrms->prdMat;
        float*      prdMat_real             = new float[numPrdPerCell];
        float*      prdMat_imag             = new float[numPrdPerCell];

        for (int hIdx = 0; hIdx < numPrdPerCell; hIdx++) {
            prdMat_real[hIdx] = prdMat[hIdx].x;
            prdMat_imag[hIdx] = prdMat[hIdx].y;
        }

        dims[0] = static_cast<hsize_t>(numPrdPerCell);
        H5::DataSpace dataspacePrdMat_real(1, dims);
        dataset = file.createDataSet("prdMat_real", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_real);
        dataset.write(prdMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspacePrdMat_imag(1, dims);
        dataset = file.createDataSet("prdMat_imag", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_imag);
        dataset.write(prdMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] prdMat_real;
        delete[] prdMat_imag;
    }

    if (cellGrpPrms->detMat) {
        int numDetPerCell;
        if (DL == 1) { // DL
            numDetPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
        } else { // UL
            numDetPerCell = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        }
         
        cuComplex*  detMat      = cellGrpPrms->detMat; 
        float*      detMat_real = new float[numDetPerCell];
        float*      detMat_imag = new float[numDetPerCell];

        for (int hIdx = 0; hIdx < numDetPerCell; hIdx++) {
            detMat_real[hIdx] = detMat[hIdx].x;
            detMat_imag[hIdx] = detMat[hIdx].y;
        }

        dims[0] = static_cast<hsize_t>(numDetPerCell);
        H5::DataSpace dataspaceDetMat_real(1, dims);
        dataset = file.createDataSet("detMat_real", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_real);
        dataset.write(detMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceDetMat_imag(1, dims);
        dataset = file.createDataSet("detMat_imag", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_imag);
        dataset.write(detMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] detMat_real;
        delete[] detMat_imag;
    }

    if (cellGrpPrms->cellId) {
        uint16_t* cellId = cellGrpPrms->cellId;

        dims[0] = {static_cast<hsize_t>(cellGrpPrms->nCell)};
        H5::DataSpace dataspaceCellId(1, dims);
        dataset = file.createDataSet("cellId", H5::PredType::NATIVE_UINT16, dataspaceCellId);
        dataset.write(cellId, H5::PredType::NATIVE_UINT16);
    }

    if (cellGrpPrms->numUeSchdPerCellTTIArr) {
        uint8_t* numUeSchdPerCellTTIArr = cellGrpPrms->numUeSchdPerCellTTIArr;

        dims[0] = {static_cast<hsize_t>(cellGrpPrms->nCell)};
        H5::DataSpace dataspaceNumUeSchdArr(1, dims);
        dataset = file.createDataSet("numUeSchdPerCellTTIArr", H5::PredType::NATIVE_UINT8, dataspaceNumUeSchdArr);
        dataset.write(numUeSchdPerCellTTIArr, H5::PredType::NATIVE_UINT8);
    }
    
    if (cellGrpUeStatus) {
        if (cellGrpUeStatus->avgRates) {
            float* avgRates = cellGrpUeStatus->avgRates;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceAvgRates(1, dims);
            dataset = file.createDataSet("avgRates", H5::PredType::NATIVE_FLOAT, dataspaceAvgRates);
            dataset.write(avgRates, H5::PredType::NATIVE_FLOAT);
        }

        if (cellGrpUeStatus->avgRatesActUe) {
            float* avgRatesActUe = cellGrpUeStatus->avgRatesActUe;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceAvgRatesActUe(1, dims);
            dataset = file.createDataSet("avgRatesActUe", H5::PredType::NATIVE_FLOAT, dataspaceAvgRatesActUe);
            dataset.write(avgRatesActUe, H5::PredType::NATIVE_FLOAT);
        }

        if (cellGrpUeStatus->tbErrLast) {
            int8_t* tbErrLast = cellGrpUeStatus->tbErrLast;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceTbErrLast(1, dims);
            dataset = file.createDataSet("tbErrLast", H5::PredType::NATIVE_INT8, dataspaceTbErrLast);
            dataset.write(tbErrLast, H5::PredType::NATIVE_INT8);
        }

        if (cellGrpUeStatus->tbErrLastActUe) {
            int8_t*     tbErrLastActUe = cellGrpUeStatus->tbErrLastActUe;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceTbErrLastActUe(1, dims);
            dataset = file.createDataSet("tbErrLastActUe", H5::PredType::NATIVE_INT8, dataspaceTbErrLastActUe);
            dataset.write(tbErrLastActUe, H5::PredType::NATIVE_INT8);
        }

        if (cellGrpUeStatus->newDataActUe) {
            int8_t* newDataActUe = cellGrpUeStatus->newDataActUe; 

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe);
            H5::DataSpace dataspaceNdActUe(1, dims);
            dataset = file.createDataSet("newDataActUe", H5::PredType::NATIVE_INT8, dataspaceNdActUe);
            dataset.write(newDataActUe, H5::PredType::NATIVE_INT8);
        }

        if (cellGrpUeStatus->allocSolLastTx) {
            int16_t* allocSolLastTx = cellGrpUeStatus->allocSolLastTx;

            dims[0] = static_cast<hsize_t>(2*cellGrpPrms->nUe);
            H5::DataSpace dataspaceAllocSolLastTx(1, dims);
            dataset = file.createDataSet("allocSolLastTx", H5::PredType::NATIVE_INT16, dataspaceAllocSolLastTx);
            dataset.write(allocSolLastTx, H5::PredType::NATIVE_INT16);
        }

        if (cellGrpUeStatus->mcsSelSolLastTx) {
            int16_t* mcsSelSolLastTx = cellGrpUeStatus->mcsSelSolLastTx;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceMcsSelSolLastTx(1, dims);
            dataset = file.createDataSet("mcsSelSolLastTx", H5::PredType::NATIVE_INT16, dataspaceMcsSelSolLastTx);
            dataset.write(mcsSelSolLastTx, H5::PredType::NATIVE_INT16);
        }

        if (cellGrpUeStatus->layerSelSolLastTx) {
            uint8_t* layerSelSolLastTx = cellGrpUeStatus->layerSelSolLastTx;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceLayerSelSolLastTx(1, dims);
            dataset = file.createDataSet("layerSelSolLastTx", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSolLastTx);
            dataset.write(layerSelSolLastTx, H5::PredType::NATIVE_UINT8);
        }
    }

    if (cellGrpPrms->cellAssoc) {
        uint8_t* cellAssoc = cellGrpPrms->cellAssoc;

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nUe);
        H5::DataSpace dataspaceCellAssoc(1, dims);
        dataset = file.createDataSet("cellAssoc", H5::PredType::NATIVE_UINT8, dataspaceCellAssoc);
        dataset.write(cellAssoc, H5::PredType::NATIVE_UINT8);
    }

    if (cellGrpPrms->cellAssocActUe) {
        uint8_t* cellAssocActUe = cellGrpPrms->cellAssocActUe;

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nActiveUe);
        H5::DataSpace dataspaceCellAssocActUe(1, dims);
        dataset = file.createDataSet("cellAssocActUe", H5::PredType::NATIVE_UINT8, dataspaceCellAssocActUe);
        dataset.write(cellAssocActUe, H5::PredType::NATIVE_UINT8);
    }

    if (cellGrpPrms->postEqSinr) {
        float* postEqSinr = cellGrpPrms->postEqSinr;

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspacePostEqSinr(1, dims);
        dataset = file.createDataSet("postEqSinr", H5::PredType::NATIVE_FLOAT, dataspacePostEqSinr);
        dataset.write(postEqSinr, H5::PredType::NATIVE_FLOAT);
    }

    if (cellGrpPrms->wbSinr) {
        float*      wbSinr = cellGrpPrms->wbSinr;

        dims[0] = static_cast<hsize_t>(cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceWbSinr(1, dims);
        dataset = file.createDataSet("wbSinr", H5::PredType::NATIVE_FLOAT, dataspaceWbSinr);
        dataset.write(wbSinr, H5::PredType::NATIVE_FLOAT);
    }
    
    if (schdSol) {
        if (schdSol->setSchdUePerCellTTI) {
            uint16_t* setSchdUePerCellTTI = schdSol->setSchdUePerCellTTI;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->numUeSchdPerCellTTI);
            H5::DataSpace dataspaceSetSchdUe(1, dims);
            dataset = file.createDataSet("setSchdUePerCellTTI", H5::PredType::NATIVE_UINT16, dataspaceSetSchdUe);
            dataset.write(setSchdUePerCellTTI, H5::PredType::NATIVE_UINT16);
        }

        if (schdSol->allocSol) {
            int16_t*    allocSol = schdSol->allocSol;

            if (param.allocType == 1) { // type-1 allocation
                dims[0] = static_cast<hsize_t>(2*cellGrpPrms->nUe);
            } else { // type-0 allocation
                dims[0] = static_cast<hsize_t>(cellGrpPrms->nCell*cellGrpPrms->nPrbGrp);
            }
            H5::DataSpace dataspaceAllocSol(1, dims);
            dataset = file.createDataSet("allocSol", H5::PredType::NATIVE_INT16, dataspaceAllocSol);
            dataset.write(allocSol, H5::PredType::NATIVE_INT16);
        }

        if (schdSol->mcsSelSol) {
            int16_t*    mcsSelSol = schdSol->mcsSelSol;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceMcsSelSol(1, dims);
            dataset = file.createDataSet("mcsSelSol", H5::PredType::NATIVE_INT16, dataspaceMcsSelSol);
            dataset.write(mcsSelSol, H5::PredType::NATIVE_INT16);
        }

        if (schdSol->layerSelSol) {
            uint8_t*    layerSelSol = schdSol->layerSelSol;

            dims[0] = static_cast<hsize_t>(cellGrpPrms->nUe);
            H5::DataSpace dataspaceLayerSelSol(1, dims);
            dataset = file.createDataSet("layerSelSol", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSol);
            dataset.write(layerSelSol, H5::PredType::NATIVE_UINT8);
        }
    }

    if (cellGrpPrms->prgMsk) {
        dims[0] = static_cast<hsize_t>(cellGrpPrms->nPrbGrp);
        for (int cIdx = 0; cIdx < cellGrpPrms->nCell; cIdx++) {
            std::string prgMskFieldName = "prgMsk" + std::to_string(cIdx);
            H5::DataSpace dataspacePrgMsk(1, dims);
            dataset = file.createDataSet(prgMskFieldName, H5::PredType::NATIVE_UINT8, dataspacePrgMsk);
            dataset.write(cellGrpPrms->prgMsk[cIdx], H5::PredType::NATIVE_UINT8);
        }
    }
}

void saveToH5_testMAC_perCell(const std::string&                 filename,
                              uint16_t                           cellId,
                              cumac::cumacCellGrpUeStatus*       cellGrpUeStatus,
                              cumac::cumacCellGrpPrms*           cellGrpPrms,
                              cumac::cumacSchdSol*               schdSol) {

    uint8_t DL = cellGrpPrms->dlSchInd;

    // determine the number of active UEs in the given cell, and the mapping to the active UEs in the entire cell group
    int numActiveUeCell = 0;
    std::vector<int> actUeId;

    std::vector<uint8_t> cellAssocActUe(cellGrpPrms->nCell*cellGrpPrms->nActiveUe);
    CUDA_CHECK_ERR(cudaMemcpy(cellAssocActUe.data(), cellGrpPrms->cellAssocActUe, cellGrpPrms->nCell*cellGrpPrms->nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

    for (int ueIdx = 0; ueIdx < cellGrpPrms->nActiveUe; ueIdx++) {
        if (cellAssocActUe[cellId*cellGrpPrms->nActiveUe + ueIdx] == 1) {
            actUeId.push_back(ueIdx);
        }
    }
    numActiveUeCell = actUeId.size();

    // define TV file names
    std::string MAC_SCH_CONFIG_REQUEST_filename = filename + "MAC_SCH_CONFIG_REQUEST_cell_" + std::to_string(cellId) + ".h5";
    std::string MAC_SCH_TTI_REQUEST_filename = filename + "MAC_SCH_TTI_REQUEST_cell_" + std::to_string(cellId) + ".h5";

    // define data types
    H5::IntType datatype_UINT8(H5::PredType::NATIVE_UINT8);
    H5::IntType datatype_UINT16(H5::PredType::NATIVE_UINT16);
    H5::FloatType datatype_FLOAT(H5::PredType::NATIVE_FLOAT);
    H5::IntType datatype_UINT32(H5::PredType::NATIVE_UINT32);

    // ***********************************
    //MAC_SCH_CONFIG_REQUEST

    cumac::MAC_SCH_CONFIG_REQUEST config_request;
        
    config_request.harqEnabledInd = 0;
    config_request.mcsSelCqi = 0;
    config_request.nMaxCell = cellGrpPrms->nCell;
    config_request.nMaxActUePerCell = cellGrpPrms->nMaxActUePerCell;
    config_request.nMaxSchUePerCell = cellGrpPrms->numUeSchdPerCellTTI;
    config_request.nMaxPrg = cellGrpPrms->nPrbGrp;
    config_request.nPrbPerPrg = 4; // to-do: may be possibly assigned to other values
    config_request.nMaxBsAnt = cellGrpPrms->nBsAnt;
    config_request.nMaxUeAnt = cellGrpPrms->nUeAnt;
    config_request.scSpacing = 30000; // to-do: may be possibly assigned to other values
    config_request.allocType = cellGrpPrms->allocType;
    config_request.precoderType = cellGrpPrms->precodingScheme;
    config_request.receiverType = cellGrpPrms->receiverScheme;
    config_request.colMajChanAccess = 1; // to-do: may be possibly assigned to row major access
    config_request.betaCoeff = cellGrpPrms->betaCoeff;
    config_request.sinValThr = cellGrpPrms->sinValThr;
    config_request.corrThr = cellGrpPrms->corrThr;
    config_request.mcsSelSinrCapThr = cellGrpPrms->mcsSelSinrCapThr;
    config_request.mcsSelLutType = cellGrpPrms->mcsSelLutType;
    config_request.prioWeightStep = cellGrpPrms->prioWeightStep;
    config_request.blerTarget = 0.1;

    H5::H5File file_MAC_SCH_CONFIG_REQUEST(MAC_SCH_CONFIG_REQUEST_filename, H5F_ACC_TRUNC);

    hsize_t dims = 1; // Scalar dataset has dimension 1
    H5::DataSpace dataspace_MAC_SCH_CONFIG_REQUEST(1, &dims); // 1D dataspace

    H5::DataSet dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("harqEnabledInd", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.harqEnabledInd, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("mcsSelCqi", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.mcsSelCqi, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nMaxCell", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nMaxCell, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nMaxActUePerCell", datatype_UINT16, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nMaxActUePerCell, datatype_UINT16);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nMaxSchUePerCell", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nMaxSchUePerCell, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nMaxPrg", datatype_UINT16, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nMaxPrg, datatype_UINT16);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nPrbPerPrg", datatype_UINT16, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nPrbPerPrg, datatype_UINT16);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nMaxBsAnt", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nMaxBsAnt, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("nMaxUeAnt", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.nMaxUeAnt, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("scSpacing", datatype_UINT32, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.scSpacing, datatype_UINT32);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("allocType", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.allocType, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("precoderType", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.precoderType, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("receiverType", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.receiverType, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("colMajChanAccess", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.colMajChanAccess, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("betaCoeff", datatype_FLOAT, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.betaCoeff, datatype_FLOAT);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("sinValThr", datatype_FLOAT, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.sinValThr, datatype_FLOAT);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("corrThr", datatype_FLOAT, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.corrThr, datatype_FLOAT);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("mcsSelSinrCapThr", datatype_FLOAT, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.mcsSelSinrCapThr, datatype_FLOAT);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("mcsSelLutType", datatype_UINT8, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.mcsSelLutType, datatype_UINT8);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("prioWeightStep", datatype_UINT16, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.prioWeightStep, datatype_UINT16);

    dataset = file_MAC_SCH_CONFIG_REQUEST.createDataSet("blerTarget", datatype_FLOAT, dataspace_MAC_SCH_CONFIG_REQUEST);
    dataset.write(&config_request.blerTarget, datatype_FLOAT);

    // Close the MAC_SCH_CONFIG_REQUEST TV file
    file_MAC_SCH_CONFIG_REQUEST.close();

    // ***********************************
    // MAC_SCH_TTI_REQUEST
    cumac::MAC_SCH_TTI_REQUEST TTI_reqiest;

    TTI_reqiest.cellID = cellId;
    TTI_reqiest.ULDLSch = cellGrpPrms->dlSchInd;

    TTI_reqiest.nActiveUe = numActiveUeCell;
    TTI_reqiest.nSrsUe = cellGrpPrms->numUeSchdPerCellTTI;
    TTI_reqiest.nPrbGrp = cellGrpPrms->nPrbGrp;
    TTI_reqiest.nBsAnt = cellGrpPrms->nBsAnt;
    TTI_reqiest.nUeAnt = cellGrpPrms->nUeAnt;
    TTI_reqiest.sigmaSqrd = cellGrpPrms->sigmaSqrd;

    H5::H5File file_MAC_SCH_TTI_REQUEST(MAC_SCH_TTI_REQUEST_filename, H5F_ACC_TRUNC);

    dims = 1; // Scalar dataset has dimension 1
    H5::DataSpace dataspace_MAC_SCH_TTI_REQUEST(1, &dims); // 1D dataspace

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("cellID", datatype_UINT16, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.cellID, datatype_UINT16);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("ULDLSch", datatype_UINT8, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.ULDLSch, datatype_UINT8);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("nActiveUe", datatype_UINT16, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.nActiveUe, datatype_UINT16);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("nSrsUe", datatype_UINT16, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.nSrsUe, datatype_UINT16);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("nPrbGrp", datatype_UINT16, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.nPrbGrp, datatype_UINT16);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("nBsAnt", datatype_UINT8, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.nBsAnt, datatype_UINT8);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("nUeAnt", datatype_UINT8, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.nUeAnt, datatype_UINT8);

    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("sigmaSqrd", datatype_FLOAT, dataspace_MAC_SCH_TTI_REQUEST);
    dataset.write(&TTI_reqiest.sigmaSqrd, datatype_FLOAT);

    int numPrgMsk = cellGrpPrms->nPrbGrp;
    hsize_t dims2[] = {static_cast<hsize_t>(numPrgMsk)};

    uint8_t* prgMsk = new uint8_t[numPrgMsk];

    for (int prgIdx = 0; prgIdx < numPrgMsk; prgIdx++) {
        prgMsk[prgIdx] = 1;
    }

    H5::DataSpace dataspace_prgMsk(1, dims2);
    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("prgMsk", H5::PredType::NATIVE_UINT8, dataspace_prgMsk);
    dataset.write(prgMsk, H5::PredType::NATIVE_UINT8);

    delete[] prgMsk;

    uint16_t* CRNTI = new uint16_t[numActiveUeCell];

    for (int ueIdx = 0; ueIdx < numActiveUeCell; ueIdx++) {
        CRNTI[ueIdx] = ueIdx;
    }

    dims2[0] = static_cast<hsize_t>(numActiveUeCell);

    H5::DataSpace dataspace_CRNTI(1, dims2);
    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("CRNTI", H5::PredType::NATIVE_UINT16, dataspace_CRNTI);
    dataset.write(CRNTI, H5::PredType::NATIVE_UINT16);

    delete[] CRNTI;

    uint16_t* srsCRNTI = new uint16_t[cellGrpPrms->numUeSchdPerCellTTI];

    uint16_t* setSchdUePerCellTTICellGrp = new uint16_t[cellGrpPrms->nCell*cellGrpPrms->numUeSchdPerCellTTI];
    uint16_t* setSchdUePerCellTTI = new uint16_t[cellGrpPrms->numUeSchdPerCellTTI];

    if (schdSol->setSchdUePerCellTTI) {
        CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTICellGrp, schdSol->setSchdUePerCellTTI, cellGrpPrms->nCell*cellGrpPrms->numUeSchdPerCellTTI*sizeof(uint16_t), cudaMemcpyDeviceToHost)); 

        for (int ueIdx = 0; ueIdx < cellGrpPrms->numUeSchdPerCellTTI; ueIdx++) {
            int selUeIdx = setSchdUePerCellTTICellGrp[cellId*cellGrpPrms->numUeSchdPerCellTTI + ueIdx];
            auto it = std::find(actUeId.begin(), actUeId.end(), selUeIdx);
            setSchdUePerCellTTI[ueIdx] = std::distance(actUeId.begin(), it);
            srsCRNTI[ueIdx] = setSchdUePerCellTTI[ueIdx];
        }
    } else {
        for (int ueIdx = 0; ueIdx < cellGrpPrms->numUeSchdPerCellTTI; ueIdx++) {
            setSchdUePerCellTTI[ueIdx] = 0xFFFF;
            srsCRNTI[ueIdx] = 0xFFFF;
        }
    }

    dims2[0] = static_cast<hsize_t>(cellGrpPrms->numUeSchdPerCellTTI);
    H5::DataSpace dataspace_srsCRNTI(1, dims2);
    dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("srsCRNTI", H5::PredType::NATIVE_UINT16, dataspace_srsCRNTI);
    dataset.write(srsCRNTI, H5::PredType::NATIVE_UINT16);

    delete[] srsCRNTI;

    if (cellGrpPrms->wbSinr) {
        float* wbSinrCellGrp = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt];
        float* wbSinr = new float[numActiveUeCell*cellGrpPrms->nUeAnt];

        CUDA_CHECK_ERR(cudaMemcpy(wbSinrCellGrp, cellGrpPrms->wbSinr, cellGrpPrms->nActiveUe*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < numActiveUeCell; idx++) {
            int ueIdx = actUeId[idx];
            for (int antIdx = 0; antIdx < cellGrpPrms->nUeAnt; antIdx++) {
                wbSinr[idx*cellGrpPrms->nUeAnt + antIdx] = wbSinrCellGrp[ueIdx*cellGrpPrms->nUeAnt + antIdx];
            }
        }

        dims2[0] = static_cast<hsize_t>(numActiveUeCell*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspace_wbSinr(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("wbSinr", H5::PredType::NATIVE_FLOAT, dataspace_wbSinr);
        dataset.write(wbSinr, H5::PredType::NATIVE_FLOAT);

        delete[] wbSinrCellGrp;
        delete[] wbSinr;
    }

    if (cellGrpPrms->postEqSinr) {
        float* postEqSinrCellGrp = new float[cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        float* postEqSinr = new float[numActiveUeCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];

        CUDA_CHECK_ERR(cudaMemcpy(postEqSinrCellGrp, cellGrpPrms->postEqSinr, cellGrpPrms->nActiveUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < numActiveUeCell; idx++) {
            int ueIdx = actUeId[idx];
            for (int prgIdx = 0; prgIdx < cellGrpPrms->nPrbGrp; prgIdx++) {
                for (int layerIdx = 0; layerIdx < cellGrpPrms->nUeAnt; layerIdx++) {
                    postEqSinr[idx*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt + prgIdx*cellGrpPrms->nUeAnt + layerIdx] = 
                        postEqSinrCellGrp[ueIdx*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt + prgIdx*cellGrpPrms->nUeAnt + layerIdx];
                }
            }
        }

        dims2[0] = static_cast<hsize_t>(numActiveUeCell*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspace_postEqSinr(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("postEqSinr", H5::PredType::NATIVE_FLOAT, dataspace_postEqSinr);
        dataset.write(postEqSinr, H5::PredType::NATIVE_FLOAT);

        delete[] postEqSinrCellGrp;
        delete[] postEqSinr;
    }


    if (cellGrpPrms->estH_fr) {
        int numCfrCellGrp = cellGrpPrms->nPrbGrp*cellGrpPrms->nUe*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt;
        int numCfr = cellGrpPrms->nPrbGrp*cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt;
        dims2[0] = static_cast<hsize_t>(numCfr);

        cuComplex*  estH_frCellGrp = new cuComplex[numCfrCellGrp];
        float*      estH_fr_real = new float[numCfr];
        float*      estH_fr_imag = new float[numCfr];

        CUDA_CHECK_ERR(cudaMemcpy(estH_frCellGrp, cellGrpPrms->estH_fr, numCfrCellGrp*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int prgIdx = 0; prgIdx < cellGrpPrms->nPrbGrp; prgIdx++) {
                for (int idx = 0; idx < cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt; idx++) {
                    int indexCellGrp = prgIdx*cellGrpPrms->nUe*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt +
                                        cellId*cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt + idx;
                    int index = prgIdx*cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nCell*cellGrpPrms->nBsAnt*cellGrpPrms->nUeAnt + idx;
                    estH_fr_real[index] = estH_frCellGrp[indexCellGrp].x;
                    estH_fr_imag[index] = estH_frCellGrp[indexCellGrp].y;
            }
        }

        H5::DataSpace dataspaceEstH_fr_real(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("estH_fr_real", H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_real);
        dataset.write(estH_fr_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceEstH_fr_imag(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("estH_fr_imag", H5::PredType::NATIVE_FLOAT, dataspaceEstH_fr_imag);
        dataset.write(estH_fr_imag, H5::PredType::NATIVE_FLOAT);

        delete[] estH_frCellGrp;
        delete[] estH_fr_real;
        delete[] estH_fr_imag;
    }

    if (cellGrpPrms->prdMat) {
        int numPrdCellGrp;
        int numPrd;
        if (DL == 1) { // DL
            numPrdCellGrp = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
            numPrd = cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        } else { // UL
            numPrdCellGrp = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
            numPrd = cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
        }

        cuComplex*  prdMatCellGrp           = new cuComplex[numPrdCellGrp];
        float*      prdMat_real             = new float[numPrd];
        float*      prdMat_imag             = new float[numPrd];

        CUDA_CHECK_ERR(cudaMemcpy(prdMatCellGrp, cellGrpPrms->prdMat, numPrdCellGrp*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int idx = 0; idx < numPrd; idx++) {
            prdMat_real[idx] = prdMatCellGrp[cellId*numPrd + idx].x;
            prdMat_imag[idx] = prdMatCellGrp[cellId*numPrd + idx].y;
        }

        dims2[0] = static_cast<hsize_t>(numPrd);
        H5::DataSpace dataspacePrdMat_real(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("prdMat_real", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_real);
        dataset.write(prdMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspacePrdMat_imag(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("prdMat_imag", H5::PredType::NATIVE_FLOAT, dataspacePrdMat_imag);
        dataset.write(prdMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] prdMatCellGrp;
        delete[] prdMat_real;
        delete[] prdMat_imag;
    }

    if (cellGrpPrms->detMat) {
        int numDetCellGrp;
        int numDet;
        if (DL == 1) { // DL
            numDetCellGrp = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
            numDet = cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*cellGrpPrms->nUeAnt;
        } else { // UL
            numDetCellGrp = cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
            numDet = cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nBsAnt*cellGrpPrms->nBsAnt;
        }
         
        cuComplex*  detMatCellGrp = new cuComplex[numDetCellGrp]; 
        float*      detMat_real = new float[numDet];
        float*      detMat_imag = new float[numDet];

        CUDA_CHECK_ERR(cudaMemcpy(detMatCellGrp, cellGrpPrms->detMat, numDetCellGrp*sizeof(cuComplex), cudaMemcpyDeviceToHost));

        for (int idx = 0; idx < numDet; idx++) {
            detMat_real[idx] = detMatCellGrp[cellId*numDet + idx].x;
            detMat_imag[idx] = detMatCellGrp[cellId*numDet + idx].y;
        }

        dims2[0] = static_cast<hsize_t>(numDet);
        H5::DataSpace dataspaceDetMat_real(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("detMat_real", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_real);
        dataset.write(detMat_real, H5::PredType::NATIVE_FLOAT);

        H5::DataSpace dataspaceDetMat_imag(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("detMat_imag", H5::PredType::NATIVE_FLOAT, dataspaceDetMat_imag);
        dataset.write(detMat_imag, H5::PredType::NATIVE_FLOAT);

        delete[] detMatCellGrp;
        delete[] detMat_real;
        delete[] detMat_imag;
    }

    if (cellGrpPrms->sinVal) {
        float* sinValCellGrp = new float[cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];
        float* sinVal = new float[cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt];

        CUDA_CHECK_ERR(cudaMemcpy(sinValCellGrp, cellGrpPrms->sinVal, cellGrpPrms->nUe*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt*sizeof(float), cudaMemcpyDeviceToHost));

        for (int idx = 0; idx < cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt; idx++) {
            sinVal[idx] = sinValCellGrp[cellId*cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt + idx];
        }
        
        dims2[0] = static_cast<hsize_t>(cellGrpPrms->numUeSchdPerCellTTI*cellGrpPrms->nPrbGrp*cellGrpPrms->nUeAnt);
        H5::DataSpace dataspaceSinVal(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("sinVal", H5::PredType::NATIVE_FLOAT, dataspaceSinVal);
        dataset.write(sinVal, H5::PredType::NATIVE_FLOAT);

        delete[] sinValCellGrp;
        delete[] sinVal;
    }
    
    if (cellGrpUeStatus->avgRatesActUe) {
        float* avgRatesActUeCellGrp = new float[cellGrpPrms->nActiveUe];
        float* avgRatesActUe = new float[numActiveUeCell];

        CUDA_CHECK_ERR(cudaMemcpy(avgRatesActUeCellGrp, cellGrpUeStatus->avgRatesActUe, cellGrpPrms->nActiveUe*sizeof(float), cudaMemcpyDeviceToHost));

        for (int idx = 0; idx < numActiveUeCell; idx++) {
            int ueIdx = actUeId[idx];
            avgRatesActUe[idx] = avgRatesActUeCellGrp[ueIdx];
        }

        dims2[0] = static_cast<hsize_t>(numActiveUeCell);
        H5::DataSpace dataspace_avgRatesActUe(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("avgRatesActUe", H5::PredType::NATIVE_FLOAT, dataspace_avgRatesActUe);
        dataset.write(avgRatesActUe, H5::PredType::NATIVE_FLOAT);

        delete[] avgRatesActUeCellGrp;
        delete[] avgRatesActUe;
    }

    if (cellGrpUeStatus->tbErrLastActUe) {
        int8_t* tbErrLastActUeCellGrp = new int8_t[cellGrpPrms->nActiveUe];
        int8_t* tbErrLastActUe = new int8_t[numActiveUeCell];

        CUDA_CHECK_ERR(cudaMemcpy(tbErrLastActUeCellGrp, cellGrpUeStatus->tbErrLastActUe, cellGrpPrms->nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < numActiveUeCell; idx++) {
            int ueIdx = actUeId[idx];
            tbErrLastActUe[idx] = tbErrLastActUeCellGrp[ueIdx];
        }

        dims2[0] = static_cast<hsize_t>(numActiveUeCell);
        H5::DataSpace dataspaceTbErrLastActUe(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("tbErrLastActUe", H5::PredType::NATIVE_INT8, dataspaceTbErrLastActUe);
        dataset.write(tbErrLastActUe, H5::PredType::NATIVE_INT8);

        delete[] tbErrLastActUeCellGrp;
        delete[] tbErrLastActUe;
    }

    if (cellGrpUeStatus->newDataActUe) {
        int8_t* newDataActUeCellGrp = new int8_t[cellGrpPrms->nActiveUe]; 
        int8_t* newDataActUe = new int8_t[numActiveUeCell];

        CUDA_CHECK_ERR(cudaMemcpy(newDataActUeCellGrp, cellGrpUeStatus->newDataActUe, cellGrpPrms->nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < numActiveUeCell; idx++) {
            int ueIdx = actUeId[idx];
            newDataActUe[idx] = newDataActUeCellGrp[ueIdx];
        }

        dims2[0] = static_cast<hsize_t>(numActiveUeCell);
        H5::DataSpace dataspace_ndActUe(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("newDataActUe", H5::PredType::NATIVE_INT8, dataspace_ndActUe);
        dataset.write(newDataActUe, H5::PredType::NATIVE_INT8);

        delete[] newDataActUeCellGrp;
        delete[] newDataActUe;
    }

    // ***********************************
    // MAC_SCH_TTI_RESPONSE
    cumac::MAC_SCH_TTI_RESPONSE TTI_response;

    if (schdSol->setSchdUePerCellTTI) {
        dims2[0] = static_cast<hsize_t>(cellGrpPrms->numUeSchdPerCellTTI);
        H5::DataSpace dataspace_setSchdUe(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("setSchdUePerCellTTI_resp", H5::PredType::NATIVE_UINT16, dataspace_setSchdUe);
        dataset.write(setSchdUePerCellTTI, H5::PredType::NATIVE_UINT16);
    }

    if (schdSol->allocSol) {
        int16_t* allocSolCellGrp;
        int16_t* allocSol;

        if (cellGrpPrms->allocType == 1) { // type-1 allocation
            allocSolCellGrp = new int16_t[2*cellGrpPrms->nUe];
            allocSol = new int16_t[2*cellGrpPrms->numUeSchdPerCellTTI];
            
            CUDA_CHECK_ERR(cudaMemcpy(allocSolCellGrp, schdSol->allocSol, 2*cellGrpPrms->nUe*sizeof(int16_t), cudaMemcpyDeviceToHost));

            for (int idx = 0; idx < 2*cellGrpPrms->numUeSchdPerCellTTI; idx++) {
                allocSol[idx] = allocSolCellGrp[cellId*2*cellGrpPrms->numUeSchdPerCellTTI + idx];
            }
            
            dims2[0] = static_cast<hsize_t>(2*cellGrpPrms->numUeSchdPerCellTTI);
        } else { // type-0 allocation
            allocSolCellGrp = new int16_t[cellGrpPrms->nCell*cellGrpPrms->nPrbGrp];
            allocSol = new int16_t[cellGrpPrms->nPrbGrp];

            CUDA_CHECK_ERR(cudaMemcpy(allocSolCellGrp, schdSol->allocSol, cellGrpPrms->nCell*cellGrpPrms->nPrbGrp*sizeof(int16_t), cudaMemcpyDeviceToHost));

            for (int prgIdx = 0; prgIdx < cellGrpPrms->nPrbGrp; prgIdx++) {
                int ueIdx = allocSolCellGrp[prgIdx*cellGrpPrms->nCell + cellId] - cellId*cellGrpPrms->numUeSchdPerCellTTI;

                allocSol[prgIdx] = setSchdUePerCellTTI[ueIdx];
            }

            dims2[0] = static_cast<hsize_t>(cellGrpPrms->nPrbGrp);
        }

        H5::DataSpace dataspaceAllocSol(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("allocSol_resp", H5::PredType::NATIVE_INT16, dataspaceAllocSol);
        dataset.write(allocSol, H5::PredType::NATIVE_INT16);

        delete[] allocSolCellGrp;
        delete[] allocSol;
    }

    if (schdSol->mcsSelSol) {
        int16_t* mcsSelSolCellGrp = new int16_t[cellGrpPrms->nUe];
        int16_t* mcsSelSol = new int16_t[cellGrpPrms->numUeSchdPerCellTTI];

        CUDA_CHECK_ERR(cudaMemcpy(mcsSelSolCellGrp, schdSol->mcsSelSol, cellGrpPrms->nUe*sizeof(int16_t), cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < cellGrpPrms->numUeSchdPerCellTTI; idx++) {
            mcsSelSol[idx] = mcsSelSolCellGrp[cellId*cellGrpPrms->numUeSchdPerCellTTI + idx];
        }

        dims2[0] = static_cast<hsize_t>(cellGrpPrms->numUeSchdPerCellTTI);
        H5::DataSpace dataspaceMcsSelSol(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("mcsSelSol_resp", H5::PredType::NATIVE_INT16, dataspaceMcsSelSol);
        dataset.write(mcsSelSol, H5::PredType::NATIVE_INT16);

        delete[] mcsSelSolCellGrp;
        delete[] mcsSelSol;
    }

    if (schdSol->layerSelSol) {
        uint8_t* layerSelSolCellGrp = new uint8_t[cellGrpPrms->nUe];
        uint8_t* layerSelSol = new uint8_t[cellGrpPrms->numUeSchdPerCellTTI];
        
        CUDA_CHECK_ERR(cudaMemcpy(layerSelSolCellGrp, schdSol->layerSelSol, cellGrpPrms->nUe*sizeof(uint8_t), cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < cellGrpPrms->numUeSchdPerCellTTI; idx++) {
            layerSelSol[idx] = layerSelSolCellGrp[cellId*cellGrpPrms->numUeSchdPerCellTTI + idx];
        }

        dims2[0] = static_cast<hsize_t>(cellGrpPrms->numUeSchdPerCellTTI);
        H5::DataSpace dataspaceLayerSelSol(1, dims2);
        dataset = file_MAC_SCH_TTI_REQUEST.createDataSet("layerSelSol_resp", H5::PredType::NATIVE_UINT8, dataspaceLayerSelSol);
        dataset.write(layerSelSol, H5::PredType::NATIVE_UINT8);

        delete[] layerSelSolCellGrp;
        delete[] layerSelSol;
    }

    delete[] setSchdUePerCellTTICellGrp;
    delete[] setSchdUePerCellTTI;
    
    // close the MAC_SCH_TTI TV file
    file_MAC_SCH_TTI_REQUEST.close();

    // close dataset
    dataset.close();
}