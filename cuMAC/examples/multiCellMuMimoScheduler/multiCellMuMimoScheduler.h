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

#include "../../src/api.h"
#include "../../src/cumac.h"
#include "../parameters.h"

bool compareCpuGpuAllocSol(const std::string& resultTvName, const std::string& refTvName)
{
    H5::H5File resultsTvFile(resultTvName, H5F_ACC_RDONLY);
    H5::H5File refTvFile(refTvName, H5F_ACC_RDONLY);

    // Open the dataset
    cumac::cumacSchedulerParam data;
    H5::DataSet dataset = refTvFile.openDataSet("cumacSchedulerParam");

    // Get the compound data type
    H5::CompType compoundType = dataset.getCompType();

    // Read the data from the dataset
    dataset.read(&data, compoundType);

    // check sortedUeList
    std::vector<std::vector<uint16_t>> sortedUeList_ref;
    std::vector<std::vector<uint16_t>> sortedUeList_result;
    sortedUeList_ref.resize(data.nCell, std::vector<uint16_t>(data.nMaxActUePerCell));
    sortedUeList_result.resize(data.nCell, std::vector<uint16_t>(data.nMaxActUePerCell));

    bool pass_sortedUeList = true;
    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        std::string sortUeListFieldName = "sortedUeList_cell" + std::to_string(cIdx);
        if (H5Lexists(refTvFile.getId(), sortUeListFieldName.c_str(), H5P_DEFAULT) > 0) {
            dataset = refTvFile.openDataSet(sortUeListFieldName);
            dataset.read(sortedUeList_ref[cIdx].data(), H5::PredType::NATIVE_UINT16);
        }

        if (H5Lexists(resultsTvFile.getId(), sortUeListFieldName.c_str(), H5P_DEFAULT) > 0) {
            dataset = resultsTvFile.openDataSet(sortUeListFieldName);
            dataset.read(sortedUeList_result[cIdx].data(), H5::PredType::NATIVE_UINT16);
        }

        for (int uIdx = 0; uIdx < data.nMaxActUePerCell; uIdx++) {
            if (sortedUeList_ref[cIdx][uIdx] != sortedUeList_result[cIdx][uIdx]) {
                pass_sortedUeList = false;
                break;
            }
        }

        if (!pass_sortedUeList) {
            break;
        }
    }
    
    if (pass_sortedUeList) {
        printf("sortedUeList solution check: PASS\n");
    } else {
        printf("sortedUeList solution check: FAIL\n");
    }

    // check muMimoInd
    std::vector<uint8_t> muMimoInd_ref(data.nActiveUe);
    std::vector<uint8_t> muMimoInd_result(data.nActiveUe);
    bool pass_muMimoInd = true;

    if (H5Lexists(refTvFile.getId(), "muMimoInd", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("muMimoInd");
        dataset.read(muMimoInd_ref.data(), H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(resultsTvFile.getId(), "muMimoInd", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("muMimoInd");
        dataset.read(muMimoInd_result.data(), H5::PredType::NATIVE_UINT8);
    }

    for (int uIdx = 0; uIdx < data.nActiveUe; uIdx++) {
        if (muMimoInd_ref[uIdx] != muMimoInd_result[uIdx]) {
            pass_muMimoInd = false;
            break;
        }
    }

    if (pass_muMimoInd) {
        printf("muMimoInd solution check: PASS\n");
    } else {
        printf("muMimoInd solution check: FAIL\n");
    }

    // check setSchdUePerCellTTI
    std::vector<uint16_t> setSchdUePerCellTTI_ref(data.nCell*data.numUeForGrpPerCell);
    std::vector<uint16_t> setSchdUePerCellTTI_result(data.nCell*data.numUeForGrpPerCell);
    bool pass_setSchdUePerCellTTI = true;

    if (H5Lexists(refTvFile.getId(), "setSchdUePerCellTTI", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("setSchdUePerCellTTI");
        dataset.read(setSchdUePerCellTTI_ref.data(), H5::PredType::NATIVE_UINT16);
    }

    if (H5Lexists(resultsTvFile.getId(), "setSchdUePerCellTTI", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("setSchdUePerCellTTI");
        dataset.read(setSchdUePerCellTTI_result.data(), H5::PredType::NATIVE_UINT16);
    }

    for (int idx = 0; idx < data.nCell*data.numUeForGrpPerCell; idx++) {
        if (setSchdUePerCellTTI_ref[idx] != setSchdUePerCellTTI_result[idx]) {
            pass_setSchdUePerCellTTI = false;
            break;
        }
    }

    if (pass_setSchdUePerCellTTI) {
        printf("setSchdUePerCellTTI solution check: PASS\n");
    } else {
        printf("setSchdUePerCellTTI solution check: FAIL\n");
    }

    // check allocSol
    std::vector<int16_t> allocSol_ref(2*data.nActiveUe);
    std::vector<int16_t> allocSol_result(2*data.nActiveUe);
    bool pass_allocSol = true;

    if (H5Lexists(refTvFile.getId(), "allocSol", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("allocSol");
        dataset.read(allocSol_ref.data(), H5::PredType::NATIVE_INT16);
    }

    if (H5Lexists(resultsTvFile.getId(), "allocSol", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("allocSol");
        dataset.read(allocSol_result.data(), H5::PredType::NATIVE_INT16);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        for (int uIdx = 0; uIdx < data.numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = setSchdUePerCellTTI_ref[cIdx*data.numUeForGrpPerCell + uIdx];

            if ((allocSol_ref[2*ueId] != allocSol_result[2*ueId]) || (allocSol_ref[2*ueId + 1] != allocSol_result[2*ueId + 1])) {
                pass_allocSol = false;
                break;
            }
        }

        if (!pass_allocSol) {
            break;
        }
    }

    if (pass_allocSol) {
        printf("allocSol solution check: PASS\n");
    } else {
        printf("allocSol solution check: FAIL\n");
    }

    // check layerSelSol
    std::vector<uint8_t> layerSelSol_ref(data.nActiveUe);
    std::vector<uint8_t> layerSelSol_result(data.nActiveUe);
    bool pass_layerSelSol = true;
    
    if (H5Lexists(refTvFile.getId(), "layerSelSol", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("layerSelSol");
        dataset.read(layerSelSol_ref.data(), H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(resultsTvFile.getId(), "layerSelSol", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("layerSelSol");
        dataset.read(layerSelSol_result.data(), H5::PredType::NATIVE_UINT8);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        for (int uIdx = 0; uIdx < data.numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = setSchdUePerCellTTI_ref[cIdx*data.numUeForGrpPerCell + uIdx];

            if (layerSelSol_ref[ueId] != layerSelSol_result[ueId]) {
                pass_layerSelSol = false;
                // printf("layerSelSol_ref = %d, layerSelSol_result = %d\n", layerSelSol_ref[ueId], layerSelSol_result[ueId]);
                break;
            }
        }

        if (!pass_layerSelSol) {
            break;
        }
    }
    
    if (pass_layerSelSol) {
        printf("layerSelSol solution check: PASS\n");
    } else {
        printf("layerSelSol solution check: FAIL\n");
    }

    // check layerSelUegSol
    std::vector<uint8_t> layerSelUegSol_ref(data.nActiveUe);
    std::vector<uint8_t> layerSelUegSol_result(data.nActiveUe);
    bool pass_layerSelUegSol = true;

    if (H5Lexists(refTvFile.getId(), "layerSelUegSol", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("layerSelUegSol");
        dataset.read(layerSelUegSol_ref.data(), H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(resultsTvFile.getId(), "layerSelUegSol", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("layerSelUegSol");
        dataset.read(layerSelUegSol_result.data(), H5::PredType::NATIVE_UINT8);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        for (int uIdx = 0; uIdx < data.numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = setSchdUePerCellTTI_ref[cIdx*data.numUeForGrpPerCell + uIdx];

            if (layerSelUegSol_ref[ueId] != layerSelUegSol_result[ueId]) {
                pass_layerSelUegSol = false;
                // printf("layerSelUegSol_ref = %d, layerSelUegSol_result = %d\n", layerSelUegSol_ref[ueId], layerSelUegSol_result[ueId]);
                break;
            }
        }

        if (!pass_layerSelUegSol) {
            break;
        }
    }

    if (pass_layerSelUegSol) {
        printf("layerSelUegSol solution check: PASS\n");
    } else {
        printf("layerSelUegSol solution check: FAIL\n");
    }

    // check nSCID
    std::vector<uint8_t> nSCID_ref(data.nActiveUe*data.nUeAnt);
    std::vector<uint8_t> nSCID_result(data.nActiveUe*data.nUeAnt);
    bool pass_nSCID = true;

    if (H5Lexists(refTvFile.getId(), "nSCID", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("nSCID");
        dataset.read(nSCID_ref.data(), H5::PredType::NATIVE_UINT8);
    }

    if (H5Lexists(resultsTvFile.getId(), "nSCID", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("nSCID");
        dataset.read(nSCID_result.data(), H5::PredType::NATIVE_UINT8);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        for (int uIdx = 0; uIdx < data.numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = setSchdUePerCellTTI_ref[cIdx*data.numUeForGrpPerCell + uIdx];

            if (nSCID_ref[ueId] != nSCID_result[ueId]) {
                pass_nSCID = false;
                // printf("nSCID_ref = %d, nSCID_result = %d\n", nSCID_ref[ueId], nSCID_result[ueId]);
                break;
            }
        }

        if (!pass_nSCID) {
            break;
        }
    }

    if (pass_nSCID) {
        printf("nSCID solution check: PASS\n");
    } else {
        printf("nSCID solution check: FAIL\n");
    }

    // check rsrpCurrTx
    std::vector<float> rsrpCurrTx_ref(data.nActiveUe);
    std::vector<float> rsrpCurrTx_result(data.nActiveUe);
    bool pass_rsrpCurrTx = true;

    if (H5Lexists(refTvFile.getId(), "rsrpCurrTx", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("rsrpCurrTx");
        dataset.read(rsrpCurrTx_ref.data(), H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(resultsTvFile.getId(), "rsrpCurrTx", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("rsrpCurrTx");
        dataset.read(rsrpCurrTx_result.data(), H5::PredType::NATIVE_FLOAT);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        for (int uIdx = 0; uIdx < data.numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = setSchdUePerCellTTI_ref[cIdx*data.numUeForGrpPerCell + uIdx];

            if (rsrpCurrTx_ref[ueId] != rsrpCurrTx_result[ueId]) {
                pass_rsrpCurrTx = false;
                // printf("rsrpCurrTx_ref = %d, rsrpCurrTx_result = %d\n", rsrpCurrTx_ref[ueId], rsrpCurrTx_result[ueId]);
                break;
            }
        }

        if (!pass_rsrpCurrTx) {
            break;
        }
    }

    if (pass_rsrpCurrTx) {
        printf("rsrpCurrTx solution check: PASS\n");
    } else {
        printf("rsrpCurrTx solution check: FAIL\n");
    }

    // check rsrpLastTx
    std::vector<float> rsrpLastTx_ref(data.nActiveUe);
    std::vector<float> rsrpLastTx_result(data.nActiveUe);
    bool pass_rsrpLastTx = true;

    if (H5Lexists(refTvFile.getId(), "rsrpLastTx", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("rsrpLastTx");
        dataset.read(rsrpLastTx_ref.data(), H5::PredType::NATIVE_FLOAT);
    }

    if (H5Lexists(resultsTvFile.getId(), "rsrpLastTx", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("rsrpLastTx");
        dataset.read(rsrpLastTx_result.data(), H5::PredType::NATIVE_FLOAT);
    }

    for (int uIdx = 0; uIdx < data.nActiveUe; uIdx++) {
        if (rsrpLastTx_ref[uIdx] != rsrpLastTx_result[uIdx]) {
            pass_rsrpLastTx = false;
            break;
        }
    }

    if (pass_rsrpLastTx) {
        printf("rsrpLastTx solution check: PASS\n");
    } else {
        printf("rsrpLastTx solution check: FAIL\n");
    }

    // check mcsSelSol
    std::vector<int16_t> mcsSelSol_ref(data.nActiveUe);
    std::vector<int16_t> mcsSelSol_result(data.nActiveUe);
    bool pass_mcsSelSol = true;

    if (H5Lexists(refTvFile.getId(), "mcsSelSol", H5P_DEFAULT) > 0) {
        dataset = refTvFile.openDataSet("mcsSelSol");
        dataset.read(mcsSelSol_ref.data(), H5::PredType::NATIVE_INT16);
    }

    if (H5Lexists(resultsTvFile.getId(), "mcsSelSol", H5P_DEFAULT) > 0) {
        dataset = resultsTvFile.openDataSet("mcsSelSol");
        dataset.read(mcsSelSol_result.data(), H5::PredType::NATIVE_INT16);
    }

    for (int cIdx = 0; cIdx < data.nCell; cIdx++) {
        for (int uIdx = 0; uIdx < data.numUeForGrpPerCell; uIdx++) {
            uint16_t ueId = setSchdUePerCellTTI_ref[cIdx*data.numUeForGrpPerCell + uIdx];

            if (mcsSelSol_ref[ueId] != mcsSelSol_result[ueId]) {
                pass_mcsSelSol = false;
                // printf("mcsSelSol_ref = %d, mcsSelSol_result = %d\n", mcsSelSol_ref[ueId], mcsSelSol_result[ueId]);
                break;
            }
        }

        if (!pass_mcsSelSol) {
            break;
        }
    }

    if (pass_mcsSelSol) {
        printf("mcsSelSol solution check: PASS\n");
    } else {
        printf("mcsSelSol solution check: FAIL\n");
    }

    // check ollaParamActUe
    std::vector<cumac::ollaParam>  ollaParamActUe_ref(data.nActiveUe);
    std::vector<cumac::ollaParam>  ollaParamActUe_result(data.nActiveUe);
    bool pass_olla = true;

    std::vector<float> delta(data.nActiveUe);
    std::vector<float> delta_up(data.nActiveUe);
    std::vector<float> delta_down(data.nActiveUe);
    std::vector<float> layerBnd(data.nActiveUe);
    std::vector<float> layerBnd_up(data.nActiveUe);
    std::vector<float> layerBnd_down(data.nActiveUe);

    bool compareInd = true;
    if ((H5Lexists(refTvFile.getId(), "OLLA_delta_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(refTvFile.getId(), "OLLA_delta_up_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(refTvFile.getId(), "OLLA_delta_down_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(refTvFile.getId(), "OLLA_layerBnd_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(refTvFile.getId(), "OLLA_layerBnd_up_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(refTvFile.getId(), "OLLA_layerBnd_down_arr", H5P_DEFAULT) > 0)) {
            
        dataset = refTvFile.openDataSet("OLLA_delta_arr");
        dataset.read(delta.data(), H5::PredType::NATIVE_FLOAT);    
        dataset = refTvFile.openDataSet("OLLA_delta_up_arr");
        dataset.read(delta_up.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = refTvFile.openDataSet("OLLA_delta_down_arr");
        dataset.read(delta_down.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = refTvFile.openDataSet("OLLA_layerBnd_arr");
        dataset.read(layerBnd.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = refTvFile.openDataSet("OLLA_layerBnd_up_arr");
        dataset.read(layerBnd_up.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = refTvFile.openDataSet("OLLA_layerBnd_down_arr");
        dataset.read(layerBnd_down.data(), H5::PredType::NATIVE_FLOAT);   

        for (int uIdx = 0; uIdx < data.nActiveUe; uIdx++) {
            ollaParamActUe_ref[uIdx].delta = delta[uIdx];
            ollaParamActUe_ref[uIdx].delta_up = delta_up[uIdx];
            ollaParamActUe_ref[uIdx].delta_down = delta_down[uIdx];
        }
    } else {
        compareInd = false;
    }

    if ((H5Lexists(resultsTvFile.getId(), "OLLA_delta_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(resultsTvFile.getId(), "OLLA_delta_up_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(resultsTvFile.getId(), "OLLA_delta_down_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(resultsTvFile.getId(), "OLLA_layerBnd_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(resultsTvFile.getId(), "OLLA_layerBnd_up_arr", H5P_DEFAULT) > 0) &&
        (H5Lexists(resultsTvFile.getId(), "OLLA_layerBnd_down_arr", H5P_DEFAULT) > 0)) {
            
        dataset = resultsTvFile.openDataSet("OLLA_delta_arr");
        dataset.read(delta.data(), H5::PredType::NATIVE_FLOAT);    
        dataset = resultsTvFile.openDataSet("OLLA_delta_up_arr");
        dataset.read(delta_up.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = resultsTvFile.openDataSet("OLLA_delta_down_arr");
        dataset.read(delta_down.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = resultsTvFile.openDataSet("OLLA_layerBnd_arr");
        dataset.read(layerBnd.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = resultsTvFile.openDataSet("OLLA_layerBnd_up_arr");
        dataset.read(layerBnd_up.data(), H5::PredType::NATIVE_FLOAT);   
        dataset = resultsTvFile.openDataSet("OLLA_layerBnd_down_arr");
        dataset.read(layerBnd_down.data(), H5::PredType::NATIVE_FLOAT);  

        for (int uIdx = 0; uIdx < data.nActiveUe; uIdx++) {
            ollaParamActUe_result[uIdx].delta = delta[uIdx];
            ollaParamActUe_result[uIdx].delta_up = delta_up[uIdx];
            ollaParamActUe_result[uIdx].delta_down = delta_down[uIdx];
        }
    } else {
        compareInd = false;
    }

    if (compareInd) {
        for (int uIdx = 0; uIdx < data.nActiveUe; uIdx++) {
            if (ollaParamActUe_ref[uIdx].delta != ollaParamActUe_result[uIdx].delta) {
                pass_olla = false;
                break;
            }

            if (ollaParamActUe_ref[uIdx].delta_up != ollaParamActUe_result[uIdx].delta_up) {
                pass_olla = false;
                break;
            }

            if (ollaParamActUe_ref[uIdx].delta_down != ollaParamActUe_result[uIdx].delta_down) {
                pass_olla = false;
                break;
            }
        }
    }

    if (pass_olla) {
        printf("ollaParamActUe update check: PASS\n");
    } else {
        printf("ollaParamActUe update check: FAIL\n");
    }

    return pass_sortedUeList && pass_muMimoInd && pass_setSchdUePerCellTTI && pass_allocSol && pass_layerSelSol && pass_layerSelUegSol && pass_nSCID && pass_rsrpCurrTx && pass_rsrpLastTx && pass_mcsSelSol && pass_olla;
}

void preProcessInput(cumac::cumacCellGrpUeStatus*       cellGrpUeStatusGpu,
                     cumac::cumacCellGrpPrms*           cellGrpPrmsGpu,
                     cumac::cumacSchdSol*               schdSolGpu)
{
    std::vector<int8_t> tbErrLast(cellGrpPrmsGpu->nActiveUe, -1);

    CUDA_CHECK_ERR(cudaMemcpy(cellGrpUeStatusGpu->tbErrLast, tbErrLast.data(), cellGrpPrmsGpu->nActiveUe*sizeof(int8_t), cudaMemcpyHostToDevice));
}