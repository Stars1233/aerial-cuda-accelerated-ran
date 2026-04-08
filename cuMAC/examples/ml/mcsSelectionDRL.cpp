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

#include "trtEngine.h"
#include "mcsSelectionDRL.h"

namespace cumac_ml {

static __constant__ float channCorrArr[100] = {1,0.99975,0.99901,0.99776,0.99602,0.99379,0.99107,0.98785,0.98414,0.97995,0.97528,0.97013,0.9645,0.9584,0.95184,0.94481,0.93733,0.9294,0.92102,0.9122,0.90295,0.89328,0.88319,0.87269,0.86178,0.85048,0.83879,0.82673,0.8143,0.80151,0.78836,0.77488,0.76107,0.74693,0.73249,0.71774,0.70271,0.6874,0.67182,0.65599,0.63992,0.62361,0.60708,0.59035,0.57342,0.5563,0.53902,0.52157,0.50398,0.48626,0.46841,0.45046,0.43241,0.41428,0.39608,0.37783,0.35953,0.3412,0.32285,0.30449,0.28615,0.26783,0.24954,0.23129,0.21311,0.19499,0.17697,0.15903,0.14121,0.1235,0.10593,0.088508,0.071239,0.054137,0.037214,0.020482,0.0039507,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

cumac_ml::mcsSelectionDRL::mcsSelectionDRL(uint16_t    nActiveUe, 
                                                 uint16_t    nUe, 
                                                 uint16_t    eventQueLen, 
                                                 uint8_t     DL):
m_nActiveUe(nActiveUe),
m_nUe(nUe),
m_DL(DL),
m_EventQueLen(eventQueLen)
{
    // sanity check
    if (m_DL == 0) {
        throw std::runtime_error("Error: cuMAC DRL MCS selection does not support UL yet");
    }

    m_slotIdx = 0;

    schSetup = 0;

    //eventQueUl = std::make_unique<std::deque<drlMcsSelEvent>[]>(m_nActiveUe);
    
    eventQueDl = std::make_unique<std::deque<drlMcsSelEvent>[]>(m_nActiveUe);

    cqiReportQue = std::make_unique<std::deque<cqiReport>[]>(m_nActiveUe);

    m_setSchdUePerCellTTI = std::make_unique<uint16_t[]>(m_nUe);

    m_newDataActUe = std::make_unique<int8_t[]>(m_nActiveUe);

    m_mcsLastTx = std::make_unique<int16_t[]>(m_nUe);

    m_numFeaturesPerUe = m_EventQueLen*numFeaturesPerEvent;
    m_numOutputs = numMcsLevels;
    m_maxBatchSize = m_nUe;

    m_mcsSelSolPtrHost =std::make_unique<int16_t[]>(m_nUe);

    // Allocate TensorRT input and output buffers.
    CUDA_CHECK_ERR(cudaMalloc((void**)&m_trtInputTensorsDevice, sizeof(float) * m_maxBatchSize * m_numFeaturesPerUe));
    CUDA_CHECK_ERR(cudaMalloc((void**)&m_trtOutputTensorsDevice, sizeof(float) * m_maxBatchSize * m_numOutputs));

    m_trtInputBuffersHost = std::make_unique<float[]>(m_maxBatchSize * m_numFeaturesPerUe);
    m_trtOutputBuffersHost = std::make_unique<float[]>(m_maxBatchSize * m_numOutputs);

    m_trtInputBuffersDevice = {(void*)m_trtInputTensorsDevice};
    m_trtOutputBuffersDevice = {(void*)m_trtOutputTensorsDevice};
}

cumac_ml::mcsSelectionDRL::~mcsSelectionDRL()
{
    CUDA_CHECK_ERR(cudaFree(m_trtInputTensorsDevice));
    CUDA_CHECK_ERR(cudaFree(m_trtOutputTensorsDevice));
}

void cumac_ml::mcsSelectionDRL::build(std::string modelFile)
{
    // Tensor names and shapes hard-coded
    const std::vector<cumac_ml::trtTensorPrms_t> inputTensorPrms = {{"input_tensor", {m_maxBatchSize, m_numFeaturesPerUe}}};
    const std::vector<cumac_ml::trtTensorPrms_t> outputTensorPrms = {{"dense_4", {m_maxBatchSize, m_numOutputs}}};

    // Create the TRT engine.
    m_pTrtEngine = std::make_unique<cumac_ml::trtEngine>(modelFile.c_str(), parseFromOnnx, m_maxBatchSize, inputTensorPrms, outputTensorPrms);
}

void cumac_ml::mcsSelectionDRL::setup(uint64_t                            slotIdx,
                                            cumac_ml::cumacMlDataApi*     mlDataApi,
                                            cumac::cumacCellGrpPrms*            cellGrpPrms,
                                            cumac::cumacSchdSol*                schdSol,
                                            cumac::cumacCellGrpUeStatus*        cellGrpUeStatus,
                                            cudaStream_t                        strm,
                                            uint8_t                             in_enableHarq)
{
    enableHarq = in_enableHarq;

    // update time slot index
    m_slotIdx = slotIdx;

    // update per-UE event & CQI queues
    if (mlDataApi != nullptr) {
        // sanity check
        if (enableHarq == 1) { // HARQ enabled
            if (mlDataApi->numUeMlData > 0 && mlDataApi->newDataFlag == nullptr) {
                throw std::runtime_error("Error: missing the newDataFlag data when HARQ is enabled");
            }
        }

        updateEventCqiQueDL(mlDataApi);
    }
    
    if (cellGrpPrms != nullptr && schdSol != nullptr && cellGrpUeStatus != nullptr) {
        schSetup = 1;

        // preprocess input to the NN model
        preProcessing(schdSol, cellGrpUeStatus, strm);

        m_pTrtEngine->setup(m_trtInputBuffersDevice, m_trtOutputBuffersDevice);

        // ! assumption's that there is an external device synchronization after the setup() call
        // ! the following device synchronization is necessary if there's no external device synchronization after the setup() call
        // cudaStreamSynchronize(strm);
    } else {
        schSetup = 0;
    }
}

void cumac_ml::mcsSelectionDRL::run(cudaStream_t strm)
{
    //sanity check
    if (schSetup == 0) {
        throw std::runtime_error("Error: cuMAC DRL MCS selection run() should not be called when setup() has not been called for scheduling");
    }

    m_pTrtEngine->run(strm);
    
    postProcessing(strm);

    // ! assumption's that there is an external device synchronization after the run() call
    // ! the following device synchronization is necessary if there's no external device synchronization after the run() call
    // cudaStreamSynchronize(strm);
}

void cumac_ml::mcsSelectionDRL::preProcessing(cumac::cumacSchdSol*            schdSol,
                                                    cumac::cumacCellGrpUeStatus*    cellGrpUeStatus,
                                                    cudaStream_t             strm)
{
    m_mcsSelSolPtrDevice = schdSol->mcsSelSol;

    if (enableHarq == 1) { // HARQ enabled
        CUDA_CHECK_ERR(cudaMemcpyAsync((void*)m_newDataActUe.get(), (void*)cellGrpUeStatus->newDataActUe, m_nActiveUe*sizeof(int8_t), cudaMemcpyDeviceToHost, strm));
        CUDA_CHECK_ERR(cudaMemcpyAsync((void*)m_mcsLastTx.get(), (void*)cellGrpUeStatus->mcsSelSolLastTx, m_nUe*sizeof(int16_t), cudaMemcpyDeviceToHost, strm));
    }
    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)m_setSchdUePerCellTTI.get(), (void*)schdSol->setSchdUePerCellTTI, m_nUe*sizeof(uint16_t), cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK_ERR(cudaStreamSynchronize(strm));

    // m_maxBatchSize == m_nUe
    for (int uIdx = 0; uIdx < m_maxBatchSize; uIdx++) {
        int ueID = m_setSchdUePerCellTTI[uIdx];

        if (ueID == 0xFFFF) { // un-scheduled UE ID
            continue;
        } 

        if (enableHarq == 1) { // HARQ enabled
            if (m_newDataActUe[ueID] != 1) {// the UE is not scheduled for new-tx
                continue;
            }
        }

        int arrayStart = uIdx*m_numFeaturesPerUe;

        if (eventQueDl[ueID].size() < m_EventQueLen) { // not sufficient events in the queue. Invalid DRL model output
            for (int fIdx = 0; fIdx < m_numFeaturesPerUe; fIdx++) {
                m_trtInputBuffersHost[arrayStart + fIdx] = -1.0;
            }
        } else { // sufficient events in the queue. Valid DRL model output
            for (int eIdx = 0; eIdx < m_EventQueLen; eIdx++) {
                // 1st feature - temporal correlation
                int deltaSlot = m_slotIdx - eventQueDl[ueID][eIdx].timeStamp; 
                if (deltaSlot < 0) {
                    deltaSlot = deltaSlot + std::numeric_limits<uint64_t>::max() + 1;
                }

                if (deltaSlot >= 0 && deltaSlot < 100) {
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent] = channCorrArr[deltaSlot];
                } else if (deltaSlot >= 100) {
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent] = 0;
                } 

                // 2nd, 3rd, and 5th features - MCS level + CRC
                if (eventQueDl[ueID][eIdx].mcsLevel != -1) {
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 1] = static_cast<float>(eventQueDl[ueID][eIdx].mcsLevel)/static_cast<float>(numMcsLevels-1);
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 2] = static_cast<float>(eventQueDl[ueID][eIdx].tbErr);
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 4] = 0;
                } else {
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 1] = -1.0;
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 2] = -1.0;
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 4] = 1.0;
                }

                // 4th and 6th features - CQI
                if (eventQueDl[ueID][eIdx].cqiLevel != -1) {
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 3] = static_cast<float>(eventQueDl[ueID][eIdx].cqiLevel)/static_cast<float>(numCqiLevels-1);
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 5] = 0;
                } else {
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 3] = -1.0;
                    m_trtInputBuffersHost[arrayStart + eIdx*numFeaturesPerEvent + 5] = 1.0;
                }
            }
        }
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)m_trtInputTensorsDevice, (void*)m_trtInputBuffersHost.get(), m_maxBatchSize*m_numFeaturesPerUe*sizeof(float), cudaMemcpyHostToDevice, strm));
}

void cumac_ml::mcsSelectionDRL::postProcessing(cudaStream_t strm)
{
    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)m_trtOutputBuffersHost.get(), (void*)m_trtOutputTensorsDevice, sizeof(float)*m_maxBatchSize*m_numOutputs, cudaMemcpyDeviceToHost, strm));
    CUDA_CHECK_ERR(cudaStreamSynchronize(strm));
    
    // m_maxBatchSize == m_nUe
    for (int uIdx = 0; uIdx < m_maxBatchSize; uIdx++) {
        int ueID = m_setSchdUePerCellTTI[uIdx];

        if (ueID == 0xFFFF) {
            m_mcsSelSolPtrHost[uIdx] = -1;
            continue;
        }

        if (enableHarq == 1) { // HARQ enabled
            if (m_newDataActUe[ueID] != 1) {// the UE is not scheduled for new-tx
                m_mcsSelSolPtrHost[uIdx] = m_mcsLastTx[uIdx];
                continue;
            }
        }

        if (eventQueDl[ueID].size() < m_EventQueLen) {
            m_mcsSelSolPtrHost[uIdx] = 0;
        } else {
            int selectedMcs = 0;
            float tempMax = std::numeric_limits<float>::lowest();

            int arrayStart = uIdx*m_numOutputs;

            for (int mIdx = 0; mIdx < m_numOutputs; mIdx++) {
                if (m_trtOutputBuffersHost[arrayStart + mIdx] > tempMax) {
                    tempMax = m_trtOutputBuffersHost[arrayStart + mIdx];
                    selectedMcs = mIdx;
                }
            }

            m_mcsSelSolPtrHost[uIdx] = selectedMcs;
        }
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)m_mcsSelSolPtrDevice, (void*)m_mcsSelSolPtrHost.get(), m_nUe*sizeof(int16_t), cudaMemcpyHostToDevice, strm));
}

void cumac_ml::mcsSelectionDRL::updateEventCqiQueDL(cumac_ml::cumacMlDataApi* mlDataApi)
{
    for (int uIdx = 0; uIdx < mlDataApi->numUeMlData; uIdx++) {
        int16_t ueID = mlDataApi->setUeMlData[uIdx];

        if (ueID >= 0) {
            drlMcsSelEvent tempEvent1, tempEvent2;
            int64_t tempTimeStamp1, tempTimeStamp2;
            uint8_t numEvents = 0;

            int8_t newDataFlag = 1;
            if (enableHarq == 1) {
                newDataFlag = mlDataApi->newDataFlag[uIdx];
            }

            // data sanity check
            if ((mlDataApi->mcsDeltaT[uIdx] != mlDataApi->tbErrDeltaT[uIdx]) || (mlDataApi->mcsLevel[uIdx]*mlDataApi->tbErr[uIdx] < 0)) { // invalid data
                throw std::runtime_error("Error: ML data passed to cuMAC API is invalid - MCS and CRC time offsets do not match, or mcsLevel and tbErr not both set to -1");
            } else if (mlDataApi->mcsDeltaT[uIdx] != -1 && mlDataApi->mcsLevel[uIdx] != -1 && newDataFlag == 1) { // valid MCS + CRC present
                numEvents = 1;

                tempTimeStamp1   = m_slotIdx - mlDataApi->mcsDeltaT[uIdx];
                if (tempTimeStamp1 < 0) {
                    tempEvent1.timeStamp = std::numeric_limits<uint64_t>::max() + tempTimeStamp1 + 1;
                } else {
                    tempEvent1.timeStamp = tempTimeStamp1;
                }

                tempEvent1.mcsLevel      = mlDataApi->mcsLevel[uIdx];
                tempEvent1.tbErr         = mlDataApi->tbErr[uIdx];

                if (mlDataApi->cqiDeltaT[uIdx] != -1 && mlDataApi->cqiLevel[uIdx] != -1) { // CQI present
                    // create CQI report
                    cqiReport cqiRep;
                    if (cqiReportQue[ueID].size() == 0) {
                        cqiRep.preCqiLevel = -1;
                    } else {
                        cqiRep.preCqiLevel = cqiReportQue[ueID].back().cqiLevel;
                    }
                    cqiRep.cqiLevel = mlDataApi->cqiLevel[uIdx];

                    // update event(s)
                    if (mlDataApi->cqiDeltaT[uIdx] == mlDataApi->mcsDeltaT[uIdx]) { // CQI and MCS + CRC in the same event
                        tempEvent1.cqiLevel = mlDataApi->cqiLevel[uIdx];

                        // determine CQI report measurement time stamp
                        cqiRep.measTimeStamp = tempEvent1.timeStamp;
                    } else { // a separate event for CQI
                        numEvents = 2;

                        tempEvent1.cqiLevel = -1;

                        tempTimeStamp2 = m_slotIdx - mlDataApi->cqiDeltaT[uIdx];
                        if (tempTimeStamp2 < 0) {
                            tempEvent2.timeStamp = std::numeric_limits<uint64_t>::max() + tempTimeStamp2 + 1;
                        } else {
                            tempEvent2.timeStamp = tempTimeStamp2;
                        }

                        tempEvent2.cqiLevel = mlDataApi->cqiLevel[uIdx];
                        tempEvent2.mcsLevel = -1;
                        tempEvent2.tbErr    = -1;

                        // determine CQI report measurement time stamp
                        cqiRep.measTimeStamp = tempEvent2.timeStamp;
                    }

                    // determine CQI report reception time stamp
                    if (mlDataApi->cqiRecDeltaT != nullptr) {
                        int64_t tempTimeStampCqiRec = m_slotIdx - mlDataApi->cqiRecDeltaT[uIdx];
                        if (tempTimeStampCqiRec < 0) {
                            cqiRep.recTimeStamp = std::numeric_limits<uint64_t>::max() + tempTimeStampCqiRec + 1;
                        } else {
                            cqiRep.recTimeStamp = tempTimeStampCqiRec;
                        }
                    } else {
                        cqiRep.recTimeStamp = cqiRep.measTimeStamp;
                    }

                    // CQI report enqueue
                    cqiReportQue[ueID].push_back(cqiRep);
                    if (cqiReportQue[ueID].size() > numPastCqiReportPerUe) {
                        cqiReportQue[ueID].pop_front();
                    }
                } else { // CQI not present
                    tempEvent1.cqiLevel = -1;
                }
            } else { // MCS + CRC not present
                if (mlDataApi->cqiDeltaT[uIdx] != -1 && mlDataApi->cqiLevel[uIdx] != -1) { // CQI present
                    // create CQI report
                    cqiReport cqiRep;
                    if (cqiReportQue[ueID].size() == 0) {
                        cqiRep.preCqiLevel = -1;
                    } else {
                        cqiRep.preCqiLevel = cqiReportQue[ueID].back().cqiLevel;
                    }
                    cqiRep.cqiLevel = mlDataApi->cqiLevel[uIdx];

                    // update event
                    numEvents = 1;

                    int64_t tempTimeStamp1 = m_slotIdx - mlDataApi->cqiDeltaT[uIdx];
                    if (tempTimeStamp1 < 0) {
                        tempEvent1.timeStamp = std::numeric_limits<uint64_t>::max() + tempTimeStamp1 + 1;
                    } else {
                        tempEvent1.timeStamp = tempTimeStamp1;
                    }

                    tempEvent1.cqiLevel = mlDataApi->cqiLevel[uIdx];

                    tempEvent1.mcsLevel      = -1;
                    tempEvent1.tbErr         = -1;

                    // determine CQI report measurement time stamp
                    cqiRep.measTimeStamp = tempEvent1.timeStamp;

                    // determine CQI report reception time stamp
                    if (mlDataApi->cqiRecDeltaT != nullptr) {
                        int64_t tempTimeStampCqiRec = m_slotIdx - mlDataApi->cqiRecDeltaT[uIdx];
                        if (tempTimeStampCqiRec < 0) {
                            cqiRep.recTimeStamp = std::numeric_limits<uint64_t>::max() + tempTimeStampCqiRec + 1;
                        } else {
                            cqiRep.recTimeStamp = tempTimeStampCqiRec;
                        }
                    } else {
                        cqiRep.recTimeStamp = cqiRep.measTimeStamp;
                    }

                    // CQI report enqueue
                    cqiReportQue[ueID].push_back(cqiRep);
                    if (cqiReportQue[ueID].size() > numPastCqiReportPerUe) {
                        cqiReportQue[ueID].pop_front();
                    }
                } else {
                    if (newDataFlag != 0) {
                        throw std::runtime_error("Error: ML data passed to cuMAC API is invalid - both MCS + CRC and CQI data not present");
                    }
                }
            }

            // enqueue
            if (numEvents == 1) {
                eventQueDl[ueID].push_back(tempEvent1);
            } else if (numEvents == 2) {
                if (tempTimeStamp1 <= tempTimeStamp2) {
                    eventQueDl[ueID].push_back(tempEvent1);
                    eventQueDl[ueID].push_back(tempEvent2);
                } else {
                    eventQueDl[ueID].push_back(tempEvent2);
                    eventQueDl[ueID].push_back(tempEvent1);
                }
            }

            uint16_t currQueLen = eventQueDl[ueID].size();
            for (int eIdx = m_EventQueLen; eIdx < currQueLen; eIdx++) {
                eventQueDl[ueID].pop_front();
            }
        }
    }
}

} // namespace cumac_ml