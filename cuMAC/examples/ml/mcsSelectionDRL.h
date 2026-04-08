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
#include "trtEngine.h"

namespace cumac_ml {

constexpr int numFeaturesPerEvent   = 6;
constexpr int numPastCqiReportPerUe = 5;
constexpr int numMcsLevels          = 28;
constexpr int numCqiLevels          = 16;
constexpr int eventQueLenDefault    =  3;

// cuMAC ML algorithm API data structures
struct cumacMlDataApi {
    uint16_t    numUeMlData; // number of UEs in the coordinated cell group that have new ML algorithm data in the current time slot
    // set to 0 if no UE in the coordinated cell group has updated ML algorithm data in the current time slot

    uint16_t    maxNumUeMlData; // maximum number of UEs in the coordinated cell group that have new ML algorithm data in the current time slot
    
    int16_t*    setUeMlData = nullptr;
    // array of global UE IDs in the coordinated cell group that have new ML algorithm data in the current time slot
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // setUeMlData[uIdx] is the uIdx-th UE's global ID
    // -1 indicates an invalid/unused element
    // ! current limitation: in each time slot, a UE can only have a single MCS + tbErr and/or a single CQI passed to this API structure.
    // ! this means that there cannot be duplicate UE IDs in setUeMlData 

    int8_t*     newDataFlag = nullptr;
    // array of new-tx/re-tx indicators of UEs in the coordinated cell group that have updated data in the current time slot
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // newDataFlag[uIdx] is the new-tx/re-tx indicator of the UE with global ID setUeMlData[uIdx]
    // 0 - retransmission, 1 - new data/initial transmission
    // -1 indicates an invalid/unused element
    // ** If K1 == 0, this buffer only contains the new-tx/re-tx indicator for the last transmission

    int8_t*     mcsLevel = nullptr;
    // array of the selected MCS levels of UEs in the coordinated cell group that have updated data in the current time slot
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // mcsLevel[uIdx] is the selected MCS level of the UE with global ID setUeMlData[uIdx]
    // -1 indicates an invalid/unused element
    // ** If K1 == 0, this buffer only contains the MCS for the last transmission

    int16_t*    mcsDeltaT = nullptr; 
    // array of time offsets (in unit of time slots) to the current time slot of the selected MCS levels
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // mcsDeltaT[uIdx] is the time slot offset for the UE with global ID setUeMlData[uIdx]
    // time offsets should be positive values
    // -1 indicates an invalid/unused element
    // ** If K1 == 0, values in this buffer can be hardcoded to 1 (previous slot)

    int8_t*     tbErr = nullptr;
    // array of the TB decoding error indicators of UEs in the coordinated cell group that have updated data in the current time slot
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // tbErr[uIdx] is the TB decoding error indicator of the UE with global ID setUeMlData[uIdx]
    // -1 indicates an invalid/unused element

    int16_t*    tbErrDeltaT = nullptr; 
    // array of time offsets (in unit of time slots) to the current time slot of the TB decoding error indicators
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // tbErrDeltaT[uIdx] is the time slot offset for the UE with global ID setUeMlData[uIdx]
    // time offsets should be positive values
    // -1 indicates an invalid/unused element

    int8_t*     cqiLevel = nullptr; 
    // array of the reported CQI levels of UEs in the coordinated cell group that have updated data in the current time slot
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // cqiLevel[uIdx] is the reported CQI level of the UE with global ID setUeMlData[uIdx]
    // -1 indicates an invalid/unused element

    int16_t*    cqiDeltaT = nullptr; 
    // array of time offsets (in unit of time slots) to the current time slot for the CQI measurement at the UE side
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // cqiDeltaT[uIdx] is the time slot offset for the UE with global ID setUeMlData[uIdx]
    // time offsets should be positive values
    // -1 indicates an invalid/unused element

    int16_t*    cqiRecDeltaT = nullptr; 
    // ! If no memory is allocated for this buffer, a zero delay is assumed between the CQI measurement and the CQI report reception for all CQI reports
    // array of time offsets (in unit of time slots) to the current time slot for the CQI report reception at the base station side
    // format: one dimensional array. array size: maxNumUeMlData
    // denote uIdx = 0, 1, …, maxNumUeMlData-1 as the UE index
    // cqiRecDeltaT[uIdx] is the time slot offset for the UE with global ID setUeMlData[uIdx]
    // time offsets should be positive values (no larger than cqiDeltaT)
    // -1 indicates an invalid/unused element
};   

struct drlMcsSelEvent {
    uint64_t timeStamp;
    // time stamp of the event
    // value: −32,768 -- 0;

    int8_t mcsLevel;
    // selected MCS level
    // value: 0 - 27
    // -1 indicates this field is invalid

    int8_t  tbErr;
    // TB decoding error indicator
    // value: 0 (CRC pass) / 1 (CRC failure)
    // -1 indicates this field is invalid

    int8_t  cqiLevel;
    // UE reported CQI level
    // value: 0 - 15;
    // -1 indicates this field is invalid
};

struct cqiReport {
    uint64_t measTimeStamp; 
    // time stamp for when the CQI is measured at the UE side

    uint64_t recTimeStamp; 
    // time stamp for when the CQI report is received at the base station side

    int8_t  cqiLevel;
    // UE reported CQI level. Value: 0 - 15

    int8_t  preCqiLevel = -1;
    // the previous UE reported CQI level. Value: 0 - 15
    // -1 indicates that the previous UE reported CQI does not exist
};

class mcsSelectionDRL {
// ! limitations of current implementation
// only support Table 5.1.3.1-2: MCS index table 2, 3GPP TS 38.214  
// only support DL MCS selection. UL not supported yet
// only support single-layer DL transmssion. Multiple layers per tx not supported yet
// only support fixed number of allocated PRBs per UE. Dynamically changing number of allocated PRBs per UE not supported yet
// assuming CQI measurement and CQI report received at the gNB are in the same time slot
// In each time slot, only a single MCS+CRC and/or a single CQI can be passed to cuMAC for each UE. This means that there cannot be duplicate UE IDs in the buffer setUeMlData
public:
    // default constructor
    mcsSelectionDRL(uint16_t    nActiveUe, 
                    uint16_t    nUe, 
                    uint16_t    eventQueLen = 10, 
                    uint8_t     DL = 1);
    // nActiveUe:   (maximum) total number of active UEs in all coordinated cells
    // nUe:         (maximum) total number of scheduled UEs per TTI in all coordinated cells
    // eventQueLen: event queue length considered for each UE's DRL model input
    // DL:          DL indicator, 0 - UL, 1 - DL

    // destructor
    ~mcsSelectionDRL();

    mcsSelectionDRL(mcsSelectionDRL const&)            = delete;
    mcsSelectionDRL& operator=(mcsSelectionDRL const&) = delete;

    // Function builds the network engine
    void build(std::string modelFile);

    // setup() function for per-TTI algorithm execution
    // ! must be called in every time slot
    void setup(uint64_t                             slotIdx,
               cumac_ml::cumacMlDataApi*      mlDataApi = nullptr,
               cumac::cumacCellGrpPrms*             cellGrpPrms = nullptr,
               cumac::cumacSchdSol*                 schdSol = nullptr,
               cumac::cumacCellGrpUeStatus*         cellGrpUeStatus = nullptr,
               cudaStream_t                         strm = 0,
               uint8_t                              in_enableHarq = 0);
    // slotIdx:         global time slot index maintained outside of cuMAC
    // mlDataApi:       cuMAC ML data in host memory
    // cellGrpPrms:     cell group cuMAC API structure in device memory
    // schdSol:         scheduling solution cuMAC API structure in device memory
    // strm:            CUDA stream
    // in_enableHarq:   HARQ enabled indicator, 0 - HARQ disabled, 1 - HARQ enabled
    // ! assumption's that there is an external device synchronization after the setup() call

    // run() function for per-TTI DRL algorithm inference
    // ! hardcode MCS/use CQI for LUT-base MCS selection when some UE doesn't have sufficient past events in the queue
    void run(cudaStream_t strm);
    // ! assumption's that there is an external device synchronization after the run() call


    // functions for debugging and testing
    // provide external read-only access to the TRT model input buffer
    const float* getInputBuffer() const {
        return m_trtInputBuffersHost.get();
    }

    // provide external read-only access to the TRT model output buffer
    const float* getOutputBuffer() const {
        return m_trtOutputBuffersHost.get();
    }

    // provide external read-only access to the DL per-UE event queues
    const std::deque<drlMcsSelEvent>* getEventQueDl() const {
        return eventQueDl.get();
    }

private:
    uint32_t                    m_nActiveUe;  // maximum total number of active UEs in all coordinated cells

    uint16_t                    m_nUe; // maximum total number of scheduled UEs in all coordinated cells per TTI

    std::unique_ptr<uint16_t[]> m_setSchdUePerCellTTI = nullptr; // array of scheduled UE IDs

    int                         m_numFeaturesPerUe; // number of features per event

    int                         m_numOutputs; // number of output elements

    int                         m_maxBatchSize; // maximum batch size

    int16_t*                    m_mcsSelSolPtrDevice = nullptr; // pointer to device MCS selection solution array

    std::unique_ptr<int16_t[]>  m_mcsSelSolPtrHost = nullptr; // pointer to host MCS selection solution array

    // time slot index of the cuMAC DRL MCS selection module
    uint64_t                    m_slotIdx = 0; // initialized to 0 and reset to 0 when time slot index reaches maximum value of uint32_t

    // indicator for DL/UL
    uint8_t                     m_DL = 1; // 0 for UL, 1 for DL 

    // indicator for HARQ enabled
    uint8_t                     enableHarq = 0;

    std::unique_ptr<int8_t[]>   m_newDataActUe; // Indicators of new-tx/re-tx for all active UEs

    std::unique_ptr<int16_t[]>  m_mcsLastTx; // MCS levels of the last transmission of the selected/scheduled UEs

    // indicator for scheduling setup
    uint8_t                     schSetup; // 0 for not setup for scheduling (in U/S slots), 1 for setup for scheduling (in D slots)

    // indicator for parse from onnx file
    const bool                  parseFromOnnx = true;

    uint16_t                    m_EventQueLen = eventQueLenDefault; // length of each UE's event queue

    std::unique_ptr<std::deque<drlMcsSelEvent>[]> eventQueUl;// all active UEs' UL event queues
    
    std::unique_ptr<std::deque<drlMcsSelEvent>[]> eventQueDl;// all active UEs' DL event queues

    std::unique_ptr<std::deque<cqiReport>[]> cqiReportQue;// all active UEs' CQI report queues

    // update DL event queue
    void updateEventCqiQueDL(cumacMlDataApi* mlDataApi); // per-UE CQI queues are also updated by this function

    // prepare input for the DRL NN
    void preProcessing(cumac::cumacSchdSol*            schdSol,
                       cumac::cumacCellGrpUeStatus*    cellGrpUeStatus,
                       cudaStream_t                    strm);

    // process output of the DRL NN
    void postProcessing(cudaStream_t strm);

    // TensorRT wrapper and input and output device buffers.
    std::unique_ptr<cumac_ml::trtEngine> m_pTrtEngine = nullptr;
    std::vector<void*> m_trtInputBuffersDevice; // Input buffer, device memory
    float* m_trtInputTensorsDevice = nullptr; // Input tensor, device memory
    std::unique_ptr<float[]> m_trtInputBuffersHost = nullptr; // Input buffer, host memory
    std::vector<void*> m_trtOutputBuffersDevice;  // Output buffer, device memory
    float* m_trtOutputTensorsDevice = nullptr;  // Output buffer, device memory
    std::unique_ptr<float[]> m_trtOutputBuffersHost = nullptr; // Output buffer, host memory
};

} // namespace cumac_ml