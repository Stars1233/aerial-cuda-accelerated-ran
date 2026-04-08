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

constexpr uint8_t SRS_COMB_SIZE = 4;

// cuMAC SRS API data structures ////////////////////////////////////////////////////
struct cumacSrsCellGrpUeStatus { // per-UE information
    //----------------- data buffers -----------------        
    int8_t*     newDataActUe = nullptr; 
    // indicators of initial transmission/retransmission for all active UEs in the coordinated cell group. 
    // format: one dimensional array. Array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the global active UE index in the coordinated cell group. 
    // newDataActUe[uIdx] is the indicator of initial transmission/retransmission for the uIdx-th active UE in the coordinated cell group. 
    // 0 – retransmission 
    // 1 - new data/initial transmission 
    // -1 indicates an invalid element
    
    uint32_t*    srsLastTxCounter  = nullptr;
    // array of time slot counter from the last scheduled SRS transmission of each active UE in the coordinated cell group. 
    // format: one-dimensional array. array size: nActiveUe.
    // Denote uidx = 0, 1, ..., nActiveUe-1 as the index in the array.
    // srsLastTxCounter[uIdx] is the uIdx-th active UE’s time slot counter from the last scheduled SRS transmission.  
    
    uint8_t*    srsNumAntPorts   =  nullptr;
    // array of the number of SRS antenna ports of each active UE in the coordinated cell group.
    // format: one-dimensional array. array size: nActiveUe.
    // Denote uidx = 0, 1, ..., nActiveUe-1 as the index in the array.
    // srsNumAntPorts[uIdx] is the uIdx-th active UE’s number of SRS antenna ports.
    
    uint8_t*    srsResourceType = nullptr;
    // array of SRS resource type of each active UE in the coordinated cell group.
    // format: one-dimensional array. array size: nActiveUe.
    // Denote uidx = 0, 1, ..., nActiveUe-1 as the index in the array.
    // srsResourceType[uIdx] is the uIdx-th active UE’s SRS resource type; currently only support 0 – aperiodic.
    
    float*      srsWbSnr = nullptr;
    // array of the reported wideband SNRs in dB measured within configured SRS bandwidth for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsWbSnr[uIdx] is the wideband SNR in dB measured within configured SRS bandwidth for the uIdx-th active UE in the coordinated cell group. 
    // If SRS channel estiamtes are not available for the uIdx-th UE, set srsWbSnr[uIdx] to -100.0
    
    float*      srsWbSnrThreshold = nullptr;
    // array of the thresholds of wideband SNR in dB measured within configured SRS bandwidth for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsWbSnrThreshold[uIdx] is the threshold of wideband SNR in dB measured within configured SRS bandwidth for the uIdx-th active UE in the coordinated cell group. 
    
    float*      srsWidebandSignalEnergy = nullptr;
    // array of the reported wideband signal energies measured within configured SRS bandwidth for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsWidebandSignalEnergy[uIdx] is the wideband signal energy measured within configured SRS bandwidth for the uIdx-th active UE in the coordinated cell group. 
    // If SRS channel estiamtes are not available for the uIdx-th UE, set srsWidebandSignalEnergy[uIdx] to -100.0
    
    float*      srsTxPwrMax = nullptr;
    // array of the UE configured maximum output powers for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsTxPwrMax[uIdx] is the configured maximum output power for the uIdx-th active UE in the coordinated cell group. 

    float*      srsPwr0 = nullptr;
    // array of the provided p0 for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsPwr0[uIdx] is the provided p0 for the uIdx-th active UE in the coordinated cell group. 

    float*      srsPwrAlpha = nullptr;
    // array of the provided alpha for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsPwrAlpha[uIdx] is the provided alpha for the uIdx-th active UE in the coordinated cell group. 

    uint8_t*    srsTpcAccumulationFlag = nullptr;
    // array of the indications of using the TPC commands via accumulatio for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsTpcAccumulationFlag[uIdx] is the accumulatio indication for the uIdx-th active UE in the coordinated cell group. 
    
    float*   srsPowerControlAdjustmentState = nullptr; 
    // array of the TPC adjustment states for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsPowerControlAdjustmentState[uIdx] is the TPC adjustment state for the uIdx-th active UE in the coordinated cell group. 
    
    uint8_t*   srsPowerHeadroomReport = nullptr; 
    // array of the power headroom reports for all active UEs in the coordinated cell group.
    // format: one dimensional array. array size: nActiveUe
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
    // srsPowerHeadroomReport[uIdx] is the power headroom report for the uIdx-th active UE in the coordinated cell group. 
};
 
struct cumacSrsCellGrpPrms { // coordinated cell group information
    uint16_t    nMaxActUePerCell; // Maximum number of active UEs per cell. 
    uint16_t    nActiveUe; // Total number of active UEs in the coordinated cell group.
    uint16_t    nCell; // Total number of coordinated cells. 
    uint16_t    nSymbsPerSlot; // Total number of symbols per slot
    uint8_t     nBsAnt; // The number of antennas in BS
    uint8_t     srsSchedulingSel; //0-Round-robin scheduling; 1-more advanced scheduling only valid for 64TR
    
    //----------------- data buffers -----------------
    uint16_t*   cellId = nullptr;
    // coordinated cell indexes
    // format: one dimensional array for coordinated cell indexes (0, 1, ... nCell-1)
    // Currently only support coordinated cell indexes starting from 0
    
    uint8_t*    cellAssocActUe = nullptr;
    // cell-UE association profile for all active UEs in the coordinated cells
    // format: one dimensional array, array size = nCell*nActiveUe
    // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index, 
    // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
    // cellAssocActUe[cIdx*nActiveUe + uIdx] == 1 means the uIdx-th active UE is associated to cIdx-th coordinated cell, 0 otherwise
};

struct cumacSrsSchdSol { // multi-cell SRS scheduling solutions
    uint16_t*   nSrsScheduledUePerCell;
    // array of the number of UEs in the coordinated cell group scheduled for SRS transmission in the current TTI.
    // format: one-dimensional array. array size: nCell.
    // Denote cIdx = 0, 1, ..., nCell-1 as the index in the array. 
    // nSrsScheduledUePerCell[cIdx] is the number of UEs scheduled for SRS transmission in the uIdx-th cell. 

    uint16_t*  srsTxUe = nullptr;
    // array of indexes of active UEs in the coordinated cell group scheduled for SRS transmission in the current TTI.
    // format: one-dimensional array. array size: nMaxActUePerCell*nCell.
    // Denote idx = 0, 1, ..., nMaxActUePerCell*nCell-1 as the index in the array. 
    // srsTxUe[cIdx*nMaxActUePerCell+uIdx] is the active UE index of the uIdx-th UE in the cIdx-th cell scheduled for SRS transmission. 

    uint16_t*   srsRxCell = nullptr;
    // array of indexes of cells in the coordinated cell group that are scheduled to receive the SRS TX signals (from UEs specified by the srsTxUe array) in the current TTI.
    // format: one dimensional array. array size: nActiveUe.
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsRxCell[idx] is the index of the cell that is scheduled to receive the SRS TX signal from the idx-th active UE.

    uint8_t*    srsNumSymb = nullptr;
    // array of numbers of SRS symbols of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe.
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array.. 
    // srsNumSymb[idx] is the number of SRS symbols for the idx-th active UE.

    uint8_t*    srsTimeStart  = nullptr;
    // array of starting symbol positions of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array.
    // srsTimeStart[idx] is the starting symbol position for the idx-th active UE.

    uint8_t*    srsNumRep  = nullptr;
    // array of repetition factors of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsNumRep[idx] is the repetition factor for the idx-th active UE.

    uint8_t*    srsConfigIndex  = nullptr;
    // array of SRS bandwidth config indexes C_SRS of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsConfigIndex[idx] is the SRS bandwidth config index for the idx-th active UE. 
    
     uint8_t*    srsBwIndex   = nullptr;
    // array of SRS bandwidth indexes B_SRS of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsBwIndex[idx] is the SRS bandwidth index for the idx-th active UE. 
    
     uint8_t*    srsCombSize    = nullptr;
    // array of SRS transmission comb sizes K_TC of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsCombSize[idx] is the SRS transmission comb size for the idx-th active UE.
    
    uint8_t*    srsCombOffset    = nullptr;
    // array of SRS transmission comb offsets kbar_TC of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsCombOffset[idx] is the SRS transmission comb offset for the idx-th active UE.
    
    uint8_t*    srsFreqStart     = nullptr;
    // array of SRS frequency domain positions of the SRS in RBs of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsFreqStart[idx] is the frequency domain starting position for the idx-th active UE.
    
    uint16_t*    srsFreqShift     = nullptr;
    // array of SRS frequency domain shifts of the SRS in RBs of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsFreqStart[idx] is the frequency domain shift for the idx-th active UE.
    
    uint8_t*    srsFreqHopping     = nullptr;
    // array of SRS frequency hopping options of the SRS in RBs of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsFreqStart[idx] is the frequency hopping option for the idx-th active UE; frequency hopping is not supported currently.
    
    uint16_t*    srsSequenceId     = nullptr;
    // array of SRS sequence IDs n^SRS_ID of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsSequenceId[idx] is the SRS sequence ID for the idx-th active UE.
    
    uint8_t*    srsGroupOrSequenceHopping     = nullptr;
    // array of SRS hopping configurations of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsGroupOrSequenceHopping[idx] is the SRS hopping configuration for the idx-th active UE. 
    
    uint8_t*    srsCyclicShift     = nullptr;
    // array of SRS cyclic shift configurations of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsCyclicShift[idx] is the SRS cyclic shift configuration for the idx-th active UE.

    float*      srsTxPwr = nullptr;
    // array of SRS transmission powers of the active UEs in the coordinated cell group that are scheduled for SRS transmission in the current TTI.
    // format: one dimensional array. Array size: nActiveUe
    // Denote idx = 0, 1, ..., nActiveUe-1 as the index in the array. 
    // srsTxPwr[idx] is the SRS transmission power for the idx-th active UE.    
        
};

// cuMAC SRS scheduler dynamic descriptor
typedef struct mcSrsSchedulerDynDescr{
    //----------------- input buffers ----------------- 
    uint16_t*    cellId = nullptr;
    uint8_t*     cellAssocActUe = nullptr;
    float*       srsWbSnr = nullptr;
    float*       srsWbSnrThreshold = nullptr;
    float*       srsWidebandSignalEnergy = nullptr;
    int8_t*      newDataActUe = nullptr;
    uint32_t*    srsLastTxCounter = nullptr; 
    uint8_t*     srsNumAntPorts = nullptr;
    uint8_t*     srsResourceType = nullptr;
    uint8_t*     muMimoInd = nullptr;
    uint16_t**   sortedUeList = nullptr;
    float*       srsTxPwrMax = nullptr;
    float*       srsPwr0 = nullptr;
    float*       srsPwrAlpha = nullptr;
    uint8_t*     srsTpcAccumulationFlag = nullptr;
    float*       srsPowerControlAdjustmentState = nullptr; 
    uint8_t*     srsPowerHeadroomReport = nullptr;
    
    //----------------- output buffers ----------------- 
    uint16_t*  nSrsScheduledUePerCell = nullptr;
    uint16_t*  srsTxUe = nullptr;
    uint16_t*  srsRxCell = nullptr;
    uint8_t*   srsNumSymb = nullptr;
    uint8_t*   srsTimeStart = nullptr;
    uint8_t*   srsNumRep = nullptr;
    uint8_t*   srsConfigIndex = nullptr;
    uint8_t*   srsBwIndex = nullptr;
    uint8_t*   srsCombSize = nullptr;   
    uint8_t*   srsCombOffset = nullptr;
    uint8_t*   srsFreqStart = nullptr;
    uint16_t*  srsFreqShift = nullptr;
    uint8_t*   srsFreqHopping = nullptr;
    uint16_t*  srsSequenceId = nullptr;
    uint8_t*   srsGroupOrSequenceHopping = nullptr;
    uint8_t*   srsCyclicShift = nullptr;
    float*     srsTxPwr = nullptr; 

    //----------------- parameters -----------------
    uint16_t    nMaxActUePerCell; // Maximum number of active UEs per cell. 
    uint16_t    nActiveUe; // Total number of active UEs in the coordinated cell group.
    uint16_t    nCell; // Total number of coordinated cells. 
    uint16_t    nSymbsPerSlot; //Total number of symbols per slot 
    
} mcSrsSchedulerDynDescr_t;

class multiCellSrsScheduler {
public:
    // default constructor
    multiCellSrsScheduler();

    // desctructor
    ~multiCellSrsScheduler();

    multiCellSrsScheduler(multiCellSrsScheduler const&)            = delete;
    multiCellSrsScheduler& operator=(multiCellSrsScheduler const&) = delete;

    // setup() function for per-TTI algorithm execution
    void setup(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
               cumacSrsCellGrpPrms*        srsCellGrpPrms,
               cumacSchdSol*               schdSol, 
               cumacSrsSchdSol*            srsSchdSol,
               cudaStream_t                strm); // requires externel synchronization

    // run() function for per-TTI algorithm execution
    void run(cudaStream_t strm);
    
    // CPU scheduler served as GPU reference
    void cpuSrsSchedulerTpc(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
                            cumacSrsSchdSol*            srsSchdSol,
                            uint16_t                    ueIdx,
                            float                       W_SRS_LAST);
    
    void cpuScheduler_v0(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
                         cumacSrsCellGrpPrms*        srsCellGrpPrms,
                         cumacSrsSchdSol*            srsSchdSol);
    
    void cpuScheduler_v1(cumacSrsCellGrpUeStatus*    srsCellGrpUeStatus,
                         cumacSrsCellGrpPrms*        srsCellGrpPrms,
                         cumacSchdSol*               schdSol, 
                         cumacSrsSchdSol*            srsSchdSol);

    // parameter/data buffer logging function for debugging purpose
    void debugLog(); // for debugging only, printing out dynamic descriptor parameters

private:
    // dynamic descriptors
    std::unique_ptr<mcSrsSchedulerDynDescr_t> pCpuDynDesc;
    mcSrsSchedulerDynDescr_t* pGpuDynDesc;

    // CUDA kernel parameters
    uint16_t numThrdBlk;
    uint16_t numThrdPerBlk;

    dim3 gridDim;
    dim3 blockDim;

    // launch configuration structure
    std::unique_ptr<launchCfg_t> pLaunchCfg;

    void kernelSelect(uint8_t kernel_version_sel);
};

typedef struct multiCellSrsScheduler*       mcSrsSchedulerHndl_t;

static __global__ void multiCellSrsSchedulerKernel(mcSrsSchedulerDynDescr_t* pDynDescr);
}