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

#include "cuda_fp16.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cuda_bf16.h"

// cuMAC namespace
namespace cumac {
    // defined constants
    constexpr uint16_t maxNumCoorCells_             = 20;
    constexpr uint16_t maxNumActUePerCell_          = 1024; // must be power of 2, max 1024
    constexpr uint16_t maxNumUegPerCell_            = 128; // must be power of 2, max 128   
    constexpr uint16_t maxNumSchdUePerCellTTI_      = 16;
    constexpr uint16_t maxNumUeSchdPerCell_         = 16;
    constexpr uint16_t maxNumLayerPerGrpDL_         = 16;
    constexpr uint16_t maxNumLayerPerGrpUL_         = 8;
    constexpr uint16_t totNumPuschDmrsPort_         = 8;
    constexpr uint16_t totNumPdschDmrsPort_         = 8;
    constexpr uint16_t maxNumBsAnt_                 = 64;
    constexpr uint16_t maxNumUeAnt_                 = 4;
    constexpr uint16_t maxNumPrgPerCell_            = 273;
    constexpr uint16_t maxNumUeForGrpPerCell_       = 128; // must be power of 2, max 128
    constexpr uint16_t maxNumUeGrpSchdPerCell_      = 16;
    constexpr uint16_t maxNumLayerPerUe_            = 4;
    constexpr uint16_t minPow2MaxNumUeGrpPercell_   = 16; // smallest power of 2 >= MaxNumUeGrpSchdPerCell_

    // multi-cell MU-MIMO UE grouping list  
    struct multiCellMuGrpList {
        uint16_t* numUeInGrp = nullptr;
        // array of number of UEs in each UEG in the coordinated cell group
        // format: one dimensional array. array size: maxNumCoorCells_*maxNumUegPerCell_
        // denote cIdx = 0, 1, ... maxNumCoorCells_-1 as the coordinated cell index
        // denote uegIdx = 0, 1, ... maxNumUegPerCell_-1 as the UEG index
        // numUeInGrp[cIdx*maxNumUegPerCell_ + uegIdx] is the number of UEs in the uegIdx-th UEG in the cIdx-th coordinated cell

        uint16_t* ueId = nullptr;
        // array of UE IDs for each UEG in the coordinated cell group
        // format: one dimensional array. array size: maxNumCoorCells_*maxNumUegPerCell_*maxNumLayerPerGrpDL_
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index
        // denote uegIdx = 0, 1, ... maxNumUegPerCell_-1 as the UEG index
        // denote uIdx = 0, 1, ... maxNumLayerPerGrpDL_-1 as the UE index
        // ueId[cIdx*maxNumUegPerCell_*maxNumLayerPerGrpDL_ + uegIdx*maxNumLayerPerGrpDL_ + uIdx] is the uIdx-th UE's active UE ID in the uegIdx-th UEG in the cIdx-th coordinated cell
        // 0xFFFF indicates an invalid element

        int16_t*  subbandId = nullptr;
        // array of subband IDs for each UEG in the coordinated cell group  
        // format: one dimensional array. array size: maxNumCoorCells_*maxNumUegPerCell_
        // denote cIdx = 0, 1, ... maxNumCoorCells_-1 as the coordinated cell index
        // denote uegIdx = 0, 1, ... maxNumUegPerCell_-1 as the UEG index
        // subbandId[cIdx*maxNumUegPerCell_ + uegIdx] is the subband ID for the uegIdx-th UEG in the cIdx-th coordinated cell
        // -1 indicates no subband allocation for a UEG
    };

    // cuMAC API data structures ////////////////////////////////////////////////////
    struct cumacCellGrpUeStatus { // per-UE information
        //----------------- data buffers -----------------
        uint32_t*   lastSchdSlotActUe = nullptr;
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // last scheduled slot index for each active UE in the coordinated cells
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cells
        // lastSchdSlotActUe[uIdx] is the last scheduled slot index for the uIdx-th active UE in the coordinated cells
        // range of each element: [0, 0xFFFFFFFE]
        // 0xFFFFFFFF indicates a UE has not been scheduled yet 
        // ** should be maintained throughout the session where cuMAC modules are being used    
        
        float*      ueTxPow = nullptr; 
        // array of each active UE’s total uplink transmit power (Watt) across all transmit antennas in the coordinated cell group.
        // format: one dimensional array. array size: nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells.
        // ueTxPow[uIdx] is the uIdx-th active UE’s total transmit power in Watt.
        
        float*      noiseVarActUe = nullptr; 
        // array of each active UE’s per-Rx-antenna noise variance (Watt) in the coordinated cell group.
        // format: one dimensional array. array size: nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells.
        // noiseVarActUe[uIdx] is the uIdx-th active UE’s per-Rx-antenna noise variance in Watt.

        float*      avgRates = nullptr; 
        //* (For 4TR SU-MIMO only)
        // per-UE long-term average data rates of the selected UEs per TTI in the coordinated cells 
        // format: one dimensional array, array size = nUe
        // denote uIdx = 0, 1, ..., nUe-1 as the selected UE index in the coordinated cells
        // avgRates[uIdx] is the uIdx-th selected UE's long-term average data rate

        float*      avgRatesActUe = nullptr; 
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // per-UE long-term average data rates of all active UEs in the coordinated cells 
        // ** should be maintained throughout the session where cuMAC modules are being used
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // avgRatesActUe[uIdx] is the uIdx-th active UE's long-term average data rate

        int8_t*     cqiActUe = nullptr; 
        // array of the most recent reported CQI levels of all active UEs in the coordinated cells 
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // cqiActUe[uIdx] is the uIdx-th active UE's most recent reported CQI level
        // -1 indicates an invalid element
        // ! If no memory is allocated for this buffer, the SRS-based MCS selection is used. Otherwise the CQI-based MCS selection is used.

        int8_t*     riActUe = nullptr; 
        // array of the most recent reported RI values of all active UEs in the coordinated cells 
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // riActUe[uIdx] is the uIdx-th active UE's most recent reported RI value
        // ** For each active UE uIdx, riActUe[uIdx] should be initialized to 1
        // -1 indicates an invalid element
        // ! If no memory is allocated for this buffer, the SRS-based layer selection is used. Otherwise the RI-based layer selection is used.
        
        int8_t*     pmiActUe = nullptr; 
        // array of the most recent reported PMI values of all active UEs in the coordinated cells 
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // pmiActUe[uIdx] is the uIdx-th active UE's most recent reported PMI value
        // -1 indicates an invalid element
        // ! assuming that for each active UE uIdx, cqiActUe[uIdx], riActUe[uIdx] and pmiActUe[uIdx] are all from the UE's most recent CSI report

        uint16_t*   prioWeightActUe = nullptr;
        //* (For 4TR SU-MIMO only)
        // Priority weights of all active UEs in the coordinated cells 
        // For priority-based UE selection
        // ** used/accessed by both the priority-based Round Robin UE selection and Round Robin PRG allocation modules
        // ** should be maintained throughout the session where cuMAC modules are being used
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // prioWeightActUe[uIdx] is the uIdx-th active UE's priority weight
        // ** For each active UE uIdx, prioWeightActUe[uIdx] should be initialized to 0
        // 0xFFFF indicates an invalid element

        uint32_t*   bufferSize = nullptr;
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // per-UE buffer depth in bytes of UEs
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the selected UE index in the coordinated cells
        // bufferSize[uIdx] is the buffer depth in bytes of the uIdx-th active UE

        int8_t*     tbErrLastActUe = nullptr;
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // TB decoding error indicators of all active UEs in the coordinated cells.
        // ** should be maintained throughout the session where cuMAC modules are being used
        // format: one dimensional array. Array size: nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cells.
        // tbErrLastActUe[uIdx] is the uIdx-th active UE's TB decoding error indicator:
        // -1 - the last transmission is not a new transmission (is a re-transmission) 
        // 0 – CRC pass 
        // 1 – CRC failure
        // *Note that if the last transmission of a UE is not a new transmission, tbErrLastActUe of that UE should be set to -1.

        int8_t*     tbErrLast = nullptr; 
        // TB decoding error indicators of the selected UEs's last transmission
        //* (for 4TR SU-MIMO)
        // format: one dimensional array, array size = nUe
        // denote uIdx = 0, 1, ..., nUe-1 as the selected UE index in the coordinated cells
        // tbErrLast[uIdx] is the uIdx-th selected UE's TB decoding error indicator
        //* (For 64TR MU-MIMO)
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cells.
        // tbErrLast[uIdx] is the uIdx-th active UE's TB decoding error indicator:
        // -1 - the last transmission is not a new transmission (is a re-transmission) 
        // 0 – CRC pass 
        // 1 – CRC failure
        // *Note that if the last transmission of a UE is not a new transmission, tbErrLast of that UE should be set to -1.

        //----------------- HARQ related parameter fields -----------------
        // HARQ-based scheduling currently only supported for type-1 PRG allocation
        int8_t*     newDataActUe = nullptr; 
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // Indicators of initial transmission/retransmission for all active UEs
        // ** should be maintained throughout the session where cuMAC modules are being used
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // newDataActUe[uIdx] is the indicator of initial transmission/retransmission for the uIdx-th active UE in the coordinated cells
        // 0 - retransmission, 1 - new data/initial transmission
        // -1 indicates an invalid element

        int16_t*    allocSolLastTx = nullptr; 
        // The PRG allocation solution of the last transmission of the selected/scheduled UEs in the coordinated cells
        // Currently only support type-1 PRG allocation
        //* (For 4TR SU-MIMO)
        // format: one dimensional array, array size = 2*nUe
        // Two elements per selected UE, 1st element represents the starting PRG index and the 2nd element represents the ending PRG index plus one
        // -1 indicates that a given UE is not being allocated to any PRG
        // Using the UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure
        //* (For 64TR MU-MIMO)
        // format: one dimensional array. Array size: 2*nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells. 
        // allocSolLastTx[2*uIdx] is the starting PRG index of the uIdx-th active UE’s last transmission. 
        // allocSolLastTx[2*uIdx + 1] is the ending PRG index of the uIdx-th active UE’s last transmission plus one. 
        // -1 indicates that a given UE is not being allocated to any PRG. 

        int16_t*    mcsSelSolLastTx = nullptr; 
        // MCS selection solution of the last transmission of the selected/scheduled UEs in the coordinated cells
        //* (For 4TR SU-MIMO)
        // format: one dimensional array, array size = nUe
        // denote uIdx = 0, 1, ..., nUe-1 as the 0-based UE index for the selected UEs in the coordinated cells
        // mcsSelSol[uIdx] indicates the MCS level for the uIdx-th selected UE in the coordinated cells.
        // Currently only support Table 5.1.3.1-2: MCS index table 2, 3GPP TS 38.214
        // Value of each element: 0, 1, ..., 27
        // -1 indicates an invalid element
        // Using the UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure
        //* (For 64TR MU-MIMO)
        // format: one dimensional array. Array size: nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells. 
        // mcsSelSolLastTx[uIdx] is the MCS level of the uIdx-th active UE’s last transmission. 
        // Range of each element: 
        // 0, 1, …, 27 (Currently only support Table 5.1.3.1-2: MCS index table 2, 3GPP TS 38.214). 
        // -1 indicates an element is invalid. 

        uint8_t*    layerSelSolLastTx = nullptr; 
        // layer selection solution of the last transmission of the selected/scheduled UEs in the coordinated cells
        //* (For 4TR SU-MIMO)
        // format: one dimensional array, array size = nUe
        // Each element represents the number of layers selected for a UE 
        // Assumption's that the selected layers are the ones with the largest singular values
        // 0xFF indicates an invalid element
        // Using the UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure
        //* (For 64TR MU-MIMO)
        // format: one dimensional array. Array size: nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells.
        // layerSelSolLastTx[uIdx] is the number of layers of the uIdx-th active UE’s last transmission.
        // Range of each element: 0, 1, …, nUeAnt-1.
        // 0xFF (255) indicates an element is invalid.

        float*      beamformGainCurrTx = nullptr;
        //* (For 64TR MU-MIMO)
        // per-UE beamforming gains in dB of all active UEs' current scheduled transmissions in the coordinated cell group
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // beamformGainCurrTx[uIdx] is the uIdx-th active UE's beamforming gain 
        // Initialize beamformGainCurrTx to -100.0 dB for all UEs
        // -100.0 indicates an element is invalid

        float*      bfGainPrgCurrTx = nullptr;
        //* (For 64TR MU-MIMO)
        // per-UE and per-PRG beamforming gains in dB of all active UEs' current scheduled transmissions in the coordinated cell group
        // format: one dimensional array, array size = nActiveUe*nPrbGrp
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // bfGainPrgCurrTx[uIdx*nPrbGrp + prgIdx] is the uIdx-th active UE's beamforming gain for the prgIdx-th PRG in the current scheduled transmission   
        // Initialize bfGainPrgCurrTx to -100.0 dB for all UEs and all PRGs
        // -100.0 indicates an element is invalid

        float*      beamformGainLastTx = nullptr;
        //* (For 64TR MU-MIMO)
        // per-UE beamforming gains in dB of all active UEs' last scheduled transmissions in the coordinated cell group
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // beamformGainLastTx[uIdx] is the uIdx-th active UE's beamforming gain 
        // Initialize beamformGainLastTx to -100.0 dB for all UEs
        // -100.0 indicates an element is invalid  
    };

    struct cumacCellGrpPrms { // coordinated cell group information
        //----------------- parameters -----------------
        uint8_t     dlSchInd = 1;
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // DL/UL scheduling indicator
        // 0 - UL scheduling
        // 1 - DL scheduling
        // default is 1 for DL scheduling

        uint8_t     harqEnabledInd = 1;
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // HARQ enabled indicator
        // 0 - HARQ disabled
        // 1 - HARQ enabled
        // default is 1 for HARQ enabled
        
        uint16_t    nCell; 
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // total number of cells in the coordinated cell group
        // range: [1, maxNumCoorCells_]

        uint16_t    nUe; 
        //* (for 4TR SU-MIMO)
        // total number of selected/scheduled UEs for all coordinated cells per TTI
        // nUe = nCell*numUeSchdPerCellTTI 
        // it is the upperbound for the number of selected/scheduled UEs for all coordinated cells when numUeSchdPerCellTTIArr is populated (not equal to nullptr)

        uint16_t    nActiveUe; 
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // total number of active UEs in all coordinated cells
        // range: [0, maxNumCoorCells_*maxNumActUePerCell_]
        
        uint8_t     numUeSchdPerCellTTI = 16; 
        //* (for 4TR SU-MIMO)
        // number of UEs selected/scheduled per TTI per cell when numUeSchdPerCellTTIArr is not populated (assigned to nullptr)
        // otherwise, it is the upperbound for the number of UEs that can be selected/scheduled per TTI per cell
        // current assumption's that 6 <= numUeSchdPerCellTTI <= 16. nUe = numUeSchdPerCellTTI*nCell
        //* (for 64TR MU-MIMO)
        // maximum total number of SU-MIMO and MU-MIMO UEs scheduled per TTI per cell 
        // current assumption's that 6 <= numUeSchdPerCellTTI <= 16.

        uint16_t    numUeForGrpPerCell = 16;
        //* (for 64TR MU-MIMO)
        // number of UEs considered for MU-MIMO UE grouping per cell

        uint8_t     nMaxLayerPerUeSuDl = 4;
        //* (for 64TR MU-MIMO)
        // maximum number of layers per UE for SU-MIMO DL   

        uint8_t     nMaxLayerPerUeSuUl = 4;
        //* (for 64TR MU-MIMO)
        // maximum number of layers per UE for SU-MIMO UL

        uint8_t     nMaxLayerPerUeMuDl = 2;
        //* (for 64TR MU-MIMO)
        // maximum number of layers per UE for MU-MIMO DL

        uint8_t     nMaxLayerPerUeMuUl = 2;
        //* (for 64TR MU-MIMO)
        // maximum number of layers per UE for MU-MIMO UL   

        uint8_t     nMaxUePerGrpUl = 8;
        //* (for 64TR MU-MIMO)
        // maximum number of UEs per UEG for UL

        uint8_t     nMaxUePerGrpDl = 16;
        //* (for 64TR MU-MIMO)
        // maximum number of UEs per UEG for DL

        uint8_t     nMaxLayerPerGrpUl = 8;
        //* (for 64TR MU-MIMO)
        // maximum number of layers per UEG for UL

        uint8_t     nMaxLayerPerGrpDl = 16;
        //* (for 64TR MU-MIMO)
        // maximum number of layers per UEG for DL

        uint8_t     nMaxUegPerCellDl = 4;
        //* (for 64TR MU-MIMO)
        // maximum number of UEGs per cell for DL
        // range: [1, 16]

        uint8_t     nMaxUegPerCellUl = 4;
        //* (for 64TR MU-MIMO)
        // maximum number of UEGs per cell for UL   
        // range: [1, 16]

        float       muGrpSrsSnrMaxGap = 100.0;
        //* (for 64TR MU-MIMO)
        // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG
        // default is 6.0 dB. 
        // set to a large value, e.g. 100.0 dB to disable the SRS SNR gap constraint for MU-MIMO UE grouping

        float       muGrpSrsSnrSplitThr = -100.0;
        //* (for 64TR MU-MIMO)
        // Threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately
        // default is 0 dB. 
        // -100.0 indicates that the SRS SNR range is not split for MU-MIMO grouping.   

        uint8_t     ueGrpMode = 0;
        //* (for 64TR MU-MIMO)
        // UE grouping mode
        // 0 - dynamic UE grouping per TTI
        // 1 - flag-triggered UE grouping (controlled by the muUeGrpTrigger flag in cumacCellGrpPrms)
        // default is 0 for dynamic UE grouping per TTI

        uint8_t     muGrpUpdate = 0;
        //* (for 64TR MU-MIMO)
        // trigger for performing MU-MIMO UE grouping in the current TTI
        // 0 - not triggering UE grouping in the current TTI
        // 1 - triggering UE grouping in the current TTI
        // default is 0
        
        uint8_t     semiStatFreqAlloc = 0;
        //* (For 64TR MU-MIMO)
        // indication for whether or not to enable semi-static subband allocation for SU UEs/MU UEGs
        // 0 - disable semi-static subband allocation
        // 1 - enable semi-static subband allocation
        // default is 0 to disable semi-static subband allocation

        uint8_t     bfPowAllocScheme = 1;
        // power allocation scheme for beamforming weights computation
        // * currently only support Scheme 0 and 1
        // 0 - equal power allocation for RX side received per-layer beamforming gain
        // 1 - equal power allocation for TX side per-layer beamforming gain
        // 2 - water-filling power allocation for per-layer beamforming gain
        // 3 - water-filling power allocation for per-UE beamforming gain
        // default is 1 

        uint8_t     mcsSelLutType = 0;
        //* (for both 4TR SU-MIMO and 64TR MU-MIMO)
        // MCS selection look-up table type
        // 0 - LUT L1T1B024
        // 1 - LUT L1T1B050PRGS01_GTC25
        // default is 0 
        // Currently only support Table 5.1.3.1-2: MCS index table 2, 3GPP TS 38.214

        float       mcsSelSinrCapThr = 25.99;
        //* (for both 4TR SU-MIMO and 64TR MU-MIMO)
        // SINR capping threshold for MCS selection
        // default is the minimum required SINR for MCS 27, i.e. 25.99 dB. May add a margin to this default value for performance tuning.  

        uint8_t     mcsSelCqi = 0;
        //* (for both 4TR SU-MIMO and 64TR MU-MIMO)
        // whether to use wideband CQI or wideband SINR for MCS selection
        // 0 - use wideband SINR for MCS selection
        // 1 - use wideband CQI for MCS selection
        // default is 0 to use wideband SINR for MCS selection

        uint16_t    nMaxActUePerCell = maxNumActUePerCell_; // Maximum number of active UEs per cell. Default value is maxNumActUePerCell
        uint16_t    nPrbGrp; // total number of PRGs
        uint8_t     nBsAnt; // Each RU’s number of TX & RX antenna ports. Value: 4 or 64
        uint8_t     nUeAnt; // Each active UE’s number of TX & RX antenna ports. Value: 2, 4. Assumption's that nUeAnt <= nBsAnt
        float       W; // frequency bandwidth (Hz) of a PRG: 12 * SCS * # of PRBs per PRG 
        float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power
        float       Pt_Rbg; // NA
        float       Pt_rbgAnt; // NA
        uint8_t     precodingScheme; // precoder type: 0 - no precoding, 1 - SVD precoding
        uint8_t     receiverScheme; // receiver type: only support 1 - MMSE-IRC
        uint8_t     allocType; // PRB allocation type: 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
        float       betaCoeff = 1.0; // coefficient for balancing cell-center and cell-edge UEs' performance in multi-cell scheduling. Default value is 1.0
        float       sinValThr = 0.1; // (For 4TR SU-MIMO only) singular value threshold for layer selection, value is in (0, 1). Default value is 0.1
        float       corrThr = 0.5; // channel vector correlation value threshold for layer selection,  value is in (0, 1). Default value is 0.5
        uint16_t    prioWeightStep = 100; // step size for UE priority weight increment per TTI if UE does not get scheduled. Default is 100
        float       muCoeff = 1.5; // Coefficient for prioritizing UEs selected for MU-MIMO transmissions.
        float       zfCoeff = 1e-16; // Scalar coefficient used for regularizing the zero-forcing beamformer.
        float       srsSnrThr = -3.0; // Threshold on SRS reported SNR in dB for determining the feasibility of MU-MIMO transmission. Value: a real number in unit of dB. Default value is -3.0 (dB).
        float       chanCorrThr = 0.7; // Threshold for the squared channel vector correlation value in UE grouping. Value: a real number between 0 and 1.0. Default: 0.1
        
        //----------------- data buffers -----------------
        uint32_t*   currSlotIdxPerCell = nullptr;
        //* (for both 4TR SU-MIMO and 64TR MU-MIMO)
        // current slot index for each cell in the coordinated cell group
        // format: one dimensional array, array size = nCell
        // denote cIdx = 0, 1, ..., nCell-1 as the coordinated cell index
        // currSlotIdxPerCell[cIdx] is the current slot index for the cIdx-th cell
        // range of each element: [0, 0xFFFFFFFE]
        // 0xFFFFFFFF indicates an invalid element  

        uint16_t*   cellId = nullptr;
        //* (for 4TR SU-MIMO)
        // coordinated cell indexes
        // format: one dimensional array for coordinated cell indexes (0, 1, ... nCell-1)
        // Currently only support coordinated cell indexes starting from 0

        uint8_t*    cellAssoc = nullptr;
        //* (for 4TR SU-MIMO)
        // cell-UE association profile for the selected UEs per TTI in the coordinated cells
        // format: one dimensional array, size: nCell*nUe
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index
        // denote uIdx = 0, 1, ..., nUe-1 as the selected UE index in the coordinated cells
        // cellAssoc[cIdx*nUe + uIdx] == 1 means the uIdx-th selected UE is associated with the cIdx-th coordinated cell, 0 otherwise
        // ** cellAssoc[cIdx*nUe + uIdx] should be set to 0 for all cIdx = 0, 1, ... nCell-1 if setSchdUePerCellTTI[uIdx] is 0xFFFF
        // ** using the global UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure 

        uint8_t*    numUeSchdPerCellTTIArr = nullptr;
        //* (for 4TR SU-MIMO)
        // array of the numbers of UEs scheduled per TTI for each cell in the coordinated cell group
        // format: one dimensional array, size: nCell
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index
        // numUeSchdPerCellTTIArr[cIdx] is the number of UEs scheduled per TTI for the cIdx-th cell
        // assumption's that numUeSchdPerCellTTIArr[cIdx] <= numUeSchdPerCellTTI for all cIdx = 0, 1, ... nCell-1

        uint8_t*    cellAssocActUe = nullptr;
        // cell-UE association profile for all active UEs in the coordinated cells
        // format: one dimensional array, array size = nCell*nActiveUe
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index, 
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // cellAssocActUe[cIdx*nActiveUe + uIdx] == 1 means the uIdx-th active UE is associated to cIdx-th coordinated cell, 0 otherwise

        float*      blerTargetActUe = nullptr;
        //* (for both 4TR SU-MIMO and 64TR MU-MIMO)
        // array of BLER targets for all active UEs in the coordinated cells
        // format: one dimensional array, array size = nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // blerTargetActUe[uIdx] is the BLER target for the uIdx-th active UE
        // default value is 0.1

        // floating type
        //* For 4TR SU-MIMO
        cuComplex*  estH_fr = nullptr; //  (FP32) estimated CFR channel coefficients per cell-UE link, per PRG, per Tx-Rx antenna pair for the selected UEs per TTI in the coordinated cells
        cuComplex*  estH_fr_actUe = nullptr; // estimated CFR channel coefficients per cell-UE link, per PRG, per Tx-Rx antenna pair for all active UEs in the coordinated cells
        cuComplex*  estH_fr_actUe_prd = nullptr;
        cuComplex*  prdMat = nullptr; 
        // DL precoding weights for all cells, per PRG, per pair of Tx antenna port and layer
        // format: one dimensional array, array size = nCell*nPrbGrp*nBsAnt*MaxNumLayerPerGrpDL_
        // denote cIdx = 0, 1, ..., nCell-1 as the coordinated cell index
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // denote antIdx = 0, 1, ..., nBsAnt-1 as the Tx antenna port index
        // denote layerIdx = 0, 1, ..., MaxNumLayerPerGrpDL_-1 as the layer index
        // prdMat[cIdx*nPrbGrp*nBsAnt*MaxNumLayerPerGrpDL_ + prgIdx*nBsAnt*MaxNumLayerPerGrpDL_ + antIdx*MaxNumLayerPerGrpDL_ + layerIdx] is the precoder weight from the layerIdx-th layer to the antIdx-th TX antenna port on the prgIdx-th PRG for the cIdx-th cell
        // un-used precoder weights entries are set to 0
        // prdMat is set to 0 for an un-allocated PRG in a cell
        
        cuComplex*  prdMat_actUe = nullptr; // precoder matrix coefficients per cell-UE link, per PRG, per symbol-Tx antenna pair for all active UEs in the coordinated cells
        cuComplex*  detMat = nullptr; // detector matrix coefficients per cell-UE link, per PRG, per symbol-Tx antenna pair for the selected/scheduled UEs
        cuComplex*  detMat_actUe = nullptr; // detector matrix coefficients per cell-UE link, per PRG, per symbol-Tx antenna pair for all active UEs in the coordinated cells
        
        float*      sinVal = nullptr; // singular values associated with each precoder matrix, per cell-UE link, per PRG, per layer (maximum number of layers is equal to the number of UE antennas) for the selected/scheduled UEs 
        //* (For 4TR SU-MIMO)
        // Array of singular values of the PRG level SRS channel estimates for all active UEs in the coordinated cell group. 
        // format: one-dimensional array, size: nActiveUe*nPrbGrp*nUeAnt
        // Per-UE, per-PRG, per-layer singular values obtained from the SVD of SRS channel matrices
        // For each active UE and on each PRG, the singular values are stored in descending order:
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the 0-based active UE index in the coordinated cells
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // denote layerIdx = 0, 1, ..., nUeAnt-1 as the layer index
        // sinVal[uIdx*nPrbGrp*nUeAnt + prgIdx*nUeAnt + layerIdx] is the UE uIdx's layerIdx-th largest singular value on PRG prgIdx

        float*      sinVal_actUe = nullptr; // singular values associated with each precoder matrix, per cell-UE link, per PRG, per layer (maximum number of layers is equal to the number of UE antennas) for all active UEs in the coordinated cells
        cuComplex** estH_fr_perUeBuffer = nullptr; // estimated CFR channel coefficients organized as per-UE buffers

        // half-precision
        __nv_bfloat162*    estH_fr_half = nullptr; // (FP16) estimated CFR channel coefficients per cell-UE link, per PRG, per Tx-Rx antenna pair for the selected UEs per TTI in the coordinated cells

        float*      postEqSinr = nullptr; // *** This SINR array will be used by multiple cuMAC modules throughput the simulation process. So it should be allocated at initialization phase and should not be allocated & freed per TTI
        //* (For 4TR SU-MIMO only)
        // per-PRG per-layer post-equalization SINRs for all active UEs in the coordinated cells
        // ** should be maintained throughout the session where cuMAC modules are being used
        // format: one dimensional array, array size = nActiveUe*nPrbGrp*nUeAnt
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // denote layerIdx = 0, 1, ..., nUeAnt-1 as the layer index
        // postEqSinr[uIdx*nPrbGrp*nUeAnt + prgIdx*nUeAnt + layerIdx] is the uIdx-th active UE's post-equalization SINR on the prgIdx-th PRG and the layerIdx-th layer

        float*      wbSinr = nullptr; 
        //* (For both 4TR SU-MIMO and 64TR MU-MIMO)
        // per-layer wideband SINRs for all active UEs in the coordinated cell group
        // * wbSinr is used for the most recent wideband SINR values from either SU-MIMO or MU-MIMO transmissions
        // format: one dimensional array, array size = nActiveUe*nUeAnt
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the global UE index for all active UEs in the coordinated cells
        // denote layerIdx = 0, 1, ..., nUeAnt-1 as the layer index
        // wbSinr[uIdx*nUeAnt + layerIdx] is the uIdx-th active UE's wideband SINR on the layerIdx-th layer
        // if a single wideband SINR is available to each UE, set all elements with layerIdx > 0 to 0;
        // Initialize wbSinr to MCS 0's required SINR for all UEs

        //----------------- HARQ related parameter fields -----------------
        // HARQ-based scheduling currently only supported for type-1 PRG allocation
        uint8_t**  prgMsk = nullptr;
        //* (For 4TR SU-MIMO only)
        // Per-cell bit map for the availability of each PRG for allocation
        // format: two-dimensional array, size: nCell X nPrbGrp
        // 1st dimension for 0-based coordinated cell indexes (0, 1, ..., nCell-1)
        // 2nd dimension for the per-PRG indicators of availability for allocation
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // prgMsk[cIdx][prgIdx] is the availability indicator for the prgIdx-th PRG in the cIdx-th coordinated cell
        // 0 - unavailable, 1 - available

        //----------------- AODT specific data buffers -----------------
        cuComplex*  prdMat_asim = nullptr;
        // Aerial Sim format SVD precoder array
        // SVD of DL/UL channel H = U*Sigma*V^H, precoder matrix is V matrix (right matrix of DL/UL channel H)
        // format: one-dimensional array, size: nUe*nPrbGrp*nBsAnt*nBsAnt (maximum size is used for mem allocate)
        // Per UE, per PRG, per data symbol-TX antenna pair SVD precoder matrix coefficients
        // denote uIdx = 0, 1, ..., nUe-1 as the 0-based UE index for the selected UEs in the coordinated cells
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // (for DL) denote antIdx = 0, 1, ..., nBsAnt-1 as the TX antenna port index
        // (for UL) denote antIdx = 0, 1, ..., nUeAnt-1 as the TX antenna port index
        // (for DL) denote symIdx = 0, 1, ..., nBsAnt-1 as the TX data symbol index
        // (for UL) denote symIdx = 0, 1, ..., nUeAnt-1 as the TX data symbol index
        // (for DL) prdMat_asim[uIdx*nPrbGrp*nBsAnt*nBsAnt + prgIdx*nBsAnt*nBsAnt + antIdx*nBsAnt + symIdx] 
        // (for UL) prdMat_asim[uIdx*nPrbGrp*nUeAnt*nUeAnt + prgIdx*nUeAnt*nUeAnt + antIdx*nUeAnt + symIdx] 
        // is the precoder weight of the uIdx-th selected UE on the prgIdx-th PRG and between the symIdx-th TX data symbol and the antIdx-th TX antenna port.
        // Using the UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure

        cuComplex*  detMat_asim = nullptr;
        // Aerial Sim format SVD detector array
        // SVD of DL/UL channel H = U*Sigma*V^H, detector matrix is U matrix (left matrix of DL/UL channel H)
        // (effective detector is U^H for DL and V^H for UL but the original U or V matrix is passed via detMat_asim)
        // format: one-dimensional array, size: nUe*nPrbGrp*nBsAnt*nBsAnt (maximum size is used for mem allocate)
        // Per UE, per PRG, per data symbol-RX antenna pair SVD detector matrix coefficients
        // denote uIdx = 0, 1, ..., nUe-1 as the 0-based UE index for the selected UEs in the coordinated cells
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // (for DL) denote symIdx = 0, 1, ..., nUeAnt-1 as the RX data symbol index
        // (for UL) denote symIdx = 0, 1, ..., nBsAnt-1 as the RX data symbol index
        // (for DL) denote antIdx = 0, 1, ..., nUeAnt-1 as the RX antenna port index
        // (for UL) denote antIdx = 0, 1, ..., nBsAnt-1 as the RX antenna port index
        // (for DL) detMat_asim[uIdx*nPrbGrp*nUeAnt*nUeAnt + prgIdx*nUeAnt*nUeAnt + symIdx*nUeAnt + antIdx]
        // (for UL) detMat_asim[uIdx*nPrbGrp*nBsAnt*nBsAnt + prgIdx*nBsAnt*nBsAnt + symIdx*nBsAnt + antIdx]
        // is the detector weight of the uIdx-th selected UE on the prgIdx-th PRG and between the symIdx-th RX data symbol and the antIdx-th RX antenna port.
        // Using the UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure

        float*      sinVal_asim = nullptr;
        // Aerial Sim format singular value array
        // format: one-dimensional array, size: nUe*nPrbGrp*nUeAnt
        // Per-UE, per-PRG, per-layer singular values obtained from SVD of CFR channel matrices
        // For each UE and on each PRG, the singular values are stored in descending order:
        // denote uIdx = 0, 1, ..., nUe-1 as the 0-based UE index for the selected UEs in the coordinated cells
        // denote prgIdx = 0, 1, ..., nPrbGrp-1 as the PRG index
        // denote layerIdx = 0, 1, ..., nUeAnt-1 as the layer index
        // sinVal_asim[uIdx*nPrbGrp*nUeAnt + prgIdx*nUeAnt + layerIdx] is the UE uIdx's layerIdx-th largest singular value on PRG prgIdx
        // Using the UE ID mapping setSchdUePerCellTTI in the cumacSchdSol structure

        ////////////////////////////////////////
        //* For 64TR MU-MIMO
        float*      bsTxPow = nullptr;
        // array of each RU’s total downlink transmit power (Watt) across all transmit antennas in the coordinated cell group.
        // format: one dimensional array, array size = nCell
        // denote cIdx = 0, 1, …, nCell-1 as the coordinated cell index.
        // bsTxPow[cIdx] is the total transmit power of the cIdx-th cell’s RU.

        cuComplex**  srsEstChan = nullptr; //  (FP32) estimated CFR channel coefficients per cell-UE link, per PRG, per Tx-Rx antenna pair for the selected UEs per TTI in the coordinated cells
        //* (For 64TR MU-MIMO only)
        // Per-cell array of the PRG level SRS channel estimates for the top-numUeForGrpPerCell UEs after UE sorting
        // UE indexing for each cell is based on the srsUeMap array defined below.  
        // SRS channel estimates are stored as the downlink channels (from RU to UE). 
        // UE transmit power for the SRS signal is excluded from the channel coefficient magnitude. 
        // format: two-dimensional array: the 1st dimension is for cells, and the 2nd dimension is for UEs, PRGs, and UE/RU antenna ports.  
        // array size: nCell X (nCell*numUeForGrpPerCell*nPrbGrp*nUeAnt*nBsAnt) 
        // denote cIdx = 0, 1, …, nCell-1 as the coordinated cell index.  
        // denote uIdx = 0, 1, …, nCell*numUeForGrpPerCell-1 as the UE index (local to the srsEstChan[cIdx] array).  
        // denote prgIdx = 0, 1, …, nPrbGrp-1 as the PRG index.  
        // denote ueAntIdx = 0, 1, …, nUeAnt as the UE antenna port index  
        // denote bsAntIdx = 0, 1, …, nBsAnt-1 as the RU antenna port index   
        // srsEstChan[cIdx][uIdx*nPrbGrp*nUeAnt*nBsAnt + prgIdx*nUeAnt*nBsAnt + ueAntIdx*nBsAnt + bsAntIdx] is the complex (downlink) channel coefficient between the cIdx-th cell and the UE with local ID uIdx on the prgIdx-th PRG, the ueAntIdx-th UE antenna port and the bsAntIdx-th RU antenna port. 

        int32_t**   srsUeMap = nullptr;
        //* (For 64TR MU-MIMO only)
        // Per-cell UE index mapping array for the SRS channel estimates stored in srsEstChan.
        // format: two-dimensional array: the 1st dimension is for cells, and the 2nd dimension is for UEs.
        // array size: nCell X nActiveUe
        // denote cIdx = 0, 1, …, nCell-1 as the coordinated cell index.  
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells. 
        // srsUeMap[cIdx][uIdx] is the uIdx-th active UE’s index in the srsEstChan[cIdx] array.
        // srsUeMap[cIdx][uIdx] == -1 means that the uIdx-th active UE’s SRS channel estimates are not available in the srsEstChan[cIdx] array.
        // it is required that srsUeMap gets updated for the top numUeForGrpPerCell-th active UEs in each cell according to sortedUeList after the MU-MIMO UE sorting module.

        float*      srsWbSnr = nullptr;
        //* (For 64TR MU-MIMO only) 
        // array of the reported wideband SNRs in dB measured within configured SRS bandwidth for all active UEs in the coordinated cell group.
        // format: one dimensional array. array size: nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group. 
        // srsWbSnr[uIdx] is the wideband SNR in dB measured within configured SRS bandwidth for the uIdx-th active UE in the coordinated cell group. 
        // set srsWbSnr[uIdx] to -100.0 (dB) when the uIdx-th active UE's SRS measurements are not available or refreshed. 
    };

    struct cumacSchdSol { // scheduling solutions
        //----------------- data buffers -----------------
        multiCellMuGrpList* muGrpList = nullptr;
        //* (For 64TR MU-MIMO only)
        // multi-cell MU-MIMO UE grouping list
        // format: a single pointer to the multiCellMuGrpList structure
        
        uint16_t*   ueOrderInGrp = nullptr;
        //* (For 64TR MU-MIMO only)
        // The order of each active UE in its MU-MIMO group
        // format: one dimensional array. array size: nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cells. 
        // ueOrderInGrp[uIdx] is the order of the uIdx-th active UE in its MU-MIMO group
        // range of each element: 0, 1, ..., MaxNumLayerPerGrpDL_-1
        // initialized to 0xFFFF    

        int16_t*    allocSol = nullptr; 
        // PRB group allocation solution for the selected UEs per TTI in the coordinated cells
        //* (For 4TR SU-MIMO)
        // format: one dimensional array; array size and content differ for type-0 and type-1 allocations
        // For type-0: array size = nCell*nPrbGrp, each element represents the selected UE index (0, 1, ..., nUe-1) that a PRG is allocated to in a coordinated cell
        // -1 indicates that a give PRG is not allocated to any UE in a given cell
        // For type-1: array size = 2*nUe, two elements per selected UE, 1st element represents the starting PRG index and the 2nd element represents the ending PRG index plus one
        // -1 indicates that a given UE is not being allocated to any PRG
        // Using the UE ID mapping setSchdUePerCellTTI
        //* (For 64TR MU-MIMO)
        // Currently only support type-1 PRB allocation
        // format: one dimensional array. Array size: 2*nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells. 
        // allocSol[2*uIdx] is the starting PRG index for the uIdx-th active UE. 
        // allocSol[2*uIdx + 1] is the ending PRG index for the uIdx-th active UE plus one. 
        // -1 indicates that a given UE is not being allocated to any PRG. 

        float*      pfMetricArr = nullptr; 
        //* (for 4TR SU-MIMO)
        // only applicable to type-1 PRB allocation, for storing computed PF metrices
        // array size = nCell * the minimum power of 2 that is no less than nPrbGrp*numUeSchdPerCellTTI
        // memory should be pre-allocated at initialization of cuMAC API

        uint16_t*   pfIdArr = nullptr; 
        //* (for 4TR SU-MIMO)
        // only applicable to type-1 PRB allocation, for storing indices (indicating PRB and UE indecies) of computed PF metrices
        // array size = nCell * the minimum power of 2 that is no less than nPrbGrp*numUeSchdPerCellTTI
        // memory should be pre-allocated at initialization of cuMAC API

        int16_t*    mcsSelSol = nullptr; 
        // MCS selection solution for the selected UEs per TTI in the coordinated cells
        //* (For 4TR SU-MIMO)
        // format: one dimensional array, array size = nUe
        // denote uIdx = 0, 1, ..., nUe-1 as the 0-based UE index for the selected UEs in the coordinated cells
        // mcsSelSol[uIdx] indicates the MCS level for the uIdx-th selected UE in the coordinated cells.
        // Currently only support Table 5.1.3.1-2: MCS index table 2, 3GPP TS 38.214
        // Value of each element: 0, 1, ..., 27
        // -1 indicates an invalid element
        // Using the UE ID mapping setSchdUePerCellTTI
        //* (For 64TR MU-MIMO)
        // format: one dimensional array. Array size: nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cells.
        // mcsSelSol[uIdx] indicates the MCS level for the uIdx-th active UE in the coordinated cells. 
        // Range of each element: 
        // 0, 1, …, 27 (Currently only support Table 5.1.3.1-2: MCS index table 2, 3GPP TS 38.214). 
        // -1 indicates an element is invalid. 

        uint16_t*   setSchdUePerCellTTI = nullptr; 
        // set of global IDs of the scheduled UEs per cell per TTI
        //* (for 4TR SU-MIMO)
        // format: one dimensional array, array size = nCell*numUeSchdPerCellTTI
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index, 
        // denote i = 0, 1, ..., numUeSchdPerCellTTI-1 as the i-th scheduled UE in a cell
        // setSchdUePerCellTTI[cIdx*numUeSchdPerCellTTI + i] is within {0, 1, ..., nActiveUe-1} and is the active UE ID of the i-th scheduled UE in the cIdx-th cell. 
        // 0xFFFF (65535) indicates an element is invalid (no scheduled UE). 
        //* (for 64TR MU-MIMO)
        // format: one dimensional array, array size = nCell*numUeForGrpPerCell
        // denote cIdx = 0, 1, ... nCell-1 as the coordinated cell index, 
        // denote i = 0, 1, ..., numUeForGrpPerCell-1 as the i-th scheduled UE in a cell
        // setSchdUePerCellTTI[cIdx*numUeForGrpPerCell + i] is within {0, 1, ..., nActiveUe-1} and is the active UE ID of the i-th scheduled UE in the cIdx-th cell. 
        // 0xFFFF (65535) indicates an element is invalid (no scheduled UE).

        uint8_t*    layerSelSol = nullptr; 
        // layer selection solution for the selected UEs per TTI in the coordinated cells
        //* (For 4TR SU-MIMO)
        // format: one dimensional array, array size = nUe
        // Each element represents the number of layers selected for a UE 
        // Assumption's that the selected layers are the ones with the largest singular values
        // 0xFF indicates an invalid element
        // Using the UE ID mapping setSchdUePerCellTTI
        //* (For 64TR MU-MIMO)
        // layer selection solution for each active UE in the coordinated cells.
        // format: one dimensional array. Array size: nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells.
        // layerSelSol[uIdx] is the number of layers selected for the uIdx-th active UE.
        // Range of each element: 0, 1, …, nUeAnt-1.
        // 0xFF (255) indicates an element is invalid (the corresponding active is not being scheduled for transmission).
        // *The selected layers are always corresponding to the UE TX/RX antenna ports starting from port 0. Note that for downlink transmissions, the UE side may use more RX antennas to improve receiver performance.

        /////////////////////////////////////////////////
        //* For 64TR MU-MIMO
        uint8_t*    muMimoInd = nullptr;
        //* (For 64TR MU-MIMO only)
        // indicators of MU-MIMO transmission feasibility for all active UEs in the coordinated cell group.
        // format: one dimensional array. array size: nActiveUe
        // denote uIdx = 0, 1, ..., nActiveUe-1 as the active UE index in the coordinated cell group.
        // muMimoInd[uIdx] is the indicator of MU-MIMO transmission feasibility for the uIdx-th active UE in the coordinated cell group.
        // muMimoInd[uIdx] == 1 means that UE uIdx is feasible for MU-MIMO. 
        // muMimoInd[uIdx] == 0 means that UE uIdx is not feasible for MU-MIMO (only feasible for SU-MIMO).

        uint16_t**  sortedUeList = nullptr;
        //* (For 64TR MU-MIMO only)
        // array of sorted UE list for each cell.
        // two-dimensional array. array size: nCell X nMaxActUePerCell
        // denote cIdx = 0, 1, …, nCell-1 as the coordinated cell index.  
        // denote uIdx = 0, 1, …, nMaxActUePerCell-1 as the UE rank in each cell.
        // sortedUeList[cIdx][uIdx] is the active UE index of the uIdx-th ranked UE in the cIdx-th cell.
        // 0xFFFF (65535) indicates an element is invalid. 

        uint8_t*    nSCID = nullptr;
        //* (For 64TR MU-MIMO only)
        // array of the DMRS sequence initialization parameter n_SCID assigned for each active UE in the coordinated cells.
        // format: one dimensional array. Array size = nActiveUe
        // denote uIdx = 0, 1, …, nActiveUe-1 as the active UE index in the coordinated cells.
        // nSCID[uIdx] is the uIdx-th active UE’s assigned n_SCID value
        // Range of each element: 0 -> 1
        // 0xFF (255) indicates an element is invalid.
    };

    // Outer-loop link adaptation (OLLA) data structure
    struct ollaParam {
        float delta_ini;
        float delta;
        float delta_up;
        float delta_down;
    };

    // Aerial Scheduler Acceleration API data structures ////////////////////////////////////////////////////
    struct MAC_SCH_CONFIG_REQUEST { // static scheduler configuration parameters, passed from L2 stack host to cuMAC-CP at the initialization phase (one-time pass)
        uint8_t     harqEnabledInd; // Indicator for whether HARQ is enabled
        uint8_t     mcsSelCqi; // Indicator for whether MCS selection is based on CQI or SINR
        uint8_t     nMaxCell; // A constant integer for the maximum number of cells in the cell group
        uint16_t    nMaxActUePerCell; // A constant integer for the maximum number of active UEs per cell
        uint8_t     nMaxSchUePerCell; // A constant integer for the maximum number of UEs that can be scheduled per TTI per cell
        uint16_t    nMaxPrg; // A constant integer for the maximum number of PRGs for allocation in each cell
        uint16_t    nPrbPerPrg; // A constant integer for the number of PRBs per PRG (PRB group)
        uint8_t     nMaxBsAnt; // A constant integer for the maximum number of BS antenna ports.  
        uint8_t     nMaxUeAnt; // A constant integer for the maximum number of UE antenna ports.  
        uint32_t    scSpacing; // Subcarrier spacing of the carrier. Value: 15000, 30000, 60000, 120000 (Hz)  
        uint8_t     allocType; // Indicator for type-0 or type-1 PRG allocation
        uint8_t     precoderType; // Indicator for the precoder type 
        uint8_t     receiverType; // Indicator for the receiver type
        uint8_t     colMajChanAccess; // Indicator for whether the estimated narrow-band SRS channel matrices are stored in column-major order or in row-major order
        float       betaCoeff; // Coefficient for adjusting the cell-edge UEs' performance in multi-cell scheduling
        float       sinValThr; // Singular value threshold for layer selection
        float       corrThr; // Channel vector correlation value threshold for layer selection
        float       mcsSelSinrCapThr; // SINR capping threshold for MCS selection
        uint8_t     mcsSelLutType; // MCS selection LUT type
        uint16_t    prioWeightStep; // Step size for UE priority weight increment per TTI if UE does not get scheduled. For priority-based UE selection
        float       blerTarget; // BLER target
    };

    struct MAC_SCH_TTI_REQUEST {
        uint16_t    cellID; // cell ID
        uint8_t     ULDLSch = 1; // Indication for UL/DL scheduling. Value - 0: UL scheduling, 1: DL scheduling
        uint16_t    nActiveUe; // total number of active UEs in the cell
        uint16_t    nSrsUe; // the number of UEs in the cell that have refreshed SRS channel estimates
        uint16_t    nPrbGrp; // the number of PRGs that can be allocated for the current TTI, excluding the PRGs that need to be reserved for HARQ re-tx's
        uint8_t     nBsAnt; // number of BS antenna ports
        uint8_t     nUeAnt; // number of UE antenna ports
        float       sigmaSqrd = 1.0; // noise variance

        // data buffer pointers
        uint16_t*   CRNTI = nullptr; // C-RNTIs of all active UEs in the cell
        uint16_t*   srsCRNTI = nullptr; // C-RNTIs of the UEs that have refreshed SRS channel estimates in the cell.
        uint8_t*    prgMsk = nullptr; // Bit map for the availability of each PRG for allocation
        float*      postEqSinr = nullptr; // array of the per-PRG per-layer post-equalizer SINRs of all active UEs in the cell
        float*      wbSinr = nullptr; // array of wideband per-layer post-equalizer SINRs of all active UEs in the cell
        cuComplex*  estH_fr = nullptr; // For FP32. array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
        cuComplex*  estH_fr_half = nullptr; // For FP16. array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
        cuComplex*  prdMat = nullptr; // array of the precoder/beamforming weights for all active UEs in the cell
        cuComplex*  detMat = nullptr; // array of the detector/beamforming weights for all active UEs in the cell
        float*      sinVal = nullptr; // array of the per-UE, per-PRG, per-layer singular values obtained from the SVD of the channel matrix
        float*      avgRatesActUe = nullptr; // array of the long-term average data rates of all active UEs in the cell
        uint16_t*   prioWeightActUe = nullptr; // For priority-based UE selection. Priority weights of all active UEs in the cell
        int8_t*     tbErrLastActUe = nullptr; // TB decoding error indicators of all active UEs in the cell
        int8_t*     newDataActUe = nullptr; // Indicators of initial transmission/retransmission for all active UEs in the cell
        int16_t*    allocSolLastTxActUe = nullptr; // The PRG allocation solution for the last transmissions of all active UEs in the cell
        int16_t*    mcsSelSolLastTxActUe = nullptr; // MCS selection solution for the last transmissions of all active UEs in the cell
        uint8_t*    layerSelSolLastTxActUe = nullptr; // Layer selection solution for the last transmissions of all active UEs in the cell
        int8_t*     cqiActUe = nullptr; // CQI values of all active UEs in the cell
    };

    struct MAC_SCH_TTI_RESPONSE {
        uint16_t*   setSchdUePerCellTTI = nullptr; // Set of IDs of the selected UEs for the cell
        int16_t*    allocSol = nullptr; // PRB group allocation solution for all active UEs in the cell
        int16_t*    mcsSelSol = nullptr; // MCS selection solution for all active UEs in the cell
        uint8_t*    layerSelSol = nullptr; // Layer selection solution for all active UEs in the cell
    };

    // cuMAC H5 TV saving/loading /////////////////////////////////////////////////
    struct cumacSchedulerParam {
        uint16_t    nUe;
        uint16_t    nCell; // number of coordinated cells
        uint16_t    totNumCell; // number of all cells in the network. (not needed if channel buffer only contains channels within coordinated cells)
        uint16_t    nPrbGrp;
        uint8_t     nBsAnt;
        uint8_t     nUeAnt; // assumption's that nUeAnt <= nBsAnt
        float       W;
        float       sigmaSqrd; // noise variance if channel is not normalized; 1/SNR if channel is normalized with transmit power, limitation: SNR (per antenna) should be <= 111 dB
        uint16_t    maxNumUePerCell; // maximum number of selected UEs per cell per TTI
        uint16_t    nMaxSchdUePerRnd; // maximum number of UEs per cell that can be scheduled per round
        float       betaCoeff; // coefficient for improving cell edge UEs' performance in multi-cell scheduling
        uint16_t    nActiveUe; // number of active UEs for all coordinated cells
        uint8_t     numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell        
        uint8_t     precodingScheme;
        uint8_t     receiverScheme;
        uint8_t     allocType;
        uint8_t     columnMajor;
        float       sinValThr; // singular value threshold for layer selection
        uint16_t    numUeForGrpPerCell;
        float       chanCorrThr;
        float       muCoeff;
        float       srsSnrThr;
        uint16_t    nMaxActUePerCell;
        float       zfCoeff;
        uint8_t     nMaxUePerGrpUl;
        uint8_t     nMaxUePerGrpDl;
        uint8_t     nMaxLayerPerGrpUl;
        uint8_t     nMaxLayerPerGrpDl;
        uint8_t     nMaxLayerPerUeSuUl;
        uint8_t     nMaxLayerPerUeSuDl;
        uint8_t     nMaxLayerPerUeMuUl; 
        uint8_t     nMaxLayerPerUeMuDl;
        uint8_t     nMaxUegPerCellDl;
        uint8_t     nMaxUegPerCellUl;   
        float       mcsSelSinrCapThr;
        float       muGrpSrsSnrMaxGap;
        float       muGrpSrsSnrSplitThr;
        uint8_t     bfPowAllocScheme;   
        uint8_t     muGrpUpdate;
        uint8_t     mcsSelLutType;
        uint8_t     semiStatFreqAlloc;
        uint8_t     harqEnabledInd;
        uint8_t     mcsSelCqi;
        uint8_t     dlSchInd;
    };
}
