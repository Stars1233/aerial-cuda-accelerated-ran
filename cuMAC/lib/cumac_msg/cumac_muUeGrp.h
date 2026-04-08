/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 #ifndef _CUMAC_MU_UE_GRP_DATA_TYPE_
 #define _CUMAC_MU_UE_GRP_DATA_TYPE_
 
 #include <stdint.h>
 #include <stddef.h>
 
 #if defined(__cplusplus)
 extern "C" {
 #endif
 
// *********************************** MU-MIMO UE grouping constants *****************************************************
/// @brief maximum number of cells for joint scheduling. Default: 6
#define MAX_NUM_CELL (6U)

/// @brief maximum number of antenna ports per UE. Default: 4
#define MAX_NUM_UE_ANT_PORT (4U)

/// @brief number of antenna ports per RU. Default: 64
#define MAX_NUM_BS_ANT_PORT (64U)

/// @brief maximum number of PRGs per cell. Default:272 PRBs/2 prbPerPrg = 136 PRGs
#define MAX_NUM_PRG (136U)

/// @brief maximum number of subbands per UE considered for UE grouping. Default: 4
#define MAX_NUM_SUBBAND (4U)

/// @brief maximum number of per-PRG SRS channel estimate samples per subband. Default: 4
#define MAX_NUM_PRG_SAMP_PER_SUBBAND (4U)

/// @brief maximum number of SRS info per slot. Default: 48, assume SRS comb: 4, max CS: 12, max SRS port: 4, max SRS UE/symbol: 12, max SRS symbols/SLOT: 4(SS), 2(UL)
#define MAX_NUM_UE_SRS_INFO_PER_SLOT (48U)

/// @brief SRS UE capacity per cell. Default: 384, i.e., totally 384 UEs can be configured for SRS channel estimation in a cell
#define MAX_NUM_SRS_UE_PER_CELL (384U)

/// @brief maximum number of scheduled UEs per cell. Default: 16, assume max 16 UEs can be scheduled per cell
#define MAX_NUM_SCHD_UE_PER_CELL (16U)

/// @brief maximum number of UEs for grouping per cell. Default: 64
#define MAX_NUM_UE_FOR_GRP_PER_CELL (64U)

/// @brief value for unavailable channel orthogonality
#define UNAVAILABLE_CHAN_ORTH_VAL (FLT_MAX)

/// @brief maximum number of UEs per UEG. Default: 16
#define MAX_NUM_UE_PER_GRP (16U)

/// @brief maximum number of layers per UEG. Default: 16
#define MAX_NUM_LAYER_PER_GRP (16U)

/// @brief maximum number of UEGs scheduled per cell per TTI. Default: 4
#define MAX_NUM_UEG_PER_CELL (4U)


/// @brief inline function to convert a float to a 32-bit integer in bits format
static inline uint32_t cumac_f32_to_u32_bits(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

// ************************************ MU-MIMO UE grouping data structures **************************************
// input to cuMAC UE grouping on GPU
/// @brief per SRS-enabled UE information structure
typedef struct {
    uint32_t    avgRate; // average rate in DL, in bits/s
    uint32_t    currRate; // current instantaneous rate in DL, in bits/s
    uint32_t    bufferSize; // current buffer size in bits
    uint16_t    rnti; // C-RNTI ranging from 1 to 65535
    uint16_t    id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1
    uint16_t    numAllocPrgLastTx; // number of PRGs allocated to the UE in the last transmission
    uint16_t    srsInfoIdx; // index of the SRS info in the SRS info array
    uint8_t     layerSelLastTx; // number of layers selected for the UE in the last transmission
    uint8_t     nUeAnt; // number of SRS TX antenna ports. Value: 2, 4
    uint8_t     flags = 0x00; 
    // 1st bit (flags & 0x01) - is a valid UE info, 0: invalid, >0: valid
    // 2nd bit (flags & 0x02) - new TX indication, 0: re-TX, >0: new TX
    // 3rd bit (flags & 0x04) - SRS chanEst available, 0: no, >0: yes
    // 4th bit (flags & 0x08) - has updated SRS info in the current slot, 0: no, >0: yes
} cumac_muUeGrp_req_ue_info_t;

/// @brief per-UE SRS channel estimation information structure with cuBB/cuMAC SRS memory bank sharing
typedef struct {
    uint32_t    srsWbSnr; // wideband SNR in dB measured within configured SRS bandwidth for the UE, stored as a 32-bit integer. Need to be populated by L2 stack.
    uint32_t    realBuffIdx; // real buffer index of the SRS channel estimates in the SRS memory bank, also the corrresping GPU buffer index. Does NOT need to be populated by L2 stack.
    uint16_t    rnti; // C-RNTI ranging from 1 to 65535. Need to be populated by L2 stack.
    uint16_t    id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1. Need to be populated by L2 stack.
    uint16_t    srsStartPrg; // Starting PRB group index for SRS allocation. Does NOT need to be populated by L2 stack.
    uint16_t    srsStartValidPrg; // First valid PRB group index. Does NOT need to be populated by L2 stack.
    uint16_t    srsNValidPrg; // Number of valid PRB groups. Does NOT need to be populated by L2 stack.
    uint8_t     nUeAnt; // number of SRS TX antenna ports. Value: 2, 4. Need to be populated by L2 stack.
    uint8_t     flags = 0x00; // 1st bit (flags & 0x01) - is a valid SRS info, 0: invalid, 1: valid. Determined by the SRS channel estimate buffer state (INIT, REQUESTED, READY, NONE)
} cumac_muUeGrp_req_srs_info_msh_t;
 
/// @brief per-UE SRS channel estimation information structure (without cuBB/cuMAC SRS memory bank sharing)
typedef struct {
    uint32_t    srsWbSnr; // wideband SNR in dB measured within configured SRS bandwidth for the UE, stored as a 32-bit integer
    uint16_t    rnti; // C-RNTI ranging from 1 to 65535
    uint16_t    id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1
    uint8_t     nUeAnt; // number of SRS TX antenna ports. Value: 2, 4
    uint8_t     flags = 0x00; // 1st bit (flags & 0x01) - is a valid SRS info, 0: invalid, 1: valid
    __half2     srsChanEst[MAX_NUM_BS_ANT_PORT*MAX_NUM_UE_ANT_PORT*MAX_NUM_SUBBAND*MAX_NUM_PRG_SAMP_PER_SUBBAND];
    // for each subband, each PRG, each UE/RU antenna port, each RU antenna port
} cumac_muUeGrp_req_srs_info_t;

/// @brief per-cell UE grouping information structure
typedef struct {
    uint32_t    betaCoeff = cumac_f32_to_u32_bits(1.0f); // exponent applied to the instantaneous rate for proportional-fair scheduling. Default value is 1.0.
    uint32_t    muCoeff = cumac_f32_to_u32_bits(1.5f); // coefficient for prioritizing UEs feasible for MU-MIMO transmissions. Default value is 1.5.
    uint32_t    chanCorrThr = cumac_f32_to_u32_bits(0.7f); // threshold on the channel vector correlation value for UE grouping. Value: a real number between 0 and 1.0. Default: 0.7
    uint32_t    srsSnrThr = cumac_f32_to_u32_bits(-3.0f); // Threshold on measured SRS SNR in dB for determining the feasibility of MU-MIMO transmission. Default value is -3.0 (dB).
    uint32_t    muGrpSrsSnrMaxGap = cumac_f32_to_u32_bits(100.0f); // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG. Value: a real number greater than 0.0. Default: 100.0
    uint32_t    muGrpSrsSnrSplitThr = cumac_f32_to_u32_bits(-100.0f); // threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately. Value: a real number greater than 0.0. Default: -100.0
    uint16_t    numUeInfo; // number of effective ueInfo in the payload
    uint16_t    numSrsInfo; // number of effective srsInfo in the payload
    uint16_t    numSubband; // number of subbands considered for UE grouping.
    uint16_t    numPrgSampPerSubband; // number of per-PRG SRS channel estimate samples per subband.
    uint16_t    numUeForGrpPerCell = 64; // number of UEs considered for UE grouping per cell. Default: 64
    uint16_t    nPrbGrp; // number of PRGs that can be allocated.
    uint8_t     nBsAnt = 64; // Each RU’s number of TX & RX antenna ports. Default: 64
    uint8_t     nMaxUeSchdPerCellTTI = 16; // maximum number of UEs scheduled per cell per TTI. Default: 16
    uint8_t     nMaxUePerGrp = 16; // maximum number of UEs per UEG. Default: 16
    uint8_t     nMaxLayerPerGrp = 16; // maximium number of layers per UEG. Default: 16
    uint8_t     nMaxLayerPerUeSu = 4; // maximium number of layers per UE for SU-MIMO. Default: 4
    uint8_t     nMaxLayerPerUeMu = 4; // maximium number of layers per UE for MU-MIMO. Default: 4
    uint8_t     nMaxUegPerCell = 4; // maximum number of UEGs per cell. Default: 4
    uint8_t     allocType = 1; // PRB allocation type. Currently only support 1: consecutive type-1 allocation.  
    cumac_muUeGrp_req_srs_info_t* srsInfo;
    cumac_muUeGrp_req_srs_info_msh_t* srsInfoMsh;
    cumac_muUeGrp_req_ue_info_t* ueInfo; 
    uint8_t     payload[];
} cumac_muUeGrp_req_info_t;

/// @brief UE grouping request message structure
typedef struct {
    uint16_t    sfn;
    uint16_t    slot;
    uint32_t    offsetData; // Offset of data payload in the nvipc_buf->data_buf
    uint8_t     extraPayload[0];   // extra payload for future use
 } cumac_muUeGrp_req_msg_t;

/*****************************************************/
// output from cuMAC UE grouping on GPU
/// @brief per-UE grouping information structure
typedef struct {
    uint16_t    rnti; // C-RNTI ranging from 1 to 65535
    uint16_t    id; // 0-based UE ID used for cuMAC scheduling, randing from 0 to MAX_NUM_SRS_UE_PER_CELL-1
    uint8_t     layerSel; 
    // bit map of layer selection for the current transmission
    // bit 0: layer corresponding to antenna port 0
    // bit 1: layer corresponding to antenna port 1
    // bit 2: layer corresponding to antenna port 2
    // bit 3: layer corresponding to antenna port 3
    uint8_t     ueOrderInGrp; // UE order in the UEG for the current transmission (to assist beamforming)
    uint8_t     nSCID; //! nSCID allocation is currently not supported. Always set to 0xFF. 
    uint8_t     flags; // flags & 0x01 - is_valid, flags & 0x02 - MU-MIMO indication (0: SU-MIMO, 1: MU-MIMO)
} cumac_muUeGrp_resp_ue_info_t;

/// @brief per-UEG grouping information structure
typedef struct {
    int16_t     allocPrgStart; // PRG index of the first PRG allocated to the UEG in the current transmission
    int16_t     allocPrgEnd; // one plus the PRG index of the last PRG allocated to the UEG in the current transmission
    uint8_t     numUeInGrp; // number of UEs in the UEG
    uint8_t     flags; // flags & 0x01 - is_valid
    cumac_muUeGrp_resp_ue_info_t ueInfo[MAX_NUM_UE_PER_GRP];
} cumac_muUeGrp_resp_ueg_info_t;

/// @brief UE grouping information structure
typedef struct {
    uint32_t    numSchdUeg; // total number of scheduled UEGs
    cumac_muUeGrp_resp_ueg_info_t schdUegInfo[MAX_NUM_UEG_PER_CELL];
} cumac_muUeGrp_resp_info_t;

/// @brief UE grouping response message structure
typedef struct {
    uint16_t    sfn; // subframe index in radio frame
    uint16_t    slot; // time slot index in radio frame
    uint32_t    offsetData; // Offset of data payload in the nvipc_buf->data_buf
    uint8_t     extraPayload[0];   // extra payload for future use
} cumac_muUeGrp_resp_msg_t;

 #if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _CUMAC_MU_UE_GRP_DATA_TYPE_ */