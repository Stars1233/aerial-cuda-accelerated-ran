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

#ifndef _NV_CUMAC_MSG_
#define _NV_CUMAC_MSG_

#include <stdint.h>
#include <stddef.h>

#include "cumac_pfm_sort.h"

#if defined(__cplusplus)
extern "C" {
#endif

//! Invalid buffer offset marker (all bits set)
#define INVALID_CUMAC_BUF_OFFSET (0xFFFFFFFF)

/**
 * Get human-readable name for cuMAC message ID
 *
 * @param[in] msg_id Message identifier from cumac_msg_t enum
 *
 * @return Pointer to message name string, or "UNKNOWN_CUMAC_MSG" if invalid
 */
const char* get_cumac_msg_name(int msg_id);

/**
 * cuMAC message type identifiers
 *
 * Defines message types for cuMAC control plane protocol
 * including configuration, lifecycle, and TTI scheduling messages
 */
typedef enum
{
    CUMAC_PARAM_REQUEST = 0x00,    //!< PARAM.request - Query parameters
    CUMAC_PARAM_RESPONSE = 0x01,   //!< PARAM.response - Parameter response
    CUMAC_CONFIG_REQUEST = 0x02,   //!< CONFIG.request - Configuration request
    CUMAC_CONFIG_RESPONSE = 0x03,  //!< CONFIG.response - Configuration response
    CUMAC_START_REQUEST = 0x04,    //!< START.request - Start cuMAC processing
    CUMAC_STOP_REQUEST = 0x05,     //!< STOP.request - Stop cuMAC processing
    CUMAC_STOP_RESPONSE = 0x06,    //!< STOP.response - Stop response
    CUMAC_ERROR_INDICATION = 0x07, //!< ERROR.indication - Error notification
    CUMAC_START_RESPONSE = 0x08,   //!< START.response - Start response

    CUMAC_DL_TTI_REQUEST = 0x80,  //!< DL_TTI.request - Downlink TTI request (Not used currently)
    CUMAC_UL_TTI_REQUEST = 0x81,  //!< UL_TTI.request - Uplink TTI request (Not used currently)

    CUMAC_SCH_TTI_REQUEST = 0x82,  //!< SCH_TTI.request - Scheduling TTI request
    CUMAC_SCH_TTI_RESPONSE = 0x83, //!< SCH_TTI.response - Scheduling TTI response

    CUMAC_TTI_END = 0x8F,               //!< TTI_END - End of TTI indication
    CUMAC_TTI_ERROR_INDICATION = 0x90,  //!< TTI_ERROR.indication - TTI-specific error
} cumac_msg_t;

/**
 * cuMAC task type enumeration
 *
 * Identifies which cuMAC scheduling algorithm to execute
 */
typedef enum
{
    CUMAC_TASK_UE_SELECTION = 0,    //!< UE selection task (cumac::multiCellUeSelection)
    CUMAC_TASK_PRB_ALLOCATION = 1,  //!< PRB allocation task (cumac::multiCellScheduler)
    CUMAC_TASK_LAYER_SELECTION = 2, //!< Layer selection task (cumac::multiCellLayerSel)
    CUMAC_TASK_MCS_SELECTION = 3,   //!< MCS selection task (cumac::mcsSelectionLUT)
    CUMAC_TASK_PFM_SORT = 4,        //!< PFM sorting task
    CUMAC_TASK_TOTAL_NUM = 5        //!< Total number of task types
} cumac_task_type_t;

/**
 * cuMAC message header structure
 *
 * Based on SCF FAPI header format (see SCF222 document). All cuMAC messages begin with this header.
 */
typedef struct
{
    uint8_t message_count;  //!< Number of messages in this transmission
    uint8_t handle_id;      //!< handle_id is used as cell_id
    uint16_t type_id;       //!< Message type identifier (cumac_msg_t)
    uint32_t body_len;      //!< Length of message body in bytes
    uint8_t body[0];        //!< Variable-length message body (flexible array member)
} cumac_msg_header_t;

/**
 * Slot-specific message header
 *
 * Extended header for TTI-related messages that includes timing information
 */
typedef struct
{
    cumac_msg_header_t header; //!< Standard message header
    uint16_t sfn;              //!< Frame Number (0-1023)
    uint16_t slot;             //!< Slot number within frame
    uint8_t payload[0];        //!< Variable-length payload (flexible array member)
} cumac_slot_msg_header_t;

/**
 * CONFIG.request message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_CONFIG_REQUEST
    uint8_t body[];            //!< Configuration payload (cumac_config_req_payload_t)
} cumac_config_req_t;

/**
 * CONFIG.response message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_CONFIG_RESPONSE
    int error_code;            //!< Error code: 0 on success, other on failure
} cumac_config_resp_t;

/**
 * START.request message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_START_REQUEST
    int start_param;           //!< Start parameters (reserved for future use)
} cumac_start_req_t;

/**
 * START.response message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_START_RESPONSE
    int error_code;            //!< Error code: 0 on success, other on failure
} cumac_start_resp_t;

/**
 * STOP.request message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_STOP_REQUEST
    int stop_param;            //!< Stop parameters (reserved for future use)
} cumac_stop_req_t;

/**
 * STOP.response message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_STOP_RESPONSE
    int error_code;            //!< Error code: 0 on success, other on failure
} cumac_stop_resp_t;

/**
 * ERROR.indication message structure
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_ERROR_INDICATION
    int msg_id;                //!< Message ID that caused the error
    int error_code;            //!< Error code identifier
    int reason_code;           //!< Detailed reason code for the error
} cumac_err_ind_t;

/**
 * CONFIG.request payload structure
 *
 * Static scheduler configuration parameters passed from L2 stack to cuMAC-CP
 * during initialization phase (one-time configuration).
 */
typedef struct
{
    uint8_t harqEnabledInd;    //!< Indicator for whether HARQ is enabled
    uint8_t mcsSelCqi;         //!< Indicator for whether MCS selection is based on CQI or SINR
    uint8_t nMaxCell;          //!< A constant integer for the maximum number of cells in the cell group
    uint16_t nMaxActUePerCell; //!< A constant integer for the maximum number of active UEs per cell
    uint8_t nMaxSchUePerCell;  //!< A constant integer for the maximum number of UEs that can be scheduled per TTI per cell
    uint16_t nMaxPrg;          //!< A constant integer for the maximum number of PRGs for allocation in each cell
    uint16_t nPrbPerPrg;       //!< A constant integer for the number of PRBs per PRG (PRB group)
    uint8_t nMaxBsAnt;         //!< A constant integer for the maximum number of BS antenna ports. 
    uint8_t nMaxUeAnt;         //!< A constant integer for the maximum number of UE antenna ports. 
    uint32_t scSpacing;        //!< Subcarrier spacing of the carrier. Value: 15000, 30000, 60000, 120000 (Hz) 
    uint8_t allocType;         //!< Indicator for type-0 or type-1 PRG allocation
    uint8_t precoderType;      //!< Indicator for the precoder type
    uint8_t receiverType;      //!< Indicator for the receiver type
    uint8_t colMajChanAccess;  //!< Indicator for whether the estimated narrow-band SRS channel matrices are stored in column-major order or in row-major order
    float betaCoeff;           //!< Coefficient for adjusting the cell-edge UEs' performance in multi-cell scheduling
    float sinValThr;           //!< Singular value threshold for layer selection
    float corrThr;             //!< Channel vector correlation value threshold for layer selection
    float mcsSelSinrCapThr;    //!< SINR capping threshold for MCS selection
    uint8_t mcsSelLutType;     //!< MCS selection LUT type
    uint16_t prioWeightStep;   //!< Step size for UE priority weight increment per TTI if UE does not get scheduled. For priority-based UE selection
    float blerTarget;          //!< BLER target (same for all active UEs; expanded per-UE in cuMAC-CP)
} cumac_config_req_payload_t;

/**
 * SCH_TTI.request buffer offsets structure
 *
 * Specifies byte offsets for data buffers in SCH_TTI.request message.
 * All offsets are relative to the start of the data payload.
 */
typedef struct
{
    uint32_t CRNTI;                  //!< C-RNTIs of all active UEs in the cell
    uint32_t srsCRNTI;               //!< C-RNTIs of the UEs that have refreshed SRS channel estimates in the cell.
    uint32_t prgMsk;                 //!< Bit map for the availability of each PRG for allocation
    uint32_t postEqSinr;             //!< Array of the per-PRG per-layer post-equalizer SINRs of all active UEs in the cell
    uint32_t wbSinr;                 //!< Array of wideband per-layer post-equalizer SINRs of all active UEs in the cell
    uint32_t estH_fr;                //!< For FP32. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    uint32_t estH_fr_half;           //!< For FP16. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    uint32_t prdMat;                 //!< Array of the precoder/beamforming weights for all active UEs in the cell
    uint32_t detMat;                 //!< Array of the detector/beamforming weights for all active UEs in the cell
    uint32_t sinVal;                 //!< Array of the per-UE, per-PRG, per-layer singular values obtained from the SVD of the channel matrix
    uint32_t avgRatesActUe;          //!< Array of the long-term average data rates of all active UEs in the cell
    uint32_t prioWeightActUe;        //!< For priority-based UE selection. Priority weights of all active UEs in the cell
    uint32_t tbErrLastActUe;         //!< TB decoding error indicators of all active UEs in the cell
    uint32_t newDataActUe;           //!< Indicators of initial transmission/retransmission for all active UEs in the cell
    uint32_t allocSolLastTxActUe;    //!< The PRG allocation solution for the last transmissions of all active UEs in the cell
    uint32_t mcsSelSolLastTxActUe;   //!< MCS selection solution for the last transmissions of all active UEs in the cell
    uint32_t layerSelSolLastTxActUe; //!< Layer selection solution for the last transmissions of all active UEs in the cell
    uint32_t pfmCellInfo;            //!< PFM sorting input buffer
} cumac_tti_req_buf_offsets_t;

/**
 * SCH_TTI.request payload structure
 *
 * Contains per-TTI scheduling parameters and buffer offset information
 */
// Payload of SCH_TTI.req
typedef struct
{
    uint32_t taskBitMask; //!< Indicate which cuMAC tasks to be scheduled. Each bit represent 1 task type defined in cumac_task_type_t
    uint16_t cellID;      //!< cell ID
    uint8_t ULDLSch;      //!< Indication for UL/DL scheduling. Value - 0: UL scheduling, 1: DL scheduling
    uint16_t nActiveUe;   //!< total number of active UEs in the cell
    uint16_t nSrsUe;      //!< the number of UEs in the cell that have refreshed SRS channel estimates
    uint16_t nPrbGrp;     //!< the number of PRGs that can be allocated for the current TTI, excluding the PRGs that need to be reserved for HARQ re-tx's
    uint8_t nBsAnt;       //!< number of BS antenna ports
    uint8_t nUeAnt;       //!< number of UE antenna ports
    float sigmaSqrd;      //!< noise variance

    cumac_tti_req_buf_offsets_t offsets; //!< Data buffer offsets for input arrays

} cumac_tti_req_payload_t;

/**
 * SCH_TTI.request message structure
 *
 * Complete message including header, timing, and payload
 */
typedef struct {
    cumac_msg_header_t header;     //!< Message header with type CUMAC_SCH_TTI_REQUEST
    uint16_t sfn;                  //!< Frame Number
    uint16_t slot;                 //!< Slot number

    cumac_tti_req_payload_t payload; //!< Request payload with scheduling parameters
} cumac_sch_tti_req_t;

/**
 * SCH_TTI.response buffer offsets structure
 *
 * Specifies byte offsets for output buffers in SCH_TTI.response message
 */
typedef struct
{
    uint32_t setSchdUePerCellTTI; //!< Set of IDs of the selected UEs for the cell
    uint32_t allocSol;            //!< PRB group allocation solution for all active UEs in the cell
    uint32_t layerSelSol;         //!< Layer selection solution for all active UEs in the cell
    uint32_t mcsSelSol;           //!< MCS selection solution for all active UEs in the cell
    uint32_t pfmSortSol;          //!< PFM sorting output buffer
} cumac_tti_resp_buf_offsets_t;

/**
 * SCH_TTI.response message structure
 *
 * Returns scheduling solutions computed by cuMAC
 */
typedef struct {
    cumac_msg_header_t header;          //!< Message header with type CUMAC_SCH_TTI_RESPONSE
    uint16_t sfn;                       //!< Frame Number
    uint16_t slot;                      //!< Slot number
    uint16_t nUeSchd;                   //!< Number of UEs actually scheduled this TTI for this cell

    cumac_tti_resp_buf_offsets_t offsets; //!< Data buffer offsets for output arrays
} cumac_sch_tti_resp_t;

/**
 * TTI_END message structure
 *
 * Indicates completion of TTI processing
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_TTI_END
    uint16_t sfn;              //!< Frame Number
    uint16_t slot;             //!< Slot number
    int end_param;             //!< End parameters (reserved for future use)
} cumac_tti_end_t;

/**
 * TTI_ERROR.indication message structure
 *
 * Reports TTI-specific errors during processing
 */
typedef struct {
    cumac_msg_header_t header; //!< Message header with type CUMAC_TTI_ERROR_INDICATION
    uint16_t sfn;              //!< Frame Number
    uint16_t slot;             //!< Slot number
    int msg_id;                //!< Message ID that caused the error
    int error_code;            //!< Error code identifier
    int reason_code;           //!< Detailed reason code for the error
} cumac_tti_err_ind_t;

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_CUMAC_MSG_ */
