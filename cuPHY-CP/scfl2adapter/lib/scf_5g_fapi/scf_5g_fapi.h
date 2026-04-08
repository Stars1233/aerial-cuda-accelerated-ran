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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef enum
{
    SCF_ERROR_CODE_MSG_OK                    = 0x0,
    SCF_ERROR_CODE_MSG_INVALID_STATE         = 0x1,
    SCF_ERROR_CODE_MSG_INVALID_CONFIG        = 0x2,
    SCF_ERROR_CODE_SFN_OUT_OF_SYNC           = 0x3,
    SCF_ERROR_CODE_MSG_SLOT_ERR              = 0x4,
    SCF_ERROR_CODE_MSG_BCH_MISSING           = 0x5,
    SCF_ERROR_CODE_MSG_INVALID_SFN           = 0x6,
    SCF_ERROR_CODE_MSG_UL_DCI_ERR            = 0x7,
    SCF_ERROR_CODE_MSG_TX_ERR                = 0x8,
//SCF FAPI v10.04 error codes ---- begin
    SCF_ERROR_CODE_MSG_INVALID_PHY_ID        = 0x9,
    SCF_ERROR_CODE_MSG_UNINSTANTIATED_PHY    = 0xA,
    SCF_ERROR_CODE_MSG_INVALID_DFE_Profile   = 0xB,
    SCF_ERROR_CODE_MSG_PHY_PROFILE_SELECTION = 0xC,
//SCF FAPI v10.04 error codes ---- end
    SCF_ERROR_CODE_FAPI_END                  = 0x32,
//Vendor specific error codes ---- begin     
    SCF_ERROR_CODE_L1_PROC_OBJ_UNAVAILABLE_ERR = 0x33,
    SCF_ERROR_CODE_MSG_LATE_SLOT_ERR         = 0x34,
    SCF_ERROR_CODE_PARTIAL_SRS_IND_ERR       = 0x35,
    SCF_ERROR_CODE_L1_DL_CPLANE_TX_ERROR = 0x36, //Indicates a DL C-plane trasmission error (Timing/Functional)
    SCF_ERROR_CODE_L1_UL_CPLANE_TX_ERROR = 0x37, //Indicates a UL C-plane trasmission error (Timing/Functional)
    SCF_ERROR_CODE_L1_DL_GPU_ERROR = 0x38,       //Indicates a DL GPU pipeline processing error
    SCF_ERROR_CODE_L1_DL_CPU_TASK_ERROR = 0x39,  //Indicates a DL CPU Task incompletion error
    SCF_ERROR_CODE_L1_UL_CPU_TASK_ERROR = 0x3A,  //Indicates a UL CPU Task incompletion error   
    SCF_ERROR_CODE_L1_P1_EXIT_ERROR = 0x3B,      //Indicates Part 1 of the error indication during L1 app exit process 
    SCF_ERROR_CODE_L1_P2_EXIT_ERROR = 0x3C,      //Indicates Part 2 of the error indication during L1 app exit process post cudaDeviceSynchronize if CUDA coredump env variables are set
    SCF_ERROR_CODE_L1_DL_CH_ERROR = 0x3D,        //Indicates DL channel run (CPU/GPU) error       
    SCF_ERROR_CODE_L1_UL_CH_ERROR = 0x3E,        //Indicates UL channel run (CPU/GPU) error
    SCF_ERROR_CODE_EARLY_HARQ_TIMING_ERROR = 0x3F, //Indicates EARLY HARQ processing failed to meet channel processing segment TLV timelines
    SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE = 0x40, //Indicates SRS Chest Buffer is in a bad state
    SCF_ERROR_CODE_BEAM_ID_OUT_OF_RANGE = 0x41, //Indicates Beam ID is out of range
    SCF_ERROR_CODE_PTP_SVC_ERROR = 0x42, //Indicates PTP service error
    SCF_ERROR_CODE_PTP_SYNCED = 0x43,   //Indicates PTP in sync
    SCF_ERROR_CODE_L1_MISSING_UL_IQ = 0x44, //Indicates OK timeout error . For L2 it means missing UL IQ data
    SCF_ERROR_CODE_MSG_CAPACITY_EXCEEDED = 0x45, //Indicates capacity exceeded error
    SCF_ERROR_CODE_RHOCP_PTP_EVENTS_ERROR = 0x46, //Indicates RHOCP PTP Events not returned as sync
    SCF_ERROR_CODE_RHOCP_PTP_EVENTS_SYNCED = 0x47,   //Indicates RHOCP PTP Events back to sync again after unsynced
    SCF_ERROR_CODE_SRS_WITHOUT_PUSCH_UNSUPPORTED = 0x48,  // SINGLE_SECT_MODE: unsupported UL channel combination (e.g. SRS without PUSCH)
// L1 limit exceeded error codes ---- begin
    SCF_FAPI_SSB_PBCH_L1_LIMIT_EXCEEDED =   0x81, // SSB/PBCH L1 limit exceeded
    SCF_FAPI_PDCCH_L1_LIMIT_EXCEEDED = 0x82, // PDCCH L1 limit exceeded
    SCF_FAPI_PDSCH_L1_LIMIT_EXCEEDED = 0x84, // PDSCH L1 limit exceeded
    SCF_FAPI_CSIRS_L1_LIMIT_EXCEEDED = 0x88, // CSIRS L1 limit exceeded

    //FH port link status error codes ---- begin
    SCF_ERROR_CODE_FH_PORT_DOWN = 0x99,  // Indicates FH port link down
    SCF_ERROR_CODE_FH_PORT_UP   = 0x9A,  // Indicates FH port link up
    //FH port link status error codes ---- end

    SCF_FAPI_PUSCH_L1_LIMIT_EXCEEDED = 0xC1, // PUSCH L1 limit exceeded
    SCF_FAPI_PUCCH_L1_LIMIT_EXCEEDED = 0xC2, // PUCCH L1 limit exceeded
    SCF_FAPI_SRS_L1_LIMIT_EXCEEDED = 0xC4, // SRS L1 limit exceeded
    SCF_FAPI_PRACH_L1_LIMIT_EXCEEDED = 0xC8, // PRACH L1 limit exceeded
// L1 limit exceeded error codes ---- end
    SCF_ERROR_CODE_RELEASED_HARQ_BUFFER_INFO = 0xD0, // HARQ buffer released
} scf_fapi_error_codes_t;

/* Table 3-4: PHY API Message Types */
typedef enum
{
    SCF_FAPI_PARAM_REQUEST      = 0x00,
    SCF_FAPI_PARAM_RESPONSE     = 0x01,
    SCF_FAPI_CONFIG_REQUEST     = 0x02,
    SCF_FAPI_CONFIG_RESPONSE    = 0x03,
    SCF_FAPI_START_REQUEST      = 0x04,
    SCF_FAPI_STOP_REQUEST       = 0x05,
    SCF_FAPI_STOP_INDICATION    = 0x06,
    SCF_FAPI_ERROR_INDICATION   = 0x07,
    /* RESERVED: 0x08-0x7F */
    SCF_FAPI_RESV_1_START       = 0x08,
    SCF_FAPI_RESV_1_END         = 0x7F,
    /* SLOT messages */
    SCF_FAPI_DL_TTI_REQUEST     = 0x80,
    SCF_FAPI_UL_TTI_REQUEST     = 0x81,
    SCF_FAPI_SLOT_INDICATION    = 0x82,
    SCF_FAPI_UL_DCI_REQUEST     = 0x83,
    SCF_FAPI_TX_DATA_REQUEST    = 0x84,
    SCF_FAPI_RX_DATA_INDICATION = 0x85,
    SCF_FAPI_CRC_INDICATION     = 0x86,
    SCF_FAPI_UCI_INDICATION     = 0x87,
    SCF_FAPI_SRS_INDICATION     = 0x88,
    SCF_FAPI_RACH_INDICATION    = 0x89,
    /* RESERVED: 0x8A-0xFF */
    SCF_FAPI_RESV_2_START       = 0x8A,
    SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION = 0x8B,
    //SCF_FAPI_RX_PF_01_INTEFERNCE_INDICATION = 0x8C,
    SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION = 0x8D,
    SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION = 0x8E,
    SCF_FAPI_SLOT_RESPONSE      = 0x8F,
    SCF_FAPI_DL_BFW_CVI_REQUEST = 0x90,
    SCF_FAPI_UL_BFW_CVI_REQUEST = 0x91,
    CV_MEM_BANK_CONFIG_REQUEST  = 0x92,
    CV_MEM_BANK_CONFIG_RESPONSE = 0x93,
    SCF_FAPI_RESV_2_END         = 0xFF,
} scf_fapi_message_id_e;

/* 3.4.2 DL_TTI PDU Type */
typedef enum
{
    DL_TTI_PDU_TYPE_PDCCH = 0,
    DL_TTI_PDU_TYPE_PDSCH = 1,
    DL_TTI_PDU_TYPE_CSI_RS = 2,
    DL_TTI_PDU_TYPE_SSB = 3,
} scf_fapi_dl_tti_pdu_type_t;

/* 3.4.2 DL_TTI PDU Type */
typedef enum
{
    UL_TTI_PDU_TYPE_PRACH = 0,
    UL_TTI_PDU_TYPE_PUSCH = 1,
    UL_TTI_PDU_TYPE_PUCCH = 2,
    UL_TTI_PDU_TYPE_SRS = 3,
} scf_fapi_ul_tti_pdu_type_t;

static constexpr int VALID_FAPI_PDU   = 0;
static constexpr int INVALID_FAPI_PDU = 1;

typedef enum
{
    UCI_IND_PDU_TYPE_PUSCH = 0,
    UCI_IND_PDU_TYPE_PUCCH_FORMAT_0_1 = 1,
    UCI_IND_PDU_TYPE_PUCCH_FORMAT_2_3_4 = 2,
} scf_fapi_uci_ind_pdu_type_t;


// [38.211, sec 6.3.2.1]
typedef enum
{
    UL_TTI_PUCCH_FORMAT_0,
    UL_TTI_PUCCH_FORMAT_1,
    UL_TTI_PUCCH_FORMAT_2,
    UL_TTI_PUCCH_FORMAT_3,
    UL_TTI_PUCCH_FORMAT_4
} scf_fapi_ul_tti_pucch_format_type_t;

typedef enum
{
    SCF_TX_DATA_INLINE_PAYLOAD = 0,
    SCF_TX_DATA_POINTER_PAYLOAD = 1,
    SCF_TX_DATA_OFFSET = 2
} scf_fapi_tx_req_payload_type_t;

typedef enum 
{
    // CELL PARAMS
    PARAM_TLVS_RELEASE_CAP = 0x0001,
    PARAM_TLVS_PHY_STATE = 0x0002,
    PARAM_TLVS_SKIP_BLANK_DL_CFG = 0x0003,
    PARAM_TLVS_SKIP_BLANK_UL_CFG = 0x0004,
    PARAM_TLVS_NUM_CONFIG_TLVS_TO_REPORT = 0x0005,

    // CARRIER PARAMS
    PARAM_TLVS_CYCLIC_PREFIX = 0x0006,
    PARAM_TLVS_SUPPORTED_SUBCARRIER_SPACING_DL = 0x0007,
    PARAM_TLVS_SUPPORTED_BANDWIDTH_DL = 0x0008,
    PARAM_TLVS_SUPPORTED_SUBCARRIER_SPACING_UL = 0x0009,
    PARAM_TLVS_SUPPORTED_BANDWIDTH_UL = 0x000A,

    // PDCCH PARAMS
    PARAM_TLVS_CCE_MAPPING_TYPE = 0x000B,
    PARAM_TLVS_CORESET_OUTSIDE_FIRST_4_OFDM_SYMS_OF_SLOT = 0x000C,
    PARAM_TLVS_PRECODER_GRANULARITY_CORESET = 0x000D,
    PARAM_TLVS_PDCCH_MU_MIMO = 0x000E,
    PARAM_TLVS_PDCCH_PRECORDER_CYCLING = 0x000F,
    PARAM_TLVS_MAX_PDCCHS_PER_SLOT = 0x0010,

    // PUCCH PARAMS
    PARAM_TLVS_PUCCH_FORMATS = 0x0011,
    PARAM_TLVS_MAX_PUCCH_PER_SLOT = 0x0012,

    // PDSCH PARAMS
    PARAM_TLVS_PDSCH_MAPPING_TYPE = 0x0013,
    PARAM_TLVS_PDSCH_ALLOC_TYPES = 0x0014,
    PARAM_TLVS_PDSCH_VRB_TO_PRB_MAPPING = 0x0015,
    PARAM_TLVS_PDSCH_CBG = 0x0016,
    PARAM_TLVS_PDSCH_DMRS_CONFIG_TYPES = 0x0017,
    PARAM_TLVS_PDSCH_DMRS_MAX_LEN = 0x0018,
    PARAM_TLVS_PDSCH_DMRS_ADDITIONAL_POS = 0x0019,
    PARAM_TLVS_MAX_PDSCHS_TBS_PER_SLOT = 0x001A,
    PARAM_TLVS_MAX_NUM_MIMO_LAYERS_PDSCH = 0x001B,
    PARAM_TLVS_SUPPORTED_MAX_MODULATION_ORDER_DL = 0x001C,
    PARAM_TLVS_MAX_MU_MIMO_USERS_DL = 0x001D,
    PARAM_TLVS_PDSCH_DATA_IN_DMRS_SYMBOLS = 0x001E,
    PARAM_TLVS_PREEMPTION_SUPPORT = 0x001F,
    PARAM_TLVS_PDSCH_NON_SLOT_SUPPORT = 0x0020,

    // PRACH PARAMS
    PARAM_TLVS_UCI_MUX_ULSCH_IN_PUSCH = 0x0021,
    PARAM_TLVS_UCI_ONLY_PUSCH = 0x0022,
    PARAM_TLVS_PUSCH_FREQ_HOPPING = 0x0023,
    PARAM_TLVS_PUSCH_DMRS_CONFIG_TYPES = 0x0024,
    PARAM_TLVS_PUSCH_DMRS_MAX_LEN = 0x0025,
    PARAM_TLVS_PUSCH_DMRS_ADDITIONAL_POS = 0x0026,
    PARAM_TLVS_PUSCH_CBG = 0x0027,
    PARAM_TLVS_PUSCH_MAPPING_TYPE = 0x0028,
    PARAM_TLVS_PUSCH_ALLOCATION_TYPES = 0x0029,
    PARAM_TLVS_PUSCH_VRB_TO_PRB_MAPPING = 0x002A,
    PARAM_TLVS_PUSCH_MAX_PTRS_PORTS = 0x002B,
    PARAM_TLVS_MAX_PDUSCHS_TBS_PER_SLOT = 0x002C,
    PARAM_TLVS_MAX_NUMBER_MIMO_LAYERS_NON_CB_PUSCH = 0x002D,
    PARAM_TLVS_SUPPORTED_MODULATION_ORDER_UL = 0x002E,
    PARAM_TLVS_MAX_MU_MIMO_USERS_UL = 0x002F,
    PARAM_TLVS_DFTS_OFDM_SUPPORT = 0x0030,
    PARAM_TLVS_PUSCH_AGGREGATION_FACTOR = 0x0031,
    PARAM_TLVS_PRACH_LONG_FORMATS = 0x0032,
    PARAM_TLVS_PRACH_SHORT_FORMATS = 0x0033,
    PARAM_TLVS_PRACH_RESTRICTED_SETS = 0x0034,
    PARAM_TLVS_MAX_PRACH_FD_OCCASIONS_IN_A_SLOT = 0x0035,
    PARAM_TLVS_RSSI_MEASUREMENT_SUPPORT = 0x0036
} scf_fapi_param_msg_id_e;

typedef enum
{
    // CARRIER CONFIG
    CONFIG_TLV_DL_BANDWIDTH = 0x1001,
    CONFIG_TLV_DL_FREQ = 0x1002,
    CONFIG_TLV_DLK0 = 0x1003,
    CONFIG_TLV_DL_GRID_SIZE = 0x1004,
    CONFIG_TLV_NUM_TX_ANT = 0x1005,
    CONFIG_TLV_UL_BANDWIDTH = 0x1006,
    CONFIG_TLV_UL_FREQ = 0x1007,
    CONFIG_TLV_ULK0 = 0x1008,
    CONFIG_TLV_UL_GRID_SIZE = 0x1009,
    CONFIG_TLV_NUM_RX_ANT = 0x100A,
    CONFIG_TLV_FREQ_SHIFT_7P_5KHZ = 0x100B,

    // CELL CONFIG
    CONFIG_TLV_PHY_CELL_ID = 0x100C,
    CONFIG_TLV_FRAME_DUPLEX_TYPE = 0x100D,

    // SSB CONFIG
    CONFIG_TLV_SSB_PBCH_POWER = 0x100E,
    CONFIG_TLV_BCH_PAYLOAD = 0x100F,
    CONFIG_TLV_SCS_COMMON = 0x1010,

    // PRACH CONFIG
    CONFIG_TLV_PRACH_SEQ_LEN = 0x1011,
    CONFIG_TLV_PRACH_SUBC_SPACING = 0x1012,
    CONFIG_TLV_RESTRICTED_SET_CONFIG = 0x1013,
    CONFIG_TLV_NUM_PRACH_FD_OCCASIONS = 0x1014,
    CONFIG_TLV_PRACH_ROOT_SEQ_INDEX = 0x1015,
    CONFIG_TLV_NUM_ROOT_SEQ = 0x1016,
    CONFIG_TLV_K1 = 0x1017,
    CONFIG_TLV_PRACH_ZERO_CORR_CONF = 0x1018,
    CONFIG_TLV_NUM_UNUSED_ROOT_SEQ = 0x1019,
    CONFIG_TLV_UNUSED_ROOT_SEQ = 0x101A,
    CONFIG_TLV_SSB_PER_RACH = 0x101B,
    CONFIG_TLV_PRACH_MULT_CARRIERS_IN_BAND = 0x101C,
    CONFIG_TLV_PRACH_CONFIG_INDEX = 0x1029,

    // SSB TABLE
    CONFIG_TLV_SSB_OFFSET_POINT_A = 0x101D,
    CONFIG_TLV_BETA_PSS = 0x101E,
    CONFIG_TLV_SSB_PERIOD = 0x101F,
    CONFIG_TLV_SSB_SUBCARRIER_OFFSET = 0x1020,
    CONFIG_TLV_MIB = 0x1021,
    CONFIG_TLV_SSB_MASK = 0x1022,
    CONFIG_TLV_BEAM_ID = 0x1023,
    CONFIG_TLV_SSB_PBCH_MULT_CARRIERS_IN_BAND = 0x1024,
    CONFIG_TLV_MULTIPLE_CELLS_SS_PBCH_IN_CARRIER = 0x1025,
    
    // TDD TABLE
    CONFIG_TLV_TDD_PERIOD = 0x1026,
    CONFIG_TLV_SLOT_CONFIG = 0x1027,

    // MEASUREMENT CONFIG
    CONFIG_TLV_RSSI_MEAS = 0x1028,
#ifdef SCF_FAPI_10_04
    CONFIG_TLV_RSRP_MEAS = 0x1040,
    CONFIG_TLV_INDICATION_INSTANCES_PER_SLOT = 0x102B,
    CONFIG_TLV_UCI_CONFIG = 0x1036,
#endif
    // The range 0xA000 – 0xAFFF are reserved for vendor-specific TLVs.
    CONFIG_TLV_VENDOR_DIGITAL_BEAM_TABLE_PDU = 0xA010,
    CONFIG_TLV_VENDOR_PRECODING_MATRIX = 0xA011,
    CONFIG_TLV_VENDOR_NOISE_VAR_MEAS = 0xA012,
    //CONFIG_TLV_VENDOR_PF_01_INTERFERENCE_MEAS = 0xA013,
    CONFIG_TLV_VENDOR_PF_234_INTERFERENCE_MEAS = 0xA014,
    CONFIG_TLV_VENDOR_PRACH_INTERFERENCE_MEAS = 0xA015,
    CONFIG_TLV_VENDOR_NUM_TX_PORT = 0xA016,
    CONFIG_TLV_VENDOR_NUM_RX_PORT = 0xA017,
    CONFIG_TLV_VENDOR_CHAN_SEGMENT = 0xA018,
#ifdef SCF_FAPI_10_04
    CONFIG_TLV_VENDOR_NUM_SRS_CHEST_BUFFERS=0xA019,
#endif
    CONFIG_TLV_VENDOR_PUSCH_AGGR_FACTOR=0xA01A
} scf_fapi_config_message_id_e;


typedef enum
{
    SCF_FAPI_MSG_OK            = 0,
    SCF_FAPI_MSG_INVALID_STATE = 1
} scf_fapi_error_code_e;

#define MAX_SRS_SYMBOLS 4
#define MAX_SRS_REPORT_TYPE 4

typedef enum
{
    SRS_REPORT_FOR_BEAM_MANAGEMENT = 0x1,
    SRS_REPORT_FOR_CODEBOOK = 0x2,
    SRS_REPORT_FOR_NON_CODEBOOK = 0x4,
    SRS_REPORT_FOR_ANTENNA_SWITCHING = 0x8,
} scf_fapi_report_type_t;

typedef enum
{
    SRS_USAGE_FOR_BEAM_MANAGEMENT = 0,
    SRS_USAGE_FOR_CODEBOOK = 1,
    SRS_USAGE_FOR_NON_CODEBOOK = 2,
    SRS_USAGE_FOR_ANTENNA_SWITCHING = 3,
} scf_fapi_srs_ind_usage_type_t;

typedef enum
{
    INDEX_IQ_REPR_16BIT_NORMALIZED_ = 0,
    INDEX_IQ_REPR_32BIT_NORMALIZED_ = 1,
    INDEX_IQ_REPR_FP32_COMPLEX = 2,
} scf_fapi_srs_iq_repr_t;

typedef enum
{
    IQ_REPR_16BIT_NORMALIZED_IQ_SIZE_2 = 2,
    IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4 = 4,
    IQ_REPR_FP32_COMPLEX_IQ_SIZE_8 = 8,
} scf_fapi_srs_iq_repr_size_t;

/* Table 3-2: PHY API Message Header */
typedef struct
{
    uint8_t message_count;
    uint8_t handle_id;
    uint8_t payload[0];
} __attribute__ ((__packed__)) scf_fapi_header_t;

/* Table 3-3: General PHY API Message Structure */
typedef struct
{
    uint16_t type_id;
    uint32_t length;
    uint8_t  data[0];
} __attribute__ ((__packed__)) scf_fapi_body_header_t;

typedef struct
{
    uint8_t error_code; /* error code                         */
    uint8_t tlv_count;  /* number of TLVs in the message body */
} __attribute__ ((__packed__)) scf_fapi_param_response_header_t;

/* Tag-Length header for "Tag-Length-Value" (TLV) triplets */
typedef struct scf_fapi_tl
{
    uint16_t tag;
#ifdef SCF_FAPI_10_04
    uint32_t length;
#else
    uint16_t length;
#endif
    uint8_t val[0];

    template <typename T>
    T AsValue() { return *reinterpret_cast<T*>(&val[0]); }

    template <typename T>
    T As() { return reinterpret_cast<T>(&val[0]); }

    auto Print()
    {
        switch (length)
        {
            case 1:
            {
                printf("tag=0x%04x val=%u", tag, *reinterpret_cast<uint8_t*>(val));
                break;
            }
            case 2:
            {
                printf("tag=0x%04x val=%u", tag, *reinterpret_cast<uint16_t*>(val));
                break;
            }
            case 4:
            {
                printf("tag=0x%04x val=%u", tag, *reinterpret_cast<uint32_t*>(val));
                break;
            }     
            default :
            {
                printf("Cannot print size tag=0x%04x length=%u\n", tag, length);
                break;
            }                   
        }

        return &val[0] + length;
    }

    template <typename T>
    auto *Set(uint16_t _tag, const uint32_t _val)
    {
        tag = _tag;
        length = sizeof(T);
        memcpy(&val[0], &_val, 4);

        // Advance to next TLV as a helper function. Align to 4 bytes, refer to Section 3.3.1.4 in SCF 222
        return &val[0] + ((length + 3) / 4) * 4;
    }

    template <typename T>
    auto *Set(uint16_t _tag, const uint8_t _val[], uint16_t len)
    {
        tag = _tag;
        length = len;
        memcpy(&val[0], _val, len);

        // Advance to next TLV as a helper function. Align to 4 bytes, refer to Section 3.3.1.4 in SCF 222
        return &val[0] + ((length + 3) / 4) * 4;
    }
} __attribute__ ((__packed__)) scf_fapi_tl_t;


typedef struct
{
    uint8_t num_tlvs;
    uint8_t tlvs[0];
} __attribute__ ((__packed__)) scf_fapi_config_request_body_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    scf_fapi_config_request_body_t msg_body;
} __attribute__ ((__packed__)) scf_fapi_config_request_msg_t;

typedef struct
{
    uint8_t error_code;
    uint8_t num_invalid_tlvs;
    uint8_t num_idle_only_tlvs;
    uint8_t num_running_only_tlvs;
    uint8_t num_missing_tlvs;
    uint8_t tlvs[0];
} __attribute__ ((__packed__)) scf_fapi_config_response_body_t;

typedef struct
{
    scf_fapi_body_header_t          msg_hdr;
    scf_fapi_config_response_body_t msg_body;
} __attribute__ ((__packed__)) scf_fapi_config_response_msg_t;

typedef struct
{
    uint8_t error_code;
} scf_fapi_generic_response_body_t;

typedef struct
{
    scf_fapi_header_t hdr;
    scf_fapi_body_header_t body_hdr;
} __attribute__ ((__packed__)) scf_fapi_generic_header_body_t;

typedef struct
{
    scf_fapi_body_header_t           msg_hdr;
    scf_fapi_generic_response_body_t msg_body;
} __attribute__ ((__packed__)) scf_fapi_generic_response_msg_t;

// Table 3-53 Rx Beamforming PDU
typedef struct
{
#ifdef SCF_FAPI_10_04_SRS
    uint8_t trp_scheme; /* "This field shall be set to 0, to identify that this table is used."
                        This lets us distinguish between a beamforming transmission and reception points (TRP) and others. */
#endif
    uint16_t num_prgs;
    uint16_t prg_size;
    uint8_t  dig_bf_interfaces;
    uint16_t beam_idx[0];
} __attribute__ ((__packed__)) scf_fapi_rx_beamforming_t;

typedef struct
{
    // BWP [TS38.213 sec 12]
    uint16_t bwp_size;
    uint16_t bwp_start;
    uint8_t  scs;
    uint8_t  cyclic_prefix;
}  __attribute__ ((__packed__)) scf_fapi_bwp_ts38_213_sec_12_t;


typedef struct
{
    uint8_t  beta_pdcch_1_0;
#ifdef SCF_FAPI_10_04
    int8_t power_control_offset_ss_profile_nr;
#else
    uint8_t  power_control_offset_ss;
#endif
} __attribute__ ((__packed__)) scf_fapi_pdcch_tx_power_info_t; // part of larger pdcch PDU message

typedef struct
{
    uint16_t payload_size_bits;
    uint8_t  payload[0];
} __attribute__ ((__packed__)) scf_fapi_pdcch_dci_payload_t; // part of larger pdcch PDU message

// Table 3-37 DL_DCI PDU
typedef struct
{
    uint16_t rnti;
    uint16_t scrambling_id;
    uint16_t scrambling_rnti;
    uint8_t  cce_index;
    uint8_t  aggregation_level;
    uint8_t  payload[0]; // precoding/beamforming, followed by tx power info
} __attribute__ ((__packed__)) scf_fapi_dl_dci_t;

typedef struct
{
    uint16_t target_code_rate;
    uint8_t  qam_mod_order;
    uint8_t  mcs_index; // valid in FAPI; only used for optional TB size check in cuPHY
    uint8_t  mcs_table; // valid in FAPI; only used for optional TB size check in cuPHY
    uint8_t  rv_index;
    uint32_t tb_size;
} __attribute__ ((__packed__)) scf_fapi_pdsch_codeword_t;


typedef struct
{
    uint16_t pdu_type;
    uint16_t pdu_size;
    uint8_t  pdu_config[0];
} __attribute__ ((__packed__)) scf_fapi_generic_pdu_info_t;

#ifdef ENABLE_L2_SLT_RSP
typedef struct
{
    uint16_t  pdu_type;
    uint16_t  rnti;
    uint8_t   pdu_index;
    uint8_t   discard_factor_pdu;
} __attribute__ ((__packed__)) scf_fapi_error_pdu_info_t;

typedef struct
{
    uint16_t  npdus_all;
    uint8_t   discard_factor_all;
    scf_fapi_error_pdu_info_t pdu_info[0];
} __attribute__ ((__packed__)) scf_fapi_error_extension_t;
#endif
/************************************************
 *  3.3.6.1 ERROR.indication
 ***********************************************/
// Table 3-30
typedef struct
{
    scf_fapi_body_header_t msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint8_t  msg_id;
    uint8_t  err_code;
#ifdef ENABLE_L2_SLT_RSP
    scf_fapi_error_extension_t extension[0];
#endif
} __attribute__ ((__packed__)) scf_fapi_error_ind_t;

typedef struct
{
    uint16_t rnti;
    uint8_t harq_pid;
    uint16_t sfn;//SFN in the last UL_TTI.request which scheduled released HARQ resource.
    uint16_t slot;//Slot in the last UL_TTI.request which scheduled released HARQ resource.
} __attribute__ ((__packed__)) scf_fapi_released_harq_buffer_info_t;
typedef struct
{
    scf_fapi_body_header_t msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint8_t  msg_id;
    uint8_t  err_code;
    uint16_t num_released_rscs;//Number of released HARQ resources.
    scf_fapi_released_harq_buffer_info_t  released_harq_buffers[0];
} __attribute__ ((__packed__)) scf_fapi_error_ind_with_released_harq_buffer_ext_t;


/************************************************
 *  3.4.1 Slot.indication
 ***********************************************/

// Table 3-34 Slot indication message body
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
} __attribute__ ((__packed__)) scf_fapi_slot_ind_t;

/************************************************
 *   Slot.response
 ***********************************************/
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
} __attribute__ ((__packed__)) scf_fapi_slot_rsp_t;

/************************************************
 *  3.4.2 DL_TTI.request
 ***********************************************/

// Table 3-35 DL_TTI.request message body
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
    uint8_t testMode; // value 0 - no test. 1 - TM1.1
#endif
    uint8_t  num_pdus;
    uint8_t  ngroup;
    uint8_t  payload[0];
} __attribute__ ((__packed__)) scf_fapi_dl_tti_req_t;

// Table 3-36 PDCCH PDU
typedef struct
{
    scf_fapi_bwp_ts38_213_sec_12_t bwp;
    uint8_t  start_sym_index;
    uint8_t  duration_sym;
    uint8_t  freq_domain_resource[6];
    uint8_t  cce_reg_mapping_type;
    uint8_t  reg_bundle_size;
    uint8_t  interleaver_size;
    uint8_t  coreset_type;
    uint16_t shift_index;
    uint8_t  precoder_granularity;
    uint16_t num_dl_dci;
    scf_fapi_dl_dci_t dl_dci[0];
} __attribute__ ((__packed__)) scf_fapi_pdcch_pdu_t;

// Table 3-38 DLSCH PDU
typedef struct
{
    uint16_t pdu_bitmap;
    uint16_t rnti;
    uint16_t pdu_index;

    scf_fapi_bwp_ts38_213_sec_12_t bwp;

    // Codeword information
    uint8_t num_codewords;
    scf_fapi_pdsch_codeword_t codewords[0];

    // Data to follow
} __attribute__ ((__packed__)) scf_fapi_pdsch_pdu_t;

typedef struct
{
    uint16_t data_scrambling_id;
    uint8_t  num_of_layers;
    uint8_t  transmission_scheme;
    uint8_t  ref_point;

    // DMRS [TS38.211 sec 7.4.1.1]
    uint16_t dl_dmrs_sym_pos;
    uint8_t  dmrs_config_type;
    uint16_t dl_dmrs_scrambling_id;
    uint8_t  sc_id;
    uint8_t  num_dmrs_cdm_grps_no_data;
    uint16_t dmrs_ports;

    // Pdsch Allocation in frequency domain [TS38.214, sec 5.1.2.2]
    uint8_t  resource_alloc;
    uint8_t  rb_bitmap[36];
    uint16_t rb_start;
    uint16_t rb_size;
    uint8_t  vrb_to_prb_mapping;

    // Resource Allocation in time domain [TS38.214, sec 5.1.2.1]
    uint8_t start_sym_index;
    uint8_t num_symbols;

    uint8_t next[0];
} __attribute__ ((__packed__)) scf_fapi_pdsch_pdu_end_t;

typedef struct
{
    // PTRS [TS38.214, sec 5.1.6.3] [SCF 222, sec 3.4.2.2]
    uint8_t  ptrs_port_index;
    uint8_t  ptrs_time_density;
    uint8_t  ptrs_freq_density;
    uint8_t  ptrs_re_offset;
    uint8_t  n_epre_ratio_of_pdsch_to_ptrs;

    uint8_t next[0];
} __attribute__ ((__packed__)) scf_fapi_pdsch_ptrs_t;


// Table 3-43 Tx Precoding and Beamforming PDU
typedef struct
{
    uint16_t pmi_idx;
    uint16_t beam_idx[0];
} __attribute__ ((__packed__)) scf_fapi_tx_pm_idx_beam_idx_t;

typedef struct 
{
    uint16_t num_prgs;
    uint16_t prg_size;
    uint8_t  dig_bf_interfaces;
    uint16_t pm_idx_and_beam_idx[0];
} __attribute__ ((__packed__)) scf_fapi_tx_precoding_beamforming_t;
typedef struct
{
    uint8_t  power_control_offset;
    uint8_t  power_control_offset_ss;
} __attribute__ ((__packed__)) scf_fapi_tx_power_info_t; // part of larger pdsch PDU message

typedef struct
{
    uint8_t  is_last_cb_present;
    uint8_t  is_inline_tb_crc;
    uint32_t dl_tb_crc;
} __attribute__ ((__packed__)) scf_fapi_pdsch_cbg_t; // part of larger pdsch PDU message

// Table 3-39 CSI-RS PDU
typedef struct
{
    scf_fapi_bwp_ts38_213_sec_12_t bwp;

    uint16_t start_rb;
    uint16_t num_of_rbs;
    uint8_t  csi_type;
    uint8_t  row;
    uint16_t freq_domain;
    uint8_t  sym_l0;
    uint8_t  sym_l1;
    uint8_t  cdm_type;
    uint8_t  freq_density;
    uint16_t scrambling_id;
    scf_fapi_tx_power_info_t tx_power;
    scf_fapi_tx_precoding_beamforming_t pc_and_bf;
} __attribute__ ((__packed__)) scf_fapi_csi_rsi_pdu_t;

// Table 3-41 MAC generated MIB PDU
typedef struct
{
    uint32_t bch_payload;
} __attribute__ ((__packed__)) scf_fapi_mac_gen_mib_pdu_t;

// Table 3-42 PHY generated MIB PDU
typedef struct
{
    uint8_t dmrs_type_a_position;
    uint8_t pdcch_config_sib_1;
    uint8_t cell_barred;
    uint8_t intra_freq_reselection;
} __attribute__ ((__packed__)) scf_fapi_phy_gen_mib_pdu_t;

// Table 3-40 SSB/PBCH PDU
typedef struct
{
    uint16_t phys_cell_id;
    uint8_t  beta_pss;
    uint8_t  ssb_block_index;
    uint8_t  ssb_subcarrier_offset;
    uint16_t ssb_offset_point_a;
    uint8_t  bch_payload_flag;

    // bchPayload See Table 3-41 and Table 3-42
    union {
        scf_fapi_phy_gen_mib_pdu_t phy;
        scf_fapi_mac_gen_mib_pdu_t mac;
        uint32_t agg;
    } mib_pdu;

    uint8_t pc_and_bf[0];
    // Precoding and Beamforming See Table 3-43
} __attribute__ ((__packed__)) scf_fapi_ssb_pdu_t;

/************************************************
 *  3.4.3 UL_TTI.request
 ***********************************************/

// Table 3-44 UL_TTI.request message body
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint8_t  num_pdus;
    uint8_t  rach_present;
    uint8_t  num_ulsch;
    uint8_t  num_ulcch;
    uint8_t  ngroup;
    uint8_t  payload[0];
} __attribute__ ((__packed__)) scf_fapi_ul_tti_req_t;

// Table 3-45 PRACH PDU
typedef struct
{
    uint16_t phys_cell_id;
    uint8_t  num_prach_ocas;
    uint8_t  prach_format;
    uint8_t  num_ra;
    uint8_t  prach_start_symbol;
    uint16_t num_cs;
    scf_fapi_rx_beamforming_t beam_index; // See Table 3-53
} __attribute__ ((__packed__)) scf_fapi_prach_pdu_t;

// Table 3-46 PUSCH PDU
typedef struct
{
    uint16_t pdu_bitmap;
    uint16_t rnti;
    uint32_t handle;

    scf_fapi_bwp_ts38_213_sec_12_t bwp;

    // Codeword information
    uint16_t target_code_rate;
    uint8_t  qam_mod_order;
    uint8_t  mcs_index;
    uint8_t  mcs_table;
    uint8_t  transform_precoding;
    uint16_t data_scrambling_id;
    uint8_t  num_of_layers;

    // DMRS [TS38.211 sec 6.4.1.1]
    uint16_t ul_dmrs_sym_pos;
    uint8_t  dmrs_config_type;
    uint16_t ul_dmrs_scrambling_id;
    uint16_t pusch_identity;
    uint8_t  scid;
    uint8_t  num_dmrs_cdm_groups_no_data;
    uint16_t dmrs_ports;

    // Pusch Allocation in frequency domain [TS38.214, sec 6.1.2.2]
    uint8_t  resource_alloc;
    uint8_t  rb_bitmap[36];
    uint16_t rb_start;
    uint16_t rb_size;
    uint8_t  vrb_to_prb_mapping;
    uint8_t  frequency_hopping;
    uint16_t tx_direct_current_location;
    uint8_t  ul_frequency_shift_7p5_khz;

    // Resource Allocation in time domain [TS38.214, sec 5.1.2.1]
    uint8_t  start_symbol_index;
    uint8_t  num_of_symbols;

    uint8_t payload[0];
    // Optional Data only included if indicated in pduBitmap
    //      puschData See Table 3-47
    //      puschUci See Table 3-48
    //      puschPtrs See Table 3-49
    //      dftsOfdm See Table 3-50
    // Beamforming See Table 3-53
} __attribute__ ((__packed__)) scf_fapi_pusch_pdu_t;

// Table 3-47 Optional puschData information
typedef struct
{
    uint8_t  rv_index;
    uint8_t  harq_process_id;
    uint8_t  new_data_indicator;
    uint32_t tb_size;
    uint16_t num_cb;
    uint8_t  cb_present_and_position[0];
} __attribute__ ((__packed__)) scf_fapi_pusch_data_t;

// Table 3-48 Optional puschUci information
typedef struct
{
    uint16_t harq_ack_bit_length;
    uint16_t csi_part_1_bit_length;
#ifdef SCF_FAPI_10_04
    uint16_t flag_csi_part2;
#else
    uint16_t csi_part_2_bit_length;
#endif
    uint8_t  alpha_scaling;
    uint8_t  beta_offset_harq_ack;
    uint8_t  beta_offset_csi_1;
    uint8_t  beta_offset_csi_2;
} __attribute__ ((__packed__)) scf_fapi_pusch_uci_t;

// Table 3-49 Optional puschPtrs information
typedef struct
{
    uint8_t  num_ptrs_ports;
    uint8_t  payload[0];
} __attribute__ ((__packed__)) scf_fapi_pusch_ptrs_t;

// Table 3-50 Optional dftsOfdm information
typedef struct
{
    uint8_t  lowPaprGroupNumber;
    uint16_t lowPaprSequenceNumber;

    uint8_t  ulPtrsSampleDensity;
    uint8_t  ulPtrsTimeDensityTransformPrecoding;
} __attribute__ ((__packed__)) scf_fapi_pusch_dftsofdm_t;

// Table 3–94 PUSCH maintenance FAPIv3 from FAPI 10.04
typedef struct
{
    uint8_t  puschTransType;
    uint16_t deltaBwp0StartFromActiveBwp;
    uint16_t initialUlBwpSize;
    uint8_t  groupOrSequenceHopping;
    uint16_t puschSecondHopPRB;
    uint8_t  ldpcBaseGraph;
    uint32_t tbSizeLbrmBytes;
} __attribute__ ((__packed__)) scf_fapi_pusch_maintenance_t;

typedef struct {
   uint16_t  paramOffsets[0];
} __attribute__ ((__packed__)) scf_uci_csip2_part_param_offset_t;

typedef struct {
   uint8_t  paramSizes[0];
} __attribute__ ((__packed__)) scf_uci_csip2_part_param_size_t;

typedef struct {
    uint16_t part2SizeMapIndex;
    uint8_t part2SizeMapScope;
} __attribute__ ((__packed__)) scf_uci_csip2_part_scope_t;

typedef struct {
    uint16_t priority;
    uint8_t numPart1Params;
} __attribute__ ((__packed__)) scf_uci_csip2_part_t;

// Table 3-95 Uci information for determining UCI Part1 to Part2 correspondence, added in FAPIv3 from 10.04
typedef struct {
    uint16_t numPart2s;
    uint8_t payload[0];
} __attribute__ ((__packed__)) scf_uci_csip2_info_t;

typedef struct
{
    uint16_t ptrs_port_index;
    uint8_t  ptrs_dmrs_port;
    uint8_t  ptrs_e_offset;
} __attribute__ ((__packed__)) scf_fapi_pusch_ptrs_port_info_t;

typedef struct
{
    uint8_t  ptrs_time_density;
    uint8_t  ptrs_freq_density;
    uint8_t  ul_ptrs_power;
} __attribute__ ((__packed__)) scf_fapi_pusch_ptrs_end_t;

typedef struct
{
    uint8_t num_ue;
    uint8_t pdu_index[0];
} __attribute__ ((__packed__)) scf_fapi_ue_group_t;

// PUSCH extension for weighted average CFO estimation
typedef struct
{
    uint8_t  n_iterations;
    uint8_t  ldpc_early_termination;
    uint8_t  fo_forget_coeff;
} __attribute__ ((__packed__)) scf_fapi_pusch_extension_t;
// Table 3-51 PUCCH PDU
typedef struct
{
    // uint16_t pdu_bitmap;
    uint16_t rnti;
    uint32_t handle;

    scf_fapi_bwp_ts38_213_sec_12_t bwp;

    uint8_t  format_type;
    uint8_t  multi_slot_tx_indicator;
    uint8_t  pi_2_bpsk;

    // Pucch Allocation in frequency domain [38.213, sec 9.2.1]
    uint16_t prb_start;
    uint16_t prb_size;

    //Pucch Allocation in time domain
    uint8_t  start_symbol_index;
    uint8_t  num_of_symbols;

    // Hopping information [38.211, sec 6.3.2.2.1]
    uint8_t  freq_hop_flag;
    uint16_t  second_hop_prb;
    uint8_t  group_hop_flag;
    uint8_t  seq_hop_flag;
    uint16_t hopping_id;
    uint16_t initial_cyclic_shift;

    uint16_t data_scrambling_id;
    uint8_t  time_domain_occ_idx;
    uint8_t  pre_dft_occ_idx;
    uint8_t  pre_dft_occ_len;

    // DMRS [38.211 sec 6.4.1.3]  
    uint8_t  add_dmrs_flag;
    uint16_t dmrs_scrambling_id;
    uint8_t  dmrs_cyclic_shift;

    uint8_t  sr_flag;
    uint16_t bit_len_harq;
    uint16_t bit_len_csi_part_1;
    uint16_t bit_len_csi_part_2;

    uint8_t payload[0];
    // Beamforming See Table 3-53
} __attribute__ ((__packed__)) scf_fapi_pucch_pdu_t;

typedef struct
{
    uint16_t srs_bandwidth_start;
    uint8_t sequence_group;
    uint8_t sequence_number;
}__attribute__ ((__packed__))srs_bw_sq_info_t;

typedef struct
{
    uint16_t srs_bw_size; /* srsBandwidthSize */
    srs_bw_sq_info_t srs_bw_sq_info[MAX_SRS_SYMBOLS];
    uint32_t usage;
    /* If a single bit is ‘1’ in Usage, then ReportType is a scalar,
    designating the reporttype for that usage whose bit is set.
    Having only 1 SRS Report for a single SRS PDU is a limitation of FPAPI 10.04
    Since usage can only be set for one type of SRS report, having an array for report_type is needed */
    //uint8_t report_type[MAX_SRS_REPORT_TYPE]; /* ReportType[] */
    uint8_t report_type;
    uint8_t sing_val_rep; /* singular Value Representation */
    uint8_t iq_repr; /* iq Representation */
    uint16_t prg_size; /* prgSize */
    uint8_t num_of_tot_ue_ant; /* numTotalUeAntennas */
    uint32_t ue_ant_in_this_srs_res_set; /* ueAntennasInThisSrsResourceSet */
    uint32_t samp_ue_ant; /* sampledUeAntennas */
    uint8_t rep_scope; /* reportScope */
    uint8_t num_ul_spat_strm_ports; /* NumULSpatialStreamsPorts */
    uint8_t ul_spat_strm_ports[0];	/* UlSpatialStreamPorts[NumULSpatialStreamsPorts] */
}__attribute__ ((__packed__)) scs_fapi_v4_srs_params_t;

// Table 3-52 SRS PDU
typedef struct
{
    uint16_t rnti;
    /* handle is a 32-bit field, 
        0 to 7 bits => For future use
        8 to 23 bits => SRS Chest Buffer Index
        24 to 31 bits => For future use  
    */
    uint32_t handle;
    scf_fapi_bwp_ts38_213_sec_12_t bwp;
    uint8_t     num_ant_ports;
    uint8_t     num_symbols;
    uint8_t     num_repetitions;
    uint8_t     time_start_position;
    uint8_t     config_index;
    uint16_t    sequenceId;
    uint8_t     bandwidth_index;
    uint8_t     comb_size;
    uint8_t     comb_offset;
    uint8_t     cyclic_shift;
    uint8_t     frequency_position;
    uint16_t    frequency_shift;
    uint8_t     frequency_hopping;
    uint8_t     group_or_sequence_hopping;
    uint8_t     resource_type;
    uint16_t    t_srs;
    uint16_t    t_offset;
    uint8_t     payload[0];
    // Beamforming See Table 3-53
} __attribute__ ((__packed__)) scf_fapi_srs_pdu_t;


/************************************************
 *  3.4.4 UL_DCI.request
 ***********************************************/

// Table 3-54 UL_DCI.request message body
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint8_t  num_pdus;
    uint8_t  payload[0]; // scf_fapi_generic_pdu_info_t
} __attribute__ ((__packed__)) scf_fapi_ul_dci_t;

/************************************************
 *  3.4.5 SLOT errors
 ***********************************************/

/************************************************
 *  3.4.6 Tx_Data.request
 ***********************************************/

// Table 3-58 TxData.request message
typedef struct
{
    uint32_t pdu_len;
    uint16_t pdu_index;
#ifdef SCF_FAPI_10_04
    uint8_t cw_index;
#endif
    uint32_t num_tlv;
    scf_fapi_tl_t  tlvs[0];
} __attribute__ ((__packed__)) scf_fapi_tx_data_pdu_info_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint16_t num_pdus;
    scf_fapi_tx_data_pdu_info_t payload[0];
} __attribute__ ((__packed__)) scf_fapi_tx_data_req_t;

#ifdef SCF_FAPI_10_04
typedef struct 
{
    int16_t ul_sinr_metric;
    uint16_t timing_advance;
    int16_t timing_advance_ns;
    uint16_t rssi;
    uint16_t rsrp;
} __attribute__ ((__packed__)) scf_fapi_ul_meas_common_t;
#endif 
/************************************************
 *  3.4.7 Rx_Data.indication
 ***********************************************/

// Table 3-61 RX_Data.indication message body
typedef struct
{
    uint32_t handle;
    uint16_t rnti;
#ifdef SCF_FAPI_10_04
    uint8_t rapid;
#endif
    uint8_t  harq_id;
#ifdef SCF_FAPI_10_04
    uint32_t pdu_len;
    uint8_t pdu_tag;
#else
    uint32_t pdu_len;
#endif

#ifndef SCF_FAPI_10_04
    uint8_t  ul_cqi;
    uint16_t timing_advance;
    uint16_t rssi;
#endif
    uint8_t  pdu[0];
} __attribute__ ((__packed__)) scf_fapi_rx_data_pdu_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
#ifdef SCF_FAPI_10_04
    uint16_t control_length;
#endif
    uint16_t num_pdus;
    scf_fapi_rx_data_pdu_t pdus[0];
} __attribute__ ((__packed__)) scf_fapi_rx_data_ind_t;

/************************************************
 *  3.4.8 CRC.indication
 ***********************************************/

// Table 3-62 CRC.indication message body
typedef struct
{
#ifdef SCF_FAPI_10_04
    scf_fapi_ul_meas_common_t measurement;
#else 
    uint8_t  ul_cqi;
    uint16_t timing_advance;
    uint16_t rssi;
#endif
} __attribute__ ((__packed__)) scf_fapi_crc_end_info_t;

typedef struct
{
    uint32_t handle;
    uint16_t rnti;
#ifdef SCF_FAPI_10_04
    uint8_t rapid;
#endif
    uint8_t  harq_id;
    uint8_t  tb_crc_status;
    uint16_t num_cb;
    uint8_t  cb_crc_status[0];
} __attribute__ ((__packed__)) scf_fapi_crc_info_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint16_t num_crcs;
    scf_fapi_crc_info_t crc_info[0];
} __attribute__ ((__packed__)) scf_fapi_crc_ind_t;

/************************************************
 *  3.4.9 UCI.indication
 ***********************************************/

// Table 3-63 UCI.indication message body
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint16_t num_ucis;
    uint8_t  payload[0];
} __attribute__ ((__packed__)) scf_fapi_uci_ind_t;

typedef struct
{
    uint16_t pdu_type;
    uint16_t pdu_size;
    uint8_t  payload[0];
} __attribute__ ((__packed__)) scf_fapi_uci_pdu_t;

// Table 3-64 UCI PUSCH PDU
typedef struct
{
    uint8_t     pdu_bitmap;
    uint32_t    handle;
    uint16_t    rnti;
#if SCF_FAPI_10_04
    scf_fapi_ul_meas_common_t measurement;
#else
    uint8_t     ul_cqi;
    uint16_t    timing_advance;
    uint16_t    rssi;
#endif
    uint8_t     payload[0];
    // HARQ information
    // CSI part1 information
    // CSI part2 information
} __attribute__ ((__packed__)) scf_fapi_uci_pusch_pdu_t;

// Table 3-65 UCI PUCCH Format 0 or 1 PDU
// Table 3-66 UCI PUCCH Format 2, 3 or 4 PDU
typedef struct
{
    uint8_t  pdu_bitmap;
    uint32_t handle;
    uint16_t rnti;
    uint8_t  pucch_format;
#if SCF_FAPI_10_04
    scf_fapi_ul_meas_common_t measurement;
#else
    uint8_t  ul_cqi;
    uint16_t timing_advance;
    uint16_t rssi;
#endif
    uint8_t  payload[0];
    // Format 0,1
    // SR information
    // HARQ information

    // Format 2,3,4
    // SR information
    // HARQ information
    // CSI part1 information
    // CSI part2 information
}  __attribute__ ((__packed__)) scf_fapi_pucch_format_hdr;

// Table 3-67 SR PDU for Format 0 or 1
typedef struct
{
    uint8_t  sr_indication;
    uint8_t  sr_confidence_level;
} __attribute__ ((__packed__)) scf_fapi_sr_format_0_1_info_t;

// Table 3-68 HARQ PDU for Format 0 or 1
typedef struct
{
    uint8_t  num_harq;
    uint8_t  harq_confidence_level;
    uint8_t  harq_value[0];
} __attribute__ ((__packed__)) scf_fapi_harq_info_t;

// Table 3-69 SR PDU for Format 2, 3 or 4
typedef struct
{
    uint16_t sr_bit_len;
    uint8_t  sr_payload[0];
} __attribute__ ((__packed__)) scf_fapi_sr_format_2_3_4_info_t;

// Table 3-70 HARQ PDU for Format 2, 3 or 4
typedef struct
{
#ifdef SCF_FAPI_10_04
    uint8_t harq_detection_status;
#else
    uint8_t  harq_crc;
#endif
    uint16_t harq_bit_len;
    uint8_t  harq_payload[0];
} __attribute__ ((__packed__)) scf_fapi_harq_format_2_3_4_info_t;

// Table 3-71 CSI Part1 PDU
typedef struct
{
#ifdef SCF_FAPI_10_04
    uint8_t csi_part1_detection_status;
#else
    uint8_t  csi_part_1_crc;
#endif
    uint16_t csi_part_1_bit_len;
    uint8_t  csi_part_1_payload[0];
} __attribute__ ((__packed__)) scf_fapi_csi_part_1_t;

// Table 3-72 CSI Part2 PDU
typedef struct
{
#ifdef SCF_FAPI_10_04
    uint8_t csi_part2_detection_status;
#else
    uint8_t  csi_part_2_crc;
#endif
    uint16_t csi_part_2_bit_len;
    uint8_t  csi_part_2_payload[0];
} __attribute__ ((__packed__)) scf_fapi_csi_part_2_t;

/************************************************
 *  3.4.10 SRS.indication
 ***********************************************/
/* Table 3–131 FAPIv3 Beamforming report, with PRG-level resolution */
typedef struct
{
    uint16_t num_prgs;
    uint8_t rb_snr[0];
} __attribute__ ((__packed__)) per_symbol_numPrg_snr_info_t;

typedef struct
{
    uint16_t prg_size;
    uint8_t num_symbols;
    uint8_t wideband_snr;
    uint8_t num_reported_symbols;
    per_symbol_numPrg_snr_info_t num_prg_snr_info[0];
} __attribute__ ((__packed__)) scf_fapi_v3_bf_report_t;

/* Table 3–132 Normalized Channel I/Q Matrix */
typedef struct
{
    uint8_t norma_iq_repr; /* Normalized iq Representation */
    uint16_t num_gnb_ant_elmts; /* numGnbAntennaElements */ 
    uint16_t num_ue_srs_ports; /* numUeSrsPorts */
    uint16_t prg_size; /* prgSize */
    uint16_t num_prgs; /* numPRGs */
    uint8_t arr_rep_ch_mat_h[0]; /* Array representing channel matrix H */
} __attribute__ ((__packed__)) scf_fapi_norm_ch_iq_matrix_info_t;

/* Table 3-130 SrsReport-TLV structure */
typedef struct
{
    uint16_t tag;
    uint32_t length;
    uint32_t value;
}__attribute__ ((__packed__))scf_fapi_srs_tl_t;

typedef struct
{
    uint16_t numRBs;
    uint8_t  rbSNR[0];
} __attribute__ ((__packed__)) scf_fapi_srs_sym_report_t;

typedef struct
{
    /* handle is a 32-bit field, 
        0 to 7 bits => For future use
        8 to 23 bits => SRS Chest Buffer Index
        24 to 31 bits => For future use  
    */
    uint32_t handle;
    uint16_t rnti;
    uint16_t timing_advance;
#ifdef SCF_FAPI_10_04_SRS
    int16_t timing_advance_ns;
    uint8_t srs_usage;
    uint8_t report_type;
    scf_fapi_srs_tl_t srs_report_tlv;
#else
    uint8_t numSymbols;
    uint8_t wideBandSNR;
    uint8_t numReportedSymbols;
    uint8_t report[0];  // packes scf_fapi_srs_sym_report_t
#endif
} __attribute__ ((__packed__)) scf_fapi_srs_info_t;


// Table 3-73 SRS.indication message body
typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
#ifdef SCF_FAPI_10_04_SRS
    uint16_t control_length;
#endif
    uint8_t  num_pdus;
    scf_fapi_srs_info_t srs_info[0];
} __attribute__ ((__packed__)) scf_fapi_srs_ind_t;

typedef struct
{
    uint32_t handle;
    uint16_t rnti;
    uint16_t timing_advance;
    uint8_t num_symbols;
    uint8_t wide_band_snr;
    uint8_t num_reported_symbols;
    uint8_t payload[0];
} __attribute__ ((__packed__)) scf_fapi_srs_ind_pdu_start_t;

typedef struct
{
    uint16_t num_rbs;
    uint8_t rb_snr[0];
} __attribute__ ((__packed__)) scf_fapi_srs_ind_pdu_end_t;


/************************************************
 *  3.4.11 RACH.indication
 ***********************************************/

// Table 3-74 RACH.indication message body
typedef struct
{
    uint8_t  preamble_index;
    uint16_t timing_advance;
    uint32_t preamble_power;
} __attribute__ ((__packed__)) scf_fapi_prach_preamble_info_t;

typedef struct
{
    uint16_t phys_cell_id;
    uint8_t  symbol_index;
    uint8_t  slot_index;
    uint8_t  freq_index;
    uint8_t  avg_rssi;
    uint8_t  avg_snr;
    uint8_t  num_preamble;
    scf_fapi_prach_preamble_info_t  preamble_info[0];
} __attribute__ ((__packed__)) scf_fapi_prach_ind_pdu_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint8_t  num_pdus;
    scf_fapi_prach_ind_pdu_t pdu_info[0];
} __attribute__ ((__packed__)) scf_fapi_rach_ind_t;

typedef struct
{
    int16_t digBeamWeightRe;
    int16_t digBeamWeightIm;
} __attribute__ ((__packed__)) scf_fapi_digBeamWeight_t;


typedef struct
{
    uint16_t         beamIdx;
    scf_fapi_digBeamWeight_t  digBeamWeightPerTxRU[0];
}__attribute__ ((__packed__)) scf_fapi_digBeam_t;

// SCF 10.02 Table 3-32
typedef struct
{
    uint16_t numDigBeams;
    uint16_t numTXRUs;
    scf_fapi_digBeam_t digBeam[0];
} __attribute__ ((__packed__)) scf_fapi_dbt_pdu_t;

typedef struct
{
    int16_t prc_wt_re;
    int16_t prc_wt_im;
} __attribute__ ((__packed__)) prc_wt_re_im_t;

// SCF 10.02 Table 3-33
typedef struct
{
    uint16_t pmi_idx;
    uint16_t num_layers;
    uint16_t num_ant_ports;
    prc_wt_re_im_t prc_wt_re_im[0];
} __attribute__ ((__packed__)) scf_fapi_pm_pdu_t;

typedef struct
{
    uint32_t handle;
    uint16_t rnti;
    uint16_t meas;
} __attribute__ ((__packed__)) scf_fapi_meas_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint16_t num_meas;
    scf_fapi_meas_t meas_info[0];
} __attribute__ ((__packed__)) scf_fapi_rx_measurement_ind_t;

typedef struct
{
    uint16_t phyCellId;
    uint8_t  freqIndex;
    uint16_t meas;
} __attribute__ ((__packed__)) scf_fapi_prach_interference_t;

typedef struct
{
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint16_t num_meas;
    scf_fapi_prach_interference_t meas_info[0];
} __attribute__ ((__packed__)) scf_fapi_prach_interference_ind_t;

typedef struct {
    uint8_t ue_ant_index;
} __attribute__((__packed__)) scf_dl_bfw_ue_ant_t;

typedef struct {
    uint16_t rnti;
    /* handle is a 32-bit field, 
        0 to 7 bits => For future use
        8 to 23 bits => SRS Chest Buffer Index
        24 to 31 bits => For future use  
    */
    uint32_t handle;
    uint16_t pduIndex;
    uint8_t  gnb_ant_index_start;
    uint8_t  gnb_ant_index_end;
    uint8_t  num_ue_ants;
    uint8_t  payload[0];
} __attribute__((__packed__)) scf_dl_bfw_config_start_t;

typedef struct {
    uint16_t rb_start;
    uint16_t rb_size;
    uint16_t num_prgs;
    uint16_t prg_size;
    uint8_t nUes;
    uint8_t payload[0];
} __attribute__((__packed__)) scf_dl_bfw_config_t;

typedef struct {
    uint16_t pdu_size;
    scf_dl_bfw_config_t dl_bfw_cvi_config;
} __attribute__((__packed__)) scf_fapi_dl_bfw_group_config_t;

typedef struct {
    scf_fapi_body_header_t         msg_hdr;
    uint16_t sfn;
    uint16_t slot;
    uint8_t npdus;
    scf_fapi_dl_bfw_group_config_t config_pdu[0];
} __attribute__((__packed__)) scf_fapi_dl_bfw_cvi_request_t;

#ifdef __cplusplus
using scf_fapi_ul_bfw_cvi_request_t = scf_fapi_dl_bfw_cvi_request_t;
using scf_fapi_ul_bfw_group_config_t = scf_fapi_dl_bfw_group_config_t;
using scf_ul_bfw_config_t = scf_dl_bfw_config_t;
using scf_ul_bfw_config_start_t = scf_dl_bfw_config_start_t;
using scf_ul_bfw_ue_ant_t = scf_dl_bfw_ue_ant_t;
#else
typedef struct scf_fapi_dl_bfw_cvi_request_t scf_fapi_ul_bfw_cvi_request_t;
typedef struct scf_fapi_dl_bfw_group_config_t scf_fapi_ul_bfw_group_config_t;
typedef struct scf_dl_bfw_config_t scf_ul_bfw_config_t;
typedef struct scf_dl_bfw_config_start_t scf_ul_bfw_config_start_t;
typedef struct scf_dl_bfw_config_end_t scf_ul_bfw_config_end_t;
typedef struct scf_dl_bfw_ue_gnb_ant_t scf_ul_bfw_ue_gnb_ant_t;
typedef struct scf_dl_bfw_ue_ant_t scf_ul_bfw_ue_ant_t;
#endif

/************************************************
 * CV MEMORY BANK CONFIG REQUEST
 * Vendor specific message - testMac to read the
 * SRS Channel Estimates from BFW TV and send it
 * using this message to Aerial. Aerial will use
 * the received SRS ChEsts to populate CV Memory
 * Bank
 ***********************************************/
typedef struct
{
    uint16_t rnti;
    uint8_t  reportType;
    uint16_t startPrbGrp;
    uint32_t srsPrbGrpSize;
    uint16_t nPrbGrps;
    uint8_t  nGnbAnt;
    uint8_t  nUeAnt;
    uint32_t offset;
} __attribute__ ((__packed__)) ue_cv_info;

typedef struct
{
    uint8_t numUes;
    ue_cv_info cv_info[0];  
} __attribute__ ((__packed__)) cv_mem_bank_config_request_body_t;

typedef struct
{
    scf_fapi_body_header_t            msg_hdr;
    cv_mem_bank_config_request_body_t msg_body;
} __attribute__ ((__packed__)) cv_mem_bank_config_request_msg_t;


typedef struct {
    uint16_t type;
    uint16_t chan_start_offset;
    uint16_t chan_duration;
} __attribute__ ((__packed__)) scf_channel_segment_info_t;

typedef struct {
    uint8_t nPduSegments;
    uint8_t payload[0];
} __attribute__ ((__packed__)) scf_channel_segment_t;

typedef enum
{
    SCF_CHAN_SEG_PDSCH            = 1<<0,
    SCF_CHAN_SEG_PUSCH            = 1<<1,
    SCF_CHAN_SEG_PBCH             = 1<<2,
    SCF_CHAN_SEG_PDCCH            = 1<<3,
    SCF_CHAN_SEG_PUCCH            = 1<<4,
    SCF_CHAN_SEG_CSIRS            = 1<<5,
    SCF_CHAN_SEG_SRS              = 1<<6,
    SCF_CHAN_SEG_PRACH            = 1<<7
} scf_fapi_ch_seg_type_t;
