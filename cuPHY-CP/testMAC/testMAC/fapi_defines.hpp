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

#ifndef _FAPI_DEFINES_HPP_
#define _FAPI_DEFINES_HPP_

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <map>
#include <vector>
#include <atomic>
#include <iostream>
#include <unordered_map>

#include "cuphy.h"
#include "common_defines.hpp"

static constexpr uint8_t MAX_DCI_PER_CORESET  = CUPHY_PDCCH_MAX_DCIS_PER_CORESET;
static constexpr uint8_t MAX_DCI_PAYLOAD_SIZE = CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES;

#define PUSCH_BITMAP_DATA (0x1)
#define PUSCH_BITMAP_UCI (0x2)
#define PUSCH_BITMAP_DFTSOFDM (0x8)

#define PUCCH_BITMAP_SR (0x1)
#define PUCCH_BITMAP_HARQ (0x2)
#define PUCCH_BITMAP_CSI_P1 (0x4)
#define PUCCH_BITMAP_CSI_P2 (0x8)

// Validation tolerance values
#define VALD_TOLERANCE_RACH_AVG_RSSI (2)
#define VALD_TOLERANCE_RACH_AVG_SNR (2)
#define VALD_TOLERANCE_RACH_PREAMBLE_PWR (1000)

#define VALD_TOLERANCE_TIMING_ADVANCE (1)
#define VALD_TOLERANCE_TIMING_ADVANCE_LOW_SNR (12) // 3us for mu of 1
#define VALD_TOLERANCE_TIMING_ADVANCE_NS (1000) //1us
#define VALD_TOLERANCE_TIMING_ADVANCE_NS_LOW_SNR (3000) //3us

#define VALD_TOLERANCE_SINR (500)
#define VALD_THRESHOLD_SINR (-5000) // -10dB
#define VALD_THRESHOLD_CQI  (-20) // -10dB
#define VALD_TOLERANCE_RSSI (10)
#define VALD_TOLERANCE_RSRP (10)
#define VALD_TOLERANCE_RSRP_LOW (40)
#define VALD_TOLERANCE_UL_CQI (2)

#define VALD_TOLERANCE_SINR_BFP9 (1500)
#define VALD_TOLERANCE_RSSI_BFP9 (30)
#define VALD_TOLERANCE_RSRP_BFP9 (30)
#define VALD_TOLERANCE_UL_CQI_BFP9 (6)

#define VALD_TOLERANCE_SINR_PF1_20DB (1500)

#define VALD_TOLERANCE_MEAS_RACH_AVG_NOISE (10)
#define VALD_TOLERANCE_MEAS_PUSCH_NOISE (10)
#define VALD_TOLERANCE_MEAS_PUSCH_NOISE_BFP9 (30)
#define VALD_TOLERANCE_MEAS_PUCCH_NOISE (10)

#define VALD_TOLERANCE_UL_SLOT_WINDOW (8)
#define VALD_TOLERANCE_UL_SLOT_WINDOW_64T64R (9)
// SRS Late deadline after T0 + 11 slots for 8 SRS PDU's, T0 + 14 for 16 SRS PDU's and T0 + 12 for 8 SRS PDU's (4 UE Ant ports 3 Cells)
#define VALD_TOLERANCE_SRS_SLOT_WINDOW_64T64R (14) //TODO: To be tuned after SRS Optimization for higher Ant ports and UE's

#define VALD_TOLERANCE_SRS_WIDEBAND_SNR (3)
#define VALD_TOLERANCE_SRS_PER_RB_SNR (5)

#define NUM_SUPPORTED_SRS_PDU 2
#define MAX_NVIPC_FOR_MEM_BANK_CV_CONFIG_REQ 96
static constexpr int  MAX_SRS_CHEST_BUFFERS_PER_CELL = 1024;
static constexpr int  MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL = 256;

#define CUPHY_N_MAX_UCI_BITS_RM (11)

// DL FAPI message order in a slot, support one instance per slot for each cell
typedef enum
{
    DL_TTI_REQ    = 0,
    UL_TTI_REQ    = 1,
    TX_DATA_REQ   = 2,
    UL_DCI_REQ    = 3,
    DL_BFW_CVI_REQ= 4,
    UL_BFW_CVI_REQ= 5,
    FAPI_REQ_SIZE = 6
} fapi_group_t;

typedef enum
{
    PUSCH       = 0,
    PDSCH       = 1,
    PDCCH_UL    = 2,
    PDCCH_DL    = 3,
    PBCH        = 4,
    PUCCH       = 5,
    PRACH       = 6,
    CSI_RS      = 7,
    SRS         = 8,
    BFW_DL      = 9,
    BFW_UL      = 10,
    CHANNEL_MAX = 11
} channel_type_t;

static constexpr const char* bfw_dl_str = "BFW_DL";
static constexpr const char* bfw_ul_str = "BFW_UL";

static inline const char* get_channel_name(int channel)
{
    switch(channel)
    {
    case PUSCH:
        return "PUSCH";
    case PDSCH:
        return "PDSCH";
    case PDCCH_UL:
        return "PDCCH_UL";
    case PDCCH_DL:
        return "PDCCH_DL";
    case PBCH:
        return "PBCH";
    case PUCCH:
        return "PUCCH";
    case PRACH:
        return "PRACH";
    case CSI_RS:
        return "CSI_RS";
    case SRS:
        return "SRS";
    case BFW_DL:
        return "BFW_DL";
    case BFW_UL:
        return "BFW_UL";
    default:
        return "INVALID";
    }
}

static inline channel_type_t get_channel_type(const char* channel_name)
{
    for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
    {
        const char* name = get_channel_name(ch);
        if(strncmp(name, channel_name, strlen(name)) == 0)
        {
            return channel_type_t(ch);
        }
    }
    return channel_type_t::CHANNEL_MAX;
}

typedef enum
{
    TV_NONE     = 0,
    TV_PBCH     = 1,
    TV_PDCCH    = 2,
    TV_PDSCH    = 3,
    TV_CSI_RS   = 4,
    TV_PRACH    = 6,
    TV_PUCCH    = 7,
    TV_PUSCH    = 8,
    TV_SRS      = 9,
    TV_BFW_DL   = 10,
    TV_BFW_UL   = 11,
    TV_PDCCH_UL = 12

} tv_channel_type_t;

typedef enum
{
    TV_GENERIC    = 0,
    TV_PRACH_MSG2 = 2,
    TV_PRACH_MSG3 = 3,
    TV_PRACH_MSG4 = 4,
    TV_HARQ       = 5,
} tv_type_t;

typedef enum
{
    BFP9  = 0,
    BFP14 = 1,
    BFP16 = 2
} bfp_t;

static inline tv_channel_type_t get_tv_channel_type(channel_type_t channel)
{
    switch(channel)
    {
    case PUSCH:
        return TV_PUSCH;
    case PDSCH:
        return TV_PDSCH;
    case PDCCH_UL:
        return TV_PDCCH_UL;
    case PDCCH_DL:
        return TV_PDCCH;
    case PBCH:
        return TV_PBCH;
    case PUCCH:
        return TV_PUCCH;
    case PRACH:
        return TV_PRACH;
    case CSI_RS:
        return TV_CSI_RS;
    case SRS:
        return TV_SRS;
    case BFW_DL:
        return TV_BFW_DL;
    case BFW_UL:
        return TV_BFW_UL;
    default:
        return TV_NONE;
    }
}

typedef struct
{
    std::vector <float>    tv_delay;
    std::vector <float>    tv_peak;
    uint32_t tv_numPrmb;
    std::vector<uint32_t>  tv_prmbIdx;
    std::vector<uint32_t>  tv_nPreambPwr;
    std::vector<uint32_t>  tv_nTA;
} preamble_params_t;

typedef struct
{
    uint32_t UL_CQI;
    int16_t  SNR;
    int16_t  SNR_ehq;
    uint32_t TimingAdvance;
    int16_t  TimingAdvanceNs;
    uint32_t RSSI;
    uint32_t RSSI_ehq;
    uint32_t RSRP;
    uint32_t RSRP_ehq;
} ul_measurement_t;

typedef enum
{
    UCI_PDU_TYPE_PUSCH = 0,
    UCI_PDU_TYPE_PF01  = 1,
    UCI_PDU_TYPE_PF234 = 2,
    UCI_PDU_TYPE_NUM   = 3,
} uci_pdu_type_t;

// FAPI validation tolerance values for each cell
typedef struct {
    ul_measurement_t ul_meas[UCI_PDU_TYPE_NUM]; // Total 3 PDU types: 0 - PUSCH, 1 - PUCCH PF01, 2 - PUCCH PF234.
    int16_t pusch_pe_noiseVardB;
} vald_tolerance_t;

typedef struct
{
    uint32_t dlGridSize;
    uint32_t ulGridSize;
    uint32_t dlBandwidth;
    uint32_t ulBandwidth;
    uint32_t numTxAnt;
    uint32_t numTxPort;
    uint32_t numRxAnt;
    uint32_t numRxPort;
    uint32_t mu;
    uint32_t phyCellId;
    uint32_t dmrsTypeAPos;
    uint32_t frameDuplexType;
    uint32_t BFP;
    uint32_t pusch_sinr_selector; // 0 - Disabled; 1 - PostEq; 2 - PreEq.
    uint32_t enableWeightedAverageCfo; // 0 - Disabled; 1 - Enabled.
    uint32_t negTV_enable;
    uint32_t enable_dynamic_BF;
    uint32_t enable_stat_dyn_bf;
    vald_tolerance_t tolerance;
} cell_configs_t;

typedef struct
{
    uint16_t prachRootSequenceIndex;
    uint8_t  numRootSequences;
    uint16_t k1;
    uint8_t  prachZeroCorrConf;
    uint16_t numUnusedRootSequences;

    std::vector<uint16_t> unusedRootSequences;
} prach_fd_occasion_config_t;

typedef struct
{
    uint8_t prachSequenceLength;
    uint8_t prachSubCSpacing;
    uint8_t restrictedSetConfig;
    uint8_t numPrachFdOccasions;
    uint8_t prachConfigIndex;
    uint8_t SsbPerRach;
    uint8_t prachMultipleCarriersInABand;

    std::vector<prach_fd_occasion_config_t> prachFdOccasions;
} prach_configs_t;

typedef struct
{
    uint32_t Crc;
    uint32_t BitLen;
    uint32_t DetectionStatus;
    std::vector<uint8_t> Payload;
} csi_part_t;

typedef struct
{
    uint8_t  ul_cqi;
    uint16_t timing_advance;
    uint16_t rssi;
} __attribute__ ((__packed__)) fapi_ul_measure_10_02_t;

// PUCCH indication validation fields (ind.type=17)
typedef struct
{
    // Common fields for all PUCCH format 0~5
    uint32_t idxInd;
    uint32_t idxPdu;
    uint32_t PucchFormat;

    ul_measurement_t meas;

    // PUCCH format 0~1 fields
    uint32_t SRindication;
    uint32_t SRconfidenceLevel;
    uint32_t NumHarq;
    uint32_t HarqconfidenceLevel;
    std::vector<uint8_t> HarqValue;

    // PUCCH format 2~4 fields
    int16_t  noiseVardB;
    uint32_t SrBitLen;
    uint32_t HarqCrc;
    uint32_t HarqBitLen;
    uint32_t HarqDetectionStatus;
    csi_part_t csi_parts[2];
    std::vector<uint8_t> SrPayload;
    std::vector<uint8_t> HarqPayload;
} pucch_uci_ind_t;

typedef struct
{
    int16_t re;
    int16_t im;
} complex_int16_t;

// SCF 222 Table 3-33
typedef struct
{
    uint16_t PMidx;
    uint16_t numLayers;
    uint16_t numAntPorts;

    std::vector<complex_int16_t> precoderWeight_v;
} precoding_matrix_t;

// SCF 222 Table 3-43. Included in PDCCH, PDSCH, CSI-RS and SSB PDUs.
typedef struct
{
    uint16_t numPRGs;
    uint16_t prgSize;
    uint8_t  digBFInterfaces;

    std::vector<uint16_t> PMidx_v;
    std::vector<uint16_t> beamIdx_v;
} tx_beamforming_data_t;

// SCF 222 Table 3-53. Included in PUCCH, PUSCH, PRACH and SRS PDUs.
typedef struct
{
    uint16_t              numPRGs;
    uint16_t              prgSize;
    uint8_t               digBFInterfaces;
    std::vector<uint16_t> beamIdx_v;
} rx_beamforming_data_t;

typedef struct
{
    uint16_t              numPRGs;
    uint16_t              prgSize;
    uint8_t               digBFInterfaces;
#ifdef SCF_FAPI_10_04
    std::vector<std::vector<uint16_t>> beamIdx_v;
#else
    std::vector<uint16_t> beamIdx_v;
#endif
} rx_srs_beamforming_data_t;

typedef struct {
    uint16_t type;
    uint16_t channel_start_offset;
    uint16_t channel_duration;
} channel_segment_t;

typedef struct
{
    uint8_t  betaPss;
    uint8_t  ssbBlockIndex;
    uint8_t  ssbSubcarrierOffset;
    uint8_t  bchPayloadFlag;
    uint16_t physCellId;
    uint16_t SsbOffsetPointA;
    uint32_t bchPayload;

    tx_beamforming_data_t tx_beam_data;
} pbch_tv_data_t;

typedef struct
{
    std::vector<pbch_tv_data_t> data;
} pbch_tv_t;

typedef struct
{
    uint32_t numPrmb;

    std::vector<uint32_t> prmbIdx_v;
    std::vector<float>    delay_v;
    std::vector<float>    peak_v;
} prach_tv_ref_t;

typedef struct
{
    // INDx
    uint32_t idxInd;
    uint32_t idxPdu;
    uint32_t SymbolIndex;
    uint32_t SlotIndex;
    uint32_t FreqIndex;
    uint32_t avgRssi;
    uint32_t avgSnr;
    uint32_t avgNoise;
    uint32_t numPreamble;

    // INDx_PreamblePwr, INDx_TimingAdvance, INDx_preambleIndex
    std::vector<uint32_t> preambleIndex_v;
    std::vector<uint32_t> TimingAdvance_v;
    std::vector<uint32_t> TimingAdvanceNano_v;
    std::vector<uint32_t> PreamblePwr_v;
} prach_ind_t;

typedef struct
{
    uint32_t type;
    uint32_t physCellID;
    uint32_t NumPrachOcas;
    uint32_t prachFormat;
    uint32_t numRa;
    uint32_t prachStartSymbol;
    uint32_t numCs;
    uint32_t prachPduIdx;

    prach_ind_t ind;

    rx_beamforming_data_t rx_beam_data;
    prach_tv_ref_t        ref;
} prach_tv_data_t;

typedef struct
{
    std::vector<prach_tv_data_t> data;
} prach_tv_t;

typedef struct dciinfo_
{
    // DATASET "DCI_DL" or "DCI_UL"
    uint16_t RNTI;
    uint16_t ScramblingId;
    uint16_t ScramblingRNTI;
    uint8_t  CceIndex;
    uint8_t  AggregationLevel;
    uint8_t  beta_PDCCH_1_0;
#ifdef SCF_FAPI_10_04
    int8_t powerControlOffsetSSProfileNR;
#else
    uint8_t  powerControlOffsetSS;
#endif
    uint16_t PayloadSizeBits;
    uint8_t  Payload[MAX_DCI_PAYLOAD_SIZE];
    tx_beamforming_data_t tx_beam_data;
} dciinfo_t;

typedef struct coresetinfo_
{
    // DATASET "PDCCH_DL" or "PDCCH_UL"
    uint16_t  BWPSize;
    uint16_t  BWPStart;
    uint8_t   SubcarrierSpacing;
    uint8_t   CyclicPrefix;
    uint8_t   StartSymbolIndex;
    uint8_t   DurationSymbols;
    uint64_t  FreqDomainResource;
    uint8_t   CceRegMappingType;
    uint8_t   RegBundleSize;
    uint8_t   InterleaverSize;
    uint8_t   CoreSetType;
    uint16_t  ShiftIndex;
    uint8_t   precoderGranularity;
    uint8_t   testModel;
    uint16_t  numDlDci;
    // dciinfo_t dciList[CUPHY_PDCCH_MAX_DCIS_PER_CORESET];
    std::vector<dciinfo_t> dciList;
} corsetinfo_t;

typedef struct
{
    std::vector<coresetinfo_> coreset;
} pdcch_tv_t;

typedef struct
{
    uint16_t priority;
    uint8_t  num_part1_paramas;
    std::vector <uint16_t> param_offsets;
    std::vector <uint8_t> param_sizes;
    uint16_t part2_size_map_index;
}uci_part1_to_part2_params_t;
typedef struct 
{
    uint16_t num_part_2s;
    std::vector <uci_part1_to_part2_params_t> uci_part1_to_part2_params;
}uci_part1_to_part2_correspondence_params_t;

typedef struct
{
    uint32_t type;
    uint32_t RNTI;
    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t SubcarrierSpacing;
    uint32_t CyclicPrefix;
    uint32_t FormatType;
    uint32_t multiSlotTxIndicator;
    uint32_t pi2Bpsk;
    uint32_t prbStart;
    uint32_t prbSize;
    uint32_t StartSymbolIndex;
    uint32_t NrOfSymbols;
    uint32_t freqHopFlag;
    uint32_t secondHopPRB;
    uint32_t groupHopFlag;
    uint32_t sequenceHopFlag;
    uint32_t hoppingId;
    uint32_t InitialCyclicShift;
    uint32_t dataScramblingId;
    uint32_t TimeDomainOccIdx;
    uint32_t PreDftOccIdx;
    uint32_t PreDftOccLen;
    uint32_t AddDmrsFlag;
    uint32_t DmrsScramblingId;
    uint32_t DMRScyclicshift;
    uint32_t SRFlag;
    uint32_t BitLenHarq;
    uint32_t BitLenCsiPart1;
    uint32_t BitLenCsiPart2;
    uint32_t nBits;
    uint32_t bitLenSr;
    uint32_t SRindication;
    uint32_t maxCodeRate;
    uint32_t pucchPduIdx;
    float    RSRP;
    float    snrdB;

    pucch_uci_ind_t uci_ind;

    rx_beamforming_data_t rx_beam_data;
    uci_part1_to_part2_correspondence_params_t uci_part1_to_part2_correspondence_params;
    std::vector<uint8_t>  Payload;
} pucch_tv_data_t;

typedef struct
{
    std::vector<pucch_tv_data_t> data;
} pucch_tv_t;

typedef struct
{
    // struct CsirsParams csirs_pars;
    uint32_t type;
    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t SubcarrierSpacing;
    uint32_t CyclicPrefix;
    uint32_t StartRB;
    uint32_t NrOfRBs;
    uint32_t CSIType;
    uint32_t Row;
    uint32_t FreqDomain;
    uint32_t SymbL0;
    uint32_t SymbL1;
    uint32_t CDMType;
    uint32_t FreqDensity;
    uint32_t ScrambId;
    uint32_t powerControlOffset;
    uint32_t powerControlOffsetSS;
    uint32_t csirsPduIdx;
    uint32_t lastCsirsPdu;

    tx_beamforming_data_t tx_beam_data;
} csirs_tv_data_t;

typedef struct
{
    std::vector<csirs_tv_data_t> data;
} csirs_tv_t;

typedef struct
{
    struct tb_pars tbpars;

    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t SubcarrierSpacing;
    uint32_t CyclicPrefix;
    uint32_t targetCodeRate;
    uint32_t FrequencyHopping;
    uint32_t txDirectCurrentLocation;
    uint32_t uplinkFrequencyShift7p5khz;
    uint32_t ref_point;

    uint32_t numDmrsCdmGrpsNoData;
    uint32_t dmrsSymLocBmsk;
    uint32_t resourceAlloc;
    uint8_t  portIndex[8];

    uint32_t VRBtoPRBMapping;
    uint32_t transmissionScheme;

    uint32_t powerControlOffset;
    uint32_t powerControlOffsetSS;

    uint8_t  testModel;

    size_t   tb_size;
    uint8_t* tb_buf;

    std::vector<uint32_t> rbBitmap;

    tx_beamforming_data_t tx_beam_data;
} pdsch_tv_data_t;

typedef struct
{
    tv_type_t type; // Whether it is a PRACH Msg 2~4, HARQ or not
    size_t    data_size;

    std::vector<uint8_t>    data_buf;
    std::vector<pdsch_tv_data_t*> data;
} pdsch_tv_t;

typedef enum
{
    IND_PUSCH_UCI  = 15,
    IND_PRACH      = 16,
    IND_PUCCH      = 17,
    IND_PUSCH_DATA = 18,
} indication_type_t;

typedef enum
{
    UCI_PDU_PF0   = 0,
    UCI_PDU_PF1   = 1,
    UCI_PDU_PF2   = 2,
    UCI_PDU_PF3   = 3,
    UCI_PDU_PF4   = 4,
    UCI_PDU_PUSCH = 5,
} uci_pdu_format_t;

// PUSCH data indication validation fields (ind.type=18)
typedef struct
{
    ul_measurement_t meas;

    uint32_t idxInd;
    uint32_t idxPdu;
    uint32_t TbCrcStatus;
    uint32_t NumCb;

    int16_t sinrdB;
    int16_t postEqSinrdB;
    int16_t noiseVardB;
    int16_t postEqNoiseVardB;

    std::vector<uint8_t> CbCrcStatus;
} pusch_data_ind_t;

// UCI on PUSCH indication validation fields (ind.type=15)
typedef struct
{
    ul_measurement_t meas;

    uint32_t idxInd;
    uint32_t idxPdu;
    uint8_t  isEarlyHarq;
    uint32_t HarqCrc;
    uint32_t HarqBitLen;
    uint32_t HarqDetStatus_earlyHarq;
    uint32_t HarqDetectionStatus;

    int16_t sinrdB;
    int16_t sinrdB_ehq;
    int16_t postEqSinrdB;
    int16_t noiseVardB;
    int16_t postEqNoiseVardB;

    csi_part_t csi_parts[2];
    std::vector<uint8_t> HarqPayload;
    std::vector<uint8_t> HarqPayload_earlyHarq;
} pusch_uci_ind_t;

typedef struct {
    uint16_t priority;
    uint8_t numPart1Params;
    std::vector<uint16_t> paramOffsets;
    std::vector<uint8_t> paramSizes;
    uint16_t part2SizeMapIndex;
    uint8_t part2SizeMapScope;
} csi_part2_info_t;

typedef struct
{
    struct tb_pars tbpars;

    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t pduBitmap;
    uint32_t harqAckBitLength;
    uint32_t csiPart1BitLength;
#ifdef SCF_FAPI_10_04
    uint16_t flagCsiPart2;
#else
    uint32_t csiPart2BitLength;
#endif
    uint32_t alphaScaling;
    uint32_t betaOffsetHarqAck;
    uint32_t betaOffsetCsi1;
    uint32_t betaOffsetCsi2;
    uint32_t harqProcessID;
    uint32_t newDataIndicator;
    uint32_t qamModOrder;
    uint32_t tbErr; // Negative test flag
    uint32_t SubcarrierSpacing;
    uint32_t CyclicPrefix;
    uint32_t targetCodeRate;
    uint32_t FrequencyHopping;
    uint32_t txDirectCurrentLocation;
    uint32_t uplinkFrequencyShift7p5khz;
    uint32_t numDmrsCdmGrpsNoData;
    uint32_t dmrsSymLocBmsk;
    bfp_t    BFP;

    // DFT-s-OFDM
    uint8_t  TransformPrecoding;
    uint16_t puschIdentity;
    uint8_t  groupOrSequenceHopping;
    uint8_t  lowPaprGroupNumber;
    uint16_t lowPaprSequenceNumber;

    // Weighted average CFO estimation
    float    foForgetCoeff;
    uint8_t  nIterations;
    uint8_t  ldpcEarlyTermination;

    // PUSCH UCI CSIP2 Info
    uint16_t numPart2s;
    std::vector<csi_part2_info_t> csip2_v3_parts;
    pusch_data_ind_t data_ind;

    pusch_uci_ind_t  uci_ind;

    size_t   tb_size;
    uint8_t* tb_buf;

    rx_beamforming_data_t rx_beam_data;
} pusch_tv_data_t;

typedef struct
{
    size_t   data_size;

    std::vector<uint8_t>    data_buf;
    std::vector<pusch_tv_data_t*> data;
    uint32_t negTV_enable;
} pusch_tv_t;

typedef struct {
    uint32_t taOffset;
    int16_t taOffsetNs;
    uint8_t wideBandSNR;
    uint16_t prgSize;
    uint8_t numSymbols;
    uint8_t numReportedSymbols;
    uint16_t numPRGs;
} srs_ind0_t;

using srs_iq_sample_t = float2;

typedef struct {
    uint16_t numUeSrsAntPorts;
    uint16_t numGnbAntennaElements;
    uint16_t prgSize;
    uint16_t numPRGs;
    std::vector<srs_iq_sample_t> report_iq_data;
} srs_ind1_t;
typedef struct
{
    uint32_t idxInd;
    uint32_t idxPdu;
    srs_ind0_t ind0;
    srs_ind1_t ind1;
} srs_ind_t;

typedef struct {
    uint32_t usage;
    uint32_t numTotalUeAntennas;
    uint32_t ueAntennasInThisSrsResourceSet;
    uint32_t sampledUeAntennas;
} srs_fapiv4_t;
typedef struct
{
    uint32_t type;
    uint32_t RNTI;
    uint32_t srsChestBufferIndex;
    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t SubcarrierSpacing;
    uint32_t CyclicPrefix;

    uint32_t numAntPorts;
    uint32_t numSymbols;
    uint32_t numRepetitions;
    uint32_t timeStartPosition;
    uint32_t configIndex;
    uint32_t sequenceId;
    uint32_t bandwidthIndex;
    uint32_t combSize;
    uint32_t combOffset;
    uint32_t cyclicShift;
    uint32_t frequencyPosition;
    uint32_t frequencyShift;
    uint32_t frequencyHopping;
    uint32_t groupOrSequenceHopping;
    uint32_t resourceType;
    uint32_t Tsrs;
    uint32_t Toffset;
    uint32_t Beamforming;
    uint32_t numPRGs;
    uint32_t prgSize;
    uint32_t digBFInterfaces;
    std::vector<uint32_t> beamIdx;

    uint8_t srsPduIdx;
    uint32_t lastSrsPdu;

    rx_srs_beamforming_data_t rx_beam_data;
    
    srs_fapiv4_t fapi_v4_params;
    srs_ind_t ind;
    std::vector<uint8_t> SNRval;
} srs_tv_data_t;

typedef struct
{
    std::vector<srs_tv_data_t> data;
} srs_tv_t;

typedef struct
{
    uint16_t RNTI;
    uint8_t reportType;
    uint32_t nGnbAnt;
    uint32_t nPrbGrps;
    uint32_t nUeAnt;
    uint32_t startPrbGrp;
    uint32_t srsPrbGrpSize;
    std::vector<uint8_t> cv_samples;
} cv_membank_config_t;

using cv_membank_config_list = std::vector<cv_membank_config_t>;

typedef struct {
    cv_membank_config_list data;
} cv_membank_config_req_t;

typedef struct {
    uint16_t RNTI;
    uint32_t srsChestBufferIndex;
    uint16_t pduIndex;
    uint8_t numOfUeAnt;
    std::vector<uint8_t> ueAntIndexes;
    uint8_t gNbAntIdxStart;
    uint8_t gNbAntIdxEnd;
} bfw_cv_ue_data_t;

using bfw_cv_ue_grp_data_t = std::vector<bfw_cv_ue_data_t>;
typedef struct {
    uint8_t nUes;
    bfw_cv_ue_grp_data_t ue_grp_data;
    uint16_t rbStart;
    uint16_t rbSize;
    uint16_t numPRGs;
    uint16_t prgSize;
    uint8_t bfwUl;
} bfw_cv_data_t;

typedef struct
{
    std::vector<bfw_cv_data_t> data;
} bfw_cv_tv_t;
using dset_list    = std::vector<std::string>;
using tv_dset_list = std::unordered_map<channel_type_t, dset_list>;

typedef struct
{
    pdcch_tv_t   pdcch_tv;
    pdsch_tv_t   pdsch_tv;
    pucch_tv_t   pucch_tv;
    pusch_tv_t   pusch_tv;
    prach_tv_t   prach_tv;
    pbch_tv_t    pbch_tv;
    csirs_tv_t   csirs_tv;
    srs_tv_t     srs_tv;
    bfw_cv_tv_t  bfw_tv;
    tv_dset_list dset_tv;
} test_vector_t;

typedef struct
{
    int              cell_idx;
    int              slot_idx;
    std::string      tv_file;
    channel_type_t   channel;
    test_vector_t*   tv_data;

    hdf5hpp::hdf5_file* h5f;
} fapi_req_t;

typedef struct
{
    std::atomic<uint64_t> ontime;
    std::atomic<uint64_t> late;
    std::atomic<uint64_t> early;
} timing_t;

// The throughput data for each channel
// typedef struct
class thrput_t {
public:
    thrput_t() {
        reset();
    }

    thrput_t(const thrput_t& obj) {
        reset();
    }

    void reset() {
        ul_thrput = 0;
        dl_thrput = 0;
        mac_ul_drop = 0;
        prmb = 0;
        sr = 0;
        harq = 0;
        csi1 = 0;
        csi2 = 0;
        srs = 0;
        error = 0;
        invalid = 0;
        for (int ch = 0; ch < CHANNEL_MAX; ch++)
        {
            slots[ch] = 0;
        }
        uci.early = 0;
        uci.late = 0;
        uci.ontime = 0;
        ul_ind.early = 0;
        ul_ind.late = 0;
        ul_ind.ontime = 0;
        prach_ind.early = 0;
        prach_ind.late = 0;
        prach_ind.ontime = 0;
        srs_ind.early = 0;
        srs_ind.late = 0;
        srs_ind.ontime = 0;
    }

    // DL/UL Data throughput
    std::atomic<uint64_t> ul_thrput;
    std::atomic<uint64_t> dl_thrput;
    std::atomic<uint64_t> mac_ul_drop; // For CRC error case

    // PUCCH parameters count
    std::atomic<uint32_t> prmb;
    std::atomic<uint32_t> sr;
    std::atomic<uint32_t> harq;
    std::atomic<uint32_t> csi1;
    std::atomic<uint32_t> csi2;
    std::atomic<uint32_t> srs;

    // ERROR.indication counter
    std::atomic<uint32_t> error;

    // Validation failure counter
    std::atomic<uint32_t> invalid;

    // Slot count for each channel - per second
    std::atomic<uint32_t> slots[CHANNEL_MAX];
    // Slot count for each channel (raw) per schedule pattern
    std::atomic<uint32_t> lp_slots[CHANNEL_MAX];

    timing_t uci;
    timing_t ul_ind;
    timing_t prach_ind;
    timing_t srs_ind;
}; // thrput_t;

typedef struct results_summary_t results_summary_t;
struct results_summary_t
{
public:
    results_summary_t() {
        uci.early = 0;
        uci.late = 0;
        uci.ontime = 0;
        ul_ind.early = 0;
        ul_ind.late = 0;
        ul_ind.ontime = 0;
        prach_ind.early = 0;
        prach_ind.late = 0;
        prach_ind.ontime = 0;
        srs_ind.early = 0;
        srs_ind.late = 0;
        srs_ind.ontime = 0;
    }
    results_summary_t(const results_summary_t&& obj) {
        uci.early = 0;
        uci.late = 0;
        uci.ontime = 0;
        ul_ind.early = 0;
        ul_ind.late = 0;
        ul_ind.ontime = 0;
        prach_ind.early = 0;
        prach_ind.late = 0;
        prach_ind.ontime = 0;
        srs_ind.early = 0;
        srs_ind.late = 0;
        srs_ind.ontime = 0;
    }
    timing_t uci;
    timing_t ul_ind;
    timing_t prach_ind;
    timing_t srs_ind;
};

typedef struct {
    uint32_t pduBitmap;
    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t SubCarrierSpacing;
    uint32_t CyclicPrefix;
    uint32_t NrOfCodeWords;
    uint32_t rvIndex;
    uint32_t dataScramblingId;
    uint32_t transmission;
    uint32_t refPoint;
    uint32_t dlDmrsScrmablingId;
    uint32_t scid;
    uint32_t resourceAlloc;
    uint32_t VRBtoPRBMapping;
    uint32_t powerControlOffset;
    uint32_t powerControlOffsetSS;

    std::vector<uint32_t> numDmrsCdmGrpsNoData;
    std::vector<uint32_t> rbBitmap;

    tx_beamforming_data_t tx_beam_data;
} pdsch_static_param_t;

typedef struct {
    uint32_t pduBitmap;
    uint32_t BWPSize;
    uint32_t BWPStart;
    uint32_t SubCarrierSpacing;
    uint32_t CyclicPrefix;
    uint32_t dataScramblingId;
    uint32_t dmrsConfigType;
    uint32_t ulDmrsScramblingId;
    uint32_t puschIdentity;
    uint32_t scid;
    uint32_t resourceAlloc;
    uint32_t VRBtoPRBMapping;
    uint32_t FrequencyHopping;
    uint32_t txDirectCurrentLocation;
    uint32_t uplinkFrequencyShift7p5khz;
    uint32_t rvIndex;
    uint32_t harqProcessID;
    uint32_t newDataIndicator;
    uint32_t numCb;
    uint32_t cbPresentAndPosition;

    std::vector<uint32_t> numDmrsCdmGrpsNoData;
    std::vector<uint32_t> rbBitmap;

    rx_beamforming_data_t rx_beam_data;
} pusch_static_param_t;

// For Spectral Efficiency test
typedef struct
{
    pdsch_static_param_t pdsch;
    pusch_static_param_t pusch;
} static_slot_param_t;

typedef struct {
    uint32_t prbStart;
    uint32_t prbEnd;
} prb_info_t;

typedef struct
{
    prb_info_t prb;
    uint32_t rnti;
    uint32_t beam;
    uint32_t layer;
    uint32_t mcs_table;
    uint32_t mcs;
    uint32_t dmrs_port_bmsk;
    uint32_t dmrs_sym_loc_bmsk;
    uint32_t nrOfSymbols;

    // Calculate from above parameters
    uint32_t modulation_order;
    uint32_t target_code_rate;
    uint32_t tb_size;
} dyn_pdu_param_t;

typedef struct
{
    channel_type_t ch_type; // PDSCH, PUSCH
    std::vector<dyn_pdu_param_t> pdus;
} dyn_slot_param_t;

using dbt_data_t = std::vector<complex_int16_t>;

struct dbt_md_ {
    bool bf_stat_dyn_enabled;
    uint16_t num_static_beamIdx;  // Number of Static Beams
    uint16_t num_TRX_beamforming; // Numeber of Baseband ports
    dbt_data_t dbt_data_buf;
    std::unordered_map<uint16_t, bool> static_beamIdx_seen;
};
using dbt_md_t = dbt_md_;

struct csi2_map_params_
{
    uint8_t numPart1Params;
    std::vector<uint8_t> sizePart1Params;
    uint8_t mapBitWidth;
    std::vector<uint16_t> map;
};
using csi2_map_params_t = csi2_map_params_;
struct csi2_maps_ {

    uint32_t nCsi2Maps;
    std::vector <csi2_map_params_t> mapParams;
    size_t totalSizeInBytes;
};

using csi2_maps_t = csi2_maps_;

#endif /* _FAPI_DEFINES_HPP_ */
