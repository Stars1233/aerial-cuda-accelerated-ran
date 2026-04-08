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
#if !defined(SLOT_COMMAND_API_HPP_INCLUDED_)
#define SLOT_COMMAND_API_HPP_INCLUDED_

#include <chrono>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include <algorithm>
#include <utility>
#include <cstring>
#include <atomic>
#include <stdio.h>

#include "nv_ipc.h"

#include "cuphy_api.h"
#include "cuphy.h"
#include "memtrace.h"

#include "aerial-fh-driver/oran.hpp"

/// Maximum DL eAxC IDs including mMIMO
#define MAX_DL_EAXCIDS 32 //TODO: Investigate why reducing to 16 causes more sendCplane timing check errors

/// Packed attribute for structure alignment
#define __sc_packed __attribute__((__packed__))
#define PRB_INFO_ARRAY
/// Number of TX ports
#define NUM_TX_PORT 16
/// Number of RX ports
#define NUM_RX_PORT 16
/**
 * @brief Slot command API namespace
 *
 * Contains data structures and constants for managing PHY slot commands,
 * including PDSCH, PUSCH, PDCCH, PUCCH, PRACH, SRS, CSI-RS, and SSB.
 */
struct ReleasedHarqBufferInfo;
namespace slot_command_api
{

using namespace std::chrono;

/// Maximum cells per cell group
inline constexpr int MAX_CELLS_PER_CELL_GROUP = 48;

/// Maximum PDCCH PDUs per cell (needed for per-PDU management vectors)
inline constexpr int MAX_PDCCH_PDUS_PER_CELL = 20;

#ifdef ENABLE_64C
inline constexpr int MAX_PUSCH_UE_GROUPS = 256;   ///< Maximum PUSCH UE groups (64 cell config)
inline constexpr int MAX_PUSCH_UE_PER_TTI = 256;  ///< Maximum PUSCH UE per TTI (64 cell config)
#else
inline constexpr int MAX_PUSCH_UE_GROUPS = 132;   ///< Maximum PUSCH UE groups (default config)
inline constexpr int MAX_PUSCH_UE_PER_TTI = 132;  ///< Maximum PUSCH UE per TTI (default config)
#endif
inline constexpr int MAX_NUM_PRGS_DBF=273;  ///< Maximum number of PRGs for dynamic beamforming
inline constexpr int MAX_NUM_PRGS=10;       ///< Maximum number of PRGs

/// Keeping it separate constant in case there is asymmetry between DL and UL
#ifdef ENABLE_64C
inline constexpr int MAX_PDSCH_UE_GROUPS = 256;   ///< Maximum PDSCH UE groups (64 cell config)
inline constexpr int MAX_PDSCH_UE_PER_TTI = 256;  ///< Maximum PDSCH UE per TTI (64 cell config)
#else
inline constexpr int MAX_PDSCH_UE_GROUPS = 192;   ///< Maximum PDSCH UE groups (default config)
inline constexpr int MAX_PDSCH_UE_PER_TTI = 192;  ///< Maximum PDSCH UE per TTI (default config)
#endif
/// Maximum PDSCH UE codewords per TTI (TODO: For SU-MIMO or MU-MIMO with > 4 layers per UE, 2 CW is needed)
inline constexpr int MAX_PDSCH_UE_CW_PER_TTI = MAX_PDSCH_UE_PER_TTI;
inline constexpr uint MAX_DCI_PAYLOAD_LENGTH_BITS = 45;  ///< Maximum DCI payload length in bits

inline constexpr int MAX_PUCCH_UE_PER_TTI = 64;         ///< Maximum PUCCH UE per TTI
inline constexpr int MAX_SSB_BLOCKS_PER_SLOT = 3;       ///< Maximum SSB blocks per slot
inline constexpr int MAX_PRACH_MAX_OCCASIONS_PER_CELL = 4;  ///< Maximum PRACH occasions per cell
inline constexpr int MAX_PRACH_OCCASIONS_PER_SLOT = (MAX_PRACH_MAX_OCCASIONS_PER_CELL * MAX_CELLS_PER_SLOT);  ///< Maximum PRACH occasions per slot
inline constexpr int MAX_CSIRS_OCCASIONS_PER_SLOT = 64; ///< Maximum CSI-RS occasions per slot
inline constexpr int MAX_SRS_UE_PER_TTI = 16;           ///< Maximum SRS UE per TTI
/// Maximum resource blocks for SRS transmission (Table 6.4.1.4.3-1: Csrs index = 61/62/63 & Bsrs = 0)
inline constexpr int MAX_RB_FOR_SRS_TRANS = 272;
inline constexpr int MIN_PRG_SIZE = 1;                   ///< Minimum PRG size
inline constexpr int NUM_GNB_TX_RX_ANT_PORTS = 64;      ///< Number of gNB TX/RX antenna ports

inline constexpr int MAX_NUM_DIGBFI = 32;  ///< Maximum number of digital beamforming interfaces

inline constexpr int IQ_REPR_FP32_COMPLEX = 2;  ///< IQ representation FP32 complex
/**
 * Maximum MU-MIMO layers
 * Up to 8 DL layers (16 layers not supported)
 * Up to 4 UL layers (8 layers not supported)
 * Up to 4 layers/UE DL (or UL) SU-MIMO
 */
#ifdef ENABLE_32DL
inline constexpr int MAX_MU_MIMO_LAYERS = 32;
#else
inline constexpr int MAX_MU_MIMO_LAYERS = 16;
#endif
 inline constexpr int MAX_PORTS_FOR_STATIC_BF = 32;   ///< Maximum ports for static beamforming
inline constexpr int MAX_DL_UL_BF_UE_PER_TTI = 8;    ///< Maximum DL/UL beamforming UE per TTI
inline constexpr int MAX_DL_UL_BF_UE_GROUPS = CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS;  ///< Maximum DL/UL BF UE groups
inline constexpr int MAX_B_SRS_INDEX = 4;             ///< Maximum B SRS index
inline constexpr int MAX_SRS_SYM = 4;                 ///< Maximum SRS symbols
inline constexpr int MAX_BFW_COFF_STORE_INDEX = 4;   ///< Maximum beamforming coefficient store index
inline constexpr int MAX_STATIC_BFW_COFF_STORE_INDEX = 32;  ///< Maximum static BFW coefficient store index
inline constexpr int MAX_N_BUNDLE = 276;              ///< Maximum N bundles
inline constexpr int MAX_PUCCH_FORMAT = 5;            ///< Maximum PUCCH format
/// Maximum SRS indications per slot (for 16 SRS PDUs in UL_TTI_REQ and nvIPC bufflen as 576000)
inline constexpr int MAX_SRS_IND_PER_SLOT = 24;

inline constexpr int  MAX_SRS_CHEST_BUFFERS_PER_CELL = 1024;
inline constexpr int  MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL = 256;
// TBD: Change to max cells once we support mutiple Cells with MU-MIMO
inline constexpr uint32_t MAX_CELLS_MU_MIMO_ENABLE = 9;
inline constexpr uint32_t MAX_SRS_PDU_PER_SLOT = 128;
inline constexpr uint32_t MAX_SRS_CHEST_BUFFERS = MAX_CELLS_MU_MIMO_ENABLE * MAX_SRS_CHEST_BUFFERS_PER_CELL;
inline constexpr int MAX_CPLANE_SPLIT = 4;
inline constexpr int MAX_AP_PER_SLOT_SRS = 64;
inline constexpr int MAX_STATIC_BFW_BEAM_ID = 1024;
inline constexpr int L2A_PROCESSING_DELTA = 2;
inline constexpr int MIN_SRS_PROCESSING_TIME_SLOTS = 1;
inline constexpr int MAX_SRS_PROCESSING_TIME_SLOTS = 11;
inline constexpr uint32_t MAX_ALLOWED_PDSCH_PDUS_PER_SLOT = 256;

/**
 * @brief O-RAN slot indication structure
 *
 * Represents slot timing in O-RAN format with frame ID, subframe ID, and slot ID.
 */
struct oran_slot_ind final
{
    uint8_t oslotid_;    ///< O-RAN slot ID (0-1)
    uint8_t osfid_;      ///< O-RAN subframe ID (0-9)
    uint8_t oframe_id_;  ///< O-RAN frame ID (0-255)
    uint8_t padding;     ///< Padding for alignment
};

/**
 * @brief 3GPP slot indication structure
 *
 * Represents slot timing with SFN, slot number, and optional timing tick information.
 */
struct slot_indication final
{
    uint64_t tick_;      ///< Timing tick in nanoseconds
    uint64_t t0_;        ///< Reference time T0
    uint16_t sfn_;       ///< System Frame Number (0-1023)
    uint16_t slot_;      ///< Slot number within frame
    bool t0_valid_;      ///< Flag indicating if T0 is valid

    /// Default constructor
    slot_indication(): sfn_(0), slot_(0),tick_(0),t0_(0),t0_valid_(false) { }

    /**
     * @brief Constructor with SFN, slot, and tick
     * @param sfn System Frame Number
     * @param slot Slot number
     * @param tick Timing tick
     */
    explicit slot_indication(const uint16_t sfn, const uint16_t slot, const uint64_t tick):
        sfn_(sfn), slot_(slot), tick_(tick), t0_(0), t0_valid_(false) {
    }
};

/**
 * @brief Convert 3GPP slot indication to O-RAN format
 * @param slot_ind 3GPP slot indication
 * @return O-RAN slot indication
 */
[[nodiscard]] inline oran_slot_ind to_oran_slot_format(const slot_indication & slot_ind)
{
    oran_slot_ind ind;
    ind.oframe_id_ = slot_ind.sfn_ % 256;
    ind.osfid_ = slot_ind.slot_/ 2;
    ind.oslotid_ = slot_ind.slot_ % 2;

    return ind;
}

/**
 * @brief Slot type enumeration for TDD configuration
 */
enum slot_type {
    SLOT_NONE = 0,      ///< No transmission
    SLOT_UPLINK = 1,    ///< Uplink slot
    SLOT_DOWNLINK = 2,  ///< Downlink slot
    SLOT_SPECIAL = 3    ///< Special slot (mixed UL/DL)
};

/**
 * @brief Physical channel type enumeration
 *
 * Ordered with downlink channels first, followed by uplink channels.
 */
enum channel_type {
    NONE = UINT32_MAX,  ///< No channel
    PDSCH_CSIRS = 0,    ///< PDSCH with CSI-RS
    PDSCH = 1,          ///< Physical Downlink Shared Channel
    CSI_RS = 2,         ///< Channel State Information Reference Signal
    PDSCH_DMRS = 3,     ///< PDSCH Demodulation Reference Signal
    PBCH = 4,           ///< Physical Broadcast Channel
    SSB_PBCH_DMRS = 5,  ///< SSB PBCH DMRS
    PDCCH_DL = 6,       ///< Physical Downlink Control Channel (DL)
    PDCCH_UL = 7,       ///< Physical Downlink Control Channel (UL grant)
    PDCCH_DMRS = 8,     ///< PDCCH DMRS
    PUSCH = 9,          ///< Physical Uplink Shared Channel
    PUCCH = 10,         ///< Physical Uplink Control Channel
    PRACH = 11,         ///< Physical Random Access Channel
    SRS = 12,           ///< Sounding Reference Signal
    BFW = 13,           ///< Beamforming Weights
    CHANNEL_MAX = 14    ///< Maximum channel type value
};

/**
 * @brief SSB PBCH component enumeration
 */
enum class ssb_pbch: uint8_t {
    PSS = 0,        ///< Primary Synchronization Signal
    SSS = 1 ,       ///< Secondary Synchronization Signal
    DMRS_DATA = 2   ///< DMRS and PBCH data
};

/**
 * @brief Beamforming weight type
 */
enum bfw_type {
    DL_BFW = 0,     ///< Downlink beamforming weights
    UL_BFW = 1,     ///< Uplink beamforming weights
    BFW_NONE        ///< No beamforming weights
};

/**
 * @brief CSI report type enumeration
 */
enum report_type{
    REPORT_TYPE_CODEBOOK = 0,       ///< Codebook-based report
    REPORT_TYPE_NON_CODEBOOK = 1,   ///< Non-codebook based report
    REPORT_MAX                      ///< Maximum report type value
};

/**
 * @brief Beamforming coefficient memory state flags
 */
enum bfw_coeff_flag_info_t
{
    BFW_COFF_MEM_FREE = 0,              ///< Memory is free
    BFW_COFF_MEM_BUSY = 1,             ///< Memory is busy
    BFW_COFF_MEM_BUSY_TO_BE_USED = 2    ///< Memory is busy but will be used
};

/**
 * @brief Boolean flag enumeration
 */
enum bool_flag_info_t
{
    NV_FALSE = 0,   ///< False value
    NV_TRUE = 1     ///< True value
};

/**
 * @brief SRS channel estimation buffer state
 */
enum srsChestBuffState {
    SRS_CHEST_BUFF_INIT = 0,        ///< Buffer initialized
    SRS_CHEST_BUFF_REQUESTED = 1,   ///< Buffer requested
    SRS_CHEST_BUFF_READY = 2,       ///< Buffer ready for use
    SRS_CHEST_BUFF_NONE = 3         ///< No buffer state
};

/**
 * @brief PDU index and RNTI list information
 *
 * Associates a PDU index with its corresponding Radio Network Temporary Identifier.
 */
struct pdu_idx_rnti_list_info_t
{
    uint16_t pdu_idx;   ///< PDU index
    uint16_t rnti;      ///< Radio Network Temporary Identifier
};


/**
 * @brief Beamforming coefficient memory information
 *
 * Manages memory buffers for beamforming coefficients, including
 * host and device buffer pointers organized by UE groups.
 */
struct bfw_coeff_mem_info_t
{
    uint8_t slotIndex;      ///< Slot index for buffer rotation
    uint16_t sfn;           ///< System Frame Number
    uint16_t slot;          ///< Slot number
    uint16_t nGnbAnt;       ///< Number of gNB antennas
    uint8_t header_size;    ///< Size of header in bytes
    uint8_t* header;        ///< Pointer to header buffer
    uint32_t buff_size;     ///< Total buffer size
    uint32_t buff_chunk_size;  ///< Size of each buffer chunk
    uint8_t num_buff_chunk_busy;  ///< Number of busy buffer chunks
    /// PDU index and RNTI list for each UE group
    pdu_idx_rnti_list_info_t pdu_idx_rnti_list[MAX_DL_UL_BF_UE_GROUPS][MAX_DL_UL_BF_UE_PER_TTI];
    /// Host buffer pointers for each UE group
    uint8_t* buff_addr_chunk_h[MAX_DL_UL_BF_UE_GROUPS];
    /// Device (GPU) buffer pointers for each UE group
    uint8_t* buff_addr_chunk_d[MAX_DL_UL_BF_UE_GROUPS];
};

/**
 * @brief Slot information structure
 *
 * Combines slot type (UL/DL/special) with 3GPP timing information.
 */
struct slot_info
{
    slot_type type;             ///< Slot type (UL/DL/special)
    slot_indication slot_3gpp;  ///< 3GPP slot timing information
};

/**
 * @brief Base channel parameters structure
 *
 * Contains cell index mappings used by all channel types.
 */
struct ch_params
{
    std::vector<int32_t> cell_index_list;      ///< Logical cell indices
    std::vector<int32_t> phy_cell_index_list;  ///< Physical cell indices
};

struct pusch_params : public ch_params
{
    uint32_t ue_tb_size[MAX_PUSCH_UE_PER_TTI];
    uint16_t forcedNumCsi2Bits;
    cuphyPuschCellGrpDynPrm_t cell_grp_info;
    cuphyPuschCellDynPrm_t cell_dyn_info[MAX_CELLS_PER_CELL_GROUP];
    cuphyPuschUeGrpPrm_t ue_grp_info[MAX_PUSCH_UE_GROUPS];
    cuphyPuschDmrsPrm_t ue_dmrs_info[MAX_PUSCH_UE_GROUPS];
    cuphyPuschUePrm_t ue_info[MAX_PUSCH_UE_PER_TTI];
    uint16_t ue_index_info[MAX_PUSCH_UE_GROUPS*MAX_PUSCH_UE_PER_TTI];
    cuphyUciOnPuschPrm_t uci_info[MAX_PUSCH_UE_PER_TTI];
    uint16_t cell_ue_group_idx_start[MAX_CELLS_PER_CELL_GROUP];
    cuphyCalcCsi2SizePrm_t csip2_v3_params[CUPHY_MAX_N_CSI2_REPORTS_PER_UE * MAX_PUSCH_UE_PER_TTI];
    std::vector<uint32_t> scf_ul_tti_handle_list;
    uint32_t nue_grps_per_cell[MAX_CELLS_PER_CELL_GROUP];
    explicit pusch_params():
    ue_tb_size{0},
    cell_grp_info(),
    cell_dyn_info{0},
    ue_grp_info{0},
    ue_dmrs_info{0},
    ue_info{0},
    ue_index_info{0},
    uci_info{0},
    cell_ue_group_idx_start{0},
    nue_grps_per_cell{0},
    csip2_v3_params{0}
    {
        /// Setup cell group of 1 per cell
        // cell_grp_info.nCells = 1;
        cell_grp_info.pCellPrms = cell_dyn_info;

        cell_grp_info.nUeGrps = 0;
        cell_grp_info.pUeGrpPrms = ue_grp_info;

        cell_grp_info.nUes = 0;
        cell_grp_info.pUePrms = ue_info;

        cell_dyn_info[0].cellPrmStatIdx = 0;
        cell_dyn_info[0].cellPrmDynIdx = 0;

        for (uint32_t i = 0; i < MAX_PUSCH_UE_GROUPS; i++)
        {
            ue_grp_info[i].nUes = 0;
            ue_grp_info[i].pUePrmIdxs = &ue_index_info[i*MAX_PUSCH_UE_PER_TTI];
            ue_grp_info[i].pDmrsDynPrm = &ue_dmrs_info[i];
        }

        for (auto& ue_grp: ue_grp_info)
        {
            ue_grp.pCellPrm = cell_dyn_info;
            // ue_grp.nUes = 0;
            // ue_grp.pUePrmIdxs = ue_index_info;
            // ue_grp.pDmrsDynPrm = ue_dmrs_info;
        }
        cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        phy_cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        scf_ul_tti_handle_list.reserve(MAX_PUSCH_UE_GROUPS*MAX_PUSCH_UE_PER_TTI);
    }

    void reset() {
        cell_grp_info.nUeGrps = 0;
        cell_grp_info.nCells = 0;
        cell_grp_info.nUes = 0;

        cell_dyn_info[0].cellPrmStatIdx = 0;
        cell_dyn_info[0].cellPrmDynIdx = 0;

        std::memset(cell_ue_group_idx_start, 0, sizeof(cell_ue_group_idx_start));
        std::memset(nue_grps_per_cell, 0, sizeof(nue_grps_per_cell));
        std::memset(ue_grp_info, 0, sizeof(ue_grp_info));

        for (uint32_t i = 0; i < MAX_PUSCH_UE_GROUPS; i++)
        {
            ue_grp_info[i].nUes = 0;
            ue_grp_info[i].pUePrmIdxs = &ue_index_info[i*MAX_PUSCH_UE_PER_TTI];
            ue_grp_info[i].pDmrsDynPrm = &ue_dmrs_info[i];
            ue_grp_info[i].pCellPrm = cell_dyn_info;
        }

        scf_ul_tti_handle_list.clear(); // will not purge memory
        cell_index_list.clear();
        phy_cell_index_list.clear();
    }
};

struct digBeamWeight_t
{
    int16_t digBeamWeightRe;
    int16_t digBeamWeightIm;
};

struct digBeam_t
{
    uint8_t beamIdxIQSentInCplane;
    std::vector<digBeamWeight_t> digBeam;
};

struct pm_weights_t {
    uint16_t layers;
    uint16_t ports;
    cuphyPmW_t weights;
};

using prc_weights_list_t = std::vector<cuphyPmW_t>;
using prc_weights_idx_list_t = std::vector<uint32_t>;

struct ra_type0_info_t_
{
    uint32_t start_prb{};
    uint32_t num_prb{};
};

struct pdsch_params : public ch_params
{
    cuphyPdschCellGrpDynPrm_t cell_grp_info;
    cuphyPdschCellAerialMetrics_t cell_metrics_info[MAX_CELLS_PER_CELL_GROUP];
    cuphyPdschCellDynPrm_t cell_dyn_info[MAX_PDSCH_UE_PER_TTI];
    cuphyPdschUeGrpPrm_t ue_grp_info[MAX_PDSCH_UE_GROUPS];
    cuphyPdschDmrsPrm_t ue_dmrs_info[MAX_PDSCH_UE_GROUPS];
    cuphyPdschUePrm_t ue_info[MAX_PDSCH_UE_PER_TTI];
    cuphyPdschCwPrm_t ue_cw_info[MAX_PDSCH_UE_CW_PER_TTI];
    cuphyCsirsRrcDynPrm_t csirs_info[MAX_PDSCH_UE_PER_TTI];
    cuphyPdschDataIn_t tb_data;
    uint16_t ue_index_info[MAX_PDSCH_UE_GROUPS*MAX_PDSCH_UE_PER_TTI]{0};
    uint16_t ue_cw_index_info[MAX_PDSCH_UE_CW_PER_TTI]{0};
    uint8_t* ue_tb_ptr[MAX_PDSCH_UE_PER_TTI]{0};
    uint16_t cell_ue_group_idx_start{0};
    typedef uint8_t rb_bitmap[MAX_RBMASK_BYTE_SIZE];
    rb_bitmap ue_rb_bitmap[MAX_PDSCH_UE_GROUPS]{0};
    ra_type0_info_t_ ra_type0_info[MAX_RBMASK_BYTE_SIZE/2+1][MAX_PDSCH_UE_GROUPS];
    //ra_type0_info_t is an array of 19. How many elements in that array
    //are valid - that number is stored in num_ra_type0_info for each UE grp
    uint32_t num_ra_type0_info[MAX_PDSCH_UE_GROUPS]{0};
    prc_weights_list_t pm_info;
    prc_weights_idx_list_t pmw_idx_cache;
    uint32_t num_csirs_info{0};
    uint32_t nue_grps_per_cell[MAX_CELLS_PER_CELL_GROUP]{0};
    uint32_t ue_grp_idx_bfw_id_map[MAX_PDSCH_UE_GROUPS]{0};

    explicit pdsch_params():
    cell_grp_info{0},
    cell_dyn_info{0},
    cell_metrics_info{0},
    ue_info{0},
    ue_grp_info{0},
    ue_dmrs_info{0},
    ue_index_info{0},
    ue_cw_info{0},
    ue_cw_index_info{0},
    ue_rb_bitmap{0},
    num_csirs_info{0},
    cell_ue_group_idx_start{0}
    {
        /// Setup cell group of 1 per cell
        // cell_grp_info.nCells = 1;
        cell_grp_info.pCellPrms = cell_dyn_info;
        cell_grp_info.nCells = 0;

        cell_grp_info.pCellMetrics = cell_metrics_info;

        cell_grp_info.nUeGrps = 0;
        cell_grp_info.pUeGrpPrms = ue_grp_info;

        cell_grp_info.nUes = 0;
        cell_grp_info.pUePrms = ue_info;
        cell_grp_info.nCsiRsPrms = 0;
        cell_grp_info.pCsiRsPrms = csirs_info;

        cell_grp_info.nPrecodingMatrices = 0;
        cell_grp_info.pPmwPrms = nullptr;

        cell_grp_info.nCws = 0;
        cell_grp_info.pCwPrms = ue_cw_info;

        for (uint32_t i = 0; i < MAX_PDSCH_UE_GROUPS; i++)
        {
            ue_grp_info[i].nUes = 0;
            ue_grp_info[i].pUePrmIdxs = &ue_index_info[i*MAX_PDSCH_UE_PER_TTI];
            ue_grp_info[i].pDmrsDynPrm = &ue_dmrs_info[i];
            ue_grp_info[i].rbBitmap = &(ue_rb_bitmap[i][0]);
            ue_grp_idx_bfw_id_map[i] = 0;
            num_ra_type0_info[i] = 0;
        }

        for(uint8_t cellIdx=0; cellIdx < MAX_CELLS_PER_CELL_GROUP; cellIdx++)
        {
            nue_grps_per_cell[cellIdx] = 0;
        }

        cell_dyn_info[0].cellPrmStatIdx = 0;
        cell_dyn_info[0].cellPrmDynIdx = 0;
        cell_dyn_info[0].nCsiRsPrms = 0;
        cell_dyn_info[0].csiRsPrmsOffset = 0;

        for (auto & ue_grp: ue_grp_info)
        {
            ue_grp.pCellPrm = &cell_dyn_info[0];
        //     ue_grp.nUes = 0;
        //     ue_grp.pUePrmIdxs = ue_index_info;
        //     ue_grp.pDmrsDynPrm = ue_dmrs_info;
        }
        std::memset(ue_tb_ptr, 0, MAX_PDSCH_UE_PER_TTI*sizeof(uint8_t *));
        tb_data.pTbInput = &ue_tb_ptr[0];
        pmw_idx_cache.reserve(PDSCH_MAX_UES_PER_CELL_GROUP);
        pm_info.reserve(PDSCH_MAX_UES_PER_CELL_GROUP);
        cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        phy_cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
    }

    void reset() {
        cell_ue_group_idx_start = 0;
        cell_grp_info.nCells = 0;
        cell_grp_info.nUes = 0;
        cell_grp_info.nCsiRsPrms = 0;
        cell_grp_info.nCws = 0;
        cell_grp_info.nUeGrps = 0;

        std::memset(ue_cw_index_info, 0 , sizeof(ue_cw_index_info));
        std::memset(cell_dyn_info, 0 , sizeof(cell_dyn_info));
        std::memset(cell_metrics_info, 0, sizeof(cell_metrics_info));
        std::memset(ue_grp_info, 0, sizeof(ue_grp_info));

        for (uint32_t i = 0; i < MAX_PDSCH_UE_GROUPS; i++)
        {
            ue_grp_info[i].nUes = 0;
            ue_grp_info[i].pUePrmIdxs = &ue_index_info[i*MAX_PDSCH_UE_PER_TTI];
            ue_grp_info[i].pDmrsDynPrm = &ue_dmrs_info[i];
            ue_grp_info[i].rbBitmap = &(ue_rb_bitmap[i][0]);
            ue_grp_idx_bfw_id_map[i] = 0;
        }

        for(uint8_t cellIdx=0; cellIdx < MAX_CELLS_PER_CELL_GROUP; cellIdx++)
        {
            nue_grps_per_cell[cellIdx] = 0;
        }

        cell_dyn_info[0].cellPrmStatIdx = 0;
        cell_dyn_info[0].cellPrmDynIdx = 0;
        cell_dyn_info[0].nCsiRsPrms = 0;
        cell_dyn_info[0].csiRsPrmsOffset = 0;
        std::memset(ue_tb_ptr, 0, MAX_PDSCH_UE_PER_TTI*sizeof(uint8_t *));

        cell_index_list.clear();
        phy_cell_index_list.clear();

        if(cell_grp_info.nPrecodingMatrices > 0)
        {
            cell_grp_info.nPrecodingMatrices = 0;
            cell_grp_info.pPmwPrms = nullptr;
            pmw_idx_cache.clear();
        }
        num_csirs_info = 0;
    }
};

struct _coreset_t;
struct _coreset_group_t;
using coreset_t = _coreset_t;
using coreset_group_t = _coreset_group_t;
using coreset_list_t = std::array<cuphyPdcchCoresetDynPrm_t, CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL>;
#if ENABLE_CELL_GROUP
using coreset_group_list_t = std::array<cuphyPdcchCoresetDynPrm_t, CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL * MAX_CELLS_PER_CELL_GROUP>;
using dci_param_list = std::array<cuphyPdcchDciPrm_t, CUPHY_PDCCH_MAX_DCIS_PER_CORESET * MAX_CELLS_PER_CELL_GROUP>;
using dci_payload_t = std::array<uint8_t, CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES>;
using dci_payload_list_t = std::array<dci_payload_t, CUPHY_PDCCH_MAX_DCIS_PER_CORESET * MAX_CELLS_PER_CELL_GROUP>;
#else
using coreset_group_list_t = std::array<cuphyPdcchCoresetDynPrm_t, CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL>;
using dci_param_list = std::array<cuphyPdcchDciPrm_t, CUPHY_PDCCH_MAX_DCIS_PER_CORESET>;
using dci_payload_t = std::array<uint8_t, CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES>;
using dci_payload_list_t = std::array<dci_payload_t, CUPHY_PDCCH_MAX_DCIS_PER_CORESET>;
#endif

struct _coreset_t
{
    cuphyPdcchCoresetDynPrm_t info;
    dci_param_list dci_params;
    dci_payload_list_t dci_payload;
    explicit _coreset_t()
    {
        // dci_params.reserve(1);
        // dci_payload.reserve(1);
    }
};

struct _coreset_group_t {
    uint16_t nCoresets;
    coreset_group_list_t csets;
    uint32_t nDcis;
    dci_param_list dcis;
    dci_payload_list_t payloads;
    explicit _coreset_group_t():
        nCoresets(0),
        nDcis(0) {
    }
};

struct pdcch_params : public ch_params
{
    coreset_list_t cs_list;
    uint16_t nCoresets;
    uint16_t nDcis;
    explicit pdcch_params():
        nCoresets(0),
        nDcis(0){
        cell_index_list.reserve(1);
        phy_cell_index_list.reserve(1);
    }
};

struct pdcch_group_params: public ch_params {
    coreset_group_t csets_group;

    explicit pdcch_group_params() {
        //Note - all other channels manage these lists on per-cell basis
        // However PDCCH manages this per PDU.  It also assumes one to one mapping between PDU and coreset
        cell_index_list.reserve(MAX_PDCCH_PDUS_PER_CELL * MAX_CELLS_PER_CELL_GROUP);
        phy_cell_index_list.reserve(MAX_PDCCH_PDUS_PER_CELL * MAX_CELLS_PER_CELL_GROUP);
    }

    void reset() {
        csets_group.nCoresets = 0;
        csets_group.nDcis = 0;
        cell_index_list.clear();
        phy_cell_index_list.clear();
    }
};

//Need information from cuPHY
struct SSTxParams
{
    float    beta_pss;   /*!< scaling factor for PSS (primary synchronization signal) */
    float    beta_sss;   /*!< scaling factor for SSS (secondary synchronization signal), PBCH data and DMRS */
    uint16_t NID;        /*!< Physical cell id */
    uint16_t nHF;        /*!< Half frame index (0 or 1) */
    uint16_t Lmax;       /*!< Max number of ss blocks in pbch period (4,8,or 64) */
    uint16_t blockIndex; /*!< SS block index (0 - L_max) */
    uint16_t f0;         /*!< Index of initial ss subcarrier */
    uint16_t t0;         /*!< Index of initial ss ofdm symbol */
    uint16_t SFN;        /*!< frame index */
    uint16_t k_SSB;      /*!< SSB subcarrier offset */
    uint16_t nF;         /*!< number of subcarriers for one slot */
    uint16_t nT;         /*!< number of symbols for one slot */
};
struct pbch_params : public ch_params
{
    //SSTxParams ssb;
    SSTxParams ssb[3];
    uint8_t pdu_bitmap;
    uint32_t mib; //In a slot, it will not change so keeping single instance for now
    explicit pbch_params():
    ssb{0},
    pdu_bitmap(0),
    mib(0)
    {}
};
struct pbch_group_data
{
    //SSTxParams ssb;
    SSTxParams ssb;   //Creating seperate Pbch data for each SSB so no need for multi indices of ssb
    uint32_t mib; //In a slot, it will not change so keeping single instance for now
    uint16_t cell_idx;
    explicit pbch_group_data():
    ssb{0},
    mib(0),
    cell_idx{0xFFFF}
    {}
};

struct pbch_group_params : public ch_params
{
    std::array<cuphyPerCellSsbDynPrms_t, MAX_CELLS_PER_CELL_GROUP> pbch_dyn_cell_params;
    std::array<cuphyPerSsBlockDynPrms_t, MAX_CELLS_PER_CELL_GROUP * MAX_SSB_BLOCKS_PER_SLOT> pbch_dyn_block_params;
    std::array<uint32_t, MAX_CELLS_PER_CELL_GROUP * MAX_SSB_BLOCKS_PER_SLOT> pbch_dyn_mib_data;

    uint8_t nSsbBlocks;
    uint8_t ncells;

    explicit pbch_group_params():
        nSsbBlocks(0),
        ncells(0) {
        cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP * MAX_SSB_BLOCKS_PER_SLOT);
        phy_cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP * MAX_SSB_BLOCKS_PER_SLOT);
    }

    void reset() {
        nSsbBlocks = 0;
        ncells = 0;
        cell_index_list.clear();
        phy_cell_index_list.clear();
    }
};
using uci_params_t = std::array<cuphyPucchUciPrm_t, CUPHY_PUCCH_F3_MAX_UCI>;
using uci_params_fmt_t = std::array<uci_params_t, MAX_PUCCH_FORMAT>;
using ul_tti_handle_list_t = std::vector<uint32_t>;
struct pucch_params : public ch_params
{
    std::array<uint16_t, CUPHY_PUCCH_F3_MAX_UCI> rnti_list;
    uci_params_fmt_t params;
    cuphyPucchCellGrpDynPrm_t grp_dyn_pars;
    cuphyPucchCellDynPrm_t dyn_pars[CUPHY_PUCCH_F3_MAX_UCI];
    std::array <ul_tti_handle_list_t, MAX_PUCCH_FORMAT> scf_ul_tti_handle_list;
    explicit pucch_params()
    {
        grp_dyn_pars.nCells = 0;
        grp_dyn_pars.pCellPrms = dyn_pars;

        grp_dyn_pars.nF0Ucis = 0;
        grp_dyn_pars.pF0UciPrms = nullptr;

        grp_dyn_pars.nF1Ucis = 0;
        grp_dyn_pars.pF1UciPrms = nullptr;

        grp_dyn_pars.nF2Ucis = 0;
        grp_dyn_pars.pF2UciPrms = nullptr;

        grp_dyn_pars.nF3Ucis = 0;
        grp_dyn_pars.pF3UciPrms = nullptr;

        grp_dyn_pars.nF4Ucis = 0;
        grp_dyn_pars.pF4UciPrms = nullptr;

        scf_ul_tti_handle_list[0].reserve(CUPHY_PUCCH_F0_MAX_GRPS * CUPHY_PUCCH_F0_MAX_UCI_PER_GRP);
        scf_ul_tti_handle_list[1].reserve(CUPHY_PUCCH_F1_MAX_GRPS * CUPHY_PUCCH_F1_MAX_UCI_PER_GRP);
        scf_ul_tti_handle_list[2].reserve(CUPHY_PUCCH_F2_MAX_UCI);
        scf_ul_tti_handle_list[3].reserve(CUPHY_PUCCH_F3_MAX_UCI);
        cell_index_list.reserve(CUPHY_PUCCH_F3_MAX_UCI);
        phy_cell_index_list.reserve(CUPHY_PUCCH_F3_MAX_UCI);
    }

    void reset() {
        grp_dyn_pars.nCells = 0;
        grp_dyn_pars.nF0Ucis = 0;
        grp_dyn_pars.nF1Ucis = 0;
        grp_dyn_pars.nF2Ucis = 0;
        grp_dyn_pars.nF3Ucis = 0;
        grp_dyn_pars.nF4Ucis = 0;
        for(ul_tti_handle_list_t & handle_list : scf_ul_tti_handle_list)
        {
            handle_list.clear();
        }
        cell_index_list.clear();
        phy_cell_index_list.clear();
    }

};

struct prach_params : public ch_params
{
    std::array<::cuphyPrachOccaDynPrms_t, MAX_PRACH_OCCASIONS_PER_SLOT> rach;
    std::array<uint8_t, MAX_PRACH_OCCASIONS_PER_SLOT> freqIndex;
    std::array<uint8_t, MAX_PRACH_OCCASIONS_PER_SLOT> startSymbols;
    uint32_t nfft;
    uint8_t mu;

    uint8_t nOccasion;

    explicit prach_params():
	nOccasion(0)
 	{
	  	cell_index_list.reserve(MAX_PRACH_OCCASIONS_PER_SLOT);
	       	phy_cell_index_list.reserve(MAX_PRACH_OCCASIONS_PER_SLOT);
        }
    void reset()
    {
        cell_index_list.clear();
        phy_cell_index_list.clear();
        nOccasion = 0;
    }
};

// Place holder for slot_command
using  srs_rb_info_per_sym_t = std::vector<std::pair<uint16_t,uint16_t>>;


struct srs_ue_per_cell_info
{
    uint8_t cell_idx;
    uint8_t num_srs_ues;
};

struct srs_params : public ch_params
{
    cuphySrsCellGrpDynPrm_t cell_grp_info;
    cuphySrsCellDynPrm_t    cell_dyn_info[MAX_CELLS_PER_CELL_GROUP];
    cuphyUeSrsPrm_t         ue_info[MAX_SRS_UE_PER_TTI * MAX_CELLS_PER_CELL_GROUP];
    uint8_t*                srs_chest_buffer[MAX_SRS_UE_PER_TTI * MAX_CELLS_PER_CELL_GROUP];
    /* Keeps tracks of the number of valid max SRS.IND index in the slot per cell*/
    int                     num_srs_ind_indexes[MAX_CELLS_PER_CELL_GROUP];
    nv_ipc_msg_t            srs_indications[MAX_CELLS_PER_CELL_GROUP][MAX_SRS_IND_PER_SLOT];
    /* To keep track of number of SRS PDUs per SRS-ind per slot per cell */
    int                     num_srs_pdus_per_srs_ind[MAX_CELLS_PER_CELL_GROUP][MAX_SRS_IND_PER_SLOT];

    uint16_t dl_ul_bwp_max_prg[MAX_CELLS_PER_CELL_GROUP];
    uint16_t nGnbAnt;
    srs_ue_per_cell_info  srs_ue_per_cell[MAX_CELLS_PER_CELL_GROUP];

    std::vector<uint32_t> scf_ul_tti_handle_list;
    std::array < std::array <srs_rb_info_per_sym_t, OFDM_SYMBOLS_PER_SLOT>, MAX_CELLS_PER_CELL_GROUP> rb_info_per_sym;
    std::array < std::array <srs_rb_info_per_sym_t, OFDM_SYMBOLS_PER_SLOT>, MAX_CELLS_PER_CELL_GROUP> final_rb_info_per_sym;
    explicit srs_params():
    cell_grp_info(),
    cell_dyn_info{0},
    nGnbAnt(0),
    ue_info{0}
    {
        cell_grp_info.pCellPrms = cell_dyn_info;
        cell_grp_info.nSrsUes = 0;
        cell_grp_info.pUeSrsPrms = ue_info;

        cell_dyn_info[0].cellPrmStatIdx = 0;
        cell_dyn_info[0].cellPrmDynIdx = 0;

        cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        phy_cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        scf_ul_tti_handle_list.reserve( MAX_CELLS_MU_MIMO_ENABLE * MAX_SRS_PDU_PER_SLOT);
        for(uint8_t cellIdx=0; cellIdx < MAX_CELLS_PER_CELL_GROUP; cellIdx++)
        {
            dl_ul_bwp_max_prg[cellIdx] = 0;
            srs_ue_per_cell[cellIdx].cell_idx = 0;
            srs_ue_per_cell[cellIdx].num_srs_ues = 0;
            for(uint8_t slotIdx=0; slotIdx < OFDM_SYMBOLS_PER_SLOT; slotIdx++)
            {
                rb_info_per_sym[cellIdx][slotIdx].reserve(MAX_SRS_UE_PER_TTI * OFDM_SYMBOLS_PER_SLOT);
                final_rb_info_per_sym[cellIdx][slotIdx].reserve(MAX_SRS_UE_PER_TTI * OFDM_SYMBOLS_PER_SLOT);
            }
        }
    }
    void reset()
    {
        cell_grp_info.nCells = 0;
        cell_grp_info.nSrsUes = 0;

        cell_dyn_info[0].cellPrmStatIdx = 0;
        cell_dyn_info[0].cellPrmDynIdx = 0;

        nGnbAnt = 0;

        scf_ul_tti_handle_list.clear(); // will not purge memory
        for(uint8_t cellIdx=0; cellIdx < MAX_CELLS_PER_CELL_GROUP; cellIdx++)
        {
            dl_ul_bwp_max_prg[cellIdx] = 0;
            srs_ue_per_cell[cellIdx].cell_idx = 0;
            srs_ue_per_cell[cellIdx].num_srs_ues = 0;
            for(uint8_t slotIdx=0; slotIdx < OFDM_SYMBOLS_PER_SLOT; slotIdx++)
            {
                rb_info_per_sym[cellIdx][slotIdx].clear();
                final_rb_info_per_sym[cellIdx][slotIdx].clear();
            }
        }
        cell_index_list.clear();
        phy_cell_index_list.clear();
    }
};

struct bfw_params : public ch_params
{
    cuphyBfwDynPrm_t bfw_dyn_info;
    cuphyBfwDataIn_t dataIn;
    cuphyBfwDataOut_t dataOutH;
    cuphyBfwDataOut_t dataOutD;
    cuphySrsChEstBuffInfo_t chEstInfo[MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_PER_TTI * MAX_DL_UL_BF_UE_GROUPS];
    uint8_t* pBfwCoefH[MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_GROUPS];
    uint8_t* pBfwCoefD[MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_GROUPS];
    cuphyBfwUeGrpPrm_t ue_grp_info[MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_GROUPS];
    cuphyBfwLayerPrm_t pBfLayerPrm[MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_GROUPS * CUPHY_BFW_COEF_COMP_N_MAX_LAYERS_PER_USER_GRP];
    uint16_t prevUeGrpChEstInfoBufIdx;
    uint16_t prevUeGrpPerLayerInfoBufIdx;
    bfw_type bfw_cvi_type;
    uint16_t dl_ul_bwp_max_prg[MAX_CELLS_MU_MIMO_ENABLE];
    uint16_t nGnbAnt;
    uint16_t nue_grps_per_cell[MAX_CELLS_MU_MIMO_ENABLE];

    explicit bfw_params():
    bfw_dyn_info(),
    dataIn(),
    dataOutH(),
    dataOutD(),
    chEstInfo{0},
    pBfwCoefH{0},
    pBfwCoefD{0},
    ue_grp_info{0},
    pBfLayerPrm{0},
    prevUeGrpChEstInfoBufIdx(0),
    prevUeGrpPerLayerInfoBufIdx(0),
    nGnbAnt(0),
    bfw_cvi_type(BFW_NONE)
    {
        bfw_dyn_info.nUeGrps = 0;
        for(uint32_t i=0; i < (MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_GROUPS); i++)
            ue_grp_info[i].pBfLayerPrm = &pBfLayerPrm[i*CUPHY_BFW_COEF_COMP_N_MAX_LAYERS_PER_USER_GRP];
        bfw_dyn_info.pUeGrpPrms = ue_grp_info;
        dataIn.pChEstInfo = chEstInfo;
        dataOutH.pBfwCoef = pBfwCoefH;
        dataOutD.pBfwCoef = pBfwCoefD;
        for(uint8_t cellIdx=0; cellIdx < MAX_CELLS_MU_MIMO_ENABLE; cellIdx++)
        {
            nue_grps_per_cell[cellIdx] = 0;
            dl_ul_bwp_max_prg[cellIdx] = 0;
        }
    }
    void reset()
    {
        bfw_dyn_info.nUeGrps = 0;
        prevUeGrpChEstInfoBufIdx = 0;
        prevUeGrpPerLayerInfoBufIdx = 0;
        bfw_cvi_type = BFW_NONE;
        nGnbAnt = 0;
        for(uint32_t i=0; i < (MAX_CELLS_MU_MIMO_ENABLE * MAX_DL_UL_BF_UE_GROUPS); i++)
            ue_grp_info[i].nBfLayers = 0;
        for(uint8_t cellIdx=0; cellIdx < MAX_CELLS_MU_MIMO_ENABLE; cellIdx++)
        {
            nue_grps_per_cell[cellIdx] = 0;
            dl_ul_bwp_max_prg[cellIdx] = 0;
        }
    }
};

inline constexpr int CSIRS_SYMBOL_LOCATION_TABLE_LENGTH = 18;
struct CsirsTables
{
    CsirsSymbLocRow rowData[CSIRS_SYMBOL_LOCATION_TABLE_LENGTH];             /*!< resource mapping table */
    //int8_t          seqTable[MAX_CDM_TYPE][CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT][2][4]; /*!< wf/wt seq table layout: 2- Wf,Wt; 4 max(maxkprimelen, maxlprimelen) */
};

struct bsrs_info_t
{
    uint16_t mSRS;
    uint8_t nb;
};

struct SrsBwConfigTable
{
    bsrs_info_t bsrs_info[MAX_B_SRS_INDEX];
};

#include "slot_command_tables.hpp"

[[nodiscard]] inline __half reinterpret_uint16_as_half(uint16_t value) {
    static_assert(sizeof(__half) == sizeof(uint16_t), "Size mismatch");
    __half result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

[[nodiscard]] inline uint16_t reinterpret_half_as_uint16(__half value) {
    static_assert(sizeof(__half) == sizeof(uint16_t), "Size mismatch");
    uint16_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

struct __attribute__ ((__packed__)) tx_precoding_beamforming_t
{
    uint16_t num_prgs;
    uint16_t prg_size;
    uint8_t  dig_bf_interfaces;
    uint16_t pm_idx_and_beam_idx[MAX_NUM_PRGS + MAX_NUM_DIGBFI*MAX_NUM_PRGS];
};

struct csirs_params_: public ch_params
{
    //reMap represents the reMap of the entire bandwidth grid - hence it's is an array of num_dl_prbs * num_OFDM_symbols
    //where each array element represents 12 subcarriers in the lower 12 bits of uint16_t
    //need to move this array to heap
    uint16_t reMap[MAX_CELLS_PER_CELL_GROUP][MAX_N_PRBS_SUPPORTED*OFDM_SYMBOLS_PER_SLOT];
    //uint16_t * pReMap[MAX_CELLS_PER_CELL_GROUP];
    //14 lower bits of symbolMapArray represent the symbols in which CSI-RS is present across all the UEs
    uint16_t symbolMapArray[MAX_CELLS_PER_CELL_GROUP];
    cuphyCsirsRrcDynPrm_t csirsList[MAX_PDSCH_UE_PER_TTI ];
    uint16_t nCsirsRrcDynPrm;
    uint16_t numPcBf;
    tx_precoding_beamforming_t pcAndBf[MAX_PDSCH_UE_PER_TTI];
    bfw_coeff_mem_info_t *static_bfwCoeff_mem_info[MAX_PDSCH_UE_PER_TTI];
    uint16_t nCells;
    uint16_t lastCell;
    cuphyCsirsCellDynPrm_t cellInfo[MAX_CELLS_PER_CELL_GROUP];
    cuphyCsirsDynPrms_t csirsDynPrms;
    explicit csirs_params_():
    reMap{0},
    csirsList{0},
    nCells(0),
    lastCell(0),
    nCsirsRrcDynPrm(0),
    numPcBf(0),
    cellInfo{0},
    pcAndBf{0},
    static_bfwCoeff_mem_info{NULL},
    csirsDynPrms{0}
    {
        cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        phy_cell_index_list.reserve(MAX_CELLS_PER_CELL_GROUP);
        csirsDynPrms.pRrcDynPrm = csirsList;
        csirsDynPrms.pCellParam = cellInfo;
    }
    void reset()
    {
        std::memset(pcAndBf, 0, numPcBf*sizeof(tx_precoding_beamforming_t));
        for(int i=0 ; i < numPcBf; i++)
        {
            static_bfwCoeff_mem_info[i] = NULL;
        }
        numPcBf = 0;
        for(int i= 0 ; i < lastCell; i++)
        {
            std::memset(reMap[i], 0, MAX_N_PRBS_SUPPORTED*OFDM_SYMBOLS_PER_SLOT*sizeof(uint16_t));
        }

        std::memset(cellInfo, 0, nCells * sizeof(cuphyCsirsCellDynPrm_t));
        nCells = 0;
        lastCell = 0;
        cell_index_list.clear();
        phy_cell_index_list.clear();
        std::memset(csirsList, 0, nCsirsRrcDynPrm * sizeof(cuphyCsirsRrcDynPrm_t));
        nCsirsRrcDynPrm = 0;
        csirsDynPrms.nCells = 0;
        csirsDynPrms.nPrecodingMatrices = 0;
        std::memset(symbolMapArray, 0, sizeof(uint16_t) * MAX_CELLS_PER_CELL_GROUP);
    }
};
using csirs_params = csirs_params_;


struct uci_output_params
{
    // Format 0 or 1;
    uint8_t numHarq;
    uint8_t harqConfidenceLevel;
    uint8_t harq_pdu[1];
    uint8_t srIndication;
    uint8_t srConfidenceLevel;
};

template <channel_type chType> struct type_traits;
template <> struct type_traits <channel_type::PUSCH> {using type = pusch_params; };
template <> struct type_traits <channel_type::PDSCH> {using type = pdsch_params; };
template <> struct type_traits <channel_type::PDCCH_DL> {using type = pdcch_params; };
template <> struct type_traits <channel_type::PDCCH_UL> {using type = pdcch_params; };
template <> struct type_traits <channel_type::PBCH> {using type =  pbch_params; };
template <> struct type_traits <channel_type::PUCCH> {using type =  pucch_params; };
template <> struct type_traits <channel_type::PRACH> {using type = prach_params; };
template <> struct type_traits <channel_type::SRS> {using type = srs_params; };
template <> struct type_traits <channel_type::CSI_RS> {using type = csirs_params; };
template <> struct type_traits <channel_type::BFW> {using type = bfw_params; };

// ru_type::MULTI_SECT_MODE 23-2 max beams for 192 max UCI groups https://nvbugs/4158154 BFW uses 1096
inline constexpr size_t MAX_BEAMS = 64;

using beamid_array_t = std::array<uint16_t, MAX_BEAMS>;

enum fh_dir_t : uint8_t {
    FH_DIR_UL = 0x00,
    FH_DIR_DL = 0x01
};

struct __sc_packed prb_info_common_t_
{
    uint16_t startPrbc;
    uint16_t numPrbc;
    uint16_t reMask;
    uint8_t extType;
    uint16_t numApIndices;
    int32_t freqOffset;
    uint8_t numSymbols;
    fh_dir_t direction;
    uint8_t filterIndex;
    uint64_t portMask;
    uint32_t ap_index;
    int      active_eaxc_ids[MAX_DL_EAXCIDS];
    uint64_t pdschPortMask;
    uint8_t useAltPrb;
    uint8_t isStaticBfwEncoded;
    uint8_t isPdschSplitAcrossPrbInfo;
    explicit prb_info_common_t_(uint16_t startPrb, uint16_t numPrb,
        uint16_t re_mask=0xFFFF, uint8_t ext_type=0x00, [[maybe_unused]] uint8_t start_ap_index = 0x00,
        uint8_t num_ap_indices = 0x01, int32_t freq_offset=0x0000, uint8_t num_symbols = 1,
        fh_dir_t dir = fh_dir_t::FH_DIR_DL, uint8_t filter_index = 0, uint64_t port_mask = 0ULL, uint8_t useAltPrb = 0, uint8_t isStaticBfwEncoded = 0, uint8_t isPdschSplitAcrossPrbInfo = 0, uint64_t pdsch_port_mask = 0ULL):
            startPrbc(startPrb),
            numPrbc(numPrb),
            reMask(re_mask),
            extType(ext_type),
            numApIndices(num_ap_indices),
            freqOffset(freq_offset),
            numSymbols(num_symbols),
            direction(dir),
            filterIndex(filter_index),
            portMask(port_mask),
            ap_index(0),
            pdschPortMask(pdsch_port_mask),
            useAltPrb(useAltPrb),
            isStaticBfwEncoded(isStaticBfwEncoded),
            isPdschSplitAcrossPrbInfo(isPdschSplitAcrossPrbInfo)
    {}

    bool operator==(const prb_info_common_t_& other) const {
        return std::tie(startPrbc, numPrbc, reMask, extType, numApIndices, freqOffset, numSymbols, direction, filterIndex, portMask, pdschPortMask, useAltPrb, isStaticBfwEncoded, isPdschSplitAcrossPrbInfo)
            == std::tie(other.startPrbc, other.numPrbc, other.reMask, other.extType, other.numApIndices, other.freqOffset, other.numSymbols, other.direction, other.filterIndex, other.portMask, other.pdschPortMask, other.useAltPrb, other.isStaticBfwEncoded, other.isPdschSplitAcrossPrbInfo);
    }
};
using prb_info_common_t = prb_info_common_t_;

inline constexpr uint16_t MAX_SE5_MASKS = 2;
struct mod_comp_info_common_t {
    uint8_t    ef;
    uint8_t    extType;
    uint8_t    nSections;
    uint8_t    udIqWidth;
};

struct mod_comp_info_section_t {
    uint16_t    mcScaleReMask;
    uint16_t    mcScaleOffset;
    uint8_t     csf;
};

using se5_array_t = std::array<mod_comp_info_section_t, MAX_SE5_MASKS>;
struct mod_comp_info_t {
    mod_comp_info_common_t common;
    se5_array_t sections;
    float bwScaler;
    std::array<float, MAX_SE5_MASKS> modCompScalingValue;
};
struct bfw_coeff_buf_info_t
{
    uint16_t num_prgs;
    uint16_t prg_size;
    uint16_t nGnbAnt;
    uint8_t  dig_bf_interfaces;
    uint8_t *header;
    uint8_t *p_buf_bfwCoef_h;
    uint8_t *p_buf_bfwCoef_d;
};

struct static_bfw_coeff_buf_info_t
{
    uint16_t num_prgs;
    uint16_t prg_size;
    uint16_t nGnbAnt;
    uint8_t  dig_bf_interfaces;
};

struct srs_rb_info_t
{
    uint16_t srs_start_prbs;
    uint16_t num_srs_prbs;
};

struct cplane_sections_info_t
{
    uint16_t startPrbc[MAX_AP_PER_SLOT_SRS][MAX_CPLANE_SPLIT];
    uint16_t numPrbc[MAX_AP_PER_SLOT_SRS][MAX_CPLANE_SPLIT];
    uint16_t section_id[MAX_AP_PER_SLOT_SRS][MAX_CPLANE_SPLIT];
    uint16_t cplane_sections_count[MAX_AP_PER_SLOT_SRS];
     //!< Combined resource element mask for all sections in the antenna port group
     //!< This is used for CSI-RS compact signaling where multiple logica antenna ports
     //!< share the same flow id.
    uint16_t combined_reMask[MAX_AP_PER_SLOT_SRS];
};

struct prb_info_t_
{
    prb_info_common_t common;
    mod_comp_info_t comp_info;
    std::array<cplane_sections_info_t*, OFDM_SYMBOLS_PER_SLOT> cplane_sections_info; // C-Plane split due to MTU
    uint16_t cplane_sections_info_sym_map; // C-Plane split due to MTU
    //cplane_sections_info_t* cplane_sections_info; // C-Plane split due to MTU
    beamid_array_t beams_array; // NOTE unused in cuphydriver
    size_t beams_array_size;    // NOTE unused in cuphydriver
    beamid_array_t beams_array2; // Used for SE5
    size_t beams_array_size2;    //  Used for SE5
    bfw_coeff_buf_info_t bfwCoeff_buf_info;
    static_bfw_coeff_buf_info_t static_bfwCoeff_buf_info;

    explicit prb_info_t_(uint16_t startPrb, uint16_t numPrb):
    common(startPrb, numPrb)
    {
        beams_array_size = 0;
        beams_array_size2 = 0;
        cplane_sections_info_sym_map = 0;
        bfwCoeff_buf_info.num_prgs = 0;
        bfwCoeff_buf_info.prg_size = 0;
        bfwCoeff_buf_info.nGnbAnt = 0;
        bfwCoeff_buf_info.dig_bf_interfaces = 0;
        bfwCoeff_buf_info.header = nullptr;
        bfwCoeff_buf_info.p_buf_bfwCoef_h = nullptr;
        bfwCoeff_buf_info.p_buf_bfwCoef_d = nullptr;
        static_bfwCoeff_buf_info.num_prgs = 0;
        static_bfwCoeff_buf_info.prg_size = 0;
        static_bfwCoeff_buf_info.nGnbAnt = NUM_GNB_TX_RX_ANT_PORTS;
        static_bfwCoeff_buf_info.dig_bf_interfaces = 0;
    }

    explicit prb_info_t_():
    common(0, 0) {
        beams_array_size = 0;
        beams_array_size2 = 0;
        comp_info.common.ef = 0;
        comp_info.common.extType = 0;
        comp_info.common.nSections = 0;
        comp_info.common.udIqWidth = 0;
        cplane_sections_info_sym_map = 0;
        bfwCoeff_buf_info.num_prgs = 0;
        bfwCoeff_buf_info.prg_size = 0;
        bfwCoeff_buf_info.nGnbAnt = 0;
        bfwCoeff_buf_info.dig_bf_interfaces = 0;
        bfwCoeff_buf_info.header = nullptr;
        bfwCoeff_buf_info.p_buf_bfwCoef_h = nullptr;
        bfwCoeff_buf_info.p_buf_bfwCoef_d = nullptr;
        static_bfwCoeff_buf_info.num_prgs = 0;
        static_bfwCoeff_buf_info.prg_size = 0;
        static_bfwCoeff_buf_info.nGnbAnt = NUM_GNB_TX_RX_ANT_PORTS;
        static_bfwCoeff_buf_info.dig_bf_interfaces = 0;
    }
};
using prb_info_t = prb_info_t_;


#define MAX_PRB_INFO 512
#define MAX_CSIRS_PORTS_MAPPED_TO_SINGLE_FLOW 16
struct overlap_csirs_port_info_t {
    uint8_t num_ports;
    uint8_t num_overlap_ports;
    std::array<std::pair<uint16_t, uint8_t>, MAX_CSIRS_PORTS_MAPPED_TO_SINGLE_FLOW> reMask_ap_idx_pairs;
    overlap_csirs_port_info_t() {
        num_ports = 0;
        num_overlap_ports = 0;
        reMask_ap_idx_pairs.fill(std::make_pair(0, 0));
    }
};

using prb_info_list_t = prb_info_t*;
using alt_csirs_prb_info_list_t = prb_info_list_t;
using prb_info_idx_list_t = std::vector<std::size_t>;
using channel_info_list_t = std::array<prb_info_idx_list_t, channel_type::CHANNEL_MAX>;
using sym_info_list_t = std::array<channel_info_list_t, OFDM_SYMBOLS_PER_SLOT>;
using alt_csirs_prb_info_idx_list_t = std::array<prb_info_idx_list_t,OFDM_SYMBOLS_PER_SLOT>;
struct slot_info_
{
    sym_info_list_t symbols;
    alt_csirs_prb_info_idx_list_t alt_csirs_prb_info_idx_list;
    size_t prbs_size;
    size_t alt_csirs_prb_info_list_size;
    prb_info_t prbs[MAX_PRB_INFO];
    std::atomic<bool>  section_id_ready;
    prb_info_t alt_csirs_prb_info_list[MAX_PRB_INFO];
    overlap_csirs_port_info_t overlap_csirs_port_info[MAX_PRB_INFO];
    size_t start_symbol_dl;
    size_t start_symbol_ul;
    explicit slot_info_()
    {
        for(auto& channel_info_list : symbols)
        {
            for (auto& prb_info_idx_list : channel_info_list)
            {
                prb_info_idx_list.reserve(MAX_PDSCH_UE_PER_TTI);
            }
        }
        for(auto& prb_info_idx_list_t: alt_csirs_prb_info_idx_list)
        {
            prb_info_idx_list_t.reserve(MAX_PDSCH_UE_PER_TTI);
        }
        prbs_size = 0;
        alt_csirs_prb_info_list_size = 0;
        section_id_ready.store(false);
    }

    void reset()
    {
        for(int i = 0; i < OFDM_SYMBOLS_PER_SLOT; ++i)
        {
            for(int j = 0; j < channel_type::CHANNEL_MAX; ++j)
            {
                if(symbols[i][j].size())
                    symbols[i][j].clear();
            }
            alt_csirs_prb_info_idx_list[i].clear();

        }
        prbs_size = 0;
        alt_csirs_prb_info_list_size = 0;
        section_id_ready.store(false);

        // Clean overlap_csirs_port_info array
        for(std::size_t i = 0; i < MAX_PRB_INFO; ++i)
        {
            overlap_csirs_port_info[i].num_ports = 0;
            overlap_csirs_port_info[i].num_overlap_ports = 0;
        }
    }
};
using slot_info_t = slot_info_;

struct phy_slot_params
{
    std::unique_ptr<pusch_params> pusch;
    std::unique_ptr<pdsch_params> pdsch;
    std::unique_ptr<pdcch_params> pdcch_ul;
    std::unique_ptr<pdcch_params> pdcch_dl;
    std::unique_ptr<pbch_params> pbch;
    std::unique_ptr<pucch_params> pucch;
    std::unique_ptr<prach_params> prach;
    std::unique_ptr<csirs_params> csi_rs;
    std::unique_ptr<srs_params> srs;

    std::unique_ptr<slot_info_t> sym_prb_info;
    std::unique_ptr<bfw_params> bfw;


    phy_slot_params()
    {
        sym_prb_info = std::make_unique<slot_info_t>();
        reset();
    }

    void reset()
    {
        // pusch.reset();
        // pdsch.reset();
        // pdcch_dl.reset();
        // pdcch_ul.reset();
        // pbch.reset();
        // pucch.reset();
        // prach.reset();
        // csi_rs.reset();
        sym_prb_info->reset();
    }

    phy_slot_params(phy_slot_params&&) = default;
    phy_slot_params& operator=(phy_slot_params&&) = default;

    phy_slot_params(const phy_slot_params&) = delete;
    phy_slot_params& operator=(const phy_slot_params&) = delete;
};

/**
 Each channel is represented by a channel type and phy params required for the channel type
 to be executed .
 */
struct channel {
    channel_type type;
    phy_slot_params params;
    explicit channel(channel_type chType):
        type(chType),
        params()
    {

    }
};

using channel_vec = std::array<channel_type, channel_type::CHANNEL_MAX>;
/**
 A cell subcommand ia list of channels to be executed for the cell.
 cell --> cell indec of the cell

 */
// per cell command
struct cell_sub_command
{
    uint16_t cell; // TODO: this should be a UUID
    slot_info slot;
    std::array<uint32_t, channel_type::CHANNEL_MAX> channel_idx;
    channel_vec channels;
    uint8_t channel_array_size;
    phy_slot_params params;

    [[nodiscard]] pusch_params* get_pusch_params()
    {
        create_if(channel_type::PUSCH);
        return params.pusch.get();
    }

    [[nodiscard]] pdsch_params* get_pdsch_params()
    {
        create_if(channel_type::PDSCH);
        return params.pdsch.get();
    }

    [[nodiscard]] pdcch_params* get_pdcch_dl_params()
    {
        create_if(channel_type::PDCCH_DL);
        return params.pdcch_dl.get();
    }

    [[nodiscard]] pdcch_params* get_pdcch_ul_params()
    {
        create_if(channel_type::PDCCH_UL);
        return params.pdcch_ul.get();
    }

    [[nodiscard]] pbch_params* get_pbch_params()
    {
        create_if(channel_type::PBCH);
        return params.pbch.get();
    }

    [[nodiscard]] pucch_params* get_pucch_params()
    {
        create_if(channel_type::PUCCH);
        return params.pucch.get();
    }

    [[nodiscard]] prach_params* get_prach_params()
    {
        create_if(channel_type::PRACH);
        return params.prach.get();
    }

    [[nodiscard]] csirs_params* get_csi_rs()
    {
        create_if(channel_type::CSI_RS);
        return params.csi_rs.get();
    }

    [[nodiscard]] auto sym_prb_info()
    {
        return params.sym_prb_info.get();
    }

    [[nodiscard]] srs_params* get_srs_params()
    {
        create_if(channel_type::SRS);
        return params.srs.get();
    }

    [[nodiscard]] bfw_params* get_bfw_params()
    {
        create_if(channel_type::BFW);
        return params.bfw.get();
    }

    void create_if(const channel_type chType)
    {
        auto index = channel_idx[chType];
        if (index == channel_type::NONE)
        {
            channel_array_size++;
            channels[channel_array_size - 1] = chType;
            index = channel_array_size - 1;
            channel_idx[chType] = index;

            switch(chType)
            {
                case channel_type::PUSCH:
                {
                    params.pusch = std::make_unique<pusch_params>();
                }
                break;
                case channel_type::PDSCH:
                {
                    params.pdsch = std::make_unique<pdsch_params>();
                }
                break;
                case channel_type::PDCCH_DL:
                {
                    params.pdcch_dl = std::make_unique<pdcch_params>();
                }
                break;
                case channel_type::PDCCH_UL:
                {
                    params.pdcch_ul = std::make_unique<pdcch_params>();
                }
                break;
                case channel_type::PBCH:
                {
                    params.pbch = std::make_unique<pbch_params>();
                }
                break;
                case channel_type::PUCCH:
                {
                    params.pucch = std::make_unique<pucch_params>();
                }
                break;
                case channel_type::PRACH:
                {
                    params.prach = std::make_unique<prach_params>();
                }
                break;
                case channel_type::CSI_RS:
                {
                    params.csi_rs = std::make_unique<csirs_params>();
                }
                break;
                case channel_type::SRS:
                {
                    params.srs = std::make_unique<srs_params>();
                }
                break;
                case channel_type::BFW:
                {
                    params.bfw = std::make_unique<bfw_params>();
                }
                break;
                default:
                break;
            }
        }
    }

    cell_sub_command()
    {
        reset();
    }

    cell_sub_command(const cell_sub_command&) = delete;
    cell_sub_command& operator=(const cell_sub_command&) = delete;

    cell_sub_command(cell_sub_command&&) = default;
    cell_sub_command& operator=(cell_sub_command&&) = default;

    void reset()
    {
       for (auto& idx: channel_idx)
        {
            idx = channel_type::NONE;
        }
        channel_array_size = 0;
        params.reset();
        slot.type = SLOT_NONE;
    }
};

struct pdsch_fh_prepare_params
{
    // Holds all of the inputs to update_prc_fh_params_pdsch
    // and update_prc_fh_params_pdsch_csirs
    cuphyPdschUeGrpPrm_t* grp;
    cuphyPdschUePrm_t* ue;
    cell_sub_command* cell_cmd;
    bfw_coeff_mem_info_t *bfwCoeff_mem_info;
    tx_precoding_beamforming_t *pc_bf;
    // TODO: both the below varibles can be combined.
    // Used to access BFW bfwCoeff_mem_info buffer for each UEG
    uint32_t ue_grp_bfw_index_per_cell;
    // Used for PDSCH
    uint16_t ue_grp_index;
    uint16_t num_dl_prb;
    uint16_t cell_index;
    bool is_new_grp;
    bool bf_enabled;
    bool pm_enabled;
    bool mmimo_enabled;
    bool csirs_compact_mode;
};

struct csirs_fh_prepare_params
{
    // Holds all of the inputs to update_fh_params_csirs_remap
    cell_sub_command* cell_cmd;
    uint32_t cell_idx;
    // same cell index of cuphy cellPrmDynIdx
    int32_t cuphy_params_cell_idx;
    uint16_t num_dl_prb;
    bool bf_enabled;
    bool mmimo_enabled;
};


struct fh_prepare_callback_params
{
    //Linear array to store pdsch_fh_params across all Cells
    // Size has been limited to MAX_PDSCH_UE_PER_TTI to avoid performance issues in FH_CALLBACK
    std::array<pdsch_fh_prepare_params, MAX_ALLOWED_PDSCH_PDUS_PER_SLOT> pdsch_fh_params = {};
    //Linear array to store TX Precoding beamforming info across all Cells
    std::array<tx_precoding_beamforming_t, MAX_ALLOWED_PDSCH_PDUS_PER_SLOT> pc_bf_arr = {};
    //Maintains a record of the start index of pdsch_fh_params per Cell
    std::array<uint16_t, DL_MAX_CELLS_PER_SLOT> start_index_pdsch_fh_params = {};
    //Number of PDSCH PDU's per Cell
    std::array<uint8_t, DL_MAX_CELLS_PER_SLOT> num_pdsch_fh_params = {};
    //Total Number of PDSCH PDUs across all the Cells
    uint32_t total_num_pdsch_pdus = {};

    std::array<csirs_fh_prepare_params, DL_MAX_CELLS_PER_SLOT> csirs_fh_params = {};
    std::array<uint8_t, DL_MAX_CELLS_PER_SLOT> is_csirs_cell = {};

    uint32_t num_csirs_cell = {};
    void reset()
    {
        for (int idx = 0; idx < total_num_pdsch_pdus; ++idx) {
            auto &params = pc_bf_arr.at(idx);
            params.num_prgs = 0;
            params.dig_bf_interfaces = 0;
        }
        for (int cell = 0; cell < DL_MAX_CELLS_PER_SLOT; ++cell) {
            is_csirs_cell[cell] = 0;
            start_index_pdsch_fh_params[cell] = 0;
            num_pdsch_fh_params[cell] = 0;
        }
        total_num_pdsch_pdus = 0;
        num_csirs_cell = 0;
    }
};

struct pm_group {

    struct CacheEntry {
        uint32_t pmwIdx;
        uint32_t nIndex;
    };
    std::array<cuphyPmWOneLayer_t,  MAX_SSB_BLOCKS_PER_SLOT * MAX_CELLS_PER_CELL_GROUP> ssb_list;
    std::array<CacheEntry,  MAX_SSB_BLOCKS_PER_SLOT * MAX_CELLS_PER_CELL_GROUP> ssb_pmw_idx_cache;

    std::array<cuphyPmWOneLayer_t,  MAX_PDSCH_UE_PER_TTI * MAX_CELLS_PER_CELL_GROUP> pdcch_list;
    std::array<CacheEntry,  MAX_PDSCH_UE_PER_TTI * MAX_CELLS_PER_CELL_GROUP> pdcch_pmw_idx_cache;

    std::array<cuphyPmWOneLayer_t, MAX_CSIRS_OCCASIONS_PER_SLOT  * MAX_CELLS_PER_CELL_GROUP> csirs_list;
    std::array<CacheEntry, MAX_CSIRS_OCCASIONS_PER_SLOT * MAX_CELLS_PER_CELL_GROUP> csirs_pmw_idx_cache;

    uint16_t nPmPdcch;
    uint16_t nPmPbch;
    uint16_t nPmCsirs;
    uint16_t nCacheEntries;
    bool precoding_enabled;

    explicit pm_group(bool enablePc, [[maybe_unused]] uint16_t max_cells = 1):
    nPmPdcch(0),
    nPmPbch(0),
    nPmCsirs(0),
    nCacheEntries(0),
    precoding_enabled(enablePc) {
        ssb_pmw_idx_cache.fill({.pmwIdx = UINT32_MAX, .nIndex = UINT32_MAX});
        pdcch_pmw_idx_cache.fill({.pmwIdx = UINT32_MAX, .nIndex = UINT32_MAX});
        csirs_pmw_idx_cache.fill({.pmwIdx = UINT32_MAX, .nIndex = UINT32_MAX});
    }


    void reset() {
        if (precoding_enabled) {
            nPmPdcch = 0;
            nPmPbch = 0;
            nPmCsirs = 0;
            nCacheEntries = 0;
            ssb_pmw_idx_cache.fill({.pmwIdx = UINT32_MAX, .nIndex = UINT32_MAX});
            pdcch_pmw_idx_cache.fill({.pmwIdx = UINT32_MAX, .nIndex = UINT32_MAX});
            csirs_pmw_idx_cache.fill({.pmwIdx = UINT32_MAX, .nIndex = UINT32_MAX});

        }
    }

};

struct cell_group_command
{
    std::unique_ptr<pusch_params> pusch;
    std::unique_ptr<pdsch_params> pdsch;
    std::unique_ptr<pucch_params> pucch;
    std::unique_ptr<csirs_params> csirs;
    std::unique_ptr<pdcch_group_params> pdcch;
    std::unique_ptr<prach_params> prach;
    std::unique_ptr<pbch_group_params> pbch;
    std::unique_ptr<pm_group> pmWeights;
    std::unique_ptr<srs_params> srs;
    std::unique_ptr<bfw_params> bfw;
    slot_info slot;
    std::array<uint32_t, channel_type::CHANNEL_MAX> channel_idx;
    channel_vec channels;
    uint8_t channel_array_size;
    fh_prepare_callback_params fh_params;
    cell_group_command():
    pusch(std::make_unique<pusch_params>()),
    pdsch(std::make_unique<pdsch_params>()),
    pdcch(std::make_unique<pdcch_group_params>()),
    pbch(std::make_unique<pbch_group_params>()),
    pucch(std::make_unique<pucch_params>()),
    prach(std::make_unique<prach_params>()),
    csirs(std::make_unique<csirs_params>()),
    srs(std::make_unique<srs_params>()),
    bfw(std::make_unique<bfw_params>()),
    pmWeights(nullptr),
    channel_array_size(0)
    {
        reset();
    }

    cell_group_command(const cell_group_command&) = delete;
    cell_group_command& operator=(const cell_group_command&) = delete;

    cell_group_command(cell_group_command&&) = default;
    cell_group_command& operator=(cell_group_command&&) = default;

    [[nodiscard]] pusch_params* get_pusch_params()
    {
        create_if(channel_type::PUSCH);
        return pusch.get();
    }

    [[nodiscard]] pdsch_params* get_pdsch_params()
    {
        create_if(channel_type::PDSCH);
        return pdsch.get();
    }

    [[nodiscard]] pucch_params* get_pucch_params()
    {
        create_if(channel_type::PUCCH);
        return pucch.get();
    }

    [[nodiscard]] csirs_params* get_csirs_params()
    {
        create_if(channel_type::CSI_RS);
        return csirs.get();
    }

    [[nodiscard]] pdcch_group_params * get_pdcch_params() {
        create_if(channel_type::PDCCH_DL);
        return pdcch.get();
    }

    [[nodiscard]] prach_params* get_prach_params() {
        create_if(channel_type::PRACH);
        return prach.get();
    }

    [[nodiscard]] pbch_group_params * get_pbch_params() {
        create_if(channel_type::PBCH);
        return pbch.get();
    }

    [[nodiscard]] srs_params* get_srs_params()
    {
        create_if(channel_type::SRS);
        return srs.get();
    }

    [[nodiscard]] bfw_params* get_bfw_params()
    {
        create_if(channel_type::BFW);
        return bfw.get();
    }

    void create_if(const channel_type chType)
    {
        auto index = channel_idx[chType];
        if (index == channel_type::NONE)
        {
            channel_array_size++;
            channels[channel_array_size - 1] = chType;
            index = channel_array_size - 1;
            channel_idx[chType] = index;

            // All channel params are eagerly allocated in constructor; create_if only registers the channel, no lazy allocation needed.
        }
    }

    void reset() {
        channel_array_size = 0;
        channels.fill(channel_type::NONE);
        channel_idx[channel_type::PUSCH] = channel_type::NONE;
        channel_idx[channel_type::PDSCH] = channel_type::NONE;
        channel_idx[channel_type::PUCCH] = channel_type::NONE;
        channel_idx[channel_type::CSI_RS] = channel_type::NONE;
        channel_idx[channel_type::PDCCH_DL] = channel_type::NONE;
        channel_idx[channel_type::PDCCH_UL] = channel_type::NONE;
        channel_idx[channel_type::PRACH] = channel_type::NONE;
        channel_idx[channel_type::PBCH] = channel_type::NONE;
        channel_idx[channel_type::SRS] = channel_type::NONE;
        channel_idx[channel_type::BFW] = channel_type::NONE;
        pdsch->reset();
        pucch->reset();
        pdcch->reset();
        prach->reset();
        csirs->reset();
        pbch->reset();
        pusch->reset();
        srs->reset();
        bfw->reset();
        fh_params.reset();
        slot.type = SLOT_NONE;
        if (pmWeights) {
            pmWeights->reset();
        }
    }

    void create_pm_group(bool enablePc, uint16_t max_cells_pm) {
        if (!pmWeights && enablePc) {
            pmWeights = std::make_unique<pm_group>(enablePc, max_cells_pm);
        }
    }

    [[nodiscard]] pm_group* get_pm_group() {
        return pmWeights.get();
    }
};

using cell_commands = std::vector<cell_sub_command>;
/**
 A slot command consists of slot (3GPP , SFN, SLOT) and list of
 cell sub commands . Each sub command again contains a list of
 channels which contain the phy parameter to run a single pipeline.
 This structure needs to be revisted when cuPHY publishes new API
 for each pipeline
 */
struct slot_command
{
    // Array of per cell commands
    cell_commands cells;
    cell_group_command cell_groups;
    nanoseconds tick_original;
};

struct ul_output_msg_buffer
{
    // need buffer type
    void* cookie; //used internally
    nv_ipc_msg_t ipc_msg;
    // CPU or GPU Buffer
    uint8_t* data_buf;
    // Intially
    std::size_t total_bytes;
    /// number of TB
    uint32_t numTB;
    // array of offset
    std::vector<uint32_t> tb_offset;
};

using ul_alloc_buffer = void (*)(void* context, ul_output_msg_buffer&, const pusch_params&);
// need more info for RNTIS for which TB are generated and TB offsets if multiple TB are generated
using ul_slot_callback = void (*)(void* context,
                                   uint32_t nCrc,
                                   ul_output_msg_buffer&,
                                   const slot_indication&,
                                   const pusch_params&,
                                   ::cuphyPuschDataOut_t const*,
                                   ::cuphyPuschStatPrms_t const*);

/*using ul_prach_callback = std::function<void ( slot_command_api::slot_indication&,
                                               const prach_params& params,
                                            const uint32_t* num_detectedPrmb,
                                            const void** prmbIndex_estimates,
                                            const void** prmbDelay_estimates,
                                            const void** prmbPower_estimates,
                                            const void* ant_rssi,
                                            const void* rssi)>; */

//Change
using ul_prach_callback = void (*)(void* context,
                                   slot_command_api::slot_indication&,
                                   const prach_params& params,
                                   const uint32_t* num_detectedPrmb,
                                   const void* prmbIndex_estimates,
                                   const void* prmbDelay_estimates,
                                   const void* prmbPower_estimates,
                                   const void* ant_rssi,
                                   const void* rssi,
                                   const void* interference);

using ul_uci_callback = void (*)(void* context,
                                 slot_indication&,
                                 const pucch_params&,
                                 const uci_output_params&);

using ul_uci_callback2 = void (*)(void* context,
                                  slot_indication&,
                                  const pucch_params&,
                                  const cuphyPucchDataOut_t&);
using ul_uci_early_callback = void (*)(void* context,
                                       const slot_indication&,
                                       const pusch_params&,
                                       ::cuphyPuschDataOut_t const*,
                                       ::cuphyPuschStatPrms_t const*,
                                       nanoseconds t0_original);
using ul_srs_callback = void (*)(void* context,
                                 ul_output_msg_buffer&,
                                 const slot_indication&,
                                 const srs_params&,
                                 ::cuphySrsDataOut_t const*,
                                 ::cuphySrsStatPrms_t const*,
                                 const std::array<bool,UL_MAX_CELLS_PER_SLOT>&);
using ul_free_harq_buffer = void (*)(void* context,
                                     const ReleasedHarqBufferInfo& freed_harq_buffer_data,
                                     slot_command_api::pusch_params* params,
                                     uint16_t sfn,
                                     uint16_t slot);

using fh_done_using_bfw_coeff_buffer = void (*)(void* context, uint8_t*);

using fh_done_using_static_bfw_coeff_buffer = void (*)(void* context, uint8_t*);

using ul_tx_error = void (*)(void* context,
                             const slot_command_api::slot_indication&,
                             uint16_t,
                             uint16_t,
                             std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>&,
                             uint8_t,
                             bool);

struct ul_slot_callbacks
{
    /// Unused
    ul_alloc_buffer alloc_fn;
    void* alloc_fn_context{};  //!< Context pointer for alloc_fn
    ul_slot_callback callback_fn;
    void* callback_fn_context{};  //!< Context pointer for callback_fn
    ul_prach_callback prach_cb_fn;
    void* prach_cb_context{};  //!< Context pointer for prach_cb_fn
    /// Used only for gNB - Harq indication
    // No support for SR and CSI
    ul_uci_callback uci_cb_fn;
    void* uci_cb_context{};  //!< Context pointer for uci_cb_fn
    ul_uci_callback2 uci_cb_fn2;
    void* uci_cb_fn2_context{};  //!< Context pointer for uci_cb_fn2
    ul_uci_early_callback callback_fn_early_uci;
    void* callback_fn_early_uci_context{};  //!< Context pointer for callback_fn_early_uci
    ul_srs_callback srs_cb_fn;
    void* srs_cb_context{};  //!< Context pointer for srs_cb_fn
    fh_done_using_bfw_coeff_buffer fh_bfw_coeff_usage_done_fn;
    void* fh_bfw_coeff_usage_done_fn_context{};  //!< Context pointer for fh_bfw_coeff_usage_done_fn
    ul_tx_error ul_tx_error_fn;
    void* ul_tx_error_fn_context{};  //!< Context pointer for ul_tx_error_fn
    ul_free_harq_buffer ul_free_harq_buffer_fn;
    void* ul_free_harq_buffer_fn_context{};  //!< Context pointer for ul_free_harq_buffer_fn
};

// may need more changes
using dl_slot_callback = void (*)(void* context, const slot_command_api::pdsch_params*);


using fh_prepare_callback = void (*)(void* context,
                                     slot_command_api::cell_group_command*,
                                     uint8_t /* cell_id */);

using dl_tx_error = void (*)(void* context,
                             const slot_command_api::slot_indication&,
                             uint16_t,
                             uint16_t,
                             std::array<uint32_t,DL_MAX_CELLS_PER_SLOT>&,
                             uint8_t);

using l1_exit_error = std::function<void(uint16_t,uint16_t,std::array<uint32_t,DL_MAX_CELLS_PER_SLOT>&,uint8_t)>;

struct dl_slot_callbacks
{
    dl_slot_callback callback_fn;
    void* callback_fn_context{};  //!< Context pointer for callback_fn

    fh_prepare_callback fh_prepare_callback_fn;
    void* fh_prepare_callback_fn_context{};  //!< Context pointer for fh_prepare_callback_fn
    fh_prepare_callback_params fh_prepare_cb_prms;
    fh_done_using_bfw_coeff_buffer fh_bfw_coeff_usage_done_fn;
    void* fh_bfw_coeff_usage_done_fn_context{};  //!< Context pointer for fh_bfw_coeff_usage_done_fn
    dl_tx_error dl_tx_error_fn;
    void* dl_tx_error_fn_context{};  //!< Context pointer for dl_tx_error_fn
    l1_exit_error l1_exit_error_fn;
};

struct callbacks {
    ul_slot_callbacks ul_cb;
    dl_slot_callbacks dl_cb;
};

}
#endif /* SLOT_COMMAND_API_HPP_INCLUDED_ */
