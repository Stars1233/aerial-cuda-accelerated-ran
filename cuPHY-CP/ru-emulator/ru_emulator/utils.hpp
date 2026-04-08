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

#ifndef UTILS_H__
#define UTILS_H__

#include <stdexcept>
#include <memory>
#include <getopt.h>
#include <signal.h>
#include <sstream>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <atomic>

#include <signal.h>
#include <sched.h>
#include <fstream>
#include <array>
#include <memory>
#include <unordered_map>
#include <type_traits>
#include <vector>
#include <map>
#include <chrono>
#include <algorithm>
#include <mutex>
#include "aerial-fh-driver/fh_mutex.hpp"

#include <bitset>
#include <unordered_set>
#include <vector_types.h>
#include <cuda_fp16.h>
#include <math.h>

#include "yaml.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/packet_stats.hpp"
#include "aerial-fh-driver/api.hpp"
#include "oran_structs.hpp"
#include "nvlog.hpp"
#include "cuphy.h"

// DPDK and threading configuration
#define PORT_ID 0                   //!< Default DPDK port identifier
#define MAX_RU_THREADS 128          //!< Maximum number of RU emulator worker threads

// Time conversion constants
#define US_X_MS 1000                //!< Microseconds per millisecond
#define US_X_S  1000000             //!< Microseconds per second
#define NS_X_US 1000                //!< Nanoseconds per microsecond
#define NS_X_S 1000000000           //!< Nanoseconds per second

// Data size conversion constants
#define BIT_X_BYTE 8                //!< Bits per byte
#define B_X_KB 1000UL               //!< Bytes per kilobyte
#define KB_X_MB 1000UL              //!< Kilobytes per megabyte
#define B_X_MB (B_X_KB * KB_X_MB)   //!< Bytes per megabyte

// Checksum configuration
#define ADLER32_MOD 65521           //!< Modulus for Adler-32 checksum algorithm

// Results table configuration
#define RESULTS_TABLE_START_INDEX   33  //!< Starting index for results table

// Packet processing configuration
#define MAX_PACKET_PER_RX_BURST     512   //!< Maximum packets to receive in a single burst
#define PACKET_RX_BUFFER_COUNT      1000  //!< Number of packet receive buffers
#define MAX_RUE_DL_CORE_COUNT       20    //!< Maximum number of DL processing cores in RU emulator
#define MAX_SRS_CELLS_PER_CORE      2     //!< Maximum number of SRS cells per processing core
#define MIN_UL_CORES_PER_CELL_MMIMO 3     //!< Minimum number of UL cores per cell for massive MIMO

// Ring buffer and packet configuration
#define RE_RING_ELEMS               8192  //!< Number of elements in RU emulator ring buffer
#define DEFAULT_MAX_PKT_SIZE        4092  //!< Default maximum packet size in bytes
#define CPLANE_DEQUEUE_BURST_SIZE   128   //!< C-plane dequeue burst size for standard mode
#define CPLANE_DEQUEUE_BURST_SIZE_MMIMO   8  //!< C-plane dequeue burst size for massive MIMO

// C-plane and U-plane limits
#define MAX_NUM_SECTIONS_PER_C_PLANE 64      //!< Maximum number of sections per C-plane message
#define MAX_NUM_SECTION_EXTENSIONS 16        //!< Maximum number of section extensions
#define MAX_NUM_PACKETS_PER_C_PLANE 3822     //!< Maximum number of packets per C-plane message
#define MAX_NUM_PACKETS_PER_C_PLANE_PER_SYM 273  //!< Maximum packets per C-plane message per symbol

// OFDM and resource block configuration
#ifdef ENABLE_32DL
#define MAX_FLOWS_PER_DL_CORE       32  //!< Maximum number of flows per DL processing core
#else
#define MAX_FLOWS_PER_DL_CORE       16  //!< Maximum number of flows per DL processing core
#endif
#define OFDM_SYMBOLS_PER_SLOT       14    //!< Number of OFDM symbols per slot
#define MAX_DL_LAYERS_PER_TB        4     //!< Maximum number of DL layers per transport block
#define SUBCARRIERS_PER_PRB         12    //!< Number of subcarriers per physical resource block
#define IQ_SAMPLE_ELEM_SIZE         4     //!< Size of IQ sample element in bytes
#define MAX_NUM_ANTENNAS            16    //!< Maximum number of antenna ports

// Channel configuration
#define MAX_UL_CHANNELS             5     //!< Maximum number of uplink channel types

// Launch pattern configuration
#define SLOT_3GPP                   (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)  //!< Total slots per 3GPP frame
#define MAX_LAUNCH_PATTERN_CYCLES   4     //!< Maximum number of launch pattern cycles
#define MAX_LAUNCH_PATTERN_SLOTS    (SLOT_3GPP * MAX_LAUNCH_PATTERN_CYCLES)   //!< Total launch pattern slots

// Processing and validation configuration
#define MAX_DL_CORES_PER_CELL       4     //!< Maximum number of DL cores per cell
#define STATS_MAX_BINS              1000  //!< Maximum number of statistics bins
#define PBCH_MAX_SUBCARRIERS        240   //!< Maximum number of PBCH subcarriers
#define PRACH_NUM_SYMBOLS           12    //!< Number of PRACH symbols
#define MAX_NUM_PRBS_PER_SYMBOL     273   //!< Maximum number of PRBs per symbol
#define BETA_DL_NOT_SET             0     //!< Sentinel value indicating beta_dl not configured

// Transmit queue configuration
#define NUM_SYMBOL_SETS_OF_TXQ 3          //!< Number of symbol sets in TX queue
#define RU_TXQ_COUNT (NUM_SYMBOL_SETS_OF_TXQ * ORAN_ALL_SYMBOLS)  //!< Total RU TX queue count
#define SRS_MAX_SYMBOLS_PER_SLOT    2     //!< Maximum SRS symbols per slot
#define SRS_MAX_EAXCIDS_PER_SYMBOL  64    //!< Maximum SRS eAxC IDs per symbol
#define SRS_MAX_EAXCIDS_PER_WINDOW  4     //!< Maximum SRS eAxC IDs per window
static constexpr uint32_t SRS_FIRST_SYMBOL = 14 - SRS_MAX_SYMBOLS_PER_SLOT;  //!< First symbol index for SRS transmission
static constexpr uint32_t SRS_TXQS_PER_SYMBOL = (SRS_MAX_EAXCIDS_PER_SYMBOL / SRS_MAX_EAXCIDS_PER_WINDOW);  //!< SRS TX queues per symbol
static constexpr uint32_t SRS_TXQS = (SRS_TXQS_PER_SYMBOL * SRS_MAX_SYMBOLS_PER_SLOT);  //!< Total SRS TX queues

// Frame caching configuration
#define ORAN_MAX_CACHED_FRAMES 32   //!< Maximum number of cached ORAN frames

// Antenna port configuration
static constexpr uint32_t MAX_AP_PER_SLOT = 32;  //!< Maximum antenna ports per slot

// Logging macros for RU Emulator
#define re_dbg(FMT, ARGS...)  NVLOGD_FMT(NVLOG_TAG_BASE_RU_EMULATOR, FMT, ##ARGS)  //!< Debug level logging
#define re_info(FMT, ARGS...) NVLOGI_FMT(NVLOG_TAG_BASE_RU_EMULATOR, FMT, ##ARGS)  //!< Info level logging
#define re_warn(FMT, ARGS...) NVLOGW_FMT(NVLOG_TAG_BASE_RU_EMULATOR, FMT, ##ARGS)  //!< Warning level logging
#define re_cons(FMT, ARGS...) NVLOGC_FMT(NVLOG_TAG_BASE_RU_EMULATOR, FMT, ##ARGS)  //!< Console level logging
#define re_err(CODE, FMT, ARGS...)  NVLOGE_FMT(NVLOG_TAG_BASE_RU_EMULATOR, CODE, FMT, ##ARGS)  //!< Error level logging
#define re_verb(FMT, ARGS...)  NVLOGV_FMT(NVLOG_TAG_BASE_RU_EMULATOR, FMT, ##ARGS)  //!< Verbose level logging

/**
 * Set thread name for debugging (max 15 characters per pthread limit)
 */
#define SET_THREAD_NAME(name) \
    { \
        char *str = strdup(name); \
        if (strlen(str) > 15) \
            str[15] = '\0'; \
        pthread_setname_np(pthread_self(), str); \
        free(str); \
    }

#define getName(var)  #var  //!< Convert variable name to string literal

// Branch prediction hints for performance optimization
#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)    //!< Hint to compiler that condition is likely true
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)  //!< Hint to compiler that condition is unlikely true
#endif

/**
 * RU Emulator error codes
 */
enum RE_ERRORS {
        RE_OK = 0,      //!< Success
        RE_ERR = -1,    //!< General error
        RE_EINVAL = -2, //!< Invalid argument
        RE_STOP = -3    //!< Stop signal
};

/**
 * Logging message level configuration
 */
enum RE_MSG_level {
    RE_MSG_VERBOSE = 1,  //!< Verbose messages
    RE_MSG_DEBUG,        //!< Debug messages
    RE_MSG_INFO,         //!< Informational messages
    RE_MSG_WARN,         //!< Warning messages
    RE_MSG_CONSOLE,      //!< Console output messages
    RE_MSG_ERROR         //!< Error messages
};

/**
 * Timer granularity level for performance profiling
 */
enum RE_TIMER_level {
    RE_TIMER_NONE = 0,   //!< No timing
    RE_TIMER_SLOT,       //!< Slot-level timing
    RE_TIMER_SYMBOL      //!< Symbol-level timing
};

/**
 * Feature enable/disable flag
 */
enum RE_ENABLE_level {
    RE_DISABLED = 0,  //!< Feature disabled
    RE_ENABLED = 1    //!< Feature enabled
};

/**
 * Downlink channel types for DL processing
 */
enum class dl_channel {
    NONE = 0,    //!< No channel
    PDSCH,       //!< Physical Downlink Shared Channel
    PBCH,        //!< Physical Broadcast Channel
    PDCCH_DL,    //!< Physical Downlink Control Channel (DL)
    PDCCH_UL,    //!< Physical Downlink Control Channel (UL grant)
    CSI_RS,      //!< Channel State Information Reference Signal
    BFW_DL,      //!< Beamforming Weights (Downlink)
    BFW_UL       //!< Beamforming Weights (Uplink)
};

/**
 * Uplink channel types for UL processing
 */
enum class ul_channel {
    NONE = 0,  //!< No channel
    PUSCH,     //!< Physical Uplink Shared Channel
    PRACH,     //!< Physical Random Access Channel
    PUCCH,     //!< Physical Uplink Control Channel
    SRS        //!< Sounding Reference Signal
};

/**
 * Test vector channel types for 5G NR simulator
 */
namespace nrsim_tv_type{
    enum nrsim_tv_type {
        SSB     = 1,  //!< Synchronization Signal Block
        PDCCH   = 2,  //!< Physical Downlink Control Channel
        PDSCH   = 3,  //!< Physical Downlink Shared Channel
        CSI_RS  = 4,  //!< Channel State Information Reference Signal
        PRACH   = 6,  //!< Physical Random Access Channel
        PUCCH   = 7,  //!< Physical Uplink Control Channel
        PUSCH   = 8,  //!< Physical Uplink Shared Channel
        SRS     = 9,  //!< Sounding Reference Signal
        BFW     = 10, //!< Beamforming Weights
    };
}

/**
 * DPDK network interface information
 */
struct dpdk_info {
    int                      socket_id;        //!< NUMA socket ID
    struct oran_ether_addr   src_eth_addr;     //!< Source Ethernet MAC address
    struct oran_ether_addr   peer_eth_addr[MAX_CELLS_PER_SLOT];  //!< Peer Ethernet MAC addresses per cell
    int                      num_peer_addr;    //!< Number of peer addresses
    int                      vlan;             //!< VLAN ID
    //Mempool
};

/**
 * Complex number representation with 16-bit integer components
 */
typedef struct
{
    int16_t re;  //!< Real component
    int16_t im;  //!< Imaginary component
} complex_int16_t;

//!< Packet type name strings for debugging
static constexpr char packet_type_array[3][20] = {"UL_C_PLANE","DL_C_PLANE", "DL_U_PLANE"};
//!< Section type name strings for debugging
static constexpr char section_type_array[4][20] = {"UL type1","UL type3", "DL type1", "AGGR"};

using dbt_data_t = std::vector<complex_int16_t>;  //!< Dynamic beamforming table data buffer type

/**
 * Dynamic beamforming table metadata
 */
struct dbt_md_
{
    bool bf_stat_dyn_enabled;       //!< Enable static/dynamic beamforming
    uint16_t num_static_beamIdx;    //!< Number of static beam indices
    uint16_t num_TRX_beamforming;   //!< Number of baseband TRX ports
    dbt_data_t dbt_data_buf;        //!< Beamforming data buffer
    std::unordered_map<uint16_t, bool> static_beamIdx_seen;  //!< Tracking map for seen static beam indices
};
using dbt_md_t = dbt_md_;  //!< Alias for dynamic beamforming table metadata

/**
 * Per-cell configuration parameters
 */
struct cell_config
{
    int pusch_txq;              //!< TX queue index for PUSCH
    int prach_txq;              //!< TX queue index for PRACH
    std::string name;           //!< Cell identifier name
    struct oran_ether_addr eth_addr;  //!< Ethernet MAC address for this cell
    std::vector<int> eAxC_UL;   //!< Uplink eAxC (antenna-carrier) ID list
    std::vector<int> eAxC_DL;   //!< Downlink eAxC (antenna-carrier) ID list
    std::vector<int> eAxC_PRACH_list;  //!< PRACH eAxC ID list
    std::vector<int> eAxC_SRS_list;    //!< SRS eAxC ID list
    int flow_rxq_c;             //!< RX queue index for C-plane flows
    std::vector<int> flow_rxq_u;  //!< RX queue indices for U-plane flows
    int num_ul_flows;           //!< Number of uplink flows
    int num_dl_flows;           //!< Number of downlink flows
    int num_valid_PRACH_flows;  //!< Number of valid PRACH flows
    int num_valid_SRS_flows;    //!< Number of valid SRS flows
    int ru_type;                //!< Radio unit type identifier
    aerial_fh::UserDataCompressionMethod dl_comp_meth;  //!< Downlink IQ compression method
    aerial_fh::UserDataCompressionMethod ul_comp_meth;  //!< Uplink IQ compression method
    int dl_bit_width;           //!< Downlink IQ bit width
    int ul_bit_width;           //!< Uplink IQ bit width
    int bfw_compression_bits;   //!< Beamforming weight compression bit width
    int dl_prb_size;            //!< Downlink PRB size
    int ul_prb_size;            //!< Uplink PRB size
    int fs_offset_dl;           //!< Frequency shift offset for downlink
    int exponent_dl;            //!< Exponent for downlink IQ scaling
    int ref_dl;                 //!< Reference value for downlink compression
    float beta_dl;              //!< Beta scaling factor for downlink
    float numerator;            //!< Numerator used to compute beta_dl field
    int peer_index;             //!< Index of peer node
    int vlan;                   //!< VLAN ID for this cell
    int nic_index;              //!< Network interface card index
    int dlBandwidth;            //!< Downlink bandwidth in Hz
    int ulBandwidth;            //!< Uplink bandwidth in Hz
    int dlGridSize;             //!< Downlink resource grid size (PRBs)
    int ulGridSize;             //!< Uplink resource grid size (PRBs)
    char uplink_oran_hdr_template[ORAN_IQ_HDR_SZ];  //!< Uplink ORAN header template
    dbt_md_t dbt_cfg;           //!< Dynamic beamforming table configuration
    std::array<std::unordered_set<uint16_t>, MAX_AP_PER_SLOT> dyn_bfw_beam_id_with_full_bw;  //!< Dynamic beamforming beam IDs with full bandwidth per antenna port
};

struct _cuphyPdcchDciDynPrm;
typedef struct _cuphyPdcchDciDynPrm cuphyPdcchDciPrm_t;  //!< PDCCH DCI parameter type
using dci_param_list = std::vector<cuphyPdcchDciPrm_t>;  //!< List of DCI parameters

/**
 * Parsed ORAN packet header information
 */
struct oran_packet_header_info
{
    uint16_t flowValue;          //!< Flow value (eAxC ID)
    uint8_t flow_index;          //!< Flow index within cell
    uint8_t symbolId;            //!< OFDM symbol ID
    uint8_t launch_pattern_slot; //!< Slot ID in launch pattern
    uint8_t rb;                  //!< Resource block indicator
    uint16_t sectionId;          //!< Section ID
    uint16_t startPrb;           //!< Starting PRB index
    uint16_t numPrb;             //!< Number of PRBs
    uint16_t payload_len;        //!< Payload length in bytes
    struct fssId fss;            //!< Frame/subframe/slot ID
};

/**
 * Optional channel parameters
 */
struct opt_channel_params
{
    int startSym;  //!< Starting OFDM symbol index
    int numSym;    //!< Number of OFDM symbols
    int startPrb;  //!< Starting PRB index
    int numPrb;    //!< Number of PRBs
};

//!< Launch pattern matrix: maps slot -> cell_index -> tv_index
typedef std::vector< std::unordered_map<int, int> > launch_pattern_matrix;

/**
 * Physical layer PDU (Protocol Data Unit) information
 */
struct pdu_info
{
    uint8_t startSym;            //!< Starting OFDM symbol index
    uint8_t numSym;              //!< Number of OFDM symbols
    uint8_t startDataSym;        //!< Starting data symbol index
    uint8_t numDataSym;          //!< Number of data symbols
    uint8_t dmrsMaxLength;       //!< DMRS maximum length
    uint16_t startPrb;           //!< Starting PRB index
    uint16_t numPrb;             //!< Number of PRBs
    uint8_t rb;                  //!< Resource block indicator
    uint8_t freqHopFlag;         //!< Frequency hopping flag
    uint16_t secondHopPrb;       //!< Second hop PRB for frequency hopping
    uint8_t numFlows;            //!< Number of flows
    uint32_t tb_size;            //!< Transport block size in bytes
    uint32_t dmrsPorts;          //!< DMRS port mask
    uint64_t freqDomainResource; //!< Frequency domain resource allocation
    uint8_t scid;                //!< Scrambling ID
    std::vector<uint8_t> flow_indices;  //!< Flow indices associated with this PDU
};

// SRS configuration constants
static constexpr int MAX_B_SRS_INDEX = 4;  //!< Maximum bandwidth configuration index for SRS
static constexpr int MAX_SRS_SYM = 4;      //!< Maximum number of SRS symbols

//!< SRS antenna index to number of ports mapping
static constexpr uint8_t srs_ant_idx_to_port[]={1,2,4};
//!< SRS symbol index to number of symbols mapping
static constexpr uint8_t srs_symb_idx_to_numSymb[]={1,2,4};
//!< SRS repetition factor index to repetition count mapping
static constexpr uint8_t srs_rep_factor_idx_to_numRepFactor[]={1,2,4};
//!< SRS comb index to comb size mapping
static constexpr uint8_t srs_comb_idx_to_combSize[]={2,4,8};

/**
 * SRS bandwidth information per configuration index
 */
typedef struct _bsrs_info_t
{
    uint16_t mSRS;  //!< Number of SRS resource blocks
    uint8_t nb;     //!< Number of bandwidth configurations
}bsrs_info_t;

/**
 * SRS bandwidth configuration table (per 3GPP spec)
 */
typedef struct _SrsBwConfigTable
{
    bsrs_info_t bsrs_info[MAX_B_SRS_INDEX];  //!< Bandwidth info for each index
}SrsBwConfigTable;

/* Table 6.4.1.4.3-1: SRS bandwidth configuration. */
static constexpr SrsBwConfigTable srs_bw_table[] =
{
    /* Csrs Index = 0 */
    {
        {
            {4, 1}, /* Bsrs Index = 0 */
            {4, 1}, /* Bsrs Index = 1 */
            {4, 1}, /* Bsrs Index = 2 */
            {4, 1}  /* Bsrs Index = 3 */
         }
    },
    /* Csrs Index = 1 */
    {
       {
            {8, 1}, /* Bsrs Index = 0 */
            {4, 2}, /* Bsrs Index = 1 */
            {4, 1}, /* Bsrs Index = 2 */
            {4, 1}  /* Bsrs Index = 3 */
        }
    },
    {{{12, 1}, {4, 3}, {4, 1}, {4, 1}}},
    {{{16, 1}, {4, 4}, {4, 1}, {4, 1}}},
    {{{16, 1}, {8, 2}, {4, 2}, {4, 1}}},
    {{{20, 1}, {4, 5}, {4, 1}, {4, 1}}},
    {{{24, 1}, {4, 6}, {4, 1}, {4, 1}}},
    {{{24, 1}, {12, 2}, {4, 3}, {4, 1}}},
    {{{28, 1}, {4, 7}, {4, 1}, {4, 1}}},
    {{{32, 1}, {16, 2}, {8, 2}, {4, 2}}},
    {{{36, 1}, {12, 3}, {4, 3}, {4, 1}}},
    {{{40, 1}, {20, 2}, {4, 5}, {4, 1}}},
    {{{48, 1}, {16, 3}, {8, 2}, {4, 2}}},
    {{{48, 1}, {24, 2}, {12, 2}, {4, 3}}},
    {{{52, 1}, {4, 13}, {4, 1}, {4, 1}}},
    {{{56, 1}, {28, 2}, {4, 7}, {4, 1}}},
    {{{60, 1}, {20, 3}, {4, 5}, {4, 1}}},
    {{{64, 1}, {32, 2}, {16, 2}, {4, 4}}},
    {{{72, 1}, {24, 3}, {12, 2}, {4, 3}}},
    {{{72, 1}, {36, 2}, {12, 3}, {4, 3}}},
    {{{76, 1}, {4, 19}, {4, 1}, {4, 1}}},
    {{{80, 1}, {40, 2}, {20, 2}, {4, 5}}},
    {{{88, 1}, {44, 2}, {4, 11}, {4, 1}}},
    {{{96, 1}, {32, 3}, {16, 2}, {4, 4}}},
    {{{96, 1}, {48, 2}, {24, 2}, {4, 6}}},
    {{{104, 1}, {52, 2}, {4, 13}, {4, 1}}},
    {{{112, 1}, {56, 2}, {28, 2}, {4, 7}}},
    {{{120, 1}, {60, 2}, {20, 3}, {4, 5}}},
    {{{120, 1}, {40, 3}, {8, 5}, {4, 2}}},
    {{{120, 1}, {24, 5}, {12, 2}, {4, 3}}},
    {{{128, 1}, {64, 2}, {32, 2}, {4, 8}}},
    {{{128, 1}, {64, 2}, {16, 4}, {4, 4}}},
    {{{128, 1}, {16, 8}, {8, 2}, {4, 2}}},
    {{{132, 1}, {44, 3}, {4, 11}, {4, 1}}},
    {{{136, 1}, {68, 2}, {4, 17}, {4, 1}}},
    {{{144, 1}, {72, 2}, {36, 2}, {4, 9}}},
    {{{144, 1}, {48, 3}, {24, 2}, {12, 2}}},
    {{{144, 1}, {48, 3}, {16, 3}, {4, 4}}},
    {{{144, 1}, {16, 9}, {8, 2}, {4, 2}}},
    {{{152, 1}, {76, 2}, {4, 19}, {4, 1}}},
    {{{160, 1}, {80, 2}, {40, 2}, {4, 10}}},
    {{{160, 1}, {80, 2}, {20, 4}, {4, 5}}},
    {{{160, 1}, {32, 5}, {16, 2}, {4, 4}}},
    {{{168, 1}, {84, 2}, {28, 3}, {4, 7}}},
    {{{176, 1}, {88, 2}, {44, 2}, {4, 11}}},
    {{{184, 1}, {92, 2}, {4, 23}, {4, 1}}},
    {{{192, 1}, {96, 2}, {48, 2}, {4, 12}}},
    {{{192, 1}, {96, 2}, {24, 4}, {4, 6}}},
    {{{192, 1}, {64, 3}, {16, 4}, {4, 4}}},
    {{{192, 1}, {24, 8}, {8, 3}, {4, 2}}},
    {{{208, 1}, {104, 2}, {52, 2}, {4, 13}}},
    {{{216, 1}, {108, 2}, {36, 3}, {4, 9}}},
    {{{224, 1}, {112, 2}, {56, 2}, {4, 14}}},
    {{{240, 1}, {120, 2}, {60, 2}, {4, 15}}},
    {{{240, 1}, {80, 3}, {20, 4}, {4, 5}}},
    {{{240, 1}, {48, 5}, {16, 3}, {8, 2}}},
    {{{240, 1}, {24, 10}, {12, 2}, {4, 3}}},
    {{{256, 1}, {128, 2}, {64, 2}, {4, 16}}},
    {{{256, 1}, {128, 2}, {32, 4}, {4, 8}}},
    {{{256, 1}, {16, 16}, {8, 2}, {4, 2}}},
    {{{264, 1}, {132, 2}, {44, 3}, {4, 11}}},
    {{{272, 1}, {136, 2}, {68, 2}, {4, 17}}},
    {{{272, 1}, {68, 4}, {4, 17}, {4, 1}}},
    {{{272, 1}, {16, 17}, {8, 2}, {4, 2}}}
};

/**
 * SRS resource block allocation information
 */
typedef struct _srs_rb_info_t
{
    uint16_t srs_start_prbs;  //!< Starting PRB index for SRS
    uint16_t num_srs_prbs;    //!< Number of PRBs allocated for SRS
}srs_rb_info_t;

/**
 * SRS PDU (Table 3-52 from SCF FAPI spec)
 */
typedef struct
{
    uint16_t rnti;                      //!< Radio Network Temporary Identifier
    uint32_t handle;                    //!< Handle for SRS PDU
//    scf_fapi_bwp_ts38_213_sec_12_t bwp;
    uint8_t     num_ant_ports;          //!< Number of antenna ports
    uint8_t     num_symbols;            //!< Number of symbols
    uint8_t     num_repetitions;        //!< Number of repetitions
    uint8_t     time_start_position;    //!< Time domain start position
    uint8_t     config_index;           //!< SRS configuration index
    uint16_t    sequenceId;             //!< SRS sequence ID
    uint8_t     bandwidth_index;        //!< Bandwidth configuration index
    uint8_t     comb_size;              //!< Transmission comb size
    uint8_t     comb_offset;            //!< Comb offset
    uint8_t     cyclic_shift;           //!< Cyclic shift
    uint8_t     frequency_position;     //!< Frequency domain position
    uint16_t    frequency_shift;        //!< Frequency shift value
    uint8_t     frequency_hopping;      //!< Frequency hopping bandwidth
    uint8_t     group_or_sequence_hopping;  //!< Group or sequence hopping
    uint8_t     resource_type;          //!< Resource type (periodic/semi-persistent/aperiodic)
    uint16_t    t_srs;                  //!< SRS periodicity
    uint16_t    t_offset;               //!< SRS slot offset
    uint8_t     payload[0];             //!< Variable length payload (beamforming, see Table 3-53)
} __attribute__ ((__packed__)) scf_fapi_srs_pdu_t;

/**
 * Test vector information base structure
 */
struct tv_info
{
    uint32_t tb_size;           //!< Transport block size in bytes
    uint8_t numFlows;           //!< Number of flows
    uint16_t startPrb;          //!< Starting PRB index
    uint16_t numPrb;            //!< Number of PRBs
    uint32_t modCompNumPrb = 0; //!< Number of PRBs for modulation/compression
    uint16_t endPrb;            //!< Ending PRB index
    uint8_t startSym;           //!< Starting OFDM symbol index
    uint8_t numSym;             //!< Number of OFDM symbols
    bool is_nr_tv = false;      //!< Flag indicating if this is an NR test vector
    std::vector<pdu_info> pdu_infos;  //!< List of PDU information structures
    std::vector<pdu_info> combined_pdu_infos;  //!< Combined PDU information (aggregated)
    std::array<std::array<bool, MAX_NUM_PRBS_PER_SYMBOL>, OFDM_SYMBOLS_PER_SLOT> prb_map{};  //!< PRB allocation map [symbol][prb]
    std::array<std::array<uint64_t, MAX_NUM_PRBS_PER_SYMBOL>, OFDM_SYMBOLS_PER_SLOT> prb_num_flow_map{};  //!< Number of flows per PRB [symbol][prb]
    std::unordered_map<int, std::unordered_map<int, std::vector<pdu_info>>> fss_pdu_infos;  //!< FSS-indexed PDU info map
    std::unordered_map<int, std::unordered_map<int, uint16_t>> fss_numPrb;  //!< FSS-indexed PRB count map
    std::unordered_map<int, std::unordered_map<int, std::array<std::array<bool, MAX_NUM_PRBS_PER_SYMBOL>, OFDM_SYMBOLS_PER_SLOT>>> fss_prb_map;  //!< FSS-indexed PRB allocation map
    /* nPrbDlBwp is used to compute beta_dl for DL channels
       It should match what cuphydriver uses. It is overriden (in ru-emulator) if fix_beta_dl
       is set to 1 in the config.yaml */
    uint16_t nPrbDlBwp = 0;     //!< Number of PRBs in DL BWP (for beta_dl calculation)
    uint16_t nPrbUlBwp = 0;     //!< Number of PRBs in UL BWP
    uint16_t numGnbAnt = 0;     //!< Number of gNodeB antenna ports

    // 4T4R beam ID validation fields (read from HDF5 test vectors)
    uint8_t digBFInterfaces = 0;                    //!< Number of digital beamforming interfaces
    std::vector<uint16_t> expected_beam_ids;         //!< Expected beam IDs (PDSCH/PBCH/PDCCH or primary CSI-RS set)
    //! All non-ZP CSI-RS beam ID sets from all PDUs (TRS + NZP CSI-RS).
    //! Each inner vector is one PDU's beamIdx array.
    //! Multiple CSI-RS types can coexist in a single TV (e.g. TRS with 1 beam ID
    //! and NZP CSI-RS with 4 beam IDs). Used for per-eAxC validation.
    std::vector<std::vector<uint16_t>> csirs_beam_id_sets;

    /** Per-PDU beam-to-PRB mapping entry.
     *  Associates a contiguous PRB range with the beam IDs carried on that range.
     */
    struct pdu_beam_entry {
        uint16_t startPrb;              //!< PRB start index (0-based, in PRB units)
        uint16_t numPrb;                //!< Number of PRBs in this range
        std::vector<uint16_t> beam_ids; //!< List of beam identifiers for the range
    };
    /** Per-PDU beam IDs for multi-PDU slots.
     *  Each entry maps a PRB range to its beam IDs, supporting slots with
     *  multiple PDSCH PDUs that carry distinct beam assignments.
     */
    std::vector<pdu_beam_entry> per_pdu_beam_ids;
};

/**
 * Beamforming weights information
 */
struct bfw_info
{
    // BFW params
    int bfwUL;                  //!< Beamforming weights for uplink flag
    int prgSize;                //!< Physical Resource Block Group size
    int bfwPrbGrpSize;          //!< Beamforming PRB group size
    int rbStart;                //!< Starting resource block index
    int rbSize;                 //!< Number of resource blocks
    int numPRGs;                //!< Number of PRB groups
    int compressBitWidth;       //!< Compression bit width for beamforming weights
    std::array<std::array<uint64_t, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> portMask;  //!< Port mask per cell and slot
    std::array<std::array<std::array<uint64_t, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT>, MAX_AP_PER_SLOT> active_eaxc_ids;  //!< Active eAxC IDs per antenna port, cell, and slot
    std::array<std::array<uint64_t, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> expect_prbs;  //!< Expected PRB count per cell and slot
};

/**
 * Downlink test vector information (extends tv_info)
 */
struct dl_tv_info : tv_info
{
    uint32_t qam_size;          //!< QAM modulation size (e.g., 16, 64, 256)
    uint8_t startDataSym;       //!< Starting data symbol index
    uint8_t numDataSym;         //!< Number of data symbols
    uint8_t dmrsMaxLength;      //!< DMRS maximum length
    uint8_t numTb;              //!< Number of transport blocks
    bool is_nr_tv = false;      //!< Flag indicating if this is an NR test vector

    //CSIRS PARAMS
    std::vector<std::vector<uint32_t>> csirsREsToValidate;  //!< CSI-RS resource elements to validate
    std::vector<uint32_t> csirsSkippedREs;    //!< CSI-RS skipped resource elements
    std::vector<uint16_t> csirsREMaskArray;   //!< CSI-RS RE mask array
    std::vector<uint16_t> csirsREMaskArrayTRSNZP;  //!< CSI-RS RE mask for TRS/NZP
    std::vector<pdu_info> csirc_pdu_infos;    //!< CSI-RS PDU information
    bool isZP;                  //!< Flag indicating zero-power CSI-RS
    uint32_t csirsNumREsSkipped = 0;  //!< Number of skipped CSI-RS REs
    uint16_t csirsNumREs = 0;   //!< Total number of CSI-RS REs
    uint8_t csirsMaxPortNum = 0;  //!< Maximum CSI-RS port number

    std::array<uint8_t, 14> numFlowsArray;  //!< Number of flows per symbol
    uint32_t csirsExpectedNumREs = 0;  //!< Expected number of CSI-RS REs

    //PDSCH + NON OVERLAPPING CSI_RS
    bool hasZPCsirsPdu = false;  //!< Flag indicating presence of ZP CSI-RS PDU
    //PDSCH + OVERLAPPING CSI_RS
    uint32_t numOverlappingCsirs = 0;     //!< Number of overlapping CSI-RS
    uint32_t numNonOverlappingCsirs = 0;  //!< Number of non-overlapping CSI-RS
    bool fullyOverlappingCsirs = false;   //!< Flag for fully overlapping CSI-RS
    std::array<std::array<uint64_t, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> total_expected_prbs = {};  //!< Total expected PRBs per cell and slot
    std::vector<bfw_info> bfw_infos;      //!< Beamforming weights information list
};

/**
 * CPU assignment for uplink cell processing
 */
struct ul_cell_cpu_assignment
{
    int thread_id;          //!< Thread ID in the thread group (SRS or UL/DL C-plane)
    int start_cell_index;   //!< Starting cell index for this thread
    int num_cells_per_core; //!< Number of cells per core (negative if multiple threads per cell)
};

using ul_cell_cpu_assignment_array = std::array<ul_cell_cpu_assignment, MAX_RU_THREADS>;  //!< Array of CPU assignments for UL cells

/**
 * Calculate CPU assignment for uplink cells
 *
 * @param[in] num_cells Number of cells to assign
 * @param[in] num_cores Number of CPU cores available
 * @param[in] enable_mmimo Enable massive MIMO mode
 * @param[in] is_srs Whether this is for SRS processing
 * @param[in] min_cores_per_cell_mmimo Minimum cores per cell for mMIMO (default: 3)
 * @return Array of CPU assignments per thread
 */
ul_cell_cpu_assignment_array get_cell_cpu_assignment(int num_cells, int num_cores, bool enable_mmimo, bool is_srs, int min_cores_per_cell_mmimo = 3);

/**
 * Uplink test vector information (extends tv_info)
 */
struct ul_tv_info : tv_info
{
    uint8_t numSections = 0;  //!< Number of sections in the C-plane message
};

/**
 * Performance timing measurements for TX symbol processing
 */
struct tx_symbol_timers {
    /* measured by tx_symbol */
    uint64_t tx_symbol_info_copy_start_t;  //!< Start time for TX symbol info copy
    uint64_t tx_symbol_info_copy_end_t;    //!< End time for TX symbol info copy
    uint64_t tx_symbol_info_copy_t;        //!< Duration of TX symbol info copy

    uint64_t packet_parsing_start_t;       //!< Start time for packet parsing
    uint64_t packet_parsing_end_t;         //!< End time for packet parsing
    uint64_t packet_parsing_t;             //!< Duration of packet parsing

    uint64_t counter_inc_start_t;          //!< Start time for counter increment
    uint64_t counter_inc_end_t;            //!< End time for counter increment
    uint64_t counter_inc_t;                //!< Duration of counter increment

    uint64_t tx_symbol_start_t;            //!< Start time for TX symbol
    uint64_t tx_symbol_end_t;              //!< End time for TX symbol
    uint64_t tx_symbol_t;                  //!< Duration of TX symbol

    uint64_t prb_match_start_t;            //!< Start time for PRB matching
    uint64_t prb_match_end_t;              //!< End time for PRB matching
    uint64_t prb_match_t;                  //!< Duration of PRB matching

    uint64_t umsg_alloc_start_t;           //!< Start time for U-plane message allocation
    uint64_t umsg_alloc_end_t;             //!< End time for U-plane message allocation
    uint64_t umsg_alloc_t;                 //!< Duration of U-plane message allocation

    uint64_t prepare_start_t;              //!< Start time for packet preparation
    uint64_t prepare_end_t;                //!< End time for packet preparation
    uint64_t send_start_t;                 //!< Start time for packet send
    uint64_t send_end_t;                   //!< End time for packet send
    uint64_t prepare_t;                    //!< Duration of packet preparation
    uint64_t send_t;                       //!< Duration of packet send

    uint64_t prepare_sum_t = 0;            //!< Cumulative preparation time
    uint64_t send_sum_t = 0;               //!< Cumulative send time
    uint64_t overall_sum_t = 0;            //!< Cumulative overall time
    /* measured from the "outside" by tx_slot */
    uint64_t overall;                      //!< Overall time measured externally

    /**
     * Clear all timing measurements
     */
    void clear()
    {
        tx_symbol_info_copy_t = 0;
        packet_parsing_t = 0;
        counter_inc_t = 0;
        tx_symbol_t = 0;
        prb_match_t = 0;
        umsg_alloc_t = 0;
        prepare_t = 0;
        send_t = 0;
    }
};

//!< Smart pointer type for aerial FH memory management
typedef std::unique_ptr<void, decltype(&aerial_fh::free_memory)> unique_void_ptr;

/**
 * Dataset container for test vector data
 */
struct Dataset {
    Dataset() : size(0), data(nullptr, aerial_fh::free_memory) {};  //!< Default constructor
    size_t size;            //!< Size of dataset in bytes
    unique_void_ptr data;   //!< Smart pointer to data buffer
};

/**
 * Slot data structure for organizing IQ samples by antenna, symbol, and PRB
 */
struct Slot {
    /**
     * Constructor initializes pointer array for fast PRB access
     * @param[in] size Number of antenna ports
     */
    Slot(size_t size) {
        for(int ant_idx = 0; ant_idx < size; ++ant_idx)
        {
            ptrs.emplace_back(std::vector<std::vector<void*>>());
            for(int sym_idx = 0; sym_idx < SLOT_NUM_SYMS; ++sym_idx)
            {
                ptrs[ant_idx].emplace_back(std::vector<void*>());
                for(int prb_idx = 0; prb_idx < MAX_NUM_PRBS_PER_SYMBOL; ++prb_idx)
                {
                    ptrs[ant_idx][sym_idx].emplace_back(nullptr);
                }
            }
        }
    }
    /* Pointers for quickly accessing pointers to PRBs */
    std::vector<std::vector<std::vector<void *>>> ptrs;  //!< 3D pointer array [antenna][symbol][prb]

    size_t data_sz = 0;         //!< Total data size in bytes
    size_t antenna_sz = 0;      //!< Size per antenna in bytes
    size_t symbol_sz = 0;       //!< Size per symbol in bytes
    size_t prb_sz = 0;          //!< Size per PRB in bytes
    size_t prbs_per_symbol = 0; //!< Number of PRBs per symbol
    size_t prbs_per_slot = 0;   //!< Total PRBs per slot
    int pkts_per_slot = 0;      //!< Number of packets per slot
    Dataset raw_data;           //!< Raw IQ data buffer
};

/**
 * Helper structure for TX symbol processing
 */
typedef struct tx_symbol_helper
{
    uint8_t cell_index;         //!< Cell index
    uint16_t eaxcId;            //!< eAxC ID (antenna-carrier ID)
    uint8_t eaxcId_index;       //!< eAxC ID index within cell
    bool valid_eaxcId;          //!< Flag indicating valid eAxC ID
    uint16_t section_id;        //!< Section ID
    uint8_t rb;                 //!< Resource block indicator
    uint8_t symInc;             //!< Symbol increment flag
    uint16_t startPrbc;         //!< Starting PRB index (compressed)
    uint16_t numPrbc;           //!< Number of PRBs (compressed)
    uint16_t reMask;            //!< Resource element mask
    uint8_t ef;                 //!< Extension flag
    uint8_t startSym;           //!< Starting OFDM symbol index
    uint16_t beamId;            //!< Beam ID for beamforming
    int32_t freqOffset;         //!< Frequency offset
    uint64_t tx_time;           //!< Transmit time in nanoseconds
    uint8_t prach_pdu_index;    //!< PRACH PDU index
    uint32_t tv_index;          //!< Test vector index
    struct fssId fss{0,0,0};    //!< Frame/subframe/slot ID
    struct ul_tv_object* tv_object;  //!< Pointer to UL test vector object
    ul_channel channel_type;    //!< Uplink channel type
} tx_symbol_helper;

/**
 * ORAN C-plane section extension information
 */
typedef struct oran_c_plane_section_ext_info_t
{
    uint8_t* ext_ptr = nullptr;  //!< Pointer to extension data
    uint16_t ext_len = 0;        //!< Extension length in bytes
    uint8_t ext_type = 0;        //!< Extension type
    uint8_t ef = 0;              //!< Extension flag
} oran_c_plane_section_ext_info_t;

/**
 * ORAN C-plane section information
 */
typedef struct oran_c_plane_section_info_t
{
    uint16_t section_id;        //!< Section ID
    uint8_t rb;                 //!< Resource block indicator
    uint8_t symInc;             //!< Symbol increment flag
    uint16_t startPrbc;         //!< Starting PRB index (compressed)
    uint16_t numPrbc;           //!< Number of PRBs (compressed)
    uint16_t reMask;            //!< Resource element mask
    uint8_t ef;                 //!< Extension flag
    uint8_t* ext11_ptr = nullptr;  //!< Pointer to extension 11 data
    uint8_t numSymbol;          //!< Number of symbols
    uint16_t beamId;            //!< Beam ID for beamforming
    int32_t freqOffset;         //!< Frequency offset
    uint8_t reserved;           //!< Reserved field

    uint64_t rx_time;           //!< Receive time in nanoseconds
    uint64_t tx_time;           //!< Transmit time in nanoseconds
    uint8_t prach_pdu_index;    //!< PRACH PDU index
    uint16_t tv_index;          //!< Test vector index
    struct ul_tv_object* tv_object;  //!< Pointer to UL test vector object
    ul_channel channel_type;    //!< Uplink channel type
    oran_c_plane_section_ext_info_t ext_infos[MAX_NUM_SECTION_EXTENSIONS];  //!< Array of section extensions
    int ext_infos_size = 0;     //!< Number of valid section extensions
    int error_status = 0;       //!< 0: no error, 1: error
    public:
    /**
     * Print section information to console for debugging
     */
    void print()
    {
        re_cons("{:>15s}: {:<5d}", getName(section_id), section_id);
        re_cons("{:>15s}: {:<5d}", getName(rb), rb);
        re_cons("{:>15s}: {:<5d}", getName(symInc), symInc);
        re_cons("{:>15s}: {:<5d}", getName(startPrbc), startPrbc);
        re_cons("{:>15s}: {:<5d}", getName(numPrbc), numPrbc);
        re_cons("{:>15s}: {:<5d}", getName(reMask), reMask);
        re_cons("{:>15s}: {:<5d}", getName(ef), ef);
        re_cons("{:>15s}: {:<5d}", getName(numSymbol), numSymbol);
        re_cons("{:>15s}: {:<5d}", getName(beamId), beamId);
        re_cons("{:>15s}: {:<5d}", getName(freqOffset), freqOffset);
        re_cons("{:>15s}: {:<5d}", getName(rx_time), rx_time);
        re_cons("{:>15s}: {:<5d}", getName(tx_time), tx_time);
        re_cons("{:>15s}: {:<5d}", getName(prach_pdu_index), prach_pdu_index);
        re_cons("{:>15s}: {:<5d}", getName(tv_index), tv_index);
    }
} oran_c_plane_section_info_t;

/**
 * Per-sectionId record for intra-slot uniqueness & multiple-citations checks.
 */
struct SlotSectionIdEntry {
    uint32_t generation{0};
    uint16_t startPrbc{0};
    uint16_t numPrbc{0};
    uint8_t  rb{0};
    uint8_t  numSymbol{0};
    uint8_t  udCompHdr{0};
};

/**
 * Tracks sectionId usage per slot for a single eAxC and a single ID range [BASE, BASE+N).
 * Used for DL (full 0-4095) and for UL split into PUSCH+PUCCH (0-1023), PRACH (2048-2111), SRS (3072-3135).
 *
 * Validates: sectionId uniqueness, "multiple citations" consistency, and (DL only) C/U-plane cross-match.
 */
template <size_t N, uint16_t BASE = 0>
struct SlotSectionIdTrackerRange
{
    static constexpr size_t NUM_ENTRIES = N;
    static constexpr uint16_t BASE_ID = BASE;
    static constexpr uint16_t MAX_SECTION_ID = (BASE + N <= 4096) ? static_cast<uint16_t>(BASE + N - 1) : 4095;

    bool in_range(uint16_t sid) const { return sid >= BASE && sid < BASE + N; }
    size_t to_index(uint16_t sid) const { return static_cast<size_t>(sid - BASE); }

    // --- Current slot state ---
    std::array<SlotSectionIdEntry, N> entries{};
    std::bitset<N> uplane_sids_seen{};
    std::bitset<N> cplane_sids_announced{};
    uint32_t current_generation{1};
    struct fssId current_fss{0, 0, 0};

    // --- Previous slot state (for 2-slot deferred cross-validation, DL only) ---
    std::bitset<N> prev_uplane_sids_seen{};
    std::bitset<N> prev_cplane_sids_announced{};
    struct fssId prev_fss{0, 0, 0};
    bool has_prev{false};

    /**
     * @brief Advances the tracker to a new slot, archiving current state as previous.
     *
     * Side-effects: copies @c uplane_sids_seen (U-plane section IDs seen this slot)
     * and @c cplane_sids_announced (C-plane section IDs announced this slot) into
     * their @c prev_ counterparts; increments @c current_generation (skipping zero),
     * used to invalidate duplicate-consistency entries across slots; resets both
     * bitsets and stores @p new_fss as @c current_fss.
     *
     * @param[in] new_fss  Frame/subframe/slot identifier for the new slot.
     */
    void advance_slot(const struct fssId& new_fss)
    {
        prev_uplane_sids_seen = uplane_sids_seen;
        prev_cplane_sids_announced = cplane_sids_announced;
        prev_fss = current_fss;
        has_prev = true;
        if (++current_generation == 0) ++current_generation;
        uplane_sids_seen.reset();
        cplane_sids_announced.reset();
        current_fss = new_fss;
    }

    /**
     * @brief Advances the tracker to a new slot with deferred cross-validation
     *        of the previous slot's U-plane sectionIds against C-plane announcements.
     *
     * Before archiving: computes the set of section IDs in @c prev_uplane_sids_seen
     * but not in @c prev_cplane_sids_announced and invokes @p warn_fn once per
     * such orphan (U-plane section without C-plane announcement). Then performs
     * the same side-effects as the single-argument overload (archive to @c prev_*,
     * increment @c current_generation, reset @c uplane_sids_seen and
     * @c cplane_sids_announced, set @c current_fss).
     *
     * @tparam WarnFn  Callable with signature <tt>void(uint16_t sid, const fssId& fss)</tt>.
     *
     * @param[in] new_fss  Frame/subframe/slot identifier for the new slot.
     * @param[in] warn_fn  Called for each unmatched U-plane sectionId from the
     *                     previous slot; receives the absolute sectionId and
     *                     the previous slot's fssId.
     */
    template <typename WarnFn>
    void advance_slot(const struct fssId& new_fss, WarnFn&& warn_fn)
    {
        if (has_prev && prev_uplane_sids_seen.any())
        {
            auto unmatched = prev_uplane_sids_seen & ~prev_cplane_sids_announced;
            if (unmatched.any())
            {
#if defined(__GLIBCXX__)
                for (size_t i = unmatched._Find_first(); i < N; i = unmatched._Find_next(i))
                    warn_fn(static_cast<uint16_t>(BASE + i), prev_fss);
#else
                for (size_t i = 0; i < N; ++i)
                {
                    if (unmatched.test(i))
                        warn_fn(static_cast<uint16_t>(BASE + i), prev_fss);
                }
#endif
            }
        }
        prev_uplane_sids_seen = uplane_sids_seen;
        prev_cplane_sids_announced = cplane_sids_announced;
        prev_fss = current_fss;
        has_prev = true;
        if (++current_generation == 0) ++current_generation;
        uplane_sids_seen.reset();
        cplane_sids_announced.reset();
        current_fss = new_fss;
    }

    /**
     * @brief Tests whether @p fss refers to the same slot currently being tracked.
     *
     * @param[in] fss  Frame/subframe/slot identifier to compare.
     * @return @c true if @p fss matches @c current_fss on all three fields.
     */
    bool is_same_slot(const struct fssId& fss) const
    {
        return fss.frameId == current_fss.frameId &&
               fss.subframeId == current_fss.subframeId &&
               fss.slotId == current_fss.slotId;
    }

    /**
     * @brief Tests whether @p fss is a forward-monotonic slot relative to the
     *        current tracked slot, accounting for frame-id wraparound.
     *
     * @param[in] fss  Frame/subframe/slot identifier to compare.
     * @return @c true if @p fss is strictly ahead of @c current_fss (within
     *         half the total slot space, to distinguish forward wrap from backward).
     */
    bool is_forward_slot(const struct fssId& fss) const
    {
        static_assert(static_cast<int64_t>(ORAN_MAX_FRAME_ID) * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID <= INT32_MAX,
                      "FSS slot linearization must fit in int32_t to avoid overflow");
        constexpr int32_t total_slots = ORAN_MAX_FRAME_ID * ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID;
        int32_t cur = current_fss.frameId * (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)
                    + current_fss.subframeId * ORAN_MAX_SLOT_ID + current_fss.slotId;
        int32_t nxt = fss.frameId * (ORAN_MAX_SUBFRAME_ID * ORAN_MAX_SLOT_ID)
                    + fss.subframeId * ORAN_MAX_SLOT_ID + fss.slotId;
        int32_t diff = nxt - cur;
        if (diff < 0) diff += total_slots;
        return diff > 0 && diff < total_slots / 2;
    }

    /**
     * @brief Records a U-plane sectionId as seen in the current slot.
     *
     * Sets the corresponding bit in @c uplane_sids_seen if @p sid falls
     * within the tracker's valid range [BASE, BASE+N).  Out-of-range IDs
     * are silently ignored.
     *
     * @param[in] sid  The sectionId from the U-plane packet header.
     */
    void record_uplane_sid(uint16_t sid)
    {
        if (in_range(sid))
            uplane_sids_seen.set(to_index(sid));
    }
};

/// DL: section IDs start from 0; max per eAxC is 448 (C-plane) or 896 (U-plane mMIMO). 1024 covers both.
static constexpr size_t DL_SECTION_ID_SIZE = 1024;
using SlotSectionIdTracker = SlotSectionIdTrackerRange<DL_SECTION_ID_SIZE, 0>;

/// UL section ID ranges: limited per-channel size and channel-specific start indices to match DU-side
/// (cuphydriver / FH driver) usage rather than ORAN max limits.
static constexpr uint16_t UL_PRACH_SECTION_ID_BASE = 2048;
static constexpr uint16_t UL_SRS_SECTION_ID_BASE   = 3072;
static constexpr size_t   UL_PRACH_SECTION_ID_SIZE = 64;
static constexpr size_t   UL_SRS_SECTION_ID_SIZE   = 64;
static constexpr size_t   UL_PUSCH_PUCCH_SECTION_ID_SIZE = 1024;

using SlotSectionIdTrackerPuschPucch = SlotSectionIdTrackerRange<UL_PUSCH_PUCCH_SECTION_ID_SIZE, 0>;
using SlotSectionIdTrackerPrach      = SlotSectionIdTrackerRange<UL_PRACH_SECTION_ID_SIZE, UL_PRACH_SECTION_ID_BASE>;
using SlotSectionIdTrackerSrs        = SlotSectionIdTrackerRange<UL_SRS_SECTION_ID_SIZE, UL_SRS_SECTION_ID_BASE>;

/** Per-eAxC UL: three trackers (PUSCH+PUCCH, PRACH, SRS) with channel-specific section ID ranges
 *  and limited sizes, aligned with DU/cuphy/FH channel-specific start indices and section counts. */
struct ULSectionIdTrackerSet
{
    SlotSectionIdTrackerPuschPucch pusch_pucch;
    SlotSectionIdTrackerPrach      prach;
    SlotSectionIdTrackerSrs       srs;

    /** Advance all three to the same slot (call when any would advance). */
    void advance_slot_all(const struct fssId& new_fss)
    {
        pusch_pucch.advance_slot(new_fss);
        prach.advance_slot(new_fss);
        srs.advance_slot(new_fss);
    }

    /** Record U-plane sectionId in the tracker that covers it (only one will). */
    void record_uplane_sid(uint16_t sid)
    {
        if (pusch_pucch.in_range(sid)) pusch_pucch.record_uplane_sid(sid);
        else if (prach.in_range(sid))  prach.record_uplane_sid(sid);
        else if (srs.in_range(sid))    srs.record_uplane_sid(sid);
    }
};

/**
 * ORAN C-plane message information (aggregated from multiple sections)
 */
typedef struct oran_c_plane_info_t
{
    struct fssId fss{0,0,0};    //!< Frame/subframe/slot ID
    uint8_t launch_pattern_slot{};  //!< Slot ID in launch pattern
    uint8_t dir{};              //!< Direction (0=UL, 1=DL)
    uint16_t eaxcId{};          //!< eAxC ID (antenna-carrier ID)
    int16_t eaxcId_index{};     //!< eAxC ID index within cell
    uint8_t section_type{};     //!< Section type (0=unused, 1=all RBs, 3=PRB mask)
    uint8_t section_id{};       //!< Section ID
    uint8_t udCompHdr{};        //!< User data compression header from C-plane common header (udCompMeth[3:0] | udIqWidth[7:4])
    uint8_t startSym{};         //!< Starting OFDM symbol index
    uint8_t numSym{};           //!< Number of OFDM symbols
    uint16_t startPrbc{};       //!< Starting PRB index (compressed)
    uint16_t numPrbc{};         //!< Number of PRBs (compressed)
    uint16_t rx_index{};        //!< Receive index
    uint64_t rx_time{};         //!< Receive time in nanoseconds
    uint64_t rte_rx_time{};     //!< DPDK RTE receive time
    uint64_t packet_processing_start{};  //!< Packet processing start time
    uint64_t ul_processing_start{};      //!< UL processing start time
    uint64_t verification_section_start{};  //!< Verification section start time
    uint64_t verification_section_end{};    //!< Verification section end time
    uint64_t tx_slot_start{};   //!< TX slot start time
    uint64_t packet_prepare_start{};  //!< Packet preparation start time
    int nb_rx{};                //!< Number of packets received
    int64_t slot_t0{};          //!< Slot t0 reference time
    uint64_t tx_offset{};       //!< TX time offset
    bool valid_eaxcId{};        //!< Flag indicating valid eAxC ID
    struct ul_tv_object* tv_object{};  //!< Pointer to UL test vector object
    struct dl_tv_object* dl_tv_object{};  //!< Pointer to DL test vector object
    ul_channel channel_type{};  //!< Uplink channel type
    bool is_mixed_channel{};    //!< True if sections have different channel types
    uint16_t tv_index{};        //!< Test vector index
    uint8_t numberOfSections{}; //!< Number of sections in this C-plane message
    // std::vector<oran_c_plane_section_info_t> section_infos;
    oran_c_plane_section_info_t section_infos[MAX_NUM_SECTIONS_PER_C_PLANE];  //!< Array of section information
    int section_infos_size{};   //!< Number of valid section information entries
} oran_c_plane_info_t;

/**
 * Slot TX information (batch of C-plane messages for processing)
 */
struct slot_tx_info
{
    oran_c_plane_info_t c_plane_infos[CPLANE_DEQUEUE_BURST_SIZE];  //!< Array of C-plane info
    int c_plane_infos_size{};       //!< Total number of C-plane messages
    int ul_c_plane_infos_size{};    //!< Number of UL C-plane messages
};

/**
 * Test vector object base structure (DL/UL common fields)
 */
struct tv_object
{
    std::string channel_string;  //!< Channel name string (e.g., "PUSCH", "PDSCH")
    std::vector<std::string> tv_names;  //!< Test vector file names
    std::unordered_map<std::string, int> tv_map;  //!< Map from TV name to TV index

    launch_pattern_matrix launch_pattern;       //!< Current launch pattern matrix
    launch_pattern_matrix init_launch_pattern;  //!< Initial launch pattern (for reset)

    std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> throughput_counters;  //!< Throughput counters per cell
    std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT> throughput_slot_counters;  //!< Slot counters for throughput
    std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT> good_slot_counters;  //!< Successful slot counters
    std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT> error_slot_counters;  //!< Error slot counters
    std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT> total_slot_counters;  //!< Total slot counters
    std::array<std::atomic<uint8_t>, MAX_CELLS_PER_SLOT> init_slot_counters;  //!< Initialization slot counters
    std::array<std::atomic<uint8_t>, MAX_CELLS_PER_SLOT> initialization_phase;  //!< Initialization phase flags

    /**
     * Check if a cell is active in a given slot
     * @param[in] cell_index Cell index
     * @param[in] slot Slot index in launch pattern
     * @return true if cell exists in slot, false otherwise
     */
    bool cell_exists_in_slot(int cell_index, int slot)
    {
        if(launch_pattern[slot].find(cell_index) == launch_pattern[slot].end())
        {
            return false;
        }
        return true;
    }
};

/**
 * Encode a floating-point scale value (0, 1) into a 15-bit modCompScaler
 * format: 4-bit exponent | 11-bit mantissa.
 * Decoded value = mantissa / (2048 * 2^exp).
 * Uses the same direct frexp-based conversion as the DU side so that
 * the encoded values match exactly for TV comparison.
 */
inline uint16_t float_to_modcompscaler(float scale)
{
    if (!(scale > 0.0f && scale < 1.0f))
    {
        return 0;
    }

    int exp = 0;
    float mantissa_float = std::frexp(scale, &exp);

    int mantissa = static_cast<int>(std::round(mantissa_float * 2048.0f));
    int custom_exp = -exp;

    mantissa = std::clamp(mantissa, 0, 0x7FF);
    custom_exp = std::clamp(custom_exp, 0, 15);

    return static_cast<uint16_t>((custom_exp << 11) | mantissa);
}

/**
 * Per-message expected SE4/SE5 extension values loaded from the TV.
 */
struct tv_mod_comp_ext_info
{
    uint32_t ext_type = 0;
    uint32_t n_mask = 0;
    uint16_t mc_scale_re_mask[2] = {0, 0};
    uint16_t mc_scale_offset_encoded[2] = {0, 0};
    uint32_t csf[2] = {0, 0};
    bool valid = false;
};

/**
 * Test vector modulation/compression object for DL
 */
struct tv_mod_comp_object
{
    //!< Modulation/compression header map: [symbol][portIdx][reMask] -> list of {startPrb, numPrb, msg_idx}
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<uint32_t, std::vector<std::array<int, 3>>>>> mod_comp_header;
    std::vector<Dataset> mod_comp_payload;  //!< Modulation/compression payload data
    std::unordered_map<int, int> global_msg_idx_to_tv_idx;  //!< Map from global message index to TV index
    //!< Payload parameters: {startPrb, numPrb, udIqWidth, skip_iq_validation}
    std::vector<std::array<int, 4>> mod_comp_payload_params;
    std::vector<tv_mod_comp_ext_info> mod_comp_ext_info;  //!< Per-message expected SE4/SE5 values from TV
    //std::array<std::array<std::array<std::array<std::array<std::unordered_map<int, int>, 64>, SLOT_NUM_SYMS>, ORAN_MAX_SLOT_X_SUBFRAME_ID>, ORAN_MAX_FRAME_ID>, MAX_CELLS_PER_SLOT> fss_mod_comp_payload_idx;
};

/**
 * Downlink test vector object (extends tv_object)
 */
struct dl_tv_object : tv_object
{
    std::array<std::vector<Dataset>, IQ_DATA_FMT_MAX> qams;  //!< QAM datasets indexed by IQ format type
    std::vector<struct tv_mod_comp_object> mod_comp_data;    //!< Modulation/compression data per TV

    std::vector<struct dl_tv_info> tv_info;  //!< DL test vector information list
    dl_channel channel_type;                 //!< Downlink channel type
    nrsim_tv_type::nrsim_tv_type nrsim_ch_type;  //!< NR simulator channel type
    //CSIRS
    std::vector<std::vector<std::vector<uint32_t>>> received_res;  //!< CSI-RS received resource elements
    std::array<std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT>, MAX_LAUNCH_PATTERN_SLOTS>  atomic_received_res;  //!< Atomic REs received per cell/slot
    std::array<std::array<std::array<std::atomic<uint32_t>, ORAN_MAX_SLOT_X_SUBFRAME_ID>, ORAN_MAX_FRAME_ID>, MAX_CELLS_PER_SLOT> fss_atomic_received_res;  //!< FSS-indexed atomic received REs
    std::array<std::array<std::array<uint64_t, ORAN_MAX_SLOT_X_SUBFRAME_ID>, ORAN_MAX_FRAME_ID>, MAX_CELLS_PER_SLOT> fss_atomic_received_res_prev_ts;  //!< Previous timestamp for FSS REs

    std::array<std::vector<bool>, MAX_CELLS_PER_SLOT> invalid_flag;  //!< Invalid slot flags per cell

    std::array<std::array<std::atomic<uint32_t>, MAX_CELLS_PER_SLOT>, MAX_LAUNCH_PATTERN_SLOTS>  atomic_received_prbs;  //!< Atomic PRBs received per cell/slot
    std::array<aerial_fh::FHMutex, MAX_CELLS_PER_SLOT> mtx;  //!< Mutex per cell for thread safety
};

/**
 * Uplink test vector object (extends tv_object)
 */
struct ul_tv_object : tv_object
{
    ul_tv_object() : blank_prbs(nullptr, aerial_fh::free_memory),channel_type(ul_channel::NONE) {};  //!< Constructor
    std::array<std::vector<Slot>, IQ_DATA_FMT_MAX> slots;  //!< UL slot data indexed by IQ format type
    std::array<std::vector<std::vector<Slot> >, IQ_DATA_FMT_MAX> prach_slots;  //!< PRACH slot data [fmt][occasion][preamble]

    ul_channel channel_type;            //!< Uplink channel type
    unique_void_ptr blank_prbs;         //!< Blank PRB data for unused resources
    std::vector<struct ul_tv_info> tv_info;  //!< UL test vector information list
    std::array<std::atomic<uint16_t>, MAX_CELLS_PER_SLOT> c_plane_rx{};  //!< C-plane packets received per cell
    std::array<std::atomic<uint16_t>, MAX_CELLS_PER_SLOT> u_plane_tx{};  //!< U-plane packets transmitted per cell
    std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> c_plane_rx_tot{};  //!< Total C-plane packets received
    std::array<std::atomic<uint64_t>, MAX_CELLS_PER_SLOT> u_plane_tx_tot{};  //!< Total U-plane packets transmitted
    std::array<std::array<std::atomic<uint16_t>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> section_rx_counters{};  //!< Section RX counters [cell][slot]
    std::array<std::array<std::atomic<uint32_t>, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> prb_rx_counters{};  //!< PRB RX counters [cell][slot]
    std::array<std::array<aerial_fh::FHMutex, MAX_LAUNCH_PATTERN_SLOTS>, MAX_CELLS_PER_SLOT> rx_counters_mtx{};  //!< RX counter mutexes [cell][slot]
};

// ---------------------------------------------------------------------------
// Pre-computed UL TX cache structures
// ---------------------------------------------------------------------------
// Leverages the deterministic launch pattern to pre-resolve per-eAxC flow
// handles, TX queue indices, and TX time offsets at init time, eliminating
// per-packet TV lookup and tx_time calculation from the hot TX path.

/**
 * @brief Pre-resolved TX parameters for one eAxC within a launch-pattern
 *        slot/cell combination.
 *
 * Built once at init by precompute_ul_tx_cache() and read on the hot TX
 * path by tx_slot_precomputed().  Every entry in the containing vector
 * is valid; no sentinel/placeholder values exist.
 */
struct PrecomputedEaxcTx {
    aerial_fh::FlowHandle flow{};               ///< ORAN flow handle for this eAxC (from ul_peer_flow_map or peer_flow_map_srs).
    ul_channel channel_type = ul_channel::NONE;  ///< UL channel type (PUSCH, PUCCH, or SRS).
    ul_tv_object* tv_object = nullptr;           ///< Owning TV object for IQ data and launch pattern lookup.
    uint16_t eaxcId = 0;                         ///< ORAN eAxC ID transmitted in the eCPRI header.
    int16_t eaxcId_index = -1;                   ///< Zero-based index into the cell's eAxC list; used to select IQ data pointers.
    int start_sym = 0;                           ///< First active symbol for this channel (from tv_info.startSym).
    int num_sym = 0;                             ///< Number of active symbols (from tv_info.numSym).
    std::array<int, ORAN_ALL_SYMBOLS> txq_index{};           ///< Per-symbol TX queue index (from get_txq_index()).
    std::array<int64_t, ORAN_ALL_SYMBOLS> tx_time_offset_ns{}; ///< Per-symbol TX time offset in ns relative to slot T0.
};

/**
 * @brief Collection of pre-resolved eAxC TX entries for one
 *        (launch-pattern slot, cell) pair.
 */
struct PrecomputedSlotCellTx {
    std::vector<PrecomputedEaxcTx> eaxc_entries; ///< All eAxC entries for this slot/cell (PUSCH + PUCCH + SRS).
};

// SIMD vector types for optimized IQ sample processing
typedef int16_t  i16x8  __attribute__((vector_size (16)));   //!< Vector to hold 8 x 16-bit samples
typedef uint8_t  u8x16  __attribute__((vector_size (16)));   //!< Vector to hold 16 x 8-bit samples
typedef uint64_t u64x4  __attribute((vector_size (32)));     //!< Vector to hold intermediate 4 x 64-bit words
typedef uint16_t u16x16 __attribute((vector_size (32)));     //!< Vector used for shuffling 16-bit data out of 64-bit words

/**
 * Union for accessing IQ samples in different formats
 */
union SampleSet_t {
	i16x8 vec;         //!< SIMD vector view (8x int16)
	__uint128_t data;  //!< 128-bit data view
	uint32_t dw[4];    //!< Doubleword array view
	int16_t w[8];      //!< Word array view
};

/**
 * Check if global force quit flag is set
 * @return true if quit requested, false otherwise
 */
bool check_force_quit();

/**
 * Set global force quit flag to request shutdown
 */
void set_force_quit();

/**
 * Check if timer has started
 * @return true if timer started, false otherwise
 */
bool check_timer_start();

/**
 * Set timer start flag
 */
void set_timer_start();

/**
 * Increment global frame/subframe/slot ID
 * @param[in] max_slot_id Maximum slot ID value (depends on TTI)
 */
void increment_glob_fss(int max_slot_id);

/**
 * Get current global frame/subframe/slot ID
 * @param[out] fss Output FSS structure
 * @param[in] max_slot_id Maximum slot ID value
 */
void get_glob_fss(struct fssId& fss, int max_slot_id);

/**
 * Set global frame/subframe/slot ID
 * @param[in] fss FSS structure to set
 * @param[in] max_slot_id Maximum slot ID value
 */
void set_glob_fss(struct fssId fss, int max_slot_id);

/**
 * Initialize channel name string mappings
 */
void channel_string_setup();

/**
 * Signal handler for graceful shutdown
 * @param[in] signum Signal number
 */
void signal_handler(int signum);

/**
 * Setup signal handlers for SIGINT, SIGTERM, etc.
 */
void signal_setup();

/**
 * Throw runtime error with message
 * @param[in] what Error message
 */
void do_throw(std::string const& what);

/**
 * Get CPU affinity list as string
 * @return Comma-separated list of CPU IDs
 */
std::string affinity_cpu_list();

/**
 * Set thread priority to maximum (real-time)
 * @return 0 on success, negative on error
 */
int set_max_thread_priority();

/**
 * Calculate Adler-32 checksum (default implementation)
 * @param[in] buffer Data buffer
 * @param[in] size Size of buffer in bytes
 * @return 32-bit checksum
 */
uint32_t adler32(void* buffer, int size);

/**
 * Calculate Adler-32 checksum (naive implementation)
 * @param[in] buffer Data buffer
 * @param[in] size Size of buffer in bytes
 * @return 32-bit checksum
 */
uint32_t adler32_naive(void* buffer, int size);

/**
 * Calculate Adler-32 checksum (optimized implementation)
 * @param[in] buffer Data buffer
 * @param[in] size Size of buffer in bytes
 * @return 32-bit checksum
 */
uint32_t adler32_optimized(unsigned char* buffer, int size);

/**
 * Calculate Adler-32 checksum (ISA-L implementation)
 * @param[in] buffer Data buffer
 * @param[in] size Size of buffer in bytes
 * @return 32-bit checksum
 */
uint32_t adler32_isal(void* buffer, int size);

/**
 * Write QAM data to file for debugging
 * @param[in] filename Output filename
 * @param[in] buffer Data buffer
 * @param[in] size Size in bytes
 */
void write_qam_to_file(std::string filename, void* buffer, size_t size);

/**
 * Print ORAN packet header information to console
 * @param[in] header_info Header information structure
 */
void print_header_info(struct oran_packet_header_info& header_info);

/**
 * Compare IQ buffers with approximate tolerance
 * @param[in] rx_buff Received buffer
 * @param[in] tv_buff Test vector buffer
 * @param[in] length Buffer length in bytes
 * @return 0 on match, negative on mismatch
 */
int compare_approx_buffer(uint8_t * rx_buff, uint8_t * tv_buff, size_t length);

/**
 * Decompress and compare IQ buffers with approximate tolerance
 * @param[in] rx_buff Received compressed buffer
 * @param[in] tv_buff Test vector buffer
 * @param[in] length Buffer length
 * @param[in] dl_bit_width DL IQ bit width
 * @param[in] beta Beta scaling factor
 * @param[in] flow Flow index (for debug)
 * @param[in] symbol_id Symbol ID (for debug)
 * @param[in] prb_start Starting PRB (for debug)
 * @return 0 on match, negative on mismatch
 */
int decompress_and_compare_approx_buffer(uint8_t * rx_buff, uint8_t * tv_buff, size_t length, int dl_bit_width, float beta, int flow, int symbol_id, int prb_start);

/**
 * Decompress and compare beamforming weight buffers
 * @param[in] rx_buff Received compressed buffer
 * @param[in] rx_exp Received exponent
 * @param[in] tv_buff Test vector buffer
 * @param[in] tv_exp Test vector exponent
 * @param[in] length Buffer length
 * @param[in] dl_bit_width DL bit width
 * @param[in] beta Beta scaling factor
 * @param[in] flow Flow index (for debug)
 * @param[in] symbol_id Symbol ID (for debug)
 * @param[in] bundle_index Bundle index (for debug)
 * @param[in] decompress_tv Whether to decompress TV buffer
 * @param[in] numGnbAnt Number of gNB antennas
 * @return 0 on match, negative on mismatch
 */
int decompress_and_compare_approx_bfw_bundle_buffer(uint8_t * rx_buff, uint8_t rx_exp, uint8_t * tv_buff, uint8_t tv_exp, size_t length, int dl_bit_width, float beta, int flow, int symbol_id, int bundle_index, bool decompress_tv, uint16_t numGnbAnt);

/**
 * Compare fixed-point beamforming bundles
 * @param[in] rx_buf Received buffer
 * @param[in] tv_buf Test vector buffer
 * @param[in] nBytes Number of bytes
 * @param[in] rx_exp Received exponent
 * @param[in] tv_exp Test vector exponent
 * @return 0 on match, negative on mismatch
 */
int fixedpt_bundle_compare(uint8_t* rx_buf, uint8_t* tv_buf, int nBytes, uint8_t rx_exp, uint8_t tv_exp);

/**
 * Compare half-precision floating point values with tolerance
 * @param[in] a First value
 * @param[in] b Second value
 * @param[in] tolf Tolerance (default 0.00098)
 * @return true if approximately equal
 */
inline bool compare_approx(const __half& a, const __half& b, const float tolf = 0.00098f);

/**
 * Get current time in nanoseconds
 * @return Current time in nanoseconds since epoch
 */
inline uint64_t get_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

/**
 * Busy wait for specified time in nanoseconds
 * @param[in] ns Time to wait in nanoseconds
 */
inline void wait_ns(uint64_t ns)
{
    uint64_t end_t = get_ns() + ns, start_t = 0;
    while ((start_t = get_ns()) < end_t) {
        for (int spin_cnt = 0; spin_cnt < 1000; ++spin_cnt) {
            __asm__ __volatile__ ("");
        }
    }
}

/**
 * Print buffer contents as hexadecimal bytes (48 bytes per line)
 * @param[in] ptr Buffer pointer
 * @param[in] size Size in bytes
 */
inline void print_bytes(void *ptr, int size)  {
    unsigned char *p = (unsigned char *)ptr;
    int i, j;
    for(i = 0; i < size;){
        for (j=0; j<48 && i < size; ++j,++i) {
            printf("%02hhX ", p[i]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * String builder helper class for constructing strings from streams
 */
class sb
{
    std::stringstream ss_;  //!< Internal string stream
public:
    /**
     * Stream insertion operator
     * @param[in] x Value to insert
     * @return Reference to this string builder
     */
    template <class T>
    sb &operator<<(T const &x) {ss_ << x; return *this;}

    /**
     * Convert to std::string
     * @return Constructed string
     */
    operator std::string() const {return ss_.str();}
};

/**
 * Get pointer to PRB data within a slot
 * @param[in] s Slot structure
 * @param[in] flow_id Flow (antenna) index
 * @param[in] symbol_id Symbol index within slot
 * @param[in] prb_id PRB index within symbol
 * @return Pointer to PRB data
 */
inline void *get_prb(Slot const& s, uint16_t flow_id, uint16_t symbol_id, size_t prb_id)
{
    return s.ptrs.at(flow_id).at(symbol_id).at(prb_id);
}

/**
 * Increment frame/subframe/slot ID with proper rollover
 * @param[in,out] fss FSS structure to increment
 * @param[in] max_slot_id Maximum slot ID before rollover (depends on TTI)
 */
inline void increment_fss(struct fssId& fss, int max_slot_id)
{
    fss.slotId++;
    if(fss.slotId > max_slot_id)
    {
        fss.subframeId++;
        fss.slotId = 0;
    }

    if(fss.subframeId >= ORAN_MAX_SUBFRAME_ID)
    {
        fss.frameId++;
        fss.subframeId = 0;
    }
/*
    if(fss.frameId >= ORAN_MAX_FRAME_ID) // Always false as frameId is of type unit8_t
    {
        fss.frameId = 0;
    }
*/
}

/**
 * Convert FSS to launch pattern slot index
 * @param[in] fss Frame/subframe/slot ID
 * @param[in] launch_pattern_slot_size Total launch pattern slots
 * @return Launch pattern slot index
 */
inline int fss_to_launch_pattern_slot(const struct fssId& fss, int launch_pattern_slot_size)
{
    return fss.frameId % (launch_pattern_slot_size / (ORAN_MAX_SLOT_X_SUBFRAME_ID)) * (ORAN_MAX_SLOT_X_SUBFRAME_ID) + fss.subframeId * (ORAN_MAX_SLOT_ID) + fss.slotId;
}

//!< Map from test vector name to dataset
typedef std::unordered_map<std::string, Dataset> dataset_map;

/**
 * Find modulation/compression message ID for given PRB allocation
 * @param[in] mod_comp_data Modulation/compression data object
 * @param[in] sym Symbol index
 * @param[in] portIdx Port index (antenna)
 * @param[in] startPrb Starting PRB
 * @param[in] numPrb Number of PRBs
 * @return Message ID if found, -1 otherwise
 */
inline int find_modcomp_msg_id(struct tv_mod_comp_object &mod_comp_data, int sym, int portIdx, int startPrb, int numPrb)
{
    auto &hdr = mod_comp_data.mod_comp_header;
    if (hdr.find(sym) != hdr.end() && hdr[sym].find(portIdx) != hdr[sym].end())
    {
        for (auto &[_, list] : hdr[sym][portIdx])
        {
            for (auto &e : list)
            {
                if (startPrb >= e[0] && startPrb + numPrb <= e[0] + e[1])
                {
                    return e[2];
                }
            }
        }
    }
    return -1;
}

/**
 * Converts ul_channel enum to its string representation
 *
 * @param[in] channel The ul_channel enum value to convert
 * @return String representation of the channel type
 */
inline std::string ul_channel_to_string(const ul_channel channel)
{
    switch (channel) {
        case ul_channel::NONE:   return "NONE";
        case ul_channel::PUSCH:  return "PUSCH";
        case ul_channel::PRACH:  return "PRACH";
        case ul_channel::PUCCH:  return "PUCCH";
        case ul_channel::SRS:    return "SRS";
        default:                 return "UNKNOWN";
    }
}

constexpr uint16_t kMaxSFN      = 1024;  //!< Maximum System Frame Number (10-bit SFN wraps at 1024)

/**
 * ORAN slot number representation
 */
struct OranSlotNumber
{
    uint8_t frame_id;     //!< Frame ID (8-bit for ORAN, wraps at 256)
    uint8_t subframe_id;  //!< Subframe ID (0-9 for 1ms subframes)
    uint8_t slot_id;      //!< Slot ID within subframe (depends on numerology)
    int SFN;              //!< System Frame Number (10-bit, 0-1023)
};

/**
 * Iterator for advancing through ORAN slot numbers with proper rollover
 */
class OranSlotIterator {
public:
    /**
     * Constructor
     * @param[in] start_slot_number Initial slot number
     */
    OranSlotIterator(OranSlotNumber start_slot_number) : slot_number_{start_slot_number}
    {
        slot_number_.SFN = slot_number_.frame_id;
    };

    /**
     * Get next slot number and advance iterator
     * @return Current slot number (before advancing)
     */
    OranSlotNumber get_next()
    {
        auto current_slot_number = slot_number_;

        slot_number_.slot_id = (slot_number_.slot_id + 1) % ORAN_MAX_SLOT_ID;

        if(slot_number_.slot_id == 0)
        {
            slot_number_.subframe_id = (slot_number_.subframe_id + 1) % ORAN_MAX_SUBFRAME_ID;
        }

        if((slot_number_.slot_id == 0) && (slot_number_.subframe_id == 0))
        {
            slot_number_.frame_id = (slot_number_.frame_id + 1) % ORAN_MAX_FRAME_ID;
            slot_number_.SFN = (slot_number_.SFN + 1) % kMaxSFN;
        }
        return current_slot_number;
    };

protected:
    OranSlotNumber slot_number_;  //!< Current slot number state
};

#endif
