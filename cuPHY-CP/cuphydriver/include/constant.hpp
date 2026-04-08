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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>
#include "aerial-fh-driver/oran.hpp"
#include "aerial-fh-driver/doca_structs.hpp"
#include "slot_command/slot_command.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// System Configuration - Core limits and resource pools
////////////////////////////////////////////////////////////////////////////////////////////////////////
#define cell_id_t uint64_t                                           ///< Unique cell identifier type (timestamp-based)

static constexpr uint32_t FINALIZE_TIMEOUT_NS=2000000000;           ///< Timeout for system finalization (2 seconds)
static constexpr uint32_t SLOT_MAP_NUM=512;                         ///< Number of slot map objects in the pool
static constexpr uint32_t SLOT_CMD_NUM=512;                         ///< Number of slot command objects in the pool
static constexpr uint32_t TASK_ITEM_NUM=2048;                       ///< Maximum task items in the system
static constexpr uint32_t TASK_LIST_SIZE=4096;                      ///< Task list buffer size
static constexpr uint32_t TIME_THRESHOLD_NS_TASK_ACCEPT=100000;     ///< Time threshold (100us) for accepting tasks
static constexpr uint32_t TASK_MAX_PER_SLOT=64;                     ///< Maximum concurrent tasks per slot
static constexpr uint32_t MAX_CPU_CORES=48;                         ///< Maximum CPU cores supported
static constexpr uint32_t MAX_NUM_TASKS = 5;                        ///< Maximum number of task types
static constexpr uint32_t DEFAULT_GPU_ID = 0;                       ///< Default GPU device ID
////////////////////////////////////////////////////////////////////////////////////////////////////////
//// PHY Channel Aggregation - Number of aggregation objects per context for each channel type
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t PHY_ULBFW_AGGR_X_CTX = 4;                 ///< UL beamforming weight aggregation objects per context
static constexpr uint32_t PRACH_MAX_OCCASIONS_AGGR = slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT;  ///< Max PRACH occasions for aggregation
static constexpr uint32_t PRACH_MAX_OCCASIONS = slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT;       ///< Max PRACH occasions per slot
static constexpr uint32_t PRACH_MAX_NUM_PREAMBLES = 64;             ///< Maximum number of PRACH preambles
static constexpr uint32_t PHY_DLBFW_AGGR_X_CTX = 10;                ///< DL beamforming weight aggregation objects per context
static constexpr uint32_t PHY_PDSCH_AGGR_X_CTX = 5;                ///< PDSCH aggregation objects per context
static constexpr uint32_t PHY_PDCCH_DL_AGGR_X_CTX = 10;             ///< DL PDCCH aggregation objects per context
static constexpr uint32_t PHY_PDCCH_UL_AGGR_X_CTX = 0;              ///< UL PDCCH aggregation objects per context (unused)
static constexpr uint32_t PHY_PBCH_AGGR_X_CTX = 10;                 ///< PBCH aggregation objects per context
static constexpr uint32_t PHY_CSIRS_AGGR_X_CTX = 10;                ///< CSI-RS aggregation objects per context

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// PUSCH Configuration Limits
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t PHY_PUSCH_MAX_FREQ_MULTI = 1;             ///< Maximum frequency multiplexing for PUSCH
static constexpr uint32_t PHY_PUSCH_MAX_MIMO = 16;                  ///< Maximum MIMO layers for PUSCH
static constexpr uint32_t PHY_PUSCH_MAX_CB_PER_TB = 148;            ///< Maximum code blocks per transport block
static constexpr uint32_t PHY_PUSCH_MAX_BYTES_PER_TB = 311386;      ///< Maximum bytes per transport block

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Downlink Buffer Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_32DL
static constexpr uint32_t DL_OUTPUT_BUFFER_SIZE = 8388608;          ///< DL output buffer size: 8MB (for 32 layer support)
#else
static constexpr uint32_t DL_OUTPUT_BUFFER_SIZE = 4194304;          ///< DL output buffer size: 4MB (for 16 layer support)
#endif
static constexpr uint32_t DL_OUTPUT_BUFFER_NUM_PER_CELL = 16;       ///< Number of DL output buffers per cell
static constexpr uint32_t DL_OUTPUT_BUFFER_BUSY_NS = 500000 * 4;    ///< DL buffer busy time: 2ms (500us x 4 slots)
                                                                    ///< Defines how long a DL output buffer remains busy after it was last used,
                                                                    ///< Specifically, for GPU initiated communications. 
static constexpr uint32_t DL_HELPER_MEMSET_BUFFERS_PER_CTX = 4;     ///< Number of helper memset buffers per context (must be power of 2)

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Antenna and Resource Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_32DL
static constexpr uint32_t MAX_AP_PER_SLOT = 32;                     ///< Maximum antenna ports per slot (32-layer DL)
#else
static constexpr uint32_t MAX_AP_PER_SLOT = 16;                     ///< Maximum antenna ports per slot (16-layer DL)
#endif
static constexpr uint32_t MAX_AP_PER_SLOT_SRS = 64;                 ///< Maximum antenna ports per slot for SRS
static constexpr uint32_t MAX_UE_SRS_ANT_PORTS = 4;                 ///< Maximum UE antenna ports for SRS

static constexpr uint32_t MAX_AP_PER_SLOT_CSI_RS = 32;              ///< Maximum antenna ports per slot for CSI-RS
static constexpr uint32_t MAX_SECTIONS_PER_CPLANE_SYMBOL = 32;      ///< Maximum C-plane sections per OFDM symbol
#ifdef ENABLE_32DL
static constexpr uint32_t MAX_SECTIONS_PER_UPLANE_SYMBOL = 64;      ///< Maximum U-plane sections per OFDM symbol (Note: increased for >32 PDCCH DCIs)
#else
static constexpr uint32_t MAX_SECTIONS_PER_UPLANE_SYMBOL = 32;      ///< Maximum U-plane sections per OFDM symbol
#endif

static constexpr uint32_t MAX_NUM_OF_NIC_SUPPORTED = 2;             ///< Maximum number of NICs supported
static constexpr uint32_t MU_SUPPORTED = 1;                         ///< Numerology (μ) supported: 1 = 30kHz SCS

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// ORAN Message Limits - Fronthaul packet and section configuration
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t TX_PKTS_SYMBOL_ANTENNA = 10;              ///< TX packets per symbol per antenna
static constexpr uint32_t TX_SYMBOL_MAX_PKTS = (MAX_AP_PER_SLOT * TX_PKTS_SYMBOL_ANTENNA); ///< Max TX packets per symbol (all antennas)

static constexpr uint32_t MAX_UPLANE_MSGS_PER_SLOT = (MAX_SECTIONS_PER_UPLANE_SYMBOL * MAX_AP_PER_SLOT * ORAN_ALL_SYMBOLS); ///< Max U-plane messages per slot
static constexpr uint32_t MAX_CPLANE_MSGS_PER_SLOT = (ORAN_ALL_SYMBOLS * MAX_AP_PER_SLOT); ///< Max C-plane messages per slot
static constexpr uint32_t MAX_CPLANE_SECTIONS_PER_SLOT_PER_AP = (ORAN_ALL_SYMBOLS * MAX_SECTIONS_PER_CPLANE_SYMBOL); ///< Max C-plane sections per slot per antenna
static constexpr uint32_t MAX_CPLANE_SECTIONS_PER_SLOT = (MAX_CPLANE_MSGS_PER_SLOT * MAX_SECTIONS_PER_CPLANE_SYMBOL); ///< Max C-plane sections per slot (all antennas)

// Section extension type 11 is used for beamforming weights
static constexpr uint32_t MAX_CPLANE_SECTIONS_EXT_PER_SLOT = (ORAN_MAX_PRB * MAX_AP_PER_SLOT); ///< Max C-plane extension sections per slot (max 273 for extType 11)

static constexpr uint32_t MAX_CPLANE_EXT_11_BUNDLES_PER_SLOT = (ORAN_MAX_PRB * ORAN_MAX_SYMBOLS * MAX_AP_PER_SLOT); ///< Max extType 11 bundles per slot
static constexpr uint16_t DEFAULT_PHY_CELL_ID = 0xFFFF;             ///< Default Physical Cell ID (indicates inactive/unconfigured cell)

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// ORAN and Data Processing Parameters
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t ORAN_RE_X_PRB = (DEFAULT_PRB_STRIDE / ORAN_RE); ///< Resource elements per PRB

static constexpr uint32_t USER_DATA_IQ_BIT_WIDTH = 16;              ///< IQ data bit width per ORAN-WG4-CUS.0-v03.00

static constexpr uint32_t GENERIC_WAIT_THRESHOLD_NS = 4000000;      ///< Generic wait threshold: 4ms
static constexpr uint32_t ORDER_KERNEL_ENABLE_THRESHOLD = GENERIC_WAIT_THRESHOLD_NS; ///< Order kernel enable threshold: 4ms
                                                                                     ///< timeout threshold for task synchronization waits throughout the uplink and downlink processing pipelines.
static constexpr uint32_t ORDER_ENTITY_NUM = 8;                     ///< Number of order kernel entities

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Uplink Task Timing Offsets - All offsets relative to T0 (slot boundary)
//// These define when each processing stage should be launched to meet latency requirements
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS=500000;           ///< Order kernel launch: T0 - 500us (PUSCH/PUCCH ordering)
static constexpr uint32_t UL_TASK1_SRS_ORDER_LAUNCH_OFFSET_FROM_T0_NS=2500000;      ///< SRS order kernel launch: T0 + 2500us

static constexpr uint32_t UL_TASK1_PUCCH_LAUNCH_OFFSET_FROM_T0_NS=500000;           ///< PUCCH processing launch: T0 - 500us

static constexpr uint32_t UL_TASK1_PUSCH_LAUNCH_OFFSET_FROM_T0_NS=400000;           ///< PUSCH processing launch: T0 - 400us
static constexpr uint32_t UL_TASK3_EARLY_UCI_IND_TASK_LAUNCH_OFFSET_FROM_T0_NS=1500000; ///< Early UCI indication task: T0 + 1500us

static constexpr uint32_t UL_TASK1_SRS_LAUNCH_OFFSET_FROM_T0_NS = UL_TASK1_SRS_ORDER_LAUNCH_OFFSET_FROM_T0_NS; ///< SRS processing launch (same as SRS order)

static constexpr uint32_t UL_TASK2_OFFSET_FROM_T0_NS=200000;                        ///< UL Task 2 (CPU init comms): T0 - 200us

///< AGGR 3 Offset from T0 - Note that if running with only 2 UL cores+EH we cannot schedule this farther than 500us from early UCI aggr 3 task
///<                        Otherwise UL cores are locked in with *4 and *5 slots, and each waits 8ms for early UCI aggr 3 task
static constexpr uint32_t UL_TASK3_AGGR3_OFFSET_FROM_T0_NS=UL_TASK3_EARLY_UCI_IND_TASK_LAUNCH_OFFSET_FROM_T0_NS-490000; ///<(T0+UL_TASK3_EARLY_UCI_IND_TASK_LAUNCH_OFFSET_FROM_T0_NS)

static constexpr uint32_t SRS_COMPLETION_TH_FROM_T0_NS=5500000;                     ///< SRS completion deadline: T0 + 5.5ms (11 slots)
static constexpr uint32_t UL_TASK3_AGGR3_MAX_BACKOFF_FROM_SRS_COMPLETION_TH_NS=4000000; ///< SRS task 3 (AGGR3) max backoff from SRS completion deadline: 4ms

static constexpr uint32_t UL_AGGR3_ULBFW_OFFSET_FROM_T0_NS=800000;                  ///< UL BFW task 3 (ULBFW) launch offset from T0: T0 - 800us

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Metrics Configuration - Prometheus metrics histogram bins and labels
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint64_t METRIC_PROCESSING_TIME_BIN_SIZE_US = 250; ///< Processing time histogram bin size: 250us
static constexpr uint64_t METRIC_PROCESSING_TIME_MAX_BIN_US = 2000; ///< Processing time histogram max: 2ms
static constexpr uint64_t METRIC_UE_PER_SLOT_BIN_SIZE = 2;          ///< UE per slot histogram bin size: 2 UEs
static constexpr uint64_t METRIC_UE_PER_SLOT_MAX_BIN = 24;          ///< UE per slot histogram max: 24 UEs
static constexpr char METRIC_CELL_KEY[] = "cell";                   ///< Prometheus label key for cell ID
static constexpr char METRIC_DIRECTION_KEY[] = "type";              ///< Prometheus label key for direction (UL/DL)
static constexpr char METRIC_CHANNEL_KEY[] = "channel";             ///< Prometheus label key for channel type

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Miscellaneous Constants
////////////////////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t MAX_MCS_INDEX = 28;                       ///< Maximum Modulation and Coding Scheme index

static constexpr uint32_t ORDER_KERNEL_MAX_RX_PKTS = 512;           ///< Maximum packets order kernel can receive per iteration
static constexpr uint32_t ORDER_KERNEL_RX_PKTS_TIMEOUT_NS = 100000; ///< Order kernel RX packet timeout in each iteration: 100us

static constexpr uint8_t TASK_NAME_RESERVE_LENGTH = 128;            ///< Reserved length for task name strings
static constexpr uint8_t TASK_LIST_RESERVE_LENGTH = 64;             ///< Reserved length for task list
static constexpr uint8_t TASK_LIST_NUM_QUEUES = 32;                 ///< Number of task queues in the system

static constexpr uint8_t UPLANE_TASKS = 2;                          ///< Number of U-plane processing tasks

static constexpr uint32_t FH_EXTENSION_DELAY_ADJUSTMENT = 100000;   ///< Fronthaul extension delay adjustment: 100us

static constexpr uint8_t  MAX_PDSCH_TB_CPY_CUDA_EVENTS = 10;        ///< Maximum CUDA events for PDSCH TB copy operations

static constexpr uint8_t  TICK_SLOT_ADVANCE_INIT_VAL=3;             ///< Initial value for tick slot advance (3 slots)

static constexpr uint8_t  SLOTS_PER_FRAME = 20;                     ///< Slots per 10ms frame for μ=1 (30kHz SCS)

static constexpr uint32_t DL_BFW_TX_RUN_START_THRESHOLD_MS = 5;     ///< DL beamforming TX start threshold: 5ms
                                                                    ///< Timeout to monitor DLBFW CUDA kernel launch and start execution.
static constexpr uint32_t DL_BFW_TX_RUN_COMPLETION_THRESHOLD_MS = 5; ///< DL beamforming TX completion threshold: 5ms
                                                                    ///< Timeout to monitor DLBFW CUDA kernel execution completion.

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Macros and Utility Classes
////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef ACCESS_ONCE
    #define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))            ///< Macro for volatile access to prevent compiler optimizations
#endif


#include "exceptions.hpp"

/**
 * Type-safe static cast wrapper with null pointer checking
 * 
 * @tparam T  Target type to cast to
 */
template <class T>
class StaticConversion {
public:
    /**
     * Construct a StaticConversion wrapper
     * 
     * @param[in] _ptr  Void pointer to be cast
     */
    explicit StaticConversion(void* _ptr) :
        ptr(_ptr){}

    /**
     * Get the casted pointer with null check
     * 
     * @return Pointer cast to type T
     * @throws PHYDRIVER_THROW_EXCEPTIONS if pointer is null
     */
    [[nodiscard]] T* get()
    {
        T* tmp = static_cast<T*>(ptr);
        if(tmp == nullptr)
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "Invalid pointer: StaticConversion can't return nullptr");
        return tmp;
    }

private:
    void* ptr;                                                       ///< Stored void pointer
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
//// States for the UL RX queue processing state machine
////////////////////////////////////////////////////////////////////////////////////////////////////////
enum ul_status {
    UL_INIT = 0,                                                     ///< Initial state: RX queue not yet locked
    UL_SETUP,                                                        ///< Setup state: RX queue locked, waiting to start order kernel
    UL_START,                                                        ///< Start state: Order kernel launched
    UL_ORDERED,                                                      ///< Ordered state: Packet ordering complete, RX queue unlocked
};
#endif
