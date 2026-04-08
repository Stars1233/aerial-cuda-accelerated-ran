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

#ifndef _COMMON_DEFINES_HPP_
#define _COMMON_DEFINES_HPP_

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <map>
#include <vector>
#include <atomic>
#include <iostream>
#include <unordered_map>

#include "hdf5hpp.hpp"
#include "nv_ipc_utils.h"

// Define common log tags which can be used among all modules
#ifndef TAG_MAC
#define TAG_MAC (NVLOG_TAG_BASE_TEST_MAC + 0) //!< Base log tag for MAC module: "MAC"
#endif
#ifndef TAG_CUMAC
#define TAG_CUMAC (NVLOG_TAG_BASE_TEST_MAC + 20) //!< Base log tag for cuMAC module: "CUMAC"
#endif

/**
 * NR numerology definitions
 * 
 * Numerology determines subcarrier spacing and slot duration:
 * - 0: 15 kHz SCS, 1ms slot
 * - 1: 30 kHz SCS, 0.5ms slot (default)
 * - 2: 60 kHz SCS, 0.25ms slot
 * - 3: 120 kHz SCS, 0.125ms slot
 * - 4: 240 kHz SCS, 0.0625ms slot
 */
#define NR_NUMEROLOGY 1 //!< Default numerology (30 kHz SCS, 20 slots per frame)
#define SLOT_INTERVAL (1000L * 1000 / (1 << NR_NUMEROLOGY)) //!< Slot interval in nanoseconds
#define SLOTS_PER_SECOND (1000L * 1000 * 1000 / SLOT_INTERVAL) //!< Number of slots per second
#define SLOTS_PER_FRAME (1000L * 1000 * 10 / SLOT_INTERVAL) //!< Number of slots per 10ms frame

#define CELL_ID_ALL 0xFFFF //!< Special cell ID representing all cells

/**
 * Slot type enumeration
 * 
 * Defines the TDD slot configuration
 */
typedef enum
{
    SLOT_NONE = 0,     //!< No specific direction
    SLOT_UPLINK = 1,   //!< Uplink slot
    SLOT_DOWNLINK = 2, //!< Downlink slot
    SLOT_SPECIAL = 3,  //!< Special slot (mixed UL/DL)
} slot_type_t;

/**
 * IPC synchronization mode
 * 
 * Controls when IPC event notifications are sent to PHY
 */
typedef enum
{
    IPC_SYNC_PER_CELL = 0, //!< Send one notification per cell (after all messages for cell are sent)
    IPC_SYNC_PER_TTI  = 1, //!< Send one notification per TTI (after all cells are sent)
    IPC_SYNC_PER_MSG  = 2, //!< Send one notification per message
} ipc_sync_mode_t;

/**
 * Scheduling information structure
 * 
 * Contains timing information for slot scheduling
 */
typedef struct
{
    sfn_slot_t ss; //!< SFN/SLOT of SLOT.indication
    int64_t ts;    //!< Timestamp of SLOT.indication in nanoseconds
} sched_info_t;

/**
 * Slot timing structure
 * 
 * Tracks timing information for UL messages per UE
 */
typedef struct
{
    sfn_slot_t ss;             //!< SFN/SLOT for this timing record
    uint32_t rnti;             //!< UE RNTI
    struct timespec ts_ul_tti; //!< Timestamp of UL TTI request
} slot_timing_t;

/**
 * Validation enable level
 * 
 * Controls what level of validation errors to report
 */
typedef enum
{
    VALD_ENABLE_NONE = 0, //!< Validation disabled
    VALD_ENABLE_ERR  = 1, //!< Report errors only
    VALD_ENABLE_WARN = 2, //!< Report errors and warnings
} vald_enable_t;

/**
 * Validation logging mode
 * 
 * Controls granularity of validation logging
 */
typedef enum
{
    VALD_LOG_PER_NONE  = 0, //!< No validation logging
    VALD_LOG_PER_MSG   = 1, //!< Log summary per message
    VALD_LOG_PER_PDU   = 2, //!< Log details per PDU
    VALD_LOG_PRINT_ALL = 3, //!< Log all validation details
} vald_log_t;

/**
 * Validation result enumeration
 * 
 * Result of a validation operation
 */
typedef enum
{
    VALD_OK   = 0, //!< Validation passed
    VALD_FAIL = 1, //!< Validation failed
} vald_result_t;

#endif /* _COMMON_DEFINES_HPP_ */
