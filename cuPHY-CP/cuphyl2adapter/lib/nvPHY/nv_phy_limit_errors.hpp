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
#include <cstdint>
#include <limits>
#include <array>
namespace nv {

#ifdef ENABLE_L2_SLT_RSP

/**
 * @brief Structure to manage TX notification settings
 */
struct TxNotificationHelper {
    static bool enable_tx_notification;

    /**
     * @brief Get the current state of TX notification
     * @return Current state of TX notification (true if enabled, false if disabled)
     */
    [[nodiscard]] static bool getEnableTxNotification() noexcept {
        return enable_tx_notification;
    }

    /**
     * @brief Set the state of TX notification
     * @param[in] enabled New state for TX notification (true to enable, false to disable)
     */
    static void setEnableTxNotification(const bool enabled) noexcept {
        enable_tx_notification = enabled;
    }
};

static constexpr uint16_t MAX_PDSCH_GROUP_ERRORS = std::numeric_limits<uint8_t>::max() + 1;
static constexpr uint16_t MAX_PDCCH_GROUP_ERRORS = std::numeric_limits<uint8_t>::max() + 1;

/**
 * @brief Structure to hold SSB/PBCH related limit errors
 */
struct ssb_pbch_limit_error_t {
    uint8_t errors;  ///< Error code for SSB/PBCH limit violations
    uint8_t parsed;  ///< Parsed status for SSB/PBCH
};

/**
 * @brief Structure to hold PDCCH PDU error context information
 *
 * This structure contains context information for a PDCCH PDU error,
 * including the associated Radio Network Temporary Identifier (RNTI).
 */
struct pdcch_pdu_error_ctxt_t {
    uint16_t rnti;      ///< Radio Network Temporary Identifier
};

/**
 * @brief Structure to hold PDCCH related limit errors
 */
struct pdcch_limit_error_t {
    uint8_t coreset_errors;    ///< Error code for CORESET limit violations
    uint8_t dci_errors;        ///< Error code for DCI limit violations
    uint8_t coreset_parsed;    ///< Parsed status for CORESET
    uint8_t dci_parsed;        ///< Parsed status for DCI
    pdcch_pdu_error_ctxt_t pdu_error_contexts[MAX_PDCCH_GROUP_ERRORS];  ///< Array of PDCCH PDU error contexts
};

/**
 * @brief Structure to hold CSI-RS related limit errors
 */
struct csirs_limit_error_t {
    uint8_t errors;  ///< Error code for CSI-RS limit violations
    uint8_t parsed;  ///< Parsed status for CSI-RS
};

/**
 * @brief Structure to hold PDSCH PDU error context information
 * 
 */
struct pdsch_pdu_error_ctxt_t {
    uint16_t rnti;      ///< Radio Network Temporary Identifier
    uint8_t pduIndex;   ///< PDU index for error tracking
};

using pdsch_pdu_error_ctxts_t = std::array<pdsch_pdu_error_ctxt_t, MAX_PDSCH_GROUP_ERRORS>;

/**
 * @brief Structure to hold PDSCH PDU error contexts info per cell
 */
struct pdsch_pdu_error_ctxts_info_t {
    uint16_t pdsch_pdu_error_ctxt_num;          ///< Number of PDSCH PDU error contexts
    pdsch_pdu_error_ctxts_t pdsch_pdu_error_contexts;  ///< Array of PDSCH PDU error contexts
};

/**
 * @brief Structure to hold PDSCH related limit errors
 */
struct pdsch_limit_error_t {
    uint8_t errors;  ///< Error code for PDSCH limit violations
    uint8_t parsed;  ///< Parsed status for PDSCH
};

/**
 * @brief Structure to hold PUSCH related limit errors
 */
struct pusch_limit_error_t {
    uint8_t errors;  ///< Error code for PUSCH limit violations
    uint8_t parsed;  ///< Parsed status for PUSCH
};

/**
 * @brief Structure to hold PUCCH related limit errors
 */
struct pucch_limit_error_t {
    uint8_t pf0_errors;    ///< Error code for PUCCH format 0 limit violations
    uint8_t pf1_errors;    ///< Error code for PUCCH format 1 limit violations
    uint8_t pf2_errors;    ///< Error code for PUCCH format 2 limit violations
    uint8_t pf3_errors;    ///< Error code for PUCCH format 3 limit violations
    uint8_t pf4_errors;    ///< Error code for PUCCH format 4 limit violations
    uint8_t pf0_parsed;    ///< Parsed status for PUCCH format 0
    uint8_t pf1_parsed;    ///< Parsed status for PUCCH format 1
    uint16_t pf2_parsed;   ///< Parsed status for PUCCH format 2 (needs uint16_t for limits up to 480)
    uint16_t pf3_parsed;   ///< Parsed status for PUCCH format 3 (needs uint16_t for limits up to 480)
    uint16_t pf4_parsed;   ///< Parsed status for PUCCH format 4 (needs uint16_t for limits up to 480)
};

/**
 * @brief Structure to hold SRS related limit errors
 */
struct srs_limit_error_t {
    uint8_t errors;  ///< Error code for SRS limit violations
    uint8_t parsed;  ///< Parsed status for SRS
};

/**
 * @brief Structure to hold PRACH related limit errors
 */
struct prach_limit_error_t {
    uint8_t errors;  ///< Error code for PRACH limit violations
    uint8_t parsed;  ///< Parsed status for PRACH
};

/**
 * @brief Structure to hold all cell-related limit errors
 */
struct slot_limit_cell_error_t {
    ssb_pbch_limit_error_t ssb_pbch_errors;  ///< SSB/PBCH related errors
    pdcch_limit_error_t pdcch_errors;        ///< PDCCH related errors
    csirs_limit_error_t csirs_errors;        ///< CSI-RS related errors
    srs_limit_error_t srs_errors;            ///< SRS related errors
    prach_limit_error_t prach_errors;        ///< PRACH related errors
    pdsch_pdu_error_ctxts_info_t pdsch_pdu_error_contexts_info;  ///< PDSCH PDU error contexts info
};

/**
 * @brief Structure to hold all cell group-related limit errors
 */
struct slot_limit_group_error_t {
    pdsch_limit_error_t pdsch_errors;  ///< PDSCH related errors
    pusch_limit_error_t pusch_errors;  ///< PUSCH related errors
    pucch_limit_error_t pucch_errors;  ///< PUCCH related errors
};

#endif
} // namespace nvphy 