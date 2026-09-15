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

#if !defined(SCF_5G_FAPI_ERRORS_HPP_INCLUDED_)
#define SCF_5G_FAPI_ERRORS_HPP_INCLUDED_

/**
 * @file scf_5g_fapi_errors.hpp
 * @brief Lightweight error types for SCF 5G FAPI slot and message parsing.
 *
 * Extracted from scf_5g_fapi_slot_concepts.hpp so that call sites (e.g.
 * nv_phy_dl_channels.cpp) can include just this header to inspect error
 * codes without pulling in the full concept/implementation chain.
 *
 * Depends only on standard library headers and tl::expected.
 */

#include <cstdint>
#include <string_view>

#include <tl/expected.hpp>
#include <wise_enum/wise_enum.h>

namespace scf_5g_fapi
{

// ---------------------------------------------------------------------------
// Shared error type for slot processing
// ---------------------------------------------------------------------------

/**
 * Error returned by DLSlotProcessor::process() and ULSlotProcessor::process().
 */
struct SlotParseError
{
    enum class Code : uint8_t
    {
        NullBuffer,       //!< msg.msg_buf is null.
        PduCountMismatch, //!< sum(nPDUsOfEachType) != num_pdus in message header.
        PerTypeMismatch,  //!< per-type dispatch count differs from nPDUsOfEachType declaration.
        BfParamsInvalid,  //!< check_bf_pc_params() rejected numPRGs/digBFInterfaces.
    };

    Code     code{};   //!< Error code identifying the failure reason.
    uint16_t sfn{};    //!< System Frame Number at the point of failure.
    uint16_t slot{};   //!< Slot number at the point of failure.
};

/**
 * Return a human-readable name for a SlotParseError::Code value.
 *
 * @param[in] code  Error code to describe.
 * @return          NUL-terminated string literal identifying the code.
 *                  Return value must be checked.
 */
[[nodiscard]] constexpr std::string_view to_string(SlotParseError::Code code) noexcept
{
    switch (code)
    {
        case SlotParseError::Code::NullBuffer:       return "NullBuffer";
        case SlotParseError::Code::PduCountMismatch: return "PduCountMismatch";
        case SlotParseError::Code::PerTypeMismatch:  return "PerTypeMismatch";
        case SlotParseError::Code::BfParamsInvalid:  return "BfParamsInvalid";
        default:                                      return "Unknown";
    }
}

/** Slot processing result: success or SlotParseError. */
using SlotParseResult = tl::expected<void, SlotParseError>;

// ---------------------------------------------------------------------------
// Message batch parser error type
// ---------------------------------------------------------------------------

/**
 * Error returned by IScfFapiMessageParser::parse_messages().
 */
struct FapiMessageParseError
{
    enum class Code : uint8_t
    {
        NullBatch,            //!< @c msgs is null.
        ZeroCount,            //!< @c count is zero.
        InconsistentMsgId,    //!< Entries disagree on @c msg_id.
        UnsupportedMessageId, //!< FAPI message id is not handled (reserved).
        NullMessageBuffer,    //!< A message has null @c msg_buf.
    };

    Code     code{};             //!< Failure reason.
    int32_t  reference_msg_id{}; //!< First @c msg_id when @c InconsistentMsgId.
};

/** Message batch parse result: success or FapiMessageParseError. */
using MessageBatchParseResult = tl::expected<void, FapiMessageParseError>;

// ---------------------------------------------------------------------------
// UE-group setup error type (PDSCH PDU parser)
// ---------------------------------------------------------------------------

/**
 * Error code returned by @c PdschPduParser::setup_ue_group(). Each enumerator
 * marks a specific early-return guard inside the function. Defined via
 * @c WISE_ENUM_CLASS so that callers can recover the enumerator name with
 * @c wise_enum::to_string for diagnostics, without maintaining a parallel
 * stringification table.
 */
WISE_ENUM_CLASS((UeGroupErrorCode, std::uint8_t),
    NoCells,            //!< params.cell_grp_info.nCells == 0 at entry.
    UeGroupsFull,       //!< nUeGrps would exceed MAX_PDSCH_UE_GROUPS.
    UeGroupAtCapacity); //!< ue_grp.nUes would exceed MAX_PDSCH_UE_PER_TTI.

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_ERRORS_HPP_INCLUDED_
