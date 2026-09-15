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

#if !defined(SCF_5G_FAPI_PARSER_HELPERS_HPP_INCLUDED_)
#define SCF_5G_FAPI_PARSER_HELPERS_HPP_INCLUDED_

#include <cstddef>
#include <cstdint>

#include "aerial/casts/casts.hpp"
#include "nv_fapi_pdu_stride.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi::detail {

/**
 * Extract a typed pointer to a FAPI message body from an IPC message descriptor.
 *
 * Skips the fixed @c scf_fapi_header_t prefix and casts the remaining payload
 * to @p T.  Shared by DL and UL slot processors to avoid duplicating the same
 * header-skip arithmetic in each class.
 *
 * @tparam T   FAPI request body type (e.g. @c scf_fapi_dl_tti_req_t,
 *             @c scf_fapi_ul_tti_req_t).
 * @param[in]  msg  IPC message descriptor to inspect.
 * @return     Pointer to the FAPI request body, or @c nullptr if
 *             @c msg.msg_buf is null or @c msg.msg_len is too small to hold
 *             @c scf_fapi_header_t + @p T.
 *             Return value must be checked.
 */
template<typename T>
[[nodiscard]] inline const T* extract_req(const nv::phy_mac_msg_desc& msg) noexcept
{
    constexpr auto k_min_len =
        static_cast<int32_t>(sizeof(scf_fapi_header_t) + sizeof(T));
    if (!msg.msg_buf || msg.msg_len < k_min_len) [[unlikely]] { return nullptr; }
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(msg.msg_buf) + sizeof(scf_fapi_header_t));
}

/// Compute ceil(x / y) — the smallest integer not less than the quotient.
/// Returns 0 if either input is 0 (avoids divide-by-zero and pre-empts the
/// (0 + y - 1) / y spurious-result corner). Used for PRG-count math; pure
/// integer arithmetic, safe for slot hot-path use.
///
/// Despite the name, this does NOT "round @p x up to the nearest multiple of
/// @p y" (that would be `((x + y - 1) / y) * y`). Name kept for continuity
/// with prior call sites; semantic is ceiling-divide.
[[nodiscard]] inline uint16_t round_up_u16(uint16_t x, uint16_t y) noexcept
{
    if (x == 0u || y == 0u)
    {
        return 0u;
    }
    return static_cast<uint16_t>((x + y - 1u) / y);
}

// The generic-PDU stride helper is nv::next_pdu (nv_fapi_pdu_stride.hpp), shared
// with the nvPHY sidecar walk so the pdu_size-advance contract has one owner.

/// End of the validated message body, used to bound a 10.02 PDU walk. Returns
/// null when @c msg_len is non-positive (length unknown -- traversal falls back
/// to @c num_pdus).
[[nodiscard]] inline const void* tti_payload_end(const nv::phy_mac_msg_desc& msg) noexcept
{
    if (msg.msg_buf == nullptr || msg.msg_len <= 0) { return nullptr; }
    return static_cast<const uint8_t*>(msg.msg_buf) + static_cast<std::size_t>(msg.msg_len);
}

/// Count PDUs of @p pdu_type in a UL_TTI.req payload (10.02 has no per-type wire array).
/// @p payload_end bounds the walk to the validated message body (see
/// @c tti_payload_end); when null the length was unknown and traversal falls
/// back to @c req.num_pdus alone.
[[nodiscard]] inline uint16_t count_ul_tti_pdus(const scf_fapi_ul_tti_req_t& req,
                                                uint16_t                     pdu_type,
                                                const void* payload_end = nullptr) noexcept
{
    uint16_t n = 0;
    const auto* pdu =
        aerial::casts::assume_cast<const scf_fapi_generic_pdu_info_t>(req.payload);
    const auto end = reinterpret_cast<uintptr_t>(payload_end);
    for (uint16_t i = 0; i < req.num_pdus && pdu != nullptr; ++i)
    {
        if (end != 0)
        {
            const auto pdu_base = reinterpret_cast<uintptr_t>(pdu);
            // Header must fit before we may read pdu_size.
            if (pdu_base + sizeof(scf_fapi_generic_pdu_info_t) > end) { break; }
            // Whole PDU body must fit too: a truncated final PDU whose header is
            // in-bounds but whose pdu_size crosses the body end is not present.
            if (pdu->pdu_size < sizeof(scf_fapi_generic_pdu_info_t) ||
                pdu_base + pdu->pdu_size > end)
            {
                break;
            }
        }
        if (pdu->pdu_type == pdu_type) { ++n; }
        pdu = nv::next_pdu(pdu);
    }
    return n;
}

/// Count PDUs of @p pdu_type in a DL_TTI.req payload (10.02 has no per-type wire array).
/// @p payload_end bounds the walk to the validated message body (see
/// @c tti_payload_end); when null the length was unknown and traversal falls
/// back to @c req.num_pdus alone.
[[nodiscard]] inline uint16_t count_dl_tti_pdus(const scf_fapi_dl_tti_req_t& req,
                                                uint16_t                     pdu_type,
                                                const void* payload_end = nullptr) noexcept
{
    uint16_t n = 0;
    const auto* pdu =
        aerial::casts::assume_cast<const scf_fapi_generic_pdu_info_t>(req.payload);
    const auto end = reinterpret_cast<uintptr_t>(payload_end);
    for (uint16_t i = 0; i < req.num_pdus && pdu != nullptr; ++i)
    {
        if (end != 0)
        {
            const auto pdu_base = reinterpret_cast<uintptr_t>(pdu);
            if (pdu_base + sizeof(scf_fapi_generic_pdu_info_t) > end) { break; }
            if (pdu->pdu_size < sizeof(scf_fapi_generic_pdu_info_t) ||
                pdu_base + pdu->pdu_size > end)
            {
                break;
            }
        }
        if (pdu->pdu_type == pdu_type) { ++n; }
        pdu = nv::next_pdu(pdu);
    }
    return n;
}

/// True iff a bounded 10.02 DL_TTI payload contains at least one PDU of @p pdu_type.
/// Non-10.04 equivalent of @c nPDUsOfEachType[pdu_type] != 0.
[[nodiscard]] inline bool has_dl_tti_pdu(const scf_fapi_dl_tti_req_t& req,
                                          uint16_t                     pdu_type,
                                          const void* payload_end = nullptr) noexcept
{
    return count_dl_tti_pdus(req, pdu_type, payload_end) != 0u;
}

} // namespace scf_5g_fapi::detail

#endif // SCF_5G_FAPI_PARSER_HELPERS_HPP_INCLUDED_
