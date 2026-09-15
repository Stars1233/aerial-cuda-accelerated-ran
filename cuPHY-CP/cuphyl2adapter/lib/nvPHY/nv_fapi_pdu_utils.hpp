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

#if !defined(NV_FAPI_PDU_UTILS_HPP_)
#define NV_FAPI_PDU_UTILS_HPP_

#include <cstddef>
#include <cstdint>
#include "aerial/casts/casts.hpp"
#include "nv_fapi_pdu_stride.hpp"
#include "nv_fapi_tti_pdu_counts.hpp"
#include "scf_5g_fapi.h"
#include "nvlog.hpp"
#include "nv_fapi_message_storage.hpp"  // phy_mac_msg_desc (full definition)

namespace nv {

// nv::next_pdu (the generic-PDU stride helper) lives in nv_fapi_pdu_stride.hpp,
// shared with the sidecar walk and the SCF FAPI parser helpers.

/// Body bytes a TTI request must have before its PDU counts can be read.
///
/// Uses @c offsetof(ReqT, payload) so the same bound works for 10.02 (no
/// @c nPDUsOfEachType[]) and 10.04. Prefer this over sizing to the per-type
/// counter array, which is 10.04-only on the wire.
template <typename ReqT>
inline constexpr std::size_t k_tti_min_body = offsetof(ReqT, payload);

/// Named view of @c NvDlTtiPduCounts for dispatch helpers.
// Wire-presence helpers over the version-neutral PDU-count sidecar
// (NvDl/UlTtiPduCounts + its named accessors). Each takes a fully materialized
// @p req -- the 10.02 walk is unbounded (no msg_len), so do not pass a truncated
// in-memory view -- and reports whether a channel is present.

[[nodiscard]] inline bool has_csi_rs(const scf_fapi_dl_tti_req_t& req) noexcept
{
    return make_dl_tti_pdu_counts(req).csi_rs() != 0u;
}

[[nodiscard]] inline bool has_pdsch(const scf_fapi_dl_tti_req_t& req) noexcept
{
    return make_dl_tti_pdu_counts(req).pdsch() != 0u;
}

[[nodiscard]] inline bool has_prach(const scf_fapi_ul_tti_req_t& req) noexcept
{
    return make_ul_tti_pdu_counts(req).prach() != 0u;
}

[[nodiscard]] inline bool has_pusch(const scf_fapi_ul_tti_req_t& req) noexcept
{
    return make_ul_tti_pdu_counts(req).pusch() != 0u;
}

[[nodiscard]] inline bool has_pucch(const scf_fapi_ul_tti_req_t& req) noexcept
{
    return make_ul_tti_pdu_counts(req).pucch() != 0u;
}

/// Walk all PDUs in a FAPI request payload with per-PDU debug logging and
/// two-tier safety guards (size validation + next_pdu null-check).
///
/// The boilerplate that was previously copy-pasted into each of the 8 DL/UL
/// channel handlers is centralised here so a fix or improvement only needs
/// to be made in one place.
///
/// @tparam tag      NVLOG tag of the calling translation unit. Must be a
///                  compile-time constant (e.g. the #define TAG macro value)
///                  because NVLOG_FMT enforces this via static_assert.
/// @param cell_id   Cell ID used in error-log context.
/// @param first_pdu Pointer to the first PDU in the payload (cast from
///                  req.payload). May be nullptr — the loop will not execute.
/// @param n_pdus    Number of PDUs declared in req.num_pdus.
/// @param fn        Callable invoked for each PDU after the size-guard step,
///                  so pdu.pdu_size is guaranteed valid on entry. PDUs whose
///                  pdu_size is 0 or below sizeof(scf_fapi_generic_pdu_info_t)
///                  are logged and skipped — fn is not called for them.
///                  Signature: void fn(uint16_t p,
///                                    const scf_fapi_generic_pdu_info_t& pdu)
///
/// @note **Last-PDU advance**: next_pdu() is called even on the final
///       iteration (p == n_pdus - 1).  A nullptr return on the last PDU is
///       expected (there is no next PDU) and is silently ignored.  The error
///       log only fires on a mid-walk truncation (p < n_pdus - 1).
///
/// The walk stops early when:
///   - pdu becomes nullptr at the top of an iteration
///   - pdu->pdu_size is 0 or < sizeof(scf_fapi_generic_pdu_info_t) (logged as error)
///   - next_pdu() returns nullptr mid-walk (p < n_pdus - 1; logged as error)
template <uint32_t tag, typename Fn>
void for_each_pdu(
    uint32_t                           cell_id,
    const scf_fapi_generic_pdu_info_t* first_pdu,
    uint16_t                           n_pdus,
    Fn&&                               fn)
{
    const scf_fapi_generic_pdu_info_t* pdu = first_pdu;
    for (uint16_t p = 0; p < n_pdus; ++p) {
        if (pdu == nullptr) { break; }
        NVLOGD_FMT(tag, "    pdu[{}]: pdu_type={} pdu_size={}",
                   p, pdu->pdu_type, pdu->pdu_size);
        if (pdu->pdu_size == 0 ||
            pdu->pdu_size < static_cast<uint16_t>(sizeof(scf_fapi_generic_pdu_info_t))) {
            NVLOGE_FMT(tag, AERIAL_L2ADAPTER_EVENT,
                       "    pdu[{}]: invalid pdu_size={} cell_id={} pdu_type={} — truncating walk",
                       p, pdu->pdu_size, cell_id, pdu->pdu_type);
            break;
        }
        fn(p, *pdu);
        const uint16_t cur_type = pdu->pdu_type;
        const uint16_t cur_size = pdu->pdu_size;
        pdu = next_pdu(pdu);
        if ((pdu == nullptr) && ((p + 1) < n_pdus)) {
            NVLOGE_FMT(tag, AERIAL_L2ADAPTER_EVENT,
                       "    pdu[{}]: next_pdu failed cell_id={} pdu_type={} pdu_size={} — truncating walk",
                       p, cell_id, cur_type, cur_size);
            break;
        }
    }
}

/// Iterate a TTI message lane from the ring slot, skipping cells rejected by
/// @p keep, and invoke @p fn for each qualifying cell.
///
/// Unified template for both DL_TTI (ReqT = scf_fapi_dl_tti_req_t) and
/// UL_TTI (ReqT = scf_fapi_ul_tti_req_t).
///
/// @tparam tag            NVLOG tag of the calling translation unit. Must be a
///                        compile-time constant because NVLOG_FMT enforces this
///                        via static_assert.
/// @tparam ReqT           FAPI request type (DL or UL TTI).
/// @param msgs            Pointer to the message descriptor array
///                        (dl_tti_messages() or ul_tti_messages()). If nullptr
///                        and n > 0, an error is logged and the function returns
///                        immediately (storage corruption). Each entry whose
///                        msg_len is too small to hold the request body is
///                        logged and skipped.
/// @param n               Message count (dl_tti_count() or ul_tti_count()).
/// @param keep            Channel-presence predicate: bool(uint16_t i, const ReqT&).
///                        Prefer lane-index sidecars on the store-replay path;
///                        wire helpers (has_csi_rs / has_pusch) remain for tests.
/// @param slot_u32        Slot identifier for error-log context.
/// @param ring_idx        Ring buffer index for error-log context.
/// @param fn              Callable:
///                        (uint16_t i, const phy_mac_msg_desc&, const ReqT&)
template <uint32_t tag, typename ReqT, typename Keep, typename Fn>
void for_each_tti_msg(
    const phy_mac_msg_desc* msgs,
    uint16_t                n,
    Keep&&                  keep,
    uint32_t                slot_u32,
    uint32_t                ring_idx,
    Fn&&                    fn)
{
    if (msgs == nullptr) {
        if (n > 0) {
            NVLOGE_FMT(tag, AERIAL_L2ADAPTER_EVENT,
                       "for_each_tti_msg: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} — skipping loop",
                       slot_u32, ring_idx, n);
        }
        return;
    }
    // Minimum body bytes past the FAPI header needed to safely read PDU counts
    // (10.02 payload walk / 10.04 nPDUsOfEachType[] via make_*_tti_pdu_counts).
    static constexpr size_t MIN_TTI_PAYLOAD = k_tti_min_body<ReqT>;

    for (uint16_t i = 0; i < n; ++i) {
        if (msgs[i].msg_buf == nullptr) {
            NVLOGE_FMT(tag, AERIAL_L2ADAPTER_EVENT,
                       "for_each_tti_msg: slot=0x{:08X} ring_idx={} msgs[{}].msg_buf=nullptr — skipping entry",
                       slot_u32, ring_idx, i);
            continue;
        }
        // msg_len is int32_t; reject negative values before the size_t comparison
        // to silence -Wsign-compare and to avoid a malformed-input footgun.
        if (msgs[i].msg_len < 0 ||
            static_cast<size_t>(msgs[i].msg_len) < sizeof(scf_fapi_header_t) + MIN_TTI_PAYLOAD) {
            NVLOGE_FMT(tag, AERIAL_L2ADAPTER_EVENT,
                       "for_each_tti_msg: slot=0x{:08X} ring_idx={} msgs[{}] msg_len={} too small "
                       "(need {}+{}={}) — skipping entry",
                       slot_u32, ring_idx, i, msgs[i].msg_len,
                       sizeof(scf_fapi_header_t), MIN_TTI_PAYLOAD,
                       sizeof(scf_fapi_header_t) + MIN_TTI_PAYLOAD);
            continue;
        }
        const auto* req = reinterpret_cast<const ReqT*>(
            static_cast<const char*>(msgs[i].msg_buf) + sizeof(scf_fapi_header_t));
        if (!keep(i, *req)) { continue; }
        fn(i, msgs[i], *req);
    }
}

} // namespace nv

#endif // NV_FAPI_PDU_UTILS_HPP_
