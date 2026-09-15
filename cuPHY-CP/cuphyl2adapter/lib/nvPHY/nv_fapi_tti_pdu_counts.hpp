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

/**
 * @file nv_fapi_tti_pdu_counts.hpp
 * @brief Version-neutral TTI PDU-count sidecars for FAPI store-replay.
 *
 * 10.04 carries @c nPDUsOfEachType[] on the wire; 10.02 does not. These helpers
 * normalize both into fixed @c by_type[] arrays so dispatch / telemetry can
 * read counts without re-walking the payload on the hot path. Indices mirror
 * the 10.04 layout but do not address wire memory on 10.02.
 */

#if !defined(NV_FAPI_TTI_PDU_COUNTS_HPP_)
#define NV_FAPI_TTI_PDU_COUNTS_HPP_

#include <cstddef>
#include <cstdint>

#include "aerial/casts/casts.hpp"
#include "nv_fapi_pdu_stride.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"

namespace nv {

// Sidecar indices mirror the 10.04 wire layout; they do not address wire memory.
#ifndef DL_TTI_NPDUS_IDX_PDCCH
#define DL_TTI_NPDUS_IDX_PDCCH  0
#define DL_TTI_NPDUS_IDX_PDSCH  1
#define DL_TTI_NPDUS_IDX_CSI_RS 2
#define DL_TTI_NPDUS_IDX_SSB    3
#define DL_TTI_NPDUS_IDX_DlDCIs 4
#endif

#ifndef UL_TTI_NPDUS_IDX_PRACH
#define UL_TTI_NPDUS_IDX_PRACH      0
#define UL_TTI_NPDUS_IDX_PUSCH      1
#define UL_TTI_NPDUS_IDX_PUCCH_F01  2
#define UL_TTI_NPDUS_IDX_PUCCH_F234 3
#define UL_TTI_NPDUS_IDX_SRS        4
#define UL_TTI_NPDUS_IDX_MsgA_PUSCH 5
#endif

inline constexpr std::size_t k_dl_tti_npdus_types = 5;
inline constexpr std::size_t k_ul_tti_npdus_types = 6;

/// Minimum body size past the FAPI header to reach @c payload (works on 10.02/10.04).
inline constexpr std::size_t k_dl_tti_min_body = offsetof(scf_fapi_dl_tti_req_t, payload);
inline constexpr std::size_t k_ul_tti_min_body = offsetof(scf_fapi_ul_tti_req_t, payload);

/// Cached per-type PDU counts for one DL_TTI.req (store-time sidecar entry).
struct NvDlTtiPduCounts
{
    uint16_t by_type[k_dl_tti_npdus_types]{};

    [[nodiscard]] bool any_channel() const noexcept
    {
        return (by_type[DL_TTI_NPDUS_IDX_PDCCH] | by_type[DL_TTI_NPDUS_IDX_PDSCH] |
                by_type[DL_TTI_NPDUS_IDX_CSI_RS] | by_type[DL_TTI_NPDUS_IDX_SSB]) != 0u;
    }

    // Named per-channel accessors over the by-index counters.
    [[nodiscard]] uint16_t pdcch()   const noexcept { return by_type[DL_TTI_NPDUS_IDX_PDCCH]; }
    [[nodiscard]] uint16_t pdsch()   const noexcept { return by_type[DL_TTI_NPDUS_IDX_PDSCH]; }
    [[nodiscard]] uint16_t csi_rs()  const noexcept { return by_type[DL_TTI_NPDUS_IDX_CSI_RS]; }
    [[nodiscard]] uint16_t ssb()     const noexcept { return by_type[DL_TTI_NPDUS_IDX_SSB]; }
    /// DCIs across all PDCCH PDUs; not signalled in 10.02.
    [[nodiscard]] uint16_t dl_dcis() const noexcept { return by_type[DL_TTI_NPDUS_IDX_DlDCIs]; }
};

/// Cached per-type PDU counts for one UL_TTI.req (store-time sidecar entry).
struct NvUlTtiPduCounts
{
    uint16_t by_type[k_ul_tti_npdus_types]{};

    [[nodiscard]] bool any_channel() const noexcept
    {
        return (by_type[UL_TTI_NPDUS_IDX_PRACH] | by_type[UL_TTI_NPDUS_IDX_PUSCH] |
                by_type[UL_TTI_NPDUS_IDX_PUCCH_F01] | by_type[UL_TTI_NPDUS_IDX_PUCCH_F234] |
                by_type[UL_TTI_NPDUS_IDX_SRS]) != 0u;
    }

    // Named per-channel accessors over the by-index counters.
    [[nodiscard]] uint16_t prach() const noexcept { return by_type[UL_TTI_NPDUS_IDX_PRACH]; }
    [[nodiscard]] uint16_t pusch() const noexcept { return by_type[UL_TTI_NPDUS_IDX_PUSCH]; }
    [[nodiscard]] uint16_t srs()   const noexcept { return by_type[UL_TTI_NPDUS_IDX_SRS]; }

    /// Combined F0/1 + F2/3/4 PUCCH count (matches legacy dispatch gating).
    [[nodiscard]] uint16_t pucch() const noexcept
    {
        return static_cast<uint16_t>(by_type[UL_TTI_NPDUS_IDX_PUCCH_F01] +
                                     by_type[UL_TTI_NPDUS_IDX_PUCCH_F234]);
    }
};

namespace detail {

/// Invoke @p bump with the pdu_type of every PDU in @p req.
/// 10.02 carries no per-type counters, so counts come from the payload itself.
/// @p payload_end bounds the walk to the validated message body (see
/// @c tti_payload_end); when null the transport reported no length and
/// traversal falls back to the @c num_pdus count alone.
template <typename ReqT, typename Fn>
void for_each_tti_pdu_type(const ReqT& req, const void* payload_end, Fn&& bump) noexcept
{
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
        bump(pdu->pdu_type);
        pdu = next_pdu(pdu);
    }
}

} // namespace detail

/// Build DL_TTI sidecar counts from a decoded request (wire or walked).
[[nodiscard]] inline NvDlTtiPduCounts make_dl_tti_pdu_counts(
    const scf_fapi_dl_tti_req_t& req, [[maybe_unused]] const void* payload_end = nullptr) noexcept
{
    NvDlTtiPduCounts counts{};
#ifdef SCF_FAPI_10_04
    for (std::size_t i = 0; i < k_dl_tti_npdus_types; ++i)
    {
        counts.by_type[i] = req.nPDUsOfEachType[i];
    }
#else
    detail::for_each_tti_pdu_type(req, payload_end, [&counts](uint16_t pdu_type) {
        switch (pdu_type)
        {
        case DL_TTI_PDU_TYPE_PDCCH:  ++counts.by_type[DL_TTI_NPDUS_IDX_PDCCH];  break;
        case DL_TTI_PDU_TYPE_PDSCH:  ++counts.by_type[DL_TTI_NPDUS_IDX_PDSCH];  break;
        case DL_TTI_PDU_TYPE_CSI_RS: ++counts.by_type[DL_TTI_NPDUS_IDX_CSI_RS]; break;
        case DL_TTI_PDU_TYPE_SSB:    ++counts.by_type[DL_TTI_NPDUS_IDX_SSB];    break;
        default: break;
        }
    });
#endif
    return counts;
}

/// Build UL_TTI sidecar counts from a decoded request (wire or walked).
[[nodiscard]] inline NvUlTtiPduCounts make_ul_tti_pdu_counts(
    const scf_fapi_ul_tti_req_t& req, [[maybe_unused]] const void* payload_end = nullptr) noexcept
{
    NvUlTtiPduCounts counts{};
#ifdef SCF_FAPI_10_04
    for (std::size_t i = 0; i < k_ul_tti_npdus_types; ++i)
    {
        counts.by_type[i] = req.nPDUsOfEachType[i];
    }
#else
    // Summary fields for PRACH/PUSCH/PUCCH; SRS needs a payload walk.
    if (req.rach_present > 0)
    {
        counts.by_type[UL_TTI_NPDUS_IDX_PRACH] = 1;
    }
    counts.by_type[UL_TTI_NPDUS_IDX_PUSCH]     = req.num_ulsch;
    counts.by_type[UL_TTI_NPDUS_IDX_PUCCH_F01] = req.num_ulcch;

    detail::for_each_tti_pdu_type(req, payload_end, [&counts](uint16_t pdu_type) {
        if (pdu_type == UL_TTI_PDU_TYPE_SRS)
        {
            ++counts.by_type[UL_TTI_NPDUS_IDX_SRS];
        }
    });
#endif
    return counts;
}

/// True when @p msg has a buffer large enough to hold at least @p min_body past
/// the FAPI header. Rejects a null buffer and any non-positive @c msg_len, so a
/// message the dispatch walk (for_each_tti_msg) would drop cannot populate a
/// non-zero sidecar -- store and dispatch stay consistent.
[[nodiscard]] inline bool msg_body_ok(const phy_mac_msg_desc& msg,
                                      std::size_t             min_body) noexcept
{
    if (msg.msg_buf == nullptr || msg.msg_len <= 0)
    {
        return false;
    }
    return static_cast<std::size_t>(msg.msg_len) >= sizeof(scf_fapi_header_t) + min_body;
}

/// End of the validated message body, used to bound the 10.02 PDU walk. Returns
/// null only defensively -- msg_body_ok already rejects msg_len <= 0 upstream, so
/// the store path always reaches this with a positive length.
[[nodiscard]] inline const void* tti_payload_end(const phy_mac_msg_desc& msg) noexcept
{
    if (msg.msg_buf == nullptr || msg.msg_len <= 0)
    {
        return nullptr;
    }
    return static_cast<const char*>(msg.msg_buf) + static_cast<std::size_t>(msg.msg_len);
}

/// Populate a DL_TTI sidecar entry from a stored message descriptor.
[[nodiscard]] inline NvDlTtiPduCounts make_dl_tti_pdu_counts_from_msg(
    const phy_mac_msg_desc& msg) noexcept
{
    if (!msg_body_ok(msg, k_dl_tti_min_body))
    {
        return {};
    }
    const auto* req = aerial::casts::assume_cast<const scf_fapi_dl_tti_req_t>(
        static_cast<const char*>(msg.msg_buf) + sizeof(scf_fapi_header_t));
    return make_dl_tti_pdu_counts(*req, tti_payload_end(msg));
}

/// Populate a UL_TTI sidecar entry from a stored message descriptor.
[[nodiscard]] inline NvUlTtiPduCounts make_ul_tti_pdu_counts_from_msg(
    const phy_mac_msg_desc& msg) noexcept
{
    if (!msg_body_ok(msg, k_ul_tti_min_body))
    {
        return {};
    }
    const auto* req = aerial::casts::assume_cast<const scf_fapi_ul_tti_req_t>(
        static_cast<const char*>(msg.msg_buf) + sizeof(scf_fapi_header_t));
    return make_ul_tti_pdu_counts(*req, tti_payload_end(msg));
}

} // namespace nv

#endif // NV_FAPI_TTI_PDU_COUNTS_HPP_
