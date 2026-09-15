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

#if !defined(SCF_5G_FAPI_UL_SLOT_PROCESSOR_HPP_INCLUDED_)
#define SCF_5G_FAPI_UL_SLOT_PROCESSOR_HPP_INCLUDED_

#include <cstdint>
#include <span>
#include <type_traits>

#include "scf_5g_fapi_parser_helpers.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi_tti_dispatch.hpp"
#include "scf_5g_fapi_ul_pdu_parsers.hpp"
#include "nv_fapi_tti_pdu_counts.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"
#include "nvlog.h"

namespace scf_5g_fapi
{

namespace detail {

#ifdef SCF_FAPI_10_04
    /**
     * Map a UL TTI PDU type enum value to its nPDUsOfEachType[] array index.
     *
     * The SCF FAPI 10.04 array indices (UL_TTI_NPDUS_IDX_*) are distinct from
     * the per-PDU wire pdu_type values (scf_fapi_ul_tti_pdu_type_t).  In
     * particular UL_TTI_PDU_TYPE_SRS == 3 but UL_TTI_NPDUS_IDX_SRS == 4, so
     * casting the enum directly to an array index gives the wrong slot for SRS.
     *
     * @param[in] t  UL TTI PDU type.
     * @return       Corresponding UL_TTI_NPDUS_IDX_* value.
     *               Return value must be checked.
     */
    [[nodiscard]] inline constexpr uint16_t ul_npdus_idx(scf_fapi_ul_tti_pdu_type_t t) noexcept
    {
        switch (t)
        {
            case UL_TTI_PDU_TYPE_PRACH: return UL_TTI_NPDUS_IDX_PRACH;
            case UL_TTI_PDU_TYPE_PUSCH: return UL_TTI_NPDUS_IDX_PUSCH;
            case UL_TTI_PDU_TYPE_PUCCH: return UL_TTI_NPDUS_IDX_PUCCH_F01;
            case UL_TTI_PDU_TYPE_SRS:   return UL_TTI_NPDUS_IDX_SRS;
            case UL_TTI_PDU_TYPE_PUCCH_2_3_4: return UL_TTI_NPDUS_IDX_PUCCH_F234;
            case UL_TTI_PDU_TYPE_MsgA_PUSCH:  return UL_TTI_NPDUS_IDX_MsgA_PUSCH;

            default:                     return 0u;
        }
    }

    /**
     * Expected dispatch count for an active UL TTI PDU type.
     *
     * PUCCH formats 0/1 and 2/3/4 share the UL_TTI_PDU_TYPE_PUCCH wire type and are
     * both parsed by PucchPduParser (parse_pucch_pdu_common switches on format_type),
     * so its expected count spans both nPDUsOfEachType lanes (F01 + F234). Every other
     * type maps to its single lane.
     *
     * @param[in] t    UL TTI PDU type.
     * @param[in] req  UL TTI request carrying nPDUsOfEachType.
     * @return         Expected number of PDUs of this type. Return value must be checked.
     */
    [[nodiscard]] inline uint16_t ul_npdus_expected_count(scf_fapi_ul_tti_pdu_type_t t,
                                                          const scf_fapi_ul_tti_req_t& req) noexcept
    {
        if (t == UL_TTI_PDU_TYPE_PUCCH)
        {
            return static_cast<uint16_t>(req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F01]
                                         + req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F234]);
        }
        return req.nPDUsOfEachType[ul_npdus_idx(t)];
    }
#endif // SCF_FAPI_10_04
} // namespace detail

// ---------------------------------------------------------------------------
// ULSlotProcessor
// ---------------------------------------------------------------------------

/**
 * Processes a batch of UL_TTI.request IPC messages for one slot.
 *
 * Only PDUs whose type is listed in the ActiveTypes template parameter pack
 * are dispatched; all other PDU types in the message are silently skipped.
 *
 * With SCF_FAPI_10_04:
 *   1. Skips messages where none of the active types have any PDUs.
 *   2. Validates that sum(nPDUsOfEachType[0..5]) == num_pdus  (pre-check).
 *   3. Dispatches active-type PDUs via TtiDispatch::dispatch_pdus_typed.
 *   4. Verifies that per-active-type dispatch counts match nPDUsOfEachType  (post-check).
 *
 * Without SCF_FAPI_10_04:
 *   1. Dispatches active-type PDUs via TtiDispatch::dispatch_pdus_typed
 *      (type filtering at dispatch_one level; no early-skip or count validation).
 *
 * Returns success (empty @c SlotParseResult) on success, or @c tl::unexpected with
 * @c SlotParseError on the first failure.
 *
 * @tparam V  Type satisfying UlModuleView (e.g. PhyModuleView).
 */
template<UlModuleView V>
class ULSlotProcessor final
{
#ifdef SCF_FAPI_10_04
    using Dispatch = TtiDispatch<scf_fapi_ul_tti_pdu_type_t,
                                  PrachPduParser<V>,
                                  PuschPduParser<V>,
                                  PucchPduParser<V>,
                                  Pucch234PduParser<V>,
                                  SrsPduParser<V>,
                                  MsgAPuschPduParser<V>>;
#else
    using Dispatch = TtiDispatch<scf_fapi_ul_tti_pdu_type_t,
                                  PrachPduParser<V>,
                                  PuschPduParser<V>,
                                  PucchPduParser<V>,
                                  Pucch234PduParser<V>,
                                  SrsPduParser<V>>;
#endif
public:
#ifdef SCF_FAPI_10_04
    explicit ULSlotProcessor(V& view) noexcept
        : dispatch_{PrachPduParser<V>{view},
                    PuschPduParser<V>{view},
                    PucchPduParser<V>{view},
                    Pucch234PduParser<V>{view},
                    SrsPduParser<V>{view},
                    MsgAPuschPduParser<V>{view}}
    {}
#else
    explicit ULSlotProcessor(V& view) noexcept
        : dispatch_{PrachPduParser<V>{view},
                    PuschPduParser<V>{view},
                    PucchPduParser<V>{view},
                    Pucch234PduParser<V>{view},
                    SrsPduParser<V>{view}}
    {}
#endif

    /**
     * Process a batch of UL_TTI.request messages, dispatching only PDUs whose
     * type appears in the ActiveTypes template parameter pack.
     *
     * @tparam ActiveTypes  One or more UL TTI PDU type enum values to filter and process;
     *                      PDUs of all other types are silently skipped.
     * @param[in] msgs   Contiguous sequence of message descriptors to process
     *                   (@c std::span<const nv::phy_mac_msg_desc>); length is @c msgs.size().
     * @param[in] ul_counts  Optional store-time PDU-count sidecars parallel to
     *                   @p msgs; when supplied, the SRS setup reuses the cached
     *                   count instead of re-walking the payload (10.02). Empty
     *                   (default) for serial/test callers with no sidecar.
     * @return           Success (empty @c SlotParseResult), or
     *                   @c tl::unexpected(@c SlotParseError) on the first failure.
     *                   Return value must be checked.
     */
    template<scf_fapi_ul_tti_pdu_type_t... ActiveTypes>
    [[nodiscard]] SlotParseResult
    process(std::span<const nv::phy_mac_msg_desc>                msgs,
            [[maybe_unused]] std::span<const nv::NvUlTtiPduCounts> ul_counts = {}) noexcept
    {
        for (const auto& msg : msgs)
        {
            const auto* req = detail::extract_req<scf_fapi_ul_tti_req_t>(msg);
            if (!req)
            {
                return tl::unexpected(SlotParseError{SlotParseError::Code::NullBuffer, 0u, 0u});
            }
            const uint16_t log_sfn = req->sfn;
            const uint16_t log_slot = req->slot;
            [[maybe_unused]] const uint16_t log_num_pdus = req->num_pdus;

#ifdef SCF_FAPI_10_04
            NVLOGD_FMT(detail::k_tag,
                       "ULSlotProcessor: cell_id={} sfn={} slot={} num_pdus={} msg_len={} "
                       "nPusch={} nPrach={} nPucchF01={} nPucchF234={} nSrs={} nMsgA={}",
                       msg.cell_id, log_sfn, log_slot,
                       log_num_pdus, msg.msg_len,
                       req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUSCH],
                       req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_PRACH],
                       req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F01],
                       req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F234],
                       req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_SRS],
                       req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_MsgA_PUSCH]);
            // Skip message if none of the active types have any PDUs.
            if (((detail::ul_npdus_expected_count(ActiveTypes, *req) == 0u) && ...))
            {
                continue;
            }

            if (!validate_sum(req))
            {
                return tl::unexpected(
                    SlotParseError{SlotParseError::Code::PduCountMismatch, req->sfn, req->slot});
            }
#endif

            // Bind per-type parsers to this message's logical carrier index
            // before dispatch. Mirrors the PDSCH path's setup_cell — the PDUs
            // themselves carry phys_cell_id (3GPP PCI 0-1007), not the logical
            // idx, so parsers cannot derive the cell index from the PDU alone.
            //
            // SRS state-reset is idempotent per (sfn, slot, cell_id); pre-10.04
            // has no per-type PDU count to gate on, so the call runs on every
            // UL slot. If the slot has no SRS PDU, dispatch_pdus_typed below
            // simply doesn't invoke SrsPduParser::parse and the reset state
            // goes unused. PRACH is gated only on 10.04 since older flows
            // didn't route PRACH through this path.
#ifdef SCF_FAPI_10_04
            if constexpr (((UL_TTI_PDU_TYPE_SRS == ActiveTypes) || ...))
            {
                if (req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_SRS] != 0u)
                {
                    dispatch_.template get_parser<SrsPduParser<V>>().setup_cell(*req, msg.cell_id);
                }
            }
            if constexpr (((UL_TTI_PDU_TYPE_PRACH == ActiveTypes) || ...))
            {
                if (req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_PRACH] != 0u)
                {
                    dispatch_.template get_parser<PrachPduParser<V>>().setup_cell(msg.cell_id);
                }
            }
            if constexpr (((UL_TTI_PDU_TYPE_PUCCH == ActiveTypes) || ...))
            {
                // PUCCH F0/1 and F2/3/4 both arrive as UL_TTI_PDU_TYPE_PUCCH and are
                // parsed by PucchPduParser, so gate setup on either lane being non-empty.
                if (detail::ul_npdus_expected_count(UL_TTI_PDU_TYPE_PUCCH, *req) != 0u)
                {
                    dispatch_.template get_parser<PucchPduParser<V>>().setup_cell(*req, msg.cell_id);
                }
            }
            if constexpr (((UL_TTI_PDU_TYPE_PUCCH_2_3_4 == ActiveTypes) || ...))
            {
                if (req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F234] != 0u)
                {
                    dispatch_.template get_parser<Pucch234PduParser<V>>().setup_cell(*req, msg.cell_id);
                }
            }
#else
            if constexpr (((UL_TTI_PDU_TYPE_SRS == ActiveTypes) || ...))
            {
                // Reuse the store-time sidecar SRS count when the driver supplies
                // it; -1 (serial/test callers with no sidecar) falls back to a walk.
                const std::size_t msg_idx = static_cast<std::size_t>(&msg - msgs.data());
                const int32_t     srs_from_sidecar =
                    (msg_idx < ul_counts.size())
                        ? static_cast<int32_t>(ul_counts[msg_idx].srs())
                        : -1;
                dispatch_.template get_parser<SrsPduParser<V>>().setup_cell(
                    *req, msg.cell_id, detail::tti_payload_end(msg), srs_from_sidecar);
            }
            if constexpr (((UL_TTI_PDU_TYPE_PRACH == ActiveTypes) || ...))
            {
                dispatch_.template get_parser<PrachPduParser<V>>().setup_cell(msg.cell_id);
            }
            if constexpr (((UL_TTI_PDU_TYPE_PUCCH == ActiveTypes) || ...))
            {
                dispatch_.template get_parser<PucchPduParser<V>>().setup_cell(*req, msg.cell_id);
            }
            if constexpr (((UL_TTI_PDU_TYPE_PUCCH_2_3_4 == ActiveTypes) || ...))
            {
                dispatch_.template get_parser<Pucch234PduParser<V>>().setup_cell(*req, msg.cell_id);
            }
#endif
            // Per-message cell setup: call setup_cell() on parsers that define it.
            // Independent `if constexpr` (not `else if`): SRS and PUSCH setup are
            // not mutually exclusive.  A future process<...SRS, PUSCH...>
            // instantiation must run both setups; chaining this to the SRS branch
            // above would silently skip PUSCH setup_cell whenever SRS is also active.
            if constexpr (((UL_TTI_PDU_TYPE_PUSCH == ActiveTypes) || ...))
            {
#ifdef SCF_FAPI_10_04
                if (req->nPDUsOfEachType[detail::ul_npdus_idx(UL_TTI_PDU_TYPE_PUSCH)] != 0u)
                {
                    dispatch_.template get_parser<PuschPduParser<V>>().setup_cell(*req, msg.cell_id);
                }
#else
                dispatch_.template get_parser<PuschPduParser<V>>().setup_cell(*req, msg.cell_id);
#endif
            }

            const auto* payload_start = req->payload;
            if (payload_start == nullptr) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "UL slot processor: sfn={} slot={} payload=null — skipping",
                           req->sfn, req->slot);
                continue;
            }
            const auto* payload_end   =
                static_cast<const uint8_t*>(msg.msg_buf) + msg.msg_len;
            if (payload_end <= payload_start) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "UL slot processor: sfn={} slot={} invalid payload bounds — "
                           "msg_buf={} msg_len={} payload_start={} payload_end={} — skipping",
                           req->sfn, req->slot,
                           msg.msg_buf, msg.msg_len,
                           static_cast<void*>(const_cast<uint8_t*>(payload_start)),
                           static_cast<void*>(const_cast<uint8_t*>(payload_end)));
                continue;
            }
            const auto  payload_bytes =
                static_cast<std::size_t>(payload_end - payload_start);

            auto result = dispatch_.template dispatch_pdus_typed<ActiveTypes...>(
                req->sfn, req->slot,
                std::span{payload_start, payload_bytes},
                req->num_pdus);
            NVLOGD_FMT(detail::k_tag,
                       "ULSlotProcessor: cell_id={} sfn={} slot={} processed={} skipped_unknown={}",
                       msg.cell_id, log_sfn, log_slot, result.processed, result.skipped_unknown);

#ifdef SCF_FAPI_10_04
            if (!verify_per_type_counts_typed<ActiveTypes...>(result, req))
            {
                return tl::unexpected(
                    SlotParseError{SlotParseError::Code::PerTypeMismatch, req->sfn, req->slot});
            }
#endif
        }
        return {};
    }

private:
    Dispatch dispatch_;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------


#ifdef SCF_FAPI_10_04
    /**
     * Pre-check: verify that the sum of all 6 PDU-type counts equals num_pdus.
     *
     * Only available when SCF_FAPI_10_04 is defined (requires nPDUsOfEachType).
     *
     * @param[in] req  UL TTI request to validate.
     * @return         true if the sum matches num_pdus; false otherwise.
     *                 Return value must be checked.
     */
    [[nodiscard]] static bool validate_sum(const scf_fapi_ul_tti_req_t* req) noexcept
    {
        // Per SCF 222.10.04 Table 3-46, nPDUsOfEachType carries six UL TTI PDU
        // counts (PRACH, PUSCH, PUCCH_F01, PUCCH_F234, SRS, MsgA-PUSCH).  Any
        // trailing array entries are not summable against num_pdus, so bound
        // the accumulator to the spec-defined PDU-type count to avoid false
        // PerTypeMismatch on releases that grow the array.
        static constexpr uint16_t k_ul_npdus_summable = 6u;
        uint32_t total = 0;
        for (uint16_t idx = 0; idx < k_ul_npdus_summable; ++idx)
        {
            total += req->nPDUsOfEachType[idx];
        }

        if (total != req->num_pdus)
        {
            NVLOGW_FMT(detail::k_tag,
                       "UL TTI sfn={} slot={}: nPDUsOfEachType sum={} != num_pdus={}",
                       req->sfn, req->slot, total, req->num_pdus);
            return false;
        }
        return true;
    }

    /**
     * Post-check: verify per-active-type dispatch counts against nPDUsOfEachType.
     *
     * Only parsers whose pdu_type is in ActiveTypes are checked; others are
     * compile-time no-ops. Only available when SCF_FAPI_10_04 is defined.
     *
     * @tparam ActiveTypes  UL TTI PDU type enum values that were dispatched.
     * @param[in] result    Dispatch result containing per-parser hit counts.
     * @param[in] req       UL TTI request containing the expected counts.
     * @return              true when all active-type counts match; false on first mismatch.
     *                      Return value must be checked.
     */
    template<scf_fapi_ul_tti_pdu_type_t... ActiveTypes>
    [[nodiscard]] static bool
    verify_per_type_counts_typed(const typename Dispatch::result_t& result,
                                  const scf_fapi_ul_tti_req_t* req) noexcept
    {
        bool err = false;

        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            (... || [&]
            {
                constexpr auto pt = Dispatch::template parser_pdu_type<Is>();
                // Only verify parsers in the active type set.
                if constexpr (!((pt == ActiveTypes) || ...)) { return false; }
                const uint16_t expected = detail::ul_npdus_expected_count(pt, *req);
                const uint16_t actual   = result.per_parser_counts[Is];
                if (expected != actual)
                {
                    NVLOGW_FMT(detail::k_tag,
                               "UL TTI sfn={} slot={} pdu_type={} expected={} got={}",
                               req->sfn, req->slot,
                               static_cast<std::underlying_type_t<scf_fapi_ul_tti_pdu_type_t>>(pt), expected, actual);
                    err = true;
                    return true;  // stop fold on first mismatch
                }
                return false;
            }());
        }(std::make_index_sequence<Dispatch::parser_count>{});

        return !err;
    }
#endif // SCF_FAPI_10_04
};

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_UL_SLOT_PROCESSOR_HPP_INCLUDED_
