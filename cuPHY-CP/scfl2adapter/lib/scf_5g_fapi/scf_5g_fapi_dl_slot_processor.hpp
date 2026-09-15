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

#if !defined(SCF_5G_FAPI_DL_SLOT_PROCESSOR_HPP_INCLUDED_)
#define SCF_5G_FAPI_DL_SLOT_PROCESSOR_HPP_INCLUDED_

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <type_traits>

#include <gsl-lite/gsl-lite.hpp>

#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi_dl_stats.hpp"
#include "scf_5g_fapi_dl_pdu_parsers.hpp"
#include "scf_5g_fapi_parser_helpers.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi_tti_dispatch.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"
#include "nvlog.h"

namespace scf_5g_fapi
{

namespace detail {

#ifdef SCF_FAPI_10_04
    /**
     * Map a DL TTI PDU type enum value to its nPDUsOfEachType[] array index.
     *
     * The SCF FAPI 10.04 array indices (DL_TTI_NPDUS_IDX_*) are distinct from
     * the per-PDU wire pdu_type values (scf_fapi_dl_tti_pdu_type_t); this helper
     * provides the mapping so callers can use named index constants instead of
     * casting the enum directly.
     *
     * @param[in] t  DL TTI PDU type.
     * @return       Corresponding DL_TTI_NPDUS_IDX_* value.
     *               Return value must be checked.
     */
    [[nodiscard]] inline constexpr uint16_t dl_npdus_idx(scf_fapi_dl_tti_pdu_type_t t) noexcept
    {
        switch (t)
        {
            case DL_TTI_PDU_TYPE_PDCCH:  return DL_TTI_NPDUS_IDX_PDCCH;
            case DL_TTI_PDU_TYPE_PDSCH:  return DL_TTI_NPDUS_IDX_PDSCH;
            case DL_TTI_PDU_TYPE_CSI_RS: return DL_TTI_NPDUS_IDX_CSI_RS;
            case DL_TTI_PDU_TYPE_SSB:    return DL_TTI_NPDUS_IDX_SSB;
            default:                      return 0u;
        }
    }
#endif // SCF_FAPI_10_04

    // PDCCH composition of a DL_TTI payload, determined in a single buffer scan.
    //   none        — no (reachable) PDCCH PDU before the buffer ends/malforms
    //   pdcch_only  — every one of num_pdus PDUs is a valid PDCCH PDU
    //   mixed       — at least one PDCCH PDU, but not pdcch_only (a non-PDCCH PDU
    //                 is present, or the buffer malforms after the first PDCCH)
    enum class pdcch_presence : uint8_t { none, mixed, pdcch_only };

    // Single-pass replacement for the old has_pdcch + is_pdcch_only pair, which
    // walked the whole PDU header chain twice per DL_TTI message on the hot path.
    // Semantics are preserved exactly:
    //   has_pdcch      <=> result != none
    //   is_pdcch_only  <=> result == pdcch_only
    [[nodiscard]] inline pdcch_presence
    dl_tti_pdcch_presence(std::span<const uint8_t> payload,
                          const uint16_t num_pdus) noexcept
    {
        if (num_pdus == 0u) { return pdcch_presence::none; }

        constexpr auto pdu_header_size = sizeof(scf_fapi_generic_pdu_info_t);
        const auto* cur = payload.data();
        const auto* const end = cur + payload.size();
        bool has_pdcch = false;
        bool has_other = false;
        uint16_t scanned = 0u;
        for (; scanned < num_pdus; ++scanned)
        {
            if (static_cast<std::size_t>(end - cur) < pdu_header_size) {
                break;
            }

            const auto* hdr = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(cur);
            if (hdr->pdu_size < pdu_header_size
                || static_cast<std::size_t>(end - cur) < hdr->pdu_size) {
                break;
            }

            if (hdr->pdu_type == DL_TTI_PDU_TYPE_PDCCH) {
                has_pdcch = true;
            } else {
                has_other = true;
            }

            // A PDCCH alongside any other PDU type is conclusively mixed; stop early.
            if (has_pdcch && has_other) {
                return pdcch_presence::mixed;
            }

            cur += hdr->pdu_size;
        }

        if (!has_pdcch) { return pdcch_presence::none; }
        // has_pdcch && !has_other here: pdcch_only iff the full chain validated;
        // a short/malformed PDU after the PDCCH leaves it mixed (matches the old
        // is_pdcch_only returning false on malformed input).
        return (scanned == num_pdus) ? pdcch_presence::pdcch_only
                                     : pdcch_presence::mixed;
    }
} // namespace detail

// ---------------------------------------------------------------------------
// DLSlotProcessor
// ---------------------------------------------------------------------------

/**
 * Processes a batch of DL_TTI.request IPC messages for one slot.
 *
 * Only PDUs whose type is listed in the ActiveTypes template parameter pack
 * are dispatched; all other PDU types in the message are silently skipped.
 *
 * With SCF_FAPI_10_04:
 *   1. Skips messages where none of the active types have any PDUs.
 *   2. Validates that sum(nPDUsOfEachType[0..3]) == num_pdus  (pre-check).
 *   3. Dispatches active-type PDUs via TtiDispatch::dispatch_pdus_typed.
 *   4. Verifies that per-active-type dispatch counts match nPDUsOfEachType  (post-check).
 *
 * Without SCF_FAPI_10_04:
 *   1. Dispatches active-type PDUs via TtiDispatch::dispatch_pdus_typed
 *      (type filtering at dispatch_one level; no early-skip or count validation).
 *
 * Returns success (empty @c SlotParseResult) on success, or @c tl::unexpected with
 * @c SlotParseError on the first failure (null buffer, PDU count mismatch, or per-type mismatch).
 *
 * @tparam V  Type satisfying DlModuleView (e.g. PhyModuleView).
 */
template<DlModuleView V>
class DLSlotProcessor final
{
    using Dispatch = TtiDispatch<scf_fapi_dl_tti_pdu_type_t,
                                  PdcchPduParser<V>,
                                  PdschPduParser<V>,
                                  CsiRsPduParser<V>,
                                  SsbPduParser<V>>;
public:
    explicit DLSlotProcessor(V& view) noexcept
        : view_{&view}
        , dispatch_{PdcchPduParser<V>{view},
                    PdschPduParser<V>{view},
                    CsiRsPduParser<V>{view},
                    SsbPduParser<V>{view}}
    {}

    /**
     * Process a batch of DL_TTI.request messages, dispatching only PDUs whose
     * type appears in the ActiveTypes template parameter pack.
     *
     * @tparam ActiveTypes  One or more DL TTI PDU type enum values to filter and process;
     *                      PDUs of all other types are silently skipped.
     * @param[in] msgs   Contiguous sequence of message descriptors to process
     *                   (@c std::span<const nv::phy_mac_msg_desc>); length is @c msgs.size().
     * @return           Success (empty @c SlotParseResult), or
     *                   @c tl::unexpected(@c SlotParseError) on the first failure.
     *                   Return value must be checked.
     */
    template<scf_fapi_dl_tti_pdu_type_t... ActiveTypes>
    [[nodiscard]] SlotParseResult
    process(std::span<const nv::phy_mac_msg_desc> msgs) noexcept
    {
        if constexpr (((DL_TTI_PDU_TYPE_PDSCH == ActiveTypes) || ...))
        {
            DlPdschStatsBatch stats_batch{};
            return process_impl<ActiveTypes...>(msgs, &stats_batch);
        }
        else
        {
            return process_impl<ActiveTypes...>(msgs, nullptr);
        }
    }

private:
    template<scf_fapi_dl_tti_pdu_type_t... ActiveTypes>
    [[nodiscard]] SlotParseResult
    process_impl(std::span<const nv::phy_mac_msg_desc> msgs,
                 DlPdschStatsBatch* stats_batch) noexcept
    {
        static constexpr bool k_processes_pdsch =
            ((DL_TTI_PDU_TYPE_PDSCH == ActiveTypes) || ...);

        auto publish_pdsch_stats = [&]() noexcept {
            if constexpr (k_processes_pdsch) {
                auto& pdsch_parser = dispatch_.template get_parser<PdschPduParser<V>>();
                pdsch_parser.end_stats_message();
                pdsch_parser.set_stats_batch(nullptr);
                view_->publish_dl_pdsch_stats(*stats_batch);
            }
        };

        auto finish_with_error = [&](SlotParseError error) noexcept -> SlotParseResult {
            publish_pdsch_stats();
            return tl::unexpected(error);
        };

        if constexpr (k_processes_pdsch) {
            dispatch_.template get_parser<PdschPduParser<V>>().set_stats_batch(stats_batch);
        }

        for (const auto& msg : msgs)
        {
            const auto* req = detail::extract_req<scf_fapi_dl_tti_req_t>(msg);
            if (!req)
            {
                return finish_with_error(
                    SlotParseError{SlotParseError::Code::NullBuffer, 0u, 0u});
            }

#ifdef SCF_FAPI_10_04
            // Skip message if none of the active types have any PDUs.
            if (((req->nPDUsOfEachType[detail::dl_npdus_idx(ActiveTypes)] == 0u) && ...))
            {
                continue;
            }

            if (!validate_sum(req))
            {
                return finish_with_error(
                    SlotParseError{SlotParseError::Code::PduCountMismatch, req->sfn, req->slot});
            }
#endif

            bool has_pdsch_for_stats = false;
            if constexpr (((DL_TTI_PDU_TYPE_PDSCH == ActiveTypes) || ...))
            {
#ifdef SCF_FAPI_10_04
                if (req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] != 0u)
                {
                    has_pdsch_for_stats = true;
                    dispatch_.template get_parser<PdschPduParser<V>>().setup_cell(
                        *req, static_cast<uint32_t>(msg.cell_id));
                }
#else
                if(detail::has_dl_tti_pdu(*req,
                                           DL_TTI_PDU_TYPE_PDSCH,
                                           detail::tti_payload_end(msg)))
                {
                    has_pdsch_for_stats = true;
                    dispatch_.template get_parser<PdschPduParser<V>>().setup_cell(
                        *req, static_cast<uint32_t>(msg.cell_id));
                }
#endif
            }

            if constexpr (((DL_TTI_PDU_TYPE_SSB == ActiveTypes) || ...))
            {
#ifdef SCF_FAPI_10_04
                if (req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_SSB] != 0u)
                {
                    dispatch_.template get_parser<SsbPduParser<V>>()
                        .setup_cell(msg.cell_id);
                }
#else
                dispatch_.template get_parser<SsbPduParser<V>>()
                    .setup_cell(msg.cell_id);
#endif
            }

            const auto* payload_start = req->payload;
            if (payload_start == nullptr) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "DL slot processor: sfn={} slot={} payload=null — skipping",
                           req->sfn, req->slot);
                continue;
            }
            const auto* payload_end   =
                static_cast<const uint8_t*>(msg.msg_buf) + msg.msg_len;
            if (payload_end <= payload_start) [[unlikely]] {
                NVLOGW_FMT(detail::k_tag,
                           "DL slot processor: sfn={} slot={} invalid payload bounds — "
                           "msg_buf={} msg_len={} payload_start={} payload_end={} — skipping",
                           req->sfn, req->slot,
                           msg.msg_buf, msg.msg_len,
                           static_cast<void*>(const_cast<uint8_t*>(payload_start)),
                           static_cast<void*>(const_cast<uint8_t*>(payload_end)));
                continue;
            }
            const auto  payload_bytes =
                static_cast<std::size_t>(payload_end - payload_start);
            const auto payload_span = std::span{payload_start, payload_bytes};

            const auto stats_cell_id = static_cast<std::uint32_t>(msg.cell_id);
            DlPdschCellDelta stats_cell_snapshot{};
            if constexpr (k_processes_pdsch) {
                if (has_pdsch_for_stats) {
                    stats_cell_snapshot = stats_batch->cell_delta(stats_cell_id);
                    dispatch_.template get_parser<PdschPduParser<V>>()
                        .begin_stats_message(stats_cell_id);
                }
            }

            // PDCCH parser setup needs per-message context before dispatch.
            // Today the temporary sym_prb_info bridge is enabled only on the
            // legacy path for PDCCH-only DL_TTI payloads.
            if constexpr (((DL_TTI_PDU_TYPE_PDCCH == ActiveTypes) || ...))
            {
                auto& parser = dispatch_.template get_parser<PdcchPduParser<V>>();
                const auto presence =
                    detail::dl_tti_pdcch_presence(payload_span, req->num_pdus);
                if (presence != detail::pdcch_presence::none) {
                    const bool enable_temporary_sym_prb_info =
                        !view_->is_fapi_to_cplane_direct_enabled()
                        && (presence == detail::pdcch_presence::pdcch_only);
                    uint8_t test_mode = 0u;
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
                    test_mode = req->testMode;
#endif
                    parser.setup_cell(static_cast<uint32_t>(msg.cell_id),
                                      test_mode,
                                      enable_temporary_sym_prb_info);
                }
            }

            auto result = dispatch_.template dispatch_pdus_typed<ActiveTypes...>(
                req->sfn, req->slot,
                payload_span,
                req->num_pdus);

            if constexpr (k_processes_pdsch) {
                if (has_pdsch_for_stats) {
                    constexpr auto k_pdsch_parser_idx =
                        Dispatch::template parser_index<PdschPduParser<V>>();
                    if (result.per_parser_counts[k_pdsch_parser_idx] != 0u) {
                        stats_batch->add_pdsch_slot(stats_cell_id);
                    }
                    dispatch_.template get_parser<PdschPduParser<V>>()
                        .end_stats_message();
                }
            }

#ifdef SCF_FAPI_10_04
            if (!verify_per_type_counts_typed<ActiveTypes...>(result, req))
            {
                if constexpr (k_processes_pdsch) {
                    if (has_pdsch_for_stats) {
                        stats_batch->set_cell_delta(stats_cell_id, stats_cell_snapshot);
                    }
                }
                return finish_with_error(
                    SlotParseError{SlotParseError::Code::PerTypeMismatch, req->sfn, req->slot});
            }
#endif
        }
        publish_pdsch_stats();
        return {};
    }

    gsl_lite::not_null<V*> view_;
    Dispatch dispatch_;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------


#ifdef SCF_FAPI_10_04
    /**
     * Pre-check: verify that the declared per-type PDU counts sum to num_pdus.
     *
     * Indices 0–3 (PDCCH, PDSCH, CSI-RS, SSB) are summed; index 4 (DlDCIs)
     * is not a PDU-payload count and is excluded.
     *
     * Only available when SCF_FAPI_10_04 is defined (requires nPDUsOfEachType).
     *
     * @param[in] req  DL TTI request to validate.
     * @return         true if the sum matches num_pdus; false otherwise.
     *                 Return value must be checked.
     */
    [[nodiscard]] static bool validate_sum(const scf_fapi_dl_tti_req_t* req) noexcept
    {
        uint32_t total = 0;
        for (auto idx : {DL_TTI_NPDUS_IDX_PDCCH,
                         DL_TTI_NPDUS_IDX_PDSCH,
                         DL_TTI_NPDUS_IDX_CSI_RS,
                         DL_TTI_NPDUS_IDX_SSB})
        {
            total += req->nPDUsOfEachType[idx];
        }

        if (total != req->num_pdus)
        {
            NVLOGW_FMT(detail::k_tag,
                       "DL TTI sfn={} slot={}: nPDUsOfEachType sum={} != num_pdus={}",
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
     * @tparam ActiveTypes  DL TTI PDU type enum values that were dispatched.
     * @param[in] result    Dispatch result containing per-parser hit counts.
     * @param[in] req       DL TTI request containing the expected counts.
     * @return              true when all active-type counts match; false on first mismatch.
     *                      Return value must be checked.
     */
    template<scf_fapi_dl_tti_pdu_type_t... ActiveTypes>
    [[nodiscard]] static bool
    verify_per_type_counts_typed(const typename Dispatch::result_t& result,
                                  const scf_fapi_dl_tti_req_t* req) noexcept
    {
        bool err = false;

        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            (... || [&]
            {
                constexpr auto pt = Dispatch::template parser_pdu_type<Is>();
                // Only verify parsers in the active type set.
                if constexpr (!((pt == ActiveTypes) || ...)) { return false; }
                const uint16_t expected = req->nPDUsOfEachType[detail::dl_npdus_idx(pt)];
                const uint16_t actual   = result.per_parser_counts[Is];
                if (expected != actual)
                {
                    NVLOGW_FMT(detail::k_tag,
                               "DL TTI sfn={} slot={} pdu_type={} expected={} got={}",
                               req->sfn, req->slot,
                               static_cast<std::underlying_type_t<scf_fapi_dl_tti_pdu_type_t>>(pt), expected, actual);
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

#endif // SCF_5G_FAPI_DL_SLOT_PROCESSOR_HPP_INCLUDED_
