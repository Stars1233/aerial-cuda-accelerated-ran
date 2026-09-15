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

#if !defined(SCF_5G_FAPI_PUCCH_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_PUCCH_PDU_PARSER_HPP_INCLUDED_

#include <cstddef>
#include <cstdint>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_pucch_pdu_parser_common.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi
{

/**
 * Parses a single PUCCH PDU from a UL_TTI.request message.
 *
 * Calls update_cell_command for PUCCH, forwarding group_command,
 * cell_sub_command, dtx_thresholds, config_options, mmimo_enabled, and
 * cell_view (cell_params.nPrbUlBwp, slot_detail) from the module view.
 *
 * @tparam V  Type satisfying UlModuleView.
 */
template<UlModuleView V>
class PucchPduParser final
{
public:
    using pdu_t = scf_fapi_pucch_pdu_t;
    static constexpr scf_fapi_ul_tti_pdu_type_t pdu_type = UL_TTI_PDU_TYPE_PUCCH;

    explicit PucchPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Cache the local cell index for the next PUCCH PDU batch on this view.
     *
     * Must be called by the dispatcher before the first @ref parse() invocation
     * for a (sfn, slot, cell) triple; @ref parse() short-circuits otherwise.
     *
     * @param[in] local_cell_id  Zero-based local cell index inside the slot's
     *                           cell array (the FAPI @c msg.cell_id), NOT the
     *                           carrier id from @c view->carrier_id().
     */
    void setup_cell(uint32_t local_cell_id) noexcept
    {
        local_cell_id_ = local_cell_id;
        cell_context_ready_ = true;
    }

    /**
     * ULSlotProcessor-friendly overload that ignores @p req for now but keeps
     * the SRS-style signature so MR3 can stash @c slot_ind_ / runtime flags
     * without a follow-up signature change.
     *
     * @param[in] req           UL_TTI request (unused in MR1; reserved).
     * @param[in] msg_cell_id   Zero-based local cell index (same semantic as
     *                          the single-arg overload).
     */
    void setup_cell([[maybe_unused]] const scf_fapi_ul_tti_req_t& req,
                    uint32_t msg_cell_id) noexcept
    {
        setup_cell(msg_cell_id);
    }

    /**
     * Process one PUCCH PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   PUCCH PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.
    uint32_t local_cell_id_{0};    //!< Local slot-command cell index, not carrier id.
    bool cell_context_ready_{false};
};

template<UlModuleView V>
bool PucchPduParser<V>::parse(uint16_t sfn, uint16_t slot,
                               const pdu_t& pdu) noexcept
{
    if (pdu.format_type > UL_TTI_PUCCH_FORMAT_4)
    {
        return false;
    }
    if (!cell_context_ready_)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PucchPduParser: setup_cell was not called before parse sfn={} slot={}",
                   sfn, slot);
        return false;
    }

    return detail::parse_pucch_pdu_common(*view_, local_cell_id_, sfn, slot, pdu, "PucchPduParser");
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PUCCH_PDU_PARSER_HPP_INCLUDED_
