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

#if !defined(SCF_5G_FAPI_PRACH_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_PRACH_PDU_PARSER_HPP_INCLUDED_

#include <cstdint>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi
{

/**
 * Per-PDU PRACH parser — drives compute -> apply_rach -> maybe apply_fh.
 *
 * Templated on @ref PrachModuleView (the refined concept narrower than the
 * universal UlModuleView). The refinement gives the parser access to
 * `phy_config(cell_id)`, `prach_addln_config(cell_id)`, `ru_type_for_cell`,
 * `get_cell_limit_errors`, and `is_fapi_to_cplane_direct()` — everything
 * needed to build the @ref prach::BuildContext from the module view alone
 * (no driver singletons on the hot path).
 *
 * @tparam V  Type satisfying @ref PrachModuleView.
 */
template<PrachModuleView V>
class PrachPduParser final
{
public:
    using pdu_t = scf_fapi_prach_pdu_t;
    static constexpr scf_fapi_ul_tti_pdu_type_t pdu_type = UL_TTI_PDU_TYPE_PRACH;

    explicit PrachPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Bind subsequent @ref parse calls to the logical carrier index of the
     * current UL_TTI.request message. Mirrors PdschPduParser::setup_cell.
     *
     * `pdu.phys_cell_id` carried in each PRACH PDU is the 3GPP PCI (0–1007),
     * NOT a logical carrier index — so @ref parse cannot derive @c cell_id
     * from the PDU payload alone. The dispatcher (ULSlotProcessor) calls
     * this once per message with `msg.cell_id` before invoking dispatch.
     *
     * @param[in] cell_id  Logical carrier index (matches @c get_carrier_id()).
     */
    void setup_cell(uint32_t cell_id) noexcept { cell_id_ = cell_id; }

    /**
     * Process one PRACH PDU for the given sfn/slot.
     *
     * Pipeline: fast-out on num_prach_ocas==0, compute_prach_params (pure
     * core), apply_prach_to_slot_command (always-on shell), and
     * apply_prach_fh_to_sym_prb_info (FH/order metadata).
     *
     * Reads @c cell_id_ stashed by the most recent @ref setup_cell call to
     * resolve every module-view lookup (cell_sub_command, phy_config,
     * prach_addln_config, ru_type_for_cell, cell_view).
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   PRACH PDU to process.
     * @return true on success; false if the pure core rejects the PDU or
     *         the shell hits a capacity refusal.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_;    //!< Non-owning; must outlive this parser.
    uint32_t               cell_id_{}; //!< Stashed by setup_cell(); consumed by parse().
};

} // namespace scf_5g_fapi

// Option A split: parse() definition lives in the sibling .tpp. The .tpp
// extension signals "template body, header-included only — NOT a standalone
// TU" (Boost-style convention). The file is NOT added to CMakeLists.txt
// target_sources; including it here ensures the template body is visible at
// every PrachPduParser<V> instantiation site without forcing the project to
// declare explicit instantiations for every concrete PrachModuleView (only
// PhyModuleView today; more in the future).
#include "scf_5g_fapi_prach_pdu_parser.tpp"

#endif // SCF_5G_FAPI_PRACH_PDU_PARSER_HPP_INCLUDED_
