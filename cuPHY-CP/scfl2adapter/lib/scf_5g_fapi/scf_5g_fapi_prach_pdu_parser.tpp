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
 * @file scf_5g_fapi_prach_pdu_parser.tpp
 * @brief Out-of-line template definitions for @ref scf_5g_fapi::PrachPduParser.
 *
 * Option A split: this file is included from scf_5g_fapi_prach_pdu_parser.hpp
 * (bottom of the header, before the include guard's #endif). The .tpp
 * extension signals "template body, header-included only — NOT a standalone
 * translation unit." It is NOT added to CMakeLists.txt target_sources; every
 * TU that instantiates @c PrachPduParser<V> picks up the body via the header
 * include.
 *
 * Part of the PRACH parser stack.
 */

#include "scf_5g_fapi_prach_slot_builder.hpp"   // BuildContext, populate_slot_command

namespace scf_5g_fapi
{

template<PrachModuleView V>
bool PrachPduParser<V>::parse(const uint16_t           sfn,
                              const uint16_t           slot,
                              const pdu_t&             pdu) noexcept
{
    // Fast-out: an empty PDU is not an error — caller may emit zero-occasion
    // PRACH messages during slot-pacing edge cases. Match legacy semantics
    // (scf_5g_fapi_phy.cpp:on_prach_pdu_info returns without touching state).
    if (pdu.num_prach_ocas == 0u) [[unlikely]] {
        return true;
    }

    // TODO: re-enable L1-limit validation here. The legacy path calls
    // validate_prach_pdu_l1_limits() and sets
    // cell_errors.error_mask |= SCF_FAPI_PRACH_L1_LIMIT_EXCEEDED on failure;
    // wiring that into the new parser is deferred to a follow-up so the
    // FCIS/DOP cutover lands with a minimal diff. The Group-F-L1Limit test
    // placeholder in test_prach_pdu_parser.cpp documents the contract.

    // Logical carrier index, stashed by setup_cell() from the dispatcher's
    // per-message msg.cell_id. NOT pdu.phys_cell_id (which is the 3GPP PCI
    // 0–1007 and is unsuitable for module-view array indexing).
    const uint32_t cell_id = cell_id_;
    const slot_command_api::slot_indication slot_ind{sfn, slot, /*tick=*/uint64_t{0}};

    // Resolve every dependency through the module view (no driver singletons).
    auto* const                                    group_ptr   = view_->group_command();
    if (group_ptr == nullptr) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PrachPduParser::parse: group_command() is null at sfn={} slot={} cell_id={}",
                   sfn, slot, cell_id);
        return false;
    }
    slot_command_api::cell_sub_command&            cell        = view_->cell_sub_command(cell_id);
    const nv::phy_config&                          phy         = view_->phy_config(cell_id);
    const nv::prach_addln_config_t&                addln       = view_->prach_addln_config(cell_id);
    const auto                                     cv          = view_->cell_view(cell_id, slot_ind);
    nv::slot_detail_t* const                       slot_detail = cv.slot_detail();

    const prach::BuildContext ctx{
        .group                  = *group_ptr,
        .cell                   = cell,
        .slot_ind               = slot_ind,
        .phy_config             = phy,
        .addln_config           = addln,
        .slot_detail            = slot_detail,
        .cell_index             = static_cast<int32_t>(cell_id),
        .ru                     = view_->ru_type_for_cell(cell_id),
        .bf_enabled             = view_->bf_enabled(),
        .mmimo_enabled          = view_->mmimo_enabled(),
        .fapi_to_cplane_direct  = view_->is_fapi_to_cplane_direct(),
    };

    if (!prach::populate_slot_command(ctx, pdu, view_->order_sym_prb_info(cell_id))) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PrachPduParser::parse: populate_slot_command failed at sfn={} slot={} cell_id={}",
                   sfn, slot, cell_id);
        return false;
    }
    return true;
}

} // namespace scf_5g_fapi
