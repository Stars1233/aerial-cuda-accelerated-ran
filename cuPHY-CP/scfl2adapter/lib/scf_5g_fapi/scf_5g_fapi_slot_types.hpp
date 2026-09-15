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

#if !defined(SCF_5G_FAPI_SLOT_TYPES_HPP_INCLUDED_)
#define SCF_5G_FAPI_SLOT_TYPES_HPP_INCLUDED_

/**
 * @file scf_5g_fapi_slot_types.hpp
 * @brief Lightweight shared types for SCF 5G FAPI slot processing.
 *
 * Contains the CellView concept, pm_weight_map_t, and module_view_traits/
 * cell_view_traits primary templates used by parser template parameters.
 *
 * Uses forward declarations for nv:: and cuphy types so that this header
 * compiles without including scf_5g_fapi_phy.hpp or nv_phy_module.hpp —
 * both of which pull in nv_phy_instance.hpp, a private nvPHY header that is
 * not present in the installed SDK.
 *
 * Concepts only require types to be DECLARED (not defined) for std::same_as
 * return-type checks.  The concrete PhyCellView / PhyModuleView classes that
 * actually USE these types live in scf_5g_fapi_message_context.hpp, which
 * continues to include the full nvPHY headers.
 *
 * Mock implementations in unit tests must include the full headers separately
 * (e.g. nv_phy_config_option.hpp) for any nv:: types they store by value.
 */

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "slot_command/slot_command.hpp"    // slot_command_api::*, bfw_coeff_mem_info_t

// ---------------------------------------------------------------------------
// Forward declarations — only type identity is needed for concept checks.
// Full definitions are in the respective nv_phy_*.hpp headers.
// ---------------------------------------------------------------------------

// cuphyCellStatPrm_t is already complete via slot_command.hpp → cuphy_h → cuphy_api.h

namespace nv
{
    // slot_detail_t is a using-alias for slot_detail_ in nv_phy_fapi_msg_common.hpp.
    // Forward-declare the underlying struct, then mirror the alias.
    struct slot_detail_;
    using  slot_detail_t = slot_detail_;
    class  phy_mac_transport; // nv_phy_mac_transport.hpp
    struct phy_config_option; // nv_phy_config_option.hpp

    struct slot_limit_cell_error_t;  // nv_phy_limit_errors.hpp
    struct slot_limit_group_error_t; // nv_phy_limit_errors.hpp

    // pucch_dtx_t_list is a type alias, not a class — define it inline.
    // Mirrors the definition in nv_phy_fapi_msg_common.hpp.
    using pucch_dtx_t_list = std::array<float, 5>;

    // PRACH PhyModuleView accessors (GT-11843, MR 2b) — declared here so the
    // PrachModuleView concept in scf_5g_fapi_slot_concepts.hpp can reference
    // these types without dragging the heavy nv_phy_fapi_msg_common.hpp into
    // every consumer.
    struct phy_config;                              // nv_phy_fapi_msg_common.hpp
    struct prach_addln_config_t_;                   // nv_phy_fapi_msg_common.hpp
    using  prach_addln_config_t = prach_addln_config_t_;
} // namespace nv

namespace scf_5g_fapi
{

/**
 * Associates a ModuleView implementation with the CellView type returned by cell_view.
 *
 * Primary template deduces cell_view_type from ModuleViewT::cell_view.
 * Specialize for a concrete module view when you want an explicit pairing
 * (see PhyModuleView in scf_5g_fapi_message_context.hpp).
 */
template <typename ModuleViewT>
struct module_view_traits {
    using cell_view_type = std::remove_cvref_t<decltype(
        std::declval<const ModuleViewT&>().cell_view(
            std::declval<uint32_t>(),
            std::declval<const slot_command_api::slot_indication&>()))>;
};

/**
 * Reverse mapping: associates a per-cell PHY view type with its ModuleView.
 *
 * Only types that belong to a known module view need a specialization
 * (see PhyCellView in scf_5g_fapi_message_context.hpp).
 */
template <typename CellViewT>
struct cell_view_traits {};

/**
 * PM weight lookup map shared across all cells.
 *
 * Key encodes cell index and PMI; value holds layer/port counts and cuPHY PM
 * weight buffer.
 */
using pm_weight_map_t = std::unordered_map<uint32_t, slot_command_api::pm_weights_t>;

/**
 * Concept for per-cell PHY data used by slot-command paths.
 *
 * - num_dl_prb()   DL BWP PRB count.
 * - slot_detail()  TDD slot detail; null when not applicable.
 * - cell_params()  Full live cell stat parameters (cuphyCellStatPrm_t).
 *
 * std::same_as checks only require the named types to be declared, not
 * complete — forward declarations above suffice for concept evaluation.
 */
template <typename T>
concept CellView = requires(const T& v) {
    { v.num_dl_prb()  } -> std::convertible_to<uint16_t>;
    { v.slot_detail() } -> std::same_as<nv::slot_detail_t*>;
    { v.cell_params() } -> std::same_as<const cuphyCellStatPrm_t&>;
};

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_SLOT_TYPES_HPP_INCLUDED_
