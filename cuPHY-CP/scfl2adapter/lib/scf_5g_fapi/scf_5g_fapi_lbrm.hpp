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

#if !defined(SCF_5G_FAPI_LBRM_HPP_INCLUDED_)
#define SCF_5G_FAPI_LBRM_HPP_INCLUDED_

#include <array>
#include <algorithm>
#include <cstdint>
#include <utility>

namespace scf_5g_fapi::lbrm
{

// ---------------------------------------------------------------------------
// Shared Limited-Buffer Rate-Matching (LBRM) helpers
//
// Used by both the PDSCH and PUSCH PDU parsers so the N_PRB_LBRM threshold
// table (TS 38.212 Table 5.4.2.1-2) and the maxQm derivation live in one
// place.  All helpers are inline constexpr free functions: fully evaluable at
// compile time for constant inputs and branchless-friendly for runtime inputs.
// ---------------------------------------------------------------------------

/// Maximum number of spatial layers per UE used for the LBRM Nref calculation.
inline constexpr uint8_t k_max_layers = 4;

/// MCS table index selecting 256-QAM (TS 38.214 Table 5.1.3.1-2 / 6.1.4.1-2).
inline constexpr uint8_t k_mcs_table_256qam = 1;

/// maxQm for 256-QAM modulation order.
inline constexpr uint8_t k_max_qm_256qam = 8;

/// maxQm for 64-QAM (and lower) modulation orders.
inline constexpr uint8_t k_max_qm_64qam = 6;

/**
 * Derive maxLayers for TBS_LBRM per TS 38.212 §5.4.2.1: min(maxMIMO-Layers, 4).
 *
 * @param[in] num_ant  Antenna port count from CONFIG.request (nTxAnt / nRxAnt);
 *                     the gNB supports 2 or more.
 * @return  num_ant, capped at k_max_layers. Return value must be checked.
 */
[[nodiscard]] inline constexpr uint8_t compute_max_layers(uint16_t num_ant) noexcept
{
    // FAPI has no PDSCH/PUSCH maintenance TLV carrying maxMIMO-Layers here, so the
    // closest value L2 has told us is the TX/RX antenna port count from
    // CONFIG.request. Disagreement with the UE's maxMIMO-Layers changes Ncb, which
    // shifts every k0 for rv != 0 and makes LBRM-limited retransmissions
    // undecodable while rv 0 still works.
    return static_cast<uint8_t>(std::min<uint16_t>(num_ant, k_max_layers));
}

/**
 * Compute N_PRB_LBRM from the BWP size per TS 38.212 Table 5.4.2.1-2.
 *
 * Uses std::ranges::lower_bound on a constexpr threshold table so the lookup
 * is branchless-friendly and fully evaluable at compile time for constant
 * inputs.  The bucket boundaries are equivalent to the legacy if-ladder in
 * compute_N_prb_lbrm() (scf_5g_slot_commands_common.hpp).
 *
 * @param[in] bwp_size  BWP bandwidth in PRBs.
 * @return  N_PRB_LBRM: the smallest TS 38.212 Table 5.4.2.1-2 bucket value
 *          ({32,66,107,135,162,217,273}) that is >= bwp_size, saturating at 273
 *          when bwp_size exceeds the largest entry. The return value must be checked.
 */
[[nodiscard]] inline constexpr uint16_t compute_n_prb_lbrm(uint16_t bwp_size) noexcept
{
    using entry_t = std::pair<uint16_t, uint16_t>;
    constexpr auto k_lbrm_table = std::to_array<entry_t>({
        {32u,  32u},  {66u,  66u},  {107u, 107u}, {135u, 135u},
        {162u, 162u}, {217u, 217u}, {273u, 273u},
    });
    const auto it = std::ranges::lower_bound(k_lbrm_table, bwp_size, {},
                                              &entry_t::first);
    return (it != k_lbrm_table.end()) ? it->second : 273u;
}

/**
 * Derive maxQm from the FAPI MCS table index.
 *
 * @param[in] mcs_table  SCF FAPI mcs_table field.
 * @return  maxQm modulation order: 8 when mcs_table selects the 256-QAM table
 *          (k_mcs_table_256qam), else 6 (64-QAM and lower). The return value
 *          must be checked.
 */
[[nodiscard]] inline constexpr uint8_t compute_max_qm(uint8_t mcs_table) noexcept
{
    return (mcs_table == k_mcs_table_256qam) ? k_max_qm_256qam : k_max_qm_64qam;
}

} // namespace scf_5g_fapi::lbrm

#endif // SCF_5G_FAPI_LBRM_HPP_INCLUDED_
