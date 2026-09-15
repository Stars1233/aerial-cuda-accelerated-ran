/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <optional>

#include "scf_5g_slot_commands_common.hpp"
#include "scf_5g_slot_commands_mod_comp.hpp"

namespace scf_5g_fapi {

    /**
     * Populate PBCH/SSB dynamic params + FH metadata for one SSB PDU.
     *
     * Shared by the legacy on_msg path and the new channel-task SsbPduParser path.
     * Takes the FAPI SSB PDU by const reference so the trailing flexible pc_and_bf
     * payload (precoding/beamforming bytes) is read from the caller's original buffer.
     *
     * @param[in,out] cell_grp_cmd   Cell-group command this PDU contributes to.
     * @param[in,out] cell_cmd       Per-cell sub-command; slot type + slot_3gpp stamped here.
     * @param[in]     cmd            SSB PDU; the underlying FAPI buffer must outlive this call.
     * @param[in]     cell_index     Logical cell index (carrier_id).
     * @param[in,out] slotinfo       Slot indication copied into cell_cmd.slot.slot_3gpp.
     * @param[in]     cell_params    Per-cell phy_config (frequencies, SCS, grid size).
     * @param[in]     l_max          Lmax derived from SSB case (4 / 8 / 64).
     * @param[in]     lmax_symbols   Symbol table for the SSB case (Lmax-sized).
     * @param[in,out] config_options Runtime overrides (staticSsb*, enableTickDynamicSfnSlot).
     * @param[in,out] pm_map         Precoding-matrix weight map.
     * @param[in]     slot_detail    TDD slot detail; nullable for non-SINGLE_SECT_MODE RUs.
     * @param[in]     mmimo_enabled  MMIMO active flag.
     */
    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, const scf_fapi_ssb_pdu_t& cmd, int32_t cell_index, slot_indication & slotinfo, const nv::phy_config& cell_params, uint8_t l_max, const uint16_t* lmax_symbols, nv::phy_config_option& config_options, pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled);

    /**
     * Populate PBCH/SSB cuPHY precoding-weight payload for one SSB block.
     *
     * Invoked by update_cell_command after the per-block dynamic params are set.
     *
     * @param[in,out] cell_grp_cmd       Cell-group command (pm_group accessed via get_pm_group).
     * @param[in,out] block              Per-block cuPHY dyn params.
     * @param[in]     ssb_cell_params    Per-cell SSB dyn params.
     * @param[in]     pdu                Tx precoding/beamforming PDU (from SSB pc_and_bf).
     * @param[in,out] config_options     Runtime overrides (precoding_enabled).
     * @param[in,out] prec_group         Pm group destination for weight rows.
     * @param[in,out] pm_map             Precoding-matrix weight map.
     * @param[in,out] cell_cmd           Per-cell sub-command.
     * @param[in]     cell_index         Logical cell index.
     * @param[in]     slot_detail        TDD slot detail; nullable.
     * @param[in]     cell_params        Per-cell phy_config.
     * @param[in]     mmimo_enabled      MMIMO active flag.
     */
    void update_pm_weights_ssb_cuphy(cell_group_command* cell_grp_cmd, cuphyPerSsBlockDynPrms_t& block, cuphyPerCellSsbDynPrms_t& ssb_cell_params, const scf_fapi_tx_precoding_beamforming_t& pdu, nv::phy_config_option& config_options,
        pm_group* prec_group, pm_weight_map_t& pm_map, cell_sub_command& cell_cmd, int32_t cell_index, nv::slot_detail_t* slot_detail, const nv::phy_config& cell_params, bool mmimo_enabled);

    /**
     * Resolve one SSB precoding matrix to its slot in the per-channel PMW cache.
     *
     * Pure: touches only @p prec_group and @p pm_map, with no driver, GPU or global
     * state — which is what makes it unit-testable, unlike its caller
     * update_pm_weights_ssb_cuphy() (that reaches nv::PHYDriverProxy::getInstance()).
     *
     * On a cache miss the entry is appended at index pm_group::nPmPbch — the
     * per-channel counter — so ssb_pmw_idx_cache and ssb_list stay in lockstep, both
     * bounded by MAX_SSB_BLOCKS_PER_SLOT * MAX_CELLS_PER_CELL_GROUP. The cross-channel
     * pm_group::nCacheEntries is deliberately NOT used: it is shared with the
     * differently-sized PDCCH and CSI-RS caches, so one counter cannot index all three.
     *
     * @param[in,out] prec_group  Pm group holding the SSB cache and ssb_list.
     * @param[in]     pm_map      Precoding-matrix weight map.
     * @param[in]     cache_pmi   Cache key: pdu_pmi | (cell_index << 16).
     * @param[out]    cache_full  Optional; set to true only when the empty result is
     *                            caused by the cache having no room. Never cleared, so
     *                            one flag can accumulate across a PRG loop. The caller
     *                            cannot infer this from prec_group alone: once the cache
     *                            is full every other failure cause would look identical.
     * @return Index into ssb_list on a cache hit or a successful append; std::nullopt
     *         when the PMI is absent from @p pm_map, is not single-layer, has more than
     *         MAX_DL_PORTS ports, or the cache is full. The result MUST be checked before
     *         use — this function is noexcept and reports every failure through the empty
     *         optional rather than throwing, so an unchecked dereference is a defect.
     */
    [[nodiscard]] std::optional<uint32_t> resolve_ssb_pmw_slot(pm_group& prec_group,
                                                               const pm_weight_map_t& pm_map,
                                                               uint32_t cache_pmi,
                                                               bool* cache_full = nullptr) noexcept;

}
