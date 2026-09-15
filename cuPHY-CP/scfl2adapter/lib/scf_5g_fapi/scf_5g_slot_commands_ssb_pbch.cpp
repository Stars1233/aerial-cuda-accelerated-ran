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

#include "scf_5g_slot_commands_ssb_pbch.hpp"
#include "aerial/casts/casts.hpp"
#include "nvlog.h"
#include "nvlog_fmt.hpp"
#include "nv_phy_module.hpp"

#include <algorithm>
#include <array>
#include <tuple>
#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

namespace scf_5g_fapi {

    using re_start_end_t = std::array<uint16_t, 2>;
    using ssb_pbch_re_offset = std::tuple<ssb_pbch, re_start_end_t>;
    using ss_pbch_segment = std::vector<ssb_pbch_re_offset>;

    static std::array<ss_pbch_segment, 4> ss_segments =  {{ 
                                        {{ssb_pbch::PSS, {56, 182}}},
                                        {{ssb_pbch::DMRS_DATA, {0, 239}}},
                                        {{ssb_pbch::DMRS_DATA, {0, 47}}, {ssb_pbch::SSS, {56, 182}}, {ssb_pbch::DMRS_DATA, {192, 239}} },
                                        {{ssb_pbch::DMRS_DATA, {0, 239}}}
                                     }};

    void update_fh_params_ssb(cuphyPerSsBlockDynPrms_t& block, uint16_t k_SSB, const scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, cell_sub_command& cell_cmd, nv::phy_config_option& config_option, uint32_t prec_ports, enum ru_type ru, int32_t cell_index, nv::slot_detail_t* slot_detail, comp_method dl_comp_method, const nv::phy_config& cell_params, bool mmimo_enabled);
    void update_mod_comp_info_section_ssb(prb_info_t& prb, ssb_pbch subtype, cuphyPerSsBlockDynPrms_t& block, uint16_t reMask);

    void update_cell_command(cell_group_command* cell_grp_cmd,cell_sub_command& cell_cmd, const scf_fapi_ssb_pdu_t& cmd, int32_t cell_index, slot_indication & slotinfo, const nv::phy_config& cell_params,
        uint8_t l_max, const uint16_t* lmax_symbols, nv::phy_config_option& config_option, pm_weight_map_t& pm_map, nv::slot_detail_t*  slot_detail, bool mmimo_enabled)
    {
        bool new_cell = false;
        // cell_cmd.create_if(channel_type::PBCH);
        pbch_group_params * grp_params = cell_grp_cmd->get_pbch_params();
        auto& cell_dyn_params = grp_params->pbch_dyn_cell_params[grp_params->ncells];
        auto& block_params = grp_params->pbch_dyn_block_params[grp_params->nSsbBlocks];
        auto& mib_data = grp_params->pbch_dyn_mib_data[grp_params->nSsbBlocks];
        cell_cmd.slot.type = SLOT_DOWNLINK;
        cell_cmd.slot.slot_3gpp = slotinfo;
        cell_grp_cmd->slot.type = SLOT_DOWNLINK;
        cell_grp_cmd->slot.slot_3gpp = slotinfo;


        auto it = std::find(grp_params->cell_index_list.begin(),grp_params->cell_index_list.end(), cell_index);

        if(it == grp_params->cell_index_list.end())
        {
            grp_params->cell_index_list.push_back(cell_index);
            grp_params->phy_cell_index_list.push_back(cell_cmd.cell);
            grp_params->ncells++;
            new_cell = true;
        }
        uint16_t slot = (config_option.staticSsbSlotNum != -1) ? config_option.staticSsbSlotNum : slotinfo.slot_;

        if (new_cell) {
            cell_dyn_params.NID = (config_option.staticSsbPcid != -1) ? config_option.staticSsbPcid : cell_params.cell_config_.phy_cell_id;
            cell_dyn_params.nHF = slot/ (5 << cell_params.ssb_config_.sub_c_common);
            cell_dyn_params.Lmax = l_max;
            if (config_option.enableTickDynamicSfnSlot)
            {
                cell_dyn_params.SFN = slotinfo.sfn_;
            }
            else
            {
                cell_dyn_params.SFN = (config_option.staticSsbSFN != -1) ? config_option.staticSsbSFN : 0;
            }
            cell_dyn_params.k_SSB = cmd.ssb_subcarrier_offset;
            cell_dyn_params.nF = cell_params.carrier_config_.dl_grid_size[cell_params.ssb_config_.sub_c_common] * CUPHY_N_TONES_PER_PRB;
            cell_dyn_params.slotBufferIdx = grp_params->cell_index_list.size() - 1;
        }

        // TODO: needs to be computed

        block_params.blockIndex = cmd.ssb_block_index;
        block_params.t0 = (lmax_symbols[block_params.blockIndex] % OFDM_SYMBOLS_PER_SLOT);

        switch (cell_params.ssb_config_.sub_c_common)
        {
            case 0: // 15Khz
                block_params.f0 = cmd.ssb_subcarrier_offset + cmd.ssb_offset_point_a * CUPHY_N_TONES_PER_PRB;

                break;
            case 1: // 30Khz
               block_params.f0 = (cmd.ssb_subcarrier_offset + cmd.ssb_offset_point_a * CUPHY_N_TONES_PER_PRB) / (1<<1);
                break;
            case 2:  // 60Khz
                break;
            case 3:  // 120Khz
                block_params.f0 = (cmd.ssb_subcarrier_offset + cmd.ssb_offset_point_a * CUPHY_N_TONES_PER_PRB) / (1<<3);
                break;
            case 4:  // 240Khz
                block_params.f0 = (cmd.ssb_subcarrier_offset + cmd.ssb_offset_point_a * CUPHY_N_TONES_PER_PRB) / (1<<4);
                break;
            default:
                break;
        }
        // Interpret pdu.betaPss according to SCS FAPI Table 3-40, 0=0 dB, 1=3 dB
        // power increase of PSS over SSS
        // SSTxParams.beta is used internally as a linear (amplitude) scaler
        switch (cmd.beta_pss)
        {
            case 0:
                block_params.beta_pss = 1;
                break;
            case 1:
                block_params.beta_pss = std::pow(10.0, 3.0 / 20.0);
                break;
            default:
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Unknown value for betaPss: {}", cmd.beta_pss);
                break;
        }
        block_params.beta_sss = 1;
        if (new_cell) {
            block_params.cell_index = grp_params->cell_index_list.size() - 1;
        } else {
            block_params.cell_index = std::distance(grp_params->cell_index_list.begin(), it);
        }

        mib_data = cmd.mib_pdu.agg;
        grp_params->nSsbBlocks++;
        auto& ssb_cell_params = grp_params->pbch_dyn_cell_params[block_params.cell_index];
        const auto& pc_bf_info = *aerial::casts::assume_cast<const scf_fapi_tx_precoding_beamforming_t>(&cmd.pc_and_bf[0]);
        auto pm_grp = cell_grp_cmd->get_pm_group();


        // update_fh_params_ssb(block_params.f0, ssb_cell_params.k_SSB, block_params.t0, pc_bf_info, cell_cmd, config_option, 0);
        update_pm_weights_ssb_cuphy(cell_grp_cmd, block_params, ssb_cell_params, pc_bf_info, config_option, pm_grp, pm_map, cell_cmd, cell_index, slot_detail, cell_params, mmimo_enabled);
    }


    std::optional<uint32_t> resolve_ssb_pmw_slot(pm_group& prec_group,
                                                 const pm_weight_map_t& pm_map,
                                                 const uint32_t cache_pmi,
                                                 bool* const cache_full) noexcept
    {
        // One index addresses both: the cache-full guard below uses
        // ssb_pmw_idx_cache.size(), then the miss path writes ssb_list[slot].
        static_assert(std::tuple_size_v<decltype(pm_group::ssb_list)>
                          == std::tuple_size_v<decltype(pm_group::ssb_pmw_idx_cache)>,
                      "SSB PMW cache and ssb_list must share a capacity: one index addresses both");

        const auto pmw_iter = pm_map.find(cache_pmi);
        if (pmw_iter == pm_map.end()) [[unlikely]] { return std::nullopt; }
        if (pmw_iter->second.layers != 1) [[unlikely]] { return std::nullopt; }
        // ssb_list entries are cuphyPmWOneLayer_t, whose matrix holds MAX_DL_PORTS
        // elements. layers == 1 here, so the std::copy below writes `ports` elements into
        // that fixed buffer; a PM config with ports > MAX_DL_PORTS would overflow it. This
        // function is noexcept, so reject the entry rather than corrupt memory. The
        // population site (scf_5g_fapi_phy.cpp) rejects such PDUs too -- this is defence in
        // depth for the SSB destination, which is narrower than the source cuphyPmW_t.
        if (pmw_iter->second.ports > MAX_DL_PORTS) [[unlikely]] { return std::nullopt; }

        // Search only the live prefix [0, nPmPbch), not the whole array. Entries beyond
        // nPmPbch hold whatever pm_group::reset() last wrote, and reset() only clears when
        // its own precoding_enabled is set -- a different flag from the
        // config_options.precoding_enabled that gates this path. Scanning past the live
        // range could therefore return an nIndex belonging to a previous slot. Size is the
        // validity bound, exactly as it is for slot_command's cleared-but-not-zeroed arrays.
        // Clamp so a corrupt nPmPbch cannot form an iterator past end() before find_if runs.
        const auto live_begin = prec_group.ssb_pmw_idx_cache.begin();
        const auto live_end   = live_begin + std::min<std::size_t>(
                                    prec_group.nPmPbch, prec_group.ssb_pmw_idx_cache.size());
        const auto iter = std::find_if(live_begin, live_end,
                                       [cache_pmi](const auto& e) { return e.pmwIdx == cache_pmi; });
        if (iter != live_end) { return iter->nIndex; }

        // Cache miss: append at the per-channel counter nPmPbch, NOT the cross-channel
        // pm_group::nCacheEntries, so this cache and ssb_list stay in lockstep -- both
        // are bounded by MAX_SSB_BLOCKS_PER_SLOT * MAX_CELLS_PER_CELL_GROUP.
        //
        // nCacheEntries is shared with the PDCCH and CSI-RS caches, which have different
        // capacities, so one counter cannot index all three correctly. It is currently
        // benign here only because SSB is the sole writer reached from the DLSlotProcessor
        // path (SsbPduParser -> update_cell_command); the remaining PDCCH/CSI-RS writers
        // sit on the legacy scf_5g_fapi_phy.cpp handlers. Routing either of those through
        // a parser would make the shared counter both mis-indexed and racy, since the
        // channel aggregation tasks run in parallel against the same pm_group. Using the
        // per-channel counter removes that coupling ahead of time. Same rationale, and
        // same fix, as scf_5g_csirs_slot_command_helpers.cpp.
        const uint32_t slot = prec_group.nPmPbch;
        if (slot >= prec_group.ssb_pmw_idx_cache.size()) [[unlikely]] {
            // Full. This function is noexcept, so .at() overflowing here would call
            // std::terminate rather than throw -- report the miss and let the caller
            // fall back to unprecoded transmission.
            //
            // Deliberately silent: this is the functional core, and it is called once per
            // PRG. Logging here would emit one line per PRG for the rest of the PDU once
            // the cache fills. The caller logs once per PDU instead -- see the
            // ssb_pmw_cache_full handling in update_pm_weights_ssb_cuphy(). Report the
            // cause here rather than let the caller re-derive it: from outside, a full
            // cache is indistinguishable from an unconfigured PMI.
            if (cache_full != nullptr) { *cache_full = true; }
            return std::nullopt;
        }
        auto& cache_entry = prec_group.ssb_pmw_idx_cache.at(slot);
        cache_entry.pmwIdx = cache_pmi;
        cache_entry.nIndex = slot;

        auto& val = prec_group.ssb_list[slot];
        val.nPorts = pmw_iter->second.weights.nPorts;
        std::copy(pmw_iter->second.weights.matrix,
                  pmw_iter->second.weights.matrix + (pmw_iter->second.layers * pmw_iter->second.ports),
                  val.matrix);
        prec_group.nPmPbch++;
        return slot;
    }

    void update_pm_weights_ssb_cuphy(cell_group_command* cell_grp_cmd, cuphyPerSsBlockDynPrms_t& block, cuphyPerCellSsbDynPrms_t& ssb_cell_params, const scf_fapi_tx_precoding_beamforming_t& pdu, nv::phy_config_option& config_options, pm_group* prec_group,
                                     pm_weight_map_t& pm_map, cell_sub_command& cell_cmd, int32_t cell_index, nv::slot_detail_t *slot_detail, const nv::phy_config& cell_params, bool mmimo_enabled) {
        block.enablePrcdBf = config_options.precoding_enabled;

        auto default_values = [&block]() {
            block.enablePrcdBf = false;
        };
        auto prec_ports = UINT32_MAX;

        // Set by resolve_ssb_pmw_slot() only when a PRG was dropped because the PMW cache
        // had no room, so an unconfigured PMI is not reported as a full cache. Reported
        // once after the loop rather than per PRG: once the cache fills, every remaining
        // PRG in this PDU takes the same path, and this runs on the slot path.
        bool ssb_pmw_cache_full = false;

        uint16_t offset = 0;
        for (uint16_t i = 0; i < pdu.num_prgs; i++) {
            uint16_t pdu_pmi = pdu.pm_idx_and_beam_idx[i + offset];
            uint32_t cache_pmi = pdu_pmi | cell_index << 16; /// PMI Unused
            block.enablePrcdBf = block.enablePrcdBf && (pdu_pmi != 0);
            if (block.enablePrcdBf) {
                const auto slot = resolve_ssb_pmw_slot(*prec_group, pm_map, cache_pmi,
                                                       &ssb_pmw_cache_full);
                if (!slot.has_value()) [[unlikely]] {
                    default_values();
                    continue;
                }
                block.pmwPrmIdx = *slot;
                prec_ports = prec_group->ssb_list[*slot].nPorts;
            }
            offset+=(pdu.dig_bf_interfaces + 1);
        }

        if (ssb_pmw_cache_full) [[unlikely]] {
            const auto capacity = prec_group->ssb_pmw_idx_cache.size();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "SSB PMW cache full ({} entries) for cell_index={}; precoding dropped "
                       "for the remaining PRGs of this PDU",
                       capacity, cell_index);
        }

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
        ru_type ru = mplane_info.ru;
        // update_fh_params_ssb(block.f0, ssb_cell_params.k_SSB, block.t0, pdu, cell_cmd, config_options, prec_ports, block.enablePrcdBf, ru, cell_index, slot_detail);
        update_fh_params_ssb(block, ssb_cell_params.k_SSB, pdu, cell_cmd, config_options, prec_ports, ru, cell_index, slot_detail, mplane_info.dl_comp_meth, cell_params, mmimo_enabled);
    }

    inline void update_fh_params_ssb( cuphyPerSsBlockDynPrms_t& block, uint16_t k_SSB, const scf_fapi_tx_precoding_beamforming_t& pmi_bf_pdu, cell_sub_command& cell_cmd, nv::phy_config_option& config_option,
                                     uint32_t prec_ports, enum ru_type ru, int32_t cell_index, nv::slot_detail_t* slot_detail, comp_method dl_comp_method, const nv::phy_config& cell_params, bool mmimo_enabled) {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        uint16_t ssb_start_prb = 0, ssb_num_prb = 0;

        auto update_new_prb_group = [sym_prbs, &prbs, &pmi_bf_pdu, slot_detail, &config_option, &ru, &dl_comp_method, cell_index] (auto t0, auto ssb_start_prb, auto ssb_num_prb, auto &enablePrcdBf, auto& prec_ports, bool mmimo_enabled) {
            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(ssb_start_prb, ssb_num_prb);
            sym_prbs->prbs_size++;
            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};
            prb_info.common.direction = fh_dir_t::FH_DIR_DL;

            if(enablePrcdBf) {
                prb_info.common.portMask = (1<< prec_ports) -1 ;
                prb_info.common.numApIndices = prec_ports;
            }
            else if (config_option.bf_enabled) {
                prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) - 1;
                prb_info.common.numApIndices = pmi_bf_pdu.dig_bf_interfaces;
            } else {
                prb_info.common.portMask = 1;
                prb_info.common.numApIndices = 1;
            }

            if(config_option.bf_enabled)
            {
                // update_beam_list only reads the Tx BF PDU (verified by inspection — it indexes
                // into pmi_bf_pdu but never mutates it), so casting away const here is safe.
                // The legacy API still takes a non-const reference; tracking the signature fix
                // under GT-12547.
                update_beam_list(prb_info.beams_array, prb_info.beams_array_size, const_cast<scf_fapi_tx_precoding_beamforming_t&>(pmi_bf_pdu), mmimo_enabled, prb_info, cell_index);
            }

            if(ru == SINGLE_SECT_MODE) {
                NVLOGD_FMT(TAG, "DL symbols = {}", slot_detail->max_dl_symbols);
                prb_info.common.numSymbols =  (slot_detail == nullptr || !slot_detail->max_dl_symbols? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);

                uint8_t start_symbol = (slot_detail == nullptr ? 0: slot_detail->start_sym_dl);
                update_prb_sym_list(*sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::PDSCH, ru);
            }
            else {
                if (dl_comp_method != comp_method::MODULATION_COMPRESSION) {
                    if(mmimo_enabled)
                    {
                        prb_info.common.numSymbols = 4;
                    }
                    update_prb_sym_list(*sym_prbs, index, t0, 4, channel_type::PBCH, ru);
                } else {
                    update_prb_sym_list(*sym_prbs, index, t0, 1, channel_type::PBCH, ru);
                }
            }
        };

        if (dl_comp_method != comp_method::MODULATION_COMPRESSION) {
            if (ru == SINGLE_SECT_MODE) {
                bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
                if (value)
                    return;

                ssb_start_prb = 0;
                ssb_num_prb = cell_params.carrier_config_.dl_grid_size[cell_params.ssb_config_.sub_c_common];
                update_new_prb_group(block.t0, ssb_start_prb, ssb_num_prb, block.enablePrcdBf, prec_ports, mmimo_enabled);
            }
            else {
                 ssb_start_prb = block.f0 / CUPHY_N_TONES_PER_PRB;
                ssb_num_prb = config_option.bf_enabled ? pmi_bf_pdu.prg_size : 20; // SSB is 240 Sub carriers in size
                if (k_SSB > 0) {
                    ssb_num_prb++;
                }
                update_new_prb_group(block.t0, ssb_start_prb, ssb_num_prb, block.enablePrcdBf, prec_ports, mmimo_enabled);
            }
        } 
        else {
            auto start_sc = block.f0;
            for (uint16_t segment = 0; segment< ss_segments.size(); segment++) {
                for (auto& comp: ss_segments[segment]) {
                    auto& ssb_type = std::get<0>(comp);
                    auto& segment_comp = std::get<1>(comp);
                    uint16_t total_res = segment_comp[1] - segment_comp[0] + 1, segment_size = 0;
                    uint16_t cur_seg_re = start_sc + segment_comp[0];
                    uint16_t end_seg_re = start_sc + segment_comp[1] + 1, parsed_res = 0;
                    uint8_t non_aligned_prb = 0;
                    while (end_seg_re - cur_seg_re) {
                        non_aligned_prb =  cur_seg_re % CUPHY_N_TONES_PER_PRB;
                        if (!!non_aligned_prb) {
                            segment_size = 1;
                        } else {
                            if ((total_res - parsed_res) < CUPHY_N_TONES_PER_PRB) {
                                segment_size = 1;
                            }
                            else {
                                segment_size = (end_seg_re - cur_seg_re) / CUPHY_N_TONES_PER_PRB;
                            }
                        }
                        uint16_t reMask = 0;
                        if (!!non_aligned_prb) {
                            parsed_res += (CUPHY_N_TONES_PER_PRB - non_aligned_prb);
                            reMask =  setPrbReMask( CUPHY_N_TONES_PER_PRB - non_aligned_prb, cur_seg_re % CUPHY_N_TONES_PER_PRB) & 0x0FFF;
                            
                        } else {
                            if ((total_res - parsed_res) < CUPHY_N_TONES_PER_PRB) {
                                reMask =  setPrbReMask(total_res - parsed_res, non_aligned_prb) & 0x0FFF;
                                parsed_res += (total_res - parsed_res);
                            } else {
                                reMask = std::numeric_limits<uint16_t>::max();
                                parsed_res += segment_size *  CUPHY_N_TONES_PER_PRB;
                            }
                        }
                        auto& idxlist{sym_prbs->symbols[block.t0 + segment][channel_type::PBCH]};

                        auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &cur_seg_re, &segment_size, &ru](const auto& e){
                            bool retval = (e < sym_prbs->prbs_size);
                            if (retval) {
                                auto& prb{prbs[e]};
                                if(ru == SINGLE_SECT_MODE)
                                {
                                    retval = (prb.common.startPrbc == 0 && prb.common.numPrbc == 273);
                                }
                                else
                                {
                                    retval = (prb.common.startPrbc == static_cast<uint16_t>(cur_seg_re/CUPHY_N_TONES_PER_PRB)  && prb.common.numPrbc == segment_size);
                                }
                            }
                            return retval;
                        });

                        if (iter != idxlist.end()) {
                            // Found a PRB which is already present
                            auto& prb{prbs[*iter]};
                            update_mod_comp_info_section_ssb(prb, ssb_type, block, reMask);
                            copy_prb_beam_list(prb);
                            // NVLOGD_FMT(TAG, " if reMask {}", reMask);
                        } else {
                            /// Create a new section
                            update_new_prb_group(block.t0 + segment, static_cast<uint16_t>(cur_seg_re/CUPHY_N_TONES_PER_PRB), segment_size,  block.enablePrcdBf, prec_ports, mmimo_enabled);
                            std::size_t index{sym_prbs->prbs_size - 1};
                            prb_info_t& prb_info{prbs[index]};
                            update_mod_comp_info_common(prb_info, getBwScaler(cell_params.carrier_config_.dl_grid_size[cell_params.ssb_config_.sub_c_common]));
                            update_mod_comp_info_section_ssb(prb_info, ssb_type, block, reMask);
                            // NVLOGD_FMT(TAG, " else reMask {}", reMask);
                        }
                        cur_seg_re = start_sc + segment_comp[0] + parsed_res;
                        std::size_t index{sym_prbs->prbs_size - 1};
                        // prb_info_t& prb_info{prbs[index]};
                        // NVLOGD_FMT(TAG, "SSB prb modcomp info SSB type {} startPrbc {} numPrbc {} c_plane section reMask {} ef {} extType {}  nSections {} udIqWidth {}", 
                        //     ssb_type, +prb_info.common.startPrbc, +prb_info.common.numPrbc, +prb_info.common.reMask, prb_info.comp_info.common.ef, 
                        //     prb_info.comp_info.common.extType, prb_info.comp_info.common.nSections, prb_info.comp_info.common.udIqWidth);
                        // for ( uint8_t j = 0 ; j <  +prb_info.comp_info.common.nSections; j++ ) {
                        //     NVLOGD_FMT(TAG, "SSB prb modcomp info SE5  section {} reMask {} csf {}  modCompScaler {}", j, +prb_info.comp_info.sections[j].mcScaleReMask, +prb_info.comp_info.sections[j].csf, 
                        //         +prb_info.comp_info.sections[j].mcScaleOffset);
                        // }
                    }
                }
            }
        }
    }

    inline void update_mod_comp_info_section_ssb(prb_info_t& prb, ssb_pbch subtype, cuphyPerSsBlockDynPrms_t& block, uint16_t reMask) {
        
        switch (subtype) {
            case ssb_pbch::PSS:
                update_mod_comp_info_section(prb, reMask, block.beta_pss, CUPHY_QAM_2, BPSK_CSF); //Special case for BPSK ( value = 4)
                break;
            case ssb_pbch::SSS:
                update_mod_comp_info_section(prb, reMask, block.beta_sss, CUPHY_QAM_2, BPSK_CSF); //Special case for BPSK ( value  = 4)
                break;
            case ssb_pbch::DMRS_DATA:
                update_mod_comp_info_section(prb, reMask, block.beta_sss, CUPHY_QAM_4, DEFAULT_CSF); //Special case for QPSK
                break;
            default:
                break;
        }
    }
}
