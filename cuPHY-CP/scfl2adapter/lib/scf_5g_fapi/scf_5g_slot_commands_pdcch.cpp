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

#include <exception>

#include "scf_5g_slot_commands_pdcch.hpp"
#include "scf_5g_slot_commands_common.hpp"
#include "nvlog.h"
#include "nv_phy_module.hpp"
#include "scf_5g_fapi_dl_validate.hpp"
#include "nv_phy_limit_errors.hpp"


#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"

namespace scf_5g_fapi {

    inline void update_new_coreset(cuphyPdcchCoresetDynPrm_t& coreset, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params);
    inline std::size_t update_new_dci(scf_fapi_dl_dci_t& msg_dci , cuphyPdcchDciPrm_t& dci, dci_payload_t& payload);
    inline void update_new_dci_pm(scf_fapi_dl_dci_t& msg_dci, cuphyPdcchDciPrm_t& dci,  pm_group* prec_group, pm_weight_map_t& pm_map,  nv::phy_config_option& config_options, int32_t cell_index);
    inline void update_fh_params_pdcch(cuphyPdcchCoresetDynPrm_t& coreset, cell_sub_command& cell_cmd, uint16_t bandwitdth, enum ru_type ru, nv::slot_detail_t* slot_detail);
    inline void update_fh_params_pdcch_per_coreset(cuphyPdcchCoresetDynPrm_t& coreset, dci_param_list& dci, cell_sub_command& cell_cmd, uint16_t bandwidth, pm_group* pm_grp, scf_fapi_pdcch_pdu_t & msg, std::size_t* fapiDciOffsets, nv::phy_config_option& config_options, enum ru_type ru, nv::slot_detail_t* slot_detail, comp_method dl_comp_method, bool mmimo_enabled, int32_t cell_index = 0);
    inline std::size_t update_per_dci_beams(scf_fapi_dl_dci_t& msg_dci,  beamid_array_t& array, size_t& array_size, bool bf_enabled, prb_info_t& prb_info, bool mmimo_enabled, int32_t cell_index);
    inline void update_mod_comp_info_pdcch_section(channel_type type, prb_info_t& prb, cuphyPdcchDciPrm_t& dci);


    inline uint16_t getDciToProcess(scf_fapi_pdcch_pdu_t& msg)
    {
        uint16_t nDcisToProcess = msg.num_dl_dci;
#if ENABLE_L2_SLT_RSP
        nDcisToProcess = std::min(+msg.num_dl_dci, CUPHY_PDCCH_MAX_DCIS_PER_CORESET);
#endif
        return nDcisToProcess;
    }

#ifdef ENABLE_L2_SLT_RSP
    void update_cell_command(cell_group_command* cell_group, cell_sub_command& cell_cmd, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params, 
                             int staticPdcchSlotNum, nv::phy_config_option& config_option, pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled, nv::pdcch_limit_error_t* pdcch_error) {
#else
    void update_cell_command(cell_group_command* cell_group, cell_sub_command& cell_cmd, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params, 
                             int staticPdcchSlotNum, nv::phy_config_option& config_option, pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled) {
#endif
        if (cell_group == nullptr) {
            return;
        }

        cell_cmd.slot.type = SLOT_DOWNLINK;
        cell_cmd.slot.slot_3gpp = slotinfo;
        cell_group->slot.type = SLOT_DOWNLINK;
        cell_group->slot.slot_3gpp = slotinfo;

        cell_group->create_if(channel_type::PDCCH_DL);
        auto pdcch_params = cell_group->get_pdcch_params();
        auto& cell_index_list = pdcch_params->cell_index_list;
        auto& phy_cell_id_list= pdcch_params->phy_cell_index_list;
        cell_index_list.push_back(cell_index);
        phy_cell_id_list.push_back( cell_params.phyCellId);

        auto& cset_group = pdcch_params->csets_group;
        auto& dci_group = cset_group.dcis;
        auto& payload_group = cset_group.payloads;

        auto& cset = cset_group.csets[cset_group.nCoresets];
        auto pm_grp = cell_group->get_pm_group();
        update_new_coreset(cset, msg, testMode, slotinfo, cell_params);
        cset_group.nCoresets++;

        if(staticPdcchSlotNum != -1)
        {
            cset.slot_number = staticPdcchSlotNum;
        }
        cset.dciStartIdx =  cset_group.nDcis;//dci_group.size();
        cset.slotBufferIdx = cell_index;

        std::size_t dciOffset = 0;
        uint8_t* dldci_buf = reinterpret_cast<uint8_t*>(msg.dl_dci);
        uint16_t nDcisToProcess = getDciToProcess(msg);
        std::size_t dciOffsets[nDcisToProcess]{0};

        auto start_prb_index = cell_cmd.sym_prb_info()->prbs_size;
        for (uint16_t i = 0; i < msg.num_dl_dci; i++) {
            auto& msg_dci = *reinterpret_cast<scf_fapi_dl_dci_t*>(dldci_buf + dciOffset);
            if (i < nDcisToProcess) {
                auto& dci = dci_group[cset_group.nDcis];
                // payload_group.emplace_back();
                auto& dci_payload = payload_group[cset_group.nDcis];
                // auto& beams_array = cell_cmd.sym_prb_info()->prbs[cell_cmd.sym_prb_info()->prbs_size-1].beams_array;
                // auto& beams_array_size = cell_cmd.sym_prb_info()->prbs[cell_cmd.sym_prb_info()->prbs_size-1].beams_array_size;
                dciOffsets[i] = dciOffset;
                auto size = update_new_dci(msg_dci, dci, dci_payload);
                update_new_dci_pm(msg_dci, dci, cell_group->get_pm_group(), pm_map, config_option, cell_index);
                dciOffset+= size;
                cset_group.nDcis++;
    #if 0
                uint8_t* tb = dci_payload.data();
                if (tb != nullptr) {
                    for (uint i = 0; i < div_round_up((int)dci.Npayload, 8); ++i) {
                        NVLOGC_FMT("{} PDCCH DCI PAYLOAD[{}] = 0x{:02X}", __FUNCTION__, i, tb[i]);
                    }
                }
    #endif
    
            } else if(nv::TxNotificationHelper::getEnableTxNotification()) {
                update_pdcch_error_contexts(msg_dci, *pdcch_error, i - nDcisToProcess);
                auto& bf = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&msg_dci.payload[0]);
                uint32_t bf_size = bf.num_prgs * sizeof(uint16_t) + bf.num_prgs * bf.dig_bf_interfaces * sizeof(uint16_t) + sizeof(bf);
                auto tx_power_info_size = sizeof(scf_fapi_pdcch_tx_power_info_t);
                auto &dci_end = *reinterpret_cast<scf_fapi_pdcch_dci_payload_t*>(&msg_dci.payload[bf_size + tx_power_info_size]);
                auto payload_size_bits = dci_end.payload_size_bits;
                if (payload_size_bits > CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT) [[unlikely]]
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                        "PDCCH DCI payload_size_bits={} exceeds max {} bits; clamping",
                        static_cast<int>(payload_size_bits), CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT);
                    payload_size_bits = CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT;
                }
                auto dci_payload_len = div_round_up(static_cast<int>(payload_size_bits), 8);
                auto size = (sizeof(scf_fapi_dl_dci_t) + bf_size + tx_power_info_size + sizeof(scf_fapi_pdcch_dci_payload_t) + dci_payload_len);
                dciOffset+= size;
            }
            // dci_group.emplace_back();
        }

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
        ru_type ru = mplane_info.ru;
        update_fh_params_pdcch_per_coreset(cset, dci_group, cell_cmd, cell_params.nPrbDlBwp, pm_grp, msg, dciOffsets, config_option, ru, slot_detail, mplane_info.dl_comp_meth, mmimo_enabled, cell_index);
    }

    void update_pdcch_sym_prb_info_for_pdcch_only_validation(
        cell_sub_command& cell_cmd,
        cuphyPdcchCoresetDynPrm_t& coreset,
        dci_param_list& dci,
        uint16_t bandwidth,
        pm_group* pm_grp,
        scf_fapi_pdcch_pdu_t& msg,
        std::size_t* fapiDciOffsets,
        nv::phy_config_option& config_option,
        nv::slot_detail_t* slot_detail,
        bool mmimo_enabled,
        int32_t cell_index)
    {
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        // getMPlaneConfig throws std::runtime_error for an out-of-range cell_index.
        // This bridge runs on the parse path (noexcept), so catch and skip the
        // fronthaul PRB update gracefully rather than letting the exception escape.
        try {
            auto& mplane_info = phyDriver.getMPlaneConfig(cell_index);
            update_fh_params_pdcch_per_coreset(
                coreset,
                dci,
                cell_cmd,
                bandwidth,
                pm_grp,
                msg,
                fapiDciOffsets,
                config_option,
                mplane_info.ru,
                slot_detail,
                mplane_info.dl_comp_meth,
                mmimo_enabled,
                cell_index);
        } catch (const std::exception& ex) {
            NVLOGW_FMT(TAG,
                       "update_pdcch_sym_prb_info_for_pdcch_only_validation: "
                       "getMPlaneConfig(cell_index={}) failed — skipping fronthaul PRB update: {}",
                       cell_index, ex.what());
        }
    }


    inline void update_new_coreset(cuphyPdcchCoresetDynPrm_t& coreset, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params) {

        uint64_t freqDomainResource = 0ULL;

        for(int i = 0; i < 6; i++) {
            freqDomainResource |= static_cast<uint64_t>(msg.freq_domain_resource[i])<<(56 - (i * 8));
        }

        coreset.n_f = cell_params.nPrbDlBwp * 12;
        coreset.slot_number = slotinfo.slot_;
        coreset.start_rb = msg.bwp.bwp_start; // CB: uncomment once freq domain is figured out
        coreset.start_sym = msg.start_sym_index;
        coreset.n_sym = msg.duration_sym;
        coreset.bundle_size = msg.reg_bundle_size;
        coreset.interleaver_size = msg.interleaver_size;
        coreset.shift_index = msg.shift_index;
        coreset.interleaved = msg.cce_reg_mapping_type;
        coreset.freq_domain_resource = freqDomainResource;

        coreset.coreset_type = msg.coreset_type;
        coreset.nDci = getDciToProcess(msg);
        coreset.testModel = testMode;
        NVLOGD_FMT(TAG,"{}:{} PDCCH testModel={}",__func__,__LINE__,testMode);
    }

    inline std::size_t update_new_dci(scf_fapi_dl_dci_t& msg_dci, cuphyPdcchDciPrm_t& dci, dci_payload_t& payload) {
        dci.rntiCrc  = msg_dci.rnti;
        dci.rntiBits = msg_dci.scrambling_rnti;
        dci.dmrs_id = msg_dci.scrambling_id;
        dci.aggr_level = msg_dci.aggregation_level;
        dci.cce_index = msg_dci.cce_index;

        auto& bf = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&msg_dci.payload[0]);
        uint32_t bf_size = bf.num_prgs * sizeof(uint16_t) + bf.num_prgs * bf.dig_bf_interfaces * sizeof(uint16_t) + sizeof(bf);
        auto& pwr_info = *reinterpret_cast<scf_fapi_pdcch_tx_power_info_t*>(&msg_dci.payload[0] + bf_size);
        #ifdef SCF_FAPI_10_04

            dci.beta_qam = std::pow(10.0, (pwr_info.power_control_offset_ss_profile_nr/20.0));
            dci.beta_dmrs = std::pow(10.0, (pwr_info.power_control_offset_ss_profile_nr/20.0));
        #else
            dci.beta_qam = std::pow(10.0, (pwr_info.power_control_offset_ss - 1)*3.0/20.0);
            dci.beta_dmrs = std::pow(10.0, (pwr_info.power_control_offset_ss - 1)*3.0/20.0);
        #endif

        auto tx_power_info_size = sizeof(scf_fapi_pdcch_tx_power_info_t);
        auto &dci_end = *reinterpret_cast<scf_fapi_pdcch_dci_payload_t*>(&msg_dci.payload[bf_size + tx_power_info_size]);
        dci.Npayload =  dci_end.payload_size_bits;
        if (dci.Npayload > CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT) [[unlikely]]
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                "PDCCH DCI payload_size_bits={} exceeds max {} bits; clamping",
                static_cast<int>(dci.Npayload), CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT);
            dci.Npayload = CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT;
        }
        auto dci_payload_len = div_round_up(static_cast<int>(dci.Npayload), 8);
        memcpy(payload.data(), dci_end.payload, dci_payload_len);
#if 0
        uint8_t* tb = payload.data();
        if (tb != nullptr) {
            for (uint i = 0; i < div_round_up((int)dci.Npayload, 8); ++i) {
                NVLOGI_FMT(TAG,"PDCCH DCI PAYLOAD[{}] = 0x{:02X}", i, tb[i]);
            }
        }
#endif
        return (sizeof(scf_fapi_dl_dci_t) + bf_size + tx_power_info_size + sizeof(scf_fapi_pdcch_dci_payload_t) + dci_payload_len);
    }

    inline void update_new_dci_pm(scf_fapi_dl_dci_t& msg_dci, cuphyPdcchDciPrm_t& dci,  pm_group* prec_group, pm_weight_map_t& pm_map,  nv::phy_config_option& config_options, int32_t cell_index) {
        auto& pdu = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&msg_dci.payload[0]);
        dci.enablePrcdBf = config_options.precoding_enabled;

        auto default_values = [&dci]() {
            dci.enablePrcdBf = false;
        };

        uint16_t offset = 0;

        for (uint16_t i = 0; i < pdu.num_prgs; i++) {
            uint16_t pdu_pmi = pdu.pm_idx_and_beam_idx[i + offset];
            uint32_t cache_pmi = pdu_pmi | cell_index << 16; /// PMI Unused
            dci.enablePrcdBf = dci.enablePrcdBf && (pdu_pmi != 0);
            if (dci.enablePrcdBf) {
                auto pmw_iter = pm_map.find(cache_pmi);
                if (pmw_iter == pm_map.end()){
                    default_values();
                    continue ;
                }

                if (pmw_iter->second.layers != 1) {
                    default_values();
                    continue;
                }

                auto iter = std::find_if(prec_group->pdcch_pmw_idx_cache.begin(), prec_group->pdcch_pmw_idx_cache.end(), [&cache_pmi](const auto& e ) {
                    return e.pmwIdx == cache_pmi;
                });

                if (iter == prec_group->pdcch_pmw_idx_cache.end()) {
                    auto& cache_entry = prec_group->pdcch_pmw_idx_cache.at(prec_group->nCacheEntries);
                    cache_entry.pmwIdx = cache_pmi;
                    cache_entry.nIndex = prec_group->nPmPdcch;
                    dci.pmwPrmIdx = prec_group->nPmPdcch;
                    prec_group->nCacheEntries++;
                    auto& val = prec_group->pdcch_list[prec_group->nPmPdcch];
                    val.nPorts = pmw_iter->second.weights.nPorts;
                    std::copy(pmw_iter->second.weights.matrix, pmw_iter->second.weights.matrix + (pmw_iter->second.layers * pmw_iter->second.ports), val.matrix);
                    prec_group->nPmPdcch++;
                } else {
                    dci.pmwPrmIdx = iter->nIndex;
                }
            }
            offset+=(pdu.dig_bf_interfaces + 1);
        }
    }

    inline void update_fh_params_pdcch(cuphyPdcchCoresetDynPrm_t& coreset, cell_sub_command& cell_cmd, uint16_t bandwidth, enum ru_type ru, nv::slot_detail_t* slot_detail) {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};

        uint16_t coreset_start_prb = coreset.start_rb + __builtin_clzll(coreset.freq_domain_resource) * 6;
        uint16_t coreset_end_prb = coreset.start_rb +(64 - find_rightmost_bit64(coreset.freq_domain_resource)) * 6;
        if(ru == SINGLE_SECT_MODE)
        {
            bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
            if(value)
                return;
            coreset_end_prb   = 273;
            coreset_start_prb = 0;
        }

        uint16_t num_prbs = std::min(static_cast<uint16_t>(coreset_end_prb - coreset_start_prb), bandwidth);
        check_prb_info_size(sym_prbs->prbs_size);
        prbs[sym_prbs->prbs_size] = prb_info_t(coreset_start_prb, num_prbs);
        sym_prbs->prbs_size++;
        std::size_t index{sym_prbs->prbs_size - 1};
        prb_info_t& prb_info{prbs[index]};
        prb_info.common.direction = fh_dir_t::FH_DIR_DL;

        if(ru == SINGLE_SECT_MODE)
        {
            prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_dl_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);
            uint8_t start_symbol = (slot_detail == nullptr ? 0 : slot_detail->start_sym_dl);
            update_prb_sym_list(*sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::PDSCH, ru);
        }
        else
        {
            update_prb_sym_list(*sym_prbs, index, coreset.start_sym, coreset.n_sym, channel_type::PDCCH_DL, ru);
        }
    }

    static constexpr uint16_t DEFAULT_RE_MASK = 0xFFFF;
    static constexpr uint16_t PDCCH_DMRS_RE_MASK = 0x0444;
    static constexpr uint16_t PDCCH_RE_MASK = ~PDCCH_DMRS_RE_MASK & DEFAULT_RE_MASK;

    inline void update_fh_params_pdcch_per_coreset(cuphyPdcchCoresetDynPrm_t& coreset, dci_param_list& dci, cell_sub_command& cell_cmd, uint16_t bandwidth,
                                                   pm_group* pm_grp,  scf_fapi_pdcch_pdu_t& msg, std::size_t* fapiDciOffsets, nv::phy_config_option& config_option, enum ru_type ru,
                                                   nv::slot_detail_t* slot_detail, comp_method dl_comp_method, bool mmimo_enabled, int32_t cell_index)
    {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};

        uint16_t startRB = 0, num_prbs = 0;

        if(ru == SINGLE_SECT_MODE)
        {
            bool value = ifAnySymbolPresent(sym_prbs->symbols, DL_CHANNEL_MASK);
            if(value)
                return;
            num_prbs = bandwidth;
            startRB  = 0;
            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(startRB, num_prbs);
            sym_prbs->prbs_size++;

            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};

            uint8_t start_symbol = (slot_detail == nullptr ? 0: slot_detail->start_sym_dl);
            
            prb_info.common.direction = fh_dir_t::FH_DIR_DL;
            prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_dl_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_dl_symbols);

            update_prb_sym_list(*sym_prbs, index, start_symbol, prb_info.common.numSymbols, channel_type::PDSCH, ru);
        }
        else
        {
            uint64_t coreset_map; /*Used as bitmask. Shifted version of freq_domain_resource */
            uint32_t rb_coreset;  /*Indicates the number of bits in coreset_map to be considered. It is # RBs divided by 6. */
            uint32_t n_CCE;

            rb_coreset  = 64 - find_rightmost_bit64(coreset.freq_domain_resource);
            coreset_map = (coreset.freq_domain_resource >> (64 - rb_coreset));
            n_CCE       = count_set_bits(coreset.freq_domain_resource) * coreset.n_sym;

            uint32_t bundles_per_coreset_bit = 6 * coreset.n_sym / coreset.bundle_size;
            uint32_t N_bundle                = n_CCE * bundles_per_coreset_bit / coreset.n_sym; // N_bundle counts all the set bits in coreset map (specific to particular DCI)
            uint32_t N_bundle_phy            = rb_coreset * bundles_per_coreset_bit;            // N_bundle_phy counts all of available bundle in a particular coreset

            uint32_t bundle_table[MAX_N_BUNDLE] = {0};                                          // Maps logical bundle ID to physical bundle ID
            uint32_t bundle_map[MAX_N_BUNDLE]   = {0};                                          // Maps old logical bundle ID to new logical bundle ID

            int log_bundle_id = 0;
            int phy_bundle_id = 0;

            for(int i = 0; i < rb_coreset; i++)
            {
                if((coreset_map >> (rb_coreset - i - 1)) & 0x1)
                {
                    for(int j = 0; j < bundles_per_coreset_bit; j++)
                    {
                        bundle_table[log_bundle_id + j] = phy_bundle_id + j;
                        NVLOGD_FMT(TAG, "Logical bundle {} maps to physical bundle {}", log_bundle_id + j, phy_bundle_id + j);
                    }
                    log_bundle_id += bundles_per_coreset_bit;
                }
                phy_bundle_id += bundles_per_coreset_bit;
            }

            uint32_t C = (coreset.interleaved) ? n_CCE * 6 / (coreset.bundle_size * coreset.interleaver_size) : 1;

            for(int i = 0; i < N_bundle; i++)
            {
                uint32_t new_bundle_id = i;
                if(coreset.interleaved)
                {
                    uint32_t c    = i / coreset.interleaver_size;
                    uint32_t r    = i % coreset.interleaver_size;
                    new_bundle_id = (r * C + c + coreset.shift_index) % N_bundle;
                    NVLOGD_FMT(TAG, "Logical bundle {} maps to new logical bundle {} in interleaved mode", i, new_bundle_id);
                }
                bundle_map[i] = new_bundle_id;
            }

            int      num_DCIs                      = coreset.nDci;
            uint32_t used_bundle_map[N_bundle_phy] = {0}; //Map of physical bundle used
            uint32_t used_bundle_dci_map[N_bundle_phy]{0};
            for(int i = 0; i < num_DCIs; i++)
            {
                cuphyPdcchDciPrm_t dci_params = dci[i + coreset.dciStartIdx];

                for(int j = 0; j < dci_params.aggr_level; j++)
                {
                    for(int used_bundle = 0; used_bundle < 6 / coreset.bundle_size; used_bundle++)
                    {
                        uint32_t log_bundle_id                           = bundle_map[(6 / coreset.bundle_size) * (dci_params.cce_index + j) + used_bundle];
                        used_bundle_map[bundle_table[log_bundle_id]]     = 1;
                        used_bundle_dci_map[bundle_table[log_bundle_id]] = i;
                        NVLOGD_FMT(TAG, "DCI {}: physical bundle {} is used, orig. logical bundle {}, new logical bundle  {}.", i, bundle_table[log_bundle_id], (6 / coreset.bundle_size) * (dci_params.cce_index + j) + used_bundle, log_bundle_id);
                        // What does it mean to say "physical bundle X" is used?
                        // It coresponds to a bit of coreset_map * 6 RBs * n_sym / bundleSize
                        NVLOGD_FMT(TAG, "It will occupy REs from: [{} to {}) per symbol", 12 * (coreset.start_rb + bundle_table[log_bundle_id] * coreset.bundle_size / coreset.n_sym), 12 * (coreset.start_rb + (bundle_table[log_bundle_id] + 1) * coreset.bundle_size / coreset.n_sym));
                    }
                }
            }

            int      count       = 0;
            uint8_t* dldci_buf   = reinterpret_cast<uint8_t*>(msg.dl_dci);
            uint32_t current_dci = UINT32_MAX;

            auto new_prb_group = [&coreset, &dci, sym_prbs, &prbs, pm_grp, dldci_buf, fapiDciOffsets, &config_option, &ru, mmimo_enabled, cell_index](auto& startRB, auto& count, auto& current_dci, channel_type ch_type, uint16_t reMask) {
                uint16_t num_prbs = (coreset.bundle_size / coreset.n_sym) * count;
                //NVLOGD_FMT(TAG, "Start PRB = {}, Number of PRBs = {}, prb-size = {} current_dci = {}", startRB, num_prbs, sym_prbs->prbs_size, current_dci);
                check_prb_info_size(sym_prbs->prbs_size);
                prbs[sym_prbs->prbs_size] = prb_info_t(startRB, num_prbs);
                sym_prbs->prbs_size++;
                std::size_t index{sym_prbs->prbs_size - 1};
                prb_info_t& prb_info{prbs[index]};
                prb_info.common.reMask = reMask;

                prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                update_prb_sym_list(*sym_prbs, index, coreset.start_sym, coreset.n_sym, ch_type, ru);
                auto& msg_dci          = *reinterpret_cast<scf_fapi_dl_dci_t*>(dldci_buf + fapiDciOffsets[current_dci]);
                auto& beams_array      = prb_info.beams_array;
                auto& beams_array_size = prb_info.beams_array_size;
                auto  size             = update_per_dci_beams(msg_dci, beams_array, beams_array_size, config_option.bf_enabled, prb_info, mmimo_enabled, cell_index);
                auto& bf               = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&msg_dci.payload[0]);

                if(config_option.precoding_enabled && dci[current_dci].enablePrcdBf)
                {
                    prb_info.common.portMask     = (1 << pm_grp->pdcch_list[dci[current_dci].pmwPrmIdx].nPorts) - 1;
                    prb_info.common.numApIndices = pm_grp->pdcch_list[dci[current_dci].pmwPrmIdx].nPorts;
                }
                else if(config_option.bf_enabled)
                {
                    prb_info.common.portMask     = (1 << bf.dig_bf_interfaces) - 1;
                    prb_info.common.numApIndices = bf.dig_bf_interfaces;
                }
                else
                {
                    prb_info.common.portMask     = 1;
                    prb_info.common.numApIndices = 1;
                }
            };

            for(int j = 0; j < N_bundle_phy; j++)
            {
                //NVLOGI_FMT(TAG, "used_bundle_map[{}] = {}, used_bundle_dci_map[{}] = {}, current_dci = {}", j,used_bundle_map[j], j,used_bundle_dci_map[j], current_dci);
                if(used_bundle_map[j] == 1)
                {
                    if(count == 0)
                    {
                        startRB     = (j * coreset.bundle_size / coreset.n_sym) + coreset.start_rb;
                        current_dci = used_bundle_dci_map[j];
                    }
                    else if(used_bundle_dci_map[j] != current_dci)
                    {
                        // Decide whether to split the PRB group at a DCI boundary.
                        //
                        // A split is required whenever the new DCI differs from the current
                        // PRB group in any of:
                        //   - dig_bf_interfaces / beam IDs  (otherwise SE-11 would stamp the
                        //     same static_beam_id on multiple bundles -> RU emulator rejects
                        //     "static beam id : N seen before!")
                        //   - enablePrcdBf flag
                        //   - pmwPrmIdx (precoding matrix index) when precoding is enabled
                        //     (otherwise the merged group would silently apply the first
                        //     DCI's PM weights to the second DCI's PRBs)
                        //
                        // Compare against the NEW DCI (used_bundle_dci_map[j]); the previous
                        // version of this code compared against current_dci which is the
                        // same DCI that already populated prb_info.beams_array, so the
                        // comparison was a self-compare and never produced a split.
                        bool needSplit = true;
                        uint32_t next_dci  = used_bundle_dci_map[j];
                        auto& msg_dci_next = *reinterpret_cast<scf_fapi_dl_dci_t*>(dldci_buf + fapiDciOffsets[next_dci]);
                        auto& bf_next      = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&msg_dci_next.payload[0]);
                        bool  beamsPresent = config_option.bf_enabled && (bf_next.dig_bf_interfaces != 0);
                        // When beamforming is absent, treat beams as "matching" so the
                        // split decision falls through to PMI alone. This preserves the
                        // pre-existing merge behavior for non-BF configurations.
                        bool  beamsMatch   = !beamsPresent;
                        if(beamsPresent && sym_prbs->prbs_size > 0)
                        {
                            uint16_t*   beam_id_list = (bf_next.pm_idx_and_beam_idx + 1);
                            std::size_t index{sym_prbs->prbs_size - 1};
                            prb_info_t& prb_info{prbs[index]};
                            // memcmp returns 0 when buffers match, non-zero when they differ.
                            // Pass byte length (num_elements * sizeof(uint16_t)).
                            if(prb_info.beams_array_size == bf_next.dig_bf_interfaces &&
                               std::memcmp(prb_info.beams_array.data(), beam_id_list,
                                           prb_info.beams_array_size * sizeof(uint16_t)) == 0)
                            {
                                beamsMatch = true;
                            }
                        }
                        bool prevPrcdBf = config_option.precoding_enabled && dci[current_dci].enablePrcdBf;
                        bool nextPrcdBf = config_option.precoding_enabled && dci[next_dci].enablePrcdBf;
                        bool pmiMatch   = (prevPrcdBf == nextPrcdBf) &&
                                          (!prevPrcdBf || dci[current_dci].pmwPrmIdx == dci[next_dci].pmwPrmIdx);
                        if(beamsMatch && pmiMatch)
                        {
                            needSplit = false;
                        }
                        if(needSplit)
                        {
                            if (dl_comp_method != comp_method::MODULATION_COMPRESSION) {
                                new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, DEFAULT_RE_MASK);
                            }
                            else {
                                new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, std::numeric_limits<uint16_t>::max());
                                std::size_t index{sym_prbs->prbs_size - 1};
                                prb_info_t& prb{prbs[index]};
                                update_mod_comp_info_common(prb, getBwScaler(bandwidth));
                                update_mod_comp_info_pdcch_section(channel_type::PDCCH_DMRS, prb, dci[current_dci]);
                                update_mod_comp_info_pdcch_section(channel_type::PDCCH_DL, prb, dci[current_dci]);
                            }
                            current_dci = used_bundle_dci_map[j];
                            startRB     = (j * coreset.bundle_size / coreset.n_sym) + coreset.start_rb;
                            count       = 0;
                        }
                    }
                    count++;
                }
                else
                {
                    if(count)
                    {
                        // num_prbs = (coreset.bundle_size / coreset.n_sym) * count;
                        // NVLOGD_FMT(TAG, "Start PRB = {}, Number of PRBs = {}, prb-size = {}", startRB, num_prbs, sym_prbs->prbs_size);
                        // prbs[sym_prbs->prbs_size] = prb_info_t(startRB, num_prbs);
                        // sym_prbs->prbs_size++;
                        // std::size_t index{sym_prbs->prbs_size - 1};
                        // prb_info_t& prb_info{prbs[index]};

                        // prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                        // update_prb_sym_list(*sym_prbs, index, coreset.start_sym, coreset.n_sym, channel_type::PDCCH_DL);
                        current_dci = used_bundle_dci_map[j];
                        if (dl_comp_method != comp_method::MODULATION_COMPRESSION) {
                            new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, DEFAULT_RE_MASK);
                        } 
                        else {
                                new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, std::numeric_limits<uint16_t>::max());
                                std::size_t index{sym_prbs->prbs_size - 1};                                
                                prb_info_t& prb{prbs[index]};
                                update_mod_comp_info_common(prb, getBwScaler(bandwidth));
                                update_mod_comp_info_pdcch_section(channel_type::PDCCH_DMRS, prb, dci[current_dci]);
                                update_mod_comp_info_pdcch_section(channel_type::PDCCH_DL, prb, dci[current_dci]);
                        }
                    }
                    count = 0;
                }
            }

            if(count)
            {
                // num_prbs = (coreset.bundle_size / coreset.n_sym) * count;
                // NVLOGD_FMT(TAG, "Start PRB = {}, Number of PRBs = {}, prb-size = {}", startRB, num_prbs, sym_prbs->prbs_size);
                // prbs[sym_prbs->prbs_size] = prb_info_t(startRB, num_prbs);
                // sym_prbs->prbs_size++;
                // std::size_t index{sym_prbs->prbs_size - 1};
                // prb_info_t& prb_info{prbs[index]};

                // prb_info.common.direction = fh_dir_t::FH_DIR_DL;
                // update_prb_sym_list(*sym_prbs, index, coreset.start_sym, coreset.n_sym, channel_type::PDCCH_DL);
                current_dci = used_bundle_dci_map[N_bundle_phy - 1];
                // new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, DEFAULT_RE_MASK);
                if (dl_comp_method != comp_method::MODULATION_COMPRESSION) {
                    new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, DEFAULT_RE_MASK);
                } 
                else {
                    new_prb_group(startRB, count, current_dci, channel_type::PDCCH_DL, std::numeric_limits<uint16_t>::max());
                    std::size_t index{sym_prbs->prbs_size - 1};
                    prb_info_t& prb{prbs[index]};
                    update_mod_comp_info_common(prb, getBwScaler(bandwidth));
                    update_mod_comp_info_pdcch_section(channel_type::PDCCH_DMRS, prb, dci[current_dci]);
                    update_mod_comp_info_pdcch_section(channel_type::PDCCH_DL, prb, dci[current_dci]);
                }
            }
        }
    }

    inline std::size_t update_per_dci_beams(scf_fapi_dl_dci_t& msg_dci,  beamid_array_t& array, size_t& array_size, bool bf_enabled, prb_info_t& prb_info, bool mmimo_enabled, int32_t cell_index) {

        auto& bf = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(&msg_dci.payload[0]);
        uint32_t bf_size = bf.num_prgs * sizeof(uint16_t) + bf.num_prgs * bf.dig_bf_interfaces * sizeof(uint16_t) + sizeof(bf);
        auto& pwr_info = *reinterpret_cast<scf_fapi_pdcch_tx_power_info_t*>(&msg_dci.payload[0] + bf_size);
        auto tx_power_info_size = sizeof(scf_fapi_pdcch_tx_power_info_t);
        auto &dci_end = *reinterpret_cast<scf_fapi_pdcch_dci_payload_t*>(&msg_dci.payload[bf_size + tx_power_info_size]);
        auto payload_size_bits = dci_end.payload_size_bits;
        if (payload_size_bits > CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT) [[unlikely]]
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                "PDCCH DCI payload_size_bits={} exceeds max {} bits; clamping",
                static_cast<int>(payload_size_bits), CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT);
            payload_size_bits = CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES * CHAR_BIT;
        }
        auto dci_payload_len = div_round_up(static_cast<int>(payload_size_bits), 8);
        if (bf_enabled) {
             update_beam_list_uniq(array, array_size, bf, prb_info, mmimo_enabled, cell_index);
        }
        return (sizeof(scf_fapi_dl_dci_t) + bf_size + tx_power_info_size + sizeof(scf_fapi_pdcch_dci_payload_t) + dci_payload_len);
    }

    inline void update_mod_comp_info_pdcch_section(channel_type type, prb_info_t& prb, cuphyPdcchDciPrm_t& dci) {
        switch (type) {
            case channel_type::PDCCH_DMRS:
                update_mod_comp_info_section(prb, PDCCH_DMRS_RE_MASK, dci.beta_dmrs, CUPHY_QAM_4, DEFAULT_CSF);
                break;
            case channel_type::PDCCH_DL:
            case channel_type::PDCCH_UL:
                update_mod_comp_info_section(prb, PDCCH_RE_MASK, dci.beta_qam, CUPHY_QAM_4, DEFAULT_CSF);
                break;
            default:
                break;
        }
    }
}
