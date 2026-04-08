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

#include "scf_5g_slot_commands.hpp"
#include "scf_5g_fapi_phy.hpp"
#include "cuphy_internal.h"
#include "cuphy.h"
#include "memtrace.h"
#include <algorithm>
#include <math.h>
using namespace slot_command_api;

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 4) // "SCF.SLOTCMD"
#define CSI_RS_START_AP_INDEX 3000
#define ROUND_UP(x, y) 1 + ((x - 1) / y);

namespace scf_5g_fapi
{
    // Tables containng Bit mask for additional dmrs (puschDuration x nAddlDmrs). Type A allocation.
    static constexpr uint16_t ADDLDMRSBMSK_MAXLENGTH1[15][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 128, 128, 128}, {0, 128, 128, 128}, {0, 512, 576, 576}, {0, 512, 576, 576}, {0, 512, 576, 2336}, {0, 2048, 2176, 2336}, {0, 2048, 2176, 2336}};
    static constexpr uint16_t ADDLDMRSBMSK_MAXLENGTH2[15][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 768}, {0, 768}, {0, 768}, {0, 3072}, {0, 3072}};
    inline uint16_t computeDmrsSymMaskTypeA(uint8_t firstUlDmrsPos, uint8_t dmrsAddlnPos, uint8_t nPuschSym, uint8_t dmrsMaxLen);
    static constexpr uint32_t PRACH_LRA_139 = 139;
    static constexpr uint32_t PRACH_LRA_839 = 839;

    static constexpr uint32_t KHZ = 1000;
    static constexpr uint16_t N_CS_DFRA_1250[16][3] = {
                                                        { 0, 15, 15},
                                                        {13, 18, 18},
                                                        {15, 22, 28},
                                                        {18, 26, 26},
                                                        {22, 32, 32},
                                                        {26, 38, 38},
                                                        {32, 46, 46},
                                                        {38, 55, 55},
                                                        {46, 68, 68},
                                                        {59, 82, 82},
                                                        {76, 100, 100},
                                                        {93, 128, 118},
                                                        {119, 158, 137},
                                                        {167, 202, 0xFFFF},
                                                        {279, 237, 0xFFFF},
                                                        {419, 0xFFFF, 0xFFFF}
                                                      };
    static constexpr uint16_t N_CS_DFRA_5000[16][3] = {
                                                        {0, 36, 36},
                                                        {13, 57, 57},
                                                        {26, 72, 60},
                                                        {33, 81, 63},
                                                        {38, 89, 65},
                                                        {41, 94, 68},
                                                        {49, 103, 71},
                                                        {55, 112, 77},
                                                        {64, 121, 81},
                                                        {76, 132, 85},
                                                        {93, 137, 97},
                                                        {119, 152, 109},
                                                        {139, 173, 122},
                                                        {209, 195, 137},
                                                        {279, 216, 0xFFFF},
                                                        {419, 237, 0xFFFF}
                                                      };

    static constexpr uint16_t N_CS_DFRA_MU[16] = {0, 2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 23, 27, 34, 46, 69};

    const uint16_t UL_NON_PRACH_CHANNEL_MASK = (1 << channel_type::PUSCH) |
                                               (1 << channel_type::PUCCH);

    const uint16_t SRS_CHANNEL_MASK = (1 << channel_type::SRS);


#ifndef ENABLE_L2_SLT_RSP
    static int32_t pusch_ue_idx = channel_type::NONE;
    static int32_t pdsch_ue_idx = channel_type::NONE;
    static int32_t pdsch_cw_idx = 0;
    static int32_t pdcch_dl_idx = channel_type::NONE;
    static int32_t pdcch_ul_idx = channel_type::NONE;
    static uint32_t pucch_ue_idx = 0;
#endif
    static bool cell_reset[MAX_CELLS_PER_SLOT] = {false};
    static bool cell_grp_reset = false;
    static constexpr int MAX_MIB_BITS = 24;

    static constexpr uint8_t srs_ant_idx_to_port[]={1,2,4};
    static constexpr uint8_t srs_symb_idx_to_numSymb[]={1,2,4};
    static constexpr uint8_t srs_rep_factor_idx_to_numRepFactor[]={1,2,4};
    static constexpr uint8_t srs_comb_idx_to_combSize[]={2,4,8};

    inline void update_fh_params_pusch(cuphyPuschUeGrpPrm_t& grp, bool is_new_grp, cuphyPuschUePrm_t& ue, const scf_fapi_rx_beamforming_t& pmi_bf_pdu, const scf_fapi_pusch_pdu_t& msg, cell_sub_command& cell_cmd, bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool bf_enabled = false, enum ru_type ru = OTHER_MODE, bool mmimo_enabled=0, nv::slot_detail_t* slot_detail = nullptr, uint32_t bfwUeGrpIndex = 0, int32_t cell_index = 0, uint16_t ul_bandwidth = MAX_N_PRBS_SUPPORTED);
    inline void update_fh_params_pucch(cuphyPucchUciPrm_t& uci_info, uint16_t prb_size, const scf_fapi_rx_beamforming_t& pmi_bf_pdu, cell_sub_command& cmd, bool bf_enabled = false, enum ru_type ru = OTHER_MODE, nv::slot_detail_t* slot_detail = nullptr, bool mmimo_enabled=0, int32_t cell_index = 0, uint16_t ul_bandwidth = MAX_N_PRBS_SUPPORTED);
    inline void update_fh_params_prach(nv::phy_config& config, nv::prach_addln_config_t& addln_config, const scf_fapi_prach_pdu_t& pdu, const scf_fapi_rx_beamforming_t & bf_pdu, cell_sub_command& cell_cmd, bool bf_enabled = false, enum ru_type ru = OTHER_MODE, nv::slot_detail_t* slot_detail = nullptr, bool mmimo_enabled=0, int32_t cell_index = 0);
    // finalize_srs_slot is now declared in the header file
    inline void update_new_coreset(cuphyPdcchCoresetDynPrm_t& coreset, scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, slot_indication & slotinfo, cuphyCellStatPrm_t& cell_params);
    inline bool ifAnySymbolPresent(sym_info_list_t& symbols, uint16_t channelMask);
    inline int8_t calc_start_prb(const scf_fapi_srs_pdu_t& msg, cell_sub_command& cell_cmd, srs_rb_info_t srs_rb_info[], uint8_t& nHops);
    inline void merge_srs_prb_interval(srs_rb_info_per_sym_t& interval, srs_rb_info_per_sym_t& final_intervals);

    static inline void to_u8_array(uint32_t bchPayload, uint8_t* mib)
    {
        for(int i = 0; i < MAX_MIB_BITS / 8; i++)
        {
            mib[i] = bchPayload >> ((MAX_MIB_BITS / 8 - i - 1) * 8);
        }
    }

    inline uint8_t get_addl_pos(uint8_t dmrsType, uint8_t numSym, uint16_t dmrsSymLocBmsk, uint8_t dmrsMaxLength)
    {
        uint8_t dmrsAddlPosition = 0;
        int bit1_count = __builtin_popcount(dmrsSymLocBmsk) / dmrsMaxLength;
        dmrsAddlPosition = bit1_count - 1;
        return dmrsAddlPosition;
    }

    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_pusch_pdu_t& msg, int32_t cell_index, slot_indication & slotinfo, int staticPuschSlotNum, uint8_t lbrm, bool bf_enabled, uint16_t cell_stat_prm_idx, float dtx_threshold, bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool mmimo_enabled, nv::slot_detail_t* slot_detail, uint16_t ul_bandwidth) {

        cell_sub_cmd.slot.type = SLOT_UPLINK;
        cell_sub_cmd.slot.slot_3gpp = slotinfo;
        cell_grp_cmd->slot.type = SLOT_UPLINK;
        cell_grp_cmd->slot.slot_3gpp = slotinfo;

        cell_grp_cmd->create_if(channel_type::PUSCH);
        pusch_params* info = cell_grp_cmd->pusch.get();
        if (info == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no pusch command");
            return;
        }

        bool newCell = false;
        auto it = std::find(info->cell_index_list.begin(), info->cell_index_list.end(), cell_index);
        if(it == info->cell_index_list.end())
        {
            info->cell_index_list.push_back(cell_index);
            info->phy_cell_index_list.push_back(cell_sub_cmd.cell);
            info->cell_ue_group_idx_start[cell_index] = info->cell_grp_info.nUeGrps;
            ++info->cell_grp_info.nCells;
            newCell = true;
        }

        //Update Cell Dyn paramters
        if(newCell)
        {
            auto& cell_info = info->cell_dyn_info[info->cell_grp_info.nCells-1];
            cell_info.slotNum = ((staticPuschSlotNum > -1)? staticPuschSlotNum : cell_grp_cmd->slot.slot_3gpp.slot_);
            cell_info.cellPrmDynIdx     = info->cell_grp_info.nCells-1;
            cell_info.cellPrmStatIdx    = cell_stat_prm_idx;
            //NVLOGD_FMT(TAG, "PUSCH new cell cell_index {} cellPrmStatIdx={} cellPrmDynIdx={}", cell_index, cell_stat_prm_idx,  cell_info.cellPrmDynIdx);
        }

        auto pusch_ue_idx = info->cell_grp_info.nUes;
        info->scf_ul_tti_handle_list.push_back(msg.handle);
        auto&  ue = info->ue_info[pusch_ue_idx];

        ue.puschIdentity    = msg.pusch_identity;
        ue.scid             = msg.scid;
        ue.dmrsPortBmsk     = msg.dmrs_ports;
        ue.mcsTableIndex    = msg.mcs_table;
        ue.mcsIndex         = msg.mcs_index;
        ue.rnti             = msg.rnti;
        ue.dataScramId      = msg.data_scrambling_id;
        ue.nUeLayers        = msg.num_of_layers;
        ue.targetCodeRate   = msg.target_code_rate;
        ue.pUciPrms         = nullptr;
        ue.qamModOrder      = msg.qam_mod_order;
        // TODO FIXME: Temp LBRM Support - Disable LBRM
        ue.i_lbrm = lbrm;
        // See 28.212 5.4.2.1 for details on the below LBRM parameters
        //ue.maxLayer = <set maxLayer from L2>
        //ue.maxQm = <set maxQm from L2>
        //ue.n_PRB_LBRM = <set n_PRB_LBRM from L2>
        if (ue.i_lbrm)
        {
            ue.maxLayers = 4;
            if(ue.mcsTableIndex == 1)
            {
                ue.maxQm = 8;
            }
            else
            {
                ue.maxQm = 6;
            }
            ue.n_PRB_LBRM = compute_N_prb_lbrm(msg.bwp.bwp_size);
        }

        auto found = [&msg] (const auto& e) {
            return (e.puschStartSym == msg.start_symbol_index &&
                    e.nPuschSym == msg.num_of_symbols &&
                    e.startPrb == (msg.bwp.bwp_start + msg.rb_start) &&
                    e.nPrb ==  msg.rb_size);
        };

        auto iter = std::find_if(info->ue_grp_info + info->cell_ue_group_idx_start[cell_index], info->ue_grp_info + MAX_PUSCH_UE_GROUPS, found);
        std::size_t ueGrpIndex;
        uint32_t bfwUeGrpIndex;
        bool newUeGrp = false;
        if (iter != std::end(info->ue_grp_info))
        {
            ueGrpIndex = std::distance(info->ue_grp_info, iter);
        }
        else
        {
            ueGrpIndex = info->cell_grp_info.nUeGrps;
            ++info->cell_grp_info.nUeGrps;
            bfwUeGrpIndex = info->nue_grps_per_cell[cell_index];
            info->nue_grps_per_cell[cell_index]++;
            newUeGrp = true;
        }
#if 0
        if(mmimo_enabled && (is_latest_bfw_coff_avail(cell_sub_cmd.slot.slot_3gpp.sfn_ , cell_sub_cmd.slot.slot_3gpp.slot_, bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)))
        {
            if((bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pusch_ue_idx].pdu_idx != pusch_ue_idx) || (bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pusch_ue_idx].rnti != msg.rnti))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,"PUSCH Index or RNTI mismatch between CVI_REQ & UL_TTI ueGrpIndex={} slotIndex={} BFW Idx={} UL_TTI Idx={} UL CVI RNTI={} PDSCH RNTI={}",
                                ueGrpIndex, bfwCoeff_mem_info->slotIndex, static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pusch_ue_idx].pdu_idx), 
                                static_cast<uint16_t>(pusch_ue_idx), static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pusch_ue_idx].rnti), static_cast<uint16_t>(msg.rnti));
                return;
            }
            else
            {
                NVLOGD_FMT(TAG,"ueGrpIndex={} slotIndex={} BFW Idx={} UL_TTI Idx={} UL CVI RNTI={} PUSCH RNTI={}",ueGrpIndex, bfwCoeff_mem_info->slotIndex,
                                static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pusch_ue_idx].pdu_idx), static_cast<uint16_t>(pusch_ue_idx),
                                static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[ueGrpIndex][pusch_ue_idx].rnti), static_cast<uint16_t>(msg.rnti));
            }
        }
#endif
        ue.ueGrpIdx = ueGrpIndex;
        ue.pUeGrpPrm = &info->ue_grp_info[ueGrpIndex];
        
        // Default value, will be updated if extension is present
        ue.foForgetCoeff = 0.0f; 
        ue.ldpcEarlyTerminationPerUe = 0;
        ue.ldpcMaxNumItrPerUe = 10;

        // update UE Group - no group present use 0 as default
        auto& ue_grp = info->ue_grp_info[ueGrpIndex];
        ue_grp.pUePrmIdxs[ue_grp.nUes] = pusch_ue_idx;
        ue_grp.nUes++;

        // Update DMRS Info
        auto& ue_dmrs = info->ue_dmrs_info[ueGrpIndex];
        ue_dmrs.dmrsMaxLen          = ((msg.ul_dmrs_sym_pos & (msg.ul_dmrs_sym_pos >> 1)) > 0) ? 2 : 1;
        ue_dmrs.dmrsAddlnPos        = get_addl_pos(msg.dmrs_config_type, msg.num_of_symbols, msg.ul_dmrs_sym_pos, ue_dmrs.dmrsMaxLen);
        ue_dmrs.nDmrsCdmGrpsNoData  = msg.num_dmrs_cdm_groups_no_data;
        ue_dmrs.dmrsScrmId          = msg.ul_dmrs_scrambling_id;

        if (newUeGrp)
        {
            ue_grp.puschStartSym = msg.start_symbol_index;
            ue_grp.nPuschSym = msg.num_of_symbols;
            // TODO do we want to update RAT0 for PUSCH while we're here?
            ue_grp.startPrb = msg.bwp.bwp_start + msg.rb_start;
            ue_grp.nPrb     = msg.rb_size;
            ue_grp.dmrsSymLocBmsk = msg.ul_dmrs_sym_pos;
            ue_grp.rssiSymLocBmsk = ue_grp.dmrsSymLocBmsk;
            ue_grp.pDmrsDynPrm = &ue_dmrs;
            ue_grp.pCellPrm = &info->cell_dyn_info[info->cell_grp_info.nCells-1];
            //NVLOGD_FMT(TAG, "PUSCH pCellPrm indexing into cell_dyn_info[{}]", info->cell_grp_info.nCells-1);
        }

        auto& cell_grp_info         = info->cell_grp_info;
        cell_grp_info.nUes++;

        const uint8_t* next = &msg.payload[0];
        ue.pduBitmap = msg.pdu_bitmap;
        // Data expected on PUSCH
        if (msg.pdu_bitmap & 0x1) {
            auto data = reinterpret_cast<const scf_fapi_pusch_data_t*>(next);
            ue.rv = data->rv_index;
            info->ue_tb_size[pusch_ue_idx] = data->tb_size;
            ue.TBSize = data->tb_size;
            ue.ndi = data->new_data_indicator;
            ue.harqProcessId = data->harq_process_id;
            next += sizeof(scf_fapi_pusch_data_t);
        }
#ifdef SCF_FAPI_10_04
        auto isCsiP2Signaled = false;
#endif 

        // PUSCH UCI Data
        if (msg.pdu_bitmap & 0x2) {
            auto data = reinterpret_cast<const scf_fapi_pusch_uci_t*>(next);
            ue.pUciPrms = &info->uci_info[pusch_ue_idx];
            auto uci_prms = ue.pUciPrms;
            uci_prms->nBitsHarq = data->harq_ack_bit_length;
            uci_prms->nBitsCsi1 = data->csi_part_1_bit_length;
            uci_prms->alphaScaling = data->alpha_scaling;
            uci_prms->betaOffsetHarqAck = data->beta_offset_harq_ack;
            uci_prms->betaOffsetCsi1 = data->beta_offset_csi_1;
            //Change for CSI part 2
            uci_prms->betaOffsetCsi2 = data->beta_offset_csi_2;

#ifdef SCF_FAPI_10_04
            uci_prms->nRanksBits = std::numeric_limits<uint8_t>::max();
            isCsiP2Signaled = (data->flag_csi_part2 == std::numeric_limits<uint16_t>::max());
#else
            if((data->csi_part_2_bit_length > 0) && (data->csi_part_2_bit_length != 255) && (data->csi_part_2_bit_length < 1707)) // check valid range of csi_part_2_bit_length
            {
                uci_prms->rankBitOffset = (data->csi_part_2_bit_length & 0xFF);
                uci_prms->nRanksBits = ((data->csi_part_2_bit_length >>8) & 0xFF);
                if ((uci_prms->nRanksBits > 4)||(uci_prms->rankBitOffset > 47)) //TODO for uci_prms->nRanksBits > 4
                    uci_prms->nRanksBits = 255;
            }
            else
            {
                uci_prms->nRanksBits = 255;
            }
#endif
            uci_prms->nCsiReports = 1;
            uci_prms->DTXthreshold = dtx_threshold;
            if(uci_prms->nRanksBits != 255)
            {
                ue.pduBitmap |= 0x20;
            }
            next += sizeof(scf_fapi_pusch_uci_t);
        }

        // PUSCH DFT-s-OFDM
        if(msg.transform_precoding==0)
        {
            ue.enableTfPrcd = 1;
            if (msg.pdu_bitmap & 0x8)
            {
                auto dftsofdm = reinterpret_cast<const scf_fapi_pusch_dftsofdm_t*>(next);
                ue.lowPaprGroupNumber = dftsofdm->lowPaprGroupNumber;
                ue.lowPaprSequenceNumber = dftsofdm->lowPaprSequenceNumber;
                ue.groupOrSequenceHopping = 0; //no use
                next += sizeof(scf_fapi_pusch_dftsofdm_t);
            }
        }
        else
        {
            ue.enableTfPrcd = 0;
        }

        // BEAMFORMING CONFIGS
        const auto& bf = *reinterpret_cast<const scf_fapi_rx_beamforming_t*>(next);
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
        ru_type ru = mplane_info.ru;
        if(mmimo_enabled && bf.dig_bf_interfaces != 0)
        {
            ue_grp.nUplinkStreams = bf.dig_bf_interfaces;
        }
        else if(mmimo_enabled && bf.dig_bf_interfaces == 0)
        {
            ue_grp.nUplinkStreams += ue.nUeLayers;
        }
        
        ue_grp.prgSize = bf.prg_size;
        
        if((mmimo_enabled) && ((ue_grp.prgSize==1) || (ue_grp.prgSize==2) || (ue_grp.prgSize==3) || (ue_grp.prgSize==4)))
        {
            ue_grp.enablePerPrgChEstPerUeg = 1;
        }
        else
        {
            ue_grp.enablePerPrgChEstPerUeg = 0;
        }
        
        update_fh_params_pusch(ue_grp, newUeGrp, ue, bf, msg, cell_sub_cmd, bfwCoeff_mem_info, bf_enabled, ru, mmimo_enabled, slot_detail,bfwUeGrpIndex, cell_index, ul_bandwidth);
        if (mmimo_enabled == 0 || bf.dig_bf_interfaces!= 0)
        {
            next += sizeof(scf_fapi_rx_beamforming_t) + sizeof(uint16_t) * bf.dig_bf_interfaces; //PUSCH only supports numPRGs = 1 now.
            if (mmimo_enabled)
            {
                bfwCoeff_mem_info = NULL;
            }
        }
        else
        {
            next += sizeof(scf_fapi_rx_beamforming_t);
        }

        // PUSCH Maintenance Parameters
    #ifdef SCF_FAPI_10_04
        auto maintenance = reinterpret_cast<const scf_fapi_pusch_maintenance_t*>(next);
        next += sizeof(scf_fapi_pusch_maintenance_t);

        // Defualts for CSIP2

        auto pUciPrms = ue.pUciPrms;
        if(pUciPrms != nullptr)
        {
            pUciPrms->nCsi2Reports = 0;
            pUciPrms->pCalcCsi2SizePrms = nullptr;
        }

        if (isCsiP2Signaled && msg.pdu_bitmap & 0x2) {
            auto csip2_info = reinterpret_cast<const scf_uci_csip2_info_t*> (next);
            auto numparts = csip2_info->numPart2s;
            NVLOGD_FMT(TAG, "CSI P2 numParts {} pusch_ue_idx {}", numparts, pusch_ue_idx);
            ue.pUciPrms = &info->uci_info[pusch_ue_idx];
            auto pUciPrms = ue.pUciPrms;
            if (!numparts) {
                pUciPrms->nCsi2Reports = 0;
                pUciPrms->pCalcCsi2SizePrms = nullptr;
            } else {
                next += sizeof(scf_uci_csip2_info_t);
                pUciPrms->nCsi2Reports = numparts;
                pUciPrms->pCalcCsi2SizePrms = &info->csip2_v3_params[pusch_ue_idx * CUPHY_MAX_N_CSI2_REPORTS_PER_UE];
                size_t offset = 0;
                for (int i = 0; i <  csip2_info->numPart2s; i++) {
                    auto csip2_part = reinterpret_cast<const scf_uci_csip2_part_t*>(next + offset);
                    auto num1PartParams = csip2_part->numPart1Params;

                    offset += sizeof(scf_uci_csip2_part_t) ;
                    auto csip2_part_offset = reinterpret_cast<const scf_uci_csip2_part_param_offset_t*>(next + offset);
                    auto csip2_part_size_offset = reinterpret_cast<const scf_uci_csip2_part_param_size_t*>(next + offset + csip2_part->numPart1Params *  sizeof(uint16_t));

                    for ( uint16_t j = 0; j < csip2_part->numPart1Params; j++) {
                            pUciPrms->pCalcCsi2SizePrms[i].prmOffsets[j] = reinterpret_cast<uint16_t>(csip2_part_offset->paramOffsets[j]);
                            pUciPrms->pCalcCsi2SizePrms[i].prmSizes[j] =  csip2_part_size_offset->paramSizes[j];
                            NVLOGD_FMT(TAG, "CSI P2 num1PartParams={} paramOffsets=[{}] prmSizes=[{}]", num1PartParams, pUciPrms->pCalcCsi2SizePrms[i].prmOffsets[j], pUciPrms->pCalcCsi2SizePrms[i].prmSizes[j]);
                    }

                    offset +=  csip2_part->numPart1Params * ((sizeof(uint16_t) + sizeof(uint8_t)));
                    auto csip2_part_scope = reinterpret_cast<const scf_uci_csip2_part_scope_t*>(&csip2_info->payload[0] + offset);

                    pUciPrms->pCalcCsi2SizePrms[i].nPart1Prms = num1PartParams;
                    pUciPrms->pCalcCsi2SizePrms[i].csi2sizeMapIdx = csip2_part_scope->part2SizeMapIndex;
                    // csip2_part_scope->part2SizeMapScope;
                    NVLOGD_FMT(TAG, "CSI P2 csi2sizeMapIdx={}",  pUciPrms->pCalcCsi2SizePrms[i].csi2sizeMapIdx);

                    offset +=sizeof(scf_uci_csip2_part_scope_t);
                }
                next += offset;
            }
        }

        if (phyDriver.l1_get_enable_weighted_average_cfo()) {    // PUSCH Extension  
            auto puschExtension = reinterpret_cast<const scf_fapi_pusch_extension_t*>(next);
            // PUSCH Extension for weighted average CFO estimation
            ue.foForgetCoeff = static_cast<float>(puschExtension->fo_forget_coeff) / 100.0f;
            ue.ldpcEarlyTerminationPerUe = puschExtension->ldpc_early_termination;
            ue.ldpcMaxNumItrPerUe = puschExtension->n_iterations;
            NVLOGD_FMT(TAG, "PUSCH Extension for weighted average CFO estimation {} LDPC n_iterations {} ldpc_early_termination {} ", 
                ue.foForgetCoeff, puschExtension->ldpc_early_termination, puschExtension->n_iterations);
            next += sizeof(scf_fapi_pusch_extension_t);
        }
#endif

        if(msg.transform_precoding==0)
        {
            ue.enableTfPrcd = 1;
            if (!(msg.pdu_bitmap & 0x8))
            {
            #ifdef SCF_FAPI_10_04
                ue.groupOrSequenceHopping = maintenance->groupOrSequenceHopping;
                ue.N_symb_slot = 14;
                ue.lowPaprGroupNumber = 0; //no use
                ue.lowPaprSequenceNumber = 0; //no use
            #else
                ue.groupOrSequenceHopping = 0; //no use
                ue.N_symb_slot = 14; //no use
                ue.lowPaprGroupNumber = 0; //no use
                ue.lowPaprSequenceNumber = 0; //no use
            #endif
            }
        }
        else
        {
            ue.enableTfPrcd = 0;
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
        coreset.nDci = msg.num_dl_dci;
        coreset.testModel = testMode;
        NVLOGD_FMT(TAG,"{}:{} PDCCH testModel={}",__func__,__LINE__,testMode);
    }


    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd, void* buffer, bool ssb, uint32_t cell_index, int buffLoc, nv::slot_detail_t*  slot_detail)
    {
        if (ssb) {
            uint8_t* ssb = static_cast<uint8_t*>(buffer);
            pbch_params* pbch = cell_cmd.params.pbch.get();
            to_u8_array(pbch->mib,ssb);

        }
        else {
            pdsch_params* info = NULL;
            if(cell_grp_cmd)
                info  = cell_grp_cmd->get_pdsch_params();
            else
                info  = cell_cmd.params.pdsch.get();

            if (info == nullptr)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no pdsch command");
                return;
            }
            // Multiple UE per TTI support is present only for tag value = 2 (SCF_TX_DATA_OFFSET)
            {
                //tb_data.pTbInput is pointed to base address of ue_tb_ptr in pdsch_params constructor
                info->ue_tb_ptr[cell_index] = static_cast<uint8_t*>(buffer);
                if(buffLoc == 3)
                {
                    info->tb_data.pBufferType = cuphyPdschDataIn_t::GPU_BUFFER;
                }
                //NVLOGI_FMT(TAG,"[{}] pBufferType = {}",__func__,info->tb_data.pBufferType);
                //NVLOGD_FMT(TAG, "PDSCH pTbInput for cell_index={}",cell_index);
#if 0
                uint8_t* tb = info->tb_data.pTbInput[cell_index];
                if (tb != nullptr) {
                    for (uint i = 0; i < 30; i+=5) {
                        NVLOGC_FMT(TAG, "{} Data Buffer[{}] = 0x{:02X} 0x{:02X} 0x{:02X} 0x{:02X} 0x{:02X}", __FUNCTION__, i, tb[i], tb[i+1], tb[i+2], tb[i+3], tb[i+4]);
                    }
                }
#endif
            }
        }
    }

    void update_cell_command(cell_group_command* grp, cell_sub_command& cell_cmd, slot_indication & slotinfo, const scf_fapi_prach_pdu_t& req, nv::phy_config& cell_params, nv::prach_addln_config_t& addln_config,
                             int32_t cell_index, bool bf_enabled, nv::slot_detail_t*  slot_detail, bool mmimo_enabled) {
        //cell_cmd.create_if(channel_type::PRACH);
	    cell_cmd.slot.type = SLOT_UPLINK;
        cell_cmd.slot.slot_3gpp = slotinfo;
        grp->slot.slot_3gpp = slotinfo;
	    grp->slot.type = SLOT_UPLINK;

        prach_params* rach_params = grp->get_prach_params();
        auto& rach = rach_params->rach[rach_params->nOccasion];
	    rach_params->cell_index_list.push_back(cell_index);
        rach_params->phy_cell_index_list.push_back(cell_params.cell_config_.phy_cell_id);
       	rach_params->freqIndex[rach_params->nOccasion]  = req.num_ra;
        rach_params->startSymbols[rach_params->nOccasion] = req.prach_start_symbol;

        rach.occaPrmStatIdx = cell_params.prach_config_.start_ro_index + req.num_ra;
        rach.occaPrmDynIdx = rach_params->nOccasion;
        rach.force_thr0 = 0.0;

        if ( mmimo_enabled && req.beam_index.dig_bf_interfaces != 0){
            rach.nUplinkStreams = req.beam_index.dig_bf_interfaces;
        }

        rach_params->mu = cell_params.prach_config_.prach_scs;
        rach_params->nfft = cell_params.prach_config_.prach_seq_length == 1 ? (nv::PRACH_SHORT_FORMAT_FFT): (nv::PRACH_LONG_FORMAT_FFT);
        rach_params->nOccasion++;

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
        ru_type ru = mplane_info.ru;
        NVLOGI_FMT(TAG, "{} PRACH occaPrmStatIdx={} occaPrmDynIdx={}, numRa ={}", __FUNCTION__, rach.occaPrmStatIdx, rach.occaPrmDynIdx, req.num_ra);
        update_fh_params_prach(cell_params, addln_config, req, req.beam_index, cell_cmd, bf_enabled, ru, slot_detail,mmimo_enabled, cell_index);
    }

    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, slot_indication & slotinfo, const scf_fapi_pucch_pdu_t& pdu, int32_t cell_index, const nv::pucch_dtx_t_list& dtx_thresholds,
                             uint16_t cell_stat_prm_idx, nv::phy_config_option& config_option, nv::slot_detail_t*  slot_detail, bool mmimo_enabled, uint16_t ul_bandwidth, uint16_t pucch_hopping_id)
    {
        // cell_sub_cmd.create_if(channel_type::PUCCH);
        cell_sub_cmd.slot.type = SLOT_UPLINK;
        cell_grp_cmd->slot.type = SLOT_UPLINK;
        cell_sub_cmd.slot.slot_3gpp = slotinfo;
        cell_grp_cmd->slot.slot_3gpp = slotinfo;
        uint32_t uci_pdu_bit_len = 0;
        uint8_t  qm = CUPHY_QAM_4;
        float    code_rate = 0;
        uint8_t  num_sub_carrier_per_prb = CUPHY_N_TONES_PER_PRB;
        uint8_t  uci_crc_len = 0;
        float    code_rate_table[] = {0.08, 0.15, 0.25, 0.35, 0.45, 0.80, 1.00};
        bool newCell = false;

        //Check to see if we can get puchh_params from cell_group_cmd instead of
        //cell_cmd
        //pucch_params* pucch_params = cell_sub_cmd.get_pucch_params();
        //cell_grp_cmd->create_if(channel_type::PUCCH);
        pucch_params* pucch_grp_params = cell_grp_cmd->get_pucch_params();

        auto it = std::find(pucch_grp_params->cell_index_list.begin(), pucch_grp_params->cell_index_list.end(), cell_index);
        if(it == pucch_grp_params->cell_index_list.end())
        {
            pucch_grp_params->cell_index_list.push_back(cell_index);
            pucch_grp_params->phy_cell_index_list.push_back(cell_sub_cmd.cell);
            newCell = true;
        }

        pucch_grp_params->scf_ul_tti_handle_list[pdu.format_type].push_back(pdu.handle);

        auto& params =  pucch_grp_params->params[pdu.format_type];
        cuphyPucchCellGrpDynPrm_t& grp = pucch_grp_params->grp_dyn_pars;

        uint16_t index = UINT16_MAX;
        switch (pdu.format_type) {
            case UL_TTI_PUCCH_FORMAT_0:
                index = grp.nF0Ucis;
                break;
            case UL_TTI_PUCCH_FORMAT_1:
                index = grp.nF1Ucis;
                break;
            case UL_TTI_PUCCH_FORMAT_2:
                index = grp.nF2Ucis;
                break;
            case UL_TTI_PUCCH_FORMAT_3:
                index = grp.nF3Ucis;
            break;
            case UL_TTI_PUCCH_FORMAT_4:
                index = grp.nF4Ucis;
            break;
        }

        cuphyPucchUciPrm_t& uci_info = params[index];
        uci_info.uciOutputIdx           = index;
        uci_info.formatType             = pdu.format_type;
        uci_info.rnti                   = pdu.rnti;
        uci_info.multiSlotTxIndicator   = pdu.multi_slot_tx_indicator;
        uci_info.pi2Bpsk                = pdu.pi_2_bpsk;
        uci_info.bwpStart               = pdu.bwp.bwp_start;
        uci_info.startPrb               = pdu.prb_start;
        uci_info.prbSize                = pdu.prb_size;
        uci_info.startSym               = pdu.start_symbol_index;
        uci_info.nSym                   = pdu.num_of_symbols;
        uci_info.freqHopFlag            = pdu.freq_hop_flag;
        uci_info.secondHopPrb           = pdu.second_hop_prb;
        uci_info.groupHopFlag           = pdu.group_hop_flag;
        uci_info.sequenceHopFlag        = pdu.seq_hop_flag;
        uci_info.initialCyclicShift     = pdu.initial_cyclic_shift;
        uci_info.timeDomainOccIdx       = pdu.time_domain_occ_idx;
        uci_info.srFlag                 = pdu.format_type > 1 ? 0 : pdu.sr_flag;
        uci_info.bitLenHarq             = pdu.bit_len_harq;
        uci_info.bitLenCsiPart1         = pdu.bit_len_csi_part_1;
        uci_info.AddDmrsFlag            = pdu.add_dmrs_flag;
        uci_info.dataScramblingId       = pdu.data_scrambling_id;
        uci_info.DmrsScramblingId       = pdu.dmrs_scrambling_id;
        uci_info.uciP1P2Crpd_t.numPart2s = 0;
        uci_info.rankBitOffset          = 0;
        uci_info.nRanksBits             = 0;
        uci_info.DTXthreshold           = dtx_thresholds[uci_info.formatType];

        uci_info.bitLenSr = pdu.sr_flag;

        if(newCell)
        {
            cuphyPucchCellDynPrm_t& dyn = pucch_grp_params->dyn_pars[grp.nCells];
            dyn.cellPrmStatIdx              = cell_stat_prm_idx;
            dyn.cellPrmDynIdx               = grp.nCells;
            dyn.slotNum                     = (config_option.staticPucchSlotNum != -1)? config_option.staticPucchSlotNum :slotinfo.slot_;
            dyn.pucchHoppingId = pucch_hopping_id;
            grp.nCells++;
        }
        uci_info.cellPrmDynIdx = grp.nCells-1;
        uci_info.cellPrmStatIdx = cell_stat_prm_idx;
        switch (pdu.format_type)
        {
            case UL_TTI_PUCCH_FORMAT_0:
            {
                grp.pF0UciPrms = params.data();
                grp.nF0Ucis++;
            }
            break;
            case UL_TTI_PUCCH_FORMAT_1:
            {
                grp.pF1UciPrms = params.data();
                grp.nF1Ucis++;
            }
            break;
            case UL_TTI_PUCCH_FORMAT_2:
            {
                grp.pF2UciPrms = params.data();
                grp.nF2Ucis++;

            }
            break;
            case UL_TTI_PUCCH_FORMAT_3:
            {
                grp.pF3UciPrms = params.data();
                grp.nF3Ucis++;
            }
            break;
            case UL_TTI_PUCCH_FORMAT_4:
            {
                grp.pF4UciPrms = params.data();
                grp.nF4Ucis++;
            }
            break;
            default:
            break;
        }
        NVLOGD_FMT(TAG, "{}: PUCCH grp.nF0Ucis={} grp.nF1Ucis={} sr_flag={} bit_len_harq={}", __FUNCTION__, grp.nF0Ucis, grp.nF1Ucis, pdu.sr_flag, static_cast<unsigned short>(pdu.bit_len_harq));

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
        ru_type ru = mplane_info.ru;

        update_fh_params_pucch(uci_info, pdu.prb_size, *reinterpret_cast<const scf_fapi_rx_beamforming_t*>(&pdu.payload[0]),cell_sub_cmd, config_option.bf_enabled, ru, slot_detail,mmimo_enabled, cell_index, ul_bandwidth);
    }

int update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd, const scf_fapi_srs_pdu_t& msg, int32_t cell_index, slot_indication & slotinfo, cuphyCellStatPrm_t cell_params, uint16_t cell_stat_prm_idx,
                                    bool bf_enabled, size_t nvIpcAllocBuffLen, int *p_srs_ind_index, int mutiple_srs_ind_allowed, nv::phy_mac_transport& transport, bool is_last_srs_pdu, bool is_last_non_prach_pdu, nv::slot_detail_t*  slot_detail,
                                    bool mmimo_enabled)
{

    if(msg.num_ant_ports > 2)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS PDU error!! num_ant_ports is {} more than 2", msg.num_ant_ports);
        return  SRS_PDU_INVALID_NUM_ANT_PORTS; /* ERROR */
    }

#ifdef SCF_FAPI_10_04_SRS
    const uint8_t* next = &msg.payload[0];
    /* It is optional for L2 to encode Rx Beamforming PDU, so L1 needs to check if it's present and if present then decode it */
    const scf_fapi_rx_beamforming_t* srs_rx_bf = reinterpret_cast<const scf_fapi_rx_beamforming_t*>(next);

    /* If TRP scheme is 0, then beamforming PDU is present, if it's not present then the next byte will be in the range of [4,272]*/
    if(srs_rx_bf->trp_scheme == 0) {
       next += (sizeof(scf_fapi_rx_beamforming_t) + (sizeof(uint16_t) * srs_rx_bf->num_prgs * srs_rx_bf->dig_bf_interfaces));
    }

    const scs_fapi_v4_srs_params_t* srs_v4_parms = reinterpret_cast<const scs_fapi_v4_srs_params_t*>(next);

    if (srs_v4_parms->rep_scope != 0)
    {
         NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "Invalid reportScope = {}, only reportScope sampledUeAntennas is supported. Hence, SRS.IND failed!", srs_v4_parms->rep_scope);
         return  SRS_PDU_INVALID_REPORT_SCOPE; /* ERROR */
    }

    /* SRS report for usage type other than Beam Management, Codebook, or Non-Codebook is not supported. */
    if (!(srs_v4_parms->usage & (SRS_REPORT_FOR_BEAM_MANAGEMENT | SRS_REPORT_FOR_CODEBOOK | SRS_REPORT_FOR_NON_CODEBOOK)))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS report for usage type {} is not supported. Hence, SRS.IND failed!", static_cast<int>(srs_v4_parms->usage));
        return  SRS_PDU_UNSUPPORTED_REPORT_USAGE; /* ERROR */
    }
#else
    scf_fapi_rx_beamforming_t l_srs_rx_bf = {0};
    scf_fapi_rx_beamforming_t *srs_rx_bf = &l_srs_rx_bf;
#endif

    cell_sub_cmd.slot.type = SLOT_UPLINK;
    cell_sub_cmd.slot.slot_3gpp = slotinfo;
    cell_grp_cmd->slot.type = SLOT_UPLINK;
    cell_grp_cmd->slot.slot_3gpp = slotinfo;

    srs_params* srs_params = cell_grp_cmd->get_srs_params();

    /* redundant check */
    if (srs_params == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no srs command");
        return  SRS_PDU_NO_SRS_CMD; /* ERROR */
    }

    bool newCell = false;
    auto it = std::find(srs_params->cell_index_list.begin(), srs_params->cell_index_list.end(), cell_index);
    if(it == srs_params->cell_index_list.end())
    {
        srs_params->cell_index_list.push_back(cell_index);
        srs_params->phy_cell_index_list.push_back(cell_sub_cmd.cell);
        ++srs_params->cell_grp_info.nCells;
        newCell = true;
    }
    
    //Update Cell Dyn paramters
    cuphySrsCellDynPrm_t *cell_info =  nullptr;
    if(newCell)
    {
        cell_info = &(srs_params->cell_dyn_info[srs_params->cell_grp_info.nCells-1]); 

        cell_info->cellPrmDynIdx = srs_params->cell_grp_info.nCells-1;
        cell_info->cellPrmStatIdx = cell_stat_prm_idx;
        cell_info->slotNum = cell_grp_cmd->slot.slot_3gpp.slot_;
        cell_info->frameNum = cell_grp_cmd->slot.slot_3gpp.sfn_;
        cell_info->srsStartSym = 0;
        cell_info->nSrsSym = 0;
        NVLOGD_FMT(TAG, "{}: SRS new cell nCells={} newCell={} cell_index {} cellPrmStatIdx={} cellPrmDynIdx={} cell_info_index={} cell_info={} local_cell_info={}", 
                        __func__, srs_params->cell_grp_info.nCells, newCell? "true" : "false", cell_index, cell_stat_prm_idx,  cell_info->cellPrmDynIdx,
                        (srs_params->cell_grp_info.nCells-1), static_cast<void *>(&srs_params->cell_dyn_info[srs_params->cell_grp_info.nCells-1]), static_cast<void *>(cell_info));
    }
    else
    {
        cell_info = &(srs_params->cell_dyn_info[it - srs_params->cell_index_list.begin()]);
        cell_info->slotNum = cell_grp_cmd->slot.slot_3gpp.slot_;
        cell_info->frameNum = cell_grp_cmd->slot.slot_3gpp.sfn_;
        NVLOGD_FMT(TAG, "{}: SRS new cell nCells={} newCell={} cell_index {} cellPrmStatIdx={} cellPrmDynIdx={} cell_info_index{}, cell_info={} local_cell_info={}",
                        __func__, srs_params->cell_grp_info.nCells, newCell? "true" : "false", cell_index, cell_stat_prm_idx,  cell_info->cellPrmDynIdx, (it - srs_params->cell_index_list.begin()),
                        static_cast<void *>(&srs_params->cell_dyn_info[it - srs_params->cell_index_list.begin()]), static_cast<void *>(cell_info));
    }

    auto srs_ue_idx = srs_params->cell_grp_info.nSrsUes;
    auto&  ue = srs_params->ue_info[srs_ue_idx];

    srs_rb_info_t srs_rb_info[MAX_SRS_SYM]={{0}};
    uint8_t nHops = 0;
    uint16_t prgSize = MIN_PRG_SIZE;
    uint16_t prgSizeL2 = MIN_PRG_SIZE;
    uint16_t num_sym_info = calc_start_prb(msg, cell_sub_cmd, srs_rb_info, nHops);
#ifdef SCF_FAPI_10_04_SRS
    // PRG SIZE in SRS PDU can be set to any integer value between 1 and 272;
    // The input of prgSize for the cuPHY internal processing should be 1, 2, or 4;
    // If PRG SIZE in SRS PDU is greater than 4 or equal to 3, then prgSize will be set to 2; otherwise prgSize will be equal to PRG SIZE in SRS PDU;
    NVLOGD_FMT(TAG, "{}: PRG SIZE in SRS PDU set to {}", __func__, static_cast<uint16_t>(srs_v4_parms->prg_size));
    prgSizeL2 = std::max(prgSizeL2, srs_v4_parms->prg_size);
    if((prgSizeL2 > 4) || (prgSizeL2 == 3))
    {
        prgSize = 2;
    }
    else
    {
        prgSize = prgSizeL2;
    }
#else
    /* For FAPI 10_02, PRG SIZE is set to 1. This field is not present in FAPI 10.02 SRS PDU but is mandatorily needed by cuPHY module.*/
    prgSize = MIN_PRG_SIZE;
    prgSizeL2 = MIN_PRG_SIZE;
#endif
    
    uint16_t num_prg_l2 = (nHops * srs_rb_info[0].num_srs_prbs)/prgSizeL2;
    uint16_t num_prg = (nHops * srs_rb_info[0].num_srs_prbs)/prgSize;

    if((srs_params->srs_indications[srs_params->cell_grp_info.nCells-1][*p_srs_ind_index].data_len + (sizeof(scf_fapi_norm_ch_iq_matrix_info_t) + (srs_ant_idx_to_port[msg.num_ant_ports] * num_prg_l2 * cell_params.nRxAntSrs * IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4))) >= nvIpcAllocBuffLen)
    {
        /* FAPI 10.04 supports splitting of SRS reports to L2 into multiple SRS IND's. 
         * Incase the mutiple SRS IND's is configured by L2 in Cell Config PDU then L1 will send multiple SRS IND's for the SRS_PDU's requested by L2 in UL_TTI.request. 
         * If the mutiple SRS IND's is not configured by L2 in Cell Config PDU then L1 will send an ERROR IND to L2 to indicate only one partial SRS.IND will follow. */
        if(!mutiple_srs_ind_allowed)
        {
            NVLOGI_FMT(TAG, "{}: SRS PDUs are overflowing NVIPC buffer with multiple_srs_ind_per_slot_NOT_allowed", __func__);
            return SRS_PDU_OVERFLOW_NVIPC_BUFF; /* Warning */
        }
        else
        {
            if((*p_srs_ind_index) < (MAX_SRS_IND_PER_SLOT-1))
            {
                /* new SRS.ind => nvIPC can be allocated */
                nv::phy_mac_msg_desc msg_desc;
                /* Using CPU_DATA would result in a sending mutiple SRS IND. Hence CPU_LARGE buffer is used for sending the mutipe
                 * SRS IND aswell because it can handle reports of a higher number of SRS PDUs compared to CPU_DATA. */
                msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
                if(transport.tx_alloc(msg_desc) < 0)
                {
                    return -1;
                }

                (*p_srs_ind_index)++;
                srs_params->srs_indications[srs_params->cell_grp_info.nCells-1][*p_srs_ind_index] = (nv_ipc_msg_t)msg_desc;
                NVLOGD_FMT(TAG, "{}: {}.{} allocating cell_index {} SRS IND srs_indications[{}][{}] = msg_id {}, cell_id {}, msg_len {}, data_len {}, data_pool {}, msg_buf {}, data_buf {}",
                                __func__, slotinfo.sfn_,slotinfo.slot_, cell_index, srs_params->cell_grp_info.nCells-1,*p_srs_ind_index,
                                msg_desc.msg_id,msg_desc.cell_id,msg_desc.msg_len,msg_desc.data_len,msg_desc.data_pool,msg_desc.msg_buf,msg_desc.data_buf);
            }
            else
            {
                NVLOGW_FMT(TAG, "SRS PDUs are overflowing NVIPC buffer with multiple_srs_ind_per_slot_allowed={} srs.ind allocted={}",mutiple_srs_ind_allowed,*p_srs_ind_index);
                return SRS_PDU_OVERFLOW_NVIPC_BUFF; /* Warning */
            }
        }
    }

    ue.cellIdx = cell_info->cellPrmDynIdx;
    ue.nAntPorts = srs_ant_idx_to_port[msg.num_ant_ports];
    ue.nSyms = srs_symb_idx_to_numSymb[msg.num_symbols];
    ue.nRepetitions = srs_rep_factor_idx_to_numRepFactor[msg.num_repetitions];
    ue.combSize = srs_comb_idx_to_combSize[msg.comb_size];
    ue.startSym = msg.time_start_position;
    ue.sequenceId = msg.sequenceId;
    ue.configIdx = msg.config_index;
    ue.bandwidthIdx = msg.bandwidth_index;
    ue.combOffset = msg.comb_offset;
    ue.cyclicShift = msg.cyclic_shift;
    ue.frequencyPosition = msg.frequency_position;
    ue.frequencyShift = msg.frequency_shift;
    ue.frequencyHopping = msg.frequency_hopping;
    ue.resourceType = msg.resource_type;
    ue.Tsrs = msg.t_srs;
    ue.Toffset = msg.t_offset;
    ue.groupOrSequenceHopping = msg.group_or_sequence_hopping;
    ue.chEstBuffIdx = srs_ue_idx;
    ue.rnti = msg.rnti;
    ue.handle = msg.handle;
    ue.prgSize = prgSizeL2;
    ue.usage = SRS_REPORT_FOR_BEAM_MANAGEMENT; /* This field is not present in FAPI 10.02. Hence setting default value for FAPI 10.02 */
    ue.nValidPrg = num_prg;
    ue.startValidPrg = floor(srs_rb_info[0].srs_start_prbs/prgSize);

    uint16_t min_srs_start_prbs = 0;
    bool firstParse = true;
    for(uint16_t symIndex = 0; symIndex < num_sym_info; symIndex++)
    {
        if(firstParse)
        {
            min_srs_start_prbs = srs_rb_info[symIndex].srs_start_prbs;
            firstParse = false;
        }
        else
        {
            min_srs_start_prbs = std::min(min_srs_start_prbs,srs_rb_info[symIndex].srs_start_prbs);
        }
    }
    // This sets the start of the SRS buffer, which is indexed from 0, not start of SRS
    ue.srsStartPrg = 0;

#ifdef SCF_FAPI_10_04
    /* SRS channel estimation buffer index is encoded by L2 in the handle field.
     * L2 should maintain these buffers and assign it to SRS PDU's in the handle field. L1 responds with the same buffer index in the handle field in SRS.IND.
     * L1 retains the channel estimates till the same buffer index is used again in SRS_PDU.
     * Currenlty L1 maintains a state machine to track the buffer index and when the buffer is in requested state the same buffer cannot be used again.
     * L2 should send the same buffer index in the handle field in DLBFW_CVI.request/ ULBFW_CVI.request which it wants L1 to use as an input for Dynamic Beamforming weights calculation.
     * Extract the SRS channel estimation buffer index from handle field (bits 8-23).
     * In FAPI 10.04, L2 encodes the buffer index in the upper bytes of the handle. */
    ue.srsChestBufferIndexL2 = static_cast<uint16_t>((msg.handle >> 8) & 0xFFFF);
#else
    ue.srsChestBufferIndexL2 = ((msg.rnti - 1) % slot_command_api::MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL);
#endif
#ifdef SCF_FAPI_10_04_SRS
    ue.usage = srs_v4_parms->usage;
#endif

    srs_params->dl_ul_bwp_max_prg[cell_index] = ROUND_UP(cell_params.nPrbDlBwp,prgSize);
    srs_params->nGnbAnt = cell_params.nRxAntSrs;
    if(newCell)
    {
       cell_info->srsStartSym = ue.startSym;
       cell_info->nSrsSym = ue.nSyms;
    }
    else
    {
       if (cell_info->srsStartSym != ue.startSym || cell_info->nSrsSym != ue.nSyms)
       {
           if (ue.startSym < cell_info->srsStartSym)
           {
               uint8_t num_sym = (cell_info->srsStartSym + cell_info->nSrsSym) - ue.startSym;
               if (num_sym >= ue.nSyms)
               {
                   cell_info->nSrsSym = num_sym;
               }
               else
               {
                   cell_info->nSrsSym = ue.nSyms;
               }
               cell_info->srsStartSym = ue.startSym;
           }
           else
           {
               uint8_t ue_num_sym = (ue.startSym + ue.nSyms) - cell_info->srsStartSym;
               if (ue_num_sym > cell_info->nSrsSym)
               {
                   cell_info->nSrsSym = ue_num_sym;
               }
           }
        }
    }
    NVLOGD_FMT(TAG, "{}: SRS: cellIdx {} rnti {} min_srs_start_prbs {} srsStartPrg {} srsChestBufferIndexL2 {} cell_index {} cellPrmStatIdx={} cellPrmDynIdx={} slotNum={} frameNum={} srsStartSym={} nSrsSym={}", 
                     __func__, ue.cellIdx, ue.rnti, min_srs_start_prbs, ue.srsStartPrg, ue.srsChestBufferIndexL2, cell_index, 
                     cell_stat_prm_idx, cell_info->cellPrmDynIdx, cell_info->slotNum, cell_info->frameNum, cell_info->srsStartSym, cell_info->nSrsSym);


    uint8_t antPortIdx = 0;
    for (uint8_t bitPos = 0; bitPos < ue.nAntPorts; bitPos++)
    {
#ifdef SCF_FAPI_10_04_SRS
        if(srs_v4_parms->samp_ue_ant & (1 << bitPos))
        {
            ue.srsAntPortToUeAntMap[antPortIdx] = antPortIdx;
            antPortIdx++;
        }
#else
        {
            /* not used in case of FAPI 10_02 */
            ue.srsAntPortToUeAntMap[antPortIdx] = antPortIdx;
            antPortIdx++;
        }
#endif
    }


    nv_ipc_msg_t* p_desc = &(srs_params->srs_indications[srs_params->cell_grp_info.nCells-1][*p_srs_ind_index]);
    auto& cell_grp_info = srs_params->cell_grp_info;
    p_desc->data_len += sizeof(scf_fapi_norm_ch_iq_matrix_info_t);
    /* update the start address of channel estimation write */
    srs_params->srs_chest_buffer[cell_grp_info.nSrsUes] = (uint8_t*)(p_desc->data_buf) + sizeof(uint8_t)*(p_desc->data_len);
    /* Adding the sizeof header plus the IQ sample length */
    p_desc->data_len += (ue.nAntPorts * num_prg_l2 * cell_params.nRxAntSrs * IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4);

    cell_grp_info.nSrsUes++;
    srs_params->srs_ue_per_cell[srs_params->cell_grp_info.nCells-1].cell_idx = cell_index;
    srs_params->srs_ue_per_cell[srs_params->cell_grp_info.nCells-1].num_srs_ues++;

    NVLOGD_FMT(TAG,"{}: Cell={} store_idx={} num_srs_ues_per_cell={} srs_chest_buffer[{}]:{} data_len={} nAntPorts={} prg_size_l2={} gnb_ant={}, nHops={} num_sym_info {} srs_ind_index={}",
                    __func__, cell_index, srs_params->cell_grp_info.nCells-1, srs_params->srs_ue_per_cell[srs_params->cell_grp_info.nCells-1].num_srs_ues,
                    cell_grp_info.nSrsUes,reinterpret_cast<void*>(srs_params->srs_chest_buffer[cell_grp_info.nSrsUes]),
                    srs_params->srs_indications[srs_params->cell_grp_info.nCells-1][*p_srs_ind_index].data_len,
                    static_cast<int>(ue.nAntPorts), static_cast<uint16_t>(num_prg_l2), static_cast<uint16_t>(cell_params.nRxAntSrs), static_cast<uint8_t>(nHops),
                    static_cast<uint8_t>(num_sym_info), *p_srs_ind_index);

    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    auto & mplane_info = phyDriver.getMPlaneConfig(cell_index);
    ru_type ru = mplane_info.ru;

    uint16_t start_rb = 0, end_rb = 0;
    uint8_t symIdx = 0, pdu_idx = 0;

    for (symIdx = 0; symIdx < num_sym_info; symIdx++)
    {
        start_rb = srs_rb_info[symIdx].srs_start_prbs;
        end_rb = start_rb + srs_rb_info[symIdx].num_srs_prbs - 1;
        srs_params->rb_info_per_sym[cell_index][ue.startSym+symIdx].push_back({start_rb, end_rb});
        NVLOGD_FMT(TAG,"ue.startSym {} symIdx {} start_rb {} end_rb {} ",ue.startSym, symIdx, start_rb, end_rb);
        for (int j = 0; j < srs_params->rb_info_per_sym[cell_index][ue.startSym+symIdx].size(); j++)
        {
            NVLOGD_FMT(TAG,"vec start_rb {} vec end_rb {}", srs_params->rb_info_per_sym[cell_index][ue.startSym+symIdx][j].first, srs_params->rb_info_per_sym[cell_index][ue.startSym+symIdx][j].second);
        }
    }

    if (is_last_srs_pdu)
    {
        finalize_srs_slot(cell_sub_cmd, *srs_rx_bf, cell_info->nSrsSym, cell_info->srsStartSym, srs_params, bf_enabled, ru, slot_detail, cell_index, is_last_non_prach_pdu);
    }

    srs_params->scf_ul_tti_handle_list.push_back(msg.handle);

    /* update the PDU count */
    (srs_params->num_srs_pdus_per_srs_ind[srs_params->cell_grp_info.nCells-1][*p_srs_ind_index])++;

    return SRS_PDU_SUCCESS;
}

#ifdef SCF_FAPI_10_04
void update_cell_command(cell_group_command* cell_grp_cmd,
                         cell_sub_command& cell_sub_cmd,
                         const scf_fapi_dl_bfw_group_config_t& bfw_msg,
                         int32_t cell_index,
                         slot_indication & slotinfo,
                         cuphyCellStatPrm_t cell_params,
                         bfw_coeff_mem_info_t *bfwCoeff_mem_info,
                         bfw_type bfwType,
                         nv::slot_detail_t*  slot_detail,
                         uint32_t &droppedBFWPdu)
{
    uint8_t report_type = REPORT_TYPE_NON_CODEBOOK;
    cell_sub_cmd.slot.type = SLOT_DOWNLINK;
    cell_grp_cmd->slot.type = SLOT_DOWNLINK;
    if(bfwType == slot_command_api::UL_BFW)
    {
        report_type = REPORT_TYPE_CODEBOOK;
        cell_sub_cmd.slot.type = SLOT_UPLINK;
        cell_grp_cmd->slot.type = SLOT_UPLINK;
    }

    cell_sub_cmd.slot.slot_3gpp = slotinfo;
    cell_grp_cmd->slot.slot_3gpp = slotinfo;

    bfw_params* bfw_params = cell_grp_cmd->get_bfw_params();

    if (bfw_params == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "no bfw command");
        return;
    }

    auto bfw_ue_grp_idx = bfw_params->bfw_dyn_info.nUeGrps;
    cuphyBfwUeGrpPrm_t& ue = bfw_params->ue_grp_info[bfw_ue_grp_idx];

    ue.startPrbGrp = (bfw_msg.dl_bfw_cvi_config.rb_start/bfw_msg.dl_bfw_cvi_config.prg_size);
    ue.nPrbGrp = bfw_msg.dl_bfw_cvi_config.num_prgs;
    //ue.nRxAnt = NUM_GNB_TX_RX_ANT_PORTS; //TODO: How to read from the cuPhyDriver.
    ue.nRxAnt = cell_params.nRxAntSrs;

    bfw_params->dl_ul_bwp_max_prg[cell_index] = ROUND_UP(cell_params.nPrbDlBwp,bfw_msg.dl_bfw_cvi_config.prg_size);
    bfw_params->nGnbAnt = cell_params.nRxAntSrs;

    uint8_t *ptr = NULL;
    uint8_t ueIdx = 0;
    uint8_t ueAntIdx = 0;

    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    const uint8_t* next = &bfw_msg.dl_bfw_cvi_config.payload[0];

    ue.beamIdOffset = (bfwType == slot_command_api::UL_BFW) ? -1 : phyDriver.l1_getDynamicBeamIdOffset(cell_index);
    //NVLOGC_FMT(TAG, "{} line {}: SFN={}:SLOT:{}  ue.beamIdOffset {}", __FUNCTION__, __LINE__,cell_grp_cmd->slot.slot_3gpp.sfn_, cell_grp_cmd->slot.slot_3gpp.slot_, ue.beamIdOffset);

    uint16_t srsChestBufferIndexL2 = 0;
    uint32_t aggr_ue_cnt = 0;
    for(ueIdx = 0; ueIdx < bfw_msg.dl_bfw_cvi_config.nUes; ueIdx++)
    {
        uint8_t  srsPrgSize     = 0;
        uint16_t srsStartPrg    = 0;
        uint16_t srsStartValidPrg = 0;
        uint16_t srsNValidPrg   = 0;
        const scf_dl_bfw_config_start_t& bfw_config_start  = *reinterpret_cast<const scf_dl_bfw_config_start_t*>(next);
        srsChestBufferIndexL2 = static_cast<uint16_t>((bfw_config_start.handle >> 8) & 0xFFFF);
        cuphyTensorDescriptor_t srsDescr = NULL;

        //Check for valid buffer
        uint16_t rnti = bfw_config_start.rnti;        
        slot_command_api::srsChestBuffState srsChestBuffState = slot_command_api::SRS_CHEST_BUFF_NONE;
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        int retVal = phyDriver.l1_cv_mem_bank_get_buffer_state(cell_index,srsChestBufferIndexL2,&srsChestBuffState);
        if(retVal != -1)
        {
            /* Validate the SRS Chest buffer state and return error if the state is in REQUESTED state during DL_BFW_CVI_REQUEST processing.
             * SRS Chest buffer state is in REQUESTED state means the buffer is under prepration and SRS.IND is not yet sent till now for this buffer.
             * Hence, this buffer cannot be used immediately till SRS.IND is sent for the input for dynamic beamforming weights calculation. */
            if(srsChestBuffState == slot_command_api::SRS_CHEST_BUFF_REQUESTED)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "cellid {} rnti {} rsrsChestBufferIndex {} state is SRS_CHEST_BUFF_REQUESTED. Dropping srs pdu from {} BFW_CVI_REQUEST",
                                        cell_index,rnti,srsChestBufferIndexL2,(bfwType == slot_command_api::UL_BFW)?"UL":"DL");
                droppedBFWPdu++;
                continue;
            }
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "l1_cv_mem_bank_get_buffer_state() returned -1. cellid {} rnti {} srsChestBufferIndex {} in DL_BFW update_cell_command",cell_index,rnti, srsChestBufferIndexL2);
            return;
        }

        NVLOGD_FMT(TAG, "cell_sub_cmd.cell={}, cell_index={}",cell_sub_cmd.cell, cell_index);
        if (!(phyDriver.l1_cv_mem_bank_retrieve_buffer(cell_index, bfw_config_start.rnti, srsChestBufferIndexL2, report_type, &srsPrgSize, &srsStartPrg, &srsStartValidPrg, &srsNValidPrg, &srsDescr, &ptr)))
        {
            bfw_params->chEstInfo[ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx].startPrbGrp         = srsStartPrg;
            bfw_params->chEstInfo[ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx].tChEstBuffer.desc   = srsDescr;
            bfw_params->chEstInfo[ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx].tChEstBuffer.pAddr  = ptr;
            bfw_params->chEstInfo[ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx].srsPrbGrpSize       = srsPrgSize;
            bfw_params->chEstInfo[ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx].startValidPrg       = srsStartValidPrg;
            bfw_params->chEstInfo[ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx].nValidPrg           = srsNValidPrg;
            NVLOGD_FMT(TAG, "BFW-SRS retrieved CV buffer cell {} rnti {} srsStartPrg {} srsPrgSize {} srsStartValidPrg {}, srsNValidPrg {}, srsChestBufferIndexL2 {} chEstInfo_Index {} chEstInfo_ptr {}", cell_sub_cmd.cell, static_cast<uint16_t>(bfw_config_start.rnti), srsStartPrg,srsPrgSize, srsStartValidPrg, srsNValidPrg, srsChestBufferIndexL2, ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx, static_cast<void *>(ptr));
        }
        else
        {
            return;
        }
        NVLOGD_FMT(TAG, "UL:{} DL:{} BFW: rnti={} pduIdx={} gNbAntStartIdx={} gNbAntEndIdx={} numUeAnt={} prg_size={} num_prgs={} srsChestBufferIndexL2={}",
         (bfwType == slot_command_api::UL_BFW)?1:0,
         (bfwType == slot_command_api::DL_BFW)?1:0,
         static_cast<uint16_t>(bfw_config_start.rnti),
         static_cast<uint16_t>(bfw_config_start.pduIndex),
         static_cast<uint8_t>(bfw_config_start.gnb_ant_index_start),
         static_cast<uint8_t>(bfw_config_start.gnb_ant_index_end),
         static_cast<uint8_t>(bfw_config_start.num_ue_ants), 
         static_cast<uint8_t>(bfw_msg.dl_bfw_cvi_config.prg_size), 
         static_cast<uint8_t>(bfw_msg.dl_bfw_cvi_config.num_prgs), 
         srsChestBufferIndexL2);

        auto pduIdx = bfw_config_start.pduIndex;

        bfwCoeff_mem_info->pdu_idx_rnti_list[bfw_ue_grp_idx][pduIdx].pdu_idx = pduIdx;
        bfwCoeff_mem_info->pdu_idx_rnti_list[bfw_ue_grp_idx][pduIdx].rnti = rnti;

        NVLOGD_FMT(TAG,"slotIndex={} bfw_ue_grp_idx={} PDSCH Idx={} CVI RNTI={}", bfwCoeff_mem_info->slotIndex, bfw_ue_grp_idx,
                        static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[bfw_ue_grp_idx][pduIdx].pdu_idx),
                        static_cast<uint16_t>(bfwCoeff_mem_info->pdu_idx_rnti_list[bfw_ue_grp_idx][pduIdx].rnti));

        const uint8_t* ue_ant_idx = &bfw_config_start.payload[0];
        for (ueAntIdx = 0; ueAntIdx < bfw_config_start.num_ue_ants; ueAntIdx++)
        {
            bfw_params->pBfLayerPrm[bfw_params->prevUeGrpPerLayerInfoBufIdx + ue.nBfLayers].chEstInfoBufIdx = ueIdx + bfw_params->prevUeGrpChEstInfoBufIdx;
            bfw_params->pBfLayerPrm[bfw_params->prevUeGrpPerLayerInfoBufIdx + ue.nBfLayers].ueLayerIndex = *(ue_ant_idx+ueAntIdx);
            NVLOGD_FMT(TAG, "chEstInfoBufIdx={}, ueLayerIndex={} ue.nBfLayers={} Index={}",bfw_params->pBfLayerPrm[bfw_params->prevUeGrpPerLayerInfoBufIdx + ue.nBfLayers].chEstInfoBufIdx, *(ue_ant_idx+ueAntIdx), ue.nBfLayers,bfw_params->prevUeGrpPerLayerInfoBufIdx);
            ue.nBfLayers++;
        }
        next += sizeof(scf_dl_bfw_config_start_t) + bfw_config_start.num_ue_ants;
        aggr_ue_cnt++;
    }

    if(aggr_ue_cnt > 0)
    {
        bfw_params->prevUeGrpChEstInfoBufIdx += aggr_ue_cnt;
        bfw_params->bfw_cvi_type = ((bfwType == UL_BFW) ? UL_BFW : DL_BFW);

        ue.coefBufIdx = bfw_ue_grp_idx;
        ue.startPrb         = bfw_msg.dl_bfw_cvi_config.rb_start;
        ue.bfwPrbGrpSize    = bfw_msg.dl_bfw_cvi_config.prg_size;
        ue.nPrbGrp          = bfw_msg.dl_bfw_cvi_config.num_prgs;

        NVLOGD_FMT(TAG,"Cell_Idx {} UL:{} DL:{} BFW: startPrb {} bfwPrbGrpSize {} nPrbGrp {} prevUeGrpChEstInfoBufIdx={} ueIdx={} aggr_ue_cnt={} sfn={} slot={}", 
            cell_index,
            (bfwType == slot_command_api::UL_BFW)?1:0,
            (bfwType == slot_command_api::DL_BFW)?1:0,
            ue.startPrb,ue.bfwPrbGrpSize,ue.nPrbGrp,
            bfw_params->prevUeGrpChEstInfoBufIdx,
            ueIdx,
            aggr_ue_cnt,
            cell_grp_cmd->slot.slot_3gpp.sfn_,
            cell_grp_cmd->slot.slot_3gpp.slot_);

        if ((*bfwCoeff_mem_info->header == BFW_COFF_MEM_FREE) &&
            (bfwCoeff_mem_info->slotIndex == (cell_grp_cmd->slot.slot_3gpp.slot_ % MAX_BFW_COFF_STORE_INDEX)))
        {
            //memset(bfwCoeff_mem_info->buff_addr_chunk_h[bfw_ue_grp_idx], 0, (MAX_MU_MIMO_LAYERS * MAX_NUM_PRGS_DBF * NUM_GNB_TX_RX_ANT_PORTS * IQ_REPR_FP32_COMPLEX * sizeof(uint32_t)));
            bfw_params->pBfwCoefH[bfw_ue_grp_idx] = bfwCoeff_mem_info->buff_addr_chunk_h[bfw_params->nue_grps_per_cell[cell_index]];
            bfw_params->pBfwCoefD[bfw_ue_grp_idx] = bfwCoeff_mem_info->buff_addr_chunk_d[bfw_params->nue_grps_per_cell[cell_index]];
            NVLOGD_FMT(TAG, "{} bfwCoeff_mem_info={} bfw_params->pBfwCoefH[{}]:{} bfw_params->pBfwCoefD[{}]:{}",__func__, reinterpret_cast<void*>(bfwCoeff_mem_info), 
                                bfw_params->nue_grps_per_cell[cell_index],reinterpret_cast<void*>(bfw_params->pBfwCoefH[bfw_ue_grp_idx]), bfw_params->nue_grps_per_cell[cell_index],reinterpret_cast<void*>(bfw_params->pBfwCoefD[bfw_ue_grp_idx]));
            bfwCoeff_mem_info->num_buff_chunk_busy = bfw_params->nue_grps_per_cell[cell_index] + 1;
            *bfwCoeff_mem_info->header = BFW_COFF_MEM_BUSY;
            bfwCoeff_mem_info->nGnbAnt = cell_params.nRxAntSrs;
            bfwCoeff_mem_info->sfn = cell_grp_cmd->slot.slot_3gpp.sfn_;
            bfwCoeff_mem_info->slot = cell_grp_cmd->slot.slot_3gpp.slot_;
        }
        else
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} line {}: SFN {}.{} memory not available for storing bfwCoeff",
                __FUNCTION__, __LINE__, cell_grp_cmd->slot.slot_3gpp.sfn_, cell_grp_cmd->slot.slot_3gpp.slot_);
        }
        ue.pBfLayerPrm = &bfw_params->pBfLayerPrm[bfw_params->prevUeGrpPerLayerInfoBufIdx];
        bfw_params->prevUeGrpPerLayerInfoBufIdx += ue.nBfLayers;
        bfw_params->dataIn.pChEstInfo = &bfw_params->chEstInfo[0];
        bfw_params->dataOutH.pBfwCoef = bfw_params->pBfwCoefH;
        bfw_params->dataOutD.pBfwCoef = bfw_params->pBfwCoefD;
        bfw_params->bfw_dyn_info.nUeGrps++;
        bfw_params->nue_grps_per_cell[cell_index]++;
        if(bfw_params->nue_grps_per_cell[cell_index] > CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} line {}: bfw_params->nue_grps_per_cell[{}]:{} > CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS:{}",
                __FUNCTION__, __LINE__, cell_index, bfw_params->nue_grps_per_cell[cell_index], CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS);
            return;
        }
    }
}
#endif

// #ifdef ENABLE_L2_SLT_RSP
//     void reset_pdsch_cw_offset()
//     {
//         next_pdsch_cw_offset = 0;
//     }
// #else
#ifndef ENABLE_L2_SLT_RSP
    void reset_cell_command(cell_sub_command& cell_cmd, slot_command_api::slot_indication& slot_ind, int32_t cell_index, bool cell_group, cell_group_command* grp_cmd)
    {
        static int sfn = 0;
        static int slot = 0;
        if(sfn != slot_ind.sfn_ || slot != slot_ind.slot_)
        {
            sfn = slot_ind.sfn_;
            slot = slot_ind.slot_;
            for(auto& reset: cell_reset)
            {
                reset = true;
            }
            cell_grp_reset = true;
        }
        if(cell_group)
        {
            if(cell_grp_reset)
            {
                // nv::PHY_module::group_command()->reset();
                grp_cmd->reset();
                cell_grp_reset = false;
                pusch_ue_idx = channel_type::NONE;
                pdsch_ue_idx = channel_type::NONE;
                pucch_ue_idx = 0;
            }
        }

        if(cell_reset[cell_index])
        {
            cell_cmd.reset();
            if(!cell_group)
            {
                pdsch_ue_idx = channel_type::NONE;
                pusch_ue_idx = channel_type::NONE;
                pucch_ue_idx = 0;
            }
            pdsch_cw_idx = 0;
            next_pdsch_cw_offset = 0;
            cell_reset[cell_index] = false;
        }
    }
#endif

    inline uint16_t computeDmrsSymMaskTypeA(uint8_t firstUlDmrsPos, uint8_t dmrsAddlnPos, uint8_t nPuschSym, uint8_t dmrsMaxLen)
    {
        uint16_t dmrsBmsk = 0;

        switch (dmrsMaxLen)
        {
            case 1:
            {
                dmrsBmsk = ADDLDMRSBMSK_MAXLENGTH1[nPuschSym][dmrsAddlnPos];
                dmrsBmsk = dmrsBmsk | (1 << firstUlDmrsPos);
            }
                break;
            case 2:
            {
                dmrsBmsk = ADDLDMRSBMSK_MAXLENGTH2[nPuschSym][dmrsMaxLen];
                dmrsBmsk = dmrsBmsk | (1 << firstUlDmrsPos) | (1 << (firstUlDmrsPos + 1));
            }
                break;
            default:
                break;
        }
        return dmrsBmsk;
    }

    inline void update_fh_params_pusch(cuphyPuschUeGrpPrm_t& grp, bool is_new_grp, cuphyPuschUePrm_t& ue, const scf_fapi_rx_beamforming_t& pmi_bf_pdu,
                    const scf_fapi_pusch_pdu_t& msg, cell_sub_command& cell_cmd, bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool bf_enabled, enum ru_type ru,
                    bool mmimo_enabled, nv::slot_detail_t *slot_detail, uint32_t bfwUeGrpIndex, int32_t cell_index, uint16_t ul_bandwidth) {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};

        if(ru == SINGLE_SECT_MODE)
        {
            bool value = ifAnySymbolPresent(sym_prbs->symbols, UL_NON_PRACH_CHANNEL_MASK);
            if(value)
                return;
        }

        if(is_new_grp)
        {
            check_prb_info_size(sym_prbs->prbs_size);
            if(ru == SINGLE_SECT_MODE)
            {
                prbs[sym_prbs->prbs_size] = prb_info_t(0, ul_bandwidth);
                sym_prbs->prbs_size++;
            }
            else
            {
                prbs[sym_prbs->prbs_size] = prb_info_t(grp.startPrb, grp.nPrb);
                sym_prbs->prbs_size++;
            }

            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};
            prb_info.common.ap_index = 0; //Init AP_INDEX as this is the first UE in the new UE-Group
            prb_info.common.direction = fh_dir_t::FH_DIR_UL;
            if(ru == SINGLE_SECT_MODE)
            {
                // prb_info.common.numSymbols = OFDM_SYMBOLS_PER_SLOT;
                prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_ul_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_ul_symbols);
            }
            /// No Precoding enabled for UL channels
            if ( mmimo_enabled && pmi_bf_pdu.dig_bf_interfaces != 0){
                prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
            }
            else if (mmimo_enabled && pmi_bf_pdu.dig_bf_interfaces == 0) {
                prb_info.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                //NVLOGD_FMT(TAG, "{}:{} ue.rnti {} ue.dmrsPortBmsk {} ue.scid {}", __FILE__, __LINE__, ue.rnti, ue.dmrsPortBmsk, ue.scid);
                track_eaxcids_fh(ue, prb_info.common);
            }

            if(bf_enabled)
            {
                update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu,mmimo_enabled, prb_info, cell_index);
                if (bfwCoeff_mem_info != NULL && *bfwCoeff_mem_info->header == BFW_COFF_MEM_BUSY)
                {
                    if ((is_latest_bfw_coff_avail(cell_cmd.slot.slot_3gpp.sfn_ , cell_cmd.slot.slot_3gpp.slot_,
                            bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot)) && (pmi_bf_pdu.dig_bf_interfaces==0))
                    {
                        NVLOGD_FMT(TAG, "PUSCH new Grp :: Header {} SFN {} SLOT {} BFW_SFN {} BFW_SLOT {}",
                                        *bfwCoeff_mem_info->header, cell_cmd.slot.slot_3gpp.sfn_, cell_cmd.slot.slot_3gpp.slot_,
                                         bfwCoeff_mem_info->sfn, bfwCoeff_mem_info->slot);
                        //prb_info.common.numApIndices = pmi_bf_pdu.dig_bf_interfaces;
                        //prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                        prb_info.common.extType = 11;
                        prb_info.bfwCoeff_buf_info.num_prgs = pmi_bf_pdu.num_prgs;
                        prb_info.bfwCoeff_buf_info.prg_size = pmi_bf_pdu.prg_size;
                        prb_info.bfwCoeff_buf_info.dig_bf_interfaces = pmi_bf_pdu.dig_bf_interfaces;
                        prb_info.bfwCoeff_buf_info.nGnbAnt = bfwCoeff_mem_info->nGnbAnt;
                        prb_info.bfwCoeff_buf_info.header = bfwCoeff_mem_info->header;
                        prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h = bfwCoeff_mem_info->buff_addr_chunk_h[bfwUeGrpIndex];
                        prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_d = bfwCoeff_mem_info->buff_addr_chunk_d[bfwUeGrpIndex];
                        NVLOGD_FMT(TAG, "PUSCH new Grp dig_bf_interfaces={}, prb_info.common.portMask {} bfwUeGrpIndex={} prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h={}",
                                    pmi_bf_pdu.dig_bf_interfaces, static_cast<uint32_t>(prb_info.common.portMask), bfwUeGrpIndex, static_cast<void *>(prb_info.bfwCoeff_buf_info.p_buf_bfwCoef_h));
                    }
                }
            }
            if(ru == SINGLE_SECT_MODE)
            {
                uint8_t start_symbol = (slot_detail == nullptr ? 0: slot_detail->start_sym_ul);
                update_prb_sym_list(*sym_prbs, index, start_symbol, 1, channel_type::PUSCH, ru);
            }
            else
            {
                prb_info.common.numSymbols = msg.num_of_symbols;
                update_prb_sym_list(*sym_prbs, index, msg.start_symbol_index, 1, channel_type::PUSCH, ru);
            }
        }
        else if (mmimo_enabled && pmi_bf_pdu.dig_bf_interfaces == 0)
        {
                auto& idxlist{sym_prbs->symbols[grp.puschStartSym][channel_type::PUSCH]};

                auto iter = std::find_if(idxlist.begin(), idxlist.end(),[&sym_prbs, &prbs, &grp, &ru, &ul_bandwidth](const auto& e){
                    bool retval = (e < sym_prbs->prbs_size);
                    if (retval) {
                        auto& prb{prbs[e]};
                        if(ru == SINGLE_SECT_MODE)
                        {
                            retval = (prb.common.startPrbc == 0 && prb.common.numPrbc == ul_bandwidth);
                        }
                        else
                        {
                            retval = (prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == grp.nPrb);
                        }
                    }
                    return retval;
                });

                if (iter != idxlist.end()){
                    auto& prb{prbs[*iter]};
                            prb.common.portMask |= calculate_dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
                            //NVLOGD_FMT(TAG, "{}:{} not pm_enabled prb_info.common.portMask {}, ue.dmrsPortBmsk {}, ue.scid {}", __FILE__, __LINE__, static_cast<uint32_t>(prb.common.portMask), static_cast<uint64_t>(ue.dmrsPortBmsk), ue.scid);
                            track_eaxcids_fh(ue, prb.common);
                    }
        }
    }

    inline void update_fh_params_pucch(cuphyPucchUciPrm_t& uci_info, uint16_t prb_size, const scf_fapi_rx_beamforming_t& pmi_bf_pdu, cell_sub_command& cell_cmd, bool bf_enabled,
                                       enum ru_type ru, nv::slot_detail_t* slot_detail, bool mmimo_enabled, int32_t cell_index, uint16_t ul_bandwidth) {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        if(ru == SINGLE_SECT_MODE)
        {
            bool value = ifAnySymbolPresent(sym_prbs->symbols, UL_NON_PRACH_CHANNEL_MASK);

            if(value)
                return;

            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(0, ul_bandwidth);
            sym_prbs->prbs_size++;
            std::size_t index{sym_prbs->prbs_size - 1};

            prb_info_t& prb_info{prbs[index]};

            prb_info.common.direction  = fh_dir_t::FH_DIR_UL;
            prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_ul_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_ul_symbols);

            if(bf_enabled)
            {
                update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu,mmimo_enabled, prb_info, cell_index);
            }
            uint8_t start_symbol = (slot_detail == nullptr ? 0 : slot_detail->start_sym_ul);
            update_prb_sym_list(*sym_prbs, index, start_symbol, 1, channel_type::PUCCH, ru);
        }
        else
        {
            // No freqHopping startPrb, prbSize should be sufficient
            // FreqHopping there will be two entries one after the other
            // (startPrb,  prbSize) (secondHopPrb, prbSize)
            uint16_t startCrb = uci_info.startPrb + uci_info.bwpStart;
            // Assume that if 2 PUCCH PDUs have the same startPrb, numPrb and numSym, then the numSym is the same too
            auto& symb = sym_prbs->symbols[uci_info.startSym];

            for(std::size_t existing_prb_index = 0; existing_prb_index < symb[channel_type::PUCCH].size(); existing_prb_index++)
            {
                if(sym_prbs->prbs_size > symb[channel_type::PUCCH][existing_prb_index])
                {
                    auto& prb = prbs[symb[channel_type::PUCCH][existing_prb_index]];
                    if(prb.common.startPrbc == startCrb && prb.common.numPrbc == uci_info.prbSize)
                    {
                        return;
                    }
                }
            }

            if(!uci_info.freqHopFlag)
            {
                check_prb_info_size(sym_prbs->prbs_size);
                prbs[sym_prbs->prbs_size] = prb_info_t(startCrb, prb_size);
                sym_prbs->prbs_size++;
                std::size_t index{sym_prbs->prbs_size - 1};
                prb_info_t& prb_info{prbs[index]};

                prb_info.common.direction = fh_dir_t::FH_DIR_UL;

                if(bf_enabled)
                {
                    update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);
                }
                prb_info.common.numSymbols = uci_info.nSym;

                if ( mmimo_enabled && pmi_bf_pdu.dig_bf_interfaces != 0){
                    prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                    uci_info.nUplinkStreams = pmi_bf_pdu.dig_bf_interfaces;
                }
                update_prb_sym_list(*sym_prbs, index, uci_info.startSym, 1, channel_type::PUCCH, ru);
            }
            else if(uci_info.nSym > 1)
            {
                /// Eg Start Sym = 3 , nSym = 11
                // firstHopStart = 3
                uint16_t firstHopStart{uci_info.startSym};
                // firstHopEnd = 8
                uint16_t hopNsym{static_cast<uint16_t>(uci_info.nSym / 2)};
                // secondHopStart = 8
                uint16_t secondHopStart{static_cast<uint16_t>(uci_info.startSym + hopNsym)};
                // secondHopEnd = 13
                uint16_t secondHopNsym{static_cast<uint16_t>(uci_info.nSym - hopNsym)};
                check_prb_info_size(sym_prbs->prbs_size);
                prbs[sym_prbs->prbs_size] = prb_info_t(startCrb, prb_size);
                sym_prbs->prbs_size++;

                std::size_t index{sym_prbs->prbs_size - 1};
                prb_info_t& prb_info{prbs[index]};
                prb_info.common.direction = fh_dir_t::FH_DIR_UL;

                if(bf_enabled)
                {
                    update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info, cell_index);
                }
                prb_info.common.numSymbols = hopNsym;

                if ( mmimo_enabled && pmi_bf_pdu.dig_bf_interfaces != 0){
                    prb_info.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                    uci_info.nUplinkStreams = pmi_bf_pdu.dig_bf_interfaces;
                }
                update_prb_sym_list(*sym_prbs, index, firstHopStart, 1, channel_type::PUCCH, ru);

                check_prb_info_size(sym_prbs->prbs_size);
                prbs[sym_prbs->prbs_size] = prb_info_t(uci_info.secondHopPrb + uci_info.bwpStart, prb_size);
                sym_prbs->prbs_size++;
                index = sym_prbs->prbs_size - 1;

                prb_info_t& prb_info2{prbs[index]};
                prb_info2.common.direction = fh_dir_t::FH_DIR_UL;

                if(bf_enabled)
                {
                    update_beam_list(prb_info2.beams_array, prb_info2.beams_array_size, pmi_bf_pdu, mmimo_enabled, prb_info2, cell_index);
                }
                prb_info2.common.numSymbols = secondHopNsym;
                if ( mmimo_enabled && pmi_bf_pdu.dig_bf_interfaces != 0){
                    prb_info2.common.portMask = (1 << pmi_bf_pdu.dig_bf_interfaces) -1;
                    uci_info.nUplinkStreams = pmi_bf_pdu.dig_bf_interfaces;
                }
                update_prb_sym_list(*sym_prbs, index, secondHopStart, 1, channel_type::PUCCH, ru);
            }
        }
    }

    inline void update_fh_params_prach(nv::phy_config& config, nv::prach_addln_config_t& addln_config, const scf_fapi_prach_pdu_t& pdu, const scf_fapi_rx_beamforming_t & bf_pdu, cell_sub_command& cell_cmd,
                                       bool bf_enabled, enum ru_type ru, nv::slot_detail_t* slot_detail, bool mmimo_enabled, int32_t cell_index) {
        uint8_t n_ra_t{pdu.num_prach_ocas};
        uint8_t l0{pdu.prach_start_symbol};
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        uint16_t startPrb{0};  //config.prach_config_.root_sequence[pdu.num_ra].k1;
        uint16_t numPrb{addln_config.n_ra_rb};
        for (uint8_t i = 0; i < n_ra_t; i++)
        {
            /// FIXME see 38.211 sec 5.3.2 for addln_config.n_ra_slot
            uint8_t startSymb  = (l0 + addln_config.n_ra_dur * i + 14 * addln_config.n_ra_slot) % OFDM_SYMBOLS_PER_SLOT;

            check_prb_info_size(sym_prbs->prbs_size);
            prbs[sym_prbs->prbs_size] = prb_info_t(startPrb, numPrb);
            sym_prbs->prbs_size++;
            std::size_t index{sym_prbs->prbs_size - 1};
            prb_info_t& prb_info{prbs[index]};

            prb_info.common.freqOffset = config.prach_config_.root_sequence[pdu.num_ra].freqOffset;
            prb_info.common.numSymbols = addln_config.n_ra_dur;
            prb_info.common.direction = fh_dir_t::FH_DIR_UL;
            switch (pdu.prach_format) {
                case 0:
                case 1:
                case 2:
                    prb_info.common.filterIndex = 1;
                break;

                case 3:
                    prb_info.common.filterIndex = 2;
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                    prb_info.common.filterIndex = 3;
                default:
                    break;
            }
            if (bf_enabled) {
                // TODO: Add support for static beamforming for PRACH by passing mmimo_enabled when we support sending static beamforming coeffs in Section Type 3 in C-plane.
                update_beam_list(prb_info.beams_array, prb_info.beams_array_size, bf_pdu, false, prb_info, cell_index);
            }
            if ( mmimo_enabled && bf_pdu.dig_bf_interfaces != 0){
                prb_info.common.portMask = (1 << bf_pdu.dig_bf_interfaces) -1;
            }
            update_prb_sym_list(*sym_prbs, index, startSymb, 1, channel_type::PRACH, ru);
        }
    }

    // Moved out of anonymous namespace to allow external access for finalization when last PDU is dropped
    void finalize_srs_slot(cell_sub_command& cell_cmd, const scf_fapi_rx_beamforming_t& pmi_bf_pdu,
        uint8_t nSrsSym, uint8_t srsStartSym, srs_params *srs_params,
        bool bf_enabled, enum ru_type ru, nv::slot_detail_t* slot_detail, int32_t cell_index, bool last_non_prach_pdu)
    {
        auto sym_prbs{cell_cmd.sym_prb_info()};
        auto& prbs{sym_prbs->prbs};
        if(ru == SINGLE_SECT_MODE)
        {
            std::size_t& prbs_size = sym_prbs->prbs_size;
            prbs[prbs_size] = prb_info_t(0, 273);
            const std::size_t index = prbs_size++;
            prb_info_t& prb_info = prbs[index];

            prb_info.common.direction  = fh_dir_t::FH_DIR_UL;
            prb_info.common.numSymbols = 1;
#if 0
        if (bf_enabled)
        {
            update_beam_list(prb_info.beams_array, prb_info.beams_array_size, pmi_bf_pdu, prb_info, cell_index);
        }
#endif
            // If we tell it there are 2 symbols countSrsPrbs() will count 2*273*2 PRBs
            const uint16_t srs_end_sym = srsStartSym + nSrsSym;
            for(uint16_t symbol = srsStartSym; symbol < srs_end_sym; symbol++) {
                update_prb_sym_list(*sym_prbs, index, symbol, 1, channel_type::SRS, ru);
            }

            // If we get to this point in PDU processing and haven't requested anything, request PUSCH
            // Does this work if we only have SRS, and does it work if we have PUSCH
            const bool value = ifAnySymbolPresent(sym_prbs->symbols, UL_NON_PRACH_CHANNEL_MASK);
            if(!value && last_non_prach_pdu)
            {
                NVLOGI_FMT(TAG, "SRS PDU: last_non_prach_pdu={} srsStartSym={} nSrsSym={}", last_non_prach_pdu, srsStartSym, nSrsSym);
                prb_info.common.numSymbols = (slot_detail == nullptr || slot_detail->max_ul_symbols == 0 ? OFDM_SYMBOLS_PER_SLOT: slot_detail->max_ul_symbols);
                const uint8_t start_symbol = (slot_detail == nullptr ? 0: slot_detail->start_sym_ul);
                update_prb_sym_list(*sym_prbs, index, start_symbol, 1, channel_type::PUSCH, ru);
            }
        }
        else
        {
            // Cache frequently accessed values to reduce pointer dereferencing
            auto& rb_info_per_sym = srs_params->rb_info_per_sym[cell_index];
            auto& final_rb_info_per_sym = srs_params->final_rb_info_per_sym[cell_index];
            std::size_t& prbs_size = sym_prbs->prbs_size;

            const uint16_t srs_end_sym = srsStartSym + nSrsSym;

            for(uint16_t sym_idx = srsStartSym; sym_idx < srs_end_sym; sym_idx++)
            {
                auto& rb_info_current_sym = rb_info_per_sym[sym_idx];
                auto& final_rb_info_current_sym = final_rb_info_per_sym[sym_idx];

                // Process RB intervals for current symbol
                if(rb_info_current_sym.size() > 1)
                {
                    merge_srs_prb_interval(rb_info_current_sym, final_rb_info_current_sym);
                }
                else
                {
                    final_rb_info_current_sym = rb_info_current_sym;
                }

                // Process all PRB intervals for this symbol
                const auto final_rb_size = final_rb_info_current_sym.size();
                for (uint16_t prb_idx = 0; prb_idx < final_rb_size; prb_idx++)
                {
                    const auto& rb_pair = final_rb_info_current_sym[prb_idx];
                    const uint16_t startRB = rb_pair.first;
                    const uint16_t numRB = (rb_pair.second - rb_pair.first) + 1;

                    NVLOGD_FMT(TAG,"sym_idx {} prb_idx{} first {} second {} startRB {} numRB {}",
                              sym_idx, prb_idx, rb_pair.first, rb_pair.second, startRB, numRB);

                    // Create PRB info and update arrays
                    prbs[prbs_size] = prb_info_t(startRB, numRB);
                    const std::size_t index = prbs_size++;

                    prb_info_t& prb_info = prbs[index];
                    prb_info.common.direction = fh_dir_t::FH_DIR_UL;
                    update_prb_sym_list(*sym_prbs, index, sym_idx, 1, channel_type::SRS, ru);
                }
            }
        }
    }

    inline int8_t calc_start_prb(const scf_fapi_srs_pdu_t& msg, cell_sub_command& cell_cmd, srs_rb_info_t srs_rb_info[], uint8_t& nHops)
    {
        uint16_t hopStartPrbs[MAX_SRS_SYM] = {0};
        uint16_t numPrbs[MAX_SRS_SYM] = {0};
        uint16_t hopStartPrbs0[MAX_SRS_SYM] = {0};
        uint16_t nHopsInSlot = 0;
        uint16_t hopIdx = 0;
        uint16_t Nb = 0;
        uint16_t m_SRS_b = 0;
        uint16_t nb = 0;
        uint16_t n_SRS = 0;
        uint16_t slotIdx = 0;
        uint16_t PI_b = 0;
        uint16_t PI_bm1 = 0;
        uint16_t Fb = 0;
        uint16_t nSyms = srs_symb_idx_to_numSymb[msg.num_symbols];
        uint16_t nRepetitions = srs_rep_factor_idx_to_numRepFactor[msg.num_repetitions];
        uint16_t frequencyShift = msg.frequency_shift;
        uint16_t bandwidthIdx = msg.bandwidth_index;
        uint16_t configIdx = msg.config_index;
        uint16_t frequencyPosition = msg.frequency_position;
        uint16_t frequencyHopping = msg.frequency_hopping;
        uint16_t resourceType = msg.resource_type;
        uint16_t Tsrs = msg.t_srs;
        uint16_t Toffset = msg.t_offset;
        uint16_t nSlotsPerFrame = 20; // TODO: for mu=1 there are 20 slots in 1 Frame. Need to define macro.
        uint16_t frameNum = cell_cmd.slot.slot_3gpp.sfn_;
        uint16_t slotNum = cell_cmd.slot.slot_3gpp.slot_;

        for (uint8_t i = 0; i < MAX_SRS_SYM ; i++)
        {
            hopStartPrbs[i] = frequencyShift;
        }
        nHopsInSlot  = nSyms / nRepetitions;

        for (hopIdx = 0 ; hopIdx <= (nHopsInSlot - 1); hopIdx++)
        {
            for (uint8_t b = 0 ; b <= bandwidthIdx ; b++)
            {
                if (frequencyHopping >= bandwidthIdx)
                {
                    Nb      = srs_bw_table[configIdx].bsrs_info[b].nb;
                    m_SRS_b = srs_bw_table[configIdx].bsrs_info[b].mSRS;
                    nb      = ((4 * frequencyPosition / m_SRS_b) % Nb);
                }
                else
                {
                    Nb      = srs_bw_table[configIdx].bsrs_info[b].nb;
                    m_SRS_b = srs_bw_table[configIdx].bsrs_info[b].mSRS;
                    if (b <= frequencyHopping)
                    {
                        nb = ((4 * frequencyPosition / m_SRS_b) % Nb);
                    }
                    else
                    {
                        if (resourceType == 0)
                        {
                            n_SRS = hopIdx;
                        }
                        else
                        {
                            slotIdx = nSlotsPerFrame * frameNum + slotNum - Toffset;
                            if ((slotIdx % Tsrs) == 0)
                            {
                                n_SRS = (slotIdx / Tsrs) * (nSyms / nRepetitions) + hopIdx;
                            }
                            else
                            {
                                NVLOGC_FMT(TAG,"Not an SRS slot ...");
                                n_SRS = 0;
                                return 0;
                            }
                        }
                        PI_bm1 = 1;
                        for (uint8_t b_prime = frequencyHopping + 1; b_prime <= b-1 ; b_prime++)
                        {
                            PI_bm1 = PI_bm1 * srs_bw_table[configIdx].bsrs_info[b_prime].nb;
                        }
                        PI_b = PI_bm1 * Nb;
                        if ((Nb % 2) == 0)
                        {
                            Fb = (Nb / 2) * ((n_SRS % PI_b) / PI_bm1) + ((n_SRS % PI_b) / (2 * PI_bm1));
                        }
                        else
                        {
                            Fb = (Nb / 2) * (n_SRS / PI_bm1);
                        }
                        nb = ((Fb + (4 * frequencyPosition / m_SRS_b)) % Nb);
                    }
                }
                hopStartPrbs[hopIdx] = hopStartPrbs[hopIdx] +  m_SRS_b * nb + msg.bwp.bwp_start;
                numPrbs[hopIdx] = m_SRS_b;
            }
        }
        hopIdx = 0;
        for (uint8_t symbIdx = 0; symbIdx < nSyms; symbIdx += nRepetitions)
        {
            for (uint8_t repIdx = 0; repIdx < nRepetitions; repIdx++)
            {
                srs_rb_info[symbIdx + repIdx].srs_start_prbs = hopStartPrbs[hopIdx];
                srs_rb_info[symbIdx + repIdx].num_srs_prbs = numPrbs[hopIdx];
            }
            hopIdx++;
        }

        nHops = 1;
        hopStartPrbs0[0] = hopStartPrbs[0];
        for(int hopIdx0 = 1; hopIdx0 < nHopsInSlot; hopIdx0++)
        {
            bool newHopFlag = true;
            for(hopIdx = 0; hopIdx < nHops; ++hopIdx)
            {
                if(hopStartPrbs[hopIdx0] == hopStartPrbs0[hopIdx])
                {
                    newHopFlag = false;
                    break;
                }
            }
            if(newHopFlag)
            {
                nHops += 1;
            }
        } 
        return nSyms;
    }

    inline void merge_srs_prb_interval(srs_rb_info_per_sym_t& intervals, srs_rb_info_per_sym_t& final_intervals)
    {
        srs_rb_info_per_sym_t &p = intervals;
        srs_rb_info_per_sym_t &p1 = final_intervals;
        std::sort(p.begin(),p.end());
        uint16_t f=p[0].first, s=p[0].second;
        for(uint16_t i=0; i<p.size()-1; i++)
        {
            uint16_t a[2];
            if(s>=p[i+1].first)
            {
                s=std::max(s,p[i+1].second);
            }
            else
            {
                a[0]=f;
                a[1]=s;
                f=p[i+1].first;
                s=p[i+1].second;
                p1.push_back({a[0],a[1]});
            }
        }
        p1.push_back({f,s});
        return;
    }
} // namespace scf_5g_fapi
