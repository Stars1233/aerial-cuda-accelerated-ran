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

#include <algorithm>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <benchmark/benchmark.h>
#include "test_utils.hpp"
#include "scf_5g_slot_commands.hpp"
#include "yaml.hpp"

#include "tests_scf_5g_slot_commands_utils.hpp"
#include "scf_5g_slot_commands_pdsch_csirs.hpp"

namespace scf_5g_fapi {
    uint16_t getPdschCsirsPrbOverlap(
        uint16_t csirs_remap,
        bool use_alt_prb,
        uint16_t startPdschPrbOffset,
        uint16_t nPdschPrb,
        uint16_t* reMap,
        int32_t cell_idx);
}

namespace {
constexpr const char* CONFIG_LAUNCH_PATTERN_PATH = "testVectors/multi-cell/";
constexpr const char* CONFIG_TEST_VECTOR_PATH = "testVectors/";
constexpr int MAX_PATH_LEN = 1024;
#ifndef CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM
  inline constexpr int CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM = 4; // Set CUBB_HOME to N level parent directory of this process. Example: 4 means "../../../../"
#endif

#ifndef CB_TB_TAG
inline constexpr int CB_TB_TAG = 1000;
#endif

#define GTEST_COUT std::cout << "[          ] [ INFO ] "

Dataset load_tv_datasets_single(hdf5hpp::hdf5_file& hdf5file, std::string const dataset)
{
    Dataset d;
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(dataset.c_str());
    d.size = dset.get_buffer_size_bytes();
    NVLOGI_FMT(CB_TB_TAG,"Opened {}-byte dataset {}", d.size, dataset.c_str());
    void * fh_mem = malloc(d.size);
    if(fh_mem == nullptr)
    {
       NVLOGC_FMT(CB_TB_TAG,"allocate_memory failure ");
       throw std::bad_alloc();
    }
    d.data.reset(memset(fh_mem, 0, d.size));
    NVLOGI_FMT(CB_TB_TAG,"Reading {}-byte dataset {}", d.size, dataset.c_str());
    dset.read(d.data.get());
    NVLOGI_FMT(CB_TB_TAG,"Read {}-byte dataset {}", d.size, dataset.c_str());
    return std::move(d);
}

void parse_tvs_and_setup_output_reference(std::vector<std::vector<tvPdschPdu>>& all_pdsch_pdus, std::vector<std::vector<tvCsirsPdu>>& all_csirs_pdus, std::vector<std::string> tvs, std::shared_ptr<slot_command_api::cell_sub_command> *sp_cell_sub_cmds, slot_indication si)
{
    auto& csirs_lookup = csirs_lookup_api::CsirsLookup::getInstance();
    for(int cell_idx = 0; cell_idx < tvs.size(); ++cell_idx)
    {
        auto sym_prbs = sp_cell_sub_cmds[cell_idx]->sym_prb_info();
        auto& prbs{sym_prbs->prbs};
        auto& tv = tvs[cell_idx];
        if(tv == "")
        {
            GTEST_COUT << "No TV scheduled for Cell " << cell_idx << std::endl;
            continue;
        }
        std::vector<tvPdschPdu> pdsch_pdus;
        std::vector<tvCsirsPdu> csirs_pdus;

        char file_path[MAX_PATH_LEN];
        get_full_path_file(file_path, CONFIG_TEST_VECTOR_PATH, tv.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        
        if(access(file_path, F_OK) != 0)
        {
            printf("TV %s file not exist: %s", tv.c_str(), file_path);
            continue;
        }
        NVLOGI_FMT(CB_TB_TAG, "Accessing TV {}", file_path);
        auto file = hdf5hpp::hdf5_file::open(file_path);
        if(!file.is_valid_dataset(NUMPDU))
        {
            continue;
        }
        auto     dset = file.open_dataset(NUMPDU);
        uint32_t nPdu = 0;
        dset.read(&nPdu);
        NVLOGI_FMT(CB_TB_TAG, "Reading {} PDUs from {}", nPdu, file_path);
        bool hasCsirs = false;
        uint32_t csitype;
        for(uint i = 0; i < nPdu; i++)
        {
            NVLOGI_FMT(CB_TB_TAG, "Reading PDUs {}", i);
            std::string name = PDU + std::to_string(i + 1);
            if(!file.is_valid_dataset(name.c_str()))
            {
                continue;
            }
            auto     pdu_ds   = file.open_dataset(name.c_str())[0];
            uint32_t pdu_type = pdu_ds["type"].as<uint32_t>();
            if(pdu_type == TV_PDSCH)
            {
                tvPdschPdu pdu;
                pdu.readPdschPduFromH5(pdu_ds);
                // pdu.printTvPdschPdu();
                pdsch_pdus.push_back(pdu);
            }
            if(pdu_type == TV_CSI_RS)
            {
                tvCsirsPdu pdu;
                pdu.readCsirsPduFromH5(pdu_ds);
                csitype = pdu.CSIType;
                hasCsirs = true;
                // pdu.printTvCsirsPdu();
                csirs_pdus.push_back(pdu);
            }
        }

        int prevStartSym = -1;
        int prevNumSym = -1;
        int prevStartPrb = -1;
        int prevNumPrb = -1;
        
        for(auto& pdu : pdsch_pdus)
        {
            if(prevStartSym != pdu.StartSymbolIndex ||
                prevNumSym != pdu.NrOfSymbols ||
                prevStartPrb != pdu.rbStart ||
                prevNumPrb != pdu.rbSize)
            {
                pdu.isNewGrpStart = true;
                prevStartSym = pdu.StartSymbolIndex;
                prevNumSym = pdu.NrOfSymbols;
                prevStartPrb = pdu.rbStart;
                prevNumPrb = pdu.rbSize;
            }
            else
            {
                pdu.isNewGrpStart = false;
            }
        }

        //Parse RE map from TV
        std::vector<uint16_t> reMaskGrid;
        std::vector<uint16_t> reMaskGridZP;
        if(hasCsirs)
        {
            if(file.is_valid_dataset(NZP_REMAP))
            {
                Dataset re_map = std::move(load_tv_datasets_single(file, NZP_REMAP));
                auto re_map_array = static_cast<uint32_t*>(re_map.data.get());
                for(uint32_t re_index = 0, reMask = 0, cnt = 0; re_index < re_map.size / sizeof(re_map_array[0]); ++re_index)
                {
                    reMask = (reMask << 1) | re_map_array[re_index];
                    if(++cnt == PRB_NUM_RE) {
                        reMaskGrid.push_back(reMask);
                        cnt = 0;
                        reMask = 0;
                    }
                }
            }
            if(file.is_valid_dataset(ZP_REMAP))
            {
                Dataset re_map = std::move(load_tv_datasets_single(file, ZP_REMAP));
                auto re_map_array = static_cast<uint32_t*>(re_map.data.get());
                for(uint32_t re_index = 0, reMask = 0, cnt = 0; re_index < re_map.size / sizeof(re_map_array[0]); ++re_index)
                {
                    reMask = (reMask << 1) | re_map_array[re_index];
                    if(++cnt == PRB_NUM_RE) {
                        reMaskGridZP.push_back(reMask);
                        cnt = 0;
                        reMask = 0;
                    }
                }
            }
        }

        std::array<uint32_t, OFDM_SYMBOLS_PER_SLOT> csiTypeSymbol;
        std::array<uint64_t, OFDM_SYMBOLS_PER_SLOT> csiTypePortMask;
        uint16_t symbolMapArray = 0;
        for(auto& portMask: csiTypePortMask)
        {
            portMask = 0;
        }
        for(auto& type: csiTypeSymbol)
        {
            type = __UINT32_MAX__;
        }
        for(int pdu_index = 0; pdu_index < csirs_pdus.size(); ++pdu_index)
        {
            auto& pdu = csirs_pdus[pdu_index];
            const csirs_lookup_api::CsirsPortData *csirs_port_data = nullptr;
            auto ret = csirs_lookup.getPortInfo(pdu.Row, pdu.FreqDomain, pdu.SymbL0, pdu.SymbL1, csirs_port_data);
            if(!ret)
            {
                continue;
            }
            auto startPrb = pdu.StartRB;
            auto numPrb = pdu.NrOfRBs;
            auto num_ports = csirs_port_data->num_ports;
            uint16_t symbolMask = 0;
            for(int port = 0; port < num_ports; port++)
            {
                symbolMask |= csirs_port_data->port_tx_locations[port].symbol_mask;
            }
            for(int sym = 0; sym < OFDM_SYMBOLS_PER_SLOT; sym++){
                if((symbolMask & (1 << sym)) == 0)
                {
                    continue;
                }
                symbolMapArray |= 1 << sym;
                
                csiTypeSymbol[sym] = pdu.CSIType;
                if(pdu.CSIType == ZP_CSI_RS)
                {
                    continue;
                }
                prbs[sym_prbs->prbs_size] = prb_info_t(startPrb, numPrb); // TODO update for RAT0
                scf_5g_fapi::update_prb_sym_list(*sym_prbs, sym_prbs->prbs_size, sym, 1, channel_type::CSI_RS, OTHER_MODE);
                prbs[sym_prbs->prbs_size].common.numApIndices = num_ports;
                prbs[sym_prbs->prbs_size].common.portMask = (1<<num_ports) -1;
                csiTypePortMask[sym] = prbs[sym_prbs->prbs_size].common.portMask;
                prbs[sym_prbs->prbs_size].common.reMask = reMaskGrid[sym * MAX_N_PRBS_SUPPORTED + startPrb]; // unused field for 4T4R DL

                NVLOGI_FMT(CB_TB_TAG, "CSI_RS PDU {} symbol {} reMaskGrid:{:x} reMaskGridZP:{:x}", pdu_index, sym, reMaskGrid[sym * MAX_N_PRBS_SUPPORTED + startPrb], reMaskGridZP[sym * MAX_N_PRBS_SUPPORTED + startPrb]);
                sym_prbs->prbs_size++;
            }
        }

        // Generate reference slot_info struct
        for(int pdu_index = 0; pdu_index < pdsch_pdus.size(); ++pdu_index)
        {
            auto& pdu = pdsch_pdus[pdu_index];
            if(!pdu.isNewGrpStart)
            {
                continue;
            }
            auto startPrb = pdu.rbStart;
            auto numPrb = pdu.rbSize;
            auto startSym = pdu.StartSymbolIndex;
            auto numSym = pdu.NrOfSymbols;
            NVLOGI_FMT(CB_TB_TAG, "PDSCH PDU {} startSym {} numSym {} startPrb {} numPrb {}", pdu_index, startSym, numSym, startPrb, numPrb);
            if(csirs_pdus.size() > 0)
            {
                for(int sym = startSym; sym < startSym+numSym; ++sym)
                {
                    uint32_t mask = 1 << sym;

                    if(sym_prbs->symbols[sym][channel_type::CSI_RS].size() > 0)
                    {
                        for(int i = 0; i < sym_prbs->symbols[sym][channel_type::CSI_RS].size(); ++i)
                        {
                            prbs[sym_prbs->prbs_size] = prb_info_t(startPrb, numPrb); // TODO update for RAT0
                            auto csirs_startPrb = sym_prbs->prbs[sym_prbs->symbols[sym][channel_type::CSI_RS][i]].common.startPrbc;
                            auto csirs_numPrb = sym_prbs->prbs[sym_prbs->symbols[sym][channel_type::CSI_RS][i]].common.numPrbc;
                            NVLOGI_FMT(CB_TB_TAG, "PDSCH PDU {} startSym {} numSym {} startPrb {} numPrb {} csirs_startPrb {} csirs_numPrb {}", pdu_index, startSym, numSym, startPrb, numPrb, csirs_startPrb, csirs_numPrb);

                            bool overlappingCondition = std::max<unsigned int>(startPrb, csirs_startPrb) < std::min<unsigned int>(startPrb+numPrb, csirs_startPrb+csirs_numPrb);
                            if(overlappingCondition)
                            {
                                //Assuming complete overlap
                                //TODO handle partial overlap - this will require some logic for splitting the PDSCH/CSI_RS blocks based on different ReMasks.
                                // currently no nrSim test cases test one PDSCH PDU overlapping with different CSI_RS types
                                // The meat of the validation is here: using the re_map from the TV to validate the reMask instead of the FH callback
                                int reMask_idx = sym * MAX_N_PRBS_SUPPORTED + csirs_startPrb;
                                if(csiTypeSymbol[sym] != ZP_CSI_RS)
                                {
                                    auto csirs_reMask = reMaskGrid[reMask_idx];
                                    prbs[sym_prbs->prbs_size].common.reMask = ~csirs_reMask;
                                }
                                else
                                {
                                    auto csirs_reMask = reMaskGridZP[reMask_idx];
                                    prbs[sym_prbs->prbs_size].common.reMask = ~csirs_reMask;
                                }
                                scf_5g_fapi::update_prb_sym_list(*sym_prbs, sym_prbs->prbs_size, sym, 1, channel_type::PDSCH_CSIRS, OTHER_MODE);
                                prbs[sym_prbs->prbs_size].common.numApIndices = 0; // unused field for 4T4R DL
                                prbs[sym_prbs->prbs_size].common.portMask |= static_cast<uint64_t>(pdu.dmrsPorts) << pdu.SCID * 8;
                                prbs[sym_prbs->prbs_size].common.portMask |= csiTypePortMask[sym];
                                NVLOGI_FMT(CB_TB_TAG, "Adding PDSCH_CSIRS symbol {} numSym hc 1 startPrb {} numPrb {} portMask {}", sym, startPrb, numPrb, static_cast<uint64_t>(prbs[sym_prbs->prbs_size].common.portMask));
                                sym_prbs->prbs_size++;
                                break;
                            }
                        }
                    }
                    else if(mask & symbolMapArray) // ZP_CSIRS with no othee CSI_RS PDUs
                    {
                        prbs[sym_prbs->prbs_size] = prb_info_t(startPrb, numPrb); // TODO update for RAT0
                        scf_5g_fapi::update_prb_sym_list(*sym_prbs, sym_prbs->prbs_size, sym, 1, channel_type::PDSCH_CSIRS, OTHER_MODE);
                        int reMask_idx = sym * MAX_N_PRBS_SUPPORTED + startPrb;
                        auto csirs_reMask = reMaskGridZP[reMask_idx];
                        prbs[sym_prbs->prbs_size].common.reMask = ~csirs_reMask;
                        prbs[sym_prbs->prbs_size].common.numApIndices = 0; // unused field for 4T4R DL
                        prbs[sym_prbs->prbs_size].common.portMask |= static_cast<uint64_t>(pdu.dmrsPorts) << pdu.SCID * 8;
                        prbs[sym_prbs->prbs_size].common.portMask |= csiTypePortMask[sym];
                        NVLOGI_FMT(CB_TB_TAG, "Adding ZP PDSCH_CSIRS symbol {} numSym hc 1 startPrb {} numPrb {} portMask {}", sym, startPrb, numPrb, static_cast<uint64_t>(prbs[sym_prbs->prbs_size].common.portMask));
                        sym_prbs->prbs_size++;
                    }
                    else
                    {
                        prbs[sym_prbs->prbs_size] = prb_info_t(startPrb, numPrb); // TODO update for RAT0
                        scf_5g_fapi::update_prb_sym_list(*sym_prbs, sym_prbs->prbs_size, sym, 1, channel_type::PDSCH, OTHER_MODE);
                        prbs[sym_prbs->prbs_size].common.numApIndices = 0; // unused field for 4T4R DL
                        prbs[sym_prbs->prbs_size].common.portMask |= static_cast<uint64_t>(pdu.dmrsPorts) << pdu.SCID * 8;
                        NVLOGI_FMT(CB_TB_TAG, "Adding PDSCH symbol {} numSym hc 1 startPrb {} numPrb {} portMask {}", sym, startPrb, numPrb, static_cast<uint64_t>(prbs[sym_prbs->prbs_size].common.portMask));
                        sym_prbs->prbs_size++;
                    }
                }
            }
            else
            {
                prbs[sym_prbs->prbs_size] = prb_info_t(startPrb, numPrb);
                NVLOGI_FMT(CB_TB_TAG, "Adding PDSCH startSym {} numSym {} startPrb {} numPrb {}", startSym, numSym, startPrb, numPrb);
                scf_5g_fapi::update_prb_sym_list(*sym_prbs, sym_prbs->prbs_size, startSym, numSym, channel_type::PDSCH, OTHER_MODE);
                prbs[sym_prbs->prbs_size].common.numApIndices = 0; // unused field for 4T4R DL
                prbs[sym_prbs->prbs_size].common.portMask |= static_cast<uint64_t>(pdu.dmrsPorts) << pdu.SCID * 8;
                sym_prbs->prbs_size++;
            }
        }

        file.close();
        all_csirs_pdus.push_back(csirs_pdus);
        all_pdsch_pdus.push_back(pdsch_pdus);
    }

#ifdef DEBUG_SYM_PRB_INFO_STRUCT
    for(int cell_idx = 0; cell_idx < tvs.size(); ++cell_idx)
    {
        scf_5g_fapi::print_sym_prb_info(si.sfn_, si.slot_, sp_cell_sub_cmds[cell_idx]->sym_prb_info(), cell_idx);
    }
#endif
}

void setup_fhcb_csirs_pdsch_input_command(std::shared_ptr<cell_group_command> cell_grp_cmd,
    std::shared_ptr<slot_command_api::cell_sub_command> *sp_cell_sub_cmds,
    std::vector<std::vector<tvPdschPdu>>& all_pdsch_pdus,
    std::vector<std::vector<tvCsirsPdu>>& all_csirs_pdus,
    slot_indication si)
{
    auto csirs_params = cell_grp_cmd->get_csirs_params();
    auto pdsch_params = cell_grp_cmd->get_pdsch_params();
    // Fill out PDSCH per-cell parameters
    int pdsch_ue_grp_idx = 0;
    int prev_ue_grp_idx = -1;
    int pdu_ue_grp_start_idx = 0;
    int pdsch_ue_idx = 0;
    auto num_cells = all_pdsch_pdus.size();
    cell_grp_cmd->fh_params.total_num_pdsch_pdus = 0;

    for(int cell_idx = 0; cell_idx < num_cells; ++cell_idx)
    {
        int pdsch_fh_param_idx = 0;
        auto nUes = all_pdsch_pdus[cell_idx].size();
        pdu_ue_grp_start_idx = pdsch_ue_grp_idx;
        for (int ue_idx = 0; ue_idx < nUes; ue_idx++)
        {
            if (cell_grp_cmd->fh_params.num_pdsch_fh_params.at(cell_idx) == 0) {
                cell_grp_cmd->fh_params.start_index_pdsch_fh_params.at(cell_idx) = cell_grp_cmd->fh_params.total_num_pdsch_pdus;
            }
            ++cell_grp_cmd->fh_params.num_pdsch_fh_params.at(cell_idx);

            auto& pdsch_pdu = all_pdsch_pdus[cell_idx][ue_idx];
            auto& pdsch_fh_params = cell_grp_cmd->fh_params.pdsch_fh_params[cell_grp_cmd->fh_params.total_num_pdsch_pdus];
            pdsch_fh_params.cell_cmd = sp_cell_sub_cmds[cell_idx].get();
            pdsch_fh_params.ue = &pdsch_params->ue_info[pdsch_ue_idx];
            pdsch_fh_params.grp = &pdsch_params->ue_grp_info[pdsch_ue_grp_idx];
            pdsch_fh_params.cell_index = cell_idx;
            pdsch_fh_params.num_dl_prb = pdsch_pdu.BWPSize;
            pdsch_fh_params.ue_grp_index = pdsch_ue_grp_idx;
            pdsch_fh_params.bf_enabled = 0;
            pdsch_fh_params.pm_enabled = 0;
            pdsch_fh_params.mmimo_enabled = 0;
            std::size_t ueGrpIndex = pdsch_pdu.idxUeg + pdu_ue_grp_start_idx;
            if(ueGrpIndex != prev_ue_grp_idx)
            {
                prev_ue_grp_idx = ueGrpIndex;
                pdsch_fh_params.is_new_grp = 1;
                pdsch_params->ue_grp_info[pdsch_ue_grp_idx].startPrb = pdsch_pdu.rbStart;
                pdsch_params->ue_grp_info[pdsch_ue_grp_idx].nPrb = pdsch_pdu.rbSize;
                pdsch_params->ue_grp_info[pdsch_ue_grp_idx].pdschStartSym = pdsch_pdu.StartSymbolIndex;
                pdsch_params->ue_grp_info[pdsch_ue_grp_idx].nPdschSym = pdsch_pdu.NrOfSymbols;
                pdsch_params->ue_grp_info[pdsch_ue_grp_idx].resourceAlloc = pdsch_pdu.resourceAlloc;
                pdsch_ue_grp_idx++;
            }
            else
            {
                pdsch_fh_params.is_new_grp = 0;
            }


            pdsch_params->ue_info[pdsch_ue_idx].scid = pdsch_pdu.SCID;
            pdsch_params->ue_info[pdsch_ue_idx].nUeLayers = pdsch_pdu.nrOfLayers;
            pdsch_params->ue_info[pdsch_ue_idx].dmrsPortBmsk = pdsch_pdu.dmrsPorts;

            pdsch_ue_idx++;
            pdsch_fh_param_idx++;
            cell_grp_cmd->fh_params.total_num_pdsch_pdus++;
        }
        cell_grp_cmd->fh_params.num_pdsch_fh_params[cell_idx] = pdsch_fh_param_idx;
    }

    pdsch_params->cell_grp_info.nUeGrps = pdsch_ue_grp_idx;
    
    int csirsList_idx = 0;
    for(int cell_idx = 0; cell_idx < all_csirs_pdus.size(); ++cell_idx)
    {
        auto nPdus = all_csirs_pdus[cell_idx].size();
        if(nPdus == 0)
        {
            csirs_params->symbolMapArray[cell_idx] = 0;
            continue;
        }
        csirs_params->cell_index_list.push_back(cell_idx);
        int csirs_start_idx = csirsList_idx;
        // Linkage between PDSCH and CSI-RS
        pdsch_params->cell_grp_info.pCellPrms[cell_idx].csiRsPrmsOffset = csirsList_idx;
        pdsch_params->cell_grp_info.pCellPrms[cell_idx].cellPrmStatIdx = cell_idx;
        pdsch_params->cell_grp_info.pCellPrms[cell_idx].cellPrmDynIdx = cell_idx;
        cell_grp_cmd->fh_params.csirs_fh_params[cell_idx].cell_cmd = sp_cell_sub_cmds[cell_idx].get();
        cell_grp_cmd->fh_params.csirs_fh_params[cell_idx].cell_idx = cell_idx;
        cell_grp_cmd->fh_params.csirs_fh_params[cell_idx].cuphy_params_cell_idx = cell_idx;
        cell_grp_cmd->fh_params.csirs_fh_params[cell_idx].num_dl_prb = 273;
        cell_grp_cmd->fh_params.csirs_fh_params[cell_idx].bf_enabled = 0;
        for (int pdu_idx = 0; pdu_idx < nPdus; pdu_idx++)
        {
            auto& csirs_pdu = all_csirs_pdus[cell_idx][pdu_idx];

            csirs_params->csirsList[csirsList_idx].startRb = csirs_pdu.StartRB;
            csirs_params->csirsList[csirsList_idx].nRb = csirs_pdu.NrOfRBs;
            csirs_params->csirsList[csirsList_idx].freqDomain = csirs_pdu.FreqDomain;
            csirs_params->csirsList[csirsList_idx].row = csirs_pdu.Row;
            csirs_params->csirsList[csirsList_idx].symbL0 = csirs_pdu.SymbL0;
            csirs_params->csirsList[csirsList_idx].symbL1 = csirs_pdu.SymbL1;
            csirs_params->csirsList[csirsList_idx].freqDensity = csirs_pdu.FreqDensity;
            csirs_params->csirsList[csirsList_idx].scrambId = csirs_pdu.ScrambId;
            csirs_params->csirsList[csirsList_idx].idxSlotInFrame = si.slot_;
            csirs_params->csirsList[csirsList_idx].csiType = static_cast<cuphyCsiType_t>(csirs_pdu.CSIType);
            csirs_params->csirsList[csirsList_idx].cdmType = static_cast<cuphyCdmType_t>(csirs_pdu.CDMType);
            csirsList_idx++;
            csirs_params->cellInfo[cell_idx].nRrcParams++;
            csirs_params->cellInfo[cell_idx].cellPrmStatIdx = cell_idx;
        }
        csirs_params->nCells++;
        pdsch_params->cell_grp_info.pCellPrms[cell_idx].nCsiRsPrms = csirsList_idx - csirs_start_idx;
        pdsch_params->cell_grp_info.pCsiRsPrms = &csirs_params->csirsList[0];
        cell_grp_cmd->fh_params.is_csirs_cell[cell_idx] = 1; // nPdus > 0 guaranteed by continue statement at line 422
    }
}

void setup_tvs(yaml::node& schedule, int slot, int num_cells, std::vector<std::string>& tvs)
{
    if(schedule.length() == 0)
    {
        NVLOGE_FMT(CB_TB_TAG, AERIAL_L2ADAPTER_EVENT, "Error: Empty schedule, cannot setup test vectors");
        return;
    }
    slot = slot % schedule.length();
    auto node = schedule[slot];
    auto slot_config = node["config"];
    auto slot_id = node["slot"].as<int>();
    for(int i = 0; i < slot_config.length(); ++i)
    {
        auto cell_node = slot_config[i];
        auto cell_id = cell_node["cell_index"].as<int>();
        if(cell_id != i)
        {
            NVLOGI_FMT(CB_TB_TAG, "Error: Launch pattern cells are no in order!");
            GTEST_COUT << "Error: Launch pattern cells are no in order!" << std::endl;
        }
        auto tv_node = cell_node["channels"];
        if(tv_node.length() == 0)
        {
            tvs.push_back("");
            NVLOGI_FMT(CB_TB_TAG, "Cell {} No TV scheduled", i);
        }
        else
        {
            auto tv_name = tv_node[(int)0].as<std::string>();
            tvs.push_back(tv_name);
            // printf("Cell %d TV: %s\n", i, tv_name.c_str());
        }
    }
}

#if 0
void calc_tf_grid(const cell_group_command& s, tf_grid_t& tf_grid)
{
    // Parse all CSI-RS PDUs first
    for (int k=0; k<s.num_csirs_params; k++)
    {
        golden_calc_csirs_re_mask(s.csirs_params[k], tf_grid);
    }

    for (int k=0; k<s.num_pdsch_params; k++)
    {
        auto& startRb {s.pdsch_params[k].rbStart};
        auto endRbPlus1 {startRb + s.pdsch_params[k].rbSize};
        auto& startSym {s.pdsch_params[k].StartSymbolIndex};
        auto endSymPlus1 {startSym + s.pdsch_params[k].NrOfSymbols};
        for (int sym=startSym; sym<endSymPlus1; sym++)
        {
            for (int rb=startRb; rb<endRbPlus1; rb++)
            {
                auto& numPorts {s.pdsch_params[k].numPorts};
                for (int portIdx=0; portIdx<numPorts; portIdx++)
                {
                    auto& beamIdx {s.pdsch_params[k].beamIdxList[portIdx]};
                    auto csirs_reMask = tf_grid[sym][rb][portIdx][0] & 0xfff;
                    if (csirs_reMask == 0)
                    {
                        tf_grid[sym][rb][portIdx][0] = (beamIdx << 16) | 0xfff;
                    }
                    else
                    {
                        tf_grid[sym][rb][portIdx][1] = (beamIdx << 16) | (~csirs_reMask & 0xfff);
                    }
                }
            }
        }
    }
}

inline void slot_info_update(slot_info_t& slot_info, channel_type channel, int startRb, int numRb, int portIdx, int beamIdx, int reMask, int symIdx, int numSym)
{
    {
        auto& prbs_size {slot_info.prbs_size};
        slot_info.prbs[prbs_size].startPrbc = startRb;
        slot_info.prbs[prbs_size].numPrbc = numRb;
        slot_info.prbs[prbs_size].reMask = reMask;
        slot_info.prbs[prbs_size].portIdxList[0] = portIdx;
        slot_info.prbs[prbs_size].beamIdxList[0] = beamIdx;
        slot_info.prbs[prbs_size].numPorts = 1;
        slot_info.prbs[prbs_size].startSym = symIdx;
        slot_info.prbs[prbs_size].numSym = 1;
        slot_info.symbols[symIdx][channel].push_back(prbs_size);
        prbs_size++;
    }
}

void calc_slot_info(const tf_grid_t& tf_grid, slot_info_t& slot_info)
{
    for (int symIdx=0; symIdx<MAX_SYMBOLS; symIdx++)
    {
        for (int portIdx=0; portIdx<MAX_PORTS; portIdx++)
        {
            for (int assignmentIdx=0; assignmentIdx<MAX_ASSIGNMENTS; assignmentIdx++)
            {
                int startRb = -1;
                uint32_t beamIdx_and_reMask;
                for (int rb=0; rb<MAX_PRBS; rb++)
                {
                    uint32_t current_beamIdx_and_reMask = tf_grid[symIdx][rb][portIdx][assignmentIdx];
                    if (current_beamIdx_and_reMask == 0)
                    {
                        if (startRb != -1)
                        {
                            int numRb = rb-startRb;
                            slot_info_update(slot_info,channel_type::PDSCH_CSIRS,startRb,numRb,portIdx,beamIdx_and_reMask >> 16,beamIdx_and_reMask & 0xfff,symIdx,1);
                            startRb = -1;
                        }
                    }
                    else
                    {
                        if (startRb == -1)
                        {
                            startRb = rb;
                            beamIdx_and_reMask = tf_grid[symIdx][rb][portIdx][assignmentIdx];
                        }
                        else if (current_beamIdx_and_reMask != beamIdx_and_reMask)
                        {
                            int numRb = rb-startRb;
                            slot_info_update(slot_info,channel_type::PDSCH_CSIRS,startRb,numRb,portIdx,beamIdx_and_reMask >> 16,beamIdx_and_reMask & 0xfff,symIdx,1);
                            startRb = rb;
                            beamIdx_and_reMask = tf_grid[symIdx][rb][portIdx][assignmentIdx];
                        }
                    }
                }
                if (startRb != -1)
                {
                    int numRb = 273-startRb;
                    slot_info_update(slot_info,channel_type::PDSCH_CSIRS,startRb,numRb,portIdx,beamIdx_and_reMask >> 16,beamIdx_and_reMask & 0xfff,symIdx,1);
                }
            }
        }
    }
}
#endif

void validate_output(std::shared_ptr<slot_command_api::cell_sub_command> *sp_cell_sub_cmds, std::shared_ptr<slot_command_api::cell_sub_command> *sp_cell_sub_cmds_ref, int num_cells)
{

    for(int cell_idx = 0; cell_idx < num_cells; ++cell_idx)
    {
        auto sym_prbs = sp_cell_sub_cmds[cell_idx]->sym_prb_info();
        auto sym_prbs_ref = sp_cell_sub_cmds_ref[cell_idx]->sym_prb_info();

        // More robust validation, flexible prb_info index
        EXPECT_EQ(sym_prbs->symbols.size(), sym_prbs_ref->symbols.size());
        for(int symbol_id = 0; symbol_id < std::min(sym_prbs->symbols.size(), sym_prbs_ref->symbols.size()); symbol_id++)
        {
            for (int channel_type = slot_command_api::channel_type::PDSCH_CSIRS; channel_type < slot_command_api::channel_type::CHANNEL_MAX; channel_type++)
            {
                EXPECT_EQ(sym_prbs->symbols[symbol_id][channel_type].size(), sym_prbs_ref->symbols[symbol_id][channel_type].size());
                for(int i = 0; i < std::min(sym_prbs->symbols[symbol_id][channel_type].size(), sym_prbs_ref->symbols[symbol_id][channel_type].size()); ++i)
                {
                    auto prb_info_idx = sym_prbs->symbols[symbol_id][channel_type][i];
                    auto prb_info_idx_ref = sym_prbs_ref->symbols[symbol_id][channel_type][i];
                    
                    auto &prb_info = sym_prbs->prbs[prb_info_idx];
                    auto &prb_info_ref = sym_prbs_ref->prbs[prb_info_idx_ref];
                    EXPECT_EQ(prb_info.common.startPrbc, prb_info_ref.common.startPrbc);
                    EXPECT_EQ(prb_info.common.numPrbc, prb_info_ref.common.numPrbc);
                    EXPECT_EQ(prb_info.common.reMask, prb_info_ref.common.reMask);
                    EXPECT_EQ(prb_info.common.extType, prb_info_ref.common.extType);
                    EXPECT_EQ(prb_info.common.numApIndices, prb_info_ref.common.numApIndices);
                    EXPECT_EQ(prb_info.common.freqOffset, prb_info_ref.common.freqOffset);
                    EXPECT_EQ(prb_info.common.numSymbols, prb_info_ref.common.numSymbols);
                    EXPECT_EQ(prb_info.common.direction, prb_info_ref.common.direction);
                    EXPECT_EQ(prb_info.common.filterIndex, prb_info_ref.common.filterIndex);
                    EXPECT_EQ(prb_info.common.portMask, prb_info_ref.common.portMask);
                    EXPECT_EQ(prb_info.common.pdschPortMask, prb_info_ref.common.pdschPortMask);
                    EXPECT_EQ(prb_info.common.useAltPrb, prb_info_ref.common.useAltPrb);
                    EXPECT_EQ(prb_info.common.isStaticBfwEncoded, prb_info_ref.common.isStaticBfwEncoded);
                    EXPECT_EQ(prb_info.common.isPdschSplitAcrossPrbInfo, prb_info_ref.common.isPdschSplitAcrossPrbInfo);
                }
            }
        }
    }
}
class FhCbTest : public ::testing::TestWithParam<std::tuple<std::string, int>> {
  // You can add common setup code here
};

TEST_P(FhCbTest,fhcb_test)
{
    int pattern_slot_number = std::get<1>(GetParam());
    std::string launch_pattern_file = std::get<0>(GetParam());
    GTEST_COUT << "Testing with TV: " << launch_pattern_file << " slot number: " << pattern_slot_number << std::endl;
    auto cell_grp_cmd = std::make_shared<cell_group_command>();
    std::shared_ptr<slot_command_api::cell_sub_command> sp_cell_sub_cmds[MAX_CELLS_PER_SLOT];
    std::shared_ptr<slot_command_api::cell_sub_command> sp_cell_sub_cmds_ref[MAX_CELLS_PER_SLOT];
    for (int k=0; k<MAX_CELLS_PER_SLOT; k++)
    {
        sp_cell_sub_cmds[k] = std::make_shared<slot_command_api::cell_sub_command>();
        sp_cell_sub_cmds_ref[k] = std::make_shared<slot_command_api::cell_sub_command>();
    }


    char pattern_file[MAX_PATH_LEN];
    get_full_path_file(pattern_file, CONFIG_LAUNCH_PATTERN_PATH, launch_pattern_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM); 

    yaml::file_parser fp(pattern_file);
    yaml::document    doc     = fp.next_document();
    yaml::node        pattern = doc.root();

    std::vector<std::string> tvs;
    yaml::node sched_node = pattern["SCHED"];
    yaml::node cell_configs = pattern["Cell_Configs"];
    int num_cells = cell_configs.length();
    setup_tvs(sched_node, pattern_slot_number, num_cells, tvs);
    slot_indication si;
    si.slot_ = pattern_slot_number % 20;
    si.sfn_ = pattern_slot_number / 20;
    std::vector<std::vector<tvPdschPdu>> all_pdsch_pdus;
    std::vector<std::vector<tvCsirsPdu>> all_csirs_pdus;
    cell_grp_cmd->slot.slot_3gpp = si;
    parse_tvs_and_setup_output_reference(all_pdsch_pdus, all_csirs_pdus, tvs, sp_cell_sub_cmds_ref, si);
    setup_fhcb_csirs_pdsch_input_command(cell_grp_cmd,sp_cell_sub_cmds,all_pdsch_pdus, all_csirs_pdus, si);

#if 0
    tf_grid_t golden_tf_grid, dut_tf_grid;
    slot_info_t golden_slot_info;
    init_tf_grid(golden_tf_grid);
    init_tf_grid(dut_tf_grid);
    calc_golden_tf_grid(cell_grp_cmd,golden_tf_grid);
    calc_golden_slot_info(golden_tf_grid, golden_slot_info);
#endif
    for (uint8_t cell = 0; cell < num_cells; ++cell) {
        scf_5g_fapi::LegacyCellGroupFhContext fh_context(*cell_grp_cmd, cell);
        scf_5g_fapi::fh_callback<false,true,false>(fh_context, nullptr);
    }

    validate_output(sp_cell_sub_cmds, sp_cell_sub_cmds_ref, num_cells);
}

INSTANTIATE_TEST_SUITE_P(
    F08_20C_59C,
    FhCbTest,
    testing::Combine(
        testing::Values("launch_pattern_F08_20C_59c.yaml"),
        testing::Range(0, 79)
    )
);

INSTANTIATE_TEST_SUITE_P(
    nrSim_90601,
    FhCbTest,
    testing::Combine(
        testing::Values("launch_pattern_nrSim_90601.yaml"),
        testing::Range(0, 39)
    )
);

INSTANTIATE_TEST_SUITE_P(
    nrSim_3337,
    FhCbTest,
    testing::Combine(
        testing::Values("launch_pattern_nrSim_3337.yaml"),
        testing::Range(0, 19)
    )
);

// INSTANTIATE_TEST_SUITE_P(
//     nrSim_3338,
//     FhCbTest,
//     testing::Combine(
//         testing::Values("launch_pattern_nrSim_3338.yaml"),
//         testing::Range(0, 19)
//     )
// );


static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int i = 0; i < 80; ++i) {
        b->Args({i});
    }
}

static void BM_scfl2adapter_fhcb_basic_csirs_pdsch(benchmark::State& state) {
    auto cell_grp_cmd = std::make_shared<cell_group_command>();
    std::shared_ptr<slot_command_api::cell_sub_command> sp_cell_sub_cmds[MAX_CELLS_PER_SLOT];
    std::shared_ptr<slot_command_api::cell_sub_command> sp_cell_sub_cmds_ref[MAX_CELLS_PER_SLOT];
    for (int k=0; k<MAX_CELLS_PER_SLOT; k++)
    {
        sp_cell_sub_cmds[k] = std::make_shared<slot_command_api::cell_sub_command>();
        sp_cell_sub_cmds_ref[k] = std::make_shared<slot_command_api::cell_sub_command>();
    }

    std::string launch_pattern_file("launch_pattern_F08_20C_59c.yaml");
    slot_indication si;
    auto pattern_slot_number = state.range(0);
    si.slot_ = pattern_slot_number % 20;
    si.sfn_ = pattern_slot_number / 20;
    char pattern_file[MAX_PATH_LEN];
    get_full_path_file(pattern_file, CONFIG_LAUNCH_PATTERN_PATH, launch_pattern_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM); 

    yaml::file_parser fp(pattern_file);
    yaml::document    doc     = fp.next_document();
    yaml::node        pattern = doc.root();

    std::vector<std::string> tvs;
    yaml::node sched_node = pattern["SCHED"];
    yaml::node cell_configs = pattern["Cell_Configs"];
    int num_cells = cell_configs.length();
    setup_tvs(sched_node, pattern_slot_number, num_cells, tvs);

    std::vector<std::vector<tvPdschPdu>> all_pdsch_pdus;
    std::vector<std::vector<tvCsirsPdu>> all_csirs_pdus;
    parse_tvs_and_setup_output_reference(all_pdsch_pdus, all_csirs_pdus, tvs, sp_cell_sub_cmds_ref, si);

    for (auto _ : state) {
            state.PauseTiming();

            cell_grp_cmd->reset();
            for (int k=0; k<num_cells; k++)
            {
                sp_cell_sub_cmds[k]->reset();
            }
            setup_fhcb_csirs_pdsch_input_command(cell_grp_cmd,sp_cell_sub_cmds,all_pdsch_pdus, all_csirs_pdus, si);
            thrash_cache();

            state.ResumeTiming();
            for (int k = 0; k < num_cells; ++k) {
                scf_5g_fapi::LegacyCellGroupFhContext fh_context(*cell_grp_cmd, k);
                scf_5g_fapi::fh_callback<false,true,false>(fh_context, nullptr);
            }
    }
}
BENCHMARK(BM_scfl2adapter_fhcb_basic_csirs_pdsch)->Apply(CustomArguments);


class ProcessConsecutiveBitsTest : public ::testing::Test {
protected:
    struct BitSequence {
        uint16_t start;
        uint16_t length;
    };
    
    std::vector<BitSequence> sequences;
    
    void SetUp() override {
        sequences.clear();
    }

    void captureSequence(uint16_t start, uint16_t length) {
        sequences.push_back({start, length});
    }
};

TEST_F(ProcessConsecutiveBitsTest, EmptyMask) {
    scf_5g_fapi::processConsecutiveBits(0, [this](uint16_t start, uint16_t length) {
        captureSequence(start, length);
    });
    EXPECT_TRUE(sequences.empty());
}

TEST_F(ProcessConsecutiveBitsTest, SingleBit) {
    scf_5g_fapi::processConsecutiveBits(0x1, [this](uint16_t start, uint16_t length) {
        captureSequence(start, length);
    });
    
    ASSERT_EQ(sequences.size(), 1);
    EXPECT_EQ(sequences[0].start, 0);
    EXPECT_EQ(sequences[0].length, 1);
}

TEST_F(ProcessConsecutiveBitsTest, ConsecutiveBits) {
    // 0b0011100 = 28
    scf_5g_fapi::processConsecutiveBits(28, [this](uint16_t start, uint16_t length) {
        captureSequence(start, length);
    });
    
    ASSERT_EQ(sequences.size(), 1);
    EXPECT_EQ(sequences[0].start, 2);
    EXPECT_EQ(sequences[0].length, 3);
}

TEST_F(ProcessConsecutiveBitsTest, MultipleSeparateSequences) {
    // 0b1100011000 = 792
    scf_5g_fapi::processConsecutiveBits(792, [this](uint16_t start, uint16_t length) {
        captureSequence(start, length);
    });
    
    ASSERT_EQ(sequences.size(), 2);
    EXPECT_EQ(sequences[0].start, 3);
    EXPECT_EQ(sequences[0].length, 2);
    EXPECT_EQ(sequences[1].start, 8);
    EXPECT_EQ(sequences[1].length, 2);
}

TEST_F(ProcessConsecutiveBitsTest, AllBitsSet) {
    // 0xFFFF = all 16 bits set
    scf_5g_fapi::processConsecutiveBits(0xFFFF, [this](uint16_t start, uint16_t length) {
        captureSequence(start, length);
    });
    
    ASSERT_EQ(sequences.size(), 1);
    EXPECT_EQ(sequences[0].start, 0);
    EXPECT_EQ(sequences[0].length, 16);
}

TEST_F(ProcessConsecutiveBitsTest, AlternatingBits) {
    // 0b0101010101 = 0x555
    scf_5g_fapi::processConsecutiveBits(0x555, [this](uint16_t start, uint16_t length) {
        captureSequence(start, length);
    });
    
    ASSERT_EQ(sequences.size(), 6);
    for (size_t i = 0; i < sequences.size(); i++) {
        EXPECT_EQ(sequences[i].start, i * 2);
        EXPECT_EQ(sequences[i].length, 1);
    }
}

TEST_F(ProcessConsecutiveBitsTest, WithExtraArguments) {
    int extra_arg = 42;
    std::string test_str = "test";
    
    scf_5g_fapi::processConsecutiveBits(0x7, 
        [this](uint16_t start, uint16_t length, int arg1, const std::string& arg2) {
            captureSequence(start, length);
            EXPECT_EQ(arg1, 42);
            EXPECT_EQ(arg2, "test");
        },
        extra_arg, test_str);
    
    ASSERT_EQ(sequences.size(), 1);
    EXPECT_EQ(sequences[0].start, 0);
    EXPECT_EQ(sequences[0].length, 3);
}

// Test real use case from the code
TEST_F(ProcessConsecutiveBitsTest, PDSCHUsagePattern) {
    uint16_t dmrsSymLocBmsk = 0x0808;  // Example DMRS symbol locations
    uint16_t numSym = 14;
    uint16_t pdschSym = 0;
    
    uint16_t pdschSymMask = ((1 << numSym) - 1) << pdschSym;
    uint16_t pdschOnlySymMask = pdschSymMask & ~dmrsSymLocBmsk;
    
    std::vector<std::pair<channel_type, BitSequence>> channel_sequences;
    
    // Process DMRS symbols
    scf_5g_fapi::processConsecutiveBits(dmrsSymLocBmsk, 
        [&channel_sequences](uint16_t start, uint16_t length) {
            channel_sequences.push_back({channel_type::PDSCH_DMRS, {start, length}});
        });
    
    // Process PDSCH-only symbols
    scf_5g_fapi::processConsecutiveBits(pdschOnlySymMask,
        [&channel_sequences](uint16_t start, uint16_t length) {
            channel_sequences.push_back({channel_type::PDSCH, {start, length}});
        });
    
    // Verify the pattern matches expected symbol allocation
    ASSERT_FALSE(channel_sequences.empty());
    for (const auto& seq : channel_sequences) {
        EXPECT_TRUE(seq.second.length > 0);
        EXPECT_LT(seq.second.start + seq.second.length, 16);
    }
}
class ProcessSetBitsTest : public ::testing::Test {
protected:
    struct BitPosition {
        uint16_t pos;
        uint16_t count;  // Will always be 1 for processSetBits
    };
    std::vector<BitPosition> positions;
    
    void SetUp() override {
        positions.clear();
    }

    void recordBit(uint16_t pos, uint16_t count) {
        positions.push_back({pos, count});
    }
};

TEST_F(ProcessSetBitsTest, EmptyMask) {
    scf_5g_fapi::processSetBits(0, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    EXPECT_TRUE(positions.empty()) << "Empty mask should not call lambda";
}

TEST_F(ProcessSetBitsTest, SingleBitAtEachPosition) {
    for (uint16_t i = 0; i < 16; ++i) {
        positions.clear();
        uint16_t mask = 1U << i;
        
        scf_5g_fapi::processSetBits(mask, [this](uint16_t pos, uint16_t count) {
            recordBit(pos, count);
        });
        
        ASSERT_EQ(positions.size(), 1) << "Failed for bit at position " << i;
        EXPECT_EQ(positions[0].pos, i) << "Wrong position for bit " << i;
        EXPECT_EQ(positions[0].count, 1) << "Count should always be 1";
    }
}

TEST_F(ProcessSetBitsTest, ConsecutiveBits) {
    // Test mask 0b0111 (7 in decimal)
    scf_5g_fapi::processSetBits(7, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    
    ASSERT_EQ(positions.size(), 3) << "Should find 3 set bits";
    for (size_t i = 0; i < positions.size(); ++i) {
        EXPECT_EQ(positions[i].pos, i) << "Wrong position for bit " << i;
        EXPECT_EQ(positions[i].count, 1) << "Count should always be 1";
    }
}

TEST_F(ProcessSetBitsTest, AlternatingBits) {
    // Test mask 0b0101010101010101 (0x5555)
    scf_5g_fapi::processSetBits(0x5555, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    
    ASSERT_EQ(positions.size(), 8) << "Should find 8 set bits";
    for (size_t i = 0; i < positions.size(); ++i) {
        EXPECT_EQ(positions[i].pos, i * 2) << "Wrong position for bit " << i;
        EXPECT_EQ(positions[i].count, 1) << "Count should always be 1";
    }
}

TEST_F(ProcessSetBitsTest, AllBitsSet) {
    scf_5g_fapi::processSetBits(0xFFFF, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    
    ASSERT_EQ(positions.size(), 16) << "Should find all 16 bits set";
    for (size_t i = 0; i < positions.size(); ++i) {
        EXPECT_EQ(positions[i].pos, i) << "Wrong position for bit " << i;
        EXPECT_EQ(positions[i].count, 1) << "Count should always be 1";
    }
}

TEST_F(ProcessSetBitsTest, DMRSPattern) {
    // Test typical DMRS pattern 0b0000100000001000 (0x0808)
    scf_5g_fapi::processSetBits(0x0808, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    
    ASSERT_EQ(positions.size(), 2) << "Should find 2 DMRS symbols";
    EXPECT_EQ(positions[0].pos, 3) << "First DMRS symbol position wrong";
    EXPECT_EQ(positions[1].pos, 11) << "Second DMRS symbol position wrong";
    for (const auto& pos : positions) {
        EXPECT_EQ(pos.count, 1) << "Count should always be 1";
    }
}

TEST_F(ProcessSetBitsTest, WithExtraArguments) {
    struct TestContext {
        int value;
        std::string str;
    };
    TestContext ctx{42, "test"};
    
    scf_5g_fapi::processSetBits(0x5, // 0b0101
        [this, &ctx](uint16_t pos, uint16_t count, const TestContext& context) {
            recordBit(pos, count);
            EXPECT_EQ(context.value, 42);
            EXPECT_EQ(context.str, "test");
        }, 
        ctx);
    
    ASSERT_EQ(positions.size(), 2) << "Should find 2 set bits";
    EXPECT_EQ(positions[0].pos, 0);
    EXPECT_EQ(positions[1].pos, 2);
    for (const auto& pos : positions) {
        EXPECT_EQ(pos.count, 1) << "Count should always be 1";
    }
}

TEST_F(ProcessSetBitsTest, PDSCHSymbolMask) {
    // Test PDSCH symbol mask with DMRS excluded
    uint16_t numSym = 14;
    uint16_t pdschSym = 0;
    uint16_t dmrsSymLocBmsk = 0x0808;  // Example DMRS locations
    
    uint16_t pdschSymMask = ((1 << numSym) - 1) << pdschSym;
    uint16_t pdschOnlySymMask = pdschSymMask & ~dmrsSymLocBmsk;
    
    scf_5g_fapi::processSetBits(pdschOnlySymMask, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    
    // Verify no DMRS positions are included
    for (const auto& pos : positions) {
        EXPECT_NE(pos.pos, 3) << "Should not include first DMRS position";
        EXPECT_NE(pos.pos, 11) << "Should not include second DMRS position";
        EXPECT_EQ(pos.count, 1) << "Count should always be 1";
    }
    
    // Verify correct number of PDSCH symbols
    EXPECT_EQ(positions.size(), 12) << "Should find 12 PDSCH symbols (14 total - 2 DMRS)";
}

TEST_F(ProcessSetBitsTest, EdgeBits) {
    // Test first and last bits
    uint16_t mask = 0x8001;  // Bits 0 and 15 set
    
    scf_5g_fapi::processSetBits(mask, [this](uint16_t pos, uint16_t count) {
        recordBit(pos, count);
    });
    
    ASSERT_EQ(positions.size(), 2) << "Should find 2 set bits";
    EXPECT_EQ(positions[0].pos, 0) << "First bit position wrong";
    EXPECT_EQ(positions[1].pos, 15) << "Last bit position wrong";
    for (const auto& pos : positions) {
        EXPECT_EQ(pos.count, 1) << "Count should always be 1";
    }
}

class PdschCsirsPrbOverlapTest : public ::testing::Test {
protected:
    static constexpr uint16_t kRemapMask = 0x0FFF;
};

TEST_F(PdschCsirsPrbOverlapTest, ReturnsFirstNonMatchingOffsetWithoutAltPrb) {
    std::array<uint16_t, 8> reMap{
        0x0123, 0x0123, 0x0123, 0x0124, 0x0124, 0x0124, 0x0124, 0x0124
    };
    const uint16_t start = 0;
    const uint16_t count = 8;
    const uint16_t csirs_remap = 0x0123 & kRemapMask;

    const uint16_t rb_idx = scf_5g_fapi::getPdschCsirsPrbOverlap(
        csirs_remap, false, start, count, reMap.data(), 0);

    EXPECT_EQ(rb_idx, 3);
}

TEST_F(PdschCsirsPrbOverlapTest, HonorsAltPrbMatchRule) {
    // With use_alt_prb enabled, val lower-12bits equal to (csirs_remap - 1) is treated as match.
    std::array<uint16_t, 6> reMap{
        0x0233, 0x0233, 0x0234, 0x0234, 0x0235, 0x0235
    };
    const uint16_t start = 0;
    const uint16_t count = 6;
    const uint16_t csirs_remap = 0x0234 & kRemapMask;

    const uint16_t rb_idx = scf_5g_fapi::getPdschCsirsPrbOverlap(
        csirs_remap, true, start, count, reMap.data(), 0);

    EXPECT_EQ(rb_idx, 4);
}

TEST_F(PdschCsirsPrbOverlapTest, ReturnsEndWhenAllEntriesMatch) {
    std::array<uint16_t, 5> reMap{
        0x0101, 0x0101, 0x0101, 0x0101, 0x0101
    };
    const uint16_t start = 0;
    const uint16_t count = 5;
    const uint16_t csirs_remap = 0x0101 & kRemapMask;

    const uint16_t rb_idx = scf_5g_fapi::getPdschCsirsPrbOverlap(
        csirs_remap, false, start, count, reMap.data(), 0);

    EXPECT_EQ(rb_idx, start + count);
}

} // anonymous namespace
