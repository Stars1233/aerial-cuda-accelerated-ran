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

#include "tv_parser.hpp"
#include "ru_emulator.hpp"
#include <set>

/**
 * @brief Try to read beam IDs from an HDF5 PDU dataset element (4T4R beam ID validation)
 *
 * Reads the digBFInterfaces and beamIdx fields from a PDU compound dataset.
 * If the fields are not present (e.g., older test vectors), the function
 * gracefully falls back to defaults and beam ID validation will be skipped.
 *
 * @param[in] pdu_pars HDF5 dataset element for the PDU
 * @param[out] info tv_info structure to populate with expected beam IDs
 */
static void try_read_beam_ids_from_pdu(hdf5hpp::hdf5_dataset_elem& pdu_pars, tv_info& info)
{
    try
    {
        info.digBFInterfaces = pdu_pars["digBFInterfaces"].as<uint16_t>();
        if (info.digBFInterfaces > 0)
        {
            info.expected_beam_ids = pdu_pars["beamIdx"].as<std::vector<uint16_t>>();
        }
    }
    catch (const std::exception&)
    {
        // beamIdx/digBFInterfaces not present in this TV - beam ID validation will be skipped
        info.digBFInterfaces = 0;
        info.expected_beam_ids.clear();
    }
}

void RU_Emulator::load_tvs()
{
    load_ul_tvs();
    load_dl_tvs();
    apply_tv_configs();
}

void RU_Emulator::apply_tv_configs()
{
    for(int cell_idx = 0; cell_idx < opt_num_cells; ++cell_idx)
    {
        for(auto& tv : prach_object.tv_info)
        {
            cell_configs[cell_idx].num_valid_PRACH_flows = std::min(cell_configs[cell_idx].num_valid_PRACH_flows, static_cast<int>(tv.numFlows));
        }

        // Assume PUSCH and PUCCH have the same number of flows
        for(auto& tv : pusch_object.tv_info)
        {
            cell_configs[cell_idx].num_ul_flows = std::min(cell_configs[cell_idx].num_ul_flows, static_cast<int>(tv.numFlows));
        }
        for(auto& tv : pucch_object.tv_info)
        {
            cell_configs[cell_idx].num_ul_flows = std::min(cell_configs[cell_idx].num_ul_flows, static_cast<int>(tv.numFlows));
        }
        for(auto& tv : srs_object.tv_info)
        {
            cell_configs[cell_idx].num_ul_flows = std::min(cell_configs[cell_idx].num_ul_flows, static_cast<int>(tv.numFlows));
        }
    }
}

void RU_Emulator::load_ul_tvs()
{
    auto t1 = get_ns();
    load_pusch_tvs();
    auto t2 = get_ns();
    re_cons("Loaded PUSCH TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);

    t1 = get_ns();
    load_prach_tvs();
    t2 = get_ns();
    re_cons("Loaded PRACH TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);

    t1 = get_ns();
    load_pucch_tvs();
    t2 = get_ns();
    re_cons("Loaded PUCCH TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);

    t1 = get_ns();
    load_srs_tvs();
    t2 = get_ns();
    re_cons("Loaded SRS TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
}

void load_ul_qams(hdf5hpp::hdf5_file& hdf5file, ul_tv_object& tv_object, ul_tv_info& tv_info, const std::vector<struct cell_config>& cell_configs, int oam)
{
    std::unordered_set<int> fmts;
    if(oam != RE_ENABLED)
    {
        for(const auto& cell_config: cell_configs)
        {
            if(cell_config.ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
            {
                fmts.insert(FIXED_POINT_16_BITS);
            }
            else
            {
                fmts.insert(cell_config.ul_bit_width);
            }
        }
    }
    else
    {
        for (int i = 0; i < IQ_DATA_FMT_MAX; ++i)
        {
            fmts.insert(i);
        }
    }

    for(auto i : fmts)
    {
        switch(i)
        {
            case BFP_COMPRESSION_9_BITS:
            {
                if(hdf5file.is_valid_dataset("X_tf_cSamples_bfp9"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_cSamples_bfp9"));
                    tv_object.slots[BFP_COMPRESSION_9_BITS].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_9F, tv_info.nPrbUlBwp, 0));
                    // tv_object.qams[BFP_COMPRESSION_9_BITS].emplace_back(std::move(d));
                }
                else
                {
                    re_cons("HDF File missing X_tf_cSamples_bfp9");
                }
                break;
            }
            case BFP_COMPRESSION_14_BITS:
            {
                if(hdf5file.is_valid_dataset("X_tf_cSamples_bfp14"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_cSamples_bfp14"));
                    tv_object.slots[BFP_COMPRESSION_14_BITS].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_14F, tv_info.nPrbUlBwp, 0));
                }
                else
                {
                    re_cons("HDF File missing X_tf_cSamples_bfp14");

                }
                break;
            }
            case BFP_NO_COMPRESSION:
            {
                if(hdf5file.is_valid_dataset("X_tf_fp16"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_fp16"));
                    tv_object.slots[BFP_NO_COMPRESSION].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_16F, tv_info.nPrbUlBwp, 0));
                }
                else
                {
                    re_cons("HDF File missing X_tf_fp16");
                }
                break;
            }
            case FIXED_POINT_16_BITS:
            {
                if(hdf5file.is_valid_dataset("X_tf_fx"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_fx"));
                    tv_object.slots[FIXED_POINT_16_BITS].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_16F, tv_info.nPrbUlBwp, 0));
                }
                else
                {
                    re_cons("HDF File missing X_tf_fx");
                }
                break;
            }
            default:
                continue;
        }
    }
}

void load_ul_qams_srs(hdf5hpp::hdf5_file& hdf5file, ul_tv_object& tv_object, ul_tv_info& tv_info, const std::vector<struct cell_config>& cell_configs, int oam)
{
    std::unordered_set<int> fmts;
    if(oam != RE_ENABLED)
    {
        for(const auto& cell_config: cell_configs)
        {
            if(cell_config.ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
            {
                fmts.insert(FIXED_POINT_16_BITS);
            }
            else
            {
                fmts.insert(cell_config.ul_bit_width);
            }
        }
    }
    else
    {
        for (int i = 0; i < IQ_DATA_FMT_MAX; ++i)
        {
            fmts.insert(i);
        }
    }

    for(auto i : fmts)
    {
        switch(i)
        {
            case BFP_COMPRESSION_9_BITS:
            {
                if(hdf5file.is_valid_dataset("X_tf_srs_cSamples_bfp9"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_srs_cSamples_bfp9"));
                    tv_object.slots[BFP_COMPRESSION_9_BITS].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_9F, tv_info.nPrbUlBwp, 0));
                    // tv_object.qams[BFP_COMPRESSION_9_BITS].emplace_back(std::move(d));
                }
                else
                {
                    re_cons("HDF File missing X_tf_cSamples_bfp9");
                }
                break;
            }
            case BFP_COMPRESSION_14_BITS:
            {
                if(hdf5file.is_valid_dataset("X_tf_srs_cSamples_bfp14"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_srs_cSamples_bfp14"));
                    tv_object.slots[BFP_COMPRESSION_14_BITS].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_14F, tv_info.nPrbUlBwp, 0));
                }
                else
                {
                    re_cons("HDF File missing X_tf_cSamples_bfp14");

                }
                break;
            }
            case BFP_NO_COMPRESSION:
            {
                if(hdf5file.is_valid_dataset("X_tf_srs_fp16"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_srs_fp16"));
                    tv_object.slots[BFP_NO_COMPRESSION].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_16F, tv_info.nPrbUlBwp, 0));
                }
                else
                {
                    re_cons("HDF File missing X_tf_srs_fp16");
                }
                break;
            }
            case FIXED_POINT_16_BITS:
            {
                if(hdf5file.is_valid_dataset("X_tf_srs_fx"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_srs_fx"));
                    tv_object.slots[FIXED_POINT_16_BITS].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_16F, tv_info.nPrbUlBwp, 0));
                }
                else
                {
                    re_cons("HDF File missing X_tf_srs_fx");
                }
                break;
            }
            default:
                continue;
        }
    }
}

void RU_Emulator::read_cell_cfg_from_tv(hdf5hpp::hdf5_file & hdf5file, struct tv_info & tv_info, std::string & tv_name)
{
    // Parse CellConfig dataset to retrieve CellConfig.ulGridSize value
    std::string dset_string = "Cell_Config";
    if(!hdf5file.is_valid_dataset(dset_string.c_str()))
    {
        do_throw(sb() << "ERROR No Cell_Config dataset found in TV " << tv_name);
    }
    hdf5hpp::hdf5_dataset dset_Cell_Config = hdf5file.open_dataset(dset_string.c_str());
    hdf5hpp::hdf5_dataset_elem cell_config_pars = dset_Cell_Config[0];

    tv_info.nPrbUlBwp = cell_config_pars["ulGridSize"].as<uint16_t>(); // FAPI parameter is uint32_t
    tv_info.nPrbDlBwp = cell_config_pars["dlGridSize"].as<uint16_t>(); // FAPI parameter is uint32_t
    tv_info.numGnbAnt = cell_config_pars["numTxAnt"].as<uint16_t>();
    //re_cons("Reading TV {} UL BWP {}, DL BWP {}, numGnbAnt {}", tv_name.c_str(), tv_info.nPrbUlBwp, tv_info.nPrbDlBwp, tv_info.numGnbAnt);
}

void RU_Emulator::get_ul_ports(hdf5hpp::hdf5_dataset_elem &pdu_pars, pdu_info &pdu_info)
{
    auto digBFInterfaces = pdu_pars["digBFInterfaces"].as<uint16_t>();
    if (digBFInterfaces == 0)
    {
        uint8_t dmrsPorts = pdu_pars["dmrsPorts"].as<uint8_t>();
        uint8_t scid = pdu_pars["SCID"].as<uint8_t>();

        for (int flow_index = 0; flow_index < sizeof(dmrsPorts) * 8; ++flow_index)
        {
            if ((dmrsPorts >> flow_index) & 0b1)
            {
                pdu_info.flow_indices.push_back(scid * 8 + flow_index);
            }
        }
    }
    else
    {
        for (int port = 0; port < digBFInterfaces; ++port)
        {
            pdu_info.flow_indices.push_back(port);
        }
    }
}

void RU_Emulator::load_pusch_tvs()
{
    // Load the requested test vectors
    // uplink_tv_flow_count = std::vector<uint16_t>(pusch_tvs.size());
    uint16_t tv_numPrb;
    for (int i = 0; i < pusch_object.tv_names.size(); ++i)
    {
        re_info("LOADING PUSCH TV {}", pusch_object.tv_names[i].c_str());
        struct ul_tv_info ul_tv_info;
        std::string dataset_name;

        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(pusch_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, pusch_object.tv_names[i]);
        uint8_t numFlows = load_num_antenna_from_nr_tv(hdf5file);

        int count = 1;
        std::string pdu = "PDU";
        bool pusch_found = false;
        ul_tv_info.tb_size = 0;
        ul_tv_info.numPrb = 0;
        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);

            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(pusch_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No PUSCH PDU found in TV " << pusch_object.tv_names[i]);
            }
            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PUSCH)
            {
                count++;
                continue;
            }
            pusch_found = true;

            // Read beam IDs for this PDU (first PDU populates expected_beam_ids
            // as fallback; all PDUs populate per_pdu_beam_ids)
            tv_info pdu_beam_info{};
            try_read_beam_ids_from_pdu(pdu_pars, pdu_beam_info);
            if (ul_tv_info.expected_beam_ids.empty())
            {
                ul_tv_info.digBFInterfaces = pdu_beam_info.digBFInterfaces;
                ul_tv_info.expected_beam_ids = pdu_beam_info.expected_beam_ids;
            }

            pdu_info pdu_info;
            pdu_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
            pdu_info.numSym = pdu_pars["NrOfSymbols"].as<uint8_t>();
            pdu_info.startPrb = pdu_pars["rbStart"].as<uint16_t>() + pdu_pars["BWPStart"].as<uint16_t>();
            pdu_info.numPrb = pdu_pars["rbSize"].as<uint16_t>();
            pdu_info.numFlows = numFlows;
            pdu_info.tb_size = pdu_pars["TBSize"].as<uint32_t>();
            pdu_info.dmrsPorts = pdu_pars["dmrsPorts"].as<uint8_t>();
            pdu_info.scid = pdu_pars["SCID"].as<uint8_t>();
            ul_tv_info.tb_size += pdu_info.tb_size;

            if (opt_enable_mmimo)
            {
                get_ul_ports(pdu_pars, pdu_info);
                ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym * pdu_info.flow_indices.size();
            }
            auto found = [&pdu_info] (const struct pdu_info& pdu) {
                return (pdu.startSym == pdu_info.startSym && pdu.numSym == pdu_info.numSym && pdu.startPrb == pdu_info.startPrb && pdu.numPrb ==  pdu_info.numPrb);
            };

            auto iter = std::find_if(ul_tv_info.pdu_infos.begin(), ul_tv_info.pdu_infos.end(), found);
            if(iter == ul_tv_info.pdu_infos.end())
            {
                ++ul_tv_info.numSections;
                /* Currently the numPrb calculation is based on the assuption that the
                 * symbol range and prb range are the same for each PDU. Need to revisit
                 * this implementation once we have different test case.
                 */

                /* Taking out flow multiplier because a 4 antenna TV could be configured for 2
                 * antennas on the FH if there is no traffic on other antennas. Adding the
                 * cell_configs num_ul_flow multiplier at the completion check
                 */

                if (!opt_enable_mmimo)
                {
                    ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym; // * pdu_info.numFlows;}
                }
            }

            if (!pdu_beam_info.expected_beam_ids.empty())
            {
                ul_tv_info.per_pdu_beam_ids.push_back({
                    pdu_info.startPrb,
                    pdu_info.numPrb,
                    pdu_beam_info.expected_beam_ids});
            }
            ul_tv_info.pdu_infos.emplace_back(std::move(pdu_info));
            count++;
        }
        ul_tv_info.numFlows = numFlows;
        load_ul_qams(hdf5file, pusch_object, ul_tv_info, cell_configs, opt_oam_cell_ctrl_cmd);
        pusch_object.tv_info.emplace_back(ul_tv_info);
        hdf5file.close();

    }

}

void RU_Emulator::load_prach_tvs()
{
    for (int i = 0; i < prach_object.tv_names.size(); ++i)
    {
        re_info("LOADING PRACH TV {}", prach_object.tv_names[i].c_str());
        struct ul_tv_info ul_tv_info;
        std::string dataset_name;
        std::string base_x_tf_dataset_name;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(prach_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, prach_object.tv_names[i]);

        int count = 1;
        std::string pdu = "PDU";
        std::string ro_config = "RO_Config_";
        bool prach_found = false;
        uint8_t numFlows = opt_enable_mmimo? 0 : load_num_antenna_from_nr_tv(hdf5file);

        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            std::string ro_dset_string = ro_config + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(!prach_found)
                {
                    do_throw(sb() << "ERROR No PRACH PDU found in TV " << prach_object.tv_names[i]);
                }
                break;
            }

            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PRACH)
            {
                count++;
                continue;
            }

            if(!prach_found)
            {
                prach_object.prach_slots[FIXED_POINT_16_BITS].emplace_back(std::vector<Slot>());
                prach_object.prach_slots[BFP_NO_COMPRESSION].emplace_back(std::vector<Slot>());
                prach_object.prach_slots[BFP_COMPRESSION_14_BITS].emplace_back(std::vector<Slot>());
                prach_object.prach_slots[BFP_COMPRESSION_9_BITS].emplace_back(std::vector<Slot>());
            }
            prach_found = true;

            pdu_info pdu_info;
            if(hdf5file.is_valid_dataset(ro_dset_string.c_str()))
            {
                hdf5hpp::hdf5_dataset dset_ro_config  = hdf5file.open_dataset(ro_dset_string.c_str());
                hdf5hpp::hdf5_dataset_elem ro_pars = dset_ro_config[0];
                pdu_info.startPrb = ro_pars["k1"].as<uint16_t>();
            }

            base_x_tf_dataset_name = "X_tf_prach_" + std::to_string(count);
            dataset_name = base_x_tf_dataset_name;
            pdu_info.numPrb = 12;
            pdu_info.numSym = 12;
            pdu_info.startSym = pdu_pars["prachStartSymbol"].as<uint8_t>();
            pdu_info.numFlows = load_num_antenna_from_nr_prach_tv(hdf5file, base_x_tf_dataset_name);
            if (opt_enable_mmimo)
            {
                get_ul_ports(pdu_pars, pdu_info);
                numFlows = pdu_pars["digBFInterfaces"].as<uint16_t>();
            }

            for(int i = 0; i < IQ_DATA_FMT_MAX; ++i)
            {
                base_x_tf_dataset_name = "X_tf_prach_" + std::to_string(count);
                dataset_name = base_x_tf_dataset_name;
                switch(i)
                {
                    case BFP_COMPRESSION_9_BITS:
                    {
                        dataset_name += "_cSamples_bfp9";
                        if(hdf5file.is_valid_dataset(dataset_name.c_str()))
                        {
                            Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", dataset_name));
                            prach_object.prach_slots[BFP_COMPRESSION_9_BITS].back().emplace_back(dataset_to_slot(std::move(d), pdu_info.numFlows, pdu_info.startSym, pdu_info.numSym, PRB_SIZE_9F, pdu_info.numPrb, pdu_info.startPrb));
                        }
                        else
                        {
                            re_cons("HDF File missing {}", dataset_name.c_str());
                        }
                        break;
                    }
                    case BFP_COMPRESSION_14_BITS:
                    {
                        dataset_name += "_cSamples_bfp14";
                        if(hdf5file.is_valid_dataset(dataset_name.c_str()))
                        {
                            Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", dataset_name));
                            prach_object.prach_slots[BFP_COMPRESSION_14_BITS].back().emplace_back(dataset_to_slot(std::move(d), pdu_info.numFlows, pdu_info.startSym, pdu_info.numSym, PRB_SIZE_14F, pdu_info.numPrb, pdu_info.startPrb));
                        }
                        else
                        {
                            re_cons("HDF File missing {}", dataset_name.c_str());
                        }
                        break;
                    }
                    case BFP_NO_COMPRESSION:
                    {
                        dataset_name += "_fp16";
                        if(hdf5file.is_valid_dataset(dataset_name.c_str()))
                        {
                            Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", dataset_name));
                            prach_object.prach_slots[BFP_NO_COMPRESSION].back().emplace_back(dataset_to_slot(std::move(d), pdu_info.numFlows, pdu_info.startSym, pdu_info.numSym, PRB_SIZE_16F, pdu_info.numPrb, pdu_info.startPrb));
                        }
                        else
                        {
                            re_cons("HDF File missing {}", dataset_name.c_str());
                        }
                        break;
                    }
                    case FIXED_POINT_16_BITS:
                    {
                        dataset_name += "_fx";
                        if(hdf5file.is_valid_dataset(dataset_name.c_str()))
                        {
                            Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", dataset_name));
                            prach_object.prach_slots[FIXED_POINT_16_BITS].back().emplace_back(dataset_to_slot(std::move(d), pdu_info.numFlows, pdu_info.startSym, pdu_info.numSym, PRB_SIZE_16F, pdu_info.numPrb, pdu_info.startPrb));
                        }
                        else
                        {
                            re_cons("HDF File missing {}", dataset_name.c_str());
                        }
                        break;
                    }
                    default:
                        continue;
                }
            }
            count++;

            ul_tv_info.pdu_infos.emplace_back(std::move(pdu_info));
        }

        ul_tv_info.numFlows = numFlows;
        ul_tv_info.startPrb = 0;
        ul_tv_info.numPrb = 12;
        ul_tv_info.startSym = 0;
        ul_tv_info.numSym = 12;
        prach_object.tv_info.emplace_back(ul_tv_info);
        hdf5file.close();
    }
}

void RU_Emulator::load_pucch_tvs()
{
    for (int i = 0; i < pucch_object.tv_names.size(); ++i)
    {
        re_info("LOADING PUCCH TV {}", pucch_object.tv_names[i].c_str());
        struct ul_tv_info ul_tv_info;
        std::string dataset_name;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(pucch_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, pucch_object.tv_names[i]);
        uint8_t  numFlows = load_num_antenna_from_nr_tv(hdf5file);

        int count = 1;
        std::string pdu = "PDU";
        bool pucch_found = false;
        ul_tv_info.tb_size = 0;
        ul_tv_info.endPrb = 0;
        ul_tv_info.startPrb = MAX_NUM_PRBS_PER_SYMBOL;
        ul_tv_info.numPrb = 0;
        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(pucch_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No PUCCH PDU found in TV " << pucch_object.tv_names[i]);
            }
            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PUCCH)
            {
                count++;
                continue;
            }
            pucch_found = true;

            // Read beam IDs for this PDU (first PDU populates expected_beam_ids
            // as fallback; all PDUs populate per_pdu_beam_ids)
            tv_info pdu_beam_info{};
            try_read_beam_ids_from_pdu(pdu_pars, pdu_beam_info);
            if (ul_tv_info.expected_beam_ids.empty())
            {
                ul_tv_info.digBFInterfaces = pdu_beam_info.digBFInterfaces;
                ul_tv_info.expected_beam_ids = pdu_beam_info.expected_beam_ids;
            }

            // Multiple PDUs, Assume all have the same params, and accumulate TB size
            pdu_info pdu_info;
            pdu_info.freqHopFlag = pdu_pars["freqHopFlag"].as<uint32_t>();
            pdu_info.secondHopPrb = pdu_pars["secondHopPRB"].as<uint32_t>() + pdu_pars["BWPStart"].as<uint16_t>();
            pdu_info.numPrb = pdu_pars["prbSize"].as<uint16_t>();
            pdu_info.startPrb = pdu_pars["prbStart"].as<uint16_t>() + pdu_pars["BWPStart"].as<uint16_t>();
            pdu_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
            pdu_info.numSym = pdu_pars["NrOfSymbols"].as<uint8_t>();
            pdu_info.numFlows = numFlows;

            if (opt_enable_mmimo)
            {
                get_ul_ports(pdu_pars, pdu_info);
            }

            if (!pdu_beam_info.expected_beam_ids.empty())
            {
                ul_tv_info.per_pdu_beam_ids.push_back({
                    pdu_info.startPrb,
                    pdu_info.numPrb,
                    pdu_beam_info.expected_beam_ids});
            }

            ul_tv_info.endPrb = std::max(static_cast<int>(pdu_info.startPrb) + static_cast<int>(pdu_info.numPrb), static_cast<int>(ul_tv_info.endPrb));
            ul_tv_info.startPrb = std::min(static_cast<int>(pdu_info.startPrb), static_cast<int>(ul_tv_info.startPrb));

            auto found = [&pdu_info] (const struct pdu_info& pdu) {
                return (pdu.startSym == pdu_info.startSym && pdu.numSym == pdu_info.numSym && pdu.startPrb == pdu_info.startPrb && pdu.numPrb ==  pdu_info.numPrb);
            };

            auto iter = std::find_if(ul_tv_info.pdu_infos.begin(), ul_tv_info.pdu_infos.end(), found);
            if(iter == ul_tv_info.pdu_infos.end())
            {
                ++ul_tv_info.numSections;
                /* Currently the numPrb calculation is based on the assuption that the
                 * symbol range and prb range are the same for each PDU. Need to revisit
                 * this implementation once we have different test case.
                 */

                /* Taking out flow multiplier because a 4 antenna TV could be configured for 2
                 * antennas on the FH if there is no traffic on other antennas. Adding the
                 * cell_configs num_ul_flow multiplier at the completion check
                 */

                if (opt_enable_mmimo)
                {
                    ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym * pdu_info.flow_indices.size();
                }
                else
                {
                    ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym; // * pdu_info.numFlows;}
                }
                ul_tv_info.pdu_infos.push_back(std::move(pdu_info));
            }

            // ul_tv_info.numPrb = pdu_pars["prbSize"].as<uint16_t>();
            // ul_tv_info.startPrb = pdu_pars["prbStart"].as<uint16_t>() + pdu_pars["BWPStart"].as<uint16_t>();
            ul_tv_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
            ul_tv_info.numSym = pdu_pars["NrOfSymbols"].as<uint8_t>();
            count++;
        }

        generate_prb_map(ul_tv_info, ul_tv_info.pdu_infos);
        ul_tv_info.numFlows = numFlows;
        load_ul_qams(hdf5file, pucch_object, ul_tv_info, cell_configs, opt_oam_cell_ctrl_cmd);
        pucch_object.tv_info.emplace_back(ul_tv_info);
        hdf5file.close();
    }
}

int8_t calc_start_prb(scf_fapi_srs_pdu_t& msg, srs_rb_info_t srs_rb_info[], uint16_t frame, uint16_t slot)
{
    uint16_t hopStartPrbs[MAX_SRS_SYM] = {0};
    uint16_t numPrbs[MAX_SRS_SYM] = {0};
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
    uint16_t frameNum = frame;//cell_cmd.slot.slot_3gpp.sfn_;
    uint16_t slotNum = slot;//cell_cmd.slot.slot_3gpp.slot_;
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
                            //NVLOGC_FMT(TAG,"Not an SRS slot ...");
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
            hopStartPrbs[hopIdx] = hopStartPrbs[hopIdx] +  m_SRS_b * nb;
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
    return nSyms;
}

void RU_Emulator::load_srs_tvs()
{
    for (int i = 0; i < srs_object.tv_names.size(); ++i)
    {
        re_info("LOADING SRS TV {}", srs_object.tv_names[i].c_str());
        struct ul_tv_info ul_tv_info;
        std::string dataset_name;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(srs_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, srs_object.tv_names[i]);
        uint8_t numFlows = load_num_antenna_from_nr_tv(hdf5file);

        int count = 1;
        std::string pdu = "PDU";
        bool srs_found = false;
        ul_tv_info.tb_size = 0;
        ul_tv_info.endPrb = 0;
        ul_tv_info.startPrb = MAX_NUM_PRBS_PER_SYMBOL;
        ul_tv_info.numPrb = 0;
        std::unordered_map<int, std::unordered_map<int, std::vector<pdu_info>>> fss_pdu_infos;
        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(srs_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No SRS PDU found in TV " << srs_object.tv_names[i]);
            }
            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::SRS)
            {
                count++;
                continue;
            }
            srs_found = true;

            srs_rb_info_t srs_rb_info[MAX_SRS_SYM]={{0}};
            scf_fapi_srs_pdu_t srs_pdu;

            srs_pdu.num_symbols = pdu_pars["numSymbols"].as<uint8_t>();
            srs_pdu.time_start_position = pdu_pars["timeStartPosition"].as<uint8_t>();
            srs_pdu.num_repetitions = pdu_pars["numRepetitions"].as<uint8_t>();
            srs_pdu.frequency_shift = pdu_pars["frequencyShift"].as<uint8_t>();
            srs_pdu.bandwidth_index = pdu_pars["bandwidthIndex"].as<uint8_t>();
            srs_pdu.config_index = pdu_pars["configIndex"].as<uint8_t>();
            srs_pdu.frequency_position = pdu_pars["frequencyPosition"].as<uint8_t>();
            srs_pdu.frequency_hopping = pdu_pars["frequencyHopping"].as<uint8_t>();
            srs_pdu.resource_type = pdu_pars["resourceType"].as<uint8_t>();
            srs_pdu.t_srs = pdu_pars["Tsrs"].as<uint16_t>();
            srs_pdu.t_offset = pdu_pars["Toffset"].as<uint16_t>();

            for(int frame = 0; frame < ORAN_MAX_FRAME_ID; frame++)
            {
                for(int slot = 0; slot < SLOT_3GPP; slot++)
                {
                    int nSyms = calc_start_prb(srs_pdu, srs_rb_info, frame, slot);
                    if(!nSyms) continue;
                    for(int k = 0; k < nSyms; k++) {
                        pdu_info pdu_info;
                        pdu_info.numPrb = srs_rb_info[k].num_srs_prbs;
                        pdu_info.startPrb = srs_rb_info[k].srs_start_prbs;
                        pdu_info.startSym = srs_pdu.time_start_position + k;
                        pdu_info.numSym = 1;
                        pdu_info.numFlows = numFlows;
                        ul_tv_info.endPrb = std::max(static_cast<int>(pdu_info.startPrb) + static_cast<int>(pdu_info.numPrb), static_cast<int>(ul_tv_info.endPrb));
                        ul_tv_info.startPrb = std::min(static_cast<int>(pdu_info.startPrb), static_cast<int>(ul_tv_info.startPrb));

                        fss_pdu_infos[frame][slot].push_back(std::move(pdu_info));
#if 0
                        auto found = [&pdu_info] (const struct pdu_info& pdu) {
                            return (pdu.startSym == pdu_info.startSym && pdu.numSym == pdu_info.numSym && pdu.startPrb == pdu_info.startPrb && pdu.numPrb ==  pdu_info.numPrb);
                        };

                        auto iter = std::find_if(ul_tv_info.fss_pdu_infos[frame][slot].begin(), ul_tv_info.fss_pdu_infos[frame][slot].end(), found);
                        if(iter == ul_tv_info.fss_pdu_infos[frame][slot].end())
                        {
                            ++ul_tv_info.numSections;
                            /* Currently the numPrb calculation is based on the assuption that the
                             * symbol range and prb range are the same for each PDU. Need to revisit
                             * this implementation once we have different test case.
                             */

                            /* Taking out flow multiplier because a 4 antenna TV could be configured for 2
                             * antennas on the FH if there is no traffic on other antennas. Adding the
                             * cell_configs num_ul_flow multiplier at the completion check
                             */

                            ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym;// * pdu_info.numFlows;
                            ul_tv_info.fss_pdu_infos[frame][slot].push_back(std::move(pdu_info));
                        }
#endif
                    }
                    ul_tv_info.startSym = srs_pdu.time_start_position;
                    ul_tv_info.numSym = nSyms;
                }
            }
            count++;
        }

        for (int frame = 0; frame < ORAN_MAX_FRAME_ID; frame++)
        {
            for (int slot = 0; slot < SLOT_3GPP; slot++)
            {
                if (fss_pdu_infos.find(frame) != fss_pdu_infos.end() && fss_pdu_infos[frame].find(slot) != fss_pdu_infos[frame].end() && fss_pdu_infos[frame][slot].size() > 0)
                {
                    sort(begin(fss_pdu_infos[frame][slot]), end(fss_pdu_infos[frame][slot]), [](auto &a, auto &b)
                         { return a.startSym == b.startSym ? a.startPrb < b.startPrb : a.startSym < b.startSym; });

                    auto &pdu_infos = ul_tv_info.fss_pdu_infos[frame][slot];
                    for (auto &p : fss_pdu_infos[frame][slot])
                    {
                        if (pdu_infos.size() == 0 || pdu_infos.back().startSym != p.startSym || pdu_infos.back().startPrb + pdu_infos.back().numPrb < p.startPrb)
                        {
                            ul_tv_info.fss_numPrb[frame][slot] += p.numPrb;
                            pdu_infos.push_back(std::move(p));
                        }
                        else
                        {
                            auto &pdu_info = pdu_infos.back();
                            ul_tv_info.fss_numPrb[frame][slot] -= pdu_info.numPrb;
                            auto end = std::max(p.startPrb + p.numPrb, pdu_info.startPrb + pdu_info.numPrb);
                            pdu_info.startPrb = std::min(p.startPrb, pdu_info.startPrb);
                            pdu_info.numPrb = end - pdu_info.startPrb;
                            ul_tv_info.fss_numPrb[frame][slot] += pdu_info.numPrb;
                        }

                        auto &pdu_info = pdu_infos.back();
                        for (int sym = pdu_info.startSym; sym < pdu_info.startSym + pdu_info.numSym; sym++)
                        {
                            for (int prb = pdu_info.startPrb; prb < pdu_info.startPrb + pdu_info.numPrb; prb++)
                            {
                                ul_tv_info.fss_prb_map[frame][slot][sym][prb] = true;
                            }
                        }
                    }
                }
            }
        }

        ul_tv_info.numFlows = load_num_antenna_from_nr_tv_srs(hdf5file);
        load_ul_qams_srs(hdf5file, srs_object, ul_tv_info, cell_configs, opt_oam_cell_ctrl_cmd);
        srs_object.tv_info.emplace_back(ul_tv_info);
        hdf5file.close();
    }
}

void RU_Emulator::load_dl_tvs()
{
    if(opt_pdsch_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_pdsch_tvs();
        auto t2 = get_ns();
        re_cons("Loaded PDSCH TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }
    if(opt_pbch_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_pbch_tvs();
        auto t2 = get_ns();
        re_cons("Loaded PBCH TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }
    if(opt_pdcch_ul_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_pdcch_ul_tvs();
        auto t2 = get_ns();
        re_cons("Loaded PDCCH_UL TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }
    if(opt_pdcch_dl_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_pdcch_dl_tvs();
        auto t2 = get_ns();
        re_cons("Loaded PDCCH_DL TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }
    if(opt_csirs_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_csirs_tvs();
        auto t2 = get_ns();
        re_cons("Loaded CSIRS TVs in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }

    if(opt_bfw_dl_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_bfw_tvs(true);
        auto t2 = get_ns();
        re_cons("Loaded BFW DL TVs\n in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }

    if(opt_bfw_ul_validation == RE_ENABLED)
    {
        auto t1 = get_ns();
        load_bfw_tvs(false);
        auto t2 = get_ns();
        re_cons("Loaded BFW UL TVs\n in {:.2f}s", ((double)(t2 - t1))/NS_X_S);
    }

}

void load_dl_qams(hdf5hpp::hdf5_file& hdf5file, dl_tv_object& tv_object, dl_tv_info& dl_tv_info, bool mod_comp_enabled, bool non_mod_comp_enabled, const std::vector<struct cell_config>& cell_configs, bool selective_load)
{
    if (non_mod_comp_enabled)
    {
        std::unordered_set<int> fmts;
        if (selective_load)
        {
            for (const auto &cell_config : cell_configs)
            {
                if (cell_config.dl_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
                {
                    fmts.insert(FIXED_POINT_16_BITS);
                }
                else
                {
                    fmts.insert(cell_config.dl_bit_width);
                }
            }
        }
        else
        {
            for (int i = 0; i < IQ_DATA_FMT_MAX; ++i)
            {
                fmts.insert(i);
            }
        }

        for (auto i : fmts)
        {
            switch (i)
            {
            case BFP_COMPRESSION_9_BITS:
            {
                if (hdf5file.is_valid_dataset("X_tf_cSamples_bfp9"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_cSamples_bfp9"));
                    tv_object.qams[BFP_COMPRESSION_9_BITS].emplace_back(std::move(d));
                }
                else
                {
                    re_cons("HDF File missing X_tf_cSamples_bfp9");
                }
                break;
            }
            case BFP_COMPRESSION_14_BITS:
            {
                if (hdf5file.is_valid_dataset("X_tf_cSamples_bfp14"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_cSamples_bfp14"));
                    tv_object.qams[BFP_COMPRESSION_14_BITS].emplace_back(std::move(d));
                }
                else
                {
                    re_cons("HDF File missing X_tf_cSamples_bfp14");
                }
                break;
            }
            case BFP_NO_COMPRESSION:
            {
                if (hdf5file.is_valid_dataset("X_tf_fp16"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_fp16"));
                    tv_object.qams[BFP_NO_COMPRESSION].emplace_back(std::move(d));
                }
                else
                {
                    re_cons("HDF File missing X_tf_fp16");
                }
                break;
            }
            case FIXED_POINT_16_BITS:
            {
                if (hdf5file.is_valid_dataset("X_tf_fx"))
                {
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_fx"));
                    tv_object.qams[FIXED_POINT_16_BITS].emplace_back(std::move(d));
                }
                else
                {
                    re_cons("HDF File missing X_tf_fx");
                }
                break;
            }
            default:
                continue;
            }
        }
    }

    if (mod_comp_enabled && hdf5file.is_valid_dataset("nMsg_new"))
    {
        uint32_t nMsg = 0;
        hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset("nMsg_new");
        dset.read(&nMsg);

        std::string msg = "MSG";
        std::string header = "_header_new";
        std::string payload = "_payload_new";

        // tv_mod_comp_object mod_comp_obj;
        tv_object.mod_comp_data.emplace_back();
        tv_mod_comp_object &mod_comp_obj = tv_object.mod_comp_data.back();
        int num_msgs = 0;

        for (int i = 1; i <= nMsg; i++)
        {
            std::string cur_header = msg + std::to_string(i) + header;
            if (!hdf5file.is_valid_dataset(cur_header.c_str()))
            {
                do_throw(sb() << "ERROR No " << cur_header << "found in TV ");
            }
            hdf5hpp::hdf5_dataset dset_msg_hdr = hdf5file.open_dataset(cur_header.c_str());
            hdf5hpp::hdf5_dataset_elem hdr = dset_msg_hdr[0];

            uint32_t sym = hdr["startSymbolid"].as<uint32_t>();
            int startPrbc = hdr["startPrbc"].as<uint32_t>();
            int numPrbc = hdr["numPrbc"].as<uint32_t>();
            int reMask = hdr["reMask"].as<uint32_t>();
            int portIdx = hdr["portIdx"].as<uint32_t>();
            int udIqWidth = hdr["udIqWidth"].as<uint32_t>();
            std::vector<uint32_t> channel_type = hdr["chanType"].as<std::vector<uint32_t>>();
            std::vector<uint16_t> mc_scale_remask;
            // check if the msg belongs to current tv
            bool is_pdsch_csirs = false;
            if (channel_type.size() == 2 && ((nrsim_tv_type::PDSCH == channel_type[0] && nrsim_tv_type::CSI_RS == channel_type[1]) || (nrsim_tv_type::PDSCH == channel_type[1] && nrsim_tv_type::CSI_RS == channel_type[0])))
            {
                is_pdsch_csirs = true;
                mc_scale_remask = hdr["mcScaleReMask"].as<std::vector<uint16_t>>();
            }

            tv_mod_comp_ext_info ext_info;
            hdf5hpp::hdf5_datatype hdr_dtype = dset_msg_hdr.get_datatype();
            bool has_ext_fields = H5Tget_member_index(hdr_dtype.id(), "extType") >= 0 &&
                                  H5Tget_member_index(hdr_dtype.id(), "nMask") >= 0 &&
                                  H5Tget_member_index(hdr_dtype.id(), "mcScaleOffset") >= 0 &&
                                  H5Tget_member_index(hdr_dtype.id(), "mcScaleReMask") >= 0 &&
                                  H5Tget_member_index(hdr_dtype.id(), "csf") >= 0;
            if (has_ext_fields)
            {
                ext_info.ext_type = hdr["extType"].as<uint32_t>();
                ext_info.n_mask = hdr["nMask"].as<uint32_t>();
                std::vector<uint32_t> tv_re_masks = hdr["mcScaleReMask"].as<std::vector<uint32_t>>();
                std::vector<double> tv_offsets = hdr["mcScaleOffset"].as<std::vector<double>>();
                std::vector<uint32_t> tv_csf = hdr["csf"].as<std::vector<uint32_t>>();
                uint32_t max_k = std::min({ext_info.n_mask, uint32_t(2),
                                           uint32_t(tv_re_masks.size()),
                                           uint32_t(tv_offsets.size()),
                                           uint32_t(tv_csf.size())});
                if (max_k < ext_info.n_mask)
                {
                    re_warn("ModComp ext vectors shorter than nMask={}: reMask={} offset={} csf={}", 
                            ext_info.n_mask, tv_re_masks.size(), tv_offsets.size(), tv_csf.size());
                }
                for (uint32_t k = 0; k < max_k; k++)
                {
                    ext_info.mc_scale_re_mask[k] = static_cast<uint16_t>(tv_re_masks[k]);
                    ext_info.mc_scale_offset_encoded[k] = float_to_modcompscaler(static_cast<float>(tv_offsets[k]));
                    ext_info.csf[k] = tv_csf[k];
                }
                ext_info.valid = true;
            }

            for (int j = 0; j < channel_type.size(); j++)
            {
                if ((tv_object.nrsim_ch_type != nrsim_tv_type::PDCCH && (int)(tv_object.nrsim_ch_type) == channel_type[j]) || (tv_object.nrsim_ch_type == nrsim_tv_type::PDCCH && dl_tv_info.prb_map[sym][startPrbc] && ((1 << portIdx) & dl_tv_info.prb_num_flow_map[sym][startPrbc])))
                {
                    int remask = is_pdsch_csirs ? mc_scale_remask[j] : reMask;
                    mod_comp_obj.mod_comp_header[sym][portIdx][remask].push_back({startPrbc, numPrbc, i - 1});
                    re_dbg("ModComp MSG {}: symbol {} startPrbc {} numPrbc {} reMask 0x{:x} portIdx {} udIqWidth {} modCompNumPrb {}",
                        i, sym, startPrbc, numPrbc, remask, portIdx, udIqWidth, dl_tv_info.modCompNumPrb);
                    bool skip_iq_validation = is_pdsch_csirs && (tv_object.nrsim_ch_type == nrsim_tv_type::CSI_RS);
                    if(dl_tv_info.modCompNumPrb + numPrbc > UINT32_MAX)
                    {
                        do_throw(sb() << "ERROR ModComp Num Prb exceeds UINT32_MAX");
                    }
                    dl_tv_info.modCompNumPrb += numPrbc;
                    std::string cur_payload = msg + std::to_string(i) + payload;
                    Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", cur_payload));
                    mod_comp_obj.mod_comp_payload.emplace_back(std::move(d));
                    mod_comp_obj.mod_comp_payload_params.push_back({startPrbc, numPrbc, udIqWidth, skip_iq_validation});
                    mod_comp_obj.mod_comp_ext_info.push_back(ext_info);
                    mod_comp_obj.global_msg_idx_to_tv_idx[i - 1] = num_msgs;
                    num_msgs++;
                    // mod_comp_obj.fss_mod_comp_payload_idx[portIdx][section_id] = i;
                    // buffer_sz = udIqwith*num_prbc*3;//(udIqWith*12*2*num_prbc)/8;
                }
                if (!is_pdsch_csirs)
                { // Assume only pdsch_csirs has two different channel types in the same MSG
                    break;
                }
            }
        }
    }
}

void load_bfw_qams(hdf5hpp::hdf5_file& hdf5file, dl_tv_object& tv_object)
{
    int ueGrpNum = 0;
    std::string uncompressed = "bfwUeGrp";
    std::string compressed = "bfwCompUeGrp";
    while(!check_force_quit())
    {
        std::string uncompressed_dset = uncompressed + std::to_string(ueGrpNum);
        std::string compressed_dset = compressed + std::to_string(ueGrpNum);
        if(!hdf5file.is_valid_dataset(uncompressed_dset.c_str()))
        {
            break;
        }

        for(int i = 0; i < BFP_NO_COMPRESSION+1; ++i)
        {
            switch(i)
            {
                case BFP_COMPRESSION_9_BITS:
                {
                    if(hdf5file.is_valid_dataset(compressed_dset.c_str()))
                    {
                        Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", compressed_dset.c_str()));
                        tv_object.qams[BFP_COMPRESSION_9_BITS].emplace_back(std::move(d));
                        re_info("Loaded {}", compressed_dset.c_str());
                    }
                    else
                    {
                        re_cons("HDF File missing {}", compressed_dset.c_str());
                    }
                    break;
                }
                case BFP_NO_COMPRESSION:
                {
                    if(hdf5file.is_valid_dataset(uncompressed_dset.c_str()))
                    {
                        Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", uncompressed_dset.c_str()));
                        tv_object.qams[BFP_NO_COMPRESSION].emplace_back(std::move(d));
                        re_info("Loaded {}", uncompressed_dset.c_str());
                    }
                    else
                    {
                        re_cons("HDF File missing {}", uncompressed_dset.c_str());
                    }
                    break;
                }
                default:
                    continue;
            }
        }
        ++ueGrpNum;
    }
}

void RU_Emulator::load_pdsch_tvs()
{
    uint32_t tv_nRx;
    uint32_t tv_numTb;
    uint32_t tv_numPrb;
    uint32_t tv_startSym;
    uint32_t tv_numSym;
    uint32_t tv_dmrsMaxLength;
    uint32_t tv_startPrb;
    uint32_t buffer_size;
    uint32_t buffer_index;
    uint32_t flow_dim;
    uint32_t sym_dim;
    uint32_t freq_dim;
    uint32_t out;
    uint32_t tb_size;

    struct tv_parsing_timers timers;
    timers.compute = 0;
    timers.load = 0;
    for(int i = 0; i < pdsch_object.tv_names.size(); ++i)
    {
        auto compute_start = get_ns();

        struct dl_tv_info dl_tv_info;
        dl_tv_info.numPrb = 0;
        dl_tv_info.tb_size = 0;
        std::string dset_name;
        re_dbg("Reading TV {}", pdsch_object.tv_names[i].c_str());
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(pdsch_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, dl_tv_info, pdsch_object.tv_names[i]);
        uint8_t numFlows = load_num_antenna_from_nr_tv(hdf5file);

        int count = 1;
        std::string pdu = "PDU";
        bool pdsch_found = false;
        dl_tv_info.tb_size = 0;
        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(pdsch_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No PDSCH PDU found in TV " << pdsch_object.tv_names[i]);
            }
            count++;
            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PDSCH)
            {
                continue;
            }
            pdsch_found = true;

            // Read beam IDs for this PDU (first PDU populates expected_beam_ids
            // as fallback; all PDUs populate per_pdu_beam_ids)
            tv_info pdu_beam_info{};
            try_read_beam_ids_from_pdu(pdu_pars, pdu_beam_info);
            if (dl_tv_info.expected_beam_ids.empty())
            {
                dl_tv_info.digBFInterfaces = pdu_beam_info.digBFInterfaces;
                dl_tv_info.expected_beam_ids = pdu_beam_info.expected_beam_ids;
            }

            // dl_tv_info.startPrb = pdu_pars["rbStart"].as<uint16_t>() + pdu_pars["BWPStart"].as<uint16_t>();
            // dl_tv_info.numPrb = pdu_pars["rbSize"].as<uint16_t>();

            // Remove concept of DMRS symbols so we check all symbols
#if 0
            uint32_t dmrsSymPos = pdu_pars["DmrsSymbPos"].as<uint32_t>();
            bool found_dmrs = false;
            int dmrsStartSym = 0;
            int dmrsNumSym = 0;

            // FIXME Assume DMRS Symbols are contiguous
            for(int i = 0; i < sizeof(uint32_t) * 8; ++i)
            {
                if(dmrsSymPos & 0x0001 == 1)
                {
                    if(found_dmrs == false)
                    {
                        found_dmrs = true;
                        dmrsStartSym = i;
                    }
                    ++dmrsNumSym;
                }
                dmrsSymPos = dmrsSymPos >> 1;
            }
#endif
            uint32_t resourceAllocType = pdu_pars["resourceAlloc"].as<uint32_t>();
            std::vector<std::pair<int, int>> prb_ranges;

            if (resourceAllocType == 0)
            {
#if 0
                prb_ranges.push_back({0, 6});
                prb_ranges.push_back({14, 8});
                prb_ranges.push_back({54, 8});
#else
                std::vector<uint32_t> rbBitmap = pdu_pars["rbBitmap"].as<std::vector<uint32_t>>();
                for (int k = 0, start = 0, cnt = 0, cur = 0, prev = 0; k < rbBitmap.size(); k++)
                {
                    re_info("rbBitmap[{}] 0x{:x}", k, rbBitmap[k]);
                    for (int j = 0; j < 8; j++)
                    {
                            cur = (rbBitmap[k] >> j) & 1;
                            if (cur && !prev) start = k * 8 + j + pdu_pars["BWPStart"].as<uint16_t>();
                            if (!cur && prev) prb_ranges.push_back({start, cnt});
                            cnt = cur * (cnt + cur);
                            prev = cur;
                    }
                }
#endif
            }
            else
            {
                prb_ranges.push_back({pdu_pars["rbStart"].as<uint16_t>() + pdu_pars["BWPStart"].as<uint16_t>(), pdu_pars["rbSize"].as<uint16_t>()});
            }

            for (auto &p : prb_ranges)
            {
                pdu_info pdu_info = {};
                dl_tv_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
                dl_tv_info.numSym = pdu_pars["NrOfSymbols"].as<uint8_t>();
                dl_tv_info.startDataSym = dl_tv_info.startSym;
                dl_tv_info.numDataSym = dl_tv_info.numSym;

                pdu_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
                pdu_info.numSym = pdu_pars["NrOfSymbols"].as<uint8_t>();
                pdu_info.startDataSym = pdu_info.startSym;
                pdu_info.numDataSym = pdu_info.numSym;
                pdu_info.startPrb = p.first;
                pdu_info.numPrb = p.second;
                pdu_info.tb_size = pdu_pars["TBSize"].as<uint32_t>();
                pdu_info.dmrsPorts = pdu_pars["dmrsPorts"].as<uint16_t>();
                pdu_info.scid = pdu_pars["SCID"].as<uint8_t>();
                //re_cons("{}:{} {}: TV dmrsPorts {} scid {}", __FILE__, __LINE__, __func__, pdu_info.dmrsPorts, pdu_info.scid);
                uint8_t pmidx = pdu_pars["PMidx"].as<uint8_t>();
                if(pmidx == 0)
                {
                    uint16_t tv_dmrsPorts = pdu_pars["dmrsPorts"].as<uint16_t>();
                    uint8_t dmrsPorts = (tv_dmrsPorts & 0xFFF);
                    uint8_t nlAbove16 = ((tv_dmrsPorts >> PDSCH_ABOVE_16_LAYERS_DMRSPORTS_BIT_LOC) & 0x1);
                    uint8_t scid = pdu_pars["SCID"].as<uint8_t>();

                    for (int flow_index = 0; flow_index < sizeof(dmrsPorts) * 8; ++flow_index)
                    {
                        if ((dmrsPorts >> flow_index) & 0x1)
                        {
                            pdu_info.flow_indices.push_back(16 * nlAbove16 + scid * 8 + flow_index);
                            //re_cons("{}:{} {}: TV dmrsPorts {} scid {} nlAbove16 {} flow_index_32dl {}", 
                            //    __FILE__, __LINE__, __func__, tv_dmrsPorts, scid, nlAbove16, 16 * nlAbove16 + scid * 8 + flow_index);
                        }
                    }
                }
                else
                {
                    std::string pmidx_dim_string = "PM" + std::to_string(pmidx) + "_dim";
                    if(!hdf5file.is_valid_dataset(pmidx_dim_string.c_str()))
                    {
                        do_throw(sb() << "PM" << (int)pmidx << "_dim not found in " << pdsch_object.tv_names[i]);
                    }
                    uint8_t nPorts = hdf5file.open_dataset(pmidx_dim_string.c_str())[0]["nPorts"].as<uint8_t>();

                    for(int port = 0; port < nPorts; ++port)
                    {
                        pdu_info.flow_indices.push_back(port);
                    }
                }

                dl_tv_info.numPrb += pdu_info.numPrb * pdu_info.flow_indices.size() * pdu_info.numDataSym;
                pdu_info.numFlows = pdu_info.flow_indices.size();
                dl_tv_info.pdu_infos.push_back(pdu_info);
                if (!pdu_beam_info.expected_beam_ids.empty())
                {
                    dl_tv_info.per_pdu_beam_ids.push_back({
                        static_cast<uint16_t>(p.first),
                        static_cast<uint16_t>(p.second),
                        pdu_beam_info.expected_beam_ids});
                }
                re_info("PDSCH PDU {} startSym {} startDataSym {} numSym {} numDataSym {} startPrb {} numPrb {} numFlows {}",
                        dl_tv_info.pdu_infos.size() - 1, pdu_info.startSym, pdu_info.startDataSym, pdu_info.numSym,
                        pdu_info.numDataSym, pdu_info.startPrb, pdu_info.numPrb, pdu_info.flow_indices.size());

                if(dl_tv_info.combined_pdu_infos.size() == 0)
                {
                    dl_tv_info.combined_pdu_infos.emplace_back(pdu_info);
                }
                else if(merge_pdu_if_identical(dl_tv_info.combined_pdu_infos, pdu_info))
                {
                    dl_tv_info.numPrb -= pdu_info.numPrb * pdu_info.flow_indices.size() * pdu_info.numDataSym;
                }
                else
                {
                    merge_pdu_if_adjacent(dl_tv_info.combined_pdu_infos, pdu_info);
                }
            }
            dl_tv_info.tb_size += pdu_pars["TBSize"].as<uint32_t>();
        }

        generate_prb_map(dl_tv_info, dl_tv_info.pdu_infos);
        dl_tv_info.numFlows = numFlows;
        auto compute_end = get_ns();
        timers.compute += compute_end - compute_start;
        auto load_qams_start = get_ns();
        re_info("Loading qams for {}", pdsch_object.tv_names[i].c_str());
        load_dl_qams(hdf5file, pdsch_object, dl_tv_info, opt_mod_comp_enabled == RE_ENABLED, opt_non_mod_comp_enabled == RE_ENABLED, cell_configs, selective_tv_load);
        auto load_qams_end = get_ns();
        timers.load += load_qams_end - load_qams_start;
        pdsch_object.tv_info.emplace_back(dl_tv_info);
        hdf5file.close();
    }
    re_cons("Load PDSCH TVs compute {:.2f}s load qams {:.2f}s", ((double)timers.compute) / NS_X_S, ((double)timers.load) / NS_X_S );
}

void RU_Emulator::load_pbch_tvs()
{
    for(int i = 0; i < pbch_object.tv_names.size(); ++i)
    {
        struct dl_tv_info dl_tv_info;
        dl_tv_info.numPrb = 0;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(pbch_object.tv_names[i].c_str());
        bool pbch_found = false;
        int count = 1;
        std::string pdu = "PDU";

        read_cell_cfg_from_tv(hdf5file, dl_tv_info, pbch_object.tv_names[i]);

        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(pbch_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No PBCH PDU found in TV " << pbch_object.tv_names[i]);
            }

            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::SSB)
            {
                count++;
                continue;
            }
            pbch_found = true;

            // Collect beam IDs from ALL SSB PDUs for 4T4R validation.
            // Multiple SSB beams share the same eAxC with different beam IDs,
            // so we accumulate all unique beam IDs across all SSB PDUs.
            {
                struct dl_tv_info temp_info{};
                try_read_beam_ids_from_pdu(pdu_pars, temp_info);
                for (uint16_t id : temp_info.expected_beam_ids)
                {
                    if (std::find(dl_tv_info.expected_beam_ids.begin(),
                                  dl_tv_info.expected_beam_ids.end(), id) == dl_tv_info.expected_beam_ids.end())
                    {
                        dl_tv_info.expected_beam_ids.push_back(id);
                    }
                }
                if (temp_info.digBFInterfaces > dl_tv_info.digBFInterfaces)
                {
                    dl_tv_info.digBFInterfaces = temp_info.digBFInterfaces;
                }
            }

            uint16_t ssbSubcarrierOffset = pdu_pars["ssbSubcarrierOffset"].as<uint16_t>();
            uint16_t ssbOffsetPointA = pdu_pars["SsbOffsetPointA"].as<uint16_t>();

            //FIXME Handle 15KHz case  f0 = (ssbSubcarrierOffset + ssbOffsetPointA * PRB_NUM_RE);
            uint16_t f0 = (ssbSubcarrierOffset + ssbOffsetPointA * PRB_NUM_RE) / 2;

            //Round up
            uint16_t startPrb = (f0 + PRB_NUM_RE - 1) / PRB_NUM_RE;
            uint16_t endPrb = (f0 + PBCH_MAX_SUBCARRIERS) / PRB_NUM_RE;
            uint16_t numPrb = endPrb - startPrb;

            pdu_info pdu_info = {};
            pdu_info.startPrb = startPrb;
            pdu_info.numPrb = numPrb;
            pdu_info.startSym = pdu_pars["nSSBStartSymbol"].as<uint8_t>();
            pdu_info.numSym = 4;

            uint8_t pmidx = pdu_pars["PMidx"].as<uint8_t>();
            uint8_t nPorts = 0;
            if(pmidx != 0)
            {
                std::string pmidx_dim_string = "PM" + std::to_string(pmidx) + "_dim";
                if(!hdf5file.is_valid_dataset(pmidx_dim_string.c_str()))
                {
                    do_throw(sb() << "PM" << (int)pmidx << "_dim not found in " << pdsch_object.tv_names[i]);
                }
                nPorts = hdf5file.open_dataset(pmidx_dim_string.c_str())[0]["nPorts"].as<uint8_t>();
            }

            if(opt_dl_approx_validation)//precoding enabled
            {
                pdu_info.numFlows = nPorts;
            }
            else if(opt_enable_beam_forming)//beam forming enabled
            {
                pdu_info.numFlows = pdu_pars["digBFInterfaces"].as<uint16_t>();
            }
            else
            {
                pdu_info.numFlows = 1;
                //pdu_info.numFlows = load_num_antenna_from_nr_tv(hdf5file);
            }

            for (int port = 0; port < pdu_info.numFlows; ++port)
            {
                pdu_info.flow_indices.push_back(port);
            }

            dl_tv_info.startPrb = startPrb;
            dl_tv_info.numSym = 4;
            dl_tv_info.numPrb += numPrb * dl_tv_info.numSym; // multiply the numFlows at validation because it is cell dependent
            dl_tv_info.startSym = pdu_pars["nSSBStartSymbol"].as<uint8_t>();
            dl_tv_info.numFlows = pdu_info.numFlows;// numFlows same for all pdus?

            dl_tv_info.pdu_infos.push_back(pdu_info);
            count++;
        }
        re_info("Loading qams for {}", pbch_object.tv_names[i].c_str());
        generate_prb_map(dl_tv_info, dl_tv_info.pdu_infos);
        load_dl_qams(hdf5file, pbch_object, dl_tv_info, opt_mod_comp_enabled == RE_ENABLED, opt_non_mod_comp_enabled == RE_ENABLED, cell_configs, selective_tv_load);
        //dl_tv_info.numFlows = load_first_dimension_dataset(hdf5file, pbch_object.tv_names[i], "X_tf");
        pbch_object.tv_info.emplace_back(dl_tv_info);
        hdf5file.close();
    }
}

void RU_Emulator::load_pdcch_ul_tvs()
{
    pdcch_ul_object.channel_type = dl_channel::PDCCH_UL;
    load_pdcch_tvs(pdcch_ul_object);
}

void RU_Emulator::load_pdcch_dl_tvs()
{
    pdcch_dl_object.channel_type = dl_channel::PDCCH_DL;
    load_pdcch_tvs(pdcch_dl_object);
}

inline uint32_t find_rightmost_bit(uint64_t val)
{
    return log2((val & (val - 1)) ^ val);
}

inline uint32_t count_set_bits(uint64_t val)
{
    uint64_t n   = val;
    uint32_t cnt = 0;
    while(n != 0)
    {
        n = n & (n - 1);
        cnt += 1;
    }    return cnt;
}

void parse_coreset(cuphyPdcchCoresetDynPrm_t& coreset, dci_param_list& dci, std::vector<std::pair<u_int16_t, u_int16_t>>& prb_pair) {
    uint64_t coreset_map;  /*Used as bitmask. Shifted version of freq_domain_resource */
    uint32_t rb_coreset;   /*Indicates the number of bits in coreset_map to be considered. It is # RBs divided by 6. */
    uint32_t n_CCE;

    rb_coreset = 64 - find_rightmost_bit(coreset.freq_domain_resource);
    coreset_map = (coreset.freq_domain_resource >> (64 - rb_coreset));
    n_CCE = count_set_bits(coreset.freq_domain_resource) * coreset.n_sym;

    uint32_t bundles_per_coreset_bit = 6 * coreset.n_sym / coreset.bundle_size;
    uint32_t N_bundle = n_CCE * bundles_per_coreset_bit / coreset.n_sym;  // N_bundle counts all the set bits in coreset map (specific to particular DCI)
    uint32_t N_bundle_phy = rb_coreset * bundles_per_coreset_bit; // N_bundle_phy counts all of available bundle in a particular coreset

    uint32_t bundle_table[slot_command_api::MAX_N_BUNDLE] = {0};  // Maps logical bundle ID to physical bundle ID
    uint32_t bundle_map[slot_command_api::MAX_N_BUNDLE]  = {0};  // Maps old logical bundle ID to new logical bundle ID

    int log_bundle_id = 0;
    int phy_bundle_id = 0;

    for(int i = 0; i < rb_coreset; i++)
    {
        if((coreset_map >> (rb_coreset - i - 1)) & 0x1)
        {
            for(int j = 0; j < bundles_per_coreset_bit; j++)
            {
                bundle_table[log_bundle_id + j] = phy_bundle_id + j;
                re_dbg("Logical bundle {} maps to physical bundle {}", log_bundle_id + j, phy_bundle_id + j);
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
            re_dbg( "Logical bundle {} maps to new logical bundle {} in interleaved mode", i, new_bundle_id);
        }
        bundle_map[i] = new_bundle_id;
    }

    int num_DCIs = coreset.nDci;
    uint16_t startRB=0, num_prbs = 0;
    uint32_t  used_bundle_map[slot_command_api::MAX_N_BUNDLE] = {0};  //Map of physical bundle used
    uint32_t used_bundle_dci_map[slot_command_api::MAX_N_BUNDLE] = {0};

    for(int i = 0; i < num_DCIs; i++)
    {
        cuphyPdcchDciPrm_t dci_params = dci[i];

        for(int j = 0; j < dci_params.aggr_level; j++)
        {
            for(int used_bundle = 0; used_bundle < 6 / coreset.bundle_size; used_bundle++)
            {
                uint32_t log_bundle_id = bundle_map[(6 / coreset.bundle_size) * (dci_params.cce_index + j) + used_bundle];
                used_bundle_map[bundle_table[log_bundle_id]] = 1;
                used_bundle_dci_map[bundle_table[log_bundle_id]] = i;
                re_info("DCI {}: physical bundle {} is used, orig. logical bundle {}, new logical bundle  {}. ",
                       i,
                       bundle_table[log_bundle_id],
                       (6 / coreset.bundle_size) * (dci_params.cce_index + j) + used_bundle,
                       log_bundle_id);
                // What does it mean to say "physical bundle X" is used?
                // It coresponds to a bit of coreset_map * 6 RBs * n_sym / bundleSize
                re_info("It will occupy REs from: [{} to {}) per symbol",
                       12 * (coreset.start_rb + bundle_table[log_bundle_id] * coreset.bundle_size / coreset.n_sym),
                       12 * (coreset.start_rb + (bundle_table[log_bundle_id] + 1) * coreset.bundle_size / coreset.n_sym));
            }
        }
    }

    int count = 0;
    uint32_t current_dci = UINT32_MAX;

    for(int j = 0; j < N_bundle_phy; j++)
    {
        if(used_bundle_map[j] == 1)
        {
            if(count == 0)
            {
                startRB = (j*coreset.bundle_size/ coreset.n_sym) + coreset.start_rb;
                current_dci = used_bundle_dci_map[j];
            }
            // else if (used_bundle_dci_map[j] != current_dci)
            // {
            //     num_prbs = (coreset.bundle_size / coreset.n_sym) * count;
            //     re_dbg("Start PRB = {}, Number of PRBs = {}", startRB, num_prbs);
            //     prb_pair.push_back(std::make_pair(startRB, num_prbs));
            //     current_dci = used_bundle_dci_map[j];
            //     startRB = (j*coreset.bundle_size/ coreset.n_sym) + coreset.start_rb;
            //     count = 0;
            // }

            count++;
        }
        else
        {
            if(count)
            {
                current_dci = used_bundle_dci_map[j];
                num_prbs = (coreset.bundle_size / coreset.n_sym) * count;
                re_dbg("Start PRB = {}, Number of PRBs = {}", startRB, num_prbs);
                prb_pair.push_back(std::make_pair(startRB, num_prbs));
            }
            count = 0;
        }
    }

    if(count)
    {
        current_dci = used_bundle_dci_map[N_bundle_phy - 1];
        num_prbs = (coreset.bundle_size / coreset.n_sym) * count;
        re_dbg("Start PRB = {}, Number of PRBs = {}", startRB, num_prbs);
        prb_pair.push_back(std::make_pair(startRB, num_prbs));
    }
}

void RU_Emulator::load_pdcch_tvs(struct dl_tv_object& tv_object)
{
    struct tv_parsing_timers timers;
    timers.compute = 0;
    timers.load = 0;
    for(int i = 0; i < tv_object.tv_names.size(); ++i)
    {
        auto compute_start = get_ns();
        Dataset d;
        struct dl_tv_info dl_tv_info;
        std::string dset_name;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(tv_object.tv_names[i].c_str());
        int count = 1;
        std::string pdu = "PDU";
        std::string dci = "DCI";
        bool pdcch_found = false;
        dl_tv_info.numPrb = 0;

        read_cell_cfg_from_tv(hdf5file, dl_tv_info, tv_object.tv_names[i]);

        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(pdcch_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No PDCCH PDU found in TV " << tv_object.tv_names[i]);
            }

            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PDCCH)
            {
                count++;
                continue;
            }
            dl_channel ch =  pdu_pars["dciUL"].as<uint8_t>()? dl_channel::PDCCH_UL: dl_channel::PDCCH_DL;
            if (ch != tv_object.channel_type) {
                count++;
                continue;
            }
            pdcch_found = true;

            // Read beam IDs for this PDU (first PDU populates expected_beam_ids
            // for fallback; all PDUs populate per_pdu_beam_ids)
            tv_info pdu_beam_info{};
            try_read_beam_ids_from_pdu(pdu_pars, pdu_beam_info);
            if (dl_tv_info.expected_beam_ids.empty())
            {
                dl_tv_info.digBFInterfaces = pdu_beam_info.digBFInterfaces;
                dl_tv_info.expected_beam_ids = pdu_beam_info.expected_beam_ids;
            }

            cuphyPdcchCoresetDynPrm_t pdcchCoresetParam;
            pdcchCoresetParam.freq_domain_resource = static_cast<uint64_t>(pdu_pars["FreqDomainResource0"].as<uint32_t>()) << 32 | pdu_pars["FreqDomainResource1"].as<uint32_t>();
            pdcchCoresetParam.start_rb = pdu_pars["BWPStart"].as<uint32_t>();
            pdcchCoresetParam.n_sym = pdu_pars["DurationSymbols"].as<uint32_t>();
            pdcchCoresetParam.bundle_size = pdu_pars["RegBundleSize"].as<uint32_t>();
            pdcchCoresetParam.interleaver_size = pdu_pars["InterleaverSize"].as<uint32_t>();
            pdcchCoresetParam.shift_index = pdu_pars["ShiftIndex"].as<uint32_t>();
            pdcchCoresetParam.interleaved = pdu_pars["CceRegMappingType"].as<uint32_t>();
            pdcchCoresetParam.nDci = pdu_pars["numDlDci"].as<uint8_t>();

            uint8_t nPorts = 0;
            dci_param_list dciParamList;
            for(int i = 0; i < pdcchCoresetParam.nDci; i++)
            {
                cuphyPdcchDciPrm_t dci_param = {};
                std::string dci_dset_string = pdu + std::to_string(count) + "_" + dci +  std::to_string(i+1);

                hdf5hpp::hdf5_dataset dset_DCI  = hdf5file.open_dataset(dci_dset_string.c_str());
                hdf5hpp::hdf5_dataset_elem dci_pars = dset_DCI[0];

                dci_param.aggr_level = dci_pars["AggregationLevel"].as<uint32_t>();
                dci_param.cce_index = dci_pars["CceIndex"].as<uint32_t>();

                uint8_t pmidx = dci_pars["PMidx"].as<uint8_t>();

                if(pmidx != 0)
                {
                    std::string pmidx_dim_string = "PM" + std::to_string(pmidx) + "_dim";
                    if(!hdf5file.is_valid_dataset(pmidx_dim_string.c_str()))
                    {
                        do_throw(sb() << "PM" << (int)pmidx << "_dim not found in " << pdsch_object.tv_names[i]);
                    }
                    nPorts = hdf5file.open_dataset(pmidx_dim_string.c_str())[0]["nPorts"].as<uint8_t>();
                }

                //Same for all DCIs?
                if(opt_dl_approx_validation)//precoding enabled
                {
                    dl_tv_info.numFlows = nPorts;
                }
                else if(opt_enable_beam_forming)//beam forming enabled
                {
                    dl_tv_info.numFlows = dci_pars["digBFInterfaces"].as<uint16_t>();
                }
                else
                {
                    dl_tv_info.numFlows = 1;
                    //dl_tv_info.numFlows = load_num_antenna_from_nr_tv(hdf5file);
                }

                dciParamList.push_back(dci_param);
            }

            std::vector<std::pair<u_int16_t, u_int16_t>> prbPair;
            parse_coreset(pdcchCoresetParam, dciParamList, prbPair);

            for(int i = 0; i< prbPair.size(); i++)
            {
                uint16_t startPrb = prbPair[i].first;
                uint16_t numPrb = prbPair[i].second;
                pdu_info pdu_info;
                for (int port = 0; port < dl_tv_info.numFlows; ++port)
                {
                    pdu_info.flow_indices.push_back(port);
                }
                pdu_info.startPrb = startPrb;
                pdu_info.numPrb = numPrb;
                pdu_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
                pdu_info.numSym = pdu_pars["DurationSymbols"].as<uint8_t>();
                //pdu_info.numFlows = load_num_antenna_from_nr_tv(hdf5file);
                pdu_info.freqDomainResource = pdcchCoresetParam.freq_domain_resource;
                dl_tv_info.startPrb = startPrb;
                dl_tv_info.numPrb += numPrb * pdu_info.numSym; // * pdu_info.numFlows, flow will be decided at validation
                dl_tv_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
                dl_tv_info.numSym = pdu_pars["DurationSymbols"].as<uint8_t>();
                if (!pdu_beam_info.expected_beam_ids.empty())
                {
                    dl_tv_info.per_pdu_beam_ids.push_back({
                        startPrb, numPrb,
                        pdu_beam_info.expected_beam_ids});
                }
                dl_tv_info.pdu_infos.push_back(std::move(pdu_info));

            }
            count++;
        }
        sort(begin(dl_tv_info.pdu_infos), end(dl_tv_info.pdu_infos), [](const pdu_info & pdu1,const pdu_info & pdu2){return pdu1.startPrb < pdu2.startPrb;});
        auto compute_end = get_ns();
        timers.compute += compute_end - compute_start;
        auto load_qams_start = get_ns();
        re_info("Loading qams for {}", tv_object.tv_names[i].c_str());
        generate_prb_map(dl_tv_info, dl_tv_info.pdu_infos);
        load_dl_qams(hdf5file, tv_object, dl_tv_info, opt_mod_comp_enabled == RE_ENABLED, opt_non_mod_comp_enabled == RE_ENABLED, cell_configs, selective_tv_load);
        auto load_qams_end = get_ns();
        timers.load += load_qams_end - load_qams_start;
        //dl_tv_info.numFlows = load_first_dimension_dataset(hdf5file, tv_object.tv_names[i], "X_tf");
        tv_object.tv_info.emplace_back(dl_tv_info);
        hdf5file.close();
    }
    re_cons("Load PDCCH TVs compute {:.2f}s load qams {:.2f}s", ((double)timers.compute) / NS_X_S, ((double)timers.load) / NS_X_S );
}

bool RU_Emulator::does_csirs_tv_have_pdsch(std::string filename)
{
    bool has_pdsch = false;
    hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(filename.c_str());
    int count = 1;
    std::string pdu = "PDU";
    while(1)
    {
        std::string dset_string = pdu + std::to_string(count);
        if(!hdf5file.is_valid_dataset(dset_string.c_str()))
        {
            return false;
        }

        hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
        hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
        if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PDSCH)
        {
            count++;
            continue;
        }
        return true;
    }
}

bool RU_Emulator::is_csirs_re_in_pdsch(std::string filename, int re_index)
{
    if(!opt_pdsch_validation)
    {
        return false;
    }
    auto it = std::find(pdsch_object.tv_names.begin(), pdsch_object.tv_names.end(), filename);
    if(it == pdsch_object.tv_names.end())
    {
        return false;
    }
    int tv_index = it - pdsch_object.tv_names.begin();

    auto prbs_per_symbol = pdsch_object.tv_info[tv_index].nPrbDlBwp;
    for(auto& pdu_info: pdsch_object.tv_info[tv_index].combined_pdu_infos)
    {
        int re_flow = re_index / (ORAN_ALL_SYMBOLS * prbs_per_symbol * PRB_NUM_RE);
        int re_symbol = (re_index / (prbs_per_symbol * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS;
        int re_prb = (re_index / PRB_NUM_RE) % prbs_per_symbol;

        auto valid_flow = std::find(pdu_info.flow_indices.begin(), pdu_info.flow_indices.end(), re_flow) != pdu_info.flow_indices.end();
        auto valid_symbol = (re_symbol >= pdu_info.startDataSym && re_symbol < pdu_info.startDataSym + pdu_info.numDataSym);
        auto valid_prb = (re_prb >= pdu_info.startPrb && re_prb < pdu_info.startPrb + pdu_info.numPrb);
        if(valid_flow && valid_symbol && valid_prb)
        {
            return true;
        }
    }
    return false;
}

void RU_Emulator::flag_pdsch_tv_with_zp_csirs(std::string filename)
{
    if(!opt_pdsch_validation)
    {
        return;
    }
    auto it = std::find(pdsch_object.tv_names.begin(), pdsch_object.tv_names.end(), filename);
    if(it == pdsch_object.tv_names.end())
    {
        return;
    }
    int tv_index = it - pdsch_object.tv_names.begin();
    re_info("Flagging DL TV {} for PDSCH with non-overlapping ZP CSIRS", filename.c_str());
    pdsch_object.tv_info[tv_index].hasZPCsirsPdu = true;
}

void RU_Emulator::add_num_overlapping_nzp_csirs_pdsch(std::string filename, uint32_t numOverlappingCsirs, bool fullyOverlapping)
{
    if(!opt_pdsch_validation)
    {
        return;
    }
    auto it = std::find(pdsch_object.tv_names.begin(), pdsch_object.tv_names.end(), filename);
    if(it == pdsch_object.tv_names.end())
    {
        return;
    }
    int tv_index = it - pdsch_object.tv_names.begin();
    re_info("Adding DL TV {} for PDSCH with overlapping NZP CSIRS count: {}", filename.c_str(), numOverlappingCsirs);
    pdsch_object.tv_info[tv_index].numOverlappingCsirs = numOverlappingCsirs;
    pdsch_object.tv_info[tv_index].fullyOverlappingCsirs = fullyOverlapping;
}

void RU_Emulator::load_csirs_tvs()
{
    float rho_vals[4] = {0.5f, 0.5f, 1, 3};
    csirs_lookup_api::CsirsLookup& lookup = csirs_lookup_api::CsirsLookup::getInstance();
    for(int i = 0; i < csirs_object.tv_names.size(); ++i)
    {
        Dataset d;
        struct dl_tv_info tv_info{};
        std::string dset_name;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(csirs_object.tv_names[i].c_str());

        read_cell_cfg_from_tv(hdf5file, tv_info, csirs_object.tv_names[i]);

        int count = 1;
        std::string pdu = "PDU";
        bool csirs_found = false;
        tv_info.isZP = true;
        uint8_t rb_indicator = 0;
        uint8_t csirs_max_port_num = 0;
        // FIXME, ASSUME ONLY 1 CSI_RS PDU
        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(csirs_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No CSI-RS PDU found in TV " << csirs_object.tv_names[i]);
            }

            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::CSI_RS)
            {
                count++;
                continue;
            }
            csirs_found = true;

            // Use CSIType from the PDU to distinguish ZP from NZP CSI-RS.
            // ZP CSI-RS beam IDs are meaningless (no signal transmitted) and stored
            // separately so the validator can identify and skip ZP sections.
            // Only NZP CSI-RS beam IDs are validated.
            uint8_t csirs_type = pdu_pars["CSIType"].as<uint8_t>();
            {
                struct dl_tv_info temp_info{};
                try_read_beam_ids_from_pdu(pdu_pars, temp_info);
                if (!temp_info.expected_beam_ids.empty())
                {
                    if (cuphyCsiType_t::ZP_CSI_RS != csirs_type)
                    {
                        // Non-ZP CSI-RS (TRS or NZP): store as a beam ID set for validation
                        tv_info.csirs_beam_id_sets.push_back(std::move(temp_info.expected_beam_ids));
                    }
                    // ZP CSI-RS: no signal transmitted, beam ID is meaningless — skip
                }
            }

            // FIXME: Assume CSI_RS PDU type will be either all ZP or all !ZP
            if(cuphyCsiType_t::ZP_CSI_RS != csirs_type)
            {
                tv_info.isZP = false;
            }
            auto freqDensity = pdu_pars["FreqDensity"].as<uint8_t>();
            if (rho_vals[freqDensity & 0x3] == 0.5f)
            {
                rb_indicator = 1;
            }
            uint8_t num_flow = 0;
            uint8_t pmidx = pdu_pars["PMidx"].as<uint8_t>();
            if(pmidx != 0 && opt_dl_approx_validation)//precoding enabled
            {
                std::string pmidx_dim_string = "PM" + std::to_string(pmidx) + "_dim";
                if(!hdf5file.is_valid_dataset(pmidx_dim_string.c_str()))
                {
                    do_throw(sb() << "PM" << (int)pmidx << "_dim not found in " << pdsch_object.tv_names[i]);
                }
                uint8_t nPorts = hdf5file.open_dataset(pmidx_dim_string.c_str())[0]["nPorts"].as<uint8_t>();
                num_flow = nPorts;
            }
            else if(opt_enable_beam_forming)//beam forming enabled
            {
                num_flow = pdu_pars["digBFInterfaces"].as<uint16_t>();
            }
            else
            {
                uint8_t csirs_row_idx = pdu_pars["Row"].as<uint8_t>()-1;
                num_flow = slot_command_api::csirs_tables.rowData[csirs_row_idx].numPorts;
            }

            if(cuphyCsiType_t::ZP_CSI_RS != csirs_type)
            {
                auto sym = pdu_pars["SymbL0"].as<uint16_t>();
                tv_info.numFlowsArray[sym] = num_flow;
                if (pdu_pars["Row"].as<uint8_t>() > 6)
                {
                    auto sym1 = pdu_pars["SymbL1"].as<uint16_t>();
                    tv_info.numFlowsArray[sym1] = num_flow;
                }
            }

            {
                uint8_t row = pdu_pars["Row"].as<uint8_t>();//-1;
                uint16_t freq_domain = pdu_pars["FreqDomain"].as<uint16_t>();
                uint8_t syml0 = pdu_pars["SymbL0"].as<uint8_t>();
                uint8_t syml1 = pdu_pars["SymbL1"].as<uint8_t>();

                const csirs_lookup_api::CsirsPortData *outPortInfo = nullptr;
                bool success = lookup.getPortInfo(row, freq_domain, syml0, syml1, outPortInfo);

                re_dbg("Input: row={}, freq_domain=0x{:x}, syml0={}, syml1={}", row, freq_domain, syml0, syml1);
                re_dbg("Result: {}", success ? "Found" : "Not Found");

                if (success)
                {
                    re_dbg("Number of ports: {}", outPortInfo->num_ports);
                    csirs_max_port_num = std::max(csirs_max_port_num, outPortInfo->num_ports);
                    for (uint8_t port = 0; port < outPortInfo->num_ports; port++)
                    {
                        re_dbg("Port : {}", port);
                        re_dbg("  Symbol mask: 0x{:x}", outPortInfo->port_tx_locations[port].symbol_mask);
                        re_dbg("  RE mask: 0x{:x}", outPortInfo->port_tx_locations[port].re_mask);
                    }
                }
            }

            count++;
        }
        tv_info.csirsMaxPortNum = csirs_max_port_num;
        re_info("Loading CSI_RS qams for {}", csirs_object.tv_names[i].c_str());
        load_dl_qams(hdf5file, csirs_object, tv_info, opt_mod_comp_enabled == RE_ENABLED, opt_non_mod_comp_enabled == RE_ENABLED, cell_configs, selective_tv_load);

        Dataset re_map = std::move(load_tv_datasets_single(hdf5file, csirs_object.tv_names[i], tv_info.isZP ? "Xtf_remap" : "Xtf_remap_trsnzp"));
        uint32_t* re_map_array = static_cast<uint32_t*>(re_map.data.get());

        for(uint32_t re_index = 0; re_index < re_map.size / sizeof(re_map_array[0]); ++re_index)
        {
            if(re_map_array[re_index] == 1)
            {
                // Skip CSI_RS REs if they are ZP and do not overlap PDSCH
                // Push back NZP if CSIRS does not overlap with PDSCH. ASSUME PDSCH is always run alongside CSI_RS
                // Count the number of overlapping REs
                bool overlap = is_csirs_re_in_pdsch(csirs_object.tv_names[i], re_index);
                if(!tv_info.isZP && !overlap) // NZP REs that do not overlap with PDSCH, CSI_RS needs to validate them
                {
                    tv_info.csirsREsToValidate.push_back(std::vector<uint32_t>({re_index, 0}));
                    tv_info.numNonOverlappingCsirs++;
                }
                else //All ZP REs or NZP REs that are overlapping with PDSCH
                {
                    tv_info.csirsSkippedREs.push_back(re_index);
                    tv_info.csirsNumREsSkipped++;
                }
                auto sym = (re_index / (tv_info.nPrbDlBwp * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS;
                tv_info.csirsExpectedNumREs += tv_info.numFlowsArray[sym];
            }
        }
        tv_info.csirsNumREs = tv_info.csirsREsToValidate.size();

        re_map = std::move(load_tv_datasets_single(hdf5file, csirs_object.tv_names[i], "Xtf_remap_trsnzp"));
        re_map_array = static_cast<uint32_t*>(re_map.data.get());
        for(uint32_t re_index = 0, reMask = 0, cnt = 0; re_index < re_map.size / sizeof(re_map_array[0]); ++re_index)
        {
            reMask = (reMask << 1) | re_map_array[re_index];
            if(++cnt == PRB_NUM_RE) {
#if 0
                static int dbg_cnt = 0;
                if((dbg_cnt++ % MAX_NUM_PRBS_PER_SYMBOL) == 0) printf("\n \n");
                printf("0x%x  ", reMask);
#endif
                tv_info.csirsREMaskArrayTRSNZP.push_back(reMask);
                cnt = 0;
                reMask = 0;
            }
        }

        re_map = std::move(load_tv_datasets_single(hdf5file, csirs_object.tv_names[i], "Xtf_remap"));
        re_map_array = static_cast<uint32_t*>(re_map.data.get());
        for(uint32_t re_index = 0, reMask = 0, cnt = 0; re_index < re_map.size / sizeof(re_map_array[0]); ++re_index)
        {
            reMask = (reMask << 1) | re_map_array[re_index];
            if(++cnt == PRB_NUM_RE) {
#if 0
                static int dbg_cnt = 0;
                if((dbg_cnt++ % tv_info.nPrbDlBwp) == 0) printf("\n \n");
                printf("0x%x  ", reMask);
#endif
                tv_info.csirsREMaskArray.push_back(reMask);
                cnt = 0;
                reMask = 0;
            }
        }

        if(does_csirs_tv_have_pdsch(csirs_object.tv_names[i].c_str()) && !opt_pdsch_validation)
        {
            re_cons("WARNING: CSI_RS TV has PDSCH, are you sure you only want to run without PDSCH validation? The IQ Samples may be mismatched.");
            sleep(3);
        }

        // If CSI_RS is ZP, assume we are testing with PDSCH
        if(tv_info.csirsREsToValidate.size() == 0 && tv_info.isZP && tv_info.csirsNumREsSkipped != 0)
        {
            flag_pdsch_tv_with_zp_csirs(csirs_object.tv_names[i]);
        }

        // If NZP CSI_RS has overlapping REs in PDSCH
        if(!tv_info.isZP && tv_info.csirsNumREsSkipped != 0)
        {
            add_num_overlapping_nzp_csirs_pdsch(csirs_object.tv_names[i], tv_info.csirsNumREsSkipped, (tv_info.csirsNumREs == 0));
        }

        //Store number of REs in the same PRB with higher index than self in second element.
        for(int re_index = 0; re_index < tv_info.csirsREsToValidate.size(); ++re_index)
        {
            for(int next_index = re_index; next_index < tv_info.csirsREsToValidate.size(); ++next_index)
            {
                int curr_re_loc = tv_info.csirsREsToValidate[re_index][0];
                int next_re_loc = tv_info.csirsREsToValidate[next_index][0];
                if(curr_re_loc % PRB_NUM_RE == next_re_loc % PRB_NUM_RE)
                {
                    tv_info.csirsREsToValidate[re_index][1]++;
                }
                else
                {
                    break;
                }
            }
        }

#if 0
        tv_info.numFlows = load_num_antenna_from_nr_tv(hdf5file);
        if(tv_info.isZP)
            tv_info.numFlows = load_num_antenna_from_nr_tv_zp_csi_rs(hdf5file);
#endif

        if(csi_rs_optimized_validation)
        {
            // Optimization to assume all CSI_RS non-overlapping REs to be after all of PDSCH and other channels
            // Also calculate startSym, numSym, startPrb, and numPrb of CSI_RS block
            std::vector<int> symbols;
            for(int re_i = 0; re_i < tv_info.csirsNumREs; ++re_i)
            {
                int symbol = (tv_info.csirsREsToValidate[re_i][0] / (tv_info.nPrbDlBwp * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS;
                if(std::find(symbols.begin(), symbols.end(), symbol) == symbols.end())
                {
                    symbols.emplace_back(symbol);
                }
            }

            // Construct a pdu info per symbol in CSI_RS
            tv_info.numPrb = 0;
            for(const auto& symbol: symbols)
            {
                pdu_info pdu_info = {};
                pdu_info.startSym = symbol;
                pdu_info.numSym = 1;
                pdu_info.startPrb = MAX_NUM_PRBS_PER_SYMBOL;
                pdu_info.numPrb = 0;
                pdu_info.rb = rb_indicator;
                pdu_info.numFlows = tv_info.numFlowsArray[symbol];
                int endPrb = 0;
                for(int re_i = 0; re_i < tv_info.csirsNumREs; ++re_i)
                {
                    if(symbol != (tv_info.csirsREsToValidate[re_i][0] / (tv_info.nPrbDlBwp * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS)
                    {
                        continue;
                    }
                    int prb = (tv_info.csirsREsToValidate[re_i][0] / PRB_NUM_RE) % tv_info.nPrbDlBwp;
                    pdu_info.startPrb = (prb < pdu_info.startPrb) ? prb : pdu_info.startPrb;
                    endPrb = (prb > endPrb) ? prb : endPrb;
                }
                pdu_info.numPrb = endPrb - pdu_info.startPrb + 1;
                tv_info.numPrb += pdu_info.numPrb * pdu_info.numFlows;
                tv_info.pdu_infos.emplace_back(pdu_info);
                tv_info.csirc_pdu_infos.emplace_back(pdu_info);
            }
        }

        // Add overlapping CSI_RS pdu info as well for reMask verification
        // Also calculate startSym, numSym, startPrb, and numPrb of CSI_RS block
        std::vector<int> symbols;
        for(int re_i = 0; re_i < tv_info.csirsSkippedREs.size(); ++re_i)
        {
            int symbol = (tv_info.csirsSkippedREs[re_i] / (tv_info.nPrbDlBwp * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS;
            if(std::find(symbols.begin(), symbols.end(), symbol) == symbols.end())
            {
                symbols.emplace_back(symbol);
            }
        }

        // Construct a pdu info per symbol in CSI_RS
        for(const auto& symbol: symbols)
        {
            pdu_info pdu_info = {};
            pdu_info.startSym = symbol;
            pdu_info.numSym = 1;
            pdu_info.startPrb = MAX_NUM_PRBS_PER_SYMBOL;
            pdu_info.numPrb = 0;
            pdu_info.rb = rb_indicator;
            pdu_info.numFlows = tv_info.numFlows;
            int endPrb = 0;
            for(int re_i = 0; re_i < tv_info.csirsSkippedREs.size(); ++re_i)
            {
                if(symbol != (tv_info.csirsSkippedREs[re_i] / (tv_info.nPrbDlBwp * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS)
                {
                    continue;
                }
                int prb = (tv_info.csirsSkippedREs[re_i] / PRB_NUM_RE) % tv_info.nPrbDlBwp;
                pdu_info.startPrb = (prb < pdu_info.startPrb) ? prb : pdu_info.startPrb;
                endPrb = (prb > endPrb) ? prb : endPrb;
            }
            pdu_info.numPrb = endPrb - pdu_info.startPrb + 1;
            tv_info.csirc_pdu_infos.emplace_back(pdu_info);
        }

        generate_prb_map(tv_info, tv_info.csirc_pdu_infos);

        csirs_object.tv_info.emplace_back(tv_info);
        re_info("CSI_RS TV {} numREs {} numFlows {}", csirs_object.tv_names[i].c_str(), tv_info.csirsNumREs, tv_info.numFlows);
        hdf5file.close();
    }

}

void RU_Emulator::generate_prb_map(struct tv_info &tv_info_, std::vector<pdu_info> &pdu_infos)
{
    for (auto &pdu_info : pdu_infos)
    {
        for (int sym = pdu_info.startSym; sym < pdu_info.startSym + pdu_info.numSym; sym++)
        {
            for (int prb = pdu_info.startPrb; prb < pdu_info.startPrb + pdu_info.numPrb; prb += pdu_info.rb + 1)
            {
                tv_info_.prb_map[sym][prb] = true;
                for(auto f : pdu_info.flow_indices)
                {
                    tv_info_.prb_num_flow_map[sym][prb] |= (uint64_t)1 << f;
                }
            }

            if (pdu_info.freqHopFlag > 0)
            {
                for (int prb = pdu_info.secondHopPrb; prb < pdu_info.secondHopPrb + pdu_info.numPrb; prb += pdu_info.rb + 1)
                {
                    tv_info_.prb_map[sym][prb] = true;
                    for (auto f : pdu_info.flow_indices)
                    {
                        tv_info_.prb_num_flow_map[sym][prb] |= (uint64_t)1 << f;
                    }
                }
            }
        }
    }
}

void RU_Emulator::load_bfw_tvs(bool dirDL)
{
    auto& tv_object = (dirDL) ? bfw_dl_object : bfw_ul_object;
    for(int i = 0; i < tv_object.tv_names.size(); ++i)
    {
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(tv_object.tv_names[i].c_str());
        struct dl_tv_info tv_info{};
        int count = 1;
        std::string pdu = "PDU";
        bool bfw_found = false;
        while(!check_force_quit())
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(bfw_found)
                {
                    break;
                }
                do_throw(sb() << "ERROR No BFW PDU found in TV " << tv_object.tv_names[i]);
            }

            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::BFW)
            {
                count++;
                continue;
            }
            bfw_found = true;
            count++;

            bfw_info bfw_info{};
            bfw_info.bfwUL = pdu_pars["bfwUL"].as<uint32_t>();
            bfw_info.prgSize = pdu_pars["prgSize"].as<uint32_t>();
            bfw_info.bfwPrbGrpSize = pdu_pars["bfwPrbGrpSize"].as<uint32_t>();
            bfw_info.rbStart = pdu_pars["rbStart"].as<uint32_t>();
            bfw_info.rbSize = pdu_pars["rbSize"].as<uint32_t>();
            bfw_info.numPRGs = pdu_pars["numPRGs"].as<uint32_t>();
            bfw_info.compressBitWidth = pdu_pars["compressBitWidth"].as<uint32_t>();
            // bfw_info.portMask = 0;// Need to read PDSCH/PUSCH TV
            // read next slot non-BFW TV, match the rbStart and rbSize to determine which UEG/PDU.
            for(int slot_idx = 0; slot_idx < tv_object.launch_pattern.size(); ++slot_idx)
            {
                for(int cell_idx = 0; cell_idx < opt_num_cells; ++cell_idx)
                {
                    uint32_t ap_index = 0;
                    auto update_active_eaxc = [&bfw_info, &ap_index, cell_idx, slot_idx](const uint32_t dmrsPorts,
                                                                                     const uint32_t scid,
                                                                                     const uint8_t nlAbove16,
                                                                                     [[maybe_unused]] const char* dir,
                                                                                     [[maybe_unused]] const char* file,
                                                                                     [[maybe_unused]] const int line)
                    {
                        uint32_t tempMask = dmrsPorts << (scid * 8);
                        re_dbg("{}:{} {}: cell_idx = {} dmrsPorts = {} scid = {} tempMask = {} nlAbove16 = {}", file, line, dir, cell_idx, dmrsPorts, scid, tempMask, nlAbove16);
                        while (tempMask) {
                            const uint32_t bit_pos = __builtin_ctz(tempMask) + nlAbove16 * 16;
                            bfw_info.active_eaxc_ids[cell_idx][slot_idx][bit_pos] = ap_index++;
                            //re_dbg("{}:{} {}: bfw_info.active_eaxc_ids[{}][{}][{}] = {}",
                            //    file, line, dir, cell_idx, slot_idx, bit_pos, ap_index - 1);
                            tempMask &= tempMask - 1;  // Clear the lowest set bit
                        }
                    };
                    if(tv_object.launch_pattern[slot_idx].find(cell_idx) != 0)
                    {
                        bfw_info.portMask[cell_idx][slot_idx] = 0;
                        if(tv_object.launch_pattern[slot_idx][cell_idx] == i)
                        {
                            if(dirDL)
                            {
                                for(const auto& pdu_info: pdsch_object.tv_info[pdsch_object.launch_pattern[slot_idx][cell_idx]].pdu_infos)
                                {
                                    if(pdu_info.startPrb == bfw_info.rbStart && pdu_info.numPrb == bfw_info.rbSize)
                                    {
                                        //re_cons("{}:{} {}: TV dmrsPorts {} scid {}", __FILE__, __LINE__, __func__, static_cast<uint32_t>(pdu_info.dmrsPorts), static_cast<uint32_t>(pdu_info.scid));
                                        uint8_t nlAbove16 = static_cast<uint8_t>((pdu_info.dmrsPorts >> PDSCH_ABOVE_16_LAYERS_DMRSPORTS_BIT_LOC) & 0x1);
                                        auto dmrsPortsWithoutAbove16 = pdu_info.dmrsPorts & 0xFFF; // extracting 12 bits of dmrsPorts
                                        bfw_info.portMask[cell_idx][slot_idx] |= (dmrsPortsWithoutAbove16 << (pdu_info.scid * 8)) << 16 * nlAbove16;
                                        //re_cons("{}:{} cell_idx = {} slot_idx = {} pdu_info.dmrsPorts = {} pdu_info.scid = {} nlAbove16 = {} bfw_info.portMask[{}][{}] = {}", 
                                        //        __FILE__, __LINE__, cell_idx, slot_idx, static_cast<uint32_t>(pdu_info.dmrsPorts), static_cast<uint32_t>(pdu_info.scid), 
                                        //        nlAbove16, cell_idx, slot_idx, bfw_info.portMask[cell_idx][slot_idx]);
                                        {
                                            update_active_eaxc(dmrsPortsWithoutAbove16,
                                                               pdu_info.scid,
                                                               nlAbove16,
                                                               "DL",
                                                               __FILE__,
                                                               __LINE__);
                                        }
                                    }
                                }
                                //re_cons("{}:{} DL: bfw_info.portMask[{}]][{}] = {}", __FILE__, __LINE__, cell_idx, slot_idx, bfw_info.portMask[cell_idx][slot_idx]);
                            }
                            else
                            {
                                for(const auto& pdu_info: pusch_object.tv_info[pusch_object.launch_pattern[slot_idx][cell_idx]].pdu_infos)
                                {
                                    if(pdu_info.startPrb == bfw_info.rbStart && pdu_info.numPrb == bfw_info.rbSize)
                                    {
                                        bfw_info.portMask[cell_idx][slot_idx] |= (pdu_info.dmrsPorts << (pdu_info.scid * 8));
                                        {
                                            update_active_eaxc(pdu_info.dmrsPorts,
                                                               pdu_info.scid,
                                                               0, // max number of layers in UL is 8. So nlAbove16 is 0.
                                                               "UL",
                                                               __FILE__,
                                                               __LINE__);
                                        }
                                    }
                                }
                                //re_cons("{}:{} UL: bfw_info.portMask[{}]][{}] = {}", __FILE__, __LINE__, cell_idx, slot_idx, bfw_info.portMask[cell_idx][slot_idx]);
                            }
                            bfw_info.expect_prbs[cell_idx][slot_idx] = bfw_info.rbSize * __builtin_popcount(bfw_info.portMask[cell_idx][slot_idx]);
                            tv_info.total_expected_prbs[cell_idx][slot_idx] += bfw_info.expect_prbs[cell_idx][slot_idx];
                        }
                    }
                }
            }

#if 0
            for (int slot_idx = 0; slot_idx < tv_object.launch_pattern.size(); ++slot_idx)
            {
                for (int cell_idx = 0; cell_idx < opt_num_cells; ++cell_idx)
                {
                    if (tv_object.launch_pattern[slot_idx].find(cell_idx) != 0)
                    {
                        bfw_info.portMask[cell_idx][slot_idx] = 0;
                        if (tv_object.launch_pattern[slot_idx][cell_idx] == i)
                        {
                            printf("cell_idx %d, slot_idx %d,  tv_info.total_expected_prbs %d\n", cell_idx, slot_idx, tv_info.total_expected_prbs[cell_idx][slot_idx]);
                        }
                    }
                }
            }
#endif
            tv_info.bfw_infos.push_back(bfw_info);
        }

        load_bfw_qams(hdf5file, tv_object);
        tv_object.tv_info.emplace_back(tv_info);
        hdf5file.close();
    }
}

Slot dataset_to_slot(Dataset d, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz, size_t tv_prbs_per_symbol, size_t start_prb)
{
    Slot ret(num_ante);

    ret.data_sz = d.size;
    if (ret.data_sz % num_symbols)
        do_throw(sb() << "Slot size " << ret.data_sz
             << " doesn't divide into the number of symbols "
             << num_symbols);

    ret.antenna_sz = d.size / num_ante;
    ret.symbol_sz = ret.antenna_sz / num_symbols;
    if (ret.symbol_sz % prb_sz)
        do_throw(sb() << "Symbol size " << ret.symbol_sz
             << " doesn't divide into the size of a PRB "
             << prb_sz);

    ret.prbs_per_symbol = ret.symbol_sz / prb_sz;
    if (ret.prbs_per_symbol > MAX_NUM_PRBS_PER_SYMBOL)
        do_throw(sb() << "Resulting number of PRBs per symbol "
             << ret.prbs_per_symbol << " is higher than MAX_NUM_PRBS_PER_SYMBOL. "
             << "Please recompile with a higher value for "
             << "MAX_NUM_PRBS_PER_SYMBOL.");
    ret.prbs_per_symbol = tv_prbs_per_symbol;
    ret.prbs_per_slot = ret.prbs_per_symbol * num_symbols * num_ante;

    char *base_ptr = (char *)d.data.get();
    for (size_t symbol_idx = start_symbol; symbol_idx < num_symbols + start_symbol; ++symbol_idx)
    {
        for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
        {
            for (size_t prb_idx = start_prb; prb_idx < ret.prbs_per_symbol + start_prb; ++prb_idx)
            {
                ret.ptrs.at(antenna_idx).at(symbol_idx).at(prb_idx) =
                    (void *)(base_ptr + antenna_idx * ret.antenna_sz + (symbol_idx - start_symbol) * ret.symbol_sz + (prb_idx - start_prb) * prb_sz);
            }
        }
    }

#if 0
    re_cons("data_sz {}", data_sz);
    re_cons("antenna_sz {}", antenna_sz);
    re_cons("symbol_sz {}", symbol_sz);
    re_cons("prb_sz {}", prb_sz);
    re_cons("prbs_per_symbol {}", prbs_per_symbol);
    // re_info("ret.prbs_per_slot {}", ret.prbs_per_slot);
#endif

    ret.raw_data = std::move(d);
    return ret;
}

Slot buffer_to_slot(uint8_t * buffer, size_t buffer_size, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz)
{
    Slot ret(num_ante);

    // The sizes below assume your buffer pointer is __half2, i.e, QAM elements. If it's bytes you'd need to multiply by 4.
    ret.data_sz = buffer_size;
    if (ret.data_sz % num_symbols)
        do_throw(sb() << "Slot size " << ret.data_sz
             << " doesn't divide into the number of symbols "
             << num_symbols);

    ret.antenna_sz = ret.data_sz / num_ante;
    ret.symbol_sz = ret.antenna_sz / num_symbols;
    ret.prb_sz = prb_sz;
    if (ret.symbol_sz % ret.prb_sz)
        do_throw(sb() << "Symbol size " << ret.symbol_sz
             << " doesn't divide into the size of a PRB "
             << ret.prb_sz);

    ret.prbs_per_symbol = ret.symbol_sz / ret.prb_sz;

    if (ret.prbs_per_symbol > MAX_NUM_PRBS_PER_SYMBOL)
        do_throw(sb() << "Resulting number of PRBs per symbol "
             << ret.prbs_per_symbol << " is higher than MAX_NUM_PRBS_PER_SYMBOL. "
             << "Please recompile with a higher value for "
             << "MAX_NUM_PRBS_PER_SYMBOL.");

    ret.prbs_per_slot = ret.prbs_per_symbol * num_symbols * num_ante;

    char *base_ptr = (char *)buffer;
    for (size_t symbol_idx = start_symbol; symbol_idx < start_symbol + num_symbols; ++symbol_idx)
    {
        for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
        {
            for (size_t prb_idx = 0; prb_idx < ret.prbs_per_symbol; ++prb_idx)
            {
                ret.ptrs.at(antenna_idx).at(symbol_idx).at(prb_idx) =
                    (void *)(base_ptr + antenna_idx * ret.antenna_sz + (symbol_idx - start_symbol) * ret.symbol_sz // PDSCH QAMS only have data symbols so need to offset
                             + prb_idx * ret.prb_sz);
            }
        }
    }
#if 0
    re_info("ret.data_sz {}", ret.data_sz);
    re_info("ret.antenna_sz {}", ret.antenna_sz);
    re_info("ret.symbol_sz {}", ret.symbol_sz);
    re_info("ret.prb_sz {}", ret.prb_sz);
    re_info("ret.prbs_per_symbol {}", ret.prbs_per_symbol);
    re_info("ret.prbs_per_slot {}", ret.prbs_per_slot);
#endif
    return ret;
}

Dataset load_tv_datasets_single(hdf5hpp::hdf5_file& hdf5file, std::string const& file, std::string const dataset)
{
    Dataset d;
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(dataset.c_str());
    d.size = dset.get_buffer_size_bytes();
    re_dbg("Opened {}-byte dataset {}", d.size, dataset.c_str());
    auto pg_sz = sysconf(_SC_PAGESIZE);
    if(pg_sz == -1)
    {
        do_throw(sb() << "failed to get page size");
    }
    void * fh_mem = aerial_fh::allocate_memory(d.size, pg_sz);
    if(fh_mem == nullptr)
    {
        do_throw(sb() << "aerial_fh::allocate_memory failure ");
    }
    d.data.reset(memset(fh_mem, 0, d.size));
    if (d.data.get() == nullptr)
        do_throw(sb() << " malloc testvector data failed");
    re_dbg("Reading {}-byte dataset {}", d.size, dataset.c_str());
    dset.read(d.data.get());
    re_dbg("Read {}-byte dataset {}", d.size, dataset.c_str());
    return std::move(d);
}

//Use to num antennas for control channel TVs
int load_first_dimension_dataset(hdf5hpp::hdf5_file& hdf5file, std::string const& file, std::string dset_name)
{
    hdf5hpp::hdf5_dataset dset  = hdf5file.open_dataset(dset_name.c_str());
    return dset.get_dataspace().get_dimensions()[0];
}

int load_num_antenna_from_nr_tv(hdf5hpp::hdf5_file& hdf5file)
{
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset("X_tf_fp16");

    if(dset.get_dataspace().get_dimensions().size() < 3)
    {
        return 1;
    }
    return dset.get_dataspace().get_dimensions()[0];
}

int load_num_antenna_from_nr_tv_srs(hdf5hpp::hdf5_file& hdf5file)
{
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset("X_tf_srs_fp16");

    if(dset.get_dataspace().get_dimensions().size() < 3)
    {
        return 1;
    }
    return dset.get_dataspace().get_dimensions()[0];
}

int load_num_antenna_from_nr_tv_zp_csi_rs(hdf5hpp::hdf5_file& hdf5file)
{
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset("X_tf_fp16");
    bool pdsch_found = false;
    int count = 1;
    std::string pdu = "PDU";
    std::set<int> flows;
    while(1)
    {
        std::string dset_string = pdu + std::to_string(count);
        if(!hdf5file.is_valid_dataset(dset_string.c_str()))
        {
            break;
        }

        hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
        hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
        if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PDSCH)
        {
            count++;
            continue;
        }
        pdsch_found = true;

        uint8_t dmrsPorts = pdu_pars["dmrsPorts"].as<uint8_t>();
        uint8_t scid = pdu_pars["SCID"].as<uint8_t>();

        for(int flow_index = 0; flow_index < sizeof(dmrsPorts) * 8; ++flow_index)
        {
            if((dmrsPorts >> flow_index) & 0b1)
            {
                flows.emplace(scid * 8 + flow_index);
            }
        }
        count++;
    }
    return flows.size();
}

int load_num_antenna_from_nr_prach_tv(hdf5hpp::hdf5_file& hdf5file, std::string dset)
{
    hdf5hpp::hdf5_dataset ds = hdf5file.open_dataset(dset.c_str());

    if(ds.get_dataspace().get_dimensions().size() < 2)
    {
        return 1;
    }
    return ds.get_dataspace().get_dimensions()[0];
}

int load_ul_num_antenna_from_tv(hdf5hpp::hdf5_file& hdf5file, std::string const& file)
{
    hdf5hpp::hdf5_dataset dset_gnb  = hdf5file.open_dataset("gnb_pars");
    hdf5hpp::hdf5_dataset_elem gnb_pars = dset_gnb[0];
    return gnb_pars["nRx"].as<uint32_t>();
}

int load_ul_tb_size_from_tv(hdf5hpp::hdf5_file& hdf5file, std::string const& file)
{
    hdf5hpp::hdf5_dataset dset_gnb  = hdf5file.open_dataset("tb_pars");
    size_t tb_size = 0;
    for(int i = 0; i < dset_gnb.get_num_elements(); ++i)
    {
        hdf5hpp::hdf5_dataset_elem gnb_pars = dset_gnb[i];
        tb_size += gnb_pars["nTbByte"].as<uint32_t>();
    }
    return tb_size;
}

bool is_nr_tv(hdf5hpp::hdf5_file& hdf5file)
{
    return hdf5file.is_valid_dataset("PDU1");
}

bool merge_pdu_if_adjacent(std::vector<pdu_info>& existing_pdus, pdu_info& new_pdu)
{
    // Assume no adjacent pdus in existing_pdus
    // combined_pdu_infos is the merged PDUs

    // handle cases
    // existing vector is empty, add to vector
    // if new PDU is not adjacent to any exsiting PDU, add to vector
    // if new PDU is after another existing PDU, change existing PDU params
    // if new PDU is before another existing PDU, change existing PDU params
    // if it is both before and after, change the PDU params of lower PDU to merge all three, delete following PDU
    // only works with same startSym/numSym PDUs.

    // Do not merge if the flows are different
    auto adj_before = [&new_pdu] (const struct pdu_info& pdu) {
        return (pdu.startSym == new_pdu.startSym && pdu.numSym == new_pdu.numSym && (pdu.startPrb + pdu.numPrb) == new_pdu.startPrb && pdu.flow_indices == new_pdu.flow_indices);
    };

    auto adj_after = [&new_pdu] (const struct pdu_info& pdu) {
        return (pdu.startSym == new_pdu.startSym && pdu.numSym == new_pdu.numSym && pdu.startPrb == (new_pdu.startPrb + new_pdu.numPrb) && pdu.flow_indices == new_pdu.flow_indices);
    };

    bool found_before = false, found_after = false;
    int before_index = -1, after_index = -1;
    {
        auto iter = std::find_if(existing_pdus.begin(), existing_pdus.end(), adj_before);
        if(iter != existing_pdus.end())
        {
            found_before = true;
            before_index = iter - existing_pdus.begin();
        }
    }

    {
        auto iter = std::find_if(existing_pdus.begin(), existing_pdus.end(), adj_after);
        if(iter != existing_pdus.end())
        {
            found_after = true;
            after_index = iter - existing_pdus.begin();
        }
    }

    if(found_before && !found_after)
    {
        existing_pdus[before_index].numPrb += new_pdu.numPrb;
    }
    else if(!found_before && found_after)
    {
        existing_pdus[after_index].startPrb = new_pdu.startPrb;
        existing_pdus[after_index].numPrb += new_pdu.numPrb;

    }
    else if(found_before && found_after)
    {
        existing_pdus[before_index].numPrb += new_pdu.numPrb;
        existing_pdus[before_index].numPrb += existing_pdus[after_index].numPrb;
        existing_pdus.erase(existing_pdus.begin()+after_index);
    }
    else // !found_before && !found_after
    {
        existing_pdus.push_back(new_pdu);
    }
    return true;
}

bool merge_pdu_if_identical(std::vector<pdu_info>& existing_pdus, pdu_info& new_pdu)
{
    auto identical = [&new_pdu] (const struct pdu_info& pdu) {
        return (pdu.startSym == new_pdu.startSym && pdu.numSym == new_pdu.numSym && pdu.startPrb == new_pdu.startPrb && pdu.numPrb == new_pdu.numPrb && pdu.flow_indices == new_pdu.flow_indices);
    };

    bool found = false;
    int found_index = -1;

    auto iter = std::find_if(existing_pdus.begin(), existing_pdus.end(), identical);
    if(iter != existing_pdus.end())
    {
        found = true;
        found_index = iter - existing_pdus.begin();
    }

    if(found)
    {
        return true;
    }
    else
    {
        return false;
    }
}
