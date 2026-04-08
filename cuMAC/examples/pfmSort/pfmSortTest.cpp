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

#include "pfmSortTest.h"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_CUMAC)

 pfm_data_manage_t::pfm_data_manage_t(const std::string& configFilePath, cudaStream_t strm) 
 : m_strm(strm) {

    int ret = loadConfigYaml(configFilePath);
    if (ret != 0) {
        throw std::runtime_error("Failed to load configuration from YAML file " + configFilePath);
    }

    ue_list.resize(m_num_cell);
    available_rnti.resize(m_num_cell);
    available_id.resize(m_num_cell);

    pfmSortTask = std::make_unique<cumac::pfmSortTask>();

    pfmSortTask->num_cell = m_num_cell;
    CUDA_CHECK_ERR(cudaMalloc(&pfmSortTask->gpu_buf, sizeof(cumac_pfm_cell_info_t)*m_num_cell));
    pfmSortTask->strm = m_strm;

    pfm_cell_info.resize(m_num_cell);
    std::memset(pfm_cell_info.data(), 0, sizeof(cumac_pfm_cell_info_t) * m_num_cell);

    pfm_output_cell_info_gpu.resize(CUMAC_PFM_MAX_NUM_CELL);
    std::memset(pfm_output_cell_info_gpu.data(), 0, sizeof(cumac_pfm_output_cell_info_t) * CUMAC_PFM_MAX_NUM_CELL);

    pfm_output_cell_info_cpu.resize(CUMAC_PFM_MAX_NUM_CELL);
    std::memset(pfm_output_cell_info_cpu.data(), 0, sizeof(cumac_pfm_output_cell_info_t) * CUMAC_PFM_MAX_NUM_CELL);

    cell_data_main_cpu.resize(CUMAC_PFM_MAX_NUM_CELL);
    std::memset(cell_data_main_cpu.data(), 0, sizeof(cumac::PFM_CELL_INFO_MANAGE) * CUMAC_PFM_MAX_NUM_CELL);

    num_lc_per_qos_type.resize(m_num_cell);

    // initialize the available RNTIs and cuMAC 0-based IDs
    for (uint16_t cell_id = 0; cell_id < m_num_cell; cell_id++) {
        num_lc_per_qos_type[cell_id].resize(CUMAC_PFM_NUM_QOS_TYPES_DL + CUMAC_PFM_NUM_QOS_TYPES_UL);

        for (uint16_t rnti = MIN_RNTI; rnti < (CUMAC_PFM_MAX_NUM_UE_PER_CELL+MIN_RNTI); rnti++) {
            available_rnti[cell_id].push_back(rnti);
        }
        // initialize the available cuMAC 0-based IDs
        for (uint16_t i = 0; i < CUMAC_PFM_MAX_NUM_UE_PER_CELL; i++) {
            available_id[cell_id].push_back(i);
        }
        // initialize the UE list for PFM sorting
        for (uint16_t i = 0; i < m_num_ue_per_cell; i++) {
            uint16_t id = available_id[cell_id].front();
            available_id[cell_id].pop_front();
            uint16_t rnti = available_rnti[cell_id].front();
            available_rnti[cell_id].pop_front();
            ue_list[cell_id].push_back(ue_id_rnti_t(id, rnti));
        }
    }
}

pfm_data_manage_t::~pfm_data_manage_t() {
    CUDA_CHECK_ERR(cudaFree(pfmSortTask->gpu_buf));
}

int pfm_data_manage_t::loadConfigYaml(const std::string& configFilePath) {
    try
    {
        // Load YAML file using yaml-cpp
        YAML::Node root = YAML::LoadFile(configFilePath);

        // Load parameters from YAML
        m_num_slot = root["NUM_SLOT"].as<uint32_t>();
        m_seed = root["SEED"].as<unsigned>();
        m_num_cell = root["NUM_CELL"].as<uint16_t>();
        m_num_ue_per_cell = root["NUM_UE_PER_CELL"].as<uint16_t>();
        m_num_dl_lc_per_ue = root["NUM_DL_LC_PER_UE"].as<uint16_t>();
        m_num_ul_lcg_per_ue = root["NUM_UL_LCG_PER_UE"].as<uint16_t>();
        m_max_num_h5_tv_created = root["MAX_NUM_H5_TV_CREATED"].as<uint16_t>();

        // validate parameters
        if (m_num_cell > CUMAC_PFM_MAX_NUM_CELL) {
            printf("ERROR: NUM_CELL is greater than %d\n", CUMAC_PFM_MAX_NUM_CELL);
            return -1;
        }
        if (m_num_ue_per_cell > CUMAC_PFM_MAX_NUM_UE_PER_CELL) {
            printf("ERROR: NUM_UE_PER_CELL is greater than %d\n", CUMAC_PFM_MAX_NUM_UE_PER_CELL);
            return -1;
        }
        if (m_num_dl_lc_per_ue > CUMAC_PFM_MAX_NUM_LC_PER_UE) {
            printf("ERROR: NUM_DL_LC_PER_UE is greater than %d\n", CUMAC_PFM_MAX_NUM_LC_PER_UE);
            return -1;
        }
        if (m_num_ul_lcg_per_ue > CUMAC_PFM_MAX_NUM_LCG_PER_UE) {
            printf("ERROR: NUM_UL_LCG_PER_UE is greater than %d\n", CUMAC_PFM_MAX_NUM_LCG_PER_UE);
            return -1;
        }

        printf("NUM_CELL: %d\n", m_num_cell);
        printf("NUM_UE_PER_CELL: %d\n", m_num_ue_per_cell);
        printf("NUM_DL_LC_PER_UE: %d\n", m_num_dl_lc_per_ue);
        printf("NUM_UL_LCG_PER_UE: %d\n", m_num_ul_lcg_per_ue);
    }
    catch (const YAML::Exception& e) {
        printf("Error loading configuration from YAML file %s: %s\n", configFilePath.c_str(), e.what());
        return -1;
    }

    return 0;
}

bool pfm_data_manage_t::add_pfm_ue(const uint16_t cell_id, const uint16_t num_new_ue) { // add new arrived UEs and assign cuMAC 0-based IDs and RNTIs
    for (int i = 0; i < num_new_ue; i++) {
        if (available_id[cell_id].empty() || available_rnti[cell_id].empty()) {
            printf("No available ID or RNTI for new UE of PFM sorting in cell %d\n", cell_id);
            return false;
        }

        if (ue_list[cell_id].size() >= CUMAC_PFM_MAX_NUM_UE_PER_CELL) {
            printf("Maximum number of UEs for PFM sorting reached in cell %d\n", cell_id);
            return false;
        }

        uint16_t id = available_id[cell_id].front();
        available_id[cell_id].pop_front();
        uint16_t rnti = available_rnti[cell_id].front();
        available_rnti[cell_id].pop_front();
        ue_list[cell_id].push_back(ue_id_rnti_t(id, rnti));
    }

    return true;
}

bool pfm_data_manage_t::remove_pfm_ue(const uint16_t cell_id, uint16_t rnti) { // remove UE from the UE list for PFM sorting with a given RNTI
    // validate RNTI value
    if (rnti < MIN_RNTI || rnti > MAX_RNTI) {
        printf("RNTI %d is out of range\n", rnti);
        return false;
    }

    auto it = std::find_if(ue_list[cell_id].begin(), ue_list[cell_id].end(), [rnti](const ue_id_rnti_t& p) {
        return p.rnti == rnti;
    });

    if (it != ue_list[cell_id].end()) {
        available_id[cell_id].push_front(it->id);
        available_rnti[cell_id].push_front(it->rnti);
        ue_list[cell_id].erase(it);

        return true;
    } else {
        printf("UE with RNTI %d is not found in the UE list for PFM sorting of cell %d\n", rnti, cell_id);
        return false;
    }
}

uint16_t pfm_data_manage_t::get_id(const uint16_t cell_id, uint16_t rnti) { // get the cuMAC 0-based ID of a UE with a given RNTI
    // validate RNTI value
    if (rnti < MIN_RNTI || rnti > MAX_RNTI) {
        printf("RNTI %d is out of range\n", rnti);
        return 0xFFFF;
    }

    auto it = std::find_if(ue_list[cell_id].begin(), ue_list[cell_id].end(), [rnti](const ue_id_rnti_t& p) {
        return p.rnti == rnti;
    });

    if (it != ue_list[cell_id].end()) {    
        return it->id;
    } else {
        printf("UE with RNTI %d is not found in the UE list for PFM sorting of cell %d\n", rnti, cell_id);
        return 0xFFFF;
    }
}

uint16_t pfm_data_manage_t::get_ue_idx_in_list(const uint16_t cell_id, uint16_t rnti) { // get the index of a UE in the UE list for PFM sorting with a given RNTI
    auto it = std::find_if(ue_list[cell_id].begin(), ue_list[cell_id].end(), [rnti](const ue_id_rnti_t& p) {
        return p.rnti == rnti;
    });

    if (it != ue_list[cell_id].end()) {
        return std::distance(ue_list[cell_id].begin(), it);
    } else {
        printf("UE with RNTI %d is not found in the UE list for PFM sorting of cell %d\n", rnti, cell_id);
        return 0xFFFF;
    }
}

ue_id_rnti_t pfm_data_manage_t::get_ue_id_rnti(const uint16_t cell_id, uint16_t ue_idx) { // get the cuMAC 0-based ID and RNTI of a UE
    if (ue_idx >= ue_list[cell_id].size()) {
        printf("UE index %d is out of range in the UE list for PFM sorting of cell %d. Current number of UEs for PFM sorting: %d\n", ue_idx, cell_id, ue_list[cell_id].size());
        return ue_id_rnti_t(0xFFFF, 0xFFFF);
    }
    return ue_list[cell_id][ue_idx];
}

void pfm_data_manage_t::prepare_pfm_data()
{
    // initialize random number generator
    std::mt19937 gen(m_seed);
    std::uniform_int_distribution<> distrib(0, INT_MAX);

    for (int cIdx = 0; cIdx < m_num_cell; cIdx++) {
        for (int idx = 0; idx < CUMAC_PFM_NUM_QOS_TYPES_DL; idx++) {
            pfm_cell_info[cIdx].num_output_sorted_lc[idx] = m_num_ue_per_cell*m_num_dl_lc_per_ue;
        }
        for (int idx = 0; idx < CUMAC_PFM_NUM_QOS_TYPES_UL; idx++) {
            pfm_cell_info[cIdx].num_output_sorted_lc[idx + CUMAC_PFM_NUM_QOS_TYPES_DL] = m_num_ue_per_cell*m_num_ul_lcg_per_ue;
        }

        pfm_cell_info[cIdx].num_ue = ue_list[cIdx].size();
        for (int uIdx = 0; uIdx < ue_list[cIdx].size(); uIdx++) {
            pfm_cell_info[cIdx].ue_info[uIdx].rcurrent_dl = 1000000 + distrib(gen) % 20000000;
            pfm_cell_info[cIdx].ue_info[uIdx].rcurrent_ul = 500000 + distrib(gen) % 10000000;
            pfm_cell_info[cIdx].ue_info[uIdx].rnti = ue_list[cIdx][uIdx].rnti;
            pfm_cell_info[cIdx].ue_info[uIdx].id = ue_list[cIdx][uIdx].id;
            pfm_cell_info[cIdx].ue_info[uIdx].num_layers_dl = distrib(gen) % 2 + 1;
            pfm_cell_info[cIdx].ue_info[uIdx].num_layers_ul = distrib(gen) % 2 + 1;
            pfm_cell_info[cIdx].ue_info[uIdx].flags = 0x03;
            for (int lcIdx = 0; lcIdx < m_num_dl_lc_per_ue; lcIdx++) {
                pfm_cell_info[cIdx].ue_info[uIdx].dl_lc_info[lcIdx].tbs_scheduled = static_cast<uint32_t>((1000000 + distrib(gen) % 20000000)*CUMAC_PFM_SLOT_DURATION);
                if (m_slot_idx == 0) {
                    uint8_t qos_type = distrib(gen) % 5;
                    pfm_cell_info[cIdx].ue_info[uIdx].dl_lc_info[lcIdx].qos_type = qos_type;
                    num_lc_per_qos_type[cIdx][qos_type]++;
                    pfm_cell_info[cIdx].ue_info[uIdx].dl_lc_info[lcIdx].flags = 0x03;
                } else {
                    pfm_cell_info[cIdx].ue_info[uIdx].dl_lc_info[lcIdx].flags = 0x01;
                }
            }

            for (int lcgIdx = 0; lcgIdx < m_num_ul_lcg_per_ue; lcgIdx++) {
                pfm_cell_info[cIdx].ue_info[uIdx].ul_lcg_info[lcgIdx].tbs_scheduled = static_cast<uint32_t>((500000 + distrib(gen) % 10000000)*CUMAC_PFM_SLOT_DURATION);
                if (m_slot_idx == 0) {
                    uint8_t qos_type = distrib(gen) % 5;
                    pfm_cell_info[cIdx].ue_info[uIdx].ul_lcg_info[lcgIdx].qos_type = qos_type;
                    num_lc_per_qos_type[cIdx][qos_type + CUMAC_PFM_NUM_QOS_TYPES_DL]++;
                    pfm_cell_info[cIdx].ue_info[uIdx].ul_lcg_info[lcgIdx].flags = 0x03;
                } else {
                    pfm_cell_info[cIdx].ue_info[uIdx].ul_lcg_info[lcgIdx].flags = 0x01;
                }
            }
        }
    }

    CUDA_CHECK_ERR(cudaMemcpy(pfmSortTask->gpu_buf, pfm_cell_info.data(), sizeof(cumac_pfm_cell_info_t)*m_num_cell, cudaMemcpyHostToDevice));
}

void pfm_data_manage_t::cpu_pfm_sort()
{
    // for debugging purpose
    // std::copy(std::begin(pfm_output_cell_info_gpu), std::end(pfm_output_cell_info_gpu), std::begin(pfm_output_cell_info_cpu));
    
    for (int cIdx = 0; cIdx < m_num_cell; cIdx++) {
        for (int qosType = 0; qosType < CUMAC_PFM_NUM_QOS_TYPES_DL; qosType++) {
            std::vector<pfmTuple<float>> pfmList;

            for (int ueLcIdx = 0; ueLcIdx < pfm_cell_info[cIdx].num_ue*CUMAC_PFM_MAX_NUM_LC_PER_UE; ueLcIdx++) {
                uint32_t ue_idx = ueLcIdx / CUMAC_PFM_MAX_NUM_LC_PER_UE;
                uint32_t lc_idx = ueLcIdx - ue_idx*CUMAC_PFM_MAX_NUM_LC_PER_UE;

                uint32_t id_in_main_buff = pfm_cell_info[cIdx].ue_info[ue_idx].id;

                if (((pfm_cell_info[cIdx].ue_info[ue_idx].flags & 0x01) > 0) && 
                    ((pfm_cell_info[cIdx].ue_info[ue_idx].dl_lc_info[lc_idx].flags & 0x01) > 0) && 
                    (pfm_cell_info[cIdx].ue_info[ue_idx].dl_lc_info[lc_idx].qos_type == qosType)) {
                    
                    pfmTuple<float> pfmTupleTemp;
                    std::get<1>(pfmTupleTemp) = pfm_cell_info[cIdx].ue_info[ue_idx].rnti;
                    std::get<2>(pfmTupleTemp) = lc_idx;

                    float l_temp_ravg;
                    if ((pfm_cell_info[cIdx].ue_info[ue_idx].dl_lc_info[lc_idx].flags & 0x02) > 0) {
                        l_temp_ravg = 1.0;
                    } else {
                        l_temp_ravg = static_cast<float>(cell_data_main_cpu[cIdx].ue_info_manage[id_in_main_buff].ravg_dl_lc[lc_idx]) * (1 - CUMAC_PFM_IIR_ALPHA) +
                                      CUMAC_PFM_IIR_ALPHA * static_cast<float>(pfm_cell_info[cIdx].ue_info[ue_idx].dl_lc_info[lc_idx].tbs_scheduled) /
                                      pfm_cell_info[cIdx].ue_info[ue_idx].num_layers_dl / CUMAC_PFM_SLOT_DURATION;
                    }

                    cell_data_main_cpu[cIdx].ue_info_manage[id_in_main_buff].ravg_dl_lc[lc_idx] = static_cast<uint32_t>(l_temp_ravg);  
                        
                    std::get<0>(pfmTupleTemp) = pfm_cell_info[cIdx].ue_info[ue_idx].rcurrent_dl / l_temp_ravg;   
                    pfmList.push_back(pfmTupleTemp);
                }
            }

            std::sort(pfmList.begin(), pfmList.end(), [](pfmTuple<float> a, pfmTuple<float> b)
                                  {
                                      return (std::get<0>(a) > std::get<0>(b)) || 
                                             (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b)) ||
                                             (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) == std::get<1>(b) && std::get<2>(a) < std::get<2>(b));
                                  });

            switch (qosType) {
                case 0:
                    for (int lcIdx = 0; lcIdx < pfm_cell_info[cIdx].num_output_sorted_lc[0]; lcIdx++) {
                        if (lcIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[lcIdx].rnti = std::get<1>(pfmList[lcIdx]);
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[lcIdx].lc_id = std::get<2>(pfmList[lcIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[lcIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[lcIdx].lc_id = 0;
                        }
                    }
                    break;
                case 1:
                    for (int lcIdx = 0; lcIdx < pfm_cell_info[cIdx].num_output_sorted_lc[1]; lcIdx++) {
                        if (lcIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[lcIdx].rnti = std::get<1>(pfmList[lcIdx]);
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[lcIdx].lc_id = std::get<2>(pfmList[lcIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[lcIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[lcIdx].lc_id = 0;
                        }
                    }
                    break;
                case 2:
                    for (int lcIdx = 0; lcIdx < pfm_cell_info[cIdx].num_output_sorted_lc[2]; lcIdx++) {
                        if (lcIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[lcIdx].rnti = std::get<1>(pfmList[lcIdx]);
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[lcIdx].lc_id = std::get<2>(pfmList[lcIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[lcIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[lcIdx].lc_id = 0;
                        }
                    }
                    break;
                case 3:
                    for (int lcIdx = 0; lcIdx < pfm_cell_info[cIdx].num_output_sorted_lc[3]; lcIdx++) {
                        if (lcIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[lcIdx].rnti = std::get<1>(pfmList[lcIdx]);
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[lcIdx].lc_id = std::get<2>(pfmList[lcIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[lcIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[lcIdx].lc_id = 0;
                        }
                    }
                    break;
                case 4:
                    for (int lcIdx = 0; lcIdx < pfm_cell_info[cIdx].num_output_sorted_lc[4]; lcIdx++) {
                        if (lcIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[lcIdx].rnti = std::get<1>(pfmList[lcIdx]);
                            pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[lcIdx].lc_id = std::get<2>(pfmList[lcIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[lcIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[lcIdx].lc_id = 0;
                        }
                    }
                    break;
                default:
                    break;
            }
        }

        for (int qosType = 0; qosType < CUMAC_PFM_NUM_QOS_TYPES_UL; qosType++) {
            std::vector<pfmTuple<float>> pfmList;

            for (int ueLcIdx = 0; ueLcIdx < pfm_cell_info[cIdx].num_ue*CUMAC_PFM_MAX_NUM_LCG_PER_UE; ueLcIdx++) {
                uint32_t ue_idx = ueLcIdx / CUMAC_PFM_MAX_NUM_LCG_PER_UE;
                uint32_t lcg_idx = ueLcIdx - ue_idx*CUMAC_PFM_MAX_NUM_LCG_PER_UE;

                uint32_t id_in_main_buff = pfm_cell_info[cIdx].ue_info[ue_idx].id;

                if (((pfm_cell_info[cIdx].ue_info[ue_idx].flags & 0x02) > 0) && 
                    ((pfm_cell_info[cIdx].ue_info[ue_idx].ul_lcg_info[lcg_idx].flags & 0x01) > 0) && 
                    (pfm_cell_info[cIdx].ue_info[ue_idx].ul_lcg_info[lcg_idx].qos_type == qosType)) {
                    
                    pfmTuple<float> pfmTupleTemp;
                    std::get<1>(pfmTupleTemp) = pfm_cell_info[cIdx].ue_info[ue_idx].rnti;
                    std::get<2>(pfmTupleTemp) = lcg_idx;
                    
                    float l_temp_ravg;
                    if ((pfm_cell_info[cIdx].ue_info[ue_idx].ul_lcg_info[lcg_idx].flags & 0x02) > 0) {
                        l_temp_ravg = 1.0;
                    } else {
                        l_temp_ravg = static_cast<float>(cell_data_main_cpu[cIdx].ue_info_manage[id_in_main_buff].ravg_ul_lcg[lcg_idx]) * (1 - CUMAC_PFM_IIR_ALPHA) +
                                      CUMAC_PFM_IIR_ALPHA * static_cast<float>(pfm_cell_info[cIdx].ue_info[ue_idx].ul_lcg_info[lcg_idx].tbs_scheduled) /
                                      pfm_cell_info[cIdx].ue_info[ue_idx].num_layers_ul / CUMAC_PFM_SLOT_DURATION;
                    }

                    cell_data_main_cpu[cIdx].ue_info_manage[id_in_main_buff].ravg_ul_lcg[lcg_idx] = static_cast<uint32_t>(l_temp_ravg);  
                        
                    std::get<0>(pfmTupleTemp) = pfm_cell_info[cIdx].ue_info[ue_idx].rcurrent_ul / l_temp_ravg;    
                    pfmList.push_back(pfmTupleTemp);
                }
            }

            std::sort(pfmList.begin(), pfmList.end(), [](pfmTuple<float> a, pfmTuple<float> b)
                                  {
                                      return (std::get<0>(a) > std::get<0>(b)) || 
                                             (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b)) ||
                                             (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) == std::get<1>(b) && std::get<2>(a) < std::get<2>(b));
                                  });

            switch (qosType) {
                case 0:
                    for (int lcgIdx = 0; lcgIdx < pfm_cell_info[cIdx].num_output_sorted_lc[CUMAC_PFM_NUM_QOS_TYPES_DL + qosType]; lcgIdx++) {
                        if (lcgIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[lcgIdx].rnti = std::get<1>(pfmList[lcgIdx]);
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[lcgIdx].lcg_id = std::get<2>(pfmList[lcgIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[lcgIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[lcgIdx].lcg_id = 0;
                        }
                    }
                    break;
                case 1:
                    for (int lcgIdx = 0; lcgIdx < pfm_cell_info[cIdx].num_output_sorted_lc[CUMAC_PFM_NUM_QOS_TYPES_DL + qosType]; lcgIdx++) {
                        if (lcgIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[lcgIdx].rnti = std::get<1>(pfmList[lcgIdx]);
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[lcgIdx].lcg_id = std::get<2>(pfmList[lcgIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[lcgIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[lcgIdx].lcg_id = 0;
                        }
                        pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[lcgIdx].lcg_id = std::get<2>(pfmList[lcgIdx]);
                    }
                    break;
                case 2:
                    for (int lcgIdx = 0; lcgIdx < pfm_cell_info[cIdx].num_output_sorted_lc[CUMAC_PFM_NUM_QOS_TYPES_DL + qosType]; lcgIdx++) {
                        if (lcgIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[lcgIdx].rnti = std::get<1>(pfmList[lcgIdx]);
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[lcgIdx].lcg_id = std::get<2>(pfmList[lcgIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[lcgIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[lcgIdx].lcg_id = 0;
                        }
                    }
                    break;
                case 3:
                    for (int lcgIdx = 0; lcgIdx < pfm_cell_info[cIdx].num_output_sorted_lc[CUMAC_PFM_NUM_QOS_TYPES_DL + qosType]; lcgIdx++) {
                        if (lcgIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[lcgIdx].rnti = std::get<1>(pfmList[lcgIdx]);
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[lcgIdx].lcg_id = std::get<2>(pfmList[lcgIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[lcgIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[lcgIdx].lcg_id = 0;
                        }
                    }
                    break;
                case 4:
                    for (int lcgIdx = 0; lcgIdx < pfm_cell_info[cIdx].num_output_sorted_lc[CUMAC_PFM_NUM_QOS_TYPES_DL + qosType]; lcgIdx++) {
                        if (lcgIdx < pfmList.size()) {
                            pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[lcgIdx].rnti = std::get<1>(pfmList[lcgIdx]);
                            pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[lcgIdx].lcg_id = std::get<2>(pfmList[lcgIdx]);
                        } else {
                            pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[lcgIdx].rnti = CUMAC_INVALID_RNTI;
                            pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[lcgIdx].lcg_id = 0;
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }
}

bool pfm_data_manage_t::validate_pfm_output()
{
    for (int cIdx = 0; cIdx < m_num_cell; cIdx++) {
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][0]; idx++) {            
            if (pfm_output_cell_info_gpu[cIdx].dl_gbr_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].dl_gbr_critical[idx].lc_id != pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[idx].lc_id) {
                printf("ERROR: PFM sorting output validation failed for DL GBR critical LC %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_gpu[cIdx].dl_gbr_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].dl_gbr_critical[idx].lc_id);
                printf("CPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].dl_gbr_critical[idx].lc_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][1]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].dl_gbr_non_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].dl_gbr_non_critical[idx].lc_id != pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[idx].lc_id) {
                printf("ERROR: PFM sorting output validation failed for DL GBR non-critical LC %d in cell %d\n", idx, cIdx);
                
                printf("GPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_gpu[cIdx].dl_gbr_non_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].dl_gbr_non_critical[idx].lc_id);
                printf("CPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].dl_gbr_non_critical[idx].lc_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][2]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].dl_ngbr_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].dl_ngbr_critical[idx].lc_id != pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[idx].lc_id) {
                printf("ERROR: PFM sorting output validation failed for DL NGBR critical LC %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_gpu[cIdx].dl_ngbr_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].dl_ngbr_critical[idx].lc_id);
                printf("CPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].dl_ngbr_critical[idx].lc_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][3]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].dl_ngbr_non_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].dl_ngbr_non_critical[idx].lc_id != pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[idx].lc_id) {
                printf("ERROR: PFM sorting output validation failed for DL NGBR non-critical LC %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_gpu[cIdx].dl_ngbr_non_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].dl_ngbr_non_critical[idx].lc_id);
                printf("CPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].dl_ngbr_non_critical[idx].lc_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][4]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].dl_mbr_non_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].dl_mbr_non_critical[idx].lc_id != pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[idx].lc_id) {
                printf("ERROR: PFM sorting output validation failed for DL MBR non-critical LC %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_gpu[cIdx].dl_mbr_non_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].dl_mbr_non_critical[idx].lc_id);
                printf("CPU output: RNTI %d, LC ID %d\n", pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].dl_mbr_non_critical[idx].lc_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][5]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].ul_gbr_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].ul_gbr_critical[idx].lcg_id != pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting output validation failed for UL GBR critical LCG %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_gpu[cIdx].ul_gbr_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].ul_gbr_critical[idx].lcg_id);
                printf("CPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].ul_gbr_critical[idx].lcg_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][6]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].ul_gbr_non_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].ul_gbr_non_critical[idx].lcg_id != pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting output validation failed for UL GBR non-critical LCG %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_gpu[cIdx].ul_gbr_non_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].ul_gbr_non_critical[idx].lcg_id);
                printf("CPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].ul_gbr_non_critical[idx].lcg_id);
                return false;
            }
        }

        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][7]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].ul_ngbr_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].ul_ngbr_critical[idx].lcg_id != pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting output validation failed for UL NGBR critical LCG %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_gpu[cIdx].ul_ngbr_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].ul_ngbr_critical[idx].lcg_id);
                printf("CPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].ul_ngbr_critical[idx].lcg_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][8]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].ul_ngbr_non_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].ul_ngbr_non_critical[idx].lcg_id != pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting output validation failed for UL NGBR non-critical LCG %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_gpu[cIdx].ul_ngbr_non_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].ul_ngbr_non_critical[idx].lcg_id);
                printf("CPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].ul_ngbr_non_critical[idx].lcg_id);
                return false;
            }
        }
        
        for (int idx = 0; idx < num_lc_per_qos_type[cIdx][9]; idx++) {
            if (pfm_output_cell_info_gpu[cIdx].ul_mbr_non_critical[idx].rnti != pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[idx].rnti ||
                pfm_output_cell_info_gpu[cIdx].ul_mbr_non_critical[idx].lcg_id != pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting output validation failed for UL MBR non-critical LCG %d in cell %d\n", idx, cIdx);

                printf("GPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_gpu[cIdx].ul_mbr_non_critical[idx].rnti, pfm_output_cell_info_gpu[cIdx].ul_mbr_non_critical[idx].lcg_id);
                printf("CPU output: RNTI %d, LCG ID %d\n", pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[idx].rnti, pfm_output_cell_info_cpu[cIdx].ul_mbr_non_critical[idx].lcg_id);
                return false;
            }
        }
    }
    return true;
}

std::string pfm_data_manage_t::pfm_save_tv_H5()
{
    try {

        std::string saveTvName = "PFM_SORT_TV_" + std::to_string(m_num_cell) +"CELLS_SLOT_" + std::to_string(m_slot_idx) + ".h5";

        // Create HDF5 file
        H5::H5File file(saveTvName, H5F_ACC_TRUNC);

        for (int cIdx = 0; cIdx < m_num_cell; cIdx++) {
            // Save input cell info
            {
                hsize_t dims[1] = {sizeof(cumac_pfm_cell_info_t)};
                H5::DataSpace dataspace(1, dims);
                std::string datasetName = "INPUT_CELL_INFO_" + std::to_string(cIdx);

                H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_UINT8, dataspace);
                dataset.write(reinterpret_cast<const uint8_t*>(&pfm_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            }

            // Save output cell info
            {
                hsize_t dims[1] = {sizeof(cumac_pfm_output_cell_info_t)};
                H5::DataSpace dataspace(1, dims);
                std::string datasetName = "OUTPUT_CELL_INFO_" + std::to_string(cIdx);

                H5::DataSet dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_UINT8, dataspace);
                dataset.write(reinterpret_cast<const uint8_t*>(&pfm_output_cell_info_gpu[cIdx]), H5::PredType::NATIVE_UINT8);
            }
        }

        std::cout << "Data written to " << saveTvName << std::endl;

        return saveTvName;

    } catch (const H5::FileIException &e) {
        e.printErrorStack();
        return "";
    } catch (const H5::DataSetIException &e) {
        e.printErrorStack();
        return "";
    } catch (const H5::DataSpaceIException &e) {
        e.printErrorStack();
        return "";
    }
}


template <typename T>
int compare_array(const char *info, T *tv_buf, T *cumac_buf, uint32_t num)
{
    int check_result = 0;
    if (tv_buf == nullptr || cumac_buf == nullptr)
    {
        NVLOGW_FMT(TAG, "ERROR: PFM sorting - pointer is null: tv_buf=0x{} cumac_buf=0x{}", (void*)tv_buf, (void*)cumac_buf);
        return -1;
    }

    size_t size = sizeof(T) * num;

    if (memcmp(tv_buf, cumac_buf, size))
    {
        char info_str[64];
        T *v1 = reinterpret_cast<T *>(tv_buf);
        T *v2 = reinterpret_cast<T *>(cumac_buf);

        uint32_t i;
        for (i = 0; i < num; i++)
        {
            if (*(v1 + i) != *(v2 + i))
            {
                break;
            }
        }
        v1 += i;
        v2 += i;

        snprintf(info_str, 64, "ARRAY %s DIFF from %u TV", info, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v1, num - i);

        snprintf(info_str, 64, "ARRAY %s DIFF from %u CUMAC", info, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v2, num - i);

        check_result = -1;
    }

    if (check_result == 0) // SAME
    {
        NVLOGC_FMT(TAG, "ARRAY {} SAME", info);
    }

    return check_result;
}

int pfm_data_manage_t::unit_test(const uint16_t num_cell, const uint16_t num_slot) {
    int ret = 0;

    std::unique_ptr<cumac::pfmSortTask> pfmSortTask = std::make_unique<cumac::pfmSortTask>();
    pfmSortTask->num_cell = num_cell;
    pfmSortTask->strm = m_strm;
    CUDA_CHECK_ERR(cudaMalloc(&pfmSortTask->gpu_buf, sizeof(cumac_pfm_cell_info_t)*num_cell));

    uint8_t* output_host_buf = nullptr;
    CUDA_CHECK_ERR(cudaMallocHost(&output_host_buf, sizeof(cumac_pfm_output_cell_info_t)*num_cell));

    // create PFM sorting object
    std::vector<cumac::pfmSort*> pfmSortVec(m_max_num_h5_tv_created);
    for (uint16_t i = 0; i < m_max_num_h5_tv_created; i++) {
        pfmSortVec[i] = new cumac::pfmSort();
    }

    // Loop through each slot
    for (uint16_t slot_id = 0; slot_id < num_slot; slot_id++) {
        // TODO: use different PFM sorting object for each slot
        // cumac::pfmSort* pfmSort = pfmSortVec[slot_id % m_max_num_h5_tv_created];
        cumac::pfmSort* pfmSort = pfmSortVec[0]; // Use the first PFM sorting object for all slots
        uint16_t tv_id = slot_id % m_max_num_h5_tv_created;
        std::string tv_name = "PFM_SORT_TV_" + std::to_string(num_cell) + "CELLS_SLOT_" + std::to_string(tv_id) + ".h5";
        std::vector<cumac_pfm_cell_info_t> input_cell_info(num_cell);
        std::vector<cumac_pfm_output_cell_info_t> output_cell_info(num_cell);

        if (!pfm_load_tv_H5(tv_name, input_cell_info, output_cell_info)) {
            printf("Failed to load TV file %s\n", tv_name.c_str());
            ret = -1;
            break;
        }
        // printf("Loaded TV file %s\n", tv_name.c_str());

        CUDA_CHECK_ERR(cudaMemcpyAsync(pfmSortTask->gpu_buf, reinterpret_cast<uint8_t*>(input_cell_info.data()), sizeof(cumac_pfm_cell_info_t)*num_cell, cudaMemcpyHostToDevice, m_strm));
        CUDA_CHECK_ERR(cudaStreamSynchronize(m_strm));
        pfmSort->setup(pfmSortTask.get());
        pfmSort->run(output_host_buf);
        CUDA_CHECK_ERR(cudaStreamSynchronize(m_strm));

        ret = compare_array("OUTPUT_CELL_INFO", reinterpret_cast<uint8_t*>(output_cell_info.data()), output_host_buf, num_cell * sizeof(cumac_pfm_output_cell_info_t));
        printf("Slot %d: unit test with TV %s result: %d\n", slot_id, tv_name.c_str(), ret);
        // NVLOGC_FMT(TAG, "Slot %d: unit test with TV %s result: %s", slot_id, tv_name, ret == 0 ? "PASSED" : "FAILED");
    }

    CUDA_CHECK_ERR(cudaFree(pfmSortTask->gpu_buf));
    CUDA_CHECK_ERR(cudaFreeHost(output_host_buf));

    return ret;
}

bool pfm_data_manage_t::pfm_load_tv_H5(const std::string& tv_name, std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info)
{
    printf("Loading PFM sorting TV file %s\n", tv_name.c_str());
    try {
        H5::H5File file(tv_name, H5F_ACC_RDONLY);

        int num_cell = pfm_cell_info.size();

        if (num_cell != pfm_output_cell_info.size()) {
            printf("ERROR: PFM sorting - number of cells in the output cell info array and the input cell info array are different: %d vs %d\n", num_cell, pfm_output_cell_info.size());
            return false;
        }

        for (int cIdx = 0; cIdx < num_cell; cIdx++) {
            std::string datasetName = "INPUT_CELL_INFO_" + std::to_string(cIdx);

            // see if the dataset exists
            if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0) {
                H5::DataSet dataset = file.openDataSet(datasetName);
                dataset.read(reinterpret_cast<uint8_t*>(&pfm_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            } else {
                printf("ERROR: PFM sorting TV file %s does not contain input cell info for cell %d\n", tv_name.c_str(), cIdx);
                return false;
            }
        }

        for (int cIdx = 0; cIdx < num_cell; cIdx++) {
            std::string datasetName = "OUTPUT_CELL_INFO_" + std::to_string(cIdx);

            // see if the dataset exists
            if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0) {
                H5::DataSet dataset = file.openDataSet(datasetName);
                dataset.read(reinterpret_cast<uint8_t*>(&pfm_output_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            } else {
                printf("ERROR: PFM sorting TV file %s does not contain output cell info for cell %d\n", tv_name.c_str(), cIdx);
                return false;
            }
        }

        printf("PFM sorting TV file %s loaded successfully\n", tv_name.c_str());
        return true;
    } catch (const H5::FileIException &e) {
        e.printErrorStack();
        return false;
    }
}

bool pfm_data_manage_t::pfm_validate_tv_h5(std::string& tv_name)
{
    // get number of cells from the TV name "PFM_SORT_TV_xxCELLS_SLOT_yyyy.h5"
    const std::size_t tv_pos = tv_name.find("TV_");
    const std::size_t cells_pos = tv_name.find("CELLS");
    
    if (tv_pos == std::string::npos || cells_pos == std::string::npos) {
        printf("ERROR: PFM sorting - invalid TV file name format: %s\n", tv_name.c_str());
        return false;
    }
    
    const std::size_t num_start = tv_pos + 3;  // Position after "TV_"
    if (num_start >= cells_pos) {
        printf("ERROR: PFM sorting - invalid TV file name format: %s\n", tv_name.c_str());
        return false;
    }
    
    int num_cell{};
    try {
        num_cell = std::stoi(tv_name.substr(num_start, cells_pos - num_start));
    } catch (const std::invalid_argument& e) {
        printf("ERROR: PFM sorting - invalid number format in TV file name: %s\n", tv_name.c_str());
        return false;
    } catch (const std::out_of_range& e) {
        printf("ERROR: PFM sorting - number out of range in TV file name: %s\n", tv_name.c_str());
        return false;
    }
    
    if (num_cell != m_num_cell) {
        printf("ERROR: PFM sorting - number of cells in the TV file does not match.\n");
        return false;
    }

    std::vector<cumac_pfm_cell_info_t> temp_cell_info(num_cell);
    std::vector<cumac_pfm_output_cell_info_t> temp_output_cell_info(num_cell);

    if (!pfm_load_tv_H5(tv_name, temp_cell_info, temp_output_cell_info)) {
        return false;
    }

    for (int cIdx = 0; cIdx < num_cell; cIdx++) {
        // check if temp_cell_info[cIdx] matches pfm_cell_info[cIdx]
        if (temp_cell_info[cIdx].num_ue != pfm_cell_info[cIdx].num_ue ||
            temp_cell_info[cIdx].num_lc_per_ue != pfm_cell_info[cIdx].num_lc_per_ue ||
            temp_cell_info[cIdx].num_lcg_per_ue != pfm_cell_info[cIdx].num_lcg_per_ue) {
            printf("ERROR: PFM sorting - cell info in the TV file does not match for cell %d.\n", cIdx);
            return false;
        }

        for (int idx = 0; idx < (CUMAC_PFM_NUM_QOS_TYPES_UL + CUMAC_PFM_NUM_QOS_TYPES_DL); idx++) {
            if (temp_cell_info[cIdx].num_output_sorted_lc[idx] != pfm_cell_info[cIdx].num_output_sorted_lc[idx]) {
                printf("ERROR: PFM sorting - number of output sorted LCs in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }

        // check if temp_cell_info[cIdx].ue_info matches pfm_cell_info[cIdx].ue_info
        for (int ueIdx = 0; ueIdx < temp_cell_info[cIdx].num_ue; ueIdx++) {
            if (temp_cell_info[cIdx].ue_info[ueIdx].rcurrent_dl != pfm_cell_info[cIdx].ue_info[ueIdx].rcurrent_dl ||
                temp_cell_info[cIdx].ue_info[ueIdx].rcurrent_ul != pfm_cell_info[cIdx].ue_info[ueIdx].rcurrent_ul ||
                temp_cell_info[cIdx].ue_info[ueIdx].rnti != pfm_cell_info[cIdx].ue_info[ueIdx].rnti ||
                temp_cell_info[cIdx].ue_info[ueIdx].id != pfm_cell_info[cIdx].ue_info[ueIdx].id ||
                temp_cell_info[cIdx].ue_info[ueIdx].num_layers_dl != pfm_cell_info[cIdx].ue_info[ueIdx].num_layers_dl ||
                temp_cell_info[cIdx].ue_info[ueIdx].num_layers_ul != pfm_cell_info[cIdx].ue_info[ueIdx].num_layers_ul ||
                temp_cell_info[cIdx].ue_info[ueIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].flags) {
                printf("ERROR: PFM sorting - UE info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            } else {
                // check dl_lc_info
                for (int lcIdx = 0; lcIdx < temp_cell_info[cIdx].num_lc_per_ue; lcIdx++) {
                    if (temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].tbs_scheduled != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].tbs_scheduled ||
                        temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].flags ||
                        temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].qos_type != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].qos_type) {
                        printf("ERROR: PFM sorting - DL LC info in the TV file does not match for cell %d.\n", cIdx);
                        return false;
                    }
                }

                // check ul_lcg_info
                for (int lcgIdx = 0; lcgIdx < temp_cell_info[cIdx].num_lcg_per_ue; lcgIdx++) {
                    if (temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].tbs_scheduled != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].tbs_scheduled ||
                        temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].flags ||
                        temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].qos_type != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].qos_type) {
                        printf("ERROR: PFM sorting - UL LCG info in the TV file does not match for cell %d.\n", cIdx);
                        return false;
                    }
                }
            }
        }
    }

    // check if temp_output_cell_info matches pfm_output_cell_info
    for (int cIdx = 0; cIdx < num_cell; cIdx++) {
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[0]; idx++) {
            if (temp_output_cell_info[cIdx].dl_gbr_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].dl_gbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_gbr_critical[idx].lc_id != pfm_output_cell_info_gpu[cIdx].dl_gbr_critical[idx].lc_id) {
                printf("ERROR: PFM sorting - DL GBR critical LC info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[1]; idx++) {
            if (temp_output_cell_info[cIdx].dl_gbr_non_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].dl_gbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_gbr_non_critical[idx].lc_id != pfm_output_cell_info_gpu[cIdx].dl_gbr_non_critical[idx].lc_id) {
                printf("ERROR: PFM sorting - DL GBR non-critical LC info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[2]; idx++) {
            if (temp_output_cell_info[cIdx].dl_ngbr_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].dl_ngbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_ngbr_critical[idx].lc_id != pfm_output_cell_info_gpu[cIdx].dl_ngbr_critical[idx].lc_id) {
                printf("ERROR: PFM sorting - DL NGBR critical LC info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[3]; idx++) {
            if (temp_output_cell_info[cIdx].dl_ngbr_non_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].dl_ngbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_ngbr_non_critical[idx].lc_id != pfm_output_cell_info_gpu[cIdx].dl_ngbr_non_critical[idx].lc_id) {
                printf("ERROR: PFM sorting - DL NGBR non-critical LC info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[4]; idx++) {
                if (temp_output_cell_info[cIdx].dl_mbr_non_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].dl_mbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_mbr_non_critical[idx].lc_id != pfm_output_cell_info_gpu[cIdx].dl_mbr_non_critical[idx].lc_id) {
                printf("ERROR: PFM sorting - DL MBR non-critical LC info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[5]; idx++) {
            if (temp_output_cell_info[cIdx].ul_gbr_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].ul_gbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_gbr_critical[idx].lcg_id != pfm_output_cell_info_gpu[cIdx].ul_gbr_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting - UL GBR critical LCG info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[6]; idx++) {
            if (temp_output_cell_info[cIdx].ul_gbr_non_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].ul_gbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_gbr_non_critical[idx].lcg_id != pfm_output_cell_info_gpu[cIdx].ul_gbr_non_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting - UL GBR non-critical LCG info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[7]; idx++) {  
            if (temp_output_cell_info[cIdx].ul_ngbr_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].ul_ngbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_ngbr_critical[idx].lcg_id != pfm_output_cell_info_gpu[cIdx].ul_ngbr_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting - UL NGBR critical LCG info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[8]; idx++) {
            if (temp_output_cell_info[cIdx].ul_ngbr_non_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].ul_ngbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_ngbr_non_critical[idx].lcg_id != pfm_output_cell_info_gpu[cIdx].ul_ngbr_non_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting - UL NGBR non-critical LCG info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
        
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[9]; idx++) {
            if (temp_output_cell_info[cIdx].ul_mbr_non_critical[idx].rnti != pfm_output_cell_info_gpu[cIdx].ul_mbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_mbr_non_critical[idx].lcg_id != pfm_output_cell_info_gpu[cIdx].ul_mbr_non_critical[idx].lcg_id) {
                printf("ERROR: PFM sorting - UL MBR non-critical LCG info in the TV file does not match for cell %d.\n", cIdx);
                return false;
            }
        }
    }

    printf("PFM sorting TV file %s validated successfully\n", tv_name.c_str());
    return true;
}