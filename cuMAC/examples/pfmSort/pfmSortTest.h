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

#include "api.h"
#include "cumac.h"
#include <yaml-cpp/yaml.h>

// Path and directory configuration
#define MAX_PATH_LEN 1024 //!< Maximum path length for file operations
#define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 4 //!< Number of parent directories to traverse to reach cuBB_SDK root
#define CONFIG_PFMSORT_CONFIG_YAML "cuMAC/examples/pfmSort/config.yaml" //!< Relative path to PFM sorting configuration file

constexpr uint16_t MAX_NUM_RNTI_PER_CELL = 65535; // max number of RNTIs per cell
constexpr uint16_t MIN_RNTI = 1; // min C-RNTI
constexpr uint16_t MAX_RNTI = 65535; // max C-RNTI

template <typename T>
using pfmTuple = std::tuple<T, uint32_t, uint32_t>;

// structure for mapping UE RNTIs to cuMAC 0-based UE IDs
struct ue_id_rnti_t {
    uint16_t id; // cuMAC 0-based UE ID
    uint16_t rnti; // C-RNTI

    ue_id_rnti_t(uint16_t id_, uint16_t rnti_)
        : id(id_), rnti(rnti_) {}
};

/// @brief Class for managing the UEs for PFM sorting in a cell
/// @note For illustration purpose. A real L2 stack implementation may differ in logic and data structure.
class pfm_data_manage_t {
    public:
        pfm_data_manage_t(const std::string& configFilePath, cudaStream_t strm);
        ~pfm_data_manage_t();
        int             unit_test(const uint16_t num_cell, const uint16_t num_slot);
        bool            add_pfm_ue(const uint16_t cell_id, const uint16_t num_new_ue);
        bool            remove_pfm_ue(const uint16_t cell_id, uint16_t rnti);
        uint16_t        get_num_cell() const { return m_num_cell; }
        uint16_t        get_num_pfm_ue(const uint16_t cell_id) { return ue_list[cell_id].size(); }
        uint16_t        get_id(const uint16_t cell_id, uint16_t rnti);
        uint16_t        get_ue_idx_in_list(const uint16_t cell_id, uint16_t rnti);
        ue_id_rnti_t    get_ue_id_rnti(const uint16_t cell_id, uint16_t ue_idx);
        unsigned        get_seed() const { return m_seed; }
        uint32_t        get_num_slot() const { return m_num_slot; }
        void            increase_slot_idx() { m_slot_idx++; }
        uint8_t*        get_pfm_output_host_ptr() { return reinterpret_cast<uint8_t*>(pfm_output_cell_info_gpu.data()); }
        uint16_t        get_max_num_h5_tv_created() { return m_max_num_h5_tv_created; }
        cumac::pfmSortTask* get_pfm_sort_task() { return pfmSortTask.get(); }

        void            sync_stream() { CUDA_CHECK_ERR(cudaStreamSynchronize(m_strm)); }

        void            prepare_pfm_data();

        bool            validate_pfm_output();

        void            cpu_pfm_sort();

        std::string     pfm_save_tv_H5();

        static bool     pfm_load_tv_H5(const std::string& tv_name, std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info);
 
        bool            pfm_validate_tv_h5(std::string& tv_name);

    private:
        uint32_t        m_slot_idx{0}; // slot index for PFM sorting
        uint32_t        m_num_slot; // number of test slots for PFM sorting
        unsigned        m_seed{0}; // randomness seed
        uint16_t        m_num_cell; // number of cells for PFM sorting
        uint16_t        m_num_ue_per_cell; // number of UEs per cell for PFM sorting
        uint16_t        m_num_dl_lc_per_ue; // number of DL LCs per UE for PFM sorting
        uint16_t        m_num_ul_lcg_per_ue; // number of UL LCGs per UE for PFM sorting
        
        cudaStream_t    m_strm; // CUDA stream for PFM sorting

        // for H5 TV creation
        uint16_t        m_max_num_h5_tv_created{80};

        std::vector<std::vector<ue_id_rnti_t>>  ue_list; // list of UEs for PFM sorting for each cell
        std::vector<std::deque<uint16_t>>       available_rnti; // queue of available RNTIs for each cell
        std::vector<std::deque<uint16_t>>       available_id; // queue of available cuMAC 0-based UE IDs for each cell

        std::unique_ptr<cumac::pfmSortTask>            pfmSortTask;

        std::vector<cumac_pfm_cell_info_t>             pfm_cell_info; // PFM sorting data structure for each cell

        std::vector<cumac_pfm_output_cell_info_t>      pfm_output_cell_info_gpu; // GPU PFM sorting output data structure for each cell

        std::vector<cumac_pfm_output_cell_info_t>      pfm_output_cell_info_cpu; // CPU PFM sorting output data structure for each cell

        std::vector<cumac::PFM_CELL_INFO_MANAGE>       cell_data_main_cpu; // CPU PFM sorting data structure for each cell

        std::vector<std::vector<uint32_t>>             num_lc_per_qos_type; // number of LCs per QoS type for each cell

        // private member functions
        [[nodiscard]] int loadConfigYaml(const std::string& configFilePath); // Load configuration from YAML file
};