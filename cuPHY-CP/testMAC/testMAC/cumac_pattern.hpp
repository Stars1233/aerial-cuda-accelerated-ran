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

#ifndef _CUMAC_PATTERN_HPP_
#define _CUMAC_PATTERN_HPP_

#include <vector>
#include <unordered_map>

#include "nv_ipc_utils.h"
#include "cuphy.h"
#include "yaml.hpp"
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"

#include "cumac_defines.hpp"
#include "test_mac_configs.hpp"

// Dimensions: slot, cell, cumac_group, type
using cumac_slot_pattern_t = std::vector<std::vector<std::vector<std::vector<cumac_req_t*>>>>;

class cumac_pattern {
public:
    cumac_pattern(test_mac_configs* configs);
    ~cumac_pattern();
    int cumac_pattern_parsing(const char* lp_file_name, uint32_t ch_mask, uint64_t cell_mask);

    cumac_cell_configs_t& get_cumac_cell_configs(int cell_id)
    {
        if(cell_id < cumac_cell_configs_v.size())
        {
            return *cumac_cell_configs_v[cell_id];
        }
        else
        {
            throw std::runtime_error("invalid cell_id");
        }
    }

    int get_cumac_type()
    {
        return cumac_type;
    }

    int get_cell_num()
    {
        return cell_num;
    }

    int get_slots_per_frame()
    {
        return slots_per_frame;
    }

    int get_sched_slot_num()
    {
        return sched_slot_num;
    }

    int get_negative_test()
    {
        return negative_test;
    }

    cumac_slot_pattern_t& get_slot_cell_patterns()
    {
        return using_init_patterns ? init_slots_pattern : sched_slots_pattern;
    }

    cumac_slot_pattern_t& get_slot_cell_patterns(uint64_t slot_counter)
    {
        if (yaml_configs->app_mode != 0)
        {
            return dynamic_slots_pattern;
        }

        if(slot_counter < init_slot_num)
        {
            return init_slots_pattern;
        }
        else
        {
            return sched_slots_pattern;
        }
    }
 
    cumac_slot_pattern_t& get_sched_slots_patterns()
    {
        return sched_slots_pattern;
    }

    void check_init_pattern_finishing(uint64_t scheduled_slot_num)
    {
        if (using_init_patterns == false)
        {
            return;
        }

        if (scheduled_slot_num == init_slot_num)
        {
            using_init_patterns = false;
        }
    }

    int apply_reconfig_pattern(int cell_id);

    int populate_cumac_pattern(cumac_slot_pattern_t& slots_data, yaml::node cell_config, int cell_id, int slot_idx);

    uint32_t get_slot_in_frame(uint16_t sfn, uint16_t slot)
    {
        uint32_t index = sfn; // extend to uint32_t
        return index * slots_per_frame + slot;
    }

    int get_first_ue_group_slot()
    {
        return first_ue_group_slot;
    }

private:
    int parse_tv_pdcch(hdf5hpp::hdf5_file& file, cumac_req_t* req);

    int parse_tv_file(cumac_req_t* req);

    /**
     * @brief Load 4T4R test-vector datasets into req->tv_data.
     * @param[in,out] req Request whose tv_data is populated on success.
     * @return 0 on success, -1 on error.
     */
    [[nodiscard]] int load_4t4r_tv(cumac_req_t* req);

    int load_pfm_sorting_tv(cumac_req_t* req);

    /** Load MU UE pairing TV files. Returns 0 on success, -1 if no TV file
     *  exists for the slot/cell (caller should clear MU_UE_GRP bit in
     *  taskBitMask). */
    int load_mu_ue_grp_tv(cumac_req_t* req);

    /** Probe the cuMAC TV directory for the schedule_slot_period H5 attribute.
     *  Returns the period if any muUePairTV file exposes it, otherwise 0.
     *  Called once at the start of cumac_pattern_parsing when MU_UE_GRP is
     *  enabled in the YAML task_bitmask. */
    int probe_schedule_slot_period();

    /** Probe the cuMAC TV directory for the TV with first_slot=1.
     *  Returns the tv_slot_id (after srs_slot_lag) of the first TV, or -1 if
     *  no TV carries the first_slot attribute.
     *  Requires sched_slot_num > 0 before calling. */
    int probe_first_ue_group_slot();

    int load_h5_config_params(int cell_id, const char* config_params_h5_file);
    int update_expected_values(int cell_id, cumac_req_t* req);

    void parse_slots(yaml::node& slot_list, cumac_slot_pattern_t& slots_data);

    void free_test_vector(cumac_test_vector_t* tv);

    int cumac_type;
    int channel_mask;

    std::vector<int32_t> lp_cell_id_vec; // map<cell_id, lp_cell_id>

    int cell_num;
    int sched_slot_num;
    int first_ue_group_slot{-1}; // tv_slot_id of the TV marked first_slot=1; -1 if none
    int init_slot_num;
    int slots_per_frame;
    int config_static_harq_proc_id;

    bool using_init_patterns;

    // TestMAC YAML configs
    test_mac_configs* yaml_configs;
    test_cumac_configs* cumac_configs;

    // Expected throughput data parsed from TV
    std::vector<cumac_thrput_t> expected;

    // Dimensions: slot, cell, cumac_group, channel
    cumac_slot_pattern_t init_slots_pattern;
    cumac_slot_pattern_t sched_slots_pattern;
    cumac_slot_pattern_t reconfig_slots_pattern;

    cumac_slot_pattern_t dynamic_slots_pattern;

    // <channel, <tv_name, tv_file>*>
    std::unordered_map<int, std::unordered_map<std::string, std::string>*> tv_maps;

    std::vector<cumac_cell_configs_t*> cumac_cell_configs_v;

    // negative test flag
    int negative_test;

    std::unordered_map<std::string, hdf5hpp::hdf5_file> h5file_map;
};

#endif /* _CUMAC_PATTERN_HPP_ */
