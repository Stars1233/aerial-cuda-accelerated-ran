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

#ifndef _TEST_CUMAC_CONFIGS_HPP
#define _TEST_CUMAC_CONFIGS_HPP

#include "yaml.hpp"
#include "nv_phy_utils.hpp"
#include <vector>

// #include "common_defines.hpp"
#include "common_utils.hpp"

class test_cumac_configs {
public:
    test_cumac_configs(yaml::node yaml_node);
    ~test_cumac_configs();

    struct nv::thread_config& get_recv_thread_config()
    {
        return recv_thread_config;
    }

    struct nv::thread_config& get_sched_thread_config()
    {
        return sched_thread_config;
    }

    struct nv::thread_config& get_builder_thread_config()
    {
        return builder_thread_config;
    }

    int get_fapi_tb_loc()
    {
        return fapi_tb_loc;
    }

    int get_test_slots()
    {
        return test_slots;
    }

    int get_restart_option()
    {
        return restart_option;
    }

    int get_restart_interval()
    {
        return restart_interval;
    }

    int get_cell_run_slots()
    {
        return cell_run_slots;
    }

    int get_cell_stop_slots()
    {
        return cell_stop_slots;
    }

    int get_ipc_sync_mode()
    {
        return ipc_sync_mode;
    }

    int get_max_msg_size()
    {
        return max_msg_size;
    }

    int get_max_data_size()
    {
        return max_data_size;
    }

    void set_max_msg_size(int max_size)
    {
        max_msg_size = max_size;
    }

    void set_max_data_size(int max_size)
    {
        max_data_size = max_size;
    }

    yaml::node& get_cumac_yaml_root()
    {
        return cumac_yaml_root;
    }

    // CUMAC message validating
    int validate_enable = 0;
    int validate_log_opt = 0;

    // Enable CUMAC builder thread
    int builder_thread_enable = 0;

    // L2 scheduler total time of a slot
    std::vector<int32_t> cumac_stt;

    int cumac_cp_standalone = 0;

    // Set bits to debug: b0 - Enable printing SCH_TTI.req buffer; b1 - Enable printing SCH_TTI.resp buffer
    int debug_option = 0;

    // CUMAC task bitmask: b0 - multiCellUeSelection; b1 - multiCellScheduler; b2 - multiCellLayerSel; b3 - mcsSelectionLUT
    int task_bitmask = 0;

    uint32_t cumac_cell_num = 0;

    // Total slot number in cuMAC-CP test
    uint32_t cumac_test_slots = 0;

    // cuMAC worker cores
    std::vector<int> worker_cores;

private:
    // Below parameters are to be loaded from yaml configuration file
    int max_msg_size = 0;
    int max_data_size = 0;
    int ipc_sync_mode = 0;

    int restart_option = 0;

    // TB data location
    int fapi_tb_loc = 0;

    // All cells restart test
    int test_slots = 0;
    int restart_interval = 0;

    // Single cell restart test
    int cell_run_slots = 0;
    int cell_stop_slots = 0;

    yaml::node cumac_yaml_root;

    struct nv::thread_config recv_thread_config;
    struct nv::thread_config sched_thread_config;
    struct nv::thread_config builder_thread_config;
};

#endif /* _TEST_CUMAC_CONFIGS_HPP */
