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

#include "nvlog.hpp"

#include "cumac_configs.hpp"

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 21) // "CUMAC.CFG"

using namespace std;

test_cumac_configs::test_cumac_configs(yaml::node config_node) : cumac_yaml_root(config_node)
{
    fapi_tb_loc   = config_node["fapi_tb_loc"].as<int>();
    max_msg_size  = 0;
    max_data_size = 0;

    cumac_cp_standalone = config_node["cumac_cp_standalone"].as<int>();
    debug_option = config_node["debug_option"].as<int>();

    task_bitmask = config_node["task_bitmask"].as<int>();

    srs_slot_lag = 0;
    if (config_node.has_key("srs_slot_lag"))
    {
        srs_slot_lag = config_node["srs_slot_lag"].as<int>();
        if (srs_slot_lag < 0)
        {
            throw std::runtime_error("srs_slot_lag must be >= 0, got: " + std::to_string(srs_slot_lag));
        }
    }

    cumac_cell_num   = config_node["cumac_cell_num"].as<uint32_t>();
    cumac_test_slots = config_node["cumac_test_slots"].as<uint32_t>();

    validate_enable = config_node["validate_enable"].as<int>();
    validate_log_opt = config_node["validate_log_opt"].as<int>();

    recv_thread_config.name           = config_node["recv_thread_config"]["name"].as<std::string>();
    recv_thread_config.cpu_affinity   = config_node["recv_thread_config"]["cpu_affinity"].as<int>();
    recv_thread_config.sched_priority = config_node["recv_thread_config"]["sched_priority"].as<int>();

    sched_thread_config.name           = config_node["sched_thread_config"]["name"].as<std::string>();
    sched_thread_config.cpu_affinity   = config_node["sched_thread_config"]["cpu_affinity"].as<int>();
    sched_thread_config.sched_priority = config_node["sched_thread_config"]["sched_priority"].as<int>();

    builder_thread_config.name           = config_node["builder_thread_config"]["name"].as<std::string>();
    builder_thread_config.cpu_affinity   = config_node["builder_thread_config"]["cpu_affinity"].as<int>();
    builder_thread_config.sched_priority = config_node["builder_thread_config"]["sched_priority"].as<int>();

    builder_thread_enable = config_node["builder_thread_enable"].as<int>();

    yaml::node worker_cores_node = config_node["worker_cores"];
    if (worker_cores_node.length() > 0)
    {
        worker_cores.resize(worker_cores_node.length());
        for (int i = 0; i < worker_cores.size(); i ++)
        {
            worker_cores[i] = worker_cores_node[i].as<int>();
        }
    }

    yaml::node cumac_stt_node = config_node["schedule_total_time"];
    if (cumac_stt_node.type() == YAML_SCALAR_NODE)
    {
        cumac_stt.push_back(cumac_stt_node.as<int32_t>());
    }
    else if (cumac_stt_node.length() == 0)
    {
        cumac_stt.push_back(0);
    }
    else if (cumac_stt_node.length() > 20)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Read YAML: schedule_total_time size cannot be more than 20");
    }
    else
    {
        cumac_stt.resize(cumac_stt_node.length());
        for (int i = 0; i < cumac_stt.size(); i ++)
        {
            cumac_stt[i] = cumac_stt_node[i].as<int32_t>();
        }
    }
    NVLOGC_FMT(TAG, "cell_num={} task_bitmask=0x{:02X} srs_slot_lag={}", cumac_cell_num, task_bitmask, srs_slot_lag);
}

test_cumac_configs::~test_cumac_configs()
{
}
