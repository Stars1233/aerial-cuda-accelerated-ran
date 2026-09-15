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

#include "cumac_cp_configs.hpp"

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 2) // "CUMCP.CFG"

using namespace std;

cumac_cp_configs::cumac_cp_configs(yaml::node config_node) : yaml_root(config_node)
{
    max_msg_size  = config_node["transport"]["shm_config"]["mempool_size"]["cpu_msg"]["buf_size"].as<int>();
    max_data_size = config_node["transport"]["shm_config"]["mempool_size"]["cpu_data"]["buf_size"].as<int>();

    recv_thread_config.name           = config_node["recv_thread_config"]["name"].as<std::string>();
    recv_thread_config.cpu_affinity   = config_node["recv_thread_config"]["cpu_affinity"].as<int>();
    recv_thread_config.sched_priority = config_node["recv_thread_config"]["sched_priority"].as<int>();

    thread_num_per_core = config_node["thread_num_per_core"].as<int>();
    cell_num = config_node["cell_num"].as<uint32_t>();
    task_ring_len = config_node["task_ring_len"].as<uint32_t>();
    run_in_cpu = config_node["run_in_cpu"].as<uint32_t>();
    debug_option = config_node["debug_option"].as<int>();

    cumac_group_tv_file = config_node["cumac_group_tv_file"].as<std::string>();

    enable_tv_test = false;
    if (config_node.has_key("enable_tv_test"))
    {
        enable_tv_test = config_node["enable_tv_test"].as<int>() != 0;
    }

    gpu_id = config_node["gpu_id"].as<uint32_t>();

    cuda_block_num = config_node["cuda_block_num"].as<uint32_t>();

    // Performance tuning parameters
    group_buffer_enable = config_node["group_buffer_enable"].as<uint32_t>();
    multi_stream_enable = config_node["multi_stream_enable"].as<uint32_t>();
    slot_concurrent_enable = config_node["slot_concurrent_enable"].as<uint32_t>();
    enable_gpu_share = false;
    if (config_node.has_key("enable_gpu_share"))
    {
        enable_gpu_share = config_node["enable_gpu_share"].as<int>() != 0;
    }

    enable_cubb = false;
    if (config_node.has_key("enable_cubb"))
    {
        enable_cubb = config_node["enable_cubb"].as<int>() != 0;
    }

    srs_slot_lag = 0;
    if (config_node.has_key("srs_slot_lag"))
    {
        srs_slot_lag = config_node["srs_slot_lag"].as<int>();
        if (srs_slot_lag < 0)
        {
            NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "srs_slot_lag must be >= 0, got: {}", srs_slot_lag);
        }
    }

    task_bitmask = CUMAC_CP_TASK_MASK_DEFAULT;
    if (config_node.has_key("task_bitmask"))
    {
        task_bitmask = static_cast<uint32_t>(config_node["task_bitmask"].as<int>());
    }

    num_blocks_per_row = 8;
    if (config_node.has_key("num_blocks_per_row"))
    {
        num_blocks_per_row = static_cast<uint16_t>(config_node["num_blocks_per_row"].as<uint32_t>());
    }

    yaml::node worker_cores_node = config_node["worker_cores"];
    if (worker_cores_node.length() > 0)
    {
        worker_cores.resize(worker_cores_node.length());
        for (int i = 0; i < worker_cores.size(); i ++)
        {
            worker_cores[i] = worker_cores_node[i].as<int>();
        }
    }
    else
    {
        // Use the recv_thread_config CPU core by default
        worker_cores.resize(1);
        worker_cores[0] = recv_thread_config.cpu_affinity;
    }
}

cumac_cp_configs::~cumac_cp_configs()
{
}
