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

#include "fapi_defines.hpp"
#include "test_mac_configs.hpp"

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 6) // "MAC.CFG"

using namespace std;

/**
 * Convert FAPI group name to FAPI group ID
 *
 * @param[in] name FAPI group name
 * @return FAPI group ID
 */
static int fapi_group_id_from_string(const std::string& name)
{
    if (name == "DL_TTI_REQ")    return static_cast<int>(DL_TTI_REQ);
    if (name == "UL_TTI_REQ")    return static_cast<int>(UL_TTI_REQ);
    if (name == "TX_DATA_REQ")   return static_cast<int>(TX_DATA_REQ);
    if (name == "UL_DCI_REQ")    return static_cast<int>(UL_DCI_REQ);
    if (name == "DL_BFW_CVI_REQ") return static_cast<int>(DL_BFW_CVI_REQ);
    if (name == "UL_BFW_CVI_REQ") return static_cast<int>(UL_BFW_CVI_REQ);
    return -1;
}

/**
 * Constructor
 *
 * @param[in] config_node YAML configuration node
 */
test_mac_configs::test_mac_configs(yaml::node config_node) :
#ifdef AERIAL_CUMAC_ENABLE
    cumac_configs(nullptr),
#endif
    yaml_config(config_node)
{
    // Initiate default values
    for (int i = 0; i < MAX_CELLS_PER_SLOT; i ++)
    {
        num_mem_bank_cv_config_req_sent[i] = 0;
    }

    app_mode = 0;

    // PDSCH TB align bytes (For SCF 222 pdschMacPduBitsAlignment configuration). Valid values are 1, 2, 4, 8, 16, 32.
    pdsch_align_bytes = config_node["pdsch_align_bytes"].as<uint32_t>();
    switch(pdsch_align_bytes)
    {
    case 1:
    case 2:
    case 4:
    case 8:
    case 16:
    case 32:
        break;
    default:
        NVLOGF(TAG, AERIAL_CONFIG_EVENT, "Invalid confg: pdsch_align_bytes = {}", pdsch_align_bytes);
        break;
    }

    fapi_tb_loc         = config_node["fapi_tb_loc"].as<int>();
    test_slots          = config_node["test_slots"].as<int>();
    restart_option      = config_node["restart_option"].as<int>();
    restart_interval    = config_node["restart_interval"].as<int>();
    cell_run_slots      = config_node["cell_run_slots"].as<int>();
    cell_stop_slots     = config_node["cell_stop_slots"].as<int>();
    ipc_sync_mode       = config_node["ipc_sync_mode"].as<int>();
    if (config_node.has_key("enable_srs_l1_limit_testing"))
    {
        enable_srs_l1_limit_testing = config_node["enable_srs_l1_limit_testing"].as<int>();
    }   // default disabled
    else
    {
        enable_srs_l1_limit_testing = 0;
    }

    max_msg_size = 0;
    max_data_size = 0;

    cell_config_timeout = config_node["cell_config_timeout"].as<int>();
    cell_config_wait = config_node["cell_config_wait"].as<int>();
    cell_config_retry = config_node["cell_config_retry"].as<int>();

    validate_enable = config_node["validate_enable"].as<int>();
    validate_log_opt = config_node["validate_log_opt"].as<int>();

    tv_data_map_enable = config_node["tv_data_map_enable"].as<int>();

    const bool has_dummy_tti_node = config_node.has_key("enable_dummy_tti");
    is_dummy_tti_enabled = has_dummy_tti_node ? (config_node["enable_dummy_tti"].as<int>() != 0) : false;

    oam_cell_ctrl_cmd = config_node["oam_cell_ctrl_cmd"].as<int>();

    oam_server_addr = config_node["oam_server_addr"].as<std::string>();

    fapi_delay_bit_mask = config_node["fapi_delay_bit_mask"].as<int>();

    duplicate_fapi_bit_mask = config_node["duplicate_fapi_bit_mask"].as<int>();
    duplicate_fapi_num_max = config_node["duplicate_fapi_num_max"].as<int>();

    NVLOGC_FMT(TAG, "{}: fapi_delay_bit_mask=0x{:02X} duplicate_fapi_bit_mask=0x{:02X} duplicate_fapi_num_max=0x{:02X}",
            __func__, fapi_delay_bit_mask, duplicate_fapi_bit_mask, duplicate_fapi_num_max);

    early_harq_deadline_ns = config_node["early_harq_deadline_ns"].as<int>();
    ul_ind_deadline_ns = config_node["ul_ind_deadline_ns"].as<int>();
    prach_ind_deadline_ns = config_node["prach_ind_deadline_ns"].as<int>();
    uci_ind_deadline_ns = config_node["uci_ind_deadline_ns"].as<int>();
    srs_late_deadline_ns = config_node["srs_late_deadline_ns"].as<int>();
    srs_early_deadline_ns = config_node["srs_early_deadline_ns"].as<int>();

    srs_vald_sample_num = config_node["srs_vald_sample_num"].as<uint32_t>();

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

    estimate_send_time = config_node["estimate_send_time"].as<int>();

    yaml::node schedule_total_time_node = config_node["schedule_total_time"];
    if (schedule_total_time_node.type() == YAML_SCALAR_NODE)
    {
        schedule_total_time.push_back(schedule_total_time_node.as<int32_t>());
    }
    else if (schedule_total_time_node.length() == 0)
    {
        schedule_total_time.push_back(0);
    }
    else if (schedule_total_time_node.length() > 20)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Read YAML: schedule_total_time size cannot be more than 20");
    }
    else
    {
        schedule_total_time.resize(schedule_total_time_node.length());
        for (int i = 0; i < schedule_total_time.size(); i ++)
        {
            schedule_total_time[i] = schedule_total_time_node[i].as<int32_t>();
        }
    }

    if (config_node.has_key("fapi_tx_deadline_enable"))
    {
        fapi_tx_deadline_enable = config_node["fapi_tx_deadline_enable"].as<int>();
    }

    if (config_node.has_key("fapi_tx_time_per_msg_ns"))
    {
        fapi_tx_time_per_msg_ns = config_node["fapi_tx_time_per_msg_ns"].as<int>();
    }

    if (config_node.has_key("fapi_tx_deadline"))
    {
        yaml::node deadline_node = config_node["fapi_tx_deadline"];
        fapi_tx_deadline_ns.resize(static_cast<size_t>(FAPI_REQ_SIZE), 0);
        for (int i = 0; i < deadline_node.length(); i++)
        {
            std::string group_name = deadline_node[i]["group_id"].as<std::string>();
            int deadline_us = deadline_node[i]["deadline"].as<int>();
            int gid = fapi_group_id_from_string(group_name);
            if (gid >= 0 && gid < FAPI_REQ_SIZE)
            {
                fapi_tx_deadline_ns[static_cast<size_t>(gid)] = static_cast<int64_t>(deadline_us) * 1000; // us -> ns
            }
            else
            {
                NVLOGW_FMT(TAG, "fapi_tx_deadline: unknown group_id '{}', skipped", group_name.c_str());
            }
        }
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

    if (config_node.has_key("test_cell_update"))
    {
        yaml::node cell_update = config_node["test_cell_update"];
        yaml::node test_cells = cell_update["test_cells"];
        yaml::node test_sequence = cell_update["test_sequence"];

        int cell_enabled[32];
        memset(cell_enabled, 0, sizeof(int) * 32);
        for (int i = 0; i < test_cells.length(); i ++)
        {
            int cell_id = test_cells[i].as<int>();
            cell_enabled[cell_id] = 1;
        }

        for (int i = 0; i < test_sequence.length(); i ++)
        {
            cell_update_cmd_t cmd;
            cmd.slot_point = test_sequence[i]["slot_point"].as<int>();
            yaml::node configs = test_sequence[i]["configs"];
            for (int j = 0; j< configs.length(); j++)
            {
                int cell_id = configs[j]["cell_id"].as<int>();
                if (cell_enabled[cell_id])
                {
                    cell_param_t param;
                    param.cell_id = cell_id;
                    param.vlan = configs[j]["vlan"].as<int>();
                    param.pcp = configs[j]["pcp"].as<int>();
                    param.mac = configs[j]["dst_mac"].as<std::string>();
                    NVLOGC_FMT(TAG, "test_cell_update: slot_point={} cell_id={} dst_mac={} vlan={} pcp={}",
                            cmd.slot_point, param.cell_id, param.mac.c_str(), param.vlan, param.pcp);
                    cmd.cell_params.push_back(std::move(param));
                }
            }
            if(cmd.cell_params.size() > 0)
            {
                cell_update_commands.push_back(std::move(cmd));
            }
        }

        int size = cell_update_commands.size();
        int period = size > 0 ? cell_update_commands[size -1].slot_point : 0;
        NVLOGC_FMT(TAG, "test_cell_update: size={} period={}", cell_update_commands.size(), period);
    }

    if(config_node.has_key("conf_test"))
    {
       if(config_node["conf_test"]["conf_test_enable"].as<int>() == 1)
       {
           conformance_test_params.conformance_test_enable = true;
       }
       else
       {
           conformance_test_params.conformance_test_enable = false;
       }
       conformance_test_params.conformance_test_start_time = config_node["conf_test"]["conformance_test_start_time"].as<uint32_t>();
       conformance_test_params.conformance_test_slots = config_node["conf_test"]["conf_test_slots"].as<int>();
       conformance_test_params.prach_params.expect_prmbindexes = config_node["conf_test"]["prach"]["expect_prmbindexes"].as<int>();
       conformance_test_params.prach_params.delay_error_tolerance = config_node["conf_test"]["prach"]["delay_error_tolerance"].as<float>();
       conformance_test_params.pucch_params.ack_nack_pattern = config_node["conf_test"]["pucch"]["ack_nack_pattern"].as<int>();
       conformance_test_params.conformance_test_stats_file = config_node["conf_test"]["stats_file"].as<string>();

    }
    else
    {
        conformance_test_params.conformance_test_enable = false;
    }
    NVLOGI_FMT(TAG, "Read YAML: recv_thread_config: {} core={} priority={}", recv_thread_config.name.c_str(), recv_thread_config.cpu_affinity, recv_thread_config.sched_priority);
#ifdef SCF_FAPI_10_04
    if(config_node.has_key("indicationPerSlot"))
    {
        indication_per_slot[0] = config_node["indicationPerSlot"]["rxDataIndPerSlot"].as<uint8_t>();
        indication_per_slot[1] = config_node["indicationPerSlot"]["crcIndPerSlot"].as<uint8_t>();
        indication_per_slot[2] = config_node["indicationPerSlot"]["uciIndPerSlot"].as<uint8_t>();
        indication_per_slot[3] = config_node["indicationPerSlot"]["rachIndPerSlot"].as<uint8_t>();
        indication_per_slot[4] = config_node["indicationPerSlot"]["srsIndPerSlot"].as<uint8_t>();
        indication_per_slot[5] = config_node["indicationPerSlot"]["dlTTIrspPerSlot"].as<uint8_t>();
    }
#endif

    if (config_node.has_key("enableTickDynamicSfnSlot"))
    {
        enableTickDynamicSfnSlot = config_node["enableTickDynamicSfnSlot"].as<int>();
    }
    else
    {
        enableTickDynamicSfnSlot = 1; // default enabled
    }

    // Config validation
    if (fapi_tx_deadline_enable != 0 && builder_thread_enable == 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "builder_thread_enable must be enabled when fapi_tx_deadline_enable is enabled");
    }
}

/**
 * Destructor
 */
test_mac_configs::~test_mac_configs()
{
}
