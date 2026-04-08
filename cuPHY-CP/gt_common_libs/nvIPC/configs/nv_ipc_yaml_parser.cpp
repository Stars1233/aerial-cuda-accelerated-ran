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

#include "nv_ipc.h"
#include "nvlog.h"

#include "nvlog.hpp"
#include "nv_ipc.hpp"

#define TAG "NVIPC:YAML"

static void parse_mempool_size(nv_ipc_mempool_size_t mempools[], yaml::node& cfg_node)
{
    yaml::node size_node = cfg_node["mempool_size"];

    mempools[NV_IPC_MEMPOOL_CPU_MSG].buf_size = size_node["cpu_msg"]["buf_size"].as<int32_t>();
    mempools[NV_IPC_MEMPOOL_CPU_MSG].pool_len = size_node["cpu_msg"]["pool_len"].as<int32_t>();

    mempools[NV_IPC_MEMPOOL_CPU_DATA].buf_size = size_node["cpu_data"]["buf_size"].as<int32_t>();
    mempools[NV_IPC_MEMPOOL_CPU_DATA].pool_len = size_node["cpu_data"]["pool_len"].as<int32_t>();

    if (size_node.has_key("cpu_large"))
    {
        mempools[NV_IPC_MEMPOOL_CPU_LARGE].buf_size = size_node["cpu_large"]["buf_size"].as<int32_t>();
        mempools[NV_IPC_MEMPOOL_CPU_LARGE].pool_len = size_node["cpu_large"]["pool_len"].as<int32_t>();
    }

    if (size_node.has_key("cuda_data"))
    {
        mempools[NV_IPC_MEMPOOL_CUDA_DATA].buf_size = size_node["cuda_data"]["buf_size"].as<int32_t>();
        mempools[NV_IPC_MEMPOOL_CUDA_DATA].pool_len = size_node["cuda_data"]["pool_len"].as<int32_t>();
    }

    if (size_node.has_key("gpu_data"))
    {
        mempools[NV_IPC_MEMPOOL_GPU_DATA].buf_size = size_node["gpu_data"]["buf_size"].as<int32_t>();
        mempools[NV_IPC_MEMPOOL_GPU_DATA].pool_len = size_node["gpu_data"]["pool_len"].as<int32_t>();
    }
}

int nv_ipc_parse_yaml_node(nv_ipc_config_t* cfg, yaml::node* yaml_node, nv_ipc_module_t module_type)
{
    yaml::node& node_config    = *yaml_node;
    std::string transport_type = node_config["type"].as<std::string>();
    //------------------------------------------------------------------
    // Set up default values. The set_nv_ipc_default_config() function
    // requires the transport type to be set before calling.
    if(0 == strcasecmp(transport_type.c_str(), "udp"))
    {
        cfg->ipc_transport = NV_IPC_TRANSPORT_UDP;
    }
    else if(0 == strcasecmp(transport_type.c_str(), "shm"))
    {
        cfg->ipc_transport = NV_IPC_TRANSPORT_SHM;
    }
    else if(0 == strcasecmp(transport_type.c_str(), "dpdk"))
    {
        cfg->ipc_transport = NV_IPC_TRANSPORT_DPDK;
    }
    else if(0 == strcasecmp(transport_type.c_str(), "doca"))
    {
        cfg->ipc_transport = NV_IPC_TRANSPORT_DOCA;
    }
    else
    {
        NVLOGE_NO_FMT(TAG, AERIAL_CONFIG_EVENT, "Unexpected YAML transport type: {}", transport_type, transport_type.c_str());
    }

    set_nv_ipc_default_config(cfg, module_type);

    int pcap_max_msg_size = 8192;
    //------------------------------------------------------------------
    // Populate fields from YAML nodes, using default values set above
    // it the YAML node does not provide them
    if(NV_IPC_TRANSPORT_UDP == cfg->ipc_transport)
    {
        yaml::node udp_config = node_config["udp_config"];
        if(udp_config.has_key("local_port"))
        {
            cfg->transport_config.udp.local_port = udp_config["local_port"].as<int32_t>();
        }
        if(udp_config.has_key("remote_port"))
        {
            cfg->transport_config.udp.remote_port = udp_config["remote_port"].as<int32_t>();
        }
        if(udp_config.has_key("local_addr"))
        {
            nvlog_safe_strncpy(cfg->transport_config.udp.local_addr, udp_config["local_addr"].as<std::string>().c_str(), NV_IPV4_STRING_LEN);
        }
        if(udp_config.has_key("remote_addr"))
        {
            nvlog_safe_strncpy(cfg->transport_config.udp.remote_addr, udp_config["remote_addr"].as<std::string>().c_str(), NV_IPV4_STRING_LEN);
        }
        if(udp_config.has_key("msg_buf_size"))
        {
            cfg->transport_config.udp.msg_buf_size = udp_config["msg_buf_size"].as<int32_t>();
        }
        if(udp_config.has_key("data_buf_size"))
        {
            cfg->transport_config.udp.data_buf_size = udp_config["data_buf_size"].as<int32_t>();
        }
        pcap_max_msg_size = cfg->transport_config.udp.msg_buf_size;
    }
    else if(NV_IPC_TRANSPORT_SHM == cfg->ipc_transport)
    {
        yaml::node shm_config = node_config["shm_config"];
        std::string prefix = shm_config["prefix"].as<std::string>();
        nvlog_safe_strncpy(cfg->transport_config.shm.prefix, prefix.c_str(), NV_NAME_MAX_LEN);

        // Secondary process only need to parse the prefix
        if (!is_module_primary(module_type))
        {
            NVLOGC_FMT(TAG, "{}: loaded secondary nvipc config prefix={}", __FUNCTION__, prefix);
            return 0;
        }

        // Parse all other parameters for primary process
        cfg->transport_config.shm.cuda_device_id = shm_config["cuda_device_id"].as<int>();
        cfg->transport_config.shm.ring_len = shm_config["ring_len"].as<int32_t>();

        parse_mempool_size(cfg->transport_config.shm.mempool_size, shm_config);

        pcap_max_msg_size = cfg->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_MSG].buf_size;
    }
    else if(NV_IPC_TRANSPORT_DPDK == cfg->ipc_transport)
    {
        yaml::node dpdk_config = node_config["dpdk_config"];
        std::string prefix        = dpdk_config["prefix"].as<std::string>();
        std::string local_nic_pci = dpdk_config["local_nic_pci"].as<std::string>();
        std::string peer_nic_mac  = dpdk_config["peer_nic_mac"].as<std::string>();
        nvlog_safe_strncpy(cfg->transport_config.dpdk.prefix, prefix.c_str(), NV_NAME_MAX_LEN);
        nvlog_safe_strncpy(cfg->transport_config.dpdk.local_nic_pci, local_nic_pci.c_str(), NV_NAME_MAX_LEN);
        nvlog_safe_strncpy(cfg->transport_config.dpdk.peer_nic_mac, peer_nic_mac.c_str(), NV_NAME_MAX_LEN);

        if(dpdk_config.has_key("nic_mtu"))
        {
            cfg->transport_config.dpdk.nic_mtu = dpdk_config["nic_mtu"].as<int>();
        }
        else
        {
            // Set default MTU to 1536
            cfg->transport_config.dpdk.nic_mtu = 1536;
        }

        cfg->transport_config.dpdk.cuda_device_id = dpdk_config["cuda_device_id"].as<int>();
        cfg->transport_config.dpdk.lcore_id       = dpdk_config["lcore_id"].as<uint16_t>();
        cfg->transport_config.dpdk.need_eal_init  = dpdk_config["need_eal_init"].as<uint16_t>();

        parse_mempool_size(cfg->transport_config.dpdk.mempool_size, dpdk_config);

        pcap_max_msg_size = cfg->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_MSG].buf_size;
    }
    else if(NV_IPC_TRANSPORT_DOCA == cfg->ipc_transport)
    {
        yaml::node doca_config = node_config["doca_config"];
        std::string prefix        = doca_config["prefix"].as<std::string>();
        std::string local_nic_pci = doca_config["host_pci"].as<std::string>();
        std::string peer_nic_mac  = doca_config["dpu_pci"].as<std::string>();
        nvlog_safe_strncpy(cfg->transport_config.doca.prefix, prefix.c_str(), NV_NAME_MAX_LEN);
        nvlog_safe_strncpy(cfg->transport_config.doca.host_pci, local_nic_pci.c_str(), NV_NAME_MAX_LEN);
        nvlog_safe_strncpy(cfg->transport_config.doca.dpu_pci, peer_nic_mac.c_str(), NV_NAME_MAX_LEN);

        cfg->transport_config.doca.cuda_device_id = doca_config["cuda_device_id"].as<int>();
        cfg->transport_config.doca.cpu_core       = doca_config["cpu_core"].as<uint16_t>();

        parse_mempool_size(cfg->transport_config.doca.mempool_size, doca_config);

        pcap_max_msg_size = cfg->transport_config.shm.mempool_size[NV_IPC_MEMPOOL_CPU_MSG].buf_size;
    }
    else
    {
        throw std::runtime_error(std::string("nv_phy_mac_transport: Unexpected type '") +
                                 transport_type +
                                 std::string("'"));
    }

    // Init debug_configs to 0 for default values
    memset(&cfg->debug_configs, 0, sizeof(nv_ipc_debug_config_t));

    if(node_config.has_key("app_config") && is_module_primary(module_type))
    {
        yaml::node app_config = node_config["app_config"];
        if(app_config.has_key("grpc_forward"))
        {
            cfg->debug_configs.grpc_forward = app_config["grpc_forward"].as<int32_t>();
        }
        if(app_config.has_key("debug_timing"))
        {
            cfg->debug_configs.debug_timing = app_config["debug_timing"].as<int32_t>();
        }
        if(app_config.has_key("pcap_enable"))
        {
            cfg->debug_configs.pcap_enable = app_config["pcap_enable"].as<int32_t>();
        }
        if(app_config.has_key("pcap_sync_save"))
        {
            cfg->debug_configs.pcap_sync_save = app_config["pcap_sync_save"].as<int32_t>();
        }

        if (app_config.has_key("fapi_tb_loc"))
        {
            cfg->debug_configs.fapi_tb_loc = app_config["fapi_tb_loc"].as<int32_t>();
        }
        else
        {
            cfg->debug_configs.fapi_tb_loc = 1; // Default fapi_tb_loc = 1
        }

        cfg->debug_configs.pcap_max_msg_size = pcap_max_msg_size;
        cfg->debug_configs.pcap_max_data_size = app_config["pcap_max_data_size"].as<int32_t>();

        cfg->debug_configs.pcap_shm_caching_cpu_core = app_config["pcap_shm_caching_cpu_core"].as<int32_t>();
        cfg->debug_configs.pcap_file_saving_cpu_core = app_config["pcap_file_saving_cpu_core"].as<int32_t>();

        cfg->debug_configs.pcap_cache_size_bits = app_config["pcap_cache_size_bits"].as<int32_t>();
        cfg->debug_configs.pcap_file_size_bits = app_config["pcap_file_size_bits"].as<int32_t>();

        yaml::node msg_filter_node = app_config["msg_filter"];
        if (msg_filter_node.type() == YAML_SEQUENCE_NODE)
        {
            int i;
            for (i = 0; i < msg_filter_node.length(); i++)
            {
                int32_t msg_id = msg_filter_node[i].as<int32_t>();
                NVLOGC_FMT(TAG, "{}: PCAP msg_filter {}-{}: enable msg_id=0x{:02X}", __FUNCTION__, msg_filter_node.length(), i, msg_id);
                if (msg_id < NVIPC_MAX_MSG_ID)
                {
                    cfg->debug_configs.msg_filter[msg_id] = 1;
                }
                else
                {
                    NVLOGE_NO_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid YAML node type={} for msg_filter", static_cast<uint32_t>(msg_filter_node.type()));
                }
            }

            // Enable all message if filter is not configured
            if (msg_filter_node.length() == 0)
            {
                NVLOGC_FMT(TAG, "{}: PCAP msg_filter: enable all messages: 0x00-0x{:02X}", __FUNCTION__, NVIPC_MAX_MSG_ID - 1);
                for (int32_t msg_id = 0; msg_id < NVIPC_MAX_MSG_ID; msg_id++)
                {
                    cfg->debug_configs.msg_filter[msg_id] = 1;
                }
            }
        }
        else
        {
            NVLOGE_NO_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid YAML node type={} for msg_filter", static_cast<uint32_t>(msg_filter_node.type()));
        }

        yaml::node cell_filter_node = app_config["cell_filter"];
        if (cell_filter_node.type() == YAML_SEQUENCE_NODE)
        {
            int i;
            for (i = 0; i < cell_filter_node.length(); i++)
            {
                int32_t cell_id = cell_filter_node[i].as<int32_t>();
                NVLOGC_FMT(TAG, "{}: PCAP cell_filter {}-{}: enable cell_id={}", __FUNCTION__, cell_filter_node.length(), i, cell_id);
                if (cell_id < NVIPC_MAX_CELL_ID)
                {
                    cfg->debug_configs.cell_filter[cell_id] = 1;
                }
                else
                {
                    NVLOGE_NO_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid YAML node type={} for cell_filter", static_cast<uint32_t>(cell_filter_node.type()));
                }
            }

            // Enable all cells if filter is not configured
            if (cell_filter_node.length() == 0)
            {
                NVLOGC_FMT(TAG, "{}: PCAP cell_filter: enable all cells 0-{}", __FUNCTION__, NVIPC_MAX_CELL_ID - 1);
                for (int32_t cell_id = 0; cell_id < NVIPC_MAX_MSG_ID; cell_id++)
                {
                    cfg->debug_configs.cell_filter[cell_id] = 1;
                }
            }
        }
        else
        {
            NVLOGE_NO_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid YAML node type={} for cell_filter", static_cast<uint32_t>(cell_filter_node.type()));
        }
    }

    return 0;
}

int load_nv_ipc_yaml_config(nv_ipc_config_t* cfg, const char* yaml_path, nv_ipc_module_t module_type)
{
    NVLOGI_FMT(TAG, "{}: {}", __FUNCTION__, yaml_path);
    try
    {
        yaml::file_parser fp(yaml_path);
        yaml::document    doc        = fp.next_document();
        yaml::node        yaml_root  = doc.root();

        if (yaml_root.has_key("nvipc_log")) {
            yaml::node nvlog_node = yaml_root["nvipc_log"];
            std::string log_file = nvlog_node["fmt_log_path"].as<std::string>().append("/");
            log_file.append(nvlog_node["fmt_log_name"].as<std::string>());
            log_file.append(is_module_primary(module_type) ? "_primary.log" : "_secondary.log");
            nvlog_c_init(log_file.c_str());
            nvlog_set_log_level(nvlog_node["log_level"].as<int>());
            nvlog_set_max_file_size(nvlog_node["fmt_log_max_size"].as<uint64_t>() * 1024 * 1024);
        }

        yaml::node nvipc_node = yaml_root["transport"];
        nv_ipc_parse_yaml_node(cfg, &nvipc_node, module_type);
        return 0;
    }
    catch(const YAML::BadFile& badFile)
    {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: BadFile exception: {}", __func__, badFile.what());
        return -1;
    }
    catch(const std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: Exception: {}", __func__, e.what());
        return -1;
    }
    catch(...)
    {
        NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "{}: Unknown error", __func__);
        return -1;
    }
}