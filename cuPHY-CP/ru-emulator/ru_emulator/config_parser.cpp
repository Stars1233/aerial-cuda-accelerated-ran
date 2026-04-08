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

#include "config_parser.hpp"
#include "yaml_sdk_version.hpp"

#define TAG (NVLOG_TAG_BASE_RU_EMULATOR + 2) // "RU.PARSER"
#define TAG_TV_CONFIGS (NVLOG_TAG_BASE_RU_EMULATOR + 7) // "RU.TV_CONFIGS"

void RU_Emulator::set_default_configs()
{
    opt_tti_us = 500;
    opt_dl_up_sanity_check = 1;
    opt_max_sect_stats = 0;
    opt_bfw_dl_validation = 1;
    opt_bfw_ul_validation = 1;
    opt_beamid_validation = RE_DISABLED;
    opt_sectionid_validation = RE_DISABLED;
    opt_num_slots_ul = 25;
    opt_num_slots_dl = 100;
    opt_c_interval_us = 0;
    opt_c_plane_per_symbol = 1;
    opt_prach_c_plane_per_symbol = 0;
    opt_timer_level = 0;
    opt_timer_offset_us = -20;
    opt_symbol_offset_us = 0;
    max_slot_id = 1;
    opt_num_cells = 1;
    opt_send_slot = RE_DISABLED;
    opt_forever = RE_ENABLED;
    opt_validate_dl_timing = RE_DISABLED;
    opt_dl_warmup_slots = 512;
    opt_ul_warmup_slots = 128;
    opt_timing_histogram = RE_DISABLED;
    opt_timing_histogram_bin_size = 1000;

    nic_interfaces = {};
    dpdk.vlan = 2;
    dpdk.socket_id = AERIAL_SOCKET_ID_ANY;

    opt_oam_cell_ctrl_cmd = RE_DISABLED;
    opt_multi_section_ul = RE_ENABLED;
    opt_ul_enabled = RE_ENABLED;
    opt_prach_enabled = RE_ENABLED;
    opt_srs_enabled = RE_ENABLED;
    opt_pucch_enabled = RE_ENABLED;
    opt_pusch_enabled = RE_ENABLED;
    opt_dl_enabled = RE_ENABLED;
    opt_dlc_tb = RE_DISABLED;
    opt_mod_comp_enabled = RE_DISABLED;
    opt_non_mod_comp_enabled = RE_DISABLED;

    opt_enable_dl_proc_mt=0;
    opt_beamforming = RE_DISABLED;
    opt_dl_approx_validation = RE_DISABLED;
    opt_enable_mmimo = RE_DISABLED;
    opt_min_ul_cores_per_cell_mmimo = MIN_UL_CORES_PER_CELL_MMIMO;
    opt_enable_beam_forming = RE_DISABLED;
    opt_enable_cplane_worker_tracing = RE_DISABLED;
    // opt_pdsch_validation = RE_ENABLED;
    // opt_pbch_validation = RE_ENABLED;
    // opt_pdcch_ul_validation = RE_ENABLED;
    // opt_pdcch_dl_validation = RE_ENABLED;
    opt_num_flows_per_dl_thread = 8;

    opt_launch_pattern_file = "launch_pattern";
    opt_config_file = "config.yaml";
    opt_launch_pattern_version = 1;
    global_slot_counter = 0;
    enable_srs = false;

    opt_afh_txq_size = 2048;
    opt_afh_rxq_size = 4096;
    opt_afh_txq_request_num = 16;
    opt_afh_dpdk_thread = 1;
    opt_afh_pdump_client_thread = -1;
    opt_afh_accu_tx_sched_res_ns = 500;
    opt_aerial_fh_per_rxq_mempool = 0;
    opt_afh_cpu_mbuf_pool_size_per_rxq = 131072;
    opt_afh_cpu_mbuf_pool_tx_size = 131072;
    opt_afh_cpu_mbuf_pool_rx_size = 131072;
    opt_afh_split_rx_tx_mp = false;
    opt_afh_dpdk_file_prefix = "ru_emulator";
    opt_afh_mtu = 1514;
    opt_split_srs_txq = 1;
    opt_enable_srs_eaxcid_pacing = RE_DISABLED;
    opt_srs_pacing_s3_srs_symbols = 2;
    opt_srs_pacing_s4_srs_symbols = 0;
    opt_srs_pacing_s5_srs_symbols = 0;
    opt_srs_pacing_eaxcids_per_tx_window = 4;
    opt_srs_pacing_eaxcids_per_symbol = 64;
    num_srs_txqs = 0;
    opt_ecpri_hdr_cfg_test = 0;
    dl_cores_per_cell = 1;
    csi_rs_optimized_validation = 0;
    opt_debug_u_plane_prints = 0;
    opt_debug_u_plane_threshold = 20;
    is_cx6_nic = true;
    opt_ul_only = RE_DISABLED;
}

// usage()
void usage(std::string prog)
{
    printf("Usage: %s <Fxx> <xC> [options]\n", prog.c_str());
    printf("Options:\n");
    printf("  --channels <channel_names>  Specify channels to enable\n");
    printf("  --config <yaml_file>        Specify config YAML file\n");
    printf("  --tv <path>                 Specify test vector base path\n");
    printf("  --lp <path>                 Specify launch pattern base path\n");
    printf("  --help, -h                  Show this help message\n");
    printf("\n");
    printf("Example: %s F08 2C --channels PDSCH+PBCH\n", prog.c_str());
}

int parse_channel_mask(char* arg, uint32_t* mask)
{
    if(arg == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: null parameter", __FUNCTION__);
        return -1;
    }
    NVLOGI_FMT(TAG, "{}: argv={}", __FUNCTION__, arg);

    if(strncmp(arg, "P", 1) == 0 || strncmp(arg, "CSI_RS", 6) == 0 || strncmp(arg, "SRS", 3) == 0)
    {
        size_t total_len = strlen(arg);
        size_t offset    = 0;
        do
        {
            for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
            {
                if(strncmp(arg + offset, get_channel_name(ch), strlen(get_channel_name(ch))) == 0)
                {
                    *mask |= 1 << ch;
                    offset += strlen(get_channel_name(ch));
                    NVLOGD_FMT(TAG, "{}: channels add ch={}: [{}]", __FUNCTION__, ch, get_channel_name(ch));
                    break;
                }
                if(ch == channel_type_t::CHANNEL_MAX - 1)
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid channel name: {}", __FUNCTION__, arg + offset);
                    return -1;
                }
            }
            offset++; // 1 character delimiter
        } while(offset < total_len);
    }
    else
    {
        char* err_ptr = NULL;
        if(strncmp(arg, "0b", 2) == 0 || strncmp(arg, "0B", 2) == 0)
        {
            *mask = strtol(arg + 2, &err_ptr, 2); // Binary
        }
        else
        {
            *mask = strtol(arg, &err_ptr, 0); // Octal, Decimal, Hex
        }

        if(err_ptr == NULL || *err_ptr != '\0')
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid channel parameter: {}", __FUNCTION__, arg);
            return -1;
        }
    }

    if(*mask >= 1 << channel_type_t::CHANNEL_MAX)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: channel out of range: 0x{:02X} > 0x{:02X}", __FUNCTION__, *mask, (1 << channel_type_t::CHANNEL_MAX) - 1);
        return -1;
    }
    return 0;
}

void RU_Emulator::get_args(int argc, char ** argv)
{
	std::string yaml_file;
    uint32_t channel_mask        = 0;
    if (argc < 3) {
        usage(std::string(argv[0]));
        exit(1);
    }

    for(int i = 1; i < argc; i++)
    {
        if(strncmp(argv[i], "F08", strlen("F08")) == 0 && i < argc - 1)
        {
            csi_rs_optimized_validation = 1;
        }

        if(strncmp(argv[i], "--channels", strlen("--channels")) == 0 && i < argc - 1)
        {
            // Parse channels
            i++;
            if(parse_channel_mask(argv[i], &channel_mask) < 0)
            {
                do_throw(sb() << "No channels specified\n");
            }
        }
        else if(strncmp(argv[i], "--config", strlen("--config")) == 0 && i < argc - 1)
        {
            // Parse channels
            i++;
            opt_config_file = std::string(argv[i]);
        }
        else if(strncmp(argv[i], "--tv", strlen("--tv")) == 0 && i < argc - 1)
        {
            // Parse TV base (full) path
            i++;
            user_defined_tv_base_path = argv[i];
        }
        else if(strncmp(argv[i], "--lp", strlen("--lp")) == 0 && i < argc - 1)
        {
            // Parse launch pattern base (full) path
            i++;
            user_defined_lp_base_path = argv[i];
        }
        else
        {
            // Parse launch pattern file name
            opt_launch_pattern_file.append("_").append(argv[i]);
        }
    }

    opt_launch_pattern_file.append(".yaml");
    NVLOGC_FMT(TAG, "Run {} {} channel_mask {}", argv[0], opt_launch_pattern_file.c_str(), channel_mask);

    // By default disable all channel
    opt_pdsch_validation = RE_DISABLED;
    opt_pbch_validation = RE_DISABLED;
    opt_pdcch_ul_validation = RE_DISABLED;
    opt_pdcch_dl_validation = RE_DISABLED;
    opt_csirs_validation = RE_DISABLED;

    opt_prach_enabled = RE_DISABLED;
    opt_srs_enabled = RE_DISABLED;
    opt_pucch_enabled = RE_DISABLED;
    opt_pusch_enabled = RE_DISABLED;

    if(channel_mask == 0)
    {
        channel_mask = (1 << channel_type_t::CHANNEL_MAX) - 1;
        opt_pdsch_validation = RE_ENABLED;
    }
    std::string enabled_channels;
    for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
    {
        if(channel_mask & (1 << ch))
        {
            switch (ch) {
            case channel_type_t::PUSCH:
                enabled_channels.append(" ").append(get_channel_name(ch));
                opt_pusch_enabled = RE_ENABLED;
                re_cons("PUSCH enabled!");
                break;
            case channel_type_t::PDSCH:
                opt_pdsch_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            case channel_type_t::PDCCH_UL:
                opt_pdcch_ul_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            case channel_type_t::PDCCH_DL:
                opt_pdcch_dl_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            case channel_type_t::PBCH:
                opt_pbch_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            case channel_type_t::PUCCH:
                enabled_channels.append(" ").append(get_channel_name(ch));
                opt_pucch_enabled = RE_ENABLED;
                re_cons("PUCCH enabled!");
                break;
            case channel_type_t::PRACH:
                enabled_channels.append(" ").append(get_channel_name(ch));
                opt_prach_enabled = RE_ENABLED;
                re_cons("PRACH enabled!");
                break;
            case channel_type_t::CSI_RS:
                opt_csirs_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            case channel_type_t::SRS:
                enabled_channels.append(" ").append(get_channel_name(ch));
                opt_srs_enabled = RE_ENABLED;
                re_cons("SRS enabled!");
                break;
            case channel_type_t::DL_BFW:
                opt_bfw_dl_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            case channel_type_t::UL_BFW:
                opt_bfw_ul_validation = RE_ENABLED;
                enabled_channels.append(" ").append(get_channel_name(ch));
                break;
            default:
                do_throw(sb() << "Invalid Channel name specified, skipping\n");
            }
        }

    }
    NVLOGC_FMT(TAG, "Enabled channels:{} | channel_mask=0x{:02X}", enabled_channels.c_str(), channel_mask);
}

void RU_Emulator::parse_yaml(std::string yaml_file)
{
    char yaml_file_full_path[MAX_PATH_LEN];
    get_full_path_file(yaml_file_full_path, CONFIG_RE_YAML_FILE_PATH, yaml_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    opt_config_file = std::string(yaml_file_full_path);
    re_cons("Config file: {}", opt_config_file.c_str());
    yaml::document doc;
    try {
        yaml::file_parser fp(yaml_file_full_path);
        doc = fp.next_document();

    } catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "YAML config file not found at " << yaml_file_full_path << "\n");
    }

    yaml::node root = doc.root();
    try
    {
        aerial::check_yaml_version(root, yaml_file_full_path);
        root = root[YAML_RU_EMULATOR];
    }
        catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "No RU Emulator configs found\n");
    }

    try
    {
        auto nic_node = root["nics"];
        for(int i = 0; i < nic_node.length(); ++i)
        {
            auto nic = static_cast<std::string>(nic_node[i][YAML_NIC_INTERFACE]);
            nic_interfaces.push_back(nic);
        }
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "NIC interfaces misconfigured!\n");
    }

    if (nic_interfaces.empty())
    {
        do_throw(sb() << "At least one NIC interface required!\n");
    }

    auto peers = root["peers"];
    for(int i = 0; i < peers.length(); ++i)
    try
    {
        auto peer = peers[i];
        yaml_assign_eth(static_cast<std::string>(peer[YAML_PEER_ETH_ADDR]), dpdk.peer_eth_addr[i]);
    }
    catch(const std::exception& e)
    {
        re_dbg("{}", e.what());
        do_throw(sb() << "Peer ethernet MAC improper or missing\n");
    }
    dpdk.num_peer_addr = peers.length();

    // try_yaml_assign_int(root, YAML_VLAN, dpdk.vlan);
    try_yaml_assign_int(root, YAML_NUM_SLOTS_UL, opt_num_slots_ul);
    try_yaml_assign_int(root, YAML_NUM_SLOTS_DL, opt_num_slots_dl);
    try_yaml_assign_int(root, YAML_TTI, opt_tti_us);
    try_yaml_assign_int(root, YAML_DL_UP_SANITY_CHECK, opt_dl_up_sanity_check);
    try_yaml_assign_int(root, YAML_MAX_SECT_STATS, opt_max_sect_stats);
    try_yaml_assign_int(root, YAML_DL_BFW_VALIDATION, opt_bfw_dl_validation);
    try_yaml_assign_int(root, YAML_UL_BFW_VALIDATION, opt_bfw_ul_validation);
    try_yaml_assign_int(root, YAML_BEAMID_VALIDATION, opt_beamid_validation);
    try_yaml_assign_int(root, YAML_SECTIONID_VALIDATION, opt_sectionid_validation);
    try_yaml_assign_int(root, YAML_TIMER_LEVEL, opt_timer_level);
    try_yaml_assign_int(root, YAML_TIMER_OFFSET_US, opt_timer_offset_us);
    try_yaml_assign_int(root, YAML_SYMBOL_OFFSET_US, opt_symbol_offset_us);
    try_yaml_assign_int(root, YAML_SEND_SLOT, opt_send_slot);

    try_yaml_assign_int(root, YAML_C_INTERVAL, opt_c_interval_us);
    try_yaml_assign_int(root, YAML_C_PLANE_PER_SYMBOL, opt_c_plane_per_symbol);
    try_yaml_assign_int(root, YAML_PRACH_C_PLANE_PER_SYMBOL, opt_prach_c_plane_per_symbol);
    try_yaml_assign_int(root, YAML_VALIDATE_TIMING, opt_validate_dl_timing);
    try_yaml_assign_int(root, YAML_DL_WARMUP_SLOTS, opt_dl_warmup_slots);
    try_yaml_assign_int(root, YAML_UL_WARMUP_SLOTS, opt_ul_warmup_slots);
    try_yaml_assign_int(root, YAML_TIMING_HISTOGRAM, opt_timing_histogram);
    try_yaml_assign_int(root, YAML_TIMING_HISTOGRAM_BIN_SIZE, opt_timing_histogram_bin_size);

    try_yaml_assign_string(root, YAML_NVLOG_NAME, opt_log_name);

    try_yaml_assign_int(root, YAML_UL_ENABLED, opt_ul_enabled);
    try_yaml_assign_int(root, YAML_PRACH_ENABLED, opt_prach_enabled);
    try_yaml_assign_int(root, YAML_SRS_ENABLED, opt_srs_enabled);
    try_yaml_assign_int(root, YAML_DL_ENABLED, opt_dl_enabled);
    try_yaml_assign_int(root, YAML_DLC_TB, opt_dlc_tb);
    try_yaml_assign_int(root, YAML_FOREVER, opt_forever);

    try_yaml_assign_int(root, YAML_LOW_PRIORITY_CORE, opt_low_priority_core);
    try_yaml_assign_int(root, YAML_MULTI_SECTION_UL, opt_multi_section_ul);

    try_yaml_assign_int(root, YAML_ENABLE_DL_PROC_MT, opt_enable_dl_proc_mt);

    try_yaml_assign_int(root, YAML_OAM_CELL_CTRL_CMD, opt_oam_cell_ctrl_cmd);
    try_yaml_assign_int(root, YAML_DL_APPROX_VALIDATION, opt_dl_approx_validation);
    selective_tv_load = (opt_oam_cell_ctrl_cmd != RE_ENABLED && opt_dl_approx_validation != RE_ENABLED);

    try_yaml_assign_int(root, YAML_ENABLE_MMIMO, opt_enable_mmimo);
    try_yaml_assign_int(root, YAML_MIN_UL_CORES_PER_CELL_MMIMO, opt_min_ul_cores_per_cell_mmimo);
    try_yaml_assign_int(root, YAML_ENABLE_BEAM_FORMING, opt_enable_beam_forming);
    try_yaml_assign_int(root, YAML_ENABLE_CPLANE_WORKER_TRACING, opt_enable_cplane_worker_tracing);
    try_yaml_assign_int(root, YAML_DROP_PACKET_EVERY_TEN_SECS, opt_drop_packet_every_ten_secs);
    try_yaml_assign_int(root, YAML_SPLIT_SRS_TXQ, opt_split_srs_txq);
    try_yaml_assign_int(root, YAML_UL_ONLY, opt_ul_only);
    try_yaml_assign_int(root, YAML_ENABLE_PRECOMPUTED_TX, opt_enable_precomputed_tx);
    try_yaml_assign_int(root, YAML_ENABLE_SRS_EAXCID_PACING, opt_enable_srs_eaxcid_pacing);
    
    // SRS Pacing parameters
    try_yaml_assign_int(root, YAML_SRS_PACING_S3_SRS_SYMBOLS, opt_srs_pacing_s3_srs_symbols);
    try_yaml_assign_int(root, YAML_SRS_PACING_S4_SRS_SYMBOLS, opt_srs_pacing_s4_srs_symbols);
    try_yaml_assign_int(root, YAML_SRS_PACING_S5_SRS_SYMBOLS, opt_srs_pacing_s5_srs_symbols);
    try_yaml_assign_int(root, YAML_SRS_PACING_EAXCIDS_PER_TX_WINDOW, opt_srs_pacing_eaxcids_per_tx_window);
    try_yaml_assign_int(root, YAML_SRS_PACING_EAXCIDS_PER_SYMBOL, opt_srs_pacing_eaxcids_per_symbol);
    
    // Calculate num_srs_txqs based on configuration
    if (opt_enable_srs_eaxcid_pacing == RE_ENABLED)
    {
        // Validate parameters
        if (opt_srs_pacing_eaxcids_per_tx_window <= 0 || opt_srs_pacing_eaxcids_per_symbol <= 0)
        {
            do_throw(sb() << "SRS pacing enabled but eaxcids_per_tx_window or eaxcids_per_symbol not properly configured\n");
        }
        
        if (opt_srs_pacing_eaxcids_per_symbol % opt_srs_pacing_eaxcids_per_tx_window != 0)
        {
            do_throw(sb() << "srs_pacing_eaxcids_per_symbol must be divisible by srs_pacing_eaxcids_per_tx_window\n");
        }
        
        // Validate symbol counts are within valid range
        if (opt_srs_pacing_s3_srs_symbols < 0 || opt_srs_pacing_s3_srs_symbols > ORAN_ALL_SYMBOLS)
        {
            do_throw(sb() << "srs_pacing_s3_srs_symbols (" << opt_srs_pacing_s3_srs_symbols << ") must be between 0 and " << ORAN_ALL_SYMBOLS << "\n");
        }
        if (opt_srs_pacing_s4_srs_symbols < 0 || opt_srs_pacing_s4_srs_symbols > ORAN_ALL_SYMBOLS)
        {
            do_throw(sb() << "srs_pacing_s4_srs_symbols (" << opt_srs_pacing_s4_srs_symbols << ") must be between 0 and " << ORAN_ALL_SYMBOLS << "\n");
        }
        if (opt_srs_pacing_s5_srs_symbols < 0 || opt_srs_pacing_s5_srs_symbols > ORAN_ALL_SYMBOLS)
        {
            do_throw(sb() << "srs_pacing_s5_srs_symbols (" << opt_srs_pacing_s5_srs_symbols << ") must be between 0 and " << ORAN_ALL_SYMBOLS << "\n");
        }
        
        const int total_srs_symbols = opt_srs_pacing_s3_srs_symbols + opt_srs_pacing_s4_srs_symbols + opt_srs_pacing_s5_srs_symbols;
        if (total_srs_symbols <= 0)
        {
            do_throw(sb() << "SRS pacing enabled but no SRS symbols configured for any slot type\n");
        }
        
        num_srs_txqs = total_srs_symbols * (opt_srs_pacing_eaxcids_per_symbol / opt_srs_pacing_eaxcids_per_tx_window);
        re_info("SRS pacing enabled: num_srs_txqs = {}", num_srs_txqs);
    }
    else
    {
        // Backward compatibility: use ORAN_ALL_SYMBOLS (14) for non-paced mode
        num_srs_txqs = ORAN_ALL_SYMBOLS;
    }

    //AERIAL-FH PARAMS
    try_yaml_assign_int(root, YAML_AFH_TXQ_SIZE, opt_afh_txq_size);
    try_yaml_assign_int(root, YAML_AFH_RXQ_SIZE, opt_afh_rxq_size);
    try_yaml_assign_int(root, YAML_AFH_TX_REQUEST_NUM, opt_afh_txq_request_num);
    try_yaml_assign_int(root, YAML_AFH_DPDK_THREAD, opt_afh_dpdk_thread);
    try_yaml_assign_int(root, YAML_AFH_PDUMP_CLIENT_THREAD, opt_afh_pdump_client_thread);
    try_yaml_assign_int(root, YAML_AFH_ACCU_TX_SCHED_RES_NS, opt_afh_accu_tx_sched_res_ns);
    try_yaml_assign_string(root, YAML_AFH_DPDK_FILE_PREFIX, opt_afh_dpdk_file_prefix);
    try_yaml_assign_int(root, YAML_AFH_PER_RXQ_MEMPOOL, opt_aerial_fh_per_rxq_mempool);
    try_yaml_assign_int(root, YAML_AFH_CPU_MBUF_POOL_SIZE_PER_RXQ, opt_afh_cpu_mbuf_pool_size_per_rxq);
    try_yaml_assign_int(root, YAML_AFH_CPU_MBUF_POOL_TX_SIZE, opt_afh_cpu_mbuf_pool_tx_size);
    try_yaml_assign_int(root, YAML_AFH_CPU_MBUF_POOL_RX_SIZE, opt_afh_cpu_mbuf_pool_rx_size);
    try_yaml_assign_int(root, YAML_AFH_SPLIT_MP, opt_afh_split_rx_tx_mp);
    try_yaml_assign_int(root, YAML_AFH_MTU, opt_afh_mtu);
#ifdef STANDALONE
    try_yaml_assign_int(root, YAML_STANDALONE_CORE_ID, opt_standalone_core_id);
#endif
    int payload_validation = 1;
    try_yaml_assign_int(root, "payload_validation", payload_validation);
    if(payload_validation == 0)
    {
        opt_pdsch_validation = RE_DISABLED;
        opt_pbch_validation = RE_DISABLED;
        opt_pdcch_ul_validation = RE_DISABLED;
        opt_pdcch_dl_validation = RE_DISABLED;
        opt_csirs_validation = RE_DISABLED;
    }
    try_yaml_assign_int(root, "debug_u_plane_prints", opt_debug_u_plane_prints);
    if(opt_debug_u_plane_prints != 0)
    {
        try_yaml_assign_int(root, "debug_u_plane_threshold", opt_debug_u_plane_threshold);
    }
    // DL C-plane window relative to OTA time -470us to -419us
    // DL U-plane window relative to OTA time -345us to -134us
    // UL C-plane window relative to OTA time -336us to -125us
    oran_timing_info.dl_c_plane_timing_delay = 470;
    oran_timing_info.dl_c_plane_window_size = 51;
    oran_timing_info.ul_c_plane_timing_delay = 336;
    oran_timing_info.ul_c_plane_window_size = 51;
    oran_timing_info.dl_u_plane_timing_delay = 345;
    oran_timing_info.dl_u_plane_window_size = 51;
    oran_timing_info.ul_u_plane_tx_offset = 50;
    oran_timing_info.ul_u_plane_tx_offset_srs = 621;

    try
    {
        auto oran_node = root["oran_timing_info"];
        try_yaml_assign_int(oran_node, "dl_c_plane_timing_delay", oran_timing_info.dl_c_plane_timing_delay);
        try_yaml_assign_int(oran_node, "dl_c_plane_window_size", oran_timing_info.dl_c_plane_window_size);
        try_yaml_assign_int(oran_node, "ul_c_plane_timing_delay", oran_timing_info.ul_c_plane_timing_delay);
        try_yaml_assign_int(oran_node, "ul_c_plane_window_size", oran_timing_info.ul_c_plane_window_size);
        try_yaml_assign_int(oran_node, "dl_u_plane_timing_delay", oran_timing_info.dl_u_plane_timing_delay);
        try_yaml_assign_int(oran_node, "dl_u_plane_window_size", oran_timing_info.dl_u_plane_window_size);
        try_yaml_assign_int(oran_node, "ul_u_plane_tx_offset", oran_timing_info.ul_u_plane_tx_offset);
        try_yaml_assign_int(oran_node, "ul_u_plane_tx_offset_srs", oran_timing_info.ul_u_plane_tx_offset_srs);
    }
        catch(const std::exception& e)
    {
        re_info("{} No ORAN timings found, using default profile", e.what());
    }

    try
    {
        auto oran_node = root["oran_beam_id_info"];
        oran_beam_id_info.static_beam_id_start = static_cast<int>(oran_node["static_beam_id_start"]);
        oran_beam_id_info.static_beam_id_end = static_cast<int>(oran_node["static_beam_id_end"]);
        oran_beam_id_info.dynamic_beam_id_start = static_cast<int>(oran_node["dynamic_beam_id_start"]);
        oran_beam_id_info.dynamic_beam_id_end = static_cast<int>(oran_node["dynamic_beam_id_end"]);

        // 1. All values must be in [1, 32767]
        if (oran_beam_id_info.static_beam_id_start < 1 || oran_beam_id_info.static_beam_id_start > 32767 ||
            oran_beam_id_info.static_beam_id_end   < 1 || oran_beam_id_info.static_beam_id_end   > 32767 ||
            oran_beam_id_info.dynamic_beam_id_start < 1 || oran_beam_id_info.dynamic_beam_id_start > 32767 ||
            oran_beam_id_info.dynamic_beam_id_end   < 1 || oran_beam_id_info.dynamic_beam_id_end   > 32767) {
            re_err(AERIAL_YAML_PARSER_EVENT, "CONFIG ERROR: All beam IDs must be in the range 1 to 32767");
            do_throw(sb() << "CONFIG ERROR: All beam IDs must be in the range 1 to 32767\n");
            return;
        }

        // 2. Ranges must be valid (start ≤ end)
        if (oran_beam_id_info.static_beam_id_start > oran_beam_id_info.static_beam_id_end) {
            re_err(AERIAL_YAML_PARSER_EVENT, "CONFIG ERROR: static_beam_id_start must be <= static_beam_id_end");
            do_throw(sb() << "CONFIG ERROR: static_beam_id_start must be <= static_beam_id_end\n");
            return;
        }
        if (oran_beam_id_info.dynamic_beam_id_start > oran_beam_id_info.dynamic_beam_id_end) {
            re_err(AERIAL_YAML_PARSER_EVENT, "CONFIG ERROR: dynamic_beam_id_start must be <= dynamic_beam_id_end");
            do_throw(sb() << "CONFIG ERROR: dynamic_beam_id_start must be <= dynamic_beam_id_end\n");
            return;
        }

        // 3. Static range must be lower and not overlap with dynamic range
        if (oran_beam_id_info.static_beam_id_end >= oran_beam_id_info.dynamic_beam_id_start) {
            re_err(AERIAL_YAML_PARSER_EVENT, "CONFIG ERROR: static beam ID range [{}, {}] overlaps with or is not lower than dynamic beam ID range [{}, {}]",
                oran_beam_id_info.static_beam_id_start, oran_beam_id_info.static_beam_id_end,
                oran_beam_id_info.dynamic_beam_id_start, oran_beam_id_info.dynamic_beam_id_end);
            do_throw(sb() << "CONFIG ERROR: static beam ID range [" << oran_beam_id_info.static_beam_id_start << ", " << oran_beam_id_info.static_beam_id_end
                          << "] overlaps with or is not lower than dynamic beam ID range ["
                          << oran_beam_id_info.dynamic_beam_id_start << ", " << oran_beam_id_info.dynamic_beam_id_end << "]\n");
            return;
        }
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        re_cons("ORAN beam id info not found, using default profile");
        //Default values
        oran_beam_id_info.static_beam_id_start = 1;
        oran_beam_id_info.static_beam_id_end = 16527;
        oran_beam_id_info.dynamic_beam_id_start = 16528;
        oran_beam_id_info.dynamic_beam_id_end = 32767;
    }

    try
    {
        auto ecpri_hdr_cfg_test = root["ecpri_hdr_cfg_test"];
        opt_ecpri_hdr_cfg_test = static_cast<int>(ecpri_hdr_cfg_test["enable"]);
        if (opt_ecpri_hdr_cfg_test)
        {
            auto ecpri_hdr_cfg_file = static_cast<std::string>(ecpri_hdr_cfg_test["ecpri_hdr_cfg_file"]);

            char ecpri_hdr_cfg_file_full_path[MAX_PATH_LEN];
            get_full_path_file(ecpri_hdr_cfg_file_full_path, CONFIG_RE_YAML_FILE_PATH, ecpri_hdr_cfg_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            yaml::document ecpri_hdr_cfg_doc;
            try
            {
                yaml::file_parser fp(ecpri_hdr_cfg_file_full_path);
                ecpri_hdr_cfg_doc = fp.next_document();
            }
            catch (const std::exception &e)
            {
                re_info("{}", e.what());
                do_throw(sb() << "YAML config file not found at " << ecpri_hdr_cfg_file_full_path << "\n");
            }

            yaml::node ecpri_hdr_cfg_node = ecpri_hdr_cfg_doc.root()["ecpri_header_config"];

            // Parse ecpriVersion
            ecpri_hdr_cfg.ecpriVersion.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriVersion"]["enable"]);
            if (ecpri_hdr_cfg.ecpriVersion.enable)
            {
                ecpri_hdr_cfg.ecpriVersion.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriVersion"]["value"]);
                if (ecpri_hdr_cfg.ecpriVersion.value > 0xF)
                {
                    throw std::runtime_error("ecpriVersion value exceeds 4-bit limit");
                }
            }

            // Parse ecpriReserved
            ecpri_hdr_cfg.ecpriReserved.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriReserved"]["enable"]);
            if (ecpri_hdr_cfg.ecpriReserved.enable)
            {
                ecpri_hdr_cfg.ecpriReserved.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriReserved"]["value"]);
                if (ecpri_hdr_cfg.ecpriReserved.value > 0x7)
                {
                    throw std::runtime_error("ecpriReserved value exceeds 3-bit limit");
                }
            }

            // Parse ecpriConcatenation
            ecpri_hdr_cfg.ecpriConcatenation.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriConcatenation"]["enable"]);
            if (ecpri_hdr_cfg.ecpriConcatenation.enable)
            {
                ecpri_hdr_cfg.ecpriConcatenation.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriConcatenation"]["value"]);
                if (ecpri_hdr_cfg.ecpriConcatenation.value > 0x1)
                {
                    throw std::runtime_error("ecpriConcatenation value exceeds 1-bit limit");
                }
            }

            // Parse ecpriMessage
            ecpri_hdr_cfg.ecpriMessage.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriMessage"]["enable"]);
            ecpri_hdr_cfg.ecpriMessage.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriMessage"]["value"]);

            // Parse ecpriPayload
            ecpri_hdr_cfg.ecpriPayload.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriPayload"]["enable"]);
            ecpri_hdr_cfg.ecpriPayload.value = static_cast<uint16_t>(ecpri_hdr_cfg_node["ecpriPayload"]["value"]);


            // Parse ecpriRtcid
            ecpri_hdr_cfg.ecpriRtcid.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriRtcid"]["enable"]);
            ecpri_hdr_cfg.ecpriRtcid.value = static_cast<uint16_t>(ecpri_hdr_cfg_node["ecpriRtcid"]["value"]);

            // Parse ecpriPcid
            ecpri_hdr_cfg.ecpriPcid.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriPcid"]["enable"]);
            ecpri_hdr_cfg.ecpriPcid.value = static_cast<uint16_t>(ecpri_hdr_cfg_node["ecpriPcid"]["value"]);

            if (ecpri_hdr_cfg.ecpriRtcid.enable && ecpri_hdr_cfg.ecpriPcid.enable) {
                throw std::runtime_error("Cannot enable both ecpriRtcid and ecpriPcid");
            }

            // Parse ecpriSeqid
            ecpri_hdr_cfg.ecpriSeqid.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriSeqid"]["enable"]);
            ecpri_hdr_cfg.ecpriSeqid.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriSeqid"]["value"]);

            // Parse ecpriEbit
            ecpri_hdr_cfg.ecpriEbit.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriEbit"]["enable"]);
            if (ecpri_hdr_cfg.ecpriEbit.enable)
            {
                ecpri_hdr_cfg.ecpriEbit.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriEbit"]["value"]);
                if (ecpri_hdr_cfg.ecpriEbit.value > 0x1)
                {
                    throw std::runtime_error("ecpriEbit value exceeds 1-bit limit");
                }
            }

            // Parse ecpriSubSeqid
            ecpri_hdr_cfg.ecpriSubSeqid.enable = static_cast<int>(ecpri_hdr_cfg_node["ecpriSubSeqid"]["enable"]);
            if (ecpri_hdr_cfg.ecpriSubSeqid.enable)
            {
                ecpri_hdr_cfg.ecpriSubSeqid.value = static_cast<uint8_t>(ecpri_hdr_cfg_node["ecpriSubSeqid"]["value"]);
                if (ecpri_hdr_cfg.ecpriSubSeqid.value > 0x7F)
                {
                    throw std::runtime_error("ecpriSubSeqid value exceeds 7-bit limit");
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "Error parsing ecpri_hdr_cfg_test\n");
    }

    //Assign core-list
    try
    {
        yaml_assign_core_list(root, ul_core_list, true,false);
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "Error in UL core list parsing\n");
    }

    // Assign SRS core list (required when SRS is enabled)
    try
    {
        yaml_assign_core_list(root, ul_srs_core_list, YAML_UL_SRS_CORE_LIST);
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        // SRS core list is optional when SRS Channel is disabled - validation happens later when enable_srs is known
    }

    try
    {
        yaml_assign_core_list(root, dl_core_list, false,true);
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "Error in DL core list parsing\n");
    }

    if(opt_enable_dl_proc_mt){
        try
        {
            yaml_assign_core_list(root, dl_rx_core_list, false,false);
        }
        catch(const std::exception& e)
        {
            re_info("{}", e.what());
            do_throw(sb() << "Error in DL Rx core list parsing\n");
        }
    }

    //Need to parse Launch pattern first for the num_cells
    try
    {
        char launch_pattern_full_path[MAX_PATH_LEN];
        if (user_defined_lp_base_path.empty()) {
            get_full_path_file(launch_pattern_full_path, CONFIG_LAUNCH_PATTERN_PATH, opt_launch_pattern_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        } else {
            std::string full_path = user_defined_lp_base_path + opt_launch_pattern_file;
            std::strncpy(launch_pattern_full_path, full_path.c_str(), sizeof(launch_pattern_full_path));
        }
        opt_launch_pattern_file = std::string(launch_pattern_full_path);
        re_cons("Launch pattern file: {}", opt_launch_pattern_file.c_str());
    }
    catch(const std::exception& e)
    {
        do_throw(sb() << "Launch pattern required!\n");
    }

    try
    {
        parse_launch_pattern(opt_launch_pattern_file);
    }
    catch(const std::exception& e)
    {
        re_err(AERIAL_YAML_PARSER_EVENT,"{}", e.what());
        do_throw(sb() << "Error in launch pattern parsing: " << opt_launch_pattern_file << "\n");
    }

    try
    {
        yaml_assign_cell_configs(root, cell_configs, opt_num_cells, opt_launch_pattern_file);
        for (auto &cc : cell_configs)
        {
            if (cc.dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)
            {
                opt_mod_comp_enabled = RE_ENABLED;
            }
            else
            {
                opt_non_mod_comp_enabled = RE_ENABLED;
            }
        }
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "Error in cell config parsing\n");
    }

    try
    {
        cell_configs_from_lp_yaml(opt_launch_pattern_file);
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        do_throw(sb() << "Error in cell config parsing from lp yaml file\n");
    }
    
    // Initialize SRS slot info if pacing is enabled
    if (opt_enable_srs_eaxcid_pacing == RE_ENABLED)
    {
        initialize_srs_slot_info();
    }

}

void RU_Emulator::initialize_srs_slot_info()
{
    const int txqs_per_symbol = opt_srs_pacing_eaxcids_per_symbol / opt_srs_pacing_eaxcids_per_tx_window;
    const int64_t slot_duration_ns = opt_tti_us * NS_X_US;
    
    // Slot type 3 (reference slot)
    srs_s3_info.num_symbols = opt_srs_pacing_s3_srs_symbols;
    srs_s3_info.first_symbol = ORAN_ALL_SYMBOLS - opt_srs_pacing_s3_srs_symbols;
    srs_s3_info.txq_base_offset = 0;
    srs_s3_info.slot_time_offset_ns = 0;  // s3 is the reference
    
    // Slot type 4 (1 slot before s3)
    srs_s4_info.num_symbols = opt_srs_pacing_s4_srs_symbols;
    srs_s4_info.first_symbol = ORAN_ALL_SYMBOLS - opt_srs_pacing_s4_srs_symbols;
    srs_s4_info.txq_base_offset = opt_srs_pacing_s3_srs_symbols * txqs_per_symbol;
    srs_s4_info.slot_time_offset_ns = -slot_duration_ns;  // 1 slot before s3
    
    // Slot type 5 (2 slots before s3)
    srs_s5_info.num_symbols = opt_srs_pacing_s5_srs_symbols;
    srs_s5_info.first_symbol = ORAN_ALL_SYMBOLS - opt_srs_pacing_s5_srs_symbols;
    srs_s5_info.txq_base_offset = (opt_srs_pacing_s3_srs_symbols + opt_srs_pacing_s4_srs_symbols) * txqs_per_symbol;
    srs_s5_info.slot_time_offset_ns = -2 * slot_duration_ns;  // 2 slots before s3
    
    re_info("SRS slot info initialized:");
    re_info("  s3: first_symbol={}, num_symbols={}, txq_base_offset={}, slot_time_offset_ns={}",
            srs_s3_info.first_symbol, srs_s3_info.num_symbols, srs_s3_info.txq_base_offset, srs_s3_info.slot_time_offset_ns);
    re_info("  s4: first_symbol={}, num_symbols={}, txq_base_offset={}, slot_time_offset_ns={}",
            srs_s4_info.first_symbol, srs_s4_info.num_symbols, srs_s4_info.txq_base_offset, srs_s4_info.slot_time_offset_ns);
    re_info("  s5: first_symbol={}, num_symbols={}, txq_base_offset={}, slot_time_offset_ns={}",
            srs_s5_info.first_symbol, srs_s5_info.num_symbols, srs_s5_info.txq_base_offset, srs_s5_info.slot_time_offset_ns);
}

void RU_Emulator::parse_launch_pattern_channel(yaml::node& root, std::string key, tv_object& tv_object)
{
    yaml::node sched = root[YAML_LP_SCHED];
    yaml_assign_launch_pattern_tv(root, key, tv_object.tv_names, tv_object.tv_map);
    yaml_assign_launch_pattern(sched, key, tv_object.launch_pattern, tv_object.tv_map, opt_num_cells);
}


void RU_Emulator::launch_pattern_v2_tv_pre_processing(yaml::node& root)
{
   try
   {
       for(int slot_idx = 0; slot_idx < root.length(); ++slot_idx)
       {
           yaml::node config_node = root[slot_idx][YAML_LP_CONFIG];
           std::string node_type = config_node.type_string();
           if(node_type == "YAML_SCALAR_NODE")
           {
               continue;
           }

           for(int cell_idx = 0; cell_idx < std::min(opt_num_cells, (int)config_node.length()); ++cell_idx)
           {
               yaml::node cell_node = config_node[cell_idx];
               yaml::node channels_node = cell_node[YAML_LP_CHANNELS];
               std::string node_type = channels_node.type_string();
               if(node_type == "YAML_SCALAR_NODE")
               {
                   continue;
               }

               for(int ch_idx = 0; ch_idx < channels_node.length(); ++ch_idx)
               {

                    std::string tv_name = static_cast<std::string>(channels_node[ch_idx]);
                    if(tv_to_channel_map.find(tv_name) != tv_to_channel_map.end())
                    {
                        continue;
                    }
                    char tv_full_path[MAX_PATH_LEN];
                    if (user_defined_tv_base_path.empty()) {
                        get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, tv_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
                    } else {
                        std::string full_path = user_defined_tv_base_path + tv_name;
                        std::strncpy(tv_full_path, full_path.c_str(), sizeof(tv_full_path));
                    }
                    std::string tv_path = std::string(tv_full_path);

                    if (access(tv_full_path, F_OK) != 0)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "File {} does not exist", tv_path);
                        exit(1);
                    }

                    hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(tv_path.c_str());
                    int count = 1;
                    std::string pdu = "PDU";
                    while(1)
                    {
                       std::string dset_string = pdu + std::to_string(count);
                       if(!hdf5file.is_valid_dataset(dset_string.c_str()))
                       {
                            break;
                       }
                       hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
                       hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];

                       uint8_t channel_type = pdu_pars["type"].as<uint8_t>();
                       std::string channel_string = "NONE";

                       switch (channel_type)
                       {
                       case nrsim_tv_type::SSB:
                           channel_string = dl_channel_string[dl_channel::PBCH];
                           break;
                       case nrsim_tv_type::PDCCH:
                           channel_string = pdu_pars["dciUL"].as<uint8_t>() ? dl_channel_string[dl_channel::PDCCH_UL] : dl_channel_string[dl_channel::PDCCH_DL];
                           break;
                       case nrsim_tv_type::PDSCH:
                           channel_string = dl_channel_string[dl_channel::PDSCH];
                           break;
                       case nrsim_tv_type::CSI_RS:
                           channel_string = dl_channel_string[dl_channel::CSI_RS];
                           break;
                       case nrsim_tv_type::BFW:
                           channel_string = pdu_pars["bfwUL"].as<uint8_t>() ? dl_channel_string[dl_channel::BFW_UL] : dl_channel_string[dl_channel::BFW_DL];
                           break;
                       case nrsim_tv_type::PRACH:
                           channel_string = ul_channel_string[ul_channel::PRACH];
                           break;
                       case nrsim_tv_type::SRS:
                           channel_string = ul_channel_string[ul_channel::SRS];
                           break;
                       case nrsim_tv_type::PUCCH:
                           channel_string = ul_channel_string[ul_channel::PUCCH];
                           break;
                       case nrsim_tv_type::PUSCH:
                           channel_string = ul_channel_string[ul_channel::PUSCH];
                           break;
                       }

                       if(channel_to_tv_map[channel_string].find(tv_name) == channel_to_tv_map[channel_string].end())
                       {
                          int size = channel_to_tv_map[channel_string].size();
                          std::string short_name = "TV" + std::to_string(size+1);
                          channel_to_tv_map[channel_string].insert({tv_name, short_name});
                       }

                       tv_to_channel_map[tv_name].insert(channel_string);
                       count++;
                   }
               }
           }
       }
   }
   catch(const std::exception& e)
   {
       re_cons("{} Detected Launch pattern v2 not compatible, using v1 parsing",e.what());
   }
}

inline int16_t convert_static_beam_weight_endian(int16_t value) {
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return static_cast<int16_t>(__builtin_bswap16(static_cast<uint16_t>(value))); // Swap on little endian
    #else
        return value;  // No swap needed on big endian
    #endif
}

int RU_Emulator::parse_dbt_configs(hdf5hpp::hdf5_file& file, int cell_id) {

    double num_static_beamIdx = 0.0;
    double num_TRX_beamforming = 0.0;
    double enable_static_dynamic_beamforming = 0.0;

    if (!file.is_valid_dataset("enable_static_dynamic_beamforming") ||
        !file.is_valid_dataset("num_static_beamIdx") ||
        !file.is_valid_dataset("num_TRX_beamforming"))
    {
        return -1;
    }

    auto hdf5dset0{file.open_dataset("enable_static_dynamic_beamforming")};
    hdf5dset0.read(&enable_static_dynamic_beamforming);

    auto hdf5dset1{file.open_dataset("num_static_beamIdx")};
    hdf5dset1.read(&num_static_beamIdx);

    auto hdf5dset2{file.open_dataset("num_TRX_beamforming")};
    hdf5dset2.read(&num_TRX_beamforming);

    NVLOGC_FMT(TAG, "parse_dbt_configs enable_static_dynamic_beamforming {} num_static_beamIdx {} num_TRX_beamforming {}", enable_static_dynamic_beamforming, num_static_beamIdx, num_TRX_beamforming);
    dbt_md_t dbt_conf{!!static_cast<uint>(enable_static_dynamic_beamforming), static_cast<uint16_t>(num_static_beamIdx), static_cast<uint16_t>(num_TRX_beamforming), dbt_data_t{}};

    hdf5hpp::hdf5_dataset coef_re = file.open_dataset("DBT_real");
    hdf5hpp::hdf5_dataset coef_im = file.open_dataset("DBT_imag");

    auto count = dbt_conf.num_static_beamIdx * dbt_conf.num_TRX_beamforming;
    std::vector<int16_t> re_data(count);
    std::vector<int16_t> im_data(count);
    coef_re.read(re_data.data());
    coef_im.read(im_data.data());
    dbt_conf.dbt_data_buf.resize(count);
    for(int j = 0; j < count; j++)
    {
        dbt_conf.dbt_data_buf[j] = {convert_static_beam_weight_endian(re_data[j]), convert_static_beam_weight_endian(im_data[j])};
    }

#if 0
    for (int i = 0; i < dbt_conf.num_static_beamIdx; i++)
    {
        for (int j = 0; j < dbt_conf.num_TRX_beamforming; j++)
        {
            NVLOGC_FMT(TAG, "beam id {}, re {} im {}", i + 1, dbt_conf.dbt_data_buf[i * dbt_conf.num_TRX_beamforming + j].re, dbt_conf.dbt_data_buf[i * dbt_conf.num_TRX_beamforming + j].im);
        }
    }
#endif

    cell_configs[cell_id].dbt_cfg = std::move(dbt_conf);

    return 0;
}

int RU_Emulator::load_h5_config_params(int cell_id, const char* config_params_h5_file)
{
    char h5path[MAX_PATH_LEN];
    if (user_defined_tv_base_path.empty()) {
        get_full_path_file(h5path, CONFIG_TEST_VECTOR_PATH, config_params_h5_file, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    } else {
        std::string full_path = user_defined_tv_base_path + config_params_h5_file;
        std::strncpy(h5path, full_path.c_str(), sizeof(h5path));
    }
    if(access(h5path, F_OK) != 0)
    {
        if (user_defined_lp_base_path.empty()) {
            get_full_path_file(h5path, CONFIG_LAUNCH_PATTERN_PATH, config_params_h5_file, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        } else {
            std::string full_path = user_defined_lp_base_path + config_params_h5_file;
            std::strncpy(h5path, full_path.c_str(), sizeof(h5path));
        }
        if(access(h5path, F_OK) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Cell_Configs {} file not exist", config_params_h5_file);
            exit(1);
        }
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path, e.what());
        return -1;
    }

    if(hdf5file.is_valid_dataset("Cell_Config"))
    {
        hdf5hpp::hdf5_dataset      dset         = hdf5file.open_dataset("Cell_Config");
        hdf5hpp::hdf5_dataset_elem dset_elem    = dset[0];
        cell_configs[cell_id].dlGridSize                = dset_elem["dlGridSize"].as<unsigned int>();
        cell_configs[cell_id].ulGridSize                = dset_elem["ulGridSize"].as<unsigned int>();
        cell_configs[cell_id].dlBandwidth               = dset_elem["dlBandwidth"].as<unsigned int>();
        cell_configs[cell_id].ulBandwidth               = dset_elem["ulBandwidth"].as<unsigned int>();
        NVLOGI_FMT(TAG, "cell_id={}: dlGridSize={}, ulGridSize={} dlBandwidth={}, ulBandwidth={}", cell_id,
            cell_configs[cell_id].dlGridSize, cell_configs[cell_id].ulGridSize, cell_configs[cell_id].dlBandwidth, cell_configs[cell_id].ulBandwidth);
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: Cell_Config not found in TV {}", __FUNCTION__, h5path);
        return -1;
    }
    parse_dbt_configs(hdf5file, cell_id);
    return 0;
}

void RU_Emulator::parse_launch_pattern(std::string yaml_file)
{
    yaml::file_parser fp(yaml_file.c_str());
    yaml::document doc = fp.next_document();
    yaml::node root = doc.root();

    if(root.has_key("beam_ids"))
    {
        opt_beamforming = RE_ENABLED;
    }

    if(root.has_key(YAML_LP_NUM_CELLS))
    {
        try_yaml_assign_int(root, YAML_LP_NUM_CELLS, opt_num_cells);
    }
    else
    {
        opt_num_cells = root[YAML_LP_CELL_CONFIGS].length();
    }

    if (opt_num_cells < 1 || opt_num_cells > MAX_CELLS_PER_SLOT)
    {
        do_throw(sb() << "opt_num_cells (" << opt_num_cells
                       << ") out of valid range [1, " << MAX_CELLS_PER_SLOT << "]\n");
    }

    yaml::node sched = root[YAML_LP_SCHED];

    opt_launch_pattern_version = root.has_key(YAML_LP_TV) ? 1 : 2;

    auto t1 = get_ns();
    launch_pattern_v2_tv_pre_processing(sched);
    auto t2 = get_ns();
    NVLOGC_FMT(TAG, "Launch Pattern v2 preprocessing finished in {:.2}s", ((double)(t2 - t1))/NS_X_S);
    parse_launch_pattern_channel(root, YAML_LP_PUSCH, pusch_object);
    parse_launch_pattern_channel(root, YAML_LP_PRACH, prach_object);
    parse_launch_pattern_channel(root, YAML_LP_PDSCH, pdsch_object);
    parse_launch_pattern_channel(root, YAML_LP_PBCH, pbch_object);
    parse_launch_pattern_channel(root, YAML_LP_PDCCH_UL, pdcch_ul_object);
    parse_launch_pattern_channel(root, YAML_LP_PDCCH_DL, pdcch_dl_object);
    parse_launch_pattern_channel(root, YAML_LP_PUCCH, pucch_object);
    parse_launch_pattern_channel(root, YAML_LP_SRS, srs_object);
    parse_launch_pattern_channel(root, YAML_LP_CSIRS, csirs_object);
    parse_launch_pattern_channel(root, YAML_LP_BFW_DL, bfw_dl_object);
    parse_launch_pattern_channel(root, YAML_LP_BFW_UL, bfw_ul_object);
    auto t3 = get_ns();
    NVLOGC_FMT(TAG, "Launch Pattern parsing finished in {:.2}s", ((double)(t3 - t2))/NS_X_S);
    launch_pattern_slot_size = sched.length();

    try
    {
        yaml::node init = root[YAML_LP_INIT];
        yaml_assign_launch_pattern(init, YAML_LP_PUSCH, pusch_object.init_launch_pattern, pusch_object.tv_map, opt_num_cells);
        yaml_assign_launch_pattern(init, YAML_LP_PRACH, prach_object.init_launch_pattern, prach_object.tv_map, opt_num_cells);
        yaml_assign_launch_pattern(init, YAML_LP_SRS, srs_object.init_launch_pattern, srs_object.tv_map, opt_num_cells);
        yaml_assign_launch_pattern(init, YAML_LP_PDSCH, pdsch_object.init_launch_pattern, pdsch_object.tv_map, opt_num_cells);
        yaml_assign_launch_pattern(init, YAML_LP_PBCH, pbch_object.init_launch_pattern, pbch_object.tv_map, opt_num_cells);
        yaml_assign_launch_pattern(init, YAML_LP_PDCCH_UL, pdcch_ul_object.init_launch_pattern, pdcch_ul_object.tv_map, opt_num_cells);
        yaml_assign_launch_pattern(init, YAML_LP_PDCCH_DL, pdcch_dl_object.init_launch_pattern, pdcch_dl_object.tv_map, opt_num_cells);

        for(int i = 0; i < launch_pattern_slot_size; ++i)
        {
            for(int cell_idx = 0; cell_idx < MAX_CELLS_PER_SLOT; ++cell_idx)
            {
                pusch_object.initialization_phase[cell_idx].store(0);
                pusch_object.init_slot_counters[cell_idx].store(0);
                prach_object.initialization_phase[cell_idx].store(0);
                prach_object.init_slot_counters[cell_idx].store(0);
                srs_object.initialization_phase[cell_idx].store(0);
                srs_object.init_slot_counters[cell_idx].store(0);
                pbch_object.initialization_phase[cell_idx].store(0);
                pbch_object.init_slot_counters[cell_idx].store(0);
                pdcch_ul_object.initialization_phase[cell_idx].store(0);
                pdcch_ul_object.init_slot_counters[cell_idx].store(0);
                pdcch_dl_object.initialization_phase[cell_idx].store(0);
                pdcch_dl_object.init_slot_counters[cell_idx].store(0);
            }
        }

        for(auto& slot_pattern: pusch_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    pusch_object.initialization_phase[cell_to_tv.first].store(1);
                    ++pusch_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }
        for(auto& slot_pattern: prach_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    prach_object.initialization_phase[cell_to_tv.first].store(1);
                    ++prach_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }
        for(auto& slot_pattern: srs_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    srs_object.initialization_phase[cell_to_tv.first].store(1);
                    ++srs_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }
        for(auto& slot_pattern: pdsch_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    pdsch_object.initialization_phase[cell_to_tv.first].store(1);
                    ++pdsch_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }
        for(auto& slot_pattern: pbch_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    pbch_object.initialization_phase[cell_to_tv.first].store(1);
                    ++pbch_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }

        for(auto& slot_pattern: pdcch_dl_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    pdcch_dl_object.initialization_phase[cell_to_tv.first].store(1);
                    ++pdcch_dl_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }

        for(auto& slot_pattern: pdcch_ul_object.init_launch_pattern)
        {
            if(slot_pattern.size() > 0)
            {
                for(auto& cell_to_tv: slot_pattern)
                {
                    pdcch_ul_object.initialization_phase[cell_to_tv.first].store(1);
                    ++pdcch_ul_object.init_slot_counters[cell_to_tv.first];
                }
            }
        }
    }
    catch(const std::exception& e)
    {
        re_cons("{}, Init launch pattern missing, using only normal launch pattern", e.what());
    }
}

std::string log_level_string(int level)
{
    switch(level)
    {
        case NVLOG_NONE:
            return std::string("NVLOG_NONE");
        case NVLOG_ERROR:
            return std::string("NVLOG_ERROR");
        case NVLOG_CONSOLE:
            return std::string("NVLOG_CONSOLE");
        case NVLOG_WARN:
            return std::string("NVLOG_WARN");
        case NVLOG_INFO:
            return std::string("NVLOG_INFO");
        case NVLOG_DEBUG:
            return std::string("NVLOG_DEBUG");
        case NVLOG_VERBOSE:
            return std::string("NVLOG_VERBOSE");
        default:
            return std::string("INVALID");
    }
}

void RU_Emulator::verify_and_apply_configs()
{
    verify_configs();
    apply_configs();
}

void RU_Emulator::apply_configs()
{
    max_slot_id = (US_X_MS / opt_tti_us) - 1;
    slots_per_second = US_X_S / opt_tti_us;

    if(opt_dl_enabled == RE_DISABLED)
    {
        opt_pdsch_validation = RE_DISABLED;
        opt_pbch_validation = RE_DISABLED;
        opt_pdcch_ul_validation = RE_DISABLED;
        opt_pdcch_dl_validation = RE_DISABLED;
    }

    if(prach_object.tv_names.size() == 0)
    {
        re_cons("No PRACH TVs provided, disabling PRACH channel");
        opt_prach_enabled = RE_DISABLED;
    }

    if(pucch_object.tv_names.size() == 0)
    {
        re_cons("No PUCCH TVs provided, disabling PUCCH channel");
        opt_pucch_enabled = RE_DISABLED;
    }

    if(srs_object.tv_names.size() == 0)
    {
        re_cons("No SRS TVs provided, disabling SRS channel");
        opt_srs_enabled = RE_DISABLED;
    }

    if(pdsch_object.tv_names.size() == 0)
    {
        re_cons("No PDSCH TVs provided, disabling PDSCH Validatio");
        opt_pdsch_validation = RE_DISABLED;
    }
    if(pbch_object.tv_names.size() == 0)
    {
        re_cons("No PBCH TVs provided, disabling PBCH Validatio");
        opt_pbch_validation = RE_DISABLED;
    }
    if(pdcch_ul_object.tv_names.size() == 0)
    {
        re_cons("No PDCCH_UL TVs provided, disabling PDCCH_UL Validatio");
        opt_pdcch_ul_validation = RE_DISABLED;
    }
    if(pdcch_dl_object.tv_names.size() == 0)
    {
        re_cons("No PDCCH_DL TVs provided, disabling PDCCH_DL Validatio");
        opt_pdcch_dl_validation = RE_DISABLED;
    }
    if(csirs_object.tv_names.size() == 0)
    {
        re_cons("No CSI-RS TVs provided, disabling CSI-RS Validatio");
        opt_csirs_validation = RE_DISABLED;
    }

    if(bfw_dl_object.tv_names.size() == 0)
    {
        re_cons("No BFW DL TVs provided, disabling BFW DL Validation\n");
        opt_bfw_dl_validation = RE_DISABLED;
    }

    if(bfw_ul_object.tv_names.size() == 0)
    {
        re_cons("No BFW UL TVs provided, disabling BFW UL Validation\n");
        opt_bfw_ul_validation = RE_DISABLED;
    }

    section_id_trackers = std::make_unique<SectionIdTrackerStorage>();
    sectionid_validation_init();
}

void RU_Emulator::verify_configs()
{
    if (opt_num_cells < 1 || opt_num_cells > MAX_CELLS_PER_SLOT)
    {
        do_throw(sb() << "opt_num_cells (" << opt_num_cells
                       << ") out of valid range [1, " << MAX_CELLS_PER_SLOT << "]\n");
    }

    int pdsch_slots = 0;
    int pusch_slots = 0;
    int prach_slots = 0;
    int pucch_slots = 0;
    int srs_slots = 0;
    int pbch_slots = 0;
    int pdcch_ul_slots = 0;
    int pdcch_dl_slots = 0;
    int csirs_slots = 0;
    int bfw_dl_slots = 0;
    int bfw_ul_slots = 0;

    for(int i = 0; i < pusch_object.launch_pattern.size(); ++i)
    {
        if(pusch_object.launch_pattern[i].size() > 0)
        {
            ++pusch_slots;
        }
        if(prach_object.launch_pattern[i].size() > 0)
        {
            ++prach_slots;
        }
        if(pucch_object.launch_pattern[i].size() > 0)
        {
            ++pucch_slots;
        }
        if(srs_object.launch_pattern[i].size() > 0)
        {
            ++srs_slots;
        }
    }

    for(int i = 0; i < pdsch_object.launch_pattern.size(); ++i)
    {
        if(pdsch_object.launch_pattern[i].size() > 0)
        {
            ++pdsch_slots;
        }

        if(pbch_object.launch_pattern[i].size() > 0)
        {
            ++pbch_slots;
        }

        if(pdcch_ul_object.launch_pattern[i].size() > 0)
        {
            ++pdcch_ul_slots;
        }
        if(pdcch_dl_object.launch_pattern[i].size() > 0)
        {
            ++pdcch_dl_slots;
        }
        if(csirs_object.launch_pattern[i].size() > 0)
        {
            ++csirs_slots;
        }
        if(bfw_dl_object.launch_pattern[i].size() > 0)
        {
            ++bfw_dl_slots;
        }
        if(bfw_ul_object.launch_pattern[i].size() > 0)
        {
            ++bfw_ul_slots;
        }

    }
    if(pusch_slots == 0 && prach_slots == 0 && srs_slots == 0 && pucch_slots == 0 && pdsch_slots == 0 && pbch_slots == 0 && pdcch_ul_slots == 0 && pdcch_dl_slots == 0 && csirs_slots == 0 && bfw_dl_slots == 0 && bfw_ul_slots == 0)
    {
        do_throw(sb() << "Launch pattern has no channels scheduled!\n");
    }

    // if(pusch_slots == 0 && prach_slots == 0 && pucch_slots == 0 && opt_beamforming == 0)
    // {
    //     opt_num_slots_ul = 0;
    //     opt_ul_enabled = RE_DISABLED;
    //     re_cons("No UL slots scheduled in launch pattern, disabling ul_enabled flag and setting num_slots_ul to 0");
    // }
    if(pdsch_slots == 0 && pbch_slots == 0 && pdcch_dl_slots == 0 && pdcch_ul_slots == 0 && csirs_slots == 0)
    {
        opt_num_slots_dl = 0;
        opt_dl_enabled = RE_DISABLED;
        re_cons("No DL slots scheduled in launch pattern, disabling dl_enabled flag and setting num_slots_dl to 0 ");
    }

    if(launch_pattern_slot_size % ORAN_MAX_SLOT_X_SUBFRAME_ID != 0)
    {
        do_throw(sb() << "Launch pattern size is not a multiple of 20, observed: "<< launch_pattern_slot_size << "\n");
    }

    if(opt_send_slot != RE_ENABLED && opt_send_slot != RE_DISABLED)
    {
        do_throw(sb() << "send slot flag can only be 0 or 1, read value: " << opt_send_slot << "\n");
    }

    if(opt_ul_enabled != RE_ENABLED && opt_ul_enabled != RE_DISABLED)
    {
        do_throw(sb() << "UL enabled flag can only be 0 or 1, read value: " << opt_ul_enabled << "\n");
    }

    if(opt_dl_enabled != RE_ENABLED && opt_dl_enabled != RE_DISABLED)
    {
        do_throw(sb() << "DL enabled flag can only be 0 or 1, read value: " << opt_dl_enabled << "\n");
    }

    if(opt_dl_enabled == RE_DISABLED && opt_num_slots_dl != 0)
    {
        re_cons("DL is disabled, setting num_slots_dl to 0");
        opt_num_slots_dl = 0;
    }

    if(opt_dl_enabled == RE_ENABLED && opt_num_slots_dl == 0)
    {
        re_cons("Num_dl_slots is 0, disabling dl_enabled flag");
        opt_dl_enabled = RE_DISABLED;
    }

    if(opt_ul_enabled == RE_DISABLED && opt_dl_enabled == RE_DISABLED)
    {
        do_throw(sb() << "UL and DL both disabled! What to do?\n");
    }

    if(opt_forever == RE_ENABLED)
    {
        re_cons("Forever is enabled, ignoring num_slots_ul and num_slots_dl");
    }
}

void print_launch_pattern_slot(tv_object& tv_object, uint8_t slot)
{
    auto& cells = tv_object.launch_pattern[slot];
    if(cells.size() != 0) {
        re_info("\t{}: ", tv_object.channel_string.c_str());
        char buffer[MAX_PRINT_LOG_LENGTH];
        int buffer_index = 0;
        for( auto& cell: cells)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "Cell %d, TV %d ", cell.first, cell.second);
        }
        re_info("\t\t{}", buffer);
    }
}

void print_init_launch_pattern_slot(tv_object& tv_object, uint8_t slot)
{
    auto& cells = tv_object.init_launch_pattern[slot];
    if(cells.size() != 0) {
        re_info("\t{}: ", tv_object.channel_string.c_str());
        for( auto& cell: cells)
        {
            re_info("\t\tCell {}, TV {}", cell.first, cell.second);
        }
    }
}

#define PDU_INFO_TABLE_START_INDEX 8
void print_single_pdu_config(char* buffer, int buffer_index, pdu_info& pdu_info, channel_type_t channel)
{
    while(buffer_index < PDU_INFO_TABLE_START_INDEX)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %8u ",  pdu_info.startSym);
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %6u ",  pdu_info.numSym);
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %8u ",  pdu_info.startPrb);
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %6u ",  pdu_info.numPrb);
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %8u ",  pdu_info.numFlows);

    if(channel == channel_type_t::PUCCH)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11u ",  pdu_info.freqHopFlag);
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %13u ",  pdu_info.secondHopPrb);
    }
    if(channel == channel_type_t::PDCCH_DL || channel == channel_type_t::PDCCH_UL)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %20lX ",  pdu_info.freqDomainResource);
    }

    if(channel == channel_type_t::PDSCH)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %12u ",  pdu_info.startDataSym);
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10u ",  pdu_info.numDataSym);
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| ");
        for(int i = 0; i < pdu_info.flow_indices.size(); ++i)
        {
            buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "%u ", pdu_info.flow_indices[i]);
        }
        // buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "\n");
    }
    else
    {
        // buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|\n");
    }
    NVLOGI_FMT(TAG_TV_CONFIGS,"{}",buffer);
}

void print_pdu_configs(tv_info& tv_info, channel_type_t channel)
{
    char buffer[MAX_PRINT_LOG_LENGTH];
    int buffer_index = 0;
    while(buffer_index < PDU_INFO_TABLE_START_INDEX + 9)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, " ");
    }
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %8s ",  "startSym");
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %6s ",  "numSym");
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %8s ",  "startPrb");
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %6s ",  "numPrb");
    buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %8s ",  "numFlows");

    if(channel == channel_type_t::PUCCH)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %11s ",  "freqHopFlag");
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %13s ",  "secondHopPrb");
    }

    if(channel == channel_type_t::PDCCH_DL || channel == channel_type_t::PDCCH_UL)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %20s ",  "freqDomRes");
    }

    if(channel == channel_type_t::PDSCH)
    {
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %12s ",  "startDataSym");
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %10s ",  "numDataSym");
        buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "| %9s ",  "Flow Indices");
        // buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "\n");
    }
    else
    {
        // buffer_index += snprintf(buffer + buffer_index, MAX_PRINT_LOG_LENGTH - buffer_index, "|\n");
    }
    NVLOGI_FMT(TAG_TV_CONFIGS, "{}",buffer);

    if(channel == channel_type_t::SRS)
    {
        bool seen[OFDM_SYMBOLS_PER_SLOT][MAX_NUM_PRBS_PER_SYMBOL][MAX_NUM_PRBS_PER_SYMBOL]{};
        for(int frame = 0; frame < ORAN_MAX_FRAME_ID; frame++)
        {
            bool frame_print = false;
            for(int slot = 0; slot < SLOT_3GPP; slot++)
            {
                for(int i = 0; i < tv_info.fss_pdu_infos[frame][slot].size(); ++i)
                {
                    auto startSym = tv_info.fss_pdu_infos[frame][slot][i].startSym;
                    auto startPrb = tv_info.fss_pdu_infos[frame][slot][i].startPrb;
                    auto numPrb = tv_info.fss_pdu_infos[frame][slot][i].numPrb;
                    if(!seen[startSym][startPrb][numPrb])
                    {
                        if(!frame_print)
                        {
                            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tReference frame id: {}", frame);
                            frame_print = true;
                        }
                        buffer_index = 0;
                        std::string base = "\t\tPDU " + std::to_string(i);
                        buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", base.c_str());
                        print_single_pdu_config(buffer, buffer_index, tv_info.fss_pdu_infos[frame][slot][i], channel);
                        seen[startSym][startPrb][numPrb] = true;
                    }
                }
            }
        }
    }
    else
    {
        for(int i = 0; i < tv_info.pdu_infos.size(); ++i)
        {
            buffer_index = 0;
            std::string base = "\t\tPDU " + std::to_string(i);
            buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", base.c_str());
            print_single_pdu_config(buffer, buffer_index, tv_info.pdu_infos[i], channel);
        }
    }
}

void RU_Emulator::print_configs()
{
    NVLOGC_FMT(TAG, "================================================================================================");
    NVLOGC_FMT(TAG, "\tConfig file used: {}", opt_config_file.c_str());
    NVLOGC_FMT(TAG, "\tLaunch Pattern file used: {}", opt_launch_pattern_file.c_str());
    NVLOGC_FMT(TAG, "================================================================================================");
    std::string core_list_str = "";
    for (int i: ul_core_list) {
        core_list_str += std::to_string(i) + ' ';
    }
    NVLOGI_FMT(TAG, "\tUL Core list: {}", core_list_str.c_str());

    core_list_str = "";
    for (int i: ul_srs_core_list) {
        core_list_str += std::to_string(i) + ' ';
    }
    NVLOGI_FMT(TAG, "\tUL SRS Core list: {}", core_list_str.c_str());

    core_list_str = "";
    for (int i: dl_core_list) {
        core_list_str += std::to_string(i) + ' ';
    }
    NVLOGI_FMT(TAG, "\tDL Core list: {}", core_list_str.c_str());

    NVLOGI_FMT(TAG, "\t NICs:");
    for(const auto& nic_interface: nic_interfaces)
    {
        NVLOGI_FMT(TAG, "\t- {}",
            nic_interface.c_str()
        );
    }

    NVLOGI_FMT(TAG, "\tPeer addresses:");
    for(int i = 0; i < dpdk.num_peer_addr; ++i)
    {
        NVLOGI_FMT(TAG, "\t\t- Peer ethernet address={:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            dpdk.peer_eth_addr[i].addr_bytes[0], dpdk.peer_eth_addr[i].addr_bytes[1],
            dpdk.peer_eth_addr[i].addr_bytes[2], dpdk.peer_eth_addr[i].addr_bytes[3],
            dpdk.peer_eth_addr[i].addr_bytes[4], dpdk.peer_eth_addr[i].addr_bytes[5]
        );
    }

    NVLOGI_FMT(TAG, "\tUL enabled: {}", opt_ul_enabled == RE_ENABLED ? "ENABLED" : "DISABLED");
    if(opt_ul_enabled == RE_ENABLED)
    {
        NVLOGI_FMT(TAG, "\t\tUL send slot by slot enabled: {}", opt_send_slot == RE_ENABLED ? "ENABLED" : "DISABLED");
    }
    NVLOGI_FMT(TAG, "\tDL enabled: {}", opt_dl_enabled == RE_ENABLED ? "ENABLED" : "DISABLED");
    if(opt_dl_enabled == RE_ENABLED)
    {
        NVLOGI_FMT(TAG, "\t\tPDSCH validation channel: {}", opt_pdsch_validation == RE_ENABLED ? "ENABLED" : "DISABLED");
        NVLOGI_FMT(TAG, "\t\tPBCH validation channel: {}", opt_pbch_validation == RE_ENABLED ? "ENABLED" : "DISABLED");
        NVLOGI_FMT(TAG, "\t\tPDCCH_UL validation channel: {}", opt_pdcch_ul_validation == RE_ENABLED ? "ENABLED" : "DISABLED");
        NVLOGI_FMT(TAG, "\t\tPDCCH_DL validation channel: {}", opt_pdcch_dl_validation == RE_ENABLED ? "ENABLED" : "DISABLED");

        NVLOGI_FMT(TAG, "\t\tDL Timing Validation enabled: {}", opt_validate_dl_timing == RE_ENABLED ? "ENABLED" : "DISABLED");
    }
    NVLOGI_FMT(TAG, "\tForever enabled: {}", opt_forever == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tTiming Histogram: {}", opt_timing_histogram == RE_ENABLED ? "ENABLED" : "DISABLED");

    if(opt_timing_histogram == RE_ENABLED)
    {
        NVLOGI_FMT(TAG, "\tTiming histogram bin size: {}", opt_timing_histogram_bin_size);
    }
    if(opt_forever == RE_DISABLED)
    {
        if(opt_ul_enabled == RE_ENABLED)
        {
            NVLOGI_FMT(TAG, "\tNum slots UL: {}", opt_num_slots_ul);
        }
        if(opt_dl_enabled == RE_ENABLED)
        {
            NVLOGI_FMT(TAG, "\tNum slots DL: {}", opt_num_slots_dl);
        }
    }
    NVLOGI_FMT(TAG, "\tDL Approx Validation: {}", opt_dl_approx_validation == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tmMIMO : {}", opt_enable_mmimo == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tMin UL Cores Per Cell (mMIMO): {}", opt_min_ul_cores_per_cell_mmimo);
    NVLOGI_FMT(TAG, "\tPre-computed TX : {}", opt_enable_precomputed_tx == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tBeam forming : {}", opt_enable_beam_forming == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tCPlane Worker Tracing : {}", opt_enable_cplane_worker_tracing == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tDL Drop packet every 10 seconds: {}", opt_drop_packet_every_ten_secs == RE_ENABLED ? "ENABLED" : "DISABLED");
    NVLOGI_FMT(TAG, "\tNum flows per DL thread: {}", opt_num_flows_per_dl_thread);

    NVLOGI_FMT(TAG, "\topt_afh_txq_size: {}", opt_afh_txq_size);
    NVLOGI_FMT(TAG, "\topt_afh_rxq_size: {}", opt_afh_rxq_size);
    NVLOGI_FMT(TAG, "\topt_afh_txq_request_num: {}", opt_afh_txq_request_num);
    NVLOGI_FMT(TAG, "\topt_afh_dpdk_thread: {}", opt_afh_dpdk_thread);
    NVLOGI_FMT(TAG, "\topt_afh_pdump_client_thread: {}", opt_afh_pdump_client_thread);
    NVLOGI_FMT(TAG, "\topt_afh_accu_tx_sched_res_ns: {}", opt_afh_accu_tx_sched_res_ns);
    NVLOGI_FMT(TAG, "\topt_afh_mtu: {}", opt_afh_mtu);
    NVLOGI_FMT(TAG, "\topt_afh_split_rx_tx_mp: {}", opt_afh_split_rx_tx_mp);
    NVLOGI_FMT(TAG, "\topt_debug_u_plane_prints: {}", opt_debug_u_plane_prints);
    NVLOGI_FMT(TAG, "\topt_debug_u_plane_threshold: {}", opt_debug_u_plane_threshold);
    NVLOGI_FMT(TAG, "\topt_dl_up_sanity_check: {}", opt_dl_up_sanity_check);
    NVLOGI_FMT(TAG, "\topt_max_sect_stats: {}", opt_max_sect_stats);
    NVLOGI_FMT(TAG, "\topt_bfw_dl_validation: {}", opt_bfw_dl_validation);
    NVLOGI_FMT(TAG, "\topt_bfw_ul_validation: {}", opt_bfw_ul_validation);
    NVLOGI_FMT(TAG, "\topt_beamid_validation: {}", opt_beamid_validation);
    NVLOGI_FMT(TAG, "\topt_sectionid_validation: {}", opt_sectionid_validation);

    NVLOGI_FMT(TAG_TV_CONFIGS, "\tPUSCH TVs:");
    for (int i = 0; i < pusch_object.tv_names.size(); ++i)
    {
        auto& tv = pusch_object.tv_names[i];
        auto& tv_info = pusch_object.tv_info[i];
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t{}", tv.c_str());
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumPDUs {}", tv_info.pdu_infos.size());
        print_pdu_configs(tv_info, channel_type_t::PUSCH);
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumSections {}", tv_info.numSections);
    }
    NVLOGI_FMT(TAG_TV_CONFIGS, "\tPRACH TVs:");
    for (int i = 0; i < prach_object.tv_names.size(); ++i)
    {
        auto& tv = prach_object.tv_names[i];
        auto& tv_info = prach_object.tv_info[i];
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());

        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\tnumPDUs {}", tv_info.pdu_infos.size());
        print_pdu_configs(tv_info, channel_type_t::PRACH);
    }

    NVLOGI_FMT(TAG_TV_CONFIGS, "\tPUCCH TVs:");
    for (int i = 0; i < pucch_object.tv_names.size(); ++i)
    {
        auto& tv = pucch_object.tv_names[i];
        auto& tv_info = pucch_object.tv_info[i];
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\tnumPDUs {}", tv_info.pdu_infos.size());
        print_pdu_configs(tv_info,channel_type_t::PUCCH);
    }

    NVLOGI_FMT(TAG_TV_CONFIGS, "\tSRS TVs:");
    for (int i = 0; i < srs_object.tv_names.size(); ++i)
    {
        auto& tv = srs_object.tv_names[i];
        auto& tv_info = srs_object.tv_info[i];
        NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
        //NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\tnumPDUs {}", tv_info.pdu_infos.size());
        print_pdu_configs(tv_info,channel_type_t::SRS);
    }

    if(opt_pdsch_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tPDSCH TVs:");
        for (int i = 0; i < pdsch_object.tv_names.size(); ++i)
        {
            auto& tv = pdsch_object.tv_names[i];
            auto& tv_info = pdsch_object.tv_info[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumPDUs {}", tv_info.pdu_infos.size());
            print_pdu_configs(tv_info, channel_type_t::PDSCH);
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumPDUs combined {}", tv_info.combined_pdu_infos.size());
            for(int i = 0; i < tv_info.combined_pdu_infos.size(); ++i)
            {
                char buffer[MAX_PRINT_LOG_LENGTH];
                std::string base = "\t\tPDU " + std::to_string(i);
                int buffer_index = snprintf(buffer, MAX_PRINT_LOG_LENGTH, "%s", base.c_str());
                print_single_pdu_config(buffer, buffer_index, tv_info.combined_pdu_infos[i], channel_type_t::PDSCH);
            }

            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tTot numPrb {}", tv_info.numPrb);

            if(tv_info.hasZPCsirsPdu)
            {
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tHas Non-overlapping ZP-CSIRS");
            }
            NVLOGI_FMT(TAG_TV_CONFIGS, "");
        }
    }

    if(opt_pbch_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tPBCH TVs:");
        for (int i = 0; i < pbch_object.tv_names.size(); ++i)
        {
            auto& tv = pbch_object.tv_names[i];
            auto& tv_info = pbch_object.tv_info[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumPDUs {}", tv_info.pdu_infos.size());
            print_pdu_configs(tv_info, channel_type_t::PBCH);
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tTot numPrb {}", tv_info.numPrb);
            NVLOGI_FMT(TAG_TV_CONFIGS, "");
        }
    }
    if(opt_pdcch_ul_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tPDCCH UL TVs:");
        for (int i = 0; i < pdcch_ul_object.tv_names.size(); ++i)
        {
            auto& tv = pdcch_ul_object.tv_names[i];
            auto& tv_info = pdcch_ul_object.tv_info[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumPDUs {}", tv_info.pdu_infos.size());
            print_pdu_configs(tv_info, channel_type_t::PDCCH_UL);
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tTot numPrb {}", tv_info.numPrb);
            NVLOGI_FMT(TAG_TV_CONFIGS, "");
        }
    }

    if(opt_pdcch_dl_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tPDCCH DL TVs:");
        for (int i = 0; i < pdcch_dl_object.tv_names.size(); ++i)
        {
            auto& tv = pdcch_dl_object.tv_names[i];
            auto& tv_info = pdcch_dl_object.tv_info[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tnumPDUs {}", tv_info.pdu_infos.size());
            print_pdu_configs(tv_info, channel_type_t::PDCCH_DL);
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tTot numPrb {}", tv_info.numPrb);
            NVLOGI_FMT(TAG_TV_CONFIGS, "");
        }
    }

    if(opt_csirs_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tCSI_RS TVs:");
        for (int i = 0; i < csirs_object.tv_names.size(); ++i)
        {
            auto& tv = csirs_object.tv_names[i];
            auto& tv_info = csirs_object.tv_info[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{}", tv.c_str());
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\tZP CSI_RS: {}", (tv_info.isZP) ? 1 : 0);
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\tcsirsNumREs: {}", tv_info.csirsNumREs);
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\tcsirsNumREsSkipped (Overlapping CSI_RS): {}", tv_info.csirsNumREsSkipped);

            auto prbs_per_symbol = tv_info.nPrbDlBwp;
            for(int re_i = 0; re_i < tv_info.csirsNumREs; ++re_i)
            {
                int flow = tv_info.csirsREsToValidate[re_i][0] / (ORAN_ALL_SYMBOLS * prbs_per_symbol * PRB_NUM_RE);
                int symbol = (tv_info.csirsREsToValidate[re_i][0] / (prbs_per_symbol * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS;
                int prb = (tv_info.csirsREsToValidate[re_i][0] / PRB_NUM_RE) % prbs_per_symbol;
                re_dbg("\t\t\t\t- {} Flow {} Symbol {} Prb {}", tv_info.csirsREsToValidate[re_i][0], flow, symbol, prb);
                re_dbg("\t\t\t\t- {} Flow {} Symbol {} Prb {} Consecutive REs {}", tv_info.csirsREsToValidate[re_i][0], flow, symbol, prb, tv_info.csirsREsToValidate[re_i][1]);
            }
            for(int re_i = 0; re_i < tv_info.csirsSkippedREs.size(); ++re_i)
            {
                int flow = tv_info.csirsSkippedREs[re_i] / (ORAN_ALL_SYMBOLS * prbs_per_symbol * PRB_NUM_RE);
                int symbol = (tv_info.csirsSkippedREs[re_i] / (prbs_per_symbol * PRB_NUM_RE)) % ORAN_ALL_SYMBOLS;
                int prb = (tv_info.csirsSkippedREs[re_i] / PRB_NUM_RE) % prbs_per_symbol;
                re_dbg("\t\t\t\t- {} Skipped Flow {} Symbol {} Prb {}", tv_info.csirsSkippedREs[re_i], flow, symbol, prb);
            }
            //Optimization
            if(csi_rs_optimized_validation)
            {
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\tCSIRS Optimizations to create a bounding box");
                print_pdu_configs(tv_info, channel_type_t::CSI_RS);
            }
            NVLOGI_FMT(TAG_TV_CONFIGS, "");
        }
    }

    if(opt_bfw_dl_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tBFW DL TVs:");
        for (int i = 0; i < bfw_dl_object.tv_names.size(); ++i)
        {
            auto& tv = bfw_dl_object.tv_names[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{} num BFW PDUs: {}", tv.c_str(), bfw_dl_object.tv_info[i].bfw_infos.size());
            for(const auto& bfw_info: bfw_dl_object.tv_info[i].bfw_infos)
            {
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t-");
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "prgSize", bfw_info.prgSize);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "rbStart", bfw_info.rbStart);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "rbSize", bfw_info.rbSize);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "numPRGs", bfw_info.numPRGs);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "compressBitWidth", bfw_info.compressBitWidth);
            }
        }
    }

    if(opt_bfw_ul_validation)
    {
        NVLOGI_FMT(TAG_TV_CONFIGS, "\tBFW UL TVs:");
        for (int i = 0; i < bfw_ul_object.tv_names.size(); ++i)
        {
            auto& tv = bfw_ul_object.tv_names[i];
            NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t{} num BFW PDUs: {}", tv.c_str(), bfw_ul_object.tv_info[i].bfw_infos.size());
            for(const auto& bfw_info: bfw_ul_object.tv_info[i].bfw_infos)
            {
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t-");
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "prgSize", bfw_info.prgSize);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "rbStart", bfw_info.rbStart);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "rbSize", bfw_info.rbSize);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "numPRGs", bfw_info.numPRGs);
                NVLOGI_FMT(TAG_TV_CONFIGS, "\t\t\t{}: {}", "compressBitWidth", bfw_info.compressBitWidth);
            }
        }
    }

    NVLOGI_FMT(TAG, "\tTimer log level: {}, TTI: {}us, Num cells: {}", opt_timer_level, opt_tti_us, opt_num_cells);
    NVLOGI_FMT(TAG, "\tTimer core offset {}us", opt_timer_offset_us);

    NVLOGI_FMT(TAG, "Cell configurations:");
    for (auto cell: cell_configs)
    {
        NVLOGI_FMT(TAG, "\t\tCell name: {} Cell Eth Addr: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            cell.name.c_str(),
            cell.eth_addr.addr_bytes[0], cell.eth_addr.addr_bytes[1],
            cell.eth_addr.addr_bytes[2], cell.eth_addr.addr_bytes[3],
            cell.eth_addr.addr_bytes[4], cell.eth_addr.addr_bytes[5]);

        NVLOGI_FMT(TAG, "\t\t{} eAxC UL: ", cell.num_ul_flows);
        for(auto & eAxC: cell.eAxC_UL)
        {
            NVLOGI_FMT(TAG, "\t\t- {}", eAxC);
        }
        NVLOGI_FMT(TAG, "\t\t{} eAxC DL: ", cell.num_dl_flows);
        for(auto & eAxC: cell.eAxC_DL)
        {
            NVLOGI_FMT(TAG, "\t\t- {}", eAxC);
        }
        NVLOGI_FMT(TAG, "\t\t{} PRACH eAxC Flows: ", cell.eAxC_PRACH_list.size());
        for(auto & flow: cell.eAxC_PRACH_list)
        {
            NVLOGI_FMT(TAG, "\t\t- {}", flow);
        }
        NVLOGI_FMT(TAG, "\t\t{} SRS eAxC Flows: ", cell.eAxC_SRS_list.size());
        for(auto & flow: cell.eAxC_SRS_list)
        {
            NVLOGI_FMT(TAG, "\t\t- {}", flow);
        }

        NVLOGI_FMT(TAG, "\t\tRU_TYPE: {}", cell.ru_type);
        NVLOGI_FMT(TAG, "\t\tDL compression method: {} bit width: {} PRB size: {}", (int)cell.dl_comp_meth, cell.dl_bit_width, cell.dl_prb_size);
        NVLOGI_FMT(TAG, "\t\tUL compression method: {} bit width: {} PRB size: {}", (int)cell.ul_comp_meth, cell.ul_bit_width, cell.ul_prb_size);
        NVLOGI_FMT(TAG, "\t\tVLAN: {}", cell.vlan);
        NVLOGI_FMT(TAG, "\t\tPeer Index: {}", cell.peer_index);
        NVLOGI_FMT(TAG, "\t\tPeer Eth Addr: {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            dpdk.peer_eth_addr[cell.peer_index].addr_bytes[0], dpdk.peer_eth_addr[cell.peer_index].addr_bytes[1],
            dpdk.peer_eth_addr[cell.peer_index].addr_bytes[2], dpdk.peer_eth_addr[cell.peer_index].addr_bytes[3],
            dpdk.peer_eth_addr[cell.peer_index].addr_bytes[4], dpdk.peer_eth_addr[cell.peer_index].addr_bytes[5]);
        NVLOGI_FMT(TAG, "\t\tNic Index: {}", cell.nic_index);
        NVLOGI_FMT(TAG, "\t\tNIC Eth Addr: {}", nic_interfaces[cell.nic_index].c_str());
    }
    NVLOGI_FMT(TAG, "\t\tORAN Timing:");
    NVLOGI_FMT(TAG, "\t\t\tdl_c_plane_timing_delay: {}", oran_timing_info.dl_c_plane_timing_delay);
    NVLOGI_FMT(TAG, "\t\t\tdl_c_plane_window_size: {}", oran_timing_info.dl_c_plane_window_size);
    NVLOGI_FMT(TAG, "\t\t\tul_c_plane_timing_delay: {}", oran_timing_info.ul_c_plane_timing_delay);
    NVLOGI_FMT(TAG, "\t\t\tul_c_plane_window_size: {}", oran_timing_info.ul_c_plane_window_size);
    NVLOGI_FMT(TAG, "\t\t\tdl_u_plane_timing_delay: {}", oran_timing_info.dl_u_plane_timing_delay);
    NVLOGI_FMT(TAG, "\t\t\tdl_u_plane_window_size: {}", oran_timing_info.dl_u_plane_window_size);
    NVLOGI_FMT(TAG, "\t\t\tul_u_plane_offset: {}", oran_timing_info.ul_u_plane_tx_offset);
    NVLOGI_FMT(TAG, "\t\t\tul_u_plane_offset_srs: {}", oran_timing_info.ul_u_plane_tx_offset_srs);

    NVLOGI_FMT(TAG, "\t\t{}", "ORAN beam id info:");
    NVLOGI_FMT(TAG, "\t\t\tstatic_beam_id_start: {}", oran_beam_id_info.static_beam_id_start);
    NVLOGI_FMT(TAG, "\t\t\tstatic_beam_id_end: {}", oran_beam_id_info.static_beam_id_end);
    NVLOGI_FMT(TAG, "\t\t\tdynamic_beam_id_start: {}", oran_beam_id_info.dynamic_beam_id_start);
    NVLOGI_FMT(TAG, "\t\t\tdynamic_beam_id_end: {}", oran_beam_id_info.dynamic_beam_id_end);
}

void yaml_assign_eth(std::string eth, struct oran_ether_addr& addr)
{
    std::string tmp;
    char * pch;
    uint32_t index=0;

    if(!eth.empty())
    {
        tmp.assign(eth);
        pch = strtok((char*)tmp.c_str(),":");
        index=0;
        while (pch != NULL)
        {
            addr.addr_bytes[index] = std::stoi(pch, 0, 16);
            index++;
            pch = strtok(NULL, ":");
        }
    }
}

void yaml_assign_tv(yaml::node root, std::string key, std::vector<std::string>& tvs)
{
    yaml::node tv_list = root[key.c_str()];

    for(int i = 0; i < tv_list.length(); ++i)
    {
        tvs.push_back(static_cast<std::string>(tv_list[i]));
    }
}

void RU_Emulator::yaml_assign_launch_pattern_tv(yaml::node root, std::string key, std::vector<std::string>& tvs, std::unordered_map<std::string, int>& tv_map)
{
    try
    {
        std::string tv_short_name;
        std::string tv_path;
        if(opt_launch_pattern_version == 2)
        {
            re_dbg("launch pattern v2 TV list {} {}", key.c_str(), channel_to_tv_map[key].size());
            for(auto tv_p: channel_to_tv_map[key])
            {
                tv_short_name = tv_p.second;
                tv_path = tv_p.first;

                char tv_full_path[MAX_PATH_LEN];
                if (user_defined_tv_base_path.empty()) {
                    get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, tv_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
                } else {
                    std::string full_path = user_defined_tv_base_path + tv_path;
                    std::strncpy(tv_full_path, full_path.c_str(), sizeof(tv_full_path));
                }
                tv_path = std::string(tv_full_path);

                tv_map[tv_short_name] = tvs.size();
                tvs.push_back(tv_path);
            }
        }
        else
        {
            yaml::node tv_list = root[YAML_LP_TV][key.c_str()];

            re_dbg("TV list {} {}", key.c_str(), tv_list.length());
            for(int i = 0; i < tv_list.length(); ++i)
            {
                tv_short_name = static_cast<std::string>(tv_list[i][YAML_LP_NAME]);
                tv_path = static_cast<std::string>(tv_list[i][YAML_LP_PATH]);

                char tv_full_path[MAX_PATH_LEN];
                if (user_defined_tv_base_path.empty()) {
                    get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, tv_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
                } else {
                    std::string full_path = user_defined_tv_base_path + tv_path;
                    std::strncpy(tv_full_path, full_path.c_str(), sizeof(tv_full_path));
                }
                tv_path = std::string(tv_full_path);

                tv_map[tv_short_name] = tvs.size();
                tvs.push_back(tv_path);
            }
        }
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        re_info("Exception parsing TVs for {}, assuming no TVs found for channel {}", key.c_str(), key.c_str());
        return;
    }
}

void RU_Emulator::cell_configs_from_lp_yaml(std::string lp_yaml_file)
{
    yaml::file_parser fp(lp_yaml_file.c_str());
    yaml::document doc = fp.next_document();
    yaml::node lp_root = doc.root();

    if(lp_root.has_key("Cell_Configs"))
    {
        yaml::node cell_configs_list = lp_root["Cell_Configs"];
        if(cell_configs_list.length() != opt_num_cells)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: cell num doesn't match: cell_num={} config_list={}", __FUNCTION__, opt_num_cells, cell_configs_list.length());
        }
        for(int cell_id = 0; cell_id < opt_num_cells; cell_id++)
        {
            std::string config_tv = cell_configs_list[cell_id].as<std::string>();
            if(load_h5_config_params(cell_id, config_tv.c_str()) < 0)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: Parsing cell configs failure", __FUNCTION__);
            }
        }
    }
    else
    {
        for(int cell_id = 0; cell_id < opt_num_cells; cell_id++)
        {
            std::string config_params_file = lp_root["Cells"].as<std::string>();
            if(load_h5_config_params(cell_id, config_params_file.c_str()) < 0)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: Parsing cell configs failure", __FUNCTION__);
            }
        }
    }

}
void yaml_assign_cell_configs(yaml::node root, std::vector<struct cell_config>& cell_configs, int& num_cells, std::string lp_yaml_file)
{
    yaml::node cell_list = root[YAML_CELL_CONFIGS];
    std::string name;
    std::string eth;
    int flow;
    int dl_comp_meth;
    int ul_comp_meth;
    if(cell_list.length() < num_cells)
    {
        do_throw(sb() << "Number of cells defined in config file: " << cell_list.length() << " is less than the cells indicated in launch pattern: " << num_cells << ".\n");
    }



    for(int i = 0; i < num_cells; ++i)
    {
        struct cell_config cc{};
        try
        {
            yaml::node eAxC_UL = cell_list[i]["eAxC_UL"];
            for(int j = 0; j < eAxC_UL.length(); ++j)
            {
                int eAxC = static_cast<int>(eAxC_UL[j]);
                cc.eAxC_UL.push_back(eAxC);
            }
            cc.num_ul_flows = cc.eAxC_UL.size();
        }
        catch(const std::exception& e)
        {
            re_info("Exception parsing eAxC_UL");
        }
        try
        {
            yaml::node eAxC_DL = cell_list[i]["eAxC_DL"];
            for(int j = 0; j < eAxC_DL.length(); ++j)
            {
                int eAxC = static_cast<int>(eAxC_DL[j]);
                cc.eAxC_DL.push_back(eAxC);
            }
            cc.num_dl_flows = cc.eAxC_DL.size();
        }
        catch(const std::exception& e)
        {
            re_info("Exception parsing eAxC_UL");
        }

        yaml::node eAxC_PRACH_list = cell_list[i][YAML_CELL_EAXC_PRACH_LIST];
        for(int j = 0; j < eAxC_PRACH_list.length(); ++j)
        {
            flow = static_cast<int>(eAxC_PRACH_list[j]);
            cc.eAxC_PRACH_list.push_back(flow);
        }
        cc.num_valid_PRACH_flows = cc.eAxC_PRACH_list.size();

        yaml::node eAxC_SRS_list = cell_list[i][YAML_CELL_EAXC_SRS_LIST];
        for(int j = 0; j < eAxC_SRS_list.length(); ++j)
        {
            flow = static_cast<int>(eAxC_SRS_list[j]);
            cc.eAxC_SRS_list.push_back(flow);
        }
        cc.num_valid_SRS_flows = cc.eAxC_SRS_list.size();

        name = static_cast<std::string>(cell_list[i][YAML_CELL_NAME]);
        eth = static_cast<std::string>(cell_list[i][YAML_CELL_ETH]);

        cc.ru_type = static_cast<int>(cell_list[i][YAML_CELL_RU_TYPE]);
        cc.dl_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(static_cast<int>(cell_list[i][YAML_CELL_DL_IQ_DATA_FMT][YAML_CELL_COMP_METH]));
        cc.ul_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(static_cast<int>(cell_list[i][YAML_CELL_UL_IQ_DATA_FMT][YAML_CELL_COMP_METH]));
        cc.dl_bit_width = static_cast<int>(cell_list[i][YAML_CELL_DL_IQ_DATA_FMT][YAML_CELL_BIT_WIDTH]);
        cc.ul_bit_width = static_cast<int>(cell_list[i][YAML_CELL_UL_IQ_DATA_FMT][YAML_CELL_BIT_WIDTH]);
        cc.fs_offset_dl = static_cast<int>(cell_list[i][YAML_CELL_FS_OFFSET_DL]);
        cc.exponent_dl = static_cast<int>(cell_list[i][YAML_CELL_EXPONENT_DL]);
        cc.ref_dl = static_cast<int>(cell_list[i][YAML_CELL_REF_DL]);

        cc.peer_index = static_cast<int>(cell_list[i]["peer"]);
        cc.nic_index = static_cast<int>(cell_list[i]["nic"]);
        cc.vlan = static_cast<int>(cell_list[i][YAML_VLAN]) | (static_cast<int>(cell_list[i][YAML_PCP]) << 13);

        if (cc.dl_comp_meth == aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT)
        {
            switch (cc.dl_bit_width)
            {
            case BFP_NO_COMPRESSION:
                cc.dl_prb_size = PRB_SIZE_16F;
                break;
            case BFP_COMPRESSION_9_BITS:
                cc.dl_prb_size = PRB_SIZE_9F;
                break;
            case BFP_COMPRESSION_14_BITS:
                cc.dl_prb_size = PRB_SIZE_14F;
                break;
            default:
                break;
            }
        }
        else if (cc.dl_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            cc.dl_prb_size = PRB_SIZE_16F;
        }

        if (cc.ul_comp_meth ==  aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT)
        {
            switch (cc.ul_bit_width)
            {
            case BFP_NO_COMPRESSION:
                cc.ul_prb_size = PRB_SIZE_16F;
                break;
            case BFP_COMPRESSION_9_BITS:
                cc.ul_prb_size = PRB_SIZE_9F;
                break;
            case BFP_COMPRESSION_14_BITS:
                cc.ul_prb_size = PRB_SIZE_14F;
                break;
            default:
                break;
            }
        }
        else if (cc.ul_comp_meth == aerial_fh::UserDataCompressionMethod::NO_COMPRESSION)
        {
            cc.ul_prb_size = PRB_SIZE_16F;
        }
        yaml_assign_eth(eth, cc.eth_addr);

        for(auto& x: eth) x = toupper(x);
        cc.name = name;

        if(static_cast<int>(root[YAML_FIX_BETA_DL]))
        {
            if(cc.dl_bit_width == BFP_COMPRESSION_9_BITS)
            {
                cc.beta_dl = 65536;
            }
            else if(cc.dl_bit_width == BFP_COMPRESSION_14_BITS)
            {
                cc.beta_dl = 2097152;
            }
        }
        else
        {
            float numerator, sqrt_fs0, fs;
            sqrt_fs0       = pow(2., cc.dl_bit_width - 1) * pow(2., pow(2., cc.exponent_dl) - 1);
            fs             = sqrt_fs0 * sqrt_fs0 * pow(2., -1 * cc.fs_offset_dl);
            numerator      = fs * pow(10., cc.ref_dl / 10.);
            cc.numerator   = numerator;
            // Note beta_dl = sqrt(numerator / denominator) where denominator = 2.0 * 12 * nPrbDlBwp
            cc.beta_dl     = BETA_DL_NOT_SET; // use it as a flag that it needs to be set later, once the denominator is computed
        }
        cell_configs.emplace_back(cc);
    }
}

void yaml_assign_core_list(yaml::node root, std::vector<int>& core_list, bool UL,bool DL_proc)
{
    yaml::node config_core_list = UL ? root[YAML_UL_CORE_LIST] : (DL_proc? root[YAML_DL_CORE_LIST]:root[YAML_DL_RX_CORE_LIST]);

    for(int i = 0; i < config_core_list.length(); ++i)
    {
        int core = static_cast<int>(config_core_list[i]);
        core_list.push_back(core);
    }
}

void yaml_assign_core_list(yaml::node root, std::vector<int>& core_list, const char* key)
{
    yaml::node config_core_list = root[key];

    for(int i = 0; i < config_core_list.length(); ++i)
    {
        int core = static_cast<int>(config_core_list[i]);
        core_list.push_back(core);
    }
}

void RU_Emulator::yaml_assign_launch_pattern(yaml::node root, std::string channel, launch_pattern_matrix& launch_pattern, std::unordered_map<std::string, int>& tv_map, int num_cells)
{
    std::string tv;
    std::string channel_type;
    int cell_index;
    int slot_num;
    int tv_index;
    try
    {
        int root_length = root.length();

        // Early return if root is empty to avoid division by zero
        if (root_length == 0)
        {
            re_cons("Launch pattern root is empty");
            return;
        }

        launch_pattern = launch_pattern_matrix(root_length);

        for(int slot_idx = 0; slot_idx < root_length; ++slot_idx)
        {
            slot_num = static_cast<int>(root[slot_idx][YAML_LP_SLOT]);
            yaml::node config_node = root[slot_idx][YAML_LP_CONFIG];
            std::string node_type = config_node.type_string();
            if(node_type == "YAML_SCALAR_NODE")
            {
                continue;
            }


            for(int cell_idx = 0; cell_idx < std::min(num_cells, (int)config_node.length()); ++cell_idx)
            {
                yaml::node cell_node = config_node[cell_idx];
                cell_index = static_cast<int>(cell_node[YAML_LP_CELL_INDEX]);
                yaml::node channels_node = cell_node[YAML_LP_CHANNELS];
                std::string node_type = channels_node.type_string();
                if(node_type == "YAML_SCALAR_NODE")
                {
                    continue;
                }

                for(int ch_idx = 0; ch_idx < channels_node.length(); ++ch_idx)
                {
                    if(opt_launch_pattern_version == 2)
                    {
                        std::string tv_name = static_cast<std::string>(channels_node[ch_idx]);
                        for(auto channel_type : tv_to_channel_map[tv_name])
                        {
                            if(channel_type == channel)
                            {
                                tv_index = tv_map[channel_to_tv_map[channel_type][tv_name]];
                                if(channel == YAML_LP_BFW_DL || channel == YAML_LP_BFW_UL)
                                {
                                    launch_pattern[(slot_num + 1) % root_length][cell_index] = tv_index;
                                }
                                else
                                {
                                    launch_pattern[slot_num][cell_index] = tv_index;
                                }
                            }
                        }
                    }
                    else
                    {
                        channel_type = static_cast<std::string>(channels_node[ch_idx][YAML_LP_CHANNEL_TYPE]);
                        if(channels_node[ch_idx].has_key(YAML_LP_CHANNEL_BEAM_IDS))
                        {
                            opt_beamforming = RE_ENABLED;
                        }
                        if(channel_type == channel)
                        {
                            tv_index = tv_map[static_cast<std::string>(channels_node[ch_idx][YAML_LP_CHANNEL_TV])];

                            if(channel == YAML_LP_BFW_DL || channel == YAML_LP_BFW_UL)
                            {
                                launch_pattern[(slot_num + 1) % root_length][cell_index] = tv_index;
                            }
                            else
                            {
                                launch_pattern[slot_num][cell_index] = tv_index;
                            }
                        }
                    }

                }
            }
        }
    }
    catch(const std::exception& e)
    {
        re_cons("{} Launch pattern error",e.what());
    }

}

void try_yaml_assign_list(yaml::node& parent, std::string key, yaml::node& dest)
{
    try
    {
        auto tmp = parent[key.c_str()];
        dest = tmp;
        return;
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        return;
    }
}

void try_yaml_assign_float(yaml::node& parent, std::string key, float& dest)
{
    try
    {
        float tmp = static_cast<float>(parent[key.c_str()]);
        dest = tmp;
        return;
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        NVLOGI_FMT(TAG,"Keeping default {} = {}", key.c_str(), dest);
        return;
    }
}

void try_yaml_assign_int(yaml::node& parent, std::string key, int& dest)
{
    try
    {
        int tmp = static_cast<int>(parent[key.c_str()]);
        dest = tmp;
        return;
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        re_cons("Keeping default {} = {}", key.c_str(), dest);
        return;
    }
}

void try_yaml_assign_string(yaml::node& parent, std::string key, std::string& dest)
{
    try
    {
        std::string tmp = static_cast<std::string>(parent[key.c_str()]);
        dest = tmp;
        return;
    }
    catch(const std::exception& e)
    {
        re_info("{}", e.what());
        re_info("Keeping default {} = {}", key.c_str(), dest.c_str());
        return;
    }
}
