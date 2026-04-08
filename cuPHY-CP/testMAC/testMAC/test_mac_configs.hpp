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

#ifndef _TEST_MAC_CONFIGS_HPP
#define _TEST_MAC_CONFIGS_HPP

#include "yaml.hpp"
#include "nv_phy_utils.hpp"
#include "common_utils.hpp"
#include <vector>

#ifdef AERIAL_CUMAC_ENABLE
#include "cumac_configs.hpp"
#endif

/**
 * PRACH conformance test parameters
 */
typedef struct
{
  int expect_prmbindexes;      //!< Expected PRACH preamble indexes
  float delay_error_tolerance;  //!< Allowed timing delay error tolerance
}
prach_conformance_params_t;

/**
 * PUCCH conformance test parameters
 */
typedef struct
{
  uint8_t ack_nack_pattern; //!< ACK/NACK pattern for validation
}
pucch_conformance_params_t;

/**
 * PUSCH conformance test parameters
 */
typedef struct
{
  // Reserved for future parameters
}
pusch_conformance_params_t;

/**
 * Overall conformance test configuration
 * 
 * Controls conformance testing behavior and parameters
 */
typedef struct
{
   bool conformance_test_enable;            //!< Enable conformance testing
   int  conformance_test_slots;             //!< Number of slots to run conformance test
   uint32_t conformance_test_start_time;    //!< Start time for conformance test
   prach_conformance_params_t  prach_params; //!< PRACH-specific parameters
   pucch_conformance_params_t  pucch_params; //!< PUCCH-specific parameters
   pusch_conformance_params_t  pusch_params; //!< PUSCH-specific parameters
   std::string conformance_test_stats_file;  //!< Output file for conformance test statistics
}conformance_test_params_t;

/**
 * Cell network parameters for OAM reconfiguration
 */
typedef struct
{
    int cell_id;        //!< Cell ID to reconfigure
    int vlan;           //!< VLAN ID for network configuration
    int pcp;            //!< Priority Code Point for QoS
    std::string mac;    //!< MAC address string
} cell_param_t;

/**
 * Cell update command for dynamic reconfiguration
 * 
 * Schedules network parameter updates at specific slot times
 */
typedef struct
{
    int slot_point;                      //!< Slot number when to apply update
    std::vector<cell_param_t> cell_params; //!< Cell parameters to update
} cell_update_cmd_t;

/**
 * Test MAC configuration class
 * 
 * Manages all configuration parameters for test MAC operation including:
 * - Thread configurations
 * - IPC and buffer settings
 * - Validation and testing options
 * - Cell restart and timing parameters
 * - cuMAC configurations
 */
class test_mac_configs {
public:
    /**
     * Construct test MAC configurations from YAML
     * 
     * @param[in] yaml_node Root YAML configuration node
     */
    test_mac_configs(yaml::node yaml_node);
    ~test_mac_configs();

#ifdef AERIAL_CUMAC_ENABLE
    test_cumac_configs* cumac_configs = nullptr;
#endif

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

    int get_tick_dynamic_sfn_slot_is_enabled()
    {
        return enableTickDynamicSfnSlot;
    }

    std::vector<cell_update_cmd_t>& get_cell_update_commands()
    {
        return cell_update_commands;
    }

    yaml::node& get_yaml_config()
    {
        return yaml_config;
    }
    conformance_test_params_t * get_conformance_test_params( )
    {
        return &conformance_test_params;
    }

    /**
     * Get FAPI sending deadline for a group (nanoseconds). Returns 0 if not configured.
     *
     * @param[in] group_id FAPI group ID
     * @return FAPI sending deadline in nanoseconds
     */
    int64_t get_fapi_tx_deadline_ns(size_t group_id) const
    {
        if (group_id < fapi_tx_deadline_ns.size())
            return fapi_tx_deadline_ns[group_id];
        return 0;
    }

    /**
     * Whether FAPI sending deadline (per-message send time) is enabled.
     *
     * @return True if FAPI sending deadline is enabled, false otherwise
     */
    int get_fapi_tx_deadline_enable() const { return fapi_tx_deadline_enable; }

    /**
     * Time cost per FAPI message (ns), for deadline send_time estimate. Default 1000.
     *
     * @return Time cost per FAPI message in nanoseconds
     */
    int get_fapi_tx_time_per_msg_ns() const { return fapi_tx_time_per_msg_ns; }

    int get_early_harq_deadline_ns()
    {
        return early_harq_deadline_ns;
    }

    int get_srs_early_deadline_ns()
    {
        return srs_early_deadline_ns;
    }

    int get_srs_late_deadline_ns()
    {
        return srs_late_deadline_ns;
    }

    int get_enable_srs_l1_limit_testing()
    {
        return enable_srs_l1_limit_testing;
    }

#ifdef SCF_FAPI_10_04
    uint8_t * get_indication_per_slot()
    {
        return indication_per_slot;
    }
#endif

    /**
     * Application mode
     * - 0: Normal static pattern mode
     * - 1: Dynamic pattern mode
     * - 2: UE dynamic mode
     */
    uint64_t app_mode;

    /**
     * PDSCH transport block alignment in bytes
     * For SCF 222 pdschMacPduBitsAlignment configuration
     * Valid values: 1, 2, 4, 8, 16, 32
     */
    uint32_t pdsch_align_bytes;

    int cell_config_timeout; //!< Cell CONFIG.req to CONFIG.resp timeout in milliseconds
    int cell_config_wait;    //!< Wait time between successful CONFIG.resp and next CONFIG.req in milliseconds
    int cell_config_retry;   //!< Number of times to retry cell configuration on failure

    int validate_enable;  //!< Enable FAPI message validation (0=disabled, 1=errors, 2=errors+warnings)
    int validate_log_opt; //!< Validation logging granularity (0=none, 1=per-msg, 2=per-PDU, 3=all)

    int tv_data_map_enable; //!< Enable FAPI test vector map to reuse duplicate PDU data

    int builder_thread_enable; //!< Enable FAPI builder thread for pre-building messages

    int estimate_send_time; //!< Estimated time cost for L2 to send one message (nanoseconds)

    std::vector<int32_t> schedule_total_time; //!< L2 scheduler total time budget per slot (nanoseconds)

    std::vector<int64_t> fapi_tx_deadline_ns; //!< FAPI sending deadline per group (nanoseconds). Empty or unset means 0 (send at slot start).
    int fapi_tx_deadline_enable = 0; //!< Enable deadline-based send time and nanosleep (0=off, 1=on)
    int fapi_tx_time_per_msg_ns = 1000; //!< Time per FAPI message (ns) for send_time estimate

    int oam_cell_ctrl_cmd;      //!< Enable OAM cell control commands
    std::string oam_server_addr; //!< OAM server address

    int fapi_delay_bit_mask; //!< Bit mask for which FAPI messages to delay

    int duplicate_fapi_bit_mask; //!< Bit mask for which FAPI messages to duplicate
    int duplicate_fapi_num_max;  //!< Maximum number of duplicate FAPI messages

    uint8_t num_mem_bank_cv_config_req_sent[MAX_CELLS_PER_SLOT]; //!< Number of memory banks sent in CV CONFIG.req per cell

    int early_harq_deadline_ns;  //!< Early HARQ deadline in nanoseconds
    int ul_ind_deadline_ns;      //!< UL indication deadline in nanoseconds
    int prach_ind_deadline_ns;   //!< PRACH indication deadline in nanoseconds
    int uci_ind_deadline_ns;     //!< UCI indication deadline in nanoseconds

    int srs_early_deadline_ns; //!< SRS early deadline in nanoseconds (for L1 limit testing)

    int srs_late_deadline_ns; //!< SRS late deadline in nanoseconds (for L1 limit testing)

    int srs_vald_sample_num; //!< Number of SRS validation samples

    std::vector<int> worker_cores; //!< List of CPU cores for worker threads

    bool get_dummy_tti_enabled()
    {
        return is_dummy_tti_enabled;
    }

private:

    // Parameters loaded from YAML configuration file
    int max_msg_size = 0;  //!< Maximum IPC message buffer size
    int max_data_size = 0; //!< Maximum IPC data buffer size
    int ipc_sync_mode = 0; //!< IPC synchronization mode (per-cell, per-TTI, per-msg)

    int restart_option = 0; //!< Cell restart behavior option

    int fapi_tb_loc = 0; //!< Transport block data location (0=msg_buf, 1=CPU_DATA, 2=CUDA_DATA, 3=GPU_DATA)

    // All cells restart test configuration
    int test_slots = 0;       //!< Total slots to run before restarting all cells
    int restart_interval = 0; //!< Interval in seconds before restarting after test completion

    // Single cell restart test configuration
    int cell_run_slots = 0;  //!< Number of slots to run a single cell before stopping
    int cell_stop_slots = 0; //!< Number of slots to keep cell stopped before restarting

    yaml::node yaml_config; //!< Full YAML configuration node

    struct nv::thread_config recv_thread_config = {};    //!< Receiver thread configuration
    struct nv::thread_config sched_thread_config = {};   //!< Scheduler thread configuration
    struct nv::thread_config builder_thread_config = {}; //!< Builder thread configuration

    conformance_test_params_t conformance_test_params = {}; //!< Conformance test parameters

#ifdef SCF_FAPI_10_04
    uint8_t indication_per_slot[6]{}; //!< SCF 10.04 indication flags per slot
#endif

    std::vector<cell_update_cmd_t> cell_update_commands; //!< Network reconfiguration test procedures

    bool is_dummy_tti_enabled = false;    //!< Enable dummy TTI for testing
    int enableTickDynamicSfnSlot = 0; //!< Enable dynamic SFN/slot in tick events
    int enable_srs_l1_limit_testing = 0; //!< Enable SRS L1 limit testing
};

#endif /* _TEST_MAC_CONFIGS_HPP */
