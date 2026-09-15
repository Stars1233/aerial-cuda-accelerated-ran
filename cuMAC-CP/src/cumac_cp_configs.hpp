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

#ifndef _CUMAC_CP_CONFIG_HPP_
#define _CUMAC_CP_CONFIG_HPP_

#include <vector>

#include "yaml.hpp"
#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"

/**
 * cuMAC Control Plane configuration manager
 *
 * Parses and manages configuration parameters from YAML file.
 * Provides accessors for IPC, threading, GPU, and performance tuning settings.
 */
class cumac_cp_configs {
public:
    /**
     * Construct configuration from YAML node
     *
     * @param[in] config_node Root YAML configuration node
     */
    cumac_cp_configs(yaml::node config_node);

    /**
     * Destructor cleans up resources
     */
    ~cumac_cp_configs();

    /**
     * Get receiver thread configuration
     *
     * @return Reference to thread configuration structure
     */
    struct nv::thread_config& get_recv_thread_config()
    {
        return recv_thread_config;
    }

    /**
     * Get IPC synchronization mode
     *
     * @return IPC sync mode value
     */
    int get_ipc_sync_mode()
    {
        return ipc_sync_mode;
    }

    /**
     * Get maximum IPC message size
     *
     * @return Maximum message size in bytes
     */
    int get_max_msg_size()
    {
        return max_msg_size;
    }

    /**
     * Get maximum IPC data size
     *
     * @return Maximum data size in bytes
     */
    int get_max_data_size()
    {
        return max_data_size;
    }

    /**
     * Get YAML configuration root node
     *
     * @return Reference to YAML root node
     */
    yaml::node& get_yaml_root()
    {
        return yaml_root;
    }

    std::vector<int> worker_cores{}; //!< CPU cores for worker threads
    int thread_num_per_core{}; //!< Number of worker threads per core

    uint32_t cell_num{}; //!< Number of cells to support
    uint32_t task_ring_len{}; //!< Length of task ring buffer
    uint32_t run_in_cpu{}; //!< Execution mode: 0=GPU, 1=CPU, 2=mixed

    int debug_option{}; //!< Debug option flags (see DBG_OPT_* constants)

    int gpu_id{}; //!< GPU device ID for CUDA operations

    std::string cumac_group_tv_file{}; //!< Path to cuMAC group test vector file

    bool enable_tv_test{}; //!< When true, parse group TV files for buffer validation / testMAC

    uint32_t cuda_block_num{}; //!< Number of CUDA thread blocks

    uint32_t group_buffer_enable{}; //!< Enable contiguous group buffer allocation
    uint32_t multi_stream_enable{}; //!< Enable multiple CUDA streams
    uint32_t slot_concurrent_enable{}; //!< Enable concurrent slot processing

    bool enable_gpu_share{}; //!< Enable GPU sharing from L1 for MU UE grouping
    bool enable_cubb{false}; //!< When true, open 3 secondary lock_free_mem_pool pools shared with cuphydriver; otherwise allocate local GPU buffers
    int srs_slot_lag{0}; //!< Slot offset applied when mapping TV files to slot indices (TV slot id + lag, wrapped mod schedule_slot_period)
    uint32_t task_bitmask{CUMAC_CP_TASK_MASK_DEFAULT}; //!< cuMAC task enable bitmask (bit 0~3: 4T4R tasks, bit 4: PFM SORT, bit 5: MU UE GRP)

    uint16_t num_blocks_per_row{8}; //!< MU UE pairing: CUDA blocks per row for channel orthogonality matrix (muUePairTask)

private:
    int max_msg_size{}; //!< Maximum IPC message size in bytes
    int max_data_size{}; //!< Maximum IPC data payload size in bytes
    int ipc_sync_mode{}; //!< IPC synchronization mode

    yaml::node yaml_root; //!< Root YAML configuration node

    struct nv::thread_config recv_thread_config{}; //!< Receiver thread configuration
};

#endif /* _CUMAC_CP_CONFIG_HPP_ */
