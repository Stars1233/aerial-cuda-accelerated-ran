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

#ifndef _CUMAC_CP_HANDLER_HPP_
#define _CUMAC_CP_HANDLER_HPP_

#include <string.h>
#include <sys/time.h>
#include <semaphore.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "cumac.h"
#include "api.h"

#include "nvlog.hpp"
#include "nv_utils.h"
#include "nv_ipc_utils.h"
#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"
#include "cumac_cp_configs.hpp"

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"
#include "nv_lockfree.hpp"
#include "cumac_task.hpp"
#include "cumac_cp_tv.hpp"
#include "muMimoUserPairing/muMimoUserPairing.cuh"

struct _CVSrsChestBuff;
typedef struct _CVSrsChestBuff CVSrsChestBuff;
struct _SrsInfoUpdate;
typedef struct _SrsInfoUpdate SrsInfoUpdate;

/**
 * Number of slot buffers for pipelined processing
 *
 * Multiple buffers allow simultaneous handling of different slots:
 * e.g., 1 buffer receiving, 1 buffer handling, 1 buffer in callback
 */
#define SCHED_SLOT_BUF_NUM 4

//! Maximum number of cuMAC messages per cell per slot
#define MAX_MSG_NUM_PER_CELL 4

//! Type alias for cuMAC cell configuration structure
typedef cumac_config_req_payload_t cumac_cell_configs_t;

/**
 * Throughput statistics tracker
 *
 * Maintains counters for processed slots and error indications
 */
class cumac_cp_thrput_t {
public:
    /**
     * Default constructor
     *
     * Initializes all counters to zero
     */
    cumac_cp_thrput_t() {
        reset();
    }

    /**
     * Copy constructor
     *
     * @param[in] obj Source object (resets counters rather than copying)
     */
    cumac_cp_thrput_t(const cumac_cp_thrput_t& obj) {
        reset();
    }

    /**
     * Reset all throughput counters to zero
     */
    void reset() {
        cumac_slots = 0;
        error = 0;
    }

    std::atomic<uint32_t> cumac_slots; //!< Number of slots processed through cuMAC

    std::atomic<uint32_t> error; //!< ERROR.indication counter

}; // cumac_cp_thrput_t;

/**
 * Cell control data structure
 *
 * Placeholder for per-cell control information
 */
typedef struct {
    int cell_tti_data; //!< Cell TTI data placeholder
} cell_ctrl_data_t;

/**
 * DU configuration data structure
 *
 * Placeholder for DU (Distributed Unit) configuration
 */
typedef struct {
    int cell_tti_data; //!< Cell TTI data placeholder
} du_config_data_t;

class cumac_cp_handler;

/**
 * Message cache for per-cell slot messages
 *
 * Buffers incoming messages for a single cell to handle message reordering.
 * Supports push/pull operations with automatic index management.
 */
class cell_msg_cache
{
public:
    /**
     * Constructor initializes empty cache
     */
    cell_msg_cache() {
        push_index = 0;
        pull_index = 0;
    }

    ~cell_msg_cache() {}

    /**
     * Reset cache if all messages have been pulled
     *
     * @return 0 on success, -1 if messages still pending
     */
    int reset_cell_data()
    {
        if (pull_index == push_index) {
            pull_index = 0;
            push_index = 0;
            return 0;
        } else {
            return -1;
        }
    }

    /**
     * Push message descriptor into cache
     *
     * @param[in] msg_desc Message descriptor to cache
     *
     * @return 0 on success, -1 if cache is full
     */
    int push_msg(nv::phy_mac_msg_desc& msg_desc)
    {
        if (push_index < MAX_MSG_NUM_PER_CELL) {
            cell_msgs[push_index++] = msg_desc;
            return 0;
        } else {
            return -1;
        }
    }

    /**
     * Pull next message descriptor from cache
     *
     * @return Pointer to message descriptor, or nullptr if cache is empty
     */
    nv::phy_mac_msg_desc* pull_msg()
    {
        if (pull_index < push_index) {
            return &cell_msgs[pull_index++];
        } else {
            return nullptr;
        }
    }

    nv::phy_mac_msg_desc cell_msgs[MAX_MSG_NUM_PER_CELL]; //!< Array of cached message descriptors

private:
    uint32_t push_index; //!< Index for next push operation
    uint32_t pull_index; //!< Index for next pull operation
};

/**
 * Scheduling slot data container
 *
 * Holds per-slot state including task pointer, message caches for all cells,
 * and current processing cell ID. Used for pipelined slot processing.
 */
class sched_slot_data
{
public:
    /**
     * Default constructor initializes invalid state
     */
    sched_slot_data() {
        cell_num = 0;
        curr_cell_id = 0;
        ss_sched.u32 = SFN_SLOT_INVALID;
        task = nullptr;
        handler = nullptr;
    }

    ~sched_slot_data() {}

    /**
     * Initialize slot data for processing
     *
     * @param[in] _handler Pointer to parent cuMAC CP handler
     * @param[in] _cell_num Number of cells to support
     */
    void init_slot_data(cumac_cp_handler* _handler, uint32_t _cell_num);

    /**
     * Reset slot data for new slot
     *
     * @param[in] ss System Frame Number and slot identifier
     */
    void reset_slot_data(sfn_slot_t ss);

    uint32_t curr_cell_id{}; //!< Current cell ID being processed
    uint32_t cell_num{}; //!< Total number of cells

    sfn_slot_t ss_sched{}; //!< SFN/slot identifier for this scheduling data
    cumac_task* task{}; //!< Associated cuMAC task for this slot
    std::vector<cell_msg_cache> slot_msgs{}; //!< Per-cell message caches for reordering

private:
    cumac_cp_handler* handler{}; //!< Pointer to parent handler
};

/**
 * cuMAC Control Plane handler
 *
 * Main handler class for cuMAC CP operations. Manages configuration,
 * task scheduling, message processing, and coordination with PHY layer
 * through IPC transport. Supports multi-cell operations with pipelined
 * slot processing and GPU/CPU execution modes.
 */
class cumac_cp_handler
{
public:
    /**
     * Construct cuMAC CP handler
     *
     * @param[in] _configs Configuration parameters for cuMAC CP
     * @param[in] wrapper PHY-MAC transport wrapper for IPC communication
     */
    cumac_cp_handler(cumac_cp_configs& _configs, nv::phy_mac_transport_wrapper& wrapper);

    /**
     * Destructor cleans up resources
     */
    ~cumac_cp_handler();

    /**
     * Set task ring pool and semaphore
     *
     * @param[in] ring Pointer to lock-free ring pool for cumac_task objects
     * @param[in] sem Pointer to semaphore for task synchronization
     */
    void set_task_ring(nv::lock_free_ring_pool<cumac_task>* ring, sem_t* sem);

    /**
     * Print throughput statistics
     *
     * @param[in] slot_counter Current slot counter for timestamp
     */
    void print_cumac_cp_thrput(uint64_t slot_counter);

    /**
     * Push new cuMAC task for slot processing
     *
     * @param[in] ss System Frame Number and slot identifier
     */
    void push_cumac_task(sfn_slot_t ss);

    /**
     * Handle CONFIG.request message from PHY
     *
     * @param[in] msg IPC message containing configuration request
     */
    void on_config_request(nv_ipc_msg_t& msg);

    /**
     * Handle PARAM.request message from PHY
     *
     * @param[in] msg IPC message containing parameter request
     */
    void on_param_request(nv_ipc_msg_t& msg);

    /**
     * Handle START.request message from PHY
     *
     * @param[in] msg IPC message containing start request
     */
    void on_start_request(nv_ipc_msg_t& msg);

    /**
     * Handle STOP.request message from PHY
     *
     * @param[in] msg IPC message containing stop request
     */
    void on_stop_request(nv_ipc_msg_t& msg);

    /**
     * Handle SCH_TTI.request message
     *
     * @param[in] msg IPC message containing TTI scheduling request
     * @param[in,out] task cuMAC task to process the request
     */
    void on_sch_tti_request(nv_ipc_msg_t& msg, cumac_task* task);

    /**
     * Copy cell data from message to task
     *
     * @param[in] msg IPC message containing cell data
     * @param[in,out] task cuMAC task to receive the data
     */
    void cell_copy_task(nv_ipc_msg_t &msg, cumac_task *task);

    /**
     * Callback function for completed cuMAC task
     *
     * @param[in,out] task Completed cuMAC task
     *
     * @return 0 on success, negative on error
     */
    int cumac_task_callback(cumac_task* task);

    /**
     * Send SCH_TTI.response message to PHY
     *
     * @param[in] task cuMAC task with results
     * @param[in] cell_id Cell identifier
     *
     * @return 0 on success, negative on error
     */
    int send_sch_tti_response(cumac_task* task, int cell_id);

    /**
     * Handle incoming slot message
     *
     * @param[in,out] msg_desc Message descriptor
     * @param[in] ss_msg SFN/slot from message
     */
    void handle_slot_msg(nv::phy_mac_msg_desc& msg_desc, sfn_slot_t ss_msg);

    /**
     * Handle incoming slot message with reordering support
     *
     * @param[in,out] msg_desc Message descriptor
     * @param[in] ss_msg SFN/slot from message
     */
    void handle_slot_msg_reorder(nv::phy_mac_msg_desc& msg_desc, sfn_slot_t ss_msg);

    /**
     * Get cell control data reference
     *
     * @return Reference to cell control data
     */
    cell_ctrl_data_t& get_cell_ctrl_data()
    {
        return cell_ctrl_data;
    }

    /**
     * Get scheduling slot data for specific slot
     *
     * @param[in] ss System Frame Number and slot identifier
     *
     * @return Reference to scheduling slot data
     */
    sched_slot_data& get_sched_slot_data(sfn_slot_t ss);

    /**
     * Get cuMAC task for specific slot
     *
     * @param[in] ss System Frame Number and slot identifier
     *
     * @return Pointer to cuMAC task, or nullptr if not found
     */
    cumac_task* get_cumac_task(sfn_slot_t ss) {
        return get_sched_slot_data(ss).task;
    }

    /**
     * Get PHY-MAC transport for specific cell
     *
     * @param[in] cell_id Cell identifier
     *
     * @return Reference to transport instance
     */
    nv::phy_mac_transport& transport(int cell_id) { return trans_wrapper.get_transport(cell_id); }

    /**
     * Get PHY-MAC transport wrapper
     *
     * @return Reference to transport wrapper instance
     */
    nv::phy_mac_transport_wrapper& transport_wrapper() { return trans_wrapper; }

    nv::lock_free_ring_pool<cumac_task>* task_ring; //!< Lock-free ring pool for cuMAC tasks

    cumac_cp_configs& configs; //!< Reference to configuration parameters

    /**
     * Get cuMAC buffer element counts
     *
     * @return Reference to buffer number structure
     */
    cumac_buf_num_t& get_buf_num()
    {
        return buf_num;
    }

    uint8_t* get_ue_pair_task_out() { return ue_pair_task_out_; }

    /**
     * Load MU UE pair TV static buffers
     *
     * @param ss SFN/slot identifier
     * @param strm CUDA stream
     * @return 0 on success or no-op, negative on error
     */
    int load_ue_pair_static_buffers(sfn_slot_t ss, cudaStream_t strm);

    int load_ue_pair_shared_memory(sfn_slot_t ss, std::vector<struct nv::phy_mac_msg_desc>& tti_reqs);

    float* chan_orth_mat_buf() { return ue_pair_chan_orth_; }
    float* get_srs_snr_buf() { return ue_pair_srs_snr_; }
    uint8_t* get_srs_chan_est_buf() { return ue_pair_srs_chan_est_; }
    uint8_t* get_task_out_buf() { return ue_pair_task_out_; }
    __half2* get_cubb_srs_buf() { return ue_pair_cubb_gpu_; }

    /** Pointer to loaded group test vector, or nullptr if not parsed */
    cumac_cp_tv_t* get_group_tv_ptr() { return group_tv.parsed ? &group_tv : nullptr; }

    void init_ue_pair_gpu_resources();

    /**
     * Snapshot the three MU UE-pair GPU buffers (`ue_pair_srs_chan_est_`,
     * `ue_pair_srs_snr_`, `ue_pair_chan_orth_`) and asynchronously write a
     * single `cumac_ue_pair_buffers_<i>_SFN_<sfn>.<slot>.h5` file.
     *
     * Mirrors the cuphydriver `SrsIpcManager::dump_h5()` two-phase design:
     * phase 1 is a synchronous device-to-host snapshot into pre-pinned
     * staging buffers (runs in the caller's worker thread); phase 2 is a
     * detached `SCHED_OTHER` worker that writes the HDF5 file so disk I/O
     * cannot preempt the RT data path.
     *
     * Gated by `DBG_OPT_DUMP_UE_PAIR_H5` in `cumac_cp.yaml debug_option`
     * (caller checks the bit). The per-process cap equals the number of
     * loaded MU UE-pair TV slots; each unique `tv_slot_idx` is dumped at
     * most once. No-op if the GPU buffers are not yet allocated, no TV
     * is loaded for this slot, or this `tv_slot_idx` has already been
     * captured.
     *
     * @param sfn        System Frame Number (filename + log).
     * @param slot       Slot index            (filename + log).
     * @param outputDir  Output directory; defaults to "/tmp".
     * @return 0 on success / no-op / cap-hit; -1 on snapshot failure.
     */
    int dump_ue_pair_h5(uint16_t sfn, uint16_t slot, const char* outputDir = "/tmp");

private:
    /**
     * Initialize cuMAC task memory and resources
     *
     * @param[in,out] task Task to initialize
     *
     * @return 0 on success, negative on error
     */
    int initiate_cumac_task(cumac_task* task);

    /**
     * Verify configuration parameters and allocate task buffers
     *
     * @return 0 on success, negative on error
     */
    int check_config_params();

    /**
     * Check and validate task buffer sizes
     *
     * @param[in] task Task to validate
     *
     * @return 0 on success, negative on error
     */
    int check_task_buf_size(cumac_task* task);

    /**
     * Allocate buffer for cuMAC task
     *
     * @param[in,out] task Task owning the buffer
     * @param[out] ptr Pointer to allocated buffer
     * @param[out] num_save Optional: pointer to save element count
     * @param[in] num Number of elements to allocate
     * @param[in] force_host_mem 1 to force host memory, 0 for GPU memory
     *
     * @return 0 on success, negative on error
     */
    template <typename T>
    int malloc_cumac_buf(cumac_task *task, T **ptr, uint32_t* num_save, uint32_t num, uint32_t force_host_mem = 0);

    /**
     * Bytes required for one task's carve-out of group_buf (GPU path + group_buf_enabled),
     * matching initiate_cumac_task allocation order and alignment.
     */
    uint32_t compute_group_buf_need_bytes() const;

    uint64_t global_tick{}; //!< Global tick counter

    std::atomic<sfn_slot_t> ss_curr{}; //!< Current SFN/slot being processed

    nv::phy_mac_transport_wrapper& trans_wrapper; //!< PHY-MAC transport wrapper reference

    sem_t* task_sem{}; //!< Task semaphore pointer

    cell_ctrl_data_t cell_ctrl_data{}; //!< Per-cell control data

    sched_slot_data sched_slots[SCHED_SLOT_BUF_NUM]; //!< Per-slot scheduling data for pipelining

    struct cumac::cumacSchedulerParam group_params{}; //!< Cell group scheduler parameters

    cumac_buf_num_t buf_num{}; //!< cuMAC buffer element counts

    uint32_t group_buf_size{}; //!< Size of contiguous group buffer (bytes)
    uint32_t configured_cell_num{}; //!< Number of configured cells
    std::vector<cumac_cell_configs_t> cell_configs{}; //!< Per-cell configurations
    float* blerTargetActUe{nullptr}; //!< BLER target for active UEs (size: nActiveUe)

    cumac_cp_tv_t group_tv{}; //!< Test vector data for group validation

    std::vector<cumac_cp_thrput_t> thrputs{}; //!< Per-cell throughput statistics

    void free_mu_ue_pair_gpu();

    cumac::muMimoUserPairing* mu_ue_pair_gpu_{nullptr};

    // Static GPU buffers for muMimoUserPairing: shared across all slots/tasks
    uint8_t* ue_pair_srs_chan_est_{nullptr};
    float* ue_pair_srs_snr_{nullptr};
    float* ue_pair_chan_orth_{nullptr};
    __half2* ue_pair_cubb_gpu_{nullptr};
    uint8_t* ue_pair_task_out_{nullptr};

    // Shared GPU/CPU memory pools for muMimoUserPairing when enable_cubb=1
    std::unique_ptr<nv::lock_free_mem_pool<uint8_t>>        cubb_gpu_pool_{};
    std::unique_ptr<nv::lock_free_mem_pool<CVSrsChestBuff>> chest_buf_pool_{};
    std::unique_ptr<nv::lock_free_mem_pool<SrsInfoUpdate>>  srs_info_pool_{};

    // Buffer sizes and geometry for per-task TV allocation
    uint32_t ue_pair_num_prg_samp_per_subband_{0}; //!< Set by init_ue_pair_gpu_resources
    uint32_t ue_pair_num_subband_{0};              //!< Set by init_ue_pair_gpu_resources
    uint32_t ue_pair_num_ue_ant_port_{0};          //!< Set by init_ue_pair_gpu_resources
    uint32_t ue_pair_num_bs_ant_{0};               //!< Set by init_ue_pair_gpu_resources
    size_t ue_pair_chan_est_bytes_{0};
    size_t ue_pair_snr_bytes_{0};
    size_t ue_pair_orth_bytes_{0};
    size_t ue_pair_cubb_bytes_{0};
    size_t ue_pair_task_in_bytes_{0};
    size_t ue_pair_out_bytes_{0};

    uint16_t num_srs_info_{0}; //!< Number of SRS UEs per slot per cell
    int first_ue_pair_tv_id{-1}; //!< First slot index in UE pair TV ID. Set -1 to auto-detect the first UE pair slot scheduled from L2.

    //! Enqueue-FIFO GPU serialization across worker threads (see order_sem).
    order_sem task_order_sem_;

    uint32_t moduleBitMask{0}; //!< Indicate which cuMAC modules should be enabled. Each bit represent 1 type defined in cumac_module_type_t

    uint32_t last_thrput_print_slot{0}; //!< Last slot index in a frame for throughput print

    //========================================================================
    // MU UE-pair H5 dump state (DBG_OPT_DUMP_UE_PAIR_H5)
    //
    // Per-process cap = number of loaded MU UE-pair TV slots
    // (`mu_ue_pair_tv_loaded == true` in `group_tv.ue_pair`). Each unique
    // tv_slot_idx is dumped at most once so the file set captures one
    // snapshot per scheduled MU UE-pair slot in the TV pattern.
    //========================================================================
    //! Pinned-host staging triple shadowing the three GPU buffers.
    //! Borrowed (not owned) by detached H5 writer threads.
    struct UePairH5Staging
    {
        uint8_t* chan_est_host{nullptr};
        uint8_t* snr_host{nullptr};
        uint8_t* chan_orth_host{nullptr};
    };

    std::vector<UePairH5Staging> ue_pair_h5_staging_{}; //!< Sized to ue_pair_h5_max_slots_ on init
    std::vector<uint8_t>         ue_pair_h5_dumped_tv_slot_{}; //!< 1 == tv_slot_idx already dumped; size == group_tv.ue_pair.size()
    uint32_t                     ue_pair_h5_max_slots_{0}; //!< Count of loaded MU UE-pair TV slots; 0 == dumping disabled
    std::atomic<uint32_t>        ue_pair_h5_dump_count_{0}; //!< 0..ue_pair_h5_max_slots_
    std::atomic<bool>            ue_pair_h5_staging_ready_{false}; //!< Set once after first successful allocate
    std::mutex                   ue_pair_h5_init_mtx_{}; //!< Guards lazy staging allocation
    std::mutex                   ue_pair_h5_claim_mtx_{}; //!< Guards tv_slot dedup, staging-slot assignment, and thread tracking
    std::vector<std::thread>     ue_pair_h5_threads_{};   //!< Writer threads joined in free_ue_pair_h5_staging()

    //! Allocate one staging triple per loaded MU UE-pair TV slot, sized to
    //! the current ue_pair buffer geometry. Idempotent; called under
    //! `ue_pair_h5_init_mtx_`. Returns false if no MU UE-pair TVs are
    //! loaded (dumping stays disabled) or any allocation fails.
    bool allocate_ue_pair_h5_staging();

    //! Free every staging triple in `ue_pair_h5_staging_` and reset the
    //! dedup bookkeeping. Idempotent. Called from `free_mu_ue_pair_gpu()`
    //! so staging lifetime matches the GPU buffers it shadows.
    void free_ue_pair_h5_staging();
};

#endif /* _CUMAC_CP_HANDLER_HPP_ */