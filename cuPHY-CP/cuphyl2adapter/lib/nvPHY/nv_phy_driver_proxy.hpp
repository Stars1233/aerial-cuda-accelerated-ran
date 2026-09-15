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

#if !defined(NV_PHY_DRIVER_PROXY_HPP_INCLUDED_)
#define NV_PHY_DRIVER_PROXY_HPP_INCLUDED_

#include <memory>
#include <mutex>
#include <span>

#include "slot_command/slot_command.hpp"
#include "nv_phy_utils.hpp"
#include "nvlog.hpp"

#include "cuphydriver_api.hpp"
using mplane_config = std::vector<::cell_mplane_info>;

using namespace slot_command_api;

namespace nv
{
/**
 * @brief Singleton proxy class for PHY driver interface
 *
 * This class provides a thread-safe singleton interface to the PHY driver,
 * abstracting cell management, configuration updates, and work queue operations.
 */
class PHYDriverProxy {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the PHYDriverProxy singleton
     */
    static PHYDriverProxy& getInstance();

    /**
     * @brief Get pointer to the singleton instance (safe, no UB)
     * @return Pointer to the PHYDriverProxy singleton, or nullptr if not initialized
     * @note Use this when you need to check if the proxy exists before using it
     */
    static PHYDriverProxy* getInstancePtr();

    /**
     * @brief Create and initialize the singleton with a PHY driver handle
     * @param phy_driver Handle to the PHY driver instance
     * @param conf M-plane configuration for cells
     */
    static void make(phydriver_handle phy_driver, mplane_config& conf);
    
    /**
     * @brief Create and initialize the singleton for standalone testing
     * @param cfg Thread configuration (optional)
     * @param max_cell_num Maximum number of cells to support
     */
    static void make(thread_config* cfg = nullptr, int max_cell_num = MAX_CELLS_PER_SLOT);

    /**
     * @brief Create a PHY cell with full configuration
     * @param cell_info Cell PHY information structure
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_create(cell_phy_info& cell_info);
    
    /**
     * @brief Create a PHY cell with minimal configuration
     * @param cell_id Cell identifier
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_create(uint16_t cell_id);
    
    /**
     * @brief Destroy a PHY cell
     * @param cell_id Cell identifier
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_destroy(uint16_t cell_id);
    
    /**
     * @brief Start a PHY cell
     * @param cell_id Cell identifier
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_start(uint16_t cell_id);
    
    /**
     * @brief Stop a PHY cell
     * @param cell_id Cell identifier
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_stop(uint16_t cell_id);
    
    /**
     * @brief Enqueue PHY work command for processing.
     *
     * Thin proxy over ::l1_enqueue_phy_work(). The @p skip_mask parameter controls
     * which task categories cuphydriver excludes from the pipeline (see EnqueueSkipMask).
     * Defaults to EnqueueSkipMask::SKIP_NONE — full pipeline, identical to original behaviour.
     * Pass EnqueueSkipMask::SKIP_DL_CPLANE when C-plane work has already been performed
     * by the store-and-replay path (task_dlc before EOM).
     *
     * @param command   Slot command containing PHY work to process.
     * @param skip_mask Categories to exclude; defaults to SKIP_NONE.
     * @return 0 on success, non-zero on error.
     */
    [[nodiscard]] int l1_enqueue_phy_work(slot_command& command,
                                          EnqueueSkipMask skip_mask = EnqueueSkipMask::SKIP_NONE);
#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * Enqueue PHY work using caller-bound early slot maps (direct C-plane path).
     *
     * Store-replay overload of l1_enqueue_phy_work(): forwards to ::l1_enqueue_phy_work() with
     * use_bound_early_maps set, so the driver uses the DL/UL early slot maps supplied here
     * (already set up for this slot by the direct C-plane path) instead of the driver-context
     * early maps. When the driver is null it falls back to the simulator, which ignores
     * @p skip_mask and both slot-map pointers.
     *
     * @param[in] command                 Slot command containing PHY work to process.
     * @param[in] skip_mask               Categories to exclude; see EnqueueSkipMask.
     * @param[in] bound_early_slot_map_dl Bound early DL slot map; null if not provided.
     *                                    Ownership is not transferred (non-owning pointer).
     * @param[in] bound_early_slot_map_ul Bound early UL slot map; null if not provided.
     *                                    Ownership is not transferred (non-owning pointer).
     * @return 0 on success, non-zero on error.
     */
    [[nodiscard]] int l1_enqueue_phy_work(slot_command& command,
                                          EnqueueSkipMask skip_mask,
                                          void* bound_early_slot_map_dl,
                                          void* bound_early_slot_map_ul);
#endif

    /**
     * @brief Allocate a Task from the driver pool, initialise, and push onto the DL queue.
     *
     * Wrapper over cuphydriver l1_push_new_dl_task(). The caller passes a work-function
     * pointer with task_work_function-compatible signature without including task.hpp.
     *
     * @param ts_exec_ns  Execution timestamp in nanoseconds (from std::chrono::system_clock).
     * @param name        Human-readable task name for diagnostics.
     * @param fn          Work function — signature must match task_work_function.
     * @param arg         Opaque arg passed through to the work function.
     * @param first_cell  First-cell parameter forwarded to Task::init().
     * @param num_cells   Num-cells parameter forwarded to Task::init().
     * @param num_tasks   Num-tasks parameter forwarded to Task::init().
     * @param desired_wid Worker affinity; use INVALID_WORKER_ID for any free DL worker.
     * @return true on success; false if task allocation failed.
     */
    [[nodiscard]] bool l1_push_new_dl_task(uint64_t ts_exec_ns, const char* name,
                                           l1_task_work_fn_t fn, void* arg,
                                           int first_cell, int num_cells, int num_tasks,
                                           worker_id desired_wid);

    /**
     * @brief Allocate a Task from the driver pool, initialise, and push onto the UL queue.
     *
     * Wrapper over cuphydriver l1_push_new_ul_task(). UL tasks are dispatched to UL
     * workers, keeping them independent from the DL worker pool.
     *
     * @param ts_exec_ns  Execution timestamp in nanoseconds (from std::chrono::system_clock).
     * @param name        Human-readable task name for diagnostics.
     * @param fn          Work function — signature must match task_work_function.
     * @param arg         Opaque arg passed through to the work function.
     * @param first_cell  First-cell parameter forwarded to Task::init().
     * @param num_cells   Num-cells parameter forwarded to Task::init().
     * @param num_tasks   Num-tasks parameter forwarded to Task::init().
     * @param desired_wid Worker affinity; use INVALID_WORKER_ID for any free UL worker.
     * @return true on success; false if task allocation failed.
     */
    [[nodiscard]] bool l1_push_new_ul_task(uint64_t ts_exec_ns, const char* name,
                                           l1_task_work_fn_t fn, void* arg,
                                           int first_cell, int num_cells, int num_tasks,
                                           worker_id desired_wid);

    /**
     * @brief Allocate, initialise, and push a batch of DL tasks under a single lock.
     *
     * Wrapper over cuphydriver l1_push_new_dl_tasks_bulk(). Captures a single
     * timestamp before the call; all tasks in the batch share @p ts_exec_ns.
     *
     * @param ts_exec_ns  Shared execution timestamp in nanoseconds.
     * @param specs       Non-owning view of TaskSpec descriptors; specs.size()
     *                    must be <= TaskSpec::BULK_MAX.
     * @return Number of tasks successfully pushed.
     */
    [[nodiscard]] int l1_push_new_dl_tasks_bulk(uint64_t ts_exec_ns,
                                                std::span<const TaskSpec> specs);

    /**
     * @brief Mirrors l1_push_new_dl_tasks_bulk() for the UL worker pool.
     *
     * @param ts_exec_ns  Shared execution timestamp in nanoseconds.
     * @param specs       Non-owning view of TaskSpec descriptors; specs.size()
     *                    must be <= TaskSpec::BULK_MAX.
     * @return Number of tasks successfully pushed.
     */
    [[nodiscard]] int l1_push_new_ul_tasks_bulk(uint64_t ts_exec_ns,
                                                std::span<const TaskSpec> specs);

    /**
     * @brief Set output callbacks for PHY processing
     * @param cb Callbacks structure with function pointers
     * @return Status code (0 for success, negative for error)
     */
    int l1_set_output_callback(callbacks& cb);

    /**
     * @brief Update cell configuration for grid size
     * @param mplane_id M-plane identifier
     * @param grid_sz Grid size in resource blocks
     * @param dl true for downlink, false for uplink
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_update_cell_config(uint16_t mplane_id, uint16_t grid_sz, bool dl);
    
    /**
     * @brief Update cell configuration for MAC address and VLAN
     * @param mplane_id M-plane identifier
     * @param dst_mac Destination MAC address
     * @param vlan_tci VLAN Tag Control Information
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_update_cell_config(uint16_t mplane_id, std::string dst_mac, uint16_t vlan_tci);
    
    /**
     * @brief Update cell configuration for eAxC IDs
     * @param mplane_id M-plane identifier
     * @param eaxcids_ch_map Map of channel to eAxC ID vectors
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_update_cell_config(uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map);
    
    /**
     * @brief Update cell configuration with multiple attributes
     * @param mplane_id M-plane identifier
     * @param attrs Map of attribute names to values
     * @param res Result map for status of each attribute update
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_update_cell_config(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res);
    
    /**
     * @brief Update cell attenuation value
     * @param mplane_id M-plane identifier
     * @param attenuation_dB Attenuation value in dB
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_update_attenuation(uint16_t mplane_id, float attenuation_dB);

    /**
     * @brief Attach this thread to the PHY driver's CUDA primary context (no-op if no driver)
     */
    void l1_bind_thread_to_phy_cuda_context();
    
    /**
     * @brief Update GPS timing synchronization parameters
     * @param alpha GPS alpha value
     * @param beta GPS beta value
     * @return Status code (0 for success, negative for error)
     */
    int l1_update_gps_alpha_beta(uint64_t alpha,int64_t beta);
    
    /**
     * @brief Update full cell PHY configuration
     * @param cell_pinfo Cell PHY information structure
     * @param fn Callback function for cell update completion
     * @return Status code (0 for success, negative for error)
     */
    int l1_cell_update_cell_config(struct cell_phy_info& cell_pinfo, ::CellUpdateCallBackFn& fn);
    /**
     * @brief Update full cell PHY configuration (single caller-runs entry point).
     *
     * Routes internally: active cells -> offload PRACH handover (ENABLE_FAPI_STORE_REPLAY);
     * otherwise the synchronous in-place reconfig.
     * @param cell_pinfo Cell PHY information structure.
     * @return Status code (0 for success, negative for error).
     */
    [[nodiscard]] int l1_cell_update_cell_config_caller_runs(cell_phy_info& cell_pinfo);

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Slot-boundary drain for the offload PRACH reconfig handover.
     * Call once per SLOT.INDICATION; lock-free no-op unless a handover is armed.
     */
    void l1_try_commit_prach_offload_handover();
#endif // ENABLE_FAPI_STORE_REPLAY

    /**
     * @brief Dry-run: check if a new phyCellId can be updated successfully
     * @param cell_id cell_id
     * @param new_phy_cell_id New phyCellId
     * @return true if the phyCellId can't be updated successfully
     */
    bool l1_phy_cell_id_mismatch(int cell_id, uint16_t new_phy_cell_id);

    /**
     * @brief Lock the cell configuration update mutex
     * @return Status code (0 for success, negative for error)
     */
    int     l1_lock_update_cell_config_mutex();
    
    /**
     * @brief Unlock the cell configuration update mutex
     * @return Status code (0 for success, negative for error)
     */
    int     l1_unlock_update_cell_config_mutex();
    
    /**
     * @brief Get PRACH start RO (Resource Occasion) index
     * @param phy_cell_id PHY cell identifier
     * @return PRACH start RO index
     */
    uint8_t l1_get_prach_start_ro_index(uint16_t phy_cell_id);
    
    /**
     * @brief Allocate SRS channel estimation buffer pool
     * @param requestedBy Identifier of requester
     * @param phy_cell_id PHY cell identifier
     * @param poolSize Size of buffer pool to allocate
     * @return true if allocation successful, false otherwise
     */
    bool allocSrsChesBuffPool(uint32_t requestedBy, uint16_t phy_cell_id, uint32_t poolSize);
    
    /**
     * @brief Deallocate SRS channel estimation buffer pool
     * @param phy_cell_id PHY cell identifier
     * @return true if deallocation successful, false otherwise
     */
    bool deAllocSrsChesBuffPool(uint16_t phy_cell_id);
    
    /**
     * @brief Copy transport block to GPU buffer
     * @param cell_id Cell identifier
     * @param tb_buff Transport block buffer (host memory)
     * @param gpu_buff_ref Reference to GPU buffer pointer
     * @param tb_len Transport block length in bytes
     * @param slot_index Slot index for buffer management
     */
    void  l1_copy_TB_to_gpu_buf(uint16_t cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index, uint16_t sfn = 0);

    /**
     * @brief Phase 1 TB H2D staging — see @c l1_stage_tb_h2d in cuphydriver.
     *
     * Stages one cell's TB data for a subsequent batched H2D copy.  Must be
     * called once per cell that has TX_DATA before @c l1_launch_tb_h2d is
     * called for the same ring slot.
     *
     * @pre  Called on the **message thread** only; not thread-safe.
     * @pre  The ring slot identified by @p slot_index must not have a Phase 2
     *       launch in progress.
     *
     * @return CUPHY_STATUS_* encoded as @c int.
     */
    [[nodiscard]] int l1_stage_tb_h2d(uint16_t phy_cell_id,
                                     const uint8_t* shm_src,
                                     uint32_t       len,
                                     uint8_t        slot_index,
                                     uint8_t**      gpu_buf_out);

    /**
     * @brief Phase 2 batched TB H2D launch — see @c l1_launch_tb_h2d in cuphydriver.
     *
     * Kicks off the batched async memcpy for all cells staged via
     * @c l1_stage_tb_h2d in the current ring slot.  Must be called exactly
     * once per slot, after all Phase 1 calls for that slot have completed.
     *
     * @pre  Called on the **message thread** only; not thread-safe.
     * @pre  All Phase 1 (@c l1_stage_tb_h2d) calls for this slot must have
     *       returned successfully before this call.
     *
     * @return CUPHY_STATUS_* encoded as @c int.
     */
    [[nodiscard]] int l1_launch_tb_h2d(uint16_t slot_in_frame);

    /**
     * @brief True if the dedicated H2D copy thread is enabled (YAML @c enable_h2d_copy_thread).
     */
    [[nodiscard]] bool l1_get_h2d_copy_thread_enable() const noexcept;
    int l1_cv_mem_bank_update(uint32_t cell_id,uint16_t rnti, uint16_t buffer_idx, uint16_t reportType, uint16_t startPrbGrp,uint32_t srsPrbGrpSize,uint16_t numPrgs,
        uint8_t nGnbAnt,uint8_t nUeAnt,uint32_t offset, uint8_t* srsChEsts, uint16_t startValidPrg, uint16_t nValidPrg);
    int l1_cv_mem_bank_retrieve_buffer(uint32_t cell_id,uint16_t rnti, uint16_t buffer_idx, uint16_t reportType, uint8_t *pSrsPrgSize, uint16_t* pSrsStartPrg, uint16_t* pSrsStartValidPrg, uint16_t* pSrsNValidPrg, cuphyTensorDescriptor_t* descr, uint8_t** ptr);
    int l1_cv_mem_bank_update_buffer_state(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state);
    int l1_cv_mem_bank_get_buffer_state(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState *srs_chest_buff_state);
    int l1_bfw_coeff_retrieve_buffer(uint32_t cell_id, bfw_buffer_info* ptr);
    int l1_mMIMO_enable_info(uint8_t *pMuMIMO_enable);
    int l1_enable_srs_info(uint8_t *pEnable_srs);
    uint32_t l1_get_cell_group_num();
    int l1_get_ch_segment_proc_enable_info(uint8_t* ch_seg_proc_enable);
    bool l1_incr_recovery_slots();
    bool l1_incr_all_obj_free_slots();
    void l1_reset_all_obj_free_slots();
    void l1_reset_recovery_slots();
    bool l1_get_aggr_obj_free_status();
    int l1_storeDBTPdu(uint16_t cell_id, void*   data_buf);
    [[nodiscard]] int l1_resetDBTStorage(uint16_t cell_id);
    int l1_getBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx);
    int l1_setBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx);
    int l1_staticBFWConfigured(uint16_t cell_id);
    int16_t l1_getDynamicBeamIdOffset(uint16_t cell_id);
    /**
     * Check whether the driver is using the direct FAPI-to-C-plane path.
     *
     * @return true when direct FAPI-to-C-plane is enabled, false otherwise.
     */
    [[nodiscard]] bool l1_isFapiToCplaneDirect();
    /**
     * Record one prior-slot direct BFW CVI completion for C-plane consumption.
     *
     * @param[in] cell_id SDK cell index the record belongs to.
     * @param[in] record Producer-built BFW completion record.
     * @return 0 on success, non-zero on error.
     */
    [[nodiscard]] int l1_recordDirectBfwCviRecord(uint16_t cell_id, const direct_bfw_cvi_record& record);
    int l1_get_send_static_bfw_wt_all_cplane();
    void l1_resetBatchedMemcpyBatches();
    uint8_t l1_get_enable_weighted_average_cfo();
    [[nodiscard]] uint8_t l1_get_cplane_processing_dl_batch_size();
    [[nodiscard]] uint8_t l1_get_cplane_processing_ul_batch_size();

    [[nodiscard]] bool l1_get_split_ul_cuda_streams();

    [[nodiscard]] worker_id l1_get_dl_worker_id(int worker_index);
    [[nodiscard]] worker_id l1_get_ul_worker_id(int worker_index);

    /**
     * @brief Get the current state of DL TX notification
     * @return Current state of DL TX notification (true if enabled, false if disabled)
     */
    [[nodiscard]] bool l1_get_dl_tx_notification() const noexcept;

    [[nodiscard]] uint8_t l1_get_cpu_task_tracing_mode() const noexcept;

private:
    PHYDriverProxy(phydriver_handle driver, mplane_config const& config) :
        driver_(driver),
        config_(config) {}

    // Constructor for L2adapter_standalone test
    PHYDriverProxy(int max_cell_num)
    {
        driver_ = nullptr;
        config_.resize(max_cell_num);
        for (int i = 0; i < max_cell_num; i++)
        {
            // Set all default values to 0
            memset(&config_[i], 0, sizeof(cell_mplane_info));

            // Add L2SA test required configurations here
            config_[i].mplane_id = i + 1;
        }
    }

public:
    PHYDriverProxy(const PHYDriverProxy&) = delete;
    PHYDriverProxy&   operator=(const PHYDriverProxy&) = delete;

    // Get cell_mplane_info by cell_id
    cell_mplane_info& getMPlaneConfig(int32_t cell_id);

    // Get cell_mplane_info by mplane_id
    cell_mplane_info& getMPlaneConfigByMplaneId(uint32_t mplane_id);

    // Get cell_mplane_info list
    std::vector<cell_mplane_info>& getMPlaneConfigList();

    bool driver_exist()
    {
        return driver_ != nullptr;
    }

    phydriver_handle get_driver()
    {
        return driver_;
    }

private:
    phydriver_handle driver_;
    mplane_config    config_;
};
} // namespace nv
#endif // NV_PHY_DRIVER_PROXY_HPP_INCLUDED_
