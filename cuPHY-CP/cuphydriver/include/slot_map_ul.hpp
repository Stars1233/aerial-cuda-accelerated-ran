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

#ifndef SLOT_MAP_UL_H
#define SLOT_MAP_UL_H

#include <iostream>
#include <atomic>
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "time.hpp"
#include "phychannel.hpp"
#include "nvlog.hpp"
#include "order_entity.hpp"
#include "phypusch_aggr.hpp"
#include "phypucch_aggr.hpp"
#include "phyprach_aggr.hpp"
#include "physrs_aggr.hpp"
#include "phyulbfw_aggr.hpp"


/**
 * @brief Timing measurements for uplink slot processing stages.
 *
 * Captures start/end timestamps for each UL channel processing phase
 * (CUDA execution, completion, callbacks) and packet reception stages.
 */
struct ul_slot_timings {
    t_ns start_t_ul_cplane[UL_MAX_CELLS_PER_SLOT];            ///< Start of C-plane processing per cell
    t_ns end_t_ul_cplane[UL_MAX_CELLS_PER_SLOT];              ///< End of C-plane processing per cell
    t_ns start_t_ul_pusch_cuda[DL_MAX_CELLS_PER_SLOT];        ///< Start of PUSCH CUDA execution per cell
    t_ns end_t_ul_pusch_cuda[DL_MAX_CELLS_PER_SLOT];          ///< End of PUSCH CUDA execution per cell
    t_ns start_t_ul_bfw_cuda[DL_MAX_CELLS_PER_SLOT];          ///< Start of UL BFW CUDA execution per cell
    t_ns end_t_ul_bfw_cuda[DL_MAX_CELLS_PER_SLOT];            ///< End of UL BFW CUDA execution per cell
    t_ns start_t_ul_pucch_cuda[DL_MAX_CELLS_PER_SLOT];        ///< Start of PUCCH CUDA execution per cell
    t_ns end_t_ul_pucch_cuda[DL_MAX_CELLS_PER_SLOT];          ///< End of PUCCH CUDA execution per cell
    t_ns start_t_ul_prach_cuda[DL_MAX_CELLS_PER_SLOT];        ///< Start of PRACH CUDA execution per cell
    t_ns end_t_ul_prach_cuda[DL_MAX_CELLS_PER_SLOT];          ///< End of PRACH CUDA execution per cell
    t_ns start_t_ul_srs_cuda[DL_MAX_CELLS_PER_SLOT];          ///< Start of SRS CUDA execution per cell
    t_ns end_t_ul_srs_cuda[DL_MAX_CELLS_PER_SLOT];            ///< End of SRS CUDA execution per cell
    t_ns start_t_ul_order;                                     ///< Start of symbol ordering
    t_ns end_t_ul_order;                                       ///< End of symbol ordering
    t_ns start_t_ul_order_cuda;                                ///< Start of symbol ordering CUDA kernel
    t_ns end_t_ul_order_cuda;                                  ///< End of symbol ordering CUDA kernel
    t_ns start_t_ul_rx_pkts[UL_MAX_CELLS_PER_SLOT];           ///< Start of packet reception per cell
    t_ns end_t_ul_rx_pkts[UL_MAX_CELLS_PER_SLOT];             ///< End of packet reception per cell
    t_ns start_t_ul_rx[UL_MAX_CELLS_PER_SLOT];                ///< Start of UL data reception per cell
    t_ns end_t_ul_rx[UL_MAX_CELLS_PER_SLOT];                  ///< End of UL data reception per cell
    t_ns start_t_ul_freemsg[UL_MAX_CELLS_PER_SLOT];           ///< Start of message buffer release per cell
    t_ns end_t_ul_freemsg[UL_MAX_CELLS_PER_SLOT];             ///< End of message buffer release per cell
    t_ns start_t_ul_pusch_compl[UL_MAX_CELLS_PER_SLOT];       ///< Start of PUSCH completion processing per cell
    t_ns end_t_ul_pusch_compl[UL_MAX_CELLS_PER_SLOT];         ///< End of PUSCH completion processing per cell
    t_ns start_t_ul_pucch_compl[UL_MAX_CELLS_PER_SLOT];       ///< Start of PUCCH completion processing per cell
    t_ns end_t_ul_pucch_compl[UL_MAX_CELLS_PER_SLOT];         ///< End of PUCCH completion processing per cell
    t_ns start_t_ul_prach_compl[UL_MAX_CELLS_PER_SLOT];       ///< Start of PRACH completion processing per cell
    t_ns end_t_ul_prach_compl[UL_MAX_CELLS_PER_SLOT];         ///< End of PRACH completion processing per cell
    t_ns start_t_ul_srs_compl[UL_MAX_CELLS_PER_SLOT];         ///< Start of SRS completion processing per cell
    t_ns end_t_ul_srs_compl[UL_MAX_CELLS_PER_SLOT];           ///< End of SRS completion processing per cell
    t_ns start_t_ul_bfw_compl[UL_MAX_CELLS_PER_SLOT];         ///< Start of UL BFW completion processing per cell
    t_ns end_t_ul_bfw_compl[UL_MAX_CELLS_PER_SLOT];           ///< End of UL BFW completion processing per cell
    t_ns start_t_ul_pusch_cb[UL_MAX_CELLS_PER_SLOT];          ///< Start of PUSCH callback per cell
    t_ns end_t_ul_pusch_cb[UL_MAX_CELLS_PER_SLOT];            ///< End of PUSCH callback per cell
    t_ns start_t_ul_pucch_cb[UL_MAX_CELLS_PER_SLOT];          ///< Start of PUCCH callback per cell
    t_ns end_t_ul_pucch_cb[UL_MAX_CELLS_PER_SLOT];            ///< End of PUCCH callback per cell
    t_ns start_t_ul_prach_cb[UL_MAX_CELLS_PER_SLOT];          ///< Start of PRACH callback per cell
    t_ns end_t_ul_prach_cb[UL_MAX_CELLS_PER_SLOT];            ///< End of PRACH callback per cell
    t_ns start_t_ul_srs_cb[UL_MAX_CELLS_PER_SLOT];            ///< Start of SRS callback per cell
    t_ns end_t_ul_srs_cb[UL_MAX_CELLS_PER_SLOT];              ///< End of SRS callback per cell
    t_ns start_t_ul_bfw_cb[UL_MAX_CELLS_PER_SLOT];            ///< Start of UL BFW callback per cell
    t_ns end_t_ul_bfw_cb[UL_MAX_CELLS_PER_SLOT];              ///< End of UL BFW callback per cell
    t_ns start_t_ul_pusch_run[UL_MAX_CELLS_PER_SLOT];         ///< Start of PUSCH run per cell
    t_ns start_t_ul_bfw_run[UL_MAX_CELLS_PER_SLOT];           ///< Start of UL BFW run per cell
    t_ns start_t_ul_pucch_run[UL_MAX_CELLS_PER_SLOT];         ///< Start of PUCCH run per cell
    t_ns start_t_ul_prach_run[UL_MAX_CELLS_PER_SLOT];         ///< Start of PRACH run per cell
    t_ns start_t_ul_srs_run[UL_MAX_CELLS_PER_SLOT];           ///< Start of SRS run per cell
};

/**
 * @brief Uplink slot resource and task management container.
 *
 * Manages resources, channel aggregators (PUSCH, PUCCH, PRACH, SRS, UL BFW),
 * and synchronization for processing one uplink slot across multiple cells.
 * Coordinates multi-threaded execution of packet reception, symbol ordering,
 * channel processing, and L2 callbacks.
 */
class SlotMapUl {
public:
    /**
     * @brief Constructs an uplink slot map.
     *
     * @param _pdh cuPHYDriver handle
     * @param _id Unique slot map identifier
     */
    SlotMapUl(phydriver_handle _pdh, uint64_t _id);
    
    /**
     * @brief Destructor.
     */
    ~SlotMapUl();

    /**
     * @brief Reserves the slot map for use.
     *
     * Marks the slot map as active. Fails if already reserved.
     *
     * @return 0 on success, -1 if already active
     */
    int reserve();
    
    /**
     * @brief Releases the slot map and cleans up resources.
     *
     * Releases channel aggregators, clears cell/buffer lists, prints timing
     * information if enabled, resets atomic synchronization flags. Last thread
     * (based on num_cells) performs cleanup.
     *
     * @param num_cells Number of cells releasing this slot map
     * @param enable_task_run_times Enable printing of task execution timing information
     * @return 0 on success
     */
    int release(int num_cells, bool enable_task_run_times);

    /**
     * @brief Gets the cuPHYDriver handle.
     *
     * @return cuPHYDriver handle
     */
    phydriver_handle    getPhyDriverHandler(void) const;
    
    /**
     * @brief Gets the unique slot map identifier.
     *
     * @return Slot map ID
     */
    uint64_t            getId() const;

    /**
     * @brief Gets the dynamic beam ID offset for this slot.
     *
     * @return Dynamic beam ID offset
     */
    int16_t getDynBeamIdOffset() const;
    
    /**
     * @brief Sets the dynamic beam ID offset for this slot.
     *
     * @param beam_id_offset Dynamic beam ID offset
     */
    void    setDynBeamIdOffset(int16_t beam_id_offset);

    /**
     * @brief Checks if current task number has been reached.
     *
     * @param task_item Task item to check
     * @param tot_cells Total number of cells
     * @return 0 if not yet reached, non-zero if reached
     */
    int                 checkCurrentTask(int task_item, int tot_cells);
    
    /**
     * @brief Unlocks the next task for execution.
     *
     * @param task_item Task item to unlock
     * @param num_cells Number of cells
     * @return true if unlocked successfully, false otherwise
     */
    bool                unlockNextTask(int task_item, int num_cells);
    
    /**
     * @brief Signals task pipeline abort by setting task counter to -1.
     * 
     * Called in error paths to indicate critical failure. Other tasks can detect
     * this via tasksAborted() to stop processing.
     */
    void                abortTasks(void);
    
    /**
     * @brief Checks if tasks have been aborted.
     *
     * @return true if tasks aborted, false otherwise
     */
    bool                tasksAborted(void);
    
    /**
     * @brief Marks early UCI processing as complete.
     *
     * @return 0 on success
     */
    int                 setEarlyUciEndTask();
    
    /**
     * @brief Non-blocking check if early UCI processing is complete.
     *
     * @return true if complete, false if still processing
     */
    bool                waitEarlyUciEndTaskNonBlocking();
    
    /**
     * @brief Waits for early UCI processing to complete.
     *
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS)
     */
    int                 waitEarlyUciEndTask();
    
    /**
     * @brief Marks UL beamforming processing as complete.
     *
     * @return 0 on success
     */
    int                 setUlBfwEndTask();
    
    /**
     * @brief Waits for UL beamforming processing to complete.
     *
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS)
     */
    int                 waitUlBfwEndTask();
    
    /**
     * @brief Increments channel end task counter (thread-safe).
     *
     * @return Always returns 0
     */
    int                 addChannelEndTask();
    
    /**
     * @brief Waits for all channel end tasks to complete.
     *
     * @param num_channels Expected number of channels
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                 waitChannelEndTask(int num_channels);
    
    /**
     * @brief Increments UL C-plane tasks complete counter (thread-safe).
     *
     * @return Always returns 0
     */
    int                 addULCTasksComplete();
    
    /**
     * @brief Waits for UL C-plane tasks to complete.
     *
     * @param num_tasks Expected number of UL C-plane tasks
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                 waitULCTasksComplete(int num_tasks);
    
    /**
     * @brief Unlocks channel task for execution.
     *
     * @return true if unlocked successfully, false otherwise
     */
    bool                unlockChannelTask();
    
    /**
     * @brief Waits for channel start task signal.
     *
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                 waitChannelStartTask();
    
    /**
     * @brief Increments slot end task counter (thread-safe).
     *
     * @return Always returns 0
     */
    int                 addSlotEndTask();
    
    /**
     * @brief Waits for all slot end tasks to complete.
     *
     * @param num_tasks Expected number of slot end tasks
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                 waitSlotEndTask(int num_tasks);
    
    /**
     * @brief Adds a cell and its input buffers to the aggregated slot.
     *
     * @param c Cell configuration pointer
     * @param _phy_slot_params Slot-level PHY parameters (PRB/symbol info)
     * @param ulbuf_st1 UL input buffer type 1 (PUSCH/PUCCH)
     * @param ulbuf_st2 UL input buffer type 2 (SRS)
     * @param ul_bufst3_v Array of UL input buffer type 3 (PRACH occasions)
     * @param rach_occasion RACH occasion index
     * @param ulbuf_pcap UL PCAP capture buffer
     * @param ulbuf_pcap_ts UL PCAP timestamp buffer
     * @return 0 on success, EINVAL if null pointers, ENOMEM if exceeds UL_MAX_CELLS_PER_SLOT
     */
    int                 aggrSetCells(Cell* c, slot_command_api::phy_slot_params * _phy_slot_params, ULInputBuffer * ulbuf_st1,
                                     ULInputBuffer * ulbuf_st2,std::array<ULInputBuffer*, PRACH_MAX_OCCASIONS> ul_bufst3_v, const uint32_t rach_occasion, ULInputBuffer * ulbuf_pcap, ULInputBuffer * ulbuf_pcap_ts);
    
    /**
     * @brief Assigns the symbol ordering entity to this slot map.
     *
     * @param oentity OrderEntity pointer for symbol reordering
     * @return 0 on success, EINVAL if null pointer
     */
    int                 aggrSetOrderEntity(OrderEntity* oentity);
    
    /**
     * @brief Gets the symbol ordering entity.
     *
     * @return OrderEntity pointer (may be nullptr)
     */
    OrderEntity*        aggrGetOrderEntity();

    /**
     * @brief Assigns channel aggregators to this slot map.
     *
     * @param pusch PUSCH aggregator pointer (can be nullptr)
     * @param pucch PUCCH aggregator pointer (can be nullptr)
     * @param prach PRACH aggregator pointer (can be nullptr)
     * @param srs SRS aggregator pointer (can be nullptr)
     * @param ulbfw UL BFW aggregator pointer (can be nullptr)
     * @param aggr_slot_params Aggregated slot parameters
     * @return 0 on success, EINVAL if all aggregators are nullptr
     */
    int                 aggrSetPhy(PhyPuschAggr* pusch, PhyPucchAggr* pucch, PhyPrachAggr* prach, PhySrsAggr* srs, PhyUlBfwAggr* ulbfw,slot_params_aggr * aggr_slot_params);

    /**
     * @brief Gets the number of active cells in this slot.
     *
     * @return Number of active cells
     */
    int                                                                           getNumCells();
    
    /**
     * @brief Sets task execution timestamps.
     *
     * @param _tasks_num Number of tasks
     * @param _tasks_ts_exec Array of task execution timestamps
     * @param _tasks_ts_enq Task enqueue timestamp
     * @return 0 on success
     */
    int                                                                           setTasksTs(int _tasks_num, const std::array<t_ns, TASK_MAX_PER_SLOT + 1> _tasks_ts_exec, t_ns _tasks_ts_enq);
    
    /**
     * @brief Gets the task enqueue timestamp.
     *
     * @return Reference to task enqueue timestamp
     */
    t_ns&                                                                         getTaskTsEnq();
    
    /**
     * @brief Gets the task execution timestamp for a specific task.
     *
     * @param task_num Task number
     * @return Reference to task execution timestamp (returns empty for invalid task_num)
     */
    t_ns&                                                                         getTaskTsExec(int task_num);
    
    /**
     * @brief Clears all timing measurements.
     */
    void                                                                          cleanupTimes();
    
    /**
     * @brief Prints all timing measurements for this slot.
     *
     * @param isNonSrsUl Enable printing of non-SRS UL timings
     * @param isSrs Enable printing of SRS timings
     */
    void                                                                          printTimes(bool isNonSrsUl, bool isSrs);

    /**
     * @brief Sets the 3GPP slot indication for this slot map.
     *
     * @param si Slot indication (SFN, slot number)
     */
    void                                                                          setSlot3GPP(slot_command_api::slot_indication si);
    
    /**
     * @brief Gets the 3GPP slot indication.
     *
     * @return Slot indication structure
     */
    struct slot_command_api::slot_indication                                      getSlot3GPP() const;
    
    /**
     * @brief Gets the list of M-plane cell indices and count.
     *
     * @param cell_idx_list Output array of cell indices
     * @param pcellCount Pointer to receive cell count
     */
    void                                                                          getCellMplaneIdxList(std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& cell_idx_list, int *pcellCount);
    
    /**
     * @brief Checks if early HARQ is present in this slot.
     *
     * @return 1 if early HARQ present, 0 otherwise
     */
    uint8_t                                                                       getIsEarlyHarqPresent(){return isEarlyHarqPresent;}

    /**
     * Checks if early SCH codeblock decoding is present in this slot.
     *
     * @return 1 if early SCH codeblock decoding is present, 0 otherwise
     */
    uint8_t                                                                       getIsEarlySchCbDecodePresent(){return isEarlySchCbDecodePresent;}
    
    /**
     * @brief Checks if front-loaded DMRS is present in this slot.
     *
     * @return 1 if front-loaded DMRS present, 0 otherwise
     */
    uint8_t                                                                       getIsFrontLoadedDmrsPresent(){return isFrontLoadedDmrsPresent;}

    /**
     * Checks whether PUSCH has work in the subslot processing phase.
     *
     * @return true if early HARQ, early SCH codeblock decoding, or front-loaded DMRS processing is present
     */
    bool                                                                          hasPuschSubSlotWork(){return isEarlyHarqPresent || isEarlySchCbDecodePresent || isFrontLoadedDmrsPresent;}
    
    /**
     * @brief Sets whether early HARQ is present.
     *
     * @param val 1 if present, 0 otherwise
     */
    void                                                                          setIsEarlyHarqPresent(uint8_t val){isEarlyHarqPresent=val;}

    /**
     * Sets whether early SCH codeblock decoding is present.
     *
     * @param val 1 if present, 0 otherwise
     */
    void                                                                          setIsEarlySchCbDecodePresent(uint8_t val){isEarlySchCbDecodePresent=val;}
    
    /**
     * @brief Sets whether front-loaded DMRS is present.
     *
     * @param val 1 if present, 0 otherwise
     */
    void                                                                          setIsFrontLoadedDmrsPresent(uint8_t val){isFrontLoadedDmrsPresent=val;}
    
    /**
     * @brief Gets the T0 timestamp (first task execution timestamp).
     *
     * @return T0 timestamp
     */
    t_ns                                                                          get_t0() {return tasks_ts_exec[0];}

    ////////////////////////////////////////////
    //// Public Members
    ////////////////////////////////////////////
    ul_slot_timings  timings;                                   ///< Timing measurements for all UL processing stages
    MemFoot          mf;                                        ///< Memory footprint tracking for this slot map

    PhyPuschAggr * aggr_pusch;                                  ///< PUSCH aggregator pointer (nullptr if not scheduled)
    PhyPucchAggr * aggr_pucch;                                  ///< PUCCH aggregator pointer (nullptr if not scheduled)
    PhyPrachAggr * aggr_prach;                                  ///< PRACH aggregator pointer (nullptr if not scheduled)
    PhyUlBfwAggr * aggr_ulbfw;                                  ///< UL beamforming weight aggregator pointer (nullptr if not scheduled)
    PhySrsAggr   * aggr_srs;                                    ///< SRS aggregator pointer (nullptr if not scheduled)
    slot_params_aggr * aggr_slot_params;                        ///< Aggregated slot parameters for all channels
    std::vector<Cell *> aggr_cell_list;                         ///< List of cells scheduled in this slot
    std::vector<ULInputBuffer *> aggr_ulbuf_st1;                ///< UL input buffers type 1 (PUSCH/PUCCH)
    std::vector<ULInputBuffer *> aggr_ulbuf_st2;                ///< UL input buffers type 2 (SRS)
    std::vector<ULInputBuffer *> aggr_ulbuf_st3;                ///< UL input buffers type 3 (PRACH occasions)
    std::vector<ULInputBuffer *> aggr_ulbuf_pcap_capture;       ///< PCAP capture buffers per cell
    std::vector<ULInputBuffer *> aggr_ulbuf_pcap_capture_ts;    ///< PCAP timestamp buffers per cell
    std::vector<int> num_prach_occa;                            ///< Number of PRACH occasions per cell
    OrderEntity * aggr_order_entity;                            ///< Symbol ordering entity pointer
    slot_command_api::slot_info_t* aggr_slot_info[UL_MAX_CELLS_PER_SLOT]; ///< Slot info (PRB/symbol allocations) per cell
    std::atomic<int>   atom_ul_cplane_info_for_uplane_rdy_count; ///< Atomic counter for C-plane to U-plane readiness synchronization

    /**
     * @brief Path-aware count of UL C-plane contributors for this slot.
     *
     * Legacy path: number of @c TaskUL1AggrCplane tasks pushed by
     * @c l1_enqueue_phy_work (= @c get_num_ulc_tasks(num_ul_workers)).  Set
     * once by @c l1_enqueue_phy_work after it computes the worker-formula value.
     *
     * Framework path (@c SKIP_UL_CPLANE): number of @c task_work_fn_cplane_batch
     * invocations actually fired by @c PHY_module::fire_cplane_batch for this
     * slot — i.e., the running batch_count.  Updated by @c fire_cplane_batch as
     * each batch is scheduled; final value is established by EOM.
     *
     * Read by UL task bodies (Order Kernel, PUCCH+PUSCH, Early UCI Ind, UL3) in
     * place of the legacy @c get_num_ulc_tasks(pdctx->getNumULWorkers()) call so
     * that bodies are path-aware without per-call branching.
     *
     * Reset to 0 in @c release().
     *
     * @return Path-aware UL C-plane contributor count for the current slot.
     *         Valid only after the producer (l1_enqueue_phy_work for legacy,
     *         fire_cplane_batch for framework) has populated it.
     */
    [[nodiscard]] int getNumUlcTasks() const noexcept { return numUlcTasks_; }

    /**
     * @brief Sets the path-aware UL C-plane contributor count.
     * @param v New count.  See @c getNumUlcTasks() for semantics.
     */
    void setNumUlcTasks(int v) noexcept { numUlcTasks_ = v; }

    /**
     * UL slot reference timestamp (L2A slot tick + 1 slot).
     *
     * Dedicated slot-reference time, kept separate from @c tasks_ts_exec so that
     * task dispatch timing is not conflated with the reference time. Consumers
     * that need the "L2A slot tick + 1" reference (Order Kernel body, framework
     * @c send_ul_cplane) read this via @c getSlotRefTs() instead of
     * @c getTaskTsExec(0): on the
     * framework (SKIP_UL_CPLANE) path the compacted layout no longer places the
     * reference value at index 0. Set by the UL producer
     * (@c l1_enqueue_phy_work / @c l1_setup_early_cplane_slot_maps).
     *
     * @return Slot reference timestamp stored in @c slotRefTs_.
     *         Return value must be checked.
     */
    [[nodiscard]] t_ns getSlotRefTs() const noexcept { return slotRefTs_; }

    /**
     * Sets the UL slot reference timestamp. See @c getSlotRefTs().
     *
     * @param[in] v Slot reference timestamp to store.
     */
    void setSlotRefTs(t_ns v) noexcept { slotRefTs_ = v; }

    /**
     * Total UL task count for this slot (the @c waitSlotEndTask target).
     *
     * Published by @c l1_enqueue_phy_work via @c setTasksTs() before the tasks are
     * pushed to the UL task list, so it is visible to task bodies once they run
     * (the task-list lock provides the happens-before).  UL3 / UL3-SRS read this
     * instead of carrying the count in their @c init() argument.
     *
     * @return Total UL task count from @c tasks_num.
     *         Return value must be checked.
     */
    [[nodiscard]] int getNumTasks() const noexcept { return tasks_num; }

private:
    ////////////////////////////////////////////
    //// Private Members
    ////////////////////////////////////////////
    phydriver_handle                                         pdh;                         ///< cuPHYDriver handle
    uint64_t                                                 id;                          ///< Unique slot map identifier
    int16_t                                                  dyn_beam_id_offset;          ///< Dynamic beam ID offset
    std::atomic<int>                                         task_current_number;         ///< Current task number (atomic)
    std::atomic<int>                                         atom_ul_channel_end_threads; ///< Atomic counter: threads finished channel processing
    std::atomic<int>                                         atom_ul_end_threads;         ///< Atomic counter: threads finished slot processing
    std::atomic<int>                                         atom_ulc_tasks_complete;     ///< Atomic counter: UL C-plane tasks completed
    int                                                      numUlcTasks_                = 0; ///< Path-aware UL C-plane contributor count: legacy = num_ulc_tasks (workers), framework = batch_count. See getNumUlcTasks().
    t_ns                                                     slotRefTs_{};                ///< UL slot reference time (L2A tick + 1 slot). See getSlotRefTs().
    std::atomic<bool>                                        run_order_done;              ///< Atomic flag: symbol ordering completed
    std::atomic<bool>                                        early_uci_task_done;         ///< Atomic flag: early UCI processing completed
    std::atomic<bool>                                        ulbfw_task_done;             ///< Atomic flag: UL beamforming processing completed
    int                                                      tasks_num;                   ///< Total number of tasks
    std::array<t_ns, TASK_MAX_PER_SLOT + 1>                  tasks_ts_exec;               ///< Task execution timestamps
    t_ns                                                     tasks_ts_enq;                ///< Task enqueue timestamp
    t_ns                                                     empty;                       ///< Fallback zero timestamp returned for invalid task numbers in getTaskTsExec()
    std::array<std::pair<t_ns, t_ns>, TASK_MAX_PER_SLOT + 1> tasks_ts_record;            ///< Task timestamp records (start/end pairs)
    std::atomic<bool>                                        atom_active;                 ///< Atomic flag: slot map is active/reserved
    std::atomic<int>                                         atom_num_cells;              ///< Atomic counter: number of cells processed
    struct slot_command_api::slot_indication                 slot_3gpp;                   ///< 3GPP slot indication (SFN, slot number)
    uint8_t                                                  isEarlyHarqPresent;          ///< Flag: early HARQ present in this slot (1=yes, 0=no)
    uint8_t                                                  isEarlySchCbDecodePresent;   ///< Flag: early SCH codeblock decoding present in this slot (1=yes, 0=no)
    uint8_t                                                  isFrontLoadedDmrsPresent;    ///< Flag: front-loaded DMRS present in this slot (1=yes, 0=no)
    
};

#endif
