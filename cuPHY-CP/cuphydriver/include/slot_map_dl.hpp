/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SLOT_MAP_DL_H
#define SLOT_MAP_DL_H

#include <iostream>
#include <atomic>
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "time.hpp"
#include "phychannel.hpp"
#include "nvlog.hpp"
#include "dlbuffer.hpp"
#include "phypdsch_aggr.hpp"
#include "phypdcch_aggr.hpp"
#include "phypbch_aggr.hpp"
#include "phycsirs_aggr.hpp"
#include "phydlbfw_aggr.hpp"


/**
 * @brief Timing measurements for downlink slot processing stages.
 *
 * Captures start/end timestamps for each DL channel processing phase
 * (setup, run, completion) and U-plane preparation/transmission stages.
 */
struct dl_slot_timings {
    t_ns start_t_dl_cplane[DL_MAX_CELLS_PER_SLOT];            ///< Start of C-plane processing per cell
    t_ns end_t_dl_cplane[DL_MAX_CELLS_PER_SLOT];              ///< End of C-plane processing per cell
    t_ns start_t_dl_csirs_setup[DL_MAX_CELLS_PER_SLOT];       ///< Start of CSI-RS setup per cell
    t_ns end_t_dl_csirs_setup[DL_MAX_CELLS_PER_SLOT];         ///< End of CSI-RS setup per cell
    t_ns start_t_dl_pdsch_setup[DL_MAX_CELLS_PER_SLOT];       ///< Start of PDSCH setup per cell
    t_ns end_t_dl_pdsch_setup[DL_MAX_CELLS_PER_SLOT];         ///< End of PDSCH setup per cell
    t_ns start_t_dl_bfw_setup[DL_MAX_CELLS_PER_SLOT];         ///< Start of DL BFW setup per cell
    t_ns end_t_dl_bfw_setup[DL_MAX_CELLS_PER_SLOT];           ///< End of DL BFW setup per cell
    t_ns start_t_dl_pdcchdl_setup[DL_MAX_CELLS_PER_SLOT];     ///< Start of PDCCH DL setup per cell
    t_ns end_t_dl_pdcchdl_setup[DL_MAX_CELLS_PER_SLOT];       ///< End of PDCCH DL setup per cell
    t_ns start_t_dl_pdcchul_setup[DL_MAX_CELLS_PER_SLOT];     ///< Start of PDCCH UL setup per cell
    t_ns end_t_dl_pdcchul_setup[DL_MAX_CELLS_PER_SLOT];       ///< End of PDCCH UL setup per cell
    t_ns start_t_dl_pbch_setup[DL_MAX_CELLS_PER_SLOT];        ///< Start of PBCH setup per cell
    t_ns end_t_dl_pbch_setup[DL_MAX_CELLS_PER_SLOT];          ///< End of PBCH setup per cell
    t_ns start_t_dl_csirs_compl[DL_MAX_CELLS_PER_SLOT];       ///< Start of CSI-RS completion processing per cell
    t_ns end_t_dl_csirs_compl[DL_MAX_CELLS_PER_SLOT];         ///< End of CSI-RS completion processing per cell
    t_ns start_t_dl_pdsch_compl[DL_MAX_CELLS_PER_SLOT];       ///< Start of PDSCH completion processing per cell
    t_ns end_t_dl_pdsch_compl[DL_MAX_CELLS_PER_SLOT];         ///< End of PDSCH completion processing per cell
    t_ns start_t_dl_pdcchdl_compl[DL_MAX_CELLS_PER_SLOT];     ///< Start of PDCCH DL completion processing per cell
    t_ns end_t_dl_pdcchdl_compl[DL_MAX_CELLS_PER_SLOT];       ///< End of PDCCH DL completion processing per cell
    t_ns start_t_dl_pdcchul_compl[DL_MAX_CELLS_PER_SLOT];     ///< Start of PDCCH UL completion processing per cell
    t_ns end_t_dl_pdcchul_compl[DL_MAX_CELLS_PER_SLOT];       ///< End of PDCCH UL completion processing per cell
    t_ns start_t_dl_pbch_compl[DL_MAX_CELLS_PER_SLOT];        ///< Start of PBCH completion processing per cell
    t_ns end_t_dl_pbch_compl[DL_MAX_CELLS_PER_SLOT];          ///< End of PBCH completion processing per cell
    t_ns start_t_dl_bfw_compl[DL_MAX_CELLS_PER_SLOT];         ///< Start of DL BFW completion processing per cell
    t_ns end_t_dl_bfw_compl[DL_MAX_CELLS_PER_SLOT];           ///< End of DL BFW completion processing per cell
    t_ns start_t_dl_uprep[DL_MAX_CELLS_PER_SLOT];             ///< Start of U-plane data preparation per cell
    t_ns end_t_dl_uprep[DL_MAX_CELLS_PER_SLOT];               ///< End of U-plane data preparation per cell
    t_ns start_t_dl_utx[DL_MAX_CELLS_PER_SLOT];               ///< Start of U-plane transmission per cell
    t_ns end_t_dl_utx[DL_MAX_CELLS_PER_SLOT];                 ///< End of U-plane transmission per cell
    t_ns start_t_dl_callback;                                  ///< Start of DL callback processing
    t_ns end_t_dl_callback;                                    ///< End of DL callback processing
    t_ns start_t_dl_bfw_cb[DL_MAX_CELLS_PER_SLOT];            ///< Start of BFW callback per cell
    t_ns end_t_dl_bfw_cb[DL_MAX_CELLS_PER_SLOT];              ///< End of BFW callback per cell
    t_ns start_t_dl_compression_cuda;                          ///< Start of CUDA compression
    t_ns end_t_dl_compression_cuda;                            ///< End of CUDA compression
    t_ns start_t_dl_compression_compl;                         ///< Start of compression completion
    t_ns end_t_dl_compression_compl;                           ///< End of compression completion
    float prepare_execution_duration1[MAX_NUM_OF_NIC_SUPPORTED];  ///< GPU packet preparation phase 1: setup and memory copy per NIC (microseconds, GPU direct comm only)
    float prepare_execution_duration2[MAX_NUM_OF_NIC_SUPPORTED];  ///< GPU packet preparation phase 2: pre-preparation kernel per NIC (microseconds, GPU direct comm only)
    float prepare_execution_duration3[MAX_NUM_OF_NIC_SUPPORTED];  ///< GPU packet preparation phase 3: final preparation kernel per NIC (microseconds, GPU direct comm only)
    float prePrepare_to_compression_gap[MAX_NUM_OF_NIC_SUPPORTED]; ///< CPU-measured wall-clock gap between pre-preparation completion and compression start per NIC (microseconds)
    float channel_to_compression_gap;                          ///< Gap between channel processing and compression (microseconds)
    float compression_execution_duration;                      ///< Compression execution duration (microseconds)
    float packet_mem_copy_per_symbol_dur_us[ORAN_MAX_SYMBOLS]; ///< Packet memory copy duration per OFDM symbol (microseconds)
    t_ns start_t_dl_pdsch_run[DL_MAX_CELLS_PER_SLOT];         ///< Start of PDSCH run per cell
    t_ns start_t_dl_bfw_run[DL_MAX_CELLS_PER_SLOT];           ///< Start of BFW run per cell
    t_ns start_t_dl_csirs_run[DL_MAX_CELLS_PER_SLOT];         ///< Start of CSI-RS run per cell
    t_ns start_t_dl_pdcchdl_run[DL_MAX_CELLS_PER_SLOT];       ///< Start of PDCCH DL run per cell
    t_ns start_t_dl_pdcchul_run[DL_MAX_CELLS_PER_SLOT];       ///< Start of PDCCH UL run per cell
    t_ns start_t_dl_pbch_run[DL_MAX_CELLS_PER_SLOT];          ///< Start of PBCH run per cell
    t_ns end_t_dl_pdsch_run[DL_MAX_CELLS_PER_SLOT];           ///< End of PDSCH run per cell
    t_ns end_t_dl_bfw_run[DL_MAX_CELLS_PER_SLOT];             ///< End of BFW run per cell
    t_ns end_t_dl_csirs_run[DL_MAX_CELLS_PER_SLOT];           ///< End of CSI-RS run per cell
    t_ns end_t_dl_pdcchdl_run[DL_MAX_CELLS_PER_SLOT];         ///< End of PDCCH DL run per cell
    t_ns end_t_dl_pdcchul_run[DL_MAX_CELLS_PER_SLOT];         ///< End of PDCCH UL run per cell
    t_ns end_t_dl_pbch_run[DL_MAX_CELLS_PER_SLOT];            ///< End of PBCH run per cell
};

/**
 * @brief Downlink slot resource and task management container.
 *
 * Manages resources, channel aggregators (PDSCH, PDCCH, PBCH, CSI-RS, DL BFW),
 * and synchronization for processing one downlink slot across multiple cells.
 * Coordinates multi-threaded execution of channel setup, GPU processing, and
 * U-plane transmission stages.
 */
class SlotMapDl {
public:
    /**
     * @brief Constructs a downlink slot map.
     *
     * @param _pdh cuPHYDriver handle
     * @param _id Unique slot map identifier
     * @param _enableBatchedMemcpy Enable batched memory copy optimization (0=disabled, 1=enabled)
     */
    SlotMapDl(phydriver_handle _pdh, uint64_t _id, uint8_t _enableBatchedMemcpy);
    
    /**
     * @brief Destructor.
     */
    ~SlotMapDl();

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
     * Releases channel aggregators, clears cell/buffer lists, resets atomic
     * synchronization flags. Last thread (based on num_cells) performs cleanup.
     *
     * @param num_cells Number of cells releasing this slot map
     * @return 0 on success
     */
    int release(int num_cells);

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
     * @brief Adds a cell and its output buffer to the aggregated slot.
     *
     * @param c Cell configuration pointer
     * @param _phy_slot_params Slot-level PHY parameters (PRB/symbol info)
     * @param dlbuf Downlink output buffer for this cell
     * @return 0 on success, EINVAL if null pointers, ENOMEM if exceeds DL_MAX_CELLS_PER_SLOT
     */
    int                                                                           aggrSetCells(Cell* c, slot_command_api::phy_slot_params * _phy_slot_params, DLOutputBuffer * dlbuf);
    
    /**
     * @brief Assigns channel aggregators to this slot map.
     *
     * @param pdsch PDSCH aggregator pointer (can be nullptr)
     * @param pdcch_dl PDCCH DL aggregator pointer (can be nullptr)
     * @param pdcch_ul PDCCH UL aggregator pointer (can be nullptr)
     * @param aggr_pbch PBCH aggregator pointer (can be nullptr)
     * @param csirs CSI-RS aggregator pointer (can be nullptr)
     * @param dlbfw DL BFW aggregator pointer (can be nullptr)
     * @param _aggr_slot_params Aggregated slot parameters
     * @return 0 on success, EINVAL if all aggregators are nullptr
     */
    int                                                                           aggrSetPhy(PhyPdschAggr* pdsch, PhyPdcchAggr * pdcch_dl, PhyPdcchAggr * pdcch_ul, PhyPbchAggr * aggr_pbch, PhyCsiRsAggr * csirs,PhyDlBfwAggr* dlbfw ,slot_params_aggr * _aggr_slot_params);
    
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
     * @return Reference to task execution timestamp
     */
    t_ns&                                                                         getTaskTsExec(int task_num);
    
    /**
     * @brief Clears all timing measurements.
     */
    void                                                                          cleanupTimes();
    
    /**
     * @brief Prints all timing measurements for this slot.
     */
    void                                                                          printTimes();

    /**
     * @brief Waits for fronthaul callback (FHCB) completion.
     *
     * Blocks until all fronthaul processing is done or timeout.
     *
     * @return 0 on success, -1 on timeout (2x GENERIC_WAIT_THRESHOLD_NS)
     */
    int                                                                           waitFHCBDone();
    
    /**
     * @brief Waits for fronthaul callback completion for a specific cell.
     *
     * @param cell Cell index
     * @return 0 on success, -1 on timeout (2x GENERIC_WAIT_THRESHOLD_NS)
     */
    int                                                                           waitCellFHCBDone(int cell);
    
    /**
     * @brief Marks fronthaul callback as done.
     */
    void                                                                          setFHCBDone();
    
    /**
     * @brief Marks fronthaul callback as done for a specific cell.
     *
     * @param cell Cell index
     */
    void                                                                          setCellFHCBDone(int cell);
    
    /**
     * @brief Waits for downlink C-plane (DLC) tasks to complete.
     *
     * Used only when mMIMO is enabled to synchronize parallel DL C-plane
     * processing completion before starting U-plane packet preparation.
     *
     * @param num_dlc_tasks Expected number of DL C-plane tasks
     * @return 0 on success, -1 on timeout (2x GENERIC_WAIT_THRESHOLD_NS)
     */
    int                                                                           waitDLCDone(int num_dlc_tasks);
    
    /**
     * @brief Waits for peer update completion
     *
     * @return 0 on success, -1 on timeout
     */
    int                                                                           waitPeerUpdateDone();
    
    /**
     * @brief Increments the downlink C-plane done counter (thread-safe).
     */
    void                                                                          incDLCDone();
    
    /**
     * @brief Increments the U-plane preparation done counter (thread-safe).
     */
    void                                                                          incUplanePrepDone();
    
    /**
     * @brief Waits for U-plane preparation tasks to complete.
     *
     * @param num_uplane_prep_tasks Expected number of U-plane preparation tasks
     * @return 0 on success, -1 on timeout (2x GENERIC_WAIT_THRESHOLD_NS)
     */
    int                                                                           waitUplanePrepDone(int num_uplane_prep_tasks);

    /**
     * @brief Marks C-plane as done for a specific task number.
     *
     * Sets the corresponding bit in atom_cplane_done_mask. Used for
     * synchronization when DL affinity is disabled.
     *
     * @param task_num Task number (0-31)
     */
    void                                                                          setCplaneDoneForTask(int task_num);

    /**
     * @brief Waits for C-plane to complete for a specific task number.
     *
     * Waits until the corresponding bit in atom_cplane_done_mask is set.
     * Used for synchronization when DL affinity is disabled.
     *
     * @param task_num Task number (0-31)
     * @return 0 on success, -1 on timeout (2x GENERIC_WAIT_THRESHOLD_NS)
     */
    int                                                                           waitCplaneDoneForTask(int task_num);

    /**
     * @brief Waits for all threads to reach C-plane processing stage.
     *
     * Synchronization barrier for multi-threaded C-plane start.
     *
     * @param num_threads Expected number of threads
     * @return 0 on success, -1 on timeout (2x GENERIC_WAIT_THRESHOLD_NS)
     */
    int                                                                           waitCplaneReady(int num_threads);
    
    /**
     * @brief Marks slot end for specified number of cells (thread-safe counter).
     *
     * @param num_cells Number of cells finished
     * @return 0 on success
     */
    int                                                                           addSlotEnd(int num_cells);
    
    /**
     * @brief Increments slot end task counter.
     *
     * @return Always returns 0
     */
    int                                                                           addSlotEndTask();
    
    /**
     * @brief Increments slot channel end counter.
     *
     * @return Always returns 0
     */
    int                                                                           addSlotChannelEnd();
    
    /**
     * @brief Waits for all slot end tasks to complete.
     *
     * @param num_tasks Expected number of slot end tasks
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                                                                           waitSlotEndTask(int num_tasks);
    
    /**
     * @brief Waits for all channel end signals.
     *
     * @param num_channels Expected number of channels
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                                                                           waitSlotChannelEnd(int num_channels);
    
    /**
     * @brief Marks DL CPU doorbell task as done.
     *
     * @return 0
     */
    int                                                                           setDlCpuDoorBellTaskDone();
    
    /**
     * @brief Marks DL GPU initiated communication finished.
     *
     * @return 0
     */
    int                                                                           setDlGpuCommEnd();
    
    /**
     * @brief Marks DL compression as ended.
     *
     * @return 0
     */
    int                                                                           setDlCompEnd();
    
    /**
     * @brief Waits for DL GPU initiated communication to end.
     *
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                                                                           waitDlGpuCommEnd();
    
    /**
     * @brief Waits for DL CPU doorbell task to complete.
     *
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                                                                           waitDlCpuDoorBellTaskDone();
    
    /**
     * @brief Waits for DL compression to complete.
     *
     * @return 0 on success, -1 on timeout (GENERIC_WAIT_THRESHOLD_NS * 2)
     */
    int                                                                           waitDlCompEnd();
    
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
    const struct slot_command_api::slot_indication                                getSlot3GPP() const;
    
    /**
     * @brief Gets the list of M-plane cell indices and count.
     *
     * @param cell_idx_list Output array of cell indices
     * @param pcellCount Pointer to receive cell count
     */
    void                                                                          getCellMplaneIdxList(std::array<uint32_t,DL_MAX_CELLS_PER_SLOT>& cell_idx_list, int *pcellCount);
    
    /**
     * @brief Gets the batched memcpy helper for this slot.
     *
     * @return Reference to batched memcpy helper
     */
    cuphyBatchedMemcpyHelper&                                                     getBatchedMemcpyHelper(){ return m_batchedMemcpyHelper; }
    
    ////////////////////////////////////////////
    //// Public Members
    ////////////////////////////////////////////
    dl_slot_timings  timings;                                   ///< Timing measurements for all DL processing stages
    MemFoot          mf;                                        ///< Memory footprint tracking for this slot map

    PhyPdschAggr * aggr_pdsch;                                  ///< PDSCH aggregator pointer (nullptr if not scheduled)
    PhyPdcchAggr * aggr_pdcch_dl;                               ///< PDCCH DL aggregator pointer (nullptr if not scheduled)
    PhyPdcchAggr * aggr_pdcch_ul;                               ///< PDCCH UL (DCI for UL grants) aggregator pointer (nullptr if not scheduled)
    PhyPbchAggr * aggr_pbch;                                    ///< PBCH aggregator pointer (nullptr if not scheduled)
    PhyCsiRsAggr * aggr_csirs;                                  ///< CSI-RS aggregator pointer (nullptr if not scheduled)
    PhyDlBfwAggr * aggr_dlbfw;                                  ///< DL beamforming weight aggregator pointer (nullptr if not scheduled)
    PhyDlBfwAggr * aggr_ulbfw;                                  ///< UL beamforming weight aggregator pointer (currently unused)
    std::vector<Cell *> aggr_cell_list;                         ///< List of cells scheduled in this slot
    std::vector<DLOutputBuffer *> aggr_dlbuf_list;              ///< List of DL output buffers (one per cell)
    TxRequestGpuPercell tx_v_for_slot_map[MAX_NUM_OF_NIC_SUPPORTED];  ///< TX request structures per NIC
    slot_command_api::slot_info_t* aggr_slot_info[DL_MAX_CELLS_PER_SLOT];  ///< Slot info (PRB/symbol allocations) per cell
    slot_params_aggr * aggr_slot_params;                        ///< Aggregated slot parameters for all channels
    bool               pdsch_cb_done;                           ///< Flag indicating PDSCH callback completion
    std::atomic<int>   atom_dl_cplane_info_for_uplane_rdy_count; ///< Atomic counter for C-plane to U-plane readiness synchronization

private:
    ////////////////////////////////////////////
    //// Private Members
    ////////////////////////////////////////////
    phydriver_handle                                         pdh;                 ///< cuPHYDriver handle
    uint64_t                                                 id;                  ///< Unique slot map identifier
    int16_t                                                  dyn_beam_id_offset;  ///< Dynamic beam ID offset
    struct downlink_slot_info*                               dl_si;               ///< Downlink slot info (legacy/unused)
    std::atomic<int>                                         task_current_number; ///< Current task number (atomic)
    int                                                      tasks_num;           ///< Total number of tasks
    std::array<t_ns, TASK_MAX_PER_SLOT + 1>                  tasks_ts_exec;       ///< Task execution timestamps
    t_ns                                                     tasks_ts_enq;        ///< Task enqueue timestamp
    t_ns                                                     empty;               ///< Fallback zero timestamp returned for invalid task numbers in getTaskTsExec()
    std::array<std::pair<t_ns, t_ns>, TASK_MAX_PER_SLOT + 1> tasks_ts_record;    ///< Task timestamp records (start/end pairs)
    std::atomic<bool>                                        atom_active;         ///< Atomic flag: slot map is active/reserved
    std::atomic<int>                                         atom_num_cells;      ///< Atomic counter: number of cells processed

    std::atomic<bool>                                        atom_fhcb_done;      ///< Atomic flag: fronthaul callback completed
    std::array<std::atomic<bool>, DL_MAX_CELLS_PER_SLOT>     atom_cell_fhcb_done; ///< Atomic flags: fronthaul callback completed per cell
    std::atomic<int>                                         atom_dlc_done;       ///< Atomic counter: DL C-plane tasks completed (synchronized via waitDLCDone() in mMIMO mode)
    std::atomic<int>                                         atom_uplane_prep_done; ///< Atomic counter: U-plane preparation tasks completed
    std::atomic<uint32_t>                                    atom_cplane_done_mask; ///< Atomic bitmask: C-plane done per task_num (bit N = task_num N done)

    std::atomic<int>                                         atom_dl_cplane_start; ///< Atomic counter: threads started C-plane processing
    std::atomic<int>                                         atom_dl_end_threads;  ///< Atomic counter: threads finished slot processing
    std::atomic<int>                                         atom_dl_channel_end_threads; ///< Atomic counter: threads finished channel processing
    std::atomic<bool>                                        atom_dl_gpu_comm_end; ///< Atomic flag: DL GPU communication ended
    std::atomic<bool>                                        atom_dl_cpu_door_bell_task_done; ///< Atomic flag: DL CPU doorbell task done
    std::atomic<bool>                                        atom_dl_comp_end;     ///< Atomic flag: DL compression ended
    int                                                      num_active_cells;     ///< Number of active cells in this slot
    struct slot_command_api::slot_indication                 slot_3gpp;            ///< 3GPP slot indication (SFN, slot number)
    cuphyBatchedMemcpyHelper                                 m_batchedMemcpyHelper; ///< Batched memcpy helper object
};

#endif
