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

#ifndef CUMAC_TASK_HPP_
#define CUMAC_TASK_HPP_

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <semaphore.h>
#include <vector>

#include "cumac_app.hpp"
#include "cumac_cp_tv.hpp"
#include "cumac.h"
#include "api.h"
#include "cumac_msg.h"
#include "muMimoUserPairing/muMimoUserPairing.cuh"

#include "nv_phy_mac_transport.hpp"

//! Debug option: Print cuMAC buffer contents
static constexpr uint32_t DBG_OPT_PRINT_CUMAC_BUF{0x1};
//! Debug option: Print NVIPC buffer contents
static constexpr uint32_t DBG_OPT_PRINT_NVIPC_BUF{0x2}; 
//! Debug option: Compare group test vector buffer
static constexpr uint32_t DBG_OPT_COMPARE_GROUP_TV_BUF{0x4};
//! Debug option: Dump MU UE-pair GPU buffers (chan_est, snr, chan_orth) to HDF5 per slot
static constexpr uint32_t DBG_OPT_DUMP_UE_PAIR_H5{0x8};
//! Debug option: Workaround to copy group test vector
static constexpr uint32_t DBG_OPT_WAR_COPY_GROUP_TV{0x80};

//! Maximum buffer size for cuMAC task data (1 MB)
static constexpr std::size_t CUMAC_TASK_BUF_MAX_SIZE{1024 * 1024};
//! Maximum buffer size for debug logging (10 MB)
static constexpr std::size_t CUMAC_TASK_DEBUG_BUF_MAX_SIZE{1024 * 1024 * 100};

class cumac_cp_handler;
class cumac_task;

/**
 * Orders GPU-side task processing in enqueue FIFO across multiple worker threads.
 *
 * Maintains one POSIX semaphore per worker thread laid out in a ring. Each enqueued
 * task is assigned the current and next semaphore for the sequence; workers call
 * sem_wait(current) before setup and sem_post(next) after GPU work completes so only
 * one ordered slot uses shared GPU resources at a time while workers can still dequeue
 * in any order.
 */
class order_sem
{
public:
    /**
     * @param num_worker_threads Number of worker threads (and ring semaphores); if 0, uses 1.
     */
    explicit order_sem(std::size_t num_worker_threads);

    ~order_sem();

    order_sem(const order_sem&)            = delete;
    order_sem& operator=(const order_sem&) = delete;

    /**
     * Set cumac_task ordering semaphores for the next enqueue. No-op (nullptrs) when
     * slot_concurrent_enable is set on the task or when this object has no semaphores.
     */
    void assign_for_enqueue(cumac_task& task);

    std::size_t chain_depth() const noexcept { return sems_.size(); }

private:
    std::vector<sem_t> sems_{};
    std::atomic<uint64_t> next_seq_{0};
};

/**
 * Cell descriptor structure
 *
 * Contains per-cell buffer pointers and offsets for cuMAC processing
 */
typedef struct _cell_desc
{
    uint8_t *home{}; //!< Pointer to cell's home buffer in GPU memory
    cumac_tti_req_buf_offsets_t offsets{}; //!< Buffer offsets for TTI request data
    uint32_t muUeGrpReqCopyLen{}; //!< Bytes valid at offsets.muUeGrpReqInfo (<= per-cell stride) for group-buf scatter
} cell_desc_t;

/**
 * cuMAC task information structure
 *
 * Stores group-level parameters and scheduling solutions for GPU processing
 */
typedef struct _cumac_task_info
{
    cumac::cumacCellGrpUeStatus ueStatus{}; //!< UE status for cell group
    cumac::cumacSchdSol schdSol{}; //!< Scheduling solution for cell group
    cumac::cumacCellGrpPrms grpPrms{}; //!< Cell group parameters
    cumac_buf_num_t data_num{}; //!< Data buffer element counts
    cumac_pfm_cell_info_t *pfmCellInfo{}; //!< PFM sorting input buffer
    cumac_muUeGrp_req_info_t *muUeGrpInfo{}; //!< MU UE group input buffer
} cumac_task_info_t;

/**
 * cuMAC task execution context
 *
 * Encapsulates all state and resources for executing a single cuMAC scheduling task.
 * Tasks are processed through setup, run, and callback phases with timing measurements.
 */
class cumac_task final
{

public:
    /**
     * Construct cuMAC task instance
     *
     * Initializes all pointer members to nullptr, creates CUDA events for timing,
     * and assigns a unique task ID
     */
    cumac_task();

    /**
     * Reset task for new slot processing
     *
     * @param[in] _ss System Frame Number and slot identifier
     */
    void reset_cumac_task(sfn_slot_t _ss);

    /**
     * Initialize cuMAC processing modules
     *
     * Creates GPU or CPU instances of scheduler modules based on configuration
     */
    void init_cumac_modules();

    /**
     * Calculate output data element counts
     *
     * Computes the number of elements in each output buffer for memory allocation
     */
    void calculate_output_data_num();

    /**
     * Setup phase: prepare data for cuMAC processing
     *
     * @return 0 on success, negative on error
     */
    int setup();

    /**
     * Run phase: execute cuMAC algorithms
     *
     * @return 0 on success, negative on error
     */
    int run();

    /**
     * Callback phase: process results and send responses
     *
     * @return 0 on success, negative on error
     */
    int callback();

    /**
     * Validate buffer setup against test vectors
     *
     * @return 0 on success, negative on validation failure
     */
    int validate_buffer_setup();

    /**
     * Validate buffer callback results against test vectors
     *
     * @return 0 on success, negative on validation failure
     */
    int validate_buffer_callback();

    /**
     * Print array contents for debugging
     *
     * @param[in] info Descriptive label for the array
     * @param[in] array Pointer to array data
     * @param[in] num Number of elements to print
     * @param[in] data_in_cpu 1 if data is in CPU memory, 0 if in GPU memory
     */
    template <typename T>
    void print_array(const char *info, T *array, uint32_t num, uint32_t data_in_cpu = 0);

    /**
     * Compare array contents with test vector
     *
     * @param[in] info Descriptive label for comparison
     * @param[in] tv_buf Test vector buffer
     * @param[in] cumac_buf cuMAC output buffer
     * @param[in] num Number of elements to compare
     * @param[in] data_in_cpu 1 if data is in CPU memory, 0 if in GPU memory
     *
     * @return 0 on success, negative on mismatch
     */
    template <typename T>
    int compare_array(const char *info, T *tv_buf, T *cumac_buf, uint32_t num, uint32_t data_in_cpu = 0);

    int (*callback_fun)(cumac_task *task, void *args){}; //!< Task completion callback function pointer
    void *callback_args{}; //!< Arguments to pass to callback function

    cumac_cp_handler* cp_handler{}; //!< Pointer to parent cuMAC CP handler

    sfn_slot_t ss{}; //!< System Frame Number and slot for this task

    //! cuMAC task bitmask: b0 - multiCellUeSelection; b1 - multiCellScheduler; b2 - multiCellLayerSel; b3 - mcsSelectionLUT
    uint32_t taskBitMask{};

    //! Module-level bitmask set from CONFIG.request: controls which modules are initialized (bits 0-3 = 4T4R, bit 4 = PFM SORT, bit 5 = MU UE GRP)
    uint32_t module_bitmask{CUMAC_CP_TASK_MASK_DEFAULT};

    uint32_t run_in_cpu{}; //!< Flag: 1 to run in CPU, 0 to run in GPU
    uint32_t slot_concurrent_enable{}; //!< Flag: Enable concurrent processing of multiple slots

    //! When slot_concurrent_enable==0: assigned by order_sem at enqueue; setup waits, callback posts.
    sem_t* order_wait_sem{};
    sem_t* order_post_sem{};

    uint32_t task_id{}; //!< Unique task instance identifier
    uint32_t cell_num{}; //!< Number of cells in this task
    uint32_t debug_option{}; //!< Debug option flags (see DBG_OPT_* constants)

    uint64_t ts_start{}; //!< Timestamp: First cuMAC message sending time
    uint64_t ts_last_send{}; //!< Timestamp: Last cuMAC message sending time
    uint64_t ts_last_recv{}; //!< Timestamp: Last cuMAC message receiving time
    uint64_t ts_enqueue{}; //!< Timestamp: Task enqueued to ring
    uint64_t ts_dequeue{}; //!< Timestamp: Task dequeued from ring
    uint64_t ts_setup{}; //!< Timestamp: Setup phase start
    uint64_t ts_copy{}; //!< Timestamp: Data copy start
    uint64_t ts_run{}; //!< Timestamp: Run phase start
    uint64_t ts_callback{}; //!< Timestamp: Callback phase start
    uint64_t ts_resp{}; //!< Timestamp: Response sent
    uint64_t ts_end{}; //!< Timestamp: Task completion

    uint64_t ts_debug{}; //!< Timestamp: Debug operation

    uint64_t output_copy_time{}; //!< Duration: Output data copy time (nanoseconds)
    uint64_t run_copy_time{}; //!< Duration: Run phase copy time (nanoseconds)

    cudaEvent_t ev_start{}; //!< CUDA event: Task start
    cudaEvent_t ev_copy1{}; //!< CUDA event: First copy operation
    cudaEvent_t ev_copy2{}; //!< CUDA event: Second copy operation
    cudaEvent_t ev_setup1{}; //!< CUDA event: Setup phase checkpoint 1
    cudaEvent_t ev_setup_end{}; //!< CUDA event: Setup phase end
    cudaEvent_t ev_setup_4{}; //!< CUDA event: Setup phase checkpoint 4
    cudaEvent_t ev_setup{}; //!< CUDA event: Setup phase general

    cudaEvent_t ev_run_start{}; //!< CUDA event: Run phase start
    cudaEvent_t ev_run1{}; //!< CUDA event: Run phase checkpoint 1
    cudaEvent_t ev_run2{}; //!< CUDA event: Run phase checkpoint 2
    cudaEvent_t ev_run3{}; //!< CUDA event: Run phase checkpoint 3
    cudaEvent_t ev_run4{}; //!< CUDA event: Run phase checkpoint 4
    cudaEvent_t ev_run_end{}; //!< CUDA event: Run phase end

    cudaEvent_t ev_callback_start{}; //!< CUDA event: Callback phase start
    cudaEvent_t ev_callback_end{}; //!< CUDA event: Callback phase end

    cudaEvent_t ev_debug{}; //!< CUDA event: Debug operation

    float tm_copy1{}; //!< Elapsed time for first copy (milliseconds)
    float tm_copy2{}; //!< Elapsed time for second copy (milliseconds)

    float tm_setup{}; //!< Elapsed time for setup phase (milliseconds)

    float tm_run{}; //!< Total elapsed time for run phase (milliseconds)
    float tm_run1{}; //!< Elapsed time for run checkpoint 1 (milliseconds)
    float tm_run2{}; //!< Elapsed time for run checkpoint 2 (milliseconds)
    float tm_run3{}; //!< Elapsed time for run checkpoint 3 (milliseconds)
    float tm_run4{}; //!< Elapsed time for run checkpoint 4 (milliseconds)

    float tm_callback{}; //!< Elapsed time for callback phase (milliseconds)

    float tm_total{}; //!< Total elapsed time for entire task (milliseconds)

    float tm_debug{}; //!< Elapsed time for debug operations (milliseconds)

    uint8_t dlIndicator{}; //!< Downlink indicator: 1 for DL, 0 for UL

    uint8_t CQI{}; //!< CQI indicator: 1 for UE reported CQI-based MCS selection, 0 otherwise

    uint8_t RI{}; //!< RI indicator: 1 for UE reported RI-based layer selection, 0 otherwise

    cumac::cumacSimParam simParam{}; //!< cuMAC simulation parameters

    uint8_t lightWeight{}; //!< Light weight processing mode flag

    uint8_t halfPrecision{}; //!< Half precision (FP16) computation flag

    uint8_t baselineUlMcsInd{}; //!< Baseline single-cell MCS selection for UL scheduler

    uint8_t columnMajor{}; //!< Channel matrix layout: 1 for column-major, 0 for row-major

    uint8_t enableHarq{}; //!< HARQ enable flag

    cumac_buf_num_t data_num{}; //!< cuMAC buffer element counts (NOTE: elements, not bytes)

    cumac::multiCellUeSelection *mcUeSelGpu{}; //!< GPU multi-cell UE selection module
    cumac::multiCellScheduler *mcSchGpu{}; //!< GPU multi-cell scheduler module
    cumac::multiCellLayerSel *mcLayerSelGpu{}; //!< GPU multi-cell layer selection module
    cumac::mcsSelectionLUT *mcMcsSelGpu{}; //!< GPU MCS selection LUT module

    cumac::multiCellUeSelectionCpu *mcUeSelCpu{}; //!< CPU multi-cell UE selection module
    cumac::multiCellSchedulerCpu *mcSchCpu{}; //!< CPU multi-cell scheduler module
    cumac::multiCellLayerSelCpu *mcLayerSelCpu{}; //!< CPU multi-cell layer selection module
    cumac::mcsSelectionLUTCpu *mcMcsSelCpu{}; //!< CPU MCS selection LUT module

    cumac::multiCellSinrCal *mcSinrGpu{}; //!< GPU multi-cell SINR calculation (NOT supported yet)
    cumac::roundRobinUeSelCpu *rrUeSelCpu{}; //!< CPU round-robin UE selection (NOT supported yet)
    cumac::roundRobinSchedulerCpu *rrSchCpu{}; //!< CPU round-robin scheduler (NOT supported yet)

    cumac::cumacCellGrpUeStatus ueStatus{}; //!< Cell group UE status parameters
    cumac::cumacSchdSol schdSol{}; //!< Scheduling solution for cell group
    cumac::cumacCellGrpPrms grpPrms{}; //!< Cell group parameters

    //! PFM sorting
    cumac::pfmSort* pfmSortGpu{}; //!< GPU PFM sorting module
    cumac::pfmSortTask pfmSortTask{}; //!< PFM sorting task structure
    cumac_pfm_cell_info_t *pfmCellInfo{}; //!< PFM sorting input buffer
    cumac_pfm_output_cell_info_t *output_pfmSortSol{}; //!< PFM sorting output buffer (host-pinned memory)

    //! MU-MIMO UE grouping
    cumac::muMimoUserPairing* muMimoUserPairingGpu{}; //!< GPU MU-MIMO UE grouping module
    cumac_muUeGrp_req_info_t *muUeGrpInfo{}; //!< MU-MIMO UE grouping input buffer (GPU memory)
    cumac_muUeGrp_resp_info_t *muUeGrpSol{}; //!< MU-MIMO UE grouping output buffer (GPU memory)
    cumac_muUeGrp_resp_info_t *output_muUeGrpSol{}; //!< MU-MIMO UE grouping output buffer (host-pinned memory)
    uint16_t max_num_srs_info{}; //!< Maximum number of SRS UEs per slot per cell

    cumac_task_type_t task_type{}; //!< Task type identifier
    cudaStream_t strm{}; //!< CUDA stream for async operations

    std::vector<struct nv::phy_mac_msg_desc> tti_reqs{}; //!< TTI request message descriptors for all cells

    cell_desc_t* cpu_cell_descs{}; //!< CPU-side cell descriptors array
    cell_desc_t* gpu_cell_descs{}; //!< GPU-side cell descriptors array

    cumac_task_info_t *gpu_task_info{}; //!< GPU-side task information structure

    uint8_t *cells_buf{}; //!< Per-cell data buffers in GPU memory
    uint8_t *group_buf{}; //!< Group-level contiguous data buffer in GPU memory

    uint32_t group_buf_offset{}; //!< Current offset into group buffer (bytes)
    uint32_t group_buf_enabled{}; //!< Flag: 1 if using contiguous group buffer, 0 for individual buffers

    cumac_cp_tv_t *tv{}; //!< Test vector data pointer for validation

#if 1
    float *input_avgRatesActUe{}; //!< Input: Average data rates for active UEs
    float *input_avgRates{}; //!< Input: Average data rates for all UEs
    int8_t *input_tbErrLastActUe{}; //!< Input: Transport block error flags for active UEs
    int8_t *input_tbErrLast{}; //!< Input: Transport block error flags for all UEs
    cuComplex *input_estH_fr{}; //!< Input: Estimated channel matrix in frequency domain

    uint16_t *output_setSchdUePerCellTTI{}; //!< Output: Set of scheduled UEs per cell per TTI
    int16_t *output_allocSol{}; //!< Output: Resource allocation solution
    uint8_t *output_layerSelSol{}; //!< Output: Layer selection solution
    int16_t *output_mcsSelSol{}; //!< Output: MCS selection solution

    uint8_t *debug_buffer{}; //!< Host-pinned memory buffer for debug logging
#else
    float input_avgRatesActUe[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Input: Average data rates for active UEs (static array)
    float input_avgRates[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Input: Average data rates for all UEs (static array)
    int8_t input_tbErrLastActUe[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Input: Transport block error flags for active UEs (static array)
    int8_t input_tbErrLast[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Input: Transport block error flags for all UEs (static array)
    cuComplex input_estH_fr[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Input: Estimated channel matrix in frequency domain (static array)

    uint16_t output_setSchdUePerCellTTI[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Output: Set of scheduled UEs per cell per TTI (static array)
    int16_t output_allocSol[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Output: Resource allocation solution (static array)
    uint8_t output_layerSelSol[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Output: Layer selection solution (static array)
    int16_t output_mcsSelSol[CUMAC_TASK_BUF_MAX_SIZE]{}; //!< Output: MCS selection solution (static array)

    uint8_t debug_buffer[CUMAC_TASK_DEBUG_BUF_MAX_SIZE]{}; //!< Debug buffer (static array)
    uint8_t debug_local[CUMAC_TASK_DEBUG_BUF_MAX_SIZE]{}; //!< Local debug buffer on stack
#endif
};

#endif // CUMAC_TASK_HPP_
