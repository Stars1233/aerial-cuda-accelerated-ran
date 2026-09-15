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

#ifndef TASK_CLASS_H
#define TASK_CLASS_H

#include <queue>
#include <chrono>
#include <span>
#include "locks.hpp"
#include "time.hpp"
#include "memfoot.hpp"
#include "cuphydriver_api.hpp"
#include "constant.hpp"
#include "phychannel.hpp"
#include "worker.hpp"

/**
 * @brief Task type classification.
 */
typedef enum task_type
{
    TASK_TYPE_UL    = 1 << 0, ///< Uplink task
    TASK_TYPE_DL    = 1 << 1, ///< Downlink task
    TASK_TYPE_NONE  = 1 << 2  ///< No specific type
} task_type;

/**
 * @brief Task work function signature.
 * 
 * @param worker Worker executing the task
 * @param param Task-specific parameters (e.g., slot map)
 * @param first_cell First cell index in the range
 * @param num_cells Number of cells to process
 * @param num_tasks Number of tasks
 * @return 0 on success, non-zero on error
 */
typedef int (*task_work_function)(Worker*, void*, int, int,int);

/**
 * @brief Task time function signature.
 * 
 * @return Timestamp in nanoseconds
 */
typedef uint64_t (*task_time_f)(void);

/**
 * @brief UL aggregated BFW (beamforming weights) task function.
 */
int task_work_function_ul_aggr_bfw(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks);
/** @brief UL aggregated PUCCH and PUSCH combined processing task. */
int task_work_function_ul_aggr_1_pucch_pusch(Worker* worker, void* param, int first_cell, int num_cells, int num_ul_tasks);
/** @brief UL aggregated PUSCH processing task. */
int task_work_function_ul_aggr_1_pusch(Worker* worker, void*, int, int,int);
/** @brief UL aggregated PUCCH processing task. */
int task_work_function_ul_aggr_1_pucch(Worker* worker, void*, int, int,int);
/** @brief UL aggregated PRACH processing task. */
int task_work_function_ul_aggr_1_prach(Worker* worker, void*, int, int,int);
/** @brief UL aggregated SRS processing task. */
int task_work_function_ul_aggr_1_srs(Worker* worker, void*, int, int,int);
/** @brief UL aggregated ordering kernel task (stage 1). */
int task_work_function_ul_aggr_1_orderKernel(Worker* worker, void*, int, int,int);
/** @brief UL aggregated C-plane processing task. */
int task_work_function_ul_aggr_1_cplane(Worker* worker, void*, int, int,int);
/** @brief UL aggregated stage 2 processing task. */
int task_work_function_ul_aggr_2(Worker* worker, void*, int, int,int);
/** @brief UL aggregated stage 3 early UCI indication task. */
int task_work_function_ul_aggr_3_early_uci_ind(Worker* worker, void*,int, int,int);
/** @brief UL aggregated stage 3 processing task. */
int task_work_function_ul_aggr_3(Worker* worker, void*, int, int,int);
/** @brief UL aggregated stage 3 SRS processing task. */
int task_work_function_ul_aggr_3_srs(Worker* worker, void*, int, int,int);
/** @brief UL aggregated stage 3 UL BFW processing task. */
int task_work_function_ul_aggr_3_ulbfw(Worker* worker, void*, int, int,int);
/** @brief DL aggregated BFW (beamforming weights) task function. */
int task_work_function_dl_aggr_bfw(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks);
/** @brief DL aggregated stage 1 PDSCH processing task. */
int task_work_function_dl_aggr_1_pdsch(Worker* worker, void*, int, int,int);
/** @brief DL aggregated control channel processing task (PBCH/PDCCH). */
int task_work_function_dl_aggr_control(Worker* worker, void*, int, int,int);
/** @brief DL aggregated stage 1 compression task. */
int task_work_function_dl_aggr_1_compression(Worker* worker, void*, int, int,int);
/** @brief C-plane processing task. */
int task_work_function_cplane(Worker* worker, void* param, int task_num, int num_cells, int first);
/** @brief DL fronthaul callback task. */
int task_work_function_dl_fh_cb(Worker* worker, void* param, int first_cell, int num_cells, int num_dl_tasks);
/** @brief DL aggregated stage 2 processing task. */
int task_work_function_dl_aggr_2(Worker* worker, void*, int, int,int);
/** @brief DL aggregated stage 2 GPU communication task. */
int task_work_function_dl_aggr_2_gpu_comm(Worker* worker, void*, int, int,int);
/** @brief DL aggregated stage 2 GPU communication transmit task. */
int task_work_function_dl_aggr_2_gpu_comm_tx(Worker* worker, void*, int, int,int);
/** @brief DL aggregated stage 2 GPU communication prepare task. */
int task_work_function_dl_aggr_2_gpu_comm_prepare(Worker* worker, void*,int,int,int);
/** @brief DL aggregated stage 2 CPU doorbell task. */
int task_work_function_dl_aggr_2_ring_cpu_doorbell(Worker* worker, void*, int, int,int);
/** @brief DL aggregated stage 3 buffer cleanup task. */
int task_work_function_dl_aggr_3_buf_cleanup(Worker* worker, void*, int, int,int);
/** @brief Debug task function. */
int task_work_function_debug(Worker* worker, void*, int, int, int);
/** @brief DL validation task function. */
int task_work_function_dl_validation(Worker* worker, void*, int, int, int);

/**
 * @brief Wait state enumeration for tracking asynchronous operation progress.
 */
typedef enum wait_state
{
    WAIT_STATE_NOT_STARTED = 0, ///< Operation not yet started
    WAIT_STATE_STARTED = 1,     ///< Operation in progress
    WAIT_STATE_COMPLETED = 2    ///< Operation completed
} wait_state;

/**
 * @brief Wait action enumeration for state transitions.
 */
typedef enum wait_action
{
    WAIT_ACTION_NONE = 0,      ///< No action required
    WAIT_ACTION_STARTED = 1,   ///< Operation has started
    WAIT_ACTION_COMPLETED = 2  ///< Operation has completed
} wait_action;

/**
 * @brief Waiter class for monitoring PhyChannel operation completion.
 */
class PhyWaiter {
public:
    /**
     * @brief Constructs a PhyWaiter for a specific channel.
     * 
     * @param channel Pointer to the PhyChannel to monitor
     */
    PhyWaiter(PhyChannel* channel);

    /**
     * @brief Checks if the channel operation is still in progress.
     * 
     * @return true if still waiting, false if completed
     */
    bool stillWaiting();
    
    /**
     * @brief Checks and returns the current action based on state transitions.
     * 
     * @return Current wait_action (NONE, STARTED, or COMPLETED)
     */
    wait_action checkAction();
    
    /**
     * @brief Gets the current wait state.
     * 
     * @return Current wait_state
     */
    wait_state getState();
    
    /**
     * @brief Sets the wait state.
     * 
     * @param state New wait_state to set
     */
    void setState(wait_state state);
private:
    PhyChannel* channel;         ///< Pointer to the PhyChannel being monitored
    wait_state current_state;    ///< Current state of the operation
};

/**
 * @brief Waiter class for monitoring order kernel launch and completion.
 */
class OrderWaiter {
public:
    /**
     * @brief Constructs an OrderWaiter for a specific order entity.
     * 
     * @param oe Pointer to the OrderEntity to monitor
     */
    OrderWaiter(OrderEntity* oe);

    /**
     * @brief Checks if the order operation is still in progress.
     * 
     * @return true if still waiting, false if completed
     */
    bool stillWaiting();
    
    /**
     * @brief Checks and returns the current action based on state transitions.
     * 
     * @param isSrs Flag indicating if this is an SRS order
     * @return Current wait_action (NONE, STARTED, or COMPLETED)
     */
    wait_action checkAction(bool isSrs);
    
    /**
     * @brief Gets the current wait state.
     * 
     * @return Current wait_state
     */
    wait_state getState();
    
    /**
     * @brief Sets the wait state.
     * 
     * @param state New wait_state to set
     */
    void       setState(wait_state);
private:
    OrderEntity* oe;              ///< Pointer to the OrderEntity being monitored
    wait_state current_state;     ///< Current state of the operation
};

/**
 * @brief Task class representing a unit of work to be executed by a Worker.
 */
class Task {
public:
    /**
     * @brief Constructs a Task.
     * 
     * @param _pdh cuPHYdriver handle
     * @param _id Unique task identifier
     */
    Task(phydriver_handle _pdh, uint64_t _id);
    
    /**
     * @brief Destructor.
     */
    ~Task();

    /**
     * @brief Initializes the task with execution parameters.
     * 
     * @param _ts_exec Execution timestamp
     * @param _name Task name
     * @param _work_f Task work function pointer
     * @param _work_f_arg Arguments for the work function
     * @param _first_cell First cell index
     * @param _num_cells Number of cells
     * @param _num_tasks Number of tasks
     * @param _desired_wid Desired worker ID
     * @return 0 on success
     */
    int init(t_ns _ts_exec, const char* _name, task_work_function _work_f, void* _work_f_arg, int _first_cell, int _num_cells, int _num_tasks, worker_id _desired_wid);
    
    /**
     * @brief Gets the cuPHYdriver handle.
     * 
     * @return PHY driver handle
     */
    phydriver_handle getPhyDriverHandler(void) const;
    
    /**
     * @brief Gets the task ID.
     * 
     * @return Task ID
     */
    uint64_t         getId() const;
    
    /**
     * @brief Gets the task creation timestamp (mutable reference).
     * 
     * @return Reference to creation timestamp
     */
    t_ns&            getTsCreate();
    
    /**
     * @brief Gets the task execution timestamp.
     * 
     * @return Execution timestamp
     */
    [[nodiscard]] t_ns            getTsExec() const noexcept;
    
    /**
     * @brief Gets the task name.
     * 
     * @return Task name
     */
    std::string_view      getName() const;
    
    /**
     * @brief Gets the first cell index.
     * 
     * @return First cell index
     */
    int              getFirstCell() const;
    
    /**
     * @brief Gets the number of cells.
     * 
     * @return Number of cells
     */
    int              getNumCells() const;
    
    /**
     * @brief Gets the desired worker ID.
     * 
     * @return Desired worker ID
     */
    worker_id        getDesiredWID() const;
    
    /**
     * @brief Gets the task arguments pointer.
     * 
     * @return Pointer to task arguments
     */
    void*            getTaskArgs();
    
    /**
     * @brief Executes the task on a worker.
     * 
     * @param worker Pointer to the worker executing the task
     * @return Task work function return value (0 on success)
     */
    int              run(Worker* worker);
    
    MemFoot          mf; ///< Memory footprint tracker
private:
    phydriver_handle   pdh;          ///< cuPHYdriver handle
    uint64_t           id;           ///< Task unique identifier
    std::string        name;         ///< Task name
    t_ns               ts_create;    ///< Task creation timestamp
    t_ns               ts_init;      ///< Task initialization timestamp
    t_ns               ts_exec;      ///< Task execution timestamp
    task_type          type;         ///< Task type (UL/DL/NONE)
    uint8_t            priority;     ///< Task priority
    task_work_function work_f;       ///< Task work function pointer
    void*              work_f_arg;   ///< Task work function arguments
    int                first_cell;   ///< First cell index in range
    int                num_cells;    ///< Number of cells to process
    int                num_tasks;    ///< Number of tasks
    worker_id          desired_wid;  ///< Desired worker ID
};

/**
 * @brief Comparator for ordering tasks by execution timestamp.
 */
struct CompareTaskExecTimestamp final
{
    /**
     * @brief Compares two tasks by execution timestamp.
     * 
     * @param p1 First task
     * @param p2 Second task
     * @return true if p1 should be ordered before p2 (higher timestamp = lower priority in min-heap)
     */
    [[nodiscard]] bool operator()(Task const* p1, Task const* p2) const noexcept
    {
        // return "true" if "p1" is ordered before "p2", for example:
        return p1->getTsExec() > p2->getTsExec();
    }
};

/**
 * @brief Priority queue for tasks, ordered by execution timestamp.
 * 
 * Extends std::priority_queue to add inspection capabilities for task readiness.
 */
class task_priority_queue : public std::priority_queue<Task*, std::vector<Task*>, CompareTaskExecTimestamp> {

public:

    //Inherit all constructors
    using priority_queue::priority_queue;

    /**
     * @brief Checks if any tasks are ready for execution.
     * 
     * Inspects all queued tasks to determine if any match the requesting worker
     * and have passed their execution time threshold.
     * 
     * @param requesting_wid Worker ID requesting tasks
     * @param time_threshold_ns Time threshold in nanoseconds for task readiness
     * @return true if at least one task is ready, false otherwise
     */
    bool anyTasksReady(worker_id requesting_wid, t_ns time_threshold_ns) {

        //Search directly in the internal vector
        for (auto it = this->c.cbegin(); it != this->c.cend(); ++it) {

            //Only inspect tasks that are either non-specific or match our requesting worker id
            worker_id desired_wid = (*it)->getDesiredWID();
            if(desired_wid == 0 || desired_wid == requesting_wid) {

                //Make sure our time trigger has expired
                t_ns ts_now = Time::nowNs();
                t_ns ts_trigger = (*it)->getTsExec();
                if((ts_now - ts_trigger) > -time_threshold_ns) {
                    //This task is ready
                    return true;

                }

            }
        }
        return false;
    }

    /**
     * @brief Gets the capacity of the underlying container.
     * 
     * @return Capacity of the internal vector
     */
    [[nodiscard]] size_t get_capacity() const {
        return this->c.capacity();
    }
};

/**
 * @brief Thread-safe task list with per-worker priority queues.
 * 
 * Manages task scheduling with separate priority queues for each worker
 * and a generic queue for unassigned tasks.
 */
class TaskList {
public:
    /**
     * @brief Constructs a TaskList.
     * 
     * @param _pdh cuPHYdriver handle
     * @param _id TaskList unique identifier
     * @param _size Initial size
     */
    TaskList(phydriver_handle _pdh, uint32_t _id, uint32_t _size);
    
    /**
     * @brief Destructor.
     */
    ~TaskList();

    /**
     * @brief Gets the cuPHYdriver handle.
     * 
     * @return cuPHYdriver handle
     */
    phydriver_handle getPhyDriverHandler(void) const;
    
    /**
     * @brief Gets the TaskList ID.
     * 
     * @return TaskList ID
     */
    uint32_t         getId();
    
    /**
     * @brief Locks the task list for thread-safe access.
     * 
     * @return 0 on success
     */
    int              lock();
    
    /**
     * @brief Unlocks the task list.
     * 
     * @return 0 on success
     */
    int              unlock();
    
    /**
     * @brief Pushes a task into the appropriate priority queue.
     * 
     * @param t Pointer to the task to add
     * @return 0 on success
     */
    int              push(Task* t);

    /**
     * @brief Pushes multiple pre-allocated tasks under a single lock acquisition.
     *
     * @param[in] tasks Non-owning view of Task pointers to push.
     * @return Number of tasks successfully pushed; must be checked for partial failures.
     */
    [[nodiscard]] int push_bulk(std::span<Task* const> tasks);

    /**
     * @brief Gets the next ready task for a worker.
     * 
     * @param requesting_wid Worker ID requesting a task
     * @param time_threshold_ns Time threshold for task readiness
     * @return Pointer to a ready task, or nullptr if none available
     */
    Task*            get_task(worker_id requesting_wid, t_ns time_threshold_ns);
    
    /**
     * @brief Clears all tasks from all queues.
     */
    void             clear_task_all();
    
    /**
     * @brief Pre-allocates memory for priority queues.
     * 
     * @param per_queue_size Capacity per queue
     * @param num_queues Number of worker-specific queues
     */
    void             initListWithReserveSize(size_t per_queue_size, size_t num_queues);
    
    MemFoot          mf; ///< Memory footprint tracker

private:
    static constexpr worker_id INVALID_WORKER_ID = 0; ///< Sentinel value for unused worker ID slots

    /**
     * Get the index for a worker ID if it exists
     * @param wid The worker ID to look up
     * @return Index into priority_lists array, or -1 if not found
     */
    [[nodiscard]] int findWorkerIndex(worker_id wid) const;

    /**
     * Create a new index mapping for a worker ID
     * @param wid The worker ID to map
     * @return Index into priority_lists array, or -1 if no space available
     */
    int createWorkerIndex(worker_id wid);

    phydriver_handle                                pdh;                       ///< cuPHYdriver handle
    uint32_t                                        id;                        ///< TaskList unique identifier
    Mutex                                           mutex_task_list;           ///< Mutex for thread-safe access
    std::vector<task_priority_queue>                priority_lists;            ///< Priority queues for each worker ID
    task_priority_queue                             generic_priority_list;     ///< Priority queue for tasks with no specific worker ID
    size_t                                          total_tasks{0};            ///< Total number of tasks across all queues
    std::vector<worker_id>                          index_to_wid;              ///< Maps indices to worker IDs (INVALID_WORKER_ID means unused)
    int                                             next_worker_index{0};      ///< Next available index in priority_lists
};

#endif
