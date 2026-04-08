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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 9) // "DRV.TASK"

#include "task.hpp"

////////////////////////////////////////////////////////////////////////////////
//Implementation of PhyWaiter class
///////////////////////////////////////////////////////////////////////////////////////////
PhyWaiter::PhyWaiter(PhyChannel* channel):channel(channel) {
    if(channel == nullptr) {
        current_state = WAIT_STATE_COMPLETED;
    } else {
        current_state = WAIT_STATE_NOT_STARTED;
    }
}
bool PhyWaiter::stillWaiting() {
    return (current_state==WAIT_STATE_NOT_STARTED || current_state==WAIT_STATE_STARTED);
};
wait_action PhyWaiter::checkAction() {
    if(stillWaiting()) {
        switch(current_state) {
            case WAIT_STATE_NOT_STARTED:
                if(channel->waitStartRunEventNonBlocking()==1) {
                    current_state = WAIT_STATE_STARTED;
                    return WAIT_ACTION_STARTED;
                }
                break;
            case WAIT_STATE_STARTED:
                if(channel->waitRunCompletionEventNonBlocking()==1) {
                    current_state = WAIT_STATE_COMPLETED;
                    return WAIT_ACTION_COMPLETED;
                }
                break;
            default:
                printf("ERROR :: Unknown Wait State encountered in PhyWaiter\n");
                break;
        }
    }
    return WAIT_ACTION_NONE;
};
wait_state PhyWaiter::getState() {
    return current_state;
}

void PhyWaiter::setState(wait_state state) {
    current_state = state;
}

////////////////////////////////////////////////////////////////////////////////
//Implementation of OrderWaiter class
////////////////////////////////////////////////////////////////////////////////
OrderWaiter::OrderWaiter(OrderEntity* oe):oe(oe) {
    //Please note: OrderWaiter only uses two states (started and complete)
    // as we do not have a wait for order start
    if(oe == nullptr) {
        current_state = WAIT_STATE_COMPLETED;
    } else {
        current_state = WAIT_STATE_STARTED;
    }
}
bool OrderWaiter::stillWaiting() {
    if(current_state == WAIT_STATE_COMPLETED) {
        return false;
    } else {
        return true;
    }
}
wait_action OrderWaiter::checkAction(bool isSrs) {
    if(stillWaiting()) {
        if(oe->checkOrderCPU(isSrs) == 1) {
            current_state = WAIT_STATE_COMPLETED;
            return WAIT_ACTION_COMPLETED;
        }
    }
    return WAIT_ACTION_NONE;
}
wait_state OrderWaiter::getState() {
    return current_state;
}
void OrderWaiter::setState(wait_state state) {
    current_state=state;
}
////////////////////////////////////////////////////////////////////////////////
//// Task
////////////////////////////////////////////////////////////////////////////////

Task::Task(
    phydriver_handle _pdh,
    uint64_t         _id) :
    pdh(_pdh),
    id(_id)
{
    mf.init(_pdh, std::string("Task"), sizeof(Task));

    ts_create = Time::zeroNs();
    ts_init = Time::zeroNs();
    ts_exec = Time::zeroNs();
    type = TASK_TYPE_NONE;
    priority = 0;
    work_f = nullptr;
    work_f_arg = nullptr;
    first_cell = 0;
    num_cells = 0;
    num_tasks = 0;
    desired_wid = 0;
    name.reserve(TASK_NAME_RESERVE_LENGTH);
}

Task::~Task()
{
}

int Task::init(t_ns               _ts_exec,
               const char*        _name,
               task_work_function _work_f,
               void*              _work_f_arg,
               int                _first_cell,
               int                _num_cells,
               int                _num_tasks,
               worker_id          _desired_wid)
{
    ts_exec    = _ts_exec;
    ts_create  = Time::nowNs();
    name       = _name; //.assign(_name);
    if(name.length() >= TASK_NAME_RESERVE_LENGTH)
    {
        NVLOGW_FMT(TAG, "Warning: Task name {} exceeds length that was used to reserve ({}), you may see high latency due to reallocation!", name.c_str(), TASK_NAME_RESERVE_LENGTH);
    }
    work_f     = _work_f;
    work_f_arg = _work_f_arg;
    first_cell = _first_cell;
    num_cells  = _num_cells;
    num_tasks  = _num_tasks;
    desired_wid = _desired_wid;

    return 0;
}

phydriver_handle Task::getPhyDriverHandler(void) const
{
    return pdh;
}

uint64_t Task::getId() const
{
    return id;
}

t_ns& Task::getTsCreate()
{
    return ts_create;
}

t_ns Task::getTsExec() const noexcept
{
    return ts_exec;
}

std::string_view Task::getName() const
{
    return name;
}

int Task::getFirstCell() const
{
    return first_cell;
}

int Task::getNumCells() const
{
    return num_cells;
}

worker_id Task::getDesiredWID() const {
    return desired_wid;
}

void* Task::getTaskArgs()
{
    return work_f_arg;
}
int Task::run(Worker* worker)
{
    return work_f(worker, work_f_arg, first_cell, num_cells, num_tasks);
}

////////////////////////////////////////////////////////////////////////////////
//// Task List
////////////////////////////////////////////////////////////////////////////////

TaskList::TaskList(
    phydriver_handle _pdh,
    uint32_t         _id,
    uint32_t         _size) :
    pdh(_pdh),
    id(_id)
{
    mf.init(_pdh, std::string("TaskList"), sizeof(TaskList));
}

TaskList::~TaskList() 
{
    clear_task_all();
}

phydriver_handle TaskList::getPhyDriverHandler(void) const
{
    return pdh;
}

uint32_t TaskList::getId()
{
    return id;
}

int TaskList::lock()
{
    mutex_task_list.lock();
    return 0;
}

int TaskList::unlock()
{
    mutex_task_list.unlock();
    return 0;
}

int TaskList::findWorkerIndex(worker_id wid) const
{
    if (wid == INVALID_WORKER_ID) {
        return -1;  // Special case for generic tasks
    }

    // Linear search through the index_to_wid array
    for (int i = 0; i < next_worker_index; i++) {
        if (index_to_wid[i] == wid) {
            return i;
        }
    }
    return -1;
}

int TaskList::createWorkerIndex(worker_id wid)
{
    if (wid == INVALID_WORKER_ID) {
        return -1;  // Special case for generic tasks
    }

    if (next_worker_index < priority_lists.size()) {
        int index = next_worker_index++;
        index_to_wid[index] = wid;
        return index;
    }

    NVLOGW_FMT(TAG, "Warning: Maximum number of worker IDs ({}) reached, treating task as generic", priority_lists.size());
    return -1;
}

int TaskList::push(Task* t)
{
    worker_id wid = t->getDesiredWID();
    
    // First try to find existing index
    int worker_index = findWorkerIndex(wid);
    
    // If not found and not generic task (wid != 0), try to create new index
    if (worker_index == -1 && wid != 0) {
        worker_index = createWorkerIndex(wid);
    }
    
    if (worker_index >= 0) {
        auto& queue = priority_lists[worker_index];
        if (queue.size() < queue.get_capacity()) {
            queue.push(std::move(t));
            total_tasks++;
            return 0;
        }
        return -1;  // Queue is full
    } else {
        if (generic_priority_list.size() < generic_priority_list.get_capacity()) {
            generic_priority_list.push(std::move(t));
            total_tasks++;
            return 0;
        }
        return -1;  // Generic queue is full
    }
}

Task* TaskList::get_task(worker_id requesting_wid, t_ns time_threshold_ns)
{
    // Check the specific queue for this worker ID if it exists
    int worker_index = findWorkerIndex(requesting_wid);
    if (worker_index >= 0 && !priority_lists[worker_index].empty()) {
        auto& queue = priority_lists[worker_index];
        Task* task = queue.top();
        t_ns ts_now = Time::nowNs();
        t_ns ts_trigger = task->getTsExec();
        
        if ((ts_now - ts_trigger) > -time_threshold_ns) {
            queue.pop();
            total_tasks--;
            return task;
        }
    }

    // Then check the generic queue
    if (!generic_priority_list.empty()) {
        Task* task = generic_priority_list.top();
        t_ns ts_now = Time::nowNs();
        t_ns ts_trigger = task->getTsExec();
        
        if ((ts_now - ts_trigger) > -time_threshold_ns) {
            generic_priority_list.pop();
            total_tasks--;
            return task;
        }
    }

    return nullptr;
}

void TaskList::clear_task_all()
{
    // Clean up tasks in generic queue
    while (!generic_priority_list.empty()) {
        Task* task = generic_priority_list.top();
        generic_priority_list.pop();
        delete task;
    }

    // Clean up tasks in worker queues
    for (auto& queue : priority_lists) {
        while (!queue.empty()) {
            Task* task = queue.top();
            queue.pop();
            delete task;
        }
    }

    // Clear the queues and mappings
    priority_lists.clear();
    index_to_wid.clear();
    next_worker_index = 0;
    total_tasks = 0;
}

void TaskList::initListWithReserveSize(size_t per_queue_size, size_t num_queues)
{
    // Clean up any existing tasks first
    clear_task_all();

    // Initialize generic queue
    std::vector<Task*> v;
    v.reserve(per_queue_size);
    task_priority_queue generic_queue(CompareTaskExecTimestamp(), std::move(v));
    generic_priority_list = std::move(generic_queue);

    // Initialize worker queues
    priority_lists.clear();
    priority_lists.reserve(num_queues);
    for (size_t i = 0; i < num_queues; i++) {
        std::vector<Task*> worker_v;
        worker_v.reserve(per_queue_size);
        task_priority_queue worker_queue(CompareTaskExecTimestamp(), std::move(worker_v));
        priority_lists.push_back(std::move(worker_queue));
    }

    // Initialize worker ID mapping array
    index_to_wid.clear();
    index_to_wid.resize(num_queues, INVALID_WORKER_ID);
    
    // Reset state
    next_worker_index = 0;
    total_tasks = 0;
}
