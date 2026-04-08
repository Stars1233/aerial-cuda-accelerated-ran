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

#ifndef WORKER_CLASS_H
#define WORKER_CLASS_H

//Required by pthread_setname_np
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "nvlog.hpp"
#include <iostream>
#include <atomic>
#include <pthread.h>
#include <string>
#include <sched.h>
#include <cinttypes>
#include "cuphydriver_api.hpp"
#include "mps.hpp"
#include "pmu_reader.hpp"

#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#if 0
/**
 * Macro to define a per worker variable "var" of type "type", don't
 * use keywords like "static" or "volatile" in type, just prefix the
 * whole macro.
 */
#define DEFINE_PER_WORKER(type, name) \
    __thread __typeof__(type) per_worker_##name

/**
 * Macro to declare an extern per worker variable "var" of type "type"
 */
#define DECLARE_PER_WORKER(type, name) \
    extern __thread __typeof__(type) per_worker_##name

/**
 * Read/write the per-worker variable value
 */
#define PER_WORKER(name) (per_worker_##name)
#endif

// class e_worker_init: public exception
// {
//     virtual const char* what() const throw()
//     {
//         return "Worker init error";
//     }
// } e_worker_init;

// DECLARE_PER_WORKER(unsigned, _windex);
// DECLARE_PER_WORKER(uintptr_t, _wptr);

/**
 * @brief Arguments for worker thread initialization.
 */
struct worker_args
{
    worker_routine      start_routine; ///< Worker thread entry point function
    void*               arg;           ///< Arguments passed to start routine
    phydriverwrk_handle whandler;      ///< Worker handle
};

class Task; ///< Forward declaration to avoid include issues

/**
 * @brief Worker thread initialization routine.
 * 
 * @param arg Pointer to worker_args structure
 * @return Thread exit value
 */
void*     worker_init(void* arg);

/**
 * @brief Gets the task ID for a worker based on task and worker type.
 * 
 * @param t Pointer to the task
 * @param w_type Worker default type
 * @return Task ID
 */
int get_worker_task_id(Task* t,worker_default_type w_type);

/**
 * @brief Creates a unique worker ID.
 * 
 * @return Unique worker identifier
 */
worker_id create_worker_id();

/**
 * @brief Default worker routine that consumes tasks from a task list.
 * 
 * @param wh Worker handle
 * @param arg Worker arguments
 * @return 0 on success
 */
int worker_default(phydriverwrk_handle wh, void* arg);

/**
 * @brief Worker thread class for executing tasks.
 * 
 * Manages a pthread with CPU affinity, scheduling policy, and PMU performance counters.
 */
class Worker {
public:
    /**
     * @brief Constructs a Worker.
     * 
     * @param _pdh cuPHYdriver handle
     * @param _wunique_id Unique worker identifier
     * @param _type Worker default type (task consumer type)
     * @param name Worker thread name
     * @param _cpucore CPU core for affinity binding
     * @param _sched_priority Scheduling priority
     * @param _pmu_metrics PMU metrics configuration flags
     * @param _start_routine Worker thread entry point function
     * @param _arg Arguments passed to start routine
     */
    Worker(
        phydriver_handle         _pdh,
        uint64_t                 _wunique_id,
        enum worker_default_type _type,
        const char*              name,
        uint8_t                  _cpucore,
        uint32_t                 _sched_priority,
        uint8_t                  _pmu_metrics,
        worker_routine           _start_routine,
        void*                    _arg);
    
    /**
     * @brief Destructor.
     */
    ~Worker();

    /**
     * @brief Gets the cuPHYdriver handle.
     * 
     * @return cuPHYdriver handle
     */
    phydriver_handle         getPhyDriverHandler(void) const;
    
    /**
     * @brief Starts the worker thread.
     * 
     * @return 0 on success, non-zero on error
     */
    int                      run();
    
    /**
     * @brief Gets the worker thread name.
     * 
     * @return Worker thread name
     */
    const char*              getName();
    
    /**
     * @brief Gets the worker unique ID.
     * 
     * @return Worker unique identifier
     */
    uint64_t                 getId();
    
    /**
     * @brief Gets the scheduling priority.
     * 
     * @return Scheduling priority value
     */
    int                      getPriority();
    
    /**
     * @brief Gets the scheduling policy.
     * 
     * @return Scheduling policy (e.g., SCHED_FIFO, SCHED_RR)
     */
    int                      getPolicy();
    
    /**
     * @brief Gets the CPU core affinity.
     * 
     * @return CPU core number
     */
    uint8_t                  getCPUAffinity();
    
    /**
     * @brief Sets the worker exit flag.
     * 
     * @param val Exit flag value (true = exit requested)
     */
    void                     setExitValue(bool val);
    
    /**
     * @brief Gets the worker exit flag status.
     * 
     * @return true if exit requested, false otherwise
     */
    bool                     getExitValue();
    
    /**
     * @brief Waits for the worker thread to exit.
     * 
     * @return 0 on success
     */
    int                      waitExit();
    
    /**
     * @brief Gets the worker default type.
     * 
     * @return Worker default type
     */
    enum worker_default_type getType();
    
    MemFoot                  mf; ///< Memory footprint tracker

    /**
     * @brief Creates PMU counters for this worker's thread.
     * Must be called from the worker thread (not the main thread).
     */
    void initPMU();

    /**
     * @brief Destroys PMU performance counters.
     */
    void destroyPMU();

    /**
     * @brief Gets the PMU delta summarizer for direct access.
     * @return Pointer to PMUDeltaSummarizer (may be nullptr before initPMU)
     */
    PMUDeltaSummarizer* getPMU() { return pmuds; }

private:
    phydriver_handle         pdh;            ///< cuPHYdriver handle
    uint64_t                 wunique_id;     ///< Worker unique identifier
    enum worker_default_type type;           ///< Worker default type (task consumer type)
    pthread_t                wid;            ///< pthread identifier
    pthread_attr_t           wattr;          ///< pthread attributes
    std::string              wname;          ///< Worker thread name
    cpu_set_t                wcpuset;        ///< CPU affinity set
    uint8_t                  cpucore;        ///< CPU core number for affinity binding
    uint32_t                 sched_priority; ///< Scheduling priority
    PMU_TYPE                 pmu_type;       ///< PMU metrics type configuration
    int                      schedpol;       ///< Scheduling policy
    struct worker_args       wargs;          ///< Worker thread arguments
    std::atomic<bool>        exit;           ///< Exit flag (true = exit requested)
    worker_routine           start_routine;  ///< Worker thread entry point function
    void*                    arg;            ///< Arguments passed to start routine
    bool                     running;        ///< Running status flag

    PMUDeltaSummarizer*      pmuds;          ///< PMU delta summarizer for performance metrics
};

#endif