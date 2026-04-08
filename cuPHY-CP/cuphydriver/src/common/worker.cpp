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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 10) // "DRV.WORKER"

#include "cuphydriver_api.hpp"
#include "constant.hpp"
#include "context.hpp"
#include "time.hpp"
#include "task.hpp"
#include "slot_map_ul.hpp"
#include "slot_map_dl.hpp"
#include "worker.hpp"
#include "phychannel.hpp"
#include "exceptions.hpp"
#include "cuphyoam.hpp"
#include "memtrace.h"
#include <rte_lcore.h>

using namespace std;

// DEFINE_PER_WORKER(unsigned, _windex) = 0;
// DEFINE_PER_WORKER(uintptr_t, _wptr) = 0;

Worker::Worker(
    phydriver_handle         _pdh,
    uint64_t                 _wunique_id,
    enum worker_default_type _type,
    const char*              name,
    uint8_t                  _cpucore,
    uint32_t                 _sched_priority,
    uint8_t                  _pmu_metrics,
    worker_routine           _start_routine,
    void*                    _arg) :
    pdh(_pdh),
    wunique_id(_wunique_id),
    type(_type),
    cpucore(_cpucore),
    sched_priority(_sched_priority),
    start_routine(_start_routine),
    arg(_arg)
{
    wargs.start_routine = start_routine;
    wargs.arg           = arg;
    wargs.whandler      = static_cast<phydriverwrk_handle>(this);
    wid = 0;
    pmuds = nullptr;

    CPU_ZERO(&wcpuset);
    CPU_SET(cpucore, &wcpuset);

    wname.clear();
    if(name)
        wname.assign(name);
    running = false;
    setExitValue(false);
    //pthread_barrier_wait(&params->configured);

    if(type == WORKER_UL)
        mf.init(_pdh, std::string("WorkerUL"), sizeof(Worker));
    else if(type == WORKER_DL)
        mf.init(_pdh, std::string("WorkerDL"), sizeof(Worker));
    else
        mf.init(_pdh, std::string("WorkerGeneric"), sizeof(Worker));

    pmu_type = static_cast<PMU_TYPE>(_pmu_metrics);
#if !defined(__arm__) && !defined(__aarch64__)
    if(pmu_type != PMU_TYPE_DISABLED && pmu_type != PMU_TYPE_GENERAL) {
        NVLOGW_FMT(TAG, "Unable to set pmu_type={} for non Grace system.  Disabling pmu_metrics.", +pmu_type);
        pmu_type = PMU_TYPE_DISABLED;
    }
#endif

    //NVLOGD_FMT(TAG, "Worker {} Initialized!", wunique_id);
};

Worker::~Worker()
{
    if(running == true)
    {
        setExitValue(true);
        waitExit();
    }

    //Stop performance counters
    destroyPMU();

};

phydriver_handle Worker::getPhyDriverHandler(void) const
{
    return pdh;
}

int Worker::run()
{
    int                ret = 0;
    struct sched_param schedprm{.__sched_priority = sched_priority};
    int schedpol;

    if(running == true)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker {} is already running", wunique_id);
        return -1;
    }

    ret = pthread_create(&wid, NULL /* attr */, worker_init, (void*)(&wargs));
    if(ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Worker creation error {}", ret);
        return -1;
    }

    if(!wname.empty())
    {
        ret = pthread_setname_np(wid, wname.c_str());
        if(ret != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Worker set name error {}", ret);
            return -1;
        }
    }

    ret = pthread_setaffinity_np(wid, sizeof(wcpuset), &wcpuset);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Worker set affinity error {}", ret);
        return -1;
    }

#ifdef ENABLE_SCHED_FIFO_ALL_RT
    ret = pthread_setschedparam(wid, SCHED_FIFO, &schedprm);
    if (ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Worker set priority error {}", ret);
        return -1;
    }
#endif

    ret = pthread_getschedparam(wid, &schedpol, &schedprm);
    if (ret != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Could not get thread scheduling info");
        return -1;
    }

    if (schedpol != SCHED_FIFO)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Failed to apply SCHED_FIFO policy ({})", schedpol);
    }

    running = true;

    return 0;
}

const char* Worker::getName()
{
    return wname.c_str();
}

uint64_t Worker::getId()
{
    return wunique_id; //(int)wid;
}

int Worker::getPriority()
{
    struct sched_param _schedprm;
    int                _schedpol;

    if(running == true)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker is not running");
        return -1;
    }

    if(pthread_getschedparam(wid, &_schedpol, &_schedprm) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "Couldn't retrieve thread scheduling info");
        return -1;
    }

    // if(_schedprm.sched_priority != sched_priority)
    //     throw std::runtime_error("Priority changed after creation");

    return 0; //FIXME //sched_priority;
}

int Worker::getPolicy()
{
    struct sched_param _schedprm;
    int                _schedpol;

    if(running == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker is not running");
        return -1;
    }

    if(pthread_getschedparam(wid, &_schedpol, &_schedprm) != 0)
        PHYDRIVER_THROW_EXCEPTIONS(errno, "Could not get thread scheduling info");

    // if(_schedpol != schedpol)
    //     throw std::runtime_error("Policy changed after creation");

    return schedpol;
}

uint8_t Worker::getCPUAffinity()
{
    return cpucore;
}

void Worker::setExitValue(bool val)
{
    return exit.store(val);
}

bool Worker::getExitValue()
{
    return exit.load();
}

int Worker::waitExit()
{
    void* retval;

    if(running == false)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker is not running");
        return -1;
    }
    return pthread_join(wid, &retval);
}

void Worker::initPMU() {
    // PMUDeltaSummarizer must be constructed on the thread it will measure
    // (perf_event_open uses pid=0 = calling thread). Worker is constructed on
    // the main thread, but this runs on the worker thread after pthread_create.
    pmuds = new PMUDeltaSummarizer(pmu_type);
}

void Worker::destroyPMU() {
    delete pmuds;
    pmuds = nullptr;
}

enum worker_default_type Worker::getType()
{
    return type;
}

void* worker_init(void* arg)
{
    PhyDriverCtx*       pdctx                        = nullptr;
    Worker*             w                            = nullptr;
    struct worker_args* params                       = (struct worker_args*)arg;
    int (*start_routine)(phydriverwrk_handle, void*) = params->start_routine;
    void*               routine_arg                  = params->arg;
    phydriverwrk_handle whandler                     = (phydriverwrk_handle)params->whandler;

    w = static_cast<Worker*>(whandler);

    // PER_WORKER(_windex) = wptr->getIndex();
    // PER_WORKER(_wptr) = (uintptr_t)wptr;

    int ret = start_routine(whandler, routine_arg);

    return NULL;
}

//Experimental, should this be exposed?
// unsigned worker_get_id() {
//     return PER_WORKER(_windex);
// }

int worker_default(phydriverwrk_handle whandler, void* arg)
{
    PhyDriverCtx* pdctx = nullptr;
    Worker*       w     = nullptr;
    TaskList*     tList = nullptr;
    Task*         nTask = nullptr;
    t_ns          unlock_t, start_t, ts_now, acceptns((uint64_t)TIME_THRESHOLD_NS_TASK_ACCEPT);
    uint32_t      coreId = 0;
    CuphyOAM *oam = CuphyOAM::getInstance();
    sleep(1);
    try
    {
        w     = static_cast<Worker*>(whandler);
        pdctx = static_cast<PhyDriverCtx*>(w->getPhyDriverHandler());
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    w->initPMU();

    // Get coreId - only use first core in unlikely scenario worker is assigned multiple cores
    coreId = w->getCPUAffinity();
    oam->core_active[coreId] = 1;

    nvlog_fmtlog_thread_init(w->getName());
    NVLOGC_FMT(TAG, "Thread {}({}) on CPU {} initialized fmtlog", w->getName(), __FUNCTION__, coreId);

#ifdef ENABLE_DPDK_TX_PKT_TRACING
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if(rte_thread_register() != 0)
#pragma GCC diagnostic pop
    {
        NVLOGF_FMT(TAG, AERIAL_DPDK_API_EVENT, "rte_thread_register failed for coreID {}!",coreId);
    }
#endif

    if(w->getType() == WORKER_UL)
        tList = pdctx->getTaskListUl();
    else if (w->getType() == WORKER_DL)
        tList = pdctx->getTaskListDl();
    else if (w->getType() == WORKER_GENERIC) // Debug task uses generic type
        tList = pdctx->getTaskListDebug();
    else if (w->getType() == WORKER_DL_VALIDATION) // Debug task uses generic type
        tList = pdctx->getTaskListDlVal();
    else {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid worker type {}", +w->getType());
        return 1;
    }

    GpuDevice *gpu_device = pdctx->getFirstGpu();

    NVLOGI_FMT(TAG, "Worker {} type {} started using task list {} with gpu_id {} core {}",
                l1_worker_get_id(whandler), (w->getType() == WORKER_UL ? "Uplink" : (w->getType() == WORKER_DL ? "Downlink" : "Debug")),
                tList->getId(),gpu_device->getId(), coreId);

    // force CUDA API dynamic memory allocation prior to entering real-time code path
    cudaSetDevice(gpu_device->getId());

    // enable dynamic memory allocation tracing in real-time code path
    // Use LD_PRELOAD=<special .so> when running cuphycontroller, otherwise this does nothing.
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

    t_ns utilization_start_t = Time::nowNs();
    t_ns t_with_work = 0s;
    while(l1_worker_check_exit(whandler) == false)
    {
        start_t = Time::nowNs();

        tList->lock();
        nTask = tList->get_task(w->getId(), acceptns);
        tList->unlock();

        if(!nTask)
        {
            t_ns t_period = start_t - utilization_start_t;
            if (t_period >= 1s)
            {
                uint32_t utilization_x1000 = (t_with_work.count()*1000) / t_period.count();
                oam->cpu_utilization_x1000[coreId] = utilization_x1000;
                utilization_start_t = start_t;
                t_with_work = 0s;
            }

            std::this_thread::sleep_for(std::chrono::nanoseconds(1000));

            if (w->getType() == WORKER_GENERIC) //Debug thread
                //Adding this sleep because of the while(1) and SCHED FIFO issue where it hangs the other cores.
                std::this_thread::sleep_for(std::chrono::nanoseconds(5000));
        }
        else
        {
            ts_now = Time::nowNs();

            NVLOGD_FMT(TAG, "Starting Task {} at {} pop/lock latency time {} accept latency time {} "\
                        "Task Create {} Task Exec {} after {} us from Task creation and {} "\
                        "us before the execution timestamp\n",
                        nTask->getId(), Time::nowNs().count(),
                        Time::getDifferenceNowToNs(start_t).count(),
                        Time::getDifferenceNowToNs(ts_now).count(),
                        nTask->getTsCreate().count(),
                        nTask->getTsExec().count(),
                        (Time::NsToUs(ts_now - nTask->getTsCreate())).count(),
                        (Time::NsToUs(nTask->getTsExec() - ts_now)).count());


            if(w->getType() == WORKER_UL || w->getType() == WORKER_DL)
            {
                if(simulated_cpu_stall_checkpoint(w->getType() == WORKER_UL ? CUPHYDRIVER_UL_WORKER_THREAD : CUPHYDRIVER_DL_WORKER_THREAD,get_worker_task_id(nTask,w->getType())))
                {
                    if(w->getType() == WORKER_DL)
                        NVLOGI_FMT(TAG, "DL CPU stall simulated on core {} Slot Map {}",coreId,((SlotMapDl*)nTask->getTaskArgs())->getId());
                    else
                        NVLOGI_FMT(TAG, "UL CPU stall simulated on core {} Slot Map {}",coreId,((SlotMapUl*)nTask->getTaskArgs())->getId());
                }
            }

            nTask->run(w);
            unlock_t = Time::nowNs();
            t_with_work += unlock_t - ts_now;
        }

        // NVLOGC_FMT(TAG, "Consumer {} executed task {} at TS %" PRIu64 " (diff time %" PRIu64 " ns) in late: {} ."
        //         "lock time: %" PRIu64 " ns, "
        //         "work time: %" PRIu64 " ns, "
        //         "unlock time: %" PRIu64 " ns, "
        //         "tot time: %" PRIu64 " ns\n",
        //         l1_worker_get_id(wh), nTask->getName().c_str(), nTask->getTsExec(),
        //         nTask->getTsExec() - ts_create,
        //         (current_t > nTask->getTsExec() ? "Yes" : "No"),
        //         current_t - start_t, unlock_t - current_t,
        //         get_ns()-unlock_t, get_ns()-start_t);
    }

    return 0;

// err:
//     NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) << "Worker " << w->getId() << " is quitting with error";
//     return 1;
}

int get_worker_task_id(Task* t,worker_default_type w_type)
{
    std::string_view task_name=t->getName();
    int task_id=0;
    if(WORKER_UL==w_type)
    {
        //NVLOGC_FMT(TAG,"UL Task name {}",task_name.c_str());
        if(task_name=="TaskUL1AggrPucchPusch")
            task_id=0;
        else if(task_name=="TaskUL3AggrEarlyUciInd")
            task_id=1;
        else if(task_name=="TaskUL1AggrPrach")
            task_id=2;
        else if(task_name=="TaskUL1AggrSrs")
            task_id=3;
        else if(task_name=="TaskUL1AggrOrderkernel1")
            task_id=4;
        else if(task_name=="TaskUL3Aggr")
            task_id=5;
        else if(task_name=="TaskULAggrUlBfw")
            task_id=6;
        else
            task_id=7; //C-plane
    }
    else
    {
        //NVLOGC_FMT(TAG,"DL Task name {}",task_name.c_str());
        if(task_name=="TaskDL1AggrPdsch")
            task_id=0;
        else if(task_name=="TaskDL1AggrControl")
            task_id=1;
        else if(task_name=="TaskDLFHCb")
            task_id=2;
        else if(task_name=="TaskDL1AggrCompression")
            task_id=3;
        else if(task_name=="TaskDL2Aggr")
            task_id=4;
        else if(task_name=="TaskDL3Aggr")
            task_id=5;
        else if(task_name=="TaskDLAggrDlBfw")
            task_id=6;        
        else
            task_id=7; //C-plane
    }
    return task_id;
}

worker_id create_worker_id()
{
    return get_ns();
}
