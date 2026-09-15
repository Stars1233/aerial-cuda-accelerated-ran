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

// Must be included before #define TAG, since it has template with parameter <TAG> and the below
// will impact the parser.
#include "backward.hpp"

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 4) // "DRV.API"
#define TAG_STARTUP_TIMES (NVLOG_TAG_BASE_CUPHY_CONTROLLER + 5) // "CTL.STARTUP_TIMES"

#include "app_config.hpp"
#include "app_utils.hpp"
#include "constant.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include "context.hpp"
#include "framework_cplane_service.hpp"
#include "time.hpp"
#include "task.hpp"
#include "cell.hpp"
#include "slot_map_ul.hpp"
#include "slot_map_dl.hpp"
#include "worker.hpp"
#include "gpudevice.hpp"
#include "time.hpp"
#include "cuphydriver_api.hpp"
#include "exceptions.hpp"
#include "nvlog.hpp"
#include "cuphyoam.hpp"
#include "oran_utils/conversion.hpp"
#include "aerial/casts/casts.hpp"  // aerial::casts::assume_cast (alignment-checked)
#include <cuda_profiler_api.h>
#include <unistd.h>
#include "scf_5g_fapi.h"
#include "ti_generic.hpp"
#include <rte_trace.h>

#include "ptp_service_status_checking.hpp"
#include "rhocp_ptp_event_consumer.hpp"
#include "enum_utils.hpp"
#include <algorithm>
#include <range/v3/view/enumerate.hpp>

// Verify that the manually duplicated constant in compression_types.hpp stays in sync with
// the UserDataCompressionMethod enum in aerial-fh-driver/api.hpp.
// RESERVED (0b0111 = 7) is the first invalid enumerator, so its value equals the count of
// valid compression methods. If a new method is inserted before RESERVED, this fires.
static_assert(
    NUM_USER_DATA_COMPRESSION_METHODS ==
        static_cast<std::size_t>(aerial_fh::UserDataCompressionMethod::RESERVED),
    "NUM_USER_DATA_COMPRESSION_METHODS in compression_types.hpp is out of sync with "
    "UserDataCompressionMethod in aerial-fh-driver/api.hpp - update both together");

#define COMBINE_DL_TASKS_WITH_GPU_INIT_COMMS 0

phydriver_handle l1_pdh;
pthread_t gBg_thread_id;

void* ptp_svc_monitoring_func(void* arg);
void* rhocp_ptp_events_monitoring_func(void* arg);

int l1_init(phydriver_handle* _pdh, const context_config& ctx_cfg)
{
    TI_GENERIC_INIT("l1_init",8);

    TI_GENERIC_ADD("Start Task");

    PhyDriverCtx* pdctx    = nullptr;
    int           ret      = 0;
    int           gpu_id   = DEFAULT_GPU_ID;
    bool          init_gdr = false;
    GpuDevice * gDev;
    uint32_t cell_idx=0;

    try
    {
        TI_GENERIC_ADD("PhyDriverCtx construct");
        pdctx = new PhyDriverCtx(ctx_cfg);
        TI_GENERIC_ADD("PhyDriverCtx convert");
        *_pdh = StaticConversion<void>(pdctx).get();

        TI_GENERIC_ADD("Add each cell");
        uint32_t count = 0;
        // Add cells with M-plane parameters

        if(ctx_cfg.cell_mplane_list.size() > 0 && ctx_cfg.mMIMO_enable == 1)
        {
            auto t1a_min_cp_dl_ns = ctx_cfg.cell_mplane_list[0].t1a_min_cp_dl_ns;
            auto t1a_max_cp_dl_ns = ctx_cfg.cell_mplane_list[0].t1a_max_cp_dl_ns;
            auto t1a_min_cp_ul_ns = ctx_cfg.cell_mplane_list[0].t1a_min_cp_ul_ns;
            auto t1a_max_cp_ul_ns = ctx_cfg.cell_mplane_list[0].t1a_max_cp_ul_ns;

            for(auto& m : ctx_cfg.cell_mplane_list)
            {
                if(m.t1a_min_cp_dl_ns != t1a_min_cp_dl_ns)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_min_cp_dl_ns across cells");
                    goto exit;
                }
                if(m.t1a_max_cp_dl_ns != t1a_max_cp_dl_ns)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_max_cp_dl_ns across cells");
                    goto exit;
                }
                if(ctx_cfg.mMIMO_enable)
                {
                    if(m.t1a_min_cp_ul_ns != t1a_min_cp_ul_ns)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_min_cp_ul_ns across cells for MIMO case");
                        goto exit;
                    }
                    if(m.t1a_max_cp_ul_ns != t1a_max_cp_ul_ns)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "We don't support different t1a_max_cp_ul_ns across cells for MIMO case");
                        goto exit;
                    }
                }
            }
        }

        for(auto& m : ctx_cfg.cell_mplane_list)
        {
            ret = pdctx->addNewCell(m,cell_idx);
            if(ret)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "New cell creation error {}", ret);
                goto exit;
            }
            cell_idx++;
            if(++count == ctx_cfg.cell_group_num)
                break;
        }

        TI_GENERIC_ADD("PhyDriverCtx start");
        ret = pdctx->start();
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't start cuPHYDriver context {}", ret);
            goto exit;
        }
        if(AppConfig::getInstance().isPtpSvcMonitoringEnabled())
        {
            pthread_t thread_id;
            int       status = pthread_create(&thread_id, nullptr, ptp_svc_monitoring_func, *_pdh);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create ptp_service_monitoring_thread_func failed with status : {}", std::strerror(status));
                goto exit;
            }
        } 
        else if (AppConfig::getInstance().isRhocpPtpEventsMonitoringEnabled())
        {
            pthread_t thread_id;
            NVLOGC_FMT(TAG, "Now will start RHOCP PTP events monitoring thread");
            int       status = pthread_create(&thread_id, nullptr, rhocp_ptp_events_monitoring_func, *_pdh);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create rhocp_ptp_events_monitoring_thread_func failed with status : {}", std::strerror(status));
                goto exit;
            }
        }


        TI_GENERIC_ADD("End Task");
        TI_GENERIC_ALL_NVLOGI(TAG_STARTUP_TIMES);
        return ret;
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

exit:
    // All goto triggers are from errors above
    return -1;
}

int l1_finalize(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        delete pdctx;
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_worker_start_generic(phydriver_handle pdh, phydriverwrk_handle* _wh, const char* name, uint8_t affinity_core, uint32_t sched_priority, worker_routine wr, void* args)
{
    PhyDriverCtx* pdctx = nullptr;
    Worker*       w     = nullptr;
    int           ret   = 0;
    worker_id     wid   = 0;

    wid = create_worker_id();
    try
    {
        if(_wh == nullptr)
            PHYDRIVER_THROW_EXCEPTIONS(errno, "Worker handler provided is nullptr");

        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        w     = new Worker(pdh, wid, WORKER_GENERIC, name, affinity_core,
                           sched_priority, 0, wr, args); //Type here is useless
        *_wh  = w;

        // Add this metrics generic worker to the generic worker map - will be started in context.cpp
        if(pdctx->addGenericWorker(std::unique_ptr<Worker>(std::move(w))))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "New generic worker can't be added to the context {}: {}", errno , std::strerror(errno));
            PHYDRIVER_THROW_EXCEPTIONS(errno, "addNewWorker");
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

worker_id l1_worker_get_id(phydriverwrk_handle whandler)
{
    Worker* w = nullptr;

    try
    {
        w = StaticConversion<Worker>(whandler).get();
        return (worker_id)w->getId();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

phydriver_handle l1_worker_get_phydriver_handler(phydriverwrk_handle whandler)
{
    Worker* w = nullptr;

    try
    {
        w = StaticConversion<Worker>(whandler).get();
        return w->getPhyDriverHandler();
    }
    PHYDRIVER_CATCH_EXCEPTIONS_RETVAL(nullptr);

    return nullptr;
}

bool l1_worker_check_exit(phydriverwrk_handle whandler)
{
    Worker* w = nullptr;

    try
    {
        w = StaticConversion<Worker>(whandler).get();
        return w->getExitValue();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return true;
}

int l1_worker_stop(phydriverwrk_handle whandler)
{
    int           ret   = 0;
    worker_id     wid   = 0;
    Worker*       w     = nullptr;
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        w     = StaticConversion<Worker>(whandler).get();
        pdctx = StaticConversion<PhyDriverCtx>(w->getPhyDriverHandler()).get();
        wid   = l1_worker_get_id(whandler);
        if(wid == 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Worker doesn't exist");
            ret = -1;
        }

        ret = pdctx->removeWorker(wid);
        if(ret)
            PHYDRIVER_THROW_EXCEPTIONS(-1, "Can't remove worker");
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

exit:
    return ret;
}

int l1_set_output_callback(phydriver_handle pdh, struct slot_command_api::callbacks& cb)
{
    int           ret   = 0;
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    ret = pdctx->setUlCb(cb.ul_cb);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't set Uplink callback {}", ret);
        return -1;
    }

    ret = pdctx->setDlCb(cb.dl_cb);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Couldn't set Downlink callback {}", ret);
        return -1;
    }

    return 0;
}

int get_num_ulc_tasks(int num_workers) {
    //Previosly we scaled this based on number of cells
    //odd -> round up  (num_cells=15 > num_cplane_tasks=8)
    //even -> num_cells == 2*num_cplane_tasks (num_cells=16 > num_cplane_tasks=8)
    // return (num_cells / 2) + (num_cells%2);
    return num_workers;
}

int get_num_dlc_tasks(int num_workers,bool commViaCpu, uint8_t mMIMO_enable) {
    // Ensure minimum number of workers required
    if (num_workers < 2) {
        return 0;  // Not enough workers to handle DLC tasks
    }

    //Previosly we scaled this based on number of cells
    //odd -> round up  (num_cells=15 > num_cplane_tasks=8)
    //even -> num_cells == 2*num_cplane_tasks (num_cells=16 > num_cplane_tasks=8)
    // return (num_cells / 2) + (num_cells%2);
    // For muMIMO case keeping the num dlc tasks to 2 as here total number of DL 
    // cores are set to 4
    if(commViaCpu)
    {
        if(mMIMO_enable)
            return num_workers-3;
        else
            return num_workers-2;
    }
    else if(mMIMO_enable)
    {
        if(num_workers > 5)
            return num_workers-2;
        else
            return num_workers;
    }
    else
        return num_workers-1;
}

/// Returns true when @p flag is set in @p mask (i.e. that task category should be skipped).
/// @c EnqueueSkipMask is defined in cuphydriver_api.hpp (public API header).
[[nodiscard]] static bool skip_flag(EnqueueSkipMask mask, EnqueueSkipMask flag) noexcept
{
    return (to_underlying(mask) & to_underlying(flag)) != 0;
}

namespace {

// ---------------------------------------------------------------------------
// Enqueue helper types (TU-private): per-slot aggregator-pointer bundles, the
// DL layout bundle, and the shared timing+enqueue TaskRegistry.  Used only by the
// UL/DL emit helpers and enqueue_{ul,dl}_tasks below.
// ---------------------------------------------------------------------------

/// UL aggregator pointers for a slot (nullptr when that channel is not scheduled).
/// Collated so the UL emit helpers take one argument instead of five.
struct UlAggrPtrs final
{
    PhyPuschAggr* pusch = nullptr;
    PhyPucchAggr* pucch = nullptr;
    PhyPrachAggr* prach = nullptr;
    PhySrsAggr*   srs   = nullptr;
    PhyUlBfwAggr* ulbfw = nullptr;

    /// True when SRS is the only scheduled UL channel.
    [[nodiscard]] bool isSrsOnly() const noexcept
    {
        return srs != nullptr && pusch == nullptr && pucch == nullptr && prach == nullptr;
    }
};

/// DL aggregator pointers for a slot (nullptr when that channel is not scheduled).
/// Collated so the DL emit helpers take one argument instead of six.
struct DlAggrPtrs final
{
    PhyPdschAggr* pdsch    = nullptr;
    PhyPdcchAggr* pdcch_dl = nullptr;
    PhyPdcchAggr* pdcch_ul = nullptr;
    PhyPbchAggr*  pbch     = nullptr;
    PhyCsiRsAggr* csirs    = nullptr;
    PhyDlBfwAggr* dlbfw    = nullptr;

    /// True when DL BFW is the only scheduled DL work.
    [[nodiscard]] bool dlbfwOnly() const noexcept
    {
        return dlbfw != nullptr && !(pdsch || pdcch_dl || pdcch_ul || pbch || csirs);
    }
    /// True when any control-plane DL channel (PDCCH DL/UL or PBCH) is scheduled.
    [[nodiscard]] bool hasControl() const noexcept
    {
        return pdcch_dl || pdcch_ul || pbch;
    }
};

/// DL worker/layout configuration for a slot (collated to keep emit args small).
/// gpu_via_cpu/mMIMO/gpu_dl cache the corresponding PhyDriverCtx flags once per
/// slot so the emit helpers and set_dl_wait_counts don't re-read them through the
/// (out-of-line) accessors on every use.
struct DlLayout final
{
    int  num_cells        = 0;
    int  num_dlc_tasks    = 0;
    int  num_dl_workers   = 0;
    int  dl_worker_offset = 0;
    int  dlbfw_core_index = 0;
    bool single_dl_worker = false;
    bool enable_affinity  = false;
    bool gpu_via_cpu      = false;
    bool mMIMO            = false;
    bool gpu_dl           = false;
};

/**
 * @brief Co-locates task timing with task creation for a UL or DL slot map.
 *
 * One @c addTask() assigns the task's launch timestamp into @c task_ts_exec AND
 * creates+inits the task with that timestamp, then advances @c task_index — so
 * the timing and the enqueue for a task live at one call site.  Direction-specific
 * behaviour is expressed via defaulted args rather than separate types:
 *   - @p w is the worker affinity (0 = none, the UL default).
 *   - @p offset_by_index selects the usual per-task stagger (ts + task_index) vs.
 *     a raw timestamp (used by TaskDL3Aggr).
 * Task counts are published on the slot map (setTasksTs) and read by task bodies
 * via SlotMap{Ul,Dl}::getNumTasks(), so the init's third int is the vestigial
 * count slot.  Instantiated as @c UlTaskRegistry / @c DlTaskRegistry below.
 */
template <typename SlotMapT>
struct TaskRegistry final
{
    PhyDriverCtx* pdctx    = nullptr;                       // non-owning observer (Task pool owner)
    SlotMapT*     slot_map = nullptr;                       // non-owning observer
    // +1 vs task_ptr_list to match SlotMap{Ul,Dl}::setTasksTs() / the slot map's
    // tasks_ts_exec (std::array<t_ns, TASK_MAX_PER_SLOT + 1>), into which it is copied
    // wholesale; only [0..task_index) is ever populated.
    std::array<t_ns, TASK_MAX_PER_SLOT + 1> task_ts_exec;   // owned: per-task launch timestamps (published via setTasksTs)
    std::array<Task*, TASK_MAX_PER_SLOT>    task_ptr_list;  // owned: per-task pointers, drained to the TaskList after emission
    int           task_index = 0;

    /// The working arrays are intentionally left uninitialized; only [0..task_index) is written/read.
    TaskRegistry(PhyDriverCtx* p, SlotMapT* sm) : pdctx(p), slot_map(sm) {}

    // Single-use, drained once into the TaskList; non-owning Task* must not be duplicated.
    TaskRegistry(const TaskRegistry&)            = delete;
    TaskRegistry& operator=(const TaskRegistry&) = delete;
    TaskRegistry(TaskRegistry&&)                 = delete;
    TaskRegistry& operator=(TaskRegistry&&)      = delete;

    template <typename Fn>
    [[nodiscard]] bool addTask(t_ns ts, const char* name, Fn work_fn, int a, int b, int c,
                            worker_id w = 0, bool offset_by_index = true)
    {
        if (task_index >= TASK_MAX_PER_SLOT) [[unlikely]]
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Task index exceeds TASK_MAX_PER_SLOT! Unable to enqueue {} task", name);
            return false;
        }
        task_ts_exec[task_index]  = ts;
        task_ptr_list[task_index] = pdctx->getNextTask();
        const t_ns dispatch = offset_by_index ? ts + static_cast<t_ns>(task_index) : ts;
        task_ptr_list[task_index]->init(dispatch, name, work_fn, static_cast<void*>(slot_map), a, b, c, w);
        ++task_index;
        return true;
    }
};

using UlTaskRegistry = TaskRegistry<SlotMapUl>;
using DlTaskRegistry = TaskRegistry<SlotMapDl>;

/// Format "<prefix><idx>" into @p buf (NUL-terminated) for a per-task name.
template <std::size_t N>
void format_indexed_task_name(char (&buf)[N], std::string_view prefix, int idx)
{
    // format_to_n writes at most N-1 chars and returns .out one past the last char
    // written; reserve the final byte for the NUL. Over-long names truncate silently,
    // which is acceptable for trace-only task names.
    *fmt::format_to_n(buf, N - 1, "{}{}", prefix, idx).out = '\0';
}

/// Emit UL C-plane tasks (legacy path only).  Skipped under SKIP_UL_CPLANE
/// (the fapi_to_cplane_direct path emits UL C-plane via fire_cplane_batch).
[[nodiscard]] bool emit_ul_cplane(
    UlTaskRegistry& b, slot_command_api::slot_command* const sc,
    const bool skip_ul_cplane, const int num_ulc_tasks)
{
    if (skip_ul_cplane)
    {
        return true;
    }

    const t_ns cplane_ts = sc->tick_original + t_ns(Cell::getTtiNsFromMu(MU_SUPPORTED));
    for (int c = 0; c < num_ulc_tasks; c++)
    {
        char buf[32];
        format_indexed_task_name(buf, "TaskUL1AggrCplane", c + 1);
        if (!b.addTask(cplane_ts, buf, task_work_function_ul_aggr_1_cplane, c, 0, num_ulc_tasks))
        {
            return false;
        }
    }
    return true;
}

/// Order-kernel path when OK testbench mode is enabled: single Order Kernel only.
[[nodiscard]] bool emit_ul_order_kernel_tb(
    UlTaskRegistry& b, const uint64_t t0_slot, const UlAggrPtrs& aggr)
{
    const t_ns ok_ts = t_ns(t0_slot - UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
    return b.addTask(ok_ts, "TaskUL1AggrOrderkernel1",
                     task_work_function_ul_aggr_1_orderKernel, 0, 1, aggr.isSrsOnly() ? 1 : 0);
}

/// Order-kernel path for normal (non-TB) operation: one or two Order Kernel tasks.
[[nodiscard]] bool emit_ul_order_kernel(
    UlTaskRegistry& b, PhyDriverCtx* const pdctx, const uint64_t t0_slot,
    const ru_type ru_type_for_srs_proc, const UlAggrPtrs& aggr)
{
    const bool isSrsOnly = aggr.isSrsOnly();
    const t_ns ok_nonsrs = t_ns(t0_slot - UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);
    const t_ns ok_srs    = pdctx->gpuCommEnabledViaCpu()
                               ? t_ns(t0_slot + pdctx->getUlSrsTask1OrderLaunchOffsetNs())
                               : t_ns(t0_slot + UL_TASK1_ORDER_LAUNCH_OFFSET_FROM_T0_NS);

    if (aggr.srs && (ru_type_for_srs_proc != SINGLE_SECT_MODE))
    {
        if (isSrsOnly)
        {
            // Single SRS only OK
            return b.addTask(ok_srs, "TaskUL1AggrOrderkernel1",
                             task_work_function_ul_aggr_1_orderKernel, 0, 1 /* OK Task # */, 1 /* isSRS */);
        }
        // If other channels are present along with SRS, enqueue a non-SRS OK along with SRS OK as two
        // separate tasks. 
        if (!b.addTask(ok_nonsrs, "TaskUL1AggrOrderkernel1",
                       task_work_function_ul_aggr_1_orderKernel, 0, 1 /* OK Task # */, 0 /* isSRS */))
        {
            return false;
        }
        return b.addTask(ok_srs, "TaskUL1AggrOrderkernel2",
                         task_work_function_ul_aggr_1_orderKernel, 0, 2 /* OK Task # */, 1 /* isSRS */);
    }
    // Single non-SRS OK (or combined under SINGLE_SECT_MODE)
    return b.addTask(ok_nonsrs, "TaskUL1AggrOrderkernel1",
                     task_work_function_ul_aggr_1_orderKernel, 0, 1 /* OK Task # */, isSrsOnly ? 1 : 0);
}

/// Emit the UL channel tasks (non-orderKernel-TB path): PUCCH/PUSCH, UL2, Early-UCI,
/// UL BFW, PRACH, SRS.  @p num_cells == slot_map_ul->getNumCells().
[[nodiscard]] bool emit_ul_channel_tasks(
    UlTaskRegistry& b, PhyDriverCtx* const pdctx, slot_command_api::slot_command* const sc,
    const uint64_t t0_slot, const int num_cells, const UlAggrPtrs& aggr)
{
    if (aggr.pucch || aggr.pusch)
    {
        if (!b.addTask(t_ns(t0_slot - UL_TASK1_PUCCH_LAUNCH_OFFSET_FROM_T0_NS),
                    "TaskUL1AggrPucchPusch", task_work_function_ul_aggr_1_pucch_pusch, 0, num_cells, 0))
        {
            return false;
        }
    }
    if (pdctx->cpuCommEnabled())
    {
        if (!b.addTask(t_ns(t0_slot + UL_TASK2_OFFSET_FROM_T0_NS),
                    "TaskUL2Aggr", task_work_function_ul_aggr_2, 0, num_cells, 0))
        {
            return false;
        }
    }
    if (aggr.pusch)
    {
        // Early UCI
        const t_ns early_uci_ts = t_ns(t0_slot + UL_TASK3_EARLY_UCI_IND_TASK_LAUNCH_OFFSET_FROM_T0_NS);

        if (!b.addTask(early_uci_ts, "TaskUL3AggrEarlyUciInd", task_work_function_ul_aggr_3_early_uci_ind,
                    0, num_cells, 0))
        {
            return false;
        }
    }
    if (aggr.ulbfw)
    {
        // BFW: immediate action time; then UL3 BFW before T0.
        if (!b.addTask(sc->tick_original, "TaskULAggrUlBfw", task_work_function_ul_aggr_bfw, 0, num_cells, 0))
        {
            return false;
        }
        if (!b.addTask(t_ns(t0_slot - UL_AGGR3_ULBFW_OFFSET_FROM_T0_NS),
                    "TaskUL3AggrUlBfw", task_work_function_ul_aggr_3_ulbfw, 0, num_cells, 0))
        {
            return false;
        }
    }
    if (aggr.prach)
    {
        if (!b.addTask(t_ns(t0_slot + Cell::getTtiNsFromMu(MU_SUPPORTED)),
                    "TaskUL1AggrPrach", task_work_function_ul_aggr_1_prach, 0, num_cells, 0))
        {
            return false;
        }
    }
    if (aggr.srs)
    {
        const t_ns srs_ts = pdctx->gpuCommEnabledViaCpu()
            ? t_ns(t0_slot + pdctx->getUlSrsTask1OrderLaunchOffsetNs())
            : t_ns(t0_slot + Cell::getTtiNsFromMu(MU_SUPPORTED));
        if (!b.addTask(srs_ts, "TaskUL1AggrSrs", task_work_function_ul_aggr_1_srs, 0, num_cells, 0))
        {
            return false;
        }
    }
    return true;
}

/// Emit UL3 wait/callback tasks (do-while over cell groups).
[[nodiscard]] bool emit_ul_aggr3(
    UlTaskRegistry& b, SlotMapUl* const slot_map_ul, const uint64_t t0_slot, const int num_cells)
{
    const t_ns ul3_ts = t_ns(t0_slot + UL_TASK3_AGGR3_OFFSET_FROM_T0_NS);
    int first_cell = 0;
    do
    {
        if (!b.addTask(ul3_ts, "TaskUL3Aggr", task_work_function_ul_aggr_3, first_cell, num_cells, 0))
        {
            return false;
        }
        first_cell += num_cells;
    } while (first_cell < slot_map_ul->getNumCells() && b.task_index < TASK_MAX_PER_SLOT);
    return true;
}

/// Emit UL3-SRS wait task (FX RU: dedicated AGGR3 awaiting SRS completion).
[[nodiscard]] bool emit_ul_aggr3_srs(
    UlTaskRegistry& b, PhyDriverCtx* const pdctx, SlotMapUl* const slot_map_ul,
    const uint64_t t0_slot, const int num_cells)
{
    uint32_t task_offset_from_t0 =
        (pdctx->getUlSrsAggr3TaskLaunchOffsetNs() > UL_TASK3_AGGR3_MAX_BACKOFF_FROM_SRS_COMPLETION_TH_NS)
            ? UL_TASK3_AGGR3_MAX_BACKOFF_FROM_SRS_COMPLETION_TH_NS
            : pdctx->getUlSrsAggr3TaskLaunchOffsetNs();
    task_offset_from_t0 = SRS_COMPLETION_TH_FROM_T0_NS - task_offset_from_t0;
    const t_ns ul3srs_ts = t_ns(t0_slot + task_offset_from_t0);
    int first_cell = 0;
    do
    {
        if (!b.addTask(ul3srs_ts, "TaskUL3AggrSrs", task_work_function_ul_aggr_3_srs, first_cell, num_cells, 0))
        {
            return false;
        }
        first_cell += num_cells;
    } while (first_cell < slot_map_ul->getNumCells() && b.task_index < TASK_MAX_PER_SLOT);
    return true;
}

} // namespace

/**
 * @brief UL phase of l1_enqueue_phy_work: emit (timestamp + create + enqueue) UL tasks.
 *
 * Single-pass co-location via UlTaskRegistry: each task's launch timestamp is
 * assigned together with its creation/init (see emit_ul_* helpers).  The total
 * UL task count is derived from the emission — @c task_index plus the fapi_to_cplane_direct
 * virtual C-plane batch-done contributors (num_ulc_tasks under SKIP_UL_CPLANE) —
 * and published via @c setTasksTs() before the tasks are pushed; UL3 / UL3-SRS
 * read it via @c SlotMapUl::getNumTasks().  @p ul_task_count (the legacy
 * pre-accumulated budget) is no longer used here.
 *
 * @return 0 on success; non-zero to request the caller's cleanup_err path.
 */
[[nodiscard]] static int enqueue_ul_tasks(
    PhyDriverCtx* const                      pdctx,
    SlotMapUl* const                         slot_map_ul,
    slot_command_api::slot_command* const    sc,
    const uint64_t                           t0_slot,
    const EnqueueSkipMask                    skip_mask,
    const bool                               en_orderKernel_tb,
    const ru_type                            ru_type_for_srs_proc,
    const UlAggrPtrs&                        aggr,
    slot_params_aggr* const                  current_slot_params_aggr,
    [[maybe_unused]] const int               ul_task_count)
{
    // UL slot reference time (L2A tick + 1 slot); see SlotMapUl::getSlotRefTs().
    slot_map_ul->setSlotRefTs(sc->tick_original + t_ns(Cell::getTtiNsFromMu(MU_SUPPORTED)));

    const bool skip_ul_cplane = skip_flag(skip_mask, EnqueueSkipMask::SKIP_UL_CPLANE);
    
    // In the non-FAPI to CplaneDirect path, the # of ULC tasks are to be recorded in the
    // newly allocated slot_map. In the FAPI to CplaneDirect path, this is done by 
    // l1_set_num_ulc_tasks_for_slot
    if (!pdctx->isFapiToCplaneDirect())
    {
        slot_map_ul->setNumUlcTasks(get_num_ulc_tasks(pdctx->getNumULWorkers()));
    }

    const int num_ulc_tasks = slot_map_ul->getNumUlcTasks();

    if (!(slot_map_ul->getNumCells() > 0 || aggr.ulbfw))
    {
        return 1;
    }

    if (slot_map_ul->aggrSetPhy(aggr.pusch, aggr.pucch, aggr.prach, aggr.srs, aggr.ulbfw, current_slot_params_aggr))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMapUL aggrSetPhy");
        return 1;
    }
    // Skip enqueue if an L1 exit is in flight.
    if (pExitHandler.test_exit_in_flight())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "L1 Exit in flight during Slot Map {}", slot_map_ul->getId());
        return 1;
    }

    const t_ns map_ul_start = Time::nowNs();
    PUSH_RANGE_PHYDRV("SLOT_UL", 4);
    const int num_cells = slot_map_ul->getNumCells();

    UlTaskRegistry b{pdctx, slot_map_ul};

    if (en_orderKernel_tb) [[unlikely]]
    {
        // OK testbench: C-plane + single Order Kernel + UL3 only.
        if (!emit_ul_cplane(b, sc, skip_ul_cplane, num_ulc_tasks))
        {
            return 1;
        }
        if (!emit_ul_order_kernel_tb(b, t0_slot, aggr))
        {
            return 1;
        }
        if (!emit_ul_aggr3(b, slot_map_ul, t0_slot, num_cells))
        {
            return 1;
        }
    }
    else
    {
        // Normal path: C-Plane + Order Kernel(s), channel tasks, then UL3 + optional UL3-SRS.
        if (!emit_ul_cplane(b, sc, skip_ul_cplane, num_ulc_tasks))
        {
            return 1;
        }
        if (!emit_ul_order_kernel(b, pdctx, t0_slot, ru_type_for_srs_proc, aggr))
        {
            return 1;
        }
        if (!emit_ul_channel_tasks(b, pdctx, sc, t0_slot, num_cells, aggr))
        {
            return 1;
        }
        if (!emit_ul_aggr3(b, slot_map_ul, t0_slot, num_cells))
        {
            return 1;
        }
        if (aggr.srs && ru_type_for_srs_proc != SINGLE_SECT_MODE)
        {
            if (!emit_ul_aggr3_srs(b, pdctx, slot_map_ul, t0_slot, num_cells))
            {
                return 1;
            }
        }
    }
    

    // Authoritative UL task count = emitted driver tasks + fapi_to_cplane_direct virtual UL
    // C-plane batch-done signals (num_ulc_tasks under SKIP_UL_CPLANE).  Publish on
    // the slot map before pushing; UL3 / UL3-SRS read it via getNumTasks().
    const int ul_task_count_actual = b.task_index + (skip_ul_cplane ? num_ulc_tasks : 0);
    slot_map_ul->setTasksTs(ul_task_count_actual, b.task_ts_exec, Time::nowNs());

    // Push the emitted (non-virtual) tasks under a single lock; bound by task_index
    // (the fapi_to_cplane_direct path's batch-done contributors are counted but not pushed).
    TaskList* tListUl = pdctx->getTaskListUl();
    const int ul_pushed = tListUl->push_bulk(
        std::span<Task* const>{b.task_ptr_list.data(), static_cast<std::size_t>(b.task_index)});
    if (ul_pushed != b.task_index) [[unlikely]]
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "UL push_bulk enqueued {}/{} tasks for Map {}",
                   ul_pushed, b.task_index, slot_map_ul->getId());
        POP_RANGE
        return 1;
    }

    POP_RANGE
    const t_ns map_ul_end = Time::nowNs();
    NVLOGI_FMT(TAG, "Enqueue UL tasks START: {} END: {} DURATION: {} us", map_ul_start.count(), map_ul_end.count(), Time::NsToUs(map_ul_end - map_ul_start).count());
    return 0;
}

namespace {

/// DL worker-ID assignment using the same abstraction as the legacy code:
/// gated by DL core affinity, delegating to PhyDriverCtx::getDLWorkerID.
[[nodiscard]] inline worker_id dl_worker(PhyDriverCtx* const pdctx, const bool enable_affinity, const int worker_index)
{
    return enable_affinity ? pdctx->getDLWorkerID(worker_index) : 0;
}

/**
 * Number of DL channel-task workers available in this layout.
 *
 * Single-worker mode collapses every channel role onto worker 0; standard
 * mode advertises @c num_dl_workers minus the reserved offset (the trailing
 * TX / doorbell workers).
 *
 * @param[in] lay  The DL layout for the current slot.
 * @return         @c 1 when @c lay.single_dl_worker is set, otherwise
 *                 @c lay.num_dl_workers - @c lay.dl_worker_offset.
 */
[[nodiscard]] inline int dl_channel_worker_count(const DlLayout& lay)
{
    return lay.single_dl_worker ? 1 : lay.num_dl_workers - lay.dl_worker_offset;
}

/**
 * Resolve the @c worker_id for a DL task from a layout + logical index.
 *
 * Routes the requested @c worker_index to worker 0 in single-worker mode, and
 * to @c dl_worker(pdctx, enable_affinity, worker_index) otherwise. Callers use
 * this instead of @c dl_worker directly so single-worker deployments do not
 * model a non-existent second worker.
 *
 * @param[in] pdctx         Driver context (never null on the DL emit paths).
 * @param[in] lay           The DL layout for the current slot.
 * @param[in] worker_index  Logical worker index the caller would use in the
 *                          non-single-worker path.
 * @return                  Concrete @c worker_id for the task.
 */
[[nodiscard]] inline worker_id dl_layout_worker(
    PhyDriverCtx* const pdctx, const DlLayout& lay, const int worker_index)
{
    return dl_worker(pdctx, lay.enable_affinity, lay.single_dl_worker ? 0 : worker_index);
}

/// Task1 channel tasks: FH callback, PDSCH, control (PDCCH/PBCH/CSI-RS).
/// Caller emits these only on a non-DLBFW-only slot (gating lives in enqueue_dl_tasks).
[[nodiscard]] bool emit_dl_channel_tasks(
    DlTaskRegistry& b, const t_ns base_ts, const DlLayout& lay, const EnqueueSkipMask skip_mask, const DlAggrPtrs& aggr)
{
    PhyDriverCtx* const pdctx = b.pdctx;
    // FH callback — excluded under SKIP_DL_FHCB (the fapi_to_cplane_direct path emits FHCB-done via batch).
    if (!skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_FHCB))
    {
        if (!b.addTask(base_ts, "TaskDLFHCb", task_work_function_dl_fh_cb, 0, lay.num_cells, 0,
                    dl_layout_worker(pdctx, lay, dl_channel_worker_count(lay))))
        {
            return false;
        }
    }
    if (aggr.pdsch)
    {
        const int wi = lay.mMIMO ? (lay.num_dl_workers - lay.dl_worker_offset) : 0;
        if (!b.addTask(base_ts, "TaskDL1AggrPdsch", task_work_function_dl_aggr_1_pdsch, 0, lay.num_cells, 0,
                    dl_layout_worker(pdctx, lay, wi)))
        {
            return false;
        }
    }
    if (aggr.hasControl() || aggr.csirs)
    {
        const int wi = lay.mMIMO ? lay.dlbfw_core_index : 1;
        if (!b.addTask(base_ts, "TaskDL1AggrControl", task_work_function_dl_aggr_control, 0, lay.num_cells, 0,
                    dl_layout_worker(pdctx, lay, wi)))
        {
            return false;
        }
    }
    return true;
}

/// DL BFW task.  Emitted whenever DL BFW is scheduled (alone, or alongside channel tasks).
[[nodiscard]] bool emit_dl_bfw(DlTaskRegistry& b, const t_ns base_ts, const DlLayout& lay)
{
    return b.addTask(base_ts, "TaskDLAggrDlBfw", task_work_function_dl_aggr_bfw, 0, lay.num_cells, 0,
                  dl_layout_worker(b.pdctx, lay, lay.dlbfw_core_index));
}

/// Task2 part A: U-plane prepare + TX (GPU comm) or DL2 aggregate, then DL C-plane
/// (with per-DLC mMIMO U-plane prepare).  Caller ensures !dlbfwOnly().
[[nodiscard]] bool emit_dl_uplane_and_cplane(
    DlTaskRegistry& b, const t_ns base_ts, const DlLayout& lay, const EnqueueSkipMask skip_mask)
{
    PhyDriverCtx* const pdctx = b.pdctx;
    const worker_id w_txcore = dl_layout_worker(pdctx, lay, dl_channel_worker_count(lay));

    if (lay.gpu_dl)
    {
        // Non-mMIMO GPU-comm prepare (excluded under SKIP_DL_GPU_COMM_PREPARE).
        if (!lay.mMIMO && !skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_GPU_COMM_PREPARE))
        {
            for (int c = lay.num_dlc_tasks - 1; c >= 0; c--)
            {
                char buf[32];
                format_indexed_task_name(buf, "TaskDL2AggrPrepare", c + 1);
                if (!b.addTask(base_ts, buf, task_work_function_dl_aggr_2_gpu_comm_prepare, c, 0, lay.num_dlc_tasks, w_txcore))
                {
                    return false;
                }
            }
        }
        if (!b.addTask(base_ts, "TaskDL2AggrTx", task_work_function_dl_aggr_2_gpu_comm_tx, 0, lay.num_cells, lay.num_dlc_tasks, w_txcore))
        {
            return false;
        }
    }
    else
    {
        if (!b.addTask(base_ts, "TaskDL2Aggr", task_work_function_dl_aggr_2, 0, lay.num_cells, 0, w_txcore))
        {
            return false;
        }
    }

    // DL C-plane (+ per-DLC mMIMO U-plane prepare).
    if (!skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_CPLANE))
    {
        for (int c = lay.num_dlc_tasks - 1; c >= 0; c--)
        {
            char buf[32];
            format_indexed_task_name(buf, "TaskDL1AggrCplane", c + 1);
            const int affinity_worker_index = c % dl_channel_worker_count(lay);
            if (!b.addTask(base_ts, buf, task_work_function_cplane, c, 0, lay.num_dlc_tasks,
                        dl_layout_worker(pdctx, lay, affinity_worker_index)))
            {
                return false;
            }
            if (lay.mMIMO && !skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_GPU_COMM_PREPARE))
            {
                format_indexed_task_name(buf, "TaskDL2AggrPrepare", c + 1);
                if (!b.addTask(base_ts, buf, task_work_function_dl_aggr_2_gpu_comm_prepare, c, 0, lay.num_dlc_tasks,
                            dl_layout_worker(pdctx, lay, affinity_worker_index)))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

/// Task2 part B: Compression, then CPU doorbell (gpuCommEnabledViaCpu).
/// Caller ensures !dlbfwOnly().
[[nodiscard]] bool emit_dl_compress_and_doorbell(
    DlTaskRegistry& b, const t_ns base_ts, const DlLayout& lay, const DlAggrPtrs& aggr)
{
    PhyDriverCtx* const pdctx = b.pdctx;
    const int comp_wi = aggr.dlbfw ? lay.dlbfw_core_index : (lay.num_dl_workers - lay.dl_worker_offset);
    if (!b.addTask(base_ts, "TaskDL1AggrCompression", task_work_function_dl_aggr_1_compression, 0, lay.num_cells, 0,
                dl_layout_worker(pdctx, lay, comp_wi)))
    {
        return false;
    }

    if (lay.gpu_via_cpu)
    {
        if (!b.addTask(base_ts, "TaskDL2RingCpuDoorbell", task_work_function_dl_aggr_2_ring_cpu_doorbell, 0, lay.num_cells, 0,
                    dl_layout_worker(pdctx, lay, lay.num_dl_workers - 1)))
        {
            return false;
        }
    }
    return true;
}

/// DL3 buffer-cleanup task (always scheduled).  Launches at T0 + 1 slot with a raw
/// timestamp (no per-index offset).
[[nodiscard]] bool emit_dl_aggr3(DlTaskRegistry& b, const uint64_t t0_slot, const DlLayout& lay)
{
    return b.addTask(t_ns(t0_slot + Cell::getTtiNsFromMu(MU_SUPPORTED)), "TaskDL3Aggr",
                  task_work_function_dl_aggr_3_buf_cleanup, 0, lay.num_cells, 0,
                  dl_layout_worker(b.pdctx, lay, dl_channel_worker_count(lay)),
                  /*offset_by_index=*/false);
}

/// Publish the DL2Tx / buf-cleanup wait targets on the slot map (read by the
/// compression and buf-cleanup task bodies).  No-op for DLBFW-only slots.
void set_dl_wait_counts(
    SlotMapDl* const slot_map_dl, const DlLayout& lay, const EnqueueSkipMask skip_mask,
    const int num_dlc_tasks, const DlAggrPtrs& aggr, const int dl_task_count)
{
    if (aggr.dlbfwOnly())
    {
        return;
    }
    const bool skip_fhcb    = skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_FHCB);
    const bool skip_prepare = skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_GPU_COMM_PREPARE);
    // U-plane prepare tasks are emitted by emit_dl_uplane_and_cplane only when !skip_prepare AND
    // (mMIMO || gpuCommDlEnabled).  dl_task_count is emission-derived, so the ignore-list must use
    // the same predicate (not the legacy budget's unconditional num_dlc_tasks); otherwise in the
    // non-GPU-comm-DL config it would over-subtract num_dlc_tasks and under-count the wait target.
    const bool prepare_emitted = !skip_prepare && (lay.mMIMO || lay.gpu_dl);

    // Compression wait: ignore DLBFW + Compression + CPU-Doorbell(iff gpu_via_cpu)
    // + BufCleanup + FHCB + DLC + UPlane-prepare + batch-done.
    // Uses waitSlotChannelEnd (atom_dl_channel_end_threads counter), which the batch-done
    // virtual task does NOT contribute to — it is therefore added to the ignore list so
    // numTasksWaitDl2Tx stays invariant under the fapi_to_cplane_direct path.
    // Ignore-list: DLBFW + Compression(1) + CPU-Doorbell(iff gpu_via_cpu) + BufCleanup(1) + FHCB + DLC + UPlane prepare(iff emitted) + batch-done(iff fapi_to_cplane_direct).
    //   Identity (SKIP_NONE, dl_batch_done_w=0, prepare emitted = N):
    //     gpu_via_cpu=true,  dlbfw_present=true:  1 + 1 + 1 + 1 + 1 + N + N + 0 = 5 + 2N  → num_dl_tasks - (5 + 2N)
    //     gpu_via_cpu=true,  dlbfw_present=false: 0 + 1 + 1 + 1 + 1 + N + N + 0 = 4 + 2N  → num_dl_tasks - (4 + 2N)
    //     gpu_via_cpu=false, dlbfw_present=true:  1 + 1 + 0 + 1 + 1 + N + N + 0 = 4 + 2N  → num_dl_tasks - (4 + 2N)
    //     gpu_via_cpu=false, dlbfw_present=false: 0 + 1 + 0 + 1 + 1 + N + N + 0 = 3 + 2N  → num_dl_tasks - (3 + 2N)
    const int dl_ignore_compress =
          (aggr.dlbfw ? 1 : 0)
        + 1                                         // Compression
        + (lay.gpu_via_cpu ? 1 : 0)                 // CPU Doorbell
        + 1                                         // DL3 Buf Cleanup
        + (skip_fhcb    ? 0 : 1)                    // when FhCb is skipped, no need to wait for it
        + num_dlc_tasks                             // C-Plane tasks (either fapi_to_cplane direct path or not, shall emit num_dlc_tasks signals). 
        + (prepare_emitted ? num_dlc_tasks : 0);    // U-plane prepare: only ignore when actually emitted (see prepare_emitted above).
    slot_map_dl->setNumTasksWaitDl2Tx(dl_task_count - dl_ignore_compress);

    // Buf-cleanup wait: ignore dl_fixed + DLC.
    const int dl_fixed = lay.gpu_via_cpu ? 4 : 3;
    const int dl_ignore_buf_cleanup = dl_fixed + num_dlc_tasks;
    slot_map_dl->setNumTasksWaitBufCleanup(dl_task_count - dl_ignore_buf_cleanup);
}

} // namespace

/**
 * @brief DL phase of l1_enqueue_phy_work: emit (timestamp + create + enqueue) DL tasks.
 *
 * Single-pass co-location via DlTaskRegistry (mirrors the UL path).  The total DL
 * task count is derived from the emission — @c task_index plus the fapi_to_cplane_direct
 * virtual C-plane batch-done contributors (num_dlc_tasks under SKIP_DL_FHCB) —
 * published via @c setTasksTs() before the tasks are pushed; the compression and
 * buf-cleanup bodies read it via @c SlotMapDl::getNumTasks(), and their wait
 * targets via getNumTasksWaitDl2Tx()/BufCleanup().  @p dl_task_count (legacy
 * pre-accumulated budget) and @p dl_from_early are no longer used here
 * (dl_from_early's index-0 value equals sc->tick_original on this path).
 *
 * @return 0 on success; non-zero to request the caller's cleanup_err path.
 */
[[nodiscard]] static int enqueue_dl_tasks(
    PhyDriverCtx* const                      pdctx,
    SlotMapDl* const                         slot_map_dl,
    slot_command_api::slot_command* const    sc,
    const uint64_t                           t0_slot,
    const EnqueueSkipMask                    skip_mask,
    [[maybe_unused]] const bool              dl_from_early,
    const DlAggrPtrs&                        aggr,
    slot_params_aggr* const                  current_slot_params_aggr,
    [[maybe_unused]] const int               dl_task_count)
{

    // DL slot reference time (L2A tick); see SlotMapDl::getSlotRefTs().
    // In fapi_to_cplane_direct the early-cplane setup already set SlotRefTs to this same value,
    // so skip the redundant rewrite: it lets the direct C-plane send (send_dl_cplane) read
    // SlotRefTs off the shared early map without racing this write. Mirrors the existing
    // !dl_from_early guard on setSlot3GPP.
    if (!dl_from_early)
    {
        slot_map_dl->setSlotRefTs(sc->tick_original);
    }

    // Cache PhyDriverCtx flags once per slot (out-of-line accessors; values are
    // slot-invariant). Used here, threaded into DlLayout, and consumed by the emit
    // helpers / set_dl_wait_counts.
    const bool fapi_direct    = pdctx->isFapiToCplaneDirect();
    const bool gpu_via_cpu    = pdctx->gpuCommEnabledViaCpu();
    const bool mMIMO          = pdctx->getmMIMO_enable();
    const bool gpu_dl         = pdctx->gpuCommDlEnabled();
    const int  num_dl_workers = pdctx->getNumDLWorkers();

    // Store/replay supports the non-mMIMO GPU-communication graph on one DL
    // worker. All DL roles map to worker 0 and the layout retains one real
    // C-plane contributor rather than modelling a nonexistent second worker.
    const bool single_dl_worker =
#ifdef ENABLE_FAPI_STORE_REPLAY
        num_dl_workers == 1 && !mMIMO && !gpu_via_cpu;
#else
        false;
#endif

    // In the non-FAPI to CplaneDirect path, the # of DLC tasks are to be recorded in the
    // newly allocated slot_map. In the FAPI to CplaneDirect path, this is done by 
    // l1_set_num_dlc_tasks_for_slot
    if (!fapi_direct)
    {
        slot_map_dl->setNumDlcTasks(
            single_dl_worker ? 1 : get_num_dlc_tasks(num_dl_workers, gpu_via_cpu, mMIMO));
    }

    const int num_dlc_tasks = slot_map_dl->getNumDlcTasks();

    const int dl_worker_offset = gpu_via_cpu ? 2 : 1;
    int dlbfw_core_index;
    if (mMIMO && gpu_via_cpu)
    {
        if (num_dl_workers < 3)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Insufficient DL workers ({}) for mMIMO with GPU comm via CPU", num_dl_workers);
            return 1;
        }
        dlbfw_core_index = num_dl_workers - 3;
    }
    else
    {
        if (num_dl_workers < 2 && !single_dl_worker)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Insufficient DL workers ({})", num_dl_workers);
            return 1;
        }
        dlbfw_core_index = single_dl_worker ? 0 : num_dl_workers - 2;
    }

    if (!(slot_map_dl->getNumCells() > 0 || aggr.dlbfw))
    {
        return 1;
    }

    if (slot_map_dl->aggrSetPhy(aggr.pdsch, aggr.pdcch_dl, aggr.pdcch_ul, aggr.pbch, aggr.csirs, aggr.dlbfw, current_slot_params_aggr))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMapDL aggrSetPhy");
        return 1;
    }
    // Skip enqueue if an L1 exit is in flight.
    if (pExitHandler.test_exit_in_flight())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "L1 Exit in flight during Slot Map {}", slot_map_dl->getId());
        return 1;
    }

    const t_ns map_dl_start = Time::nowNs();
    PUSH_RANGE_PHYDRV("SLOT_DL", 5);
    const int  num_cells  = slot_map_dl->getNumCells();
    const t_ns dl_base_ts = sc->tick_original;

    const DlLayout lay{.num_cells        = num_cells,
                       .num_dlc_tasks    = num_dlc_tasks,
                       .num_dl_workers   = num_dl_workers,
                       .dl_worker_offset = dl_worker_offset,
                       .dlbfw_core_index = dlbfw_core_index,
                       .single_dl_worker = single_dl_worker,
                       .enable_affinity  = (pdctx->get_enable_dl_core_affinity() != 0),
                       .gpu_via_cpu      = gpu_via_cpu,
                       .mMIMO            = mMIMO,
                       .gpu_dl           = gpu_dl};
    DlTaskRegistry b{pdctx, slot_map_dl};

    // DL task layout, gated in one place: a slot with real channel work emits the
    // full pipeline (channels, optional BFW, U-plane/C-plane, compression/doorbell);
    // a BFW-only slot emits just BFW; DL3 buffer-cleanup is always emitted last.
    // Emission order is preserved exactly (dispatch time = ts + task_index).
    if (!aggr.dlbfwOnly())
    {
        if (!emit_dl_channel_tasks(b, dl_base_ts, lay, skip_mask, aggr))
        {
            return 1;
        }
        if (aggr.dlbfw && !emit_dl_bfw(b, dl_base_ts, lay))
        {
            return 1;
        }
        if (!emit_dl_uplane_and_cplane(b, dl_base_ts, lay, skip_mask))
        {
            return 1;
        }
        if (!emit_dl_compress_and_doorbell(b, dl_base_ts, lay, aggr))
        {
            return 1;
        }
    }
    else
    {
        // dlbfwOnly() implies aggr.dlbfw != nullptr.
        if (!emit_dl_bfw(b, dl_base_ts, lay))
        {
            return 1;
        }
    }
    if (!emit_dl_aggr3(b, t0_slot, lay))
    {
        return 1;
    }

    // Authoritative DL task count = emitted driver tasks + fapi_to_cplane_direct virtual DL
    // C-plane batch-done signals (num_dlc_tasks under SKIP_DL_FHCB).  Publish on
    // the slot map and derive the wait targets before pushing; the compression and
    // buf-cleanup bodies read getNumTasks() / getNumTasksWaitDl2Tx()/BufCleanup().
    const bool skip_fhcb = skip_flag(skip_mask, EnqueueSkipMask::SKIP_DL_FHCB);
    const int dl_task_count_actual = b.task_index + (skip_fhcb ? num_dlc_tasks : 0);
    set_dl_wait_counts(slot_map_dl, lay, skip_mask, num_dlc_tasks, aggr, dl_task_count_actual);
    slot_map_dl->setTasksTs(dl_task_count_actual, b.task_ts_exec, Time::nowNs());
    NVLOGI_FMT(TAG, "Map {} dl_task_count {} dlbfw_only {}", slot_map_dl->getId(), dl_task_count_actual, aggr.dlbfwOnly());

    // Push the emitted tasks under a single lock; bound by task_index (scheduled tasks),
    // not the count — the fapi_to_cplane_direct path's virtual batch-done contributors are
    // counted but not pushed.
    TaskList* tListDl = pdctx->getTaskListDl();
    const int dl_pushed = tListDl->push_bulk(
        std::span<Task* const>{b.task_ptr_list.data(), static_cast<std::size_t>(b.task_index)});
    if (dl_pushed != b.task_index) [[unlikely]]
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "DL push_bulk enqueued {}/{} tasks for Map {}",
                   dl_pushed, b.task_index, slot_map_dl->getId());
        POP_RANGE
        return 1;
    }

    if (pdctx->debug_worker_enabled())
    {
        auto debug_task = pdctx->getNextTask();
        debug_task->init(dl_base_ts, "Debug", task_work_function_debug, static_cast<void*>(slot_map_dl),
                         0, num_cells, dl_task_count_actual, 0);
        auto dl = pdctx->getTaskListDebug();
        dl->lock();
        dl->push(debug_task);
        dl->unlock();
    }
    POP_RANGE
    const t_ns map_dl_end = Time::nowNs();
    NVLOGI_FMT(TAG, "Enqueue DL tasks START: {} END: {} DURATION: {} us", map_dl_start.count(), map_dl_end.count(), Time::NsToUs(map_dl_end - map_dl_start).count());
    return 0;
}

namespace {

/**
 * Resolve the next OrderEntity for the UL slot.
 *
 * Under @c ENABLE_FAPI_STORE_REPLAY, cell ordinals must match SlotMap
 * (@c aggr_cell_list). Otherwise use channel-arrival @p cell_ul_list.
 * The store/replay vs production selection stays inside this helper so
 * @c l1_enqueue_phy_work call sites stay free of @c #ifdef noise.
 *
 * Named to match the legacy @c getNextOrderEntity style in this path.
 *
 * @param[in] pdctx            Driver context used to allocate/lookup the entity.
 * @param[in] slot_map_ul      UL SlotMap; used under store/replay for SlotMap order.
 * @param[in] cell_ul_list     Channel-arrival UL cell phy-id list (non-store/replay).
 * @param[in] cell_ul_list_idx Number of valid entries in @p cell_ul_list.
 * @param[in] existing         Existing OrderEntity for this slot, or nullptr.
 * @param[in] create_new       If true, request a new OrderEntity when needed.
 * @return Pointer to OrderEntity, or nullptr on allocation/lookup failure.
 *         Return value must be checked.
 */
[[nodiscard]] OrderEntity* getNextUlOrderEntity(PhyDriverCtx* const pdctx,
                                               [[maybe_unused]] SlotMapUl* const slot_map_ul,
                                               [[maybe_unused]] int32_t* const cell_ul_list,
                                               [[maybe_unused]] const uint8_t cell_ul_list_idx,
                                               OrderEntity* const existing,
                                               const bool create_new)
{
#ifdef ENABLE_FAPI_STORE_REPLAY
    std::array<int32_t, UL_MAX_CELLS_PER_SLOT> order_cell_list{};
    const auto order_cell_count =
        static_cast<uint8_t>(slot_map_ul->aggr_cell_list.size());
    for (uint8_t order_cell_idx = 0; order_cell_idx < order_cell_count; ++order_cell_idx)
    {
        order_cell_list[order_cell_idx] =
            slot_map_ul->aggr_cell_list[order_cell_idx]->getPhyId();
    }
    return pdctx->getNextOrderEntity(
        order_cell_list.data(), order_cell_count, existing, create_new);
#else
    return pdctx->getNextOrderEntity(
        cell_ul_list, cell_ul_list_idx, existing, create_new);
#endif
}

} // namespace

/**
 * @brief Enqueue PHY work for the current slot.
 *
 * Schedules UL and DL task pipelines for all cells in @p sc. The @p skip_mask controls
 * which task categories are excluded — see EnqueueSkipMask in cuphydriver_api.hpp for the
 * full contract and extensibility notes.
 *
 * Six sites inside this function are gated by @p skip_mask:
 *  1. @c dl_task_count budget (non-mMIMO path) — C-plane task slots omitted when
 *     @c SKIP_DL_CPLANE is set.
 *  2. @c dl_task_count budget (mMIMO path) — same omission.
 *  3. @c TaskDL1AggrCplane / @c TaskDL2AggrPrepare enqueue loop — skipped entirely when
 *     @c SKIP_DL_CPLANE is set.
 *  4. @c ul_task_count budget (orderKernel path) — ULC task slots omitted when
 *     @c SKIP_UL_CPLANE is set.
 *  5. @c ul_task_count budget (non-orderKernel paths) — same omission.
 *  6. @c TaskUL1AggrCplane enqueue loop — skipped entirely when @c SKIP_UL_CPLANE is set.
 *
 * @param pdh       cuPHYDriver handler; must be non-null.
 * @param sc        Slot command containing UL/DL channel parameters for all cells.
 * @param skip_mask Bitmask of categories to exclude; see EnqueueSkipMask. Callers that
 *                  use the default (SKIP_NONE) receive the original full-pipeline behaviour.
 * @param use_bound_early_maps   (ENABLE_FAPI_STORE_REPLAY only) When true, use the
 *                  caller-supplied early slot maps below instead of the driver-context
 *                  early maps (getEarlySlotMapDl / getEarlySlotMapUl).
 * @param bound_early_slot_map_dl (ENABLE_FAPI_STORE_REPLAY only) Caller-bound early DL slot
 *                  map; consulted iff use_bound_early_maps. May be null.
 * @param bound_early_slot_map_ul (ENABLE_FAPI_STORE_REPLAY only) Caller-bound early UL slot
 *                  map; consulted iff use_bound_early_maps. May be null.
 *
 * @return 0 on success, non-zero on error; return value must be checked.
 */
[[nodiscard]] int l1_enqueue_phy_work(phydriver_handle pdh,
                                      slot_command_api::slot_command* sc,
                                      EnqueueSkipMask skip_mask
#ifdef ENABLE_FAPI_STORE_REPLAY
                                      ,
                                      bool use_bound_early_maps,
                                      SlotMapDl* bound_early_slot_map_dl,
                                      SlotMapUl* bound_early_slot_map_ul
#endif
                                      )
{
    PhyDriverCtx*                           pdctx       = nullptr;
    SlotMapUl*                              slot_map_ul = nullptr;
    SlotMapDl*                              slot_map_dl = nullptr;
    int                                     tentative = 0, mu = 0, num_cells = 0, first_cell = 0, task_index = 0, max_ul_uc_delay = 0, min_slot_ahead = 100,dl_task_count=0,ul_task_count=0;
    bool pucch_or_pusch_found = false;
    std::array<t_ns, TASK_MAX_PER_SLOT + 1> task_ts_enq;
    t_ns                                    waitns(10 * 1000);
    t_ns                                    t0, t1, t2, t3, t4, t5, t6;
    t_ns                                    start_task;
    struct slot_params*                     current_slot_params = nullptr;
    struct slot_params_aggr*                current_slot_params_aggr = nullptr;
    bool                                    ulbuffer_st1_needed = true;
    std::array<ULInputBuffer*, PRACH_MAX_OCCASIONS> ulbuf_st3_v = {nullptr};
    int rach_occasion = 0;
    slot_command_api::slot_info& slot {sc->cell_groups.slot};
    bool order_entity_set=false;
    /*
     * Here we assume L1 supports only homogeneous cells with MU = 1
     */
    ULInputBuffer * ulbuf_st1    = nullptr;
    ULInputBuffer * ulbuf_st2    = nullptr;
    ULInputBuffer * ulbuf_pcap_capture    = nullptr;
    ULInputBuffer * ulbuf_pcap_capture_ts  = nullptr;
    DLOutputBuffer* dlbuf        = nullptr;
    OrderEntity*    oentity_ptr  = nullptr;
    OrderEntity*    oentity_ptr_tmp  = nullptr;

    PhyPuschAggr*   aggr_pusch_ptr = nullptr;
    PhyPucchAggr*   aggr_pucch_ptr = nullptr;
    PhyPrachAggr*   aggr_prach_ptr = nullptr;
    PhySrsAggr*   aggr_srs_ptr   = nullptr;
    PhyPdschAggr*   aggr_pdsch_ptr = nullptr;
    PhyPdcchAggr*   aggr_pdcch_dl_ptr = nullptr;
    PhyPdcchAggr*   aggr_pdcch_ul_ptr = nullptr;
    PhyPbchAggr*    aggr_pbch_ptr = nullptr;
    PhyDlBfwAggr*   aggr_dlbfw_ptr=nullptr;
    PhyUlBfwAggr*   aggr_ulbfw_ptr=nullptr;
    PhyCsiRsAggr*   aggr_csirs_ptr = nullptr;
    int32_t cell_dl_list[DL_MAX_CELLS_PER_SLOT];
    uint32_t cell_dl_list_idx = 0, tmpdl = 0;
    int32_t cell_ul_list[UL_MAX_CELLS_PER_SLOT];
    uint32_t cell_ul_list_idx = 0, tmpul = 0 ;
    std::vector<int32_t> * phy_cell_index_list;
    std::vector<int32_t> * cell_index_list;
    bool isAggrObjAvail=true;
    bool isUlDlBufAvail=true;
    bool isOKobjAvail=true;
    bool en_orderKernel_tb=false;
    ru_type ru_type_for_srs_proc = OTHER_MODE;


    if(pdh == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "l1_enqueue_phy_work returned error for pdh == nullptr");
        return EINVAL;
    }

    t0 = Time::nowNs();

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        if(pdctx->isActive() == false)
        {
            NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "This cuPHYDriver context is not active, can't enqueue any work");
            return -1;
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    aggr_obj_error_info_t* errorInfoDl = pdctx->getAggrObjErrInfo(true);
    aggr_obj_error_info_t* errorInfoUl = pdctx->getAggrObjErrInfo(false);

    const bool direct_cplane = pdctx->isFapiToCplaneDirect();
#ifdef ENABLE_FAPI_STORE_REPLAY
    const bool dl_from_early = direct_cplane &&
        (use_bound_early_maps ? bound_early_slot_map_dl != nullptr : pdctx->getEarlySlotMapDl() != nullptr);
    const bool ul_from_early = direct_cplane &&
        (use_bound_early_maps ? bound_early_slot_map_ul != nullptr : pdctx->getEarlySlotMapUl() != nullptr);
#else
    const bool dl_from_early = direct_cplane && pdctx->getEarlySlotMapDl() != nullptr;
    const bool ul_from_early = direct_cplane && pdctx->getEarlySlotMapUl() != nullptr;
#endif

    min_slot_ahead = pdctx->get_slot_advance();
    en_orderKernel_tb = pdctx->enableOKTb();

    t1 = Time::nowNs();

    // t3 = Time::nowNs();

    for (auto &cell : sc->cells)
    {
        if (slot.type != slot_command_api::slot_type::SLOT_NONE)
            break;
        else
            slot = cell.slot;
    }

    if (slot.type != slot_command_api::slot_type::SLOT_NONE)
    {
        // Clean-up sc->tick_original if L2 Adapter screwed it up
        uint64_t t0_slot = sfn_to_tai(slot.slot_3gpp.sfn_, slot.slot_3gpp.slot_, sc->tick_original.count() + AppConfig::getInstance().getTaiOffset(), (int64_t)pdctx->get_gps_alpha(),pdctx->get_gps_beta(), 1) - AppConfig::getInstance().getTaiOffset();
        uint64_t correct_tick = t0_slot - ((min_slot_ahead) * Cell::getTtiNsFromMu(MU_SUPPORTED));
        slot.slot_3gpp.t0_ = t0_slot;
        slot.slot_3gpp.t0_valid_ = true;

        NVLOGD_FMT(TAG,"SFN {}.{} L2A tick {} correct tick {} error {}, gps_alpha={}, gps_beta={}",
               slot.slot_3gpp.sfn_,
               slot.slot_3gpp.slot_,
               sc->tick_original.count(),
               correct_tick,
               sc->tick_original.count()-correct_tick,
               pdctx->get_gps_alpha(),
               pdctx->get_gps_beta());

        if (correct_tick != sc->tick_original.count())
        {
            sc->tick_original = t_ns(correct_tick);
        }
    }
    NVLOGD_FMT(TAG, "[LEGACY_TICK] sfn={} slot={} tick_original={}", slot.slot_3gpp.sfn_, slot.slot_3gpp.slot_, sc->tick_original.count());

    uint64_t t0_slot = slot.slot_3gpp.t0_valid_ ? slot.slot_3gpp.t0_ : (sc->tick_original + t_ns(min_slot_ahead * Cell::getTtiNsFromMu(MU_SUPPORTED))).count();

    ////////////////////////////////////////////////////////////////////////////
    //// Split cells by direction UL/DL
    ////////////////////////////////////////////////////////////////////////////
    {
        CuphyOAM *oam = CuphyOAM::getInstance();
        // FIXME: HACK for the H5Dump ACK
        if(oam->puschH5dumpInProgress.load())
        {
            static int drop_count = 0;
            NVLOGI_FMT(TAG, "SFN {}.{} Drop slot command due to H5Dump Mechanism (Total dropped this run {})", static_cast<unsigned>(sc->cells[0].slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cells[0].slot.slot_3gpp.slot_), drop_count++);
            goto cleanup_err;
        }
        current_slot_params_aggr = pdctx->getNextSlotCmd(); //new slot_params_aggr(sc->cell_groups.slot.slot_3gpp, &sc->cell_groups);
        current_slot_params_aggr->populate(&(sc->cell_groups.slot.slot_3gpp), &(sc->cell_groups));
        for (uint8_t i = 0; i < sc->cell_groups.channel_array_size; i++)
        // for(auto& ch : sc->cell_groups.channels)
        {
            auto ch = sc->cell_groups.channels[i];
            /* Generic DL items */
            if(ch == slot_command_api::PDSCH || ch == slot_command_api::PDCCH_DL || ch == slot_command_api::PDCCH_UL || ch == slot_command_api::PBCH || ch == slot_command_api::CSI_RS || (ch == slot_command_api::BFW && current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::DL_BFW))
            {
                if(slot_map_dl == nullptr)
                {
                    if (dl_from_early) {
#ifdef ENABLE_FAPI_STORE_REPLAY
                        slot_map_dl = use_bound_early_maps ? bound_early_slot_map_dl : pdctx->getEarlySlotMapDl();
                        if (!use_bound_early_maps)
                        {
                            pdctx->setEarlySlotMapDl(nullptr);
                        }
#else
                        slot_map_dl = pdctx->getEarlySlotMapDl();
                        pdctx->setEarlySlotMapDl(nullptr);
#endif
                        slot_map_dl->setDynBeamIdOffset(pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot());
                    } else {
                        slot_map_dl = pdctx->getNextSlotMapDl();
                    }
                    if(slot_map_dl == nullptr)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} SlotMap DL error", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} direction DL at {}", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_dl->getId(), Time::nowNs().count());

                    if (!dl_from_early) {
                        slot_map_dl->setSlot3GPP(*current_slot_params_aggr->si);
                        auto dyn_beam_id_offset = pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot();
                        slot_map_dl->setDynBeamIdOffset(dyn_beam_id_offset);
                    }
                }

                phy_cell_index_list = nullptr;
                cell_index_list = nullptr;
                if(ch == slot_command_api::PDSCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pdsch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pdsch->cell_index_list);
                    //Enable this code snippet to Start/Stop profiling in the middle of a test run

                    // if (3==(slot_map_dl->getId()))    {
                    //     NVLOGC_FMT(TAG, "Starting profiler");
                    //     cudaProfilerStart();
                    // }
                    // if (10==(slot_map_dl->getId())) {
                    //     NVLOGC_FMT(TAG, "Stopping profiler");
                    //     cudaProfilerStop();
                    // }
#if 0 
                     if ((sc->cell_groups.slot.slot_3gpp.sfn_ == 0) && (sc->cell_groups.slot.slot_3gpp.slot_ == 7)) {
                        NVLOGC(TAG, "Starting profiler");
                        CUDA_DRIVER_CHECK(cuProfilerStart());
                    }
                     if ((sc->cell_groups.slot.slot_3gpp.sfn_ == 1) && (sc->cell_groups.slot.slot_3gpp.slot_ == 2)) {
                        NVLOGC(TAG, "Stopping profiler");
                        CUDA_DRIVER_CHECK(cuProfilerStop());
                    }
 
#endif

                }
                else if(ch == slot_command_api::PDCCH_DL || ch == slot_command_api::PDCCH_UL)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pdcch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pdcch->cell_index_list);
                }
                else if(ch == slot_command_api::PBCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pbch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pbch->cell_index_list);
                }
                else if(ch == slot_command_api::CSI_RS)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->csirs->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->csirs->cell_index_list);
                }

                if(ch != slot_command_api::BFW)
                {
                    int cell_index=0;
                    for(auto& cell_phy_id : *phy_cell_index_list)
                    {
                        // auto findptr = std::find(std::begin(cell_dl_list), std::end(cell_dl_list), (int) cell_phy_id);
                        // if (findptr != std::end(cell_dl_list)) {
                        //     cell_index++;
                        //     continue;
                        // }

                        for (tmpdl=0; tmpdl < cell_dl_list_idx; tmpdl++) {
                            if (cell_dl_list[tmpdl] == (int) cell_phy_id)
                                break;
                        }

                        if (tmpdl < cell_dl_list_idx) {
                            cell_index++;
                            continue;
                        }

                        dlbuf = nullptr;

                        Cell* cell_ptr = pdctx->getCellByPhyId(cell_phy_id);
                        if(cell_ptr == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} is not present in the PhyDriver context", cell_phy_id);
                            cell_index++;
                            continue;
                        }

                        if(cell_ptr->isActive() == false)
                        {
                            NVLOGW_FMT(TAG, "Cell {} is not active, can't run anything", cell_ptr->getPhyId());
                            cell_index++;
                            continue;
                        }

                        // Assume homogeneous cells
                        mu = cell_ptr->getMu();

                        if (dl_from_early) {
                            for (int j = 0, n = static_cast<int>(slot_map_dl->aggr_cell_list.size()); j < n; ++j) {
                                if (slot_map_dl->aggr_cell_list[j] == cell_ptr) {
                                    slot_map_dl->aggr_slot_info[j] =
                                        sc->cells[(*cell_index_list)[cell_index]].params.sym_prb_info.get();
                                    break;
                                }
                            }
                        } else {
                            dlbuf = cell_ptr->getNextDlBuffer();
                            if(dlbuf == nullptr)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available DL output buffers for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                isUlDlBufAvail=false;
                                goto cleanup_err;
                            }
                            else
                                NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} DLBuffer {} at {}",
                                            static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_),
                                            slot_map_dl->getId(), cell_ptr->getPhyId(), mu, dlbuf->getId(), Time::nowNs().count());
                            if (slot_map_dl->aggrSetCells(cell_ptr, &sc->cells[(*cell_index_list)[cell_index]].params, dlbuf)) {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMap DL can't set another cell");
                                goto cleanup_err;
                            }
                        }

                        // Avoid to waste UL thread time starting it too early
                        if(max_ul_uc_delay < cell_ptr->getT1aMaxCpUlNs())
                            max_ul_uc_delay = cell_ptr->getT1aMaxCpUlNs();

                        // cell_dl_list.push_back(cell_phy_id);
                        cell_dl_list[cell_dl_list_idx] = cell_phy_id;
                        cell_dl_list_idx++;
                        cell_index++;
                    }
                }
            }

            /* Generic UL items */
            if(ch == slot_command_api::PUSCH ||ch == slot_command_api::PUCCH || ch == slot_command_api::PRACH || ch == slot_command_api::SRS || (ch == slot_command_api::BFW && current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::UL_BFW))
            {
                if(slot_map_ul == nullptr)
                {
                    if (ul_from_early) {
#ifdef ENABLE_FAPI_STORE_REPLAY
                        slot_map_ul = use_bound_early_maps ? bound_early_slot_map_ul : pdctx->getEarlySlotMapUl();
                        if (!use_bound_early_maps)
                        {
                            pdctx->setEarlySlotMapUl(nullptr);
                        }
#else
                        slot_map_ul = pdctx->getEarlySlotMapUl();
                        pdctx->setEarlySlotMapUl(nullptr);
#endif
                        slot_map_ul->setDynBeamIdOffset(pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot());
                    } else {
                        slot_map_ul = pdctx->getNextSlotMapUl();
                    }
                    if(slot_map_ul == nullptr)
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} SlotMap UL error", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} direction UL at {}", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), Time::nowNs().count());

                    if (!ul_from_early) {
                        auto dyn_beam_id_offset = pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot();
                        slot_map_ul->setDynBeamIdOffset(dyn_beam_id_offset);
                    }
                }

                if (!ul_from_early) {
                    slot_map_ul->setSlot3GPP(*current_slot_params_aggr->si);
                }

                int cell_index=0;
                /*By default */
                phy_cell_index_list = nullptr;
                cell_index_list = nullptr;
                if(ch == slot_command_api::PUSCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pusch->cell_index_list);
                }
                else if(ch == slot_command_api::PUCCH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->pucch->cell_index_list);
                }
                else if(ch == slot_command_api::PRACH)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->prach->cell_index_list);
                }
                else if(ch == slot_command_api::SRS)
                {
                    phy_cell_index_list = &(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list);
                    cell_index_list = &(current_slot_params_aggr->cgcmd->srs->cell_index_list);
                }

                if(ch != slot_command_api::BFW)
                {
                    bool ru_type_found = false;
                    /* Check if all the cells are valid */
                    for(auto& cell_phy_id : *phy_cell_index_list)
                    {
                        for(tmpul = 0; tmpul < cell_ul_list_idx; tmpul++)
                        {
                            if(cell_ul_list[tmpul] == (int) cell_phy_id)
                            {
                                break;
                            }
                        }
                        if (tmpul < cell_ul_list_idx)
                        {
                            cell_index++;
                            continue;
                        }

                        oentity_ptr = nullptr;
                        // ulbuf_st3_v.clear();
                        rach_occasion = 0;

                        Cell* cell_ptr = pdctx->getCellByPhyId(cell_phy_id);
                        //NVLOGC_FMT(TAG, "Cell phy id = {}, cell index = {}", cell_phy_id, (*cell_index_list)[cell_index]);
                        if(cell_ptr == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell {} is not present in the PhyDriver context", cell_phy_id);
                            cell_index++;
                            continue;
                        }

                        if(ru_type_found == false)
                        {
                            ru_type_for_srs_proc = cell_ptr->getRUType();
                            pdctx->set_ru_type_for_srs_proc(ru_type_for_srs_proc);
                            ru_type_found = true;
                        }

                        if(cell_ptr->isActive() == false)
                        {
                            NVLOGW_FMT(TAG, "Cell {} is not active, can't run anything", cell_ptr->getPhyId());
                            cell_index++;
                            continue;
                        }

                        // Assume homogeneous cells
                        mu = cell_ptr->getMu();

                        if(ch == slot_command_api::PUSCH ||ch == slot_command_api::PUCCH)
                        {
                            ulbuf_st1 = nullptr;
                            ulbuf_st2 = nullptr;
                            cell_ptr->setPuschDynPrmIndex(sc->cell_groups.slot.slot_3gpp.slot_, -1);
                            // ulbuf_st3_v.clear();
                            rach_occasion = 0;
                            ulbuf_st1 = cell_ptr->getNextUlBufferST1();
                            if(ulbuf_st1 == nullptr)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st1 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                isUlDlBufAvail=false;
                                goto cleanup_err;
                            }
                            else
                                NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST1 {} at {} ch = PUSCH|PUCCH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st1->getId(), Time::nowNs().count());

                            /* Is the same cell there also for PRACH? */
                            if(current_slot_params_aggr->cgcmd->prach) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list)) {
                                    // ulbuf_st3_v.clear();
                                    rach_occasion = 0;
                                   // if(current_slot_params_aggr->cgcmd->prach->rach.size() > PRACH_MAX_OCCASIONS)
                                    if(current_slot_params_aggr->cgcmd->prach->nOccasion > PRACH_MAX_OCCASIONS)
    				                {
                                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Too many RACH objects (occasions) {} for this cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, current_slot_params_aggr->cgcmd->prach->nOccasion, cell_ptr->getPhyId());
                                        goto cleanup_err;
                                    }

                                    rach_occasion = cell_ptr->getPrachOccaSize();
                                    //NVLOGI_FMT(TAG,"NUmber of RO per cell = {}", rach_occasion);

                                    //Create UL buffer entries for each rach occasion
                                    for(int ro = 0; ro < rach_occasion; ro++)
                                    {
                                        ULInputBuffer * ulbuf_st3 = cell_ptr->getNextUlBufferST3();
                                        if(ulbuf_st3 == nullptr)
                                        {
                                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st3 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                            isUlDlBufAvail=false;
                                            goto cleanup_err;
                                        }
                                        else
                                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST3 {} at {} ch = PRACH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st3->getId(), Time::nowNs().count());

                                        // ulbuf_st3_v.push_back(ulbuf_st3);
                                        ulbuf_st3_v[ro] = ulbuf_st3;
                                    }
                                }
                            }
                            if( current_slot_params_aggr->cgcmd->srs) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list)) {
                                    ulbuf_st2 = nullptr;
                                    ulbuf_st2 = cell_ptr->getNextUlBufferST2();
                                    if(ulbuf_st2 == nullptr)
                                    {
                                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st2 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                        isUlDlBufAvail=false;
                                        goto cleanup_err;
                                    }
                                    else
                                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST2 {} at {} ch = SRS", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st2->getId(), Time::nowNs().count());
                                    }
                            }
                        }
                        else if(ch == slot_command_api::PRACH)
                        {
                            // ulbuf_st3_v.clear();
                            rach_occasion = 0;
                            //if(current_slot_params_aggr->cgcmd->prach->rach.size() > PRACH_MAX_OCCASIONS)
                            if(current_slot_params_aggr->cgcmd->prach->nOccasion > PRACH_MAX_OCCASIONS)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Too many RACH objects (occasions) {} for this cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, current_slot_params_aggr->cgcmd->prach->nOccasion, cell_ptr->getPhyId());
                                goto cleanup_err;
                            }

                            rach_occasion = cell_ptr->getPrachOccaSize();
                            //NVLOGC_FMT(TAG,"NUmber of RO per cell  {} with PHY ID {} = {}", cell_index, cell_phy_id, rach_occasion);

                            //Create UL buffer entries for each rach occasion
                            for(int ro = 0; ro < rach_occasion; ro++)
                            {
                                ULInputBuffer * ulbuf_st3 = cell_ptr->getNextUlBufferST3();
                                if(ulbuf_st3 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st3 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST3 {} at {} ch = PRACH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st3->getId(), Time::nowNs().count());

                                // ulbuf_st3_v.push_back(ulbuf_st3);
                                ulbuf_st3_v[ro] = ulbuf_st3;
                            }

                            bool is_pusch = false;
                            bool is_pucch = false;
                            bool is_srs = false;
                            ulbuf_st1 = nullptr;
                            ulbuf_st2 = nullptr;

                            /* Is the same cell there also for PUSCH? */
                            if(current_slot_params_aggr->cgcmd->pusch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list)) {
                                    is_pusch = true;
                                }
                            }

                            /* Is the same cell there also for PUSCH? */
                            if(is_pusch == false && current_slot_params_aggr->cgcmd->pucch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list)) {
                                    is_pucch = true;
                                }
                            }

                            if(current_slot_params_aggr->cgcmd->srs) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->srs->phy_cell_index_list)) {
                                    is_srs = true;
                                }
                            }
                            /* Get ULBuffer ST1 */
                            if(is_pusch == true || is_pucch == true ) {
                                ulbuf_st1 = nullptr;
                                ulbuf_st1 = cell_ptr->getNextUlBufferST1();
                                if(ulbuf_st1 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st1 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST1 {} at {} ch = PUSCH|PUCCH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st1->getId(), Time::nowNs().count());
                            }

                            if(is_srs == true){
                                ulbuf_st2 = nullptr;
                                ulbuf_st2 = cell_ptr->getNextUlBufferST2();
                                if(ulbuf_st2 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st2 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST2 {} at {} ch = SRS", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st2->getId(), Time::nowNs().count());
                            }
                        }
                        else if (ch == slot_command_api::SRS){
                            ulbuf_st2 = nullptr;
                            ulbuf_st2 = cell_ptr->getNextUlBufferST2();
                            if(ulbuf_st2 == nullptr)
                            {
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st2 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                isUlDlBufAvail=false;
                                goto cleanup_err;
                            }
                            else
                                NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST2 {} at {} ch = SRS", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st2->getId(), Time::nowNs().count());
                            /* Is the same cell there also for PRACH? */
                            if(current_slot_params_aggr->cgcmd->prach) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->prach->phy_cell_index_list)) {
                                    // ulbuf_st3_v.clear();
                                    rach_occasion = 0;
                                   // if(current_slot_params_aggr->cgcmd->prach->rach.size() > PRACH_MAX_OCCASIONS)
                                    if(current_slot_params_aggr->cgcmd->prach->nOccasion > PRACH_MAX_OCCASIONS)
                                    {
                                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Too many RACH objects (occasions) {} for this cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, current_slot_params_aggr->cgcmd->prach->nOccasion, cell_ptr->getPhyId());
                                        goto cleanup_err;
                                    }

                                    rach_occasion = cell_ptr->getPrachOccaSize();
                                    //NVLOGI_FMT(TAG,"NUmber of RO per cell = {}", rach_occasion);

                                    //Create UL buffer entries for each rach occasion
                                    for(int ro = 0; ro < rach_occasion; ro++)
                                    {
                                        ULInputBuffer * ulbuf_st3 = cell_ptr->getNextUlBufferST3();
                                        if(ulbuf_st3 == nullptr)
                                        {
                                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st3 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                            isUlDlBufAvail=false;
                                            goto cleanup_err;
                                        }
                                        else
                                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST3 {} at {} ch = PRACH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st3->getId(), Time::nowNs().count());

                                        // ulbuf_st3_v.push_back(ulbuf_st3);
                                        ulbuf_st3_v[ro] = ulbuf_st3;
                                    }
                                }
                            }

                            bool is_pusch = false;
                            bool is_pucch = false;
                            ulbuf_st1 = nullptr;
                            /* Is the same cell there also for PUSCH? */
                            if(current_slot_params_aggr->cgcmd->pusch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pusch->phy_cell_index_list)) {
                                    is_pusch = true;
                                }
                            }

                            /* Is the same cell there also for PUSCH? */
                            if(is_pusch == false && current_slot_params_aggr->cgcmd->pucch) {
                                auto findcell = std::find(std::begin(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list), (int) cell_phy_id);
                                if (findcell != std::end(current_slot_params_aggr->cgcmd->pucch->phy_cell_index_list)) {
                                    is_pucch = true;
                                }
                            }
                            /* Get ULBuffer ST1 */
                            if(is_pusch == true || is_pucch == true ) {
                                ulbuf_st1 = nullptr;
                                ulbuf_st1 = cell_ptr->getNextUlBufferST1();
                                if(ulbuf_st1 == nullptr)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available UL input buffers st1 for cell {}", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, cell_ptr->getPhyId());
                                    isUlDlBufAvail=false;
                                    goto cleanup_err;
                                }
                                else
                                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} Cell {} with MU {} ULBuffer ST1 {} at {} ch = PUSCH|PUCCH", static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.sfn_), static_cast<unsigned>(sc->cell_groups.slot.slot_3gpp.slot_), slot_map_ul->getId(), cell_ptr->getPhyId(), mu, ulbuf_st1->getId(), Time::nowNs().count());
                            }
                        }

                        ulbuf_pcap_capture = nullptr;
                        ulbuf_pcap_capture_ts = nullptr;
                        if(pdctx->get_ul_pcap_capture_enable())
                        {
                            ulbuf_pcap_capture = cell_ptr->getUlBufferPcap();
                            ulbuf_pcap_capture_ts = cell_ptr->getUlBufferPcapTs();
                        }
                        if (ul_from_early) {
                            for (int j = 0, n = static_cast<int>(slot_map_ul->aggr_cell_list.size()); j < n; ++j) {
                                if (slot_map_ul->aggr_cell_list[j] == cell_ptr) {
                                    slot_map_ul->aggr_slot_info[j] =
                                        sc->cells[(*cell_index_list)[cell_index]].params.sym_prb_info.get();
                                    slot_map_ul->aggr_ulbuf_st1[j] = ulbuf_st1;
                                    slot_map_ul->aggr_ulbuf_st2[j] = ulbuf_st2;
                                    slot_map_ul->aggr_ulbuf_pcap_capture[j] = ulbuf_pcap_capture;
                                    slot_map_ul->aggr_ulbuf_pcap_capture_ts[j] = ulbuf_pcap_capture_ts;
                                    slot_map_ul->num_prach_occa[j] = rach_occasion;
                                    for (int ro = 0; ro < rach_occasion; ++ro) {
                                        slot_map_ul->aggr_ulbuf_st3.push_back(ulbuf_st3_v[ro]);
                                    }
                                    break;
                                }
                            }
                        } else if (slot_map_ul->aggrSetCells(cell_ptr, &sc->cells[(*cell_index_list)[cell_index]].params,
                                                    ulbuf_st1,ulbuf_st2,ulbuf_st3_v, rach_occasion, ulbuf_pcap_capture, ulbuf_pcap_capture_ts)) {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SlotMap UL can't set another cell");
                            goto cleanup_err;
                        }


                        // Avoid to waste UL thread time starting it too early
                        if(max_ul_uc_delay < cell_ptr->getT1aMaxCpUlNs())
                            max_ul_uc_delay = cell_ptr->getT1aMaxCpUlNs();
                        cell_ul_list[cell_ul_list_idx] = cell_phy_id;
                        cell_ul_list_idx++;
                        cell_index++;
                    }
                    //Single order kernel entity per slot
                    if(!order_entity_set)
                    {
                        oentity_ptr = getNextUlOrderEntity(
                            pdctx, slot_map_ul, &cell_ul_list[0], cell_ul_list_idx,
                            nullptr, true);
                        if(oentity_ptr == nullptr)
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No available Order Kernel object for SFN {}.{} Map {} ", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId());
                            isOKobjAvail=false;
                            goto cleanup_err;
                        }
                        else
                        {
                            NVLOGI_FMT(TAG, "SFN {}.{} Map {} Oentity {} at {} for ch({})", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),oentity_ptr->getId(), Time::nowNs().count(),(int)ch);
                            slot_map_ul->aggrSetOrderEntity(oentity_ptr);
                            order_entity_set=true;
                        }
                    }
                    else
                    {
                        oentity_ptr = getNextUlOrderEntity(
                            pdctx, slot_map_ul, &cell_ul_list[0], cell_ul_list_idx,
                            slot_map_ul->aggrGetOrderEntity(), false);
                        if(oentity_ptr!=slot_map_ul->aggrGetOrderEntity())
                        {
                            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Order entity cannot be different for the same slot");
                            goto cleanup_err;
                        }
                    }
                }
            }

            /* Uplink */
            if(ch == slot_command_api::BFW)
            {
                if(current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::UL_BFW)
                {
                    aggr_ulbfw_ptr = pdctx->getNextUlBfwAggr(current_slot_params_aggr);
                    if(aggr_ulbfw_ptr == nullptr)
                    {
                        NVLOGI_FMT(TAG, "No available Aggr ULBFW objects");
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got ULBFW Aggr obj {:x} at {}",
                                sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                                mu, aggr_ulbfw_ptr->getId(), Time::nowNs().count()
                            );
                    ul_task_count+=2; //ULBFW + AGGR3_ULBFW
                }
            }

            if(ch == slot_command_api::PUSCH || ch == slot_command_api::PUCCH) {
                if(!pucch_or_pusch_found) {
                    pucch_or_pusch_found = true;
                    ul_task_count += 1;
                }
            }

            if(ch == slot_command_api::PUSCH)
            {
                aggr_pusch_ptr = pdctx->getNextPuschAggr(current_slot_params_aggr);
                if(aggr_pusch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PUSCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PUSCH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_pusch_ptr->getId(), Time::nowNs().count()
                        );
                ul_task_count+=1; //Early UCI IND Task
            }

            if(ch == slot_command_api::PUCCH)
            {
                aggr_pucch_ptr = pdctx->getNextPucchAggr(current_slot_params_aggr);
                if(aggr_pucch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PUCCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PUCCH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_pucch_ptr->getId(), Time::nowNs().count()
                        );
            }

            if(ch == slot_command_api::PRACH)
            {
                aggr_prach_ptr = pdctx->getNextPrachAggr(current_slot_params_aggr);
                if(aggr_prach_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PRACH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PRACH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_prach_ptr->getId(), Time::nowNs().count()
                        );
                ul_task_count++;
            }

            if(ch == slot_command_api::SRS)
            {
                aggr_srs_ptr = pdctx->getNextSrsAggr(current_slot_params_aggr);
                if(aggr_srs_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr SRS objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got SRS Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_ul->getId(),
                            mu, aggr_srs_ptr->getId(), Time::nowNs().count()
                        );
                if(pdctx->get_ru_type_for_srs_proc() == SINGLE_SECT_MODE)
                {
                    ul_task_count+=1; //SRS only
                }
                else
                {
                    ul_task_count+=2; //SRS + AGGR3_SRS
                }
            }

            /* Downlink */

            if(ch == slot_command_api::BFW)
            {
                if(current_slot_params_aggr->cgcmd->get_bfw_params()->bfw_cvi_type==slot_command_api::DL_BFW)
                {
                    aggr_dlbfw_ptr = pdctx->getNextDlBfwAggr(current_slot_params_aggr);
                    if(aggr_dlbfw_ptr == nullptr)
                    {
                        NVLOGI_FMT(TAG, "No available Aggr DLBFW objects");
                        goto cleanup_err;
                    }
                    else
                        NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got DLBFW Aggr obj {:x} at {}",
                                sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                                mu, aggr_dlbfw_ptr->getId(), Time::nowNs().count()
                            );
                    dl_task_count++;
                }
            }
            else if(ch == slot_command_api::PDSCH)
            {
                aggr_pdsch_ptr = pdctx->getNextPdschAggr(current_slot_params_aggr);
                if(aggr_pdsch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PDSCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got PDSCH Aggr obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pdsch_ptr->getId(), Time::nowNs().count()
                        );
                dl_task_count++;
                // When prepone TB H2D is enabled and the dedicated H2D copy thread is off, this
                // block runs the same sequence as l1_launch_tb_h2d (setCtx, start event,
                // performBatchedMemcpy, reset batches, complete event) on the slot-command path.
                //
                // isBatchLaunchedForSlot() is true when l1_launch_tb_h2d (split-phase path)
                // already executed the memcpy and recorded events for this slot — skip to
                // avoid double-submitting the DMA and overwriting the events.
                {
                    auto* h2dMgr = pdctx->getH2DCopyManager();
                    const uint8_t slot = sc->cell_groups.slot.slot_3gpp.slot_;
                    if(h2dMgr->isPreponeEnabled() && !h2dMgr->isBatchLaunchedForSlot(slot))
                    {
                        if(!h2dMgr->isThreadEnabled())
                        {
                            h2dMgr->setCtx();

                            CUDA_CHECK(cudaEventRecord(h2dMgr->getStartEvent(slot), h2dMgr->getStream()));

                            cuphyStatus_t batched_memcpy_status = h2dMgr->performBatchedMemcpy();
                            h2dMgr->resetBatchedMemcpyBatches();

                            if (batched_memcpy_status != CUPHY_STATUS_SUCCESS) [[unlikely]]
                            {
                                // Do NOT record the complete event — a downstream
                                // cudaStreamWaitEvent would otherwise proceed even
                                // though the DMA failed (silent data corruption).
                                // Matches l1_launch_tb_h2d's failure handling.
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                                           "l1_enqueue_phy_work inline H2D: performBatchedMemcpy failed st={} slot={} - "
                                           "complete event NOT recorded; PDSCH will time out",
                                           static_cast<int>(batched_memcpy_status), slot);
                            }
                            else
                            {
                                CUDA_CHECK(cudaEventRecord(h2dMgr->getCompleteEvent(slot), h2dMgr->getStream()));
                            }
                        }
                        else
                        {
                            l1_set_h2d_copy_done_cur_slot_flag(pdh,(int)slot);
                        }
                    }
                    else if(h2dMgr->isBatchLaunchedForSlot(slot))
                    {
                        h2dMgr->clearBatchLaunched(slot);
                    }
                }
            }
            else if(ch == slot_command_api::PDCCH_DL)
            {
                aggr_pdcch_dl_ptr = pdctx->getNextPdcchDlAggr(current_slot_params_aggr);
                if(aggr_pdcch_dl_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PDCCH DL objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr PDCCH DL obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pdcch_dl_ptr->getId(), Time::nowNs().count()
                        );
            }
            else if(ch == slot_command_api::PDCCH_UL)
            {
                aggr_pdcch_ul_ptr = pdctx->getNextPdcchUlAggr(current_slot_params_aggr);
                if(aggr_pdcch_ul_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PDCCH UL objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr PDCCH UL obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pdcch_ul_ptr->getId(), Time::nowNs().count()
                        );
            }
            else if(ch == slot_command_api::PBCH)
            {
                aggr_pbch_ptr = pdctx->getNextPbchAggr(current_slot_params_aggr);
                if(aggr_pbch_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr PBCH objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr PBCH obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_pbch_ptr->getId(), Time::nowNs().count()
                        );
            }
            else if(ch == slot_command_api::CSI_RS)
            {
                aggr_csirs_ptr = pdctx->getNextCsiRsAggr(current_slot_params_aggr);
                if(aggr_csirs_ptr == nullptr)
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} No available Aggr CSI_RS objects", sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_);
                    isAggrObjAvail=false;
                    goto cleanup_err;
                }
                else
                    NVLOGI_FMT(TAG, "SFN {}.{} Map {} with MU {} got Aggr CSI_RS obj {:x} at {}",
                            sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_, slot_map_dl->getId(),
                            mu, aggr_csirs_ptr->getId(), Time::nowNs().count()
                        );
            }

        }

        if (aggr_pdcch_dl_ptr || aggr_pdcch_ul_ptr || aggr_pbch_ptr || aggr_csirs_ptr) {
            dl_task_count++;
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //// CreateMap + CreateTask + EnqueueTask()
    ////////////////////////////////////////////////////////////////////////

    if(slot_map_ul != nullptr)
    {
        const UlAggrPtrs ul_aggr{aggr_pusch_ptr, aggr_pucch_ptr, aggr_prach_ptr, aggr_srs_ptr, aggr_ulbfw_ptr};
        if(enqueue_ul_tasks(pdctx, slot_map_ul, sc, t0_slot, skip_mask,
                            en_orderKernel_tb, ru_type_for_srs_proc,
                            ul_aggr, current_slot_params_aggr, ul_task_count) != 0)
        {
            goto cleanup_err;
        }
    }

    if(slot_map_dl != nullptr)
    {
        const DlAggrPtrs dl_aggr{aggr_pdsch_ptr, aggr_pdcch_dl_ptr, aggr_pdcch_ul_ptr, aggr_pbch_ptr, aggr_csirs_ptr, aggr_dlbfw_ptr};
        if(enqueue_dl_tasks(pdctx, slot_map_dl, sc, t0_slot, skip_mask, dl_from_early,
                            dl_aggr, current_slot_params_aggr, dl_task_count) != 0)
        {
            goto cleanup_err;
        }
    }


    if(slot_map_dl!=nullptr){
        errorInfoDl->prevSlotNonAvail=false;
        errorInfoDl->nonAvailCount=0;
        NVLOGD_FMT(TAG,"[DL Slot] Aggr Objects Available prevSlotNonAvail {},nonAvailCount {}",errorInfoDl->prevSlotNonAvail,errorInfoDl->nonAvailCount);
    }
    if(slot_map_ul!=nullptr){
        errorInfoUl->prevSlotNonAvail=false;
        errorInfoUl->nonAvailCount=0;
        NVLOGD_FMT(TAG,"[UL Slot] Aggr Objects Available prevSlotNonAvail {},nonAvailCount {}",errorInfoUl->prevSlotNonAvail,errorInfoUl->nonAvailCount);
    }

    pdctx->getFhProxy()->updateDynamicBeamIdOffset();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    t6 = Time::nowNs();
    NVLOGI_FMT(TAG, "l1_enqueue_phy_work START: {} END: {} DURATION: {} us", t0.count(), t6.count(), Time::NsToUs(t6 - t0).count());

    return 0;

cleanup_err:
    // NVSLOGE(TAG, AERIAL_CUPHYDRV_API_EVENT) << "Exit error";

    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "[ENQ_ERR] SFN {}.{} cleanup_err: slot_map_ul_id={} slot_map_dl_id={} aggr_pusch={} aggr_pucch={} aggr_prach={} aggr_srs={} aggr_pdsch={} aggr_ulbfw={} isAggrObjAvail={} isUlDlBufAvail={} isOKobjAvail={}",
               sc->cell_groups.slot.slot_3gpp.sfn_, sc->cell_groups.slot.slot_3gpp.slot_,
               slot_map_ul ? slot_map_ul->getId() : -1,
               slot_map_dl ? slot_map_dl->getId() : -1,
               aggr_pusch_ptr != nullptr, aggr_pucch_ptr != nullptr, aggr_prach_ptr != nullptr,
               aggr_srs_ptr != nullptr, aggr_pdsch_ptr != nullptr, aggr_ulbfw_ptr != nullptr,
               isAggrObjAvail, isUlDlBufAvail, isOKobjAvail);

    pdctx->getFhProxy()->updateDynamicBeamIdOffset();

    if(slot_map_ul) slot_map_ul->release(UL_MAX_CELLS_PER_SLOT,false);
    if(oentity_ptr) oentity_ptr->release();
    if(slot_map_dl) slot_map_dl->release(DL_MAX_CELLS_PER_SLOT);
    if(dlbuf) dlbuf->release();
    if(current_slot_params) delete current_slot_params;

    if(ulbuf_st1) ulbuf_st1->release();
    // if(ulbuf_st3_v.size() > 0)
    // {
    //     for(auto& p : ulbuf_st3_v)
    //     {
    //         if(p)
    //         {
    //             p->release();
    //             p = nullptr;
    //         }
    //     }
    // }
    // ulbuf_st3_v.clear();

    if(aggr_pusch_ptr) aggr_pusch_ptr->release();
    if(aggr_pucch_ptr) aggr_pucch_ptr->release();
    if(aggr_prach_ptr) aggr_prach_ptr->release();
    if(aggr_srs_ptr)   aggr_srs_ptr->release();
    if(aggr_pdsch_ptr) aggr_pdsch_ptr->release();

    /*Error handling for Non-availability of DL or UL Aggr Object OR DL or UL Buffers OR Order kernel object for this slot*/
    if(!isAggrObjAvail || !isUlDlBufAvail || !isOKobjAvail){
        if(slot_map_dl!=nullptr)
        {
            errorInfoDl->nonAvailCount++;
            if(errorInfoDl->prevSlotNonAvail){
                if(errorInfoDl->nonAvailCount>pdctx->getAggr_obj_non_avail_th()) //Greater than Yaml configured threshold, Declare fatal error and exit
                {
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"[DL] Successive Non-availability of Aggregated DL objects OR DL Buffers reached Threshold {}, Exiting!!!",pdctx->getAggr_obj_non_avail_th());
                    ENTER_L1_RECOVERY()
                }
            }
            errorInfoDl->prevSlotNonAvail=true;
            NVLOGD_FMT(TAG,"[DL] Aggr Objects OR DL Buffers Non-available prevSlotNonAvail {},nonAvailCount {} isAggrObjAvail {} isUlDlBufAvail {}",errorInfoDl->prevSlotNonAvail,errorInfoDl->nonAvailCount,isAggrObjAvail,isUlDlBufAvail);
            return -2;//Change error code to -2 s.t L2A can handle this error case differently
        }
        if(slot_map_ul!=nullptr)
        {
            errorInfoUl->nonAvailCount++;
            if(errorInfoUl->prevSlotNonAvail){
                if(errorInfoUl->nonAvailCount>pdctx->getAggr_obj_non_avail_th()) //Greater than Yaml configured threshold, Declare fatal error and exit
                {
                    NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"[UL] Successive Non-availability of Aggregated UL objects OR UL Buffers OR Order Kernel Objects reached Threshold {}, Exiting!!!",pdctx->getAggr_obj_non_avail_th());
                    ENTER_L1_RECOVERY()
                }
            }
            errorInfoUl->prevSlotNonAvail=true;
            NVLOGD_FMT(TAG,"[UL] Aggr Objects OR UL Buffers OR Order Kernel Objects Non-available prevSlotNonAvail {},nonAvailCount {} isAggrObjAvail {} isUlDlBufAvail {} isOKobjAvail {}",errorInfoUl->prevSlotNonAvail,errorInfoUl->nonAvailCount,isAggrObjAvail,isUlDlBufAvail,isOKobjAvail);
            return -2;//Change error code to -2 s.t L2A can handle this error case differently
        }
    }

    return -1;
}



namespace {

// RAII guard for TaskList::lock() / TaskList::unlock().
// TaskList::lock/unlock return int, so std::lock_guard<TaskList> does not
// satisfy BasicLockable (which requires void return).  This minimal wrapper
// provides the same exception-safety guarantee without any extra dependencies.
// Precondition: list must be non-null.  Callers (l1_push_task_dl/ul) guarantee
/// RAII lock/unlock guard for TaskList.
/// Callers must guarantee @p list is non-null before constructing this guard:
/// null-check pdctx before calling getTaskListDl/Ul(), and null-check tlist
/// before constructing the guard (see l1_push_task_generic).
struct TaskListLockGuard final
{
    TaskList* list = nullptr;
    explicit TaskListLockGuard(TaskList* l) : list(l) { list->lock(); }
    ~TaskListLockGuard()                               { list->unlock(); }
    TaskListLockGuard(const TaskListLockGuard&)            = delete;
    TaskListLockGuard& operator=(const TaskListLockGuard&) = delete;
    TaskListLockGuard(TaskListLockGuard&&)                 = delete;
    TaskListLockGuard& operator=(TaskListLockGuard&&)      = delete;
};

void l1_push_task_dl(phydriver_handle pdh, Task* task)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_task_dl: null pdctx; task dropped");
        return;
    }
    TaskList* tList = pdctx->getTaskListDl();
    if (tList == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_task_dl: null tList; task dropped");
        return;
    }
    TaskListLockGuard guard(tList);
    tList->push(task);
}

void l1_push_task_ul(phydriver_handle pdh, Task* task)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_task_ul: null pdctx; task dropped");
        return;
    }
    TaskList* tList = pdctx->getTaskListUl();
    if (tList == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_task_ul: null tList; task dropped");
        return;
    }
    TaskListLockGuard guard(tList);
    tList->push(task);
}

void l1_push_task_generic([[maybe_unused]] phydriver_handle pdh, Task* task, TaskList* tlist)
{
    if (tlist == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_task_generic: null tlist; task dropped");
        return;
    }
    TaskListLockGuard guard(tlist);
    tlist->push(task);
}

[[nodiscard]] Task* l1_get_next_task(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_get_next_task: null pdctx; returning nullptr");
        return nullptr;
    }
    return pdctx->getNextTask();
}

// Shared implementation for l1_push_new_dl_tasks_bulk and l1_push_new_ul_tasks_bulk.
// Callers are responsible for the pdctx null check and for selecting the correct tList;
// tList null is handled internally.
[[nodiscard]] int push_tasks_bulk_impl(PhyDriverCtx* pdctx, TaskList* tList,
                                       uint64_t ts_exec_ns,
                                       std::span<const TaskSpec> specs,
                                       const char* fn_name)
{
    if (tList == nullptr)
    {
        NVLOGW_FMT(TAG, "{}: null tList; {} tasks dropped", fn_name, specs.size());
        return 0;
    }

    // Allocate and init all tasks before acquiring the lock so that the
    // critical section only contains pushes (no heap allocation inside).

    // Runtime bounds check: clamp to prevent a stack buffer overflow of ready[].
    if (specs.size() > static_cast<std::size_t>(TaskSpec::BULK_MAX))
    {
        NVLOGW_FMT(TAG, "{}: specs.size()={} exceeds BULK_MAX={}; clamping to BULK_MAX",
                   fn_name, specs.size(), TaskSpec::BULK_MAX);
        specs = specs.first(TaskSpec::BULK_MAX);
    }

    Task* ready[TaskSpec::BULK_MAX]{};
    int   n_ready = 0;
    const t_ns ts{ static_cast<t_ns::rep>(ts_exec_ns) };

    for (auto&& [i, spec] : ranges::views::enumerate(specs))
    {
        Task* task = pdctx->getNextTask();
        if (task == nullptr)
        {
            NVLOGW_FMT(TAG, "{}: task pool exhausted at spec[{}] '{}'; "
                       "{} of {} tasks skipped", fn_name, i, spec.name,
                       specs.size() - static_cast<std::size_t>(i), specs.size());
            break;
        }
        // l1_task_work_fn_t and task_work_function are identical types — no cast required.
        if (task->init(ts, spec.name.data(), spec.fn, spec.arg,
                       spec.first_cell, spec.num_cells, spec.num_tasks,
                       spec.wid) != 0)
        {
            NVLOGW_FMT(TAG, "{}: task->init() failed for '{}'; skipped", fn_name, spec.name);
            // Slot consumed from the ring; reused on next wrap-around.
            continue;
        }
        ready[n_ready++] = task;
    }

    // Single lock/unlock for the entire batch.
    return tList->push_bulk(std::span<Task* const>{ready, static_cast<std::size_t>(n_ready)});
}

} // namespace

worker_id l1_get_dl_worker_id(phydriver_handle pdh, int worker_index)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_get_dl_worker_id: null pdctx; returning INVALID_WORKER_ID");
        return INVALID_WORKER_ID;
    }
    return pdctx->getDLWorkerID(worker_index);
}

worker_id l1_get_ul_worker_id(phydriver_handle pdh, int worker_index)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_get_ul_worker_id: null pdctx; returning INVALID_WORKER_ID");
        return INVALID_WORKER_ID;
    }
    return pdctx->getULWorkerID(worker_index);
}

uint8_t l1_get_cpu_task_tracing_mode(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_get_cpu_task_tracing_mode: null pdctx; returning DISABLED (0)");
        return 0;
    }
    return pdctx->enableCPUTaskTracing();
}

PMUDeltaSummarizer* l1_get_worker_pmu(Worker* worker)
{
    return worker != nullptr ? worker->getPMU() : nullptr;
}

[[nodiscard]] bool l1_push_new_dl_task(phydriver_handle pdh, uint64_t ts_exec_ns, const char* name,
                                       l1_task_work_fn_t fn, void* arg,
                                       int first_cell, int num_cells, int num_tasks,
                                       worker_id desired_wid)
{
    if (StaticConversion<PhyDriverCtx>(pdh).get() == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "l1_push_new_dl_task: null pdh; task '{}' dropped", name);
        return false;
    }
    Task* task = l1_get_next_task(pdh);
    if (task == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_new_dl_task: task pool exhausted, dropping task '{}' "
                   "(first_cell={}, num_cells={}, ts_ns={})",
                   name, first_cell, num_cells, ts_exec_ns);
        return false;
    }

    // l1_task_work_fn_t and task_work_function are identical types
    // (int(*)(Worker*, void*, int, int, int)) — no cast required.
    if (task->init(t_ns(static_cast<t_ns::rep>(ts_exec_ns)), name,
                   fn, arg, first_cell, num_cells, num_tasks, desired_wid) != 0)
    {
        NVLOGW_FMT(TAG, "l1_push_new_dl_task: task->init() failed for '{}'; task dropped", name);
        // NOTE: the slot is consumed from the ring (TASK_ITEM_NUM=2048) but not
        // pushed — it is reused naturally when task_item_index next wraps around.
        // init() failure is not observed in practice; if it becomes frequent,
        // add an l1_return_task() API to recycle the slot immediately.
        return false;
    }
    l1_push_task_dl(pdh, task);
    return true;
}

[[nodiscard]] bool l1_push_new_ul_task(phydriver_handle pdh, uint64_t ts_exec_ns, const char* name,
                                       l1_task_work_fn_t fn, void* arg,
                                       int first_cell, int num_cells, int num_tasks,
                                       worker_id desired_wid)
{
    if (StaticConversion<PhyDriverCtx>(pdh).get() == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "l1_push_new_ul_task: null pdh; task '{}' dropped", name);
        return false;
    }
    Task* task = l1_get_next_task(pdh);
    if (task == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_new_ul_task: task pool exhausted, dropping task '{}' "
                   "(first_cell={}, num_cells={}, ts_ns={})",
                   name, first_cell, num_cells, ts_exec_ns);
        return false;
    }

    // l1_task_work_fn_t and task_work_function are identical types
    // (int(*)(Worker*, void*, int, int, int)) — no cast required.
    if (task->init(t_ns(static_cast<t_ns::rep>(ts_exec_ns)), name,
                   fn, arg, first_cell, num_cells, num_tasks, desired_wid) != 0)
    {
        NVLOGW_FMT(TAG, "l1_push_new_ul_task: task->init() failed for '{}'; task dropped", name);
        // NOTE: the slot is consumed from the ring (TASK_ITEM_NUM=2048) but not
        // pushed — it is reused naturally when task_item_index next wraps around.
        // init() failure is not observed in practice; if it becomes frequent,
        // add an l1_return_task() API to recycle the slot immediately.
        return false;
    }
    l1_push_task_ul(pdh, task);
    return true;
}

[[nodiscard]] int l1_push_new_dl_tasks_bulk(phydriver_handle pdh, uint64_t ts_exec_ns,
                                            std::span<const TaskSpec> specs)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_new_dl_tasks_bulk: null pdctx; {} tasks dropped", specs.size());
        return 0;
    }
    return push_tasks_bulk_impl(pdctx, pdctx->getTaskListDl(),
                                ts_exec_ns, specs,
                                "l1_push_new_dl_tasks_bulk");
}

[[nodiscard]] int l1_push_new_ul_tasks_bulk(phydriver_handle pdh, uint64_t ts_exec_ns,
                                            std::span<const TaskSpec> specs)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        NVLOGW_FMT(TAG, "l1_push_new_ul_tasks_bulk: null pdctx; {} tasks dropped", specs.size());
        return 0;
    }
    return push_tasks_bulk_impl(pdctx, pdctx->getTaskListUl(),
                                ts_exec_ns, specs,
                                "l1_push_new_ul_tasks_bulk");
}

int l1_cell_create(phydriver_handle pdh, struct cell_phy_info& cell_pinfo)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        pdctx->setPuschEarlyHarqEn(cell_pinfo.is_early_harq_detection_enabled);
        pdctx->setPuschAggrFactor(cell_pinfo.pusch_aggr_factor);

        ret = pdctx->setCellPhyByMplane(cell_pinfo);
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "setCellPhyByMplane() failed with error {}", ret);
            return ret;
        }

        Cell* c = pdctx->getCellByPhyId(cell_pinfo.phy_stat.phyCellId);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't getCellByPhyId {}", cell_pinfo.phy_stat.phyCellId);
            return -1;
        }

    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_cell_destroy(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return pdctx->removeCell(cell_id);
}

int l1_cell_start(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    Cell* c = pdctx->getCellByPhyId(cell_id);
    if(c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't find any cell with id {}", cell_id);
        return -1;
    }

    c->start();
    AppConfig::getInstance().cellActivated(c->getMplaneId());
    return 0;
}

int l1_cell_stop(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    Cell* c = pdctx->getCellByPhyId(cell_id);
    if(c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't find any cell with id {}", cell_id);
        return -1;
    }
    c->stop();
    AppConfig::getInstance().cellDeactivated(c->getMplaneId());
    return 0;
}

int l1_set_log_error_handler(phydriver_handle pdh, log_handler_fn_t log_fn)
{
    PhyDriverCtx* pdctx = nullptr;

    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        try
        {
            pdctx->set_error_logger(log_fn);
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_set_log_info_handler(phydriver_handle pdh, log_handler_fn_t log_fn)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        try
        {
            pdctx->set_info_logger(log_fn);
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_set_log_debug_handler(phydriver_handle pdh, log_handler_fn_t log_fn)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        try
        {
            pdctx->set_debug_logger(log_fn);
        }
        PHYDRIVER_CATCH_EXCEPTIONS();
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_set_log_level(phydriver_handle pdh, l1_log_level log_lvl)
{
    PhyDriverCtx* pdctx = nullptr;
    try
    {
        pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        pdctx->set_level_logger(log_lvl);
        NVLOGC_FMT(TAG, "Log level set to {}", +log_lvl);
    }
    PHYDRIVER_CATCH_EXCEPTIONS();
    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, uint16_t grid_sz, bool dl)
{
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* c = pdctx->getCellByMplaneId(mplane_id);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Could't getCellByMplaneId ", __func__, mplane_id);
            return -1;
        }

        if(c->isActive())
        {
           NVLOGC_FMT(TAG, "Cell active, cannot update config!");
           return -1;
        }

        if(dl)
        {
            NVLOGC_FMT(TAG, "Update cell: mplane_id={} dl_grid_sz={} ", mplane_id, grid_sz);
            c->setDLGridSize(grid_sz);
        }
        else
        {
            NVLOGC_FMT(TAG, "Update cell: mplane_id={} ul_grid_sz={} ", mplane_id, grid_sz);
            c->setULGridSize(grid_sz);
        }
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::string dst_mac, uint16_t vlan_tci)
{
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* c = pdctx->getCellByMplaneId(mplane_id);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Could't getCellByMplaneId ", __func__, mplane_id);
            return -1;
        }

        if(c->isActive())
        {
           NVLOGC_FMT(TAG, "Cell active, cannot update config!");
           return -1;
        }

        NVLOGC_FMT(TAG, "Update cell: mplane_id={} dst_mac={} vlan_tci=0x{:X}", mplane_id, dst_mac.c_str(), vlan_tci);
        c->updateCellConfig(dst_mac, vlan_tci);
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

int l1_cell_update_attenuation(phydriver_handle pdh, uint16_t mplane_id, float attenuation_dB)
{
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        Cell* c = pdctx->getCellByMplaneId(mplane_id);
        if(c == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Could't getCellByMplaneId ", __func__, mplane_id);
            return -1;
        }

        NVLOGC_FMT(TAG, "Update cell: mplane_id={} attenuation_dB={} dB", mplane_id, attenuation_dB);
        c->setAttenuation_dB(attenuation_dB);
    }
    PHYDRIVER_CATCH_EXCEPTIONS();

    return 0;
}

void l1_bind_thread_to_phy_cuda_context(phydriver_handle pdh)
{
    if(pdh == nullptr)
    {
        return;
    }
    try
    {
        PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
        GpuDevice* gpu = pdctx->getFirstGpu();
        if(gpu == nullptr)
        {
            NVLOGW_FMT(TAG, "{}: no GPU device", __func__);
            return;
        }
        gpu->setDevice();
        NVLOGC_FMT(TAG, "{}: thread bound to PHY CUDA primary context", __func__);
    }
    PHYDRIVER_CATCH_EXCEPTIONS_VOID();
}

int l1_update_gps_alpha_beta(phydriver_handle pdh,uint64_t alpha,int64_t beta)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->set_gps_alpha(alpha);
    pdctx->set_gps_beta(beta);

    return 0;
}

void* ptp_svc_monitoring_func(void* arg)
{
    NVLOGI_FMT(TAG, "ptp_svc_monitoring_func thread");

    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "cuphydriver thread {} affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "PtpMonitoring") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();

    std::string syslogPath = "/host/var/log/syslog";
    auto ptp_rms_threshold = appConfig.getPtpRmsThreshold();
    int ptp_status = 0;
    /* Single thread runs this (one pthread from l1_initialize); no synchronization needed. */
    static time_t last_reported_link_down_ts = time(nullptr);
    static time_t last_reported_link_up_ts = time(nullptr);

    while(1)
    {
        int new_ptp_status = AppUtils::checkPtpServiceStatus(syslogPath, ptp_rms_threshold, ptp_rms_threshold);
        if(ptp_status != new_ptp_status)
        {
            //Send PTP ERROR.indication
            slot_command_api::dl_slot_callbacks      dl_cb{};
            std::array<uint32_t, MAX_CELLS_PER_SLOT> cell_idx_list = {};
            const auto                               cell_count    = pdctx->getCellIdxList(cell_idx_list);
            auto                                     error_code    = new_ptp_status ? SCF_ERROR_CODE_PTP_SVC_ERROR : SCF_ERROR_CODE_PTP_SYNCED;
            if(pdctx->getDlCb(dl_cb))
            {
                if(cell_count > 0)
                {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, error_code, cell_idx_list, cell_count);
                }
            }
            ptp_status = new_ptp_status;
        }

        time_t link_down_ts = static_cast<time_t>(-1);
        time_t link_up_ts = static_cast<time_t>(-1);
        AppUtils::getPtpPortLinkEvents(syslogPath, link_down_ts, link_up_ts);
        const bool new_down = (link_down_ts != static_cast<time_t>(-1) && link_down_ts > last_reported_link_down_ts);
        const bool new_up = (link_up_ts != static_cast<time_t>(-1) && link_up_ts > last_reported_link_up_ts);
        if (new_down || new_up) {
            if (new_down) last_reported_link_down_ts = link_down_ts;
            if (new_up) last_reported_link_up_ts = link_up_ts;

            slot_command_api::dl_slot_callbacks dl_cb{};
            std::array<uint32_t, MAX_CELLS_PER_SLOT> cell_idx_list = {};
            const auto cell_count = pdctx->getCellIdxList(cell_idx_list);
            if (pdctx->getDlCb(dl_cb) && cell_count > 0 && dl_cb.l1_exit_error_fn) {
                if (new_down && new_up) {
                    /* Ordering is best-effort when link_down_ts == link_up_ts (syslog has 1s granularity). */
                    if (link_down_ts <= link_up_ts) {
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_DOWN, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link down indication (0x99)");
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_UP, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link up indication (0x9A)");
                    } else {
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_UP, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link up indication (0x9A)");
                        dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_DOWN, cell_idx_list, cell_count);
                        NVLOGI_FMT(TAG, "FH port link down indication (0x99)");
                    }
                } else if (new_down) {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_DOWN, cell_idx_list, cell_count);
                    NVLOGI_FMT(TAG, "FH port link down indication (0x99)");
                } else if (new_up) {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, SCF_ERROR_CODE_FH_PORT_UP, cell_idx_list, cell_count);
                    NVLOGI_FMT(TAG, "FH port link up indication (0x9A)");
                }
            }
        }

        usleep(200000);  /* 200 ms poll for faster port link indication */
    }

    NVLOGI_FMT(TAG, "ptp_svc_monitoring thread exit");
    return nullptr;
}

void* rhocp_ptp_events_monitoring_func(void* arg)
{
    NVLOGC_FMT(TAG, "RhocpPtpMonitor thread");
    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGC_FMT(TAG, "cuphydriver thread {} affinity set to cpu core {}", __func__, low_priority_core);
    // Set thread affinity to low priority core
    nv_assign_thread_cpu_core(low_priority_core); 

    if(pthread_setname_np(pthread_self(), "RhocpPtpMon") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();

    auto rhocp_ptp_publisher = appConfig.getRhocpPtpPublisher();
    auto rhocp_ptp_node_name = appConfig.getRhocpPtpNodeName();
    auto rhocp_ptp_consumer = appConfig.getRhocpPtpConsumer();

    // FYI: startEventServer will create another event consumer thread.
    std::shared_ptr<httplib::Server>  svr = AppUtils::startEventServer(rhocp_ptp_consumer);
    if (!svr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to start event consumer server for RHOCP PTP events");
        return nullptr;
    }

    // Give it a moment to start before subscription
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // To pull, we need subscriptions exist first.
    AppUtils::subscribeToEvents(rhocp_ptp_publisher, rhocp_ptp_node_name, rhocp_ptp_consumer);
    // Starting from RHOCP v4.18, pull also requires the consumer server to be running, otherwise subscription will be deleted. 
    // svr->stop();

    int ptp_status = 0;

    while (1) {
        int new_ptp_status = AppUtils::pullEvents(rhocp_ptp_publisher, rhocp_ptp_node_name);
        NVLOGD_FMT(TAG, "RHOCP PTP events status: {} -> {}", ptp_status, new_ptp_status); 
        if(ptp_status != new_ptp_status)
        {
             //Send RHOCP PTP ERROR.indication
            slot_command_api::dl_slot_callbacks      dl_cb{};
            std::array<uint32_t, MAX_CELLS_PER_SLOT> cell_idx_list = {};
            const auto                               cell_count    = pdctx->getCellIdxList(cell_idx_list);
            auto                                     error_code    = new_ptp_status ? SCF_ERROR_CODE_RHOCP_PTP_EVENTS_ERROR : SCF_ERROR_CODE_RHOCP_PTP_EVENTS_SYNCED;
            if(pdctx->getDlCb(dl_cb))
            {
                if(cell_count > 0)
                {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION, error_code, cell_idx_list, cell_count);
                    if (new_ptp_status == 0)
                        NVLOGW_FMT(TAG, "RHOCP PTP events status backed to normal. Sending indication SCF_ERROR_CODE_RHOCP_PTP_EVENTS_SYNCED code=0x{:X}", +error_code);   
                    else
                        NVLOGW_FMT(TAG, "RHOCP PTP events ERROR detected. Sending indication SCF_ERROR_CODE_RHOCP_PTP_EVENTS_ERROR code=0x{:X}", +error_code);
                }
            }
            ptp_status = new_ptp_status;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    NVLOGC_FMT(TAG, "rhocp_ptp_events_monitoring thread exited");
    return nullptr;
}

void* cell_update_config_func(void* arg)
{
    NVLOGI_FMT(TAG, "cell_update_config_func thread");

    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "cuphydriver thread {} affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "cell_update_cfg") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    phydriver_handle pdh = reinterpret_cast<phydriver_handle>(arg);
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx->createPrachObjects()!= 0)
    {
        if(pdctx->cellUpdateCbExists())
        {
            NVLOGW_FMT(TAG, "{}: createPrachObjects failed, calling cell_update_cb for cell_id={} with error_code=0x{:X}", __func__, pdctx->updateCellConfigCellId, +SCF_ERROR_CODE_MSG_INVALID_CONFIG);
            pdctx->cell_update_cb(pdctx->updateCellConfigCellId, SCF_ERROR_CODE_MSG_INVALID_CONFIG);
        }
        pdctx->updateCellConfigMutex.unlock();
        NVLOGI_FMT(TAG, "cell_update_config_func: createPrachObjects failed. Send INVALID_CONFIG in CONFIG.RSP. Thread exit");
        return nullptr;
    }

    Cell* cell_list[MAX_CELLS_PER_SLOT];
    uint32_t cellCount = 0;
    pdctx->getCellList(cell_list,&cellCount);

    bool any_cell_active = false;
    for(uint32_t i = 0; i < cellCount; i++)
    {
        auto& cell_ptr = cell_list[i];
        if(cell_ptr->isActive())
            any_cell_active = true;
    }

    if(!any_cell_active)
    {
        pdctx->replacePrachObjects();
    }

    NVLOGI_FMT(TAG, "cell_update_config_func thread exit");
    return nullptr;
}

int l1_lock_update_cell_config_mutex(phydriver_handle pdh)
{

    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->updateCellConfigMutex.try_lock();
}

int l1_unlock_update_cell_config_mutex(phydriver_handle pdh)
{

    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->updateCellConfigMutex.unlock();
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* c = pdctx->getCellByMplaneId(mplane_id);
    if(c == nullptr)
    {
        NVLOGC_FMT(TAG, "Cell does not exist, cannot update config!");
        return -1;
    }

    if(c->isActive())
    {
        NVLOGC_FMT(TAG, "Cell active, cannot update eAxCIDs!");
        return -1;
    }
    else
    {
        c->updateeAxCIds(eaxcids_ch_map);
    }
    return 0;
}

int l1_cell_update_cell_config(phydriver_handle pdh, uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* c = pdctx->getCellByMplaneId(mplane_id);
    if(c == nullptr)
    {
        NVLOGC_FMT(TAG, "Cell does not exist, cannot update config!");
        return -1;
    }

    if(c->isActive())
    {
        std::unordered_set<std::string> del;
        for(auto& p : attrs)
        {
            if(strcmp(p.first.c_str(), CELL_PARAM_NIC) == 0) continue;
            NVLOGC_FMT(TAG, "Cell active, skip updating '{}' ", p.first);
            res[p.first] = -1;
            del.insert(p.first);
        }
        for(auto key : del)
        {
            attrs.erase(key);
        }
    }

    if(attrs.find(CELL_PARAM_DL_COMP_METH) != attrs.end() || attrs.find(CELL_PARAM_DL_BIT_WIDTH) != attrs.end())
    {
        if(attrs.find(CELL_PARAM_DL_COMP_METH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "dl_comp_meth is missing, skip updating dl IQ data format...");
            res[CELL_PARAM_DL_BIT_WIDTH] = -1;
        }
        else if(attrs.find(CELL_PARAM_DL_BIT_WIDTH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "dl_bit_width is missing, skip updating dl IQ data format...");
            res[CELL_PARAM_DL_COMP_METH] = -1;
        }
        else
        {
            auto    comp_meth = static_cast<UserDataCompressionMethod>(attrs[CELL_PARAM_DL_COMP_METH]);
            uint8_t bit_width = attrs[CELL_PARAM_DL_BIT_WIDTH];
            if(comp_meth == UserDataCompressionMethod::NO_COMPRESSION && bit_width != 16)
            {
                NVLOGC_FMT(TAG, "Error dl_bit_width {}, Fix point currently only supports 16 bit_width, skip updating dl IQ data format...", bit_width);
                res[CELL_PARAM_DL_BIT_WIDTH] = -1;
                res[CELL_PARAM_DL_COMP_METH] = -1;
            }
            else if(comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT && (bit_width != 9 && bit_width != 14 && bit_width != 16))
            {
                NVLOGC_FMT(TAG, "Error dl_bit_width {}, BFP currently only supports 9, 14, 16 bit_width, skip updating dl IQ data format...", bit_width);
                res[CELL_PARAM_DL_BIT_WIDTH] = -1;
                res[CELL_PARAM_DL_COMP_METH] = -1;
            }
            else
            {
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_DL_COMP_METH, attrs[CELL_PARAM_DL_COMP_METH]);
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_DL_BIT_WIDTH, attrs[CELL_PARAM_DL_BIT_WIDTH]);
                c->setDLIQDataFmt(comp_meth, bit_width);
            }
        }
        attrs.erase(CELL_PARAM_DL_COMP_METH);
        attrs.erase(CELL_PARAM_DL_BIT_WIDTH);
    }

    if(attrs.find(CELL_PARAM_UL_COMP_METH) != attrs.end() || attrs.find(CELL_PARAM_UL_BIT_WIDTH) != attrs.end())
    {
        if(attrs.find(CELL_PARAM_UL_COMP_METH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "ul_comp_meth is missing, skip updating ul IQ data format...");
            res[CELL_PARAM_UL_BIT_WIDTH] = -1;
        }
        else if(attrs.find(CELL_PARAM_UL_BIT_WIDTH) == attrs.end())
        {
            NVLOGC_FMT(TAG, "ul_bit_width is missing, skip updating ul IQ data format...");
            res[CELL_PARAM_UL_COMP_METH] = -1;
        }
        else
        {
            auto    comp_meth = static_cast<UserDataCompressionMethod>(attrs[CELL_PARAM_UL_COMP_METH]);
            uint8_t bit_width = attrs[CELL_PARAM_UL_BIT_WIDTH];
            if(comp_meth == UserDataCompressionMethod::NO_COMPRESSION && bit_width != 16)
            {
                NVLOGC_FMT(TAG, "Error ul_bit_width {}, Fix point currently only supports 16 bit_width, skip updating ul IQ data format...", bit_width);
                res[CELL_PARAM_UL_BIT_WIDTH] = -1;
                res[CELL_PARAM_UL_COMP_METH] = -1;
            }
            else if(comp_meth == UserDataCompressionMethod::BLOCK_FLOATING_POINT && (bit_width != 9 && bit_width != 14 && bit_width != 16))
            {
                NVLOGC_FMT(TAG, "Error ul_bit_width {}, BFP currently only supports 9, 14, 16 bit_width, skip updating ul IQ data format...", bit_width);
                res[CELL_PARAM_UL_BIT_WIDTH] = -1;
                res[CELL_PARAM_UL_COMP_METH] = -1;
            }
            else
            {
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_UL_COMP_METH, attrs[CELL_PARAM_UL_COMP_METH]);
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_UL_BIT_WIDTH, attrs[CELL_PARAM_UL_BIT_WIDTH]);
                c->setULIQDataFmt(comp_meth, bit_width);
            }
        }
        attrs.erase(CELL_PARAM_UL_COMP_METH);
        attrs.erase(CELL_PARAM_UL_BIT_WIDTH);
    }

    for(auto& p : attrs)
    {
        if(strcmp(p.first.c_str(), CELL_PARAM_UL_GAIN_CALIBRATION) == 0 || strcmp(p.first.c_str(), CELL_PARAM_LOWER_GUARD_BW) == 0)
        {
            continue;
        }

        if(strcmp(p.first.c_str(), CELL_PARAM_NIC) != 0 && strcmp(p.first.c_str(), CELL_PARAM_DST_MAC_ADDR) != 0
        && strcmp(p.first.c_str(), CELL_PARAM_VLAN_ID) != 0 && strcmp(p.first.c_str(), CELL_PARAM_PCP) != 0)
        {
            NVLOGC_FMT(TAG, "{} updated to {:.0f} ", p.first.c_str(), p.second);
        }

        if(strcmp(p.first.c_str(), CELL_PARAM_DST_MAC_ADDR) == 0)
        {
            if(attrs.find(CELL_PARAM_VLAN_ID) == attrs.end() || attrs.find(CELL_PARAM_PCP) == attrs.end())
            {
                NVLOGC_FMT(TAG, "No vlan_id/pcp provided, skip update mac ... ");
            }
            uint64_t mac = p.second;
            std::string dst_mac;
            for(int i = 0; i < 6; i++)
            {
                dst_mac = (dst_mac.size() > 0 ? ":" : "") + dst_mac;
                std::ostringstream ss;
                ss << std::setfill('0') << std::setw(2) << std::hex << (mac & 0xFF);
                dst_mac = ss.str() + dst_mac;
                mac >>= 8;
            }
            uint32_t vlan_id = attrs[CELL_PARAM_VLAN_ID];
            uint32_t pcp = attrs[CELL_PARAM_PCP];
            uint32_t vlan_tci = (pcp << 13) | vlan_id;
            NVLOGC_FMT(TAG, "dst_mac updated to {} ", dst_mac.c_str());
            NVLOGC_FMT(TAG, "vlan_id updated to {} ", vlan_id);
            NVLOGC_FMT(TAG, "pcp updated to {} ", pcp);
            //NVLOGC_FMT(TAG, "dst_mac={} vlan_tci=0x{:X}", dst_mac.c_str(), vlan_tci);
            c->updateCellConfig(dst_mac, vlan_tci);
        }

        if(strcmp(p.first.c_str(), CELL_PARAM_RU_TYPE) == 0)
        {
            c->setRUType(static_cast<ru_type>(p.second));
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_EXPONENT_DL) == 0)
        {
            c->setDlExponent(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_EXPONENT_UL) == 0)
        {
            c->setUlExponent(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_MAX_AMP_UL) == 0)
        {
            c->setUlMaxAmp(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_PUSCH_PRB_STRIDE) == 0)
        {
            if(pdctx->enableL1ParamSanityCheck())
            {
                if(p.second>ORAN_MAX_PRB) //Exceeds spec limit
                {
                    NVLOGW_FMT(TAG,"Invalid L1 Param value {} provided for CELL_PARAM_PUSCH_PRB_STRIDE.Setting default value of {}",p.second,ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL);
                    p.second=ORAN_PUSCH_PRBS_X_PORT_X_SYMBOL;
                }
                c->setPuschPrbStride(p.second);
            }
            else
                c->setPuschPrbStride(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_PRACH_PRB_STRIDE) == 0)
        {
            if(pdctx->enableL1ParamSanityCheck())
            {
                if(p.second>ORAN_PRACH_PRB) //Exceeds spec limit
                {
                    NVLOGW_FMT(TAG,"Invalid L1 Param value {} provided for CELL_PARAM_PRACH_PRB_STRIDE.Setting default value of {}",p.second,ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL);
                    p.second=ORAN_PRACH_B4_PRBS_X_PORT_X_SYMBOL;
                }
                c->setPrachPrbStride(p.second);
            }
            else
                c->setPrachPrbStride(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_SECTION_3_TIME_OFFSET) == 0)
        {
            c->setSection3TimeOffset(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_FH_DISTANCE_RANGE) == 0)
        {
            c->updateFhLenConfig(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_REF_DL) == 0)
        {
            c->setRefDl(p.second);
        }
        else if(strcmp(p.first.c_str(), CELL_PARAM_NIC) == 0)
        {
            uint64_t address = p.second;
            uint32_t domain = (address >> 20) & 0xFFFF;
            uint32_t bus = (address >> 12) & 0xFF;
            uint32_t device = (address >> 4) & 0xFF;
            uint32_t function = address & 0xF;

            std::stringstream ss;
            ss << std::hex << std::setw(4) << std::setfill('0') << domain << ":"
               << std::setw(2) << std::setfill('0') << bus << ":"
               << std::setw(2) << std::setfill('0') << device << "."
               << std::setw(1) << std::setfill('0') << function;

            std::string pcie_address = ss.str();
            NVLOGC_FMT(TAG, "nic updated to {} ", pcie_address.c_str());
            auto ret = c->setNicName(pcie_address);
            if(ret != 0)
            {
                res[p.first] = ret;
            }
        }
    }
    return 0;
}

bool l1_phy_cell_id_mismatch(phydriver_handle pdh, uint16_t mplane_id, uint16_t new_phy_cell_id)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: failed to get PhyDriverCtx", __func__);
        return true;
    }

    Cell* cell_ptr = pdctx->getCellByMplaneId(mplane_id);
    if (cell_ptr == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: failed to getCellByMplaneId {}", __func__ , mplane_id);
        return true;
    }
    return pdctx->phyCellIdMismatch(cell_ptr->getPhyId(), new_phy_cell_id, cell_ptr->getId());
}

int l1_cell_update_cell_config(phydriver_handle pdh, struct cell_phy_info& cell_pinfo, CellUpdateCallBackFn& callback)
{
    PhyDriverCtx* pdctx = nullptr;
    int           ret   = 0;

    pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* cell_ptr = pdctx->getCellByMplaneId(cell_pinfo.mplane_id);
    if(cell_ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:Could't getCellByPhyId {}", __func__ , cell_pinfo.phy_stat.phyCellId);
        return -1;
    }

    NVLOGI_FMT(TAG, "Update cell config received for PCI {} new PCI {}",cell_ptr->getPhyId(), cell_pinfo.phy_stat.phyCellId);

    if(cell_ptr->getPhyId() != cell_pinfo.phy_stat.phyCellId)
    {
        //If the PCI has been changed, update cell_index_map to use the new phyCellId for this cell
        ret = pdctx->setCellPhyId(cell_ptr->getPhyId(),cell_pinfo.phy_stat.phyCellId,cell_ptr->getId());
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:setCellPhyId error {}", __func__ , ret);
            return ret;
        }
    }

    //Update the static parameters of the cell
    ret = cell_ptr->setPhyStatic(cell_pinfo);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:setPhyStaticInfo error {}", __func__ , ret);
        return ret;
    }

    if(pdctx->updateCellConfig(cell_ptr->getId(),cell_pinfo) == 1)
    {
        pdctx->setCellUpdateCb(callback);
        pdctx->updateCellConfigCellId = cell_pinfo.mplane_id;
        pthread_t thread_id;
        int status=pthread_create(&thread_id, nullptr, cell_update_config_func, pdh);

        if(status == 0)
        {
            NVLOGC_FMT(TAG, "launch a new thread cell_update_config_func for mplane_id={}. Return 1", cell_pinfo.mplane_id);
            // Switch to a low_priority_core to avoid blocking time critical thread
            auto&     appConfig         = AppConfig::getInstance();
            auto      low_priority_core = appConfig.getLowPriorityCore();
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(low_priority_core, &cpuset);
            status = pthread_setaffinity_np(thread_id, sizeof(cpu_set_t), &cpuset);
            if(status)
            {
                NVLOGW_FMT(TAG, "cell_update_config_func setaffinity_np failed with status : {}", std::strerror(status));
            }
            return 1;
        }
    }
    else if(ret == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "updateCellConfig failed");
        return ret;
    }
    NVLOGI_FMT(TAG, "Successful update of cell config. Return 0") ;
    return 0;
}

namespace {

[[nodiscard]] int update_cell_static_config(PhyDriverCtx& pdctx, Cell& cell, cell_phy_info& cell_pinfo)
{
    if(cell.getPhyId() != cell_pinfo.phy_stat.phyCellId)
    {
        const int ret = pdctx.setCellPhyId(cell.getPhyId(), cell_pinfo.phy_stat.phyCellId, cell.getId());
        if(ret)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:setCellPhyId error {}", __func__ , ret);
            return ret;
        }
    }

    const int ret = cell.setPhyStatic(cell_pinfo);
    if(ret)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}:setPhyStaticInfo error {}", __func__ , ret);
    }
    return ret;
}

[[nodiscard]] int finish_prach_update_caller_runs(PhyDriverCtx& pdctx)
{
    if(pdctx.createPrachObjects() != 0)
    {
        return -1;
    }
    return pdctx.replacePrachObjectsCallerRuns();
}

// Resolved (PhyDriverCtx, Cell) pair for a cell reconfiguration request; both null on
// either lookup failure (already logged).
struct ResolvedCellCtx
{
    PhyDriverCtx* pdctx;
    Cell*         cell;
};

// Shared preamble for the caller-runs reconfig entry: phydriver_handle + mplane_id ->
// (PhyDriverCtx&, Cell&), logging and returning {nullptr, nullptr} on failure.
[[nodiscard]] ResolvedCellCtx resolve_cell_ctx(phydriver_handle pdh, const cell_phy_info& cell_pinfo, const char* caller)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: failed to get PhyDriverCtx", caller);
        return {nullptr, nullptr};
    }
    Cell* cell = pdctx->getCellByMplaneId(cell_pinfo.mplane_id);
    if(cell == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: could not get cell for mplane_id {}", caller, cell_pinfo.mplane_id);
        return {nullptr, nullptr};
    }
    return {pdctx, cell};
}

} // namespace

int l1_cell_update_cell_config_caller_runs(phydriver_handle pdh, cell_phy_info& cell_pinfo)
{
    const ResolvedCellCtx ctx = resolve_cell_ctx(pdh, cell_pinfo, __func__);
    if(ctx.pdctx == nullptr)
    {
        return -1;
    }

    // Single entry point; the active? branch lives here rather than in the caller. The
    // decision uses cuphydriver's own active-cell state (hasActiveCells) -- the same truth
    // the inactive path already gated on, and the one the cuPHY PRACH handles depend on.
    if(ctx.pdctx->hasActiveCells())
    {
#ifdef ENABLE_FAPI_STORE_REPLAY
        // Other cells are running: reconfigure without disrupting them via the offload
        // PRACH handover. reconfigure() owns the stage -> create -> static-config ->
        // commit -> arm ordering and the rollback; the live vectors publish and the
        // per-aggregator cuPHY handle swap then drains at slot boundaries with no blocking
        // wait (mirrors legacy cell_update_config_func + getNextPrachAggr).
        NVLOGI_FMT(TAG, "Offload active-cell PRACH reconfig for PCI {} new PCI {}",
            ctx.cell->getPhyId(), cell_pinfo.phy_stat.phyCellId);

        PrachOffloadReconfig&        reconfig = ctx.pdctx->prachOffloadReconfig();
        const PrachReconfigOutcome   outcome  = reconfig.reconfigure(
            *ctx.cell, cell_pinfo,
            [&]() -> int { return update_cell_static_config(*ctx.pdctx, *ctx.cell, cell_pinfo); });

        if(outcome != PrachReconfigOutcome::Committed)
        {
            // The failing step logs its own cause; this is the summary (outcome value
            // included for correlation).
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "{}: active-cell PRACH reconfig failed for PCI {} (outcome={})",
                __func__, ctx.cell->getPhyId(), static_cast<int>(outcome));
            return -1;
        }
        return 0;
#else
        NVLOGW_FMT(TAG, "{}: active-cell caller-runs reconfig is not supported", __func__);
        return -1;
#endif
    }

    // No active cells: synchronous caller-runs reconfig (legacy). Static config first,
    // then the PRACH objects are (re)created and swapped all at once -- safe while idle.
    NVLOGI_FMT(TAG, "Caller-runs update cell config received for PCI {} new PCI {}",
        ctx.cell->getPhyId(), cell_pinfo.phy_stat.phyCellId);

    const int ret = update_cell_static_config(*ctx.pdctx, *ctx.cell, cell_pinfo);
    if(ret != 0)
    {
        return ret;
    }

    const int update_ret = ctx.pdctx->updateCellConfig(ctx.cell->getId(), cell_pinfo);
    if(update_ret == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "updateCellConfig failed");
        return update_ret;
    }
    if(update_ret == 1)
    {
        return finish_prach_update_caller_runs(*ctx.pdctx);
    }
    NVLOGI_FMT(TAG, "Successful caller-runs update of cell config. Return 0");
    return 0;
}

#ifdef ENABLE_FAPI_STORE_REPLAY
void l1_try_commit_prach_offload_handover(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx != nullptr)
    {
        // Lock-free no-op unless an offload PRACH handover is armed; otherwise
        // drains one incremental per-aggregator handle swap for this slot.
        pdctx->prachOffloadReconfig().tryCommit();
    }
}
#endif // ENABLE_FAPI_STORE_REPLAY

uint8_t l1_get_prach_start_ro_index(phydriver_handle pdh, uint16_t phyCellId)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    Cell* c = pdctx->getCellByPhyId(phyCellId);
    if(c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Could't find any cell with id {}", phyCellId);
        return -1;
    }
    return c->getPrachOccaPrmStatIdx();
}

bool l1_allocSrsChesBuffPool(phydriver_handle pdh, uint32_t requestedBy, uint16_t phyCellId, uint32_t poolSize)
{
    bool retVal = false;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx != nullptr)
    {
        CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();
        if(cv != nullptr)
        {
            retVal = cv->memPoolAllocatePerCell(requestedBy, phyCellId, poolSize);
        }
    }
    return retVal;
}

bool l1_deAllocSrsChesBuffPool(phydriver_handle pdh, uint16_t phyCellId)
{
    bool retVal = false;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(pdctx != nullptr)
    {
        CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();
        if(cv != nullptr)
        {
            retVal = cv->memPoolDeAllocatePerCell(phyCellId);
        }
    }
    return retVal;
}

void l1_copy_TB_to_gpu_buf(phydriver_handle pdh, uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    auto* mgr = pdctx->getH2DCopyManager();
    Cell* c = pdctx->getCellByPhyId(phy_cell_id);
    (*gpu_buff_ref) = (uint8_t *)c->get_pdsch_tb_buffer(slot_index);
    mgr->setPreponeEnabled(true);
    NVLOGI_FMT(TAG, "l1_copy_TB_to_gpu_buf pdh={} cell_id={} tb_buff={} gpu_buff_ref={} tb_len={} slot_index={}",(void*)pdh,phy_cell_id,(void*)tb_buff,(void*)gpu_buff_ref,tb_len,slot_index);
    mgr->setCtx();

    if(mgr->getBuffCopyCount() == 0)
    {
        CUDA_CHECK(cudaEventRecord(mgr->getStartEvent(slot_index), mgr->getStream()));
    }

    CUDA_CHECK(cudaMemcpyAsync(aerial::casts::assume_cast<uint32_t>(*gpu_buff_ref),
                               tb_buff,
                               tb_len,
                               cudaMemcpyHostToDevice,
                               mgr->getStream()));
    if(mgr->incBuffCopyCount() == pdctx->getCellNum())
    {
        mgr->setBuffCopyCount(0);
        CUDA_CHECK(cudaEventRecord(mgr->getCompleteEvent(slot_index), mgr->getStream()));
    }
}

void l1_copy_TB_to_gpu_buf_thread_func(phydriver_handle pdh, std::stop_token st)
{
    NVLOGI_FMT(TAG,"l1_copy_TB_to_gpu_buf_thread_func Entry");
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    auto* mgr = pdctx->getH2DCopyManager();
    h2d_copy_prepone_info_t h2d_cpy_info{};
    Cell*    c{nullptr};
    uint16_t h2d_ri{};
    uint16_t h2d_wi{};
    int      h2d_done_slot_idx{-1};

    std::vector<cuphyBatchedMemcpyHelper> batched_memcpy_helper;
    batched_memcpy_helper.reserve(PDSCH_MAX_GPU_BUFFS);
    std::generate_n(std::back_inserter(batched_memcpy_helper), PDSCH_MAX_GPU_BUFFS,
        [mgr]() {
            return cuphyBatchedMemcpyHelper(DL_MAX_CELLS_PER_SLOT, batchedMemcpySrcHint::srcIsHost, batchedMemcpyDstHint::dstIsDevice, (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1) && mgr->getUseBatchedMemcpy());
        });
    std::for_each(batched_memcpy_helper.begin(), batched_memcpy_helper.end(),
        [](auto& helper) { helper.reset(); });

    cudaStream_t h2d_copy_stream = mgr->getStream();
    std::array<int32_t, PDSCH_MAX_GPU_BUFFS> active_batch_sfn;
    active_batch_sfn.fill(-1);

    // Set context on this thread once at startup, so any overhead when running through
    // tools, e.g., for the first CUDA API call like cuCtxSetCurrent, is paid here and not
    // on the critical path later.
    mgr->setCtx();

    // Cooperative shutdown via std::jthread's built-in stop_token:
    // PdschH2DCopyManager::stopThread() calls request_stop() and joins.
    // st.stop_requested() flips to true on the next iteration; the loop
    // exits and the destructor tears down the CUDA stream/events without
    // use-after-free on the captured `mgr`/`pdctx`.
    while (!st.stop_requested())
    {
            h2d_ri = mgr->readIdx();
            h2d_wi = mgr->writeIdx();
            h2d_done_slot_idx = mgr->doneCurSlotIdx()[mgr->doneCurSlotReadIdx()].load(std::memory_order_acquire);
            if(h2d_ri != h2d_wi)
            {
                h2d_cpy_info = (h2d_copy_prepone_info_t)(*mgr->getPreponeInfo(mgr->readIdx()));
                c = pdctx->getCellByPhyId(h2d_cpy_info.phy_cell_id);
                NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_func pdh={} cell_id={} tb_buff={} gpu_buff_ref={} tb_len={} slot_index={} h2d_read_idx={} h2d_write_idx={}",pdh,h2d_cpy_info.phy_cell_id,(void*)h2d_cpy_info.tb_buff,(void*)h2d_cpy_info.gpu_buff_ref,h2d_cpy_info.tb_len,h2d_cpy_info.slot_index,h2d_ri,h2d_wi);
                mgr->setCtx();

                uint8_t si = h2d_cpy_info.slot_index;
                if (batched_memcpy_helper[si].getMemcpyCount() > 0 &&
                    active_batch_sfn[si] != static_cast<int32_t>(h2d_cpy_info.sfn))
                {
                    NVLOGW_FMT(TAG, "{}: Stale batch detected for slot_index={} (SFN {} vs {}), discarding {}/{} copies (slot was likely dropped)",
                               __func__, (int)si, active_batch_sfn[si], (int)h2d_cpy_info.sfn,
                               batched_memcpy_helper[si].getMemcpyCount(),
                               batched_memcpy_helper[si].getMaxMemcopiesCount());
                    batched_memcpy_helper[si].reset();
                }
                active_batch_sfn[si] = static_cast<int32_t>(h2d_cpy_info.sfn);

                batched_memcpy_helper[h2d_cpy_info.slot_index].updateMemcpy((uint32_t*)c->get_pdsch_tb_buffer(h2d_cpy_info.slot_index),
                                                   h2d_cpy_info.tb_buff,
                                                   h2d_cpy_info.tb_len,
                                                   cudaMemcpyHostToDevice,
                                                   h2d_copy_stream);

                mgr->readIdx() = (mgr->readIdx() + 1) % (DL_MAX_CELLS_PER_SLOT * PDSCH_MAX_GPU_BUFFS);
                NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_func cell_id={},h2d_read_idx={},h2d_write_idx={}", h2d_cpy_info.phy_cell_id,(int)h2d_ri,(int)h2d_wi);
            }
            if(h2d_done_slot_idx >= 0)
            {
                uint8_t slot_index = h2d_done_slot_idx % PDSCH_MAX_GPU_BUFFS;
                h2d_cpy_info = (h2d_copy_prepone_info_t)(*mgr->getPreponeInfo(mgr->readIdx()));
                if ((slot_index == h2d_cpy_info.slot_index) && (h2d_ri != h2d_wi))
                {
                    continue;
                }
                mgr->setCtx();

                cuphyStatus_t launch_status = CUPHY_STATUS_SUCCESS;
                if (batched_memcpy_helper[slot_index].getMemcpyCount() != 0)
                {
                    CUDA_CHECK(cudaEventRecord(mgr->getStartEvent(h2d_done_slot_idx), mgr->getStream()));

                    NVLOGI_FMT(TAG, "Launching batched memcpy with {} copies for slot {} ", batched_memcpy_helper[slot_index].getMemcpyCount(), slot_index);
                    launch_status = batched_memcpy_helper[slot_index].launchBatchedMemcpy(h2d_copy_stream);
                    if (launch_status != CUPHY_STATUS_SUCCESS) [[unlikely]]
                    {
                        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                                   "{}: launchBatchedMemcpy failed st={} slot={} - "
                                   "complete event NOT recorded; PDSCH will time out",
                                   __func__, static_cast<int>(launch_status), slot_index);
                    }
                    batched_memcpy_helper[slot_index].reset();
                }
                active_batch_sfn[slot_index] = -1;

                NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_func triggering cudaEventRecord h2d_read_idx={},h2d_write_idx={}",(int)h2d_ri,(int)h2d_wi);
                // Only record complete event on launch success — otherwise a
                // downstream cudaStreamWaitEvent would proceed even though the
                // DMA never completed, producing silent data corruption. The
                // flag below is still set so the consumer's spin exits and
                // PDSCH fails fast with a clear "TB never arrived" symptom.
                if (launch_status == CUPHY_STATUS_SUCCESS) [[likely]]
                {
                    CUDA_CHECK(cudaEventRecord(mgr->getCompleteEvent(h2d_done_slot_idx), mgr->getStream()));
                }
                // Release: synchronizes-with the acquire-load in
                // waitH2dCopyCudaEventRec (phypdsch_aggr.cpp). The consumer
                // then issues cudaStreamWaitEvent on the just-recorded
                // complete event — that ordering must be observable on ARM
                // (Grace), where a relaxed store would allow reordering of
                // the cudaEventRecord after the flag write.
                mgr->cudaEventRecDone()[slot_index].store(true, std::memory_order_release);
                mgr->doneCurSlotIdx()[mgr->doneCurSlotReadIdx()].store(-1, std::memory_order_relaxed);
                mgr->doneCurSlotReadIdx() = (mgr->doneCurSlotReadIdx() + 1) % PDSCH_MAX_GPU_BUFFS;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::nanoseconds(5000));
            }
    }
}

void l1_copy_TB_to_gpu_buf_thread_offload(phydriver_handle pdh, uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref, uint32_t tb_len, uint8_t slot_index, uint16_t sfn)
{
    h2d_copy_prepone_info_t h2d_cpy_info;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    auto* mgr = pdctx->getH2DCopyManager();
    h2d_cpy_info.pdh=pdh;
    h2d_cpy_info.phy_cell_id=phy_cell_id;
    h2d_cpy_info.tb_buff=tb_buff;
    h2d_cpy_info.gpu_buff_ref=gpu_buff_ref;
    h2d_cpy_info.tb_len=tb_len;
    h2d_cpy_info.slot_index=slot_index;
    h2d_cpy_info.sfn=sfn;
    Cell* c = pdctx->getCellByPhyId(phy_cell_id);
    (*gpu_buff_ref) = (uint8_t *)c->get_pdsch_tb_buffer(slot_index);
    mgr->setPreponeEnabled(true);
    NVLOGD_FMT(TAG, "l1_copy_TB_to_gpu_buf_thread_offload pdh={} cell_id={} tb_buff={} gpu_buff_ref={} tb_len={} slot_index={} h2d_copy_thread_enable={}",(void*)pdh,phy_cell_id,(void*)tb_buff,(void*)gpu_buff_ref,tb_len,slot_index,mgr->isThreadEnabled());
    if(!mgr->isThreadEnabled())
    {
        mgr->updateBatchedMemcpyInfo((uint32_t*)c->get_pdsch_tb_buffer(h2d_cpy_info.slot_index),
                                h2d_cpy_info.tb_buff,
                                h2d_cpy_info.tb_len);
    }
    else
    {
        uint16_t h2d_widx = mgr->writeIdx();
        h2d_copy_prepone_info_t* h2d_cpy_info_mgr = mgr->getPreponeInfo(h2d_widx);
        *h2d_cpy_info_mgr = h2d_cpy_info;
        h2d_widx = (h2d_widx + 1) % (DL_MAX_CELLS_PER_SLOT * PDSCH_MAX_GPU_BUFFS);
        mgr->writeIdx() = h2d_widx;
    }
}

void l1_set_h2d_copy_done_cur_slot_flag(phydriver_handle pdh,int slot_idx)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    auto* mgr = pdctx->getH2DCopyManager();
    mgr->doneCurSlotIdx()[mgr->doneCurSlotWriteIdx()].store(slot_idx, std::memory_order_release);
    mgr->doneCurSlotWriteIdx() = (mgr->doneCurSlotWriteIdx() + 1) % PDSCH_MAX_GPU_BUFFS;
    NVLOGD_FMT(TAG, "l1_set_h2d_copy_done_cur_slot_flag Set h2d_copy_done_cur_slot to true for slot_idx(%d)",slot_idx);
}

int l1_stage_tb_h2d(const phydriver_handle pdh,
                    const uint16_t         phy_cell_id,
                    const uint8_t*         shm_src,
                    const uint32_t         len,
                    const uint8_t          slot_index,
                    uint8_t**              gpu_buf_out)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr || shm_src == nullptr || gpu_buf_out == nullptr || len == 0U)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "l1_stage_tb_h2d: invalid args pdctx=0x{:x} shm_src=0x{:x} gpu_buf_out=0x{:x} len={}",
                   reinterpret_cast<uintptr_t>(pdctx), reinterpret_cast<uintptr_t>(shm_src),
                   reinterpret_cast<uintptr_t>(gpu_buf_out), len);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    auto* mgr = pdctx->getH2DCopyManager();
    Cell* c = pdctx->getCellByPhyId(phy_cell_id);
    if (c == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "l1_stage_tb_h2d: cell not found cell_id={}", phy_cell_id);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    void* const gpu = c->get_pdsch_tb_buffer(slot_index);
    if (gpu == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "l1_stage_tb_h2d: null GPU buffer cell_id={} slot_index={}", phy_cell_id, slot_index);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *gpu_buf_out = static_cast<uint8_t*>(gpu);
    mgr->setPreponeEnabled(true);
    mgr->updateBatchedMemcpyInfo(aerial::casts::assume_cast<uint32_t>(gpu),
                                 shm_src,
                                 static_cast<std::size_t>(len));
    NVLOGD_FMT(TAG,
               "l1_stage_tb_h2d: OK cell_id={} len={} slot_index={} gpu=0x{:x}",
               phy_cell_id, len, slot_index, reinterpret_cast<uintptr_t>(gpu));
    return CUPHY_STATUS_SUCCESS;
}

int l1_launch_tb_h2d(const phydriver_handle pdh, const uint16_t slot_in_frame)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        return static_cast<int>(CUPHY_STATUS_INVALID_ARGUMENT);
    }
    auto* mgr = pdctx->getH2DCopyManager();
    const uint8_t ev_slot = static_cast<uint8_t>(slot_in_frame);
    mgr->setCtx();
    CUDA_CHECK(cudaEventRecord(mgr->getStartEvent(ev_slot), mgr->getStream()));
    cuphyStatus_t const st = mgr->performBatchedMemcpy();
    mgr->resetBatchedMemcpyBatches();
    if (st != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(TAG,
                   AERIAL_CUPHYDRV_API_EVENT,
                   "l1_launch_tb_h2d: performBatchedMemcpy failed st={} slot={} - "
                   "TB H2D complete event NOT recorded; PDSCH will time out",
                   static_cast<int>(st),
                   ev_slot);
        return static_cast<int>(st);
    }
    CUDA_CHECK(cudaEventRecord(mgr->getCompleteEvent(ev_slot), mgr->getStream()));
    // The buffer-ring index is semantically independent from SLOTS_PER_FRAME
    // even though both happen to be 20 today, so wrap explicitly to match
    // what the copy-thread path at L3570 does.
    const uint8_t buf_slot = ev_slot % PDSCH_MAX_GPU_BUFFS;
    // Release: synchronizes-with the acquire-load in waitH2dCopyCudaEventRec
    // (phypdsch_aggr.cpp). Mirrors the copy-thread path; required on Grace.
    mgr->cudaEventRecDone()[buf_slot].store(true, std::memory_order_release);
    mgr->markBatchLaunched(buf_slot);
    return static_cast<int>(st);
}

bool l1_get_h2d_copy_thread_enable(const phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr)
    {
        return false;
    }
    auto* mgr = pdctx->getH2DCopyManager();
    return mgr->isThreadEnabled();
}

int l1_cv_mem_bank_update(phydriver_handle pdh,uint32_t cell_id,uint16_t rnti,uint16_t buffer_idx,uint16_t reportType,uint16_t startPrbGrp,uint32_t srsPrbGrpSize ,uint16_t numPrgs,
        uint8_t nGnbAnt,uint8_t nUeAnt,uint32_t offset, uint8_t* srsChEsts, uint16_t startValidPrg, uint16_t nValidPrg)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }

    NVLOGD_FMT(TAG, "cell_id {} rnti {} startPrbGrp {} srsPrbGrpSize{} ",cell_id,rnti,startPrbGrp,srsPrbGrpSize);

    CVSrsChestBuff *buffer = nullptr;
    if(cv->preAllocateBuffer(cell_id, rnti, buffer_idx, reportType, &buffer, nullptr))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: allocateBuffer returned error", __func__);
        return -1;
    }
    buffer->configSrsInfo(numPrgs, nGnbAnt, nUeAnt, srsPrbGrpSize, startPrbGrp, startValidPrg, nValidPrg);

    //uint8_t* ptr = srsChEsts+offset;
    //NVLOGD(TAG, "%s: srsChEst = %d %d %d %d\n",__func__,*ptr, *(ptr+1), *(ptr+2), *(ptr+3));
    uint8_t* dst = nullptr;
    /* Each ChEst Buffer is of geometry - nPrbGrp x nGnbAnt x nUeLayer with nPrbGrp being the fastest changing dimension
     * In C style ChEst buffer dimensions are -  [nUeLayer][nGnbAnt][nPrbGrp].
     */
    uint32_t size_of_half2 = sizeof(uint32_t);

    MemtraceDisableScope mds;

    pdctx->getSrsMpsCtx()->setCtx();
    for(uint32_t i=0; i < nUeAnt; i++)
    {
        for(uint32_t j=0; j < nGnbAnt; j++)
        {
            uint32_t k = startPrbGrp;
            uint32_t addrOffset=size_of_half2 * (i*nGnbAnt*numPrgs + j*numPrgs + k);
            dst = buffer->getAddr() + addrOffset;
            // TODO replace w/ 3D memcopy
            CUDA_DRIVER_CHECK(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), srsChEsts + offset + addrOffset, size_of_half2 * numPrgs));
        }
    }
    return 0;
}
    
int l1_cv_mem_bank_retrieve_buffer(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint16_t reportType,uint8_t *pSrsPrgSize, uint16_t* pSrsStartPrg, uint16_t* pSrsStartValidPrg, uint16_t* pSrsNValidPrg, cuphyTensorDescriptor_t* descr, uint8_t** ptr)
{
    if(ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Invalid input - nullptr pointer", __func__);
        return -1;
    }
    else
        *ptr = nullptr;

    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }

    CVSrsChestBuff *buffer = nullptr;
    if(cv->retrieveBuffer(cell_id, rnti, buffer_idx, reportType, &buffer))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: retrieveBuffer returned error for cell_id {} rnti {} reportType {}", __func__, cell_id, rnti, reportType);
        return -1;
    }

    *ptr = buffer->getAddr();
    buffer->getSrsPrgInfo(pSrsPrgSize, pSrsStartPrg, pSrsStartValidPrg, pSrsNValidPrg);
    *descr = buffer->getSrsDescr();
    return 0;
}

int l1_cv_mem_bank_update_buffer_state(phydriver_handle pdh,uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    cv->updateSrsChestBufferState(cell_id, buffer_idx, srs_chest_buff_state);
    return 0;
    
}

int l1_cv_mem_bank_get_buffer_state(phydriver_handle pdh,uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState* srs_chest_buff_state)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    *srs_chest_buff_state = cv->getSrsChestBufferState(cell_id, buffer_idx);
    return 0;
    
}

int l1_cv_mem_bank_update_buffer_usage(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint32_t usage)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    cv->updateSrsChestBufferUsage(cell_id, rnti, buffer_idx, usage);
    return 0;
}

int l1_cv_mem_bank_get_buffer_usage(phydriver_handle pdh,uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint32_t* usage)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    CvSrsChestMemoryBank* cv = pdctx->getCvSrsChestMemoryBank();

    if(cv == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CV Memory Bank does not exist", __func__);
        return -1;
    }
    *usage = cv->getSrsChestBufferUsage(cell_id, rnti, buffer_idx);
    return 0;
}

int l1_bfw_coeff_retrieve_buffer(phydriver_handle pdh, uint32_t cell_id, bfw_buffer_info* bfw_buffer_info)
{
    if(bfw_buffer_info == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: Invalid input - nullptr pointer", __func__);
        return -1;
    }
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    *bfw_buffer_info = *pdctx->getBfwCoeffBuffer(cell_id);
    return 0;
}


int l1_mMIMO_enable_info(phydriver_handle pdh, uint8_t *pMuMIMO_enable)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *pMuMIMO_enable = pdctx->getmMIMO_enable();
    return 0;
}

int l1_enable_srs_info(phydriver_handle pdh, uint8_t *pEnable_srs)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *pEnable_srs = pdctx->get_enable_srs();
    return 0;
}

int l1_get_cell_group_num(phydriver_handle pdh, uint8_t *cell_group_num)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *cell_group_num = pdctx->getCellGroupNum();
    return 0;
}

int l1_get_ch_segment_proc_enable_info(phydriver_handle pdh, uint8_t* ch_seg_proc_enable)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    *ch_seg_proc_enable = pdctx->get_ch_segment_proc_enable();
    return 0;
}

bool l1_incr_recovery_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->incrL1RecoverySlots();
}

bool l1_incr_all_obj_free_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->incrAllObjFreeSlots();
}

void l1_reset_all_obj_free_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->resetAllObjFreeSlots();
}

void l1_reset_recovery_slots(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->resetL1RecoverySlots();
}

int l1_storeDBTPduInFH(phydriver_handle pdh, uint16_t cell_id, void* data_buf)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->storeDBTPdu(cell_id, data_buf);
}

int l1_resetDBTStorageInFH(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->resetDBTStorage(cell_id);
}

int l1_getBeamWeightsSentFlagInFH(phydriver_handle pdh, uint16_t cell_id, uint16_t beamIdx)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->getBeamWeightsSentFlag(cell_id,beamIdx);
}

int l1_setBeamWeightsSentFlagInFH(phydriver_handle pdh, uint16_t cell_id, uint16_t beamIdx)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->setBeamWeightsSentFlag(cell_id,beamIdx);
}

int16_t l1_getDynamicBeamIdOffset(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return (fhproxy->getBfwCPlaneChainingMode() == aerial_fh::BfwCplaneChainingMode::NO_CHAINING) ? -1 : fhproxy->getDynamicBeamIdOffset();
}

int l1_staticBFWConfiguredInFH(phydriver_handle pdh, uint16_t cell_id)
{
    PhyDriverCtx* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    if (fhproxy == nullptr)
    {
        return -1;
    }
    return fhproxy->staticBFWConfigured(cell_id);
}

int l1_clear_task_list(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    TaskList*     tListUL = pdctx->getTaskListUl();
    TaskList*     tListDL = pdctx->getTaskListDl();

    tListUL->lock();
    tListUL->clear_task_all();
    tListUL->unlock();

    tListDL->lock();
    tListDL->clear_task_all();
    tListDL->unlock();
    NVLOGI_FMT(TAG,"{}: Clearing DL/UL Task Lists", __func__);
    return 0;
}


phydriver_handle l1_getPhydriverHandle()
{
    return l1_pdh;
}

pthread_t l1_getFmtLogThreadId()
{
    return gBg_thread_id;
}

int l1_get_send_static_bfw_wt_all_cplane(phydriver_handle pdh)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->get_send_static_bfw_wt_all_cplane();
}

// Reason for macro is not to add extra call of function to be part of the stack.
// And since we have few locations where we need it, macro is a proper use here.
#define AERIAL_PRINT_BACKTRACE(TRACE_CNT_MAX)      \
    do {                                           \
        backward::StackTrace st;                   \
        std::ignore = st.load_here(TRACE_CNT_MAX); \
        backward::Printer p;                       \
        p.print(st);                               \
    } while(false)

void l1_exit_handler()
{
    static constexpr auto trace_cnt_max = 32ULL;
    NVLOGC_FMT(TAG,"Triggering L1 exit handler");

    //PhyDriver initialization failure
    if(l1_getPhydriverHandle() == nullptr)
    {
        AERIAL_PRINT_BACKTRACE(32ULL);
        NVLOGW_FMT(TAG, "L1 exit handler: PhyDriver handle is null, cleanup may be incomplete");
        return;
    }

#ifdef ENABLE_DPDK_TX_PKT_TRACING
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    rte_trace_save();
#pragma GCC diagnostic pop 
#endif   

    //Step 1 : Send ERROR.indication (Part1) here
    auto* pdctx = StaticConversion<PhyDriverCtx>(l1_getPhydriverHandle()).get();
    slot_command_api::dl_slot_callbacks dl_cb{};
    std::array<uint32_t,MAX_CELLS_PER_SLOT> cell_idx_list={};
    const auto cell_count = pdctx->getCellIdxList(cell_idx_list);
    if(pdctx->getDlCb(dl_cb))
    {
        if(cell_count>0)
        {
            dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION,SCF_ERROR_CODE_L1_P1_EXIT_ERROR,cell_idx_list,cell_count);
        }
    }
    NVLOGC_FMT(TAG,"L1 exit handler: end of step 1");

    //Step 2 : Clear UL/DL Tasks which are still in the Task queue
    l1_clear_task_list(l1_getPhydriverHandle());

    NVLOGC_FMT(TAG,"L1 exit handler: end of step 2");

    //Step 3 : Issue cuCtxSynchronize if CUDA coredump env variables are set
    if(const auto *enable_core = getenv("CUDA_ENABLE_COREDUMP_ON_EXCEPTION"); enable_core)
    {
        if(1==std::stoi(enable_core))
        {
            MemtraceDisableScope md;
            NVLOGC_FMT(TAG,"L1 exit handler: step 3 (CUDA_ENABLE_COREDUMP_ON_EXCEPTION = 1)");

            // If a CUDA coredump generation is triggered, it is possible some of these steps won't take place,
            // if an abort is triggered by default at the end of the GPU core dump generation process.

            //Synchronize for all MPS contexts
            for(int i=0;i<pdctx->mpsCtxList.size();i++)
            {
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - MPS context {} work start", i);
                pdctx->mpsCtxList[i]->setCtx();
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuCtxSynchronize());
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - MPS context {} work end", i);
            }

            //Synchronize Primary context as well
            {
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - primary context work start");
                CUcontext  CurCtx;
                CUcontext  PrimaryCtx;
                CUdevice   cuDev;
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuCtxGetDevice(&cuDev));
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuDevicePrimaryCtxRetain(&PrimaryCtx,cuDev));
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuCtxSynchronize());
                NVLOGC_FMT(TAG,"L1 exit handler: step 3 - primary context work end");
                CU_CHECK_L1_EXIT_PHYDRIVER_NONFATAL(cuDevicePrimaryCtxRelease(cuDev));
            }
            //Step 4 : Send ERROR.indication (Part2) here
            NVLOGC_FMT(TAG,"L1 exit handler: step 4");
            if(pdctx->getDlCb(dl_cb))
            {
                if(cell_count>0)
                {
                    dl_cb.l1_exit_error_fn(SCF_FAPI_ERROR_INDICATION,SCF_ERROR_CODE_L1_P2_EXIT_ERROR,cell_idx_list,cell_count);
                }
            }
            NVLOGC_FMT(TAG,"L1 exit handler: will close nvlog");
            nvlog_fmtlog_close(l1_getFmtLogThreadId());
            printf("L1 exit handler: closed nvlog\n");
            std::fflush(nullptr);
            asm volatile("" : : : "memory"); //Memory barrier inserted here to prevent compiler from reordering and thereby prematurely triggering the system abort before SCF_ERROR_CODE_L1_P2_EXIT_ERROR is sent to L2
            AERIAL_PRINT_BACKTRACE(trace_cnt_max);
            return;
        }
        else
        {
            NVLOGC_FMT(TAG,"L1 exit handler: step 3 (CUDA_ENABLE_COREDUMP_ON_EXCEPTION = 0)");
            asm volatile("" : : : "memory"); //Memory barrier inserted here to prevent compiler from reordering
            MemtraceDisableScope md;
            AERIAL_PRINT_BACKTRACE(trace_cnt_max);
            return;
        }
    }
    else // core dump set to 0
    {
        NVLOGC_FMT(TAG,"L1 exit handler: step 3 (CUDA_ENABLE_COREDUMP_ON_EXCEPTION unset)");
        asm volatile("" : : : "memory"); //Memory barrier inserted here to prevent compiler from reordering
        MemtraceDisableScope md;
        AERIAL_PRINT_BACKTRACE(trace_cnt_max);
        return;
    }
}

bool l1_check_cuphy_objects_status(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getAggrObjFreeStatus(); 
}

void l1_resetBatchedMemcpyBatches(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    pdctx->getH2DCopyManager()->resetBatchedMemcpyBatches();
}

uint8_t l1_get_enable_weighted_average_cfo(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getEnableWeightedAverageCfo();
}

uint8_t l1_get_cplane_processing_dl_batch_size(phydriver_handle pdh)
{
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getCplaneProcessingDlBatchSize();
}

uint8_t l1_get_cplane_processing_ul_batch_size(phydriver_handle pdh)
{
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getCplaneProcessingUlBatchSize();
}

bool l1_get_split_ul_cuda_streams(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->splitUlCudaStreamsEnabled();
}

bool l1_get_dl_tx_notification(phydriver_handle pdh)
{
    auto* pdctx =  StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx->getEnableTxNotification();
}

[[nodiscard]] int l1_init_cplane_generator(phydriver_handle pdh,
                                           bool bf_enabled,
                                           bool precoding_enabled)
{
    if (!pdh) return 0;
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    auto* svc = pdctx->getFrameworkCPlaneService();
    if (!svc) return 0;
    return svc->init(pdctx, pdctx->getSortedCells(), bf_enabled, precoding_enabled);
}

[[nodiscard]] int l1_recordDirectBfwCviRecord(const phydriver_handle pdh,
                                              const uint16_t cell_id,
                                              const direct_bfw_cvi_record& record)
{
    if (!pdh) {
        return -1;
    }
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (pdctx == nullptr) {
        return -1;
    }
    return pdctx->recordDirectBfwCviRecord(cell_id, record);
}

[[nodiscard]] int l1_send_dl_cplane_from_stored_msg(phydriver_handle pdh,
                                                     const nv::phy_mac_msg_desc* dl_tti,
                                                     const nv::phy_mac_msg_desc* ul_dci,
                                                     std::size_t transaction_id,
                                                     SlotMapDl* slot_map_dl)
{
    if (!pdh) return 0;
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    auto* svc = pdctx->getFrameworkCPlaneService();
    if (!svc || !svc->is_active()) return 0;
    return svc->send_dl_cplane(dl_tti, ul_dci, transaction_id, slot_map_dl);
}

[[nodiscard]] int l1_send_ul_cplane_from_stored_msg(phydriver_handle pdh,
                                                     const nv::phy_mac_msg_desc* ul_tti,
                                                     std::size_t transaction_id,
                                                     SlotMapUl* slot_map_ul)
{
    if (!pdh) return 0;
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    auto* svc = pdctx->getFrameworkCPlaneService();
    if (!svc || !svc->is_active()) return 0;
    return svc->send_ul_cplane(ul_tti, transaction_id, slot_map_ul);
}

bool l1_is_fapi_to_cplane_direct(phydriver_handle pdh)
{
    if (!pdh) return false;
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    return pdctx && pdctx->isFapiToCplaneDirect();
}

void l1_signal_dl_cplane_batch_done(void* slot_map_dl_ptr,
                                    uint8_t batch_id,
                                    uint8_t total_batches,
                                    uint8_t n_cells_in_batch,
                                    nv::CplaneBatchDirection direction)
{
    auto* slot_map = static_cast<SlotMapDl*>(slot_map_dl_ptr);
    if (!slot_map) {
        NVLOGW_FMT(TAG, "[DL_BATCH_DONE] slot_map=NULL batch_id={} total_batches={} n_cells={} direction={} - skipping signals",
                   batch_id, total_batches, n_cells_in_batch, static_cast<int>(direction));
        return;
    }
    NVLOGD_FMT(TAG, "[DL_BATCH_DONE] slot_map_id={} batch_id={} total_batches={} n_cells={} direction={}",
               slot_map->getId(), batch_id, total_batches, n_cells_in_batch, static_cast<int>(direction));
    // Each batch-done invocation = one DL C-plane contributor for this slot.  Across
    // the slot's k batches the handler is called k times, producing k contributions
    // to each counter.  Consumers' wait targets are slot_map_dl->getNumDlcTasks(),
    // which equals k after PHY_module::fire_cplane_batch has populated it.
    //
    // Signals emitted per call:
    //   - incDLCDone           — consumed by task_work_function_dl_aggr_3_buf_cleanup's
    //                            waitDLCDone(slot_map->getNumDlcTasks()) on the mMIMO path.
    //   - incUplanePrepDone    — consumed by task_work_function_dl_aggr_2_gpu_comm_tx's
    //                            waitUplanePrepDone(slot_map->getNumDlcTasks()).
    //   - addSlotEndTask       — consumed by task_work_function_dl_aggr_3_buf_cleanup's
    //                            waitSlotEndTask(dl_task_count).
    //
    // FHCB-done signals (setCellFHCBDone / setFHCBDone) are intentionally NOT emitted
    // here under the fapi_to_cplane_direct path: the only legacy callers of waitFHCBDone are
    // task_work_function_dl_aggr_2_gpu_comm and _gpu_comm_prepare, both of which are
    // skipped in the fapi_to_cplane_direct path (SKIP_DL_GPU_COMM_PREPARE / split-TX routing).
    slot_map->incDLCDone();
    slot_map->incUplanePrepDone();
    slot_map->addSlotEndTask();
}

void l1_signal_ul_cplane_batch_done(void* slot_map_ul_ptr,
                                    uint8_t batch_id,
                                    uint8_t total_batches,
                                    uint8_t n_cells_in_batch,
                                    nv::CplaneBatchDirection direction)
{
    auto* slot_map = static_cast<SlotMapUl*>(slot_map_ul_ptr);
    if (!slot_map) {
        NVLOGW_FMT(TAG, "[UL_BATCH_DONE] slot_map=NULL batch_id={} total_batches={} n_cells={} direction={} - skipping signals",
                   batch_id, total_batches, n_cells_in_batch, static_cast<int>(direction));
        return;
    }
    NVLOGD_FMT(TAG, "[UL_BATCH_DONE] slot_map_id={} batch_id={} total_batches={} n_cells={} direction={}",
               slot_map->getId(), batch_id, total_batches, n_cells_in_batch, static_cast<int>(direction));
    // Each batch-done invocation = one UL C-plane contributor for this slot.  Across
    // the slot's k batches the handler is called k times, producing k contributions to
    // each counter.  Consumers' wait targets are slot_map_ul->getNumUlcTasks(), which
    // equals k after PHY_module::fire_cplane_batch has populated it.
    //
    // Signals emitted per call:
    //   - addULCTasksComplete  — consumed by waitULCTasksComplete(getNumUlcTasks()) in
    //                            Order Kernel, PUCCH+PUSCH, Early UCI Ind, UL3 task bodies.
    //   - addSlotEndTask       — consumed by UL3's waitSlotEndTask(ul_task_count).
    slot_map->addULCTasksComplete();
    slot_map->addSlotEndTask();
}

// ---------------------------------------------------------------------------
// fapi_to_cplane_direct C-plane batch error handlers.
//
// Invoked by task_work_fn_cplane_batch (nv_cplane_batch_tasks.cpp) when one
// or more cells in a batch returned a non-zero code from process_cell.
//
// Wired into CplaneBatchTaskArg::on_batch_error via the per-direction
// Policy::on_batch_error_fn() method.
// ---------------------------------------------------------------------------

void l1_handle_ul_cplane_batch_error(void* slot_map_ul_ptr,
                                     std::span<const uint32_t> err_cell_ids)
{
    if (!slot_map_ul_ptr || err_cell_ids.empty()) {
        return;
    }
    auto* slot_map = static_cast<SlotMapUl*>(slot_map_ul_ptr);

#ifdef ENABLE_FAPI_STORE_REPLAY
    slot_map->abortTasks();
#endif

    auto* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();

    slot_command_api::ul_slot_callbacks ul_cb;
    if (pdctx && pdctx->getUlCb(ul_cb) && ul_cb.ul_tx_error_fn != nullptr) {
        std::array<uint32_t, UL_MAX_CELLS_PER_SLOT> err_arr{};
        const auto copy_n = std::min(err_cell_ids.size(), static_cast<std::size_t>(UL_MAX_CELLS_PER_SLOT));
        std::copy_n(err_cell_ids.begin(), copy_n, err_arr.begin());
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "Calling ul_tx_error_fn ({} failed cells, slot_map_id={})",
                   copy_n, slot_map->getId());
        ul_cb.ul_tx_error_fn(ul_cb.ul_tx_error_fn_context,
                             slot_map->getSlot3GPP(),
                             SCF_FAPI_UL_TTI_REQUEST,
                             SCF_ERROR_CODE_L1_UL_CPLANE_TX_ERROR,
                             err_arr,
                             static_cast<uint8_t>(copy_n),
                             false);
    }
#ifndef ENABLE_FAPI_STORE_REPLAY
    slot_map->abortTasks();
#endif
}

void l1_handle_dl_cplane_batch_error(void* slot_map_dl_ptr,
                                     std::span<const uint32_t> err_cell_ids)
{
    if (!slot_map_dl_ptr || err_cell_ids.empty()) {
        return;
    }
    auto* slot_map = static_cast<SlotMapDl*>(slot_map_dl_ptr);

    auto* pdctx = StaticConversion<PhyDriverCtx>(slot_map->getPhyDriverHandler()).get();

    slot_command_api::dl_slot_callbacks dl_cb;
    if (pdctx && pdctx->getDlCb(dl_cb) && dl_cb.dl_tx_error_fn != nullptr) {
        // dl_tx_error_fn expects std::array<uint32_t, DL_MAX_CELLS_PER_SLOT>&
        // and no trailing bool (asymmetric with ul_tx_error_fn).
        std::array<uint32_t, DL_MAX_CELLS_PER_SLOT> err_arr{};
        const auto copy_n = std::min(err_cell_ids.size(), static_cast<std::size_t>(DL_MAX_CELLS_PER_SLOT));
        std::copy_n(err_cell_ids.begin(), copy_n, err_arr.begin());
        // fapi_to_cplane_direct C-plane batch fires from SLOT.resp arrival, BEFORE the legacy
        // DL aggregation populates slot_map->aggr_pdcch_ul / aggr_dlbfw. Those
        // pointers are guaranteed null at this point, so the legacy three-branch
        // selection (task_function_dl_aggr.cpp:1158-1168) cannot be faithfully
        // mirrored here. Collapse to DL_TTI_REQUEST: the fapi_to_cplane_direct batch is always
        // entered from a DL_TTI flow, and the cell-id list + error code carry the
        // actionable information for L2.
        const uint16_t msg_id = static_cast<uint16_t>(SCF_FAPI_DL_TTI_REQUEST);
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "Calling dl_tx_error_fn ({} failed cells, slot_map_id={}, msg_id={:#x})",
                   copy_n, slot_map->getId(), msg_id);
        dl_cb.dl_tx_error_fn(dl_cb.dl_tx_error_fn_context,
                             slot_map->getSlot3GPP(),
                             msg_id,
                             SCF_ERROR_CODE_L1_DL_CPLANE_TX_ERROR,
                             err_arr,
                             static_cast<uint8_t>(copy_n));
    }
}

void l1_set_num_dlc_tasks_for_slot(void* slot_map_dl_ptr, int count)
{
    auto* slot_map = static_cast<SlotMapDl*>(slot_map_dl_ptr);
    if (!slot_map) {
        return;
    }
    slot_map->setNumDlcTasks(count);
}

void l1_set_num_ulc_tasks_for_slot(void* slot_map_ul_ptr, int count)
{
    auto* slot_map = static_cast<SlotMapUl*>(slot_map_ul_ptr);
    if (!slot_map) {
        return;
    }
    slot_map->setNumUlcTasks(count);
}

[[nodiscard]] EarlyCplaneSlotMaps l1_setup_early_cplane_slot_maps(
    phydriver_handle pdh,
    uint16_t sfn, uint16_t slot,
    std::chrono::nanoseconds t0,
    uint64_t dl_cell_bitmap,
    uint64_t ul_cell_bitmap)
{
    EarlyCplaneSlotMaps result{};
    if (!pdh) return result;
    auto* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if (!pdctx || !pdctx->isFapiToCplaneDirect()) {
        return result;
    }

    slot_command_api::slot_indication si(sfn, slot, 0);
    si.t0_ = t0.count();
    si.t0_valid_ = true;

    const int slot_advance = pdctx->get_slot_advance();
    const int64_t tti_ns = static_cast<int64_t>(Cell::getTtiNsFromMu(MU_SUPPORTED));
    const t_ns tick_original = t_ns(t0.count() - slot_advance * tti_ns);
    // Direct C-plane consumes these maps before l1_enqueue_phy_work(), which
    // normally initializes this field. Set the same previous-slot base here
    // so a recycled map cannot expose its stale dynamic-BFW offset.
    const auto dynamic_beam_id_offset = static_cast<int16_t>(
        pdctx->getFhProxy()->getDynamicBeamIdOffsetOfPrevSlot());

    uint32_t active_count{};
    Cell* clist[DL_MAX_CELLS_PER_SLOT]{};
    pdctx->getCellList(clist, &active_count);
    std::sort(clist, clist + active_count,
              [](const Cell* a, const Cell* b) { return a->getIdx() < b->getIdx(); });

    if (dl_cell_bitmap != 0) {
        auto* sm = pdctx->getNextSlotMapDl();
        if (sm) {
            bool dl_buffer_exhausted = false;
            sm->setSlot3GPP(si);
            sm->setDynBeamIdOffset(dynamic_beam_id_offset);
            for (uint32_t i = 0; i < active_count; ++i) {
                if (!clist[i] || !clist[i]->isActive()) continue;
                uint32_t idx = static_cast<uint32_t>(clist[i]->getIdx());
                if (!(dl_cell_bitmap & (uint64_t{1} << idx))) continue;
                DLOutputBuffer* dlbuf = clist[i]->getNextDlBuffer();
                if (dlbuf == nullptr) {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                        "l1_setup_early_cplane_slot_maps: SFN {}.{} map={} "
                        "DL output buffer unavailable for cell_idx={} phy_id={}",
                        sfn, slot, sm->getId(), idx, clist[i]->getPhyId());
                    dl_buffer_exhausted = true;
                    break;
                }
                sm->aggr_cell_list.push_back(clist[i]);
                sm->aggr_dlbuf_list.push_back(dlbuf);
            }
            if (dl_buffer_exhausted) {
                sm->release(static_cast<int>(sm->aggr_cell_list.size()));
            } else {
                std::array<t_ns, TASK_MAX_PER_SLOT + 1> ts_exec{};
                ts_exec[0] = tick_original;
                sm->setTasksTs(1, ts_exec, Time::nowNs());
                // fapi_to_cplane_direct send_dl_cplane reads this before l1_enqueue_phy_work runs.
                sm->setSlotRefTs(tick_original);
                // Freeze waitPeerUpdateDone to this early-map cell count.
                sm->set_num_dl_cplane_peer_ready_targets(static_cast<int>(sm->aggr_cell_list.size()));
                result.dl = sm;
            }
        }
    }

    if (ul_cell_bitmap != 0) {
        auto* sm = pdctx->getNextSlotMapUl();
        if (sm) {
            sm->setSlot3GPP(si);
            sm->setDynBeamIdOffset(dynamic_beam_id_offset);
            for (uint32_t i = 0; i < active_count; ++i) {
                if (!clist[i] || !clist[i]->isActive()) continue;
                uint32_t idx = static_cast<uint32_t>(clist[i]->getIdx());
                if (!(ul_cell_bitmap & (uint64_t{1} << idx))) continue;
                sm->aggr_cell_list.push_back(clist[i]);
            }
            uint32_t ul_count = static_cast<uint32_t>(sm->aggr_cell_list.size());
            sm->aggr_ulbuf_st1.resize(ul_count, nullptr);
            sm->aggr_ulbuf_st2.resize(ul_count, nullptr);
            sm->aggr_ulbuf_pcap_capture.resize(ul_count, nullptr);
            sm->aggr_ulbuf_pcap_capture_ts.resize(ul_count, nullptr);
            sm->num_prach_occa.resize(ul_count, 0);
            std::array<t_ns, TASK_MAX_PER_SLOT + 1> ts_exec{};
            const t_ns ul_task_ts = tick_original + t_ns(tti_ns);
            const int ul_ts_count = static_cast<int>(std::min<uint32_t>(
                ul_count, static_cast<uint32_t>(TASK_MAX_PER_SLOT + 1)));
            for (int i = 0; i < ul_ts_count; ++i) {
                ts_exec[static_cast<std::size_t>(i)] = ul_task_ts;
            }
            sm->setTasksTs((ul_ts_count > 0) ? ul_ts_count : 1, ts_exec, Time::nowNs());
            // fapi_to_cplane_direct send_ul_cplane reads this before l1_enqueue_phy_work runs.
            sm->setSlotRefTs(ul_task_ts);
            result.ul = sm;
        }
    }

    pdctx->setEarlySlotMapDl(result.dl);
    pdctx->setEarlySlotMapUl(result.ul);

    return result;
}
