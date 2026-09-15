/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// All task work functions call only through their respective dispatch tables and
// DlcTaskArg / DlAggrTaskArg, so neither nv_phy_module.hpp nor nv_phy_instance.hpp
// is needed here — all virtual dispatch happens through the function pointers
// stored in the arg structs.
//
// The shared task_work_fn_cplane_batch (DL+UL) lives in nv_cplane_batch_tasks.cpp.
#include "nv_dl_aggr_tasks.hpp"
#include "nv_l2a_task_tracing.hpp"
#include "nvlog.hpp"
#include "task_instrumentation/task_instrumentation_v3.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 13) // "L2A.DL_CHANNELS"

/**
 * @brief Worker function for the cross-cell PDSCH aggregation task (task_aggr_pdsch).
 *
 * Delegates iteration and PDU filtering to
 * @ref PHY_module::process_aggr_pdsch_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::DlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_pdsch(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_dl_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_pdsch: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A DL Aggr PDSCH", 4);
    ti.add("Start Task");
    ctx->dispatch->process_pdsch(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell CSI-RS aggregation task (task_aggr_csirs).
 *
 * Delegates iteration and PDU filtering to
 * @ref PHY_module::process_aggr_csirs_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::DlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_csirs(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_dl_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_csirs: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A DL Aggr CSIRS", 4);
    ti.add("Start Task");
    ctx->dispatch->process_csirs(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell PDCCH aggregation task (task_aggr_pdcch).
 *
 * Delegates iteration over DL_TTI PDCCH PDUs and UL_DCI messages to
 * @ref PHY_module::process_aggr_pdcch_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::DlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_pdcch(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_dl_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_pdcch: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A DL Aggr PDCCH", 4);
    ti.add("Start Task");
    ctx->dispatch->process_pdcch(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell SSB aggregation task (task_aggr_ssb).
 *
 * Delegates iteration and PDU filtering to
 * @ref PHY_module::process_aggr_ssb_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::DlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_ssb(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_dl_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_ssb: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A DL Aggr SSB", 4);
    ti.add("Start Task");
    ctx->dispatch->process_ssb(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell DL beamforming weight aggregation task (task_aggr_dlbfw).
 *
 * Delegates iteration over stored DL_BFW_CVI_REQUEST messages to
 * @ref PHY_module::process_aggr_dlbfw_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::DlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_dlbfw(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_dl_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_dlbfw: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A DL Aggr DLBFW", 4);
    ti.add("Start Task");
    ctx->dispatch->process_dlbfw(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}
