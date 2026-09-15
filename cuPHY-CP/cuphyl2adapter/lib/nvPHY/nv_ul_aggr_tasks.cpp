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

// All task work functions call only through the UlChannelDispatch table or the
// CplaneBatchTaskArg::process_cell function pointer, keeping all
// process_aggr_*_channel / on_slot_channel_task_complete private.
// Neither nv_phy_module.hpp nor nv_phy_instance.hpp is needed here.
#include "nv_ul_aggr_tasks.hpp"
#include "nv_l2a_task_tracing.hpp"
#include "nvlog.hpp"
#include "task_instrumentation/task_instrumentation_v3.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 14) // "L2A.UL_CHANNELS"

/**
 * @brief Worker function for the cross-cell PUSCH aggregation task (task_aggr_pusch).
 *
 * Delegates iteration and PDU filtering to
 * @ref PHY_module::process_aggr_pusch_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::UlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_pusch(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_ul_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_pusch: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A UL Aggr PUSCH", 4);
    ti.add("Start Task");
    ctx->dispatch->process_pusch(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell PRACH aggregation task (task_aggr_prach).
 *
 * Delegates iteration and PDU filtering to
 * @ref PHY_module::process_aggr_prach_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::UlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_prach(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_ul_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_prach: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A UL Aggr PRACH", 4);
    ti.add("Start Task");
    ctx->dispatch->process_prach(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell PUCCH aggregation task (task_aggr_pucch).
 *
 * Delegates iteration to @ref PHY_module::process_aggr_pucch_channel via the
 * dispatch table. Handles both PUCCH format 0/1 (UL_TTI_NPDUS_IDX_PUCCH_F01)
 * and format 2/3/4 (UL_TTI_NPDUS_IDX_PUCCH_F234); the task is enqueued when
 * either index is non-zero. Then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::UlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_pucch(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_ul_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_pucch: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A UL Aggr PUCCH", 4);
    ti.add("Start Task");
    ctx->dispatch->process_pucch(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell SRS aggregation task (task_aggr_srs).
 *
 * Delegates iteration and PDU filtering to
 * @ref PHY_module::process_aggr_srs_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::UlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_srs(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_ul_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_srs: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A UL Aggr SRS", 4);
    ti.add("Start Task");
    ctx->dispatch->process_srs(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}

/**
 * @brief Worker function for the cross-cell UL beamforming weight aggregation task (task_aggr_ulbfw).
 *
 * Delegates iteration over stored UL_BFW_CVI_REQUEST messages to
 * @ref PHY_module::process_aggr_ulbfw_channel via the dispatch table,
 * then signals task completion.
 *
 * @param arg  Pointer to a @ref nv::UlAggrTaskArg describing the slot context.
 * @return     0 on success; aborts on type-tag mismatch to prevent tasks_in_flight deadlock.
 */
int task_work_fn_aggr_ulbfw(Worker* worker, void* arg, [[maybe_unused]] int first_cell, [[maybe_unused]] int num_cells, [[maybe_unused]] int num_tasks)
{
    auto* ctx = nv::checked_cast_ul_aggr(arg);
    if (ctx == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_work_fn_aggr_ulbfw: type-tag mismatch arg={} — aborting to prevent tasks_in_flight deadlock",
                   arg);
        std::abort();
    }
    auto ti_ctx = nv::makeL2AInstrumentationContext(*ctx, worker);
    TaskInstrumentation ti(ti_ctx, "L2A UL Aggr ULBFW", 4);
    ti.add("Start Task");
    ctx->dispatch->process_ulbfw(ctx->phy_module, *ctx);
    ti.add("On Complete");
    ctx->dispatch->on_complete(ctx->phy_module, ctx->ring_idx, ctx->slot_u32);
    ti.add("End Task");
    return 0;
}
