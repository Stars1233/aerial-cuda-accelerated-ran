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

#include "nv_cplane_batch_tasks.hpp"
#include "nv_l2a_task_tracing.hpp"
#include <task_instrumentation/task_instrumentation_v3.hpp>
#include "nvlog.hpp"

#include <array>
#include <limits>
#include <string_view>
#include <fmt/format.h>

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 4) // "DRV.API"

int task_work_fn_cplane_batch(Worker* /*worker*/, void* arg, int /*first_cell*/, int /*num_cells*/, int /*num_tasks*/)
{
    auto* ctx = static_cast<nv::CplaneBatchTaskArg*>(arg);

    constexpr std::string_view kDlBatchNameFmt = "DL Task C-Plane {}";
    constexpr std::string_view kUlBatchNameFmt = "UL Task CPlane {}";
    // Buffer fits longest output: format string with "{}" replaced by max digits of batch_id, plus NUL.
    constexpr std::size_t kBatchNameBufSize =
        std::max(kDlBatchNameFmt.size(), kUlBatchNameFmt.size()) - 2U +                  // strip "{}"
        std::numeric_limits<decltype(ctx->batch_id)>::digits10 + 1U +                    // max digits
        1U;                                                                              // NUL

    char name_buf[kBatchNameBufSize];
    const auto end = (ctx->direction == nv::CplaneBatchDirection::DOWNLINK
        ? fmt::format_to_n(name_buf, kBatchNameBufSize - 1U, kDlBatchNameFmt, ctx->batch_id + 1)
        : fmt::format_to_n(name_buf, kBatchNameBufSize - 1U, kUlBatchNameFmt, ctx->batch_id + 1)).out;
    *end = '\0';

    const auto ss = std::bit_cast<std::array<uint16_t, 2>>(ctx->slot_u32);
    const uint16_t sfn = ss[0];
    const uint16_t slot = ss[1];
    const TaskInstrumentationContext ti_ctx(
        TracingMode::FULL_TRACING,
        static_cast<uint64_t>(ctx->ring_idx),
        sfn,
        slot);

    TaskInstrumentation ti(ti_ctx, name_buf, 4);
    ti.add("Start Task");
    ti.add("CPlane Prepare");

    // Per-cell err list, stack-local, mirrors legacy
    // task_function_ul_aggr.cpp:1334-1335 (cplane_tx_err_cell_idx_list).
    // Sized at nv::CPLANE_BATCH_CELL_LIMIT (>= MAX_CELLS_PER_SLOT, so all
    // batch cells fit). Re-packed into the SCF direction-specific bound
    // (UL_MAX_CELLS_PER_SLOT / DL_MAX_CELLS_PER_SLOT) inside the handlers.
    std::array<uint32_t, nv::CPLANE_BATCH_CELL_LIMIT> err_cell_ids{};
    int err_count = 0;

    for (uint8_t i = 0; i < ctx->n_cells; ++i)
    {
        const auto cid = static_cast<uint32_t>(ctx->cell_ids[i]);
        const int ret = ctx->process_cell(ctx->phy_module,
                                          ctx->ring_idx,
                                          cid,
                                          ctx->transaction_id);
        if (ret != 0) {
            err_cell_ids[static_cast<std::size_t>(err_count++)] = cid;
        }
    }
    ti.add("Signal Completion");

    // Snapshot ctx before signalling completion. on_complete() releases this ring slot
    // once tasks_in_flight reaches 0, after which the message thread may re-arm the slot
    // and overwrite *ctx (the batch arg lives in the ring's batch-arg pool). The error
    // fan-out and slot-end signal below must therefore touch only these locals.
    auto* const    phy_module     = ctx->phy_module;
    const uint32_t ring_idx       = ctx->ring_idx;
    const uint32_t slot_u32       = ctx->slot_u32;
    const auto     on_complete    = ctx->on_complete;
    void* const    slot_map       = ctx->slot_map;
    const auto     on_batch_error = ctx->on_batch_error;
    const auto     post_batch     = ctx->post_batch;
    const uint8_t  batch_id       = ctx->batch_id;
    const uint8_t  total_batches  = ctx->total_batches;
    const uint8_t  n_cells        = ctx->n_cells;
    const auto     direction      = ctx->direction;

    auto notify_batch_error = [&]() {
        if (err_count > 0 && on_batch_error != nullptr) {
            on_batch_error(slot_map,
                           std::span<const uint32_t>{err_cell_ids.data(),
                                                     static_cast<std::size_t>(err_count)});
        }
    };

    const bool is_ul_direction = direction == nv::CplaneBatchDirection::UPLINK;
    if (is_ul_direction) {
        if (post_batch != nullptr) {
            post_batch(slot_map, batch_id, total_batches, n_cells, direction);
        }
        // UL consumers wait on addULCTasksComplete(). If the C-plane batch failed,
        // mark the slot-map aborted before releasing the L2A task counter; otherwise
        // on_complete() can publish the slot and start UL order consumers first.
        notify_batch_error();
        on_complete(phy_module, ring_idx, slot_u32);
    }
    else {
        // DL keeps the historical ordering because post_batch only signals slot-end
        // state there, and changing it perturbs DL/PDSCH behavior.
        // Invariant: post_batch runs for error batches too (err_count is not consulted),
        // so every pushed DL batch posts its incUplanePrepDone contribution and the
        // U-plane TX task's waitUplanePrepDone is released by construction — a failed
        // C-plane batch reports via on_batch_error but never strands the wire TX.
        on_complete(phy_module, ring_idx, slot_u32);
        notify_batch_error();
        if (post_batch != nullptr) {
            post_batch(slot_map, batch_id, total_batches, n_cells, direction);
        }
    }

    ti.add("End Task");
    return 0;
}
