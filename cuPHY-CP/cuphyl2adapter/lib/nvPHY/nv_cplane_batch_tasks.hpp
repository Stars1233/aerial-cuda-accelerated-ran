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

#ifndef NV_CPLANE_BATCH_TASKS_HPP
#define NV_CPLANE_BATCH_TASKS_HPP

#include <cstdint>
#include <span>

class Worker;

namespace nv
{

class PHY_module;

// ---------------------------------------------------------------------------
// Direction-neutral types shared by DL and UL batched C-plane tasks.
// ---------------------------------------------------------------------------

/// Signature for the shared slot task-completion callback (on_slot_channel_task_complete).
/// Used by both DL and UL dispatch tables and CplaneBatchTaskArg::on_complete.
using task_complete_fn_t = void(*)(PHY_module*, uint32_t ring_idx, uint32_t slot_u32);

/// Upper bound for CplaneBatchTaskArg::cell_ids[].
/// Must be >= MAX_CELLS_PER_SLOT (40).  Verified by static_assert in nv_phy_module.hpp.
inline constexpr uint8_t CPLANE_BATCH_CELL_LIMIT = 64;

/// Per-cell work function type: build cplane for one cell.
/// Receives ring_idx so the implementation can index ring-local storage.
/// Receives transaction_id so each concurrent batch maps to a unique
/// CPlaneGenerator transaction workspace.
/// Returns SEND_CPLANE_NO_ERROR on success or a non-zero error code on failure
/// (e.g., SEND_UL_CPLANE_TIMING_ERROR from framework_cplane_service timing gate).
/// The dispatcher accumulates failing cell_ids for batch-end fan-out via on_batch_error.
/// DL: PHY_module::s_process_dlc_cell.  UL: PHY_module::su_process_ulc_cell.
using cplane_process_cell_fn_t =
    int(*)(PHY_module*, uint32_t ring_idx, uint32_t cell_id, std::size_t transaction_id);

/// Direction-specific batch-end error handler.
/// Invoked by task_work_fn_cplane_batch when one or more cells in the batch
/// reported a non-zero return from process_cell. Mirrors legacy cleanup:
/// L2 error notification via *_tx_error_fn + slot_map->abortTasks().
/// @param slot_map      ctx->slot_map (SlotMapUl* for UL, SlotMapDl* for DL)
/// @param err_cell_ids  failing cell IDs from the dispatcher's stack-local array
/// DL: l1_handle_dl_cplane_batch_error.  UL: l1_handle_ul_cplane_batch_error.
using cplane_on_batch_error_fn_t =
    void(*)(void* slot_map, std::span<const uint32_t> err_cell_ids);

enum class CplaneBatchDirection : uint8_t
{
    DOWNLINK = 0U,
    UPLINK   = 1U,
};

/// Post-batch signaling callback — invoked once after all cells in a batch
/// have been processed.
/// @param slot_map         SlotMapDl* (DL) or SlotMapUl* (UL), may be nullptr.
/// @param batch_id         0-based batch index for this direction+slot.
/// @param total_batches    Total planned batches for this direction+slot.
/// @param n_cells_in_batch Number of cells processed by this batch.
/// @param direction        Direction marker (DOWNLINK or UPLINK).
using post_batch_fn_t = void(*)(void* slot_map,
                                uint8_t batch_id,
                                uint8_t total_batches,
                                uint8_t n_cells_in_batch,
                                CplaneBatchDirection direction);

/**
 * @brief Shared arg for one batched DLC or ULC task.
 *
 * One instance covers up to cplane_batch_size cells.  Stored in
 * SlotTaskPool::dlc_batch_task_args[] or ulc_batch_task_args[].
 * Index: ring_idx * MAX_CELLS_PER_SLOT + batch_count.
 *
 * cell_ids[] is filled during accumulation (accumulate_dlc/ulc_for_cell).
 * All other fields are filled by fire_cplane_batch<> at fire time.
 * Read-only after the task is pushed to the worker queue.
 */
struct CplaneBatchTaskArg final
{
    PHY_module*               phy_module;    ///< Back-pointer to PHY_module
    uint32_t                  ring_idx;      ///< slot_index % SLOT_STORAGE_DEPTH
    uint32_t                  slot_u32;      ///< Packed SFN/slot
    uint8_t                   n_cells;       ///< Number of cells in this batch
    uint8_t                   cell_ids[CPLANE_BATCH_CELL_LIMIT]; ///< Cells in batch
    cplane_process_cell_fn_t  process_cell;  ///< Per-cell: build cplane + clear guard (now returns int)
    task_complete_fn_t        on_complete;   ///< → PHY_module::on_slot_channel_task_complete
    void*                     slot_map{};    ///< SlotMapDl* (DL) or SlotMapUl* (UL)
    post_batch_fn_t           post_batch{};  ///< Deferred signaling after all cells complete
    cplane_on_batch_error_fn_t on_batch_error{}; ///< Direction-specific batch-end error fan-out (may be null)
    CplaneBatchDirection      direction{CplaneBatchDirection::DOWNLINK}; ///< Direction set by fire_cplane_batch<Policy>
    uint8_t                   batch_id{0};   ///< 0-based batch index for this slot+direction
    uint8_t                   total_batches{1}; ///< Planned total batches for this slot+direction
    std::size_t               transaction_id{0}; ///< Framework C-plane transaction id for this batch
};

/**
 * @brief Per-ring-slot accumulation state for batched DLC or ULC task enqueue.
 *
 * Tracks how many cells have been accumulated into the current in-progress
 * batch and how many batch tasks have already been submitted for this ring slot.
 * Reset by reset_for_slot() at each new slot.
 */
struct CplaneBatchAccum final
{
    uint8_t n_pending{0};      ///< cells accumulated, not yet fired as a task
    uint8_t batch_count{0};    ///< batch tasks submitted so far for this ring slot
    void reset() noexcept { n_pending = 0; batch_count = 0; }
};

} // namespace nv

// ---------------------------------------------------------------------------
// Task work function declaration
// Signature matches task_work_function: int f(Worker*, void*, int, int, int)
// [[nodiscard]] is for direct-call documentation only —
// not enforced when invoked through l1_task_work_fn_t.
// ---------------------------------------------------------------------------

/// Batched DLC or ULC task — processes CplaneBatchTaskArg::n_cells cells in one worker invocation.
/// process_cell function pointer in arg encodes DL vs UL direction.
[[nodiscard]] int task_work_fn_cplane_batch(Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);

#endif // NV_CPLANE_BATCH_TASKS_HPP
