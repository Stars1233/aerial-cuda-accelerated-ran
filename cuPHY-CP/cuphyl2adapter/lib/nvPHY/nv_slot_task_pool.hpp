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

/**
 * @file nv_slot_task_pool.hpp
 * @brief Lightweight header providing SlotTaskPool, SLOT_STORAGE_DEPTH, and
 *        slot_index_from_u32() without pulling in the full nv_phy_module.hpp
 *        include chain (cuphydriver, CUDA, gRPC, yaml, …).
 *
 * Intended for:
 *   - nv_phy_module.hpp  (canonical consumer — includes this header)
 *   - Unit-tests and benchmarks that need the task pool types without the
 *     full PHY_module dependency chain
 *
 * All heavy headers (app_config.hpp, nv_phy_instance.hpp,
 * slot_command/slot_command.hpp, cuphydriver_api.hpp, …) are intentionally
 * absent so this header can be included in isolated test builds.
 */

#ifndef NV_SLOT_TASK_POOL_HPP
#define NV_SLOT_TASK_POOL_HPP

#include "nv_cplane_batch_tasks.hpp"  // CplaneBatchTaskArg, CplaneBatchAccum
#include "nv_dl_aggr_tasks.hpp"       // DlcTaskArg, DlAggrTaskArg
#include "nv_ul_aggr_tasks.hpp"       // UlcTaskArg, UlAggrTaskArg
#include "nv_ipc_utils.h"             // SFN_SLOT_INVALID

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace nv {

// ---------------------------------------------------------------------------
// Ring-buffer depth
// ---------------------------------------------------------------------------

/// Number of simultaneously live slots tracked in the ring buffer.
inline constexpr std::size_t SLOT_STORAGE_DEPTH = 3;
static_assert(SLOT_STORAGE_DEPTH > 0, "SLOT_STORAGE_DEPTH must be positive");

// ---------------------------------------------------------------------------
// SlotTaskPool
// ---------------------------------------------------------------------------

/**
 * @brief Cohesive bundle of per-slot task scheduling state for one PHY_module instance.
 *
 * Groups the ring-indexed in-flight counters, task argument pools, and per-cell
 * DLC/ULC guards. PHY_module holds exactly one instance as @c task_pool_.
 *
 * A single @c tasks_in_flight sentinel counter covers all tasks for a slot
 * (DLC + ULC + DL channel aggr + UL channel aggr) so that exactly one
 * @c l1_enqueue_phy_work() call fires when all tasks complete, correctly
 * handling both DL-only slots and Special-S mixed DL+UL slots.
 *
 * Note: worker IDs for cache-warm core co-location (pdsch_worker_id_,
 * pusch_worker_id_) are kept as members of PHY_module directly to avoid a
 * cuphydriver_api.hpp dependency in this lightweight header.
 */
struct SlotTaskPool final
{
    /// Sentinel loaded into tasks_in_flight[] at slot start to hold the counter
    /// above zero until EOM. Chosen as 2^30 (~1 billion) — far beyond any
    /// realistic task count — so it can never be reached by real increments,
    /// making premature zero-crossing impossible.
    static constexpr int32_t SLOT_TASK_SENTINEL = 0x4000'0000;

    /// Ring-indexed counter tracking all in-flight tasks (DLC batches + ULC batches +
    /// DL/UL channel aggr) for a slot. Index = slot_index % SLOT_STORAGE_DEPTH.
    ///
    /// Lifecycle per slot:
    ///   1. Armed with SLOT_TASK_SENTINEL by reset_for_slot() at slot start.
    ///   2. Incremented by +1 per DLC batch fired (each batch covers
    ///      1..cplane_processing_dl_batch_size cells at EOM, or all accumulated
    ///      DL cells when the configured size is invalid).
    ///   3. Incremented by +1 per ULC batch fired (same rule with
    ///      cplane_processing_ul_batch_size).
    ///   4. At EOM: fire_cplane_batch flushes incomplete batches (< batch_size cells);
    ///      then fetch_add(n_dl_ch_tasks + n_ul_ch_tasks) and fetch_sub(SLOT_TASK_SENTINEL)
    ///      atomically replaces the sentinel with the real total task count.
    ///   5. Each task completion decrements by 1; reaching zero triggers
    ///      publish_slot_command() (channel-done) and finalize_slot_cleanup() (all-done).
    std::array<std::atomic<int32_t>, SLOT_STORAGE_DEPTH> tasks_in_flight{};

    /// Per-ring count of channel-aggr tasks ONLY (not C-plane batches). Same sentinel
    /// lifecycle as tasks_in_flight: armed with SLOT_TASK_SENTINEL by reset_for_slot(),
    /// += n_channel at channel push, sentinel dropped at EOM. Reaching zero (all channel
    /// tasks done) triggers publish_slot_command() — independent of C-plane completion.
    /// ONLY channel-aggr task completion (on_slot_channel_aggr_complete) decrements this;
    /// C-plane batch completion must never touch it (would re-fire publish).
    std::array<std::atomic<int32_t>, SLOT_STORAGE_DEPTH> channel_tasks_in_flight{};

    /// True iff this ring slot holds a pure sentinel -- armed by reset_for_slot() (step 1)
    /// but its EOM has not fired. Step 4's fetch_sub(SLOT_TASK_SENTINEL) is the only writer
    /// that displaces the sentinel, so == SLOT_TASK_SENTINEL uniquely means "armed, nothing
    /// fired" (vs > sentinel = batches in flight pre-EOM, or < sentinel = fired/draining).
    /// Centralizes the magic-value comparison so call sites do not repeat it.
    /// @param[in] ring_idx Ring slot index (slot_index % SLOT_STORAGE_DEPTH).
    /// @return true if the ring slot is armed and its EOM has not fired.
    [[nodiscard]] bool is_armed_unfired(uint32_t ring_idx) const noexcept
    {
        return tasks_in_flight[ring_idx].load(std::memory_order_acquire) == SLOT_TASK_SENTINEL;
    }

    /// Same predicate for an already-loaded counter value (call sites that loaded it for
    /// other checks or logging), so the SLOT_TASK_SENTINEL comparison stays in one place.
    /// @param[in] in_flight A previously loaded tasks_in_flight value.
    /// @return true if @p in_flight is the pure sentinel.
    [[nodiscard]] static bool is_sentinel(int32_t in_flight) noexcept
    {
        return in_flight == SLOT_TASK_SENTINEL;
    }

    /// DLC task arguments indexed by ring slot: [ring_idx * MAX_CELLS_PER_SLOT + cell_id].
    /// Using ring-local storage eliminates cross-slot aliasing — slot N and slot N+3
    /// (same ring_idx) write to different rows, so no per-cell in-flight guard is needed.
    std::array<DlcTaskArg, SLOT_STORAGE_DEPTH * MAX_CELLS_PER_SLOT> dlc_task_args{};

    /// Bitmap of cells accumulated into a DLC batch for this ring slot.
    /// Bit N is set when cell N's SLOT.resp has been accepted into the batch.
    /// Reset by reset_for_slot() — which is only ever called from the msg_processing
    /// thread, matching the writers (accumulate_dlc/ulc_for_cell). No atomics needed.
    /// uint64_t supports up to 64 cells; MAX_CELLS_PER_SLOT must stay ≤ 64.
    static_assert(MAX_CELLS_PER_SLOT <= 64,
                  "dlc/ulc_accumulated_bitmap is uint64_t - MAX_CELLS_PER_SLOT must not exceed 64");
    std::array<uint64_t, SLOT_STORAGE_DEPTH> dlc_accumulated_bitmap{};

    /// ULC task arguments indexed by ring slot: [ring_idx * MAX_CELLS_PER_SLOT + cell_id].
    /// Mirrors dlc_task_args — see above for rationale.
    std::array<UlcTaskArg, SLOT_STORAGE_DEPTH * MAX_CELLS_PER_SLOT> ulc_task_args{};

    /// Bitmap of cells accumulated into a ULC batch for this ring slot.
    /// Mirrors dlc_accumulated_bitmap for the UL path.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> ulc_accumulated_bitmap{};

    /// Shared argument for all EOM DL channel aggr tasks — one per ring slot.
    std::array<DlAggrTaskArg, SLOT_STORAGE_DEPTH> dl_aggr_task_args{};

    /// Shared argument for all EOM UL channel aggr tasks — one per ring slot.
    std::array<UlAggrTaskArg, SLOT_STORAGE_DEPTH> ul_aggr_task_args{};

    // -----------------------------------------------------------------------
    // Batched DLC/ULC task infrastructure
    // Index convention: ring_idx * MAX_CELLS_PER_SLOT + batch_count
    // Worst case (batch_size=1): one batch per cell → MAX_CELLS_PER_SLOT entries per ring slot.
    // -----------------------------------------------------------------------

    /// Batch arg pool for DLC (DL C-plane) batched tasks.
    std::array<CplaneBatchTaskArg, SLOT_STORAGE_DEPTH * MAX_CELLS_PER_SLOT> dlc_batch_task_args{};

    /// Batch arg pool for ULC (UL C-plane) batched tasks.
    std::array<CplaneBatchTaskArg, SLOT_STORAGE_DEPTH * MAX_CELLS_PER_SLOT> ulc_batch_task_args{};

    /// Per-ring-slot DLC accumulation state — reset by reset_for_slot().
    std::array<CplaneBatchAccum, SLOT_STORAGE_DEPTH> dlc_batch_accum{};

    /// Per-ring-slot ULC accumulation state — reset by reset_for_slot().
    std::array<CplaneBatchAccum, SLOT_STORAGE_DEPTH> ulc_batch_accum{};

    /// Per-ring-slot EOM bitmap for Path B (ENABLE_FAPI_STORE_REPLAY).
    /// Tracks which cells have delivered SLOT.resp for the ring slot's active slot.
    /// Indexed by ring_idx derived from ss_msg (not ss_curr) — immune to ss_curr
    /// rollback races.  Cleared in reset_for_slot() and inline after EOM fires.
    /// Path A uses the separate fapi_eom_rcvd_bitmap (uint64_t) — these never interact.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> store_eom_bitmap{};

    /// Committed active-cell bitmap snapshotted when a ring slot is armed.
    /// The EOM gate uses this stable snapshot instead of live active-cell state
    /// so mid-slot START/STOP changes cannot affect the current slot's mask.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> active_cell_bitmap_snapshot{};

    /// Most recently finalized (submitted or released) slot per ring index. A SLOT.resp whose
    /// slot equals this must NOT re-arm the ring -- it is a straggler for an already-finalized
    /// slot (e.g. a deferred-commit cell's first produced slot arriving after the established
    /// cells' EOM already closed the slot); re-arming would re-finalize and double-submit it.
    /// Written only at finalize and read only at the SLOT.resp arm check, both on the
    /// msg-processing thread -- so plain (not atomic). SFN_SLOT_INVALID means "none yet".
    std::array<uint32_t, SLOT_STORAGE_DEPTH> last_finalized_slot = []
    {
        std::array<uint32_t, SLOT_STORAGE_DEPTH> init{};
        init.fill(SFN_SLOT_INVALID);
        return init;
    }();

    /// Popcount-derived active-cell count paired with active_cell_bitmap_snapshot.
    /// Reserved for read sites that need a stable per-slot active-cell count.
    std::array<uint8_t, SLOT_STORAGE_DEPTH> active_cell_count_snapshot{};

    /// Timestamp of first FAPI payload message arrival per ring slot (nanoseconds since epoch).
    /// Set in store_slot_message() on the first payload msg for the slot; reset by reset_for_slot().
    std::array<std::chrono::nanoseconds, SLOT_STORAGE_DEPTH> fapi_first_arrival_ts{};

    /// Set after process_phy_commands(true) returns in enqueue_channel_tasks (msg thread).
    /// Checked in on_slot_channel_task_complete() to detect ordering violations.
    std::array<std::atomic<bool>, SLOT_STORAGE_DEPTH> enqueue_phy_work_done{};

    // --- Per-ring drop diagnostics ---------------------------------------------
    // All counters and slot markers below are plain (not atomic): they are written
    // and read only on the msg-processing thread (reset_for_slot / store_slot_message /
    // enqueue_channel_tasks). Concurrent access from worker or other threads is
    // forbidden unless these are converted to atomics or mutex-guarded.

    /// Consecutive SLOT.ind resets that found live tasks still using the ring slot.
    /// The msg-processing thread owns updates; used to allow short startup/latency slips
    /// while preserving a fatal escape hatch for a permanently stuck task.
    std::array<uint32_t, SLOT_STORAGE_DEPTH> ring_reuse_consecutive_drops{};

    /// Total slot-boundary ring-reuse drops per ring slot for diagnostics.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> ring_reuse_total_drops{};

    /// Total payload messages dropped because their ring slot was still owned by
    /// live tasks from an older slot.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> ring_reuse_payload_drops{};

    /// Consecutive direct C-plane slots dropped because their FAPI payloads
    /// arrived too late to satisfy C-plane send timing.
    std::array<uint32_t, SLOT_STORAGE_DEPTH> direct_cplane_stale_slot_drops{};

    /// Total direct C-plane stale-slot drops per ring slot for diagnostics.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> direct_cplane_total_stale_drops{};

    /// Direct C-plane slot already dropped as stale at payload ingress. Subsequent
    /// payloads for the same slot are released before they can enter slot storage
    /// or TX_DATA H2D staging.
    std::array<uint32_t, SLOT_STORAGE_DEPTH> direct_cplane_stale_dropped_slot = []
    {
        std::array<uint32_t, SLOT_STORAGE_DEPTH> init{};
        init.fill(SFN_SLOT_INVALID);
        return init;
    }();

    /// Total payload messages released because their direct C-plane slot was
    /// already classified stale at ingress.
    std::array<uint64_t, SLOT_STORAGE_DEPTH> direct_cplane_stale_payload_drops{};

    /// Incoming slot dropped at SLOT.ind because the ring was still occupied by
    /// live tasks. Payload messages for this same slot are dropped too so a
    /// partial slot cannot be rebound after the older task drains.
    std::array<uint32_t, SLOT_STORAGE_DEPTH> ring_reuse_dropped_slot = []
    {
        std::array<uint32_t, SLOT_STORAGE_DEPTH> init{};
        init.fill(SFN_SLOT_INVALID);
        return init;
    }();

    /// Slot whose payload failed to store. Distinguishes empty vs degraded at EOM
    /// (both leave storage unbound). Set on store failure; cleared (slot-conditional)
    /// on both EOM finalize paths -- EOM-complete and the SLOT.ind backstop
    /// (finalize_slot_on_boundary). Deliberately NOT cleared on reset_for_slot, which
    /// runs at SLOT.resp *before* classification: the marker must outlive it so EOM
    /// can still see the drop. Reads compare the stored value against the exact
    /// current slot (u32 equality), so a stale entry from a prior ring occupant never
    /// matches -- that slot-keyed equality is the load-bearing invariant that keeps
    /// "empty" vs "degraded" distinct.
    std::array<uint32_t, SLOT_STORAGE_DEPTH> payload_dropped_slot = []
    {
        std::array<uint32_t, SLOT_STORAGE_DEPTH> init{};
        init.fill(SFN_SLOT_INVALID);
        return init;
    }();
};

// ---------------------------------------------------------------------------
// slot_index_from_u32
// ---------------------------------------------------------------------------

/**
 * @brief Convert a packed slot_u32 value into a linear (monotonic) slot index.
 *
 * The packed representation stores the SFN in the upper 16 bits and the
 * within-frame slot number in the lower 16 bits:
 * @code
 *   slot_u32 = (sfn << 16) | slot_in_frame
 * @endcode
 * The linear index is then: @c sfn * slot_per_frame + slot_in_frame.
 *
 * @note When @p slot_u32 equals @c SFN_SLOT_INVALID the function returns 0
 *       rather than producing an undefined result, so callers can safely pass
 *       an uninitialised slot handle without a prior validity check.
 *
 * @param slot_u32       Packed SFN/slot value as returned by nv_ipc_get_sfn_slot().
 * @param slot_per_frame Number of slots per radio frame (numerology-dependent,
 *                       e.g. 20 for µ=1).
 * @return Linear slot index (@c sfn * @p slot_per_frame + @c slot_in_frame),
 *         or 0 if @p slot_u32 is @c SFN_SLOT_INVALID.
 */
[[nodiscard]] constexpr uint32_t slot_index_from_u32(const uint32_t slot_u32,
                                                      const uint32_t slot_per_frame) noexcept
{
    return (slot_u32 == SFN_SLOT_INVALID)
        ? 0U
        : ((slot_u32 >> 16U) & 0xFFFFU) * slot_per_frame + (slot_u32 & 0xFFFFU);
}

}  // namespace nv

#endif // NV_SLOT_TASK_POOL_HPP
