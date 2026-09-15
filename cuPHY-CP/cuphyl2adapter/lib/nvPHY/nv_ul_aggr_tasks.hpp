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

#ifndef NV_UL_AGGR_TASKS_HPP
#define NV_UL_AGGR_TASKS_HPP

#include "nv_aggr_task_common.hpp"
#include "nv_cplane_batch_tasks.hpp"
#include "nv_fapi_message_storage.hpp" // FapiSlotMessageStorage, phy_mac_msg_desc
#include "nv_bfw_dispatch.hpp"         // dispatch_bfw_messages
#include <cstdint>
#include <type_traits>
#include <span>

class Worker;
class SlotMapUl;

namespace nv
{

// Forward declaration needed by the dispatch table aliases below.
struct UlAggrTaskArg;

/**
 * @brief Bitmask of active UL channel types at EOM.
 *
 * Bit positions match UL_TTI_NPDUS_IDX_* values from scf_5g_fapi.h:
 *   PRACH = 0, PUSCH = 1, PUCCH = 2, SRS = 4.
 * Index [3] (PUCCH 2/3/4) is merged into PUCCH (bit 2), and
 * index [5] (MsgA-PUSCH) is always zero by contract; neither is tested
 * when building this mask.
 */
enum class UlChMask : std::uint8_t
{
    NONE  = 0,
    PRACH = 1U << 0, ///< UL_TTI_NPDUS_IDX_PRACH    = 0
    PUSCH = 1U << 1, ///< UL_TTI_NPDUS_IDX_PUSCH    = 1
    PUCCH = 1U << 2, ///< UL_TTI_NPDUS_IDX_PUCCH_F01 = 2 (F234 merged here)
    SRS   = 1U << 4, ///< UL_TTI_NPDUS_IDX_SRS      = 4
};

// ---------------------------------------------------------------------------
// Function pointer aliases for the UL dispatch table.
// Project rule: `using Alias = R(*)(Args...)` — never typedef, never add_pointer_t.
// ---------------------------------------------------------------------------

/// Signature for per-channel UL processing functions (e.g. process_aggr_pusch_channel).
using ul_channel_fn_t  = void(*)(PHY_module*, const UlAggrTaskArg&);

/// Backward-compatible alias — prefer task_complete_fn_t from nv_cplane_batch_tasks.hpp.
using ul_complete_fn_t = task_complete_fn_t;

/**
 * @brief Static dispatch table embedded in UlAggrTaskArg.
 *
 * PHY_module populates this at EOM enqueue time (pointing to its private
 * UL_DISPATCH_TABLE). Worker threads call through it so that all
 * process_aggr_*_channel methods and the shared completion callback
 * remain private inside PHY_module.
 *
 * All members are non-null when properly initialised.
 */
struct UlChannelDispatch final
{
    ul_channel_fn_t    process_pusch  = nullptr; ///< → PHY_module::process_aggr_pusch_channel
    ul_channel_fn_t    process_prach  = nullptr; ///< → PHY_module::process_aggr_prach_channel
    ul_channel_fn_t    process_pucch  = nullptr; ///< → PHY_module::process_aggr_pucch_channel (format 0/1 only)
    ul_channel_fn_t    process_srs    = nullptr; ///< → PHY_module::process_aggr_srs_channel
    ul_channel_fn_t    process_ulbfw  = nullptr; ///< → PHY_module::process_aggr_ulbfw_channel
    task_complete_fn_t on_complete    = nullptr; ///< → PHY_module::on_slot_channel_task_complete (via su_done shim)
};

/**
 * @brief Per-cell data for one UL C-plane cell.
 *
 * Stored in SlotTaskPool::ulc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cell_id].
 * Populated by accumulate_ulc_for_cell(); read by PHY_module::su_process_ulc_cell()
 * inside task_work_fn_cplane_batch.  Lifetime: valid from accumulate time until
 * reset_for_slot() reclaims the ring slot (guaranteed only after tasks_in_flight == 0).
 */
struct UlcTaskArg final
{
    PHY_instance*     phy_inst     = nullptr; ///< Direct pointer to the per-cell PHY instance; resolved at
                                              ///< accumulate time via PHY_instances()[cell_id].get().
    phy_mac_msg_desc* ul_tti       = nullptr; ///< Pointer into ring buffer UL_TTI_REQ lane
    SlotMapUl*        slot_map_ul  = nullptr; ///< Early-enqueued UL SlotMap (set when isFapiToCplaneDirect)
};

/**
 * @brief Shared context passed to all EOM UL channel aggregation tasks.
 *
 * All four EOM UL channel aggr tasks for the same slot share one instance.
 * Read-only from task work functions — never written after EOM enqueue.
 * Points into ring buffer storage — valid until all channel tasks complete.
 */
struct UlAggrTaskArg final
{
    TaskArgBase              base{TaskArgType::UL_AGGR};              ///< Type sentinel — verified by checked_cast_ul_aggr
    PHY_module*              phy_module        = nullptr;           ///< Back-pointer to access phy_refs_, slot_command_array
    const UlChannelDispatch* dispatch          = nullptr;           ///< Points to PHY_module::UL_DISPATCH_TABLE; set at EOM
    uint32_t                 slot_u32          = 0;                 ///< Packed SFN/slot for slot_message_storage() lookup
    uint32_t                 ring_idx          = 0;                 ///< slot_index % SLOT_STORAGE_DEPTH
    uint32_t                 active_ul_ch_mask = 0;                 ///< Bitmask of active UL channel types (UlChMask bits)
    uint32_t                 slot_cmd_idx      = 0;                 ///< slot_command_array index captured at EOM before
                                                                    ///< update_slot_cmds_indexes() is called.
                                                                    ///< IMPORTANT: access slot_command_array[slot_cmd_idx] directly —
                                                                    ///< do NOT use group_command() / cell_sub_command() (already advanced).
    SlotTypeMask             slot_type_mask    = SlotTypeMask::NONE; ///< OR of SlotTypeMask::{DL,UL,CSIRS} bits, snapshotted at EOM
};

/// Type-safe cast from void* to UlAggrTaskArg*.
///
/// Reads the type tag through a pointer-interconvertible TaskArgBase*
/// (well-defined for standard-layout structs per [basic.compound]/4)
/// before down-casting to the concrete type.  Returns nullptr on any
/// mismatch so callers can return a non-zero error code rather than
/// dereferencing a bad pointer.
[[nodiscard]] inline UlAggrTaskArg* checked_cast_ul_aggr(void* arg) noexcept
{
    static_assert(std::is_standard_layout_v<UlAggrTaskArg>,
                  "UlAggrTaskArg must be standard-layout for TaskArgBase cast");
    if (arg == nullptr) { return nullptr; }
    auto* base = reinterpret_cast<TaskArgBase*>(arg);
    if (base->type_tag != TaskArgType::UL_AGGR) { return nullptr; }
    return static_cast<UlAggrTaskArg*>(arg);
}

/**
 * Dispatch stored UL_BFW lane messages to per-cell handlers with bounds checking.
 *
 * @param slot_store         Slot storage containing the UL_BFW lane.
 * @param num_phy_instances  Upper bound for valid cell_id (exclusive).
 * @param per_cell           Callable void(int cell_id, const phy_mac_msg_desc& msg).
 * @param slot_u32           Packed SFN/slot for logging.
 * @param ring_idx           Ring index for logging.
 */
template<typename Fn>
void dispatch_ul_bfw_messages(const FapiSlotMessageStorage& slot_store,
                              std::size_t num_phy_instances,
                              Fn&&        per_cell,
                              uint32_t    slot_u32,
                              uint32_t    ring_idx)
{
    dispatch_bfw_messages(
        std::span<const phy_mac_msg_desc>{slot_store.ul_bfw_messages(),
                                          slot_store.ul_bfw_count()},
        num_phy_instances, std::forward<Fn>(per_cell),
        slot_u32, ring_idx, "dispatch_ul_bfw");
}
 
} // namespace nv

// ---------------------------------------------------------------------------
// Task work function declarations
// Signature matches task_work_function: int f(Worker*, void*, int, int, int)
// [[nodiscard]] is for direct-call documentation only —
// not enforced when invoked through l1_task_work_fn_t.
// ---------------------------------------------------------------------------

/// EOM UL channel aggregation tasks — called in parallel after global EOM.
[[nodiscard]] int task_work_fn_aggr_pusch(Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_prach(Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_pucch(Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_srs  (Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_ulbfw(Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);

#endif // NV_UL_AGGR_TASKS_HPP
