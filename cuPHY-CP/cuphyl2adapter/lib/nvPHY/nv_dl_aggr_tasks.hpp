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

#ifndef NV_DL_AGGR_TASKS_HPP
#define NV_DL_AGGR_TASKS_HPP

#include "nv_aggr_task_common.hpp"
#include "nv_cplane_batch_tasks.hpp"
#include "nv_fapi_message_storage.hpp"   // FapiSlotMessageStorage, phy_mac_msg_desc
#include "nv_bfw_dispatch.hpp"           // dispatch_bfw_messages
#include "nvlog.hpp"                     // NVLOGD_FMT, NVLOGW_FMT
#include <cstdint>
#include <type_traits>
#include <span>

class Worker;
class SlotMapDl;

namespace nv
{

// Forward declaration needed by the dispatch table aliases below.
struct DlAggrTaskArg;

/// Signature for per-channel processing functions (e.g. process_aggr_pdsch_channel).
using dl_channel_fn_t  = void(*)(PHY_module*, const DlAggrTaskArg&);

/// Backward-compatible alias — prefer task_complete_fn_t from nv_cplane_batch_tasks.hpp.
using dl_complete_fn_t = task_complete_fn_t;

/**
 * @brief Static dispatch table embedded in DlAggrTaskArg.
 *
 * PHY_module populates this at EOM enqueue time (pointing to its private
 * DL_DISPATCH_TABLE). Worker threads call through it so that all
 * process_aggr_*_channel methods and the completion callback remain
 * private inside PHY_module.
 *
 * All members are non-null when properly initialised.
 */
struct DlChannelDispatch final
{
    dl_channel_fn_t    process_pdsch  = nullptr; ///< → PHY_module::process_aggr_pdsch_channel
    dl_channel_fn_t    process_csirs  = nullptr; ///< → PHY_module::process_aggr_csirs_channel
    dl_channel_fn_t    process_pdcch  = nullptr; ///< → PHY_module::process_aggr_pdcch_channel
    dl_channel_fn_t    process_ssb    = nullptr; ///< → PHY_module::process_aggr_ssb_channel
    dl_channel_fn_t    process_dlbfw  = nullptr; ///< → PHY_module::process_aggr_dlbfw_channel
    task_complete_fn_t on_complete    = nullptr; ///< → PHY_module::on_slot_channel_task_complete
};

/**
 * @brief Per-cell data for one DL C-plane cell.
 *
 * Stored in SlotTaskPool::dlc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cell_id].
 * Populated by accumulate_dlc_for_cell(); read by PHY_module::s_process_dlc_cell()
 * inside task_work_fn_cplane_batch.  Lifetime: valid from accumulate time until
 * reset_for_slot() reclaims the ring slot (guaranteed only after tasks_in_flight == 0).
 */
struct DlcTaskArg final
{
    PHY_instance*     phy_inst     = nullptr; ///< Direct pointer to the per-cell PHY instance; resolved at
                                              ///< accumulate time via PHY_instances()[cell_id].get().
    phy_mac_msg_desc* dl_tti       = nullptr; ///< Pointer into ring buffer DL_TTI_REQ lane (null if absent)
    phy_mac_msg_desc* ul_dci       = nullptr; ///< Pointer into ring buffer UL_DCI_REQ lane (null if absent)
    SlotMapDl*        slot_map_dl  = nullptr; ///< Early-enqueued DL SlotMap (set when isFapiToCplaneDirect)
};

/**
 * @brief Shared context passed to all EOM DL channel aggregation tasks.
 *
 * All five EOM channel aggr tasks for the same slot share one instance.
 * Read-only from task work functions — never written after EOM enqueue.
 * Points into ring buffer storage — valid until all channel tasks complete.
 */
struct DlAggrTaskArg final
{
    TaskArgBase              base{TaskArgType::DL_AGGR};            ///< Type sentinel — verified by checked_cast_dl_aggr
    PHY_module*              phy_module     = nullptr;           ///< Back-pointer to access phy_refs_, slot_command_array
    const DlChannelDispatch* dispatch       = nullptr;           ///< Points to PHY_module::DL_DISPATCH_TABLE; set at EOM enqueue
    uint32_t                 slot_u32       = 0;                 ///< Packed SFN/slot for slot_message_storage() lookup
    uint32_t                 ring_idx       = 0;                 ///< slot_index % SLOT_STORAGE_DEPTH
    uint32_t                 active_ch_mask = 0;                 ///< Bitmask of active channel types (DL_TTI_PDU_TYPE_* bits)
    uint32_t                 slot_cmd_idx   = 0;                 ///< slot_command_array index captured at EOM before
                                                                 ///< update_slot_cmds_indexes() is called.
                                                                 ///< IMPORTANT: access slot_command_array[slot_cmd_idx] directly —
                                                                 ///< do NOT use group_command() / cell_sub_command() (already advanced).
    SlotTypeMask             slot_type_mask = SlotTypeMask::NONE; ///< OR of SlotTypeMask::{DL,UL,CSIRS} bits, snapshotted at EOM
    bool                     fapi_to_cplane_direct = false;       ///< Cached is_fapi_to_cplane_direct_enabled() — avoids repeated driver lookups from worker threads

    /// Frozen snapshot of staged TB GPU pointers set at EOM after Phase 2 launch.
    /// Points into PHY_module::txdata_staged_gpu_ptr_[ring_idx] — valid and immutable
    /// until all channel tasks for this slot complete.  nullptr when split-phase is inactive.
    uint8_t* const*          staged_tb_ptrs   = nullptr;
    uint32_t                 n_staged_tb_ptrs = 0;               ///< Array size (MAX_CELLS_PER_SLOT), for bounds checking

    /// Frozen snapshot of TX_DATA.req FAPI msg_buf pointers, indexed by cell_id.
    /// Used to extract per-UE tbStartOffset from TLV metadata in process_aggr_pdsch_channel.
    /// nullptr when split-phase is inactive or no TX_DATA was staged.
    const void* const*       staged_tx_msg_bufs = nullptr;
};

/// Type-safe cast from void* to DlAggrTaskArg*.
///
/// Reads the type tag through a pointer-interconvertible TaskArgBase*
/// (well-defined for standard-layout structs per [basic.compound]/4)
/// before down-casting to the concrete type.  Returns nullptr on any
/// mismatch so callers can return a non-zero error code rather than
/// dereferencing a bad pointer.
[[nodiscard]] inline DlAggrTaskArg* checked_cast_dl_aggr(void* arg) noexcept
{
    static_assert(std::is_standard_layout_v<DlAggrTaskArg>,
                  "DlAggrTaskArg must be standard-layout for TaskArgBase cast");
    if (arg == nullptr) { return nullptr; }
    auto* base = reinterpret_cast<TaskArgBase*>(arg);
    if (base->type_tag != TaskArgType::DL_AGGR) { return nullptr; }
    return static_cast<DlAggrTaskArg*>(arg);
}

/**
 * Dispatch stored DL_BFW lane messages to per-cell handlers with bounds checking.
 *
 * @param slot_store         Slot storage containing the DL_BFW lane.
 * @param num_phy_instances  Upper bound for valid cell_id (exclusive).
 * @param per_cell           Callable void(int cell_id, const phy_mac_msg_desc& msg).
 * @param slot_u32           Packed SFN/slot for logging.
 * @param ring_idx           Ring index for logging.
 */
template<typename Fn>
void dispatch_dl_bfw_messages(const FapiSlotMessageStorage& slot_store,
                              std::size_t num_phy_instances,
                              Fn&&        per_cell,
                              uint32_t    slot_u32,
                              uint32_t    ring_idx)
{
    dispatch_bfw_messages(
        std::span<const phy_mac_msg_desc>{slot_store.dl_bfw_messages(),
                                          slot_store.dl_bfw_count()},
        num_phy_instances, std::forward<Fn>(per_cell),
        slot_u32, ring_idx, "dispatch_dl_bfw");
}

} // namespace nv

// ---------------------------------------------------------------------------
// Task work function declarations
// Signature matches task_work_function: int f(Worker*, void*, int, int, int)
// [[nodiscard]] is for direct-call documentation only —
// not enforced when invoked through l1_task_work_fn_t.
// ---------------------------------------------------------------------------

/// EOM channel aggregation tasks — called in parallel after global EOM.
[[nodiscard]] int task_work_fn_aggr_pdsch (Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_csirs (Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_pdcch (Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_ssb   (Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);
[[nodiscard]] int task_work_fn_aggr_dlbfw (Worker* worker, void* arg, int first_cell, int num_cells, int num_tasks);

#endif // NV_DL_AGGR_TASKS_HPP
