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

// nv_phy_slot_dispatch.cpp
//
// Store-and-dispatch slot path for NvPhyModule.
// Compiled only when ENABLE_FAPI_STORE_REPLAY=ON (see CMakeLists.txt).
// No #ifdef ENABLE_FAPI_STORE_REPLAY guards inside this file.

#include "nv_phy_module.hpp"
#include "nv_l2a_task_tracing.hpp"
#include "nv_phy_driver_proxy.hpp"
#include "nv_scope_exit.hpp"
#include "aerial/casts/casts.hpp"
#include "scf_5g_fapi.h"
#include "scf_5g_fapi_msg_helpers.hpp"
#include "nv_dl_aggr_tasks.hpp"
#include "nv_ul_aggr_tasks.hpp"
#include "oran_utils/conversion.hpp"
#include <algorithm>
#include <task_instrumentation/task_instrumentation_v3.hpp>
#include "nv_tx_data_h2d_helpers.hpp"
#include "nv_fapi_message_storage.hpp"
#include "nv_fapi_pdu_utils.hpp"
#include "enum_utils.hpp"
#include "scf_5g_fapi_message_context.hpp"
#include <bit>
#include <cassert>
#include <cstddef>
#include <string_view>
#include <type_traits>

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 6) // "L2A.MODULE"
// Mirrors the canonical definition in nv_phy_module.cpp:34 — kept here so
// publish_slot_command can emit the same per-slot diagnostic line as
// process_phy_commands. Keep the two definitions in sync.
#define TAG_PROCESSING_TIMES (NVLOG_TAG_BASE_L2_ADAPTER + 11) // "L2A.PROCESSING_TIMES"

namespace nv
{

namespace {
constexpr uint32_t CH_UL_PUSCH = static_cast<uint32_t>(nv::UlChMask::PUSCH);
constexpr uint32_t CH_UL_PRACH = static_cast<uint32_t>(nv::UlChMask::PRACH);
constexpr uint32_t CH_UL_PUCCH = static_cast<uint32_t>(nv::UlChMask::PUCCH);
constexpr uint32_t CH_UL_SRS   = static_cast<uint32_t>(nv::UlChMask::SRS);

// Allow a short burst of ring-reuse drops for startup / mixed DL+UL latency slips,
// but keep a fatal escape hatch for a permanently stuck task.
constexpr uint32_t kMaxConsecutiveRingReuseDrops = 8U;

// Direct C-plane timing is checked against the slot-map reference timestamp.
// If FAPI store-replay delivers a startup slot tens of slots behind the current
// SLOT.ind, that slot cannot be recovered by workers; drop it before allocating
// C-plane slot maps. Keep a bounded guard so a permanent producer/consumer skew
// still fails loudly.
constexpr uint32_t kMaxDirectCplaneStaleSlotInterval = 1U;
constexpr uint32_t kMaxDirectCplaneConsecutiveStaleDrops = 80U;
constexpr uint16_t kStartupDirectCplaneWarmupSfns = 4U;

[[nodiscard]] bool is_startup_direct_cplane_warmup_slot(uint32_t slot_u32)
{
    const sfn_slot_t ss{.u32 = slot_u32};
    return ss.u16.sfn < kStartupDirectCplaneWarmupSfns;
}

[[nodiscard]] std::chrono::nanoseconds slot_dispatch_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
}
} // namespace

slot_command_api::slot_info_t*
PHY_module::ul_order_scratch_sym_prb_info(uint32_t ring_idx, uint32_t channel_idx, uint32_t cell_idx)
{
#ifdef ENABLE_FAPI_STORE_REPLAY
    if (ring_idx >= SLOT_STORAGE_DEPTH || channel_idx >= kUlOrderScratchChannels
        || cell_idx >= MAX_CELLS_PER_SLOT) [[unlikely]]
    {
        return nullptr;
    }

    // Every scratch entry is pre-allocated at init (PHY_module ctor) and re-armed per
    // slot by reset_ul_order_scratch, so the live worker path must not allocate here: a
    // make_unique would heap-alloc on the hot path and throw across
    // resolve_ul_order_scratch's noexcept boundary. Return the pre-allocated entry, or
    // nullptr if unexpectedly absent (callers null-check).
    return ul_order_scratch_[ring_idx][channel_idx][cell_idx].get();
#else
    static_cast<void>(ring_idx);
    static_cast<void>(channel_idx);
    static_cast<void>(cell_idx);
    return nullptr;
#endif
}

void PHY_module::prealloc_ul_order_scratch()
{
#ifdef ENABLE_FAPI_STORE_REPLAY
    // Construction-time allocation of every UL order-scratch entry, so the per-slot
    // reset_ul_order_scratch and worker-side ul_order_scratch_sym_prb_info paths never
    // allocate on live traffic. An entry that fails to allocate stays null; merge,
    // reset and resolve all treat a null entry as empty.
    for (auto& by_ring : ul_order_scratch_)
    {
        for (auto& by_cell : by_ring)
        {
            for (auto& scratch : by_cell)
            {
                if (scratch != nullptr)
                {
                    continue;
                }
                try
                {
                    scratch = std::make_unique<slot_command_api::slot_info_t>();
                }
                catch (...)
                {
                    // Fail fast: without every scratch entry the per-slot paths would hit a
                    // null entry, drop that cell/channel's UL order PRBs and wedge the order
                    // kernel at runtime - a construction-time abort is the clearer failure.
                    NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "prealloc_ul_order_scratch: failed to allocate a UL order scratch entry; cannot run UL ordering");
                }
            }
        }
    }
#endif
}

void PHY_module::clear_direct_cplane_task_args_for_ring(uint32_t ring_idx) noexcept
{
    if (ring_idx >= SLOT_STORAGE_DEPTH) [[unlikely]]
    {
        return;
    }

    constexpr auto NUM_CELLS = static_cast<uint32_t>(MAX_CELLS_PER_SLOT);
    const uint32_t arg_base  = ring_idx * NUM_CELLS;
    for (uint32_t cell_idx = 0; cell_idx < NUM_CELLS; ++cell_idx)
    {
        task_pool_.dlc_task_args[arg_base + cell_idx]       = {};
        task_pool_.ulc_task_args[arg_base + cell_idx]       = {};
        task_pool_.dlc_batch_task_args[arg_base + cell_idx] = {};
        task_pool_.ulc_batch_task_args[arg_base + cell_idx] = {};
    }
}

void PHY_module::reset_ul_order_scratch(uint32_t ring_idx) noexcept
{
#ifdef ENABLE_FAPI_STORE_REPLAY
    if (ring_idx >= SLOT_STORAGE_DEPTH) [[unlikely]]
    {
        return;
    }

    // Entries are pre-allocated once by prealloc_ul_order_scratch() at construction, so
    // this per-slot path only clears them - it never allocates. A null entry (a
    // construction-time allocation that failed) is skipped and stays null.
    for (auto& by_cell : ul_order_scratch_[ring_idx])
    {
        for (auto& scratch : by_cell)
        {
            if (scratch != nullptr)
            {
                scratch->reset();
            }
        }
    }
#else
    static_cast<void>(ring_idx);
#endif
}

void PHY_module::merge_ul_order_scratch(uint32_t ring_idx) noexcept
{
#ifdef ENABLE_FAPI_STORE_REPLAY
    if (ring_idx >= SLOT_STORAGE_DEPTH) [[unlikely]]
    {
        return;
    }

    const auto slot_cmd_idx = task_pool_.ul_aggr_task_args[ring_idx].slot_cmd_idx;
    if (slot_cmd_idx >= slot_command_array.size()) [[unlikely]]
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "merge_ul_order_scratch: slot_cmd_idx={} out of range size={} ring_idx={}",
                   slot_cmd_idx, slot_command_array.size(), ring_idx);
        return;
    }

    auto& slot_cmd = slot_command_array[slot_cmd_idx];
    for (uint32_t channel_idx = 0; channel_idx < kUlOrderScratchChannels; ++channel_idx)
    {
        for (uint32_t cell_idx = 0; cell_idx < MAX_CELLS_PER_SLOT && cell_idx < slot_cmd.cells.size(); ++cell_idx)
        {
            const auto& scratch_ptr = ul_order_scratch_[ring_idx][channel_idx][cell_idx];
            if (scratch_ptr == nullptr || scratch_ptr->prbs_size == 0u)
            {
                continue;
            }

            auto* const dst = slot_cmd.cells[cell_idx].sym_prb_info();
            if (dst == nullptr) [[unlikely]]
            {
                continue;
            }
            const auto& src = *scratch_ptr;
            const auto base = dst->prbs_size;
            const auto add = src.prbs_size;
            constexpr auto max_prb_info = static_cast<decltype(base)>(MAX_PRB_INFO);
            if (base > max_prb_info || add > (max_prb_info - base)) [[unlikely]]
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "merge_ul_order_scratch: sym_prb_info overflow ring_idx={} cell_idx={} "
                           "channel_idx={} base={} add={} capacity={}; dropping scratch entries",
                           ring_idx, cell_idx, channel_idx, base, add, max_prb_info);
                continue;
            }

            for (std::size_t prb_idx = 0; prb_idx < add; ++prb_idx)
            {
                dst->prbs[base + prb_idx] = src.prbs[prb_idx];
            }
            dst->prbs_size = base + add;

            for (std::size_t sym_idx = 0; sym_idx < src.symbols.size(); ++sym_idx)
            {
                for (std::size_t ch = 0; ch < src.symbols[sym_idx].size(); ++ch)
                {
                    auto& dst_list = dst->symbols[sym_idx][ch];
                    for (const auto src_prb_idx : src.symbols[sym_idx][ch])
                    {
                        // Never grow past the construction reserve: a reallocation here
                        // would be a heap allocation on the per-slot merge path. Drop and
                        // log the overflow instead (matches the prbs[] capacity guard above).
                        if (dst_list.size() >= dst_list.capacity()) [[unlikely]]
                        {
                            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                                       "merge_ul_order_scratch: symbol-index list full (capacity={}) "
                                       "ring_idx={} cell_idx={} channel_idx={} sym_idx={} ch={}; dropping remaining entries",
                                       dst_list.capacity(), ring_idx, cell_idx, channel_idx, sym_idx, ch);
                            break;
                        }
                        dst_list.push_back(base + src_prb_idx);
                    }
                }
            }

            if (src.start_symbol_ul != 0u || dst->start_symbol_ul == 0u)
            {
                dst->start_symbol_ul = src.start_symbol_ul;
            }
        }
    }
#else
    static_cast<void>(ring_idx);
#endif
}

namespace
{

constexpr int IPC_NOTIFY_VALUE = 1;

/// @brief Returns true if @p msg_id identifies a message that must be
///        processed immediately, bypassing slot storage entirely.
[[nodiscard]] bool is_immediate_non_slot_msg(uint8_t msg_id)
{
    switch(msg_id)
    {
    case CV_MEM_BANK_CONFIG_REQUEST:
    case SCF_FAPI_PARAM_REQUEST:
        return true;
    default:
        return false;
    }
}

/// @brief Returns true if @p msg_id belongs to the control lane.
[[nodiscard]] bool is_control_lane_msg(uint8_t msg_id)
{
    switch(msg_id)
    {
    case SCF_FAPI_SLOT_INDICATION:
    case SCF_FAPI_SLOT_RESPONSE:
    case SCF_FAPI_ERROR_INDICATION:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool is_parallel_nonslot_msg(const uint8_t msg_id)
{
    // Single source of truth for the worker-owned routing set lives in nv_phy_non_slot_dispatch
    // (is_nonslot_lp_message); delegate so this dispatch gate and the executor filter stay in sync.
    return is_nonslot_lp_message(msg_id);
}

void send_nonslot_error_indication(PHY_module& module,
                                   uint16_t cell_id,
                                   scf_fapi_message_id_e request_id,
                                   scf_fapi_error_codes_t error_code)
{
    auto& transport = module.transport(cell_id);
    phy_mac_msg_desc msg_desc{};
    if (transport.tx_alloc(msg_desc) < 0)
    {
        NVLOGE_FMT(TAG,
                   AERIAL_NVIPC_API_EVENT,
                   "{}: failed to allocate ERROR.ind for cell_id={} msg_id=0x{:02X} err_code=0x{:02X}",
                   __func__, cell_id, +request_id, +error_code);
        return;
    }

    auto* fapi = scf_5g_fapi::add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, cell_id, false);
    auto* rsp = aerial::casts::assume_cast<scf_fapi_error_ind_t>(fapi);
    rsp->sfn = 0;
    rsp->slot = 0;
    rsp->msg_id = request_id;
    rsp->err_code = error_code;
    transport.tx_send(msg_desc);
    transport.notify(IPC_NOTIFY_VALUE);
}

/**
 * Convert a SlotTypeMask + timing into a slot_command_api::slot_info.
 *
 * @param[in] mask  Slot-type bitmask snapshotted at EOM.
 * @param[in] sfn   System frame number extracted from the EOM sfn_slot.
 * @param[in] slot  Slot number extracted from the EOM sfn_slot.
 * @return          slot_info with type derived from mask and slot_3gpp set to {sfn, slot, 0}.
 *                  Return value must be checked.
 */
[[nodiscard]] inline slot_command_api::slot_info
slot_type_from_mask(const SlotTypeMask mask, const uint16_t sfn, const uint16_t slot) noexcept
{
    const auto m      = to_underlying(mask);
    const bool has_dl = (m & to_underlying(SlotTypeMask::DL)) != 0u;
    const bool has_ul = (m & to_underlying(SlotTypeMask::UL)) != 0u;
    return {
        (has_dl && has_ul) ? slot_command_api::SLOT_SPECIAL :
        has_dl             ? slot_command_api::SLOT_DOWNLINK :
        has_ul             ? slot_command_api::SLOT_UPLINK :
                             slot_command_api::SLOT_NONE,
        slot_command_api::slot_indication{sfn, slot, 0u}};
}

[[nodiscard]] bool has_real_phy_driver()
{
    auto* proxy = PHYDriverProxy::getInstancePtr();
    return proxy && proxy->get_driver() != nullptr;
}

} // namespace

// ---------------------------------------------------------------------------
// process_fapi_messages — drain transport RX queue, classify all FAPI
// messages (slot and non-slot) into ring storage; triggers EOM-complete and
// per-slot task dispatch.  Renamed from recv_msg_to_store.
// ---------------------------------------------------------------------------

/**
 * @brief Drain the transport receive queue, classify each message, and
 *        place it in the appropriate storage (slot or non-slot).
 * @note  Replaces recv_msg() under the ENABLE_FAPI_STORE_REPLAY
 *        build flag.  Triggers EOM-complete and slot-boundary reset
 *        logic internally; does not return a status value.
 * @thread_safety Called exclusively from the msg-processing thread.
 */
void PHY_module::process_fapi_messages()
{
    tti_event_count++;

    phy_mac_msg_desc smsg;
    if (transport_wrapper().rx_recv(smsg) < 0)
        return;

    TaskInstrumentationContext msg_ti_ctx(
        nv::detail::l2a_cpu_tracing_mode(), static_cast<uint64_t>(ss_curr.u32),
        ss_curr.u16.sfn, ss_curr.u16.slot);
    TaskInstrumentation msg_ti(msg_ti_ctx, "L2A Msg Thread", 8);
    msg_ti.add("Start Task");

    do {
        simulated_cpu_stall_checkpoint(L2A_MSG_THREAD, 0);
        sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&smsg);
        NVLOGD_FMT(TAG,
                   "process_fapi_messages: rx msg_id=0x{:02X} cell_id={} ss=0x{:08X}",
                   smsg.msg_id,
                   smsg.cell_id,
                   ss_msg.u32);
        if(is_immediate_non_slot_msg(smsg.msg_id))
        {
            process_immediate_non_slot_msg(smsg);
            continue;
        }

        if(ss_msg.u32 == SFN_SLOT_INVALID)
        {
            // Non-slot dispatch: CONFIG/START/STOP are offloaded to the nonslot_lp worker
            // (parallel dispatch is the only mode; the serial toggle is gone). PARAM and
            // CV mem-bank config never reach here -- they are handled earlier as immediate
            // messages (see is_immediate_non_slot_msg above). Any other non-slot message
            // falls through to the per-cell store-and-process path below.
            if (is_parallel_nonslot_msg(smsg.msg_id))
            {
                if (ipc_sync_mode != SYNC_MODE_PER_SLOT)
                {
                    // Non-slot CONFIG/START/STOP must not consume per-cell slot-sync budget.
                    tti_event_count--;
                }

                if (!enqueue_nonslot_lp_work(smsg))
                {
                    auto release_rx_desc = nv::make_scope_exit([&] {
                        transport_wrapper().rx_release(smsg);
                    });
                    if (smsg.cell_id >= 0 && static_cast<std::size_t>(smsg.cell_id) < phy_refs_.size())
                    {
                        send_nonslot_error_indication(*this,
                                                      static_cast<uint16_t>(smsg.cell_id),
                                                      static_cast<scf_fapi_message_id_e>(smsg.msg_id),
                                                      SCF_ERROR_CODE_NON_SLOT_OFFLOAD_REJECTED);
                    }
                }
            }
            else if (store_non_slot_message(smsg, ss_msg))
            {
                process_pending_non_slot_messages(static_cast<uint16_t>(smsg.cell_id));
            }
        }
        else
        {
            // Control lane: process immediately, do not place in slot payload storage.
            if(is_control_lane_msg(smsg.msg_id))
            {
                if(smsg.cell_id < 0)
                {
                    NVLOGW_FMT(TAG, "process_fapi_messages: invalid cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
                    transport_wrapper().rx_release(smsg);
                    continue;
                }
                const auto cell_id = static_cast<std::size_t>(smsg.cell_id);
                if(cell_id >= phy_refs_.size())
                {
                    NVLOGW_FMT(TAG, "process_fapi_messages: invalid cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
                    transport_wrapper().rx_release(smsg);
                    continue;
                }

                // Path B: handle control-lane messages directly — no on_msg() needed.
                // SLOT.IND and SLOT.RESP contributions are trivially inlined here.
                // ERROR.IND uses ring-buffer-aware recovery instead of Path A cuphydriver reset.
                switch(smsg.msg_id)
                {
                case SCF_FAPI_SLOT_INDICATION: {
                    // Slot-boundary backstop: complete the prior slot if its active-cell
                    // EOM never fired, before advancing ss_curr. Deferred by one SLOT.ind
                    // so slot N is finalized at SLOT.ind for N+2, giving late-arriving
                    // messages within the widened lag > 2 acceptance window time to
                    // complete before ring reuse.
                    finalize_slot_on_boundary(deferred_prior_slot_u32_);
                    deferred_prior_slot_u32_ = ss_curr.u32;

                    // ss_msg already holds sfn/slot from nv_ipc_get_sfn_slot — use directly.
                    set_curr_sfn_slot(ss_msg);
                    commit_staged_active_cell_bitmap();
                    // Drain the offload PRACH reconfig handover. Runs once per slot: the
                    // loopback SLOT.ind is emitted once per tick (send_slot_indication on a
                    // single cell), not per cell. tryCommit() is a single atomic-load fast
                    // path when nothing is armed; it takes the handover + aggregator locks
                    // and swaps idle aggregators' handles only while a handover is armed.
                    PHYDriverProxy::getInstance().l1_try_commit_prach_offload_handover();
                    NVLOGD_FMT(TAG, "process_fapi_messages: SLOT.ind cell_id={} ss={}.{}",
                               cell_id, ss_msg.u16.sfn, ss_msg.u16.slot);
                    break;
                }
                case SCF_FAPI_SLOT_RESPONSE:
                {
                    // Per-ring last_fapi_msg_tick stamp.
                    const uint32_t ring_idx_sr = ring_idx_from_slot(ss_msg.u32);
                    slot_telemetry_[ring_idx_sr].last_fapi_msg_tick =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch());
                    NVLOGD_FMT(TAG, "process_fapi_messages: SLOT.resp cell_id={} ss={}.{}",
                               cell_id, ss_msg.u16.sfn, ss_msg.u16.slot);
                    // EOM bitmap + enqueue logic follows immediately below.
                    break;
                }
                case SCF_FAPI_ERROR_INDICATION:
                    handle_error_ind_path_b(cell_id, ss_msg);
                    transport_wrapper().rx_release(smsg);
                    continue;
                default:
                    // is_control_lane_msg() only returns true for the three IDs above, so this
                    // branch is unreachable today.  Log a warning if that ever changes.
                    NVLOGW_FMT(TAG, "process_fapi_messages: unhandled control-lane msg_id=0x{:X} cell_id={} ss={}.{}", smsg.msg_id, cell_id, ss_msg.u16.sfn, ss_msg.u16.slot);
                    break;
                }
                transport_wrapper().rx_release(smsg);

                // EOM-complete gate — SLOT.RESP path.
                if(smsg.msg_id == SCF_FAPI_SLOT_RESPONSE)
                {
                    // Derive ring slot directly from the message's SFN/slot — no rollback of
                    // ss_curr needed because store_eom_bitmap is indexed per ring slot.
                    const uint32_t ring_idx_sr = ring_idx_from_slot(ss_msg.u32);
                    if(task_pool_.ring_reuse_dropped_slot[ring_idx_sr] == ss_msg.u32)
                    {
                        NVLOGD_FMT(TAG,
                                   "task_framework: ignoring SLOT.resp for dropped ring-reuse slot "
                                   "slot=0x{:08X} cell_id={} ring_idx={} consecutive_slot_drops={} "
                                   "total_ring_drops={}",
                                   ss_msg.u32,
                                   cell_id,
                                   ring_idx_sr,
                                   task_pool_.ring_reuse_consecutive_drops[ring_idx_sr],
                                   task_pool_.ring_reuse_total_drops[ring_idx_sr]);
                        continue;
                    }

                    // Arm sentinel for this ring slot if not already armed -- but NOT for a
                    // straggler SLOT.resp whose slot was just finalized at this ring. That
                    // happens with deferred commit: a newly-joined cell's first produced slot
                    // is not yet committed, so the established cells' EOM closes the slot, and
                    // the new cell's late SLOT.resp must not re-arm it (re-arming would
                    // re-finalize and double-submit). The straggler is dropped; its stored data
                    // is overwritten when the ring is reused by a later slot.
                    if(task_pool_.tasks_in_flight[ring_idx_sr].load(std::memory_order_acquire) == 0 &&
                       ss_msg.u32 != task_pool_.last_finalized_slot[ring_idx_sr])
                    {
                        reset_for_slot(ss_msg.u32);
                    }

                    // Accumulate DLC C-plane task for this cell. Fire if DL_TTI.req is present
                    // (DL/S slots) OR if UL_DCI.req is present (UL DCI in a DL-looking slot).
                    {
                        auto& ss = slot_message_storage(ss_msg.u32);
                        if(ss.dl_tti_count() > 0 || ss.ul_dci_count() > 0)
                        {
                            accumulate_dlc_for_cell(static_cast<uint32_t>(cell_id), ss_msg);
                        }
                    }

                    // Update per-ring-slot EOM bitmap directly — Path B never calls update_eom_rcvd_bitmap.
                    task_pool_.store_eom_bitmap[ring_idx_sr] |= (uint64_t{1} << cell_id);

                    const auto active_snapshot = task_pool_.active_cell_bitmap_snapshot[ring_idx_sr];
                    const bool eom_complete_trigger = (active_snapshot &&
                        ((task_pool_.store_eom_bitmap[ring_idx_sr] & active_snapshot) == active_snapshot));
                    NVLOGD_FMT(TAG,
                        "task_framework: EOM decision - SLOT.resp cell_id={} ss=0x{:08X} ring_idx={} "
                        "store_eom_bitmap=0x{:X} active_cell_bitmap=0x{:X} eom_trigger={} "
                        "tasks_in_flight={}",
                        cell_id,
                        ss_msg.u32,
                        ring_idx_sr,
                        task_pool_.store_eom_bitmap[ring_idx_sr],
                        active_cell_bitmap,
                        eom_complete_trigger,
                        task_pool_.tasks_in_flight[ring_idx_sr].load(std::memory_order_acquire));
                    if (eom_complete_trigger)
                    {
                        auto&      msg_slot_store = slot_message_storage(ss_msg.u32);
                        const bool store_ready    = msg_slot_store.has_slot() &&
                                                 msg_slot_store.tracked_slot() == ss_msg.u32;
                        NVLOGD_FMT(TAG,
                                   "task_framework: EOM triggered ss=0x{:08X} ring_idx={} store_ready={}",
                                   ss_msg.u32, ring_idx_sr, store_ready);
                        msg_ti.add("EOM Complete");
                        // enqueue_channel_tasks handles EOM — fires C-plane batch tasks
                        // and pushes channel-aggr tasks. Sentinel drop + slot publish +
                        // release are handled by the task drain path.
                        if(store_ready)
                        {
                            msg_slot_store.set_ready(true);
                            msg_ti.add("Enqueue Tasks Start");
                            enqueue_channel_tasks(ss_msg);
                            msg_ti.add("Enqueue Tasks End");
                        }
                        else
                        {
                            // Unbound store: empty slot (no payload) or degraded (payload dropped /
                            // ring collision). Distinguish via drop markers below.
                            const bool payload_dropped =
                                msg_slot_store.has_slot() ||
                                task_pool_.payload_dropped_slot[ring_idx_sr] == ss_msg.u32 ||
                                task_pool_.direct_cplane_stale_dropped_slot[ring_idx_sr] == ss_msg.u32;

                            // If the ring still has live tasks, they belong to an older occupant —
                            // leave cleanup to its completion path.
                            const int32_t cur_in_flight =
                                task_pool_.tasks_in_flight[ring_idx_sr].load(std::memory_order_acquire);
                            if(cur_in_flight != 0 && !nv::SlotTaskPool::is_sentinel(cur_in_flight))
                            {
                                NVLOGW_FMT(TAG,
                                           "task_framework: dropping degraded slot at EOM - ring_idx={} still "
                                           "owned by draining tasks (tasks_in_flight={}); leaving ring state to "
                                           "its completion path ss=0x{:08X}",
                                           ring_idx_sr, cur_in_flight, ss_msg.u32);
                            }
                            else
                            {
                                if(payload_dropped)
                                {
                                    NVLOGW_FMT(TAG,
                                               "task_framework: dropping degraded slot at EOM - payload lost "
                                               "before slot storage (see the ingest drop reason logged for this "
                                               "slot) ss=0x{:08X} ring_idx={} store_bound_to_other_slot={}",
                                               ss_msg.u32,
                                               ring_idx_sr,
                                               msg_slot_store.has_slot());
                                }
                                else
                                {
                                    NVLOGD_FMT(TAG,
                                               "task_framework: closing empty slot at EOM - SLOT.resp carried no "
                                               "DL/UL payload, nothing to aggregate ss=0x{:08X} ring_idx={}",
                                               ss_msg.u32, ring_idx_sr);
                                }
                                auto& store = slot_message_storage(ss_msg.u32);
                                store.set_ready(true);
                                reset_txdata_h2d_state_for_ring(ring_idx_sr);
                                release_stored_slot_messages(ss_msg.u32);
                                // Idle (0) or this slot's sentinel — safe to clear for next occupant.
                                task_pool_.tasks_in_flight[ring_idx_sr].store(0, std::memory_order_release);
                                task_pool_.channel_tasks_in_flight[ring_idx_sr].store(0, std::memory_order_release);
                                task_pool_.dlc_batch_accum[ring_idx_sr].reset();
                                task_pool_.ulc_batch_accum[ring_idx_sr].reset();
                                task_pool_.dlc_accumulated_bitmap[ring_idx_sr] = 0;
                                task_pool_.ulc_accumulated_bitmap[ring_idx_sr] = 0;
                                clear_direct_cplane_task_args_for_ring(ring_idx_sr);
                            }
                        }
                        msg_ti.add("Post Enqueue processing");
                        task_pool_.store_eom_bitmap[ring_idx_sr] = 0;
                        // Mark this slot finalized at its ring so a later straggler SLOT.resp
                        // for the same slot (deferred-commit join) cannot re-arm/re-submit it.
                        task_pool_.last_finalized_slot[ring_idx_sr] = ss_msg.u32;
                        // Cleared here: reset_for_slot() runs at SLOT.resp before classification.
                        task_pool_.payload_dropped_slot[ring_idx_sr] = SFN_SLOT_INVALID;
                        run_slot_boundary_reset(true /*EomComplete*/);
                    }
                }
                // SLOT.ind always triggers slot-boundary reset path.
                else if(smsg.msg_id == SCF_FAPI_SLOT_INDICATION)
                {
                    run_slot_boundary_reset(false /*SlotIndication*/);
                }
                continue;
            }

            // Payload lane: store now, process through worker tasks after EOM-complete.
            // TODO: temporary fix — uncomment the block below if L2A.PROCESSING_TIMES
            // instrumentation is required (l2a_start_time will be 0 otherwise).
            // This will be removed once a complete fix is done.
            //
            // if (new_slot_ &&
            //     (smsg.msg_id == SCF_FAPI_DL_TTI_REQUEST ||
            //      smsg.msg_id == SCF_FAPI_UL_TTI_REQUEST))
            // {
            //     new_slot_ = false;
            //     l2a_start_tick_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            //         std::chrono::system_clock::now().time_since_epoch());
            // }
            const uint32_t ring_idx_payload = ring_idx_from_slot(ss_msg.u32);
            auto* proxy = PHYDriverProxy::getInstancePtr();
            const auto payload_driver = (proxy != nullptr) ? proxy->get_driver() : nullptr;
            const auto direct_cplane_payload =
                payload_driver != nullptr &&
                l1_is_fapi_to_cplane_direct(payload_driver);
            if(direct_cplane_payload)
            {
                if(task_pool_.direct_cplane_stale_dropped_slot[ring_idx_payload] == ss_msg.u32)
                {
                    // reset_for_slot() clears the stale marker at SLOT.resp, before EOM
                    // classifies; this one survives so the slot is still seen as degraded.
                    task_pool_.payload_dropped_slot[ring_idx_payload] = ss_msg.u32;
                    const uint64_t payload_drops =
                        ++task_pool_.direct_cplane_stale_payload_drops[ring_idx_payload];
                    NVLOGD_FMT(TAG,
                               "DIRECT_CPLANE_STALE_PAYLOAD_DROP slot=0x{:08X} msg_id=0x{:02X} "
                               "cell_id={} ring_idx={} payload_drops={} consecutive_slot_drops={} "
                               "total_slot_drops={}",
                               ss_msg.u32,
                               smsg.msg_id,
                               smsg.cell_id,
                               ring_idx_payload,
                               payload_drops,
                               task_pool_.direct_cplane_stale_slot_drops[ring_idx_payload],
                               task_pool_.direct_cplane_total_stale_drops[ring_idx_payload]);
                    transport_wrapper().rx_release(smsg);
                    continue;
                }

                uint32_t slot_interval = 0;
                sfn_slot_t ss_tick_snapshot{};
                std::chrono::nanoseconds curr_tick_snapshot{};
                {
                    const std::lock_guard<std::mutex> lock(tick_lock);
                    ss_tick_snapshot = ss_tick.load();
                    slot_interval = get_fapi_latency(ss_msg);
                    curr_tick_snapshot = current_tick_;
                }

                if(slot_interval > kMaxDirectCplaneStaleSlotInterval)
                {
                    const auto now_ns = slot_dispatch_now_ns();
                    // Count each stale slot once toward the consecutive-drop budget and
                    // exempt startup warmup (mirrors try_drop_stale_direct_cplane_slot):
                    // skip the increment during warmup and when this slot was already
                    // counted - an earlier payload for it, or the EOM stale-drop path - so
                    // the shared counter measures distinct steady-state stale slots.
                    const bool startup_warmup_drop =
                        !startup_boot_complete_ && is_startup_direct_cplane_warmup_slot(ss_msg.u32);
                    const bool already_counted =
                        task_pool_.direct_cplane_stale_dropped_slot[ring_idx_payload] == ss_msg.u32;
                    const uint32_t consecutive_drops = (startup_warmup_drop || already_counted)
                        ? task_pool_.direct_cplane_stale_slot_drops[ring_idx_payload]
                        : ++task_pool_.direct_cplane_stale_slot_drops[ring_idx_payload];
                    const uint64_t total_drops =
                        ++task_pool_.direct_cplane_total_stale_drops[ring_idx_payload];
                    const uint64_t payload_drops =
                        ++task_pool_.direct_cplane_stale_payload_drops[ring_idx_payload];
                    const int64_t slot_age_ns =
                        static_cast<int64_t>(slot_interval) *
                        static_cast<int64_t>(mu_to_ns(tick_updater_.mu_highest_));

                    task_pool_.direct_cplane_stale_dropped_slot[ring_idx_payload] = ss_msg.u32;
                    // Survives reset_for_slot() so EOM still classifies this slot as degraded.
                    task_pool_.payload_dropped_slot[ring_idx_payload] = ss_msg.u32;

                    if(!startup_warmup_drop && consecutive_drops > kMaxDirectCplaneConsecutiveStaleDrops)
                    {
                        transport_wrapper().rx_release(smsg);
                        NVLOGF_FMT(TAG,
                                   AERIAL_L2ADAPTER_EVENT,
                                   "DIRECT_CPLANE_STALE_PAYLOAD_DROP slot=0x{:08X} sfn={} slot={} "
                                   "ring_idx={} msg_id=0x{:02X} cell_id={} slot_interval={} "
                                   "max_interval={} slot_age_ns={} ss_tick={}.{} current_tick_ns={} "
                                   "now_ns={} consecutive_slot_drops={} total_slot_drops={} "
                                   "payload_drops={} - direct C-plane FAPI payloads remain too late; exiting",
                                   ss_msg.u32,
                                   ss_msg.u16.sfn,
                                   ss_msg.u16.slot,
                                   ring_idx_payload,
                                   smsg.msg_id,
                                   smsg.cell_id,
                                   slot_interval,
                                   kMaxDirectCplaneStaleSlotInterval,
                                   slot_age_ns,
                                   ss_tick_snapshot.u16.sfn,
                                   ss_tick_snapshot.u16.slot,
                                   curr_tick_snapshot.count(),
                                   now_ns.count(),
                                   consecutive_drops,
                                   total_drops,
                                   payload_drops);
                        return;
                    }

                    NVLOGD_FMT(TAG,
                               "DIRECT_CPLANE_STALE_PAYLOAD_DROP slot=0x{:08X} sfn={} slot={} "
                               "ring_idx={} msg_id=0x{:02X} cell_id={} slot_interval={} "
                               "max_interval={} slot_age_ns={} ss_tick={}.{} current_tick_ns={} "
                               "now_ns={} consecutive_slot_drops={}/{} total_slot_drops={} "
                               "payload_drops={} - dropping expired direct C-plane payload before "
                               "slot storage/H2D staging",
                               ss_msg.u32,
                               ss_msg.u16.sfn,
                               ss_msg.u16.slot,
                               ring_idx_payload,
                               smsg.msg_id,
                               smsg.cell_id,
                               slot_interval,
                               kMaxDirectCplaneStaleSlotInterval,
                               slot_age_ns,
                               ss_tick_snapshot.u16.sfn,
                               ss_tick_snapshot.u16.slot,
                               curr_tick_snapshot.count(),
                               now_ns.count(),
                               consecutive_drops,
                               kMaxDirectCplaneConsecutiveStaleDrops,
                               total_drops,
                               payload_drops);
                    transport_wrapper().rx_release(smsg);
                    continue;
                }

                task_pool_.direct_cplane_stale_slot_drops[ring_idx_payload] = 0;
                task_pool_.direct_cplane_stale_dropped_slot[ring_idx_payload] = SFN_SLOT_INVALID;
            }

            if (!store_slot_message(smsg, ss_msg))
            {
                // So EOM can tell dropped-payload apart from an empty slot.
                task_pool_.payload_dropped_slot[ring_idx_from_slot(ss_msg.u32)] = ss_msg.u32;
                NVLOGW_FMT(TAG,
                           "process_fapi_messages: failed to store slot msg msg_id=0x{:02X} cell_id={}",
                           smsg.msg_id,
                           smsg.cell_id);
                continue;
            }

            // Per-ring telemetry stamp at ingestion (store path bypasses the
            // SCF dispatcher for payload msgs). CSI-RS presence comes from the
            // store-time sidecar — no payload walk.
            {
                bool has_csirs = false;
                if (smsg.msg_id == SCF_FAPI_DL_TTI_REQUEST)
                {
                    auto& store = slot_message_storage(ss_msg.u32);
                    const uint16_t n = store.dl_tti_count();
                    if (n > 0)
                    {
                        has_csirs =
                            store.dl_tti_pdu_counts(static_cast<uint16_t>(n - 1))
                                .by_type[DL_TTI_NPDUS_IDX_CSI_RS] > 0;
                    }
                }
                stamp_ring_telemetry_on_ingest(ss_msg, smsg.msg_id, has_csirs);
            }

            if (smsg.msg_id == SCF_FAPI_UL_TTI_REQUEST && smsg.cell_id >= 0)
            {
                if(smsg.msg_buf != nullptr)
                {
                    static constexpr std::size_t MIN_UL_TTI_NUM_PDUS =
                        offsetof(scf_fapi_ul_tti_req_t, num_pdus) +
                        sizeof(scf_fapi_ul_tti_req_t::num_pdus);
                    if(smsg.msg_len < sizeof(scf_fapi_header_t) + MIN_UL_TTI_NUM_PDUS)
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_fapi_messages: UL_TTI msg_len={} too small (need {}), slot=0x{:08X}", smsg.msg_len, sizeof(scf_fapi_header_t) + MIN_UL_TTI_NUM_PDUS, ss_msg.u32);
                    }
                    else
                    {
                        const auto* ul_tti = aerial::casts::assume_cast<const scf_fapi_ul_tti_req_t>(
                            static_cast<const char*>(smsg.msg_buf) + sizeof(scf_fapi_header_t));
                        if(ul_tti->num_pdus > 0)
                        {
                            const uint32_t ring_idx_ul = ring_idx_from_slot(ss_msg.u32);
                            if(task_pool_.tasks_in_flight[ring_idx_ul].load(std::memory_order_acquire) == 0)
                            {
                                reset_for_slot(ss_msg.u32);
                            }
                            accumulate_ulc_for_cell(static_cast<uint32_t>(smsg.cell_id), ss_msg);
                        }
                    }
                }
            }
            else if(smsg.msg_id == SCF_FAPI_TX_DATA_REQUEST && smsg.cell_id >= 0)
            {
                // Split-phase TX_DATA H2D (Phase 1): stage the GPU H2D copy on ingest;
                // launch_tx_data_h2d() completes it (Phase 2) at EOM and the channel-task
                // PDSCH body consumes the staged pointers. Unconditional — serial replay
                // (which bound the host TB pointer inline) has been removed.
                const uint32_t ring_idx_tx = ring_idx_from_slot(ss_msg.u32);
                stage_tx_data_h2d(static_cast<uint32_t>(smsg.cell_id), smsg, ring_idx_tx, ss_msg);
            }
        }
    } while (transport_wrapper().rx_recv(smsg) >= 0);

    msg_ti.add("End Task");
}

// ---------------------------------------------------------------------------

/**
 * @brief Perform slot-boundary housekeeping on SLOT.ind or EOM-complete.
 *
 * On SLOT.indication (@p slot_end_rcvd = false): arms the ring sentinel for the
 * new slot via @c reset_for_slot().  The EOM bitmap is left untouched because the
 * next slot's loopback 0x82 can arrive while the current slot's SLOT.resp messages
 * are still accumulating — premature reset would prevent EOM from firing.
 *
 * On EOM-complete (@p slot_end_rcvd = true): @c store_eom_bitmap[ring_idx] is already
 * reset inline in @c process_fapi_messages before this call; clears per-slot state flags.
 *
 * @param slot_end_rcvd  @c true on EOM-complete, @c false on SLOT.indication.
 */
void PHY_module::run_slot_boundary_reset(bool slot_end_rcvd)
{
    // Slot command submission is handled by publish_slot_command() on the completing
    // DL worker — not from this thread.  On SLOT.indication, arm the ring sentinel for
    // the incoming slot.  On EOM-complete (Path B), store_eom_bitmap[ring_idx] is already
    // cleared inline in process_fapi_messages before this call.
    if(!slot_end_rcvd)
    {
        reset_for_slot(ss_curr.u32);
    }
    else
    {
        // Path B: store_eom_bitmap[ring_idx] already reset inline before this call.
        // Path A: never calls this function — fapi_eom_rcvd_bitmap reset is in recv_msg().
    }
    tti_event_count = 0;
    new_slot_       = true;
    is_ul_slot_     = false;
    is_dl_slot_     = false;
    is_csirs_slot_  = false;
    reset_l1_limit_errors();
}

// ---------------------------------------------------------------------------

/**
 * Slot-boundary backstop: finalize the prior slot when its active-cell EOM never fired.
 *
 * Acts only if the slot still holds a pure SLOT_TASK_SENTINEL (armed but never fired);
 * the active-cell EOM replaces the sentinel with the real task count on fire, so an
 * already-finalized slot is skipped here and never double-fired.
 * @param[in] prior_slot_u32 Packed SFN/slot of the slot whose window just closed.
 */
void PHY_module::finalize_slot_on_boundary(const uint32_t prior_slot_u32)
{
    if(prior_slot_u32 == SFN_SLOT_INVALID)
    {
        return;
    }

    const uint32_t ring_idx = ring_idx_from_slot(prior_slot_u32);

    // Only act on a slot still holding a pure sentinel: armed by reset_for_slot() but its
    // active-cell EOM never completed (silent just-started cell, empty/stale startup
    // snapshot). The EOM path replaces the sentinel with the real task count (fetch_sub)
    // the moment it fires, so anything else means the slot was already handled (or drained,
    // or never armed) — skip it to avoid a double-fire. (is_armed_unfired centralizes the
    // sentinel comparison on SlotTaskPool.)
    if(!task_pool_.is_armed_unfired(ring_idx))
    {
        return;
    }

    // Pick the log level by why the slot stranded:
    //  - empty snapshot: no committed cell was ever expected for this slot. A just-STARTed
    //    cell stays out of the snapshot until its first message (commit-on-first-message), so
    //    the EOM trigger (which requires a non-empty snapshot) can never fire. This is the
    //    expected cell-startup transient -> INFO, not a warning.
    //  - non-empty snapshot: a committed, producing cell missed the boundary. Keep this at
    //    INFO to avoid noisy logs during transient startup/mixed-load recovery.
    const uint64_t active_snapshot = task_pool_.active_cell_bitmap_snapshot[ring_idx];
    if(active_snapshot == 0)
    {
        NVLOGI_FMT(TAG,
                   "task_framework: SLOT.ind backstop - slot=0x{:08X} ring_idx={} no committed "
                   "active cell for this slot (cell startup, pre-first-message); dropping degraded slot",
                   prior_slot_u32, ring_idx);
    }
    else
    {
        NVLOGI_FMT(TAG,
                   "task_framework: SLOT.ind backstop - slot=0x{:08X} ring_idx={} active-cell EOM "
                   "did not complete before slot boundary (expected=0x{:X}); dropping degraded slot",
                   prior_slot_u32, ring_idx, active_snapshot);
    }

    // Missing active-cell EOM means the slot is degraded. Do not enqueue partial
    // channel work here: launching a partial backstop batch can keep a task live
    // until the ring wraps, tripping reset_for_slot's UAF guard. Drop the slot and
    // reset the ring state; complete slots still publish through the normal EOM path.
    auto& store = slot_message_storage(prior_slot_u32);
    store.set_ready(true);
    reset_txdata_h2d_state_for_ring(ring_idx);
    release_stored_slot_messages(prior_slot_u32);
    task_pool_.tasks_in_flight[ring_idx].store(0, std::memory_order_release);
    task_pool_.channel_tasks_in_flight[ring_idx].store(0, std::memory_order_release);
    task_pool_.dlc_batch_accum[ring_idx].reset();
    task_pool_.ulc_batch_accum[ring_idx].reset();
    task_pool_.dlc_accumulated_bitmap[ring_idx] = 0;
    task_pool_.ulc_accumulated_bitmap[ring_idx] = 0;
    clear_direct_cplane_task_args_for_ring(ring_idx);
    reset_ul_order_scratch(ring_idx);
    task_pool_.store_eom_bitmap[ring_idx] = 0;
    // Mark this slot finalized at its ring so a later straggler SLOT.resp for the same slot
    // (deferred-commit join) cannot re-arm/re-submit it.
    task_pool_.last_finalized_slot[ring_idx] = prior_slot_u32;
    // Clear this slot's payload-dropped marker on the backstop path too, mirroring the
    // EOM-complete clear. Slot-conditional so a newer slot's marker on the same ring
    // (the slot-keyed-equality invariant) is preserved.
    if(task_pool_.payload_dropped_slot[ring_idx] == prior_slot_u32)
    {
        task_pool_.payload_dropped_slot[ring_idx] = SFN_SLOT_INVALID;
    }
}

/**
 * @brief Path B ring-buffer-aware ERROR.ind recovery.
 *
 * Steps, in order:
 *   1. Release stored FAPI payload messages for the errored slot N so the IPC
 *      buffers are returned to the transport.
 *   2. Run the slot-boundary reset (@c run_slot_boundary_reset(false)) which
 *      internally calls @c reset_for_slot(ss_curr.u32) — clears the
 *      tasks_in_flight sentinel, store_eom_bitmap, DLC/ULC accumulators, and
 *      re-snapshots active-cell state for ring_idx(N).
 *   3. Advance @c ss_curr to N+2 so stale post-error payload messages are
 *      dropped by @c store_slot_message's lag/lead window check.
 *
 * Step 1 and step 2 are order-independent: @c release_stored_slot_messages
 * only touches per-ring slot-storage entries, and @c reset_for_slot only
 * touches the per-ring task / accumulator / snapshot bookkeeping. Step 2 is
 * placed before step 3 because @c reset_for_slot reads @c ss_curr internally —
 * if @c ss_curr were advanced to N+2 first, the boundary reset would target
 * ring_idx(N+2) ≡ ring_idx(N-1) (mod SLOT_STORAGE_DEPTH=3), potentially
 * corrupting the prior slot's in-flight counter.
 *
 * @param cell_id Cell that sent the ERROR.ind.
 * @param ss_msg  SFN/slot of the errored slot.
 */
void PHY_module::handle_error_ind_path_b(std::size_t cell_id, sfn_slot_t ss_msg)
{
    const uint32_t ring_idx = ring_idx_from_slot(ss_msg.u32);
    const int32_t  cur      = task_pool_.tasks_in_flight[ring_idx].load(std::memory_order_acquire);
    if(cur == 0)
    {
        NVLOGW_FMT(TAG,
                   "handle_error_ind_path_b: ignoring late ERROR.ind cell_id={} ss={}.{} "
                   "ring_idx={} - EOM already completed (tasks_in_flight=0)",
                   cell_id,
                   ss_msg.u16.sfn,
                   ss_msg.u16.slot,
                   ring_idx);
        return;
    }

    NVLOGW_FMT(TAG,
               "process_fapi_messages: ERROR.ind cell_id={} ss={}.{} ring_idx={} "
               "tasks_in_flight={} - resetting Path B ring slot",
               cell_id,
               ss_msg.u16.sfn,
               ss_msg.u16.slot,
               ring_idx,
               cur);

    // Reset only if the ring slot still holds a sentinel (tasks not yet drained).
    // When tasks_in_flight is already 0, the slot has fully drained and re-arming
    // the sentinel would cause a FATAL on the next slot that maps to this ring index.
    {
        const uint32_t ring_idx = ring_idx_from_slot(ss_msg.u32);
        const int32_t cur = task_pool_.tasks_in_flight[ring_idx].load(std::memory_order_acquire);
        if (cur != 0)
        {
            reset_for_slot(ss_msg.u32);
        }
    }

    // Release all stored FAPI payload messages for this slot so IPC buffers are returned.
    // Uses the PHY_module-level method — not a method on the slot store object returned
    // by slot_message_storage(). Order-independent w.r.t. the boundary reset below:
    // release_stored_slot_messages only touches per-ring slot storage entries, not the
    // tasks_in_flight / accumulator / snapshot state cleared inside reset_for_slot().
    // No H2D/driver callback fires for an errored slot, so clear `deferred` first so the
    // release below frees the TX_DATA lane (reset() itself is staging-only).
    reset_txdata_h2d_state_for_ring(ring_idx);
    release_stored_slot_messages(ss_msg.u32);

    // Call boundary reset BEFORE advancing ss_curr — internally invokes
    // reset_for_slot(ss_curr.u32), which clears tasks_in_flight sentinel, store_eom_bitmap,
    // DLC/ULC accumulated bitmaps, and re-snapshots active-cell state. ss_curr is still
    // the errored slot N here (see ordering rationale in Doxygen above), so the reset
    // targets the errored slot's ring entry.
    //
    // Pre-condition: EOM has not fired for the errored slot — tasks_in_flight is 0 or
    // SENTINEL (no worker tasks have been enqueued yet for this ring slot).
    // ERROR.IND arrives when L1 fails to process the slot, before EOM completes, so this
    // is safe to call unconditionally.
    run_slot_boundary_reset(false /*force boundary reset — not EOM*/);

    // Advance ss_curr to N+2 AFTER the boundary reset so that store_slot_message's
    // lag/lead window check (lag > 1) drops any stale payload messages for the
    // errored slot that arrive after the reset.
    // Note: SFN_SLOT_INVALID cannot be used — the window check is bypassed when
    // ss_curr == SFN_SLOT_INVALID, which would allow stale messages through.
    sfn_slot_t ss_skip = get_next_sfn_slot(ss_msg);
    ss_skip            = get_next_sfn_slot(ss_skip);
    set_curr_sfn_slot(ss_skip);
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// reset_for_slot
// ---------------------------------------------------------------------------

/**
 * @brief Arm the per-ring-slot in-flight sentinel for a new slot.
 *
 * Stores @c nv::SlotTaskPool::SLOT_TASK_SENTINEL into @c task_pool_.tasks_in_flight[ring_idx], marking the
 * ring slot as "owned" by @p slot_u32.  All subsequent task enqueues increment
 * from this base; when the last task completes the counter drops to zero and
 * @c release_stored_slot_messages() is triggered.
 *
 * Safe to call when the counter is already at @c nv::SlotTaskPool::SLOT_TASK_SENTINEL (stale
 * sentinel from a UL-only slot that received no EOM).  Logs a debug message in
 * that case and overwrites the stale value.  If live tasks are still in flight
 * (counter != 0 and != sentinel), drops the incoming slot for a bounded number
 * of consecutive ring reuses and aborts once the guard threshold is exceeded.
 *
 * @param slot_u32  Packed SFN/slot for the incoming slot.
 *
 * Deliberately does not clear @c slot_telemetry_[ring_idx]: the per-ring
 * telemetry snapshot is owned by bind_ring_slot(), which re-zeroes it on every
 * FreshBind/Rebound. Every slot that reaches publish stores its first message
 * through store_slot_message, whose rebind path runs bind_ring_slot() when the
 * ring is empty or holds a different slot - so the snapshot is always reset
 * before the new slot stamps or the worker reads it. A prior slot's telemetry
 * (is_csirs_slot, per-channel counters) can therefore never leak into a
 * later slot's snapshot, even though the EOM / error reset paths route here.
 */
void PHY_module::reset_for_slot(const uint32_t slot_u32)
{
    const uint32_t ring_idx = ring_idx_from_slot(slot_u32);

    // Latch out of the startup window the first time a slot at or past it is armed,
    // so the SFN-based warmup / ring-reuse tolerance never re-trigger on SFN rollover.
    const sfn_slot_t ss_arm{.u32 = slot_u32};
    if(!startup_boot_complete_ && ss_arm.u16.sfn >= kStartupDirectCplaneWarmupSfns)
    {
        startup_boot_complete_ = true;
    }

    const int32_t cur = task_pool_.tasks_in_flight[ring_idx].load(std::memory_order_acquire);
    if(nv::SlotTaskPool::is_sentinel(cur))
    {
        // Stale sentinel: prior slot at this ring_idx received no EOM (e.g. UL-only slot).
        // No tasks were enqueued so there is no UAF risk — reclaim safely.
        NVLOGD_FMT(TAG,
                   "task_framework: reset_for_slot: ring_idx={} slot=0x{:08X} - stale sentinel "
                   "reclaimed (prior slot had no EOM); overwriting with fresh sentinel",
                   ring_idx,
                   slot_u32);
    }
    else if(cur != 0)
    {
        const uint64_t total_drops       = ++task_pool_.ring_reuse_total_drops[ring_idx];
        const uint32_t consecutive_drops = ++task_pool_.ring_reuse_consecutive_drops[ring_idx];
        if(consecutive_drops <= kMaxConsecutiveRingReuseDrops)
        {
            NVLOGW_FMT(TAG,
                       "reset_for_slot: ring_idx={} slot=0x{:08X} tasks_in_flight={} != 0 "
                       "- dropping incoming slot to avoid UAF consecutive_drops={}/{} "
                       "total_ring_drops={} payload_drops={}",
                       ring_idx,
                       slot_u32,
                       cur,
                       consecutive_drops,
                       kMaxConsecutiveRingReuseDrops,
                       total_drops,
                       task_pool_.ring_reuse_payload_drops[ring_idx]);
            task_pool_.ring_reuse_dropped_slot[ring_idx] = slot_u32;
            return;
        }

        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "reset_for_slot: ring_idx={} slot=0x{:08X} tasks_in_flight={} != 0 "
                   "- prior slot tasks still live after {} consecutive ring-reuse drops "
                   "(total_ring_drops={} payload_drops={}); exiting to avoid hiding a stuck task",
                   ring_idx,
                   slot_u32,
                   cur,
                   consecutive_drops,
                   total_drops,
                   task_pool_.ring_reuse_payload_drops[ring_idx]);
        return;
    }
    task_pool_.ring_reuse_consecutive_drops[ring_idx] = 0;
    task_pool_.ring_reuse_dropped_slot[ring_idx] = SFN_SLOT_INVALID;
    task_pool_.direct_cplane_stale_dropped_slot[ring_idx] = SFN_SLOT_INVALID;
    task_pool_.tasks_in_flight[ring_idx].store(nv::SlotTaskPool::SLOT_TASK_SENTINEL, std::memory_order_release);
    task_pool_.channel_tasks_in_flight[ring_idx].store(nv::SlotTaskPool::SLOT_TASK_SENTINEL, std::memory_order_release);

    // Reset batch accumulators and duplicate-cell bitmaps for this ring slot.
    task_pool_.dlc_batch_accum[ring_idx].reset();
    task_pool_.ulc_batch_accum[ring_idx].reset();
    task_pool_.dlc_accumulated_bitmap[ring_idx] = 0;
    task_pool_.ulc_accumulated_bitmap[ring_idx] = 0;
    clear_direct_cplane_task_args_for_ring(ring_idx);
    // Defensive clear: store_eom_bitmap is also reset inline in process_fapi_messages
    // immediately after EOM fires (before this function is called via run_slot_boundary_reset).
    // This second reset handles the edge case where no EOM was received for this ring slot
    // (e.g. slot dropped / ring overflow) so the bitmap is clean for the next occupant.
    task_pool_.store_eom_bitmap[ring_idx] = 0;
    snapshot_active_cell_state(slot_u32);
    task_pool_.fapi_first_arrival_ts[ring_idx] = std::chrono::nanoseconds::zero();
    task_pool_.enqueue_phy_work_done[ring_idx].store(false, std::memory_order_release);
    reset_ul_order_scratch(ring_idx);
}

// ---------------------------------------------------------------------------
// Batched DLC/ULC policy push_task implementations and fire helper
// ---------------------------------------------------------------------------

[[nodiscard]] bool PHY_module::DlcBatchPolicy::push_task(uint64_t ts, l1_task_work_fn_t fn, void* arg)
{
    return PHYDriverProxy::getInstance()
        .l1_push_new_dl_task(ts, "task_dlc_batch", fn, arg, 0, 1, 1, INVALID_WORKER_ID);
}

[[nodiscard]] bool PHY_module::UlcBatchPolicy::push_task(uint64_t ts, l1_task_work_fn_t fn, void* arg)
{
    return PHYDriverProxy::getInstance()
        .l1_push_new_ul_task(ts, "task_ulc_batch", fn, arg, 0, 1, 1, INVALID_WORKER_ID);
}

template <CplaneBatchPolicy Policy>
void PHY_module::fire_cplane_batch(
    uint32_t                          ring_idx,
    sfn_slot_t                        ss_curr,
    std::span<nv::CplaneBatchAccum>   accums,
    std::span<nv::CplaneBatchTaskArg> batch_pool,
    uint8_t local_batch_idx,
    uint8_t n_cells_in_batch,
    uint8_t batch_id,
    uint8_t total_batches,
    uint64_t task_ts_ns)
{
    auto& accum = accums[ring_idx];
    if (n_cells_in_batch == 0) { return; }

    const uint32_t batch_idx =
        ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT) + local_batch_idx;
    if (batch_idx >= batch_pool.size())
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "fire_cplane_batch: batch_idx={} out of range [0,{}) ring_idx={} - "
                   "logic error (local_batch_idx={} n_cells={}); dropping batch",
                   batch_idx, batch_pool.size(), ring_idx,
                   local_batch_idx, n_cells_in_batch);
        accum.batch_count = 0;
        return;
    }
    auto& barg        = batch_pool[batch_idx];
    barg.phy_module   = this;
    barg.ring_idx     = ring_idx;
    barg.slot_u32     = ss_curr.u32;
    barg.n_cells      = n_cells_in_batch;
    barg.process_cell = Policy::process_cell_fn();
    barg.on_complete  = &PHY_module::s_done;
    barg.slot_map     = Policy::get_slot_map(this, ring_idx, barg.cell_ids[0]);
    barg.post_batch   = (barg.slot_map != nullptr) ? Policy::post_batch_fn() : nullptr;
    barg.on_batch_error = (barg.slot_map != nullptr) ? Policy::on_batch_error_fn() : nullptr;
    const bool is_dl_direction = std::is_same_v<Policy, DlcBatchPolicy>;
    barg.direction    = is_dl_direction
        ? nv::CplaneBatchDirection::DOWNLINK
        : nv::CplaneBatchDirection::UPLINK;
    barg.batch_id     = batch_id;
    barg.total_batches = total_batches;
    const std::size_t max_num_dl_batches = config_options_.cplane_max_num_dl_batches;

    // Use non-overlapping transaction-ID ranges per direction:
    // DL uses [0, max_num_dl_batches), UL is offset by max_num_dl_batches
    // so concurrent DL/UL batches never collide in framework transaction state.
    barg.transaction_id = is_dl_direction
        ? batch_id
        : max_num_dl_batches + batch_id;

    // Increment BEFORE push_task to prevent premature zero-crossing: a concurrent
    // on_slot_channel_task_complete() could otherwise see the counter reach 0 between
    // the push and the first on_complete callback, triggering an early slot release.
    // Policy::on_drop() undoes this increment if push_task fails.
    task_pool_.tasks_in_flight[ring_idx].fetch_add(1, std::memory_order_acq_rel);

    const bool push_ok = Policy::push_task(task_ts_ns, task_work_fn_cplane_batch, &barg);
    if(!push_ok)
    {
        Policy::on_drop(this, ring_idx, ss_curr.u32);
        NVLOGW_FMT(TAG, "fire_cplane_batch: ring_idx={} n_cells={} slot=0x{:08X} - "
                   "push_task() failed; batch dropped",
                   ring_idx, n_cells_in_batch, ss_curr.u32);
    }
    else
    {
        NVLOGD_FMT(TAG, "task_framework: fired cplane_batch ring_idx={} n_cells={} "
                   "batch_count={} slot=0x{:08X} tasks_in_flight={}",
                   ring_idx, n_cells_in_batch, accum.batch_count, ss_curr.u32,
                   task_pool_.tasks_in_flight[ring_idx].load());
    }
    if (push_ok)
    {
        ++accum.batch_count;
    }

    // Stash the running batch count on the slot map so cuphydriver-side task bodies
    // (Compression, GPU Comm TX, BufCleanup, Debug on DL; Order Kernel, PUCCH+PUSCH,
    // Early UCI Ind, UL3 on UL) can read a path-aware contributor count via
    // SlotMap{Dl,Ul}::getNum{Dlc,Ulc}Tasks() instead of recomputing from worker count.
    // Updated on every fire so the slot map holds the final count once all
    // batches for this slot have been scheduled.  No-op
    // when slot_map is null (e.g., partial dispatch failure paths).
    if (barg.direction == nv::CplaneBatchDirection::DOWNLINK) {
        l1_set_num_dlc_tasks_for_slot(barg.slot_map, static_cast<int>(accum.batch_count));
    } else {
        l1_set_num_ulc_tasks_for_slot(barg.slot_map, static_cast<int>(accum.batch_count));
    }
}

// ---------------------------------------------------------------------------

/**
 * @brief Accumulate a per-cell DL C-plane entry for EOM-only batch scheduling.
 *
 * Called from the @c msg_processing thread on each SLOT.response, before the
 * global EOM gate fires.  Locates this cell's DL_TTI.req and UL_DCI.req in the
 * ring buffer, sets the dlc_accumulated_bitmap bit for the cell, stores per-cell data in
 * dlc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cell_id], and pre-partitions cell_id
 * into dlc_batch_task_args for EOM firing. enqueue_channel_tasks() only finalizes
 * metadata and fires the prebuilt DLC batches.
 *
 * tasks_in_flight[ring_idx] is incremented by 1 per batch (not per cell).
 *
 * @param cell_id  FAPI cell identifier (0 … NUM_CELLS-1).
 * @param ss_curr  Packed SFN/slot of the current slot.
 */
void PHY_module::accumulate_dlc_for_cell(uint32_t cell_id, sfn_slot_t ss_curr)
{
    if(cell_id >= MAX_CELLS_PER_SLOT)
    {
        NVLOGW_FMT(TAG, "accumulate_dlc_for_cell: cell_id={} out of range slot=0x{:08X}", cell_id, ss_curr.u32);
        return;
    }
    if (!has_real_phy_driver())
    {
        return;
    }

    const uint32_t ring_idx = ring_idx_from_slot(ss_curr.u32);
    const uint32_t arg_idx = ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT) + cell_id;

    // Duplicate-cell guard: skip if this cell's SLOT.resp was already accumulated
    // for this ring slot (e.g. duplicate FAPI delivery).
    const uint64_t cell_bit = uint64_t{1} << cell_id;
    if(task_pool_.dlc_accumulated_bitmap[ring_idx] & cell_bit)
    {
        NVLOGW_FMT(TAG,
                   "accumulate_dlc_for_cell: cell_id={} slot=0x{:08X} - "
                   "duplicate SLOT.resp for this ring slot, skipping",
                   cell_id,
                   ss_curr.u32);
        return;
    }
    task_pool_.dlc_accumulated_bitmap[ring_idx] |= cell_bit;
    // Bitmap is set here, BEFORE the early-exit PDU check below, intentionally:
    // a SLOT.resp with no DL_TTI/UL_DCI PDUs for this cell is still a valid delivery.
    // Marking the bit now ensures a duplicate SLOT.resp for the same ring slot is
    // suppressed even when no batch task is enqueued for this cell.

    auto& slot_store = slot_message_storage(ss_curr.u32);

    // Locate this cell's DL_TTI.req and UL_DCI.req in one pass over the ring store.
    phy_mac_msg_desc* dl_tti     = nullptr;
    phy_mac_msg_desc* ul_dci     = nullptr;
    uint16_t          dl_tti_idx = 0;

    const uint16_t n_dl = slot_store.dl_tti_count();
    const uint16_t n_ul = slot_store.ul_dci_count();

    for(uint16_t i = 0; i < std::max(n_dl, n_ul); ++i)
    {
        if(dl_tti == nullptr && i < n_dl &&
           static_cast<uint32_t>(slot_store.dl_tti_messages()[i].cell_id) == cell_id)
        {
            dl_tti     = &slot_store.dl_tti_messages()[i];
            dl_tti_idx = i;
        }
        if(ul_dci == nullptr && i < n_ul &&
           static_cast<uint32_t>(slot_store.ul_dci_messages()[i].cell_id) == cell_id)
        {
            ul_dci = &slot_store.ul_dci_messages()[i];
        }
        if(dl_tti != nullptr && ul_dci != nullptr)
        {
            break;
        }
    }

    // Gate on a non-zero PDU count so that a DL_TTI.req with no PDUs does not
    // fire a degenerate batch (matches the channel mask scan in
    // enqueue_channel_tasks). Uses the store-time sidecar — no payload walk.
    const bool has_dl_pdus =
        (dl_tti != nullptr) && slot_store.dl_tti_pdu_counts(dl_tti_idx).any_channel();
    const bool has_ul_dcis = (ul_dci != nullptr);

    if(!has_dl_pdus && !has_ul_dcis)
    {
        task_pool_.dlc_task_args[arg_idx] = {};
        return;
    }

    // Store per-cell data in ring-indexed storage — no cross-slot aliasing possible.
    // Accessed by s_process_dlc_cell inside the batch worker.
    auto& arg = task_pool_.dlc_task_args[arg_idx];
    arg       = {&PHY_instances()[cell_id].get(), dl_tti, ul_dci};

    // Pre-partition this cell into its final EOM batch slot so EOM only fires.
    auto& accum = task_pool_.dlc_batch_accum[ring_idx];
    const uint8_t effective_batch_size = config_options_.cplane_processing_dl_batch_size;
    const uint8_t cell_ord = accum.n_pending;
    const uint8_t local_batch_idx = static_cast<uint8_t>(cell_ord / effective_batch_size);
    const uint8_t cell_offset = static_cast<uint8_t>(cell_ord % effective_batch_size);
    const uint32_t batch_base = ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT);
    const uint32_t batch_idx = batch_base + local_batch_idx;
    if (batch_idx >= task_pool_.dlc_batch_task_args.size())
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "accumulate_dlc_for_cell: batch_idx={} out of range [0,{}) ring_idx={} "
                   "slot=0x{:08X} cell_id={} n_pending={}",
                   batch_idx, task_pool_.dlc_batch_task_args.size(), ring_idx,
                   ss_curr.u32, cell_id, accum.n_pending);
        return;
    }
    auto& batch_arg = task_pool_.dlc_batch_task_args[batch_idx];
    batch_arg.cell_ids[cell_offset] = static_cast<uint8_t>(cell_id);
    batch_arg.n_cells = static_cast<uint8_t>(cell_offset + 1u);
    ++accum.n_pending;

    NVLOGD_FMT(TAG,
               "task_framework: accumulated DLC cell_id={} slot=0x{:08X} ring_idx={} "
               "has_dl_pdus={} has_ul_dcis={} n_pending={} dl_eom_batch_size={}",
               cell_id, ss_curr.u32, ring_idx, has_dl_pdus, has_ul_dcis,
               accum.n_pending, config_options_.cplane_processing_dl_batch_size);
}

/**
 * @brief Drop a direct C-plane slot whose FAPI input arrived too late to launch.
 *
 * A slot is dropped when it is a startup warm-up slot or its @p slot_interval exceeds
 * @c kMaxDirectCplaneStaleSlotInterval. On a drop the ring is fully reset (both counters
 * zeroed, accumulators/bitmaps cleared, stored FAPI released, TX_DATA H2D reset) so the
 * next occupant arms cleanly. After @c kMaxDirectCplaneConsecutiveStaleDrops consecutive
 * non-startup drops the ring is left untouched and a fatal is logged so a genuinely stuck
 * input is not masked by silent resets.
 *
 * Must be called before any C-plane batch is fired for the slot, so zeroing the counters
 * cannot race an in-flight task completion.
 *
 * @param ss_curr            Packed SFN/slot reaching EOM.
 * @param ring_idx           Ring index for @p ss_curr.
 * @param slot_interval      FAPI latency (in slots) snapshotted under @c tick_lock.
 * @param ss_tick_snapshot   SFN/slot tick snapshot, for diagnostics only.
 * @param curr_tick_snapshot current_tick_ snapshot, for diagnostics only.
 * @return @c true if the slot was dropped and the caller must return; @c false to proceed.
 */
[[nodiscard]] bool PHY_module::try_drop_stale_direct_cplane_slot(const sfn_slot_t               ss_curr,
                                                                 const uint32_t                 ring_idx,
                                                                 const uint32_t                 slot_interval,
                                                                 const sfn_slot_t               ss_tick_snapshot,
                                                                 const std::chrono::nanoseconds curr_tick_snapshot)
{
    const bool startup_warmup_drop = !startup_boot_complete_ && is_startup_direct_cplane_warmup_slot(ss_curr.u32);
    if(!(startup_warmup_drop || slot_interval > kMaxDirectCplaneStaleSlotInterval))
    {
        return false;
    }

    const auto now_ns = slot_dispatch_now_ns();
    // Warmup drops must not consume the fatal consecutive-drop budget (mirrors the
    // ring-reuse startup-tolerance path): only genuine stale drops increment it, so
    // the abort threshold measures steady-state lateness, not boot transients.
    // Dedup against the payload-ingress path: if this slot already counted a stale
    // drop (direct_cplane_stale_dropped_slot tracks it), don't count it twice.
    const bool already_counted = task_pool_.direct_cplane_stale_dropped_slot[ring_idx] == ss_curr.u32;
    const uint32_t consecutive_drops = (startup_warmup_drop || already_counted)
        ? task_pool_.direct_cplane_stale_slot_drops[ring_idx]
        : ++task_pool_.direct_cplane_stale_slot_drops[ring_idx];
    const uint64_t total_drops = ++task_pool_.direct_cplane_total_stale_drops[ring_idx];
    const int64_t slot_age_ns =
        static_cast<int64_t>(slot_interval) * static_cast<int64_t>(mu_to_ns(tick_updater_.mu_highest_));

    if(!startup_warmup_drop && consecutive_drops > kMaxDirectCplaneConsecutiveStaleDrops)
    {
        NVLOGF_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "DIRECT_CPLANE_STALE_DROP slot=0x{:08X} sfn={} slot={} ring_idx={} "
                   "slot_interval={} max_interval={} slot_age_ns={} ss_tick={}.{} "
                   "current_tick_ns={} now_ns={} consecutive_drops={} total_drops={} "
                   "- direct C-plane FAPI input remains too late; exiting",
                   ss_curr.u32,
                   ss_curr.u16.sfn,
                   ss_curr.u16.slot,
                   ring_idx,
                   slot_interval,
                   kMaxDirectCplaneStaleSlotInterval,
                   slot_age_ns,
                   ss_tick_snapshot.u16.sfn,
                   ss_tick_snapshot.u16.slot,
                   curr_tick_snapshot.count(),
                   now_ns.count(),
                   consecutive_drops,
                   total_drops);
        return true;
    }

    if(startup_warmup_drop)
    {
        NVLOGD_FMT(TAG,
                   "DIRECT_CPLANE_STARTUP_WARMUP_DROP slot=0x{:08X} sfn={} slot={} "
                   "ring_idx={} warmup_sfns={} slot_interval={} slot_age_ns={} "
                   "ss_tick={}.{} current_tick_ns={} now_ns={} total_slot_drops={} "
                   "- dropping startup direct C-plane slot before task launch",
                   ss_curr.u32,
                   ss_curr.u16.sfn,
                   ss_curr.u16.slot,
                   ring_idx,
                   kStartupDirectCplaneWarmupSfns,
                   slot_interval,
                   slot_age_ns,
                   ss_tick_snapshot.u16.sfn,
                   ss_tick_snapshot.u16.slot,
                   curr_tick_snapshot.count(),
                   now_ns.count(),
                   total_drops);
    }
    else
    {
        NVLOGD_FMT(TAG,
                   "DIRECT_CPLANE_STALE_DROP slot=0x{:08X} sfn={} slot={} ring_idx={} "
                   "slot_interval={} max_interval={} slot_age_ns={} ss_tick={}.{} "
                   "current_tick_ns={} now_ns={} consecutive_drops={}/{} total_drops={} "
                   "- dropping expired direct C-plane slot before task launch",
                   ss_curr.u32,
                   ss_curr.u16.sfn,
                   ss_curr.u16.slot,
                   ring_idx,
                   slot_interval,
                   kMaxDirectCplaneStaleSlotInterval,
                   slot_age_ns,
                   ss_tick_snapshot.u16.sfn,
                   ss_tick_snapshot.u16.slot,
                   curr_tick_snapshot.count(),
                   now_ns.count(),
                   consecutive_drops,
                   kMaxDirectCplaneConsecutiveStaleDrops,
                   total_drops);
    }

    auto& store = slot_message_storage(ss_curr.u32);
    store.set_ready(true);
    reset_txdata_h2d_state_for_ring(ring_idx);
    release_stored_slot_messages(ss_curr.u32);
    task_pool_.tasks_in_flight[ring_idx].store(0, std::memory_order_release);
    task_pool_.channel_tasks_in_flight[ring_idx].store(0, std::memory_order_release);
    task_pool_.dlc_batch_accum[ring_idx].reset();
    task_pool_.ulc_batch_accum[ring_idx].reset();
    task_pool_.dlc_accumulated_bitmap[ring_idx] = 0;
    task_pool_.ulc_accumulated_bitmap[ring_idx] = 0;
    clear_direct_cplane_task_args_for_ring(ring_idx);
    task_pool_.store_eom_bitmap[ring_idx] = 0;
    task_pool_.last_finalized_slot[ring_idx] = ss_curr.u32;
    task_pool_.direct_cplane_stale_dropped_slot[ring_idx] = ss_curr.u32;
    reset_ul_order_scratch(ring_idx);
    return true;
}

/**
 * @brief Enqueue EOM-triggered DL/UL channel work for a completed slot.
 *
 * Called from the @c msg_processing thread once the EOM gate fires (all cells have
 * delivered their SLOT.response). A single task-based path handles every slot:
 *
 *   1. Wire up the early C-plane DL/UL slot maps and fire the DLC/ULC C-plane batch
 *      tasks accumulated during ingest (via @c fire_cplane_batch, finalised by
 *      @c plan_and_fire_eom_batches).
 *   2. Launch the split-phase TX_DATA H2D DMA (Phase 2).
 *   3. Push one worker task per active channel type (PDSCH, CSI-RS, PDCCH, SSB,
 *      DL_BFW on DL; PUSCH, PRACH, PUCCH, SRS, UL_BFW on UL).
 *
 * Two counters gate the slot: @c channel_tasks_in_flight (channel-aggr tasks only) and
 * @c tasks_in_flight (all tasks). When the channel counter drains, @c publish_slot_command
 * enqueues the U-plane work — independent of C-plane; when the all-tasks counter drains,
 * @c finalize_slot_cleanup releases stored FAPI messages. Either step runs inline here if its
 * set already drained before the sentinels are dropped, else from the last completing task.
 * publish_slot_command applies the direct C-plane skip mask, so it enqueues only U-plane work
 * — the C-plane was already published by the batch tasks — and handles the empty-slot
 * (channel_array_size
 * == 0) case itself. No serial replay runs on the msg thread.
 *
 * @param ss_curr  Packed SFN/slot of the slot that just reached EOM.
 */

namespace {

/**
 * @brief C++20 zero-size type tag pairing a channel-mask bit with a
 *        @c cell_group_command member function pointer via @c auto NTTP.
 *
 * All members are @c static @c constexpr; the type has no runtime storage.
 *
 * @tparam Bit  Bitmask value (e.g. @c CH_PDSCH) that enables this channel.
 * @tparam MFn  Pointer to the @c cell_group_command getter (e.g.
 *              @c &cell_group_command::get_pdsch_params) invoked when
 *              @c bit is set in the active channel mask.
 */
template <uint32_t Bit, auto MFn>
struct Ch final {
    static constexpr uint32_t bit = Bit;
    static constexpr auto      fn = MFn;
};

/**
 * @brief For each tag type @c Cs whose bit is set in @p mask, invokes
 *        @c (grp.*Cs::fn)() to register the corresponding channel in @p grp.
 *
 * Return values from the getters are intentionally discarded; the call is
 * made solely for the @c create_if side-effect that populates
 * @c cell_group_command::channel_idx[].
 *
 * @tparam Cs   Pack of @c Ch<Bit,MFn> tag types to evaluate.
 * @param  grp  @c cell_group_command whose @c channel_idx[] will be populated.
 * @param  mask Bitmask of active channels; a getter is called only when
 *              the corresponding @c Cs::bit is set.
 */
template <typename... Cs>
void prime_channels_from_mask(
    slot_command_api::cell_group_command& grp,
    uint32_t mask) noexcept
{
    auto prime_one = [&](auto channel) noexcept {
        using C = decltype(channel);
        if (mask & C::bit) {
            std::invoke(C::fn, grp);
        }
    };

    (prime_one(Cs{}), ...);
}

/**
 * @brief Registers all active slot channels in @p grp on the message thread,
 *        before any worker reads @c cell_group_command::channel_idx[].
 *
 * DL_TTI PDCCH and UL_DCI both feed the DL PDCCH aggregate path.  The legacy
 * on_msg path validates UL_DCI with the PDCCH_UL mask but registers the shared
 * PDCCH params as PDCCH_DL; keep store/replay aligned with that contract
 * because the cuphydriver PDCCH_UL aggregate pool is intentionally unused.
 * All other DL and UL channels are registered
 * via @c prime_channels_from_mask using the bit constants passed as NTTPs.
 *
 * Side effects: may invoke @c cg_t::get_pdcch_params, @c get_pdsch_params,
 * @c get_csirs_params, @c get_pbch_params, @c get_pusch_params,
 * @c get_prach_params, @c get_pucch_params, and @c get_srs_params.
 *
 * @tparam ChPdsch      Bit value for PDSCH in the DL mask.
 * @tparam ChCsirs      Bit value for CSI-RS in the DL mask.
 * @tparam ChSsb        Bit value for SSB/PBCH in the DL mask.
 * @tparam ChPusch      Bit value for PUSCH in the UL mask.
 * @tparam ChPrach      Bit value for PRACH in the UL mask.
 * @tparam ChPucch      Bit value for PUCCH in the UL mask.
 * @tparam ChSrs        Bit value for SRS in the UL mask.
 * @param  grp          @c cell_group_command to populate.
 * @param  dl_mask      Active DL channel bitmask (excluding PDCCH).
 * @param  ul_mask      Active UL channel bitmask.
 * @param  has_pdcch_dl @c true if a DL_TTI.req contained PDCCH PDUs.
 * @param  has_ul_dci   @c true if a UL_DCI.req is present for this slot.
 */
template <uint32_t ChPdsch, uint32_t ChCsirs, uint32_t ChSsb,
          uint32_t ChPusch, uint32_t ChPrach, uint32_t ChPucch, uint32_t ChSrs>
void
prime_slot_command_channel_params(
    slot_command_api::cell_group_command& grp,
    const uint32_t dl_mask,
    const uint32_t ul_mask,
    const bool has_pdcch_dl,
    const bool has_ul_dci) noexcept
{
    using cg_t = slot_command_api::cell_group_command;

    if (has_pdcch_dl || has_ul_dci) {
        grp.ensure_pdcch_params(slot_command_api::PDCCH_DL);
    }

    prime_channels_from_mask<
        Ch<ChPdsch, &cg_t::get_pdsch_params>,
        Ch<ChCsirs, &cg_t::get_csirs_params>,
        Ch<ChSsb,   &cg_t::get_pbch_params>
    >(grp, dl_mask);

    prime_channels_from_mask<
        Ch<ChPusch, &cg_t::get_pusch_params>,
        Ch<ChPrach, &cg_t::get_prach_params>,
        Ch<ChPucch, &cg_t::get_pucch_params>,
        Ch<ChSrs,   &cg_t::get_srs_params>
    >(grp, ul_mask);
}

} // namespace

void PHY_module::enqueue_channel_tasks(sfn_slot_t ss_curr)
{
    const uint32_t ring_idx = ring_idx_from_slot(ss_curr.u32);
    auto& slot_store = slot_message_storage(ss_curr.u32);

    if (!has_real_phy_driver())
    {
        // No driver to publish to and no workers to run channel tasks: drop the slot.
        // Zero the counter and accumulators so the next slot at this ring index arms
        // cleanly. No serial replay runs on the msg thread.
        NVLOGW_FMT(TAG,
                   "task_framework: dropping slot at EOM - no real PHY driver "
                   "(test/init condition) ss=0x{:08X} ring_idx={}",
                   ss_curr.u32, ring_idx);
        release_stored_slot_messages(ss_curr.u32);
        task_pool_.tasks_in_flight[ring_idx].store(0, std::memory_order_release);
        task_pool_.channel_tasks_in_flight[ring_idx].store(0, std::memory_order_release);
        task_pool_.dlc_batch_accum[ring_idx].reset();
        task_pool_.ulc_batch_accum[ring_idx].reset();
        task_pool_.dlc_accumulated_bitmap[ring_idx] = 0;
        task_pool_.ulc_accumulated_bitmap[ring_idx] = 0;
        clear_direct_cplane_task_args_for_ring(ring_idx);
        task_pool_.direct_cplane_stale_slot_drops[ring_idx] = 0;
        task_pool_.direct_cplane_stale_dropped_slot[ring_idx] = SFN_SLOT_INVALID;
        return;
    }

    // Defense-in-depth: the sentinel must already be armed by reset_for_slot().
    // Check it before any counter-modifying work (slot-map setup is counter-neutral,
    // but the C-plane batch fire below increments tasks_in_flight). Catching a
    // missing/leaked sentinel here prevents fired batches from masking an off-by-k
    // counter (SENTINEL - k + n_batches >= SENTINEL) and avoids firing tasks into a
    // corrupt slot.
    if(task_pool_.tasks_in_flight[ring_idx].load(std::memory_order_acquire) < nv::SlotTaskPool::SLOT_TASK_SENTINEL)
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "enqueue_channel_tasks: sentinel missing at ring_idx={} slot=0x{:08X} "
                                                "tasks_in_flight_={} - sentinel corrupted or reset_for_slot() guard missed",
                   ring_idx,
                   ss_curr.u32,
                   task_pool_.tasks_in_flight[ring_idx].load());
        return;
    }

    uint64_t ul_cplane_task_ts_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    // Direct FAPI-to-C-plane is the only mode: always wire up the early C-plane slot maps.
    {
        auto& slot_cmd = slot_command_array.at(current_slot_cmd_index);
        uint32_t slot_interval = 0;
        sfn_slot_t ss_tick_snapshot{};
        std::chrono::nanoseconds curr_tick_snapshot{};
        {
            const std::lock_guard<std::mutex> lock(tick_lock);
            ss_tick_snapshot = ss_tick.load();
            slot_interval = get_fapi_latency(ss_curr);
            curr_tick_snapshot = current_tick_;
            slot_cmd.tick_original =
                curr_tick_snapshot -
                std::chrono::nanoseconds(slot_interval * mu_to_ns(tick_updater_.mu_highest_));
        }

        if(try_drop_stale_direct_cplane_slot(ss_curr, ring_idx, slot_interval,
                                             ss_tick_snapshot, curr_tick_snapshot))
        {
            return;
        }

        task_pool_.direct_cplane_stale_slot_drops[ring_idx] = 0;
        task_pool_.direct_cplane_stale_dropped_slot[ring_idx] = SFN_SLOT_INVALID;

        const auto t0 = std::chrono::nanoseconds(
            sfn_to_tai(ss_curr.u16.sfn, ss_curr.u16.slot,
                       slot_cmd.tick_original.count() + AppConfig::getInstance().getTaiOffset(),
                       gps_alpha_, gps_beta_, tick_updater_.mu_highest_) -
            AppConfig::getInstance().getTaiOffset());

        // UL_DCI-only cells also need a DL output buffer: PDCCH carries the UL
        // grant on the downlink and writes IQ samples into the DL output tensor.
        // The raw accumulated bitmaps are duplicate-delivery guards; early slot maps
        // must only count cells that have current C-plane work and a live task arg.
        uint64_t dl_cplane_cell_bitmap = 0;
        uint64_t ul_cplane_cell_bitmap = 0;
        for (uint32_t cid = 0; cid < MAX_CELLS_PER_SLOT; ++cid)
        {
            const uint64_t cell_bit = uint64_t{1} << cid;
            const uint32_t arg_idx = ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT) + cid;

            if ((task_pool_.dlc_accumulated_bitmap[ring_idx] & cell_bit) != 0)
            {
                const auto& dl_arg = task_pool_.dlc_task_args[arg_idx];
                if (dl_arg.phy_inst != nullptr && (dl_arg.dl_tti != nullptr || dl_arg.ul_dci != nullptr))
                {
                    dl_cplane_cell_bitmap |= cell_bit;
                }
            }

            if ((task_pool_.ulc_accumulated_bitmap[ring_idx] & cell_bit) != 0)
            {
                const auto& ul_arg = task_pool_.ulc_task_args[arg_idx];
                if (ul_arg.phy_inst != nullptr && ul_arg.ul_tti != nullptr)
                {
                    ul_cplane_cell_bitmap |= cell_bit;
                }
            }
        }

        const auto slot_maps = l1_setup_early_cplane_slot_maps(
            PHYDriverProxy::getInstance().get_driver(),
            ss_curr.u16.sfn,
            ss_curr.u16.slot,
            t0,
            dl_cplane_cell_bitmap,
            ul_cplane_cell_bitmap);

        if (dl_cplane_cell_bitmap != 0 && slot_maps.dl == nullptr)
        {
            NVLOGW_FMT(TAG,
                       "enqueue_channel_tasks: direct C-plane DL slot map unavailable "
                       "slot=0x{:08X} ring_idx={} dl_bitmap=0x{:X}",
                       ss_curr.u32, ring_idx, dl_cplane_cell_bitmap);
        }
        if (ul_cplane_cell_bitmap != 0 && slot_maps.ul == nullptr)
        {
            NVLOGW_FMT(TAG,
                       "enqueue_channel_tasks: direct C-plane UL slot map unavailable "
                       "slot=0x{:08X} ring_idx={} ul_bitmap=0x{:X}",
                       ss_curr.u32, ring_idx, ul_cplane_cell_bitmap);
        }

        for (uint32_t cid = 0; cid < MAX_CELLS_PER_SLOT; ++cid)
        {
            const uint64_t cell_bit = uint64_t{1} << cid;
            if (dl_cplane_cell_bitmap & cell_bit)
            {
                task_pool_.dlc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cid].slot_map_dl =
                    slot_maps.dl;
            }
            if (ul_cplane_cell_bitmap & cell_bit)
            {
                task_pool_.ulc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cid].slot_map_ul =
                    slot_maps.ul;
            }
        }
    }

    // EOM planner: batches are pre-partitioned during accumulation; EOM only computes
    // final metadata and fires DL/UL batch tasks. Slot maps were set up once above.
    const auto plan_and_fire_eom_batches =
        [ring_idx, slot_u32 = ss_curr.u32](auto& accum,
                                           auto& batch_pool,
                                           uint8_t batch_size,
                                           std::string_view direction_tag,
                                           auto&& fire_batch) {
            const uint8_t n_total_cells = accum.n_pending;
            if (n_total_cells == 0) { return; }

            const uint8_t effective_batch_size = batch_size;
            const uint8_t total_batches = static_cast<uint8_t>(
                (n_total_cells + effective_batch_size - 1u) / effective_batch_size);

            const uint32_t batch_base = ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT);
            for (uint8_t local_batch_idx = 0; local_batch_idx < total_batches; ++local_batch_idx)
            {
                const uint32_t batch_idx = batch_base + local_batch_idx;
                if (batch_idx >= batch_pool.size())
                {
                    NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "enqueue_channel_tasks: {} batch_idx={} out of range [0,{}) "
                               "ring_idx={} slot=0x{:08X} - dropping remaining EOM batches",
                               direction_tag, batch_idx, batch_pool.size(), ring_idx, slot_u32);
                    accum.reset();
                    return;
                }
                const uint8_t n_cells_this_batch = batch_pool[batch_idx].n_cells;
                if (n_cells_this_batch == 0)
                {
                    NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "enqueue_channel_tasks: {} batch_idx={} has zero cells "
                               "ring_idx={} slot=0x{:08X} - dropping remaining EOM batches",
                               direction_tag, batch_idx, ring_idx, slot_u32);
                    accum.reset();
                    return;
                }
                fire_batch(local_batch_idx, local_batch_idx, total_batches, n_cells_this_batch);
            }
            accum.n_pending = 0;
        };
    // Fire all planned DL C-plane batch tasks for this slot.
    const uint64_t dl_cplane_task_ts_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    plan_and_fire_eom_batches(task_pool_.dlc_batch_accum[ring_idx],
                              task_pool_.dlc_batch_task_args,
                              config_options_.cplane_processing_dl_batch_size,
                              "DL",
                              [this, ring_idx, ss_curr, dl_cplane_task_ts_ns](uint8_t local_batch_idx,
                                                                              uint8_t batch_id,
                                                                              uint8_t total_batches,
                                                                              uint8_t n_cells_in_batch) {
                                  fire_cplane_batch<DlcBatchPolicy>(ring_idx, ss_curr,
                                                                    task_pool_.dlc_batch_accum,
                                                                    task_pool_.dlc_batch_task_args,
                                                                    local_batch_idx,
                                                                    n_cells_in_batch,
                                                                    batch_id,
                                                                    total_batches,
                                                                    dl_cplane_task_ts_ns);
                              });
    // Fire all planned UL C-plane batch tasks for this slot.
    plan_and_fire_eom_batches(task_pool_.ulc_batch_accum[ring_idx],
                              task_pool_.ulc_batch_task_args,
                              config_options_.cplane_processing_ul_batch_size,
                              "UL",
                              [this, ring_idx, ss_curr, ul_cplane_task_ts_ns](uint8_t local_batch_idx,
                                                                              uint8_t batch_id,
                                                                              uint8_t total_batches,
                                                                              uint8_t n_cells_in_batch) {
                                  fire_cplane_batch<UlcBatchPolicy>(ring_idx, ss_curr,
                                                                    task_pool_.ulc_batch_accum,
                                                                    task_pool_.ulc_batch_task_args,
                                                                    local_batch_idx,
                                                                    n_cells_in_batch,
                                                                    batch_id,
                                                                    total_batches,
                                                                    ul_cplane_task_ts_ns);
                              });
    // Split-phase TX_DATA H2D (Phase 2): the channel-task PDSCH body consumes the
    // staged GPU TB pointers, so launch H2D for every slot.
    launch_tx_data_h2d(ss_curr, ring_idx);
    // -----------------------------------------------------------------------
    // DL channel mask scan — counts come from the store-time sidecar
    // (populated in store_message); no payload walk / nPDUsOfEachType[] read.
    // -----------------------------------------------------------------------
    constexpr uint32_t CH_PDCCH      = 1U << DL_TTI_PDU_TYPE_PDCCH;
    constexpr uint32_t CH_PDSCH      = 1U << DL_TTI_PDU_TYPE_PDSCH;
    constexpr uint32_t CH_CSIRS      = 1U << DL_TTI_PDU_TYPE_CSI_RS;
    constexpr uint32_t CH_SSB        = 1U << DL_TTI_PDU_TYPE_SSB;
    constexpr uint32_t CH_DL_TTI_ALL = CH_PDCCH | CH_PDSCH | CH_CSIRS | CH_SSB;

    uint32_t active_dl_ch_mask = 0;
    bool has_pdcch_dl = false;

    for(uint16_t i = 0;
        i < slot_store.dl_tti_count() && active_dl_ch_mask != CH_DL_TTI_ALL;
        ++i)
    {
        const NvDlTtiPduCounts counts = slot_store.dl_tti_pdu_counts(i);
        if(counts.by_type[DL_TTI_NPDUS_IDX_PDCCH] > 0)
        {
            active_dl_ch_mask |= CH_PDCCH;
            has_pdcch_dl = true;
        }
        if(counts.by_type[DL_TTI_NPDUS_IDX_PDSCH] > 0)
        {
            active_dl_ch_mask |= CH_PDSCH;
        }
        if(counts.by_type[DL_TTI_NPDUS_IDX_CSI_RS] > 0)
        {
            active_dl_ch_mask |= CH_CSIRS;
        }
        if(counts.by_type[DL_TTI_NPDUS_IDX_SSB] > 0)
        {
            active_dl_ch_mask |= CH_SSB;
        }
    }

    // DlDCIs consistency check -- kept out of the mask scan above, which
    // early-exits once all channel types are seen and would skip later cells.
    // Index 4 (DlDCIs) is an aggregate DCI count across PDCCH PDUs, not a PDU
    // type: non-zero with a PDCCH PDU present is normal (DCIs are parsed per-PDU
    // via num_dl_dci). Warn only on DlDCIs with no PDCCH PDU to carry them --
    // malformed FAPI. No-op on 10.02 (the sidecar never sets index 4).
    for(uint16_t i = 0; i < slot_store.dl_tti_count(); ++i)
    {
        const NvDlTtiPduCounts counts = slot_store.dl_tti_pdu_counts(i);
        if(counts.by_type[DL_TTI_NPDUS_IDX_DlDCIs] != 0 &&
           counts.by_type[DL_TTI_NPDUS_IDX_PDCCH] == 0)
        {
            NVLOGW_FMT(TAG,
                       "enqueue_channel_tasks: DlDCIs={} without a PDCCH PDU; "
                       "slot=0x{:08X}",
                       counts.by_type[DL_TTI_NPDUS_IDX_DlDCIs],
                       ss_curr.u32);
        }
    }

    // UL_DCI.req presence alone requires a PDCCH task (carries DL-DCI payloads).
    // No #ifdef: this TU is compiled only under ENABLE_FAPI_STORE_REPLAY (CMake-gated).
    const bool has_ul_dci = slot_store.ul_dci_count() > 0;
    if(has_ul_dci)
    {
        active_dl_ch_mask |= CH_PDCCH;
        // Touch pdcch_params so the slot command pre-allocates storage for the
        // UL_DCI-only path — the PDCCH worker writes here when no DL_TTI PDU
        // pre-allocated for this slot.
        slot_command_array[current_slot_cmd_index].cell_groups.ensure_pdcch_params(slot_command_api::PDCCH_UL);
    }

    bool has_dl_bfw = slot_store.dl_bfw_count() > 0;

    // -----------------------------------------------------------------------
    // UL channel mask scan — PUCCH task is enqueued if either F01 or F234
    // PDUs are present; both format groups are handled by process_aggr_pucch_channel.
    // Counts come from the store-time sidecar (same as the DL scan above).
    // UL_TTI_NPDUS_IDX_MsgA_PUSCH is always 0 by contract; not tested.
    // -----------------------------------------------------------------------
    uint32_t active_ul_ch_mask = 0;

    for(uint16_t i = 0; i < slot_store.ul_tti_count(); ++i)
    {
        const NvUlTtiPduCounts counts = slot_store.ul_tti_pdu_counts(i);
        if(counts.by_type[UL_TTI_NPDUS_IDX_PUSCH] > 0) { active_ul_ch_mask |= CH_UL_PUSCH; }
        if(counts.by_type[UL_TTI_NPDUS_IDX_PRACH] > 0) { active_ul_ch_mask |= CH_UL_PRACH; }
        if(counts.pucch() > 0)                         { active_ul_ch_mask |= CH_UL_PUCCH; }
        if(counts.by_type[UL_TTI_NPDUS_IDX_SRS] > 0)   { active_ul_ch_mask |= CH_UL_SRS; }
    }

    bool has_ul_bfw = slot_store.ul_bfw_count() > 0;
    if(has_dl_bfw && has_ul_bfw)
    {
        NVLOGW_FMT(TAG,
                   "enqueue_channel_tasks: DL_BFW and UL_BFW are both present in slot=0x{:08X}; "
                   "skipping BFW parallel aggregation for this slot",
                   ss_curr.u32);
        has_dl_bfw = false;
        has_ul_bfw = false;
    }

    // Prime all active channel entries on the message thread before any worker
    // reads cell_groups. Grouping the channel_idx writes here (after both scan
    // loops) instead of inline inside the loops means all writes hit the same
    // cell_group_command object consecutively — one cache-coherent pass.
    prime_slot_command_channel_params<CH_PDSCH, CH_CSIRS, CH_SSB,
                                      CH_UL_PUSCH, CH_UL_PRACH, CH_UL_PUCCH, CH_UL_SRS>(
        slot_command_array[current_slot_cmd_index].cell_groups,
        active_dl_ch_mask,
        active_ul_ch_mask,
        has_pdcch_dl,
        has_ul_dci);

    // Re-derive channel masks from slot_command.channel_idx so that task counts and
    // slot_type reflect what was actually registered, not raw FAPI PDU counts.
    {
        const auto& ch_idx = slot_command_array[current_slot_cmd_index].cell_groups.channel_idx;
        constexpr auto NONE = slot_command_api::channel_type::NONE;

        active_dl_ch_mask = 0;
        if(ch_idx[slot_command_api::PDCCH_DL] != NONE ||
           ch_idx[slot_command_api::PDCCH_UL] != NONE) { active_dl_ch_mask |= CH_PDCCH; }
        if(ch_idx[slot_command_api::PDSCH]  != NONE) { active_dl_ch_mask |= CH_PDSCH; }
        if(ch_idx[slot_command_api::CSI_RS] != NONE) { active_dl_ch_mask |= CH_CSIRS; }
        if(ch_idx[slot_command_api::PBCH]   != NONE) { active_dl_ch_mask |= CH_SSB; }

        active_ul_ch_mask = 0;
        if(ch_idx[slot_command_api::PUSCH] != NONE) { active_ul_ch_mask |= CH_UL_PUSCH; }
        if(ch_idx[slot_command_api::PRACH] != NONE) { active_ul_ch_mask |= CH_UL_PRACH; }
        if(ch_idx[slot_command_api::PUCCH] != NONE) { active_ul_ch_mask |= CH_UL_PUCCH; }
        if(ch_idx[slot_command_api::SRS]   != NONE) { active_ul_ch_mask |= CH_UL_SRS; }
    }

    const int32_t n_dl_tasks =
        static_cast<int32_t>(std::popcount(active_dl_ch_mask)) + (has_dl_bfw ? 1 : 0);

    // Derive slot_type from the already-computed channel masks rather than the
    // Path-A-only is_dl_slot_ / is_ul_slot_ / is_csirs_slot_ flags (which are
    // never set in the store-and-replay path). CSI-RS is a downlink channel, so
    // any DL channel (CSI-RS included) or DL BFW makes this a DL slot — mirroring
    // legacy, which set is_dl_slot on every DL_TTI and is_csirs_slot as a separate
    // supplementary marker. The CSIRS bit is set in addition to DL, not instead of
    // it: slot_type_from_mask() classifies on DL/UL, so excluding CSI-RS from the
    // DL bit would misclassify CSI-RS-only slots as SLOT_NONE and drop the work.
    const nv::SlotTypeMask slot_type =
        static_cast<nv::SlotTypeMask>(
            (((active_dl_ch_mask != 0) || has_dl_bfw) ? to_underlying(nv::SlotTypeMask::DL) : 0u) |
            (((active_ul_ch_mask != 0) || has_ul_bfw) ? to_underlying(nv::SlotTypeMask::UL) : 0u) |
            (((active_dl_ch_mask & CH_CSIRS) != 0) ? to_underlying(nv::SlotTypeMask::CSIRS) : 0u));

    const int32_t n_ul_tasks =
        static_cast<int32_t>(std::popcount(active_ul_ch_mask)) + (has_ul_bfw ? 1 : 0);

    const int32_t n_total = n_dl_tasks + n_ul_tasks;

    NVLOGD_FMT(TAG,
               "task_framework: enqueue_channel_tasks slot=0x{:08X} ring_idx={} "
               "active_dl_ch_mask=0x{:X} active_ul_ch_mask=0x{:X} "
               "has_dl_bfw={} has_ul_bfw={} n_dl_tasks={} n_ul_tasks={} n_total={}",
               ss_curr.u32,
               ring_idx,
               active_dl_ch_mask,
               active_ul_ch_mask,
               has_dl_bfw,
               has_ul_bfw,
               n_dl_tasks,
               n_ul_tasks,
               n_total);

    // -----------------------------------------------------------------------
    // Two-counter publish path: channel-aggr tasks share channel_tasks_in_flight[ring_idx]
    // (publish gate) and, with C-plane batches, tasks_in_flight[ring_idx] (cleanup gate).
    // Push one worker task per active channel. When the channel counter drains,
    // publish_slot_command enqueues U-plane work (independent of C-plane); when the all-tasks
    // counter drains, finalize_slot_cleanup releases stored FAPI messages. Either runs inline
    // below if its set already drained before the sentinels drop, else from the last
    // completing task (channel -> on_slot_channel_aggr_complete, C-plane ->
    // on_slot_channel_task_complete). publish_slot_command applies the direct C-plane skip
    // mask (C-plane already published by the batch tasks) and handles the empty-slot
    // (channel_array_size == 0) case itself.
    // No serial replay runs on the msg thread.
    //
    // Capture the submitted slot command index, then advance to the next buffer.
    // Slot maps keep aggr_slot_info pointers into slot_command_array[slot_cmd_idx],
    // so do not reset that submitted buffer here or in the completion callback.
    // Reset the newly-current buffer instead so the next replay starts cleanly.
    // -----------------------------------------------------------------------
    const uint32_t slot_cmd_idx = current_slot_cmd_index;
    update_slot_cmds_indexes();
    // Next-slot buffer reset is deferred until after channel-task push. It
    // targets the advanced current_slot_cmd_index, distinct from the captured
    // slot_cmd_idx the pushed tasks read, so it need not gate the push.
    // Reset deferred to on_slot_channel_task_complete()

    // Initialise slot type and timing on the captured slot-command entry before
    // any worker tasks are enqueued.  cell_groups.slot and every cell_sub_command
    // .slot mirror the same slot_info so per-cell code sees consistent timing.
    // slot_cmd_idx is always valid: update_slot_cmds_indexes() wraps it modulo
    // slot_command_array.size(), so an out-of-range index is impossible unless
    // the array is empty (caught below).
    assert(slot_cmd_idx < slot_command_array.size());
    const auto slot_info = slot_type_from_mask(slot_type, ss_curr.u16.sfn, ss_curr.u16.slot);
    auto& slot_cmd_entry = slot_command_array[slot_cmd_idx];
    slot_cmd_entry.cell_groups.slot = slot_info;
    // cells is built once at construction with exactly phy_instances_.size()
    // entries (lockstep with phy_refs_ / PHY_instances()), and is never resized,
    // so iterate the vector directly — no bounds-checked .at() needed.
    for(auto& cell : slot_cmd_entry.cells)
    {
        cell.slot = slot_info;
    }

    // Snapshot per-slot diagnostic state and stamp tick_original on the msg
    // thread BEFORE any worker can run. publish_slot_command will read these
    // values from slot_telemetry_[ring_idx] without re-reading singleton
    // scalars that the msg thread may have already overwritten for the next
    // in-flight slot.
    stash_slot_telemetry_for_worker_publish(ring_idx, ss_curr, slot_cmd_idx);

    // ORDERING CONTRACT: dl/ul_aggr_task_args[ring_idx] must be fully visible
    // to any worker thread that executes the tasks enqueued below.  Visibility is
    // guaranteed by the mutex inside l1_push_task_dl / l1_push_task_ul (TaskList
    // lock/unlock), which provides acquire/release ordering between this write and
    // the worker's read.  Do not change the push mechanism without re-establishing
    // this guarantee.
    // .base is intentionally omitted — C++20 designated init leaves it
    // default-initialized to {TaskArgType::DL_AGGR} / {TaskArgType::UL_AGGR}
    // from the default member initializer in the struct definition.
    // split_phase_ok mirrors launch_tx_data_h2d()'s arm outcome (is_valid() ==
    // expected_cells > 0) and is used only to size n_staged_tb_ptrs below. TX_DATA
    // release ownership (`deferred`) is armed inside launch_tx_data_h2d() on the
    // H2D-launch success path only — not here — so a failed/absent launch leaves
    // `deferred` false and the inline path releases the lane.
    const bool split_phase_ok = tx_data_ring_.is_valid(ring_idx);
    task_pool_.dl_aggr_task_args[ring_idx] = {
        .phy_module         = this,
        .dispatch           = &DL_DISPATCH_TABLE,
        .slot_u32           = ss_curr.u32,
        .ring_idx           = ring_idx,
        .active_ch_mask     = active_dl_ch_mask,
        .slot_cmd_idx       = slot_cmd_idx,
        .slot_type_mask     = slot_type,
        .fapi_to_cplane_direct = true,  // mandatory mode
        .staged_tb_ptrs     = tx_data_ring_.gpu_ptrs(ring_idx),
        .n_staged_tb_ptrs   = split_phase_ok ? static_cast<uint32_t>(MAX_CELLS_PER_SLOT) : 0U,
        .staged_tx_msg_bufs = tx_data_ring_.msg_bufs(ring_idx)};
    task_pool_.ul_aggr_task_args[ring_idx] = {
        .phy_module        = this,
        .dispatch          = &UL_DISPATCH_TABLE,
        .slot_u32          = ss_curr.u32,
        .ring_idx          = ring_idx,
        .active_ul_ch_mask = active_ul_ch_mask,
        .slot_cmd_idx      = slot_cmd_idx,
        .slot_type_mask    = slot_type};

    // Add the channel task count on top of the sentinel (and on top of any DLC/ULC
    // C-plane batch tasks already fired by plan_and_fire_eom_batches above, which
    // incremented the counter at fire time).
    task_pool_.tasks_in_flight[ring_idx].fetch_add(n_total, std::memory_order_acq_rel);
    // Channel-only counter: counts channel-aggr tasks only, not C-plane batches.
    // Its 0-crossing fires publish independent of C-plane completion.
    task_pool_.channel_tasks_in_flight[ring_idx].fetch_add(n_total, std::memory_order_acq_rel);
    NVLOGD_FMT(TAG,
               "task_framework: EOM fetch_add slot=0x{:08X} ring_idx={} n_total_added={}",
               ss_curr.u32,
               ring_idx,
               n_total);

    // OPT-2: Capture a single timestamp shared by all tasks in this slot batch.
    // All DL and UL channel tasks for this EOM event use the same ts_exec_ns,
    // eliminating repeated system_clock::now() vDSO calls inside the loop.
    const uint64_t ts_exec_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    // -----------------------------------------------------------------------
    // OPT-1: Batch-enqueue DL channel tasks under a single TaskList lock.
    // Split-phase: Phase 2 batched H2D already ran in launch_tx_data_h2d.
    // Non-split-phase: H2D was launched on the message thread above — no worker task needed.
    // -----------------------------------------------------------------------
    TaskSpec dl_specs[TaskSpec::BULK_MAX];
    int      n_dl_specs = 0;

    if (active_dl_ch_mask & CH_PDSCH)
    {
        dl_specs[n_dl_specs++] = {"task_aggr_pdsch", task_work_fn_aggr_pdsch, &task_pool_.dl_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (active_dl_ch_mask & CH_CSIRS)
    {
        dl_specs[n_dl_specs++] = {"task_aggr_csirs", task_work_fn_aggr_csirs, &task_pool_.dl_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (active_dl_ch_mask & CH_PDCCH)
    {
        dl_specs[n_dl_specs++] = {"task_aggr_pdcch", task_work_fn_aggr_pdcch, &task_pool_.dl_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (active_dl_ch_mask & CH_SSB)
    {
        dl_specs[n_dl_specs++] = {"task_aggr_ssb", task_work_fn_aggr_ssb, &task_pool_.dl_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (has_dl_bfw)
    {
        dl_specs[n_dl_specs++] = {"task_aggr_dlbfw", task_work_fn_aggr_dlbfw, &task_pool_.dl_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }

    const int n_dl_pushed =
        (n_dl_specs > 0) ? PHYDriverProxy::getInstance().l1_push_new_dl_tasks_bulk(ts_exec_ns, {dl_specs, static_cast<std::size_t>(n_dl_specs)}) : 0;

    // For any DL task that could not be allocated, fire the completion callback
    // directly so that on_slot_channel_task_complete() can still release the slot.
    for(int i = n_dl_pushed; i < n_dl_specs; ++i)
    {
        NVLOGW_FMT(TAG, "enqueue_channel_tasks: DL task '{}' ring_idx={} dropped; "
                        "calling completion callback",
                   dl_specs[i].name,
                   ring_idx);
        on_slot_channel_aggr_complete(ring_idx, task_pool_.dl_aggr_task_args[ring_idx].slot_u32);
    }

    NVLOGD_FMT(TAG, "task_framework: pushed {} of {} DL specs ring_idx={} slot=0x{:08X}", n_dl_pushed, n_dl_specs, ring_idx, ss_curr.u32);

    // -----------------------------------------------------------------------
    // OPT-1: Batch-enqueue UL channel tasks under a single TaskList lock.
    // -----------------------------------------------------------------------
    TaskSpec ul_specs[TaskSpec::BULK_MAX];
    int      n_ul_specs = 0;

    if (active_ul_ch_mask & CH_UL_PUSCH)
    {
        ul_specs[n_ul_specs++] = {"task_aggr_pusch", task_work_fn_aggr_pusch, &task_pool_.ul_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (active_ul_ch_mask & CH_UL_PRACH)
    {
        ul_specs[n_ul_specs++] = {"task_aggr_prach", task_work_fn_aggr_prach, &task_pool_.ul_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (active_ul_ch_mask & CH_UL_PUCCH)
    {
        ul_specs[n_ul_specs++] = {"task_aggr_pucch", task_work_fn_aggr_pucch, &task_pool_.ul_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (active_ul_ch_mask & CH_UL_SRS)
    {
        ul_specs[n_ul_specs++] = {"task_aggr_srs", task_work_fn_aggr_srs, &task_pool_.ul_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }
    if (has_ul_bfw)
    {
        ul_specs[n_ul_specs++] = {"task_aggr_ulbfw", task_work_fn_aggr_ulbfw, &task_pool_.ul_aggr_task_args[ring_idx], INVALID_WORKER_ID};
    }

    const int n_ul_pushed =
        (n_ul_specs > 0) ? PHYDriverProxy::getInstance().l1_push_new_ul_tasks_bulk(ts_exec_ns, {ul_specs, static_cast<std::size_t>(n_ul_specs)}) : 0;

    for(int i = n_ul_pushed; i < n_ul_specs; ++i)
    {
        NVLOGW_FMT(TAG, "enqueue_channel_tasks: UL task '{}' ring_idx={} dropped; "
                        "calling completion callback",
                   ul_specs[i].name,
                   ring_idx);
        on_slot_channel_aggr_complete(ring_idx, task_pool_.ul_aggr_task_args[ring_idx].slot_u32);
    }

    NVLOGD_FMT(TAG, "task_framework: pushed {} of {} UL specs ring_idx={} slot=0x{:08X}", n_ul_pushed, n_ul_specs, ring_idx, ss_curr.u32);

    // Drop BOTH sentinels. Publish is gated on the channel-only counter (fires as soon
    // as all channel-aggr tasks drain, independent of C-plane); cleanup (release) is gated on
    // the all-tasks counter (waits for C-plane too, the last stored-FAPI reader). If either
    // set already drained before its sentinel-drop, that step runs inline here; otherwise the
    // last completing task fires it (channel task -> on_slot_channel_aggr_complete;
    // C-plane -> on_slot_channel_task_complete). Publish before cleanup is guaranteed because
    // all >= channel at all times.
    const int32_t after_ch =
        task_pool_.channel_tasks_in_flight[ring_idx].fetch_sub(
            nv::SlotTaskPool::SLOT_TASK_SENTINEL, std::memory_order_acq_rel)
        - nv::SlotTaskPool::SLOT_TASK_SENTINEL;
    if (after_ch == 0)
    {
        NVLOGD_FMT(TAG, "task_framework: publish inline (channel drained) ring_idx={} slot=0x{:08X}",
                   ring_idx, ss_curr.u32);
        merge_ul_order_scratch(ring_idx);
        publish_slot_command(ring_idx);
    }
    // Ordering invariant: drop the all-tasks sentinel only AFTER publish returns.
    // Until this drop, tasks_in_flight stays >= SENTINEL, so finalize_slot_cleanup
    // (release_stored_slot_messages) cannot fire on any thread while publish runs.
    // Keep this fetch_sub below publish_slot_command.
    const int32_t after =
        task_pool_.tasks_in_flight[ring_idx].fetch_sub(
            nv::SlotTaskPool::SLOT_TASK_SENTINEL, std::memory_order_acq_rel)
        - nv::SlotTaskPool::SLOT_TASK_SENTINEL;
    NVLOGD_FMT(TAG,
               "task_framework: sentinel dropped slot=0x{:08X} ring_idx={} after_ch={} after_all={}",
               ss_curr.u32, ring_idx, after_ch, after);
    if (after == 0)
    {
        NVLOGD_FMT(TAG,
                   "task_framework: cleanup inline (all drained) slot=0x{:08X} ring_idx={}",
                   ss_curr.u32, ring_idx);
        finalize_slot_cleanup(ring_idx, ss_curr.u32);
    }

    // Reset the next-slot buffer last (off the publish path). Targets the advanced
    // current_slot_cmd_index, distinct from the captured slot_cmd_idx workers/submit
    // read — different slot_command_array entries, so no race with a concurrent
    // worker submit in the after_val > 0 case.
    group_command()->reset();
    for(uint32_t i = 0; i < static_cast<uint32_t>(phy_refs_.size()); ++i)
    {
        cell_sub_command(i).reset();
    }
}

/**
 * @brief Called by every C-plane batch task and EOM channel-aggr task on completion.
 *
 * Atomically decrements @c task_pool_.tasks_in_flight[ring_idx]. When the counter
 * reaches zero every task (channel-aggr + C-plane) is done, so @c finalize_slot_cleanup
 * releases stored FAPI messages. The U-plane command was already published by
 * @c publish_slot_command when the channel-only counter drained (independent of C-plane);
 * publish applied the direct C-plane skip mask, so the C-plane already published by the
 * batch tasks was not re-enqueued. The driver may still consume aggr_slot_info pointers into
 * that command buffer; the per-slot buffer reset happens when the ring advances to the next
 * slot command. If every task drained before the
 * sentinel was dropped, the msg thread already published inline in
 * enqueue_channel_tasks and this callback does not fire for that slot.
 *
 * Must be called exactly once per task, from the task's work function, after the
 * task has finished writing to shared structures.
 *
 * @param ring_idx  Index into @c task_pool_.tasks_in_flight (slot % SLOT_STORAGE_DEPTH).
 * @param slot_u32  Packed SFN/slot — passed to release helpers for logging.
 */
void PHY_module::on_slot_channel_task_complete(uint32_t ring_idx, uint32_t slot_u32)
{
    // C-plane batch completion path: decrements the all-tasks counter ONLY. It must never
    // touch channel_tasks_in_flight (that would re-fire publish). Reaching 0 means every
    // task (channel + C-plane) is done -> finalize_slot_cleanup (release stored FAPI).
    const int32_t remaining =
        task_pool_.tasks_in_flight[ring_idx].fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining < 0) {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_framework: tasks_in_flight underflow ring_idx={} slot=0x{:08X} remaining={}"
                   " - double-completion bug; ignoring",
                   ring_idx, slot_u32, remaining);
        return;
    }
    if (remaining == 0)
    {
        NVLOGD_FMT(TAG, "task_framework: all tasks complete (C-plane last) ring_idx={} slot=0x{:08X}",
                   ring_idx, slot_u32);
        finalize_slot_cleanup(ring_idx, slot_u32);
    }
}

void PHY_module::on_slot_channel_aggr_complete(uint32_t ring_idx, uint32_t slot_u32)
{
    // Channel-aggr completion path: decrement BOTH counters. Channel 0-crossing publishes
    // (independent of C-plane); all-tasks 0-crossing finalizes. all >= channel at all times,
    // so if this task crosses both it publishes then finalizes in order.
    const int32_t ch_remaining =
        task_pool_.channel_tasks_in_flight[ring_idx].fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (ch_remaining < 0) {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_framework: channel_tasks_in_flight underflow ring_idx={} slot=0x{:08X} remaining={}"
                   " - double-completion bug; ignoring channel counter",
                   ring_idx, slot_u32, ch_remaining);
    }
    else if (ch_remaining == 0)
    {
        NVLOGD_FMT(TAG, "task_framework: channel tasks complete ring_idx={} slot=0x{:08X}; publishing",
                   ring_idx, slot_u32);
        merge_ul_order_scratch(ring_idx);
        publish_slot_command(ring_idx);
    }

    // Ordering invariant: this task keeps its +1 on tasks_in_flight until AFTER publish
    // returns, so the counter stays >= 1 for the whole publish call and finalize_slot_cleanup
    // (release_stored_slot_messages) cannot fire on any thread while publish runs. Keep this
    // decrement below publish_slot_command.
    const int32_t remaining =
        task_pool_.tasks_in_flight[ring_idx].fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining < 0) {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "task_framework: tasks_in_flight underflow ring_idx={} slot=0x{:08X} remaining={}"
                   " - double-completion bug; ignoring",
                   ring_idx, slot_u32, remaining);
        return;
    }
    if (remaining == 0)
    {
        NVLOGD_FMT(TAG, "task_framework: all tasks complete (channel last) ring_idx={} slot=0x{:08X}",
                   ring_idx, slot_u32);
        finalize_slot_cleanup(ring_idx, slot_u32);
    }
}

void PHY_module::finalize_slot_cleanup(uint32_t ring_idx, uint32_t slot_u32)
{
    // All tasks (channel + C-plane) are done. C-plane was the last reader of the stored FAPI,
    // so releasing storage is now safe. Per-PHY reset_slot(false) belongs here too once
    // implemented (GT-11913); today submit's success path does no reset_slot.
    NVLOGD_FMT(TAG, "task_framework: finalize_slot_cleanup ring_idx={} slot=0x{:08X}", ring_idx, slot_u32);
    release_stored_slot_messages(slot_u32);
}

// ---------------------------------------------------------------------------
// UL task framework — accumulate_ulc_for_cell
// ---------------------------------------------------------------------------

/**
 * @brief Accumulate a per-cell UL C-plane entry for EOM-only batch scheduling.
 *
 * Called from @c process_fapi_messages() on UL_TTI.req arrival (before EOM).
 * Guards against duplicate UL_TTI.req via @c task_pool_.ulc_accumulated_bitmap[ring_idx].
 * Stores per-cell data in ulc_task_args[ring_idx * MAX_CELLS_PER_SLOT + cell_id] and
 * pre-partitions cell_id into ulc_batch_task_args for EOM firing. enqueue_channel_tasks()
 * only finalizes metadata and fires the prebuilt ULC batches.
 *
 * tasks_in_flight[ring_idx] is incremented by 1 per batch (not per cell).
 *
 * @param cell_id  Cell index from the UL_TTI.req message.
 * @param ss_curr  Current packed SFN/slot.
 */
void PHY_module::accumulate_ulc_for_cell(uint32_t cell_id, sfn_slot_t ss_curr)
{
    if(cell_id >= MAX_CELLS_PER_SLOT)
    {
        NVLOGW_FMT(TAG, "accumulate_ulc_for_cell: cell_id={} out of range slot=0x{:08X}", cell_id, ss_curr.u32);
        return;
    }
    if (!has_real_phy_driver())
    {
        return;
    }

    const uint32_t ring_idx = ring_idx_from_slot(ss_curr.u32);
    const uint32_t arg_idx = ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT) + cell_id;

    // Duplicate-cell guard: skip if this cell's UL_TTI.req was already accumulated
    // for this ring slot (e.g. duplicate FAPI delivery).
    const uint64_t cell_bit = uint64_t{1} << cell_id;
    if(task_pool_.ulc_accumulated_bitmap[ring_idx] & cell_bit)
    {
        NVLOGW_FMT(TAG,
                   "accumulate_ulc_for_cell: cell_id={} slot=0x{:08X} - "
                   "duplicate UL_TTI.req for this ring slot, skipping",
                   cell_id,
                   ss_curr.u32);
        return;
    }
    task_pool_.ulc_accumulated_bitmap[ring_idx] |= cell_bit;
    // Bitmap is set before the early-exit PDU check below, intentionally:
    // a UL_TTI.req with no PDUs for this cell is still a valid delivery.
    // Marking the bit now ensures a duplicate UL_TTI.req for the same ring slot is
    // suppressed even when no batch task is enqueued for this cell.

    auto& slot_store = slot_message_storage(ss_curr.u32);

    // Locate this cell's UL_TTI.req in the ring store.
    phy_mac_msg_desc* ul_tti     = nullptr;
    uint16_t          ul_tti_idx = 0;
    const uint16_t    n_ul       = slot_store.ul_tti_count();
    for(uint16_t i = 0; i < n_ul; ++i)
    {
        if(slot_store.ul_tti_messages()[i].cell_id == static_cast<int>(cell_id))
        {
            ul_tti     = &slot_store.ul_tti_messages()[i];
            ul_tti_idx = i;
            break;
        }
    }

    if(ul_tti == nullptr)
    {
        // UL_TTI.req not yet stored or has no entry for this cell — skip accumulation.
        // Bitmap already set above so a duplicate delivery is still suppressed.
        // This guards against storing {nullptr} in ulc_task_args and firing a batch
        // task that would call build_and_send_ul_cplane(nullptr).
        NVLOGW_FMT(TAG,
                   "accumulate_ulc_for_cell: cell_id={} slot=0x{:08X} ring_idx={} - "
                   "UL_TTI.req not found in ring store; skipping batch accumulation",
                   cell_id,
                   ss_curr.u32,
                   ring_idx);
        task_pool_.ulc_task_args[arg_idx] = {};
        return;
    }

    // Gate on a non-zero PDU count so that a UL_TTI.req with no PDUs does not
    // fire a degenerate batch (matches the channel mask scan in
    // enqueue_channel_tasks). Uses the store-time sidecar — no payload walk.
    if(!slot_store.ul_tti_pdu_counts(ul_tti_idx).any_channel())
    {
        task_pool_.ulc_task_args[arg_idx] = {};
        return;
    }

    // Store per-cell data in ring-indexed storage — no cross-slot aliasing possible.
    // Accessed by su_process_ulc_cell inside the batch worker.
    auto& arg = task_pool_.ulc_task_args[arg_idx];
    arg       = {&PHY_instances()[cell_id].get(), ul_tti};

    // Pre-partition this cell into its final EOM batch slot so EOM only fires.
    auto& accum = task_pool_.ulc_batch_accum[ring_idx];
    const uint8_t effective_batch_size = config_options_.cplane_processing_ul_batch_size;
    const uint8_t cell_ord = accum.n_pending;
    const uint8_t local_batch_idx = static_cast<uint8_t>(cell_ord / effective_batch_size);
    const uint8_t cell_offset = static_cast<uint8_t>(cell_ord % effective_batch_size);
    const uint32_t batch_base = ring_idx * static_cast<uint32_t>(MAX_CELLS_PER_SLOT);
    const uint32_t batch_idx = batch_base + local_batch_idx;
    if (batch_idx >= task_pool_.ulc_batch_task_args.size())
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "accumulate_ulc_for_cell: batch_idx={} out of range [0,{}) ring_idx={} "
                   "slot=0x{:08X} cell_id={} n_pending={}",
                   batch_idx, task_pool_.ulc_batch_task_args.size(), ring_idx,
                   ss_curr.u32, cell_id, accum.n_pending);
        return;
    }
    auto& batch_arg = task_pool_.ulc_batch_task_args[batch_idx];
    batch_arg.cell_ids[cell_offset] = static_cast<uint8_t>(cell_id);
    batch_arg.n_cells = static_cast<uint8_t>(cell_offset + 1u);
    ++accum.n_pending;

    NVLOGD_FMT(TAG,
               "task_framework: accumulated ULC cell_id={} ring_idx={} slot=0x{:08X} "
               "n_pending={} ul_eom_batch_size={}",
               cell_id, ring_idx, ss_curr.u32,
               accum.n_pending, config_options_.cplane_processing_ul_batch_size);
}

// ---------------------------------------------------------------------------
// store_slot_message — ring-buffer slot binding for incoming slot messages.
// Moved from nv_phy_module.cpp; only called from process_fapi_messages().

// ---------------------------------------------------------------------------

/**
 * @brief Store a slot-scoped message with per-slot ring tracking.
 *
 * Only current-slot and next-slot (early-arrival cache) messages are accepted.
 * Messages beyond that window are dropped. On ring index collision with a
 * different tracked slot, the old slot is force-released, the entry is rebound,
 * and processing continues with the incoming message. SLOT.indication and
 * SLOT.response are control-only in store-first mode and are not stored as
 * payload. The NVIPC buffer is released
 * on failure/drop via a scope guard, and retained only when payload storage
 * succeeds.
 *
 * @return true if stored or handled, false on drop/failure.
 */
bool PHY_module::store_slot_message(phy_mac_msg_desc& smsg, sfn_slot_t ss_msg)
{
    auto release_guard = make_scope_exit([this, &smsg]() {
        transport_wrapper().rx_release(smsg);
    });

    // Accept messages in a 3-slot window [curr-1, curr, curr+1]:
    //   lag  = slots ss_msg is BEHIND ss_curr  (0 = current, 1 = one slot old)
    //   lead = slots ss_msg is AHEAD  of ss_curr (0 = current, 1 = one slot early)
    // get_slot_interval() is wrap-aware and already handles SFN 1023->0 rollover.
    //
    // The one-slot backward tolerance (lag <= 1) mirrors the prev_slot acceptance in
    // check_sfn_slot() (scf_5g_fapi_phy.cpp:222-226) used by the non-store-replay path.
    // It is needed because ss_curr advances on the first SLOT.indication received from
    // any cell, so late-arriving payload messages from high-numbered cells (e.g. cell 18)
    // for the previous slot would otherwise be dropped as too_old, preventing EOM and
    // causing ring slot collisions.
    //
    // When ss_msg == ss_curr: lag=0, lead=0 -> condition false -> accepted.
    // drop_reason tie-break (lead==lag, i.e. ~half-hyperframe away, ~500 ms for mu=1):
    //   labels "too_future" — unreachable in practice.
    //
    // Note: a late curr-1 message accepted here after release_stored_slot_messages()
    // has already run for that slot will trigger a fresh ring allocation for curr-1.
    // Collision with slot curr+2 (same ring index for SLOT_STORAGE_DEPTH=3) requires
    // the message to be delayed by more than 2 slot periods — outside the operational
    // tolerance window.
    if(ss_curr.u32 != SFN_SLOT_INVALID)
    {
        const uint32_t lag  = get_slot_interval(ss_msg, ss_curr);
        const uint32_t lead = get_slot_interval(ss_curr, ss_msg);
        // lag > 2 is the maximum given SLOT_STORAGE_DEPTH=3; lag=3 would collide
        // with ring reuse at slot N+3 (same ring_idx as slot N).
        if(lag > 2 && lead > 1)
        {
            const char* drop_reason = (lead <= lag) ? "too_future" : "too_old";
            NVLOGW_FMT(TAG,
                       "store_slot_message: drop out-of-window slot message reason={} cell_id={} msg_id=0x{:02X} sfn={} slot={} curr=0x{:08X} lag={} lead={}",
                       drop_reason,
                       smsg.cell_id,
                       smsg.msg_id,
                       ss_msg.u16.sfn,
                       ss_msg.u16.slot,
                       ss_curr.u32,
                       lag,
                       lead);
            return false;
        }
    }

    const uint32_t sfn_slot_u32 = ss_msg.u32;
    const uint32_t ring_idx_msg = ring_idx_from_slot(sfn_slot_u32);
    if(task_pool_.ring_reuse_dropped_slot[ring_idx_msg] == sfn_slot_u32)
    {
        const uint64_t payload_drops = ++task_pool_.ring_reuse_payload_drops[ring_idx_msg];
        NVLOGW_FMT(TAG,
                   "store_slot_message: dropping message for previously dropped ring-reuse slot "
                   "slot=0x{:08X} msg_id=0x{:02X} cell_id={} ring_idx={} payload_drops={} "
                   "consecutive_slot_drops={} total_ring_drops={}",
                   sfn_slot_u32,
                   smsg.msg_id,
                   smsg.cell_id,
                   ring_idx_msg,
                   payload_drops,
                   task_pool_.ring_reuse_consecutive_drops[ring_idx_msg],
                   task_pool_.ring_reuse_total_drops[ring_idx_msg]);
        return false;
    }

    auto&          slot_store   = slot_message_storage(sfn_slot_u32);
    if(!slot_store.has_slot() || slot_store.tracked_slot() != sfn_slot_u32)
    {
        if(slot_store.has_slot())
        {
            const uint32_t old_tracked_slot_u32 = slot_store.tracked_slot();
            // old_ring_idx == ring_idx_from_slot(sfn_slot_u32) by construction:
            // both slots hash to the same ring entry, which is why we collided.
            const uint32_t old_ring_idx = ring_idx_from_slot(old_tracked_slot_u32);
            const int32_t  in_flight =
                task_pool_.tasks_in_flight[old_ring_idx].load(std::memory_order_acquire);

            NVLOGC_FMT(TAG,
                       "store_slot_message: ring slot collision check old=0x{:08X} new=0x{:08X} "
                       "ring_idx={} tasks_in_flight={} is_sentinel={} is_zero={}",
                       old_tracked_slot_u32,
                       sfn_slot_u32,
                       old_ring_idx,
                       in_flight,
                       nv::SlotTaskPool::is_sentinel(in_flight),
                       (in_flight == 0));

            if(in_flight != 0)
            {
                const uint64_t payload_drops = ++task_pool_.ring_reuse_payload_drops[old_ring_idx];
                NVLOGW_FMT(TAG,
                           "store_slot_message: ring slot collision old=0x{:08X} new=0x{:08X} "
                           "msg_id=0x{:02X} ring_idx={} tasks_in_flight={} is_sentinel={} - "
                           "dropping incoming message to avoid UAF payload_drops={} "
                           "consecutive_slot_drops={} total_ring_drops={}",
                           old_tracked_slot_u32,
                           sfn_slot_u32,
                           smsg.msg_id,
                           old_ring_idx,
                           in_flight,
                           nv::SlotTaskPool::is_sentinel(in_flight),
                           payload_drops,
                           task_pool_.ring_reuse_consecutive_drops[old_ring_idx],
                           task_pool_.ring_reuse_total_drops[old_ring_idx]);
                return false;
            }

            // tasks_in_flight == 0: old slot fully drained, safe to release and rebind.
            NVLOGW_FMT(TAG,
                       "store_slot_message: ring slot collision old=0x{:08X} new=0x{:08X} "
                       "msg_id=0x{:02X} ring_idx={} - old slot drained, force releasing",
                       old_tracked_slot_u32,
                       sfn_slot_u32,
                       smsg.msg_id,
                       old_ring_idx);
            reset_txdata_h2d_state_for_ring(old_ring_idx);
            release_stored_slot_messages(old_tracked_slot_u32);
            task_pool_.ring_reuse_consecutive_drops[old_ring_idx] = 0;
            task_pool_.ring_reuse_dropped_slot[old_ring_idx] = SFN_SLOT_INVALID;
        }
        //NVLOGD_FMT(TAG,
        //           "trace store_slot_message: bind ring slot old=0x{:08X} new=0x{:08X}",
        //           slot_store.has_slot() ? slot_store.tracked_slot() : 0xFFFFFFFFU,
        //           sfn_slot_u32);
        // Storage rebind + telemetry-snapshot reset in one owner (cannot desync).
        // The enclosing "slot changed" branch plus the collision-safety above
        // (tasks_in_flight eviction) guarantee a state change here, so the outcome
        // is FreshBind or Rebound - never AlreadyBound. Assert it so a future refactor
        // that reaches this site with the same slot bound fails loudly instead of
        // silently skipping the coupled storage + telemetry reset.
        [[maybe_unused]] const auto bind_outcome = bind_ring_slot(sfn_slot_u32);
        assert(bind_outcome != FapiSlotMessageStorage::BindOutcome::AlreadyBound &&
               "store_slot_message rebind must transition state (FreshBind/Rebound)");

        const uint32_t ring_idx_new = ring_idx_from_slot(sfn_slot_u32);
        task_pool_.fapi_first_arrival_ts[ring_idx_new] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch());
    }

    if(smsg.cell_id < 0)
    {
        NVLOGW_FMT(TAG, "store_slot_message: invalid cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
        return false;
    }
    const auto cell_id = static_cast<std::size_t>(smsg.cell_id);
    if(cell_id >= phy_refs_.size())
    {
        NVLOGW_FMT(TAG, "store_slot_message: invalid cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
        return false;
    }

    const bool stored = phy_refs_[cell_id].get().on_msg_to_store(smsg, ss_msg.u32);
    if(!stored)
    {
        NVLOGW_FMT(TAG, "store_slot_message: message not stored cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
        return false;
    }

    // Defer-commit: record that this staged cell has produced a slot message. The cell is
    // committed (enters the EOM expectation) at the NEXT SLOT.IND -- not here -- so established
    // cells never wait on a not-yet-producing cell, and the per-slot snapshot is fixed for the
    // whole slot (a late first-slot message cannot reopen an already-finalized slot). See
    // ActiveCellMask: committed = staged & produced.
    mark_cell_produced(static_cast<uint16_t>(cell_id));

    release_guard.release();
    return true;
}

// ---------------------------------------------------------------------------
// release_stored_slot_messages — return ring-slot IPC buffers to transport.

// ---------------------------------------------------------------------------

void PHY_module::release_stored_slot_messages(uint32_t slot_u32)
{
    const uint32_t idx = ring_idx_from_slot(slot_u32);

    auto& store = slot_message_storage_[idx];
    NVLOGD_FMT(TAG,
               "task_framework: release_stored_slot_messages slot=0x{:08X} ring_idx={} "
               "dl_tti={} ul_tti={} ul_dci={} tx_data={} dl_bfw={} ul_bfw={} tx_data_deferred={}",
               slot_u32,
               idx,
               store.dl_tti_count(),
               store.ul_tti_count(),
               store.ul_dci_count(),
               store.tx_data_count(),
               store.dl_bfw_count(),
               store.ul_bfw_count(),
               tx_data_ring_.is_deferred(idx));

    using MT = FapiSlotMessageStorage::MsgType;
    auto& tw = transport_wrapper();
    store.release_lane(MT::DL_TTI, tw);
    store.release_lane(MT::UL_TTI, tw);
    store.release_lane(MT::UL_DCI, tw);
    store.release_lane(MT::DL_BFW, tw);
    store.release_lane(MT::UL_BFW, tw);

    // TX_DATA release ownership (inline / message-thread path). While a slot is
    // armed deferred the H2D DMA may still be reading the IPC buffers, so skip
    // the inline release and let the H2D-completion callback own it. The callback
    // clears `deferred` (release store) AFTER its atomic release_lane, so a later
    // inline release for the same slot may observe deferred==false and call
    // release_lane here — that is safe: release_lane atomically exchanges the lane
    // count, so whichever of the callback / this path runs second finds count==0
    // and frees nothing. The `deferred` gate is thus an in-flight-DMA guard, not
    // the double-free guard (the atomic exchange in release_lane is).
    if(!tx_data_ring_.is_deferred(idx))
    {
        store.release_lane(MT::TX_DATA, tw);
    }
    store.unbind();
}

// ---------------------------------------------------------------------------
// submit_slot_command / submit_slot_command_empty — slot command dispatch.

// ---------------------------------------------------------------------------

// Thin wrapper around nv::stamp_ring_telemetry.
void PHY_module::stamp_ring_telemetry_on_ingest(sfn_slot_t ss_msg,
                                                int32_t    msg_id,
                                                bool       has_csirs)
{
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch());
    nv::stamp_ring_telemetry(slot_telemetry_[ring_idx_from_slot(ss_msg.u32)],
                             now, msg_id, has_csirs);
}

FapiSlotMessageStorage::BindOutcome PHY_module::bind_ring_slot(uint32_t slot_u32)
{
    // Single owner of the coupled ring-slot rebind: the storage binding and the
    // per-ring telemetry snapshot transition together so they cannot desync. The
    // snapshot is cleared on any bind that changes state (FreshBind/Rebound),
    // mirroring the storage's own clear-on-(re)bind in bind_to_slot().
    const auto outcome = slot_message_storage(slot_u32).bind_to_slot(slot_u32);
    if (outcome != FapiSlotMessageStorage::BindOutcome::AlreadyBound)
    {
        slot_telemetry_[ring_idx_from_slot(slot_u32)] = nv::SlotTelemetrySnapshot{};
    }
    return outcome;
}

void PHY_module::release_srs_indication_buffers(slot_command_api::srs_params& srs_params,
                                                const slot_command_api::slot_indication& slot)
{
    const auto n_cells = std::min<std::size_t>(srs_params.cell_grp_info.nCells,
                                               srs_params.cell_index_list.size());
    if (n_cells == 0u)
    {
        return;
    }

    NVLOGW_FMT(TAG,
               "SFN {}.{}: l1_enqueue_phy_work failed - releasing SRS.IND nvIPC buffers",
               slot.sfn_, slot.slot_);

    for (std::size_t cell_store_idx = 0; cell_store_idx < n_cells; ++cell_store_idx)
    {
        const int cell_index = srs_params.cell_index_list[cell_store_idx];
        const int max_srs_ind_index = srs_params.num_srs_ind_indexes[cell_store_idx];
        if (cell_index < 0 || max_srs_ind_index < 0
            || max_srs_ind_index >= slot_command_api::MAX_SRS_IND_PER_SLOT)
        {
            NVLOGW_FMT(TAG,
                       "SFN {}.{}: skipping invalid SRS IND cleanup row cell_store_idx={} "
                       "cell_index={} max_srs_ind_index={}",
                       slot.sfn_, slot.slot_, cell_store_idx, cell_index, max_srs_ind_index);
            continue;
        }

        for (int srs_ind_index = 0; srs_ind_index <= max_srs_ind_index; ++srs_ind_index)
        {
            auto& desc = srs_params.srs_indications[cell_store_idx][srs_ind_index];
            if (desc.data_buf == nullptr)
            {
                continue;
            }

            NVLOGD_FMT(TAG,
                       "{}.{}: releasing SRS IND cell_index {} srs_indications[{}][{}] = "
                       "msg_id {}, cell_id {} msg_len {} data_len {} data_pool {} msg_buf {} data_buf {}",
                       slot.sfn_, slot.slot_, cell_index, cell_store_idx, srs_ind_index,
                       desc.msg_id, desc.cell_id, desc.msg_len, desc.data_len, desc.data_pool,
                       desc.msg_buf, desc.data_buf);
            nv::phy_mac_msg_desc msg_desc(desc);
            try
            {
                transport(cell_index).tx_release(msg_desc);
                desc = {};
            }
            catch (const std::runtime_error& ex)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "SFN {}.{}: tx_release threw during SRS.IND cleanup "
                           "cell_index={} idx={} - leaking buffer, continuing: {}",
                           slot.sfn_, slot.slot_, cell_index, srs_ind_index, ex.what());
            }
        }
        srs_params.num_srs_ind_indexes[cell_store_idx] = 0;
    }
}

/**
 * @brief Stash EOM-derived per-slot fields into @c slot_telemetry_[ring_idx].
 *
 * Ingestion-time fields are populated by @c stamp_ring_telemetry_on_ingest
 * and the SLOT.RESP per-ring stamp, so they are not touched here.
 */
void PHY_module::stash_slot_telemetry_for_worker_publish(uint32_t   ring_idx,
                                                         sfn_slot_t ss_curr,
                                                         uint32_t   slot_cmd_idx)
{
    // Locking: tick_lock only synchronises current_tick_ <-> ss_tick. The
    // per-slot-mod-10 arrays (current_tick_list_, l1_slot_ind_tick_) are
    // written by the tick thread without holding tick_lock, so reading them
    // here lock-free matches process_phy_commands(). Only get_fapi_latency()
    // touches state mutated under tick_lock, hence the narrow critical
    // section below. Msg-thread only — a future move off this thread must
    // reconsider visibility of the singleton scalars read below.
    auto& snap = slot_telemetry_[ring_idx];
    snap.ss               = ss_curr;
    snap.l1_slot_ind_tick = l1_slot_ind_tick_[ss_curr.u16.slot % 10];
    snap.current_tick     = current_tick_list_[ss_curr.u16.slot % 10];
    snap.l2a_end_tick     = std::chrono::nanoseconds{};
    // Ingestion-time fields populated by stamp_ring_telemetry_on_ingest + SLOT.RESP.

    // tick_original mirrors process_phy_commands (nv_phy_module.cpp:1188): use
    // the singleton current_tick_ under tick_lock, NOT snap.current_tick (the
    // per-slot current_tick_list_[slot%10] entry). The two normally agree, but
    // diverge if the tick thread has advanced past the slot being processed —
    // in which case current_tick_ holds the next slot's tick while the list
    // entry still holds this slot's. process_phy_commands uses the singleton
    // for tick_original and the per-slot list for the downstream
    // L2A.PROCESSING_TIMES timestamps; we keep that exact split.
    std::chrono::nanoseconds curr_tick{};
    {
        const std::lock_guard lock(tick_lock);
        snap.slot_interval = get_fapi_latency(ss_curr);
        curr_tick          = current_tick_;
    }

    // mu_highest_ is set once at config init and never mutated thereafter,
    // so it is read lock-free.
    assert(slot_cmd_idx < slot_command_array.size());
    auto& slot_cmd = slot_command_array[slot_cmd_idx];
    slot_cmd.tick_original =
        curr_tick - std::chrono::nanoseconds(
                        static_cast<int64_t>(snap.slot_interval * mu_to_ns(tick_updater_.mu_highest_)));
}

/**
 * @brief Submit the completed slot command (DL + UL) to the PHY driver.
 *
 * Called at last-task drain (from @c on_slot_channel_task_complete, or inline in
 * @c enqueue_channel_tasks when all tasks drained before the sentinel-drop). Uses the
 * slot command index captured at EOM in @c task_pool_.dl_aggr_task_args[ring_idx].slot_cmd_idx
 * rather than @c current_slot_cmd_index, which has already been advanced by the time this runs.
 *
 * fapi-to-cplane-direct is mandatory, so @c l1_enqueue_phy_work() is always called with
 * the direct skip mask (SKIP_DL_FHCB | SKIP_DL_CPLANE | SKIP_UL_CPLANE |
 * SKIP_DL_GPU_COMM_PREPARE): the DL/UL C-plane was already published by the batch tasks,
 * so this enqueues only U-plane work. Handles the empty-slot (channel_array_size == 0) case.
 *
 * @param ring_idx  Ring slot index used to look up slot_cmd_idx and slot_u32.
 */
void PHY_module::publish_slot_command(uint32_t ring_idx)
{
    const auto& ctx = task_pool_.dl_aggr_task_args[ring_idx];
    assert(ctx.slot_cmd_idx < slot_command_array.size());
    auto& slot_cmd  = slot_command_array[ctx.slot_cmd_idx];
    auto& group_cmd = slot_cmd.cell_groups;

    // Per-slot telemetry: read from the EOM-time snapshot populated by the
    // msg thread inside enqueue_channel_tasks. tick_original was already
    // stamped there, so this path no longer needs tick_lock.
    // Telemetry is computed/emitted up-front so the channel_array_size == 0
    // skip path still produces one slot_latency + one L2A.PROCESSING_TIMES
    // line per publish_slot_command invocation (stub-era channel-aggr tasks
    // are no-ops and leave channel_array_size at 0, but the L2A latency for
    // the slot is still meaningful and worth reporting).
    //
    // Value-copy slot_telemetry_[ring_idx] for body-local consistency. The msg-thread
    // writers of this element (EOM stash + rebind reset in store_slot_message) cannot
    // race this read: the completing task holds its +1 on tasks_in_flight[ring_idx]
    // until AFTER publish_slot_command returns (the ordering invariant in
    // on_slot_channel_aggr_complete and the inline-drained path), and store_slot_message
    // only rebinds/resets a ring whose tasks_in_flight == 0 - it drops the incoming
    // message otherwise. So no rebind can fire while this worker reads the snapshot.
    nv::SlotTelemetrySnapshot snap   = slot_telemetry_[ring_idx];
    const std::size_t cells_size = slot_cmd.cells.size();
    const uint64_t    mu_ns      = mu_to_ns(tick_updater_.mu_highest_);

    const auto l2a_end = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    snap.l2a_end_tick = l2a_end;
    int64_t latency =
        (l2a_end - snap.current_tick).count() + static_cast<int64_t>(snap.slot_interval * mu_ns);
    {
        // stat_log_add() does unlocked RMW on min/max/sum/carry/counter
        // (cuPHY/nvlog/src/stat_log.c:252). Multiple cuphydriver workers
        // can land here concurrently for different ring slots, so the
        // mutex is required even though contention is rare.
        const std::lock_guard lock(slot_latency_mu_);
        slot_latency->add(slot_latency, latency);
    }

    // Tick-advance / per-slot duration components shared by the skipped and
    // full branches. SFN/slot are taken from snap.ss (populated by the msg
    // thread at EOM in enqueue_channel_tasks), which is identical to
    // slot_cmd.cell_groups.slot.slot_3gpp.* once written by the same code
    // path but is valid even when channel_array_size == 0.
    const std::chrono::nanoseconds current_t0_timestamp(
        sfn_to_tai(snap.ss.u16.sfn, snap.ss.u16.slot,
                   snap.current_tick.count() + AppConfig::getInstance().getTaiOffset(),
                   gps_alpha_, gps_beta_, tick_updater_.mu_highest_)
        - AppConfig::getInstance().getTaiOffset());
    const auto tick_advance      = current_t0_timestamp - snap.current_tick;
    const auto l1_slot_ind_delay = snap.l1_slot_ind_tick   - snap.current_tick;
    const auto l2_estimate       = snap.last_fapi_msg_tick - snap.l1_slot_ind_tick;
    const auto fapi_proc_dur     = snap.l2a_end_tick       - snap.l2a_start_tick;

    if (group_cmd.channel_array_size == 0)
    {
        // No channels built — skip driver enqueue, but still emit
        // slot_latency + L2A.PROCESSING_TIMES with cells=cmd_size=0,
        // L1 Enqueue Dur=0, and skipped=1 so the per-slot worker telemetry
        // remains 1:1 with publish_slot_command invocations. This is the
        // common path during the current stub era of
        // process_aggr_{pdsch,pusch,...}_channel(); once channel-aggr
        // workers populate cell_groups.channel_array, this branch becomes
        // rare. Reporting cells=0 (instead of slot_cmd.cells.size()) avoids
        // making a no-channel skip look like a full-cell command in the log.
        const std::size_t skipped_cells = 0;
        NVLOGI_FMT(TAG, "SFN {}.{} {} cells={} cmd_size={} slot_latency={} skipped=1",
                   snap.ss.u16.sfn, snap.ss.u16.slot, __func__,
                   skipped_cells, skipped_cells, latency);

        NVLOGI_FMT(TAG_PROCESSING_TIMES,
            "SFN {}.{} {}: cells={} cmd_size={} Tick Advance={}ns L1 Slot Ind Delay={}ns L2 Estimate={}ns FAPI Proc Dur={}ns L1 Enqueue Dur={}ns l1_slot_ind_tick={} l2a_start_time={} l2a_end_time={} last_fapi_msg_tick={} l1_enqueue_complete_time={} UL={} DL={} CSIRS={} skipped=1",
            snap.ss.u16.sfn, snap.ss.u16.slot, __func__,
            skipped_cells, skipped_cells,
            tick_advance.count(),
            l1_slot_ind_delay.count(),
            l2_estimate.count(),
            fapi_proc_dur.count(),
            int64_t{0},                              // L1 Enqueue Dur — nothing enqueued
            snap.l1_slot_ind_tick.count(), snap.l2a_start_tick.count(), snap.l2a_end_tick.count(),
            snap.last_fapi_msg_tick.count(), snap.l2a_end_tick.count(),
            snap.is_ul_slot ? 1 : 0, snap.is_dl_slot ? 1 : 0, snap.is_csirs_slot ? 1 : 0);

        NVLOGD_FMT(TAG,
                   "task_framework: publish_slot_command skipped (channel_array_size=0) "
                   "slot=0x{:08X} ring_idx={} slot_cmd_idx={} (stub-era channel-aggr no-op)",
                   ctx.slot_u32, ctx.ring_idx, ctx.slot_cmd_idx);
        for (auto& phy : phy_refs_)
        {
            phy.get().reset_slot(false /*partial_cmd*/);
        }
        return;
    }
    // Full publish path: emit the slot_latency INFO line with real cell counts.
    NVLOGI_FMT(TAG, "SFN {}.{} {} cells={} cmd_size={} slot_latency={}",
               snap.ss.u16.sfn, snap.ss.u16.slot, __func__,
               cells_size, cells_size, latency);

    // Bracket l1_enqueue_phy_work with start/end timestamps so the
    // l1_enqueue_phy_work and L2A.PROCESSING_TIMES diagnostic lines below
    // can report the same fields as process_phy_commands.
    const auto start_process_command_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch());

    // Re-anchor l2a_end_tick to the enqueue boundary so l1_enqueue_duration and
    // fapi_proc_dur exclude the telemetry-compute overhead above (stat_log add,
    // sfn_to_tai derivations, slot_latency emit), matching process_phy_commands'
    // anchor in nv_phy_module.cpp. The skipped branch above keeps the function-entry
    // stamp - it has no l1_enqueue_phy_work, so the anchor difference does not apply.
    snap.l2a_end_tick = start_process_command_time;
    const auto fapi_proc_dur_enqueue = snap.l2a_end_tick - snap.l2a_start_tick;

    // The unified EOM path always publishes DL/UL C-plane via the batch tasks fired in
    // enqueue_channel_tasks, so the U-plane submit must always skip C-plane (plus the
    // legacy FHCB / GPU-prepare the direct path elides). fapi-to-cplane-direct is the
    // only supported mode, so this mask is unconditional — a SKIP_NONE fallback would
    // re-enqueue and double-publish the C-plane.
    const auto skip = EnqueueSkipMask::SKIP_DL_FHCB
                      | EnqueueSkipMask::SKIP_DL_CPLANE
                      | EnqueueSkipMask::SKIP_UL_CPLANE
                      | EnqueueSkipMask::SKIP_DL_GPU_COMM_PREPARE;
    void* slot_map_dl = nullptr;
    void* slot_map_ul = nullptr;
    for (uint32_t cid = 0; cid < MAX_CELLS_PER_SLOT; ++cid)
    {
        const uint64_t cell_bit = uint64_t{1} << cid;
        const auto idx = ring_idx * MAX_CELLS_PER_SLOT + cid;
        if (slot_map_dl == nullptr && (task_pool_.dlc_accumulated_bitmap[ring_idx] & cell_bit) != 0)
        {
            slot_map_dl = task_pool_.dlc_task_args[idx].slot_map_dl;
        }
        if (slot_map_ul == nullptr && (task_pool_.ulc_accumulated_bitmap[ring_idx] & cell_bit) != 0)
        {
            slot_map_ul = task_pool_.ulc_task_args[idx].slot_map_ul;
        }
        if (slot_map_dl != nullptr && slot_map_ul != nullptr)
        {
            break;
        }
    }
    const int ret = PHYDriverProxy::getInstance().l1_enqueue_phy_work(slot_cmd, skip, slot_map_dl, slot_map_ul);

    const auto end_process_command_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch());
    const auto diff = end_process_command_time - start_process_command_time;

    NVLOGI_FMT(TAG,
        "SFN {}.{} {}: l1_enqueue_phy_work after status = {} slot cmd size ={}  phy_refs : size = {} l1_enqueue_phy_work start: {} l1_enqueue_phy_work duration: {} ns UL {} DL {} direct={} skip_mask=0x{:X}",
        snap.ss.u16.sfn, snap.ss.u16.slot, __func__, ret,
        cells_size, phy_refs_.size(),
        start_process_command_time.count(), diff.count(),
        snap.is_ul_slot ? 1 : 0, snap.is_dl_slot ? 1 : 0,
        1, static_cast<uint32_t>(skip));

    // L2A.PROCESSING_TIMES diagnostic — mirrors nv_phy_module.cpp:1234-1254
    // but reads per-slot fields from snap (no cross-thread races).
    // tick_advance / l1_slot_ind_delay / l2_estimate / fapi_proc_dur were
    // computed once above and are reused here so the skipped branch reports
    // the same sfn_to_tai()-derived values as this branch.
    const auto l1_enqueue_duration = end_process_command_time - snap.l2a_end_tick;

    NVLOGI_FMT(TAG_PROCESSING_TIMES,
        "SFN {}.{} {}: cells={} cmd_size={} Tick Advance={}ns L1 Slot Ind Delay={}ns L2 Estimate={}ns FAPI Proc Dur={}ns L1 Enqueue Dur={}ns l1_slot_ind_tick={} l2a_start_time={} l2a_end_time={} last_fapi_msg_tick={} l1_enqueue_complete_time={} UL={} DL={} CSIRS={}",
        snap.ss.u16.sfn, snap.ss.u16.slot, __func__,
        cells_size, cells_size,
        tick_advance.count(),
        l1_slot_ind_delay.count(),
        l2_estimate.count(),
        fapi_proc_dur_enqueue.count(),
        l1_enqueue_duration.count(),
        snap.l1_slot_ind_tick.count(), snap.l2a_start_tick.count(), snap.l2a_end_tick.count(),
        snap.last_fapi_msg_tick.count(), end_process_command_time.count(),
        snap.is_ul_slot ? 1 : 0, snap.is_dl_slot ? 1 : 0, snap.is_csirs_slot ? 1 : 0);

    if (ret != 0)
    {
        NVLOGW_FMT(TAG,
                   "publish_slot_command: l1_enqueue_phy_work failed "
                   "ret={} slot=0x{:08X} - resetting PHY instances",
                   ret,
                   ctx.slot_u32);
        if(auto* srs_params = group_cmd.srs.get())
        {
            release_srs_indication_buffers(*srs_params, group_cmd.slot.slot_3gpp);
        }
        for(auto& phy : phy_refs_)
        {
            phy.get().reset_slot(true /*partial_cmd*/);
        }
        // Failure branch only: cuphydriver's tx_data_release_fn will NOT fire
        // (the worker task was never enqueued). Clear the ring slot here so
        // the subsequent release_stored_slot_messages() observes
        // is_deferred == false and releases the TX_DATA lane immediately.
        reset_txdata_h2d_state_for_ring(ring_idx);
    }
    else
    {
        NVLOGD_FMT(TAG,
                   "task_framework: publish_slot_command success slot=0x{:08X} slot_cmd_idx={} channel_array_size={}",
                   ctx.slot_u32,
                   ctx.slot_cmd_idx,
                   group_cmd.channel_array_size);
        // Success branch: leave deferred=true. The cuphydriver worker still
        // has DMA in flight reading from the TX_DATA IPC buffers; the
        // tx_data_release_fn callback (release_deferred_tx_data) is the only
        // safe point to release the TX_DATA lane and reset the ring slot.
        // Clearing deferred here would let the immediately-following
        // release_stored_slot_messages() free the IPC buffers prematurely
        // (use-after-free on the worker's DMA source).
    }

    // TODO(GT-11913): handler-specific cleanup deferred to task handler implementation phase:
    //       DL TB buffer lifecycle handoff, UL TB buffer lifecycle, per-cell reset_slot().
}

/**
 * @brief Perform housekeeping for a slot with no channel tasks (DL or UL).
 *
 * Called when both n_dl_tasks and n_ul_tasks are zero at EOM.
 * Resets each PHY instance's per-slot state without submitting a driver command.
 *
 * @param slot_u32  Packed SFN/slot (unused; retained for future logging).
 */
void PHY_module::submit_slot_command_empty(uint32_t /*slot_u32*/)
{
    for(auto& phy : phy_refs_)
    {
        phy.get().reset_slot(false /*partial_cmd*/);
    }
}

// ---------------------------------------------------------------------------
// stage_tx_data_h2d / launch_tx_data_h2d — TX_DATA H2D DMA stubs.
// Moved from nv_phy_dl_channels.cpp.

// ---------------------------------------------------------------------------

void PHY_module::stage_tx_data_h2d(uint32_t                cell_id,
                                   const phy_mac_msg_desc& smsg,
                                   uint32_t                ring_idx,
                                   sfn_slot_t              ss_msg)
{
    if(ring_idx >= SLOT_STORAGE_DEPTH)
    {
        return;
    }
    if(smsg.cell_id < 0 || smsg.data_buf == nullptr)
    {
        NVLOGW_FMT(TAG,
                   "stage_tx_data_h2d: invalid TX_DATA cell_id={} data_buf={}",
                   smsg.cell_id,
                   static_cast<void*>(smsg.data_buf));
        return;
    }
    if(cell_id >= MAX_CELLS_PER_SLOT || cell_id >= PHY_instances().size())
    {
        NVLOGW_FMT(TAG,
                   "stage_tx_data_h2d: cell_id={} out of range [0,{})",
                   cell_id,
                   std::min(static_cast<std::size_t>(MAX_CELLS_PER_SLOT),
                            PHY_instances().size()));
        return;
    }
    const uint16_t phy_cell_id = PHY_instances()[cell_id].get().get_phy_cell_id();
    const uint8_t  tb_slot     = tx_data_h2d::tb_gpu_buffer_index(ss_msg.u32);

    uint8_t*  gpu_ptr = nullptr;
    const int st      = PHYDriverProxy::getInstance().l1_stage_tb_h2d(
        phy_cell_id,
        static_cast<const uint8_t*>(smsg.data_buf),
        smsg.data_len,
        tb_slot,
        &gpu_ptr);

    if(st == static_cast<int>(CUPHY_STATUS_SUCCESS) && gpu_ptr != nullptr)
    {
        tx_data_ring_.stage(ring_idx, cell_id, gpu_ptr, smsg.msg_buf);
        NVLOGD_FMT(TAG,
                   "stage_tx_data_h2d: OK cell_id={} pci={} ring_idx={} slot=0x{:08X} "
                   "gpu_ptr={} cell_count={} tb_slot={}",
                   cell_id,
                   phy_cell_id,
                   ring_idx,
                   ss_msg.u32,
                   static_cast<void*>(gpu_ptr),
                   tx_data_ring_.staged_count(ring_idx),
                   tb_slot);
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "stage_tx_data_h2d: l1_stage_tb_h2d failed ring_idx={} cell_id={} pci={} st={} - "
                                                "Phase 2 count mismatch will skip PDSCH for this slot",
                   ring_idx,
                   cell_id,
                   phy_cell_id,
                   static_cast<int>(st));
    }
}

void PHY_module::launch_tx_data_h2d(sfn_slot_t ss_curr, uint32_t ring_idx)
{
    auto&   slot_store           = slot_message_storage(ss_curr.u32);
    uint8_t expected_pdsch_cells = 0;
    for(uint16_t i = 0; i < slot_store.dl_tti_count(); ++i)
    {
        if(slot_store.dl_tti_has_pdsch_pdu(i))
        {
            ++expected_pdsch_cells;
        }
    }

    // Save staged count BEFORE arm() — arm() calls reset() on failure, zeroing cell_count.
    const uint8_t staged = tx_data_ring_.staged_count(ring_idx);

    if(expected_pdsch_cells == 0U)
    {
        if(staged > 0U)
        {
            NVLOGW_FMT(TAG,
                       "launch_tx_data_h2d: no DL_TTI PDSCH but "
                       "TX_DATA staged cell_count={} ring_idx={} slot=0x{:08X}",
                       staged,
                       ring_idx,
                       ss_curr.u32);
            reset_batched_memcpy_unless_sibling_staged(ring_idx);
        }
        tx_data_ring_.reset(ring_idx);
        return;
    }

    if(!tx_data_ring_.arm(ring_idx, expected_pdsch_cells))
    {
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "launch_tx_data_h2d: TX_DATA vs DL_TTI PDSCH "
                   "count mismatch expected={} staged={} ring_idx={} slot=0x{:08X} - "
                   "clearing split-phase state; PDSCH will be skipped (pTbInput null)",
                   expected_pdsch_cells,
                   staged,
                   ring_idx,
                   ss_curr.u32);
        if(staged > 0U)
        {
            reset_batched_memcpy_unless_sibling_staged(ring_idx);
        }
        // arm() already reset the ring on failure.
        return;
    }

    NVLOGD_FMT(TAG,
               "launch_tx_data_h2d: Phase2 PDSCH-vs-TX_DATA verified OK - launching "
               "expected_pdsch={} staged_tx_data={} ring_idx={} slot=0x{:08X}",
               expected_pdsch_cells,
               staged,
               ring_idx,
               ss_curr.u32);

    const uint16_t slot_in_frame = tx_data_h2d::slot_in_frame_from_slot_u32(ss_curr.u32);
    const int      launch_st     = PHYDriverProxy::getInstance().l1_launch_tb_h2d(slot_in_frame);
    if(launch_st != static_cast<int>(CUPHY_STATUS_SUCCESS))
    {
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "launch_tx_data_h2d: l1_launch_tb_h2d returned {} "
                   "ring_idx={} slot=0x{:08X} - clearing split-phase state; PDSCH will be skipped",
                   launch_st,
                   ring_idx,
                   ss_curr.u32);
        tx_data_ring_.reset(ring_idx);
    }
    else
    {
        NVLOGD_FMT(TAG,
                   "launch_tx_data_h2d: Phase2 H2D DMA launched OK "
                   "slot_in_frame={} ring_idx={} slot=0x{:08X} expected_cells={}",
                   slot_in_frame,
                   ring_idx,
                   ss_curr.u32,
                   expected_pdsch_cells);
        // Arm TX_DATA release ownership ONLY here, on the H2D-launch success
        // path: the cuphydriver H2D-completion callback (release_deferred_tx_data)
        // is now guaranteed to fire and owns the release, so the inline
        // release_stored_slot_messages() path must skip the TX_DATA lane.
        // `deferred` is false by default and is cleared by the callback after each
        // release, so the early-return / failure exits above deliberately leave it
        // false (release handled inline) — no explicit clear is needed.
        //
        // Replay-present direct mode arms the same token in process_phy_commands()
        // instead (see nv_phy_module.cpp); the two sites are mutually exclusive at
        // runtime. No-replay builds must arm only here — never in enqueue_channel_tasks.
        tx_data_ring_.set_deferred(ring_idx, true);
    }
    // Note: a release fence here would be a no-op — there is no
    // subsequent atomic store on this thread to pair with, and the
    // downstream GPU consumer synchronizes via cudaStreamWaitEvent
    // (CUDA-event domain), not a C++ acquire load. Ordering for the
    // CPU→GPU handoff is established by the cudaEventRecord issued
    // inside l1_launch_tb_h2d.
}

// ---------------------------------------------------------------------------
// Split-phase TX_DATA H2D helpers
// ---------------------------------------------------------------------------

void PHY_module::reset_txdata_h2d_state_for_ring(std::size_t ring_idx)
{
    if(ring_idx >= SLOT_STORAGE_DEPTH)
    {
        return;
    }
    // Abandon path (enqueue failure / ring collision): the H2D-completion
    // callback will not fire for this slot, so hand release back to the inline
    // message-thread path by clearing deferred. reset() itself is staging-only
    // and no longer clears deferred, so this must be explicit here to avoid the
    // TX_DATA IPC buffer leak on enqueue failure.
    tx_data_ring_.set_deferred(static_cast<uint32_t>(ring_idx), false);
    reset_batched_memcpy_unless_sibling_staged(static_cast<uint32_t>(ring_idx));
    tx_data_ring_.reset(static_cast<uint32_t>(ring_idx));
}

void PHY_module::reset_batched_memcpy_unless_sibling_staged(uint32_t ring_idx)
{
    // The driver's batched-memcpy accumulator is a single shared (not
    // slot-keyed) list flushed by l1_launch_tb_h2d. Resetting it while a
    // SIBLING ring slot holds staged-but-unlaunched TX_DATA would cancel the
    // sibling's pending copies; its launch would still record the complete
    // event and PDSCH would transmit stale GPU TB data. Reset only when no
    // sibling has staged entries. Otherwise leave the accumulator: copies
    // land in per-slot GPU buffers, so this ring's orphaned entries are
    // flushed harmlessly by the sibling's launch.
    for(uint32_t r = 0; r < SLOT_STORAGE_DEPTH; ++r)
    {
        if(r != ring_idx && tx_data_ring_.staged_count(r) > 0U)
        {
            NVLOGW_FMT(TAG,
                       "reset_batched_memcpy_unless_sibling_staged: skipping global batched-memcpy "
                       "reset for ring_idx={} - sibling ring_idx={} has staged_count={}",
                       ring_idx, r, tx_data_ring_.staged_count(r));
            return;
        }
    }
    PHYDriverProxy::getInstance().l1_resetBatchedMemcpyBatches();
}

bool PHY_module::use_split_phase_txdata_h2d() noexcept
{
#ifdef ENABLE_FAPI_STORE_REPLAY
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// TX_DATA deferred-release ownership contract (exactly-once per ring slot):
//
// Arm sites (build-aware, mutually exclusive per slot lifecycle):
//   - Replay ON + legacy split-phase: launch_tx_data_h2d() after l1_launch_tb_h2d success.
//   - Replay ON + direct mode: process_phy_commands() after l1_enqueue_phy_work success.
//   - Replay OFF: launch_tx_data_h2d() only — never process_phy_commands or enqueue_channel_tasks.
//
// Callback: release_deferred_tx_data() via tx_data_release_fn — at most once per arm.
// Disarm after arm: release_deferred_tx_data() clears deferred after release_lane(TX_DATA).
// Abandon without arm: reset_txdata_h2d_state_for_ring() clears deferred; inline releases.
//
// TxDataRingBuffer::reset() is staging-only and must not touch deferred.
// ---------------------------------------------------------------------------

void PHY_module::s_tx_data_release(void* ctx, uint16_t sfn, uint8_t slot)
{
    auto* self = static_cast<PHY_module*>(ctx);
    self->release_deferred_tx_data(sfn, slot);
}

void PHY_module::release_deferred_tx_data(uint16_t sfn, uint8_t slot)
{
    sfn_slot_t ss;
    ss.u16.sfn              = sfn;
    ss.u16.slot             = slot;
    const uint32_t ring_idx = ring_idx_from_slot(ss.u32);

    // TX_DATA release ownership (callback path). Any callback with deferred==false
    // is a contract violation (duplicate callback, missing arm, or spurious path).
    // Logged at ERROR, not FATAL: the path below is idempotent (release_lane
    // atomically no-ops on an already-claimed lane, then deferred/reset are
    // cleared), so the violation is surfaced loudly without aborting L1.
    if(!tx_data_ring_.is_deferred(ring_idx))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "release_deferred_tx_data: callback fired but deferred=false "
                   "(duplicate callback or missing arm) ring_idx={} slot=0x{:08X} "
                   "tx_data_count={}",
                   ring_idx,
                   ss.u32,
                   slot_message_storage_[ring_idx].tx_data_count());
    }

    auto& store = slot_message_storage_[ring_idx];
    NVLOGD_FMT(TAG,
               "task_framework: release_deferred_tx_data callback from cuphydriver - "
               "sfn={} slot={} ring_idx={} tx_data_count={}",
               sfn,
               slot,
               ring_idx,
               store.tx_data_count());
    store.release_lane(FapiSlotMessageStorage::MsgType::TX_DATA, transport_wrapper());
    // Hand ownership back to the inline path: clear `deferred` (release store)
    // AFTER the atomic release_lane above has claimed and zeroed the lane count.
    // A racing/later inline release for this slot that observes deferred==false
    // then calls release_lane and finds count==0 (visible via this release store
    // paired with is_deferred()'s acquire load), so it frees nothing — no
    // double-free. Clearing here (rather than leaving it true) also means the
    // next slot at this ring index that finalizes via an abandon path sees
    // deferred==false and releases its own TX_DATA inline instead of leaking.
    tx_data_ring_.set_deferred(ring_idx, false);
    // Clear the H2D staging fields for the next slot at this ring index.
    tx_data_ring_.reset(ring_idx);
}

// ---------------------------------------------------------------------------
// Non-slot message helpers and slot payload helpers.
// Moved from nv_phy_module.cpp; all callers are in process_fapi_messages().

// ---------------------------------------------------------------------------

/**
 * @brief Dispatch a message that bypasses slot storage entirely.
 *
 * Called for messages identified by is_immediate_non_slot_msg().  Adjusts
 * tti_event_count in non-per-slot sync mode, validates cell_id, invokes
 * on_msg(), and releases the NVIPC buffer when the PHY does not retain it.
 *
 * @param msg Message descriptor received from transport.
 */
void PHY_module::process_immediate_non_slot_msg(phy_mac_msg_desc& msg)
{
    // Messages selected by is_immediate_non_slot_msg() are always
    // processed immediately and never queued in deferred storage.
    if(ipc_sync_mode != SYNC_MODE_PER_SLOT)
    {
        tti_event_count--;
    }
    if(msg.cell_id < 0)
    {
        NVLOGW_FMT(TAG, "process_immediate_non_slot_msg: invalid cell_id={} msg_id=0x{:02X}", msg.cell_id, msg.msg_id);
        transport_wrapper().rx_release(msg);
        return;
    }
    const auto cell_id = static_cast<std::size_t>(msg.cell_id);
    if(cell_id >= phy_refs_.size())
    {
        NVLOGW_FMT(TAG, "process_immediate_non_slot_msg: invalid cell_id={} msg_id=0x{:02X}", msg.cell_id, msg.msg_id);
        transport_wrapper().rx_release(msg);
        return;
    }
    if(phy_refs_[cell_id].get().on_msg(msg))
    {
        transport_wrapper().rx_release(msg);
    }
}

/**
 * @brief Store a non-slot message via the per-cell storage path.
 *
 * Validates cell_id, delegates storage to the PHY instance, and releases the
 * NVIPC buffer on failure (RAII guard). Returns true only when the message is
 * successfully stored.
 *
 * @return true if stored, false otherwise.
 */
bool PHY_module::store_non_slot_message(phy_mac_msg_desc& smsg, sfn_slot_t ss_msg)
{
    auto release_guard = make_scope_exit([this, &smsg]() {
        transport_wrapper().rx_release(smsg);
    });

    if(ipc_sync_mode != SYNC_MODE_PER_SLOT)
    {
        tti_event_count--;
    }

    if(smsg.cell_id < 0)
    {
        NVLOGW_FMT(TAG, "store_non_slot_message: invalid cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
        return false;
    }
    const auto cell_id = static_cast<std::size_t>(smsg.cell_id);
    if(cell_id >= phy_refs_.size())
    {
        NVLOGW_FMT(TAG, "store_non_slot_message: invalid cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
        return false;
    }

    const bool stored = phy_refs_[cell_id].get().on_msg_to_store(smsg, ss_msg.u32);
    if(!stored)
    {
        NVLOGW_FMT(TAG, "store_non_slot_message: non-slot message not stored cell_id={} msg_id=0x{:02X}", smsg.cell_id, smsg.msg_id);
        return false;
    }

    release_guard.release();
    return true;
}

void PHY_module::process_pending_non_slot_messages(uint16_t cell_id)
{
    if(const auto sz = phy_refs_.size(); cell_id >= sz)
    {
        NVLOGW_FMT(TAG,
                   "process_pending_non_slot_messages: cell_id={} out of range (size={})",
                   cell_id,
                   sz);
        return;
    }
    auto& storage     = non_slot_message_storage(cell_id);
    auto  process_one = [this, cell_id](phy_mac_msg_desc* stored_msg) {
        if(stored_msg == nullptr)
        {
            return;
        }
        if(phy_refs_[cell_id].get().on_msg(*stored_msg))
        {
            transport_wrapper().rx_release(*stored_msg);
        }
    };

    process_one(storage.config_request_mut());
    process_one(storage.start_request_mut());
    process_one(storage.stop_request_mut());

    // Messages handled in this pass should not be processed again.
    storage.clear();
}

uint32_t PHY_module::ring_idx_from_slot(uint32_t slot_u32) const noexcept
{
    return slot_index_from_u32(slot_u32, nv::mu_to_slot_in_sf(tick_updater_.mu_highest_)) % SLOT_STORAGE_DEPTH;
}

FapiSlotMessageStorage& PHY_module::slot_message_storage(uint32_t slot_u32)
{
    return slot_message_storage_[ring_idx_from_slot(slot_u32)];
}

bool PHY_module::store_message(const phy_mac_msg_desc& msg, uint32_t slot_u32)
{
    if(slot_u32 == SFN_SLOT_INVALID)
    {
        if(msg.cell_id < 0)
        {
            NVLOGE_FMT(TAG,
                       AERIAL_L2ADAPTER_EVENT,
                       "store_message: invalid cell_id={} max_cells={}",
                       msg.cell_id,
                       nonslot_dispatch_state_.message_storage.size());
            return false;
        }
        const auto cell_id = static_cast<std::size_t>(msg.cell_id);
        if (const auto sz = nonslot_dispatch_state_.message_storage.size(); cell_id >= sz)
        {
            NVLOGE_FMT(TAG,
                       AERIAL_L2ADAPTER_EVENT,
                       "store_message: invalid cell_id={} max_cells={}",
                       msg.cell_id,
                       sz);
            return false;
        }
        return non_slot_message_storage(cell_id)
            .store_message(msg, transport_wrapper());
    }

    const StoreResult result = slot_message_storage(slot_u32).store_message(msg);
    return result == StoreResult::Stored;
}

} // namespace nv
