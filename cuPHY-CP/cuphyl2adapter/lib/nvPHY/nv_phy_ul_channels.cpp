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
 * @file nv_phy_ul_channels.cpp
 * @brief UL channel aggregation handlers for PHY_module.
 *
 * Implements PHY_module::process_aggr_*_channel for the UL path (PUSCH,
 * PRACH, PUCCH, SRS, UL-BFW). Each function is invoked by a worker thread
 * via the UlChannelDispatch table wired up in nv_phy_module.hpp and
 * populated at EOM in enqueue_channel_tasks().
 *
 * Anonymous-namespace helpers encapsulate common FAPI iteration patterns
 * so each channel handler contains only its channel-specific logic.
 * @see nv_phy_dl_channels.cpp for the symmetric DL path.
 */

#include "nv_phy_module.hpp"
#include "nv_fapi_pdu_utils.hpp"
#include "nv_ul_aggr_tasks.hpp"
#include "nv_bfw_debug_cleanup.hpp"
#include "scf_5g_fapi_ul_slot_processor.hpp"
#include "scf_5g_fapi_message_context.hpp"
#include "scf_5g_slot_commands_common.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

static_assert(std::is_standard_layout_v<scf_fapi_ul_bfw_cvi_request_t>,
              "scf_fapi_ul_bfw_cvi_request_t must be standard-layout for reinterpret_cast from raw FAPI buffer");
static_assert(std::is_standard_layout_v<scf_fapi_ul_bfw_group_config_t>,
              "scf_fapi_ul_bfw_group_config_t must be standard-layout for reinterpret_cast from raw FAPI buffer");

namespace {
/// Minimum `phy_mac_msg_desc::msg_len` for a UL BFW CVI request after `scf_fapi_header_t`
/// (fixed `scf_fapi_ul_bfw_cvi_request_t` prefix; `config_pdu[]` is not counted in sizeof).
constexpr std::size_t kUlBfwCviMsgMinBytes =
    sizeof(scf_fapi_header_t) + sizeof(scf_fapi_ul_bfw_cvi_request_t);

/// Internal enum (anonymous namespace) identifying a UL ordering-scratch channel.
enum class UlOrderScratchChannel : uint32_t
{
    Prach = 0, //!< PRACH ordering scratch
    Pusch = 1, //!< PUSCH ordering scratch
    Pucch = 2, //!< PUCCH ordering scratch
    Srs   = 3, //!< SRS ordering scratch
};

/// Internal POD (anonymous namespace) referencing one ring's UL ordering-scratch state.
struct UlOrderScratchContext final
{
    nv::PHY_module*       owner{};    //!< Non-owning; must outlive this context.
    uint32_t              ring_idx{}; //!< Index into the ordering-scratch ring.
    UlOrderScratchChannel channel{};  //!< Associated UL ordering-scratch channel.
};

/// Order-kernel scratch resolver: maps (@p ctx, @p cell_idx) to the per-ring,
/// per-channel UL ordering sym_prb_info scratch slot. Used as a callback pointer.
/// @return Scratch slot_info for the cell, or nullptr if @p ctx or its owner is null.
[[nodiscard]] slot_command_api::slot_info_t* resolve_ul_order_scratch(void* ctx, uint32_t cell_idx) noexcept
{
    auto* scratch_ctx = static_cast<UlOrderScratchContext*>(ctx);
    if (scratch_ctx == nullptr || scratch_ctx->owner == nullptr)
    {
        return nullptr;
    }
    return scratch_ctx->owner->ul_order_scratch_sym_prb_info(
        scratch_ctx->ring_idx, static_cast<uint32_t>(scratch_ctx->channel), cell_idx);
}

/// Compact UL_TTI lane to messages whose store-time sidecar satisfies @p keep.
///
/// @param slot_store  Current ring-slot storage (sidecar source).
/// @param filtered    Output descriptors (capacity @c MAX_CELLS_PER_SLOT).
/// @param keep        Predicate on @c NvUlTtiPduCounts (e.g. PUSCH/PRACH/PUCCH/SRS).
/// @return Number of entries written to @p filtered.
template<std::predicate<const nv::NvUlTtiPduCounts&> Predicate>
[[nodiscard]] uint16_t filter_ul_tti_messages(const nv::FapiSlotMessageStorage& slot_store,
                                              std::array<nv::phy_mac_msg_desc, MAX_CELLS_PER_SLOT>& filtered,
                                              Predicate&& keep)
{
    const nv::phy_mac_msg_desc* msgs = slot_store.ul_tti_messages();
    const uint16_t              n    = slot_store.ul_tti_count();
    uint16_t out = 0;
    for (uint16_t i = 0; i < n && out < filtered.size(); ++i)
    {
        if (!keep(slot_store.ul_tti_pdu_counts(i)))
        {
            continue;
        }
        filtered[out++] = msgs[i];
    }
    return out;
}
} // namespace

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 14) // "L2A.UL_CHANNELS"

namespace nv {

// ---------------------------------------------------------------------------
// UL channel aggregation handlers
// ---------------------------------------------------------------------------

void PHY_module::process_aggr_pusch_channel(const UlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];

    NVLOGD_FMT(TAG, "process_aggr_pusch: slot=0x{:08X} ring_idx={} ul_tti_msgs={}",
               ctx.slot_u32, ctx.ring_idx, slot_store.ul_tti_count());

    const phy_mac_msg_desc* msgs = slot_store.ul_tti_messages();
    const uint16_t          n    = slot_store.ul_tti_count();
    if (msgs == nullptr && n > 0u) [[unlikely]] {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_pusch: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} - skipping",
                   ctx.slot_u32, ctx.ring_idx, n);
        return;
    }
    if (n != 0u) [[likely]] {
        std::array<phy_mac_msg_desc, MAX_CELLS_PER_SLOT> pusch_msgs{};
        const uint16_t pusch_n = filter_ul_tti_messages(
            slot_store, pusch_msgs,
            [](const NvUlTtiPduCounts& c) { return c.pusch() > 0; });
        if (pusch_n == 0u)
        {
            return;
        }
        UlOrderScratchContext order_scratch_ctx{this, ctx.ring_idx, UlOrderScratchChannel::Pusch};
        scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                               detail::query_l1_srs_enabled(), ctx.slot_cmd_idx,
                                               resolve_ul_order_scratch, &order_scratch_ctx};
        auto* grp_cmd = module_view.group_command();
        if (!grp_cmd || !grp_cmd->pusch) [[unlikely]] {
            // Reached only when PUSCH PDUs are present (task_aggr_pusch is enqueued
            // solely for a non-zero PUSCH count), so a missing slot sub-command
            // here drops live PUSCH — never silent.
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_pusch: slot=0x{:08X} ring_idx={} dropping {} PUSCH message(s): {}",
                       ctx.slot_u32, ctx.ring_idx, pusch_n,
                       !grp_cmd ? "group_command=nullptr"
                                : "cell_group.pusch unset (no PUSCH slot sub-command)");
            return;
        }

        scf_5g_fapi::ULSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};
        if (auto const result = processor.process<UL_TTI_PDU_TYPE_PUSCH>(
                std::span{pusch_msgs.data(), static_cast<std::size_t>(pusch_n)});
            !result.has_value())
        {
            // Log-only by design: the aggregation task must still run to completion
            // (task_work_fn_aggr_pusch returns 0 + on_complete) or tasks_in_flight
            // deadlocks. There is no actionable per-slot recovery for a parse failure,
            // so the error is surfaced via the log/NVLOG event, not the (void) return.
            const auto& err = result.error();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                        "process_aggr_pusch: slot=0x{:08X} ULSlotProcessor failed code={} sfn={} slot={}",
                        ctx.slot_u32,
                        scf_5g_fapi::to_string(err.code),
                        err.sfn,
                        err.slot);
        }
    }
}

void PHY_module::process_aggr_prach_channel(const UlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];

    NVLOGD_FMT(TAG, "process_aggr_prach: slot=0x{:08X} ring_idx={} ul_tti_msgs={}",
               ctx.slot_u32, ctx.ring_idx, slot_store.ul_tti_count());

    const phy_mac_msg_desc* msgs = slot_store.ul_tti_messages();
    const uint16_t          n    = slot_store.ul_tti_count();
    if (msgs == nullptr && n > 0u) [[unlikely]] {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_prach: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} - skipping",
                   ctx.slot_u32, ctx.ring_idx, n);
        return;
    }
    if (n != 0u) [[likely]] {
        std::array<phy_mac_msg_desc, MAX_CELLS_PER_SLOT> prach_msgs{};
        const uint16_t prach_n = filter_ul_tti_messages(
            slot_store, prach_msgs,
            [](const NvUlTtiPduCounts& c) {
                return c.by_type[UL_TTI_NPDUS_IDX_PRACH] > 0;
            });
        if (prach_n == 0u)
        {
            return;
        }

        UlOrderScratchContext order_scratch_ctx{this, ctx.ring_idx, UlOrderScratchChannel::Prach};
        scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                               detail::query_l1_srs_enabled(), ctx.slot_cmd_idx,
                                               resolve_ul_order_scratch, &order_scratch_ctx};
        auto* const grp_cmd = module_view.group_command();
        if (grp_cmd == nullptr || grp_cmd->get_prach_params() == nullptr) [[unlikely]] {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_prach: slot=0x{:08X} ring_idx={} no work scheduled "
                       "(group_command or prach_params unavailable)",
                       ctx.slot_u32, ctx.ring_idx);
            return;
        }

        scf_5g_fapi::ULSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};
        const auto result = processor.process<UL_TTI_PDU_TYPE_PRACH>(
            std::span{prach_msgs.data(), static_cast<std::size_t>(prach_n)});
        if (!result.has_value())
        {
            const auto& err = result.error();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_prach: slot=0x{:08X} ULSlotProcessor failed code={} sfn={} slot={}",
                       ctx.slot_u32,
                       scf_5g_fapi::to_string(err.code),
                       err.sfn,
                       err.slot);
        }
    }
}

void PHY_module::process_aggr_pucch_channel(const UlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];
    const auto* msgs = slot_store.ul_tti_messages();
    const auto  n    = slot_store.ul_tti_count();

    NVLOGD_FMT(TAG, "process_aggr_pucch: slot=0x{:08X} ring_idx={} ul_tti_msgs={}",
               ctx.slot_u32, ctx.ring_idx, n);

    if (msgs == nullptr && n > 0u) [[unlikely]]
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_pucch: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} - skipping",
                   ctx.slot_u32, ctx.ring_idx, n);
        return;
    }

    if (n != 0u) [[likely]]
    {
        std::array<phy_mac_msg_desc, MAX_CELLS_PER_SLOT> pucch_msgs{};
        const uint16_t pucch_n = filter_ul_tti_messages(
            slot_store, pucch_msgs,
            [](const NvUlTtiPduCounts& c) { return c.pucch() > 0; });
        if (pucch_n == 0u)
        {
            return;
        }

        UlOrderScratchContext order_scratch_ctx{this, ctx.ring_idx, UlOrderScratchChannel::Pucch};
        scf_5g_fapi::PhyModuleView module_view{
            *this, detail::query_l1_mmimo_enabled(), detail::query_l1_srs_enabled(), ctx.slot_cmd_idx,
            resolve_ul_order_scratch, &order_scratch_ctx};
        const auto* grp_cmd = module_view.group_command();
        if (!grp_cmd)
        {
            // Slot-command buffer must be non-null at this point; mirrors the SRS
            // sibling check at process_aggr_srs_channel.  Log loudly and drop the
            // slot — the dispatcher is upstream of the parser, so retrying without
            // a buffer would just hit the same condition.
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_pucch: slot=0x{:08X} ring_idx={} group_command()==nullptr "
                       "- dropping slot (invariant violation)",
                       ctx.slot_u32, ctx.ring_idx);
            return;
        }

        scf_5g_fapi::ULSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};

        // PUCCH F0/1 and F2/3/4 share the UL_TTI_PDU_TYPE_PUCCH wire type; PucchPduParser
        // (parse_pucch_pdu_common) handles all formats via format_type, and its expected
        // count spans both F01+F234 sidecar lanes (NvUlTtiPduCounts::pucch()).
        const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(
            std::span{pucch_msgs.data(), static_cast<std::size_t>(pucch_n)});
        if (!result.has_value())
        {
            const auto& err = result.error();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_pucch: slot=0x{:08X} ULSlotProcessor failed code={} sfn={} slot={}",
                       ctx.slot_u32, scf_5g_fapi::to_string(err.code), err.sfn, err.slot);
        }
    }
}

void PHY_module::process_aggr_srs_channel(const UlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];

    NVLOGD_FMT(TAG, "process_aggr_srs: slot=0x{:08X} ring_idx={} ul_tti_msgs={}",
               ctx.slot_u32, ctx.ring_idx, slot_store.ul_tti_count());

    const phy_mac_msg_desc* msgs = slot_store.ul_tti_messages();
    const uint16_t          n    = slot_store.ul_tti_count();
    if (msgs == nullptr && n > 0u) [[unlikely]] {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_srs: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} - skipping",
                   ctx.slot_u32, ctx.ring_idx, n);
        return;
    }

    if (n != 0u) [[likely]] {
        UlOrderScratchContext order_scratch_ctx{this, ctx.ring_idx, UlOrderScratchChannel::Srs};
        scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                               detail::query_l1_srs_enabled(), ctx.slot_cmd_idx,
                                               resolve_ul_order_scratch, &order_scratch_ctx};
        auto* grp_cmd = module_view.group_command();
        if (!grp_cmd || !grp_cmd->srs) [[unlikely]] {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_srs: slot=0x{:08X} ring_idx={} grp_cmd_null={} srs_null={} - "
                       "invariant violation, skipping SRS UL dispatch",
                       ctx.slot_u32, ctx.ring_idx,
                       grp_cmd == nullptr,
                       grp_cmd == nullptr || grp_cmd->srs.get() == nullptr);
            return;
        }

        scf_5g_fapi::ULSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};
        if (auto const result = processor.process<UL_TTI_PDU_TYPE_SRS>(
                std::span{msgs, n},
                std::span<const NvUlTtiPduCounts>{slot_store.ul_tti_pdu_counts_data(), n});
            !result.has_value())
        {
            const auto& err = result.error();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_srs: slot=0x{:08X} ULSlotProcessor failed code={} sfn={} slot={}",
                       ctx.slot_u32,
                       scf_5g_fapi::to_string(err.code),
                       err.sfn,
                       err.slot);
        }
        NVLOGD_FMT(TAG,
                   "process_aggr_srs: slot=0x{:08X} ring_idx={} ul_tti_msgs={} nCells={} nSrsUes={}",
                   ctx.slot_u32,
                   ctx.ring_idx,
                   n,
                   grp_cmd->srs->cell_grp_info.nCells,
                   grp_cmd->srs->cell_grp_info.nSrsUes);
    }
}

void PHY_module::process_aggr_ulbfw_channel(const UlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];

    auto& slot_cmd_entry = slot_command_array[ctx.slot_cmd_idx];
    auto* grp_cmd = &slot_cmd_entry.cell_groups;

    dispatch_ul_bfw_messages(slot_store, phy_refs_.size(),
        [this, &slot_cmd_entry, grp_cmd, slot_u32 = ctx.slot_u32](
            const int cell_id, const phy_mac_msg_desc& msg) {

            if (msg.msg_buf == nullptr) {
                NVLOGW_FMT(TAG, "process_aggr_ulbfw: msg_buf=nullptr cell_id={} slot=0x{:08X}",
                           cell_id, slot_u32);
                return;
            }

            if (msg.msg_len < 0) {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "process_aggr_ulbfw: negative msg_len={} cell_id={} slot=0x{:08X}",
                           msg.msg_len, cell_id, slot_u32);
                return;
            }
            if (static_cast<std::size_t>(msg.msg_len) < kUlBfwCviMsgMinBytes) {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "process_aggr_ulbfw: msg_len={} < min={} (fapi_hdr + cvi_req fixed) cell_id={} slot=0x{:08X}",
                           msg.msg_len, kUlBfwCviMsgMinBytes, cell_id, slot_u32);
                return;
            }

            const auto* req = reinterpret_cast<const scf_fapi_ul_bfw_cvi_request_t*>(
                static_cast<const char*>(msg.msg_buf) + sizeof(scf_fapi_header_t));

            NVLOGD_FMT(TAG, "  ul_bfw: cell_id={} sfn={} slot={} npdus={}",
                       cell_id, req->sfn, req->slot, req->npdus);

            if (!phy_refs_[cell_id].get().is_ul_bfw_cvi_feature_allowed())
            {
                NVLOGW_FMT(TAG,
                           "process_aggr_ulbfw: UL BFW CVI rejected (SRS or mMIMO disabled) cell_id={} "
                           "slot=0x{:08X} sfn={} slot={}",
                           cell_id, slot_u32, req->sfn, req->slot);
                phy_refs_[cell_id].get().reject_ul_bfw_cvi_feature_disabled(
                    req->msg_hdr.type_id, req->sfn, req->slot);
                return;
            }

            auto& cell_cmd = slot_cmd_entry.cells[cell_id];

            slot_command_api::slot_indication slot_ind{req->sfn, req->slot, /*tick=*/0};

            auto* const bfw_info = acquire_free_bfw_coeff_buff(
                static_cast<uint32_t>(cell_id),
                req->slot % MAX_BFW_COFF_STORE_INDEX);

            const std::size_t available_bytes =
                static_cast<std::size_t>(msg.msg_len) - kUlBfwCviMsgMinBytes;
            const uint8_t* data = reinterpret_cast<const uint8_t*>(req->config_pdu);
            std::size_t offset = 0;
            uint32_t droppedPdu = 0;

            for (uint8_t p = 0; p < req->npdus; ++p) {
                if (offset > available_bytes ||
                    available_bytes - offset < sizeof(scf_fapi_ul_bfw_group_config_t)) {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "process_aggr_ulbfw: pdu[{}] header truncated offset={} available={} "
                               "cell_id={} slot=0x{:08X} - aborting PDU loop",
                               p, offset, available_bytes, cell_id, slot_u32);
                    break;
                }
                const auto& pdu = *reinterpret_cast<const scf_fapi_ul_bfw_group_config_t*>(
                    data + offset);
                const uint16_t pdu_size = pdu.pdu_size;
                if (pdu_size == 0) {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                                 "process_aggr_ulbfw: pdu[{}] has pdu_size=0 cell_id={} slot=0x{:08X} - aborting PDU loop",
                                 p, cell_id, slot_u32);
                    break;
                }
                NVLOGD_FMT(TAG, "    pdu[{}]: pdu_size={}", p, pdu_size);
                if (pdu_size > available_bytes - offset) {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                               "process_aggr_ulbfw: pdu[{}] pdu_size={} exceeds remaining={} "
                               "offset={} available={} cell_id={} slot=0x{:08X} - aborting PDU loop",
                               p, pdu_size, available_bytes - offset,
                               offset, available_bytes, cell_id, slot_u32);
                    break;
                }
                phy_refs_[cell_id].get().apply_ul_bfw_pdu(
                    pdu, slot_ind, grp_cmd, cell_cmd, bfw_info, droppedPdu);
                offset += pdu_size;
            }

            // Send ERROR IND for each dropped UL BFW PDU
            if (droppedPdu > 0) {
                NVLOGW_FMT(TAG,
                    "process_aggr_ulbfw: cell_id={} sfn={} slot={} dropped={}/{} PDUs - sending error indications",
                    cell_id, req->sfn, req->slot, droppedPdu, req->npdus);
                phy_refs_[cell_id].get().send_bfw_error_indications(
                    req->msg_hdr.type_id, req->sfn, req->slot, droppedPdu);
            }

            const bool bfw_coeff_header_freed = detail::debug_free_busy_bfw_coeff_header(bfw_info);
            if (!bfw_coeff_header_freed)
            {
                NVLOGW_FMT(TAG,
                    "process_aggr_ulbfw: BFW coeff debug cleanup skipped for cell_id={} sfn={} slot={}; "
                    "header was null or not BUSY",
                    cell_id, req->sfn, req->slot);
            }
        },
        ctx.slot_u32, ctx.ring_idx);
}

} // namespace nv
