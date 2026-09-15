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
 * @file nv_phy_dl_channels.cpp
 * @brief DL channel aggregation handlers for PHY_module.
 *
 * Implements PHY_module::process_aggr_*_channel for the DL path (PDSCH,
 * CSI-RS, PDCCH, SSB, DL-BFW) and the TX_DATA H2D staging/launch stubs.
 * Each function is invoked by a worker thread via the DlChannelDispatch
 * table wired up in nv_phy_module.hpp and populated at EOM in
 * enqueue_channel_tasks().
 *
 * Anonymous-namespace helpers encapsulate common FAPI iteration patterns
 * so each channel handler contains only its channel-specific logic.
 * @see nv_phy_ul_channels.cpp for the symmetric UL path.
 */

#include "nv_phy_module.hpp"
#include "nv_phy_driver_proxy.hpp"
#include "aerial/casts/casts.hpp"     // aerial::casts::assume_cast (alignment-checked)
#include "nv_fapi_pdu_utils.hpp"
#include "nv_dl_aggr_tasks.hpp"
#include "nv_bfw_debug_cleanup.hpp"
#include "scf_5g_fapi_dl_slot_processor.hpp"
#include "scf_5g_fapi_message_context.hpp"
#include "scf_5g_csirs_slot_command_helpers.hpp"
#include "nv_tx_data_h2d_helpers.hpp"
#include "nv_pdsch_tb_merge.hpp"
#include "scf_5g_fapi.h"
#include "scf_5g_slot_commands_common.hpp"
#include "cuphy_api.h"

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

static_assert(std::is_standard_layout_v<scf_fapi_dl_bfw_cvi_request_t>,
              "scf_fapi_dl_bfw_cvi_request_t must be standard-layout for reinterpret_cast from raw FAPI buffer");
static_assert(std::is_standard_layout_v<scf_fapi_dl_bfw_group_config_t>,
              "scf_fapi_dl_bfw_group_config_t must be standard-layout for reinterpret_cast from raw FAPI buffer");

namespace
{
/// Minimum `phy_mac_msg_desc::msg_len` for a DL BFW CVI request after `scf_fapi_header_t`
/// (fixed `scf_fapi_dl_bfw_cvi_request_t` prefix; `config_pdu[]` is not counted in sizeof).
constexpr std::size_t kDlBfwCviMsgMinBytes =
    sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_bfw_cvi_request_t);
} // namespace

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 13) // "L2A.DL_CHANNELS"

namespace nv
{

// ---------------------------------------------------------------------------
// DL channel aggregation handlers
// ---------------------------------------------------------------------------

void PHY_module::process_aggr_pdsch_channel(const DlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];
    NVLOGD_FMT(TAG, "process_aggr_pdsch: slot=0x{:08X} ring_idx={} dl_tti_msgs={} "
                    "split_phase={}",
               ctx.slot_u32,
               ctx.ring_idx,
               slot_store.dl_tti_count(),
               ctx.staged_tb_ptrs != nullptr);
    const phy_mac_msg_desc* msgs = slot_store.dl_tti_messages();
    const uint16_t          n    = slot_store.dl_tti_count();
    if(msgs == nullptr && n > 0u) [[unlikely]]
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_pdsch: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} - skipping", ctx.slot_u32, ctx.ring_idx, n);
        return;
    }
    if(n != 0u) [[likely]]
    {
        scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                               detail::query_l1_srs_enabled(), ctx.slot_cmd_idx};
        auto*                      grp_cmd = module_view.group_command();
        if(!grp_cmd || !grp_cmd->pdsch) { return; }

        scf_5g_fapi::DLSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};
        if(auto const result = processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{msgs, n});
           !result.has_value())
        {
            const auto& err = result.error();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_pdsch: slot=0x{:08X} DLSlotProcessor failed code={} sfn={} slot={}", ctx.slot_u32, scf_5g_fapi::to_string(err.code), err.sfn, err.slot);
        }
    }
    // Split-phase TX_DATA H2D: staged_tb_ptrs is non-null only when
    // launch_tx_data_h2d (Phase 2) succeeded and all per-cell GPU TB pointers
    // were frozen into the task arg at EOM.  On mismatch/failure paths both
    // staged_tb_ptrs and staged_tx_msg_bufs are nullptr, skipping this block
    // entirely and leaving pTbInput null so cuPHY skips PDSCH for this slot.
    if(ctx.staged_tb_ptrs != nullptr)
    {
        auto& slot_cmd_entry = slot_command_array[ctx.slot_cmd_idx];
        auto* pdsch          = slot_cmd_entry.cell_groups.get_pdsch_params();
        if(pdsch != nullptr) [[likely]]
        {
            const auto gpu_span = std::span<uint8_t* const>{ctx.staged_tb_ptrs, ctx.n_staged_tb_ptrs};
            const auto msg_span = (ctx.staged_tx_msg_bufs != nullptr) ? std::span<const void* const>{ctx.staged_tx_msg_bufs, ctx.n_staged_tb_ptrs} : std::span<const void* const>{};
            const auto stats    = nv::pdsch_merge::merge_and_patch(
                slot_store, gpu_span, msg_span, *pdsch, ctx.slot_u32);
            NVLOGD_FMT(TAG,
                       "process_aggr_pdsch: split-phase merge slot=0x{:08X} "
                       "ue_tb_assigned={} tb_offsets_applied={} overflow={} "
                       "bufferType=GPU",
                       ctx.slot_u32,
                       stats.ue_tb_assigned,
                       stats.tb_offsets_applied,
                       stats.overflow_count);
        }
    }
}

void PHY_module::process_aggr_csirs_channel(const DlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];
    const uint16_t n_dl_tti = slot_store.dl_tti_count();

    NVLOGD_FMT(TAG, "process_aggr_csirs: slot=0x{:08X} ring_idx={} dl_tti_msgs={}",
               ctx.slot_u32, ctx.ring_idx, n_dl_tti);

    // Precompute PDSCH dynamic-info ordinals from DL_TTI storage order.
    // pdsch_ordinal[i] = number of PDSCH-carrying DL_TTI messages at
    // indices 0..i-1, which is the dyn_idx into cell_dyn_info[] for
    // cell i when it has PDSCH.
    std::array<uint32_t, MAX_CELLS_PER_SLOT> pdsch_ordinal{};
    uint32_t pdsch_cell_count = 0;
    for (uint16_t j = 0; j < n_dl_tti && j < pdsch_ordinal.size(); ++j)
    {
        pdsch_ordinal[j] = std::exchange(
            pdsch_cell_count,
            pdsch_cell_count + (slot_store.dl_tti_has_pdsch_pdu(j) ? 1u : 0u));
    }

    const bool mmimo_enabled = detail::query_l1_mmimo_enabled();

    // Iterate DL_TTI lane cells that carry CSI-RS (store-time sidecar; no
    // nPDUsOfEachType[] / payload walk). For each qualifying cell, invoke the
    // handler with the message index, descriptor, and decoded request.
    for_each_tti_msg<TAG, scf_fapi_dl_tti_req_t>(slot_store.dl_tti_messages(), slot_store.dl_tti_count(),
        [&slot_store](uint16_t i, const scf_fapi_dl_tti_req_t&) {
            return slot_store.dl_tti_pdu_counts(i).by_type[DL_TTI_NPDUS_IDX_CSI_RS] > 0;
        },
        ctx.slot_u32, ctx.ring_idx,
        [&slot_store, &pdsch_ordinal, mmimo_enabled, &ctx, this]
        (uint16_t i, const phy_mac_msg_desc& msg, const scf_fapi_dl_tti_req_t& req) {
            NVLOGD_FMT(TAG, "  csirs[{}]: cell_id={} num_pdus={} num_csirs_pdus={}",
                       i, msg.cell_id, req.num_pdus,
                       slot_store.dl_tti_pdu_counts(i).by_type[DL_TTI_NPDUS_IDX_CSI_RS]);

            const int32_t carrier_id   = PHY_instances()[msg.cell_id].get().get_carrier_id();
            const uint16_t phy_cell_id = PHY_instances()[msg.cell_id].get().get_phy_cell_id();
            const bool has_pdsch       = slot_store.dl_tti_has_pdsch_pdu(i);
            const uint32_t dyn_idx     = pdsch_ordinal[i];
            bool captured_csirs_offset = false;
            uint32_t csirs_offset      = 0;

            auto& slot_cmd_entry = slot_command_array[ctx.slot_cmd_idx];
            auto& cell_grp_cmd   = slot_cmd_entry.cell_groups;

            if (carrier_id < 0 ||
                static_cast<size_t>(carrier_id) >= slot_cmd_entry.cells.size()) {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "process_aggr_csirs: carrier_id={} out of range (cells.size={}) "
                           "cell_id={} slot=0x{:08X} - skipping DL_TTI msg",
                           carrier_id, slot_cmd_entry.cells.size(),
                           msg.cell_id, ctx.slot_u32);
                return;
            }
            auto& cell_cmd       = slot_cmd_entry.cells[carrier_id];
            auto& slotinfo       = cell_grp_cmd.slot.slot_3gpp;

            cell_cmd.cell = phy_cell_id;

            for_each_pdu<TAG>(msg.cell_id,
                aerial::casts::assume_cast<const scf_fapi_generic_pdu_info_t>(req.payload),
                req.num_pdus,
                [&cell_grp_cmd, &cell_cmd, &slotinfo,
                 &captured_csirs_offset, &csirs_offset,
                 carrier_id, has_pdsch, mmimo_enabled, dyn_idx,
                 &msg, this]
                (uint16_t p, const scf_fapi_generic_pdu_info_t& pdu) {
                    if (pdu.pdu_type == DL_TTI_PDU_TYPE_CSI_RS) {
                        const auto& csi_pdu =
                            *aerial::casts::assume_cast<const scf_fapi_csi_rsi_pdu_t>(pdu.pdu_config);
                        NVLOGD_FMT(TAG,
                                   "    csirs pdu[{}]: applying slot_cmd cell_id={} carrier_id={} has_pdsch={} "
                                   "csirs_offset={} captured={} mmimo={} pdsch_dyn_idx={}",
                                   p, msg.cell_id, carrier_id, has_pdsch,
                                   csirs_offset, captured_csirs_offset, mmimo_enabled, dyn_idx);
                        scf_5g_fapi::apply_csirs_dl_aggr_slot_command_for_pdu(
                            cell_grp_cmd, cell_cmd, csi_pdu, slotinfo,
                            carrier_id, has_pdsch,
                            captured_csirs_offset, csirs_offset,
                            config_options(), pm_map(), mmimo_enabled,
                            dyn_idx);
                    }
                });

            // TEMPORARY(csirs-fh-aggr-path): begin -- populate FH params for CSI-RS cell
            {
                slot_command_api::csirs_params* csirs = cell_grp_cmd.csirs.get();
                const uint32_t nCsirs = csirs
                    ? csirs->cellInfo[csirs->nCells > 0 ? csirs->nCells - 1 : 0].nRrcParams
                    : 0;
                if (has_pdsch || nCsirs > 0)
                {
                    const uint16_t num_dl_prb =
                        static_cast<scf_5g_fapi::phy&>(PHY_instances()[msg.cell_id].get())
                            .get_phy_cell_params().nPrbDlBwp;
                    scf_5g_fapi::populate_csirs_fh_params_for_cell(
                        cell_grp_cmd.fh_params, cell_cmd,
                        carrier_id, has_pdsch,
                        static_cast<int32_t>(dyn_idx),
                        bf_enabled(), mmimo_enabled, num_dl_prb);
                }
            }
            // TEMPORARY(csirs-fh-aggr-path): end
        });
}

void PHY_module::process_aggr_pdcch_dl_tti(const DlAggrTaskArg& ctx,
                                           const phy_mac_msg_desc* msgs, uint16_t n)
{
    scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                           detail::query_l1_srs_enabled(), ctx.slot_cmd_idx};
    auto* grp_cmd = module_view.group_command();
    if (grp_cmd == nullptr
        || grp_cmd->channel_idx[slot_command_api::channel_type::PDCCH_DL]
            == slot_command_api::channel_type::NONE) {
        NVLOGD_FMT(TAG,
                   "process_aggr_pdcch: slot=0x{:08X} no DL PDCCH common registration; skipping DL_TTI PDCCH aggregation",
                   ctx.slot_u32);
        return;
    }

    scf_5g_fapi::DLSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};
    auto const result = processor.process<DL_TTI_PDU_TYPE_PDCCH>(std::span{msgs, n});
    if (!result.has_value())
    {
        const auto& err = result.error();
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_pdcch: slot=0x{:08X} DLSlotProcessor failed code={} sfn={} slot={}",
                   ctx.slot_u32,
                   scf_5g_fapi::to_string(err.code),
                   err.sfn,
                   err.slot);
    }
}

void PHY_module::process_aggr_pdcch_channel(const DlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];

    NVLOGD_FMT(TAG, "process_aggr_pdcch: slot=0x{:08X} ring_idx={} dl_tti_msgs={} ul_dci_msgs={}",
               ctx.slot_u32, ctx.ring_idx, slot_store.dl_tti_count(), slot_store.ul_dci_count());

    const phy_mac_msg_desc* msgs = slot_store.dl_tti_messages();
    const uint16_t          n    = slot_store.dl_tti_count();
    if (msgs == nullptr && n > 0u) [[unlikely]] {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_pdcch: slot=0x{:08X} ring_idx={} dl_tti msgs=nullptr but n={} - skipping DL PDCCH",
                   ctx.slot_u32, ctx.ring_idx, n);
    }
    else if (n != 0u) [[likely]] {
        process_aggr_pdcch_dl_tti(ctx, msgs, n);
    }

    // UL_DCI.req: carries downlink DCI for UL grants — iterate all messages.
    const phy_mac_msg_desc* ul_msgs = slot_store.ul_dci_messages();
    const uint16_t          ul_n    = slot_store.ul_dci_count();
    if(ul_msgs == nullptr && ul_n > 0)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci msgs=nullptr but n={} - skipping loop", ctx.slot_u32, ctx.ring_idx, ul_n);
        return;
    }
    if(ul_n == 0u)
    {
        return;
    }

    scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                           detail::query_l1_srs_enabled(), ctx.slot_cmd_idx};
    scf_5g_fapi::PdcchPduParser<scf_5g_fapi::PhyModuleView> parser{module_view};

    for(uint16_t i = 0; i < ul_n; ++i)
    {
        if(ul_msgs[i].msg_buf == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci[{}].msg_buf=nullptr - skipping", ctx.slot_u32, ctx.ring_idx, i);
            continue;
        }
        constexpr std::size_t k_min_ul_dci_msg_len =
            sizeof(scf_fapi_header_t) + sizeof(scf_fapi_ul_dci_t);
        // msg_len is int32_t; reject negative values before the size_t comparison
        // (matches the nv_fapi_pdu_utils for_each_tti_msg pattern).
        if(ul_msgs[i].msg_len < 0 ||
           static_cast<std::size_t>(ul_msgs[i].msg_len) < k_min_ul_dci_msg_len)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci[{}].msg_len={} too small (need {}) - skipping",
                       ctx.slot_u32, ctx.ring_idx, i, ul_msgs[i].msg_len, k_min_ul_dci_msg_len);
            continue;
        }
        const std::size_t msg_len = static_cast<std::size_t>(ul_msgs[i].msg_len);

        const auto* msg_base    = static_cast<const uint8_t*>(ul_msgs[i].msg_buf);
        const auto* req         = aerial::casts::assume_cast<const scf_fapi_ul_dci_t>(
            msg_base + sizeof(scf_fapi_header_t));
        const auto* payload     = req->payload;
        const auto* payload_end = msg_base + msg_len;

        if (req->num_pdus == 0u)
        {
            // Nothing to walk; skip the setup_cell push entirely so a zero-PDU
            // UL_DCI message does not register a dangling carrier on the parser.
            continue;
        }

        // UL_DCI path: testMode is always 0; bridge flag derived from the
        // view's mode (legacy enables, direct disables).
        const bool ul_dci_bridge =
            !module_view.is_fapi_to_cplane_direct_enabled();
        parser.setup_cell(static_cast<uint32_t>(ul_msgs[i].cell_id),
                          /*test_mode=*/0u,
                          ul_dci_bridge);
        auto* cursor = payload;
        // SCF FAPI 10.04 UL_DCI.req Table 3-55: only PDCCH PDUs (pdu_type==0)
        // are defined for this message.
        constexpr uint8_t k_ul_dci_pdu_type_pdcch = 0u;
        for(uint8_t pdu_idx = 0u; pdu_idx < req->num_pdus; ++pdu_idx)
        {
            if(static_cast<std::size_t>(payload_end - cursor) < sizeof(scf_fapi_generic_pdu_info_t))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci[{}] pdu[{}] header exceeds payload - stopping",
                           ctx.slot_u32, ctx.ring_idx, i, pdu_idx);
                break;
            }
            const auto* pdu_info = aerial::casts::assume_cast<const scf_fapi_generic_pdu_info_t>(cursor);
            // Enforce the combined generic-header + PDCCH-body floor here so the
            // assume_cast<scf_fapi_pdcch_pdu_t>(pdu_info->pdu_config) below has
            // at least sizeof(scf_fapi_pdcch_pdu_t) bytes to read. Mirrors the
            // k_min_size pattern in scf_5g_fapi_tti_dispatch.hpp.
            constexpr std::size_t k_min_pdcch_pdu_size =
                sizeof(scf_fapi_generic_pdu_info_t) + sizeof(scf_fapi_pdcch_pdu_t);
            if(pdu_info->pdu_size < k_min_pdcch_pdu_size
               || cursor + pdu_info->pdu_size > payload_end)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci[{}] pdu[{}] invalid pdu_size={} (need >= {}) - stopping",
                           ctx.slot_u32, ctx.ring_idx, i, pdu_idx, pdu_info->pdu_size, k_min_pdcch_pdu_size);
                break;
            }
            if(pdu_info->pdu_type != k_ul_dci_pdu_type_pdcch)
            {
                // Unexpected pdu_type in UL_DCI.req: treat as malformed message and stop
                // walking — we cannot trust pdu_size from a wrong-typed header.
                NVLOGW_FMT(TAG,
                           "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci[{}] pdu[{}] unexpected pdu_type={} - stopping",
                           ctx.slot_u32, ctx.ring_idx, i, pdu_idx, pdu_info->pdu_type);
                break;
            }

            const auto* pdcch = aerial::casts::assume_cast<const scf_fapi_pdcch_pdu_t>(pdu_info->pdu_config);
            if(!parser.parse(req->sfn, req->slot, *pdcch))
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                           "process_aggr_pdcch: slot=0x{:08X} ring_idx={} ul_dci[{}] pdu[{}] PDCCH parser failed",
                           ctx.slot_u32, ctx.ring_idx, i, pdu_idx);
            }
            cursor += pdu_info->pdu_size;
        }
    }
}

void PHY_module::process_aggr_ssb_channel(const DlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];
    const auto* msgs = slot_store.dl_tti_messages();
    const auto  n    = slot_store.dl_tti_count();

    NVLOGD_FMT(TAG, "process_aggr_ssb: slot=0x{:08X} ring_idx={} dl_tti_msgs={}",
               ctx.slot_u32, ctx.ring_idx, n);

    if(msgs == nullptr && n > 0u) [[unlikely]]
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                   "process_aggr_ssb: slot=0x{:08X} ring_idx={} msgs=nullptr but n={} - skipping",
                   ctx.slot_u32, ctx.ring_idx, n);
        return;
    }

    if(n != 0u) [[likely]]
    {
        scf_5g_fapi::PhyModuleView module_view{*this, detail::query_l1_mmimo_enabled(),
                                               detail::query_l1_srs_enabled(), ctx.slot_cmd_idx};
        auto* grp_cmd = module_view.group_command();
        if(!grp_cmd) { return; }

        scf_5g_fapi::DLSlotProcessor<scf_5g_fapi::PhyModuleView> processor{module_view};
        // On partial-PDU failure pbch_group_params may already carry accepted entries;
        // the slot is still submitted downstream — mirrors process_aggr_pdsch_channel.
        if(auto const result = processor.process<DL_TTI_PDU_TYPE_SSB>(std::span{msgs, n});
           !result.has_value())
        {
            const auto& err = result.error();
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT,
                       "process_aggr_ssb: slot=0x{:08X} DLSlotProcessor failed code={} sfn={} slot={}",
                       ctx.slot_u32,
                       scf_5g_fapi::to_string(err.code),
                       err.sfn,
                       err.slot);
        }
    }
}

void PHY_module::process_aggr_dlbfw_channel(const DlAggrTaskArg& ctx)
{
    const FapiSlotMessageStorage& slot_store = slot_message_storage_[ctx.ring_idx];

    auto& slot_cmd_entry = slot_command_array[ctx.slot_cmd_idx];
    auto* grp_cmd        = &slot_cmd_entry.cell_groups;

    dispatch_dl_bfw_messages(
        slot_store, phy_refs_.size(), [this, &slot_cmd_entry, grp_cmd, slot_u32 = ctx.slot_u32](const int cell_id, const phy_mac_msg_desc& msg) {
            if(msg.msg_buf == nullptr)
            {
                NVLOGW_FMT(TAG, "process_aggr_dlbfw: msg_buf=nullptr cell_id={} slot=0x{:08X}", cell_id, slot_u32);
                return;
            }

            if(msg.msg_len < 0)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_dlbfw: negative msg_len={} cell_id={} slot=0x{:08X}", msg.msg_len, cell_id, slot_u32);
                return;
            }
            if(static_cast<std::size_t>(msg.msg_len) < kDlBfwCviMsgMinBytes)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_dlbfw: msg_len={} < min={} (fapi_hdr + cvi_req fixed) cell_id={} slot=0x{:08X}", msg.msg_len, kDlBfwCviMsgMinBytes, cell_id, slot_u32);
                return;
            }

            const auto* req = aerial::casts::assume_cast<const scf_fapi_dl_bfw_cvi_request_t>(
                static_cast<const char*>(msg.msg_buf) + sizeof(scf_fapi_header_t));

            NVLOGD_FMT(TAG, "  dl_bfw: cell_id={} sfn={} slot={} npdus={}", cell_id, req->sfn, req->slot, req->npdus);

            if(!phy_refs_[cell_id].get().is_dl_bfw_cvi_feature_allowed())
            {
                NVLOGW_FMT(TAG,
                           "process_aggr_dlbfw: DL BFW CVI rejected (SRS or mMIMO disabled) cell_id={} "
                           "slot=0x{:08X} sfn={} slot={}",
                           cell_id,
                           slot_u32,
                           req->sfn,
                           req->slot);
                phy_refs_[cell_id].get().reject_dl_bfw_cvi_feature_disabled(
                    req->msg_hdr.type_id, req->sfn, req->slot);
                return;
            }

            auto& cell_cmd = slot_cmd_entry.cells[cell_id];

            slot_command_api::slot_indication slot_ind{req->sfn, req->slot, /*tick=*/0};

            auto* const bfw_info = acquire_free_bfw_coeff_buff(
                static_cast<uint32_t>(cell_id),
                req->slot % MAX_BFW_COFF_STORE_INDEX);

            const std::size_t available_bytes =
                static_cast<std::size_t>(msg.msg_len) - kDlBfwCviMsgMinBytes;
            const uint8_t* data       = aerial::casts::assume_cast<const uint8_t>(req->config_pdu);
            std::size_t    offset     = 0;
            uint32_t       droppedPdu = 0;

            for(uint8_t p = 0; p < req->npdus; ++p)
            {
                if(offset > available_bytes ||
                   available_bytes - offset < sizeof(scf_fapi_dl_bfw_group_config_t))
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_dlbfw: pdu[{}] header truncated offset={} available={} "
                                                            "cell_id={} slot=0x{:08X} - aborting PDU loop",
                               p,
                               offset,
                               available_bytes,
                               cell_id,
                               slot_u32);
                    break;
                }
                const auto& pdu = *aerial::casts::assume_cast<const scf_fapi_dl_bfw_group_config_t>(
                    data + offset);
                const uint16_t pdu_size = pdu.pdu_size;
                if(pdu_size == 0)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_dlbfw: pdu[{}] has pdu_size=0 cell_id={} slot=0x{:08X} - aborting PDU loop", p, cell_id, slot_u32);
                    break;
                }
                NVLOGD_FMT(TAG, "    pdu[{}]: pdu_size={}", p, pdu_size);
                if(pdu_size > available_bytes - offset)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "process_aggr_dlbfw: pdu[{}] pdu_size={} exceeds remaining={} "
                                                            "offset={} available={} cell_id={} slot=0x{:08X} - aborting PDU loop",
                               p,
                               pdu_size,
                               available_bytes - offset,
                               offset,
                               available_bytes,
                               cell_id,
                               slot_u32);
                    break;
                }
                phy_refs_[cell_id].get().apply_dl_bfw_pdu(
                    pdu, slot_ind, grp_cmd, cell_cmd, bfw_info, droppedPdu);
                offset += pdu_size;
            }
            if(droppedPdu > 0)
            {
                NVLOGW_FMT(TAG,
                           "process_aggr_dlbfw: cell_id={} sfn={} slot={} dropped={}/{} PDUs - sending error indications",
                           cell_id,
                           req->sfn,
                           req->slot,
                           droppedPdu,
                           req->npdus);
                phy_refs_[cell_id].get().send_bfw_error_indications(
                    req->msg_hdr.type_id, req->sfn, req->slot, droppedPdu);
            }

            const bool bfw_coeff_header_freed = detail::debug_free_busy_bfw_coeff_header(bfw_info);
            if (!bfw_coeff_header_freed)
            {
                NVLOGW_FMT(TAG,
                           "process_aggr_dlbfw: BFW coeff debug cleanup skipped for cell_id={} sfn={} slot={}; "
                           "header was null or not BUSY",
                           cell_id,
                           req->sfn,
                           req->slot);
            }
        },
        ctx.slot_u32,
        ctx.ring_idx);
}

} // namespace nv
