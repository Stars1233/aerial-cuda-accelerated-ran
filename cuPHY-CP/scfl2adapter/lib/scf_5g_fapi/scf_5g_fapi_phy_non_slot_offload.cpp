/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "scf_5g_fapi_phy.hpp"
#include "scf_5g_fapi_config_decode.hpp"
#include "aerial/casts/casts.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 3) // "SCF.PHY"

namespace scf_5g_fapi
{

// fake_phy_cell_id is shared with the serial path; see scf_5g_fapi_phy.hpp.

// IPC threading note: this function and on_cell_stop_request_offload() below
// run on the LP non-slot worker thread (NonSlotLpExecutor). They publish FAPI
// indications via send_error_indication() and transport.tx_send(), which
// forward into NVIPC's lock-free MPMC queue (see the threading note in
// nv_phy_non_slot_dispatch.cpp's V1 worker-exception contract for full
// rationale). Concurrent producers from the message-processing thread and the
// LP worker on the same per-cell transport are explicitly supported by NVIPC,
// so calling send_error_indication() from here is safe.
// metrics_ is also safe here: maps are initialized before runtime and
// prometheus::Counter::Increment() is thread-safe.
void phy::on_config_request_offload(int32_t cell_id, nv::phy_mac_msg_desc& msg)
{
    const char* func = __FUNCTION__;
    // payload is a flexible array member, so body is non-null iff hdr is;
    // both reduce to msg_buf being valid.
    gsl_Expects(msg.msg_buf != nullptr);
    auto* hdr = aerial::casts::assume_cast<scf_fapi_header_t>(msg.msg_buf);
    auto* body = aerial::casts::assume_cast<scf_fapi_body_header_t>(hdr->payload);
    gsl_Expects(body->type_id == SCF_FAPI_CONFIG_REQUEST);

    auto* config_request = aerial::casts::assume_cast<scf_fapi_config_request_msg_t>(body);
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();

    auto process_reconfig = [&] {
        NVLOGC_FMT(TAG, "{}: offloaded CONFIG.req received for cell_id={} in CONFIGURED state", func, cell_id);
        if(phyDriver.l1_lock_update_cell_config_mutex() == false)
        {
            NVLOGC_FMT(TAG, "{}: send CONFIG.res for cell_id={} - try lock failed", func, cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_INVALID_STATE);
            state = fapi_state_t::FAPI_STATE_CONFIGURED;
            return;
        }

        update_cells_stats(cell_id);
        // Shared with the serial reconfig path; see scf_5g_fapi_config_decode.hpp.
        const bool dbt_pdu_present = decode_reconfig_tlvs(*config_request, cell_update_config);

        if (phyDriver.l1_phy_cell_id_mismatch(cell_id, cell_update_config.cell_config_.phy_cell_id))
        {
            NVLOGW_FMT(TAG, "{}: cell_id={} can't be updated because phyCellId doesn't match: old={} new={}",
                func, cell_id, phy_driver_info.phy_stat.phyCellId, cell_update_config.cell_config_.phy_cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_INVALID_CONFIG);
            phyDriver.l1_unlock_update_cell_config_mutex();
            return;
        }

        if (dbt_pdu_present)
        {
            if (msg.data_pool == NV_IPC_MEMPOOL_CPU_LARGE && msg.data_buf != nullptr && phy_module().bf_enabled())
            {
                const int ret = update_dbt_pdu_table_ptr(cell_id, msg.data_buf);
                if (ret != 0) {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Failed to store DBT PDU for cell_id={} in CONFIGURED state", func, cell_id);
                }
            }
            else
            {
                NVLOGW_FMT(TAG, "{}: Beamforming not enabled or invalid buffer for DBT PDU in CONFIGURED state", func);
            }
        }

        update_phy_driver_info_reconfig(phyDriver, cell_id);
        MemtraceDisableScope md;
        // Single caller-runs entry: it branches internally on active cells -- when others
        // are running it drives the staged slot-boundary PRACH handover (non-disruptive),
        // otherwise the synchronous in-place swap. The active? decision now lives in the
        // driver (on its own hasActiveCells state) rather than being made here.
        const int32_t ret = phyDriver.l1_cell_update_cell_config_caller_runs(cell_reconfig_phy_driver_info);

        if (ret == 0)
        {
            cell_update_success(phyDriver, cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_OK);
        }
        else
        {
            NVLOGW_FMT(TAG, "{}: offloaded cell config update failed cell_id={} phyCellId={}", func, cell_id, cell_update_config.cell_config_.phy_cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_INVALID_CONFIG);
        }

        phyDriver.l1_unlock_update_cell_config_mutex();
        phy_config.prach_config_.start_ro_index = phyDriver.l1_get_prach_start_ro_index(phy_cell_params.phyCellId);
        cell_update_config.prach_config_.start_ro_index = phy_config.prach_config_.start_ro_index;
        NVLOGD_FMT(TAG, "{}: start_ro_index={}", func, cell_update_config.prach_config_.start_ro_index);
    };

    auto process_initial_config = [&]() -> uint8_t {
        if(state == fapi_state_t::FAPI_STATE_RUNNING)
        {
            NVLOGW_FMT(TAG, "{}: CONFIG.req rejected for cell_id={} - FAPI_STATE_RUNNING", func, cell_id);
            return SCF_ERROR_CODE_MSG_INVALID_STATE;
        }

        // Fresh-config TLV decode is shared with the serial path; see decode_config_tlvs().
        // PerCallIndexed: this offload path keeps its per-call, bounds-checked SSB_MASK handling.
        uint8_t error_code = decode_config_tlvs(*config_request, cell_id,
                                                std::span<uint8_t>{static_cast<uint8_t*>(msg.data_buf),
                                                                   (msg.data_len > 0) ? static_cast<std::size_t>(msg.data_len) : 0U},
                                                msg.data_pool, SsbMaskDecode::PerCallIndexed);

        if(error_code == SCF_ERROR_CODE_MSG_OK)
        {
            error_code = create_cell_configs();
            // We should track how many TLVs were valid above, but for now, we just report that any CONFIG request is valid
            NVLOGC_FMT(TAG, "{}: create_cell_configs for cell_id={} phy_cell_id={} returned error_code={}", func, cell_id, phy_config.cell_config_.phy_cell_id, error_code);
        }
        if (error_code == SCF_ERROR_CODE_MSG_OK)
        {
            state = fapi_state_t::FAPI_STATE_CONFIGURED;
        }

        phy_module().create_cell_update_call_back();
        return error_code;
    };

    auto finish_config = [&]() -> bool {
        if (state != fapi_state_t::FAPI_STATE_CONFIGURED)
        {
            return true;
        }

        phy_module().transport_wrapper().set_cell_configured(cell_id);
        auto& config = phy_module().config_options();
        if (config.duplicateConfigAllCells && !first_config_req && phyDriver.driver_exist()) {
            first_config_req = true;
            auto& instances = phy_module().PHY_instances();
            // Publish the configured cell count so print_cell_stats() iterates
            // every fanned-out cell, matching the serial CONFIG path.
            total_cell_num = phyDriver.l1_get_cell_group_num();
            const uint32_t cell_group_num = total_cell_num;
            NVLOGI_FMT(TAG, "{}: duplicateConfigAllCells copying CONFIG from cell_id={} to {} cells",
                       func, cell_id, cell_group_num);
            for (uint32_t i = 0; i < cell_group_num; i++) {
                auto& phy = instances[i];
                if (static_cast<int32_t>(i) == cell_id) {
                    continue;
                }
                auto& instance = reinterpret_cast<scf_5g_fapi::phy&>(phy.get());
                instance.copy_phy_configs_from(phy_config);
                instance.copy_csi2_maps_from(nCsi2Maps, csi2MapCpuBuffer.get(), csi2MapParamsCpuBuffer.get());
                instance.phy_config.cell_config_.phy_cell_id = fake_phy_cell_id[i];
                instance.phy_config.cell_config_.carrier_idx = static_cast<int32_t>(i);
                instance.create_cell_configs();
                instance.copy_precoding_configs_to(static_cast<int32_t>(i));
                // Update the FH
                instance.update_dbt_pdu_table_ptr(instance.phy_config.cell_config_.carrier_idx , this->dbt_pdu_table_ptr);
                // Clear the dbt_pdu_table_ptr for the instance cell - No FH storing
                instance.update_dbt_pdu_table_ptr(instance.phy_config.cell_config_.carrier_idx, nullptr);
                instance.update_cell_state(fapi_state_t::FAPI_STATE_CONFIGURED);
                phy_module().transport_wrapper().set_cell_configured(i);
            }
            update_dbt_pdu_table_ptr(phy_config.cell_config_.carrier_idx, nullptr);
        }

        if (phy_module().transport_wrapper().get_all_cells_configured()) {
            memfoot_global_print_all();

            if (l1_init_cplane_generator(phyDriver.get_driver(),
                                         config.bf_enabled,
                                         config.precoding_enabled) != 0) {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "l1_init_cplane_generator failed");
                return false;
            }
            phy_module().set_all_cells_configured(true);
        }
        return true;
    };

    NVLOGI_FMT(TAG,
               "{}: received offloaded CONFIG.req cell_id={} handle_id={} state={}",
               func, cell_id, +hdr->handle_id, static_cast<uint32_t>(state.load()));
    if (state == fapi_state_t::FAPI_STATE_CONFIGURED)
    {
        process_reconfig();
        return;
    }

    uint8_t response_code = process_initial_config();
    const bool post_ok = response_code == SCF_ERROR_CODE_MSG_OK && finish_config();
    NVLOGI_FMT(TAG,
               "{}: completed offloaded CONFIG.req cell_id={} response_code={} post_ok={} state={}",
               func, cell_id, response_code, post_ok, static_cast<uint32_t>(state.load()));
    if (!post_ok)
    {
        if (response_code == SCF_ERROR_CODE_MSG_OK)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: CONFIG post-processing failed cell_id={}", func, cell_id);
            response_code = SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }
    }
    send_cell_config_response(cell_id, response_code);
}

void phy::on_cell_start_request_offload(int32_t cell_id)
{
    NVLOGI_FMT(TAG,
               "{}: Cell {} received offloaded START.req carrier_id={}",
               __FUNCTION__,
               phy_config.cell_config_.phy_cell_id,
               phy_config.cell_config_.carrier_idx);

    if (state != fapi_state_t::FAPI_STATE_CONFIGURED)
    {
        NVLOGW_FMT(TAG,
                   "{}: offloaded START.req rejected - state not CONFIGURED (current={})",
                   __FUNCTION__,
                   static_cast<uint32_t>(state.load()));
        send_error_indication(SCF_FAPI_START_REQUEST, SCF_ERROR_CODE_MSG_INVALID_STATE, 0, 0);
        return;
    }

    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    phyDriver.l1_cell_start(phy_config.cell_config_.phy_cell_id);
    phy_module().send_call_backs();

    if (get_mMIMO_enable_info())
    {
        bfw_buffer_info buffer_info;
        phyDriver.l1_bfw_coeff_retrieve_buffer(cell_id, &buffer_info);
        phy_module().set_bfw_coeff_buff_info(cell_id, &buffer_info);
    }

    if (phyDriver.l1_staticBFWConfigured(cell_id))
    {
        const int dbt_store_status = phyDriver.l1_resetDBTStorage(cell_id);
        if (dbt_store_status == -1)
        {
            NVLOGE_FMT(TAG,
                       AERIAL_L2ADAPTER_EVENT,
                       "{}: l1_resetDBTStorage failed for cell_id={} - sending error indication",
                       __FUNCTION__,
                       cell_id);
            send_error_indication(SCF_FAPI_START_REQUEST, SCF_ERROR_CODE_BEAM_ID_OUT_OF_RANGE, 0, 0);
            return;
        }
    }

    if (get_enable_srs_info())
    {
        const uint32_t srs_chest_buff_size = getSrsChestBuffSize();
        NVLOGI_FMT(TAG,
                   "{}: SRS chest buffer requested for size={} from offloaded START.req",
                   __FUNCTION__,
                   srs_chest_buff_size);
        const bool ok = phyDriver.allocSrsChesBuffPool(SCF_FAPI_START_REQUEST, cell_id, srs_chest_buff_size);
        if (!ok)
        {
            NVLOGE_FMT(TAG,
                       AERIAL_L2ADAPTER_EVENT,
                       "{}: SRS chest buffer allocation failed size={} - sending error response",
                       __FUNCTION__,
                       srs_chest_buff_size);
            send_error_indication(SCF_FAPI_START_REQUEST, SCF_ERROR_CODE_MSG_INVALID_CONFIG, 0, 0);
            return;
        }
    }

    phy_module().set_tti_flag(true);

    // NOTE: The legacy phy::on_cell_start_request() (scf_5g_fapi_phy.cpp) also
    // calls phy_module().incr_active_cells() at this point to bump the legacy
    // num_cells_active counter. The store-replay/offload path intentionally
    // omits that immediate update: stage_cell_started() below records the
    // transition in the staged bitmap, and the slot-boundary
    // commit_staged_active_cell_bitmap() refreshes both
    // NonSlotDispatchState's committed bitmap and num_cells_active (see
    // nv_phy_non_slot_dispatch.cpp::commit_staged_active_cell_bitmap()).
    // Consumers in this build path should read the active count via
    // PHY_module::get_active_cell_count() (popcount of the committed bitmap)
    // rather than the raw num_cells_active field to stay consistent with the
    // staged-then-committed model; the staleness window is bounded to one slot.
    state = fapi_state_t::FAPI_STATE_RUNNING;
    phy_module().transport(cell_id).set_started_cells_mask(cell_id, true);
#ifdef ENABLE_L2_SLT_RSP
    phy_module().stage_cell_started(static_cast<uint16_t>(phy_config.cell_config_.carrier_idx));
#endif
}

void phy::on_cell_stop_request_offload(int32_t cell_id)
{
    NVLOGI_FMT(TAG,
               "{}: Cell {} received offloaded STOP.req carrier_id={}",
               __FUNCTION__,
               phy_config.cell_config_.phy_cell_id,
               phy_config.cell_config_.carrier_idx);

    if (state != fapi_state_t::FAPI_STATE_RUNNING)
    {
        NVLOGW_FMT(TAG,
                   "{}: offloaded STOP.req rejected - state not RUNNING (current={})",
                   __FUNCTION__,
                   static_cast<uint32_t>(state.load()));
        send_error_indication(SCF_FAPI_STOP_REQUEST, SCF_ERROR_CODE_MSG_INVALID_STATE, 0, 0);
        return;
    }

    // NOTE: The legacy phy::on_cell_stop_request() calls
    // phy_module().decr_active_cells() at this point. The offload path
    // intentionally defers the legacy num_cells_active update to the
    // slot-boundary commit_staged_active_cell_bitmap(). Stage STOP before
    // l1_cell_stop() so the next slot snapshot cannot include a carrier that
    // L1 may already have stopped serving; see the matching note in
    // on_cell_start_request_offload() for full rationale.
    const int32_t stopped_carrier_idx = phy_config.cell_config_.carrier_idx;
#ifdef ENABLE_L2_SLT_RSP
    phy_module().stage_cell_stopped(static_cast<uint16_t>(stopped_carrier_idx));
#endif

    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    phyDriver.l1_cell_stop(phy_config.cell_config_.phy_cell_id);
    phyDriver.deAllocSrsChesBuffPool(cell_id);
    state = fapi_state_t::FAPI_STATE_CONFIGURED;

    nv::phy_mac_transport& transport = phy_module().transport(stopped_carrier_idx);
    nv::phy_mac_msg_desc msg_desc;
    if (transport.tx_alloc(msg_desc) < 0)
    {
        // tx_alloc failure means we cannot deliver STOP.indication to L2, but
        // L1 has already been stopped (l1_cell_stop above) and state is
        // already FAPI_STATE_CONFIGURED. Run the same local-bookkeeping
        // cleanup that the happy path performs at the bottom of this function
        // so PHY's internal model is consistent with the real (stopped) L1
        // state — otherwise tti_flag and started_cells_mask would remain in
        // their pre-stop values and stale
        // DL/UL traffic for this carrier could be processed until the next
        // STOP attempt. L2 will not see a STOP.indication on this failure
        // path; the upstream watchdog/STOP retry must reconcile the L2-side
        // state.
        NVLOGE_FMT(TAG,
                   AERIAL_L2ADAPTER_EVENT,
                   "{}: tx_alloc failed for STOP.indication on carrier_idx={} - "
                   "applying local STOP cleanup without notifying L2",
                   __FUNCTION__,
                   stopped_carrier_idx);
        phy_module().set_tti_flag(false);
        phy_module().transport(cell_id).set_started_cells_mask(cell_id, false);
        return;
    }

    auto* hdr = aerial::casts::assume_cast<scf_fapi_header_t>(msg_desc.msg_buf);
    hdr->message_count = 1;
    hdr->handle_id = stopped_carrier_idx;

    auto* body = aerial::casts::assume_cast<scf_fapi_body_header_t>(hdr->payload);
    body->type_id = SCF_FAPI_STOP_INDICATION;
    body->length = 0;

    msg_desc.msg_id = SCF_FAPI_STOP_INDICATION;
    msg_desc.cell_id = stopped_carrier_idx;
    msg_desc.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
    msg_desc.data_len = 0;
    transport.tx_send(msg_desc);
    transport.notify(IPC_NOTIFY_VALUE);
    metrics_.incr_tx_packet_count(SCF_FAPI_STOP_INDICATION);

    phy_module().set_tti_flag(false);
    phy_module().transport(cell_id).set_started_cells_mask(cell_id, false);
}

} // namespace scf_5g_fapi
