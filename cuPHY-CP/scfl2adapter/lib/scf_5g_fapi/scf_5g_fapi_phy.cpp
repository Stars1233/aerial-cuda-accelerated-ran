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
#include "scf_5g_fapi_rx_msg.hpp"
#include "scf_5g_slot_commands.hpp"
#include "scf_5g_fapi_msg_helpers.hpp"
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "nv_phy_utils.hpp"
#include "nv_phy_fapi_msg_common.hpp"
#include "nv_phy_limit_errors.hpp"
#include "cuphydriver_api.hpp"
#include "memfoot_global.h"
#include <cerrno>
#include <functional>

#define TAG (NVLOG_TAG_BASE_SCF_L2_ADAPTER + 3) // "SCF.PHY"
namespace scf_5g_fapi
{

uint32_t total_cell_num = 0;
std::atomic<uint64_t> ul_thrput[MAX_CELLS_PER_SLOT];
std::atomic<uint64_t> dl_thrput[MAX_CELLS_PER_SLOT];
std::atomic<uint32_t> ul_slot[MAX_CELLS_PER_SLOT];
std::atomic<uint32_t> dl_slot[MAX_CELLS_PER_SLOT];
std::atomic<uint32_t> ul_crc_err[MAX_CELLS_PER_SLOT];
std::atomic<uint32_t> ul_crc_err_total[MAX_CELLS_PER_SLOT];

inline void reset_cell_stats(uint32_t cell_id)
{
    ul_thrput[cell_id] = 0;
    dl_thrput[cell_id] = 0;
    ul_slot[cell_id] = 0;
    dl_slot[cell_id] = 0;
    ul_crc_err[cell_id] = 0;
}
 
void phy::print_cell_stats(slot_command_api::slot_indication* slot_3gpp)
{
    static constexpr double BYTES_TO_MBPS_FACTOR = 8.0 / 1000000.0;
    
    for (int cell_id = 0; cell_id < total_cell_num; cell_id++) {
        const double dl_mbps = static_cast<double>(dl_thrput[cell_id]) * BYTES_TO_MBPS_FACTOR;
        const double ul_mbps = static_cast<double>(ul_thrput[cell_id]) * BYTES_TO_MBPS_FACTOR;
        const uint32_t dl_slots = dl_slot[cell_id].load();
        const uint32_t ul_slots = ul_slot[cell_id].load();
        const uint32_t crc_err = ul_crc_err[cell_id].load();
        const uint32_t crc_err_total = ul_crc_err_total[cell_id].load();
        
        if (slot_3gpp) {
            NVLOGC_FMT(TAG, "Cell {:2d} | DL {:7.2f} Mbps {:4d} Slots | UL {:7.2f} Mbps {:4d} Slots CRC {:3d} ({:6d}) | Tick {}", 
                    cell_id, dl_mbps, dl_slots, ul_mbps, ul_slots, crc_err, crc_err_total, slot_3gpp->tick_);
            reset_cell_stats(cell_id);
        }
        else {
            NVLOGC_FMT(TAG, "Cell {:2d} | DL {:7.2f} Mbps {:4d} Slots | UL {:7.2f} Mbps {:4d} Slots CRC {:3d} ({:6d})", 
                    cell_id, dl_mbps, dl_slots, ul_mbps, ul_slots, crc_err, crc_err_total);
        }
    }
}

bool phy::first_config_req = false;
static constexpr uint16_t fake_phy_cell_id[MAX_CELLS_PER_SLOT] = {1008, 1009, 1010, 1011, 1012, 1013,
                                                                  1014, 1015, 1016, 1017, 1018, 1019,
                                                                  1020, 1021, 1022, 1023, 1024, 1025,
                                                                  1026, 1027};
std::vector<uint32_t> phy::first_config_req_pmidxes = {};

phy::phy(nv::PHY_module& phy_module, yaml::node node_config) :
    nv::PHY_instance(phy_module, node_config),
    phy_cell_params(),
    cell_update_config(),
    cell_reconfig_phy_cell_params{0},
    prach_addln_config({0}),
    ssb_case(nv::ssb_case::CASE_UNKNOWN),
    lmax_symbol_list(nullptr),
    prach_ta_offset_usec_(0.0),
    non_prach_ta_offset_usec_(0.0),
    //pf_01_interference(0),
    pusch_cell_stat_params{0},
    pucch_cell_stat_params{0},
    csi2MapCpuBuffer(cuphy::make_unique_pinned<uint16_t>(CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL)),
    csi2MapParamsCpuBuffer(cuphy::make_unique_pinned<cuphyCsi2MapPrm_t>(CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL)),
    nCsi2Maps(0)
{
    l_max = 0;
    beta = 0;
    beta_sq = 0;
    fs_offset_ul = 0;
    ul_bitwidth = 0;
    dl_pdu_index_size = 0;
    allowed_fapi_latency = 0;

    pdsch_cw_idx_start = 0;

    tx_data_req_meta_data_.num_pdus = 0;
    tx_data_req_meta_data_.data = nullptr;
    tx_data_req_meta_data_.buf = nullptr;

#ifndef ENABLE_L2_SLT_RSP
    allowed_fapi_latency = phy_module.get_allowed_fapi_latency();
    NVLOGI_FMT(TAG, "scf_5g_fapi::phy::phy(): allowed_fapi_latency={}", allowed_fapi_latency);
#endif
    if (node_config.has_key("prach_ta_offset_usec")) {
        prach_ta_offset_usec_ = node_config["prach_ta_offset_usec"].as<float>();
    }

    if (node_config.has_key("non_prach_ta_offset_usec")) {
        non_prach_ta_offset_usec_ = node_config["non_prach_ta_offset_usec"].as<float>();
    }

    cell_stat_prm_idx = INVALID_CELL_CFG_IDX;
    cell_reconfig_phy_cell_params.pPuschCellStatPrms = nullptr;
    cell_reconfig_phy_cell_params.pPucchCellStatPrms = nullptr;
}
 
phy::~phy() {

}

void phy::reset_slot(bool partial_cmd)
{
    if (cur_dl_msg.data_buf != nullptr)
    {
        scf_fapi_header_t* fapi_hdr = reinterpret_cast<scf_fapi_header_t*>(cur_dl_msg.msg_buf);
        scf_fapi_tx_data_req_t* req = reinterpret_cast<scf_fapi_tx_data_req_t*>(fapi_hdr->payload);

        if(partial_cmd)
            NVLOGI_FMT(TAG, "{}: TX_DATA.request not released: SFN {}.{} cell_id={} due to partial command. Releasing now", __FUNCTION__, static_cast<unsigned>(req->sfn), static_cast<unsigned>(req->slot), cur_dl_msg.cell_id);
        else
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: TX_DATA.request not released: SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(req->sfn), static_cast<unsigned>(req->slot), cur_dl_msg.cell_id);
        phy_module().transport_wrapper().rx_release(cur_dl_msg);
    }

    cur_dl_msg.reset();
    dl_pdu_index_size = 0;
    pdsch_rejected_ = false;

    // Reset duplicate message tracking flags for the new slot
    duplicate_dl_tti_req = false;
    duplicate_ul_tti_req = false;
    duplicate_tx_data_req = false;
    duplicate_ul_dci_req = false;
    duplicate_dl_bfw_cvi_req = false;
    duplicate_ul_bfw_cvi_req = false;
}

// Reset PHY state and clean up resources when L2 reconnects
int phy::reset()
{
    int32_t cell_id = phy_config.cell_config_.carrier_idx;
    nv::phy_mac_transport& transport = phy_module().transport(cell_id);

    NVLOGC_FMT(TAG, "{}: PHY reset initiated: transport_id={} phy_cell_id={} mac_cell_id={} current_state={}", __FUNCTION__, transport.get_transport_id(), cell_id, transport.get_mac_cell_id(cell_id), static_cast<uint32_t>(state.load()));

    // Stop RUNNING state cell and move to CONFIGURED state. For other states, just keep the state as is.
    // Use atomic CAS operation to compare state and exchange if matches.
    fapi_state_t expected_state = fapi_state_t::FAPI_STATE_RUNNING;
    if (state.compare_exchange_strong(expected_state, fapi_state_t::FAPI_STATE_CONFIGURED))
    {
        // Stop cell in PHYDriver for proper cleanup before reset
        nv::PHYDriverProxy::getInstance().l1_cell_stop(phy_config.cell_config_.phy_cell_id);
        nv::PHYDriverProxy::getInstance().deAllocSrsChesBuffPool(cell_id);

        // Decrement active cell count for running cells
        phy_module().set_tti_flag(false);
        phy_module().decr_active_cells();
        transport.set_started_cells_mask(cell_id, false);
#ifdef ENABLE_L2_SLT_RSP
        phy_module().unset_active_cell_bitmap(cell_id);
#endif
    }

    // Reset slot data and duplicate message tracking flags
    reset_slot(false);

    return 0;
}

#ifdef ENABLE_L2_SLT_RSP
int phy::check_sfn_slot(int cell_id, int msg_id, sfn_slot_t ss_msg)
{
    sfn_slot_t& ss_curr = phy_module().get_curr_sfn_slot();

    if ((msg_id != SCF_FAPI_SLOT_INDICATION) && (ss_curr.u32 != ss_msg.u32))
    {
        // SFN/SLOT changed: process pending messages from previous slot
        NVLOGW_FMT(TAG, "{}: SFN mismatch cell_id={} expected={}.{} received={}.{} msg_id=0x{:02X} dropped", __FUNCTION__, 
                   cell_id, ss_curr.u16.sfn, ss_curr.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot, msg_id);

        switch(msg_id)
        {
            case SCF_FAPI_DL_TTI_REQUEST:
            case SCF_FAPI_UL_TTI_REQUEST:
                send_error_indication(static_cast<scf_fapi_message_id_e>(msg_id), SCF_ERROR_CODE_SFN_OUT_OF_SYNC, ss_msg.u16.sfn, ss_msg.u16.slot);
                break;
            case SCF_FAPI_UL_DCI_REQUEST:
            case SCF_FAPI_TX_DATA_REQUEST:
                send_error_indication(static_cast<scf_fapi_message_id_e>(msg_id), SCF_ERROR_CODE_MSG_INVALID_SFN, ss_msg.u16.sfn, ss_msg.u16.slot);
                break;
            default:
                break;
        }
        return -1;
    }
    else
        return 0;
}
#else
int phy::check_sfn_slot(int cell_id, int msg_id, sfn_slot_t ss_msg)
{
    sfn_slot_t& ss_curr = phy_module().get_curr_sfn_slot();
    sfn_slot_t& ss_last = phy_module().get_last_sfn_slot();
    uint32_t fapi_latency = phy_module().get_fapi_latency(ss_msg);

    if (ss_curr.u32 != ss_msg.u32 && ss_curr.u32 != ss_last.u32)
    {
        // SFN/SLOT changed, process remaining messages of previous slot
        NVLOGW_FMT(TAG, "{}: SFN mismatch cell_id={} current={}.{} incoming={}.{} msg_id=0x{:02X} latency={} - processing pending commands", __FUNCTION__,
                   cell_id, ss_curr.u16.sfn, ss_curr.u16.slot, ss_msg.u16.sfn, ss_msg.u16.slot, msg_id, fapi_latency);
        phy_module().process_phy_commands(true);
    }

    ss_curr.u32 = ss_msg.u32;
    if (fapi_latency <= allowed_fapi_latency)
    {
        NVLOGI_FMT(TAG, "{}: Processing FAPI msg SFN {}.{} cell_id={} msg_id=0x{:02X} latency={}", __FUNCTION__,
                static_cast<unsigned>(ss_msg.u16.sfn), static_cast<unsigned>(ss_msg.u16.slot), cell_id, msg_id, fapi_latency);
        return 0;
    }
    else
    {
        NVLOGW_FMT(TAG, "{}: FAPI msg dropped due to latency SFN {}.{} cell_id={} msg_id=0x{:02X} latency={}", __FUNCTION__,
                ss_msg.u16.sfn, ss_msg.u16.slot, cell_id, msg_id, fapi_latency);

        switch(msg_id)
        {
            case SCF_FAPI_DL_TTI_REQUEST:
            case SCF_FAPI_UL_TTI_REQUEST:
                send_error_indication(static_cast<scf_fapi_message_id_e>(msg_id), SCF_ERROR_CODE_SFN_OUT_OF_SYNC, ss_msg.u16.sfn, ss_msg.u16.slot);
                break;
            case SCF_FAPI_UL_DCI_REQUEST:
            case SCF_FAPI_TX_DATA_REQUEST:
                send_error_indication(static_cast<scf_fapi_message_id_e>(msg_id), SCF_ERROR_CODE_MSG_INVALID_SFN, ss_msg.u16.sfn, ss_msg.u16.slot);
                break;
            default:
                break;
        }
        return -1;
    }
}
#endif

inline void update_l1_recovery_cnts(sfn_slot_t& ss_msg) {
    if(false == nv::PHYDriverProxy::getInstance().l1_incr_recovery_slots())
    {
        EXIT_L1(EXIT_FAILURE);
        return;
    }

    if(nv::PHYDriverProxy::getInstance().l1_get_aggr_obj_free_status())
    {
        if(nv::PHYDriverProxy::getInstance().l1_incr_all_obj_free_slots())
        {
            NVLOGW_FMT(TAG, "{}: SFN {}.{} Transition L1 to running", __FUNCTION__, ss_msg.u16.sfn, ss_msg.u16.slot);
            nv::PHYDriverProxy::getInstance().l1_reset_recovery_slots();
            pExitHandler.set_exit_handler_flag(exit_handler::l1_state::L1_RUNNING);
        }
    }
    else
        nv::PHYDriverProxy::getInstance().l1_reset_all_obj_free_slots();
    }
 
bool phy::on_msg(nv_ipc_msg_t& msg)
{
    bool ready_to_free = true;
    bool data_buf = (msg.data_pool ==  NV_IPC_MEMPOOL_CPU_DATA) &&
                    (phy_module().dl_tb_location() == nv::dl_tb_loc::TB_LOC_INLINE);
    
    // This array indicates invalid PDSCH PDUs for a given cell.
    uint8_t pdsch_pdu_valid_flag[MAX_CELLS_PER_SLOT] = {0};
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    rx_msg_reader reader(msg);
    for(rx_msg_reader::iterator it = reader.begin(); it != reader.end(); ++it)
    {
        scf_fapi_body_header_t& body_hdr = (*it);
        uint8_t                 typeID   = body_hdr.type_id;

        // Check whether the msg_len is set correctly. Note: there's only 1 FAPI message in the iterator
        uint32_t head_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
        uint32_t body_len = body_hdr.length;
        if (msg.msg_len != head_len + body_len)
        {
            NVLOGW_FMT(TAG, "{}: Incorrect msg length cell_id={} msg_id=0x{:02X} expected={} received={}", __FUNCTION__,
                    msg.cell_id, msg.msg_id, head_len + body_len, msg.msg_len);
        }

        if(state != fapi_state_t::FAPI_STATE_IDLE)
        {
            metrics_.incr_rx_packet_count(static_cast<scf_fapi_message_id_e>(typeID));
        }

        sfn_slot_t& ss_msg = *(reinterpret_cast<sfn_slot_t*>(body_hdr.data));
        NVLOGI_FMT(TAG, "{}: SFN {}.{} received: cell_id={} msg_id=0x{:02X}", __FUNCTION__, ss_msg.u16.sfn, ss_msg.u16.slot, msg.cell_id, msg.msg_id);

        scf_fapi_header_t *hdr = reinterpret_cast<scf_fapi_header_t*>(msg.msg_buf);
        if(hdr->handle_id != msg.cell_id || msg.cell_id < 0 || msg.cell_id >= MAX_CELLS_PER_SLOT)
        {
            NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: Incorrect cell_id={} msg_id=0x{:02X} handle_id={} pool={}", __FUNCTION__, msg.cell_id, msg.msg_id, hdr->handle_id, msg.data_pool);
            return ready_to_free;
        }

        // Validate and process SFN/SLOT for slot messages
#ifdef ENABLE_L2_SLT_RSP
        if (typeID > SCF_FAPI_RESV_1_END)
#else
        if (typeID > SCF_FAPI_RESV_1_END && typeID < SCF_FAPI_RESV_2_START)
#endif
        {
            if(state == fapi_state_t::FAPI_STATE_RUNNING && check_sfn_slot(msg.cell_id, typeID, ss_msg) < 0)
            {
#ifndef ENABLE_L2_SLT_RSP
                // Force reset to prevent duplicate SFN/SLOT
                auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
                slot_command_api::slot_indication slot_ind;
                sfn_slot_t& ss_curr = phy_module().get_curr_sfn_slot();
                slot_ind.sfn_ = ss_curr.u16.sfn;
                slot_ind.slot_ = ss_curr.u16.slot;
                reset_cell_command(slot_cmd, slot_ind, get_carrier_id(),  phy_module().cell_group(), phy_module().group_command());
#endif
                continue;
            }
        }

        if(pExitHandler.get_l1_state() != exit_handler::l1_state::L1_RUNNING) {

            if(pExitHandler.get_l1_state() == exit_handler::l1_state::L1_EXIT || typeID != SCF_FAPI_SLOT_INDICATION) {
                    NVLOGW_FMT(TAG, "{}: SFN {}.{} received: cell_id={} msg_id=0x{:02X}: L1 in recovery, drop msg", __FUNCTION__,
                ss_msg.u16.sfn, ss_msg.u16.slot, msg.cell_id, msg.msg_id);

                return ready_to_free;
            }

            update_l1_recovery_cnts(ss_msg);
        }


        switch(typeID)
        {
            case SCF_FAPI_CONFIG_REQUEST:
            {
                scf_fapi_header_t &hdr = reader.header();
                on_config_request(reinterpret_cast<scf_fapi_config_request_msg_t&>(body_hdr), msg.cell_id, hdr.handle_id, msg);
                if (state == fapi_state_t::FAPI_STATE_IDLE)
                {
                    // Return if configuration failed or already in running state
                    return ready_to_free;
                }

                phy_module().transport_wrapper().set_cell_configured(msg.cell_id);

                // Duplicate config for all other cells if duplicateConfigAllCells is true
                auto& config = phy_module().config_options();
                if (config.duplicateConfigAllCells && !first_config_req && phyDriver.driver_exist()) {
                    first_config_req = true;
                    auto& instances =  phy_module().PHY_instances();
                    total_cell_num = phyDriver.l1_get_cell_group_num();
                    for (auto i = 0; i < total_cell_num; i++) {
                        auto& phy =  instances[i];
                        if (i == msg.cell_id) {
                            continue;
                        } else {
                            auto& instance = reinterpret_cast<scf_5g_fapi::phy&>(phy.get());
                            instance.copy_phy_configs_from(phy_config);
                            instance.copy_csi2_maps_from(nCsi2Maps, csi2MapCpuBuffer.get(), csi2MapParamsCpuBuffer.get());
                            instance.phy_config.cell_config_.phy_cell_id = fake_phy_cell_id[i];
                            instance.phy_config.cell_config_.carrier_idx = i;
                            instance.create_cell_configs();
                            instance.copy_precoding_configs_to(i);
                            instance.update_dbt_pdu_table_ptr(instance.phy_config.cell_config_.carrier_idx , this->dbt_pdu_table_ptr); // Update the FH
                            instance.update_dbt_pdu_table_ptr(instance.phy_config.cell_config_.carrier_idx, nullptr); // Clear the dbt_pdu_table_ptr for the instance cell - No FH storing
                            instance.update_cell_state(fapi_state_t::FAPI_STATE_CONFIGURED);
                            phy_module().transport_wrapper().set_cell_configured(i);
                        }
                    }
                    update_dbt_pdu_table_ptr(phy_config.cell_config_.carrier_idx, nullptr);
                }

                if (phy_module().transport_wrapper().get_all_cells_configured()) {
                    phy_module().set_all_cells_configured(true);
                    // Print memory info after all cells are configured
                    memfoot_global_print_all();
                }
            }
                break;
            case SCF_FAPI_START_REQUEST:
                on_cell_start_request(msg.cell_id);
                break;
            case SCF_FAPI_STOP_REQUEST:
                on_cell_stop_request(msg.cell_id);
                break;
            case SCF_FAPI_PARAM_REQUEST:
                on_param_request();
                break;
            case SCF_FAPI_DL_TTI_REQUEST:
            {
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                phy_module().is_dl_slot(true);
                if(phy_module().new_slot() == true)
                {
                    phy_module().new_slot(false);
                    phy_module().l2a_start_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                }
                on_dl_tti_request(reinterpret_cast<scf_fapi_dl_tti_req_t&>(body_hdr), msg, pdsch_pdu_valid_flag);
#ifdef ENABLE_L2_SLT_RSP
                auto& cell_error = phy_module().get_cell_limit_errors(get_carrier_id());
                auto& group_error = phy_module().get_group_limit_errors();
                auto [total_errors, error_mask] = check_dl_tti_l1_limit_errors(cell_error, group_error);
                if(total_errors > 0)
                {
                    send_error_indication(static_cast<scf_fapi_message_id_e>(typeID), static_cast<scf_fapi_error_codes_t>(error_mask), ss_msg.u16.sfn, ss_msg.u16.slot, false, total_errors, &cell_error, &group_error);
                    duplicate_dl_tti_req = false;
                }
#endif
            }
                break;
            case SCF_FAPI_UL_TTI_REQUEST:
            {
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                phy_module().is_ul_slot(true);
                if(phy_module().new_slot() == true)
                {
                    phy_module().new_slot(false);
                    phy_module().l2a_start_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                }
                on_ul_tti_request(reinterpret_cast<scf_fapi_ul_tti_req_t&>(body_hdr), msg);
#ifdef ENABLE_L2_SLT_RSP
                auto& cell_error = phy_module().get_cell_limit_errors(get_carrier_id());
                auto& group_error = phy_module().get_group_limit_errors();
                auto [total_errors, error_mask] = check_ul_tti_l1_limit_errors(cell_error, group_error);
                if(total_errors > 0)
                {
                    send_error_indication(static_cast<scf_fapi_message_id_e>(typeID), static_cast<scf_fapi_error_codes_t>(error_mask), ss_msg.u16.sfn, ss_msg.u16.slot);
                }
#endif
            }
            break;
            case SCF_FAPI_UL_DCI_REQUEST:
            {
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                phy_module().is_dl_slot(true);
                if(phy_module().new_slot() == true)
                {
                    phy_module().new_slot(false);
                    phy_module().l2a_start_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                }
                on_ul_dci_request(reinterpret_cast<scf_fapi_ul_dci_t&>(body_hdr), msg);
#ifdef ENABLE_L2_SLT_RSP
                auto& cell_error = phy_module().get_cell_limit_errors(get_carrier_id());
                auto [total_errors, error_mask] = check_ul_dci_l1_limit(cell_error);
                if(total_errors > 0)
                {
                    send_error_indication(static_cast<scf_fapi_message_id_e>(typeID),  static_cast<scf_fapi_error_codes_t>(error_mask), ss_msg.u16.sfn, ss_msg.u16.slot);
                }
#endif
            }
            break;
            case SCF_FAPI_TX_DATA_REQUEST:
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                if (on_phy_dl_tx_request(reinterpret_cast<scf_fapi_tx_data_req_t&>(body_hdr), msg, pdsch_pdu_valid_flag))
                {
                    // Reserve nvipc buffer since data_buf is in use
                    ready_to_free = false;
                }
                break;
#ifdef SCF_FAPI_10_04
            /* Handle Downlink Beamforming Weights and Channel Vector Information (CVI) request from L2 which
             * is received one slot before the corresponding PDSCH_PDU in DL_TTI.request.
             * This message is used in massive MIMO scenarios to instruct L1 to calculate beamforming weights
             * for downlink transmission. Requires both SRS and mMIMO feature flags to be enabled in cuphycontroller_config_xxx.yaml. */
            case SCF_FAPI_DL_BFW_CVI_REQUEST:
            {
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                phy_module().is_dl_slot(true);
                if(phy_module().new_slot() == true)
                {
                    phy_module().new_slot(false);
                    phy_module().l2a_start_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                }
                if(get_enable_srs_info() && get_mMIMO_enable_info())
                {
                    on_dl_bfw_request(reinterpret_cast<scf_fapi_dl_bfw_cvi_request_t&>(body_hdr), msg);
                }
                else
                {
                    send_error_indication(static_cast<scf_fapi_message_id_e>(typeID), SCF_ERROR_CODE_MSG_SLOT_ERR, ss_msg.u16.sfn, ss_msg.u16.slot);
                }
            }
            break;
            /* Handle Uplink Beamforming Weights and Channel Vector Information (CVI) request from L2 which
             * is received one slot before the corresponding PUSCH_PDU in UL_TTI.request.
             * This message is used in massive MIMO scenarios to instruct L1 to calculate beamforming weights
             * for uplink transmission. Requires both SRS and mMIMO feature flags to be enabled in cuphycontroller_config_xxx.yaml. */
            case SCF_FAPI_UL_BFW_CVI_REQUEST:
            {
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                phy_module().is_ul_slot(true);
                if(phy_module().new_slot() == true)
                {
                    phy_module().new_slot(false);
                    phy_module().l2a_start_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                }
                if(get_enable_srs_info() && get_mMIMO_enable_info())
                {
                    on_ul_bfw_request(reinterpret_cast<scf_fapi_ul_bfw_cvi_request_t&>(body_hdr), msg);
                }
                else
                {
                    send_error_indication(static_cast<scf_fapi_message_id_e>(typeID), SCF_ERROR_CODE_MSG_SLOT_ERR, ss_msg.u16.sfn, ss_msg.u16.slot);
                }
            }
            break;
#endif
#ifdef ENABLE_L2_SLT_RSP
            case SCF_FAPI_SLOT_INDICATION:
            {
                auto fapi = reinterpret_cast<scf_fapi_slot_ind_t&>(body_hdr);
                NVLOGI_FMT(TAG, "{}: Slot indication received SFN {}.{}", __FUNCTION__, static_cast<int>(fapi.sfn), static_cast<int>(fapi.slot));
                sfn_slot_t ss_curr; ss_curr.u16.sfn = fapi.sfn; ss_curr.u16.slot = fapi.slot;
                phy_module().set_curr_sfn_slot(ss_curr);
                //phy_module().l1_slot_ind_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                break;
            }
            case SCF_FAPI_SLOT_RESPONSE:
            {
                phy_module().last_fapi_msg_tick(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()));
                auto fapi = reinterpret_cast<scf_fapi_slot_rsp_t&>(body_hdr);
                NVLOGI_FMT(TAG, "{}: Slot response received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<int>(fapi.sfn), static_cast<int>(fapi.slot), msg.cell_id);
                sfn_slot_t& ss_curr = phy_module().get_curr_sfn_slot();
                if(fapi.sfn == ss_curr.u16.sfn && fapi.slot == ss_curr.u16.slot)
                    phy_module().update_eom_rcvd_bitmap(msg.cell_id);
                break;
            }
            case SCF_FAPI_ERROR_INDICATION:
            {
                auto fapi = reinterpret_cast<scf_fapi_error_ind_t&>(body_hdr);
                NVLOGI_FMT(TAG, "{}: Late slot error SFN {}.{}", __FUNCTION__, static_cast<int>(fapi.sfn), static_cast<int>(fapi.slot));
                sfn_slot_t ss_curr; ss_curr.u16.sfn = 1024; ss_curr.u16.slot = nv::mu_to_slot_in_sf(phy_cell_params.mu);
                phy_module().set_curr_sfn_slot(ss_curr);
                NVLOGI_FMT(TAG, "{}: Set curr SFN=1024 slot={}, clear slot command and keep dropping FAPI messages till next slot indication", __FUNCTION__,
                    ss_curr.u16.slot);
                on_slot_error_indication(reinterpret_cast<scf_fapi_error_ind_t&>(body_hdr), msg);
                break;
            }
#endif
            case CV_MEM_BANK_CONFIG_REQUEST:
            {
                on_cv_mem_bank_config_request(reinterpret_cast<cv_mem_bank_config_request_body_t *>(body_hdr.data), msg.cell_id, msg);
                break;
            }
            default:
                NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}:Error unknown FAPI message: cell_id={} msg_id=0x{:02X} type_id=0x{:02X}", __func__, msg.cell_id, msg.msg_id, typeID);
                break;
        }
    }
    return ready_to_free;
}

void phy::send_slot_error_indication(slot_command_api::slot_indication& slot_3gpp)
{
    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Late slot error encountered for SFN {}.{}", __FUNCTION__, slot_3gpp.sfn_, slot_3gpp.slot_);

    // Broadcast error indication to all transport instances
    for (nv::phy_mac_transport* ptransport : phy_module().transport_wrapper().get_transports())
    {
        nv::phy_mac_transport& transport = *ptransport;
        if (!transport.is_started()) {
            continue;
        }
        int32_t cell_id = transport.get_phy_cell_id(0);
        send_error_indication_l1(SCF_FAPI_SLOT_INDICATION, SCF_ERROR_CODE_MSG_LATE_SLOT_ERR, slot_3gpp.sfn_, slot_3gpp.slot_, cell_id);
    }
#ifdef ENABLE_L2_SLT_RSP
    // TODO: remove tx_send_loopback for SHM
    nv::phy_mac_transport* loopback_transport = phy_module().transport_wrapper().get_loopback_transport();
    if(loopback_transport == nullptr)
    {
        return;
    }
    if (loopback_transport->get_nv_ipc_config()->ipc_transport == NV_IPC_TRANSPORT_SHM) {
        nv::phy_mac_msg_desc msg_desc;
        if(loopback_transport->tx_alloc(msg_desc) < 0)
        {
            return;
        }
        auto fapi_loopback = add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, 0, false);

        auto err_ind_loopback = reinterpret_cast<scf_fapi_error_ind_t*>(fapi_loopback);
        err_ind_loopback->sfn = slot_3gpp.sfn_;
        err_ind_loopback->slot = slot_3gpp.slot_;
        err_ind_loopback->msg_id = SCF_FAPI_SLOT_INDICATION;
        err_ind_loopback->err_code = SCF_ERROR_CODE_MSG_LATE_SLOT_ERR;
        NVLOGI_FMT(TAG, "{}: Loopback: Send error indication for late slot msg_id=0x{:02X} err_code=0x{:02X}", __FUNCTION__, err_ind_loopback->msg_id, err_ind_loopback->err_code);
        loopback_transport->tx_send_loopback(msg_desc);
    } else {
        NVLOGI_FMT(TAG, "Late slot error encountered SFN {}.{}", static_cast<int>(slot_3gpp.sfn_), static_cast<int>(slot_3gpp.slot_));
        sfn_slot_t ss_curr = { .u16 = { .sfn = 1024, .slot = nv::mu_to_slot_in_sf(phy_cell_params.mu) } };
        phy_module().set_curr_sfn_slot(ss_curr);
        NVLOGI_FMT(TAG, "{}: Set curr SFN=1024 slot={}, clear slot command and keep dropping FAPI messages till next slot indication", __FUNCTION__,
            ss_curr.u16.slot);
        nv::phy_mac_msg_desc msg_desc;
        scf_fapi_error_ind_t err_rsp;
        err_rsp.sfn = slot_3gpp.sfn_;
        err_rsp.slot = slot_3gpp.slot_;
        err_rsp.msg_id = SCF_FAPI_SLOT_INDICATION;
        err_rsp.err_code = SCF_ERROR_CODE_MSG_LATE_SLOT_ERR;
        on_slot_error_indication(err_rsp, msg_desc);
    }
#endif
    metrics_.incr_tx_packet_count(SCF_FAPI_ERROR_INDICATION);
}

void phy::send_slot_indication(slot_command_api::slot_indication& slot_3gpp)
{
    // pExitHandler.test_exit_in_flight() is checked in nv_phy_module.cpp
#ifdef ENABLE_L2_SLT_RSP
    // TODO: remove tx_send_loopback for SHM
    nv::phy_mac_transport* loopback_transport = phy_module().transport_wrapper().get_loopback_transport();
    if(loopback_transport == nullptr)
    {
        return;
    }
    if (loopback_transport->get_nv_ipc_config()->ipc_transport == NV_IPC_TRANSPORT_SHM) {
        nv::phy_mac_msg_desc msg_desc;
        if(loopback_transport->tx_alloc(msg_desc) < 0)
        {
            return;
        }
        auto fapi_loopback = add_scf_fapi_hdr<scf_fapi_slot_ind_t>(msg_desc, SCF_FAPI_SLOT_INDICATION, 0, false);

        auto& slot_ind_loopback = *reinterpret_cast<scf_fapi_slot_ind_t*>(fapi_loopback);
        slot_ind_loopback.sfn = slot_3gpp.sfn_;
        slot_ind_loopback.slot = slot_3gpp.slot_;
        loopback_transport->tx_send_loopback(msg_desc);
    } else {
        sfn_slot_t ss_curr = { .u16 = { .sfn = slot_3gpp.sfn_, .slot = slot_3gpp.slot_ } };
        phy_module().set_curr_sfn_slot(ss_curr);
    }
#endif

    // Broadcast slot indication to all transport instances
    for (nv::phy_mac_transport* ptransport : phy_module().transport_wrapper().get_transports())
    {
        nv::phy_mac_transport& transport = *ptransport;
        if (!transport.is_started()) {
            continue;
        }
        nv::phy_mac_msg_desc msg_desc;
        if(transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }
        int32_t cell_id = transport.get_phy_cell_id(0);
        auto fapi = add_scf_fapi_hdr<scf_fapi_slot_ind_t>(msg_desc, SCF_FAPI_SLOT_INDICATION, cell_id, false);
        auto& slot_ind = *reinterpret_cast<scf_fapi_slot_ind_t*>(fapi);
        slot_ind.sfn = slot_3gpp.sfn_;
        slot_ind.slot = slot_3gpp.slot_;
        NVLOGI_FMT(TAG, "{}: Send slot indication SFN {}.{}", __FUNCTION__, static_cast<int>(slot_ind.sfn), static_cast<int>(slot_ind.slot));
        // Send the message over the transport
        transport.tx_send(msg_desc);
        transport.notify(1);
    }

    metrics_.incr_tx_packet_count(SCF_FAPI_SLOT_INDICATION);

    // Print DL/UL throughput statistics
    if (slot_3gpp.tick_ % 2000 == 0) {
        print_cell_stats(&slot_3gpp);
    }
}

bool phy::process_dl_tx_request()
{
    uint32_t offset   = 0;
    bool     handled  = false;
    bool     isDlSlot = false;

    if(tx_data_req_meta_data_.num_pdus == 0 || tx_data_req_meta_data_.num_pdus != dl_pdu_index_size)
    {
        if(tx_data_req_meta_data_.num_pdus != 0)
        {
            NVLOGW_FMT(TAG, "{}: PDU count mismatch DL_TTI={} TX_DATA={}", __FUNCTION__, dl_pdu_index_size, tx_data_req_meta_data_.num_pdus);
        }
        dl_pdu_index_size = 0;
        return handled;
    }
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    uint8_t* data          = tx_data_req_meta_data_.data;
    uint32_t cellPrmDynIdx = 0;
    cell_group_command* group_cmd = phy_module().group_command();
    pdsch_params*       info      = group_cmd->get_pdsch_params();
    if(info == NULL)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PDSCH params not set pdsch_params={} nCells={}", __FUNCTION__, reinterpret_cast<void*>(info), info == nullptr ? -1 : info->cell_grp_info.nCells);
        return handled;
    }

    for(uint32_t i = 0; i < info->cell_grp_info.nCells; i++)
    {
        if(phy_module().get_cell_id_from_stat_prm_idx(info->cell_dyn_info[i].cellPrmStatIdx) == get_carrier_id())
        {
            cellPrmDynIdx = info->cell_dyn_info[i].cellPrmDynIdx;
            NVLOGD_FMT(TAG, "{}: PHY cell_id={} cellPrmDynIdx={}", __FUNCTION__, get_carrier_id(), cellPrmDynIdx);
            break;
        }
    }

    int32_t buffLoc = phy_module().dl_tb_location();

    for(uint16_t i = 0; i < tx_data_req_meta_data_.num_pdus; i++)
    {
        auto& dl_pdu = *(reinterpret_cast<scf_fapi_tx_data_pdu_info_t*>(data + offset));
        if(dl_pdu.pdu_len == 0 || dl_pdu.num_tlv == 0)
        {
            NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: Incorrect PDU length or number of TLVs: pdu_len={} num_tlv={}", __FUNCTION__, static_cast<unsigned>(dl_pdu.pdu_len), static_cast<unsigned>(dl_pdu.num_tlv));
            continue;
        }

        if(dl_pdu_index_size <= dl_pdu.pdu_index)
        {
            NVLOGW_FMT(TAG, "{}: DL PDU index mismatch: size={}, dl_pdu.nPduIdx={}", __FUNCTION__, dl_pdu_index_size, static_cast<int>(dl_pdu.pdu_index));
            continue;
        }

        auto pduType = dl_pdu_index[dl_pdu.pdu_index];
        NVLOGD_FMT(TAG, "{}: DL PDU index size={}, dl_pdu.nPduIdx={} pduType {}", __FUNCTION__, dl_pdu_index_size, static_cast<int>(dl_pdu.pdu_index), static_cast<int>(pduType));

        auto&   tlv_info = *(reinterpret_cast<scf_fapi_tl_t*>(&dl_pdu.tlvs[0]));
        uint32_t pdu_offset;
        auto& ue_cw = info->ue_cw_info[pdsch_cw_idx_start + i];
        switch(pduType)
        {
            case DL_TTI_PDU_TYPE_PDSCH: {
                void*  buf         = nullptr;
                size_t ttl_pdu_len = dl_pdu.pdu_len;

                //Multiple UE per TTI support is present only for tag value=2 (SCF_TX_DATA_OFFSET)
                switch(tlv_info.tag)
                {
                    case SCF_TX_DATA_INLINE_PAYLOAD:
                        buf = reinterpret_cast<decltype(buf)>(&tlv_info.val[0]);
                        break;
                    case SCF_TX_DATA_POINTER_PAYLOAD:
                        buf = reinterpret_cast<void*>(*(reinterpret_cast<uint32_t*>(&tlv_info.val[0])));
                        break;
                    case SCF_TX_DATA_OFFSET:
                        //Value field in the TLV which corresponds to tb offset in the TB buffer is not used
                        //TB size field in DL_TTI for the corresponding PDSCH PDU is used to figure out the
                        //offset of this TB in the TB buffer
                        buf = (uint8_t*)(tx_data_req_meta_data_.buf);
                        
                        pdu_offset = *reinterpret_cast<uint32_t*>(tlv_info.val);
                        NVLOGD_FMT(TAG, "{}:  SFN {}.{} cell_id={} TX_DATA.req: PDU {}-{}-{} tb_size={} pdu_offset padded: {} -> {}, PDSCH PDU buf={}",
                                __func__, group_cmd->slot.slot_3gpp.sfn_, group_cmd->slot.slot_3gpp.slot_, get_carrier_id(), pdsch_cw_idx_start + i, tx_data_req_meta_data_.num_pdus, i, ue_cw.tbSize, ue_cw.tbStartOffset, pdu_offset, buf);

                        // Pass the tbStartOffset to cuPHY
                        ue_cw.tbStartOffset = pdu_offset;

                        if(phy_module().prepone_h2d_copy())
                            buffLoc = 3;
                        break;
                    default:
                        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: Invalid TLV tag {} received in TX_DATA.request", __FUNCTION__, static_cast<unsigned>(tlv_info.tag));
                        break;
                }
                if(buf != nullptr)
                {
                    update_cell_command(phy_module().group_command(), slot_cmd, buf, false, cellPrmDynIdx, buffLoc, nullptr);
                    handled    = true;
                    isDlSlot   = true;
                }

                set_valid_dci_rx_slot(false);
            }
            break;
            case DL_TTI_PDU_TYPE_PDCCH:
            case DL_TTI_PDU_TYPE_CSI_RS:
            default:
                NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: Invalid DL PDU type.", __FUNCTION__);
                break;
        }

        offset += sizeof(scf_fapi_tx_data_pdu_info_t) + dl_pdu.num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t));
    }

    dl_pdu_index_size = 0;
    return handled;
}

bool phy::on_phy_dl_tx_request(scf_fapi_tx_data_req_t& request, nv_ipc_msg_t& ipc_msg, uint8_t* pdsch_pdu_valid_flag)
{
    if(duplicate_tx_data_req)
    {
        NVLOGW_FMT(TAG, "{}: Duplicate TX_DATA.req received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(request.sfn), static_cast<unsigned>(request.slot), static_cast<unsigned>(ipc_msg.cell_id));
        return false;
    }
    duplicate_tx_data_req = true;

    slot_command_api::slot_indication slot_ind;
    slot_ind.sfn_       = request.sfn;
    slot_ind.slot_      = request.slot;
    uint32_t tb_len     = 0;
    uint32_t offset     = 0;
    bool     handled    = false;
    bool     isDlSlot   = false;
    uint8_t  slot_index = slot_ind.slot_ % PDSCH_MAX_GPU_BUFFS;

    NVLOGD_FMT(TAG, "{}: TX_DATA.req received for SFN = {}, Slot ={} Number of PDUs = {}", __FUNCTION__, static_cast<int>(slot_ind.sfn_), static_cast<int>(slot_ind.slot_), static_cast<int>(request.num_pdus));
    if (state != fapi_state_t::FAPI_STATE_RUNNING) {
        NVLOGW_FMT(TAG, "{}: TX_DATA.req rejected - FAPI state not RUNNING (state={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        send_error_indication(static_cast<scf_fapi_message_id_e>(request.msg_hdr.type_id), SCF_ERROR_CODE_MSG_INVALID_STATE, request.sfn, request.slot);
        return handled;
    }
    
    // The PDSCH PDU for this cell is invalid or rejected (e.g. check_bf_pc_params failed), drop TX_DATA
    if(pdsch_pdu_valid_flag[phy_config.cell_config_.carrier_idx] == 1 || pdsch_rejected_)
    {
        NVLOGW_FMT(TAG, "{}: Dropping TX_DATA.req for cell_id={} since invalid/rejected PDSCH PDU for SFN {}.{}", __FUNCTION__, phy_config.cell_config_.carrier_idx, slot_ind.sfn_, slot_ind.slot_);
        if(cur_dl_msg.data_buf != nullptr)
        {
            phy_module().transport_wrapper().rx_release(cur_dl_msg);
            cur_dl_msg.reset();
        }
        return handled;
    }
    
    if(cur_dl_msg.data_buf != nullptr)
    {
        scf_fapi_header_t*      fapi_hdr = reinterpret_cast<scf_fapi_header_t*>(cur_dl_msg.msg_buf);
        scf_fapi_tx_data_req_t* last_req = reinterpret_cast<scf_fapi_tx_data_req_t*>(fapi_hdr->payload);
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Last TX_DATA.req not released: SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(request.sfn), static_cast<unsigned>(request.slot), static_cast<unsigned>(last_req->sfn), static_cast<unsigned>(last_req->slot), cur_dl_msg.cell_id);
        phy_module().transport_wrapper().rx_release(cur_dl_msg);
        cur_dl_msg.reset();
    }
    tx_data_req_meta_data_.num_pdus = request.num_pdus;
    tx_data_req_meta_data_.data     = reinterpret_cast<uint8_t*>(&request.payload[0]);
    for(int i = 0; i < request.num_pdus; i++)
    {
        auto& dl_pdu = *(reinterpret_cast<scf_fapi_tx_data_pdu_info_t*>(tx_data_req_meta_data_.data + offset));
#ifdef SCF_FAPI_10_04
        // Expect numTLV == 1: all transport blocks (TBs) are stored contiguously in a single flat buffer
        tb_len += dl_pdu.tlvs[0].length;
#else
        tb_len+= dl_pdu.pdu_len;
#endif
        offset += sizeof(scf_fapi_tx_data_pdu_info_t) + dl_pdu.num_tlv * (sizeof(scf_fapi_tl_t) + sizeof(uint32_t));
    }
    if(phy_module().prepone_h2d_copy())
    {
        // Check total padding bytes range: 0 ~ nPdu * 31
        if (ipc_msg.data_len < tb_len || ipc_msg.data_len > tb_len + request.num_pdus * 31) {
            NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: SFN {}.{}: cell {} NVIPC message data length mismatch: actual={} expected={}", __FUNCTION__,
                    static_cast<int>(request.sfn), static_cast<int>(request.slot), get_carrier_id(), ipc_msg.data_len, tb_len);
        }
        auto start_process_command_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        phyDriver.l1_copy_TB_to_gpu_buf(phy_cell_params.phyCellId, (uint8_t *)ipc_msg.data_buf, &tx_data_req_meta_data_.buf, ipc_msg.data_len, slot_index, request.sfn);
        auto end_process_command_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end_process_command_time_ - start_process_command_time_);
        NVLOGI_FMT(TAG, "{}: TB copy completed for cell_id {} SFN {}.{}: start_time={} duration={}ns", __FUNCTION__, get_carrier_id(), static_cast<int>(request.sfn), static_cast<int>(request.slot), start_process_command_time_.count(), diff.count());
    }
    else
    {
        tx_data_req_meta_data_.buf = (uint8_t *)ipc_msg.data_buf;
    }
    handled = process_dl_tx_request();

    if(ipc_msg.data_buf != nullptr )
    {
        if(handled)
            cur_dl_msg = reinterpret_cast<nv::phy_mac_msg_desc&>(ipc_msg);
        isDlSlot = true;
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: Invalid DL TB location configured = {}", __FUNCTION__, +phy_module().dl_tb_location());
    }

    if(isDlSlot)
    {
        dl_slot[ipc_msg.cell_id]++;
    }
    return handled;
}


void phy::prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pdcch_pdu_t& pdu)
{
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
#ifdef ENABLE_L2_SLT_RSP    
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_pdcch_pdu_l1_limits(pdu, cell_l1_limit.pdcch_errors);
    if (phy_module().cell_group() && !ret) {
        update_cell_command(grp_cmd, slot_cmd, pdu, 0, get_carrier_id(), slot_ind, phy_cell_params, phy_module().staticPdcchSlotNum(), phy_module().config_options(), phy_module().pm_map(), get_slot_detail(slot_ind), get_mMIMO_enable_info(), &cell_l1_limit.pdcch_errors);
    }
#else
    auto ret = VALID_FAPI_PDU;
    if (phy_module().cell_group() && !ret) {
        update_cell_command(grp_cmd, slot_cmd, pdu, 0, get_carrier_id(), slot_ind, phy_cell_params, phy_module().staticPdcchSlotNum(), phy_module().config_options(), phy_module().pm_map(), get_slot_detail(slot_ind), get_mMIMO_enable_info());
    }
#endif

}

void phy::on_ul_dci_request(scf_fapi_ul_dci_t& request, nv_ipc_msg_t& ipc_msg)
{
    if(duplicate_ul_dci_req)
    {
        NVLOGW_FMT(TAG, "{}: Duplicate UL_DCI.req received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(request.sfn), static_cast<unsigned>(request.slot), static_cast<unsigned>(ipc_msg.cell_id));
        return;
    }
    duplicate_ul_dci_req = true;

    slot_command_api::slot_indication slot_ind;
    uint16_t msg_len = 0;
    slot_ind.sfn_ = request.sfn;
    slot_ind.slot_ = request.slot;

    if (state != fapi_state_t::FAPI_STATE_RUNNING) {
        NVLOGW_FMT(TAG, "{}: UL_DCI.req rejected - FAPI state not RUNNING (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        send_error_indication(static_cast<scf_fapi_message_id_e>(request.msg_hdr.type_id), SCF_ERROR_CODE_MSG_INVALID_STATE, request.sfn, request.slot);
        return;
    }

    msg_len += sizeof(scf_fapi_ul_dci_t) + sizeof(scf_fapi_header_t);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    slot_cmd.cell = phy_cell_params.phyCellId;

    uint8_t pdus = request.num_pdus;
    NVLOGD_FMT(TAG, "{}: UL_DCI.req received for SFN = {}, Slot ={}", __FUNCTION__, slot_ind.sfn_, slot_ind.slot_);

    uint8_t* data = reinterpret_cast<uint8_t*>(request.payload);
#ifndef ENABLE_L2_SLT_RSP
    auto cell_group = phy_module().cell_group();
    if (!cell_group) {
    if (slot_cmd.channel_array_size == 0)
        {
            reset_cell_command(slot_cmd, slot_ind, get_carrier_id(), cell_group, phy_module().group_command());
        }
    } else {
        // Safely resets cell_group command if UL_DCI is the first message; otherwise, no effect.
        reset_cell_command(slot_cmd, slot_ind, get_carrier_id(), cell_group, phy_module().group_command());
    }
#endif
    for (uint8_t pdu = 0; pdu < pdus; pdu++) {
        //// Aggregate all DCI
        auto &payload_info = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data));
        if (payload_info.pdu_type != 0) {
            NVLOGW_FMT(TAG, "{}: Incorrect PDU type {} received in UL_DCI.req", __FUNCTION__, static_cast<int>(payload_info.pdu_type));
            send_error_indication(static_cast<scf_fapi_message_id_e>(request.msg_hdr.type_id), SCF_ERROR_CODE_MSG_UL_DCI_ERR, request.sfn, request.slot);
            return;
        }

        auto& pdcch_config = *reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&payload_info.pdu_config[0]);

        // Verify if UL DCI FAPI PDU check is enabled
        auto validate_mask = phy_module().fapi_config_check_mask();
        uint64_t pdcch_mask = (1 << channel_type::PDCCH_UL);
        pdcch_mask = validate_mask & pdcch_mask;
        
        // Validate UL DCI if the check is enabled 
        if(pdcch_mask)
        {
            if(validate_pdcch_pdu(pdcch_config) == INVALID_FAPI_PDU)
            {
                send_error_indication(static_cast<scf_fapi_message_id_e> (request.msg_hdr.type_id), scf_fapi_error_codes_t::SCF_ERROR_CODE_MSG_SLOT_ERR, request.sfn, request.slot);
                return;
            }
        }
       
        prepare_ul_slot_command(slot_ind, pdcch_config);

        data += payload_info.pdu_size;
        msg_len += payload_info.pdu_size;
    }

    if(msg_len > ipc_msg.msg_len)
    {
        NVLOGW_FMT(TAG, "{}: UL_DCI.req message size mismatch: FAPI={} IPC={}", __FUNCTION__, msg_len, ipc_msg.msg_len);
        send_error_indication(static_cast<scf_fapi_message_id_e>(request.msg_hdr.type_id), SCF_ERROR_CODE_MSG_UL_DCI_ERR, request.sfn, request.slot);
        return;
    }
}

void phy::on_param_request()
{
    NVLOGW_FMT(TAG, "{}: PARAM.req is not supported yet", __FUNCTION__);
}

void phy::on_dl_tti_request(scf_fapi_dl_tti_req_t &msg, nv_ipc_msg_t& ipc_msg, uint8_t* pdsch_valid_flag)
{
    if(duplicate_dl_tti_req)
    {
        NVLOGW_FMT(TAG, "{}: Duplicate DL_TTI.req received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(msg.sfn), static_cast<unsigned>(msg.slot), static_cast<unsigned>(ipc_msg.cell_id));
        return;
    }
    duplicate_dl_tti_req = true;

    slot_command_api::slot_indication slot_ind;
    slot_ind.sfn_ = msg.sfn;
    slot_ind.slot_ = msg.slot;
    uint8_t testMode = 0; /* default value '0' => Test Model disabled */
#ifdef ENABLE_CONFORMANCE_TM_PDSCH_PDCCH
    testMode = msg.testMode;
    NVLOGD_FMT(TAG, "{}: testMode = {}", __FUNCTION__,testMode);
#endif
    bool has_ssb_pdu = false;
    uint16_t msg_len = 0;

    if (state != fapi_state_t::FAPI_STATE_RUNNING) {
        NVLOGW_FMT(TAG, "{}: DL_TTI.req rejected - FAPI state not RUNNING (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_INVALID_STATE, msg.sfn, msg.slot);
        return;
    }

    
    auto validate_mask = phy_module().fapi_config_check_mask();

    if(validate_mask)
    {
        bool pdsch_pdu_check = true;
        /* Validate DL_TTI_REQ */
        if(validate_dl_tti_req(msg, validate_mask, pdsch_pdu_check) == INVALID_FAPI_PDU)
        {
            send_error_indication(static_cast<scf_fapi_message_id_e> (msg.msg_hdr.type_id), scf_fapi_error_codes_t::SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
            if(!pdsch_pdu_check){pdsch_valid_flag[phy_config.cell_config_.carrier_idx]=1;}
            return;
        }
    }
    
    msg_len += sizeof(scf_fapi_dl_tti_req_t) + sizeof(scf_fapi_header_t);
    //reset_current_fh_command();
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
#ifdef ENABLE_L2_SLT_RSP
    reset_pdsch_cw_offset();
#else
    reset_cell_command(slot_cmd, slot_ind, get_carrier_id(), phy_module().cell_group(), phy_module().group_command());
#endif

    dl_pdu_index_size = 0;
    slot_cmd.cell = phy_cell_params.phyCellId;
    //uint8_t numDCI = request.nDCI; <-- not in DL TTI
    uint16_t numPDU = msg.num_pdus;
    NVLOGD_FMT(TAG, "{}: DL_TTI.req received for SFN = {}, Slot ={} number of PDUs = {}", __FUNCTION__, slot_ind.sfn_, slot_ind.slot_, numPDU);
    uint8_t numGroup = msg.ngroup;
    uint16_t offset = 0;
    uint8_t *data = reinterpret_cast<uint8_t*>(msg.payload);

    bool dci = false;
    bool pdsch = false;
    uint32_t cell_idx = 0;
    bool first_csirs = false;
    uint32_t csirs_offset = 0;
    //If cell groups have to be disabled, set the flag to true so that it doesn't populate cell group parameters
    if(!phy_module().cell_group())
        pdsch = true;
    else
    {
        cell_group_command* group_cmd = phy_module().group_command();
        pdsch_params* pdsch_info = group_cmd->pdsch.get();
        if(pdsch_info)
        {
            pdsch_info->cell_ue_group_idx_start = pdsch_info->cell_grp_info.nUeGrps;
            pdsch_cw_idx_start = pdsch_info->cell_grp_info.nCws;
        }
    }

    for (uint16_t i = 0; i < numPDU; i++) {
        auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
        NVLOGD_FMT(TAG, "{}: PDU Type ={}, PDU Size={}, Offset={}", __FUNCTION__, static_cast<int>(pdu.pdu_type), static_cast<int>(pdu.pdu_size), offset);
        switch (pdu.pdu_type)
        {
            case DL_TTI_PDU_TYPE_PDCCH:
            {
                auto &pdu_dat = *reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&pdu.pdu_config[0]);
                prepare_dl_slot_command(slot_ind, pdu_dat, testMode);
                dci = true;
                break;
            }
            case DL_TTI_PDU_TYPE_PDSCH:
            {
                auto& pdu_dat = *reinterpret_cast<scf_fapi_pdsch_pdu_t*>(&pdu.pdu_config[0]);
                if(!pdsch)
                {
                    // cell_group_command* group_cmd = nv::PHY_module::group_command();
                    cell_group_command* group_cmd = phy_module().group_command();
                    group_cmd->slot.type = SLOT_DOWNLINK;
                    group_cmd->slot.slot_3gpp = slot_ind;
                    pdsch_params* pdsch_info = group_cmd->get_pdsch_params();
                    cell_idx = pdsch_info->cell_grp_info.nCells;
                    pdsch_info->cell_dyn_info[cell_idx].cellPrmStatIdx = cell_stat_prm_idx;
                    pdsch_info->cell_dyn_info[cell_idx].cellPrmDynIdx = cell_idx;
                    pdsch_info->cell_dyn_info[cell_idx].slotNum = ((phy_module().staticPdschSlotNum() > -1)? phy_module().staticPdschSlotNum() : msg.slot);
                    pdsch_info->cell_dyn_info[cell_idx].pdschStartSym = 0;
                    pdsch_info->cell_dyn_info[cell_idx].nPdschSym = 0;
                    pdsch_info->cell_dyn_info[cell_idx].dmrsSymLocBmsk = 0;
                    pdsch_info->cell_grp_info.nCells++;

                    NVLOGD_FMT(TAG, "{}: PDSCH nCells={}, phy_cell_id={}, cellPrmDynIdx={} cell_stat_prm_idx {}",__FUNCTION__,pdsch_info->cell_grp_info.nCells,get_carrier_id(),cell_idx,cell_stat_prm_idx);
                    pdsch_info->cell_index_list.push_back(get_carrier_id());
                    pdsch_info->phy_cell_index_list.push_back(phy_cell_params.phyCellId);
                    pdsch = true;
                }
                if (!prepare_dl_slot_command(slot_ind, pdu_dat, testMode))
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PDSCH rejected (check_bf_pc_params etc) - skip entire cell for this slot SFN {}.{}", __FUNCTION__, static_cast<unsigned>(msg.sfn), static_cast<unsigned>(msg.slot));
                    /* Rollback all PDSCH added this slot so we don't send slot cmd with PDSCH but no TB data */
                    auto* grp_cmd = phy_module().group_command();
                    if (grp_cmd)
                    {
                        if (grp_cmd->pdsch)
                            grp_cmd->pdsch->reset();
                        grp_cmd->fh_params.reset();
                    }
#ifdef ENABLE_L2_SLT_RSP
                    reset_pdsch_cw_offset();
#endif
                    pdsch_valid_flag[phy_config.cell_config_.carrier_idx] = 1;
                    pdsch_rejected_ = true;
                    dl_pdu_index_size = 0;
                    send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
                    break;
                }
                if(dl_pdu_index_size >= MAX_PDSCH_UE_GROUPS)
                {
                    NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: DL PDU index exceeds limit current={} max_allowed={}", __FUNCTION__, dl_pdu_index_size, MAX_PDSCH_UE_GROUPS);
                    send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
                    break;
                }

                dl_pdu_index[dl_pdu_index_size] = pdu.pdu_type;
                ++dl_pdu_index_size;
                int cw_offset = 0;
                for(int cw_i = 0; cw_i < pdu_dat.num_codewords; ++cw_i)
                {
                    auto& cw_dat = *reinterpret_cast<scf_fapi_pdsch_codeword_t*>(pdu_dat.codewords + cw_offset);
                    dl_thrput[ipc_msg.cell_id] += cw_dat.tb_size;
                }

                break;
            }
            case DL_TTI_PDU_TYPE_CSI_RS:
            {
                break;
            }
            case DL_TTI_PDU_TYPE_SSB:
            {
                has_ssb_pdu = true;
                auto &pdu_dat = *reinterpret_cast<scf_fapi_ssb_pdu_t*>(&pdu.pdu_config[0]);
                prepare_dl_slot_command(slot_ind, pdu_dat);
                break;
            }
            default:
            {
                NVLOGW_FMT(TAG, "{}: Unknown DL_TTI PDU type {}", __FUNCTION__, static_cast<int>(pdu.pdu_type));
                send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
                break;
            }
        }
        offset += pdu.pdu_size;
    }
    msg_len += offset;
    set_valid_dci_rx_slot(dci);
    offset = 0;

    /* Following are the cases for CSI-RS and PDSCH -
     * ZP CSI-RS + PDSCH - PDSCH will look into the RrcDynPrms. CSI-RS pipeline is not called.
     * NZP CSI-RS only - only CSI-RS look into RrcDynPrms. PDSCH pipeline is not called.
     * PDSCH only - neither PDSCH nor CSI-RS look into RrcDynPrms. CSI-RS pipeline is not called
     * NZP CSI-RS + PDSCH - both PDSCH & CSI-RS look into RrcDynPrms. Both pipelines called indepedently
     * ZP CSI-RS - no pipeline called
     */
    //Process CSI-RS at the end to identify if RrcDynPrms of PDSCH need to be updated or not
    for (uint16_t i = 0; i < numPDU; i++) {
        auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));

        if(pdu.pdu_type == DL_TTI_PDU_TYPE_CSI_RS)
        {
            NVLOGD_FMT(TAG, "{}: Processing CSI-RS:PDU Type ={}, PDU Size={}, Offset={} cell_id ={}", __FUNCTION__, static_cast<int>(pdu.pdu_type), static_cast<int>(pdu.pdu_size), offset, get_carrier_id());
            auto &pdu_dat = *reinterpret_cast<scf_fapi_csi_rsi_pdu_t*>(&pdu.pdu_config[0]);
            {
                //For the first CSI-RS of the cell, we need to capture the CSI-RS offset in the cell groups
                if(!first_csirs)
                {
                    if(pdsch)
                    {
                        cell_group_command* group_cmd = phy_module().group_command();
                        pdsch_params* pdsch_info = group_cmd->get_pdsch_params();
                        csirs_offset = pdsch_info->cell_grp_info.nCsiRsPrms;
                        NVLOGD_FMT(TAG, "{}: CSI-RS offset = pdsch_info->cell_grp_info.nCsiRsPrms = {} pdsch_exist={}", __FUNCTION__,
                            pdsch_info->cell_grp_info.nCsiRsPrms,pdsch);
                    }
                    first_csirs = true;
                }
            }
            prepare_dl_slot_command(slot_ind, pdu_dat, csirs_offset, pdsch);
        }
        offset += pdu.pdu_size;
    }

    if (first_csirs) // Rename to is_csirs_present_in_cell
    {
        csirs_params * csirs = nullptr;
        cell_group_command* group_cmd = phy_module().group_command();
        csirs = group_cmd->csirs.get();
        uint32_t nCsirs = (csirs ? csirs->cellInfo[csirs->nCells].nRrcParams : 0);
        if(pdsch || nCsirs)
        {
            csirs_fh_prepare_params &csirs_fh_params = group_cmd->fh_params.csirs_fh_params.at(get_carrier_id()); 

            csirs_fh_params.cell_idx = get_carrier_id();
            csirs_fh_params.cuphy_params_cell_idx = (pdsch ? cell_idx : -1);
            csirs_fh_params.cell_cmd = &phy_module().cell_sub_command(get_carrier_id());
            csirs_fh_params.bf_enabled = phy_module().bf_enabled();
            csirs_fh_params.mmimo_enabled = get_mMIMO_enable_info();
            csirs_fh_params.num_dl_prb = phy_cell_params.nPrbDlBwp;
            
            group_cmd->fh_params.is_csirs_cell.at(get_carrier_id()) = 1;
            group_cmd->fh_params.num_csirs_cell++; 
        }
        if(nCsirs)
        {
            csirs->nCells++;
            csirs->lastCell = std::max(csirs->lastCell, static_cast<uint16_t>(get_carrier_id() + 1));
            csirs->cellInfo[csirs->nCells].rrcParamsOffset = csirs->nCsirsRrcDynPrm;
            phy_module().is_csirs_slot(true);
        }
        NVLOGD_FMT(TAG, "{}: pdsch={} nCsirs={} CSIRS.nCells={}", __FUNCTION__,pdsch,nCsirs,csirs->nCells);
    }

    // TODO: do not report SSB missing before fixing the reporting logic
    has_ssb_pdu = true;
    if (has_ssb_pdu == false && ssb_slot_index.find(slot_ind.slot_) != ssb_slot_index.end() ) {
        NVLOGW_FMT(TAG, "{}: Missing BCH PDU SFN {}.{} ssb_period={}", __FUNCTION__, static_cast<int>(msg.sfn), static_cast<int>(msg.slot), phy_config.ssb_table_.ssb_period);
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_BCH_MISSING, msg.sfn, msg.slot);
        return;
    }

    if(msg_len > ipc_msg.msg_len)
    {
        NVLOGW_FMT(TAG, "{}: DL_TTI.req message size mismatch: FAPI={} IPC={}", __FUNCTION__, msg_len, ipc_msg.msg_len);
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
        return;
    }
}

void phy::on_ul_tti_request(scf_fapi_ul_tti_req_t& msg, nv_ipc_msg_t& ipc_msg)
{
    if(duplicate_ul_tti_req)
    {
        NVLOGW_FMT(TAG, "{}: Duplicate UL_TTI.req received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(msg.sfn), static_cast<unsigned>(msg.slot), static_cast<unsigned>(ipc_msg.cell_id));
        return;
    }
    duplicate_ul_tti_req = true;

    int max_srs_flag = 0;
    size_t nvIpcAllocBuffLen = 0;
    slot_command_api::slot_indication slot_ind;
    slot_ind.sfn_ = msg.sfn;
    slot_ind.slot_ = msg.slot;
    uint16_t msg_len = 0;
    if (state != fapi_state_t::FAPI_STATE_RUNNING) {
        NVLOGW_FMT(TAG, "{}: UL_TTI.req rejected - FAPI state not RUNNING (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_INVALID_STATE, msg.sfn, msg.slot);
        return;
    }

    
    auto validate_mask = phy_module().fapi_config_check_mask();

    if(validate_mask)
    {
        /* Validate UL_TTI_REQ */
        if(validate_ul_tti_req(msg, validate_mask) == INVALID_FAPI_PDU)
        {
            send_error_indication(static_cast<scf_fapi_message_id_e> (msg.msg_hdr.type_id), scf_fapi_error_codes_t::SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
            return;
        }
    }

    msg_len += sizeof(scf_fapi_ul_tti_req_t) + sizeof(scf_fapi_header_t);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
#ifndef ENABLE_L2_SLT_RSP
    reset_cell_command(slot_cmd, slot_ind, get_carrier_id(), phy_module().cell_group(), phy_module().group_command());
#endif
    slot_cmd.cell = phy_cell_params.phyCellId;
    uint num_pdu_rx = msg.num_pdus;
    uint8_t* data = reinterpret_cast<uint8_t*>(msg.payload);

    NVLOGD_FMT(TAG, "{}: UL_TTI.req received for SFN = {}, Slot ={} number of PDUs ={}", __FUNCTION__, slot_ind.sfn_, slot_ind.slot_, num_pdu_rx);

    uint offset = 0;
    bool first_srs = false;
    bool last_srs = false;
    uint8_t num_errors = 0;
    uint8_t num_srs_pdus = 0;
    uint8_t srs_pdu_ctr = 0;
    uint8_t last_puxch_pdu_idx = 0;
    bool last_non_prach_pdu = false;
    const uint8_t enable_srs_flag = get_enable_srs_info();

    int *p_srs_ind_index = NULL;

    // Reset PUCCH hopping ID for this slot
    pucch_hopping_id_ = 0;
    bool pucch_hopping_id_found = false;

    for (uint i = 0 ; i < num_pdu_rx; i++)
    {
        auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
        switch (pdu.pdu_type)
        {
            case UL_TTI_PDU_TYPE_SRS:
            {
                num_srs_pdus++;
                break;
            }
            case UL_TTI_PDU_TYPE_PUSCH:
            //case UL_TTI_PDU_TYPE_PUCCH: // TODO: Fix so it also works with just PUSCH+SRS. This is worked around in the next switch of UL_TTI_PDU_TYPE_SRS
            {
                last_puxch_pdu_idx = i;
                break;
            }
            case UL_TTI_PDU_TYPE_PUCCH:
            {
                // Iterate through PDUs to identify the first PUCCH PDU of format 0, 1, 3, or 4 and extract its hopping_id for use in subsequent processing
                // This is needed because hopping_id invalid for PUCCH FORMAT 2 per FAPI spec, so we need to extract it from the first PUCCH PDU of format 0, 1, 3, or 4.
                if (!pucch_hopping_id_found) {
                    auto &pdu_dat = *reinterpret_cast<scf_fapi_pucch_pdu_t*>(&pdu.pdu_config[0]);
                    if (pdu_dat.format_type != UL_TTI_PUCCH_FORMAT_2) {
                        pucch_hopping_id_ = pdu_dat.hopping_id;
                        pucch_hopping_id_found = true;
                    }
                }
                break;
            }
        }
        offset += pdu.pdu_size;
        if (num_srs_pdus == slot_command_api::MAX_SRS_PDU_PER_SLOT) {
            NVLOGD_FMT(TAG, "{}: num_srs_pdus={}", __FUNCTION__, num_srs_pdus);
            break;
        }
    }

    offset = 0;

    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    auto cell_id = phy_config.cell_config_.carrier_idx;
    ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
    ru_type ru = mplane.ru;

#ifdef SCF_FAPI_10_04_SRS
    // SRS without PUSCH in SINGLE_SECT_MODE: send error indication once, then skip only SRS PDUs in the loop below (other PDUs still processed).
    if (ru == SINGLE_SECT_MODE && num_srs_pdus > 0 && msg.num_ulsch == 0) {
        NVLOGI_FMT(TAG,
            "{}: SRS without PUSCH is not supported in SINGLE_SECT_MODE. SFN {}.{} cell_id={}",
            __FUNCTION__, static_cast<unsigned>(msg.sfn), static_cast<unsigned>(msg.slot), ipc_msg.cell_id);
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id),
            SCF_ERROR_CODE_SRS_WITHOUT_PUSCH_UNSUPPORTED, msg.sfn, msg.slot, true);
    }
#endif

    for (uint i = 0 ; i < num_pdu_rx; i++) {
        auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(data + offset));
        switch (pdu.pdu_type)
        {
            case UL_TTI_PDU_TYPE_PRACH:
            {
                auto &pdu_dat = *reinterpret_cast<scf_fapi_prach_pdu_t*>(&pdu.pdu_config[0]);
                on_prach_pdu_info(pdu_dat, slot_ind);
                break;
            }
            case UL_TTI_PDU_TYPE_PUSCH:
            {
                auto &pdu_dat = *reinterpret_cast<scf_fapi_pusch_pdu_t*>(&pdu.pdu_config[0]);
                if (on_pusch_pdu_info(pdu_dat)) {
                    prepare_ul_slot_command(slot_ind, pdu_dat);
                } else {
                   num_errors++;
                }
                break;
            }
            case UL_TTI_PDU_TYPE_PUCCH:
            {
                auto &pdu_dat = *reinterpret_cast<scf_fapi_pucch_pdu_t*>(&pdu.pdu_config[0]);
                on_pucch_pdu_info(pdu_dat, slot_ind);
                break;
            }
            case UL_TTI_PDU_TYPE_SRS:
            {
#ifndef SCF_FAPI_10_04_SRS
                // Skip SRS PDU processing for SINGLE_SECT_MODE type, as some L2s do not support SRS.
                if (ru == SINGLE_SECT_MODE) { break;}
#else
                // Skip only SRS PDUs when SRS without PUSCH in SINGLE_SECT_MODE (error already sent before this loop).
                if (ru == SINGLE_SECT_MODE && msg.num_ulsch == 0) { break; }
#endif
                if (!enable_srs_flag)
                {
                    NVLOGI_FMT(TAG, "{}: cell_id={} received SRS PDU while enable_srs=false - dropping PDU", __FUNCTION__, ipc_msg.cell_id);
                    break;
                }
                if(!max_srs_flag)
                {
                    auto &pdu_dat = *reinterpret_cast<scf_fapi_srs_pdu_t*>(&pdu.pdu_config[0]);
#ifdef SCF_FAPI_10_04
                    uint16_t rnti = pdu_dat.rnti;
                    uint16_t srsChestBufferIndex = static_cast<uint16_t>((pdu_dat.handle >> 8) & 0xFFFF);
                    NVLOGD_FMT(TAG, "{}: SFN {}.{} cellid {} rnti {} handle {} srsChestBufferIndex {}",__FUNCTION__,slot_ind.sfn_,slot_ind.slot_,ipc_msg.cell_id,rnti, static_cast<unsigned int>(pdu_dat.handle), srsChestBufferIndex);
                    slot_command_api::srsChestBuffState srsChestBuffState = slot_command_api::SRS_CHEST_BUFF_NONE;
                    if(phyDriver.l1_cv_mem_bank_get_buffer_state(ipc_msg.cell_id, srsChestBufferIndex, &srsChestBuffState) == -1)
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Buffer state check failed cell_id={} rnti={} index={}", __FUNCTION__, ipc_msg.cell_id, rnti, srsChestBufferIndex);
                        return;
                    }
                    /* Validate the SRS Chest buffer state and return error if the state is already in REQUESTED state during UL_TTI.req SRS_PDU processing 
                     * SRS Chest buffer state is in REQUESTED state means the buffer is under prepration and SRS.IND is not yet sent till now for this buffer.
                     * Hence, this buffer cannot be reused immediately till SRS.IND is sent. */
                    if(srsChestBuffState == slot_command_api::SRS_CHEST_BUFF_REQUESTED)
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS PDU dropped as buffer state is SRS_CHEST_BUFF_REQUESTED cell_id={}, rnti={}, srsChestBufferIndex={}", __FUNCTION__, ipc_msg.cell_id, rnti, srsChestBufferIndex);
                        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE, msg.sfn, msg.slot);
                        
                        /* Increment srs_pdu_ctr before dropping to maintain counter consistency.
                         * 
                         * Case 1 - Non-last PDU dropped: Incrementing the counter ensures that when subsequent 
                         * PDUs are processed, the actual last PDU will correctly trigger finalization 
                         * (last_srs = true) and update_fh_params_srs will be called with accumulated state.
                         * 
                         * Case 2 - Last PDU dropped: If this is the last PDU and previous PDUs were successfully 
                         * processed, those PDUs have accumulated state in srs_params->rb_info_per_sym[] but 
                         * update_fh_params_srs won't be called because no PDU will have last_srs=true.
                         * We handle this by manually triggering slot finalization when the last PDU is dropped.
                         */
                        srs_pdu_ctr++;
                        
                        /* Check if this was the last PDU and we need to finalize.
                         * Three conditions must be met:
                         * 1. srs_pdu_ctr == num_srs_pdus: This is the last PDU in the slot
                         * 2. srs_pdu_ctr > 1: At least one previous PDU exists (not the only PDU)
                         * 3. first_srs: At least one PDU was successfully processed
                         * 
                         * IMPORTANT: first_srs (NOT !first_srs) is the correct check
                         * - first_srs starts as false (line 1299)
                         * - first_srs becomes true when first SUCCESSFUL PDU allocates buffer (line 1528)
                         * - first_srs=true means: buffer allocated, p_srs_ind_index valid, nCells > 0
                         * - first_srs=false means: all PDUs dropped, no buffer, nothing to finalize
                         * 
                         * The first_srs check is CRITICAL for safety:
                         * - Prevents accessing cell_dyn_info[nCells-1] when nCells=0 (all PDUs dropped)
                         * - Prevents dereferencing NULL p_srs_ind_index
                         * - Ensures finalization only occurs when there's accumulated state to finalize
                         * 
                         * Scenario matrix:
                         * - PDU1 processed, PDU2 dropped: first_srs=true  → MUST finalize PDU1's state ✓
                         * - PDU1 dropped, PDU2 processed: first_srs=true  → normal path handles it ✓
                         * - PDU1 dropped, PDU2 dropped:   first_srs=false → nothing to finalize, skip ✓
                         */
                        if (srs_pdu_ctr == num_srs_pdus && srs_pdu_ctr > 1 && first_srs)
                        {
                            // Last SRS PDU dropped, but previous PDUs were successfully processed
                            // Need to trigger finalization to populate PRB parameters for those PDUs
                            NVLOGW_FMT(TAG, "{}: Last SRS PDU dropped after {} previous PDUs processed. Triggering slot finalization. "
                                      "SFN {}.{} cell_id={}", 
                                      __FUNCTION__, srs_pdu_ctr - 1, slot_ind.sfn_, slot_ind.slot_, ipc_msg.cell_id);
                            
                            try {
                                // Get necessary objects for finalization
                                auto cell_grp_cmd = phy_module().group_command();
                                srs_params* srs_params_ptr = cell_grp_cmd->get_srs_params();
                                if (srs_params_ptr != nullptr && srs_params_ptr->cell_grp_info.nCells > 0)
                                {
                                    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
                                    auto& cell_info = srs_params_ptr->cell_dyn_info[srs_params_ptr->cell_grp_info.nCells-1];
                                    
                                    // Get slot detail for the cell
                                    nv::slot_detail_t* slot_detail = get_slot_detail(slot_ind);
                                    
                                    // Determine if this is the last non-PRACH PDU
                                    bool is_last_non_prach = i >= last_puxch_pdu_idx;
                                    
                                    // Get beamforming configuration from PDU payload
                                    #ifdef SCF_FAPI_10_04_SRS
                                    scf_fapi_rx_beamforming_t* srs_rx_bf = reinterpret_cast<scf_fapi_rx_beamforming_t*>(&pdu_dat.payload[0]);
                                    bool bf_enabled = (srs_rx_bf->trp_scheme == 0 && srs_rx_bf->num_prgs > 0);
                                    #else
                                    scf_fapi_rx_beamforming_t dummy_bf = {0};
                                    scf_fapi_rx_beamforming_t* srs_rx_bf = &dummy_bf;
                                    bool bf_enabled = false;
                                    #endif
                                    
                                    // Call finalization to populate PRB parameters based on accumulated state
                                    finalize_srs_slot(slot_cmd, *srs_rx_bf, 
                                                     cell_info.nSrsSym, cell_info.srsStartSym, srs_params_ptr, 
                                                     bf_enabled, ru, slot_detail, get_carrier_id(), is_last_non_prach);
                                    
                                    NVLOGD_FMT(TAG, "{}: Slot finalization completed for {} successfully processed SRS PDUs", 
                                              __FUNCTION__, srs_pdu_ctr - 1);
                                }
                            }
                            catch (const std::exception& e)
                            {
                                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Exception during slot finalization after last PDU drop: {}", 
                                          __FUNCTION__, e.what());
                            }
                        }
                        else
                        {
                            NVLOGD_FMT(TAG, "{}: SRS PDU dropped, srs_pdu_ctr incremented to {} of {} total", 
                                      __FUNCTION__, srs_pdu_ctr, num_srs_pdus);
                        }
                        break;
                    }
#endif
                    auto cell_grp_cmd = phy_module().group_command();
                    srs_params* srs_params = cell_grp_cmd->get_srs_params();
                    if (srs_params == nullptr)
                    {
                        NVLOGW_FMT(TAG, "{}: No SRS params found for cell_id {}. Skipping SRS PDU processing.", __FUNCTION__, get_carrier_id());
                        continue; // Skip processing if no SRS params
                    }
                    
                    if(!first_srs)
                    {
                        nv::phy_mac_transport& transport = phy_module().transport(get_carrier_id());
                        nv::phy_mac_msg_desc msg_desc;
                        /* Using CPU_DATA would result in a sending mutiple SRS IND. Hence CPU_LARGE buffer is used for sending SRS IND because it can handle reports of a higher number of SRS PDUs compared to CPU_DATA. */
                        msg_desc.data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
                        if(transport.tx_alloc(msg_desc) < 0)
                        {
                            return;
                        }

                        switch(transport.get_nv_ipc_config()->ipc_transport)
                        {
                            case NV_IPC_TRANSPORT_SHM:
                                nvIpcAllocBuffLen = transport.get_nv_ipc_config()->transport_config.shm.mempool_size[msg_desc.data_pool].buf_size;
                                break;
                            case NV_IPC_TRANSPORT_DPDK:
                                nvIpcAllocBuffLen = transport.get_nv_ipc_config()->transport_config.dpdk.mempool_size[msg_desc.data_pool].buf_size;
                                break;
                            case NV_IPC_TRANSPORT_DOCA:
                                nvIpcAllocBuffLen = transport.get_nv_ipc_config()->transport_config.doca.mempool_size[msg_desc.data_pool].buf_size;
                                break;
                            case NV_IPC_TRANSPORT_UDP:
                            default:
                                NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Invalid transport type {}", __FUNCTION__, +transport.get_nv_ipc_config()->ipc_transport);
                        }
                        first_srs = true;
                        /*Init */
                        srs_params->num_srs_ind_indexes[srs_params->cell_grp_info.nCells] = 0;
                        p_srs_ind_index = &srs_params->num_srs_ind_indexes[srs_params->cell_grp_info.nCells];
                        srs_params->srs_indications[srs_params->cell_grp_info.nCells][*p_srs_ind_index] = (nv_ipc_msg_t)msg_desc;
                        NVLOGD_FMT(TAG, "{}: SRS ind allocation SFN {}.{} cell_index={} msg_id={} msg_len={} data_len={} data_pool={} msg_buf={} data_buf={} buff_len={} offset={} sizeof_matrix={} nCells={} srs_ind_index={}", __FUNCTION__,
                            slot_ind.sfn_, slot_ind.slot_, get_carrier_id(), msg_desc.msg_id, msg_desc.msg_len, msg_desc.data_len, msg_desc.data_pool, msg_desc.msg_buf, msg_desc.data_buf, nvIpcAllocBuffLen,
                            srs_params->srs_indications[srs_params->cell_grp_info.nCells][*p_srs_ind_index].data_len, sizeof(scf_fapi_norm_ch_iq_matrix_info_t), srs_params->cell_grp_info.nCells, *p_srs_ind_index);

                        /*Init number of SRS PDUs per SRS Ind for current slot */
                        for(int i = 0; i < MAX_SRS_IND_PER_SLOT; i++)
                        {
                            srs_params->num_srs_pdus_per_srs_ind[srs_params->cell_grp_info.nCells][i] = 0;
                        }
                    }
                    
                    srs_pdu_ctr++;
                    if (srs_pdu_ctr == num_srs_pdus)
                    {
                        last_srs = true;
                        last_non_prach_pdu = i >= last_puxch_pdu_idx;
                    }
                    NVLOGD_FMT(TAG, "{}: srs_pdu_ctr={} num_srs_pdus={} last_srs={} last_non_prach_pdu={}", __FUNCTION__, srs_pdu_ctr, num_srs_pdus, last_srs, last_non_prach_pdu);
                    int retVal = on_srs_pdu_info(pdu_dat, slot_ind, nvIpcAllocBuffLen, p_srs_ind_index, last_srs, last_non_prach_pdu);
                    switch(retVal)
                    {
                        case SRS_PDU_SUCCESS:
                            NVLOGD_FMT(TAG, "{}: Final SRS IND index={}", __FUNCTION__, *p_srs_ind_index);
                            break;
                        case SRS_PDU_OVERFLOW_NVIPC_BUFF:
                            {
                                max_srs_flag = 1;  //Ignore any more SRS PDUs   

                                //Send an ERROR IND to L2 to indicate a partial SRS.IND will follow
                                NVLOGD_FMT(TAG, "{}: Partial SRS Ind to follow", __FUNCTION__);
                                send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_PARTIAL_SRS_IND_ERR, msg.sfn, msg.slot);
                            }
                            break;
                        case SRS_PDU_L1_LIMIT_ERROR:
                            {
                                NVLOGD_FMT(TAG, "{}: SRS PDU L1 limit error", __FUNCTION__);
                                max_srs_flag = 1; /* Ignore any more SRS PDUs*/
                                send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_FAPI_SRS_L1_LIMIT_EXCEEDED, msg.sfn, msg.slot);
                            }
                            break;
                        default:
                            NVLOGE_FMT(TAG,AERIAL_L2ADAPTER_EVENT,"{}: on_srs_pdu_info returned error {}", __FUNCTION__,retVal);
                            break;
                    }
                }
            break;
        }
        default:
        {
            NVLOGW_FMT(TAG, "{}: Unknown UL_TTI PDU type {}", __FUNCTION__, static_cast<int>(pdu.pdu_type));
            send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
            break;
        }
    }
    offset += pdu.pdu_size;
    }
    msg_len += offset;
    if(msg_len > ipc_msg.msg_len)
    {
        NVLOGW_FMT(TAG, "{}: UL_TTI.req message size mismatch: FAPI={} IPC={}", __FUNCTION__, msg_len, ipc_msg.msg_len);
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
        return;
    }
    if (num_errors > 0) {
        send_error_indication(static_cast<scf_fapi_message_id_e> (msg.msg_hdr.type_id), scf_fapi_error_codes_t::SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
    }

}

#ifdef SCF_FAPI_10_04
void phy::on_dl_bfw_request(scf_fapi_dl_bfw_cvi_request_t& msg, nv_ipc_msg_t& ipc_msg)
{
    if(duplicate_dl_bfw_cvi_req)
    {
        NVLOGW_FMT(TAG, "{}: Duplicate DL_BFW_CVI.req received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(msg.sfn), static_cast<unsigned>(msg.slot), static_cast<unsigned>(ipc_msg.cell_id));
        return;
    }
    duplicate_dl_bfw_cvi_req = true;

    slot_command_api::slot_indication slot_ind;
    slot_ind.sfn_ = msg.sfn;
    slot_ind.slot_ = msg.slot;
    uint16_t msg_len = 0;
    if (state != fapi_state_t::FAPI_STATE_RUNNING)
    {
        NVLOGW_FMT(TAG, "{}: DL_BFW_CVI.req rejected - FAPI state not RUNNING (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_INVALID_STATE, msg.sfn, msg.slot);
        return;
    }

    auto validate_mask = phy_module().fapi_config_check_mask();

    if(validate_mask)
    {
        /* Validate DL_BFW_CVI_REQ */
        if(validate_ul_dl_bfw_cvi_req(msg, DL_BFW, validate_mask) == INVALID_FAPI_PDU)
        {
            send_error_indication(static_cast<scf_fapi_message_id_e> (msg.msg_hdr.type_id), scf_fapi_error_codes_t::SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
            return;
        }
    }

    msg_len += sizeof(scf_fapi_dl_bfw_cvi_request_t) + sizeof(scf_fapi_header_t);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
#ifndef ENABLE_L2_SLT_RSP
    reset_cell_command(slot_cmd, slot_ind, get_carrier_id(), phy_module().cell_group(), phy_module().group_command());
#endif
    slot_cmd.cell = phy_cell_params.phyCellId;
    uint num_pdu_rx = msg.npdus;
    uint8_t* data = reinterpret_cast<uint8_t*>(msg.config_pdu);
    NVLOGD_FMT(TAG, "{}: DL_BFW_CVI.req received for SFN = {}, Slot ={} number of PDUs ={}", __FUNCTION__, slot_ind.sfn_, slot_ind.slot_, num_pdu_rx);

    uint offset = 0;
    uint32_t droppedDlBFWPdu = 0;
    for (uint i = 0 ; i < num_pdu_rx; i++)
    {
        auto &pdu = *(reinterpret_cast<scf_fapi_dl_bfw_group_config_t*>(data + offset));
        on_dl_bfw_pdu_info(pdu, slot_ind, droppedDlBFWPdu);
        offset += pdu.pdu_size;
    }
    // Send ERROR IND for each dropped DL BFW PDU
    for(uint32_t i = 0; i < droppedDlBFWPdu; i++)
    {
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE, msg.sfn, msg.slot);
    }
}

void phy::on_ul_bfw_request(scf_fapi_ul_bfw_cvi_request_t& msg, nv_ipc_msg_t& ipc_msg)
{
    if(duplicate_ul_bfw_cvi_req)
    {
        NVLOGW_FMT(TAG, "{}: Duplicate UL_BFW_CVI.req received SFN {}.{} cell_id={}", __FUNCTION__, static_cast<unsigned>(msg.sfn), static_cast<unsigned>(msg.slot), static_cast<unsigned>(ipc_msg.cell_id));
        return;
    }
    duplicate_ul_bfw_cvi_req = true;

    slot_command_api::slot_indication slot_ind;
    slot_ind.sfn_ = msg.sfn;
    slot_ind.slot_ = msg.slot;
    uint16_t msg_len = 0;
    if (state != fapi_state_t::FAPI_STATE_RUNNING)
    {
        NVLOGW_FMT(TAG, "{}: UL_BFW_CVI.req rejected - FAPI state not RUNNING (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_MSG_INVALID_STATE, msg.sfn, msg.slot);
        return;
    }

    auto validate_mask = phy_module().fapi_config_check_mask();

    if(validate_mask)
    {
        // Validate UL_BFW_CVI_REQ
        if(validate_ul_dl_bfw_cvi_req(msg, UL_BFW, validate_mask) == INVALID_FAPI_PDU)
        {
            send_error_indication(static_cast<scf_fapi_message_id_e> (msg.msg_hdr.type_id), scf_fapi_error_codes_t::SCF_ERROR_CODE_MSG_SLOT_ERR, msg.sfn, msg.slot);
            return;
        }
    }

    msg_len += sizeof(scf_fapi_ul_bfw_cvi_request_t) + sizeof(scf_fapi_header_t);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
#ifndef ENABLE_L2_SLT_RSP
    reset_cell_command(slot_cmd, slot_ind, get_carrier_id(), phy_module().cell_group(), phy_module().group_command());
#endif
    slot_cmd.cell = phy_cell_params.phyCellId;
    uint num_pdu_rx = msg.npdus;
    uint8_t* data = reinterpret_cast<uint8_t*>(msg.config_pdu);
    NVLOGD_FMT(TAG, "{}: UL_BFW_CVI.req received for SFN = {}, Slot ={} number of PDUs ={}", __FUNCTION__, slot_ind.sfn_, slot_ind.slot_, num_pdu_rx);

    uint offset = 0;
    uint32_t droppedUlBFWPdu = 0;

    for (uint i = 0 ; i < num_pdu_rx; i++)
    {
        auto &pdu = *(reinterpret_cast<scf_fapi_ul_bfw_group_config_t*>(data + offset));
        on_ul_bfw_pdu_info(pdu, slot_ind, droppedUlBFWPdu);
        offset += pdu.pdu_size;
    }
    // Send ERROR IND for each dropped UL BFW PDU
    for(uint32_t i = 0; i < droppedUlBFWPdu; i++)
    {
        send_error_indication(static_cast<scf_fapi_message_id_e>(msg.msg_hdr.type_id), SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE, msg.sfn, msg.slot);
    }
}
#endif

inline uint16_t get_prev_slotIdx(uint16_t curr_slot)
{
    int slot_per_frame = nv::mu_to_slot_in_sf(1);
    if (curr_slot == 0)
    {
        return ( curr_slot + slot_per_frame - 1);
    }
    else
    {
        return (curr_slot - 1);
    }
}

void phy::on_pucch_pdu_info(scf_fapi_pucch_pdu_t& pdu_info, slot_command_api::slot_indication& slot_ind)
{
    prepare_ul_slot_command(slot_ind, pdu_info, pucch_hopping_id_);
}

void phy::prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pucch_pdu_t& pdu, uint16_t pucch_hopping_id)
{
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
#ifdef ENABLE_L2_SLT_RSP
    auto& group_limit_errors = phy_module().get_group_limit_errors();  
    auto ret = validate_pucch_pdu_l1_limits(pdu, group_limit_errors.pucch_errors);
#else
    auto ret = VALID_FAPI_PDU;
#endif
    if(phy_module().cell_group() && ret == VALID_FAPI_PDU)
    {
        update_cell_command(grp_cmd, slot_cmd, slot_ind, pdu, get_carrier_id(), phy_module().dtx_thresholds(),(uint16_t) cell_stat_prm_idx, phy_module().config_options(), get_slot_detail(slot_ind), get_mMIMO_enable_info(), phy_cell_params.nPrbUlBwp, pucch_hopping_id);
    }
}

bool phy::on_pusch_pdu_info(scf_fapi_pusch_pdu_t& pdu_info)
{
    bool value = true;
    auto validate_mask = phy_module().fapi_config_check_mask();
    uint64_t pusch_mask = 1 << channel_type::PUSCH;
    if (!validate_mask || ((validate_mask & pusch_mask) != pusch_mask)) {
        return value;
    } else {
        value = false;
        if (pdu_info.rnti == 0) { value = true; }
        else {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid RNTI={}", __FUNCTION__, static_cast<int>(pdu_info.rnti));  
            value = false;
            return value;
        }
        if (pdu_info.bwp.bwp_size > 0  && pdu_info.bwp.bwp_size < 276 ) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid BWP size={}", __FUNCTION__, static_cast<int>(pdu_info.bwp.bwp_size));
            value = false;
            return value;
        }
        if (pdu_info.bwp.bwp_start < 275) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid BWP start={}", __FUNCTION__, static_cast<int>(pdu_info.bwp.bwp_start));
            value = false;
            return value;
        }
        if (pdu_info.bwp.scs < 4) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid BWP SCS ={}", __FUNCTION__, static_cast<int>(pdu_info.bwp.scs));
            value = false;
            return value;
        }
        if (pdu_info.mcs_index < 32){value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid MCS Index={}", __FUNCTION__, static_cast<int>(pdu_info.mcs_index));
            value = false;
            return value;
        }
        if (pdu_info.mcs_table < 5) { value = true; }
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid MCS Table={}", __FUNCTION__, static_cast<int>(pdu_info.mcs_table));
            value = false;
            return value;
        }
        if (pdu_info.transform_precoding < 2 ) { value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Transform Precoding={}", __FUNCTION__, static_cast<int>(pdu_info.transform_precoding));
            value = false;
            return value;
        }
        // data_scrambling_id ranges from 0 through 65535.
        // if (pdu_info.data_scrambling_id )
        if (pdu_info.num_of_layers > 0 && pdu_info.num_of_layers < 5) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Number of Layers={}", __FUNCTION__, static_cast<int>(pdu_info.num_of_layers));
            value = false;
            return value;
        }
        if (pdu_info.ul_dmrs_sym_pos !=0 && pdu_info.ul_dmrs_scrambling_id < (1 << (OFDM_SYMBOLS_PER_SLOT + 1)) ) { value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid UL DMRS Symbol Position={}", __FUNCTION__, static_cast<int>(pdu_info.ul_dmrs_sym_pos));
            value = false;
            return value;
        }
        if (pdu_info.dmrs_config_type < 2 ) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid DMRS Config Type={}", __FUNCTION__, static_cast<int>(pdu_info.dmrs_config_type));
            value = false;
            return value;
        }
        // ul_dmrs_scrambling_id  ranges from 0 through 65535.
        // if (pdu_info.ul_dmrs_scrambling_id)
        if (pdu_info.pusch_identity < 1008) { value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid PUSCH Identity={}", __FUNCTION__, static_cast<int>(pdu_info.pusch_identity));
            value = false;
            return value;
        }
        if (pdu_info.scid < 2) { value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid SCID={}", __FUNCTION__, static_cast<int>(pdu_info.scid));
            value = false;
            return value;
        }
        if (pdu_info.num_dmrs_cdm_groups_no_data > 0 && pdu_info.num_dmrs_cdm_groups_no_data < 4) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid numDmrsCdmGrpsNoData={}", __FUNCTION__, static_cast<int>(pdu_info.num_dmrs_cdm_groups_no_data));
            value = false;
            return value;
        }
        if (pdu_info.dmrs_ports > 0 && pdu_info.dmrs_ports < (1 << 12)) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid DMRS Ports={}", __FUNCTION__, static_cast<int>(pdu_info.dmrs_ports));
            value = false;
            return value;
        }
        if (pdu_info.resource_alloc != 1) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid RA-Type={}", __FUNCTION__, static_cast<int>(pdu_info.resource_alloc));
            value = false;
            return value;
        }
        // pdu_info.rb_bitmap pending
        if (pdu_info.resource_alloc == 1) {
            if (pdu_info.rb_start < 275) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid RBStart={}", __FUNCTION__, static_cast<int>(pdu_info.rb_start));
                value = false;
                return value;
            }
            if (pdu_info.rb_size > 0 && pdu_info.rb_size < 276) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid RB Size={}", __FUNCTION__, static_cast<int>(pdu_info.rb_size));
                value = false;
                return value;
            }
        }

        // if (pdu_info.vrb_to_prb_mapping < 2) {value = true;}
        // else {
         NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH VRB To PRB mapping={}", __FUNCTION__, static_cast<int>(pdu_info.vrb_to_prb_mapping));
        //     value = false;
        //     return value;
        // }
        // if (pdu_info.frequency_hopping <  2) {value = true;}
        // else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Frequency Hopping={}", __FUNCTION__, static_cast<int>(pdu_info.frequency_hopping));
        //     value = false;
        //     return value;
        // }
        // if (pdu_info.tx_direct_current_location < (1<<12)) {value = true;}
        // else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Tx Direct current location={}", __FUNCTION__, static_cast<int>(pdu_info.tx_direct_current_location));
        //     value = false;
        //     return value;
        // }
        // if (pdu_info.ul_frequency_shift_7p5_khz < 2 ) {value = true;}
        // else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Uplink Frequency Shift 7.5Khz={}", __FUNCTION__, static_cast<int>(pdu_info.ul_frequency_shift_7p5_khz));
        //     value = false;
        //     return value;
        // }
        if (pdu_info.start_symbol_index < OFDM_SYMBOLS_PER_SLOT) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Start Symbol={}", __FUNCTION__, static_cast<int>(pdu_info.start_symbol_index));
            value = false;
            return value;
        }
        if (pdu_info.num_of_symbols > 0 && pdu_info.num_of_symbols < OFDM_SYMBOLS_PER_SLOT) {value = true;}
        else {
             NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Num of Symbols={}", __FUNCTION__, static_cast<int>(pdu_info.num_of_symbols));
            value = false;
            return value;
        }
        //puschData
        uint8_t* next = &pdu_info.payload[0];

        if (pdu_info.pdu_bitmap & 0x01) {
            auto data = reinterpret_cast<scf_fapi_pusch_data_t*>(next);
            if (data->rv_index < 4) {value = true;}
                else {
                     NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid RV Index={}", __FUNCTION__, static_cast<int>(data->rv_index));
                    value = false;
                    return value;
                }
            if (data->harq_process_id < 16) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid HARQ Process Id={}", __FUNCTION__, static_cast<int>(data->harq_process_id));
                value = false;
                return value;
            }
            if (data->new_data_indicator < 2 ) { value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid New Data Indicator={}", __FUNCTION__, static_cast<int>(data->new_data_indicator));
                value = false;
                return value;
            }
            if (data->tb_size < MAX_BYTES_PER_TRANSPORT_BLOCK ) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid TB Size={}", __FUNCTION__, static_cast<int>(data->tb_size));
                value = false;
                return value;
            }
            next += sizeof(scf_fapi_pusch_data_t);
        }
        if (pdu_info.pdu_bitmap & 0x02) {
            auto data = reinterpret_cast<scf_fapi_pusch_uci_t*>(next);
            if (data->harq_ack_bit_length < 13) { value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid HARQ Ack Bit length={}", __FUNCTION__, static_cast<int>(data->harq_ack_bit_length));
                value = false;
                return value;
            }
            if (data->csi_part_1_bit_length < 13 ) { value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid CSI Part 1 Bit Length={}", __FUNCTION__, static_cast<int>(data->csi_part_1_bit_length));
                value = false;
                return value;
            }
#ifdef SCF_FAPI_10_04
            if (data->flag_csi_part2 == 0 || data->flag_csi_part2 == std::numeric_limits<uint16_t>::max()) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid flag_csi_part2 Length={}", __FUNCTION__, static_cast<int>(data->flag_csi_part2));
                value = false;
                return value;
            }
#else
            if (data->csi_part_2_bit_length < 13 ) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid CSI Part 2 Bit Length={}", __FUNCTION__, static_cast<int>(data->csi_part_2_bit_length));
                value = false;
                return value;
            }
#endif
            if (data->alpha_scaling < 4) { value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid alpha Scaling={}", __FUNCTION__, static_cast<int>(data->alpha_scaling));
                value = false;
                return value;
            }
            if (data->beta_offset_harq_ack < 16) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Beta Offset HARQ Ack={}", __FUNCTION__, static_cast<int>(data->beta_offset_harq_ack));
                value = false;
                return value;
            }
            if (data->beta_offset_csi_1 < 16) { value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Beta offset CSI Part 1={}", __FUNCTION__, static_cast<int>(data->beta_offset_csi_1));
                value = false;
                return value;
            }
            if (data->beta_offset_csi_2 < 16) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Beta Offset CSI Part 2={}", __FUNCTION__, static_cast<int>(data->beta_offset_csi_2));
                value = false;
                return value;
            }

            next += sizeof(scf_fapi_pusch_uci_t);
        }
        if (!pdu_info.transform_precoding && (pdu_info.pdu_bitmap & 0x08)) {
            auto data = reinterpret_cast<scf_fapi_pusch_dftsofdm_t*>(next);
            if (data->lowPaprGroupNumber < 30) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid Low PAPR Group Number={}", __FUNCTION__, static_cast<int>(data->lowPaprGroupNumber));
                value = false;
                return value;
            }
            // No range specified in FAPI
            // if (data->lowPaprSequenceNumber < )
            if (data->ulPtrsSampleDensity < 9) {value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid UL PTRS Sample Density={}", __FUNCTION__, static_cast<int>(data->ulPtrsSampleDensity));
                value = false;
                return value;
            }
            if (data->ulPtrsTimeDensityTransformPrecoding < 5) { value = true;}
            else {
                 NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: PUSCH Invalid UL PTRS Time Density Transform Precoding={}", __FUNCTION__, static_cast<int>(data->ulPtrsTimeDensityTransformPrecoding));
                value = false;
                return value;
            }
            next += sizeof(scf_fapi_pusch_dftsofdm_t);
        }

        return value;
    }
    return value;
}


void phy::prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pusch_pdu_t& pdu)
{
#ifdef ENABLE_L2_SLT_RSP
    auto& group_l1_limit = phy_module().get_group_limit_errors();
    auto ret = validate_pusch_pdu_l1_limits(pdu, group_l1_limit.pusch_errors);
    if (ret == INVALID_FAPI_PDU) {
        return;
    }
#endif
    NVLOGD_FMT(TAG, "{}: Preparing UL PUSCH FH Command, Delay Tick:{}, SFN:{}, Slot:{}", __FUNCTION__, slot_ind.tick_, slot_ind.sfn_, slot_ind.slot_);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
    uint8_t prv_slot_idx = get_prev_slotIdx(slot_ind.slot_);
    bfw_coeff_mem_info_t* bfwCoeff_mem_info = nullptr;
    if (get_mMIMO_enable_info())
    {
        bfwCoeff_mem_info = phy_module().get_bfw_coeff_buff_info(get_carrier_id(), prv_slot_idx % MAX_BFW_COFF_STORE_INDEX);
    }

    if(phy_module().cell_group())
    {
        update_cell_command(grp_cmd, slot_cmd, pdu, get_carrier_id(), slot_ind, phy_module().staticPuschSlotNum(), phy_module().lbrm(), phy_module().bf_enabled(), (uint16_t) cell_stat_prm_idx, phy_module().dtx_thresholds_pusch(),bfwCoeff_mem_info, get_mMIMO_enable_info(), get_slot_detail(slot_ind), phy_cell_params.nPrbUlBwp);
    }
}

int phy::on_srs_pdu_info(scf_fapi_srs_pdu_t& pdu_info, slot_command_api::slot_indication& slot_ind, size_t nvIpcAllocBuffLen, int *p_srs_ind_index, bool is_last_srs_pdu, bool is_last_non_prach_pdu)
{
    return prepare_ul_slot_command(slot_ind, pdu_info, nvIpcAllocBuffLen,p_srs_ind_index, is_last_srs_pdu, is_last_non_prach_pdu);
}

int phy::prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_srs_pdu_t& pdu, size_t nvIpcAllocBuffLen, int *p_srs_ind_index, bool is_last_srs_pdu, bool is_last_non_prach_pdu)
{
    NVLOGD_FMT(TAG, "{}: Preparing UL SRS FH Command, Delay Tick:{}, SFN:{}, Slot:{}", __FUNCTION__, slot_ind.tick_, slot_ind.sfn_, slot_ind.slot_);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
    nv::phy_mac_transport& transport = phy_module().transport(get_carrier_id());
    int mutiple_srs_ind_allowed = 0; /* disabled by default*/
#ifdef ENABLE_L2_SLT_RSP
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_srs_pdu_l1_limits(pdu, cell_l1_limit.srs_errors);
    if (ret == INVALID_FAPI_PDU) {
        return SRS_PDU_L1_LIMIT_ERROR;
    }
#endif

#ifdef SCF_FAPI_10_04
    /* FAPI 10.04 supports splitting of SRS reports to L2 into multiple SRS IND's for which L2 can configure in Cell Config PDU thebelow field to MULTI_MSG_INSTANCE_PER_SLOT.
     * This is needed if one NVIPC buffer is not enough to send all requested SRS PDU reports in one SRS IND. 
     * If this field is not set to MULTI_MSG_INSTANCE_PER_SLOT and L1 is not able to accomodate all the requested SRS PDU reports in one SRS IND 
     * then L1 will send an ERROR IND to L2 to indicate only one partial SRS.IND will follow. */
    mutiple_srs_ind_allowed = (phy_module().get_phy_config().indication_instances_per_slot[nv::SRS_DATA_IND_IDX] ==  nv::MULTI_MSG_INSTANCE_PER_SLOT) ? 1 : 0;
#endif
    return scf_5g_fapi::update_cell_command(grp_cmd, slot_cmd, pdu, get_carrier_id(), slot_ind, phy_cell_params, (uint16_t) cell_stat_prm_idx, phy_module().bf_enabled(), nvIpcAllocBuffLen, p_srs_ind_index, mutiple_srs_ind_allowed, transport, is_last_srs_pdu, is_last_non_prach_pdu, get_slot_detail(slot_ind), get_mMIMO_enable_info());
}

void phy::prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pdcch_pdu_t& pdu, uint8_t testMode)
{
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
#ifdef ENABLE_L2_SLT_RSP
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_pdcch_pdu_l1_limits(pdu, cell_l1_limit.pdcch_errors);
    if (phy_module().cell_group() && !ret) {
        // update_cell_command(nv::PHY_module::group_command(), slot_cmd, pdu, get_carrier_id(), slot_ind, phy_cell_params, false, phy_module().staticPdcchSlotNum(), phy_module().bf_enabled());
        update_cell_command(grp_cmd, slot_cmd, pdu, testMode, get_carrier_id(), slot_ind, phy_cell_params, phy_module().staticPdcchSlotNum(), phy_module().config_options(), phy_module().pm_map(), get_slot_detail(slot_ind), get_mMIMO_enable_info(), &cell_l1_limit.pdcch_errors);
    }
#else
    auto ret = VALID_FAPI_PDU;
    if (phy_module().cell_group() && !ret) {
        update_cell_command(grp_cmd, slot_cmd, pdu, testMode, get_carrier_id(), slot_ind, phy_cell_params, phy_module().staticPdcchSlotNum(), phy_module().config_options(), phy_module().pm_map(), get_slot_detail(slot_ind), get_mMIMO_enable_info());
    }
#endif

}

bool phy::prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pdsch_pdu_t& pdu, uint8_t testMode)
{
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();

    uint8_t prv_slot_idx = get_prev_slotIdx(slot_ind.slot_);

    bfw_coeff_mem_info_t* bfwCoeff_mem_info = nullptr;
#ifdef ENABLE_L2_SLT_RSP
    auto& group_l1_limit = phy_module().get_group_limit_errors();
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_pdsch_pdu_l1_limits(pdu, group_l1_limit.pdsch_errors, cell_l1_limit.pdsch_pdu_error_contexts_info);
    if (ret == INVALID_FAPI_PDU) {
        return false;
    }
#endif
    if (get_mMIMO_enable_info())
    {
        bfwCoeff_mem_info = phy_module().get_bfw_coeff_buff_info(get_carrier_id(), prv_slot_idx % MAX_BFW_COFF_STORE_INDEX);
    }

    if(phy_module().cell_group())
    {
        if(grp_cmd->fh_params.total_num_pdsch_pdus >= MAX_ALLOWED_PDSCH_PDUS_PER_SLOT)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Capacity Check Failed!! PDSCH total_num_pdsch_pdus={} >= MAX_ALLOWED_PDSCH_PDUS_PER_SLOT={}", __FUNCTION__, grp_cmd->fh_params.total_num_pdsch_pdus, MAX_ALLOWED_PDSCH_PDUS_PER_SLOT);
            send_error_indication(SCF_FAPI_DL_TTI_REQUEST, SCF_ERROR_CODE_MSG_CAPACITY_EXCEEDED, slot_ind.sfn_, slot_ind.slot_);
            return false;
        }
        return update_cell_command(grp_cmd, slot_cmd, pdu, testMode, slot_ind, get_carrier_id(), nv::PHY_module::pm_map(), phy_module().pm_enabled(), phy_module().bf_enabled(), phy_cell_params.nPrbDlBwp, bfwCoeff_mem_info, get_mMIMO_enable_info(), get_slot_detail(slot_ind));
    }
    return true;
}

void phy::prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_ssb_pdu_t& pdu)
{
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
#ifdef ENABLE_L2_SLT_RSP
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_ssb_pdu_l1_limits(pdu, cell_l1_limit.ssb_pbch_errors);
#else
    auto ret = VALID_FAPI_PDU;
#endif

    if(phy_module().cell_group() && !ret)
    {
        update_cell_command(grp_cmd,slot_cmd, pdu, get_carrier_id(), slot_ind, phy_config, l_max, lmax_symbol_list, phy_module().config_options(), phy_module().pm_map(), get_slot_detail(slot_ind), get_mMIMO_enable_info());
    }
}

void phy::prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_csi_rsi_pdu_t& pdu, uint32_t csirs_offset, bool pdsch_exist)
{
    auto& phy_mod = phy_module();
    auto& slot_cmd = phy_mod.cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_mod.group_command();
#ifdef ENABLE_L2_SLT_RSP    
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_csirs_pdu_l1_limits(pdu, cell_l1_limit.csirs_errors);
    if (ret == INVALID_FAPI_PDU) {
        return;
    }
#endif
    update_cell_command(grp_cmd,slot_cmd, pdu, slot_ind, get_carrier_id(), phy_cell_params,phy_module().config_options(), phy_module().pm_map(), csirs_offset, pdsch_exist, cell_stat_prm_idx, get_mMIMO_enable_info(), get_slot_detail(slot_ind));
}

void phy::prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_prach_pdu_t& req)
{
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
#ifdef ENABLE_L2_SLT_RSP
    auto& cell_l1_limit = phy_module().get_cell_limit_errors(get_carrier_id());
    auto ret = validate_prach_pdu_l1_limits(req, cell_l1_limit.prach_errors);
    if (ret == INVALID_FAPI_PDU) {
        return;
    }
#else
    auto ret = VALID_FAPI_PDU;
#endif
    if (phy_module().cell_group() && !ret) {
        update_cell_command(grp_cmd, slot_cmd, slot_ind, req, phy_config, prach_addln_config, get_carrier_id(), phy_module().bf_enabled(), get_slot_detail(slot_ind), get_mMIMO_enable_info());
    }
}

void phy::on_prach_pdu_info(scf_fapi_prach_pdu_t &request, slot_command_api::slot_indication& slot_ind)
{
    if (request.num_prach_ocas == 0)
    {
        return;
    }

    prepare_ul_slot_command(slot_ind, request);
}

#ifdef SCF_FAPI_10_04
void phy::prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_ul_bfw_group_config_t& pdu, uint32_t &droppedUlBFWPdu)
{
    NVLOGD_FMT(TAG, "{}: Preparing UL BFW cuPHY pipeline, for cell_Idx:{} Delay Tick:{}, SFN:{}, Slot:{}", __FUNCTION__, get_carrier_id(), slot_ind.tick_, slot_ind.sfn_, slot_ind.slot_);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
    bfw_coeff_mem_info_t* bfwCoeff_mem_info = phy_module().get_bfw_coeff_buff_info(get_carrier_id(), slot_ind.slot_ % MAX_BFW_COFF_STORE_INDEX);

   if (*bfwCoeff_mem_info->header != BFW_COFF_MEM_FREE)
    {
       NVLOGD_FMT(TAG, "{}: bfwCoeff_mem_info header not marked as FREE (possible UL C-plane send error)", __FUNCTION__);
       *bfwCoeff_mem_info->header = BFW_COFF_MEM_FREE;
    }

    update_cell_command(grp_cmd, slot_cmd, pdu, get_carrier_id(), slot_ind, phy_cell_params, bfwCoeff_mem_info, slot_command_api::UL_BFW, get_slot_detail(slot_ind),droppedUlBFWPdu);
}

void phy::on_ul_bfw_pdu_info(scf_fapi_ul_bfw_group_config_t &pdu_info, slot_command_api::slot_indication& slot_ind, uint32_t &droppedUlBFWPdu)
{
    prepare_ul_slot_command(slot_ind, pdu_info, droppedUlBFWPdu);
}

void phy::prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_dl_bfw_group_config_t& pdu, uint32_t &droppedDlBFWPdu)
{
    NVLOGD_FMT(TAG, "{}: Preparing DL BFW cuPHY pipeline, for cell_Idx:{} Delay Tick:{}, SFN:{}, Slot:{}", __FUNCTION__, get_carrier_id(), slot_ind.tick_, slot_ind.sfn_, slot_ind.slot_);
    auto& slot_cmd = phy_module().cell_sub_command(get_carrier_id());
    auto grp_cmd = phy_module().group_command();
    bfw_coeff_mem_info_t* bfwCoeff_mem_info = phy_module().get_bfw_coeff_buff_info(get_carrier_id(), slot_ind.slot_ % MAX_BFW_COFF_STORE_INDEX);

    if (*bfwCoeff_mem_info->header != BFW_COFF_MEM_FREE)
    {
       NVLOGD_FMT(TAG, "{}: bfwCoeff_mem_info header not marked as FREE (possible DL C-plane send error)", __FUNCTION__);
       *bfwCoeff_mem_info->header = BFW_COFF_MEM_FREE;
    }
    update_cell_command(grp_cmd, slot_cmd, pdu, get_carrier_id(), slot_ind, phy_cell_params, bfwCoeff_mem_info, slot_command_api::DL_BFW, get_slot_detail(slot_ind), droppedDlBFWPdu);
}

void phy::on_dl_bfw_pdu_info(scf_fapi_dl_bfw_group_config_t &pdu_info, slot_command_api::slot_indication& slot_ind, uint32_t &droppedDlBFWPdu)
{
    prepare_dl_slot_command(slot_ind, pdu_info, droppedDlBFWPdu);
}
#endif

void phy::on_config_request(scf_fapi_config_request_msg_t& config_request, const int32_t cell_id, uint8_t handle_id, nv_ipc_msg_t& ipc_msg)
{
    NVLOGC_FMT(TAG, "{}: CONFIG.req received for cell_id={} numTLVs={} state={}", __FUNCTION__, cell_id, config_request.msg_body.num_tlvs, state.load());

    uint8_t *body_ptr = &config_request.msg_body.tlvs[0];
    auto tlvs = config_request.msg_body.num_tlvs;
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    uint8_t muMIMO_enable_flag = get_mMIMO_enable_info();
    uint8_t enable_srs_flag = get_enable_srs_info();
    uint32_t srsChest_buff_size = 0;
    uint8_t error_code = SCF_ERROR_CODE_MSG_OK;

    if(state == fapi_state_t::FAPI_STATE_RUNNING)
    {
        NVLOGW_FMT(TAG, "{}: send CONFIG.res for cell_id={} - FAPI_STATE_RUNNING", __FUNCTION__, cell_id);
        send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_INVALID_STATE);
        return;
    }

    if(state == fapi_state_t::FAPI_STATE_CONFIGURED)
    {
        NVLOGC_FMT(TAG, "{}: CONFIG.req received for cell_id={} in CONFIGURED state", __FUNCTION__, cell_id);

        /*Try to attain the PhyDriverCtx::updateCellConfigMutex. This lock protects running creation and deletion of PRACH objects
          at the same time. If the lock is available, call l1_cell_update_cell_config. If this function returns 0, means PRACH
          object are not yet created and config change is successful. Unlock the mutex in this condition. If this function returns 1
          it means that new PRACH objects need to be created and older ones need to be destroyed. The mutex will be unlocked in the
          thread that deletes the PRACH objects - see delete_prach_obj_func
        */
        if(phyDriver.l1_lock_update_cell_config_mutex() == false)
        {
            NVLOGC_FMT(TAG, "{}: send CONFIG.res for cell_id={} - try lock failed", __FUNCTION__, cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_INVALID_STATE);
            state = fapi_state_t::FAPI_STATE_CONFIGURED;
            return;
        }

        update_cells_stats(cell_id);
        int32_t prach_fd_index = -1;
        uint32_t prach_root_seq_unused_seq_index = 0;
        while (tlvs)
        {
            scf_fapi_tl_t *hdr = reinterpret_cast<scf_fapi_tl_t*>(body_ptr);
            switch(hdr->tag)
            {
                case CONFIG_TLV_DL_BANDWIDTH:
                    cell_update_config.carrier_config_.dl_bandwidth = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Carrier DL Bandwidth (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_bandwidth);
                    break;
                case CONFIG_TLV_UL_BANDWIDTH:
                    cell_update_config.carrier_config_.ul_bandwidth = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Carrier UL Bandwidth (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_bandwidth);
                    break;
                case CONFIG_TLV_PHY_CELL_ID:
                    cell_update_config.cell_config_.phy_cell_id = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Physical Cell ID (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.cell_config_.phy_cell_id);
                    break;
                case CONFIG_TLV_NUM_PRACH_FD_OCCASIONS:
                    cell_update_config.prach_config_.num_prach_fd_occasions = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: Number of PRACH FD Occasions (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.num_prach_fd_occasions);
                    break;
                case CONFIG_TLV_PRACH_ROOT_SEQ_INDEX:
                    prach_fd_index++;
                    prach_root_seq_unused_seq_index = 0;
                    if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                    {
                        cell_update_config.prach_config_.root_sequence[prach_fd_index].seq_index = hdr->AsValue<uint16_t>();
                        NVLOGI_FMT(TAG, "{} config request: PRACH Root Sequence Index (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].seq_index);
                    }
                    break;
                case CONFIG_TLV_NUM_ROOT_SEQ:
                    if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                    {
                        cell_update_config.prach_config_.root_sequence[prach_fd_index].number_root_sequence = hdr->AsValue<uint8_t>();
                        NVLOGI_FMT(TAG, "{} config request: Number of Root Sequence (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].number_root_sequence);
                    }
                    break;
                case CONFIG_TLV_K1:
                    if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                    {
                        cell_update_config.prach_config_.root_sequence[prach_fd_index].k1 = hdr->AsValue<uint16_t>();
                        NVLOGI_FMT(TAG, "{} config request: Frequency Offset K1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].k1);
                    }
                    break;
                case CONFIG_TLV_PRACH_ZERO_CORR_CONF:
                    if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                    {
                        cell_update_config.prach_config_.root_sequence[prach_fd_index].zero_conf = hdr->AsValue<uint8_t>();
                        NVLOGI_FMT(TAG, "{} config request: PRACH Zero Correlation Config (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].zero_conf, prach_fd_index);
                    }
                    break;
                case CONFIG_TLV_NUM_UNUSED_ROOT_SEQ:
                    if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                    {
                        cell_update_config.prach_config_.root_sequence[prach_fd_index].number_unused_sequence = hdr->AsValue<uint16_t>();
                        NVLOGI_FMT(TAG, "{} config request: Number of Unused Root Sequence (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].number_unused_sequence, prach_fd_index);
                    }
                    break;
                case CONFIG_TLV_UNUSED_ROOT_SEQ:
                    if(prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM && prach_root_seq_unused_seq_index < nv::NV_MAX_UNUSED_ROOT_SEQUENCE_NUM)
                    {
                        cell_update_config.prach_config_.root_sequence[prach_fd_index].unused_sequence[prach_root_seq_unused_seq_index] = hdr->AsValue<uint16_t>();
                        NVLOGI_FMT(TAG, "{} config request: Unused Root Sequence (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].unused_sequence[prach_root_seq_unused_seq_index], prach_fd_index);
                    }
                    prach_root_seq_unused_seq_index++;
                    break;
                case CONFIG_TLV_PRACH_CONFIG_INDEX:
                    cell_update_config.prach_config_.prach_conf_index = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: PRACH Config Index (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.prach_conf_index);
                    break;
                case CONFIG_TLV_RESTRICTED_SET_CONFIG:
                    cell_update_config.prach_config_.restricted_set_config = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: PRACH Restricted Set Config (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.restricted_set_config);
                    break;
                case CONFIG_TLV_VENDOR_DIGITAL_BEAM_TABLE_PDU:
                {
                    /* Handle DBT PDU during reconfiguration in CONFIGURED state */
                    if (ipc_msg.data_pool == NV_IPC_MEMPOOL_CPU_LARGE && ipc_msg.data_buf != nullptr &&  phy_module().bf_enabled())
                    {
                        int ret = update_dbt_pdu_table_ptr(cell_id, ipc_msg.data_buf);
                        if (ret != 0) {
                            error_code = SCF_ERROR_CODE_BEAM_ID_OUT_OF_RANGE;
                            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: Failed to store DBT PDU for cell_id={} in CONFIGURED state", __FUNCTION__, cell_id);
                        }
                    }
                    else
                    {
                        NVLOGW_FMT(TAG, "{}: Beamforming not enabled or invalid buffer for DBT PDU in CONFIGURED state", __FUNCTION__);
                    }
                }
                break;
            }
            // Round up TLV length to 4-byte boundary according to specs
            body_ptr += sizeof(scf_fapi_tl_t) + ((hdr->length + 3) / 4) * 4;
            tlvs--;
        }

        update_phy_driver_info_reconfig(phyDriver, cell_id);
        MemtraceDisableScope md;
        int32_t ret = phyDriver.l1_cell_update_cell_config(cell_reconfig_phy_driver_info, phy_module().cell_update_cb());

        if (ret == 0)
        {
            //If l1_cell_update_cell_config returns 0 means operation is complete. Send CONFIG.resp
            cell_update_success(phyDriver, cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_OK);
            phyDriver.l1_unlock_update_cell_config_mutex();
            phy_config.prach_config_.start_ro_index = phyDriver.l1_get_prach_start_ro_index(phy_cell_params.phyCellId);
            cell_update_config.prach_config_.start_ro_index = phy_config.prach_config_.start_ro_index;
        }
        else if (ret == -1)
        {
            NVLOGW_FMT(TAG, "{}: Cell config update failed cell_id={}", __FUNCTION__, cell_id);
            send_cell_config_response(cell_id, SCF_ERROR_CODE_MSG_INVALID_CONFIG);
            phyDriver.l1_unlock_update_cell_config_mutex();
            phy_config.prach_config_.start_ro_index = phyDriver.l1_get_prach_start_ro_index(phy_cell_params.phyCellId);
            cell_update_config.prach_config_.start_ro_index = phy_config.prach_config_.start_ro_index;
        }
        else if (ret == 1) {
            NVLOGI_FMT(TAG, "{}: l1_cell_update_cell_config returned 1 for carrier_id={}. CONFIG.res to be sent in another thread context", __FUNCTION__,
                phy_config.cell_config_.carrier_idx);
            cell_update_config.prach_config_.start_ro_index = phyDriver.l1_get_prach_start_ro_index(phy_cell_params.phyCellId);

        }

        NVLOGD_FMT(TAG, "{}: start_ro_index={}", __FUNCTION__, cell_update_config.prach_config_.start_ro_index);
        return;
    }

    phy_config.cell_config_.carrier_idx = cell_id;
    update_cells_stats(cell_id);
    // All starting points for TLVs that are part of a loop
    uint32_t slot_cfg_idx = 0;
    int32_t prach_fd_index = -1;
    uint32_t prach_root_seq_unused_seq_index = 0;

    // default
    phy_cell_params.nTxAnt    = phy_config.carrier_config_.num_tx_ants;
    phy_cell_params.nRxAnt    = phy_config.carrier_config_.num_rx_ants;
    phy_cell_params.nRxAntSrs = phy_config.carrier_config_.num_rx_ants;

    bool srsChestTlvPresent = false;
    bool txPortTlvPresent = false;
    bool rxPortTlvPresent = false;

    uint8_t mCh_segment_proc_enable = 0;
    auto retval = phyDriver.l1_get_ch_segment_proc_enable_info(&mCh_segment_proc_enable);

    while (tlvs)
    {
        scf_fapi_tl_t *hdr = reinterpret_cast<scf_fapi_tl_t*>(body_ptr);
        switch(hdr->tag)
        {
            // CELL PARAMETERS
            case CONFIG_TLV_DL_BANDWIDTH:
                phy_config.carrier_config_.dl_bandwidth = hdr->AsValue<uint16_t>();
                NVLOGI_FMT(TAG, "{} config request: Carrier DL Bandwidth (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_bandwidth);
                break;
            case CONFIG_TLV_DL_FREQ:
                phy_config.carrier_config_.dl_freq_abs_A = hdr->AsValue<uint32_t>();
                NVLOGI_FMT(TAG, "{} config request: Absolute frequency of DL point A (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_freq_abs_A);
                break;
            case CONFIG_TLV_DLK0:
                memcpy(reinterpret_cast<void*>(&phy_config.carrier_config_.dlk0), hdr->As<void*>(), sizeof(phy_config.carrier_config_.dlk0));
                NVLOGI_FMT(TAG, "{} config request: DL K0 for numerology 0 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dlk0[0]);
                NVLOGI_FMT(TAG, "{} config request: DL K0 for numerology 1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dlk0[1]);
                NVLOGI_FMT(TAG, "{} config request: DL K0 for numerology 2 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dlk0[2]);
                NVLOGI_FMT(TAG, "{} config request: DL K0 for numerology 3 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dlk0[3]);
                NVLOGI_FMT(TAG, "{} config request: DL K0 for numerology 4 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dlk0[4]);
                break;
            case CONFIG_TLV_NUM_TX_ANT:
                phy_config.carrier_config_.num_tx_ants = hdr->AsValue<uint16_t>();
                if(!muMIMO_enable_flag && phy_config.carrier_config_.num_tx_ants > 4)
                {
                    NVLOGW_FMT(TAG, "{}: TX antennas {} exceeds 4T4R limit - setting to maximum supported value of 4", __FUNCTION__, phy_config.carrier_config_.num_tx_ants);
                    phy_cell_params.nTxAnt = 4;
                }
                else
                {
                    phy_cell_params.nTxAnt = phy_config.carrier_config_.num_tx_ants;
                }
                NVLOGI_FMT(TAG, "{} config request: Number of TX Antennas (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.num_tx_ants);
                break;
            case CONFIG_TLV_UL_BANDWIDTH:
                phy_config.carrier_config_.ul_bandwidth = hdr->AsValue<uint16_t>();
                NVLOGI_FMT(TAG, "{} config request: Carrier UL Bandwidth (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_bandwidth);
                break;
            case CONFIG_TLV_UL_FREQ:
                phy_config.carrier_config_.ul_freq_abs_A = hdr->AsValue<uint32_t>();
                NVLOGI_FMT(TAG, "{} config request: Absolute frequency of UL point A (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_freq_abs_A);
                break;
            case CONFIG_TLV_ULK0:
                memcpy(reinterpret_cast<void*>(&phy_config.carrier_config_.ulk0), hdr->As<void*>(), sizeof(phy_config.carrier_config_.ulk0));
                NVLOGI_FMT(TAG, "{} config request: UL K0 for numerology 0 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ulk0[0]);
                NVLOGI_FMT(TAG, "{} config request: UL K0 for numerology 1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ulk0[1]);
                NVLOGI_FMT(TAG, "{} config request: UL K0 for numerology 2 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ulk0[2]);
                NVLOGI_FMT(TAG, "{} config request: UL K0 for numerology 3 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ulk0[3]);
                NVLOGI_FMT(TAG, "{} config request: UL K0 for numerology 4 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ulk0[4]);
                break;
            case CONFIG_TLV_NUM_RX_ANT:
                phy_config.carrier_config_.num_rx_ants = hdr->AsValue<uint16_t>();
                 if(!muMIMO_enable_flag && phy_config.carrier_config_.num_rx_ants > 4)
                {
                    NVLOGW_FMT(TAG, "{}: RX antennas {} exceeds 4T4R limit - setting to maximum supported value of 4", __FUNCTION__, phy_config.carrier_config_.num_rx_ants);
                    phy_cell_params.nRxAnt = 4;
                }
                else
                {
                    phy_cell_params.nRxAnt = phy_config.carrier_config_.num_rx_ants;
                }
                phy_cell_params.nRxAntSrs = phy_config.carrier_config_.num_rx_ants;
                phy_config.carrier_config_.num_rx_port = phy_config.carrier_config_.num_rx_ants;
                NVLOGI_FMT(TAG, "{} config request: Number of RX Antennas (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.num_rx_ants);
                break;
            case CONFIG_TLV_PHY_CELL_ID:
                // IDLE STATE
                phy_config.cell_config_.phy_cell_id = hdr->AsValue<uint16_t>();
                NVLOGI_FMT(TAG, "{} config request: Physical Cell ID (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.cell_config_.phy_cell_id);
                break;
            case CONFIG_TLV_FRAME_DUPLEX_TYPE:
                phy_config.cell_config_.frame_duplex_type = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: Frame Duplex Type (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.cell_config_.frame_duplex_type);
                break;
            case CONFIG_TLV_SSB_PBCH_POWER:
                phy_config.ssb_config_.ssb_pbch_power = hdr->AsValue<uint32_t>();
                NVLOGI_FMT(TAG, "{} config request: SSB Block Power (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_config_.ssb_pbch_power);
                break;
            case CONFIG_TLV_SCS_COMMON:
                phy_config.ssb_config_.sub_c_common = hdr->AsValue<uint8_t>();
                phy_cell_params.mu = phy_config.ssb_config_.sub_c_common;
                NVLOGI_FMT(TAG, "{} config request: Common Subcarrier Spacing (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_config_.sub_c_common);
                break;
            case CONFIG_TLV_PRACH_SEQ_LEN:
                phy_config.prach_config_.prach_seq_length = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Sequence Length (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.prach_seq_length);
                break;
            case CONFIG_TLV_PRACH_SUBC_SPACING: {
                phy_config.prach_config_.prach_scs = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Subcarrier Spacing (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.prach_scs);
                } break;
            case CONFIG_TLV_RESTRICTED_SET_CONFIG:
                phy_config.prach_config_.restricted_set_config = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Restricted Set Config (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.restricted_set_config);
                break;
            case CONFIG_TLV_NUM_PRACH_FD_OCCASIONS:
            {
                phy_config.prach_config_.num_prach_fd_occasions = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: Number of PRACH FD Occasions (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.num_prach_fd_occasions);
                break;
            }
            case CONFIG_TLV_PRACH_CONFIG_INDEX: {
                phy_config.prach_config_.prach_conf_index = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Config Index (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.prach_conf_index);
            } break;
            case CONFIG_TLV_PRACH_ROOT_SEQ_INDEX:
                prach_fd_index++;
                prach_root_seq_unused_seq_index = 0;
                if (prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    phy_config.prach_config_.root_sequence[prach_fd_index].seq_index = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: PRACH Root Sequence Index (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].seq_index);
                }
                break;
            case CONFIG_TLV_NUM_ROOT_SEQ:
                if (prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    phy_config.prach_config_.root_sequence[prach_fd_index].number_root_sequence = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: Number of Root Sequence (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].number_root_sequence);
                }
                break;
            case CONFIG_TLV_K1:
                if (prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    phy_config.prach_config_.root_sequence[prach_fd_index].k1 = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request: Frequency Offset K1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].k1);
                }
                break;
            case CONFIG_TLV_PRACH_ZERO_CORR_CONF:
                if (prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    phy_config.prach_config_.root_sequence[prach_fd_index].zero_conf = hdr->AsValue<uint8_t>();
                    NVLOGI_FMT(TAG, "{} config request: PRACH Zero Correlation Configuration (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].zero_conf, prach_fd_index);
                }
                break;
            case CONFIG_TLV_NUM_UNUSED_ROOT_SEQ:
                if (prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM)
                {
                    phy_config.prach_config_.root_sequence[prach_fd_index].number_unused_sequence = hdr->AsValue<uint16_t>();   // Loss of precision!
                    NVLOGI_FMT(TAG, "{} config request: Number of Unused Root Sequence (message ID {:X}) value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].number_unused_sequence, prach_fd_index);
                }
                break;
            case CONFIG_TLV_UNUSED_ROOT_SEQ:
                if (prach_fd_index >= 0 && prach_fd_index < nv::NV_MAX_PRACH_FD_OCCASION_NUM && prach_root_seq_unused_seq_index < nv::NV_MAX_UNUSED_ROOT_SEQUENCE_NUM)
                {
                    phy_config.prach_config_.root_sequence[prach_fd_index].unused_sequence[prach_root_seq_unused_seq_index] = hdr->AsValue<uint16_t>();
                    NVLOGI_FMT(TAG, "{} config request message ID {:X} value {} prach_fd_index {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.root_sequence[prach_fd_index].unused_sequence[prach_root_seq_unused_seq_index], prach_fd_index);
                    prach_root_seq_unused_seq_index++;
                }
                break; 

            case CONFIG_TLV_SSB_PER_RACH:
                phy_config.prach_config_.ssb_per_rach = hdr->AsValue<uint8_t>(); 
                NVLOGI_FMT(TAG, "{} config request: SSB Per RACH Occasion (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.ssb_per_rach);
                break;
            case CONFIG_TLV_PRACH_MULT_CARRIERS_IN_BAND: {
                phy_config.prach_config_.multiple_carriers_prach = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Multiple Carriers In Band (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.prach_config_.multiple_carriers_prach);
            }
            break;

            // SSB table
            case CONFIG_TLV_SSB_OFFSET_POINT_A:
                phy_config.ssb_table_.ssb_offset_point_A = hdr->AsValue<uint16_t>(); 
                NVLOGI_FMT(TAG, "{} config request: SSB Offset Point A (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.ssb_offset_point_A);
                break;
            case CONFIG_TLV_BETA_PSS:
                phy_config.ssb_table_.beta_pss = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: Beta PSS (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.beta_pss);
                break;
            case CONFIG_TLV_SSB_PERIOD:
                phy_config.ssb_table_.ssb_period = hdr->AsValue<uint8_t>(); 
                NVLOGI_FMT(TAG, "{} config request: SSB Periodicity (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.ssb_period);
                break;
            case CONFIG_TLV_SSB_SUBCARRIER_OFFSET:
                phy_config.ssb_table_.ssb_sub_carrier_offset = hdr->AsValue<uint8_t>(); 
                NVLOGI_FMT(TAG, "{} config request: SSB Subcarrier Offset (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.ssb_sub_carrier_offset);
                break;
            case CONFIG_TLV_MIB:
                memcpy(phy_config.ssb_table_.mib, hdr->As<char*>(), sizeof(phy_config.ssb_table_.mib)); 
                NVLOGI_FMT(TAG, "{} config request: MIB Payload Byte 0 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.mib[0]);
                NVLOGI_FMT(TAG, "{} config request: MIB Payload Byte 1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.mib[1]);
                NVLOGI_FMT(TAG, "{} config request: MIB Payload Byte 2 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.mib[2]);
                break;
            case CONFIG_TLV_SSB_MASK:
                static bool first = true;
                if(first)
                {
                    phy_config.ssb_table_.ssb_mask[0] = hdr->AsValue<uint32_t>();
                    NVLOGI_FMT(TAG, "{} config request: SSB Bit Mask(message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.ssb_mask[0]);
                }
                else
                {
                    phy_config.ssb_table_.ssb_mask[1] = hdr->AsValue<uint32_t>();
                    NVLOGI_FMT(TAG, "{} config request: SSB Bit Mask (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.ssb_mask[1]);
                }
                first = false;
                break;
            case CONFIG_TLV_SSB_PBCH_MULT_CARRIERS_IN_BAND:
                phy_config.ssb_table_.ss_pbch_multiple_carriers = hdr->AsValue<uint8_t>(); 
                NVLOGI_FMT(TAG, "{} config request: SSB PBCH Multiple Carriers In Band (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.ss_pbch_multiple_carriers);
                break;
            case CONFIG_TLV_MULTIPLE_CELLS_SS_PBCH_IN_CARRIER:
                phy_config.ssb_table_.multiple_cells_pbch = hdr->AsValue<uint8_t>(); 
                NVLOGI_FMT(TAG, "{} config request: Multiple Cells SSB In Single Carrier (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.ssb_table_.multiple_cells_pbch);
                break;
            case CONFIG_TLV_TDD_PERIOD:
                phy_config.tdd_table_.tdd_period_num = hdr->AsValue<uint8_t>(); 
                NVLOGI_FMT(TAG, "{} config request: DL UL Transmission Periodicity (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.tdd_table_.tdd_period_num);
                break;

            case CONFIG_TLV_SLOT_CONFIG: {
                    uint8_t* arrs = reinterpret_cast<uint8_t*>(hdr->val);
                    auto mu = phy_config.ssb_config_.sub_c_common;
                    auto tti = nv::mu_to_ns(phy_config.ssb_config_.sub_c_common);
                    auto slot_duration =  std::chrono::microseconds(tti);
                    auto num_slots = nv::get_duration(phy_config.tdd_table_.tdd_period_num)/std::chrono::duration<float, std::milli>(1);
                    uint16_t valid_entries = (1 << phy_config.ssb_config_.sub_c_common) * num_slots;
                    uint16_t k = 0;
                    for (int i = 0; i < valid_entries; i++) {
                        auto& slot_cfg = phy_config.tdd_table_.s_detail[i];
                        slot_cfg.max_dl_symbols = 0;
                        slot_cfg.max_ul_symbols = 0;
                        slot_cfg.start_sym_dl = -1;
                        slot_cfg.start_sym_ul = -1;
                        for (int j = 0; j < OFDM_SYMBOLS_PER_SLOT; j++) {
                            switch((arrs[k + j])) {
                                case nv::SlotConfig::DL_SLOT:
                                    slot_cfg.max_dl_symbols++;
                                    if(slot_cfg.start_sym_dl == -1)
                                        slot_cfg.start_sym_dl = j;
                                    break;
                                case nv::SlotConfig::UL_SLOT:
                                    slot_cfg.max_ul_symbols++;
                                    if(slot_cfg.start_sym_ul == -1)
                                        slot_cfg.start_sym_ul = j;
                                    break;
                            }
                        }
                        if (slot_cfg.max_dl_symbols < OFDM_SYMBOLS_PER_SLOT && slot_cfg.max_ul_symbols < OFDM_SYMBOLS_PER_SLOT) {
                            slot_cfg.type = nv::slot_type::SLOT_SPECIAL;
                        } else if (slot_cfg.max_dl_symbols == OFDM_SYMBOLS_PER_SLOT) {
                            slot_cfg.type = nv::slot_type::SLOT_DOWNLINK;
                        } else if (slot_cfg.max_ul_symbols == OFDM_SYMBOLS_PER_SLOT) {
                            slot_cfg.type = nv::slot_type::SLOT_UPLINK;
                        } else {
                            slot_cfg.type = nv::slot_type::SLOT_NONE;
                        }
                        NVLOGD_FMT(TAG, "{}: Slot type = {}, DL symbols = {}, UL symbols = {} Start DL = {} Start UL = {}", __FUNCTION__, +slot_cfg.type, slot_cfg.max_dl_symbols, slot_cfg.max_ul_symbols, slot_cfg.start_sym_dl, slot_cfg.start_sym_ul);
                        k += OFDM_SYMBOLS_PER_SLOT;
                    }
                } 
            break;
            case CONFIG_TLV_DL_GRID_SIZE:
            {
                uint16_t* tmp = hdr->As<uint16_t*>();
                phy_config.carrier_config_.dl_grid_size[0] = tmp[0];
                phy_config.carrier_config_.dl_grid_size[1] = tmp[1];
                phy_config.carrier_config_.dl_grid_size[2] = tmp[2];
                phy_config.carrier_config_.dl_grid_size[3] = tmp[3];
                phy_config.carrier_config_.dl_grid_size[4] = tmp[4];
                NVLOGI_FMT(TAG, "{} config request: DL Grid Size for numerology 0 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_grid_size[0]);
                NVLOGI_FMT(TAG, "{} config request: DL Grid Size for numerology 1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_grid_size[1]);
                NVLOGI_FMT(TAG, "{} config request: DL Grid Size for numerology 2 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_grid_size[2]);
                NVLOGI_FMT(TAG, "{} config request: DL Grid Size for numerology 3 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_grid_size[3]);
                NVLOGI_FMT(TAG, "{} config request: DL Grid Size for numerology 4 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.dl_grid_size[4]);
            }
            break;
            case CONFIG_TLV_UL_GRID_SIZE:
            {
                uint16_t* tmp = hdr->As<uint16_t*>();
                phy_config.carrier_config_.ul_grid_size[0] = tmp[0];
                phy_config.carrier_config_.ul_grid_size[1] = tmp[1];
                phy_config.carrier_config_.ul_grid_size[2] = tmp[2];
                phy_config.carrier_config_.ul_grid_size[3] = tmp[3];
                phy_config.carrier_config_.ul_grid_size[4] = tmp[4];
                NVLOGI_FMT(TAG, "{} config request: UL Grid Size for numerology 0 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_grid_size[0]);
                NVLOGI_FMT(TAG, "{} config request: UL Grid Size for numerology 1 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_grid_size[1]);
                NVLOGI_FMT(TAG, "{} config request: UL Grid Size for numerology 2 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_grid_size[2]);
                NVLOGI_FMT(TAG, "{} config request: UL Grid Size for numerology 3 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_grid_size[3]);
                NVLOGI_FMT(TAG, "{} config request: UL Grid Size for numerology 4 (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.ul_grid_size[4]);
            }
            break;
            case CONFIG_TLV_VENDOR_DIGITAL_BEAM_TABLE_PDU:
            {
                /* CPU_LARGE Buffer is used for encoding Digital Beam Table (DBT) PDU as it can handle the large size of DBT PDU when static beamforming needs to be used by L2.
                 * The DBT PDU is used for encoding the predetermined static beamforming weights for the Users that are not paired or for common channels. 
                 * For the non-paired UEs and channels for which L2 wants to use static beamforming, the beam IDs are indicated in Tx Precoding and Beamforming PDU 
                 * and corresponding weights are looked-up and encoded in the C-Plane message by the L1.*/
                if (ipc_msg.data_pool == NV_IPC_MEMPOOL_CPU_LARGE && ipc_msg.data_buf!= nullptr &&  phy_module().bf_enabled())
                {
                    // transferring the DBT info to FH
                    // int ret = phyDriver.l1_storeDBTPdu(cell_id, ipc_msg.data_buf);
                    // if(ret == -1)
                    // {
                    //     error_code = SCF_ERROR_CODE_BEAM_ID_OUT_OF_RANGE;
                    //     break;
                    // }
                    int ret = update_dbt_pdu_table_ptr(cell_id, ipc_msg.data_buf);
                    if (ret != 0) {
                        error_code = SCF_ERROR_CODE_BEAM_ID_OUT_OF_RANGE;
                        break;
                    }
#if 0
                    // Below code is added for validtion of unordered map if the is the values are stored properly in the map.
                    auto & dbt_static_weight_map_print = nv::PHY_module::static_digBeam_map();
                    for ( uint16_t digBeamIdx = 1; digBeamIdx <= numDigBeams ; digBeamIdx++)
                    {
                        beamIdx = (digBeamIdx) + (cell_id * MAX_STATIC_BFW_BEAM_ID);
                        auto dbt_iter = dbt_static_weight_map_print.find((beamIdx));G
                        if(dbt_iter != dbt_static_weight_map_print.end())
                        {
                            for (const auto& elem: dbt_iter->second.digBeam)
                            {
                                int16_t real = elem.digBeamWeightRe, imaginary = elem.digBeamWeightIm;
                                NVLOGD_FMT(TAG, "{}: cell_id={} Index={}, bemidx={} flag={} real = {}, imag = {}", __FUNCTION__,
                                            cell_id, beamIdx, dbt_iter->first, dbt_iter->second.beamIdxIQSentInCplane,
                                            static_cast<int16_t>(real),
                                            static_cast<int16_t>(imaginary));
                            }
                        }
                        else
                        {
                            NVLOGW_FMT(TAG, "{}: No entry found for BeamIdx={} in DBT PDU Table", __FUNCTION__, digBeamIdx);
                        }
                    }
#endif
                }
                else
                {
                    NVLOGW_FMT(TAG, "{}: Beamforming is not enabled. Hence not storing the DBT PDU for static Beamforming", __FUNCTION__);
                }
            }
            break;
            case CONFIG_TLV_VENDOR_PRECODING_MATRIX:
            {
                scf_fapi_pm_pdu_t& pm_pdu = *hdr->As<scf_fapi_pm_pdu_t*>();

                if (phy_module().pm_enabled() && pm_pdu.pmi_idx != 0) {
                    uint32_t pmidx = pm_pdu.pmi_idx | cell_id << 16;
                    auto layers = pm_pdu.num_layers;
                    auto ports = pm_pdu.num_ant_ports;
                    prc_wt_re_im_t* pm_start = &pm_pdu.prc_wt_re_im[0];
                    auto & pm_weight_map = nv::PHY_module::pm_map();
                    // Disabling memtrace here to suppress dynamic allocation due to pm_weight_map.insert. This is only happening at the start up phase per cell per pmi index
                    MemtraceDisableScope md; // disable memtrace while this variable is in scope
                    pm_weight_map.insert(std::make_pair(pmidx, pm_weights_t{layers, ports, cuphyPmW_t()}));
                    auto& weights = pm_weight_map[pmidx];
                    weights.weights.nPorts = ports;
                    for ( uint16_t i = 0 ; i < layers ; i++) {
                        for (uint16_t j = 0; j < ports; j++) {
                            std::size_t index = static_cast<std::size_t>(i * ports) + j;
                            prc_wt_re_im_t&pm = pm_start[index];
                            int16_t real = pm.prc_wt_re, imaginary = pm.prc_wt_im;
                            half re = *reinterpret_cast<half*>(&real);
                            half im = *reinterpret_cast<half*>(&imaginary);
                            weights.weights.matrix[index] = __half2(re,im);
#if DBG_PRECODER
                            NVLOGC_FMT(TAG, "index {}. real = {} imag = {}", index,  static_cast<float>(weights.weights.matrix[index].x), static_cast<float>(weights.weights.matrix[index].y));
#endif
                        }
                    }
                    if (!first_config_req && phy_module().config_options().duplicateConfigAllCells) {
                        first_config_req_pmidxes.push_back(pmidx);
                    }
                }
            }
            break;
            case CONFIG_TLV_RSSI_MEAS:
            {
                uint8_t rssi_value = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: RSSI Measurement unit (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), rssi_value);
                
                setRssiMeasurement(rssi_value);
                if(getRssiMeasurement() > 1)
                {
                    NVLOGW_FMT(TAG, "{}: Invalid RSSI config value {} - setting to 0", __FUNCTION__, rssi_value);
                    setRssiMeasurement(0);
                }
            }
            break;
            // Unsupported values
            case CONFIG_TLV_FREQ_SHIFT_7P_5KHZ:
            {
                NVLOGW_FMT(TAG, "{} config request: FREQ_SHIFT_7P_5KHZ (message ID {:X}) not supported", __FUNCTION__, static_cast<int>(hdr->tag)); break;
            }
            break;
            case CONFIG_TLV_BCH_PAYLOAD:
            {
                NVLOGW_FMT(TAG, "{} config request: BCH_PAYLOAD (message ID {:X}) not supported", __FUNCTION__, static_cast<int>(hdr->tag)); break;
            }
            break;
            case CONFIG_TLV_BEAM_ID:
            {
                NVLOGW_FMT(TAG, "{} config request: BEAM_ID (message ID {:X}) not supported", __FUNCTION__, static_cast<int>(hdr->tag)); break;
            }
            break;
#ifdef SCF_FAPI_10_04
            case CONFIG_TLV_RSRP_MEAS: {
                uint8_t rsrp_value = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: RSRP Measurement unit (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), rsrp_value);
                setRsrpMeasurement(rsrp_value);
                if(getRsrpMeasurement() > 1)
                {
                    NVLOGW_FMT(TAG, "{}: Invalid RSRP Measurement config value {}, setting to 0.", __FUNCTION__, rsrp_value);
                    setRsrpMeasurement(0);
                }
            }
            break;
            /* FAPI 10.04 allows generation of more than one message instance per slot for UL FAPI Messages.
             * If L2 wants to split SRS reports into multiple SRS IND's, then L2 can configure this TLV and encode Index=5 (SRS.Indication) as value=2 (MULTI_MSG_INSTANCE_PER_SLOT).
             * This is needed if one NVIPC buffer size is not enough to send all requested SRS PDU reports in one SRS IND.
             * If Index=5 (SRS.Indication) is not set to value=2 (MULTI_MSG_INSTANCE_PER_SLOT) and L1 is not able to accommodate all the requested SRS PDU reports in one SRS IND,
             * then L1 will send an ERROR IND to L2 to indicate only one partial SRS.IND will follow. */
            case CONFIG_TLV_INDICATION_INSTANCES_PER_SLOT:{
                uint8_t *tmp = hdr->As<uint8_t*>();
                if(tmp ==nullptr)
                {
                    NVLOGC_FMT(TAG, "{} config request: Indication Instances Per Slot IndPerSlotPtr is NULL", __FUNCTION__);
                }
                else
                {
                    for (int i = 0; i < nv::MAX_IND_INDEX; i++)
                    {
                        if(tmp[i] > phy_module().get_phy_config().indication_instances_per_slot[i])
                            phy_module().get_phy_config().indication_instances_per_slot[i] = tmp[i];
                        NVLOGI_FMT(TAG, "{} config request: Indication Instances Per Slot IndPerSlotPtr[{}] = {}", __FUNCTION__, i, tmp[i]);
                    }
                }
            }
            break;
            case CONFIG_TLV_UCI_CONFIG: {
                auto *uci_buf = hdr->As<uint8_t*>();
                uint16_t numUci2Maps = *reinterpret_cast<uint16_t*>(uci_buf);
                uci_buf+= sizeof(uint16_t);
                auto offset = 0;
                auto mapOffset = 0;
                auto destMapBuf = static_cast<uint16_t*>(csi2MapCpuBuffer.get());
                auto destMapParamBuf = static_cast<cuphyCsi2MapPrm_t*>(csi2MapParamsCpuBuffer.get());
                for (uint16_t i = 0; i < numUci2Maps; i++) {
                    uint8_t numPart1Params = *(uci_buf + offset);
                    offset += sizeof(uint8_t);
                    auto sizesPart1Part1Params = reinterpret_cast<uint8_t*>(uci_buf + offset);
                    auto sigma = 0;
                    sigma = std::accumulate(sizesPart1Part1Params, sizesPart1Part1Params + numPart1Params, 0);
                    offset += sizeof(uint8_t) * numPart1Params;
                    size_t uciMapSize = 1 << sigma;
                    auto map = reinterpret_cast<uint16_t*>(uci_buf + offset);
                    std::copy(map, map + uciMapSize, destMapBuf + mapOffset);
                    destMapParamBuf[i].csi2MapSize = uciMapSize;
                    destMapParamBuf[i].csi2MapStartIdx = mapOffset;
                    NVLOGD_FMT(TAG, "{}: numUci2Maps {} i {} , numPart1Params {} sigma {} uciMapSize {} mapParams[ csi2MapSize {} csi2MapStartIdx {}] map[ first 0x{:04X} last 0x{:04X}] offset 0x{:02X}", __FUNCTION__, numUci2Maps, i, numPart1Params, sigma, uciMapSize, 
                        destMapParamBuf[i].csi2MapSize, destMapParamBuf[i].csi2MapStartIdx, *(destMapBuf + destMapParamBuf[i].csi2MapStartIdx), *(destMapBuf+ destMapParamBuf[i].csi2MapSize - 1), offset ) ;
                    mapOffset+= uciMapSize;
                    offset+= sizeof(uint16_t) * uciMapSize;
                }
                nCsi2Maps = numUci2Maps;
            }
            break;
#endif
            case CONFIG_TLV_VENDOR_NOISE_VAR_MEAS: {
                uint8_t pn_value = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PE Noise Measurement unit (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), pn_value);
                setPnMeasurement(pn_value);
                if(getPnMeasurement() > 1)
                {
                    NVLOGW_FMT(TAG, "Invalid PE Noise Measurement config value {}, setting to 0.", pn_value);
                    setPnMeasurement(0);
                }
            }
            break;
#if 0
            case CONFIG_TLV_VENDOR_PF_01_INTERFERENCE_MEAS: {
                uint8_t pf_01_interference_meas = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PF_01 Interference Measurement unit (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), pf_01_interference_meas);
                if(pf_01_interference_meas > 1)
                {
                    NVLOGW_FMT(TAG, "{}: Invalid PF_01 Interference Measurement config value {}, setting to invalid 0.", __FUNCTION__, pf_01_interference_meas);
                    pf_01_interference_meas = 0;
                }
            }
            break;
#endif
            case CONFIG_TLV_VENDOR_PF_234_INTERFERENCE_MEAS: {
                uint8_t pf_234_interference_meas = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PF_234 Interference Measurement unit (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), pf_234_interference_meas);
                setPf234Interference(pf_234_interference_meas);
                if(getPf234Interference() > 1)
                {
                    NVLOGW_FMT(TAG, "{}: Invalid PF_234 Interference Measurement config value {}, setting to 0.", __FUNCTION__, pf_234_interference_meas);
                    setPf234Interference(0);
                }
            }
            break;
            case CONFIG_TLV_VENDOR_PRACH_INTERFERENCE_MEAS: {
                uint8_t prach_interference_meas = hdr->AsValue<uint8_t>();
                NVLOGI_FMT(TAG, "{} config request: PRACH Interference Measurement unit (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), prach_interference_meas);
                setPrachInterference(prach_interference_meas);
                if(getPrachInterference() > 1)
                {
                    NVLOGW_FMT(TAG, "Invalid PRACH Interference Measurement config value {}, setting to invalid 0.", prach_interference_meas);
                    setPrachInterference(0);
                }
            }
            break;
            case CONFIG_TLV_VENDOR_NUM_TX_PORT:
            {
                if(muMIMO_enable_flag)
                {
                    phy_config.carrier_config_.num_tx_port = hdr->AsValue<uint16_t>();
                    phy_cell_params.nTxAnt = phy_config.carrier_config_.num_tx_port;
                    NVLOGI_FMT(TAG, "{} config request: Number of Tx Ports (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.num_tx_port);
                    txPortTlvPresent = true;
                }
            }
            break;
            case CONFIG_TLV_VENDOR_NUM_RX_PORT:
            {
                if(muMIMO_enable_flag)
                {
                    phy_config.carrier_config_.num_rx_port = hdr->AsValue<uint16_t>();
                    phy_cell_params.nRxAnt = phy_config.carrier_config_.num_rx_port;
                    NVLOGI_FMT(TAG, "{} config request: Number of Rx Ports (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), phy_config.carrier_config_.num_rx_port);
                    rxPortTlvPresent = true;
                }
            }
            break;
            case CONFIG_TLV_VENDOR_CHAN_SEGMENT: 
            {
                if (mCh_segment_proc_enable != 0) {
                    auto& ch_segment = *hdr->As<scf_channel_segment_t*>();
                    auto nseg = ch_segment.nPduSegments;
                    uint8_t* segment_buf = ch_segment.payload;
                    NVLOGD_FMT(TAG, "{} config request: Number of Channel Segments (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), nseg);

                    for (uint8_t i = 0; i < ch_segment.nPduSegments; i++) {
                        auto seg_info = reinterpret_cast<scf_channel_segment_info_t*>(segment_buf);
                        auto type = seg_info->type;
                        auto chan_start_offset = seg_info->chan_start_offset;
                        auto chan_duration = seg_info->chan_duration;
                        NVLOGD_FMT(TAG, "{} config request: Channel Segment Type {} Chan Start Offset {} Chan Duration {}", __FUNCTION__, type, chan_start_offset, chan_duration);
                        auto& ch_seg_timeline =  phy_module().get_ch_timeline(type << 16 | phy_config.cell_config_.carrier_idx);
                        auto& ch_seg_indexes = phy_module().get_ch_proc_indexes(type << 16 | phy_config.cell_config_.carrier_idx);
                        ch_seg_timeline[ch_seg_indexes[type]] = std::make_pair(chan_start_offset, chan_duration);
                        ch_seg_indexes[type]++;
                        segment_buf += sizeof(scf_channel_segment_info_t);
                    }
                }
            }
            break;
#ifdef SCF_FAPI_10_04
            /* L1 Reserves a fixed number of SRS Chest buffers for the entire system during bring-up of the cell using the paramerter total_num_srs_chest_buffers in the cuphycontroller_xxx.yaml file.
             * L2 reserves a fixed number of SRS Chest buffers per cell using the TLV CONFIG_TLV_VENDOR_NUM_SRS_CHEST_BUFFERS in Cell Config Request.
             * This number of buffers is reserved for the entire cell life cycle and will be released on cell stop. */
            case CONFIG_TLV_VENDOR_NUM_SRS_CHEST_BUFFERS:
            {
                srsChestTlvPresent = true;
                srsChest_buff_size = hdr->AsValue<uint32_t>();
                NVLOGD_FMT(TAG, "{} config request: Number of SRS Chest Buffers (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), srsChest_buff_size);

                if(!enable_srs_flag)
                {
                    NVLOGW_FMT(TAG, "{}: cell_id={} received CONFIG_TLV_VENDOR_NUM_SRS_CHEST_BUFFERS while enable_srs_flag=false so no buffer allocation is done.", __func__, cell_id);
                    break;
                }

                NVLOGD_FMT(TAG, "SRS enabled and TLV present - proceeding with SRS chest buffer allocation");
                
                if ((!muMIMO_enable_flag) && (srsChest_buff_size > MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL))
                {
                    NVLOGI_FMT(TAG, "{}: SRS chest buffer size {} exceeds maximum allowed {} for non-muMIMO configuration - clamping to maximum", __FUNCTION__, srsChest_buff_size, MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL);
                    srsChest_buff_size = MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL;
                }
                
                bool ret = phyDriver.allocSrsChesBuffPool(SCF_FAPI_CONFIG_REQUEST, cell_id, srsChest_buff_size);
                if(!ret)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SRS chest buffer allocation failed for size {} - sending error response", srsChest_buff_size);
                    error_code = SCF_ERROR_CODE_MSG_INVALID_CONFIG;
                    break;
                }
                
                setSrsChestBuffSize(srsChest_buff_size);
            }
            break;
#endif
            case CONFIG_TLV_VENDOR_PUSCH_AGGR_FACTOR:
            {
                nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
                uint8_t pusch_aggr_factor = hdr->AsValue<uint8_t>();
                // Set pusch_aggr_factor to the position of the most significant bit set, plus 1
                auto get_msb_pos = [](const uint8_t val) {
                    uint8_t msb_pos{};
                    uint8_t temp_val = val;
                    while (temp_val) {
                        msb_pos++;
                        temp_val >>= 1;
                    }
                    return msb_pos;
                };
                pusch_aggr_factor = (pusch_aggr_factor == 0) ? 1 : get_msb_pos(pusch_aggr_factor);
                if (phyDriver.l1_get_split_ul_cuda_streams()) {
                    pusch_aggr_factor = 1;
                }
                else if (pusch_aggr_factor > 2) {
                    NVLOGW_FMT(TAG, "{}: Invalid PUSCH Aggregation Factor config value {}, setting to 1.", __FUNCTION__, pusch_aggr_factor);
                    pusch_aggr_factor = 1;
                }
                NVLOGI_FMT(TAG, "{} config request: PUSCH Aggregation Factor (message ID {:X}) value {}", __FUNCTION__, static_cast<int>(hdr->tag), pusch_aggr_factor);
                setPuschAggrFactor(pusch_aggr_factor);
            }
            break;
            default:
            {
                NVLOGW_FMT(TAG, "{} config request: Unknown TLV {}", __FUNCTION__, static_cast<int>(hdr->tag));
            }
        }
        // Round up TLV length to 4-byte boundary according to specs
        body_ptr += sizeof(scf_fapi_tl_t) + ((hdr->length + 3) / 4) * 4;
        tlvs--;
    }

    if(!srsChestTlvPresent)
    {
        if(enable_srs_flag)
        {
            /* L1 Reserves a fixed number of SRS Chest buffers for the entire system during bring-up of the cell using the paramerter total_num_srs_chest_buffers in the cuphycontroller_xxx.yaml file.
             * If L2 does not send a TLV for NUM_SRS_CHEST_BUFFERS in Cell Config Request then L1 will reserve a default fixed number of SRS Chest buffers per cell.
             * These buffers are reserved for the entire cell life cycle and will be released on cell stop. */
            srsChest_buff_size = muMIMO_enable_flag ? MAX_SRS_CHEST_BUFFERS_PER_CELL : MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL;
            NVLOGI_FMT(TAG, "{}: SRS chest buffer size TLV absent - using default size {}", __FUNCTION__, srsChest_buff_size);
            
            bool ret = phyDriver.allocSrsChesBuffPool(SCF_FAPI_CONFIG_REQUEST, cell_id, srsChest_buff_size);
            if (!ret)
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer allocation failed for size {} - sending error response", __FUNCTION__, srsChest_buff_size);
                error_code = SCF_ERROR_CODE_MSG_INVALID_CONFIG;
            }
            else
            {
                setSrsChestBuffSize(srsChest_buff_size);
            }
        }
        else
        {
            NVLOGI_FMT(TAG, "{}: SRS chest buffer TLV absent and SRS disabled - no buffer allocation", __FUNCTION__);
        }
    }

    /* In case muMIMO flag is set but the Tx port and Rx port TLVs are absent then set the number of Tx and Rx ports to default value */
    if(muMIMO_enable_flag && !txPortTlvPresent)
    {
        phy_cell_params.nTxAnt = (phy_cell_params.nTxAnt > 4) ? NUM_TX_PORT : phy_cell_params.nTxAnt;
        NVLOGW_FMT(TAG, "{}: muMIMO flag is enabled but txPortTlvPresent absent. Setting nTxAnt to {}", __FUNCTION__, phy_cell_params.nTxAnt);
    }
    if(muMIMO_enable_flag && !rxPortTlvPresent)
    {
        phy_cell_params.nRxAnt = (phy_cell_params.nRxAnt > 4) ? NUM_RX_PORT : phy_cell_params.nRxAnt;
        NVLOGW_FMT(TAG, "{}: muMIMO flag is enabled but rxPortTlvPresent absent. Setting nRxAnt to {}", __FUNCTION__, phy_cell_params.nRxAnt);
    }

    if(error_code == SCF_ERROR_CODE_MSG_OK)
    {
        error_code = create_cell_configs();
        // We should track how many TLVs were valid above, but for now, we just report that any CONFIG request is valid
        NVLOGC_FMT(TAG, "{}: create_cell_configs for cell_id={} phy_cell_id={} returned error_code={}", __FUNCTION__, cell_id, phy_config.cell_config_.phy_cell_id, error_code);
    }
    send_cell_config_response(cell_id, error_code);

    if (error_code == SCF_ERROR_CODE_MSG_OK)
    {
        state = fapi_state_t::FAPI_STATE_CONFIGURED;
    }

    phy_module().create_cell_update_call_back();
}

void phy::on_cell_start_request(const int32_t cell_id)
{
    NVLOGC_FMT(TAG, "{}: Cell {} Received START.request, sending START.response... phy_cell_id={}", __FUNCTION__, cell_id, phy_config.cell_config_.phy_cell_id);
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    if (state != fapi_state_t::FAPI_STATE_CONFIGURED) {
        NVLOGW_FMT(TAG, "{}: START.req rejected - state not CONFIGURED (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        nv::phy_mac_transport& transport = phy_module().transport(phy_config.cell_config_.carrier_idx);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }
        auto fapi = add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, phy_config.cell_config_.carrier_idx, false);
        auto rsp = reinterpret_cast<scf_fapi_error_ind_t*>(fapi);
        rsp->sfn = 0;
        rsp->slot = 0;
        rsp->msg_id = SCF_FAPI_START_REQUEST;
        rsp->err_code = SCF_ERROR_CODE_MSG_INVALID_STATE;
        transport.tx_send(msg_desc);
        transport.notify(IPC_NOTIFY_VALUE);
        metrics_.incr_tx_packet_count(SCF_FAPI_ERROR_INDICATION);
        return;
    }

    phyDriver.l1_cell_start(phy_config.cell_config_.phy_cell_id);
    phy_module().send_call_backs();
    phy_module().set_tti_flag(true);

    state = fapi_state_t::FAPI_STATE_RUNNING;
    phy_module().incr_active_cells();
    phy_module().transport(cell_id).set_started_cells_mask(cell_id, true);
#ifdef ENABLE_L2_SLT_RSP
    phy_module().set_active_cell_bitmap(phy_config.cell_config_.carrier_idx);
#endif
    bfw_buffer_info buffer_info;
    phyDriver.l1_bfw_coeff_retrieve_buffer(cell_id, &buffer_info);
    phy_module().set_bfw_coeff_buff_info(cell_id, &buffer_info);

    if(phyDriver.l1_staticBFWConfigured(cell_id))
    {
        int dbtStoreStatus = phyDriver.l1_resetDBTStorage(cell_id);
        if(dbtStoreStatus == -1)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}:l1_resetDBTStorage failed for cell_id={} Sending error indication", __FUNCTION__, cell_id);
            nv::phy_mac_transport& transport = phy_module().transport(phy_config.cell_config_.carrier_idx);
            nv::phy_mac_msg_desc msg_desc;
            if (transport.tx_alloc(msg_desc) < 0)
            {
                return;
            }
            auto fapi = add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, phy_config.cell_config_.carrier_idx, false);
            auto rsp = reinterpret_cast<scf_fapi_error_ind_t*>(fapi);
            rsp->sfn = 0;
            rsp->slot = 0;
            rsp->msg_id = SCF_FAPI_START_REQUEST;
            rsp->err_code = SCF_ERROR_CODE_BEAM_ID_OUT_OF_RANGE;
            transport.tx_send(msg_desc);
            transport.notify(IPC_NOTIFY_VALUE);
        }
    }
    if(get_enable_srs_info())
    {
        uint32_t srsChest_buff_size = getSrsChestBuffSize();
        NVLOGI_FMT(TAG, "{}: SRS chest buffer requested for size={} from START.req", __FUNCTION__, srsChest_buff_size);
        bool ret = phyDriver.allocSrsChesBuffPool(SCF_FAPI_START_REQUEST, cell_id, srsChest_buff_size);
        if (!ret)
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer allocation failed size={} - sending error response", __FUNCTION__, srsChest_buff_size);
            nv::phy_mac_transport& transport = phy_module().transport(phy_config.cell_config_.carrier_idx);
            nv::phy_mac_msg_desc msg_desc;
            if (transport.tx_alloc(msg_desc) < 0)
            {
                return;
            }
            auto fapi = add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, phy_config.cell_config_.carrier_idx, false);
            auto rsp = reinterpret_cast<scf_fapi_error_ind_t*>(fapi);
            rsp->sfn = 0;
            rsp->slot = 0;
            rsp->msg_id = SCF_FAPI_START_REQUEST;
            rsp->err_code = SCF_ERROR_CODE_MSG_INVALID_CONFIG;
            transport.tx_send(msg_desc);
            transport.notify(IPC_NOTIFY_VALUE);
            metrics_.incr_tx_packet_count(SCF_FAPI_ERROR_INDICATION);
        }
    }
}


void phy::on_cell_stop_request(const int32_t cell_id)
{
    NVLOGI_FMT(TAG, "{}: Cell {} received STOP.req, sending STOP.response... carrier_id={}", __FUNCTION__, phy_config.cell_config_.phy_cell_id, phy_config.cell_config_.carrier_idx);

    if (state != fapi_state_t::FAPI_STATE_RUNNING) {
        NVLOGW_FMT(TAG, "{}: STOP.req rejected  state not RUNNING (current={})", __FUNCTION__, static_cast<uint32_t>(state.load()));
        nv::phy_mac_transport& transport = phy_module().transport(phy_config.cell_config_.carrier_idx);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }
        auto fapi = add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, phy_config.cell_config_.carrier_idx, false);
        auto rsp = reinterpret_cast<scf_fapi_error_ind_t*>(fapi);
        rsp->sfn = 0;
        rsp->slot = 0;
        rsp->msg_id = SCF_FAPI_STOP_REQUEST;
        rsp->err_code = SCF_ERROR_CODE_MSG_INVALID_STATE;
        transport.tx_send(msg_desc);
        transport.notify(IPC_NOTIFY_VALUE);
        metrics_.incr_tx_packet_count(SCF_FAPI_ERROR_INDICATION);
        return;
    }

    nv::PHYDriverProxy::getInstance().l1_cell_stop(phy_config.cell_config_.phy_cell_id);

    nv::PHYDriverProxy::getInstance().deAllocSrsChesBuffPool(cell_id);

    state = fapi_state_t::FAPI_STATE_CONFIGURED;

    // Carrier index for STOP.ind and bitmap update; must not read hdr after tx_send (buffer may be reused).
    const int32_t stopped_carrier_idx = phy_config.cell_config_.carrier_idx;

    nv::phy_mac_transport& transport = phy_module().transport(phy_config.cell_config_.carrier_idx);
    nv::phy_mac_msg_desc msg_desc;
    if (transport.tx_alloc(msg_desc) < 0)
    {
        return;
    }
    scf_fapi_header_t *hdr;
    hdr = reinterpret_cast<scf_fapi_header_t*>(msg_desc.msg_buf);

    hdr->message_count     = 1;
    hdr->handle_id         = stopped_carrier_idx;

    auto *body = reinterpret_cast<scf_fapi_body_header_t*>(hdr->payload);
    body->type_id          = SCF_FAPI_STOP_INDICATION;
    body->length           = 0;

    msg_desc.msg_id = SCF_FAPI_STOP_INDICATION;
    msg_desc.cell_id = stopped_carrier_idx;
    msg_desc.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);
    msg_desc.data_len = 0;
    transport.tx_send(msg_desc);
    transport.notify(IPC_NOTIFY_VALUE);
    metrics_.incr_tx_packet_count(SCF_FAPI_STOP_INDICATION);
    phy_module().set_tti_flag(false);
    phy_module().decr_active_cells();
    phy_module().transport(cell_id).set_started_cells_mask(cell_id, false);
#ifdef ENABLE_L2_SLT_RSP
    phy_module().unset_active_cell_bitmap(static_cast<uint16_t>(stopped_carrier_idx));
#endif
}

inline void ul_ind_ta(uint16_t& ta, float &rawTA, float& ta_offset_usec_, uint8_t mu, uint16_t ta_max, uint8_t ta_base_offset) {
    // Use signed arithmetic to avoid overflow
    int32_t tmp = std::round((rawTA - ta_offset_usec_) * TA_BASE_SCALE * (1 << mu)) + ta_base_offset;
    ta = static_cast<uint16_t>(std::clamp(tmp, 0, static_cast<int32_t>(ta_max)));
}

void phy::send_rach_indication(slot_command_api::slot_indication& slot,
                                const prach_params& params,
                                const uint32_t* num_detectedPrmb,
                                const void* prmbIndex_estimates,
                                const void* prmbDelay_estimates,
                                const void* prmbPower_estimates,
                                const void* ant_rssi,
                                const void* rssi,
                                const void* interference)
{
    if (num_detectedPrmb == nullptr || prmbIndex_estimates == nullptr || prmbDelay_estimates == nullptr||
            prmbPower_estimates == nullptr || ant_rssi == nullptr || rssi == nullptr)
    {
        NVLOGW_FMT(TAG, "{}: Insufficient parameters for RACH indication", __FUNCTION__);
        return;
    }

    std::size_t total_rach_occasions = params.nOccasion;
    uint32_t curr = 0;
    uint32_t interferenceIndex = 0;
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();

    NVLOGD_FMT(TAG, "{}: Total RACH occasions = {}", __FUNCTION__, total_rach_occasions);

    for(curr = 0; curr < total_rach_occasions; )
    {
        auto numPrmb = num_detectedPrmb[curr];

        if(!numPrmb)
        {
            NVLOGI_FMT(TAG, "{}: SFN {}.{} No Preambles detected", __FUNCTION__, slot.sfn_, slot.slot_);
            curr++;
            continue;
        }

        NVLOGI_FMT(TAG, "{}: SFN {}.{} cell_index={} phy_cell_id={} Yes Preambles detected, num_detectedPrmb: {}", 
                   __FUNCTION__, slot.sfn_, slot.slot_, 
                   params.cell_index_list[curr], params.phy_cell_index_list[curr], numPrmb);

        nv::phy_mac_transport& transport = phy_module().transport(params.cell_index_list[curr]);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }
        msg_desc.cell_id = params.cell_index_list[curr];
        auto fapi = add_scf_fapi_hdr<scf_fapi_rach_ind_t>(msg_desc, SCF_FAPI_RACH_INDICATION, params.cell_index_list[curr], false);

        int8_t numPdus = 0;
        int32_t cell_id = params.phy_cell_index_list[curr];
        while((curr+numPdus) < total_rach_occasions && cell_id == params.phy_cell_index_list[curr+numPdus])
        {
            numPdus++;
        }

        auto& hdr = *reinterpret_cast<scf_fapi_rach_ind_t*>(fapi);
        hdr.num_pdus = numPdus;
        hdr.slot = slot.slot_;
        hdr.sfn = slot.sfn_;

        nv::phy_mac_msg_desc interference_msg_desc;
        if (transport.tx_alloc(interference_msg_desc) < 0)
        {
            return;
        }
        interference_msg_desc.cell_id = params.cell_index_list[curr];
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(params.cell_index_list[curr]);
        float ul_configured_gain = mplane.ul_gain_calibration;

        auto interference_fapi = add_scf_fapi_hdr<scf_fapi_prach_interference_ind_t>(interference_msg_desc, SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION, params.cell_index_list[curr], false);

        auto& interference_hdr = *reinterpret_cast<scf_fapi_prach_interference_ind_t*>(interference_fapi);
        interference_hdr.slot = slot.slot_;
        interference_hdr.sfn = slot.sfn_;
        interference_hdr.num_meas = 0;

        uint32_t offset = 0;
        uint8_t* data = reinterpret_cast<uint8_t*>(&hdr.pdu_info[0]);
        int i = 0;
        for(int i = 0; i < hdr.num_pdus; ++i)
        {
            auto& prach_pdu = *reinterpret_cast<scf_fapi_prach_ind_pdu_t*>(&data[offset]);
            prach_pdu.phys_cell_id = params.phy_cell_index_list[curr + i];
            prach_pdu.symbol_index = params.startSymbols[curr+i];
            prach_pdu.slot_index = slot.slot_;
            prach_pdu.freq_index = params.freqIndex[curr+i];
            const float * avg_rssi = static_cast<const float*>(rssi);
            NVLOGD_FMT(TAG, "{}: avg_rssi ={:5.6f}", __FUNCTION__, avg_rssi[curr+i]);
            
            // TODO: FAPI 10.04: Value: 0->170000 representing -140dB to 30dB with a step size of 0.001dB
            // But FAPI 10.04 define avgRssi as uint16_t, it should be uint32_t
            // prach_pdu.avg_rssi = avg_rssi[curr+i] * 1000 + 140000;

            // FAPI 10.02: Value: 0->254 representing -63dB to 30dB with a step size of 0.5dB
            prach_pdu.avg_rssi = avg_rssi[curr+i] * 2 + 128 + 0.5; // Add 0.5 to round during float to int
            prach_pdu.avg_snr = 0xff;

            numPrmb = num_detectedPrmb[curr+i];
            prach_pdu.num_preamble = numPrmb;

            if(getPrachInterference())
            {
                auto& prach_interference_pdu = *reinterpret_cast<scf_fapi_prach_interference_t*>(&interference_hdr.meas_info[i]);
                prach_interference_pdu.phyCellId = params.phy_cell_index_list[curr+i];
                prach_interference_pdu.freqIndex = params.freqIndex[curr+i];
                const float * interference_value = static_cast<const float*>(interference);
                float meas = interference_value[curr+i];
                float meas_dbm = meas - ul_configured_gain; // Applied UL configured gain to raw interference measurement
                prach_interference_pdu.meas = std::round(std::clamp(10 * (meas_dbm + 152.0), 0.0, 1520.0));
                NVLOGD_FMT(TAG, "{}: PRACH Interference nOcc={} raw interference={} ul_configured_gain={} FAPI value={}", __FUNCTION__,
                    (curr+i), meas, ul_configured_gain, static_cast<int>(prach_interference_pdu.meas));
                interference_fapi->length += sizeof(scf_fapi_prach_interference_t);
                interference_hdr.num_meas ++;
            }

            const uint32_t * index_est = static_cast<const uint32_t*>(prmbIndex_estimates) + (curr+i)*PRACH_MAX_NUM_PREAMBLES;
            const float * delay_time_est = static_cast<const float*>(prmbDelay_estimates) + (curr+i)*PRACH_MAX_NUM_PREAMBLES;
            const float * peak_dest = static_cast<const float*>(prmbPower_estimates) + (curr+i)*PRACH_MAX_NUM_PREAMBLES;

            float step = nv::STEP_CONST/(1 << params.mu);
            for (size_t prmb_idx = 0 ; prmb_idx < numPrmb; prmb_idx++)
            {
                auto& preamble = *(reinterpret_cast<scf_fapi_prach_preamble_info_t*>(&prach_pdu.preamble_info[prmb_idx]));
                preamble.preamble_index = index_est[prmb_idx];

                uint16_t ta = UINT16_MAX;
                float rawTA = delay_time_est[prmb_idx]/TA_MICROSECOND_TO_SECOND;
                ul_ind_ta(ta, rawTA, prach_ta_offset_usec_ , phy_module().get_mu_highest(), TA_MAX_PRACH, 0);
                preamble.timing_advance = ta;

                preamble.preamble_power = 1000 * (10 * std::log10(peak_dest[prmb_idx]) + 140 - ul_configured_gain) + 0.5; // Add 0.5 to round during float to int

                NVLOGI_FMT(TAG, "{}: RACH_INDICATION: pdu idx [{}] prmb_idx [{}] preambleIndex={}, PreamblePwr={}, TA={} raw TA {} us peak={:5.6f} preamblePower={:5.6f}", __FUNCTION__,
                       i, prmb_idx, preamble.preamble_index, static_cast<int>(preamble.preamble_power), static_cast<unsigned>(preamble.timing_advance), rawTA, peak_dest[prmb_idx], 10 * std::log10(peak_dest[prmb_idx]));
            }
            offset += sizeof(scf_fapi_prach_ind_pdu_t) + prach_pdu.num_preamble * sizeof(scf_fapi_prach_preamble_info_t);
        }
        fapi->length += offset;
        msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);

        transport.tx_send(msg_desc);
        transport.notify(1);
        metrics_.incr_tx_packet_count(SCF_FAPI_RACH_INDICATION);

        if(getPrachInterference())
        {
            interference_msg_desc.msg_len = interference_fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
            transport.tx_send(interference_msg_desc);
            transport.notify(1);
        }
        else
           transport.tx_release(interference_msg_desc);

        // Advance to next cell's occasions
        curr += numPdus;
    }
}

static void pucch_ul_cqi_f234(scf_fapi_pucch_format_hdr& pdu_hdr, const cuphyPucchDataOut_t& cuphy_out, const cuphyPucchF234OutOffsets_t& uci, float ul_configured_gain)
{
    float adjusted_snr;
    float raw_snr = cuphy_out.pSinr[uci.snrOffset];
#ifdef SCF_FAPI_10_04
    NVLOGD_FMT(TAG, "{}: Raw SINR={} RSRP={} ul_configured_gain={}", __FUNCTION__, raw_snr, cuphy_out.pRsrp[uci.snrOffset], ul_configured_gain);

    adjusted_snr = (raw_snr < -65.534) ? (-65.534) : ((raw_snr > 65.534) ? 65.534 : raw_snr);
    pdu_hdr.measurement.ul_sinr_metric = (adjusted_snr * 500);
    float rsrp_dbm = cuphy_out.pRsrp[uci.snrOffset] - ul_configured_gain; // Applied UL configured gain to raw RSRP measurement
    //clamp between -140 and -12. Add 140 to the result and multiply by 10 to get a value between 0 to 1280
    rsrp_dbm = (rsrp_dbm < -140) ? -140 : ((rsrp_dbm > -12) ? -12 : rsrp_dbm);
    pdu_hdr.measurement.rsrp = std::round(rsrp_dbm + 140.0) * 10;
#else
    NVLOGD_FMT(TAG, "{}: Raw SINR={} ul_configured_gain={}", __FUNCTION__, raw_snr, ul_configured_gain);
    adjusted_snr = std::clamp(raw_snr, -64.0f, 63.0f);
    uint16_t adjusted_snr_int = (uint16_t)((adjusted_snr + 64.0f) * 2);
    pdu_hdr.ul_cqi = adjusted_snr_int;
#endif
}

inline void pucch_rssi_f234(uint8_t rssiMeasurement, uint16_t conf_rssi, float ul_configured_gain, scf_fapi_pucch_format_hdr& pdu_hdr, const cuphyPucchDataOut_t& cuphyOut, const cuphyPucchF234OutOffsets_t& uci)
{
    float rssi = cuphyOut.pRssi[uci.RSSIoffset];
    cuphyPucchF234OutOffsets_t* offsets;

    switch(rssiMeasurement)
    {
        case 0: // Do not report RSSI
        {
#ifdef SCF_FAPI_10_04
            pdu_hdr.measurement.rssi = conf_rssi;
#else
            pdu_hdr.rssi = conf_rssi; // 0xffff by default unless config file specifies otherwise
#endif
        }
        break;
        case 1: // dBm = ULConfiguredGainConstant + dBFS 128dBm to 0dB, with a step size of 0.1dB
        {
            NVLOGD_FMT(TAG, "{}: Raw RSSI={} ul_configured_gain={}", __FUNCTION__, rssi, ul_configured_gain);
            float rssi_dbm = rssi - ul_configured_gain; // Applied UL configured gain to raw RSSI measurement
            //clamp between -128 and 0. Add 128 to the result and multiply by 10 to get a value between 0 to 1280
            rssi_dbm = (rssi_dbm < -128) ? -128 : ((rssi_dbm > 0) ? 0 : rssi_dbm);
#ifdef SCF_FAPI_10_04
            pdu_hdr.measurement.rssi = std::round((rssi_dbm + 128.0)) * 10;
#else
            pdu_hdr.rssi = std::round((rssi_dbm + 128.0)) * 10;
#endif
        }
        break;
#if 0
        case 2: // dBFS 128dBFs to 0dBFS with a step size of 0.1dB
        {
#ifdef SCF_FAPI_10_04
            pdu_hdr.measurement.rssi = (rssi > 0) ? 0 : static_cast<int>(rssi * -10);
#else
            pdu_hdr.rssi = (rssi > 0) ? 0 : static_cast<int>(rssi * -10);
#endif
        }
        break;
#endif
        default:
        break;
    }
}

inline uint16_t pucch_rssi_f01(uint8_t rssiMeasurement, uint16_t conf_rssi, float ul_configured_gain, float rssi)
{
    switch(rssiMeasurement)
    {
        case 0: // Do not report RSSI
        {
            return conf_rssi;
        }
        break;
        case 1: // dBm = ULConfiguredGainConstant + dBFS 128dBm to 0dB, with a step size of 0.1dB
        {
            NVLOGD_FMT(TAG, "{}: Raw RSSI={} ul_configured_gain={}", __FUNCTION__, rssi, ul_configured_gain);
            float rssi_dbm = rssi - ul_configured_gain; // Applied UL configured gain to raw RSSI measurement
            //clamp between -128 and 0. Add 128 to the result and multiply by 10 to get a value between 0 to 1280
            rssi_dbm = (rssi_dbm < -128) ? -128 : ((rssi_dbm > 0) ? 0 : rssi_dbm);
            return std::round((rssi_dbm + 128.0)) * 10;
        }
        break;
        default:
            return conf_rssi;
        break;
    }
}

inline uint16_t pucch_rsrp_f01(uint8_t rsrpMeasurement, uint16_t conf_rsrp, float ul_configured_gain, float rsrp)
{
    switch(rsrpMeasurement)
    {
        case 0: // Do not report RSRP
        {
            return conf_rsrp;
        }
        break;
        case 1:
        {
            NVLOGD_FMT(TAG, "{}: Raw RSRP={} ul_configured_gain={}", __FUNCTION__, rsrp, ul_configured_gain);
            float rsrp_dbm = rsrp - ul_configured_gain; // Applied UL configured gain to raw RSRP measurement 
            //clamp between -140 and -12. Add 140 to the result and multiply by 10 to get a value between 0 to 1280
            rsrp_dbm = (rsrp_dbm < -140) ? -140 : ((rsrp_dbm > -12) ? -12 : rsrp_dbm);
            return std::round((rsrp_dbm + 140.0)) * 10;
        }
        break;
        default:
            return conf_rsrp;
        break;
    }
    return conf_rsrp;
}

inline void pucch_ul_ta_f234(scf_fapi_pucch_format_hdr& pdu_hdr, const cuphyPucchDataOut_t& cuphy_out, const cuphyPucchF234OutOffsets_t& uci, float& non_prach_ta_offset_usec_, uint8_t mu)
{
    float rawTA = cuphy_out.pTaEst[uci.taEstOffset];
    uint16_t ta = INVALID_TA;

    NVLOGD_FMT(TAG, "{}: Raw TA={} us", __FUNCTION__, rawTA);

    ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec_, mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
    bool invalidTa = (cuphy_out.HarqDetectionStatus[uci.HarqDetectionStatusOffset] == CUPHY_FAPI_DTX);
    ta = invalidTa ? INVALID_TA : ta;
#ifdef SCF_FAPI_10_04
    pdu_hdr.measurement.timing_advance = ta;
    pdu_hdr.measurement.timing_advance_ns = invalidTa ? INVALID_TA : static_cast<int16_t>(rawTA * 1000);// rawTA from cuPHY is usecs
#else
    pdu_hdr.timing_advance = ta;
#endif
}

void phy::send_uci_indication(slot_command_api::slot_indication& slot,
            const slot_command_api::pucch_params& params,
            const cuphyPucchDataOut_t& out)
{
    if (out.pF0UcisOut == nullptr && out.pF1UcisOut == nullptr && out.pPucchF3OutOffsets == nullptr)
    {
        NVLOGI_FMT(TAG, "{}: No PUCCH output data", __FUNCTION__);
        return;
    }

    cuphyPucchF0F1UciOut_t* ucisF01[2] = { out.pF0UcisOut , out.pF1UcisOut};
    cuphyPucchF234OutOffsets_t* ucisF234[5] = { nullptr,
                                                nullptr,
                                                out.pPucchF2OutOffsets,
                                                out.pPucchF3OutOffsets,
                                                out.pPucchF4OutOffsets};

    uint16_t nUCIs[5] = {   params.grp_dyn_pars.nF0Ucis,
                            params.grp_dyn_pars.nF1Ucis,
                            params.grp_dyn_pars.nF2Ucis,
                            params.grp_dyn_pars.nF3Ucis,
                            params.grp_dyn_pars.nF4Ucis};

    cuphyPucchUciPrm_t* uciParams[5] = {params.grp_dyn_pars.pF0UciPrms,
                                        params.grp_dyn_pars.pF1UciPrms,
                                        params.grp_dyn_pars.pF2UciPrms,
                                        params.grp_dyn_pars.pF3UciPrms,
                                        params.grp_dyn_pars.pF4UciPrms};
    bool uci_for_cell_found = false;
    for(int i = 0; i < params.grp_dyn_pars.nCells ; i++)
    {
        nv::phy_mac_transport& transport = phy_module().transport(params.cell_index_list[i]);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }
        msg_desc.cell_id = params.cell_index_list[i];
        NVLOGD_FMT(TAG, "{}: UCI indication on PUCCH cell_id={}", __FUNCTION__, params.cell_index_list[i]);

        auto fapi = add_scf_fapi_hdr<scf_fapi_uci_ind_t>(msg_desc, SCF_FAPI_UCI_INDICATION, params.cell_index_list[i], false);
        auto& ind = *reinterpret_cast<scf_fapi_uci_ind_t*>(fapi);
        uint8_t *next = reinterpret_cast<uint8_t*>(ind.payload);
        std::size_t offset = 0;
        ind.sfn =  slot.sfn_;
        ind.slot = slot.slot_;
        ind.num_ucis = 0;

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(params.cell_index_list[i]);

        nv::phy_mac_msg_desc pucch_234_msg_desc;
        if (transport.tx_alloc(pucch_234_msg_desc) < 0)
        {
            return;
        }
        pucch_234_msg_desc.cell_id = params.cell_index_list[i];
        auto fapi_pucch_234 = add_scf_fapi_hdr<scf_fapi_rx_measurement_ind_t>(pucch_234_msg_desc, SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION, params.cell_index_list[i], false);
        auto& ind_pucch_234 = *reinterpret_cast<scf_fapi_rx_measurement_ind_t*>(fapi_pucch_234);
        ind_pucch_234.sfn =  slot.sfn_;
        ind_pucch_234.slot = slot.slot_;
        ind_pucch_234.num_meas = 0;
        uint8_t* next_pucch_234 = reinterpret_cast<uint8_t*>(ind_pucch_234.meas_info);
        uint16_t offset_pucch_234 = 0;

        for (uint16_t f = 0; f < 5 ; f++)
        {
            // PF4 is not yet supported
            if(f == 4)
            {
                continue;
            }

            switch(f)
            {
                case 0:
                case 1:
                {
                    if (ucisF01[f] == nullptr)
                    {
                        NVLOGD_FMT(TAG, "{}: Skipping UCI processing for PUCCH format {} (UCI pointer is null)", __FUNCTION__, f);
                        continue;
                    }
                    for (uint16_t n = 0; n < nUCIs[f]; n++)
                    {
                        cuphyPucchF0F1UciOut_t& uci = ucisF01[f][n];
                        cuphyPucchUciPrm_t& uciParam = uciParams[f][n];
                        if(params.cell_index_list[i] != phy_module().get_cell_id_from_stat_prm_idx(uciParam.cellPrmStatIdx))
                        {
                            if(uci_for_cell_found)
                            {
                                break;
                            }
                            continue;
                        }
                        uci_for_cell_found = true;
                        scf_fapi_uci_pdu_t& uci_pdu = *reinterpret_cast<scf_fapi_uci_pdu_t*>(next);
                        uci_pdu.pdu_size = sizeof(scf_fapi_uci_pdu_t);
                        next = uci_pdu.payload;

                        uci_pdu.pdu_type = 1; // SCF222: UCI indication PDU carried on PUCCH Format 0 or 1, see Section 3.4.9.2

                        scf_fapi_pucch_format_hdr& pdu_hdr = *reinterpret_cast<scf_fapi_pucch_format_hdr*>(next);
                        pdu_hdr.pdu_bitmap = 0;
                        pdu_hdr.handle = params.scf_ul_tti_handle_list[f][uciParam.uciOutputIdx];
                        pdu_hdr.rnti = uciParam.rnti;
                        pdu_hdr.pucch_format = uciParam.formatType;
#ifdef SCF_FAPI_10_04
                        float adjusted_snr = (uci.SinrDB < -65.534) ? (-65.534) : ((uci.SinrDB > 65.534) ? 65.534 : uci.SinrDB);
                        float rawTA = uci.taEstMicroSec;
                        pdu_hdr.measurement.rssi = pucch_rssi_f01(getRssiMeasurement(), phy_module().rssi(), mplane.ul_gain_calibration,uci.RSSI);
                        uint16_t ta = INVALID_TA;
                        ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec_, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
                        bool invalidTa = (f == 0 || uci.HarqValues[0] == 2); // report invalid TA for PF0 (not supported) or HARQ DTX
                        pdu_hdr.measurement.timing_advance = invalidTa ? INVALID_TA : ta;
                        pdu_hdr.measurement.timing_advance_ns = invalidTa ? INVALID_TA : static_cast<int16_t>(rawTA * 1000); // rawTA from cuPHY is usecs
                        pdu_hdr.measurement.ul_sinr_metric = (f == 0) ? 0xFFFF : static_cast<int16_t>(std::round(adjusted_snr * 500)); // PF0 not supported
                        pdu_hdr.measurement.rsrp = pucch_rsrp_f01(getRsrpMeasurement(), UINT16_MAX, mplane.ul_gain_calibration, uci.RSRP);

                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {}: raw_sinr={} raw_rssi={} raw_rsrp={} raw_ta={} -> ul_sinr={} ta={} ta_ns={} rssi={} rsrp={}", __FUNCTION__,
                            uciParam.formatType,
                            uci.SinrDB, uci.RSSI, uci.RSRP, uci.taEstMicroSec,
                            static_cast<int>(pdu_hdr.measurement.ul_sinr_metric),
                            static_cast<int>(pdu_hdr.measurement.timing_advance),
                            static_cast<int>(pdu_hdr.measurement.timing_advance_ns),
                            static_cast<unsigned>(pdu_hdr.measurement.rssi),
                            static_cast<unsigned>(pdu_hdr.measurement.rsrp));
#else
                        float adjusted_snr = (uci.SinrDB < -64.0) ? (-64.0) : ((uci.SinrDB > 63.0) ? 63.0 : uci.SinrDB);
                        pdu_hdr.ul_cqi = (f == 0) ? 0xFFFF : static_cast<int16_t>(std::round((adjusted_snr + 64.0) * 2)); // PF0 not supported

                        float rawTA = uci.taEstMicroSec;
                        uint16_t ta = INVALID_TA;
                        bool invalidTa = (f == 0 || uci.HarqValues[0] == 2); // report invalid TA for PF0 (not supported) or HARQ DTX
                        ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec_, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
                        pdu_hdr.timing_advance = invalidTa ? INVALID_TA : ta; // PF0 not supported

                        pdu_hdr.rssi = pucch_rssi_f01(getRssiMeasurement(), phy_module().rssi(), mplane.ul_gain_calibration,uci.RSSI);
                        
                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {}: raw_sinr={} raw_rssi={} raw_ta={} -> ul_cqi={} ta={} rssi={}", __FUNCTION__,
                            uciParam.formatType, uci.SinrDB, uci.RSSI, uci.taEstMicroSec,
                            pdu_hdr.ul_cqi, static_cast<unsigned>(pdu_hdr.timing_advance), static_cast<unsigned>(pdu_hdr.rssi));
#endif

                        uci_pdu.pdu_size += sizeof(scf_fapi_pucch_format_hdr);
                        next = pdu_hdr.payload;

                        // Add SR if requested by L2
                        if (uciParam.srFlag > 0)
                        {
                            scf_fapi_sr_format_0_1_info_t* sr = reinterpret_cast<scf_fapi_sr_format_0_1_info_t*>(next);
                            pdu_hdr.pdu_bitmap |= 0x1; // Bit 0: SR
                            sr->sr_indication = uci.SRindication;
                            sr->sr_confidence_level = uci.SRconfidenceLevel;
                            uci_pdu.pdu_size += sizeof(scf_fapi_sr_format_0_1_info_t);
                            next += sizeof(scf_fapi_sr_format_0_1_info_t);
                        }

                        // Add HARQ if exists
                        if (uciParam.bitLenHarq > 0)
                        {
                            scf_fapi_harq_info_t& harq = *reinterpret_cast<scf_fapi_harq_info_t*>(next);
                            pdu_hdr.pdu_bitmap |= 0x2; // Bit 1: HARQ
                            harq.num_harq = uci.NumHarq;
                            harq.harq_confidence_level = uci.HarqconfidenceLevel;
                            for (int i = 0; i < harq.num_harq; i++)
                            {
#ifdef SCF_FAPI_10_04
                                harq.harq_value[i] = uci.HarqValues[i];
#else
                                // In FAPI 10.02, HARQ values are encoded as follows:
                                //0 = pass,1 = fail, 2 = not present
                                harq.harq_value[i] = (uci.HarqValues[i] <= 1) ? (1 - uci.HarqValues[i]) : uci.HarqValues[i];
#endif
                                NVLOGD_FMT(TAG, "{}: UCI PUCCH{} HARQ[{}]={}", __FUNCTION__, uciParam.formatType, i, harq.harq_value[i]);
                            }
                            uci_pdu.pdu_size += sizeof(scf_fapi_harq_info_t) + harq.num_harq;
                            next = harq.harq_value + harq.num_harq;
                        }

                        ind.num_ucis ++;
                        offset += uci_pdu.pdu_size;

                        NVLOGD_FMT(TAG, "{}: UCI.indication: PUCCH Format {}: bitmap=0x{:X} SRindication={} NumHarq={}", __FUNCTION__, uciParam.formatType, pdu_hdr.pdu_bitmap, uci.SRindication, uci.NumHarq);
                    }
                    uci_for_cell_found = false;
                }
                break;
                case 2:
                case 3:
                {
                    if (ucisF234[f] == nullptr)
                    {
                        NVLOGD_FMT(TAG, "{}: Skipping UCI processing for PUCCH format {} (UCI pointer is null)", __FUNCTION__, f);
                        continue;
                    }
                    for (uint16_t n = 0; n < nUCIs[f]; n++)
                    {
                        cuphyPucchF234OutOffsets_t& uci = ucisF234[f][n];
                        cuphyPucchUciPrm_t& uciParam = uciParams[f][n];

                        if(params.cell_index_list[i] != phy_module().get_cell_id_from_stat_prm_idx(uciParam.cellPrmStatIdx))
                        {
                            if(uci_for_cell_found)
                            {
                                break;
                            }
                            continue;
                        }
                        uci_for_cell_found = true;

                        scf_fapi_uci_pdu_t& uci_pdu = *reinterpret_cast<scf_fapi_uci_pdu_t*>(next);
                        uci_pdu.pdu_size = sizeof(scf_fapi_uci_pdu_t);
                        next = uci_pdu.payload;
                        uci_pdu.pdu_type = 2; // SCF222: UCI indication PDU carried on PUCCH Format 2, 3 or 4, see Section 3.4.9.3.

                        scf_fapi_pucch_format_hdr& pdu_hdr = *reinterpret_cast<scf_fapi_pucch_format_hdr*>(next);
                        pdu_hdr.pdu_bitmap = 0;
                        pdu_hdr.handle = params.scf_ul_tti_handle_list[f][uciParam.uciOutputIdx];
                        pdu_hdr.rnti = uciParam.rnti;
                        //Field pucch_format has value 0,1,2 for PUCCH format 2,3,4
                        pdu_hdr.pucch_format = uciParam.formatType-2;
                        pucch_ul_cqi_f234(pdu_hdr, out, uci, mplane.ul_gain_calibration);
                        pucch_ul_ta_f234(pdu_hdr, out, uci, non_prach_ta_offset_usec_, phy_cell_params.mu);
                        pucch_rssi_f234(getRssiMeasurement(), phy_module().rssi(), mplane.ul_gain_calibration, pdu_hdr, out, uci);
#ifdef SCF_FAPI_10_04
                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : ul_sinr={} ta={} ta-ns={} rssi={} rsrp={}", __FUNCTION__, uciParam.formatType,
                               static_cast<int>(pdu_hdr.measurement.ul_sinr_metric),
                               static_cast<unsigned>(pdu_hdr.measurement.timing_advance),
                               static_cast<int>(pdu_hdr.measurement.timing_advance_ns),
                               static_cast<unsigned>(pdu_hdr.measurement.rssi),
                               static_cast<unsigned>(pdu_hdr.measurement.rsrp));
#else
                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : ul_sinr={} ta={} rssi={} ", __FUNCTION__, uciParam.formatType, pdu_hdr.ul_cqi, static_cast<unsigned>(pdu_hdr.timing_advance), static_cast<unsigned>(pdu_hdr.rssi));
#endif
                        uci_pdu.pdu_size += sizeof(scf_fapi_pucch_format_hdr);
                        next = pdu_hdr.payload;
                        uint32_t uciSeg1PayloadByteOffset = uci.uciSeg1PayloadByteOffset;
                        uint8_t xtra_bits = 0;
                        uint32_t uci_payload_len_bits = uciParam.bitLenHarq +  uciParam.bitLenSr + uciParam.bitLenCsiPart1;
                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : Harq_Len {}, SR len {} CSI len = {} uci_offset = {}", __FUNCTION__, uciParam.formatType, uciParam.bitLenHarq, uciParam.bitLenSr, uciParam.bitLenCsiPart1, uciSeg1PayloadByteOffset);
                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : Harq_offset {}, SR offset {} CSI offset = {} ", __FUNCTION__, uciParam.formatType, uci.harqPayloadByteOffset, uci.srPayloadByteOffset, uci.csi1PayloadByteOffset);

#if 0
                        //for(int i = uciSeg1PayloadByteOffset; i < uciSeg1PayloadByteOffset + (uci_payload_len_bits + 7)>>3; i++)
                        for(int i = uciSeg1PayloadByteOffset; i < uciSeg1PayloadByteOffset + 4; i++)
                        {
                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : UCI_payload[{}] = {}", __FUNCTION__, uciParam.formatType, i, out.pUciPayloads[i]);
                        }
#endif

                        if(uciParam.bitLenSr > 0)
                        {
                            //In UCI bit stream SR bits are after HARQ bits so find the offset of SR bits.
                            int num_bytes = 1;
                            uint32_t srPayloadOffset = uci.srPayloadByteOffset;
                            scf_fapi_sr_format_2_3_4_info_t& sr_pdu = *reinterpret_cast<scf_fapi_sr_format_2_3_4_info_t*>(next);
                            uci_pdu.pdu_size += sizeof(scf_fapi_sr_format_2_3_4_info_t);
                            pdu_hdr.pdu_bitmap |= 0x1; // Bit 0: SR
                            sr_pdu.sr_bit_len = uciParam.bitLenSr;
                            sr_pdu.sr_payload[0] = out.pUciPayloads[srPayloadOffset];
                            sr_pdu.sr_payload[0] =  sr_pdu.sr_payload[0] & (~(0xFF << sr_pdu.sr_bit_len));
                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : SR payload = {} !", __FUNCTION__, uciParam.formatType, sr_pdu.sr_payload[0]);
                            uci_pdu.pdu_size += num_bytes;
                            next += sizeof(scf_fapi_sr_format_2_3_4_info_t) + num_bytes;
                        }

                        if(uciParam.bitLenHarq > 0)
                        {
                            scf_fapi_harq_format_2_3_4_info_t& harq_pdu = *reinterpret_cast<scf_fapi_harq_format_2_3_4_info_t*>(next);
                            uci_pdu.pdu_size += sizeof(scf_fapi_harq_format_2_3_4_info_t);
                            pdu_hdr.pdu_bitmap |= 0x2; // Bit 1: HARQ
#ifdef SCF_FAPI_10_04
                            harq_pdu.harq_detection_status = out.HarqDetectionStatus[uci.HarqDetectionStatusOffset];
                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : HARQ detection status {}", __FUNCTION__, uciParam.formatType, harq_pdu.harq_detection_status);
#else
                            harq_pdu.harq_crc = (uci_payload_len_bits > 11) ? out.pCrcFlags[uci.harqCrcFlagOffset]: 2;
                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : HARQ CRC {}", __FUNCTION__, uciParam.formatType, harq_pdu.harq_crc);
#endif
                            harq_pdu.harq_bit_len = uciParam.bitLenHarq;
                            xtra_bits = harq_pdu.harq_bit_len & 7;
                            int num_bytes = (harq_pdu.harq_bit_len + 7) >> 3 ;
                            std::copy(out.pUciPayloads + uci.harqPayloadByteOffset ,out.pUciPayloads + uci.harqPayloadByteOffset + num_bytes, harq_pdu.harq_payload);
                            uint8_t * last_payload_byte = &harq_pdu.harq_payload[num_bytes -1];
                            if(xtra_bits)
                            {
                                *last_payload_byte = *last_payload_byte &(~ ( 0xFF << xtra_bits));
                            }

                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : HARQ bitlen {}", __FUNCTION__, uciParam.formatType, static_cast<int>(harq_pdu.harq_bit_len));
                            uci_pdu.pdu_size += num_bytes;
                            next += sizeof(scf_fapi_harq_format_2_3_4_info_t) + num_bytes;
                        }

                        if(uciParam.bitLenCsiPart1 > 0)
                        {
                            int num_bytes = 0;

                            scf_fapi_csi_part_1_t& csip1_pdu = *reinterpret_cast<scf_fapi_csi_part_1_t*>(next);
                            uci_pdu.pdu_size += sizeof(scf_fapi_csi_part_1_t);
                            pdu_hdr.pdu_bitmap |= 0x4; // Bit 2: CSIP1
#ifdef SCF_FAPI_10_04
                            csip1_pdu.csi_part1_detection_status = out.CsiP1DetectionStatus[uci.CsiP1DetectionStatusOffset];
                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : CSI Part1 detection offset {} status {}", __FUNCTION__, uciParam.formatType,
                                uci.CsiP1DetectionStatusOffset, csip1_pdu.csi_part1_detection_status);
#else
                            csip1_pdu.csi_part_1_crc = out.pCrcFlags[uci.csi1CrcFlagOffset];
#endif
                            csip1_pdu.csi_part_1_bit_len = uciParam.bitLenCsiPart1;

                            xtra_bits = csip1_pdu.csi_part_1_bit_len & 0x7;
                            num_bytes = (csip1_pdu.csi_part_1_bit_len + 7) >> 3;

                            std::copy(out.pUciPayloads + uci.csi1PayloadByteOffset ,out.pUciPayloads + uci.csi1PayloadByteOffset + num_bytes, csip1_pdu.csi_part_1_payload);

                            if(xtra_bits)
                            {
                                csip1_pdu.csi_part_1_payload[num_bytes - 1] = csip1_pdu.csi_part_1_payload[num_bytes - 1] & (~(0XFF << xtra_bits));
                            }
                            uci_pdu.pdu_size += num_bytes;
                            next += sizeof(scf_fapi_csi_part_1_t) + num_bytes;
                        }
                        //Interference Indication
                        if(getPf234Interference())
                        {
                            scf_fapi_meas_t& meas_info = *(reinterpret_cast<scf_fapi_meas_t*>(next_pucch_234));
                            meas_info.handle =  params.scf_ul_tti_handle_list[f][uciParam.uciOutputIdx];
                            meas_info.rnti = uciParam.rnti;
                            float pn_dbm = out.pInterf[uci.InterfOffset] - mplane.ul_gain_calibration; // conversion from db to dBm
                            //clamp between -152 and 0. Add 152 to the result and multiply by 10 to get a value between 0 to 1520
                            pn_dbm = (pn_dbm < -152.0) ? -152.0 : ((pn_dbm > 0) ? 0 : pn_dbm);
                            meas_info.meas = (pn_dbm + 152.0) * 10;
                            next_pucch_234 += sizeof(scf_fapi_meas_t);
                            offset_pucch_234 +=sizeof(scf_fapi_meas_t);
                            ind_pucch_234.num_meas++;
                            NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : interference Raw={} dbm {} numMeasurements {}", __FUNCTION__, uciParam.formatType,
                                out.pInterf[uci.InterfOffset], static_cast<int>(meas_info.meas), static_cast<int>(ind_pucch_234.num_meas));
                        }

                        ind.num_ucis++;
                        offset += uci_pdu.pdu_size;

                        NVLOGD_FMT(TAG, "{}: UCI PUCCH Format {} : bitmap=0x{:X} numUcis {}", __FUNCTION__, uciParam.formatType, pdu_hdr.pdu_bitmap, static_cast<int>(ind.num_ucis));
                    }
                    uci_for_cell_found = false;
                }
                break;
                default:
                    NVLOGE_FMT(TAG, AERIAL_FAPI_EVENT, "{}: PUCCH format {} not supported yet!", __FUNCTION__, f);
                break;
            }
        }
        fapi->length += offset;
        msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
        transport.tx_send(msg_desc);
        transport.notify(1);
        metrics_.incr_tx_packet_count(SCF_FAPI_UCI_INDICATION);

        if(ind_pucch_234.num_meas)
        {
            fapi_pucch_234->length += offset_pucch_234;
            pucch_234_msg_desc.msg_len = fapi_pucch_234->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
            transport.tx_send(pucch_234_msg_desc);
            transport.notify(1);
            metrics_.incr_tx_packet_count(SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION);
        }
        else
        {
           transport.tx_release(pucch_234_msg_desc);
        }
    }
}

inline void ul_ind_rssi(uint16_t& rssi_out, const cuphyPuschDataOut_t& out, const slot_command_api::pusch_params& params, int ue_index, const cuphyPuschStatPrms_t& puschStatPrms, uint8_t rssiMeasurement, float ul_configured_gain, uint16_t default_value)
{
    if(puschStatPrms.enableRssiMeasurement && out.pRssi != nullptr)
    {
        switch(rssiMeasurement)
        {
            case 0: // Do not report RSSI
            {
                rssi_out = default_value; // 0xffff by default unless config file specifies otherwise
            }
            break;
            case 1: // dBm = ULConfiguredGainConstant + dBFS 128dBm to 0dB, with a step size of 0.1dB
            {
                float rssi_dbm = 0.0;
                NVLOGD_FMT(TAG, "{}: Raw RSSI={} db ul_configured_gain={}", __FUNCTION__, out.pRssi[params.ue_info[ue_index].ueGrpIdx], ul_configured_gain);
                rssi_dbm = out.pRssi[params.ue_info[ue_index].ueGrpIdx] - ul_configured_gain; // Applied UL configured gain to raw RSSI measurement
                //clamp between -128 and 0. Add 128 to the result and multiply by 10 to get a value between 0 to 1280
                rssi_dbm = (rssi_dbm < -128) ? -128 : ((rssi_dbm > 0) ? 0 : rssi_dbm);
                rssi_out = std::round(rssi_dbm + 128.0) * 10;
            }
            break;
#if 0
            //TODO: ul_configured_gain for dBFs is different from dB. Need to fix this to support dBFs
            case 2: // dBFS 128dBFs to 0dBFS with a step size of 0.1dB
            {
                float rssi_dBFS = 0.0;
                NVLOGD_FMT(TAG, "{}: Raw RSSI={} dBFs ul_configured_gain={}", __FUNCTION__, out.pRssi[params.ue_info[ue_index].ueGrpIdx], ul_configured_gain);
                rssi_dBFS = out.pRssi[params.ue_info[ue_index].ueGrpIdx] - ul_configured_gain; // Applied UL configured gain to raw RSSI measurement
                //clamp between -128 and 0. Add 128 to the result and multiply by 10 to get a value between 0 to 1280
                rssi_out = (std::clamp(rssi_dBFs,-128,0) + 128) * 10;
            }
            break;
#endif
            default:
            break;
        }
    }
    else
    {
        rssi_out = default_value; // 0xffff by default unless config file specifies otherwise
    }
}

inline void ul_ind_rsrp(uint16_t& rsrp_out, const cuphyPuschDataOut_t& out, const slot_command_api::pusch_params& params, int ue_index, const cuphyPuschStatPrms_t& puschStatPrms, uint8_t rsrpMeasurement, float ul_configured_gain, uint16_t default_value) {
    if(puschStatPrms.enableSinrMeasurement && out.pRsrp != nullptr)
    {
        switch(rsrpMeasurement)
        {
            case 0: // Do not report RSRP
            {
                rsrp_out = default_value; // 0xffff by default unless config file specifies otherwise
            }
            break;
            case 1: // dBm = ULConfiguredGainConstant + dBFS 128dBm to 0dB, with a step size of 0.1dB
            {
                float rsrp_dbm = 0.0;
                NVLOGD_FMT(TAG, "{}: ue_index ={} Raw RSRP ={} db ul_configured_gain ={}", __FUNCTION__, ue_index, out.pRsrp[ue_index], ul_configured_gain);
                rsrp_dbm = out.pRsrp[ue_index] - ul_configured_gain; // Applied UL configured gain to raw RSRP measurement
                //clamp between -140 to -12 , add 140 and multiply by 10 to get a value between 0 and 1280
                rsrp_dbm = (rsrp_dbm < -140) ? -140 : ((rsrp_dbm > -12) ? -12 : rsrp_dbm);
                rsrp_out = std::round((rsrp_dbm + 140.0)) * 10;
            }
            break;
#if 0
            //TODO: ul configured gain is different for dBFs. Need to fix this to support dBFS
            case 2: // dBFS 128dBFs to 0dBFS with a step size of 0.1dB
            {
                float rsrp_dBFs = 0.0;
                NVLOGD_FMT(TAG, "{}: Raw RSRP ={} dBFs ul_configured_gain = {}", __FUNCTION__, out.pRsrp[params.ue_info[ue_index].ueGrpIdx], ul_configured_gain);
                rsrp_dBFs = out.pRsrp[params.ue_info[ue_index].ueGrpIdx] - ul_configured_gain;
                //clamp between -140 to -12 , add 140 and multiply by 10 to get a value between 0 and 1280
                rsrp_out = ( (std::clamp(rsrp_dBFs, -140, -12) + 140) * 10;
            }
            break;
#endif
            default:
            break;
        }
    }
    else
    {
        rsrp_out = default_value; // 0xffff by default unless config file specifies otherwise
    }
}
template <typename T, int res>
inline void ul_ind_ul_cqi(T& ul_cqi, const cuphyPuschDataOut_t& out, int ue_index, const cuphyPuschStatPrms_t& puschStatPrms)
{
    if(puschStatPrms.enableSinrMeasurement)
    {
        float adjusted_snr;
        if(out.pSinrPostEq) {
            adjusted_snr = out.pSinrPostEq[ue_index];
            NVLOGD_FMT(TAG, "{}: Post Eq Raw SINR={}", __FUNCTION__, adjusted_snr);

        }
        else {
            adjusted_snr = out.pSinrPreEq[ue_index];
            NVLOGD_FMT(TAG, "{}: Pre Eq SINR={}", __FUNCTION__, adjusted_snr);
        }
        adjusted_snr = std::clamp(adjusted_snr, -64.0f, 63.0f);
        uint8_t adjusted_snr_int = (uint8_t)(std::round(adjusted_snr + 64.0) * res);
        ul_cqi = adjusted_snr_int;
    }
    else
    {
        ul_cqi = std::numeric_limits<T>::max();
    }
}

inline void ul_ind_ul_cqi_fapi_10_04(int16_t& ul_cqi, const cuphyPuschDataOut_t& out, int ue_index, const cuphyPuschStatPrms_t& puschStatPrms) {
    if(puschStatPrms.enableSinrMeasurement)
    {
        float raw_snr;
        if(out.pSinrPostEq)
            raw_snr = out.pSinrPostEq[ue_index];
        else
            raw_snr = out.pSinrPreEq[ue_index];
    
        NVLOGD_FMT(TAG, "{}: Raw SINR={}", __FUNCTION__, raw_snr);
        int32_t sinr_db = (int32_t)(std::round(raw_snr*500));
        ul_cqi = (sinr_db < -32767) ? -32767 : ((sinr_db > 32767) ? 32767 : sinr_db);
    }
    else
    {
        ul_cqi = 0xFFFF;
    }
}

void phy::send_uci_indication(const slot_command_api::slot_indication& slot,
            const slot_command_api::pusch_params& params,
            const cuphyPuschDataOut_t& out, ::cuphyPuschStatPrms_t const* puschStatPrms)
{
    uint16_t nUes = params.cell_grp_info.nUes;
    uint32_t ue_idx = 0;
    auto taEsts{out.pTaEsts};
    auto cfoEsts{out.pCfoHz};
    for(uint32_t cell_idx = 0; cell_idx < params.cell_grp_info.nCells; cell_idx++)
    {
        uint32_t cell_id = params.cell_index_list[cell_idx];
        nv::phy_mac_transport& transport = phy_module().transport(cell_id);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);

        auto fapi = add_scf_fapi_hdr<scf_fapi_uci_ind_t>(msg_desc, SCF_FAPI_UCI_INDICATION, cell_id, false);
        auto& ind = *reinterpret_cast<scf_fapi_uci_ind_t*>(fapi);
        uint8_t *next = reinterpret_cast<uint8_t*>(ind.payload);
        std::size_t offset = 0;

        ind.sfn =  slot.sfn_;
        ind.slot = slot.slot_;
        ind.num_ucis = 0;

        for(;ue_idx < nUes; ++ue_idx)
        {
            NVLOGD_FMT(TAG, "{}: UCI indication on PUSCH for ue {} cell-id {}", __FUNCTION__, ue_idx, cell_id);
            if(phy_module().cell_group() &&
                phy_module().get_cell_id_from_stat_prm_idx(params.ue_info[ue_idx].pUeGrpPrm->pCellPrm->cellPrmStatIdx) != cell_id)
            {
                break;
            }

            if(params.ue_info[ue_idx].pUciPrms == nullptr)
            {
                continue;
            }

            auto uci = &out.pUciOnPuschOutOffsets[ue_idx];
#ifdef SCF_FAPI_10_04
            auto processed_early_harq = ((phy_module().get_phy_config().indication_instances_per_slot[nv::UCI_DATA_IND_IDX] ==  nv::MULTI_MSG_INSTANCE_PER_SLOT) && (uci->isEarlyHarq));

            #if 0
            // If already processed early and there no other UCI to process dont send UCI for this UE
            if(  (processed_early_harq) &&
                 (params.ue_info[ue_idx].pUciPrms->nBitsCsi1 == 0) &&
                 (params.ue_info[ue_idx].pUciPrms->nRanksBits == 255) )
            {
                continue;
            }
            #endif
#endif
            scf_fapi_uci_pdu_t& uci_pdu = *reinterpret_cast<scf_fapi_uci_pdu_t*>(next);
            uci_pdu.pdu_size = sizeof(scf_fapi_uci_pdu_t);
            next = uci_pdu.payload;
            uci_pdu.pdu_type = 0; // UCI indication PDU carried on PUSCH, see Section 3.4.9.1
            float rawTA =0.0f;
            if (puschStatPrms->enableToEstimation) {
                rawTA = taEsts[ue_idx];
                NVLOGD_FMT(TAG, "{}: UE Idx [{}] TO in microseconds= {}", __FUNCTION__, ue_idx, rawTA);
            }
            if (puschStatPrms->enableCfoCorrection) {
                float rawCfoEstHz{cfoEsts[ue_idx]};
                NVLOGD_FMT(TAG, "{}: UE Idx [{}] Raw CFO Estimate in Hz= {}", __FUNCTION__, ue_idx, rawCfoEstHz);
            }
            scf_fapi_uci_pusch_pdu_t& pdu_hdr = *reinterpret_cast<scf_fapi_uci_pusch_pdu_t*>(next);
            pdu_hdr.pdu_bitmap = 0;
            pdu_hdr.handle = params.scf_ul_tti_handle_list[ue_idx];
            pdu_hdr.rnti = params.ue_info[ue_idx].rnti;
            uint16_t val = UINT16_MAX;
#ifdef SCF_FAPI_10_04
            int16_t sinr = 0;
            ul_ind_ul_cqi_fapi_10_04(sinr, out, ue_idx, *puschStatPrms);
            pdu_hdr.measurement.ul_sinr_metric = sinr;
            uint16_t ta = UINT16_MAX;
            float non_prach_ta_offset_usec = 0.0f;
            ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
            pdu_hdr.measurement.timing_advance = ta;
            pdu_hdr.measurement.timing_advance_ns = static_cast<int16_t>(rawTA * 1000);// rawTA from cuPHY is usecs
            ul_ind_rssi(val, out, params, ue_idx, *puschStatPrms, getRssiMeasurement(), mplane.ul_gain_calibration, phy_module().rssi());
            pdu_hdr.measurement.rssi = val;
            ul_ind_rsrp(val, out, params, ue_idx, *puschStatPrms, getRsrpMeasurement(), mplane.ul_gain_calibration, phy_module().rsrp());
            pdu_hdr.measurement.rsrp = val;
            NVLOGD_FMT(TAG, "{}: UCI on PUSCH ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}", __FUNCTION__,
                   static_cast<int>(pdu_hdr.measurement.ul_sinr_metric),
                   static_cast<unsigned>(pdu_hdr.measurement.timing_advance),
                   static_cast<int>(pdu_hdr.measurement.timing_advance_ns),
                   static_cast<unsigned>(pdu_hdr.measurement.rssi),
                   static_cast<unsigned>(pdu_hdr.measurement.rsrp));
#else
            ul_ind_ul_cqi<uint8_t, 2>(pdu_hdr.ul_cqi, out, ue_idx, *puschStatPrms);
            uint16_t ta = UINT16_MAX;
            ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec_, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
            pdu_hdr.timing_advance = ta;
            ul_ind_rssi(val, out, params, ue_idx, *puschStatPrms, getRssiMeasurement(), mplane.ul_gain_calibration, phy_module().rssi());
            pdu_hdr.rssi = val;
            NVLOGD_FMT(TAG, "{}: UCI on PUSCH ul-cqi={} ta={} rssi={} ", __FUNCTION__, pdu_hdr.ul_cqi, static_cast<unsigned>(pdu_hdr.timing_advance), static_cast<unsigned>(pdu_hdr.rssi));
#endif
            uci_pdu.pdu_size += sizeof(scf_fapi_uci_pusch_pdu_t);
            next = pdu_hdr.payload;
#ifdef SCF_FAPI_10_04
            if((params.ue_info[ue_idx].pUciPrms->nBitsHarq > 0) && (!processed_early_harq))
#else
            if(params.ue_info[ue_idx].pUciPrms->nBitsHarq > 0)
#endif
            {
                scf_fapi_harq_format_2_3_4_info_t& harq_pdu = *reinterpret_cast<scf_fapi_harq_format_2_3_4_info_t*>(next);
                uci_pdu.pdu_size += sizeof(scf_fapi_harq_format_2_3_4_info_t);
                pdu_hdr.pdu_bitmap |= 0x2; // Bit 1: HARQ
                harq_pdu.harq_bit_len = params.ue_info[ue_idx].pUciPrms->nBitsHarq;
#ifdef SCF_FAPI_10_04
                harq_pdu.harq_detection_status = out.HarqDetectionStatus[uci->HarqDetectionStatusOffset];
                NVLOGD_FMT(TAG, "{}: UCI on PUSCH HARQ detection status {}", __FUNCTION__, harq_pdu.harq_detection_status);
                if(((params.ue_info[ue_idx].pUciPrms->nBitsHarq >= 12) &&
                    (harq_pdu.harq_detection_status == 1))||
                    ((params.ue_info[ue_idx].pUciPrms->nBitsHarq <= 11) &&
                    (harq_pdu.harq_detection_status == 4)))
    #else
                harq_pdu.harq_crc = out.pUciCrcFlags[uci->harqCrcFlagOffset];
                if(((params.ue_info[ue_idx].pUciPrms->nBitsHarq >= 12) &&
                    (harq_pdu.harq_crc == 0))||
                    (params.ue_info[ue_idx].pUciPrms->nBitsHarq <= 11))
    #endif
                {
                    int num_bytes = (harq_pdu.harq_bit_len + 8 - 1) / 8;
                    memcpy(harq_pdu.harq_payload, &out.pUciPayloads[uci->harqPayloadByteOffset], num_bytes);
                    NVLOGD_FMT(TAG, "{}: UCI on PUSCH HARQ bitlen {}", __FUNCTION__, static_cast<int>(harq_pdu.harq_bit_len));
                    uci_pdu.pdu_size += num_bytes;
                    next += sizeof(scf_fapi_harq_format_2_3_4_info_t) + num_bytes;
                }
                else
                    next += sizeof(scf_fapi_harq_format_2_3_4_info_t);
            }

            if(params.ue_info[ue_idx].pUciPrms->nBitsCsi1 > 0)
            {
                scf_fapi_csi_part_1_t& csip1_pdu = *reinterpret_cast<scf_fapi_csi_part_1_t*>(next);
                uci_pdu.pdu_size += sizeof(scf_fapi_csi_part_1_t);
                pdu_hdr.pdu_bitmap |= 0x4; // Bit 2: CSIP1
                csip1_pdu.csi_part_1_bit_len = params.ue_info[ue_idx].pUciPrms->nBitsCsi1;

#ifdef SCF_FAPI_10_04
                csip1_pdu.csi_part1_detection_status = out.CsiP1DetectionStatus[uci->CsiP1DetectionStatusOffset];
                NVLOGD_FMT(TAG, "{}: UCI on PUSCH CSI Part1 detection status {} CSI P1 bit len {}", __FUNCTION__, csip1_pdu.csi_part1_detection_status,static_cast<int>(csip1_pdu.csi_part_1_bit_len));

                if(((params.ue_info[ue_idx].pUciPrms->nBitsCsi1 >= 12) &&
                    (csip1_pdu.csi_part1_detection_status == 1))||
                    ((params.ue_info[ue_idx].pUciPrms->nBitsCsi1 <= 11) &&
                    (csip1_pdu.csi_part1_detection_status == 4)))
#else
                csip1_pdu.csi_part_1_crc = out.pUciCrcFlags[uci->csi1CrcFlagOffset];
                if(((params.ue_info[ue_idx].pUciPrms->nBitsCsi1 >= 12) &&
                    (csip1_pdu.csi_part_1_crc == 0))||
                    (params.ue_info[ue_idx].pUciPrms->nBitsCsi1 <= 11))
#endif
                {
                    int num_bytes = (csip1_pdu.csi_part_1_bit_len + 8 - 1) / 8;
                    memcpy(csip1_pdu.csi_part_1_payload, &out.pUciPayloads[uci->csi1PayloadByteOffset], num_bytes);
                    uci_pdu.pdu_size += num_bytes;
                    next += sizeof(scf_fapi_csi_part_1_t) + num_bytes;
                }
                else
                    next += sizeof(scf_fapi_csi_part_1_t);
            }

#ifdef SCF_FAPI_10_04
            if(params.ue_info[ue_idx].pUciPrms->nCsi2Reports != 0)
#else 
            if(params.ue_info[ue_idx].pUciPrms->nRanksBits != 255)
#endif 
            {
                scf_fapi_csi_part_2_t& csip2_pdu = *reinterpret_cast<scf_fapi_csi_part_2_t*>(next);
                uci_pdu.pdu_size += sizeof(scf_fapi_csi_part_2_t);
                pdu_hdr.pdu_bitmap |= 0x8; // Bit 3: CSIP2
                csip2_pdu.csi_part_2_bit_len = out.pNumCsi2Bits[uci->numCsi2BitsOffset];
#ifdef SCF_FAPI_10_04
                csip2_pdu.csi_part2_detection_status = out.CsiP2DetectionStatus[uci->CsiP2DetectionStatusOffset];
                NVLOGD_FMT(TAG, "{}: UCI on PUSCH CSI Part2 detection status {} CSI P2 len {}", __FUNCTION__, csip2_pdu.csi_part2_detection_status, static_cast<int>(csip2_pdu.csi_part_2_bit_len));
                if(((csip2_pdu.csi_part_2_bit_len >= 12) &&
                    (csip2_pdu.csi_part2_detection_status == 1))||
                    ((csip2_pdu.csi_part_2_bit_len <= 11) &&
                    (csip2_pdu.csi_part2_detection_status == 4)))
#else
                csip2_pdu.csi_part_2_crc = out.pUciCrcFlags[uci->csi2CrcFlagOffset];
                NVLOGD_FMT(TAG, "{}: UCI on PUSCH CSI Part2 CRC {} CSI P2 len {}", __FUNCTION__, csip2_pdu.csi_part_2_crc, static_cast<int>(csip2_pdu.csi_part_2_bit_len));
                if(((csip2_pdu.csi_part_2_bit_len >= 12) &&
                    (csip2_pdu.csi_part_2_crc == 0))||
                    (csip2_pdu.csi_part_2_bit_len <= 11))
#endif
                {
                    int num_bytes = (csip2_pdu.csi_part_2_bit_len + 8 - 1) / 8;
                    //memcpy(csip2_pdu.csi_part_2_payload, &out.pUciPayloads[uci->csi2PayloadByteOffset], num_bytes);
                    std::copy(&out.pUciPayloads[uci->csi2PayloadByteOffset], &out.pUciPayloads[uci->csi2PayloadByteOffset] + num_bytes, csip2_pdu.csi_part_2_payload);

                    uci_pdu.pdu_size += num_bytes;
                    next += sizeof(scf_fapi_csi_part_2_t) + num_bytes;
                }
                else
                {
                    csip2_pdu.csi_part_2_bit_len = 0;
                    next += sizeof(scf_fapi_csi_part_2_t);
                }
            }

            NVLOGD_FMT(TAG, "{}: UCI on PUSCH bitmap=0x{:X}", __FUNCTION__, pdu_hdr.pdu_bitmap);

            ind.num_ucis ++;
            offset += uci_pdu.pdu_size;
        }

        fapi->length += offset;
        msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
        transport.tx_send(msg_desc);
        transport.notify(1);
        metrics_.incr_tx_packet_count(SCF_FAPI_UCI_INDICATION);
    }
}

void phy::send_early_uci_indication(const slot_command_api::slot_indication& slot,
            const slot_command_api::pusch_params& params,
            const cuphyPuschDataOut_t& out, ::cuphyPuschStatPrms_t const* puschStatPrms, nanoseconds& t0_original)
{
    uint16_t nUes = params.cell_grp_info.nUes;
    uint32_t ue_idx = 0;

    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    uint8_t mCh_segment_proc_enable = 0;
    phyDriver.l1_get_ch_segment_proc_enable_info(&mCh_segment_proc_enable);


    uint64_t ch_seg_cell_error_mask = 0;
    for(uint32_t cell_idx = 0; cell_idx < params.cell_grp_info.nCells; cell_idx++)
    {
        // if (out.isEarlyHarqPresentCellMask & (1 << cell_idx)  == 0 ) {
        //     continue;
        // }
        uint32_t cell_id = params.cell_index_list[cell_idx];
        nv::phy_mac_transport& transport = phy_module().transport(cell_id);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }

        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);

        auto fapi = add_scf_fapi_hdr<scf_fapi_uci_ind_t>(msg_desc, SCF_FAPI_UCI_INDICATION, cell_id, false);
        auto& ind = *reinterpret_cast<scf_fapi_uci_ind_t*>(fapi);
        uint8_t *next = reinterpret_cast<uint8_t*>(ind.payload);
        std::size_t offset = 0;

        ind.sfn =  slot.sfn_;
        ind.slot = slot.slot_;
        ind.num_ucis = 0;

        for(;ue_idx < nUes; ++ue_idx)
        {
            NVLOGD_FMT(TAG, "{}: UCI indication on PUSCH for ue {} cell-id {}",__FUNCTION__,ue_idx,cell_id);
            if(phy_module().get_cell_id_from_stat_prm_idx(params.ue_info[ue_idx].pUeGrpPrm->pCellPrm->cellPrmStatIdx) != cell_id)
            {
                break;
            }

            if(params.ue_info[ue_idx].pUciPrms == nullptr)
            {
                continue;
            }

            auto uci = &out.pUciOnPuschOutOffsets[ue_idx];
            if(!uci->isEarlyHarq)
            {
                continue;
            }
            scf_fapi_uci_pdu_t& uci_pdu = *reinterpret_cast<scf_fapi_uci_pdu_t*>(next);
            uci_pdu.pdu_size = sizeof(scf_fapi_uci_pdu_t);
            next = uci_pdu.payload;
            uci_pdu.pdu_type = 0; // UCI indication PDU carried on PUSCH, see Section 3.4.9.1
            scf_fapi_uci_pusch_pdu_t& pdu_hdr = *reinterpret_cast<scf_fapi_uci_pusch_pdu_t*>(next);
            pdu_hdr.pdu_bitmap = 0;
            pdu_hdr.handle = params.scf_ul_tti_handle_list[ue_idx];
            pdu_hdr.rnti = params.ue_info[ue_idx].rnti;
#ifdef SCF_FAPI_10_04
            int16_t sinr{INT16_MAX};
            if (out.pSinrPreEq) {
                ul_ind_ul_cqi_fapi_10_04(sinr, out, ue_idx, *puschStatPrms);
            }
            pdu_hdr.measurement.ul_sinr_metric = sinr;
            pdu_hdr.measurement.timing_advance = UINT16_MAX;
            pdu_hdr.measurement.timing_advance_ns = UINT16_MAX;
            
            uint16_t val = UINT16_MAX;
            ul_ind_rssi(val, out, params, ue_idx, *puschStatPrms, getRssiMeasurement(), mplane.ul_gain_calibration, phy_module().rssi());
            pdu_hdr.measurement.rssi = val;
            ul_ind_rsrp(val, out, params, ue_idx, *puschStatPrms, getRsrpMeasurement(), mplane.ul_gain_calibration, phy_module().rsrp());
            pdu_hdr.measurement.rsrp = val;
#else
            pdu_hdr.ul_cqi = UINT8_MAX;
            pdu_hdr.timing_advance = UINT16_MAX;
            pdu_hdr.rssi = UINT16_MAX;
            NVLOGD_FMT(TAG, "{}: UCI on PUSCH ul-cqi={} ta={} rssi={} ", __FUNCTION__, pdu_hdr.ul_cqi, static_cast<unsigned>(pdu_hdr.timing_advance), static_cast<unsigned>(pdu_hdr.rssi));
#endif
            uci_pdu.pdu_size += sizeof(scf_fapi_uci_pusch_pdu_t);
            next = pdu_hdr.payload;
            if(params.ue_info[ue_idx].pUciPrms->nBitsHarq > 0)
            {
                scf_fapi_harq_format_2_3_4_info_t& harq_pdu = *reinterpret_cast<scf_fapi_harq_format_2_3_4_info_t*>(next);
                uci_pdu.pdu_size += sizeof(scf_fapi_harq_format_2_3_4_info_t);
                pdu_hdr.pdu_bitmap |= 0x2; // Bit 1: HARQ
                harq_pdu.harq_bit_len = params.ue_info[ue_idx].pUciPrms->nBitsHarq;
#ifdef SCF_FAPI_10_04
                harq_pdu.harq_detection_status = out.HarqDetectionStatus[uci->HarqDetectionStatusOffset];
                NVLOGD_FMT(TAG, "{}: UCI on PUSCH HARQ detection status {}", __FUNCTION__, harq_pdu.harq_detection_status);
                if(((params.ue_info[ue_idx].pUciPrms->nBitsHarq >= 12) &&
                    (harq_pdu.harq_detection_status == 1))||
                    ((params.ue_info[ue_idx].pUciPrms->nBitsHarq <= 11) &&
                    (harq_pdu.harq_detection_status == 4)))
#else
                harq_pdu.harq_crc = (params.ue_info[ue_idx].pUciPrms->nBitsHarq > 11) ? out.pUciCrcFlags[uci->harqCrcFlagOffset] : 2;

                if(((params.ue_info[ue_idx].pUciPrms->nBitsHarq >= 12) &&
                    (harq_pdu.harq_crc == 0))||
                    (params.ue_info[ue_idx].pUciPrms->nBitsHarq <= 11))
#endif
                {
                    int num_bytes = (harq_pdu.harq_bit_len + 8 - 1) / 8;
                    memcpy(harq_pdu.harq_payload, &out.pUciPayloads[uci->harqPayloadByteOffset], num_bytes);
                    NVLOGD_FMT(TAG, "{}: UCI on PUSCH HARQ bitlen {}", __FUNCTION__, static_cast<int>(harq_pdu.harq_bit_len));
                    uci_pdu.pdu_size += num_bytes;
                    next += sizeof(scf_fapi_harq_format_2_3_4_info_t) + num_bytes;
                }
                else
                    next += sizeof(scf_fapi_harq_format_2_3_4_info_t);
            }

            NVLOGD_FMT(TAG, "{}: UCI on PUSCH bitmap=0x{:X}", __FUNCTION__, pdu_hdr.pdu_bitmap);

            ind.num_ucis ++;
            offset += uci_pdu.pdu_size;
        }

        fapi->length += offset;
        msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
        transport.tx_send(msg_desc);
        transport.notify(1);
        metrics_.incr_tx_packet_count(SCF_FAPI_UCI_INDICATION);
        if (mCh_segment_proc_enable) {
            nanoseconds now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
            auto dur = duration_cast<microseconds>(now - t0_original).count();
            auto tx_send_dur = duration_cast<microseconds>(now - nanoseconds(transport.get_ts_send(msg_desc))).count();
            auto& ch_seg_timelines = phy_module().get_ch_timeline(SCF_CHAN_SEG_PUSCH << 16 | cell_id);
            auto& ch_seg_indexes = phy_module().get_ch_proc_indexes(SCF_CHAN_SEG_PUSCH << 16 | cell_id);
            if (ch_seg_indexes[SCF_CHAN_SEG_PUSCH] < 2) {
                continue;
            }
            NVLOGD_FMT(TAG, "{}: total duration {} pusch_timeline {}", __FUNCTION__, dur + tx_send_dur, ch_seg_timelines[1].second);
            if (dur + tx_send_dur >  ch_seg_timelines[1].second) {
                ch_seg_cell_error_mask |= (1 << cell_id);
            }
        }
    }

    if (mCh_segment_proc_enable) {
        for(uint32_t cell_idx = 0; cell_idx < params.cell_grp_info.nCells; cell_idx++) {
            if (ch_seg_cell_error_mask & (1 << cell_idx)) {
                send_error_indication_l1(SCF_FAPI_UL_TTI_REQUEST, SCF_ERROR_CODE_EARLY_HARQ_TIMING_ERROR, slot.sfn_, slot.slot_, cell_idx);
            }
        }
    }
}

/**
 * Static wrapper for UL alloc buffer callback - allows using function pointer instead of std::function
 * 
 * @param context Unused (nullptr)
 * @param buffer Output message buffer
 * @param params PUSCH parameters
 */
static void ul_alloc_buffer_wrapper(void* context,
                                     ul_output_msg_buffer& buffer,
                                     const slot_command_api::pusch_params& params)
{
    // Dummy allocation - currently unused
    (void)context;
    (void)buffer;
    (void)params;
}

/**
 * Static wrapper for UL slot callback - allows using function pointer instead of std::function
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param nCrc Number of CRC errors
 * @param buffer Output message buffer
 * @param slot Slot indication
 * @param params PUSCH parameters
 * @param out PUSCH output data
 * @param puschStatPrms PUSCH static parameters
 */
static void ul_slot_callback_wrapper(void* context,
                                      uint32_t nCrc,
                                      ul_output_msg_buffer& buffer,
                                      const slot_command_api::slot_indication& slot,
                                      const slot_command_api::pusch_params& params,
                                      ::cuphyPuschDataOut_t const* out,
                                      ::cuphyPuschStatPrms_t const* puschStatPrms)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    (void)nCrc;  // Unused parameter
    (void)buffer;  // Unused parameter
    
    if (out != nullptr)
    {
        //TODO:
        auto crcFails = phy_instance->send_crc_indication(slot, params, out, puschStatPrms);
        phy_instance->send_rx_data_indication(slot, params, out, puschStatPrms);
        // bool allCrcFail{crcFails == params.cell_grp_info.nUes};
        // if (!allCrcFail)
        // {
        //     phy_instance->send_rx_data_indication(slot, params, out, puschStatPrms);
        // }
        // else
        // {
        //     NVLOGI_FMT(TAG, "NO ULSCH Indication due to CRC {} errors", crcFails);
        //     // ul_crc_err_total[params.cell_index_list[0]]+=crcFails;
        //     // ul_crc_err[params.cell_index_list[0]]+=crcFails;
        // }

        if(out->totNumUciSegs > 0)
        {
            phy_instance->send_uci_indication(slot, params, *out, puschStatPrms);
        }

        if(phy_instance->getPnMeasurement())
            phy_instance->send_rx_pe_noise_var_indication(slot, params, out, puschStatPrms);
    }
    else
    {
        NVLOGD_FMT(TAG, "{}: No CRC or ULSCH indication", __FUNCTION__);
    }
}

/**
 * Static wrapper for UL PRACH callback - allows using function pointer instead of std::function
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param slot Slot indication
 * @param params PRACH parameters
 * @param num_detectedPrmb Number of detected preambles
 * @param prmbIndex_estimates Preamble index estimates
 * @param prmbDelay_estimates Preamble delay estimates
 * @param prmbPower_estimates Preamble power estimates
 * @param ant_rssi Antenna RSSI
 * @param rssi RSSI
 * @param interference Interference measurements
 */
static void ul_prach_callback_wrapper(void* context,
                                       slot_command_api::slot_indication& slot,
                                       const slot_command_api::prach_params& params,
                                       const uint32_t* num_detectedPrmb,
                                       const void* prmbIndex_estimates,
                                       const void* prmbDelay_estimates,
                                       const void* prmbPower_estimates,
                                       const void* ant_rssi,
                                       const void* rssi,
                                       const void* interference)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->send_rach_indication(slot, params, num_detectedPrmb, prmbIndex_estimates,
                                       prmbDelay_estimates, prmbPower_estimates,
                                       ant_rssi, rssi, interference);
}

/**
 * Static wrapper for UL UCI callback 2 - allows using function pointer instead of std::function
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param slot Slot indication
 * @param params PUCCH parameters
 * @param outParams PUCCH output data
 */
static void ul_uci_callback2_wrapper(void* context,
                                      slot_command_api::slot_indication& slot,
                                      const slot_command_api::pucch_params& params,
                                      const cuphyPucchDataOut_t& outParams)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->send_uci_indication(slot, params, outParams);
}

/**
 * Static wrapper for UL UCI early callback - allows using function pointer instead of std::function
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param slot Slot indication
 * @param params PUSCH parameters
 * @param out PUSCH output data
 * @param puschStatPrms PUSCH static parameters
 * @param t0_original Original T0 timestamp
 */
static void ul_uci_early_callback_wrapper(void* context,
                                           const slot_command_api::slot_indication& slot,
                                           const slot_command_api::pusch_params& params,
                                           ::cuphyPuschDataOut_t const* out,
                                           ::cuphyPuschStatPrms_t const* puschStatPrms,
                                           nanoseconds t0_original)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    if(out->isEarlyHarqPresent && out->isEarlyHarqPresentCellMask != 0)
    {
#ifdef SCF_FAPI_10_04
        phy_instance->send_early_uci_indication(slot, params, *out, puschStatPrms, t0_original);
#endif
    }
    else
    {
        NVLOGW_FMT(TAG, "{}: SFN {}.{} Early UCI callback called, when early harq is not present",
                   __FUNCTION__, slot.sfn_, slot.slot_);
    }
}

/**
 * Static wrapper for DL slot callback
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param params PDSCH parameters
 */
static void dl_slot_callback_wrapper(void* context, const slot_command_api::pdsch_params* params)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->on_dl_tb_processed_callback(params);
}

/**
 * Static wrappers for FH prepare callbacks - different template instantiations
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param grp_cmd Cell group command
 * @param cell Cell ID
 */
static void fh_prepare_callback_wrapper_tff(void* context,
                                             slot_command_api::cell_group_command* grp_cmd,
                                             uint8_t cell)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->fh_prepare_callback_wrapper_tff(grp_cmd, cell);
}

static void fh_prepare_callback_wrapper_tft(void* context,
                                             slot_command_api::cell_group_command* grp_cmd,
                                             uint8_t cell)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->fh_prepare_callback_wrapper_tft(grp_cmd, cell);
}

static void fh_prepare_callback_wrapper_ttf(void* context,
                                             slot_command_api::cell_group_command* grp_cmd,
                                             uint8_t cell)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->fh_prepare_callback_wrapper_ttf(grp_cmd, cell);
}

static void fh_prepare_callback_wrapper_ttt(void* context,
                                             slot_command_api::cell_group_command* grp_cmd,
                                             uint8_t cell)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->fh_prepare_callback_wrapper_ttt(grp_cmd, cell);
}

/**
 * Static wrapper for FH BFW coeff usage done callback
 * 
 * @param context Unused (nullptr)
 * @param header_addr BFW coefficient header address
 */
static void fh_bfw_coeff_usage_done_wrapper(void* context, uint8_t* header_addr)
{
    (void)context;
    NVLOGD_FMT(TAG, "{}: callback_fn: fh_bfw_coeff_usage_done_fn {}", __FUNCTION__, *header_addr);
    *header_addr = BFW_COFF_MEM_FREE;
}

/**
 * Static wrapper for DL TX error callback
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param slot Slot indication
 * @param msg_id Message ID
 * @param error_id Error ID
 * @param cell_idx_list List of cell indices
 * @param num_cells Number of cells
 */
static void dl_tx_error_wrapper(void* context,
                                 const slot_command_api::slot_indication& slot,
                                 uint16_t msg_id,
                                 uint16_t error_id,
                                 std::array<uint32_t,DL_MAX_CELLS_PER_SLOT>& cell_idx_list,
                                 uint8_t num_cells)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    for(int i = 0; i < num_cells; ++i)
    {
        phy_instance->send_error_indication_l1((scf_fapi_message_id_e)msg_id,
                                                (scf_fapi_error_codes_t)error_id,
                                                slot.sfn_, slot.slot_,
                                                cell_idx_list[i]);
    }
}

/**
 * Static wrapper for UL TX error callback
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param slot Slot indication
 * @param msg_id Message ID
 * @param error_id Error ID
 * @param cell_idx_list List of cell indices
 * @param num_cells Number of cells
 * @param log_info Whether to log info
 */
static void ul_tx_error_wrapper(void* context,
                                 const slot_command_api::slot_indication& slot,
                                 uint16_t msg_id,
                                 uint16_t error_id,
                                 std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& cell_idx_list,
                                 uint8_t num_cells,
                                 bool log_info)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    for(int i = 0; i < num_cells; ++i)
    {
        phy_instance->send_error_indication_l1((scf_fapi_message_id_e)msg_id,
                                                (scf_fapi_error_codes_t)error_id,
                                                slot.sfn_, slot.slot_,
                                                cell_idx_list[i], log_info);
    }
}

/**
 * Static wrapper for UL free HARQ buffer callback
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param freed_harq_buffer_data Released HARQ buffer information
 * @param params PUSCH parameters
 * @param sfn System frame number
 * @param slot Slot number
 */
static void ul_free_harq_buffer_wrapper(void* context,
                                         const ReleasedHarqBufferInfo& freed_harq_buffer_data,
                                         slot_command_api::pusch_params* params,
                                         uint16_t sfn,
                                         uint16_t slot)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->send_released_harq_buffer_error_indication(freed_harq_buffer_data, params, sfn, slot);
}

/**
 * Static wrapper for SRS callback - allows using function pointer instead of std::function
 * 
 * @param context Pointer to phy instance (casted from void*)
 * @param buffer Output message buffer
 * @param slot Slot indication
 * @param params SRS parameters
 * @param out SRS output data
 * @param srsStatPrms SRS static parameters
 * @param srs_order_cell_timeout_list Timeout list per cell
 */
static void srs_callback_wrapper(void* context,
                                  ul_output_msg_buffer& buffer,
                                  const slot_command_api::slot_indication& slot,
                                  const slot_command_api::srs_params& params,
                                  ::cuphySrsDataOut_t const* out,
                                  ::cuphySrsStatPrms_t const* srsStatPrms,
                                  const std::array<bool,UL_MAX_CELLS_PER_SLOT>& srs_order_cell_timeout_list)
{
    auto* phy_instance = static_cast<scf_5g_fapi::phy*>(context);
    phy_instance->send_srs_indication(slot, params, out, srsStatPrms, srs_order_cell_timeout_list);
}

void phy::on_dl_tb_processed_callback(const slot_command_api::pdsch_params* params)
{
    phy_module().on_dl_tb_processed(params);
}

void phy::fh_prepare_callback_wrapper_tff(slot_command_api::cell_group_command* grp_cmd, uint8_t cell)
{
    LegacyCellGroupFhContext fh_context(*grp_cmd, cell);
    fh_callback<true,false,false>(fh_context, get_slot_detail(grp_cmd->slot.slot_3gpp));
}

void phy::fh_prepare_callback_wrapper_tft(slot_command_api::cell_group_command* grp_cmd, uint8_t cell)
{
    LegacyCellGroupFhContext fh_context(*grp_cmd, cell);
    fh_callback<true,false,true>(fh_context, get_slot_detail(grp_cmd->slot.slot_3gpp));
}

void phy::fh_prepare_callback_wrapper_ttf(slot_command_api::cell_group_command* grp_cmd, uint8_t cell)
{
    LegacyCellGroupFhContext fh_context(*grp_cmd, cell);
    fh_callback<true,true,false>(fh_context, get_slot_detail(grp_cmd->slot.slot_3gpp));
}

void phy::fh_prepare_callback_wrapper_ttt(slot_command_api::cell_group_command* grp_cmd, uint8_t cell)
{
    LegacyCellGroupFhContext fh_context(*grp_cmd, cell);
    fh_callback<true,true,true>(fh_context, get_slot_detail(grp_cmd->slot.slot_3gpp));
}

void phy::create_ul_dl_callbacks(slot_command_api::callbacks &cb)
{
    cb.dl_cb.callback_fn = &dl_slot_callback_wrapper;
    cb.dl_cb.callback_fn_context = this;

    if ((!phy_module().config_options().precoding_enabled) && (!phy_module().config_options().bf_enabled))
    {
        cb.dl_cb.fh_prepare_callback_fn = &scf_5g_fapi::fh_prepare_callback_wrapper_tff;
        cb.dl_cb.fh_prepare_callback_fn_context = this;
    }
    else if ((!phy_module().config_options().precoding_enabled) && (phy_module().config_options().bf_enabled))
    {
        cb.dl_cb.fh_prepare_callback_fn = &scf_5g_fapi::fh_prepare_callback_wrapper_tft;
        cb.dl_cb.fh_prepare_callback_fn_context = this;
    }
    else if ((phy_module().config_options().precoding_enabled) && (!phy_module().config_options().bf_enabled))
    {
        cb.dl_cb.fh_prepare_callback_fn = &scf_5g_fapi::fh_prepare_callback_wrapper_ttf;
        cb.dl_cb.fh_prepare_callback_fn_context = this;
    }
    else // ((phy_module().config_options().precoding_enabled) && (phy_module().config_options().bf_enabled))
    {
        cb.dl_cb.fh_prepare_callback_fn = &scf_5g_fapi::fh_prepare_callback_wrapper_ttt;
        cb.dl_cb.fh_prepare_callback_fn_context = this;
    }

    cb.dl_cb.fh_bfw_coeff_usage_done_fn = &fh_bfw_coeff_usage_done_wrapper;
    cb.dl_cb.fh_bfw_coeff_usage_done_fn_context = nullptr;  // No instance needed

    cb.ul_cb.fh_bfw_coeff_usage_done_fn = &fh_bfw_coeff_usage_done_wrapper;
    cb.ul_cb.fh_bfw_coeff_usage_done_fn_context = nullptr;  // No instance needed

    cb.dl_cb.dl_tx_error_fn = &dl_tx_error_wrapper;
    cb.dl_cb.dl_tx_error_fn_context = this;

    cb.dl_cb.l1_exit_error_fn = [this] (uint16_t msg_id,uint16_t error_id,std::array<uint32_t,DL_MAX_CELLS_PER_SLOT>& cell_idx_list,uint8_t num_cells)
    {
            sfn_slot_t& ss_curr = phy_module().get_curr_sfn_slot();
            for(int i = 0; i < num_cells; ++i)
            {
                send_error_indication_l1((scf_fapi_message_id_e)msg_id,(scf_fapi_error_codes_t)error_id,ss_curr.u16.sfn,ss_curr.u16.slot,cell_idx_list[i]);
            }
    };

    /// 1 TB need to revisit for multiple TB
    cb.ul_cb.alloc_fn = &ul_alloc_buffer_wrapper;
    cb.ul_cb.alloc_fn_context = nullptr;  // Unused

    cb.ul_cb.callback_fn = &ul_slot_callback_wrapper;
    cb.ul_cb.callback_fn_context = this;

    cb.ul_cb.prach_cb_fn = &ul_prach_callback_wrapper;
    cb.ul_cb.prach_cb_context = this;

    cb.ul_cb.uci_cb_fn2 = &ul_uci_callback2_wrapper;
    cb.ul_cb.uci_cb_fn2_context = this;
    cb.ul_cb.callback_fn_early_uci = &ul_uci_early_callback_wrapper;
    cb.ul_cb.callback_fn_early_uci_context = this;
    cb.ul_cb.srs_cb_fn = &srs_callback_wrapper;
    cb.ul_cb.srs_cb_context = this;
    
    cb.ul_cb.ul_tx_error_fn = &ul_tx_error_wrapper;
    cb.ul_cb.ul_tx_error_fn_context = this;
    
    cb.ul_cb.ul_free_harq_buffer_fn = &ul_free_harq_buffer_wrapper;
    cb.ul_cb.ul_free_harq_buffer_fn_context = this;
}

void phy::send_phy_l1_enqueue_error_indication(uint16_t sfn,uint16_t slot,bool ul_slot,std::array<int32_t, MAX_CELLS_PER_SLOT>& cell_id_list,int32_t& index)
{
    auto it=std::find(cell_id_list.begin(),cell_id_list.end(),phy_config.cell_config_.carrier_idx);
    if(it==cell_id_list.end())
    {
        NVLOGW_FMT(TAG, "{}: L1 enqueue error SFN {}.{} cell_id={}", __FUNCTION__, sfn, slot, phy_config.cell_config_.carrier_idx);
        cell_id_list[index]=phy_config.cell_config_.carrier_idx;
        index++;

        scf_fapi_message_id_e msg_id = (ul_slot == 1) ? SCF_FAPI_UL_TTI_REQUEST : SCF_FAPI_DL_TTI_REQUEST;
        send_error_indication(msg_id, SCF_ERROR_CODE_L1_PROC_OBJ_UNAVAILABLE_ERR, sfn, slot);
    }
}

inline void ul_ind_ul_pn(uint16_t& pn, const cuphyPuschDataOut_t& out, int ue_index, const cuphyPuschStatPrms_t& puschStatPrms, float ul_configured_gain)
{
    if(puschStatPrms.enableSinrMeasurement)
    {
        float raw_noise;
        if(out.pNoiseVarPostEq)
            raw_noise = out.pNoiseVarPostEq[ue_index];
        else
            raw_noise = out.pNoiseVarPreEq[ue_index];
        NVLOGD_FMT(TAG, "{}: Raw PE Noise variance={} ul_configured_gain={}", __FUNCTION__, raw_noise, ul_configured_gain);
        float pn_dbm = raw_noise - ul_configured_gain; // Applied UL configured gain to raw noise variance measurement
        pn_dbm = (pn_dbm < -152.0) ? -152.0 : ((pn_dbm > 0) ? 0 : pn_dbm);
        pn = (uint16_t)(std::round((pn_dbm + 152.0) * 10));
    }
    else
    {
        NVLOGD_FMT(TAG, "{}: Sinr measurement is disabled", __FUNCTION__);
        pn = 0xffff;
    }
}

uint16_t phy::send_crc_indication(const slot_command_api::slot_indication& slot,
    const slot_command_api::pusch_params& params,
    ::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms)
{
    uint16_t nUes = params.cell_grp_info.nUes;
    uint16_t ue_index = 0;
    uint32_t start_ue_index = 0;
    uint16_t nCrcFail = 0;
    float rssi;
    for(int i = 0; i < params.cell_grp_info.nCells; ++i)
    {
        int cell_index = params.cell_index_list[i];
        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_index);

        NVLOGD_FMT(TAG, "{}: Sending CRC indication for cell_id = {}...", __FUNCTION__, cell_index);
        nv::phy_mac_transport& transport = phy_module().transport(cell_index);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return 0;
        }

        auto fapi = add_scf_fapi_hdr<scf_fapi_crc_ind_t>(msg_desc, SCF_FAPI_CRC_INDICATION, cell_index, false);
        auto& indication = *reinterpret_cast<scf_fapi_crc_ind_t*>(fapi);

        // HARQ Indication
        indication.sfn =  slot.sfn_;
        indication.slot = slot.slot_;
        indication.num_crcs = 0;
        NVLOGD_FMT(TAG, "{}: start_ue_index {} params.cell_grp_info.nCells {}", __FUNCTION__, start_ue_index, params.cell_grp_info.nCells);
        uint8_t* next = reinterpret_cast<uint8_t*>(indication.crc_info);
        uint16_t offset = 0;
        nCrcFail = 0;
        while(start_ue_index < nUes)
        {
            int i = start_ue_index;
            // Logic to split CRC indication for multiple cells
            if(phy_module().cell_group() &&
                phy_module().get_cell_id_from_stat_prm_idx(params.ue_info[i].pUeGrpPrm->pCellPrm->cellPrmStatIdx )!= cell_index)
            {
                break;
            }
            if((params.ue_info[i].pduBitmap & 0x1) == 0)
            {
                start_ue_index++;
                continue;
            }
            scf_fapi_crc_info_t& crcInfo = *(reinterpret_cast<scf_fapi_crc_info_t*>(next));
            uint32_t offsetsTbCrc = out->pStartOffsetsTbCrc[i];
            uint32_t crc_value = out->pTbCrcs[offsetsTbCrc];
            float rawTA = 0.0f;
            if (puschStatPrms->enableToEstimation) {
                rawTA = out->pTaEsts[i];
                NVLOGD_FMT(TAG, "{}: UE Idx [{}] Raw TO in microseconds= {}", __FUNCTION__, i, rawTA);
            }
            if (puschStatPrms->enableCfoCorrection) {
                float rawCfoEstHz{out->pCfoHz[i]};
                NVLOGD_FMT(TAG, "{}: UE Idx [{}] SFN {}.{} Raw CFO Estimate in Hz= {} puschStatPrms->enableWeightedAverageCfo={}", __FUNCTION__, i, static_cast<int>(indication.sfn), static_cast<int>(indication.slot), rawCfoEstHz, puschStatPrms->enableWeightedAverageCfo);
            }
            crcInfo.handle = params.scf_ul_tti_handle_list[i];
            crcInfo.tb_crc_status = !(crc_value == 0);
            if (crcInfo.tb_crc_status) {
                nCrcFail++;
            }
            crcInfo.rnti = params.ue_info[i].rnti;
            crcInfo.harq_id = params.ue_info[i].harqProcessId; // Update once we know what this is
#ifdef SCF_FAPI_10_04
            crcInfo.rapid = UINT8_MAX;
#endif

            // CB CRCs
            uint32_t cbCrcOffset = out->pStartOffsetsCbCrc[i];
            // crcInfo.num_cb = i == nUes - 1 ? out->totNumCbs - cbCrcOffset : out->pStartOffsetsCbCrc[i + 1] - cbCrcOffset;
            crcInfo.num_cb = 0; // No plan to support CBG yet, hard-code to 0
            int nbytes = (crcInfo.num_cb + 7) / 8;
            next = crcInfo.cb_crc_status + nbytes;

            memset(crcInfo.cb_crc_status, 0, nbytes);
            for (int cb_id = 0; cb_id < crcInfo.num_cb; cb_id ++)
            {
                if(out->pCbCrcs[cbCrcOffset + cb_id] != 0)
                {
                    crcInfo.cb_crc_status[cb_id / 8] |= 1 << cb_id % 8;
                }
            }
            NVLOGD_FMT(TAG, ">>> SCF_FAPI_CRC_INDICATION PHY CRC[{}]: RNTI={} HarqID={} TbCrcStatus={} NumCb={} cbCrcOffset[{}-{}]={}",
                    i, static_cast<unsigned>(crcInfo.rnti), crcInfo.harq_id, crcInfo.tb_crc_status, static_cast<unsigned>(crcInfo.num_cb), nUes, i, cbCrcOffset);

            auto *end = reinterpret_cast<scf_fapi_crc_end_info_t*>(next);
            uint16_t val = UINT16_MAX;
            ul_ind_rssi(val, *out, params, i, *puschStatPrms, getRssiMeasurement(), mplane.ul_gain_calibration, phy_module().rssi());
#ifdef SCF_FAPI_10_04
            end->measurement.rssi = val;
            int16_t sinr = 0;
            ul_ind_ul_cqi_fapi_10_04(sinr, *out, i, *puschStatPrms);
            end->measurement.ul_sinr_metric = sinr;
            uint16_t ta = UINT16_MAX;
            float non_prach_ta_offset_usec = 0.0f;
            ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
            end->measurement.timing_advance = ta;
            end->measurement.timing_advance_ns = static_cast<int16_t>(rawTA * 1000);// rawTA from cuPHY is usecs
            ul_ind_rsrp(val, *out, params, i, *puschStatPrms, getRsrpMeasurement(), mplane.ul_gain_calibration, phy_module().rsrp());
            end->measurement.rsrp = val;
            NVLOGD_FMT(TAG, ">>> SCF_FAPI_CRC_INDICATION 10.04 ul-sinr={} ta={} ta-ns={} rssi={} rsrp={}",
                   static_cast<int>(end->measurement.ul_sinr_metric),
                   static_cast<unsigned>(end->measurement.timing_advance),
                   static_cast<int>(end->measurement.timing_advance_ns),
                   static_cast<unsigned>(end->measurement.rssi),
                   static_cast<unsigned>(end->measurement.rsrp));
#else
            end->rssi = val;

            uint16_t ta = UINT16_MAX;
            ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec_, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
            end->timing_advance = ta;
            //NVLOGI_FMT(TAG, "{} rawTA ={:5.6f} TA reported={}", __FUNCTION__, rawTA, end->timing_advance);
            ul_ind_ul_cqi<uint8_t, 2>(end->ul_cqi, *out, i, *puschStatPrms);
            NVLOGD_FMT(TAG, "{}: CRC indication ul-sinr={} ta={} rssi={}", __FUNCTION__, end->ul_cqi,static_cast<unsigned>(end->timing_advance),static_cast<unsigned>(end->rssi));
#endif
            //crcInfo.nUEID = i; // CB ???
            offset += sizeof(scf_fapi_crc_info_t) + sizeof(scf_fapi_crc_end_info_t) + nbytes;
            next += sizeof(scf_fapi_crc_end_info_t);

            ++start_ue_index;
            ++indication.num_crcs;
        }
        if(nCrcFail > 0)
        {
            ul_crc_err_total[params.cell_index_list[i]]++;
            ul_crc_err[params.cell_index_list[i]]++;
        }
        fapi->length += offset;
        msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);

        // Send the message over the transport
        NVLOGD_FMT(TAG, "{}: CRC indication PHY SFN {}.{} num_crc={}", __FUNCTION__, static_cast<int>(indication.sfn), static_cast<int>(indication.slot), static_cast<int>(indication.num_crcs));
        transport.tx_send(msg_desc);
        transport.notify(IPC_NOTIFY_VALUE);
        metrics_.incr_tx_packet_count(SCF_FAPI_CRC_INDICATION);
    }
    return nCrcFail;
}

void phy::send_rx_data_indication(const slot_command_api::slot_indication& slot,
    const slot_command_api::pusch_params& params,
    ::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms)
{
    const uint32_t* tbSize = params.ue_tb_size;
    uint16_t nUes = params.cell_grp_info.nUes;
    auto& ue_prms = params.ue_info;
    uint8_t* tbdecoded = out->pTbPayloads;
    float* taEsts{out->pTaEsts};
    int tb_size = 0;
    int data_len = 0;
    auto tbcrcStatus{out->pTbCrcs};
    float rssi;
    uint32_t start_ue_index = 0;
    uint32_t end_ue_index = 0;
    NVLOGD_FMT(TAG, "{}: params.cell_grp_info.nCells {}", __FUNCTION__, params.cell_grp_info.nCells);
    for(int cell_index = 0; cell_index < params.cell_grp_info.nCells; ++cell_index)
    {
        bool found_crc = false;
        int tb_dest_offset = 0;
        tb_size = 0;
        data_len = 0;
        while( end_ue_index < nUes)
        {
            // Logic for cell_group to split RX_DATA indication for multiple cells
            if(phy_module().cell_group() &&
                phy_module().get_cell_id_from_stat_prm_idx(params.ue_info[end_ue_index].pUeGrpPrm->pCellPrm->cellPrmStatIdx) != params.cell_index_list[cell_index])
            {
                break;
            }
            else
            {
                found_crc = true;
            }
            ++end_ue_index;
        }
        if(!found_crc)
        {
            continue;
        }

        /// Start nvIPC message
        nv::phy_mac_transport& transport = phy_module().transport(params.cell_index_list[cell_index]);
        nv::phy_mac_msg_desc desc;
        desc.data_pool = NV_IPC_MEMPOOL_CPU_DATA;
        if(transport.tx_alloc(desc) < 0)
        {
            return;
        }

        desc.cell_id = params.cell_index_list[cell_index];

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(desc.cell_id);

        scf_fapi_body_header_t* hdr_ptr = nullptr;
        switch(phy_module().dl_tb_location())
        {
            case nv::TB_LOC_INLINE:
            case nv::TB_LOC_EXT_HOST_BUF:
            {
                hdr_ptr = add_scf_fapi_hdr<scf_fapi_rx_data_ind_t>(desc, SCF_FAPI_RX_DATA_INDICATION, params.cell_index_list[cell_index], false);
            }
            break;
            case nv::TB_LOC_EXT_GPU_BUF:
            default:
            break;
        }
        if (hdr_ptr == nullptr) {
            return;
        }
        auto &indication = *reinterpret_cast<scf_fapi_rx_data_ind_t*>(hdr_ptr);

        desc.msg_len = 0;
        desc.data_len = data_len; // desc.data_len was reset by add_scf_fapi_hdr(), set it again
        indication.slot = slot.slot_;
        indication.sfn = slot.sfn_;
#ifdef SCF_FAPI_10_04
        indication.control_length = 0;
#endif
        indication.num_pdus = 0;

        uint8_t* next = reinterpret_cast<uint8_t*>(indication.pdus);
        uint32_t offset = 0;
        ul_slot[desc.cell_id] ++;
        for(int i = start_ue_index; i < end_ue_index; ++i)
        {
            if((params.ue_info[i].pduBitmap & 0x1) == 0)
            {
                continue;
            }
            indication.num_pdus++;
            uint8_t* tb_src = out->pTbPayloads + out->pStartOffsetsTbPayload[i];
            auto& ulschPdu = *(reinterpret_cast<scf_fapi_rx_data_pdu_t*>(next));
            ulschPdu.rnti = ue_prms[i].rnti;
            ulschPdu.handle = params.scf_ul_tti_handle_list[i];
            ulschPdu.harq_id = ue_prms[i].harqProcessId;
#ifdef SCF_FAPI_10_04
            ulschPdu.rapid = UINT8_MAX;
#else
            float rawTA{taEsts[i]};
            NVLOGD_FMT(TAG, "{}: UE Idx [{}] Raw TO in microseconds= {}", __FUNCTION__, i, rawTA);

            uint16_t ta = UINT16_MAX;
            ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec_, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
            ulschPdu.timing_advance = ta;
            uint16_t val = UINT16_MAX;
            ul_ind_rssi(val, *out, params, i, *puschStatPrms, getRssiMeasurement(), mplane.ul_gain_calibration, phy_module().rssi());
            ulschPdu.rssi = val;

            ul_ind_ul_cqi<uint8_t, 2>(ulschPdu.ul_cqi, *out, i, *puschStatPrms);
            NVLOGD_FMT(TAG, "{}: RX data indication ul-sinr={} ta={} rssi={}", __FUNCTION__, ulschPdu.ul_cqi,static_cast<unsigned>(ulschPdu.timing_advance),static_cast<unsigned>(ulschPdu.rssi));
#endif
            uint32_t offsetsTbCrc = out->pStartOffsetsTbCrc[i];
            ///tbcrcStatus[offsetTbCrc] 0 - pass, >1 -fail
            if (!tbcrcStatus[offsetsTbCrc])
            {
                data_len += tbSize[i];
                //TODO FIXME PDU_LEN OVERFLOW
                ulschPdu.pdu_len = tbSize[i];
#ifdef SCF_FAPI_10_04
                ulschPdu.pdu_tag = 1;
#endif
                tb_size += tbSize[i];
                ul_thrput[desc.cell_id] += tbSize[i];
                uint8_t* tb_dest = nullptr;
                switch(phy_module().dl_tb_location())
                {
                    case nv::dl_tb_loc::TB_LOC_INLINE:
                    {
                        NVLOGW_FMT(TAG, "{}: Inline RX data buffers are not supported in L2 adapter", __FUNCTION__);
                    return;
                    }
                    break;
                    case nv::dl_tb_loc::TB_LOC_EXT_HOST_BUF:
                    {
            // NOTE: For nvipc RX TB data is populated in separate data buffer, this doesn't conform to SCF 222
            // Just ignore the ulschPdu.pdu[]
                tb_dest = static_cast<uint8_t*>(desc.data_buf) + tb_dest_offset;
            desc.data_len += tbSize[i];
            std::copy(tb_src, tb_src + tbSize[i], tb_dest);
            tb_dest_offset += tbSize[i];
                    }
                }
            }
            else
            {
                NVLOGD_FMT(TAG, "{}: Skipping UE Index {} RNTI {}", __FUNCTION__, i, ue_prms[i].rnti);
                ulschPdu.pdu_len = 0;
            }
            // reinterpret_cast<uint8_t*>(ulschPdu.pduData);
            // // The inevitable memcopy
            // // ptrdiff_t diff = tb_dest - data;
            // // NVLOGD_FMT(TAG,"ptr diff ={}", diff);
            // offset+= ULSCH_IND_FAPIMSG_OFFSET + ulschPdu.nPduLen;
            // data += offset;
            // hdr.msgLen+=offset;
#if 0
            for (uint i = 0; i < 30; i+=5) {
                NVLOGI_FMT(TAG, "{}: Data PDU[{}] = 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x}", __FUNCTION__, i, tb_dest[i],
                        tb_dest[i+1], tb_dest[i+2], tb_dest[i+3], tb_dest[i+4]);
            }
#endif
            offset += sizeof(scf_fapi_rx_data_pdu_t);
            next = ulschPdu.pdu;
        }

        hdr_ptr->length += offset;
        desc.msg_len = hdr_ptr->length + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t);

        NVLOGI_FMT(TAG, "{}: RX data indication PHY cell_id={} SFN {}.{} numPDUs={} tb_size={}", __FUNCTION__, desc.cell_id,
                static_cast<int>(indication.sfn),
                static_cast<int>(indication.slot),
                static_cast<int>(indication.num_pdus),
                tb_size);

        transport.tx_send(desc);
        transport.notify(IPC_NOTIFY_VALUE);
        metrics_.incr_tx_packet_count(SCF_FAPI_RX_DATA_INDICATION);
        // Increment start_ue_index to end
        start_ue_index = end_ue_index;
    }
}

void phy::send_rx_pe_noise_var_indication(const slot_command_api::slot_indication& slot,
    const slot_command_api::pusch_params& params,
    ::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms) {
    uint16_t nUes = params.cell_grp_info.nUes;
    uint16_t ue_index = 0;
    uint32_t start_ue_index = 0;

    for(int i = 0; i < params.cell_grp_info.nCells; ++i) {
        int cell_index = params.cell_index_list[i];
        nv::phy_mac_transport& transport = phy_module().transport(cell_index);
        nv::phy_mac_msg_desc msg_desc;
        if (transport.tx_alloc(msg_desc) < 0)
        {
            return;
        }

        nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_index);

        auto fapi = add_scf_fapi_hdr<scf_fapi_rx_measurement_ind_t>(msg_desc, SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION, cell_index, false);
        auto& indication = *reinterpret_cast<scf_fapi_rx_measurement_ind_t*>(fapi);
        indication.sfn =  slot.sfn_;
        indication.slot = slot.slot_;
        indication.num_meas = 0;
        uint8_t* next = reinterpret_cast<uint8_t*>(indication.meas_info);
        uint16_t offset = 0;
        NVLOGD_FMT(TAG, "{}: RX_PE_NOISE_VARIANCE_INDICATION for cell_id = {} start_ue_index={} params.cell_grp_info.nCells={} nUes={}", __FUNCTION__,
            cell_index, start_ue_index, params.cell_grp_info.nCells, nUes);

        while(start_ue_index < nUes)
        {
            int it = start_ue_index;
            uint16_t adjPn = UINT16_MAX;
            // Logic to split CRC indication for multiple cells
            if(phy_module().cell_group() &&
                phy_module().get_cell_id_from_stat_prm_idx(params.ue_info[it].pUeGrpPrm->pCellPrm->cellPrmStatIdx) != cell_index)
            {
                break;
            }
            scf_fapi_meas_t& pn_meas_info = *(reinterpret_cast<scf_fapi_meas_t*>(next));
            pn_meas_info.handle =  params.scf_ul_tti_handle_list[it];
            pn_meas_info.rnti = params.ue_info[it].rnti;
            ul_ind_ul_pn(adjPn, *out, it, *puschStatPrms,mplane.ul_gain_calibration);
            pn_meas_info.meas = adjPn;
            next+= sizeof(scf_fapi_meas_t);
            offset+=sizeof(scf_fapi_meas_t);
            start_ue_index++;
            indication.num_meas++;
        }
        fapi->length += offset;
        msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);

        // Send the message over the transport
        NVLOGD_FMT(TAG, "{}: RX_PE_NOISE_VARIANCE_INDICATION: PHY SFN {}.{} num_meas={} meas[0]={}", __FUNCTION__, static_cast<int>(indication.sfn), static_cast<int>(indication.slot), static_cast<int>(indication.num_meas), static_cast<unsigned>(indication.meas_info[0].meas));
        transport.tx_send(msg_desc);
        transport.notify(IPC_NOTIFY_VALUE);
        metrics_.incr_tx_packet_count(SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION);
    }
}

void phy::send_cell_config_response(int32_t cell_id, uint8_t response_code){
    nv::phy_mac_transport& transport = phy_module().transport(cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if(transport.tx_alloc(msg_desc) < 0)
    {
        return;
    }
    NVLOGC_FMT(TAG, "{}: Send CONFIG.response: cell_id={} error_code=0x{:X}", __FUNCTION__, cell_id, response_code);
    auto fapi = add_scf_fapi_hdr<scf_fapi_config_response_msg_t>(msg_desc, SCF_FAPI_CONFIG_RESPONSE, cell_id, false);
    auto rsp = reinterpret_cast<scf_fapi_config_response_msg_t*>(fapi);
    rsp->msg_body.error_code = response_code;
    rsp->msg_body.num_invalid_tlvs = 0;
    rsp->msg_body.num_idle_only_tlvs = 0;
    rsp->msg_body.num_running_only_tlvs = 0;
    rsp->msg_body.num_missing_tlvs = 0;
    int error_tlv_size = 0; // Add error TLV size if exists
    fapi->length += error_tlv_size;
    msg_desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
    transport.tx_send(msg_desc);
    transport.notify(IPC_NOTIFY_VALUE);
    metrics_.incr_tx_packet_count(SCF_FAPI_CONFIG_RESPONSE);
}

/* CV_MEM_BANK_CONFIG_REQUEST is sent by testMac to L1 to allocate SRS Chest buffers for the each cell and store the preloaded SRS channel estimates from the testVectors.
 * This is valid only for cuBB Testing for MU-MIMO scenarios where SRS Channels are not configured by testMAC */
void phy::on_cv_mem_bank_config_request(cv_mem_bank_config_request_body_t * cv_mem_bank_config_request_body, uint32_t cell_id,nv_ipc_msg_t& ipc_msg)
{
    NVLOGI_FMT(TAG, "{}: Received CV Mem Bank Config Request for Cell {}", __FUNCTION__, cell_id);
    if(state == fapi_state_t::FAPI_STATE_IDLE)
        //This message is expected in IDLE state
        NVLOGW_FMT(TAG, "{}: CV_MEMBANK_CONFIG_REQUEST received in state={}", __FUNCTION__, static_cast<uint32_t>(state.load()));
    scf_fapi_error_codes_t ret_code = SCF_ERROR_CODE_MSG_OK;
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    uint8_t muMIMO_enable_flag = get_mMIMO_enable_info();
    uint8_t enable_srs_flag = get_enable_srs_info();
    
    if(enable_srs_flag && muMIMO_enable_flag)
    {
        bool ret = phyDriver.allocSrsChesBuffPool(CV_MEM_BANK_CONFIG_REQUEST, cell_id, MAX_SRS_CHEST_BUFFERS_PER_CELL);
        uint32_t numUes = cv_mem_bank_config_request_body->numUes;
        uint8_t * ptr = reinterpret_cast<uint8_t*>(cv_mem_bank_config_request_body->cv_info);
        uint32_t srsChestBufferIndex = 0;
        for(uint32_t i=0; i < numUes; i++)
        {
            ue_cv_info* data = reinterpret_cast<ue_cv_info*> (ptr);
            srsChestBufferIndex = (data->rnti - 1);
            NVLOGD_FMT(TAG, "{}: rnti={}, srsChestBufferIndex={},reportType={} startPrbGrp={} srsPrbGrpSize={} nPrbGrps={} nGnbAnt={} nUeAnt={} offset={} \n",__FUNCTION__,
                static_cast<uint16_t>(data->rnti), srsChestBufferIndex,static_cast<uint8_t>(data->reportType),
                static_cast<uint16_t>(data->startPrbGrp),static_cast<uint32_t>(data->srsPrbGrpSize) , static_cast<uint16_t>(data->nPrbGrps),
                static_cast<uint8_t>(data->nGnbAnt), static_cast<uint8_t>(data->nUeAnt),static_cast<uint32_t>(data->offset));
            if(phyDriver.l1_cv_mem_bank_update(cell_id,data->rnti,srsChestBufferIndex,data->reportType,data->startPrbGrp,data->srsPrbGrpSize,data->nPrbGrps,
                data->nGnbAnt,data->nUeAnt,data->offset,(uint8_t*)ipc_msg.data_buf, data->startPrbGrp, data->nPrbGrps))
            {
                NVLOGW_FMT(TAG, "{}: CV memory bank update failed", __FUNCTION__);
                ret_code = SCF_ERROR_CODE_MSG_TX_ERR;
                goto send_response;
            }
            slot_command_api::srsChestBuffState curr_srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_NONE;
            if (!(phyDriver.l1_cv_mem_bank_get_buffer_state(cell_id, srsChestBufferIndex, &curr_srs_chest_buff_state)))
            {
                if(curr_srs_chest_buff_state == slot_command_api::SRS_CHEST_BUFF_REQUESTED)
                {
                    curr_srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_READY;
                    if(phyDriver.l1_cv_mem_bank_update_buffer_state(cell_id, srsChestBufferIndex, curr_srs_chest_buff_state))
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer state update failed", __FUNCTION__);
                        ret_code = SCF_ERROR_CODE_MSG_TX_ERR;
                        goto send_response;
                    }
                }
                else
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}:Updating SRS Chest buffer in invalid state. Current state={}",__FUNCTION__, static_cast<uint8_t>(curr_srs_chest_buff_state));
                    ret_code = SCF_ERROR_CODE_MSG_TX_ERR;
                    goto send_response;
                }
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer state fetching failed", __FUNCTION__);
                ret_code = SCF_ERROR_CODE_MSG_TX_ERR;
                goto send_response;
            }
            /*else
                release data->buffer */
            ptr += sizeof(ue_cv_info);
        }
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer allocation failed: enable_srs={} muMIMO_enable={} - both flags must be enabled together", __FUNCTION__, enable_srs_flag, muMIMO_enable_flag);
        ret_code = SCF_ERROR_CODE_MSG_TX_ERR;
        goto send_response;
    }
    
send_response:
    nv::phy_mac_transport& transport = phy_module().transport(cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if(transport.tx_alloc(msg_desc) < 0)
    {
        return;
    }
    auto fapi = add_scf_fapi_hdr<scf_fapi_generic_response_msg_t>(msg_desc, CV_MEM_BANK_CONFIG_RESPONSE, cell_id, false);
    auto rsp = reinterpret_cast<scf_fapi_generic_response_msg_t*>(fapi);
    rsp->msg_body.error_code = ret_code;
    transport.tx_send(msg_desc);
    transport.notify(IPC_NOTIFY_VALUE);
    metrics_.incr_tx_packet_count(CV_MEM_BANK_CONFIG_RESPONSE);
    return;
}

void phy::on_slot_error_indication(scf_fapi_error_ind_t& error_msg, nv_ipc_msg_t& ipc_msg)
{
    if(error_msg.err_code != SCF_ERROR_CODE_MSG_LATE_SLOT_ERR)
    {
        NVLOGI_FMT(TAG, "{}: Unhandled error code {}", __FUNCTION__, error_msg.err_code);
        return;
    }
    /* Cleanup slot command */
    phy_module().update_slot_cmds_indexes();
    phy_module().group_command()->reset();
    uint32_t i = 0;
    for (auto& phy: phy_module().PHY_instances())
    {
        phy.get().reset_slot(false);
        phy_module().cell_sub_command(i).reset();
        i++;
    }
}

inline uint8_t convertdBToFapi(float value)
{
    float dbMin = -64.0f;
    float dbMax = 63.0f;

    value = std::clamp(value, dbMin, dbMax);
    value = std::round(value * 2.0f) / 2.0f; // Rounding it to nearest 0.5 dB

    return static_cast<uint8_t>((value - dbMin) / 0.5f); // Map to uint8_t
}

void phy::send_srs_indication(const slot_command_api::slot_indication& slot,
                                        const slot_command_api::srs_params& params,
                                        ::cuphySrsDataOut_t const* out,
                                        ::cuphySrsStatPrms_t const* srsStatPrms,
                                        const std::array<bool,UL_MAX_CELLS_PER_SLOT>& srs_order_cell_timeout_list)
{
    float rssi;
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    uint32_t start_ue_index = 0;
    for(int i = 0; i < params.cell_grp_info.nCells; ++i)
    {
        int cell_index = params.cell_index_list[i];
        uint16_t cellPrmDynIdx = params.cell_dyn_info[i].cellPrmDynIdx;
        bool srsOrderTimeout = srs_order_cell_timeout_list[cellPrmDynIdx];

        for(int srs_ind_index = 0; srs_ind_index <= params.num_srs_ind_indexes[i]; srs_ind_index++)
        {
            nv::phy_mac_transport& transport = phy_module().transport(cell_index);
            NVLOGD_FMT(TAG, "{}: SFN {}.{} sending SRS indication for cell_index = {} srs.ind {} num PDUs = {}", __FUNCTION__,
                        slot.sfn_, slot.slot_, cell_index, srs_ind_index, params.num_srs_pdus_per_srs_ind[i][srs_ind_index]);
            if(params.num_srs_pdus_per_srs_ind[i][srs_ind_index] == 0)
                break;
#ifndef SCF_FAPI_10_04_SRS
            nv::phy_mac_msg_desc tx_msg_desc;
            if (transport.tx_alloc(tx_msg_desc) < 0)
            {
                return;
            }

            nv_ipc_msg_t desc = tx_msg_desc;
            nv_ipc_msg_t ori_desc = params.srs_indications[i][srs_ind_index];
            nv::phy_mac_msg_desc ori_msg_desc(ori_desc);
            transport.tx_release(ori_msg_desc);
#else
            nv_ipc_msg_t desc = params.srs_indications[i][srs_ind_index];
#endif
            //function add_scf_fapi_hdr will set data_len = 0, so take a copy
            uint32_t cv_offset = desc.data_len;
            NVLOGD_FMT(TAG, "{}: ipc_msg.msg_buf={} ipc_msg.data_buf={} ipc_msg.data_len={}",__FUNCTION__,desc.msg_buf,desc.data_buf,desc.data_len);

            auto fapi = add_scf_fapi_hdr<scf_fapi_srs_ind_t>(desc, SCF_FAPI_SRS_INDICATION, cell_index, false);
            auto& indication = *reinterpret_cast<scf_fapi_srs_ind_t*>(fapi);

            indication.sfn =  slot.sfn_;
            indication.slot = slot.slot_;
#ifdef SCF_FAPI_10_04_SRS
            indication.control_length = 0; /*reports are included inline */
#endif
            indication.num_pdus = 0;
            NVLOGD_FMT(TAG, "start_ue_index {} params.cell_grp_info.nCells {}", start_ue_index, params.cell_grp_info.nCells);
            uint8_t* next = reinterpret_cast<uint8_t*>(indication.srs_info);
            uint16_t srs_ind_msg_offset = 0;

            desc.data_len = cv_offset;
            cv_offset = 0;
            uint32_t numSRSPdus = 0;
            uint16_t numRBs = 0;

            while(numSRSPdus < params.num_srs_pdus_per_srs_ind[i][srs_ind_index])
            {
                // Bounds check to prevent out-of-bounds access
                if(start_ue_index >= params.cell_grp_info.nSrsUes)
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: start_ue_index {} exceeds nSrsUes {}, breaking loop", __FUNCTION__, start_ue_index, params.cell_grp_info.nSrsUes);
                    break;
                }
                
                slot_command_api::srsChestBuffState curr_srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_NONE;
#ifdef SCF_FAPI_10_04_SRS
                uint32_t usage = params.ue_info[start_ue_index].usage;
                while (usage != 0)
                {
#endif
                    int j = start_ue_index;
                    float rawTA = 0.0f;
                    rawTA = out->pSrsReports[j].toEstMicroSec;
                    NVLOGD_FMT(TAG, "{}: UE Idx [{}] Raw TO in microseconds= {}", __FUNCTION__, j, rawTA);
                    scf_fapi_srs_info_t& srsInfo = *(reinterpret_cast<scf_fapi_srs_info_t*>(next));
                    srsInfo.handle = params.ue_info[start_ue_index].handle;
                    srsInfo.rnti = params.ue_info[start_ue_index].rnti;
                    NVLOGD_FMT(TAG, "{}: cell_index {} SFN {}.{} SRS.IND  srsInfo.handle={} srsInfo.rnti={}", __FUNCTION__, cell_index,
                                    static_cast<unsigned>(slot.sfn_), static_cast<unsigned>(slot.slot_), static_cast<unsigned int>(srsInfo.handle), static_cast<unsigned int>(srsInfo.rnti ));
                    if (!(phyDriver.l1_cv_mem_bank_get_buffer_state(cell_index, params.ue_info[start_ue_index].srsChestBufferIndexL2, &curr_srs_chest_buff_state)))
                    {
                        /* Validate the SRS Chest buffer state and return error if the state is not in REQUESTED state during SRS.IND preparation.
                         * SRS Chest buffer state is not in REQUESTED state means the GPU Setup stage has failed. */
                        if(curr_srs_chest_buff_state != slot_command_api::SRS_CHEST_BUFF_REQUESTED)
                        {
                            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer invalid state current={}", __FUNCTION__, static_cast<uint8_t>(curr_srs_chest_buff_state));
                            return;
                        }
                    }
                    else
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer state fetching failed", __FUNCTION__);
                        return;
                    }
#ifdef SCF_FAPI_10_04_SRS
                    NVLOGD_FMT(TAG, "{}: srsOrderTimeout {} SRS.IND cell_index {} rnti {} start_ue_index= {} usage={} rawTA= {}", __FUNCTION__, srsOrderTimeout ? "true" : "false", cell_index, params.ue_info[start_ue_index].rnti, start_ue_index, usage, rawTA);
                    if(srsOrderTimeout)
                    {
                        srsInfo.timing_advance = 0xFFFF; // Invalid TA
                        srsInfo.timing_advance_ns = 0x8000; // SCF 10.02 through 222.05 have this as 0xffff, but that's -1, so within [-16800, 16800]
                    }
                    else
                    {
                        uint16_t ta = UINT16_MAX;
                        float non_prach_ta_offset_usec = 0.0f;
                        ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
                        srsInfo.timing_advance = ta;

                        if(!(std::isnan(rawTA)))
                        {
                            srsInfo.timing_advance_ns = static_cast<int16_t>(rawTA * 1000);// rawTA from cuPHY is usecs
                        }
                        else
                        {
                            /* Special Case: When rawTA is reportd as NaN from cuPHY API,  timing_advance value is 0 and we set the timing_advance_ns to -16800 nanoseconds */
                            srsInfo.timing_advance_ns = - 16800;
                        }
                        NVLOGD_FMT(TAG, "{}: UE Idx [{}]  Timing Advance Value = {} Timing Advance in nanoseconds = {}", __FUNCTION__, j, static_cast<uint16_t>(srsInfo.timing_advance), static_cast<int16_t>(srsInfo.timing_advance_ns));
                    }

                    if (usage & SRS_REPORT_FOR_BEAM_MANAGEMENT)
                    {
                        srsInfo.srs_usage = SRS_USAGE_FOR_BEAM_MANAGEMENT;
                        // Resetting the usage
                        usage &= ~SRS_REPORT_FOR_BEAM_MANAGEMENT;
                    }
                    else if (usage & SRS_REPORT_FOR_CODEBOOK)
                    {
                        srsInfo.srs_usage = SRS_USAGE_FOR_CODEBOOK;
                        usage &= ~SRS_REPORT_FOR_CODEBOOK;
                    }
                    else if (usage & SRS_REPORT_FOR_NON_CODEBOOK)
                    {
                        srsInfo.srs_usage = SRS_USAGE_FOR_NON_CODEBOOK;
                        usage &= ~SRS_REPORT_FOR_NON_CODEBOOK;
                    }
                    else
                    {
                        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS Usage Type {} not supported", __FUNCTION__, usage);
                        usage = 0;
                    }

                    /* The type of report included in or pointed to by Report TLV depends on the SRS usage:
                        Beam management:- 1: FAPIv3 Beamforming report (Table 3â€“131)
                        Codebook:- 1: Normalized Channel I/Q Matrix (Table 3â€“132)
                        nonCodebook:- 1: Normalized Channel I/Q Matrix (Table 3â€“132)
                        antennaSwitch:- 1: Channel SVD Representation (Table 3â€“133) */
                    srsInfo.report_type = 1;
                    if(srsOrderTimeout)
                    {
                        //- 0: null report (e.g. no report requested, or PHY could not compute a report). 
                        //L1 may choose to skip reporting for an UE if it is not scheduled
                        srsInfo.report_type = 0;
                        if (srsInfo.srs_usage == SRS_USAGE_FOR_BEAM_MANAGEMENT)
                        {
                            srsInfo.srs_report_tlv.tag = 2;
                        }
                        else if ((srsInfo.srs_usage & SRS_USAGE_FOR_CODEBOOK) || (srsInfo.srs_usage & SRS_USAGE_FOR_NON_CODEBOOK))
                        {
                            srsInfo.srs_report_tlv.tag = 1;
                        }
                        else
                        {
                            srsInfo.srs_report_tlv.tag = 0;
                            NVLOGE_FMT(TAG,AERIAL_FAPI_EVENT, "{}: Invalid SRS report type, SRS Usage Type {} not supported", __FUNCTION__, srsInfo.srs_usage);
                        }

                        srsInfo.srs_report_tlv.length = 0; // Because of SRS ORDER TIMEOUT, we'll not report anything
                        srsInfo.srs_report_tlv.value = 0;

                        next += sizeof(scf_fapi_srs_info_t);
                    }
                    else
                    {
                        if (srsInfo.srs_usage == SRS_USAGE_FOR_BEAM_MANAGEMENT)
                        {
                            /* Custom tag value of 2 is defined"
                                PDUsTag=2: Report (Table 3-131) is placed in the msg_buf portion of NVIPC message after the value field.
                                Filed "srsInfo.srs_report_tlv.value" is a 32 bit(4 Bytes) field set to 0xFFFFFFFF and should be ignored by L2.
                                Currently the stored BEAM_MANAGEMENT report is not 32 bit (4 Bytes) alligned.
                            */
                            srsInfo.srs_report_tlv.tag = 2;
                            srsInfo.srs_report_tlv.value = 0xFFFFFFFF;
                            //next = reinterpret_cast<uint8_t*>(&srsInfo.srs_report_tlv.value + sizeof(uint32_t));
                            next = reinterpret_cast<uint8_t*>(&srsInfo.srs_report_tlv.value);
                            next += sizeof(srsInfo.srs_report_tlv.value);
                            scf_fapi_v3_bf_report_t& srs_bf_report = *(reinterpret_cast<scf_fapi_v3_bf_report_t*>(reinterpret_cast<uint8_t*>(next)));
                            srs_bf_report.prg_size = out->pSrsChEstToL2[start_ue_index].prbGrpSize;
                            /* cuPHY-API provides a report for each unique frequency allocation
                            (i.e. averages across SRS symbols which have the same freq allocation) */
                            srs_bf_report.num_symbols = params.ue_info[start_ue_index].nSyms;
                            srs_bf_report.wideband_snr = convertdBToFapi(out->pSrsReports[start_ue_index].widebandSnr);
                            srs_bf_report.num_reported_symbols = 1;  // all symbols are aggregated
                            next = reinterpret_cast<uint8_t*>(srs_bf_report.num_prg_snr_info);
                            per_symbol_numPrg_snr_info_t& snr_per_prg = *(reinterpret_cast<per_symbol_numPrg_snr_info_t*>(next));
                            snr_per_prg.num_prgs = out->pSrsChEstToL2[start_ue_index].nPrbGrps;
                            next += sizeof(uint16_t);

                            const uint16_t prbGrpSize = out->pSrsChEstToL2[start_ue_index].prbGrpSize;
                            uint32_t rb_snr_buff_offset = out->pRbSnrBuffOffsets[start_ue_index];
                            const uint16_t numPrgs = snr_per_prg.num_prgs;

                            // Optimize based on known prbGrpSize values: 1, 2, 4, 16
                            // For 272 PRBs: size=1 (272 PRGs), size=2 (136 PRGs), size=4 (68 PRGs), size=16 (17 PRGs)
                            switch (prbGrpSize) {
                                case 1: {
                                    // No averaging needed - direct copy with 8x loop unrolling
                                    uint16_t prgIdx = 0;
                                    for (; prgIdx + 7 < numPrgs; prgIdx += 8) {
                                        snr_per_prg.rb_snr[prgIdx + 0] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 0]);
                                        snr_per_prg.rb_snr[prgIdx + 1] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 1]);
                                        snr_per_prg.rb_snr[prgIdx + 2] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 2]);
                                        snr_per_prg.rb_snr[prgIdx + 3] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 3]);
                                        snr_per_prg.rb_snr[prgIdx + 4] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 4]);
                                        snr_per_prg.rb_snr[prgIdx + 5] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 5]);
                                        snr_per_prg.rb_snr[prgIdx + 6] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 6]);
                                        snr_per_prg.rb_snr[prgIdx + 7] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx + 7]);
                                    }
                                    for (; prgIdx < numPrgs; ++prgIdx) {
                                        snr_per_prg.rb_snr[prgIdx] = convertdBToFapi(out->pRbSnrBuffer[rb_snr_buff_offset + prgIdx]);
                                    }
                                    break;
                                }

                                case 2: {
                                    // Average 2 PRBs per group with 8x loop unrolling
                                    uint16_t prgIdx = 0;
                                    for (; prgIdx + 7 < numPrgs; prgIdx += 8) {
                                        uint16_t prbIdx = prgIdx << 1;
                                        snr_per_prg.rb_snr[prgIdx + 0] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  0] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  1]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 1] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  2] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  3]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 2] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  4] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  5]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 3] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  6] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  7]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 4] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  8] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  9]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 5] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 10] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 11]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 6] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 12] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 13]) * 0.5f);
                                        snr_per_prg.rb_snr[prgIdx + 7] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 14] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 15]) * 0.5f);
                                    }
                                    for (; prgIdx < numPrgs; ++prgIdx) {
                                        const uint16_t prbIdx = prgIdx << 1;
                                        snr_per_prg.rb_snr[prgIdx] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 1]) * 0.5f);
                                    }
                                    break;
                                }

                                case 4: {
                                    // Average 4 PRBs per group with 4x loop unrolling
                                    uint16_t prgIdx = 0;
                                    for (; prgIdx + 3 < numPrgs; prgIdx += 4) {
                                        uint16_t prbIdx = prgIdx << 2;
                                        snr_per_prg.rb_snr[prgIdx + 0] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  0] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  1] +
                                                                                         out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  2] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  3]) * 0.25f);
                                        snr_per_prg.rb_snr[prgIdx + 1] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  4] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  5] +
                                                                                         out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  6] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  7]) * 0.25f);
                                        snr_per_prg.rb_snr[prgIdx + 2] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  8] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  9] +
                                                                                         out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 10] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 11]) * 0.25f);
                                        snr_per_prg.rb_snr[prgIdx + 3] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 12] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 13] +
                                                                                         out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 14] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 15]) * 0.25f);
                                    }
                                    for (; prgIdx < numPrgs; ++prgIdx) {
                                        const uint16_t prbIdx = prgIdx << 2;
                                        snr_per_prg.rb_snr[prgIdx] = convertdBToFapi((out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 1] +
                                                                                     out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 2] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 3]) * 0.25f);
                                    }
                                    break;
                                }

                                case 16: {
                                    // Average 16 PRBs per group - no outer loop unrolling needed (only 17 iterations)
                                    for (uint16_t prgIdx = 0; prgIdx < numPrgs; ++prgIdx) {
                                        const uint16_t prbIdx = prgIdx << 4;
                                        const float avg_snr = (out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  0] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  1] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  2] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  3] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  4] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  5] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  6] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  7] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  8] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx +  9] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 10] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 11] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 12] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 13] +
                                                              out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 14] + out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + 15]) * 0.0625f;
                                        snr_per_prg.rb_snr[prgIdx] = convertdBToFapi(avg_snr);
                                    }
                                    break;
                                }

                                default:
                                    // Fallback for unexpected sizes
                                    for (uint16_t prgIdx = 0; prgIdx < numPrgs; ++prgIdx) {
                                        const uint16_t prbIdx = prbGrpSize * prgIdx;
                                        float avg_snr = 0.0f;
                                        for (uint16_t i = 0; i < prbGrpSize; ++i) {
                                            avg_snr += out->pRbSnrBuffer[rb_snr_buff_offset + prbIdx + i];
                                        }
                                        avg_snr /= prbGrpSize;
                                        snr_per_prg.rb_snr[prgIdx] = convertdBToFapi(avg_snr);
                                    }
                                    break;
                            }

                            // Move pointer increment outside loop
                            next += snr_per_prg.num_prgs * sizeof(uint8_t);
                            srsInfo.srs_report_tlv.length = sizeof(scf_fapi_v3_bf_report_t) + sizeof(per_symbol_numPrg_snr_info_t) + 
                                                            (out->pSrsChEstToL2[start_ue_index].nPrbGrps * sizeof(uint8_t));
                            //TODO: Enable below code if we want BEAM_MANAGEMENT report to be alligned to 32 bit (4 Bytes)
#if 0
                            uint8_t padding_bytes = 4 -  (srsInfo.srs_report_tlv.length % 4);
                            next += padding_bytes;
                            srs_ind_msg_offset += (sizeof(scf_fapi_v3_bf_report_t) + sizeof(per_symbol_numPrg_snr_info_t) +
                                        (out->pSrsChEstToL2[start_ue_index].nPrbGrps * sizeof(uint8_t)) + padding_bytes);
#else
                            srs_ind_msg_offset += (sizeof(scf_fapi_v3_bf_report_t) + sizeof(per_symbol_numPrg_snr_info_t) +
                                        (out->pSrsChEstToL2[start_ue_index].nPrbGrps * sizeof(uint8_t)));
#endif
                            if(!usage)
                            {
                                cv_offset += ((sizeof(scf_fapi_norm_ch_iq_matrix_info_t)) + (out->pSrsChEstToL2[start_ue_index].nPrbGrps * 
                                            srsStatPrms->pCellStatPrms->nRxAntSrs * params.ue_info[start_ue_index].nAntPorts * IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4));
                            }
                            NVLOGD_FMT(TAG, "{}: length={} offset={} cv_offset={} next={}", __FUNCTION__, static_cast<uint32_t>(srsInfo.srs_report_tlv.length), srs_ind_msg_offset, cv_offset, reinterpret_cast<void*>(next));
                        }
                        else if ((srsInfo.srs_usage & SRS_USAGE_FOR_CODEBOOK) || (srsInfo.srs_usage & SRS_USAGE_FOR_NON_CODEBOOK))
                        {
                            /* Custom tag value of 1 is defined
                            1: Report (Table 3-132) is placed in the data_buf portion of NVIPC message for all SRS
                            PDUsTag=1 - Offset (in bytes) into the data_buf portion of NVIPC message for each SRS PDU */
                            srsInfo.srs_report_tlv.tag = 1;
                            scf_fapi_norm_ch_iq_matrix_info_t& srs_iq_info = *(reinterpret_cast<scf_fapi_norm_ch_iq_matrix_info_t*>(reinterpret_cast<uint8_t*>(desc.data_buf)+cv_offset));
                            srs_iq_info.norma_iq_repr = INDEX_IQ_REPR_32BIT_NORMALIZED_;
                            srs_iq_info.num_gnb_ant_elmts = srsStatPrms->pCellStatPrms->nRxAntSrs;
                            srs_iq_info.num_ue_srs_ports = params.ue_info[start_ue_index].nAntPorts;
                            srs_iq_info.prg_size = out->pSrsChEstToL2[start_ue_index].prbGrpSize;
                            srs_iq_info.num_prgs = out->pSrsChEstToL2[start_ue_index].nPrbGrps;
                            srsInfo.srs_report_tlv.length = (sizeof(scf_fapi_norm_ch_iq_matrix_info_t)) +
                                (srs_iq_info.num_prgs * srs_iq_info.num_gnb_ant_elmts * srs_iq_info.num_ue_srs_ports * IQ_REPR_32BIT_NORMALIZED_IQ_SIZE_4);
                            srsInfo.srs_report_tlv.value = cv_offset;
                            cv_offset += srsInfo.srs_report_tlv.length;
                            next += sizeof(scf_fapi_srs_info_t);
                            NVLOGD_FMT(TAG, "{}: offset={} offset in data_buf={} data_len={} data_buf={}",
                                    __FUNCTION__,
                                    static_cast<int>(srsInfo.srs_report_tlv.length),
                                    static_cast<unsigned int>(srsInfo.srs_report_tlv.value),
                                    desc.data_len,
                                    reinterpret_cast<void*>(desc.data_buf));
#if 0
                            uint8_t* buffer_ptr = (uint8_t*)desc.data_buf + (sizeof(scf_fapi_norm_ch_iq_matrix_info_t));
                            half2* value = (half2*)buffer_ptr;
                            NVLOGD_FMT(TAG, "{}: First SRS CV = {} + j{}", __FUNCTION__,(float)value[0].x, (float)value[0].y);
                            NVLOGD_FMT(TAG, "num_prg={}, num_gnb_ant_elmts={}, num_ue_srs_ports={}, cv_offset={}\n",
                                        static_cast<int>(srs_iq_info.num_prgs), static_cast<int>(srs_iq_info.num_gnb_ant_elmts),
                                        static_cast<int>(srs_iq_info.num_ue_srs_ports), static_cast<int>(cv_offset));

#endif
                        }
                        else
                        {
                            NVLOGE_FMT(TAG,AERIAL_FAPI_EVENT, "{}: Invalid SRS report type, SRS Usage Type {} not supported", __FUNCTION__, srsInfo.srs_usage);
                        }
                    }
#else
                    //FAPI 10.02
                    srsInfo.numSymbols = params.ue_info[start_ue_index].nSyms;
                    if(srsOrderTimeout)
                    {
                        srsInfo.timing_advance = 0xFFFF; // Invalid TA
                        srsInfo.wideBandSNR = 0xFF; // Invalid SNR
                    }
                    else
                    {
                        uint16_t ta = UINT16_MAX;
                        float non_prach_ta_offset_usec = 0.0f;
                        ul_ind_ta(ta, rawTA, non_prach_ta_offset_usec, phy_cell_params.mu, TA_MAX_NON_PRACH, TA_BASE_OFFSET);
                        srsInfo.timing_advance = ta;
                        srsInfo.wideBandSNR = convertdBToFapi(out->pSrsReports[start_ue_index].widebandSnr); //Note: refet to pucch or pusch, if any conversion to FAPI is needed or not?
                    }
                    srsInfo.numReportedSymbols = 1; // all symbols are aggregated
                    NVLOGD_FMT(TAG, "{}: numSymbols={} fapi_wideBandSNR={} cuphy_WBandSnr={} numReportedSymbols={}", __FUNCTION__,srsInfo.numSymbols,srsInfo.wideBandSNR,out->pSrsReports[start_ue_index].widebandSnr,srsInfo.numReportedSymbols);
                    uint32_t* pRbSnrBuffOffsets = out->pRbSnrBuffOffsets;
                    next = srsInfo.report;
                    if(srsOrderTimeout)
                    {
                        out->pSrsChEstToL2[start_ue_index].nPrbGrps = 0; // Reset numRB to '0' because of srs prder kernel timeout
                    }
                    for(int numRepSRSSymIndx = 0; numRepSRSSymIndx < srsInfo.numReportedSymbols; numRepSRSSymIndx++)
                    {
                        numRBs = out->pSrsChEstToL2[start_ue_index].prbGrpSize * out->pSrsChEstToL2[start_ue_index].nPrbGrps;
                        NVLOGD_FMT(TAG, "{}: numRepSRSSymIndx ={} numRBs={} prbGrpSize {} nPrbGrps {}", __FUNCTION__,numRepSRSSymIndx,numRBs, out->pSrsChEstToL2[start_ue_index].prbGrpSize, out->pSrsChEstToL2[start_ue_index].nPrbGrps);
                        *reinterpret_cast<uint16_t *>(next) = numRBs;
                        next += sizeof(uint16_t);

                        for(int prbIdx = 0; prbIdx < numRBs; prbIdx++)
                        {
                            float rbSnrCuphy = out->pRbSnrBuffer[pRbSnrBuffOffsets[start_ue_index] + prbIdx];
                            *reinterpret_cast<uint8_t *>(next) = convertdBToFapi(rbSnrCuphy);
                            NVLOGD_FMT(TAG, "{}: prbIdx={} rbSnrCuphy = {} fapiSnr={}", __FUNCTION__,prbIdx,rbSnrCuphy,(*reinterpret_cast<uint8_t *>(next)));
                            next += sizeof(uint8_t);
                        }
                    }
#endif
                    ++indication.num_pdus;
#ifdef SCF_FAPI_10_04_SRS
                    srs_ind_msg_offset += sizeof(scf_fapi_srs_info_t);
                }
#else
                    srs_ind_msg_offset += sizeof(scf_fapi_srs_info_t) + sizeof(scf_fapi_srs_sym_report_t) + (numRBs * sizeof(uint8_t));
#endif
                curr_srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_READY;
                /* Update the SRS Chest buffer state to READY once the SRS_IND is prepared and ready to be sent */
                if(phyDriver.l1_cv_mem_bank_update_buffer_state (cell_index, params.ue_info[start_ue_index].srsChestBufferIndexL2, curr_srs_chest_buff_state))
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: SRS chest buffer state update failed", __FUNCTION__);
                    return;
                }
                ++start_ue_index;
                numSRSPdus++;
            }
            fapi->length += srs_ind_msg_offset;
            desc.msg_len = fapi->length + sizeof(scf_fapi_body_header_t) + sizeof(scf_fapi_header_t);
            // Send the message over the transport
            NVLOGD_FMT(TAG, ">>> cell_index {} SCF_FAPI_SRS_INDICATION: PHY SFN {}.{}", cell_index, static_cast<int>(indication.sfn), static_cast<int>(indication.slot));
            nv::phy_mac_msg_desc msg_desc(desc);
            transport.tx_send(msg_desc);
            transport.notify(IPC_NOTIFY_VALUE);
            metrics_.incr_tx_packet_count(SCF_FAPI_SRS_INDICATION);
        }
    }
}


void phy::send_error_indication(scf_fapi_message_id_e msg_id,  scf_fapi_error_codes_t error_code, uint16_t sfn , uint16_t slot, bool log_info, uint16_t total_errors, nv::slot_limit_cell_error_t* cell_error, nv::slot_limit_group_error_t* group_error) {
    int32_t cell_idx = phy_config.cell_config_.carrier_idx;
    send_error_indication_l1(msg_id, error_code, sfn, slot, cell_idx, log_info, total_errors, cell_error, group_error);
}

void phy::send_error_indication_l1(scf_fapi_message_id_e msg_id, scf_fapi_error_codes_t error_code, uint16_t sfn , uint16_t slot, int32_t cell_idx, bool log_info, uint16_t total_errors, nv::slot_limit_cell_error_t* cell_error, nv::slot_limit_group_error_t* group_error) {
    nv::phy_mac_transport& transport = phy_module().transport(cell_idx);
    nv::phy_mac_msg_desc   msg_desc;
    if(transport.tx_alloc(msg_desc) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Skip sending ERR.ind for SFN {}.{} cell_id={} msg_id=0x{:02X} err_code=0x{:02X}", sfn, slot, cell_idx, +msg_id, +error_code);
        return;
    }

    if(log_info)
        NVLOGI_FMT(TAG, "Send Err.ind for SFN {}.{} cell_id={} msg_id=0x{:02X} err_code=0x{:02X}", sfn, slot, cell_idx, +msg_id, +error_code);
    else
        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Send Err.ind for SFN {}.{} cell_id={} msg_id=0x{:02X} err_code=0x{:02X}", sfn, slot, cell_idx, +msg_id, +error_code);
    auto fapi = add_scf_fapi_hdr<scf_fapi_error_ind_t>(msg_desc, SCF_FAPI_ERROR_INDICATION, cell_idx, false);
    auto rsp = reinterpret_cast<scf_fapi_error_ind_t*>(fapi);
    rsp->sfn = sfn;
    rsp->slot = slot;
    rsp->msg_id = msg_id;
    rsp->err_code = error_code;
#ifdef ENABLE_L2_SLT_RSP
    if (nv::TxNotificationHelper::getEnableTxNotification())       
    {
        // NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Enable Tx Notification is enabled for Error Indication msg_id={} err_code={}", +msg_id, +rsp->err_code);
        switch (msg_id)
        {
        case SCF_FAPI_DL_TTI_REQUEST:
        case SCF_FAPI_UL_DCI_REQUEST:
            {
                /* code */
                auto extension = reinterpret_cast<scf_fapi_error_extension_t*>(&rsp->extension[0]);
                extension->npdus_all = total_errors;
                extension->discard_factor_all = error_code;
                if (total_errors == 0) {
                    return;
                }
                auto offset = 0u;
                auto& pdu_info = extension->pdu_info;
                if ((error_code & SCF_FAPI_SSB_PBCH_L1_LIMIT_EXCEEDED) == SCF_FAPI_SSB_PBCH_L1_LIMIT_EXCEEDED) {
                    for (uint8_t i = 0; i < cell_error->ssb_pbch_errors.errors; i++) {
                        pdu_info[offset].pdu_type = DL_TTI_PDU_TYPE_SSB;
                        pdu_info[offset].rnti = 0;
                        pdu_info[offset].pdu_index = 0;
                        pdu_info[offset].discard_factor_pdu = SCF_FAPI_SSB_PBCH_L1_LIMIT_EXCEEDED;
                        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error extension SSB PBCH limit exceeded pdu_info[{}].pdu_type={} pdu_info[{}].rnti={} pdu_info[{}].pdu_index={} pdu_info[{}].discard_factor_pdu={} ssb_errors={}", offset, +pdu_info[offset].pdu_type, offset, +pdu_info[offset].rnti, offset, pdu_info[offset].pdu_index, offset, pdu_info[offset].discard_factor_pdu, cell_error->ssb_pbch_errors.errors);
                        offset++;
                    }
                }

                if ((error_code & SCF_FAPI_CSIRS_L1_LIMIT_EXCEEDED) == SCF_FAPI_CSIRS_L1_LIMIT_EXCEEDED) {
                    // NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "csirs_errors={}", cell_error->csirs_errors.errors);
                    for (uint8_t i = 0; i < cell_error->csirs_errors.errors; i++) {
                        pdu_info[offset].pdu_type = DL_TTI_PDU_TYPE_CSI_RS;
                        pdu_info[offset].rnti = 0;
                        pdu_info[offset].pdu_index = 0;
                        pdu_info[offset].discard_factor_pdu = SCF_FAPI_CSIRS_L1_LIMIT_EXCEEDED;
                        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error extension CSIRS limit exceeded pdu_info[{}].pdu_type={} pdu_info[{}].rnti={} pdu_info[{}].pdu_index={} pdu_info[{}].discard_factor_pdu={}", offset, +pdu_info[offset].pdu_type, offset, +pdu_info[offset].rnti, offset, pdu_info[offset].pdu_index, offset, pdu_info[offset].discard_factor_pdu);
                        offset++;
                    }
                }

                // TODO: Add PDCCH limit error handling
                if ((error_code & SCF_FAPI_PDCCH_L1_LIMIT_EXCEEDED) == SCF_FAPI_PDCCH_L1_LIMIT_EXCEEDED) {
                    // NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error extension PDCCH limit exceeded pdcch_errors={}", cell_error->pdcch_errors.dci_errors);
                    for (uint8_t i = 0; i < cell_error->pdcch_errors.dci_errors; i++) {
                        pdu_info[offset].pdu_type = DL_TTI_PDU_TYPE_PDCCH;
                        pdu_info[offset].rnti = cell_error->pdcch_errors.pdu_error_contexts[i].rnti;
                        pdu_info[offset].pdu_index = 0;
                        pdu_info[offset].discard_factor_pdu = SCF_FAPI_PDCCH_L1_LIMIT_EXCEEDED;
                        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error extension PDCCH limit exceeded pdu_info[{}].pdu_type={} pdu_info[{}].rnti={} pdu_info[{}].pdu_index={} pdu_info[{}].discard_factor_pdu={}", offset, +pdu_info[offset].pdu_type, offset, +pdu_info[offset].rnti, offset, pdu_info[offset].pdu_index, offset, pdu_info[offset].discard_factor_pdu);
                        offset++;
                    }
                }

                // TODO: Add PDSCH limit error handling
                if ((error_code & SCF_FAPI_PDSCH_L1_LIMIT_EXCEEDED) == SCF_FAPI_PDSCH_L1_LIMIT_EXCEEDED) {
                    for (uint8_t i = 0; i < cell_error->pdsch_pdu_error_contexts_info.pdsch_pdu_error_ctxt_num; i++) {
                        pdu_info[offset].pdu_type = DL_TTI_PDU_TYPE_PDSCH;
                        pdu_info[offset].rnti = cell_error->pdsch_pdu_error_contexts_info.pdsch_pdu_error_contexts[i].rnti;
                        pdu_info[offset].pdu_index = cell_error->pdsch_pdu_error_contexts_info.pdsch_pdu_error_contexts[i].pduIndex;
                        pdu_info[offset].discard_factor_pdu = SCF_FAPI_PDSCH_L1_LIMIT_EXCEEDED;
                        NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Error extension PDSCH limit exceeded pdu_info[{}].pdu_type={} pdu_info[{}].rnti={} pdu_info[{}].pdu_index={} pdu_info[{}].discard_factor_pdu={}", offset, +pdu_info[offset].pdu_type, offset, +pdu_info[offset].rnti, offset, pdu_info[offset].pdu_index, offset, pdu_info[offset].discard_factor_pdu);
                        offset++;
                    }
                }
            }
            break;
            default:
                break;
        }
    }
#endif
    transport.tx_send(msg_desc);
    transport.notify(IPC_NOTIFY_VALUE);
    metrics_.incr_tx_packet_count(SCF_FAPI_ERROR_INDICATION);
}

inline void phy::send_released_harq_buffer_error_indication(const ReleasedHarqBufferInfo &released_harq_buffer_info, const slot_command_api::pusch_params* params, uint16_t sfn, uint16_t slot) {
    for(int i =0; i< params->cell_grp_info.nCells; i++)
    {
        int32_t cell_idx = params->cell_dyn_info[i].cellPrmStatIdx;
        nv::phy_mac_transport& transport = phy_module().transport(cell_idx);
        nv::phy_mac_msg_desc   msg_desc;

        if(transport.tx_alloc(msg_desc) < 0)
        {
            NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "Skip sending ERR.ind for SFN {}.{} cell_id={} msg_id=0x{:02X} err_code=0x{:02X}", sfn, slot, cell_idx, +SCF_FAPI_ERROR_INDICATION, +SCF_ERROR_CODE_RELEASED_HARQ_BUFFER_INFO);
            continue;
        }
        scf_fapi_header_t *hdr = reinterpret_cast<scf_fapi_header_t*>(msg_desc.msg_buf);
        hdr->message_count     = 1;
        hdr->handle_id         = cell_idx;

        auto *body = reinterpret_cast<scf_fapi_body_header_t*>(hdr->payload);
        body->type_id          = SCF_FAPI_ERROR_INDICATION;
        scf_fapi_error_ind_with_released_harq_buffer_ext_t* msg = 
            reinterpret_cast<scf_fapi_error_ind_with_released_harq_buffer_ext_t*>(body);
        msg->sfn = sfn;
        msg->slot = slot;
        msg->msg_id = SCF_FAPI_ERROR_INDICATION;
        msg->err_code = SCF_ERROR_CODE_RELEASED_HARQ_BUFFER_INFO;
        
        msg->num_released_rscs = 0;
        for(int j = 0; j < released_harq_buffer_info.num_released_harq_buffers; j++)
        {
            if(released_harq_buffer_info.released_harq_buffer_list[j].cell_id == cell_idx)
            {  auto k = msg->num_released_rscs;
                msg->released_harq_buffers[k].rnti = released_harq_buffer_info.released_harq_buffer_list[j].rnti;
                msg->released_harq_buffers[k].harq_pid = released_harq_buffer_info.released_harq_buffer_list[j].harq_pid;
                msg->released_harq_buffers[k].sfn = released_harq_buffer_info.released_harq_buffer_list[j].sfn;
                msg->released_harq_buffers[k].slot = released_harq_buffer_info.released_harq_buffer_list[j].slot;
                msg->num_released_rscs++;
            }
        }
        if(msg->num_released_rscs == 0)
        {
            transport.tx_release(msg_desc);
            continue;
        }
        body->length           = static_cast<uint32_t>(sizeof(scf_fapi_error_ind_with_released_harq_buffer_ext_t) - sizeof(scf_fapi_body_header_t) + 
                                    msg->num_released_rscs * sizeof(scf_fapi_released_harq_buffer_info_t));
        msg_desc.msg_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + body->length;
        msg_desc.data_len = 0;
        msg_desc.cell_id = cell_idx;
        transport.tx_send(msg_desc);
        transport.notify(IPC_NOTIFY_VALUE);
        metrics_.incr_tx_packet_count(SCF_FAPI_ERROR_INDICATION);
        NVLOGI_FMT(TAG, "SFN {}.{} cell_idx={} num_released_rscs={} sent successfully", sfn, slot, cell_idx, reinterpret_cast<uint16_t>(msg->num_released_rscs));
    }
}


inline void phy::update_ssb_config() {
    //SSB case as in 3GPP TS 38.213 section 4.1
    ssb_case = nv::getSSBCase(phy_config.carrier_config_.dl_freq_abs_A/ 1000, phy_config.carrier_config_.ul_freq_abs_A/1000, phy_config.ssb_config_.sub_c_common);
    if(phy_config.carrier_config_.dl_freq_abs_A <= 3000000)
    {
        l_max = 4;
        auto& l_max_symbols = L_MAX_4_SYMBOLS[ssb_case];
        for(int i = 0; i < l_max; ++i)
        {
            ssb_slot_index.insert(l_max_symbols[i]/14);
        }
        lmax_symbol_list = L_MAX_4_SYMBOLS[ssb_case];
    }
    else if (phy_config.carrier_config_.dl_freq_abs_A <= 6000000)
    {
        l_max = 8;
        auto& l_max_symbols = L_MAX_8_SYMBOLS[ssb_case];
        for(int i = 0; i < l_max; ++i)
        {
            // Disabling memtrace here to suppress dynamic allocation due to ssb_slot_index.insert. This is only happening at the start up phase per cell with max number of ssb beams
            MemtraceDisableScope md; // disable memtrace while this variable is in scope
            ssb_slot_index.insert(l_max_symbols[i]/14);
        }
        lmax_symbol_list = L_MAX_8_SYMBOLS[ssb_case];
    }
    else
    {
        l_max = 64;
        auto& l_max_symbols = L_MAX_64_SYMBOLS[ssb_case];
        for(int i = 0; i < l_max; ++i)
        {
            ssb_slot_index.insert(l_max_symbols[i]/14);
        }
        lmax_symbol_list = L_MAX_64_SYMBOLS[ssb_case];
    }
    char buf[512];
    int offset = 0;
    for (auto it = ssb_slot_index.begin(); it != ssb_slot_index.end(); it++)
    {
        offset += snprintf(buf + offset, 32, " %u", *it);
    }
    buf[offset] = '\0';
    NVLOGI_FMT(TAG, "{}: SSB case={} dl_freq_abs_A={} l_max={} ssb_slot_index:{}", __FUNCTION__, +ssb_case, phy_config.carrier_config_.dl_freq_abs_A, l_max, buf);
}

inline void phy::update_prach_addln_configs() {
    switch (phy_config.cell_config_.frame_duplex_type) {
        case 0:
            if (phy_config.prach_config_.prach_conf_index >= 198 && phy_config.prach_config_.prach_conf_index <= 218)
            {
                prach_addln_config.l_ra = 139;
                prach_addln_config.n_ra_slot = 1;
                prach_addln_config.n_ra_dur = 12;
                prach_addln_config.n_ra_rb = 12;
                prach_addln_config.n_ra_t = 1;
            }
            break;
        case 1:
            if (phy_config.prach_config_.prach_conf_index >= 146 && phy_config.prach_config_.prach_conf_index <= 168)
            {
                prach_addln_config.l_ra = 139;
                prach_addln_config.n_ra_slot = 1;
                prach_addln_config.n_ra_dur = 12;
                prach_addln_config.n_ra_rb = 12;
                prach_addln_config.n_ra_t = 1;
            }
            break;
    }
}

inline void phy::update_prach_configs_l1(nv::PHYDriverProxy& phyDriver) {
    bool muMIMO_enable = static_cast<bool>(get_mMIMO_enable_info());
    auto numPrachFdOccasions = phy_config.prach_config_.num_prach_fd_occasions;
    int32_t scs_Khz = 15 * (1<<phy_config.ssb_config_.sub_c_common);
    auto guardBW_Khz = static_cast<float>(getGuardband(scs_Khz, phy_config.carrier_config_.ul_bandwidth));
    int32_t halfBW_Khz = - (static_cast<int32_t>(phy_config.carrier_config_.ul_bandwidth) * 1000)/2;
    auto cell_id = phy_config.cell_config_.carrier_idx;
    if(phyDriver.driver_exist()) {
        auto& prachStatParams =  phy_driver_info.prachStatParams;
        prachStatParams.nFdmOccasions = numPrachFdOccasions;
        prachStatParams.occaStartIdx = 0;
        prachStatParams.configurationIndex = phy_config.prach_config_.prach_conf_index;
        prachStatParams.restrictedSet = phy_config.prach_config_.restricted_set_config;
        if(muMIMO_enable)
        {
            // TODO: Remove this hardcondig once cuPHY parameter for receiving the digBFinterface is introduced.   
            prachStatParams.N_ant = 4; //phy_config.carrier_config_.num_rx_port;
        }
        else
        {
            prachStatParams.N_ant = phy_config.carrier_config_.num_rx_ants;
        }
        prachStatParams.FR = 1;
        prachStatParams.duplex = phy_config.cell_config_.frame_duplex_type;
        prachStatParams.mu = phy_config.prach_config_.prach_scs;
        prachStatParams.configurationIndex = phy_config.prach_config_.prach_conf_index;
        prachStatParams.restrictedSet = phy_config.prach_config_.restricted_set_config;
        
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
        if (mplane.lower_guard_bw >= static_cast<int32_t>(guardBW_Khz)) {
            guardBW_Khz = mplane.lower_guard_bw;
        } else {
            NVLOGW_FMT(TAG, "{}: Using Min guard Bandwidth {:5.4f} kHz for cell id {}", __FUNCTION__, guardBW_Khz, cell_id);
        }
        
        for (uint8_t i = 0; i < numPrachFdOccasions; i++) {
            nv::prach_root_seq& root_seq{phy_config.prach_config_.root_sequence[i]};
            root_seq.freqOffset = 2 * (halfBW_Khz + guardBW_Khz + (root_seq.k1 * scs_Khz * CUPHY_N_TONES_PER_PRB) + scs_Khz/2)/(scs_Khz);
            phy_driver_info.prach_configs.emplace_back();
            auto& prach_cuphy_params =  phy_driver_info.prach_configs.back();
            prach_cuphy_params.prachRootSequenceIndex = root_seq.seq_index;
            prach_cuphy_params.prachZeroCorrConf = root_seq.zero_conf;
        NVLOGI_FMT(TAG, "{}: prachStatParams: prach_configs[{}] RootSeqIndex {} ZCC {} k1 {} freqOffset {}", __FUNCTION__, phy_driver_info.prach_configs.size()-1,
            prach_cuphy_params.prachRootSequenceIndex,prach_cuphy_params.prachZeroCorrConf, root_seq.k1, root_seq.freqOffset);
        }
    }
}

inline void phy::update_phy_stat_configs_l1(nv::PHYDriverProxy& phyDriver) {
    phy_cell_params.phyCellId = phy_config.cell_config_.phy_cell_id;
    phy_cell_params.mu = phy_config.ssb_config_.sub_c_common;
    phy_cell_params.nPrbUlBwp = phy_config.carrier_config_.ul_grid_size[phy_cell_params.mu];
    phy_cell_params.nPrbDlBwp = phy_config.carrier_config_.dl_grid_size[phy_cell_params.mu];
    //Changes for CSI part 2, populate cuphyPuschCellStatPrm_t with default values.
    pusch_cell_stat_params.nCsirsPorts = 4;
    pusch_cell_stat_params.N1 = 2;
    pusch_cell_stat_params.N2 = 1;
    pusch_cell_stat_params.csiReportingBand = 0;
    pusch_cell_stat_params.codebookType = 0;
    pusch_cell_stat_params.codebookMode = 1;
    pusch_cell_stat_params.isCqi = 0;
    pusch_cell_stat_params.isLi = 0;
#ifdef SCF_FAPI_10_04
    pusch_cell_stat_params.nCsi2Maps = nCsi2Maps;
    pusch_cell_stat_params.pCsi2MapBuffer = static_cast<uint16_t*>(csi2MapCpuBuffer.get());
    pusch_cell_stat_params.pCsi2MapPrm = static_cast<cuphyCsi2MapPrm_t*>(csi2MapParamsCpuBuffer.get());
#else
    pusch_cell_stat_params.nCsi2Maps = 0;
    pusch_cell_stat_params.pCsi2MapBuffer = nullptr;
    pusch_cell_stat_params.pCsi2MapPrm = nullptr;
#endif
    phy_cell_params.pPuschCellStatPrms = &pusch_cell_stat_params;
    //Changes for CSI part 2, populate cuphyPucchCellStatPrm_t with default values.
    pucch_cell_stat_params.nCsirsPorts = 4;
    pucch_cell_stat_params.N1 = 2;
    pucch_cell_stat_params.N2 = 1;
    pucch_cell_stat_params.csiReportingBand = 0;
    pucch_cell_stat_params.codebookType = 0;
    pucch_cell_stat_params.codebookMode = 1;
    pucch_cell_stat_params.isCqi = 0;
    pucch_cell_stat_params.isLi = 0;
    phy_cell_params.pPucchCellStatPrms = &pucch_cell_stat_params;
    auto cell_id = phy_config.cell_config_.carrier_idx;
    if(phyDriver.driver_exist()) {
        // MemtraceDisableScope md;
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
        phy_driver_info.name = name();
        phy_driver_info.phy_stat = phy_cell_params;
        phy_driver_info.tti = nv::mu_to_ns(phy_config.ssb_config_.sub_c_common);
        phy_driver_info.slot_ahead = phy_module().get_slot_advance();
        phy_driver_info.mplane_id = mplane.mplane_id;
        phy_driver_info.pusch_aggr_factor = phy_config.vendor_config_.pusch_aggr_factor;
        NVLOGC_FMT(TAG, "{}: PHY Cell Id = {}, M-Plane Id= {}", __FUNCTION__, phy_driver_info.phy_stat.phyCellId, phy_driver_info.mplane_id);
    }
}

inline void phy::update_pusch_power_control_configs(nv::PHYDriverProxy& phyDriver) {
    auto cell_id = phy_config.cell_config_.carrier_idx;
    if(phyDriver.driver_exist()) {
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
        // PUSCH power control calculation uses YAML-configured parameters (fs_offset_ul, ul_bitwidth, max_amp_ul, exponent_ul)
        fs_offset_ul = mplane.fs_offset_ul;
        ul_bitwidth = static_cast<int>(mplane.ul_bit_width);
        int64_t minimumValueOfIAt0dBFS = -(std::pow(2, ul_bitwidth - 1) * std::pow(2, std::pow(2, mplane.exponent_ul) - 1));
        int64_t FS = std::pow(minimumValueOfIAt0dBFS, 2) * std::pow(2, -(fs_offset_ul));
        beta = (float)mplane.max_amp_ul / std::sqrt(FS);
        beta_sq = std::pow(beta, 2);
        NVLOGD_FMT(TAG, "{}: FS_OFFSET_UL {} ul_bitwidth {} minimumValueOfIAt0dBFS {} FS {} beta {} beta^2 {}", __FUNCTION__, fs_offset_ul, ul_bitwidth, minimumValueOfIAt0dBFS, FS, beta, beta_sq);
#ifdef SCF_FAPI_10_04
        phy_driver_info.is_early_harq_detection_enabled = (phy_module().get_phy_config().indication_instances_per_slot[nv::UCI_DATA_IND_IDX] ==  nv::MULTI_MSG_INSTANCE_PER_SLOT) ? true : false;
#else //10.02
        phy_driver_info.is_early_harq_detection_enabled = false;
#endif
    }
}

inline void phy::update_cell_stat_prm_idx() {
    if(cell_stat_prm_idx == INVALID_CELL_CFG_IDX) {
        cell_stat_prm_idx = phy_module().get_stat_prm_idx_to_cell_id_map_size();
        phy_module().insert_cell_id_in_stat_prm_map(get_carrier_id(), cell_stat_prm_idx);
    }    
}
    
inline void phy::update_prach_start_ro_index(nv::PHYDriverProxy& phyDriver) {
    phy_config.prach_config_.start_ro_index = phyDriver.l1_get_prach_start_ro_index(phy_cell_params.phyCellId);
    NVLOGD_FMT(TAG, "{}: start_ro_index={}", __FUNCTION__, phy_config.prach_config_.start_ro_index);
}

inline uint8_t phy::create_cell_l1(nv::PHYDriverProxy& phyDriver) {
    uint8_t error_code = 0;
   if(!cell_created)
    {
        auto create_rsp = phyDriver.l1_cell_create(phy_driver_info);
        if(create_rsp != 0)
        {
                error_code = SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }
        else
        {
                cell_created = true;
                update_cell_stat_prm_idx();
                update_prach_start_ro_index(phyDriver);
        }
    }
    return error_code;
}

inline uint8_t phy::create_cell_configs() {
    uint8_t error_code = SCF_ERROR_CODE_MSG_OK;
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    update_tx_rx_ants();
    update_ssb_config();
    update_prach_addln_configs();
    MemtraceDisableScope md;
    update_phy_stat_configs_l1(phyDriver);
    update_prach_configs_l1(phyDriver);
    auto cell_id = phy_config.cell_config_.carrier_idx;
    NVLOGI_FMT(TAG, "{}: config request: cell_id={} PHY_CELL_ID {} nPrbDlBwp {} nPrbUlBwp {} NUM_RX_ANT {} NUM_TX_ANT {} SCS_COMMON {} PRACH_SUBC_SPACING {}", __FUNCTION__,
        cell_id,
        phy_cell_params.phyCellId,
        phy_cell_params.nPrbDlBwp,
        phy_cell_params.nPrbUlBwp,
        phy_cell_params.nRxAnt,
        phy_cell_params.nTxAnt,
        phy_cell_params.mu,
        phy_cell_params.mu);
    if (phyDriver.driver_exist()) {
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
        update_pusch_power_control_configs(phyDriver);
        cuphyPrachStatPrms_t const prach_stat_params{
            .pOutInfo = nullptr,
            .nMaxCells = 1,
            .pCellPrms = &phy_driver_info.prachStatParams,
            .pOccaPrms = phy_driver_info.prach_configs.data(),
            .nMaxOccaProc = static_cast<uint16_t>(phy_driver_info.prach_configs.size()),
        };
        auto status = cuphyValidatePrachParams(&prach_stat_params);
        if (status != CUPHY_STATUS_SUCCESS) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "{}: cuphyValidatePrachParams returned error {} for cell_id={}", __FUNCTION__, cuphyGetErrorString(status), phy_config.cell_config_.carrier_idx);         
            return SCF_ERROR_CODE_MSG_INVALID_CONFIG;
        }
        error_code = create_cell_l1(phyDriver);

        phyDriver.l1_cell_update_cell_config(mplane.mplane_id, phy_cell_params.nPrbDlBwp, true);
        phyDriver.l1_cell_update_cell_config(mplane.mplane_id, phy_cell_params.nPrbUlBwp, false);
    } else {
        update_cell_stat_prm_idx();
    }
    update_cell_reconfig_params();
    Singleton<PdschFhContext>::makeInstance();
    Singleton<PdschCsirsFhContext>::makeInstance();
    return error_code;
}

inline void phy::copy_phy_configs_from(nv::phy_config& phy_config_origin) {
    uint8_t* orig = reinterpret_cast<uint8_t*>(&phy_config_origin);
    uint8_t* dest = reinterpret_cast<uint8_t*>(&phy_config);
    std::copy(orig, orig + sizeof(nv::phy_config), dest);
}

inline void phy::copy_csi2_maps_from(uint16_t nCsi2MapsOther, uint16_t* csi2MapBufferOther, cuphyCsi2MapPrm_t * csi2MapParamsBufferOther) {
    if (!nCsi2MapsOther || !csi2MapBufferOther || !csi2MapParamsBufferOther) {
        return;
    }
    nCsi2Maps = nCsi2MapsOther;
    auto size = csi2MapParamsBufferOther[nCsi2MapsOther - 1].csi2MapStartIdx + csi2MapParamsBufferOther[nCsi2MapsOther - 1].csi2MapSize;
    std::copy(csi2MapBufferOther, csi2MapBufferOther + size, csi2MapCpuBuffer.get());
    std::copy(csi2MapParamsBufferOther, csi2MapParamsBufferOther + nCsi2Maps, csi2MapParamsCpuBuffer.get());
}

inline int phy::update_dbt_pdu_table_ptr(int32_t cell_id, void* dbt_pdu_table_ptr) {
    if (phy_module().bf_enabled() == false) {
        return -1;
    }
    this->dbt_pdu_table_ptr = dbt_pdu_table_ptr;
    NVLOGD_FMT(TAG, "{}: [cell_id={}] dbt_pdu_table_ptr={}", __FUNCTION__, cell_id, dbt_pdu_table_ptr);
    auto& phyDriver = nv::PHYDriverProxy::getInstance();
    if(phyDriver.driver_exist() && dbt_pdu_table_ptr) {
        return phyDriver.l1_storeDBTPdu(cell_id, dbt_pdu_table_ptr);
    }
    return -1;
}

inline void phy::update_cell_state(fapi_state_t other_state) {
    if (state != other_state) {
        state = other_state;
    }
}

inline void phy::update_tx_rx_ants() {
    bool muMIMO_enable = static_cast<bool>(get_mMIMO_enable_info());
    
    phy_cell_params.nRxAntSrs = phy_config.carrier_config_.num_rx_ants;
    if (muMIMO_enable) {
        phy_cell_params.nRxAnt = phy_config.carrier_config_.num_rx_port;
        phy_cell_params.nTxAnt = phy_config.carrier_config_.num_tx_port;
    } else {
        phy_cell_params.nRxAnt = phy_config.carrier_config_.num_rx_ants;
        phy_cell_params.nTxAnt = phy_config.carrier_config_.num_tx_ants;
    }
}

inline void phy::update_cells_stats(int32_t cell_id) {
    if (cell_id >= total_cell_num) {
        total_cell_num = cell_id + 1;
    }
    reset_cell_stats(cell_id);
    metrics_.update_carrier_id(cell_id);
    ul_crc_err_total[cell_id] = 0;
}

inline void phy::copy_precoding_configs_to(int32_t cell_id) {
    auto pm_enabled = phy_module().pm_enabled();
    auto& pm_map = phy_module().pm_map();
    if (!pm_enabled || pm_map.empty()) {
        return;
    }
    for (auto& idx : first_config_req_pmidxes) {
        auto& pm = pm_map[idx];
        uint32_t newpmIdx = (idx & static_cast<uint32_t>(0x0000FFFF)) | cell_id << 16;
        MemtraceDisableScope md;
        pm_map.insert(std::make_pair(newpmIdx, pm));
    }
}

inline uint8_t phy::get_mMIMO_enable_info(){
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    uint8_t mMIMO_enable = 0;
    phyDriver.l1_mMIMO_enable_info(&mMIMO_enable);
    return mMIMO_enable;
}

inline uint8_t phy::get_enable_srs_info(){
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    uint8_t enable_srs = 0;
    phyDriver.l1_enable_srs_info(&enable_srs);
    return enable_srs;
}

inline ru_type phy::get_ru_type() {
    nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
    auto cell_id = phy_config.cell_config_.carrier_idx;
    ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
    return mplane.ru;
}

inline nv::slot_detail_t* phy::get_slot_detail(slot_command_api::slot_indication& slot) {
    auto ru = get_ru_type();
    if (ru == SINGLE_SECT_MODE) {
        
        auto repeat_slots = (1 << phy_config.ssb_config_.sub_c_common) *  (nv::get_duration(phy_config.tdd_table_.tdd_period_num)/std::chrono::duration<float, std::milli>(1));
        auto slot_index = slot.slot_ % static_cast<int>(repeat_slots);
        return &phy_config.tdd_table_.s_detail[slot_index];
    }
    return nullptr;
}

void phy::update_cell_reconfig_params() {
  cell_update_config.cell_config_ = phy_config.cell_config_;
  cell_update_config.carrier_config_ = phy_config.carrier_config_;
  cell_update_config.prach_config_.num_prach_fd_occasions = phy_config.prach_config_.num_prach_fd_occasions;
  std::copy(phy_config.prach_config_.root_sequence, phy_config.prach_config_.root_sequence + phy_config.prach_config_.num_prach_fd_occasions, cell_update_config.prach_config_.root_sequence);
  cell_update_config.prach_config_.prach_conf_index = phy_config.prach_config_.prach_conf_index;
  cell_update_config.prach_config_.restricted_set_config = phy_config.prach_config_.restricted_set_config;
  cell_update_config.prach_config_.start_ro_index = phy_config.prach_config_.start_ro_index;
}

void phy::update_phy_driver_info_reconfig(nv::PHYDriverProxy& phyDriver, const int32_t cell_id){
    cell_reconfig_phy_driver_info = phy_driver_info;
    cell_reconfig_phy_cell_params = phy_cell_params;
    cell_reconfig_phy_cell_params.phyCellId = cell_update_config.cell_config_.phy_cell_id;
    cell_reconfig_phy_driver_info.phy_stat.phyCellId = cell_reconfig_phy_cell_params.phyCellId;
    auto& prachStatParams =  cell_reconfig_phy_driver_info.prachStatParams;
    auto numPrachFdOccasions = cell_update_config.prach_config_.num_prach_fd_occasions;
    //Remove the existing config from prach_configs
    cell_reconfig_phy_driver_info.prach_configs.resize(0);
    prachStatParams.nFdmOccasions = numPrachFdOccasions;
    prachStatParams.configurationIndex = cell_update_config.prach_config_.prach_conf_index;
    prachStatParams.restrictedSet = cell_update_config.prach_config_.restricted_set_config;

    int32_t scs_Khz = 15 * (1<<phy_config.ssb_config_.sub_c_common);
    auto guardBW_Khz = static_cast<float>(getGuardband(scs_Khz, cell_update_config.carrier_config_.ul_bandwidth));
    int32_t halfBW_Khz = - (static_cast<int32_t>(cell_update_config.carrier_config_.ul_bandwidth) * 1000)/2;

    if(phyDriver.driver_exist())
    {
        ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
        if (mplane.lower_guard_bw >= static_cast<int32_t>(guardBW_Khz)) {
            guardBW_Khz = mplane.lower_guard_bw;
        } else {
            NVLOGW_FMT(TAG, "{}: Using Min guard Bandwidth {:5.4f} kHz for cell id {}", __FUNCTION__, guardBW_Khz, cell_id);
        }
    }
    NVLOGD_FMT(TAG, "{}: guardBW = {:5.4f} kHz", __FUNCTION__, guardBW_Khz);
    NVLOGI_FMT(TAG, "{}: prachStatParams: configIndex={} restrictedSet={} nFdmOcc={}", __FUNCTION__,
        prachStatParams.configurationIndex,prachStatParams.restrictedSet,prachStatParams.nFdmOccasions);
    for (uint8_t i = 0; i < numPrachFdOccasions; i++)
    {
        nv::prach_root_seq& root_seq{cell_update_config.prach_config_.root_sequence[i]};
        //-(Channel BW(in KHz)/2) + (GB (from lowest edge of BWP, in KHz) + (k1*SCS(in KHz)*12)+(SCS(in KHz)/2)]/(0.5*SCS(in KHz))
        root_seq.freqOffset = 2 * (halfBW_Khz + guardBW_Khz + (root_seq.k1 * scs_Khz * CUPHY_N_TONES_PER_PRB) + scs_Khz/2)/(scs_Khz);
        cell_reconfig_phy_driver_info.prach_configs.emplace_back();
        auto& prach_cuphy_params =  cell_reconfig_phy_driver_info.prach_configs.back();
        //prach_cuphy_params.cellPrmStatIdx should be filled by driver when it's creating cuphyPrachStatPrms_t for all cells
        prach_cuphy_params.prachRootSequenceIndex = root_seq.seq_index;
        prach_cuphy_params.prachZeroCorrConf = root_seq.zero_conf;
        NVLOGI_FMT(TAG, "{}: prachStatParams: prach_configs[{}] RootSeqIndex {} ZCC {} k1 {} = freqOffset {}", __FUNCTION__, cell_reconfig_phy_driver_info.prach_configs.size()-1,
            prach_cuphy_params.prachRootSequenceIndex,prach_cuphy_params.prachZeroCorrConf,root_seq.k1, root_seq.freqOffset );
    }
}

void phy::cell_update_success(nv::PHYDriverProxy& phyDriver, const int32_t cell_id) {
    phy_config.carrier_config_.dl_bandwidth = cell_update_config.carrier_config_.dl_bandwidth;
    phy_config.carrier_config_.ul_bandwidth = cell_update_config.carrier_config_.ul_bandwidth;
    phy_config.cell_config_.phy_cell_id = cell_update_config.cell_config_.phy_cell_id;
    phy_config.prach_config_.num_prach_fd_occasions = cell_update_config.prach_config_.num_prach_fd_occasions;
    std::copy(cell_update_config.prach_config_.root_sequence, cell_update_config.prach_config_.root_sequence + cell_update_config.prach_config_.num_prach_fd_occasions, phy_config.prach_config_.root_sequence);
    phy_config.prach_config_.prach_conf_index = cell_update_config.prach_config_.prach_conf_index;
    phy_config.prach_config_.restricted_set_config = cell_update_config.prach_config_.restricted_set_config;
    phy_config.prach_config_.start_ro_index = cell_update_config.prach_config_.start_ro_index;
    phy_driver_info = cell_reconfig_phy_driver_info;
    phy_cell_params.phyCellId = cell_reconfig_phy_cell_params.phyCellId;
}

void phy::handle_cell_config_response(int32_t cell_id, uint8_t response_code) { 
    switch (response_code) {
        case SCF_ERROR_CODE_MSG_OK: {
            NVLOGI_FMT(TAG, "{}: cell_id={} cell_config_response OK", __FUNCTION__, cell_id);
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            cell_update_success(phyDriver, cell_id);
            phy_config.prach_config_.start_ro_index = phyDriver.l1_get_prach_start_ro_index(phy_cell_params.phyCellId);
            NVLOGI_FMT(TAG, "{}: start_ro_index={}", __FUNCTION__, phy_config.prach_config_.start_ro_index);
        }
            break;
        default:
            break;
    }
    send_cell_config_response(cell_id, response_code);
}
} // namespace scf_5g_fapi
