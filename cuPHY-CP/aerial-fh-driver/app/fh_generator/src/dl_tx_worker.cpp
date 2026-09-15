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

#include "worker.hpp"

#include <algorithm>

#include "fh_generator.hpp"
#include "utils.hpp"
#include "gpudevice.hpp"
#include "yaml_parser.hpp"
#include "cuphy_pti.hpp"

namespace fh_gen
{

static void tx_complete_callback(void* addr, void* opaque)
{
}

static void fill_cplane_msg_template(const WorkerContext& info, std::array<aerial_fh::CPlaneMsgSendInfo, kMaxMsgSendInfoCount>& cplane_send_infos, std::array<aerial_fh::CPlaneSectionInfo, kMaxSectionCount>& cplane_sections)
{
    memset(&cplane_send_infos, 0, kMaxMsgSendInfoCount * sizeof(aerial_fh::CPlaneMsgSendInfo));
    memset(&cplane_sections[0], 0, kMaxSectionCount * sizeof(aerial_fh::CPlaneSectionInfo));

    for(size_t i = 0; i < kMaxMsgSendInfoCount; i++)
    {
        auto& cplane_section_hdr = cplane_send_infos[i].section_common_hdr;
        auto& radio_app_hdr          = cplane_section_hdr.sect_1_common_hdr.radioAppHdr;
        radio_app_hdr.dataDirection  = DIRECTION_DOWNLINK;
        radio_app_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
        radio_app_hdr.filterIndex    = ORAN_DEF_FILTER_INDEX;
        radio_app_hdr.sectionType    = ORAN_CMSG_SECTION_TYPE_1;

        auto iq_sample_size                            = static_cast<uint8_t>(info.ud_comp_info.method);
        auto iq_comp_method                            = (static_cast<uint8_t>(info.ud_comp_info.iq_sample_size) & 0xF) << 4;
        cplane_section_hdr.sect_1_common_hdr.udCompHdr = iq_comp_method | iq_sample_size;
    }

    for(size_t i = 0; i < kMaxSectionCount; i++)
    {
        auto& section_hdr     = cplane_sections[i].sect_1;
        section_hdr.sectionId = i;
        section_hdr.rb        = ORAN_RB_ALL;
        section_hdr.symInc    = ORAN_SYMCINC_NO;
        section_hdr.startPrbc = i;
        section_hdr.numPrbc   = 1;
        section_hdr.reMask    = ORAN_REMASK_ALL;
        section_hdr.ef        = ORAN_EF_NO;
        section_hdr.beamId    = ORAN_BEAMFORMING_NO;
    }
}

static void fill_uplane_msg_template(const WorkerContext& info, aerial_fh::UPlaneMsgSendInfo& uplane_msg_info)
{
    memset(&uplane_msg_info, 0, sizeof(uplane_msg_info));

    auto& radio_app_hdr          = uplane_msg_info.radio_app_hdr;
    radio_app_hdr.dataDirection  = DIRECTION_DOWNLINK;
    radio_app_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
    radio_app_hdr.filterIndex    = ORAN_DEF_FILTER_INDEX;

    auto& section_hdr   = uplane_msg_info.section_info;
    section_hdr.rb      = ORAN_RB_ALL;
    section_hdr.sym_inc = ORAN_SYMCINC_NO;
}

void fronthaul_generator_dl_tx_worker(Worker* worker)
{
    usleep(500);
    uint32_t cpu;
    int ret;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        THROW(StringBuilder() << "getcpu failed for " << __FUNCTION__);
    }

    NVLOGC_FMT(TAG,"Start DL TX U worker on CPU {}", cpu);

    PrimaryCtxGuard ctx_guard(0);

    auto& context = worker->get_context();
    // auto& nic = context.nic;
    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SLOT_ID;
    frame_cycle_time_ns *= context.slot_duration;
    //FIX ME hardcoded
    std::chrono::nanoseconds symbol_duration(35714);
    // Prepare U-plane message descriptor template
    // std::array<std::array<aerial_fh::UPlaneMsgSendInfo, 7168>, kMaxCells> uplane_msg_infos;
    std::array<aerial_fh::UPlaneMsgSendInfo, MAX_UPLANE_MSGS_PER_SLOT> uplane_msg_infos;
    int uplane_msg_infos_num = 0;
    aerial_fh::UPlaneMsgSendInfo uplane_msg_info;
    fill_uplane_msg_template(context, uplane_msg_info);

    aerial_fh::UPlaneTxCompleteNotification tx_complete_notification{
        .callback     = tx_complete_callback,
        .callback_arg = nullptr,
    };

    // GPU Comms setup
    aerial_fh::PreparePRBInfo prb_info{{nullptr}};
    // aerial_fh::TxRequestGpuPercell tx_v = {0};
    aerial_fh::TxRequestGpuPercell tx_v_cells[kMaxNicsSupported] = {};
    aerial_fh::TxRequestGpuCommHandle txrq_gpu[kMaxCells];
    CUdeviceptr prb_ptrs_dptr;
    CHECK_CU_THROW(cuMemAlloc(&prb_ptrs_dptr, sizeof(uint8_t*)*kMaxPrbsPerSymbol*kMaxSymbols*kMaxAntennas*kMaxCells));
    uint8_t** prb_ptrs = reinterpret_cast<uint8_t**>(prb_ptrs_dptr);
    ACCESS_ONCE(((uint32_t*)context.buffer_ready_gdr[0]->addrh())[0]) = 1;
    prb_info.ready_flag = (uint32_t*)context.buffer_ready_gdr[0]->addrd();
    prb_info.wait_val   = 1;
    prb_info.comm_start_evt = nullptr;
    prb_info.comm_copy_evt = nullptr;
    prb_info.comm_preprep_stop_evt = nullptr;
    prb_info.compression_stop_evt = nullptr;
    prb_info.comm_stop_evt = nullptr;
    prb_info.trigger_end_evt = nullptr;
    prb_info.enable_prepare_tracing = false;
    prb_info.enable_dl_cqe_tracing = false;
    prb_info.disable_empw = false;
    prb_info.prb_ptrs[0] = prb_ptrs;
    prb_info.num_antennas[0] = 1;
    prb_info.max_num_prb_per_symbol[0] = ORAN_MAX_PRB_X_SLOT;

    auto     prb_size       = get_prb_size(context.ud_comp_info.iq_sample_size, context.ud_comp_info.method);
    auto     symbol_size    = prb_size * ORAN_MAX_PRB_X_SLOT;
    uint32_t slot_index_tdd = 0;
    uint64_t test_slot_count = 0;

    // Make sure that time_anchor is in the future
    aerial_fh::Ns next_window = context.time_anchor;// + frame_cycle_time_ns;
    // wait_ns(next_window - context.tx_time_advance);

    do
    {
        auto oran_slot_number = context.oran_slot_iterator.get_next();
        CuphyPtiSetIndexScope cuphy_pti_index_scope(test_slot_count % CUPHY_PTI_INDEX_MAX);

        for(int i = 0; i < context.dl_tx_worker_context.peers.size(); ++i)
        {
            uplane_msg_infos_num = 0;
            for(auto const& tx : context.dl_tx_worker_context.uplane[i][slot_index_tdd])
            {
                uplane_msg_info.flow                      = tx.flow;
                uplane_msg_info.eaxcid                    = tx.eAxC;
                uplane_msg_info.tx_window.tx_window_start = next_window + tx.slot_offset;
                auto& radio_app_hdr      = uplane_msg_info.radio_app_hdr;
                radio_app_hdr.frameId    = oran_slot_number.frame_id;
                radio_app_hdr.subframeId = oran_slot_number.subframe_id;
                radio_app_hdr.slotId     = oran_slot_number.slot_id;
                radio_app_hdr.symbolId   = tx.symbol_id;

                uplane_msg_info.section_info.section_id = tx.section_id;

                for(auto& section: tx.section_list)
                {
                    if(section.num_prbu == ORAN_MAX_PRB_X_SLOT)
                    {
                        uplane_msg_info.section_info.start_prbu     = section.start_prbu;
                        uplane_msg_info.section_info.num_prbu       = 0;
                        uplane_msg_info.section_info.iq_data_buffer = static_cast<uint8_t*>(tx.iq_data_buffer) + (tx.symbol_id * symbol_size) + (section.start_prbu * prb_size);
                        memcpy(&uplane_msg_infos[uplane_msg_infos_num++], &uplane_msg_info, sizeof(uplane_msg_info));
                    }
                    else if (section.num_prbu > ORAN_MAX_PRB_X_SECTION)
                    {
                        uplane_msg_info.section_info.start_prbu     = section.start_prbu;
                        uplane_msg_info.section_info.num_prbu = ORAN_MAX_PRB_X_SECTION;
                        uplane_msg_info.section_info.iq_data_buffer = static_cast<uint8_t*>(tx.iq_data_buffer) + (tx.symbol_id * symbol_size) + (uplane_msg_info.section_info.start_prbu * prb_size);
                        memcpy(&uplane_msg_infos[uplane_msg_infos_num++], &uplane_msg_info, sizeof(uplane_msg_info));
                        uplane_msg_info.section_info.start_prbu = ORAN_MAX_PRB_X_SECTION;
                        uplane_msg_info.section_info.num_prbu = section.num_prbu - ORAN_MAX_PRB_X_SECTION;
                        uplane_msg_info.section_info.iq_data_buffer = static_cast<uint8_t*>(tx.iq_data_buffer) + (tx.symbol_id * symbol_size) + (uplane_msg_info.section_info.start_prbu * prb_size);
                        memcpy(&uplane_msg_infos[uplane_msg_infos_num++], &uplane_msg_info, sizeof(uplane_msg_info));
                    }
                    else
                    {
                        uplane_msg_info.section_info.start_prbu     = section.start_prbu;
                        uplane_msg_info.section_info.num_prbu       = section.num_prbu;
                        uplane_msg_info.section_info.iq_data_buffer = static_cast<uint8_t*>(tx.iq_data_buffer) + (tx.symbol_id * symbol_size) + (section.start_prbu * prb_size);
                        memcpy(&uplane_msg_infos[uplane_msg_infos_num++], &uplane_msg_info, sizeof(uplane_msg_info));
                    }
                }
            }
            std::chrono::nanoseconds start_time(uplane_msg_info.tx_window.tx_window_start);
            if(uplane_msg_infos_num > 0)
            {
                // NOTE: We cannot support different tx_time_advances due to the nature of this loop
                if(aerial_fh::prepare_uplane_gpu_comm(context.dl_tx_worker_context.peers[i], &uplane_msg_infos[0], uplane_msg_infos_num, &txrq_gpu[i], start_time, symbol_duration,false))
                {
                    NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT,"Failed to prepare U-plane");
                    THROW(StringBuilder() << "Failed to prepare U-plane");
                }
                if(int rc = aerial_fh::set_gpu_request_cell_idx(txrq_gpu[i], static_cast<uint8_t>(i)); rc != 0)
                {
                    NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "set_gpu_request_cell_idx failed (cell={}, rc={})", i, rc);
                    THROW(StringBuilder() << "set_gpu_request_cell_idx failed");
                }

                auto& tx_v = tx_v_cells[context.dl_tx_worker_context.peer_nic_ids[i]];
                tx_v.nic_name = context.dl_tx_worker_context.peer_nic_names[i];
                tx_v.tx_v_per_nic[tx_v.size] = txrq_gpu[i];
                tx_v.size++;
                prb_info.prb_ptrs[i] = &prb_ptrs[kMaxPrbsPerSymbol*kMaxSymbols*kMaxAntennas*i];
                // wait_ns(uplane_msg_infos[0].tx_window.tx_window_start - context.dl_u_enq_time_advance_ns * 2);
            }
        }
        if(uplane_msg_infos_num > 0)
        {
            for(int i = 0; i < kMaxNicsSupported; i++)
            {
                auto& tx_v = tx_v_cells[i];
                if(tx_v.size > 0)
                {
                    auto nic = context.fhgen->get_nic_handle_from_name(tx_v.nic_name);
                    if(nic == nullptr)
                    {
                        NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT,"send error nic not found");
                        THROW(StringBuilder() << "send error nic not found");
                    }
                    if(0 != aerial_fh::send_uplane_gpu_comm(nic, &tx_v, prb_info))
                    {
                        NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT,"aerial_fh::send_uplane_gpu_comm error");
                        THROW(StringBuilder() << "aerial_fh::send_uplane_gpu_comm error");
                    }
                    tx_v.size = 0;
                }
            }
        }

        slot_index_tdd = (slot_index_tdd + 1) % context.slot_count;
        next_window += context.slot_duration;
        wait_ns(next_window - context.dl_u_enq_time_advance_ns * 2);
        ++test_slot_count;
    } while(!worker->exit_signal() && (context.test_slots == 0 || test_slot_count < context.test_slots));
    sleep(3);
    NVLOGC_FMT(TAG, "DL TX Worker on CPU {} exit, test_slot_count {} reached test_slots {}", cpu, test_slot_count, context.test_slots);
    context.fhgen->set_workers_exit_signal();
    sleep(1);
    context.fhgen->set_exit_signal();
}

void prepare_c_plane(const struct fh_gen::OranSlotNumber& oran_slot_number, std::array<aerial_fh::CPlaneMsgSendInfo, kMaxMsgSendInfoCount>& cplane_msg_infos, std::array<aerial_fh::CPlaneSectionInfo, kMaxSectionCount>& cplane_sections, const fh_gen::CPlaneTX &tx, int& cplane_msg_infos_num, int& cplane_sections_num, uint64_t next_window)
{
    cplane_msg_infos[cplane_msg_infos_num].sections = &cplane_sections[cplane_sections_num];
    cplane_msg_infos[cplane_msg_infos_num].flow                      = tx.flow;
    cplane_msg_infos[cplane_msg_infos_num].tx_window.tx_window_start = next_window + tx.slot_offset;
    cplane_msg_infos[cplane_msg_infos_num].data_direction = tx.direction;
    auto& radio_app_hdr         = cplane_msg_infos[cplane_msg_infos_num].section_common_hdr.sect_1_common_hdr.radioAppHdr;
    radio_app_hdr.frameId       = oran_slot_number.frame_id;
    radio_app_hdr.subframeId    = oran_slot_number.subframe_id;
    radio_app_hdr.slotId        = oran_slot_number.slot_id;
    radio_app_hdr.startSymbolId = tx.symbol_id;
    radio_app_hdr.dataDirection = tx.direction;
    radio_app_hdr.numberOfSections = 0;
    radio_app_hdr.sectionType = 1; // FIXME PRACH section type 3
    for(auto const& section_info: tx.section_list)
    {
        auto& section = cplane_sections[cplane_sections_num];
        if(tx.direction == DIRECTION_DOWNLINK)
        {
            if(section_info.num_prbc == ORAN_MAX_PRB_X_SLOT)
            {
                section.sect_1.startPrbc = section_info.start_prbc;
                section.sect_1.numPrbc = 0;
                section.sect_1.numSymbol = 1;
                ++cplane_sections_num;
                ++radio_app_hdr.numberOfSections;
            }
            else if (section_info.num_prbc > ORAN_MAX_PRB_X_SECTION)
            {
                section.sect_1.startPrbc = section_info.start_prbc;
                section.sect_1.numPrbc = ORAN_MAX_PRB_X_SECTION;
                section.sect_1.numSymbol = 1;
                ++cplane_sections_num;
                ++radio_app_hdr.numberOfSections;
                auto& new_section = cplane_sections[cplane_sections_num];
                new_section.sect_1.startPrbc = ORAN_MAX_PRB_X_SECTION;
                new_section.sect_1.numPrbc = section_info.num_prbc - ORAN_MAX_PRB_X_SECTION;
                new_section.sect_1.numSymbol = 1;
                ++cplane_sections_num;
                ++radio_app_hdr.numberOfSections;
            }
            else
            {
                section.sect_1.startPrbc = section_info.start_prbc;
                section.sect_1.numPrbc = section_info.num_prbc;
                section.sect_1.numSymbol = 1;
                ++cplane_sections_num;
                ++radio_app_hdr.numberOfSections;
            }
        }
        else
        {
            if(tx.symbol_id == section_info.start_sym)
            {
                section.sect_1.startPrbc = section_info.start_prbc;
                section.sect_1.numPrbc = section_info.num_prbc;
                section.sect_1.numSymbol = section_info.num_sym;
                ++cplane_sections_num;
                ++radio_app_hdr.numberOfSections;
                if(section_info.num_prbc == ORAN_MAX_PRB_X_SLOT)
                {
                    section.sect_1.startPrbc = section_info.start_prbc;
                    section.sect_1.numPrbc = 0;
                    section.sect_1.numSymbol = section_info.num_sym;
                    ++cplane_sections_num;
                    ++radio_app_hdr.numberOfSections;
                }
                else if (section_info.num_prbc > ORAN_MAX_PRB_X_SECTION)
                {
                    section.sect_1.startPrbc = section_info.start_prbc;
                    section.sect_1.numPrbc = ORAN_MAX_PRB_X_SECTION;
                    section.sect_1.numSymbol = section_info.num_sym;
                    ++cplane_sections_num;
                    ++radio_app_hdr.numberOfSections;
                    auto& new_section = cplane_sections[cplane_sections_num];
                    new_section.sect_1.startPrbc = ORAN_MAX_PRB_X_SECTION;
                    new_section.sect_1.numPrbc = section_info.num_prbc - ORAN_MAX_PRB_X_SECTION;
                    new_section.sect_1.numSymbol = section_info.num_sym;
                    ++cplane_sections_num;
                    ++radio_app_hdr.numberOfSections;
                }
                else
                {
                    section.sect_1.startPrbc = section_info.start_prbc;
                    section.sect_1.numPrbc = section_info.num_prbc;
                    section.sect_1.numSymbol = section_info.num_sym;
                    ++cplane_sections_num;
                    ++radio_app_hdr.numberOfSections;
                }
            }
            else
            {
                continue;
            }
        }
    }

    if(radio_app_hdr.numberOfSections > 0)
    {
        ++cplane_msg_infos_num;
    }
}


void fronthaul_generator_dl_tx_c_worker(Worker* worker)
{
    usleep(500);
    uint32_t cpu;
    int ret;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        THROW(StringBuilder() << "getcpu failed for " << __FUNCTION__);
    }
    char threadname[30];
    sprintf(threadname, "%s", "DLTX");
    SET_THREAD_NAME(threadname);
    NVLOGC_FMT(TAG,"Start DL C TX worker on CPU {}", cpu);

    auto& context = worker->get_context();
    // Prepare C-plane message descriptor template
    std::array<aerial_fh::CPlaneMsgSendInfo, kMaxMsgSendInfoCount> cplane_msg_infos;
    std::array<aerial_fh::CPlaneSectionInfo, kMaxSectionCount> cplane_sections;
    int cplane_msg_infos_num = 0;
    int cplane_sections_num = 0;
    fill_cplane_msg_template(context, cplane_msg_infos, cplane_sections);
    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID;
    frame_cycle_time_ns *= ORAN_MAX_SLOT_ID;
    frame_cycle_time_ns *= context.slot_duration;

    aerial_fh::UPlaneTxCompleteNotification tx_complete_notification{
        .callback     = tx_complete_callback,
        .callback_arg = nullptr,
    };

    uint32_t slot_index_tdd = 0;
    uint64_t test_slot_count = 0;

    // Make sure that time_anchor is in the future
    aerial_fh::Ns next_window = context.time_anchor;// + frame_cycle_time_ns;
    wait_ns(next_window - context.dl_c_enq_time_advance_ns);

    do
    {
        auto oran_slot_number = context.oran_slot_iterator.get_next();
        for(int i = 0; i < context.dl_tx_worker_context.peers.size(); ++i)
        {
            auto& peer = context.dl_tx_worker_context.peers[i];
            cplane_msg_infos_num = 0;
            cplane_sections_num = 0;
            for(auto const& tx : context.dl_tx_worker_context.cplane[i][slot_index_tdd])
            {
                prepare_c_plane(oran_slot_number, cplane_msg_infos, cplane_sections, tx, cplane_msg_infos_num, cplane_sections_num, next_window);
            }

            if(cplane_msg_infos_num > 0)
            {
                if(unlikely(0 == aerial_fh::send_cplane(peer, &cplane_msg_infos[0], cplane_msg_infos_num)))
                {
                    NVLOGE_FMT(TAG,AERIAL_ORAN_FH_EVENT,"Failed to send C-plane");
                    THROW(StringBuilder() << "Failed to send C-plane");
                }
            }
        }
        slot_index_tdd = (slot_index_tdd + 1) % context.slot_count;
        ++test_slot_count;
        next_window += context.slot_duration;
        wait_ns(next_window - context.dl_c_enq_time_advance_ns);
    } while(!worker->exit_signal() && (context.test_slots == 0 || test_slot_count < context.test_slots));
    NVLOGC_FMT(TAG, "DL TX C Worker on CPU {} exit, test_slot_count {} reached test_slots {}", cpu, test_slot_count, context.test_slots);
    sleep(3);
    context.fhgen->set_workers_exit_signal();
}

}