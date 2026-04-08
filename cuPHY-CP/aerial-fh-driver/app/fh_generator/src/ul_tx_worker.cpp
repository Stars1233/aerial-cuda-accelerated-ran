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

namespace fh_gen
{

static void fill_uplane_msg_template(const WorkerContext& info, aerial_fh::UPlaneMsgSendInfo& uplane_msg_info)
{
    memset(&uplane_msg_info, 0, sizeof(uplane_msg_info));

    auto& radio_app_hdr          = uplane_msg_info.radio_app_hdr;
    radio_app_hdr.dataDirection  = DIRECTION_UPLINK;
    radio_app_hdr.payloadVersion = ORAN_DEF_PAYLOAD_VERSION;
    radio_app_hdr.filterIndex    = ORAN_DEF_FILTER_INDEX;

    auto& section_hdr   = uplane_msg_info.section_info;
    section_hdr.rb      = ORAN_RB_ALL;
    section_hdr.sym_inc = ORAN_SYMCINC_NO;
}

static void tx_complete_callback(void* addr, void* opaque)
{
}

void prepare_u_plane_single_section(const struct fh_gen::OranSlotNumber& oran_slot_number, aerial_fh::UPlaneMsgMultiSectionSendInfo& uplane_msg, const fh_gen::UPlaneTX &tx, uint64_t next_window, uint64_t slot_duration, void* fh_mem, size_t prb_size, size_t symbol_size, int section_num)
{
    auto section = tx.section_list[section_num];
    aerial_fh::UPlaneSectionInfo& uplane_section = uplane_msg.section_infos[uplane_msg.section_num++];
    aerial_fh::MsgSendWindow& msg_send_window = uplane_msg.tx_window;
    msg_send_window.tx_window_start = next_window + tx.slot_offset + tx.symbol_id * slot_duration / ORAN_ALL_SYMBOLS;
    if(uplane_msg.section_num == 1)
    {
        struct oran_umsg_iq_hdr& iq_df = uplane_msg.radio_app_hdr;
        iq_df.frameId          = oran_slot_number.frame_id;
        iq_df.subframeId       = oran_slot_number.subframe_id;
        iq_df.slotId           = oran_slot_number.slot_id;
        iq_df.symbolId         = tx.symbol_id;
        uplane_msg.flow        = tx.flow;
        uplane_msg.eaxcid      = tx.eAxC;
    }
    uplane_section.iq_data_buffer = static_cast<uint8_t*>(fh_mem) + (tx.symbol_id * symbol_size) + (section.start_prbu * prb_size);;
    uplane_section.section_id = tx.section_id;
    uplane_section.rb = 0;
    uplane_section.sym_inc = 0;
    uplane_section.start_prbu = section.start_prbu;
    uplane_section.num_prbu = section.num_prbu;
}

void prepare_u_plane(const struct fh_gen::OranSlotNumber& oran_slot_number, aerial_fh::UPlaneMsgMultiSectionSendInfo& uplane_msg, const fh_gen::UPlaneTX &tx, uint64_t next_window, uint64_t slot_duration, void* fh_mem, size_t prb_size, size_t symbol_size)
{
    for(int i = 0; i < tx.section_list.size(); ++i)
    {
        prepare_u_plane_single_section(oran_slot_number, uplane_msg, tx, next_window, slot_duration, fh_mem, prb_size, symbol_size, i);
    }
}

void fronthaul_generator_ul_tx_worker(Worker* worker)
{
    uint32_t cpu;
    int ret;
    ret = getcpu(&cpu, nullptr);
    if(ret != 0)
    {
        THROW(StringBuilder() << "getcpu failed for " << __FUNCTION__);
    }

    auto& context = worker->get_context();
    auto& nic = context.nic;
    NVLOGC_FMT(TAG,"Start UL TX worker on CPU {}, handling {} cells", cpu, context.ul_tx_worker_context.num_peers);

    char threadname[30];
    sprintf(threadname, "%s", "ULTX");
    SET_THREAD_NAME(threadname);

    void * fh_mem = aerial_fh::allocate_memory(kMaxPrbsPerSymbol * kMaxSymbols * kMaxAntennas * kMaxPrbSize * sizeof(uint8_t), (unsigned long)sysconf(_SC_PAGESIZE));

    aerial_fh::UPlaneTxCompleteNotification tx_complete_notification{
        .callback     = tx_complete_callback,
        .callback_arg = nullptr,
    };

    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID; // 256
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID; // 10
    frame_cycle_time_ns *= ORAN_MAX_SLOT_ID; // 2
    frame_cycle_time_ns *= 500; // slot duration
    frame_cycle_time_ns *= 1000; // NS X US

    // fill_uplane_msg_template(context, uplane_msg_info);

    if(fh_mem == nullptr)
    {
        THROW(StringBuilder() << "aerial_fh::allocate_memory failure ");
    }
    
    auto     prb_size       = get_prb_size(context.ud_comp_info.iq_sample_size, context.ud_comp_info.method);
    auto     symbol_size    = prb_size * ORAN_MAX_PRB_X_SLOT;
    uint32_t slot_index_tdd = 0;
    uint64_t test_slot_count = 0;

    aerial_fh::Ns next_window = context.time_anchor;
    aerial_fh::TxRequestHandle tx_request;

    // All cells use the same tx_request pool, use peer 0 for placeholder
    aerial_fh::alloc_tx_request(context.ul_tx_worker_context.peers[0], &tx_request);
    // All cells use the same txqs, use peer 0 for placeholder
    aerial_fh::TxqHandle txqs[ORAN_ALL_SYMBOLS * 2];
    size_t num_txqs = ORAN_ALL_SYMBOLS * 2;
    int num_tx_msgs = 0;
    aerial_fh::get_uplane_txqs(context.ul_tx_worker_context.peers[0], txqs, &num_txqs);
    if(num_txqs != ORAN_ALL_SYMBOLS * 2){ // 2
        THROW(StringBuilder() << "Failed to get " << ORAN_ALL_SYMBOLS * 2 << " uplane txqs only got " << num_txqs << "\n");
    }
    wait_ns(next_window - context.ul_u_enq_time_advance_ns);

    do
    {
        auto oran_slot_number = context.oran_slot_iterator.get_next();
        for(int i = 0; i < context.ul_tx_worker_context.num_peers; ++i)
        {
            aerial_fh::UPlaneMsgMultiSectionSendInfo uplane_msg = {};
            if(context.ul_tx_worker_context.uplane[i][slot_index_tdd].size() == 0)
            {
                // Preallocate mbuf for TX when nb_rx == 0, we assume it is a down time for the thread, so it can spend some cycles to pre-allocate the mbufs
                // needed for UL TX, 512 is a heuristic for the UL peak patterns for now.
                aerial_fh::preallocate_mbufs(context.ul_tx_worker_context.peers[i], &tx_request, 512);
            }
            for(auto const& tx : context.ul_tx_worker_context.uplane[i][slot_index_tdd])
            {
                prepare_u_plane(oran_slot_number, uplane_msg, tx, next_window, context.slot_duration, fh_mem, prb_size, symbol_size);
                if(uplane_msg.section_num > 0)
                {
                    auto txq_index = uplane_msg.radio_app_hdr.symbolId + uplane_msg.radio_app_hdr.slotId * ORAN_ALL_SYMBOLS;
                    if (txq_index >= num_txqs) {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "txq_index {} out of bounds, max {}", txq_index, num_txqs - 1);
                        continue;
                    }
                    aerial_fh::prepare_uplane_with_preallocated_tx_request(context.ul_tx_worker_context.peers[i], &uplane_msg, tx_complete_notification, &tx_request, txq_index);
                    auto tx_cnt = aerial_fh::send_uplane_without_freeing_tx_request(tx_request, txqs[txq_index]);
                    uplane_msg.section_num = 0;
                    NVLOGI_FMT(TAG_TX_TIMINGS,"[UL] {} F{}S{}S{} Cell {} TX Time {} Enqueue Time {} Sym {} Num Packets {} Queue {}",
                                "NONE",
                                oran_slot_number.frame_id,
                                oran_slot_number.subframe_id,
                                oran_slot_number.slot_id,
                                context.ul_tx_worker_context.peer_ids[i],
                                uplane_msg.tx_window.tx_window_start,
                                now_ns(),
                                uplane_msg.radio_app_hdr.symbolId.get(),
                                tx_cnt,
                                txq_index);
                }
            }
        }
        slot_index_tdd = (slot_index_tdd + 1) % context.slot_count;
        wait_ns(next_window - context.ul_u_enq_time_advance_ns);
        next_window += context.slot_duration;
        ++test_slot_count;

    } while(!worker->exit_signal() && (context.test_slots == 0 || test_slot_count < context.test_slots));

    // All cells use the same tx_request pool, use peer 0 for placeholder
    aerial_fh::free_preallocated_mbufs(context.ul_tx_worker_context.peers[0], &tx_request);

    NVLOGC_FMT(TAG, "UL TX Worker on CPU {} exit, test_slot_count {} reached test_slots {}", cpu, test_slot_count, context.test_slots);
    sleep(1);
    context.fhgen->set_workers_exit_signal();
    sleep(3);
    context.fhgen->set_exit_signal();
}

}