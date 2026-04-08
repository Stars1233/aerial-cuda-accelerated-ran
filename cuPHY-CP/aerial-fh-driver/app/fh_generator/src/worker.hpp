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

#ifndef FH_GENERATOR_WORKER_HPP__
#define FH_GENERATOR_WORKER_HPP__

#include <atomic>
#include <thread>
#include <vector>

#include "utils.hpp"
#include "oran_slot_iterator.hpp"
#include "gpudevice.hpp"
#include "doca_utils.hpp"
#include "order_entity.hpp"

namespace fh_gen
{
class FhGenerator;

struct CPlaneTXSection
{
    uint8_t start_sym;
    uint8_t num_sym;
    uint16_t start_prbc;
    uint16_t num_prbc;
};

using CPlaneTXSectionList = std::vector<CPlaneTXSection>;
struct CPlaneTX
{
    aerial_fh::FlowHandle flow;
    aerial_fh::FlowId     eAxC;
    uint16_t              vlan_id;
    int64_t               slot_offset;
    uint16_t              section_count;
    uint8_t               symbol_id;
    oran_pkt_dir          direction;
    CPlaneTXSectionList   section_list;
    bool                  operator<(const CPlaneTX& a)
    {
        return slot_offset < a.slot_offset;
    }
};

struct UPlaneTXSection
{
    uint16_t start_prbu;
    uint16_t num_prbu;
};
using UPlaneTXSectionList = std::vector<UPlaneTXSection>;

struct UPlaneTX
{
    aerial_fh::FlowHandle flow;
    aerial_fh::FlowId     eAxC;
    uint16_t              vlan_id;
    int64_t               slot_offset;
    void*                 iq_data_buffer;
    uint8_t               symbol_id;
    uint16_t              section_id;
    UPlaneTXSectionList   section_list;

    bool operator<(const UPlaneTX& a)
    {
        return slot_offset < a.slot_offset;
    }
};

using CPlaneTXs = std::array<std::vector<std::vector<CPlaneTX>>, kMaxCells>;
using UPlaneTXs = std::array<std::vector<std::vector<UPlaneTX>>, kMaxCells>;
using GDR             = std::vector<struct gpinned_buffer*>;

enum WorkerType
{
    DL_TX_U,
    DL_TX_C,
    DL_RX,
    UL_TX,
    UL_RX
};

using PeerHandles = std::vector<aerial_fh::PeerHandle>;

struct DLTXWorkerContext
{
    PeerHandles peers;
    UPlaneTXs                          uplane;
    CPlaneTXs                          cplane;
    std::vector<std::string>           peer_nic_names;
    std::vector<int>                   peer_nic_ids;
};

struct DLRXWorkerContext
{
    uint8_t                            num_peers;
    PeerHandles                        peers;
    std::vector<int>                   peer_ids;
    std::vector<aerial_fh::Ns>         tcp_adv_dl_ns;
    std::vector<aerial_fh::Ns>         t1a_max_cp_ul_ns;
    std::vector<aerial_fh::Ns>         t1a_max_up_ns;
    std::vector<aerial_fh::Ns>         window_end_ns;
};

struct ULTXWorkerContext
{
    uint8_t                            num_peers;
    PeerHandles                        peers;
    UPlaneTXs                          uplane;
    std::vector<std::string>           peer_nic_names;
    std::vector<int>                   peer_nic_ids;
    std::vector<int>                   peer_ids;
};

struct ULRXWorkerContext
{
    PeerHandles     peers;
    OrderEntity*    order_entities[kOrderEntityNum];
    GDR             exit_flag;
    struct doca_rx_items d_rxq_info[fh_gen::kMaxCells];
    uint16_t*       expected_rx_prbs_d;
    uint16_t        expected_rx_prbs_h[MAX_LAUNCH_PATTERN_SLOTS][fh_gen::kMaxCells];
    bool            has_expected_rx_prbs[MAX_LAUNCH_PATTERN_SLOTS];
    uint64_t*         ta4_min_ns_d;
    uint64_t*         ta4_max_ns_d;
    std::vector<aerial_fh::Ns>         ta4_min_ns;
    std::vector<aerial_fh::Ns>         ta4_max_ns;
};

struct WorkerContext
{
    FhGenerator*                       fhgen;
    uint16_t                           index;
    aerial_fh::PeerHandle              peer;
    aerial_fh::NicHandle               nic;
    std::string                        nic_name;
    aerial_fh::Ns                      time_anchor;
    aerial_fh::Ns                      slot_duration;
    uint8_t                            slot_count;
    uint64_t                           test_slots;
    aerial_fh::Ns                      enq_time_advance;
    aerial_fh::Ns                      tcp_adv_dl_ns;
    aerial_fh::Ns                      t1a_max_cp_ul_ns;
    aerial_fh::Ns                      t1a_max_up_ns;
    aerial_fh::Ns                      dl_u_enq_time_advance_ns;
    aerial_fh::Ns                      dl_c_enq_time_advance_ns;
    aerial_fh::Ns                      ul_u_tx_time_advance_ns;
    aerial_fh::Ns                      ul_u_enq_time_advance_ns;
    aerial_fh::Ns                      window_end_ns;
    aerial_fh::UserDataCompressionInfo ud_comp_info;
    OranSlotIterator                   oran_slot_iterator;
    GDR                                buffer_ready_gdr; //GpuComm required
    DLRXWorkerContext                  dl_rx_worker_context;
    ULRXWorkerContext                  ul_rx_worker_context;
    ULTXWorkerContext                  ul_tx_worker_context;
    DLTXWorkerContext                  dl_tx_worker_context;
};

struct WorkerInfo
{
    WorkerType                 worker_type_;
    FhGenType                  fh_gen_type_;
    uint32_t                   cpu_core;
    int32_t                    priority;
    aerial_fh::PeerId          peer_id;
    aerial_fh::Ns              tcp_adv_dl_ns;
    aerial_fh::Ns              t1a_max_cp_ul_ns;
    std::default_random_engine random_engine;
    uint32_t                   cplane_shuffle_count;
    uint32_t                   uplane_shuffle_count;
    int64_t                    max_tx_shift;
};

class Worker {
public:
    Worker(WorkerContext context, WorkerInfo info);
    ~Worker();
    WorkerContext& get_context();
    volatile bool  exit_signal() const;
    void           set_exit_signal();
protected:
    WorkerInfo                   info_;
    WorkerContext                context_;
    std::unique_ptr<std::thread> thread_{};
    std::atomic<bool>            exit_signal_{false};

    void set_priority();
    void set_affinity();
};

void prepare_c_plane(const struct fh_gen::OranSlotNumber& oran_slot_number, std::array<aerial_fh::CPlaneMsgSendInfo, kMaxMsgSendInfoCount>& cplane_msg_infos, std::array<aerial_fh::CPlaneSectionInfo, kMaxSectionCount>& cplane_sections, const fh_gen::CPlaneTX &tx, int& cplane_msg_infos_num, int& cplane_sections_num, uint64_t next_window);
void prepare_u_plane(const struct fh_gen::OranSlotNumber& oran_slot_number, aerial_fh::UPlaneMsgMultiSectionSendInfo& uplane_msg, const fh_gen::UPlaneTX &tx, uint64_t next_window, uint64_t slot_duration, void* fh_mem, size_t prb_size, size_t symbol_size);
void prepare_u_plane_single_section(const struct fh_gen::OranSlotNumber& oran_slot_number, aerial_fh::UPlaneMsgMultiSectionSendInfo& uplane_msg, const fh_gen::UPlaneTX &tx, uint64_t next_window, uint64_t slot_duration, void* fh_mem, size_t prb_size, size_t symbol_size, int section_num);

} // namespace fh_gen

#endif //ifndef FH_GENERATOR_WORKER_HPP__
