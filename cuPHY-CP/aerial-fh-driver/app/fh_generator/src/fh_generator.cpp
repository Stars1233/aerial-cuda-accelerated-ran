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

#include "fh_generator.hpp"
#include "oran_slot_iterator.hpp"
#include "worker.hpp"
#include "cuphy_pti.hpp"

#undef TAG
#define TAG "FHGEN.GEN"

namespace fh_gen
{

std::atomic<bool>           FhGenerator::synced_with_peer;
std::atomic<bool>           FhGenerator::ru_send_ack;
std::atomic<aerial_fh::Ns>  FhGenerator::time_anchor_;

enum SyncState
{
    S0 = 0,
    S1 = 1,
    S2 = 2
};

FhGenerator::FhGenerator(const std::string& config_file, FhGenType type) :
    fh_gen_type_(type),
    yaml_parser_(this, config_file)
{
    initialize_random_number_generator();
    if(fh_gen_type_ == FhGenType::DU)
    {
        cudaFree(0);
        CUDA_CHECK(cudaSetDevice(0));
    }
    setup_fh_driver();

    add_nics();
    add_peers();
    add_flows();
    if(fh_gen_type_ == FhGenType::DU)
    {
        add_iq_data_buffers();
        add_gpu_comm_ready_flags();
        add_doca_rx_kernel_exit_flag();
    }
    add_cells();

    if(fh_gen_type_ == FhGenType::DU)
    {
        initialize_order_entities();
    }

    setup_oam();
    calculate_expected_packet_counts();
    synchronize_peer();
    create_workers();
}

FhGenerator::~FhGenerator()
{
    NVLOGC_FMT(TAG,"Exitting, printing summary for {}: ", (fh_gen_type_ ==  FhGenType::RU) ? "FHGen RU" : "FHGen DU");
    if(fh_gen_type_ == FhGenType::RU)
    {
        print_ru_summary_stats();
        if(ulc_stats.active())
        {
            NVLOGC_FMT(TAG,"UL C summary: ");
            ulc_stats.flush_counters(yaml_parser_.get_peer_info().size());
        }
        if(dlc_stats.active())
        {
            NVLOGC_FMT(TAG,"DL C summary: ");
            dlc_stats.flush_counters(yaml_parser_.get_peer_info().size());
        }
        if(dlu_stats.active())
        {
            NVLOGC_FMT(TAG,"DL U summary: ");
            dlu_stats.flush_counters(yaml_parser_.get_peer_info().size());
        }
    }
    else
    {
        if(ulu_stats.active())
        {
            NVLOGC_FMT(TAG,"UL U summary: ");
            ulu_stats.flush_counters(yaml_parser_.get_peer_info().size());
        }
        // Keep file because CICD uses it even if it is empty
        // ulu_stats.flush_counters_file(yaml_parser_.get_peer_info().size(), std::string("/tmp/fhgen_du_ul_packet_times.txt"));
    }

    bool pass = true;
    if(!check_packet_count_pass_criteria())
    {
        pass = false;
    }

    float threshold;
    if(fh_gen_type_ == FhGenType::DU)
    {
        for(int peer_index = 0; peer_index < yaml_parser_.get_peer_info().size(); ++peer_index)
        {
            threshold = yaml_parser_.ulu_ontime_pass_percentage();
            if(!ulu_stats.pass_slot_percentage(peer_index, threshold))
            {
                NVLOGC_FMT(TAG, "Cell {} ULU did not pass {}", peer_index, threshold);
                pass = false;
            }
            else
            {
                NVLOGC_FMT(TAG, "Cell {} ULU passed {}", peer_index, threshold);
            }
        }
    }
    else
    {
        for(int peer_index = 0; peer_index < yaml_parser_.get_peer_info().size(); ++peer_index)
        {
            threshold = yaml_parser_.ulc_ontime_pass_percentage();
            if(!ulc_stats.pass_slot_percentage(peer_index, threshold))
            {
                NVLOGC_FMT(TAG, "Cell {} ULC did not pass {}", peer_index, threshold);
                pass = false;
            }
            else
            {
                NVLOGC_FMT(TAG, "Cell {} ULC passed {}", peer_index, threshold);
            }

            threshold = yaml_parser_.dlc_ontime_pass_percentage();
            if(!dlc_stats.pass_slot_percentage(peer_index, threshold))
            {
                NVLOGC_FMT(TAG, "Cell {} DLC did not pass {}", peer_index, threshold);
                pass = false;
            }
            else
            {
                NVLOGC_FMT(TAG, "Cell {} DLC passed {}", peer_index, threshold);
            }

            threshold = yaml_parser_.dlu_ontime_pass_percentage();
            if(!dlu_stats.pass_slot_percentage(peer_index, threshold))
            {
                NVLOGC_FMT(TAG, "Cell {} DLU did not pass {}", peer_index, threshold);
                pass = false;
            }
            else
            {
                NVLOGC_FMT(TAG, "Cell {} DLU passed {}", peer_index, threshold);
            }
        }
    }

    workers_.clear();

    for(auto nic: nics_)
    {
        aerial_fh::print_stats(nic.second, true);
    }
    free_resources();
    aerial_fh::close(fhi_);

    if(pass)
    {
        NVLOGC_FMT(TAG, "TEST PASS");
    }
    else
    {
        NVLOGC_FMT(TAG, "TEST FAIL");
    }
    NVLOGC_FMT(TAG,"Closing FH traffic generator");
}

void FhGenerator::initialize_order_entities()
{
    for(int i = 0; i < kOrderEntityNum; i++)
    {
        order_entities.emplace_back(std::make_unique<OrderEntity>(gpus_[0].get(), i));
        auto& order_entity = order_entities.back();
        for(int cell_idx = 0; cell_idx < cells.size(); ++cell_idx)
        {
            order_entity->order_kernel_config_params->next_slot_early_rx_packets[cell_idx] = (uint32_t*)cells[cell_idx]->next_slot_early_rx_packets->addr();
            order_entity->order_kernel_config_params->next_slot_on_time_rx_packets[cell_idx] = (uint32_t*)cells[cell_idx]->next_slot_on_time_rx_packets->addr();
            order_entity->order_kernel_config_params->next_slot_late_rx_packets[cell_idx] = (uint32_t*)cells[cell_idx]->next_slot_late_rx_packets->addr();
            order_entity->order_kernel_config_params->next_slot_num_prb[cell_idx] = (uint32_t*)cells[cell_idx]->next_slot_num_prb->addr();
            order_entity->order_kernel_config_params->ta4_min_ns[cell_idx] = cells[cell_idx]->peer_info.ta4_min_ns;
            order_entity->order_kernel_config_params->ta4_max_ns[cell_idx] = cells[cell_idx]->peer_info.ta4_max_ns;
            order_entity->order_kernel_config_params->next_slot_rx_packets_count[cell_idx] = (uint32_t *)cells[cell_idx]->next_slot_rx_packets_count->addr();
            order_entity->order_kernel_config_params->next_slot_rx_packets_ts[cell_idx] = (uint64_t *)cells[cell_idx]->next_slot_rx_packets_ts->addr();
        }
    }
}

void FhGenerator::synchronize_peer()
{
    int64_t beginning_of_time = 0;
    int64_t frame_cycle_time_ns = ORAN_MAX_FRAME_ID; // 256
    frame_cycle_time_ns *= ORAN_MAX_SUBFRAME_ID; // 10
    frame_cycle_time_ns *= ORAN_MAX_SLOT_ID; // 2
    frame_cycle_time_ns *= 500; // slot duration
    frame_cycle_time_ns *= 1000; // NS X US
    int64_t sfn_frame_cycle_time_ns = frame_cycle_time_ns * 4; // 1024

    beginning_of_time = now_ns();
    beginning_of_time /= (10000000ULL * 1024ULL);
    beginning_of_time++;
    beginning_of_time *= (1024ULL * 10000000ULL);
    beginning_of_time += (364ULL * 10000000ULL); // adjust to SFN = 0, accounting for GPS vs TIA conversion

    time_anchor_ = beginning_of_time;

    if(yaml_parser_.get_sync_info().enable)
    {
        // Use DU as the primary anchor
        // Following a simplified version of TCP protocol SYN, SYN+ACK, ACK handshake.
        // DU sends SYN with proposed timestamp, RU sends reply ACK with corresponding timestamp, DU sends final ACK with timestamp
        if(fh_gen_type_ == FhGenType::DU)
        {
            int state = 0;
            bool synced = false;
            while(!exit_signal_.load() && !synced)
            {
                switch (state)
                {
                case SyncState::S0:
                {
                    while(time_anchor_.load() < now_ns() + 4 * frame_cycle_time_ns)
                    {
                        time_anchor_ += 4 * frame_cycle_time_ns;
                    }
                    if(send_sfn_slot_sync_grpc_command())
                    {
                        sleep(1);
                        continue;
                    }
                    else
                    {
                        state = SyncState::S1;
                    }
                    break;
                }
                case SyncState::S1:
                {
                    auto now = now_ns();
                    state = SyncState::S0;
                    while(now_ns() - now < (uint64_t)(2) * 1000 * 1000 * 1000)
                    {
                        if(synced_with_peer.load())
                        {
                            send_sfn_slot_sync_grpc_command();
                            state = SyncState::S2;
                            break;
                        }
                    }
                    break;
                }
                case SyncState::S2:
                {
                    synced = true;
                    NVLOGC_FMT(TAG, "DU Synchronized! Time Anchor {}", time_anchor_.load());
                    break;
                }
                default:
                    break;
                }
                usleep(500000);
            }
        }
        else
        {
            NVLOGC_FMT(TAG, "gRPC Sync enabled, waiting for DU to send sync request with time anchor");
            int state = 0;
            bool synced = false;
            time_anchor_.store(0);
            while(!exit_signal_.load() && !synced)
            {
                switch (state)
                {
                case SyncState::S0:
                {
                    if(!ru_send_ack.load())
                    {
                        continue;
                    }
                    else
                    {
                        NVLOGC_FMT(TAG, "Received Time Anchor {}, Send ACK! ", time_anchor_.load());
                        state = SyncState::S1;
                    }
                    break;
                }
                case SyncState::S1:
                {
                    if(send_sfn_slot_sync_grpc_command())
                    {
                        NVLOGC_FMT(TAG, "Send ACK failed! Time Anchor {}", time_anchor_.load());
                        state = SyncState::S0;
                    }
                    else
                    {
                        state = SyncState::S2;
                        NVLOGC_FMT(TAG, "Send ACK succeeded for Time Anchor {}", time_anchor_.load());
                    }
                    NVLOGC_FMT(TAG, "Set ru_send_ack to false");
                    ru_send_ack.store(false);
                    break;
                }
                case SyncState::S2:
                {
                    auto now = now_ns();
                    while(now_ns() - now < (uint64_t)(2) * 1000 * 1000 * 1000)
                    {
                        if(synced_with_peer.load())
                        {
                            synced = true;
                            NVLOGC_FMT(TAG, "RU Synchronized! Time Anchor {}", time_anchor_.load());
                            break;
                        }
                    }
                    state = SyncState::S0;
                    break;
                }
                default:
                    break;
                }
                usleep(500000);
            }
        }
    }
}

void FhGenerator::setup_oam()
{
    auto enable = yaml_parser_.get_sync_info().enable;
    auto du_server_addr = yaml_parser_.get_sync_info().du_server_addr;
    auto ru_server_addr = yaml_parser_.get_sync_info().ru_server_addr;
    NVLOGC_FMT(TAG, "Setting up OAM, SFN SLOT sync {} {} {} {} {}", enable ? "enabled" : "disabled",
        enable ? "DU IP:" : "",
        enable ? du_server_addr : "",
        enable ? "RU IP:" : "",
        enable ? ru_server_addr : ""
    );

    synced_with_peer.store(false);
    ru_send_ack.store(false);
    if(enable){
        oam_init();
        peer_oam_addr = (fh_gen_type_ == FhGenType::DU) ? ru_server_addr : du_server_addr;

        if(fh_gen_type_ == FhGenType::DU)
        {
            pthread_t thread_id;
            auto status=pthread_create(&thread_id, NULL, du_sfn_slot_sync_cmd_thread_func, this);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create du_sfn_slot_sync_cmd_thread_func failed with status : {}", std::strerror(status));
            }
        }
        else
        {
            pthread_t thread_id;
            auto status=pthread_create(&thread_id, NULL, ru_sfn_slot_sync_cmd_thread_func, this);
            if(status)
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create ru_sfn_slot_sync_cmd_thread_func failed with status : {}", std::strerror(status));
            }
        }
    }
}

void FhGenerator::oam_init()
{
    CuphyOAM* oam = CuphyOAM::getInstance();
    if (oam->init_everything())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error oam init");
    }
    else
    {
        NVLOGC_FMT(TAG, "OAM initialized!");
    }
}

void FhGenerator::set_workers_exit_signal()
{
    for(auto& worker: workers_)
    {
        worker->set_exit_signal();
    }
}

volatile bool FhGenerator::exit_signal() const
{
    return exit_signal_.load();
}

void FhGenerator::set_exit_signal()
{
    exit_signal_ = true;
}

void FhGenerator::setup_fh_driver()
{
    auto fh_info              = const_cast<aerial_fh::FronthaulInfo*>(&yaml_parser_.get_fh_info());
    auto iq_data_buffer_infos = yaml_parser_.get_iq_data_buffer_info();
    std::set<aerial_fh::GpuId> unique_cuda_device_ids;

    for(auto& iq_data_buffer_info : iq_data_buffer_infos)
    {
        auto cuda_device_id = iq_data_buffer_info.cuda_device_id;
        if(cuda_device_id >= 0)
        {
            unique_cuda_device_ids.insert(cuda_device_id);
        }
    }
    if(fh_gen_type_ == FhGenType::DU)
    {
        fh_info->cuda_device_ids.assign(unique_cuda_device_ids.begin(), unique_cuda_device_ids.end());
    }
    else
    {
        fh_info->cuda_device_ids.clear();
        fh_info->cuda_device_ids_for_compute.clear();
    }
    if(aerial_fh::open(fh_info, &fhi_))
    {
        THROW("Failed to open FH driver");
    }
}

void FhGenerator::add_nics()
{
    auto nic_infos = yaml_parser_.get_nic_info();

    for(auto& nic_info : nic_infos)
    {
        if(fh_gen_type_ == FhGenType::RU)
        {
            nic_info.cuda_device = -1;
            nic_info.txq_count_gpu = 0;
            nic_info.rx_ts_enable = true;
            nic_info.split_cpu_mp = true;
        }

        aerial_fh::NicHandle nic{};
        auto                 name = std::string(nic_info.name);

        if ((fh_gen_type_ == FhGenType::DU) && (cuphy_pti_initialized_ == false))
        {
            cuphy_pti_init(name.c_str());
            cuphy_pti_initialized_ = true;
        }

        if(aerial_fh::add_nic(fhi_, &nic_info, &nic))
        {
            THROW(StringBuilder() << "Failed to add NIC " << name);
        }
        NVLOGC_FMT(TAG, "Added NIC {}", name.c_str());
        nics_[name] = nic;
        resources_to_free_.nics.emplace_back(nic, aerial_fh::remove_nic);
    }
}

void FhGenerator::add_iq_data_buffers()
{
    auto iq_data_buffer_infos = yaml_parser_.get_iq_data_buffer_info();
    auto send_utc_anchor      = yaml_parser_.send_utc_anchor();

    UtcAnchor utc_anchor{.anchor = __builtin_bswap64(time_anchor_.load())};

    for(auto& iq_data_buffer_info : iq_data_buffer_infos)
    {
        aerial_fh::MemRegHandle memreg{};

        auto  id             = iq_data_buffer_info.id;
        auto  cuda_device_id = iq_data_buffer_info.cuda_device_id;
        auto& addr           = iq_data_buffer_info.info.addr;
        auto& len            = iq_data_buffer_info.info.len;
        auto& page_sz        = iq_data_buffer_info.info.page_sz;
        auto  input_file     = iq_data_buffer_info.input_file;

        if(cuda_device_id < 0)
        {
            THROW(StringBuilder() << "We do not support FH generator with CPU U-plane");
        }
        else
        {
            NVLOGI_FMT(TAG, "Allocating GPU memory from CUDA device {} for IQ data buffer #{} of size {}", cuda_device_id , id , len);
            CHECK_CUDA_THROW(cudaSetDevice(cuda_device_id));
            CHECK_CUDA_THROW(cudaMalloc(&addr, len));

            NVLOGI_FMT(TAG, "Filling IQ data buffer #{} with byte value 0x{:x}", id , kGpuIqBufferMagicChar);
            CHECK_CUDA_THROW(cudaMemset(addr, kGpuIqBufferMagicChar, len));

            if(send_utc_anchor)
            {
                CHECK_CUDA_THROW(cudaMemcpy(addr, &utc_anchor, sizeof(UtcAnchor), cudaMemcpyHostToDevice));
            }

            resources_to_free_.gpu_buffers.emplace_back(addr, cuda_deallocator);
        }

        iq_data_buffers_[id] = std::make_pair(addr, len);
        resources_to_free_.memregs.emplace_back(memreg, aerial_fh::unregister_memory);
    }
}

void FhGenerator::add_peers()
{
    auto peer_infos = yaml_parser_.get_peer_info();

    for(auto& peer_info : peer_infos)
    {
        aerial_fh::PeerHandle peer{};
        std::vector<uint16_t> eAxC_list_ul,eAxC_list_srs, eAxC_list_dl;

        aerial_fh::MacAddr temp;
        if(fh_gen_type_ == FhGenType::RU)
        {
            std::memcpy(&temp, &peer_info.info.src_mac_addr, sizeof(uint8_t) * 6);
            std::memcpy(&peer_info.info.src_mac_addr, &peer_info.info.dst_mac_addr, sizeof(uint8_t) * 6);
            std::memcpy(&peer_info.info.dst_mac_addr, &temp, sizeof(uint8_t) * 6);
        }

        auto  id       = peer_info.info.id;
        auto& nic_name = peer_info.nic_name;
        if(fh_gen_type_ == FhGenType::RU)
        {
            peer_info.info.txq_count_uplane_gpu = 0;
            peer_info.info.txq_count_uplane = ORAN_ALL_SYMBOLS * 2;
            peer_info.info.txq_cplane = false;
            peer_info.info.share_txqs = true;
        }
        else
        {
            if(!yaml_parser_.enable_c_plane())
            {
                peer_info.info.txq_cplane = false;
            }
        }

        if(nics_.find(nic_name) == nics_.end())
        {
            THROW(StringBuilder() << "Failed to find NIC '" << nic_name << "' specified by Cell " << id);
        }

        // Get UL eAxCs
        for(auto flow_info : yaml_parser_.get_flow_info())
        {
            auto eAxC    = flow_info.info.eAxC;
            auto vlan_id = flow_info.info.vlan_tag.vid;
            auto cell_id = flow_info.cell_id;
            if(cell_id != id)
            {
                continue;
            }
            if(std::find(eAxC_list_ul.begin(), eAxC_list_ul.end(), eAxC) == eAxC_list_ul.end())
            {
                eAxC_list_ul.push_back(eAxC);
            }
        }

        if(aerial_fh::add_peer(nics_[nic_name], &peer_info.info, &peer, eAxC_list_ul, eAxC_list_srs, eAxC_list_dl))
        {
            THROW(StringBuilder() << "Failed to add Cell " << id << " to NIC " << nic_name);
        }

        peers_[id] = peer;
        resources_to_free_.peers.emplace_back(peer, aerial_fh::remove_peer);
    }
}

void FhGenerator::add_cells()
{
    for(auto& peer_info: yaml_parser_.get_peer_info())
    {
        //Assume support for 1 GPU only
        if(fh_gen_type_ == FhGenType::DU)
        {
            cells.emplace_back(std::make_unique<Cell>(this, gpus_[0].get(), peer_info));
            cells.back()->allocate_buffers();
        }
        else
        {
            cells.emplace_back(std::make_unique<Cell>(this, nullptr, peer_info));
        }
    }
}

void FhGenerator::add_flows()
{
    for(auto flow_info : yaml_parser_.get_flow_info())
    {
        aerial_fh::FlowHandle flow{};

        auto eAxC    = flow_info.info.eAxC;
        auto vlan_id = flow_info.info.vlan_tag.vid;
        auto cell_id = flow_info.cell_id;

        if(cplane_flows_[cell_id].find(eAxC) == cplane_flows_[cell_id].end())
        {
            if(peers_.find(cell_id) == peers_.end())
            {
                THROW(StringBuilder() << "Failed to find Cell '" << cell_id << "' specified by flow with eAxC=" << eAxC << ", vlan_id=" << vlan_id);
            }

            flow_info.info.type = aerial_fh::FlowType::CPLANE;
            flow_info.info.request_new_rxq = false;
            if(aerial_fh::add_flow(peers_[cell_id], &flow_info.info, &flow))
            {
                THROW(StringBuilder() << "Failed to add C-plane flow with eAxC=" << eAxC << ", vlan_id=" << vlan_id << " to Cell " << cell_id);
            }

            cplane_flows_[cell_id][eAxC] = flow;
            resources_to_free_.flows.emplace_back(flow, aerial_fh::remove_flow);
        }

        if(uplane_flows_[cell_id].find(eAxC) == uplane_flows_[cell_id].end())
        {
            flow_info.info.type = aerial_fh::FlowType::UPLANE;
            if(aerial_fh::add_flow(peers_[cell_id], &flow_info.info, &flow))
            {
                THROW(StringBuilder() << "Failed to add C-plane flow with eAxC=" << eAxC << ", vlan_id=" << vlan_id << " to Cell " << cell_id);
            }

            uplane_flows_[cell_id][eAxC] = flow;
            resources_to_free_.flows.emplace_back(flow, aerial_fh::remove_flow);
        }
    }
}

void FhGenerator::add_gpu_comm_ready_flags()
{
    auto fh_info              = const_cast<aerial_fh::FronthaulInfo*>(&yaml_parser_.get_fh_info());
    for(auto cuda_device: fh_info->cuda_device_ids)
    {
        NVLOGC_FMT(TAG, "Creating GPU instance with GDRCopy for Device ID {}", cuda_device);
        gpus_.push_back(std::make_unique<GpuDevice>(cuda_device, true));
        buffer_ready_gdr.push_back(gpus_.back()->newGDRbuf(1 * sizeof(uint32_t)));
        ((uint32_t*)buffer_ready_gdr.back()->addrh())[0] = 1;
    }
}

void FhGenerator::add_doca_rx_kernel_exit_flag()
{
    auto fh_info              = const_cast<aerial_fh::FronthaulInfo*>(&yaml_parser_.get_fh_info());
    for(auto cuda_device: fh_info->cuda_device_ids)
    {
        NVLOGC_FMT(TAG, "Creating exit_flag with GDRCopy for Device ID {}", cuda_device);
        exit_flag.push_back(gpus_.back()->newGDRbuf(1 * sizeof(uint32_t)));
        ((uint32_t*)exit_flag.back()->addrh())[0] = 1;
    }
}

static void validate_iq_data_buffer_size(void* addr, size_t buffer_size, aerial_fh::UserDataCompressionInfo ud_comp_info)
{
    auto    prb_size             = get_prb_size(ud_comp_info.iq_sample_size, ud_comp_info.method);
    auto    expected_buffer_size = prb_size * ORAN_MAX_PRB_X_SLOT * ORAN_ALL_SYMBOLS;
    int32_t buffer_size_diff     = buffer_size - expected_buffer_size;

    if(buffer_size < expected_buffer_size)
    {
        THROW(StringBuilder() << "The buffer @" << addr << " is " << (expected_buffer_size - buffer_size)
                              << " byte(s) too small to contain an entire slot of IQ data (273 PRBs for each of 14 symbols), given "
                              << ud_comp_info.iq_sample_size << " bit IQ sample size and ud_comp_meth=" << static_cast<uint16_t>(ud_comp_info.method));
    }
    else if(buffer_size > expected_buffer_size)
    {
        NVLOGW_FMT(TAG, "The IQ data buffer @{} is larger than required to contain an entire slot of IQ data (273 PRBs for each of 14 symbols) by {} byte(s), given {} bit IQ sample size and ud_comp_meth={}",
            addr , (buffer_size - expected_buffer_size) , ud_comp_info.iq_sample_size, static_cast<uint16_t>(ud_comp_info.method));
    }
}

void FhGenerator::create_workers()
{
    if(is_ru())
    {
        create_ru_workers();
    }

    if(is_du())
    {
        create_du_workers();
    }

    NVLOGI_FMT(TAG, "UTC time anchor: {}", time_anchor_.load());
    NVLOGI_FMT(TAG,"FH traffic generator is running...");
}

void FhGenerator::create_du_workers()
{
    auto peer_infos      = yaml_parser_.get_peer_info();
    auto& ul_cpus_list    = yaml_parser_.get_ul_cpus();
    auto& dl_cpus_list    = yaml_parser_.get_dl_cpus();

    auto worker_priority = yaml_parser_.get_workers_priority();
    auto shuffle_info    = yaml_parser_.get_shuffle_info();
    auto start_slot_info = yaml_parser_.get_start_slot_info();
    uint8_t slot_count   = yaml_parser_.slot_count();
    auto slot_duration   = yaml_parser_.slot_duration();
    auto test_slots      = yaml_parser_.test_slots();
    auto dl_u_enq_time_advance_ns      = yaml_parser_.dl_u_enq_time_advance_ns();
    auto dl_c_enq_time_advance_ns      = yaml_parser_.dl_c_enq_time_advance_ns();
    OranSlotIterator oran_slot_iterator{{start_slot_info.frame_id, start_slot_info.subframe_id, start_slot_info.slot_id}};

    // Launch UL RX workers
    if(yaml_parser_.enable_ulu())
    {
        auto ul_num_cpus = ul_cpus_list.size();
        auto cpu_to_use = ul_cpus_list.begin();
        WorkerContext worker_context{
            .fhgen              = this,
            .index              = 0,
            .peer               = nullptr,
            .nic                = nullptr,
            .time_anchor        = time_anchor_.load(),
            .slot_duration      = slot_duration,
            .slot_count         = slot_count,
            .test_slots         = test_slots,
            .oran_slot_iterator = oran_slot_iterator,
            .buffer_ready_gdr   = buffer_ready_gdr
        };

        WorkerInfo worker_info{
            .fh_gen_type_         = fh_gen_type_,
            .cpu_core             = *(cpu_to_use++),
            .priority             = worker_priority,
            .peer_id              = 0,
        };

        for(auto& peer_info : peer_infos)
        {
            auto peer_id          = peer_info.info.id;
            auto peer             = peers_[peer_id];
            auto ta4_min_ns       = peer_info.ta4_min_ns;
            auto ta4_max_ns       = peer_info.ta4_max_ns;

            worker_context.ul_rx_worker_context.peers.push_back(peer);
            worker_context.ul_rx_worker_context.ta4_min_ns.push_back(ta4_min_ns);
            worker_context.ul_rx_worker_context.ta4_max_ns.push_back(ta4_max_ns);
        }

        for(int i = 0; i < order_entities.size(); ++i)
        {
            worker_context.ul_rx_worker_context.order_entities[i] = order_entities[i].get();
            int index = 0;
            for(auto& peer_info : peer_infos)
            {
                auto peer_id          = peer_info.info.id;
                auto peer             = peers_[peer_id];
                struct doca_rx_items & d_rxq_info = worker_context.ul_rx_worker_context.d_rxq_info[index];
                aerial_fh::get_doca_rxq_items(peer,&d_rxq_info);
                worker_context.ul_rx_worker_context.order_entities[i]->order_kernel_config_params->rxq_info_gpu[index] = d_rxq_info.eth_rxq_gpu;
                worker_context.ul_rx_worker_context.order_entities[i]->order_kernel_config_params->sem_gpu[index]=d_rxq_info.sem_gpu;
                worker_context.ul_rx_worker_context.order_entities[i]->order_kernel_config_params->sem_order_num[index]=d_rxq_info.nitems;
                ++index;
            }
        }

        memset((void*)worker_context.ul_rx_worker_context.has_expected_rx_prbs, 0, sizeof(bool) * MAX_LAUNCH_PATTERN_SLOTS);
        for(auto& flow_info : yaml_parser_.get_flow_info())
        {
            for(auto& uplane_info : flow_info.uplane_rx_info)
            {
                for(auto& section: uplane_info.section_list)
                {
                    worker_context.ul_rx_worker_context.expected_rx_prbs_h[uplane_info.slot_id][flow_info.cell_id] += section.num_prbu;
                }

                if(worker_context.ul_rx_worker_context.expected_rx_prbs_h[uplane_info.slot_id][flow_info.cell_id] > 0)
                {
                    worker_context.ul_rx_worker_context.has_expected_rx_prbs[uplane_info.slot_id] = true;
                }
            }
        }

        for(int i = 0; i < MAX_LAUNCH_PATTERN_SLOTS; ++i)
        {
            for(int j = 0; j < fh_gen::kMaxCells; ++j)
            {
                if(worker_context.ul_rx_worker_context.expected_rx_prbs_h[i][j] > 0)
                {
                    NVLOGI_FMT(TAG, "Expected PRBs for slot {} cell {} : {}", i, j, worker_context.ul_rx_worker_context.expected_rx_prbs_h[i][j]);
                }
            }
        }

        worker_context.ul_rx_worker_context.exit_flag = exit_flag;
        worker_info.worker_type_ = WorkerType::UL_RX;

        workers_.push_back(std::make_unique<Worker>(worker_context, worker_info));
    }

    // Launch DL TX workers
    {
        auto dl_num_cpus = dl_cpus_list.size();
        // Split 1/2 CPUs for DL C 1/2 fo DL U
        auto dl_c_num_cpus = (dl_num_cpus + 1) / 2; // Round up for DL C (DL C is using CPU based TX)
        auto dl_u_num_cpus = dl_num_cpus / 2; // Round down for DL U

        // Assign slots to CPUs in round robin fashion

        auto cpu_to_use = dl_cpus_list.begin() + dl_c_num_cpus;

        // Launch DL TX C workers
        if(yaml_parser_.enable_c_plane()) {
            for(int cpu_index = 0; cpu_index < dl_c_num_cpus; ++cpu_index)
            {
                WorkerContext worker_context{
                    .fhgen              = this,
                    .index              = (uint16_t)cpu_index,
                    .peer               = nullptr,
                    .time_anchor        = time_anchor_.load(),
                    .slot_duration      = slot_duration,
                    .slot_count         = slot_count,
                    .test_slots         = test_slots,
                    // Assumption that all cells have same tx_time_advance_c and ud_comp_info
                    .dl_u_enq_time_advance_ns    = dl_u_enq_time_advance_ns,
                    .dl_c_enq_time_advance_ns    = dl_c_enq_time_advance_ns,
                    .ud_comp_info       = peer_infos[0].info.ud_comp_info,
                    .oran_slot_iterator = oran_slot_iterator,
                };

                WorkerInfo worker_info{
                    .fh_gen_type_         = fh_gen_type_,
                    .cpu_core             = dl_cpus_list[cpu_index],
                    .priority             = worker_priority,
                    .peer_id              = 0,
                };

                for(int i = 0; i < peer_infos.size(); ++i)
                {
                    auto& peer_info = peer_infos[i];
                    auto peer_id          = peer_info.info.id;
                    auto peer             = peers_[peer_id];

                    auto tcp_adv_dl_ns    = peer_info.tcp_adv_dl_ns;
                    auto t1a_max_cp_ul_ns = peer_info.t1a_max_cp_ul_ns;
                    auto t1a_max_up_ns    = peer_info.t1a_max_up_ns;
                    auto ud_comp_info     = peer_info.info.ud_comp_info;
                    worker_context.dl_tx_worker_context.cplane[i].resize(slot_count);
                    worker_context.dl_tx_worker_context.peers.push_back(peer);

                    for(auto& flow_info : yaml_parser_.get_flow_info())
                    {
                        auto cell_id = flow_info.cell_id;

                        if(peer_id == cell_id)
                        {
                            auto eAxC    = flow_info.info.eAxC;
                            auto vlan_id = flow_info.info.vlan_tag.vid;
                            for(auto& cplane_tx_info : flow_info.cplane_tx_info)
                            {
                                auto         slot_id          = cplane_tx_info.slot_id;
                                if((slot_id % dl_c_num_cpus) == cpu_index)
                                {
                                    auto         symbol_id        = cplane_tx_info.symbol_id;
                                    oran_pkt_dir direction        = std::string("UL") == cplane_tx_info.direction ? DIRECTION_UPLINK : DIRECTION_DOWNLINK;
                                    auto         cplane_tx_offset = direction == DIRECTION_UPLINK ? t1a_max_cp_ul_ns : tcp_adv_dl_ns + t1a_max_up_ns;
                                    int64_t      slot_offset      = (slot_duration / ORAN_ALL_SYMBOLS) * symbol_id - cplane_tx_offset;
                                    auto         section_count    = cplane_tx_info.section_count;

                                    if(!yaml_parser_.enable_dlc() && direction == DIRECTION_DOWNLINK)
                                    {
                                        continue;
                                    }

                                    if(!yaml_parser_.enable_ulc() && direction == DIRECTION_UPLINK)
                                    {
                                        continue;
                                    }

                                    CPlaneTX cplane{
                                        .flow          = cplane_flows_[cell_id][eAxC],
                                        .eAxC          = eAxC,
                                        .vlan_id       = vlan_id,
                                        .slot_offset   = slot_offset,
                                        .section_count = section_count,
                                        .symbol_id     = symbol_id,
                                        .direction     = direction,
                                    };
                                    for(auto& section_info: cplane_tx_info.section_list)
                                    {
                                        CPlaneTXSection section {
                                            .start_sym = section_info.start_sym,
                                            .num_sym = section_info.num_sym,
                                            .start_prbc = section_info.start_prbc,
                                            .num_prbc = section_info.num_prbc
                                        };
                                        cplane.section_list.emplace_back(section);
                                    }
                                    if(slot_id >= slot_count)
                                    {
                                        THROW(StringBuilder() << "Invalid slot_id (" << slot_id << ") for C-plane flow with eAxC=" << eAxC << ". Cell " << cell_id << " slot count is " << slot_count);
                                    }

                                    worker_context.dl_tx_worker_context.cplane[i][slot_id].push_back(cplane);
                                }
                            }
                        }
                    }
                }
                worker_info.worker_type_ = WorkerType::DL_TX_C;
                workers_.push_back(std::make_unique<Worker>(worker_context, worker_info));
            }
        }

        // Launch DL TX U workers
        if(yaml_parser_.enable_dlu())
        {
            for(int cpu_index = dl_c_num_cpus; cpu_index < dl_c_num_cpus + dl_u_num_cpus; ++cpu_index)
            {
                WorkerContext worker_context{
                    .fhgen              = this,
                    .index              = 0,
                    .peer               = nullptr,
                    .nic                = nics_[peer_infos[0].nic_name],
                    .nic_name           = peer_infos[0].nic_name,
                    .time_anchor        = time_anchor_.load(),
                    .slot_duration      = slot_duration,
                    .slot_count         = slot_count,
                    .test_slots         = test_slots,
                    // Assumption that all cells have same tx_time_advance_c and ud_comp_info
                    .dl_u_enq_time_advance_ns    = dl_u_enq_time_advance_ns,
                    .dl_c_enq_time_advance_ns    = dl_c_enq_time_advance_ns,
                    .ud_comp_info       = peer_infos[0].info.ud_comp_info,
                    .oran_slot_iterator = oran_slot_iterator,
                    .buffer_ready_gdr   = buffer_ready_gdr
                };

                WorkerInfo worker_info{
                    .fh_gen_type_         = fh_gen_type_,
                    .cpu_core             = dl_cpus_list[cpu_index],
                    .priority             = worker_priority,
                    .peer_id              = peer_infos[0].info.id,
                };

                for(int i = 0; i < peer_infos.size(); ++i)
                {
                    auto& peer_info = peer_infos[i];
                    auto peer_id          = peer_info.info.id;
                    auto peer             = peers_[peer_id];

                    auto tcp_adv_dl_ns    = peer_info.tcp_adv_dl_ns;
                    auto t1a_max_cp_ul_ns = peer_info.t1a_max_cp_ul_ns;
                    auto t1a_max_up_ns    = peer_info.t1a_max_up_ns;
                    auto ud_comp_info     = peer_info.info.ud_comp_info;

                    worker_context.dl_tx_worker_context.uplane[i].resize(slot_count);
                    worker_context.dl_tx_worker_context.peers.push_back(peer);
                    worker_context.dl_tx_worker_context.peer_nic_names.push_back(peer_info.nic_name);
                    worker_context.dl_tx_worker_context.peer_nic_ids.push_back(peer_info.nic_id);

                    for(auto& flow_info : yaml_parser_.get_flow_info())
                    {
                        auto cell_id = flow_info.cell_id;

                        if(peer_id == cell_id)
                        {
                            auto eAxC    = flow_info.info.eAxC;
                            auto vlan_id = flow_info.info.vlan_tag.vid;

                            for(auto& uplane_info : flow_info.uplane_tx_info)
                            {
                                auto    slot_id        = uplane_info.slot_id;
                                if((slot_id % dl_u_num_cpus) == cpu_index - dl_c_num_cpus)
                                {
                                    auto    symbol_id      = uplane_info.symbol_id;
                                    auto    iq_data_buffer = uplane_info.iq_data_buffer;
                                    int64_t slot_offset    = 0 - t1a_max_up_ns;

                                    UPlaneTX uplane{
                                        .flow           = uplane_flows_[cell_id][eAxC],
                                        .eAxC           = eAxC,
                                        .vlan_id        = vlan_id,
                                        .slot_offset    = slot_offset,
                                        .iq_data_buffer = iq_data_buffers_[iq_data_buffer].first,
                                        .symbol_id      = symbol_id,
                                        .section_id     = uplane_info.section_id,
                                    };

                                    for(auto& section_info: uplane_info.section_list)
                                    {
                                        UPlaneTXSection section {
                                            .start_prbu = section_info.start_prbu,
                                            .num_prbu = section_info.num_prbu
                                        };
                                        uplane.section_list.emplace_back(section);
                                    }
                                    if(slot_id >= slot_count)
                                    {
                                        THROW(StringBuilder() << "Invalid slot_id (" << slot_id << ") for U-plane flow with eAxC=" << eAxC << ", vlan_id=" << vlan_id << ". Cell " << cell_id << " slot count is " << slot_count);
                                    }

                                    worker_context.dl_tx_worker_context.uplane[i][slot_id].push_back(uplane);
                                }
                            }
                        }
                    }
                }

                worker_info.worker_type_ = WorkerType::DL_TX_U;
                workers_.push_back(std::make_unique<Worker>(worker_context, worker_info));
            }
        }
    }
}

void FhGenerator::create_ru_workers()
{
    auto peer_infos      = yaml_parser_.get_peer_info();
    auto& ul_cpus_list    = yaml_parser_.get_ul_cpus();
    auto& dl_cpus_list    = yaml_parser_.get_dl_cpus();

    auto worker_priority = yaml_parser_.get_workers_priority();
    auto shuffle_info    = yaml_parser_.get_shuffle_info();
    auto start_slot_info = yaml_parser_.get_start_slot_info();
    uint8_t slot_count   = yaml_parser_.slot_count();
    auto slot_duration   = yaml_parser_.slot_duration();
    auto test_slots      = yaml_parser_.test_slots();
    auto ul_u_enq_time_advance_ns   = yaml_parser_.ul_u_enq_time_advance_ns();
    auto ul_u_tx_time_advance_ns   = yaml_parser_.ul_u_tx_time_advance_ns();
    OranSlotIterator oran_slot_iterator{{start_slot_info.frame_id, start_slot_info.subframe_id, start_slot_info.slot_id}};

    // Launch UL TX workers
    if(yaml_parser_.enable_ulu())
    {
        auto ul_num_cpus = ul_cpus_list.size();
        auto cpu_to_use = ul_cpus_list.begin();

        std::vector<int> num_peers_per_cpu;
        for(int i = 0; i < ul_cpus_list.size(); ++i)
        {
            num_peers_per_cpu.push_back(0);
        }

        auto num_peers_added = 0;
        while(num_peers_added < peer_infos.size())
        {
            ++num_peers_per_cpu[num_peers_added%num_peers_per_cpu.size()];
            ++num_peers_added;
        }
        num_peers_added = 0;
        for(int i = 0; i < ul_cpus_list.size(); ++i)
        {
            WorkerContext worker_context{
                .fhgen              = this,
                .index              = (uint16_t)i,
                .peer               = nullptr,
                .nic                = nics_[peer_infos[0].nic_name],
                .nic_name           = peer_infos[0].nic_name,
                .time_anchor        = time_anchor_.load(),
                .slot_duration      = slot_duration,
                .slot_count         = slot_count,
                .test_slots         = test_slots,
                .ul_u_tx_time_advance_ns   = ul_u_tx_time_advance_ns,
                .ul_u_enq_time_advance_ns   = ul_u_enq_time_advance_ns,
                .ud_comp_info       = peer_infos[0].info.ud_comp_info,
                .oran_slot_iterator = oran_slot_iterator,
                .buffer_ready_gdr   = buffer_ready_gdr
            };

            WorkerInfo worker_info{
                .fh_gen_type_         = fh_gen_type_,
                .cpu_core             = *(cpu_to_use++),
                .priority             = worker_priority,
                .peer_id              = peer_infos[0].info.id,
            };
            worker_context.ul_tx_worker_context.num_peers = 0;

            for(int peer_index = num_peers_added; peer_index < num_peers_added + num_peers_per_cpu[i]; ++peer_index)
            {
                auto& peer_info       = peer_infos[peer_index];
                auto peer_id          = peer_info.info.id;
                auto peer             = peers_[peer_id];

                auto tcp_adv_dl_ns    = peer_info.tcp_adv_dl_ns;
                auto t1a_max_cp_ul_ns = peer_info.t1a_max_cp_ul_ns;
                auto t1a_max_up_ns    = peer_info.t1a_max_up_ns;
                auto ud_comp_info     = peer_info.info.ud_comp_info;

                worker_context.ul_tx_worker_context.uplane[worker_context.ul_tx_worker_context.num_peers].resize(slot_count);
                worker_context.ul_tx_worker_context.peers.push_back(peer);
                worker_context.ul_tx_worker_context.peer_nic_names.push_back(peer_info.nic_name);
                worker_context.ul_tx_worker_context.peer_nic_ids.push_back(peer_info.nic_id);
                worker_context.ul_tx_worker_context.peer_ids.push_back(peer_id);

                for(auto& flow_info : yaml_parser_.get_flow_info())
                {
                    auto cell_id = flow_info.cell_id;

                    if(peer_id == cell_id)
                    {
                        auto eAxC    = flow_info.info.eAxC;
                        auto vlan_id = flow_info.info.vlan_tag.vid;

                        for(auto& uplane_info : flow_info.uplane_rx_info)
                        {
                            auto    slot_id        = uplane_info.slot_id;
                            auto    symbol_id      = uplane_info.symbol_id;
                            auto    iq_data_buffer = uplane_info.iq_data_buffer;
                            int64_t slot_offset    = 0 + ul_u_tx_time_advance_ns;

                            UPlaneTX uplane{
                                .flow           = uplane_flows_[cell_id][eAxC],
                                .eAxC           = eAxC,
                                .vlan_id        = vlan_id,
                                .slot_offset    = slot_offset,
                                .iq_data_buffer = iq_data_buffers_[iq_data_buffer].first,
                                .symbol_id      = symbol_id,
                                .section_id     = uplane_info.section_id,
                            };

                            for(auto& section_info: uplane_info.section_list)
                            {
                                UPlaneTXSection section {
                                    .start_prbu = section_info.start_prbu,
                                    .num_prbu = section_info.num_prbu
                                };
                                uplane.section_list.emplace_back(section);
                            }

                            if(slot_id >= slot_count)
                            {
                                THROW(StringBuilder() << "Invalid slot_id (" << slot_id << ") for U-plane flow with eAxC=" << eAxC << ", vlan_id=" << vlan_id << ". Cell " << cell_id << " slot count is " << slot_count);
                            }
                            worker_context.ul_tx_worker_context.uplane[worker_context.ul_tx_worker_context.num_peers][slot_id].push_back(uplane);
                        }
                    }
                }
                ++worker_context.ul_tx_worker_context.num_peers;
            }
            num_peers_added += num_peers_per_cpu[i];
            worker_info.worker_type_ = WorkerType::UL_TX;

            if((is_ru() && yaml_parser_.enable_ulu()) && num_peers_per_cpu[i] > 0)
                workers_.push_back(std::make_unique<Worker>(worker_context, worker_info));
        }
    }

    // Launch DL RX workers
    {
        auto dl_num_cpus = dl_cpus_list.size();
        auto cpu_to_use = dl_cpus_list.begin();

        std::vector<int> num_peers_per_cpu;
        for(int i = 0; i < dl_cpus_list.size(); ++i)
        {
            num_peers_per_cpu.push_back(0);
        }

        auto num_peers_added = 0;
        while(num_peers_added < peer_infos.size())
        {
            ++num_peers_per_cpu[num_peers_added%num_peers_per_cpu.size()];
            ++num_peers_added;
        }
        num_peers_added = 0;

        for(int i = 0; i < dl_cpus_list.size(); ++i)
        {
            WorkerContext worker_context{
                .fhgen              = this,
                .index              = 0,
                .peer               = nullptr,
                .nic                = nullptr,
                .time_anchor        = time_anchor_.load(),
                .slot_duration      = slot_duration,
                .slot_count         = slot_count,
                .test_slots         = test_slots,
                .oran_slot_iterator = oran_slot_iterator,
                .buffer_ready_gdr   = buffer_ready_gdr
            };

            WorkerInfo worker_info{
                .fh_gen_type_         = fh_gen_type_,
                .cpu_core             = *(cpu_to_use++),
                .priority             = worker_priority,
                .peer_id              = 0,
            };

            worker_context.dl_rx_worker_context.num_peers = 0;

            for(int peer_index = num_peers_added; peer_index < num_peers_added + num_peers_per_cpu[i]; ++peer_index)
            {
                auto& peer_info       = peer_infos[peer_index];
                auto peer_id          = peer_info.info.id;
                auto peer             = peers_[peer_id];
                auto tcp_adv_dl_ns    = peer_info.tcp_adv_dl_ns;
                auto t1a_max_cp_ul_ns = peer_info.t1a_max_cp_ul_ns;
                auto t1a_max_up_ns    = peer_info.t1a_max_up_ns;
                auto ud_comp_info     = peer_info.info.ud_comp_info;

                worker_context.dl_rx_worker_context.peers.push_back(peer);
                worker_context.dl_rx_worker_context.tcp_adv_dl_ns.push_back(peer_info.tcp_adv_dl_ns);
                worker_context.dl_rx_worker_context.t1a_max_cp_ul_ns.push_back(peer_info.t1a_max_cp_ul_ns);
                worker_context.dl_rx_worker_context.t1a_max_up_ns.push_back(peer_info.t1a_max_up_ns);
                worker_context.dl_rx_worker_context.window_end_ns.push_back(peer_info.window_end_ns);
                worker_context.dl_rx_worker_context.peer_ids.push_back(peer_id);
                ++worker_context.dl_rx_worker_context.num_peers;
            }
            num_peers_added += num_peers_per_cpu[i];
            worker_info.worker_type_ = WorkerType::DL_RX;
            if(num_peers_per_cpu[i] > 0)
                workers_.push_back(std::make_unique<Worker>(worker_context, worker_info));
        }
    }

}

void cuda_deallocator(void* addr)
{
    CHECK_CUDA_THROW(cudaFree(addr));
}

void FhGenerator::initialize_random_number_generator()
{
    auto random_seed = yaml_parser_.get_shuffle_info().random_seed;

    if(random_seed < 0)
    {
        NVLOGI_FMT(TAG, "Initializing random device with non-deterministic seed value");
        std::random_device random_device;
        random_engine_.seed(random_device());
    }
    else
    {
        NVLOGI_FMT(TAG, "Initializing random device with seed value {}" , random_seed);
        random_engine_.seed(random_seed);
    }
}

void FhGenerator::free_resources()
{
    resources_to_free_.memregs.clear();
    resources_to_free_.cpu_buffers.clear();
    resources_to_free_.gpu_buffers.clear();
    resources_to_free_.flows.clear();
    resources_to_free_.peers.clear();
    resources_to_free_.nics.clear();
}

void FhGenerator::increment_dl_counter(int cell_index, int type, int slot, int timing)
{
    switch (type)
    {
    case DLPacketCounterType::ULC:
        ulc_stats.increment_counters(cell_index, timing, slot);
        break;
    case DLPacketCounterType::DLC:
        dlc_stats.increment_counters(cell_index, timing, slot);
        break;
    case DLPacketCounterType::DLU:
        dlu_stats.increment_counters(cell_index, timing, slot);
        break;
    default:
        break;
    }
}

uint64_t FhGenerator::get_cell_type_timing_count(int cell_index, int type, int timing)
{
    switch (type)
    {
    case DLPacketCounterType::ULC:
        return ulc_stats.get_cell_timing_count(cell_index, timing);
        break;
    case DLPacketCounterType::DLC:
        return dlc_stats.get_cell_timing_count(cell_index, timing);
        break;
    case DLPacketCounterType::DLU:
        return dlu_stats.get_cell_timing_count(cell_index, timing);
        break;
    default:
        return 0;
        break;
    }
}

uint64_t FhGenerator::get_cell_total_count(int cell_index, int type)
{
    switch (type)
    {
    case DLPacketCounterType::ULC:
        return ulc_stats.get_cell_total_count(cell_index);
        break;
    case DLPacketCounterType::DLC:
        return dlc_stats.get_cell_total_count(cell_index);
        break;
    case DLPacketCounterType::DLU:
        return dlu_stats.get_cell_total_count(cell_index);
        break;
    default:
        return 0;
        break;
    }
}

float FhGenerator::get_cell_timing_percentage(int cell_index, int type, int timing)
{
    switch (type)
    {
    case DLPacketCounterType::ULC:
        return ulc_stats.get_cell_timing_percentage(cell_index, timing);
        break;
    case DLPacketCounterType::DLC:
        return dlc_stats.get_cell_timing_percentage(cell_index, timing);
        break;
    case DLPacketCounterType::DLU:
        return dlu_stats.get_cell_timing_percentage(cell_index, timing);
        break;
    default:
        return 0;
        break;
    }
}

float FhGenerator::get_ulu_cell_timing_percentage(int cell_index, int type, int timing)
{
    return ulu_stats.get_cell_timing_percentage(cell_index, timing);
}

uint64_t FhGenerator::get_ulu_counter_value(int cell_index, int timing)
{
    return ulu_stats.get_cell_timing_count(cell_index, timing);
}

void FhGenerator::print_periodic_counters()
{
    char buffer[kMaxLogLength];
    int buffer_index = 0;

    if(fh_gen_type_ == FhGenType::RU)
    {
        for(int cell = 0; cell < peers_.size(); ++cell)
        {
            buffer_index = snprintf(buffer, kMaxLogLength, "Cell %2d ", cell);
            for(int type = 0; type < DLPacketCounterType::DLCounterTypeMax; ++type)
            {
                buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "%s | ", packet_type_to_char_string(type));
                for(int timing = 0; timing < PacketCounterTiming::CounterTimingMax; ++timing)
                {
                    buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "%8lu %s | ", get_cell_type_timing_count(cell, type, timing), packet_timing_to_char_string(timing));
                }
            }
            // buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "\n");
            NVLOGC_FMT(TAG,"{}",buffer);
        }
    }
    else
    {
        for(int cell = 0; cell < peers_.size(); ++cell)
        {
            buffer_index = snprintf(buffer, kMaxLogLength, "Cell %2d ", cell);
            buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "%s | ", "UL U");
            for(int timing = 0; timing < PacketCounterTiming::CounterTimingMax; ++timing)
            {
                buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "%8lu %s | ", get_ulu_counter_value(cell, timing), packet_timing_to_char_string(timing));
            }
            // buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "\n");
            NVLOGC_FMT(TAG,"{}",buffer);
        }
    }
}

void FhGenerator::print_cell_stats(int cell)
{
    char buffer[kMaxLogLength];
    int buffer_index = 0;

    buffer_index = snprintf(buffer, kMaxLogLength, "Cell %2d ", cell);
    for(int type = 0; type < DLPacketCounterType::DLCounterTypeMax; ++type)
    {
        buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "%s | ", packet_type_to_char_string(type));
        for(int timing = 0; timing < PacketCounterTiming::CounterTimingMax; ++timing)
        {
            float percentage = get_cell_timing_percentage(cell, type, timing);
            buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "%0.2f%s %s | ", percentage, "%", packet_timing_to_char_string(timing));
        }

        buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "total rx %lu | ", get_cell_total_count(cell, type));
    }
    // buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "\n");
    NVLOGC_FMT(TAG,"{}",buffer);
}


void FhGenerator::print_ru_summary_stats()
{
    for(int cell = 0; cell < peers_.size(); ++cell)
    {
        print_cell_stats(cell);
    }
    // print_late_packet_stats();
}


void FhGenerator::print_late_packet_stats()
{
    char buffer[kMaxLogLength];
    NVLOGC_FMT(TAG, "Late U-Plane packets per slot: ");
    int buffer_index = 0;
    // buffer_index = snprintf(buffer, kMaxLogLength, "Late U-Plane packets per slot: ");
    uint64_t total = 0;
    for(int slot = 0; slot < yaml_parser_.slot_count(); ++slot)
    {
        buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "Slot %2d %5lu | ", slot, late_packet_counters[slot].load());
        total += late_packet_counters[slot].load();
        if(slot % 10 == 9)
        {
            NVLOGC_FMT(TAG,"{}",buffer);
            buffer_index = 0;
        }
    }
    if(buffer_index != 0)
    {
        NVLOGC_FMT(TAG,"{}",buffer);
    }
    // NVLOGC_FMT(TAG, "Latest slots {} of total late packets: ", "%");
    // buffer_index = 0;
    // total = (total == 0) ? 1 : total;
    // for(int slot = 0; slot < yaml_parser_.slot_count(); ++slot)
    // {
    //     buffer_index += snprintf(buffer + buffer_index, kMaxLogLength - buffer_index, "Slot %2d %5.2f | ", slot, (float)(late_packet_counters[slot].load() * 100) / total);
    //     if(slot % 10 == 9)
    //     {
    //         NVLOGC_FMT(TAG,"{}",buffer);
    //         buffer_index = 0;
    //     }
    // }
    // if(buffer_index != 0)
    // {
    //     NVLOGC_FMT(TAG,"{}",buffer);
    // }
}

aerial_fh::NicHandle FhGenerator::get_nic_handle_from_name(std::string nic_name)
{
    if(nics_.find(nic_name) == nics_.end())
    {
        return nullptr;
    }
    return nics_[nic_name];
}

} // namespace fh_gen
