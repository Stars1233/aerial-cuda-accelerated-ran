/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "fh.hpp"

void RU_Emulator::open_fh_driver(){
    aerial_fh::FronthaulInfo fh_info{};
    fh_info.dpdk_thread            = opt_afh_dpdk_thread;
    fh_info.accu_tx_sched_res_ns   = opt_afh_accu_tx_sched_res_ns;
    fh_info.pdump_client_thread    = opt_afh_pdump_client_thread;
    fh_info.fh_stats_dump_cpu_core = -1;

    if (fh_info.accu_tx_sched_res_ns > 0)
        fh_info.accu_tx_sched_disable = false;
    else
        fh_info.accu_tx_sched_disable = true;

    fh_info.dpdk_verbose_logs = false;
    fh_info.dpdk_file_prefix = opt_afh_dpdk_file_prefix;
    fh_info.cuda_device_ids = {};
    fh_info.cuda_device_ids_for_compute = {};
    fh_info.rivermax = false;
    int ret = aerial_fh::open(&fh_info, &fronthaul);
    if(ret)
    {
        do_throw(sb() << "FH driver init error\n");
    }
}

void RU_Emulator::add_nics(){

    for (int i = 0; i < nic_interfaces.size(); ++i)
    {
        auto nic_interface = nic_interfaces[i];
        aerial_fh::NicInfo nic_info;
        aerial_fh::NicHandle nic;
        bool cx6_nic=true;

        nic_info.mtu = opt_afh_mtu;
        nic_info.per_rxq_mempool = (opt_aerial_fh_per_rxq_mempool==1)?true:false;
        nic_info.cpu_mbuf_num = 0;
        nic_info.cpu_mbuf_rx_num_per_rxq = opt_afh_cpu_mbuf_pool_size_per_rxq;
        nic_info.cpu_mbuf_tx_num = opt_afh_cpu_mbuf_pool_tx_size;
        nic_info.cpu_mbuf_rx_num = opt_afh_cpu_mbuf_pool_rx_size;
        nic_info.tx_request_num = opt_afh_txq_request_num;
        nic_info.cuda_device = -1;
        nic_info.rx_ts_enable = true;
        nic_info.txq_count_gpu = 0;
        nic_info.txq_count = 0;
        nic_info.rxq_count = 0;
        for(auto cell: cell_configs)
        {
            if(cell.nic_index == i)
            {
                // TODO: fix for only assigned 1 RXQ for DLU
                nic_info.rxq_count += cell.eAxC_DL.size(); // rxq * flows
                nic_info.rxq_count += 1; // rxq for Cplane
                nic_info.rxq_count += 1; // rxq for SRS
            }
        }

        // TODO: update add nic API to be able to determine CX6 NIC without providing RXQ count
        // if(enable_srs)
        // {
        //     nic_info.rxq_count = 0;
        //     for(auto cell: cell_configs)
        //     {
        //         if(cell.nic_index == i)
        //         {
        //             nic_info.rxq_count += 1; // rxq * DLU
        //             nic_info.rxq_count += 1; // rxq for Cplane
        //             nic_info.rxq_count += 1; // rxq for SRS
        //         }
        //     }
        // }
        nic_info.txq_count =  ORAN_ALL_SYMBOLS * 2;
        if(opt_split_srs_txq)
        {
            nic_info.txq_count += num_srs_txqs;
        }

        nic_info.txq_size = opt_afh_txq_size;
        nic_info.rxq_size = opt_afh_rxq_size;

        nic_info.name = nic_interface.c_str();
        nic_info.split_cpu_mp = opt_afh_split_rx_tx_mp;
        auto ret = aerial_fh::add_nic(fronthaul, &nic_info, &nic);
        if(ret)
        {
            do_throw(sb() << "Failed to add NIC " << nic_info.name << "\n");
        }
        else
        {
            if(i==0) //Check only for the first NIC interface. Presume that any additional NIC interfaces are all of the first type
            {
                aerial_fh::is_cx6_nic(nic,cx6_nic);
                is_cx6_nic=cx6_nic;
            }
        }
        nic_info_list.emplace_back(nic_info);
        nic_list.emplace_back(nic);
    }
}

void RU_Emulator::add_peers(){

    // ADD PEERS
    aerial_fh::PeerId peer_id_counter = 0;

    for (auto cell_conf : cell_configs)
    {
        aerial_fh::PeerInfo peer_info{};
        aerial_fh::PeerHandle peer{};
        std::vector<uint16_t> eAxC_list_uplink;
        std::vector<uint16_t> eAxC_list_srs;
        std::vector<uint16_t> eAxC_list_dl;
        peer_info.id = peer_id_counter;

        memcpy( peer_info.src_mac_addr.bytes, cell_conf.eth_addr.addr_bytes, sizeof(cell_conf.eth_addr.addr_bytes)); // cell_eth
        memcpy( peer_info.dst_mac_addr.bytes, dpdk.peer_eth_addr[cell_conf.peer_index].addr_bytes, sizeof(dpdk.peer_eth_addr[cell_conf.peer_index].addr_bytes)); // peer_eth

        if(cell_conf.ul_comp_meth != aerial_fh::UserDataCompressionMethod::NO_COMPRESSION && cell_conf.ul_comp_meth != aerial_fh::UserDataCompressionMethod::BLOCK_FLOATING_POINT)
        {
            do_throw(sb() << "Compression scheme not supported " << (int)cell_conf.ul_comp_meth);
        }

        peer_info.ud_comp_info = {(size_t)(cell_conf.ul_bit_width), cell_conf.ul_comp_meth};
        peer_info.max_num_prbs_per_symbol = cell_conf.ulGridSize;
        peer_info.txq_count_uplane = ORAN_ALL_SYMBOLS * 2;
        if(opt_split_srs_txq)
        {
            peer_info.txq_count_uplane += num_srs_txqs;
        }
        peer_info.txq_count_uplane_gpu = 0;
        peer_info.rx_mode = aerial_fh::RxApiMode::HYBRID;
        peer_info.vlan.tci = cell_conf.vlan;
        peer_info.txq_cplane = false;
        peer_info.share_txqs = true;


        for(auto eAxC : cell_conf.eAxC_UL)
        {
            eAxC_list_uplink.push_back(eAxC);
        }

        for(auto eAxC : cell_conf.eAxC_PRACH_list)
        {
            auto it = std::find(std::begin(eAxC_list_uplink), std::end(eAxC_list_uplink), (uint16_t) eAxC);
            if(it!=std::end(eAxC_list_uplink))
                continue;
            eAxC_list_uplink.push_back(eAxC);
        }

        for(auto eAxC : cell_conf.eAxC_SRS_list)
        {
            eAxC_list_srs.push_back(eAxC);
        }

        bool has_overlap = false; //Assume that if there is overlap then the SRS is not enabled
        std::set<uint16_t> uplink_set(eAxC_list_uplink.begin(), eAxC_list_uplink.end());

        for (const auto& srs_val : eAxC_list_srs) {
            if (uplink_set.find(srs_val) != uplink_set.end()) {
                has_overlap = true;
                break;
            }
        }

        for (auto eAxC : cell_conf.eAxC_DL)
        {
            if (uplink_set.find(eAxC) != uplink_set.end())
            {
                continue;
            }
            eAxC_list_dl.push_back(eAxC);
        }

        if(has_overlap)
        {
            enable_srs = false;
            peer_info.enable_srs = enable_srs;
            eAxC_list_srs.clear();
            eAxC_list_uplink.clear();
        }
        else if(opt_enable_mmimo)
        {
            enable_srs = true;
            peer_info.enable_srs = enable_srs;
        }

        // Note: eAxC_list_dl is unused for now
        auto ret = aerial_fh::add_peer(nic_list[cell_conf.nic_index], &peer_info, &peer, eAxC_list_uplink, eAxC_list_srs, eAxC_list_dl);
        if(ret)
        {
            do_throw(sb() << "Failed to add Peer " << peer_info.id << "\n");
        }

        peer_list[peer_id_counter] = peer;
        peer_id_counter++;
    }

}

void RU_Emulator::add_flows(){

   // Find unique set of flows
    std::vector<int> num_of_flows_with_rq;
    std::vector<std::vector<int>> unique_flows;
    void* rxq;
    for(int cell_idx = 0; cell_idx < opt_num_cells; ++cell_idx)
    {
        // Add flow list

        std::set<int> flow_set(cell_configs[cell_idx].eAxC_DL.begin(),  cell_configs[cell_idx].eAxC_DL.end());
        unique_flows.push_back(cell_configs[cell_idx].eAxC_DL);
        num_of_flows_with_rq.push_back(cell_configs[cell_idx].eAxC_DL.size());


        //Below flows do not need RXQ
        // Add UL list
        for(auto it : cell_configs[cell_idx].eAxC_UL ){
            if (flow_set.find(it) != flow_set.end())
                continue;

            unique_flows[cell_idx].push_back(it);
        }

        // Add PRACH list
        for(auto it : cell_configs[cell_idx].eAxC_PRACH_list ){
            if (flow_set.find(it) != flow_set.end())
                continue;

            unique_flows[cell_idx].push_back(it);
        }

        // Add SRS list
        for(auto it : cell_configs[cell_idx].eAxC_SRS_list ){
            if (flow_set.find(it) != flow_set.end())
                continue;

            unique_flows[cell_idx].push_back(it);
        }
    }

    // Add DL UPlane flows
    for(uint16_t i = 0; i < opt_num_cells; ++i)
    {
        rxq=nullptr;
        for(int flow_idx = 0; flow_idx < cell_configs[i].eAxC_DL.size(); ++flow_idx)
        {
            aerial_fh::FlowHandle flow;
            aerial_fh::FlowInfo flowinfo{};
            aerial_fh::VlanTci vlan_tci;
            vlan_tci.tci = cell_configs[i].vlan;
            flowinfo.flow_rx_mode = aerial_fh::FlowRxApiMode::TXANDRX;
            flowinfo.eAxC = cell_configs[i].eAxC_DL[flow_idx];
            flowinfo.type = aerial_fh::FlowType::UPLANE;
            flowinfo.vlan_tag = vlan_tci;
            if(!is_cx6_nic)
            {
                if(flow_idx==0)
                {
                    flowinfo.request_new_rxq=true;
                }
                else
                {
                    flowinfo.request_new_rxq=false;
                    flowinfo.rxq=rxq;
                }
            }
            else //Request a new RxQ for every flow ID for CX-6 NIC HW to avoid perf degredation on ULU timing
            {
                flowinfo.request_new_rxq=true;
            }
            auto ret = aerial_fh::add_flow(peer_list[i], &flowinfo, &flow);
            if(ret)
            {
                do_throw(sb() << "Failed to add U-Plane Flow" << "\n");
            }
            if(!is_cx6_nic)
            {
                if(flow_idx==0)
                {
                    rxq=flowinfo.rxq;
                }
            }
            dl_peer_flow_map[i].push_back(flow);
        }
        // std::cout<<"Number of U-Plane flows added for Peer "<<i<<" : "<<peer_flow_map[i].size()<<"\n";
    }

    for(uint16_t i = 0; i < opt_num_cells; ++i)
    {
        ul_peer_flow_map.emplace_back(std::vector<aerial_fh::FlowHandle>());
        peer_flow_map_prach.emplace_back(std::vector<aerial_fh::FlowHandle>());
        peer_flow_map_srs.emplace_back(std::vector<aerial_fh::FlowHandle>());
    }

    // Add UL Section type 1 flows
    for(uint16_t i = 0; i < opt_num_cells; ++i)
    {
        for(int flow_idx = 0; flow_idx < cell_configs[i].eAxC_UL.size(); ++flow_idx)
        {
            aerial_fh::FlowHandle flow;
            aerial_fh::FlowInfo flowinfo;
            aerial_fh::VlanTci vlan_tci;
            vlan_tci.tci = cell_configs[i].vlan;
            flowinfo.flow_rx_mode = aerial_fh::FlowRxApiMode::TXONLY;
            flowinfo.eAxC = cell_configs[i].eAxC_UL[flow_idx];
            flowinfo.type = aerial_fh::FlowType::UPLANE;
            flowinfo.vlan_tag = vlan_tci;

            auto ret = aerial_fh::add_flow(peer_list[i], &flowinfo, &flow);
            if(ret)
            {
                do_throw(sb() << "Failed to add U-Plane Flow" << "\n");
            }
            ul_peer_flow_map[i].emplace_back(flow);
        }
    }

    // Add UL Section type 3 flows
    for(uint16_t i = 0; opt_prach_enabled == RE_ENABLED && i < opt_num_cells; ++i)
    {
        for(int flow_idx = 0; flow_idx < cell_configs[i].eAxC_PRACH_list.size(); ++flow_idx)
        {
            aerial_fh::FlowHandle flow;
            aerial_fh::FlowInfo flowinfo;
            aerial_fh::VlanTci vlan_tci;
            vlan_tci.tci = cell_configs[i].vlan;
            flowinfo.flow_rx_mode = aerial_fh::FlowRxApiMode::TXONLY;
            flowinfo.eAxC = cell_configs[i].eAxC_PRACH_list[flow_idx];
            flowinfo.type = aerial_fh::FlowType::UPLANE;
            flowinfo.vlan_tag = vlan_tci;
            auto ret = aerial_fh::add_flow(peer_list[i], &flowinfo, &flow);
            if(ret)
            {
                do_throw(sb() << "Failed to add U-Plane Flow" << "\n");
            }
            peer_flow_map_prach[i].push_back(flow);
        }
    }


    // Add UL Section type 1 SRS flows
    for(uint16_t i = 0; opt_srs_enabled == RE_ENABLED && i < opt_num_cells; ++i)
    {
        for(int flow_idx = 0; flow_idx < cell_configs[i].eAxC_SRS_list.size(); ++flow_idx)
        {
            aerial_fh::FlowHandle flow;
            aerial_fh::FlowInfo flowinfo;
            aerial_fh::VlanTci vlan_tci;
            vlan_tci.tci = cell_configs[i].vlan;
            flowinfo.flow_rx_mode = aerial_fh::FlowRxApiMode::TXONLY;
            flowinfo.eAxC = cell_configs[i].eAxC_SRS_list[flow_idx];
            flowinfo.type = aerial_fh::FlowType::UPLANE;
            flowinfo.vlan_tag = vlan_tci;
            auto ret = aerial_fh::add_flow(peer_list[i], &flowinfo, &flow);
            if(ret)
            {
                do_throw(sb() << "Failed to add U-Plane Flow" << "\n");
            }
            peer_flow_map_srs[i].push_back(flow);
        }
    }
}

void RU_Emulator::init_fh(){

    open_fh_driver();
    add_nics();
    add_peers();

    // add_flows();
    // start_fh_driver();
}

