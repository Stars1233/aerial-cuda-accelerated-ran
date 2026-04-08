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

#include "yaml_parser.hpp"

#include <cstring>
#include <string>

#include <sstream>

#include "fh_generator.hpp"

#undef TAG
#define TAG "FHGEN.YAML"

namespace fh_gen
{
YamlParser::YamlParser(FhGenerator* fh_generator, const std::string& config_file) :
    fh_gen_{fh_generator}
{
    if(!file_exists(config_file))
    {
        THROW(StringBuilder() << "Config file: '" << config_file << "' not found!");
    }

    yaml::file_parser fp{config_file.c_str()};
    auto              doc  = fp.next_document();
    auto              root = doc.root();

    NVLOGI_FMT(TAG, "Parsing config file {}", config_file.c_str());

    workers_priority_             = root[kWorkerThreadSchedFifoPrioYaml].as<int32_t>();
    slot_duration_                = root[kSlotDurationYaml].as<aerial_fh::Ns>();
    slot_count_                   = root[kSlotCountYaml].as<uint8_t>();
    enable_ulu_                    = static_cast<bool>(root[kEnableULUYaml].as<uint32_t>());
    enable_dlu_                    = static_cast<bool>(root[kEnableDLUYaml].as<uint32_t>());
    enable_ulc_                    = static_cast<bool>(root[kEnableULCYaml].as<uint32_t>());
    enable_dlc_                    = static_cast<bool>(root[kEnableDLCYaml].as<uint32_t>());
    test_slots_                   = root[kTestSlotsYaml].as<uint64_t>();
    dl_u_enq_time_advance_ns_     = root[kDLUEnqTimeAdvanceYaml].as<aerial_fh::Ns>();
    dl_c_enq_time_advance_ns_     = root[kDLCEnqTimeAdvanceYaml].as<aerial_fh::Ns>();
    ul_u_tx_time_advance_ns_      = root[kULUTxTimeAdvanceYaml].as<aerial_fh::Ns>();
    ul_u_enq_time_advance_ns_      = root[kULUEnqTimeAdvanceYaml].as<aerial_fh::Ns>();
    ulc_ontime_pass_percentage_      = root[kULCOntimePassPctYaml].as<float>();
    ulu_ontime_pass_percentage_      = root[kULUOntimePassPctYaml].as<float>();
    dlc_ontime_pass_percentage_      = root[kDLCOntimePassPctYaml].as<float>();
    dlu_ontime_pass_percentage_      = root[kDLUOntimePassPctYaml].as<float>();
    parse_start_slot_info(root);
    parse_fh_info(root);
    if(fh_gen_->is_du())
    {
        parse_nic_info(root[kNicsYaml]);
    }
    else
    {
        parse_nic_info(root[kRuNicsYaml]);
    }
    parse_iq_data_buffer_info(root[kIqDataBuffersYaml]);
    parse_peer_info(root[kPeersYaml]);
    parse_flow_info(root[kFlowsYaml]);
    parse_sfn_slot_sync_info(root[kSfnSlotSyncYaml]);
    if(fh_gen_->is_du())
    {
        parse_cpu_info(root[kDuCpusYaml]);
    }
    else
    {
        parse_cpu_info(root[kRuCpusYaml]);
    }
}

const aerial_fh::FronthaulInfo& YamlParser::get_fh_info() const
{
    return fh_info_;
}

const std::vector<aerial_fh::NicInfo>& YamlParser::get_nic_info() const
{
    return nic_info_;
}

const std::vector<IqDataBufferInfo>& YamlParser::get_iq_data_buffer_info() const
{
    return iq_data_buffer_info_;
}

const std::vector<PeerInfo>& YamlParser::get_peer_info() const
{
    return peer_info_;
}

const std::vector<fh_gen::FlowInfo>& YamlParser::get_flow_info() const
{
    return flow_info_;
}

const std::vector<CpuId>& YamlParser::get_cpus() const
{
    return cpus_;
}

const std::vector<CpuId>& YamlParser::get_ul_cpus() const
{
    return ul_cpus_;
}

const std::vector<CpuId>& YamlParser::get_dl_cpus() const
{
    return dl_cpus_;
}

int32_t YamlParser::get_workers_priority() const
{
    return workers_priority_;
}

bool YamlParser::send_utc_anchor() const
{
    return send_utc_anchor_;
}

bool YamlParser::validate_iq_data_buffer_size() const
{
    return validate_iq_data_buffer_size_;
}

bool YamlParser::enable_ulu() const
{
    return enable_ulu_;
}

bool YamlParser::enable_c_plane() const
{
    return enable_dlc_ ||  enable_ulc_;
}

bool YamlParser::enable_dlc() const
{
    return enable_dlc_;
}

bool YamlParser::enable_ulc() const
{
    return enable_ulc_;
}

bool YamlParser::enable_dlu() const
{
    return enable_dlu_;
}

ShuffleInfo YamlParser::get_shuffle_info() const
{
    return shuffle_info_;
}

StartSlotInfo YamlParser::get_start_slot_info() const
{
    return start_slot_info_;
}

SfnSlotSyncInfo YamlParser::get_sync_info() const
{
    return sync_info_;
}


aerial_fh::Ns YamlParser::slot_duration() const
{
    return slot_duration_;
}

aerial_fh::Ns YamlParser::dl_u_enq_time_advance_ns() const
{
    return dl_u_enq_time_advance_ns_;
}

aerial_fh::Ns YamlParser::dl_c_enq_time_advance_ns() const
{
    return dl_c_enq_time_advance_ns_;
}

aerial_fh::Ns YamlParser::ul_u_tx_time_advance_ns() const
{
    return ul_u_tx_time_advance_ns_;
}

aerial_fh::Ns YamlParser::ul_u_enq_time_advance_ns() const
{
    return ul_u_enq_time_advance_ns_;
}

uint16_t YamlParser::slot_count() const
{
    return slot_count_;
}

uint64_t YamlParser::test_slots() const
{
    return test_slots_;
}

float YamlParser::ulc_ontime_pass_percentage() const
{
    return ulc_ontime_pass_percentage_;
}

float YamlParser::ulu_ontime_pass_percentage() const
{
    return ulu_ontime_pass_percentage_;
}

float YamlParser::dlc_ontime_pass_percentage() const
{
    return dlc_ontime_pass_percentage_;
}

float YamlParser::dlu_ontime_pass_percentage() const
{
    return dlu_ontime_pass_percentage_;
}


void YamlParser::parse_fh_info(yaml::node node)
{
    aerial_fh::FronthaulInfo fh_info{
        .dpdk_thread           = node[kFhDpdkThreadYaml].as<uint32_t>(),
        .accu_tx_sched_res_ns  = node[kFhAccuTxSchedResNsYaml].as<uint32_t>(),
        .pdump_client_thread   = -1,
        .accu_tx_sched_disable = false,
        .dpdk_verbose_logs     = static_cast<bool>(node[kFhDpdkVerboseLogsYaml].as<uint32_t>()),
        .dpdk_file_prefix      = "fh_generator",
        .cuda_device_ids       = {0},
        .cuda_device_ids_for_compute       = {0},
        .rivermax              = false,
        .fh_stats_dump_cpu_core = -1,
        .cpu_rx_only            = false
    };

    fh_info_ = fh_info;
}

void YamlParser::parse_nic_info(yaml::node nic_list)
{
    for(size_t i = 0; i < nic_list.length(); ++i)
    {
        auto node = nic_list[i];

        aerial_fh::NicInfo nic_info{
            .name           = node[kNicNameYaml].as<std::string>().c_str(),
            .mtu            = node[kNicMtuYaml].as<uint16_t>(),
            .per_rxq_mempool = false,
            .cpu_mbuf_num   = node[kNicCpuMbufsYaml].as<uint32_t>(),
            .cpu_mbuf_tx_num = (fh_gen_->is_du()) ? (uint32_t)0 : (uint32_t)262144,
            .cpu_mbuf_rx_num = (fh_gen_->is_du()) ? (uint32_t)0 : (uint32_t)262144,
            .cpu_mbuf_rx_num_per_rxq = 0,
            .tx_request_num = node[kNicUplaneTxHandlesYaml].as<uint32_t>(),
            .txq_count      = node[kNicTxqCountYaml].as<uint16_t>(),
            .txq_count_gpu  = node[kNicTxqCountYaml].as<uint16_t>(),
            .rxq_count      = node[kNicRxqCountYaml].as<uint16_t>(),
            .txq_size       = node[kNicTxqSizeYaml].as<uint16_t>(),
            .rxq_size       = node[kNicRxqSizeYaml].as<uint16_t>(),
            .cuda_device    = node[kNicCudaDeviceIdYaml].as<uint16_t>(),
            .rx_ts_enable = false
        };

        nic_info_.push_back(nic_info);
    }
}

void YamlParser::parse_iq_data_buffer_info(yaml::node iq_data_buffer_list)
{
    for(size_t i = 0; i < iq_data_buffer_list.length(); ++i)
    {
        auto node           = iq_data_buffer_list[i];
        auto cuda_device_id = node[kIqCudaDeviceIdYaml].as<int32_t>();

        aerial_fh::MemRegInfo iq_data_buffer_info{
            .addr    = nullptr,
            .len     = node[kIqDataBufferSizeYaml].as<size_t>(),
            .page_sz = cuda_device_id < 0 ? static_cast<size_t>(sysconf(_SC_PAGESIZE)) : kNvGpuPageSize,
        };

        std::string input_file = node.has_key(kIqInputFileYaml) ? node[kIqInputFileYaml].as<std::string>() : "";

        IqDataBufferInfo iq_data_buffer_info_ext{
            .info           = iq_data_buffer_info,
            .id             = node[kIqDataBufferIdYaml].as<MemId>(),
            .cuda_device_id = cuda_device_id,
            .input_file     = input_file,
        };

        auto buffer_len = iq_data_buffer_info.len;
        if(buffer_len < kMinIqBufferSize)
        {
            THROW(StringBuilder() << "IQ data buffer size must be at least " << kMinIqBufferSize << " bytes. Found: " << buffer_len);
        }

        iq_data_buffer_info_.push_back(iq_data_buffer_info_ext);
    }
}

static void parse_eth_addr(const std::string& eth_addr_str, aerial_fh::MacAddr& eth_addr)
{
    char* pch = nullptr;
    int   i = 0, tmp = 0;

    pch = strtok(const_cast<char*>(eth_addr_str.c_str()), ":");
    while(pch != nullptr)
    {
        std::stringstream str;
        std::string       stok;
        stok.assign(pch);
        str << stok;
        str >> std::hex >> tmp;
        eth_addr.bytes[i] = static_cast<uint8_t>(tmp);
        pch               = strtok(nullptr, ":");
        i++;
    }
}

void YamlParser::parse_peer_info(yaml::node peer_list)
{
    for(size_t i = 0; i < peer_list.length(); ++i)
    {
        auto node = peer_list[i];

        aerial_fh::VlanTci vlan;
        vlan.pcp = node[kPeerVlanPcpYaml].as<uint16_t>();
        vlan.dei = 0;
        vlan.vid = node[kPeerVlanIdYaml].as<uint16_t>();

        aerial_fh::PeerInfo peer_info{
            .id = node[kPeerIdYaml].as<uint16_t>(),
            .src_mac_addr{},
            .dst_mac_addr{},
            .vlan = vlan,
            .ud_comp_info     = {node[kPeerUdIqWithYaml].as<size_t>(), static_cast<aerial_fh::UserDataCompressionMethod>(node[kPeerUdCompMethYaml].as<int>())},
            .txq_count_uplane = node[kPeerTxqCountUplaneYaml].as<uint8_t>(),
            .txq_count_uplane_gpu = node[kPeerTxqCountUplaneYaml].as<uint8_t>(),
            .rx_mode          = aerial_fh::RxApiMode::PEER,
            .txq_cplane       = true,
        };

        parse_eth_addr(static_cast<std::string>(node[kPeerSrcMacAddrYaml]), peer_info.src_mac_addr);
        parse_eth_addr(static_cast<std::string>(node[kPeerDstMacAddrYaml]), peer_info.dst_mac_addr);

        auto nic_name = node[kNicNameYaml].as<std::string>();
        int nic_id = -1;
        for(int j = 0; j < nic_info_.size(); ++j)
        {
            if(nic_info_[j].name.compare(nic_name) == 0)
            {
                nic_id = j;
                break;
            }
        }

        if(fh_gen_->is_du() && nic_id == -1)
        {
            THROW(StringBuilder() << "NIC name " << nic_name << " of peer " << i << " does not match any listed NICs");
        }

        PeerInfo peer_info_ext{
            .index            = static_cast<uint8_t>(i),
            .info             = peer_info,
            .nic_name         = node[kNicNameYaml].as<std::string>(),
            .nic_id           = nic_id,
            .tcp_adv_dl_ns    = node[kPeerTcpAdvDlYaml].as<aerial_fh::Ns>(),
            .t1a_max_cp_ul_ns = node[kPeerT1aMaxCpUlYaml].as<aerial_fh::Ns>(),
            .t1a_max_up_ns    = node[kPeerT1aMaxUpYaml].as<aerial_fh::Ns>(),
            .ta4_min_ns       = node[kPeerTa4MinYaml].as<aerial_fh::Ns>(),
            .ta4_max_ns       = node[kPeerTa4MaxYaml].as<aerial_fh::Ns>(),
            .window_end_ns    = node[kPeerWindowEndYaml].as<aerial_fh::Ns>(),
        };

        if(fh_gen_->is_ru())
        {
            peer_info_ext.nic_name = node[kRuNicNameYaml].as<std::string>();
        }
        peer_info_.push_back(peer_info_ext);
    }
}

void YamlParser::parse_flow_info(yaml::node flow_list)
{
    for(size_t i = 0; i < flow_list.length(); ++i)
    {
        auto node = flow_list[i];

        aerial_fh::VlanTci vlan;
        auto cell_id = node[kFlowPeerIdYaml].as<uint16_t>();
        vlan = peer_info_[cell_id].info.vlan;

        aerial_fh::FlowInfo flow_info{
            .eAxC     = node[kFlowEaxcYaml].as<uint16_t>(),
            .type     = aerial_fh::FlowType::CPLANE,
            .vlan_tag = vlan,
        };

        fh_gen::FlowInfo flow_info_ext{
            .info    = flow_info,
            .cell_id = cell_id,
        };

        if(node.has_key(kCPlaneTxYaml))
        {
            parse_cplane_tx_info(node[kCPlaneTxYaml], flow_info_ext.cplane_tx_info);
        }

        if(node.has_key(kUPlaneTxYaml))
        {
            parse_uplane_info(node[kUPlaneTxYaml], flow_info_ext.uplane_tx_info);
        }

        if(node.has_key(kUPlaneRxYaml))
        {
            parse_uplane_info(node[kUPlaneRxYaml], flow_info_ext.uplane_rx_info);
        }

        flow_info_.push_back(flow_info_ext);
    }
}

void YamlParser::parse_cplane_tx_info(yaml::node cplane_tx_list, std::vector<CPlaneTxInfo>& cplane_tx_info)
{
    for(size_t i = 0; i < cplane_tx_list.length(); ++i)
    {
        auto node = cplane_tx_list[i];

        CPlaneTxInfo tx_info{
            .slot_id       = node[kTxSlotIdYaml].as<uint8_t>(),
            .symbol_id     = node[kTxSymbolIdYaml].as<uint8_t>(),
            .section_count = node[kTxSectionCount].as<uint16_t>(),
            .direction     = node[kTxDataDirectionYaml].as<std::string>(),
        };
        auto section_count = tx_info.section_count;

        yaml::node section = node[kTxSections];
        for(size_t j = 0; j < section.length(); ++j)
        {
            auto section_node = section[j];
            CPlaneSection section_info{
                .start_sym = section_node[kTxSectionStartSymYaml].as<uint8_t>(),
                .num_sym = section_node[kTxSectionNumSymYaml].as<uint8_t>(),
                .start_prbc = section_node[kTxSectionStartPrbYaml].as<uint16_t>(),
                .num_prbc = section_node[kTxSectionNumPrbYaml].as<uint16_t>()
            };
            tx_info.section_list.emplace_back(section_info);
        }

        if((section_count == 0) || (section_count > kMaxSectionNum))
        {
            THROW(StringBuilder() << "Invalid C-plane section count: " << section_count
                                  << " (valid range: 1-" << kMaxSectionNum << ")");
        }

        cplane_tx_info.push_back(tx_info);
    }
}

void YamlParser::parse_uplane_info(yaml::node uplane_tx_list, std::vector<UPlaneInfo>& uplane_tx_info)
{
    for(size_t i = 0; i < uplane_tx_list.length(); ++i)
    {
        auto     node       = uplane_tx_list[i];
        uint16_t section_id = node.has_key(kTxSectionIdYaml) ? node[kTxSectionIdYaml].as<uint16_t>() : 0;

        UPlaneInfo tx_info{
            .slot_id        = node[kTxSlotIdYaml].as<uint8_t>(),
            .symbol_id      = node[kTxSymbolIdYaml].as<uint8_t>(),
            .iq_data_buffer = node[kTxIqDataBufferYaml].as<MemId>(),
            .section_id     = section_id,
        };

        yaml::node section = node[kTxSections];
        for(size_t j = 0; j < section.length(); ++j)
        {
            auto section_node = section[j];
            UPlaneSection section_info{
                .start_prbu = section_node[kTxSectionStartPrbYaml].as<uint16_t>(),
                .num_prbu = section_node[kTxSectionNumPrbYaml].as<uint16_t>()
            };
            tx_info.section_list.emplace_back(section_info);
        }

        uplane_tx_info.push_back(tx_info);
    }
}

void YamlParser::parse_cpu_info(yaml::node cpu_list)
{
    auto ul_list = cpu_list["ul"];
    auto dl_list = cpu_list["dl"];
    auto cpu_count  = ul_list.length() + dl_list.length();

    for(size_t i = 0; i < ul_list.length(); ++i)
    {
        cpus_.push_back(ul_list[i].as<CpuId>());
        ul_cpus_.push_back(ul_list[i].as<CpuId>());
    }

    for(size_t i = 0; i < dl_list.length(); ++i)
    {
        cpus_.push_back(dl_list[i].as<CpuId>());
        dl_cpus_.push_back(dl_list[i].as<CpuId>());
    }
}

void YamlParser::parse_shuffle_info(yaml::node node)
{
    shuffle_info_.random_seed          = node[kRandomSeedYaml].as<int32_t>();
    shuffle_info_.cplane_shuffle_count = node[kShuffleCplaneTxYaml].as<uint32_t>();
    shuffle_info_.uplane_shuffle_count = node[kShuffleUplaneTxYaml].as<uint32_t>();
    shuffle_info_.max_tx_shift         = static_cast<int64_t>(node[kMaxTxTimestampDiffYaml].as<uint64_t>());
}

void YamlParser::parse_start_slot_info(yaml::node node)
{
    if(node.has_key(kStartFrameIdYaml)) start_slot_info_.frame_id = node[kStartFrameIdYaml].as<uint16_t>();
    if(node.has_key(kStartSubframeIdYaml)) start_slot_info_.subframe_id = node[kStartSubframeIdYaml].as<uint16_t>() & 0xf;
    if(node.has_key(kStartSectionIdYaml)) start_slot_info_.slot_id = node[kStartSectionIdYaml].as<uint16_t>() & 0x3f;
}

void YamlParser::parse_sfn_slot_sync_info(yaml::node node)
{
    sync_info_.enable          = node["enable"].as<uint32_t>() != 0 ? true : false;
    if(sync_info_.enable)
    {
        sync_info_.ru_server_addr  = node["ru_server_addr"].as<std::string>();
        sync_info_.du_server_addr  = node["du_server_addr"].as<std::string>();
    }
}

} // namespace fh_gen

