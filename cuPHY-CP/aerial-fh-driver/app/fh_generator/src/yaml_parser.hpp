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

#ifndef FH_GENERATOR_YAML_PARSER_HPP__
#define FH_GENERATOR_YAML_PARSER_HPP__

#include <vector>

#include "aerial-fh-driver/api.hpp"
#include "yaml.hpp"

namespace fh_gen
{
class FhGenerator;

struct PeerInfo
{
    uint8_t             index;
    aerial_fh::PeerInfo info;
    std::string         nic_name;
    int                 nic_id;
    aerial_fh::Ns       tcp_adv_dl_ns;
    aerial_fh::Ns       t1a_max_cp_ul_ns;
    aerial_fh::Ns       t1a_max_up_ns;
    aerial_fh::Ns       ta4_min_ns;
    aerial_fh::Ns       ta4_max_ns;
    aerial_fh::Ns       window_end_ns;
};

using MemId  = uint32_t;
using CpuId  = uint32_t;
using FlowId = uint32_t;

struct IqDataBufferInfo
{
    aerial_fh::MemRegInfo info;
    MemId                 id;
    int32_t               cuda_device_id{-1};
    std::string           input_file{};
};



struct CPlaneSection
{
    uint8_t start_sym;
    uint8_t num_sym;
    uint16_t start_prbc;
    uint16_t num_prbc;
};

using CPlaneSectionList = std::vector<CPlaneSection>;

struct CPlaneTxInfo
{
    uint8_t           slot_id;
    uint8_t           symbol_id;
    uint16_t          section_count;
    std::string       direction;
    CPlaneSectionList section_list;
};

struct UPlaneSection
{
    uint16_t start_prbu;
    uint16_t num_prbu;
};

using UPlaneSectionList = std::vector<UPlaneSection>;

struct UPlaneInfo
{
    uint8_t  slot_id;
    uint8_t  symbol_id;
    MemId    iq_data_buffer;
    uint16_t section_id;
    std::string direction;
    UPlaneSectionList section_list;
};

struct FlowInfo
{
    aerial_fh::FlowInfo       info;
    aerial_fh::PeerId         cell_id;
    std::vector<CPlaneTxInfo> cplane_tx_info;
    std::vector<UPlaneInfo>   uplane_tx_info;
    std::vector<UPlaneInfo>   uplane_rx_info;
};

struct ShuffleInfo
{
    int32_t  random_seed{0};
    uint32_t cplane_shuffle_count{0};
    uint32_t uplane_shuffle_count{0};
    int64_t  max_tx_shift{0};
};

struct StartSlotInfo
{
    uint8_t frame_id{0};
    uint8_t subframe_id{0};
    uint8_t slot_id{0};
};

struct SfnSlotSyncInfo
{
    bool enable;
    std::string du_server_addr;
    std::string ru_server_addr;
};

class YamlParser {
public:
    YamlParser(FhGenerator* fh_generator, const std::string& config_file);

    const aerial_fh::FronthaulInfo&              get_fh_info() const;
    const std::vector<aerial_fh::NicInfo>&       get_nic_info() const;
    const std::vector<fh_gen::IqDataBufferInfo>& get_iq_data_buffer_info() const;
    const std::vector<fh_gen::PeerInfo>&         get_peer_info() const;
    const std::vector<fh_gen::FlowInfo>&         get_flow_info() const;
    const std::vector<CpuId>&                    get_cpus() const;
    const std::vector<CpuId>&                    get_ul_cpus() const;
    const std::vector<CpuId>&                    get_dl_cpus() const;
    uint32_t                                     get_startup_delay() const;
    int32_t                                      get_workers_priority() const;
    bool                                         send_utc_anchor() const;
    ShuffleInfo                                  get_shuffle_info() const;
    bool                                         validate_iq_data_buffer_size() const;
    bool                                         enable_ulu() const;
    bool                                         enable_dlu() const;
    bool                                         enable_c_plane() const;
    bool                                         enable_ulc() const;
    bool                                         enable_dlc() const;
    StartSlotInfo                                get_start_slot_info() const;
    aerial_fh::Ns                                slot_duration() const;
    uint16_t                                     slot_count() const;
    uint64_t                                     test_slots() const;
    SfnSlotSyncInfo                              get_sync_info() const;
    aerial_fh::Ns                                dl_u_enq_time_advance_ns() const;
    aerial_fh::Ns                                dl_c_enq_time_advance_ns() const;
    aerial_fh::Ns                                ul_u_tx_time_advance_ns() const;
    aerial_fh::Ns                                ul_u_enq_time_advance_ns() const;
    float                                ulc_ontime_pass_percentage() const;
    float                                ulu_ontime_pass_percentage() const;
    float                                dlc_ontime_pass_percentage() const;
    float                                dlu_ontime_pass_percentage() const;

protected:
    FhGenerator* fh_gen_{};

    // Config values
    aerial_fh::FronthaulInfo              fh_info_;
    std::vector<aerial_fh::NicInfo>       nic_info_;
    std::vector<fh_gen::IqDataBufferInfo> iq_data_buffer_info_;
    std::vector<fh_gen::PeerInfo>         peer_info_;
    std::vector<fh_gen::FlowInfo>         flow_info_;
    std::vector<CpuId>                    cpus_;
    std::vector<CpuId>                    ru_cpus_;
    std::vector<CpuId>                    ul_cpus_;
    std::vector<CpuId>                    dl_cpus_;
    SfnSlotSyncInfo                       sync_info_;

    uint32_t      startup_delay_{1};
    int32_t       workers_priority_{30};
    bool          send_utc_anchor_{true};
    ShuffleInfo   shuffle_info_;
    bool          validate_iq_data_buffer_size_{false};
    StartSlotInfo start_slot_info_;
    aerial_fh::Ns slot_duration_;
    uint16_t       slot_count_;
    bool          enable_ulu_{true};
    bool          enable_dlu_{true};
    bool          enable_ulc_{true};
    bool          enable_dlc_{true};
    uint64_t      test_slots_{0};
    aerial_fh::Ns       dl_u_enq_time_advance_ns_;
    aerial_fh::Ns       dl_c_enq_time_advance_ns_;
    aerial_fh::Ns       ul_u_tx_time_advance_ns_;
    aerial_fh::Ns       ul_u_enq_time_advance_ns_;
    float       ulc_ontime_pass_percentage_;
    float       ulu_ontime_pass_percentage_;
    float       dlc_ontime_pass_percentage_;
    float       dlu_ontime_pass_percentage_;
    void parse_fh_info(yaml::node node);
    void parse_nic_info(yaml::node nic_list);
    void parse_iq_data_buffer_info(yaml::node iq_data_buffer_list);
    void parse_peer_info(yaml::node peer_list);
    void parse_flow_info(yaml::node flow_list);
    void parse_cplane_tx_info(yaml::node cplane_tx_list, std::vector<CPlaneTxInfo>& cplane_tx_info);
    void parse_uplane_info(yaml::node uplane_tx_list, std::vector<UPlaneInfo>& uplane_tx_info);
    void parse_cpu_info(yaml::node cpu_list);
    void parse_ru_cpu_info(yaml::node cpu_list);
    void parse_shuffle_info(yaml::node node);
    void parse_start_slot_info(yaml::node node);
    void parse_sfn_slot_sync_info(yaml::node node);
};

} // namespace fh_gen

#endif // #ifndef FH_GENERATOR_YAML_PARSER_HPP__
