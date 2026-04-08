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

#include <iostream>
#include <memory>
#include <stdexcept>

// Include cuphydriver headers first to establish their symbols
#include "fh.hpp"
#include "yamlparser.hpp"

// Include aerial_fh headers after cuphydriver headers
#include "peer.hpp"
#include "nic.hpp"
#include "fronthaul.hpp"

#include "aerial-fh-driver/api.hpp"
#include <slot_command/slot_command.hpp>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_eal.h>
#include <rte_lcore.h>

#include "scf_fapi_handler.hpp"
#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"
#include "nv_phy_mac_transport.hpp"

#include <gtest/gtest.h>
#include <tuple>

#define TAG_UNIT_TB_BASE    NVLOG_TAG_BASE_DLC_TESTBENCH
#define TAG_UNIT_TB_COMMON  TAG_UNIT_TB_BASE + 1
#define TAG_UNIT_TB_DLC     TAG_UNIT_TB_BASE + 2

// Wrapper class that exposes protected methods from scf_fapi_handler
class TestFapiHandler : public scf_fapi_handler {
public:
    TestFapiHandler(nv::phy_mac_transport& transport, test_mac_configs* configs, launch_pattern* lp, ch8_conformance_test_stats* stats)
        : scf_fapi_handler(transport, configs, lp, stats) {}
    
    // Expose protected methods as public
    std::vector<fapi_req_t*>& get_fapi_req_list_public(int cell_id, sfn_slot_t ss, fapi_group_t group_id) {
        return get_fapi_req_list(cell_id, ss, group_id);
    }

    int build_dl_tti_request_public(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_tti_req_t& req) {
        return build_dl_tti_request(cell_id, fapi_reqs, req); 
    }
    
    int build_ul_tti_request_public(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_tti_req_t& req) {
        return build_ul_tti_request(cell_id, fapi_reqs, req); 
    }

    int build_ul_dci_request_public(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_dci_t& req) {
        return build_ul_dci_request(cell_id, fapi_reqs, req); 
    }
    
};

// Forward declarations
class PhyDriverCtx;

namespace aerial_fh
{

// Global configuration struct for command line arguments
struct TestConfig {
    std::string pattern_number{};     // Default pattern number
    bool enable_pcap{false};    // Default: pcap disabled
    bool enable_benchmark{false};    // Default: benchmarking disabled (in testing mode)
    bool verify_cplane{true};   // Default: RUE verification of cplane packets
    bool use_perf_profiler{false};   // Default: Disable perf-profiling while running benchmarks
    std::string pcap_file_name{"cplane_packets.pcap"};   // Default: RUE verification of cplane packets
    static TestConfig& instance() {
        static TestConfig config;
        return config;
    }
    bool is_nrsim() const {
        try {
            int num = std::stoi(pattern_number);
            return (num >= 90000 && num < 100000);
        } catch (...) {
            return false;
        }
    }
};

struct LocalFapiMessage {
    static constexpr size_t FAPI_PAYLOAD_BUFFER_SIZE = 64 * 1024;
    scf_fapi_dl_tti_req_t* dl_tti_req; 
    scf_fapi_ul_dci_t* ul_dci_req;
    // scf_fapi_ul_tti_req_t* ul_tti_req; 

    LocalFapiMessage() 
    {
        dl_tti_req = static_cast<scf_fapi_dl_tti_req_t*>(std::malloc(sizeof(scf_fapi_dl_tti_req_t) + FAPI_PAYLOAD_BUFFER_SIZE));
        dl_tti_req->num_pdus = 0;
        ul_dci_req = static_cast<scf_fapi_ul_dci_t*>(std::malloc(sizeof(scf_fapi_ul_dci_t) + FAPI_PAYLOAD_BUFFER_SIZE));
        ul_dci_req->num_pdus = 0; 
        // ul_tti_req = new (8*1024); 
    }

    void Reset() {
        if (dl_tti_req) {
            dl_tti_req->num_pdus = 0; 
        }
        if (ul_dci_req) {
            ul_dci_req->num_pdus = 0; 
        }
    }
    
    ~LocalFapiMessage() 
    {
        if (dl_tti_req) {
            std::free(dl_tti_req); 
            dl_tti_req = nullptr; 
        }
        if (ul_dci_req) {
            std::free(ul_dci_req);
            ul_dci_req = nullptr; 
        }
    }
    
    void print_dl_tti_req_params() {

        int offset = 0; 

        for (int i = 0; i < dl_tti_req->num_pdus; ++i) {
            auto& pdu    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(&dl_tti_req->payload[0 + offset]));

            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "DL_TTI_REQ SFN,Slot: {}.{} "
               "numPDUS: {} "
               "PDUType: {} "
               "PDUSize: {} ",
               (int) dl_tti_req->sfn, (int)dl_tti_req->slot,
               dl_tti_req->num_pdus,
               (int) pdu.pdu_type,
               (int) pdu.pdu_size);

            offset += pdu.pdu_size;
        }

        for (int i = 0; i < ul_dci_req->num_pdus; ++i) {
            auto& pdu    = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(&dl_tti_req->payload[0 + offset]));

            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "UL_DCI_REQ SFN,Slot: {}.{} "
               "numPDUS: {} "
               "PDUType: {} "
               "PDUSize: {} ",
               (int) ul_dci_req->sfn, (int)ul_dci_req->slot,
               ul_dci_req->num_pdus,
               (int) pdu.pdu_type,
               (int) pdu.pdu_size);

            offset += pdu.pdu_size;
        }
        
    }
};

class SendCPlaneUnitTest {

public:
    SendCPlaneUnitTest ();
    ~SendCPlaneUnitTest(); 
    // std::shared_ptr<FhProxy> fh_proxy_; 

    void Setup(int); 
    void print_ctx_cfg (); 
    void parse_eAxC(yaml::node node, std::vector<uint16_t> &vec); 
    void Setup_TestMAC_Integration(); 
    void Setup_MockTransport();
    void Setup_OneTimeConfigs(); 
    void TearDown(); 
    void Run(slot_command_api::slot_indication &slot_info); 
    void Reset() { 
        cell_group_.fh_params.total_num_pdsch_pdus = 0;
        cell_group_.reset(); 
        cell_cmd_.reset();
        singleton_.Reset(); 
    }

    [[nodiscard]] std::tuple<bool,bool> GenerateFapiMessagesForSlot(uint16_t sfn, uint16_t slot);
    void TestUpdateCellCommand(slot_command_api::slot_indication &slot_info); 
    void TestSendCPlaneComplete(slot_command_api::slot_indication &slot_ind);

    slot_command_api::cell_sub_command &get_cell_cmd() { return cell_cmd_; }
    slot_command_api::cell_group_command &get_cell_group_cmd() { return cell_group_; }
    
    uint16_t get_cell_dl_comp_meth(int cell_index) { return static_cast<uint16_t>(mplane_configs.at(cell_index).dl_comp_meth); }
    const context_config &get_ctx_cfg () { return ctx_cfg_; }

private:
    std::string ctx_config_file_;
    YamlParser yaml_parser_;
    context_config ctx_cfg_{};
    std::vector<::cell_mplane_info> mplane_configs{};

    // RU Emulator instance for test functionality (using opaque pointer to avoid header conflicts)
    void* ru_emulator_;
        
    // Create cell_group_command and cell_sub_command objects (use slot_command_api namespace)
    slot_command_api::cell_group_command cell_group_{};
    slot_command_api::cell_sub_command cell_cmd_{};

    aerial_fh::FronthaulHandle fronthaul;

    aerial_fh::NicInfo nic_info{};
    aerial_fh::NicHandle nic{};
    aerial_fh::PeerHandle peer{};
    aerial_fh::PeerInfo peer_info{};
    peer_id_t  peer_id_{}; 

    std::unique_ptr<PhyDriverCtx> ctx_; 
    std::shared_ptr<TestFapiHandler> fapi_handler_;
    launch_pattern* launch_pattern_;
    std::shared_ptr<test_mac_configs> testmac_configs_;
    std::unique_ptr<nv::phy_mac_transport> mock_transport_; 
    uint32_t nDLAbsFrePointA_{}, nULAbsFrePointA_{};
    LocalFapiMessage singleton_; 
}; // class SendCPlaneUnitTest

// Inherits from Peer to allow for intercepting the FH C-Plane flow right before the NIC send. 
class MockPeer : public Peer {
public:
    MockPeer(Nic* nic, PeerInfo const* info, std::vector<uint16_t>& eAxC_list_ul, std::vector<uint16_t>& eAxC_list_srs, std::vector<uint16_t>& eAxC_list_dl)
        : Peer(nic, info, eAxC_list_ul, eAxC_list_srs, eAxC_list_dl) {}
    
    // Override the key methods that send_cplane_mmimo calls to intercept populated MBUFs
    void send_cplane_packets_dl(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts) override;
    void send_cplane_packets_ul(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts) override;
    void send_cplane_enqueue_nic(Txq *txq, rte_mbuf* mbufs[], size_t num_packets); 
   
    static void add_mock_peer(void* nic, PeerInfo const* info, void** output_handle, std::vector<uint16_t>& eAxC_list_ul, std::vector<uint16_t>& eAxC_list_srs, std::vector<uint16_t>& eAxC_list_dl); 
    
private:
    void capture_packets(rte_mbuf** mbufs, size_t num_packets, bool dump_pcap);
};

// This summarizes the test result using the nvlog framework.
class SummaryLogger : public ::testing::EmptyTestEventListener {
public:
    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
        int num_tests = unit_test.total_test_count();
        int passed_tests = unit_test.successful_test_count();
        int failed_tests = unit_test.failed_test_count();
        
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "SendCPlaneUnitTest test result.. \n#Total Tests:{} \n#Passed Tests:{} \n#Failed Tests:{}",
                   num_tests, passed_tests, failed_tests);
    }
};



// Dynamic test and benchmark registration functions
// Call these from main() after CLI parsing to ensure pattern_number is available
void RegisterDynamicBenchmarks();

}  // namespace aerial_fh

