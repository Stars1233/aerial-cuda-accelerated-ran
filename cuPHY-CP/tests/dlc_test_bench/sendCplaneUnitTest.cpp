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

#include <iterator>
#include <fstream>
#include <cstdio>  // for std::remove
#include <unordered_map>  // for std::unordered_map
#include <chrono>  // for timing stub
#include "yamlparser.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <rte_eal.h>
#include "test_utils.hpp"

// Include context.hpp first to establish global Time class
#include "context.hpp"
#include "time.hpp"

// Now include our header with aerial_fh types
#include "sendCplaneUnitTest.hpp"
#include "pcap_writer.h"

// Undefine TAG from gpudevice.hpp before including nv_phy_module.hpp to avoid redefinition
#ifdef TAG
#undef TAG
#endif
#include "nv_phy_module.hpp"  // For nv::PHY_module class
#include "cuphy_api.h"        // For cuphyPmW_t type and cuphyCellStatPrm_t
#include "cuda_fp16.h"        // For __half2 and half types

// Required includes for update_cell_command API
#include "scf_5g_fapi.h"                   // For scf_fapi_pdcch_pdu_t
#include "scf_5g_fh_callback_context.hpp"  // For scf_5g_fapi::LegacyCellGroupFhContext
#include "slot_command/slot_command.hpp"    // For cell_group_command, cell_sub_command
#include "nv_phy_fapi_msg_common.hpp"       // For slot_indication
#include "nv_phy_config_option.hpp"         // For nv::phy_config_option
#include "nv_phy_limit_errors.hpp"         // For nv::pdcch_limit_error_t

// Use opaque wrapper to avoid type conflicts with RU emulator headers
#include "ru_emulator_wrapper.hpp"

// Google Benchmark for performance measurement
#include <benchmark/benchmark.h>

// Forward declaration to avoid namespace conflicts from including scf_5g_slot_commands_pdcch.hpp
namespace scf_5g_fapi {
    // Forward declare the types we need to avoid conflicts
    using pm_weight_map_t = std::unordered_map<uint32_t, slot_command_api::pm_weights_t>;

    // Extern declaration of the update_cell_command function with correct signature
#ifdef ENABLE_L2_SLT_RSP
void update_cell_command(slot_command_api::cell_group_command* cell_group, slot_command_api::cell_sub_command& cell_cmd,
    scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, int32_t cell_index, slot_command_api::slot_indication& slotinfo,
    cuphyCellStatPrm_t& cell_params, int staticPdcchSlotNum, nv::phy_config_option& config_option,
    pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled, nv::pdcch_limit_error_t* pdcch_error);
#else
    void update_cell_command(slot_command_api::cell_group_command* cell_group, slot_command_api::cell_sub_command& cell_cmd,
                            scf_fapi_pdcch_pdu_t& msg, uint8_t testMode, int32_t cell_index, slot_command_api::slot_indication& slotinfo,
                            cuphyCellStatPrm_t& cell_params, int staticPdcchSlotNum, nv::phy_config_option& config_option,
                            pm_weight_map_t& pm_map, nv::slot_detail_t* slot_detail, bool mmimo_enabled);
#endif
    void update_cell_command(slot_command_api::cell_group_command* cell_grp_cmd,cell_sub_command& cell_cmd,
                             scf_fapi_ssb_pdu_t& cmd, int32_t cell_index, slot_command_api::slot_indication& slotinfo,
                             nv::phy_config& cell_params, uint8_t l_max, const uint16_t* lmax_symbols,
                             nv::phy_config_option& config_option, pm_weight_map_t& pm_map, nv::slot_detail_t*  slot_detail, bool mmimo_enabled);

    void update_cell_command(cell_group_command* cell_grp_cmd, cell_sub_command& cell_sub_cmd,
                             const scf_fapi_pdsch_pdu_t& msg, uint8_t testMode, slot_command_api::slot_indication& slotinfo,
                             int32_t cell_index, pm_weight_map_t& pm_map, bool pm_enabled, bool bf_enabled, uint16_t num_dl_prb,
                             bfw_coeff_mem_info_t *bfwCoeff_mem_info, bool mmimo_enabled, nv::slot_detail_t*  slot_detail);

    void update_cell_command(slot_command_api::cell_group_command* cell_grp_cmd, cell_sub_command& cell_cmd,
                             const scf_fapi_csi_rsi_pdu_t& msg, slot_command_api::slot_indication & slotinfo, int32_t cell_index,
                             cuphyCellStatPrm_t cell_params, nv::phy_config_option& config_option,
                             pm_weight_map_t& pm_map,uint32_t csirs_offset, bool pdsch_exist,
                             uint16_t cell_stat_prm_idx, bool mmimo_enabled, nv::slot_detail_t*  slot_detail);

    template <bool mplane_configured_ru_type, bool config_options_precoding_enabled, bool config_options_bf_enabled>
    void fh_callback(IFhCallbackContext& fh_context, nv::slot_detail_t* slot_detail);
}

namespace aerial_fh {


static bfw_buffer_info_header bfw_header[MAX_CELLS_PER_SLOT]{};
static bfw_coeff_mem_info_t bfwCoeff_mem_info[MAX_CELLS_PER_SLOT]{};

// Missing global variables
FhProxy* fh_proxy_;  // Global for TestSendCPlaneComplete()
peer_id_t  peer_id_g; // Global for TestSendCPlaneComplete


constexpr size_t L3_CACHE_SIZE = 116736 * 1024; // in bytes
constexpr size_t BUF_SIZE = L3_CACHE_SIZE * 4;  // 4x L3 size, ~456 MB
constexpr size_t CACHELINE = 64;                // Typical cache line size

std::vector<char> buf(BUF_SIZE);

// static void thrash_cache(int k) {
//     for (size_t offset = 0; offset < BUF_SIZE; offset += CACHELINE) {
//         buf[offset] = k++;
//     }
// }

void MockPeer::add_mock_peer(void* nic, PeerInfo const* info, void** output_handle, std::vector<uint16_t>& eAxC_list_ul, std::vector<uint16_t>& eAxC_list_srs, std::vector<uint16_t>& eAxC_list_dl)
{
    auto nic_       = static_cast<Nic*>(nic);
    *output_handle = new MockPeer(nic_, info,eAxC_list_ul,eAxC_list_srs,eAxC_list_dl);
}

void *ru_emulator_global;

SendCPlaneUnitTest::SendCPlaneUnitTest()
    : ru_emulator_(nullptr)
{
    // Initialize all members to safe defaults
    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest constructor called..");

    // Initialize RU emulator using wrapper to avoid header conflicts
    ru_emulator_ = create_ru_emulator();
    ru_emulator_global = ru_emulator_;
    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: RU_Emulator initialized..");
}

SendCPlaneUnitTest::~SendCPlaneUnitTest()
{
    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest destructor called..");

    // Clean up RU emulator using wrapper
    if (ru_emulator_) {
        destroy_ru_emulator(ru_emulator_);
        ru_emulator_ = nullptr;
        ru_emulator_global = nullptr;
    }

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest destructor completed..");
}

static void Setup_bfwMemInfo(int cell_index) {
    /*  Total Buffer is of size: maxCplaneProcSlots (4) * nUeG(8) x nUeLayers(16) * nPrbGrpBfw(273) * nRxAnt(64)  * (2 *sizeof(uint32_t))
               4 x 8 x 16 x 273 x 64 x 8 bytes = 71565312 bytes = ~ 72MB + 4 * 1 bytes(Header Metadata per buffer chunk) */
    uint32_t bfwCoffBuffChunkSize  =  (MAX_DL_UL_BF_UE_GROUPS * MAX_MU_MIMO_LAYERS * MAX_NUM_PRGS_DBF *
        NUM_GNB_TX_RX_ANT_PORTS * IQ_REPR_FP32_COMPLEX * sizeof(uint32_t));
    uint32_t bfwCoffBuffUegSize =  (MAX_MU_MIMO_LAYERS * MAX_NUM_PRGS_DBF * NUM_GNB_TX_RX_ANT_PORTS *
        IQ_REPR_FP32_COMPLEX * sizeof(uint32_t));
    uint8_t uegIdx = 0;
    uint8_t slotIdx = 0;

    bfw_header[cell_index].state[0] = BFW_COFF_MEM_BUSY;
    bfwCoeff_mem_info[cell_index].slotIndex = 0;
    bfwCoeff_mem_info[cell_index].sfn = 0;
    bfwCoeff_mem_info[cell_index].slot = 0;
    bfwCoeff_mem_info[cell_index].nGnbAnt = NUM_GNB_TX_RX_ANT_PORTS;
    bfwCoeff_mem_info[cell_index].header_size = 1;
    bfwCoeff_mem_info[cell_index].header = &bfw_header[cell_index].state[0 /* always use buffer-0 in this test */];
    bfwCoeff_mem_info[cell_index].buff_size = bfwCoffBuffChunkSize;
    bfwCoeff_mem_info[cell_index].buff_chunk_size = bfwCoffBuffUegSize;
    bfwCoeff_mem_info[cell_index].num_buff_chunk_busy = 0;
}


void SendCPlaneUnitTest::Setup_OneTimeConfigs()
{

    const int cell_idx = 0; // TODO
    auto matrix_vec = launch_pattern_->get_precoding_matrix_v(cell_idx);

    auto& pm_weight_map = nv::PHY_module::pm_map();


    for (const auto& matrix : matrix_vec) {
        const auto pmidx = matrix.PMidx;
        const auto nlayers = matrix.numLayers;
        const auto nports = matrix.numAntPorts;

        // Insert new entry in the map
        pm_weight_map.insert(std::make_pair(pmidx, pm_weights_t{nlayers, nports, cuphyPmW_t()}));
        auto& weights = pm_weight_map[pmidx];
        weights.weights.nPorts = nports;

        // Process precoder weights with proper indexing
        std::size_t index = 0;
        for(const complex_int16_t& precoderWeight : matrix.precoderWeight_v)
        {
            if (index < (MAX_DL_LAYERS_PER_TB * MAX_DL_PORTS)) {
                // Convert int16 to half using reinterpret_cast like the working example
                const half re = *reinterpret_cast<const half*>(&precoderWeight.re);
                const half im = *reinterpret_cast<const half*>(&precoderWeight.im);
                weights.weights.matrix[index] = __half2(re, im);
                index++;
            }
        }
    }
}

void SendCPlaneUnitTest::parse_eAxC(yaml::node node, std::vector<uint16_t> &vec)
{
    for(int i = 0; i < node.length(); ++i)
    {
        auto eAxC_id = static_cast<int>(node[i]);
        vec.push_back(eAxC_id);
    }
}

void SendCPlaneUnitTest::print_ctx_cfg() {
    NVLOGC_FMT(TAG_UNIT_TB_COMMON,
               "ctx_cfg_.enable_gpu_comm_via_cpu: {}\n"
               "ctx_cfg_.static_beam_id_start: {}\n"
               "ctx_cfg_.static_beam_id_end: {}\n"
               "ctx_cfg_.dynamic_beam_id_start: {}\n"
               "ctx_cfg_.dynamic_beam_id_end: {}\n"
               "ctx_cfg_.bfw_c_plane_chaining_mode: {}\n"
               "ctx_cfg_.dlc_bfw_enable_divide_per_cell: {}\n"
               "ctx_cfg_.ulc_bfw_enable_divide_per_cell: {}\n"
               "ctx_cfg_.send_static_bfw_wt_all_cplane: {}\n",
               ctx_cfg_.enable_gpu_comm_via_cpu,
               ctx_cfg_.static_beam_id_start,
               ctx_cfg_.static_beam_id_end,
               ctx_cfg_.dynamic_beam_id_start,
               ctx_cfg_.dynamic_beam_id_end,
               ctx_cfg_.bfw_c_plane_chaining_mode,
               ctx_cfg_.dlc_bfw_enable_divide_per_cell,
               ctx_cfg_.ulc_bfw_enable_divide_per_cell,
               ctx_cfg_.send_static_bfw_wt_all_cplane);
}

void SendCPlaneUnitTest::Setup(int cell_index)
{
    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Setup called..");
    try {
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Setup Creating nic_cfg structure...");
        struct nic_cfg niccfg;
        niccfg.nic_bus_addr = "0000:01:00.0";
        niccfg.nic_mtu = 8192;
        niccfg.cpu_mbuf_num = 196608;
        niccfg.tx_req_num = 64;
        niccfg.txq_count_uplane = 4;
        niccfg.txq_count_cplane = 4;
        niccfg.rxq_count = 2;
        niccfg.txq_size = 16384;
        niccfg.rxq_size = 16384;

        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Creating context_config structure...");

        // Read from the Cuphycontroller config file
        std::string cuphy_yaml_name;
        if (TestConfig::instance().is_nrsim()) {
            cuphy_yaml_name = "cuphycontroller_nrSim_SCF_CG1_" + TestConfig::instance().pattern_number + ".yaml";
        } else {
            cuphy_yaml_name = "cuphycontroller_F08_CG1.yaml";
        }
        char temp_config_file[MAX_PATH_LEN];
        get_full_path_file(temp_config_file, CONFIG_YAML_FILE_PATH, cuphy_yaml_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

        // Parse YAML config
        yaml::file_parser parser(temp_config_file);
        yaml::document doc = parser.next_document();
        yaml::node config_node = doc.root();

        // NOTE: Setting gpu_id to 0 but will be overridden by CPU-only flags in FhProxy
        ctx_cfg_.gpu_id = 0;  // Use valid GPU ID, but CPU-only mode will override this
        ctx_cfg_.standalone = 1;
        ctx_cfg_.validation = 1;  // Enable validation mode for safer testing
        ctx_cfg_.fh_cpu_core = 0;
        ctx_cfg_.prometheus_cpu_core = 2;
        ctx_cfg_.data_core = -1;
        ctx_cfg_.mMIMO_enable = (config_node["cuphydriver_config"]["mMIMO_enable"].as<uint>() > 0);

        // DPDK configuration for testing
        ctx_cfg_.dpdk_verbose_logs = false;
        ctx_cfg_.dpdk_file_prefix = "testdlc";
        ctx_cfg_.accu_tx_sched_disable = true;  // Disable scheduling for simpler testing
        ctx_cfg_.accu_tx_sched_res_ns = 0;
        ctx_cfg_.pdump_client_thread = -1;      // Disable pdump
        ctx_cfg_.fh_stats_dump_cpu_core = -1;   // Disable stats dump
        ctx_cfg_.enable_ul_cuphy_graphs = false;
        ctx_cfg_.enable_dl_cuphy_graphs = false;
        ctx_cfg_.enable_cpu_init_comms = 0;
        ctx_cfg_.enable_gpu_comm_dl = 1;
        ctx_cfg_.nic_configs.emplace_back(niccfg);

        // Initialize additional fields that FhProxy constructor may use
        ctx_cfg_.enable_gpu_comm_via_cpu = false;  // CPU-only mode
        ctx_cfg_.static_beam_id_start = config_node["cuphydriver_config"]["static_beam_id_start"].as<uint>();
        ctx_cfg_.static_beam_id_end = config_node["cuphydriver_config"]["static_beam_id_end"].as<uint>();
        ctx_cfg_.dynamic_beam_id_start = config_node["cuphydriver_config"]["dynamic_beam_id_start"].as<uint>();
        ctx_cfg_.dynamic_beam_id_end = config_node["cuphydriver_config"]["dynamic_beam_id_end"].as<uint>();
        ctx_cfg_.bfw_c_plane_chaining_mode = 0; // TODO Support chaining mode.
        ctx_cfg_.dlc_bfw_enable_divide_per_cell = false;
        ctx_cfg_.ulc_bfw_enable_divide_per_cell = false;
        ctx_cfg_.send_static_bfw_wt_all_cplane = true;
        ctx_cfg_.enable_weighted_average_cfo = false;

        print_ctx_cfg ();

        yaml::node yaml_cell = config_node["cuphydriver_config"]["cells"][cell_index];
        cell_mplane_info mplane_cfg{};
        mplane_cfg.dl_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(static_cast<int>(yaml_cell["dl_iq_data_fmt"]["comp_meth"]));
        mplane_cfg.dl_bit_width = static_cast<int>(yaml_cell["dl_iq_data_fmt"]["bit_width"]);
        mplane_cfg.ru = MULTI_SECT_MODE;

        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Creating Minimal PhydriverContext instance...");
        bool minimal = true;
        cudaFree(0);
        cudaSetDevice(ctx_cfg_.gpu_id);        
        // This is a minimally constructed phydriver object that is setup for FhProxy creation.
        ctx_ = std::make_unique<PhyDriverCtx>(ctx_cfg_, minimal);
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest PhyDriverCtx created successfully!");
        fh_proxy_ = ctx_->getFhProxy();

        // TODO hardcoded mac address to be fixed
        peer_info.dst_mac_addr.bytes[0]       = 0x20;
        peer_info.dst_mac_addr.bytes[1]       = 0x4;
        peer_info.dst_mac_addr.bytes[2]       = 0x9B;
        peer_info.dst_mac_addr.bytes[3]       = 0x9E;
        peer_info.dst_mac_addr.bytes[4]       = 0x27;
        peer_info.dst_mac_addr.bytes[5]       = 0x00;

        peer_info.id = 0;
        peer_info.max_num_prbs_per_symbol = 273; // TODO From config files
        peer_info.txq_count_uplane = 0;
        peer_info.txq_count_uplane_gpu = 4;
        peer_info.rx_mode = aerial_fh::RxApiMode::TXONLY;
        peer_info.vlan.tci = 0;
        peer_info.share_txqs = false;
        peer_info.txq_cplane = true;
        peer_info.mMIMO_enable = ctx_cfg_.mMIMO_enable;

        peer_info.enable_srs = false;
        peer_info.ud_comp_info.iq_sample_size = mplane_cfg.dl_bit_width;
        peer_info.ud_comp_info.method = mplane_cfg.dl_comp_meth;
        peer_info.txq_bfw_cplane = true;
        peer_info.bfw_cplane_info.bfw_chain_mode = aerial_fh::BfwCplaneChainingMode::NO_CHAINING; // TODO from config files

        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Adding mock peer...");

        parse_eAxC(yaml_cell["eAxC_id_ssb_pbch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PBCH]);
        parse_eAxC(yaml_cell["eAxC_id_pdcch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PDCCH_DL]);
        parse_eAxC(yaml_cell["eAxC_id_pdcch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PDCCH_UL]);
        parse_eAxC(yaml_cell["eAxC_id_pdsch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PDSCH]);
        parse_eAxC(yaml_cell["eAxC_id_pdsch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PDSCH_CSIRS]);
        parse_eAxC(yaml_cell["eAxC_id_csirs"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::CSI_RS]);
        parse_eAxC(yaml_cell["eAxC_id_pusch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PUSCH]);
        parse_eAxC(yaml_cell["eAxC_id_pucch"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PUCCH]);
        parse_eAxC(yaml_cell["eAxC_id_srs"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::SRS]);
        parse_eAxC(yaml_cell["eAxC_id_prach"], mplane_cfg.eAxC_ids[slot_command_api::channel_type::PRACH]);

        std::vector<uint16_t> eAxC_list_uplink{};
        for (int channel = slot_command_api::channel_type::PUSCH; channel < slot_command_api::channel_type::SRS; channel++)
        {
            for (auto eAxC : mplane_cfg.eAxC_ids[channel])
            {
                auto it = std::find(std::begin(eAxC_list_uplink), std::end(eAxC_list_uplink), (uint16_t) eAxC);
                if(it!=std::end(eAxC_list_uplink))
                    continue;
                eAxC_list_uplink.push_back(eAxC);
            }
        }

        // Why a mock peer? Because we're overriding methods of the Peer class to intercept the generated packtes.
        MockPeer::add_mock_peer(fh_proxy_->getNic(niccfg.nic_bus_addr), &peer_info, &peer,
                                eAxC_list_uplink,
                                mplane_cfg.eAxC_ids[slot_command_api::channel_type::SRS],
                                mplane_cfg.eAxC_ids[slot_command_api::channel_type::PDSCH]);

        // TODO support multi-cell.
        mplane_configs.push_back(mplane_cfg);
        peer_id_ = ::Time::nowNs().count();
        peer_id_g = peer_id_;
        fh_proxy_->updatePeerMap(peer_id_,
                                 std::move(std::unique_ptr<fh_peer_t>(new fh_peer_t(peer_id_, peer, peer_info, mplane_configs[cell_index].dl_comp_meth, mplane_configs[cell_index].dl_bit_width))));

        // Register flows of all the channels.
        for(int channel = slot_command_api::channel_type::PDSCH_CSIRS; channel < slot_command_api::channel_type::CHANNEL_MAX; channel++)
        {
            for(auto eAxC : mplane_cfg.eAxC_ids[channel])
            {
                fh_proxy_->registerFlow(peer_id_, eAxC, 0 /* vlan_tci */, static_cast<slot_command_api::channel_type>(channel));
            }
        }


        Setup_MockTransport();
        Setup_TestMAC_Integration();
        Setup_OneTimeConfigs();
        Setup_bfwMemInfo(0);

        // Use pattern number from command line arguments
        if (TestConfig::instance().is_nrsim()) {
            std::string ru_config = "ru_emulator_config_nrSim_SCF_CG1_" + TestConfig::instance().pattern_number + ".yaml";
            const char *argv[5] = {"ru_emulator", "nrSim", TestConfig::instance().pattern_number.c_str(), "--config", ru_config.c_str()};
            ru_emulator_init(ru_emulator_, 5, (char **)argv);
        } else {
            const char *argv[6] = {"ru_emulator", "F08", "1C", TestConfig::instance().pattern_number.c_str(), "--config", "config.yaml"};
            ru_emulator_init(ru_emulator_, 6, (char **)argv);
        }
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest RU Emulator instance initialized with pattern {}...",
                   TestConfig::instance().pattern_number);

        // Load STATIC BFW.
        auto* dbt = launch_pattern_->get_dbt_info(0);

        if (dbt) {

            uint16_t numDigBeams = dbt->num_static_beamIdx;
            uint16_t numTXRUs = dbt->num_TRX_beamforming;
            std::size_t size = ((sizeof(uint16_t) * 2)+ (numDigBeams * (sizeof(uint16_t) + (sizeof(complex_int16_t) * numTXRUs))));

            void* data_buf = std::aligned_alloc(32, size);

            uint16_t* dbt_data = reinterpret_cast<uint16_t*>(data_buf);
            *dbt_data++        = numDigBeams;
            *dbt_data++        = numTXRUs;

            for(int i = 0; i < numDigBeams; i++)
            {
                *dbt_data++ = i+1;
                complex_int16_t* dbt_wt_data = reinterpret_cast<complex_int16_t*>(dbt_data);
                for(int j = 0; j < numTXRUs; j++)
                {
                    dbt_wt_data->re = dbt->dbt_data_buf[(i*numTXRUs)+j].re;
                    dbt_wt_data->im = dbt->dbt_data_buf[(i*numTXRUs)+j].im;
                    dbt_wt_data++;
                }
                dbt_data = reinterpret_cast<uint16_t*>(dbt_wt_data);
            }

            int rc = fh_proxy_->storeDBTPdu(0, data_buf);
            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest DBT setup completed numDigBeams:{} numTXRUs:{}", numDigBeams, numTXRUs);
        }

        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest setup completed...");

    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Error: SendCPlaneUnitTest Error in setup: {}", e.what());
        throw;
    }
    fmtlog::poll(true);
}

void SendCPlaneUnitTest::TearDown()
{
    if (TestConfig::instance().verify_cplane) {
        // Summary & assert checks are not done in benchmark mode.
        ru_emulator_finalize(ru_emulator_global);

        ru_emulator_total_slot_counters_t rue_slt_counters{};
        ru_emulator_get_total_slt_counters (ru_emulator_global, 0/* cell_idx */, rue_slt_counters);

        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "PDSCH {}:{}",   launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PDSCH].load(),    rue_slt_counters.pdsch);
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "PDCCH_UL {}:{}",launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PDCCH_UL].load(), rue_slt_counters.pdcch_ul);
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "PDCCH_DL {}:{}",launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PDCCH_DL].load(), rue_slt_counters.pdcch_dl);
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "PBCH {}:{}",    launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PBCH].load(),     rue_slt_counters.pbch);
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "CSI_RS {}:{}",  launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::CSI_RS].load(),   rue_slt_counters.csi_rs);
        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "BFW_DL {}:{}",  launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::BFW_DL].load(),   rue_slt_counters.bfw_dl);

        ASSERT_EQ(launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PDSCH].load(),    rue_slt_counters.pdsch);
        ASSERT_EQ(launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PDCCH_UL].load(), rue_slt_counters.pdcch_ul);
        ASSERT_EQ(launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PDCCH_DL].load(), rue_slt_counters.pdcch_dl);
        ASSERT_EQ(launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::PBCH].load(),     rue_slt_counters.pbch);
        ASSERT_EQ(launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::CSI_RS].load(),   rue_slt_counters.csi_rs);
        ASSERT_EQ(launch_pattern_->get_expected_values().at(0 /* cell */).lp_slots[channel_type_t::BFW_DL].load(),   rue_slt_counters.bfw_dl);

        uint64_t num_err_sections = 0xFFF;
        ru_emulator_get_cplane_err_sections(ru_emulator_global, 0/* cell_idx */, num_err_sections);
        ASSERT_EQ(num_err_sections, 0);

        uint64_t num_tot_sections = 0x0;
        ru_emulator_get_cplane_tot_sections(ru_emulator_global, 0/* cell_idx */, num_tot_sections);
        ASSERT_GT(num_tot_sections, 0);
    }

    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test cleanup completed...");
}

void SendCPlaneUnitTest::Run(slot_command_api::slot_indication &slot_info)
{
    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test Run Start...");

    auto [run_dl_cplane, run_ul_cplane] = GenerateFapiMessagesForSlot(slot_info.sfn_,slot_info.slot_);

    if (run_dl_cplane) { // TODO support UL
        TestUpdateCellCommand(slot_info);

        TestSendCPlaneComplete(slot_info);
    } else {
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test Run skipped for SFN:{} SLT:{}...", slot_info.sfn_, slot_info.slot_);
    }

    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test Run Completed...");
}

// MockPeer overriden method implementations
void MockPeer::send_cplane_packets_dl(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts)
{

    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test DL Run Result - BFW mbufs:{} Regular mbufs:{} total packets:{} Created:{}",
           (int)mbufs_bfw.size(), (int)mbufs_regular.size(), num_packets, created_pkts);

    int num_pkts = (mbufs_regular.size() > 0) ? mbufs_regular.size() : mbufs_bfw.size();
    auto mbuf = (mbufs_regular.size() > 0) ? mbufs_regular.data() : mbufs_bfw.data();

    // Capture packets for verification (controlled by command line flag)
    const bool enable_pcap = TestConfig::instance().enable_pcap;
    capture_packets(mbuf, num_pkts, enable_pcap);

    const bool verify_cplane = TestConfig::instance().verify_cplane;
    if (verify_cplane) {
        try {
            for (int i = 0; i < num_pkts; ++i) {
                ru_emulator_verify_dl_cplane_content(ru_emulator_global, rte_pktmbuf_mtod(mbuf[i], uint8_t *), rte_pktmbuf_pkt_len(mbuf[i]), 0 /* cell_index */);
            }
        } catch (const std::exception& e) {
            NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Error in {}:{}", __func__, e.what());
            EXPECT_TRUE(false);
        }
    }

    for (int i = 0; i < num_pkts; ++i) {
        rte_pktmbuf_free(mbuf[i]);
    }
}

void MockPeer::send_cplane_packets_ul(MbufArray& mbufs_bfw, MbufArray& mbufs_regular, rte_mbuf* mbufs[], size_t num_packets, size_t created_pkts)
{

    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test UL Run Result - BFW mbufs:{} Regular mbufs:{} total packets:{} Created:{}",
           (int)mbufs_bfw.size(), (int)mbufs_regular.size(), num_packets, created_pkts);

    int num_pkts = (mbufs_regular.size() > 0) ? mbufs_regular.size() : mbufs_bfw.size();
    auto mbuf = (mbufs_regular.size() > 0) ? mbufs_regular.data() : mbufs_bfw.data();

    // Capture packets for verification (controlled by command line flag)
    const bool enable_pcap = TestConfig::instance().enable_pcap;
    capture_packets(mbuf, num_pkts, enable_pcap);

    const bool verify_cplane = TestConfig::instance().verify_cplane;
    if (verify_cplane) {
        try {
            for (int i = 0; i < num_pkts; ++i) {
                ru_emulator_verify_dl_cplane_content(ru_emulator_global, rte_pktmbuf_mtod(mbuf[i], uint8_t *), rte_pktmbuf_pkt_len(mbuf[i]), 0 /* cell_index */);
            }
        } catch (const std::exception& e) {
            NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Error in {}:{}", __func__, e.what());
            EXPECT_TRUE(false);
        }
    }

    for (int i = 0; i < num_pkts; ++i) {
        rte_pktmbuf_free(mbuf[i]);
    }
}


void MockPeer::send_cplane_enqueue_nic(Txq *txq, rte_mbuf* mbuf[], size_t num_pkts)
{
    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test Run Result - total packets:{}", num_pkts);

    // Capture packets for verification (controlled by command line flag)
    const bool enable_pcap = TestConfig::instance().enable_pcap;
    capture_packets(mbuf, num_pkts, enable_pcap);

    const bool verify_cplane = TestConfig::instance().verify_cplane;
    if (verify_cplane) {
        try {
            for (int i = 0; i < num_pkts; ++i) {
                ru_emulator_verify_dl_cplane_content(ru_emulator_global, rte_pktmbuf_mtod(mbuf[i], uint8_t *), rte_pktmbuf_pkt_len(mbuf[i]), 0 /* cell_index */);
            }
        } catch (const std::exception& e) {
            NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Error in {}:{}", __func__, e.what());
            EXPECT_TRUE(false);
        }
    }

    for (int i = 0; i < num_pkts; ++i) {
        rte_pktmbuf_free(mbuf[i]);
    }

}

void MockPeer::capture_packets(rte_mbuf** mbufs, size_t num_packets, bool dump_pcap)
{

    if (!dump_pcap || num_packets == 0) {
        return;
    }

    static bool first_time = true;
    std::string path = "/tmp/";
    std::string pcap_file_full = path + TestConfig::instance().pcap_file_name;
    const char* file_name = pcap_file_full.c_str();

    if (first_time) {
        // Create the pcap file for the first time logging (typically bfw packets)
        int rc = pcap_write_mbufs(file_name,
                                  mbufs,
                                  num_packets,
                                  false /* bool use_timestamps */);
        first_time = false;
    } else {
        // Append to the existing pcap file for the subsequent logging (typically non-bfw packets)
        int rc = pcap_append_mbufs(file_name,
                                   mbufs,
                                   num_packets,
                                   false /* bool use_timestamps */);
    }

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test {} packets written to PCAP file:{} ", num_packets, file_name);
}

/**
 * Exercises the unit under test to produce packets.
 */
void SendCPlaneUnitTest::TestSendCPlaneComplete(slot_command_api::slot_indication &slot_ind)
{

    // Verify initialization
    slot_command_api::slot_info_t &slot_info = *cell_cmd_.sym_prb_info();
    ASSERT_TRUE(slot_info.prbs_size > 0);
    ASSERT_TRUE(slot_info.section_id_ready.load() == false);
    // Create slot indication
    slot_command_api::oran_slot_ind slot_indication = to_oran_slot_format(slot_ind);

    uint8_t* bfw_header_p = nullptr;

    ti_subtask_info tis{};
    int cell_id = 0;

    int ret = fh_proxy_->prepareCPlaneInfo(
        cell_id,                                    // cell_id
        MULTI_SECT_MODE,                            // ru_type
        peer_id_g,                                  // peer_id
        static_cast<uint16_t>(mplane_configs[cell_id].dl_comp_meth),      // dl_comp_meth
        t_ns(::Time::nowNs().count() + 1000000UL),  // start_tx_time (1ms in future)
        0,                                          // tx_cell_start_ofs_ns
        DIRECTION_DOWNLINK,                         // direction
        slot_indication,                            // slot_indication
        slot_info,                                  // Slot info
        0,                                          // time_offset
        fh_proxy_->getDynamicBeamIdStart(),         // Dyn Beam ID
        0,                                          // frame_structure
        0,                                          // cp_length
        &bfw_header_p,                              // bfw_header
        t_ns(0),                                    // start_ch_task_time
        true,                                       // prevSlotBfwCompStatus
        tis                                         // ti_info
    );

    // Verify successful completion
    ASSERT_TRUE(ret == SEND_CPLANE_NO_ERROR);
    if (ctx_cfg_.mMIMO_enable) {
        int ret1 = fh_proxy_->sendCPlaneMMIMO (true /* isBFW */, cell_id, peer_id_g, DIRECTION_DOWNLINK, tis);  
        int ret2 = fh_proxy_->sendCPlaneMMIMO (false /* isBFW */, cell_id, peer_id_g, DIRECTION_DOWNLINK, tis);  
        ASSERT_TRUE(ret1 == SEND_CPLANE_NO_ERROR);  
        ASSERT_TRUE(ret2 == SEND_CPLANE_NO_ERROR); 
    }
    NVLOGC_FMT(TAG_UNIT_TB_DLC, "Debug: sendCPlane completed successfully SFN:{} SLT:{}!", slot_ind.sfn_, slot_ind.slot_);

}

static constexpr uint16_t L_MAX_4_SYMBOLS[3][4] = {{2, 8, 16, 22}, {4, 8, 16, 20}, {2, 8, 16, 22}};
static constexpr uint16_t L_MAX_8_SYMBOLS[3][8] = {{2, 8, 16, 22, 30, 36, 44, 50}, {4, 8, 16, 20, 32, 36, 44, 48}, {2, 8, 16, 22, 30, 36, 44, 50}};
static constexpr uint16_t L_MAX_64_SYMBOLS[2][64] = {   {4,8,16,20,32,36,44,48,
                                                        60,64,72,76,88,92,100,104,
                                                        144,148,156,160,172,176,184,188,
                                                        200,204,212,216,228,232,240,244,
                                                        284,288,296,300,312,316,324,328,
                                                        340,344,352,356,368,372,380,384,
                                                        424,428,436,440,452,456,464,468,
                                                        480,484,492,496,508,512,520,524},
                                                        {8,12,16,20,32,36,40,44,
                                                        64,68,72,76,88,92,96,100,
                                                        120,124,128,132,144,148,152,156,
                                                        176,180,184,188,200,204,208,212,
                                                        288,292,296,300,312,316,320,324,
                                                        344,348,352,356,368,372,376,380,
                                                        400,404,408,412,424,428,432,436,
                                                        456,460,464,468,480,484,488,492}};

static inline void update_ssb_config_static(uint32_t dl_freq_abs_A, uint32_t ul_freq_abs_A, uint32_t sub_c_common, const uint16_t** lmax_symbol_list, uint16_t *l_max)
{
    //SSB case as in 3GPP TS 38.213 section 4.1
    nv::ssb_case ssb_case = nv::getSSBCase(dl_freq_abs_A/1000, ul_freq_abs_A/1000, sub_c_common);
    if(dl_freq_abs_A <= 3000000)
    {
        *l_max = 4;
        *lmax_symbol_list = L_MAX_4_SYMBOLS[ssb_case];
    }
    else if (dl_freq_abs_A <= 6000000)
    {
        *l_max = 8;
        *lmax_symbol_list = L_MAX_8_SYMBOLS[ssb_case];
    }
    else
    {
        *l_max = 64;
        *lmax_symbol_list = L_MAX_64_SYMBOLS[ssb_case];
    }
}


/**
 * Test function to demonstrate setup for update_cell_command API from scf_5g_slot_commands_pdcch.cpp
 */
void SendCPlaneUnitTest::TestUpdateCellCommand(slot_command_api::slot_indication &slot_info)
{
    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test Starting {} - demonstrating parameter setup", __func__);

    try {
        // Create a mock PDCCH PDU
        scf_fapi_pdcch_pdu_t pdcch_pdu{};
        // Create cell parameters
        cuphyCellStatPrm_t cell_params{};
        cell_configs_t& cell_configs = launch_pattern_->get_cell_configs(0);
        cell_params.phyCellId = 0;             // Physical cell ID
        cell_params.nRxAnt = cell_configs.numRxPort;                // Number of RX antennas
        cell_params.nRxAntSrs = cell_configs.numRxAnt;             // Number of SRS RX antennas
        cell_params.nTxAnt = cell_configs.numRxPort;                // Number of TX antennas
        cell_params.nPrbUlBwp = cell_configs.ulGridSize;           // UL BWP PRBs
        cell_params.nPrbDlBwp = cell_configs.dlGridSize;           // DL BWP PRBs
        cell_params.mu = 1;                    // Numerology (30 kHz SCS)

        nv::phy_config ssb_cell_params{};
        ssb_cell_params.cell_config_.phy_cell_id = 0;
        ssb_cell_params.ssb_config_.sub_c_common = 1;
        ssb_cell_params.carrier_config_.dl_grid_size[0] = cell_configs.dlGridSize;
        ssb_cell_params.carrier_config_.dl_grid_size[1] = cell_configs.dlGridSize;
        ssb_cell_params.carrier_config_.dl_grid_size[2] = cell_configs.dlGridSize;
        ssb_cell_params.carrier_config_.dl_grid_size[3] = cell_configs.dlGridSize;
        ssb_cell_params.carrier_config_.dl_grid_size[4] = cell_configs.dlGridSize;

        // Read from the L2A config file
        std::string l2a_yaml_name;
        if (TestConfig::instance().is_nrsim()) {
            std::string cuphy_cfg_name = "cuphycontroller_nrSim_SCF_CG1_" + TestConfig::instance().pattern_number + ".yaml";
            char cuphy_cfg_path[MAX_PATH_LEN];
            get_full_path_file(cuphy_cfg_path, CONFIG_YAML_FILE_PATH, cuphy_cfg_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            yaml::file_parser cuphy_parser(cuphy_cfg_path);
            yaml::document cuphy_doc = cuphy_parser.next_document();
            l2a_yaml_name = cuphy_doc.root()["l2adapter_filename"].as<std::string>();
        } else {
            l2a_yaml_name = "l2_adapter_config_F08_CG1.yaml";
        }
        char temp_config_file[MAX_PATH_LEN];
        get_full_path_file(temp_config_file, CONFIG_YAML_FILE_PATH, l2a_yaml_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

        // Parse YAML config
        yaml::file_parser parser(temp_config_file);
        yaml::document doc = parser.next_document();
        yaml::node config_node = doc.root();

        // Create PHY config options
        nv::phy_config_option config_options{};

        if (config_node.has_key("enable_precoding")) {
            config_options.precoding_enabled = (config_node["enable_precoding"].as<uint>() > 0);
        } else {
            config_options.precoding_enabled = false;
        }
        if (config_node.has_key("enable_beam_forming")) {
            config_options.bf_enabled = (config_node["enable_beam_forming"].as<uint>() > 0);
        } else {
            config_options.bf_enabled = false;
        }


        config_options.staticPdcchSlotNum = -1;  // Dynamic slot numbering

        // Test parameters
        const uint8_t testMode = 0;             // No test mode
        const int32_t cell_index = 0;           // Cell index
        const int staticPdcchSlotNum = -1;      // Dynamic slot numbering
        nv::slot_detail_t* slot_detail = nullptr; // No slot detail for this test
        const bool mmimo_enabled = ctx_cfg_.mMIMO_enable;        // Enable mMIMO

        auto& pm_map = nv::PHY_module::pm_map();

        // Ensure pm_group is created for precoding-enabled nrSim patterns
        cell_group_.create_pm_group(config_options.precoding_enabled, 1);

        // Makes an instance with phydriverctx that just stores our
        // modified fh_proxy_ -> this allows L2A APIs to execute
        // beam related queries.
        nv::PHYDriverProxy::make(ctx_.get(), mplane_configs);

        uint16_t l_max;
        const uint16_t *lmax_symbols;
        yaml::node&     yaml_config  = testmac_configs_->get_yaml_config();

        update_ssb_config_static(nDLAbsFrePointA_, nULAbsFrePointA_, ssb_cell_params.ssb_config_.sub_c_common, &lmax_symbols, &l_max);
        bool run_fh_cb = false;
        // Process the DL TTI Request (PDCCH_DL, PBCH, PDSCH)
        int offset = 0;

        bool constructed_bfw_memory = false;
        bool pdsch = false;

        for (int i = 0; i < singleton_.dl_tti_req->num_pdus; ++i)
        {
            auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(&singleton_.dl_tti_req->payload[0 + offset]));
            switch (pdu.pdu_type)
            {
                case DL_TTI_PDU_TYPE_PDCCH:
                {
                    auto &pdcch_pdu = *reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&pdu.pdu_config[0]);

                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test PDCCH parameters for update_cell_command");
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - PDCCH PDU: BWP size={}, start symbol={}, duration={} symbols",
                           (int) pdcch_pdu.bwp.bwp_size, pdcch_pdu.start_sym_index, pdcch_pdu.duration_sym);
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - Slot info: SFN={}, Slot={}\n", slot_info.sfn_, slot_info.slot_);
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - Cell params: ID={}, RX antennas={}, TX antennas={}",
                           (int) cell_params.phyCellId, cell_params.nRxAnt, cell_params.nTxAnt);
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - Config: Precoding={}, Beamforming={}\n",
                           config_options.precoding_enabled ? "enabled" : "disabled",
                           config_options.bf_enabled ? "enabled" : "disabled");

                    scf_5g_fapi::update_cell_command(&cell_group_, cell_cmd_, pdcch_pdu, testMode,
                                                     cell_index, slot_info, cell_params, config_options.staticPdcchSlotNum,
                                                     config_options, pm_map, slot_detail, mmimo_enabled, nullptr);
                    break;
                }
                case DL_TTI_PDU_TYPE_SSB:
                {

                    auto &ssb_pdu = *reinterpret_cast<scf_fapi_ssb_pdu_t*>(&pdu.pdu_config[0]);

                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test SSB parameters for update_cell_command");
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - SSB PDU: cell_id={}, beta_pss={}", (int) ssb_pdu.phys_cell_id, ssb_pdu.beta_pss);

                    scf_5g_fapi::update_cell_command(&cell_group_, cell_cmd_, ssb_pdu,
                                                     cell_index, slot_info, ssb_cell_params,
                                                     l_max, lmax_symbols, config_options,
                                                     pm_map, slot_detail, mmimo_enabled);

                    break;
                }
                case DL_TTI_PDU_TYPE_PDSCH:
                {
                    if (constructed_bfw_memory == false && mmimo_enabled == true) {
                        std::vector<uint8_t*> bfw_start_ptr_vec{};
                        ru_emulator_construct_bfw(ru_emulator_, slot_info.sfn_, slot_info.slot_, 0 /* cell id */, bfw_start_ptr_vec);
                        int ueg_idx = 0;
                        for (auto bfw_start_ptr : bfw_start_ptr_vec) {
                            bfwCoeff_mem_info[cell_index].buff_addr_chunk_h[ueg_idx++] = bfw_start_ptr;
                        }
                        if (bfw_start_ptr_vec.size()) {
                            // The bfwCoeff_mem_info is filled w/ timing parameter of the previous slot
                            // when the BFW FAPI PDU was sent to the DU. So, mimic that.
                            int lp_slot = slot_info.sfn_ * 20 + slot_info.slot_;
                            if (lp_slot == 0) {
                                bfwCoeff_mem_info[cell_index].sfn = FAPI_SFN_MAX-1;
                                bfwCoeff_mem_info[cell_index].slot = nv::mu_to_slot_in_sf(1) - 1;
                            } else {
                                bfwCoeff_mem_info[cell_index].sfn = (lp_slot - 1) / 20;
                                bfwCoeff_mem_info[cell_index].slot = (lp_slot - 1) % 20;
                            }
                        }
                        // Do this only once per all the PDUs.
                        constructed_bfw_memory = true;
                    }

                    if(!pdsch)
                    {
                        int cell_stat_prm_idx = cell_index;
                        cell_group_command* group_cmd = &cell_group_;
                        group_cmd->slot.type = slot_command_api::slot_type::SLOT_DOWNLINK;
                        group_cmd->slot.slot_3gpp = slot_info;
                        pdsch_params* pdsch_info = group_cmd->get_pdsch_params();
                        uint32_t cell_idx = pdsch_info->cell_grp_info.nCells;
                        pdsch_info->cell_dyn_info[cell_index].cellPrmStatIdx = cell_stat_prm_idx;
                        pdsch_info->cell_dyn_info[cell_index].cellPrmDynIdx = cell_index;
                        pdsch_info->cell_dyn_info[cell_index].slotNum = 0; // ((nv::phy_module().staticPdschSlotNum() > -1)? nv::phy_module().staticPdschSlotNum() : singleton_.dl_tti_req->slot);
                        pdsch_info->cell_dyn_info[cell_index].pdschStartSym = 0;
                        pdsch_info->cell_dyn_info[cell_index].nPdschSym = 0;
                        pdsch_info->cell_dyn_info[cell_index].dmrsSymLocBmsk = 0;
                        pdsch_info->cell_grp_info.nCells++;

                        NVLOGD_FMT(TAG_UNIT_TB_COMMON, "{}: PDSCH nCells={}, cellPrmDynIdx={} cell_stat_prm_idx {}",
                                   __FUNCTION__,pdsch_info->cell_grp_info.nCells, cell_idx, cell_stat_prm_idx);

                        pdsch_info->cell_index_list.push_back(cell_index);
                        pdsch_info->phy_cell_index_list.push_back(cell_index);
                        pdsch = true;
                    }

                    auto &pdsch_pdu = *reinterpret_cast<scf_fapi_pdsch_pdu_t *>(&pdu.pdu_config[0]);

                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test PDSCH parameters for update_cell_command");
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - PDSCH PDU: num_codewords={}, pdu_index={}",
                           pdsch_pdu.num_codewords, (int) pdsch_pdu.pdu_index);
                    NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - PDSCH PDU: BFW SFN={} Slot={}",
                           bfwCoeff_mem_info[cell_index].sfn, bfwCoeff_mem_info[cell_index].slot);

                    scf_5g_fapi::update_cell_command(&cell_group_, cell_cmd_, pdsch_pdu, testMode, slot_info,
                                        cell_index, pm_map, config_options.precoding_enabled, config_options.bf_enabled,
                                        273 /* num_dl_prb */, &bfwCoeff_mem_info[cell_index],
                                        mmimo_enabled, slot_detail);

                    run_fh_cb = true;
                    break;
                }
                default:
                {
                    NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Debug: SendCPlaneUnitTest Test Unhandled PDU Type:{}", (int)pdu.pdu_type);
                }
            }
            offset += pdu.pdu_size;
        } // for (int i = 0; i < singleton_.dl_tti_req->num_pdus; ++i)

        offset = 0;
        bool first_csirs = false;
        uint32_t csirs_offset = 0;
        for (int i = 0; i < singleton_.dl_tti_req->num_pdus; ++i) {
            auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(&singleton_.dl_tti_req->payload[0 + offset]));

            if(pdu.pdu_type == DL_TTI_PDU_TYPE_CSI_RS)
            {
                NVLOGD_FMT(TAG_UNIT_TB_COMMON, "{}: Processing CSI-RS:PDU Type ={}, PDU Size={}, Offset={} cell_id ={}",
                           __FUNCTION__, static_cast<int>(pdu.pdu_type), static_cast<int>(pdu.pdu_size), offset, cell_index);
                auto &pdu_dat = *reinterpret_cast<scf_fapi_csi_rsi_pdu_t*>(&pdu.pdu_config[0]);
                {
                    //For the first CSI-RS of the cell, we need to capture the CSI-RS offset in the cell groups
                    if(!first_csirs)
                    {
                        if(pdsch)
                        {
                            cell_group_command* group_cmd = &cell_group_;
                            pdsch_params* pdsch_info = group_cmd->get_pdsch_params();
                            csirs_offset = pdsch_info->cell_grp_info.nCsiRsPrms;
                            NVLOGC_FMT(TAG_UNIT_TB_COMMON, "{}: CSI-RS offset = pdsch_info->cell_grp_info.nCsiRsPrms = {} pdsch_exist={}",
                                       __FUNCTION__, pdsch_info->cell_grp_info.nCsiRsPrms,pdsch);
                        }
                        first_csirs = true;
                    }
                }

                auto &csirs_pdu  = *reinterpret_cast<scf_fapi_csi_rsi_pdu_t*>(&pdu.pdu_config[0]);
                scf_5g_fapi::update_cell_command(&cell_group_, cell_cmd_, csirs_pdu, slot_info,
                                    cell_index, cell_params, config_options, pm_map,
                                    csirs_offset, pdsch, 0 /* cell_stat_prm_idx */,
                                    mmimo_enabled, slot_detail);

                run_fh_cb = true; // Run FHCB if either PDSCH or CSIRS PDUs are present.
            }
            offset += pdu.pdu_size;
        }

        if (first_csirs) // Rename to is_csirs_present_in_cell
        {
            csirs_params * csirs = nullptr;
            cell_group_command* group_cmd = &cell_group_;
            csirs = group_cmd->csirs.get();
            uint32_t nCsirs = (csirs ? csirs->cellInfo[csirs->nCells].nRrcParams : 0);
            if(pdsch || nCsirs)
            {
                csirs_fh_prepare_params &csirs_fh_params = group_cmd->fh_params.csirs_fh_params.at(cell_index);

                csirs_fh_params.cell_idx = cell_index;
                csirs_fh_params.cuphy_params_cell_idx = (pdsch ? cell_index : -1);
                csirs_fh_params.cell_cmd = &cell_cmd_;
                csirs_fh_params.bf_enabled = config_options.bf_enabled;
                csirs_fh_params.mmimo_enabled = mmimo_enabled;
                csirs_fh_params.num_dl_prb = 273;

                group_cmd->fh_params.is_csirs_cell.at(cell_index) = 1;
                group_cmd->fh_params.num_csirs_cell++;
            }
            if(nCsirs)
            {
                csirs->nCells++;
                csirs->lastCell = cell_index+1;
                csirs->cellInfo[csirs->nCells].rrcParamsOffset = csirs->nCsirsRrcDynPrm;
                // nv::phy_module().is_csirs_slot(true);
            }
            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "{}: pdsch={} nCsirs={} CSIRS.nCells={}", __FUNCTION__,pdsch,nCsirs,csirs->nCells);
        }

        // Process the UL DCI Request PDU (PDCCH UL)
        offset = 0;
        for (int i = 0; i < singleton_.ul_dci_req->num_pdus; ++i)
        {
            auto &pdu = *(reinterpret_cast<scf_fapi_generic_pdu_info_t*>(&singleton_.ul_dci_req->payload[0 + offset]));
            auto &pdcch_pdu = *reinterpret_cast<scf_fapi_pdcch_pdu_t*>(&pdu.pdu_config[0]);

            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test PDCCH_UL parameters for update_cell_command");
            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - PDCCH PDU: BWP size={}, start symbol={}, duration={} symbols",
                       (int) pdcch_pdu.bwp.bwp_size, pdcch_pdu.start_sym_index, pdcch_pdu.duration_sym);
            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - Slot info: SFN={}, Slot={}", slot_info.sfn_, slot_info.slot_);
            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - Cell params: ID={}, RX antennas={}, TX antennas={}",
                       (int) cell_params.phyCellId, cell_params.nRxAnt, cell_params.nTxAnt);
            NVLOGD_FMT(TAG_UNIT_TB_COMMON, "  - Config: Precoding={}, Beamforming={}\n",
                       config_options.precoding_enabled ? "enabled" : "disabled",
                       config_options.bf_enabled ? "enabled" : "disabled");

            scf_5g_fapi::update_cell_command(&cell_group_, cell_cmd_, pdcch_pdu, testMode,
                                             cell_index, slot_info, cell_params, staticPdcchSlotNum,
                                             config_options, pm_map, slot_detail, mmimo_enabled, nullptr);
            offset += pdu.pdu_size;
        }

        // Run fronthaul callback to process PDSCH/CSIRS PDUs
        if (run_fh_cb) {
            if ((!config_options.precoding_enabled) && (!config_options.bf_enabled))
            {
                scf_5g_fapi::LegacyCellGroupFhContext fh_context(cell_group_, cell_index);
                scf_5g_fapi::fh_callback<true /* mplane_configured_ru_type */,false /* config_options_precoding_enabled */,false /*config_options_bf_enabled */>(fh_context, slot_detail);
            }
            else if ((!config_options.precoding_enabled) && (config_options.bf_enabled))
            {
                scf_5g_fapi::LegacyCellGroupFhContext fh_context(cell_group_, cell_index);
                scf_5g_fapi::fh_callback<true /* mplane_configured_ru_type */,false /* config_options_precoding_enabled */,true /*config_options_bf_enabled */>(fh_context, slot_detail);
            }
            else if ((config_options.precoding_enabled) && (!config_options.bf_enabled))
            {
                scf_5g_fapi::LegacyCellGroupFhContext fh_context(cell_group_, cell_index);
                scf_5g_fapi::fh_callback<true /* mplane_configured_ru_type */,true /* config_options_precoding_enabled */,false /*config_options_bf_enabled */>(fh_context, slot_detail);
            }
            else // ((config_options.precoding_enabled) && (config_options.bf_enabled))
            {
                scf_5g_fapi::LegacyCellGroupFhContext fh_context(cell_group_, cell_index);
                scf_5g_fapi::fh_callback<true /* mplane_configured_ru_type */,true /* config_options_precoding_enabled */,true /*config_options_bf_enabled */>(fh_context, slot_detail);
            }
        }

        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest update_cell_command complete!"
                                       "NUM_DL_TTI_PDUS:{} NUM_UL_DCI_REQ_PDUS:{}",
                                        singleton_.dl_tti_req->num_pdus, singleton_.ul_dci_req->num_pdus);

    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Error in {}:{}", __func__, e.what());
        throw;
    } catch (...) {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Unknown Error in {}", __func__);
        throw;
    }
}

// Add this to your SendCPlaneUnitTest class
void SendCPlaneUnitTest::Setup_MockTransport()
{
    nv_ipc_config_t ipc_config{};
    ipc_config.module_type = NV_IPC_MODULE_PRIMARY;
    ipc_config.ipc_transport = NV_IPC_TRANSPORT_SHM;
    ipc_config.transport_config.shm.cuda_device_id = 0;
    ipc_config.transport_config.shm.ring_len = 1;
    ipc_config.transport_config.shm.mempool_size[0].buf_size = 0;
    ipc_config.transport_config.shm.mempool_size[1].buf_size = 0;
    ipc_config.transport_config.shm.mempool_size[2].buf_size = 0;
    ipc_config.transport_config.shm.mempool_size[3].buf_size = 0;
    ipc_config.transport_config.shm.mempool_size[4].buf_size = 0;
    ipc_config.transport_config.shm.mempool_size[0].pool_len = 0;
    ipc_config.transport_config.shm.mempool_size[1].pool_len = 0;
    ipc_config.transport_config.shm.mempool_size[2].pool_len = 0;
    ipc_config.transport_config.shm.mempool_size[3].pool_len = 0;
    ipc_config.transport_config.shm.mempool_size[4].pool_len = 0;
    strncpy (ipc_config.transport_config.shm.prefix, "testbench", strlen("testbench") + 1);

    try {
        // Create transport (this will handle NVIPC setup internally)
        mock_transport_ = std::make_unique<nv::phy_mac_transport>(ipc_config);
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: SendCPlaneUnitTest Test MockTransport created successfully");

    } catch (const std::exception& e) {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Error creating mock transport:{}", e.what());
        throw;
    }
}

void SendCPlaneUnitTest::Setup_TestMAC_Integration()
{

    char test_mac_yaml_array[MAX_PATH_LEN];
    std::string temp_path = std::string(CONFIG_TESTMAC_YAML_PATH).append(CONFIG_TESTMAC_YAML_NAME);
    get_full_path_file(test_mac_yaml_array, NULL, temp_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    // Parse YAML config
    yaml::file_parser parser(test_mac_yaml_array);
    yaml::document doc = parser.next_document();
    yaml::node config_node = doc.root();

    // Create testMAC configs with yaml::node
    testmac_configs_ = std::make_shared<test_mac_configs>(config_node);

    // Build pattern filename dynamically based on pattern type
    std::string pattern_file;
    if (TestConfig::instance().is_nrsim()) {
        pattern_file = "launch_pattern_nrSim_" + TestConfig::instance().pattern_number + ".yaml";
    } else {
        pattern_file = "launch_pattern_F08_1C_" + TestConfig::instance().pattern_number + ".yaml";
    }

    // Create and parse launch pattern
    launch_pattern_ = new launch_pattern(testmac_configs_.get());
    if (launch_pattern_->launch_pattern_parsing(pattern_file.c_str(), 0xFFFFFFF, 0x1) < 0)
    {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Launch pattern parsing failed for pattern {}!",
                   TestConfig::instance().pattern_number);
        throw std::runtime_error("Launch pattern parsing failed");
    }

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Launch pattern {} loaded successfully", TestConfig::instance().pattern_number);
    for (int cell = 0; cell < 1 /* num cells is 1 */; ++cell) {

        std::string exp_slot = "Launch pattern expected slot schedule: Cell=" + std::to_string(cell);
        for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
        {
            exp_slot.append(" ").append(get_channel_name(ch)).append("=").append(std::to_string(launch_pattern_->get_expected_values().at(cell).lp_slots[ch]));
        }
        NVLOGC_FMT(TAG_UNIT_TB_COMMON,"{}", exp_slot);
    }

    // For now, skip launch pattern parsing since we don't have actual pattern files
    // This would normally require actual test vector files
    nv_ipc_config_t config_{};
    // 3. Create FAPI handler with raw pointers using our wrapper class
    fapi_handler_ = std::make_shared<TestFapiHandler>(
        *mock_transport_.get(),
        testmac_configs_.get(),
        launch_pattern_,
        nullptr
    );

    nDLAbsFrePointA_ = config_node["data"]["nDLAbsFrePointA"].as<uint32_t>();
    nULAbsFrePointA_ = config_node["data"]["nULAbsFrePointA"].as<uint32_t>();
    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "testMAC integration setup completed!");

}

std::tuple<bool,bool> SendCPlaneUnitTest::GenerateFapiMessagesForSlot(uint16_t sfn, uint16_t slot)
{
    sfn_slot_t ss{};
    ss.u16.sfn = sfn;
    ss.u16.slot = slot;

    bool run_dl_cplane = false, run_ul_cplane = false;

    singleton_.dl_tti_req->sfn = ss.u16.sfn;
    singleton_.dl_tti_req->slot = ss.u16.slot;
    singleton_.ul_dci_req->sfn = ss.u16.sfn;
    singleton_.ul_dci_req->slot = ss.u16.slot;

    static uint8_t ping_pong_idx = 0;


    // Generate FAPI messages for this slot
    for (int group_id = 0; group_id < FAPI_REQ_SIZE; group_id++) {
        // phy_mac_msg_desc msg_desc = create_local_msg_desc(local_msg);
        // Get FAPI requests for this slot/group
        vector<fapi_req_t*>& fapi_reqs = fapi_handler_->get_fapi_req_list_public(0/*cell-id*/, ss, (fapi_group_t)group_id);

        if (fapi_reqs.size() > 0) {
            // Build specific FAPI message type
            switch(group_id) {
                case DL_TTI_REQ:
                    fapi_handler_->build_dl_tti_request_public(0, fapi_reqs, *singleton_.dl_tti_req);
                    run_dl_cplane = true;
                    break;
                case UL_DCI_REQ:
                    fapi_handler_->build_ul_dci_request_public(0, fapi_reqs, *singleton_.ul_dci_req);
                    run_dl_cplane = true;
                    break;
                // case DL_BFW_CVI_REQ:
                //     Uses RU Emulator's parsed data objects to construct the BFW memory as RUE has already parsed the
                //     compressed BFW from the TV for verification purposes.
                //     break;
                // case UL_TTI_REQ:
                //     fapi_handler_->build_ul_tti_request_public(0, fapi_reqs, singleton_.ul_tti_req);
                //     break;
                // case TX_DATA_REQ:
                //     fapi_handler_->build_tx_data_request_public(0, fapi_reqs, local_msg.tx_data_req, msg_desc);
                //     break;
                // Add other message types as needed
            }
            NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Debug: DONE Generating FAPI messages for group:{} SFN:{}.SLT:{} sz:{}", group_id, sfn, slot, fapi_reqs.size());
        }
    }

    singleton_.print_dl_tti_req_params();

    return std::make_tuple(run_dl_cplane, run_ul_cplane);
}

#ifdef GENERATE_SAMPLE_SLOT_CMD
/**
 * Creates a fully functional slot_info_t structure for comprehensive sendCPlane testing
 * This ensures all code paths are exercised and the function completes successfully
 */
static slot_command_api::slot_info_t* createComprehensivePdcchSlotInfo() {
    static slot_command_api::slot_info_t slot_info{};

    // Critical: Reset and initialize the structure properly
    slot_info.reset();

    // Initialize the atomic flag (critical for sendCPlane completion)
    slot_info.section_id_ready.store(false);

    // PDCCH typically spans symbols 0-2 in a slot
    const uint8_t PDCCH_START_SYMBOL = 0;
    const uint8_t PDCCH_NUM_SYMBOLS = 3;
    const uint16_t START_PRB = 0;
    const uint16_t NUM_PRBS = 24;    // Typical CORESET0 bandwidth

    // Add PRB info entries for each PDCCH symbol
    for (uint8_t sym = PDCCH_START_SYMBOL; sym < PDCCH_START_SYMBOL + PDCCH_NUM_SYMBOLS; sym++) {
        // Add PRB allocation for this symbol
        slot_info.prbs[slot_info.prbs_size].common.startPrbc = START_PRB;
        slot_info.prbs[slot_info.prbs_size].common.numPrbc = NUM_PRBS;
        // Symbol info will be set in section mapping
        slot_info.prbs[slot_info.prbs_size].common.numSymbols = 1;
        // Beam ID not in common struct
        // UE ID not in common struct
        // Compression method not in common struct
        // IQ width not in common struct
        // Compression param not in common struct
        slot_info.prbs[slot_info.prbs_size].common.filterIndex = 0;
        // EF not in common struct
        // mcScale fields not in common struct
        // mcScale fields not in common struct
        // symInc not in common struct
        // iqSample not in common struct
        slot_info.prbs[slot_info.prbs_size].common.portMask = 1;
        slot_info.prbs[slot_info.prbs_size].common.numApIndices = 1;

        // Add to symbol mapping
        slot_info.symbols[sym][slot_command_api::channel_type::PDCCH_DL].push_back(slot_info.prbs_size);

        slot_info.prbs_size++;
    }

    // Set start symbols for DL processing
    slot_info.start_symbol_dl = PDCCH_START_SYMBOL;
    slot_info.start_symbol_ul = 14; // No UL in this slot

    return &slot_info;
}
#endif // #ifdef GENERATE_SAMPLE_SLOT_CMD

}  // namespace aerial_fh

// ==================== Google Test Framework Integration ====================

/**
 * Google Test fixture for SendCPlane tests with parameterized SFN and Slot
 * Uses SetUpTestSuite/TearDownTestSuite for one-time setup/teardown across all test cases
 */
class SendCPlaneTest : public ::testing::TestWithParam<int> {
protected:
    // Static test instance shared across all parameterized tests
    static std::unique_ptr<aerial_fh::SendCPlaneUnitTest> test_instance;

    /**
     * SetUpTestSuite runs ONCE before all parameterized test cases
     * This is where we do the expensive one-time initialization
     */
    static void SetUpTestSuite() {
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Google Test SetUpTestSuite: Running ONE-TIME setup ===");

        // Create test instance once for all tests
        test_instance = std::make_unique<aerial_fh::SendCPlaneUnitTest>();

        // Run one-time setup (RU emulator, NIC, peers, transport, etc.)
        test_instance->Setup(0 /* cell index */);

        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Google Test SetUpTestSuite: ONE-TIME setup complete ===");
    }

    /**
     * TearDownTestSuite runs ONCE after all parameterized test cases
     * This is where we do cleanup
     */
    static void TearDownTestSuite() {
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Google Test TearDownTestSuite: Running ONE-TIME cleanup ===");

        if (test_instance) {
            test_instance->TearDown();
            test_instance.reset();
        }

        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Google Test TearDownTestSuite: ONE-TIME cleanup complete ===");
    }

    /**
     * SetUp runs before EACH test case (optional, for per-test initialization)
     * Leave empty if no per-test setup is needed
     */
    void SetUp() override {
        auto pattern_slot = GetParam();
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Starting test for SFN={}, Slot={} ===", pattern_slot / 20, pattern_slot % 20);
    }

    /**
     * TearDown runs after EACH test case (optional, for per-test cleanup)
     * Leave empty if no per-test cleanup is needed
     */
    void TearDown() override {
        auto pattern_slot = GetParam();
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Finished test for SFN={}, Slot={} ===", pattern_slot / 20, pattern_slot % 20);
        test_instance->Reset();
    }
};

// Define the static member variable
std::unique_ptr<aerial_fh::SendCPlaneUnitTest> SendCPlaneTest::test_instance = nullptr;

/**
 * Parameterized test for different SFN and Slot combinations
 */
TEST_P(SendCPlaneTest, TestSendCPlaneForSlot) {

    auto pattern_slot_number = GetParam();

    slot_command_api::slot_indication slot_info{};
    slot_info.sfn_ = pattern_slot_number / 20;
    slot_info.slot_ = pattern_slot_number % 20;
    slot_info.tick_ = ::Time::nowNs().count();

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "\n\n\n=== Testing SFN={}, Slot={} ===", slot_info.sfn_, slot_info.slot_);

    test_instance->Run(slot_info);

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Test PASSED for SFN={}, Slot={} ===\n\n\n", slot_info.sfn_, slot_info.slot_);
}

static std::vector<int> GenerateLaunchPatternSlots() {
    const std::string& pattern_num = TestConfig::instance().pattern_number;
    if (pattern_num.empty()) {
        // Return empty vector - tests will be dynamically added later
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "{} Empty slots", __func__);
        return std::vector<int>{};
    }

    std::string pattern_file;
    if (TestConfig::instance().is_nrsim()) {
        pattern_file = "launch_pattern_nrSim_" + pattern_num + ".yaml";
    } else {
        pattern_file = "launch_pattern_F08_1C_" + pattern_num + ".yaml";
    }

    char pattern_file_array[MAX_PATH_LEN];
    get_full_path_file(pattern_file_array, CONFIG_LAUNCH_PATTERN_PATH, pattern_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    // Quick parse just to get slot count
    yaml::file_parser parser(pattern_file_array);
    yaml::document doc = parser.next_document();
    yaml::node root = doc.root();
    yaml::node sched = root["SCHED"];
    const int slot_count = sched.length();

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "{} Scheduling Pattern:{} Length:{} slots", __func__, pattern_num, slot_count);

    // Generate slot numbers [0, slot_count)
    std::vector<int> slots(slot_count);
    std::iota(slots.begin(), slots.end(), 0);
    return slots;
}

/**
 * Instantiate the parameterized test with launch pattern slots
 *
 * Note: SetUpTestSuite() runs ONCE before all these test cases
 *       TearDownTestSuite() runs ONCE after all these test cases
 *       SetUp()/TearDown() run before/after EACH test case
 */
INSTANTIATE_TEST_SUITE_P(
    MultipleSlots,
    SendCPlaneTest,
    ::testing::ValuesIn(GenerateLaunchPatternSlots())
    //::testing::Range(6, 7) // - alternatively, you can choose a selective slot to run
);

// ==================== Google Benchmark Integration ====================

/**
 * Global test instance for benchmarks
 * Must be initialized before running benchmarks
 */
std::unique_ptr<aerial_fh::SendCPlaneUnitTest> g_benchmark_test_instance = nullptr;

/**
 * Benchmark setup/teardown for sendCPlane benchmarks
 */
static void BenchmarkGlobalSetup() {
    // Initialize test instance once (check if already initialized)
    if (!g_benchmark_test_instance) {

        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Benchmark SetUp: Initializing test environment ===");
        fmtlog::poll(true);  // Flush immediately

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(4, &cpuset);  // pin to core #4
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        g_benchmark_test_instance = std::make_unique<aerial_fh::SendCPlaneUnitTest>();
        g_benchmark_test_instance->Setup(0 /* cell index */);

        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "=== Benchmark SetUp: Initialization complete ===");
        fmtlog::poll(true);  // Flush immediately before benchmark starts
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Give time for logs to print
    }
}

static void BenchmarkGlobalTeardown(const ::benchmark::State& state) {
    // Flush any logs from this benchmark iteration
    fmtlog::poll(true);

    if (g_benchmark_test_instance) {
        // This only resets the internal command structs and doesn't teardown all the resources.
        g_benchmark_test_instance->Reset();
    }

    // Note: We don't clean up g_benchmark_test_instance here because it's
    // shared across all benchmarks. Cleanup happens in main() [see: dlc_testbench.cpp] after all benchmarks complete.
}

/**
 * Register benchmarks dynamically - call from main() after CLI parsing
 * This ensures pattern_number is available when GenerateLaunchPatternSlots() runs
 */
namespace aerial_fh {
void RegisterDynamicBenchmarks() {
    const auto slots = GenerateLaunchPatternSlots();

    // Only register if we have slots (i.e., pattern_number was parsed)
    if (slots.empty()) {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                   "Cannot register benchmarks - no slots found (pattern not loaded)");
        return;
    }

    BenchmarkGlobalSetup(); 

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Registering {} slot benchmarks", slots.size());

    // Register a benchmark for each slot (only DL slots)
    for (const auto& lp_slot : slots) {
        const uint16_t sfn = lp_slot / 20;
        const uint16_t slot = lp_slot % 20;

        // Pre-check if this slot has DL C-Plane (avoid registering UL-only slots)
        auto [run_dl, run_ul] = g_benchmark_test_instance->GenerateFapiMessagesForSlot(sfn, slot);
        g_benchmark_test_instance->Reset();  // Clean up after check
        
        if (!run_dl) {
            NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Skipping UL-only slot {}/{} from benchmark registration", sfn, slot);
            continue;  // Skip UL-only slots
        }

        const string bench_name = fmt::format("SendCPlane_{}_{}", sfn, slot); 

        ::benchmark::RegisterBenchmark(
            bench_name.c_str(),
            [lp_slot, bench_name](benchmark::State& state) {
                const uint16_t sfn = lp_slot / 20;
                const uint16_t slot = lp_slot  % 20;
                
                // thrash_cache();

                NVLOGC_FMT(TAG_UNIT_TB_COMMON, "[BENCHMARK] Starting {} benchmark!", bench_name);
                fmtlog::poll(true);  // Flush immediately

                // Create slot indication
                slot_command_api::slot_indication slot_info{};
                slot_info.sfn_ = sfn;
                slot_info.slot_ = slot;

                // Prepare FAPI messages once before benchmark loop
                // Note: UL-only slots are filtered out during registration, so this slot has DL C-Plane
                auto [run_dl_cplane, run_ul_cplane] = g_benchmark_test_instance->GenerateFapiMessagesForSlot(sfn, slot);

                g_benchmark_test_instance->TestUpdateCellCommand(slot_info);

                // Flush all setup logs before starting benchmark iterations
                fmtlog::poll(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                const int cell_id = 0;

                slot_info.tick_ = ::Time::nowNs().count(); // Update timestamp each iteration

                // Get slot info
                slot_command_api::slot_info_t &slot_info_ref = *g_benchmark_test_instance->get_cell_cmd().sym_prb_info();

                std::optional<PerfProfiler> profile;
                if (aerial_fh::TestConfig::instance().use_perf_profiler) {
                    std::filesystem::path dirname = "/tmp/perf_data";
                    profile.emplace(9999, dirname);
                    std::string s = fmt::format("{}_{}.data", sfn, slot);
                    profile->start(s);
                }

                // Benchmark loop - measures only the sendCPlane call
                for (auto _ : state) {

                    state.PauseTiming();
                    // Reset section_id_ready flag for next iteration
                    slot_info_ref.section_id_ready.store(false);

                    // Create slot indication format
                    slot_command_api::oran_slot_ind slot_indication = to_oran_slot_format(slot_info);

                    uint8_t* bfw_header = nullptr;
                    ti_subtask_info tis{};

                    state.ResumeTiming();
                    int ret = aerial_fh::fh_proxy_->prepareCPlaneInfo(
                        0,                                  // cell_id
                        MULTI_SECT_MODE,                    // ru_type
                        aerial_fh::peer_id_g,               // peer_id
                        g_benchmark_test_instance->get_cell_dl_comp_meth(cell_id),      // dl_comp_meth
                        t_ns(::Time::nowNs().count() + 1000000UL), // start_tx_time
                        0,                                  // tx_cell_start_ofs_ns
                        DIRECTION_DOWNLINK,                 // direction
                        slot_indication,                    // slot_indication
                        slot_info_ref,                      // Slot info
                        0,                                  // time_offset
                        fh_proxy_->getDynamicBeamIdStart(), // Dyn Beam ID
                        0,                                  // frame_structure
                        0,                                  // cp_length
                        &bfw_header,                        // bfw_header
                        t_ns(0),                            // start_ch_task_time
                        true,                               // prevSlotBfwCompStatus
                        tis                                 // ti_info
                    );
                    
                    if (g_benchmark_test_instance->get_ctx_cfg().mMIMO_enable) {
                        int ret1 = aerial_fh::fh_proxy_->sendCPlaneMMIMO (true /* isBFW */, 0, aerial_fh::peer_id_g, DIRECTION_DOWNLINK, tis);  
                        int ret2 = aerial_fh::fh_proxy_->sendCPlaneMMIMO (false /* isBFW */, 0, aerial_fh::peer_id_g, DIRECTION_DOWNLINK, tis);  
                        if (ret1 != SEND_CPLANE_NO_ERROR || ret2 != SEND_CPLANE_NO_ERROR) {
                            state.SkipWithError("prepareCPlaneInfo failed!");
                            break;
                        }
                    }

                    // Verify successful completion
                    if (ret != SEND_CPLANE_NO_ERROR) {
                        state.SkipWithError("prepareCPlaneInfo failed!");
                        break;
                    }
                }

                if (profile.has_value()) {
                    profile->stop();
                }

                // Report additional statistics
                state.SetItemsProcessed(state.iterations());
                state.SetLabel(bench_name);

                NVLOGC_FMT(TAG_UNIT_TB_COMMON, "[BENCHMARK] Completed {} benchmark: {} iterations!", bench_name, state.iterations());
                fmtlog::poll(true);

                // Cleanup after benchmark
                g_benchmark_test_instance->Reset();
            }
        )
        ->Unit(benchmark::kMicrosecond)
        // ->Repetitions(10) // Turn this ON to repeat the benchmark several times. Provides mean/std-dev statistics
        ->Iterations(200000) 
        ->DisplayAggregatesOnly(false);
    }
}
} // namespace aerial_fh

