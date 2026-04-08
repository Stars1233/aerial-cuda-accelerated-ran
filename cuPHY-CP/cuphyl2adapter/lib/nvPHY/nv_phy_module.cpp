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

#include "nv_phy_module.hpp"
#include "nv_phy_factory.hpp"
#include "nv_phy_driver_proxy.hpp"
#include "memtrace.h"
#include "oran_utils/conversion.hpp"
#include <grpcpp/grpcpp.h>
#include "aerial_common.grpc.pb.h"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 6) // "L2A.MODULE"
#define TAG_PROCESSING_TIMES (NVLOG_TAG_BASE_L2_ADAPTER + 11) // "L2A.PROCESSING_TIMES"
#define TAG_TICK_TIMES (NVLOG_TAG_BASE_L2_ADAPTER + 12) // "L2A.TICK_TIMES"

#include "cuphyoam.hpp"

#include <unistd.h> // usleep(), temporary!
#include <chrono>
#include <atomic>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

namespace nv
{

static int oam_update_cell_attenuation(int32_t mplane_id, float attenuation_dB) {
    return PHYDriverProxy::getInstance().l1_cell_update_attenuation(mplane_id, attenuation_dB);
}

int PHY_module::send_sfn_slot_sync_grpc_command()
{
      int ret=0;  
      NVLOGI_FMT(TAG,"{} : server_addr {}",__func__,server_addr);  
      aerial::Common::Stub stub(grpc::CreateChannel(server_addr, grpc::InsecureChannelCredentials()));
      nanoseconds ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
      uint64_t curr_time=ts_now.count();
      int32_t sync_done=2;

      aerial::SfnSlotSyncRequest request;
      request.set_sync_done(sync_done);

      aerial::DummyReply reply;
      ClientContext context;

      Status status = stub.SendSfnSlotSyncCmd(&context, request, &reply);
      if (status.ok())
      {
        NVLOGI_FMT(TAG,"gRPC message sent successfully for SFN/slot sync at time {}",curr_time);
        ret=0;
      }
      else
      {
        NVLOGE_FMT(TAG,AERIAL_CUPHYDRV_API_EVENT,"gRPC message send failure");
        ret=1;
      }
      return ret;
}

int phy_module_reset_callback(phy_mac_transport* transp, PHY_module* phy_module)
{
    if(transp == nullptr || phy_module == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: invalid pointer", __func__);
        return -1;
    }
    return phy_module->reset_transport(transp);
}

int PHY_module::reset_transport(phy_mac_transport* transp)
{
    int32_t transport_id = transp->get_transport_id();
    uint64_t started_cells_mask = transp->get_started_cells_mask();
    uint64_t configured_cells_mask = transport_wrapper().get_configured_cells_mask();

    NVLOGC_FMT(TAG, "{}: phy_instances_.size={} dl_tbs_queue_.size={} transport_id={} started_cells_mask=0x{:X} configured_cells_mask=0x{:X}",
            __func__, phy_instances_.size(), dl_tbs_queue_.size(), transport_id, started_cells_mask, configured_cells_mask);

    // Reset phy instances which bounded to this transport or in error state
    int phy_cell_id = 0;
    while(configured_cells_mask != 0)
    {
        if(configured_cells_mask & 0x1)
        {
            if(transport_id == transport_wrapper_.get_transport_id(phy_cell_id))
            {
                // Reset all phy instances which belongs to this transport
                phy_instances_[phy_cell_id].get()->reset();
            }
            else
            {
                // The phy instance belongs to another transport, only occurs in Multi-L2 case
                phy_mac_transport& other_transport = transport_wrapper_.get_transport(phy_cell_id);
                if (other_transport.get_error_flag())
                {
                    // The transport is in error state, need reset it to inactive the cell (avoid cell update locking problem)
                    phy_instances_[phy_cell_id].get()->reset();
                }
                else
                {
                    // The transport is not in error state, skip reset it
                    NVLOGC_FMT(TAG, "{}: skip reset phy for phy_cell_id={} transport_id={}", __func__, phy_cell_id, other_transport.get_transport_id());
                }
            }
        }
        configured_cells_mask >>= 1;
        phy_cell_id++;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////
// PHY_module::PHY_module()
using namespace std::chrono;
// slot_command_api::cell_group_command PHY_module::group_command_ {};
std::unordered_map<uint32_t, pm_weights_t> PHY_module::pm_weight_map_ {};
std::unordered_map<uint16_t, digBeam_t> PHY_module::static_digBeam_weight_map_ {};

PHY_module::~PHY_module()
{
    if (slot_latency != nullptr)
    {
        slot_latency->close(slot_latency);
        slot_latency = nullptr;
    }
    if (tick_logger != nullptr)
    {
        tick_logger->close(tick_logger);
        tick_logger = nullptr;
    }
}

PHY_module::PHY_module(yaml::node node_config) :
    total_cell_num(nv::PHYDriverProxy::getInstance().l1_get_cell_group_num()),
    transport_wrapper_(node_config, NV_IPC_MODULE_PHY, nv::PHYDriverProxy::getInstance().l1_get_cell_group_num()),
    epoll_ctx_p(new phy_epoll_context()),
    dl_tbs_queue_(MAX_CELLS_PER_SLOT * 20), // There's 1 TX_DATA.req per cell per slot, total supports maximum 20 slots in the queue.
    tick_updater_(node_config),
    tti_module_(node_config, *this),
    current_tick_(0ns),
    dtx_thresholds_({CUPHY_DEFAULT_EXT_DTX_THRESHOLD}),
    dtx_thresholds_pusch_(CUPHY_DEFAULT_EXT_DTX_THRESHOLD),
    current_slot_cmd_index(0),
    num_cells_active(0),
#ifdef ENABLE_L2_SLT_RSP
    active_cell_bitmap(0),
    fapi_eom_rcvd_bitmap(0)
{
#else
    num_msgs(0),
    ipc_sync_mode(sync_mode_t::SYNC_MODE_PER_CELL),
    config_options_(),
    fapi_config_check_mask_(0),
    cell_limit_errors_{},
    group_limit_errors_()
{
    if (node_config.has_key("allowed_fapi_latency"))
    {
        allowed_fapi_latency = node_config["allowed_fapi_latency"].as<uint32_t>();
    }
    else
    {
        allowed_fapi_latency = 0;
    }
#endif

    NVLOGC_FMT(TAG, "{}: transport_num={} total_cell_num={}", __func__, transport_wrapper_.get_transport_num(), total_cell_num);

    next_slot_fapi_num = 0;

    //------------------------------------------------------------------
    // Create a PHY instance for each element of the "instances" YAML
    // sequence. The type will be dictated by the "phy_class" value.
    std::string phy_class   = node_config["phy_class"].as<std::string>();
    yaml::node  phy_configs = node_config["instances"];
    
    // Validate that the number of instances is sufficient for the number of cells
    if (phy_configs.length() < total_cell_num) {
        std::string error_msg = fmt::format(
            "Configuration error: number of instances ({}) is less than the number of cells ({}). "
            "Please ensure that the 'instances' array length in l2adapter configuration YAML file is at least equal to the 'cell_group_num' value from the cuPHYcontroller configuration YAML file.",
            phy_configs.length(), total_cell_num);
        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "{}", error_msg);
        throw std::runtime_error(error_msg);
    }
    
    for(size_t i = 0; i < phy_configs.length(); ++i)
    {
        phy_instances_.push_back(PHY_instance_ptr(nv::phy_factory::create(phy_class.c_str(),
                                                                          *this,
                                                                          phy_configs[i])));
        // Maintain a vector of PHY instance references that we will
        // expose via the public interface
        phy_refs_.push_back(*phy_instances_.back());
    }
    std::unique_ptr<member_event_callback<PHY_module>> mcb_p(new member_event_callback<PHY_module>(this, &PHY_module::msg_processing));
    for (phy_mac_transport* ptransport : transport_wrapper_.get_transports())
    {
        ptransport->set_reset_callback(phy_module_reset_callback, this);
        epoll_ctx_p->add_fd(ptransport->get_fd(), mcb_p.get());
    }
    msg_mcb_p = std::move(mcb_p);
    if (node_config.has_key("message_thread_config")) {
        yaml::node cfg = node_config["message_thread_config"];
        thread_cfg_.reset(new thread_config);
        thread_cfg_->name = cfg["name"].as<std::string>();
        thread_cfg_->cpu_affinity = cfg["cpu_affinity"].as<int32_t>();
        thread_cfg_->sched_priority = cfg["sched_priority"].as<int32_t>();
    }

    if(node_config.has_key("sfn_slot_sync_se")){
        yaml::node sfn_slot_sync_se_config = node_config["sfn_slot_sync_se"];
        server_addr = sfn_slot_sync_se_config["server_addr"].as<std::string>();
        target_node = sfn_slot_sync_se_config["target_node"].as<uint8_t>();
        enable_se_sync_cmd = sfn_slot_sync_se_config["enable"].as<uint8_t>();
    }
    else
    {
        enable_se_sync_cmd=0;
    }
    sync_rcvd_from_ue=false;
    sync_rcvd_from_du=false;

    if (node_config.has_key("dl_tb_loc"))
    {
        uint32_t loc = node_config["dl_tb_loc"].as<uint32_t>();
        if (loc > dl_tb_loc::TB_LOC_EXT_GPU_BUF)
        {
            dl_tb_location_ = dl_tb_loc::TB_LOC_INLINE;
        }
        else
        {
            dl_tb_location_ = static_cast<dl_tb_loc>(loc);
        }
    }
    else
    {
        dl_tb_location_ = dl_tb_loc::TB_LOC_INLINE;
    }

    if (node_config.has_key("timer_thread_wakeup_threshold"))
    {
        timer_thread_wakeup_threshold_ = node_config["timer_thread_wakeup_threshold"].as<uint64_t>();
    }

    if (node_config.has_key("l2a_allowed_latency"))
    {
        l2a_allowed_latency_ = node_config["l2a_allowed_latency"].as<uint32_t>();
    }

    if (node_config.has_key("enableTickDynamicSfnSlot"))
    {
        config_options_.enableTickDynamicSfnSlot = node_config["enableTickDynamicSfnSlot"].as<int>();
    }

    if (node_config.has_key("staticPuschSlotNum"))
    {
        config_options_.staticPuschSlotNum =  node_config["staticPuschSlotNum"].as<int>();
    }

    if (node_config.has_key("staticPdschSlotNum"))
    {
        config_options_.staticPdschSlotNum = node_config["staticPdschSlotNum"].as<int>();
    }

    if (node_config.has_key("staticPdcchSlotNum"))
    {
        config_options_.staticPdcchSlotNum =  node_config["staticPdcchSlotNum"].as<int>();
    }

    if (node_config.has_key("staticCsiRsSlotNum"))
    {
       config_options_.staticCsiRsSlotNum = node_config["staticCsiRsSlotNum"].as<int>();
    }

    if (node_config.has_key("staticSsbPcid"))
    {
        config_options_.staticSsbPcid = node_config["staticSsbPcid"].as<int>();
    }

    if (node_config.has_key("staticSsbSFN"))
    {
        config_options_.staticSsbSFN = node_config["staticSsbSFN"].as<int>();
    }

    if (node_config.has_key("staticSsbSlotNum"))
    {
        config_options_.staticSsbSlotNum = node_config["staticSsbSlotNum"].as<int>();
    }

    if (node_config.has_key("staticPucchSlotNum")) {
        config_options_.staticPucchSlotNum = node_config["staticPucchSlotNum"].as<int>();
    }

    if (node_config.has_key("lbrm"))
    {
        if(node_config["lbrm"].as<unsigned int>() > 0)
        {
            lbrm_ = 1;
        }
    }

    if (node_config.has_key("enable_precoding")) {
        config_options_.precoding_enabled = (node_config["enable_precoding"].as<uint>() > 0);
        
        // Check if precoding is enabled and any cell has modulation compression enabled
        if (config_options_.precoding_enabled) {

            bool modulation_compression_enabled = false;
            
            for (const auto& cfg : PHYDriverProxy::getInstance().getMPlaneConfigList()) {
                if (cfg.dl_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION ||
                    cfg.ul_comp_meth == aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION) {
                    modulation_compression_enabled = true;
                    break;
                }
            }
            
            // Only throw error if both precoding and modulation compression are enabled
            if (modulation_compression_enabled) {
                std::string error_msg = fmt::format(
                    "Configuration error: Setting enable_precoding to 1 is incompatible with modulation compression (comp_meth = 4) enabled in cell configurations. "
                    "Please set enable_precoding to 0 when using modulation compression or use a different compression method.");
                NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "{}", error_msg);
                throw std::runtime_error(error_msg);
            }
        }
    }

    if (node_config.has_key("enable_beam_forming")) {
        config_options_.bf_enabled = (node_config["enable_beam_forming"].as<uint>() > 0);
    }

    if (node_config.has_key("ptp"))
    {
        if(node_config["ptp"].has_key("gps_alpha"))
        {
            gps_alpha_ = node_config["ptp"]["gps_alpha"].as<uint64_t>();
        }
        if(node_config["ptp"].has_key("gps_beta"))
        {
            gps_beta_ = node_config["ptp"]["gps_beta"].as<int>();
        }
    }

    if (node_config.has_key("ul_cqi"))
    {
        ul_cqi_ = node_config["ul_cqi"].as<uint8_t>();
    }

    if (node_config.has_key("rssi"))
    {
        rssi_ = node_config["rssi"].as<uint16_t>();
    }

    if (node_config.has_key("rsrp"))
    {
        rssi_ = node_config["rsrp"].as<uint16_t>();
    }

    if (node_config.has_key("test_type"))
    {
        test_type = node_config["test_type"].as<int32_t>();
    }

    if (node_config.has_key("cell_group"))
    {
        cell_group_ = (node_config["cell_group"].as<uint>() > 0);
    }

    if (node_config.has_key("allowed_tick_error"))
    {
        allowed_tick_error = node_config["allowed_tick_error"].as<uint32_t>();
    }
    else
    {
        allowed_tick_error = 10; // Unit: us
    }

    if (node_config.has_key("prepone_h2d_copy"))
    {
        prepone_h2d_copy_ = node_config["prepone_h2d_copy"].as<uint8_t>();
    }
    else
    {
        prepone_h2d_copy_ = 0; // Unit: us
    }

    if (node_config.has_key("fapi_config_check_mask")) {
        fapi_config_check_mask_ = node_config["fapi_config_check_mask"].as<uint64_t>();
    } else {
        fapi_config_check_mask_ = 0;
    }

    dtx_thresholds_.fill(1.0);
    if (node_config.has_key("pucch_dtx_thresholds")) {
        auto dtx_root{node_config["pucch_dtx_thresholds"]};
        for (uint i = 0; i < dtx_thresholds_.size(); i++) {
            dtx_thresholds_[i] = dtx_root[i].as<float>();
        }
    }

    dtx_thresholds_pusch_ = 1.0;
    if (node_config.has_key("pusch_dtx_thresholds")) {
        dtx_thresholds_pusch_ = node_config["pusch_dtx_thresholds"].as<float>();
    }

    if (node_config.has_key("ipc_sync_mode")) {
        ipc_sync_mode = static_cast<sync_mode_t>(node_config["ipc_sync_mode"].as<uint8_t>());
    } else {
        ipc_sync_mode = SYNC_MODE_PER_CELL;
    }
    NVLOGI_FMT(TAG, "ipc_sync_mode = {}", static_cast<int>(ipc_sync_mode));

    if (node_config.has_key("duplicate_config_all_cells")) {
        config_options_.duplicateConfigAllCells = (static_cast<uint>(node_config["duplicate_config_all_cells"]) > 0);
    }
    // Tick interval error statistic logger
    int64_t stat_period = 1E9 / mu_to_ns(tick_updater_.mu_highest_) * 5;
    tick_logger = stat_log_open("TICK.ERROR", STAT_MODE_COUNTER, stat_period); // Print every 5 seconds
    if (tick_logger)
        tick_logger->set_limit(tick_logger, allowed_tick_error * (-1000), allowed_tick_error * 1000);

    slot_latency = stat_log_open("SLOT.LATENCY", STAT_MODE_COUNTER, 10000); // Print every 10000 slots, 5 seconds
#ifdef ENABLE_L2_SLT_RSP
    NVLOGI_FMT(TAG, "{}: allowed_tick_error={} us", __func__, allowed_tick_error);
#else
    NVLOGI_FMT(TAG, "{}: allowed_fapi_latency={} allowed_tick_error={} us", __func__, allowed_fapi_latency, allowed_tick_error);
    ss_last.u32 = SFN_SLOT_INVALID;
#endif

    tti_event_count = 0;
    ss_tick.store({.u32 = 0});
    ss_curr.u32 = SFN_SLOT_INVALID;

    int slot_command_array_size = 30;
    if (node_config.has_key("slot_command_array_size"))
    {
        slot_command_array_size = node_config["slot_command_array_size"].as<uint>();
    }
    slot_command_array.reserve(slot_command_array_size);
    for (int i = 0 ; i < slot_command_array_size; i++) {
        slot_command_array.push_back(slot_command_api::slot_command());
        auto& slot_cmd = slot_command_array.back();
        for (int j = 0; j < phy_instances_.size(); j++) {
            slot_cmd.cells.push_back(slot_command_api::cell_sub_command());
            auto& slot_sub_cmd = slot_cmd.cells.back();
        /// TODO: create the channels
        }
        if (config_options_.precoding_enabled) {
            slot_cmd.cell_groups.create_pm_group(config_options_.precoding_enabled,  phy_instances_.size());
        }

    }
    csirs_lookup_api::CsirsLookup& lookup = csirs_lookup_api::CsirsLookup::getInstance();
    stat_prm_idx_to_cell_id_map.clear();

    CuphyOAM::getInstance()->callback.update_cell_attenuation = oam_update_cell_attenuation;
    TxNotificationHelper::setEnableTxNotification(PHYDriverProxy::getInstance().l1_get_dl_tx_notification());
}

////////////////////////////////////////////////////////////////////////
// PHY_module::join()
void PHY_module::join()
{
    if (thread_.joinable()) {
        thread_.join();
    }

    // Tick thread is not started at initial, so join it after the msg_processing thread joined
    tti_module_.timer_thread_join();
}

////////////////////////////////////////////////////////////////////////
// PHY_module::start()
void PHY_module::start()
{
    //------------------------------------------------------------------
    // Create a thread that will invoke the virtual thread_func()
    std::thread t(&PHY_module::thread_func, this);
    thread_.swap(t);
    int status = pthread_setname_np(thread_.native_handle(), "msg_processing");
    if (status != 0) {
        NVLOGW_FMT(TAG, "PHY_module::msg_processing pthread_setname_np failed with error({})", std::strerror(status));
    }

    NVLOGI_FMT(TAG, "{}: test_type={}", __func__, test_type);
    if (test_type == 2)
    {
        // Tick unit test, start tick generator immediately
        set_tti_flag(true);
    }

    if (thread_cfg_.get() != nullptr) {

       sched_param sch;
       int policy;
       status = 0;
       status = pthread_getschedparam(thread_.native_handle(), &policy, &sch);
       if(status != 0)
       {
           NVLOGW_FMT(TAG, "msg_processing thread pthread_getschedparam failed with status : {}" ,std::strerror(status));
       }
       sch.sched_priority = thread_cfg_->sched_priority;
#ifdef ENABLE_SCHED_FIFO_ALL_RT
       status = pthread_setschedparam(thread_.native_handle(), SCHED_FIFO, &sch);
       if(status != 0)
       {
          NVLOGW_FMT(TAG, "msg_processing setschedparam failed with status : {}", std::strerror(status));
       }
#endif

       //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
       // Set thread CPU affinity
       cpu_set_t cpuset;
       CPU_ZERO(&cpuset);
       CPU_SET(thread_cfg_->cpu_affinity, &cpuset);
       status = pthread_setaffinity_np(thread_.native_handle(), sizeof(cpu_set_t), &cpuset);
       if(status)
       {
           NVLOGW_FMT(TAG, "msg_processing setaffinity_np  failed with status : {}", std::strerror(status));
       }
    }

    pthread_t thread_id;
    status=pthread_create(&thread_id, NULL, cell_update_thread_func, this);
    if(status)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create cell_update_thread_func failed with status : {}", std::strerror(status));
    }
    if(enable_se_sync_cmd){
        status=pthread_create(&thread_id, NULL, sfn_slot_sync_cmd_thread_func, this);
        if(status)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create sfn_slot_sync_cmd_thread_func failed with status : {}", std::strerror(status));
        }        
    }

    CuphyOAM::getInstance()->cell_eaxcids_update_callback = [this] (uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map) {
        oam_cell_eaxcids_update(mplane_id, eaxcids_ch_map);
    };

    CuphyOAM::getInstance()->cell_multi_attri_update_callback = [this] (uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res) {
        oam_cell_multi_attri_update(mplane_id, attrs, res);
    };

    if(AppConfig::getInstance().isCUSPortFailoverEnabled())
    {
        status = pthread_create(&thread_id, NULL, cus_conn_mgr_thread_func, this);
        if(status)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pthread_create cus_conn_mgr_thread_func failed with status : {}", std::strerror(status));
        }
    }
}

/**
    * Stop the PHY module
    *
    * Signals the module thread and tick generator to stop.
    */
void PHY_module::stop()
{
    stop_tick_generator();
    if (epoll_ctx_p)
    {
        epoll_ctx_p->terminate();
    }
}

////////////////////////////////////////////////////////////////////////
// PHY_module::recv_msg()
#ifdef ENABLE_L2_SLT_RSP
bool PHY_module::recv_msg()
{
    phy_mac_msg_desc smsg;
    static uint32_t last_processed_slot = 0xFFFFFFFF;

    while(transport_wrapper().rx_recv(smsg) >= 0)
    {
        if(simulated_cpu_stall_checkpoint(L2A_MSG_THREAD,-1))
            NVLOGI_FMT(TAG, "Simulate CPU Stall on L2A msg thread");

        sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&smsg);
        NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SFN {}.{}",
                ss_curr.u16.sfn, ss_curr.u16.slot, smsg.cell_id, smsg.msg_id, ss_msg.u16.sfn, ss_msg.u16.slot);

        // Reset L1 limit errors when we start processing a NEW slot
        // This prevents counter accumulation across slots
        if (ss_msg.u32 != SFN_SLOT_INVALID && ss_msg.u32 != last_processed_slot) {
            NVLOGD_FMT(TAG, "New slot detected: SFN {}.{} (was {}.{}), resetting L1 limit errors (PDSCH parsed was {})",
                       ss_msg.u16.sfn, ss_msg.u16.slot,
                       (last_processed_slot >> 16) & 0xFFFF, last_processed_slot & 0xFFFF,
                       group_limit_errors_.pdsch_errors.parsed);
            reset_l1_limit_errors();
            last_processed_slot = ss_msg.u32;
        }

        // For disordered FAPI in Multi-L2: new slot messages comes before processing current slot and updating ss_curr
        if(ss_msg.u32 != SFN_SLOT_INVALID && smsg.msg_id != 0x82 && ss_msg.u32 == get_next_sfn_slot(ss_curr).u32) {
            // Save next slot FAPI message to next_slot_fapi_cache, then continue
            if (next_slot_fapi_num < next_slot_fapi_cache.size()) {
                next_slot_fapi_cache[next_slot_fapi_num++] = smsg;
                NVLOGW_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SFN {}.{} - early received next slot message",
                        ss_curr.u16.sfn, ss_curr.u16.slot, smsg.cell_id, smsg.msg_id, ss_msg.u16.sfn, ss_msg.u16.slot);
            } else {
                NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} SFN {}.{} - next_slot_fapi_cache out of boundary, drop",
                        ss_curr.u16.sfn, ss_curr.u16.slot, smsg.cell_id, smsg.msg_id, ss_msg.u16.sfn, ss_msg.u16.slot);
                transport_wrapper().rx_release(smsg);
            }
            continue;
        }

        if(phy_refs_[smsg.cell_id].get().on_msg(smsg))
        {
            // Release the NVIPC buffers if FAPI message handle finished in on_msg()
            transport_wrapper().rx_release(smsg);
        }

        // Process slot command at one of below cases:
        // (1) Loopback SLOT.ind was received
        // (2) All active cells received SLOT.resp for current SFN/SLOT
        if (smsg.msg_id == 0x82 || (active_cell_bitmap && (fapi_eom_rcvd_bitmap & active_cell_bitmap) == active_cell_bitmap))
        {
            bool slot_end_rcvd = smsg.msg_id == 0x82 ? false : true;
            process_phy_commands(slot_end_rcvd);
            fapi_eom_rcvd_bitmap = 0;
            new_slot_ = true;
            is_ul_slot_ = false;
            is_dl_slot_ = false;
            is_csirs_slot_ = false;

            // Process cached new slot FAPI messages and reset the counter
            for (uint32_t i = 0; i < next_slot_fapi_num; i++)
            {
                auto& desc = next_slot_fapi_cache[i];
                if(phy_refs_[desc.cell_id].get().on_msg(desc))
                {
                    // The IPC message handle finished, release the buffers explicitly
                    transport_wrapper().rx_release(desc);
                }
            }
            next_slot_fapi_num = 0;
            // Log PDSCH counter before reset for debugging
            NVLOGD_FMT(TAG, "Resetting L1 limit errors at slot end - PDSCH parsed={}, errors={}",
                       group_limit_errors_.pdsch_errors.parsed,
                       group_limit_errors_.pdsch_errors.errors);
            reset_l1_limit_errors();
        }
    }

    return true;
}
#else
bool PHY_module::recv_msg()
{
    tti_event_count++;

    phy_mac_msg_desc smsg;
    // NVLOGI_FMT(TAG, "tti_event_count={}", tti_event_count);

        while(transport_wrapper().rx_recv(smsg) >= 0)
        {
            simulated_cpu_stall_checkpoint(L2A_MSG_THREAD,0);
            sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&smsg);
            if(ss_msg.u32 == SFN_SLOT_INVALID)
            {
                if (ipc_sync_mode != SYNC_MODE_PER_SLOT) {
                    tti_event_count --;
                }
                /// CONFIG, START and STOP Messages have no SFN/SF
                if(phy_refs_[smsg.cell_id].get().on_msg(smsg))
                {
                    // The IPC message handle finished, release the buffers explicitly
                    transport_wrapper().rx_release(smsg);
                }
            } else {
                if (num_msgs >= slot_msgs_received.size())
                {
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "SFN {}.{}: received msg_id=0x{:02X} for SFN {}.{}, num_msgs={} exceeds array boundary", ss_curr.u16.sfn, ss_curr.u16.slot, smsg.msg_id, ss_msg.u16.sfn, ss_msg.u16.slot, num_msgs);
                    transport_wrapper().rx_release(smsg);
                    continue;
                }
                slot_msgs_received[num_msgs] = smsg;
                num_msgs++;
                // NVLOGI_FMT(TAG, "slot_msgs_received[num_msgs].cell_id]= {} num_msgs={}",slot_msgs_received[num_msgs -1].cell_id, num_msgs);
            }
        }

        NVLOGD_FMT(TAG, "SFN {}.{} recv_msg: tti_event_count={}",
                ss_curr.u16.sfn, ss_curr.u16.slot, tti_event_count);

        if ((ipc_sync_mode == SYNC_MODE_PER_SLOT && tti_event_count == 1) ||(ipc_sync_mode == SYNC_MODE_PER_CELL && tti_event_count == num_cells_active) )
        // if (tti_event_count == num_cells_active)
        {
            for (uint i = 0; i < num_msgs; i++) {
                auto& desc = slot_msgs_received[i];
                if(phy_refs_[desc.cell_id].get().on_msg(desc))
                {
                    // The IPC message handle finished, release the buffers explicitly
                    transport_wrapper().rx_release(desc);
                }
            }
            if (num_msgs) {
                process_phy_commands(true);
            }
            new_slot_ = true;
            is_ul_slot_ = false;
            is_dl_slot_ = false;
            is_csirs_slot_ = false;
            num_msgs = 0;
            tti_event_count = 0;
        }
    return true;
}
#endif

#define epoll_fd //use epoll fd instead of semaphore

void PHY_module::oam_cell_eaxcids_update(uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map)
{
    PHYDriverProxy::getInstance().l1_cell_update_cell_config(mplane_id, eaxcids_ch_map);
}

void PHY_module::oam_cell_multi_attri_update(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res)
{
    //NVLOGC_FMT(TAG, "Update cell: mplane_id={} ", config->cell_id);
    string gps_alpha_key = "gps_alpha";
    string gps_beta_key  = "gps_beta";

    bool update_gps_alpha_beta_l1 = false;
    if(attrs.find(gps_alpha_key) != attrs.end())
    {
        if(this->num_cells_active)
        {
            NVLOGC_FMT(TAG, "gps_alpha has to be configured when all cells are in Idle state, ignoring...");
        }
        else
        {
            this->gps_alpha_ = attrs[gps_alpha_key];
            NVLOGC_FMT(TAG, "Update gps_alpha_={} ", this->gps_alpha_);
            update_gps_alpha_beta_l1 = true;
        }
        attrs.erase(gps_alpha_key);
    }
    if(attrs.find(gps_beta_key) != attrs.end())
    {
        if(this->num_cells_active)
        {
            NVLOGC_FMT(TAG, "gps_beta has to be configured when all cells are in Idle state, ignoring...");
        }
        else
        {
            this->gps_beta_ = attrs[gps_beta_key];
            NVLOGC_FMT(TAG, "Update gps_beta_={} ", this->gps_beta_);
            update_gps_alpha_beta_l1 = true;
        }
        attrs.erase(gps_beta_key);
    }
    if(update_gps_alpha_beta_l1)
    {
        update_gps_alpha_beta_l1 = false;
        PHYDriverProxy::getInstance().l1_update_gps_alpha_beta(this->gps_alpha_, this->gps_beta_);
    }

    if(!attrs.empty())
    {
        if(mplane_id == 0xFF)//For all cells
        {
            for(auto& cfg : PHYDriverProxy::getInstance().getMPlaneConfigList())
            {
                PHYDriverProxy::getInstance().l1_cell_update_cell_config(cfg.mplane_id, attrs, res);
            }
        }
        else
        {
            PHYDriverProxy::getInstance().l1_cell_update_cell_config(mplane_id, attrs, res);
        }
    }
}

void* PHY_module::cell_update_thread_func(void* arg)
{
    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGC_FMT(TAG, "{}: OAM thread affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);

    if(pthread_setname_np(pthread_self(), "oam_cell_update") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    sleep(4);//wait for readiness of OAM server

    PHY_module *phy_mod = (PHY_module *)arg;

    while(1)
    {
         CuphyOAM *oam = CuphyOAM::getInstance();
         CuphyOAMCellConfig *config;
         while ((config = oam->get_cell_config()) != nullptr)
         {
            if (config->update_network_cfg)
            {
                NVLOGC_FMT(TAG, "Update cell: mplane_id={} dst_mac={} vlan_tci=0x{:X}",
                       config->cell_id, config->dst_mac_addr.c_str(), config->vlan_tci);
                if (PHYDriverProxy::getInstance().l1_cell_update_cell_config(config->cell_id,config->dst_mac_addr, config->vlan_tci)) {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to update cell {} dst_mac={} vlan_tci=0x{:X}",
                            config->cell_id, config->dst_mac_addr.c_str(), config->vlan_tci);
                }
            }
            else if (config->multi_attrs_cfg)
            {
                phy_mod->oam_cell_multi_attri_update(config->cell_id, config->attrs, config->res);
            }
            oam->free_cell_config(config);
         }
         usleep(100000);
    }
    NVLOGI_FMT(TAG, "cell_update_thread_func exit");
    return nullptr;
}

void* PHY_module::sfn_slot_sync_cmd_thread_func(void* arg)
{
    // Switch to a low_priority_core to avoid blocking time critical thread
    auto& appConfig = AppConfig::getInstance();
    auto low_priority_core = appConfig.getLowPriorityCore();
    NVLOGD_FMT(TAG, "{}: OAM thread affinity set to cpu core {}", __func__, low_priority_core);
    nv_assign_thread_cpu_core(low_priority_core);
    nanoseconds ts_now;

    if(pthread_setname_np(pthread_self(), "sfn_slot_sync_cmd") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    sleep(4);//wait for readiness of OAM server

    PHY_module *phy_mod = (PHY_module *)arg;

    while(1)
    {
         CuphyOAM *oam = CuphyOAM::getInstance();
         CuphyOAMSfnSlotSyncCmd* sfn_slot_sync_cmd;
         while ((sfn_slot_sync_cmd = oam->get_sfn_slot_sync_cmd()) != nullptr)
         {
            if(phy_mod->target_node==0) //Target:UE, Source : DU
            {
                phy_mod->sync_rcvd_from_ue=true;
                ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
                NVLOGI_FMT(TAG, "{} ==> sync_done from cmd: {}, sync_rcvd_from_ue: {} curr_time: {}",__func__,sfn_slot_sync_cmd->sync_done,phy_mod->sync_rcvd_from_ue.load(),ts_now.count());
            }
            else //Target:DU, Source : UE
            {
                phy_mod->sync_rcvd_from_du=true;
                ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
                NVLOGI_FMT(TAG, "{} ==> sync_done from cmd: {}, sync_rcvd_from_du: {} curr_time: {}",__func__,sfn_slot_sync_cmd->sync_done,phy_mod->sync_rcvd_from_du.load(),ts_now.count());
            }                
            oam->free_sfn_slot_sync_cmd(sfn_slot_sync_cmd);
         }
         usleep(100000);
    }
    NVLOGI_FMT(TAG, "sfn_slot_sync_cmd_thread_func exit");
    return nullptr;
}


////////////////////////////////////////////////////////////////////////
// PHY_module::msg_processing()
#ifdef ENABLE_L2_SLT_RSP
void PHY_module::msg_processing()
{
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    try
    {
        // The while loop in recv_msg() returns only when there is no more message to receive. No need to add additional polling here.
        recv_msg();

#ifdef FORCE_SLEEP_OF_L2A_MSG_THREAD
        // Add sleep to avoid system blocking by SCHED_FIFO + polling thread
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
#endif
    }
    catch(std::exception& e)
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PHY_module::msg_processing() exception: {}", e.what());
    }
    catch(...)
    {
        // Need rethrow abi::__forced_unwind exception to avoid "FATAL: exception not rethrown" in pthread_join()
        NVLOGC_FMT(TAG, "PHY_module::msg_processing() re-throwing unknown exception");
        throw;
    }
}
#else
void PHY_module::msg_processing()
{
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    try
    {
#ifndef epoll_fd
        transport_wrapper_.rx_wait();
        NVLOGD_FMT(TAG, "nv::PHY_module::msg_processing(): transport.rx_wait() returned");
#else
        transport_wrapper_.get_value();
#endif
        recv_msg();

#ifndef epoll_fd
        usleep(1000);
#endif
    }
    catch(std::exception& e)
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PHY_module::msg_processing() exception: {}", e.what());
    }
    catch(...)
    {
        NVLOGF_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "PHY_module::msg_processing() unknown exception");
    }
}
#endif

////////////////////////////////////////////////////////////////////////
// PHY_module::thread_func()
#ifdef ENABLE_L2_SLT_RSP

void PHY_module::thread_func()
{
    // TODO: load thread_cfg_ from yaml file and pass thread_cfg_->cpu_affinity
    //assign_thread_cpu_core(21);
    config_thread_property(*thread_cfg_);
    nvlog_fmtlog_thread_init("msg_processing");
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, thread_cfg_->cpu_affinity);

    while(!pExitHandler.test_exit_in_flight())
    {
        msg_processing();
    }
    NVLOGC_FMT(TAG, "Thread msg_processing exiting");
}
#else
void PHY_module::thread_func()
{
    // TODO: load thread_cfg_ from yaml file and pass thread_cfg_->cpu_affinity
    //assign_thread_cpu_core(21);
    config_thread_property(*thread_cfg_);
    nvlog_fmtlog_thread_init("msg_processing");
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, thread_cfg_->cpu_affinity);
    // enable dynamic memory allocation tracing in real-time code path
    // Use LD_PRELOAD=<special .so> when running cuphycontroller, otherwise this does nothing.
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

#ifdef epoll_fd
    epoll_ctx_p->start_event_loop();
#else

#if 0
    while(1)
#else
    do
#endif
    {
        msg_processing();
#if 0
    }
#else
    } while(0); // Just run once
#endif

#endif
    NVLOGD_FMT(TAG, "PHY_module::thread_func() returning");
}
#endif


bool PHY_module::check_time_threshold(std::chrono::nanoseconds now, uint16_t slot, bool slot_end_rcvd)
{
    //If slot_end_rcvd == false, the subtract 1 from slot because ss_curr is updated to the new slot
    //while this function is called for l1_enqueue of previous slot
    if(!slot_end_rcvd)
        slot = (slot + nv::mu_to_slot_in_sf(get_mu_highest()) - 1)%10;

    NVLOGD_FMT(TAG, "{} slot = {} now={} l1_slot_ind_tick_={} l2a_allowed_latency_={} diff={}.",__func__,
         slot, now.count(), l1_slot_ind_tick_[slot%10].count(),l2a_allowed_latency_,(now - l1_slot_ind_tick_[slot%10]).count());

    if((now - l1_slot_ind_tick_[slot%10]).count() > (nv::mu_to_ns(get_mu_highest()) + l2a_allowed_latency_))
    {
        // Warmup latency seen for first UL and first DL slot
        if(first_dl_slot_ || first_ul_slot_)
            NVLOGI_FMT(TAG, "L2+L2A processing taking > 500us (l2+l2a_duration={} ns) for first UL/DL slot. Drop slot.",
                (now - l1_slot_ind_tick_[slot%10]).count());
        else
            NVLOGW_FMT(TAG, "L2+L2A processing taking > 500us for slot={} now={} l1_slot_ind_tick={} diff={} ns l2a_allowed_latency={} ns. Drop slot",
                slot, now.count(),l1_slot_ind_tick_[slot%10].count(),(now - l1_slot_ind_tick_[slot%10]).count(), l2a_allowed_latency_);
        return false;
    }
    else
        return true;
}
/*
 processes the cell command for the slot
 For the cell configure and cell start,
 they are handled in PHY_instance itself
*/
    void PHY_module::process_phy_commands(bool slot_end_rcvd)
    {
#ifndef ENABLE_L2_SLT_RSP
        tti_event_count --;

        if (ss_last.u32 != SFN_SLOT_INVALID)
        {
            if(ss_last.u32 == ss_curr.u32 || get_slot_interval(ss_last, ss_curr) > FAPI_SFN_MAX)
            {
                NVLOGD_FMT(TAG, "{}: SFN {}.{} error slots: ss_last=SFN {}.{} slot_latency={} tti_event_counter={}",
                        __func__, ss_curr.u16.sfn, ss_curr.u16.slot, ss_last.u16.sfn, ss_last.u16.slot, get_slot_interval(ss_curr, ss_tick.load()), tti_event_count);
            }
        }
#endif
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
        // Each PHY_instance produces cell_sub_command
        // The cell_sub_command are added to slot_cmd
        slot_command_api::slot_command& slot_cmd = slot_command();

        uint32_t phy_list_size = 0;
        uint32_t phy_list[MAX_CELLS_PER_CELL_GROUP];

        bool partial_cmd = false;
        bool pdsch_exist = false;
        cell_group_command& group_cmd = *(group_command());
        auto group_cmd_size = group_cmd.channel_array_size;

        if (!group_cmd_size) {
            NVLOGD_FMT(TAG, "SFN {}.{} {}: NO channel", ss_curr.u16.sfn, ss_curr.u16.slot, __func__);
#ifndef ENABLE_L2_SLT_RSP
            ss_last.u32 = ss_curr.u32;
#endif  
            if(!partial_cmd)
            {
                for (auto& phy: phy_refs_)
                {
                    phy.get().reset_slot(partial_cmd);
                }
            }
            return;
        }
        auto slotType = slot_command_api::slot_type::SLOT_NONE;
        slotType = group_cmd.slot.type;

        slot_command_api::pdsch_params* dlParam = group_cmd.pdsch.get();
        if(dlParam->cell_grp_info.nCws != 0)
        {
            for(uint32_t idx = 0; idx < dlParam->cell_grp_info.nCells; idx++)
            {
                auto staticIdx = dlParam->cell_dyn_info[idx].cellPrmStatIdx;
                uint32_t phy_id = stat_prm_idx_to_cell_id_map[staticIdx];
                auto& cell_cmd = cell_sub_command(phy_id);
                if (dlParam->tb_data.pTbInput[idx] == nullptr)
                {
                    partial_cmd = true;
                    NVLOGW_FMT(TAG, "Current SFN {}.{}, Previous slot received={} {}: cell_id={} channels={} - invalid PDSCH pTbInput={} data_buf={}",
                            static_cast<unsigned>(ss_curr.u16.sfn),
                            static_cast<unsigned>(ss_curr.u16.slot),
                            slot_end_rcvd,
                            __func__,
                            phy_id,
                            cell_cmd.channel_array_size,
                            reinterpret_cast<void*>(dlParam->tb_data.pTbInput[idx]),
                            reinterpret_cast<void*>(phy_refs_[staticIdx].get().cur_dl_msg.data_buf));
                    continue;
                }

                if(phy_list_size >= MAX_CELLS_PER_CELL_GROUP)
                {
                    partial_cmd = true;
                    NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "phy_list_size {} >= MAX_CELLS_PER_CELL_GROUP {}", phy_list_size, MAX_CELLS_PER_CELL_GROUP);
                    continue;
                }
                phy_list[phy_list_size] = stat_prm_idx_to_cell_id_map[staticIdx];
                ++phy_list_size;
                NVLOGD_FMT(TAG, "SFN {}.{} {}: cell_id={} channels={} - command added",
                    ss_curr.u16.sfn, ss_curr.u16.slot, __func__, phy_id, cell_cmd.channel_array_size);
            } // for i=0->nCells
        } // if slot is DL or S

        tick_lock.lock();
        uint32_t slot_interval = get_fapi_latency(ss_curr);
        nanoseconds curr_tick = current_tick_.load();
        tick_lock.unlock();
        slot_cmd.tick_original = curr_tick - std::chrono::nanoseconds(slot_interval * mu_to_ns(tick_updater_.mu_highest_));
        // NVLOGD_FMT(TAG, "{}: SFN {}.{} curr tick {} slot_interval {}", __func__, ss_curr.u16.sfn, ss_curr.u16.slot, curr_tick, slot_interval);

        std::chrono::nanoseconds now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
        l2a_end_tick(now);
        int64_t latency = std::chrono::duration_cast<std::chrono::nanoseconds>(now - curr_tick).count();
        latency += slot_interval * mu_to_ns(tick_updater_.mu_highest_);
        slot_latency->add(slot_latency, latency);
        std::size_t cells_size = slot_cmd.cells.size();
        NVLOGI_FMT(TAG, "SFN {}.{} {} cells={} cmd_size={} slot_latency={}",
                ss_curr.u16.sfn, ss_curr.u16.slot, __func__, phy_list_size, cells_size, latency);

        int ret = -1;
        int to_clean = 0;
        bool clean_srs_ind_buffers = false;

        // publish to PHYDriver
        if ((check_time_threshold(now,ss_curr.u16.slot,slot_end_rcvd)) && (phy_list_size <= cells_size) && !(partial_cmd))
        {
            auto start_process_command_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
            ret = PHYDriverProxy::getInstance().l1_enqueue_phy_work(slot_cmd);
            to_clean = 1;
            auto end_process_command_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch());
            auto diff = end_process_command_time - start_process_command_time;
            NVLOGI_FMT(TAG, "SFN {}.{} {}: l1_enqueue_phy_work after status = {} slot cmd size ={}  phy_refs : size = {} l1_enqueue_phy_work start: {} l1_enqueue_phy_work duration: {} ns UL {} DL {}",
                 slot_cmd.cell_groups.slot.slot_3gpp.sfn_, slot_cmd.cell_groups.slot.slot_3gpp.slot_, __func__, ret,
                 cells_size, phy_refs_.size(), start_process_command_time.count(), diff.count(), is_ul_slot_ ? 1 : 0, is_dl_slot_ ? 1 : 0);

            //Calculate the expected t0, which can be used to verify the actual tick time
            int current_sfn = slot_cmd.cell_groups.slot.slot_3gpp.sfn_;
            int current_slot = slot_cmd.cell_groups.slot.slot_3gpp.slot_;
            std::chrono::nanoseconds current_t0_timestamp(sfn_to_tai(current_sfn, current_slot, current_tick_list_[ss_curr.u16.slot%10].count() + AppConfig::getInstance().getTaiOffset(), gps_alpha_, gps_beta_, tick_updater_.mu_highest_) - AppConfig::getInstance().getTaiOffset());

            //Calculate several useful statistics to be made human-readable in L2A.PROCESSING_TIMES message
            auto tick_advance = current_t0_timestamp - current_tick_list_[ss_curr.u16.slot%10];
            auto l1_slot_ind_delay = l1_slot_ind_tick_[ss_curr.u16.slot%10] - current_tick_list_[ss_curr.u16.slot%10];
            auto l2_estimate = last_fapi_msg_tick_ - l1_slot_ind_tick_[ss_curr.u16.slot%10];
            auto fapi_proc_dur = l2a_end_tick_ - l2a_start_tick_;
            auto l1_enqueue_duration = end_process_command_time - l2a_end_tick_;

            NVLOGI_FMT(TAG_PROCESSING_TIMES, "SFN {}.{} {}: cells={} cmd_size={} Tick Advance={}ns L1 Slot Ind Delay={}ns L2 Estimate={}ns FAPI Proc Dur={}ns L1 Enqueue Dur={}ns l1_slot_ind_tick={} l2a_start_time={} l2a_end_time={} last_fapi_msg_tick={} l1_enqueue_complete_time={} UL={} DL={} CSIRS={}",
                slot_cmd.cell_groups.slot.slot_3gpp.sfn_, slot_cmd.cell_groups.slot.slot_3gpp.slot_, __func__, phy_list_size, cells_size,
                tick_advance.count(),//Tick Advance (ns)
                l1_slot_ind_delay.count(),//L1 Slot Ind Delay (ns)
                l2_estimate.count(),//L2 Estimate (ns)
                fapi_proc_dur.count(),// FAPI Proc Dur (ns)
                l1_enqueue_duration.count(),// L1 Enqueue Dur (ns)
                l1_slot_ind_tick_[ss_curr.u16.slot%10].count(),l2a_start_tick_.count(),l2a_end_tick_.count(),last_fapi_msg_tick_.count(),end_process_command_time.count(),
                is_ul_slot_ ? 1 : 0, is_dl_slot_ ? 1 : 0, is_csirs_slot_ ? 1 : 0);
#ifndef ENABLE_L2_SLT_RSP
            update_slot_cmds_indexes();
#endif
        }
#ifndef ENABLE_L2_SLT_RSP
        else if (!slot_interval) {
            NVLOGD_FMT(TAG, "Current SFN {}.{} is same as Last SFN {} {} not enqueueing", ss_curr.u16.sfn, ss_curr.u16.slot, ss_last.u16.sfn, ss_last.u16.slot);
            return;
        }
#endif
        else {
            to_clean = 1;
            // Need to clean the nvIPC buffers allocated for SRS.IND
            clean_srs_ind_buffers = true;
            auto slot = ss_curr.u16.slot;
            if(!slot_end_rcvd)
                slot = (slot + nv::mu_to_slot_in_sf(get_mu_highest()) - 1)% (nv::mu_to_slot_in_sf(get_mu_highest()));
            NVLOGW_FMT(TAG, "Dropping the slot command for SFN {}.{}", ss_curr.u16.sfn, slot);
            PHYDriverProxy::getInstance().l1_resetBatchedMemcpyBatches(); //used to guard against case when the slot is dropped and PDSCH H2D copies are batched with preponing (prepone_h2d_copy=1) enabled and without separate copy thread (enable_h2d_copy_thread in cuphycontroller yaml = 0)
        }

        if (to_clean == 1) {
            // Clean up slot command and FAPI message
            for (int i = 0 ; i < phy_list_size; i++)
            {
                uint32_t phy_id = phy_list[i];
                phy_mac_msg_desc &ipc = phy_refs_[phy_id].get().cur_dl_msg;

                if (ipc.data_buf != nullptr)
                {
                    if (ret == 0)
                    {
                        std::lock_guard<std::mutex> lock(dl_tbs_lock);
                        dl_tbs_queue_.push(ipc);
                        NVLOGI_FMT(TAG, "DL TB Queue Size = {}", dl_tbs_queue_.size());
                    }
                    else
                    {
                        NVLOGI_FMT(TAG, "Clearing the previous DL TB due to l1_enqueue_phy_work failure");
                        transport_wrapper().rx_release(reinterpret_cast<nv::phy_mac_msg_desc&>(ipc));
                    }
                }
                phy_refs_[phy_id].get().cur_dl_msg.reset();
            }
            if(ret==-2){
                std::array<int32_t,MAX_CELLS_PER_SLOT> cell_id_list;
                std::fill(cell_id_list.begin(),cell_id_list.end(),-1);
                int32_t index=0;
                for (auto& phy: phy_refs_)
                {
                    phy.get().send_phy_l1_enqueue_error_indication(slot_cmd.cell_groups.slot.slot_3gpp.sfn_,slot_cmd.cell_groups.slot.slot_3gpp.slot_,is_ul_slot_,cell_id_list,index);
                }
            }
            // To free nvipc reserved buffer for SRS.IND
            if(clean_srs_ind_buffers)
            {
                slot_command_api::srs_params* srs_params = slot_cmd.cell_groups.get_srs_params();
                if(srs_params != nullptr)
                {
                    NVLOGW_FMT(TAG, "SFN {}.{}: releasing pre-allocated nvIPC buffers for SRS INDs", ss_curr.u16.sfn, ss_curr.u16.slot);
                    for(int i =0; i < srs_params->cell_grp_info.nCells; i++)
                    {
                        int cell_index = srs_params->cell_index_list[i];
                        for(int j =0; j <= srs_params->num_srs_ind_indexes[i]; j++)
                        {
                            NVLOGD_FMT(TAG, "{}.{}: releasing SRS IND cell_index {} srs_indications[{}][{}] = msg_id {}, cell_id {} msg_len {} data_len {} data_pool {} msg_buf {} data_buf {}", 
                                            ss_curr.u16.sfn, ss_curr.u16.slot, cell_index, i, j, 
                                            srs_params->srs_indications[i][j].msg_id, 
                                            srs_params->srs_indications[i][j].cell_id, 
                                            srs_params->srs_indications[i][j].msg_len, 
                                            srs_params->srs_indications[i][j].data_len, 
                                            srs_params->srs_indications[i][j].data_pool,
                                            srs_params->srs_indications[i][j].msg_buf,
                                            srs_params->srs_indications[i][j].data_buf);
                            nv::phy_mac_msg_desc msg_desc(srs_params->srs_indications[i][j]);
                            transport(cell_index).tx_release(msg_desc);
                        }
                        srs_params->num_srs_ind_indexes[i] = 0;
                    }
                }
            }
        }

#ifdef ENABLE_L2_SLT_RSP
        update_slot_cmds_indexes();
        group_command()->reset();
#else
        ss_last.u32 = ss_curr.u32;
#endif
        uint32_t i = 0;
        for (auto& phy: phy_refs_)
        {
            phy.get().reset_slot(partial_cmd);
#ifdef ENABLE_L2_SLT_RSP
            cell_sub_command(i).reset();
            i++;
#endif
        }

        if(is_ul_slot_)
            first_ul_slot_ = false;

        if(is_dl_slot_)
            first_dl_slot_ = false;

        last_fapi_msg_tick_ = std::chrono::nanoseconds(0);
    }

    void PHY_module::tick_received(std::chrono::nanoseconds& tick)
    {
        if (!all_cells_configured) {
            // Skip until all cells are configured
            return;
        }

        nanoseconds ts_now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());

        NVLOGI_FMT(TAG_TICK_TIMES,"SFN {}.{} current_time={}, tick={}",
                   static_cast<unsigned>(tick_updater_.slot_info_.sfn_),
                   static_cast<unsigned>(tick_updater_.slot_info_.slot_),
                   ts_now.count(),
                   tick.count());

        if (first_tick)
        {
            first_tick = false;

            uint64_t slot_temp;
            uint8_t SLOT_MAX=20;
            // Initialize SFN.  Convert from system time (which is synchronized to TIA/PTP time) to GPS time
            // NOTE: Calculation assumes O-RAN WG4 CUS 9.7.2 parameters alpha=0 and beta=0
            uint64_t tia_to_gps_offset_ns = (315964800ULL + 19ULL) * 1000000000ULL;
            int64_t gps_offset = ((gps_beta_ * 1000000000LL) / 100LL) + ((gps_alpha_ * 10000ULL) / 12288ULL);
            uint64_t framePeriodInNanoseconds = 10000000;
            uint64_t slotPeriodInNanoseconds = 500000;
            uint64_t gps_ns = tick.count() - (tia_to_gps_offset_ns) + AppConfig::getInstance().getTaiOffset();
            uint64_t numerator = gps_ns - gps_offset;
            tick_updater_.slot_info_.sfn_ = ((numerator / framePeriodInNanoseconds)) % FAPI_SFN_MAX;
            /*Slot value needs to be deduced assuming random first_tick value*/
            slot_temp=  numerator - (framePeriodInNanoseconds*(numerator / framePeriodInNanoseconds));
            slot_temp = (slot_temp/slotPeriodInNanoseconds)%SLOT_MAX;
            if((slot_temp+tick_updater_.slot_advance_)>=SLOT_MAX) //Advance SFN value by 1 if slot_temp exceeds SLOT_MAX
            {
                tick_updater_.slot_info_.sfn_ = (tick_updater_.slot_info_.sfn_+1)%FAPI_SFN_MAX;
            }
            tick_updater_.slot_info_.slot_ = (slot_temp+tick_updater_.slot_advance_)%SLOT_MAX;
            NVLOGI_FMT(TAG,"FIRST tick received: SFN {}.{} tick={} seconds since epoch = {} nanoseconds = {}",
                 static_cast<unsigned>(tick_updater_.slot_info_.sfn_),
                 static_cast<unsigned>(tick_updater_.slot_info_.slot_),
                 tick.count(),
                 tick.count() / 1000000000ULL,
                 tick.count() % 1000000000ULL);
        }

        // Stop tick generator and skip sending SLOT.indication if app exit is in flight
        if (pExitHandler.test_exit_in_flight())
        {
            NVLOGC_FMT(TAG, "{}: Stop tick generator because app exit is in flight", __FUNCTION__);
            tti_module_.stop_tick_generator();
            return;
        }

        sfn_slot_t sfn_slot;
        sfn_slot.u16.sfn = tick_updater_.slot_info_.sfn_;
        sfn_slot.u16.slot = tick_updater_.slot_info_.slot_;

        tick_lock.lock();
        ss_tick.store(sfn_slot);
        current_tick_.store(tick);
        tick_lock.unlock();
        // NVLOGI_FMT(TAG, "{}: SFN {}.{} curr tick {}", __func__, sfn_slot.u16.sfn, sfn_slot.u16.slot, tick);
        int64_t tick_err = ts_now.count() - tick.count();
        if (test_type != 2) // Do not send SLOT.indication in tick unit test
        {
            if(tick_err < timer_thread_wakeup_threshold_)
            {
                current_tick_list_[sfn_slot.u16.slot%10] = tick;
                l1_slot_ind_tick_[sfn_slot.u16.slot%10] = ts_now;
                NVLOGI_FMT(TAG,"{}: SFN {}.{} tick={} ts_now={} tick_err={} threshold={}",
                __func__,static_cast<unsigned>(tick_updater_.slot_info_.sfn_),
                static_cast<unsigned>(tick_updater_.slot_info_.slot_),tick.count(), ts_now.count(),tick_err,timer_thread_wakeup_threshold_);
                phy_refs_[0].get().send_slot_indication(tick_updater_.slot_info_);
            }
            else
            {
                NVLOGC_FMT(TAG, "{}: tick_err {}", __func__, tick_err);
                current_tick_list_[sfn_slot.u16.slot%10] = std::chrono::nanoseconds(0);
                l1_slot_ind_tick_[sfn_slot.u16.slot%10] = std::chrono::nanoseconds(0);
                phy_refs_[0].get().send_slot_error_indication(tick_updater_.slot_info_);
            }
        }

        // Normal mode, integrating phydriver mode
        if (test_type == 0)
        {
            CuphyOAM *oam = CuphyOAM::getInstance();
            oam->status_sfn_slot = (tick_updater_.slot_info_.sfn_<<8) | tick_updater_.slot_info_.slot_;
        }

        if (1)
        {
            static int count = 0;
            if (count == 0)
            {
                NVLOGD_FMT(TAG,"tick received: SFN {}.{} tick={} seconds since epoch = {} nanoseconds = {}",
                     static_cast<unsigned>(tick_updater_.slot_info_.sfn_),
                     static_cast<unsigned>(tick_updater_.slot_info_.slot_),
                     tick.count(),
                     tick.count() / 1000000000ULL,
                     tick.count() % 1000000000ULL);
            }
            count++;
            if (count >= 20*100)
            {
                count = 0;
            }
        }

        tick_updater_();

        // Print statistics of the tick interval
        tick_logger->add(tick_logger, tick_err);
    }

    // Get next slot SFN/SLOT
    sfn_slot_t PHY_module::get_next_sfn_slot(sfn_slot_t& ss)
    {
        uint16_t slot_per_frame = nv::mu_to_slot_in_sf(tick_updater_.mu_highest_);

        sfn_slot_t next = ss;
        next.u16.slot++;
        if(next.u16.slot >= slot_per_frame)
        {
            next.u16.slot = 0;
            next.u16.sfn  = next.u16.sfn >= FAPI_SFN_MAX - 1 ? 0 : next.u16.sfn + 1;
        }
        return next;
    }

    uint32_t PHY_module::get_slot_interval(sfn_slot_t ss_old, sfn_slot_t ss_new)
    {
        // This is the most likely case, add for quick return
        if (ss_old.u32 == ss_new.u32)
        {
            return 0;
        }

        uint32_t mu = tick_updater_.mu_highest_;
        uint32_t slot_per_frame = nv::mu_to_slot_in_sf(mu);

        uint32_t new_slots = ss_new.u16.sfn * slot_per_frame + ss_new.u16.slot;
        uint32_t old_slots = ss_old.u16.sfn * slot_per_frame + ss_old.u16.slot;

        // Old SFN/SLOT should not latter than new SFN/SLOT
        if (old_slots > new_slots)
        {
            new_slots += FAPI_SFN_MAX * slot_per_frame;
        }
        return new_slots - old_slots;
    }

    uint32_t PHY_module::get_fapi_latency(sfn_slot_t ss_msg)
    {
        return get_slot_interval(ss_msg, ss_tick.load());
    }

    void PHY_module::send_call_backs()
    {
        std::call_once(cb_flag, [this]() {
            auto& phy = phy_refs_[0].get();
            phy.create_ul_dl_callbacks(callbacks_);
            /// Send to L1
            PHYDriverProxy::getInstance().l1_set_output_callback(callbacks_);
        });

    }

    void PHY_module::create_cell_update_call_back() {
        std::call_once(cell_update_flag, [this]() {
            cell_update_cb_fn = [this] (int32_t mplane_id, uint8_t response_code) {
                if(mplane_id < 0)
                {
                    NVLOGW_FMT(TAG, "{}: invalid mplane_id={}", __func__, mplane_id);
                    return;
                }
                nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
                for(uint32_t cell_id=0; cell_id < phy_refs_.size(); cell_id++)
                {
                    ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(cell_id);
                    if(mplane.mplane_id == mplane_id)
                    {
                        // phy_refs_[cell_id].get().send_cell_config_response(cell_id, response_code);
                        phy_refs_[cell_id].get().handle_cell_config_response(cell_id, response_code);
                        return;
                    }
                }
                NVLOGW_FMT(TAG, "{}: no cell-id found with mplane_id={}", __func__, mplane_id);
            };
        });
    }

    void PHY_module::on_dl_tb_processed()
    {
        std::lock_guard<std::mutex> lock(dl_tbs_lock);
        NVLOGD_FMT(TAG, "Queue Size before = {}", dl_tbs_queue_.size());
        if (dl_tbs_queue_.size() > 0)
        {
            phy_mac_msg_desc& ipc_msg = dl_tbs_queue_.front();
            transport_wrapper().rx_release(ipc_msg);
            dl_tbs_queue_.pop();
            NVLOGD_FMT(TAG, "on_dl_tb_processed Queue size={}", dl_tbs_queue_.size());
        }
    }


    // TODO: this function requires that PDSCH commands processed and callback function called strictly in order, is this true?
    void PHY_module::on_dl_tb_processed(const slot_command_api::pdsch_params* params)
    {
        uint32_t i = params->cell_grp_info.nCells;
        std::lock_guard<std::mutex> lock(dl_tbs_lock);

        NVLOGD_FMT(TAG, "{}: size={} nCells={}", __func__, dl_tbs_queue_.size(), i);
        if (dl_tbs_queue_.size() < i)
        {
            i = dl_tbs_queue_.size();
        }

        while(i)
        {
            phy_mac_msg_desc& ipc_msg = dl_tbs_queue_.front();
            transport_wrapper().rx_release(ipc_msg);
            dl_tbs_queue_.pop();
            i--;
        }
    }

    // TODO: this function requires that DL TTI commands processed and callback function called strictly in order, is this true?
    void PHY_module::on_dl_tti_processed()
    {
        uint32_t i = num_cells_active;
        std::lock_guard<std::mutex> lock(dl_tti_lock);

        NVLOGD_FMT(TAG, "{}: size={} nCells={}", __func__, dl_tti_queue_.size(), i);
        if (dl_tti_queue_.size() < i)
        {
            i = dl_tti_queue_.size();
        }

        while(i)
        {
            phy_mac_msg_desc& ipc_msg = dl_tti_queue_.front();
            transport_wrapper().rx_release(ipc_msg);
            dl_tti_queue_.pop();
            i--;
        }
    }

    void PHY_module::on_dl_tti_processed(int num_dl_tti)
    {
        int i = num_dl_tti;
        std::lock_guard<std::mutex> lock(dl_tti_lock);

        NVLOGD_FMT(TAG, "{}: size={} nCells={}", __func__, dl_tti_queue_.size(), i);
        if (dl_tti_queue_.size() < i)
        {
            i = dl_tti_queue_.size();
        }

        while(i)
        {
            phy_mac_msg_desc& ipc_msg = dl_tti_queue_.front();
            transport_wrapper().rx_release(ipc_msg);
            dl_tti_queue_.pop();
            i--;
        }
    }

    void PHY_module::set_tti_flag(bool value)
    {
        NVLOGI_FMT(TAG, "{} value={}", __FUNCTION__, value);

        if (value) {
            tti_module_.start_tick_generator();
            if(!transport_wrapper().get_all_cells_configured()) {
                NVLOGC_FMT(TAG, "SLOT.indication cannot be sent until all cells are configured");
            }
        }
    }

    void PHY_module::stop_tick_generator()
    {
        NVLOGI_FMT(TAG, "{}", __FUNCTION__);
        tti_module_.stop_tick_generator();
    }

    void PHY_module::set_bfw_coeff_buff_info(uint32_t cell_index, bfw_buffer_info* buff)
    {
        /*  Total Buffer is of size: maxCplaneProcSlots (4) * nUeG(8) x nUeLayers(16) * nPrbGrpBfw(273) * nRxAnt(64)  * (2 *sizeof(uint32_t))
               4 x 8 x 16 x 273 x 64 x 8 bytes = 71565312 bytes = ~ 72MB + 4 * 1 bytes(Header Metadata per buffer chunk) */
        uint32_t bfwCoffBuffChunkSize  =  (MAX_DL_UL_BF_UE_GROUPS * MAX_MU_MIMO_LAYERS * MAX_NUM_PRGS_DBF *
                                           NUM_GNB_TX_RX_ANT_PORTS * IQ_REPR_FP32_COMPLEX * sizeof(uint32_t));
        uint32_t bfwCoffBuffUegSize =  (MAX_MU_MIMO_LAYERS * MAX_NUM_PRGS_DBF * NUM_GNB_TX_RX_ANT_PORTS *
                                       IQ_REPR_FP32_COMPLEX * sizeof(uint32_t));
        uint8_t uegIdx = 0;
        uint8_t slotIdx = 0;
        uint8_t* ptr_h = nullptr;
        uint8_t* ptr_d = nullptr;
        for (slotIdx = 0; slotIdx < MAX_BFW_COFF_STORE_INDEX; slotIdx++)
        {
            bfwCoeff_mem_info[cell_index][slotIdx].slotIndex = slotIdx;
            bfwCoeff_mem_info[cell_index][slotIdx].sfn = 0xFFFF;
            bfwCoeff_mem_info[cell_index][slotIdx].slot = 0xFFFF;
            bfwCoeff_mem_info[cell_index][slotIdx].nGnbAnt = NUM_GNB_TX_RX_ANT_PORTS;
            bfwCoeff_mem_info[cell_index][slotIdx].header_size = 1;
            bfwCoeff_mem_info[cell_index][slotIdx].header = &buff->header->state[slotIdx];
            bfwCoeff_mem_info[cell_index][slotIdx].buff_size = bfwCoffBuffChunkSize;
            bfwCoeff_mem_info[cell_index][slotIdx].buff_chunk_size = bfwCoffBuffUegSize;
            bfwCoeff_mem_info[cell_index][slotIdx].num_buff_chunk_busy = 0;
            ptr_h = std::next(buff->dataH, bfwCoffBuffChunkSize * slotIdx);

            if(buff->dataD)
            {
                ptr_d = std::next(buff->dataD, bfwCoffBuffChunkSize * slotIdx);
            }
            for (uegIdx = 0; uegIdx < MAX_DL_UL_BF_UE_GROUPS ; uegIdx++)
            {
                bfwCoeff_mem_info[cell_index][slotIdx].buff_addr_chunk_h[uegIdx] = std::next(ptr_h, bfwCoffBuffUegSize * uegIdx);
                if(buff->dataD)
                {
                    bfwCoeff_mem_info[cell_index][slotIdx].buff_addr_chunk_d[uegIdx] = std::next(ptr_d, bfwCoffBuffUegSize * uegIdx);
                }
                else
                {
                    bfwCoeff_mem_info[cell_index][slotIdx].buff_addr_chunk_d[uegIdx] = nullptr;
                }
            }
        }
    }
    
} // namespace nv

