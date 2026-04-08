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

#if !defined(NV_PHY_MODULE_HPP_INCLUDED_)
#define NV_PHY_MODULE_HPP_INCLUDED_

#include "app_config.hpp"
#include "yaml.hpp"
#include "nv_phy_instance.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "slot_command/csirs_lookup.hpp"
#include "nv_phy_tick_updater.hpp"
#include "nv_tick_generator.hpp"
#include "nvlog.hpp"
#include "stat_log.h"
#include "nv_phy_driver_proxy.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_utils.hpp"
#include "cuphyoam.hpp"
#include "nv_phy_limit_errors.hpp"

#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <chrono>
#include <queue>
#include <map>

namespace nv
{

#if 1
#ifdef SCF_FAPI_10_04
    typedef enum
    {
        RX_DATA_IND_IDX = 0,
        CRC_DATA_IND_IDX,
        UCI_DATA_IND_IDX,
        RACH_DATA_IND_IDX,
        SRS_DATA_IND_IDX,
        DL_TTI_RSP_IDX,
        MAX_IND_INDEX
    }indication_index;
    typedef enum
    {
        NO_CHANGE,
        ONE_MSG_INSTANCE_PER_SLOT,
        MULTI_MSG_INSTANCE_PER_SLOT,
        RESERVED
    }indication_instance_val;
    struct PHY_config{
        //Value, for each entry: • 0: no (change in) configuration • 1: limit to one message instance per slot & numerology • 2: allow generation of more than one message instance per slot &numerology • other values are reserved
        //Scope of each entry: [0]: Rx_Data.indication [1]: CRC.indication [2]: UCI.indication [3]: RACH.indication [4]: SRS.indication [5]: DL_TTI.response
        uint8_t indication_instances_per_slot[6];
        PHY_config()
        {
            for(uint8_t & indication : indication_instances_per_slot)
                indication = 1;
        }
    };
#endif
#endif

using namespace std::chrono;
typedef std::reference_wrapper<PHY_instance> PHY_instance_ref;
struct phy_mac_msg_desc;

enum dl_tb_loc
{
    TB_LOC_INLINE = 0,
    TB_LOC_EXT_HOST_BUF = 1,
    TB_LOC_EXT_GPU_BUF = 2
};

#define MAX_PER_CH_PROC_SEGMENTS 4

using ch_segment = pair<uint16_t, uint16_t>;
using ch_seg_timelines = array<ch_segment, MAX_PER_CH_PROC_SEGMENTS>;
using ch_indexes = array<uint, 8>;

/**
 * @brief Interface class for message-type specific module dispatchers
 *
 * A module dispatcher determines which PHY instance a message should be
 * dispatched to using its knowledge of the message format. The
 * implementation of this function should call the on_msg() function of
 * the PHY instance that is the target of the message.
 */
class PHY_module_dispatch {
public:
    virtual ~PHY_module_dispatch() {}
    
    /**
     * @brief Dispatch a message to the appropriate PHY instance
     * @param msg The IPC message to dispatch
     * @param instances Vector of PHY instance references
     * @return true if dispatch was successful, false otherwise
     */
    virtual bool dispatch(nv_ipc_msg_t&                  msg,
                          std::vector<PHY_instance_ref>& instances) = 0;
};

enum sync_mode_t: uint8_t {
    SYNC_MODE_PER_CELL,
    SYNC_MODE_PER_SLOT
};

class PHY_module;

/**
 * @brief Callback function for PHY module reset
 * @param transp Pointer to the PHY-MAC transport
 * @param phy_module Pointer to the PHY module instance
 * @return Status code (0 for success)
 */
int phy_module_reset_callback(phy_mac_transport* transp, PHY_module* phy_module);

/**
 * @brief Main PHY module class managing PHY instances and message processing
 *
 * The PHY_module class is responsible for managing multiple PHY instances,
 * handling message routing, tick generation, and coordinating communication
 * between MAC and PHY layers.
 */
class PHY_module {
public:
    /**
     * @brief Destructor
     */
    ~PHY_module();

    /**
     * @brief Constructor
     * @param node_config YAML configuration node containing module settings
     */
    PHY_module(yaml::node node_config);

    PHY_module(const PHY_module&) = delete;
    PHY_module& operator=(const PHY_module&) = delete;
    
    /**
     * @brief Move constructor
     * @param other PHY_module instance to move from
     */
    PHY_module(PHY_module&& other) :
        transport_wrapper_(std::move(other.transport_wrapper_)),
        thread_(std::move(other.thread_)),
        phy_instances_(std::move(other.phy_instances_)),
        phy_refs_(std::move(other.phy_refs_)),
        epoll_ctx_p(std::move(other.epoll_ctx_p)),
        thread_cfg_(std::move(other.thread_cfg_)),
        dl_tbs_queue_(std::move(other.dl_tbs_queue_)),
        tick_updater_(std::move(other.tick_updater_)),
        callbacks_(std::move(other.callbacks_)),
        tti_module_(std::move(other.tti_module_)),
        dl_tb_location_(std::move(other.dl_tb_location_)),
        lbrm_(std::move(other.lbrm_)),
        gps_alpha_(std::move(other.gps_alpha_)),
        gps_beta_(std::move(other.gps_beta_)),
        ul_cqi_(std::move(other.ul_cqi_)),
        test_type(std::move(other.test_type)),
        ss_curr(std::move(other.ss_curr)),
        tti_event_count(std::move(other.tti_event_count)),
        allowed_tick_error(std::move(other.allowed_tick_error)),
        prepone_h2d_copy_(std::move(other.prepone_h2d_copy_)),
        rssi_(std::move(other.rssi_)),
        cell_group_(std::move(other.cell_group_)),
        dtx_thresholds_(std::move(other.dtx_thresholds_)),
        dtx_thresholds_pusch_(std::move(other.dtx_thresholds_pusch_)),
        server_addr(std::move(other.server_addr)),        
        target_node(std::move(other.target_node)),
        enable_se_sync_cmd(std::move(other.enable_se_sync_cmd)),
        current_tick_list_(std::move(other.current_tick_list_)),
        l1_slot_ind_tick_(std::move(other.l1_slot_ind_tick_)),
        l2a_start_tick_(std::move(other.l2a_start_tick_)),
        l2a_end_tick_(std::move(other.l2a_end_tick_)),
        last_fapi_msg_tick_(std::move(other.last_fapi_msg_tick_)),
        new_slot_(std::move(other.new_slot_)),
        is_ul_slot_(std::move(other.is_ul_slot_)),
        is_dl_slot_(std::move(other.is_dl_slot_)),
        is_csirs_slot_(std::move(other.is_csirs_slot_)),
        // group_command_(std::move(other.group_command_))
        current_slot_cmd_index(other.current_slot_cmd_index),
        slot_command_array(std::move(other.slot_command_array)),
        num_cells_active(other.num_cells_active),
        total_cell_num(other.total_cell_num),
        next_slot_fapi_num(other.next_slot_fapi_num),
#ifdef ENABLE_L2_SLT_RSP
        active_cell_bitmap(std::move(other.active_cell_bitmap)),
        fapi_eom_rcvd_bitmap(std::move(other.fapi_eom_rcvd_bitmap)),
#else
        slot_msgs_received(std::move(other.slot_msgs_received)),
        num_msgs(other.num_msgs),
        allowed_fapi_latency(std::move(other.allowed_fapi_latency)),
        ss_last(std::move(other.ss_last)),
#endif
        cell_update_cb_fn(std::move(other.cell_update_cb_fn)),
        current_tick_(other.current_tick_.load()),
        ipc_sync_mode(std::move(other.ipc_sync_mode)),
        first_dl_slot_(std::move(other.first_dl_slot_)),
        first_ul_slot_(std::move(other.first_ul_slot_)),
        config_options_(std::move(other.config_options_)),
        bfwCoeff_mem_info(std::move(other.bfwCoeff_mem_info)),
        timer_thread_wakeup_threshold_(std::move(other.timer_thread_wakeup_threshold_)),
        l2a_allowed_latency_(std::move(other.l2a_allowed_latency_)),
        fapi_config_check_mask_(std::move(other.fapi_config_check_mask_)),
        ch_proc_seg_indexes(std::move(other.ch_proc_seg_indexes)),
        ch_proc_seg_timelines(std::move(other.ch_proc_seg_timelines)),
#ifdef ENABLE_L2_SLT_RSP
        cell_limit_errors_(std::move(other.cell_limit_errors_)),
        group_limit_errors_(std::move(other.group_limit_errors_))
#endif
    {
        // Replace pointer to parent module in PHY instances, since that
        // module no longer exists...
        for(auto& phy : phy_instances_)
        {
            phy->set_module(*this);
        }

        // Old module does not exist, remap the transport fd event callback.
        std::unique_ptr<member_event_callback<PHY_module>> mcb_p(new member_event_callback<PHY_module>(this, &PHY_module::msg_processing));
        for (phy_mac_transport* ptransport : transport_wrapper_.get_transports())
        {
            ptransport->set_reset_callback(phy_module_reset_callback, this);
            epoll_ctx_p->add_fd(ptransport->get_fd(), mcb_p.get());
        }
        msg_mcb_p = std::move(mcb_p);
        tti_module_.set_module(*this);

        tick_logger = other.tick_logger;
        slot_latency = other.slot_latency;
        other.slot_latency = nullptr;
        other.tick_logger = nullptr;
    }
    
    /**
     * @brief Get references to all PHY instances
     * @return Vector of PHY instance references
     */
    std::vector<PHY_instance_ref>& PHY_instances() { return phy_refs_; }
    
    /**
     * @brief Start the PHY module thread
     *
     * Initiates the PHY_module processing thread that handles message
     * routing and coordination between MAC and PHY instances.
     */
    void start();
    
    /**
     * Stop the PHY module
     *
     * Signals the module thread and tick generator to stop.
     */
    void stop();

    /**
     * @brief Join the PHY module thread
     *
     * Blocks until all PHY instances have completed their execution.
     */
    void join();
    
    /**
     * @brief Get transport instance for a specific cell
     * @param cell_id Cell identifier
     * @return Reference to the PHY-MAC transport for the specified cell
     */
    phy_mac_transport& transport(int cell_id) { return transport_wrapper_.get_transport(cell_id); }

    /**
     * @brief Get the transport wrapper instance
     * @return Reference to the transport wrapper managing all transports
     */
    phy_mac_transport_wrapper& transport_wrapper() { return transport_wrapper_; }

    /**
     * @brief Reset a transport connection
     * @param transp Pointer to the transport to reset
     * @return Status code (0 for success)
     */
    int reset_transport(phy_mac_transport* transp);

private:
    /**
     * @brief Main thread function for PHY module processing
     */
    void thread_func();
    
    /**
     * @brief Process incoming messages from transport layer
     */
    void msg_processing();
    
    /**
     * @brief Thread function for cell configuration updates
     * @param arg Pointer to PHY_module instance
     * @return void pointer (unused)
     */
    static void* cell_update_thread_func(void* arg);

    /**
     * @brief Thread function for SFN/slot synchronization commands
     * @param arg Pointer to PHY_module instance
     * @return void pointer (unused)
     */
    static void* sfn_slot_sync_cmd_thread_func(void* arg);
    
    /**
     * @brief Receive a message from transport
     * @return true if message was received successfully, false otherwise
     */
    bool recv_msg();
    
    /**
     * @brief Check if time threshold has been exceeded
     * @param time_ns Time in nanoseconds
     * @param slot_num Slot number
     * @param is_ul true for uplink, false for downlink
     * @return true if threshold exceeded, false otherwise
     */
    bool check_time_threshold(std::chrono::nanoseconds,uint16_t,bool);

public:
    /**
     * @brief Process PHY commands for current slot
     * @param force_process Force processing even if conditions not met
     */
    void process_phy_commands(bool);

    /**
     * @brief Callback when DL transport block has been processed
     */
    void on_dl_tb_processed();
    
    /**
     * @brief Callback when DL transport block has been processed
     * @param params Pointer to PDSCH parameters
     */
    void on_dl_tb_processed(const slot_command_api::pdsch_params* params);

    /**
     * @brief Callback when DL TTI has been processed (UNUSED)
     */
    void on_dl_tti_processed();
    
    /**
     * @brief Callback when DL TTI has been processed (UNUSED)
     * @param num_dl_tti Number of DL TTI messages processed
     */
    void on_dl_tti_processed(int num_dl_tti);
    
    /**
     * @brief Handle received timing tick
     * @param tick_time Tick timestamp in nanoseconds
     */
    void tick_received(std::chrono::nanoseconds&);
    
    /**
     * @brief Set the TTI flag
     * @param flag TTI flag value
     */
    void set_tti_flag(bool flag);

    /**
     * @brief Stop the tick generator
     */
    void stop_tick_generator();

    /**
     * @brief Set whether all cells are configured
     * @param value true if all cells configured, false otherwise
     */
    void set_all_cells_configured(bool value) { all_cells_configured = value; }
    
    /**
     * @brief Send callbacks to registered handlers
     */
    void send_call_backs();
    
    /**
     * @brief Get the highest numerology (mu) configured
     * @return Highest mu value
     */
    uint32_t get_mu_highest() {return tick_updater_.mu_highest_;}
    
    /**
     * @brief Get the slot advance value
     * @return Number of slots to advance
     */
    uint32_t get_slot_advance() {return tick_updater_.slot_advance_;}
    
    /**
     * @brief Get previous tick slot indication
     * @return Reference to previous slot indication
     */
    slot_command_api::slot_indication& get_prev_tick() {return tick_updater_.prev_slot_info_;}
    
    /**
     * @brief Get downlink transport block location type
     * @return DL TB location (inline, host buffer, or GPU buffer)
     */
    dl_tb_loc dl_tb_location() { return dl_tb_location_; }

    /**
     * @brief Check if dynamic SFN/slot tick is enabled
     * @return 1 if enabled, 0 otherwise
     */
    int tickDynamicSfnSlotIsEnabled() { return config_options_.enableTickDynamicSfnSlot; }
    
    /**
     * @brief Get static PUSCH slot number for testing
     * @return PUSCH slot number
     */
    int staticPuschSlotNum() { return config_options_.staticPuschSlotNum; }
    
    /**
     * @brief Get static PDSCH slot number for testing
     * @return PDSCH slot number
     */
    int staticPdschSlotNum() { return config_options_.staticPdschSlotNum; }
    
    /**
     * @brief Get static PDCCH slot number for testing
     * @return PDCCH slot number
     */
    int staticPdcchSlotNum() { return config_options_.staticPdcchSlotNum; }
    
    /**
     * @brief Get static CSI-RS slot number for testing
     * @return CSI-RS slot number
     */
    int staticCsiRsSlotNum() { return config_options_.staticCsiRsSlotNum; }
    
    /**
     * @brief Get static SSB physical cell ID for testing
     * @return SSB PCID
     */
    int staticSsbPcid() { return config_options_.staticSsbPcid; }
    
    /**
     * @brief Get static SSB SFN for testing
     * @return SSB SFN
     */
    int staticSsbSFN() { return config_options_.staticSsbSFN; }
    
    /**
     * @brief Get static SSB slot number for testing
     * @return SSB slot number
     */
    int staticSsbSlotNum() { return config_options_.staticSsbSlotNum; }
    
    /**
     * @brief Get static PUCCH slot number for testing
     * @return PUCCH slot number
     */
    int staticPucchSlotNum() {return config_options_.staticPucchSlotNum; }
    
    /**
     * @brief Check if beamforming is enabled
     * @return true if beamforming enabled, false otherwise
     */
    bool bf_enabled() { return config_options_.bf_enabled; }
    
    /**
     * @brief Check if precoding/precoding matrix is enabled
     * @return true if precoding enabled, false otherwise
     */
    bool pm_enabled() { return config_options_.precoding_enabled; }
    
    /**
     * @brief Get Limited Buffer Rate Matching (LBRM) value
     * @return LBRM configuration value
     */
    uint8_t lbrm() { return lbrm_; }
    
    /**
     * @brief Get GPS alpha value for timing synchronization
     * @return GPS alpha value
     */
    uint64_t gps_alpha() { return gps_alpha_; }
    
    /**
     * @brief Get GPS beta value for timing synchronization
     * @return GPS beta value
     */
    int64_t gps_beta() { return gps_beta_; }
    
    /**
     * @brief Get uplink CQI value
     * @return UL CQI value (0xff if not set)
     */
    uint8_t ul_cqi() { return ul_cqi_; }
    
    /**
     * @brief Get RSSI (Received Signal Strength Indicator) value
     * @return RSSI value (0xffff if not set)
     */
    uint16_t rssi() { return rssi_; }
    
    /**
     * @brief Get RSRP (Reference Signal Received Power) value
     * @return RSRP value (0xffff if not set)
     */
    uint16_t rsrp() { return rsrp_; }
    // static cell_group_command* group_command() {
    //     return &group_command_;
    // }
    static std::unordered_map<uint32_t, pm_weights_t>& pm_map() {
        return pm_weight_map_;
    }
    static std::unordered_map<uint16_t, digBeam_t>& static_digBeam_map() {
        return static_digBeam_weight_map_;
    }
    bool cell_group() { return cell_group_; }
    const pucch_dtx_t_list&  dtx_thresholds() const { return dtx_thresholds_; }
    const float&  dtx_thresholds_pusch() const { return dtx_thresholds_pusch_; }

    std::chrono::nanoseconds l2a_start_tick() { return l2a_start_tick_; }
    std::chrono::nanoseconds l2a_end_tick() { return l2a_end_tick_; }
    void l2a_start_tick(std::chrono::nanoseconds time) { l2a_start_tick_ = time; }
    void l2a_end_tick(std::chrono::nanoseconds time) { l2a_end_tick_ = time; }
    void last_fapi_msg_tick(std::chrono::nanoseconds time) { last_fapi_msg_tick_ = time; }
    bool new_slot() { return new_slot_; }
    void new_slot(bool new_slot) { new_slot_ = new_slot; }
    bool is_ul_slot() { return is_ul_slot_; }
    bool is_dl_slot() { return is_dl_slot_; }
    void is_ul_slot(bool is_ul_slot) { is_ul_slot_ = is_ul_slot; }
    void is_dl_slot(bool is_dl_slot) { is_dl_slot_ = is_dl_slot; }
    bool is_csirs_slot() { return is_csirs_slot_; }
    void is_csirs_slot(bool is_csirs_slot) { is_csirs_slot_ = is_csirs_slot; }
    sfn_slot_t& get_curr_sfn_slot() { return ss_curr; }
    sfn_slot_t get_next_sfn_slot(sfn_slot_t& ss);
    uint32_t get_fapi_latency(sfn_slot_t ss_msg);
    uint32_t get_slot_interval(sfn_slot_t ss_old, sfn_slot_t ss_new);
#ifdef ENABLE_L2_SLT_RSP
    void set_curr_sfn_slot(sfn_slot_t sfnslot) { ss_curr.u32 = sfnslot.u32; }
    uint32_t get_active_cell_bitmap() { return active_cell_bitmap; }
    void set_active_cell_bitmap(uint16_t cell_id) { active_cell_bitmap |= 1ULL << cell_id; }
    void unset_active_cell_bitmap(uint16_t cell_id) { active_cell_bitmap &= ~(1ULL << cell_id); }
    void update_eom_rcvd_bitmap(uint16_t cell_id) {  fapi_eom_rcvd_bitmap |= 1ULL << cell_id; }
    uint32_t get_eom_rcvd_bitmap() {return fapi_eom_rcvd_bitmap; }
#else
    uint32_t get_allowed_fapi_latency() { return allowed_fapi_latency; }
    sfn_slot_t& get_last_sfn_slot() { return ss_last; }
#endif
    void set_first_tick(bool ft) { first_tick = ft; }
    slot_command_api::cell_sub_command& cell_sub_command(uint32_t cell_index) {return slot_command_array.at(current_slot_cmd_index).cells.at(cell_index);}
    slot_command_api::slot_command& slot_command() {return slot_command_array.at(current_slot_cmd_index);}
    cell_group_command* group_command() {
        return &(slot_command_array.at(current_slot_cmd_index).cell_groups);
    }
    void update_slot_cmds_indexes() {
        current_slot_cmd_index = (current_slot_cmd_index + 1 )% slot_command_array.size();
    }

    bfw_coeff_mem_info_t* get_bfw_coeff_buff_info(uint32_t cell_index, uint8_t slot_index){return &(bfwCoeff_mem_info[cell_index][slot_index]);}

    void set_bfw_coeff_buff_info(uint32_t cell_index, bfw_buffer_info* buff);

    void incr_active_cells() {
        num_cells_active++;
    }

    void decr_active_cells() {
        num_cells_active--;
    }

    void oam_cell_eaxcids_update(uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map);

    void oam_cell_multi_attri_update(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res);

    void create_cell_update_call_back();

    ::CellUpdateCallBackFn& cell_update_cb() {
        return cell_update_cb_fn;
    }
    uint8_t prepone_h2d_copy(){ return prepone_h2d_copy_;};

    inline phy_config_option& config_options() { return config_options_; }
    uint16_t get_stat_prm_idx_to_cell_id_map_size(){return stat_prm_idx_to_cell_id_map.size();};
    void insert_cell_id_in_stat_prm_map(uint16_t cell_id, uint16_t stat_idx){ stat_prm_idx_to_cell_id_map.insert({stat_idx,cell_id}); };
    uint16_t get_cell_id_from_stat_prm_idx(uint16_t stat_idx ){ return stat_prm_idx_to_cell_id_map[stat_idx];};
    uint64_t fapi_config_check_mask() {return fapi_config_check_mask_;};

#ifdef SCF_FAPI_10_04
    PHY_config& get_phy_config(){return phy_config;};
#endif
    int send_sfn_slot_sync_grpc_command();
    bool get_sfn_slot_sync_cmd_sent(){return sfn_slot_sync_cmd_sent;};
    void set_sfn_slot_sync_cmd_sent(bool val){sfn_slot_sync_cmd_sent=val;};
    uint8_t get_enable_se_sync_cmd(){return enable_se_sync_cmd;};
    uint8_t get_target_node(){return target_node;};
    bool check_sync_rcvd_from_ue(){return(sync_rcvd_from_ue==true);};
    bool check_sync_rcvd_from_du(){return(sync_rcvd_from_du==true);};
    ch_indexes& get_ch_proc_indexes(uint type) {return ch_proc_seg_indexes[type]; }
    ch_seg_timelines& get_ch_timeline(uint type) { return ch_proc_seg_timelines[type];}

    // Accessor methods for slot limit errors
    slot_limit_cell_error_t& get_cell_limit_errors(uint16_t cell_id) {
        return cell_limit_errors_[cell_id];
    }

    slot_limit_group_error_t& get_group_limit_errors() {
        return group_limit_errors_;
    }

    void reset_l1_limit_errors()
    {
        // Reset all cell errors
        for (auto& cell_error : cell_limit_errors_) {
            std::fill_n(reinterpret_cast<std::uint8_t*>(&cell_error),
            sizeof(cell_error), 0);
        }

        // Reset all group errors
        std::fill_n(reinterpret_cast<std::uint8_t*>(&group_limit_errors_),
            sizeof(group_limit_errors_), 0);
    }
private:
    typedef std::unique_ptr<PHY_instance> PHY_instance_ptr;
    //------------------------------------------------------------------
    // Data
    phy_mac_transport_wrapper transport_wrapper_;
    std::thread       thread_;    // module thread
    std::unique_ptr<thread_config>     thread_cfg_;

    std::vector<PHY_instance_ptr> phy_instances_;
    std::vector<PHY_instance_ref> phy_refs_;

    std::unique_ptr<member_event_callback<PHY_module>> msg_mcb_p;
    std::unique_ptr<phy_epoll_context>                 epoll_ctx_p;
    ///// Queue holding the DL TB to released in DL callback
    nv_preallocated_queue<phy_mac_msg_desc> dl_tbs_queue_;
    std::mutex dl_tbs_lock;

    ///// Queue holding the DL TTI msg to be released in FH prepare callback
    std::queue<phy_mac_msg_desc> dl_tti_queue_;
    std::mutex dl_tti_lock;

    // Tick member variables
    // Atomic variables can be used in both tick_generator and msg_processing threads
    nv::TickUpdater tick_updater_;
    bool first_tick = true;
    std::atomic<sfn_slot_t> ss_tick;
    std::mutex tick_lock;
    std::atomic<nanoseconds> current_tick_;
    nv::tti_gen tti_module_;

    // TODO: Remove the limitation that all cells need to be configured at initial
    bool all_cells_configured = false;

    bool sfn_slot_sync_cmd_sent = false;

    std::once_flag cb_flag;
    slot_command_api::callbacks callbacks_;
    dl_tb_loc dl_tb_location_;
    uint64_t timer_thread_wakeup_threshold_ = 15000; //15 us
    uint32_t l2a_allowed_latency_ = 100000; //100 us
    uint8_t lbrm_ = 0;
    uint64_t gps_alpha_ = 0;
    int64_t gps_beta_ = 0;
    uint8_t ul_cqi_ = 0xff;
    uint16_t rssi_ = 0xffff;
    uint16_t rsrp_ = 0xffff;

    // test_type: 0 - normal; 1 - l2adapter standalone; 2 - tick unit test
    int32_t test_type = 0;

    bool cell_group_ = false;

    uint32_t total_cell_num = 0;

    // Tick interval deviation statistic logger
    int32_t allowed_tick_error;
    uint8_t prepone_h2d_copy_;
    uint64_t fapi_config_check_mask_ = 0x0UL;
    stat_log_t* tick_logger;
    stat_log_t* slot_latency;
    pucch_dtx_t_list dtx_thresholds_;
    float            dtx_thresholds_pusch_;

    //SE sync params
    std::string server_addr;
    uint8_t     target_node;
    uint8_t        enable_se_sync_cmd;
    std::atomic<bool>        sync_rcvd_from_ue;
    std::atomic<bool>        sync_rcvd_from_du;

    bool new_slot_ = true;
    std::array<std::chrono::nanoseconds,10> current_tick_list_;//Array for storing last 10 ticks (indexed based on slot%10)
    std::array<std::chrono::nanoseconds,10> l1_slot_ind_tick_;//Array for storing last 10 slot indicator times (indexed based on slot%10)
    std::chrono::nanoseconds l2a_start_tick_;
    std::chrono::nanoseconds l2a_end_tick_;
    std::chrono::nanoseconds last_fapi_msg_tick_;
    bool is_ul_slot_ = false;
    bool is_dl_slot_ = false;
    bool is_csirs_slot_ = false;

    // Below variables should only use in msg_processing thread
    sfn_slot_t ss_curr;
    int32_t tti_event_count;
    uint32_t current_slot_cmd_index;
    uint num_cells_active;
    uint16_t next_slot_fapi_num;
    std::array<nv::phy_mac_msg_desc, MAX_CELLS_PER_SLOT * 12> next_slot_fapi_cache;
#ifdef ENABLE_L2_SLT_RSP
    uint64_t active_cell_bitmap;
    uint64_t fapi_eom_rcvd_bitmap;
#else
    uint32_t allowed_fapi_latency;
    std::array<nv::phy_mac_msg_desc, 128> slot_msgs_received;
    uint16_t num_msgs;
    sfn_slot_t ss_last;
#endif
    ::CellUpdateCallBackFn cell_update_cb_fn;
#ifdef SCF_FAPI_10_04
    PHY_config  phy_config;
#endif
    public:
    // static slot_command_api::cell_group_command group_command_;
    static std::unordered_map<uint32_t, pm_weights_t> pm_weight_map_;
    static std::unordered_map<uint16_t, digBeam_t> static_digBeam_weight_map_;
    std::vector<slot_command_api::slot_command> slot_command_array;
    std::once_flag cell_update_flag;
    /// 0 - Sync per cell, 1 - sync per slot
    sync_mode_t ipc_sync_mode;
    bool first_dl_slot_ = true;
    bool first_ul_slot_ = true;

    phy_config_option config_options_;
    std::map<uint16_t, uint16_t> stat_prm_idx_to_cell_id_map;

    bfw_coeff_mem_info_t bfwCoeff_mem_info[MAX_CELLS_PER_SLOT][MAX_BFW_COFF_STORE_INDEX];
    bfw_coeff_mem_info_t static_bfwCoeff_mem_info[MAX_CELLS_PER_SLOT][MAX_STATIC_BFW_COFF_STORE_INDEX];
    unordered_map<uint, ch_indexes> ch_proc_seg_indexes;
    unordered_map<uint, ch_seg_timelines> ch_proc_seg_timelines;

#ifdef ENABLE_L2_SLT_RSP
    std::array<slot_limit_cell_error_t, MAX_CELLS_PER_SLOT> cell_limit_errors_;
    slot_limit_group_error_t group_limit_errors_;
#endif
};

} // namespace nv

#endif // !defined(NV_PHY_MODULE_HPP_INCLUDED_)
