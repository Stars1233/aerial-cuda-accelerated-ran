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

#if !defined(SCF_5G_FAPI_PHY_HPP_INCLUDED_)
#define SCF_5G_FAPI_PHY_HPP_INCLUDED_

#include "nv_phy_instance.hpp"
#include "nv_phy_module.hpp"
#include "scf_5g_fapi.h"
#include "scf_5g_fapi_metrics.hpp"
#include "scf_5g_fapi_ul_validate.hpp"
#include "scf_5g_fapi_dl_validate.hpp"
#include "nv_phy_driver_proxy.hpp"
#include "cuda_fp16.h"
#include "cuphy.hpp"

#include <map>
#include <unordered_map>
#include <set>
#include <chrono>

/// Debug flag for precoder (0 = disabled, 1 = enabled)
#define DBG_PRECODER 0

/**
 * @name SSB Symbol Configuration Constants
 * @brief Lookup tables for SSB symbol positions in different frequency ranges
 * @{
 */

/// L_max symbols for SSB with 4 possible symbols (Case A, B, C)
static constexpr uint16_t L_MAX_4_SYMBOLS[3][4] = {{2, 8, 16, 22}, {4, 8, 16, 20}, {2, 8, 16, 22}};

/// L_max symbols for SSB with 8 possible symbols
static constexpr uint16_t L_MAX_8_SYMBOLS[3][8] = {{2, 8, 16, 22, 30, 36, 44, 50}, {4, 8, 16, 20, 32, 36, 44, 48}, {2, 8, 16, 22, 30, 36, 44, 50}};

/// L_max symbols for SSB with 64 possible symbols (FR2 mmWave)
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
/** @} */

/**
 * @name PHY Configuration Constants
 * @{
 */
static constexpr int32_t INVALID_CELL_CFG_IDX = -1;  ///< Invalid cell configuration index
static constexpr int IPC_NOTIFY_VALUE = 1;           ///< IPC notification value
/** @} */

/**
 * @name Timing Advance Constants
 * @brief Constants for calculating timing advance values
 * @{
 */
static constexpr uint16_t INVALID_TA = 0xFFFF;       ///< Invalid timing advance value
static constexpr uint32_t TA_BASE_OFFSET = 31;       ///< Base offset for TA calculation
static constexpr float TA_MICROSECOND_TO_SECOND = 1e-6f;  ///< Microsecond to second conversion
/// Timing advance base scale factor
static constexpr float TA_BASE_SCALE = (480000.0f * 4096.0f)/(16.0f * 64.0f) * TA_MICROSECOND_TO_SECOND;
static constexpr uint16_t TA_MAX_PRACH = 3846;       ///< Maximum TA for PRACH
static constexpr uint16_t TA_MAX_NON_PRACH = 63;     ///< Maximum TA for non-PRACH channels
/** @} */


/**
 * @brief SRS PDU processing error codes
 */
typedef enum
{
    SRS_PDU_SUCCESS                       = 0,  ///< SRS PDU processed successfully
    SRS_PDU_NO_SRS_CMD                    = 1,  ///< No SRS command available
    SRS_PDU_INVALID_REPORT_SCOPE          = 2,  ///< Invalid report scope
    SRS_PDU_UNSUPPORTED_REPORT_USAGE      = 3,  ///< Unsupported report usage
    SRS_PDU_OVERFLOW_NVIPC_BUFF           = 4,  ///< NVIPC buffer overflow
    SRS_PDU_INVALID_NUM_ANT_PORTS         = 5,  ///< Invalid number of antenna ports
    SRS_PDU_INVALID_TRP_SCHEME            = 6,  ///< Invalid TRP scheme
    SRS_PDU_L1_LIMIT_ERROR                = 7   ///< L1 limit error
} srs_error_code_t;

/**
 * @brief SCF 5G FAPI namespace
 *
 * Contains implementations of the Small Cell Forum 5G FAPI specification
 * for PHY-MAC interface communication.
 */
namespace scf_5g_fapi
{

/**
 * @brief FAPI state machine enumeration
 */
enum class fapi_state_t {
    FAPI_STATE_IDLE,        ///< PHY is idle, not configured
    FAPI_STATE_CONFIGURED,  ///< PHY is configured but not running
    FAPI_STATE_RUNNING      ///< PHY is configured and running
} ;

/**
 * @brief TX data request metadata
 *
 * Contains metadata for processing downlink transport blocks
 * from TX_DATA.request messages.
 */
typedef struct tx_data_req_meta_data_t
{
    uint16_t num_pdus;   ///< Number of PDUs (to be compared with dl_pdu_index.size())
    uint8_t* data;       ///< Pointer to data buffer
    uint8_t* buf;        ///< Pointer to working buffer
}
tx_data_req_meta_data_t;

/**
 * @brief PHY instance for SCF 5G FAPI
 *
 * Implements the Small Cell Forum 5G FAPI interface for communication
 * between MAC and PHY layers. Handles FAPI message processing, cell
 * configuration, and uplink/downlink slot commands.
 */
class phy : public nv::PHY_instance
{
public:
    /**
     * @brief Constructor
     * @param module Reference to parent PHY module
     * @param node_config YAML configuration node
     */
    phy(nv::PHY_module& module, yaml::node node_config);
    
    /**
     * @brief Destructor
     */
    virtual ~phy();
    
    /**
     * @brief Process incoming FAPI message
     * @param msg IPC message containing FAPI command
     * @return true if message processed successfully, false otherwise
     */
    virtual bool on_msg(nv_ipc_msg_t& msg) override;
    
    /**
     * @brief Reset slot-specific state
     * @param force_reset Force reset even if conditions not met
     */
    virtual void reset_slot(bool) override;

    /**
     * @brief Reset PHY state for L2 reconnection
     * @return 0 on success, negative error code on failure
     */
    virtual int reset() override;

    /**
     * @brief Get carrier ID for this PHY instance
     * @return Carrier identifier
     */
    const int32_t get_carrier_id() {return phy_config.cell_config_.carrier_idx;}
    
    /**
     * @brief Process CONFIG.request message
     * @param config_request Configuration request parameters
     * @param cell_id Cell identifier
     * @param handle_id Handle identifier for response
     * @param ipc_msg IPC message structure
     */
    void on_config_request(scf_fapi_config_request_msg_t& config_request, const int32_t cell_id, uint8_t handle_id, nv_ipc_msg_t& ipc_msg);
    
    /**
     * @brief Process CV memory bank configuration request
     * @param cv_mem_bank_config_request_body Configuration parameters
     * @param cell_id Cell identifier
     * @param ipc_msg IPC message structure
     */
    void on_cv_mem_bank_config_request(cv_mem_bank_config_request_body_t * cv_mem_bank_config_request_body, uint32_t cell_id, nv_ipc_msg_t& ipc_msg);
    
    /**
     * @brief Handle unknown FAPI message
     * @param hdr Message header
     */
    void on_unknown_msg(scf_fapi_body_header_t& hdr);
    
    /**
     * @brief Process PARAM.request message
     */
    void on_param_request();
    
    /**
     * @brief Process START.request message
     * @param cell_id Cell identifier
     */
    void on_cell_start_request(const int32_t cell_id);
    
    /**
     * @brief Process STOP.request message
     * @param cell_id Cell identifier
     */
    void on_cell_stop_request(const int32_t cell_id);
    
    /**
     * @brief Process DL_TTI.request message
     * @param msg DL TTI request parameters
     * @param ipc_msg IPC message structure
     * @param pdsch_valid_flag Array of PDSCH validation flags
     */
    void on_dl_tti_request(scf_fapi_dl_tti_req_t &msg, nv_ipc_msg_t& ipc_msg, uint8_t* pdsch_valid_flag);
    
    /**
     * @brief Process UL_TTI.request message
     * @param request UL TTI request parameters
     * @param ipc_msg IPC message structure
     */
    void on_ul_tti_request(scf_fapi_ul_tti_req_t& request, nv_ipc_msg_t& ipc_msg);
    
    /**
     * @brief Process DL beamforming weight/CVI request
     * @param msg DL BFW request parameters
     * @param ipc_msg IPC message structure
     */
    void on_dl_bfw_request(scf_fapi_dl_bfw_cvi_request_t& msg, nv_ipc_msg_t& ipc_msg);
    
    /**
     * @brief Process UL beamforming weight/CVI request
     * @param msg UL BFW request parameters
     * @param ipc_msg IPC message structure
     */
    void on_ul_bfw_request(scf_fapi_ul_bfw_cvi_request_t& msg, nv_ipc_msg_t& ipc_msg);
    void on_prach_pdu_info(scf_fapi_prach_pdu_t &request, slot_command_api::slot_indication& slot_ind);
    void on_ul_dci_request(scf_fapi_ul_dci_t& request, nv_ipc_msg_t& ipc_msg);
    bool on_pusch_pdu_info(scf_fapi_pusch_pdu_t& pdu_info);
    void on_pucch_pdu_info(scf_fapi_pucch_pdu_t& pdu_info, slot_command_api::slot_indication& slot_ind);
    bool on_phy_dl_tx_request(scf_fapi_tx_data_req_t& request, nv_ipc_msg_t& ipc_msg, uint8_t* pdsch_valid_flag);
    int on_srs_pdu_info(scf_fapi_srs_pdu_t& srs_pdu, slot_command_api::slot_indication& slot_ind, size_t nvIpcAllocBuffLen, int *p_srs_ind_index, bool is_last_srs_pdu, bool is_last_non_prach_pdu);
    void on_slot_error_indication(scf_fapi_error_ind_t& error_msg, nv_ipc_msg_t& ipc_msg);
    void send_rach_indication(slot_command_api::slot_indication& slot,
                                const prach_params& params,
                                const uint32_t* num_detectedPrmb,
                                const void* prmbIndex_estimates,
                                const void* prmbDelay_estimates,
                                const void* prmbPower_estimates,
                                const void* ant_rssi,
                                const void* rssi,
                                const void* interference);
    uint16_t send_crc_indication(const slot_command_api::slot_indication& slot,
        const slot_command_api::pusch_params& params,
        ::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms);
    void send_uci_indication(slot_command_api::slot_indication& slot,
                const slot_command_api::pucch_params& params,
                const slot_command_api::uci_output_params& outParams);
    void send_uci_indication(slot_command_api::slot_indication& slot,
                const slot_command_api::pucch_params& params,
                const cuphyPucchDataOut_t& out);
    void send_uci_indication(const slot_command_api::slot_indication& slot,
                const slot_command_api::pusch_params& params,
                const cuphyPuschDataOut_t& out,
                ::cuphyPuschStatPrms_t const* puschStatPrms);
    void send_early_uci_indication(const slot_command_api::slot_indication& slot,
                const slot_command_api::pusch_params& params,
                const cuphyPuschDataOut_t& out,
                ::cuphyPuschStatPrms_t const* puschStatPrms, nanoseconds& to_orig);
    void send_srs_indication(const slot_command_api::slot_indication& slot,
                const slot_command_api::srs_params& params,
                ::cuphySrsDataOut_t const* out,
                ::cuphySrsStatPrms_t const* srsStatPrms,
                const std::array<bool,UL_MAX_CELLS_PER_SLOT>& srs_order_cell_timeout_list);
    void create_ul_dl_callbacks(slot_command_api::callbacks &cb);
    
    /**
     * @brief Wrapper for DL TB processed callback - provides public access for static wrapper
     * @param params PDSCH parameters
     */
    void on_dl_tb_processed_callback(const slot_command_api::pdsch_params* params);
    
    /**
     * @brief FH prepare callback wrappers - provide public access for static wrappers
     * @param grp_cmd Cell group command
     * @param cell Cell ID
     */
    void fh_prepare_callback_wrapper_tff(slot_command_api::cell_group_command* grp_cmd, uint8_t cell);  // <true,false,false>
    void fh_prepare_callback_wrapper_tft(slot_command_api::cell_group_command* grp_cmd, uint8_t cell);  // <true,false,true>
    void fh_prepare_callback_wrapper_ttf(slot_command_api::cell_group_command* grp_cmd, uint8_t cell);  // <true,true,false>
    void fh_prepare_callback_wrapper_ttt(slot_command_api::cell_group_command* grp_cmd, uint8_t cell);  // <true,true,true>
    
    void send_rx_data_indication(const slot_command_api::slot_indication& slot,
        const slot_command_api::pusch_params&, ::cuphyPuschDataOut_t const*, ::cuphyPuschStatPrms_t const* puschStatPrms);
    void send_rx_pe_noise_var_indication(const slot_command_api::slot_indication& slot, const slot_command_api::pusch_params& params,
        ::cuphyPuschDataOut_t const* out, ::cuphyPuschStatPrms_t const* puschStatPrms);

    void prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_csi_rsi_pdu_t& pdu,uint32_t,bool);
    void prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pdcch_pdu_t& pdu, uint8_t testMode);

    /**
     * Prepare the DL slot command for PDSCH
     * @param slot_ind The slot information, including SFN, slot, and tick
     * @param pdu The scf_fapi_pdsch_pdu_t data structure
     * @param testMode The test mode
     * @return true if PDSCH accepted, false if rejected (e.g. check_bf_pc_params failed). When false, caller must set pdsch_valid_flag. */
    bool prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pdsch_pdu_t& pdu, uint8_t testMode);

    void prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_ssb_pdu_t& pdu);
    void prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_prach_pdu_t& req);
    void prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pdcch_pdu_t& pdu);
    void prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pusch_pdu_t& pdu);
    void prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_pucch_pdu_t& pdu, uint16_t pucch_hopping_id);
    int prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_srs_pdu_t& pdu, size_t nvIpcAllocBuffLen, int *p_srs_ind_index, bool is_last_srs_pdu, bool is_last_non_prach_pdu);
    void prepare_dl_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_dl_bfw_group_config_t& pdu, uint32_t &droppedDlBFWPdu);
    void prepare_ul_slot_command(slot_command_api::slot_indication& slot_ind, scf_fapi_ul_bfw_group_config_t& pdu, uint32_t &droppedUlBFWPdu);

    void on_dl_bfw_pdu_info(scf_fapi_dl_bfw_group_config_t &pdu_info, slot_command_api::slot_indication& slot_ind, uint32_t &droppedDlBFWPdu);
    void on_ul_bfw_pdu_info(scf_fapi_ul_bfw_group_config_t &pdu_info, slot_command_api::slot_indication& slot_ind, uint32_t &droppedUlBFWPdu);

    bool is_valid_dci_rx_slot() { return valid_dci_rx;}
    void set_valid_dci_rx_slot(bool value) {valid_dci_rx = value;}

    virtual void send_slot_indication(slot_command_api::slot_indication& slot_3gpp) override;
    virtual void send_slot_error_indication(slot_command_api::slot_indication& slot_3gpp) override;
    virtual void send_phy_l1_enqueue_error_indication(uint16_t sfn,uint16_t slot,bool ul_slot,std::array<int32_t,MAX_CELLS_PER_SLOT>& cell_id_list,int32_t& index) override;
    void print_cell_stats(slot_command_api::slot_indication* slot_3gpp);
    virtual void send_cell_config_response(int32_t cell_id, uint8_t response_code) override;
    virtual void handle_cell_config_response(int32_t cell_id, uint8_t response_code) override;
    bool process_dl_tx_request();
    /** @brief Send an SCF FAPI error indication to the MAC for this PHY instance's configured carrier.
     *  @param[in] msg_id FAPI message ID the error is associated with.
     *  @param[in] error_code SCF error code placed in the indication.
     *  @param[in] sfn System frame number for the indication.
     *  @param[in] slot Slot index for the indication.
     *  @param[in] log_info If true, log the outgoing indication at info level; if false, at error/event level.
     *  @param[in] total_errors Optional PDU count when populating error extension fields.
     *  @param[in] cell_error Optional per-cell L1 limit error context for extension payload.
     *  @param[in] group_error Optional group L1 limit error context for extension payload.
     */
    void send_error_indication(scf_fapi_message_id_e msg_id,  scf_fapi_error_codes_t error_code, uint16_t sfn, uint16_t slot, bool log_info = false, uint16_t total_errors = 0, nv::slot_limit_cell_error_t* cell_error = nullptr, nv::slot_limit_group_error_t* group_error = nullptr);
    /** @brief Send an SCF FAPI error indication for an explicit cell index (transport / L1 path).
     *  @param[in] msg_id FAPI message ID the error is associated with.
     *  @param[in] error_code SCF error code placed in the indication.
     *  @param[in] sfn System frame number for the indication.
     *  @param[in] slot Slot index for the indication.
     *  @param[in] cell_idx Carrier index used to select the IPC transport.
     *  @param[in] log_info If true, log the outgoing indication at info level; if false, at error/event level.
     *  @param[in] total_errors Optional PDU count when populating error extension fields.
     *  @param[in] cell_error Optional per-cell L1 limit error context for extension payload.
     *  @param[in] group_error Optional group L1 limit error context for extension payload.
     */
    void send_error_indication_l1(scf_fapi_message_id_e msg_id,  scf_fapi_error_codes_t error_code, uint16_t sfn, uint16_t slot, int32_t cell_idx, bool log_info = false, uint16_t total_errors=0, nv::slot_limit_cell_error_t* cell_error = nullptr, nv::slot_limit_group_error_t* group_error = nullptr);
    void send_released_harq_buffer_error_indication(const ReleasedHarqBufferInfo &released_harq_buffer_info, const slot_command_api::pusch_params* params, uint16_t sfn, uint16_t slot);
    void update_ssb_config();
    void update_prach_addln_configs();
    void update_prach_configs_l1(nv::PHYDriverProxy& phyDriver);
    void update_phy_stat_configs_l1(nv::PHYDriverProxy& phyDriver);
    void update_pusch_power_control_configs(nv::PHYDriverProxy& phyDriver);
    void update_cell_stat_prm_idx();
    void update_prach_start_ro_index(nv::PHYDriverProxy& phyDriver);
    void update_phy_driver_info_reconfig(nv::PHYDriverProxy& phyDriver, const int32_t cell_id);
    void update_cell_reconfig_params();
    void cell_update_success(nv::PHYDriverProxy& phyDriver, const int32_t cell_id);
    uint8_t create_cell_l1(nv::PHYDriverProxy& phyDriver);
    uint8_t create_cell_configs();
    void copy_phy_configs_from(nv::phy_config& phy_config_origin);
    void update_cell_state(fapi_state_t other_state);
    void update_tx_rx_ants();
    void update_cells_stats(int32_t);
    void copy_precoding_configs_to(int32_t cell_id);
    inline ru_type get_ru_type();
    inline nv::slot_detail_t* get_slot_detail(slot_command_api::slot_indication& slot);
    inline bfw_coeff_mem_info_t* get_free_static_bfw_index(int32_t cell_id);
    inline uint8_t get_mMIMO_enable_info();
    inline uint8_t get_enable_srs_info();
    // RSSI
    inline void setRssiMeasurement(uint8_t rssiMeasurement_) {phy_config.meas_config_.rssiMeasurement = rssiMeasurement_; }
    inline uint8_t getRssiMeasurement() {return phy_config.meas_config_.rssiMeasurement;}
    // RSRP
    inline void setRsrpMeasurement(uint8_t rsrpMeasurement_) {phy_config.meas_config_.rsrpMeasurement = rsrpMeasurement_; }
    inline uint8_t getRsrpMeasurement() {return phy_config.meas_config_.rsrpMeasurement;}
    // PN Measurement
    inline void setPnMeasurement(uint8_t pnMeasurement_) {phy_config.vendor_config_.pnMeasurement = pnMeasurement_;}
    inline uint8_t getPnMeasurement() {return phy_config.vendor_config_.pnMeasurement;}

    //  PF 234 interference
    inline void setPf234Interference(uint8_t pf_234_interference_) {phy_config.vendor_config_.pf_234_interference = pf_234_interference_;}
    inline uint8_t getPf234Interference () {return phy_config.vendor_config_.pf_234_interference;}

    // PRACH Interference
    inline void setPrachInterference (uint8_t prach_interference_) { phy_config.vendor_config_.prach_interference = prach_interference_;}
    inline uint8_t getPrachInterference() {return phy_config.vendor_config_.prach_interference;}

    // SRS Chest Buffer Size requested by L2
    inline void setSrsChestBuffSize (uint32_t srsChest_buff_size_) { phy_config.vendor_config_.srsChest_buff_size = srsChest_buff_size_;}
    inline uint32_t getSrsChestBuffSize() {return phy_config.vendor_config_.srsChest_buff_size;}

    // PUSCH Aggregation Factor requested by L2
    inline void setPuschAggrFactor(uint8_t pusch_aggr_factor_) { phy_config.vendor_config_.pusch_aggr_factor = pusch_aggr_factor_;}
    inline uint8_t getPuschAggrFactor() {return phy_config.vendor_config_.pusch_aggr_factor;}

    // CSI2 Maps
    void copy_csi2_maps_from(uint16_t nCsi2MapsOther, uint16_t* csi2MapBufferOther, cuphyCsi2MapPrm_t * csi2MapParamsBufferOther);
    
    // DBT PDU Table Pointer
    int update_dbt_pdu_table_ptr(int32_t cell_id, void* dbt_pdu_table_ptr);
private:
    int check_sfn_slot(int cell_id, int msg_id, sfn_slot_t ss_msg);
    bool can_handle_msg(uint8_t typeId);

    // The FAPI state of the cell, accessed by multiple threads so use atomic
    std::atomic<fapi_state_t> state = fapi_state_t::FAPI_STATE_IDLE;

    nv::phy_config phy_config;
    nv::cell_update_config cell_update_config;
    ::cell_phy_info phy_driver_info;
    //temporay hold phy_driver_info for cell update till reconfiguration is verified
    ::cell_phy_info cell_reconfig_phy_driver_info;
    cuphyCellStatPrm_t phy_cell_params;
    //temporay hold phy_cell_params for cell update till reconfiguration is verified
    cuphyCellStatPrm_t cell_reconfig_phy_cell_params;
    //Change for CSI part 2
    cuphyPuschCellStatPrm_t pusch_cell_stat_params;
    cuphyPucchCellStatPrm_t pucch_cell_stat_params;
    uint32_t dl_pdu_index[MAX_PDSCH_UE_GROUPS];
    uint32_t dl_pdu_index_size;
    bool valid_dci_rx = false;

    /** True when PDSCH was rejected this slot (e.g. check_bf_pc_params failed). Persists across messages so TX_DATA can be dropped when it arrives separately. */
    bool pdsch_rejected_ = false;

    // Duplicate message check
    bool duplicate_dl_tti_req = false;
    bool duplicate_ul_tti_req = false;
    bool duplicate_tx_data_req = false;
    bool duplicate_ul_dci_req = false;
    bool duplicate_dl_bfw_cvi_req = false;
    bool duplicate_ul_bfw_cvi_req = false;

    // The start index of cuphyPdschCwPrm_t array for updating tbStartOffset from TX_DATA.req
    uint32_t pdsch_cw_idx_start;

    std::vector<uint64_t> layer_map;
    uint8_t l_max;
    std::set<uint8_t> ssb_slot_index;
    metrics metrics_;
    nv::prach_addln_config_t prach_addln_config;

    uint16_t pucch_hopping_id_{};

    // UL RSSI
    float beta;
    float beta_sq;
    int fs_offset_ul;
    int ul_bitwidth;
    //uint8_t pf_01_interference;

    nv::ssb_case ssb_case;
    const uint16_t* lmax_symbol_list;

    uint32_t allowed_fapi_latency;
    float prach_ta_offset_usec_;
    float non_prach_ta_offset_usec_;

    bool cell_created = false;
    tx_data_req_meta_data_t tx_data_req_meta_data_;
    int8_t cell_stat_prm_idx;
    static bool first_config_req;
    static std::vector<uint32_t> first_config_req_pmidxes;
    uint16_t nCsi2Maps;
    cuphy::unique_pinned_ptr<uint16_t> csi2MapCpuBuffer;
    cuphy::unique_pinned_ptr<cuphyCsi2MapPrm_t> csi2MapParamsCpuBuffer;

    void* dbt_pdu_table_ptr{nullptr};
};

} // namespace scf_5g_fapi

#endif // !defined(SCF_5G_FAPI_PHY_HPP_INCLUDED_)
