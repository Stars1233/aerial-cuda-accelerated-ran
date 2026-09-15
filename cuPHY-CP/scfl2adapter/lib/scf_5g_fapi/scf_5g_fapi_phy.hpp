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
#include "scf_5g_fapi_tags.hpp"

#include <wise_enum/wise_enum.h>

#include <map>
#include <unordered_map>
#include <set>
#include <chrono>
#include <span>

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

//! Highest configured cell index + 1; bounds phy::print_cell_stats() iteration.
//! Defined in scf_5g_fapi_phy.cpp.
extern uint32_t total_cell_num;

//! Synthetic phyCellId values assigned per cell index when duplicating one
//! cell's config across all cells (config_options().duplicateConfigAllCells).
//! Shared by the serial (on_config_request) and offload (on_config_request_offload)
//! paths so the two never drift.
inline constexpr std::array<uint16_t, MAX_CELLS_PER_SLOT> fake_phy_cell_id = {
    1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
    1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027};

//! SSB_MASK TLV decode policy for the shared phy::decode_config_tlvs(). Kept
//! deliberately distinct so this dedup does not change the serial path. Defined
//! via WISE_ENUM_CLASS so the mode is recoverable as a string for diagnostics.
WISE_ENUM_CLASS((SsbMaskDecode, std::uint8_t),
    LegacyStaticToggle,  //!< Serial path: original process-static toggle (mask[0] then mask[1]; state persists across requests). Preserved verbatim.
    PerCallIndexed);     //!< Offload path: per-call, bounds-checked index.

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
    [[nodiscard]] bool on_msg(nv_ipc_msg_t& msg, FapiSkip mask) override;
    
    /**
     * @brief Reset slot-specific state
     * @param force_reset Force reset even if conditions not met
     */
    virtual void reset_slot(bool) override;

    void on_store_replay_slot_complete() override;

    /**
     * @brief Send DL O-RAN Section Type 1 C-plane to RU for this cell.
     * Called from task_work_fn_dlc() on a DL worker thread.
     * Implementation deferred — will extract C-plane from on_dl_tti_request().
     *
     * @param[in] dl_tti Stored DL_TTI.request descriptor.
     * @param[in] ul_dci Stored UL_DCI.request descriptor.
     * @param[in] slot_map_dl Slot-map pointer for DL C-plane task accounting.
     */
    [[nodiscard]] int build_and_send_dl_cplane(
        const nv::phy_mac_msg_desc* dl_tti,
        const nv::phy_mac_msg_desc* ul_dci,
        std::size_t transaction_id,
        SlotMapDl* slot_map_dl = nullptr) override;

    /**
     * @brief Send UL O-RAN Section Type 3 C-plane to RU for this cell.
     * Called from task_work_fn_ulc() on a UL worker thread.
     * Implementation deferred — will extract C-plane from on_ul_tti_request().
     *
     * @param[in] ul_tti Stored UL_TTI.request descriptor.
     * @param[in] slot_map_ul Slot-map pointer for UL C-plane task accounting.
     */
    [[nodiscard]] int build_and_send_ul_cplane(
        const nv::phy_mac_msg_desc* ul_tti,
        std::size_t transaction_id,
        SlotMapUl* slot_map_ul = nullptr) override;

    /**
     * @brief Reset PHY state for L2 reconnection
     * @return 0 on success, negative error code on failure
     */
    virtual int reset() override;

    [[nodiscard]] int32_t get_carrier_id() const noexcept override { return phy_config.cell_config_.carrier_idx; }

    /** @brief Return the PHY cell ID (PCI) for this instance. @return PCI (0-1007). */
    [[nodiscard]] uint16_t get_phy_cell_id() const noexcept override { return phy_config.cell_config_.phy_cell_id; }
    
    /**
     * @brief Process CONFIG.request message
     * @param config_request Configuration request parameters
     * @param cell_id Cell identifier
     * @param handle_id Handle identifier for response
     * @param ipc_msg IPC message structure
     */
    void on_config_request(scf_fapi_config_request_msg_t& config_request, const int32_t cell_id, uint8_t handle_id, nv_ipc_msg_t& ipc_msg);

    /**
     * @brief Decode the full (initial-config) CONFIG.request TLV set + default fixups.
     *
     * Fills phy_config / phy_cell_params and applies the SRS-chest / muMIMO port
     * default fixups. Shared verbatim by the serial (on_config_request) and offload
     * (on_config_request_offload) fresh-config paths so the ~68-case TLV decoder is
     * defined once. The DBT PDU buffer is supplied via @p dbt_data / @p dbt_data_pool
     * (nv_ipc_msg_t in the serial path, phy_mac_msg_desc in the offload path).
     *
     * @param[in,out] config_request CONFIG.request message (non-const: scf_fapi_tl_t accessors are non-const).
     * @param[in]     cell_id        Cell being configured.
     * @param[in]     dbt_data       DBT PDU data buffer (may be empty).
     * @param[in]     dbt_data_pool  DBT PDU data pool id (compared against NV_IPC_MEMPOOL_CPU_LARGE).
     * @param[in]     ssb_mask_mode  SSB_MASK decode policy: the serial path passes
     *                               LegacyStaticToggle to preserve its original behavior byte-for-byte;
     *                               the offload path passes PerCallIndexed (bounds-checked per-call).
     * @return FAPI error code (SCF_ERROR_CODE_MSG_OK on success).
     */
    [[nodiscard]] uint8_t decode_config_tlvs(scf_fapi_config_request_msg_t& config_request, int32_t cell_id, std::span<uint8_t> dbt_data, int32_t dbt_data_pool, SsbMaskDecode ssb_mask_mode);

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

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Process CONFIG.request on the low-priority offload worker.
     * @param cell_id Cell identifier
     * @param msg IPC message descriptor containing the CONFIG.request payload.
     */
    void on_config_request_offload(int32_t cell_id, nv::phy_mac_msg_desc& msg) override;

    /**
     * @brief Process START.request on the low-priority offload worker.
     * @param cell_id Cell identifier
     *
     * @note Declared only when ENABLE_FAPI_STORE_REPLAY is defined; the
     *       definition lives in scf_5g_fapi_phy_non_slot_offload.cpp, which
     *       is compiled only under the same guard (see CMakeLists.txt). When
     *       store-replay is disabled, the no-op default in
     *       nv::PHY_instance::on_cell_start_request_offload is inherited.
     */
    void on_cell_start_request_offload(int32_t cell_id) override;
#endif

    /**
     * @brief Process STOP.request message
     * @param cell_id Cell identifier
     */
    void on_cell_stop_request(const int32_t cell_id);

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Process STOP.request on the low-priority offload worker.
     * @param cell_id Cell identifier
     *
     * @note Declared only when ENABLE_FAPI_STORE_REPLAY is defined; see the
     *       sibling start-offload override above for the rationale.
     */
    void on_cell_stop_request_offload(int32_t cell_id) override;
#endif
    /**
     * @brief Process DL_TTI.request message
     * @param msg DL TTI request parameters
     * @param ipc_msg IPC message structure
     * @param pdsch_valid_flag Array of PDSCH validation flags
     * @param mask Bitmask of legacy paths to skip (see FapiSkip enum)
     */
    void on_dl_tti_request(scf_fapi_dl_tti_req_t &msg, nv_ipc_msg_t& ipc_msg, uint8_t* pdsch_valid_flag, FapiSkip mask);
    
    /**
     * @brief Process UL_TTI.request message
     * @param request UL TTI request parameters
     * @param ipc_msg IPC message structure
     * @param mask Bitmask of legacy paths to skip (see FapiSkip enum)
     */
    void on_ul_tti_request(scf_fapi_ul_tti_req_t& request, nv_ipc_msg_t& ipc_msg, FapiSkip mask);
    
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
    
    /**
     * Send RX data indication for PUSCH transport block
     *
     * @param[in] slot Slot indication containing SFN and slot number
     * @param[in] params PUSCH parameters for the transport block
     * @param[in] out PUSCH data output from cuPHY processing
     * @param[in] puschStatPrms PUSCH static parameters
     * @param[in] ext_buffer Optional pre-allocated output message buffer for zero-copy.
     *            If nullptr (default), buffer is allocated internally via tx_alloc.
     *            If non-null, the provided nvIPC message(s) are used directly for
     *            the indication payload, avoiding an extra memcpy. Ownership is
     *            transferred on success (consumed by tx_send); on failure or when
     *            zero-copy conditions are not met, the buffer is released internally.
     */
    void send_rx_data_indication(const slot_command_api::slot_indication& slot,
        const slot_command_api::pusch_params& params, ::cuphyPuschDataOut_t const* out,
        ::cuphyPuschStatPrms_t const* puschStatPrms,
        slot_command_api::ul_output_msg_buffer* ext_buffer = nullptr);

    /**
     * Allocate external buffer for zero-copy UL transport block data
     *
     * Acquires pre-allocated nvIPC buffer(s) from the transport layer for the
     * given PUSCH parameters, enabling zero-copy D2H of uplink TB data.
     *
     * @param[out] buffer Output message buffer structure to populate with allocated buffer(s)
     * @param[in] params PUSCH parameters identifying the cells and UEs for this slot
     */
    void alloc_ul_tb_buffer(slot_command_api::ul_output_msg_buffer& buffer,
        const slot_command_api::pusch_params& params);
    /**
     * Release external buffer acquired for UL transport block data
     *
     * Returns nvIPC buffer(s) to the transport pool. Must be called for every
     * buffer successfully allocated via alloc_ul_tb_buffer that was not consumed
     * by tx_send (i.e., when zero-copy was not used or on error paths).
     *
     * @param[in,out] buffer Message buffer structure containing the buffer(s) to release
     * @param[in] params PUSCH parameters identifying the cells for transport lookup
     */
    void release_ul_tb_buffer(slot_command_api::ul_output_msg_buffer& buffer,
        const slot_command_api::pusch_params& params);
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

    void apply_dl_bfw_pdu(
        const scf_fapi_dl_bfw_group_config_t& pdu,
        slot_command_api::slot_indication& slot_ind,
        slot_command_api::cell_group_command* grp_cmd,
        slot_command_api::cell_sub_command& cell_cmd,
        slot_command_api::bfw_coeff_mem_info_t* bfw_info,
        uint32_t& droppedBFWPdu) override;
        
    void apply_ul_bfw_pdu(
        const scf_fapi_ul_bfw_group_config_t& pdu,
        slot_command_api::slot_indication& slot_ind,
        slot_command_api::cell_group_command* grp_cmd,
        slot_command_api::cell_sub_command& cell_cmd,
        slot_command_api::bfw_coeff_mem_info_t* bfw_info,
        uint32_t& droppedBFWPdu) override;

    virtual void send_bfw_error_indications(uint16_t msg_type_id,
        uint16_t sfn,
        uint16_t slot,
        uint32_t dropped_count) override;

    [[nodiscard]] bool is_dl_bfw_cvi_feature_allowed() noexcept override;
    void reject_dl_bfw_cvi_feature_disabled(uint16_t msg_type_id,
                                          uint16_t sfn,
                                          uint16_t slot) noexcept override;
    [[nodiscard]] bool is_ul_bfw_cvi_feature_allowed() noexcept override;
    void reject_ul_bfw_cvi_feature_disabled(uint16_t msg_type_id,
                                          uint16_t sfn,
                                          uint16_t slot) noexcept override;

    bool is_valid_dci_rx_slot() { return valid_dci_rx;}
    void set_valid_dci_rx_slot(bool value) {valid_dci_rx = value;}

    virtual void send_slot_indication(slot_command_api::slot_indication& slot_3gpp) override;
    virtual void send_slot_error_indication(slot_command_api::slot_indication& slot_3gpp) override;
    virtual void send_phy_l1_enqueue_error_indication(uint16_t sfn,uint16_t slot,bool ul_slot,std::array<int32_t,MAX_CELLS_PER_SLOT>& cell_id_list,int32_t& index) override;
    void publish_dl_pdsch_stats(uint32_t cell_id, uint64_t bytes, uint32_t slots) noexcept override;
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
    void send_fapi_error_indication(scf_fapi_message_id_e msg_id, scf_fapi_error_codes_t error_code, uint16_t sfn, uint16_t slot) override
    {
        send_error_indication(msg_id, error_code, sfn, slot);
    }
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
    inline bfw_coeff_mem_info_t* get_free_static_bfw_index(int32_t cell_id);
    /** Query whether multi-user MIMO is enabled on the L1 driver.
     *
     * Thin pass-through to nv::PHYDriverProxy::l1_mMIMO_enable_info(); the
     * proxy is fetched via its singleton accessor on every call.
     *
     * @return Non-zero when mMIMO is enabled in the L1 configuration, zero
     *         when disabled. The uint8_t width is preserved from the L1
     *         driver API to avoid implicit truncation at call sites that
     *         compose this with bitwise/boolean expressions.
     */
    [[nodiscard]] uint8_t get_mMIMO_enable_info();

    /** Query whether SRS (Sounding Reference Signal) processing is enabled
     *  on the L1 driver.
     *
     * Thin pass-through to nv::PHYDriverProxy::l1_enable_srs_info(); the
     * proxy is fetched via its singleton accessor on every call.
     *
     * @return Non-zero when SRS processing is enabled in the L1 configuration,
     *         zero when disabled. The uint8_t width is preserved from the L1
     *         driver API to avoid implicit truncation at call sites.
     */
    [[nodiscard]] uint8_t get_enable_srs_info();
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

    /**
     * Live L1 cell stat parameters (read-only).
     *
     * @return  Const reference to the live @c cuphyCellStatPrm_t for this cell.
     *          Return value must be checked.
     */
    [[nodiscard]] const cuphyCellStatPrm_t& get_phy_cell_params() const noexcept { return phy_cell_params; }
    /**
     * Live PHY config (read-only).
     *
     * Exposes the internal @c nv::phy_config to module-view accessors. Used by the
     * new PRACH parser (GT-11843) via @c PhyModuleView::phy_config(cell_id) to read
     * @c prach_config_ / @c cell_config_ without going through @c update_cell_command.
     *
     * @return  Const reference to the live @c nv::phy_config for this cell.
     *          Return value must be checked.
     */
    [[nodiscard]] const nv::phy_config& get_phy_config() const noexcept { return phy_config; }
    /**
     * Live PRACH additional config (read-only).
     *
     * Exposes @c prach_addln_config (l_ra / n_ra_slot / n_ra_dur / n_ra_rb / n_ra_t)
     * to module-view accessors. Used by the new PRACH parser to read additional
     * PRACH timing data populated by @c update_prach_addln_configs() at cell setup.
     *
     * @return  Const reference to the @c nv::prach_addln_config_t for this cell.
     *          Return value must be checked.
     */
    [[nodiscard]] const nv::prach_addln_config_t& get_prach_addln_config() const noexcept { return prach_addln_config; }
    /**
     * Staged cell stat parameters during cell reconfiguration (read-only).
     *
     * @return  Const reference to the staged @c cuphyCellStatPrm_t populated during
     *          a cell reconfiguration sequence.
     *          Return value must be checked.
     */
    [[nodiscard]] const cuphyCellStatPrm_t& get_cell_reconfig_phy_cell_params() const noexcept { return cell_reconfig_phy_cell_params; }
    /**
     * Per-cell static parameter index assigned during cell creation (read-only).
     *
     * @return  The @c int8_t static parameter index used to look up this cell's
     *          entry in the cuPHY static parameter table.
     *          Return value must be checked.
     */
    [[nodiscard]] int8_t get_cell_stat_prm_idx() const noexcept { return cell_stat_prm_idx; }

    /**
     * Return the RU type for this carrier from the M-plane configuration.
     *
     * Catches any exception thrown by @c getMPlaneConfig() (e.g.
     * @c std::runtime_error on invalid carrier_id) and returns
     * @c OTHER_MODE so callers remain @c noexcept-safe.
     *
     * @return RU type for this carrier; @c OTHER_MODE on lookup failure.
     */
    [[nodiscard]] ru_type get_ru_type() const noexcept
    {
        try {
            nv::PHYDriverProxy& phyDriver = nv::PHYDriverProxy::getInstance();
            ::cell_mplane_info& mplane = phyDriver.getMPlaneConfig(get_carrier_id());
            return mplane.ru;
        } catch (const std::runtime_error& ex) {
            NVLOGW_FMT(detail::k_tag,
                       "get_ru_type: getMPlaneConfig failed for carrier_id={} — returning OTHER_MODE: {}",
                       get_carrier_id(), ex.what());
            return OTHER_MODE;
        }
    }

    /**
     * Look up the TDD slot detail for the current slot.
     *
     * Returns a pointer into @c phy_config.tdd_table_.s_detail indexed by
     * @c slot.slot_ modulo @c repeat_slots_int, where @c repeat_slots_int is
     * derived from @c (1 << phy_config.ssb_config_.sub_c_common) multiplied by
     * the TDD period in milliseconds.  Only valid when the RU type is
     * @c SINGLE_SECT_MODE; returns @c nullptr for all other RU types.
     *
     * @param[in]  slot  Slot indication carrying the current slot index.
     *
     * @return  Pointer to the @c nv::slot_detail_t entry within
     *          @c phy_config.tdd_table_.s_detail for the computed slot index.
     *          The pointer refers into @c phy_config and must not be freed by
     *          the caller.  Return value must be checked.
     * @retval  nullptr  If @c get_ru_type() != @c SINGLE_SECT_MODE.
     * @retval  nullptr  If the computed @c repeat_slots_int is outside
     *                   <tt>[1, nv::NV_MAX_TDD_PERIODICITY]</tt>; a warning is
     *                   logged in this case.
     *
     * @note  This function is @c noexcept; exceptions from @c get_ru_type()
     *        are swallowed internally and cause @c OTHER_MODE to be returned,
     *        which triggers the @c nullptr path here.
     */
    [[nodiscard]] inline nv::slot_detail_t* get_slot_detail(
        slot_command_api::slot_indication const& slot) noexcept
    {
        auto ru = get_ru_type();
        if (ru == SINGLE_SECT_MODE) {
            const auto repeat_slots_float =
                static_cast<float>(1 << phy_config.ssb_config_.sub_c_common)
                * (nv::get_duration(phy_config.tdd_table_.tdd_period_num)
                   / std::chrono::duration<float, std::milli>(1));
            const auto repeat_slots_int = static_cast<int>(repeat_slots_float);
            if (repeat_slots_int <= 0 || repeat_slots_int > nv::NV_MAX_TDD_PERIODICITY) {
                NVLOGW_FMT(detail::k_tag,
                           "get_slot_detail: repeat_slots={} out of range [1,{}] — returning nullptr",
                           repeat_slots_int, nv::NV_MAX_TDD_PERIODICITY);
                return nullptr;
            }
            const auto slot_index = slot.slot_ % repeat_slots_int;
            return &phy_config.tdd_table_.s_detail[static_cast<std::size_t>(slot_index)];
        }
        return nullptr;
    }

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
