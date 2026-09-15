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

#if !defined(NV_PHY_INSTANCE_HPP_INCLUDED_)
#define NV_PHY_INSTANCE_HPP_INCLUDED_

#include "yaml.hpp"
#include "nv_ipc.h"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi.h"

#include "slot_command/slot_command.hpp"

#include <unordered_map>
#include <cstdint>
#include <type_traits>

using prach_dyn_idx_map_t = std::unordered_map <uint32_t, uint16_t>;

// TODO: remove once all channels are in parallel worker path
//
// Bit layout aligned with SCF FAPI PDU-type enums:
//   Bits 0-3  : DL_TTI.req PDU types (matches scf_fapi_dl_tti_pdu_type_t)
//   Bits 4-7  : UL_TTI.req PDU types (matches scf_fapi_ul_tti_pdu_type_t + 4)
//   Bits 8-11 : Single-operation messages (TX_DATA, UL_DCI, DL_BFW, UL_BFW)
enum class FapiSkip : uint32_t {
    None      = 0,
    PDCCH     = 1U << 0,   // DL_TTI_PDU_TYPE_PDCCH
    PDSCH     = 1U << 1,   // DL_TTI_PDU_TYPE_PDSCH
    CSI_RS    = 1U << 2,   // DL_TTI_PDU_TYPE_CSI_RS
    SSB       = 1U << 3,   // DL_TTI_PDU_TYPE_SSB
    PRACH     = 1U << 4,   // UL_TTI_PDU_TYPE_PRACH
    PUSCH     = 1U << 5,   // UL_TTI_PDU_TYPE_PUSCH
    PUCCH     = 1U << 6,   // UL_TTI_PDU_TYPE_PUCCH
    SRS       = 1U << 7,   // UL_TTI_PDU_TYPE_SRS
    TX_DATA   = 1U << 8,   // SCF_FAPI_TX_DATA_REQUEST
    UL_DCI    = 1U << 9,   // SCF_FAPI_UL_DCI_REQUEST
    DL_BFW    = 1U << 10,  // SCF_FAPI_DL_BFW_CVI_REQUEST
    UL_BFW    = 1U << 11,  // SCF_FAPI_UL_BFW_CVI_REQUEST
};

constexpr FapiSkip operator|(FapiSkip a, FapiSkip b) noexcept {
    using U = std::underlying_type_t<FapiSkip>;
    return static_cast<FapiSkip>(static_cast<U>(a) | static_cast<U>(b));
}

constexpr bool operator&(FapiSkip a, FapiSkip b) noexcept {
    using U = std::underlying_type_t<FapiSkip>;
    return (static_cast<U>(a) & static_cast<U>(b)) != 0;
}

class SlotMapDl;
class SlotMapUl;

namespace nv
{
class PHY_module;
    
/**
 * @brief Base class for PHY instances
 *
 * Provides the foundation for specific PHY implementations. Derived classes
 * implement message handlers for their specific protocol needs.
 */
class PHY_instance
{
public:
    /**
     * @brief Constructor
     *
     * @param phy_module Reference to the parent PHY module
     * @param node_config YAML configuration node containing instance settings
     */
    explicit PHY_instance(PHY_module& phy_module, yaml::node node_config) :
        module_(&phy_module),
        name_(node_config["name"].as<std::string>()),
        current_cmd()
    {
    }

    PHY_instance(const PHY_instance&) = delete;
    PHY_instance& operator=(const PHY_instance&) = delete;

    virtual ~PHY_instance() {}

    /**
     * @brief Get the instance name
     * @return C-string containing the instance name
     */
    const char* name() { return name_.c_str(); }
    
    /** Handle incoming IPC message (convenience wrapper).
     *
     * Non-virtual overload that forwards to the two-parameter form with
     * @p mask set to FapiSkip::None (all legacy processing paths active).
     *
     * @param[in] msg  Reference to the IPC message.
     * @return true if message was handled successfully, false otherwise.
     * @sa on_msg(nv_ipc_msg_t&, FapiSkip) for the overridable implementation.
     */
    [[nodiscard]] bool on_msg(nv_ipc_msg_t& msg) { return on_msg(msg, FapiSkip::None); }

    /** Handle incoming IPC message (implementation entry point).
     *
     * Pure virtual function that derived classes must override. Callers that
     * need to selectively disable legacy processing paths should invoke this
     * overload directly with the appropriate @p mask bits.
     *
     * @param[in] msg   Reference to the IPC message.
     * @param[in] mask  Bitmask of legacy processing paths to skip.
     *                  See @c FapiSkip enum above for the bit constants.
     *                  Pass FapiSkip::None (default via convenience wrapper)
     *                  to run all paths.
     * @return true if message was handled successfully, false otherwise.
     * @note  Derived classes must override this overload, not the
     *        single-parameter convenience wrapper.
     */
    [[nodiscard]] virtual bool on_msg(nv_ipc_msg_t& msg, FapiSkip mask) = 0;

#ifdef ENABLE_FAPI_STORE_REPLAY
    /**
     * @brief Handle incoming IPC message for storage-only paths
     *
     * @param[in] msg Reference to the IPC message.
     * @param[in] slot_u32 Slot in u32 SFN/SLOT format.
     * @return true if message was stored successfully, false otherwise
     *
     * @note Store-replay only: routes to PHY_module::store_message, whose
     *       definition lives in the store-replay-gated translation unit.
     */
    [[nodiscard]] bool on_msg_to_store(const phy_mac_msg_desc& msg, uint32_t slot_u32);
#endif

    /**
     * @brief Reset slot state
     * @param reset_flag Flag indicating type of reset
     */
    virtual void reset_slot(bool) = 0;

    /**
     * @brief Finalize minimal store-replay slot state after submit.
     */
    virtual void on_store_replay_slot_complete() {}

    /**
     * @brief Execute an offloaded CONFIG.request on the low-priority worker.
     */
    virtual void on_config_request_offload([[maybe_unused]] int32_t cell_id,
                                           [[maybe_unused]] phy_mac_msg_desc& msg) {}

    /**
     * @brief Execute an offloaded START.request on the low-priority worker.
     * @param cell_id Carrier index associated with the queued request.
     * @note Default implementation is a no-op for PHY_instance types that do
     *       not support non-slot offload execution.
     */
    virtual void on_cell_start_request_offload([[maybe_unused]] int32_t cell_id) {}

    /**
     * @brief Execute an offloaded STOP.request on the low-priority worker.
     * @param cell_id Carrier index associated with the queued request.
     * @note Default implementation is a no-op for PHY_instance types that do
     *       not support non-slot offload execution.
     */
    virtual void on_cell_stop_request_offload([[maybe_unused]] int32_t cell_id) {}

    /**
     * @brief Publish DL PDSCH byte/slot counter deltas for this PHY instance type.
     *
     * Store-replay DL aggregation runs in nvPHY, while the SCF adapter owns the
     * PHY counter storage.  The default implementation is a no-op for non-SCF
     * instances; scf_5g_fapi::phy overrides it and updates its counters.
     *
     * Called once per active cell at DL_TTI batch boundary, not per PDU.
     */
    virtual void publish_dl_pdsch_stats([[maybe_unused]] uint32_t cell_id,
                                        [[maybe_unused]] uint64_t bytes,
                                        [[maybe_unused]] uint32_t slots) noexcept
    {}

    /**
     * @brief Send DL O-RAN Section Type 1 C-plane to RU for this cell.
     * Called from task_work_fn_dlc() on a DL worker thread.
     * Pure virtual — all concrete subclasses must provide an implementation.
     * @param dl_tti Message descriptor for DL_TTI_REQ (may be null).
     * @param ul_dci Message descriptor for UL_DCI_REQ (may be null).
     *               At least one is non-null with actual PDU content.
     * @return 0 on success; non-zero error code on failure (e.g., timing-window
     *         miss). Return value must be checked.
     */
    [[nodiscard]] virtual int build_and_send_dl_cplane(
        const phy_mac_msg_desc* dl_tti,
        const phy_mac_msg_desc* ul_dci,
        std::size_t transaction_id,
        SlotMapDl* slot_map_dl = nullptr) = 0;

    /**
     * @brief Build and send UL C-plane for a single cell.
     *
     * Invoked from task_work_fn_ulc (worker thread) once per cell on UL_TTI.req
     * arrival. Derived classes populate the UL slot command with PUSCH/PRACH/
     * PUCCH/SRS parameters from the stored UL_TTI.req message.
     *
     * @param ul_tti  Message descriptor for UL_TTI_REQ (non-null when invoked).
     * @return 0 on success; non-zero error code on failure (e.g., timing-window
     *         miss). Return value must be checked.
     */
    [[nodiscard]] virtual int build_and_send_ul_cplane(
        const phy_mac_msg_desc* ul_tti,
        std::size_t transaction_id,
        SlotMapUl* slot_map_ul = nullptr) = 0;

    /**
     * @brief Whether DL BFW CVI may be applied for this cell (SRS + mMIMO enabled).
     *
     * Matches the legacy @c on_msg gate around @c on_dl_bfw_request().
     * Default @c true for stubs; @c scf_5g_fapi::phy overrides using driver feature flags.
     */
    [[nodiscard]] virtual bool is_dl_bfw_cvi_feature_allowed() noexcept { return true; }

    /**
     * @brief Reject a stored DL BFW CVI when @ref is_dl_bfw_cvi_feature_allowed() is false.
     *
     * Legacy sends @c SCF_ERROR_CODE_MSG_SLOT_ERR. Default is no-op.
     */
    virtual void reject_dl_bfw_cvi_feature_disabled(uint16_t msg_type_id,
                                                    uint16_t sfn,
                                                    uint16_t slot) noexcept
    {
        (void)msg_type_id;
        (void)sfn;
        (void)slot;
    }

    /**
     * @brief Whether UL BFW CVI may be applied for this cell (SRS + mMIMO enabled).
     *
     * Matches the legacy @c on_msg gate around @c on_ul_bfw_request().
     * Default @c true for stubs; @c scf_5g_fapi::phy overrides using driver feature flags.
     */
    [[nodiscard]] virtual bool is_ul_bfw_cvi_feature_allowed() noexcept { return true; }

    /**
     * @brief Reject a stored UL BFW CVI when @ref is_ul_bfw_cvi_feature_allowed() is false.
     *
     * Legacy sends @c SCF_ERROR_CODE_MSG_SLOT_ERR. Default is no-op.
     */
    virtual void reject_ul_bfw_cvi_feature_disabled(uint16_t msg_type_id,
                                                    uint16_t sfn,
                                                    uint16_t slot) noexcept
    {
        (void)msg_type_id;
        (void)sfn;
        (void)slot;
    }

    /**
     * @brief Apply a single DL BFW PDU to the slot command.
     *
     * Called from PHY_module::process_aggr_dlbfw_channel() once per PDU
     * in a DL_BFW_CVI_REQUEST message. Pure virtual — each concrete PHY
     * implementation must supply this (typically by calling
     * update_cell_command() with per-cell static parameters).
     *
     * @param[in]     pdu           DL BFW group configuration PDU
     * @param[in]     slot_ind      Slot indication (SFN/slot from the request)
     * @param[in,out] grp_cmd       Cell group command for this slot (from slot_command_array)
     * @param[in,out] cell_cmd      Cell sub-command for this cell (from slot_command_array)
     * @param[in,out] bfw_info      BFW coefficient memory buffer for this cell/slot
     * @param[in,out] droppedBFWPdu Running count of dropped BFW PDUs (incremented on failure)
     */
    virtual void apply_dl_bfw_pdu(
        const scf_fapi_dl_bfw_group_config_t& pdu,
        slot_command_api::slot_indication& slot_ind,
        slot_command_api::cell_group_command* grp_cmd,
        slot_command_api::cell_sub_command& cell_cmd,
        slot_command_api::bfw_coeff_mem_info_t* bfw_info,
        uint32_t& droppedBFWPdu) = 0;

    /**
     * @brief Apply a single UL BFW PDU to the slot command.
     *
     * Called from PHY_module::process_aggr_ulbfw_channel() once per PDU
     * in a UL_BFW_CVI_REQUEST message. Pure virtual — each concrete PHY
     * implementation must supply this (typically by calling
     * update_cell_command() with per-cell static parameters).
     *
     * @param[in]     pdu           UL BFW group configuration PDU
     * @param[in]     slot_ind      Slot indication (SFN/slot from the request)
     * @param[in,out] grp_cmd       Cell group command for this slot (from slot_command_array)
     * @param[in,out] cell_cmd      Cell sub-command for this cell (from slot_command_array)
     * @param[in,out] bfw_info      BFW coefficient memory buffer for this cell/slot
     * @param[in,out] droppedBFWPdu Running count of dropped BFW PDUs (incremented on failure)
     */
     virtual void apply_ul_bfw_pdu(
        const scf_fapi_ul_bfw_group_config_t& pdu,
        slot_command_api::slot_indication& slot_ind,
        slot_command_api::cell_group_command* grp_cmd,
        slot_command_api::cell_sub_command& cell_cmd,
        slot_command_api::bfw_coeff_mem_info_t* bfw_info,
        uint32_t& droppedBFWPdu) = 0;

    /**
     * @brief Send SCF FAPI error indications for BFW PDUs that were dropped.
     *
     * Called from PHY_module::process_aggr_dlbfw_channel() /
     * process_aggr_ulbfw_channel() once per BFW request after all PDUs in
     * the request have been applied, when one or more PDUs were dropped by
     * apply_dl_bfw_pdu() / apply_ul_bfw_pdu().
     *
     * Pure virtual — each concrete PHY implementation must supply this
     * (e.g. emit one SCF_ERROR_CODE_SRS_CHEST_BUFF_BAD_STATE error indication
     * per dropped PDU), matching the non-aggregated on_dl_bfw_request() /
     * on_ul_bfw_request() path when @p dropped_count is greater than zero.
     *
     * @param[in] msg_type_id   FAPI message type id (e.g. SCF_FAPI_DL_BFW_CVI_REQUEST)
     * @param[in] sfn           SFN of the request
     * @param[in] slot          Slot of the request
     * @param[in] dropped_count Number of dropped BFW PDUs; when zero, do nothing
     */
    virtual void send_bfw_error_indications(uint16_t msg_type_id,
        uint16_t sfn,
        uint16_t slot,
        uint32_t dropped_count) = 0;

    virtual void send_fapi_error_indication(scf_fapi_message_id_e,
                                            scf_fapi_error_codes_t,
                                            uint16_t,
                                            uint16_t) {}

    /**
     * @brief Reset PHY state for L2 reconnecting
     * @return 0 on success, negative error code on failure
     */
    virtual int reset() = 0;

    /**
     * @brief Get the 3GPP Physical Cell ID (PCI) for this instance.
     *
     * Per-cell identity accessor used by all parallel aggregation paths that
     * need to address cell-level slot commands via the PCI registered with
     * cuphydriver (CSI-RS today; TX_DATA.req and other DL/UL channels as
     * they migrate to the parallel aggregation path).
     *
     * @return PCI (0-1007) as registered with cuphydriver via l1_cell_start.
     */
    [[nodiscard]] virtual uint16_t get_phy_cell_id() const noexcept = 0;

    /**
     * @brief Get the carrier index for this instance.
     *
     * Per-cell identity accessor used by all parallel aggregation paths to
     * index into slot_command_array cell entries and to pass the correct
     * cell_index to slot-command helpers (CSI-RS today; TX_DATA.req and other
     * DL/UL channels as they migrate to the parallel aggregation path).
     *
     * @return Carrier index (0-based ordinal assigned at cell configuration).
     */
    [[nodiscard]] virtual int32_t get_carrier_id() const noexcept = 0;

protected:
    /**
     * @brief Get reference to parent PHY module
     * @return Reference to the PHY_module
     */
    PHY_module& phy_module() { return *module_; }
    
private:
    friend class PHY_module;
    
    /**
     * @brief Set the parent PHY module
     * @param m Reference to the PHY_module to set
     */
    void set_module(PHY_module& m) { module_ = &m; }
    
    PHY_module* module_;  ///< Pointer to parent PHY module
    std::string name_;    ///< Instance name
    
protected:
    /**
     * @brief Command type enumeration
     */
    enum class command_t : uint32_t
    {
        COMMAND_SLOT = 0,         ///< Per slot command
        COMMAND_CONFIGURE = 1,    ///< Configure command
        COMMAND_START = 2,        ///< Start command
        COMMAND_STOP = 3          ///< Stop command
    };

    command_t current_cmd;              ///< Current command being processed
    nv::phy_mac_msg_desc cur_dl_msg;    ///< Store the current TB received
    
protected:
    /**
     * @brief Send slot indication message
     * @param slot_3gpp Slot indication structure
     */
    virtual void send_slot_indication(slot_command_api::slot_indication& slot_3gpp) {}
    
    /**
     * @brief Send slot error indication message
     * @param slot_3gpp Slot indication structure
     */
    virtual void send_slot_error_indication(slot_command_api::slot_indication& slot_3gpp) {}
    
    /**
     * @brief Create uplink/downlink callbacks
     * @param cb Callbacks structure to populate
     */
    virtual void create_ul_dl_callbacks(slot_command_api::callbacks& cb) {}
    
    /**
     * @brief Send PHY L1 enqueue error indication
     * @param sfn System frame number
     * @param slot Slot number
     * @param ul_slot True if uplink slot, false otherwise
     * @param cell_id_list Array of cell IDs
     * @param index Index into cell_id_list
     */
    virtual void send_phy_l1_enqueue_error_indication(uint16_t sfn,uint16_t slot,bool ul_slot,std::array<int32_t,MAX_CELLS_PER_SLOT>& cell_id_list,int32_t& index) {}
    
    /**
     * @deprecated Use handle_cell_config_response() instead
     * @brief Send cell configuration response
     * @param cell_id Cell identifier
     * @param response_code Response code value
     */
    [[deprecated("Use handle_cell_config_response()")]]
    virtual void send_cell_config_response(int32_t cell_id, uint8_t response_code)
    {
        handle_cell_config_response(cell_id, response_code);
    }

    /**
     * @brief Handle cell configuration response
     * @param cell_id Cell identifier
     * @param response_code Response code value
     */
    virtual void handle_cell_config_response(int32_t cell_id, uint8_t response_code) {}
};

} // namespace nv

#endif // !defined(NV_PHY_INSTANCE_HPP_INCLUDED_)
