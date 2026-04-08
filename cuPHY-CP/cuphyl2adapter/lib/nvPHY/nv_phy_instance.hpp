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

#include "slot_command/slot_command.hpp"

#include <unordered_map>

using prach_dyn_idx_map_t = std::unordered_map <uint32_t, uint16_t>;

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
    
    /**
     * @brief Handle incoming IPC message
     *
     * Pure virtual function that must be implemented by derived classes.
     *
     * @param msg Reference to the IPC message
     * @return true if message was handled successfully, false otherwise
     */
    virtual bool on_msg(nv_ipc_msg_t& msg) = 0;

    /**
     * @brief Reset slot state
     * @param reset_flag Flag indicating type of reset
     */
    virtual void reset_slot(bool) = 0;

    /**
     * @brief Reset PHY state for L2 reconnecting
     * @return 0 on success, negative error code on failure
     */
    virtual int reset() = 0;

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
