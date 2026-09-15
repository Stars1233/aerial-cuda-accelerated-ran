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

#if !defined(NV_PHY_MAC_TRANSPORT_HPP_INCLUDED_)
#define NV_PHY_MAC_TRANSPORT_HPP_INCLUDED_

// #include "nv_phy_module.hpp"
#include "nv_ipc.hpp"
#include "yaml.hpp"
#include "nvlog.hpp"

#include <limits>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <utility>

namespace nv
{

#define INVALID_CELL_ID (-1)

class PHY_module;

////////////////////////////////////////////////////////////////////////
// phy_mac_msg_desc
// Descriptor for a message allocated and released by the nvIPC library
struct phy_mac_msg_desc : public nv_ipc_msg_t
{
    phy_mac_msg_desc()
    {
        reset();
    }

    phy_mac_msg_desc(nv_ipc_msg_t msg)
    {
        cell_id = msg.cell_id;
        msg_id = msg.msg_id;
        msg_len = msg.msg_len;
        data_len = msg.data_len;
        msg_buf = msg.msg_buf;
        data_buf = msg.data_buf;
        data_pool = msg.data_pool;
    }

public:
    void reset()
    {
        cell_id = msg_id = msg_len = data_len = 0;
        data_pool = NV_IPC_MEMPOOL_CPU_MSG;
        msg_buf = data_buf = nullptr;
    }
};

////////////////////////////////////////////////////////////////////////
// phy_mac_transport
// Wrapper for the nv_ipc_t object in the nvIPC library
class phy_mac_transport
{
public:
    /**
     * Construct from a populated NVIPC configuration (e.g. after load_nv_ipc_yaml_config).
     *
     * @param[in,out] config      NVIPC settings passed to create_nv_ipc_interface (stored in config_).
     * @param[in]     cell_num_in MAC cell count used for bookkeeping and phy_cell mapping limits.
     */
    phy_mac_transport(nv_ipc_config_t& config, uint32_t cell_num_in);

    /**
     * Construct by parsing NVIPC fields from a YAML node, then opening the IPC interface.
     *
     * For SHM primary modules, per-pool pool_len values from YAML are multiplied by @p cell_num_in so total
     * SHM capacity scales with the number of cells.
     *
     * @param[in] node_config      YAML subtree containing NVIPC / transport settings.
     * @param[in] module_type      MAC, PHY, or primary module role for parsing and validation.
     * @param[in] cell_num_in      Number of MAC cells (SHM pool scaling, phy_cells list size).
     * @param[in] transport_id_in  Logical transport id for multi-transport logging and mapping.
     * @param[in] map_enable       When true, enables mapped / pinned buffer paths as configured.
     */
    phy_mac_transport(yaml::node node_config, nv_ipc_module_t module_type, uint32_t cell_num_in, int32_t transport_id_in = 0, bool map_enable = false);
    phy_mac_transport(phy_mac_transport&& t) :
        ipc_(std::move(t.ipc_)),
        config_(std::move(t.config_)),
        mapped(std::move(t.mapped)),
        transport_id(std::move(t.transport_id))
    {
    }
    //------------------------------------------------------------------
    // rx_wait()
    // (throws std::system_error on error)
    void rx_wait()
    {
        if(0 != ipc_->rx_tti_sem_wait(ipc_.get())) throw std::system_error(errno, std::generic_category(), "ipc connection error");
    }
    //------------------------------------------------------------------
    // tx_post()
    // (throws std::system_error on error)
    void tx_post()
    {
        if(0 != ipc_->tx_tti_sem_post(ipc_.get())) throw std::system_error(errno, std::generic_category(), "ipc connection error");
    }
    //------------------------------------------------------------------
    // tx_alloc()
    int tx_alloc(phy_mac_msg_desc& msg_desc, uint32_t options = 0);
    //------------------------------------------------------------------
    // tx_copy()
    void tx_copy(void* dst, const void* src, size_t size, int32_t data_pool);
    //------------------------------------------------------------------
    // rx_alloc()
    int rx_alloc(phy_mac_msg_desc& msg_desc, uint32_t options = 0);
    //------------------------------------------------------------------
    // tx_send()
    void tx_send(phy_mac_msg_desc& msg_desc);
#ifdef ENABLE_L2_SLT_RSP
    //------------------------------------------------------------------
    // tx_send_loopback() Send msg to itself
    void tx_send_loopback(phy_mac_msg_desc& msg_desc);
    //------------------------------------------------------------------
    // poll()
    int poll();
#endif
    //------------------------------------------------------------------
    // rx_recv()
    int rx_recv(phy_mac_msg_desc& msg_desc);
    //------------------------------------------------------------------
    // tx_release()
    void tx_release(phy_mac_msg_desc& msg_desc);
    //------------------------------------------------------------------
    // rx_release()
    void rx_release(phy_mac_msg_desc& msg_desc);
    //------------------------------------------------------------------
    // get_fd()
    int get_fd();
    //------------------------------------------------------------------
    // notify() Write tx_fd to notify the event
    int notify(int value);
    //------------------------------------------------------------------
    // get_value() Read rx_fd to clear the event
    int get_value();

    // Copy memory from msg_desc.data_buf to CPU dst_buf. Return 0 on success; return < 0 on failure.
    int copy_from_data_buf(nv_ipc_msg_t& msg_desc, uint32_t src_offset, void* dst_buf, size_t size);

    // Copy memory from CPU src_buf to msg_desc.data_buf. Return 0 on success; return < 0 on failure.
    int copy_to_data_buf(nv_ipc_msg_t& msg_desc, uint32_t dst_offset, void* src_buf, size_t size);

    nv_ipc_config_t* get_nv_ipc_config()
    {
        return &config_;
    }

    int64_t get_ts_send(nv_ipc_msg_t& msg_desc);

    int set_reset_callback(int (*callback)(phy_mac_transport*, PHY_module*), nv::PHY_module* phy_module)
    {
        reset_callback = callback;
        _phy_module    = phy_module;
        return 0;
    }

    bool is_started() {
        return started_cells_mask != 0 && error_flag == false;
    }

    bool get_mapped() {
        return mapped;
    }

    void set_mapped(bool val) {
        mapped = val;
    }

    int32_t get_transport_id() {
        return transport_id;
    }

    void set_transport_id(int32_t tid) {
        transport_id = tid;
    }

    int32_t get_phy_cell_id(int32_t mac_cell_id) {
        return mapped ? phy_cell_map[mac_cell_id] : mac_cell_id;
    }

    int32_t get_mac_cell_id(int32_t phy_cell_id) {
        return mapped ? mac_cell_map[phy_cell_id] : phy_cell_id;
    }

    std::unordered_map<int32_t, int32_t>& get_phy_cell_map()
    {
        return phy_cell_map;
    }

    bool get_error_flag()
    {
        return error_flag;
    }

    void set_error_flag(bool val)
    {
        error_flag = val;
    }

    /**
     * Sets the started state for a specific PHY cell in the cells mask
     *
     * Updates the internal bitmask to track which PHY cells are currently started/running.
     * Each bit in the mask represents a cell ID, where bit position corresponds to the cell ID.
     * This is used to track the operational state of individual cells in the transport.
     *
     * @param[in] phy_cell_id The PHY cell ID to set the state for (0-63)
     * @param[in] started True to mark the cell as started, false to mark as stopped
     */
    void set_started_cells_mask(int32_t phy_cell_id, bool started) {
        // Check if the cell_id is out of range
        if (phy_cell_id < 0 || phy_cell_id >= std::numeric_limits<uint64_t>::digits) {
            throw std::system_error(errno, std::generic_category(), "phy_cell_id out of range for the started_cells_mask");
        }

        if (started) {
            // Set the bit for the cell_id
            started_cells_mask.fetch_or(1ULL << phy_cell_id);
        } else {
            // Clear the bit for the cell_id
            started_cells_mask.fetch_and(~(1ULL << phy_cell_id));
        }
    }

    void reset_started_cells_mask() {
        started_cells_mask = 0;
    }

    uint64_t get_started_cells_mask() {
        return started_cells_mask;
    }

    int (*reset_callback)(phy_mac_transport* transp, PHY_module* phy_module) = nullptr;
    PHY_module *_phy_module = nullptr;

private:
    phy_mac_transport& operator=(const phy_mac_transport&) = delete;
    phy_mac_transport(const phy_mac_transport&) = delete;
    //------------------------------------------------------------------
    // cleanup_nv_ipc()
    // Custom deleter for unique_ptr
    static void cleanup_nv_ipc(nv_ipc_t* ipc);

    typedef std::unique_ptr<nv_ipc_t, decltype(&cleanup_nv_ipc)> nv_ipc_unique_ptr;

    //------------------------------------------------------------------
    // Data
    nv_ipc_unique_ptr ipc_;
    nv_ipc_config_t config_;

    uint32_t cell_num = 0; // Number of cells in this transport
    bool mapped = false; // False for single L2; True for Multi-L2.
    int32_t transport_id = 0;

    std::atomic<bool> error_flag = false;
    std::atomic<uint64_t> started_cells_mask = 0;

    std::unordered_map<int32_t, int32_t> mac_cell_map; // map<phy_cell_id, mac_cell_id>
    std::unordered_map<int32_t, int32_t> phy_cell_map; // map<mac_cell_id, phy_cell_id>
};

class phy_mac_transport_wrapper {

public:

    phy_mac_transport_wrapper(yaml::node node_config, nv_ipc_module_t module_type, uint32_t total_cell_num);

    ~phy_mac_transport_wrapper();

    void init(yaml::node& node_config, nv_ipc_module_t module_type);

    // Call the wrapper poll() to transverse all transport instances to see if there is any enqued msgs
    int poll();

    // Call the wrapper rx_recv() to transverse all transport instances
    int rx_recv(phy_mac_msg_desc& msg_desc);

    // Call the wrapper rx_release() when don't know which transport instance the message belongs to
    void rx_release(phy_mac_msg_desc& msg_desc);

    int32_t get_transport_id(int32_t phy_cell_id);

    // Get transport instance by L1 unique cell_id
    phy_mac_transport& get_transport(int32_t phy_cell_id);

    std::vector<phy_mac_transport*>& get_transports() {
        return transport_vec;
    }

    // Return a started transport. Return nullptr if not available.
    phy_mac_transport* get_loopback_transport()
    {
        for(phy_mac_transport* transport : transport_vec)
        {
            if(transport != nullptr && transport->is_started())
            {
                return transport;
            }
        }
        return nullptr;
    }

    uint32_t get_transport_num() {
        return transport_num;
    }

    void set_cell_configured(int32_t phy_cell_id) {
        configured_cells_mask |= 1ULL << phy_cell_id;
    }

    uint64_t get_configured_cells_mask() {
        return configured_cells_mask;
    }

    bool get_all_cells_configured();

private:
    bool cell_map_enabled = false;
    int32_t test_type = 0; // 0: normal test, 1: l2adapter standalone test
    uint32_t transport_num = 0;
    uint32_t total_cell_num = 0;
    uint64_t configured_cells_mask = 0;
    std::vector<int32_t> transport_id_vec; // Map<phy_cell_id, transport_id>
    std::vector<phy_mac_transport*> transport_vec;

    // Round poll index of the transport instances
    uint32_t round_poll_id = 0;

    // Handle for limitation that all cells need to be configured at initial
    int32_t wait_all_cell_config = 1;
};

} // namespace nv

#endif // !defined(NV_PHY_MAC_TRANSPORT_HPP_INCLUDED_)
