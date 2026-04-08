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

#include "nv_phy_mac_transport.hpp"
#include <strings.h>
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 5) // "L2A.TRANSPORT"

namespace
{
////////////////////////////////////////////////////////////////////////
// assign_char_array()
// Copy a string into a fixed length char array, throwing an exception
// if the length of the string is larger than the buffer.
template <int NUM_BUF_CHAR>
void assign_char_array(char (&buf)[NUM_BUF_CHAR], const std::string& str)
{
    if(str.size() >= NUM_BUF_CHAR)
    {
        throw std::runtime_error(std::string("nv_phy_mac_transport: YAML string length (") +
                                 std::to_string(str.size()) +
                                 std::string(") greater than buffer size (") +
                                 std::to_string(NUM_BUF_CHAR) +
                                 std::string(")"));
    }
    // str.size() is less than bufsz, so we are guaranteed a
    // terminating NULL in buf.
    strncpy(buf, str.c_str(), NUM_BUF_CHAR);
}

////////////////////////////////////////////////////////////////////////
// ipc_from_yaml()
// Create an nv_ipc_t instance from a YAML node with the appropriate
// fields
nv_ipc_t* ipc_from_yaml(nv_ipc_config_t& config, yaml::node node_config, nv_ipc_module_t module_type)
{
    nv_ipc_parse_yaml_node(&config, &node_config, module_type);
    return create_nv_ipc_interface(&config);
}

} // namespace

namespace nv
{

int transport_reset_callback(void* args)
{
    phy_mac_transport* transp = reinterpret_cast<phy_mac_transport*>(args);
    if(transp == nullptr || transp->reset_callback == nullptr)
    {
        NVLOGC_FMT(TAG, "{}: args == nullptr or reset_callback was not set", __func__);
        return -1;
    }
    transp->reset_started_cells_mask();
    transp->set_error_flag(false);
    return transp->reset_callback(transp, transp->_phy_module);
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport()
phy_mac_transport::phy_mac_transport(nv_ipc_config_t& config) :
    ipc_(create_nv_ipc_interface(&config), &cleanup_nv_ipc)
{
    NVLOGC_FMT(TAG, "{} construct 1", __func__);

    transport_id = 0;
    mapped = false;
    started_cells_mask = 0;
    config_ = config;

    if(!ipc_)
    {
        throw std::runtime_error("Error returned from create_nv_ipc_interface()");
    }
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport()
phy_mac_transport::phy_mac_transport(yaml::node node_config, nv_ipc_module_t module_type, uint32_t cell_num, int32_t transport_id, bool map_enable) :
    ipc_(ipc_from_yaml(config_, node_config, module_type), &cleanup_nv_ipc)
{
    if(!ipc_)
    {
        throw std::runtime_error("Error returned from create_nv_ipc_interface()");
    }

    // Check if all necessary data memory pools (cpu_data and cpu_large) are host pinned memory
    nv_ipc_check_host_pinned_memory(ipc_.get());

    nv_ipc_set_reset_callback(ipc_.get(), transport_reset_callback, this);

    this->transport_id = transport_id;
    this->cell_num = cell_num;
    mapped = map_enable;
    started_cells_mask = 0;
    if (node_config.has_key("phy_cells"))
    {
        yaml::node phy_cells = node_config["phy_cells"];
        for (int mac_cell_id = 0; mac_cell_id < cell_num; mac_cell_id++) {
            int phy_cell_id = phy_cells[mac_cell_id];
            if (std::find_if(phy_cell_map.begin(), phy_cell_map.end(), [phy_cell_id](auto&& p) { return p.second == phy_cell_id; }) != std::end(phy_cell_map)) {
                NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "duplicated cell_id configred in phy_cells list: phy_cells[{}]={}", mac_cell_id, phy_cell_id);
            }
            phy_cell_map[mac_cell_id] = phy_cell_id;
            mac_cell_map[phy_cell_id] = mac_cell_id;
        }
    }

    NVLOGC_FMT(TAG, "{}[{}] created. phy_cell_map.size={} mac_cell_map.size={} mapped={} cell_num={}", __func__, transport_id, phy_cell_map.size(), mac_cell_map.size(), mapped, cell_num);
    for (int mac_cell_id = 0; mac_cell_id < phy_cell_map.size(); mac_cell_id++) {
        NVLOGI_FMT(TAG, "{}[{}] cell_id map: mac {} <-> phy {}", __func__, transport_id, mac_cell_id, phy_cell_map[mac_cell_id]);
    }
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::rx_alloc()
int phy_mac_transport::rx_alloc(phy_mac_msg_desc& msg_desc, uint32_t options)
{
    int ret = ipc_->rx_allocate(ipc_.get(), &msg_desc, options);
    if(ret < 0)
    {
        if(error_flag == false)
        {
            NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: NVIPC memory pool is full: transport_id={}", __func__, transport_id);
        }
        error_flag = true;
    }
    return ret;
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::tx_alloc()
int phy_mac_transport::tx_alloc(phy_mac_msg_desc& msg_desc, uint32_t options)
{
    int ret = ipc_->tx_allocate(ipc_.get(), &msg_desc, options);
    if(ret < 0)
    {
        if (error_flag == false)
        {
            NVLOGE_FMT(TAG, AERIAL_NVIPC_API_EVENT, "{}: NVIPC memory pool is full: transport_id={}", __func__, transport_id);
        }
        error_flag = true;
    }
    return ret;
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::tx_copy()
void phy_mac_transport::tx_copy(void* dst, const void* src, size_t size, int32_t data_pool)
{
    if(data_pool == NV_IPC_MEMPOOL_GPU_DATA)
    {
        //NVLOGI_FMT(TAG,"[{}]: dst = 0x{:x} src = 0x{:x} size = {}",__func__,dst,(void *)src,size);
        if(0 != ipc_->gdr_memcpy_to_device(ipc_.get(), dst,src,size))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: tx_copy() error: dst={} src={} size={}", __FUNCTION__, dst,(void *)src,size);
            throw std::runtime_error("phy_mac_transport::tx_copy() failure");
        }
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: wrong data_pool used {}", __FUNCTION__,data_pool);
        throw std::runtime_error("phy_mac_transport::tx_copy() failure wrong data_pool used");
    }
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::tx_send()
void phy_mac_transport::tx_send(phy_mac_msg_desc& msg_desc)
{
    if (mapped) {
        // Convert L1 cell_id to L2 cell_id
        msg_desc.cell_id = get_mac_cell_id(msg_desc.cell_id);
        nv_ipc_set_handle_id(&msg_desc, msg_desc.cell_id);
    }

    if(0 != ipc_->tx_send_msg(ipc_.get(), &msg_desc))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: error: msg_id=0x{:02X}", __FUNCTION__, msg_desc.msg_id);
        throw std::runtime_error("phy_mac_transport::tx_send() failure");
    }
}

#ifdef ENABLE_L2_SLT_RSP
////////////////////////////////////////////////////////////////////////
// phy_mac_transport::tx_send_loopback()
void phy_mac_transport::tx_send_loopback(phy_mac_msg_desc& msg_desc)
{
    // if (ipc_->tx_send_loopback == nullptr)
    // {
    //     throw std::runtime_error("phy_mac_transport::tx_send_loopback() not supported");
    // }

    if(0 != nv_ipc_shm_send_loopback(ipc_.get(), &msg_desc))
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: error: msg_id=0x{:02X}", __FUNCTION__, msg_desc.msg_id);
        throw std::runtime_error("phy_mac_transport::tx_send() failure");
    }
}
////////////////////////////////////////////////////////////////////////
// phy_mac_transport::poll()
int phy_mac_transport::poll()
{
    if (config_.ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        return nv_ipc_shm_rx_poll(ipc_.get());
    } else {
        throw std::runtime_error("phy_mac_transport::poll() only supported in SHM IPC");
    }
}
#endif
////////////////////////////////////////////////////////////////////////
// phy_mac_transport::rx_release()
void phy_mac_transport::rx_release(phy_mac_msg_desc& msg_desc)
{
    if(0 != ipc_->rx_release(ipc_.get(), &msg_desc))
    {
        sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: error: transport_id={} SFN {}.{} cell_id={} msg_id=0x{:02X} msg_buf={} data_buf={} ts_send={}",
                __FUNCTION__, transport_id, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, msg_desc.msg_buf, msg_desc.data_buf, get_ts_send(msg_desc));
        throw std::runtime_error(fmt::format("{}: error: transport_id={} SFN {}.{} cell_id={} msg_id=0x{:02X} msg_buf={} data_buf={} ts_send={}",
                __FUNCTION__, transport_id, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, msg_desc.msg_buf, msg_desc.data_buf, get_ts_send(msg_desc)));
    }
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::tx_release()
void phy_mac_transport::tx_release(phy_mac_msg_desc& msg_desc)
{
    if(0 != ipc_->tx_release(ipc_.get(), &msg_desc))
    {
        sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg_desc);
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: error: transport_id={} SFN {}.{} cell_id={} msg_id=0x{:02X} msg_buf={} data_buf={} ts_send={}",
            __FUNCTION__, transport_id, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, msg_desc.msg_buf, msg_desc.data_buf, get_ts_send(msg_desc));
        throw std::runtime_error(fmt::format("{}: error: transport_id={} SFN {}.{} cell_id={} msg_id=0x{:02X} msg_buf={} data_buf={} ts_send={}",
                __FUNCTION__, transport_id, ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, msg_desc.msg_buf, msg_desc.data_buf, get_ts_send(msg_desc)));
    }
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::rx_recv()
int phy_mac_transport::rx_recv(phy_mac_msg_desc& msg_desc)
{
    int ret = ipc_->rx_recv_msg(ipc_.get(), &msg_desc);
    if (ret < 0) {
        return ret;
    }

    if(msg_desc.cell_id < 0 || msg_desc.cell_id >= cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: invalid cell_id={} cell_num={} transport_id={} msg_id=0x{:02X}",
                __func__, msg_desc.cell_id, cell_num, transport_id, msg_desc.msg_id);
        rx_release(msg_desc);
        return -1;
    }

    if (mapped) {
        msg_desc.cell_id = get_phy_cell_id(msg_desc.cell_id);
        nv_ipc_set_handle_id(&msg_desc, msg_desc.cell_id);
    }

    return ret;
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::cleanup_nv_ipc()
void phy_mac_transport::cleanup_nv_ipc(nv_ipc_t* ipc)
{
    NVLOGI_FMT(TAG, "phy_mac_transport::cleanup_nv_ipc");
    if(ipc)
    {
        ipc->ipc_destroy(ipc);
    }
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::get_fd()
int phy_mac_transport::get_fd()
{
    return ipc_->get_fd(ipc_.get());
}

////////////////////////////////////////////////////////////////////////
// phy_mac_transport::notify()
int phy_mac_transport::notify(int value)
{
    return ipc_->notify(ipc_.get(), value);
}

////////////////////////////////////////////////////////////////////////
int phy_mac_transport::get_value()
{
    return ipc_->get_value(ipc_.get());
}

int64_t phy_mac_transport::get_ts_send(nv_ipc_msg_t& msg_desc)
{
    if (msg_desc.msg_buf == nullptr)
    {
        throw std::runtime_error("IPC message buffer was not allocated");
    }

    if (config_.ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        // Sending time stamp only supported by SHM IPC
        return nv_ipc_get_ts_send(ipc_.get(), &msg_desc);
    }
    else
    {
        struct timespec ts;
        nvlog_gettime_rt(&ts);
        return ts.tv_sec * 1000000000LL + ts.tv_nsec;
    }
}

// Copy memory from msg_desc.data_buf to CPU dst_buf. Return 0 on success; return < 0 on failure.
int phy_mac_transport::copy_from_data_buf(nv_ipc_msg_t& msg_desc, uint32_t src_offset, void* dst_buf, size_t size)
{
    if(size <= 0)
    {
        // No buffer to copy, skip
        return 0;
    }

    uint8_t* src_buf = reinterpret_cast<uint8_t*>(msg_desc.data_buf) + src_offset;
    if(msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_DATA)
    {
        memcpy(dst_buf, src_buf, size);
    }
    else if(msg_desc.data_pool == NV_IPC_MEMPOOL_GPU_DATA)
    {
        if(0 != ipc_->gdr_memcpy_to_host(ipc_.get(), dst_buf, src_buf, size))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: error: data_pool={} dst={} src={} size={}", __func__, msg_desc.data_pool, dst_buf, (void*)src_buf, size);
            return -1;
        }
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: unknown data_pool {}", __func__, msg_desc.data_pool);
    }

    return 0;
}

// Copy memory from CPU src_buf to msg_desc.data_buf. Return 0 on success; return < 0 on failure.
int phy_mac_transport::copy_to_data_buf(nv_ipc_msg_t& msg_desc, uint32_t dst_offset, void* src_buf, size_t size)
{
    if(size <= 0)
    {
        // No buffer to copy, skip
        return -1;
    }

    uint8_t* dst_buf = reinterpret_cast<uint8_t*>(msg_desc.data_buf) + dst_offset;
    if(msg_desc.data_pool == NV_IPC_MEMPOOL_CPU_DATA)
    {
        memcpy(dst_buf, src_buf, size);
    }
    else if(msg_desc.data_pool == NV_IPC_MEMPOOL_GPU_DATA)
    {
        if(0 != ipc_->gdr_memcpy_to_device(ipc_.get(), dst_buf, src_buf, size))
        {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: error: data_pool={} dst={} src={} size={}", __func__, msg_desc.data_pool, dst_buf, (void*)src_buf, size);
            return -1;
        }
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}: unknown data_pool {}", __func__, msg_desc.data_pool);
        return -1;
    }

    return 0;
}

phy_mac_transport_wrapper::phy_mac_transport_wrapper(yaml::node node_config, nv_ipc_module_t module_type, uint32_t total_cell_num) {
    this->total_cell_num = total_cell_num;

    if (node_config.has_key("test_type")) {
        test_type = node_config["test_type"].as<int32_t>();
    }

    if (node_config.has_key("nvipc_config_file")) {
        std::string file_name = node_config["nvipc_config_file"].as<std::string>();
        if (file_name.length() == 0 || file_name.compare("null") == 0) {
            init(node_config, module_type);
            return;
        }

        // cuphycontroller app relative path: build/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf
        std::string relative_path = std::string("../../../../cuPHY-CP/cuphycontroller/config/").append(file_name.c_str());
        if (node_config.has_key("nvipc_config_file_path")) {
            relative_path = node_config["nvipc_config_file_path"].as<std::string>();
            relative_path.append(file_name);
        }

        char        yaml_file[1024];
        nv_get_absolute_path(yaml_file, relative_path.c_str());
        NVLOGC_FMT(TAG, "loaded nvipc config from {}", yaml_file);
        yaml::file_parser nvipc_file(yaml_file);
        yaml::document nvipc_doc = nvipc_file.next_document();
        yaml::node nvipc_root = nvipc_doc.root();
        init(nvipc_root, module_type);
    } else {
        init(node_config, module_type);
    }
}

void phy_mac_transport_wrapper::init(yaml::node& node_config, nv_ipc_module_t module_type) {
    configured_cells_mask = 0;
    wait_all_cell_config = 1;
    round_poll_id = 0;

    if (node_config.has_key("wait_all_cell_config")) {
        wait_all_cell_config = node_config["wait_all_cell_config"].as<int32_t>();
    }

    yaml::node transport_node = node_config["transport"];
    if(transport_node.type() == YAML_SEQUENCE_NODE) {
        int transport_cell_sum = 0;
        cell_map_enabled = true;
        transport_num = transport_node.length();
        for(size_t i = 0; i < transport_num; i++) {
            if (transport_node[i]["transport_id"].as<int32_t>() != i) {
                NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "Please config transport_id in order and start from 0");
            }
            uint32_t cell_num = transport_node[i]["phy_cells"].length();
            phy_mac_transport* ptransport = new phy_mac_transport(transport_node[i], module_type, cell_num, i, true);
            transport_vec.push_back(ptransport);
            transport_cell_sum += cell_num;
        }

        if (test_type != 0) {
            // There's no cell_group_num configuration in l2adapter standalone test, total_cell_num is the instances list size, should have transport_cell_sum <= total_cell_num
            if (transport_cell_sum > total_cell_num) {
                NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "Check YAML config for L2SA test: total_cell_num={} for all L2 instances != transport_cell_sum={}", total_cell_num, transport_cell_sum);
            }
            // Update total_cell_num to transport_cell_sum
            total_cell_num = transport_cell_sum;
        }

        if (transport_cell_sum != total_cell_num) {
            NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "Check YAML config: total_cell_num={} for all L2 instances != transport_cell_sum={}", total_cell_num, transport_cell_sum);
        }
        transport_id_vec.resize(total_cell_num);
        for (int transport_id = 0; transport_id < transport_vec.size(); transport_id++)
        {
            std::unordered_map<int32_t, int32_t>& cell_map = transport_vec[transport_id]->get_phy_cell_map();
            NVLOGC_FMT(TAG, "wrapper: transport_id {} size={}", transport_id, cell_map.size());
            for (int mac_cell_id = 0; mac_cell_id < cell_map.size(); mac_cell_id++)
            {
                int phy_cell_id = cell_map[mac_cell_id];

                // Validate cell_id config: phy_cell_id < total_cell_num
                if (phy_cell_id >= total_cell_num) {
                    NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "cell_id exceeds range: transport_id={} phy_cells[{}]={} total_cell_num={}", transport_id, mac_cell_id, phy_cell_id, total_cell_num);
                }

                // Validate cell_id config: non duplicate cell_id in phy_cell_maps
                for (int prev_transp_id = 0; prev_transp_id < transport_id; prev_transp_id++) {
                    std::unordered_map<int32_t, int32_t>& prev_phy_cell_map = transport_vec[prev_transp_id]->get_phy_cell_map();
                    if (std::find_if(prev_phy_cell_map.begin(), prev_phy_cell_map.end(), [phy_cell_id](auto&& p) { return p.second == phy_cell_id; }) != std::end(prev_phy_cell_map)) {
                        NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "duplicated cell_id: transport[{}].phy_cells[{}]={} already exists in transport[{}]", transport_id, mac_cell_id, phy_cell_id, prev_transp_id);
                    }
                }

                transport_id_vec[phy_cell_id] = transport_id;
                NVLOGC_FMT(TAG, "wrapper map: phy {} <-> transport {} - mac {}", phy_cell_id, transport_id, mac_cell_id);
            }
        }
    } else {
        cell_map_enabled = false;
        transport_num = 1;
        phy_mac_transport* ptransport = new phy_mac_transport(transport_node, module_type, total_cell_num);
        transport_vec.push_back(ptransport);
    }

    NVLOGC_FMT(TAG, "{}: loaded {} nvipc instance, total_cell_num={}", __func__, transport_num, total_cell_num);
}

phy_mac_transport_wrapper::~phy_mac_transport_wrapper() {

}

int32_t phy_mac_transport_wrapper::get_transport_id(int32_t phy_cell_id) {
    if (transport_num == 1) {
        return 0;
    }

    if (phy_cell_id >= 0 && phy_cell_id < transport_id_vec.size()) {
        return transport_id_vec[phy_cell_id];
    } else {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} wrapper: invalid phy_cell_id {}. transport_id_vec.size={} transport_num={}",
                __FUNCTION__, phy_cell_id, transport_id_vec.size(), transport_num);
        return -1;
    }
}

phy_mac_transport& phy_mac_transport_wrapper::get_transport(int phy_cell_id)
{
    int32_t transport_id = get_transport_id(phy_cell_id);
    if (transport_id >=0 && transport_id < transport_vec.size()) {
        return *transport_vec[transport_id];
    } else {
        NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{} wrapper: invalid transport_id {}. phy_cell_id={} transport_num={} transport_vec.size={}",
                __FUNCTION__, transport_id, phy_cell_id, transport_num, transport_vec.size());
        throw std::runtime_error("phy_mac_transport_wrapper: invalid transport_id");
    }
}

bool phy_mac_transport_wrapper::get_all_cells_configured() {
    if (wait_all_cell_config == 0) {
        // Skip if not configured to requiring all cells configured
        return true;
    }

    uint64_t expected = (1ULL << total_cell_num) - 1;
    bool ret = (configured_cells_mask == expected);
    NVLOGC_FMT(TAG, "wrapper: configured_cells_mask=0x{:X} expected_mask=0x{:X} all_cells_configured={} - {}",
            configured_cells_mask, expected, ret, ret ? "all cells configured" : "not all cells configured yet");
    return ret;
}

int phy_mac_transport_wrapper::poll() {
    for (uint32_t id = 0; id < transport_num; id ++) {
        phy_mac_transport* ptransport = transport_vec[id];
        if(ptransport->poll()) return 1;
    }
    return 0;
}

int phy_mac_transport_wrapper::rx_recv(phy_mac_msg_desc& msg_desc) {
    int ret;
    for (uint32_t id = 0; id < transport_num; id ++) {
        phy_mac_transport* ptransport = transport_vec[(round_poll_id + id) % transport_num];
        if ((ret = ptransport->rx_recv(msg_desc)) >= 0) {
            // Next time poll the next transport instance
            round_poll_id = (round_poll_id + id + 1) % transport_num;
            return ret;
        }
    }

    // No incoming message in all transports
    return -1;
}

void phy_mac_transport_wrapper::rx_release(phy_mac_msg_desc& msg_desc) {
    if (msg_desc.msg_buf != nullptr) {
        int32_t transport_id = get_transport_id(msg_desc.cell_id);
        if (transport_id >= 0 && transport_id < transport_num) {
            transport_vec[transport_id]->rx_release(msg_desc);
        } else {
            NVLOGE_FMT(TAG, AERIAL_L2ADAPTER_EVENT, "{}[{}] wrapper: invalid transport_id. cell_id={} msg_id=0x{:02X} msg_buf={} data_buf={}",
                    __FUNCTION__, transport_id, msg_desc.cell_id, msg_desc.msg_id, msg_desc.msg_buf, msg_desc.data_buf);
        }
    }
}

} // namespace nv
