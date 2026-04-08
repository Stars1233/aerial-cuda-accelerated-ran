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

#include "rivermax.hpp"
#include "utils.hpp"
#include <arpa/inet.h>
#include <fstream>
#include <sys/types.h>
#include <ifaddrs.h>

#define TAG "FH.RMAX"

#define IN_MULTICAST_N(a) (((long int)(a) & htonl(0xf0000000)) == htonl(0xe0000000))

namespace aerial_fh
{

static std::string get_dev_mac(char* if_name)
{
    std::stringstream mac_addr_path;

    mac_addr_path << "/sys/class/net/" << if_name << "/address";

    std::ifstream is (mac_addr_path.str(), std::ios::in);

    if (is.good()) {

        std::string mac_str;
        is >> mac_str;
        for(int i=0; i<RTE_ETHER_ADDR_LEN-1; i++) {
            mac_str.erase(std::find(mac_str.begin(),mac_str.end(),':'));
        }

        return mac_str;
    }
    return 0;
}

static uint64_t convert_mac_addr(std::string mac_str)
{
    // std::cout << "mac " << mac_str << std::endl;;
    // for(int i = 0; i < RTE_ETHER_ADDR_LEN-1; i++) {
    //     mac_str.erase(std::find(mac_str.begin(),mac_str.end(),':'));
    // }
    uint64_t d_mac = strtoul(mac_str.c_str(), nullptr, 16);

    return d_mac;
}

static std::string get_inet_ntoa(const in_addr &in)
{
    char buff[INET_ADDRSTRLEN];
    const char* result = inet_ntop(AF_INET, (void *)&in.s_addr, buff, sizeof(buff));
    if (result == nullptr) {
        THROW_FH(errno, StringBuilder() << "inet_ntop failed for address conversion");
    }
    return std::string(result);
}

static rmax_status_t get_ip_from_ifface(std::string mac_address, struct sockaddr_in* sockh)
{
    rmax_status_t ret_val = RMAX_ERR_NO_DEVICE;
    int num_mapped_devices = 0;
    ifaddrs *ifa = nullptr;
    ifaddrs *ifa_start = nullptr;
    std::string mac_address_;

    if (getifaddrs(&ifa)) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "getifaddrs failed errno {}", errno);
        return RMAX_ERR_UNKNOWN_ISSUE;
    }
    ifa_start = ifa;

    for (; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET &&
            !IN_MULTICAST_N(((sockaddr_in*)ifa->ifa_addr)->sin_addr.s_addr)) {
            std::string hw_dev_name;
            const char* hw_ifname = hw_dev_name.size() ? hw_dev_name.c_str() : ifa->ifa_name;
            std::string ip_address = get_inet_ntoa(((sockaddr_in *)ifa->ifa_addr)->sin_addr);
            mac_address_ = get_dev_mac(ifa->ifa_name);

            NVLOGI_FMT(TAG, "examining IP: {} MAC: {} device: {} hw_ifname {}",
                ip_address.c_str(), mac_address_.c_str(), ifa->ifa_name, hw_ifname);

            if(mac_address.compare(mac_address_) == 0) {

                sockh->sin_family = AF_INET;
                int rc = inet_pton(AF_INET, ip_address.c_str(), &(sockh->sin_addr));
                NVLOGC_FMT(TAG, "inet_pton rc {} ip {}", rc, ip_address.c_str());
                if (rc != 1) {
                    NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "Failed to parse local network address {}", rc);
                    ret_val = RMAX_ERR_LAST;
                }
                else
                    num_mapped_devices++;
                break;
            }
        }
    }

    if (ifa_start)
        freeifaddrs(ifa_start);

    if (num_mapped_devices)
        ret_val = RMAX_OK; // if at least single device is mapped return success

    return ret_val;
}

RivermaxPrx::RivermaxPrx(Fronthaul* fhi) :
    fhi_{fhi}
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(-1, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    const char *rmax_version = rmax_get_version_string();
    static std::string app_version =
        std::to_string(RMAX_API_MAJOR) + std::string(".") +
        std::to_string(RMAX_API_MINOR) + std::string(".") +
        std::to_string(RMAX_RELEASE_VERSION) + std::string(".") +
        std::to_string(RMAX_BUILD);

    NVLOGI_FMT(TAG, "RivermaxPrx library version: {}", rmax_version);
    NVLOGI_FMT(TAG, "Application version: {}", app_version.c_str());

    // Stream default attrs
    rx_type_ = RMAX_APP_PROTOCOL_PACKET_PAYLOAD_L2;
    attr_flags_ = RMAX_IN_BUFFER_ATTER_STREAM_ECPRI_SYMBOL_PRB_PLACEMENT_ORDER;
    timestamp_format_ = RMAX_PACKET_TIMESTAMP_SYNCED;
    // use_checksum_header_ = false;
    header_size_ = ECPRI_HEADER_SIZE;

    // Configure the buffer attributes.
    memset(&m_buffer, 0, sizeof(m_buffer));
    m_buffer.attr_flags = attr_flags_;
    // m_buffer.num_of_elements = buffer_elements;

    // Configure the payload buffer.
    memset(&m_data, 0, sizeof(m_data));
    // m_data.min_size = payload_size;
    // m_data.max_size = payload_size;
    m_buffer.data = &m_data;

    // Configure the header buffer.
    memset(&m_header, 0, sizeof(m_header));
    m_header.min_size = header_size_;
    m_header.max_size = header_size_;
    m_header.ptr = NULL;
    m_buffer.hdr = &m_header;

#endif
}

int RivermaxPrx::init_nic(std::string mac_address, socket_handle* sockh)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    rmax_init_config init_config_;
    rmax_status_t status;
    struct sockaddr_in * local_socket_ = (struct sockaddr_in *) calloc(1, sizeof(struct sockaddr_in));

    NVLOGI_FMT(TAG, "Init Rivermax NIC with MAC address '{}'", mac_address.c_str());

    memset(local_socket_, 0, sizeof(struct sockaddr_in));

    status = get_ip_from_ifface(mac_address, local_socket_);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to get IP for device {}", mac_address.c_str());
        return -1;
    }

    // Initialize RivermaxPrx library.
    memset(&init_config_, 0, sizeof(init_config_));
    init_config_.flags |= RIVERMAX_ENABLE_GLOBAL_PROTOCOL_CONFIGURATION_ECPRI | RIVERMAX_HANDLE_SIGNAL;
    init_config_.global_protocol_device.ip_address = local_socket_->sin_addr;

    status = rmax_init(&init_config_);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed initializing RivermaxPrx {}", status);
        return -1;
    }

    *sockh = static_cast<socket_handle>(local_socket_);

    NVLOGC_FMT(TAG, "RivermaxPrx sockh_ {} *sockh {} local_socket_ {}", sockh, *sockh, local_socket_);
    char buffer[INET_ADDRSTRLEN];
    inet_ntop( AF_INET, &(local_socket_->sin_addr), buffer, sizeof( buffer ));
    NVLOGC_FMT(TAG,  "RivermaxPrx address:{}", buffer );
#endif

    return 0;
}

int RivermaxPrx::query_buffer_size(int buffer_elements, int payload_unit_size, socket_handle* sockh,
                                                                size_t * payload_len, size_t * header_len)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else
    rmax_status_t status;
    struct sockaddr_in * local_socket_ = static_cast<struct sockaddr_in*>(*sockh);

    m_buffer.num_of_elements = buffer_elements;
    m_data.min_size = payload_unit_size;
    m_data.max_size = payload_unit_size;

    status = rmax_in_query_buffer_size(rx_type_, local_socket_, &m_buffer, payload_len, header_len);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed calling rmax_in_query_buffer_size; error: {}", status);
        return -1;
    }

#endif

    return 0;
}

int RivermaxPrx::create_stream(void* addr, socket_handle* sockh, rmax_stream_id* stream_id)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    rmax_status_t status;
    struct sockaddr_in * local_socket_ = static_cast<struct sockaddr_in*>(*sockh);

    m_buffer.data->ptr = addr;

    status = rmax_in_create_stream(rx_type_, local_socket_, &m_buffer,
                                        timestamp_format_,
                                        RMAX_IN_CREATE_STREAM_INFO_PER_PACKET,
                                        stream_id);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed calling rmax_in_create_stream; error: {}", status);
        return -1;
    }

        NVLOGI_FMT(TAG, "Created stream for {} packets.    Payload stride: {}    Header stride: {}",
            m_buffer.num_of_elements, m_buffer.data->stride_size,
            (m_buffer.hdr ? m_buffer.hdr->stride_size : 0));

#endif

    return 0;
}


int RivermaxPrx::destroy_stream(rmax_stream_id stream_id)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    rmax_status_t status = rmax_in_destroy_stream(stream_id);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Failed to destroy stream; error: {}", status);
        return -1;
    }

#endif

    return 0;
}

int RivermaxPrx::attach_flow(int flow_id, std::string destination_mac, std::string source_mac, uint64_t vlan_id,
                            uint16_t pc_id, int idx_slot, int sectionMask, int sectionId, rmax_stream_id stream_id,
                            socket_handle* sockh, rmax_in_flow_attr_ex* in_flow)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    rmax_status_t status;
    struct sockaddr_in * local_socket_ = static_cast<struct sockaddr_in*>(*sockh);

    memset(in_flow, 0, sizeof(in_flow));
    in_flow->flow_id = flow_id;
    in_flow->app_layer_type = RMAX_IN_FLOW_APP_LAYER_ECPRI;

    in_flow->l2.mask.destination_mac = 0xFFFFFFFFFFFF;
    in_flow->l2.mask.source_mac = 0xFFFFFFFFFFFF;
    in_flow->l2.mask.vlan_id = 0xFFF;
    in_flow->l2.value.destination_mac = convert_mac_addr(destination_mac);
    in_flow->l2.value.source_mac = convert_mac_addr(source_mac);
    in_flow->l2.value.vlan_id = vlan_id;

    in_flow->app_layer.ecpri.mask.pc_id = 0xFFFF;
    in_flow->app_layer.ecpri.mask.slot_id = 0x1;
    in_flow->app_layer.ecpri.mask.subframe_id = 0x1;
    in_flow->app_layer.ecpri.mask.section_id = 0xFFF; //sectionMask;
    in_flow->app_layer.ecpri.value.pc_id = (uint16_t)pc_id;
    if (idx_slot == 0) {
        in_flow->app_layer.ecpri.value.slot_id = 0;
        in_flow->app_layer.ecpri.value.subframe_id = 0;
    }
    if (idx_slot == 1) {
        in_flow->app_layer.ecpri.value.slot_id = 1;
        in_flow->app_layer.ecpri.value.subframe_id = 0;
    }
    if (idx_slot == 2) {
        in_flow->app_layer.ecpri.value.slot_id = 0;
        in_flow->app_layer.ecpri.value.subframe_id = 1;
    }
    if (idx_slot == 3) {
        in_flow->app_layer.ecpri.value.slot_id = 1;
        in_flow->app_layer.ecpri.value.subframe_id = 1;
    }

    in_flow->app_layer.ecpri.value.section_id = sectionId;

    status = rmax_in_attach_flow_ex(stream_id, in_flow);
    if (status != RMAX_OK)
        THROW_FH(ENOMEM, StringBuilder() << "Failed to attach flow; error:" << status);

    NVLOGD_FMT(TAG, "Created flow for Stream {} eAxC {} flow_id {}", stream_id, pc_id, flow_id);

#endif

    return 0;
}

int RivermaxPrx::detach_flow(rmax_stream_id stream_id, rmax_in_flow_attr_ex* in_flow)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    rmax_status_t status = rmax_in_detach_flow_ex(stream_id, in_flow);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT,  "Failed to detach flow; error: {}", status);
        return -1;
    }

#endif

    return 0;
}

int RivermaxPrx::get_next_chunk(rmax_stream_id stream_id, int timeout, uint64_t* rx_bytes)
{
#ifndef ENABLE_RIVERMAX
    THROW_FH(ENOTSUP, StringBuilder() << "Aerial FH not build with Rivermax");
#else

    const int min_packets = 0;
    const int max_packets = ECPRI_NUMBER_OF_PRB_IN_SYMBOL * ECPRI_NUMBER_OF_SYMBOLS_IN_SLOT;
    const int flags = 0;
    uint64_t rx_bytes_ = 0;
    struct rmax_in_completion rx_completion;

    rmax_status_t status = rmax_in_get_next_chunk(stream_id, min_packets, max_packets, timeout, flags, &rx_completion);
    if (status != RMAX_OK) {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT,  "Failed to get next chunk in; error: {}", status);
        return -1;
    }

    for (int idx = 0; idx < rx_completion.chunk_size; idx++) {
        rx_bytes_ += rx_completion.packet_info_arr->data_size;
        // FIXME: should we report min and max packets' timestamps?
    }

    (*rx_bytes) = rx_bytes_;

#endif

    return 0;
}


RivermaxPrx::~RivermaxPrx()
{
#ifdef ENABLE_RIVERMAX

    NVLOGD_FMT(TAG, "Destroying RivermaxPrx context");
    rmax_cleanup();

#endif
}

Fronthaul* RivermaxPrx::get_fronthaul() const
{
    return fhi_;
}

} // namespace aerial_fh
