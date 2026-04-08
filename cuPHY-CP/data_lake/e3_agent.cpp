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

#include "e3_agent.hpp"
#include "data_lake.hpp"
#include "fmt/format.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

// Constructor
E3Agent::E3Agent(
    DataLake* dl,
    const uint16_t repPort,
    const uint16_t pubPort,
    const uint16_t subPort,
    const int rowsFh,
    const int rowsPusch,
    const int rowsHest,
    const uint32_t fhSamples,
    const uint32_t maxHestSamples
) :
    dataLake(dl),
    e3RepPort(repPort),
    e3PubPort(pubPort),
    e3SubPort(subPort),
    numRowsToInsertFh(rowsFh),
    numRowsToInsertPusch(rowsPusch),
    numRowsToInsertHest(rowsHest),
    numFhSamples(fhSamples),
    maxHestSamplesPerRow(maxHestSamples),
    zmq_context(1),
    e3_rep_socket(zmq_context, ZMQ_REP),
    e3_pub_socket(zmq_context, ZMQ_PUB),
    e3_sub_socket(zmq_context, ZMQ_SUB)
{
}

// Destructor
E3Agent::~E3Agent()
{
    shutdown();

    if (shm_data_ptr != nullptr && shm_data_ptr != MAP_FAILED) {
        munmap(shm_data_ptr, shm_data_size);
    }
    if (shm_data_fd != -1) {
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
    }
}

// Initialize E3 agent - bind sockets and start threads
bool E3Agent::init()
{
    if (e3_running.load()) {
        return true;
    }

    NVLOGC_FMT(TAG_E3, "Initializing E3 Agent...");

    try {
        e3_rep_socket.bind("tcp://*:" + std::to_string(e3RepPort));
        e3_pub_socket.bind("tcp://*:" + std::to_string(e3PubPort));
        e3_sub_socket.bind("tcp://*:" + std::to_string(e3SubPort));

        e3_rep_socket.set(zmq::sockopt::tcp_keepalive, 1);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_idle, 5);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_intvl, 2);
        e3_rep_socket.set(zmq::sockopt::tcp_keepalive_cnt, 3);

        e3_sub_socket.set(zmq::sockopt::subscribe, "");
        e3_sub_socket.set(zmq::sockopt::linger, 1000);  // 1 second linger to allow graceful shutdown

        NVLOGC_FMT(TAG_E3, "E3 sockets initialized - REP: {}, PUB: {}, SUB: {}", e3RepPort, e3PubPort, e3SubPort);
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Failed to initialize E3 sockets: {}", e.what());
        return false;
    }

    e3_running = true;
    e3_data_thread = std::thread(&E3Agent::dataServerThread, this);

    e3_reaper_running = true;
    e3_reaper_thread = std::thread(&E3Agent::reaperThread, this);

    e3_sub_running = true;
    e3_sub_thread = std::thread(&E3Agent::managerSubscriptionThread, this);

    {
        std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
        dataLake->e3_buffer_info = {};
    }

    NVLOGC_FMT(TAG_E3, "E3 Agent initialized successfully.");
    return true;
}

// Shutdown E3 agent - stop threads and close sockets
void E3Agent::shutdown()
{
    if (e3_running) {
        e3_running = false;
        if (e3_data_thread.joinable()) {
            e3_data_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 data server thread shutdown");
    }

    if (e3_sub_running) {
        e3_sub_running = false;
        if (e3_sub_thread.joinable()) {
            e3_sub_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 subscription thread shutdown");
    }

    if (e3_reaper_running) {
        e3_reaper_running = false;
        if (e3_reaper_thread.joinable()) {
            e3_reaper_thread.join();
        }
        NVLOGC_FMT(TAG_E3, "E3 reaper thread shutdown");
    }

    try {
        e3_pub_socket.close();
        e3_rep_socket.close();
        e3_sub_socket.close();
        NVLOGC_FMT(TAG_E3, "E3 sockets closed successfully");
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Error closing E3 sockets: {}", e.what());
    }
}

// Create shared memory buffers for data exchange
bool E3Agent::createSharedMemoryBuffers(
    fhInfo_t** pFh,
    fhInfo_t** pInsertFh,
    puschInfo_t** p,
    puschInfo_t** pInsertPusch,
    hestInfo_t** pHest,
    hestInfo_t** pInsertHest
)
{
    NVLOGC_FMT(TAG_E3, "Creating shared memory buffers for E3");

    const size_t fh_buffer_size = numFhSamples * numRowsToInsertFh * sizeof(fhDataType);
    const size_t pusch_buffer_size = 80000 * numRowsToInsertPusch;
    const size_t hest_buffer_size = maxHestSamplesPerRow * numRowsToInsertHest * sizeof(hestDataType);

    const size_t total_size = sizeof(SharedMemoryHeader) +
                             (2 * fh_buffer_size) +
                             (2 * pusch_buffer_size) +
                             (2 * hest_buffer_size);

    shm_data_fd = shm_open(E3_SHARED_MEMORY_KEY.data(), O_CREAT | O_RDWR, 0666);
    if (shm_data_fd == -1) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to create shared memory, errno: {}", errno);
        return false;
    }

    if (ftruncate(shm_data_fd, total_size) == -1) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to set shared memory size, errno: {}", errno);
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
        return false;
    }

    shm_data_ptr = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_data_fd, 0);
    if (shm_data_ptr == MAP_FAILED) {
        NVLOGE_FMT(TAG_E3, AERIAL_SYSTEM_API_EVENT, "Failed to map shared memory, errno: {}", errno);
        close(shm_data_fd);
        shm_unlink(E3_SHARED_MEMORY_KEY.data());
        return false;
    }

    shm_data_size = total_size;

    SharedMemoryHeader* header = static_cast<SharedMemoryHeader*>(shm_data_ptr);
    memset(header, 0, sizeof(SharedMemoryHeader));
    header->version = 1;
    header->fh_buffer_size = fh_buffer_size;
    header->pusch_buffer_size = pusch_buffer_size;
    header->hest_buffer_size = hest_buffer_size;
    header->num_fh_samples = numFhSamples;
    header->num_fh_rows = numRowsToInsertFh;
    header->num_pusch_rows = numRowsToInsertPusch;
    header->num_hest_rows = numRowsToInsertHest;
    header->max_hest_samples_per_row = maxHestSamplesPerRow;

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(header + 1);

    (*pFh)->pDataAlloc = reinterpret_cast<fhDataType*>(base_ptr);
    (*pInsertFh)->pDataAlloc = reinterpret_cast<fhDataType*>(base_ptr + fh_buffer_size);

    (*p)->pDataAlloc = reinterpret_cast<uint8_t*>(base_ptr + 2 * fh_buffer_size);
    (*pInsertPusch)->pDataAlloc = reinterpret_cast<uint8_t*>(base_ptr + 2 * fh_buffer_size + pusch_buffer_size);

    (*pHest)->pDataAlloc = reinterpret_cast<hestDataType*>(base_ptr + 2 * fh_buffer_size + 2 * pusch_buffer_size);
    (*pInsertHest)->pDataAlloc = reinterpret_cast<hestDataType*>(base_ptr + 2 * fh_buffer_size + 2 * pusch_buffer_size + hest_buffer_size);

    NVLOGC_FMT(TAG_E3, "Shared memory buffers created successfully");
    NVLOGC_FMT(TAG_E3, "  Total size: {} bytes", total_size);
    NVLOGC_FMT(TAG_E3, "  FH buffers: {} bytes each", fh_buffer_size);
    NVLOGC_FMT(TAG_E3, "  PUSCH buffers: {} bytes each", pusch_buffer_size);
    NVLOGC_FMT(TAG_E3, "  H estimates buffers: {} bytes each", hest_buffer_size);

    return true;
}


// Notify subscribers that data is ready
void E3Agent::notifyDataReady()
{
    if (!e3_running) {
        return;
    }

    NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: e3NotifyDataReady entry at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());

    E3BufferInfo buffer_info;
    {
        std::lock_guard<std::mutex> lock(dataLake->e3_buffer_mutex);
        buffer_info = dataLake->e3_buffer_info;
    }

    std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
    for (auto& [sub_id, sub] : e3_subscriptions) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::microseconds>(now - sub.last_update).count() >= sub.periodicity_us) {
            json notif_json;
            notif_json["type"] = "indicationMessage";
            notif_json["id"] = generateMessageId();
            notif_json["dAppIdentifier"] = sub.dapp_id;
            notif_json["ranFunctionIdentifier"] = sub.ran_function_id;
            notif_json["subscriptionId"] = sub.subscription_id;

            json protocolData;
            uint64_t remaining_streams = static_cast<uint64_t>(sub.stream_bitfield);
            while (remaining_streams != 0) {
                // Find the lowest set bit
                const uint64_t lowest_bit = remaining_streams & (~remaining_streams + 1);
                const e3::StreamType stream_flag = static_cast<e3::StreamType>(lowest_bit);

                switch (stream_flag) {
                    case e3::StreamType::IQ_SAMPLES: {
                        json iq_shm_data;
                        iq_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                        iq_shm_data["fh_buffer_index"] = static_cast<int>(buffer_info.current_fh_buffer);
                        iq_shm_data["fh_write_index"] = buffer_info.fh_write_index;
                        protocolData["iq_samples"] = iq_shm_data;
                        break;
                    }
                    case e3::StreamType::PDU_DATA: {
                        json pdu_shm_data;
                        pdu_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                        pdu_shm_data["pusch_buffer_index"] = static_cast<int>(buffer_info.current_pusch_buffer);
                        pdu_shm_data["pusch_write_index"] = buffer_info.pusch_write_index;
                        protocolData["pdu_data"] = pdu_shm_data;
                        break;
                    }
                    case e3::StreamType::H_ESTIMATES: {
                        json hest_shm_data;
                        hest_shm_data["shm_name"] = E3_SHARED_MEMORY_KEY;
                        hest_shm_data["hest_buffer_index"] = static_cast<int>(buffer_info.current_hest_buffer);
                        hest_shm_data["hest_write_index"] = buffer_info.hest_write_index;
                        hest_shm_data["hest_data_size"] = buffer_info.hest_data_size;
                        protocolData["h_estimates"] = hest_shm_data;
                        break;
                    }
                    case e3::StreamType::TIMESTAMP: {
                        protocolData["timestamp"] = buffer_info.timestamp_ns;
                        break;
                    }
                    case e3::StreamType::SFN: {
                        protocolData["sfn"] = buffer_info.sfn;
                        break;
                    }
                    case e3::StreamType::SLOT: {
                        protocolData["slot"] = buffer_info.slot;
                        break;
                    }
                    case e3::StreamType::CELL_ID: {
                        protocolData["cell_id"] = buffer_info.cell_id;
                        break;
                    }
                    case e3::StreamType::N_RX_ANT: {
                        protocolData["n_rx_ant"] = buffer_info.n_rx_ant;
                        break;
                    }
                    case e3::StreamType::N_RX_ANT_SRS: {
                        protocolData["n_rx_ant_srs"] = buffer_info.n_rx_ant_srs;
                        break;
                    }
                    case e3::StreamType::N_CELLS: {
                        protocolData["n_cells"] = buffer_info.n_cells;
                        break;
                    }
                    case e3::StreamType::N_BS_ANTS: {
                        protocolData["n_bs_ants"] = buffer_info.n_bs_ants;
                        break;
                    }
                    case e3::StreamType::N_LAYERS: {
                        protocolData["n_layers"] = buffer_info.n_layers;
                        break;
                    }
                    case e3::StreamType::N_SUBCARRIERS: {
                        protocolData["n_subcarriers"] = buffer_info.n_subcarriers;
                        break;
                    }
                    case e3::StreamType::N_DMRS_ESTIMATES: {
                        protocolData["n_dmrs_estimates"] = buffer_info.n_dmrs_estimates;
                        break;
                    }
                    case e3::StreamType::DMRS_SYMB_POS: {
                        protocolData["dmrs_symb_pos"] = buffer_info.dmrs_symb_pos;
                        break;
                    }
                    case e3::StreamType::TB_CRC_FAIL: {
                        protocolData["tb_crc_fail"] = buffer_info.tb_crc_fail;
                        break;
                    }
                    case e3::StreamType::CB_ERRORS: {
                        protocolData["cb_errors"] = buffer_info.cb_errors;
                        break;
                    }
                    case e3::StreamType::RSRP: {
                        protocolData["rsrp"] = buffer_info.rsrp;
                        break;
                    }
                    case e3::StreamType::CQI: {
                        protocolData["cqi"] = buffer_info.cqi;
                        break;
                    }
                    case e3::StreamType::CB_COUNT: {
                        protocolData["cb_count"] = buffer_info.cb_count;
                        break;
                    }
                    case e3::StreamType::RSSI: {
                        protocolData["rssi"] = buffer_info.rssi;
                        break;
                    }
                    case e3::StreamType::QAM_MOD_ORDER: {
                        protocolData["qam_mod_order"] = buffer_info.qam_mod_order;
                        break;
                    }
                    case e3::StreamType::MCS_INDEX: {
                        protocolData["mcs_index"] = buffer_info.mcs_index;
                        break;
                    }
                    case e3::StreamType::MCS_TABLE_INDEX: {
                        protocolData["mcs_table_index"] = buffer_info.mcs_table_index;
                        break;
                    }
                    case e3::StreamType::RB_START: {
                        protocolData["rb_start"] = buffer_info.rb_start;
                        break;
                    }
                    case e3::StreamType::RB_SIZE: {
                        protocolData["rb_size"] = buffer_info.rb_size;
                        break;
                    }
                    case e3::StreamType::START_SYMBOL_INDEX: {
                        protocolData["start_symbol_index"] = buffer_info.start_symbol_index;
                        break;
                    }
                    case e3::StreamType::NR_OF_SYMBOLS: {
                        protocolData["nr_of_symbols"] = buffer_info.nr_of_symbols;
                        break;
                    }
                    case e3::StreamType::NONE:
                    default: {
                        // Ignore unrecognized or unset stream types
                        break;
                    }
                }
                remaining_streams &= ~lowest_bit;
            }

            if (!protocolData.empty()) {
                notif_json["protocolData"] = protocolData;
            }

            // No ZMQ topic prefix: dApps filter client-side on dAppIdentifier
            try {
                NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: Before ZMQ send at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
                const std::string message = notif_json.dump();
                {
                    std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                    e3_pub_socket.send(zmq::buffer(message), zmq::send_flags::dontwait);
                }
                NVLOGD_FMT(TAG_E3, "TIMESTAMP_LOG: After ZMQ send at {}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
                NVLOGD_FMT(TAG_E3, "Sent E3 indication to dApp {} for subscription {}", sub.dapp_id, sub.subscription_id);
            } catch (const zmq::error_t& e) {
                NVLOGD_FMT(TAG_E3, "No subscribers for E3 indication");
            }
            sub.last_update = now;
        }
    }
} 

// Thread functions

// E3 data server thread - handles ZMQ request/reply
void E3Agent::dataServerThread()
{
    e3_rep_socket.set(zmq::sockopt::rcvtimeo, 1000);

    NVLOGC_FMT(TAG_E3, "E3 data server thread started");

    while (e3_running) {
        zmq::message_t request;
        if (e3_rep_socket.recv(request, zmq::recv_flags::none)) {
            std::string response;
            try {
                const json req_json = json::parse(std::string(static_cast<char*>(request.data()), request.size()));
                const std::string type = req_json.value("type", "");

                NVLOGC_FMT(TAG_E3, "Received E3 request: {}", req_json.dump());

                if (type == "setupRequest") {
                    handleSetupRequest(req_json, response);
                } else {
                    json error_resp;
                    error_resp["type"] = type;
                    error_resp["id"] = generateMessageId();
                    error_resp["requestId"] = req_json.value("id", 0u);
                    error_resp["responseCode"] = "negative";
                    error_resp["message"] = "unknown request type";
                    response = error_resp.dump();
                }
            } catch (const json::parse_error& e) {
                json error_resp;
                error_resp["responseCode"] = "negative";
                error_resp["message"] = "invalid JSON format";
                response = error_resp.dump();
                NVLOGC_FMT(TAG_E3, "Failed to parse request: {}", e.what());
            }
            e3_rep_socket.send(zmq::buffer(response));
        }
    }
    NVLOGC_FMT(TAG_E3, "E3 data server thread stopped");
}

// E3 reaper thread - cleanup disconnected dApps
void E3Agent::reaperThread()
{
    NVLOGC_FMT(TAG_E3, "E3 reaper thread started");

    while (e3_reaper_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        reapTimedOutDapps();
    }
    NVLOGC_FMT(TAG_E3, "E3 reaper thread stopped");
}

// Remove timed-out dApps
// NOTE: dApps with active subscriptions are kept alive even if inactive, since indications
// are fire-and-forget (no ACK). A crashed dApp with subscriptions won't be reaped until
// an explicit release message is sent.
void E3Agent::reapTimedOutDapps()
{
    constexpr auto ACTIVITY_TIMEOUT_SECONDS = 1800;

    const auto now = std::chrono::steady_clock::now();

    // Expire time-bounded subscriptions
    {
        std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
        for (auto it = e3_subscriptions.begin(); it != e3_subscriptions.end(); ) {
            if (now >= it->second.expiry_time) {
                NVLOGC_FMT(TAG_E3, "Subscription {} expired for dApp {}", it->first, it->second.dapp_id);
                it = e3_subscriptions.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<uint32_t> timed_out_dapps;

    {
        std::lock_guard<std::mutex> lock(e3_dapps_mutex);
        for (auto const& [dapp_id, conn_info] : e3_connected_dapps) {
            if (std::chrono::duration_cast<std::chrono::seconds>(now - conn_info.last_activity_time).count() > ACTIVITY_TIMEOUT_SECONDS) {
                bool has_active_subscriptions = false;
                {
                    std::lock_guard<std::mutex> subs_lock(e3_subscriptions_mutex);
                    for (const auto& [sub_id, sub] : e3_subscriptions) {
                        if (sub.dapp_id == dapp_id) {
                            has_active_subscriptions = true;
                            break;
                        }
                    }
                }

                if (!has_active_subscriptions) {
                    timed_out_dapps.push_back(dapp_id);
                }
            }
        }
    }

    for (const uint32_t dapp_id : timed_out_dapps) {
        NVLOGC_FMT(TAG_E3, "dApp {} timed out after {} seconds of inactivity. Releasing.", dapp_id, ACTIVITY_TIMEOUT_SECONDS);
        sendRelease(dapp_id);
    }
}

// Manager subscription thread - receives commands from E3 Manager
void E3Agent::managerSubscriptionThread()
{
    NVLOGC_FMT(TAG_E3, "E3 Manager subscription thread started");

    while (e3_sub_running) {
        try {
            zmq::message_t msg;
            const auto result = e3_sub_socket.recv(msg, zmq::recv_flags::dontwait);

            if (result) {
                try {
                    const json msg_json = json::parse(msg.to_string());
                    handleManagerMessage(msg_json);
                } catch (const json::exception& e) {
                    NVLOGC_FMT(TAG_E3, "Failed to parse dApp message: {}", e.what());
                }
            }
        } catch (const zmq::error_t& e) {
            if (e.num() == ETERM) {
                // Context terminated during shutdown - exit gracefully
                break;
            } else if (e.num() != EAGAIN) {
                NVLOGC_FMT(TAG_E3, "Error receiving from Manager: {}", e.what());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    NVLOGC_FMT(TAG_E3, "E3 Manager subscription thread stopped");
}

// Handle messages asynchronously received from E3 Manager via PUB-SUB
void E3Agent::handleManagerMessage(const json& message)
{
    NVLOGD_FMT(TAG_E3, "Handling Manager message: {}", message.dump());

    const std::string type = message.value("type", "");

    if (type == "releaseMessage") {
        const uint32_t dapp_id = message.value("dAppIdentifier", 0u);
        if (dapp_id == 0) {
            NVLOGC_FMT(TAG_E3, "Received e3_release with invalid dAppIdentifier");
            return;
        }
        NVLOGC_FMT(TAG_E3, "Received e3_release from dApp {}", dapp_id);
        releaseDapp(dapp_id);

    } else if (type == "subscriptionRequest" || type == "subscriptionDelete") {
        // Subscription Request/Delete via PUB-SUB: process and publish response on Agent PUB
        const uint32_t dapp_id = message.value("dAppIdentifier", 0u);

        // Silently ignore requests for dApps we don't own (multi-agent correctness)
        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            if (e3_connected_dapps.find(dapp_id) == e3_connected_dapps.end()) {
                return;
            }
        }

        std::string response;
        if (type == "subscriptionRequest") {
            handleSubscriptionRequest(message, response);
        } else if (type == "subscriptionDelete") {
            handleSubscriptionDelete(message, response);
        }

        // Publish response on Agent PUB socket
        try {
            {
                std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                e3_pub_socket.send(zmq::buffer(response), zmq::send_flags::dontwait);
            }
            NVLOGC_FMT(TAG_E3, "Published subscription response to dApp {}", dapp_id);
        } catch (const zmq::error_t& e) {
            NVLOGC_FMT(TAG_E3, "Failed to publish subscription response to dApp {}: {}", dapp_id, e.what());
        }

    } else if (type == "dAppControlAction") {
        const uint32_t dapp_id = message.value("dAppIdentifier", 0u);

        {
            std::lock_guard<std::mutex> lock(e3_dapps_mutex);
            if (e3_connected_dapps.find(dapp_id) == e3_connected_dapps.end()) {
                return;
            }
        }

        std::string response;
        handleControlMessage(message, response);

        // Optional ack to control message
        try {
            {
                std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
                e3_pub_socket.send(zmq::buffer(response), zmq::send_flags::dontwait);
            }
        } catch (const zmq::error_t& e) {
            NVLOGC_FMT(TAG_E3, "Failed to publish control ack to dApp {}: {}", dapp_id, e.what());
        }
    }
}

bool E3Agent::updateDappActivity(uint32_t dapp_id)
{
    std::lock_guard<std::mutex> lock(e3_dapps_mutex);
    auto it = e3_connected_dapps.find(dapp_id);
    if (it == e3_connected_dapps.end()) return false;
    it->second.last_activity_time = std::chrono::steady_clock::now();
    return true;
}

// Release a dApp: remove all subscriptions and connection state
void E3Agent::releaseDapp(uint32_t dapp_id)
{
    std::lock_guard<std::mutex> dapps_lock(e3_dapps_mutex);
    std::lock_guard<std::mutex> subs_lock(e3_subscriptions_mutex);

    auto it = e3_connected_dapps.find(dapp_id);
    if (it == e3_connected_dapps.end()) {
        NVLOGC_FMT(TAG_E3, "Release: dApp {} not found, ignoring", dapp_id);
        return;
    }

    e3_connected_dapps.erase(it);

    for (auto sub_it = e3_subscriptions.begin(); sub_it != e3_subscriptions.end(); ) {
        if (sub_it->second.dapp_id == dapp_id) {
            NVLOGC_FMT(TAG_E3, "Release: removing subscription {} for dApp {}", sub_it->first, dapp_id);
            sub_it = e3_subscriptions.erase(sub_it);
        } else {
            ++sub_it;
        }
    }

    NVLOGC_FMT(TAG_E3, "dApp {} released successfully", dapp_id);
}

// Send e3_release to a dApp via PUB socket
bool E3Agent::sendRelease(uint32_t dapp_id)
{
    // Verify dApp exists before publishing
    {
        std::lock_guard<std::mutex> lock(e3_dapps_mutex);
        if (e3_connected_dapps.find(dapp_id) == e3_connected_dapps.end()) {
            NVLOGC_FMT(TAG_E3, "sendRelease: dApp {} not found", dapp_id);
            return false;
        }
    }

    json release_msg;
    release_msg["type"] = "releaseMessage";
    release_msg["id"] = generateMessageId();
    release_msg["dAppIdentifier"] = dapp_id;

    const std::string message = release_msg.dump();

    try {
        {
            std::lock_guard<std::mutex> lock(e3_pub_socket_mutex_);
            e3_pub_socket.send(zmq::buffer(message), zmq::send_flags::dontwait);
        }
        NVLOGC_FMT(TAG_E3, "Sent e3_release to dApp {}", dapp_id);
    } catch (const zmq::error_t& e) {
        NVLOGC_FMT(TAG_E3, "Failed to send e3_release to dApp {}: {}", dapp_id, e.what());
        return false;
    }

    releaseDapp(dapp_id);
    return true;
}

// E3AP Message helpers

uint32_t E3Agent::generateMessageId()
{
    static std::atomic<uint32_t> message_counter{1};
    return message_counter.fetch_add(1);
}

uint32_t E3Agent::generateDappId()
{
    static std::atomic<uint32_t> dapp_counter{1};
    return dapp_counter.fetch_add(1);
}

uint32_t E3Agent::generateSubscriptionId()
{
    static std::atomic<uint32_t> sub_counter{1};
    return sub_counter.fetch_add(1);
}

// Stream creation helpers

json E3Agent::createIndicationPayloadDelivery(const std::string& stream_id) const
{
    json delivery;
    delivery["transport_type"] = "protocolData";
    delivery["keyword"] = stream_id;
    delivery["encoding"] = "json";
    return delivery;
}

json E3Agent::createIndicationPayloadStream(
    const std::string& stream_id,
    const std::string& data_type,
    const std::string& description
) const
{
    json stream;
    uint64_t val = static_cast<uint64_t>(e3::streamNameToType(stream_id));
    if (val == 0) {
        NVLOGC_FMT(TAG_E3, "createIndicationPayloadStream: unknown stream_id '{}', telemetryIdentifier will be 0", stream_id);
        stream["telemetryIdentifier"] = 0;
    } else {
        uint32_t pos = 0;
        while (val >>= 1) ++pos;
        stream["telemetryIdentifier"] = pos + 1;
    }
    stream["stream_id"] = stream_id;
    stream["data_type"] = data_type;
    stream["description"] = description;
    stream["status"] = "available";
    stream["delivery_method"] = createIndicationPayloadDelivery(stream_id);
    return stream;
}

json E3Agent::createSharedMemoryStream(
    const std::string& stream_id,
    const std::string& data_type,
    const std::string& description,
    const size_t memory_size_bytes,
    const uint32_t max_elements,
    const json& additional_shm_info,
    const json& data_schema
) const
{
    json stream;
    uint64_t val = static_cast<uint64_t>(e3::streamNameToType(stream_id));
    if (val == 0) {
        NVLOGC_FMT(TAG_E3, "createSharedMemoryStream: unknown stream_id '{}', telemetryIdentifier will be 0", stream_id);
        stream["telemetryIdentifier"] = 0;
    } else {
        uint32_t pos = 0;
        while (val >>= 1) ++pos;
        stream["telemetryIdentifier"] = pos + 1;
    }
    stream["stream_id"] = stream_id;
    stream["data_type"] = data_type;
    stream["description"] = description;
    stream["status"] = "available";

    json delivery;
    delivery["transport_type"] = "shared_memory";

    json shm_info;
    shm_info["memory_key"] = E3_SHARED_MEMORY_KEY;
    shm_info["memory_size_bytes"] = memory_size_bytes;
    shm_info["access_pattern"] = "double_buffer";
    shm_info["max_elements"] = max_elements;

    if (!additional_shm_info.empty()) {
        shm_info.update(additional_shm_info);
    }

    delivery["shared_memory_info"] = shm_info;
    stream["delivery_method"] = delivery;

    if (!data_schema.empty()) {
        stream["data_schema"] = data_schema;
    }

    return stream;
}

// Request handlers

void E3Agent::handleSetupRequest(const json& request, std::string& response) {
	json response_json;
	json e3_setup_response;
	uint32_t request_id = 0;
	
	try {
		const json& e3_setup_req = request;
		
		request_id = e3_setup_req.value("id", 0u);
		std::string protocol_version = e3_setup_req.value("e3apProtocolVersion", "");
		std::string dapp_name = e3_setup_req.value("dAppName", "unknown");
		std::string dapp_version = e3_setup_req.value("dAppVersion", "unknown");
		std::string vendor = e3_setup_req.value("vendor", "unknown");
		
		NVLOGC_FMT(TAG_E3, "E3 Setup Request from dApp '{}' v{} by {} (E3AP v{})",
				   dapp_name, dapp_version, vendor, protocol_version);
		
		if (protocol_version != e3::E3AP_PROTOCOL_VERSION) {
			NVLOGC_FMT(TAG_E3, "E3 Setup rejected: protocol version mismatch (received '{}', expected '{}')",
					   protocol_version, e3::E3AP_PROTOCOL_VERSION);
			json error_resp;
			error_resp["type"] = "setupResponse";
			error_resp["id"] = generateMessageId();
			error_resp["requestId"] = request_id;
			error_resp["responseCode"] = "negative";
			error_resp["message"] = "protocol version mismatch";
			error_resp["e3apProtocolVersion"] = e3::E3AP_PROTOCOL_VERSION;
			response = error_resp.dump();
			return;
		}
		
		// Generate dApp ID during setup phase
		const uint32_t dapp_id = generateDappId();
		{
			std::lock_guard<std::mutex> lock(e3_dapps_mutex);
			e3_connected_dapps[dapp_id] = {std::chrono::steady_clock::now()};
		}
		
		// Create E3AP Setup Response
		e3_setup_response["type"] = "setupResponse";
		e3_setup_response["id"] = generateMessageId();
		e3_setup_response["requestId"] = request_id;
		e3_setup_response["responseCode"] = "positive";
		e3_setup_response["e3apProtocolVersion"] = e3::E3AP_PROTOCOL_VERSION;
		e3_setup_response["dAppIdentifier"] = dapp_id;
		e3_setup_response["ranIdentifier"] = e3::RAN_IDENTIFIER;
		
		// Available data streams
		json available_data_streams = json::array();
		
		// IQ Samples stream (FH Data)
		available_data_streams.push_back(createSharedMemoryStream(
			"iq_samples",
			"array(int16)",
			"Raw IQ samples (Fronthaul data)",
			numFhSamples * numRowsToInsertFh * sizeof(fhDataType),
			numRowsToInsertFh
		));

		// PDU Data stream (PUSCH Data)
		available_data_streams.push_back(createSharedMemoryStream(
			"pdu_data",
			"array(uint8)",
			"PUSCH PDU data",
			80000 * numRowsToInsertPusch,
			numRowsToInsertPusch
		));
		
		// H Estimates stream
		json hest_shm_info;
		hest_shm_info["max_samples_per_row"] = maxHestSamplesPerRow;
		
		json hest_schema;
		hest_schema["dimensions"] = "Variable: (N_BS_ANTS, N_LAYERS, NF, NH) for first UE group";
		hest_schema["N_BS_ANTS"] = "Number of base station antennas (limited to 4)";
		hest_schema["N_LAYERS"] = "Number of spatial layers";
		hest_schema["NF"] = "Number of subcarriers (PRBs * 12)";
		hest_schema["NH"] = "Number of DMRS estimates";
		
		available_data_streams.push_back(createSharedMemoryStream(
			"h_estimates",
			"array(complex64)",
			"PUSCH H matrix estimates (first UE group only)",
			maxHestSamplesPerRow * numRowsToInsertHest * sizeof(hestDataType),
			numRowsToInsertHest,
			hest_shm_info,
			hest_schema
		));
		
		// Timing streams
		available_data_streams.push_back(createIndicationPayloadStream("timestamp", "uint64", "L1 software timestamp in nanoseconds"));
		available_data_streams.push_back(createIndicationPayloadStream("sfn", "uint16", "Network frame timing information"));
		available_data_streams.push_back(createIndicationPayloadStream("slot", "uint16", "Network slot timing information"));
		available_data_streams.push_back(createIndicationPayloadStream("cell_id", "uint16", "Physical Cell ID"));
		
		// Antenna and cell configuration streams
		available_data_streams.push_back(createIndicationPayloadStream("n_rx_ant", "uint16", "Number of receive antennas"));
		available_data_streams.push_back(createIndicationPayloadStream("n_rx_ant_srs", "uint16", "Number of SRS receive antennas"));
		available_data_streams.push_back(createIndicationPayloadStream("n_cells", "uint16", "Number of cells"));
		
		// H Estimates metadata streams
		available_data_streams.push_back(createIndicationPayloadStream("n_bs_ants", "uint8", "Number of base station antennas in H estimates"));
		available_data_streams.push_back(createIndicationPayloadStream("n_layers", "uint8", "Number of spatial layers in H estimates"));
		available_data_streams.push_back(createIndicationPayloadStream("n_subcarriers", "uint16", "Number of subcarriers (PRBs * 12) in H estimates"));
		available_data_streams.push_back(createIndicationPayloadStream("n_dmrs_estimates", "uint8", "Number of DMRS estimates in H matrix"));
		available_data_streams.push_back(createIndicationPayloadStream("dmrs_symb_pos", "uint16", "DMRS symbol positions bitmap"));
		
		// Quality and error metrics streams
		available_data_streams.push_back(createIndicationPayloadStream("tb_crc_fail", "uint8", "Transport Block CRC aggregated failure indicator (0=success,1=failure)"));
		available_data_streams.push_back(createIndicationPayloadStream("cb_errors", "uint32", "Code Block CRC error count per UE"));
		available_data_streams.push_back(createIndicationPayloadStream("rsrp", "float32", "Reference Signal Received Power per UE in dB"));
		available_data_streams.push_back(createIndicationPayloadStream("cqi", "float32", "SINR post-equalization per UE in dB (also known as CQI)"));
		available_data_streams.push_back(createIndicationPayloadStream("cb_count", "uint16", "Number of Code Blocks per UE transport block"));
		available_data_streams.push_back(createIndicationPayloadStream("rssi", "float32", "Received Signal Strength Indicator per UE group in dB"));
		
		// Modulation and coding scheme streams
		available_data_streams.push_back(createIndicationPayloadStream("qam_mod_order", "uint8", "QAM modulation order (2,4,6,8 if transform precoding disabled; 1,2,4,6,8 if enabled)"));
		available_data_streams.push_back(createIndicationPayloadStream("mcs_index", "uint8", "MCS index (should match value sent in DCI, range 0-31)"));
		available_data_streams.push_back(createIndicationPayloadStream("mcs_table_index", "uint8", "MCS-Table-PUSCH index (0=notqam256, 1=qam256, 2=qam64LowSE, etc.)"));
		
		// Resource allocation streams
		available_data_streams.push_back(createIndicationPayloadStream("rb_start", "uint16", "Starting resource block within the BWP for this PUSCH (resource allocation type 1)"));
		available_data_streams.push_back(createIndicationPayloadStream("rb_size", "uint16", "Number of resource blocks for this PUSCH (resource allocation type 1)"));
		available_data_streams.push_back(createIndicationPayloadStream("start_symbol_index", "uint8", "Start symbol index of PUSCH mapping from the start of the slot (range 0-13)"));
		available_data_streams.push_back(createIndicationPayloadStream("nr_of_symbols", "uint8", "PUSCH duration in symbols (range 1-14)"));
		
		// E3-RanFunctionDefinition: telemetry IDs 1..STREAM_TYPE_COUNT
		json telemetry_id_list = json::array();
		for (uint32_t id = 1; id <= e3::STREAM_TYPE_COUNT; ++id) {
			telemetry_id_list.push_back(id);
		}

		json ran_function;
		ran_function["ranFunctionIdentifier"] = e3::RAN_FUNCTION_ID_NVIDIA_KPM;
		ran_function["telemetryIdentifierList"] = telemetry_id_list;
		ran_function["controlIdentifierList"] = json::array();
		ran_function["ranFunctionData"] = available_data_streams;

		e3_setup_response["ranFunctionList"] = json::array({ran_function});
		
		response_json = e3_setup_response;
		
		NVLOGC_FMT(TAG_E3, "E3 Setup successful for dApp '{}' assigned ID: {}", dapp_name, dapp_id);
		
	} catch (const json::exception& e) {
		NVLOGC_FMT(TAG_E3, "Error processing E3 Setup Request: {}", e.what());
		json error_resp;
		error_resp["type"] = "setupResponse";
		error_resp["id"] = generateMessageId();
		error_resp["requestId"] = request_id;
		error_resp["responseCode"] = "negative";
		error_resp["message"] = "invalid setup request format";
		response = error_resp.dump();
		return;
	}
	
	response = response_json.dump();
}

void E3Agent::handleSubscriptionRequest(const json& request, std::string& response)
{
    json response_json;
    json e3_sub_response;
    uint32_t dapp_id = 0;
    uint32_t request_id = 0;

    try {
        const json& e3_sub_req = request;
        dapp_id = e3_sub_req.at("dAppIdentifier").get<uint32_t>();
        request_id = e3_sub_req.value("id", 0u);

        if (!updateDappActivity(dapp_id)) {
            NVLOGC_FMT(TAG_E3, "Subscription rejected for non-connected dApp {}", dapp_id);
            e3_sub_response["responseCode"] = "negative";
            e3_sub_response["message"] = "dApp not connected or timed out";
        } else {
            uint32_t ran_func_id = e3_sub_req.at("ranFunctionIdentifier").get<uint32_t>();
            if (ran_func_id != e3::RAN_FUNCTION_ID_NVIDIA_KPM) {
                NVLOGC_FMT(TAG_E3, "Subscription rejected: unsupported ranFunctionIdentifier {} (expected {})",
                           ran_func_id, e3::RAN_FUNCTION_ID_NVIDIA_KPM);
                e3_sub_response["responseCode"] = "negative";
                e3_sub_response["message"] = "unsupported ranFunctionIdentifier";
            } else {
                auto telemetry_ids = e3_sub_req.value("telemetryIdentifierList", std::vector<uint32_t>{});
                auto control_ids = e3_sub_req.value("controlIdentifierList", std::vector<uint32_t>{});
                uint32_t periodicity_us = e3_sub_req.value("periodicity", 100000u);
                uint32_t subscription_time_s = e3_sub_req.value("subscriptionTime", 0u);

                e3::StreamType stream_bitfield = e3::StreamType::NONE;
                bool valid = true;

                // NVIDIA KPM: telemetry-only, no control dispatch yet.
                // Relax to (telemetry_ids.empty() && control_ids.empty()) when controls are implemented.
                if (telemetry_ids.empty()) {
                    NVLOGC_FMT(TAG_E3, "Subscription rejected: empty telemetryIdentifierList");
                    e3_sub_response["responseCode"] = "negative";
                    e3_sub_response["message"] = "telemetryIdentifierList must not be empty";
                    valid = false;
                }

                // Validate all telemetry IDs
                for (uint32_t tid : telemetry_ids) {
                    e3::StreamType st = e3::telemetryIdToStreamType(tid);
                    if (st == e3::StreamType::NONE) {
                        NVLOGC_FMT(TAG_E3, "Subscription rejected: invalid telemetry ID {}", tid);
                        e3_sub_response["responseCode"] = "negative";
                        e3_sub_response["message"] = "invalid telemetry identifier";
                        valid = false;
                        break;
                    }
                    stream_bitfield |= st;
                }

                if (valid) {
                    uint32_t sub_id = generateSubscriptionId();
                    auto now = std::chrono::steady_clock::now();
                    auto expiry = (subscription_time_s > 0)
                        ? now + std::chrono::seconds(subscription_time_s)
                        : std::chrono::steady_clock::time_point::max();

                    {
                        std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
                        e3_subscriptions[sub_id] = {sub_id, dapp_id, ran_func_id, telemetry_ids, stream_bitfield, periodicity_us, now, expiry};
                    }

                    std::string ids_str = fmt::format("[{}]", fmt::join(telemetry_ids, ","));
                    NVLOGC_FMT(TAG_E3, "E3 Subscription {} created for dApp {} (ranFunction={}, telemetryIds={})",
                               sub_id, dapp_id, ran_func_id, ids_str);

                    e3_sub_response["responseCode"] = "positive";
                    e3_sub_response["subscriptionId"] = sub_id;
                    e3_sub_response["ranFunctionIdentifier"] = ran_func_id;
                    e3_sub_response["telemetryGrantedList"] = telemetry_ids;
                    e3_sub_response["controlGrantedList"] = json::array();
                    e3_sub_response["periodicity"] = periodicity_us;
                }
            }
        }

        e3_sub_response["type"] = "subscriptionResponse";
        e3_sub_response["id"] = generateMessageId();
        e3_sub_response["requestId"] = request_id;
        e3_sub_response["dAppIdentifier"] = dapp_id;
        response_json = e3_sub_response;

    } catch (const json::exception& e) {
        NVLOGC_FMT(TAG_E3, "Error processing E3 Subscription Request: {}", e.what());
        json e3_err_resp;
        e3_err_resp["type"] = "subscriptionResponse";
        e3_err_resp["id"] = generateMessageId();
        e3_err_resp["requestId"] = request_id;
        e3_err_resp["responseCode"] = "negative";
        e3_err_resp["message"] = "missing or invalid parameters in subscription request";
        e3_err_resp["dAppIdentifier"] = dapp_id;
        response_json = e3_err_resp;
    }
    response = response_json.dump();
}

void E3Agent::handleSubscriptionDelete(const json& request, std::string& response)
{
    json response_json;
    json e3_unsub_response;
    uint32_t dapp_id = 0;
    uint32_t sub_id = 0;
    uint32_t request_id = 0;

    try {
        const json& e3_unsub_req = request;
        dapp_id = e3_unsub_req.at("dAppIdentifier").get<uint32_t>();
        sub_id = e3_unsub_req.at("subscriptionId").get<uint32_t>();
        request_id = e3_unsub_req.value("id", 0u);

        bool found = false;
        {
            std::lock_guard<std::mutex> lock(e3_subscriptions_mutex);
            auto it = e3_subscriptions.find(sub_id);
            if (it != e3_subscriptions.end() && it->second.dapp_id == dapp_id) {
                e3_subscriptions.erase(it);
                found = true;
            }
        }

        updateDappActivity(dapp_id);

        if (found) {
            NVLOGC_FMT(TAG_E3, "E3 Subscription Delete successful for subscription {}", sub_id);
            e3_unsub_response["responseCode"] = "positive";
        } else {
            NVLOGC_FMT(TAG_E3, "E3 Subscription Delete failed for sub_id {}, dApp_id {}", sub_id, dapp_id);
            e3_unsub_response["responseCode"] = "negative";
            e3_unsub_response["message"] = "subscription not found or dApp ID mismatch";
        }

        e3_unsub_response["subscriptionId"] = sub_id;
        e3_unsub_response["type"] = "subscriptionResponse";
        e3_unsub_response["id"] = generateMessageId();
        e3_unsub_response["requestId"] = request_id;
        e3_unsub_response["dAppIdentifier"] = dapp_id;
        response_json = e3_unsub_response;

    } catch (const json::exception& e) {
        NVLOGC_FMT(TAG_E3, "Error processing E3 Subscription Delete Request: {}", e.what());
        json e3_err_resp;
        e3_err_resp["type"] = "subscriptionResponse";
        e3_err_resp["id"] = generateMessageId();
        e3_err_resp["requestId"] = request_id;
        e3_err_resp["responseCode"] = "negative";
        e3_err_resp["message"] = "missing or invalid parameters in subscription delete request";
        e3_err_resp["dAppIdentifier"] = dapp_id;
        e3_err_resp["subscriptionId"] = sub_id;
        response_json = e3_err_resp;
    }
    response = response_json.dump();
}


void E3Agent::handleControlMessage(const json& request, std::string& response)
{
    json ack;
    uint32_t dapp_id = 0;
    uint32_t request_id = 0;

    try {
        dapp_id = request.at("dAppIdentifier").get<uint32_t>();
        request_id = request.value("id", 0u);

        updateDappActivity(dapp_id);

        // Control actions not implemented
        ack["responseCode"] = "negative";
        ack["message"] = "control actions not implemented";

    } catch (const json::exception& e) {
        ack["responseCode"] = "negative";
        ack["message"] = "invalid control message format";
        NVLOGC_FMT(TAG_E3, "Error processing E3 Control Message: {}. Request: {}", e.what(), request.dump());
    }

    ack["type"] = "messageAck";
    ack["id"] = generateMessageId();
    ack["requestId"] = request_id;
    ack["dAppIdentifier"] = dapp_id;

    response = ack.dump();
}
