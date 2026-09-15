/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file pcap_writer.cpp
 * @brief Implementation of standalone PCAP file writer for DPDK mbufs
 * 
 * This module provides functionality to write DPDK rte_mbuf packets to PCAP files
 * that can be opened and analyzed in Wireshark. The implementation follows the
 * standard PCAP file format specification without using any DPDK services.
 */

#include "pcap_writer.h"
#include <time.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * @brief Write PCAP global header to file
 * 
 * @param[in] file  File pointer to write to
 * @return 0 on success, -1 on failure
 */
static int write_pcap_global_header(FILE* file)
{
    const pcap_global_header_t global_header = {
        .magic_number = PCAP_MAGIC_NUMBER,
        .version_major = PCAP_VERSION_MAJOR,
        .version_minor = PCAP_VERSION_MINOR,
        .thiszone = 0,
        .sigfigs = 0,
        .snaplen = PCAP_SNAPLEN,
        .network = PCAP_NETWORK_ETHERNET
    };

    const size_t written = fwrite(&global_header, sizeof(global_header), 1, file);
    return (written == 1) ? 0 : -1;
}

/**
 * @brief Write PCAP packet header to file
 * 
 * @param[in] file      File pointer to write to
 * @param[in] ts_sec    Timestamp seconds since epoch
 * @param[in] ts_usec   Timestamp microseconds
 * @param[in] pkt_len   Packet length in bytes
 * @return 0 on success, -1 on failure
 */
static int write_pcap_packet_header(FILE* file, uint32_t ts_sec, uint32_t ts_usec, uint32_t pkt_len)
{
    const pcap_packet_header_t packet_header = {
        .ts_sec = ts_sec,
        .ts_usec = ts_usec,
        .incl_len = pkt_len,
        .orig_len = pkt_len
    };

    const size_t written = fwrite(&packet_header, sizeof(packet_header), 1, file);
    return (written == 1) ? 0 : -1;
}

/**
 * @brief Write mbuf data to file, handling chained mbufs
 * 
 * @param[in] file  File pointer to write to
 * @param[in] mbuf  Head of mbuf chain to write
 * @return 0 on success, -1 on failure
 */
static int write_mbuf_data(FILE* file, const struct rte_mbuf* mbuf)
{
    const struct rte_mbuf* current = mbuf;
    
    while (current != nullptr) {
        // Get pointer to packet data in this segment
        const uint8_t* data = rte_pktmbuf_mtod(current, const uint8_t*);
        const uint16_t data_len = rte_pktmbuf_data_len(current);
        
        if (data_len > 0) {
            const size_t written = fwrite(data, 1, data_len, file);
            if (written != data_len) {
                return -1;
            }
        }
        
        // Move to next segment in chain
        current = current->next;
    }
    
    return 0;
}

/**
 * @brief Get current timestamp
 * 
 * @param[out] ts_sec   Timestamp seconds since epoch
 * @param[out] ts_usec  Timestamp microseconds
 */
static void get_current_timestamp(uint32_t* ts_sec, uint32_t* ts_usec)
{
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    
    *ts_sec = static_cast<uint32_t>(ts.tv_sec);
    *ts_usec = static_cast<uint32_t>(ts.tv_nsec / 1000); // Convert nanoseconds to microseconds
}

/**
 * @brief Check if file exists and has valid PCAP header
 * 
 * @param[in] filename  File path to check
 * @return true if file exists with valid PCAP header, false otherwise
 */
static bool is_valid_pcap_file(const char* filename)
{
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return false;
    }
    
    pcap_global_header_t header{};
    const size_t read_bytes = fread(&header, sizeof(header), 1, file);
    fclose(file);
    
    return (read_bytes == 1 && header.magic_number == PCAP_MAGIC_NUMBER);
}

int pcap_write_mbufs(const char* filename, 
                     struct rte_mbuf** mbufs, 
                     uint32_t num_packets,
                     bool use_timestamps)
{
    // Validate input parameters
    if (!filename || !mbufs || num_packets == 0) {
        return -5;
    }

    // Open file for writing (binary mode)
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return -1;
    }

    // Write PCAP global header
    if (write_pcap_global_header(file) != 0) {
        fclose(file);
        return -2;
    }

    // Sequential timestamp for non-timestamped mode (starts at Unix epoch + 1 day)
    uint32_t sequential_sec = 86400; // 1 day after epoch
    uint32_t sequential_usec = 0;

    // Write each packet
    for (uint32_t i = 0; i < num_packets; ++i) {
        struct rte_mbuf* mbuf = mbufs[i];
        
        // Validate mbuf
        if (!pcap_validate_mbuf(mbuf)) {
            continue; // Skip invalid mbufs
        }

        uint32_t ts_sec{};
        uint32_t ts_usec{};
        
        if (use_timestamps) {
            get_current_timestamp(&ts_sec, &ts_usec);
        } else {
            // Use sequential timestamps (1ms apart)
            ts_sec = sequential_sec;
            ts_usec = sequential_usec;
            sequential_usec += 1000; // Add 1ms
            if (sequential_usec >= 1000000) {
                sequential_usec = 0;
                ++sequential_sec;
            }
        }

        const uint32_t pkt_len = pcap_get_mbuf_total_len(mbuf);

        // Write packet header
        if (write_pcap_packet_header(file, ts_sec, ts_usec, pkt_len) != 0) {
            fclose(file);
            return -3;
        }

        // Write packet data
        if (write_mbuf_data(file, mbuf) != 0) {
            fclose(file);
            return -4;
        }
    }

    fclose(file);
    return 0;
}

int pcap_write_mbufs_with_timestamps(const char* filename,
                                     struct rte_mbuf** mbufs,
                                     const struct timespec* timestamps,
                                     uint32_t num_packets)
{
    // Validate input parameters
    if (!filename || !mbufs || !timestamps || num_packets == 0) {
        return -5;
    }

    // Open file for writing (binary mode)
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return -1;
    }

    // Write PCAP global header
    if (write_pcap_global_header(file) != 0) {
        fclose(file);
        return -2;
    }

    // Write each packet with custom timestamp
    for (uint32_t i = 0; i < num_packets; ++i) {
        struct rte_mbuf* mbuf = mbufs[i];
        
        // Validate mbuf
        if (!pcap_validate_mbuf(mbuf)) {
            continue; // Skip invalid mbufs
        }

        const uint32_t ts_sec = static_cast<uint32_t>(timestamps[i].tv_sec);
        const uint32_t ts_usec = static_cast<uint32_t>(timestamps[i].tv_nsec / 1000);
        const uint32_t pkt_len = pcap_get_mbuf_total_len(mbuf);

        // Write packet header
        if (write_pcap_packet_header(file, ts_sec, ts_usec, pkt_len) != 0) {
            fclose(file);
            return -3;
        }

        // Write packet data
        if (write_mbuf_data(file, mbuf) != 0) {
            fclose(file);
            return -4;
        }
    }

    fclose(file);
    return 0;
}

int pcap_append_mbufs(const char* filename,
                      struct rte_mbuf** mbufs,
                      uint32_t num_packets,
                      bool use_timestamps)
{
    // Validate input parameters
    if (!filename || !mbufs || num_packets == 0) {
        return -5;
    }

    // Check if file exists and has valid PCAP header
    if (!is_valid_pcap_file(filename)) {
        // File doesn't exist or is invalid, create new one
        return pcap_write_mbufs(filename, mbufs, num_packets, use_timestamps);
    }

    // Open file for appending (binary mode)
    FILE* file = fopen(filename, "ab");
    if (!file) {
        return -1;
    }

    // Sequential timestamp for non-timestamped mode
    uint32_t sequential_sec = 86400; // 1 day after epoch
    uint32_t sequential_usec = 0;

    // Write each packet
    for (uint32_t i = 0; i < num_packets; ++i) {
        struct rte_mbuf* mbuf = mbufs[i];
        
        // Validate mbuf
        if (!pcap_validate_mbuf(mbuf)) {
            continue; // Skip invalid mbufs
        }

        uint32_t ts_sec{};
        uint32_t ts_usec{};
        
        if (use_timestamps) {
            get_current_timestamp(&ts_sec, &ts_usec);
        } else {
            // Use sequential timestamps (1ms apart)
            ts_sec = sequential_sec;
            ts_usec = sequential_usec;
            sequential_usec += 1000; // Add 1ms
            if (sequential_usec >= 1000000) {
                sequential_usec = 0;
                ++sequential_sec;
            }
        }

        const uint32_t pkt_len = pcap_get_mbuf_total_len(mbuf);

        // Write packet header
        if (write_pcap_packet_header(file, ts_sec, ts_usec, pkt_len) != 0) {
            fclose(file);
            return -3;
        }

        // Write packet data
        if (write_mbuf_data(file, mbuf) != 0) {
            fclose(file);
            return -4;
        }
    }

    fclose(file);
    return 0;
}





