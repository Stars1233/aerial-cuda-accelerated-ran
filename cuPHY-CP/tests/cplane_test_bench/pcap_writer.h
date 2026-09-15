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
 * @file pcap_writer.h
 * @brief Standalone PCAP file writer for DPDK mbufs
 * 
 * This module provides functionality to write DPDK rte_mbuf packets to PCAP files
 * that can be opened and analyzed in Wireshark. The implementation follows the
 * standard PCAP file format specification without using any DPDK services.
 */

#ifndef PCAP_WRITER_H
#define PCAP_WRITER_H

#include <stdint.h>
#include <stdio.h>
#include <rte_mbuf.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief PCAP file format constants
 */
#define PCAP_MAGIC_NUMBER       0xa1b2c3d4  //!< PCAP magic number (native byte order)
#define PCAP_VERSION_MAJOR      2           //!< PCAP major version
#define PCAP_VERSION_MINOR      4           //!< PCAP minor version
#define PCAP_SNAPLEN            65535       //!< Maximum packet length to capture
#define PCAP_NETWORK_ETHERNET   1           //!< Ethernet link type

/**
 * @brief PCAP global header structure
 * 
 * This header appears at the beginning of every PCAP file and defines
 * the overall properties of the capture file.
 */
typedef struct {
    uint32_t magic_number;   //!< Magic number identifying the file format
    uint16_t version_major;  //!< Major version number
    uint16_t version_minor;  //!< Minor version number
    int32_t  thiszone;       //!< GMT to local correction (always 0)
    uint32_t sigfigs;        //!< Accuracy of timestamps (always 0)
    uint32_t snaplen;        //!< Maximum length of captured packets
    uint32_t network;        //!< Data link type (1 = Ethernet)
} __attribute__((packed)) pcap_global_header_t;

/**
 * @brief PCAP packet header structure
 * 
 * This header precedes each packet in the PCAP file and contains
 * timestamp and length information.
 */
typedef struct {
    uint32_t ts_sec;         //!< Timestamp seconds since epoch
    uint32_t ts_usec;        //!< Timestamp microseconds
    uint32_t incl_len;       //!< Number of octets of packet saved in file
    uint32_t orig_len;       //!< Actual length of packet
} __attribute__((packed)) pcap_packet_header_t;

/**
 * @brief Write DPDK mbufs to a PCAP file
 * 
 * This function creates a PCAP file that can be opened in Wireshark for
 * packet analysis. It handles chained mbufs properly and generates
 * appropriate timestamps.
 * 
 * @param[in] filename      Output PCAP file path
 * @param[in] mbufs         Array of rte_mbuf pointers to write
 * @param[in] num_packets   Number of packets in the array
 * @param[in] use_timestamps If true, use current system time; if false, use sequential timestamps
 * 
 * @return 0 on success, negative error code on failure
 *   - -1: Failed to open output file
 *   - -2: Failed to write global header
 *   - -3: Failed to write packet header
 *   - -4: Failed to write packet data
 *   - -5: Invalid input parameters
 */
int pcap_write_mbufs(const char* filename, 
                     struct rte_mbuf** mbufs, 
                     uint32_t num_packets,
                     bool use_timestamps);

/**
 * @brief Write DPDK mbufs to a PCAP file with custom timestamps
 * 
 * Advanced version that allows specifying custom timestamps for each packet.
 * Useful when packets have specific timing requirements or when replaying
 * captured traffic with original timing.
 * 
 * @param[in] filename      Output PCAP file path
 * @param[in] mbufs         Array of rte_mbuf pointers to write
 * @param[in] timestamps    Array of timestamps (seconds since epoch)
 * @param[in] num_packets   Number of packets in the arrays
 * 
 * @return 0 on success, negative error code on failure
 */
int pcap_write_mbufs_with_timestamps(const char* filename,
                                     struct rte_mbuf** mbufs,
                                     const struct timespec* timestamps,
                                     uint32_t num_packets);

/**
 * @brief Append DPDK mbufs to an existing PCAP file
 * 
 * This function appends packets to an existing PCAP file. The file must
 * already have a valid PCAP global header.
 * 
 * @param[in] filename      Existing PCAP file path
 * @param[in] mbufs         Array of rte_mbuf pointers to append
 * @param[in] num_packets   Number of packets in the array
 * @param[in] use_timestamps If true, use current system time; if false, use sequential timestamps
 * 
 * @return 0 on success, negative error code on failure
 */
int pcap_append_mbufs(const char* filename,
                      struct rte_mbuf** mbufs,
                      uint32_t num_packets,
                      bool use_timestamps);

/**
 * @brief Get the total data length of an mbuf chain
 * 
 * Helper function to calculate the total data length of a potentially
 * chained mbuf structure.
 * 
 * @param[in] mbuf  Head of the mbuf chain
 * @return Total data length in bytes
 */
static inline uint32_t pcap_get_mbuf_total_len(const struct rte_mbuf* mbuf)
{
    if (!mbuf) {
        return 0;
    }
    return mbuf->pkt_len;
}

/**
 * @brief Validate mbuf for PCAP writing
 * 
 * Helper function to validate that an mbuf is suitable for writing to PCAP.
 * Checks for null pointers, zero length, and other common issues.
 * 
 * @param[in] mbuf  Mbuf to validate
 * @return true if mbuf is valid, false otherwise
 */
static inline bool pcap_validate_mbuf(const struct rte_mbuf* mbuf)
{
    return (mbuf != NULL && mbuf->pkt_len > 0 && mbuf->data_len > 0);
}

#ifdef __cplusplus
}
#endif

#endif /* PCAP_WRITER_H */





