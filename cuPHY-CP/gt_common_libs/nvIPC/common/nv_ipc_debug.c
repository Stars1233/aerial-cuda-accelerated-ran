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

#define _GNU_SOURCE /* See feature_test_macros(7) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <pcap/pcap.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <sched.h>

#include "nv_ipc.h"
#include "nv_ipc_debug.h"
#include "nv_ipc_efd.h"
#include "nv_ipc_sem.h"
#include "nv_ipc_epoll.h"
#include "nv_ipc_mempool.h"
#include "nv_ipc_utils.h"
#include "array_queue.h"
#include "nv_ipc_forward.h"
#include "nv_ipc_ring.h"
#include "nv_utils.h"

#define LOG_PATH_LEN 64

#define TAG (NVLOG_TAG_BASE_NVIPC + 10) //"NVIPC.DEBUG"
#define SHM_DEBUG_SUFFIX "_debug"

/*********** NVIPC Library Version **************
 * Note:
 * 1. Update the version number if NVIPC has significant change which requires L2 sync NVIPC accordingly.
 * 2. If partner L2 doesn't sync the same nvipc version, will see "NVIPC lib version doesn't match" log at NVIPC initialization.
 */
#define LIBRARY_VERSION (2530) // 25.3.0

#define CONFIG_LOG_ALLOCATE_TIME 1
#define CONFIG_LOG_SEND_TIME 1

// Enable nvipc pcap capture or not.
#define ENV_NVIPC_DEBUG_PCAP "NVIPC_DEBUG_EN"
#define CONFIG_DEBUG_PCAP 0

#define ENV_NVIPC_DEBUG_TIMING "NVIPC_DEBUG_TIMING"
#define CONFIG_DEBUG_TIMING 0

#define U64_1E9 (1000 * 1000 * 1000LL)
#define TIMING_STAT_INTERVAL (U64_1E9 * 1) // Print timing statistic log every 1 second

#define LOG_SHM_NAME_LEN 32

#define ENABLE_PCAP_CAPTURE 1

#define SCF_PCAP_PROTOCOL_PORT 9000

// Data link type (LINKTYPE_*): 1 - Ethernet; 113 - Linux cooked capture.
#define PCAP_DATA_LINK_TYPE 113

// Sync with scf_5g_fapi.h
#define SCF_FAPI_ERROR_INDICATION 0x07
#define SCF_FAPI_RESV_1_START 0x08
#define SCF_FAPI_RESV_1_END 0x7F
#define SCF_FAPI_RESV_2_START 0x8A
#define SCF_FAPI_RESV_2_END 0xFF

// INVALID SFN/SLOT for non slot messages like CONFIG.req
#define SFN_SLOT_INVALID 0xFFFFFFFF

static const sfn_slot_t sfn_slot_invalid = {.u32 = SFN_SLOT_INVALID};

static int id_counter = 0x1000;

static FILE*           pcapfile;
static pthread_mutex_t pcapmutex;

#ifdef DEBUG_HIGH_RESOLUTION_TIME
#define debug_get_timestamp(ts) clock_gettime(CLOCK_REALTIME, (ts))
#define debug_get_ts_interval(ts1, ts2) nvlog_timespec_interval((ts1), (ts2))
#else
#define debug_get_timestamp(ts) gettimeofday((ts), NULL)
#define debug_get_ts_interval(ts1, ts2) nvlog_timeval_interval((ts1), (ts2))
#endif

static void save_ipc_msg(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t flags);

typedef enum
{
    FAPI_TYPE_UNKNOWN = 0,
    SCF_FAPI    = 1,
} fapi_type_t;

static const int32_t fapi_type = SCF_FAPI;

typedef struct
{
    uint8_t  mac[6];
    char     ip[16];
    uint16_t port;
} net_addr_t;

// NVIDIA MAC prefix: 00:04:4B
static net_addr_t mac_addr = {
    {0x00, 0x04, 0x4B, 0x34, 0x35, 0x36},
    "192.168.1.8",
    38555};

static net_addr_t phy_addr = {
    {0x00, 0x04, 0x4B, 0x44, 0x45, 0x46},
    "192.168.1.9",
    38556};

// Record header for each packet. See https://wiki.wireshark.org/Development/LibpcapFileFormat
typedef struct
{
    uint32_t ts_sec;   /* timestamp seconds */
    uint32_t ts_usec;  /* timestamp microseconds */
    uint32_t incl_len; /* number of octets of packet saved in file */
    uint32_t orig_len; /* actual length of packet */
} record_header_t;

// SLL header for "Linux cooked capture" 16B
typedef struct
{
    uint16_t packet_type;
    uint16_t arphrd_type;
    uint16_t link_layer_addr_len;
    uint8_t  link_layer_addr[6];
    uint16_t padding;
    uint16_t protocol_type;
} sll_header_t;

// Ethernet header 14B
typedef struct
{
    uint8_t  dst_mac[6];
    uint8_t  src_mac[6];
    uint16_t type_len;
} eth_header_t;

//IP header 20B
typedef struct
{
    uint8_t  ver_hlen; // Version (4 bits) + Internet header length (4 bits)
    uint8_t  tos;      // Type of service
    uint16_t len;      // Total length
    uint16_t id;       // Identification
    uint16_t flags;
    uint8_t  ttl; // Time to live
    uint8_t  protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
} ip_header_t;

//UDP header 8B
typedef struct _udp_hdr
{
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t len;
    uint16_t checksum;
} udp_header_t;

// SCF FAPI header, copied from scf_5g_fapi.h
typedef struct
{
    uint8_t message_count;
    uint8_t handle_id;
    uint8_t payload[0];
} __attribute__((__packed__)) scf_fapi_header_t;

// SCF FAPI body header, copied from scf_5g_fapi.h
typedef struct
{
    uint16_t type_id;
    uint32_t length;
    uint8_t  next[0];
} __attribute__((__packed__)) scf_fapi_body_header_t;

typedef struct
{
    uint16_t sfn;
    uint16_t slot;
    uint8_t  next[0];
} __attribute__((__packed__)) scf_fapi_sfn_slot_t;

typedef struct
{
    scf_fapi_header_t      head;
    scf_fapi_body_header_t body;
    sfn_slot_t             sfn_slot;
} __attribute__((__packed__)) scf_fapi_slot_header_t;

// Get integer system environment value
long get_env_long(const char* name, long def)
{
    if(name == NULL)
    {
        return def;
    }

    // If CUBB_HOME was set in system environment variables, return it
    char* env = getenv(name);
    if(env == NULL)
    {
        return def;
    }

    long  val;
    char* err_ptr = NULL;
    if(strncmp(env, "0b", 2) == 0 || strncmp(env, "0B", 2) == 0)
    {
        val = strtol(env + 2, &err_ptr, 2); // Binary
    }
    else
    {
        val = strtol(env, &err_ptr, 0); // Octal, Decimal, Hex
    }

    if(err_ptr == NULL || *err_ptr != '\0')
    {
        NVLOGI(TAG, "%s: invalid variable: %s=%s", __FUNCTION__, name, env);
        return def;
    }
    else
    {
        return val;
    }
}

static int32_t parse_fapi_id(uint8_t* fapi_buf)
{
    if(fapi_type == SCF_FAPI)
    {
        scf_fapi_header_t* scf_fapi = (scf_fapi_header_t*)fapi_buf;
        if(scf_fapi->message_count == 0)
        {
            return -1;
        }
        else
        {
            scf_fapi_body_header_t* body = (scf_fapi_body_header_t*)scf_fapi->payload;
            return body->type_id;
        }
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_FAPI_EVENT, "%s unknown FAPI type %d", __func__, fapi_type);
        return -1;
    }
}

static int32_t parse_fapi_length(uint8_t* fapi_buf, int32_t max_data_size)
{
    int32_t fapi_len = 0; // The FAPI payload length, included in FAPI header

    if(fapi_type == SCF_FAPI)
    {
        scf_fapi_header_t* scf_fapi = (scf_fapi_header_t*)fapi_buf;
        fapi_len                    = sizeof(scf_fapi_header_t);
        int offset                  = 0;
        for(int i = 0; i < scf_fapi->message_count; i++)
        {
            scf_fapi_body_header_t* body = (scf_fapi_body_header_t*)(scf_fapi->payload + offset);
            fapi_len += sizeof(scf_fapi_body_header_t) + body->length;
            offset += sizeof(scf_fapi_body_header_t) + body->length;
        }
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_FAPI_EVENT, "%s unknown FAPI type %d", __func__, fapi_type);
        return -1;
    }

    return fapi_len;
}

// Write mocking common headers ahead of FAPI payload: PCAP + SLL/ETH + IP + UDP
int pcap_write_common_headers(record_t* record, int32_t fapi_len)
{
    int         ret = 0;
    net_addr_t *src, *dst;
    int32_t     dir = record->flags >> 16;
    if(dir == NVIPC_SECONDARY_TO_PRIMARY)
    {
        src = &phy_addr;
        dst = &mac_addr;
    }
    else
    {
        src = &mac_addr;
        dst = &phy_addr;
    }

    if(fapi_type == SCF_FAPI)
    {
        src->port = SCF_PCAP_PROTOCOL_PORT;
        dst->port = SCF_PCAP_PROTOCOL_PORT;
    }

    // PCAP header: 16B
    record_header_t record_hdr;
    record_hdr.ts_sec  = record->tv.tv_sec;
    record_hdr.ts_usec = record->tv.tv_usec;
#if(PCAP_DATA_LINK_TYPE == 1)
    record_hdr.incl_len = fapi_len + 8 + 20 + 14; // FAPI + UDP 8 + IP 20 + ETH 14
#elif(PCAP_DATA_LINK_TYPE == 113)
    record_hdr.incl_len = fapi_len + 8 + 20 + 16; // FAPI + UDP 8 + IP 20 + SLL 16
#else
#error Unsupported PCAP_DATA_LINK_TYPE
#endif
    record_hdr.orig_len = record_hdr.incl_len;
    if(fwrite(&record_hdr, sizeof(record_header_t), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        ret = -1;
    }

#if(PCAP_DATA_LINK_TYPE == 1)
    // ETH header: 14B
    eth_header_t eth_hdr;
    for(int i = 0; i < 6; i++)
    {
        eth_hdr.src_mac[i] = src->mac[i];
        eth_hdr.dst_mac[i] = dst->mac[i];
    }
    eth_hdr.type_len = htons(0x0800); // 0x0800 is IP(v4)
    if(fwrite(&eth_hdr, sizeof(eth_header_t), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        ret = -1;
    }
#elif(PCAP_DATA_LINK_TYPE == 113)
    // SLL header for "Linux cooked capture": 16B
    sll_header_t sll_header;
    sll_header.packet_type         = 0;
    sll_header.arphrd_type         = htons(772);
    sll_header.link_layer_addr_len = htons(6);
    for(int i = 0; i < 6; i++)
    {
        sll_header.link_layer_addr[i] = 0;
    }
    sll_header.padding       = 0;
    sll_header.protocol_type = htons(0x0800);
    if(fwrite(&sll_header, sizeof(sll_header_t), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        ret = -1;
    }
#else
#error Unsupported PCAP_DATA_LINK_TYPE
#endif

    // IP header: 20B
    ip_header_t ip_hdr;
    ip_hdr.ver_hlen = 0x45; // Version and header length
    ip_hdr.tos      = 0x00;
    ip_hdr.len      = htons(fapi_len + 8 + 20); // FAPI + UDP 8 + IP 20
    ip_hdr.id       = htons(id_counter++);
    ip_hdr.flags    = 0x0040;
    ip_hdr.ttl      = 64;
    ip_hdr.protocol = 0x11; //UDP
    ip_hdr.checksum = 0x0996;
    uint32_t addr;
    inet_pton(AF_INET, src->ip, &addr);
    ip_hdr.src_ip = addr;
    inet_pton(AF_INET, dst->ip, &addr);
    ip_hdr.dst_ip = addr;
    if(fwrite(&ip_hdr, sizeof(ip_header_t), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        ret = -1;
    }

    // UDP Header: 8B
    udp_header_t udp_hdr;
    udp_hdr.src_port = htons(src->port);
    udp_hdr.dst_port = htons(dst->port);
    udp_hdr.len      = htons(fapi_len + 8); // FAPI + UDP 8
    udp_hdr.checksum = 0xd6db;
    if(fwrite(&udp_hdr, sizeof(udp_header_t), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        ret = -1;
    }
    return ret;
}

static int pcap_write_record(record_t* record, int max_data_size)
{
    int32_t msg_id  = record->flags & 0xFFFF;
    int32_t fapi_id = parse_fapi_id((uint8_t*)record->buf);
    if(msg_id != fapi_id)
    {
        NVLOGW(TAG, "%s: msg_id not match: msg_id=0x%02X fapi_id=0x%02X", __func__, msg_id, fapi_id);
    }

    // The FAPI payload length, included in FAPI header
    int32_t fapi_len = parse_fapi_length((uint8_t*)record->buf, max_data_size);
    int32_t record_packet_size = fapi_len + (record->data_len > max_data_size ? max_data_size : record->data_len);

    if(record_packet_size != record->buf_size)
    {

        NVLOGW(TAG, "%s: msg_len not match: msg_id=0x%02X record.buf_size=%d record.data_len=%d fapi_len=%d max_data_size=%d record_packet_size=%d",
               __func__, msg_id, record->buf_size, record->data_len, fapi_len, max_data_size, record_packet_size);
    }

    if(pcap_write_common_headers(record, record->buf_size) < 0)
    {
        return -1;
    }

    // Write FAPI payload
    if(fwrite(record->buf, record->buf_size, 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "Error: %s line %d - %s", __func__, __LINE__, strerror(errno));
        return -1;
    }
    else
    {
        NVLOGD(TAG, "%s: write OK. fapi_id=0x%02X record->buf_size=%d", __func__, fapi_id, record->buf_size);
        return 0;
    }
}

static int pcap_write_file_header()
{
    struct pcap_file_header file_header;
    file_header.magic         = 0xA1B2C3D4;
    file_header.version_major = PCAP_VERSION_MAJOR;
    file_header.version_minor = PCAP_VERSION_MINOR;
    file_header.thiszone      = 0;          /* gmt to local correction */
    file_header.sigfigs       = 0;          /* accuracy of timestamps */
    file_header.snaplen       = 0x00040000; /* max length saved portion of each pkt */
    file_header.linktype      = PCAP_DATA_LINK_TYPE;
    NVLOGI(TAG, "%s: write pcap_write_file_header size=%lu", __func__, sizeof(struct pcap_file_header));

    if(fwrite(&file_header, sizeof(struct pcap_file_header), 1, pcapfile) != 1)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: write pcap_write_file_header failed", __func__);
        return -1;
    }
    else
    {
        return 0;
    }
}

int pcap_file_open(const char* filename)
{
    // Create or lookup the semaphore
    pthread_mutex_init(&pcapmutex, NULL);
    char path[LOG_SHM_NAME_LEN * 2];
    snprintf(path, LOG_SHM_NAME_LEN * 2, "%s", filename);

    // Open a temperate file to store the logs
    if((pcapfile = fopen(path, "w")) == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: failed to open file %s", __func__, path);
        return -1;
    }
    pcap_write_file_header();
    NVLOGI(TAG, "%s: opened file %s for PCAP log", __func__, path);
    return 0;
}

#define MAX_SHMLOG_BUF_SIZE (3 * 1024 * 1024)

static int convert_pcap(FILE* record_file, FILE* pcap_file, long start, long end, int32_t max_msg_size, int32_t max_data_size)
{
    int       ret    = 0;
    record_t* record = malloc(MAX_SHMLOG_BUF_SIZE);
    if (record == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: malloc failed", __func__);
        return -1;
    }
    record->buf_size = 0;
    record->flags = 0;

    NVLOGC(TAG, "%s: start=0x%lX end=0x%lX", __func__, start, end);

    int32_t max_record_size = sizeof(record_t) + max_msg_size + max_data_size + 4;

    long pos_forward = start;
    while(pos_forward + sizeof(record_t) < end)
    {
        // Set file offset
        if(fseek(record_file, pos_forward, SEEK_SET) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fseek error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }

        // Read the record_t header
        if(fread(record, sizeof(record_t), 1, record_file) != 1)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fread header error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }
        NVLOGV(TAG, "%s: record: dir=%d msg_id=0x%X record->buf_size=%d", __func__, record->flags >> 16, record->flags & 0xFFFF, record->buf_size);

        if(pos_forward + record->buf_size + sizeof(record->buf_size) > end)
        {
            NVLOGI(TAG, "%s: The last record was overridden, skip pos_forward=0x%lX record->buf_size=0x%X", __func__, pos_forward, record->buf_size);
            break;
        }

        // Error check
        if(record->buf_size <= 0 || record->buf_size > max_record_size - sizeof(record_t) - 4)
        {
            NVLOGC(TAG, "%s: error record: pos_forward=0x%lX record->buf_size=0x%X", __func__, pos_forward, record->buf_size);
            ret = -1;
            break;
        }

        // Read the payload
        if(fread(record->buf, record->buf_size, 1, record_file) != 1)
        {
            NVLOGC(TAG, "%s: fread payload error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }

        int32_t buf_size = -1;
        if(fread(&buf_size, sizeof(record->buf_size), 1, record_file) != 1)
        {
            NVLOGC(TAG, "%s: fread size error: pos_forward=0X%lX err=%d - %s", __func__, pos_forward, errno, strerror(errno));
            ret = -1;
            break;
        }

        // For debug: error check
        if(buf_size != record->buf_size)
        {
            NVLOGC(TAG, "%s: record file format error: pos_forward=0x%lX record->buf_size=%d buf_size=%d", __func__, pos_forward, record->buf_size, buf_size);
            ret = -1;
            break;
        }

        // Write to pcap file
        pcap_write_record(record, max_data_size);

        // Check and move file offset
        int record_size = get_record_size(record);
        if(pos_forward + record_size + sizeof(record_t) >= end)
        {
            break;
        }
        else
        {
            pos_forward += record_size;
        }
    }

    if (ret < 0)
    {
        if (end - pos_forward <= max_record_size * 2)
        {
            // Ignore the last 2 records parsing failure
            ret = 0;
            NVLOGC(TAG, "%s: The last %lu bytes was not integrative. pos_forward=0x%lX record->buf_size=0x%X msg_id=0x%02X",
                    __func__, end - pos_forward, pos_forward, record->buf_size, record->flags & 0xFFFF);
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: convert error: remain_size=%lu pos_forward=0x%lX record->buf_size=0x%X msg_id=0x%02X",
                    __func__, end - pos_forward, pos_forward, record->buf_size, record->flags & 0xFFFF);
        }
    }

    free(record);
    fflush(pcapfile);
    NVLOGC(TAG, "%s: ret=%d start=0x%lX end=0x%lX converted_pos=0x%lX - %ld", __func__, ret, start, end, pos_forward, pos_forward);
    return ret;
}

int64_t nv_ipc_convert_pcap(FILE* record_file, char* pcap_filepath, long shm_cache_size, int32_t max_msg_size, int32_t max_data_size, long total_size, uint64_t break_offset)
{
    if(record_file == NULL || fseek(record_file, 0, SEEK_END) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: record_file error", __func__);
        return -1;
    }

    int64_t file_size = ftell(record_file);
    if (file_size < 0)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: record_file size error", __func__);
        return -1;
    }

    NVLOGC(TAG, "%s: fapi_type=%d shm_cache_size=0x%lX file_size=0x%lX total_size=0x%lX", __func__, fapi_type, shm_cache_size, file_size, total_size);

    int ret = 0;
    pcap_file_open(pcap_filepath);

    // Log rotation enabled
    // Convert the first SHM block
    long pcap_end = break_offset == 0 ? file_size : shm_cache_size;
    ret = convert_pcap(record_file, pcapfile, 0, pcap_end, max_msg_size, max_data_size);
    if(ret != 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: convert first SHM block failed shm_cache_size=0x%lX file_size=0x%lX", __func__, shm_cache_size, file_size);
    }

    int64_t pos_break = (break_offset & ((shm_cache_size -1) >> 1)) + shm_cache_size;

    NVLOGC(TAG, "%s: converted first block size=0x%lX=%ld MB file_size=0x%lX=%ld MB pos_break=0x%lX", __func__,
            shm_cache_size, shm_cache_size >> 20, file_size, file_size >> 20, pos_break);

    if (break_offset != 0)
    {
        // Some logs may have been overwritten, find the earliest record which hasn't been overwritten
        int64_t pos_backward = file_size;
        record_t record;
    #if 0 // Below code are for debug, do not delete
        while(pos_backward > shm_cache_size)
        {
            if(fseek(record_file, pos_backward - sizeof(record.buf_size), SEEK_SET) < 0)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fseek error: pos_backward=0x%lX err=%d - %s", __func__, pos_backward, errno, strerror(errno));
                ret = -1;
                break;
            }

            // Read the record size
            if(fread(&record.buf_size, sizeof(record.buf_size), 1, record_file) != 1)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: fread error: pos_backward=0x%lX err=%d - %s", __func__, pos_backward, errno, strerror(errno));
                ret = -1;
                break;
            }

            NVLOGD(TAG, "%s: buf_size=%ld - 0x%X pos_backward=0x%lX", __func__, record.buf_size, record.buf_size, pos_backward);

            if(record.buf_size <= 0)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: error buffer size: pos_backward=0x%lX=%u MB", __func__, pos_backward, pos_backward / 1024/1024);
                ret = -1;
                break;
            }

            // Move backward to previous record
            int record_size = get_record_size(&record);
            if(pos_backward - record_size < shm_cache_size + sizeof(record.buf_size))
            {
                break;
            }
            else
            {
                pos_backward -= record_size;
            }
        }

    #else
        pos_backward = pos_break;
    #endif

        NVLOGC(TAG, "%s: found shm_cache_size=%ld pos_backward=0x%lX break_offset=0x%lX file_size=%ld",
                __func__, shm_cache_size, pos_backward, break_offset, file_size);

        if (pos_backward != pos_break)
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: break_offset error: shm_cache_size=%ld pos_backward=0x%lX break_offset=0x%lX file_size=%ld",
                    __func__, shm_cache_size, pos_backward, break_offset, file_size);
        }

        if(ret == 0)
        {
            ret = convert_pcap(record_file, pcapfile, pos_backward, file_size, max_msg_size, max_data_size);
        }
    }
    int64_t pcap_size = ftell(pcapfile);
    fclose(pcapfile);
    return ret == 0 ? pcap_size : -1;
}

static int pcap_shmlogger_init(nv_ipc_debug_t *ipc_debug)
{
    if (ipc_debug->shmlogger != NULL)
    {
        // shmlogger already created, skip
        return 0;
    }

    char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    nvlog_safe_strncpy(name, ipc_debug->prefix, NV_NAME_MAX_LEN);
    strncat(name, "_pcap", NV_NAME_SUFFIX_MAX_LEN);
    shmlogger_config_t cfg;
    cfg.save_to_file = 1;                                                       // Start a background thread to save SHM cache to file before overflow
    cfg.shm_cache_size = (1L << ipc_debug->debug_configs.pcap_cache_size_bits); // 512MB, shared memory size, a SHM file will be created at /dev/shm/${prefix}_pcap
    cfg.max_file_size = (1L << ipc_debug->debug_configs.pcap_file_size_bits);   // 2GB Max file size, a disk file will be created at /var/log/aerial/${prefix}_pcap
    cfg.file_saving_core = ipc_debug->debug_configs.pcap_file_saving_cpu_core;  // CPU core ID for the background file saving if enabled.
    cfg.shm_caching_core = ipc_debug->debug_configs.pcap_shm_caching_cpu_core;  // CPU core ID for the background copying to shared memory if enabled.
    cfg.max_data_size = ipc_debug->debug_configs.pcap_max_data_size;
    cfg.max_msg_size = ipc_debug->debug_configs.pcap_max_msg_size;
    ipc_debug->shmlogger = shmlogger_open(ipc_debug->primary, name, &cfg);
    NVLOGC(TAG, "Open PCAP logger: prefix=%s shm_caching_core=%d file_saving_core=%d max_data_size=%d cache_size=%ldMB round_save_size=%luMB",
           name, cfg.shm_caching_core, cfg.file_saving_core, cfg.max_data_size, cfg.shm_cache_size >> 20, cfg.max_file_size >> 20);
    return 0;
}

void* pcap_shm_caching_thread_func(void* arg)
{
    nv_ipc_debug_t *ipc_debug = arg;
    nv_ipc_t *ipc = ipc_debug->ipc;

    if(ipc == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_INVALID_PARAM_EVENT, "%s: get nvipc instance failed", __func__);
        return NULL;
    }

    char thread_name[SHMLOG_NAME_MAX_LEN];
    int ret = snprintf(thread_name, 16, "%s_pcap_shm", ipc_debug->prefix); // Thread name length has to be <= 15 characters
    if (ret < 0) abort();
    if(pthread_setname_np(pthread_self(), thread_name) != 0)
    {
        NVLOGE_NO(TAG, AERIAL_THREAD_API_EVENT, "%s: set thread name %s failed", __func__, thread_name);
    }

    nv_set_sched_fifo_priority(90);
    if (ipc_debug->debug_configs.pcap_shm_caching_cpu_core >= 0)
    {
        nv_assign_thread_cpu_core(ipc_debug->debug_configs.pcap_shm_caching_cpu_core);
    }

    // Print thread name
    pthread_getname_np(pthread_self(), thread_name, 16);
    NVLOGC(TAG, "%s: thread [%s] started on CPU core [%d] ...", __func__, thread_name, sched_getcpu());

    // If enabled by YAML, create shmlogger for PCAP at initial. Otherwize create until received "sudo pcap start" command.
    if (ipc_debug->debug_configs.pcap_enable)
    {
        pcap_shmlogger_init(ipc_debug);
    }

    nv_ipc_msg_t msg;
    nvipc_cmd_t nvipc_cmd;

    while(1)
    {
        if (sem_wait(ipc_debug->debug_sem) < 0)
        {
            NVLOGE(TAG, AERIAL_SYSTEM_API_EVENT, "%s: sem_wait returned with errno=%d - %s", __func__, errno, strerror(errno));
        }

        if (ipc_debug->cmd_ring->dequeue(ipc_debug->cmd_ring, &nvipc_cmd) == 0)
        {
            NVLOGC(TAG, "%s: received nvipc command: %d", __func__, nvipc_cmd.cmd_id);

            switch (nvipc_cmd.cmd_id)
            {
            case IPC_CMD_PCAP_ENABLE:
                // Will be called at most one time during the process lifetime
                pcap_shmlogger_init(ipc_debug);
                ipc_debug->debug_configs.pcap_enable = 1; // Enable in cuphycontroller process
                ipc_debug->shm_data->primary_configs.debug_configs.pcap_enable = 1; // Enable in shared memory config
                nvipc_fw_start(ipc, 0);
                break;
            case IPC_CMD_PCAP_CONFIG_MSG_FILTER:
                memcpy(ipc_debug->debug_configs.msg_filter, ipc_debug->shm_data->primary_configs.debug_configs.msg_filter, NVIPC_MAX_MSG_ID * sizeof(uint8_t));
                NVLOGC(TAG, "%s: [%s] loaded new msg_filter", __func__, ipc_debug->prefix);
                break;
            case IPC_CMD_PCAP_CONFIG_CELL_FILTER:
                memcpy(ipc_debug->debug_configs.cell_filter, ipc_debug->shm_data->primary_configs.debug_configs.cell_filter, NVIPC_MAX_MSG_ID * sizeof(uint8_t));
                NVLOGC(TAG, "%s: [%s] loaded new cell_filter", __func__, ipc_debug->prefix);
                break;
            case IPC_CMD_PCAP_CLEAN:
                NVLOGC(TAG, "%s: [%s] stop and reset the shm logger for pcap", __func__, ipc_debug->prefix);
                // Stop and reset PCAP capturing
                nvipc_fw_reset(ipc);

                // Drop the cached messages if exists
                while (nvipc_fw_dequeue(ipc, &msg) >= 0)
                {
                    // Free the IPC buffers
                    nvipc_fw_free(ipc, &msg);
                }

                // Reset SHM logger
                shmlogger_reset(ipc_debug->shmlogger);
                break;
            default:
                NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s: [%s] unsupported nvipc command: %d", __func__, ipc_debug->prefix, nvipc_cmd.cmd_id);
                break;
            }
        }

        while(nvipc_fw_dequeue(ipc, &msg) >= 0)
        {
            // IPC message dequeued from the fw_ring queue
            NVLOGD(TAG, "Forwarder: dequeue msg_id=0x%02X", msg.msg_id);

            nvipc_msg_dir_t direction = nv_ipc_get_msg_direction(ipc, &msg);

            if (ipc_debug->debug_configs.cell_filter[msg.cell_id] == 0 || ipc_debug->debug_configs.msg_filter[msg.msg_id] == 0)
            {
                // Skip filtered out messages
            }
            else if (direction == NVIPC_PRIMARY_LOOPBACK)
            {
                // Skip the loop-back sending of SLOT.indication
            }
            else
            {
                // Asynchronized saving PCAP log in background thread
                save_ipc_msg(ipc_debug, &msg, (direction << 16) | msg.msg_id);
            }

            // Free the IPC buffers
            nvipc_fw_free(ipc, &msg);
        }
    }

    // nvipc_fw_stop(ipc);

    return NULL;
}

int nv_ipc_dump_config(nv_ipc_config_t *cfg)
{
    if (cfg == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid parameter", __func__);
        return -1;
    }

    char *prefix;
    nv_ipc_mempool_size_t *mempool;

    if (cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        nv_ipc_config_shm_t *transp_cfg = &cfg->transport_config.shm;
        prefix = transp_cfg->prefix;
        mempool = transp_cfg->mempool_size;
        NVLOGC(TAG, "[%s]: transport=%d ring_len=%d cuda_device_id=%d",
                prefix, cfg->ipc_transport, transp_cfg->ring_len, transp_cfg->cuda_device_id);
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        nv_ipc_config_dpdk_t *transp_cfg = &cfg->transport_config.dpdk;
        prefix = transp_cfg->prefix;
        mempool = transp_cfg->mempool_size;
        NVLOGC(TAG, "[%s]: transport=%d cuda_device_id=%d", prefix, cfg->ipc_transport, transp_cfg->cuda_device_id);
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        nv_ipc_config_doca_t *transp_cfg = &cfg->transport_config.doca;
        prefix = transp_cfg->prefix;
        mempool = transp_cfg->mempool_size;
        NVLOGC(TAG, "[%s]: transport=%d cuda_device_id=%d", prefix, cfg->ipc_transport, transp_cfg->cuda_device_id);
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: unsupported transport type: %d", __func__, cfg->ipc_transport);
        return -1;
    }

    for (int pool_id = 0; pool_id < NV_IPC_MEMPOOL_NUM; pool_id++)
    {
        NVLOGC(TAG, "[%s]: memory pool %d buf_size=%d pool_len=%d", prefix, pool_id, mempool->buf_size, mempool->pool_len);
        mempool++;
    }

    nv_ipc_debug_config_t* dbg = &cfg->debug_configs;
    NVLOGC(TAG, "========== Dump PCAP configs ======================");
    NVLOGC(TAG, "[%s]: config grpc_forward=%d", prefix, dbg->grpc_forward);
    NVLOGC(TAG, "[%s]: config debug_timing=%d", prefix, dbg->debug_timing);
    NVLOGC(TAG, "[%s]: config pcap_enable=%d", prefix, dbg->pcap_enable);
    NVLOGC(TAG, "[%s]: config pcap_shm_caching_cpu_core=%d", prefix, dbg->pcap_shm_caching_cpu_core);
    NVLOGC(TAG, "[%s]: config pcap_file_saving_cpu_core=%d", prefix, dbg->pcap_file_saving_cpu_core);

    if (dbg->pcap_cache_size_bits > 20)
    {
        NVLOGC(TAG, "[%s]: config pcap_cache_size_bits=%d size=%uMB", prefix, dbg->pcap_cache_size_bits, 1U << (dbg->pcap_cache_size_bits - 20));
    }
    else if (dbg->pcap_cache_size_bits > 10)
    {
        NVLOGC(TAG, "[%s]: config pcap_cache_size_bits=%d size=%uKB", prefix, dbg->pcap_cache_size_bits, 1U << (dbg->pcap_cache_size_bits - 10));
    }
    else
    {
        NVLOGC(TAG, "[%s]: config pcap_cache_size_bits=%d size=%uKB", prefix, dbg->pcap_cache_size_bits, 1U << dbg->pcap_cache_size_bits);
    }

    if (dbg->pcap_file_size_bits > 20)
    {
        NVLOGC(TAG, "[%s]: config pcap_file_size_bits=%d size=%uMB", prefix, dbg->pcap_file_size_bits, 1U << (dbg->pcap_file_size_bits - 20));
    }
    else if (dbg->pcap_file_size_bits > 10)
    {
        NVLOGC(TAG, "[%s]: config pcap_file_size_bits=%d size=%uKB", prefix, dbg->pcap_file_size_bits, 1U << (dbg->pcap_file_size_bits - 10));
    }
    else
    {
        NVLOGC(TAG, "[%s]: config pcap_file_size_bits=%d size=%uKB", prefix, dbg->pcap_file_size_bits, 1U << dbg->pcap_file_size_bits);
    }

    char tmp_buf[NVIPC_MAX_MSG_ID * 5 + 1];

    // Check whether all are enabled
    int counter = 0;
    int offset = 0;
    for (int i = 0; i < NVIPC_MAX_MSG_ID; i++)
    {
        if (dbg->msg_filter[i] != 0)
        {
            offset += snprintf(tmp_buf + offset, 6, "0x%02X ", i);
            counter++;
        }
    }
    tmp_buf[offset - 1] = '\0';
    if (counter == NVIPC_MAX_MSG_ID)
    {
        snprintf(tmp_buf, 16, "all are enabled");
    }
    NVLOGC(TAG, "[%s]: msg_filter[%d]: %s", prefix, counter, tmp_buf);

    counter = 0;
    offset = 0;
    for (int i = 0; i < NVIPC_MAX_CELL_ID; i++)
    {
        if (dbg->cell_filter[i] != 0)
        {
            offset += snprintf(tmp_buf + offset, 5, "%d ", i);
            counter++;
        }
    }
    tmp_buf[offset - 1] = '\0';
    if (counter == NVIPC_MAX_MSG_ID)
    {
        snprintf(tmp_buf, 16, "all are enabled");
    }
    NVLOGC(TAG, "[%s]: cell_filter[%d]: %s", prefix, counter, tmp_buf);

    NVLOGC(TAG, "========== Dump captured packet number ============");
    shmlogger_t *logger = NULL;
    if (dbg->pcap_enable)
    {
        shmlogger_config_t config;
        char logger_name[SHMLOG_NAME_MAX_LEN * 2];
        snprintf(logger_name, SHMLOG_NAME_MAX_LEN * 2, "%s_%s", prefix, "pcap");
        if ((logger = shmlogger_open(0, logger_name, &config)) == NULL)
        {
            NVLOGC(TAG, "%s: no /dev/shm/%s_pcap, logger may have been created properly", __func__, prefix);
        }
    }
    NVLOGC(TAG, "[%s]: captured_num=%lu", prefix, shmlogger_get_packet_count(logger));

    return 0;
}

int nv_ipc_lookup_config(nv_ipc_config_t *cfg, const char *prefix, nv_ipc_module_t module_type)
{
    if (cfg == NULL || prefix == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: invalid parameter", __func__);
        return -1;
    }

    char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    nvlog_safe_strncpy(name, prefix, NV_NAME_MAX_LEN);
    strncat(name, SHM_DEBUG_SUFFIX, NV_NAME_SUFFIX_MAX_LEN);

    nv_ipc_shm_t *shmpool = nv_ipc_shm_open(0, name, 0);
    if (shmpool == NULL)
    {
        return -1;
    }

    debug_shm_data_t *debug_shm_data = (debug_shm_data_t *)shmpool->get_mapped_addr(shmpool);
    if (debug_shm_data == NULL)
    {
        shmpool->close(shmpool);
        return -1;
    }

    memcpy(cfg, &debug_shm_data->primary_configs, sizeof(nv_ipc_config_t));

    cfg->module_type = module_type;

    // Set "primary" to 0 by default
    int ret = 0;
    if (cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
    {
        // No need to configure primary for SHM IPC
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DPDK)
    {
        cfg->transport_config.dpdk.primary = 0;
    }
    else if (cfg->ipc_transport == NV_IPC_TRANSPORT_DOCA)
    {
        cfg->transport_config.doca.primary = 0;
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unknown transport type %d", __func__, cfg->ipc_transport);
        ret = -1;
    }

    if (shmpool->close(shmpool) < 0)
    {
        ret = -1;
    }

    return ret;
}

static int ipc_debug_open(nv_ipc_debug_t* ipc_debug, nv_ipc_config_t* ipc_cfg)
{
    char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
    nvlog_safe_strncpy(name, ipc_debug->prefix, NV_NAME_MAX_LEN);
    strncat(name, SHM_DEBUG_SUFFIX, NV_NAME_SUFFIX_MAX_LEN);

    // For primary app, calculate total required shared memory size with configured pool_len
    // For secondary app, the total_shm_size passed to nv_ipc_shm_open() doesn't matter
    ipc_debug->msg_pool_len = ipc_debug->primary ? nv_ipc_get_pool_len(ipc_cfg, NV_IPC_MEMPOOL_CPU_MSG) : 0;
    size_t shm_data_size    = sizeof(debug_shm_data_t);
    size_t msg_timing_size  = sizeof(msg_timing_t) * ipc_debug->msg_pool_len;
    size_t sync_timing_size = sizeof(sync_timing_t) * ipc_debug->msg_pool_len;
    size_t total_shm_size   = shm_data_size + msg_timing_size + sync_timing_size * 2;

    int shm_primary = ipc_debug->transport == NV_IPC_TRANSPORT_SHM ? ipc_debug->primary : 1;
    if((ipc_debug->shmpool = nv_ipc_shm_open(shm_primary, name, total_shm_size)) == NULL)
    {
        return -1;
    }

    int8_t* shm_addr = ipc_debug->shmpool->get_mapped_addr(ipc_debug->shmpool);
    ipc_debug->shm_data = (debug_shm_data_t*)shm_addr;

    if(shm_primary)
    {
        memset(shm_addr, 0, total_shm_size);

        // Save debug_configs to shared memory by primary process
        memcpy(&ipc_debug->shm_data->primary_configs, ipc_cfg, sizeof(nv_ipc_config_t));

        // Write NVIPC lib version by primary app
        ipc_debug->shm_data->primary_configs.debug_configs.nvipc_version = LIBRARY_VERSION;

        // Overwrite pcap_enable yaml config if exported NVIPC_DEBUG_EN
        if (get_env_long(ENV_NVIPC_DEBUG_PCAP, CONFIG_DEBUG_PCAP))
        {
            ipc_debug->shm_data->primary_configs.debug_configs.pcap_enable = 1;
        }
    }
    else
    {
        if (ipc_cfg->ipc_transport == NV_IPC_TRANSPORT_SHM)
        {
            memcpy(ipc_cfg, &ipc_debug->shm_data->primary_configs, sizeof(nv_ipc_config_t));
        }
        else
        {
            NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s not supported transport type %d", __func__, ipc_cfg->ipc_transport);
            return -1;
        }

        // Read msg_pool_len from shared memory and update shared memory size calculation
        ipc_debug->msg_pool_len = nv_ipc_get_pool_len(&ipc_debug->shm_data->primary_configs, NV_IPC_MEMPOOL_CPU_MSG);
        msg_timing_size  = sizeof(msg_timing_t) * ipc_debug->msg_pool_len;
        sync_timing_size = sizeof(sync_timing_t) * ipc_debug->msg_pool_len;
        total_shm_size   = shm_data_size + msg_timing_size + sync_timing_size * 2;
    }

    shm_addr += shm_data_size;
    ipc_debug->msg_timing = (msg_timing_t*)shm_addr;
    shm_addr += msg_timing_size;

    ipc_debug->sync_timing_m2s = (sync_timing_t*)shm_addr;
    shm_addr += sync_timing_size;

    ipc_debug->sync_timing_s2m = (sync_timing_t*)shm_addr;
    shm_addr += sync_timing_size;

    // Copy debug_configs from shared memory, for both primary and secondary app
    memcpy(&ipc_debug->debug_configs, &ipc_debug->shm_data->primary_configs.debug_configs, sizeof(nv_ipc_debug_config_t));

    // Check nvipc version for secondary app
    if (ipc_debug->debug_configs.nvipc_version != LIBRARY_VERSION)
    {
        int32_t libver = ipc_debug->debug_configs.nvipc_version;
        NVLOGF(TAG, AERIAL_NVIPC_API_EVENT, "NVIPC lib version doesn't match: expected_version=%d.%d.%d current_version=%d.%d.%d",
               libver / 100, libver / 10 % 10, libver % 10, LIBRARY_VERSION / 100, LIBRARY_VERSION / 10 % 10, LIBRARY_VERSION % 10);
    }

    // Get frequently used pointers to improve performance
    ipc_debug->primary_debug_configs = &ipc_debug->shm_data->primary_configs.debug_configs;
    ipc_debug->cell_filters = ipc_debug->shm_data->primary_configs.debug_configs.cell_filter;
    ipc_debug->msg_filters  = ipc_debug->shm_data->primary_configs.debug_configs.msg_filter;

    if(ipc_debug->debug_configs.debug_timing)
    {
        if(ipc_debug->primary)
        {
            ipc_debug->stat_msg_build     = stat_log_open("DL_MSG_BUILD", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_msg_transport = stat_log_open("DL_MSG_TRANSPORT", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_msg_handle    = stat_log_open("DL_MSG_HANDLE", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_msg_total     = stat_log_open("DL_MSG_TOTAL", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_wait_delay    = stat_log_open("DL_WAIT_DELAY", STAT_MODE_COUNTER, 10 * 1000);
            ipc_debug->stat_post_interval = stat_log_open("UL_POST_INTERVAL", STAT_MODE_COUNTER, 10 * 1000);
        }
        else
        {
            ipc_debug->stat_msg_build     = stat_log_open("UL_MSG_BUILD", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_msg_transport = stat_log_open("UL_MSG_TRANSPORT", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_msg_handle    = stat_log_open("UL_MSG_HANDLE", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_msg_total     = stat_log_open("UL_MSG_TOTAL", STAT_MODE_COUNTER, 50 * 1000);
            ipc_debug->stat_wait_delay    = stat_log_open("UL_WAIT_DELAY", STAT_MODE_COUNTER, 10 * 1000);
            ipc_debug->stat_post_interval = stat_log_open("DL_POST_INTERVAL", STAT_MODE_COUNTER, 10 * 1000);
        }
        //    ipc_debug->stat_msg_build->set_limit(ipc_debug->stat_msg_build, 0, 600 * 1000);
        //    ipc_debug->stat_msg_ipc->set_limit(ipc_debug->stat_msg_ipc, 0, 600 * 1000);
        //    ipc_debug->stat_msg_handle->set_limit(ipc_debug->stat_msg_handle, 0, 600 * 1000);
        //    ipc_debug->stat_msg_total->set_limit(ipc_debug->stat_msg_total, 0, 1200 * 1000);
    }

    nvlog_safe_strncpy(name, ipc_debug->prefix, NV_NAME_MAX_LEN);
    strncat(name, "_dbg_sem", NV_NAME_SUFFIX_MAX_LEN);
    if ((ipc_debug->debug_sem = sem_open(name, O_CREAT, 0600, 0)) == SEM_FAILED)
    {
        NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "sem_open failed: name=%s", name);
        return -1;
    }

    nvlog_safe_strncpy(name, ipc_debug->prefix, NV_NAME_MAX_LEN);
    strncat(name, "_cmd_ring", NV_NAME_SUFFIX_MAX_LEN);
    if (ipc_debug->primary)
    {
        NVLOGI(TAG, "Semaphore create: %s", name);
        sem_init(ipc_debug->debug_sem, 1, 0); // Initiate the semaphore to be shared and set value to 0

        ipc_debug->cmd_ring = nv_ipc_ring_open(RING_TYPE_SHM_PRIMARY, name, 64, sizeof(nvipc_cmd_t));

        // Create the background thread for receiving dynamic PCAP and other commands
        if(pthread_create(&ipc_debug->debug_thread_id, NULL, pcap_shm_caching_thread_func, ipc_debug) != 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "pthread_create failed: name=%s", name);
        }
    }
    else
    {
        NVLOGI(TAG, "Semaphore lookup: %s", name);
        ipc_debug->cmd_ring = nv_ipc_ring_open(RING_TYPE_SHM_SECONDARY, name, 64, sizeof(nvipc_cmd_t));
    }

    return 0;
}

static int ipc_debug_close(nv_ipc_debug_t* ipc_debug)
{
    int ret = 0;

    // Stop and join the background thread if it was created (primary only)
    if(ipc_debug->primary && ipc_debug->debug_thread_id != 0)
    {
        // Cancel the thread
        pthread_cancel(ipc_debug->debug_thread_id);
        // Wait for it to finish
        pthread_join(ipc_debug->debug_thread_id, NULL);
        ipc_debug->debug_thread_id = 0;
    }

    // Close cmd_ring
    if(ipc_debug->cmd_ring != NULL && ipc_debug->cmd_ring->close != NULL)
    {
        if(ipc_debug->cmd_ring->close(ipc_debug->cmd_ring) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: close cmd_ring failed", __func__);
            ret = -1;
        }
        ipc_debug->cmd_ring = NULL;
    }

    // Close debug semaphore
    if(ipc_debug->debug_sem != NULL && ipc_debug->debug_sem != SEM_FAILED)
    {
        char name[NV_NAME_MAX_LEN + NV_NAME_SUFFIX_MAX_LEN];
        nvlog_safe_strncpy(name, ipc_debug->prefix, NV_NAME_MAX_LEN);
        strncat(name, "_dbg_sem", NV_NAME_SUFFIX_MAX_LEN);

        if(sem_close(ipc_debug->debug_sem) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: sem_close failed: %s", __func__, strerror(errno));
            ret = -1;
        }

        // Unlink the semaphore if primary
        if(ipc_debug->primary)
        {
            if(sem_unlink(name) < 0 && errno != ENOENT)
            {
                NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: sem_unlink failed: %s", __func__, strerror(errno));
                ret = -1;
            }
        }
        ipc_debug->debug_sem = NULL;
    }

    // Close stat_log objects if they were opened
    if(ipc_debug->debug_configs.debug_timing)
    {
        if(ipc_debug->stat_msg_build != NULL && ipc_debug->stat_msg_build->close != NULL)
        {
            ipc_debug->stat_msg_build->close(ipc_debug->stat_msg_build);
            ipc_debug->stat_msg_build = NULL;
        }
        if(ipc_debug->stat_msg_transport != NULL && ipc_debug->stat_msg_transport->close != NULL)
        {
            ipc_debug->stat_msg_transport->close(ipc_debug->stat_msg_transport);
            ipc_debug->stat_msg_transport = NULL;
        }
        if(ipc_debug->stat_msg_handle != NULL && ipc_debug->stat_msg_handle->close != NULL)
        {
            ipc_debug->stat_msg_handle->close(ipc_debug->stat_msg_handle);
            ipc_debug->stat_msg_handle = NULL;
        }
        if(ipc_debug->stat_msg_total != NULL && ipc_debug->stat_msg_total->close != NULL)
        {
            ipc_debug->stat_msg_total->close(ipc_debug->stat_msg_total);
            ipc_debug->stat_msg_total = NULL;
        }
        if(ipc_debug->stat_wait_delay != NULL && ipc_debug->stat_wait_delay->close != NULL)
        {
            ipc_debug->stat_wait_delay->close(ipc_debug->stat_wait_delay);
            ipc_debug->stat_wait_delay = NULL;
        }
        if(ipc_debug->stat_post_interval != NULL && ipc_debug->stat_post_interval->close != NULL)
        {
            ipc_debug->stat_post_interval->close(ipc_debug->stat_post_interval);
            ipc_debug->stat_post_interval = NULL;
        }
    }

    // Close SHM pool
    if(ipc_debug->shmpool != NULL)
    {
        if(ipc_debug->shmpool->close(ipc_debug->shmpool) < 0)
        {
            NVLOGE_NO(TAG, AERIAL_SYSTEM_API_EVENT, "%s: close SHM pool failed", __func__);
            ret = -1;
        }
        ipc_debug->shmpool = NULL;
    }

    free(ipc_debug);

    if(ret == 0)
    {
        NVLOGI(TAG, "%s: OK", __func__);
    }
    return 0;
}

static int ipc_debug_post_hook(nv_ipc_debug_t* ipc_debug)
{
    if(ipc_debug->debug_configs.debug_timing == 0)
    {
        return 0;
    }

    unsigned long  counter;
    sync_timing_t* sync_timing;
    if(ipc_debug->primary)
    {
        counter     = atomic_fetch_add(&ipc_debug->shm_data->post_counter_m2s, 1);
        sync_timing = ipc_debug->sync_timing_m2s + counter % ipc_debug->msg_pool_len;
    }
    else
    {
        counter     = atomic_fetch_add(&ipc_debug->shm_data->post_counter_s2m, 1);
        sync_timing = ipc_debug->sync_timing_s2m + counter % ipc_debug->msg_pool_len;
    }
    debug_get_timestamp(&sync_timing->ts_post);

    if(counter > 0)
    {
        sync_timing_t* sync_timing_last;
        if(counter % ipc_debug->msg_pool_len == 0)
        {
            sync_timing_last = sync_timing + ipc_debug->msg_pool_len - 1;
        }
        else
        {
            sync_timing_last = sync_timing - 1;
        }
        long tti_interval = debug_get_ts_interval(&sync_timing_last->ts_post, &sync_timing->ts_post);
        ipc_debug->stat_post_interval->add(ipc_debug->stat_post_interval, tti_interval);
    }
    return 0;
}

static int ipc_debug_wait_hook(nv_ipc_debug_t* ipc_debug)
{
    if(ipc_debug->debug_configs.debug_timing == 0)
    {
        return 0;
    }

    sync_timing_t* sync_timing;
    if(ipc_debug->primary)
    {
        unsigned long counter = atomic_fetch_add(&ipc_debug->shm_data->wait_counter_s2m, 1) % ipc_debug->msg_pool_len;
        sync_timing           = ipc_debug->sync_timing_s2m + counter;
    }
    else
    {
        unsigned long counter = atomic_fetch_add(&ipc_debug->shm_data->wait_counter_m2s, 1) % ipc_debug->msg_pool_len;
        sync_timing           = ipc_debug->sync_timing_m2s + counter;
    }
    debug_get_timestamp(&sync_timing->ts_wait);
    long sync_delay = debug_get_ts_interval(&sync_timing->ts_post, &sync_timing->ts_wait);
    ipc_debug->stat_wait_delay->add(ipc_debug->stat_wait_delay, sync_delay);
    return 0;
}

static int ipc_debug_alloc_hook(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index)
{
    if(CONFIG_LOG_ALLOCATE_TIME || ipc_debug->debug_configs.debug_timing)
    {
        msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
        debug_get_timestamp(&msg_timing->ts_alloc);
    }
    return 0;
}

sfn_slot_t nv_ipc_get_sfn_slot(nv_ipc_msg_t* msg)
{
    if(fapi_type == SCF_FAPI && msg->msg_id > SCF_FAPI_RESV_1_END)
    {
        scf_fapi_slot_header_t* header = msg->msg_buf;
        return header->sfn_slot;
    }
    else
    {
        return sfn_slot_invalid;
    }
}

void nv_ipc_set_handle_id(nv_ipc_msg_t* msg, uint8_t handle_id)
{
    if(fapi_type == SCF_FAPI)
    {
        scf_fapi_slot_header_t* header = msg->msg_buf;
        header->head.handle_id = handle_id;
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: unsupported fapi_type %d", __func__, fapi_type);
    }
}

static int ipc_debug_free_hook(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index)
{
    if(ipc_debug->debug_configs.debug_timing == 0)
    {
        return 0;
    }

    msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
    debug_get_timestamp(&msg_timing->ts_free);

    int64_t build_time     = debug_get_ts_interval(&msg_timing->ts_alloc, &msg_timing->ts_send);
    int64_t transport_time = debug_get_ts_interval(&msg_timing->ts_send, &msg_timing->ts_recv);
    int64_t handle_time    = debug_get_ts_interval(&msg_timing->ts_recv, &msg_timing->ts_free);
    int64_t total_time     = debug_get_ts_interval(&msg_timing->ts_alloc, &msg_timing->ts_free);

    sfn_slot_t ss = nv_ipc_get_sfn_slot(msg);
    NVLOGI(TAG, "SFN %d.%d IPC free FAPI=0x%02X Timing: build=%ld transport=%ld handle=%ld total=%ld", ss.u16.sfn, ss.u16.slot, msg->msg_id, build_time, transport_time, handle_time, total_time);

    int ret = 0;
    if(ipc_debug->stat_msg_build->add(ipc_debug->stat_msg_build, build_time) != 0)
    {
        ret |= 1;
    }
    if(ipc_debug->stat_msg_transport->add(ipc_debug->stat_msg_transport, transport_time) != 0)
    {
        ret |= 2;
    }
    if(ipc_debug->stat_msg_handle->add(ipc_debug->stat_msg_handle, handle_time) != 0)
    {
        ret |= 4;
    }
    if(ipc_debug->stat_msg_total->add(ipc_debug->stat_msg_total, total_time) != 0)
    {
        ret |= 8;
    }

    // Clear all
    memset(msg_timing, 0, sizeof(msg_timing_t));
    return ret;
}

void verify_msg(nv_ipc_t* ipc, nv_ipc_msg_t *msg, const char *info)
{
    scf_fapi_header_t *scf_fapi = (scf_fapi_header_t *)msg->msg_buf;
    scf_fapi_body_header_t *body = (scf_fapi_body_header_t *)scf_fapi->payload;
    scf_fapi_sfn_slot_t* ss = (scf_fapi_sfn_slot_t*)(body->next);
    int fapi_id = body->type_id;
    int32_t fapi_len = sizeof(scf_fapi_header_t) + sizeof(scf_fapi_body_header_t) + body->length;

    int32_t msg_index = get_msg_index(ipc, msg);

    if (msg->msg_id != fapi_id || msg->msg_len != fapi_len)
    {
        NVLOGW(TAG, "SFN %u.%u %s: not match: msg_index=%d msg_id=0x%02X fapi_id=0x%02X msg_len=%d fapi_len=%d", ss->sfn, ss->slot, info, msg_index, msg->msg_id, fapi_id, msg->msg_len, fapi_len);
    }
    else
    {
        NVLOGI(TAG, "SFN %u.%u %s: matched: msg_index=%d msg_id=0x%02X fapi_id=0x%02X msg_len=%d fapi_len=%d", ss->sfn, ss->slot, info, msg_index, msg->msg_id, fapi_id, msg->msg_len, fapi_len);
    }
}

static void save_ipc_msg(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t flags)
{
    int     ret = 0;
    void*   fapi_msg;
    int32_t ipc_len;

    // verify_msg(ipc_debug->ipc, msg, "SAVE");

    if(ipc_debug->debug_configs.fapi_tb_loc == 0)
    {
        NVLOGC(TAG, "%s: fapi_tb_loc=%d, please confirm", __func__, ipc_debug->debug_configs.fapi_tb_loc);
        if(msg->data_pool == NV_IPC_MEMPOOL_CPU_DATA || msg->data_pool == NV_IPC_MEMPOOL_CPU_LARGE)
        {
            fapi_msg = msg->data_buf;
            ipc_len  = msg->data_len;
        }
        else
        {
            fapi_msg = msg->msg_buf;
            ipc_len  = msg->msg_len;
        }
        int32_t fapi_len = parse_fapi_length(fapi_msg, ipc_debug->debug_configs.pcap_max_data_size);
        shmlogger_save_buffer(ipc_debug->shmlogger, (const char*)fapi_msg, fapi_len, flags);
#if 0 // Do not call to improve performance
        int32_t fapi_id = parse_fapi_id(fapi_msg);
        if(fapi_id != msg->msg_id)
        {
            NVLOGI(TAG, "%s: fapi_id not match: msg_id=0x%02X fapi_id=0x%02X", __func__, msg->msg_id, fapi_id);
        }

        if(fapi_len != ipc_len)
        {
            NVLOGI(TAG, "%s: fapi_len not match: msg_id=0x%02X fapi_id=0x%02X fapi_len=%d ipc_len=%d", __func__, msg->msg_id, fapi_id, fapi_len, ipc_len);
        }
#endif
    }
    else if(ipc_debug->debug_configs.fapi_tb_loc == 1)
    {
        shmlogger_save_ipc_msg(ipc_debug->shmlogger, msg, flags, ipc_debug->debug_configs.pcap_max_data_size);
        return;
    }
    else
    {
        NVLOGI(TAG, "%s: fapi_tb_loc=%d is not supported", __func__, ipc_debug->debug_configs.fapi_tb_loc);
        return;
    }
}

int64_t nv_ipc_get_buffer_ts_send(nv_ipc_debug_t* ipc_debug, int32_t buf_index)
{
    msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
    return msg_timing->ts_send.tv_sec * 1000000000LL + msg_timing->ts_send.tv_nsec;
}

static int ipc_debug_send_hook(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index)
{
    int ret = 0;
    if(CONFIG_LOG_SEND_TIME || ipc_debug->debug_configs.debug_timing)
    {
        msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
        debug_get_timestamp(&msg_timing->ts_send);
    }

    // Check forward enable status from atomic variable in shared memory for both primary and secondary apps
    if (atomic_load(ipc_debug->p_forward_started) == 0)
    {
        return 0;
    }

    if (ipc_debug->packet_infos[buf_index].direction == NVIPC_PRIMARY_LOOPBACK)
    {
        // Skip the loop-back sending of SLOT.indication
        return 0;
    }

    // Read PCAP filter configs from shared memory
    if (ipc_debug->cell_filters[msg->cell_id] == 0 || ipc_debug->msg_filters[msg->msg_id] == 0)
    {
        // Skip filtered out messages
        return 0;
    }

    // Capture PCAP packet
    if(ipc_debug->primary_debug_configs->pcap_sync_save)
    {
        // PCAP synchronized mode saves PCAP packet only in primary process, in both send and receive
        if (ipc_debug->primary)
        {
            save_ipc_msg(ipc_debug, msg, (NVIPC_PRIMARY_TO_SECONDARY << 16) | msg->msg_id);
        }
    }
    else
    {
        // For asynchronized mode, forward the packet in both primary and secondary apps, only in send function
        nvipc_fw_enqueue(ipc_debug->ipc, msg);
    }
    return 0;
}

static int ipc_debug_recv_hook(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index)
{
    if(ipc_debug->debug_configs.debug_timing)
    {
        msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
        debug_get_timestamp(&msg_timing->ts_recv);
    }

    // PCAP synchronized mode saves PCAP packet only in primary process, in both send and receive
    if(ipc_debug->primary &&ipc_debug->primary_debug_configs->pcap_sync_save)
    {
        // Check forward enable status from atomic variable in shared memory for both primary app in synchronized save mode
        if (atomic_load(ipc_debug->p_forward_started) == 0)
        {
            return 0;
        }

        if (ipc_debug->cell_filters[msg->cell_id] == 0 || ipc_debug->msg_filters[msg->msg_id] == 0)
        {
            // Skip filtered out messages
            return 0;
        }

        if (ipc_debug->packet_infos[buf_index].direction == NVIPC_SECONDARY_LOOPBACK)
        {
            // Skip the loop-back receiving of SLOT.indication
            return 0;
        }

        save_ipc_msg(ipc_debug, msg, (NVIPC_SECONDARY_TO_PRIMARY << 16) | msg->msg_id);
    }
    return 0;
}

static int ipc_debug_fw_deq_hook(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index)
{
    if(ipc_debug->debug_configs.debug_timing)
    {
        msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
        debug_get_timestamp(&msg_timing->ts_fw_deq);
    }
    return 0;
}

static int ipc_debug_fw_free_hook(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index)
{
    if(ipc_debug->debug_configs.debug_timing)
    {
        msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
        debug_get_timestamp(&msg_timing->ts_fw_free);
    }
    return 0;
}

int nv_ipc_dump_msg(nv_ipc_debug_t* ipc_debug, nv_ipc_msg_t* msg, int32_t buf_index, const char* info)
{
    char log_buf[1024];
    int  offset = snprintf(log_buf, 128, "%s:", info);

    if(fapi_type == SCF_FAPI)
    {
        sfn_slot_t ss = nv_ipc_get_sfn_slot(msg);
        offset += snprintf(log_buf + offset, 64, " SFN %d.%d", ss.u16.sfn, ss.u16.slot);
    }

    offset += snprintf(log_buf + offset, 64, " cell_id=%d msg_id=0x%02X buf_id=%d", msg->cell_id, msg->msg_id, buf_index);

    if(ipc_debug != NULL)
    {
        msg_timing_t* msg_timing = ipc_debug->msg_timing + buf_index;
        if(msg_timing->ts_alloc.tv_sec != 0)
        {
            offset += snprintf(log_buf + offset, 64, " Allocate: ");

            struct tm _tm;
            if(localtime_r(&msg_timing->ts_alloc.tv_sec, &_tm) != NULL)
            {
                // size +19
                offset += strftime(log_buf + offset, sizeof("1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S", &_tm);
#ifdef DEBUG_HIGH_RESOLUTION_TIME
                // size +10
                offset += snprintf(log_buf + offset, 11, ".%09ld", msg_timing->ts_alloc.tv_nsec);
#else
                // size +7
                offset += snprintf(log_buf + offset, 8, ".%06ld", msg_timing->ts_alloc.tv_usec);
#endif
            }

            int64_t build_time     = msg_timing->ts_send.tv_sec == 0 ? -1 : debug_get_ts_interval(&msg_timing->ts_alloc, &msg_timing->ts_send);
            int64_t transport_time = msg_timing->ts_recv.tv_sec == 0 ? -1 : debug_get_ts_interval(&msg_timing->ts_send, &msg_timing->ts_recv);
            int64_t handle_time    = msg_timing->ts_free.tv_sec == 0 ? -1 : debug_get_ts_interval(&msg_timing->ts_recv, &msg_timing->ts_free);
            int64_t total_time     = msg_timing->ts_free.tv_sec == 0 ? -1 : debug_get_ts_interval(&msg_timing->ts_alloc, &msg_timing->ts_free);
            offset += snprintf(log_buf + offset, 256, " Interval: build=%ld transport=%ld handle=%ld total=%ld", build_time, transport_time, handle_time, total_time);
        }
    }

    log_buf[offset] = '\0';

    NVLOGC(TAG, "%s", log_buf);
    return 0;
}

nv_ipc_debug_t* nv_ipc_debug_open(nv_ipc_t* ipc, nv_ipc_config_t* cfg)
{
    int             size      = sizeof(nv_ipc_debug_t);
    nv_ipc_debug_t* ipc_debug = malloc(size);
    if(ipc_debug == NULL)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: memory malloc failed", __func__);
        return NULL;
    }
    memset(ipc_debug, 0, size);
    ipc_debug->ipc       = ipc;
    ipc_debug->transport = cfg->ipc_transport;
    ipc_debug->primary   = is_module_primary(cfg->module_type);

    if(ipc_debug->transport == NV_IPC_TRANSPORT_SHM)
    {
        nvlog_safe_strncpy(ipc_debug->prefix, cfg->transport_config.shm.prefix, NV_NAME_MAX_LEN);
    }
    else if(ipc_debug->transport == NV_IPC_TRANSPORT_DPDK)
    {
        nvlog_safe_strncpy(ipc_debug->prefix, cfg->transport_config.dpdk.prefix, NV_NAME_MAX_LEN);
    }
    else if(ipc_debug->transport == NV_IPC_TRANSPORT_DOCA)
    {
        nvlog_safe_strncpy(ipc_debug->prefix, cfg->transport_config.doca.prefix, NV_NAME_MAX_LEN);
    }
    else
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s unknown transport type %d", __func__, cfg->ipc_transport);
        free(ipc_debug);
        return NULL;
    }

    ipc_debug->alloc_hook = ipc_debug_alloc_hook;
    ipc_debug->free_hook  = ipc_debug_free_hook;
    ipc_debug->send_hook  = ipc_debug_send_hook;
    ipc_debug->recv_hook  = ipc_debug_recv_hook;
    ipc_debug->post_hook  = ipc_debug_post_hook;
    ipc_debug->wait_hook  = ipc_debug_wait_hook;

    ipc_debug->fw_deq_hook  = ipc_debug_fw_deq_hook;
    ipc_debug->fw_free_hook = ipc_debug_fw_free_hook;

    ipc_debug->close = ipc_debug_close;

    if(ipc_debug_open(ipc_debug, cfg) < 0)
    {
        NVLOGE_NO(TAG, AERIAL_NVIPC_API_EVENT, "%s: prefix=%s Failed", __func__, ipc_debug->prefix);
        ipc_debug_close(ipc_debug);
        return NULL;
    }
    else
    {
        NVLOGC(TAG, "%s: prefix=%s fapi_type=%d nvipc_version=%d.%d.%d OK", __func__, ipc_debug->prefix,
               fapi_type, LIBRARY_VERSION / 100, LIBRARY_VERSION / 10 % 10, LIBRARY_VERSION % 10);
        return ipc_debug;
    }

    return 0;
}
