/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _NVLOG_H_
#define _NVLOG_H_

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "aerial_event_code.h"

#ifdef __cplusplus /* For both C and C++ */
extern "C" {
#endif

// Default nvlog configuration file
#define NVLOG_DEFAULT_CONFIG_FILE "cuPHY/nvlog/config/nvlog_config.yaml"

// Module tag base numbers
#define NVLOG_TAG_BASE_RESERVED 0            // reserved
#define NVLOG_TAG_BASE_NVLOG 10              // nvlog
#define NVLOG_TAG_BASE_NVIPC 30              // nvIPC
#define NVLOG_TAG_BASE_UTIL 80              // aerial_utils
#define NVLOG_TAG_BASE_CUPHY_CONTROLLER 100  // cuphycontroller
#define NVLOG_TAG_BASE_CUPHY_DRIVER 200      // cuphydriver
#define NVLOG_TAG_BASE_L2_ADAPTER 300        // cuphyl2adapter
#define NVLOG_TAG_BASE_SCF_L2_ADAPTER 330    // scfl2adapter
#define NVLOG_TAG_BASE_TEST_MAC 400          // testMAC
#define NVLOG_TAG_BASE_CUMAC_CP 450          // cuMAC-CP
#define NVLOG_TAG_BASE_RU_EMULATOR 500       // ru-emulator
#define NVLOG_TAG_BASE_FH_DRIVER 600         // aerial-fh-driver
#define NVLOG_TAG_BASE_FH_GENERATOR 650      // fh_generator
#define NVLOG_TAG_BASE_COMPRESSION 700       // compression_decompression
#define NVLOG_TAG_BASE_CUPHY_OAM 800         // cuphyoam
#define NVLOG_TAG_BASE_CUPHY 900             // cuPHY
#define NVLOG_TAG_BASE_TESTBENCH 1000        // test bench (currently phase-3)
#define NVLOG_TAG_BASE_DLC_TESTBENCH 1010        // DLC test bench 
#define NVLOG_TAG_BASE_APP_CFG_UTILS 1100    // App cfg and utils
#define NVLOG_TAG_BASE_CUMAC 1200            // cuMAC

// Log levels
#define NVLOG_NONE 0 // Set log level to NVLOG_NONE can disable all log
#define NVLOG_FATAL 1
#define NVLOG_ERROR 2
#define NVLOG_CONSOLE 3
#define NVLOG_WARN 4
#define NVLOG_INFO 5
#define NVLOG_DEBUG 6
#define NVLOG_VERBOSE 7

#define NVLOG_DEFAULT_TAG_NUM 1024
#define NVLOG_NAME_MAX_LEN 32                  // Log name string length should be less than 32

void nvlog_c_print(int level, int itag, const char* format, ...);
void nvlog_e_c_print(int level, int itag, const char* event, const char* format, ...);

#define NVLOG_C(level, itag, format, ...) nvlog_c_print(level, itag, format, ##__VA_ARGS__)
#define NVLOG_E_C(level, itag, event, format, ...) nvlog_e_c_print(level, itag, #event, format, ##__VA_ARGS__)

#define NVLOGV(tag, format, ...) NVLOG_C(NVLOG_VERBOSE, tag, format, ##__VA_ARGS__)
#define NVLOGD(tag, format, ...) NVLOG_C(NVLOG_DEBUG, tag, format, ##__VA_ARGS__)
#define NVLOGI(tag, format, ...) NVLOG_C(NVLOG_INFO, tag, format, ##__VA_ARGS__)
#define NVLOGW(tag, format, ...) NVLOG_C(NVLOG_WARN, tag, format, ##__VA_ARGS__)
#define NVLOGC(tag, format, ...) NVLOG_C(NVLOG_CONSOLE, tag, format, ##__VA_ARGS__)
#define NVLOGE(tag, event, format, ...) NVLOG_E_C(NVLOG_ERROR, tag, event, format, ##__VA_ARGS__)
#define NVLOGF(tag, event, format, ...)                            \
    do {                                                           \
        NVLOG_E_C(NVLOG_FATAL, tag, event, format, ##__VA_ARGS__); \
        usleep(100000);                                            \
        exit(1);                                                   \
    } while(0)

#define NVLOGE_NO(tag, event, format, ...) NVLOG_E_C(NVLOG_ERROR, tag, event, format, ##__VA_ARGS__)

void nvlog_set_log_level(int log_level);
void nvlog_set_max_file_size(size_t size);
void nvlog_c_init(const char *file);
void nvlog_c_close();

// Copy at most (dest_size - 1) bytes and make sure it is terminated by '\0'.
static inline char* nvlog_safe_strncpy(char* dest, const char* src, size_t dest_size)
{
    if(dest == NULL)
    {
        return dest;
    }

    if(src == NULL)
    {
        *dest = '\0'; // Set destination to empty string
        return dest;
    }

    char* ret_dest          = strncpy(dest, src, dest_size - 1); // Reserve 1 byte for '\0'
    *(dest + dest_size - 1) = '\0';                              // Safely terminate the string with '\0'
    return ret_dest;
}

// Get monotonic time stamp
static inline int nvlog_gettime(struct timespec* ts)
{
    return clock_gettime(CLOCK_MONOTONIC, ts);
}

// Get real-time time stamp
static inline int nvlog_gettime_rt(struct timespec* ts)
{
    return clock_gettime(CLOCK_REALTIME, ts);
}

static inline int64_t nvlog_timespec_interval(struct timespec* t1, struct timespec* t2)
{
    return (t2->tv_sec - t1->tv_sec) * 1000000000LL + t2->tv_nsec - t1->tv_nsec;
}

static inline void nvlog_timespec_add(struct timespec* ts, int64_t ns)
{
    ns += ts->tv_nsec;
    ts->tv_sec += ns / 1000000000L;
    ts->tv_nsec = ns % 1000000000L;
}

static inline int64_t nvlog_get_interval(struct timespec* start)
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return (now.tv_sec - start->tv_sec) * 1000000000LL + now.tv_nsec - start->tv_nsec;
}

// struct timeval
static inline int64_t nvlog_timeval_interval(struct timeval* t1, struct timeval* t2)
{
    return (t2->tv_sec - t1->tv_sec) * 1000000LL + t2->tv_usec - t1->tv_usec;
}

#define NVLOG_TIME_STRING_LEN (16)

// Call gettimeofday to get time and format the string. buf_size should >=16 including '\0'. Style: 01:00:00.123456
static inline void nvlog_gettimeofday_string(char* ts_buf, int32_t buf_size)
{
    if(ts_buf == NULL || buf_size < 16)
    {
        return;
    }

    struct timeval tv;
    struct tm      ptm;
    gettimeofday(&tv, NULL);
    if(localtime_r(&tv.tv_sec, &ptm) != NULL)
    {
        // String style: 01:00:00.123456, size = 8 + 7 = 15
        size_t size = strftime(ts_buf, sizeof("00:00:00"), "%H:%M:%S", &ptm);
        size += snprintf(ts_buf + size, 8, ".%06ld", tv.tv_usec);
    }
}

#if defined(__cplusplus) /* For both C and C++ */
} /* extern "C" */
#endif

#endif /* _NVLOG_H_ */
