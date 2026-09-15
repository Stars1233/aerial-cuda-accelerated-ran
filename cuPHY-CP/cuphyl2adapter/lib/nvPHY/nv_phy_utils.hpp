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

#if !defined(NV_PHY_UTILS_HPP_)
#define NV_PHY_UTILS_HPP_

#include <sys/types.h>  /* system data type definitions */
#include <sys/socket.h> /* socket specific definitions */
#include <netinet/in.h> /* INET constants and stuff */
#include <arpa/inet.h>  /* IP address conversion stuff */
#include <unistd.h>

#include <sys/ipc.h>
#include <sys/msg.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "nv_phy_fapi_msg_common.hpp"

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

namespace nv {

/// Maximum SFN (System Frame Number) value in 5G NR
#define FAPI_SFN_MAX 1024

    /**
     * @brief Thread configuration structure
     *
     * Contains parameters for configuring thread properties including
     * name, CPU affinity, and scheduling priority.
     */
    struct thread_config
    {
        std::string name;        ///< Thread name for identification
        int    cpu_affinity;     ///< CPU core to bind thread to
        int    sched_priority;   ///< Real-time scheduling priority
    };

    /**
     * @brief Assign a name to the calling thread
     * @param name Thread name (max 16 characters including null terminator)
     * @return 0 on success, error code on failure
     */
    inline int assign_thread_name(const char* name)
    {
        return pthread_setname_np(pthread_self(), name);
    }

    /**
     * @brief Bind the calling thread to a specific CPU core
     * @param cpu_id CPU core identifier
     * @return 0 on success, error code on failure
     */
    inline int assign_thread_cpu_core(int cpu_id)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_id, &mask);
        return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    }

    /**
     * @brief Set real-time FIFO scheduling priority for the calling thread
     * @param priority Priority level (higher values = higher priority)
     * @return 0 on success, error code on failure
     */
    inline int assign_thread_priority(int priority)
    {
        struct sched_param param;
        param.__sched_priority = priority;
        pthread_t thread_me = pthread_self();
        return pthread_setschedparam(thread_me, SCHED_FIFO, &param);
    }

    /**
     * @brief Configure all thread properties (name, affinity, priority)
     * @param config Thread configuration structure
     * @return 0 on success, negative error code on failure
     *         -1: failed to set thread name
     *         -2: failed to set CPU affinity
     *         -3: failed to set scheduling priority
     */
    inline int config_thread_property(thread_config & config)
    {
        int ret = 0;
        if (assign_thread_name(config.name.c_str()) != 0) {
            ret = -1;
        }
        if (assign_thread_cpu_core(config.cpu_affinity) != 0) {
            ret = -2;
        }
        if (assign_thread_priority(config.sched_priority) != 0) {
            ret = -3;
        }
        return ret;
    }

#if 0
    std::size_t create_udp_socket(const std::size_t address, const std::size_t port) {
        int sock = -1;
        int flags = 0;
        ::sockaddr_in addr;

        // Attempt to initialize socket
        sock = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock < 0) {
            perror("Opening socket");
            EXIT_L1(EXIT_FAILURE);
        }

        // Return socket now if binding is not needed
        if (address == 0 && port == 0) {
            return sock;
        }

        // Initialize address structure
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        if (address == 0) {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            addr.sin_addr.s_addr = htonl(address);
        }
        addr.sin_port = htons(port);
        // std::cout<<addr.sin_port<<" "<<::ntohs(addr.sin_port)<<" "<<port<<std::endl;
        // Attempt to bind socket to the port/address specified
        if (bind(sock, (sockaddr *) &addr, sizeof(addr)) < 0) {
            ::fprintf(stderr, "Binding address: %lu port: %lu\n", (long unsigned int) address, port);
            ::perror("Binding");
            EXIT_L1(EXIT_FAILURE);
        }

        fprintf(stderr, "Created socket: fd=%d, addr=%lu port=%lu\n", sock, (long unsigned int) address,
                port);

        return sock;
    }

    void set_sockaddr_in(::sockaddr_in* addr, const char* ip, uint16_t port) {
        std::cout<<"Input IP:"<<ip<<" Input Port:"<<port<<std::endl;
        bzero(addr, sizeof(::sockaddr_in));
        addr->sin_family = AF_INET;
        addr->sin_port = htons(port);
        addr->sin_addr.s_addr = inet_addr(ip);
        // std::cout<<"Output IP:"<<::inet_ntoa(addr->sin_addr)<<" Output Port:"<<::ntohs(addr->sin_port)<<std::endl;
    }

    void set_blocking_fd(int fd, bool blocking) {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags == -1 ) {
            perror("failed to F_GETFL");
        }
        if (blocking) {
            flags &= ~O_NONBLOCK;
        } else {
            flags |= O_NONBLOCK;
        }
        if (fcntl(fd, F_SETFL, flags) == -1) {
            perror("Failed to set F_SETFL");
        }
    }

    static constexpr std::size_t NS_PER_SEC = 1 * 1000 * 1000 * 1000;

    static inline void normalize_ts(timespec * spec) {
        for(;spec->tv_nsec > NS_PER_SEC;) {
            spec->tv_nsec-= NS_PER_SEC;
            spec->tv_sec++;
        }
    }
#endif
    /**
     * @brief Convert numerology (mu) to slot duration in nanoseconds
     * @param mu Numerology value (0-4 for 15kHz, 30kHz, 60kHz, 120kHz, 240kHz)
     * @return Slot duration in nanoseconds
     * @throws std::runtime_error if mu > 4
     */
    inline std::size_t mu_to_ns(uint8_t mu) {
         if (mu <= 4) {
            return 1000000 >> mu; // Use shift to do quick calculation: 1000000 / 2 ^ mu
        } else {
            throw std::runtime_error("mu_to_ns error: invalid mu " + std::to_string(mu));
        }
    }

    /**
     * @brief Convert numerology (mu) to number of slots per radio frame
     * @param mu Numerology value (0-4)
     * @return Number of slots per 10ms radio frame (= 10 * 2^mu)
     * @throws std::runtime_error if mu > 4
     */
    inline uint16_t mu_to_slot_in_sf(uint8_t mu) {
        if (mu <= 4) {
            return 10 << mu; // Use shift to do quick calculation: 10 * 2 ^ mu
        } else {
            throw std::runtime_error("mu_to_slot_in_sf: invalid mu " + std::to_string(mu));
        }
    }

    /**
     * @brief Get NR band information based on DL/UL grid sizes
     * @param dlGridSizemHz Downlink grid size in MHz
     * @param ulGridSizemHz Uplink grid size in MHz
     * @return Pointer to NR band entry, or bands.end() if not found
     */
    inline const nr_band_entry* getNrBand(uint32_t dlGridSizemHz, uint32_t ulGridSizemHz) {
        auto iter = std::find_if(bands.begin(), bands.end(), [&dlGridSizemHz, &ulGridSizemHz](const auto& elem) {
            return (ulGridSizemHz >= elem.ul_band_low  && ulGridSizemHz <= elem.ul_band_hi) &&
            (dlGridSizemHz >= elem.dl_band_low && dlGridSizemHz <= elem.dl_band_hi);
        });
        
        return iter;
    }

    /**
     * @brief Determine SSB (Synchronization Signal Block) case based on frequency and SCS
     * @param dlGridSizemHz Downlink grid size in MHz
     * @param ulGridSizemHz Uplink grid size in MHz
     * @param ssb_scs SSB subcarrier spacing
     * @return SSB case (A, B, C, D, E) or CASE_UNKNOWN
     */
    inline ssb_case getSSBCase(uint32_t dlGridSizemHz, uint32_t ulGridSizemHz, uint8_t ssb_scs) {
        
        auto iter = getNrBand(dlGridSizemHz, ulGridSizemHz);

        if (iter == bands.end()) {
            return ssb_case::CASE_UNKNOWN;
        }
        nr_bands band = iter->band;

        auto iter2 = std::find_if(ss_raster.begin(), ss_raster.end(), [&band, &ssb_scs] (const auto& elem) {
            return (elem.band == band && elem.ssb_scs == ssb_scs); 
        });
        if (iter2 == ss_raster.end()) {
            return ssb_case::CASE_UNKNOWN;
        }
        return iter2->block_pattern;
    }

    /**
     * @brief Check if a band is in FR2 (Frequency Range 2, mmWave)
     * @param band NR band identifier
     * @return true if band is FR2 (n257, n258, n260, n261), false otherwise
     */
    inline bool isFr2Band(const nr_bands& band) {
        return ((band == nr_bands::N257) || (band == nr_bands::N258) || (band == nr_bands::N260) || (band == nr_bands::N261));
    }

    inline float get_fr1_lower_bw(uint8_t mu, uint16_t ulBandwidth) {
        auto & mu_table = fr1_bw_config_table[mu];
        switch(ulBandwidth) {
            case 5: return mu_table[0];
            case 10: return mu_table[1];
            case 15: return mu_table[2];            
            case 20: return mu_table[3];            
            case 25: return mu_table[4];            
            case 30: return mu_table[5];            
            case 40: return mu_table[6];            
            case 50: return mu_table[7];            
            case 60: return mu_table[8];            
            case 70: return mu_table[9];            
            case 80: return mu_table[10];            
            case 90: return mu_table[11];            
            case 100: return mu_table[12];
        }
        return 0.0f;
    }

    inline float get_fr2_lower_bw(uint8_t mu, uint16_t ulBandwidth) {
        if (mu == 3 || mu == 4) {
            auto& mu_table = fr2_bw_config_table[ mu - 3];
            switch (ulBandwidth) {
                case 50: return mu_table[0];
                case 100: return mu_table[1];
                case 200: return mu_table[2]; 
                case 400: return mu_table[3];
            }
        } else if (mu == 5) {
            return fr2_240_khz_bw_config_table[(ulBandwidth/100) >> 1];
        }
        return 0.0f;
    }

    inline float get_lower_bw(uint8_t mu,  uint32_t dlGridSizemHz, uint32_t ulGridSizemHz, uint16_t ulBandwidth) {

        auto iter = getNrBand(dlGridSizemHz, ulGridSizemHz);
        if (iter == bands.end()) {
            return 0.0f;
        }

        auto isFr2 = isFr2Band(iter->band);
        if (!isFr2) {
            return get_fr1_lower_bw(mu, ulBandwidth);
        } else {
            return get_fr2_lower_bw(mu, ulBandwidth);
        }
    }

    using namespace std::chrono_literals;

    inline auto get_duration(uint8_t tdd_period) {
        switch(tdd_period) {
            case 0: return 500us;
            case 1: return 625us;
            case 2: return 1000us;
            case 3: return 1250us;
            case 4: return 2000us;
            case 5: return 2500us;
            case 6: return 5000us;
            case 7: return 10000us;
            default:  return 5000us;
        }
    }
}
#endif
