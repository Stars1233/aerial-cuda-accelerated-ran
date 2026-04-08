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

#ifndef _TEST_MAC_HPP_
#define _TEST_MAC_HPP_

#include <map>

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"

#include "common_defines.hpp"
#include "fapi_handler.hpp"
#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"
#include "test_mac_stats.hpp"

#include "cuphyoam.hpp"

#ifdef AERIAL_CUMAC_ENABLE
#include "cumac_handler.hpp"
#endif

/**
 * Main test MAC application class
 * 
 * Manages the MAC layer test application including transport, FAPI handlers,
 * and cuMAC handlers for multi-cell operation
 */
class test_mac {
public:
    /**
     * Construct test MAC instance
     * 
     * @param[in] ipc_config YAML configuration for IPC transport
     * @param[in] cell_num Number of cells to support
     */
    test_mac(yaml::node ipc_config, uint32_t cell_num);
    virtual ~test_mac();

    /**
     * Start MAC receiver thread
     */
    void start();

    /**
     * Wait for MAC receiver thread to complete
     */
    void join();

    /**
     * Configure launch pattern and test MAC configurations
     * 
     * @param[in] _configs Test MAC configurations
     * @param[in] _lp Launch pattern for test execution
     */
    void set_launch_pattern_and_configs(test_mac_configs* _configs, launch_pattern* _lp);

#ifdef AERIAL_CUMAC_ENABLE
    /**
     * Set cuMAC pattern for accelerated MAC processing
     * 
     * @param[in] _cp cuMAC pattern configuration
     */
    void set_cumac_pattern(cumac_pattern* _cp);
    cumac_handler* _cumac_handler = nullptr; //!< cuMAC handler instance
#endif

    /**
     * Get reference to MAC-PHY transport layer
     * 
     * @return Reference to transport object
     */
    phy_mac_transport& transport() {
        return *_transport;
    }

    /**
     * Get test MAC configurations
     * 
     * @return Pointer to configuration object
     */
    test_mac_configs* get_configs() {
        return configs;
    }

    /**
     * Get FAPI handler instance
     * 
     * @return Pointer to FAPI handler
     */
    fapi_handler* get_fapi_handler() {
        return _fapi_handler;
    }

private:

    pthread_t mac_recv_tid = 0; //!< MAC receiver thread ID
    pthread_t mac_sched_tid = 0; //!< MAC scheduler thread ID

    phy_mac_transport* _transport = nullptr; //!< MAC-PHY transport object for IPC communication

    test_mac_configs* configs = nullptr;     //!< Test MAC configuration parameters
    fapi_handler*     _fapi_handler = nullptr; //!< FAPI message handler

    yaml::node testmac_yaml; //!< YAML configuration node
    ch8_conformance_test_stats conformance_test_stats; //!< Conformance test statistics
};

#endif /* _TEST_MAC_HPP_ */
