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
#include <memory>

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"
#include "yaml.hpp"

#include "common_defines.hpp"
#include "fapi_handler.hpp"
#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"
#include "test_mac_stats.hpp"
#include "scf_5g_fapi.h"
#include "nv_ipc.hpp"
#include "cuphyoam.hpp"

#ifdef AERIAL_CUMAC_ENABLE
#include "cumac_handler.hpp"
#endif

/**
 * Main test MAC application class
 *
 * Owns YAML-backed configuration, launch pattern, optional NVIPC transport, and the FAPI handler.
 * Typical sequence: construct from test MAC YAML, load_launch_pattern(), optional prebuild_fapi_messages(),
 * then start() which creates the transport (if absent), calls set_transport() on the handler, and spawns threads.
 */
class test_mac {
public:
    /** Fallback max IPC message buffer (bytes) until nv_ipc sizes are applied in start/prebuild. */
    static constexpr int default_max_msg_size = 65536;
    /** Fallback max IPC data buffer (bytes) until nv_ipc sizes are applied in start/prebuild. */
    static constexpr int default_max_data_size = 10 * 1024 * 1024;

    /**
     * Construct test MAC instance from the test MAC application YAML file.
     *
     * Parses the first YAML document, validates schema version, loads NVIPC config into ipc_config,
     * builds test_mac_configs, and applies default_max_msg_size / default_max_data_size until transport exists.
     * Keeps the file_parser and document alive so yaml::node views inside configs remain valid.
     *
     * @param[in] config_yaml_path Path to test_mac_config (or override) YAML on disk.
     */
    explicit test_mac(const char* config_yaml_path);
    virtual ~test_mac();

    /**
     * Parse launch pattern YAML, build launch_pattern and scf_fapi_handler (transport not set yet).
     *
     * @param[in] launch_pattern_path Path to launch pattern YAML (or dynamic pattern asset, per app mode).
     * @param[in] cell_mask           Bit mask of active cells.
     * @param[in] channel_mask        Bit mask of active channel types.
     * @return 0 on success, -1 if pattern parsing fails.
     */
    [[nodiscard]] int load_launch_pattern(const char* launch_pattern_path, uint64_t cell_mask, uint32_t channel_mask);

    /**
     * Build pre-serialized downlink FAPI messages (CONFIG.req and per-slot TX requests) into host memory.
     *
     * @return 0 on success, -1 if the FAPI handler was not created (call load_launch_pattern first), or -1 if prebuild was already done.
     *
     * @note Call after load_launch_pattern() and before start(). At most one successful prebuild per FAPI handler instance.
     *       Temporary build buffers use test_mac_configs::get_max_msg_size() / get_max_data_size()
     *       (constructor defaults until start() refreshes from NVIPC).
     */
    [[nodiscard]] int prebuild_fapi_messages();

    /**
     * Log a human-readable dump of pre-built CONFIG.req and slot messages (for --prebuild diagnostics).
     */
    void print_prebuilt_fapi_messages();

    /**
     * Access pre-built CONFIG.req for one cell (after prebuild_fapi_messages()).
     *
     * @param[in] cell_id MAC cell index.
     * @return Pointer to cached descriptor, or nullptr if the launch pattern was not loaded or @p cell_id is invalid.
     * @note The returned pointer aliases storage owned by the FAPI handler; do not free.
     */
    [[nodiscard]] const nv::phy_mac_msg_desc* get_prebuilt_config_req(int cell_id) const;

    /**
     * Access pre-built slot TX messages for one cell and SFN/slot (after prebuild_fapi_messages()).
     *
     * @param[in] cell_id MAC cell index.
     * @param[in] ss      Slot selector (SFN/slot) used to index the launch-pattern schedule.
     * @return View over message descriptors for that slot/cell, or an empty span if the handler
     *         is missing or @p cell_id is invalid.
     * @note The returned span aliases storage owned by the FAPI handler.
     */
    [[nodiscard]] std::span<const nv::phy_mac_msg_desc> get_prebuilt_slot_messages(int cell_id, sfn_slot_t ss) const;

    /**
     * Create NVIPC transport (if needed), attach it to the FAPI handler, refresh IPC buffer sizes, and start worker threads.
     *
     * @param[in] enable_uplink   If true, start the MAC receive thread (uplink / IPC RX path).
     * @param[in] enable_downlink If true, start the MAC scheduler thread (downlink scheduling path).
     *
     * @pre load_launch_pattern() must have completed successfully so _fapi_handler and _lp exist.
     *
     * @note Downlink thread depends on uplink thread, please set enable_uplink=true while enable_downlink=true
     */
    void start(bool enable_uplink = true, bool enable_downlink = true);

    /**
     * Join scheduler and MAC receive threads (recv thread may be cancelled first).
     */
    void join();

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
     * Get reference to the MAC-PHY NVIPC transport.
     *
     * @return Reference to the live transport instance.
     * @pre start() has created the transport; behavior is undefined if _transport is still null.
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
        return _configs.get();
    }

    /**
     * Get FAPI handler instance
     *
     * @return Pointer to FAPI handler
     */
    fapi_handler* get_fapi_handler() {
        return _fapi_handler.get();
    }

    /**
     * Get launch pattern instance
     *
     * @return Pointer to launch pattern
     */
    launch_pattern* get_launch_pattern() {
        return _lp.get();
    }

private:

    pthread_t mac_recv_tid = 0; //!< MAC receiver thread ID
    pthread_t mac_sched_tid = 0; //!< MAC scheduler thread ID

    std::unique_ptr<phy_mac_transport> _transport = nullptr; //!< MAC-PHY transport object for IPC communication

    //! Declared before _configs so destruction runs _configs first, then document/parser (yaml::nodes must not outlive document).
    yaml::document _config_yaml_document{};
    std::unique_ptr<yaml::file_parser> _config_yaml_parser = nullptr;

    std::unique_ptr<test_mac_configs> _configs = nullptr;     //!< Test MAC configuration parameters
    std::unique_ptr<launch_pattern> _lp = nullptr; //!< Launch pattern for test execution
    std::unique_ptr<fapi_handler> _fapi_handler = nullptr; //!< FAPI message handler

    ch8_conformance_test_stats conformance_test_stats; //!< Conformance test statistics

    nv_ipc_config_t ipc_config{}; //!< IPC configuration for transport
};

#endif /* _TEST_MAC_HPP_ */
