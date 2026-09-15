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

#pragma once

#include <string>
#include <cstdint>

#include "nvlog_fmt.hpp"

#define TAG_CPTB_BASE    NVLOG_TAG_BASE_DLC_TESTBENCH
#define TAG_CPTB_COMMON  TAG_CPTB_BASE + 1

namespace cplane_tb {

/// Global configuration from command-line arguments (singleton).
struct TestConfig {
    std::string pattern_number{};
    bool enable_pcap{false};
    bool verify_cplane{true};
    std::string pcap_file_name{"cplane_tb_packets.pcap"};

    static TestConfig& instance() {
        static TestConfig config;
        return config;
    }

    /// nrSim patterns are in the 90000-99999 range.
    [[nodiscard]] bool is_nrsim() const {
        try {
            int num = std::stoi(pattern_number);
            return (num >= 90000 && num < 100000);
        } catch (...) {
            return false;
        }
    }
};

} // namespace cplane_tb
