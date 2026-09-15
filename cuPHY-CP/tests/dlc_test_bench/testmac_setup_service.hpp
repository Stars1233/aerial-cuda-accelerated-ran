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

#ifndef TESTMAC_SETUP_SERVICE_HPP
#define TESTMAC_SETUP_SERVICE_HPP

#include <cstdint>
#include <memory>
#include <string>

#include "scf_fapi_handler.hpp"
#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"
#include "nv_phy_mac_transport.hpp"
#include "yaml.hpp"

namespace aerial_fh {

class TestFapiHandler final : public scf_fapi_handler {
public:
    TestFapiHandler(test_mac_configs* configs, launch_pattern* lp, ch8_conformance_test_stats* stats)
        : scf_fapi_handler(configs, lp, stats) {}

    ~TestFapiHandler() override = default;

    using scf_fapi_handler::get_fapi_req_list;
    using scf_fapi_handler::build_dl_tti_request;
    using scf_fapi_handler::build_ul_tti_request;
    using scf_fapi_handler::build_ul_dci_request;
};

enum class TestMacPatternMode : uint8_t {
    F08SingleCell,
    NrSim
};

class TestMacSetupService final {
public:
    TestMacSetupService(const std::string& pattern_number, TestMacPatternMode pattern_mode, uint32_t n_cells);
    ~TestMacSetupService() = default;

    TestMacSetupService(const TestMacSetupService&) = delete;
    TestMacSetupService& operator=(const TestMacSetupService&) = delete;
    TestMacSetupService(TestMacSetupService&&) = delete;
    TestMacSetupService& operator=(TestMacSetupService&&) = delete;

    [[nodiscard]] TestFapiHandler* fapi_handler() { return fapi_handler_.get(); }

    [[nodiscard]] test_mac_configs* configs() { return testmac_configs_.get(); }

    [[nodiscard]] launch_pattern* get_launch_pattern() { return launch_pattern_.get(); }

    [[nodiscard]] const thrput_t& expected_values_for_cell(uint32_t cell);

    [[nodiscard]] uint32_t dl_abs_freq_point_a() const { return nDLAbsFrePointA_; }
    [[nodiscard]] uint32_t ul_abs_freq_point_a() const { return nULAbsFrePointA_; }

private:
    yaml::document                          testmac_yaml_document_;
    std::unique_ptr<test_mac_configs>       testmac_configs_;
    std::unique_ptr<launch_pattern>         launch_pattern_;
    std::unique_ptr<TestFapiHandler>        fapi_handler_;
    uint32_t                                 n_cells_{};
    uint32_t nDLAbsFrePointA_{};
    uint32_t nULAbsFrePointA_{};
};

} // namespace aerial_fh

#endif // TESTMAC_SETUP_SERVICE_HPP
