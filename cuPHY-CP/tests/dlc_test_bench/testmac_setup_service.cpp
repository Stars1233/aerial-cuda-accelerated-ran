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

#include "testmac_setup_service.hpp"
#include "common_utils.hpp"

#include <fmt/format.h>

#define TAG_UNIT_TB_BASE    NVLOG_TAG_BASE_DLC_TESTBENCH
#define TAG_UNIT_TB_COMMON  TAG_UNIT_TB_BASE + 1

namespace aerial_fh {
namespace {
constexpr uint32_t kChannelMask = 0x0FFFFFFF;
constexpr const char* kIpcPrefix = "testbench";
}  // namespace

TestMacSetupService::TestMacSetupService(const std::string& pattern_number, TestMacPatternMode pattern_mode, uint32_t n_cells)
    : n_cells_(n_cells)
{
    if (n_cells_ == 0) {
        throw std::invalid_argument("n_cells must be greater than 0");
    }

    char test_mac_yaml_array[MAX_PATH_LEN];
    std::string temp_path = std::string(CONFIG_TESTMAC_YAML_PATH).append(CONFIG_TESTMAC_YAML_NAME);
    get_full_path_file(test_mac_yaml_array, nullptr, temp_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    yaml::file_parser parser(test_mac_yaml_array);
    testmac_yaml_document_ = parser.next_document();
    yaml::node config_node = testmac_yaml_document_.root();

    testmac_configs_ = std::make_unique<test_mac_configs>(config_node);

    const std::string pattern_file = (pattern_mode == TestMacPatternMode::NrSim)
        ? fmt::format("launch_pattern_nrSim_{}.yaml", pattern_number)
        : fmt::format("launch_pattern_F08_1C_{}.yaml", pattern_number);

    const uint64_t cell_mask = (n_cells_ >= 64U) ? ~0ULL : ((1ULL << n_cells_) - 1ULL);
    launch_pattern_ = std::make_unique<launch_pattern>(testmac_configs_.get());
    if (launch_pattern_->launch_pattern_parsing(pattern_file.c_str(), kChannelMask, cell_mask) < 0) {
        NVLOGE_FMT(TAG_UNIT_TB_COMMON, AERIAL_INVALID_PARAM_EVENT, "Launch pattern parsing failed for pattern {}!", pattern_number);
        throw std::runtime_error("Launch pattern parsing failed");
    }

    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "Launch pattern {} loaded successfully", pattern_number);
    const std::size_t expected_cells = launch_pattern_->get_expected_values().size();
    for (uint32_t cell = 0; cell < n_cells_ && cell < expected_cells; ++cell) {
        std::string exp_slot = fmt::format("Launch pattern expected slot schedule: Cell={}", cell);
        for (int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++) {
            exp_slot += fmt::format(
                " {}={}",
                get_channel_name(ch),
                launch_pattern_->get_expected_values().at(cell).lp_slots[ch].load()
            );
        }
        NVLOGC_FMT(TAG_UNIT_TB_COMMON, "{}", exp_slot);
    }

    fapi_handler_ = std::make_unique<TestFapiHandler>(
        testmac_configs_.get(),
        launch_pattern_.get(),
        nullptr
    );

    nDLAbsFrePointA_ = config_node["data"]["nDLAbsFrePointA"].as<uint32_t>();
    nULAbsFrePointA_ = config_node["data"]["nULAbsFrePointA"].as<uint32_t>();
    NVLOGC_FMT(TAG_UNIT_TB_COMMON, "TestMacSetupService: initialization complete");
}

const thrput_t& TestMacSetupService::expected_values_for_cell(uint32_t cell)
{
    return launch_pattern_->get_expected_values().at(cell);
}

} // namespace aerial_fh
