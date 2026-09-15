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

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "launch_pattern.hpp"

// TEST(TestMac,LaunchParsePattern)
// {
//     char test_mac_yaml[MAX_PATH_LEN];
//     std::string temp_path = std::string(CONFIG_TESTMAC_YAML_PATH).append(CONFIG_TESTMAC_YAML_NAME);
//     get_cubb_full_path(test_mac_yaml, NULL, temp_path.c_str());

//     yaml::file_parser fp(test_mac_yaml);
//     yaml::document    doc       = fp.next_document();
//     yaml::node        yaml_root = doc.root();
//     test_mac_configs* configs = new test_mac_configs(yaml_root);
//     launch_pattern *lp = new launch_pattern(configs);

//     uint32_t channel_mask = 0x7ff;
//     uint64_t cell_mask = 0x3;
//     std::string launch_pattern_file("launch_pattern_F08_2C_59c.yaml");
//     int result = lp->launch_pattern_parsing(launch_pattern_file.c_str(), channel_mask, cell_mask);
//     EXPECT_TRUE(result >= 0);
// }
