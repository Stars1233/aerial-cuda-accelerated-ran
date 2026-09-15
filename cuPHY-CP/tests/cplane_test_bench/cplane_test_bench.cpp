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

// Include yamlparser first, then undef SLOTS_PER_FRAME to avoid conflict with testMAC macro

// GCC 12 emits a false-positive -Wnull-dereference when CLI::detail::split
// (basic_string SSO move via vector::emplace_back) is inlined into app.parse
// here at -O2. Scope the suppression to this instantiation point only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
#include "yamlparser.hpp"
#pragma GCC diagnostic pop

#ifdef SLOTS_PER_FRAME
#undef SLOTS_PER_FRAME
#endif

#include "cplane_test_bench.hpp"
#include "cplane_test_fixture.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <chrono>

#include "nvlog_fmt.hpp"
#include "CLI/CLI.hpp"

namespace {
// Emit the final pass/fail summary through nvlog so it flows on the same async
// fmtlog stream as the per-test logs. gtest prints its summary synchronously to
// stdout from inside RUN_ALL_TESTS, so async nvlog lines can flush after it;
// routing the result here (fired at OnTestProgramEnd) plus the post-run
// fmtlog::poll(true) makes the result the last line in the nvlog output.
class SummaryLogger : public ::testing::EmptyTestEventListener {
    /// Log the total/passed/failed test counts via nvlog at program end.
    void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
        NVLOGC_FMT(TAG_CPTB_COMMON,
                   "C-Plane Test Bench result -- Total:{} Passed:{} Failed:{}",
                   unit_test.total_test_count(), unit_test.successful_test_count(),
                   unit_test.failed_test_count());
    }
};
} // namespace

int main(int argc, char** argv)
{
    // Initialize NVLOG
    char root[1024];
    get_root_path(root, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    std::string yaml_path = std::string(root).append(NVLOG_DEFAULT_CONFIG_FILE);
    pthread_t bg_thread_id = nvlog_fmtlog_init(yaml_path.c_str(), "cplane_tb_nvlog.log", NULL);

    if (bg_thread_id == static_cast<pthread_t>(-1)) {
        std::cerr << "nvlog_fmtlog_init() failed! yaml_path: " << yaml_path << std::endl;
        return -1;
    }

    // Parse CLI arguments
    CLI::App app{"C-Plane Test Bench — framework CPlaneGenerator -> RU Emulator verification"};

    auto& cfg = cplane_tb::TestConfig::instance();
    app.add_option("-p,--pattern", cfg.pattern_number, "Test vector pattern number")->required();
    app.add_flag("--enable-pcap", cfg.enable_pcap, "Enable PCAP packet capture to file");
    app.add_option("--pcap-file", cfg.pcap_file_name, "PCAP output filename (saved in /tmp/)");

    try {
        app.allow_extras(); // Don't fail on GTest-specific args
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    // Re-emit the result via nvlog so it's the last (correctly ordered) line.
    // listeners().Append() takes ownership of the raw pointer and deletes it at
    // shutdown; release() hands ownership over without a naked new.
    ::testing::UnitTest::GetInstance()->listeners().Append(
        std::make_unique<SummaryLogger>().release());

    // Log configuration
    NVLOGC_FMT(TAG_CPTB_COMMON, "=============================================================");
    NVLOGC_FMT(TAG_CPTB_COMMON, "C-Plane Test Bench Configuration");
    NVLOGC_FMT(TAG_CPTB_COMMON, "=============================================================");
    NVLOGC_FMT(TAG_CPTB_COMMON, "  Pattern:    {}", cfg.pattern_number);
    NVLOGC_FMT(TAG_CPTB_COMMON, "  PCAP:       {}", cfg.enable_pcap ? "Enabled" : "Disabled");
    NVLOGC_FMT(TAG_CPTB_COMMON, "  Verify:     {}", cfg.verify_cplane ? "Enabled" : "Disabled");
    NVLOGC_FMT(TAG_CPTB_COMMON, "=============================================================");

    fmtlog::poll(true);

    int result = RUN_ALL_TESTS();

    fmtlog::poll(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    nvlog_fmtlog_close(bg_thread_id);

    return result;
}
