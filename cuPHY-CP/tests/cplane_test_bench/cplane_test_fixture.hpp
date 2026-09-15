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

#ifndef CPLANE_TEST_FIXTURE_HPP__
#define CPLANE_TEST_FIXTURE_HPP__

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "oran/cplane_generator.hpp"
#include "oran/cplane_types.hpp"
#include "oran/vec_buf.hpp"

#include "fapi_source.hpp"
#include "fapi_source_testmac.hpp"

struct rte_mempool;

namespace cplane_tb {

/// GTest fixture that wires CPlaneGenerator output directly to the RU emulator.
///
/// SetUpTestSuite: creates CPlaneGenerator, FapiTestMacSource, RU emulator.
/// Each TEST_F calls RunSlot() which: generates FAPI -> prepare -> generate -> verify.
/// TearDownTestSuite: checks RU emulator error counters.
class CPlaneTestFixture : public ::testing::Test {
protected:
    static void SetUpTestSuite();
    static void TearDownTestSuite();

    /// Run a single slot through the pipeline:
    ///   FAPI source -> CPlaneGenerator -> VecBuf -> RU emulator verify
    void RunSlotDl(size_t slot_index, size_t cell_index = 0);

    /// Run a single UL slot through the pipeline:
    ///   FAPI source -> CPlaneGenerator::prepare_ul -> generate_ul_packets -> RU emulator verify
    void RunSlotUl(size_t slot_index, size_t cell_index = 0);

    /// Capture generated packets to /tmp/<pcap_file_name> via the pcap_writer library.
    /// VecBufs are copied into rte_mbufs from pcap_mbuf_pool_ to use the mbuf-based PCAP API
    /// (same library DLC test bench uses). No-op if TestConfig::enable_pcap is false.
    void capture_packets(const ran::oran::VecBuf* bufs, size_t num_packets);

    // Shared state across all tests in the suite
    static std::unique_ptr<ran::oran::CPlaneGenerator> generator_;
    static std::unique_ptr<FapiTestMacSource> fapi_source_;
    static void* ru_emulator_;
    static std::vector<ran::oran::VecBuf> packet_buffers_;
    static rte_mempool* pcap_mbuf_pool_;

    // Config derived from YAML
    static ran::oran::RuConfig ru_config_;
    static ran::oran::OranTxWindows tx_windows_;

    // mMIMO state: when enabled, per-slot BFW completion records are fed to
    // prepare_dl/prepare_ul. Disabled => non-mMIMO path is byte-for-byte unchanged.
    static bool mmimo_enabled_;
    static ran::oran::BfwConfig bfw_config_;
};

} // namespace cplane_tb

#endif // CPLANE_TEST_FIXTURE_HPP__
