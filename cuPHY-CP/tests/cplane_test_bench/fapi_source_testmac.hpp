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

#include "fapi_source.hpp"

#include <memory>
#include <string>
#include <cstdlib>

#include "scf_fapi_handler.hpp"
#include "test_mac_configs.hpp"
#include "launch_pattern.hpp"

namespace cplane_tb {

/// Wrapper that exposes protected methods from scf_fapi_handler for test use.
/// No transport is set: the bench only invokes the build_*_request / get_fapi_req_list
/// methods which compose FAPI messages from the launch pattern and never touch IPC.
class TestFapiHandler : public scf_fapi_handler {
public:
    TestFapiHandler(test_mac_configs* test_configs, launch_pattern* pattern,
                    ch8_conformance_test_stats* stats)
        : scf_fapi_handler(test_configs, pattern, stats)
    {
    }

    std::vector<fapi_req_t*>& get_fapi_req_list_public(int cell_id, sfn_slot_t ss, fapi_group_t group_id) {
        return get_fapi_req_list(cell_id, ss, group_id);
    }

    int build_dl_tti_request_public(int cell_id, std::vector<fapi_req_t*>& fapi_reqs,
                                    scf_fapi_dl_tti_req_t& req) {
        return build_dl_tti_request(cell_id, fapi_reqs, req);
    }

    int build_ul_tti_request_public(int cell_id, std::vector<fapi_req_t*>& fapi_reqs,
                                    scf_fapi_ul_tti_req_t& req) {
        return build_ul_tti_request(cell_id, fapi_reqs, req);
    }

    int build_ul_dci_request_public(int cell_id, std::vector<fapi_req_t*>& fapi_reqs,
                                    scf_fapi_ul_dci_t& req) {
        return build_ul_dci_request(cell_id, fapi_reqs, req);
    }
};

/// FAPI source backed by testMAC launch patterns and scf_fapi_handler.
/// Generates FAPI DL_TTI_REQ, UL_DCI_REQ, UL_TTI_REQ from pattern YAML files.
class FapiTestMacSource : public FapiSource {
public:
    /// Construct from a pattern number string (e.g., "79", "90001").
    explicit FapiTestMacSource(const std::string& pattern_number, bool is_nrsim);
    ~FapiTestMacSource() override;

    [[nodiscard]] size_t get_total_slots() const override;
    [[nodiscard]] size_t get_num_cells() const override;
    [[nodiscard]] SlotFapiMessages get_slot(size_t slot_index, size_t cell_index) override;

    /// Access the launch pattern for expected-value queries in teardown.
    [[nodiscard]] launch_pattern* get_launch_pattern() const { return launch_pattern_; }

private:
    static constexpr size_t FAPI_PAYLOAD_BUFFER_SIZE = 64 * 1024;

    std::shared_ptr<test_mac_configs> testmac_configs_;
    launch_pattern* launch_pattern_{nullptr};
    std::shared_ptr<TestFapiHandler> fapi_handler_;

    // Reusable FAPI message buffers
    scf_fapi_dl_tti_req_t* dl_tti_buf_{nullptr};
    scf_fapi_ul_dci_t* ul_dci_buf_{nullptr};
    scf_fapi_ul_tti_req_t* ul_tti_buf_{nullptr};
};

} // namespace cplane_tb
