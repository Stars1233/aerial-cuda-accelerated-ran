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
#include "yamlparser.hpp"
#ifdef SLOTS_PER_FRAME
#undef SLOTS_PER_FRAME
#endif

#include "fapi_source_testmac.hpp"
#include "cplane_test_bench.hpp"

#include "nvlog_fmt.hpp"

#include <unistd.h>  // access()

namespace cplane_tb {

FapiTestMacSource::FapiTestMacSource(const std::string& pattern_number, bool is_nrsim)
{
    // --- testMAC configs ---
    char test_mac_yaml_array[MAX_PATH_LEN];
    std::string temp_path = std::string(CONFIG_TESTMAC_YAML_PATH).append(CONFIG_TESTMAC_YAML_NAME);
    get_full_path_file(test_mac_yaml_array, NULL, temp_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    yaml::file_parser parser(test_mac_yaml_array);
    yaml::document doc = parser.next_document();
    yaml::node config_node = doc.root();
    testmac_configs_ = std::make_shared<test_mac_configs>(config_node);

    // --- Launch pattern ---
    // Prefer launch_pattern_nrSim_<p>.yaml when it exists — some sub-90xxx patterns
    // (e.g. 0103) are nrSim test vectors but fall outside the 90000-99999 range
    // that TestConfig::is_nrsim() uses for detection. Fall back to F08_1C naming
    // for pure 4T4R patterns.
    std::string pattern_file;
    {
        char probe_path[MAX_PATH_LEN];
        const std::string nrsim_name = "launch_pattern_nrSim_" + pattern_number + ".yaml";
        get_full_path_file(probe_path, CONFIG_LAUNCH_PATTERN_PATH, nrsim_name.c_str(),
                           CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        if (access(probe_path, R_OK) == 0) {
            pattern_file = nrsim_name;
        } else if (is_nrsim) {
            // 90xxx range but file missing — keep the nrSim name so the original
            // error surfaces rather than silently masking it with F08_1C.
            pattern_file = nrsim_name;
        } else {
            pattern_file = "launch_pattern_F08_1C_" + pattern_number + ".yaml";
        }
    }

    launch_pattern_ = new launch_pattern(testmac_configs_.get());
    if (launch_pattern_->launch_pattern_parsing(pattern_file.c_str(), 0xFFFFFFF, 0x1) < 0) {
        NVLOGE_FMT(TAG_CPTB_COMMON, AERIAL_INVALID_PARAM_EVENT,
                   "Launch pattern parsing failed for pattern {}!", pattern_number);
        throw std::runtime_error("Launch pattern parsing failed");
    }

    NVLOGC_FMT(TAG_CPTB_COMMON, "Launch pattern {} loaded successfully ({} slots)",
               pattern_number, launch_pattern_->get_sched_slot_num());

    // --- FAPI handler ---
    fapi_handler_ = std::make_shared<TestFapiHandler>(
        testmac_configs_.get(), launch_pattern_, nullptr);

    // --- Allocate reusable FAPI buffers ---
    dl_tti_buf_ = static_cast<scf_fapi_dl_tti_req_t*>(
        std::malloc(sizeof(scf_fapi_dl_tti_req_t) + FAPI_PAYLOAD_BUFFER_SIZE));
    ul_dci_buf_ = static_cast<scf_fapi_ul_dci_t*>(
        std::malloc(sizeof(scf_fapi_ul_dci_t) + FAPI_PAYLOAD_BUFFER_SIZE));
    ul_tti_buf_ = static_cast<scf_fapi_ul_tti_req_t*>(
        std::malloc(sizeof(scf_fapi_ul_tti_req_t) + FAPI_PAYLOAD_BUFFER_SIZE));
    if ((dl_tti_buf_ == nullptr) || (ul_dci_buf_ == nullptr) || (ul_tti_buf_ == nullptr)) {
        std::free(dl_tti_buf_);
        std::free(ul_dci_buf_);
        std::free(ul_tti_buf_);
        dl_tti_buf_ = nullptr;
        ul_dci_buf_ = nullptr;
        ul_tti_buf_ = nullptr;
        throw std::runtime_error("Failed to allocate reusable FAPI buffers");
    }
}

FapiTestMacSource::~FapiTestMacSource()
{
    // fapi_handler::~fapi_handler() explicitly calls lp->~launch_pattern() (destructor only,
    // no delete). We must destroy fapi_handler first, then free the launch_pattern memory
    // without calling the destructor again.
    fapi_handler_.reset();

    // launch_pattern_ destructor already called by fapi_handler — just free the raw memory.
    ::operator delete(launch_pattern_);

    std::free(dl_tti_buf_);
    std::free(ul_dci_buf_);
    std::free(ul_tti_buf_);
}

size_t FapiTestMacSource::get_total_slots() const
{
    return static_cast<size_t>(launch_pattern_->get_sched_slot_num());
}

size_t FapiTestMacSource::get_num_cells() const
{
    return 1; // TODO: support multi-cell
}

SlotFapiMessages FapiTestMacSource::get_slot(size_t slot_index, size_t cell_index)
{
    SlotFapiMessages result{};

    // Convert linear slot index to SFN/slot (20 slots per frame for mu=1)
    result.sfn = static_cast<uint16_t>(slot_index / 20);
    result.slot = static_cast<uint16_t>(slot_index % 20);

    sfn_slot_t ss{};
    ss.u16.sfn = result.sfn;
    ss.u16.slot = result.slot;

    // Reset buffers
    dl_tti_buf_->num_pdus = 0;
    dl_tti_buf_->sfn = result.sfn;
    dl_tti_buf_->slot = result.slot;
#ifdef SCF_FAPI_10_04
    // build_dl_tti_request_public uses ++ to accumulate these counts; the
    // adjacent num_pdus reset didn't cover this array, leaving stale values.
    for (int i = 0; i < 5; ++i) dl_tti_buf_->nPDUsOfEachType[i] = 0;
#endif
    ul_dci_buf_->num_pdus = 0;
    ul_dci_buf_->sfn = result.sfn;
    ul_dci_buf_->slot = result.slot;
    ul_tti_buf_->num_pdus = 0;
    ul_tti_buf_->sfn = result.sfn;
    ul_tti_buf_->slot = result.slot;

    int cell_id = static_cast<int>(cell_index);

    // Generate FAPI messages for each group
    for (int group_id = 0; group_id < FAPI_REQ_SIZE; group_id++) {
        std::vector<fapi_req_t*>& fapi_reqs =
            fapi_handler_->get_fapi_req_list_public(cell_id, ss, static_cast<fapi_group_t>(group_id));

        if (fapi_reqs.empty()) {
            continue;
        }

        switch (group_id) {
            case DL_TTI_REQ:
                fapi_handler_->build_dl_tti_request_public(cell_id, fapi_reqs, *dl_tti_buf_);
                result.has_dl = true;
                break;
            case UL_DCI_REQ:
                fapi_handler_->build_ul_dci_request_public(cell_id, fapi_reqs, *ul_dci_buf_);
                result.has_dl = true; // UL_DCI is part of DL C-Plane
                break;
            case UL_TTI_REQ:
                fapi_handler_->build_ul_tti_request_public(cell_id, fapi_reqs, *ul_tti_buf_);
                result.has_ul = true;
                break;
            default:
                break;
        }
    }

    if (result.has_dl) {
        result.dl_tti_req = dl_tti_buf_;
        result.dl_body_len = FAPI_PAYLOAD_BUFFER_SIZE; // Upper bound; CPlaneGenerator handles actual parsing
        result.ul_dci_req = ul_dci_buf_;
        result.ul_dci_body_len = FAPI_PAYLOAD_BUFFER_SIZE;
    }

    if (result.has_ul) {
        result.ul_tti_req = ul_tti_buf_;
        result.ul_body_len = FAPI_PAYLOAD_BUFFER_SIZE;
    }

    return result;
}

} // namespace cplane_tb
