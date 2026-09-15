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

/**
 * @file pusch_parser_benchmark.cpp
 * @brief Throughput benchmark for the PUSCH parser paths.
 *
 * Feeds real FAPI UL_TTI.request PUSCH PDUs (from a testMAC launch pattern) into:
 *   - BM_PuschParserNewPath    : ULSlotProcessor::process<UL_TTI_PDU_TYPE_PUSCH>
 *   - BM_PuschParserLegacyPath : scf_5g_fapi::update_cell_command(pusch)
 *
 * Each benchmark resolves the same data assets as the functional test and
 * skips-with-error when they are unavailable. The new path is measured from the
 * raw message (it includes message validation + PDU walk); the legacy path is
 * measured from an already-parsed request, so the two numbers are not directly
 * comparable as a like-for-like ratio.
 */

#include <benchmark/benchmark.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "fapi_message_source.hpp"  // testMAC-backed message generator (clean boundary)
#include "ul_module_view_mock.hpp"
#include "pusch_parser_drivers.hpp"

#include "scf_5g_fapi.h"
#include "scf_5g_slot_commands.hpp"
#include "scf_5g_fapi_ul_slot_processor.hpp"

#include "nv_fapi_pdu_utils.hpp"

namespace cuphy_cp::tests
{
namespace
{

/// One captured legacy-path request paired with the cell id of its carrying
/// message, so the benchmark drives the legacy populator with the same cell id
/// the new path reads from @c phy_mac_msg_desc::cell_id (not a hardcoded 0).
struct LegacyRequest final
{
    const scf_fapi_ul_tti_req_t* req{nullptr};
    int                          cell_id{0};
};

/// Inputs gathered once (outside the timed region) from a source's PUSCH PDUs.
struct PuschBenchInputs final
{
    std::optional<FapiMessageSource>         source{std::nullopt};
    std::vector<const nv::phy_mac_msg_desc*> messages{};  // new-path entry
    std::vector<LegacyRequest>               requests{};  // legacy-path entry
};

/**
 * Open a source and collect every PUSCH-bearing UL_TTI message / request.
 *
 * Opens a testMAC-backed @c FapiMessageSource and, for every @c SCF_FAPI_UL_TTI_REQUEST
 * carrying PUSCH PDUs, records the new-path @c phy_mac_msg_desc and the legacy-path
 * @c scf_fapi_ul_tti_req_t so both parser front-ends can be driven from one capture.
 *
 * @return Populated @c PuschBenchInputs whose @c messages / @c requests reference the
 *         PUSCH-bearing UL_TTI entries; the contained @c source owns the backing
 *         storage, so those pointers stay valid for the lifetime of the returned
 *         object. On failure the collection is empty: @c source is @c std::nullopt
 *         when the assets cannot be opened, and @c messages / @c requests are empty
 *         when no PUSCH data is present.
 * @note   Marked @c [[nodiscard]]: the gathered inputs are the sole product of the
 *         call and must not be ignored.
 */
[[nodiscard]] PuschBenchInputs gather_pusch_inputs()
{
    PuschBenchInputs in{};
    in.source = FapiMessageSource::open_pusch();
    if (!in.source.has_value())
    {
        return in;
    }

    for (const SourceMessage& sm : in.source->messages())
    {
        if (sm.desc.msg_id != SCF_FAPI_UL_TTI_REQUEST) { continue; }
        nv::for_each_tti_msg<kPuschParserTestTag, scf_fapi_ul_tti_req_t>(
            &sm.desc, 1u,
            [](uint16_t, const scf_fapi_ul_tti_req_t& req) { return nv::has_pusch(req); },
            sm.sfn_slot, /*ring_idx=*/0u,
            [&](uint16_t, const nv::phy_mac_msg_desc& m, const scf_fapi_ul_tti_req_t& req) {
                in.messages.push_back(&m);
                in.requests.push_back(LegacyRequest{&req, m.cell_id});
            });
    }
    return in;
}

void BM_PuschParserNewPath(benchmark::State& state)
{
    PuschBenchInputs in = gather_pusch_inputs();
    if (in.messages.empty())
    {
        state.SkipWithError("PUSCH data assets unavailable (set cuBB_SDK or "
                            "PUSCH_E2E_LAUNCH_PATTERN / PARSER_E2E_TESTMAC_YAML)");
        return;
    }

    for (auto _ : state)
    {
        for (const nv::phy_mac_msg_desc* msg : in.messages)
        {
            MockUlModuleView view{};
            view.carrier_id_val_ = msg->cell_id;
            scf_5g_fapi::ULSlotProcessor<MockUlModuleView> processor{view};
            auto result = processor.process<UL_TTI_PDU_TYPE_PUSCH>(std::span{msg, 1});
            benchmark::DoNotOptimize(result);
            benchmark::DoNotOptimize(view.slot_cmd_);
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(in.messages.size()));
}
BENCHMARK(BM_PuschParserNewPath);

void BM_PuschParserLegacyPath(benchmark::State& state)
{
    PuschBenchInputs in = gather_pusch_inputs();
    if (in.requests.empty())
    {
        state.SkipWithError("PUSCH data assets unavailable (set cuBB_SDK or "
                            "PUSCH_E2E_LAUNCH_PATTERN / PARSER_E2E_TESTMAC_YAML)");
        return;
    }

    for (auto _ : state)
    {
        for (const LegacyRequest& lr : in.requests)
        {
            slot_command_api::cell_group_command grp{};
            slot_command_api::cell_sub_command   cell{};
            run_legacy_pusch(*lr.req, lr.cell_id, grp, cell);
            benchmark::DoNotOptimize(grp);
            benchmark::DoNotOptimize(cell);
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(in.requests.size()));
}
BENCHMARK(BM_PuschParserLegacyPath);

} // namespace
} // namespace cuphy_cp::tests

BENCHMARK_MAIN();
