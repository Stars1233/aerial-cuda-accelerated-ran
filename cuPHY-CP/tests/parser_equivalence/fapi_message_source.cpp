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
 * @file fapi_message_source.cpp
 * @brief testMAC-side implementation of @ref cuphy_cp::tests::FapiMessageSource.
 *
 * This is the ONLY translation unit that includes the testMAC FAPI headers
 * (via parser_equivalence_harness.hpp). It must not include the scf_5g_fapi
 * slot-command headers (scf_5g_slot_commands.hpp, scf_5g_fapi_ul_slot_processor.hpp,
 * ...): the two header families declare clashing global names. See
 * fapi_message_source.hpp for the rationale.
 */

#include "fapi_message_source.hpp"

#include <utility>
#include <vector>

#include "parser_equivalence_harness.hpp"  // TestMacSlotSession + asset resolution (testMAC side)

namespace cuphy_cp::tests
{
namespace
{

// PUSCH data-selection constants (formerly pusch_equivalence_fixture.hpp). Kept
// in the implementation TU so the public header stays free of testMAC details.

/// Default single-cell launch pattern shipped under testVectors/multi-cell/.
/// It schedules many channels, but kPuschChannelMask keeps only PUSCH for every
/// slot, so the parser sees PUSCH PDUs on each UL slot the pattern defines.
constexpr const char* kPuschDefaultPattern = "launch_pattern_F08_1C_59.yaml";

/// Env-var overrides for custom CI vectors.
constexpr const char* kEnvTestmacYaml   = "PARSER_E2E_TESTMAC_YAML";
constexpr const char* kEnvLaunchPattern = "PUSCH_E2E_LAUNCH_PATTERN";

/// Activate cell 0 only.
constexpr uint64_t kPuschCellMask = 0x1ULL;

/// Schedule PUSCH only (channel_type_t::PUSCH == bit 0).
constexpr uint32_t kPuschChannelMask = 1u;

/// Cell exercised by the default single-cell launch pattern.
constexpr int kCellId = 0;

} // namespace

/// Owns the testMAC session and the flattened raw descriptors it produced.
class FapiMessageSource::Impl final
{
public:
    explicit Impl(TestMacSlotSession session) : session_{std::move(session)}
    {
        session_.for_each_slot_message(kCellId,
            [this](sfn_slot_t ss, const nv::phy_mac_msg_desc& msg) {
                messages_.push_back(SourceMessage{msg, ss.u32});
            });
    }

    [[nodiscard]] std::span<const SourceMessage> messages() const noexcept { return messages_; }

private:
    TestMacSlotSession         session_;
    std::vector<SourceMessage> messages_{};
};

FapiMessageSource::FapiMessageSource(std::unique_ptr<Impl> impl) noexcept : impl_{std::move(impl)} {}

FapiMessageSource::FapiMessageSource(FapiMessageSource&&) noexcept            = default;
FapiMessageSource& FapiMessageSource::operator=(FapiMessageSource&&) noexcept = default;
FapiMessageSource::~FapiMessageSource()                                       = default;

std::optional<FapiMessageSource> FapiMessageSource::open_pusch()
{
    std::optional<TestMacSlotSession> session =
        TestMacSlotSession::try_open(resolve_testmac_yaml(kEnvTestmacYaml),
                                     resolve_launch_pattern(kEnvLaunchPattern, kPuschDefaultPattern),
                                     kPuschCellMask,
                                     kPuschChannelMask);
    if (!session.has_value())
    {
        return std::nullopt;
    }
    return FapiMessageSource{std::make_unique<Impl>(std::move(*session))};
}

std::span<const SourceMessage> FapiMessageSource::messages() const noexcept
{
    return impl_->messages();
}

const char* FapiMessageSource::pusch_default_pattern() noexcept
{
    return kPuschDefaultPattern;
}

} // namespace cuphy_cp::tests
