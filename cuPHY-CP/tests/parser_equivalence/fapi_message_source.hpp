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

#ifndef CUPHY_CP_TESTS_FAPI_MESSAGE_SOURCE_HPP_
#define CUPHY_CP_TESTS_FAPI_MESSAGE_SOURCE_HPP_

/**
 * @file fapi_message_source.hpp
 * @brief Clean boundary between the testMAC FAPI generator and the
 *        scf_5g_fapi parser under test.
 *
 * The testMAC FAPI headers (fapi_defines.hpp / common_defines.hpp) declare a
 * global @c prb_info_t, a global @c slot_indication and a @c SLOTS_PER_FRAME
 * macro that collide with the scf_5g_fapi slot-command headers
 * (@c slot_command_api::prb_info_t, @c slot_command_api::slot_indication, ...).
 * The two header families therefore cannot be compiled in the same translation
 * unit. This interface hides every testMAC type behind a pimpl so the
 * parser-side TUs (functional test + benchmark) consume only raw FAPI byte
 * buffers -- exactly the boundary that exists in production between the MAC and
 * the L2 adapter.
 *
 * Implementation lives in fapi_message_source.cpp, which is the only TU that
 * includes the testMAC headers.
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include "nv_phy_mac_transport.hpp"  // nv::phy_mac_msg_desc (lightweight nvPHY header)

namespace cuphy_cp::tests
{

/**
 * @brief One prebuilt FAPI message plus the slot it belongs to.
 *
 * @c desc.msg_buf points into testMAC-owned memory that stays valid for the
 * lifetime of the @ref FapiMessageSource that produced this descriptor.
 */
struct SourceMessage final
{
    nv::phy_mac_msg_desc desc{};
    uint32_t             sfn_slot{0};  ///< Packed sfn_slot_t.u32, for the PDU walker + diagnostics.
};

/**
 * @brief Opaque, move-only owner of a testMAC session whose FAPI messages have
 *        been prebuilt and flattened into raw descriptors.
 *
 * Construct via @ref open_pusch. All testMAC types are confined to the
 * implementation TU; consumers see only @ref SourceMessage / @c phy_mac_msg_desc.
 *
 * @thread_safety Not thread-safe; intended for single-threaded test/benchmark
 *                setup and iteration.
 */
class FapiMessageSource final
{
public:
    /// @brief Move ctor; transfers ownership of the testMAC session.
    FapiMessageSource(FapiMessageSource&&) noexcept;
    /// @brief Move assignment; transfers ownership of the testMAC session.
    FapiMessageSource& operator=(FapiMessageSource&&) noexcept;
    FapiMessageSource(const FapiMessageSource&)            = delete;
    FapiMessageSource& operator=(const FapiMessageSource&) = delete;
    /// @brief Destroys the owned testMAC session and its prebuilt messages.
    ~FapiMessageSource();

    /**
     * @brief Open a session loaded with the default (or env-overridden) PUSCH
     *        launch pattern and prebuild its FAPI messages.
     * @return A ready source, or std::nullopt when the data assets are absent or
     *         any construction/load/prebuild step fails.
     */
    [[nodiscard]] static std::optional<FapiMessageSource> open_pusch();

    /// @brief All prebuilt messages for the active cell, flattened across all
    ///        scheduled slots.
    /// @return Span valid for the lifetime of this object.
    [[nodiscard]] std::span<const SourceMessage> messages() const noexcept;

    /// @brief Default PUSCH launch-pattern file name (for skip diagnostics).
    /// @return Null-terminated pattern file name; never nullptr.
    [[nodiscard]] static const char* pusch_default_pattern() noexcept;

private:
    class Impl;
    explicit FapiMessageSource(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace cuphy_cp::tests

#endif // CUPHY_CP_TESTS_FAPI_MESSAGE_SOURCE_HPP_
