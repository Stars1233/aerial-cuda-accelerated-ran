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
 * @file test_tti_dispatch.cpp
 * @brief Unit tests for TtiDispatch and DLSlotProcessor.
 *
 * Covers:
 *   Group A — TtiDispatch::dispatch_pdus: routing and unknown-type handling
 *             (table-driven; 3-parser dispatch throughout)
 *   Group B — TtiDispatch::dispatch_pdus_typed: active-type filtering
 *             (table-driven; two sub-tests by active-type set)
 *   Group C — DLSlotProcessor::process: null buffer, empty payload, batch
 *             first-failure-wins, and sfn/slot propagation in error payloads
 *   Group D — TtiDispatch: per-parser body-size guard (A-1 fix)
 *             pdu_size < sizeof(header)+sizeof(pdu_t) → rejected before deref
 *   Group E — DLSlotProcessor: extract_req msg_len guard (B-4 fix)
 *             msg_len too small → nullptr → NullBuffer error
 *
 * Invariants verified in every table-driven case:
 *   Group A: processed + skipped_unknown == total PDUs in payload
 *            sum(per_parser_counts) == processed
 *   Group B: skipped_unknown == 0 for all non-malformed inputs
 *            (dispatch_pdus_typed silently skips inactive/unknown types;
 *             skipped_unknown only fires on malformed PDU headers)
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#include "scf_5g_fapi_dl_stats.hpp"
#include "scf_5g_fapi_tti_dispatch.hpp"
#include "scf_5g_fapi_dl_slot_processor.hpp"
#include "scf_5g_fapi_slot_concepts.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "scf_5g_fapi.h"

namespace {

// ---------------------------------------------------------------------------
// Minimal stub parsers (satisfy PduParser<scf_fapi_dl_tti_pdu_type_t>)
// ---------------------------------------------------------------------------

/// Minimal PDU body type — one byte is enough for pointer validity.
/// Kept trivial (no default member initializer) so it satisfies
/// aerial::casts::assume_cast's is_trivial constraint, matching the real
/// (packed, trivial) FAPI pdu_t types the dispatch template casts to.
struct MinPdu
{
    uint8_t dummy;
};

/// Counting parser for DL_TTI_PDU_TYPE_PDSCH.
struct PdschCountParser
{
    using pdu_t = MinPdu;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_PDSCH;

    uint16_t count{};

    [[nodiscard]] bool parse(uint16_t /*sfn*/, uint16_t /*slot*/,
                             const pdu_t& /*pdu*/) noexcept
    {
        ++count;
        return true;
    }
};

/// Counting parser for DL_TTI_PDU_TYPE_PDCCH.
struct PdcchCountParser
{
    using pdu_t = MinPdu;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_PDCCH;

    uint16_t count{};

    [[nodiscard]] bool parse(uint16_t /*sfn*/, uint16_t /*slot*/,
                             const pdu_t& /*pdu*/) noexcept
    {
        ++count;
        return true;
    }
};

/// Counting parser for DL_TTI_PDU_TYPE_CSI_RS.
struct CsiRsCountParser
{
    using pdu_t = MinPdu;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_CSI_RS;

    uint16_t count{};

    [[nodiscard]] bool parse(uint16_t /*sfn*/, uint16_t /*slot*/,
                             const pdu_t& /*pdu*/) noexcept
    {
        ++count;
        return true;
    }
};

// ---------------------------------------------------------------------------
// Shared 3-parser dispatch type (Groups A and B)
//   Index 0 = PDSCH, 1 = PDCCH, 2 = CSI_RS
// ---------------------------------------------------------------------------

using Dispatch3 = scf_5g_fapi::TtiDispatch<
    scf_fapi_dl_tti_pdu_type_t,
    PdschCountParser,
    PdcchCountParser,
    CsiRsCountParser>;

// ---------------------------------------------------------------------------
// PDU payload builder helpers
// ---------------------------------------------------------------------------

/// Build a minimal PDU entry: scf_fapi_generic_pdu_info_t header + 1 byte body.
/// pdu_size covers the full entry (header + MinPdu).
std::vector<uint8_t> make_minimal_pdu(scf_fapi_dl_tti_pdu_type_t pdu_type)
{
    std::vector<uint8_t> buf(sizeof(scf_fapi_generic_pdu_info_t) + sizeof(MinPdu), 0u);
    auto* hdr = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(buf.data());
    hdr->pdu_type = static_cast<uint16_t>(pdu_type);
    hdr->pdu_size = static_cast<uint16_t>(buf.size());
    return buf;
}

/// Build a concatenated PDU payload from a vector of PDU types.
std::vector<uint8_t> build_payload(
    const std::vector<scf_fapi_dl_tti_pdu_type_t>& types)
{
    std::vector<uint8_t> result;
    for (auto t : types) {
        auto pdu = make_minimal_pdu(t);
        result.insert(result.end(), pdu.begin(), pdu.end());
    }
    return result;
}

/// Full DL_TTI FAPI message buffer (scf_fapi_header_t +
/// scf_fapi_dl_tti_req_t fixed header + PDU payload) with a filled-in
/// phy_mac_msg_desc pointing into it.
struct FapiDlMsg
{
    std::vector<uint8_t>  buf;
    nv::phy_mac_msg_desc  desc{};

    FapiDlMsg(uint16_t sfn, uint16_t slot, uint16_t num_pdus,
              const std::vector<uint8_t>& pdu_payload)
    {
        const std::size_t hdr_sz = sizeof(scf_fapi_header_t);
        const std::size_t req_sz = sizeof(scf_fapi_dl_tti_req_t);
        buf.resize(hdr_sz + req_sz + pdu_payload.size(), 0u);

        auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(buf.data() + hdr_sz);
        req->sfn      = sfn;
        req->slot     = slot;
        req->num_pdus = num_pdus;

        std::memcpy(buf.data() + hdr_sz + req_sz,
                    pdu_payload.data(), pdu_payload.size());

        desc.msg_buf  = buf.data();
        desc.msg_len  = static_cast<uint32_t>(buf.size());
        desc.cell_id  = 0;
    }
};

// ---------------------------------------------------------------------------
// Minimal DlModuleView mock for Group C (DLSlotProcessor error path tests)
// ---------------------------------------------------------------------------

/// Satisfies the DlModuleView concept without any PHY-specific logic.
/// Used only for DLSlotProcessor instantiation where no parser body executes
/// (null buffer or zero-PDU message).
struct NullModuleView
{
    mutable slot_command_api::slot_command       slot_cmd_{};
    mutable slot_command_api::cell_sub_command   cell_sub_cmd_{};
    mutable scf_5g_fapi::pm_weight_map_t         pm_map_{};
    mutable nv::phy_config_option                config_opt_{};
    mutable nv::phy_config                       phy_config_{};
    mutable nv::slot_limit_cell_error_t          limit_errors_{};

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }
    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t) const noexcept
    {
        return cell_sub_cmd_;
    }
    [[nodiscard]] slot_command_api::slot_info_t* order_sym_prb_info(uint32_t) const noexcept
    {
        return cell_sub_cmd_.sym_prb_info();
    }
    [[nodiscard]] scf_5g_fapi::pm_weight_map_t& pm_map() const noexcept { return pm_map_; }
    [[nodiscard]] bool pm_enabled()    const noexcept { return false; }
    [[nodiscard]] bool bf_enabled()    const noexcept { return false; }
    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t* bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept
    {
        return nullptr;
    }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return true; }
    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPdcchSlotNum() const noexcept { return -1; }
    [[nodiscard]] int staticPdschSlotNum() const noexcept { return -1; }
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 0u; }
    [[nodiscard]] int32_t  carrier_id(uint32_t)        const noexcept { return 0; }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t)       const noexcept { return 0u; }
    [[nodiscard]] const nv::phy_config& phy_config([[maybe_unused]] uint32_t cell_idx) const noexcept
    {
        return phy_config_;
    }
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }
    // P3 / P5 — required by extended DlModuleView concept.
    [[nodiscard]] uint16_t num_dl_prb(uint32_t)              const noexcept { return 100u; }
    [[nodiscard]] bool     is_fapi_to_cplane_direct_enabled() const noexcept { return false; }
    void publish_dl_pdsch_stats(const scf_5g_fapi::DlPdschStatsBatch&) const noexcept {}

    struct CellV
    {
        cuphyCellStatPrm_t stat_{};
        [[nodiscard]] uint16_t num_dl_prb() const noexcept { return 100u; }
        [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
        [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_; }
    };

    [[nodiscard]] CellV cell_view(uint32_t, const slot_command_api::slot_indication&) const noexcept
    {
        return CellV{};
    }
};

static_assert(scf_5g_fapi::DlModuleView<NullModuleView>,
              "NullModuleView must satisfy DlModuleView");

// ===========================================================================
// Group A — TtiDispatch::dispatch_pdus (table-driven, 3-parser dispatch)
//
// dispatch_pdus routes every PDU to the first parser whose pdu_type matches.
// PDUs whose type is not in any parser increment skipped_unknown.
//
// Invariants checked for every case:
//   processed + skipped_unknown == total PDUs in payload
//   sum(per_parser_counts[0..2]) == processed
// ===========================================================================

TEST(TtiDispatch, Routing_TableDriven)
{
    struct Case {
        std::string_view                        description;
        std::vector<scf_fapi_dl_tti_pdu_type_t> input_types;
        uint16_t                                expected_processed;
        uint16_t                                expected_skipped;
        // per_parser_counts: index 0 = PDSCH, 1 = PDCCH, 2 = CSI_RS
        std::array<uint16_t, 3>                 expected_counts;
    };

    static const Case k_cases[] = {
        {
            "single PDSCH → PDSCH parser hit, others zero",
            {DL_TTI_PDU_TYPE_PDSCH},
            1u, 0u, {1u, 0u, 0u},
        },
        {
            "single PDCCH → PDCCH parser hit, PDSCH zero",
            {DL_TTI_PDU_TYPE_PDCCH},
            1u, 0u, {0u, 1u, 0u},
        },
        {
            "single CSI_RS → CSI_RS parser hit, others zero",
            {DL_TTI_PDU_TYPE_CSI_RS},
            1u, 0u, {0u, 0u, 1u},
        },
        {
            "2×PDSCH + 1×PDCCH + 1×CSI_RS: each routed to correct parser",
            {DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_PDCCH,
             DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_CSI_RS},
            4u, 0u, {2u, 1u, 1u},
        },
        {
            "SSB (not in dispatch table) → skipped_unknown=1, processed=0",
            {DL_TTI_PDU_TYPE_SSB},
            0u, 1u, {0u, 0u, 0u},
        },
        {
            "PDSCH then SSB: first processed, second skipped",
            {DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_SSB},
            1u, 1u, {1u, 0u, 0u},
        },
        {
            "zero PDUs: all counters zero",
            {},
            0u, 0u, {0u, 0u, 0u},
        },
    };

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        Dispatch3 dispatch{PdschCountParser{}, PdcchCountParser{}, CsiRsCountParser{}};
        const auto payload = build_payload(tc.input_types);

        // Act
        const auto result = dispatch.dispatch_pdus(
            0u, 0u,
            std::span{payload},
            static_cast<uint16_t>(tc.input_types.size()));

        // Assert — per-case expectations
        EXPECT_EQ(result.processed,            tc.expected_processed);
        EXPECT_EQ(result.skipped_unknown,      tc.expected_skipped);
        EXPECT_EQ(result.per_parser_counts[0], tc.expected_counts[0]); // PDSCH
        EXPECT_EQ(result.per_parser_counts[1], tc.expected_counts[1]); // PDCCH
        EXPECT_EQ(result.per_parser_counts[2], tc.expected_counts[2]); // CSI_RS

        // Assert — structural invariants
        EXPECT_EQ(result.processed + result.skipped_unknown,
                  static_cast<uint16_t>(tc.input_types.size()))
            << "processed + skipped_unknown must equal total PDUs in payload";
        EXPECT_EQ(result.per_parser_counts[0]
                + result.per_parser_counts[1]
                + result.per_parser_counts[2],
                  result.processed)
            << "sum of per-parser counts must equal processed";
    }
}

// ===========================================================================
// Group B — TtiDispatch::dispatch_pdus_typed (table-driven, 3-parser dispatch)
//
// dispatch_pdus_typed<ActiveTypes...> only dispatches PDUs whose type is in
// ActiveTypes.  Crucially:
//
//   - PDUs whose type is NOT in ActiveTypes are silently skipped (cursor
//     advances, no counter is touched) — regardless of whether the type is
//     in the dispatch table.  This differs from dispatch_pdus where unknown
//     types increment skipped_unknown.
//   - skipped_unknown is only incremented for malformed PDU headers.
//     All well-formed cases below must therefore have skipped_unknown == 0.
//
// Two sub-tests cover different active-type sets.
// ===========================================================================

TEST(TtiDispatch, TypedDispatch_PdschActiveOnly_TableDriven)
{
    struct Case {
        std::string_view                        description;
        std::vector<scf_fapi_dl_tti_pdu_type_t> input_types;
        uint16_t                                expected_processed;
        // per_parser_counts: index 0 = PDSCH, 1 = PDCCH, 2 = CSI_RS
        std::array<uint16_t, 3>                 expected_counts;
    };

    static const Case k_cases[] = {
        {
            "PDSCH payload → active type matched, processed=1",
            {DL_TTI_PDU_TYPE_PDSCH},
            1u, {1u, 0u, 0u},
        },
        {
            "PDCCH payload → known type but not active: silently skipped, processed=0",
            {DL_TTI_PDU_TYPE_PDCCH},
            0u, {0u, 0u, 0u},
        },
        {
            "SSB payload → unknown type silently skipped (same as inactive), processed=0",
            {DL_TTI_PDU_TYPE_SSB},
            0u, {0u, 0u, 0u},
        },
        {
            "2×PDSCH + PDCCH(inactive): PDSCH processed, PDCCH silently skipped",
            {DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_PDCCH, DL_TTI_PDU_TYPE_PDSCH},
            2u, {2u, 0u, 0u},
        },
        {
            "CSI_RS only (inactive): silently skipped, processed=0",
            {DL_TTI_PDU_TYPE_CSI_RS},
            0u, {0u, 0u, 0u},
        },
    };

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        Dispatch3 dispatch{PdschCountParser{}, PdcchCountParser{}, CsiRsCountParser{}};
        const auto payload = build_payload(tc.input_types);

        // Act
        const auto result = dispatch.dispatch_pdus_typed<DL_TTI_PDU_TYPE_PDSCH>(
            0u, 0u,
            std::span{payload},
            static_cast<uint16_t>(tc.input_types.size()));

        // Assert
        EXPECT_EQ(result.processed,            tc.expected_processed);
        // dispatch_pdus_typed never increments skipped_unknown for inactive/unknown
        // types — only for malformed headers.  All cases above have well-formed PDUs.
        EXPECT_EQ(result.skipped_unknown,      0u);
        EXPECT_EQ(result.per_parser_counts[0], tc.expected_counts[0]); // PDSCH
        EXPECT_EQ(result.per_parser_counts[1], tc.expected_counts[1]); // PDCCH (always 0)
        EXPECT_EQ(result.per_parser_counts[2], tc.expected_counts[2]); // CSI_RS (always 0)
    }
}

TEST(TtiDispatch, TypedDispatch_PdschAndCsiRsActive_TableDriven)
{
    struct Case {
        std::string_view                        description;
        std::vector<scf_fapi_dl_tti_pdu_type_t> input_types;
        uint16_t                                expected_processed;
        // per_parser_counts: index 0 = PDSCH, 1 = PDCCH, 2 = CSI_RS
        std::array<uint16_t, 3>                 expected_counts;
    };

    static const Case k_cases[] = {
        {
            "PDSCH+CSI_RS: both active, both processed",
            {DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_CSI_RS},
            2u, {1u, 0u, 1u},
        },
        {
            "PDCCH only (inactive): silently skipped, processed=0",
            {DL_TTI_PDU_TYPE_PDCCH},
            0u, {0u, 0u, 0u},
        },
        {
            "PDSCH + PDCCH(inactive) + CSI_RS: inactive silently skipped",
            {DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_PDCCH, DL_TTI_PDU_TYPE_CSI_RS},
            2u, {1u, 0u, 1u},
        },
        {
            "SSB (unknown) + PDSCH + CSI_RS: unknown silently skipped",
            {DL_TTI_PDU_TYPE_SSB, DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_CSI_RS},
            2u, {1u, 0u, 1u},
        },
    };

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        Dispatch3 dispatch{PdschCountParser{}, PdcchCountParser{}, CsiRsCountParser{}};
        const auto payload = build_payload(tc.input_types);

        // Act
        const auto result =
            dispatch.dispatch_pdus_typed<DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_CSI_RS>(
                0u, 0u,
                std::span{payload},
                static_cast<uint16_t>(tc.input_types.size()));

        // Assert
        EXPECT_EQ(result.processed,            tc.expected_processed);
        EXPECT_EQ(result.skipped_unknown,      0u); // inactive/unknown silently skipped
        EXPECT_EQ(result.per_parser_counts[0], tc.expected_counts[0]); // PDSCH
        EXPECT_EQ(result.per_parser_counts[1], tc.expected_counts[1]); // PDCCH (always 0)
        EXPECT_EQ(result.per_parser_counts[2], tc.expected_counts[2]); // CSI_RS
    }
}

TEST(TtiDispatch, TypedDispatch_PdschAndPdcchUseStandardParserSignature)
{
    Dispatch3 dispatch{PdschCountParser{}, PdcchCountParser{}, CsiRsCountParser{}};
    const auto payload = build_payload({DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_PDCCH});

    const auto result =
        dispatch.dispatch_pdus_typed<DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_PDCCH>(
            0u, 0u,
            std::span{payload},
            2u);

    EXPECT_EQ(result.processed, 2u);
    EXPECT_EQ(result.per_parser_counts[0], 1u); // PDSCH legacy 3-arg parser
    EXPECT_EQ(result.per_parser_counts[1], 1u); // PDCCH cell-aware parser

    const auto& pdsch = dispatch.get_parser<PdschCountParser>();
    const auto& pdcch = dispatch.get_parser<PdcchCountParser>();
    EXPECT_EQ(pdsch.count, 1u);
    EXPECT_EQ(pdcch.count, 1u);
}

// ===========================================================================
// Group C — DLSlotProcessor::process (table-driven)
//
// Verifies the outer message-loop error paths and success paths.
//
// Invariants checked for every case:
//   - On success: result.has_value() is true.
//   - On failure: error().code, error().sfn, and error().slot are all
//     asserted (not just has_value()), confirming sfn/slot propagation.
//
// Scenarios covered:
//   NullBuf        — single message with null msg_buf
//   EmptySpan      — no messages at all (loop never executes)
//   EmptyPayload   — single message, zero PDUs (two sfn/slot variants)
//   TwoEmpty       — two empty messages, both skipped, batch succeeds
//   SecondNull     — two messages: first empty succeeds, second null fails
//                    (first-failure-wins: error carries sfn=0, slot=0)
// #ifdef SCF_FAPI_10_04 only:
//   CountMismatch  — nPDUsOfEachType[PDSCH]=1 but num_pdus=0; error carries
//                    the actual sfn/slot from the message header
// ===========================================================================

TEST(DLSlotProcessor, Process_TableDriven)
{
    enum class BatchSetup : uint8_t {
        NullBuf,       ///< Single msg with null msg_buf.
        EmptySpan,     ///< No messages at all.
        EmptyPayload,  ///< Single msg, zero PDUs; sfn/slot from tc.sfn/tc.slot.
        TwoEmpty,      ///< Two msgs, both empty payloads.
        SecondNull,    ///< Two msgs: first empty, second null.
#ifdef SCF_FAPI_10_04
        CountMismatch, ///< nPDUsOfEachType[PDSCH]=1 but num_pdus=0.
#endif
    };

    struct Case {
        std::string_view             description;
        BatchSetup                   setup;
        uint16_t                     sfn;            ///< sfn for primary message.
        uint16_t                     slot;           ///< slot for primary message.
        bool                         expected_ok;
        scf_5g_fapi::SlotParseError::Code expected_code;  ///< Checked when !expected_ok.
        uint16_t                     expected_sfn;   ///< Checked when !expected_ok.
        uint16_t                     expected_slot;  ///< Checked when !expected_ok.
    };

    static const std::array k_cases = std::to_array<Case>({
        {
            "null msg_buf → NullBuffer error; sfn and slot are 0 (no req to read)",
            BatchSetup::NullBuf, 0u, 0u,
            false, scf_5g_fapi::SlotParseError::Code::NullBuffer, 0u, 0u,
        },
        {
            "empty span → success (loop never executes)",
            BatchSetup::EmptySpan, 0u, 0u,
            true, {}, 0u, 0u,
        },
        {
            "single msg zero PDUs sfn=0 slot=0 → success",
            BatchSetup::EmptyPayload, 0u, 0u,
            true, {}, 0u, 0u,
        },
        {
            "single msg zero PDUs sfn=1023 slot=19 → success (boundary sfn/slot)",
            BatchSetup::EmptyPayload, 1023u, 19u,
            true, {}, 0u, 0u,
        },
        {
            "two empty msgs → both skipped, batch succeeds",
            BatchSetup::TwoEmpty, 0u, 0u,
            true, {}, 0u, 0u,
        },
        {
            "two msgs: first empty ok, second null → NullBuffer; sfn=0 slot=0",
            BatchSetup::SecondNull, 0u, 0u,
            false, scf_5g_fapi::SlotParseError::Code::NullBuffer, 0u, 0u,
        },
#ifdef SCF_FAPI_10_04
        {
            "nPDUsOfEachType[PDSCH]=1 vs num_pdus=0 sfn=42 slot=3 → PduCountMismatch",
            BatchSetup::CountMismatch, 42u, 3u,
            false, scf_5g_fapi::SlotParseError::Code::PduCountMismatch, 42u, 3u,
        },
#endif
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        NullModuleView view{};
        scf_5g_fapi::DLSlotProcessor<NullModuleView> processor{view};

        FapiDlMsg            msg_a{tc.sfn, tc.slot, 0u, {}};  // primary message
        FapiDlMsg            msg_b{0u, 0u, 0u, {}};           // secondary (TwoEmpty/SecondNull)
        nv::phy_mac_msg_desc null_desc{};                      // msg_buf zero-initialised = nullptr

#ifdef SCF_FAPI_10_04
        // For CountMismatch: declare one PDSCH PDU in nPDUsOfEachType while
        // leaving num_pdus=0 so validate_sum() returns false.
        if (tc.setup == BatchSetup::CountMismatch)
        {
            auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(
                msg_a.buf.data() + sizeof(scf_fapi_header_t));
            req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
        }
#endif

        // Act — lambda builds the span for each setup variant.
        const scf_5g_fapi::SlotParseResult result = [&]() -> scf_5g_fapi::SlotParseResult
        {
            switch (tc.setup)
            {
            case BatchSetup::NullBuf:
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&null_desc, 1u});
            case BatchSetup::EmptySpan:
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(
                    std::span<const nv::phy_mac_msg_desc>{});
            case BatchSetup::EmptyPayload:
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg_a.desc, 1u});
            case BatchSetup::TwoEmpty:
            {
                const std::array descs{msg_a.desc, msg_b.desc};
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(
                    std::span{descs.data(), descs.size()});
            }
            case BatchSetup::SecondNull:
            {
                const std::array descs{msg_a.desc, null_desc};
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(
                    std::span{descs.data(), descs.size()});
            }
#ifdef SCF_FAPI_10_04
            case BatchSetup::CountMismatch:
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg_a.desc, 1u});
#endif
            default:
                return {};
            }
        }();

        // Assert
        if (tc.expected_ok)
        {
            EXPECT_TRUE(result.has_value());
        }
        else
        {
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code, tc.expected_code);
            EXPECT_EQ(result.error().sfn,  tc.expected_sfn);
            EXPECT_EQ(result.error().slot, tc.expected_slot);
        }
    }
}

// ===========================================================================
// Group D — TtiDispatch: per-parser body-size guard (A-1 fix)
//
// The new guard inside dispatch_one rejects any PDU whose
// pdu_size < sizeof(generic_header) + sizeof(pdu_t) before the reinterpret_cast.
// Guard 2 (outer) still passes because pdu_size >= sizeof(generic_header).
//
// All cases: pdu_size == sizeof(scf_fapi_generic_pdu_info_t) (4 bytes).
// k_min_size for Dispatch3 parsers = 4 + sizeof(MinPdu) = 4 + 1 = 5.
// Expected: processed == 0 (PDU rejected before body dereference).
// ===========================================================================

TEST(TtiDispatch, BodySizeGuard_PduSizeBelowMinSizeof_Rejected)
{
    struct Case {
        std::string_view              description;
        scf_fapi_dl_tti_pdu_type_t    pdu_type;
        uint16_t                      pdu_size;   ///< set on the generic PDU header
        uint16_t                      expected_processed;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "PDSCH, pdu_size = sizeof(generic_header) only → body-size guard fires",
            DL_TTI_PDU_TYPE_PDSCH,
            static_cast<uint16_t>(sizeof(scf_fapi_generic_pdu_info_t)),
            0u,
        },
        {
            "PDCCH, pdu_size = sizeof(generic_header) only → body-size guard fires",
            DL_TTI_PDU_TYPE_PDCCH,
            static_cast<uint16_t>(sizeof(scf_fapi_generic_pdu_info_t)),
            0u,
        },
        {
            "PDSCH, pdu_size = sizeof(generic_header) + sizeof(MinPdu) → guard passes",
            DL_TTI_PDU_TYPE_PDSCH,
            static_cast<uint16_t>(sizeof(scf_fapi_generic_pdu_info_t) + sizeof(MinPdu)),
            1u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Build a PDU buffer that is large enough for the actual body bytes,
        // but set pdu_size to tc.pdu_size to trigger (or not) the guard.
        auto payload = make_minimal_pdu(tc.pdu_type);
        auto* hdr = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(payload.data());
        hdr->pdu_size = tc.pdu_size;

        Dispatch3 dispatch{PdschCountParser{}, PdcchCountParser{}, CsiRsCountParser{}};
        const auto result = dispatch.dispatch_pdus(0u, 0u, std::span{payload}, 1u);

        EXPECT_EQ(result.processed, tc.expected_processed);
        EXPECT_EQ(result.per_parser_counts[0] + result.per_parser_counts[1]
                  + result.per_parser_counts[2],
                  tc.expected_processed);
    }
}

// ===========================================================================
// Group E — DLSlotProcessor: extract_req msg_len guard (B-4 fix)
//
// extract_req<T>() returns nullptr when msg_len < sizeof(header) + sizeof(T).
// DLSlotProcessor::process() maps this to SlotParseError::Code::NullBuffer.
//
// Cases: msg_len exactly 0, msg_len = sizeof(scf_fapi_header_t) - 1,
//        msg_len = minimum - 1, msg_len = minimum (boundary, must succeed).
// ===========================================================================

TEST(DLSlotProcessor, ExtractReq_ShortMsgLen_ReturnsNullBuffer)
{
    using Code = scf_5g_fapi::SlotParseError::Code;

    static constexpr uint32_t k_min_len =
        static_cast<uint32_t>(sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_tti_req_t));

    struct Case {
        std::string_view description;
        uint32_t         msg_len;
        bool             expected_ok;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "msg_len == 0 → nullptr → NullBuffer",
            0u, false,
        },
        {
            "msg_len == sizeof(fapi_header) - 1 → nullptr → NullBuffer",
            static_cast<uint32_t>(sizeof(scf_fapi_header_t)) - 1u, false,
        },
        {
            "msg_len == k_min_len - 1 → nullptr → NullBuffer",
            k_min_len - 1u, false,
        },
        {
            "msg_len == k_min_len (empty payload) → req valid; empty span → ok",
            k_min_len, true,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        NullModuleView view{};
        scf_5g_fapi::DLSlotProcessor<NullModuleView> processor{view};

        FapiDlMsg msg{0u, 0u, 0u, {}};
        msg.desc.msg_len = tc.msg_len;

        const auto result =
            processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg.desc, 1u});

        if (tc.expected_ok) {
            EXPECT_TRUE(result.has_value());
        } else {
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code, Code::NullBuffer);
        }
    }
}

} // namespace
