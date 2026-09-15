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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <vector>

#include "aerial/casts/casts.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "nv_phy_mac_transport.hpp"
#include "scf_5g_fapi_pucch234_pdu_parser.hpp"
#include "scf_5g_fapi_pucch_pdu_parser.hpp"
#include "scf_5g_fapi_ul_slot_processor.hpp"

// Instrumentation recorded by the append_pucch_order_prbs() stub in test_stubs.cpp.
// The cuPHY param fill itself happens in pucch::populate_slot_command(), which is
// inline and therefore un-stubbable; tests assert that half directly against the
// slot command (see expect_single_pdu_accepted below).
namespace scf_5g_fapi::test_support
{
extern int      pucch_order_append_count;
extern int32_t  pucch_last_cell_index;
extern uint16_t pucch_last_cell_stat_prm_idx;
extern uint16_t pucch_last_ul_bandwidth;
extern uint16_t pucch_last_prb_size;
extern uint8_t  pucch_last_format;
extern uint16_t pucch_last_rnti;
extern bool     pucch_last_mmimo_enabled;
extern bool     pucch_last_slot_detail_present;
extern void reset_pucch_stub_state() noexcept;
} // namespace scf_5g_fapi::test_support

namespace
{

namespace pucch_test_defaults
{
constexpr uint16_t k_sfn                = 7u;
constexpr uint16_t k_slot               = 3u;
constexpr uint32_t k_local_cell_0       = 0u;
constexpr uint32_t k_local_cell_1       = 1u;
constexpr int32_t  k_carrier_id_base    = 10;   // carrier_id(cell_idx) = base + cell_idx
constexpr uint16_t k_cell_stat_idx_base = 41u;  // cell_stat_prm_idx(cell_idx) = base + cell_idx
constexpr uint16_t k_phy_cell_id_base   = 100u; // phy_cell_id(cell_idx) = base + cell_idx
constexpr int32_t  k_carrier_1          = k_carrier_id_base + static_cast<int32_t>(k_local_cell_1);
constexpr uint16_t k_cell_stat_1        = static_cast<uint16_t>(k_cell_stat_idx_base + k_local_cell_1);
constexpr uint16_t k_phy_cell_1         = static_cast<uint16_t>(k_phy_cell_id_base + k_local_cell_1);
constexpr uint16_t k_ul_bandwidth       = 273u;
constexpr uint16_t k_hopping_id         = 321u;
constexpr uint16_t k_rnti               = 0x1234u;
constexpr uint16_t k_prb_size           = 1u;
constexpr uint16_t k_prb_start          = 12u;
constexpr uint8_t  k_start_sym          = 2u;
constexpr uint8_t  k_num_sym            = 2u;
} // namespace pucch_test_defaults

struct MockCellView final
{
    cuphyCellStatPrm_t stat_prm_{};

    explicit MockCellView(uint16_t ul_prb = 273u) noexcept
    {
        stat_prm_.nPrbUlBwp = ul_prb;
    }

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return 0u; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_prm_; }
    // Consumed only by the PUSCH single-sector FH path, which the PUCCH parser never
    // exercises at runtime; present so the bundled PUSCH parser compiles against this mock.
    [[nodiscard]] uint8_t ul_start_symbol() const noexcept { return 0u; }
    [[nodiscard]] uint8_t ul_max_symbols(uint8_t fallback) const noexcept { return fallback; }
};

static_assert(scf_5g_fapi::CellView<MockCellView>);

struct MockUlModuleView final
{
    mutable slot_command_api::slot_command       slot_cmd_{};
    mutable slot_command_api::cell_sub_command   cell_sub_cmd_{};
    // Separate from cell_sub_cmd_.sym_prb_info(): the PUCCH parser routes its
    // Order PRB metadata to order_sym_prb_info() so the two destinations must
    // be distinct objects for a test to distinguish them (same fix as PUSCH/SRS mocks).
    mutable std::unique_ptr<slot_command_api::slot_info_t> order_sym_prbs_{
        std::make_unique<slot_command_api::slot_info_t>()};
    mutable nv::phy_config_option                config_opt_{};
    mutable nv::slot_limit_group_error_t         group_limit_errors_{};
    nv::pucch_dtx_t_list                         dtx_{};
    float                                        dtx_pusch_{0.0F};
    bool                                         mmimo_enabled_{true};
    bool                                         bf_enabled_{false};
    bool                                         group_command_null_{false};
    int32_t                                      carrier_id_base_{pucch_test_defaults::k_carrier_id_base};
    mutable uint32_t                             last_cell_sub_idx_{};
    mutable uint32_t                             last_cell_view_id_{};

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        if (group_command_null_) { return nullptr; }
        return &slot_cmd_.cell_groups;
    }

    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t cell_idx) const noexcept
    {
        last_cell_sub_idx_ = cell_idx;
        return cell_sub_cmd_;
    }

    [[nodiscard]] slot_command_api::slot_info_t* order_sym_prb_info(uint32_t) const noexcept
    {
        return order_sym_prbs_.get();
    }

    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }

    [[nodiscard]] bool bf_enabled() const noexcept { return bf_enabled_; }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPuschSlotNum() const noexcept { return -1; }
    [[nodiscard]] uint8_t lbrm() const noexcept { return 0u; }
    [[nodiscard]] const nv::pucch_dtx_t_list& dtx_thresholds() const noexcept { return dtx_; }
    [[nodiscard]] const float& dtx_thresholds_pusch() const noexcept { return dtx_pusch_; }
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t cell_idx) const noexcept
    {
        return static_cast<uint16_t>(pucch_test_defaults::k_cell_stat_idx_base + cell_idx);
    }
    [[nodiscard]] int32_t carrier_id(uint32_t cell_idx) const noexcept
    {
        return carrier_id_base_ + static_cast<int32_t>(cell_idx);
    }
    [[nodiscard]] nv::slot_limit_group_error_t& get_group_limit_errors() const noexcept
    {
        return group_limit_errors_;
    }

    [[nodiscard]] nv::phy_mac_transport& transport([[maybe_unused]] int carrier) const noexcept
    {
        // PUCCH parser must not reach the SRS transport accessor.
        // nv::phy_mac_transport has no default ctor, so a `static dummy{}` does not
        // compile; dereferencing nullptr to satisfy the return type is UB (LTO can
        // miscompile the surrounding code). Hard-fail instead — matches the PRACH
        // mock pattern in test_prach_pdu_parser.cpp.
        ADD_FAILURE() << "PUCCH MockUlModuleView::transport() must not be called "
                         "(SRS-only accessor; PUCCH parser never invokes it)";
        std::abort(); // [[noreturn]] — satisfies the non-void return type
    }

    [[nodiscard]] MockCellView cell_view(uint32_t cell_id,
        const slot_command_api::slot_indication&) const noexcept
    {
        last_cell_view_id_ = cell_id;
        return MockCellView{pucch_test_defaults::k_ul_bandwidth};
    }

    // UlModuleView concept stubs not exercised by the PUCCH parser path.
    mutable nv::slot_limit_cell_error_t cell_limit_errors_{};

    [[nodiscard]] uint16_t phy_cell_id(uint32_t cell_idx) const noexcept
    {
        return static_cast<uint16_t>(pucch_test_defaults::k_phy_cell_id_base + cell_idx);
    }
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return cell_limit_errors_;
    }
    [[nodiscard]] uint8_t indication_instances_per_slot(uint32_t) const noexcept { return 0u; }
    void send_fapi_error_indication(uint32_t,
                                    scf_fapi_message_id_e,
                                    scf_fapi_error_codes_t,
                                    uint16_t,
                                    uint16_t) const noexcept {}
    [[nodiscard]] bool    srs_enabled() const noexcept { return false; }
    [[nodiscard]] ru_type ru(uint32_t) const noexcept { return OTHER_MODE; }
    // PuschModuleView concept stubs — required because ULSlotProcessor bundles the
    // PUSCH parser for all V; the PUCCH parser path never exercises these.
    [[nodiscard]] bool enable_weighted_avg_cfo() const noexcept { return false; }
    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t*
    bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept { return nullptr; }
    [[nodiscard]] scf_5g_fapi::SrsChestBuffVerdict
    classify_srs_chest_buffer(uint32_t, const scf_fapi_srs_pdu_t&) const noexcept
    {
        return scf_5g_fapi::SrsChestBuffVerdict::Accept;
    }
    // Required by the UlModuleView concept (mode gate used by MR3 to suppress
    // legacy PUCCH replay).  PUCCH parser unit tests don't exercise direct
    // mode; presence-only to satisfy the concept.
    [[nodiscard]] bool fapi_to_cplane_direct_enabled() const noexcept { return false; }

    // ---- PrachModuleView refinement stubs --------------------------------
    // ULSlotProcessor<V> instantiates PrachPduParser<V>, which requires
    // PrachModuleView<V>. PUCCH parser never invokes these accessors; present
    // here as concept-only stubs (same ISP pattern as transport() above).
    nv::phy_config                       phy_config_obj_{};
    nv::prach_addln_config_t             addln_config_obj_{};

    [[nodiscard]] const nv::phy_config& phy_config(uint32_t) const noexcept
    {
        return phy_config_obj_;
    }
    [[nodiscard]] const nv::prach_addln_config_t& prach_addln_config(uint32_t) const noexcept
    {
        return addln_config_obj_;
    }
    [[nodiscard]] ru_type ru_type_for_cell(uint32_t) const noexcept { return OTHER_MODE; }
    [[nodiscard]] bool    is_fapi_to_cplane_direct() const noexcept { return false; }
};

static_assert(scf_5g_fapi::UlModuleView<MockUlModuleView>);

[[nodiscard]] scf_fapi_pucch_pdu_t make_pucch_pdu(const uint8_t format_type) noexcept
{
    scf_fapi_pucch_pdu_t pdu{};
    pdu.rnti = pucch_test_defaults::k_rnti;
    pdu.handle = 0xABCDEFu;
    pdu.bwp.bwp_start = 0u;
    pdu.bwp.bwp_size = pucch_test_defaults::k_ul_bandwidth;
    pdu.format_type = format_type;
    pdu.prb_start = pucch_test_defaults::k_prb_start;
    pdu.prb_size = pucch_test_defaults::k_prb_size;
    pdu.start_symbol_index = pucch_test_defaults::k_start_sym;
    pdu.num_of_symbols = pucch_test_defaults::k_num_sym;
    pdu.hopping_id = pucch_test_defaults::k_hopping_id;
    pdu.bit_len_harq = 1u;
    pdu.sr_flag = 1u;
    return pdu;
}

constexpr std::array<uint8_t, 5> k_all_pucch_formats{
    static_cast<uint8_t>(UL_TTI_PUCCH_FORMAT_0),
    static_cast<uint8_t>(UL_TTI_PUCCH_FORMAT_1),
    static_cast<uint8_t>(UL_TTI_PUCCH_FORMAT_2),
    static_cast<uint8_t>(UL_TTI_PUCCH_FORMAT_3),
    static_cast<uint8_t>(UL_TTI_PUCCH_FORMAT_4)};

// Per-format UCI counters live in separate grp_dyn_pars fields; map format ->
// counter so callers can stay format-agnostic.
[[nodiscard]] uint16_t uci_count_for_format(const slot_command_api::pucch_params& params,
                                           const uint8_t format_type)
{
    const auto& grp = params.grp_dyn_pars;
    switch (format_type)
    {
        case UL_TTI_PUCCH_FORMAT_0: return grp.nF0Ucis;
        case UL_TTI_PUCCH_FORMAT_1: return grp.nF1Ucis;
        case UL_TTI_PUCCH_FORMAT_2: return grp.nF2Ucis;
        case UL_TTI_PUCCH_FORMAT_3: return grp.nF3Ucis;
        case UL_TTI_PUCCH_FORMAT_4: return grp.nF4Ucis;
        default:
            ADD_FAILURE() << "unexpected PUCCH format_type "
                          << static_cast<unsigned>(format_type);
            return 0u;
    }
}

// Asserts that exactly one PUCCH PDU of format_type was accepted for local cell 1.
// Checks the cuPHY params the parser actually produced (populate_slot_command is
// inline, so there is no stub to count), plus the pass-through arguments recorded
// by the append_pucch_order_prbs seam.
void expect_single_pdu_accepted(const MockUlModuleView& view, const uint8_t format_type)
{
    using namespace scf_5g_fapi::test_support;

    // Cell resolution: parser indexed the mock with the local cell and stamped the
    // PHY cell id onto the cell sub-command.
    EXPECT_EQ(view.last_cell_sub_idx_, pucch_test_defaults::k_local_cell_1);
    EXPECT_EQ(view.last_cell_view_id_, pucch_test_defaults::k_local_cell_1);
    EXPECT_EQ(view.cell_sub_cmd_.cell, pucch_test_defaults::k_phy_cell_1);
    EXPECT_EQ(view.cell_sub_cmd_.slot.type, SLOT_UPLINK);
    EXPECT_EQ(view.cell_sub_cmd_.slot.slot_3gpp.sfn_, pucch_test_defaults::k_sfn);
    EXPECT_EQ(view.cell_sub_cmd_.slot.slot_3gpp.slot_, pucch_test_defaults::k_slot);

    auto* const params = view.slot_cmd_.cell_groups.get_pucch_params();
    ASSERT_NE(params, nullptr);
    const auto& grp = params->grp_dyn_pars;

    // One cell registered, one UCI of this format, and no other format touched.
    EXPECT_EQ(grp.nCells, 1u);
    ASSERT_EQ(params->cell_index_list.size(), 1u);
    EXPECT_EQ(params->cell_index_list[0], pucch_test_defaults::k_carrier_1);
    ASSERT_EQ(params->phy_cell_index_list.size(), 1u);
    EXPECT_EQ(params->phy_cell_index_list[0], pucch_test_defaults::k_phy_cell_1);
    ASSERT_EQ(uci_count_for_format(*params, format_type), 1u);
    for (const uint8_t other : k_all_pucch_formats)
    {
        if (other == format_type)
        {
            continue;
        }
        EXPECT_EQ(uci_count_for_format(*params, other), 0u)
            << "format " << static_cast<unsigned>(other) << " counter was touched";
    }

    // Per-cell dynamic params: hopping id and cell static index are forwarded here,
    // which is why the stub no longer needs to capture pucch_hopping_id.
    EXPECT_EQ(params->dyn_pars[0].cellPrmStatIdx, pucch_test_defaults::k_cell_stat_1);
    EXPECT_EQ(params->dyn_pars[0].pucchHoppingId, pucch_test_defaults::k_hopping_id);
    EXPECT_EQ(params->dyn_pars[0].cellPrmDynIdx, 0u);
    // staticPuschSlotNum() is -1 in the mock, so slotNum falls back to the slot.
    EXPECT_EQ(params->dyn_pars[0].slotNum, pucch_test_defaults::k_slot);

    // The UCI entry itself.
    const auto& uci = params->params[format_type][0];
    EXPECT_EQ(uci.uciOutputIdx, 0u);
    EXPECT_EQ(uci.formatType, format_type);
    EXPECT_EQ(uci.rnti, pucch_test_defaults::k_rnti);
    EXPECT_EQ(uci.prbSize, pucch_test_defaults::k_prb_size);
    EXPECT_EQ(uci.startPrb, pucch_test_defaults::k_prb_start);
    EXPECT_EQ(uci.startSym, pucch_test_defaults::k_start_sym);
    EXPECT_EQ(uci.nSym, pucch_test_defaults::k_num_sym);
    EXPECT_EQ(uci.cellPrmStatIdx, pucch_test_defaults::k_cell_stat_1);
    EXPECT_EQ(uci.cellPrmDynIdx, 0u);
    // srFlag is suppressed for F2/3/4 (sr bits ride in the UCI payload instead).
    EXPECT_EQ(uci.srFlag, format_type > UL_TTI_PUCCH_FORMAT_1 ? 0u : 1u);

    // The Order-metadata seam ran once with the parser's pass-through arguments.
    EXPECT_EQ(pucch_order_append_count, 1);
    EXPECT_EQ(pucch_last_cell_index, pucch_test_defaults::k_carrier_1);
    EXPECT_EQ(pucch_last_cell_stat_prm_idx, pucch_test_defaults::k_cell_stat_1);
    EXPECT_EQ(pucch_last_ul_bandwidth, pucch_test_defaults::k_ul_bandwidth);
    EXPECT_EQ(pucch_last_prb_size, pucch_test_defaults::k_prb_size);
    EXPECT_EQ(pucch_last_format, format_type);
    EXPECT_EQ(pucch_last_rnti, pucch_test_defaults::k_rnti);
    EXPECT_TRUE(pucch_last_mmimo_enabled);
    // MockCellView::slot_detail() returns nullptr, so the parser must forward that.
    EXPECT_FALSE(pucch_last_slot_detail_present);
}

// Builds a UL_TTI.request carrying PUCCH PDUs. There is no per-entry wire type to
// choose: every PUCCH format is carried as UL_TTI_PDU_TYPE_PUCCH, and the format
// comes from the PDU's own format_type field.
struct FapiUlMsg final
{
    std::vector<uint8_t> buf;
    nv::phy_mac_msg_desc desc{};

    FapiUlMsg(const uint16_t sfn,
              const uint16_t slot,
              const uint32_t cell_id,
              std::initializer_list<scf_fapi_pucch_pdu_t> pdus)
    {
        const std::size_t hdr_sz      = sizeof(scf_fapi_header_t);
        const std::size_t req_sz      = sizeof(scf_fapi_ul_tti_req_t);
        const std::size_t gen_hdr_sz  = sizeof(scf_fapi_generic_pdu_info_t);
        // scf_fapi_pucch_pdu_t ends in a flexible payload[0] that the parser always
        // reads as scf_fapi_rx_beamforming_t (see the append_pucch_order_prbs call in
        // parse_pucch_pdu_common), so every entry on the wire carries one beamforming
        // struct after the fixed PDU body. Counting it in entry_sz keeps pdu_size,
        // msg_len and the dispatch cursor consistent: without it each PDU but the last
        // would present the following generic header as its beamforming data.
        const std::size_t bf_sz       = sizeof(scf_fapi_rx_beamforming_t);
        const std::size_t entry_sz    = gen_hdr_sz + sizeof(scf_fapi_pucch_pdu_t) + bf_sz;
        const std::size_t payload_len = pdus.size() * entry_sz;
        buf.resize(hdr_sz + req_sz + payload_len, 0u);

        auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(buf.data() + hdr_sz);
        req->sfn = sfn;
        req->slot = slot;
        req->num_pdus = static_cast<decltype(req->num_pdus)>(pdus.size());
#ifdef SCF_FAPI_10_04
        for (const auto& pdu : pdus)
        {
            // nPDUsOfEachType is keyed by PUCCH *format*, not by the wire pdu_type:
            // F0/F1 and F2/3/4 have separate counters even though both arrive as
            // UL_TTI_PDU_TYPE_PUCCH on the wire.
            const auto idx = (pdu.format_type > UL_TTI_PUCCH_FORMAT_1)
                ? UL_TTI_NPDUS_IDX_PUCCH_F234
                : UL_TTI_NPDUS_IDX_PUCCH_F01;
            ++req->nPDUsOfEachType[idx];
        }
#else
        req->num_ulcch = static_cast<uint8_t>(pdus.size());
#endif

        auto* next = req->payload;
        for (const auto& pdu : pdus)
        {
            auto* hdr = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(next);
            // All PUCCH formats share one wire pdu_type. UL_TTI_PDU_TYPE_PUCCH_2_3_4
            // is an internal dispatch tag, not a value that appears in the FAPI
            // stream, so emitting it here would not match what the L2 sends.
            hdr->pdu_type = static_cast<uint16_t>(UL_TTI_PDU_TYPE_PUCCH);
            hdr->pdu_size = static_cast<uint16_t>(entry_sz);
            std::memcpy(hdr->pdu_config, &pdu, sizeof(pdu));
            next += entry_sz;
        }

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(hdr_sz + req_sz + payload_len);
        desc.cell_id = cell_id;
    }
};

} // namespace

TEST(PucchPduParser, Format01PopulatesSlotCommand)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::PucchPduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_TRUE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_1);
#ifdef ENABLE_L2_SLT_RSP
    // pf*_parsed counters are only bumped inside the ENABLE_L2_SLT_RSP block of
    // parse_pucch_pdu_common (validate_pucch_pdu_l1_limits is gated). Wrap the
    // counter assertion so the test still passes when the gate is off.
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf1_parsed, 1u);
#endif
}

TEST(PucchPduParser, Format0PopulatesSlotCommand)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::PucchPduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_0);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_TRUE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_0);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf0_parsed, 1u);
#endif
}

TEST(Pucch234PduParser, Format234PopulatesSlotCommand)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::Pucch234PduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_3);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_TRUE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_3);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf3_parsed, 1u);
#endif
}

TEST(Pucch234PduParser, Format2PopulatesSlotCommand)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::Pucch234PduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_2);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_TRUE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_2);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf2_parsed, 1u);
#endif
}

TEST(Pucch234PduParser, Format4PopulatesSlotCommand)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::Pucch234PduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_4);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_TRUE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_4);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf4_parsed, 1u);
#endif
}

#ifdef ENABLE_L2_SLT_RSP
// The L1-limit pre-check that drops the PDU before it reaches the slot command only
// exists when ENABLE_L2_SLT_RSP is defined (see parse_pucch_pdu_common()).
// Build the test conditionally so the count==0 / pf0_errors==1 expectations match
// the compiled path; mirrors the equivalent guard in test_srs_pdu_parser.cpp.
TEST(PucchPduParser, L1LimitDropSkipsSlotCommandPopulation)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    view.group_limit_errors_.pucch_errors.pf0_parsed = 0xFFu;
    scf_5g_fapi::PucchPduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_0);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_TRUE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    EXPECT_EQ(pucch_order_append_count, 0);
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf0_errors, 1u);
}
#endif // ENABLE_L2_SLT_RSP

TEST(PucchPduParser, ParseWithoutSetupReturnsFalse)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::PucchPduParser parser{view};

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_FALSE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    EXPECT_EQ(pucch_order_append_count, 0);
}

TEST(PucchPduParser, NegativeCarrierIdDropsPdu)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    view.carrier_id_base_ = -2;
    scf_5g_fapi::PucchPduParser parser{view};
    parser.setup_cell(pucch_test_defaults::k_local_cell_1);

    const auto pdu = make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1);
    ASSERT_NO_FATAL_FAILURE(
        EXPECT_FALSE(parser.parse(pucch_test_defaults::k_sfn, pucch_test_defaults::k_slot, pdu)));

    EXPECT_EQ(pucch_order_append_count, 0);
}

TEST(ULSlotProcessor, Process_PucchF01_HappyPath_PopulatesGroupParams)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_1,
                  {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1)}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_1);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf1_parsed, 1u);
#endif
}

// F2/3/4 PDUs are dispatched through UL_TTI_PDU_TYPE_PUCCH, same as F0/1: the
// processor selects the Pucch234PduParser from the PDU's format_type, not from a
// distinct wire type. Dispatching UL_TTI_PDU_TYPE_PUCCH_2_3_4 here would look
// plausible but exercises a path the L2 never produces.
TEST(ULSlotProcessor, Process_PucchF234_HappyPath_PopulatesGroupParams)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_1,
                  {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_3)}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_3);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf3_parsed, 1u);
#endif
}

TEST(ULSlotProcessor, Process_PucchF0_HappyPath_PopulatesGroupParams)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_1,
                  {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_0)}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_0);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf0_parsed, 1u);
#endif
}

TEST(ULSlotProcessor, Process_PucchF2_HappyPath_PopulatesGroupParams)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_1,
                  {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_2)}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_2);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf2_parsed, 1u);
#endif
}

TEST(ULSlotProcessor, Process_PucchF4_HappyPath_PopulatesGroupParams)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_1,
                  {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_4)}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    expect_single_pdu_accepted(view, UL_TTI_PUCCH_FORMAT_4);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf4_parsed, 1u);
#endif
}

TEST(ULSlotProcessor, Process_NoPucch_IsNoOp)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_0,
                  {}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(pucch_order_append_count, 0);
}

// Two cells, each carrying one F1 and one F3 PDU. A single UL_TTI_PDU_TYPE_PUCCH
// pass consumes all four: the per-type expected count for PUCCH covers the F01 and
// F234 counters together, so splitting this into two passes would trip the
// per-type mismatch check (expected=2 got=1) on each pass.
TEST(ULSlotProcessor, Process_TwoCellAccumulation)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg cell0{pucch_test_defaults::k_sfn,
                    pucch_test_defaults::k_slot,
                    pucch_test_defaults::k_local_cell_0,
                    {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1),
                     make_pucch_pdu(UL_TTI_PUCCH_FORMAT_3)}};
    FapiUlMsg cell1{pucch_test_defaults::k_sfn,
                    pucch_test_defaults::k_slot,
                    pucch_test_defaults::k_local_cell_1,
                    {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1),
                     make_pucch_pdu(UL_TTI_PUCCH_FORMAT_3)}};
    const std::array<nv::phy_mac_msg_desc, 2> msgs = {cell0.desc, cell1.desc};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{msgs});
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(pucch_order_append_count, 4);
#ifdef ENABLE_L2_SLT_RSP
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf1_parsed, 2u);
    EXPECT_EQ(view.group_limit_errors_.pucch_errors.pf3_parsed, 2u);
#endif

    // Both cells accumulated into one group: 2 cells, 2 F1 UCIs, 2 F3 UCIs.
    auto* const params = view.slot_cmd_.cell_groups.get_pucch_params();
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->grp_dyn_pars.nCells, 2u);
    EXPECT_EQ(params->grp_dyn_pars.nF1Ucis, 2u);
    EXPECT_EQ(params->grp_dyn_pars.nF3Ucis, 2u);
    ASSERT_EQ(params->cell_index_list.size(), 2u);
    EXPECT_EQ(params->cell_index_list[0], pucch_test_defaults::k_carrier_id_base);
    EXPECT_EQ(params->cell_index_list[1], pucch_test_defaults::k_carrier_1);

    // Cell 1 was processed last.
    EXPECT_EQ(pucch_last_cell_index, pucch_test_defaults::k_carrier_1);
}

TEST(ULSlotProcessor, Process_NullGroupCommand_PropagatesParserFailure)
{
    using namespace scf_5g_fapi::test_support;
    reset_pucch_stub_state();
    MockUlModuleView view;
    view.group_command_null_ = true;
    scf_5g_fapi::ULSlotProcessor processor{view};
    FapiUlMsg msg{pucch_test_defaults::k_sfn,
                  pucch_test_defaults::k_slot,
                  pucch_test_defaults::k_local_cell_1,
                  {make_pucch_pdu(UL_TTI_PUCCH_FORMAT_1)}};

    const auto result = processor.process<UL_TTI_PDU_TYPE_PUCCH>(std::span{&msg.desc, 1u});

#ifdef SCF_FAPI_10_04
    EXPECT_FALSE(result.has_value());
    if (!result.has_value())
    {
        EXPECT_EQ(result.error().code, scf_5g_fapi::SlotParseError::Code::PerTypeMismatch);
    }
#else
    EXPECT_TRUE(result.has_value());
#endif
    EXPECT_EQ(pucch_order_append_count, 0);
}
