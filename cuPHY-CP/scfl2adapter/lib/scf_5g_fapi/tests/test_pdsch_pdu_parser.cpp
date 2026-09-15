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
 * @file test_pdsch_pdu_parser.cpp
 * @brief Unit tests for PdschPduParser and DLSlotProcessor (PDSCH path).
 *
 * Covers:
 *   Group A — setup_cell: slot selection (req vs static override); CSI-RS count
 *             is intentionally NOT forwarded (owned by the CSI-RS channel processor)
 *   Group B — parse(): UE fields, codeword fields, and UE-group fields
 *   Group C — LBRM lookup: exhaustive bucket table (BWP 25–300)
 *   Group D — power-control beta LUT: offset clamping and exponent formula
 *   Group E — UE-group reuse: same vs different resource allocation drives nUeGrps
 *   Group F — DLSlotProcessor PDSCH path: single message and batch dispatch
 *   Group G — apply_pm_weights: PM weight insertion and guard paths
 *             (pm_enabled=true / mmimo=false required to invoke the function)
 *
 * All groups are table-driven: static case arrays + SCOPED_TRACE.
 *
 * Design notes:
 *   - MockDlModuleView satisfies the DlModuleView concept without linking PHY or CUDA kernels.
 *   - check_bf_pc_params() is provided by test_stubs.cpp (always returns true).
 *   - All test PDUs have mmimo_enabled=true and pm_enabled=false so apply_pm_weights()
 *     is not invoked, keeping the mock free of pm_weight_map setup.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "scf_5g_fapi_dl_stats.hpp"
#include "scf_5g_fapi_pdsch_pdu_parser.hpp"
#include "scf_5g_fapi_dl_slot_processor.hpp"
#include "scf_5g_fapi_lbrm.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "scf_5g_fapi.h"

// ---------------------------------------------------------------------------
// Mock CellView (satisfies scf_5g_fapi::CellView)
// ---------------------------------------------------------------------------

namespace {

struct MockCellView
{
    uint16_t           bwp_size_{59};
    cuphyCellStatPrm_t stat_prm_{};

    MockCellView() noexcept
    {
        // Default matches prior hardcoded LBRM maxLayers (=4).
        stat_prm_.nRxAnt = scf_5g_fapi::lbrm::k_max_layers;
        stat_prm_.nTxAnt = scf_5g_fapi::lbrm::k_max_layers;
    }

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return bwp_size_; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_prm_; }
};

static_assert(scf_5g_fapi::CellView<MockCellView>,
              "MockCellView must satisfy CellView");

// ---------------------------------------------------------------------------
// Mock DlModuleView (satisfies scf_5g_fapi::DlModuleView)
// ---------------------------------------------------------------------------

struct MockDlModuleView
{
    // Configurable flags — default: mmimo only (no PM weights, no BF)
    bool pm_enabled_    {false};
    bool bf_enabled_    {false};
    bool mmimo_enabled_ {true};
    int  static_pdsch_slot_{-1};

    // FH gate (P5): when true, parser populates fh_params; when false, NullFhCallable path.
    bool fh_populate_enabled_{true};
    // Per-cell DL PRB count surfaced by num_dl_prb() — defaults to bwp_size of MockCellView.
    uint16_t num_dl_prb_{59};
    // Optional BFW pointer returned by bfw_coeff_mem_info(). Owned by test fixture.
    slot_command_api::bfw_coeff_mem_info_t* bfw_{nullptr};
    // Per-cell carrier_id. carrier_id_lookup_[cell_idx] = carrier_id used by FH params.
    // Default: identity (carrier_id == cell_idx) so multi-cell tests use distinct cell_index.
    std::array<int32_t, slot_command_api::MAX_CELLS_PER_CELL_GROUP> carrier_id_lookup_{};
    std::array<uint16_t, slot_command_api::MAX_CELLS_PER_CELL_GROUP> stat_prm_idx_lookup_{};
    std::array<uint16_t, slot_command_api::MAX_CELLS_PER_CELL_GROUP> phy_cell_id_lookup_{};

    mutable slot_command_api::slot_command     slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable scf_5g_fapi::pm_weight_map_t       pm_map_{};
    mutable nv::phy_config_option              config_opt_{};
    mutable nv::phy_config                     phy_config_{};
    mutable nv::slot_limit_cell_error_t        limit_errors_{};
    mutable std::array<scf_5g_fapi::DlPdschCellDelta, MAX_CELLS_PER_SLOT> published_stats_{};
    mutable uint32_t                            publish_nonzero_cells_{};
    MockCellView                               cell_view_{};

    MockDlModuleView() noexcept {
        for (int32_t i = 0; i < static_cast<int32_t>(carrier_id_lookup_.size()); ++i) {
            carrier_id_lookup_[static_cast<std::size_t>(i)] = i;
        }
    }

    // ---- DlModuleView interface ------------------------------------------

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }

    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t) const noexcept
    {
        return cell_sub_cmd_;
    }

    [[nodiscard]] scf_5g_fapi::pm_weight_map_t& pm_map() const noexcept { return pm_map_; }

    [[nodiscard]] bool pm_enabled()    const noexcept { return pm_enabled_; }
    [[nodiscard]] bool bf_enabled()    const noexcept { return bf_enabled_; }

    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t*
    bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept { return bfw_; }

    [[nodiscard]] bool mmimo_enabled() const noexcept { return mmimo_enabled_; }

    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }

    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPdcchSlotNum() const noexcept { return -1; }
    [[nodiscard]] int staticPdschSlotNum() const noexcept { return static_pdsch_slot_; }

    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t cell_idx) const noexcept
    {
        return (cell_idx < stat_prm_idx_lookup_.size()) ? stat_prm_idx_lookup_[cell_idx] : 0u;
    }
    [[nodiscard]] int32_t  carrier_id(uint32_t cell_idx) const noexcept {
        return (cell_idx < carrier_id_lookup_.size())
            ? carrier_id_lookup_[cell_idx] : 0;
    }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t cell_idx) const noexcept
    {
        return (cell_idx < phy_cell_id_lookup_.size()) ? phy_cell_id_lookup_[cell_idx] : 0u;
    }
    [[nodiscard]] const nv::phy_config& phy_config([[maybe_unused]] uint32_t cell_idx) const noexcept
    {
        return phy_config_;
    }

    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }

    [[nodiscard]] MockCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_;
    }

    // P3: per-cell DL BWP PRB count (FH params consumer).
    [[nodiscard]] uint16_t num_dl_prb(uint32_t) const noexcept { return num_dl_prb_; }

    // P5: FH runtime gate. False (default) means fh_params should be populated.
    [[nodiscard]] bool is_fapi_to_cplane_direct_enabled() const noexcept
    {
        return !fh_populate_enabled_;
    }

    void publish_dl_pdsch_stats(const scf_5g_fapi::DlPdschStatsBatch& batch) const noexcept
    {
        for (std::uint32_t cell_id = 0U; cell_id < MAX_CELLS_PER_SLOT; ++cell_id) {
            const auto delta = batch.cell_delta(cell_id);
            if (delta.bytes != 0U || delta.slots != 0U) {
                auto& published = published_stats_[cell_id];
                published.bytes += delta.bytes;
                published.slots += delta.slots;
                ++publish_nonzero_cells_;
            }
        }
    }
};

static_assert(scf_5g_fapi::DlModuleView<MockDlModuleView>,
              "MockDlModuleView must satisfy DlModuleView");

// ---------------------------------------------------------------------------
// PDSCH PDU binary buffer builder
// ---------------------------------------------------------------------------

/**
 * Test parameter container for one serialised PDSCH PDU.
 *
 * Defaults produce a single-codeword, RA-Type 1, no-PTRS PDU suitable for
 * most field-population tests.  Pass an instance to @c build_pdsch_pdu_body()
 * to obtain the corresponding binary buffer.
 */
struct PdschPduParams
{
    uint16_t rnti              {0x1234};  //!< UE RNTI.
    uint16_t bwp_size          {59};      //!< Bandwidth part size in PRBs.
    uint16_t bwp_start         {0};       //!< Bandwidth part start PRB.
    uint8_t  num_codewords     {1};       //!< Number of codewords (1 or 2).
    uint16_t pdu_bitmap        {0};       //!< PDU options bitmap (bit 0 = has_ptrs).

    // Codeword (first/only)
    uint8_t  mcs_table         {0};       //!< MCS table index.
    uint8_t  mcs_index         {5};       //!< MCS index within the table.
    uint16_t target_code_rate  {193};     //!< Target code rate × 10.
    uint8_t  qam_mod_order     {2};       //!< Modulation order (2/4/6/8).
    uint8_t  rv_index          {0};       //!< Redundancy version (0–3).
    uint32_t tb_size           {1000};    //!< Transport block size in bytes.

    // PDU end (scf_fapi_pdsch_pdu_end_t fields)
    uint16_t data_scrambling_id    {42};      //!< Data scrambling ID.
    uint8_t  num_of_layers         {1};       //!< Number of MIMO layers.
    uint8_t  ref_point             {0};       //!< Reference point (0 = CRB 0).
    uint16_t dl_dmrs_sym_pos       {0x0080};  //!< DMRS symbol position bitmap.
    uint16_t dl_dmrs_scrambling_id {100};     //!< DMRS scrambling ID.
    uint8_t  sc_id                 {0};       //!< DMRS sequence ID.
    uint8_t  num_dmrs_cdm_grps     {1};       //!< Number of DMRS CDM groups.
    uint16_t dmrs_ports            {1};       //!< DMRS antenna port bitmap.
    uint8_t  resource_alloc        {1};       //!< Resource allocation type (1 = RA-Type 1).
    uint16_t rb_start              {0};       //!< Starting RB index (RA-Type 1).
    uint16_t rb_size               {50};      //!< Number of allocated RBs.
    uint8_t  start_sym_index       {2};       //!< Starting OFDM symbol index.
    uint8_t  num_symbols           {12};      //!< Number of allocated OFDM symbols.

    // PM/BF fixed header (scf_fapi_pdsch_pdu_pm_bf_t)
    uint16_t num_prgs          {1};   //!< Number of PRGs.
    uint16_t prg_size          {52};  //!< PRG size in PRBs.
    uint8_t  dig_bf_interfaces {0};   //!< Digital BF interfaces per PRG (0 = mMIMO path).

    /** PM index and beam index pairs: @c num_prgs × (@c dig_bf_interfaces + 1) entries.
     *  Leave empty for the mMIMO path (@c dig_bf_interfaces == 0 skips weight lookup). */
    std::vector<uint16_t> pm_idx_and_beam_idx{};

    // TX power (scf_fapi_tx_power_info_t)
    uint8_t  power_ctrl_offset    {8};  //!< Power control offset (maps to beta LUT).
    uint8_t  power_ctrl_offset_ss {1};  //!< SS power control offset.
};

/**
 * Build the binary PDSCH PDU body (starting at the fixed pdu_t header).
 *
 * Layout (all packed, no padding):
 *   [scf_fapi_pdsch_pdu_t fixed fields]
 *   [num_codewords × scf_fapi_pdsch_codeword_t]
 *   [scf_fapi_pdsch_pdu_end_t]
 *   [5 bytes: pm_bf fixed header (num_prgs:2, prg_size:2, dig_bf_interfaces:1)]
 *   [scf_fapi_tx_power_info_t]
 *
 * @param[in]  p  PDSCH PDU parameters to serialise.
 *
 * @return  Byte buffer containing the serialised PDU body, sized exactly to
 *          hold all fields derived from @p p.  Return value must be checked.
 */
[[nodiscard]] std::vector<uint8_t> build_pdsch_pdu_body(const PdschPduParams& p)
{
    // Compute total size using sizeof() on packed structs so we don't need
    // to hard-code byte offsets for each field.
    static constexpr std::size_t k_pm_bf_fixed = 5u; // 2+2+1 bytes
    const std::size_t k_pm_idx_bytes = p.pm_idx_and_beam_idx.size() * sizeof(uint16_t);
    const std::size_t total =
          sizeof(scf_fapi_pdsch_pdu_t)
        + p.num_codewords * sizeof(scf_fapi_pdsch_codeword_t)
        + sizeof(scf_fapi_pdsch_pdu_end_t)
        + k_pm_bf_fixed
        + k_pm_idx_bytes
        + sizeof(scf_fapi_tx_power_info_t);

    std::vector<uint8_t> buf(total, 0u);
    uint8_t* ptr = buf.data();

    // ---- PDU header -------------------------------------------------------
    auto* pdu = reinterpret_cast<scf_fapi_pdsch_pdu_t*>(ptr);
    pdu->pdu_bitmap    = p.pdu_bitmap;
    pdu->rnti          = p.rnti;
    pdu->bwp.bwp_start = p.bwp_start;
    pdu->bwp.bwp_size  = p.bwp_size;
    pdu->num_codewords = p.num_codewords;
    ptr += sizeof(scf_fapi_pdsch_pdu_t);

    // ---- Codewords --------------------------------------------------------
    for (uint8_t cw = 0; cw < p.num_codewords; ++cw) {
        auto* cw_entry = reinterpret_cast<scf_fapi_pdsch_codeword_t*>(ptr);
        cw_entry->mcs_table        = p.mcs_table;
        cw_entry->mcs_index        = p.mcs_index;
        cw_entry->target_code_rate = p.target_code_rate;
        cw_entry->qam_mod_order    = p.qam_mod_order;
        cw_entry->rv_index         = p.rv_index;
        cw_entry->tb_size          = p.tb_size;
        ptr += sizeof(scf_fapi_pdsch_codeword_t);
    }

    // ---- PDU end ----------------------------------------------------------
    auto* end = reinterpret_cast<scf_fapi_pdsch_pdu_end_t*>(ptr);
    end->data_scrambling_id      = p.data_scrambling_id;
    end->num_of_layers           = p.num_of_layers;
    end->ref_point               = p.ref_point;
    end->dl_dmrs_sym_pos         = p.dl_dmrs_sym_pos;
    end->dl_dmrs_scrambling_id   = p.dl_dmrs_scrambling_id;
    end->sc_id                   = p.sc_id;
    end->num_dmrs_cdm_grps_no_data = p.num_dmrs_cdm_grps;
    end->dmrs_ports              = p.dmrs_ports;
    end->resource_alloc          = p.resource_alloc;
    end->rb_start                = p.rb_start;
    end->rb_size                 = p.rb_size;
    end->start_sym_index         = p.start_sym_index;
    end->num_symbols             = p.num_symbols;
    ptr += sizeof(scf_fapi_pdsch_pdu_end_t);

    // ---- pm_bf fixed header (5 bytes) ------------------------------------
    // num_prgs : uint16_t, prg_size : uint16_t, dig_bf_interfaces : uint8_t
    *reinterpret_cast<uint16_t*>(ptr)     = p.num_prgs;
    *reinterpret_cast<uint16_t*>(ptr + 2) = p.prg_size;
    *(ptr + 4)                            = p.dig_bf_interfaces;
    ptr += k_pm_bf_fixed;

    // ---- pm_bf variable section (pm_idx_and_beam_idx entries) ------------
    if (!p.pm_idx_and_beam_idx.empty()) {
        std::memcpy(ptr, p.pm_idx_and_beam_idx.data(), k_pm_idx_bytes);
        ptr += k_pm_idx_bytes;
    }

    // ---- tx_power ---------------------------------------------------------
    auto* tx = reinterpret_cast<scf_fapi_tx_power_info_t*>(ptr);
    tx->power_control_offset    = p.power_ctrl_offset;
    tx->power_control_offset_ss = p.power_ctrl_offset_ss;

    return buf;
}

/**
 * @brief DL TTI request buffer: FAPI header + DL_TTI.request + one PDSCH PDU.
 *
 * Constructs a complete FAPI message byte buffer (scf_fapi_header_t +
 * scf_fapi_dl_tti_req_t + scf_fapi_generic_pdu_info_t + PDSCH body) and
 * populates a @c phy_mac_msg_desc that points into it, ready to pass to
 * @c DLSlotProcessor::process().
 *
 * @param[in]  sfn          System frame number written into the DL_TTI request.
 * @param[in]  slot         Slot number written into the DL_TTI request.
 * @param[in]  pdsch_body   Pre-serialised PDSCH PDU body produced by
 *                          @c build_pdsch_pdu_body().
 * @param[in]  csirs_count  With @c SCF_FAPI_10_04: written to
 *                          @c nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS]
 *                          (default 0). Ignored when @c SCF_FAPI_10_04 is off.
 *
 * @note  The constructor sets @c desc.msg_buf to @c buf.data(),
 *        @c desc.msg_len to the total buffer size, and @c desc.cell_id to 0.
 *        @c buf owns the memory; @c desc is valid only while @c buf is alive.
 *
 * @see build_pdsch_pdu_body
 */
struct FapiDlMsg
{
    std::vector<uint8_t>  buf;
    nv::phy_mac_msg_desc  desc{};

    FapiDlMsg(uint16_t sfn, uint16_t slot,
              const std::vector<uint8_t>& pdsch_body,
              uint16_t csirs_count = 0u)
    {
        const std::size_t hdr_sz     = sizeof(scf_fapi_header_t);
        const std::size_t req_sz     = sizeof(scf_fapi_dl_tti_req_t);
        const std::size_t gen_hdr_sz = sizeof(scf_fapi_generic_pdu_info_t);
        const std::size_t total      = hdr_sz + req_sz + gen_hdr_sz + pdsch_body.size();

        buf.resize(total, 0u);

        // DL TTI request header
        auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(buf.data() + hdr_sz);
        req->sfn      = sfn;
        req->slot     = slot;
        req->num_pdus = 1u;
#ifdef SCF_FAPI_10_04
        req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH]  = 1u;
        req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS] = csirs_count;
#endif

        // Generic PDU header (wraps PDSCH body)
        auto* gen = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(
            buf.data() + hdr_sz + req_sz);
        gen->pdu_type = static_cast<uint16_t>(DL_TTI_PDU_TYPE_PDSCH);
        gen->pdu_size = static_cast<uint16_t>(gen_hdr_sz + pdsch_body.size());

        // PDSCH body (starts at gen->pdu_config, offset 4 from gen header start)
        std::memcpy(buf.data() + hdr_sz + req_sz + gen_hdr_sz,
                    pdsch_body.data(), pdsch_body.size());

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(buf.size());
        desc.cell_id = 0u;
    }
};

struct FapiDlNoPdschMsg
{
    std::vector<uint8_t>  buf;
    nv::phy_mac_msg_desc  desc{};

    FapiDlNoPdschMsg(uint16_t sfn, uint16_t slot, uint32_t cell_id = 0u)
    {
        const std::size_t hdr_sz = sizeof(scf_fapi_header_t);
        const std::size_t req_sz = sizeof(scf_fapi_dl_tti_req_t);
        buf.resize(hdr_sz + req_sz, 0u);

        auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(buf.data() + hdr_sz);
        req->sfn      = sfn;
        req->slot     = slot;
        req->num_pdus = 0u;

        desc.msg_buf = buf.data();
        desc.msg_len = static_cast<uint32_t>(buf.size());
        desc.cell_id = cell_id;
    }
};

[[nodiscard]] scf_fapi_generic_pdu_info_t* generic_pdu_header(FapiDlMsg& msg) noexcept
{
    return reinterpret_cast<scf_fapi_generic_pdu_info_t*>(
        msg.buf.data() + sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_tti_req_t));
}

/**
 * Return the pdsch_params pointer from the slot_command stored in @p view.
 *
 * @param[in,out]  view  Mock DL module view whose @c slot_cmd_.cell_groups
 *                       holds the pdsch_params populated by the parser under test.
 *
 * @return  Pointer to @c slot_command_api::pdsch_params obtained via
 *          @c view.slot_cmd_.cell_groups.get_pdsch_params().
 *          Return value must be checked — may be @c nullptr if no PDSCH
 *          params have been set up.
 */
[[nodiscard]] slot_command_api::pdsch_params* get_params(MockDlModuleView& view)
{
    return view.slot_cmd_.cell_groups.get_pdsch_params();
}

// ===========================================================================
// Group A — setup_cell (table-driven)
//
// Verifies slot selection (req slot vs static override). CSI-RS counts are
// deliberately NOT forwarded by setup_cell — that field family is owned by
// process_aggr_csirs_channel — so nCsiRsPrms must stay 0 for every case even
// when nPDUsOfEachType[CSI_RS] is non-zero.
//
// Invariants for every case:
//   nCells == 1, cellPrmDynIdx == 0, csiRsPrmsOffset == 0 (first call),
//   pdschStartSym/nPdschSym/dmrsSymLocBmsk == 0, cell_ue_group_idx_start == 0
//
// Per-case variables: req_slot, static_pdsch_slot, csirs_count → expected_slot
// ===========================================================================

TEST(PdschPduParser, SetupCell_TableDriven)
{
    struct Case {
        std::string_view description;
        uint16_t         req_slot;
        int              static_pdsch_slot;  ///< -1 = no override.
        uint16_t         csirs_count;        ///< Injected into req.nPDUsOfEachType[CSI_RS]; must NOT be forwarded (nCsiRsPrms stays 0).
        uint16_t         expected_slot;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "req slot used when no static override; csirs input NOT forwarded",
            3u, -1, 2u, 3u,
        },
        {
            "static slot override wins over req slot",
            3u,  7, 0u, 7u,
        },
        {
            "boundary: slot=0, no override, no CSI-RS",
            0u, -1, 0u, 0u,
        },
        {
            "static override + non-zero CSI-RS input (still not forwarded)",
            5u,  9, 4u, 9u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        view.static_pdsch_slot_ = tc.static_pdsch_slot;
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        req.slot = tc.req_slot;
        // Feed the CSI-RS PDU count into the request so the assertion below
        // proves setup_cell() ignores it (the count is owned by the CSI-RS
        // channel processor). nPDUsOfEachType is a FAPI-10.04-only field; this
        // suite only builds under SCF_FAPI_10_04, so the guard is always taken.
#ifdef SCF_FAPI_10_04
        req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS] = tc.csirs_count;
#endif

        // Act
        parser.setup_cell(req, 0u);

        // Assert
        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);

        // Structural invariants — cell-group level
        EXPECT_EQ(params->cell_grp_info.nCells,             1u);
        EXPECT_EQ(params->cell_grp_info.nUes,               0u);  // no UEs parsed yet
        EXPECT_EQ(params->cell_grp_info.nUeGrps,            0u);
        EXPECT_EQ(params->cell_grp_info.nCws,               0u);
        EXPECT_EQ(params->cell_ue_group_idx_start,          0u);

        // Structural invariants — cell_dyn_info
        EXPECT_EQ(params->cell_dyn_info[0].cellPrmDynIdx,   0u);
        EXPECT_EQ(params->cell_dyn_info[0].csiRsPrmsOffset, 0u);
        EXPECT_EQ(params->cell_dyn_info[0].pdschStartSym,   0u);
        EXPECT_EQ(params->cell_dyn_info[0].nPdschSym,       0u);
        EXPECT_EQ(params->cell_dyn_info[0].dmrsSymLocBmsk,  0u);

        // Per-case assertions
        EXPECT_EQ(params->cell_dyn_info[0].slotNum, tc.expected_slot);
        // setup_cell() no longer forwards the CSI-RS count: those fields are
        // owned exclusively by process_aggr_csirs_channel (see one-writer-per-
        // field-family note on slot_command_api::pdsch_params). The PDSCH parser
        // must leave nCsiRsPrms untouched regardless of nPDUsOfEachType[CSI_RS].
        EXPECT_EQ(params->cell_dyn_info[0].nCsiRsPrms, 0u);
        EXPECT_EQ(params->cell_grp_info.nCsiRsPrms,    0u);
    }
}

// Regression: destination ordinal (nCells) can differ from msg_cell_id when
// only a non-zero cell carries PDSCH. Source lookups must use msg_cell_id.
TEST(PdschPduParser, SetupCell_UsesMessageCellForSourceLookups)
{
    MockDlModuleView view{};
    view.carrier_id_lookup_[1] = 7;
    view.stat_prm_idx_lookup_[1] = 11;
    view.phy_cell_id_lookup_[1] = 51;

    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};
    scf_fapi_dl_tti_req_t req{};
    req.slot = 7;
    parser.setup_cell(req, 1u);

    auto* params = get_params(view);
    ASSERT_NE(params, nullptr);
    ASSERT_EQ(params->cell_grp_info.nCells, 1u);
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmDynIdx, 0u);
    EXPECT_EQ(params->cell_dyn_info[0].cellPrmStatIdx, 11u);
    EXPECT_EQ(params->cell_index_list[0], 7);
    EXPECT_EQ(params->phy_cell_index_list[0], 51);
}

TEST(PdschPduParser, SetupCell_SkipsOutOfRangeMessageCell)
{
    MockDlModuleView view{};
    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};
    scf_fapi_dl_tti_req_t req{};

    parser.setup_cell(req, static_cast<uint32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP));

    auto* params = get_params(view);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_grp_info.nCells, 0u);
    EXPECT_TRUE(params->cell_index_list.empty());
    EXPECT_TRUE(params->phy_cell_index_list.empty());

    // Failed setup must hard-stop parse() so prior-cell state cannot be reused.
    // Use a structurally valid PDU so failure cannot be attributed to malformed
    // input if the setup gate regresses.
    PdschPduParams p{};
    const auto body = build_pdsch_pdu_body(p);
    const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body.data());
    EXPECT_FALSE(parser.parse(/*sfn=*/0u, /*slot=*/0u, pdu));
}

// ===========================================================================
// Group B — parse() field population (table-driven, three sub-groups)
//
// B1: UE-level fields  (rnti, BWPStart, nCw, nUeLayers, scrambling IDs, ports)
// B2: Codeword fields  (MCS table/index, code rate, QAM order, RV, TB size,
//                       maxQm, n_PRB_LBRM, tbStartOffset, maxLayers)
// B3: UE-group fields  (DMRS bitmap, start symbol, num symbols, nUes, pointers)
//
// All three sub-groups use bwp_size=59 → n_PRB_LBRM=66 unless noted.
// LBRM boundary coverage is in Group C; beta LUT coverage is in Group D.
// ===========================================================================

TEST(PdschPduParser, Parse_UeFields_TableDriven)
{
    struct Case {
        std::string_view description;
        uint16_t rnti;
        uint16_t bwp_start;
        uint8_t  num_of_layers;
        uint16_t data_scrambling_id;
        uint16_t dmrs_ports;
        uint8_t  sc_id;
        uint16_t dl_dmrs_scrambling_id;
        uint8_t  ref_point;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "multi-layer UE, non-zero BWP start, two DMRS ports",
            0xABCDu, 5u, 2u, 99u, 0x0003u, 1u, 77u, 0u,
        },
        {
            "single-layer UE, zero BWP start, single DMRS port",
            0x0001u, 0u, 1u,  0u, 0x0001u, 0u,  0u, 0u,
        },
        {
            "four-layer UE, large BWP start, four DMRS ports, refPoint=1",
            0xFFFFu, 100u, 4u, 255u, 0x000Fu, 1u, 200u, 1u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        parser.setup_cell(req, 0u);

        PdschPduParams p{};
        p.rnti                  = tc.rnti;
        p.bwp_start             = tc.bwp_start;
        p.num_of_layers         = tc.num_of_layers;
        p.data_scrambling_id    = tc.data_scrambling_id;
        p.dmrs_ports            = tc.dmrs_ports;
        p.sc_id                 = tc.sc_id;
        p.dl_dmrs_scrambling_id = tc.dl_dmrs_scrambling_id;
        p.ref_point             = tc.ref_point;

        // Act
        const auto body = build_pdsch_pdu_body(p);
        const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu));

        // Assert
        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);
        ASSERT_EQ(params->cell_grp_info.nUes, 1u);

        const auto& ue = params->ue_info[0];
        EXPECT_EQ(ue.rnti,         tc.rnti);
        EXPECT_EQ(ue.BWPStart,     tc.bwp_start);
        EXPECT_EQ(ue.nCw,          1u);
        EXPECT_EQ(ue.nUeLayers,    tc.num_of_layers);
        EXPECT_EQ(ue.dataScramId,  tc.data_scrambling_id);
        EXPECT_EQ(ue.dmrsPortBmsk, tc.dmrs_ports & 0xFFFu);
        EXPECT_EQ(ue.scid,         tc.sc_id);
        EXPECT_EQ(ue.dmrsScrmId,   tc.dl_dmrs_scrambling_id);
        EXPECT_EQ(ue.refPoint,     tc.ref_point);
    }
}

TEST(PdschPduParser, Parse_CodewordFields_TableDriven)
{
    struct Case {
        std::string_view description;
        uint8_t  mcs_table;
        uint8_t  mcs_index;
        uint16_t target_code_rate;
        uint8_t  qam_mod_order;
        uint8_t  rv_index;
        uint32_t tb_size;
        uint8_t  expected_max_qm;  ///< 256-QAM table (mcs_table=1) → maxQm=8.
    };

    // All cases use bwp_size=59 → n_PRB_LBRM=66 (LBRM tested in Group C).
    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "256-QAM: mcs_idx=20 rv=0 tb=8000 → maxQm=8",
            1u, 20u, 948u, 8u, 0u, 8000u, 8u,
        },
        {
            "256-QAM: mcs_idx=5 rv=1 tb=1000 → maxQm=8",
            1u,  5u, 193u, 2u, 1u, 1000u, 8u,
        },
        {
            "256-QAM: mcs_idx=28 rv=3 tb=50000 → maxQm=8",
            1u, 28u, 948u, 8u, 3u, 50000u, 8u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        parser.setup_cell(req, 0u);

        PdschPduParams p{};
        p.bwp_size         = 59u;
        p.mcs_table        = tc.mcs_table;
        p.mcs_index        = tc.mcs_index;
        p.target_code_rate = tc.target_code_rate;
        p.qam_mod_order    = tc.qam_mod_order;
        p.rv_index         = tc.rv_index;
        p.tb_size          = tc.tb_size;

        // Act
        const auto body = build_pdsch_pdu_body(p);
        const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu));

        // Assert
        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);
        ASSERT_EQ(params->cell_grp_info.nCws, 1u);

        const auto& cw = params->ue_cw_info[0];
        EXPECT_EQ(cw.mcsTableIndex,  tc.mcs_table);
        EXPECT_EQ(cw.mcsIndex,       tc.mcs_index);
        EXPECT_EQ(cw.targetCodeRate, tc.target_code_rate);
        EXPECT_EQ(cw.qamModOrder,    tc.qam_mod_order);
        EXPECT_EQ(cw.rv,             tc.rv_index);
        EXPECT_EQ(cw.tbSize,         tc.tb_size);
        EXPECT_EQ(cw.maxQm,          tc.expected_max_qm);
        EXPECT_EQ(cw.n_PRB_LBRM,    66u);   // BWP=59 → bucket 33–66 → 66
        EXPECT_EQ(cw.tbStartOffset,  0u);   // first (only) codeword
        EXPECT_EQ(cw.maxLayers,      4u);   // k_max_layers invariant
    }
}

TEST(PdschPduParser, Parse_UeGroupFields_TableDriven)
{
    struct Case {
        std::string_view description;
        uint16_t dl_dmrs_sym_pos;
        uint8_t  start_sym;
        uint8_t  num_sym;
        uint16_t rb_start;
        uint16_t rb_size;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "non-zero DMRS pattern, sym 3–12, rb 10–34",
            0x0088u, 3u, 10u, 10u, 25u,
        },
        {
            "single-bit DMRS pattern, sym 2–13, full 50-PRB allocation",
            0x0080u, 2u, 12u,  0u, 50u,
        },
        {
            "two-bit DMRS pattern, sym 1–14, rb 5–34",
            0x00C0u, 1u, 14u,  5u, 30u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        parser.setup_cell(req, 0u);

        PdschPduParams p{};
        p.dl_dmrs_sym_pos = tc.dl_dmrs_sym_pos;
        p.start_sym_index = tc.start_sym;
        p.num_symbols     = tc.num_sym;
        p.resource_alloc  = 1u;   // RA-Type 1
        p.rb_start        = tc.rb_start;
        p.rb_size         = tc.rb_size;
        p.bwp_start       = 0u;

        // Act
        const auto body = build_pdsch_pdu_body(p);
        const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu));

        // Assert
        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);
        ASSERT_EQ(params->cell_grp_info.nUeGrps, 1u);

        // Structural invariants
        EXPECT_EQ(params->cell_grp_info.nCells,  1u);
        EXPECT_EQ(params->cell_grp_info.nUes,    1u);
        EXPECT_EQ(params->cell_grp_info.nCws,    1u);

        const auto& grp = params->ue_grp_info[0];
        EXPECT_EQ(grp.dmrsSymLocBmsk, tc.dl_dmrs_sym_pos);
        EXPECT_EQ(grp.pdschStartSym,  tc.start_sym);
        EXPECT_EQ(grp.nPdschSym,      tc.num_sym);
        EXPECT_EQ(grp.nUes,           1u);
        EXPECT_EQ(grp.resourceAlloc,  1u);   // RA-Type 1 as set in all cases
        EXPECT_NE(grp.pCellPrm,       nullptr);
        EXPECT_NE(grp.pDmrsDynPrm,    nullptr);
    }
}

// ===========================================================================
// Group C — LBRM lookup (table-driven)
// ===========================================================================

TEST(PdschPduParser, LbrmLookup_TableDriven)
{
    struct LbrmCase
    {
        std::string_view description;
        uint16_t         bwp_size;
        uint16_t         expected_lbrm;
    };

    static constexpr std::array k_lbrm_cases = std::to_array<LbrmCase>({
        {"BWP=25: first bucket (≤32 → 32)",         25u,  32u},
        {"BWP=32: exact first-bucket boundary",      32u,  32u},
        {"BWP=33: second bucket start (33–66 → 66)", 33u,  66u},
        {"BWP=59: 59C scenario (33–66 → 66)",        59u,  66u},
        {"BWP=66: exact second-bucket boundary",     66u,  66u},
        {"BWP=67: third bucket start (67–107 → 107)",67u, 107u},
        {"BWP=69: 69-PRB scenario",                  69u, 107u},
        {"BWP=273: maximum table entry",            273u, 273u},
        {"BWP=300: beyond table → clamped to 273",  300u, 273u},
    });

    for (const auto& tc : k_lbrm_cases) {
        SCOPED_TRACE(tc.description);

        MockDlModuleView view{};
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        parser.setup_cell(req, 0u);

        PdschPduParams p{};
        p.bwp_size = tc.bwp_size;

        const auto body = build_pdsch_pdu_body(p);
        const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu));

        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);
        EXPECT_EQ(params->ue_cw_info[0].n_PRB_LBRM, tc.expected_lbrm);
    }
}

// ===========================================================================
// Group D — power-control beta LUT (table-driven)
// ===========================================================================

TEST(PdschPduParser, PowerControlBeta_TableDriven)
{
    struct BetaCase
    {
        std::string_view description;
        uint8_t          offset;
        uint8_t          offset_ss;
        float            expected_beta;
        float            tolerance;
    };

    // Formula: exponent = (offset - 8 + (offset_ss - 1) * 3) / 20.0f
    // beta = 10^exponent; offset is clamped to [0, 23].
    static const std::array k_beta_cases = std::to_array<BetaCase>({
        // offset=8, ss=1 → exponent=(0+0)/20=0.0 → 10^0=1.0
        {"offset=8 ss=1: exponent=0.0 → beta=1.0",
         8u, 1u, 1.0f, 1e-6f},
        // offset=0, ss=1 → exponent=(-8+0)/20=-0.4 → 10^-0.4≈0.39811
        {"offset=0 ss=1: exponent=-0.4 → beta≈0.398",
         0u, 1u, std::pow(10.0f, -0.4f), 1e-4f},
        // offset=23, ss=3 → exponent=(15+6)/20=1.05 → 10^1.05≈11.22
        {"offset=23 ss=3: exponent=1.05 → beta≈11.22",
         23u, 3u, std::pow(10.0f, 1.05f), 1e-3f},
        // offset=255 clamped to 23, ss=1 → exponent=(15+0)/20=0.75 → 10^0.75≈5.623
        {"offset=255 ss=1: clamped to 23, exponent=0.75 → beta≈5.623",
         255u, 1u, std::pow(10.0f, 0.75f), 1e-3f},
    });

    for (const auto& tc : k_beta_cases) {
        SCOPED_TRACE(tc.description);

        MockDlModuleView view{};
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        parser.setup_cell(req, 0u);

        PdschPduParams p{};
        p.power_ctrl_offset    = tc.offset;
        p.power_ctrl_offset_ss = tc.offset_ss;

        const auto body = build_pdsch_pdu_body(p);
        const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu));

        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);

        const auto& ue = params->ue_info[0];
        EXPECT_NEAR(ue.beta_qam,  tc.expected_beta, tc.tolerance);
        EXPECT_FLOAT_EQ(ue.beta_qam, ue.beta_dmrs)
            << "beta_qam and beta_dmrs must be identical";
    }
}

// ===========================================================================
// Group E — UE-group reuse (table-driven)
//
// Two UEs are always parsed. The second UE's resource allocation varies per
// case.  When the second UE matches the first UE's full resource key
// (rb_start, rb_size, start_sym, num_sym, dmrs_sym_pos), the group is reused
// and nUeGrps stays at 1; any difference creates a new group (nUeGrps=2).
//
// Invariants for every case:
//   nUes == 2, nCws == 2
//   ue_cw_info[0].tbStartOffset == 0
//   ue_cw_info[1].tbStartOffset == p1.tb_size (= 1000u)
// ===========================================================================

TEST(PdschPduParser, UeGroupReuse_TableDriven)
{
    struct Case {
        std::string_view description;
        // Second UE's resource allocation (first UE is fixed as reference).
        uint16_t ue2_rb_start;
        uint16_t ue2_rb_size;
        uint8_t  ue2_start_sym;
        uint8_t  ue2_num_sym;
        // Expected group count
        uint16_t expected_nue_grps;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "identical resource allocation → group reused (nUeGrps=1)",
            0u, 50u, 2u, 12u, 1u,
        },
        {
            "different rb_start → new group (nUeGrps=2)",
            25u, 50u, 2u, 12u, 2u,
        },
        {
            "different rb_size → new group (nUeGrps=2)",
            0u, 25u, 2u, 12u, 2u,
        },
        {
            "different start_sym → new group (nUeGrps=2)",
            0u, 50u, 3u, 12u, 2u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        parser.setup_cell(req, 0u);

        // First UE — fixed reference parameters
        PdschPduParams p1{};
        p1.rnti            = 0x0001u;
        p1.resource_alloc  = 1u;
        p1.rb_start        = 0u;
        p1.rb_size         = 50u;
        p1.start_sym_index = 2u;
        p1.num_symbols     = 12u;
        p1.tb_size         = 1000u;

        const auto body1 = build_pdsch_pdu_body(p1);
        const auto& pdu1 = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body1.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu1));

        // Second UE — resource allocation varies by case
        PdschPduParams p2   = p1;
        p2.rnti             = 0x0002u;
        p2.tb_size          = 2000u;
        p2.rb_start         = tc.ue2_rb_start;
        p2.rb_size          = tc.ue2_rb_size;
        p2.start_sym_index  = tc.ue2_start_sym;
        p2.num_symbols      = tc.ue2_num_sym;

        const auto body2 = build_pdsch_pdu_body(p2);
        const auto& pdu2 = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body2.data());
        EXPECT_TRUE(parser.parse(0u, 0u, pdu2));

        // Assert
        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);

        // Structural invariants
        EXPECT_EQ(params->cell_grp_info.nCells,         1u);
        EXPECT_EQ(params->cell_grp_info.nUes,           2u);
        EXPECT_EQ(params->cell_grp_info.nCws,           2u);
        EXPECT_EQ(params->ue_cw_info[0].tbStartOffset,  0u);
        EXPECT_EQ(params->ue_cw_info[1].tbStartOffset,  p1.tb_size);  // cumulative

        // UE identity preserved regardless of grouping
        EXPECT_EQ(params->ue_info[0].rnti, 0x0001u);
        EXPECT_EQ(params->ue_info[1].rnti, 0x0002u);

        // Per-case assertion
        EXPECT_EQ(params->cell_grp_info.nUeGrps, tc.expected_nue_grps);

        if (tc.expected_nue_grps == 1u)
        {
            // Group reused: both UEs in the single group.
            EXPECT_EQ(params->ue_grp_info[0].nUes, 2u);
        }
        else
        {
            // New group allocated: each group owns exactly one UE.
            EXPECT_EQ(params->ue_grp_info[0].nUes, 1u);
            EXPECT_EQ(params->ue_grp_info[1].nUes, 1u);
        }
    }
}

// ===========================================================================
// Group F — DLSlotProcessor PDSCH round-trip (table-driven)
//
// Verifies that process<PDSCH>() populates pdsch_params correctly for both
// single-message and multi-message batches.
//
// Invariants for every single-message case:
//   result.has_value(), nCells==1, nUes==1, nCws==1
//   cell_dyn_info[0].slotNum == expected_slot
//   ue_info[0].rnti          == expected_rnti
//
// The TwoMsgs case additionally verifies nCells==2, nUes==2.
// ===========================================================================

TEST(DLSlotProcessor, Process_Pdsch_TableDriven)
{
    enum class BatchSetup : uint8_t {
        SingleMsg, ///< One message; sfn/slot/rnti from tc.
        TwoMsgs,   ///< Two messages; second uses rnti=0x2222, slot=0.
    };

    struct Case {
        std::string_view description;
        BatchSetup       setup;
        uint16_t         sfn;
        uint16_t         slot;
        uint16_t         rnti;
        // Expected
        uint16_t         expected_n_cells;
        uint16_t         expected_n_ues;
        uint16_t         expected_slot;   ///< cell_dyn_info[0].slotNum
        uint16_t         expected_rnti;   ///< ue_info[0].rnti
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "single msg sfn=2 slot=5 rnti=0x5678 → slot and rnti propagated",
            BatchSetup::SingleMsg, 2u, 5u, 0x5678u,
            1u, 1u, 5u, 0x5678u,
        },
        {
            "single msg sfn=0 slot=0 rnti=0x1234 → boundary sfn/slot",
            BatchSetup::SingleMsg, 0u, 0u, 0x1234u,
            1u, 1u, 0u, 0x1234u,
        },
        {
            "single msg sfn=1023 slot=19 rnti=0xABCD → max sfn/slot boundary",
            BatchSetup::SingleMsg, 1023u, 19u, 0xABCDu,
            1u, 1u, 19u, 0xABCDu,
        },
        {
            "two-msg batch → nCells=2 nUes=2; first slot/rnti checked",
            BatchSetup::TwoMsgs, 0u, 7u, 0x1111u,
            2u, 2u, 7u, 0x1111u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

        PdschPduParams p1{};
        p1.rnti     = tc.rnti;
        p1.bwp_size = 59u;
        FapiDlMsg msg1{tc.sfn, tc.slot, build_pdsch_pdu_body(p1)};

        PdschPduParams p2{};
        p2.rnti     = 0x2222u;
        p2.bwp_size = 59u;
        FapiDlMsg msg2{0u, 0u, build_pdsch_pdu_body(p2)};

        // Act
        const scf_5g_fapi::SlotParseResult result =
            [&]() -> scf_5g_fapi::SlotParseResult
        {
            if (tc.setup == BatchSetup::SingleMsg)
            {
                return processor.process<DL_TTI_PDU_TYPE_PDSCH>(
                    std::span{&msg1.desc, 1u});
            }
            const std::array descs{msg1.desc, msg2.desc};
            return processor.process<DL_TTI_PDU_TYPE_PDSCH>(
                std::span{descs.data(), descs.size()});
        }();

        // Assert
        ASSERT_TRUE(result.has_value());

        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);

        // Primary per-case assertions
        EXPECT_EQ(params->cell_grp_info.nCells,     tc.expected_n_cells);
        EXPECT_EQ(params->cell_grp_info.nUes,       tc.expected_n_ues);
        EXPECT_EQ(params->cell_grp_info.nCws,       tc.expected_n_ues);  // 1 CW per UE
        EXPECT_EQ(params->cell_dyn_info[0].slotNum, tc.expected_slot);
        EXPECT_EQ(params->ue_info[0].rnti,          tc.expected_rnti);

        // Secondary assertions: LBRM and first CW offset are stable across all cases
        EXPECT_EQ(params->ue_cw_info[0].n_PRB_LBRM,   66u);  // bwp_size=59 → 66
        EXPECT_EQ(params->ue_cw_info[0].tbStartOffset, 0u);

        // For the two-message case verify the second cell/UE as well.
        if (tc.setup == BatchSetup::TwoMsgs)
        {
            EXPECT_EQ(params->ue_info[1].rnti,              0x2222u);
            EXPECT_EQ(params->cell_dyn_info[1].slotNum,     0u);   // msg2 uses slot=0
            EXPECT_EQ(params->ue_cw_info[1].tbStartOffset,  0u);   // second cell, first CW
        }
    }
}

TEST(DLSlotProcessor, Process_PdschStats_SingleCodeword)
{
    MockDlModuleView view{};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    PdschPduParams p{};
    p.tb_size = 1234u;
    FapiDlMsg msg{2u, 5u, build_pdsch_pdu_body(p)};
    msg.desc.cell_id = 3u;

    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(view.published_stats_[3].bytes, 1234u);
    EXPECT_EQ(view.published_stats_[3].slots, 1u);
    EXPECT_EQ(view.publish_nonzero_cells_, 1u);
}

TEST(DLSlotProcessor, Process_PdschStats_TwoCodewords)
{
    MockDlModuleView view{};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    PdschPduParams p{};
    p.num_codewords = 2u;
    p.tb_size       = 777u;
    FapiDlMsg msg{2u, 5u, build_pdsch_pdu_body(p)};
    msg.desc.cell_id = 4u;

    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(view.published_stats_[4].bytes, 1554u);
    EXPECT_EQ(view.published_stats_[4].slots, 1u);
    EXPECT_EQ(view.publish_nonzero_cells_, 1u);
}

TEST(DLSlotProcessor, Process_PdschStats_TwoCells)
{
    MockDlModuleView view{};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    PdschPduParams p0{};
    p0.tb_size = 1000u;
    FapiDlMsg msg0{0u, 1u, build_pdsch_pdu_body(p0)};
    msg0.desc.cell_id = 0u;

    PdschPduParams p1{};
    p1.tb_size = 2000u;
    FapiDlMsg msg1{0u, 1u, build_pdsch_pdu_body(p1)};
    msg1.desc.cell_id = 1u;

    const std::array descs{msg0.desc, msg1.desc};
    const auto result = processor.process<DL_TTI_PDU_TYPE_PDSCH>(
        std::span{descs.data(), descs.size()});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(view.published_stats_[0].bytes, 1000u);
    EXPECT_EQ(view.published_stats_[0].slots, 1u);
    EXPECT_EQ(view.published_stats_[1].bytes, 2000u);
    EXPECT_EQ(view.published_stats_[1].slots, 1u);
    EXPECT_EQ(view.publish_nonzero_cells_, 2u);
}

TEST(DLSlotProcessor, Process_PdschStats_ZeroPdschNoPublish)
{
    MockDlModuleView view{};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    FapiDlNoPdschMsg msg{0u, 1u, 2u};

    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg.desc, 1u});

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(view.published_stats_[2].bytes, 0u);
    EXPECT_EQ(view.published_stats_[2].slots, 0u);
    EXPECT_EQ(view.publish_nonzero_cells_, 0u);
}

TEST(DLSlotProcessor, Process_PdschStats_MalformedPdschNotCounted)
{
    MockDlModuleView view{};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    PdschPduParams p{};
    p.tb_size = 4096u;
    FapiDlMsg msg{0u, 1u, build_pdsch_pdu_body(p)};
    msg.desc.cell_id = 5u;
    generic_pdu_header(msg)->pdu_size =
        static_cast<uint16_t>(sizeof(scf_fapi_generic_pdu_info_t));

    const auto result =
        processor.process<DL_TTI_PDU_TYPE_PDSCH>(std::span{&msg.desc, 1u});

#ifdef SCF_FAPI_10_04
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, scf_5g_fapi::SlotParseError::Code::PerTypeMismatch);
#else
    ASSERT_TRUE(result.has_value());
#endif
    EXPECT_EQ(view.published_stats_[5].bytes, 0u);
    EXPECT_EQ(view.published_stats_[5].slots, 0u);
    EXPECT_EQ(view.publish_nonzero_cells_, 0u);
}

TEST(DLSlotProcessor, Process_PdschStats_PublishesPriorDeltasOnMalformedLaterMessage)
{
    MockDlModuleView view{};
    scf_5g_fapi::DLSlotProcessor<MockDlModuleView> processor{view};

    PdschPduParams good{};
    good.tb_size = 1111u;
    FapiDlMsg good_msg{0u, 1u, build_pdsch_pdu_body(good)};
    good_msg.desc.cell_id = 0u;

    PdschPduParams bad{};
    bad.tb_size = 2222u;
    FapiDlMsg bad_msg{0u, 1u, build_pdsch_pdu_body(bad)};
    bad_msg.desc.cell_id = 1u;
    generic_pdu_header(bad_msg)->pdu_size =
        static_cast<uint16_t>(sizeof(scf_fapi_generic_pdu_info_t));

    const std::array descs{good_msg.desc, bad_msg.desc};
    const auto result = processor.process<DL_TTI_PDU_TYPE_PDSCH>(
        std::span{descs.data(), descs.size()});

#ifdef SCF_FAPI_10_04
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, scf_5g_fapi::SlotParseError::Code::PerTypeMismatch);
#else
    ASSERT_TRUE(result.has_value());
#endif
    EXPECT_EQ(view.published_stats_[0].bytes, 1111u);
    EXPECT_EQ(view.published_stats_[0].slots, 1u);
    EXPECT_EQ(view.published_stats_[1].bytes, 0u);
    EXPECT_EQ(view.published_stats_[1].slots, 0u);
    EXPECT_EQ(view.publish_nonzero_cells_, 1u);
}

// ===========================================================================
// Group G — apply_pm_weights (table-driven)
//
// apply_pm_weights() is reached only when pm_enabled=true AND mmimo=false.
// setup_beamforming() sets ue.enablePrcdBf = pm_enabled() before the call;
// apply_pm_weights() resets it to 0 on every early-exit path (map miss,
// pm_idx==0, dig_bf_interfaces==0).
//
// Per-case variables: dig_bf_interfaces, pm_idx, pm_map populated
// Expected:           nPrecodingMatrices, enablePrcdBf
// ===========================================================================

TEST(PdschPduParser, ApplyPmWeights_TableDriven)
{
    struct Case {
        std::string_view description;
        uint8_t          dig_bf_interfaces;
        uint16_t         pm_idx;       ///< put in pm_idx_and_beam_idx[0]
        bool             populate_pm_map;
        // Expected
        uint32_t         expected_n_prec_matrices;
        uint8_t          expected_enable_prcd_bf;
    };

    static constexpr std::array k_cases = std::to_array<Case>({
        {
            "empty pm_map, non-zero pm_idx → map miss, enablePrcdBf cleared",
            /*dig_bf_interfaces=*/ 1u, /*pm_idx=*/ 0x0001u,
            /*populate_pm_map=*/ false,
            /*expected_n_prec_matrices=*/ 0u, /*expected_enable_prcd_bf=*/ 0u,
        },
        {
            "populated pm_map, matching pm_idx → weights inserted, enablePrcdBf kept",
            /*dig_bf_interfaces=*/ 1u, /*pm_idx=*/ 0x0001u,
            /*populate_pm_map=*/ true,
            /*expected_n_prec_matrices=*/ 1u, /*expected_enable_prcd_bf=*/ 1u,
        },
        {
            "pm_idx == 0 → dynamic-BFW skip, enablePrcdBf cleared",
            /*dig_bf_interfaces=*/ 1u, /*pm_idx=*/ 0x0000u,
            /*populate_pm_map=*/ true,
            /*expected_n_prec_matrices=*/ 0u, /*expected_enable_prcd_bf=*/ 0u,
        },
        {
            "dig_bf_interfaces == 0 → dynamic-BFW path skip, enablePrcdBf cleared",
            /*dig_bf_interfaces=*/ 0u, /*pm_idx=*/ 0x0001u,
            /*populate_pm_map=*/ true,
            /*expected_n_prec_matrices=*/ 0u, /*expected_enable_prcd_bf=*/ 0u,
        },
    });

    for (const auto& tc : k_cases) {
        SCOPED_TRACE(tc.description);

        // Arrange
        MockDlModuleView view{};
        view.pm_enabled_    = true;
        view.mmimo_enabled_ = false;  // enables apply_pm_weights

        if (tc.populate_pm_map) {
            slot_command_api::pm_weights_t w{};
            w.layers         = 1u;
            w.ports          = 2u;
            w.weights.nPorts = 2u;  // must match ports for consistency check
            // key = pm_idx | (carrier_id << 16); carrier_id() returns 0.
            view.pm_map_.emplace(static_cast<uint32_t>(tc.pm_idx), w);
        }

        scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

        scf_fapi_dl_tti_req_t req{};
        req.slot = 0u;
#ifdef SCF_FAPI_10_04
        req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
#endif
        parser.setup_cell(req, 0u);

        PdschPduParams p{};
        p.dig_bf_interfaces = tc.dig_bf_interfaces;
        p.num_prgs          = 1u;
        // num_prgs * (dig_bf_interfaces + 1) entries; always provide one entry
        // so the loop can read pm_idx_and_beam_idx[0] regardless of dig_bf.
        const uint16_t n_entries =
            static_cast<uint16_t>(p.num_prgs)
            * static_cast<uint16_t>(p.dig_bf_interfaces + 1u);
        p.pm_idx_and_beam_idx.assign(n_entries, 0u);
        p.pm_idx_and_beam_idx[0] = tc.pm_idx;

        const auto pdu_body = build_pdsch_pdu_body(p);
        const auto& pdu =
            *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(pdu_body.data());

        // Act
        const bool ok = parser.parse(0u, 0u, pdu);

        // Assert
        ASSERT_TRUE(ok);

        auto* params = get_params(view);
        ASSERT_NE(params, nullptr);

        EXPECT_EQ(params->cell_grp_info.nPrecodingMatrices, tc.expected_n_prec_matrices);
        EXPECT_EQ(params->ue_info[0].enablePrcdBf,          tc.expected_enable_prcd_bf);

        if (tc.expected_n_prec_matrices > 0u) {
            EXPECT_EQ(params->ue_info[0].pmwPrmIdx,        0u);
            EXPECT_NE(params->cell_grp_info.pPmwPrms,      nullptr);
            EXPECT_EQ(params->pm_info[0].nPorts,           2u);
        }
    }
}

// ===========================================================================
// Group H — FH params population
//
// Verifies that the parser correctly populates cell_group_command::fh_params
// when is_fapi_to_cplane_direct_enabled() == false (the standard path), and
// skips population (NullFhCallable specialization) when it returns true.
//
// Coverage:
//   H1. Single PDU → 1 FH entry; field-by-field check
//   H2. Direct mode (is_fapi_to_cplane_direct_enabled == true) → 0 FH entries
//   H3. Multiple PDUs in same cell → counters increment, start_index preserved
//   H4. mMIMO dynamic BFW (mmimo && dig_bf_ifaces == 0) → bfwCoeff_mem_info == nullptr
// ===========================================================================

TEST(PdschPduParser, FhParams_PopulatedOnSinglePdu)
{
    // Arrange
    MockDlModuleView view{};
    view.fh_populate_enabled_ = true;   // legacy / populate mode
    view.num_dl_prb_          = 59u;
    view.carrier_id_lookup_[0] = 7;      // distinguishable cell_index in fh_params

    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

    scf_fapi_dl_tti_req_t req{};
    req.slot = 4u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
#endif
    parser.setup_cell(req, 0u);

    PdschPduParams p{};
    const auto pdu_body = build_pdsch_pdu_body(p);
    const auto& pdu     = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(pdu_body.data());

    // Act
    const bool ok = parser.parse(0u, req.slot, pdu);

    // Assert
    ASSERT_TRUE(ok);
    auto* group_cmd = view.group_command();
    ASSERT_NE(group_cmd, nullptr);
    auto& fh = group_cmd->fh_params;

    EXPECT_EQ(fh.total_num_pdsch_pdus,       1u);
    EXPECT_EQ(fh.num_pdsch_fh_params[7],     1u);
    EXPECT_EQ(fh.start_index_pdsch_fh_params[7], 0u);

    const auto& entry = fh.pdsch_fh_params[0];
    EXPECT_EQ(entry.cell_index,       7u);
    EXPECT_EQ(entry.num_dl_prb,       59u);
    EXPECT_EQ(entry.bf_enabled,       view.bf_enabled_);
    EXPECT_EQ(entry.pm_enabled,       view.pm_enabled_);
    EXPECT_EQ(entry.mmimo_enabled,    view.mmimo_enabled_);
    EXPECT_EQ(entry.csirs_compact_mode, false);
    EXPECT_EQ(entry.is_new_grp,       true);
    EXPECT_EQ(entry.ue_grp_index,     0u);
    EXPECT_EQ(entry.pc_bf,            &fh.pc_bf_arr[0]);
    EXPECT_EQ(entry.ue,               &get_params(view)->ue_info[0]);
    EXPECT_EQ(entry.grp,              &get_params(view)->ue_grp_info[0]);

    // mmimo=true + dig_bf_interfaces==0 → dynamic BFW path → real pointer
    // queried from the view. MockDlModuleView::bfw_ defaults to nullptr, so
    // we expect nullptr here even though the dynamic-BFW path is taken.
    EXPECT_EQ(entry.bfwCoeff_mem_info, view.bfw_);

    EXPECT_EQ(fh.pc_bf_arr[0].num_prgs,          p.num_prgs);
    EXPECT_EQ(fh.pc_bf_arr[0].prg_size,          p.prg_size);
    EXPECT_EQ(fh.pc_bf_arr[0].dig_bf_interfaces, p.dig_bf_interfaces);
}

TEST(PdschPduParser, FhParams_SkippedInDirectMode)
{
    // Arrange: direct mode on → is_fapi_to_cplane_direct_enabled() returns true.
    MockDlModuleView view{};
    view.fh_populate_enabled_ = false;
    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

    scf_fapi_dl_tti_req_t req{};
    req.slot = 4u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
#endif
    parser.setup_cell(req, 0u);

    PdschPduParams p{};
    const auto pdu_body = build_pdsch_pdu_body(p);
    const auto& pdu     = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(pdu_body.data());

    // Act
    const bool ok = parser.parse(0u, req.slot, pdu);

    // Assert: parser succeeded but FH counters remain at zero.
    ASSERT_TRUE(ok);
    auto& fh = view.group_command()->fh_params;
    EXPECT_EQ(fh.total_num_pdsch_pdus, 0u);
    for (uint8_t v : fh.num_pdsch_fh_params) { EXPECT_EQ(v, 0u); }

    // Non-FH parser state was still populated.
    auto* params = get_params(view);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->cell_grp_info.nUes,    1u);
    EXPECT_EQ(params->cell_grp_info.nUeGrps, 1u);
}

TEST(PdschPduParser, FhParams_MultiplePdusSameCell)
{
    // Arrange
    MockDlModuleView view{};
    view.fh_populate_enabled_ = true;
    view.carrier_id_lookup_[0] = 0;
    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

    scf_fapi_dl_tti_req_t req{};
    req.slot = 4u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 2u;
#endif
    parser.setup_cell(req, 0u);

    PdschPduParams p1{};
    p1.rnti = 0x1111;
    p1.rb_start = 0;  p1.rb_size = 25;
    PdschPduParams p2{};
    p2.rnti = 0x2222;
    p2.rb_start = 25; p2.rb_size = 25;  // distinct allocation → new UE group

    const auto body1 = build_pdsch_pdu_body(p1);
    const auto body2 = build_pdsch_pdu_body(p2);
    const auto& pdu1 = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body1.data());
    const auto& pdu2 = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(body2.data());

    // Act
    ASSERT_TRUE(parser.parse(0u, req.slot, pdu1));
    ASSERT_TRUE(parser.parse(0u, req.slot, pdu2));

    // Assert
    auto& fh = view.group_command()->fh_params;
    EXPECT_EQ(fh.total_num_pdsch_pdus,           2u);
    EXPECT_EQ(fh.num_pdsch_fh_params[0],         2u);
    EXPECT_EQ(fh.start_index_pdsch_fh_params[0], 0u);   // first PDU's offset

    EXPECT_EQ(fh.pdsch_fh_params[0].ue_grp_index, 0u);
    EXPECT_EQ(fh.pdsch_fh_params[0].is_new_grp,   true);
    EXPECT_EQ(fh.pdsch_fh_params[1].ue_grp_index, 1u);
    EXPECT_EQ(fh.pdsch_fh_params[1].is_new_grp,   true);
}

TEST(PdschPduParser, FhParams_MmimoDynamicBfw_PreservesPointer)
{
    // Arrange: mmimo + dig_bf_interfaces == 0 → dynamic BFW path.
    // Legacy contract (prepare_dl_slot_command L2317-2320 + update_cell_command L441):
    // only this path carries a real bfwCoeff_mem_info pointer through to fh_params.
    MockDlModuleView view{};
    view.fh_populate_enabled_ = true;
    view.mmimo_enabled_       = true;
    slot_command_api::bfw_coeff_mem_info_t fake_bfw{};
    view.bfw_ = &fake_bfw;

    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

    scf_fapi_dl_tti_req_t req{};
    req.slot = 0u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
#endif
    parser.setup_cell(req, 0u);

    PdschPduParams p{};
    p.num_prgs          = 1u;
    p.dig_bf_interfaces = 0u;  // dynamic BFW path → bfw pointer preserved

    const auto pdu_body = build_pdsch_pdu_body(p);
    const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(pdu_body.data());

    // Act
    ASSERT_TRUE(parser.parse(0u, req.slot, pdu));

    // Assert
    auto& fh = view.group_command()->fh_params;
    ASSERT_EQ(fh.total_num_pdsch_pdus, 1u);
    EXPECT_EQ(fh.pdsch_fh_params[0].bfwCoeff_mem_info, &fake_bfw);
    EXPECT_EQ(fh.pc_bf_arr[0].dig_bf_interfaces, 0u);
}

TEST(PdschPduParser, FhParams_MmimoStaticBfw_NullsPointer)
{
    // Arrange: mmimo + dig_bf_interfaces != 0 → static BFW path.
    // Even though the view exposes a non-null BFW, the parser must zero it
    // (legacy update_cell_command L441).
    MockDlModuleView view{};
    view.fh_populate_enabled_ = true;
    view.mmimo_enabled_       = true;
    slot_command_api::bfw_coeff_mem_info_t fake_bfw{};
    view.bfw_ = &fake_bfw;

    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

    scf_fapi_dl_tti_req_t req{};
    req.slot = 0u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
#endif
    parser.setup_cell(req, 0u);

    PdschPduParams p{};
    p.num_prgs          = 1u;
    p.dig_bf_interfaces = 2u;  // static BFW path → bfw nulled
    p.pm_idx_and_beam_idx = {1u, 10u, 20u};  // numPRGs * (1 + digBFI) = 3 entries copied

    const auto pdu_body = build_pdsch_pdu_body(p);
    const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(pdu_body.data());

    // Act
    ASSERT_TRUE(parser.parse(0u, req.slot, pdu));

    // Assert
    auto& fh = view.group_command()->fh_params;
    ASSERT_EQ(fh.total_num_pdsch_pdus, 1u);
    EXPECT_EQ(fh.pdsch_fh_params[0].bfwCoeff_mem_info, nullptr);  // L441 contract
    EXPECT_EQ(fh.pc_bf_arr[0].dig_bf_interfaces, 2u);
    // PM/beam indices still copied: !mmimo || dig>0 path runs.
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[0], 1u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[1], 10u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[2], 20u);
}

TEST(PdschPduParser, FhParams_NonMmimo_NullBfw)
{
    // Arrange: mmimo == false. Legacy caller never fetches bfwCoeff_mem_info,
    // so it lands in fh_params as nullptr regardless of dig_bf_interfaces.
    MockDlModuleView view{};
    view.fh_populate_enabled_ = true;
    view.mmimo_enabled_       = false;
    slot_command_api::bfw_coeff_mem_info_t fake_bfw{};
    view.bfw_ = &fake_bfw;   // view exposes one, but parser must not use it.

    scf_5g_fapi::PdschPduParser<MockDlModuleView> parser{view};

    scf_fapi_dl_tti_req_t req{};
    req.slot = 0u;
#ifdef SCF_FAPI_10_04
    req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1u;
#endif
    parser.setup_cell(req, 0u);

    PdschPduParams p{};
    p.num_prgs          = 1u;
    p.dig_bf_interfaces = 1u;
    p.pm_idx_and_beam_idx = {5u, 7u};  // 1 * 2 = 2 entries

    const auto pdu_body = build_pdsch_pdu_body(p);
    const auto& pdu = *reinterpret_cast<const scf_fapi_pdsch_pdu_t*>(pdu_body.data());

    // Act
    ASSERT_TRUE(parser.parse(0u, req.slot, pdu));

    // Assert
    auto& fh = view.group_command()->fh_params;
    ASSERT_EQ(fh.total_num_pdsch_pdus, 1u);
    EXPECT_EQ(fh.pdsch_fh_params[0].bfwCoeff_mem_info, nullptr);
}

// ===========================================================================
// Group I — fill_pdsch_fh_entry as a standalone builder (no parser instance)
//
// Exercises the free-function primitive directly with synthetic inputs.
// This is the cheapest test surface for field-by-field parity, capacity
// overflow handling, and the NullFhCallable specialization.
// ===========================================================================

TEST(FillPdschFhEntry, NullFhCallable_IsNoOp)
{
    slot_command_api::fh_prepare_callback_params fh{};
    // No FhFillArgs needed — NullFhCallable specialization ignores everything.
    const bool ok = scf_5g_fapi::append_fh_entry(fh, /*cell_index*/ 0u,
                                                 scf_5g_fapi::NullFhCallable{});
    EXPECT_TRUE(ok);  // NullFhCallable specialization always returns true.
    EXPECT_EQ(fh.total_num_pdsch_pdus, 0u);
    for (uint8_t v : fh.num_pdsch_fh_params) { EXPECT_EQ(v, 0u); }
}

TEST(FillPdschFhEntry, RealFill_PopulatesFieldsAndAdvancesCounters)
{
    slot_command_api::fh_prepare_callback_params fh{};
    slot_command_api::cell_sub_command cell_cmd{};
    cuphyPdschUeGrpPrm_t                ue_grp{};
    cuphyPdschUePrm_t                   ue{};
    scf_fapi_tx_precoding_beamforming_t pm_bf{};
    pm_bf.num_prgs          = 1u;
    pm_bf.prg_size          = 52u;
    pm_bf.dig_bf_interfaces = 0u;

    const scf_5g_fapi::FhFillArgs args{
        .cell_cmd     = cell_cmd,
        .ue_grp       = ue_grp,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = nullptr,
        .bfw_idx      = 0u,
        .ue_grp_index = 0u,
        .num_dl_prb   = 59u,
        .cell_index   = 3u,
        .flags        = scf_5g_fapi::make_fh_flags(/*is_new_grp*/    true,
                                                    /*bf_enabled*/    false,
                                                    /*pm_enabled*/    false,
                                                    /*mmimo_enabled*/ true),
    };

    const bool ok = scf_5g_fapi::fill_pdsch_fh_entry(fh, args);
    ASSERT_TRUE(ok);
    EXPECT_EQ(fh.total_num_pdsch_pdus,   1u);
    EXPECT_EQ(fh.num_pdsch_fh_params[3], 1u);

    EXPECT_EQ(fh.pdsch_fh_params[0].cell_index, 3u);
    EXPECT_EQ(fh.pdsch_fh_params[0].num_dl_prb, 59u);
    EXPECT_EQ(fh.pdsch_fh_params[0].ue,         &ue);
    EXPECT_EQ(fh.pdsch_fh_params[0].grp,        &ue_grp);
    EXPECT_EQ(fh.pdsch_fh_params[0].cell_cmd,   &cell_cmd);
    EXPECT_EQ(fh.pdsch_fh_params[0].pc_bf,      &fh.pc_bf_arr[0]);
    EXPECT_EQ(fh.pdsch_fh_params[0].csirs_compact_mode, false);
}

TEST(FillPdschFhEntry, CapacityOverflow_ReturnsFalse)
{
    slot_command_api::fh_prepare_callback_params fh{};
    fh.total_num_pdsch_pdus = slot_command_api::MAX_ALLOWED_PDSCH_PDUS_PER_SLOT;

    slot_command_api::cell_sub_command cell_cmd{};
    cuphyPdschUeGrpPrm_t                ue_grp{};
    cuphyPdschUePrm_t                   ue{};
    scf_fapi_tx_precoding_beamforming_t pm_bf{};

    const scf_5g_fapi::FhFillArgs args{
        .cell_cmd     = cell_cmd,
        .ue_grp       = ue_grp,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = nullptr,
        .bfw_idx      = 0u,
        .ue_grp_index = 0u,
        .num_dl_prb   = 59u,
        .cell_index   = 0u,
        .flags        = scf_5g_fapi::make_fh_flags(/*is_new_grp*/    false,
                                                    /*bf_enabled*/    false,
                                                    /*pm_enabled*/    false,
                                                    /*mmimo_enabled*/ true),
    };

    const bool ok = scf_5g_fapi::fill_pdsch_fh_entry(fh, args);
    EXPECT_FALSE(ok);
    // Counter not advanced past MAX.
    EXPECT_EQ(fh.total_num_pdsch_pdus, slot_command_api::MAX_ALLOWED_PDSCH_PDUS_PER_SLOT);
}

TEST(FillPdschFhEntry, PmIdxAndBeamIdx_CopiedWhenNotDynamicBfw)
{
    slot_command_api::fh_prepare_callback_params fh{};
    slot_command_api::cell_sub_command cell_cmd{};
    cuphyPdschUeGrpPrm_t                ue_grp{};
    cuphyPdschUePrm_t                   ue{};
    scf_fapi_tx_precoding_beamforming_t pm_bf{};
    pm_bf.num_prgs          = 2u;
    pm_bf.prg_size          = 52u;
    pm_bf.dig_bf_interfaces = 1u;     // 2 * (1+1) = 4 uint16_t entries copied
    pm_bf.pm_idx_and_beam_idx[0] = 11u;
    pm_bf.pm_idx_and_beam_idx[1] = 22u;
    pm_bf.pm_idx_and_beam_idx[2] = 33u;
    pm_bf.pm_idx_and_beam_idx[3] = 44u;

    const scf_5g_fapi::FhFillArgs args{
        .cell_cmd     = cell_cmd,
        .ue_grp       = ue_grp,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = nullptr,
        .bfw_idx      = 0u,
        .ue_grp_index = 0u,
        .num_dl_prb   = 59u,
        .cell_index   = 0u,
        // mmimo=false → copy runs
        .flags        = scf_5g_fapi::make_fh_flags(/*is_new_grp*/    false,
                                                    /*bf_enabled*/    false,
                                                    /*pm_enabled*/    false,
                                                    /*mmimo_enabled*/ false),
    };

    ASSERT_TRUE(scf_5g_fapi::fill_pdsch_fh_entry(fh, args));
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[0], 11u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[1], 22u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[2], 33u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[3], 44u);
}

TEST(FillPdschFhEntry, PmIdxAndBeamIdx_SkippedForDynamicBfw)
{
    slot_command_api::fh_prepare_callback_params fh{};
    // Seed destination with a sentinel that we expect to NOT be overwritten.
    fh.pc_bf_arr[0].pm_idx_and_beam_idx[0] = 0xBEEFu;

    slot_command_api::cell_sub_command cell_cmd{};
    cuphyPdschUeGrpPrm_t                ue_grp{};
    cuphyPdschUePrm_t                   ue{};
    scf_fapi_tx_precoding_beamforming_t pm_bf{};
    pm_bf.num_prgs          = 1u;
    pm_bf.dig_bf_interfaces = 0u;
    pm_bf.pm_idx_and_beam_idx[0] = 0xDEADu;

    const scf_5g_fapi::FhFillArgs args{
        .cell_cmd     = cell_cmd,
        .ue_grp       = ue_grp,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = nullptr,
        .bfw_idx      = 0u,
        .ue_grp_index = 0u,
        .num_dl_prb   = 59u,
        .cell_index   = 0u,
        // mmimo=true && dig_bf_interfaces==0 → skip copy
        .flags        = scf_5g_fapi::make_fh_flags(/*is_new_grp*/    false,
                                                    /*bf_enabled*/    false,
                                                    /*pm_enabled*/    false,
                                                    /*mmimo_enabled*/ true),
    };

    ASSERT_TRUE(scf_5g_fapi::fill_pdsch_fh_entry(fh, args));
    // Sentinel preserved — copy was skipped.
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[0], 0xBEEFu);
}

TEST(FillPdschFhEntry, PmIdxAndBeamIdx_RejectsOversizedStaticList)
{
    slot_command_api::fh_prepare_callback_params fh{};
    slot_command_api::cell_sub_command cell_cmd{};
    cuphyPdschUeGrpPrm_t                ue_grp{};
    cuphyPdschUePrm_t                   ue{};

    std::array<uint8_t, sizeof(scf_fapi_tx_precoding_beamforming_t)> pm_storage{};
    auto& pm_bf = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(pm_storage.data());
    pm_bf.num_prgs = static_cast<uint16_t>(
        scf_5g_fapi::detail::pm_idx_and_beam_idx_capacity() + 1u);
    pm_bf.dig_bf_interfaces = 0u;

    const scf_5g_fapi::FhFillArgs args{
        .cell_cmd     = cell_cmd,
        .ue_grp       = ue_grp,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = nullptr,
        .bfw_idx      = 0u,
        .ue_grp_index = 0u,
        .num_dl_prb   = 59u,
        .cell_index   = 0u,
        .flags        = scf_5g_fapi::make_fh_flags(/*is_new_grp*/    false,
                                                    /*bf_enabled*/    false,
                                                    /*pm_enabled*/    false,
                                                    /*mmimo_enabled*/ false),
    };

    EXPECT_FALSE(scf_5g_fapi::fill_pdsch_fh_entry(fh, args));
    EXPECT_EQ(fh.total_num_pdsch_pdus, 0u);
    EXPECT_EQ(fh.num_pdsch_fh_params[0], 0u);
}

TEST(FillPdschFhEntry, PmIdxAndBeamIdx_AllowsBoundaryStaticList)
{
    slot_command_api::fh_prepare_callback_params fh{};
    slot_command_api::cell_sub_command cell_cmd{};
    cuphyPdschUeGrpPrm_t                ue_grp{};
    cuphyPdschUePrm_t                   ue{};

    constexpr std::size_t k_capacity =
        scf_5g_fapi::detail::pm_idx_and_beam_idx_capacity();
    std::vector<uint8_t> pm_storage(
        sizeof(scf_fapi_tx_precoding_beamforming_t) + k_capacity * sizeof(uint16_t));
    auto& pm_bf = *reinterpret_cast<scf_fapi_tx_precoding_beamforming_t*>(pm_storage.data());
    pm_bf.num_prgs = static_cast<uint16_t>(k_capacity);
    pm_bf.dig_bf_interfaces = 0u;
    for (std::size_t i = 0u; i < k_capacity; ++i)
    {
        pm_bf.pm_idx_and_beam_idx[i] = static_cast<uint16_t>(i);
    }

    const scf_5g_fapi::FhFillArgs args{
        .cell_cmd     = cell_cmd,
        .ue_grp       = ue_grp,
        .ue           = ue,
        .pm_bf        = pm_bf,
        .bfw          = nullptr,
        .bfw_idx      = 0u,
        .ue_grp_index = 0u,
        .num_dl_prb   = 59u,
        .cell_index   = 0u,
        .flags        = scf_5g_fapi::make_fh_flags(/*is_new_grp*/    false,
                                                    /*bf_enabled*/    false,
                                                    /*pm_enabled*/    false,
                                                    /*mmimo_enabled*/ false),
    };

    ASSERT_TRUE(scf_5g_fapi::fill_pdsch_fh_entry(fh, args));
    EXPECT_EQ(fh.total_num_pdsch_pdus, 1u);
    EXPECT_EQ(fh.num_pdsch_fh_params[0], 1u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[0], 0u);
    EXPECT_EQ(fh.pc_bf_arr[0].pm_idx_and_beam_idx[k_capacity - 1u],
              static_cast<uint16_t>(k_capacity - 1u));
}

} // namespace
