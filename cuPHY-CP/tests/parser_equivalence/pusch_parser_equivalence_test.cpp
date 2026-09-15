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
 * @file pusch_parser_equivalence_test.cpp
 * @brief Functional acceptance test: legacy vs new PUSCH parser equivalence.
 *
 * Generates real FAPI UL_TTI.request messages with PUSCH PDUs from a testMAC
 * launch pattern (via @ref cuphy_cp::tests::FapiMessageSource, which keeps every
 * testMAC header out of this parser-side TU), then feeds the same PDUs through:
 *
 *   - the legacy populator : @c scf_5g_fapi::update_cell_command(pusch)
 *   - the new parser path   : @c ULSlotProcessor<V>::process<UL_TTI_PDU_TYPE_PUSCH>
 *
 * and asserts the two resulting @c pusch_params snapshots agree field by field.
 * The legacy populator is the oracle; this is a differential test, not a
 * recomputation of the parser's own arithmetic (cpp-checklist §13.6).
 *
 * Data-driven: GTEST_SKIPs when the launch pattern / config assets are absent.
 *
 * Reference: docs/plans/pusch_parser_hld.plan.md §10 "Phase 4 (MR-4)".
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>  // std::getenv / ::setenv / ::unsetenv (env-override asset tests)
#include <optional>
#include <span>
#include <string>

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

// ---------------------------------------------------------------------------
// UCI-on-PUSCH comparison (only reached when both UEs carry a UCI block).
//
// Covers every field the PUSCH parse step writes into cuphyUciOnPuschPrm_t,
// including the FAPIv3 CSI-Part2 sizing parameters reached through
// pCalcCsi2SizePrms (compared by pointee, never by pointer value -- the two
// paths point into their own csip2_v3_params arrays).
// ---------------------------------------------------------------------------
void expect_uci_match(const cuphyUciOnPuschPrm_t& a, const cuphyUciOnPuschPrm_t& b, uint16_t i)
{
    EXPECT_EQ(a.nBitsHarq,         b.nBitsHarq)         << "uci[" << i << "].nBitsHarq";
    EXPECT_EQ(a.nBitsCsi1,         b.nBitsCsi1)         << "uci[" << i << "].nBitsCsi1";
    EXPECT_EQ(a.alphaScaling,      b.alphaScaling)      << "uci[" << i << "].alphaScaling";
    EXPECT_EQ(a.betaOffsetHarqAck, b.betaOffsetHarqAck) << "uci[" << i << "].betaOffsetHarqAck";
    EXPECT_EQ(a.betaOffsetCsi1,    b.betaOffsetCsi1)    << "uci[" << i << "].betaOffsetCsi1";
    EXPECT_EQ(a.betaOffsetCsi2,    b.betaOffsetCsi2)    << "uci[" << i << "].betaOffsetCsi2";
    EXPECT_EQ(a.rankBitOffset,     b.rankBitOffset)     << "uci[" << i << "].rankBitOffset";
    EXPECT_EQ(a.nRanksBits,        b.nRanksBits)        << "uci[" << i << "].nRanksBits";
    EXPECT_EQ(a.nCsiReports,       b.nCsiReports)       << "uci[" << i << "].nCsiReports";
    EXPECT_FLOAT_EQ(a.DTXthreshold, b.DTXthreshold)     << "uci[" << i << "].DTXthreshold";
    EXPECT_EQ(a.nCsi2Reports,      b.nCsi2Reports)      << "uci[" << i << "].nCsi2Reports";

    ASSERT_EQ((a.pCalcCsi2SizePrms != nullptr), (b.pCalcCsi2SizePrms != nullptr))
        << "uci[" << i << "].pCalcCsi2SizePrms presence";
    if (a.pCalcCsi2SizePrms == nullptr || b.pCalcCsi2SizePrms == nullptr) { return; }
    for (uint16_t r = 0; r < a.nCsi2Reports; ++r)
    {
        const auto& ca = a.pCalcCsi2SizePrms[r];
        const auto& cb = b.pCalcCsi2SizePrms[r];
        EXPECT_EQ(ca.nPart1Prms,    cb.nPart1Prms)    << "uci[" << i << "].csi2[" << r << "].nPart1Prms";
        EXPECT_EQ(ca.csi2sizeMapIdx, cb.csi2sizeMapIdx) << "uci[" << i << "].csi2[" << r << "].csi2sizeMapIdx";
        for (uint8_t p = 0; p < ca.nPart1Prms; ++p)
        {
            EXPECT_EQ(ca.prmSizes[p],   cb.prmSizes[p])
                << "uci[" << i << "].csi2[" << r << "].prmSizes[" << +p << "]";
            EXPECT_EQ(ca.prmOffsets[p], cb.prmOffsets[p])
                << "uci[" << i << "].csi2[" << r << "].prmOffsets[" << +p << "]";
        }
    }
}

// ---------------------------------------------------------------------------
// Field-by-field comparison of two pusch_params snapshots.
//
// Compares every value the PUSCH parse step (legacy update_cell_command vs new
// ULSlotProcessor) writes -- all UE, UE-group, DMRS, UCI, per-cell-dynamic and
// cell-group bookkeeping fields. Pointer members (pUePrms, pCellPrm, pDmrsDynPrm,
// pUePrmIdxs, pUciPrms, pCalcCsi2SizePrms) are compared by pointee where they
// carry parsed data and never by pointer value, since each path points into its
// own arrays. Fields no PUSCH parse path writes are intentionally excluded:
// ue_info.N_slot_frame / nlAbove16 (never assigned) and
// pusch_params.forcedNumCsi2Bits (not initialised by the ctor). EXPECT_
// throughout so every mismatch surfaces in one run.
// ---------------------------------------------------------------------------
void expect_pusch_params_match(const slot_command_api::pusch_params& legacy,
                               const slot_command_api::pusch_params& fresh)
{
    ASSERT_EQ(legacy.cell_grp_info.nCells,  fresh.cell_grp_info.nCells)  << "nCells";
    ASSERT_EQ(legacy.cell_grp_info.nUes,    fresh.cell_grp_info.nUes)    << "nUes";
    ASSERT_EQ(legacy.cell_grp_info.nUeGrps, fresh.cell_grp_info.nUeGrps) << "nUeGrps";

    // --- Cell-group bookkeeping ---
    ASSERT_EQ(legacy.cell_index_list.size(), fresh.cell_index_list.size()) << "cell_index_list size";
    ASSERT_EQ(legacy.phy_cell_index_list.size(), fresh.phy_cell_index_list.size())
        << "phy_cell_index_list size";
    ASSERT_EQ(legacy.scf_ul_tti_handle_list.size(), fresh.scf_ul_tti_handle_list.size())
        << "scf_ul_tti_handle_list size";
    for (std::size_t c = 0; c < legacy.cell_index_list.size(); ++c)
    {
        EXPECT_EQ(legacy.cell_index_list[c], fresh.cell_index_list[c])
            << "cell_index_list[" << c << "]";
        EXPECT_EQ(legacy.phy_cell_index_list[c], fresh.phy_cell_index_list[c])
            << "phy_cell_index_list[" << c << "]";
        // Per-logical-cell (carrier-indexed) group bookkeeping.
        const auto cell = static_cast<std::size_t>(legacy.cell_index_list[c]);
        EXPECT_EQ(legacy.cell_ue_group_idx_start[cell], fresh.cell_ue_group_idx_start[cell])
            << "cell_ue_group_idx_start[" << cell << "]";
        EXPECT_EQ(legacy.nue_grps_per_cell[cell], fresh.nue_grps_per_cell[cell])
            << "nue_grps_per_cell[" << cell << "]";
    }
    for (std::size_t h = 0; h < legacy.scf_ul_tti_handle_list.size(); ++h)
    {
        EXPECT_EQ(legacy.scf_ul_tti_handle_list[h], fresh.scf_ul_tti_handle_list[h])
            << "scf_ul_tti_handle_list[" << h << "]";
    }

    // --- Per-cell dynamic params (dense over [0, nCells)) ---
    const uint16_t n_cells = legacy.cell_grp_info.nCells;
    for (uint16_t c = 0; c < n_cells; ++c)
    {
        const auto& ca = legacy.cell_dyn_info[c];
        const auto& cb = fresh.cell_dyn_info[c];
        EXPECT_EQ(ca.cellPrmStatIdx, cb.cellPrmStatIdx) << "cell_dyn_info[" << c << "].cellPrmStatIdx";
        EXPECT_EQ(ca.cellPrmDynIdx,  cb.cellPrmDynIdx)  << "cell_dyn_info[" << c << "].cellPrmDynIdx";
        EXPECT_EQ(ca.slotNum,        cb.slotNum)        << "cell_dyn_info[" << c << "].slotNum";
    }

    // --- Per-UE params ---
    const uint16_t n_ues = legacy.cell_grp_info.nUes;
    for (uint16_t i = 0; i < n_ues; ++i)
    {
        const auto& a = legacy.ue_info[i];
        const auto& b = fresh.ue_info[i];
        EXPECT_EQ(a.rnti,           b.rnti)           << "ue_info[" << i << "].rnti";
        EXPECT_EQ(a.puschIdentity,  b.puschIdentity)  << "ue_info[" << i << "].puschIdentity";
        EXPECT_EQ(a.scid,           b.scid)           << "ue_info[" << i << "].scid";
        EXPECT_EQ(a.dmrsPortBmsk,   b.dmrsPortBmsk)   << "ue_info[" << i << "].dmrsPortBmsk";
        EXPECT_EQ(a.mcsTableIndex,  b.mcsTableIndex)  << "ue_info[" << i << "].mcsTableIndex";
        EXPECT_EQ(a.mcsIndex,       b.mcsIndex)       << "ue_info[" << i << "].mcsIndex";
        EXPECT_EQ(a.dataScramId,    b.dataScramId)    << "ue_info[" << i << "].dataScramId";
        EXPECT_EQ(a.nUeLayers,      b.nUeLayers)      << "ue_info[" << i << "].nUeLayers";
        EXPECT_EQ(a.targetCodeRate, b.targetCodeRate) << "ue_info[" << i << "].targetCodeRate";
        EXPECT_EQ(a.qamModOrder,    b.qamModOrder)    << "ue_info[" << i << "].qamModOrder";
        EXPECT_EQ(a.pduBitmap,      b.pduBitmap)      << "ue_info[" << i << "].pduBitmap";
        EXPECT_EQ(a.enableTfPrcd,   b.enableTfPrcd)   << "ue_info[" << i << "].enableTfPrcd";
        EXPECT_EQ(a.harqProcessId,  b.harqProcessId)  << "ue_info[" << i << "].harqProcessId";
        EXPECT_EQ(a.rv,             b.rv)             << "ue_info[" << i << "].rv";
        EXPECT_EQ(a.ndi,            b.ndi)            << "ue_info[" << i << "].ndi";
        EXPECT_EQ(a.TBSize,         b.TBSize)         << "ue_info[" << i << "].TBSize";
        EXPECT_EQ(a.ueGrpIdx,       b.ueGrpIdx)       << "ue_info[" << i << "].ueGrpIdx";
        // DFT-s-OFDM / sequence hopping.
        EXPECT_EQ(a.groupOrSequenceHopping, b.groupOrSequenceHopping)
            << "ue_info[" << i << "].groupOrSequenceHopping";
        EXPECT_EQ(a.N_symb_slot,           b.N_symb_slot)           << "ue_info[" << i << "].N_symb_slot";
        EXPECT_EQ(a.lowPaprGroupNumber,    b.lowPaprGroupNumber)    << "ue_info[" << i << "].lowPaprGroupNumber";
        EXPECT_EQ(a.lowPaprSequenceNumber, b.lowPaprSequenceNumber) << "ue_info[" << i << "].lowPaprSequenceNumber";
        // LBRM.
        EXPECT_EQ(a.i_lbrm,    b.i_lbrm)    << "ue_info[" << i << "].i_lbrm";
        EXPECT_EQ(a.maxLayers, b.maxLayers) << "ue_info[" << i << "].maxLayers";
        EXPECT_EQ(a.maxQm,     b.maxQm)     << "ue_info[" << i << "].maxQm";
        EXPECT_EQ(a.n_PRB_LBRM, b.n_PRB_LBRM) << "ue_info[" << i << "].n_PRB_LBRM";
        // LDPC / weighted-average CFO.
        EXPECT_FLOAT_EQ(a.foForgetCoeff, b.foForgetCoeff) << "ue_info[" << i << "].foForgetCoeff";
        EXPECT_EQ(a.ldpcEarlyTerminationPerUe, b.ldpcEarlyTerminationPerUe)
            << "ue_info[" << i << "].ldpcEarlyTerminationPerUe";
        EXPECT_EQ(a.ldpcMaxNumItrPerUe, b.ldpcMaxNumItrPerUe)
            << "ue_info[" << i << "].ldpcMaxNumItrPerUe";

        EXPECT_EQ(legacy.ue_tb_size[i], fresh.ue_tb_size[i]) << "ue_tb_size[" << i << "]";

        ASSERT_EQ((a.pUciPrms != nullptr), (b.pUciPrms != nullptr))
            << "ue_info[" << i << "].pUciPrms presence";
        if (a.pUciPrms != nullptr && b.pUciPrms != nullptr)
        {
            expect_uci_match(*a.pUciPrms, *b.pUciPrms, i);
        }
    }

    // --- Per-UE-group params (+ DMRS, + membership index list) ---
    const uint16_t n_grps = legacy.cell_grp_info.nUeGrps;
    for (uint16_t g = 0; g < n_grps; ++g)
    {
        const auto& a = legacy.ue_grp_info[g];
        const auto& b = fresh.ue_grp_info[g];
        EXPECT_EQ(a.nUes,          b.nUes)          << "ue_grp_info[" << g << "].nUes";
        EXPECT_EQ(a.puschStartSym, b.puschStartSym) << "ue_grp_info[" << g << "].puschStartSym";
        EXPECT_EQ(a.nPuschSym,     b.nPuschSym)     << "ue_grp_info[" << g << "].nPuschSym";
        EXPECT_EQ(a.startPrb,      b.startPrb)      << "ue_grp_info[" << g << "].startPrb";
        EXPECT_EQ(a.nPrb,          b.nPrb)          << "ue_grp_info[" << g << "].nPrb";
        EXPECT_EQ(a.prgSize,       b.prgSize)       << "ue_grp_info[" << g << "].prgSize";
        EXPECT_EQ(a.enablePerPrgChEstPerUeg, b.enablePerPrgChEstPerUeg)
            << "ue_grp_info[" << g << "].enablePerPrgChEstPerUeg";
        EXPECT_EQ(a.nUplinkStreams, b.nUplinkStreams) << "ue_grp_info[" << g << "].nUplinkStreams";
        EXPECT_EQ(a.dmrsSymLocBmsk, b.dmrsSymLocBmsk) << "ue_grp_info[" << g << "].dmrsSymLocBmsk";
        EXPECT_EQ(a.rssiSymLocBmsk, b.rssiSymLocBmsk) << "ue_grp_info[" << g << "].rssiSymLocBmsk";

        // UE membership list (compared by pointee for [0, nUes)).
        ASSERT_NE(a.pUePrmIdxs, nullptr) << "ue_grp_info[" << g << "].pUePrmIdxs (legacy)";
        ASSERT_NE(b.pUePrmIdxs, nullptr) << "ue_grp_info[" << g << "].pUePrmIdxs (fresh)";
        for (uint16_t u = 0; u < a.nUes; ++u)
        {
            EXPECT_EQ(a.pUePrmIdxs[u], b.pUePrmIdxs[u])
                << "ue_grp_info[" << g << "].pUePrmIdxs[" << u << "]";
        }

        const auto& da = legacy.ue_dmrs_info[g];
        const auto& db = fresh.ue_dmrs_info[g];
        EXPECT_EQ(da.dmrsMaxLen,         db.dmrsMaxLen)         << "ue_dmrs_info[" << g << "].dmrsMaxLen";
        EXPECT_EQ(da.dmrsAddlnPos,       db.dmrsAddlnPos)       << "ue_dmrs_info[" << g << "].dmrsAddlnPos";
        EXPECT_EQ(da.nDmrsCdmGrpsNoData, db.nDmrsCdmGrpsNoData) << "ue_dmrs_info[" << g << "].nDmrsCdmGrpsNoData";
        EXPECT_EQ(da.dmrsScrmId,         db.dmrsScrmId)         << "ue_dmrs_info[" << g << "].dmrsScrmId";
    }
}

// ===========================================================================
// Fronthaul (C-plane) sym_prb comparison -- SEPARATE, TOGGLEABLE concern.
//
// This block validates the fronthaul PRB/symbol map
// (cell_sub_command::sym_prb_info()), which the new parser currently fills via
// PuschPduParser::update_fh_params. That responsibility is planned to move out
// of the parser. When it does, set @ref kCompareFhParams to false to drop the
// FH comparison (its single call site below is a plain runtime guard, so the
// build stays warning-clean), or delete this block plus that call site. None of
// it touches the pusch_params equivalence above, so the two concerns stay
// decoupled.
// ===========================================================================

/// Master switch for the fronthaul sym_prb comparison. Flip to false once the
/// new parser no longer populates cell_sub_command::sym_prb_info().
inline constexpr bool kCompareFhParams = true;

/// Compare the parsed fields of one fronthaul PRB section (pointer/handle
/// members in prb_info_t -- bfwCoeff buffers, cplane split caches -- are not
/// parse outputs and are excluded).
void expect_prb_common_match(const slot_command_api::prb_info_common_t& a,
                             const slot_command_api::prb_info_common_t& b, std::size_t p)
{
    EXPECT_EQ(a.startPrbc,     b.startPrbc)     << "prbs[" << p << "].startPrbc";
    EXPECT_EQ(a.numPrbc,       b.numPrbc)       << "prbs[" << p << "].numPrbc";
    EXPECT_EQ(a.numSymbols,    b.numSymbols)    << "prbs[" << p << "].numSymbols";
    EXPECT_EQ(a.reMask,        b.reMask)        << "prbs[" << p << "].reMask";
    EXPECT_EQ(a.extType,       b.extType)       << "prbs[" << p << "].extType";
    EXPECT_EQ(a.numApIndices,  b.numApIndices)  << "prbs[" << p << "].numApIndices";
    EXPECT_EQ(a.freqOffset,    b.freqOffset)    << "prbs[" << p << "].freqOffset";
    EXPECT_EQ(a.filterIndex,   b.filterIndex)   << "prbs[" << p << "].filterIndex";
    EXPECT_EQ(a.direction,     b.direction)     << "prbs[" << p << "].direction";
    EXPECT_EQ(a.portMask,      b.portMask)      << "prbs[" << p << "].portMask";
    EXPECT_EQ(a.ap_index,      b.ap_index)      << "prbs[" << p << "].ap_index";
    EXPECT_EQ(a.pdschPortMask, b.pdschPortMask) << "prbs[" << p << "].pdschPortMask";
}

// Compare the fronthaul PRB list and the per-symbol/per-channel index maps that
// drive C-plane section emission (the start-symbol-only vs every-symbol
// population difference surfaces in the symbols[] maps).
void expect_fh_sym_prb_match(const slot_command_api::slot_info_t& legacy,
                             const slot_command_api::slot_info_t& fresh)
{
    ASSERT_EQ(legacy.prbs_size, fresh.prbs_size) << "sym_prb_info.prbs_size";
    for (std::size_t p = 0; p < legacy.prbs_size; ++p)
    {
        expect_prb_common_match(legacy.prbs[p].common, fresh.prbs[p].common, p);
        EXPECT_EQ(legacy.prbs[p].beams_array_size, fresh.prbs[p].beams_array_size)
            << "prbs[" << p << "].beams_array_size";
        const std::size_t n_beams =
            std::min(legacy.prbs[p].beams_array_size, fresh.prbs[p].beams_array_size);
        for (std::size_t bm = 0; bm < n_beams; ++bm)
        {
            EXPECT_EQ(legacy.prbs[p].beams_array[bm], fresh.prbs[p].beams_array[bm])
                << "prbs[" << p << "].beams_array[" << bm << "]";
        }
    }

    for (std::size_t sym = 0; sym < legacy.symbols.size(); ++sym)
    {
        for (int ch = 0; ch < slot_command_api::channel_type::CHANNEL_MAX; ++ch)
        {
            const auto& la = legacy.symbols[sym][static_cast<std::size_t>(ch)];
            const auto& fb = fresh.symbols[sym][static_cast<std::size_t>(ch)];
            ASSERT_EQ(la.size(), fb.size())
                << "symbols[" << sym << "][ch=" << ch << "] size";
            for (std::size_t e = 0; e < la.size(); ++e)
            {
                EXPECT_EQ(la[e], fb[e]) << "symbols[" << sym << "][ch=" << ch << "][" << e << "]";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Asset-resolution failure coverage (every-runner: no CUDA, no data assets).
//
// FapiMessageSource::open_pusch() is the only entry the parser-side TU sees, so
// the deterministic nullopt branch reachable here is the first asset-resolution
// step. The deeper failure paths (CUDA acquire, load_launch_pattern, prebuild,
// catch) live behind the testMAC headers and are exercised by the integration
// run; this guards the optional contract and the asset-missing path on every
// runner. (B-20's override-mismatch path also returns nullopt, but reaching it
// needs a resolvable testMAC YAML, which is not guaranteed asset-less.)
// ---------------------------------------------------------------------------

/// RAII override of an environment variable, restored on scope exit so an
/// asset-resolution test cannot leak env state into sibling tests.
class ScopedEnv final
{
public:
    ScopedEnv(const char* name, const char* value) : name_{name}
    {
        if (const char* prev = std::getenv(name); prev != nullptr)
        {
            had_prev_ = true;
            prev_     = prev;
        }
        ::setenv(name, value, /*overwrite=*/1);
    }
    ScopedEnv(const ScopedEnv&)            = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;
    ScopedEnv(ScopedEnv&&)                 = delete;
    ScopedEnv& operator=(ScopedEnv&&)      = delete;
    ~ScopedEnv() noexcept
    {
        if (had_prev_) { ::setenv(name_, prev_.c_str(), /*overwrite=*/1); }
        else           { ::unsetenv(name_); }
    }

private:
    const char* name_;
    bool        had_prev_{false};
    std::string prev_{};
};

TEST(FapiMessageSourceAssets, MissingTestmacYamlYieldsNullopt)
{
    // Force the first resolution step to fail: open_pusch() must return nullopt
    // (not throw, not proceed) before it ever needs CUDA or a launch pattern.
    const ScopedEnv yaml{"PARSER_E2E_TESTMAC_YAML",
                         "/nonexistent/parser_equivalence/no_such_testmac.yaml"};
    EXPECT_FALSE(FapiMessageSource::open_pusch().has_value());
}

// ---------------------------------------------------------------------------
// Fixture: open the message source once for the whole suite.
// ---------------------------------------------------------------------------
class TestMacPuschEquivalence : public ::testing::Test
{
protected:
    static void SetUpTestSuite() { source_ = FapiMessageSource::open_pusch(); }
    static void TearDownTestSuite() { source_.reset(); }

    static std::optional<FapiMessageSource> source_;
};

std::optional<FapiMessageSource> TestMacPuschEquivalence::source_{std::nullopt};

// ---------------------------------------------------------------------------
// Legacy vs new parser produce identical pusch_params for every slot that
// carries PUSCH PDUs in the loaded launch pattern.
// ---------------------------------------------------------------------------
TEST_F(TestMacPuschEquivalence, LegacyVsNewParserEquivalence)
{
    if (!source_.has_value())
    {
        GTEST_SKIP() << "testMAC session could not be opened (default pattern '"
                     << FapiMessageSource::pusch_default_pattern()
                     << "'). Run the binary directly to see the failing step on stderr; set "
                        "cuBB_SDK, or override via PUSCH_E2E_LAUNCH_PATTERN / "
                        "PARSER_E2E_TESTMAC_YAML, to enable.";
    }

    int slots_with_pusch = 0;

    for (const SourceMessage& sm : source_->messages())
    {
        if (sm.desc.msg_id != SCF_FAPI_UL_TTI_REQUEST) { continue; }

        // Production bounds-safe walker: null-checks msg_buf, validates that
        // msg_len covers the FAPI header + nPDUsOfEachType[], and only fires the
        // callback when this message carries >=1 PUSCH PDU.
        nv::for_each_tti_msg<kPuschParserTestTag, scf_fapi_ul_tti_req_t>(
            &sm.desc, 1u,
            [](uint16_t, const scf_fapi_ul_tti_req_t& req) { return nv::has_pusch(req); },
            sm.sfn_slot, /*ring_idx=*/0u,
            [&](uint16_t, const nv::phy_mac_msg_desc& m, const scf_fapi_ul_tti_req_t& req) {
                ++slots_with_pusch;

                SCOPED_TRACE("sfn=" + std::to_string(req.sfn) +
                             " slot=" + std::to_string(req.slot) +
                             " cell=" + std::to_string(m.cell_id));

                // Physical cell id fed identically to both paths: the legacy
                // populator sources phy_cell_index_list from cell_sub_cmd.cell,
                // while the new parser sources it from phy_cell_id(). Feed both
                // paths the SAME phy_cell_id, but make it deliberately DISTINCT
                // from the logical carrier id (m.cell_id) so a bug that wrongly
                // sourced the index list from the carrier id surfaces as a
                // mismatch instead of being masked by phy_cell_id == carrier_id.
                constexpr uint16_t k_phy_cell_id_offset = 0x100u;
                const auto phy_cell_id = static_cast<uint16_t>(
                    static_cast<uint16_t>(m.cell_id) + k_phy_cell_id_offset);

                // --- Legacy path (oracle) ---
                slot_command_api::cell_group_command legacy_grp{};
                slot_command_api::cell_sub_command   legacy_cell{};
                legacy_cell.cell = phy_cell_id;
                run_legacy_pusch(req, m.cell_id, legacy_grp, legacy_cell);
                const auto* legacy_params = legacy_grp.get_pusch_params();
                ASSERT_NE(legacy_params, nullptr);

                // --- New path (mirrors process_aggr_pusch_channel) ---
                MockUlModuleView view{};
                view.carrier_id_val_ = m.cell_id;
                view.phy_cell_id_val_ = phy_cell_id;
                scf_5g_fapi::ULSlotProcessor<MockUlModuleView> processor{view};
                const auto result = processor.process<UL_TTI_PDU_TYPE_PUSCH>(std::span{&m, 1});
                // ASSERT (not EXPECT): a failed parse leaves fresh_params partial,
                // so stop this PDU before the field-by-field compare emits noise.
                ASSERT_TRUE(result.has_value())
                    << "ULSlotProcessor failed at sfn=" << req.sfn << " slot=" << req.slot;
                const auto* fresh_params = view.slot_cmd_.cell_groups.get_pusch_params();
                ASSERT_NE(fresh_params, nullptr);

                expect_pusch_params_match(*legacy_params, *fresh_params);

                // Fronthaul (C-plane) sym_prb map -- separate, toggleable concern
                // (see kCompareFhParams). Remove this guarded call together with
                // the FH block when the parser stops filling FH params.
                if (kCompareFhParams)
                {
                    expect_fh_sym_prb_match(*legacy_cell.sym_prb_info(),
                                            *view.cell_sub_cmd_.sym_prb_info());
                }
            });
    }

    EXPECT_GT(slots_with_pusch, 0)
        << "loaded launch pattern produced no PUSCH PDUs; check channel_mask / pattern selection";
}

} // namespace
} // namespace cuphy_cp::tests
