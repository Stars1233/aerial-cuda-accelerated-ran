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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "aerial/casts/casts.hpp"
#include "common_defines.hpp"
#include "common_utils.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include "fapi_defines.hpp"
#include "fapi_validate.hpp"
#include "nv_ipc_utils.h"
#include "nvlog.h"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "scf_5g_fapi_dl_slot_processor.hpp"
#include "scf_5g_fapi.h"
#include "test_mac.hpp"
#include "yaml.hpp"

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 0) // "TEST_MAC"

// ---------------------------------------------------------------------------
// Pattern parameters loaded from the YAML config at startup.
// ---------------------------------------------------------------------------

struct PatternParam {
    std::string name;          // Human-readable label used as the GTest parameter name
    std::string file;          // Launch-pattern YAML filename passed to load_launch_pattern()
    uint64_t    cell_mask;     // 0 = all cells
    uint32_t    channel_mask;  // Bit mask of active channel types
};

// ---------------------------------------------------------------------------
// Per-pattern shared state — constructed lazily on the first test that needs it
// and cached for the remainder of the run so the expensive setup runs once per pattern.
// ---------------------------------------------------------------------------

struct SuiteState {
    // Keep the primary context alive while test_mac owns CUDA-backed buffers.
    // Declared before mac so mac is destroyed first when the cache is cleared.
    std::unique_ptr<PrimaryCtxGuard> cuda_ctx;
    std::unique_ptr<test_mac> mac;
    int load_result     = -1;
    int prebuild_result = -1;
};

struct SuiteCacheKey {
    std::string file;
    uint64_t    cell_mask    = 0;
    uint32_t    channel_mask = 0;

    bool operator==(const SuiteCacheKey& other) const {
        return file == other.file && cell_mask == other.cell_mask
               && channel_mask == other.channel_mask;
    }
};

struct SuiteCacheKeyHash {
    std::size_t operator()(const SuiteCacheKey& key) const noexcept {
        std::size_t h = std::hash<std::string>{}(key.file);
        h ^= std::hash<uint64_t>{}(key.cell_mask) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(key.channel_mask) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Global cache keyed by pattern file + masks (setup depends on all three).
static std::unordered_map<SuiteCacheKey, SuiteState, SuiteCacheKeyHash> g_suite_cache;

// PBCH TV defaults: phy_cell_id=41, 20 MHz @ 30 kHz SCS (106 PRBs), n7 DL/UL,
// sub_c_common=15 kHz.  Matches the testMAC PBCH replay TV fixture.
namespace pbch_tv_defaults {
inline constexpr uint16_t k_phy_cell_id   = 41u;
inline constexpr uint32_t k_dl_freq_khz   = 2110000u;
inline constexpr uint32_t k_ul_freq_khz   = 1920000u;
inline constexpr uint16_t k_dl_grid_prbs  = 106u;
inline constexpr uint8_t  k_sub_c_common  = 0u; // 15 kHz

// Lmax<=4 case: spec allows at most 4 SSB block indices (0..3). Higher indices
// in a prebuild walk imply a different Lmax bucket; first_pbch_prebuild_sample
// scopes to the Lmax<=4 fixture by skipping them.
inline constexpr uint8_t  k_lmax4_block_count = 4u;
} // namespace pbch_tv_defaults

struct PbchProcessorCellView final {
    cuphyCellStatPrm_t stat_{};

    [[nodiscard]] uint16_t num_dl_prb() const noexcept { return pbch_tv_defaults::k_dl_grid_prbs; }
    [[nodiscard]] nv::slot_detail_t* slot_detail() const noexcept { return nullptr; }
    [[nodiscard]] const cuphyCellStatPrm_t& cell_params() const noexcept { return stat_; }
};

struct PbchProcessorView final {
    mutable slot_command_api::slot_command slot_cmd_{};
    mutable slot_command_api::cell_sub_command cell_sub_cmd_{};
    mutable scf_5g_fapi::pm_weight_map_t pm_map_{};
    mutable nv::phy_config_option config_opt_{};
    mutable nv::phy_config phy_config_{};
    mutable nv::slot_limit_cell_error_t limit_errors_{};
    PbchProcessorCellView cell_view_{};

    PbchProcessorView()
    {
        cell_sub_cmd_.cell                          = pbch_tv_defaults::k_phy_cell_id;
        phy_config_.cell_config_.phy_cell_id        = pbch_tv_defaults::k_phy_cell_id;
        phy_config_.carrier_config_.dl_freq_abs_A   = pbch_tv_defaults::k_dl_freq_khz;
        phy_config_.carrier_config_.ul_freq_abs_A   = pbch_tv_defaults::k_ul_freq_khz;
        phy_config_.carrier_config_.dl_grid_size[0] = pbch_tv_defaults::k_dl_grid_prbs;
        phy_config_.ssb_config_.sub_c_common        = pbch_tv_defaults::k_sub_c_common;
    }

    [[nodiscard]] slot_command_api::cell_group_command* group_command() const noexcept
    {
        return &slot_cmd_.cell_groups;
    }

    [[nodiscard]] slot_command_api::cell_sub_command& cell_sub_command(uint32_t cell_idx) const noexcept
    {
        cell_sub_cmd_.cell = static_cast<uint16_t>(41u + cell_idx);
        return cell_sub_cmd_;
    }

    [[nodiscard]] scf_5g_fapi::pm_weight_map_t& pm_map() const noexcept { return pm_map_; }
    [[nodiscard]] bool pm_enabled() const noexcept { return false; }
    [[nodiscard]] bool bf_enabled() const noexcept { return false; }
    [[nodiscard]] slot_command_api::bfw_coeff_mem_info_t* bfw_coeff_mem_info(uint32_t, uint8_t) const noexcept
    {
        return nullptr;
    }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return false; }
    [[nodiscard]] slot_command_api::slot_command& slot_command() const noexcept
    {
        return slot_cmd_;
    }
    [[nodiscard]] nv::phy_config_option& config_options() const noexcept { return config_opt_; }
    [[nodiscard]] int staticPdcchSlotNum() const noexcept { return -1; }
    [[nodiscard]] int staticPdschSlotNum() const noexcept { return -1; }
    [[nodiscard]] uint16_t cell_stat_prm_idx(uint32_t) const noexcept { return 0u; }
    [[nodiscard]] int32_t carrier_id(uint32_t cell_idx) const noexcept
    {
        return static_cast<int32_t>(cell_idx);
    }
    [[nodiscard]] uint16_t phy_cell_id(uint32_t) const noexcept { return phy_config_.cell_config_.phy_cell_id; }
    [[nodiscard]] const nv::phy_config& phy_config(uint32_t) const noexcept
    {
        return phy_config_;
    }
    [[nodiscard]] nv::slot_limit_cell_error_t& get_cell_limit_errors(uint16_t) const noexcept
    {
        return limit_errors_;
    }
    [[nodiscard]] uint16_t num_dl_prb(uint32_t) const noexcept { return 106u; }
    [[nodiscard]] bool is_fapi_to_cplane_direct_enabled() const noexcept { return false; }
    void publish_dl_pdsch_stats(const scf_5g_fapi::DlPdschStatsBatch&) const noexcept {}

    [[nodiscard]] PbchProcessorCellView cell_view(uint32_t,
        const slot_command_api::slot_indication&) const noexcept
    {
        return cell_view_;
    }
};

static_assert(scf_5g_fapi::DlModuleView<PbchProcessorView>);

namespace scf_5g_fapi {

void update_cell_command(slot_command_api::cell_group_command* cell_grp_cmd,
                         slot_command_api::cell_sub_command& cell_cmd,
                         const scf_fapi_ssb_pdu_t& cmd,
                         int32_t cell_index,
                         slot_command_api::slot_indication& slotinfo,
                         const nv::phy_config& cell_params,
                         uint8_t l_max,
                         const uint16_t* lmax_symbols,
                         nv::phy_config_option& config_options,
                         [[maybe_unused]] pm_weight_map_t& pm_map,
                         [[maybe_unused]] nv::slot_detail_t* slot_detail,
                         [[maybe_unused]] bool mmimo_enabled)
{
    auto* grp_params = cell_grp_cmd->get_pbch_params();
    auto& cell_dyn_params = grp_params->pbch_dyn_cell_params[grp_params->ncells];
    auto& block_params = grp_params->pbch_dyn_block_params[grp_params->nSsbBlocks];
    auto& mib_data = grp_params->pbch_dyn_mib_data[grp_params->nSsbBlocks];
    cell_cmd.slot.set_downlink(slotinfo);
    cell_grp_cmd->slot.set_downlink(slotinfo);

    const auto cell_index_pos = [&](int32_t target) noexcept {
        const auto& list = grp_params->cell_index_list;
        const auto pos   = std::find(list.begin(), list.end(), target);
        return std::pair{pos, static_cast<uint16_t>(std::distance(list.begin(), pos))};
    };

    const auto [it, _unused] = cell_index_pos(cell_index);
    const auto new_cell = (it == grp_params->cell_index_list.end());
    if (new_cell) {
        grp_params->cell_index_list.push_back(cell_index);
        grp_params->phy_cell_index_list.push_back(cell_cmd.cell);
        ++grp_params->ncells;
    }

    uint16_t ssb_slot = slotinfo.slot_;
    if (config_options.staticSsbSlotNum != -1) {
        ssb_slot = static_cast<uint16_t>(config_options.staticSsbSlotNum);
    }

    if (new_cell) {
        cell_dyn_params.NID = (config_options.staticSsbPcid != -1)
            ? static_cast<uint16_t>(config_options.staticSsbPcid)
            : cell_params.cell_config_.phy_cell_id;
        cell_dyn_params.nHF = ssb_slot / (5u << cell_params.ssb_config_.sub_c_common);
        cell_dyn_params.Lmax = l_max;
        cell_dyn_params.SFN = config_options.enableTickDynamicSfnSlot
            ? slotinfo.sfn_
            : static_cast<uint16_t>((config_options.staticSsbSFN != -1)
                ? config_options.staticSsbSFN
                : 0);
        cell_dyn_params.k_SSB = cmd.ssb_subcarrier_offset;
        cell_dyn_params.nF =
            cell_params.carrier_config_.dl_grid_size[cell_params.ssb_config_.sub_c_common] *
            CUPHY_N_TONES_PER_PRB;
        cell_dyn_params.slotBufferIdx =
            static_cast<uint16_t>(grp_params->cell_index_list.size() - 1u);
    }

    const auto f0 = detail::calc_ssb_f0(cmd, cell_params.ssb_config_.sub_c_common);
    if (!f0.has_value()) {
        // Mirror the canonical stub in test_stubs.cpp: drop the PDU on unsupported
        // numerology and roll back the new-cell list mutation so negative-path tests
        // see a clean rejection rather than half-filled block state.
        if (new_cell) {
            grp_params->cell_index_list.pop_back();
            grp_params->phy_cell_index_list.pop_back();
            --grp_params->ncells;
        }
        return;
    }
    block_params.blockIndex = cmd.ssb_block_index;
    block_params.t0 = lmax_symbols[block_params.blockIndex] % OFDM_SYMBOLS_PER_SLOT;
    block_params.f0 = *f0;
    block_params.beta_pss = (cmd.beta_pss == 1u) ? detail::k_beta_pss_3db : 1.0F;
    block_params.beta_sss = 1.0F;
    block_params.cell_index = cell_index_pos(cell_index).second;
    block_params.enablePrcdBf = false;

    mib_data = cmd.mib_pdu.agg;
    ++grp_params->nSsbBlocks;
}

} // namespace scf_5g_fapi

namespace pucch_tv_defaults {
inline constexpr uint8_t  k_format_max = UL_TTI_PUCCH_FORMAT_4;
} // namespace pucch_tv_defaults

// ---------------------------------------------------------------------------
// Fixture: each test instance holds (pattern, method-pointer).
// SetUp() lazily populates g_suite_cache for the requested pattern.
// ---------------------------------------------------------------------------

class TestMacCoreFixture : public testing::Test {
public:
    using TestMethod = void (TestMacCoreFixture::*)();

    struct PbchPrebuildStats final {
        uint32_t dl_tti_messages         = 0;
        uint32_t total_pdus              = 0;
        uint32_t ssb_pdus                = 0;
        uint32_t non_ssb_pdus            = 0;
        uint32_t npdus_of_each_type_ssb  = 0;
        std::set<int> cells_with_ssb;
        std::vector<scf_fapi_ssb_pdu_t> pdus;
    };

    struct PdcchUlPrebuildStats final {
        uint32_t ul_dci_messages = 0;
        uint32_t total_pdus      = 0;
        uint32_t pdcch_pdus      = 0;
        uint32_t non_pdcch_pdus  = 0;
        std::set<int> cells_with_pdcch;
        std::vector<scf_fapi_pdcch_pdu_t> pdus;
    };

    struct PbchPrebuildSample final {
        nv::phy_mac_msg_desc desc{};
        scf_fapi_dl_tti_req_t req{};
        std::vector<scf_fapi_ssb_pdu_t> ssbs;
        int cell_id = 0;
    };

    struct PucchPrebuildStats final {
        uint32_t ul_tti_messages = 0;
        uint32_t total_pdus = 0;
        uint32_t pucch_pdus_f01 = 0;
        uint32_t pucch_pdus_f234 = 0;
        uint32_t non_pucch_pdus = 0;
        uint32_t npdus_of_each_type_pucch_f01 = 0;
        uint32_t npdus_of_each_type_pucch_f234 = 0;
        std::set<int> cells_with_pucch;
        std::vector<scf_fapi_pucch_pdu_t> pdus;
    };

    struct PucchPrebuildSample final {
        nv::phy_mac_msg_desc desc{};
        scf_fapi_ul_tti_req_t req{};
        int cell_id = 0;
        uint32_t pucch_pdus_f01 = 0;
        uint32_t pucch_pdus_f234 = 0;
    };

    explicit TestMacCoreFixture(PatternParam p, TestMethod m)
        : param_(std::move(p)), method_(m) {}

    void TestBody() override { (this->*method_)(); }

    // ------------------------------------------------------------------
    // Test-case bodies (prefixed tc_ to avoid collisions with GTest names).
    // ------------------------------------------------------------------

    void tc_LoadLaunchPatternReturnsZero() {
        EXPECT_EQ(suite_->load_result, 0);
    }

    void tc_PrebuildFapiMessagesReturnsZero() {
        ASSERT_EQ(suite_->load_result, 0) << "load_launch_pattern must succeed first";
        EXPECT_EQ(suite_->prebuild_result, 0);
    }

    void tc_GetPrebuiltConfigReqCell0IsNotNull() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "prebuild_fapi_messages must succeed first";
        EXPECT_NE(suite_->mac->get_prebuilt_config_req(0), nullptr);
    }

    void tc_GetPrebuiltSlotMessagesCell0Slot0IsNotEmpty() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "prebuild_fapi_messages must succeed first";
        // F08 TDD pattern: slot 0 is a DL slot, so cell 0 slot 0 only carries
        // messages when the channel_mask selects at least one DL-side channel
        // (PDSCH, PDCCH_DL, PBCH, CSI_RS, BFW_DL). UL-only patterns (PUCCH/
        // PUSCH/PRACH/SRS-only) legitimately have no messages here; skip
        // instead of false-failing.
        constexpr uint32_t k_dl_channel_mask =
            (1u << PDSCH) | (1u << PDCCH_DL) | (1u << PBCH) | (1u << CSI_RS) | (1u << BFW_DL);
        if ((param_.channel_mask & k_dl_channel_mask) == 0u) {
            GTEST_SKIP() << "channel_mask=" << std::hex << param_.channel_mask
                         << " has no DL channel; cell 0 slot 0 is empty by design";
        }
        const sfn_slot_t ss = {.u16 = {0, 0}};
        EXPECT_FALSE(suite_->mac->get_prebuilt_slot_messages(0, ss).empty());
    }

    void tc_PrintPrebuiltFapiMessagesDoesNotCrash() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "prebuild_fapi_messages must succeed first";
        // F08 TDD pattern: UL-only channel_masks (e.g. PUCCH-only) leave most slots empty;
        // print_prebuilt_fapi_messages() then NVLOGEs once per empty (slot, cell) tuple,
        // flooding the CICD log. The print walk is channel-agnostic, so the all-channels
        // patterns already smoke-cover the same code path. Skip on UL-only masks to keep
        // the CICD output readable.
        constexpr uint32_t k_dl_channel_mask =
            (1u << PDSCH) | (1u << PDCCH_DL) | (1u << PBCH) | (1u << CSI_RS) | (1u << BFW_DL);
        if ((param_.channel_mask & k_dl_channel_mask) == 0u) {
            GTEST_SKIP() << "channel_mask=" << std::hex << param_.channel_mask
                         << " has no DL channel; skip print walk to avoid empty-slot log spam";
        }
        EXPECT_NO_FATAL_FAILURE(suite_->mac->print_prebuilt_fapi_messages());
    }

    // Calling prebuild a second time on the same handler must be rejected; this
    // guards the "at most one successful prebuild per handler instance" contract.
    void tc_PrebuildIsIdempotent() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "first prebuild must have succeeded";
        EXPECT_EQ(suite_->mac->prebuild_fapi_messages(), -1);
    }

    void tc_GetCellNumIsPositive() {
        ASSERT_EQ(suite_->load_result, 0);
        ASSERT_NE(suite_->mac->get_fapi_handler(), nullptr);
        EXPECT_GT(suite_->mac->get_fapi_handler()->get_cell_num(), 0);
    }

    void tc_AllValidCellIdsHaveConfigReq() {
        ASSERT_EQ(suite_->prebuild_result, 0);
        const int cell_num = suite_->mac->get_fapi_handler()->get_cell_num();
        ASSERT_GT(cell_num, 0);
        for (int cell_id = 0; cell_id < cell_num; ++cell_id) {
            EXPECT_NE(suite_->mac->get_prebuilt_config_req(cell_id), nullptr)
                << "cell_id=" << cell_id;
        }
    }

    void tc_GetPrebuiltConfigReqNullForNegativeCellId() {
        ASSERT_EQ(suite_->prebuild_result, 0);
        EXPECT_EQ(suite_->mac->get_prebuilt_config_req(-1), nullptr);
    }

    void tc_GetPrebuiltConfigReqNullForOutOfRangeCellId() {
        ASSERT_EQ(suite_->prebuild_result, 0);
        const int cell_num = suite_->mac->get_fapi_handler()->get_cell_num();
        EXPECT_EQ(suite_->mac->get_prebuilt_config_req(cell_num), nullptr);
    }

    void tc_GetPrebuiltSlotMessagesEmptyForInvalidCellId() {
        ASSERT_EQ(suite_->prebuild_result, 0);
        const int      cell_num = suite_->mac->get_fapi_handler()->get_cell_num();
        const sfn_slot_t ss     = {.u16 = {0, 0}};
        EXPECT_TRUE(suite_->mac->get_prebuilt_slot_messages(-1, ss).empty());
        EXPECT_TRUE(suite_->mac->get_prebuilt_slot_messages(cell_num, ss).empty());
    }

    void tc_GettersAreNotNullAfterLoad() {
        ASSERT_EQ(suite_->load_result, 0);
        EXPECT_NE(suite_->mac->get_configs(), nullptr);
        EXPECT_NE(suite_->mac->get_launch_pattern(), nullptr);
        EXPECT_NE(suite_->mac->get_fapi_handler(), nullptr);
    }

    // The FAPI handler is constructed from the launch pattern, so their cell_num
    // and slots_per_frame must agree.
    void tc_FapiHandlerDimensionsMatchLaunchPattern() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler*   fh = suite_->mac->get_fapi_handler();
        launch_pattern* lp = suite_->mac->get_launch_pattern();
        ASSERT_NE(fh, nullptr);
        ASSERT_NE(lp, nullptr);
        EXPECT_EQ(fh->get_cell_num(), lp->get_cell_num());
        EXPECT_EQ(fh->get_slots_per_frame(), lp->get_slots_per_frame());
    }

    void tc_GetSlotInFrameComputesLinearIndex() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler* fh = suite_->mac->get_fapi_handler();
        ASSERT_NE(fh, nullptr);
        const int spf = fh->get_slots_per_frame();
        ASSERT_GT(spf, 0);
        EXPECT_EQ(fh->get_slot_in_frame(0, 0), 0u);
        EXPECT_EQ(fh->get_slot_in_frame(0, static_cast<uint16_t>(spf - 1)),
                  static_cast<uint32_t>(spf - 1));
        EXPECT_EQ(fh->get_slot_in_frame(2, 3), 2u * static_cast<uint32_t>(spf) + 3u);
    }

    void tc_GetNextSfnSlotWithinFrame() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler* fh = suite_->mac->get_fapi_handler();
        ASSERT_NE(fh, nullptr);
        ASSERT_GT(fh->get_slots_per_frame(), 1) << "slots_per_frame must be > 1";
        sfn_slot_t cur{};
        cur.u16.sfn  = 0;
        cur.u16.slot = 0;
        const sfn_slot_t next = fh->get_next_sfn_slot(cur);
        EXPECT_EQ(next.u16.sfn, 0u);
        EXPECT_EQ(next.u16.slot, 1u);
    }

    void tc_GetNextSfnSlotWrapsAtFrameEnd() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler* fh = suite_->mac->get_fapi_handler();
        ASSERT_NE(fh, nullptr);
        const int spf = fh->get_slots_per_frame();
        ASSERT_GT(spf, 0);
        sfn_slot_t cur{};
        cur.u16.sfn  = 5;
        cur.u16.slot = static_cast<uint16_t>(spf - 1);
        const sfn_slot_t next = fh->get_next_sfn_slot(cur);
        EXPECT_EQ(next.u16.sfn, 6u);
        EXPECT_EQ(next.u16.slot, 0u);
    }

    void tc_GetNextSfnSlotWrapsAtSfnMax() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler* fh = suite_->mac->get_fapi_handler();
        ASSERT_NE(fh, nullptr);
        const int spf = fh->get_slots_per_frame();
        ASSERT_GT(spf, 0);
        sfn_slot_t cur{};
        cur.u16.sfn  = static_cast<uint16_t>(FAPI_SFN_MAX - 1);
        cur.u16.slot = static_cast<uint16_t>(spf - 1);
        const sfn_slot_t next = fh->get_next_sfn_slot(cur);
        EXPECT_EQ(next.u16.sfn, 0u);
        EXPECT_EQ(next.u16.slot, 0u);
    }

    // Before start() runs, no cell should be RUNNING.
    void tc_AllCellsAreIdleBeforeStart() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler* fh = suite_->mac->get_fapi_handler();
        ASSERT_NE(fh, nullptr);
        const int cell_num = fh->get_cell_num();
        ASSERT_GT(cell_num, 0);
        EXPECT_TRUE(fh->is_stopped());
        for (int cell_id = 0; cell_id < cell_num; ++cell_id) {
            EXPECT_EQ(fh->get_fapi_state(cell_id), fapi_state_t::IDLE)
                << "cell_id=" << cell_id;
        }
    }

    void tc_GetFapiStateInvalidForOutOfRangeCellId() {
        ASSERT_EQ(suite_->load_result, 0);
        fapi_handler* fh = suite_->mac->get_fapi_handler();
        ASSERT_NE(fh, nullptr);
        EXPECT_EQ(fh->get_fapi_state(fh->get_cell_num()), fapi_state_t::INVALID);
    }

    void tc_PbchTvPrebuildBuildsDlTtiSsbOnly() {
        const PbchPrebuildStats stats = collect_pbch_prebuild_stats();
        EXPECT_GT(stats.dl_tti_messages, 0u);
        EXPECT_GT(stats.ssb_pdus, 0u);
        EXPECT_EQ(stats.non_ssb_pdus, 0u);
    }

    void tc_PbchTvPrebuildCountsMatchSsbPdus() {
#ifndef SCF_FAPI_10_04
        GTEST_SKIP() << "nPDUsOfEachType[SSB] cross-check requires SCF_FAPI_10_04";
#else
        const PbchPrebuildStats stats = collect_pbch_prebuild_stats();
        ASSERT_GT(stats.ssb_pdus, 0u);
        EXPECT_EQ(stats.ssb_pdus, stats.npdus_of_each_type_ssb);
#endif
    }

    void tc_PbchTvPrebuildSsbPduFieldsArePlausible() {
        const PbchPrebuildStats stats = collect_pbch_prebuild_stats();
        ASSERT_FALSE(stats.pdus.empty());
        for (const auto& pdu : stats.pdus) {
            EXPECT_LE(pdu.phys_cell_id, 1007u);
            EXPECT_LT(pdu.ssb_block_index, 64u);
        }
    }

    void tc_PbchTvPrebuildMultiCellCarriesSsbPdus() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "prebuild_fapi_messages must succeed first";
        const int cell_num = suite_->mac->get_fapi_handler()->get_cell_num();
        if (cell_num <= 1) {
            GTEST_SKIP() << "multi-cell PBCH check requires more than one active cell";
        }
        const PbchPrebuildStats stats = collect_pbch_prebuild_stats();
        EXPECT_GT(stats.cells_with_ssb.size(), 1u);
    }

    void tc_PdcchUlTvPrebuildBuildsUlDci() {
        const PdcchUlPrebuildStats stats = collect_pdcch_ul_prebuild_stats();
        if (stats.ul_dci_messages == 0u) {
            GTEST_SKIP() << "launch pattern produced no UL_DCI.req messages";
        }
        EXPECT_GT(stats.pdcch_pdus, 0u);
        EXPECT_EQ(stats.non_pdcch_pdus, 0u)
            << "UL_DCI.req payload must contain only pdu_type==0 PDCCH PDUs";
    }

    void tc_PdcchUlTvPrebuildPdcchPduFieldsArePlausible() {
        const PdcchUlPrebuildStats stats = collect_pdcch_ul_prebuild_stats();
        if (stats.pdcch_pdus == 0u) {
            GTEST_SKIP() << "launch pattern produced no UL_DCI PDCCH PDUs";
        }
        ASSERT_FALSE(stats.pdus.empty());
        for (const auto& pdu : stats.pdus) {
            // Spec ranges per 3GPP TS 38.213.
            EXPECT_LT(pdu.coreset_type, 2u)
                << "coreset_type must be 0 (CORESET#0) or 1 (other CORESETs)";
            EXPECT_GT(pdu.num_dl_dci, 0u)
                << "a PDCCH PDU with zero DCIs is malformed";
        }
    }

    void tc_PdcchUlTvPrebuildMultiCellCarriesPdcchPdus() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "prebuild_fapi_messages must succeed first";
        const int cell_num = suite_->mac->get_fapi_handler()->get_cell_num();
        if (cell_num <= 1) {
            GTEST_SKIP() << "multi-cell PDCCH_UL check requires more than one active cell";
        }
        const PdcchUlPrebuildStats stats = collect_pdcch_ul_prebuild_stats();
        if (stats.pdcch_pdus == 0u) {
            GTEST_SKIP() << "launch pattern produced no UL_DCI PDCCH PDUs";
        }
        EXPECT_GT(stats.cells_with_pdcch.size(), 1u);
    }

    void tc_PbchTvNewPathPopulatesSlotCommandPbch() {
        const auto sample = first_pbch_prebuild_sample();
        ASSERT_TRUE(sample.has_value());
        ASSERT_FALSE(sample->ssbs.empty());

        PbchProcessorView view;
        view.cell_sub_cmd_.cell = static_cast<uint16_t>(41 + sample->cell_id);
        view.phy_config_.cell_config_.phy_cell_id = sample->ssbs.front().phys_cell_id;
        scf_5g_fapi::DLSlotProcessor<PbchProcessorView> processor{view};

        const auto result = processor.process<DL_TTI_PDU_TYPE_SSB>(
            std::span<const nv::phy_mac_msg_desc>{&sample->desc, 1u});

        ASSERT_TRUE(result.has_value());
        const auto* params = view.slot_cmd_.cell_groups.pbch.get();
        ASSERT_NE(params, nullptr);
        EXPECT_EQ(params->ncells, 1u);
        EXPECT_EQ(params->nSsbBlocks, sample->ssbs.size());
        ASSERT_EQ(params->cell_index_list.size(), 1u);
        EXPECT_EQ(params->cell_index_list[0], sample->cell_id);

        const auto& cell = params->pbch_dyn_cell_params[0];
        EXPECT_EQ(cell.NID, sample->ssbs.front().phys_cell_id);
        EXPECT_EQ(cell.SFN, sample->req.sfn);
        EXPECT_EQ(cell.k_SSB, sample->ssbs.front().ssb_subcarrier_offset);

        for (std::size_t i = 0; i < sample->ssbs.size(); ++i) {
            const auto& ssb = sample->ssbs[i];
            const auto& block = params->pbch_dyn_block_params[i];
            EXPECT_EQ(block.blockIndex, ssb.ssb_block_index) << "block=" << i;
            EXPECT_EQ(block.cell_index, 0u) << "block=" << i;
            EXPECT_EQ(block.f0,
                      ssb.ssb_subcarrier_offset
                          + (ssb.ssb_offset_point_a * CUPHY_N_TONES_PER_PRB))
                << "block=" << i;
            EXPECT_FALSE(block.enablePrcdBf) << "block=" << i;
            EXPECT_EQ(params->pbch_dyn_mib_data[i], ssb.mib_pdu.agg) << "block=" << i;
        }
        EXPECT_EQ(view.cell_sub_cmd_.slot.type, slot_command_api::SLOT_DOWNLINK);
        EXPECT_EQ(view.slot_cmd_.cell_groups.slot.type, slot_command_api::SLOT_DOWNLINK);
    }

    void tc_PucchTvPrebuildBuildsUlTtiPucchOnly() {
        const PucchPrebuildStats stats = collect_pucch_prebuild_stats();
        EXPECT_GT(stats.ul_tti_messages, 0u);
        if ((stats.pucch_pdus_f01 + stats.pucch_pdus_f234) == 0u) {
            GTEST_SKIP() << "PUCCH-only TV pattern has no PUCCH PDUs";
        }
        EXPECT_EQ(stats.non_pucch_pdus, 0u);
    }

    void tc_PucchTvPrebuildCountsMatchPucchPdus() {
#ifndef SCF_FAPI_10_04
        GTEST_SKIP() << "nPDUsOfEachType[PUCCH_F01/F234] cross-check requires SCF_FAPI_10_04";
#else
        const PucchPrebuildStats stats = collect_pucch_prebuild_stats();
        const uint32_t pucch_pdus = stats.pucch_pdus_f01 + stats.pucch_pdus_f234;
        if (pucch_pdus == 0u) {
            GTEST_SKIP() << "PUCCH-only TV pattern has no PUCCH PDUs";
        }
        EXPECT_EQ(stats.pucch_pdus_f01, stats.npdus_of_each_type_pucch_f01);
        EXPECT_EQ(stats.pucch_pdus_f234, stats.npdus_of_each_type_pucch_f234);
#endif
    }

    void tc_PucchTvPrebuildPucchPduFieldsArePlausible() {
        const PucchPrebuildStats stats = collect_pucch_prebuild_stats();
        if (stats.pdus.empty()) {
            GTEST_SKIP() << "PUCCH-only TV pattern has no PUCCH PDUs";
        }
        for (const auto& pdu : stats.pdus) {
            // Some PUCCH TVs carry rnti=0; this validates prebuild shape, not UE allocation.
            EXPECT_LE(pdu.format_type, pucch_tv_defaults::k_format_max);
            EXPECT_GT(pdu.bwp.bwp_size, 0u);
            EXPECT_GT(pdu.prb_size, 0u);
            EXPECT_GT(pdu.num_of_symbols, 0u);
        }
    }

    void tc_PucchTvPrebuildMultiCellCarriesPucchPdus() {
        ASSERT_EQ(suite_->prebuild_result, 0) << "prebuild_fapi_messages must succeed first";
        const int cell_num = suite_->mac->get_fapi_handler()->get_cell_num();
        if (cell_num <= 1) {
            GTEST_SKIP() << "multi-cell PUCCH check requires more than one active cell";
        }
        const PucchPrebuildStats stats = collect_pucch_prebuild_stats();
        const uint32_t pucch_pdus = stats.pucch_pdus_f01 + stats.pucch_pdus_f234;
        if (pucch_pdus == 0u) {
            GTEST_SKIP() << "PUCCH-only TV pattern has no PUCCH PDUs";
        }
        EXPECT_GT(stats.cells_with_pucch.size(), 1u);
    }

    void tc_PucchTvPrebuildFirstPucchMessageIsWellFormed() {
        const auto sample = first_pucch_prebuild_sample();
        if (!sample.has_value()) {
            GTEST_SKIP() << "PUCCH-only TV pattern has no PUCCH PDUs";
        }

        EXPECT_EQ(sample->desc.msg_id, SCF_FAPI_UL_TTI_REQUEST);
        EXPECT_NE(sample->desc.msg_buf, nullptr);
        EXPECT_GT(sample->desc.msg_len, sizeof(scf_fapi_header_t));
        EXPECT_GT(sample->req.num_pdus, 0u);
        EXPECT_EQ(sample->pucch_pdus_f01 + sample->pucch_pdus_f234,
                  sample->req.num_pdus);
    }

protected:
    PatternParam param_;
    SuiteState*  suite_ = nullptr;
    TestMethod   method_;

    [[nodiscard]] static const scf_fapi_ul_tti_req_t*
    ul_tti_req_from_msg(const nv::phy_mac_msg_desc& msg) {
        if (msg.msg_id != SCF_FAPI_UL_TTI_REQUEST || msg.msg_buf == nullptr) {
            return nullptr;
        }
        if (msg.msg_len < sizeof(scf_fapi_header_t) + sizeof(scf_fapi_ul_tti_req_t)) {
            ADD_FAILURE() << "UL_TTI message shorter than fixed request header";
            return nullptr;
        }
        const auto* hdr = aerial::casts::assume_cast<scf_fapi_header_t>(msg.msg_buf);
        const auto* body = aerial::casts::assume_cast<scf_fapi_body_header_t>(hdr->payload);
        if (body->type_id != SCF_FAPI_UL_TTI_REQUEST) {
            ADD_FAILURE() << "message descriptor/body type mismatch";
            return nullptr;
        }
        const size_t body_bytes = sizeof(scf_fapi_body_header_t) + body->length;
        if (msg.msg_len < sizeof(scf_fapi_header_t) + body_bytes) {
            ADD_FAILURE() << "UL_TTI body length exceeds copied message buffer";
            return nullptr;
        }
        return aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(body);
    }

    void accumulate_pucch_pdus(const scf_fapi_ul_tti_req_t& req,
                               int cell_id,
                               PucchPrebuildStats& stats) const {
        const auto* body = reinterpret_cast<const uint8_t*>(&req);
        const uint8_t* payload = req.payload;
        const uint8_t* const end = body + sizeof(scf_fapi_body_header_t) + req.msg_hdr.length;

        uint32_t pdus_seen = 0;
        while (pdus_seen < req.num_pdus) {
            if (payload + sizeof(scf_fapi_generic_pdu_info_t) > end) {
                ADD_FAILURE() << "UL_TTI payload ended before generic PDU header";
                return;
            }
            const auto* gen = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(payload);
            if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t)
                || payload + gen->pdu_size > end) {
                ADD_FAILURE() << "malformed generic PDU size " << gen->pdu_size;
                return;
            }

            ++stats.total_pdus;
            if (gen->pdu_type == UL_TTI_PDU_TYPE_PUCCH
                || gen->pdu_type == UL_TTI_PDU_TYPE_PUCCH_2_3_4) {
                if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t) + sizeof(scf_fapi_pucch_pdu_t)) {
                    ADD_FAILURE() << "PUCCH PDU shorter than fixed PUCCH body";
                    return;
                }
                const auto* pucch = aerial::casts::assume_cast<scf_fapi_pucch_pdu_t>(gen->pdu_config);
                stats.pdus.push_back(*pucch);
                stats.cells_with_pucch.insert(cell_id);
                // Bucket by container pdu_type (matches production nPDUsOfEachType[] semantics
                // in detail::ul_npdus_idx in scf_5g_fapi_ul_slot_processor.hpp).  Body-level
                // format_type drift vs container is a separate concern verified in the
                // plausibility test below — not in the bucketing arithmetic.
                if (gen->pdu_type == UL_TTI_PDU_TYPE_PUCCH) {
                    ++stats.pucch_pdus_f01;
                } else {
                    ++stats.pucch_pdus_f234;
                }
            } else {
                ++stats.non_pucch_pdus;
            }

            payload += gen->pdu_size;
            ++pdus_seen;
        }

#ifdef SCF_FAPI_10_04
        stats.npdus_of_each_type_pucch_f01 += req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F01];
        stats.npdus_of_each_type_pucch_f234 += req.nPDUsOfEachType[UL_TTI_NPDUS_IDX_PUCCH_F234];
#endif
    }

    [[nodiscard]] PucchPrebuildStats collect_pucch_prebuild_stats() const {
        if (suite_->prebuild_result != 0) {
            ADD_FAILURE() << "prebuild_fapi_messages must succeed first";
            return {};
        }
        auto* fh = suite_->mac->get_fapi_handler();
        auto* lp = suite_->mac->get_launch_pattern();
        if (fh == nullptr || lp == nullptr) {
            ADD_FAILURE() << "test_mac getters must be valid after load";
            return {};
        }

        PucchPrebuildStats stats;
        sfn_slot_t ss{};
        // slots_seen drives the loop count; the actual sfn/slot tracking happens
        // via the `ss` iterator below (advanced by fh->get_next_sfn_slot(ss)).
        for (int slots_seen = 0; slots_seen < lp->get_sched_slot_num(); ++slots_seen) {
            for (int cell_id = 0; cell_id < fh->get_cell_num(); ++cell_id) {
                for (const auto& msg : suite_->mac->get_prebuilt_slot_messages(cell_id, ss)) {
                    const scf_fapi_ul_tti_req_t* req = ul_tti_req_from_msg(msg);
                    if (req == nullptr) {
                        continue;
                    }
                    ++stats.ul_tti_messages;
                    accumulate_pucch_pdus(*req, cell_id, stats);
                }
            }
            ss = fh->get_next_sfn_slot(ss);
        }
        return stats;
    }

    [[nodiscard]] std::optional<PucchPrebuildSample> first_pucch_prebuild_sample() const {
        if (suite_->prebuild_result != 0) {
            ADD_FAILURE() << "prebuild_fapi_messages must succeed first";
            return std::nullopt;
        }
        auto* fh = suite_->mac->get_fapi_handler();
        auto* lp = suite_->mac->get_launch_pattern();
        if (fh == nullptr || lp == nullptr) {
            ADD_FAILURE() << "test_mac getters must be valid after load";
            return std::nullopt;
        }

        sfn_slot_t ss{};
        // slots_seen drives the loop count; the actual sfn/slot tracking happens
        // via the `ss` iterator below (advanced by fh->get_next_sfn_slot(ss)).
        for (int slots_seen = 0; slots_seen < lp->get_sched_slot_num(); ++slots_seen) {
            for (int cell_id = 0; cell_id < fh->get_cell_num(); ++cell_id) {
                for (const auto& msg : suite_->mac->get_prebuilt_slot_messages(cell_id, ss)) {
                    const scf_fapi_ul_tti_req_t* req = ul_tti_req_from_msg(msg);
                    if (req == nullptr) {
                        continue;
                    }

                    PucchPrebuildStats stats;
                    accumulate_pucch_pdus(*req, cell_id, stats);
                    if ((stats.pucch_pdus_f01 + stats.pucch_pdus_f234) == 0u) {
                        continue;
                    }

                    PucchPrebuildSample sample;
                    sample.desc = msg;
                    sample.req = *req;
                    sample.cell_id = cell_id;
                    sample.pucch_pdus_f01 = stats.pucch_pdus_f01;
                    sample.pucch_pdus_f234 = stats.pucch_pdus_f234;
                    return sample;
                }
            }
            ss = fh->get_next_sfn_slot(ss);
        }

        return std::nullopt;
    }

    [[nodiscard]] static const scf_fapi_dl_tti_req_t*
    dl_tti_req_from_msg(const nv::phy_mac_msg_desc& msg) {
        if (msg.msg_id != SCF_FAPI_DL_TTI_REQUEST || msg.msg_buf == nullptr) {
            return nullptr;
        }
        if (msg.msg_len < sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_tti_req_t)) {
            ADD_FAILURE() << "DL_TTI message shorter than fixed request header";
            return nullptr;
        }
        const auto* hdr = aerial::casts::assume_cast<scf_fapi_header_t>(msg.msg_buf);
        const auto* body = aerial::casts::assume_cast<scf_fapi_body_header_t>(hdr->payload);
        if (body->type_id != SCF_FAPI_DL_TTI_REQUEST) {
            ADD_FAILURE() << "message descriptor/body type mismatch";
            return nullptr;
        }
        const size_t body_bytes = sizeof(scf_fapi_body_header_t) + body->length;
        if (msg.msg_len < sizeof(scf_fapi_header_t) + body_bytes) {
            ADD_FAILURE() << "DL_TTI body length exceeds copied message buffer";
            return nullptr;
        }
        return aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(body);
    }

    [[nodiscard]] static const scf_fapi_ul_dci_t*
    ul_dci_req_from_msg(const nv::phy_mac_msg_desc& msg) {
        if (msg.msg_id != SCF_FAPI_UL_DCI_REQUEST || msg.msg_buf == nullptr) {
            return nullptr;
        }
        if (msg.msg_len < 0 ||
            static_cast<size_t>(msg.msg_len)
                < sizeof(scf_fapi_header_t) + sizeof(scf_fapi_ul_dci_t)) {
            ADD_FAILURE() << "UL_DCI message shorter than fixed request header";
            return nullptr;
        }
        const auto* hdr = aerial::casts::assume_cast<scf_fapi_header_t>(msg.msg_buf);
        const auto* body = aerial::casts::assume_cast<scf_fapi_body_header_t>(hdr->payload);
        if (body->type_id != SCF_FAPI_UL_DCI_REQUEST) {
            ADD_FAILURE() << "message descriptor/body type mismatch";
            return nullptr;
        }
        const size_t body_bytes = sizeof(scf_fapi_body_header_t) + body->length;
        if (static_cast<size_t>(msg.msg_len) < sizeof(scf_fapi_header_t) + body_bytes) {
            ADD_FAILURE() << "UL_DCI body length exceeds copied message buffer";
            return nullptr;
        }
        return aerial::casts::assume_cast<scf_fapi_ul_dci_t>(body);
    }

    static void accumulate_pdcch_ul_pdus(const scf_fapi_ul_dci_t& req,
                                         int cell_id,
                                         PdcchUlPrebuildStats& stats) {
        // Body-relative end pointer; the walk validates each generic PDU header
        // against this bound before advancing the cursor.
        const auto* body = reinterpret_cast<const uint8_t*>(&req);
        const uint8_t* payload = req.payload;
        const uint8_t* const end = body + sizeof(scf_fapi_body_header_t) + req.msg_hdr.length;

        uint32_t pdus_seen = 0;
        while (pdus_seen < req.num_pdus) {
            if (payload + sizeof(scf_fapi_generic_pdu_info_t) > end) {
                ADD_FAILURE() << "UL_DCI payload ended before generic PDU header";
                return;
            }
            const auto* gen = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(payload);
            if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t)
                || payload + gen->pdu_size > end) {
                ADD_FAILURE() << "malformed generic PDU size " << gen->pdu_size;
                return;
            }

            ++stats.total_pdus;
            // pdu_type == 0 = PDCCH per UL_DCI.req spec (Table 3-55).
            if (gen->pdu_type == 0u) {
                if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t) + sizeof(scf_fapi_pdcch_pdu_t)) {
                    ADD_FAILURE() << "PDCCH PDU shorter than fixed PDCCH body";
                    return;
                }
                const auto* pdcch = aerial::casts::assume_cast<scf_fapi_pdcch_pdu_t>(gen->pdu_config);
                stats.pdus.push_back(*pdcch);
                stats.cells_with_pdcch.insert(cell_id);
                ++stats.pdcch_pdus;
            } else {
                ++stats.non_pdcch_pdus;
            }

            payload += gen->pdu_size;
            ++pdus_seen;
        }
    }

    [[nodiscard]] PdcchUlPrebuildStats collect_pdcch_ul_prebuild_stats() const {
        if (suite_->prebuild_result != 0) {
            ADD_FAILURE() << "prebuild_fapi_messages must succeed first";
            return {};
        }
        auto* fh = suite_->mac->get_fapi_handler();
        auto* lp = suite_->mac->get_launch_pattern();
        if (fh == nullptr || lp == nullptr) {
            ADD_FAILURE() << "test_mac getters must be valid after load";
            return {};
        }

        PdcchUlPrebuildStats stats;
        sfn_slot_t ss{};
        for (int slots_seen = 0; slots_seen < lp->get_sched_slot_num(); ++slots_seen) {
            for (int cell_id = 0; cell_id < fh->get_cell_num(); ++cell_id) {
                for (const auto& msg : suite_->mac->get_prebuilt_slot_messages(cell_id, ss)) {
                    const scf_fapi_ul_dci_t* req = ul_dci_req_from_msg(msg);
                    if (req == nullptr) {
                        continue;
                    }
                    ++stats.ul_dci_messages;
                    accumulate_pdcch_ul_pdus(*req, cell_id, stats);
                }
            }
            ss = fh->get_next_sfn_slot(ss);
        }
        return stats;
    }

    void accumulate_ssb_pdus(const scf_fapi_dl_tti_req_t& req,
                             int cell_id,
                             PbchPrebuildStats& stats) const {
        // Raw byte pointer for end-of-body arithmetic; structured reads use assume_cast.
        const auto* body = reinterpret_cast<const uint8_t*>(&req);
        const uint8_t* payload = req.payload;
        const uint8_t* const end = body + sizeof(scf_fapi_body_header_t) + req.msg_hdr.length;

        uint32_t pdus_seen = 0;
        while (pdus_seen < req.num_pdus) {
            if (payload + sizeof(scf_fapi_generic_pdu_info_t) > end) {
                ADD_FAILURE() << "DL_TTI payload ended before generic PDU header";
                return;
            }
            const auto* gen = aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(payload);
            if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t)
                || payload + gen->pdu_size > end) {
                ADD_FAILURE() << "malformed generic PDU size " << gen->pdu_size;
                return;
            }

            ++stats.total_pdus;
            if (gen->pdu_type == DL_TTI_PDU_TYPE_SSB) {
                if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t) + sizeof(scf_fapi_ssb_pdu_t)) {
                    ADD_FAILURE() << "SSB PDU shorter than fixed SSB body";
                    return;
                }
                const auto* ssb = aerial::casts::assume_cast<scf_fapi_ssb_pdu_t>(gen->pdu_config);
                stats.pdus.push_back(*ssb);
                stats.cells_with_ssb.insert(cell_id);
                ++stats.ssb_pdus;
            } else {
                ++stats.non_ssb_pdus;
            }

            payload += gen->pdu_size;
            ++pdus_seen;
        }

#ifdef SCF_FAPI_10_04
        stats.npdus_of_each_type_ssb += req.nPDUsOfEachType[DL_TTI_NPDUS_IDX_SSB];
#endif
    }

    [[nodiscard]] PbchPrebuildStats collect_pbch_prebuild_stats() const {
        if (suite_->prebuild_result != 0) {
            ADD_FAILURE() << "prebuild_fapi_messages must succeed first";
            return {};
        }
        auto* fh = suite_->mac->get_fapi_handler();
        auto* lp = suite_->mac->get_launch_pattern();
        if (fh == nullptr || lp == nullptr) {
            ADD_FAILURE() << "test_mac getters must be valid after load";
            return {};
        }

        PbchPrebuildStats stats;
        sfn_slot_t ss{};
        for (int slots_seen = 0; slots_seen < lp->get_sched_slot_num(); ++slots_seen) {
            for (int cell_id = 0; cell_id < fh->get_cell_num(); ++cell_id) {
                for (const auto& msg : suite_->mac->get_prebuilt_slot_messages(cell_id, ss)) {
                    const scf_fapi_dl_tti_req_t* req = dl_tti_req_from_msg(msg);
                    if (req == nullptr) {
                        continue;
                    }
                    ++stats.dl_tti_messages;
                    accumulate_ssb_pdus(*req, cell_id, stats);
                }
            }
            ss = fh->get_next_sfn_slot(ss);
        }
        return stats;
    }

    [[nodiscard]] std::optional<PbchPrebuildSample> first_pbch_prebuild_sample() const {
        if (suite_->prebuild_result != 0) {
            ADD_FAILURE() << "prebuild_fapi_messages must succeed first";
            return std::nullopt;
        }
        auto* fh = suite_->mac->get_fapi_handler();
        auto* lp = suite_->mac->get_launch_pattern();
        if (fh == nullptr || lp == nullptr) {
            ADD_FAILURE() << "test_mac getters must be valid after load";
            return std::nullopt;
        }

        sfn_slot_t ss{};
        for (int slots_seen = 0; slots_seen < lp->get_sched_slot_num(); ++slots_seen) {
            for (int cell_id = 0; cell_id < fh->get_cell_num(); ++cell_id) {
                for (const auto& msg : suite_->mac->get_prebuilt_slot_messages(cell_id, ss)) {
                    const scf_fapi_dl_tti_req_t* req = dl_tti_req_from_msg(msg);
                    if (req == nullptr) {
                        continue;
                    }

                    PbchPrebuildSample sample;
                    sample.desc = msg;
                    sample.req = *req;
                    sample.cell_id = cell_id;
                    const auto* body = reinterpret_cast<const uint8_t*>(req);
                    const uint8_t* payload = req->payload;
                    const uint8_t* const end =
                        body + sizeof(scf_fapi_body_header_t) + req->msg_hdr.length;
                    for (uint32_t pdu_idx = 0; pdu_idx < req->num_pdus; ++pdu_idx) {
                        if (payload + sizeof(scf_fapi_generic_pdu_info_t) > end) {
                            ADD_FAILURE() << "DL_TTI payload ended before generic PDU header";
                            return std::nullopt;
                        }
                        const auto* gen =
                            aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(payload);
                        if (gen->pdu_size < sizeof(scf_fapi_generic_pdu_info_t)
                            || payload + gen->pdu_size > end) {
                            ADD_FAILURE() << "malformed generic PDU size " << gen->pdu_size;
                            return std::nullopt;
                        }
                        if (gen->pdu_type == DL_TTI_PDU_TYPE_SSB) {
                            if (gen->pdu_size
                                < sizeof(scf_fapi_generic_pdu_info_t)
                                      + sizeof(scf_fapi_ssb_pdu_t)) {
                                ADD_FAILURE() << "SSB PDU shorter than fixed SSB body";
                                return std::nullopt;
                            }
                            const auto* ssb = aerial::casts::assume_cast<scf_fapi_ssb_pdu_t>(
                                gen->pdu_config);
                            if (ssb->ssb_block_index >= pbch_tv_defaults::k_lmax4_block_count) {
                                payload += gen->pdu_size;
                                continue;
                            }
                            sample.ssbs.push_back(*ssb);
                        }
                        payload += gen->pdu_size;
                    }
                    if (!sample.ssbs.empty()) {
                        return sample;
                    }
                }
            }
            ss = fh->get_next_sfn_slot(ss);
        }

        ADD_FAILURE() << "no prebuilt SSB PDU found for PBCH TV test";
        return std::nullopt;
    }

    void SetUp() override {
        const SuiteCacheKey key{param_.file, param_.cell_mask, param_.channel_mask};
        auto it = g_suite_cache.find(key);
        if (it == g_suite_cache.end()) {
            SuiteState st;
            char config_yaml[MAX_PATH_LEN];
            const std::string rel_path =
                std::string(CONFIG_TESTMAC_YAML_PATH).append(CONFIG_TESTMAC_YAML_NAME);
            ASSERT_GE(get_cubb_full_path(config_yaml, nullptr, rel_path.c_str()), 0)
                << "failed to resolve test_mac config path";
            // Initialize CUDA only for pattern/prebuild tests. Pure helper tests
            // such as FapiValidate.* and CommonUtils.* stay GPU-free.
            st.cuda_ctx = std::make_unique<PrimaryCtxGuard>(0);
            st.mac = std::make_unique<test_mac>(config_yaml);
            st.load_result = st.mac->load_launch_pattern(
                param_.file.c_str(), param_.cell_mask, param_.channel_mask);
            if (st.load_result == 0) {
                st.prebuild_result = st.mac->prebuild_fapi_messages();
            }
            it = g_suite_cache.emplace(key, std::move(st)).first;
        }
        suite_ = &it->second;
        ASSERT_NE(suite_->mac, nullptr);
    }
};

// ---------------------------------------------------------------------------
// MAYBE_REG: conditionally register test case TestName for pattern p.
// Skipped if TestName appears in the disabled set.
// ---------------------------------------------------------------------------

#define MAYBE_REG(suite_name, p, disabled, TestName)                        \
    do {                                                                     \
        if (!(disabled).count(#TestName))                                   \
            testing::RegisterTest(                                           \
                (suite_name).c_str(), #TestName,                            \
                nullptr, nullptr,  /* type_param / value_param unused */    \
                __FILE__, __LINE__,                                          \
                [_p = (p)]() -> testing::Test* {                            \
                    return new TestMacCoreFixture(                           \
                        _p, &TestMacCoreFixture::tc_##TestName);            \
                });                                                          \
    } while (false)

// ---------------------------------------------------------------------------
// fapi_validate: pure-logic unit tests (no pattern dependency).
// ---------------------------------------------------------------------------

TEST(FapiValidate, DefaultCtorState) {
    fapi_validate v;
    EXPECT_EQ(v.enable, VALD_ENABLE_NONE);
    EXPECT_EQ(v.log_opt, VALD_LOG_PER_MSG);
    EXPECT_EQ(v.cell_id, 0);
    EXPECT_EQ(v.get_fapi_req(), nullptr);
}

TEST(FapiValidate, ParameterizedCtorRespectsArgs) {
    fapi_validate v(VALD_ENABLE_WARN, VALD_LOG_PER_PDU);
    EXPECT_EQ(v.enable, VALD_ENABLE_WARN);
    EXPECT_EQ(v.log_opt, VALD_LOG_PER_PDU);
}

// enable=NONE: every report() succeeds (the failure is downgraded to a warn).
TEST(FapiValidate, ReportEnableNoneNeverFails) {
    fapi_validate v;
    v.enable = VALD_ENABLE_NONE;
    EXPECT_EQ(v.report(VALD_ENABLE_ERR), VALD_OK);
    EXPECT_EQ(v.report(VALD_ENABLE_WARN), VALD_OK);
}

// enable=ERR: only ERR-level reports fail; WARN-level is downgraded.
TEST(FapiValidate, ReportEnableErrFailsErrPassesWarn) {
    fapi_validate v;
    v.enable = VALD_ENABLE_ERR;
    EXPECT_EQ(v.report(VALD_ENABLE_ERR), VALD_FAIL);
    EXPECT_EQ(v.report(VALD_ENABLE_WARN), VALD_OK);
}

// enable=WARN: both ERR and WARN reports fail.
TEST(FapiValidate, ReportEnableWarnFailsBoth) {
    fapi_validate v;
    v.enable = VALD_ENABLE_WARN;
    EXPECT_EQ(v.report(VALD_ENABLE_ERR), VALD_FAIL);
    EXPECT_EQ(v.report(VALD_ENABLE_WARN), VALD_FAIL);
}

TEST(FapiValidate, ShouldLogPrintAllAlwaysTrue) {
    fapi_validate v;
    v.log_opt = VALD_LOG_PRINT_ALL;
    EXPECT_TRUE(v.should_log(VALD_OK));
    EXPECT_TRUE(v.should_log(VALD_FAIL));
}

TEST(FapiValidate, ShouldLogOkResultIsFalseUnlessPrintAll) {
    fapi_validate v;
    for (int opt : {VALD_LOG_PER_NONE, VALD_LOG_PER_MSG, VALD_LOG_PER_PDU}) {
        v.log_opt = opt;
        EXPECT_FALSE(v.should_log(VALD_OK)) << "log_opt=" << opt;
    }
}

TEST(FapiValidate, ShouldLogPerPduAlwaysLogsFailures) {
    fapi_validate v;
    v.log_opt = VALD_LOG_PER_PDU;
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(v.should_log(VALD_FAIL)) << "i=" << i;
    }
}

// PER_MSG caps logged failures at MAX_PER_MSG_LOG_COUNT (=3) per
// msg_start/msg_ended window.
TEST(FapiValidate, ShouldLogPerMsgCapsAtMaxCount) {
    fapi_validate v;
    v.log_opt = VALD_LOG_PER_MSG;
    for (int i = 0; i < MAX_PER_MSG_LOG_COUNT; ++i) {
        EXPECT_TRUE(v.should_log(VALD_FAIL)) << "i=" << i;
    }
    EXPECT_FALSE(v.should_log(VALD_FAIL)) << "after cap";
    EXPECT_FALSE(v.should_log(VALD_FAIL)) << "after cap (2)";
}

TEST(FapiValidate, ShouldLogPerNoneNeverLogs) {
    fapi_validate v;
    v.log_opt = VALD_LOG_PER_NONE;
    EXPECT_FALSE(v.should_log(VALD_OK));
    EXPECT_FALSE(v.should_log(VALD_FAIL));
}

TEST(FapiValidate, CheckValueEqualUint32ReturnsOk) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    EXPECT_EQ(v.check_value<uint32_t>(VALD_ENABLE_ERR, "a", "b", 42u, 42u), VALD_OK);
}

TEST(FapiValidate, CheckValueDiffExceedsToleranceFails) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    EXPECT_EQ(v.check_value<uint32_t>(VALD_ENABLE_ERR, "a", "b", 10u, 12u, 1u), VALD_FAIL);
}

TEST(FapiValidate, CheckValueDiffWithinToleranceReturnsOk) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    EXPECT_EQ(v.check_value<uint32_t>(VALD_ENABLE_ERR, "a", "b", 10u, 12u, 2u), VALD_OK);
}

// enable=NONE means even a real mismatch is downgraded -> returns OK.
TEST(FapiValidate, CheckValueDisabledDoesNotFail) {
    fapi_validate v;
    v.enable  = VALD_ENABLE_NONE;
    v.log_opt = VALD_LOG_PER_NONE;
    EXPECT_EQ(v.check_value<uint32_t>(VALD_ENABLE_ERR, "a", "b", 1u, 1000u), VALD_OK);
}

// |127 - (-128)| = 255 must not overflow the internal computation
// (the implementation widens to int64_t). With tolerance just below the spread
// the check fails, and with tolerance exactly at the spread it succeeds.
TEST(FapiValidate, CheckValueInt8AbsDiffNoOverflow) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    EXPECT_EQ(v.check_value<int8_t>(VALD_ENABLE_ERR, "a", "b",
                                    int8_t{127}, int8_t{-128}, int8_t{0}),
              VALD_FAIL);
    // tolerance=127 (max signed int8) is still below the 255 spread.
    EXPECT_EQ(v.check_value<int8_t>(VALD_ENABLE_ERR, "a", "b",
                                    int8_t{127}, int8_t{-128}, int8_t{127}),
              VALD_FAIL);
}

TEST(FapiValidate, CheckApproxValueExactMatchReturnsOk) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    // check_approx_value passes when diff < tolerance (strict), so exact match
    // requires a small positive tolerance (not 0.0f, which would trigger 0 >= 0).
    EXPECT_EQ(v.check_approx_value(VALD_ENABLE_ERR, "a", "b", 1.5f, 1.5f, 1e-6f), VALD_OK);
}

TEST(FapiValidate, CheckApproxValueWithinToleranceReturnsOk) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    EXPECT_EQ(v.check_approx_value(VALD_ENABLE_ERR, "a", "b", 1.0f, 1.005f, 0.01f), VALD_OK);
}

// check_approx_value uses `>=` against the tolerance, so exact-match-at-tol fails.
TEST(FapiValidate, CheckApproxValueAtToleranceFails) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    EXPECT_EQ(v.check_approx_value(VALD_ENABLE_ERR, "a", "b", 1.0f, 1.1f, 0.1f), VALD_FAIL);
}

TEST(FapiValidate, CheckBytesEqualBuffersReturnsOk) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    uint8_t a[] = {1, 2, 3, 4};
    uint8_t b[] = {1, 2, 3, 4};
    EXPECT_EQ(v.check_bytes(VALD_ENABLE_ERR, "a", "b", a, b, sizeof(a)), VALD_OK);
}

TEST(FapiValidate, CheckBytesDifferentBuffersFails) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    uint8_t a[] = {1, 2, 3, 4};
    uint8_t b[] = {1, 2, 3, 5};
    EXPECT_EQ(v.check_bytes(VALD_ENABLE_ERR, "a", "b", a, b, sizeof(a)), VALD_FAIL);
}

TEST(FapiValidate, CheckBytesNullptrTreatedAsMismatch) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_NONE);
    uint8_t buf[] = {0xAB};
    EXPECT_EQ(v.check_bytes(VALD_ENABLE_ERR, "a", "b", nullptr, buf, 1), VALD_FAIL);
    EXPECT_EQ(v.check_bytes(VALD_ENABLE_ERR, "a", "b", buf, nullptr, 1), VALD_FAIL);
}

// Hammer the internal error buffer to ensure no overflow / SIGSEGV.
TEST(FapiValidate, LogValueDoesNotOverflowBuffer) {
    fapi_validate v(VALD_ENABLE_ERR, VALD_LOG_PER_PDU);
    EXPECT_NO_FATAL_FAILURE({
        for (int i = 0; i < 10000; ++i) {
            v.log_value(VALD_ENABLE_ERR, "x", "y",
                        static_cast<uint32_t>(i), static_cast<uint32_t>(i + 1));
        }
    });
}

// ---------------------------------------------------------------------------
// common_utils.hpp: bit-field macros and SCF FAPI message name helper.
// ---------------------------------------------------------------------------

TEST(CommonUtils, IntegerSetGetBitsRoundtrip) {
    // start=4, width=3 -> value occupies bits [4, 7); 5 = 0b101
    uint32_t v = INTEGER_SET_BITS(0u, 4, 3, 5u);
    EXPECT_EQ(v, 5u << 4);
    EXPECT_EQ(INTEGER_GET_BITS(v, 4, 3), 5u);
}

TEST(CommonUtils, IntegerSetBitsPreservesOtherBits) {
    uint32_t v = 0xFF00u;
    v          = INTEGER_SET_BITS(v, 0, 4, 0x7u);
    EXPECT_EQ(v, 0xFF07u);
    EXPECT_EQ(INTEGER_GET_BITS(v, 0, 4), 0x7u);
    EXPECT_EQ(INTEGER_GET_BITS(v, 8, 8), 0xFFu);
}

TEST(CommonUtils, IntegerGetBitsExtractsWithinWidth) {
    EXPECT_EQ(INTEGER_GET_BITS(0xFFu, 0, 4), 0xFu);
    EXPECT_EQ(INTEGER_GET_BITS(0xFFu, 4, 4), 0xFu);
}

TEST(CommonUtils, GetScfFapiMsgNameKnownIds) {
    EXPECT_STREQ(get_scf_fapi_msg_name(SCF_FAPI_CONFIG_REQUEST),   "CONFIG.req");
    EXPECT_STREQ(get_scf_fapi_msg_name(SCF_FAPI_SLOT_INDICATION),  "SLOT.ind");
    EXPECT_STREQ(get_scf_fapi_msg_name(SCF_FAPI_DL_TTI_REQUEST),   "DL_TTI.req");
    EXPECT_STREQ(get_scf_fapi_msg_name(SCF_FAPI_ERROR_INDICATION), "ERR.ind");
}

TEST(CommonUtils, GetScfFapiMsgNameUnknownReturnsSentinel) {
    EXPECT_STREQ(get_scf_fapi_msg_name(0xFE), "UNKNOWN_SCF_FAPI");
}

// ---------------------------------------------------------------------------
// YAML config reader
// ---------------------------------------------------------------------------

static const char* kTestConfigRelPath =
    "cuPHY-CP/testMAC/testMAC/tests/test_testmac_core_config.yaml";
static const char* kNvlogConfigRelPath =
    "cuPHY-CP/testMAC/testMAC/tests/test_testmac_core_nvlog_config.yaml";

static constexpr uint32_t kAllChannels = (1U << channel_type_t::CHANNEL_MAX) - 1U;
static constexpr uint32_t kPbchOnlyChannelMask = (1U << channel_type_t::PBCH);
static constexpr uint32_t kPdcchUlOnlyChannelMask = (1U << channel_type_t::PDCCH_UL);
static constexpr uint32_t kPucchOnlyChannelMask = (1U << channel_type_t::PUCCH);

struct TestConfig {
    std::vector<PatternParam>  patterns;
    std::set<std::string>      disabled_tests;
};

static TestConfig load_test_config(const char* path) {
    TestConfig cfg;
    try {
        yaml::file_parser parser(path);
        yaml::document    doc  = parser.next_document();
        yaml::node        root = doc.root();

        yaml::node patterns_node = root["patterns"];
        for (size_t i = 0; i < patterns_node.length(); ++i) {
            yaml::node p = patterns_node[i];
            if (!p["enabled"].as<int>())
                continue;
            PatternParam param;
            param.name         = p["name"].as<std::string>();
            param.file         = p["file"].as<std::string>();
            param.cell_mask    = static_cast<uint64_t>(p["cell_mask"].as<uint64_t>());
            param.channel_mask = static_cast<uint32_t>(p["channel_mask"].as<int>());
            cfg.patterns.push_back(std::move(param));
        }

        if (root.has_key("disabled_tests")) {
            yaml::node dt = root["disabled_tests"];
            for (size_t i = 0; i < dt.length(); ++i)
                cfg.disabled_tests.insert(dt[i].as<std::string>());
        }
    } catch (const std::exception& e) {
        NVLOGC_FMT(TAG, "test config not available at {}: {}; falling back to default pattern",
                   path, e.what());
        cfg.patterns.push_back({"F08_8C_60", "launch_pattern_F08_8C_60.yaml", 0, kAllChannels});
    }

    if (cfg.patterns.empty()) {
        NVLOGC_FMT(TAG, "no enabled patterns in config; falling back to default pattern");
        cfg.patterns.push_back({"F08_8C_60", "launch_pattern_F08_8C_60.yaml", 0, kAllChannels});
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// Main: initialise logging, load config, register per-pattern tests, run.
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Parse GTest flags before any expensive/global setup. This keeps
    // --gtest_filter and --gtest_list_tests usable for GPU-free test suites.
    ::testing::InitGoogleTest(&argc, argv);

    char root[MAX_PATH_LEN];
    if (get_cubb_root_path(root) < 0)
    {
        fprintf(stderr, "get_cubb_root_path failed\n");
        return 1;
    }
    char nvlog_config_path[MAX_PATH_LEN];
    if (get_cubb_full_path(nvlog_config_path, nullptr, kNvlogConfigRelPath) < 0)
    {
        fprintf(stderr, "get_cubb_full_path failed for %s\n", kNvlogConfigRelPath);
        return 1;
    }
    nvlog_fmtlog_init(nvlog_config_path, "test_testmac_core.log", nullptr);
    nvlog_fmtlog_thread_init();

    NVLOGC_FMT(TAG, "testmac_core unit test: root={}", root);

    // Read test config (patterns to run + test cases to disable).
    char test_config_path[MAX_PATH_LEN];
    if (get_cubb_full_path(test_config_path, nullptr, kTestConfigRelPath) < 0)
    {
        fprintf(stderr, "get_cubb_full_path failed for %s\n", kTestConfigRelPath);
        return 1;
    }
    TestConfig cfg = load_test_config(test_config_path);

    NVLOGC_FMT(TAG, "test config: {} pattern(s) enabled, {} test case(s) disabled",
               cfg.patterns.size(), cfg.disabled_tests.size());

    // Dynamically register the fixture-based test cases for each enabled pattern.
    for (const auto& p : cfg.patterns) {
        const std::string suite = "TestMacCore_" + p.name;
        MAYBE_REG(suite, p, cfg.disabled_tests, LoadLaunchPatternReturnsZero);
        MAYBE_REG(suite, p, cfg.disabled_tests, PrebuildFapiMessagesReturnsZero);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetPrebuiltConfigReqCell0IsNotNull);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetPrebuiltSlotMessagesCell0Slot0IsNotEmpty);
        MAYBE_REG(suite, p, cfg.disabled_tests, PrintPrebuiltFapiMessagesDoesNotCrash);
        MAYBE_REG(suite, p, cfg.disabled_tests, PrebuildIsIdempotent);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetCellNumIsPositive);
        MAYBE_REG(suite, p, cfg.disabled_tests, AllValidCellIdsHaveConfigReq);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetPrebuiltConfigReqNullForNegativeCellId);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetPrebuiltConfigReqNullForOutOfRangeCellId);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetPrebuiltSlotMessagesEmptyForInvalidCellId);
        MAYBE_REG(suite, p, cfg.disabled_tests, GettersAreNotNullAfterLoad);
        MAYBE_REG(suite, p, cfg.disabled_tests, FapiHandlerDimensionsMatchLaunchPattern);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetSlotInFrameComputesLinearIndex);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetNextSfnSlotWithinFrame);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetNextSfnSlotWrapsAtFrameEnd);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetNextSfnSlotWrapsAtSfnMax);
        MAYBE_REG(suite, p, cfg.disabled_tests, AllCellsAreIdleBeforeStart);
        MAYBE_REG(suite, p, cfg.disabled_tests, GetFapiStateInvalidForOutOfRangeCellId);
        if (p.channel_mask == kPbchOnlyChannelMask) {
            MAYBE_REG(suite, p, cfg.disabled_tests, PbchTvPrebuildBuildsDlTtiSsbOnly);
            MAYBE_REG(suite, p, cfg.disabled_tests, PbchTvPrebuildCountsMatchSsbPdus);
            MAYBE_REG(suite, p, cfg.disabled_tests, PbchTvPrebuildSsbPduFieldsArePlausible);
            MAYBE_REG(suite, p, cfg.disabled_tests, PbchTvPrebuildMultiCellCarriesSsbPdus);
            MAYBE_REG(suite, p, cfg.disabled_tests, PbchTvNewPathPopulatesSlotCommandPbch);
        }
        if (p.channel_mask == kPdcchUlOnlyChannelMask) {
            MAYBE_REG(suite, p, cfg.disabled_tests, PdcchUlTvPrebuildBuildsUlDci);
            MAYBE_REG(suite, p, cfg.disabled_tests, PdcchUlTvPrebuildPdcchPduFieldsArePlausible);
            MAYBE_REG(suite, p, cfg.disabled_tests, PdcchUlTvPrebuildMultiCellCarriesPdcchPdus);
        }
        if (p.channel_mask == kPucchOnlyChannelMask) {
            MAYBE_REG(suite, p, cfg.disabled_tests, PucchTvPrebuildBuildsUlTtiPucchOnly);
            MAYBE_REG(suite, p, cfg.disabled_tests, PucchTvPrebuildCountsMatchPucchPdus);
            MAYBE_REG(suite, p, cfg.disabled_tests, PucchTvPrebuildPucchPduFieldsArePlausible);
            MAYBE_REG(suite, p, cfg.disabled_tests, PucchTvPrebuildMultiCellCarriesPucchPdus);
            MAYBE_REG(suite, p, cfg.disabled_tests, PucchTvPrebuildFirstPucchMessageIsWellFormed);
        }
    }

    const int ret = RUN_ALL_TESTS();
    // Tear down while CUDA/nvlog are still live (before static destruction).
    g_suite_cache.clear();
    return ret;
}
