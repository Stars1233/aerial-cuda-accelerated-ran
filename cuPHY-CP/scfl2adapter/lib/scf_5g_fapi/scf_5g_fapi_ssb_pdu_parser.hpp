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

#if !defined(SCF_5G_FAPI_SSB_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_SSB_PDU_PARSER_HPP_INCLUDED_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"
#include "aerial_event_code.h"
#include "cuphy.h"
#include "nv_phy_config_option.hpp"
#include "nv_phy_limit_errors.hpp"
#include "nv_phy_utils.hpp"
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi
{

void update_cell_command(slot_command_api::cell_group_command* cell_grp_cmd,
                         slot_command_api::cell_sub_command& cell_cmd,
                         const scf_fapi_ssb_pdu_t& cmd,
                         int32_t cell_index,
                         slot_command_api::slot_indication& slotinfo,
                         const nv::phy_config& cell_params,
                         uint8_t l_max,
                         const uint16_t* lmax_symbols,
                         nv::phy_config_option& config_options,
                         pm_weight_map_t& pm_map,
                         nv::slot_detail_t* slot_detail,
                         bool mmimo_enabled);

namespace detail
{

// scf_fapi_ssb_pdu_t ends with flexible pc_and_bf[0] payload. Keep parser/helper
// calls by reference so the trailing bytes remain available to FH preparation.
static_assert(sizeof(scf_fapi_ssb_pdu_t) == offsetof(scf_fapi_ssb_pdu_t, pc_and_bf),
              "scf_fapi_ssb_pdu_t has trailing pc_and_bf payload; do not copy by value");

struct SsbSymbolInfo final
{
    uint8_t l_max{};
    const uint16_t* symbols{};
};

// SSB candidate counts per 3GPP TS 38.213 Table 4.1-1: 4 for FR1 sub-3 GHz
// (Case A/C TDD), 8 for FR1 3-6 GHz (Case B), 64 for FR2 (Cases D/E).
inline constexpr std::size_t k_ssb_lmax4 = 4u;
inline constexpr std::size_t k_ssb_lmax8 = 8u;
inline constexpr std::size_t k_ssb_lmax64 = 64u;
// Number of SSB cases that share each Lmax table: A/B/C in FR1 (3 each for
// Lmax 4 and Lmax 8); D and E in FR2 (2 for Lmax 64).
inline constexpr std::size_t k_ssb_fr1_case_count = 3u;
inline constexpr std::size_t k_ssb_fr2_case_count = 2u;

inline constexpr std::array<std::array<uint16_t, k_ssb_lmax4>, k_ssb_fr1_case_count> k_lmax4_symbols{{
    {2, 8, 16, 22},
    {4, 8, 16, 20},
    {2, 8, 16, 22},
}};

inline constexpr std::array<std::array<uint16_t, k_ssb_lmax8>, k_ssb_fr1_case_count> k_lmax8_symbols{{
    {2, 8, 16, 22, 30, 36, 44, 50},
    {4, 8, 16, 20, 32, 36, 44, 48},
    {2, 8, 16, 22, 30, 36, 44, 50},
}};

inline constexpr std::array<std::array<uint16_t, k_ssb_lmax64>, k_ssb_fr2_case_count> k_lmax64_symbols{{
    {4,   8,   16,  20,  32,  36,  44,  48,
     60,  64,  72,  76,  88,  92,  100, 104,
     144, 148, 156, 160, 172, 176, 184, 188,
     200, 204, 212, 216, 228, 232, 240, 244,
     284, 288, 296, 300, 312, 316, 324, 328,
     340, 344, 352, 356, 368, 372, 380, 384,
     424, 428, 436, 440, 452, 456, 464, 468,
     480, 484, 492, 496, 508, 512, 520, 524},
    {8,   12,  16,  20,  32,  36,  40,  44,
     64,  68,  72,  76,  88,  92,  96,  100,
     120, 124, 128, 132, 144, 148, 152, 156,
     176, 180, 184, 188, 200, 204, 208, 212,
     288, 292, 296, 300, 312, 316, 320, 324,
     344, 348, 352, 356, 368, 372, 376, 380,
     400, 404, 408, 412, 424, 428, 432, 436,
     456, 460, 464, 468, 480, 484, 488, 492},
}};

// 3 dB power offset in linear amplitude: 10^(3/20).
inline constexpr float k_beta_pss_3db = 1.4125375446227545F;

/**
 * Derive the SSB candidate set (Lmax + symbol table) from cell PHY config.
 *
 * Per 3GPP TS 38.213 Table 4.1-1:
 *   - FR1 sub-3 GHz (Cases A/C TDD): Lmax = 4 → k_lmax4_symbols
 *   - FR1 3-6 GHz (Case B):           Lmax = 8 → k_lmax8_symbols
 *   - FR2 (Cases D/E):                Lmax = 64 → k_lmax64_symbols
 *
 * Indexes into the table arrays via the integer value of @c nv::ssb_case.
 *
 * @param[in] cell_params Per-cell PHY config (provides DL/UL absolute freqs and SCS).
 * @return  SsbSymbolInfo with Lmax + symbol pointer on success, or
 *          @c std::nullopt for any unsupported SSB case (e.g. mu=2 / 60 kHz SCS
 *          which is not a valid SSB numerology in 3GPP).
 */
[[nodiscard]] inline std::optional<SsbSymbolInfo>
derive_ssb_symbol_info(const nv::phy_config& cell_params) noexcept
{
    const auto ssb_case =
        nv::getSSBCase(cell_params.carrier_config_.dl_freq_abs_A / 1000,
                       cell_params.carrier_config_.ul_freq_abs_A / 1000,
                       cell_params.ssb_config_.sub_c_common);

    const auto case_index = static_cast<uint8_t>(ssb_case);
    if (cell_params.carrier_config_.dl_freq_abs_A <= 3000000u)
    {
        if (case_index >= k_lmax4_symbols.size()) { return std::nullopt; }
        return SsbSymbolInfo{4u, k_lmax4_symbols[case_index].data()};
    }

    if (cell_params.carrier_config_.dl_freq_abs_A <= 6000000u)
    {
        if (case_index >= k_lmax8_symbols.size()) { return std::nullopt; }
        return SsbSymbolInfo{8u, k_lmax8_symbols[case_index].data()};
    }

    if (ssb_case == nv::ssb_case::CASE_D)
    {
        return SsbSymbolInfo{64u, k_lmax64_symbols[0].data()};
    }
    if (ssb_case == nv::ssb_case::CASE_E)
    {
        return SsbSymbolInfo{64u, k_lmax64_symbols[1].data()};
    }
    return std::nullopt;
}

/**
 * Compute the SSB f0 (subcarrier offset) for a PDU at the given numerology.
 *
 * @c f0 = (@c ssb_subcarrier_offset + @c ssb_offset_point_a × N_TONES_PER_PRB) >> @p mu
 * for valid SSB numerologies.
 *
 * @param[in] pdu  SSB PDU (only ssb_subcarrier_offset / ssb_offset_point_a are read).
 * @param[in] mu   Subcarrier-spacing index: 0 = 15 kHz, 1 = 30 kHz, 3 = 120 kHz, 4 = 240 kHz.
 * @return f0 in subcarriers, or @c std::nullopt when @p mu is not a valid SSB
 *         numerology (mu = 2 / 60 kHz is structurally rejected — no SSB raster
 *         maps to it in 3GPP).
 */
[[nodiscard]] inline std::optional<uint16_t>
calc_ssb_f0(const scf_fapi_ssb_pdu_t& pdu, const uint8_t mu) noexcept
{
    const uint32_t offset =
        static_cast<uint32_t>(pdu.ssb_subcarrier_offset) +
        (static_cast<uint32_t>(pdu.ssb_offset_point_a) * CUPHY_N_TONES_PER_PRB);
    if (mu == 0u)
    {
        return static_cast<uint16_t>(offset);
    }
    // FR1 30 kHz and FR2 120/240 kHz SSB cases scale kSSB/pointA by mu.
    // mu=2 is not a valid SSB numerology, so callers should drop it explicitly.
    if (mu == 1u || mu == 3u || mu == 4u)
    {
        return static_cast<uint16_t>(offset >> mu);
    }
    return std::nullopt;
}

} // namespace detail

/**
 * Parses a single SSB PDU from a DL_TTI.request message.
 *
 * Populates the cuPHY PBCH/SSB dynamic parameters consumed by the channel-task
 * path. The final slot-command write is delegated to the shared legacy SSB
 * helper so the non-direct worker path carries the same PBCH PM/FH
 * compatibility metadata as the legacy on_msg path.
 *
 * @tparam V  Type satisfying DlModuleView.
 */
template<DlModuleView V>
class SsbPduParser final
{
public:
    using pdu_t = scf_fapi_ssb_pdu_t;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_SSB;

    explicit SsbPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Cache the per-message cell context before dispatching SSB PDUs.
     *
     * DLSlotProcessor will call this in the activation MR. Parser unit tests call
     * it directly so MR1 can validate field population without scheduling work.
     */
    void setup_cell(uint32_t msg_cell_id) noexcept;

    /**
     * Process one SSB PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   SSB PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.
    uint32_t logical_cell_id_{};
    uint32_t cell_index_{};
    bool cell_context_ready_{false};
};

template<DlModuleView V>
void SsbPduParser<V>::setup_cell(const uint32_t msg_cell_id) noexcept
{
    logical_cell_id_ = msg_cell_id;
    const int32_t carrier_id = view_->carrier_id(msg_cell_id);
    if (carrier_id < 0)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SsbPduParser: carrier_id<0 for msg_cell_id={}; cell setup skipped, "
                   "subsequent parse() calls will early-return",
                   msg_cell_id);
        cell_context_ready_ = false;
        return;
    }
    cell_index_ = static_cast<uint32_t>(carrier_id);
    cell_context_ready_ = true;
}

template<DlModuleView V>
bool SsbPduParser<V>::parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept
{
    try
    {
        if (!cell_context_ready_) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: setup_cell was not called before parse sfn={} slot={}",
                       sfn, slot);
            return false;
        }

        auto* grp_cmd = view_->group_command();
        if (grp_cmd == nullptr) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: group_command is null sfn={} slot={}",
                       sfn, slot);
            return false;
        }

#ifdef ENABLE_L2_SLT_RSP
        // get_cell_limit_errors is keyed by zero-based local cell index, not by carrier_id.
        // PDSCH (scf_5g_fapi_pdsch_pdu_parser.hpp:972) uses the same split.
        auto& ssb_limit = view_->get_cell_limit_errors(static_cast<uint16_t>(logical_cell_id_))
                              .ssb_pbch_errors;
        if (ssb_limit.parsed < CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT)
        {
            ++ssb_limit.parsed;
        }
        else
        {
            ++ssb_limit.errors;
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: SSB/PBCH L1 limit exceeded sfn={} slot={} cell_index={}",
                       sfn, slot, cell_index_);
            return true;
        }
#endif

        const auto& cell_params = view_->phy_config(logical_cell_id_);
        const auto symbol_info = detail::derive_ssb_symbol_info(cell_params);
        if (!symbol_info.has_value()) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: unable to derive SSB case for sfn={} slot={} cell_index={}",
                       sfn, slot, cell_index_);
            return true;
        }

        if (pdu.ssb_block_index >= symbol_info->l_max) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: ssb_block_index={} >= Lmax={}; dropping PDU",
                       pdu.ssb_block_index, symbol_info->l_max);
            return true;
        }

        auto* grp_params = grp_cmd->get_pbch_params();
        if (grp_params == nullptr) [[unlikely]] { return false; }

        if (grp_params->nSsbBlocks >= grp_params->pbch_dyn_block_params.size()) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: nSsbBlocks={} exceeds capacity {}; dropping PDU",
                       grp_params->nSsbBlocks, grp_params->pbch_dyn_block_params.size());
            return true;
        }

        const auto cell_key = static_cast<int32_t>(cell_index_);
        const auto it = std::find(grp_params->cell_index_list.begin(),
                                  grp_params->cell_index_list.end(),
                                  cell_key);
        const bool new_cell = (it == grp_params->cell_index_list.end());

        if (new_cell && grp_params->ncells >= grp_params->pbch_dyn_cell_params.size()) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "SsbPduParser: ncells={} exceeds capacity {}; dropping PDU",
                       grp_params->ncells, grp_params->pbch_dyn_cell_params.size());
            return true;
        }

        slot_command_api::slot_indication slot_ind{sfn, slot, 0};
        // cell_sub_command is keyed by zero-based local cell index; carrier_id (cell_index_)
        // is only used as the PBCH cell_index_list key.  PDSCH precedent:
        // scf_5g_fapi_pdsch_pdu_parser.hpp:972 uses local idx for cell_sub_command and
        // carrier_id only for cell_index_list.
        auto& cell_cmd = view_->cell_sub_command(logical_cell_id_);
        cell_cmd.cell = view_->phy_cell_id(logical_cell_id_);
        update_cell_command(grp_cmd,
                            cell_cmd,
                            pdu,
                            cell_key,
                            slot_ind,
                            cell_params,
                            symbol_info->l_max,
                            symbol_info->symbols,
                            view_->config_options(),
                            view_->pm_map(),
                            view_->cell_view(logical_cell_id_, slot_ind).slot_detail(),
                            view_->mmimo_enabled());
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SsbPduParser::parse sfn={} slot={} threw: {}",
                   sfn, slot, ex.what());
        return false;
    }
    catch (...)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "SsbPduParser::parse sfn={} slot={} threw unknown exception",
                   sfn, slot);
        return false;
    }
    return true;
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_SSB_PDU_PARSER_HPP_INCLUDED_
